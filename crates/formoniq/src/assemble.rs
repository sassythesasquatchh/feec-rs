use crate::operators::{
  ApplyWeight, CoordAwareElMatProvider, DofIdx, ElMatProvider, ElVecProvider,
  InnerProductWeightClosure, Nc1MassElmat, Nc2MassElmat, NcLumpedMassElmat,
};

use common::{
  linalg::nalgebra::{CooMatrix, CooMatrixExt, CsrMatrix, Matrix, Vector},
  util,
};
use ddf::CoordSimplexExt;
use exterior::{
  field::{DifferentialMultiForm, ExteriorField},
  ExteriorGrade,
};
use itertools::Itertools;
use manifold::{
  geometry::{
    coord::{
      mesh::MeshCoords,
      quadrature::SimplexQuadRule,
      simplex::{SimplexCoords, SimplexHandleExt},
      CoordRef,
    },
    metric::{mesh::MeshLengths, simplex::SimplexLengths},
    refsimp_vol,
  },
  topology::{
    complex::Complex,
    handle::{KSimplexIdx, SimplexHandle, SimplexIdx},
    simplex::Simplex,
  },
  Dim,
};

use itertools::izip;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::ops::{AddAssign, Mul};

pub type GalMat = CooMatrix;

#[derive(Debug, Clone, Copy)]
pub struct BarycentricDualSparseInverseConfig {
  pub stabilization_factor: f64,
  pub validation_tolerance: f64,
}

impl Default for BarycentricDualSparseInverseConfig {
  fn default() -> Self {
    Self {
      stabilization_factor: 1.0,
      validation_tolerance: 1e-10,
    }
  }
}

fn assert_supported_nc_grade(dim: Dim, grade: ExteriorGrade) {
  assert!(
    grade < dim,
    "NC support requires grade < intrinsic dimension, got grade {grade} on dimension {dim}."
  );
}

fn nc_slots(grade: ExteriorGrade) -> usize {
  grade + 1
}

fn nc_global_dof(simplex_kidx: usize, slot: usize, grade: ExteriorGrade) -> usize {
  nc_slots(grade) * simplex_kidx + slot
}

fn nc_ndofs(topology: &Complex, grade: ExteriorGrade) -> usize {
  assert_supported_nc_grade(topology.dim(), grade);
  nc_slots(grade) * topology.skeleton(grade).len()
}

fn nc_vertex_dofs(topology: &Complex, grade: ExteriorGrade) -> Vec<Vec<usize>> {
  assert_supported_nc_grade(topology.dim(), grade);

  let mut vertex_dofs = vec![Vec::new(); topology.vertices().len()];
  for simplex in topology.skeleton(grade).handle_iter() {
    for slot in 0..nc_slots(grade) {
      vertex_dofs[simplex[slot]].push(nc_global_dof(simplex.kidx(), slot, grade));
    }
  }
  for dofs in &mut vertex_dofs {
    dofs.sort_unstable();
  }
  vertex_dofs
}

fn nc_lumped_mass_inverse_blocks(
  topology: &Complex,
  grade: ExteriorGrade,
  lumped_mass: &GalMat,
) -> (Vec<Vec<usize>>, Vec<Matrix>) {
  let ndofs = nc_ndofs(topology, grade);
  assert_eq!(lumped_mass.nrows(), ndofs);
  assert_eq!(lumped_mass.ncols(), ndofs);

  let vertex_dofs = nc_vertex_dofs(topology, grade);
  let mut dof_locations = vec![None; ndofs];
  let mut blocks = Vec::with_capacity(vertex_dofs.len());
  for (ivertex, dofs) in vertex_dofs.iter().enumerate() {
    for (ilocal, &dof) in dofs.iter().enumerate() {
      assert!(dof_locations[dof].is_none());
      dof_locations[dof] = Some((ivertex, ilocal));
    }
    blocks.push(Matrix::zeros(dofs.len(), dofs.len()));
  }

  for (r, c, &v) in lumped_mass.triplet_iter() {
    let Some((ivertex_r, ilocal_r)) = dof_locations[r] else {
      panic!("NC{grade} dof {r} is not associated with a mesh vertex.");
    };
    let Some((ivertex_c, ilocal_c)) = dof_locations[c] else {
      panic!("NC{grade} dof {c} is not associated with a mesh vertex.");
    };

    if ivertex_r != ivertex_c {
      assert!(
        v.abs() <= 1e-12,
        "NC{grade} lumped mass must not couple dofs attached to different vertices."
      );
      continue;
    }
    blocks[ivertex_r][(ilocal_r, ilocal_c)] += v;
  }

  let inverse_blocks = blocks
    .into_iter()
    .map(|block| {
      block
        .clone()
        .cholesky()
        .unwrap_or_else(|| panic!("NC{grade} lumped mass vertex block must be positive definite."))
        .inverse()
    })
    .collect();

  (vertex_dofs, inverse_blocks)
}

fn assemble_nc_lumped_mass_inverse_from_blocks(
  ndofs: usize,
  vertex_dofs: &[Vec<usize>],
  inverse_blocks: &[Matrix],
) -> GalMat {
  let mut galmat = GalMat::new(ndofs, ndofs);
  for (dofs, block) in vertex_dofs.iter().zip(inverse_blocks) {
    for (ilocal, &iglobal) in dofs.iter().enumerate() {
      for (jlocal, &jglobal) in dofs.iter().enumerate() {
        let value = block[(ilocal, jlocal)];
        if value != 0.0 {
          galmat.push(iglobal, jglobal, value);
        }
      }
    }
  }
  galmat
}

fn assemble_whitney_projected_sparse_inverse_from_blocks(
  topology: &Complex,
  grade: ExteriorGrade,
  vertex_dofs: &[Vec<usize>],
  inverse_blocks: &[Matrix],
) -> GalMat {
  let nsimps = topology.skeleton(grade).len();
  let slots = nc_slots(grade);
  let scale = (slots as f64).recip().powi(2);
  let mut galmat = GalMat::new(nsimps, nsimps);
  for (dofs, block) in vertex_dofs.iter().zip(inverse_blocks) {
    for (ilocal, &idof) in dofs.iter().enumerate() {
      let isimp = idof / slots;
      for (jlocal, &jdof) in dofs.iter().enumerate() {
        let jsimp = jdof / slots;
        let value = scale * block[(ilocal, jlocal)];
        if value != 0.0 {
          galmat.push(isimp, jsimp, value);
        }
      }
    }
  }
  galmat
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BarycentricDualLocalBlock {
  simplex_kidxs: Vec<KSimplexIdx>,
  volume: f64,
  primal: Matrix,
  dual: Matrix,
  inverse_block: Matrix,
}

fn validate_barycentric_dual_config(
  topology: &Complex,
  coords: &MeshCoords,
  grade: ExteriorGrade,
  config: BarycentricDualSparseInverseConfig,
) -> Result<(), String> {
  if topology.dim() != 3 {
    return Err(format!(
      "barycentric-dual sparse inverse Hodge is implemented only for tetrahedral 3D meshes, got topological dimension {}",
      topology.dim()
    ));
  }
  if coords.dim() != 3 {
    return Err(format!(
      "barycentric-dual sparse inverse Hodge requires 3D coordinates, got coordinate dimension {}",
      coords.dim()
    ));
  }
  if !matches!(grade, 1 | 2) {
    return Err(format!(
      "barycentric-dual sparse inverse Hodge is implemented only for Whitney 1- and 2-forms, got grade {grade}"
    ));
  }
  if !config.stabilization_factor.is_finite() || config.stabilization_factor <= 0.0 {
    return Err("barycentric-dual stabilization factor must be finite and positive".to_string());
  }
  if !config.validation_tolerance.is_finite() || config.validation_tolerance <= 0.0 {
    return Err("barycentric-dual validation tolerance must be finite and positive".to_string());
  }
  Ok(())
}

fn barycentric_dual_sparse_inverse_local_block(
  topology: &Complex,
  coords: &MeshCoords,
  vertex_kidx: KSimplexIdx,
  grade: ExteriorGrade,
  config: BarycentricDualSparseInverseConfig,
) -> Result<BarycentricDualLocalBlock, String> {
  validate_barycentric_dual_config(topology, coords, grade, config)?;

  let simplex_kidxs = topology
    .skeleton(grade)
    .handle_iter()
    .filter(|simplex| simplex.contains(vertex_kidx))
    .map(|simplex| simplex.kidx())
    .collect::<Vec<_>>();
  if simplex_kidxs.len() < 3 {
    return Err(format!(
      "vertex {vertex_kidx} has only {} incident {grade}-simplices; barycentric-dual local reconstruction requires at least 3",
      simplex_kidxs.len()
    ));
  }

  let local_rows = simplex_kidxs
    .iter()
    .copied()
    .enumerate()
    .map(|(local, global)| (global, local))
    .collect::<HashMap<_, _>>();
  let mut primal = Matrix::zeros(simplex_kidxs.len(), 3);
  let mut dual = Matrix::zeros(simplex_kidxs.len(), 3);

  for &simplex_kidx in &simplex_kidxs {
    let simplex = topology.skeleton(grade).handle_by_kidx(simplex_kidx);
    let row = local_rows[&simplex_kidx];
    let vector = match grade {
      1 => scale3(edge_vector(coords, &simplex)?, 0.5),
      2 => scale3(face_vector(coords, &simplex)?, 1.0 / 3.0),
      _ => unreachable!(),
    };
    set_row3(&mut primal, row, vector);
  }

  let mut volume: f64 = 0.0;
  for cell in topology.cells().handle_iter() {
    if !cell.contains(vertex_kidx) {
      continue;
    }
    volume += cell.coord_simplex(coords).vol() / 4.0;

    match grade {
      1 => accumulate_edge_dual_rows(topology, coords, cell, vertex_kidx, &local_rows, &mut dual)?,
      2 => accumulate_face_dual_rows(topology, coords, cell, vertex_kidx, &local_rows, &mut dual)?,
      _ => unreachable!(),
    }
  }

  if !volume.is_finite() || volume <= 0.0 {
    return Err(format!(
      "vertex {vertex_kidx} has non-positive barycentric dual volume {volume}"
    ));
  }

  validate_reconstruction_identity(
    vertex_kidx,
    grade,
    &primal,
    &dual,
    volume,
    config.validation_tolerance,
  )?;
  let inverse_block = barycentric_dual_inverse_block(
    vertex_kidx,
    grade,
    &primal,
    &dual,
    volume,
    config.stabilization_factor,
  )?;

  Ok(BarycentricDualLocalBlock {
    simplex_kidxs,
    volume,
    primal,
    dual,
    inverse_block,
  })
}

fn accumulate_edge_dual_rows(
  topology: &Complex,
  coords: &MeshCoords,
  cell: SimplexHandle,
  vertex_kidx: KSimplexIdx,
  local_rows: &HashMap<KSimplexIdx, usize>,
  dual: &mut Matrix,
) -> Result<(), String> {
  for edge in cell
    .mesh_subsimps(1)
    .filter(|edge| edge.contains(vertex_kidx))
  {
    let opposite_vertices = cell
      .iter()
      .filter(|vertex| !edge.contains(*vertex))
      .collect::<Vec<_>>();
    if opposite_vertices.len() != 2 {
      return Err("tetrahedral edge should have exactly two opposite vertices".to_string());
    }
    let face = Simplex::from(vec![
      vertex_kidx,
      opposite_vertices[0],
      opposite_vertices[1],
    ])
    .sorted();
    let face = topology.skeleton(2).handle_by_simplex(&face);

    let edge_vec = edge_vector(coords, &edge)?;
    let face_vec = face_vector(coords, &face)?;
    let sign = oriented_pair_sign(&face_vec, &edge_vec)?;
    let row = local_rows[&edge.kidx()];
    add_row3(dual, row, scale3(face_vec, sign / 6.0));
  }
  Ok(())
}

fn accumulate_face_dual_rows(
  topology: &Complex,
  coords: &MeshCoords,
  cell: SimplexHandle,
  vertex_kidx: KSimplexIdx,
  local_rows: &HashMap<KSimplexIdx, usize>,
  dual: &mut Matrix,
) -> Result<(), String> {
  for face in cell
    .mesh_subsimps(2)
    .filter(|face| face.contains(vertex_kidx))
  {
    let opposite_vertices = cell
      .iter()
      .filter(|vertex| !face.contains(*vertex))
      .collect::<Vec<_>>();
    if opposite_vertices.len() != 1 {
      return Err("tetrahedral face should have exactly one opposite vertex".to_string());
    }
    let edge = Simplex::from(vec![vertex_kidx, opposite_vertices[0]]).sorted();
    let edge = topology.skeleton(1).handle_by_simplex(&edge);

    let face_vec = face_vector(coords, &face)?;
    let edge_vec = edge_vector(coords, &edge)?;
    let sign = oriented_pair_sign(&face_vec, &edge_vec)?;
    let row = local_rows[&face.kidx()];
    add_row3(dual, row, scale3(edge_vec, sign / 4.0));
  }
  Ok(())
}

fn validate_reconstruction_identity(
  vertex_kidx: KSimplexIdx,
  grade: ExteriorGrade,
  primal: &Matrix,
  dual: &Matrix,
  volume: f64,
  tolerance: f64,
) -> Result<(), String> {
  let actual = primal.transpose() * dual;
  let expected = Matrix::identity(3, 3) * volume;
  let max_abs = actual
    .iter()
    .zip(expected.iter())
    .map(|(a, e)| (a - e).abs())
    .fold(0.0, f64::max);
  let allowed = tolerance * volume.abs().max(1.0);
  if max_abs > allowed {
    return Err(format!(
      "barycentric-dual reconstruction identity failed for vertex {vertex_kidx}, grade {grade}: max abs error {max_abs} exceeds {allowed}"
    ));
  }
  Ok(())
}

fn barycentric_dual_inverse_block(
  vertex_kidx: KSimplexIdx,
  grade: ExteriorGrade,
  primal: &Matrix,
  dual: &Matrix,
  volume: f64,
  stabilization_factor: f64,
) -> Result<Matrix, String> {
  let consistent = (primal * primal.transpose()) / volume;
  let rank_complement = primal.nrows().saturating_sub(3);
  let stabilization = if rank_complement == 0 {
    Matrix::zeros(primal.nrows(), primal.nrows())
  } else {
    let gram = dual.transpose() * dual;
    let Some(gram_inverse) = gram.try_inverse() else {
      return Err(format!(
        "barycentric-dual dual reconstruction matrix is rank deficient for vertex {vertex_kidx}, grade {grade}"
      ));
    };
    let projector =
      Matrix::identity(primal.nrows(), primal.nrows()) - dual * gram_inverse * dual.transpose();
    let trace_scale = consistent.trace() / rank_complement as f64;
    projector * (stabilization_factor * trace_scale)
  };

  let block = consistent + stabilization;
  let block = (&block + block.transpose()) * 0.5;
  if block.clone().cholesky().is_none() {
    return Err(format!(
      "barycentric-dual local inverse block is not positive definite for vertex {vertex_kidx}, grade {grade}"
    ));
  }
  Ok(block)
}

fn edge_vector(coords: &MeshCoords, edge: &SimplexHandle) -> Result<[f64; 3], String> {
  if edge.dim() != 1 {
    return Err(format!("expected edge dimension 1, got {}", edge.dim()));
  }
  let vertices = edge.iter().collect::<Vec<_>>();
  if vertices.len() != 2 {
    return Err(
      "failed to convert edge simplex to two vertices for barycentric-dual assembly".to_string(),
    );
  }
  let (v0, v1) = (vertices[0], vertices[1]);
  Ok(sub3(coord3(coords, v1)?, coord3(coords, v0)?))
}

fn face_vector(coords: &MeshCoords, face: &SimplexHandle) -> Result<[f64; 3], String> {
  if face.dim() != 2 {
    return Err(format!("expected face dimension 2, got {}", face.dim()));
  }
  let vertices = face.iter().collect::<Vec<_>>();
  if vertices.len() != 3 {
    return Err(
      "failed to convert face simplex to three vertices for barycentric-dual assembly".to_string(),
    );
  }
  let (v0, v1, v2) = (vertices[0], vertices[1], vertices[2]);
  let e01 = sub3(coord3(coords, v1)?, coord3(coords, v0)?);
  let e02 = sub3(coord3(coords, v2)?, coord3(coords, v0)?);
  Ok(scale3(cross3(e01, e02), 0.5))
}

fn coord3(coords: &MeshCoords, vertex: usize) -> Result<[f64; 3], String> {
  if coords.dim() != 3 {
    return Err(format!(
      "expected 3D coordinates, got dimension {}",
      coords.dim()
    ));
  }
  let coord = coords.coord(vertex);
  Ok([coord[0], coord[1], coord[2]])
}

fn oriented_pair_sign(face_vec: &[f64; 3], edge_vec: &[f64; 3]) -> Result<f64, String> {
  let dot = dot3(*face_vec, *edge_vec);
  if dot.abs() <= 1e-14 {
    return Err(
      "degenerate tetrahedral geometry produced an orthogonal opposite edge/face pair".to_string(),
    );
  }
  Ok(if dot > 0.0 { 1.0 } else { -1.0 })
}

fn set_row3(matrix: &mut Matrix, row: usize, values: [f64; 3]) {
  for col in 0..3 {
    matrix[(row, col)] = values[col];
  }
}

fn add_row3(matrix: &mut Matrix, row: usize, values: [f64; 3]) {
  for col in 0..3 {
    matrix[(row, col)] += values[col];
  }
}

fn sub3(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
  [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

fn scale3(vector: [f64; 3], scale: f64) -> [f64; 3] {
  [scale * vector[0], scale * vector[1], scale * vector[2]]
}

fn dot3(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
  lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn cross3(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
  [
    lhs[1] * rhs[2] - lhs[2] * rhs[1],
    lhs[2] * rhs[0] - lhs[0] * rhs[2],
    lhs[0] * rhs[1] - lhs[1] * rhs[0],
  ]
}

/// Assembly algorithm for the Galerkin Matrix.
fn assemble_galmat_impl<M>(
  topology: &Complex,
  geometry: &MeshLengths,
  row_grade: ExteriorGrade,
  col_grade: ExteriorGrade,
  eval: impl Fn(&SimplexLengths, &Simplex) -> M + Sync,
) -> GalMat
where
  M: std::ops::Index<(usize, usize), Output = f64> + Send,
{
  let nsimps_row = topology.skeleton(row_grade).len();
  let nsimps_col = topology.skeleton(col_grade).len();

  let triplets: Vec<(usize, usize, f64)> = topology
    .cells()
    .handle_iter()
    .par_bridge()
    .flat_map(|cell| {
      let geo = geometry.simplex_lengths(cell);
      let elmat = eval(&geo, &cell);

      let row_subs: Vec<_> = cell.mesh_subsimps(row_grade).collect();
      let col_subs: Vec<_> = cell.mesh_subsimps(col_grade).collect();

      let mut local_triplets = Vec::new();
      for (ilocal, &iglobal) in row_subs.iter().enumerate() {
        for (jlocal, &jglobal) in col_subs.iter().enumerate() {
          let val = elmat[(ilocal, jlocal)];
          if val != 0.0 {
            local_triplets.push((iglobal.kidx(), jglobal.kidx(), val));
          }
        }
      }

      local_triplets
    })
    .collect();

  let (rows, cols, values) = triplets.into_iter().multiunzip();
  GalMat::try_from_triplets(nsimps_row, nsimps_col, rows, cols, values).unwrap()
}

fn assemble_nc_galmat_impl<M>(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  eval: impl Fn(&SimplexLengths, &Simplex) -> M + Sync,
) -> GalMat
where
  M: std::ops::Index<(usize, usize), Output = f64> + Send,
{
  let ndofs = nc_ndofs(topology, grade);
  let slots = nc_slots(grade);

  let triplets: Vec<(usize, usize, f64)> = topology
    .cells()
    .handle_iter()
    .par_bridge()
    .flat_map(|cell| {
      let geo = geometry.simplex_lengths(cell);
      let elmat = eval(&geo, &cell);

      let local_simps: Vec<_> = cell.mesh_subsimps(grade).collect();
      let nlocal_dofs = slots * local_simps.len();

      let mut local_triplets = Vec::new();
      for ilocal in 0..nlocal_dofs {
        let iglobal = nc_global_dof(local_simps[ilocal / slots].kidx(), ilocal % slots, grade);

        for jlocal in 0..nlocal_dofs {
          let jglobal = nc_global_dof(local_simps[jlocal / slots].kidx(), jlocal % slots, grade);
          let val = elmat[(ilocal, jlocal)];
          if val != 0.0 {
            local_triplets.push((iglobal, jglobal, val));
          }
        }
      }

      local_triplets
    })
    .collect();

  let (rows, cols, values) = triplets.into_iter().multiunzip();
  GalMat::try_from_triplets(ndofs, ndofs, rows, cols, values).unwrap()
}

pub fn assemble_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
  elmat: impl ElMatProvider + Sync,
) -> GalMat {
  let (r, c) = (elmat.row_grade(), elmat.col_grade());
  assemble_galmat_impl(topology, geometry, r, c, move |geo, _cell| elmat.eval(geo))
}

pub fn assemble_galmat_coord_aware(
  topology: &Complex,
  geometry: &MeshLengths,
  elmat: impl CoordAwareElMatProvider + Sync,
) -> GalMat {
  let (r, c) = (elmat.row_grade(), elmat.col_grade());
  assemble_galmat_impl(topology, geometry, r, c, move |geo, cell| {
    elmat.eval_with_coords(geo, cell)
  })
}

pub fn assemble_nc1_mass_galmat(topology: &Complex, geometry: &MeshLengths) -> GalMat {
  let dim = topology.dim();
  assemble_nc_galmat_impl(topology, geometry, 1, move |geo, _cell| {
    Nc1MassElmat::new(dim).eval(geo)
  })
}

pub fn assemble_nc2_mass_galmat(topology: &Complex, geometry: &MeshLengths) -> GalMat {
  let dim = topology.dim();
  assemble_nc_galmat_impl(topology, geometry, 2, move |geo, _cell| {
    Nc2MassElmat::new(dim).eval(geo)
  })
}

pub fn assemble_nc2_mass_galmat_weighted<T>(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<T>,
) -> GalMat
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  let dim = topology.dim();
  let elmat = Nc2MassElmat::new_weighted(dim, coords, qr, weight);
  assemble_nc_galmat_impl(topology, geometry, 2, move |geo, cell| {
    elmat.eval_with_coords(geo, cell)
  })
}

pub fn assemble_nc1_mass_galmat_weighted<T>(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<T>,
) -> GalMat
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  let dim = topology.dim();
  let elmat = Nc1MassElmat::new_weighted(dim, coords, qr, weight);
  assemble_nc_galmat_impl(topology, geometry, 1, move |geo, cell| {
    elmat.eval_with_coords(geo, cell)
  })
}

pub fn assemble_nc_lumped_mass_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
) -> GalMat {
  let dim = topology.dim();
  assemble_nc_galmat_impl(topology, geometry, grade, move |geo, _cell| {
    NcLumpedMassElmat::new(dim, grade).eval(geo)
  })
}

pub fn assemble_nc1_lumped_mass_galmat(topology: &Complex, geometry: &MeshLengths) -> GalMat {
  assemble_nc_lumped_mass_galmat(topology, geometry, 1)
}

pub fn assemble_nc2_lumped_mass_galmat(topology: &Complex, geometry: &MeshLengths) -> GalMat {
  assemble_nc_lumped_mass_galmat(topology, geometry, 2)
}

pub fn assemble_nc_lumped_mass_galmat_weighted(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<f64>,
) -> GalMat {
  let dim = topology.dim();
  let elmat = NcLumpedMassElmat::new_weighted(dim, grade, coords, qr, weight);
  assemble_nc_galmat_impl(topology, geometry, grade, move |geo, cell| {
    elmat.eval_with_coords(geo, cell)
  })
}

pub fn assemble_nc2_lumped_mass_galmat_weighted(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<f64>,
) -> GalMat {
  assemble_nc_lumped_mass_galmat_weighted(topology, geometry, 2, coords, qr, weight)
}

pub fn assemble_nc1_lumped_mass_galmat_weighted(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<f64>,
) -> GalMat {
  assemble_nc_lumped_mass_galmat_weighted(topology, geometry, 1, coords, qr, weight)
}

pub fn assemble_nc_lumped_mass_inverse_galmat_for_grade(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
) -> GalMat {
  let lumped_mass = assemble_nc_lumped_mass_galmat(topology, geometry, grade);
  let (vertex_dofs, inverse_blocks) = nc_lumped_mass_inverse_blocks(topology, grade, &lumped_mass);
  assemble_nc_lumped_mass_inverse_from_blocks(lumped_mass.nrows(), &vertex_dofs, &inverse_blocks)
}

pub fn assemble_nc1_lumped_mass_inverse_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
) -> GalMat {
  assemble_nc_lumped_mass_inverse_galmat_for_grade(topology, geometry, 1)
}

pub fn assemble_nc2_lumped_mass_inverse_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
) -> GalMat {
  assemble_nc_lumped_mass_inverse_galmat_for_grade(topology, geometry, 2)
}

pub fn assemble_nc_lumped_mass_inverse_galmat_weighted_for_grade(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<f64>,
) -> GalMat {
  let lumped_mass =
    assemble_nc_lumped_mass_galmat_weighted(topology, geometry, grade, coords, qr, weight);
  let (vertex_dofs, inverse_blocks) = nc_lumped_mass_inverse_blocks(topology, grade, &lumped_mass);
  assemble_nc_lumped_mass_inverse_from_blocks(lumped_mass.nrows(), &vertex_dofs, &inverse_blocks)
}

pub fn assemble_nc2_lumped_mass_inverse_galmat_weighted(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<f64>,
) -> GalMat {
  assemble_nc_lumped_mass_inverse_galmat_weighted_for_grade(
    topology, geometry, 2, coords, qr, weight,
  )
}

pub fn assemble_nc1_lumped_mass_inverse_galmat_weighted(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<f64>,
) -> GalMat {
  assemble_nc_lumped_mass_inverse_galmat_weighted_for_grade(
    topology, geometry, 1, coords, qr, weight,
  )
}

pub fn assemble_nc_to_whitney_projection_galmat(
  topology: &Complex,
  grade: ExteriorGrade,
) -> GalMat {
  assert_supported_nc_grade(topology.dim(), grade);
  let nsimps = topology.skeleton(grade).len();
  let slots = nc_slots(grade);
  let mut galmat = GalMat::new(nsimps, slots * nsimps);
  for isimp in 0..nsimps {
    for slot in 0..slots {
      galmat.push(
        isimp,
        nc_global_dof(isimp, slot, grade),
        (slots as f64).recip(),
      );
    }
  }
  galmat
}

pub fn assemble_whitney_to_nc_embedding_galmat(topology: &Complex, grade: ExteriorGrade) -> GalMat {
  assert_supported_nc_grade(topology.dim(), grade);
  let nsimps = topology.skeleton(grade).len();
  let slots = nc_slots(grade);
  let mut galmat = GalMat::new(slots * nsimps, nsimps);
  for isimp in 0..nsimps {
    for slot in 0..slots {
      galmat.push(nc_global_dof(isimp, slot, grade), isimp, 1.0);
    }
  }
  galmat
}

pub fn assemble_nc1_to_whitney_projection_galmat(topology: &Complex) -> GalMat {
  assemble_nc_to_whitney_projection_galmat(topology, 1)
}

pub fn assemble_whitney_to_nc1_embedding_galmat(topology: &Complex) -> GalMat {
  assemble_whitney_to_nc_embedding_galmat(topology, 1)
}

pub fn assemble_nc2_to_whitney_projection_galmat(topology: &Complex) -> GalMat {
  assemble_nc_to_whitney_projection_galmat(topology, 2)
}

pub fn assemble_whitney_to_nc2_embedding_galmat(topology: &Complex) -> GalMat {
  assemble_whitney_to_nc_embedding_galmat(topology, 2)
}

pub fn assemble_whitney_projected_sparse_inverse_galmat_for_grade(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
) -> GalMat {
  let lumped_mass = assemble_nc_lumped_mass_galmat(topology, geometry, grade);
  let (vertex_dofs, inverse_blocks) = nc_lumped_mass_inverse_blocks(topology, grade, &lumped_mass);
  assemble_whitney_projected_sparse_inverse_from_blocks(
    topology,
    grade,
    &vertex_dofs,
    &inverse_blocks,
  )
}

pub fn assemble_whitney_projected_sparse_inverse_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
) -> GalMat {
  assemble_whitney_projected_sparse_inverse_galmat_for_grade(topology, geometry, 1)
}

pub fn assemble_whitney_projected_sparse_inverse_galmat_weighted_for_grade(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<f64>,
) -> GalMat {
  let lumped_mass =
    assemble_nc_lumped_mass_galmat_weighted(topology, geometry, grade, coords, qr, weight);
  let (vertex_dofs, inverse_blocks) = nc_lumped_mass_inverse_blocks(topology, grade, &lumped_mass);
  assemble_whitney_projected_sparse_inverse_from_blocks(
    topology,
    grade,
    &vertex_dofs,
    &inverse_blocks,
  )
}

pub fn assemble_whitney_projected_sparse_inverse_galmat_weighted(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<f64>,
) -> GalMat {
  assemble_whitney_projected_sparse_inverse_galmat_weighted_for_grade(
    topology, geometry, 1, coords, qr, weight,
  )
}

pub fn assemble_whitney_2form_projected_sparse_inverse_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
) -> GalMat {
  assemble_whitney_projected_sparse_inverse_galmat_for_grade(topology, geometry, 2)
}

pub fn assemble_whitney_2form_projected_sparse_inverse_galmat_weighted(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure<f64>,
) -> GalMat {
  assemble_whitney_projected_sparse_inverse_galmat_weighted_for_grade(
    topology, geometry, 2, coords, qr, weight,
  )
}

pub fn assemble_barycentric_dual_sparse_inverse_galmat_for_grade(
  topology: &Complex,
  coords: &MeshCoords,
  grade: ExteriorGrade,
  config: BarycentricDualSparseInverseConfig,
) -> Result<GalMat, String> {
  validate_barycentric_dual_config(topology, coords, grade, config)?;
  let ndofs = topology.skeleton(grade).len();
  let mut galmat = GalMat::new(ndofs, ndofs);

  for vertex in topology.vertices().handle_iter() {
    let local =
      barycentric_dual_sparse_inverse_local_block(topology, coords, vertex.kidx(), grade, config)?;
    for (ilocal, &irow) in local.simplex_kidxs.iter().enumerate() {
      for (jlocal, &jcol) in local.simplex_kidxs.iter().enumerate() {
        let value = local.inverse_block[(ilocal, jlocal)];
        if value != 0.0 {
          galmat.push(irow, jcol, value);
        }
      }
    }
  }

  Ok(galmat)
}

pub fn assemble_barycentric_dual_1form_sparse_inverse_galmat(
  topology: &Complex,
  coords: &MeshCoords,
  config: BarycentricDualSparseInverseConfig,
) -> Result<GalMat, String> {
  assemble_barycentric_dual_sparse_inverse_galmat_for_grade(topology, coords, 1, config)
}

pub fn assemble_barycentric_dual_2form_sparse_inverse_galmat(
  topology: &Complex,
  coords: &MeshCoords,
  config: BarycentricDualSparseInverseConfig,
) -> Result<GalMat, String> {
  assemble_barycentric_dual_sparse_inverse_galmat_for_grade(topology, coords, 2, config)
}

pub type GalVec = Vector;
/// Assembly algorithm for the Galerkin Vector.
pub fn assemble_galvec(
  topology: &Complex,
  geometry: &MeshLengths,
  elvec: impl ElVecProvider,
) -> GalVec {
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let entries: Vec<(usize, f64)> = topology
    .cells()
    .handle_iter()
    .par_bridge()
    .flat_map(|cell| {
      let geo = geometry.simplex_lengths(cell);
      let elvec = elvec.eval(&geo, &cell);

      let subs: Vec<_> = cell.mesh_subsimps(grade).collect();

      let mut local_entries = Vec::new();
      for (ilocal, &iglobal) in subs.iter().enumerate() {
        if elvec[ilocal] != 0.0 {
          local_entries.push((iglobal.kidx(), elvec[ilocal]));
        }
      }

      local_entries
    })
    .collect();

  let mut galvec = Vector::zeros(nsimps);
  for (irow, val) in entries {
    galvec[irow] += val;
  }
  galvec
}

/// Assembly algorithm for the Galerkin Vector.
pub fn assemble_boundary_galvec<P>(
  topology: &Complex,
  geometry: &MeshLengths,
  elvec: impl ElVecProvider,
  boundary_selector: P,
) -> GalVec
where
  P: Fn(KSimplexIdx) -> bool + Sync,
{
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let entries: Vec<(usize, f64)> = topology
    .boundary_facets()
    .into_par_iter()
    .filter(|fidx| boundary_selector(fidx.kidx))
    .flat_map(|fidx| {
      let facet = fidx.handle(topology);
      let geo = geometry.simplex_lengths(facet);
      let elvec = elvec.eval(&geo, &facet);

      let subs: Vec<_> = facet.mesh_subsimps(grade).collect();

      let mut local_entries = Vec::new();
      for (ilocal, &iglobal) in subs.iter().enumerate() {
        if elvec[ilocal] != 0.0 {
          local_entries.push((iglobal.kidx(), elvec[ilocal]));
        }
      }

      local_entries
    })
    .collect();

  let mut galvec = Vector::zeros(nsimps);
  for (irow, val) in entries {
    galvec[irow] += val;
  }
  galvec
}

/// Return simplices of a given grade whose barycenter satisfies a predicate.
///
/// Useful for partitioning boundaries by geometric location.
pub fn boundary_simplices_where_barycenter<P>(
  topology: &Complex,
  coords: &MeshCoords,
  dim: Dim,
  predicate: P,
) -> Vec<KSimplexIdx>
where
  P: Fn(CoordRef) -> bool + Sync,
{
  assert!(
    dim < topology.dim(),
    "Simplex dimension exceeds boundary dimension."
  );
  topology
    .boundary_subcomplex_simplices(dim)
    .into_par_iter()
    .filter_map(|simp_idx| {
      let simp = simp_idx.handle(topology);
      let simplex_coords = SimplexCoords::from_simplex_and_coords(&simp, coords);
      let barycenter = simplex_coords.barycenter();
      predicate(barycenter.as_view()).then_some(simp_idx.kidx)
    })
    .collect()
}
// Assemble a boundary (Neumann) Galerkin vector
// $\int_{\Gamma_N} \mathrm{tr}\, \omega \wedge g_N$ on a selectable subset of boundary facets.
pub fn assemble_boundary_integral_term<F: DifferentialMultiForm>(
  topology: &Complex,
  coords: &MeshCoords,
  test_grade: ExteriorGrade,
  boundary_data: &F,
  qr: Option<SimplexQuadRule>,
  boundary_selector: &dyn Fn(KSimplexIdx) -> bool,
) -> GalVec {
  let nsimps = topology.skeleton(test_grade).len();
  if nsimps == 0 {
    return Vector::zeros(0);
  }

  let boundary_dim = topology.dim().saturating_sub(1);
  assert!(
    test_grade <= boundary_dim,
    "Test form grade exceeds boundary dimension."
  );
  assert!(
    boundary_data.grade() + test_grade == boundary_dim,
    "Boundary data grade does not match (n-1 - grade(test)). Data grade: {}, test grade: {}, boundary dimension: {}",
    boundary_data.grade(),
    test_grade,
    boundary_dim
  );

  let qr = qr.unwrap_or_else(|| SimplexQuadRule::barycentric(boundary_dim));

  // TODO make safe for parallel execution
  let entries: Vec<(usize, f64)> = topology
    .boundary_facets()
    .into_iter()
    .filter(|fidx| boundary_selector(fidx.kidx))
    .flat_map(|fidx| {
      let facet = fidx.handle(topology);
      let facet_coords = SimplexCoords::from_simplex_and_coords(&facet, coords);

      let orientation_sign = if boundary_dim == 0 {
        1.0
      } else {
        boundary_orientation_sign(facet, coords)
      };

      let elvec = boundary_elvec_for_facet(
        test_grade,
        facet,
        &facet_coords,
        boundary_data,
        &qr,
        orientation_sign,
      );

      facet
        .mesh_subsimps(test_grade)
        .enumerate()
        .map(|(iloc, sub)| (sub.kidx(), elvec[iloc]))
        .collect::<Vec<_>>()
    })
    .collect();

  let mut galvec = Vector::zeros(nsimps);
  for (irow, val) in entries {
    galvec[irow] += val;
  }
  galvec
}

fn boundary_elvec_for_facet<F: DifferentialMultiForm>(
  test_grade: ExteriorGrade,
  facet: SimplexHandle,
  facet_coords: &SimplexCoords,
  boundary_data: &F,
  qr: &SimplexQuadRule,
  orientation_sign: f64,
) -> Vector {
  let subs: Vec<_> = facet.mesh_subsimps(test_grade).collect();
  if subs.is_empty() {
    return Vector::zeros(0);
  }

  let mut elvec = Vector::zeros(subs.len());
  let multivector = facet_coords.spanning_multivector();
  let vol = refsimp_vol(facet_coords.dim_intrinsic());

  for (iloc, sub) in subs.iter().enumerate() {
    let local_sub = sub.relative_to(&facet);
    let lsf = ddf::whitney::lsf::WhitneyLsf::from_coords(facet_coords.clone(), local_sub);
    let f = |xi: CoordRef| {
      let global = facet_coords.local2global(xi);
      let phi = lsf.at_point(global.as_view());
      let g = if boundary_data.dim_ambient() == facet_coords.dim_ambient() {
        boundary_data.at_point(global.as_view())
      } else if boundary_data.dim_ambient() == facet_coords.dim_intrinsic()
        && facet_coords.dim_ambient() != facet_coords.dim_intrinsic()
      {
        facet_coords.lift_form(&boundary_data.at_point(global.as_view()))
      } else {
        panic!(
          "Boundary data ambient dimension {} is incompatible with facet dimensions ({}, {}).",
          boundary_data.dim_ambient(),
          facet_coords.dim_intrinsic(),
          facet_coords.dim_ambient()
        );
      };
      let integrand = phi.wedge(&g);
      orientation_sign * integrand.apply_form_to_vector(&multivector)
    };
    elvec[iloc] = qr.integrate_local(&f, vol);
  }

  elvec
}

/// Orientation factor for an oriented boundary facet induced by its unique parent cell.
///
/// The factor combines
/// - the sign of the facet in the boundary chain of its parent cell, and
/// - the orientation of that cell with respect to the ambient coordinates.
pub fn boundary_orientation_sign(facet: SimplexHandle, coords: &MeshCoords) -> f64 {
  let parent_cell = facet
    .cocells()
    .next()
    .expect("Boundary facet should have exactly one parent cell.");

  let facet_sign = parent_cell
    .boundary_chain()
    .find_map(|(sign, subfacet)| (subfacet == facet).then_some(sign))
    .expect("Boundary facet must appear in boundary of its parent cell.")
    .as_f64();

  let parent_coords = SimplexCoords::from_simplex_and_coords(&parent_cell, coords);
  if parent_coords.is_same_dim() {
    facet_sign * parent_coords.orientation().as_f64()
  } else {
    facet_sign
  }
}

pub fn drop_boundary_dofs_galmat(complex: &Complex, galmat: &mut GalMat) {
  drop_dofs_galmat(&complex.boundary_vertices().into_iter().collect(), galmat)
}

// Build old-index -> new-index map (None if dropped).
fn build_index_map(n_old: usize, drop: &HashSet<usize>) -> Vec<Option<usize>> {
  let mut map = vec![None; n_old];
  let mut next = 0usize;
  for (i, slot) in map.iter_mut().enumerate() {
    if !drop.contains(&i) {
      *slot = Some(next);
      next += 1;
    }
  }
  map
}

pub fn drop_dofs_rectangular_galmat(
  drop_rows: &HashSet<usize>,
  drop_cols: &HashSet<usize>,
  galmat: &mut GalMat,
) {
  let nrows_old = galmat.nrows();
  let ncols_old = galmat.ncols();

  assert!(drop_rows.len() <= nrows_old);
  assert!(drop_cols.len() <= ncols_old);
  assert!(drop_rows.iter().all(|&r| r < nrows_old));
  assert!(drop_cols.iter().all(|&c| c < ncols_old));

  let nrows_new = nrows_old - drop_rows.len();
  let ncols_new = ncols_old - drop_cols.len();

  let row_map = build_index_map(nrows_old, drop_rows);
  let col_map = build_index_map(ncols_old, drop_cols);

  let (rows, cols, values) = std::mem::replace(galmat, GalMat::new(0, 0)).disassemble();
  let nnz_old = values.len();

  let mut new_rows = Vec::with_capacity(nnz_old);
  let mut new_cols = Vec::with_capacity(nnz_old);
  let mut new_vals = Vec::with_capacity(nnz_old);

  for (r, c, v) in izip!(rows, cols, values) {
    if let (Some(r2), Some(c2)) = (row_map[r], col_map[c]) {
      new_rows.push(r2);
      new_cols.push(c2);
      new_vals.push(v);
    }
  }

  *galmat = GalMat::try_from_triplets(nrows_new, ncols_new, new_rows, new_cols, new_vals).unwrap();
}

// pub fn drop_dofs_galmat(dofs: &HashSet<DofIdx>, galmat: &mut GalMat) {
//   assert!(galmat.nrows() == galmat.ncols());
//   let ndofs_old = galmat.ncols();
//   let ndofs_new = ndofs_old - dofs.len();

//   let (rows, cols, values) = std::mem::replace(galmat, GalMat::new(0, 0)).disassemble();

//   let (rows, cols, values): (Vec<_>, Vec<_>, Vec<_>) = multizip((rows, cols, values))
//     .filter(|(r, c, _)| !dofs.contains(r) && !dofs.contains(c))
//     .map(|(mut r, mut c, v)| {
//       let diffr = dofs.iter().filter(|&&idof| idof < r).count();
//       let diffc = dofs.iter().filter(|&&idof| idof < c).count();
//       r -= diffr;
//       c -= diffc;
//       (r, c, v)
//     })
//     .multiunzip();

//   *galmat = GalMat::try_from_triplets(ndofs_new, ndofs_new, rows, cols, values).unwrap();
// }

pub fn drop_dofs_galmat(dofs: &HashSet<usize>, galmat: &mut GalMat) {
  drop_dofs_rectangular_galmat(dofs, dofs, galmat);
}

pub fn drop_dofs_galvec(dofs: &[DofIdx], galvec: &mut GalVec) {
  *galvec = std::mem::take(galvec).remove_rows_at(dofs);
}

pub fn reintroduce_boundary_dofs_galsols(complex: &Complex, galsols: &mut Matrix) {
  reintroduce_dropped_dofs_galsols(complex.boundary_vertices(), galsols)
}

pub fn reintroduce_dropped_dofs_galsols(mut dofs: Vec<DofIdx>, galsols: &mut Matrix) {
  dofs.sort_unstable();
  dofs.dedup();

  let mut galsol_owned = std::mem::take(galsols);
  for dof in dofs {
    galsol_owned = galsol_owned.insert_row(dof, 0.0);
  }
  *galsols = galsol_owned;
}

pub fn reintroduce_non_homogenous_dofs_galsols(dof_coeffs: &[(DofIdx, f64)], galsols: &mut Vector) {
  let mut pairs: Vec<(DofIdx, f64)> = dof_coeffs
    .iter()
    .map(|(dof, coeff)| (*dof, *coeff))
    .collect();

  // Sort by dof index
  pairs.sort_unstable_by_key(|(a, _)| *a);

  let initial_len = pairs.len();
  pairs.dedup_by(|(a, _), (b, _)| a == b);
  let deduped_len = pairs.len();
  assert!(
    deduped_len == initial_len,
    "Duplicate dof indices found in reintroduction of non-homogeneous dofs."
  );

  // Insert rows with the provided coefficient
  let mut owned = std::mem::take(galsols);
  for (dof, coeff) in pairs {
    owned = owned.insert_row(dof, coeff);
  }
  *galsols = owned;
}

pub fn enforce_homogeneous_dirichlet_bc(
  complex: &Complex,
  galmat: &mut GalMat,
  galvec: &mut Vector,
) {
  fix_dofs_zero(&complex.boundary_vertices(), galmat, galvec);
}

pub fn enforce_dirichlet_bc_partial<F>(
  complex: &Complex,
  boundary_coeff_map: F,
  galmat: &mut GalMat,
  galvec: &mut Vector,
  boundary_selector: Option<&dyn Fn(usize) -> bool>,
) where
  F: Fn(DofIdx) -> f64,
{
  let boundary_selector = boundary_selector.unwrap_or(&|_: usize| true);
  let boundary_dofs = complex.boundary_vertices();
  let dof_coeffs: Vec<_> = boundary_dofs
    .into_iter()
    .filter(|sidx| boundary_selector(*sidx))
    .map(|idof| (idof, boundary_coeff_map(idof)))
    .collect();

  fix_dofs_coeff(&dof_coeffs, galmat, galvec);
}

pub fn enforce_dirichlet_bc<F>(
  complex: &Complex,
  boundary_coeff_map: F,
  galmat: &mut GalMat,
  galvec: &mut Vector,
) where
  F: Fn(DofIdx) -> f64,
{
  enforce_dirichlet_bc_partial(complex, boundary_coeff_map, galmat, galvec, None);
}

pub fn enforce_essential_bc<F>(
  grade: ExteriorGrade,
  complex: &Complex,
  boundary_coeff_map: F,
  galmat: &mut GalMat,
  galvec: &mut Vector,
  boundary_selector: Option<&dyn Fn(SimplexIdx) -> bool>,
) where
  F: Fn(DofIdx) -> f64,
{
  let boundary_selector = boundary_selector.unwrap_or(&|_sidx: SimplexIdx| true);
  let boundary_dofs = complex.boundary_subcomplex_simplices(grade);
  let dof_coeffs: Vec<_> = boundary_dofs
    .into_iter()
    .filter(|sidx| boundary_selector(*sidx))
    .map(|simp| {
      let idof = simp.kidx;
      (idof, boundary_coeff_map(idof))
    })
    .collect();
  fix_dofs_coeff(&dof_coeffs, galmat, galvec);
}

pub fn fix_dofs_zero(dofs: &[DofIdx], galmat: &mut GalMat, galvec: &mut Vector) {
  let ndofs = galmat.nrows();
  let dof_flags = util::indicies_to_flags(dofs, ndofs);
  galmat.set_zero(|i, j| dof_flags[i] || dof_flags[j]);
  for &idof in dofs {
    galmat.push(idof, idof, 1.0);
    galvec[idof] = 0.0;
  }
}

/// Fix DOFs of FE solution.
///
/// Modifies supplied galerkin matrix and galerkin vector,
/// such that the FE solution has the optionally given coefficents on the dofs.
/// $mat(A_0, 0; 0, I) vec(mu_0, mu_diff) = vec(phi - A_(0 diff) gamma, gamma)$
pub fn fix_dofs_coeff(dof_coeffs: &[(DofIdx, f64)], galmat: &mut GalMat, galvec: &mut Vector) {
  let ndofs = galmat.nrows();

  let dof_coeffs_opt = util::sparse_to_dense_data(dof_coeffs.to_vec(), ndofs);
  let dof_coeffs_zeroed =
    Vector::from_iterator(ndofs, dof_coeffs_opt.iter().map(|v| v.unwrap_or(0.0)));

  // Modify galvec.
  let galmat_csr = CsrMatrix::from(&*galmat);
  *galvec -= galmat_csr * dof_coeffs_zeroed;

  // Set galvec to prescribed coefficents.
  dof_coeffs.iter().for_each(|&(i, v)| galvec[i] = v);

  // Set entires zero that share a (row or column) index with a fixed dof.
  galmat.set_zero(|r, c| dof_coeffs_opt[r].is_some() || dof_coeffs_opt[c].is_some());

  // Set galmat diagonal for dofs to one.
  for &(i, _) in dof_coeffs {
    galmat.push(i, i, 1.0);
  }
}

/// $mat(A_0, A_(0 diff); 0, I) vec(mu_0, mu_diff) = vec(phi, gamma)$
//#[allow(unused_variables, unreachable_code)]
pub fn fix_dofs_coeff_alt(dof_coeffs: &[(DofIdx, f64)], galmat: &mut GalMat, galvec: &mut Vector) {
  tracing::warn!("use of `fix_dofs_coeff_alt` probably doesn't work.");

  let ndofs = galmat.nrows();
  let dof_coeffs_opt = util::sparse_to_dense_data(dof_coeffs.to_vec(), ndofs);

  // Set entires zero that share a row index with a fixed dof.
  galmat.set_zero(|r, _| dof_coeffs_opt[r].is_some());

  // Set galmat diagonal for dofs to one.
  for &(i, _) in dof_coeffs {
    galmat.push(i, i, 1.0);
  }

  // Set galvec to prescribed coefficents.
  for &(i, v) in dof_coeffs.iter() {
    galvec[i] = v
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;
  use exterior::field::DiffFormClosure;
  use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::mesh::standard_coord_complex,
  };

  fn max_abs_matrix_diff(lhs: &Matrix, rhs: &Matrix) -> f64 {
    assert_eq!(lhs.nrows(), rhs.nrows());
    assert_eq!(lhs.ncols(), rhs.ncols());
    lhs
      .iter()
      .zip(rhs.iter())
      .map(|(l, r)| (l - r).abs())
      .fold(0.0, f64::max)
  }

  fn simplices_share_vertex(topology: &Complex, grade: ExteriorGrade, i: usize, j: usize) -> bool {
    let simplex_i = topology.skeleton(grade).handle_by_kidx(i);
    let simplex_j = topology.skeleton(grade).handle_by_kidx(j);
    simplex_i.iter().any(|vertex| simplex_j.contains(vertex))
  }

  #[test]
  fn barycentric_dual_local_reconstruction_identity_on_single_tetrahedron() {
    let (topology, coords) = standard_coord_complex(3);
    let config = BarycentricDualSparseInverseConfig::default();

    for grade in [1, 2] {
      for vertex in topology.vertices().handle_iter() {
        let local = barycentric_dual_sparse_inverse_local_block(
          &topology,
          &coords,
          vertex.kidx(),
          grade,
          config,
        )
        .expect("local barycentric-dual block should build");
        let actual = local.primal.transpose() * &local.dual;
        let expected = Matrix::identity(3, 3) * local.volume;

        assert!(
          max_abs_matrix_diff(&actual, &expected) <= 1e-12,
          "grade {grade}, vertex {} failed local reconstruction",
          vertex.kidx()
        );
        let reconstructed_primal = &local.inverse_block * &local.dual;
        assert!(
          max_abs_matrix_diff(&reconstructed_primal, &local.primal) <= 1e-12,
          "grade {grade}, vertex {} failed stabilized inverse consistency",
          vertex.kidx()
        );
        assert!(local.inverse_block.clone().cholesky().is_some());
      }
    }
  }

  #[test]
  fn barycentric_dual_stabilization_preserves_reconstruction_on_cube() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();

    for stabilization_factor in [1e-3, 1.0, 10.0] {
      let config = BarycentricDualSparseInverseConfig {
        stabilization_factor,
        ..BarycentricDualSparseInverseConfig::default()
      };
      for grade in [1, 2] {
        for vertex in topology.vertices().handle_iter() {
          let local = barycentric_dual_sparse_inverse_local_block(
            &topology,
            &coords,
            vertex.kidx(),
            grade,
            config,
          )
          .expect("local barycentric-dual block should build");
          let reconstructed_primal = &local.inverse_block * &local.dual;
          let tolerance = 1e-10 * local.primal.norm().max(1.0);
          assert!(
            max_abs_matrix_diff(&reconstructed_primal, &local.primal) <= tolerance,
            "grade {grade}, vertex {}, stabilization {stabilization_factor} failed stabilized inverse consistency",
            vertex.kidx()
          );
          assert!(local.inverse_block.clone().cholesky().is_some());
        }
      }
    }
  }

  #[test]
  fn barycentric_dual_sparse_inverse_is_symmetric_positive_and_vertex_local() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let config = BarycentricDualSparseInverseConfig::default();

    for grade in [1, 2] {
      let inverse = Matrix::from(
        &assemble_barycentric_dual_sparse_inverse_galmat_for_grade(
          &topology, &coords, grade, config,
        )
        .expect("barycentric-dual sparse inverse should assemble"),
      );

      assert_eq!(inverse.nrows(), topology.skeleton(grade).len());
      assert_eq!(inverse.ncols(), topology.skeleton(grade).len());
      assert!(max_abs_matrix_diff(&inverse, &inverse.transpose()) <= 1e-10);
      assert!(inverse.clone().cholesky().is_some());

      for i in 0..inverse.nrows() {
        assert!(inverse[(i, i)] > 0.0);
        for j in 0..inverse.ncols() {
          if !simplices_share_vertex(&topology, grade, i, j) {
            assert_abs_diff_eq!(inverse[(i, j)], 0.0, epsilon = 1e-12);
          }
        }
      }
    }
  }

  #[test]
  fn barycentric_dual_sparse_inverse_rejects_unsupported_meshes_and_grades() {
    let (surface_topology, surface_coords) = standard_coord_complex(2);
    let config = BarycentricDualSparseInverseConfig::default();
    let err = assemble_barycentric_dual_sparse_inverse_galmat_for_grade(
      &surface_topology,
      &surface_coords.embed_euclidean(3),
      1,
      config,
    )
    .expect_err("surface mesh should be rejected");
    assert!(err.contains("topological dimension 2"));

    let (topology, coords) = standard_coord_complex(3);
    let err =
      assemble_barycentric_dual_sparse_inverse_galmat_for_grade(&topology, &coords, 0, config)
        .expect_err("0-forms should be rejected");
    assert!(err.contains("Whitney 1- and 2-forms"));
  }

  #[test]
  fn boundary_term_interval_endpoints() {
    let (topology, coords) = standard_coord_complex(1);

    let a = 2.0;
    let b = -3.0;

    let boundary_data = DiffFormClosure::scalar(
      move |x| {
        if (x[0]).abs() < 1e-12 {
          a
        } else {
          b
        }
      },
      coords.dim(),
    );

    let v = assemble_boundary_integral_term(&topology, &coords, 0, &boundary_data, None, &|_| true);

    assert_eq!(v.len(), 2);
    assert_abs_diff_eq!(v[0], a, epsilon = 1e-12);
    assert_abs_diff_eq!(v[1], b, epsilon = 1e-12);
  }

  #[test]
  fn boundary_term_single_edge_whitney_integral_is_one() {
    let (topology, coords) = standard_coord_complex(2);

    let g = DiffFormClosure::scalar(|_| 1.0, coords.dim());

    let target_edge_kidx = topology.boundary_facets()[0].kidx;

    let v = assemble_boundary_integral_term(&topology, &coords, 1, &g, None, &|facet_kidx| {
      facet_kidx == target_edge_kidx
    });

    let val = v[target_edge_kidx];

    assert_abs_diff_eq!(val.abs(), 1.0, epsilon = 1e-10);
    for (idx, entry) in v.iter().enumerate() {
      if idx != target_edge_kidx {
        assert_abs_diff_eq!(*entry, 0.0, epsilon = 1e-12);
      }
    }
  }

  #[test]
  fn boundary_term_vertex_hat_integral_edge_length_half() {
    let (topology, coords) = standard_coord_complex(2);

    let target_edge_kidx = topology.boundary_facets()[0].kidx;
    let edge = topology.edges().handle_by_kidx(target_edge_kidx);
    let [v0_idx, v1_idx]: [usize; 2] = (*edge).clone().try_into().unwrap();

    let p0 = coords.coord(v0_idx);
    let p1 = coords.coord(v1_idx);
    let tangent = p1 - p0;
    let edge_length = tangent.norm();
    let unit_tangent = tangent / edge_length;

    let unit_tangent_clone = unit_tangent.clone_owned();
    let boundary_data =
      DiffFormClosure::one_form(move |_| unit_tangent_clone.clone(), coords.dim());

    let v =
      assemble_boundary_integral_term(&topology, &coords, 0, &boundary_data, None, &|facet_kidx| {
        facet_kidx == target_edge_kidx
      });

    assert_eq!(v.len(), topology.skeleton(0).len());
    assert_abs_diff_eq!(v[v0_idx], edge_length / 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(v[v1_idx], edge_length / 2.0, epsilon = 1e-10);

    for (idx, entry) in v.iter().enumerate() {
      if idx != v0_idx && idx != v1_idx {
        assert_abs_diff_eq!(*entry, 0.0, epsilon = 1e-12);
      }
    }
  }
}
