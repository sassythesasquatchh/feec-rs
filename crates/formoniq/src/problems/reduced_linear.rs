use crate::assemble::{
  assemble_barycentric_dual_sparse_inverse_galmat_for_grade, assemble_galmat,
  assemble_whitney_projected_sparse_inverse_galmat,
  assemble_whitney_projected_sparse_inverse_galmat_for_grade,
  assemble_whitney_projected_sparse_inverse_galmat_weighted, BarycentricDualSparseInverseConfig,
};
use crate::operators::{HodgeMassElmat, InnerProductWeightClosure};
use crate::problems::hodge_laplace::MixedGalmats;
use crate::problems::laplace_beltrami::LaplaceBeltramiGalmats;
use crate::reduction::{DofLayout, EssentialBoundarySpec, PrescribedDof};
use common::linalg::nalgebra::{CooMatrix, CsrMatrix, Matrix, Vector};
use manifold::geometry::coord::{mesh::MeshCoords, quadrature::SimplexQuadRule};
use manifold::geometry::metric::mesh::MeshLengths;
use manifold::topology::{complex::Complex, handle::KSimplexIdx};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy)]
pub enum MassInverseApproximation {
  RowSumLumped,
  WhitneyProjected,
  BarycentricDual(BarycentricDualSparseInverseConfig),
  ExactTopDegreeDiagonal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReducedHodgeOperators {
  pub mass: CsrMatrix,
  pub laplacian: CsrMatrix,
  pub lower_mass_inverse: Option<CsrMatrix>,
  pub state_mass_inverse: CsrMatrix,
  pub layout: DofLayout,
}

pub fn assemble_mass_inverse(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: Option<&MeshCoords>,
  grade: usize,
  approximation: MassInverseApproximation,
) -> Result<CsrMatrix, String> {
  if grade > topology.dim() {
    return Err(format!(
      "mass inverse grade {grade} exceeds mesh dimension {}",
      topology.dim()
    ));
  }
  let mass = CsrMatrix::from(&assemble_galmat(
    topology,
    geometry,
    HodgeMassElmat::new(topology.dim(), grade),
  ));
  match approximation {
    MassInverseApproximation::RowSumLumped => Ok(diag_matrix(&invert_diag(&lumped_diag(&mass)))),
    MassInverseApproximation::WhitneyProjected => {
      if grade == 0 || grade == topology.dim() {
        return Err(format!(
          "Whitney-projected inverse requires 0 < grade < {}, got {grade}",
          topology.dim()
        ));
      }
      Ok(CsrMatrix::from(
        &assemble_whitney_projected_sparse_inverse_galmat_for_grade(topology, geometry, grade),
      ))
    }
    MassInverseApproximation::BarycentricDual(config) => {
      let coords =
        coords.ok_or_else(|| "barycentric-dual inverse requires mesh coordinates".to_string())?;
      Ok(CsrMatrix::from(
        &assemble_barycentric_dual_sparse_inverse_galmat_for_grade(
          topology, coords, grade, config,
        )?,
      ))
    }
    MassInverseApproximation::ExactTopDegreeDiagonal => {
      if grade != topology.dim() {
        return Err(format!(
          "exact top-degree diagonal inverse requires grade {}, got {grade}",
          topology.dim()
        ));
      }
      Ok(diag_matrix(&invert_diag(&matrix_diag(&mass))))
    }
  }
}

pub fn assemble_reduced_hodge_operators(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: Option<&MeshCoords>,
  grade: usize,
  boundary: &EssentialBoundarySpec,
  lower_inverse: Option<MassInverseApproximation>,
  state_inverse: MassInverseApproximation,
) -> Result<ReducedHodgeOperators, String> {
  if grade == 0 {
    ensure_no_auxiliary_regions(boundary)?;
    let galmats = LaplaceBeltramiGalmats::compute(topology, geometry);
    let layout = build_state_layout(galmats.mass_csr().nrows(), &boundary.state)?;
    let mass = reduce_square_with_layout(&galmats.mass_csr(), &layout)?;
    let laplacian = reduce_square_with_layout(&galmats.stiffness_csr(), &layout)?;
    let inverse = assemble_mass_inverse(topology, geometry, coords, grade, state_inverse)?;
    return Ok(ReducedHodgeOperators {
      mass,
      laplacian,
      lower_mass_inverse: None,
      state_mass_inverse: reduce_square_with_layout(&inverse, &layout)?,
      layout,
    });
  }

  let lower_inverse = lower_inverse.ok_or_else(|| {
    format!(
      "grade-{grade} Hodge assembly requires a grade-{} inverse",
      grade - 1
    )
  })?;
  let galmats = MixedGalmats::compute(topology, geometry, grade);
  let context = build_mixed_1form_boundary_context(&galmats, boundary)?;
  let sigma_predicate = |kidx: KSimplexIdx| context.auxiliary_fixed_set.contains(&kidx);
  let state_predicate = |kidx: KSimplexIdx| context.state_fixed_set.contains(&kidx);
  let sigma_data = |kidx: KSimplexIdx| context.auxiliary_fixed_map[kidx].unwrap_or(0.0);
  let state_data = |kidx: KSimplexIdx| context.state_fixed_map[kidx].unwrap_or(0.0);
  let reduced_sigma_len = galmats.free_sigma_len(&sigma_predicate);
  let reduced_u_len = galmats.free_u_len(&state_predicate);
  let mut sigma_rhs = Vector::zeros(galmats.sigma_len());
  let mut u_rhs = Vector::zeros(galmats.u_len());
  let (mixed, _) = galmats.mixed_hodge_laplacian_with_strong_bc_via_elimination(
    &sigma_predicate,
    &sigma_data,
    &state_predicate,
    &state_data,
    &mut sigma_rhs,
    &mut u_rhs,
    &Matrix::zeros(reduced_u_len, 0),
  );
  let (mass_sigma, a12, a21, codifdif) =
    split_reduced_mixed_blocks(&mixed, reduced_sigma_len, reduced_u_len);
  let lower_layout = DofLayout::from_prescribed(galmats.sigma_len(), boundary.auxiliary.clone())?;
  let lower_full = assemble_mass_inverse(topology, geometry, coords, grade - 1, lower_inverse)?;
  let lower_reduced = reduce_square_with_layout(&lower_full, &lower_layout)?;
  if lower_reduced.nrows() != mass_sigma.nrows() {
    return Err(format!(
      "lower-form inverse dimension {} does not match reduced auxiliary dimension {}",
      lower_reduced.nrows(),
      mass_sigma.nrows()
    ));
  }
  let laplacian = if mass_sigma.nrows() == 0 {
    codifdif
  } else {
    add_sparse(
      &codifdif,
      &(&a21 * &lower_reduced * &scale_matrix(&a12, -1.0)),
    )
  };
  let mass = reduce_square_with_layout(&CsrMatrix::from(galmats.mass_u()), &context.layout)?;
  let state_full = assemble_mass_inverse(topology, geometry, coords, grade, state_inverse)?;
  let state_mass_inverse = reduce_square_with_layout(&state_full, &context.layout)?;

  Ok(ReducedHodgeOperators {
    mass,
    laplacian,
    lower_mass_inverse: Some(lower_reduced),
    state_mass_inverse,
    layout: context.layout,
  })
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReducedLinearPdeAssembly {
  pub operator: CsrMatrix,
  pub residual_bias: Vector,
  pub state_mass: CsrMatrix,
  pub state_mass_inverse: Option<CsrMatrix>,
  pub layout: DofLayout,
  pub forcing_operator: CsrMatrix,
  pub neumann_operator: CsrMatrix,
}

impl ReducedLinearPdeAssembly {
  pub fn residual_dimension(&self) -> usize {
    self.operator.nrows()
  }

  pub fn state_dimension(&self) -> usize {
    self.operator.ncols()
  }
}

pub fn build_reduced_laplace_beltrami_system(
  topology: &Complex,
  geometry: &MeshLengths,
  boundary: &EssentialBoundarySpec,
) -> Result<ReducedLinearPdeAssembly, String> {
  ensure_no_auxiliary_regions(boundary)?;
  let galmats = LaplaceBeltramiGalmats::compute(topology, geometry);
  let full_operator = galmats.stiffness_csr();
  let full_mass = galmats.mass_csr();
  let layout = build_state_layout(full_operator.nrows(), &boundary.state)?;
  let reduced_operator = reduce_square_with_layout(&full_operator, &layout)?;
  let reduced_mass = reduce_square_with_layout(&full_mass, &layout)?;
  let residual_bias = hard_dirichlet_bias(&full_operator, &layout);
  let residual_dim = reduced_operator.nrows();

  Ok(ReducedLinearPdeAssembly {
    operator: reduced_operator,
    residual_bias,
    state_mass: reduced_mass,
    state_mass_inverse: None,
    layout,
    forcing_operator: scaled_identity(residual_dim, -1.0),
    neumann_operator: scaled_identity(residual_dim, -1.0),
  })
}

pub fn build_reduced_hodge_laplace_1form_system(
  topology: &Complex,
  geometry: &MeshLengths,
  boundary: &EssentialBoundarySpec,
) -> Result<ReducedLinearPdeAssembly, String> {
  let galmats = MixedGalmats::compute(topology, geometry, 1);
  let state_mass_inverse = CsrMatrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
    topology, geometry,
  ));
  build_reduced_hodge_laplace_1form_system_with_galmats(&galmats, boundary, &state_mass_inverse)
}

pub fn build_reduced_weighted_hodge_laplace_1form_system(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
  boundary: &EssentialBoundarySpec,
) -> Result<ReducedLinearPdeAssembly, String> {
  let galmats = MixedGalmats::compute_weighted(topology, geometry, 1, coords, qr.clone(), weight);
  let state_mass_inverse =
    CsrMatrix::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
      topology, geometry, coords, qr, weight,
    ));
  build_reduced_hodge_laplace_1form_system_with_galmats(&galmats, boundary, &state_mass_inverse)
}

pub fn build_reduced_hodge_laplace_1form_system_with_galmats(
  galmats: &MixedGalmats,
  boundary: &EssentialBoundarySpec,
  state_mass_inverse: &CsrMatrix,
) -> Result<ReducedLinearPdeAssembly, String> {
  let context = build_mixed_1form_boundary_context(galmats, boundary)?;
  let (schur, schur_rhs) = schur_reduced_mixed_system(
    galmats,
    &context,
    &Vector::zeros(galmats.sigma_len()),
    &Vector::zeros(galmats.u_len()),
  )?;
  let reduced_mass =
    reduce_square_with_layout(&CsrMatrix::from(galmats.mass_u()), &context.layout)?;
  let reduced_mass_inverse = reduce_square_with_layout(state_mass_inverse, &context.layout)?;
  let residual_dim = reduced_mass.nrows();

  Ok(ReducedLinearPdeAssembly {
    operator: schur,
    residual_bias: -schur_rhs,
    state_mass: reduced_mass,
    state_mass_inverse: Some(reduced_mass_inverse),
    layout: context.layout,
    forcing_operator: scaled_identity(residual_dim, -1.0),
    neumann_operator: scaled_identity(residual_dim, -1.0),
  })
}

pub fn reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
  galmats: &MixedGalmats,
  boundary: &EssentialBoundarySpec,
  sigma_rhs: &Vector,
  u_rhs: &Vector,
) -> Result<Vector, String> {
  if sigma_rhs.len() != galmats.sigma_len() {
    return Err(format!(
      "sigma rhs length {} must match sigma dimension {}",
      sigma_rhs.len(),
      galmats.sigma_len()
    ));
  }
  if u_rhs.len() != galmats.u_len() {
    return Err(format!(
      "u rhs length {} must match state dimension {}",
      u_rhs.len(),
      galmats.u_len()
    ));
  }
  let context = build_mixed_1form_boundary_context(galmats, boundary)?;
  let (_, schur_rhs) = schur_reduced_mixed_system(galmats, &context, sigma_rhs, u_rhs)?;
  Ok(schur_rhs)
}

fn ensure_no_auxiliary_regions(boundary: &EssentialBoundarySpec) -> Result<(), String> {
  if !boundary.auxiliary.is_empty() {
    return Err("0-form systems do not support auxiliary boundary regions".to_string());
  }
  Ok(())
}

fn build_state_layout(
  full_dimension: usize,
  prescribed: &[PrescribedDof],
) -> Result<DofLayout, String> {
  DofLayout::from_prescribed(full_dimension, prescribed.to_vec())
}

fn fixed_map(
  full_dimension: usize,
  fixed_dofs: &[PrescribedDof],
) -> Result<Vec<Option<f64>>, String> {
  let mut fixed_map = vec![None; full_dimension];
  for fixed in fixed_dofs {
    if fixed_map[fixed.index].is_some() {
      return Err(format!("duplicate fixed dof {}", fixed.index));
    }
    fixed_map[fixed.index] = Some(fixed.value);
  }
  Ok(fixed_map)
}

fn reduced_index_map(layout: &DofLayout) -> Vec<Option<usize>> {
  let mut map = vec![None; layout.full_dimension];
  for (reduced, full) in layout.active_dofs.iter().copied().enumerate() {
    map[full] = Some(reduced);
  }
  map
}

fn reduce_square_with_layout(matrix: &CsrMatrix, layout: &DofLayout) -> Result<CsrMatrix, String> {
  if matrix.nrows() != layout.full_dimension || matrix.ncols() != layout.full_dimension {
    return Err(format!(
      "matrix reduction expected a {}x{} operator, got {}x{}",
      layout.full_dimension,
      layout.full_dimension,
      matrix.nrows(),
      matrix.ncols()
    ));
  }
  let reduced_map = reduced_index_map(layout);
  let mut reduced = CooMatrix::new(layout.reduced_dimension(), layout.reduced_dimension());
  for (row, col, value) in matrix.triplet_iter() {
    let Some(reduced_row) = reduced_map[row] else {
      continue;
    };
    let Some(reduced_col) = reduced_map[col] else {
      continue;
    };
    reduced.push(reduced_row, reduced_col, *value);
  }
  Ok(CsrMatrix::from(&reduced))
}

fn hard_dirichlet_bias(matrix: &CsrMatrix, layout: &DofLayout) -> Vector {
  let reduced_map = reduced_index_map(layout);
  let mut bias = Vector::zeros(layout.reduced_dimension());
  let fixed_map = layout
    .prescribed_dofs
    .iter()
    .map(|entry| (entry.index, entry.value))
    .collect::<BTreeMap<_, _>>();
  for (row, col, value) in matrix.triplet_iter() {
    let Some(reduced_row) = reduced_map[row] else {
      continue;
    };
    if let Some(fixed_value) = fixed_map.get(&col) {
      bias[reduced_row] += *value * *fixed_value;
    }
  }
  bias
}

fn split_reduced_mixed_blocks(
  matrix: &CsrMatrix,
  reduced_sigma_len: usize,
  reduced_u_len: usize,
) -> (CsrMatrix, CsrMatrix, CsrMatrix, CsrMatrix) {
  let mut mass_sigma = CooMatrix::new(reduced_sigma_len, reduced_sigma_len);
  let mut a12 = CooMatrix::new(reduced_sigma_len, reduced_u_len);
  let mut a21 = CooMatrix::new(reduced_u_len, reduced_sigma_len);
  let mut k_matrix = CooMatrix::new(reduced_u_len, reduced_u_len);
  let u_offset = reduced_sigma_len;
  let total = reduced_sigma_len + reduced_u_len;

  for (row, col, value) in matrix.triplet_iter() {
    if row >= total || col >= total {
      continue;
    }
    if row < reduced_sigma_len {
      if col < reduced_sigma_len {
        mass_sigma.push(row, col, *value);
      } else {
        a12.push(row, col - u_offset, *value);
      }
    } else if col < reduced_sigma_len {
      a21.push(row - u_offset, col, *value);
    } else {
      k_matrix.push(row - u_offset, col - u_offset, *value);
    }
  }

  (
    CsrMatrix::from(&mass_sigma),
    CsrMatrix::from(&a12),
    CsrMatrix::from(&a21),
    CsrMatrix::from(&k_matrix),
  )
}

#[derive(Debug, Clone)]
struct Mixed1FormBoundaryContext {
  layout: DofLayout,
  auxiliary_fixed_map: Vec<Option<f64>>,
  state_fixed_map: Vec<Option<f64>>,
  auxiliary_fixed_set: BTreeSet<usize>,
  state_fixed_set: BTreeSet<usize>,
}

fn build_mixed_1form_boundary_context(
  galmats: &MixedGalmats,
  boundary: &EssentialBoundarySpec,
) -> Result<Mixed1FormBoundaryContext, String> {
  boundary.validate(galmats.u_len(), galmats.sigma_len())?;
  let layout = build_state_layout(galmats.u_len(), &boundary.state)?;
  let auxiliary_fixed = boundary.auxiliary.clone();
  Ok(Mixed1FormBoundaryContext {
    auxiliary_fixed_map: fixed_map(galmats.sigma_len(), &auxiliary_fixed)?,
    state_fixed_map: fixed_map(galmats.u_len(), &layout.prescribed_dofs)?,
    auxiliary_fixed_set: auxiliary_fixed.iter().map(|entry| entry.index).collect(),
    state_fixed_set: layout
      .prescribed_dofs
      .iter()
      .map(|entry| entry.index)
      .collect(),
    layout,
  })
}

fn schur_reduced_mixed_system(
  galmats: &MixedGalmats,
  context: &Mixed1FormBoundaryContext,
  sigma_rhs: &Vector,
  u_rhs: &Vector,
) -> Result<(CsrMatrix, Vector), String> {
  let sigma_predicate = |kidx: KSimplexIdx| context.auxiliary_fixed_set.contains(&kidx);
  let state_predicate = |kidx: KSimplexIdx| context.state_fixed_set.contains(&kidx);
  let sigma_data = |kidx: KSimplexIdx| context.auxiliary_fixed_map[kidx].unwrap_or(0.0);
  let state_data = |kidx: KSimplexIdx| context.state_fixed_map[kidx].unwrap_or(0.0);
  let reduced_sigma_len = galmats.free_sigma_len(&sigma_predicate);
  let reduced_u_len = galmats.free_u_len(&state_predicate);

  let mut sigma_rhs = sigma_rhs.clone();
  let mut u_rhs = u_rhs.clone();
  let (reduced_mixed, reduced_rhs) = galmats.mixed_hodge_laplacian_with_strong_bc_via_elimination(
    &sigma_predicate,
    &sigma_data,
    &state_predicate,
    &state_data,
    &mut sigma_rhs,
    &mut u_rhs,
    &Matrix::zeros(context.layout.reduced_dimension(), 0),
  );
  Ok(schur_reduce_eliminated_mixed_system(
    &reduced_mixed,
    &reduced_rhs,
    reduced_sigma_len,
    reduced_u_len,
  ))
}

fn schur_reduce_eliminated_mixed_system(
  reduced_mixed: &CsrMatrix,
  reduced_rhs: &Vector,
  reduced_sigma_len: usize,
  reduced_u_len: usize,
) -> (CsrMatrix, Vector) {
  let (mass_sigma, a12, a21, k_matrix) =
    split_reduced_mixed_blocks(reduced_mixed, reduced_sigma_len, reduced_u_len);
  let mass_sigma_inv = diag_matrix(&invert_diag(&lumped_diag(&mass_sigma)));
  let schur = if mass_sigma.nrows() == 0 {
    k_matrix
  } else {
    add_sparse(
      &k_matrix,
      &(&a21 * &mass_sigma_inv * &scale_matrix(&a12, -1.0)),
    )
  };

  let (rhs_sigma, rhs_u) = split_reduced_rhs(reduced_rhs, reduced_sigma_len, reduced_u_len);
  let schur_rhs = if mass_sigma.nrows() == 0 {
    rhs_u
  } else {
    let sigma_correction = &mass_sigma_inv * &rhs_sigma;
    rhs_u - &a21 * sigma_correction
  };
  (schur, schur_rhs)
}

fn split_reduced_rhs(
  rhs: &Vector,
  reduced_sigma_len: usize,
  reduced_u_len: usize,
) -> (Vector, Vector) {
  let rhs_sigma = Vector::from_iterator(
    reduced_sigma_len,
    (0..reduced_sigma_len).map(|index| rhs[index]),
  );
  let rhs_u = Vector::from_iterator(
    reduced_u_len,
    (0..reduced_u_len).map(|index| rhs[reduced_sigma_len + index]),
  );
  (rhs_sigma, rhs_u)
}

fn lumped_diag(mat: &CsrMatrix) -> Vec<f64> {
  let mut diag = vec![0.0; mat.nrows()];
  for (row, _col, value) in mat.triplet_iter() {
    diag[row] += *value;
  }
  diag
}

fn matrix_diag(mat: &CsrMatrix) -> Vec<f64> {
  let mut diag = vec![0.0; mat.nrows()];
  for (row, col, value) in mat.triplet_iter() {
    if row == col {
      diag[row] += *value;
    }
  }
  diag
}

fn invert_diag(diag: &[f64]) -> Vec<f64> {
  let eps = 1e-12;
  diag
    .iter()
    .map(|value| if value.abs() < eps { 0.0 } else { 1.0 / value })
    .collect()
}

fn diag_matrix(diag: &[f64]) -> CsrMatrix {
  let mut coo = CooMatrix::new(diag.len(), diag.len());
  for (index, value) in diag.iter().copied().enumerate() {
    if value != 0.0 {
      coo.push(index, index, value);
    }
  }
  CsrMatrix::from(&coo)
}

fn scale_matrix(matrix: &CsrMatrix, scale: f64) -> CsrMatrix {
  let mut coo = CooMatrix::new(matrix.nrows(), matrix.ncols());
  for (row, col, value) in matrix.triplet_iter() {
    let scaled = *value * scale;
    if scaled != 0.0 {
      coo.push(row, col, scaled);
    }
  }
  CsrMatrix::from(&coo)
}

fn add_sparse(lhs: &CsrMatrix, rhs: &CsrMatrix) -> CsrMatrix {
  let mut coo = CooMatrix::from(lhs);
  for (row, col, value) in rhs.triplet_iter() {
    coo.push(row, col, *value);
  }
  CsrMatrix::from(&coo)
}

fn scaled_identity(dimension: usize, scale: f64) -> CsrMatrix {
  diag_matrix(&vec![scale; dimension])
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{assemble, operators::InnerProductWeightClosure};
  use manifold::gen::cartesian::CartesianMeshInfo;

  #[test]
  fn every_mass_inverse_strategy_is_grade_aware_and_finite() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let cases = [
      (0, MassInverseApproximation::RowSumLumped),
      (1, MassInverseApproximation::WhitneyProjected),
      (
        1,
        MassInverseApproximation::BarycentricDual(BarycentricDualSparseInverseConfig::default()),
      ),
      (3, MassInverseApproximation::ExactTopDegreeDiagonal),
    ];

    for (grade, strategy) in cases {
      let inverse = assemble_mass_inverse(&topology, &geometry, Some(&coords), grade, strategy)
        .expect("supported mass-inverse strategy should assemble");
      let dimension = topology.skeleton(grade).len();
      assert_eq!(inverse.nrows(), dimension);
      assert_eq!(inverse.ncols(), dimension);
      assert!(inverse
        .triplet_iter()
        .all(|(_, _, value)| value.is_finite()));
    }

    assert!(assemble_mass_inverse(
      &topology,
      &geometry,
      Some(&coords),
      1,
      MassInverseApproximation::ExactTopDegreeDiagonal,
    )
    .is_err());
  }

  #[test]
  fn square_reduction_and_prescribed_bias_use_one_canonical_layout() {
    let mut full = CooMatrix::new(3, 3);
    full.push(0, 0, 5.0);
    full.push(0, 1, 3.0);
    full.push(2, 1, -4.0);
    full.push(2, 2, 7.0);
    let full = CsrMatrix::from(&full);
    let layout = DofLayout::from_prescribed(
      3,
      vec![PrescribedDof {
        index: 1,
        value: 2.0,
      }],
    )
    .unwrap();

    let reduced = reduce_square_with_layout(&full, &layout).unwrap();
    let bias = hard_dirichlet_bias(&full, &layout);

    assert_eq!(layout.active_dofs, vec![0, 2]);
    assert_eq!(reduced.nrows(), 2);
    assert_eq!(reduced.ncols(), 2);
    assert_eq!(
      dense_from_sparse(&reduced),
      Matrix::from_diagonal(&Vector::from_vec(vec![5.0, 7.0]))
    );
    assert_eq!(bias.as_slice(), &[6.0, -8.0]);
  }

  #[test]
  fn two_form_hodge_assembly_keeps_lower_and_state_inverse_dimensions_distinct() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let assembled = assemble_reduced_hodge_operators(
      &topology,
      &geometry,
      Some(&coords),
      2,
      &EssentialBoundarySpec::default(),
      Some(MassInverseApproximation::WhitneyProjected),
      MassInverseApproximation::WhitneyProjected,
    )
    .expect("3D 2-form Hodge assembly should use the lower 1-form inverse");

    let lower = assembled
      .lower_mass_inverse
      .expect("mixed 2-form assembly should retain its lower inverse");
    assert_eq!(lower.nrows(), topology.nsimplices(1));
    assert_eq!(lower.ncols(), topology.nsimplices(1));
    assert_eq!(assembled.state_mass_inverse.nrows(), topology.nsimplices(2));
    assert_eq!(assembled.state_mass_inverse.ncols(), topology.nsimplices(2));
  }

  #[test]
  fn reduced_laplace_beltrami_system_eliminates_prescribed_dofs() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let soft = topology
      .boundary_subcomplex_simplices(0)
      .into_iter()
      .take(2)
      .map(|simp| simp.kidx)
      .collect::<Vec<_>>();
    let boundary = EssentialBoundarySpec::default().with_state(
      soft
        .iter()
        .copied()
        .zip([1.0, 2.0])
        .map(|(index, value)| PrescribedDof { index, value }),
    );
    let system = build_reduced_laplace_beltrami_system(&topology, &geometry, &boundary)
      .expect("0-form system should assemble");

    assert_eq!(
      system.layout.reduced_dimension(),
      system.layout.full_dimension - soft.len()
    );
    assert_eq!(system.layout.prescribed_dofs.len(), soft.len());
  }

  fn dense_from_sparse(matrix: &CsrMatrix) -> Matrix {
    let mut dense = Matrix::zeros(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
      dense[(row, col)] += *value;
    }
    dense
  }

  #[test]
  fn weighted_reduced_hodge_laplace_1form_matches_unweighted_for_unit_weight() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let auxiliary_dofs =
      assemble::boundary_simplices_where_barycenter(&topology, &coords, 0, |point| point[0] == 0.0)
        .into_iter()
        .take(1)
        .collect::<Vec<_>>();
    let boundary = EssentialBoundarySpec::default().with_auxiliary(
      auxiliary_dofs
        .into_iter()
        .map(|index| PrescribedDof { index, value: 0.0 }),
    );

    let unweighted = build_reduced_hodge_laplace_1form_system(&topology, &geometry, &boundary)
      .expect("unweighted system should assemble");
    let weighted = build_reduced_weighted_hodge_laplace_1form_system(
      &topology,
      &geometry,
      &coords,
      None,
      &InnerProductWeightClosure::new(|_| 1.0),
      &boundary,
    )
    .expect("weighted unit system should assemble");

    assert_eq!(unweighted.layout, weighted.layout);
    assert_eq!(
      unweighted.residual_bias.len(),
      weighted.residual_bias.len(),
      "residual bias lengths should match"
    );
    assert_eq!(
      unweighted.state_mass_inverse.as_ref().map(CsrMatrix::nrows),
      weighted.state_mass_inverse.as_ref().map(CsrMatrix::nrows),
      "state mass inverse dimensions should match"
    );

    let operator_diff =
      dense_from_sparse(&unweighted.operator) - dense_from_sparse(&weighted.operator);
    let mass_diff =
      dense_from_sparse(&unweighted.state_mass) - dense_from_sparse(&weighted.state_mass);
    let mass_inverse_diff = dense_from_sparse(
      unweighted
        .state_mass_inverse
        .as_ref()
        .expect("unweighted 1-form system should expose a reduced mass inverse"),
    ) - dense_from_sparse(
      weighted
        .state_mass_inverse
        .as_ref()
        .expect("weighted unit 1-form system should expose a reduced mass inverse"),
    );
    let bias_diff = &unweighted.residual_bias - &weighted.residual_bias;

    assert!(operator_diff.norm() <= 1e-10);
    assert!(mass_diff.norm() <= 1e-10);
    assert!(mass_inverse_diff.norm() <= 1e-10);
    assert!(bias_diff.norm() <= 1e-10);
  }

  #[test]
  fn weighted_reduced_hodge_laplace_1form_exposes_reduced_nc1_inverse() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let boundary = EssentialBoundarySpec::default();
    let weight = InnerProductWeightClosure::new(|point| 1.0 + point[0] + 0.5 * point[1]);

    let system = build_reduced_weighted_hodge_laplace_1form_system(
      &topology, &geometry, &coords, None, &weight, &boundary,
    )
    .expect("weighted 1-form system should assemble");

    let reduced_inverse = system
      .state_mass_inverse
      .as_ref()
      .expect("weighted 1-form system should expose a reduced NC1 projected inverse");
    assert_eq!(reduced_inverse.nrows(), system.state_dimension());
    assert_eq!(reduced_inverse.ncols(), system.state_dimension());
    assert!(reduced_inverse
      .triplet_iter()
      .all(|(_, _, value)| value.is_finite()));
  }
}
