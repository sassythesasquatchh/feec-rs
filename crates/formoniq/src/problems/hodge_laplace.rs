use crate::{
  assemble::{self, assemble_galmat, assemble_galmat_coord_aware, GalMat, GalVec},
  operators::{DofIdx, HodgeMassElmat, InnerProductWeightClosure},
  problems::transient::{validate_time_grid, ThetaMethod},
};

use {
  common::linalg::faer::FaerCholesky,
  common::linalg::petsc::{
    petsc_ghep_reduced_with_which, petsc_ghiep, petsc_ghiep_largest, petsc_saddle_point,
    GhepReducedOperators, GhiepReducedSolve, GhiepWhich,
  },
  ddf::{cochain::Cochain, ManifoldComplexExt},
  exterior::ExteriorGrade,
  manifold::geometry::coord::mesh::MeshCoords,
  manifold::geometry::coord::quadrature::SimplexQuadRule,
  manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex},
};

use common::linalg::nalgebra::{CooMatrix, CooMatrixExt, CsrMatrix, Matrix, Vector};
use itertools::Itertools;
use std::{collections::HashSet, mem};

use crate::operators::DofCoeff;
use manifold::topology::handle::KSimplexIdx;

#[derive(Debug, Clone)]
pub struct MixedTransientState {
  pub sigma: Cochain,
  pub u: Cochain,
}

pub struct MixedTransientConfig<'a> {
  pub times: &'a [f64],
  pub method: ThetaMethod,
  pub sigma_rhs_at: &'a dyn Fn(f64) -> GalVec,
  pub u_rhs_at: &'a dyn Fn(f64) -> GalVec,
  pub k_strong_bc_predicate: Option<&'a dyn Fn(KSimplexIdx) -> bool>,
  pub k_strong_bc_data_at: Option<&'a dyn Fn(f64, KSimplexIdx) -> DofCoeff>,
  pub k_minus_one_strong_bc_predicate: Option<&'a dyn Fn(KSimplexIdx) -> bool>,
  pub k_minus_one_strong_bc_data_at: Option<&'a dyn Fn(f64, KSimplexIdx) -> DofCoeff>,
}

impl MixedTransientConfig<'_> {
  fn validate(&self) {
    validate_time_grid(self.times);
    assert!(
      self.k_strong_bc_predicate.is_some() == self.k_strong_bc_data_at.is_some(),
      "k-form boundary predicate and data must either both be present or both be absent."
    );
    assert!(
      self.k_minus_one_strong_bc_predicate.is_some()
        == self.k_minus_one_strong_bc_data_at.is_some(),
      "k-1-form boundary predicate and data must either both be present or both be absent."
    );
  }
}

pub fn solve_hodge_laplace_source(
  topology: &Complex,
  geometry: &MeshLengths,
  source_galvec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_inner(
    topology,
    geometry,
    None,
    source_galvec,
    grade,
    homology_dim,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
  )
}

// Low-level FEEC compatibility entry point: every argument selects an independently
// assembled part of the weighted mixed problem. The integration crate supplies the
// higher-level builder API; weighted convergence studies cover this exact route.
#[allow(clippy::too_many_arguments)]
pub fn solve_weighted_hodge_laplace_source(
  topology: &Complex,
  geometry: &MeshLengths,
  source_galvec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_inner(
    topology,
    geometry,
    None,
    source_galvec,
    grade,
    homology_dim,
    Some(coords),
    qr,
    Some(weight),
    None,
    None,
    None,
    None,
  )
}

// The four boundary callbacks represent two mathematically distinct trace spaces;
// keeping them explicit prevents accidental cross-grade reuse. Mixed-BC regression
// tests exercise the resulting eliminated system.
#[allow(clippy::too_many_arguments)]
pub fn solve_weighted_hodge_laplace_source_with_boundary_conditions(
  topology: &Complex,
  geometry: &MeshLengths,
  sigma_vec: Option<GalVec>,
  u_vec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
  k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
  k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_minus_one_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_inner(
    topology,
    geometry,
    sigma_vec,
    u_vec,
    grade,
    homology_dim,
    Some(coords),
    qr,
    Some(weight),
    Some(k_strong_bc_predicate),
    Some(k_strong_bc_data),
    Some(k_minus_one_strong_bc_predicate),
    Some(k_minus_one_strong_bc_data),
  )
}

/// Recommended flow when reusing assembled matrices:
/// ```text
/// let galmats = MixedGalmats::compute(&topology, &metric, grade);
/// let harmonics = solve_hodge_laplace_harmonics_with_galmats(
///   &topology, &galmats, grade, homology_dim, None, None
/// );
/// let (sigma, u, p) = solve_hodge_laplace_source_with_galmats(
///   &topology, &galmats, source, grade, homology_dim
/// );
/// ```
/// This avoids reassembling FEEC operators when solving multiple related problems.
/// Variant that reuses preassembled mixed matrices.
pub fn solve_hodge_laplace_source_with_galmats(
  topology: &Complex,
  galmats: &MixedGalmats,
  source_galvec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_with_galmats_inner(
    topology,
    galmats,
    None,
    source_galvec,
    grade,
    homology_dim,
    None,
    None,
    None,
    None,
  )
}

/// Variant that reuses preassembled mixed matrices with strong boundary conditions.
// Boundary selectors and data remain paired by form degree; the focused strong-BC
// source-solver test covers this compatibility entry point.
#[allow(clippy::too_many_arguments)]
pub fn solve_hodge_laplace_source_with_galmats_and_boundary_conditions(
  topology: &Complex,
  galmats: &MixedGalmats,
  sigma_vec: Option<GalVec>,
  u_vec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
  k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_minus_one_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
) -> (Cochain, Cochain, Cochain) {
  solve_hodge_laplace_source_with_galmats_inner(
    topology,
    galmats,
    sigma_vec,
    u_vec,
    grade,
    homology_dim,
    Some(k_strong_bc_predicate),
    Some(k_strong_bc_data),
    Some(k_minus_one_strong_bc_predicate),
    Some(k_minus_one_strong_bc_data),
  )
}

pub fn solve_hodge_laplace_transient(
  topology: &Complex,
  geometry: &MeshLengths,
  initial_u: Cochain,
  initial_sigma: Option<Cochain>,
  grade: ExteriorGrade,
  config: MixedTransientConfig<'_>,
) -> Vec<MixedTransientState> {
  let galmats = MixedGalmats::compute(topology, geometry, grade);
  solve_hodge_laplace_transient_with_galmats(
    topology,
    &galmats,
    initial_u,
    initial_sigma,
    grade,
    config,
  )
}

// Weighted assembly and time integration are independent FEEC concerns at this
// low-level boundary. Transient zero-state and boundary-reintroduction tests cover it.
#[allow(clippy::too_many_arguments)]
pub fn solve_weighted_hodge_laplace_transient(
  topology: &Complex,
  geometry: &MeshLengths,
  initial_u: Cochain,
  initial_sigma: Option<Cochain>,
  grade: ExteriorGrade,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
  config: MixedTransientConfig<'_>,
) -> Vec<MixedTransientState> {
  let galmats = MixedGalmats::compute_weighted(topology, geometry, grade, coords, qr, weight);
  solve_hodge_laplace_transient_with_galmats(
    topology,
    &galmats,
    initial_u,
    initial_sigma,
    grade,
    config,
  )
}

pub fn solve_hodge_laplace_transient_with_galmats(
  topology: &Complex,
  galmats: &MixedGalmats,
  initial_u: Cochain,
  initial_sigma: Option<Cochain>,
  grade: ExteriorGrade,
  config: MixedTransientConfig<'_>,
) -> Vec<MixedTransientState> {
  config.validate();
  debug_assert_eq!(galmats.u_len(), topology.nsimplices(grade));
  if grade > 0 {
    debug_assert_eq!(galmats.sigma_len(), topology.nsimplices(grade - 1));
  }

  assert_eq!(
    initial_u.dim(),
    grade,
    "initial u cochain must have grade {grade}, got {}.",
    initial_u.dim()
  );
  assert_eq!(
    initial_u.len(),
    galmats.u_len(),
    "initial u cochain length must equal {} dofs, got {}.",
    galmats.u_len(),
    initial_u.len()
  );

  let initial_sigma = initial_sigma_from_state(
    galmats,
    &initial_u,
    initial_sigma,
    grade,
    config.times[0],
    &config,
  );

  let mut solution = Vec::with_capacity(config.times.len());
  solution.push(MixedTransientState {
    sigma: initial_sigma,
    u: initial_u,
  });

  for t01 in config.times.windows(2) {
    let [t0, t1] = t01 else { unreachable!() };
    let prev = solution.last().unwrap();
    let next = solve_hodge_laplace_transient_step(galmats, grade, prev, *t0, *t1, &config);
    solution.push(next);
  }

  solution
}

// This private kernel receives the complete mixed weak problem: forcing, weighted
// assembly, harmonic dimension, and trace data. Focused mixed-BC tests validate it.
#[allow(clippy::too_many_arguments)]
fn solve_hodge_laplace_source_inner(
  topology: &Complex,
  geometry: &MeshLengths,
  sigma_vec: Option<GalVec>,
  u_vec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_data: Option<&dyn Fn(KSimplexIdx) -> DofCoeff>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_minus_one_strong_bc_data: Option<&dyn Fn(KSimplexIdx) -> DofCoeff>,
) -> (Cochain, Cochain, Cochain) {
  let galmats = if let (Some(coords), Some(weight)) = (coords, weight) {
    MixedGalmats::compute_weighted(topology, geometry, grade, coords, qr.clone(), weight)
  } else {
    MixedGalmats::compute(topology, geometry, grade)
  };

  solve_hodge_laplace_source_with_galmats_inner(
    topology,
    &galmats,
    sigma_vec,
    u_vec,
    grade,
    homology_dim,
    k_strong_bc_predicate,
    k_strong_bc_data,
    k_minus_one_strong_bc_predicate,
    k_minus_one_strong_bc_data,
  )
}

// Each callback belongs to a distinct k or (k-1) trace block in the mixed system.
// The source solver's strong-boundary regression test validates their placement.
#[allow(clippy::too_many_arguments)]
fn solve_hodge_laplace_source_with_galmats_inner(
  topology: &Complex,
  galmats: &MixedGalmats,
  sigma_vec: Option<GalVec>,
  u_vec: GalVec,
  grade: ExteriorGrade,
  homology_dim: usize,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_data: Option<&dyn Fn(KSimplexIdx) -> DofCoeff>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_minus_one_strong_bc_data: Option<&dyn Fn(KSimplexIdx) -> DofCoeff>,
) -> (Cochain, Cochain, Cochain) {
  debug_assert_eq!(galmats.u_len(), topology.nsimplices(grade));
  if grade > 0 {
    debug_assert_eq!(galmats.sigma_len(), topology.nsimplices(grade - 1));
  }

  let harmonics = solve_hodge_laplace_harmonics_with_galmats(
    topology,
    galmats,
    grade,
    homology_dim,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
  );

  let sigma_len = if let Some(pred) = k_minus_one_strong_bc_predicate {
    galmats.free_sigma_len(pred)
  } else {
    galmats.sigma_len()
  };

  let u_len = if let Some(pred) = k_strong_bc_predicate {
    galmats.free_u_len(pred)
  } else {
    galmats.u_len()
  };

  let sigma_rhs_len = galmats.sigma_len();
  let u_rhs_len = galmats.u_len();

  let sigma_vec = if let Some(sigma_vec) = sigma_vec {
    assert_eq!(
      sigma_vec.len(),
      sigma_rhs_len,
      "sigma rhs must have full length {}, got {}.",
      sigma_rhs_len,
      sigma_vec.len()
    );
    sigma_vec
  } else {
    Vector::zeros(sigma_rhs_len)
  };
  assert_eq!(
    u_vec.len(),
    u_rhs_len,
    "u rhs must have full length {}, got {}.",
    u_rhs_len,
    u_vec.len()
  );
  let mut owned_sigma_vec = sigma_vec.into_owned();
  let mut owned_u_vec = u_vec;

  let (system_matrix, rhs) = galmats.mixed_hodge_laplacian_with_strong_bc_via_elimination(
    k_minus_one_strong_bc_predicate.unwrap_or(&|_| false),
    k_minus_one_strong_bc_data.unwrap_or(&|_| 0.0),
    k_strong_bc_predicate.unwrap_or(&|_| false),
    k_strong_bc_data.unwrap_or(&|_| 0.0),
    &mut owned_sigma_vec,
    &mut owned_u_vec,
    &harmonics,
  );

  let galsol = petsc_saddle_point(&system_matrix, &rhs, harmonics.ncols() > 0).into_owned();

  let sigma = if let (Some(k_minus_one_strong_bc_predicate), Some(k_minus_one_strong_bc_data)) =
    (k_minus_one_strong_bc_predicate, k_minus_one_strong_bc_data)
  {
    let boundary_data = (0..galmats.sigma_len())
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .map(|i| (i, k_minus_one_strong_bc_data(i)))
      .collect::<Vec<_>>();
    let mut temp_sigma = galsol.view_range(..sigma_len, 0).into_owned();
    assemble::reintroduce_non_homogenous_dofs_galsols(&boundary_data, &mut temp_sigma);
    Cochain::new(sigma_grade(grade), temp_sigma)
  } else {
    Cochain::new(
      sigma_grade(grade),
      galsol.view_range(..sigma_len, 0).into_owned(),
    )
  };

  let u = if let (Some(k_strong_bc_predicate), Some(k_strong_bc_data)) =
    (k_strong_bc_predicate, k_strong_bc_data)
  {
    let boundary_data = (0..galmats.u_len())
      .filter(|&i| k_strong_bc_predicate(i))
      .map(|i| (i, k_strong_bc_data(i)))
      .collect::<Vec<_>>();
    let mut temp_u = galsol
      .view_range(sigma_len..sigma_len + u_len, 0)
      .into_owned();
    assemble::reintroduce_non_homogenous_dofs_galsols(&boundary_data, &mut temp_u);
    Cochain::new(grade, temp_u)
  } else {
    Cochain::new(
      grade,
      galsol
        .view_range(sigma_len..sigma_len + u_len, 0)
        .into_owned(),
    )
  };

  let p = Cochain::new(
    grade,
    galsol.view_range(sigma_len + u_len.., 0).into_owned(),
  );
  (sigma, u, p)
}

fn sigma_grade(grade: ExteriorGrade) -> ExteriorGrade {
  grade.saturating_sub(1)
}

fn boundary_data_pairs(
  ndofs: usize,
  time: f64,
  predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  data_at: Option<&dyn Fn(f64, KSimplexIdx) -> DofCoeff>,
) -> Vec<(DofIdx, f64)> {
  let (Some(predicate), Some(data_at)) = (predicate, data_at) else {
    return Vec::new();
  };

  (0..ndofs)
    .filter(|&i| predicate(i))
    .map(|i| (i, data_at(time, i)))
    .collect()
}

fn rhs_at(rhs_at: &dyn Fn(f64) -> GalVec, time: f64, expected_len: usize, label: &str) -> Vector {
  let rhs = rhs_at(time);
  assert_eq!(
    rhs.len(),
    expected_len,
    "{label} must have length {expected_len}, got {} at time {time}.",
    rhs.len()
  );
  rhs
}

fn consistent_sigma_from_u(
  galmats: &MixedGalmats,
  u: &Cochain,
  time: f64,
  config: &MixedTransientConfig<'_>,
) -> Cochain {
  let sigma_grade = sigma_grade(u.dim());
  if galmats.sigma_len() == 0 {
    return Cochain::new(sigma_grade, Vector::zeros(0));
  }

  let mass_sigma = galmats.mass_sigma().clone();
  let codif_u = CsrMatrix::from(galmats.codif_u());
  let sigma_rhs = rhs_at(config.sigma_rhs_at, time, galmats.sigma_len(), "sigma rhs");
  let rhs = sigma_rhs + codif_u * u.coeffs();

  let k_minus_one_boundary_data = boundary_data_pairs(
    galmats.sigma_len(),
    time,
    config.k_minus_one_strong_bc_predicate,
    config.k_minus_one_strong_bc_data_at,
  );

  let mut mass_sigma = mass_sigma;
  let mut rhs = rhs;
  fix_dofs_coeff_strong_coo(&k_minus_one_boundary_data, &mut mass_sigma, &mut rhs);

  let sigma = FaerCholesky::new(CsrMatrix::from(&mass_sigma)).solve(&rhs);
  Cochain::new(sigma_grade, sigma)
}

fn initial_sigma_from_state(
  galmats: &MixedGalmats,
  initial_u: &Cochain,
  initial_sigma: Option<Cochain>,
  grade: ExteriorGrade,
  time: f64,
  config: &MixedTransientConfig<'_>,
) -> Cochain {
  let consistent = consistent_sigma_from_u(galmats, initial_u, time, config);
  if let Some(initial_sigma) = initial_sigma {
    assert_eq!(
      initial_sigma.dim(),
      sigma_grade(grade),
      "initial sigma cochain must have grade {}, got {}.",
      sigma_grade(grade),
      initial_sigma.dim()
    );
    assert_eq!(
      initial_sigma.len(),
      galmats.sigma_len(),
      "initial sigma cochain length must equal {} dofs, got {}.",
      galmats.sigma_len(),
      initial_sigma.len()
    );

    let max_diff = initial_sigma
      .coeffs()
      .iter()
      .zip(consistent.coeffs().iter())
      .map(|(lhs, rhs)| (lhs - rhs).abs())
      .fold(0.0, f64::max);
    assert!(
      max_diff <= 1e-9,
      "initial sigma is inconsistent with the mixed algebraic relation at t0; max abs diff = {max_diff}."
    );
    initial_sigma
  } else {
    consistent
  }
}

// Arguments are the four algebraic blocks, two right-hand sides, and the two
// grade-specific boundary maps of one elimination equation. Transient boundary
// reintroduction tests exercise this kernel.
#[allow(clippy::too_many_arguments)]
fn transient_system_with_strong_bc_via_elimination(
  mass_sigma: &GalMat,
  codif_u: &GalMat,
  dif_sigma: &GalMat,
  u_block: &GalMat,
  sigma_rhs: &Vector,
  u_rhs: &Vector,
  k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_minus_one_boundary_data: &[(DofIdx, f64)],
  k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_boundary_data: &[(DofIdx, f64)],
) -> (CsrMatrix, Vector) {
  let mut mass_sigma = mass_sigma.clone();
  let mut dif_sigma = dif_sigma.clone();
  let mut codif_u_neg = codif_u.clone().neg();
  let mut u_block = u_block.clone();
  let mut sigma_rhs = sigma_rhs.clone();
  let mut u_rhs = u_rhs.clone();

  fix_dofs_coeff_strong_coo(k_minus_one_boundary_data, &mut mass_sigma, &mut sigma_rhs);
  fix_dofs_coeff_strong_coo(k_boundary_data, &mut u_block, &mut u_rhs);

  fix_dofs_coeff_strong_coo_rectangular(
    k_minus_one_boundary_data,
    k_strong_bc_predicate,
    &mut dif_sigma,
    &mut u_rhs,
  );

  fix_dofs_coeff_strong_coo_rectangular(
    k_boundary_data,
    k_minus_one_strong_bc_predicate,
    &mut codif_u_neg,
    &mut sigma_rhs,
  );

  let k_strongly_enforced_dofs = (0..u_block.nrows())
    .filter(|&i| k_strong_bc_predicate(i))
    .collect::<HashSet<_>>();

  let k_minus_one_strongly_enforced_dofs = (0..mass_sigma.nrows())
    .filter(|&i| k_minus_one_strong_bc_predicate(i))
    .collect::<HashSet<_>>();

  assemble::drop_dofs_galmat(&k_minus_one_strongly_enforced_dofs, &mut mass_sigma);
  assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut u_block);
  assemble::drop_dofs_rectangular_galmat(
    &k_strongly_enforced_dofs,
    &k_minus_one_strongly_enforced_dofs,
    &mut dif_sigma,
  );
  assemble::drop_dofs_rectangular_galmat(
    &k_minus_one_strongly_enforced_dofs,
    &k_strongly_enforced_dofs,
    &mut codif_u_neg,
  );

  let mut sigma_drop = k_minus_one_strongly_enforced_dofs
    .iter()
    .copied()
    .collect::<Vec<_>>();
  sigma_drop.sort_unstable();
  let mut u_drop = k_strongly_enforced_dofs.iter().copied().collect::<Vec<_>>();
  u_drop.sort_unstable();

  assemble::drop_dofs_galvec(&sigma_drop, &mut sigma_rhs);
  assemble::drop_dofs_galvec(&u_drop, &mut u_rhs);

  let system_matrix = CsrMatrix::from(&CooMatrix::block(&[
    &[&mass_sigma, &codif_u_neg],
    &[&dif_sigma, &u_block],
  ]));
  let rhs = Vector::from_iterator(
    sigma_rhs.len() + u_rhs.len(),
    sigma_rhs.iter().chain(u_rhs.iter()).copied(),
  );

  (system_matrix, rhs)
}

fn solve_hodge_laplace_transient_step(
  galmats: &MixedGalmats,
  grade: ExteriorGrade,
  prev: &MixedTransientState,
  t0: f64,
  t1: f64,
  config: &MixedTransientConfig<'_>,
) -> MixedTransientState {
  let dt = t1 - t0;
  let theta = config.method.theta();
  let sigma_rhs = rhs_at(config.sigma_rhs_at, t1, galmats.sigma_len(), "sigma rhs");
  let u_rhs_0 = rhs_at(config.u_rhs_at, t0, galmats.u_len(), "u rhs");
  let u_rhs_1 = rhs_at(config.u_rhs_at, t1, galmats.u_len(), "u rhs");
  let u_rhs_theta = (1.0 - theta) * u_rhs_0 + theta * u_rhs_1;

  let dif_sigma = CsrMatrix::from(galmats.dif_sigma());
  let codifdif_u = CsrMatrix::from(galmats.codifdif_u());
  let mass_u = galmats.mass_u_csr();

  let prev_coupling = &dif_sigma * prev.sigma.coeffs() + &codifdif_u * prev.u.coeffs();
  let u_rhs =
    (1.0 / dt) * (&mass_u * prev.u.coeffs()) - (1.0 - theta) * prev_coupling + u_rhs_theta;

  let dif_sigma_theta = CooMatrix::from(&(theta * &dif_sigma));
  let u_block = CooMatrix::from(&((1.0 / dt) * &mass_u + theta * &codifdif_u));

  let k_boundary_data = boundary_data_pairs(
    galmats.u_len(),
    t1,
    config.k_strong_bc_predicate,
    config.k_strong_bc_data_at,
  );
  let k_minus_one_boundary_data = boundary_data_pairs(
    galmats.sigma_len(),
    t1,
    config.k_minus_one_strong_bc_predicate,
    config.k_minus_one_strong_bc_data_at,
  );

  let (system_matrix, rhs) = transient_system_with_strong_bc_via_elimination(
    galmats.mass_sigma(),
    galmats.codif_u(),
    &dif_sigma_theta,
    &u_block,
    &sigma_rhs,
    &u_rhs,
    config.k_minus_one_strong_bc_predicate.unwrap_or(&|_| false),
    &k_minus_one_boundary_data,
    config.k_strong_bc_predicate.unwrap_or(&|_| false),
    &k_boundary_data,
  );

  let solution = petsc_saddle_point(&system_matrix, &rhs, false);

  let sigma_len = if let Some(pred) = config.k_minus_one_strong_bc_predicate {
    galmats.free_sigma_len(pred)
  } else {
    galmats.sigma_len()
  };
  let u_len = if let Some(pred) = config.k_strong_bc_predicate {
    galmats.free_u_len(pred)
  } else {
    galmats.u_len()
  };

  let mut sigma = solution.rows(0, sigma_len).into_owned();
  assemble::reintroduce_non_homogenous_dofs_galsols(&k_minus_one_boundary_data, &mut sigma);

  let mut u = solution.rows(sigma_len, u_len).into_owned();
  assemble::reintroduce_non_homogenous_dofs_galsols(&k_boundary_data, &mut u);

  MixedTransientState {
    sigma: Cochain::new(sigma_grade(grade), sigma),
    u: Cochain::new(grade, u),
  }
}

pub fn solve_hodge_laplace_harmonics(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  homology_dim: usize,
) -> Matrix {
  solve_hodge_laplace_harmonics_inner(
    topology,
    geometry,
    grade,
    homology_dim,
    None,
    None,
    None,
    None,
    None,
  )
}

pub fn solve_weighted_hodge_laplace_harmonics(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> Matrix {
  solve_hodge_laplace_harmonics_inner(
    topology,
    geometry,
    grade,
    homology_dim,
    Some(coords),
    qr,
    Some(weight),
    None,
    None,
  )
}

/// Variant that reuses preassembled mixed matrices.
pub fn solve_hodge_laplace_harmonics_with_galmats(
  topology: &Complex,
  galmats: &MixedGalmats,
  grade: ExteriorGrade,
  homology_dim: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
) -> Matrix {
  debug_assert_eq!(galmats.u_len(), topology.nsimplices(grade));
  if grade > 0 {
    debug_assert_eq!(galmats.sigma_len(), topology.nsimplices(grade - 1));
  }

  if homology_dim == 0 {
    return Matrix::zeros(galmats.u_len(), 0);
  }

  let (eigenvals, _, harmonics) = solve_hodge_laplace_evp_with_galmats(
    galmats,
    homology_dim,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
  );

  if !eigenvals.iter().all(|&eigenval| eigenval <= 1e-12) {
    panic!(
      "Expected zero eigenvalues for harmonic forms, but got eigenvalues: {:?}",
      eigenvals
    );
  }
  harmonics
}

// Weighted metric data and the two grade-specific boundary restrictions define
// independent parts of the harmonic eigenproblem. Harmonic-dimension tests cover it.
#[allow(clippy::too_many_arguments)]
fn solve_hodge_laplace_harmonics_inner(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  homology_dim: usize,
  coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
) -> Matrix {
  let galmats = if let (Some(coords), Some(weight)) = (coords, weight) {
    MixedGalmats::compute_weighted(topology, geometry, grade, coords, qr, weight)
  } else {
    MixedGalmats::compute(topology, geometry, grade)
  };

  solve_hodge_laplace_harmonics_with_galmats(
    topology,
    &galmats,
    grade,
    homology_dim,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
  )
}

pub fn solve_hodge_laplace_evp(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner(
    topology,
    geometry,
    grade,
    neigen_values,
    None,
    None,
    None,
    None,
    None,
    GhiepWhich::Smallest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_hodge_laplace_evp_config(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner(
    topology,
    geometry,
    grade,
    neigen_values,
    None,
    None,
    None,
    None,
    None,
    which,
    mass_solve,
  )
}

pub fn solve_hodge_laplace_evp_largest(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner_with_solver(
    topology,
    geometry,
    grade,
    neigen_values,
    None,
    None,
    None,
    None,
    None,
    GhiepWhich::Largest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_weighted_hodge_laplace_evp(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner(
    topology,
    geometry,
    grade,
    neigen_values,
    Some(coords),
    qr,
    Some(weight),
    None,
    None,
    GhiepWhich::Smallest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_weighted_hodge_laplace_evp_largest(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner_with_solver(
    topology,
    geometry,
    grade,
    neigen_values,
    Some(coords),
    qr,
    Some(weight),
    None,
    None,
    GhiepWhich::Largest,
    GhiepReducedSolve::Direct,
  )
}

// Low-level weighted EVP compatibility API; each boundary selector belongs to a
// different FEEC trace space. EVP regression studies exercise this path.
#[allow(clippy::too_many_arguments)]
pub fn solve_weighted_hodge_laplace_evp_with_boundary_conditions(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
  k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner(
    topology,
    geometry,
    grade,
    neigen_values,
    Some(coords),
    qr,
    Some(weight),
    Some(k_minus_one_strong_bc_predicate),
    Some(k_strong_bc_predicate),
    GhiepWhich::Smallest,
    GhiepReducedSolve::Direct,
  )
}

// Largest-mode counterpart to the weighted EVP compatibility API above.
#[allow(clippy::too_many_arguments)]
pub fn solve_weighted_hodge_laplace_evp_with_boundary_conditions_largest(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
  k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner_with_solver(
    topology,
    geometry,
    grade,
    neigen_values,
    Some(coords),
    qr,
    Some(weight),
    Some(k_minus_one_strong_bc_predicate),
    Some(k_strong_bc_predicate),
    GhiepWhich::Largest,
    GhiepReducedSolve::Direct,
  )
}

/// Variant that reuses preassembled mixed matrices.
pub fn solve_hodge_laplace_evp_with_galmats(
  galmats: &MixedGalmats,
  neigen_values: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_with_galmats_impl(
    galmats,
    neigen_values,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    GhiepWhich::Smallest,
    GhiepReducedSolve::Direct,
  )
}

pub fn solve_hodge_laplace_evp_with_galmats_config(
  galmats: &MixedGalmats,
  neigen_values: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_with_galmats_impl(
    galmats,
    neigen_values,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    which,
    mass_solve,
  )
}

pub fn solve_hodge_laplace_evp_largest_with_galmats(
  galmats: &MixedGalmats,
  neigen_values: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_with_galmats_impl(
    galmats,
    neigen_values,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    GhiepWhich::Largest,
    GhiepReducedSolve::Direct,
  )
}

fn solve_hodge_laplace_evp_with_galmats_impl(
  galmats: &MixedGalmats,
  neigen_values: usize,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  let (evals, evec_sigmas, evec_us) = if which == GhiepWhich::Smallest {
    let (lhs, sigma_len, u_len) = if let (Some(km1_pred), Some(k_pred)) =
      (k_minus_one_strong_bc_predicate, k_strong_bc_predicate)
    {
      (
        galmats.reduced_mixed_hodge_laplacian(km1_pred, k_pred),
        galmats.free_sigma_len(km1_pred),
        galmats.free_u_len(k_pred),
      )
    } else {
      (
        galmats.mixed_hodge_laplacian(),
        galmats.sigma_len(),
        galmats.u_len(),
      )
    };

    let mut rhs = CooMatrix::zeros(sigma_len + u_len, sigma_len + u_len);

    let mass_u = if let Some(k_pred) = k_strong_bc_predicate {
      let mut mass_u = galmats.mass_u().clone();
      let k_strong_bc_set = (0..mass_u.nrows())
        .filter(|&i| k_pred(i))
        .collect::<HashSet<_>>();
      assemble::drop_dofs_galmat(&k_strong_bc_set, &mut mass_u);
      mass_u
    } else {
      galmats.mass_u().clone()
    };

    for (mut r, mut c, &v) in mass_u.triplet_iter() {
      r += sigma_len;
      c += sigma_len;
      rhs.push(r, c, v);
    }

    let (eigenvals, eigenvectors) = petsc_ghiep(&(&lhs).into(), &(&rhs).into(), neigen_values);

    let eigen_sigmas = eigenvectors.rows(0, sigma_len).into_owned();
    let eigen_us = eigenvectors.rows(sigma_len, u_len).into_owned();

    (eigenvals, eigen_sigmas, eigen_us)
  } else {
    let mut mass_sigma = galmats.mass_sigma().clone();
    let mut dif_sigma = galmats.dif_sigma().clone();
    let mut codif_u = galmats.codif_u().clone();
    let mut codifdif_u = galmats.codifdif_u().clone();
    let mut mass_u = galmats.mass_u().clone();

    if let (Some(km1_pred), Some(k_pred)) = (k_minus_one_strong_bc_predicate, k_strong_bc_predicate)
    {
      // TODO migrate reduced matrix construction (for individual matrices) into galmat class
      let k_minus_one_strongly_enforced_dofs = (0..mass_sigma.nrows())
        .filter(|&i| km1_pred(i))
        .collect::<HashSet<_>>();
      let k_strongly_enforced_dofs = (0..mass_u.nrows())
        .filter(|&i| k_pred(i))
        .collect::<HashSet<_>>();

      assemble::drop_dofs_galmat(&k_minus_one_strongly_enforced_dofs, &mut mass_sigma);
      assemble::drop_dofs_rectangular_galmat(
        &k_strongly_enforced_dofs,
        &k_minus_one_strongly_enforced_dofs,
        &mut dif_sigma,
      );
      assemble::drop_dofs_rectangular_galmat(
        &k_minus_one_strongly_enforced_dofs,
        &k_strongly_enforced_dofs,
        &mut codif_u,
      );
      assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut codifdif_u);
      assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut mass_u);
    }

    let sigma_len = mass_sigma.nrows();

    if sigma_len == 0 {
      let l = CsrMatrix::from(&codifdif_u);
      let mk = CsrMatrix::from(&mass_u);

      // TODO untested, useful only for laplace beltrami problems which are currently handled elsewhere
      let (eigenvals, eigen_us) = petsc_ghiep_largest(&l, &mk, neigen_values);
      let eigen_sigmas = Matrix::zeros(0, eigen_us.ncols());
      return (eigenvals, eigen_sigmas, eigen_us);
    }

    // Solve the generalized saddle point problem for largest eigenvalues, which
    // requires eliminating sigma to avoid infinite eigenvalues.
    let l = CsrMatrix::from(&codifdif_u);
    let d = CsrMatrix::from(&dif_sigma);
    let c = CsrMatrix::from(&codif_u);
    let mkm1 = CsrMatrix::from(&mass_sigma);
    let mk = CsrMatrix::from(&mass_u);

    let (eigenvals, eigen_sigmas, eigen_us) = petsc_ghep_reduced_with_which(
      GhepReducedOperators {
        l: &l,
        d: &d,
        c: &c,
        mkm1: &mkm1,
        mk: &mk,
      },
      neigen_values,
      which,
      mass_solve,
    );

    (eigenvals, eigen_sigmas, eigen_us)
  };

  (evals, evec_sigmas, evec_us)
}

// Private dispatch preserves the separately testable choices of weighting,
// boundary restriction, spectrum end, and reduced mass solve.
#[allow(clippy::too_many_arguments)]
fn solve_hodge_laplace_evp_inner(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  solve_hodge_laplace_evp_inner_with_solver(
    topology,
    geometry,
    grade,
    neigen_values,
    coords,
    qr,
    weight,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    which,
    mass_solve,
  )
}

// Each option changes a distinct operator in the generalized mixed eigenproblem;
// the Hodge EVP studies test both direct and reduced solve routes.
#[allow(clippy::too_many_arguments)]
fn solve_hodge_laplace_evp_inner_with_solver(
  topology: &Complex,
  geometry: &MeshLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
  coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
  k_minus_one_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  k_strong_bc_predicate: Option<&dyn Fn(KSimplexIdx) -> bool>,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  let galmats = if let (Some(coords), Some(weight)) = (coords, weight) {
    MixedGalmats::compute_weighted(topology, geometry, grade, coords, qr, weight)
  } else {
    MixedGalmats::compute(topology, geometry, grade)
  };

  solve_hodge_laplace_evp_with_galmats_impl(
    &galmats,
    neigen_values,
    k_minus_one_strong_bc_predicate,
    k_strong_bc_predicate,
    which,
    mass_solve,
  )
}

pub struct MixedGalmats {
  mass_sigma: GalMat,
  dif_sigma: GalMat,
  codif_u: GalMat,
  codifdif_u: GalMat,
  mass_u: GalMat,
}
impl MixedGalmats {
  pub fn compute(topology: &Complex, geometry: &MeshLengths, grade: ExteriorGrade) -> Self {
    let dim = topology.dim();
    assert!(grade <= dim);

    let mass_u = assemble_galmat(topology, geometry, HodgeMassElmat::new(dim, grade));
    let mass_u_csr = CsrMatrix::from(&mass_u);

    let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
      let mass_sigma = assemble_galmat(topology, geometry, HodgeMassElmat::new(dim, grade - 1));

      let exdif_sigma = topology.exterior_derivative_operator(grade - 1);
      let exdif_sigma = CsrMatrix::from(&exdif_sigma);

      let dif_sigma = &mass_u_csr * &exdif_sigma;
      let dif_sigma = CooMatrix::from(&dif_sigma);

      let codif_u = &exdif_sigma.transpose() * &mass_u_csr;
      let codif_u = CooMatrix::from(&codif_u);

      (mass_sigma, dif_sigma, codif_u)
    } else {
      (GalMat::new(0, 0), GalMat::new(0, 0), GalMat::new(0, 0))
    };

    let codifdif_u = if grade < topology.dim() {
      let mass_plus = assemble_galmat(topology, geometry, HodgeMassElmat::new(dim, grade + 1));
      let mass_plus = CsrMatrix::from(&mass_plus);
      let exdif_u = topology.exterior_derivative_operator(grade);
      let exdif_u = CsrMatrix::from(&exdif_u);
      let codifdif_u = exdif_u.transpose() * mass_plus * exdif_u;
      CooMatrix::from(&codifdif_u)
    } else {
      GalMat::new(0, 0)
    };

    Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      mass_u,
    }
  }

  pub fn compute_weighted(
    topology: &Complex,
    geometry: &MeshLengths,
    grade: ExteriorGrade,
    coords: &MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &InnerProductWeightClosure,
  ) -> Self {
    let dim = topology.dim();
    assert!(grade <= dim);

    let mass_u = assemble_galmat_coord_aware(
      topology,
      geometry,
      HodgeMassElmat::new_weighted(dim, grade, coords, qr.clone(), weight),
    );
    let mass_u_csr = CsrMatrix::from(&mass_u);

    let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
      let mass_sigma = assemble_galmat_coord_aware(
        topology,
        geometry,
        HodgeMassElmat::new_weighted(dim, grade - 1, coords, qr.clone(), weight),
      );

      let exdif_sigma = topology.exterior_derivative_operator(grade - 1);
      let exdif_sigma = CsrMatrix::from(&exdif_sigma);

      let dif_sigma = &mass_u_csr * &exdif_sigma;
      let dif_sigma = CooMatrix::from(&dif_sigma);

      let codif_u = &exdif_sigma.transpose() * &mass_u_csr;
      let codif_u = CooMatrix::from(&codif_u);

      (mass_sigma, dif_sigma, codif_u)
    } else {
      (GalMat::new(0, 0), GalMat::new(0, 0), GalMat::new(0, 0))
    };

    let codifdif_u = if grade < topology.dim() {
      let mass_plus = assemble_galmat_coord_aware(
        topology,
        geometry,
        HodgeMassElmat::new_weighted(dim, grade + 1, coords, qr.clone(), weight),
      );
      let mass_plus = CsrMatrix::from(&mass_plus);
      let exdif_u = topology.exterior_derivative_operator(grade);
      let exdif_u = CsrMatrix::from(&exdif_u);
      let codifdif_u = exdif_u.transpose() * mass_plus * exdif_u;
      CooMatrix::from(&codifdif_u)
    } else {
      GalMat::new(0, 0)
    };

    Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      mass_u,
    }
  }

  pub fn sigma_len(&self) -> usize {
    self.mass_sigma.nrows()
  }
  pub fn u_len(&self) -> usize {
    self.mass_u.nrows()
  }

  pub fn mass_sigma(&self) -> &GalMat {
    &self.mass_sigma
  }
  pub fn dif_sigma(&self) -> &GalMat {
    &self.dif_sigma
  }
  pub fn codif_u(&self) -> &GalMat {
    &self.codif_u
  }
  pub fn codifdif_u(&self) -> &GalMat {
    &self.codifdif_u
  }
  pub fn mass_u(&self) -> &GalMat {
    &self.mass_u
  }

  pub fn free_sigma_len(
    &self,
    k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  ) -> usize {
    let total = self.mass_sigma.nrows();
    let n_strong_bc = (0..total)
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .count();
    total - n_strong_bc
  }

  pub fn free_u_len(&self, k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool) -> usize {
    let total = self.mass_u.nrows();
    let n_strong_bc = (0..total).filter(|&i| k_strong_bc_predicate(i)).count();
    total - n_strong_bc
  }

  pub fn mass_u_csr(&self) -> CsrMatrix {
    CsrMatrix::from(&self.mass_u)
  }

  /// Schur complement of the mixed formulation using a lumped inverse for the sigma mass matrix.
  pub fn hodge_laplacian_schur_complement_lumped(&self) -> CsrMatrix {
    if self.mass_sigma.nrows() == 0 {
      return CsrMatrix::from(&self.codifdif_u);
    }

    let mass_sigma = CsrMatrix::from(&self.mass_sigma);
    let dif_sigma = CsrMatrix::from(&self.dif_sigma);
    let codif_u = CsrMatrix::from(&self.codif_u);
    let codifdif_u = CsrMatrix::from(&self.codifdif_u);

    let mass_sigma_inv = invert_diag(&lumped_diag(&mass_sigma));
    let codif_u_scaled = scale_rows(&codif_u, &mass_sigma_inv);

    let schur = &dif_sigma * &codif_u_scaled;
    add_sparse(&codifdif_u, &schur)
  }

  pub fn mixed_hodge_laplacian(&self) -> CooMatrix {
    let Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      ..
    } = self;
    let codif_u = codif_u.clone();
    CooMatrix::block(&[&[mass_sigma, &(codif_u.neg())], &[dif_sigma, codifdif_u]])
  }

  pub fn reduced_mixed_hodge_laplacian(
    &self,
    k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
  ) -> CooMatrix {
    let Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      ..
    } = self;
    let k_minus_one_strongly_enforced_dofs = (0..mass_sigma.nrows())
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .collect::<HashSet<_>>();
    let k_strongly_enforced_dofs = (0..codifdif_u.nrows())
      .filter(|&i| k_strong_bc_predicate(i))
      .collect::<HashSet<_>>();

    let mut mass_sigma = mass_sigma.clone();
    let mut dif_sigma = dif_sigma.clone();
    let mut codif_u = codif_u.clone();
    let mut codifdif_u = codifdif_u.clone();
    assemble::drop_dofs_galmat(&k_minus_one_strongly_enforced_dofs, &mut mass_sigma);
    // TODO check if rows/cols semantics are correct
    assemble::drop_dofs_rectangular_galmat(
      &k_strongly_enforced_dofs,
      &k_minus_one_strongly_enforced_dofs,
      &mut dif_sigma,
    );
    assemble::drop_dofs_rectangular_galmat(
      &k_minus_one_strongly_enforced_dofs,
      &k_strongly_enforced_dofs,
      &mut codif_u,
    );
    assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut codifdif_u);
    CooMatrix::block(&[&[&mass_sigma, &(codif_u.neg())], &[&dif_sigma, &codifdif_u]])
  }

  // This method assembles the four mixed blocks, two trace maps, two right-hand
  // sides, and harmonic constraint of one KKT system. Strong-BC tests compare the
  // reconstructed solution and boundary values.
  #[allow(clippy::too_many_arguments)]
  pub fn mixed_hodge_laplacian_with_strong_bc_via_elimination(
    &self,
    k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_minus_one_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
    k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_strong_bc_data: &dyn Fn(KSimplexIdx) -> DofCoeff,
    sigma_rhs: &mut Vector,
    u_rhs: &mut Vector,
    harmonics: &Matrix, // Assumed to already be in the reduced basis
  ) -> (CsrMatrix, Vector) {
    let Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      codifdif_u,
      mass_u,
    } = self;

    let mut mass_sigma = mass_sigma.clone();
    let mut dif_sigma = dif_sigma.clone();
    let mut codif_u_neg = codif_u.clone().neg();
    let mut codifdif_u = codifdif_u.clone();
    let mut mass_u = mass_u.clone();

    let k_minus_one_strongly_enforced_data = (0..mass_sigma.nrows())
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .map(|i| (i, k_minus_one_strong_bc_data(i)))
      .collect::<Vec<_>>();

    let k_strongly_enforced_data = (0..mass_u.nrows())
      .filter(|&i| k_strong_bc_predicate(i))
      .map(|i| (i, k_strong_bc_data(i)))
      .collect::<Vec<_>>();

    fix_dofs_coeff_strong_coo(
      &k_minus_one_strongly_enforced_data,
      &mut mass_sigma,
      sigma_rhs,
    );

    fix_dofs_coeff_strong_coo(&k_strongly_enforced_data, &mut codifdif_u, u_rhs);

    fix_dofs_coeff_strong_coo_rectangular(
      &k_minus_one_strongly_enforced_data,
      k_strong_bc_predicate,
      &mut dif_sigma,
      u_rhs,
    );

    fix_dofs_coeff_strong_coo_rectangular(
      &k_strongly_enforced_data,
      k_minus_one_strong_bc_predicate,
      &mut codif_u_neg,
      sigma_rhs,
    );

    let k_strongly_enforced_dofs = (0..mass_u.nrows())
      .filter(|&i| k_strong_bc_predicate(i))
      .collect::<HashSet<_>>();

    let k_minus_one_strongly_enforced_dofs = (0..mass_sigma.nrows())
      .filter(|&i| k_minus_one_strong_bc_predicate(i))
      .collect::<HashSet<_>>();

    assemble::drop_dofs_galmat(&k_minus_one_strongly_enforced_dofs, &mut mass_sigma);
    assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut codifdif_u);
    assemble::drop_dofs_rectangular_galmat(
      &k_strongly_enforced_dofs,
      &k_minus_one_strongly_enforced_dofs,
      &mut dif_sigma,
    );
    assemble::drop_dofs_rectangular_galmat(
      &k_minus_one_strongly_enforced_dofs,
      &k_strongly_enforced_dofs,
      &mut codif_u_neg,
    );

    let k_strongly_enforced_dofs_slice =
      k_strongly_enforced_dofs.iter().cloned().collect::<Vec<_>>();
    let k_minus_one_strongly_enforced_dofs_slice = k_minus_one_strongly_enforced_dofs
      .iter()
      .cloned()
      .collect::<Vec<_>>();

    assemble::drop_dofs_galvec(&k_minus_one_strongly_enforced_dofs_slice, sigma_rhs);

    assemble::drop_dofs_galvec(&k_strongly_enforced_dofs_slice, u_rhs);

    let mut galmat = CooMatrix::block(&[&[&mass_sigma, &codif_u_neg], &[&dif_sigma, &codifdif_u]]);

    let rhs_vec = if harmonics.ncols() > 0 {
      let mut harmonics_rhs = Vector::zeros(mass_u.nrows());

      // Restricts mass_u to free dofs and makes rhs equal to -M_ID * u_D
      fix_dofs_coeff_strong_coo(&k_strongly_enforced_data, &mut mass_u, &mut harmonics_rhs);

      assemble::drop_dofs_galvec(&k_strongly_enforced_dofs_slice, &mut harmonics_rhs);

      // RHS finally equal to -H^T * M_ID * u_D
      harmonics_rhs = &harmonics.transpose() * &harmonics_rhs;

      assemble::drop_dofs_galmat(&k_strongly_enforced_dofs, &mut mass_u);

      let mass_u_csr = CsrMatrix::from(&mass_u);

      let mass_harmonics = &mass_u_csr * harmonics;

      galmat.grow(harmonics.ncols(), harmonics.ncols());

      let reduced_sigma_len = mass_sigma.nrows();
      let reduced_u_len = codifdif_u.nrows();

      for (mut r, mut c) in (0..mass_harmonics.nrows()).cartesian_product(0..mass_harmonics.ncols())
      {
        let v = mass_harmonics[(r, c)];
        r += reduced_sigma_len;
        c += reduced_sigma_len + reduced_u_len;
        galmat.push(r, c, v);
      }
      for (mut r, mut c) in (0..mass_harmonics.nrows()).cartesian_product(0..mass_harmonics.ncols())
      {
        let v = mass_harmonics[(r, c)];
        // transpose
        mem::swap(&mut r, &mut c);
        r += reduced_sigma_len + reduced_u_len;
        c += reduced_sigma_len;
        galmat.push(r, c, v);
      }
      Vector::from_iterator(
        sigma_rhs.len() + u_rhs.len() + harmonics_rhs.len(),
        sigma_rhs
          .iter()
          .chain(u_rhs.iter())
          .chain(harmonics_rhs.iter())
          .copied(),
      )
    } else {
      Vector::from_iterator(
        sigma_rhs.len() + u_rhs.len(),
        sigma_rhs.iter().chain(u_rhs.iter()).copied(),
      )
    };

    let system_matrix = CsrMatrix::from(&galmat);

    (system_matrix, rhs_vec)
  }
}

fn lumped_diag(mat: &CsrMatrix) -> Vec<f64> {
  let mut diag = vec![0.0; mat.nrows()];
  for (row, _col, value) in mat.triplet_iter() {
    diag[row] += *value;
  }
  diag
}

fn invert_diag(diag: &[f64]) -> Vec<f64> {
  let eps = 1e-12;
  diag
    .iter()
    .map(|v| if v.abs() < eps { 0.0 } else { 1.0 / v })
    .collect()
}

fn scale_rows(mat: &CsrMatrix, row_scales: &[f64]) -> CsrMatrix {
  assert_eq!(mat.nrows(), row_scales.len());
  let mut coo = CooMatrix::new(mat.nrows(), mat.ncols());
  for (row, col, value) in mat.triplet_iter() {
    let scaled = *value * row_scales[row];
    if scaled != 0.0 {
      coo.push(row, col, scaled);
    }
  }
  CsrMatrix::from(&coo)
}

fn add_sparse(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
  assert_eq!(a.nrows(), b.nrows());
  assert_eq!(a.ncols(), b.ncols());

  let mut coo = CooMatrix::from(a);
  for (row, col, value) in b.triplet_iter() {
    coo.push(row, col, *value);
  }
  CsrMatrix::from(&coo)
}
// TODO same functionality as assemble::fix_dofs_coeff but different implementation
// Benchmark then decide which one to keep
pub fn fix_dofs_coeff_strong_coo(
  dof_coeffs: &[(DofIdx, f64)],
  galmat: &mut GalMat,
  galvec: &mut Vector,
) {
  let ndofs = galmat.nrows();

  let mut fixed_val: Vec<Option<f64>> = vec![None; ndofs];
  for &(i, v) in dof_coeffs {
    fixed_val[i] = Some(v);
  }

  let mut new_mat = GalMat::new(ndofs, ndofs);

  for (r, c, a_ref) in galmat.triplet_iter() {
    let a = *a_ref;

    if fixed_val[r].is_some() {
      continue;
    }

    if let Some(vc) = fixed_val[c] {
      // Move known contribution to RHS: b_r -= A_{r,c} * v_c
      galvec[r] -= a * vc;
      continue;
    }

    // Free/free coupling survives.
    new_mat.push(r, c, a);
  }

  // Impose Dirichlet rows: set b_i = v_i and set A_{i,i} = 1.
  for &(i, v) in dof_coeffs {
    galvec[i] = v;
    new_mat.push(i, i, 1.0);
  }

  *galmat = new_mat;
}

pub fn fix_dofs_coeff_strong_coo_rectangular(
  col_dof_coeffs: &[(DofIdx, f64)],
  row_predicate: &dyn Fn(DofIdx) -> bool,
  galmat: &mut GalMat,
  galvec: &mut Vector,
) {
  let nrows = galmat.nrows();
  let ncols = galmat.ncols();

  let mut fixed_val: Vec<Option<f64>> = vec![None; ncols];
  for &(i, v) in col_dof_coeffs {
    fixed_val[i] = Some(v);
  }

  let mut excluded_rows: Vec<bool> = vec![false; nrows];
  for (i, excluded) in excluded_rows.iter_mut().enumerate() {
    if row_predicate(i) {
      *excluded = true;
    }
  }

  let mut new_mat = GalMat::new(nrows, ncols);

  for (r, c, a_ref) in galmat.triplet_iter() {
    let a = *a_ref;

    if excluded_rows[r] {
      continue;
    }

    if let Some(vc) = fixed_val[c] {
      // Move known contribution to RHS: b_r -= A_{r,c} * v_c
      galvec[r] -= a * vc;
      continue;
    }

    // Free/free coupling survives.
    new_mat.push(r, c, a);
  }

  // Impose Dirichlet rows: set b_i = v_i and set A_{i,i} = 1.
  // for &(i, v) in col_dof_coeffs {
  //   galvec[i] = v;
  //   new_mat.push(i, i, 1.0);
  // }

  *galmat = new_mat;
}

#[cfg(test)]
mod tests {
  use super::*;
  use manifold::gen::cartesian::CartesianMeshInfo;

  #[cfg(feature = "external-solver-tests")]
  fn assert_vectors_close(lhs: &Vector, rhs: &Vector, tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    let max_diff = lhs
      .iter()
      .zip(rhs.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0, f64::max);
    assert!(
      max_diff <= tol,
      "vector entries differ by up to {max_diff}, tolerance {tol}"
    );
  }

  #[test]
  fn schur_complement_lumped_dimensions() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let galmats = MixedGalmats::compute(&topology, &metric, 1);
    let mass_u = galmats.mass_u_csr();
    let laplacian = galmats.hodge_laplacian_schur_complement_lumped();

    assert!(mass_u.nrows() > 0);
    assert_eq!(mass_u.nrows(), mass_u.ncols());
    assert_eq!(laplacian.nrows(), mass_u.nrows());
    assert_eq!(laplacian.ncols(), mass_u.ncols());
  }

  #[test]
  fn harmonics_with_galmats_zero_dim() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let galmats = MixedGalmats::compute(&topology, &metric, 1);
    let harmonics =
      solve_hodge_laplace_harmonics_with_galmats(&topology, &galmats, 1, 0, None, None);

    assert_eq!(harmonics.ncols(), 0);
    assert_eq!(harmonics.nrows(), galmats.u_len());
  }

  #[test]
  #[cfg(feature = "external-solver-tests")]
  fn source_solver_with_strong_boundary_conditions_accepts_full_rhs_lengths() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let galmats = MixedGalmats::compute(&topology, &metric, 1);
    let boundary_u = topology
      .boundary_subcomplex_simplices(1)
      .into_iter()
      .map(|simp| simp.kidx)
      .collect::<HashSet<_>>();
    let boundary_sigma = topology
      .boundary_subcomplex_simplices(0)
      .into_iter()
      .map(|simp| simp.kidx)
      .collect::<HashSet<_>>();
    let k_strong_bc_predicate = |kidx: KSimplexIdx| boundary_u.contains(&kidx);
    let k_minus_one_strong_bc_predicate = |kidx: KSimplexIdx| boundary_sigma.contains(&kidx);
    let zero_data = |_kidx: KSimplexIdx| 0.0;

    let (sigma, u, p) = solve_hodge_laplace_source_with_galmats_and_boundary_conditions(
      &topology,
      &galmats,
      None,
      Vector::zeros(galmats.u_len()),
      1,
      0,
      &k_strong_bc_predicate,
      &zero_data,
      &k_minus_one_strong_bc_predicate,
      &zero_data,
    );

    assert_eq!(sigma.coeffs().len(), galmats.sigma_len());
    assert_eq!(u.coeffs().len(), galmats.u_len());
    assert_eq!(p.coeffs().len(), 0);
  }

  #[test]
  #[cfg(feature = "external-solver-tests")]
  fn transient_solver_zero_state_stays_zero() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let galmats = MixedGalmats::compute(&topology, &metric, 1);

    let initial_u = Cochain::new(1, Vector::zeros(galmats.u_len()));
    let sigma_len = galmats.sigma_len();
    let u_len = galmats.u_len();
    let sigma_rhs_at = move |_| Vector::zeros(sigma_len);
    let u_rhs_at = move |_| Vector::zeros(u_len);
    let config = MixedTransientConfig {
      times: &[0.0, 0.1, 0.2],
      method: ThetaMethod::BACKWARD_EULER,
      sigma_rhs_at: &sigma_rhs_at,
      u_rhs_at: &u_rhs_at,
      k_strong_bc_predicate: None,
      k_strong_bc_data_at: None,
      k_minus_one_strong_bc_predicate: None,
      k_minus_one_strong_bc_data_at: None,
    };

    let solution =
      solve_hodge_laplace_transient_with_galmats(&topology, &galmats, initial_u, None, 1, config);

    for state in solution {
      assert_vectors_close(state.sigma.coeffs(), &Vector::zeros(sigma_len), 1e-12);
      assert_vectors_close(state.u.coeffs(), &Vector::zeros(u_len), 1e-12);
    }
  }

  #[test]
  #[cfg(feature = "external-solver-tests")]
  fn transient_solver_recovers_missing_sigma_consistently() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let galmats = MixedGalmats::compute(&topology, &metric, 1);

    let initial_u = Cochain::new(1, Vector::from_element(galmats.u_len(), 0.25));
    let sigma_len = galmats.sigma_len();
    let u_len = galmats.u_len();
    let sigma_rhs_at = move |_| Vector::zeros(sigma_len);
    let u_rhs_at = move |_| Vector::zeros(u_len);
    let make_config = || MixedTransientConfig {
      times: &[0.0, 0.1],
      method: ThetaMethod::BACKWARD_EULER,
      sigma_rhs_at: &sigma_rhs_at,
      u_rhs_at: &u_rhs_at,
      k_strong_bc_predicate: None,
      k_strong_bc_data_at: None,
      k_minus_one_strong_bc_predicate: None,
      k_minus_one_strong_bc_data_at: None,
    };
    let consistent_sigma = consistent_sigma_from_u(&galmats, &initial_u, 0.0, &make_config());

    let recovered = solve_hodge_laplace_transient_with_galmats(
      &topology,
      &galmats,
      initial_u.clone(),
      None,
      1,
      make_config(),
    );
    let explicit = solve_hodge_laplace_transient_with_galmats(
      &topology,
      &galmats,
      initial_u,
      Some(consistent_sigma),
      1,
      make_config(),
    );

    assert_eq!(recovered.len(), explicit.len());
    for (lhs, rhs) in recovered.iter().zip(explicit.iter()) {
      assert_vectors_close(lhs.sigma.coeffs(), rhs.sigma.coeffs(), 1e-12);
      assert_vectors_close(lhs.u.coeffs(), rhs.u.coeffs(), 1e-12);
    }
  }

  #[test]
  #[cfg(feature = "external-solver-tests")]
  fn transient_solver_reintroduces_strong_boundary_data() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let galmats = MixedGalmats::compute(&topology, &metric, 1);
    let boundary_u = topology
      .boundary_subcomplex_simplices(1)
      .into_iter()
      .map(|simp| simp.kidx)
      .collect::<HashSet<_>>();
    let boundary_sigma = topology
      .boundary_subcomplex_simplices(0)
      .into_iter()
      .map(|simp| simp.kidx)
      .collect::<HashSet<_>>();

    let sigma_len = galmats.sigma_len();
    let u_len = galmats.u_len();
    let sigma_rhs_at = move |_| Vector::zeros(sigma_len);
    let u_rhs_at = move |_| Vector::zeros(u_len);
    let k_strong_bc_predicate = |kidx: KSimplexIdx| boundary_u.contains(&kidx);
    let k_strong_bc_data_at = |time: f64, kidx: KSimplexIdx| time * (1.0 + kidx as f64);
    let k_minus_one_strong_bc_predicate = |kidx: KSimplexIdx| boundary_sigma.contains(&kidx);
    let k_minus_one_strong_bc_data_at = |time: f64, kidx: KSimplexIdx| -time * (1.0 + kidx as f64);
    let config = MixedTransientConfig {
      times: &[0.0, 0.25],
      method: ThetaMethod::BACKWARD_EULER,
      sigma_rhs_at: &sigma_rhs_at,
      u_rhs_at: &u_rhs_at,
      k_strong_bc_predicate: Some(&k_strong_bc_predicate),
      k_strong_bc_data_at: Some(&k_strong_bc_data_at),
      k_minus_one_strong_bc_predicate: Some(&k_minus_one_strong_bc_predicate),
      k_minus_one_strong_bc_data_at: Some(&k_minus_one_strong_bc_data_at),
    };

    let solution = solve_hodge_laplace_transient_with_galmats(
      &topology,
      &galmats,
      Cochain::new(1, Vector::zeros(u_len)),
      None,
      1,
      config,
    );
    let state = solution.last().unwrap();

    for &kidx in &boundary_u {
      assert!((state.u[kidx] - 0.25 * (1.0 + kidx as f64)).abs() <= 1e-12);
    }
    for &kidx in &boundary_sigma {
      assert!((state.sigma[kidx] + 0.25 * (1.0 + kidx as f64)).abs() <= 1e-12);
    }
  }
}
