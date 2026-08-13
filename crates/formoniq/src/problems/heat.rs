//! Module for the Heat Equation, the prototypical parabolic PDE.

use common::linalg::{faer::FaerCholesky, nalgebra::CsrMatrix};

use crate::{
  assemble,
  operators::{self, DofIdx},
  problems::transient::{validate_time_grid, ThetaMethod},
};

use {
  ddf::cochain::Cochain,
  manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex},
};

/// Convenience wrapper for constant-data Backward Euler.
#[allow(clippy::too_many_arguments)]
pub fn solve_heat<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  nsteps: usize,
  dt: f64,
  boundary_data: F,
  initial_data: Cochain,
  source_data: Cochain,
  diffusion_coeff: f64,
) -> Vec<Cochain>
where
  F: Fn(DofIdx) -> f64,
{
  let times: Vec<_> = (0..=nsteps).map(|istep| istep as f64 * dt).collect();
  solve_heat_theta(
    topology,
    geometry,
    &times,
    ThetaMethod::BACKWARD_EULER,
    |_, idof| boundary_data(idof),
    initial_data,
    |_| source_data.clone(),
    diffusion_coeff,
  )
}

#[allow(clippy::too_many_arguments)]
pub fn solve_heat_theta<FBoundary, FSource>(
  topology: &Complex,
  geometry: &MeshLengths,
  times: &[f64],
  method: ThetaMethod,
  boundary_data_at: FBoundary,
  initial_data: Cochain,
  source_data_at: FSource,
  diffusion_coeff: f64,
) -> Vec<Cochain>
where
  FBoundary: Fn(f64, DofIdx) -> f64,
  FSource: Fn(f64) -> Cochain,
{
  validate_time_grid(times);

  assert_eq!(
    initial_data.dim(),
    0,
    "heat solver expects a 0-cochain initial condition, got grade {}.",
    initial_data.dim()
  );
  assert_eq!(
    initial_data.len(),
    topology.vertices().len(),
    "heat solver initial condition must have {} coefficients, got {}.",
    topology.vertices().len(),
    initial_data.len()
  );

  let dim = topology.dim();
  let laplace = CsrMatrix::from(&assemble::assemble_galmat(
    topology,
    geometry,
    operators::LaplaceBeltramiElmat::new(dim),
  ));
  let mass = CsrMatrix::from(&assemble::assemble_galmat(
    topology,
    geometry,
    operators::ScalarMassElmat::new(),
  ));

  let theta = method.theta();
  let mut solution = Vec::with_capacity(times.len());
  solution.push(initial_data);

  for t01 in times.windows(2) {
    let [t0, t1] = t01 else { unreachable!() };
    let dt = t1 - t0;

    let source_0 = source_vector_at(topology, &source_data_at, &mass, *t0);
    let source_1 = source_vector_at(topology, &source_data_at, &mass, *t1);
    let source_theta = (1.0 - theta) * source_0 + theta * source_1;

    let prev = solution.last().unwrap().coeffs();
    let lhs_csr = assemble_heat_system_matrix(&mass, &laplace, diffusion_coeff, theta, dt);
    let mut lhs = common::linalg::nalgebra::CooMatrix::from(&lhs_csr);
    let mut rhs = (1.0 / dt) * (&mass * prev) - (1.0 - theta) * diffusion_coeff * (&laplace * prev)
      + source_theta;

    assemble::enforce_dirichlet_bc(
      topology,
      |idof| boundary_data_at(*t1, idof),
      &mut lhs,
      &mut rhs,
    );

    let next = FaerCholesky::new(CsrMatrix::from(&lhs)).solve(&rhs);
    solution.push(Cochain::new(0, next));
  }

  solution
}

fn source_vector_at<FSource>(
  topology: &Complex,
  source_data_at: &FSource,
  mass: &CsrMatrix,
  time: f64,
) -> common::linalg::nalgebra::Vector
where
  FSource: Fn(f64) -> Cochain,
{
  let source = source_data_at(time);
  assert_eq!(
    source.dim(),
    0,
    "heat solver source must be a 0-cochain, got grade {} at time {time}.",
    source.dim()
  );
  assert_eq!(
    source.len(),
    topology.vertices().len(),
    "heat solver source must have {} coefficients, got {} at time {time}.",
    topology.vertices().len(),
    source.len()
  );
  mass * source.coeffs()
}

fn assemble_heat_system_matrix(
  mass: &CsrMatrix,
  laplace: &CsrMatrix,
  diffusion_coeff: f64,
  theta: f64,
  dt: f64,
) -> CsrMatrix {
  (1.0 / dt) * mass + theta * diffusion_coeff * laplace
}
