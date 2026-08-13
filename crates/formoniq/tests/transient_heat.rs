use common::linalg::nalgebra::Vector;
use ddf::cochain::Cochain;
use formoniq::problems::{
  heat::{solve_heat, solve_heat_theta},
  transient::ThetaMethod,
};
use manifold::gen::cartesian::CartesianMeshInfo;

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
fn backward_euler_wrapper_matches_theta_solver() {
  let mesh = CartesianMeshInfo::new_unit_scaled(1, 4, 1.0);
  let (topology, coords) = mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  let dt = 0.1;
  let nsteps = 3;
  let times: Vec<_> = (0..=nsteps).map(|i| i as f64 * dt).collect();
  let diffusion_coeff = 0.7;
  let initial = Cochain::zero(&topology.vertices());
  let source = Cochain::constant(1.0, &topology.vertices());

  let wrapped = solve_heat(
    &topology,
    &metric,
    nsteps,
    dt,
    |_| 0.0,
    initial.clone(),
    source.clone(),
    diffusion_coeff,
  );
  let theta = solve_heat_theta(
    &topology,
    &metric,
    &times,
    ThetaMethod::BACKWARD_EULER,
    |_, _| 0.0,
    initial,
    |_| source.clone(),
    diffusion_coeff,
  );

  assert_eq!(wrapped.len(), theta.len());
  for (lhs, rhs) in wrapped.iter().zip(theta.iter()) {
    assert_vectors_close(lhs.coeffs(), rhs.coeffs(), 1e-12);
  }
}

#[test]
fn theta_heat_tracks_linear_exact_solution() {
  let mesh = CartesianMeshInfo::new_unit_scaled(1, 8, 1.0);
  let (topology, coords) = mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let x = coords.matrix().row(0).transpose().into_owned();
  let times = [0.0, 0.1, 0.2, 0.3];
  let diffusion_coeff = 1.7;

  let initial = Cochain::zero(&topology.vertices());
  let boundary_data = |time: f64, idof: usize| time * x[idof];
  let source_data = |_: f64| Cochain::new(0, x.clone());

  let backward_euler = solve_heat_theta(
    &topology,
    &metric,
    &times,
    ThetaMethod::BACKWARD_EULER,
    boundary_data,
    initial.clone(),
    source_data,
    diffusion_coeff,
  );
  let crank_nicolson = solve_heat_theta(
    &topology,
    &metric,
    &times,
    ThetaMethod::CRANK_NICOLSON,
    boundary_data,
    initial,
    source_data,
    diffusion_coeff,
  );

  for ((&time, be), cn) in times
    .iter()
    .zip(backward_euler.iter())
    .zip(crank_nicolson.iter())
  {
    let expected = time * &x;
    assert_vectors_close(be.coeffs(), &expected, 1e-10);
    assert_vectors_close(cn.coeffs(), &expected, 1e-10);
  }
}
