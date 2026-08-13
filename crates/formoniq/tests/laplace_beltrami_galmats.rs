use common::linalg::nalgebra::Matrix;
use formoniq::{
  assemble,
  operators::{InnerProductWeightClosure, LaplaceBeltramiElmat, ScalarMassElmat},
  problems::laplace_beltrami::LaplaceBeltramiGalmats,
};
use manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::quadrature::SimplexQuadRule};

#[test]
fn laplace_beltrami_galmats_match_direct_assembly() {
  let dim = 2;
  let nboxes = 1;
  let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes, 1.0);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  let mats = LaplaceBeltramiGalmats::compute(&topology, &metric);
  let stiffness_ref = assemble::assemble_galmat(&topology, &metric, LaplaceBeltramiElmat::new(dim));
  let mass_ref = assemble::assemble_galmat(&topology, &metric, ScalarMassElmat::new());

  assert_matrix_close(
    &Matrix::from(mats.stiffness()),
    &Matrix::from(&stiffness_ref),
    1e-12,
  );
  assert_matrix_close(&Matrix::from(mats.mass()), &Matrix::from(&mass_ref), 1e-12);

  let weight = InnerProductWeightClosure::new(|_p| 2.0);
  let qr: Option<SimplexQuadRule> = None;
  let mats_weighted =
    LaplaceBeltramiGalmats::compute_weighted(&topology, &metric, &coords, qr.clone(), &weight);
  let stiffness_weighted_ref = assemble::assemble_galmat_coord_aware(
    &topology,
    &metric,
    LaplaceBeltramiElmat::new_weighted(dim, &coords, qr.clone(), &weight),
  );
  let mass_weighted_ref = assemble::assemble_galmat_coord_aware(
    &topology,
    &metric,
    ScalarMassElmat::new_weighted(&coords, qr, &weight),
  );

  assert_matrix_close(
    &Matrix::from(mats_weighted.stiffness()),
    &Matrix::from(&stiffness_weighted_ref),
    1e-12,
  );
  assert_matrix_close(
    &Matrix::from(mats_weighted.mass()),
    &Matrix::from(&mass_weighted_ref),
    1e-12,
  );
}

fn assert_matrix_close(lhs: &Matrix, rhs: &Matrix, tol: f64) {
  assert_eq!(lhs.shape(), rhs.shape());
  let diff = lhs - rhs;
  assert!(
    diff.iter().all(|v| v.abs() <= tol),
    "matrix entries differ by more than {tol}"
  );
}
