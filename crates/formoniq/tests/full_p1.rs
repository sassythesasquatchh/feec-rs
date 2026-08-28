use approx::assert_relative_eq;
use common::linalg::nalgebra::Matrix;
use ddf::ManifoldComplexExt;
use formoniq::{
  assemble::assemble_galmat,
  full_p1::{assemble_full_p1_one_form_metric_3d, FullP1ReferenceComplex3d},
  operators::HodgeMassElmat,
};
use manifold::{geometry::coord::mesh::standard_coord_complex, topology::complex::Complex};

const TOLERANCE: f64 = 1e-12;

#[test]
fn reference_complex_has_exact_dimensions_ranks_and_nilpotency() {
  let reference = FullP1ReferenceComplex3d::new();
  let d0 = reference.p2_to_full_p1();
  let d1 = reference.full_p1_to_whitney_two();
  let d2 = reference.whitney_two_to_constant_three();

  assert_eq!(d0.shape(), (12, 10));
  assert_eq!(d1.shape(), (4, 12));
  assert_eq!(d2.shape(), (1, 4));
  assert_eq!(d0.rank(TOLERANCE), 9);
  assert_eq!(d1.rank(TOLERANCE), 3);
  assert_eq!(d2.rank(TOLERANCE), 1);
  assert_eq!(d0.ncols() - d0.rank(TOLERANCE), 1);
  assert_eq!(d1.ncols() - d1.rank(TOLERANCE), d0.rank(TOLERANCE));
  assert_eq!(d2.ncols() - d2.rank(TOLERANCE), d1.rank(TOLERANCE));

  assert_eq!(d1 * d0, Matrix::zeros(4, 10));
  assert_eq!(d2 * d1, Matrix::zeros(1, 12));
}

#[test]
fn whitney_inclusion_commutes_with_the_outgoing_derivative() {
  let reference = FullP1ReferenceComplex3d::new();
  let topology = Complex::standard(3);
  let whitney_derivative = Matrix::from(&topology.exterior_derivative_operator(1));

  assert_eq!(
    reference.full_p1_to_whitney_two() * reference.whitney_one_embedding(),
    whitney_derivative
  );
  assert_eq!(
    reference.full_p1_to_whitney_one_retraction() * reference.whitney_one_embedding(),
    Matrix::identity(6, 6)
  );
}

#[test]
fn exact_metric_facade_has_two_dofs_per_edge_and_restricts_to_whitney_mass() {
  let (topology, coords) = standard_coord_complex(3);
  let geometry = coords.to_edge_lengths(&topology);
  let full = assemble_full_p1_one_form_metric_3d(&topology, &geometry).unwrap();

  assert_eq!(full.dimension(), 2 * topology.edges().len());
  assert_eq!(full.dimension(), 12);

  let mass = Matrix::from(full.exact_mass());
  assert_relative_eq!(&mass, &mass.transpose(), epsilon = TOLERANCE);
  assert!(mass.clone().cholesky().is_some());

  let embedding = Matrix::from(full.whitney_embedding());
  let retraction = Matrix::from(full.whitney_retraction());
  assert_eq!(&retraction * &embedding, Matrix::identity(6, 6));

  let whitney_mass = Matrix::from(&assemble_galmat(
    &topology,
    &geometry,
    HodgeMassElmat::new(3, 1),
  ));
  assert_relative_eq!(
    &(embedding.transpose() * mass * embedding),
    &whitney_mass,
    epsilon = TOLERANCE,
    max_relative = TOLERANCE
  );
}

#[test]
fn exact_metric_facade_rejects_unscoped_dimensions() {
  let (topology, coords) = standard_coord_complex(2);
  let geometry = coords.to_edge_lengths(&topology);
  let error = assemble_full_p1_one_form_metric_3d(&topology, &geometry).unwrap_err();
  assert_eq!(
    error.to_string(),
    "full P1 one-form assembly requires a tetrahedral 3D complex, got dimension 2"
  );
}
