use common::linalg::nalgebra::Vector;
use ddf::cochain::cochain_projection;
use exterior::field::{DiffFormClosure, EmbeddedDiffFormClosure};
use formoniq::{
  assemble::{assemble_boundary_integral_term, assemble_galvec},
  fe::fe_l2_error,
  operators::SourceElVec,
};
use manifold::geometry::coord::mesh::standard_coord_complex;

fn embedded_triangle() -> (
  manifold::topology::complex::Complex,
  manifold::geometry::coord::mesh::MeshCoords,
) {
  let (topology, coords_2d) = standard_coord_complex(2);
  let coords = coords_2d.embed_euclidean(3);
  (topology, coords)
}

#[test]
fn embedded_source_elvec_matches_flat_mesh() {
  let (topology, coords_2d) = standard_coord_complex(2);
  let metric_2d = coords_2d.to_edge_lengths(&topology);
  let coords_3d = coords_2d.clone().embed_euclidean(3);
  let metric_3d = coords_3d.to_edge_lengths(&topology);

  let intrinsic = DiffFormClosure::one_form(|_| Vector::from_vec(vec![1.0, -0.5]), 2);
  let ambient =
    EmbeddedDiffFormClosure::ambient_one_form(|_| Vector::from_vec(vec![1.0, -0.5, 0.0]), 3, 2);

  let flat = assemble_galvec(
    &topology,
    &metric_2d,
    SourceElVec::new(&intrinsic, &coords_2d, None),
  );
  let embedded = assemble_galvec(
    &topology,
    &metric_3d,
    SourceElVec::new(&ambient, &coords_3d, None),
  );

  assert!((flat - embedded).norm() < 1e-12);
}

#[test]
fn embedded_projection_has_zero_l2_error() {
  let (topology, coords) = embedded_triangle();
  let exact =
    EmbeddedDiffFormClosure::ambient_one_form(|_| Vector::from_vec(vec![1.0, -0.5, 0.0]), 3, 2);

  let projected = cochain_projection(&exact, &topology, &coords, None);
  let error = fe_l2_error(&projected, &exact, &topology, &coords);

  assert!(error < 1e-12, "expected near-zero error, got {error}");
}

#[test]
fn embedded_boundary_integral_matches_flat_mesh() {
  let (topology, coords_2d) = standard_coord_complex(2);
  let coords_3d = coords_2d.clone().embed_euclidean(3);
  let intrinsic = DiffFormClosure::one_form(|_| Vector::from_vec(vec![1.0, -0.5]), 2);
  let ambient =
    EmbeddedDiffFormClosure::ambient_one_form(|_| Vector::from_vec(vec![1.0, -0.5, 0.0]), 3, 2);
  let selector = |_kidx| true;

  let flat = assemble_boundary_integral_term(&topology, &coords_2d, 0, &intrinsic, None, &selector);
  let embedded =
    assemble_boundary_integral_term(&topology, &coords_3d, 0, &ambient, None, &selector);

  assert!((flat - embedded).norm() < 1e-12);
}
