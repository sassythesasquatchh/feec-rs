use std::f64::consts::PI;

use common::linalg::nalgebra::Vector;
use exterior::field::DiffFormClosure;
use formoniq::{
  assemble::{assemble_boundary_galvec, boundary_simplices_where_barycenter},
  operators::SourceElVec,
};
use manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef};

#[test]
fn boundary_vector_equivalence() {
  let resolution = 4;
  let box_mesh = CartesianMeshInfo::new_unit_scaled(2, resolution, 1.);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let _neumann_dofs =
    boundary_simplices_where_barycenter(&topology, &coords, 1, |p: CoordRef| p[1] == 1.0);

  let neumann_data = DiffFormClosure::scalar(
    |p| {
      if p[0] == 0.0 {
        -(1. + p[1] + PI * (PI * p[1]).sin())
      } else if p[0] == 1.0 {
        1. + p[1] - PI * (PI * p[1]).sin()
      } else if p[1] == 0.0 {
        -(1. + p[0] + PI * (PI * p[0]).sin())
      } else if p[1] == 1.0 {
        1. + p[0] - PI * (PI * p[0]).sin()
      } else {
        0.0
      }
    },
    1,
  );

  // let neumann_dof_selector = |kidx| neumann_dofs.contains(&kidx);
  let neumann_dof_selector = |_kidx| true;

  let neumann_rhs = assemble_boundary_galvec(
    &topology,
    &metric,
    SourceElVec::new(&neumann_data, &coords, None),
    neumann_dof_selector,
  );

  let neumann_one_form_data = DiffFormClosure::one_form(
    |p| {
      Vector::from_column_slice(&[
        -(1. + p[0] + PI * (PI * p[0]).sin() * (PI * p[1]).cos()),
        (1. + p[1] + PI * (PI * p[0]).cos() * (PI * p[1]).sin()),
      ])
    },
    2,
  );

  let neumann_rhs_2 = formoniq::assemble::assemble_boundary_integral_term(
    &topology,
    &coords,
    0,
    &neumann_one_form_data,
    None,
    &neumann_dof_selector,
  );

  assert!(
    (neumann_rhs.clone() - neumann_rhs_2.clone())
      .iter()
      .all(|x| x.abs() < 1e-10),
    "neumann_rhs and neumann_rhs_2 differ by more than floating point error"
  );
}
