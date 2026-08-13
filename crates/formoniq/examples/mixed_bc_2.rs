use common::linalg::nalgebra::Vector;
use ddf::cochain::cochain_projection;
use exterior::field::DiffFormClosure;
use formoniq::{
  fe::fe_l2_error, io::write_cochain_vtk, operators::InnerProductWeightClosure,
  problems::laplace_beltrami,
};
use manifold::gen::cartesian::CartesianMeshInfo;
use manifold::geometry::coord::CoordRef;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::fs;

fn main() {
  tracing_subscriber::fmt::init();
  let path = "out/examples/mixed_bc_2";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;
  let resolution = 100;

  let exact_solution = DiffFormClosure::scalar(
    |p| 1. + p[0] + p[1] + p[0] * p[1] + (PI * p[0]).sin() * (PI * p[1]).sin(),
    dim,
  );

  let rhs = DiffFormClosure::scalar(
    |p| 2. * PI * PI * (PI * p[1]).sin() * (PI * p[0]).sin(),
    dim,
  );

  let inner_product_weight = InnerProductWeightClosure::new(|_p| 1.0);

  let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, resolution, 1.);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  let rhs_vec = formoniq::assemble::assemble_galvec(
    &topology,
    &metric,
    formoniq::operators::SourceElVec::new_weighted(&rhs, &coords, None, &inner_product_weight),
  );

  // TODO for efficiency, implement projection only onto dirichlet boundary
  let solution_projected = cochain_projection(&exact_solution, &topology, &coords, None);

  // let strong_dof_predicate = |p: CoordRef| p[1] != 1.0 || (p[0] == 0.0 || p[0] == 1.0);
  let strong_dof_predicate = |p: CoordRef| p[1] == 1.0;
  let weak_dof_predicate = |p: CoordRef| !strong_dof_predicate(p);

  let dirichlet_dofs = formoniq::assemble::boundary_simplices_where_barycenter(
    &topology,
    &coords,
    0,
    strong_dof_predicate,
  )
  .into_iter()
  .collect::<HashSet<usize>>();

  let dirichlet_dof_selector = |sidx: usize| dirichlet_dofs.contains(&sidx);
  let dirichlet_boundary_data = |vidx: usize| solution_projected[vidx];

  // Flux is through edges
  let neumann_dofs = formoniq::assemble::boundary_simplices_where_barycenter(
    &topology,
    &coords,
    1,
    weak_dof_predicate,
  );

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
  let neumann_dof_selector =
    |kidx: manifold::topology::handle::KSimplexIdx| neumann_dofs.contains(&kidx);

  let neumann_rhs = formoniq::assemble::assemble_boundary_galvec(
    &topology,
    &metric,
    formoniq::operators::SourceElVec::new(&neumann_data, &coords, None),
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

  // let neumann_rhs_formatted: Vec<String> =
  //   neumann_rhs.iter().map(|x| format!("{:.3e}", x)).collect();
  // let neumann_rhs_2_formatted: Vec<String> =
  //   neumann_rhs_2.iter().map(|x| format!("{:.3e}", x)).collect();
  // println!("neumann_rhs: {:?}", neumann_rhs_formatted);
  // println!("neumann_rhs_2: {:?}", neumann_rhs_2_formatted);

  assert!(
    (neumann_rhs.clone() - neumann_rhs_2.clone())
      .iter()
      .all(|x| x.abs() < 1e-10),
    "neumann_rhs and neumann_rhs_2 differ by more than floating point error"
  );

  let total_rhs = rhs_vec + neumann_rhs;

  let galsol = laplace_beltrami::solve_laplace_beltrami_source(
    &topology,
    &metric,
    total_rhs,
    dirichlet_boundary_data,
    Some(&dirichlet_dof_selector),
  );

  let error_l2 = fe_l2_error(&galsol, &exact_solution, &topology, &coords);
  println!("L2 error: {error_l2:.6e}");

  let solution_difference = galsol.clone() - solution_projected.clone();

  let vtk_solution_path = format!("{path}/solution.vtk");
  write_cochain_vtk(&vtk_solution_path, &coords, &topology, &galsol, "solution")
    .expect("failed to write VTK output");
  let vtk_exact_path = format!("{path}/solution_exact.vtk");
  write_cochain_vtk(
    &vtk_exact_path,
    &coords,
    &topology,
    &solution_projected,
    "solution_exact",
  )
  .expect("failed to write VTK output");
  let vtk_difference_path = format!("{path}/solution_difference.vtk");
  write_cochain_vtk(
    &vtk_difference_path,
    &coords,
    &topology,
    &solution_difference,
    "solution_difference",
  )
  .expect("failed to write VTK output for difference");

  println!("Wrote solution and gradient VTK outputs to {path}");
}
