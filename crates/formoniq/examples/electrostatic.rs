use common::linalg::nalgebra::Vector;
use ddf::cochain::{cochain_projection, Cochain};
use exterior::field::DiffFormClosure;
use formoniq::{
  io::write_cochain_vtk, operators::InnerProductWeightClosure, problems::laplace_beltrami,
};
use manifold::gen::cartesian::CartesianMeshInfo;
use manifold::io::{save_coords_to_file, save_skeleton_to_file};
use std::{f64::consts::PI, fs, io::Write};

use formoniq::fe::fe_l2_error;

fn write_cochain(path: &str, cochain: &Cochain) -> std::io::Result<()> {
  let mut file = fs::File::create(path)?;
  for coeff in cochain.coeffs.iter() {
    writeln!(file, "{coeff:.12}")?;
  }
  Ok(())
}

fn main() {
  tracing_subscriber::fmt::init();
  let path = "out/electrodynamics/electrostatic_poisson";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let permittivity: f64 = 8.854e-12;
  // let permittivity: f64 = 0.01;

  let dim = 3;

  println!("Solving Electrostatic Poisson in {dim}d.");

  let solution_exact =
    DiffFormClosure::scalar(|p| p.iter().map(|&pi| (PI * pi).sin()).product(), dim);
  let grad_solution_exact = DiffFormClosure::one_form(
    |p| {
      Vector::from_iterator(
        p.len(),
        (0..p.len()).map(|i| {
          let sin_prod = p
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &pi)| (PI * pi).sin())
            .product::<f64>();
          PI * (PI * p[i]).cos() * sin_prod
        }),
      )
    },
    dim,
  );

  // let rhs = DiffFormClosure::scalar(
  //   move |p| 3. * PI * PI * permittivity * p.iter().map(|&pi| (PI * pi).sin()).product::<f64>(),
  //   dim,
  // );

  let rhs = DiffFormClosure::scalar(
    move |p| 3. * PI * PI * p.iter().map(|&pi| (PI * pi).sin()).product::<f64>(),
    dim,
  );

  let inner_product_weight = InnerProductWeightClosure::new(move |_p| permittivity);
  let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, 10, 1.);

  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  let load_vector = formoniq::assemble::assemble_galvec(
    &topology,
    &metric,
    formoniq::operators::SourceElVec::new_weighted(&rhs, &coords, None, &inner_product_weight),
  );

  let solution_projected = cochain_projection(&solution_exact, &topology, &coords, None);
  let field_exact = cochain_projection(&grad_solution_exact, &topology, &coords, None).scaled(-1.0);

  let boundary_data = |ivertex| solution_projected[ivertex];

  let galsol = laplace_beltrami::solve_laplace_beltrami_source_weighted(
    &topology,
    &metric,
    load_vector,
    boundary_data,
    None,
    &coords,
    None,
    &inner_product_weight,
  );

  let error_l2 = fe_l2_error(&galsol, &solution_exact, &topology, &coords);
  println!("L2 error: {error_l2:.6e}");

  // Electric field E = -grad(phi) approximated by the exterior derivative of the potential.
  let electric_field = galsol.dif(&topology).scaled(-1.0);

  // Mesh + data exports for visualization.
  let vtk_path = format!("{path}/solution.vtk");
  write_cochain_vtk(&vtk_path, &coords, &topology, &galsol, "potential")
    .expect("failed to write VTK output");
  let vtk_exact_path = format!("{path}/solution_exact.vtk");
  write_cochain_vtk(
    &vtk_exact_path,
    &coords,
    &topology,
    &solution_projected,
    "potential_exact",
  )
  .expect("failed to write VTK output");
  let vtk_difference_path = format!("{path}/solution_difference.vtk");
  let difference = galsol.clone() - solution_projected.clone();
  write_cochain_vtk(
    &vtk_difference_path,
    &coords,
    &topology,
    &difference,
    "potential_difference",
  )
  .expect("failed to write VTK output for difference");
  let field_vtk_path = format!("{path}/electric_field.vtk");
  write_cochain_vtk(
    &field_vtk_path,
    &coords,
    &topology,
    &electric_field,
    "electric_field",
  )
  .expect("failed to write VTK output for electric field");
  let field_exact_vtk_path = format!("{path}/electric_field_exact.vtk");
  write_cochain_vtk(
    &field_exact_vtk_path,
    &coords,
    &topology,
    &field_exact,
    "electric_field_exact",
  )
  .expect("failed to write VTK output for exact electric field");

  // Also export the raw mesh and cochains for external visualization tools.
  save_coords_to_file(&coords, format!("{path}/vertices.coords")).unwrap();
  save_skeleton_to_file(&topology, dim, format!("{path}/cells.skel")).unwrap();
  save_skeleton_to_file(&topology, 1, format!("{path}/edges.skel")).unwrap();
  write_cochain(&format!("{path}/electric_field.cochain"), &electric_field).unwrap();
  write_cochain(
    &format!("{path}/electric_field_exact.cochain"),
    &field_exact,
  )
  .unwrap();
  write_cochain(&format!("{path}/potential.cochain"), &galsol).unwrap();
  write_cochain(
    &format!("{path}/potential_exact.cochain"),
    &solution_projected,
  )
  .unwrap();

  println!("Wrote solution for visualization to {vtk_path} and edge field to {field_vtk_path}");
  println!("Also exported exact solutions and raw mesh/cochains in {path}");
}
