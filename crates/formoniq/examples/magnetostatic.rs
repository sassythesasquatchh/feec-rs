use formoniq::{
  io::{write_1form_vector_field_vtk, write_2form_vector_field_vtk, write_cochain_vtk},
  operators::InnerProductWeightClosure,
};
use manifold::topology::handle::KSimplexIdx;

use {
  common::linalg::nalgebra::Vector,
  exterior::field::DiffFormClosure,
  formoniq::{assemble::assemble_galvec, operators::SourceElVec, problems::hodge_laplace},
  manifold::geometry::coord::CoordRef,
};

use std::{collections::HashSet, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();
  let path = "out/examples/magnetostatic_toroidal_inductor";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let _target_coil_cell_size = 0.06;
  let target_air_cell_size = 0.25;

  let mesh_path = "meshes/toroidal_inductor.msh";
  let path_bytes = fs::read(mesh_path)?;
  let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&path_bytes);
  let metric = coords.to_edge_lengths(&topology);

  let mu_0 = 4e-7 * std::f64::consts::PI;
  let mu_0_inverse = 1.0 / mu_0;

  let inverse_permeability = InnerProductWeightClosure::new(move |_| mu_0_inverse);

  // ---------------- Boundary Handling ----------------

  let major_radius = 2.;
  let core_minor_radius = 0.60;
  let coil_minor_radius = 0.85;
  let box_half_length = 6.0;

  // Outer Boundary (far field approximation)
  let strong_dof_predicate = |p: CoordRef| {
    let d = (box_half_length - p[0].abs())
      .min(box_half_length - p[1].abs())
      .min(box_half_length - p[2].abs());

    d < 5. * 2.0 * target_air_cell_size
  };

  let strong_k_dofs = formoniq::assemble::boundary_simplices_where_barycenter(
    &topology,
    &coords,
    1,
    strong_dof_predicate,
  )
  .into_iter()
  .collect::<HashSet<usize>>();

  let strong_k_minus_one_dofs = formoniq::assemble::boundary_simplices_where_barycenter(
    &topology,
    &coords,
    0,
    strong_dof_predicate,
  )
  .into_iter()
  .collect::<HashSet<usize>>();

  let strong_k_dof_predicate = |sidx: KSimplexIdx| strong_k_dofs.contains(&sidx);
  let strong_k_minus_one_dof_predicate =
    |sidx: KSimplexIdx| strong_k_minus_one_dofs.contains(&sidx);

  let strong_k_data = |_| 0.0;
  let strong_k_minus_one_data = |_| 0.0;

  //   let weak_dof_predicate = |p: CoordRef| {
  //     let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
  //     let s = ((rho - major_radius).powi(2) + p[2] * p[2]).sqrt();

  //     (s - core_minor_radius).abs() < 2.0 * target_coil_cell_size
  //   };

  //---------------- Source Definition ----------------

  let j0: f64 = 1.0; // amplitude (scale as you like)
  let sigma: f64 = 0.18; // Gaussian width in s
  let eps: f64 = 0.03; // smooth cutoff thickness in s

  let current_density = DiffFormClosure::one_form(
    move |p| {
      let x = p[0];
      let y = p[1];
      let z = p[2];

      let rho = (x * x + y * y).sqrt();
      if rho < 1e-12 {
        return Vector::from_column_slice(&[0.0, 0.0, 0.0]);
      }

      let s = ((rho - major_radius).powi(2) + z * z).sqrt();

      // Smoothstep function: C^1, value 0..1
      let smoothstep = |t: f64| t * t * (3.0 - 2.0 * t);

      // Cutoff to (approximately) r_c + eps <= s <= r_w - eps
      let inner = core_minor_radius + eps;
      let outer = coil_minor_radius - eps;

      let tin = ((s - inner) / eps).clamp(0.0, 1.0); // ramps up
      let tout = ((outer - s) / eps).clamp(0.0, 1.0); // ramps down
      let cutoff = smoothstep(tin) * smoothstep(tout);

      // Profile g(s): centered in the pack (optional; adjust as desired)
      let s0 = 0.5 * (core_minor_radius + coil_minor_radius);
      let gauss = (-((s - s0) * (s - s0)) / (sigma * sigma)).exp();

      let amp = mu_0 * j0 * gauss * cutoff;
      // let amp = j0 * gauss * cutoff;

      // e_phi = (-y/rho, x/rho, 0)
      Vector::from_column_slice(&[amp * (-y / rho), amp * (x / rho), 0.0])
    },
    3,
  );

  let source_galvec = assemble_galvec(
    &topology,
    &metric,
    // SourceElVec::new(&current_density, &coords, None),
    SourceElVec::new_weighted(&current_density, &coords, None, &inverse_permeability),
  );

  // --------------- Homology Dimension ----------------
  let strong_k_plus_one_dofs = formoniq::assemble::boundary_simplices_where_barycenter(
    &topology,
    &coords,
    2,
    strong_dof_predicate,
  )
  .into_iter()
  .collect::<HashSet<usize>>();

  let _strong_k_plus_one_dof_predicate = |sidx: KSimplexIdx| strong_k_plus_one_dofs.contains(&sidx);

  let homology_dim = 1;

  //---------------- Solve ----------------

  println!("Preparing to solve...");
  // TODO extract and visualise the harmonic basis
  let (_, u_galsol, _) =
    hodge_laplace::solve_weighted_hodge_laplace_source_with_boundary_conditions(
      &topology,
      &metric,
      None, // Homogenous Neumann BC for potential (gauge)
      source_galvec,
      1,
      homology_dim,
      &coords,
      None,
      &inverse_permeability,
      &strong_k_dof_predicate,
      &strong_k_data, // Homogenous Dirichlet BC
      &strong_k_minus_one_dof_predicate,
      &strong_k_minus_one_data, // Homogenous Dirichlet BC
    );

  //---------------- Output ----------------

  write_1form_vector_field_vtk(
    format!("{path}/magnetic_vector_potential.vtk"),
    &coords,
    &topology,
    &u_galsol,
    "A",
  )?;

  write_cochain_vtk(
    format!("{path}/magnetic_vector_potential_cochain.vtk"),
    &coords,
    &topology,
    &u_galsol,
    "magnetic_vector_potential",
  )?;

  let b_cochain = u_galsol.dif(&topology); // B = dA (2-form)

  write_2form_vector_field_vtk(
    format!("{path}/B_vector.vtk"),
    &coords,
    &topology,
    &b_cochain,
    "B_vector",
  )?;

  write_cochain_vtk(
    format!("{path}/magnetic_flux_density_cochain.vtk"),
    &coords,
    &topology,
    &b_cochain,
    "magnetic_flux_density",
  )?;

  Ok(())
}
