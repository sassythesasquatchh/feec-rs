use ddf::cochain::Cochain;
use exterior::field::DiffFormClosure;
use formoniq::{
  io::write_cochain_vtk, operators::InnerProductWeightClosure, problems::laplace_beltrami,
};
use manifold::gen::cartesian::CartesianMeshInfo;
use manifold::{
  geometry::coord::CoordRef,
  io::{save_coords_to_file, save_skeleton_to_file},
};
use std::{fs, io::Write};

fn write_cochain(path: &str, cochain: &Cochain) -> std::io::Result<()> {
  let mut file = fs::File::create(path)?;
  for coeff in cochain.coeffs.iter() {
    writeln!(file, "{coeff:.12}")?;
  }
  Ok(())
}

fn main() {
  tracing_subscriber::fmt::init();
  let path = "out/weiland/darcy_flow";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let dim = 2;

  let rhs = DiffFormClosure::scalar(|_p| 1., dim);

  pub fn make_darcy_coeff() -> impl for<'a> Fn(CoordRef) -> f64 {
    let lo = 3.0;
    let hi = 12.0;

    // z(x,y) = Σ c cos(pi k x) cos(pi l y), then threshold at 0.
    let terms: Vec<(u32, u32, f64)> = vec![
      (0, 0, 0.15),
      (1, 0, -0.70),
      (0, 1, 0.55),
      (2, 1, 0.45),
      (3, 2, -0.35),
      (4, 4, -0.28),
      (7, 0, 0.20),
    ];

    move |p: CoordRef| -> f64 {
      assert!(p.len() >= 2, "expected at least 2 coordinates");
      let x = p[0].clamp(0.0, 1.0);
      let y = p[1].clamp(0.0, 1.0);

      let mut z = 0.0;
      for &(k, l, c) in &terms {
        let kf = k as f64;
        let lf = l as f64;
        z += c * (std::f64::consts::PI * kf * x).cos() * (std::f64::consts::PI * lf * y).cos();
      }

      if z >= 0.0 {
        hi
      } else {
        lo
      }
    }
  }

  let inner_product_weight = InnerProductWeightClosure::new(make_darcy_coeff());

  let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, 100, 1.);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);

  let load_vector = formoniq::assemble::assemble_galvec(
    &topology,
    &metric,
    formoniq::operators::SourceElVec::new_weighted(&rhs, &coords, None, &inner_product_weight),
  );

  let boundary_data = |_ivertex| 0.;

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

  // Export potential (0-form) for visualization.
  let vtk_path = format!("{path}/potential.vtk");
  write_cochain_vtk(&vtk_path, &coords, &topology, &galsol, "pressure")
    .expect("failed to write VTK output");

  // Also export the raw mesh and cochains for external visualization tools.
  save_coords_to_file(&coords, format!("{path}/vertices.coords")).unwrap();
  save_skeleton_to_file(&topology, dim, format!("{path}/cells.skel")).unwrap();
  save_skeleton_to_file(&topology, 1, format!("{path}/edges.skel")).unwrap();
  write_cochain(&format!("{path}/pressure.cochain"), &galsol).unwrap();
}
