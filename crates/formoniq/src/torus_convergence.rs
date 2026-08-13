use crate::{
  assemble::assemble_galvec,
  fe::{fe_l2_error, hd_error},
  io::write_1form_vector_field_vtk,
  operators::SourceElVec,
  problems::hodge_laplace,
};

use common::linalg::nalgebra::Vector;
use ddf::cochain::cochain_projection;
use exterior::{field::EmbeddedDiffFormClosure, ExteriorElement};
use manifold::geometry::coord::CoordRef;

use std::{
  fs::{self, File},
  io::{self, BufWriter, Write},
  path::{Path, PathBuf},
  process::Command,
  time::Instant,
};

const MINOR_RADIUS: f64 = 0.3;
const MAJOR_RADIUS: f64 = 1.0;
const GRADE: usize = 1;
const HOMOLOGY_DIM: usize = 2;
const DEFAULT_RESOLUTIONS: [usize; 6] = [0, 1, 2, 3, 4, 5];
const BASE_TORUS_MESH_SIZE: f64 = 0.05;

#[derive(Debug, Clone, PartialEq)]
pub struct TorusConvergenceRecord {
  pub resolution: usize,
  pub h: f64,
  pub l2_error: f64,
  pub l2_rate: f64,
  pub hd_error: f64,
  pub hd_rate: f64,
  pub wall_seconds: f64,
}

fn torus_angles(p: CoordRef, major_radius: f64) -> (f64, f64, f64) {
  let x = p[0];
  let y = p[1];
  let z = p[2];

  let s = (x * x + y * y).sqrt();

  let phi = y.atan2(x);
  let theta = z.atan2(s - major_radius);
  let rho = s;

  (theta, phi, rho)
}

fn torus_covectors(p: CoordRef, major_radius: f64) -> (Vector<f64>, Vector<f64>) {
  let x = p[0];
  let y = p[1];
  let z = p[2];

  let s = (x * x + y * y).sqrt();
  let q = (s - major_radius).powi(2) + z * z;

  let dtheta_x = -z * x / (s * q);
  let dtheta_y = -z * y / (s * q);
  let dtheta_z = (s - major_radius) / q;

  let dphi_x = -y / (s * s);
  let dphi_y = x / (s * s);
  let dphi_z = 0.0;

  let dphi = Vector::from_vec(vec![dphi_x, dphi_y, dphi_z]);
  let dtheta = Vector::from_vec(vec![dtheta_x, dtheta_y, dtheta_z]);
  (dtheta, dphi)
}

fn chart_one_form_to_xyz(p: CoordRef, major_radius: f64, a_theta: f64, a_phi: f64) -> Vector<f64> {
  let (dtheta, dphi) = torus_covectors(p, major_radius);
  a_theta * dtheta + a_phi * dphi
}

fn chart_two_form_to_xyz(p: CoordRef, major_radius: f64, coeff_theta_phi: f64) -> Vector<f64> {
  let (dtheta, dphi) = torus_covectors(p, major_radius);
  coeff_theta_phi
    * ExteriorElement::line(dtheta)
      .wedge(&ExteriorElement::line(dphi))
      .into_coeffs()
}

pub fn build_torus_reference_fields() -> (EmbeddedDiffFormClosure, EmbeddedDiffFormClosure) {
  let u_exact = EmbeddedDiffFormClosure::ambient_one_form(
    move |p: CoordRef| {
      let (theta, phi, rho_val) = torus_angles(p, MAJOR_RADIUS);

      let a_theta = 2.0 * (2.0 * theta).cos() * (3.0 * phi).cos()
        - 2.0 * MINOR_RADIUS * theta.cos() / rho_val * (2.0 * phi).cos();

      let a_phi = -3.0 * (2.0 * theta).sin() * (3.0 * phi).sin()
        - rho_val / MINOR_RADIUS * theta.sin() * (2.0 * phi).sin();

      chart_one_form_to_xyz(p, MAJOR_RADIUS, a_theta, a_phi)
    },
    3,
    2,
  );

  let dif_solution_exact = EmbeddedDiffFormClosure::ambient_k_form(
    move |p: CoordRef| {
      let (theta, phi, rho_val) = torus_angles(p, MAJOR_RADIUS);

      let coeff_theta_phi = (theta.sin().powi(2)
        - (rho_val / MINOR_RADIUS + 4.0 * MINOR_RADIUS / rho_val) * theta.cos())
        * (2.0 * phi).sin();

      chart_two_form_to_xyz(p, MAJOR_RADIUS, coeff_theta_phi)
    },
    3,
    2,
    2,
  );

  (u_exact, dif_solution_exact)
}

fn resolve_example_input_path(relative_path: impl AsRef<Path>) -> io::Result<PathBuf> {
  let relative_path = relative_path.as_ref();
  let cwd_candidate = PathBuf::from(relative_path);
  if cwd_candidate.exists() {
    return Ok(cwd_candidate);
  }

  let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("..")
    .join("..")
    .join("..");

  let workspace_root_candidate = workspace_root.join(relative_path);
  if workspace_root_candidate.exists() {
    return Ok(workspace_root_candidate);
  }

  let parent_candidate = workspace_root.join("..").join(relative_path);
  if parent_candidate.exists() {
    return Ok(parent_candidate);
  }

  Err(io::Error::new(
    io::ErrorKind::NotFound,
    format!(
      "could not find input {:?}; tried {:?}, {:?}, and {:?}",
      relative_path, cwd_candidate, workspace_root_candidate, parent_candidate
    ),
  ))
}

fn workspace_target_path(relative_path: impl AsRef<Path>) -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("..")
    .join("..")
    .join("..")
    .join("target")
    .join(relative_path)
}

fn torus_mesh_path(resolution: usize) -> io::Result<PathBuf> {
  let relative_mesh_path = format!("meshes/torus_shell_resolution_{resolution}.msh");
  match resolve_example_input_path(&relative_mesh_path) {
    Ok(path) => Ok(path),
    Err(missing_mesh_error) => generate_torus_mesh(resolution).map_err(|generation_error| {
      io::Error::new(
        generation_error.kind(),
        format!(
          "{missing_mesh_error}; failed to generate resolution {resolution} mesh with Gmsh: {generation_error}"
        ),
      )
    }),
  }
}

fn generate_torus_mesh(resolution: usize) -> io::Result<PathBuf> {
  let geometry_path = resolve_example_input_path("geometries/torus_shell.geo")?;
  let output_path = workspace_target_path(format!(
    "torus_convergence_meshes/torus_shell_resolution_{resolution}.msh"
  ));
  if output_path.exists() {
    return Ok(output_path);
  }

  let output_dir = output_path.parent().ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::InvalidInput,
      "invalid torus mesh output path",
    )
  })?;
  fs::create_dir_all(output_dir)?;

  let mesh_size = 0.2 * 2.0_f64.powi(-(resolution as i32));
  let clscale = mesh_size / BASE_TORUS_MESH_SIZE;
  let output = Command::new("gmsh")
    .arg("-2")
    .arg(&geometry_path)
    .arg("-clscale")
    .arg(format!("{clscale:.16e}"))
    .arg("-format")
    .arg("msh41")
    .arg("-o")
    .arg(&output_path)
    .output()?;

  if !output.status.success() {
    return Err(io::Error::other(format!(
      "gmsh exited with status {}; stderr: {}",
      output
        .status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated by signal".to_string()),
      String::from_utf8_lossy(&output.stderr)
    )));
  }

  Ok(output_path)
}

pub fn resolution_output_dir(output_dir: impl AsRef<Path>, resolution: usize) -> PathBuf {
  output_dir.as_ref().join(format!("resolution_{resolution}"))
}

pub fn computed_vector_field_output_path(
  output_dir: impl AsRef<Path>,
  resolution: usize,
) -> PathBuf {
  resolution_output_dir(output_dir, resolution).join("solution_computed_vector_field.vtk")
}

pub fn projected_exact_vector_field_output_path(
  output_dir: impl AsRef<Path>,
  resolution: usize,
) -> PathBuf {
  resolution_output_dir(output_dir, resolution).join("solution_projected_exact_vector_field.vtk")
}

pub fn convergence_csv_output_path(output_dir: impl AsRef<Path>) -> PathBuf {
  output_dir.as_ref().join("convergence.csv")
}

pub fn run_torus_convergence(
  output_dir: impl AsRef<Path>,
) -> Result<Vec<TorusConvergenceRecord>, Box<dyn std::error::Error>> {
  run_torus_convergence_for_resolutions(output_dir, &DEFAULT_RESOLUTIONS)
}

pub fn run_torus_convergence_for_resolutions(
  output_dir: impl AsRef<Path>,
  resolutions: &[usize],
) -> Result<Vec<TorusConvergenceRecord>, Box<dyn std::error::Error>> {
  if resolutions.is_empty() {
    return Err(
      io::Error::new(
        io::ErrorKind::InvalidInput,
        "at least one resolution is required",
      )
      .into(),
    );
  }

  let output_dir = output_dir.as_ref();
  let _ = fs::remove_dir_all(output_dir);
  fs::create_dir_all(output_dir)?;

  let (u_exact, dif_solution_exact) = build_torus_reference_fields();

  let f_exact = EmbeddedDiffFormClosure::ambient_one_form(
    move |p: CoordRef| {
      let (theta, phi, rho_val) = torus_angles(p, MAJOR_RADIUS);

      let a = (4.0 / MINOR_RADIUS.powi(2) + 9.0 / rho_val.powi(2)) * (2.0 * theta).sin()
        + 2.0 * theta.sin() * (2.0 * theta).cos() / (MINOR_RADIUS * rho_val);

      let b = (1.0 / MINOR_RADIUS.powi(2) + 4.0 / rho_val.powi(2)) * theta.cos()
        - theta.sin().powi(2) / (MINOR_RADIUS * rho_val);

      let a_prime =
        2.0 * (4.0 / MINOR_RADIUS.powi(2) + 9.0 / rho_val.powi(2)) * (2.0 * theta).cos()
          + 18.0 * MINOR_RADIUS * theta.sin() * (2.0 * theta).sin() / rho_val.powi(3)
          + 2.0 * (theta.cos() * (2.0 * theta).cos() - 2.0 * theta.sin() * (2.0 * theta).sin())
            / (MINOR_RADIUS * rho_val)
          + 2.0 * theta.sin().powi(2) * (2.0 * theta).cos() / rho_val.powi(2);

      let b_prime = 8.0 * MINOR_RADIUS * theta.sin() * theta.cos() / rho_val.powi(3)
        - (1.0 / MINOR_RADIUS.powi(2) + 4.0 / rho_val.powi(2)) * theta.sin()
        - 2.0 * theta.sin() * theta.cos() / (MINOR_RADIUS * rho_val)
        - theta.sin().powi(3) / rho_val.powi(2);

      let f_theta =
        a_prime * (3.0 * phi).cos() - 2.0 * MINOR_RADIUS / rho_val * b * (2.0 * phi).cos();

      let f_phi =
        -3.0 * a * (3.0 * phi).sin() + rho_val / MINOR_RADIUS * b_prime * (2.0 * phi).sin();

      chart_one_form_to_xyz(p, MAJOR_RADIUS, f_theta, f_phi)
    },
    3,
    2,
  );

  let mut errors_l2 = Vec::new();
  let mut errors_hd = Vec::new();
  let mut mesh_widths = Vec::new();
  let mut records = Vec::with_capacity(resolutions.len());

  for &resolution in resolutions {
    let level_start = Instant::now();
    let mesh_path = torus_mesh_path(resolution)?;
    let mesh_bytes = fs::read(&mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let h = metric.mesh_width_max();

    let source_data = assemble_galvec(
      &topology,
      &metric,
      SourceElVec::new(&f_exact, &coords, None),
    );

    let (_, galsol, _) = hodge_laplace::solve_hodge_laplace_source(
      &topology,
      &metric,
      source_data,
      GRADE,
      HOMOLOGY_DIM,
    );

    let u_projected = cochain_projection(&u_exact, &topology, &coords, None);

    let resolution_dir = resolution_output_dir(output_dir, resolution);
    fs::create_dir_all(&resolution_dir)?;
    write_1form_vector_field_vtk(
      computed_vector_field_output_path(output_dir, resolution),
      &coords,
      &topology,
      &galsol,
      "solution_computed_vector_field",
    )?;
    write_1form_vector_field_vtk(
      projected_exact_vector_field_output_path(output_dir, resolution),
      &coords,
      &topology,
      &u_projected,
      "solution_projected_exact_vector_field",
    )?;

    let conv_rate = |errors: &[f64], curr: f64| {
      errors
        .last()
        .zip(mesh_widths.last())
        .map(|(&prev_error, &prev_h)| convergence_rate(prev_h, prev_error, h, curr))
        .unwrap_or(f64::INFINITY)
    };

    let l2_error = fe_l2_error(&galsol, &u_exact, &topology, &coords);
    let l2_rate = conv_rate(&errors_l2, l2_error);
    errors_l2.push(l2_error);

    let dif_galsol = galsol.dif(&topology);
    let derivative_error = fe_l2_error(&dif_galsol, &dif_solution_exact, &topology, &coords);
    let hd_error = hd_error(l2_error, derivative_error);
    let hd_rate = conv_rate(&errors_hd, hd_error);
    errors_hd.push(hd_error);
    mesh_widths.push(h);

    records.push(TorusConvergenceRecord {
      resolution,
      h,
      l2_error,
      l2_rate,
      hd_error,
      hd_rate,
      wall_seconds: level_start.elapsed().as_secs_f64(),
    });
  }

  write_convergence_csv(convergence_csv_output_path(output_dir), &records)?;

  Ok(records)
}

fn convergence_rate(prev_h: f64, prev_error: f64, h: f64, error: f64) -> f64 {
  (prev_error / error).ln() / (prev_h / h).ln()
}

fn write_convergence_csv(
  path: impl AsRef<Path>,
  records: &[TorusConvergenceRecord],
) -> io::Result<()> {
  let file = File::create(path)?;
  let mut writer = BufWriter::new(file);
  writeln!(
    writer,
    "resolution,h,l2_error,l2_rate,hd_error,hd_rate,wall_seconds"
  )?;
  for record in records {
    let l2_rate = finite_rate_field(record.l2_rate);
    let hd_rate = finite_rate_field(record.hd_rate);
    writeln!(
      writer,
      "{},{:.12e},{:.12e},{},{:.12e},{},{:.12e}",
      record.resolution,
      record.h,
      record.l2_error,
      l2_rate,
      record.hd_error,
      hd_rate,
      record.wall_seconds,
    )?;
  }
  Ok(())
}

fn finite_rate_field(rate: f64) -> String {
  if rate.is_finite() {
    format!("{rate:.12e}")
  } else {
    String::new()
  }
}
