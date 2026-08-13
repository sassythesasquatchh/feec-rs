use crate::{
  assemble::{
    assemble_boundary_integral_term, assemble_galvec, boundary_simplices_where_barycenter,
  },
  fe::{fe_l2_error, hd_error},
  operators::{InnerProductWeightClosure, SourceElVec},
  problems::hodge_laplace,
};

use common::linalg::nalgebra::Vector;
use ddf::cochain::partial_cochain_projection;
use exterior::{field::DiffFormClosure, ExteriorElement};
use manifold::{
  gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef, topology::handle::KSimplexIdx,
};

use std::{
  collections::HashSet,
  f64::consts::PI,
  fs::{self, File},
  io::{self, BufWriter, Write},
  path::{Path, PathBuf},
  time::Instant,
};

pub const DEFAULT_RESOLUTIONS: &[usize] = &[2, 4, 8, 16, 32];

#[derive(Debug, Clone, PartialEq)]
pub struct MixedBcHodgeLaplacianConvergenceRecord {
  pub resolution: usize,
  pub h: f64,
  pub u_dofs: usize,
  pub l2_error: f64,
  pub l2_rate: f64,
  pub hd_error: f64,
  pub hd_rate: f64,
  pub wall_seconds: f64,
}

pub fn convergence_csv_output_path(output_dir: impl AsRef<Path>) -> PathBuf {
  output_dir.as_ref().join("convergence.csv")
}

pub fn run_mixed_bc_hodge_laplacian_convergence(
  output_dir: impl AsRef<Path>,
  resolutions: &[usize],
) -> Result<Vec<MixedBcHodgeLaplacianConvergenceRecord>, Box<dyn std::error::Error>> {
  if resolutions.is_empty() {
    return Err("at least one resolution is required".into());
  }
  if resolutions.contains(&0) {
    return Err("resolutions must be positive".into());
  }

  let output_dir = output_dir.as_ref();
  let _ = fs::remove_dir_all(output_dir);
  fs::create_dir_all(output_dir)?;

  let solution_exact = solution_exact();
  let sigma_exact = sigma_exact();
  let laplacian_exact = laplacian_exact();
  let solution_neumann_exact = solution_neumann_exact();
  let sigma_neumann_exact = sigma_neumann_exact();
  let dif_solution_exact = dif_solution_exact();
  let unit_weight = InnerProductWeightClosure::new(|_p| 1.0);

  let mut records: Vec<MixedBcHodgeLaplacianConvergenceRecord> =
    Vec::with_capacity(resolutions.len());

  for &resolution in resolutions {
    let level_start = Instant::now();
    let box_mesh = CartesianMeshInfo::new_unit_scaled(3, resolution, 1.0);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let h = metric.mesh_width_max();

    let strong_dof_predicate = |p: CoordRef| p[0] == 0.0 || p[1] == 0.0 || p[2] == 0.0;
    let weak_dof_predicate = |p: CoordRef| !strong_dof_predicate(p);

    let strong_k_dofs =
      boundary_simplices_where_barycenter(&topology, &coords, 1, strong_dof_predicate)
        .into_iter()
        .collect::<HashSet<usize>>();
    let strong_k_minus_one_dofs =
      boundary_simplices_where_barycenter(&topology, &coords, 0, strong_dof_predicate)
        .into_iter()
        .collect::<HashSet<usize>>();
    let weak_face_dofs =
      boundary_simplices_where_barycenter(&topology, &coords, 2, weak_dof_predicate)
        .into_iter()
        .collect::<HashSet<usize>>();

    let strong_k_dof_predicate = |sidx: KSimplexIdx| strong_k_dofs.contains(&sidx);
    let strong_k_minus_one_dof_predicate =
      |sidx: KSimplexIdx| strong_k_minus_one_dofs.contains(&sidx);
    let weak_face_dof_predicate = |sidx: KSimplexIdx| weak_face_dofs.contains(&sidx);

    let solution_essential_data_map = partial_cochain_projection(
      &solution_exact,
      &topology,
      &coords,
      &strong_k_dof_predicate,
      None,
    );
    let solution_essential_data = |kidx: KSimplexIdx| solution_essential_data_map[&kidx];
    let sigma_essential_data_map = partial_cochain_projection(
      &sigma_exact,
      &topology,
      &coords,
      &strong_k_minus_one_dof_predicate,
      None,
    );
    let sigma_essential_data = |kidx: KSimplexIdx| sigma_essential_data_map[&kidx];

    let sigma_neumann_galvec = assemble_boundary_integral_term(
      &topology,
      &coords,
      0,
      &sigma_neumann_exact,
      None,
      &weak_face_dof_predicate,
    );
    let solution_neumann_galvec = assemble_boundary_integral_term(
      &topology,
      &coords,
      1,
      &solution_neumann_exact,
      None,
      &weak_face_dof_predicate,
    );
    let source_galvec = assemble_galvec(
      &topology,
      &metric,
      SourceElVec::new(&laplacian_exact, &coords, None),
    );

    let (_, u_galsol, _) =
      hodge_laplace::solve_weighted_hodge_laplace_source_with_boundary_conditions(
        &topology,
        &metric,
        Some(sigma_neumann_galvec),
        source_galvec + solution_neumann_galvec,
        1,
        0,
        &coords,
        None,
        &unit_weight,
        &strong_k_dof_predicate,
        &solution_essential_data,
        &strong_k_minus_one_dof_predicate,
        &sigma_essential_data,
      );

    let l2_error = fe_l2_error(&u_galsol, &solution_exact, &topology, &coords);
    let dif_galsol = u_galsol.dif(&topology);
    let derivative_error = fe_l2_error(&dif_galsol, &dif_solution_exact, &topology, &coords);
    let hd_error = hd_error(l2_error, derivative_error);
    let l2_rate = records
      .last()
      .map(|previous| convergence_rate(previous.h, previous.l2_error, h, l2_error))
      .unwrap_or(f64::INFINITY);
    let hd_rate = records
      .last()
      .map(|previous| convergence_rate(previous.h, previous.hd_error, h, hd_error))
      .unwrap_or(f64::INFINITY);

    records.push(MixedBcHodgeLaplacianConvergenceRecord {
      resolution,
      h,
      u_dofs: u_galsol.len(),
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

fn solution_exact() -> DiffFormClosure {
  DiffFormClosure::one_form(
    |p| {
      Vector::from_column_slice(&[
        (PI * p[0]).sin() + p[0] * p[0] + p[1] + p[2],
        (PI * p[1]).cos() + p[0] + p[2] * p[2],
        (PI * p[2]).sin() + p[0] * p[0] + p[1] + p[2] * p[2],
      ])
    },
    3,
  )
}

fn sigma_exact() -> DiffFormClosure {
  DiffFormClosure::scalar(
    |p| PI * (-(PI * p[0]).cos() + (PI * p[1]).sin() - (PI * p[2]).cos()) - 2.0 * (p[0] + p[2]),
    3,
  )
}

fn laplacian_exact() -> DiffFormClosure {
  DiffFormClosure::one_form(
    |p| {
      Vector::from_column_slice(&[
        (PI * PI) * (PI * p[0]).sin() - 2.0,
        (PI * PI) * (PI * p[1]).cos() - 2.0,
        (PI * PI) * (PI * p[2]).sin() - 4.0,
      ])
    },
    3,
  )
}

fn dif_solution_exact() -> DiffFormClosure {
  DiffFormClosure::new(
    Box::new(|p| {
      ExteriorElement::new(
        Vector::from_column_slice(&[0.0, 2.0 * p[0] - 1.0, 1.0 - 2.0 * p[2]]),
        3,
        2,
      )
    }),
    3,
    2,
  )
}

fn solution_neumann_exact() -> DiffFormClosure {
  DiffFormClosure::new(
    Box::new(|p| {
      ExteriorElement::new(
        Vector::from_column_slice(&[1.0 - 2.0 * p[2], 1.0 - 2.0 * p[0], 0.0]),
        3,
        1,
      )
    }),
    3,
    1,
  )
}

fn sigma_neumann_exact() -> DiffFormClosure {
  DiffFormClosure::new(
    Box::new(|p| {
      ExteriorElement::new(
        Vector::from_column_slice(&[
          -((PI * p[2]).sin() + p[0] * p[0] + p[1] + p[2] * p[2]),
          (PI * p[1]).cos() + p[0] + p[2] * p[2],
          -((PI * p[0]).sin() + p[0] * p[0] + p[1] + p[2]),
        ]),
        3,
        2,
      )
    }),
    3,
    2,
  )
}

fn convergence_rate(prev_h: f64, prev_error: f64, h: f64, error: f64) -> f64 {
  (prev_error / error).ln() / (prev_h / h).ln()
}

fn write_convergence_csv(
  path: impl AsRef<Path>,
  records: &[MixedBcHodgeLaplacianConvergenceRecord],
) -> io::Result<()> {
  let file = File::create(path)?;
  let mut writer = BufWriter::new(file);
  writeln!(
    writer,
    "resolution,h,u_dofs,l2_error,l2_rate,hd_error,hd_rate,wall_seconds"
  )?;
  for record in records {
    let l2_rate = finite_rate_field(record.l2_rate);
    let hd_rate = finite_rate_field(record.hd_rate);
    writeln!(
      writer,
      "{},{:.12e},{},{:.12e},{},{:.12e},{},{:.12e}",
      record.resolution,
      record.h,
      record.u_dofs,
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
