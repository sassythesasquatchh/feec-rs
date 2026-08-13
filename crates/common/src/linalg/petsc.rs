use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
  fs::File,
  io::{BufReader, BufWriter, Write},
};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

use super::nalgebra::{CsrMatrix, Matrix, Vector};

const PETSC_SOLVER_ENV: &str = "PETSC_SOLVER_PATH";
static PETSC_WORKSPACE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn petsc_solver_path() -> PathBuf {
  if let Ok(path) = std::env::var(PETSC_SOLVER_ENV) {
    let path = resolve_petsc_solver_path(PathBuf::from(path));
    if petsc_solver_binaries_exist(&path) {
      return path;
    }
  }

  if let Ok(current_exe) = std::env::current_exe() {
    let current_exe = canonicalize_if_possible(current_exe);
    if let Some(path) = search_petsc_solver_from(&current_exe) {
      return path;
    }
  }

  if let Ok(current_dir) = std::env::current_dir() {
    let current_dir = canonicalize_if_possible(current_dir);
    if let Some(path) = search_petsc_solver_from(&current_dir) {
      return path;
    }
  }

  let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
  let path = search_petsc_solver_from(manifest_dir)
    .unwrap_or_else(|| manifest_dir.join("..").join("..").join("petsc-solver"));

  canonicalize_if_possible(path)
}

fn petsc_solver_binaries_exist(path: &Path) -> bool {
  ["ghiep.out", "ghep_reduced.out", "hils.out"]
    .iter()
    .all(|binary| path.join(binary).exists())
}

fn resolve_petsc_solver_path(path: PathBuf) -> PathBuf {
  if path.is_absolute() {
    return canonicalize_if_possible(path);
  }

  if petsc_solver_binaries_exist(&path) {
    return canonicalize_if_possible(path);
  }

  search_relative_path_upwards(&path).unwrap_or(path)
}

fn search_petsc_solver_from(start: &Path) -> Option<PathBuf> {
  start
    .ancestors()
    .map(|ancestor| ancestor.join("petsc-solver"))
    .find(|path| petsc_solver_binaries_exist(path))
    .or_else(|| {
      start
        .ancestors()
        .map(|ancestor| ancestor.join("petsc-solver"))
        .find(|path| path.exists())
    })
    .map(canonicalize_if_possible)
}

fn search_relative_path_upwards(relative_path: &Path) -> Option<PathBuf> {
  let mut starts = Vec::new();

  if let Ok(current_exe) = std::env::current_exe() {
    starts.push(canonicalize_if_possible(current_exe));
  }

  if let Ok(current_dir) = std::env::current_dir() {
    starts.push(canonicalize_if_possible(current_dir));
  }

  for start in &starts {
    if let Some(path) = start
      .ancestors()
      .map(|ancestor| ancestor.join(relative_path))
      .find(|path| petsc_solver_binaries_exist(path))
    {
      return Some(canonicalize_if_possible(path));
    }
  }

  for start in starts {
    if let Some(path) = start
      .ancestors()
      .map(|ancestor| ancestor.join(relative_path))
      .find(|path| path.exists())
    {
      return Some(canonicalize_if_possible(path));
    }
  }

  None
}

fn canonicalize_if_possible(path: PathBuf) -> PathBuf {
  std::fs::canonicalize(&path).unwrap_or(path)
}

fn run_petsc_command<S>(binary: &Path, current_dir: &Path, args: &[S]) -> std::process::ExitStatus
where
  S: AsRef<std::ffi::OsStr>,
{
  std::process::Command::new(binary)
    .current_dir(current_dir)
    .args(args)
    .status()
    .unwrap_or_else(|error| {
      panic!(
        "failed to launch PETSc helper {:?} in {:?}: {}",
        binary, current_dir, error
      )
    })
}

fn create_petsc_workspace(solver_path: &Path) -> PathBuf {
  let workspace_id = PETSC_WORKSPACE_COUNTER.fetch_add(1, Ordering::Relaxed);
  let workspace = solver_path
    .join("tmp")
    .join(format!("run-{}-{workspace_id}", std::process::id()));

  if workspace.exists() {
    std::fs::remove_dir_all(&workspace)
      .unwrap_or_else(|error| panic!("failed to clear PETSc workspace {:?}: {}", workspace, error));
  }

  std::fs::create_dir_all(workspace.join("in")).unwrap_or_else(|error| {
    panic!(
      "failed to create PETSc workspace input dir {:?}: {}",
      workspace.join("in"),
      error
    )
  });
  std::fs::create_dir_all(workspace.join("out")).unwrap_or_else(|error| {
    panic!(
      "failed to create PETSc workspace output dir {:?}: {}",
      workspace.join("out"),
      error
    )
  });

  workspace
}

const PETSC_MAT_FILE_CLASSID: i32 = 1211216;
const PETSC_VEC_FILE_CLASSID: i32 = 1211214;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhiepWhich {
  Smallest,
  Largest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhiepReducedSolve {
  Direct,
  Iterative,
}

pub fn petsc_write_matrix(matrix: &CsrMatrix, filename: &str) -> std::io::Result<()> {
  let file = File::create(filename)?;
  let mut writer = BufWriter::new(file);

  writer.write_i32::<BigEndian>(PETSC_MAT_FILE_CLASSID)?;

  let nrows = matrix.nrows() as i32;
  let ncols = matrix.ncols() as i32;
  let nnz = matrix.nnz() as i32;
  writer.write_i32::<BigEndian>(nrows)?;
  writer.write_i32::<BigEndian>(ncols)?;
  writer.write_i32::<BigEndian>(nnz)?;

  let row_offsets = matrix.row_offsets();
  for i in 0..nrows as usize {
    let row_nnz = (row_offsets[i + 1] - row_offsets[i]) as i32;
    writer.write_i32::<BigEndian>(row_nnz)?;
  }

  let col_indices = matrix.col_indices();
  for &col in col_indices {
    writer.write_i32::<BigEndian>(col as i32)?;
  }

  let values = matrix.values();
  for &value in values {
    writer.write_f64::<BigEndian>(value)?;
  }

  writer.flush()?;
  Ok(())
}

pub fn petsc_write_vector(vector: &Vector, filename: &str) -> std::io::Result<()> {
  let file = File::create(filename)?;
  let mut writer = BufWriter::new(file);

  writer.write_i32::<BigEndian>(PETSC_VEC_FILE_CLASSID)?;

  let nrows = vector.nrows() as i32;
  writer.write_i32::<BigEndian>(nrows)?;

  for &value in vector {
    writer.write_f64::<BigEndian>(value)?;
  }

  writer.flush()?;
  Ok(())
}

pub fn petsc_read_vector(filename: &str) -> std::io::Result<Vector> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let magic = reader.read_i32::<BigEndian>()?;
  assert_eq!(magic, PETSC_VEC_FILE_CLASSID);

  let nrows = reader.read_i32::<BigEndian>()? as usize;

  let mut vector = Vector::zeros(nrows);
  for i in 0..nrows {
    vector[i] = reader.read_f64::<BigEndian>()?;
  }

  Ok(vector)
}

pub fn petsc_read_eigenvals(filename: &str) -> std::io::Result<Vector> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let neigenvals = reader.read_i32::<BigEndian>()? as usize;
  let mut eigenvals = Vector::zeros(neigenvals);

  for i in 0..neigenvals {
    eigenvals[i] = reader.read_f64::<BigEndian>()?;
  }

  Ok(eigenvals)
}

pub fn petsc_read_eigenvecs(filename: &str) -> std::io::Result<nalgebra::DMatrix<f64>> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let nrows = reader.read_i32::<BigEndian>()? as usize;
  let ncols = reader.read_i32::<BigEndian>()? as usize;

  let mut data = Vec::with_capacity(nrows * ncols);
  for _ in 0..ncols {
    let magic = reader.read_i32::<BigEndian>()?;
    assert_eq!(magic, PETSC_VEC_FILE_CLASSID);

    let this_nrows = reader.read_i32::<BigEndian>()? as usize;
    assert_eq!(this_nrows, nrows);

    for _ in 0..nrows {
      data.push(reader.read_f64::<BigEndian>()?);
    }
  }
  Ok(Matrix::from_column_slice(nrows, ncols, &data))
}

fn ghiep_args(which: GhiepWhich, neigen_values: usize) -> Vec<String> {
  match which {
    GhiepWhich::Smallest => vec![
      "-st_pc_factor_mat_solver_type".into(),
      "mumps".into(),
      "-st_type".into(),
      "sinvert".into(),
      "-st_shift".into(),
      "0.1".into(),
      "-eps_target".into(),
      "0.".into(),
      "-eps_nev".into(),
      neigen_values.to_string(),
    ],
    GhiepWhich::Largest => vec![
      "-st_pc_factor_mat_solver_type".into(),
      "mumps".into(),
      "-st_pc_type".into(),
      "lu".into(),
      "st_ksp_type".into(),
      "preonly".into(),
      "-eps_nev".into(),
      neigen_values.to_string(),
      "-eps_largest_magnitude".into(),
    ],
  }
}

fn ghep_reduced_args(
  which: GhiepWhich,
  neigen_values: usize,
  mass_solve: GhiepReducedSolve,
) -> Vec<String> {
  let mut args = vec![
    "-eps_type".into(),
    "krylovschur".into(),
    "-eps_nev".into(),
    neigen_values.to_string(),
    "-eps_tol".into(),
    "1e-10".into(),
  ];

  match which {
    GhiepWhich::Smallest => {
      args.push("-eps_smallest_real".into());
    }
    GhiepWhich::Largest => {
      args.push("-eps_largest_real".into());
    }
  }

  let ncv = (neigen_values.saturating_mul(4)).max(neigen_values);
  args.push("-eps_ncv".into());
  args.push(ncv.to_string());

  match mass_solve {
    GhiepReducedSolve::Direct => {
      args.extend(
        [
          "-st_ksp_type",
          "preonly",
          "-st_pc_type",
          "lu",
          "-st_pc_factor_mat_solver_type",
          "mumps",
          "-mkm1_ksp_type",
          "preonly",
          "-mkm1_pc_type",
          "lu",
          "-mkm1_pc_factor_mat_solver_type",
          "mumps",
        ]
        .iter()
        .map(|s| (*s).into()),
      );
    }
    GhiepReducedSolve::Iterative => {
      args.extend(
        [
          "-st_ksp_type",
          "cg",
          "-st_pc_type",
          "jacobi",
          "-st_ksp_rtol",
          "1e-10",
          "-st_ksp_max_it",
          "50",
          "-mkm1_ksp_type",
          "cg",
          "-mkm1_pc_type",
          "jacobi",
          "-mkm1_ksp_rtol",
          "1e-10",
          "-mkm1_ksp_max_it",
          "50",
        ]
        .iter()
        .map(|s| (*s).into()),
      );
    }
  }

  args
}

pub fn petsc_ghiep_with_which(
  lhs: &CsrMatrix,
  rhs: &CsrMatrix,
  neigen_values: usize,
  which: GhiepWhich,
) -> (Vector, Matrix) {
  let solver_path = petsc_solver_path();
  let workspace = create_petsc_workspace(&solver_path);

  let path = workspace.join("in");
  petsc_write_matrix(lhs, path.join("A.bin").to_str().unwrap()).unwrap();
  petsc_write_matrix(rhs, path.join("B.bin").to_str().unwrap()).unwrap();

  let binary = solver_path.join("ghiep.out");
  let args = ghiep_args(which, neigen_values);

  let status = run_petsc_command(&binary, &workspace, &args);
  assert!(status.success());

  let eigenvals =
    petsc_read_eigenvals(workspace.join("out/eigenvals.bin").to_str().unwrap()).unwrap();
  let eigenvecs =
    petsc_read_eigenvecs(workspace.join("out/eigenvecs.bin").to_str().unwrap()).unwrap();

  let k = neigen_values.min(eigenvals.len());

  let eigenvals = eigenvals.rows(0, k).into_owned();
  let eigenvecs = eigenvecs.columns(0, k).into_owned();
  (eigenvals, eigenvecs)
}

pub fn petsc_ghiep(lhs: &CsrMatrix, rhs: &CsrMatrix, neigen_values: usize) -> (Vector, Matrix) {
  petsc_ghiep_with_which(lhs, rhs, neigen_values, GhiepWhich::Smallest)
}

pub fn petsc_ghiep_largest(
  lhs: &CsrMatrix,
  rhs: &CsrMatrix,
  neigen_values: usize,
) -> (Vector, Matrix) {
  petsc_ghiep_with_which(lhs, rhs, neigen_values, GhiepWhich::Largest)
}

/// Operators defining the reduced generalized Hodge eigenproblem.
///
/// Keeping these matrices together makes the discrete equation explicit and
/// prevents callers from accidentally swapping same-shaped operator blocks.
pub struct GhepReducedOperators<'a> {
  pub l: &'a CsrMatrix,
  pub d: &'a CsrMatrix,
  pub c: &'a CsrMatrix,
  pub mkm1: &'a CsrMatrix,
  pub mk: &'a CsrMatrix,
}

pub fn petsc_ghep_reduced_with_which(
  operators: GhepReducedOperators<'_>,
  neigen_values: usize,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> (Vector, Matrix, Matrix) {
  let solver_path = petsc_solver_path();
  let workspace = create_petsc_workspace(&solver_path);

  let in_path = workspace.join("in");
  petsc_write_matrix(operators.l, in_path.join("L.bin").to_str().unwrap()).unwrap();
  petsc_write_matrix(operators.d, in_path.join("D.bin").to_str().unwrap()).unwrap();
  petsc_write_matrix(operators.c, in_path.join("C.bin").to_str().unwrap()).unwrap();
  petsc_write_matrix(operators.mkm1, in_path.join("Mkm1.bin").to_str().unwrap()).unwrap();
  petsc_write_matrix(operators.mk, in_path.join("Mk.bin").to_str().unwrap()).unwrap();

  let binary = solver_path.join("ghep_reduced.out");
  let args = ghep_reduced_args(which, neigen_values, mass_solve);

  let status = run_petsc_command(&binary, &workspace, &args);
  assert!(status.success());

  let eigenvals =
    petsc_read_eigenvals(workspace.join("out/eigenvals.bin").to_str().unwrap()).unwrap();

  let sigma_eigenvecs =
    petsc_read_eigenvecs(workspace.join("out/eigenvecs_sigma.bin").to_str().unwrap()).unwrap();
  let u_eigenvecs =
    petsc_read_eigenvecs(workspace.join("out/eigenvecs_u.bin").to_str().unwrap()).unwrap();

  let k = neigen_values.min(eigenvals.len());
  let eigenvals = eigenvals.rows(0, k).into_owned();

  let u_eigenvecs = u_eigenvecs.columns(0, k).into_owned();
  let sigma_eigenvecs = sigma_eigenvecs.columns(0, k).into_owned();

  (eigenvals, sigma_eigenvecs, u_eigenvecs)
}

pub fn petsc_ghep_reduced_direct(
  l: &CsrMatrix,
  d: &CsrMatrix,
  c: &CsrMatrix,
  mkm1: &CsrMatrix,
  mk: &CsrMatrix,
  neigen_values: usize,
  which: GhiepWhich,
) -> (Vector, Matrix, Matrix) {
  petsc_ghep_reduced_with_which(
    GhepReducedOperators { l, d, c, mkm1, mk },
    neigen_values,
    which,
    GhiepReducedSolve::Direct,
  )
}

pub fn petsc_ghep_reduced_iterative(
  l: &CsrMatrix,
  d: &CsrMatrix,
  c: &CsrMatrix,
  mkm1: &CsrMatrix,
  mk: &CsrMatrix,
  neigen_values: usize,
  which: GhiepWhich,
) -> (Vector, Matrix, Matrix) {
  petsc_ghep_reduced_with_which(
    GhepReducedOperators { l, d, c, mkm1, mk },
    neigen_values,
    which,
    GhiepReducedSolve::Iterative,
  )
}

pub fn petsc_saddle_point(lhs: &CsrMatrix, rhs: &Vector, has_harmonics: bool) -> Vector {
  let solver_path = petsc_solver_path();
  let workspace = create_petsc_workspace(&solver_path);

  let in_path = workspace.join("in");
  petsc_write_matrix(lhs, in_path.join("A.bin").to_str().unwrap()).unwrap();
  petsc_write_vector(rhs, in_path.join("b.bin").to_str().unwrap()).unwrap();

  let binary = solver_path.join("hils.out");

  let preferred_args = if has_harmonics {
    harmonic_saddle_point_args(true)
  } else {
    direct_saddle_point_args()
  };
  let mut status = run_petsc_command(&binary, &workspace, &preferred_args);

  if has_harmonics && !status.success() {
    eprintln!(
      "PETSc harmonic saddle-point solve failed with fieldsplit preconditioning; retrying with a direct LU factorization."
    );

    status = run_petsc_command(&binary, &workspace, &direct_saddle_point_args());
  }

  assert!(status.success());

  let out_path = workspace.join("out");
  petsc_read_vector(out_path.join("x.bin").to_str().unwrap()).unwrap()
}

fn harmonic_saddle_point_args(has_harmonics: bool) -> Vec<&'static str> {
  if !has_harmonics {
    return Vec::new();
  }

  vec![
    // Outer Krylov
    "-ksp_type",
    "gmres",
    "-ksp_max_it",
    "1000",
    "-ksp_rtol",
    "1e-9",
    "-ksp_error_if_not_converged",
    // Saddle-point aware preconditioner (Schur complement)
    "-pc_type",
    "fieldsplit",
    "-pc_fieldsplit_type",
    "schur",
    "-pc_fieldsplit_detect_saddle_point",
    // Use upper factorization
    "-pc_fieldsplit_schur_fact_type",
    "upper",
    // How to (approximately) solve the (non-saddle) block
    "-fieldsplit_0_ksp_type",
    "preonly",
    "-fieldsplit_0_pc_type",
    "ilu",
    // How to handle the Schur block
    "-fieldsplit_1_ksp_type",
    "preonly",
    "-fieldsplit_1_pc_type",
    "none",
  ]
}

fn direct_saddle_point_args() -> Vec<&'static str> {
  vec![
    "-ksp_type",
    "preonly",
    "-pc_type",
    "lu",
    "-pc_factor_mat_solver_type",
    "mumps",
  ]
}

#[cfg(test)]
mod tests {
  use super::*;

  fn assert_petsc_solver_sources_exist(path: &Path) {
    for source in ["Makefile", "ghiep.c", "ghep_reduced.c", "hils.c"] {
      assert!(path.join(source).is_file(), "missing {source} in {path:?}");
    }
  }

  #[test]
  fn petsc_solver_path_defaults_to_release_source_directory() {
    let path = petsc_solver_path();
    assert_eq!(
      path.file_name().and_then(|p| p.to_str()),
      Some("petsc-solver")
    );
    assert_petsc_solver_sources_exist(&path);
  }

  #[test]
  fn relative_petsc_solver_override_resolves_to_release_source_directory() {
    let path = resolve_petsc_solver_path(PathBuf::from("petsc-solver"));
    assert!(path.is_absolute(), "resolved path: {path:?}");
    assert_petsc_solver_sources_exist(&path);
  }

  #[test]
  fn ghiep_args_smallest_uses_shift_invert() {
    let args = ghiep_args(GhiepWhich::Smallest, 4);
    assert!(args.iter().any(|arg| arg == "-st_type"));
    assert!(args.iter().any(|arg| arg == "sinvert"));
    assert!(args.iter().any(|arg| arg == "4"));
  }

  #[test]
  fn ghiep_args_largest_requests_largest_magnitude() {
    let args = ghiep_args(GhiepWhich::Largest, 3);
    assert!(args.iter().any(|arg| arg == "-eps_largest_magnitude"));
    assert!(args.iter().any(|arg| arg == "3"));
    assert!(!args.iter().any(|arg| arg == "-st_type"));
  }

  #[test]
  fn ghep_reduced_args_direct_uses_lu() {
    let args = ghep_reduced_args(GhiepWhich::Largest, 4, GhiepReducedSolve::Direct);
    assert!(args.iter().any(|arg| arg == "-mkm1_ksp_type"));
    assert!(args.iter().any(|arg| arg == "preonly"));
    assert!(args.iter().any(|arg| arg == "-mkm1_pc_type"));
    assert!(args.iter().any(|arg| arg == "lu"));
  }

  #[test]
  fn ghep_reduced_args_iterative_uses_cg() {
    let args = ghep_reduced_args(GhiepWhich::Largest, 4, GhiepReducedSolve::Iterative);
    assert!(args.iter().any(|arg| arg == "-mkm1_ksp_type"));
    assert!(args.iter().any(|arg| arg == "cg"));
    assert!(args.iter().any(|arg| arg == "-mkm1_pc_type"));
    assert!(args.iter().any(|arg| arg == "jacobi"));
  }

  #[test]
  fn harmonic_saddle_point_args_without_harmonics_are_empty() {
    let args = harmonic_saddle_point_args(false);
    assert!(args.is_empty());
  }

  #[test]
  fn harmonic_saddle_point_args_with_harmonics_use_fieldsplit() {
    let args = harmonic_saddle_point_args(true);
    assert!(args.contains(&"-pc_type"));
    assert!(args.contains(&"fieldsplit"));
    assert!(args.contains(&"-ksp_error_if_not_converged"));
  }

  #[test]
  fn direct_saddle_point_args_use_lu_with_mumps() {
    let args = direct_saddle_point_args();
    assert!(args.contains(&"-pc_type"));
    assert!(args.contains(&"lu"));
    assert!(args.contains(&"-pc_factor_mat_solver_type"));
    assert!(args.contains(&"mumps"));
  }
}
