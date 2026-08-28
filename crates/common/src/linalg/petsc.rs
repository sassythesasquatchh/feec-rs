use std::fmt;
use std::io;
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
  try_run_petsc_command(binary, current_dir, args).unwrap_or_else(|error| panic!("{error}"))
}

fn create_petsc_workspace(solver_path: &Path) -> PathBuf {
  try_create_petsc_workspace(solver_path).unwrap_or_else(|error| panic!("{error}"))
}

fn try_run_petsc_command<S>(
  binary: &Path,
  current_dir: &Path,
  args: &[S],
) -> Result<std::process::ExitStatus, PetscError>
where
  S: AsRef<std::ffi::OsStr>,
{
  std::process::Command::new(binary)
    .current_dir(current_dir)
    .args(args)
    .status()
    .map_err(|source| PetscError::HelperLaunch {
      binary: binary.to_path_buf(),
      workspace: current_dir.to_path_buf(),
      source,
    })
}

fn try_create_petsc_workspace(solver_path: &Path) -> Result<PathBuf, PetscError> {
  let workspace_root = solver_path.join("tmp");
  std::fs::create_dir_all(&workspace_root).map_err(|source| PetscError::WorkspaceCreate {
    path: workspace_root.clone(),
    source,
  })?;

  loop {
    let workspace_id = PETSC_WORKSPACE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let workspace = workspace_root.join(format!("run-{}-{workspace_id}", std::process::id()));
    match std::fs::create_dir(&workspace) {
      Ok(()) => {
        for child in ["in", "out"] {
          let path = workspace.join(child);
          std::fs::create_dir(&path)
            .map_err(|source| PetscError::WorkspaceCreate { path, source })?;
        }
        return Ok(workspace);
      }
      Err(source) if source.kind() == io::ErrorKind::AlreadyExists => continue,
      Err(source) => {
        return Err(PetscError::WorkspaceCreate {
          path: workspace,
          source,
        })
      }
    }
  }
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

/// Typed failures from the external PETSc/SLEPc helper boundary.
#[derive(Debug)]
pub enum PetscError {
  InvalidGeneralizedEigenproblemDimensions {
    lhs_rows: usize,
    lhs_cols: usize,
    rhs_rows: usize,
    rhs_cols: usize,
  },
  InvalidRequestedEigenpairs {
    requested: usize,
    dimension: usize,
  },
  InvalidReducedGeneralizedEigenproblemDimensions {
    l_rows: usize,
    l_cols: usize,
    d_rows: usize,
    d_cols: usize,
    c_rows: usize,
    c_cols: usize,
    mkm1_rows: usize,
    mkm1_cols: usize,
    mk_rows: usize,
    mk_cols: usize,
  },
  NonFiniteInputEntry {
    operator: &'static str,
    row: usize,
    col: usize,
    value: f64,
  },
  WorkspaceCreate {
    path: PathBuf,
    source: io::Error,
  },
  InputWrite {
    binary: PathBuf,
    workspace: PathBuf,
    path: PathBuf,
    source: io::Error,
  },
  HelperLaunch {
    binary: PathBuf,
    workspace: PathBuf,
    source: io::Error,
  },
  HelperExit {
    binary: PathBuf,
    workspace: PathBuf,
    status: std::process::ExitStatus,
  },
  OutputRead {
    binary: PathBuf,
    workspace: PathBuf,
    path: PathBuf,
    source: io::Error,
  },
  InvalidHelperOutput {
    binary: PathBuf,
    workspace: PathBuf,
    detail: String,
  },
  InvalidReducedHelperOutputShape {
    binary: PathBuf,
    workspace: PathBuf,
    eigenvalue_count: usize,
    sigma_rows: usize,
    sigma_cols: usize,
    u_rows: usize,
    u_cols: usize,
    expected_sigma_rows: usize,
    expected_u_rows: usize,
  },
  InsufficientReducedEigenpairConvergence {
    binary: PathBuf,
    workspace: PathBuf,
    requested: usize,
    converged: usize,
  },
  ReducedEigenproblemResourceOverflow {
    quantity: &'static str,
  },
}

impl fmt::Display for PetscError {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidGeneralizedEigenproblemDimensions {
        lhs_rows,
        lhs_cols,
        rhs_rows,
        rhs_cols,
      } => write!(
        formatter,
        "PETSc generalized eigenproblem requires matching square matrices, got lhs {lhs_rows}x{lhs_cols} and rhs {rhs_rows}x{rhs_cols}"
      ),
      Self::InvalidRequestedEigenpairs {
        requested,
        dimension,
      } => write!(
        formatter,
        "PETSc generalized eigenproblem rank must be in 1..={dimension}, got {requested}"
      ),
      Self::InvalidReducedGeneralizedEigenproblemDimensions {
        l_rows,
        l_cols,
        d_rows,
        d_cols,
        c_rows,
        c_cols,
        mkm1_rows,
        mkm1_cols,
        mk_rows,
        mk_cols,
      } => write!(
        formatter,
        "PETSc reduced generalized Hodge eigenproblem has incompatible blocks: L={l_rows}x{l_cols}, D={d_rows}x{d_cols}, C={c_rows}x{c_cols}, Mkm1={mkm1_rows}x{mkm1_cols}, Mk={mk_rows}x{mk_cols}"
      ),
      Self::NonFiniteInputEntry {
        operator,
        row,
        col,
        value,
      } => write!(
        formatter,
        "PETSc generalized eigenproblem {operator} entry ({row}, {col}) is non-finite: {value}"
      ),
      Self::WorkspaceCreate { path, source } => {
        write!(formatter, "failed to create PETSc workspace {path:?}: {source}")
      }
      Self::InputWrite {
        binary,
        workspace,
        path,
        source,
      } => write!(
        formatter,
        "failed to write PETSc helper input {path:?} for binary {binary:?} in workspace {workspace:?}: {source}"
      ),
      Self::HelperLaunch {
        binary,
        workspace,
        source,
      } => write!(
        formatter,
        "failed to launch PETSc helper {binary:?} in workspace {workspace:?}: {source}"
      ),
      Self::HelperExit {
        binary,
        workspace,
        status,
      } => write!(
        formatter,
        "PETSc helper {binary:?} exited with {status} in workspace {workspace:?}"
      ),
      Self::OutputRead {
        binary,
        workspace,
        path,
        source,
      } => write!(
        formatter,
        "failed to read PETSc helper output {path:?} from binary {binary:?} in workspace {workspace:?}: {source}"
      ),
      Self::InvalidHelperOutput {
        binary,
        workspace,
        detail,
      } => write!(
        formatter,
        "invalid PETSc helper output from binary {binary:?} in workspace {workspace:?}: {detail}"
      ),
      Self::InvalidReducedHelperOutputShape {
        binary,
        workspace,
        eigenvalue_count,
        sigma_rows,
        sigma_cols,
        u_rows,
        u_cols,
        expected_sigma_rows,
        expected_u_rows,
      } => write!(
        formatter,
        "invalid reduced PETSc helper output from binary {binary:?} in workspace {workspace:?}: {eigenvalue_count} eigenvalues, sigma {sigma_rows}x{sigma_cols} (expected {expected_sigma_rows} rows), u {u_rows}x{u_cols} (expected {expected_u_rows} rows)"
      ),
      Self::InsufficientReducedEigenpairConvergence {
        binary,
        workspace,
        requested,
        converged,
      } => write!(
        formatter,
        "PETSc reduced Hodge helper {binary:?} converged {converged} eigenpairs, fewer than requested {requested}, in workspace {workspace:?}"
      ),
      Self::ReducedEigenproblemResourceOverflow { quantity } => write!(
        formatter,
        "PETSc reduced Hodge eigenproblem resource diagnostic overflow for {quantity}"
      ),
    }
  }
}

impl std::error::Error for PetscError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    match self {
      Self::WorkspaceCreate { source, .. }
      | Self::InputWrite { source, .. }
      | Self::HelperLaunch { source, .. }
      | Self::OutputRead { source, .. } => Some(source),
      _ => None,
    }
  }
}

/// Diagnostics for one partial generalized Hermitian eigenproblem helper run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PetscGeneralizedEigenDiagnostics {
  pub dimension: usize,
  pub requested_rank: usize,
  pub converged_rank: usize,
  pub lhs_nonzeros: usize,
  pub rhs_nonzeros: usize,
  pub workspace: PathBuf,
  pub binary: PathBuf,
}

/// Partial generalized eigenpairs plus the exact helper/run context.
#[derive(Debug)]
pub struct PetscGeneralizedEigenpairs {
  pub eigenvalues: Vector,
  pub eigenvectors: Matrix,
  pub diagnostics: PetscGeneralizedEigenDiagnostics,
}

/// Diagnostics for one sparse mixed/reduced generalized Hodge helper run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PetscReducedGeneralizedEigenDiagnostics {
  pub sigma_dimension: usize,
  pub u_dimension: usize,
  pub requested_rank: usize,
  pub converged_rank: usize,
  pub l_nonzeros: usize,
  pub d_nonzeros: usize,
  pub c_nonzeros: usize,
  pub mkm1_nonzeros: usize,
  pub mk_nonzeros: usize,
  pub input_nonzeros: usize,
  pub which: GhiepWhich,
  pub mass_solve: GhiepReducedSolve,
  pub workspace: PathBuf,
  pub binary: PathBuf,
}

/// Mixed/reduced generalized Hodge eigenpairs with retained helper context.
#[derive(Debug)]
pub struct PetscReducedGeneralizedEigenpairs {
  pub eigenvalues: Vector,
  pub sigma_eigenvectors: Matrix,
  pub u_eigenvectors: Matrix,
  pub diagnostics: PetscReducedGeneralizedEigenDiagnostics,
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
  if magic != PETSC_VEC_FILE_CLASSID {
    return Err(io::Error::new(
      io::ErrorKind::InvalidData,
      format!("invalid PETSc vector class id {magic}"),
    ));
  }

  let nrows = read_nonnegative_i32(&mut reader, "PETSc vector row count")?;

  let mut vector = Vector::zeros(nrows);
  for i in 0..nrows {
    vector[i] = reader.read_f64::<BigEndian>()?;
  }

  Ok(vector)
}

pub fn petsc_read_eigenvals(filename: &str) -> std::io::Result<Vector> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let neigenvals = read_nonnegative_i32(&mut reader, "PETSc eigenvalue count")?;
  let mut eigenvals = Vector::zeros(neigenvals);

  for i in 0..neigenvals {
    eigenvals[i] = reader.read_f64::<BigEndian>()?;
  }

  Ok(eigenvals)
}

pub fn petsc_read_eigenvecs(filename: &str) -> std::io::Result<nalgebra::DMatrix<f64>> {
  let file = File::open(filename)?;
  let mut reader = BufReader::new(file);

  let nrows = read_nonnegative_i32(&mut reader, "PETSc eigenvector row count")?;
  let ncols = read_nonnegative_i32(&mut reader, "PETSc eigenvector column count")?;
  let entry_count = nrows.checked_mul(ncols).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::InvalidData,
      "PETSc eigenvector dimensions overflow usize",
    )
  })?;

  let mut data = Vec::with_capacity(entry_count);
  for _ in 0..ncols {
    let magic = reader.read_i32::<BigEndian>()?;
    if magic != PETSC_VEC_FILE_CLASSID {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("invalid PETSc eigenvector class id {magic}"),
      ));
    }

    let this_nrows = read_nonnegative_i32(&mut reader, "PETSc eigenvector row count")?;
    if this_nrows != nrows {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("PETSc eigenvector column has {this_nrows} rows, expected {nrows}"),
      ));
    }

    for _ in 0..nrows {
      data.push(reader.read_f64::<BigEndian>()?);
    }
  }
  Ok(Matrix::from_column_slice(nrows, ncols, &data))
}

fn read_nonnegative_i32(reader: &mut impl ReadBytesExt, field: &'static str) -> io::Result<usize> {
  let value = reader.read_i32::<BigEndian>()?;
  usize::try_from(value).map_err(|_| {
    io::Error::new(
      io::ErrorKind::InvalidData,
      format!("{field} must be non-negative, got {value}"),
    )
  })
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
  let result = try_petsc_ghiep_with_which(lhs, rhs, neigen_values, which)
    .unwrap_or_else(|error| panic!("{error}"));
  (result.eigenvalues, result.eigenvectors)
}

/// Solve a partial generalized Hermitian eigenproblem without panicking.
///
/// The helper solves `lhs * v = lambda * rhs * v`. Both matrices must be
/// finite, square, and defined on the same coefficient space. The unique
/// workspace is intentionally retained and returned in diagnostics so failed
/// or suspicious external runs can be inspected; this function never removes
/// a pre-existing workspace and never falls back to a dense solver.
pub fn try_petsc_ghiep_with_which(
  lhs: &CsrMatrix,
  rhs: &CsrMatrix,
  neigen_values: usize,
  which: GhiepWhich,
) -> Result<PetscGeneralizedEigenpairs, PetscError> {
  validate_generalized_eigenproblem(lhs, rhs, neigen_values)?;
  let solver_path = petsc_solver_path();
  let binary = solver_path.join("ghiep.out");
  let workspace = try_create_petsc_workspace(&solver_path)?;

  let input_path = workspace.join("in");
  write_generalized_eigenproblem_input(lhs, &input_path.join("A.bin"), &binary, &workspace)?;
  write_generalized_eigenproblem_input(rhs, &input_path.join("B.bin"), &binary, &workspace)?;

  let args = ghiep_args(which, neigen_values);

  let status = try_run_petsc_command(&binary, &workspace, &args)?;
  if !status.success() {
    return Err(PetscError::HelperExit {
      binary,
      workspace,
      status,
    });
  }

  let eigenvalue_path = workspace.join("out/eigenvals.bin");
  let eigenvector_path = workspace.join("out/eigenvecs.bin");
  let eigenvalues = read_eigenvalues_with_context(&eigenvalue_path, &binary, &workspace)?;
  let eigenvectors = read_eigenvectors_with_context(&eigenvector_path, &binary, &workspace)?;
  validate_generalized_eigenproblem_output(
    &eigenvalues,
    &eigenvectors,
    lhs.nrows(),
    &binary,
    &workspace,
  )?;

  let converged_rank = eigenvalues.len();
  Ok(PetscGeneralizedEigenpairs {
    eigenvalues,
    eigenvectors,
    diagnostics: PetscGeneralizedEigenDiagnostics {
      dimension: lhs.nrows(),
      requested_rank: neigen_values,
      converged_rank,
      lhs_nonzeros: lhs.nnz(),
      rhs_nonzeros: rhs.nnz(),
      workspace,
      binary,
    },
  })
}

/// Typed smallest-eigenvalue compatibility companion to [`petsc_ghiep`].
pub fn try_petsc_ghiep(
  lhs: &CsrMatrix,
  rhs: &CsrMatrix,
  neigen_values: usize,
) -> Result<PetscGeneralizedEigenpairs, PetscError> {
  try_petsc_ghiep_with_which(lhs, rhs, neigen_values, GhiepWhich::Smallest)
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

fn validate_generalized_eigenproblem(
  lhs: &CsrMatrix,
  rhs: &CsrMatrix,
  requested_rank: usize,
) -> Result<(), PetscError> {
  if lhs.nrows() != lhs.ncols() || rhs.nrows() != rhs.ncols() || lhs.nrows() != rhs.nrows() {
    return Err(PetscError::InvalidGeneralizedEigenproblemDimensions {
      lhs_rows: lhs.nrows(),
      lhs_cols: lhs.ncols(),
      rhs_rows: rhs.nrows(),
      rhs_cols: rhs.ncols(),
    });
  }
  let dimension = lhs.nrows();
  if requested_rank == 0 || requested_rank > dimension {
    return Err(PetscError::InvalidRequestedEigenpairs {
      requested: requested_rank,
      dimension,
    });
  }
  validate_finite_operator(lhs, "lhs")?;
  validate_finite_operator(rhs, "rhs")?;
  Ok(())
}

fn validate_finite_operator(operator: &CsrMatrix, name: &'static str) -> Result<(), PetscError> {
  if let Some((row, col, value)) = operator
    .triplet_iter()
    .find(|(_, _, value)| !value.is_finite())
  {
    return Err(PetscError::NonFiniteInputEntry {
      operator: name,
      row,
      col,
      value: *value,
    });
  }
  Ok(())
}

fn write_generalized_eigenproblem_input(
  matrix: &CsrMatrix,
  path: &Path,
  binary: &Path,
  workspace: &Path,
) -> Result<(), PetscError> {
  let filename = path.to_string_lossy();
  petsc_write_matrix(matrix, &filename).map_err(|source| PetscError::InputWrite {
    binary: binary.to_path_buf(),
    workspace: workspace.to_path_buf(),
    path: path.to_path_buf(),
    source,
  })
}

fn read_eigenvalues_with_context(
  path: &Path,
  binary: &Path,
  workspace: &Path,
) -> Result<Vector, PetscError> {
  let filename = path.to_string_lossy();
  petsc_read_eigenvals(&filename).map_err(|source| PetscError::OutputRead {
    binary: binary.to_path_buf(),
    workspace: workspace.to_path_buf(),
    path: path.to_path_buf(),
    source,
  })
}

fn read_eigenvectors_with_context(
  path: &Path,
  binary: &Path,
  workspace: &Path,
) -> Result<Matrix, PetscError> {
  let filename = path.to_string_lossy();
  petsc_read_eigenvecs(&filename).map_err(|source| PetscError::OutputRead {
    binary: binary.to_path_buf(),
    workspace: workspace.to_path_buf(),
    path: path.to_path_buf(),
    source,
  })
}

fn validate_generalized_eigenproblem_output(
  eigenvalues: &Vector,
  eigenvectors: &Matrix,
  dimension: usize,
  binary: &Path,
  workspace: &Path,
) -> Result<(), PetscError> {
  let invalid = if eigenvalues.is_empty() {
    Some("the helper returned no converged eigenpairs".to_string())
  } else if eigenvectors.nrows() != dimension {
    Some(format!(
      "the eigenvector matrix has {} rows, expected {dimension}",
      eigenvectors.nrows()
    ))
  } else if eigenvectors.ncols() != eigenvalues.len() {
    Some(format!(
      "the helper returned {} eigenvalues and {} eigenvector columns",
      eigenvalues.len(),
      eigenvectors.ncols()
    ))
  } else {
    None
  };
  if let Some(detail) = invalid {
    return Err(PetscError::InvalidHelperOutput {
      binary: binary.to_path_buf(),
      workspace: workspace.to_path_buf(),
      detail,
    });
  }
  Ok(())
}

/// Operators defining the reduced generalized Hodge eigenproblem.
///
/// Keeping these matrices together makes the discrete equation explicit and
/// prevents callers from accidentally swapping same-shaped operator blocks.
#[derive(Debug, Clone, Copy)]
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
  let result = try_petsc_ghep_reduced_with_which(operators, neigen_values, which, mass_solve)
    .unwrap_or_else(|error| panic!("{error}"));
  let k = neigen_values;
  (
    result.eigenvalues.rows(0, k).into_owned(),
    result.sigma_eigenvectors.columns(0, k).into_owned(),
    result.u_eigenvectors.columns(0, k).into_owned(),
  )
}

/// Solve the sparse mixed/reduced generalized Hodge eigenproblem without panicking.
///
/// The helper applies
/// `K = L + D M_(k-1)^(-1) C` as a shell operator and solves
/// `K u = lambda M_k u`.  Every block is written sparsely; this Rust boundary
/// never materializes the marginal `K`, never falls back to a dense solver,
/// and retains the unique helper workspace in its diagnostics.
pub fn try_petsc_ghep_reduced_with_which(
  operators: GhepReducedOperators<'_>,
  neigen_values: usize,
  which: GhiepWhich,
  mass_solve: GhiepReducedSolve,
) -> Result<PetscReducedGeneralizedEigenpairs, PetscError> {
  validate_reduced_generalized_eigenproblem(operators, neigen_values)?;
  let solver_path = petsc_solver_path();
  let binary = solver_path.join("ghep_reduced.out");
  let workspace = try_create_petsc_workspace(&solver_path)?;

  let in_path = workspace.join("in");
  for (name, operator) in [
    ("L.bin", operators.l),
    ("D.bin", operators.d),
    ("C.bin", operators.c),
    ("Mkm1.bin", operators.mkm1),
    ("Mk.bin", operators.mk),
  ] {
    write_generalized_eigenproblem_input(operator, &in_path.join(name), &binary, &workspace)?;
  }

  let args = ghep_reduced_args(which, neigen_values, mass_solve);

  let status = try_run_petsc_command(&binary, &workspace, &args)?;
  if !status.success() {
    return Err(PetscError::HelperExit {
      binary,
      workspace,
      status,
    });
  }

  let eigenvalues =
    read_eigenvalues_with_context(&workspace.join("out/eigenvals.bin"), &binary, &workspace)?;
  let sigma_eigenvectors = read_eigenvectors_with_context(
    &workspace.join("out/eigenvecs_sigma.bin"),
    &binary,
    &workspace,
  )?;
  let u_eigenvectors =
    read_eigenvectors_with_context(&workspace.join("out/eigenvecs_u.bin"), &binary, &workspace)?;
  validate_reduced_generalized_eigenproblem_output(
    &eigenvalues,
    &sigma_eigenvectors,
    &u_eigenvectors,
    operators.mkm1.nrows(),
    operators.mk.nrows(),
    neigen_values,
    &binary,
    &workspace,
  )?;

  let converged_rank = eigenvalues.len();
  let input_nonzeros = [
    operators.l.nnz(),
    operators.d.nnz(),
    operators.c.nnz(),
    operators.mkm1.nnz(),
    operators.mk.nnz(),
  ]
  .into_iter()
  .try_fold(0usize, usize::checked_add)
  .ok_or(PetscError::ReducedEigenproblemResourceOverflow {
    quantity: "input nonzero count",
  })?;
  Ok(PetscReducedGeneralizedEigenpairs {
    eigenvalues,
    sigma_eigenvectors,
    u_eigenvectors,
    diagnostics: PetscReducedGeneralizedEigenDiagnostics {
      sigma_dimension: operators.mkm1.nrows(),
      u_dimension: operators.mk.nrows(),
      requested_rank: neigen_values,
      converged_rank,
      l_nonzeros: operators.l.nnz(),
      d_nonzeros: operators.d.nnz(),
      c_nonzeros: operators.c.nnz(),
      mkm1_nonzeros: operators.mkm1.nnz(),
      mk_nonzeros: operators.mk.nnz(),
      input_nonzeros,
      which,
      mass_solve,
      workspace,
      binary,
    },
  })
}

fn validate_reduced_generalized_eigenproblem(
  operators: GhepReducedOperators<'_>,
  requested_rank: usize,
) -> Result<(), PetscError> {
  let sigma_dimension = operators.mkm1.nrows();
  let u_dimension = operators.mk.nrows();
  let valid = sigma_dimension > 0
    && u_dimension > 0
    && operators.mkm1.ncols() == sigma_dimension
    && operators.mk.ncols() == u_dimension
    && operators.l.nrows() == u_dimension
    && operators.l.ncols() == u_dimension
    && operators.d.nrows() == u_dimension
    && operators.d.ncols() == sigma_dimension
    && operators.c.nrows() == sigma_dimension
    && operators.c.ncols() == u_dimension;
  if !valid {
    return Err(
      PetscError::InvalidReducedGeneralizedEigenproblemDimensions {
        l_rows: operators.l.nrows(),
        l_cols: operators.l.ncols(),
        d_rows: operators.d.nrows(),
        d_cols: operators.d.ncols(),
        c_rows: operators.c.nrows(),
        c_cols: operators.c.ncols(),
        mkm1_rows: operators.mkm1.nrows(),
        mkm1_cols: operators.mkm1.ncols(),
        mk_rows: operators.mk.nrows(),
        mk_cols: operators.mk.ncols(),
      },
    );
  }
  if requested_rank == 0 || requested_rank > u_dimension {
    return Err(PetscError::InvalidRequestedEigenpairs {
      requested: requested_rank,
      dimension: u_dimension,
    });
  }
  for (name, operator) in [
    ("L", operators.l),
    ("D", operators.d),
    ("C", operators.c),
    ("Mkm1", operators.mkm1),
    ("Mk", operators.mk),
  ] {
    validate_finite_operator(operator, name)?;
  }
  Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_reduced_generalized_eigenproblem_output(
  eigenvalues: &Vector,
  sigma_eigenvectors: &Matrix,
  u_eigenvectors: &Matrix,
  sigma_dimension: usize,
  u_dimension: usize,
  requested_rank: usize,
  binary: &Path,
  workspace: &Path,
) -> Result<(), PetscError> {
  let converged = eigenvalues.len();
  if sigma_eigenvectors.nrows() != sigma_dimension
    || u_eigenvectors.nrows() != u_dimension
    || sigma_eigenvectors.ncols() != converged
    || u_eigenvectors.ncols() != converged
  {
    return Err(PetscError::InvalidReducedHelperOutputShape {
      binary: binary.to_path_buf(),
      workspace: workspace.to_path_buf(),
      eigenvalue_count: converged,
      sigma_rows: sigma_eigenvectors.nrows(),
      sigma_cols: sigma_eigenvectors.ncols(),
      u_rows: u_eigenvectors.nrows(),
      u_cols: u_eigenvectors.ncols(),
      expected_sigma_rows: sigma_dimension,
      expected_u_rows: u_dimension,
    });
  }
  if converged < requested_rank {
    return Err(PetscError::InsufficientReducedEigenpairConvergence {
      binary: binary.to_path_buf(),
      workspace: workspace.to_path_buf(),
      requested: requested_rank,
      converged,
    });
  }
  if let Some((index, value)) = eigenvalues
    .iter()
    .copied()
    .enumerate()
    .find(|(_, value)| !value.is_finite())
  {
    return Err(PetscError::InvalidHelperOutput {
      binary: binary.to_path_buf(),
      workspace: workspace.to_path_buf(),
      detail: format!("eigenvalue {index} is non-finite: {value}"),
    });
  }
  for (name, matrix) in [
    ("sigma eigenvector", sigma_eigenvectors),
    ("u eigenvector", u_eigenvectors),
  ] {
    if let Some((index, value)) = matrix
      .iter()
      .copied()
      .enumerate()
      .find(|(_, value)| !value.is_finite())
    {
      return Err(PetscError::InvalidHelperOutput {
        binary: binary.to_path_buf(),
        workspace: workspace.to_path_buf(),
        detail: format!("{name} entry {index} is non-finite: {value}"),
      });
    }
  }
  Ok(())
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
  use crate::linalg::nalgebra::CooMatrix;

  fn diagonal(values: &[f64]) -> CsrMatrix {
    let mut coo = CooMatrix::new(values.len(), values.len());
    for (index, value) in values.iter().copied().enumerate() {
      coo.push(index, index, value);
    }
    CsrMatrix::from(&coo)
  }

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
  fn typed_ghiep_rejects_nonmatching_dimensions_before_launch() {
    let lhs = diagonal(&[1.0, 2.0]);
    let mut rhs_coo = CooMatrix::new(2, 3);
    rhs_coo.push(0, 0, 1.0);
    rhs_coo.push(1, 1, 1.0);
    let rhs = CsrMatrix::from(&rhs_coo);
    let error = try_petsc_ghiep(&lhs, &rhs, 1).unwrap_err();
    assert!(matches!(
      error,
      PetscError::InvalidGeneralizedEigenproblemDimensions { .. }
    ));
  }

  #[test]
  fn typed_ghiep_rejects_zero_and_oversized_rank_before_launch() {
    let lhs = diagonal(&[1.0, 2.0]);
    let rhs = diagonal(&[1.0, 1.0]);
    for requested in [0, 3] {
      let error = try_petsc_ghiep(&lhs, &rhs, requested).unwrap_err();
      assert!(matches!(
        error,
        PetscError::InvalidRequestedEigenpairs {
          requested: actual,
          dimension: 2
        } if actual == requested
      ));
    }
  }

  #[test]
  fn typed_ghiep_rejects_nonfinite_input_before_launch() {
    let lhs = diagonal(&[1.0, f64::NAN]);
    let rhs = diagonal(&[1.0, 1.0]);
    let error = try_petsc_ghiep(&lhs, &rhs, 1).unwrap_err();
    assert!(matches!(
      error,
      PetscError::NonFiniteInputEntry {
        operator: "lhs",
        row: 1,
        col: 1,
        value
      } if value.is_nan()
    ));
  }

  #[test]
  fn typed_helper_launch_error_contains_binary_and_workspace() {
    let binary = PathBuf::from("/definitely/missing/ghiep.out");
    let workspace = PathBuf::from("/definitely/missing/workspace");
    let error = try_run_petsc_command(&binary, &workspace, &[] as &[&str]).unwrap_err();
    match error {
      PetscError::HelperLaunch {
        binary: actual_binary,
        workspace: actual_workspace,
        ..
      } => {
        assert_eq!(actual_binary, binary);
        assert_eq!(actual_workspace, workspace);
      }
      other => panic!("expected helper launch error, got {other}"),
    }
  }

  #[test]
  #[ignore = "requires built PETSc/SLEPc ghiep helper"]
  fn typed_ghiep_solves_bounded_diagonal_problem_with_helper() {
    let lhs = diagonal(&[1.0, 2.0, 4.0]);
    let rhs = diagonal(&[1.0, 1.0, 1.0]);
    let result = try_petsc_ghiep(&lhs, &rhs, 2).expect("bounded helper solve should succeed");
    assert_eq!(result.diagnostics.dimension, 3);
    assert_eq!(result.diagnostics.requested_rank, 2);
    assert!(result.diagnostics.converged_rank >= 2);
    assert_eq!(result.eigenvectors.nrows(), 3);
    assert_eq!(
      result.eigenvectors.ncols(),
      result.diagnostics.converged_rank
    );
    let mut eigenvalues = result.eigenvalues.iter().copied().collect::<Vec<_>>();
    eigenvalues.sort_by(f64::total_cmp);
    assert!((eigenvalues[0] - 1.0).abs() <= 1.0e-8);
    assert!((eigenvalues[1] - 2.0).abs() <= 1.0e-8);
    eprintln!(
      "retained PETSc helper workspace for inspection: {:?}",
      result.diagnostics.workspace
    );
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
  fn typed_ghep_reduced_rejects_incompatible_blocks_and_rank_before_launch() {
    let l = diagonal(&[1.0, 2.0, 4.0]);
    let mk = diagonal(&[1.0, 1.0, 1.0]);
    let mkm1 = diagonal(&[1.0, 1.0]);
    let d = CsrMatrix::from(&CooMatrix::new(3, 2));
    let invalid_c = CsrMatrix::from(&CooMatrix::new(3, 2));
    let error = try_petsc_ghep_reduced_with_which(
      GhepReducedOperators {
        l: &l,
        d: &d,
        c: &invalid_c,
        mkm1: &mkm1,
        mk: &mk,
      },
      1,
      GhiepWhich::Smallest,
      GhiepReducedSolve::Direct,
    )
    .unwrap_err();
    assert!(matches!(
      error,
      PetscError::InvalidReducedGeneralizedEigenproblemDimensions { .. }
    ));

    let c = CsrMatrix::from(&CooMatrix::new(2, 3));
    for requested in [0, 4] {
      let error = try_petsc_ghep_reduced_with_which(
        GhepReducedOperators {
          l: &l,
          d: &d,
          c: &c,
          mkm1: &mkm1,
          mk: &mk,
        },
        requested,
        GhiepWhich::Smallest,
        GhiepReducedSolve::Direct,
      )
      .unwrap_err();
      assert!(matches!(
        error,
        PetscError::InvalidRequestedEigenpairs {
          requested: actual,
          dimension: 3
        } if actual == requested
      ));
    }
  }

  #[test]
  fn typed_ghep_reduced_distinguishes_shape_and_convergence_failures() {
    let binary = Path::new("/test/ghep_reduced.out");
    let workspace = Path::new("/test/workspace");
    let shape_error = validate_reduced_generalized_eigenproblem_output(
      &Vector::from_vec(vec![1.0, 2.0]),
      &Matrix::zeros(2, 1),
      &Matrix::zeros(3, 2),
      2,
      3,
      2,
      binary,
      workspace,
    )
    .unwrap_err();
    assert!(matches!(
      shape_error,
      PetscError::InvalidReducedHelperOutputShape { .. }
    ));

    let convergence_error = validate_reduced_generalized_eigenproblem_output(
      &Vector::from_vec(vec![1.0]),
      &Matrix::zeros(2, 1),
      &Matrix::zeros(3, 1),
      2,
      3,
      2,
      binary,
      workspace,
    )
    .unwrap_err();
    assert!(matches!(
      convergence_error,
      PetscError::InsufficientReducedEigenpairConvergence {
        requested: 2,
        converged: 1,
        ..
      }
    ));
  }

  #[test]
  #[ignore = "requires built PETSc/SLEPc ghep_reduced helper"]
  fn typed_ghep_reduced_solves_bounded_sparse_problem_with_helper() {
    let l = diagonal(&[1.0, 2.0, 4.0]);
    let mk = diagonal(&[1.0, 1.0, 1.0]);
    let mkm1 = diagonal(&[1.0, 1.0]);
    let d = CsrMatrix::from(&CooMatrix::new(3, 2));
    let c = CsrMatrix::from(&CooMatrix::new(2, 3));
    let result = try_petsc_ghep_reduced_with_which(
      GhepReducedOperators {
        l: &l,
        d: &d,
        c: &c,
        mkm1: &mkm1,
        mk: &mk,
      },
      2,
      GhiepWhich::Smallest,
      GhiepReducedSolve::Direct,
    )
    .expect("bounded mixed helper solve should succeed");
    assert_eq!(result.diagnostics.sigma_dimension, 2);
    assert_eq!(result.diagnostics.u_dimension, 3);
    assert_eq!(result.diagnostics.requested_rank, 2);
    assert!(result.diagnostics.converged_rank >= 2);
    assert_eq!(result.diagnostics.which, GhiepWhich::Smallest);
    assert_eq!(result.diagnostics.mass_solve, GhiepReducedSolve::Direct);
    assert_eq!(result.diagnostics.input_nonzeros, 8);
    assert_eq!(result.sigma_eigenvectors.nrows(), 2);
    assert_eq!(result.u_eigenvectors.nrows(), 3);
    let mut eigenvalues = result.eigenvalues.iter().copied().collect::<Vec<_>>();
    eigenvalues.sort_by(f64::total_cmp);
    assert!((eigenvalues[0] - 1.0).abs() <= 1.0e-8);
    assert!((eigenvalues[1] - 2.0).abs() <= 1.0e-8);
    eprintln!(
      "retained reduced PETSc helper workspace for inspection: {:?}",
      result.diagnostics.workspace
    );
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
