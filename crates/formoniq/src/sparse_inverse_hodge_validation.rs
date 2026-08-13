use std::{collections::HashSet, f64::consts::PI, fmt::Write as _, fs};

use {
  crate::{
    assemble::{
      self, assemble_barycentric_dual_1form_sparse_inverse_galmat, assemble_galvec,
      assemble_whitney_projected_sparse_inverse_galmat, BarycentricDualSparseInverseConfig,
    },
    fe::{fe_l2_error, l2_norm},
    operators::SourceElVec,
    problems::hodge_laplace::MixedGalmats,
  },
  common::{
    linalg::{
      faer::{FaerCholesky, FaerLu},
      nalgebra::{CooMatrix, CsrMatrix, Matrix, Vector},
    },
    util::algebraic_convergence_rate,
  },
  ddf::cochain::Cochain,
  exterior::{field::DiffFormClosure, ExteriorElement},
  manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::coord::{mesh::MeshCoords, CoordRef},
    topology::{complex::Complex, handle::KSimplexIdx},
  },
};

const DIM: usize = 3;
const GRADE: usize = 2;
const KAPPA: f64 = 0.0;
const SOLUTION_WEIGHTS: [f64; 3] = [1.0, 2.0, -1.0];

#[derive(Debug, Clone)]
pub struct SparseInverseHodgeValidationConfig {
  pub output_dir: std::path::PathBuf,
  pub max_refinement: u32,
}

impl Default for SparseInverseHodgeValidationConfig {
  fn default() -> Self {
    Self {
      output_dir: "out/examples/hodge_laplace_2form_nc1_lumping".into(),
      max_refinement: 4,
    }
  }
}

fn two_form(coeffs: [f64; 3]) -> ExteriorElement {
  ExteriorElement::new(Vector::from_column_slice(&coeffs), DIM, GRADE)
}

fn three_form(coeff: f64) -> ExteriorElement {
  ExteriorElement::new(Vector::from_column_slice(&[coeff]), DIM, GRADE + 1)
}

fn scalar_factor(p: CoordRef) -> f64 {
  let (sx, sy, sz) = (p[0].sin(), p[1].sin(), p[2].sin());
  sx.powi(2) * sy.powi(2) * sz.powi(2)
}

fn scalar_factor_hodge_laplacian(p: CoordRef) -> f64 {
  let (sx, sy, sz) = (p[0].sin(), p[1].sin(), p[2].sin());
  let (sx2, sy2, sz2) = (sx.powi(2), sy.powi(2), sz.powi(2));
  -2.0
    * ((2.0 * p[0]).cos() * sy2 * sz2
      + (2.0 * p[1]).cos() * sx2 * sz2
      + (2.0 * p[2]).cos() * sx2 * sy2)
}

fn exact_solution_coeffs(p: CoordRef) -> [f64; 3] {
  let scale = scalar_factor(p);
  SOLUTION_WEIGHTS.map(|weight| weight * scale)
}

fn exact_shifted_source_coeffs(p: CoordRef, kappa: f64) -> [f64; 3] {
  let scale = scalar_factor_hodge_laplacian(p) + kappa * kappa * scalar_factor(p);
  SOLUTION_WEIGHTS.map(|weight| weight * scale)
}

fn exact_dif_coeff(p: CoordRef) -> f64 {
  let (sx, sy, sz) = (p[0].sin(), p[1].sin(), p[2].sin());
  let (sx2, sy2, sz2) = (sx.powi(2), sy.powi(2), sz.powi(2));
  let dx = (2.0 * p[0]).sin() * sy2 * sz2;
  let dy = sx2 * (2.0 * p[1]).sin() * sz2;
  let dz = sx2 * sy2 * (2.0 * p[2]).sin();
  SOLUTION_WEIGHTS[2] * dx - SOLUTION_WEIGHTS[1] * dy + SOLUTION_WEIGHTS[0] * dz
}

fn scale_sparse(mat: &CsrMatrix, scale: f64) -> CsrMatrix {
  let mut coo = CooMatrix::new(mat.nrows(), mat.ncols());
  if scale == 0.0 {
    return CsrMatrix::from(&coo);
  }

  for (row, col, value) in mat.triplet_iter() {
    coo.push(row, col, scale * *value);
  }
  CsrMatrix::from(&coo)
}

fn add_sparse(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
  assert_eq!(a.nrows(), b.nrows());
  assert_eq!(a.ncols(), b.ncols());

  let mut coo = CooMatrix::from(a);
  for (row, col, value) in b.triplet_iter() {
    coo.push(row, col, *value);
  }
  CsrMatrix::from(&coo)
}

fn add_shifted_block(
  system: &CsrMatrix,
  block: &CsrMatrix,
  offset: usize,
  scale: f64,
) -> CsrMatrix {
  let mut coo = CooMatrix::from(system);
  if scale != 0.0 {
    for (row, col, value) in block.triplet_iter() {
      coo.push(offset + row, offset + col, scale * *value);
    }
  }
  CsrMatrix::from(&coo)
}

fn sorted_dofs(dofs: &HashSet<usize>) -> Vec<usize> {
  let mut sorted = dofs.iter().copied().collect::<Vec<_>>();
  sorted.sort_unstable();
  sorted
}

fn expand_reduced_solution(
  reduced: &Vector,
  total_len: usize,
  strongly_enforced: &HashSet<usize>,
) -> Vector {
  let mut full = Vector::zeros(total_len);
  let mut ifree = 0;

  for idof in 0..total_len {
    if strongly_enforced.contains(&idof) {
      continue;
    }
    full[idof] = reduced[ifree];
    ifree += 1;
  }

  full
}

fn boundary_dofs(topology: &Complex, grade: usize) -> HashSet<usize> {
  topology
    .boundary_subcomplex_simplices(grade)
    .into_iter()
    .map(|simplex| simplex.kidx)
    .collect()
}

fn reduced_square_csr(galmat: &assemble::GalMat, drop_dofs: &HashSet<usize>) -> CsrMatrix {
  let mut reduced = galmat.clone();
  assemble::drop_dofs_galmat(drop_dofs, &mut reduced);
  CsrMatrix::from(&reduced)
}

fn reduced_rectangular_csr(
  galmat: &assemble::GalMat,
  drop_rows: &HashSet<usize>,
  drop_cols: &HashSet<usize>,
) -> CsrMatrix {
  let mut reduced = galmat.clone();
  assemble::drop_dofs_rectangular_galmat(drop_rows, drop_cols, &mut reduced);
  CsrMatrix::from(&reduced)
}

fn row_sum_lumped_inverse(mass: &CsrMatrix) -> CsrMatrix {
  assert_eq!(mass.nrows(), mass.ncols());

  let mut row_sums = vec![0.0; mass.nrows()];
  for (row, _col, value) in mass.triplet_iter() {
    row_sums[row] += *value;
  }

  let mut diagonal = CooMatrix::new(mass.nrows(), mass.ncols());
  for (irow, row_sum) in row_sums.into_iter().enumerate() {
    if row_sum.abs() > 1e-12 {
      diagonal.push(irow, irow, 1.0 / row_sum);
    }
  }

  CsrMatrix::from(&diagonal)
}

fn assemble_reduced_schur_operator(
  galmats: &MixedGalmats,
  sigma_bc: &HashSet<usize>,
  u_bc: &HashSet<usize>,
  sigma_inverse: &CsrMatrix,
  kappa: f64,
) -> CsrMatrix {
  let mass_u = reduced_square_csr(galmats.mass_u(), u_bc);
  let codifdif_u = reduced_square_csr(galmats.codifdif_u(), u_bc);
  let dif_sigma = reduced_rectangular_csr(galmats.dif_sigma(), u_bc, sigma_bc);
  let codif_u = reduced_rectangular_csr(galmats.codif_u(), sigma_bc, u_bc);

  let shift = scale_sparse(&mass_u, kappa * kappa);
  let mut operator = add_sparse(&codifdif_u, &shift);

  let schur_mid = &dif_sigma * sigma_inverse;
  let schur = schur_mid * &codif_u;
  operator = add_sparse(&operator, &schur);

  operator
}

fn assemble_reduced_projected_nc1_operator(
  topology: &Complex,
  metric: &manifold::geometry::metric::mesh::MeshLengths,
  galmats: &MixedGalmats,
  sigma_bc: &HashSet<usize>,
  u_bc: &HashSet<usize>,
  kappa: f64,
) -> CsrMatrix {
  let mut sigma_inverse = assemble_whitney_projected_sparse_inverse_galmat(topology, metric);
  assemble::drop_dofs_galmat(sigma_bc, &mut sigma_inverse);
  let sigma_inverse = CsrMatrix::from(&sigma_inverse);
  assemble_reduced_schur_operator(galmats, sigma_bc, u_bc, &sigma_inverse, kappa)
}

fn assemble_reduced_barycentric_dual_operator(
  topology: &Complex,
  coords: &MeshCoords,
  galmats: &MixedGalmats,
  sigma_bc: &HashSet<usize>,
  u_bc: &HashSet<usize>,
  kappa: f64,
) -> Result<CsrMatrix, String> {
  let mut sigma_inverse = assemble_barycentric_dual_1form_sparse_inverse_galmat(
    topology,
    coords,
    BarycentricDualSparseInverseConfig::default(),
  )?;
  assemble::drop_dofs_galmat(sigma_bc, &mut sigma_inverse);
  let sigma_inverse = CsrMatrix::from(&sigma_inverse);
  Ok(assemble_reduced_schur_operator(
    galmats,
    sigma_bc,
    u_bc,
    &sigma_inverse,
    kappa,
  ))
}

fn assemble_reduced_rowsum_lumped_operator(
  galmats: &MixedGalmats,
  sigma_bc: &HashSet<usize>,
  u_bc: &HashSet<usize>,
  kappa: f64,
) -> CsrMatrix {
  let reduced_mass_sigma = reduced_square_csr(galmats.mass_sigma(), sigma_bc);
  let sigma_inverse = row_sum_lumped_inverse(&reduced_mass_sigma);
  assemble_reduced_schur_operator(galmats, sigma_bc, u_bc, &sigma_inverse, kappa)
}

fn solve_reduced_mixed_u(
  galmats: &MixedGalmats,
  rhs_u: &Vector,
  sigma_bc: &HashSet<usize>,
  u_bc: &HashSet<usize>,
  kappa: f64,
) -> Vector {
  let sigma_predicate = |idof: KSimplexIdx| sigma_bc.contains(&idof);
  let u_predicate = |idof: KSimplexIdx| u_bc.contains(&idof);
  let zero_data = |_idof: KSimplexIdx| 0.0;

  let mut sigma_rhs = Vector::zeros(galmats.sigma_len());
  let mut u_rhs = rhs_u.clone();
  let harmonics = Matrix::zeros(galmats.free_u_len(&u_predicate), 0);

  let (system, rhs) = galmats.mixed_hodge_laplacian_with_strong_bc_via_elimination(
    &sigma_predicate,
    &zero_data,
    &u_predicate,
    &zero_data,
    &mut sigma_rhs,
    &mut u_rhs,
    &harmonics,
  );

  let mut reduced_mass_u = galmats.mass_u().clone();
  assemble::drop_dofs_galmat(u_bc, &mut reduced_mass_u);
  let reduced_mass_u = CsrMatrix::from(&reduced_mass_u);

  let sigma_len_reduced = galmats.free_sigma_len(&sigma_predicate);
  let system = add_shifted_block(&system, &reduced_mass_u, sigma_len_reduced, kappa * kappa);

  let solution = FaerLu::new(system).solve(&rhs);
  let reduced_u = solution
    .rows(sigma_len_reduced, galmats.free_u_len(&u_predicate))
    .into_owned();

  expand_reduced_solution(&reduced_u, galmats.u_len(), u_bc)
}

#[derive(Debug, Clone, Copy)]
pub struct SparseInverseHodgeConvergenceRow {
  pub refine: u32,
  pub nboxes_per_dim: usize,
  pub mixed_l2: f64,
  pub mixed_l2_rate: f64,
  pub mixed_hd: f64,
  pub mixed_hd_rate: f64,
  pub nc1_l2: f64,
  pub nc1_l2_rate: f64,
  pub nc1_hd: f64,
  pub nc1_hd_rate: f64,
  pub rowsum_l2: f64,
  pub rowsum_l2_rate: f64,
  pub rowsum_hd: f64,
  pub rowsum_hd_rate: f64,
  pub bary_l2: f64,
  pub bary_l2_rate: f64,
  pub bary_hd: f64,
  pub bary_hd_rate: f64,
  pub mixed_nc1_l2: f64,
  pub mixed_rowsum_l2: f64,
  pub mixed_bary_l2: f64,
}

fn last_rate(errors: &[f64], current: f64) -> f64 {
  errors
    .last()
    .map(|&previous| algebraic_convergence_rate(current, previous))
    .unwrap_or(f64::INFINITY)
}

pub fn run_sparse_inverse_hodge_validation(
  config: &SparseInverseHodgeValidationConfig,
) -> Result<Vec<SparseInverseHodgeConvergenceRow>, Box<dyn std::error::Error>> {
  let _ = fs::remove_dir_all(&config.output_dir);
  fs::create_dir_all(&config.output_dir)?;

  let solution_exact =
    DiffFormClosure::new(Box::new(|p| two_form(exact_solution_coeffs(p))), DIM, GRADE);
  let source_exact = DiffFormClosure::new(
    Box::new(|p| two_form(exact_shifted_source_coeffs(p, KAPPA))),
    DIM,
    GRADE,
  );
  let dif_solution_exact =
    DiffFormClosure::new(Box::new(|p| three_form(exact_dif_coeff(p))), DIM, GRADE + 1);

  let mut rows = Vec::new();
  let mut mixed_l2_errors = Vec::new();
  let mut mixed_hd_errors = Vec::new();
  let mut nc1_l2_errors = Vec::new();
  let mut nc1_hd_errors = Vec::new();
  let mut rowsum_l2_errors = Vec::new();
  let mut rowsum_hd_errors = Vec::new();
  let mut bary_l2_errors = Vec::new();
  let mut bary_hd_errors = Vec::new();

  println!("2-form Hodge-Laplacian convergence in 3d");
  println!("max refinement = {}", config.max_refinement);
  println!("kappa = {KAPPA:.3}");
  println!("homogeneous strong boundary conditions are enforced on edges and faces");
  println!(
    "| {:>2} | {:>5} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>10} | {:>10} |",
    "k",
    "n",
    "mixed L2",
    "rate",
    "mixed Hd",
    "rate",
    "nc1 L2",
    "rate",
    "nc1 Hd",
    "rate",
    "rsum L2",
    "rate",
    "rsum Hd",
    "rate",
    "bd L2",
    "rate",
    "bd Hd",
    "rate",
    "mix-nc1",
    "mix-rsum",
    "mix-bd",
  );

  for refine in 0..=config.max_refinement {
    let nboxes_per_dim = 2usize.pow(refine);
    let box_mesh = CartesianMeshInfo::new_unit_scaled(DIM, nboxes_per_dim, PI);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let sigma_bc = boundary_dofs(&topology, GRADE - 1);
    let u_bc = boundary_dofs(&topology, GRADE);

    let source_galvec = assemble_galvec(
      &topology,
      &metric,
      SourceElVec::new(&source_exact, &coords, None),
    );
    let mut reduced_rhs = source_galvec.clone();
    let reduced_u_drop = sorted_dofs(&u_bc);
    assemble::drop_dofs_galvec(&reduced_u_drop, &mut reduced_rhs);

    let galmats = MixedGalmats::compute(&topology, &metric, GRADE);
    let mixed_solution = Cochain::new(
      GRADE,
      solve_reduced_mixed_u(&galmats, &source_galvec, &sigma_bc, &u_bc, KAPPA),
    );

    let nc1_operator = assemble_reduced_projected_nc1_operator(
      &topology, &metric, &galmats, &sigma_bc, &u_bc, KAPPA,
    );
    let nc1_reduced = FaerCholesky::new(nc1_operator).solve(&reduced_rhs);
    let nc1_solution = Cochain::new(
      GRADE,
      expand_reduced_solution(&nc1_reduced, galmats.u_len(), &u_bc),
    );

    let rowsum_operator =
      assemble_reduced_rowsum_lumped_operator(&galmats, &sigma_bc, &u_bc, KAPPA);
    let rowsum_reduced = FaerCholesky::new(rowsum_operator).solve(&reduced_rhs);
    let rowsum_solution = Cochain::new(
      GRADE,
      expand_reduced_solution(&rowsum_reduced, galmats.u_len(), &u_bc),
    );

    let bary_operator = assemble_reduced_barycentric_dual_operator(
      &topology, &coords, &galmats, &sigma_bc, &u_bc, KAPPA,
    )
    .map_err(std::io::Error::other)?;
    let bary_reduced = FaerCholesky::new(bary_operator).solve(&reduced_rhs);
    let bary_solution = Cochain::new(
      GRADE,
      expand_reduced_solution(&bary_reduced, galmats.u_len(), &u_bc),
    );

    let mixed_l2 = fe_l2_error(&mixed_solution, &solution_exact, &topology, &coords);
    let mixed_hd = fe_l2_error(
      &mixed_solution.dif(&topology),
      &dif_solution_exact,
      &topology,
      &coords,
    );
    let nc1_l2 = fe_l2_error(&nc1_solution, &solution_exact, &topology, &coords);
    let nc1_hd = fe_l2_error(
      &nc1_solution.dif(&topology),
      &dif_solution_exact,
      &topology,
      &coords,
    );
    let rowsum_l2 = fe_l2_error(&rowsum_solution, &solution_exact, &topology, &coords);
    let rowsum_hd = fe_l2_error(
      &rowsum_solution.dif(&topology),
      &dif_solution_exact,
      &topology,
      &coords,
    );
    let bary_l2 = fe_l2_error(&bary_solution, &solution_exact, &topology, &coords);
    let bary_hd = fe_l2_error(
      &bary_solution.dif(&topology),
      &dif_solution_exact,
      &topology,
      &coords,
    );
    let mixed_nc1_l2 = l2_norm(
      &(mixed_solution.clone() - nc1_solution.clone()),
      &topology,
      &metric,
    );
    let mixed_rowsum_l2 = l2_norm(
      &(mixed_solution.clone() - rowsum_solution.clone()),
      &topology,
      &metric,
    );
    let mixed_bary_l2 = l2_norm(
      &(mixed_solution.clone() - bary_solution.clone()),
      &topology,
      &metric,
    );

    let row = SparseInverseHodgeConvergenceRow {
      refine,
      nboxes_per_dim,
      mixed_l2,
      mixed_l2_rate: last_rate(&mixed_l2_errors, mixed_l2),
      mixed_hd,
      mixed_hd_rate: last_rate(&mixed_hd_errors, mixed_hd),
      nc1_l2,
      nc1_l2_rate: last_rate(&nc1_l2_errors, nc1_l2),
      nc1_hd,
      nc1_hd_rate: last_rate(&nc1_hd_errors, nc1_hd),
      rowsum_l2,
      rowsum_l2_rate: last_rate(&rowsum_l2_errors, rowsum_l2),
      rowsum_hd,
      rowsum_hd_rate: last_rate(&rowsum_hd_errors, rowsum_hd),
      bary_l2,
      bary_l2_rate: last_rate(&bary_l2_errors, bary_l2),
      bary_hd,
      bary_hd_rate: last_rate(&bary_hd_errors, bary_hd),
      mixed_nc1_l2,
      mixed_rowsum_l2,
      mixed_bary_l2,
    };

    println!(
      "| {:>2} | {:>5} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>10.2e} | {:>10.2e} |",
      row.refine,
      row.nboxes_per_dim,
      row.mixed_l2,
      row.mixed_l2_rate,
      row.mixed_hd,
      row.mixed_hd_rate,
      row.nc1_l2,
      row.nc1_l2_rate,
      row.nc1_hd,
      row.nc1_hd_rate,
      row.rowsum_l2,
      row.rowsum_l2_rate,
      row.rowsum_hd,
      row.rowsum_hd_rate,
      row.bary_l2,
      row.bary_l2_rate,
      row.bary_hd,
      row.bary_hd_rate,
      row.mixed_nc1_l2,
      row.mixed_rowsum_l2,
      row.mixed_bary_l2,
    );

    mixed_l2_errors.push(mixed_l2);
    mixed_hd_errors.push(mixed_hd);
    nc1_l2_errors.push(nc1_l2);
    nc1_hd_errors.push(nc1_hd);
    rowsum_l2_errors.push(rowsum_l2);
    rowsum_hd_errors.push(rowsum_hd);
    bary_l2_errors.push(bary_l2);
    bary_hd_errors.push(bary_hd);
    rows.push(row);
  }

  let mut summary = String::new();
  writeln!(summary, "2-form Hodge-Laplacian convergence in 3d").unwrap();
  writeln!(summary, "kappa = {KAPPA:.3}").unwrap();
  writeln!(
    summary,
    "Homogeneous strong boundary conditions are enforced on boundary edges and faces."
  )
  .unwrap();
  writeln!(
    summary,
    "The mixed solve uses sparse LU on the reduced system. The nc1 path uses a reduced projected-NC1 Schur operator. The rsum path uses a reduced Schur operator with row-sum lumping of the consistent (k-1)-mass. The bd path uses the FEEC barycentric-dual sparse inverse on the sigma 1-form mass. All operator paths use sparse Cholesky."
  )
  .unwrap();
  writeln!(
    summary,
    "Maximum refinement for this resolved run is {}.",
    config.max_refinement
  )
  .unwrap();
  writeln!(summary).unwrap();
  writeln!(
    summary,
    "| {:>2} | {:>5} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>6} | {:>10} | {:>10} | {:>10} |",
    "k",
    "n",
    "mixed L2",
    "rate",
    "mixed Hd",
    "rate",
    "nc1 L2",
    "rate",
    "nc1 Hd",
    "rate",
    "rsum L2",
    "rate",
    "rsum Hd",
    "rate",
    "bd L2",
    "rate",
    "bd Hd",
    "rate",
    "mix-nc1",
    "mix-rsum",
    "mix-bd",
  )
  .unwrap();

  for row in &rows {
    writeln!(
      summary,
      "| {:>2} | {:>5} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>6.2} | {:>10.2e} | {:>10.2e} | {:>10.2e} |",
      row.refine,
      row.nboxes_per_dim,
      row.mixed_l2,
      row.mixed_l2_rate,
      row.mixed_hd,
      row.mixed_hd_rate,
      row.nc1_l2,
      row.nc1_l2_rate,
      row.nc1_hd,
      row.nc1_hd_rate,
      row.rowsum_l2,
      row.rowsum_l2_rate,
      row.rowsum_hd,
      row.rowsum_hd_rate,
      row.bary_l2,
      row.bary_l2_rate,
      row.bary_hd,
      row.bary_hd_rate,
      row.mixed_nc1_l2,
      row.mixed_rowsum_l2,
      row.mixed_bary_l2,
    )
    .unwrap();
  }

  fs::write(config.output_dir.join("summary.txt"), summary)?;

  Ok(rows)
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_relative_eq;

  #[test]
  fn row_sum_lumped_inverse_matches_diagonal_inverse_of_row_sums() {
    let mut coo = CooMatrix::new(3, 3);
    coo.push(0, 0, 2.0);
    coo.push(0, 1, 1.0);
    coo.push(1, 0, 1.0);
    coo.push(1, 1, 3.0);
    coo.push(1, 2, 2.0);
    coo.push(2, 1, 2.0);
    coo.push(2, 2, 4.0);

    let inverse = row_sum_lumped_inverse(&CsrMatrix::from(&coo));
    let inverse_dense = Matrix::from(&inverse);

    assert_relative_eq!(inverse_dense[(0, 0)], 1.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(inverse_dense[(1, 1)], 1.0 / 6.0, epsilon = 1e-12);
    assert_relative_eq!(inverse_dense[(2, 2)], 1.0 / 6.0, epsilon = 1e-12);
    assert_relative_eq!(inverse_dense[(0, 1)], 0.0, epsilon = 1e-12);
    assert_relative_eq!(inverse_dense[(1, 2)], 0.0, epsilon = 1e-12);
  }

  #[test]
  fn barycentric_dual_operator_builds_positive_definite_reduced_system() {
    let box_mesh = CartesianMeshInfo::new_unit_scaled(DIM, 1, PI);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let sigma_bc = boundary_dofs(&topology, GRADE - 1);
    let u_bc = boundary_dofs(&topology, GRADE);
    let galmats = MixedGalmats::compute(&topology, &metric, GRADE);

    let operator = assemble_reduced_barycentric_dual_operator(
      &topology, &coords, &galmats, &sigma_bc, &u_bc, KAPPA,
    )
    .expect("barycentric-dual reduced Schur operator should assemble");

    assert_eq!(
      operator.nrows(),
      galmats.free_u_len(&|idof| u_bc.contains(&idof))
    );
    assert_eq!(operator.ncols(), operator.nrows());
    FaerCholesky::new(operator);
  }
}
