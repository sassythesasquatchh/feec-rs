//! Module for the Poisson Equation, the prototypical elliptic PDE.

use crate::{
  assemble::{self, GalMat, GalVec},
  operators::{self, DofCoeff, InnerProductWeightClosure},
};

use common::linalg::{
  faer::FaerCholesky,
  nalgebra::{CsrMatrix, Matrix, Vector},
  petsc::petsc_ghiep,
};
use ddf::cochain::Cochain;
use manifold::geometry::coord::mesh::MeshCoords;
use manifold::geometry::coord::quadrature::SimplexQuadRule;
use manifold::{
  geometry::metric::mesh::MeshLengths,
  topology::{complex::Complex, handle::KSimplexIdx},
};

/// Preassembled Laplace-Beltrami stiffness and mass Galerkin matrices.
pub struct LaplaceBeltramiGalmats {
  stiffness: GalMat,
  mass: GalMat,
}

impl LaplaceBeltramiGalmats {
  pub fn compute(topology: &Complex, geometry: &MeshLengths) -> Self {
    let stiffness = assemble_laplace_beltrami_galmat(topology, geometry, None, None, None);
    let mass = assemble_scalar_mass_galmat(topology, geometry, None, None, None);
    Self { stiffness, mass }
  }

  pub fn compute_weighted(
    topology: &Complex,
    geometry: &MeshLengths,
    mesh_coords: &MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &InnerProductWeightClosure,
  ) -> Self {
    let stiffness = assemble_laplace_beltrami_galmat(
      topology,
      geometry,
      Some(mesh_coords),
      qr.clone(),
      Some(weight),
    );
    let mass = assemble_scalar_mass_galmat(topology, geometry, Some(mesh_coords), qr, Some(weight));
    Self { stiffness, mass }
  }

  pub fn stiffness(&self) -> &GalMat {
    &self.stiffness
  }

  pub fn mass(&self) -> &GalMat {
    &self.mass
  }

  pub fn stiffness_csr(&self) -> CsrMatrix {
    CsrMatrix::from(&self.stiffness)
  }

  pub fn mass_csr(&self) -> CsrMatrix {
    CsrMatrix::from(&self.mass)
  }
}

pub fn solve_laplace_beltrami_source<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  source_galvec: GalVec,
  essential_boundary_data: F,
  essential_boundary_selector: Option<&dyn Fn(KSimplexIdx) -> bool>,
) -> Cochain
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  solve_laplace_beltrami_source_inner(
    topology,
    geometry,
    source_galvec,
    essential_boundary_data,
    essential_boundary_selector,
    None,
    None,
    None,
  )
}

// Weighted quadrature and essential trace data are independent low-level FEEC
// inputs. Laplace-Beltrami convergence tests exercise this compatibility API.
#[allow(clippy::too_many_arguments)]
pub fn solve_laplace_beltrami_source_weighted<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  source_galvec: GalVec,
  essential_boundary_data: F,
  essential_boundary_selector: Option<&dyn Fn(KSimplexIdx) -> bool>,
  mesh_coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> Cochain
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  solve_laplace_beltrami_source_inner(
    topology,
    geometry,
    source_galvec,
    essential_boundary_data,
    essential_boundary_selector,
    Some(mesh_coords),
    qr,
    Some(weight),
  )
}

fn matrix_to_cochain(matrix: Matrix) -> Vec<Cochain> {
  matrix
    .column_iter()
    .map(|c| Cochain::new(0, c.into_owned()))
    .collect()
}

pub fn solve_laplace_beltrami_evp(
  topology: &Complex,
  geometry: &MeshLengths,
  neigen_values: usize,
) -> (Vector, Vec<Cochain>) {
  let (eigenvals, eigenvecs) =
    solve_laplace_beltrami_evp_inner(topology, geometry, neigen_values, None, None, None);
  let eigenvecs = matrix_to_cochain(eigenvecs);
  (eigenvals, eigenvecs)
}

pub fn solve_laplace_beltrami_evp_weighted(
  topology: &Complex,
  geometry: &MeshLengths,
  neigen_values: usize,
  mesh_coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: &InnerProductWeightClosure,
) -> (Vector, Vec<Cochain>) {
  let (eigenvals, eigenvecs) = solve_laplace_beltrami_evp_inner(
    topology,
    geometry,
    neigen_values,
    Some(mesh_coords),
    qr,
    Some(weight),
  );

  let eigenvecs = matrix_to_cochain(eigenvecs);
  (eigenvals, eigenvecs)
}

pub fn solve_laplace_beltrami_evp_as_matrix(
  topology: &Complex,
  geometry: &MeshLengths,
  neigen_values: usize,
) -> (Vector, Matrix) {
  solve_laplace_beltrami_evp_inner(topology, geometry, neigen_values, None, None, None)
}

// Private kernel retains explicit weighted-assembly and essential-boundary inputs;
// source-solver tests cover both weighted and unweighted dispatch.
#[allow(clippy::too_many_arguments)]
fn solve_laplace_beltrami_source_inner<F>(
  topology: &Complex,
  geometry: &MeshLengths,
  mut source_galvec: GalVec,
  essential_boundary_data: F,
  essential_boundary_selector: Option<&dyn Fn(KSimplexIdx) -> bool>,
  mesh_coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
) -> Cochain
where
  F: Fn(KSimplexIdx) -> DofCoeff,
{
  let mut laplace_galmat =
    assemble_laplace_beltrami_galmat(topology, geometry, mesh_coords, qr, weight);
  assemble::enforce_dirichlet_bc_partial(
    topology,
    essential_boundary_data,
    &mut laplace_galmat,
    &mut source_galvec,
    essential_boundary_selector,
  );

  let laplace = CsrMatrix::from(&laplace_galmat);
  let sol = FaerCholesky::new(laplace).solve(&source_galvec);
  Cochain::new(0, sol)
}

/// Eigenvalue problem of Laplace-Beltrami operator.
fn solve_laplace_beltrami_evp_inner(
  topology: &Complex,
  geometry: &MeshLengths,
  neigen_values: usize,
  mesh_coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
) -> (Vector, Matrix) {
  let laplace_galmat =
    assemble_laplace_beltrami_galmat(topology, geometry, mesh_coords, qr.clone(), weight);
  let mass_galmat = assemble_scalar_mass_galmat(topology, geometry, mesh_coords, qr, weight);
  let (eigenvals, eigenvecs) = petsc_ghiep(
    &CsrMatrix::from(&laplace_galmat),
    &CsrMatrix::from(&mass_galmat),
    neigen_values,
  );

  (eigenvals, eigenvecs)
}

fn assemble_laplace_beltrami_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
  mesh_coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
) -> GalMat {
  let dim = topology.dim();
  if let (Some(mesh_coords), Some(weight)) = (mesh_coords, weight) {
    assemble::assemble_galmat_coord_aware(
      topology,
      geometry,
      operators::LaplaceBeltramiElmat::new_weighted(dim, mesh_coords, qr, weight),
    )
  } else {
    assemble::assemble_galmat(
      topology,
      geometry,
      operators::LaplaceBeltramiElmat::new(dim),
    )
  }
}

fn assemble_scalar_mass_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
  mesh_coords: Option<&MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&InnerProductWeightClosure>,
) -> GalMat {
  if let (Some(mesh_coords), Some(weight)) = (mesh_coords, weight) {
    assemble::assemble_galmat_coord_aware(
      topology,
      geometry,
      operators::ScalarMassElmat::new_weighted(mesh_coords, qr, weight),
    )
  } else {
    assemble::assemble_galmat(topology, geometry, operators::ScalarMassElmat::new())
  }
}
