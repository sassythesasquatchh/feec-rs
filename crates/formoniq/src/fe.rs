use crate::{
  assemble::assemble_galmat,
  operators::{CodifDifElmat, HodgeMassElmat},
};

use {
  common::linalg::nalgebra::{quadratic_form_sparse, CsrMatrix},
  ddf::{cochain::Cochain, whitney::form::WhitneyForm},
  exterior::{field::ExteriorField, term::multi_gramian},
  manifold::{
    geometry::{
      coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexHandleExt},
      metric::mesh::MeshLengths,
    },
    topology::complex::Complex,
  },
};

/// Combine the coefficient and exterior-derivative errors into the FEEC
/// graph-norm error
///
/// `||u - u_h||_{H(d)} = (||u - u_h||² + ||d(u - u_h)||²)^(1/2)`.
///
/// Keeping this operation here ensures convergence studies report the same
/// mathematical norm instead of accidentally substituting its derivative
/// seminorm.
pub fn hd_error(l2_error: f64, derivative_l2_error: f64) -> f64 {
  l2_error.hypot(derivative_l2_error)
}

pub fn l2_norm(fe: &Cochain, topology: &Complex, geometry: &MeshLengths) -> f64 {
  let mass = assemble_galmat(
    topology,
    geometry,
    HodgeMassElmat::new(topology.dim(), fe.dim()),
  );
  let mass = CsrMatrix::from(&mass);
  quadratic_form_sparse(&mass, fe.coeffs()).sqrt()
}

pub fn hdif_norm(fe: &Cochain, topology: &Complex, geometry: &MeshLengths) -> f64 {
  let codifdif = assemble_galmat(
    topology,
    geometry,
    CodifDifElmat::new(topology.dim(), fe.dim),
  );
  let codifdif = CsrMatrix::from(&codifdif);
  quadratic_form_sparse(&codifdif, fe.coeffs()).sqrt()
}

pub fn fe_l2_error<E: ExteriorField>(
  fe_cochain: &Cochain,
  exact: &E,
  topology: &Complex,
  coords: &MeshCoords,
) -> f64 {
  let dim = topology.dim();
  let qr = SimplexQuadRule::order3(dim);
  let fe_whitney = WhitneyForm::new(fe_cochain.clone(), topology, coords);
  let mut error_sq = 0.0;
  for cell in topology.cells().handle_iter() {
    let cell_coords = cell.coord_simplex(coords);
    let exact_is_ambient = exact.dim_ambient() == cell_coords.dim_ambient();
    let exact_is_intrinsic = exact.dim_ambient() == cell_coords.dim_intrinsic()
      && cell_coords.dim_ambient() != cell_coords.dim_intrinsic();
    assert!(
      exact_is_ambient || exact_is_intrinsic,
      "Exact field ambient dimension {} is incompatible with cell dimensions ({}, {}).",
      exact.dim_ambient(),
      cell_coords.dim_intrinsic(),
      cell_coords.dim_ambient()
    );
    let inner = multi_gramian(&cell_coords.metric_tensor().inverse(), fe_cochain.dim());
    error_sq += qr.integrate_local(
      &|local| {
        let global = cell_coords.local2global(local);
        let exact_local = if exact_is_ambient {
          cell_coords.pullback_form(&exact.at_point(global.as_view()))
        } else {
          exact.at_point(global.as_view())
        };
        let discrete_local = cell_coords.pullback_form(&fe_whitney.eval_known_cell(cell, &global));
        inner.norm_sq((exact_local - discrete_local).coeffs())
      },
      cell_coords.vol(),
    );
  }
  error_sq.sqrt()
}

#[cfg(test)]
mod tests {
  use super::hd_error;

  #[test]
  fn hd_error_is_the_exterior_derivative_graph_norm() {
    assert_eq!(hd_error(3.0, 4.0), 5.0);
    assert_eq!(hd_error(2.0, 0.0), 2.0);
  }
}
