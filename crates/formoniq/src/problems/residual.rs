//! Native FEEC contract for assembled nonlinear residuals and Jacobians.

use common::linalg::nalgebra::{CsrMatrix, Vector};

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualEvaluation {
  pub residual: Vector,
  pub jacobian: CsrMatrix,
}

impl ResidualEvaluation {
  pub fn validate(&self, residual_dimension: usize, state_dimension: usize) -> Result<(), String> {
    if self.residual.len() != residual_dimension
      || self.jacobian.nrows() != residual_dimension
      || self.jacobian.ncols() != state_dimension
    {
      return Err("nonlinear residual/Jacobian dimensions do not match the model".to_string());
    }
    Ok(())
  }
}

pub trait ResidualModel {
  fn state_dimension(&self) -> usize;

  fn residual_dimension(&self) -> usize;

  fn residual(&self, state: &[f64]) -> Result<Vector, String> {
    self
      .residual_and_jacobian(state)
      .map(|evaluation| evaluation.residual)
  }

  fn residual_and_jacobian(&self, state: &[f64]) -> Result<ResidualEvaluation, String>;
}
