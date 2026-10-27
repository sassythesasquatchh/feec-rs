use std::ops::{AddAssign, Mul};

use super::{
  mesh::MeshCoords,
  simplex::{barycenter_local, SimplexCoords},
  CoordRef,
};
use crate::{
  topology::{complex::Complex, handle::SimplexHandle},
  Dim,
};

use common::linalg::nalgebra::{Matrix, Vector};

/// A quadrature rule defined on the reference simplex.
///
/// Can be used to integrate functions defined on the reference simplex.
/// Functions can be scalar or matrix valued.
#[derive(Debug, Clone)]
pub struct SimplexQuadRule {
  /// Points in local coordinates.
  points: na::DMatrix<f64>,
  /// Normalized weights that sum to 1.
  weights: na::DVector<f64>,
}
impl SimplexQuadRule {
  pub fn dim(&self) -> Dim {
    self.points.nrows()
  }
  pub fn npoints(&self) -> usize {
    self.points.ncols()
  }

  /// Uses a local coordinate function `f`.
  pub fn integrate_local<T, F>(&self, f: &F, vol: f64) -> T
  where
    F: Fn(CoordRef) -> T,
    T: AddAssign + Mul<f64, Output = T>,
  {
    let n = self.npoints();
    debug_assert!(n > 0, "quadrature rule has no points");
    let mut integral = f(self.points.column(0)) * self.weights[0];
    for i in 1..n {
      integral += f(self.points.column(i)) * self.weights[i];
    }
    integral * vol
  }

  /// Uses a global coordinate function `f`.
  pub fn integrate_coord<F>(&self, f: &F, coords: &SimplexCoords) -> f64
  where
    F: Fn(CoordRef) -> f64,
  {
    self.integrate_local(
      &|local_coord| f(coords.local2global(local_coord).as_view()),
      coords.vol(),
    )
  }

  /// Uses a global coordinate function `f`.
  pub fn integrate_mesh<F>(&self, f: &F, complex: &Complex, coords: &MeshCoords) -> f64
  where
    F: Fn(CoordRef, SimplexHandle) -> f64,
  {
    let mut integral = 0.0;
    for cell in complex.cells().handle_iter() {
      let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
      integral += self.integrate_coord(&|x| f(x, cell), &cell_coords);
    }
    integral
  }
}

impl SimplexQuadRule {
  pub fn dim0() -> Self {
    let points = Matrix::zeros(0, 1);
    let weights = na::dvector![1.0];
    Self { points, weights }
  }

  /// Integrates 1st order affine linear functions exactly.
  pub fn barycentric(dim: Dim) -> Self {
    let barycenter = barycenter_local(dim);
    let points = Matrix::from_columns(&[barycenter]);
    let weight = 1.0;
    let weights = Vector::from_element(1, weight);
    Self { points, weights }
  }

  pub fn vertices(dim: Dim) -> Self {
    let nvertices = dim + 1;
    let mut points = Matrix::zeros(dim, nvertices);
    for ivertex in 1..nvertices {
      points[(ivertex - 1, ivertex)] = 1.0;
    }
    let weight = (nvertices as f64).recip();
    let weights = Vector::from_element(nvertices, weight);
    Self { points, weights }
  }
}

impl SimplexQuadRule {
  /// Composite two-point Gauss--Legendre rule on the reference segment [0, 1].
  /// Each of the equally sized panels integrates cubic polynomials exactly.
  ///
  /// # Panics
  /// Panics when `panels` is zero.
  pub fn segment_gauss_legendre(panels: usize) -> Self {
    assert!(
      panels > 0,
      "the number of quadrature panels must be positive"
    );
    let points = Matrix::from_fn(1, 2 * panels, |_, i| {
      let node = if i % 2 == 0 { -1.0 } else { 1.0 } / 3.0_f64.sqrt();
      (i / 2) as f64 / panels as f64 + (1.0 + node) / (2 * panels) as f64
    });
    let weights = Vector::from_element(2 * panels, 1.0 / (2 * panels) as f64);
    Self { points, weights }
  }

  pub fn order3(dim: Dim) -> Self {
    match dim {
      0 => Self::dim0(),
      1 => Self::dim1_order3(),
      2 => Self::dim2_order3(),
      3 => Self::dim3_order3(),
      _ => unimplemented!("No order 3 Quadrature available for dim {dim}."),
    }
  }

  /// Simposons Rule
  pub fn dim1_order3() -> Self {
    let points = na::dmatrix![0.0, 0.5, 1.0];
    let weights = na::dvector![1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];
    Self { points, weights }
  }
  pub fn dim2_order3() -> Self {
    let points = na::dmatrix![
      1.0/3.0, 1.0/5.0, 3.0/5.0, 1.0/5.0;
      1.0/3.0, 1.0/5.0, 1.0/5.0, 3.0/5.0;
    ];
    let weights = na::dvector![-27.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0];
    Self { points, weights }
  }
  pub fn dim3_order3() -> Self {
    let a: f64 = 1.0 / 4.0; // centroid
    let b: f64 = 1.0 / 6.0;
    let c: f64 = 1.0 / 2.0;

    let points = na::dmatrix![
        a, b, c, b, b;
        a, b, b, c, b;
        a, b, b, b, c;
    ];
    let weights = na::dvector![-4.0 / 5.0, 9.0 / 20.0, 9.0 / 20.0, 9.0 / 20.0, 9.0 / 20.0];
    Self { points, weights }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::geometry::refsimp_vol;
  use approx::assert_abs_diff_eq;

  #[test]
  fn composite_segment_gauss_rule_is_cubic_exact_and_resolves_smooth_fields() {
    for panels in [1, 7, 32] {
      let rule = SimplexQuadRule::segment_gauss_legendre(panels);
      for power in 0..=3 {
        let value = rule.integrate_local(&|p: CoordRef| p[0].powi(power), 1.0);
        assert_abs_diff_eq!(value, 1.0 / (power + 1) as f64, epsilon = 1e-14);
      }
    }
    let exact = std::f64::consts::E - 1.0;
    let error = |panels| {
      (SimplexQuadRule::segment_gauss_legendre(panels)
        .integrate_local(&|p: CoordRef| p[0].exp(), 1.0)
        - exact)
        .abs()
    };
    assert!(error(32) < error(8));
    let gauss_error_bound = std::f64::consts::E / (4320.0 * 32.0_f64.powi(4));
    assert!(error(32) < gauss_error_bound);
  }

  #[test]
  #[should_panic(expected = "must be positive")]
  fn composite_segment_gauss_rule_rejects_zero_panels() {
    SimplexQuadRule::segment_gauss_legendre(0);
  }

  // Assume: type CoordRef<'a> = &'a na::DVectorSlice<'a, f64>;

  #[test]
  fn test_dim0() {
    let rule = SimplexQuadRule::dim0();
    let vol = refsimp_vol(0);
    let f_const = |_p: CoordRef| 1.0;
    assert_abs_diff_eq!(rule.integrate_local(&f_const, vol), vol, epsilon = 1e-12);
  }

  #[test]
  fn test_barycentric() {
    for dim in 1..=3 {
      let rule = SimplexQuadRule::barycentric(dim);
      let vol = refsimp_vol(dim);

      let f_const = |_p: CoordRef| 1.0;
      assert_abs_diff_eq!(rule.integrate_local(&f_const, vol), vol, epsilon = 1e-12);

      if dim > 0 {
        let f_linear = |p: CoordRef| p[0];
        let exact_linear = vol / (dim as f64 + 1.0);
        assert_abs_diff_eq!(
          rule.integrate_local(&f_linear, vol),
          exact_linear,
          epsilon = 1e-12
        );
      }
    }
  }

  #[test]
  fn test_order3_dim1() {
    let dim = 1;
    let rule = SimplexQuadRule::order3(dim);
    let vol = refsimp_vol(dim);

    assert_abs_diff_eq!(
      rule.integrate_local(&|p| p[0].powi(2), vol),
      1.0 / 3.0,
      epsilon = 1e-12
    );
    assert_abs_diff_eq!(
      rule.integrate_local(&|p| p[0].powi(3), vol),
      0.25,
      epsilon = 1e-12
    );
  }

  #[test]
  fn test_order3_dim2() {
    let dim = 2;
    let rule = SimplexQuadRule::order3(dim);
    let vol = refsimp_vol(dim);

    assert_abs_diff_eq!(
      rule.integrate_local(&|p| p[1].powi(2), vol),
      vol / 6.0,
      epsilon = 1e-12
    );
    assert_abs_diff_eq!(
      rule.integrate_local(&|p| p[0].powi(3), vol),
      vol / 10.0,
      epsilon = 1e-12
    );
    assert_abs_diff_eq!(
      rule.integrate_local(&|p| p[0].powi(2) * p[1], vol),
      vol / 30.0,
      epsilon = 1e-12
    );
  }

  #[test]
  fn test_order3_dim3() {
    let dim = 3;
    let rule = SimplexQuadRule::order3(dim);
    let vol = refsimp_vol(dim);
    assert_abs_diff_eq!(
      rule.integrate_local(&|p| p[0].powi(2), vol),
      vol / 10.0,
      epsilon = 1e-12
    );
    assert_abs_diff_eq!(
      rule.integrate_local(&|p| p[0] * p[1], vol),
      vol / 20.0,
      epsilon = 1e-12
    );
    assert_abs_diff_eq!(
      rule.integrate_local(&|p| p[0].powi(3), vol),
      vol / 20.0,
      epsilon = 1e-12
    );
  }
}
