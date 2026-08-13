use common::{
  affine::AffineTransform,
  linalg::nalgebra::{Vector, VectorView},
};

use crate::{Dim, ExteriorElement, ExteriorGrade};

/// Type-erased differential-form field evaluated in ambient coordinates.
pub type DifferentialFormFunction = dyn Fn(VectorView<f64>) -> ExteriorElement + Sync;

pub trait ExteriorField {
  fn dim_ambient(&self) -> Dim;
  fn dim_intrinsic(&self) -> Dim;
  fn grade(&self) -> ExteriorGrade;
  fn at_point<'a>(&self, coord: impl Into<VectorView<'a>>) -> ExteriorElement;
}

// Trait aliases.
pub trait MultiVectorField: ExteriorField {}
impl<T: ExteriorField> MultiVectorField for T {}
pub trait MultiFormField: ExteriorField {}
impl<T: ExteriorField> MultiFormField for T {}
pub trait DifferentialMultiForm: MultiFormField {}
impl<T: MultiFormField> DifferentialMultiForm for T {}

pub struct EmbeddedDiffFormClosure {
  closure: Box<DifferentialFormFunction>,
  dim_ambient: Dim,
  dim_intrinsic: Dim,
  grade: ExteriorGrade,
}

impl EmbeddedDiffFormClosure {
  pub fn new(
    closure: Box<DifferentialFormFunction>,
    dim_ambient: Dim,
    dim_intrinsic: Dim,
    grade: ExteriorGrade,
  ) -> Self {
    assert!(grade <= dim_intrinsic);
    Self {
      closure,
      dim_ambient,
      dim_intrinsic,
      grade,
    }
  }

  pub fn ambient_scalar(
    f: impl Fn(VectorView<f64>) -> f64 + Sync + 'static,
    dim_ambient: Dim,
    dim_intrinsic: Dim,
  ) -> Self {
    let wrapper = move |x: VectorView<f64>| crate::ExteriorElement::scalar(f(x), dim_ambient);
    Self::new(Box::new(wrapper), dim_ambient, dim_intrinsic, 0)
  }

  pub fn ambient_one_form(
    f: impl Fn(VectorView<f64>) -> Vector + Sync + 'static,
    dim_ambient: Dim,
    dim_intrinsic: Dim,
  ) -> Self {
    Self::ambient_k_form(f, dim_ambient, dim_intrinsic, 1)
  }

  pub fn ambient_k_form(
    f: impl Fn(VectorView<f64>) -> Vector + Sync + 'static,
    dim_ambient: Dim,
    dim_intrinsic: Dim,
    grade: ExteriorGrade,
  ) -> Self {
    let wrapper = move |x: VectorView<f64>| crate::ExteriorElement::new(f(x), dim_ambient, grade);
    Self::new(Box::new(wrapper), dim_ambient, dim_intrinsic, grade)
  }
}

pub struct DiffFormClosure {
  embedded: EmbeddedDiffFormClosure,
}

impl DiffFormClosure {
  pub fn new(closure: Box<DifferentialFormFunction>, dim: Dim, grade: ExteriorGrade) -> Self {
    Self {
      embedded: EmbeddedDiffFormClosure::new(closure, dim, dim, grade),
    }
  }
}

// Convenience methods specifically for DiffFormClosure
impl DiffFormClosure {
  /// Create a scalar field (0-form).
  pub fn scalar(f: impl Fn(VectorView<f64>) -> f64 + Sync + 'static, dim: Dim) -> Self {
    Self {
      embedded: EmbeddedDiffFormClosure::ambient_scalar(f, dim, dim),
    }
  }
  /// Create a 1-form (covector field).
  pub fn one_form(f: impl Fn(VectorView<f64>) -> Vector + Sync + 'static, dim: Dim) -> Self {
    Self {
      embedded: EmbeddedDiffFormClosure::ambient_one_form(f, dim, dim),
    }
  }

  pub fn top_form(f: impl Fn(VectorView<f64>) -> f64 + Sync + 'static, dim: Dim) -> Self {
    let wrapper =
      move |x: VectorView<f64>| crate::ExteriorElement::new(na::dvector![f(x)], dim, dim);
    Self::new(Box::new(wrapper), dim, dim)
  }

  /// Create a constant scalar field.
  pub fn constant_scalar(value: f64, dim: Dim) -> Self {
    Self::scalar(move |_| value, dim)
  }
  /// Create a scalar field that extracts a specific coordinate component.
  pub fn coordinate_component(icomp: usize, dim: Dim) -> Self {
    assert!(icomp < dim, "Component index out of bounds");
    Self::scalar(move |x| x[icomp], dim)
  }
  /// Create a scalar field of the radial distance from a center point.
  pub fn radial_scalar(center: Vector, dim: Dim) -> Self {
    Self::scalar(move |x| (&center - x).norm(), dim)
  }
}
impl ExteriorField for EmbeddedDiffFormClosure {
  fn dim_ambient(&self) -> Dim {
    self.dim_ambient
  }
  fn dim_intrinsic(&self) -> Dim {
    self.dim_intrinsic
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn at_point<'a>(&self, coord: impl Into<VectorView<'a>>) -> ExteriorElement {
    let value = (self.closure)(coord.into());
    assert_eq!(value.dim(), self.dim_ambient);
    assert_eq!(value.grade(), self.grade);
    value
  }
}

impl ExteriorField for DiffFormClosure {
  fn dim_ambient(&self) -> Dim {
    self.embedded.dim_ambient()
  }
  fn dim_intrinsic(&self) -> Dim {
    self.embedded.dim_intrinsic()
  }
  fn grade(&self) -> ExteriorGrade {
    self.embedded.grade()
  }
  fn at_point<'a>(&self, coord: impl Into<VectorView<'a>>) -> ExteriorElement {
    self.embedded.at_point(coord)
  }
}

pub struct FormPullback<F: DifferentialMultiForm> {
  form: F,
  affine_transform: AffineTransform,
}
impl<F: DifferentialMultiForm> FormPullback<F> {
  pub fn new(form: F, affine_transform: AffineTransform) -> Self {
    Self {
      form,
      affine_transform,
    }
  }
}
impl<F: DifferentialMultiForm> ExteriorField for FormPullback<F> {
  fn dim_ambient(&self) -> Dim {
    self.affine_transform.dim_domain()
  }
  fn dim_intrinsic(&self) -> Dim {
    self.affine_transform.dim_domain()
  }
  fn grade(&self) -> ExteriorGrade {
    self.form.grade()
  }
  fn at_point<'a>(&self, local: impl Into<VectorView<'a>>) -> ExteriorElement {
    let local = local.into();
    let global = self.affine_transform.apply_forward(local);
    let form_ref = self.form.at_point(&global);
    let pushforward = &self.affine_transform.linear;
    form_ref.precompose_form(pushforward)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn embedded_field_preserves_dimensions() {
    let field = EmbeddedDiffFormClosure::ambient_one_form(|_p| na::dvector![1.0, 2.0, 3.0], 3, 2);

    let value = field.at_point(na::dvector![0.1, 0.2, 0.3].as_view());
    assert_eq!(field.dim_ambient(), 3);
    assert_eq!(field.dim_intrinsic(), 2);
    assert_eq!(value.dim(), 3);
    assert_eq!(value.grade(), 1);
  }

  #[test]
  fn top_form_constructor_returns_true_top_form() {
    let field = DiffFormClosure::top_form(|p| p.sum(), 2);
    let value = field.at_point(na::dvector![0.25, 0.75].as_view());
    assert_eq!(value.dim(), 2);
    assert_eq!(value.grade(), 2);
    assert_eq!(value.coeffs().len(), 1);
    assert_eq!(value.coeffs()[0], 1.0);
  }
}
