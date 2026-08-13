use {
  common::{
    combo::{factorial, Sign},
    linalg::nalgebra::{Matrix, Vector},
  },
  ddf::{whitney::lsf::WhitneyLsf, ManifoldComplexExt},
  exterior::{
    exterior_dim, field::ExteriorField, list::ExteriorElementList, term::multi_gramian, Dim,
    ExteriorGrade,
  },
  manifold::{
    geometry::{
      coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexCoords, CoordRef},
      metric::simplex::SimplexLengths,
    },
    topology::{
      complex::Complex,
      simplex::{nsubsequence_simplices, standard_subsimps, Simplex},
    },
  },
  std::{
    ops::{AddAssign, Mul},
    sync::Arc,
  },
};

pub type DofIdx = usize;
pub type DofCoeff = f64;

pub type ElMat = Matrix;
pub trait ElMatProviderBase: Sync {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
}

pub trait ElMatProvider: ElMatProviderBase {
  fn eval(&self, geometry: &SimplexLengths) -> ElMat;
}

pub trait CoordAwareElMatProvider: ElMatProviderBase {
  fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> ElMat;
}

pub struct InnerProductWeightClosure<T = f64>
where
  T: AddAssign + Mul<f64, Output = T>,
{
  f: Arc<dyn Fn(CoordRef) -> T + Sync + Send>,
}

impl<T> InnerProductWeightClosure<T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  pub fn new<F>(f: F) -> Self
  where
    F: Fn(CoordRef) -> T + Sync + Send + 'static,
  {
    Self { f: Arc::new(f) }
  }

  pub fn apply(&self, x: CoordRef, coeffs: &Vector) -> Vector {
    (self.f)(x).apply(coeffs)
  }

  pub fn apply_batched(&self, x: CoordRef, coeffs: &Matrix) -> Matrix {
    (self.f)(x).apply_batched(coeffs)
  }
}

pub trait ApplyWeight: Send + Sync {
  fn apply_batched(&self, coeffs: &Matrix) -> Matrix;
  fn apply(&self, coeffs: &Vector) -> Vector;
}

impl ApplyWeight for f64 {
  fn apply_batched(&self, coeffs: &Matrix) -> Matrix {
    *self * coeffs
  }
  fn apply(&self, coeffs: &Vector) -> Vector {
    *self * coeffs
  }
}

impl ApplyWeight for Matrix {
  fn apply_batched(&self, coeffs: &Matrix) -> Matrix {
    self * coeffs
  }
  fn apply(&self, coeffs: &Vector) -> Vector {
    self * coeffs
  }
}

struct InnerProductWeight<T: ApplyWeight> {
  weight: T,
}

impl<T: ApplyWeight> InnerProductWeight<T> {
  pub fn apply_batched(&self, coeffs: &Matrix) -> Matrix {
    self.weight.apply_batched(coeffs)
  }
}

fn apply_optional_weight<T: ApplyWeight>(
  weight: Option<&InnerProductWeight<T>>,
  coeffs: &Matrix,
) -> Matrix {
  if let Some(w) = weight {
    w.apply_batched(coeffs)
  } else {
    coeffs.clone()
  }
}

fn averaged_cell_weight<T>(
  cell: &Simplex,
  coords: &MeshCoords,
  qr: &SimplexQuadRule,
  weight: &InnerProductWeightClosure<T>,
) -> InnerProductWeight<T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  let cell_coords = SimplexCoords::from_simplex_and_coords(cell, coords);

  // vol is set to 1 because we are not integrating over the simplex volume here,
  // we simply want the average value of the weight function over the simplex.
  let quadrature_result = qr.integrate_local(
    &|local: CoordRef| {
      let global = cell_coords.local2global(local);
      (weight.f)(global.as_view())
    },
    1.0,
  );

  InnerProductWeight {
    weight: quadrature_result,
  }
}

fn scalar_cell_weight(
  cell: &Simplex,
  weight_function: Option<&InnerProductWeightClosure<f64>>,
  coords: Option<&MeshCoords>,
  qr: Option<&SimplexQuadRule>,
) -> f64 {
  if let (Some(weight_function), Some(coords), Some(qr)) = (weight_function, coords, qr) {
    averaged_cell_weight(cell, coords, qr, weight_function).weight
  } else {
    1.0
  }
}

fn assert_supported_nc_grade(dim: Dim, grade: ExteriorGrade) {
  assert!(
    grade < dim,
    "NC support requires grade < intrinsic dimension, got grade {grade} on dimension {dim}."
  );
}

fn assert_supported_nc_exact_mass_dim(dim: Dim, grade: ExteriorGrade) {
  assert_supported_nc_grade(dim, grade);
  assert!(
    dim <= 3,
    "NC exact mass support currently requires an order-3 simplex quadrature rule, available only through dimension 3."
  );
}

fn assert_supported_nc1_dim(dim: Dim) {
  assert_supported_nc_grade(dim, 1);
}

fn assert_supported_nc2_dim(dim: Dim) {
  assert_supported_nc_grade(dim, 2);
}

fn assert_supported_nc1_exact_mass_dim(dim: Dim) {
  assert_supported_nc_exact_mass_dim(dim, 1);
}

fn assert_supported_nc2_exact_mass_dim(dim: Dim) {
  assert_supported_nc_exact_mass_dim(dim, 2);
}

fn nc_local_nsimps(dim: Dim, grade: ExteriorGrade) -> usize {
  assert_supported_nc_grade(dim, grade);
  nsubsequence_simplices(dim, grade)
}

fn nc_local_ndofs(dim: Dim, grade: ExteriorGrade) -> usize {
  (grade + 1) * nc_local_nsimps(dim, grade)
}

#[allow(dead_code)]
fn nc1_local_nedges(dim: Dim) -> usize {
  nc_local_nsimps(dim, 1)
}

#[allow(dead_code)]
fn nc1_local_ndofs(dim: Dim) -> usize {
  nc_local_ndofs(dim, 1)
}

fn nc_local_embedding_matrix(dim: Dim, grade: ExteriorGrade) -> Matrix {
  let nsimps = nc_local_nsimps(dim, grade);
  let slots = grade + 1;
  let mut embedding = Matrix::zeros(nc_local_ndofs(dim, grade), nsimps);
  for isimp in 0..nsimps {
    let group_start = slots * isimp;
    for slot in 0..slots {
      embedding[(group_start + slot, isimp)] = 1.0;
    }
  }
  embedding
}

fn nc_local_projection_matrix(dim: Dim, grade: ExteriorGrade) -> Matrix {
  let slots = grade + 1;
  (slots as f64).recip() * nc_local_embedding_matrix(dim, grade).transpose()
}

fn nc_local_dof_vertex(dim: Dim, grade: ExteriorGrade, local_dof: usize) -> usize {
  assert_supported_nc_grade(dim, grade);

  let slots = grade + 1;
  let local_simps: Vec<_> = standard_subsimps(dim, grade).collect();
  let simp = &local_simps[local_dof / slots];
  simp[local_dof % slots]
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn nc1_local_embedding_matrix(dim: Dim) -> Matrix {
  nc_local_embedding_matrix(dim, 1)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn nc1_local_projection_matrix(dim: Dim) -> Matrix {
  nc_local_projection_matrix(dim, 1)
}

#[cfg_attr(not(test), allow(dead_code))]
fn nc1_local_dof_vertex(dim: Dim, local_dof: usize) -> usize {
  nc_local_dof_vertex(dim, 1, local_dof)
}

#[allow(dead_code)]
fn nc2_local_nfaces(dim: Dim) -> usize {
  nc_local_nsimps(dim, 2)
}

#[allow(dead_code)]
fn nc2_local_ndofs(dim: Dim) -> usize {
  nc_local_ndofs(dim, 2)
}

#[allow(dead_code)]
pub(crate) fn nc2_local_embedding_matrix(dim: Dim) -> Matrix {
  nc_local_embedding_matrix(dim, 2)
}

#[allow(dead_code)]
pub(crate) fn nc2_local_projection_matrix(dim: Dim) -> Matrix {
  nc_local_projection_matrix(dim, 2)
}

fn nc_local_basis_coeffs(dim: Dim, grade: ExteriorGrade, coord: CoordRef) -> Matrix {
  assert_supported_nc_grade(dim, grade);

  let mut barys = Vec::with_capacity(dim + 1);
  barys.push(1.0 - coord.iter().sum::<f64>());
  barys.extend(coord.iter().copied());

  let local_simps: Vec<_> = standard_subsimps(dim, grade).collect();
  let slots = grade + 1;
  let mut coeffs = Matrix::zeros(exterior_dim(dim, grade), slots * local_simps.len());
  let scale = factorial(grade) as f64;

  for (isimp, simp) in local_simps.into_iter().enumerate() {
    let lsf = WhitneyLsf::standard(dim, simp.clone());
    for slot in 0..slots {
      let sign = Sign::from_parity(slot).as_f64();
      let column = scale * sign * barys[simp[slot]] * lsf.wedge_term(slot).into_coeffs();
      coeffs.set_column(slots * isimp + slot, &column);
    }
  }

  coeffs
}

fn nc2_local_basis_coeffs(dim: Dim, coord: CoordRef) -> Matrix {
  nc_local_basis_coeffs(dim, 2, coord)
}

fn nc1_local_basis_coeffs(dim: Dim, coord: CoordRef) -> Matrix {
  nc_local_basis_coeffs(dim, 1, coord)
}

/// Element matrix for the full first-order H(curl) space NC1 = P1 Lambda^1 on simplices.
pub struct Nc1MassElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  dim: Dim,
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<T>>,
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> Nc1MassElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    assert_supported_nc1_exact_mass_dim(dim);
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, Some(coords), Some(qr), Some(weight))
  }

  fn _new(
    dim: Dim,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    assert_supported_nc1_exact_mass_dim(dim);
    Self {
      dim,
      coords,
      qr,
      weight,
    }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let weight_to_apply = if let Some(weight) = &self.weight {
      let topology =
        topology.expect("Weighted Nc1MassElmat requires a cell (topology) to evaluate the weight.");
      let qr = self
        .qr
        .as_ref()
        .expect("Inner product weight provided, but no quadrature rule specified.");
      let coords = self
        .coords
        .as_ref()
        .expect("Inner product weight provided, but no mesh coordinates specified.");

      Some(averaged_cell_weight(topology, coords, qr, weight))
    } else {
      None
    };

    let inner = multi_gramian(&geometry.to_metric_tensor().inverse(), 1);
    let qr = SimplexQuadRule::order3(self.dim);
    qr.integrate_local(
      &|local: CoordRef| {
        let basis_coeffs = nc1_local_basis_coeffs(self.dim, local);
        inner.inner_mat(
          &apply_optional_weight(weight_to_apply.as_ref(), &basis_coeffs),
          &basis_coeffs,
        )
      },
      geometry.vol(),
    )
  }
}

impl<'a> Nc1MassElmat<'a, f64> {
  pub fn new(dim: Dim) -> Self {
    Self::_new(dim, None, None, None)
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> Nc1MassElmat<'a, T> {
  pub fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    debug_assert!(self.weight.is_none());
    self._eval(geometry, None)
  }

  pub fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> Matrix {
    self._eval(geometry, Some(cell))
  }
}

/// Element matrix for the mass-lumped vertex-split first-order space on k-forms.
pub struct NcLumpedMassElmat<'a> {
  dim: Dim,
  grade: ExteriorGrade,
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<f64>>,
}

impl<'a> NcLumpedMassElmat<'a> {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::_new(dim, grade, None, None, None)
  }

  pub fn new_weighted(
    dim: Dim,
    grade: ExteriorGrade,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<f64>,
  ) -> Self {
    assert_supported_nc_grade(dim, grade);
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, grade, Some(coords), Some(qr), Some(weight))
  }

  fn _new(
    dim: Dim,
    grade: ExteriorGrade,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<f64>>,
  ) -> Self {
    assert_supported_nc_grade(dim, grade);
    Self {
      dim,
      grade,
      coords,
      qr,
      weight,
    }
  }

  fn _eval(&self, geometry: &SimplexLengths, cell: Option<&Simplex>) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let cell_weight = if let Some(weight) = self.weight {
      let cell = cell.expect("Weighted NcLumpedMassElmat requires a cell (topology).");
      scalar_cell_weight(cell, Some(weight), self.coords, self.qr.as_ref())
    } else {
      1.0
    };

    let inner = multi_gramian(&geometry.to_metric_tensor().inverse(), self.grade);
    let qr = SimplexQuadRule::vertices(self.dim);
    cell_weight
      * qr.integrate_local(
        &|local: CoordRef| {
          let basis_coeffs = nc_local_basis_coeffs(self.dim, self.grade, local);
          inner.inner_mat(&basis_coeffs, &basis_coeffs)
        },
        geometry.vol(),
      )
  }

  pub fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    debug_assert!(self.weight.is_none());
    self._eval(geometry, None)
  }

  pub fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> Matrix {
    self._eval(geometry, Some(cell))
  }
}

/// Compatibility wrapper for the mass-lumped vertex-split first-order space on 1-forms.
pub struct Nc1LumpedMassElmat<'a> {
  dim: Dim,
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<f64>>,
}

/// Element matrix for the full first-order H(div)-analogue auxiliary 2-form space in 3D.
///
/// The local basis follows the projected BDM1/RT0 construction from the variational
/// Yee-like scheme, expressed in 2-form coefficients ordered as (dx^dy, dx^dz, dy^dz).
pub struct Nc2MassElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  dim: Dim,
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<T>>,
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> Nc2MassElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    assert_supported_nc2_exact_mass_dim(dim);
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, Some(coords), Some(qr), Some(weight))
  }

  fn _new(
    dim: Dim,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    assert_supported_nc2_exact_mass_dim(dim);
    Self {
      dim,
      coords,
      qr,
      weight,
    }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let weight_to_apply = if let Some(weight) = &self.weight {
      let topology =
        topology.expect("Weighted Nc2MassElmat requires a cell (topology) to evaluate the weight.");
      let qr = self
        .qr
        .as_ref()
        .expect("Inner product weight provided, but no quadrature rule specified.");
      let coords = self
        .coords
        .as_ref()
        .expect("Inner product weight provided, but no mesh coordinates specified.");

      Some(averaged_cell_weight(topology, coords, qr, weight))
    } else {
      None
    };

    let inner = multi_gramian(&geometry.to_metric_tensor().inverse(), 2);
    let qr = SimplexQuadRule::order3(self.dim);
    qr.integrate_local(
      &|local: CoordRef| {
        let basis_coeffs = nc2_local_basis_coeffs(self.dim, local);
        inner.inner_mat(
          &apply_optional_weight(weight_to_apply.as_ref(), &basis_coeffs),
          &basis_coeffs,
        )
      },
      geometry.vol(),
    )
  }
}

impl<'a> Nc2MassElmat<'a, f64> {
  pub fn new(dim: Dim) -> Self {
    Self::_new(dim, None, None, None)
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> Nc2MassElmat<'a, T> {
  pub fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    debug_assert!(self.weight.is_none());
    self._eval(geometry, None)
  }

  pub fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> Matrix {
    self._eval(geometry, Some(cell))
  }
}

/// Compatibility wrapper for the mass-lumped vertex-split first-order space on 2-forms.
pub struct Nc2LumpedMassElmat<'a> {
  dim: Dim,
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<f64>>,
}

impl<'a> Nc2LumpedMassElmat<'a> {
  pub fn new(dim: Dim) -> Self {
    Self::_new(dim, None, None, None)
  }

  pub fn new_weighted(
    dim: Dim,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<f64>,
  ) -> Self {
    assert_supported_nc2_dim(dim);
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, Some(coords), Some(qr), Some(weight))
  }

  fn _new(
    dim: Dim,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<f64>>,
  ) -> Self {
    assert_supported_nc2_dim(dim);
    Self {
      dim,
      coords,
      qr,
      weight,
    }
  }

  fn _eval(&self, geometry: &SimplexLengths, cell: Option<&Simplex>) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let cell_weight = if let Some(weight) = self.weight {
      let cell = cell.expect("Weighted Nc2LumpedMassElmat requires a cell (topology).");
      scalar_cell_weight(cell, Some(weight), self.coords, self.qr.as_ref())
    } else {
      1.0
    };

    let inner = multi_gramian(&geometry.to_metric_tensor().inverse(), 2);
    let qr = SimplexQuadRule::vertices(self.dim);
    cell_weight
      * qr.integrate_local(
        &|local: CoordRef| {
          let basis_coeffs = nc2_local_basis_coeffs(self.dim, local);
          inner.inner_mat(&basis_coeffs, &basis_coeffs)
        },
        geometry.vol(),
      )
  }

  pub fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    debug_assert!(self.weight.is_none());
    self._eval(geometry, None)
  }

  pub fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> Matrix {
    self._eval(geometry, Some(cell))
  }
}

impl<'a> Nc1LumpedMassElmat<'a> {
  pub fn new(dim: Dim) -> Self {
    Self::_new(dim, None, None, None)
  }

  pub fn new_weighted(
    dim: Dim,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<f64>,
  ) -> Self {
    assert_supported_nc1_dim(dim);
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, Some(coords), Some(qr), Some(weight))
  }

  fn _new(
    dim: Dim,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<f64>>,
  ) -> Self {
    assert_supported_nc1_dim(dim);
    Self {
      dim,
      coords,
      qr,
      weight,
    }
  }

  fn _eval(&self, geometry: &SimplexLengths, cell: Option<&Simplex>) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let cell_weight = if let Some(weight) = self.weight {
      let cell = cell.expect("Weighted Nc1LumpedMassElmat requires a cell (topology).");
      scalar_cell_weight(cell, Some(weight), self.coords, self.qr.as_ref())
    } else {
      1.0
    };

    let inner = multi_gramian(&geometry.to_metric_tensor().inverse(), 1);
    let qr = SimplexQuadRule::vertices(self.dim);
    cell_weight
      * qr.integrate_local(
        &|local: CoordRef| {
          let basis_coeffs = nc1_local_basis_coeffs(self.dim, local);
          inner.inner_mat(&basis_coeffs, &basis_coeffs)
        },
        geometry.vol(),
      )
  }

  pub fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    debug_assert!(self.weight.is_none());
    self._eval(geometry, None)
  }

  pub fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> Matrix {
    self._eval(geometry, Some(cell))
  }
}

/// Exact Element Matrix Provider for the Laplace-Beltrami operator.
///
/// $A = [(dif lambda_tau, dif lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct LaplaceBeltramiElmat<'a> {
  dim: Dim,
  ref_difbarys: Matrix,
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<f64>>,
}
impl<'a> LaplaceBeltramiElmat<'a> {
  pub fn new(dim: Dim) -> Self {
    let ref_difbarys = SimplexCoords::standard(dim).difbarys().transpose();
    Self {
      dim,
      ref_difbarys,
      coords: None,
      qr: None,
      weight: None,
    }
  }

  pub fn new_weighted(
    dim: Dim,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<f64>,
  ) -> Self {
    let ref_difbarys = SimplexCoords::standard(dim).difbarys().transpose();
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self {
      dim,
      ref_difbarys,
      coords: Some(coords),
      qr: Some(qr),
      weight: Some(weight),
    }
  }
}

impl<'a> ElMatProviderBase for LaplaceBeltramiElmat<'a> {
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
    0
  }
}
impl<'a> ElMatProvider for LaplaceBeltramiElmat<'a> {
  fn eval(&self, geometry: &SimplexLengths) -> ElMat {
    assert!(self.dim == geometry.dim());
    geometry.vol()
      * geometry
        .to_metric_tensor()
        .inverse()
        .norm_sq_mat(&self.ref_difbarys)
  }
}

impl<'a> CoordAwareElMatProvider for LaplaceBeltramiElmat<'a> {
  fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> ElMat {
    scalar_cell_weight(cell, self.weight, self.coords, self.qr.as_ref()) * self.eval(geometry)
  }
}

/// Exact Element Matrix Provider for scalar mass bilinear form.
pub struct ScalarMassElmat<'a> {
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<f64>>,
}
impl<'a> Default for ScalarMassElmat<'a> {
  fn default() -> Self {
    Self::new()
  }
}

impl<'a> ScalarMassElmat<'a> {
  pub fn new() -> Self {
    Self {
      coords: None,
      qr: None,
      weight: None,
    }
  }
}

impl<'a> ScalarMassElmat<'a> {
  pub fn new_weighted(
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<f64>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(coords.dim()));
    Self {
      coords: Some(coords),
      qr: Some(qr),
      weight: Some(weight),
    }
  }
}
impl<'a> ElMatProviderBase for ScalarMassElmat<'a> {
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
    0
  }
}
impl<'a> ElMatProvider for ScalarMassElmat<'a> {
  fn eval(&self, geometry: &SimplexLengths) -> ElMat {
    let ndofs = geometry.nvertices();
    let dim = geometry.dim();
    let v = geometry.vol() / ((dim + 1) * (dim + 2)) as f64;
    let mut elmat = Matrix::from_element(ndofs, ndofs, v);
    elmat.fill_diagonal(2.0 * v);
    elmat
  }
}

impl<'a> CoordAwareElMatProvider for ScalarMassElmat<'a> {
  fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> ElMat {
    scalar_cell_weight(cell, self.weight, self.coords, self.qr.as_ref()) * self.eval(geometry)
  }
}

/// Approximated Element Matrix Provider for scalar mass bilinear form,
/// obtained through trapezoidal quadrature rule.
pub struct ScalarLumpedMassElmat;
impl ElMatProviderBase for ScalarLumpedMassElmat {
  fn row_grade(&self) -> ExteriorGrade {
    0
  }
  fn col_grade(&self) -> ExteriorGrade {
    0
  }
}
impl ElMatProvider for ScalarLumpedMassElmat {
  fn eval(&self, geomery: &SimplexLengths) -> ElMat {
    let n = geomery.nvertices();
    let v = geomery.vol() / n as f64;
    Matrix::from_diagonal_element(n, n, v)
  }
}

/// Element Matrix for the weak Hodge star operator / the mass bilinear form.
///
/// $M = [inner(star lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))$
pub struct HodgeMassElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  dim: Dim,
  grade: ExteriorGrade,
  simplices: Vec<Simplex>,
  wedge_terms: Vec<ExteriorElementList>,
  coords: Option<&'a MeshCoords>,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<T>>,
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> HodgeMassElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    grade: ExteriorGrade,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, grade, Some(coords), Some(qr), Some(weight))
  }

  fn _new(
    dim: Dim,
    grade: ExteriorGrade,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    let simplices: Vec<_> = standard_subsimps(dim, grade).collect();
    let wedge_terms: Vec<ExteriorElementList> = simplices
      .iter()
      .cloned()
      .map(|simp| WhitneyLsf::standard(dim, simp).wedge_terms().collect())
      .collect();

    Self {
      dim,
      grade,
      simplices,
      wedge_terms,
      coords,
      qr,
      weight,
    }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let scalar_mass = ScalarMassElmat::new().eval(geometry);
    let mut elmat = Matrix::zeros(self.simplices.len(), self.simplices.len());

    let weight_to_apply = if let Some(weight) = &self.weight {
      let topology = topology
        .expect("Weighted HodgeMassElmat requires a cell (topology) to evaluate the weight.");
      let qr = self
        .qr
        .as_ref()
        .expect("Inner product weight provided, but no quadrature rule specified.");
      let coords = self
        .coords
        .as_ref()
        .expect("Inner product weight provided, but no mesh coordinates specified.");

      Some(averaged_cell_weight(topology, coords, qr, weight))
    } else {
      None
    };

    for (i, asimp) in self.simplices.iter().enumerate() {
      for (j, bsimp) in self.simplices.iter().enumerate() {
        let wedge_terms_a = &self.wedge_terms[i];
        let wedge_terms_b = &self.wedge_terms[j];
        let wedge_inners = multi_gramian(&geometry.to_metric_tensor().inverse(), self.grade)
          .inner_mat(
            &apply_optional_weight(weight_to_apply.as_ref(), wedge_terms_a.coeffs()),
            wedge_terms_b.coeffs(),
          );

        let nvertices = self.grade + 1;
        let mut sum = 0.0;
        for avertex in 0..nvertices {
          for bvertex in 0..nvertices {
            let sign = Sign::from_parity(avertex + bvertex);

            let inner = wedge_inners[(avertex, bvertex)];

            sum += sign.as_f64() * inner * scalar_mass[(asimp[avertex], bsimp[bvertex])];
          }
        }

        elmat[(i, j)] = sum;
      }
    }

    factorial(self.grade).pow(2) as f64 * elmat
  }
}

impl<'a> HodgeMassElmat<'a, f64> {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::_new(dim, grade, None, None, None)
  }
}

impl<'a, T> ElMatProviderBase for HodgeMassElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn row_grade(&self) -> ExteriorGrade {
    self.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.grade
  }
}

impl<'a, T> CoordAwareElMatProvider for HodgeMassElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn eval_with_coords(&self, geometry: &SimplexLengths, cell: &Simplex) -> ElMat {
    self._eval(geometry, Some(cell))
  }
}

impl<'a, T> ElMatProvider for HodgeMassElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn eval(&self, geometry: &SimplexLengths) -> ElMat {
    // This is only valid for the unweighted case
    debug_assert!(self.weight.is_none());
    self._eval(geometry, None)
  }
}
/// Element Matrix Provider for the weak mixed exterior derivative $(dif sigma, v)$.
///
/// $A = [inner(dif lambda_J, lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_, J in Delta_(k-1) (K))$
pub struct DifElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  mass: HodgeMassElmat<'a, T>,
  dif: Matrix,
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> DifElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    grade: ExteriorGrade,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, grade, Some(coords), Some(qr), Some(weight))
  }

  pub fn _new(
    dim: Dim,
    grade: ExteriorGrade,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    let mass = HodgeMassElmat::_new(dim, grade, coords, qr, weight);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade - 1);
    let dif = Matrix::from(&dif);
    Self { mass, dif }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    let mass = self.mass._eval(geometry, topology);
    mass * &self.dif
  }
}

impl<'a> DifElmat<'a, f64> {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::_new(dim, grade, None, None, None)
  }
}

impl<'a, T> ElMatProviderBase for DifElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> ElMatProvider for DifElmat<'a, T> {
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    // This is only valid for the unweighted case
    debug_assert!(self.mass.weight.is_none());
    self._eval(geometry, None)
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CoordAwareElMatProvider
  for DifElmat<'a, T>
{
  fn eval_with_coords(&self, geometry: &SimplexLengths, topology: &Simplex) -> Matrix {
    self._eval(geometry, Some(topology))
  }
}

/// Element Matrix Provider for the weak mixed codifferential $(u, dif tau)$.
///
/// $A = [inner(lambda_J, dif lambda_I)_(L^2 Lambda^k (K))]_(I in Delta_(k-1), J in Delta_k (K))$
pub struct CodifElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  mass: HodgeMassElmat<'a, T>,
  codif: Matrix,
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CodifElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    grade: ExteriorGrade,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, grade, Some(coords), Some(qr), Some(weight))
  }

  pub fn _new(
    dim: Dim,
    grade: ExteriorGrade,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    let mass = HodgeMassElmat::_new(dim, grade, coords, qr, weight);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade - 1);
    let dif = Matrix::from(&dif);
    let codif = dif.transpose();
    Self { mass, codif }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    let mass = self.mass._eval(geometry, topology);
    &self.codif * mass
  }
}

impl<'a> CodifElmat<'a, f64> {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::_new(dim, grade, None, None, None)
  }
}

impl<'a, T> ElMatProviderBase for CodifElmat<'a, T>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CoordAwareElMatProvider
  for CodifElmat<'a, T>
{
  fn eval_with_coords(&self, geometry: &SimplexLengths, topology: &Simplex) -> Matrix {
    self._eval(geometry, Some(topology))
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> ElMatProvider for CodifElmat<'a, T> {
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    // This is only valid for the unweighted case
    debug_assert!(self.mass.weight.is_none());
    self._eval(geometry, None)
  }
}

/// Element Matrix Provider for the $(dif u, dif v)$ bilinear form.
///
/// $A = [inner(dif lambda_J, dif lambda_I)_(L^2 Lambda^(k+1) (K))]_(I,J in Delta_k (K))$
pub struct CodifDifElmat<'a, T = f64>
where
  T: AddAssign + Mul<f64, Output = T> + ApplyWeight,
{
  mass: HodgeMassElmat<'a, T>,
  dif: Matrix,
  codif: Matrix,
}
impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CodifDifElmat<'a, T> {
  pub fn new_weighted(
    dim: Dim,
    grade: ExteriorGrade,
    coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(dim));
    Self::_new(dim, grade, Some(coords), Some(qr), Some(weight))
  }

  pub fn _new(
    dim: Dim,
    grade: ExteriorGrade,
    coords: Option<&'a MeshCoords>,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    let mass = HodgeMassElmat::_new(dim, grade + 1, coords, qr, weight);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade);
    let dif = Matrix::from(&dif);
    let codif = dif.transpose();

    Self { mass, dif, codif }
  }

  fn _eval(&self, geometry: &SimplexLengths, topology: Option<&Simplex>) -> Matrix {
    let mass = self.mass._eval(geometry, topology);
    &self.codif * mass * &self.dif
  }
}

impl<'a> CodifDifElmat<'a, f64> {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    Self::_new(dim, grade, None, None, None)
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> ElMatProviderBase
  for CodifDifElmat<'a, T>
{
  fn row_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
  fn col_grade(&self) -> ExteriorGrade {
    self.mass.grade - 1
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> CoordAwareElMatProvider
  for CodifDifElmat<'a, T>
{
  fn eval_with_coords(&self, geometry: &SimplexLengths, topology: &Simplex) -> Matrix {
    self._eval(geometry, Some(topology))
  }
}

impl<'a, T: AddAssign + Mul<f64, Output = T> + ApplyWeight> ElMatProvider for CodifDifElmat<'a, T> {
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    // This is only valid for the unweighted case
    debug_assert!(self.mass.weight.is_none());
    self._eval(geometry, None)
  }
}

pub type ElVec = Vector;
pub trait ElVecProvider: Sync {
  fn grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec;
}

pub struct SourceElVec<'a, F, T>
where
  F: ExteriorField,
  T: ApplyWeight + AddAssign + Mul<f64, Output = T>,
{
  source: &'a F,
  mesh_coords: &'a MeshCoords,
  qr: Option<SimplexQuadRule>,
  weight: Option<&'a InnerProductWeightClosure<T>>,
}
impl<'a, F, T> SourceElVec<'a, F, T>
where
  F: ExteriorField,
  T: ApplyWeight + AddAssign + Mul<f64, Output = T>,
{
  pub fn new_weighted(
    source: &'a F,
    mesh_coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: &'a InnerProductWeightClosure<T>,
  ) -> Self {
    Self::_new(source, mesh_coords, qr, Some(weight))
  }

  pub fn _new(
    source: &'a F,
    mesh_coords: &'a MeshCoords,
    qr: Option<SimplexQuadRule>,
    weight: Option<&'a InnerProductWeightClosure<T>>,
  ) -> Self {
    Self {
      source,
      mesh_coords,
      qr,
      weight,
    }
  }
}

impl<'a, F> SourceElVec<'a, F, f64>
where
  F: ExteriorField,
{
  pub fn new(source: &'a F, mesh_coords: &'a MeshCoords, qr: Option<SimplexQuadRule>) -> Self {
    Self::_new(source, mesh_coords, qr, None)
  }
}
impl<F, T> ElVecProvider for SourceElVec<'_, F, T>
where
  F: Sync + ExteriorField,
  T: ApplyWeight + AddAssign + Mul<f64, Output = T>,
{
  fn grade(&self) -> ExteriorGrade {
    self.source.grade()
  }
  fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec {
    let cell_coords = SimplexCoords::from_simplex_and_coords(topology, self.mesh_coords);
    let dim = cell_coords.dim_intrinsic();
    let source_is_ambient = self.source.dim_ambient() == cell_coords.dim_ambient();
    let source_is_intrinsic =
      self.source.dim_ambient() == dim && cell_coords.dim_ambient() != cell_coords.dim_intrinsic();
    assert!(
      source_is_ambient || source_is_intrinsic,
      "Source field ambient dimension {} is incompatible with cell dimensions ({}, {}).",
      self.source.dim_ambient(),
      cell_coords.dim_intrinsic(),
      cell_coords.dim_ambient()
    );
    let grade = self.grade();
    let qr = self
      .qr
      .clone()
      .unwrap_or_else(|| SimplexQuadRule::barycentric(dim));
    assert_eq!(qr.dim(), dim);
    let dof_simps: Vec<_> = standard_subsimps(dim, grade).collect();
    let whitneys: Vec<_> = dof_simps
      .iter()
      .cloned()
      .map(|dof_simp| WhitneyLsf::standard(dim, dof_simp))
      .collect();

    let inner = multi_gramian(&geometry.to_metric_tensor().inverse(), grade);

    let mut elvec = ElVec::zeros(whitneys.len());
    for (iwhitney, whitney) in whitneys.iter().enumerate() {
      let inner_pointwise = |local: CoordRef| {
        let global = cell_coords.local2global(local);
        let ref_source = if source_is_ambient {
          cell_coords.pullback_form(&self.source.at_point(&global))
        } else {
          self.source.at_point(global.as_view())
        };

        let source_coeffs = ref_source.coeffs();

        let weighted_owned = self
          .weight
          .map(|weight| weight.apply(global.as_view(), source_coeffs));

        let weighted_source = weighted_owned.as_ref().unwrap_or(source_coeffs);

        inner.inner(weighted_source, whitney.at_point(local).coeffs())
      };
      let value = qr.integrate_local(&inner_pointwise, geometry.vol());
      elvec[iwhitney] = value;
    }
    elvec
  }
}

#[cfg(test)]
mod test {
  use crate::operators::{
    nc1_local_dof_vertex, nc1_local_embedding_matrix, nc1_local_projection_matrix, CodifDifElmat,
    CodifElmat, CoordAwareElMatProvider, DifElmat, ElMatProvider, HodgeMassElmat,
    InnerProductWeightClosure, LaplaceBeltramiElmat, Matrix, Nc1LumpedMassElmat, Nc1MassElmat,
    ScalarMassElmat,
  };

  use approx::assert_relative_eq;
  use ddf::whitney::lsf::WhitneyLsf;
  use exterior::term::multi_gramian;
  use manifold::geometry::coord::mesh::MeshCoords;
  use manifold::topology::complex::Complex;
  use manifold::{geometry::metric::simplex::SimplexLengths, topology::simplex::standard_subsimps};

  #[test]
  fn codifdif0_is_laplace_beltrami() {
    let grade = 0;
    for dim in 1..=3 {
      let geo = SimplexLengths::standard(dim);
      let hodge_laplace = CodifDifElmat::new(dim, grade).eval(&geo);
      let laplace_beltrami = LaplaceBeltramiElmat::new(dim).eval(&geo);
      assert_relative_eq!(&hodge_laplace, &laplace_beltrami);
    }
  }

  #[test]
  fn hodge_mass0_is_scalar_mass() {
    let grade = 0;
    for dim in 0..=3 {
      let geo = SimplexLengths::standard(dim);
      let hodge_mass = HodgeMassElmat::new(dim, grade).eval(&geo);
      let scalar_mass = ScalarMassElmat::new().eval(&geo);
      assert_relative_eq!(&hodge_mass, &scalar_mass);
    }
  }

  #[test]
  fn hodge_mass_dim2_grade1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = HodgeMassElmat::new(dim, grade).eval(&geo);
    let expected = na::dmatrix![
      1./3.,1./6.,0.   ;
      1./6.,1./3.,0.   ;
      0.   ,0.   ,1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn dif_n2_k1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = DifElmat::new(dim, grade).eval(&geo);
    let expected = na::dmatrix![
      -1./2., 1./3.,1./6.;
      -1./2., 1./6.,1./3.;
       0.   ,-1./6.,1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn codif_n2_k1() {
    let dim = 2;
    let grade = 1;
    let geo = SimplexLengths::standard(dim);
    let computed = CodifElmat::new(dim, grade).eval(&geo);
    let expected = na::dmatrix![
      -1./2., -1./2., 0.   ;
       1./3.,  1./6.,-1./6.;
       1./6.,  1./3., 1./6.;
    ];
    assert_relative_eq!(&computed, &expected);
  }

  #[test]
  fn dif_dif_is_norm_of_difwhitneys() {
    for dim in 1..=3 {
      let geo = SimplexLengths::standard(dim);
      for grade in 0..dim {
        let difdif = CodifDifElmat::new(dim, grade).eval(&geo);

        let difwhitneys: Vec<_> = standard_subsimps(dim, grade)
          .map(|simp| WhitneyLsf::standard(dim, simp).dif())
          .collect();
        let mut inner = Matrix::zeros(difwhitneys.len(), difwhitneys.len());
        for (i, awhitney) in difwhitneys.iter().enumerate() {
          for (j, bwhitney) in difwhitneys.iter().enumerate() {
            inner[(i, j)] = multi_gramian(&geo.to_metric_tensor().inverse(), grade + 1)
              .inner(awhitney.coeffs(), bwhitney.coeffs());
          }
        }
        inner *= geo.vol();
        assert_relative_eq!(&difdif, &inner);
      }
    }
  }

  #[test]
  fn weighted_hodge_mass_scales_with_constant_scalar_weight() {
    const W: f64 = 2.5;
    const RTOL: f64 = 1e-12;

    for dim in 1..=3 {
      let geo = SimplexLengths::standard(dim);
      let topo = Complex::standard(dim);
      let cell = topo.cells().handle_iter().next().unwrap();

      for grade in 0..=dim {
        let unweighted = HodgeMassElmat::new(dim, grade).eval(&geo);

        let coords = MeshCoords::standard(dim);
        let weight = InnerProductWeightClosure::new(|_| W);

        let weighted = HodgeMassElmat::new_weighted(dim, grade, &coords, None, &weight)
          .eval_with_coords(&geo, &cell);

        let expected = W * &unweighted;
        assert_relative_eq!(&weighted, &expected, max_relative = RTOL);
      }
    }
  }

  #[test]
  fn weighted_hodge_mass_uses_cell_average_for_affine_weight_with_barycentric_qr() {
    // With barycentric quadrature and an affine weight w(x),
    // the implementation uses the (approx) cell-average, which matches w(barycenter).
    const RTOL: f64 = 1e-12;

    let dim = 2;
    let grade = 0;

    let geo = SimplexLengths::standard(dim);
    let topo = Complex::standard(dim);
    let cell = topo.cells().handle_iter().next().unwrap();

    // affine weight: w(x) = 1 + x0
    // on the standard simplex in R^dim, avg(x0) = 1/(dim+1), hence avg(w) = 1 + 1/(dim+1)
    let expected_w_avg = 1.0 + 1.0 / (dim as f64 + 1.0);

    let unweighted = HodgeMassElmat::new(dim, grade).eval(&geo);

    let coords = MeshCoords::standard(dim);
    let weight = InnerProductWeightClosure::new(|x| 1.0 + x[0]);

    let weighted = HodgeMassElmat::new_weighted(dim, grade, &coords, None, &weight)
      .eval_with_coords(&geo, &cell);

    let expected = expected_w_avg * &unweighted;
    assert_relative_eq!(&weighted, &expected, max_relative = RTOL);
  }

  #[test]
  fn weighted_hodge_mass_matrix_identity_matches_unweighted() {
    // sanity check that matrix-valued weights work and identity leaves the result unchanged
    const RTOL: f64 = 1e-12;

    let dim = 2;
    let grade = 1;

    let geo = SimplexLengths::standard(dim);
    let topo = Complex::standard(dim);
    let cell = topo.cells().handle_iter().next().unwrap();

    let unweighted = HodgeMassElmat::<f64>::new(dim, grade).eval(&geo);

    let coords = MeshCoords::standard(dim);

    // For 1-forms in 2D, coeff dimension is 2, so use 2x2 identity.
    let weight = InnerProductWeightClosure::new(|_| Matrix::identity(2, 2));

    let weighted = HodgeMassElmat::<Matrix>::new_weighted(dim, grade, &coords, None, &weight)
      .eval_with_coords(&geo, &cell);

    assert_relative_eq!(&weighted, &unweighted, max_relative = RTOL);
  }

  #[test]
  fn nc1_projection_times_embedding_is_identity() {
    for dim in [2, 3] {
      let projection = nc1_local_projection_matrix(dim);
      let embedding = nc1_local_embedding_matrix(dim);
      let identity = Matrix::identity(projection.nrows(), projection.nrows());
      assert_relative_eq!(&(projection * embedding), &identity);
    }
  }

  #[test]
  fn nc1_mass_matches_whitney_mass_via_embedding() {
    for dim in [2, 3] {
      let geo = SimplexLengths::standard(dim);
      let embedding = nc1_local_embedding_matrix(dim);
      let nc1_mass = Nc1MassElmat::new(dim).eval(&geo);
      let whitney_mass = HodgeMassElmat::new(dim, 1).eval(&geo);
      assert_relative_eq!(
        &(embedding.transpose() * nc1_mass * embedding),
        &whitney_mass
      );
    }
  }

  #[test]
  fn weighted_nc1_mass_scales_with_constant_scalar_weight() {
    const W: f64 = 2.5;
    const RTOL: f64 = 1e-12;

    for dim in [2, 3] {
      let geo = SimplexLengths::standard(dim);
      let topo = Complex::standard(dim);
      let cell = topo.cells().handle_iter().next().unwrap();

      let unweighted = Nc1MassElmat::new(dim).eval(&geo);

      let coords = MeshCoords::standard(dim);
      let weight = InnerProductWeightClosure::new(|_| W);

      let weighted =
        Nc1MassElmat::new_weighted(dim, &coords, None, &weight).eval_with_coords(&geo, &cell);

      assert_relative_eq!(&weighted, &(W * unweighted), max_relative = RTOL);
    }
  }

  #[test]
  fn weighted_nc1_mass_matrix_identity_matches_unweighted() {
    const RTOL: f64 = 1e-12;

    let dim = 3;
    let geo = SimplexLengths::standard(dim);
    let topo = Complex::standard(dim);
    let cell = topo.cells().handle_iter().next().unwrap();

    let unweighted = Nc1MassElmat::<f64>::new(dim).eval(&geo);
    let coords = MeshCoords::standard(dim);
    let weight = InnerProductWeightClosure::new(move |_| Matrix::identity(dim, dim));

    let weighted = Nc1MassElmat::<Matrix>::new_weighted(dim, &coords, None, &weight)
      .eval_with_coords(&geo, &cell);

    assert_relative_eq!(&weighted, &unweighted, max_relative = RTOL);
  }

  #[test]
  fn nc1_lumped_mass_is_vertex_block_diagonal() {
    const RTOL: f64 = 1e-12;

    for dim in [2, 3] {
      let geo = SimplexLengths::standard(dim);
      let lumped = Nc1LumpedMassElmat::new(dim).eval(&geo);

      for i in 0..lumped.nrows() {
        for j in 0..lumped.ncols() {
          if nc1_local_dof_vertex(dim, i) != nc1_local_dof_vertex(dim, j) {
            assert_relative_eq!(lumped[(i, j)], 0.0, max_relative = RTOL, epsilon = RTOL);
          }
        }
      }
    }
  }

  #[test]
  fn weighted_nc1_lumped_mass_scales_with_constant_scalar_weight() {
    const W: f64 = 2.5;
    const RTOL: f64 = 1e-12;

    for dim in [2, 3] {
      let geo = SimplexLengths::standard(dim);
      let topo = Complex::standard(dim);
      let cell = topo.cells().handle_iter().next().unwrap();

      let unweighted = Nc1LumpedMassElmat::new(dim).eval(&geo);

      let coords = MeshCoords::standard(dim);
      let weight = InnerProductWeightClosure::new(|_| W);
      let weighted =
        Nc1LumpedMassElmat::new_weighted(dim, &coords, None, &weight).eval_with_coords(&geo, &cell);

      assert_relative_eq!(&weighted, &(W * unweighted), max_relative = RTOL);
    }
  }

  #[test]
  fn nc1_local_projected_sparse_inverse_matches_dense_formula() {
    const RTOL: f64 = 1e-12;

    for dim in [2, 3] {
      let lumped = Nc1LumpedMassElmat::new(dim).eval(&SimplexLengths::standard(dim));
      let projection = nc1_local_projection_matrix(dim);
      let expected = &projection * lumped.clone().try_inverse().unwrap() * projection.transpose();

      let mut inverse = Matrix::zeros(lumped.nrows(), lumped.ncols());
      for ivertex in 0..=dim {
        let dofs = (0..lumped.nrows())
          .filter(|&idof| nc1_local_dof_vertex(dim, idof) == ivertex)
          .collect::<Vec<_>>();

        let mut block = Matrix::zeros(dofs.len(), dofs.len());
        for (iblock, &iglobal) in dofs.iter().enumerate() {
          for (jblock, &jglobal) in dofs.iter().enumerate() {
            block[(iblock, jblock)] = lumped[(iglobal, jglobal)];
          }
        }

        let inv_block = block
          .clone()
          .cholesky()
          .expect("NC1 lumped mass vertex block must be positive definite.")
          .inverse();

        for (iblock, &iglobal) in dofs.iter().enumerate() {
          for (jblock, &jglobal) in dofs.iter().enumerate() {
            inverse[(iglobal, jglobal)] = inv_block[(iblock, jblock)];
          }
        }
      }

      let actual = &projection * inverse * projection.transpose();
      assert_relative_eq!(&actual, &expected, max_relative = RTOL, epsilon = RTOL);
    }
  }
}
