//! Nonlinear magnetostatic FEEC residual assembly.
//!
//! V1 implements the physically standard 2D reduction
//! `A = A_z(x, y) dz`, so the active unknown is a Whitney 0-form scalar on
//! vertices. The same residual/Jacobian contract is intended to carry the future
//! 3D Whitney 1-form vector-potential model, where `b = d1 a` and the residual
//! has the shape `d1^T M2(nu(|B|^2), a) d1 a - j`.

use crate::problems::residual::{
  ResidualEvaluation as NonlinearResidualEvaluation, ResidualModel as NonlinearResidualModel,
};
use crate::{
  operators::InnerProductWeightClosure,
  problems::reduced_linear::build_reduced_weighted_hodge_laplace_1form_system,
  reduction::{DofLayout, EssentialBoundarySpec, PrescribedDof},
};
use common::linalg::nalgebra::{CooMatrix, CsrMatrix, Vector};
use ddf::whitney::lsf::WhitneyLsf;
use exterior::field::ExteriorField;
use manifold::{
  geometry::coord::{
    mesh::MeshCoords,
    simplex::{barycenter_local, SimplexHandleExt},
    CoordRef,
  },
  topology::{complex::Complex, handle::SimplexHandle},
};
use std::{
  collections::{BTreeMap, BTreeSet},
  sync::Arc,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearReluctivityLaw {
  pub nu0: f64,
  pub beta: f64,
}

impl NonlinearReluctivityLaw {
  pub fn new(nu0: f64, beta: f64) -> Result<Self, String> {
    let law = Self { nu0, beta };
    law.validate()?;
    Ok(law)
  }

  pub fn validate(&self) -> Result<(), String> {
    if !self.nu0.is_finite() || self.nu0 <= 0.0 {
      return Err("nonlinear reluctivity nu0 must be finite and positive".to_string());
    }
    if !self.beta.is_finite() || self.beta < 0.0 {
      return Err("nonlinear reluctivity beta must be finite and nonnegative".to_string());
    }
    Ok(())
  }

  pub fn nu(&self, s: f64) -> f64 {
    self.nu0 * (1.0 + self.beta * s)
  }

  pub fn d_nu_ds(&self, _s: f64) -> f64 {
    self.nu0 * self.beta
  }
}

/// Deterministic constitutive policy used by native magnetostatic assembly.
pub trait SpatialReluctivity: std::fmt::Debug + Send + Sync {
  fn validate(&self) -> Result<(), String>;
  fn nu(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64;
  fn d_nu_ds(&self, point: [f64; 3], magnetic_flux_squared: f64) -> f64;
  fn linear_reference_reluctivity(&self, point: [f64; 3]) -> f64;
}

impl SpatialReluctivity for NonlinearReluctivityLaw {
  fn validate(&self) -> Result<(), String> {
    self.validate()
  }

  fn nu(&self, _point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
    self.nu(magnetic_flux_squared)
  }

  fn d_nu_ds(&self, _point: [f64; 3], magnetic_flux_squared: f64) -> f64 {
    self.d_nu_ds(magnetic_flux_squared)
  }

  fn linear_reference_reluctivity(&self, _point: [f64; 3]) -> f64 {
    self.nu0
  }
}

#[derive(Debug, Clone)]
pub struct NonlinearMagnetostaticAssemblyConfig {
  pub material: Arc<dyn SpatialReluctivity>,
  /// Source vector on the reduced active-vertex layout.
  pub source: Option<Vec<f64>>,
  pub boundary: EssentialBoundarySpec,
}

impl NonlinearMagnetostaticAssemblyConfig {
  pub fn new(material: impl SpatialReluctivity + 'static, boundary: EssentialBoundarySpec) -> Self {
    Self {
      material: Arc::new(material),
      source: None,
      boundary,
    }
  }

  pub fn with_source(mut self, source: impl AsRef<[f64]>) -> Self {
    self.source = Some(source.as_ref().to_vec());
    self
  }

  pub fn with_shared_material(
    material: Arc<dyn SpatialReluctivity>,
    boundary: EssentialBoundarySpec,
  ) -> Self {
    Self {
      material,
      source: None,
      boundary,
    }
  }
}

#[derive(Debug, Clone)]
pub struct ReducedScalarAzMagnetostatic2d {
  material: Arc<dyn SpatialReluctivity>,
  layout: DofLayout,
  reduced_index_by_full: Vec<Option<usize>>,
  source: Vec<f64>,
  elements: Vec<TriangleElement>,
}

impl ReducedScalarAzMagnetostatic2d {
  pub fn reduced_dimension(&self) -> usize {
    self.layout.reduced_dimension()
  }

  pub fn layout(&self) -> &DofLayout {
    &self.layout
  }

  pub fn material(&self) -> &dyn SpatialReluctivity {
    self.material.as_ref()
  }

  pub fn source(&self) -> &[f64] {
    &self.source
  }

  pub fn num_elements(&self) -> usize {
    self.elements.len()
  }

  pub fn with_source(mut self, source: impl AsRef<[f64]>) -> Result<Self, String> {
    let source = source.as_ref().to_vec();
    if source.len() != self.reduced_dimension() {
      return Err(format!(
        "magnetostatic reduced source length {} must match active dimension {}",
        source.len(),
        self.reduced_dimension()
      ));
    }
    if !source.iter().all(|value| value.is_finite()) {
      return Err("magnetostatic reduced source must contain only finite values".to_string());
    }
    self.source = source;
    Ok(self)
  }

  pub fn manufactured_source(&self, reduced_state: &[f64]) -> Result<Vector, String> {
    self
      .assemble_source_free(reduced_state)
      .map(|evaluation| evaluation.residual)
  }

  pub fn source_free_residual_and_jacobian(
    &self,
    reduced_state: &[f64],
  ) -> Result<NonlinearResidualEvaluation, String> {
    self.assemble_source_free(reduced_state)
  }

  pub fn lift_reduced_state(&self, reduced_state: &[f64]) -> Result<Vec<f64>, String> {
    lift_reduced_with_layout(&self.layout, reduced_state)
  }

  fn assemble_source_free(
    &self,
    reduced_state: &[f64],
  ) -> Result<NonlinearResidualEvaluation, String> {
    if reduced_state.len() != self.reduced_dimension() {
      return Err(format!(
        "magnetostatic state length {} must match active dimension {}",
        reduced_state.len(),
        self.reduced_dimension()
      ));
    }
    if !reduced_state.iter().all(|value| value.is_finite()) {
      return Err("magnetostatic state must contain only finite values".to_string());
    }

    let full_state = lift_reduced_with_layout(&self.layout, reduced_state)?;
    let mut residual = vec![0.0; self.reduced_dimension()];
    let mut jacobian_values = BTreeMap::<(usize, usize), f64>::new();

    for element in &self.elements {
      let grad_a = element.gradient_of(&full_state);
      let s = dot2(grad_a, grad_a);
      let nu = self.material.nu(element.barycenter, s);
      let dnu = self.material.d_nu_ds(element.barycenter, s);
      let flux = [nu * grad_a[0], nu * grad_a[1]];
      let tangent = [
        [
          nu + 2.0 * dnu * grad_a[0] * grad_a[0],
          2.0 * dnu * grad_a[0] * grad_a[1],
        ],
        [
          2.0 * dnu * grad_a[1] * grad_a[0],
          nu + 2.0 * dnu * grad_a[1] * grad_a[1],
        ],
      ];

      for local_row in 0..3 {
        let full_row = element.vertices[local_row];
        let Some(row) = self.reduced_index_by_full[full_row] else {
          continue;
        };
        let grad_row = element.gradients[local_row];
        residual[row] += element.area * dot2(flux, grad_row);

        for local_col in 0..3 {
          let full_col = element.vertices[local_col];
          let Some(col) = self.reduced_index_by_full[full_col] else {
            continue;
          };
          let grad_col = element.gradients[local_col];
          let value = element.area * bilinear2(grad_row, tangent, grad_col);
          if value != 0.0 {
            *jacobian_values.entry((row, col)).or_insert(0.0) += value;
          }
        }
      }
    }

    Ok(NonlinearResidualEvaluation {
      residual: Vector::from_vec(residual),
      jacobian: csr_from_triplets(
        self.reduced_dimension(),
        self.reduced_dimension(),
        jacobian_values
          .into_iter()
          .filter(|(_, value)| *value != 0.0)
          .map(|((row, col), value)| (row, col, value)),
      ),
    })
  }
}

impl NonlinearResidualModel for ReducedScalarAzMagnetostatic2d {
  fn state_dimension(&self) -> usize {
    self.reduced_dimension()
  }

  fn residual_dimension(&self) -> usize {
    self.reduced_dimension()
  }

  fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
    let mut evaluation = self.assemble_source_free(state)?;
    for (value, source) in evaluation.residual.iter_mut().zip(self.source.iter()) {
      *value -= *source;
    }
    Ok(evaluation)
  }
}

#[derive(Debug, Clone)]
pub struct ReducedVectorPotentialMagnetostatic3d {
  material: Arc<dyn SpatialReluctivity>,
  layout: DofLayout,
  reduced_index_by_full: Vec<Option<usize>>,
  source: Vec<f64>,
  elements: Vec<TetrahedronElement>,
  boundary_edge_dofs: Vec<usize>,
  hodge_coulomb_operator: CsrMatrix,
  hodge_coulomb_bias: Vec<f64>,
  state_mass: CsrMatrix,
  state_mass_inverse: Option<CsrMatrix>,
}

impl ReducedVectorPotentialMagnetostatic3d {
  pub fn reduced_dimension(&self) -> usize {
    self.layout.reduced_dimension()
  }

  pub fn layout(&self) -> &DofLayout {
    &self.layout
  }

  pub fn material(&self) -> &dyn SpatialReluctivity {
    self.material.as_ref()
  }

  pub fn source(&self) -> &[f64] {
    &self.source
  }

  pub fn num_elements(&self) -> usize {
    self.elements.len()
  }

  pub fn boundary_edge_dofs(&self) -> &[usize] {
    &self.boundary_edge_dofs
  }

  pub fn gauge_edge_dofs(&self) -> &[usize] {
    &[]
  }

  pub fn state_mass(&self) -> &CsrMatrix {
    &self.state_mass
  }

  pub fn state_mass_inverse(&self) -> Option<&CsrMatrix> {
    self.state_mass_inverse.as_ref()
  }

  pub fn with_source(mut self, source: impl AsRef<[f64]>) -> Result<Self, String> {
    let source = source.as_ref().to_vec();
    if source.len() != self.reduced_dimension() {
      return Err(format!(
        "3D magnetostatic reduced source length {} must match active dimension {}",
        source.len(),
        self.reduced_dimension()
      ));
    }
    if !source.iter().all(|value| value.is_finite()) {
      return Err("3D magnetostatic reduced source must contain only finite values".to_string());
    }
    self.source = source;
    Ok(self)
  }

  pub fn without_coulomb_gauge(mut self) -> Self {
    let dimension = self.reduced_dimension();
    self.hodge_coulomb_operator = zero_csr(dimension, dimension);
    self.hodge_coulomb_bias = vec![0.0; dimension];
    self
  }

  pub fn manufactured_source(&self, reduced_state: &[f64]) -> Result<Vector, String> {
    self
      .assemble_source_free(reduced_state)
      .map(|evaluation| evaluation.residual)
  }

  pub fn residual_with_material<M>(
    &self,
    material: &M,
    reduced_state: &[f64],
  ) -> Result<Vec<f64>, String>
  where
    M: SpatialReluctivity + ?Sized,
  {
    let mut residual = self.assemble_source_free_residual_with_material(material, reduced_state)?;
    for (value, source) in residual.iter_mut().zip(self.source.iter()) {
      *value -= *source;
    }
    Ok(residual)
  }

  pub fn residual_and_jacobian_with_material<M>(
    &self,
    material: &M,
    reduced_state: &[f64],
  ) -> Result<NonlinearResidualEvaluation, String>
  where
    M: SpatialReluctivity + ?Sized,
  {
    let mut evaluation = self.assemble_source_free_with_material(material, reduced_state)?;
    for (value, source) in evaluation.residual.iter_mut().zip(self.source.iter()) {
      *value -= *source;
    }
    Ok(evaluation)
  }

  pub fn source_free_residual_and_jacobian(
    &self,
    reduced_state: &[f64],
  ) -> Result<NonlinearResidualEvaluation, String> {
    self.assemble_source_free(reduced_state)
  }

  pub fn source_free_residual(&self, reduced_state: &[f64]) -> Result<Vec<f64>, String> {
    self.assemble_source_free_residual(reduced_state)
  }

  pub fn material_sensitivity_columns_with_material<M, F>(
    &self,
    material: &M,
    reduced_state: &[f64],
    parameter_count: usize,
    dnu_dparam: F,
  ) -> Result<CsrMatrix, String>
  where
    M: SpatialReluctivity + ?Sized,
    F: Fn(&M, [f64; 3], f64) -> Result<Vec<f64>, String>,
  {
    if reduced_state.len() != self.reduced_dimension() {
      return Err(format!(
        "3D magnetostatic state length {} must match active dimension {}",
        reduced_state.len(),
        self.reduced_dimension()
      ));
    }
    if !reduced_state.iter().all(|value| value.is_finite()) {
      return Err("3D magnetostatic state must contain only finite values".to_string());
    }
    assemble_3d_curl_material_sensitivity_columns(
      material,
      &self.layout,
      &self.reduced_index_by_full,
      &self.elements,
      reduced_state,
      parameter_count,
      dnu_dparam,
    )
  }

  pub fn lift_reduced_state(&self, reduced_state: &[f64]) -> Result<Vec<f64>, String> {
    lift_reduced_with_layout(&self.layout, reduced_state)
  }

  fn assemble_source_free(
    &self,
    reduced_state: &[f64],
  ) -> Result<NonlinearResidualEvaluation, String> {
    self.assemble_source_free_with_material(self.material.as_ref(), reduced_state)
  }

  fn assemble_source_free_with_material<M>(
    &self,
    material: &M,
    reduced_state: &[f64],
  ) -> Result<NonlinearResidualEvaluation, String>
  where
    M: SpatialReluctivity + ?Sized,
  {
    if reduced_state.len() != self.reduced_dimension() {
      return Err(format!(
        "3D magnetostatic state length {} must match active dimension {}",
        reduced_state.len(),
        self.reduced_dimension()
      ));
    }
    if !reduced_state.iter().all(|value| value.is_finite()) {
      return Err("3D magnetostatic state must contain only finite values".to_string());
    }

    let mut evaluation = assemble_3d_curl_source_free(
      material,
      &self.layout,
      &self.reduced_index_by_full,
      &self.elements,
      reduced_state,
    )?;
    add_sparse_affine_term(
      &mut evaluation.residual,
      &mut evaluation.jacobian,
      &self.hodge_coulomb_operator,
      &self.hodge_coulomb_bias,
      reduced_state,
    )?;
    Ok(evaluation)
  }

  fn assemble_source_free_residual(&self, reduced_state: &[f64]) -> Result<Vec<f64>, String> {
    self.assemble_source_free_residual_with_material(self.material.as_ref(), reduced_state)
  }

  fn assemble_source_free_residual_with_material<M>(
    &self,
    material: &M,
    reduced_state: &[f64],
  ) -> Result<Vec<f64>, String>
  where
    M: SpatialReluctivity + ?Sized,
  {
    if reduced_state.len() != self.reduced_dimension() {
      return Err(format!(
        "3D magnetostatic state length {} must match active dimension {}",
        reduced_state.len(),
        self.reduced_dimension()
      ));
    }
    if !reduced_state.iter().all(|value| value.is_finite()) {
      return Err("3D magnetostatic state must contain only finite values".to_string());
    }

    let mut residual = assemble_3d_curl_source_free_residual(
      material,
      &self.layout,
      &self.reduced_index_by_full,
      &self.elements,
      reduced_state,
    )?;
    add_sparse_affine_residual(
      &mut residual,
      &self.hodge_coulomb_operator,
      &self.hodge_coulomb_bias,
      reduced_state,
    )?;
    Ok(residual)
  }
}

fn assemble_3d_curl_source_free<M>(
  material: &M,
  layout: &DofLayout,
  reduced_index_by_full: &[Option<usize>],
  elements: &[TetrahedronElement],
  reduced_state: &[f64],
) -> Result<NonlinearResidualEvaluation, String>
where
  M: SpatialReluctivity + ?Sized,
{
  let full_state = lift_reduced_with_layout(layout, reduced_state)?;
  let mut residual = vec![0.0; layout.reduced_dimension()];
  let mut jacobian_values = BTreeMap::<(usize, usize), f64>::new();

  for element in elements {
    let magnetic_flux = element.magnetic_flux(&full_state);
    let s = dot3(magnetic_flux, magnetic_flux);
    let nu = material.nu(element.barycenter, s);
    let dnu = material.d_nu_ds(element.barycenter, s);
    let weighted_flux = [
      nu * magnetic_flux[0],
      nu * magnetic_flux[1],
      nu * magnetic_flux[2],
    ];
    let tangent = [
      [
        nu + 2.0 * dnu * magnetic_flux[0] * magnetic_flux[0],
        2.0 * dnu * magnetic_flux[0] * magnetic_flux[1],
        2.0 * dnu * magnetic_flux[0] * magnetic_flux[2],
      ],
      [
        2.0 * dnu * magnetic_flux[1] * magnetic_flux[0],
        nu + 2.0 * dnu * magnetic_flux[1] * magnetic_flux[1],
        2.0 * dnu * magnetic_flux[1] * magnetic_flux[2],
      ],
      [
        2.0 * dnu * magnetic_flux[2] * magnetic_flux[0],
        2.0 * dnu * magnetic_flux[2] * magnetic_flux[1],
        nu + 2.0 * dnu * magnetic_flux[2] * magnetic_flux[2],
      ],
    ];

    for local_row in 0..6 {
      let full_row = element.edges[local_row];
      let Some(row) = reduced_index_by_full[full_row] else {
        continue;
      };
      let curl_row = element.curls[local_row];
      residual[row] += element.volume * dot3(weighted_flux, curl_row);

      for local_col in 0..6 {
        let full_col = element.edges[local_col];
        let Some(col) = reduced_index_by_full[full_col] else {
          continue;
        };
        let curl_col = element.curls[local_col];
        let value = element.volume * bilinear3(curl_row, tangent, curl_col);
        if value != 0.0 {
          *jacobian_values.entry((row, col)).or_insert(0.0) += value;
        }
      }
    }
  }

  Ok(NonlinearResidualEvaluation {
    residual: Vector::from_vec(residual),
    jacobian: csr_from_triplets(
      layout.reduced_dimension(),
      layout.reduced_dimension(),
      jacobian_values
        .into_iter()
        .filter(|(_, value)| *value != 0.0)
        .map(|((row, col), value)| (row, col, value)),
    ),
  })
}

fn assemble_3d_curl_source_free_residual<M>(
  material: &M,
  layout: &DofLayout,
  reduced_index_by_full: &[Option<usize>],
  elements: &[TetrahedronElement],
  reduced_state: &[f64],
) -> Result<Vec<f64>, String>
where
  M: SpatialReluctivity + ?Sized,
{
  let full_state = lift_reduced_with_layout(layout, reduced_state)?;
  let mut residual = vec![0.0; layout.reduced_dimension()];

  for element in elements {
    let magnetic_flux = element.magnetic_flux(&full_state);
    let s = dot3(magnetic_flux, magnetic_flux);
    let nu = material.nu(element.barycenter, s);
    let weighted_flux = [
      nu * magnetic_flux[0],
      nu * magnetic_flux[1],
      nu * magnetic_flux[2],
    ];

    for local_row in 0..6 {
      let full_row = element.edges[local_row];
      let Some(row) = reduced_index_by_full[full_row] else {
        continue;
      };
      residual[row] += element.volume * dot3(weighted_flux, element.curls[local_row]);
    }
  }

  Ok(residual)
}

fn assemble_3d_curl_material_sensitivity_columns<M, F>(
  material: &M,
  layout: &DofLayout,
  reduced_index_by_full: &[Option<usize>],
  elements: &[TetrahedronElement],
  reduced_state: &[f64],
  parameter_count: usize,
  dnu_dparam: F,
) -> Result<CsrMatrix, String>
where
  M: SpatialReluctivity + ?Sized,
  F: Fn(&M, [f64; 3], f64) -> Result<Vec<f64>, String>,
{
  let full_state = lift_reduced_with_layout(layout, reduced_state)?;
  let mut values = BTreeMap::<(usize, usize), f64>::new();

  for element in elements {
    let magnetic_flux = element.magnetic_flux(&full_state);
    let s = dot3(magnetic_flux, magnetic_flux);
    let sensitivities = dnu_dparam(material, element.barycenter, s)?;
    if sensitivities.len() != parameter_count {
      return Err(format!(
        "material sensitivity callback returned {} columns, expected {}",
        sensitivities.len(),
        parameter_count
      ));
    }

    for local_row in 0..6 {
      let full_row = element.edges[local_row];
      let Some(row) = reduced_index_by_full[full_row] else {
        continue;
      };
      let row_factor = element.volume * dot3(magnetic_flux, element.curls[local_row]);
      if row_factor == 0.0 {
        continue;
      }
      for (parameter, sensitivity) in sensitivities.iter().enumerate() {
        let value = row_factor * *sensitivity;
        if value != 0.0 {
          *values.entry((row, parameter)).or_insert(0.0) += value;
        }
      }
    }
  }

  Ok(csr_from_triplets(
    layout.reduced_dimension(),
    parameter_count,
    values
      .into_iter()
      .filter(|(_, value)| *value != 0.0)
      .map(|((row, col), value)| (row, col, value)),
  ))
}

fn add_sparse_affine_term(
  residual: &mut Vector,
  jacobian: &mut CsrMatrix,
  operator: &CsrMatrix,
  bias: &[f64],
  state: &[f64],
) -> Result<(), String> {
  if residual.len() != operator.nrows() || bias.len() != operator.nrows() {
    return Err("affine correction dimensions do not match residual dimension".to_string());
  }
  if state.len() != operator.ncols() {
    return Err("affine correction state dimension mismatch".to_string());
  }
  for (out, value) in residual.iter_mut().zip(bias.iter()) {
    *out += *value;
  }
  for (row, col, value) in operator.triplet_iter() {
    residual[row] += value * state[col];
  }

  *jacobian = add_csr(jacobian, operator);
  Ok(())
}

fn add_sparse_affine_residual(
  residual: &mut [f64],
  operator: &CsrMatrix,
  bias: &[f64],
  state: &[f64],
) -> Result<(), String> {
  if residual.len() != operator.nrows() || bias.len() != operator.nrows() {
    return Err("affine correction dimensions do not match residual dimension".to_string());
  }
  if state.len() != operator.ncols() {
    return Err("affine correction state dimension mismatch".to_string());
  }
  for (out, value) in residual.iter_mut().zip(bias.iter()) {
    *out += *value;
  }
  for (row, col, value) in operator.triplet_iter() {
    residual[row] += value * state[col];
  }
  Ok(())
}

impl NonlinearResidualModel for ReducedVectorPotentialMagnetostatic3d {
  fn state_dimension(&self) -> usize {
    self.reduced_dimension()
  }

  fn residual_dimension(&self) -> usize {
    self.reduced_dimension()
  }

  fn residual(&self, state: &[f64]) -> Result<Vector, String> {
    let mut residual = self.assemble_source_free_residual(state)?;
    for (value, source) in residual.iter_mut().zip(self.source.iter()) {
      *value -= *source;
    }
    Ok(Vector::from_vec(residual))
  }

  fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
    let mut evaluation = self.assemble_source_free(state)?;
    for (value, source) in evaluation.residual.iter_mut().zip(self.source.iter()) {
      *value -= *source;
    }
    Ok(evaluation)
  }
}

#[derive(Debug, Clone)]
pub struct LocalMagneticStrongProbe3d {
  layout: DofLayout,
  reduced_index_by_full: Vec<Option<usize>>,
  weak_model: ReducedVectorPotentialMagnetostatic3d,
  elements: Vec<TetrahedronElement>,
  selected_cells: Vec<usize>,
  state_mass_inverse: CsrMatrix,
}

impl LocalMagneticStrongProbe3d {
  pub fn from_vector_potential_model(
    model: &ReducedVectorPotentialMagnetostatic3d,
    selected_cells: Vec<usize>,
  ) -> Result<Self, String> {
    let mut seen = BTreeSet::new();
    for cell in &selected_cells {
      if *cell >= model.elements.len() {
        return Err(format!(
          "selected local magnetic probe cell {cell} is outside cell count {}",
          model.elements.len()
        ));
      }
      if !seen.insert(*cell) {
        return Err(format!(
          "selected local magnetic probe cell {cell} appears more than once"
        ));
      }
    }
    let state_mass_inverse = model.state_mass_inverse.clone().ok_or_else(|| {
      "local magnetic strong probes require the FEEC projected sparse 1-form mass inverse"
        .to_string()
    })?;
    Ok(Self {
      layout: model.layout.clone(),
      reduced_index_by_full: model.reduced_index_by_full.clone(),
      weak_model: model.clone(),
      elements: model.elements.clone(),
      selected_cells,
      state_mass_inverse,
    })
  }

  pub fn selected_cells(&self) -> &[usize] {
    &self.selected_cells
  }

  pub fn cell_count(&self) -> usize {
    self.elements.len()
  }

  pub fn reduced_dimension(&self) -> usize {
    self.layout.reduced_dimension()
  }

  fn assemble_projected_weak_residual(
    &self,
    reduced_state: &[f64],
  ) -> Result<ProjectedIntensityEvaluation, String> {
    if reduced_state.len() != self.reduced_dimension() {
      return Err(format!(
        "local magnetic strong probe state length {} must match active dimension {}",
        reduced_state.len(),
        self.reduced_dimension()
      ));
    }
    if !reduced_state.iter().all(|value| value.is_finite()) {
      return Err("local magnetic strong probe state must contain only finite values".to_string());
    }

    let weak_evaluation = self.weak_model.residual_and_jacobian(reduced_state)?;
    let coefficients = sparse_mul_vec(
      &self.state_mass_inverse,
      weak_evaluation.residual.as_slice(),
    )?;
    let jacobian = sparse_matmul(&self.state_mass_inverse, &weak_evaluation.jacobian)?;
    Ok(ProjectedIntensityEvaluation {
      coefficients,
      jacobian,
    })
  }
}

struct ProjectedIntensityEvaluation {
  coefficients: Vec<f64>,
  jacobian: CsrMatrix,
}

impl NonlinearResidualModel for LocalMagneticStrongProbe3d {
  fn state_dimension(&self) -> usize {
    self.reduced_dimension()
  }

  fn residual_dimension(&self) -> usize {
    3 * self.selected_cells.len()
  }

  fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
    let projected = self.assemble_projected_weak_residual(state)?;
    let full_residual = lift_active_to_full_zero_fixed(&self.layout, &projected.coefficients)?;
    let projected_jacobian_rows = sparse_rows(&projected.jacobian);
    let mut residual = vec![0.0; self.residual_dimension()];
    let mut jacobian_values = BTreeMap::<(usize, usize), f64>::new();

    for (probe_index, &cell_index) in self.selected_cells.iter().enumerate() {
      let element = &self.elements[cell_index];
      for component in 0..3 {
        let mut value = 0.0;
        for local_edge in 0..6 {
          value += full_residual[element.edges[local_edge]] * element.values[local_edge][component];
        }
        residual[3 * probe_index + component] = value;
      }

      for local_edge in 0..6 {
        let full_edge = element.edges[local_edge];
        let Some(projected_row) = self.reduced_index_by_full[full_edge] else {
          continue;
        };
        for component in 0..3 {
          let basis_value = element.values[local_edge][component];
          if basis_value == 0.0 {
            continue;
          }
          let residual_row = 3 * probe_index + component;
          for (col, projected_value) in &projected_jacobian_rows[projected_row] {
            let value = basis_value * projected_value;
            if value != 0.0 {
              *jacobian_values.entry((residual_row, *col)).or_insert(0.0) += value;
            }
          }
        }
      }
    }

    Ok(NonlinearResidualEvaluation {
      residual: Vector::from_vec(residual),
      jacobian: csr_from_triplets(
        self.residual_dimension(),
        self.reduced_dimension(),
        jacobian_values
          .into_iter()
          .filter(|(_, value)| value.abs() > 0.0)
          .map(|((row, col), value)| (row, col, value)),
      ),
    })
  }
}

pub fn build_reduced_scalar_az_magnetostatic_2d(
  topology: &Complex,
  coords: &MeshCoords,
  config: NonlinearMagnetostaticAssemblyConfig,
) -> Result<ReducedScalarAzMagnetostatic2d, String> {
  config.material.validate()?;
  ensure_no_auxiliary_regions(&config.boundary)?;
  if topology.dim() != 2 {
    return Err(format!(
      "2D scalar A_z magnetostatic reduction requires topology dimension 2, got {}",
      topology.dim()
    ));
  }
  if coords.dim() != 2 {
    return Err(format!(
      "2D scalar A_z magnetostatic reduction requires coordinate dimension 2, got {}",
      coords.dim()
    ));
  }
  if coords.nvertices() != topology.nsimplices(0) {
    return Err(format!(
      "coordinate vertex count {} must match topology vertex count {}",
      coords.nvertices(),
      topology.nsimplices(0)
    ));
  }

  let layout = build_state_layout(topology.nsimplices(0), &config.boundary.state)?;
  let reduced_index_by_full = reduced_index_map(&layout);
  let source = match config.source {
    Some(source) => {
      if source.len() != layout.reduced_dimension() {
        return Err(format!(
          "magnetostatic reduced source length {} must match active dimension {}",
          source.len(),
          layout.reduced_dimension()
        ));
      }
      if !source.iter().all(|value| value.is_finite()) {
        return Err("magnetostatic reduced source must contain only finite values".to_string());
      }
      source
    }
    None => vec![0.0; layout.reduced_dimension()],
  };
  let elements = topology
    .cells()
    .handle_iter()
    .map(|cell| TriangleElement::from_cell_vertices(cell.vertices.clone(), coords))
    .collect::<Result<Vec<_>, _>>()?;
  if elements.is_empty() {
    return Err("2D scalar A_z magnetostatic reduction requires at least one triangle".to_string());
  }

  Ok(ReducedScalarAzMagnetostatic2d {
    material: config.material,
    layout,
    reduced_index_by_full,
    source,
    elements,
  })
}

pub fn build_reduced_vector_potential_magnetostatic_3d(
  topology: &Complex,
  coords: &MeshCoords,
  config: NonlinearMagnetostaticAssemblyConfig,
) -> Result<ReducedVectorPotentialMagnetostatic3d, String> {
  config.material.validate()?;
  if topology.dim() != 3 {
    return Err(format!(
      "3D vector-potential magnetostatic model requires topology dimension 3, got {}",
      topology.dim()
    ));
  }
  if coords.dim() != 3 {
    return Err(format!(
      "3D vector-potential magnetostatic model requires coordinate dimension 3, got {}",
      coords.dim()
    ));
  }
  if coords.nvertices() != topology.nsimplices(0) {
    return Err(format!(
      "coordinate vertex count {} must match topology vertex count {}",
      coords.nvertices(),
      topology.nsimplices(0)
    ));
  }

  let (layout, boundary_edge_dofs) = build_3d_edge_layout(topology, &config.boundary.state)?;
  let reduced_index_by_full = reduced_index_map(&layout);
  let source = match config.source {
    Some(source) => {
      if source.len() != layout.reduced_dimension() {
        return Err(format!(
          "3D magnetostatic reduced source length {} must match active dimension {}",
          source.len(),
          layout.reduced_dimension()
        ));
      }
      if !source.iter().all(|value| value.is_finite()) {
        return Err("3D magnetostatic reduced source must contain only finite values".to_string());
      }
      source
    }
    None => vec![0.0; layout.reduced_dimension()],
  };
  let elements = topology
    .cells()
    .handle_iter()
    .map(|cell| TetrahedronElement::from_cell(cell, coords))
    .collect::<Result<Vec<_>, _>>()?;
  if elements.is_empty() {
    return Err(
      "3D vector-potential magnetostatic model requires at least one tetrahedron".to_string(),
    );
  }
  let zero_state = vec![0.0; layout.reduced_dimension()];
  let beta_zero_curl = assemble_3d_curl_source_free(
    config.material.as_ref(),
    &layout,
    &reduced_index_by_full,
    &elements,
    &zero_state,
  )?;
  let metric = coords.to_edge_lengths(topology);
  let weight_material = Arc::clone(&config.material);
  let weight = InnerProductWeightClosure::new(move |point| {
    weight_material.linear_reference_reluctivity(coord_ref_to_point3(point))
  });
  let linear_hodge = build_reduced_weighted_hodge_laplace_1form_system(
    topology,
    &metric,
    coords,
    None,
    &weight,
    &config.boundary,
  )?;
  if linear_hodge.layout.active_dofs != layout.active_dofs {
    return Err("nonlinear 3D layout does not match reduced Hodge-Laplacian layout".to_string());
  }
  let hodge_coulomb_operator = sparse_difference(&linear_hodge.operator, &beta_zero_curl.jacobian)?;
  let hodge_coulomb_bias = linear_hodge
    .residual_bias
    .iter()
    .zip(beta_zero_curl.residual.iter())
    .map(|(linear, curl)| linear - curl)
    .collect::<Vec<_>>();

  Ok(ReducedVectorPotentialMagnetostatic3d {
    material: config.material,
    layout,
    reduced_index_by_full,
    source,
    elements,
    boundary_edge_dofs,
    hodge_coulomb_operator,
    hodge_coulomb_bias,
    state_mass: linear_hodge.state_mass,
    state_mass_inverse: linear_hodge.state_mass_inverse,
  })
}

#[derive(Debug, Clone, PartialEq)]
struct TriangleElement {
  vertices: [usize; 3],
  area: f64,
  barycenter: [f64; 3],
  gradients: [[f64; 2]; 3],
}

#[derive(Debug, Clone, PartialEq)]
struct TetrahedronElement {
  edges: [usize; 6],
  volume: f64,
  barycenter: [f64; 3],
  values: [[f64; 3]; 6],
  curls: [[f64; 3]; 6],
}

impl TetrahedronElement {
  fn from_cell(cell: SimplexHandle<'_>, coords: &MeshCoords) -> Result<Self, String> {
    let cell_coords = cell.coord_simplex(coords);
    if cell.vertices.len() != 4 {
      return Err(
        "3D vector-potential magnetostatic assembly expects tetrahedral cells".to_string(),
      );
    }
    let volume = cell_coords.vol();
    if !volume.is_finite() || volume <= 1e-14 {
      return Err(
        "3D vector-potential magnetostatic assembly encountered a degenerate tetrahedron"
          .to_string(),
      );
    }

    let mut edges = Vec::with_capacity(6);
    let mut values = Vec::with_capacity(6);
    let mut curls = Vec::with_capacity(6);
    let local_barycenter = barycenter_local(3);
    for edge in cell.mesh_subsimps(1) {
      let local_edge = edge.relative_to(&cell);
      let lsf = WhitneyLsf::standard(3, local_edge);
      let one_form = cell_coords.lift_form(&lsf.at_point(local_barycenter.as_view()));
      let one_coeffs = one_form.coeffs();
      if one_coeffs.len() != 3 {
        return Err(format!(
          "expected three 1-form coefficients for a 3D Whitney value, found {}",
          one_coeffs.len()
        ));
      }
      let two_form = cell_coords.lift_form(&lsf.dif());
      let coeffs = two_form.coeffs();
      if coeffs.len() != 3 {
        return Err(format!(
          "expected three 2-form coefficients for a 3D Whitney curl, found {}",
          coeffs.len()
        ));
      }
      edges.push(edge.kidx());
      values.push(one_form_coeffs_to_vector(one_coeffs));
      curls.push(two_form_coeffs_to_vector(coeffs));
    }

    Ok(Self {
      edges: edges
        .try_into()
        .map_err(|_| "tetrahedral cells must have six edge dofs".to_string())?,
      volume,
      barycenter: coord_to_point3(cell_coords.barycenter()),
      values: values
        .try_into()
        .map_err(|_| "tetrahedral cells must have six Whitney 1-form values".to_string())?,
      curls: curls
        .try_into()
        .map_err(|_| "tetrahedral cells must have six Whitney 1-form curls".to_string())?,
    })
  }

  fn magnetic_flux(&self, full_state: &[f64]) -> [f64; 3] {
    let mut out = [0.0, 0.0, 0.0];
    for local in 0..6 {
      let value = full_state[self.edges[local]];
      out[0] += value * self.curls[local][0];
      out[1] += value * self.curls[local][1];
      out[2] += value * self.curls[local][2];
    }
    out
  }
}

impl TriangleElement {
  fn from_cell_vertices(vertices: Vec<usize>, coords: &MeshCoords) -> Result<Self, String> {
    let vertices: [usize; 3] = vertices
      .try_into()
      .map_err(|_| "2D scalar A_z magnetostatic assembly expects triangular cells".to_string())?;
    for vertex in vertices {
      if vertex >= coords.nvertices() {
        return Err(format!(
          "triangle references vertex {vertex} outside coordinate vertex count {}",
          coords.nvertices()
        ));
      }
    }

    let p0 = coords.coord(vertices[0]);
    let p1 = coords.coord(vertices[1]);
    let p2 = coords.coord(vertices[2]);
    let x0 = p0[0];
    let y0 = p0[1];
    let x1 = p1[0];
    let y1 = p1[1];
    let x2 = p2[0];
    let y2 = p2[1];
    let twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if !twice_area.is_finite() || twice_area.abs() <= 1e-14 {
      return Err(
        "2D scalar A_z magnetostatic assembly encountered a degenerate triangle".to_string(),
      );
    }
    let area = 0.5 * twice_area.abs();
    let gradients = [
      [(y1 - y2) / twice_area, (x2 - x1) / twice_area],
      [(y2 - y0) / twice_area, (x0 - x2) / twice_area],
      [(y0 - y1) / twice_area, (x1 - x0) / twice_area],
    ];
    Ok(Self {
      vertices,
      area,
      barycenter: [(x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0, 0.0],
      gradients,
    })
  }

  fn gradient_of(&self, full_state: &[f64]) -> [f64; 2] {
    let mut gradient = [0.0, 0.0];
    for local in 0..3 {
      let value = full_state[self.vertices[local]];
      gradient[0] += value * self.gradients[local][0];
      gradient[1] += value * self.gradients[local][1];
    }
    gradient
  }
}

fn dot2(lhs: [f64; 2], rhs: [f64; 2]) -> f64 {
  lhs[0] * rhs[0] + lhs[1] * rhs[1]
}

fn dot3(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
  lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn bilinear2(lhs: [f64; 2], matrix: [[f64; 2]; 2], rhs: [f64; 2]) -> f64 {
  lhs[0] * (matrix[0][0] * rhs[0] + matrix[0][1] * rhs[1])
    + lhs[1] * (matrix[1][0] * rhs[0] + matrix[1][1] * rhs[1])
}

fn bilinear3(lhs: [f64; 3], matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> f64 {
  lhs[0] * (matrix[0][0] * rhs[0] + matrix[0][1] * rhs[1] + matrix[0][2] * rhs[2])
    + lhs[1] * (matrix[1][0] * rhs[0] + matrix[1][1] * rhs[1] + matrix[1][2] * rhs[2])
    + lhs[2] * (matrix[2][0] * rhs[0] + matrix[2][1] * rhs[1] + matrix[2][2] * rhs[2])
}

fn one_form_coeffs_to_vector(coeffs: &common::linalg::nalgebra::Vector) -> [f64; 3] {
  [coeffs[0], coeffs[1], coeffs[2]]
}

fn two_form_coeffs_to_vector(coeffs: &common::linalg::nalgebra::Vector) -> [f64; 3] {
  [coeffs[2], -coeffs[1], coeffs[0]]
}

fn coord_to_point3(coord: common::linalg::nalgebra::Vector) -> [f64; 3] {
  let mut point = [0.0, 0.0, 0.0];
  for index in 0..coord.len().min(3) {
    point[index] = coord[index];
  }
  point
}

fn coord_ref_to_point3(coord: CoordRef<'_>) -> [f64; 3] {
  let mut point = [0.0, 0.0, 0.0];
  for index in 0..coord.len().min(3) {
    point[index] = coord[index];
  }
  point
}

fn sparse_difference(lhs: &CsrMatrix, rhs: &CsrMatrix) -> Result<CsrMatrix, String> {
  if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
    return Err(format!(
      "sparse difference dimension mismatch: {}x{} vs {}x{}",
      lhs.nrows(),
      lhs.ncols(),
      rhs.nrows(),
      rhs.ncols()
    ));
  }
  let mut values = BTreeMap::<(usize, usize), f64>::new();
  for (row, col, value) in lhs.triplet_iter() {
    *values.entry((row, col)).or_insert(0.0) += *value;
  }
  for (row, col, value) in rhs.triplet_iter() {
    *values.entry((row, col)).or_insert(0.0) -= *value;
  }
  Ok(csr_from_triplets(
    lhs.nrows(),
    lhs.ncols(),
    values
      .into_iter()
      .filter(|(_, value)| value.abs() > 1e-14)
      .map(|((row, col), value)| (row, col, value)),
  ))
}

fn sparse_mul_vec(matrix: &CsrMatrix, vector: &[f64]) -> Result<Vec<f64>, String> {
  if matrix.ncols() != vector.len() {
    return Err(format!(
      "sparse matrix-vector dimension mismatch: matrix has {} columns but vector has length {}",
      matrix.ncols(),
      vector.len()
    ));
  }
  let mut out = vec![0.0; matrix.nrows()];
  for (row, col, value) in matrix.triplet_iter() {
    out[row] += *value * vector[col];
  }
  Ok(out)
}

fn sparse_rows(matrix: &CsrMatrix) -> Vec<Vec<(usize, f64)>> {
  let mut rows = vec![Vec::new(); matrix.nrows()];
  for (row, col, value) in matrix.triplet_iter() {
    rows[row].push((col, *value));
  }
  rows
}

fn sparse_matmul(lhs: &CsrMatrix, rhs: &CsrMatrix) -> Result<CsrMatrix, String> {
  if lhs.ncols() != rhs.nrows() {
    return Err(format!(
      "sparse matrix product dimension mismatch: lhs is {}x{}, rhs is {}x{}",
      lhs.nrows(),
      lhs.ncols(),
      rhs.nrows(),
      rhs.ncols()
    ));
  }
  let rhs_rows = sparse_rows(rhs);
  let mut values = BTreeMap::<(usize, usize), f64>::new();
  for (lhs_row, lhs_col, lhs_value) in lhs.triplet_iter() {
    for (rhs_col, rhs_value) in &rhs_rows[lhs_col] {
      *values.entry((lhs_row, *rhs_col)).or_insert(0.0) += *lhs_value * rhs_value;
    }
  }
  Ok(csr_from_triplets(
    lhs.nrows(),
    rhs.ncols(),
    values
      .into_iter()
      .filter(|(_, value)| value.abs() > 1e-14)
      .map(|((row, col), value)| (row, col, value)),
  ))
}

fn csr_from_triplets(
  nrows: usize,
  ncols: usize,
  triplets: impl IntoIterator<Item = (usize, usize, f64)>,
) -> CsrMatrix {
  let mut coo = CooMatrix::new(nrows, ncols);
  for (row, col, value) in triplets {
    if value != 0.0 {
      coo.push(row, col, value);
    }
  }
  CsrMatrix::from(&coo)
}

fn zero_csr(nrows: usize, ncols: usize) -> CsrMatrix {
  CsrMatrix::from(&CooMatrix::new(nrows, ncols))
}

fn add_csr(lhs: &CsrMatrix, rhs: &CsrMatrix) -> CsrMatrix {
  let mut coo = CooMatrix::from(lhs);
  for (row, col, value) in rhs.triplet_iter() {
    coo.push(row, col, *value);
  }
  CsrMatrix::from(&coo)
}

fn ensure_no_auxiliary_regions(boundary: &EssentialBoundarySpec) -> Result<(), String> {
  if !boundary.auxiliary.is_empty() {
    return Err(
      "nonlinear magnetostatic assembly does not support auxiliary boundary regions".to_string(),
    );
  }
  Ok(())
}

fn build_3d_edge_layout(
  topology: &Complex,
  prescribed: &[PrescribedDof],
) -> Result<(DofLayout, Vec<usize>), String> {
  let edge_count = topology.nsimplices(1);
  let layout = DofLayout::from_prescribed(edge_count, prescribed.to_vec())?;
  let boundary_edge_dofs = layout
    .prescribed_dofs
    .iter()
    .map(|entry| entry.index)
    .collect::<Vec<_>>();
  Ok((layout, boundary_edge_dofs))
}

fn build_state_layout(
  full_dimension: usize,
  prescribed: &[PrescribedDof],
) -> Result<DofLayout, String> {
  DofLayout::from_prescribed(full_dimension, prescribed.to_vec())
}

fn reduced_index_map(layout: &DofLayout) -> Vec<Option<usize>> {
  let mut map = vec![None; layout.full_dimension];
  for (reduced, full) in layout.active_dofs.iter().copied().enumerate() {
    map[full] = Some(reduced);
  }
  map
}

fn lift_reduced_with_layout(layout: &DofLayout, reduced_state: &[f64]) -> Result<Vec<f64>, String> {
  if reduced_state.len() != layout.reduced_dimension() {
    return Err(format!(
      "reduced vector length {} does not match layout active dimension {}",
      reduced_state.len(),
      layout.reduced_dimension()
    ));
  }
  let mut full = vec![0.0; layout.full_dimension];
  for fixed in &layout.prescribed_dofs {
    full[fixed.index] = fixed.value;
  }
  for (reduced, full_index) in layout.active_dofs.iter().copied().enumerate() {
    full[full_index] = reduced_state[reduced];
  }
  Ok(full)
}

fn lift_active_to_full_zero_fixed(
  layout: &DofLayout,
  reduced_state: &[f64],
) -> Result<Vec<f64>, String> {
  if reduced_state.len() != layout.reduced_dimension() {
    return Err(format!(
      "reduced vector length {} does not match layout active dimension {}",
      reduced_state.len(),
      layout.reduced_dimension()
    ));
  }
  let mut full = vec![0.0; layout.full_dimension];
  for (reduced, full_index) in layout.active_dofs.iter().copied().enumerate() {
    full[full_index] = reduced_state[reduced];
  }
  Ok(full)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    operators::InnerProductWeightClosure,
    problems::reduced_linear::build_reduced_weighted_hodge_laplace_1form_system,
  };
  use manifold::gen::cartesian::CartesianMeshInfo;
  use std::f64::consts::PI;

  fn essential_boundary(dofs: Vec<usize>, values: Vec<f64>) -> EssentialBoundarySpec {
    EssentialBoundarySpec::default().with_state(
      dofs
        .into_iter()
        .zip(values)
        .map(|(index, value)| PrescribedDof { index, value }),
    )
  }

  fn tiny_problem() -> (ReducedScalarAzMagnetostatic2d, Vec<f64>) {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 3, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let boundary_vertices = topology
      .boundary_subcomplex_simplices(0)
      .into_iter()
      .map(|simplex| simplex.kidx)
      .collect::<Vec<_>>();
    let boundary = essential_boundary(
      boundary_vertices.clone(),
      vec![0.0; boundary_vertices.len()],
    );
    let material = NonlinearReluctivityLaw::new(1.3, 0.4).unwrap();
    let model = build_reduced_scalar_az_magnetostatic_2d(
      &topology,
      &coords,
      NonlinearMagnetostaticAssemblyConfig::new(material, boundary),
    )
    .unwrap();
    let truth = model
      .layout()
      .active_dofs
      .iter()
      .map(|&vertex| {
        let point = coords.coord(vertex);
        (PI * point[0]).sin() * (PI * point[1]).sin()
      })
      .collect::<Vec<_>>();
    (model, truth)
  }

  fn vector_problem_3d(
    level: usize,
  ) -> (
    Complex,
    MeshCoords,
    ReducedVectorPotentialMagnetostatic3d,
    Vec<f64>,
  ) {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, level, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let material = NonlinearReluctivityLaw::new(1.0, 0.25).unwrap();
    let model = build_reduced_vector_potential_magnetostatic_3d(
      &topology,
      &coords,
      NonlinearMagnetostaticAssemblyConfig::new(material, EssentialBoundarySpec::default()),
    )
    .unwrap();
    let truth = smooth_edge_truth_3d(&model, &topology, &coords, 0.2);
    (topology, coords, model, truth)
  }

  fn smooth_edge_truth_3d(
    model: &ReducedVectorPotentialMagnetostatic3d,
    topology: &Complex,
    coords: &MeshCoords,
    scale: f64,
  ) -> Vec<f64> {
    let mut full = vec![0.0; topology.nsimplices(1)];
    for edge in topology.edges().handle_iter() {
      let [v0, v1]: [usize; 2] = edge.vertices.clone().try_into().unwrap();
      let p0 = coords.coord(v0);
      let p1 = coords.coord(v1);
      let midpoint = [
        0.5 * (p0[0] + p1[0]),
        0.5 * (p0[1] + p1[1]),
        0.5 * (p0[2] + p1[2]),
      ];
      let tangent = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
      let vector_potential = smooth_vector_potential_3d(midpoint, scale);
      full[edge.kidx()] = dot3(vector_potential, tangent);
    }
    model
      .layout()
      .active_dofs
      .iter()
      .map(|&edge| full[edge])
      .collect()
  }

  fn smooth_vector_potential_3d(point: [f64; 3], scale: f64) -> [f64; 3] {
    [
      scale * (PI * point[1]).sin() * (PI * point[2]).sin(),
      scale * (PI * point[2]).sin() * (PI * point[0]).sin(),
      scale * (PI * point[0]).sin() * (PI * point[1]).sin(),
    ]
  }

  fn matrix_entry(matrix: &CsrMatrix, row: usize, col: usize) -> f64 {
    matrix
      .triplet_iter()
      .filter(|(r, c, _)| *r == row && *c == col)
      .map(|(_, _, value)| *value)
      .sum()
  }

  #[test]
  fn nonlinear_reluctivity_derivative_matches_finite_differences() {
    let law = NonlinearReluctivityLaw::new(2.0, 0.7).unwrap();
    let s = 1.4;
    let eps = 1e-6;
    let finite_difference = (law.nu(s + eps) - law.nu(s - eps)) / (2.0 * eps);
    assert!((finite_difference - law.d_nu_ds(s)).abs() <= 1e-9);
  }

  #[test]
  fn manufactured_truth_has_zero_residual() {
    let (model, truth) = tiny_problem();
    let source = model.manufactured_source(&truth).unwrap();
    let model = model.with_source(source).unwrap();
    let residual = model.residual_and_jacobian(&truth).unwrap().residual;
    let norm = residual
      .iter()
      .map(|value| value * value)
      .sum::<f64>()
      .sqrt();
    assert!(norm <= 1e-11, "manufactured residual norm was {norm}");
  }

  #[test]
  fn sparse_jacobian_matches_finite_differences() {
    let (model, truth) = tiny_problem();
    let source = model.manufactured_source(&truth).unwrap();
    let model = model.with_source(source).unwrap();
    let evaluation = model.residual_and_jacobian(&truth).unwrap();
    let eps = 1e-6;

    for col in 0..model.reduced_dimension() {
      let mut plus = truth.clone();
      let mut minus = truth.clone();
      plus[col] += eps;
      minus[col] -= eps;
      let plus_residual = model.residual_and_jacobian(&plus).unwrap().residual;
      let minus_residual = model.residual_and_jacobian(&minus).unwrap().residual;
      for row in 0..model.reduced_dimension() {
        let finite_difference = (plus_residual[row] - minus_residual[row]) / (2.0 * eps);
        let assembled = matrix_entry(&evaluation.jacobian, row, col);
        assert!(
          (assembled - finite_difference).abs() <= 2e-5,
          "Jacobian mismatch at ({row}, {col}): assembled {assembled}, finite difference {finite_difference}"
        );
      }
    }
  }

  #[test]
  fn nonlinear_magnetostatic_3d_layout_uses_only_supplied_hard_edges() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let mut supplied_edges = topology
      .boundary_subcomplex_simplices(1)
      .into_iter()
      .take(3)
      .map(|simplex| simplex.kidx)
      .collect::<Vec<_>>();
    supplied_edges.sort_unstable();
    let boundary = essential_boundary(supplied_edges.clone(), vec![0.0; supplied_edges.len()]);
    let material = NonlinearReluctivityLaw::new(1.0, 0.25).unwrap();
    let model = build_reduced_vector_potential_magnetostatic_3d(
      &topology,
      &coords,
      NonlinearMagnetostaticAssemblyConfig::new(material, boundary),
    )
    .unwrap();
    assert_eq!(model.boundary_edge_dofs(), supplied_edges.as_slice());
    assert!(model.gauge_edge_dofs().is_empty());
    assert!(model.reduced_dimension() > 0);
    assert_eq!(
      model.reduced_dimension(),
      topology.nsimplices(1) - supplied_edges.len()
    );
  }

  #[test]
  fn nonlinear_magnetostatic_3d_manufactured_truth_has_zero_residual() {
    let (_topology, _coords, model, truth) = vector_problem_3d(2);
    let source = model.manufactured_source(&truth).unwrap();
    let model = model.with_source(source).unwrap();
    let residual = model.residual_and_jacobian(&truth).unwrap().residual;
    let norm = residual
      .iter()
      .map(|value| value * value)
      .sum::<f64>()
      .sqrt();
    assert!(norm <= 1e-10, "3D manufactured residual norm was {norm}");
  }

  #[test]
  fn nonlinear_magnetostatic_3d_ungauged_model_preserves_source_and_residual() {
    let (_topology, _coords, model, truth) = vector_problem_3d(1);
    let ungauged = model.clone().without_coulomb_gauge();
    let source = ungauged.manufactured_source(&truth).unwrap();
    let ungauged = ungauged.with_source(source.clone()).unwrap();
    assert_eq!(ungauged.source(), source.as_slice());

    let evaluation = ungauged.residual_and_jacobian(&truth).unwrap();
    let residual_norm = evaluation
      .residual
      .iter()
      .map(|value| value * value)
      .sum::<f64>()
      .sqrt();
    assert!(
      residual_norm <= 1e-10,
      "ungauged manufactured residual norm was {residual_norm}"
    );
    assert!(evaluation.jacobian.nnz() > 0);
    assert!(evaluation
      .jacobian
      .triplet_iter()
      .all(|(_, _, value)| value.is_finite()));
  }

  #[test]
  fn nonlinear_magnetostatic_3d_sparse_jacobian_matches_finite_differences() {
    let (_topology, _coords, model, truth) = vector_problem_3d(1);
    let source = model.manufactured_source(&truth).unwrap();
    let model = model.with_source(source).unwrap();
    let evaluation = model.residual_and_jacobian(&truth).unwrap();
    let eps = 1e-6;

    for col in 0..model.reduced_dimension() {
      let mut plus = truth.clone();
      let mut minus = truth.clone();
      plus[col] += eps;
      minus[col] -= eps;
      let plus_residual = model.residual_and_jacobian(&plus).unwrap().residual;
      let minus_residual = model.residual_and_jacobian(&minus).unwrap().residual;
      for row in 0..model.reduced_dimension() {
        let finite_difference = (plus_residual[row] - minus_residual[row]) / (2.0 * eps);
        let assembled = matrix_entry(&evaluation.jacobian, row, col);
        assert!(
          (assembled - finite_difference).abs() <= 2e-5,
          "3D Jacobian mismatch at ({row}, {col}): assembled {assembled}, finite difference {finite_difference}"
        );
      }
    }
  }

  #[test]
  fn local_magnetic_strong_probe_dimensions_are_selected_cells_by_active_edges() {
    let (topology, _coords, model, _truth) = vector_problem_3d(1);
    let selected_cells = vec![0, topology.nsimplices(3) - 1];
    let probe =
      LocalMagneticStrongProbe3d::from_vector_potential_model(&model, selected_cells).unwrap();
    assert_eq!(probe.state_dimension(), model.reduced_dimension());
    assert_eq!(probe.residual_dimension(), 6);

    let evaluation = probe
      .residual_and_jacobian(&vec![0.0; model.reduced_dimension()])
      .unwrap();
    assert_eq!(evaluation.residual.len(), 6);
    assert_eq!(evaluation.jacobian.nrows(), 6);
    assert_eq!(evaluation.jacobian.ncols(), model.reduced_dimension());
  }

  #[test]
  fn local_magnetic_strong_probe_jacobian_matches_finite_differences() {
    let (_topology, _coords, model, truth) = vector_problem_3d(1);
    let selected_cells = (0..model.num_elements()).collect::<Vec<_>>();
    let probe =
      LocalMagneticStrongProbe3d::from_vector_potential_model(&model, selected_cells).unwrap();
    let evaluation = probe.residual_and_jacobian(&truth).unwrap();
    let eps = 1e-6;

    for col in 0..model.reduced_dimension() {
      let mut plus = truth.clone();
      let mut minus = truth.clone();
      plus[col] += eps;
      minus[col] -= eps;
      let plus_residual = probe.residual_and_jacobian(&plus).unwrap().residual;
      let minus_residual = probe.residual_and_jacobian(&minus).unwrap().residual;
      for row in 0..probe.residual_dimension() {
        let finite_difference = (plus_residual[row] - minus_residual[row]) / (2.0 * eps);
        let assembled = matrix_entry(&evaluation.jacobian, row, col);
        assert!(
          (assembled - finite_difference).abs() <= 2e-5,
          "local strong-probe Jacobian mismatch at ({row}, {col}): assembled {assembled}, finite difference {finite_difference}"
        );
      }
    }
  }

  #[test]
  fn local_magnetic_strong_probe_zero_field_no_source_has_small_curl_residual() {
    let (_topology, _coords, model, _truth) = vector_problem_3d(1);
    let selected_cells = (0..model.num_elements()).collect::<Vec<_>>();
    let probe =
      LocalMagneticStrongProbe3d::from_vector_potential_model(&model, selected_cells).unwrap();
    let state = vec![0.0; model.reduced_dimension()];
    let residual = probe.residual_and_jacobian(&state).unwrap().residual;
    let norm = residual
      .iter()
      .map(|value| value * value)
      .sum::<f64>()
      .sqrt();
    assert!(norm <= 1e-12, "zero-field local probe residual was {norm}");
  }

  #[test]
  fn local_magnetic_strong_probe_beta_zero_tangent_is_finite_and_sparse() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let material = NonlinearReluctivityLaw::new(1.7, 0.0).unwrap();
    let model = build_reduced_vector_potential_magnetostatic_3d(
      &topology,
      &coords,
      NonlinearMagnetostaticAssemblyConfig::new(material, EssentialBoundarySpec::default()),
    )
    .unwrap();
    let probe = LocalMagneticStrongProbe3d::from_vector_potential_model(&model, vec![0]).unwrap();
    let evaluation = probe
      .residual_and_jacobian(&vec![0.0; model.reduced_dimension()])
      .unwrap();
    assert!(evaluation.residual.iter().all(|value| value.is_finite()));
    assert!(evaluation.jacobian.nnz() > 0);
    assert!(evaluation
      .jacobian
      .triplet_iter()
      .all(|(_, _, value)| value.is_finite()));
  }

  #[test]
  fn nonlinear_beta_zero_jacobian_matches_reduced_hodge_laplacian() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let material = NonlinearReluctivityLaw::new(2.5, 0.0).unwrap();
    let boundary = EssentialBoundarySpec::default();
    let model = build_reduced_vector_potential_magnetostatic_3d(
      &topology,
      &coords,
      NonlinearMagnetostaticAssemblyConfig::new(material, boundary.clone()),
    )
    .unwrap();
    let evaluation = model
      .source_free_residual_and_jacobian(&vec![0.0; model.reduced_dimension()])
      .unwrap();

    let metric = coords.to_edge_lengths(&topology);
    let weight = InnerProductWeightClosure::new(|_| 2.5);
    let linear = build_reduced_weighted_hodge_laplace_1form_system(
      &topology, &metric, &coords, None, &weight, &boundary,
    )
    .unwrap();
    assert_eq!(model.layout().active_dofs, linear.layout.active_dofs);
    for row_reduced in 0..model.reduced_dimension() {
      for col_reduced in 0..model.reduced_dimension() {
        let assembled = matrix_entry(&evaluation.jacobian, row_reduced, col_reduced);
        let expected = matrix_entry(&linear.operator, row_reduced, col_reduced);
        assert!(
          (assembled - expected).abs() <= 1e-9,
          "linear Hodge-Laplacian mismatch at reduced ({row_reduced}, {col_reduced}): assembled {assembled}, expected {expected}"
        );
      }
    }
  }
}
