use crate::problems::residual::{
  ResidualEvaluation as NonlinearResidualEvaluation, ResidualModel as NonlinearResidualModel,
};
use crate::{
  assemble::{
    assemble_galmat, assemble_galmat_coord_aware, assemble_galvec,
    assemble_whitney_projected_sparse_inverse_galmat,
  },
  operators::{HodgeMassElmat, InnerProductWeightClosure, SourceElVec},
  reduction::{DofLayout, EssentialBoundarySpec, PrescribedDof},
};
use common::linalg::nalgebra::{CooMatrix, CsrMatrix, Vector};
use ddf::whitney::lsf::WhitneyLsf;
use ddf::ManifoldComplexExt;
use exterior::field::{DiffFormClosure, ExteriorField};
use manifold::{
  geometry::{
    coord::{
      mesh::MeshCoords,
      quadrature::SimplexQuadRule,
      simplex::{barycenter_local, SimplexHandleExt},
    },
    metric::mesh::MeshLengths,
  },
  topology::{complex::Complex, handle::SimplexHandle},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub struct ReducedEddyCurrent1FormSystem {
  pub curl_curl: CsrMatrix,
  pub conductivity_mass: CsrMatrix,
  pub state_mass: CsrMatrix,
  pub state_mass_inverse: CsrMatrix,
  pub layout: DofLayout,
  pub curl_curl_fixed_bias: Vector,
  pub conductivity_fixed_bias: Vector,
}

impl ReducedEddyCurrent1FormSystem {
  pub fn reduced_dimension(&self) -> usize {
    self.layout.reduced_dimension()
  }
}

pub fn build_reduced_eddy_current_1form_system(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  inverse_permeability: &InnerProductWeightClosure<f64>,
  conductivity: &InnerProductWeightClosure<f64>,
  boundary: &EssentialBoundarySpec,
) -> Result<ReducedEddyCurrent1FormSystem, String> {
  ensure_no_auxiliary_regions(boundary)?;
  let dim = topology.dim();
  if dim < 2 {
    return Err("eddy-current 1-form system requires topology dimension at least 2".to_string());
  }
  if coords.dim() != dim {
    return Err(format!(
      "eddy-current assembly requires matching intrinsic/ambient dimensions, got topology dim {dim} and coordinate dim {}",
      coords.dim()
    ));
  }

  let curl_curl =
    assemble_weighted_curl_curl_1form(topology, geometry, coords, qr.clone(), inverse_permeability);
  let conductivity_mass = assemble_galmat_coord_aware(
    topology,
    geometry,
    HodgeMassElmat::new_weighted(dim, 1, coords, qr, conductivity),
  );
  let state_mass = assemble_galmat(topology, geometry, HodgeMassElmat::new(dim, 1));
  let state_mass_inverse = CsrMatrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
    topology, geometry,
  ));
  let layout = build_state_layout(curl_curl.nrows(), &boundary.state)?;

  let conductivity_mass = CsrMatrix::from(&conductivity_mass);
  let state_mass = CsrMatrix::from(&state_mass);

  let reduced_curl_curl = reduce_square_with_layout(&curl_curl, &layout)?;
  let reduced_conductivity_mass = reduce_square_with_layout(&conductivity_mass, &layout)?;
  let reduced_state_mass = reduce_square_with_layout(&state_mass, &layout)?;
  let reduced_state_mass_inverse = reduce_square_with_layout(&state_mass_inverse, &layout)?;

  Ok(ReducedEddyCurrent1FormSystem {
    curl_curl: reduced_curl_curl,
    conductivity_mass: reduced_conductivity_mass,
    state_mass: reduced_state_mass,
    state_mass_inverse: reduced_state_mass_inverse,
    layout: layout.clone(),
    curl_curl_fixed_bias: hard_dirichlet_bias(&curl_curl, &layout),
    conductivity_fixed_bias: hard_dirichlet_bias(&conductivity_mass, &layout),
  })
}

pub fn assemble_weighted_curl_curl_1form(
  topology: &Complex,
  geometry: &MeshLengths,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  inverse_permeability: &InnerProductWeightClosure<f64>,
) -> CsrMatrix {
  let dim = topology.dim();
  let mass_2 = assemble_galmat_coord_aware(
    topology,
    geometry,
    HodgeMassElmat::new_weighted(dim, 2, coords, qr, inverse_permeability),
  );
  let mass_2 = CsrMatrix::from(&mass_2);
  let d1 = CsrMatrix::from(&topology.exterior_derivative_operator(1));
  d1.transpose() * mass_2 * d1
}

pub fn reduce_eddy_current_source(layout: &DofLayout, source: &Vector) -> Result<Vector, String> {
  reduce_vector_with_layout(layout, source)
}

pub fn screened_eddy_current_sinusoidal_source_value(
  point: [f64; 3],
  amplitude: f64,
) -> Result<[f64; 3], String> {
  if !amplitude.is_finite() || amplitude <= 0.0 {
    return Err("screened eddy-current source amplitude must be finite and positive".to_string());
  }
  let pi = std::f64::consts::PI;
  Ok([
    amplitude * (pi * point[1]).sin() * (pi * point[2]).sin(),
    amplitude * (pi * point[2]).sin() * (pi * point[0]).sin(),
    amplitude * (pi * point[0]).sin() * (pi * point[1]).sin(),
  ])
}

pub fn screened_eddy_current_sinusoidal_source(amplitude: f64) -> Result<DiffFormClosure, String> {
  if !amplitude.is_finite() || amplitude <= 0.0 {
    return Err("screened eddy-current source amplitude must be finite and positive".to_string());
  }
  Ok(DiffFormClosure::one_form(
    move |point| {
      let value =
        screened_eddy_current_sinusoidal_source_value([point[0], point[1], point[2]], amplitude)
          .expect("validated source amplitude should remain valid");
      Vector::from_column_slice(&value)
    },
    3,
  ))
}

pub fn assemble_reduced_eddy_current_1form_source<F>(
  topology: &Complex,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  boundary: &EssentialBoundarySpec,
  source: &F,
) -> Result<Vec<f64>, String>
where
  F: ExteriorField + Sync,
{
  ensure_no_auxiliary_regions(boundary)?;
  if topology.dim() != 3 {
    return Err(format!(
      "screened eddy-current source assembly requires topology dimension 3, got {}",
      topology.dim()
    ));
  }
  if coords.dim() != 3 {
    return Err(format!(
      "screened eddy-current source assembly requires coordinate dimension 3, got {}",
      coords.dim()
    ));
  }
  if source.grade() != 1 {
    return Err(format!(
      "screened eddy-current source must be a 1-form, got grade {}",
      source.grade()
    ));
  }
  if source.dim_ambient() != 3 {
    return Err(format!(
      "screened eddy-current source ambient dimension must be 3, got {}",
      source.dim_ambient()
    ));
  }
  let metric = coords.to_edge_lengths(topology);
  let full_source = assemble_galvec(topology, &metric, SourceElVec::new(source, coords, qr));
  let layout = build_state_layout(topology.nsimplices(1), &boundary.state)?;
  reduce_vector_with_layout(&layout, &full_source).map(|source| source.as_slice().to_vec())
}

pub fn assemble_reduced_screened_eddy_current_sinusoidal_source(
  topology: &Complex,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  boundary: &EssentialBoundarySpec,
  amplitude: f64,
) -> Result<Vec<f64>, String> {
  let source = screened_eddy_current_sinusoidal_source(amplitude)?;
  assemble_reduced_eddy_current_1form_source(topology, coords, qr, boundary, &source)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearEddyCurrentReluctivityLaw {
  pub nu0: f64,
  pub beta: f64,
}

impl NonlinearEddyCurrentReluctivityLaw {
  pub fn new(nu0: f64, beta: f64) -> Result<Self, String> {
    let law = Self { nu0, beta };
    law.validate()?;
    Ok(law)
  }

  pub fn validate(&self) -> Result<(), String> {
    if !self.nu0.is_finite() || self.nu0 <= 0.0 {
      return Err("nonlinear eddy-current reluctivity nu0 must be finite and positive".to_string());
    }
    if !self.beta.is_finite() || self.beta < 0.0 {
      return Err(
        "nonlinear eddy-current reluctivity beta must be finite and nonnegative".to_string(),
      );
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

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearScreenedEddyCurrentAssemblyConfig {
  pub reluctivity: NonlinearEddyCurrentReluctivityLaw,
  pub sigma: f64,
  pub source: Option<Vec<f64>>,
  pub boundary: EssentialBoundarySpec,
}

impl NonlinearScreenedEddyCurrentAssemblyConfig {
  pub fn new(
    reluctivity: NonlinearEddyCurrentReluctivityLaw,
    sigma: f64,
    boundary: EssentialBoundarySpec,
  ) -> Self {
    Self {
      reluctivity,
      sigma,
      source: None,
      boundary,
    }
  }

  pub fn with_source(mut self, source: impl AsRef<[f64]>) -> Self {
    self.source = Some(source.as_ref().to_vec());
    self
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReducedNonlinearScreenedEddyCurrent1Form {
  reluctivity: NonlinearEddyCurrentReluctivityLaw,
  sigma: f64,
  layout: DofLayout,
  reduced_index_by_full: Vec<Option<usize>>,
  source: Vec<f64>,
  elements: Vec<EddyCurrentTetrahedronElement>,
  conductivity_mass: CsrMatrix,
  conductivity_bias: Vector,
  state_mass: CsrMatrix,
  state_mass_inverse: CsrMatrix,
  beta_zero_operator: CsrMatrix,
  beta_zero_bias: Vector,
}

impl ReducedNonlinearScreenedEddyCurrent1Form {
  pub fn reduced_dimension(&self) -> usize {
    self.layout.reduced_dimension()
  }

  pub fn layout(&self) -> &DofLayout {
    &self.layout
  }

  pub fn reluctivity(&self) -> NonlinearEddyCurrentReluctivityLaw {
    self.reluctivity
  }

  pub fn sigma(&self) -> f64 {
    self.sigma
  }

  pub fn source(&self) -> &[f64] {
    &self.source
  }

  pub fn num_elements(&self) -> usize {
    self.elements.len()
  }

  pub fn state_mass(&self) -> &CsrMatrix {
    &self.state_mass
  }

  pub fn state_mass_inverse(&self) -> &CsrMatrix {
    &self.state_mass_inverse
  }

  pub fn beta_zero_operator(&self) -> &CsrMatrix {
    &self.beta_zero_operator
  }

  pub fn beta_zero_bias(&self) -> &[f64] {
    self.beta_zero_bias.as_slice()
  }

  pub fn with_source(mut self, source: impl AsRef<[f64]>) -> Result<Self, String> {
    let source = source.as_ref().to_vec();
    validate_reduced_source(
      "nonlinear screened eddy-current",
      self.reduced_dimension(),
      &source,
    )?;
    self.source = source;
    Ok(self)
  }

  pub fn manufactured_source(&self, reduced_state: &[f64]) -> Result<Vector, String> {
    self
      .source_free_residual_and_jacobian(reduced_state)
      .map(|evaluation| evaluation.residual)
  }

  pub fn source_free_residual_and_jacobian(
    &self,
    reduced_state: &[f64],
  ) -> Result<NonlinearResidualEvaluation, String> {
    if reduced_state.len() != self.reduced_dimension() {
      return Err(format!(
        "nonlinear screened eddy-current state length {} must match active dimension {}",
        reduced_state.len(),
        self.reduced_dimension()
      ));
    }
    if !reduced_state.iter().all(|value| value.is_finite()) {
      return Err(
        "nonlinear screened eddy-current state must contain only finite values".to_string(),
      );
    }

    let mut evaluation = assemble_screened_eddy_current_curl_source_free(
      self.reluctivity,
      &self.layout,
      &self.reduced_index_by_full,
      &self.elements,
      reduced_state,
    )?;
    add_sparse_affine_term(
      &mut evaluation.residual,
      &mut evaluation.jacobian,
      &self.conductivity_mass,
      self.conductivity_bias.as_slice(),
      reduced_state,
    )?;
    Ok(evaluation)
  }

  pub fn cell_vector_potential_operator(&self) -> CsrMatrix {
    let mut operator = CooMatrix::new(3 * self.elements.len(), self.reduced_dimension());
    for (cell_index, element) in self.elements.iter().enumerate() {
      for local in 0..6 {
        let edge = element.edges[local];
        let Some(reduced) = self.reduced_index_by_full[edge] else {
          continue;
        };
        let value = element.values[local];
        operator.push(3 * cell_index, reduced, value[0]);
        operator.push(3 * cell_index + 1, reduced, value[1]);
        operator.push(3 * cell_index + 2, reduced, value[2]);
      }
    }
    CsrMatrix::from(&operator)
  }

  pub fn cell_volumes(&self) -> Vec<f64> {
    self.elements.iter().map(|element| element.volume).collect()
  }

  pub fn lift_reduced_state(&self, reduced_state: &[f64]) -> Result<Vec<f64>, String> {
    lift_reduced_with_layout(&self.layout, reduced_state)
  }
}

impl NonlinearResidualModel for ReducedNonlinearScreenedEddyCurrent1Form {
  fn state_dimension(&self) -> usize {
    self.reduced_dimension()
  }

  fn residual_dimension(&self) -> usize {
    self.reduced_dimension()
  }

  fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
    let mut evaluation = self.source_free_residual_and_jacobian(state)?;
    for (value, source) in evaluation.residual.iter_mut().zip(self.source.iter()) {
      *value -= *source;
    }
    Ok(evaluation)
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalEddyCurrentResidualProbe3d {
  layout: DofLayout,
  reduced_index_by_full: Vec<Option<usize>>,
  weak_model: ReducedNonlinearScreenedEddyCurrent1Form,
  elements: Vec<EddyCurrentTetrahedronElement>,
  selected_cells: Vec<usize>,
  state_mass_inverse: CsrMatrix,
}

impl LocalEddyCurrentResidualProbe3d {
  pub fn from_model(
    model: &ReducedNonlinearScreenedEddyCurrent1Form,
    selected_cells: Vec<usize>,
  ) -> Result<Self, String> {
    let mut seen = BTreeSet::new();
    for cell in &selected_cells {
      if *cell >= model.elements.len() {
        return Err(format!(
          "selected local eddy-current probe cell {cell} is outside cell count {}",
          model.elements.len()
        ));
      }
      if !seen.insert(*cell) {
        return Err(format!(
          "selected local eddy-current probe cell {cell} appears more than once"
        ));
      }
    }
    Ok(Self {
      layout: model.layout.clone(),
      reduced_index_by_full: model.reduced_index_by_full.clone(),
      weak_model: model.clone(),
      elements: model.elements.clone(),
      selected_cells,
      state_mass_inverse: model.state_mass_inverse.clone(),
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
  ) -> Result<ProjectedWeakResidualEvaluation, String> {
    if reduced_state.len() != self.reduced_dimension() {
      return Err(format!(
        "local eddy-current probe state length {} must match active dimension {}",
        reduced_state.len(),
        self.reduced_dimension()
      ));
    }
    if !reduced_state.iter().all(|value| value.is_finite()) {
      return Err("local eddy-current probe state must contain only finite values".to_string());
    }

    let weak_evaluation = self.weak_model.residual_and_jacobian(reduced_state)?;
    let coefficients = sparse_mul_vec(
      &self.state_mass_inverse,
      weak_evaluation.residual.as_slice(),
    )?;
    let jacobian = sparse_matmul(&self.state_mass_inverse, &weak_evaluation.jacobian)?;
    Ok(ProjectedWeakResidualEvaluation {
      coefficients,
      jacobian,
    })
  }
}

struct ProjectedWeakResidualEvaluation {
  coefficients: Vec<f64>,
  jacobian: CsrMatrix,
}

impl NonlinearResidualModel for LocalEddyCurrentResidualProbe3d {
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

pub fn build_reduced_nonlinear_screened_eddy_current_1form(
  topology: &Complex,
  coords: &MeshCoords,
  qr: Option<SimplexQuadRule>,
  config: NonlinearScreenedEddyCurrentAssemblyConfig,
) -> Result<ReducedNonlinearScreenedEddyCurrent1Form, String> {
  config.reluctivity.validate()?;
  if !config.sigma.is_finite() || config.sigma <= 0.0 {
    return Err("nonlinear screened eddy-current sigma must be finite and positive".to_string());
  }
  ensure_no_auxiliary_regions(&config.boundary)?;
  if topology.dim() != 3 {
    return Err(format!(
      "nonlinear screened eddy-current model requires topology dimension 3, got {}",
      topology.dim()
    ));
  }
  if coords.dim() != 3 {
    return Err(format!(
      "nonlinear screened eddy-current model requires coordinate dimension 3, got {}",
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

  let metric = coords.to_edge_lengths(topology);
  let nu0 = config.reluctivity.nu0;
  let sigma = config.sigma;
  let inverse_permeability = InnerProductWeightClosure::new(move |_| nu0);
  let conductivity = InnerProductWeightClosure::new(move |_| sigma);
  let linear = build_reduced_eddy_current_1form_system(
    topology,
    &metric,
    coords,
    qr,
    &inverse_permeability,
    &conductivity,
    &config.boundary,
  )?;
  let layout = linear.layout.clone();
  let reduced_index_by_full = reduced_index_map(&layout);
  let source = match config.source {
    Some(source) => {
      validate_reduced_source(
        "nonlinear screened eddy-current",
        layout.reduced_dimension(),
        &source,
      )?;
      source
    }
    None => vec![0.0; layout.reduced_dimension()],
  };
  let elements = topology
    .cells()
    .handle_iter()
    .map(|cell| EddyCurrentTetrahedronElement::from_cell(cell, coords))
    .collect::<Result<Vec<_>, _>>()?;
  if elements.is_empty() {
    return Err(
      "nonlinear screened eddy-current model requires at least one tetrahedron".to_string(),
    );
  }

  let beta_zero_operator = sparse_sum(&linear.curl_curl, &linear.conductivity_mass)?;
  let beta_zero_bias = &linear.curl_curl_fixed_bias + &linear.conductivity_fixed_bias;

  Ok(ReducedNonlinearScreenedEddyCurrent1Form {
    reluctivity: config.reluctivity,
    sigma,
    layout,
    reduced_index_by_full,
    source,
    elements,
    conductivity_mass: linear.conductivity_mass,
    conductivity_bias: linear.conductivity_fixed_bias,
    state_mass: linear.state_mass,
    state_mass_inverse: linear.state_mass_inverse,
    beta_zero_operator,
    beta_zero_bias,
  })
}

#[derive(Debug, Clone, PartialEq)]
struct EddyCurrentTetrahedronElement {
  edges: [usize; 6],
  volume: f64,
  values: [[f64; 3]; 6],
  curls: [[f64; 3]; 6],
}

impl EddyCurrentTetrahedronElement {
  fn from_cell(cell: SimplexHandle<'_>, coords: &MeshCoords) -> Result<Self, String> {
    let cell_coords = cell.coord_simplex(coords);
    if cell.vertices.len() != 4 {
      return Err("nonlinear screened eddy-current assembly expects tetrahedral cells".to_string());
    }
    let volume = cell_coords.vol();
    if !volume.is_finite() || volume <= 1e-14 {
      return Err(
        "nonlinear screened eddy-current assembly encountered a degenerate tetrahedron".to_string(),
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
      let two_coeffs = two_form.coeffs();
      if two_coeffs.len() != 3 {
        return Err(format!(
          "expected three 2-form coefficients for a 3D Whitney curl, found {}",
          two_coeffs.len()
        ));
      }
      edges.push(edge.kidx());
      values.push(one_form_coeffs_to_vector(one_coeffs));
      curls.push(two_form_coeffs_to_vector(two_coeffs));
    }

    Ok(Self {
      edges: edges
        .try_into()
        .map_err(|_| "tetrahedral cells must have six edge dofs".to_string())?,
      volume,
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

fn assemble_screened_eddy_current_curl_source_free(
  reluctivity: NonlinearEddyCurrentReluctivityLaw,
  layout: &DofLayout,
  reduced_index_by_full: &[Option<usize>],
  elements: &[EddyCurrentTetrahedronElement],
  reduced_state: &[f64],
) -> Result<NonlinearResidualEvaluation, String> {
  let full_state = lift_reduced_with_layout(layout, reduced_state)?;
  let mut residual = Vector::zeros(layout.reduced_dimension());
  let mut jacobian_values = BTreeMap::<(usize, usize), f64>::new();

  for element in elements {
    let magnetic_flux = element.magnetic_flux(&full_state);
    let s = dot3(magnetic_flux, magnetic_flux);
    let nu = reluctivity.nu(s);
    let dnu = reluctivity.d_nu_ds(s);
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
    residual,
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

fn validate_reduced_source(name: &str, dimension: usize, source: &[f64]) -> Result<(), String> {
  if source.len() != dimension {
    return Err(format!(
      "{name} reduced source length {} must match active dimension {dimension}",
      source.len()
    ));
  }
  if !source.iter().all(|value| value.is_finite()) {
    return Err(format!(
      "{name} reduced source must contain only finite values"
    ));
  }
  Ok(())
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
    residual[row] += *value * state[col];
  }

  *jacobian = sparse_sum(jacobian, operator)?;
  Ok(())
}

fn sparse_sum(lhs: &CsrMatrix, rhs: &CsrMatrix) -> Result<CsrMatrix, String> {
  if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
    return Err(format!(
      "sparse sum dimension mismatch: {}x{} vs {}x{}",
      lhs.nrows(),
      lhs.ncols(),
      rhs.nrows(),
      rhs.ncols()
    ));
  }
  let mut values = BTreeMap::<(usize, usize), f64>::new();
  for (row, col, value) in lhs.triplet_iter().chain(rhs.triplet_iter()) {
    *values.entry((row, col)).or_insert(0.0) += *value;
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

fn dot3(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
  lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
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

fn ensure_no_auxiliary_regions(boundary: &EssentialBoundarySpec) -> Result<(), String> {
  if !boundary.auxiliary.is_empty() {
    return Err("eddy-current systems do not support auxiliary boundary regions".to_string());
  }
  Ok(())
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

fn reduce_square_with_layout(matrix: &CsrMatrix, layout: &DofLayout) -> Result<CsrMatrix, String> {
  if matrix.nrows() != layout.full_dimension || matrix.ncols() != layout.full_dimension {
    return Err(format!(
      "matrix reduction expected a {}x{} operator, got {}x{}",
      layout.full_dimension,
      layout.full_dimension,
      matrix.nrows(),
      matrix.ncols()
    ));
  }
  let reduced_map = reduced_index_map(layout);
  let mut reduced = CooMatrix::new(layout.reduced_dimension(), layout.reduced_dimension());
  for (row, col, value) in matrix.triplet_iter() {
    let Some(reduced_row) = reduced_map[row] else {
      continue;
    };
    let Some(reduced_col) = reduced_map[col] else {
      continue;
    };
    reduced.push(reduced_row, reduced_col, *value);
  }
  Ok(CsrMatrix::from(&reduced))
}

fn reduce_vector_with_layout(layout: &DofLayout, full: &Vector) -> Result<Vector, String> {
  if full.len() != layout.full_dimension {
    return Err(format!(
      "full vector length {} does not match layout dimension {}",
      full.len(),
      layout.full_dimension
    ));
  }
  Ok(Vector::from_iterator(
    layout.reduced_dimension(),
    layout.active_dofs.iter().map(|&index| full[index]),
  ))
}

fn hard_dirichlet_bias(matrix: &CsrMatrix, layout: &DofLayout) -> Vector {
  let reduced_map = reduced_index_map(layout);
  let mut bias = Vector::zeros(layout.reduced_dimension());
  let fixed_map = layout
    .prescribed_dofs
    .iter()
    .map(|entry| (entry.index, entry.value))
    .collect::<BTreeMap<_, _>>();
  for (row, col, value) in matrix.triplet_iter() {
    let Some(reduced_row) = reduced_map[row] else {
      continue;
    };
    if let Some(fixed_value) = fixed_map.get(&col) {
      bias[reduced_row] += *value * *fixed_value;
    }
  }
  bias
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::assemble::boundary_simplices_where_barycenter;
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

  #[test]
  fn reduced_eddy_current_blocks_have_free_edge_dimensions() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let outer_edges =
      boundary_simplices_where_barycenter(&topology, &coords, 1, |point| point[0] == 0.0);
    let boundary = essential_boundary(outer_edges.clone(), vec![0.0; outer_edges.len()]);
    let unit = InnerProductWeightClosure::new(|_| 1.0);

    let system = build_reduced_eddy_current_1form_system(
      &topology, &metric, &coords, None, &unit, &unit, &boundary,
    )
    .expect("eddy-current blocks should assemble");

    let free_dim = topology.nsimplices(1) - outer_edges.len();
    assert_eq!(system.reduced_dimension(), free_dim);
    assert_eq!(system.curl_curl.nrows(), free_dim);
    assert_eq!(system.curl_curl.ncols(), free_dim);
    assert_eq!(system.conductivity_mass.nrows(), free_dim);
    assert_eq!(system.conductivity_mass.ncols(), free_dim);
    assert_eq!(system.state_mass_inverse.nrows(), free_dim);
    assert_eq!(system.state_mass_inverse.ncols(), free_dim);
  }

  #[test]
  fn reduced_eddy_current_bias_folds_nonzero_fixed_dofs() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let fixed = vec![0usize];
    let boundary = essential_boundary(fixed, vec![2.0]);
    let unit = InnerProductWeightClosure::new(|_| 1.0);

    let system = build_reduced_eddy_current_1form_system(
      &topology, &metric, &coords, None, &unit, &unit, &boundary,
    )
    .expect("eddy-current blocks should assemble");

    assert_eq!(
      system.curl_curl_fixed_bias.len(),
      system.reduced_dimension()
    );
    assert_eq!(
      system.conductivity_fixed_bias.len(),
      system.reduced_dimension()
    );
    assert!(
      system
        .conductivity_fixed_bias
        .iter()
        .any(|value| value.abs() > 1e-12),
      "nonzero fixed data should fold into the conductivity bias"
    );
  }

  #[test]
  fn screened_eddy_current_sinusoidal_source_matches_reference_formula() {
    let value = screened_eddy_current_sinusoidal_source_value([0.5, 0.5, 0.5], 2.0)
      .expect("source value should evaluate");
    assert!((value[0] - 2.0).abs() < 1e-14);
    assert!((value[1] - 2.0).abs() < 1e-14);
    assert!((value[2] - 2.0).abs() < 1e-14);

    let value = screened_eddy_current_sinusoidal_source_value([0.0, 0.5, 0.5], 3.0)
      .expect("source value should evaluate");
    assert!((value[0] - 3.0).abs() < 1e-14);
    assert!(value[1].abs() < 1e-14);
    assert!(value[2].abs() < 1e-14);
  }

  #[test]
  fn screened_eddy_current_source_reduction_respects_hard_boundary_edges() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let boundary_edges = topology
      .boundary_subcomplex_simplices(1)
      .into_iter()
      .map(|simplex| simplex.kidx)
      .collect::<Vec<_>>();
    let boundary = essential_boundary(boundary_edges.clone(), vec![0.0; boundary_edges.len()]);
    let source = assemble_reduced_screened_eddy_current_sinusoidal_source(
      &topology, &coords, None, &boundary, 1.0,
    )
    .expect("analytic source should assemble");
    assert_eq!(source.len(), topology.nsimplices(1) - boundary_edges.len());
    assert!(source.iter().all(|value| value.is_finite()));
    assert!(
      source.iter().any(|value| value.abs() > 1e-14),
      "analytic source should produce a nonzero reduced load"
    );
  }

  fn nonlinear_eddy_problem(
    level: usize,
    beta: f64,
  ) -> (
    Complex,
    MeshCoords,
    ReducedNonlinearScreenedEddyCurrent1Form,
  ) {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, level, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let model = build_reduced_nonlinear_screened_eddy_current_1form(
      &topology,
      &coords,
      None,
      NonlinearScreenedEddyCurrentAssemblyConfig::new(
        NonlinearEddyCurrentReluctivityLaw::new(1.3, beta).unwrap(),
        0.7,
        EssentialBoundarySpec::default(),
      ),
    )
    .expect("nonlinear screened eddy-current model should assemble");
    (topology, coords, model)
  }

  fn smooth_edge_truth(
    model: &ReducedNonlinearScreenedEddyCurrent1Form,
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
      let vector_potential = [
        scale * (PI * midpoint[1]).sin() * (PI * midpoint[2]).sin(),
        scale * (PI * midpoint[2]).sin() * (PI * midpoint[0]).sin(),
        scale * (PI * midpoint[0]).sin() * (PI * midpoint[1]).sin(),
      ];
      full[edge.kidx()] = vector_potential[0] * tangent[0]
        + vector_potential[1] * tangent[1]
        + vector_potential[2] * tangent[2];
    }
    model
      .layout()
      .active_dofs
      .iter()
      .map(|&edge| full[edge])
      .collect()
  }

  fn sparse_apply(matrix: &CsrMatrix, vector: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
      out[row] += *value * vector[col];
    }
    out
  }

  fn sparse_entry_map(matrix: &CsrMatrix) -> BTreeMap<(usize, usize), f64> {
    let mut entries = BTreeMap::new();
    for (row, col, value) in matrix.triplet_iter() {
      *entries.entry((row, col)).or_insert(0.0) += *value;
    }
    entries
  }

  fn l2_norm(values: impl AsRef<[f64]>) -> f64 {
    values
      .as_ref()
      .iter()
      .map(|value| value * value)
      .sum::<f64>()
      .sqrt()
  }

  #[test]
  fn nonlinear_eddy_current_beta_zero_jacobian_matches_linear_screened_operator() {
    let (topology, coords, model) = nonlinear_eddy_problem(1, 0.0);
    let state = smooth_edge_truth(&model, &topology, &coords, 0.1);
    let evaluation = model
      .source_free_residual_and_jacobian(&state)
      .expect("source-free residual should evaluate");

    let mut linear_residual = sparse_apply(model.beta_zero_operator(), &state);
    for (value, bias) in linear_residual.iter_mut().zip(model.beta_zero_bias()) {
      *value += *bias;
    }
    let residual_error = evaluation
      .residual
      .iter()
      .zip(linear_residual.iter())
      .map(|(actual, expected)| (actual - expected).abs())
      .fold(0.0, f64::max);
    assert!(
      residual_error < 1e-10,
      "beta-zero residual should equal the linear screened operator, max error {residual_error:.3e}"
    );

    let actual = sparse_entry_map(&evaluation.jacobian);
    let expected = sparse_entry_map(model.beta_zero_operator());
    for key in actual.keys().chain(expected.keys()) {
      let a = actual.get(key).copied().unwrap_or(0.0);
      let e = expected.get(key).copied().unwrap_or(0.0);
      assert!(
        (a - e).abs() < 1e-10,
        "beta-zero Jacobian entry {key:?} mismatch: actual={a:.6e} expected={e:.6e}"
      );
    }
  }

  #[test]
  fn nonlinear_eddy_current_manufactured_truth_has_near_zero_residual() {
    let (topology, coords, source_free) = nonlinear_eddy_problem(1, 0.4);
    let truth = smooth_edge_truth(&source_free, &topology, &coords, 0.15);
    let source = source_free
      .manufactured_source(&truth)
      .expect("manufactured source should assemble");
    let model = source_free
      .with_source(source)
      .expect("manufactured source should fit model layout");
    let residual = model
      .residual_and_jacobian(&truth)
      .expect("residual should evaluate at truth")
      .residual;
    assert!(
      l2_norm(&residual) < 1e-11,
      "manufactured truth residual should be near zero"
    );
  }

  #[test]
  fn nonlinear_eddy_current_sparse_jacobian_matches_finite_differences() {
    let (topology, coords, source_free) = nonlinear_eddy_problem(1, 0.6);
    let state = smooth_edge_truth(&source_free, &topology, &coords, 0.08);
    let source = source_free
      .manufactured_source(&state)
      .expect("manufactured source should assemble");
    let model = source_free.with_source(source).unwrap();
    let evaluation = model.residual_and_jacobian(&state).unwrap();
    let analytic = sparse_entry_map(&evaluation.jacobian);
    let eps = 1e-6;

    for col in 0..model.reduced_dimension() {
      let mut plus = state.clone();
      let mut minus = state.clone();
      plus[col] += eps;
      minus[col] -= eps;
      let r_plus = model.residual_and_jacobian(&plus).unwrap().residual;
      let r_minus = model.residual_and_jacobian(&minus).unwrap().residual;
      for row in 0..model.reduced_dimension() {
        let fd = (r_plus[row] - r_minus[row]) / (2.0 * eps);
        let exact = analytic.get(&(row, col)).copied().unwrap_or(0.0);
        assert!(
          (fd - exact).abs() <= 2e-5 * exact.abs().max(1.0),
          "Jacobian mismatch at ({row}, {col}): finite-diff={fd:.6e}, analytic={exact:.6e}"
        );
      }
    }
  }

  #[test]
  fn nonlinear_eddy_current_local_residual_probe_jacobian_matches_finite_differences() {
    let (topology, coords, source_free) = nonlinear_eddy_problem(1, 0.35);
    let state = smooth_edge_truth(&source_free, &topology, &coords, 0.06);
    let source = source_free.manufactured_source(&state).unwrap();
    let model = source_free.with_source(source).unwrap();
    let selected_cells = (0..model.num_elements().min(3)).collect::<Vec<_>>();
    let probe = LocalEddyCurrentResidualProbe3d::from_model(&model, selected_cells)
      .expect("local probe should assemble");
    assert_eq!(probe.residual_dimension(), 3 * probe.selected_cells().len());
    assert_eq!(probe.state_dimension(), model.reduced_dimension());

    let evaluation = probe.residual_and_jacobian(&state).unwrap();
    let analytic = sparse_entry_map(&evaluation.jacobian);
    let eps = 1e-6;
    for col in 0..model.reduced_dimension() {
      let mut plus = state.clone();
      let mut minus = state.clone();
      plus[col] += eps;
      minus[col] -= eps;
      let r_plus = probe.residual_and_jacobian(&plus).unwrap().residual;
      let r_minus = probe.residual_and_jacobian(&minus).unwrap().residual;
      for row in 0..probe.residual_dimension() {
        let fd = (r_plus[row] - r_minus[row]) / (2.0 * eps);
        let exact = analytic.get(&(row, col)).copied().unwrap_or(0.0);
        assert!(
          (fd - exact).abs() <= 5e-5 * exact.abs().max(1.0),
          "local probe Jacobian mismatch at ({row}, {col}): finite-diff={fd:.6e}, analytic={exact:.6e}"
        );
      }
    }
  }
}
