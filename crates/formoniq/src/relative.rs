//! Homogeneous full-relative Whitney subcomplex assembly.
//!
//! A relative boundary condition is a property of the entire discrete de Rham
//! complex, not a principal restriction applied only to the state space.  This
//! module therefore chooses the active degrees of freedom in all adjacent
//! spaces before it restricts mass matrices or exterior derivatives.

use crate::{
  assemble::assemble_galmat,
  operators::HodgeMassElmat,
  reduction::{DofLayout, PrescribedDof},
};
use common::linalg::nalgebra::{CooMatrix, CsrMatrix};
use ddf::ManifoldComplexExt;
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex, Dim};
use std::{error::Error, fmt};

const COMPLEX_TOLERANCE: f64 = 1e-12;

/// A failure to assemble a compatible homogeneous full-relative subcomplex.
#[derive(Debug, Clone, PartialEq)]
pub enum FullRelativeAssemblyError {
  InvalidDegree {
    degree: Dim,
    complex_dimension: Dim,
  },
  InvalidLayout {
    degree: Dim,
    message: String,
  },
  DimensionMismatch {
    operator: &'static str,
    expected_rows: usize,
    expected_cols: usize,
    actual_rows: usize,
    actual_cols: usize,
  },
  SubcomplexViolation {
    degree: Dim,
    maximum_residual: f64,
  },
  InjectionViolation {
    degree: Dim,
    maximum_residual: f64,
  },
}

impl fmt::Display for FullRelativeAssemblyError {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidDegree {
        degree,
        complex_dimension,
      } => write!(
        formatter,
        "form degree {degree} exceeds complex dimension {complex_dimension}"
      ),
      Self::InvalidLayout { degree, message } => {
        write!(formatter, "invalid degree-{degree} relative layout: {message}")
      }
      Self::DimensionMismatch {
        operator,
        expected_rows,
        expected_cols,
        actual_rows,
        actual_cols,
      } => write!(
        formatter,
        "{operator} must have shape {expected_rows}x{expected_cols}, got {actual_rows}x{actual_cols}"
      ),
      Self::SubcomplexViolation {
        degree,
        maximum_residual,
      } => write!(
        formatter,
        "relative exterior derivatives at degree {degree} do not compose to zero; maximum residual {maximum_residual}"
      ),
      Self::InjectionViolation {
        degree,
        maximum_residual,
      } => write!(
        formatter,
        "degree-{degree} exterior derivative does not preserve the relative subcomplex; maximum residual {maximum_residual}"
      ),
    }
  }
}

impl Error for FullRelativeAssemblyError {}

/// One active lowest-order Whitney space with its reduced weak L2 metric.
#[derive(Debug, Clone, PartialEq)]
pub struct FullRelativeWhitneySpace {
  degree: Dim,
  mass: CsrMatrix,
  layout: DofLayout,
}

impl FullRelativeWhitneySpace {
  pub fn degree(&self) -> Dim {
    self.degree
  }

  pub fn mass(&self) -> &CsrMatrix {
    &self.mass
  }

  pub fn layout(&self) -> &DofLayout {
    &self.layout
  }

  pub fn dimension(&self) -> usize {
    self.layout.reduced_dimension()
  }
}

/// A coefficient exterior derivative between two active relative spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct FullRelativeWhitneyDerivative {
  source_degree: Dim,
  full: CsrMatrix,
  active: CsrMatrix,
}

impl FullRelativeWhitneyDerivative {
  pub fn source_degree(&self) -> Dim {
    self.source_degree
  }

  pub fn full(&self) -> &CsrMatrix {
    &self.full
  }

  pub fn active(&self) -> &CsrMatrix {
    &self.active
  }
}

/// Adjacent spaces and derivatives for the exact degree-`k` Whitney target.
///
/// The lower and incoming fields are absent exactly at degree zero.  The upper
/// and outgoing fields are absent exactly at top degree; no zero-sized inverse
/// placeholder is used for either endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct FullRelativeWhitneyOperators {
  complex_dimension: Dim,
  lower: Option<FullRelativeWhitneySpace>,
  state: FullRelativeWhitneySpace,
  upper: Option<FullRelativeWhitneySpace>,
  incoming: Option<FullRelativeWhitneyDerivative>,
  outgoing: Option<FullRelativeWhitneyDerivative>,
}

impl FullRelativeWhitneyOperators {
  pub fn complex_dimension(&self) -> Dim {
    self.complex_dimension
  }

  pub fn degree(&self) -> Dim {
    self.state.degree()
  }

  pub fn lower(&self) -> Option<&FullRelativeWhitneySpace> {
    self.lower.as_ref()
  }

  pub fn state(&self) -> &FullRelativeWhitneySpace {
    &self.state
  }

  pub fn upper(&self) -> Option<&FullRelativeWhitneySpace> {
    self.upper.as_ref()
  }

  pub fn incoming(&self) -> Option<&FullRelativeWhitneyDerivative> {
    self.incoming.as_ref()
  }

  pub fn outgoing(&self) -> Option<&FullRelativeWhitneyDerivative> {
    self.outgoing.as_ref()
  }
}

/// Assemble the adjacent lowest-order Whitney spaces on the homogeneous
/// full-relative de Rham subcomplex.
///
/// For each active degree, Whitney degrees of freedom attached to simplices in
/// the boundary subcomplex are prescribed to zero.  Top-degree forms have no
/// boundary degrees of freedom.  Masses are restricted only after these three
/// compatible layouts have been chosen, and the returned derivatives satisfy
/// both `D_full P = P D_active` and `D_k D_{k-1} = 0`.
pub fn assemble_whitney_full_relative_operators(
  topology: &Complex,
  geometry: &MeshLengths,
  degree: Dim,
) -> Result<FullRelativeWhitneyOperators, FullRelativeAssemblyError> {
  let complex_dimension = topology.dim();
  if degree > complex_dimension {
    return Err(FullRelativeAssemblyError::InvalidDegree {
      degree,
      complex_dimension,
    });
  }

  let state = assemble_space(topology, geometry, degree)?;
  let lower = (degree > 0)
    .then(|| assemble_space(topology, geometry, degree - 1))
    .transpose()?;
  let upper = (degree < complex_dimension)
    .then(|| assemble_space(topology, geometry, degree + 1))
    .transpose()?;

  let incoming = match &lower {
    Some(lower) => Some(assemble_derivative(topology, lower, &state)?),
    None => None,
  };
  let outgoing = match &upper {
    Some(upper) => Some(assemble_derivative(topology, &state, upper)?),
    None => None,
  };

  if let (Some(incoming), Some(outgoing)) = (&incoming, &outgoing) {
    validate_complex_identity(degree, incoming.active(), outgoing.active())?;
  }

  Ok(FullRelativeWhitneyOperators {
    complex_dimension,
    lower,
    state,
    upper,
    incoming,
    outgoing,
  })
}

fn assemble_space(
  topology: &Complex,
  geometry: &MeshLengths,
  degree: Dim,
) -> Result<FullRelativeWhitneySpace, FullRelativeAssemblyError> {
  let full_dimension = topology.nsimplices(degree);
  let prescribed_dofs = topology
    .boundary_subcomplex_simplices(degree)
    .into_iter()
    .map(|simplex| PrescribedDof {
      index: simplex.kidx,
      value: 0.0,
    })
    .collect::<Vec<_>>();
  let layout = DofLayout::from_prescribed(full_dimension, prescribed_dofs)
    .map_err(|message| FullRelativeAssemblyError::InvalidLayout { degree, message })?;
  let full_mass = CsrMatrix::from(&assemble_galmat(
    topology,
    geometry,
    HodgeMassElmat::new(topology.dim(), degree),
  ));
  let mass = restrict_square(&full_mass, &layout, "relative mass")?;
  Ok(FullRelativeWhitneySpace {
    degree,
    mass,
    layout,
  })
}

fn assemble_derivative(
  topology: &Complex,
  source: &FullRelativeWhitneySpace,
  target: &FullRelativeWhitneySpace,
) -> Result<FullRelativeWhitneyDerivative, FullRelativeAssemblyError> {
  debug_assert_eq!(target.degree(), source.degree() + 1);
  let source_degree = source.degree();
  let full = CsrMatrix::from(&topology.exterior_derivative_operator(source_degree));
  let expected_rows = target.layout().full_dimension;
  let expected_cols = source.layout().full_dimension;
  if full.nrows() != expected_rows || full.ncols() != expected_cols {
    return Err(FullRelativeAssemblyError::DimensionMismatch {
      operator: "full exterior derivative",
      expected_rows,
      expected_cols,
      actual_rows: full.nrows(),
      actual_cols: full.ncols(),
    });
  }
  let active = restrict_rectangular(&full, source.layout(), target.layout());
  validate_injection_identity(source_degree, &full, source.layout(), target.layout())?;
  Ok(FullRelativeWhitneyDerivative {
    source_degree,
    full,
    active,
  })
}

fn reduced_index_map(layout: &DofLayout) -> Vec<Option<usize>> {
  let mut map = vec![None; layout.full_dimension];
  for (reduced, full) in layout.active_dofs.iter().copied().enumerate() {
    map[full] = Some(reduced);
  }
  map
}

fn restrict_square(
  matrix: &CsrMatrix,
  layout: &DofLayout,
  operator: &'static str,
) -> Result<CsrMatrix, FullRelativeAssemblyError> {
  if matrix.nrows() != layout.full_dimension || matrix.ncols() != layout.full_dimension {
    return Err(FullRelativeAssemblyError::DimensionMismatch {
      operator,
      expected_rows: layout.full_dimension,
      expected_cols: layout.full_dimension,
      actual_rows: matrix.nrows(),
      actual_cols: matrix.ncols(),
    });
  }
  Ok(restrict_rectangular(matrix, layout, layout))
}

fn restrict_rectangular(matrix: &CsrMatrix, source: &DofLayout, target: &DofLayout) -> CsrMatrix {
  let source_map = reduced_index_map(source);
  let target_map = reduced_index_map(target);
  let mut reduced = CooMatrix::new(target.reduced_dimension(), source.reduced_dimension());
  for (row, col, value) in matrix.triplet_iter() {
    if let (Some(reduced_row), Some(reduced_col)) = (target_map[row], source_map[col]) {
      reduced.push(reduced_row, reduced_col, *value);
    }
  }
  CsrMatrix::from(&reduced)
}

fn validate_injection_identity(
  degree: Dim,
  full: &CsrMatrix,
  source: &DofLayout,
  target: &DofLayout,
) -> Result<(), FullRelativeAssemblyError> {
  let source_map = reduced_index_map(source);
  let target_map = reduced_index_map(target);
  let maximum_residual = full
    .triplet_iter()
    .filter(|(row, col, _)| source_map[*col].is_some() && target_map[*row].is_none())
    .map(|(_, _, value)| value.abs())
    .fold(0.0_f64, f64::max);
  if maximum_residual > COMPLEX_TOLERANCE {
    return Err(FullRelativeAssemblyError::InjectionViolation {
      degree,
      maximum_residual,
    });
  }
  Ok(())
}

fn validate_complex_identity(
  degree: Dim,
  incoming: &CsrMatrix,
  outgoing: &CsrMatrix,
) -> Result<(), FullRelativeAssemblyError> {
  let composition = outgoing * incoming;
  let maximum_residual = composition
    .triplet_iter()
    .map(|(_, _, value)| value.abs())
    .fold(0.0_f64, f64::max);
  if maximum_residual > COMPLEX_TOLERANCE {
    return Err(FullRelativeAssemblyError::SubcomplexViolation {
      degree,
      maximum_residual,
    });
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use common::linalg::nalgebra::Matrix;
  use manifold::gen::cartesian::CartesianMeshInfo;

  fn fixture(dim: usize, cells_per_axis: usize) -> (Complex, MeshLengths) {
    let mesh = CartesianMeshInfo::new_unit(dim, cells_per_axis);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    (topology, geometry)
  }

  #[test]
  fn positive_degree_uses_three_compatible_relative_spaces() {
    let (topology, geometry) = fixture(2, 2);
    let operators = assemble_whitney_full_relative_operators(&topology, &geometry, 1).unwrap();

    let lower = operators.lower().unwrap();
    let upper = operators.upper().unwrap();
    assert_eq!(lower.layout().prescribed_dofs.len(), 8);
    assert_eq!(lower.dimension(), 1);
    assert!(!operators.state().layout().prescribed_dofs.is_empty());
    assert_eq!(upper.dimension(), topology.nsimplices(2));
    assert!(upper.layout().prescribed_dofs.is_empty());

    let incoming = operators.incoming().unwrap().active();
    let outgoing = operators.outgoing().unwrap().active();
    let composition = outgoing * incoming;
    assert!(composition
      .triplet_iter()
      .all(|(_, _, value)| value.abs() <= COMPLEX_TOLERANCE));
  }

  #[test]
  fn endpoint_branches_are_absent_instead_of_zero_sized_placeholders() {
    let (topology, geometry) = fixture(2, 2);
    let zero = assemble_whitney_full_relative_operators(&topology, &geometry, 0).unwrap();
    assert!(zero.lower().is_none());
    assert!(zero.incoming().is_none());
    assert!(zero.upper().is_some());
    assert!(zero.outgoing().is_some());

    let top = assemble_whitney_full_relative_operators(&topology, &geometry, 2).unwrap();
    assert!(top.lower().is_some());
    assert!(top.incoming().is_some());
    assert!(top.upper().is_none());
    assert!(top.outgoing().is_none());
    assert_eq!(top.state().dimension(), topology.nsimplices(2));
    assert!(top.state().layout().prescribed_dofs.is_empty());
  }

  #[test]
  fn tetrahedral_relative_derivatives_preserve_injection_and_compose_to_zero() {
    let (topology, geometry) = fixture(3, 2);
    for degree in [1, 2] {
      let operators =
        assemble_whitney_full_relative_operators(&topology, &geometry, degree).unwrap();
      let spaces = [
        operators.lower().unwrap(),
        operators.state(),
        operators.upper().unwrap(),
      ];
      for space in spaces {
        assert!(Matrix::from(space.mass()).cholesky().is_some());
      }
      for (derivative, source, target) in [
        (operators.incoming().unwrap(), spaces[0], spaces[1]),
        (operators.outgoing().unwrap(), spaces[1], spaces[2]),
      ] {
        let injection = |layout: &DofLayout| {
          Matrix::from_fn(
            layout.full_dimension,
            layout.reduced_dimension(),
            |full, active| {
              if full == layout.active_dofs[active] {
                1.0
              } else {
                0.0
              }
            },
          )
        };
        let full = Matrix::from(derivative.full());
        let active = Matrix::from(derivative.active());
        let residual = full * injection(source.layout()) - injection(target.layout()) * active;
        assert_eq!(residual.norm(), 0.0);
      }
      let composition =
        operators.outgoing().unwrap().active() * operators.incoming().unwrap().active();
      assert!(composition
        .triplet_iter()
        .all(|(_, _, value)| *value == 0.0));
    }
  }

  #[test]
  fn relative_grade_zero_projected_inverse_uses_full_vertex_volumes() {
    use crate::assemble::assemble_reduced_whitney_projected_sparse_inverse_galmat_for_grade;
    let (topology, geometry) = fixture(3, 2);
    let operators = assemble_whitney_full_relative_operators(&topology, &geometry, 1).unwrap();
    let layout = operators.lower().unwrap().layout();
    let inverse = Matrix::from(
      &assemble_reduced_whitney_projected_sparse_inverse_galmat_for_grade(
        &topology, &geometry, 0, layout,
      )
      .unwrap(),
    );
    let full_mass = Matrix::from(&assemble_galmat(
      &topology,
      &geometry,
      HodgeMassElmat::new(3, 0),
    ));
    let expected = Matrix::from_diagonal(&common::linalg::nalgebra::Vector::from_iterator(
      layout.reduced_dimension(),
      layout
        .active_dofs
        .iter()
        .map(|&node| 1.0 / full_mass.row(node).iter().sum::<f64>()),
    ));
    assert!((&inverse - &expected).norm() < 1e-12);
    let reduced_mass = Matrix::from(operators.lower().unwrap().mass());
    assert!((inverse[(0, 0)] - 1.0 / reduced_mass.row(0).iter().sum::<f64>()).abs() > 1e-8);
  }

  #[test]
  fn relative_assembly_rejects_a_degree_above_the_mesh_dimension() {
    let (topology, geometry) = fixture(3, 1);
    assert!(matches!(
      assemble_whitney_full_relative_operators(&topology, &geometry, 4),
      Err(FullRelativeAssemblyError::InvalidDegree {
        degree: 4,
        complex_dimension: 3
      })
    ));
  }

  #[test]
  fn state_only_principal_restriction_differs_from_relative_complex_assembly() {
    let (topology, geometry) = fixture(2, 2);
    let relative = assemble_whitney_full_relative_operators(&topology, &geometry, 1).unwrap();

    let full_lower_mass = Matrix::from(&assemble_galmat(
      &topology,
      &geometry,
      HodgeMassElmat::new(2, 0),
    ));
    let full_state_mass = Matrix::from(&assemble_galmat(
      &topology,
      &geometry,
      HodgeMassElmat::new(2, 1),
    ));
    let full_derivative = Matrix::from(&topology.exterior_derivative_operator(0));
    let full_lower_inverse = full_lower_mass.clone().try_inverse().unwrap();
    let full_lower_term = &full_state_mass
      * &full_derivative
      * full_lower_inverse
      * full_derivative.transpose()
      * &full_state_mass;
    let state_only = principal_block(&full_lower_term, relative.state().layout());

    let relative_state_mass = Matrix::from(relative.state().mass());
    let relative_lower_mass = Matrix::from(relative.lower().unwrap().mass());
    let relative_derivative = Matrix::from(relative.incoming().unwrap().active());
    let relative_lower_inverse = relative_lower_mass.try_inverse().unwrap();
    let assembled_on_boundary = &relative_state_mass
      * &relative_derivative
      * relative_lower_inverse
      * relative_derivative.transpose()
      * &relative_state_mass;

    let discrepancy = state_only - assembled_on_boundary;
    assert!(
      discrepancy.norm() > 1e-8,
      "the fixture must distinguish conditioning a full-space operator from assembling on the relative domain"
    );
  }

  fn principal_block(matrix: &Matrix, layout: &DofLayout) -> Matrix {
    Matrix::from_fn(
      layout.reduced_dimension(),
      layout.reduced_dimension(),
      |row, col| matrix[(layout.active_dofs[row], layout.active_dofs[col])],
    )
  }
}
