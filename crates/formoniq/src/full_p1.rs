//! Exact-metric primitives for the three-dimensional full first-order
//! one-form space.
//!
//! The compatible polynomial sequence containing this space is
//!
//! ```text
//! P2 Lambda0 -> P1 Lambda1 -> P1^- Lambda2 -> P0 Lambda3.
//! ```
//!
//! This module deliberately implements only the independently valid middle
//! space, its exact metric, the Whitney inclusion/retraction, and the local
//! reference-complex algebra.  It does not assemble the global quadratic
//! predecessor, claim a sparse predecessor metric, or construct a shifted
//! Hodge--Laplacian.

use crate::assemble::{
  assemble_nc1_mass_galmat, assemble_nc1_to_whitney_projection_galmat,
  assemble_whitney_to_nc1_embedding_galmat,
};
use common::linalg::nalgebra::{CsrMatrix, Matrix};
use ddf::ManifoldComplexExt;
use manifold::{
  geometry::metric::mesh::MeshLengths,
  topology::{complex::Complex, simplex::standard_subsimps},
  Dim,
};
use std::{error::Error, fmt};

const SUPPORTED_DIMENSION: Dim = 3;

/// Failure to assemble the scoped full-`P1 Lambda1` metric primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullP1AssemblyError {
  /// The first implementation is intentionally restricted to tetrahedral 3D
  /// complexes, where the compatible neighboring spaces have been audited.
  UnsupportedDimension { actual: Dim },
}

impl fmt::Display for FullP1AssemblyError {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::UnsupportedDimension { actual } => write!(
        formatter,
        "full P1 one-form assembly requires a tetrahedral 3D complex, got dimension {actual}"
      ),
    }
  }
}

impl Error for FullP1AssemblyError {}

/// Exact weak metric and canonical Whitney maps for `P1 Lambda1` in 3D.
///
/// There are two full-space coefficients per oriented mesh edge.  The
/// embedding maps a Whitney edge coefficient to both slots.  The retraction
/// averages those two slots, so `retraction * embedding` is the identity on
/// Whitney one-cochains.  This retraction is algebraic; it is not advertised
/// as an orthogonal or commuting cochain projection for the complete mixed
/// polynomial complex.
#[derive(Debug, Clone, PartialEq)]
pub struct FullP1OneFormMetric3d {
  exact_mass: CsrMatrix,
  whitney_embedding: CsrMatrix,
  whitney_retraction: CsrMatrix,
}

impl FullP1OneFormMetric3d {
  /// Exactly integrated full-space mass matrix.
  pub fn exact_mass(&self) -> &CsrMatrix {
    &self.exact_mass
  }

  /// Coefficient inclusion from Whitney one-cochains into the full space.
  pub fn whitney_embedding(&self) -> &CsrMatrix {
    &self.whitney_embedding
  }

  /// Algebraic left inverse of [`Self::whitney_embedding`].
  pub fn whitney_retraction(&self) -> &CsrMatrix {
    &self.whitney_retraction
  }

  /// Number of full-space state coefficients (`2 * number_of_edges`).
  pub fn dimension(&self) -> usize {
    self.exact_mass.nrows()
  }
}

/// Assemble the exact `P1 Lambda1` metric and canonical Whitney maps in 3D.
///
/// The existing NC1 element assembly is reused because its two barycentric
/// summands per edge are a basis of the conforming full-polynomial
/// `P1 Lambda1` finite element.  The quadratic integrand is integrated by the
/// order-three simplex rule used by that canonical assembler.
pub fn assemble_full_p1_one_form_metric_3d(
  topology: &Complex,
  geometry: &MeshLengths,
) -> Result<FullP1OneFormMetric3d, FullP1AssemblyError> {
  if topology.dim() != SUPPORTED_DIMENSION {
    return Err(FullP1AssemblyError::UnsupportedDimension {
      actual: topology.dim(),
    });
  }

  let exact_mass = CsrMatrix::from(&assemble_nc1_mass_galmat(topology, geometry));
  let whitney_embedding = CsrMatrix::from(&assemble_whitney_to_nc1_embedding_galmat(topology));
  let whitney_retraction = CsrMatrix::from(&assemble_nc1_to_whitney_projection_galmat(topology));

  debug_assert_eq!(exact_mass.nrows(), exact_mass.ncols());
  debug_assert_eq!(exact_mass.nrows(), whitney_embedding.nrows());
  debug_assert_eq!(exact_mass.nrows(), whitney_retraction.ncols());
  debug_assert_eq!(whitney_embedding.ncols(), whitney_retraction.nrows());

  Ok(FullP1OneFormMetric3d {
    exact_mass,
    whitney_embedding,
    whitney_retraction,
  })
}

/// Algebraic reference-tetrahedron complex containing `P1 Lambda1`.
///
/// The basis ordering is fixed as follows:
///
/// - `P2 Lambda0`: four Bernstein vertex functions `lambda_i^2`, followed by
///   six edge functions `2 lambda_i lambda_j` in lexicographic edge order;
/// - `P1 Lambda1`: two slots per lexicographically ordered edge `(i,j)`,
///   namely `lambda_i d lambda_j` and `-lambda_j d lambda_i`;
/// - `P1^- Lambda2`: the four oriented Whitney face forms;
/// - `P0 Lambda3`: the oriented constant Whitney volume form.
///
/// All entries are integers or dyadic rationals, hence exactly representable
/// in the stored `f64` matrices.  This object is an algebraic oracle and basis
/// contract for future assembly; it is not a global finite-element operator.
#[derive(Debug, Clone, PartialEq)]
pub struct FullP1ReferenceComplex3d {
  p2_to_full_p1: Matrix,
  full_p1_to_whitney_two: Matrix,
  whitney_two_to_constant_three: Matrix,
  whitney_one_embedding: Matrix,
  full_p1_to_whitney_one_retraction: Matrix,
}

impl Default for FullP1ReferenceComplex3d {
  fn default() -> Self {
    Self::new()
  }
}

impl FullP1ReferenceComplex3d {
  /// Construct the fixed reference-tetrahedron matrices.
  pub fn new() -> Self {
    let topology = Complex::standard(SUPPORTED_DIMENSION);
    let edges = standard_subsimps(SUPPORTED_DIMENSION, 1).collect::<Vec<_>>();

    let mut p2_to_full_p1 = Matrix::zeros(12, 10);

    // d(lambda_i^2) = -2 sum_{j != i} lambda_i d lambda_j.
    for vertex in 0..4 {
      for other in 0..4 {
        if vertex == other {
          continue;
        }
        let edge = [vertex.min(other), vertex.max(other)];
        let edge_index = edges
          .iter()
          .position(|candidate| candidate.vertices.as_slice() == edge)
          .expect("standard tetrahedron contains every vertex pair");
        let (slot, coefficient) = if vertex < other { (0, -2.0) } else { (1, 2.0) };
        p2_to_full_p1[(2 * edge_index + slot, vertex)] = coefficient;
      }
    }

    // d(2 lambda_i lambda_j) = 2(lambda_i d lambda_j + lambda_j d lambda_i).
    for edge_index in 0..edges.len() {
      let column = 4 + edge_index;
      p2_to_full_p1[(2 * edge_index, column)] = 2.0;
      p2_to_full_p1[(2 * edge_index + 1, column)] = -2.0;
    }

    let whitney_one_to_two = Matrix::from(&topology.exterior_derivative_operator(1));
    let full_p1_to_whitney_two = Matrix::from_fn(4, 12, |row, column| {
      0.5 * whitney_one_to_two[(row, column / 2)]
    });
    let whitney_two_to_constant_three = Matrix::from(&topology.exterior_derivative_operator(2));

    let whitney_one_embedding = Matrix::from_fn(
      12,
      6,
      |row, column| {
        if row / 2 == column {
          1.0
        } else {
          0.0
        }
      },
    );
    let full_p1_to_whitney_one_retraction =
      Matrix::from_fn(
        6,
        12,
        |row, column| {
          if column / 2 == row {
            0.5
          } else {
            0.0
          }
        },
      );

    Self {
      p2_to_full_p1,
      full_p1_to_whitney_two,
      whitney_two_to_constant_three,
      whitney_one_embedding,
      full_p1_to_whitney_one_retraction,
    }
  }

  /// `d: P2 Lambda0 -> P1 Lambda1`, with shape `12 x 10` and rank nine.
  pub fn p2_to_full_p1(&self) -> &Matrix {
    &self.p2_to_full_p1
  }

  /// `d: P1 Lambda1 -> P1^- Lambda2`, with shape `4 x 12` and rank three.
  pub fn full_p1_to_whitney_two(&self) -> &Matrix {
    &self.full_p1_to_whitney_two
  }

  /// `d: P1^- Lambda2 -> P0 Lambda3`, with shape `1 x 4` and rank one.
  pub fn whitney_two_to_constant_three(&self) -> &Matrix {
    &self.whitney_two_to_constant_three
  }

  /// Inclusion of the six-dimensional Whitney one-form subspace.
  pub fn whitney_one_embedding(&self) -> &Matrix {
    &self.whitney_one_embedding
  }

  /// Algebraic left inverse of [`Self::whitney_one_embedding`].
  pub fn full_p1_to_whitney_one_retraction(&self) -> &Matrix {
    &self.full_p1_to_whitney_one_retraction
  }
}
