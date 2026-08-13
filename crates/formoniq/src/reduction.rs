//! Native prescribed-DOF elimination metadata.
//!
//! These types describe deterministic FEEC reduction only. Statistical
//! interpretations of boundary data live in the integration workspace.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrescribedDof {
  pub index: usize,
  pub value: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EssentialBoundarySpec {
  pub state: Vec<PrescribedDof>,
  pub auxiliary: Vec<PrescribedDof>,
}

impl EssentialBoundarySpec {
  pub fn with_state(mut self, prescribed: impl IntoIterator<Item = PrescribedDof>) -> Self {
    self.state.extend(prescribed);
    self
  }

  pub fn with_auxiliary(mut self, prescribed: impl IntoIterator<Item = PrescribedDof>) -> Self {
    self.auxiliary.extend(prescribed);
    self
  }

  pub fn validate(&self, state_dimension: usize, auxiliary_dimension: usize) -> Result<(), String> {
    validate_prescribed_dofs("state", state_dimension, &self.state)?;
    validate_prescribed_dofs("auxiliary", auxiliary_dimension, &self.auxiliary)
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DofLayout {
  pub full_dimension: usize,
  pub active_dofs: Vec<usize>,
  pub prescribed_dofs: Vec<PrescribedDof>,
}

impl DofLayout {
  pub fn new(
    full_dimension: usize,
    active_dofs: Vec<usize>,
    prescribed_dofs: Vec<PrescribedDof>,
  ) -> Self {
    Self::from_parts(full_dimension, active_dofs, prescribed_dofs)
      .expect("invalid native FEEC dof layout")
  }

  pub fn identity(dimension: usize) -> Self {
    Self {
      full_dimension: dimension,
      active_dofs: (0..dimension).collect(),
      prescribed_dofs: Vec::new(),
    }
  }

  pub fn from_prescribed(
    full_dimension: usize,
    prescribed_dofs: Vec<PrescribedDof>,
  ) -> Result<Self, String> {
    validate_prescribed_dofs("layout", full_dimension, &prescribed_dofs)?;
    let prescribed = prescribed_dofs
      .iter()
      .map(|entry| entry.index)
      .collect::<BTreeSet<_>>();
    let active_dofs = (0..full_dimension)
      .filter(|index| !prescribed.contains(index))
      .collect();
    Ok(Self {
      full_dimension,
      active_dofs,
      prescribed_dofs,
    })
  }

  pub fn from_parts(
    full_dimension: usize,
    active_dofs: Vec<usize>,
    prescribed_dofs: Vec<PrescribedDof>,
  ) -> Result<Self, String> {
    validate_prescribed_dofs("layout", full_dimension, &prescribed_dofs)?;
    let mut active = BTreeSet::new();
    for &dof in &active_dofs {
      if dof >= full_dimension {
        return Err(format!(
          "active dof {dof} is outside full dimension {full_dimension}"
        ));
      }
      if !active.insert(dof) {
        return Err(format!("active dof {dof} appears more than once"));
      }
    }
    for prescribed in &prescribed_dofs {
      if active.contains(&prescribed.index) {
        return Err(format!(
          "prescribed dof {} must not also be active",
          prescribed.index
        ));
      }
    }
    if active_dofs.len() + prescribed_dofs.len() != full_dimension {
      return Err(format!(
        "active and prescribed dofs cover {} entries, expected {full_dimension}",
        active_dofs.len() + prescribed_dofs.len()
      ));
    }
    Ok(Self {
      full_dimension,
      active_dofs,
      prescribed_dofs,
    })
  }

  pub fn reduced_dimension(&self) -> usize {
    self.active_dofs.len()
  }
}

pub fn validate_prescribed_dofs(
  label: &str,
  full_dimension: usize,
  prescribed: &[PrescribedDof],
) -> Result<(), String> {
  let mut seen = BTreeSet::new();
  for entry in prescribed {
    if entry.index >= full_dimension {
      return Err(format!(
        "{label} prescribed dof {} is outside dimension {full_dimension}",
        entry.index
      ));
    }
    if !entry.value.is_finite() {
      return Err(format!(
        "{label} prescribed dof {} has non-finite value {}",
        entry.index, entry.value
      ));
    }
    if !seen.insert(entry.index) {
      return Err(format!(
        "{label} prescribed dof {} appears more than once",
        entry.index
      ));
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn layout_from_prescribed_builds_complement() {
    let layout = DofLayout::from_prescribed(
      4,
      vec![PrescribedDof {
        index: 1,
        value: 2.0,
      }],
    )
    .unwrap();
    assert_eq!(layout.active_dofs, vec![0, 2, 3]);
    assert_eq!(layout.reduced_dimension(), 3);
  }

  #[test]
  fn prescribed_dofs_reject_duplicates_bounds_and_nonfinite_values() {
    assert!(DofLayout::from_prescribed(
      2,
      vec![
        PrescribedDof {
          index: 0,
          value: 1.0,
        },
        PrescribedDof {
          index: 0,
          value: 2.0,
        },
      ],
    )
    .is_err());
    assert!(DofLayout::from_prescribed(
      2,
      vec![PrescribedDof {
        index: 2,
        value: 1.0,
      }],
    )
    .is_err());
    assert!(DofLayout::from_prescribed(
      2,
      vec![PrescribedDof {
        index: 1,
        value: f64::NAN,
      }],
    )
    .is_err());
  }
}
