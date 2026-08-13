use super::lsf::WhitneyLsf;
use crate::cochain::Cochain;

use exterior::{field::ExteriorField, MultiForm};
use manifold::{
  geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt, CoordRef},
  topology::{complex::Complex, handle::SimplexHandle},
};

pub struct WhitneyForm<'a> {
  cochain: Cochain,
  complex: &'a Complex,
  mesh_coords: &'a MeshCoords,
}
impl<'a> WhitneyForm<'a> {
  pub fn new(cochain: Cochain, complex: &'a Complex, mesh_coords: &'a MeshCoords) -> Self {
    Self {
      cochain,
      complex,
      mesh_coords,
    }
  }

  pub fn dif(&self) -> Self {
    Self {
      cochain: self.cochain.dif(self.complex),
      complex: self.complex,
      mesh_coords: self.mesh_coords,
    }
  }
}
impl ExteriorField for WhitneyForm<'_> {
  fn dim_ambient(&self) -> exterior::Dim {
    self.mesh_coords.dim()
  }
  fn dim_intrinsic(&self) -> exterior::Dim {
    self.complex.dim()
  }
  fn grade(&self) -> exterior::ExteriorGrade {
    self.cochain.dim()
  }
  /// Global position
  fn at_point<'a>(&self, coord: impl Into<CoordRef<'a>>) -> exterior::ExteriorElement {
    let coord = coord.into();

    // WARN: This is slow!
    let cell = self
      .mesh_coords
      .find_cell_containing(self.complex, coord)
      .unwrap();
    self.eval_known_cell(cell, coord)
  }
}
impl WhitneyForm<'_> {
  #[allow(dead_code)]
  pub(crate) fn eval_known_cell_intrinsic<'a>(
    &self,
    cell: SimplexHandle,
    coord: impl Into<CoordRef<'a>>,
  ) -> exterior::ExteriorElement {
    let coord = coord.into();
    let cell_coords = cell.coord_simplex(self.mesh_coords);
    let local = cell_coords.global2local(coord);
    self.eval_known_cell_local_intrinsic(cell, local.as_view())
  }

  #[allow(dead_code)]
  pub(crate) fn eval_known_cell_local_intrinsic<'a>(
    &self,
    cell: SimplexHandle,
    local: impl Into<CoordRef<'a>>,
  ) -> exterior::ExteriorElement {
    let local = local.into().into_owned();
    let dim = cell.dim();
    let mut value = MultiForm::zero(dim, self.grade());
    for dof_simp in cell.mesh_subsimps(self.grade()) {
      let local_dof_simp = dof_simp.relative_to(&cell);
      let lsf = WhitneyLsf::standard(dim, local_dof_simp);
      let lsf_value = lsf.at_point(local.as_view());
      let dof_value = self.cochain[dof_simp];
      value += dof_value * lsf_value;
    }
    value
  }

  pub fn eval_known_cell<'a>(
    &self,
    cell: SimplexHandle,
    coord: impl Into<CoordRef<'a>>,
  ) -> exterior::ExteriorElement {
    let coord = coord.into();
    let cell_coords = cell.coord_simplex(self.mesh_coords);
    let local = cell_coords.global2local(coord);
    let local_value = self.eval_known_cell_local_intrinsic(cell, local.as_view());
    cell_coords.lift_form(&local_value)
  }
}

pub struct DifWhitneyForm<'a> {
  cochain: &'a Cochain,
  complex: &'a Complex,
  mesh_coords: &'a MeshCoords,
}
impl<'a> DifWhitneyForm<'a> {
  pub fn new(cochain: &'a Cochain, complex: &'a Complex, mesh_coords: &'a MeshCoords) -> Self {
    Self {
      cochain,
      complex,
      mesh_coords,
    }
  }
}
impl ExteriorField for DifWhitneyForm<'_> {
  fn dim_ambient(&self) -> exterior::Dim {
    self.mesh_coords.dim()
  }
  fn dim_intrinsic(&self) -> exterior::Dim {
    self.complex.dim()
  }
  fn grade(&self) -> exterior::ExteriorGrade {
    self.cochain.dim() + 1
  }
  /// Global position
  fn at_point<'a>(&self, coord: impl Into<CoordRef<'a>>) -> exterior::ExteriorElement {
    let coord = coord.into();

    // WARN: This is slow!
    let cell = self
      .mesh_coords
      .find_cell_containing(self.complex, coord)
      .unwrap();
    let cell_coords = cell.coord_simplex(self.mesh_coords);
    let dim = cell.dim();
    let mut value = MultiForm::zero(dim, self.grade());
    for dof_simp in cell.mesh_subsimps(self.cochain.dim()) {
      let local_dof_simp = dof_simp.relative_to(&cell);
      let lsf = WhitneyLsf::standard(dim, local_dof_simp);
      let dif_lsf_value = lsf.dif();
      let dof_value = self.cochain[dof_simp];
      value += dof_value * dif_lsf_value;
    }
    cell_coords.lift_form(&value)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use common::linalg::nalgebra::Vector;
  use manifold::geometry::coord::mesh::standard_coord_complex;

  #[test]
  fn embedded_whitney_pullback_matches_intrinsic_evaluation() {
    let (topology, coords_2d) = standard_coord_complex(2);
    let coords = coords_2d.embed_euclidean(3);
    let cell = topology.cells().handle_iter().next().unwrap();
    let cell_coords = cell.coord_simplex(&coords);
    let bary = cell_coords.barycenter();
    let cochain = Cochain::new(1, Vector::from_vec(vec![1.0, -2.0, 0.5]));
    let whitney = WhitneyForm::new(cochain, &topology, &coords);

    let ambient_value = whitney.eval_known_cell(cell, &bary);
    let intrinsic_value = whitney.eval_known_cell_intrinsic(cell, &bary);

    assert!(cell_coords
      .pullback_form(&ambient_value)
      .eq_epsilon(&intrinsic_value, 1e-12));
  }
}
