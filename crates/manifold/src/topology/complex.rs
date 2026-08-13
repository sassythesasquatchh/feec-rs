use super::{
  handle::{KSimplexIdx, SimplexIdx, SkeletonHandle},
  simplex::Simplex,
  skeleton::Skeleton,
};
use crate::Dim;

use common::linalg::nalgebra::{CooMatrix, Matrix};

use itertools::Itertools;

/// A simplicial manifold complex.
#[derive(Default, Debug, Clone)]
pub struct Complex {
  skeletons: Vec<ComplexSkeleton>,
}

/// A skeleton inside of a complex.
#[derive(Default, Debug, Clone)]
pub struct ComplexSkeleton {
  skeleton: Skeleton,
  complex_data: SkeletonComplexData,
}
impl ComplexSkeleton {
  pub fn skeleton(&self) -> &Skeleton {
    &self.skeleton
  }
  pub fn complex_data(&self) -> &[SimplexComplexData] {
    &self.complex_data
  }
}

pub type SkeletonComplexData = Vec<SimplexComplexData>;

#[derive(Default, Debug, Clone)]
pub struct SimplexComplexData {
  pub cocells: Vec<SimplexIdx>,
}

impl Complex {
  pub fn skeletons(&self) -> impl Iterator<Item = SkeletonHandle<'_>> {
    (0..=self.dim()).map(|d| SkeletonHandle::new(self, d))
  }
  pub fn skeleton(&self, dim: Dim) -> SkeletonHandle<'_> {
    SkeletonHandle::new(self, dim)
  }
  pub fn mesh_skeleton_raw(&self, dim: Dim) -> &ComplexSkeleton {
    &self.skeletons[dim]
  }
  pub fn nsimplices(&self, dim: Dim) -> usize {
    self.skeleton(dim).len()
  }
  pub fn vertices(&self) -> SkeletonHandle<'_> {
    self.skeleton(0)
  }
  pub fn edges(&self) -> SkeletonHandle<'_> {
    self.skeleton(1)
  }
  pub fn facets(&self) -> SkeletonHandle<'_> {
    self.skeleton(self.dim() - 1)
  }
  pub fn cells(&self) -> SkeletonHandle<'_> {
    self.skeleton(self.dim())
  }
}

impl Complex {
  pub fn standard(dim: Dim) -> Self {
    Self::from_cells(Skeleton::standard(dim))
  }
  pub fn dim(&self) -> Dim {
    self.skeletons.len() - 1
  }

  pub fn has_boundary(&self) -> bool {
    !self.boundary_facets().is_empty()
  }

  pub fn boundary_subcomplex_simplices(&self, k: Dim) -> Vec<SimplexIdx> {
    let top = self.dim();
    if top == 0 {
      return Vec::new();
    }
    if k > top - 1 {
      return Vec::new();
    }

    let bfacets = self.boundary_facets();
    if k == top - 1 {
      return bfacets;
    }

    // Closure of boundary facets down to dimension k.
    bfacets
      .into_iter()
      .flat_map(|f: SimplexIdx| {
        f.handle(self)
          .mesh_subsimps(k)
          .map(|s| s.idx())
          .collect::<Vec<_>>()
      })
      .unique()
      .collect()
  }

  /// For a d-mesh computes the boundary, which consists of facets ((d-1)-subs).
  ///
  /// The boundary facets are characterized by the fact that they
  /// only have 1 cell as super entity.
  pub fn boundary_facets(&self) -> Vec<SimplexIdx> {
    self
      .facets()
      .handle_iter()
      .filter(|f| f.cocells().count() == 1)
      .map(|f| f.idx())
      .collect()
  }

  pub fn boundary_cells(&self) -> Vec<SimplexIdx> {
    self
      .boundary_facets()
      .into_iter()
      // the boundary has only one parent cell by definition
      .map(|facet| {
        facet
          .handle(self)
          .cocells()
          .next()
          .expect("Boundary facets have exactly one cell.")
          .idx()
      })
      .unique()
      .collect()
  }

  /// The vertices that lie on the boundary of the mesh.
  /// No particular order of vertices.
  pub fn boundary_vertices(&self) -> Vec<usize> {
    self
      .boundary_facets()
      .into_iter()
      .flat_map(|facet| facet.handle(self).vertices.clone())
      .unique()
      .collect()
  }

  /// $boundary^k: Delta_k -> Delta_(k-1)$
  pub fn boundary_operator(&self, dim: Dim) -> CooMatrix {
    let sups = &self.skeleton(dim);

    if dim == 0 {
      return CooMatrix::zeros(0, sups.len());
    }

    let subs = &self.skeleton(dim - 1);
    let mut mat = CooMatrix::zeros(subs.len(), sups.len());
    for (isup, sup) in sups.handle_iter().enumerate() {
      let sup_boundary = sup.boundary();
      for sub in sup_boundary {
        let sign = sub.sign.as_f64();
        let isub = subs.handle_by_simplex(&sub.simplex).kidx();
        mat.push(isub, isup, sign);
      }
    }
    mat
  }

  pub fn restricted_boundary_operator(
    &self,
    k: Dim,
    k_minus_one_strong_dof_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_strong_dof_predicate: &dyn Fn(KSimplexIdx) -> bool,
  ) -> CooMatrix {
    let sups = &self.skeleton(k);

    // Map k-simplex index in full basis -> index in reduced basis
    let mut col_of_full: Vec<Option<usize>> = vec![None; sups.len()];
    let mut ncols = 0usize;
    for sup in sups.handle_iter() {
      let full_idx: KSimplexIdx = sup.kidx(); // full-basis index
      debug_assert!(full_idx < sups.len());

      if !k_strong_dof_predicate(full_idx) {
        col_of_full[full_idx] = Some(ncols);
        ncols += 1;
      }
    }

    if k == 0 {
      return CooMatrix::zeros(0, ncols);
    }

    let subs = &self.skeleton(k - 1);

    // Map (k-1)-simplex index in full basis -> index in reduced basis
    let mut row_of_full: Vec<Option<usize>> = vec![None; subs.len()];
    let mut nrows = 0usize;
    for sub in subs.handle_iter() {
      let full_idx: KSimplexIdx = sub.kidx();
      debug_assert!(full_idx < subs.len());

      if !k_minus_one_strong_dof_predicate(full_idx) {
        row_of_full[full_idx] = Some(nrows);
        nrows += 1;
      }
    }

    let mut mat = CooMatrix::zeros(nrows, ncols);

    // Assemble using reduced row/col indices.
    for sup in sups.handle_iter() {
      let full_sup: KSimplexIdx = sup.kidx();
      let full_sup_u: usize = full_sup as usize; // adjust if needed
      let Some(col) = col_of_full[full_sup_u] else {
        continue;
      };

      for face in sup.boundary() {
        let full_sub: KSimplexIdx = subs.handle_by_simplex(&face.simplex).kidx();
        let full_sub_u: usize = full_sub as usize; // adjust if needed
        debug_assert!(full_sub_u < subs.len());

        let Some(row) = row_of_full[full_sub_u] else {
          continue;
        };

        mat.push(row, col, face.sign.as_f64());
      }
    }

    mat
  }

  /// Dimension of the k-th homology group.
  ///
  /// k-th Betti number.
  /// Number of k-dimensional holes in the manifold.
  /// Computed using simplicial homology.
  pub fn homology_dim(&self, dim: Dim) -> usize {
    // TODO: use sparse matrix!
    let boundary_this = Matrix::from(&self.boundary_operator(dim));
    let boundary_plus = Matrix::from(&self.boundary_operator(dim + 1));

    const RANK_TOL: f64 = 1e-12;

    let dim_image = |op: &Matrix| -> usize { op.rank(RANK_TOL) };
    let dim_kernel = |op: &Matrix| -> usize { op.ncols() - dim_image(op) };

    let dim_cycles = dim_kernel(&boundary_this);
    let dim_boundaries = dim_image(&boundary_plus);

    dim_cycles - dim_boundaries
  }

  pub fn relative_homology_dim(
    &self,
    k: Dim,
    k_minus_one_strong_dof_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_strong_dof_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_plus_one_strong_dof_predicate: &dyn Fn(KSimplexIdx) -> bool,
  ) -> usize {
    // TODO: use sparse matrix!
    let boundary_k = Matrix::from(&self.restricted_boundary_operator(
      k,
      k_minus_one_strong_dof_predicate,
      k_strong_dof_predicate,
    ));
    let boundary_k_plus_one = Matrix::from(&self.restricted_boundary_operator(
      k + 1,
      k_strong_dof_predicate,
      k_plus_one_strong_dof_predicate,
    ));

    const RANK_TOL: f64 = 1e-12;

    let dim_image = |op: &Matrix| -> usize { op.rank(RANK_TOL) };
    let dim_kernel = |op: &Matrix| -> usize { op.ncols() - dim_image(op) };

    let dim_cycles = dim_kernel(&boundary_k);
    let dim_boundaries = dim_image(&boundary_k_plus_one);

    dim_cycles - dim_boundaries
  }
}

impl Complex {
  pub fn from_cells(cells: Skeleton) -> Self {
    let dim = cells.dim();

    let mut skeletons = vec![ComplexSkeleton::default(); dim + 1];
    skeletons[0] = ComplexSkeleton {
      skeleton: Skeleton::new((0..cells.nvertices()).map(Simplex::single).collect()),
      complex_data: (0..cells.nvertices())
        .map(|_| SimplexComplexData::default())
        .collect(),
    };

    for (icell, cell) in cells.iter().enumerate() {
      for (
        dim_skeleton,
        ComplexSkeleton {
          skeleton,
          complex_data: mesh_data,
        },
      ) in skeletons.iter_mut().enumerate()
      {
        for sub in cell.subsequences(dim_skeleton) {
          let (sub_idx, is_new) = skeleton.insert(sub);
          let sub_data = if is_new {
            mesh_data.push(SimplexComplexData::default());
            mesh_data.last_mut().unwrap()
          } else {
            &mut mesh_data[sub_idx]
          };
          sub_data.cocells.push(SimplexIdx::new(dim, icell));
        }
      }
    }

    // Topology checks.
    if dim >= 1 {
      let facet_data = skeletons[dim - 1].complex_data();
      for SimplexComplexData { cocells } in facet_data {
        let nparents = cocells.len();
        let is_manifold = nparents == 2 || nparents == 1;
        assert!(is_manifold, "Topology must be manifold.");
      }
    }

    Self { skeletons }
  }
}

#[cfg(test)]
mod test {
  use crate::topology::simplex::{nsubsequence_simplices, Simplex};

  use super::*;

  #[test]
  fn incidence() {
    let dim = 3;
    let complex = Complex::standard(dim);
    let cell = complex.cells().handle_iter().next().unwrap();

    // print
    for dim_sub in 0..=dim {
      let skeleton = complex.skeleton(dim_sub);
      for simp in skeleton.handle_iter() {
        let simp_vertices = &*simp;
        print!("{simp_vertices:?},");
      }
      println!();
    }

    let cell_simplex = Simplex::standard(dim);
    for dim_sub in 0..=dim {
      let subs: Vec<_> = cell.mesh_subsimps(dim_sub).collect();
      assert_eq!(subs.len(), nsubsequence_simplices(dim, dim_sub));
      let subs_vertices: Vec<_> = cell_simplex.subsequences(dim_sub).collect();
      assert_eq!(
        subs.iter().map(|&sub| (*sub).clone()).collect::<Vec<_>>(),
        subs_vertices
      );

      for (isub, sub) in subs.iter().enumerate() {
        let sub_vertices = &subs_vertices[isub];
        for dim_sup in dim_sub..dim {
          let sups: Vec<_> = sub.mesh_supersimps(dim_sup).collect();
          sups
            .iter()
            .all(|sup| sub_vertices.is_subsequence_of(sup) && sup.is_subsequence_of(&cell_simplex));
        }
      }
    }
  }
}
