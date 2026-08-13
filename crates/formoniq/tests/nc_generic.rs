use approx::assert_relative_eq;
use common::linalg::nalgebra::Matrix;
use formoniq::{
  assemble::{
    assemble_nc1_lumped_mass_galmat, assemble_nc1_lumped_mass_inverse_galmat,
    assemble_nc2_lumped_mass_galmat, assemble_nc2_lumped_mass_inverse_galmat,
    assemble_nc2_to_whitney_projection_galmat, assemble_nc_lumped_mass_galmat,
    assemble_nc_lumped_mass_galmat_weighted, assemble_nc_lumped_mass_inverse_galmat_for_grade,
    assemble_nc_lumped_mass_inverse_galmat_weighted_for_grade,
    assemble_nc_to_whitney_projection_galmat,
    assemble_whitney_projected_sparse_inverse_galmat_for_grade,
    assemble_whitney_to_nc2_embedding_galmat,
  },
  operators::InnerProductWeightClosure,
};
use manifold::{
  geometry::{
    coord::mesh::{standard_coord_complex, MeshCoords},
    metric::mesh::MeshLengths,
  },
  topology::complex::Complex,
};

fn standard_metric_complex(dim: usize) -> (Complex, MeshCoords, MeshLengths) {
  let (topology, coords) = standard_coord_complex(dim);
  let metric = coords.to_edge_lengths(&topology);
  (topology, coords, metric)
}

fn simplices_share_vertex(topology: &Complex, grade: usize, i: usize, j: usize) -> bool {
  let simplex_i = topology.skeleton(grade).handle_by_kidx(i);
  let simplex_j = topology.skeleton(grade).handle_by_kidx(j);
  simplex_i.iter().any(|vertex| simplex_j.contains(vertex))
}

#[test]
fn generic_nc_lumped_mass_inverse_matches_dense_formula_on_4_simplex() {
  let (topology, _, metric) = standard_metric_complex(4);

  for grade in 0..topology.dim() {
    let lumped = Matrix::from(&assemble_nc_lumped_mass_galmat(&topology, &metric, grade));
    let inverse = Matrix::from(&assemble_nc_lumped_mass_inverse_galmat_for_grade(
      &topology, &metric, grade,
    ));

    assert_relative_eq!(
      &inverse,
      &lumped.clone().try_inverse().unwrap(),
      epsilon = 1e-12,
      max_relative = 1e-12
    );
  }
}

#[test]
fn generic_whitney_projected_sparse_inverse_matches_dense_formula_on_4_simplex() {
  let (topology, _, metric) = standard_metric_complex(4);

  for grade in 0..topology.dim() {
    let projection = Matrix::from(&assemble_nc_to_whitney_projection_galmat(&topology, grade));
    let lumped = Matrix::from(&assemble_nc_lumped_mass_galmat(&topology, &metric, grade));
    let projected = Matrix::from(&assemble_whitney_projected_sparse_inverse_galmat_for_grade(
      &topology, &metric, grade,
    ));

    let expected = &projection * lumped.clone().try_inverse().unwrap() * projection.transpose();
    assert_relative_eq!(&projected, &expected, epsilon = 1e-12, max_relative = 1e-12);
  }
}

#[test]
fn nc1_and_nc2_lumping_wrappers_support_4d() {
  let (topology, _, metric) = standard_metric_complex(4);

  let nc1_lumped = Matrix::from(&assemble_nc1_lumped_mass_galmat(&topology, &metric));
  let nc1_generic = Matrix::from(&assemble_nc_lumped_mass_galmat(&topology, &metric, 1));
  assert_relative_eq!(&nc1_lumped, &nc1_generic, epsilon = 1e-12);

  let nc1_inverse = Matrix::from(&assemble_nc1_lumped_mass_inverse_galmat(&topology, &metric));
  let nc1_generic_inverse = Matrix::from(&assemble_nc_lumped_mass_inverse_galmat_for_grade(
    &topology, &metric, 1,
  ));
  assert_relative_eq!(&nc1_inverse, &nc1_generic_inverse, epsilon = 1e-12);

  let nc2_lumped = Matrix::from(&assemble_nc2_lumped_mass_galmat(&topology, &metric));
  let nc2_generic = Matrix::from(&assemble_nc_lumped_mass_galmat(&topology, &metric, 2));
  assert_relative_eq!(&nc2_lumped, &nc2_generic, epsilon = 1e-12);

  let nc2_inverse = Matrix::from(&assemble_nc2_lumped_mass_inverse_galmat(&topology, &metric));
  let nc2_generic_inverse = Matrix::from(&assemble_nc_lumped_mass_inverse_galmat_for_grade(
    &topology, &metric, 2,
  ));
  assert_relative_eq!(&nc2_inverse, &nc2_generic_inverse, epsilon = 1e-12);

  let projection = Matrix::from(&assemble_nc2_to_whitney_projection_galmat(&topology));
  let embedding = Matrix::from(&assemble_whitney_to_nc2_embedding_galmat(&topology));
  let ntriangles = topology.skeleton(2).len();
  assert_eq!(projection.nrows(), ntriangles);
  assert_eq!(projection.ncols(), 3 * ntriangles);
  assert_relative_eq!(
    &(projection * embedding),
    &Matrix::identity(ntriangles, ntriangles),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn weighted_generic_nc_lumped_inverse_scales_with_reciprocal_constant_weight_on_4_simplex() {
  let (topology, coords, metric) = standard_metric_complex(4);
  let weight = InnerProductWeightClosure::new(|_| 2.5);

  for grade in [1, 2] {
    let unweighted_inverse = Matrix::from(&assemble_nc_lumped_mass_inverse_galmat_for_grade(
      &topology, &metric, grade,
    ));
    let weighted_lumped = Matrix::from(&assemble_nc_lumped_mass_galmat_weighted(
      &topology, &metric, grade, &coords, None, &weight,
    ));
    let weighted_inverse =
      Matrix::from(&assemble_nc_lumped_mass_inverse_galmat_weighted_for_grade(
        &topology, &metric, grade, &coords, None, &weight,
      ));

    assert_relative_eq!(
      &weighted_lumped,
      &(2.5 * Matrix::from(&assemble_nc_lumped_mass_galmat(&topology, &metric, grade,))),
      epsilon = 1e-12,
      max_relative = 1e-12
    );
    assert_relative_eq!(
      &weighted_inverse,
      &(unweighted_inverse / 2.5),
      epsilon = 1e-12,
      max_relative = 1e-12
    );
  }
}

#[test]
fn projected_sparse_inverse_is_vertex_local_for_4d_1forms_and_2forms() {
  let (topology, _, metric) = standard_metric_complex(4);

  for grade in [1, 2] {
    let projected = Matrix::from(&assemble_whitney_projected_sparse_inverse_galmat_for_grade(
      &topology, &metric, grade,
    ));

    assert_relative_eq!(
      &projected,
      &projected.transpose(),
      epsilon = 1e-12,
      max_relative = 1e-12
    );

    for i in 0..projected.nrows() {
      assert!(projected[(i, i)] > 0.0);
      for j in 0..projected.ncols() {
        if !simplices_share_vertex(&topology, grade, i, j) {
          assert_relative_eq!(projected[(i, j)], 0.0, epsilon = 1e-12);
        }
      }
    }
  }
}
