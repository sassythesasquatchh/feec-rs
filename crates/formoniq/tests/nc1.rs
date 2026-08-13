use approx::assert_relative_eq;
use common::linalg::nalgebra::Matrix;
use formoniq::{
  assemble::{
    assemble_galmat, assemble_galmat_coord_aware, assemble_nc1_lumped_mass_galmat,
    assemble_nc1_lumped_mass_galmat_weighted, assemble_nc1_lumped_mass_inverse_galmat,
    assemble_nc1_lumped_mass_inverse_galmat_weighted, assemble_nc1_mass_galmat,
    assemble_nc1_mass_galmat_weighted, assemble_nc1_to_whitney_projection_galmat,
    assemble_whitney_projected_sparse_inverse_galmat,
    assemble_whitney_projected_sparse_inverse_galmat_weighted,
    assemble_whitney_to_nc1_embedding_galmat,
  },
  operators::{HodgeMassElmat, InnerProductWeightClosure},
};
use manifold::{
  gen::cartesian::CartesianMeshInfo,
  geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
  topology::complex::Complex,
  Dim,
};

fn cartesian_metric_complex(dim: Dim, ncells_axis: usize) -> (Complex, MeshCoords, MeshLengths) {
  let (topology, coords) = CartesianMeshInfo::new_unit(dim, ncells_axis).compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  (topology, coords, metric)
}

fn assert_projection_shape(topology: &Complex, projection: &Matrix) {
  let nedges = topology.edges().len();
  assert_eq!(projection.nrows(), nedges);
  assert_eq!(projection.ncols(), 2 * nedges);

  for iedge in 0..nedges {
    let nonzeros = projection
      .row(iedge)
      .iter()
      .enumerate()
      .filter_map(|(icol, &value)| (value.abs() > 1e-12).then_some((icol, value)))
      .collect::<Vec<_>>();

    assert_eq!(nonzeros.len(), 2);
    assert_eq!(nonzeros[0].0, 2 * iedge);
    assert_eq!(nonzeros[1].0, 2 * iedge + 1);
    assert_relative_eq!(nonzeros[0].1, 0.5, epsilon = 1e-12);
    assert_relative_eq!(nonzeros[1].1, 0.5, epsilon = 1e-12);
  }
}

fn assert_nc1_whitney_consistency(topology: &Complex, metric: &MeshLengths) {
  let nc1_mass = Matrix::from(&assemble_nc1_mass_galmat(topology, metric));
  let embedding = Matrix::from(&assemble_whitney_to_nc1_embedding_galmat(topology));
  let whitney_mass = Matrix::from(&assemble_galmat(
    topology,
    metric,
    HodgeMassElmat::new(topology.dim(), 1),
  ));

  assert_relative_eq!(
    &(embedding.transpose() * nc1_mass * &embedding),
    &whitney_mass,
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

fn assert_weighted_nc1_whitney_consistency_f64(
  topology: &Complex,
  metric: &MeshLengths,
  coords: &MeshCoords,
  weight: &InnerProductWeightClosure<f64>,
) {
  let nc1_mass = Matrix::from(&assemble_nc1_mass_galmat_weighted(
    topology, metric, coords, None, weight,
  ));
  let embedding = Matrix::from(&assemble_whitney_to_nc1_embedding_galmat(topology));
  let whitney_mass = Matrix::from(&assemble_galmat_coord_aware(
    topology,
    metric,
    HodgeMassElmat::new_weighted(topology.dim(), 1, coords, None, weight),
  ));

  assert_relative_eq!(
    &(embedding.transpose() * nc1_mass * &embedding),
    &whitney_mass,
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

fn assert_nc1_lumped_mass_inverse_matches_dense_formula(topology: &Complex, metric: &MeshLengths) {
  let lumped_mass = Matrix::from(&assemble_nc1_lumped_mass_galmat(topology, metric));
  let lumped_inverse = Matrix::from(&assemble_nc1_lumped_mass_inverse_galmat(topology, metric));

  assert_relative_eq!(
    &lumped_inverse,
    &lumped_mass.clone().try_inverse().unwrap(),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

fn assert_whitney_projected_sparse_inverse_matches_dense_formula(
  topology: &Complex,
  metric: &MeshLengths,
) {
  let projection = Matrix::from(&assemble_nc1_to_whitney_projection_galmat(topology));
  let lumped_mass = Matrix::from(&assemble_nc1_lumped_mass_galmat(topology, metric));
  let projected_sparse_inverse = Matrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
    topology, metric,
  ));

  let expected = &projection * lumped_mass.clone().try_inverse().unwrap() * projection.transpose();
  assert_relative_eq!(
    &projected_sparse_inverse,
    &expected,
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

fn edges_share_vertex(topology: &Complex, iedge: usize, jedge: usize) -> bool {
  let edge_i = topology.edges().handle_by_kidx(iedge);
  let edge_j = topology.edges().handle_by_kidx(jedge);
  edge_i.iter().any(|vertex| edge_j.contains(vertex))
}

#[test]
fn nc1_projection_and_embedding_are_inverses_on_triangle_meshes() {
  let (topology, _, _) = cartesian_metric_complex(2, 2);

  let projection = Matrix::from(&assemble_nc1_to_whitney_projection_galmat(&topology));
  let embedding = Matrix::from(&assemble_whitney_to_nc1_embedding_galmat(&topology));

  assert_projection_shape(&topology, &projection);
  assert_relative_eq!(
    &(projection * embedding),
    &Matrix::identity(topology.edges().len(), topology.edges().len()),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn nc1_projection_and_embedding_are_inverses_on_tetra_meshes() {
  let (topology, _, _) = cartesian_metric_complex(3, 1);

  let projection = Matrix::from(&assemble_nc1_to_whitney_projection_galmat(&topology));
  let embedding = Matrix::from(&assemble_whitney_to_nc1_embedding_galmat(&topology));

  assert_projection_shape(&topology, &projection);
  assert_relative_eq!(
    &(projection * embedding),
    &Matrix::identity(topology.edges().len(), topology.edges().len()),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn nc1_mass_matches_whitney_mass_on_triangle_meshes() {
  let (topology, _, metric) = cartesian_metric_complex(2, 2);
  assert_nc1_whitney_consistency(&topology, &metric);
}

#[test]
fn nc1_mass_matches_whitney_mass_on_tetra_meshes() {
  let (topology, _, metric) = cartesian_metric_complex(3, 1);
  assert_nc1_whitney_consistency(&topology, &metric);
}

#[test]
fn weighted_nc1_mass_matches_whitney_mass_for_constant_scalar_weight() {
  let weight = InnerProductWeightClosure::new(|_| 2.5);

  let (topology2, coords2, metric2) = cartesian_metric_complex(2, 2);
  assert_weighted_nc1_whitney_consistency_f64(&topology2, &metric2, &coords2, &weight);

  let (topology3, coords3, metric3) = cartesian_metric_complex(3, 1);
  assert_weighted_nc1_whitney_consistency_f64(&topology3, &metric3, &coords3, &weight);
}

#[test]
fn weighted_nc1_mass_matches_whitney_mass_for_identity_weight_in_3d() {
  let (topology, coords, metric) = cartesian_metric_complex(3, 1);
  let weight = InnerProductWeightClosure::new(|_| Matrix::identity(3, 3));

  let nc1_mass = Matrix::from(&assemble_nc1_mass_galmat_weighted(
    &topology, &metric, &coords, None, &weight,
  ));
  let embedding = Matrix::from(&assemble_whitney_to_nc1_embedding_galmat(&topology));
  let whitney_mass = Matrix::from(&assemble_galmat_coord_aware(
    &topology,
    &metric,
    HodgeMassElmat::new_weighted(topology.dim(), 1, &coords, None, &weight),
  ));

  assert_relative_eq!(
    &(embedding.transpose() * nc1_mass * &embedding),
    &whitney_mass,
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn weighted_nc1_mass_scales_with_constant_scalar_weight() {
  let (topology2, coords2, metric2) = cartesian_metric_complex(2, 2);
  let (topology3, coords3, metric3) = cartesian_metric_complex(3, 1);
  let weight = InnerProductWeightClosure::new(|_| 2.5);

  let unweighted2 = Matrix::from(&assemble_nc1_mass_galmat(&topology2, &metric2));
  let weighted2 = Matrix::from(&assemble_nc1_mass_galmat_weighted(
    &topology2, &metric2, &coords2, None, &weight,
  ));
  assert_relative_eq!(
    &weighted2,
    &(2.5 * unweighted2),
    epsilon = 1e-12,
    max_relative = 1e-12
  );

  let unweighted3 = Matrix::from(&assemble_nc1_mass_galmat(&topology3, &metric3));
  let weighted3 = Matrix::from(&assemble_nc1_mass_galmat_weighted(
    &topology3, &metric3, &coords3, None, &weight,
  ));
  assert_relative_eq!(
    &weighted3,
    &(2.5 * unweighted3),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn weighted_nc1_mass_identity_weight_matches_unweighted_in_3d() {
  let (topology, coords, metric) = cartesian_metric_complex(3, 1);
  let weight = InnerProductWeightClosure::new(|_| Matrix::identity(3, 3));

  let unweighted = Matrix::from(&assemble_nc1_mass_galmat(&topology, &metric));
  let weighted = Matrix::from(&assemble_nc1_mass_galmat_weighted(
    &topology, &metric, &coords, None, &weight,
  ));

  assert_relative_eq!(
    &weighted,
    &unweighted,
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn nc1_lumped_mass_inverse_matches_dense_formula_on_triangle_meshes() {
  let (topology, _, metric) = cartesian_metric_complex(2, 2);
  assert_nc1_lumped_mass_inverse_matches_dense_formula(&topology, &metric);
}

#[test]
fn nc1_lumped_mass_inverse_matches_dense_formula_on_tetra_meshes() {
  let (topology, _, metric) = cartesian_metric_complex(3, 1);
  assert_nc1_lumped_mass_inverse_matches_dense_formula(&topology, &metric);
}

#[test]
fn weighted_nc1_lumped_mass_inverse_matches_dense_formula_for_constant_scalar_weight() {
  let weight = InnerProductWeightClosure::new(|_| 2.5);

  let (topology2, coords2, metric2) = cartesian_metric_complex(2, 2);
  let lumped2 = Matrix::from(&assemble_nc1_lumped_mass_galmat_weighted(
    &topology2, &metric2, &coords2, None, &weight,
  ));
  let inverse2 = Matrix::from(&assemble_nc1_lumped_mass_inverse_galmat_weighted(
    &topology2, &metric2, &coords2, None, &weight,
  ));
  assert_relative_eq!(
    &inverse2,
    &lumped2.clone().try_inverse().unwrap(),
    epsilon = 1e-12,
    max_relative = 1e-12
  );

  let (topology3, coords3, metric3) = cartesian_metric_complex(3, 1);
  let lumped3 = Matrix::from(&assemble_nc1_lumped_mass_galmat_weighted(
    &topology3, &metric3, &coords3, None, &weight,
  ));
  let inverse3 = Matrix::from(&assemble_nc1_lumped_mass_inverse_galmat_weighted(
    &topology3, &metric3, &coords3, None, &weight,
  ));
  assert_relative_eq!(
    &inverse3,
    &lumped3.clone().try_inverse().unwrap(),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn whitney_projected_sparse_inverse_matches_dense_formula_on_triangle_meshes() {
  let (topology, _, metric) = cartesian_metric_complex(2, 2);
  assert_whitney_projected_sparse_inverse_matches_dense_formula(&topology, &metric);
}

#[test]
fn whitney_projected_sparse_inverse_matches_dense_formula_on_tetra_meshes() {
  let (topology, _, metric) = cartesian_metric_complex(3, 1);
  assert_whitney_projected_sparse_inverse_matches_dense_formula(&topology, &metric);
}

#[test]
fn projected_sparse_inverse_is_symmetric_positive_diagonal_and_vertex_local() {
  for (dim, ncells_axis) in [(2, 2), (3, 1)] {
    let (topology, _, metric) = cartesian_metric_complex(dim, ncells_axis);
    let projected_sparse_inverse = Matrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
      &topology, &metric,
    ));

    assert_relative_eq!(
      &projected_sparse_inverse,
      &projected_sparse_inverse.transpose(),
      epsilon = 1e-12,
      max_relative = 1e-12
    );

    for iedge in 0..topology.edges().len() {
      assert!(projected_sparse_inverse[(iedge, iedge)] > 0.0);
      for jedge in 0..topology.edges().len() {
        if !edges_share_vertex(&topology, iedge, jedge) {
          assert_relative_eq!(
            projected_sparse_inverse[(iedge, jedge)],
            0.0,
            epsilon = 1e-12,
            max_relative = 1e-12
          );
        }
      }
    }
  }
}

#[test]
fn weighted_projected_sparse_inverse_scales_with_reciprocal_constant_scalar_weight() {
  let weight = InnerProductWeightClosure::new(|_| 2.5);

  let (topology2, coords2, metric2) = cartesian_metric_complex(2, 2);
  let unweighted2 = Matrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
    &topology2, &metric2,
  ));
  let weighted2 = Matrix::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
    &topology2, &metric2, &coords2, None, &weight,
  ));
  assert_relative_eq!(
    &weighted2,
    &(unweighted2 / 2.5),
    epsilon = 1e-12,
    max_relative = 1e-12
  );

  let (topology3, coords3, metric3) = cartesian_metric_complex(3, 1);
  let unweighted3 = Matrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
    &topology3, &metric3,
  ));
  let weighted3 = Matrix::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
    &topology3, &metric3, &coords3, None, &weight,
  ));
  assert_relative_eq!(
    &weighted3,
    &(unweighted3 / 2.5),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}
