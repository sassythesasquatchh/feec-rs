use approx::assert_relative_eq;
use common::linalg::nalgebra::Matrix;
use formoniq::{
  assemble::{
    assemble_galmat, assemble_galmat_coord_aware, assemble_nc2_lumped_mass_galmat,
    assemble_nc2_lumped_mass_inverse_galmat, assemble_nc2_lumped_mass_inverse_galmat_weighted,
    assemble_nc2_mass_galmat, assemble_nc2_mass_galmat_weighted,
    assemble_nc2_to_whitney_projection_galmat,
    assemble_whitney_2form_projected_sparse_inverse_galmat,
    assemble_whitney_2form_projected_sparse_inverse_galmat_weighted,
    assemble_whitney_to_nc2_embedding_galmat,
  },
  operators::{HodgeMassElmat, InnerProductWeightClosure},
};
use manifold::{
  gen::cartesian::CartesianMeshInfo,
  geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
  topology::complex::Complex,
};

fn cartesian_metric_complex_3d(ncells_axis: usize) -> (Complex, MeshCoords, MeshLengths) {
  let (topology, coords) = CartesianMeshInfo::new_unit(3, ncells_axis).compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  (topology, coords, metric)
}

fn assert_projection_shape(topology: &Complex, projection: &Matrix) {
  let nfaces = topology.facets().len();
  assert_eq!(projection.nrows(), nfaces);
  assert_eq!(projection.ncols(), 3 * nfaces);

  for iface in 0..nfaces {
    let nonzeros = projection
      .row(iface)
      .iter()
      .enumerate()
      .filter_map(|(icol, &value)| (value.abs() > 1e-12).then_some((icol, value)))
      .collect::<Vec<_>>();

    assert_eq!(nonzeros.len(), 3);
    assert_eq!(nonzeros[0].0, 3 * iface);
    assert_eq!(nonzeros[1].0, 3 * iface + 1);
    assert_eq!(nonzeros[2].0, 3 * iface + 2);
    assert_relative_eq!(nonzeros[0].1, 1.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(nonzeros[1].1, 1.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(nonzeros[2].1, 1.0 / 3.0, epsilon = 1e-12);
  }
}

fn assert_nc2_whitney_consistency(topology: &Complex, metric: &MeshLengths) {
  let nc2_mass = Matrix::from(&assemble_nc2_mass_galmat(topology, metric));
  let embedding = Matrix::from(&assemble_whitney_to_nc2_embedding_galmat(topology));
  let whitney_mass = Matrix::from(&assemble_galmat(
    topology,
    metric,
    HodgeMassElmat::new(topology.dim(), 2),
  ));

  assert_relative_eq!(
    &(embedding.transpose() * nc2_mass * &embedding),
    &whitney_mass,
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

fn assert_weighted_nc2_whitney_consistency_f64(
  topology: &Complex,
  metric: &MeshLengths,
  coords: &MeshCoords,
  weight: &InnerProductWeightClosure<f64>,
) {
  let nc2_mass = Matrix::from(&assemble_nc2_mass_galmat_weighted(
    topology, metric, coords, None, weight,
  ));
  let embedding = Matrix::from(&assemble_whitney_to_nc2_embedding_galmat(topology));
  let whitney_mass = Matrix::from(&assemble_galmat_coord_aware(
    topology,
    metric,
    HodgeMassElmat::new_weighted(topology.dim(), 2, coords, None, weight),
  ));

  assert_relative_eq!(
    &(embedding.transpose() * nc2_mass * &embedding),
    &whitney_mass,
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

fn assert_nc2_lumped_mass_inverse_matches_dense_formula(topology: &Complex, metric: &MeshLengths) {
  let lumped_mass = Matrix::from(&assemble_nc2_lumped_mass_galmat(topology, metric));
  let lumped_inverse = Matrix::from(&assemble_nc2_lumped_mass_inverse_galmat(topology, metric));

  assert_relative_eq!(
    &lumped_inverse,
    &lumped_mass.clone().try_inverse().unwrap(),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

fn assert_whitney_2form_projected_sparse_inverse_matches_dense_formula(
  topology: &Complex,
  metric: &MeshLengths,
) {
  let projection = Matrix::from(&assemble_nc2_to_whitney_projection_galmat(topology));
  let lumped_mass = Matrix::from(&assemble_nc2_lumped_mass_galmat(topology, metric));
  let projected_sparse_inverse = Matrix::from(
    &assemble_whitney_2form_projected_sparse_inverse_galmat(topology, metric),
  );

  let expected = &projection * lumped_mass.clone().try_inverse().unwrap() * projection.transpose();
  assert_relative_eq!(
    &projected_sparse_inverse,
    &expected,
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

fn faces_share_vertex(topology: &Complex, iface: usize, jface: usize) -> bool {
  let face_i = topology.facets().handle_by_kidx(iface);
  let face_j = topology.facets().handle_by_kidx(jface);
  face_i.iter().any(|vertex| face_j.contains(vertex))
}

#[test]
fn nc2_projection_and_embedding_are_inverses_on_tetra_meshes() {
  let (topology, _, _) = cartesian_metric_complex_3d(1);

  let projection = Matrix::from(&assemble_nc2_to_whitney_projection_galmat(&topology));
  let embedding = Matrix::from(&assemble_whitney_to_nc2_embedding_galmat(&topology));

  assert_projection_shape(&topology, &projection);
  assert_relative_eq!(
    &(projection * embedding),
    &Matrix::identity(topology.facets().len(), topology.facets().len()),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn nc2_mass_matches_whitney_mass_on_tetra_meshes() {
  let (topology, _, metric) = cartesian_metric_complex_3d(1);
  assert_nc2_whitney_consistency(&topology, &metric);
}

#[test]
fn weighted_nc2_mass_matches_whitney_mass_for_constant_scalar_weight() {
  let (topology, coords, metric) = cartesian_metric_complex_3d(1);
  let weight = InnerProductWeightClosure::new(|_| 2.5);
  assert_weighted_nc2_whitney_consistency_f64(&topology, &metric, &coords, &weight);
}

#[test]
fn nc2_lumped_mass_inverse_matches_dense_formula_on_tetra_meshes() {
  let (topology, _, metric) = cartesian_metric_complex_3d(1);
  assert_nc2_lumped_mass_inverse_matches_dense_formula(&topology, &metric);
}

#[test]
fn weighted_nc2_lumped_mass_inverse_scales_with_reciprocal_constant_weight() {
  let (topology, coords, metric) = cartesian_metric_complex_3d(1);
  let weight = InnerProductWeightClosure::new(|_| 2.5);

  let unweighted = Matrix::from(&assemble_nc2_lumped_mass_inverse_galmat(&topology, &metric));
  let weighted = Matrix::from(&assemble_nc2_lumped_mass_inverse_galmat_weighted(
    &topology, &metric, &coords, None, &weight,
  ));

  assert_relative_eq!(
    &weighted,
    &(unweighted / 2.5),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn whitney_2form_projected_sparse_inverse_matches_dense_formula_on_tetra_meshes() {
  let (topology, _, metric) = cartesian_metric_complex_3d(1);
  assert_whitney_2form_projected_sparse_inverse_matches_dense_formula(&topology, &metric);
}

#[test]
fn weighted_whitney_2form_projected_sparse_inverse_scales_with_reciprocal_constant_weight() {
  let (topology, coords, metric) = cartesian_metric_complex_3d(1);
  let weight = InnerProductWeightClosure::new(|_| 2.5);

  let unweighted = Matrix::from(&assemble_whitney_2form_projected_sparse_inverse_galmat(
    &topology, &metric,
  ));
  let weighted = Matrix::from(
    &assemble_whitney_2form_projected_sparse_inverse_galmat_weighted(
      &topology, &metric, &coords, None, &weight,
    ),
  );

  assert_relative_eq!(
    &weighted,
    &(unweighted / 2.5),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
}

#[test]
fn projected_2form_sparse_inverse_is_symmetric_positive_and_vertex_local() {
  let (topology, _, metric) = cartesian_metric_complex_3d(1);
  let projected = Matrix::from(&assemble_whitney_2form_projected_sparse_inverse_galmat(
    &topology, &metric,
  ));

  assert_relative_eq!(
    &projected,
    &projected.transpose(),
    epsilon = 1e-12,
    max_relative = 1e-12
  );
  for iface in 0..projected.nrows() {
    assert!(projected[(iface, iface)] > 0.0);
    for jface in 0..projected.ncols() {
      if !faces_share_vertex(&topology, iface, jface) {
        assert!(
          projected[(iface, jface)].abs() <= 1e-12,
          "projected 2-form sparse inverse should remain vertex-local after projection"
        );
      }
    }
  }
}
