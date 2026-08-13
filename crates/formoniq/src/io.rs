use std::{
  fs::{self, File},
  io::{self, BufWriter, Write},
  path::Path,
};

use ddf::{cochain::Cochain, whitney::form::WhitneyForm};
use manifold::{
  geometry::coord::{mesh::MeshCoords, simplex::SimplexCoords},
  topology::{complex::Complex, handle::SkeletonHandle},
};

pub fn write_cochain(path: &str, cochain: &Cochain) -> std::io::Result<()> {
  let mut file = fs::File::create(path)?;
  for coeff in cochain.coeffs.iter() {
    writeln!(file, "{coeff:.12e}")?;
  }
  Ok(())
}

fn vtk_cell_type(k: usize) -> Option<u32> {
  match k {
    0 => Some(1),  // VTK_VERTEX
    1 => Some(3),  // VTK_LINE
    2 => Some(5),  // VTK_TRIANGLE
    3 => Some(10), // VTK_TETRA
    _ => None,
  }
}

/// Write multiple scalar cochain fields to a legacy VTK (ASCII) unstructured grid.
///
/// The `degree` determines where the data live:
/// - degree 0 writes top-dimensional mesh geometry with POINT_DATA on vertices.
/// - degree k > 0 writes the k-skeleton with CELL_DATA on k-cells.
///
/// Coordinates with dimension < 3 are zero-padded to 3D for VTK.
pub fn write_cochain_vtk_fields(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  degree: usize,
  fields: &[(&str, &Cochain)],
) -> io::Result<()> {
  if fields.is_empty() {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "at least one cochain field is required",
    ));
  }

  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "VTK export supports up to 3D coordinates",
    ));
  }

  let topo_dim = topology.dim();
  if degree > topo_dim {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      format!("Cochain degree {degree} exceeds topology dimension {topo_dim}"),
    ));
  }

  let geom_k = if degree == 0 { topo_dim } else { degree };
  let cell_type = vtk_cell_type(geom_k).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::InvalidInput,
      format!("Unsupported cell dimension {geom_k}"),
    )
  })?;
  let geom_skeleton = topology.skeleton(geom_k);
  let ncells = geom_skeleton.len();
  let nverts_per_cell = geom_k + 1;
  let expected_data_len = if degree == 0 {
    coords.nvertices()
  } else {
    topology.skeleton(degree).len()
  };

  for (name, cochain) in fields {
    if cochain.dim() != degree {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("Expected a {degree}-cochain for field {name}"),
      ));
    }
    if cochain.len() != expected_data_len {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
          "Cochain length {} does not match expected data size {} for field {name}",
          cochain.len(),
          expected_data_len
        ),
      ));
    }
  }

  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  writeln!(w, "# vtk DataFile Version 4.2")?;
  writeln!(w, "{degree}-cochain fields")?;
  writeln!(w, "ASCII")?;
  writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

  write_vtk_points(&mut w, coords)?;

  writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
  write_skeleton_cells(&mut w, &geom_skeleton)?;

  writeln!(w, "CELL_TYPES {}", ncells)?;
  for _ in 0..ncells {
    writeln!(w, "{cell_type}")?;
  }

  if degree == 0 {
    writeln!(w, "POINT_DATA {}", expected_data_len)?;
  } else {
    writeln!(w, "CELL_DATA {}", expected_data_len)?;
  }
  for (name, cochain) in fields {
    write_vtk_scalar(&mut w, name, cochain.coeffs.iter().copied())?;
  }

  Ok(())
}

/// Write multiple scalar cochain fields to a VTK XML UnstructuredGrid (`.vtu`).
///
/// This mirrors [`write_cochain_vtk_fields`] while using the XML VTU container:
/// - degree 0 writes top-dimensional mesh geometry with `PointData` on vertices.
/// - degree k > 0 writes the k-skeleton with `CellData` on k-cells.
pub fn write_cochain_vtu_fields(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  degree: usize,
  fields: &[(&str, &Cochain)],
) -> io::Result<()> {
  if fields.is_empty() {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "at least one cochain field is required",
    ));
  }

  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "VTK export supports up to 3D coordinates",
    ));
  }

  let topo_dim = topology.dim();
  if degree > topo_dim {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      format!("Cochain degree {degree} exceeds topology dimension {topo_dim}"),
    ));
  }

  let geom_k = if degree == 0 { topo_dim } else { degree };
  let cell_type = vtk_cell_type(geom_k).ok_or_else(|| {
    io::Error::new(
      io::ErrorKind::InvalidInput,
      format!("Unsupported cell dimension {geom_k}"),
    )
  })?;
  let geom_skeleton = topology.skeleton(geom_k);
  let expected_data_len = if degree == 0 {
    coords.nvertices()
  } else {
    topology.skeleton(degree).len()
  };

  for (name, cochain) in fields {
    if cochain.dim() != degree {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("Expected a {degree}-cochain for field {name}"),
      ));
    }
    if cochain.len() != expected_data_len {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
          "Cochain length {} does not match expected data size {} for field {name}",
          cochain.len(),
          expected_data_len
        ),
      ));
    }
  }

  let scalar_fields = fields
    .iter()
    .map(|(name, cochain)| (*name, cochain.coeffs.as_slice()))
    .collect::<Vec<_>>();

  if degree == 0 {
    write_vtu_unstructured_grid(
      path,
      coords,
      &geom_skeleton,
      cell_type,
      VtuDataFields {
        point_scalars: &scalar_fields,
        point_vectors: &[],
        cell_scalars: &[],
        cell_vectors: &[],
      },
    )
  } else {
    write_vtu_unstructured_grid(
      path,
      coords,
      &geom_skeleton,
      cell_type,
      VtuDataFields {
        point_scalars: &[],
        point_vectors: &[],
        cell_scalars: &scalar_fields,
        cell_vectors: &[],
      },
    )
  }
}

/// Write a cochain to a legacy VTK (ASCII) unstructured grid.
///
/// The cochain degree determines which simplices are written:
/// - 0-cochains -> vertices as 1-vertex cells
/// - 1-cochains -> edges as line cells
/// - 2-cochains -> triangles
/// - 3-cochains -> tetrahedra
///
/// Coordinates with dimension < 3 are zero-padded to 3D for VTK.
pub fn write_cochain_vtk(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  write_cochain_vtk_fields(
    path,
    coords,
    topology,
    cochain.dim(),
    &[(data_name, cochain)],
  )
}

/// Write a cochain to a VTK XML UnstructuredGrid (`.vtu`).
pub fn write_cochain_vtu(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  write_cochain_vtu_fields(
    path,
    coords,
    topology,
    cochain.dim(),
    &[(data_name, cochain)],
  )
}

/// Sample a Whitney 1-form at cell barycenters and export as a vector field to VTK (CELL_DATA).
///
/// - The provided cochain must have degree 1.
/// - The vectors are piecewise constant: one vector per top-dimensional cell, evaluated at its barycenter.
/// - Ambient-valued Whitney evaluation handles embedded meshes directly.
pub fn sample_1form_cell_vectors(
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
) -> io::Result<Vec<[f64; 3]>> {
  if cochain.dim() != 1 {
    return Err(io::Error::other(format!(
      "Expected a 1-cochain, got dim {}",
      cochain.dim()
    )));
  }

  if coords.dim() > 3 {
    return Err(io::Error::other("VTK export supports up to 3D coordinates"));
  }

  let topo_dim = topology.dim();
  if topo_dim > coords.dim() {
    return Err(io::Error::other(format!(
      "Invalid mesh dimensions: topology dim {} > coordinate dim {}",
      topo_dim,
      coords.dim()
    )));
  }

  let edge_skeleton = topology.skeleton(1);
  if cochain.len() != edge_skeleton.len() {
    return Err(io::Error::other(format!(
      "Cochain length {} does not match edge skeleton size {}",
      cochain.len(),
      edge_skeleton.len()
    )));
  }

  let geom_skeleton = topology.skeleton(topo_dim);
  let whitney = WhitneyForm::new(cochain.clone(), topology, coords);

  let mut vectors = Vec::with_capacity(geom_skeleton.len());
  for cell in geom_skeleton.handle_iter() {
    let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
    let bary = cell_coords.barycenter();
    let value = whitney.eval_known_cell(cell, &bary).into_grade1();

    vectors.push([
      value[0],
      if value.len() > 1 { value[1] } else { 0.0 },
      if value.len() > 2 { value[2] } else { 0.0 },
    ]);
  }

  Ok(vectors)
}

/// Sample a Whitney 2-form at cell barycenters and Hodge-dual it to ambient vectors.
///
/// Assumptions:
/// - The provided cochain has degree 2 in a 3D mesh.
/// - Coordinates are Euclidean; the Hodge dual reduces to the standard
///   pseudovector mapping: (c01, c02, c12) -> (c12, -c02, c01).
pub fn sample_2form_cell_vectors(
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
) -> io::Result<Vec<[f64; 3]>> {
  if cochain.dim() != 2 {
    return Err(io::Error::other(format!(
      "Expected a 2-cochain, got dim {}",
      cochain.dim()
    )));
  }

  if coords.dim() != 3 || topology.dim() != 3 {
    return Err(io::Error::other(
      "sample_2form_cell_vectors supports 3D meshes only",
    ));
  }

  let face_skeleton = topology.skeleton(2);
  if cochain.len() != face_skeleton.len() {
    return Err(io::Error::other(format!(
      "Cochain length {} does not match face skeleton size {}",
      cochain.len(),
      face_skeleton.len()
    )));
  }

  let topo_dim = topology.dim();
  let geom_skeleton = topology.skeleton(topo_dim);
  let whitney = WhitneyForm::new(cochain.clone(), topology, coords);

  let mut vectors = Vec::with_capacity(geom_skeleton.len());
  for cell in geom_skeleton.handle_iter() {
    let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
    let bary = cell_coords.barycenter();
    let value = whitney.eval_known_cell(cell, &bary);

    let coeffs = value.coeffs();
    assert!(
      coeffs.len() == 3,
      "Expected 3 coefficients for a 2-form in 3D"
    );
    let c01 = coeffs[0];
    let c02 = coeffs[1];
    let c12 = coeffs[2];

    vectors.push([c12, -c02, c01]);
  }

  Ok(vectors)
}

/// Write vector and scalar fields defined on top-dimensional cells into a single VTK file.
pub fn write_top_cell_vtk_fields(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  vector_fields: &[(&str, &[[f64; 3]])],
  scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
  if vector_fields.is_empty() && scalar_fields.is_empty() {
    return Err(io::Error::other(
      "at least one top-cell vector or scalar field is required",
    ));
  }

  if coords.dim() > 3 {
    return Err(io::Error::other("VTK export supports up to 3D coordinates"));
  }

  let topo_dim = topology.dim();
  let cell_type = vtk_cell_type(topo_dim)
    .ok_or_else(|| io::Error::other(format!("Unsupported cell dimension {topo_dim}")))?;
  let geom_skeleton = topology.skeleton(topo_dim);
  let ncells = geom_skeleton.len();
  let nverts_per_cell = topo_dim + 1;

  for (name, vectors) in vector_fields {
    if vectors.len() != ncells {
      return Err(io::Error::other(format!(
        "Vector field length {} does not match top-cell count {} for {name}",
        vectors.len(),
        ncells,
      )));
    }
  }
  for (name, field) in scalar_fields {
    if field.len() != ncells {
      return Err(io::Error::other(format!(
        "Scalar field length {} does not match top-cell count {} for {name}",
        field.len(),
        ncells,
      )));
    }
  }

  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  writeln!(w, "# vtk DataFile Version 4.2")?;
  writeln!(w, "top-cell fields")?;
  writeln!(w, "ASCII")?;
  writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

  writeln!(w, "POINTS {} double", coords.nvertices())?;
  for coord in coords.coord_iter() {
    let x = coord[0];
    let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
    let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
    writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
  }

  writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
  write_skeleton_cells(&mut w, &geom_skeleton)?;

  writeln!(w, "CELL_TYPES {}", ncells)?;
  for _ in 0..ncells {
    writeln!(w, "{cell_type}")?;
  }

  writeln!(w, "CELL_DATA {}", ncells)?;
  for (name, vectors) in vector_fields {
    writeln!(w, "VECTORS {} double", name)?;
    for [vx, vy, vz] in vectors.iter() {
      writeln!(w, "{vx:.12e} {vy:.12e} {vz:.12e}")?;
    }
  }
  for (name, field) in scalar_fields {
    writeln!(w, "SCALARS {} double 1", name)?;
    writeln!(w, "LOOKUP_TABLE default")?;
    for value in field.iter() {
      writeln!(w, "{value:.12e}")?;
    }
  }

  Ok(())
}

/// Write vector and scalar fields defined on top-dimensional cells to VTU.
pub fn write_top_cell_vtu_fields(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  vector_fields: &[(&str, &[[f64; 3]])],
  scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
  if vector_fields.is_empty() && scalar_fields.is_empty() {
    return Err(io::Error::other(
      "at least one top-cell vector or scalar field is required",
    ));
  }

  if coords.dim() > 3 {
    return Err(io::Error::other("VTK export supports up to 3D coordinates"));
  }

  let topo_dim = topology.dim();
  let cell_type = vtk_cell_type(topo_dim)
    .ok_or_else(|| io::Error::other(format!("Unsupported cell dimension {topo_dim}")))?;
  let geom_skeleton = topology.skeleton(topo_dim);
  let ncells = geom_skeleton.len();

  for (name, vectors) in vector_fields {
    if vectors.len() != ncells {
      return Err(io::Error::other(format!(
        "Vector field length {} does not match top-cell count {} for {name}",
        vectors.len(),
        ncells,
      )));
    }
  }
  for (name, field) in scalar_fields {
    if field.len() != ncells {
      return Err(io::Error::other(format!(
        "Scalar field length {} does not match top-cell count {} for {name}",
        field.len(),
        ncells,
      )));
    }
  }

  write_vtu_unstructured_grid(
    path,
    coords,
    &geom_skeleton,
    cell_type,
    VtuDataFields {
      point_scalars: &[],
      point_vectors: &[],
      cell_scalars: scalar_fields,
      cell_vectors: vector_fields,
    },
  )
}

pub fn write_1form_vector_field_vtk(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  let vectors = sample_1form_cell_vectors(coords, topology, cochain)?;
  write_top_cell_vtk_fields(
    path,
    coords,
    topology,
    &[(data_name, vectors.as_slice())],
    &[],
  )
}

pub fn write_1form_vector_field_vtu(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  let vectors = sample_1form_cell_vectors(coords, topology, cochain)?;
  write_top_cell_vtu_fields(
    path,
    coords,
    topology,
    &[(data_name, vectors.as_slice())],
    &[],
  )
}

/// Write vector proxies for a 1-form as edge-aligned vectors (CELL_DATA on the 1-skeleton).
///
/// Each proxy is computed as:
///   v = (cochain(edge) / |edge|) * edge_direction
/// which yields a piecewise-constant vector along each oriented edge.
pub fn write_1form_vector_proxy_vtk_fields(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  vector_name: &str,
  vector_cochain: &Cochain,
  scalar_fields: &[(&str, &Cochain)],
) -> io::Result<()> {
  if vector_cochain.dim() != 1 {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      format!(
        "Expected a 1-cochain for vectors ({vector_name}), got dim {}",
        vector_cochain.dim()
      ),
    ));
  }

  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "VTK export supports up to 3D coordinates",
    ));
  }

  let edge_skeleton = topology.skeleton(1);
  let ncells = edge_skeleton.len();
  if vector_cochain.len() != ncells {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      format!(
        "Cochain length {} does not match edge skeleton size {} for vectors ({vector_name})",
        vector_cochain.len(),
        ncells
      ),
    ));
  }
  for (name, cochain) in scalar_fields {
    if cochain.dim() != 1 {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("Expected a 1-cochain for field {name}"),
      ));
    }
    if cochain.len() != ncells {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
          "Cochain length {} does not match edge skeleton size {} for field {name}",
          cochain.len(),
          ncells
        ),
      ));
    }
  }

  let cell_type = vtk_cell_type(1)
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Unsupported edge cell type"))?;

  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  writeln!(w, "# vtk DataFile Version 4.2")?;
  writeln!(w, "1-form vector proxy fields")?;
  writeln!(w, "ASCII")?;
  writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

  write_vtk_points(&mut w, coords)?;

  // Cells: edges
  let nverts_per_cell = 2;
  writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
  write_skeleton_cells(&mut w, &edge_skeleton)?;

  writeln!(w, "CELL_TYPES {}", ncells)?;
  for _ in 0..ncells {
    writeln!(w, "{cell_type}")?;
  }

  // Data: edge-aligned vector proxies
  writeln!(w, "CELL_DATA {}", ncells)?;
  writeln!(w, "VECTORS {} double", vector_name)?;

  for edge in edge_skeleton.handle_iter() {
    let v0 = coords.coord(edge.vertices[0]);
    let v1 = coords.coord(edge.vertices[1]);
    let mut dir = (v1 - v0).into_owned();
    let length = dir.norm();
    if length > 0.0 {
      let scale = vector_cochain[edge] / length;
      dir *= scale;
    } else {
      dir.fill(0.0);
    }

    let vx = dir[0];
    let vy = if dir.len() > 1 { dir[1] } else { 0.0 };
    let vz = if dir.len() > 2 { dir[2] } else { 0.0 };
    writeln!(w, "{vx:.12e} {vy:.12e} {vz:.12e}")?;
  }

  for (name, cochain) in scalar_fields {
    write_vtk_scalar(&mut w, name, cochain.coeffs.iter().copied())?;
  }

  Ok(())
}

/// Write vector proxies for a 1-form as edge-aligned vectors in VTU.
pub fn write_1form_vector_proxy_vtu_fields(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  vector_name: &str,
  vector_cochain: &Cochain,
  scalar_fields: &[(&str, &Cochain)],
) -> io::Result<()> {
  if vector_cochain.dim() != 1 {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      format!(
        "Expected a 1-cochain for vectors ({vector_name}), got dim {}",
        vector_cochain.dim()
      ),
    ));
  }

  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "VTK export supports up to 3D coordinates",
    ));
  }

  let edge_skeleton = topology.skeleton(1);
  let ncells = edge_skeleton.len();
  if vector_cochain.len() != ncells {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      format!(
        "Cochain length {} does not match edge skeleton size {} for vectors ({vector_name})",
        vector_cochain.len(),
        ncells
      ),
    ));
  }
  for (name, cochain) in scalar_fields {
    if cochain.dim() != 1 {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("Expected a 1-cochain for field {name}"),
      ));
    }
    if cochain.len() != ncells {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
          "Cochain length {} does not match edge skeleton size {} for field {name}",
          cochain.len(),
          ncells
        ),
      ));
    }
  }

  let cell_type = vtk_cell_type(1)
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Unsupported edge cell type"))?;

  let mut vectors = Vec::with_capacity(ncells);
  for edge in edge_skeleton.handle_iter() {
    let v0 = coords.coord(edge.vertices[0]);
    let v1 = coords.coord(edge.vertices[1]);
    let mut dir = (v1 - v0).into_owned();
    let length = dir.norm();
    if length > 0.0 {
      let scale = vector_cochain[edge] / length;
      dir *= scale;
    } else {
      dir.fill(0.0);
    }

    vectors.push([
      dir[0],
      if dir.len() > 1 { dir[1] } else { 0.0 },
      if dir.len() > 2 { dir[2] } else { 0.0 },
    ]);
  }

  let scalar_fields = scalar_fields
    .iter()
    .map(|(name, cochain)| (*name, cochain.coeffs.as_slice()))
    .collect::<Vec<_>>();

  write_vtu_unstructured_grid(
    path,
    coords,
    &edge_skeleton,
    cell_type,
    VtuDataFields {
      point_scalars: &[],
      point_vectors: &[],
      cell_scalars: &scalar_fields,
      cell_vectors: &[(vector_name, vectors.as_slice())],
    },
  )
}

pub fn write_1form_vector_proxy_vtk(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  write_1form_vector_proxy_vtk_fields(path, coords, topology, data_name, cochain, &[])
}

pub fn write_1form_vector_proxy_vtu(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  write_1form_vector_proxy_vtu_fields(path, coords, topology, data_name, cochain, &[])
}

/// Write polyline paths with scalar CELL_DATA to a legacy VTK (ASCII) POLYDATA file.
///
/// Each path is a sequence of vertex indices into `coords`; each scalar field must have one
/// value per path.
pub fn write_polyline_vtk_fields(
  path: impl AsRef<Path>,
  title: &str,
  coords: &MeshCoords,
  paths: &[&[usize]],
  cell_scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "VTK export supports up to 3D coordinates",
    ));
  }

  for (path_index, path_vertices) in paths.iter().enumerate() {
    if path_vertices.len() < 2 {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("Polyline path {path_index} has fewer than two vertices"),
      ));
    }
    for &vertex in *path_vertices {
      if vertex >= coords.nvertices() {
        return Err(io::Error::new(
          io::ErrorKind::InvalidInput,
          format!(
            "Polyline path {path_index} references vertex {vertex}, but mesh has {} vertices",
            coords.nvertices()
          ),
        ));
      }
    }
  }

  for (name, field) in cell_scalar_fields {
    if field.len() != paths.len() {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
          "Scalar field length {} does not match polyline count {} for field {name}",
          field.len(),
          paths.len()
        ),
      ));
    }
  }

  let total_points = paths.iter().map(|path| path.len()).sum::<usize>();
  let total_line_size = paths.iter().map(|path| path.len() + 1).sum::<usize>();

  let file = File::create(path)?;
  let mut w = BufWriter::new(file);
  writeln!(w, "# vtk DataFile Version 4.2")?;
  writeln!(w, "{title}")?;
  writeln!(w, "ASCII")?;
  writeln!(w, "DATASET POLYDATA")?;
  writeln!(w, "POINTS {} double", total_points)?;
  for path_vertices in paths {
    for &vertex in *path_vertices {
      let coord = coords.coord(vertex);
      let x = coord[0];
      let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
      let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
      writeln!(w, "{x:.12} {y:.12} {z:.12}")?;
    }
  }

  writeln!(w, "LINES {} {}", paths.len(), total_line_size)?;
  let mut point_offset = 0usize;
  for path_vertices in paths {
    write!(w, "{}", path_vertices.len())?;
    for point_index in 0..path_vertices.len() {
      write!(w, " {}", point_offset + point_index)?;
    }
    writeln!(w)?;
    point_offset += path_vertices.len();
  }

  if !cell_scalar_fields.is_empty() {
    writeln!(w, "CELL_DATA {}", paths.len())?;
    for (name, values) in cell_scalar_fields {
      write_vtk_scalar(&mut w, name, values.iter().copied())?;
    }
  }

  Ok(())
}

/// Write polyline paths with scalar `CellData` to a VTU line-cell grid.
///
/// The point coordinates are duplicated per path, matching the legacy polydata
/// writer's topology so side-by-side validation is straightforward.
pub fn write_polyline_vtu_fields(
  path: impl AsRef<Path>,
  _title: &str,
  coords: &MeshCoords,
  paths: &[&[usize]],
  cell_scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
  if coords.dim() > 3 {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "VTK export supports up to 3D coordinates",
    ));
  }

  for (path_index, path_vertices) in paths.iter().enumerate() {
    if path_vertices.len() < 2 {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("Polyline path {path_index} has fewer than two vertices"),
      ));
    }
    for &vertex in *path_vertices {
      if vertex >= coords.nvertices() {
        return Err(io::Error::new(
          io::ErrorKind::InvalidInput,
          format!(
            "Polyline path {path_index} references vertex {vertex}, but mesh has {} vertices",
            coords.nvertices()
          ),
        ));
      }
    }
  }

  for (name, field) in cell_scalar_fields {
    if field.len() != paths.len() {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
          "Scalar field length {} does not match polyline count {} for field {name}",
          field.len(),
          paths.len()
        ),
      ));
    }
  }

  let total_points = paths.iter().map(|path| path.len()).sum::<usize>();
  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  write_vtu_open(&mut w, total_points, paths.len())?;
  write_vtu_data_section(&mut w, "PointData", &[], &[])?;
  write_vtu_data_section(&mut w, "CellData", cell_scalar_fields, &[])?;
  write_vtu_polyline_points(&mut w, coords, paths)?;
  write_vtu_polyline_cells(&mut w, paths)?;
  write_vtu_close(&mut w)?;

  Ok(())
}

/// Sample a Whitney 2-form at cell barycenters, Hodge-dual it to a vector field,
/// and export as VTK (CELL_DATA).
///
/// Assumptions:
/// - The provided cochain has degree 2 in a 3D mesh.
/// - Coordinates are Euclidean; the Hodge dual reduces to the standard
///   pseudovector mapping: (c01, c02, c12) -> (c12, -c02, c01).
pub fn write_2form_vector_field_vtk(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  if cochain.dim() != 2 {
    return Err(io::Error::other(format!(
      "Expected a 2-cochain, got dim {}",
      cochain.dim()
    )));
  }

  if coords.dim() != 3 || topology.dim() != 3 {
    return Err(io::Error::other(
      "write_2form_vector_field_vtk supports 3D meshes only",
    ));
  }

  let topo_dim = topology.dim();
  let cell_type = vtk_cell_type(topo_dim)
    .ok_or_else(|| io::Error::other(format!("Unsupported cell dimension {topo_dim}")))?;

  let geom_skeleton = topology.skeleton(topo_dim);

  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  writeln!(w, "# vtk DataFile Version 4.2")?;
  writeln!(w, "{data_name}")?;
  writeln!(w, "ASCII")?;
  writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

  // Points
  writeln!(w, "POINTS {} double", coords.nvertices())?;
  for coord in coords.coord_iter() {
    let x = coord[0];
    let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
    let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
    writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
  }

  // Cells
  let nverts_per_cell = topo_dim + 1;
  let ncells = geom_skeleton.len();
  writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
  write_skeleton_cells(&mut w, &geom_skeleton)?;

  writeln!(w, "CELL_TYPES {}", ncells)?;
  for _ in 0..ncells {
    writeln!(w, "{cell_type}")?;
  }

  let vectors = sample_2form_cell_vectors(coords, topology, cochain)?;

  writeln!(w, "CELL_DATA {}", ncells)?;
  writeln!(w, "VECTORS {} double", data_name)?;

  for [vx, vy, vz] in vectors {
    writeln!(w, "{vx:.12e} {vy:.12e} {vz:.12e}")?;
  }

  Ok(())
}

/// Sample a Whitney 2-form at cell barycenters, Hodge-dual it, and export as VTU.
pub fn write_2form_vector_field_vtu(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  topology: &Complex,
  cochain: &Cochain,
  data_name: &str,
) -> io::Result<()> {
  if cochain.dim() != 2 {
    return Err(io::Error::other(format!(
      "Expected a 2-cochain, got dim {}",
      cochain.dim()
    )));
  }

  if coords.dim() != 3 || topology.dim() != 3 {
    return Err(io::Error::other(
      "write_2form_vector_field_vtu supports 3D meshes only",
    ));
  }

  let vectors = sample_2form_cell_vectors(coords, topology, cochain)?;
  write_top_cell_vtu_fields(
    path,
    coords,
    topology,
    &[(data_name, vectors.as_slice())],
    &[],
  )
}

struct VtuDataFields<'a> {
  point_scalars: &'a [(&'a str, &'a [f64])],
  point_vectors: &'a [(&'a str, &'a [[f64; 3]])],
  cell_scalars: &'a [(&'a str, &'a [f64])],
  cell_vectors: &'a [(&'a str, &'a [[f64; 3]])],
}

fn write_vtu_unstructured_grid(
  path: impl AsRef<Path>,
  coords: &MeshCoords,
  skeleton: &SkeletonHandle,
  cell_type: u32,
  fields: VtuDataFields<'_>,
) -> io::Result<()> {
  let file = File::create(path)?;
  let mut w = BufWriter::new(file);

  write_vtu_open(&mut w, coords.nvertices(), skeleton.len())?;
  write_vtu_data_section(
    &mut w,
    "PointData",
    fields.point_scalars,
    fields.point_vectors,
  )?;
  write_vtu_data_section(&mut w, "CellData", fields.cell_scalars, fields.cell_vectors)?;
  write_vtu_points(&mut w, coords)?;
  write_vtu_skeleton_cells(&mut w, skeleton, cell_type)?;
  write_vtu_close(&mut w)?;

  Ok(())
}

fn write_vtu_open(mut w: impl Write, npoints: usize, ncells: usize) -> io::Result<()> {
  writeln!(
    w,
    "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
  )?;
  writeln!(w, "  <UnstructuredGrid>")?;
  writeln!(
    w,
    "    <Piece NumberOfPoints=\"{npoints}\" NumberOfCells=\"{ncells}\">"
  )?;
  Ok(())
}

fn write_vtu_close(mut w: impl Write) -> io::Result<()> {
  writeln!(w, "    </Piece>")?;
  writeln!(w, "  </UnstructuredGrid>")?;
  writeln!(w, "</VTKFile>")?;
  Ok(())
}

fn write_vtu_data_section(
  mut w: impl Write,
  tag: &str,
  scalar_fields: &[(&str, &[f64])],
  vector_fields: &[(&str, &[[f64; 3]])],
) -> io::Result<()> {
  if scalar_fields.is_empty() && vector_fields.is_empty() {
    writeln!(w, "      <{tag}/>")?;
    return Ok(());
  }

  write!(w, "      <{tag}")?;
  if let Some((name, _)) = scalar_fields.first() {
    write!(w, " Scalars=\"{}\"", xml_escape(name))?;
  }
  if let Some((name, _)) = vector_fields.first() {
    write!(w, " Vectors=\"{}\"", xml_escape(name))?;
  }
  writeln!(w, ">")?;

  for (name, values) in scalar_fields {
    write_vtu_scalar_data_array(&mut w, name, values.iter().copied())?;
  }
  for (name, vectors) in vector_fields {
    write_vtu_vector_data_array(&mut w, name, vectors)?;
  }

  writeln!(w, "      </{tag}>")?;
  Ok(())
}

fn write_vtu_scalar_data_array(
  mut w: impl Write,
  name: &str,
  values: impl IntoIterator<Item = f64>,
) -> io::Result<()> {
  writeln!(
    w,
    "        <DataArray type=\"Float64\" Name=\"{}\" NumberOfComponents=\"1\" format=\"ascii\">",
    xml_escape(name)
  )?;
  write!(w, "          ")?;
  for value in values {
    write!(w, "{value:.12e} ")?;
  }
  writeln!(w)?;
  writeln!(w, "        </DataArray>")?;
  Ok(())
}

fn write_vtu_vector_data_array(
  mut w: impl Write,
  name: &str,
  vectors: &[[f64; 3]],
) -> io::Result<()> {
  writeln!(
    w,
    "        <DataArray type=\"Float64\" Name=\"{}\" NumberOfComponents=\"3\" format=\"ascii\">",
    xml_escape(name)
  )?;
  for [vx, vy, vz] in vectors {
    writeln!(w, "          {vx:.12e} {vy:.12e} {vz:.12e}")?;
  }
  writeln!(w, "        </DataArray>")?;
  Ok(())
}

fn write_vtu_points(mut w: impl Write, coords: &MeshCoords) -> io::Result<()> {
  writeln!(w, "      <Points>")?;
  writeln!(
    w,
    "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"
  )?;
  for coord in coords.coord_iter() {
    let x = coord[0];
    let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
    let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
    writeln!(w, "          {x:.12e} {y:.12e} {z:.12e}")?;
  }
  writeln!(w, "        </DataArray>")?;
  writeln!(w, "      </Points>")?;
  Ok(())
}

fn write_vtu_polyline_points(
  mut w: impl Write,
  coords: &MeshCoords,
  paths: &[&[usize]],
) -> io::Result<()> {
  writeln!(w, "      <Points>")?;
  writeln!(
    w,
    "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"
  )?;
  for path_vertices in paths {
    for &vertex in *path_vertices {
      let coord = coords.coord(vertex);
      let x = coord[0];
      let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
      let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
      writeln!(w, "          {x:.12e} {y:.12e} {z:.12e}")?;
    }
  }
  writeln!(w, "        </DataArray>")?;
  writeln!(w, "      </Points>")?;
  Ok(())
}

fn write_vtu_skeleton_cells(
  mut w: impl Write,
  skeleton: &SkeletonHandle,
  cell_type: u32,
) -> io::Result<()> {
  let nverts_per_cell = skeleton.dim() + 1;
  writeln!(w, "      <Cells>")?;
  writeln!(
    w,
    "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">"
  )?;
  for simplex in skeleton.handle_iter() {
    write!(w, "          ")?;
    for &vertex in simplex.vertices.iter() {
      write!(w, "{vertex} ")?;
    }
    writeln!(w)?;
  }
  writeln!(w, "        </DataArray>")?;
  writeln!(
    w,
    "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">"
  )?;
  write!(w, "          ")?;
  for offset in (1..=skeleton.len()).map(|cell_index| cell_index * nverts_per_cell) {
    write!(w, "{offset} ")?;
  }
  writeln!(w)?;
  writeln!(w, "        </DataArray>")?;
  writeln!(
    w,
    "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
  )?;
  write!(w, "          ")?;
  for _ in 0..skeleton.len() {
    write!(w, "{cell_type} ")?;
  }
  writeln!(w)?;
  writeln!(w, "        </DataArray>")?;
  writeln!(w, "      </Cells>")?;
  Ok(())
}

fn write_vtu_polyline_cells(mut w: impl Write, paths: &[&[usize]]) -> io::Result<()> {
  let cell_type = vtk_cell_type(1)
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Unsupported line cell type"))?;
  writeln!(w, "      <Cells>")?;
  writeln!(
    w,
    "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">"
  )?;
  let mut point_offset = 0usize;
  for path_vertices in paths {
    write!(w, "          ")?;
    for point_index in 0..path_vertices.len() {
      write!(w, "{} ", point_offset + point_index)?;
    }
    writeln!(w)?;
    point_offset += path_vertices.len();
  }
  writeln!(w, "        </DataArray>")?;
  writeln!(
    w,
    "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">"
  )?;
  write!(w, "          ")?;
  let mut offset = 0usize;
  for path_vertices in paths {
    offset += path_vertices.len();
    write!(w, "{offset} ")?;
  }
  writeln!(w)?;
  writeln!(w, "        </DataArray>")?;
  writeln!(
    w,
    "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
  )?;
  write!(w, "          ")?;
  for _ in paths {
    write!(w, "{cell_type} ")?;
  }
  writeln!(w)?;
  writeln!(w, "        </DataArray>")?;
  writeln!(w, "      </Cells>")?;
  Ok(())
}

fn xml_escape(value: &str) -> String {
  value
    .replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
    .replace('\'', "&apos;")
}

fn write_vtk_points(mut w: impl Write, coords: &MeshCoords) -> io::Result<()> {
  writeln!(w, "POINTS {} double", coords.nvertices())?;
  for coord in coords.coord_iter() {
    let x = coord[0];
    let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
    let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
    writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
  }
  Ok(())
}

fn write_vtk_scalar(
  mut w: impl Write,
  name: &str,
  values: impl IntoIterator<Item = f64>,
) -> io::Result<()> {
  writeln!(w, "SCALARS {name} double 1")?;
  writeln!(w, "LOOKUP_TABLE default")?;
  for value in values {
    writeln!(w, "{value:.12e}")?;
  }
  Ok(())
}

fn write_skeleton_cells(mut w: impl Write, skeleton: &SkeletonHandle) -> io::Result<()> {
  let nverts_per_cell = skeleton.dim() + 1;
  for simplex in skeleton.handle_iter() {
    write!(w, "{nverts_per_cell}")?;
    for &vertex in simplex.vertices.iter() {
      write!(w, " {}", vertex)?;
    }
    writeln!(w)?;
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use common::linalg::nalgebra::Vector;
  use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::mesh::standard_coord_complex,
  };

  #[test]
  fn write_1form_vector_proxy_vtk_smoke() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let edges = topology.skeleton(1);
    let cochain = Cochain::new(1, Vector::from_element(edges.len(), 1.0));

    let path = std::env::temp_dir().join("proxy_1form.vtk");
    write_1form_vector_proxy_vtk(&path, &coords, &topology, &cochain, "proxy").unwrap();
    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_1form_vector_proxy_vtk_fields_writes_vectors_and_scalars() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let edges = topology.skeleton(1);
    let vector = Cochain::new(1, Vector::from_element(edges.len(), 1.0));
    let weight = Cochain::new(1, Vector::from_element(edges.len(), 0.25));

    let path = std::env::temp_dir().join("proxy_1form_fields.vtk");
    write_1form_vector_proxy_vtk_fields(
      &path,
      &coords,
      &topology,
      "proxy",
      &vector,
      &[("weight", &weight)],
    )
    .unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("VECTORS proxy double"));
    assert!(content.contains("SCALARS weight double 1"));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_1form_vector_field_vtk_embedded_surface_smoke() {
    let (topology, coords_2d) = standard_coord_complex(2);
    let coords = coords_2d.embed_euclidean(3);
    let edges = topology.skeleton(1);
    let cochain = Cochain::new(1, Vector::from_element(edges.len(), 1.0));

    let path = std::env::temp_dir().join("embedded_1form_vector_field.vtk");
    write_1form_vector_field_vtk(&path, &coords, &topology, &cochain, "embedded").unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let vectors_idx = lines
      .iter()
      .position(|line| line.trim() == "VECTORS embedded double")
      .unwrap();
    let vector_line = lines[vectors_idx + 1];
    let comps: Vec<f64> = vector_line
      .split_whitespace()
      .map(|value| value.parse::<f64>().unwrap())
      .collect();
    assert_eq!(comps.len(), 3);
    assert!(comps.iter().all(|value| value.is_finite()));
    assert!(comps[2].abs() < 1e-10);

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn sample_1form_cell_vectors_embedded_surface_smoke() {
    let (topology, coords_2d) = standard_coord_complex(2);
    let coords = coords_2d.embed_euclidean(3);
    let edges = topology.skeleton(1);
    let cochain = Cochain::new(1, Vector::from_element(edges.len(), 1.0));

    let vectors = sample_1form_cell_vectors(&coords, &topology, &cochain).unwrap();
    assert_eq!(vectors.len(), topology.cells().len());
    assert!(vectors
      .iter()
      .flat_map(|vector| vector.iter())
      .all(|value| value.is_finite()));
  }

  #[test]
  fn sample_2form_cell_vectors_3d_smoke() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let faces = topology.skeleton(2);
    let cochain = Cochain::new(2, Vector::from_element(faces.len(), 1.0));

    let vectors = sample_2form_cell_vectors(&coords, &topology, &cochain).unwrap();
    assert_eq!(vectors.len(), topology.cells().len());
    assert!(vectors
      .iter()
      .flat_map(|vector| vector.iter())
      .all(|value| value.is_finite()));
  }

  #[test]
  fn write_top_cell_vtk_fields_writes_multiple_vectors_and_scalars() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let cell_count = topology.cells().len();
    let mean_vectors = vec![[1.0, 0.0, 0.5]; cell_count];
    let auxiliary_vectors = vec![[0.25, 0.5, 0.75]; cell_count];
    let magnitude = vec![1.5; cell_count];

    let path = std::env::temp_dir().join("top_cell_fields.vtk");
    write_top_cell_vtk_fields(
      &path,
      &coords,
      &topology,
      &[
        ("mean_vector", mean_vectors.as_slice()),
        ("auxiliary_vector", auxiliary_vectors.as_slice()),
      ],
      &[("magnitude", magnitude.as_slice())],
    )
    .unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("VECTORS mean_vector double"));
    assert!(content.contains("VECTORS auxiliary_vector double"));
    assert!(content.contains("SCALARS magnitude double 1"));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_top_cell_vtk_fields_writes_scalar_only_fields() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let cell_count = topology.cells().len();
    let scalar = vec![0.5; cell_count];

    let path = std::env::temp_dir().join("top_cell_scalar_only_fields.vtk");
    write_top_cell_vtk_fields(
      &path,
      &coords,
      &topology,
      &[],
      &[("scalar", scalar.as_slice())],
    )
    .unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("CELL_DATA"));
    assert!(content.contains("SCALARS scalar double 1"));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_cochain_vtk_fields_writes_multiple_scalars() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let edge_count = topology.skeleton(1).len();
    let first = Cochain::new(1, Vector::from_element(edge_count, 1.0));
    let second = Cochain::new(1, Vector::from_element(edge_count, 2.0));

    let path = std::env::temp_dir().join("multi_cochain_fields.vtk");
    write_cochain_vtk_fields(
      &path,
      &coords,
      &topology,
      1,
      &[("first", &first), ("second", &second)],
    )
    .unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("CELL_DATA"));
    assert!(content.contains("SCALARS first double 1"));
    assert!(content.contains("SCALARS second double 1"));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_polyline_vtk_fields_writes_cell_scalars() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (_topology, coords) = mesh.compute_coord_complex();
    let path_a = vec![0, 1];
    let path_b = vec![2, 3];
    let paths = vec![path_a.as_slice(), path_b.as_slice()];
    let cycle_index = vec![0.0, 1.0];

    let path = std::env::temp_dir().join("polyline_fields.vtk");
    write_polyline_vtk_fields(
      &path,
      "polyline fields",
      &coords,
      &paths,
      &[("cycle_index", cycle_index.as_slice())],
    )
    .unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("DATASET POLYDATA"));
    assert!(content.contains("LINES 2 6"));
    assert!(content.contains("SCALARS cycle_index double 1"));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_cochain_vtk_preserves_tiny_values_in_scientific_notation() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let edges = topology.skeleton(1);
    let cochain = Cochain::new(1, Vector::from_element(edges.len(), 1.0e-15));

    let path = std::env::temp_dir().join("tiny_cochain.vtk");
    write_cochain_vtk(&path, &coords, &topology, &cochain, "tiny").unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("1.000000000000e-15"));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_0cochain_vtu_fields_writes_point_data() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let field = Cochain::new(0, Vector::from_element(coords.nvertices(), 2.0));

    let path = std::env::temp_dir().join("point_cochain_fields.vtu");
    write_cochain_vtu_fields(&path, &coords, &topology, 0, &[("potential", &field)]).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("<VTKFile type=\"UnstructuredGrid\""));
    assert!(content.contains("NumberOfPoints=\"4\""));
    assert!(content.contains("NumberOfCells=\"2\""));
    assert!(content.contains("<PointData Scalars=\"potential\">"));
    assert!(content.contains("Name=\"potential\" NumberOfComponents=\"1\""));
    assert!(content.contains("<CellData/>"));
    assert!(content.contains("Name=\"types\" format=\"ascii\">\n          5 5 "));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_1cochain_vtu_fields_writes_cell_data_and_tiny_values() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let edge_count = topology.skeleton(1).len();
    let tiny = Cochain::new(1, Vector::from_element(edge_count, 1.0e-15));
    let other = Cochain::new(1, Vector::from_element(edge_count, 3.0));

    let path = std::env::temp_dir().join("edge_cochain_fields.vtu");
    write_cochain_vtu_fields(
      &path,
      &coords,
      &topology,
      1,
      &[("tiny", &tiny), ("other", &other)],
    )
    .unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("NumberOfCells=\"5\""));
    assert!(content.contains("<PointData/>"));
    assert!(content.contains("<CellData Scalars=\"tiny\">"));
    assert!(content.contains("Name=\"tiny\" NumberOfComponents=\"1\""));
    assert!(content.contains("Name=\"other\" NumberOfComponents=\"1\""));
    assert!(content.contains("1.000000000000e-15"));
    assert!(content.contains("Name=\"types\" format=\"ascii\">\n          3 3 3 3 3 "));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_top_cell_vtu_fields_writes_vectors_and_scalars() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let cell_count = topology.cells().len();
    let vectors = vec![[1.0, 0.0, 0.5]; cell_count];
    let scalar = vec![0.25; cell_count];

    let path = std::env::temp_dir().join("top_cell_fields.vtu");
    write_top_cell_vtu_fields(
      &path,
      &coords,
      &topology,
      &[("mean_vector", vectors.as_slice())],
      &[("magnitude", scalar.as_slice())],
    )
    .unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("<CellData Scalars=\"magnitude\" Vectors=\"mean_vector\">"));
    assert!(content.contains("Name=\"mean_vector\" NumberOfComponents=\"3\""));
    assert!(content.contains("Name=\"magnitude\" NumberOfComponents=\"1\""));
    assert!(content.contains("1.000000000000e0 0.000000000000e0 5.000000000000e-1"));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_1form_vector_proxy_vtu_fields_writes_vectors_and_scalars() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let edges = topology.skeleton(1);
    let vector = Cochain::new(1, Vector::from_element(edges.len(), 1.0));
    let weight = Cochain::new(1, Vector::from_element(edges.len(), 0.25));

    let path = std::env::temp_dir().join("proxy_1form_fields.vtu");
    write_1form_vector_proxy_vtu_fields(
      &path,
      &coords,
      &topology,
      "proxy",
      &vector,
      &[("weight", &weight)],
    )
    .unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("NumberOfCells=\"5\""));
    assert!(content.contains("<CellData Scalars=\"weight\" Vectors=\"proxy\">"));
    assert!(content.contains("Name=\"proxy\" NumberOfComponents=\"3\""));
    assert!(content.contains("Name=\"weight\" NumberOfComponents=\"1\""));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_1form_vector_field_vtu_writes_top_cell_vectors() {
    let (topology, coords_2d) = standard_coord_complex(2);
    let coords = coords_2d.embed_euclidean(3);
    let edges = topology.skeleton(1);
    let cochain = Cochain::new(1, Vector::from_element(edges.len(), 1.0));

    let path = std::env::temp_dir().join("embedded_1form_vector_field.vtu");
    write_1form_vector_field_vtu(&path, &coords, &topology, &cochain, "embedded").unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("<CellData Vectors=\"embedded\">"));
    assert!(content.contains("Name=\"embedded\" NumberOfComponents=\"3\""));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_2form_vector_field_vtu_writes_top_cell_vectors() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let faces = topology.skeleton(2);
    let cochain = Cochain::new(2, Vector::from_element(faces.len(), 1.0));

    let path = std::env::temp_dir().join("two_form_vector_field.vtu");
    write_2form_vector_field_vtu(&path, &coords, &topology, &cochain, "flux").unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("<CellData Vectors=\"flux\">"));
    assert!(content.contains("Name=\"flux\" NumberOfComponents=\"3\""));
    assert!(content.contains("Name=\"types\" format=\"ascii\">"));
    assert!(content.contains("10 "));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn write_polyline_vtu_fields_writes_line_cells_and_cell_scalars() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (_topology, coords) = mesh.compute_coord_complex();
    let path_a = vec![0, 1, 3];
    let path_b = vec![2, 3];
    let paths = vec![path_a.as_slice(), path_b.as_slice()];
    let cycle_index = vec![0.0, 1.0];

    let path = std::env::temp_dir().join("polyline_fields.vtu");
    write_polyline_vtu_fields(
      &path,
      "polyline fields",
      &coords,
      &paths,
      &[("cycle_index", cycle_index.as_slice())],
    )
    .unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("NumberOfPoints=\"5\""));
    assert!(content.contains("NumberOfCells=\"2\""));
    assert!(content.contains("<CellData Scalars=\"cycle_index\">"));
    assert!(content.contains("Name=\"connectivity\" format=\"ascii\">\n          0 1 2 "));
    assert!(content.contains("Name=\"offsets\" format=\"ascii\">\n          3 5 "));
    assert!(content.contains("Name=\"types\" format=\"ascii\">\n          3 3 "));

    std::fs::remove_file(path).ok();
  }

  #[test]
  fn cochain_vtk_and_vtu_report_matching_counts() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let edges = topology.skeleton(1);
    let cochain = Cochain::new(1, Vector::from_element(edges.len(), 1.0));

    let vtk_path = std::env::temp_dir().join("cochain_count_parity.vtk");
    let vtu_path = std::env::temp_dir().join("cochain_count_parity.vtu");
    write_cochain_vtk(&vtk_path, &coords, &topology, &cochain, "field").unwrap();
    write_cochain_vtu(&vtu_path, &coords, &topology, &cochain, "field").unwrap();

    let vtk = std::fs::read_to_string(&vtk_path).unwrap();
    let vtu = std::fs::read_to_string(&vtu_path).unwrap();
    assert!(vtk.contains("POINTS 4 double"));
    assert!(vtk.contains("CELLS 5 15"));
    assert!(vtu.contains("NumberOfPoints=\"4\""));
    assert!(vtu.contains("NumberOfCells=\"5\""));

    std::fs::remove_file(vtk_path).ok();
    std::fs::remove_file(vtu_path).ok();
  }
}
