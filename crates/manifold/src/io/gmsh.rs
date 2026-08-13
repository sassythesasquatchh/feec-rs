use std::{collections::HashMap, str::FromStr};

use common::linalg::nalgebra::Matrix;

use crate::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton},
};

pub fn gmsh2coord_complex(bytes: &[u8]) -> (Complex, MeshCoords) {
  let (cells, coords) = gmsh2coord_cells(bytes);
  let complex = Complex::from_cells(cells);
  (complex, coords)
}

/// Load Gmesh `.msh` file (version 4.1).
pub fn gmsh2coord_cells(bytes: &[u8]) -> (Skeleton, MeshCoords) {
  let msh = mshio::parse_msh_bytes(bytes).unwrap();
  let nodes = msh.data.nodes.as_ref().unwrap();

  let (mut mesh_vertices, tag_to_index) = dense_node_data(bytes, nodes);

  if mesh_vertices.iter().all(|coord| coord[2] == 0.0) {
    mesh_vertices
      .iter_mut()
      .for_each(|coord| *coord = na::dvector![coord[0], coord[1]])
  }

  let mesh_vertices = Matrix::from_columns(&mesh_vertices);
  let mesh_vertices = MeshCoords::new(mesh_vertices);

  let mut points = Vec::new();
  let mut edges = Vec::new();
  let mut trias = Vec::new();
  let mut quads = Vec::new();

  let elements = msh.data.elements.unwrap();
  for block in elements.element_blocks {
    type ElType = mshio::ElementType;
    let simplex_acc = match block.element_type {
      ElType::Pnt => &mut points,
      ElType::Lin2 => &mut edges,
      ElType::Tri3 => &mut trias,
      ElType::Tet4 => &mut quads,
      _ => {
        tracing::warn!("unsupported gmsh ElementType: {:?}", block.element_type);
        continue;
      }
    };
    for e in block.elements {
      let simplex: Vec<_> = e
        .nodes
        .iter()
        .map(|tag| {
          *tag_to_index
            .get(tag)
            .unwrap_or_else(|| panic!("missing node tag {tag} in Gmsh node map"))
        })
        .collect();
      let simplex = Simplex::from(simplex).sorted();
      simplex_acc.push(simplex);
    }
  }

  let skeleton = if !quads.is_empty() {
    quads
  } else if !trias.is_empty() {
    trias
  } else if !edges.is_empty() {
    edges
  } else {
    panic!("Failed to construct Triangulation from gmsh.");
  };

  (Skeleton::new(skeleton), mesh_vertices)
}

type DenseNodeData = (Vec<na::DVector<f64>>, HashMap<u64, usize>);

fn dense_node_data(bytes: &[u8], nodes: &mshio::Nodes<u64, i32, f64>) -> DenseNodeData {
  parse_ascii_dense_node_data(bytes).unwrap_or_else(|| {
    tracing::warn!(
      "falling back to mshio node ordering; this assumes the file-order node list matches the tag order"
    );
    dense_node_data_from_mshio(nodes)
  })
}

fn parse_ascii_dense_node_data(bytes: &[u8]) -> Option<DenseNodeData> {
  let text = std::str::from_utf8(bytes).ok()?;
  let (_, after_nodes) = text.split_once("$Nodes")?;
  let (nodes_section, _) = after_nodes.split_once("$EndNodes")?;
  let mut tokens = nodes_section.split_whitespace();

  let num_entity_blocks = parse_next::<usize>(&mut tokens)?;
  let num_nodes = parse_next::<usize>(&mut tokens)?;
  let _min_node_tag = parse_next::<u64>(&mut tokens)?;
  let _max_node_tag = parse_next::<u64>(&mut tokens)?;

  let mut tagged_vertices = Vec::with_capacity(num_nodes);
  for _ in 0..num_entity_blocks {
    let _entity_dim = parse_next::<i32>(&mut tokens)?;
    let _entity_tag = parse_next::<i32>(&mut tokens)?;
    let parametric = parse_next::<i32>(&mut tokens)?;
    let num_nodes_in_block = parse_next::<usize>(&mut tokens)?;
    if parametric != 0 {
      return None;
    }

    let mut tags = Vec::with_capacity(num_nodes_in_block);
    for _ in 0..num_nodes_in_block {
      tags.push(parse_next::<u64>(&mut tokens)?);
    }
    for tag in tags {
      let x = parse_next::<f64>(&mut tokens)?;
      let y = parse_next::<f64>(&mut tokens)?;
      let z = parse_next::<f64>(&mut tokens)?;
      tagged_vertices.push((tag, na::dvector![x, y, z]));
    }
  }

  if tagged_vertices.len() != num_nodes {
    return None;
  }

  tagged_vertices.sort_by_key(|(tag, _)| *tag);

  let mut mesh_vertices = Vec::with_capacity(num_nodes);
  let mut tag_to_index = HashMap::with_capacity(num_nodes);
  for (index, (tag, coord)) in tagged_vertices.into_iter().enumerate() {
    if tag_to_index.insert(tag, index).is_some() {
      return None;
    }
    mesh_vertices.push(coord);
  }

  Some((mesh_vertices, tag_to_index))
}

fn dense_node_data_from_mshio(
  nodes: &mshio::Nodes<u64, i32, f64>,
) -> (Vec<na::DVector<f64>>, HashMap<u64, usize>) {
  if nodes
    .node_blocks
    .iter()
    .all(|block| block.node_tags.is_none())
  {
    let mesh_vertices: Vec<_> = nodes
      .node_blocks
      .iter()
      .flat_map(|block| block.nodes.iter())
      .map(|node| na::dvector![node.x, node.y, node.z])
      .collect();

    let start_tag = nodes.min_node_tag;
    let tag_to_index = (0..mesh_vertices.len())
      .map(|index| (start_tag + index as u64, index))
      .collect();

    return (mesh_vertices, tag_to_index);
  }

  let mut tagged_vertices = Vec::with_capacity(nodes.num_nodes as usize);

  for block in &nodes.node_blocks {
    let node_tags = block
      .node_tags
      .as_ref()
      .expect("mixed Gmsh node-tag layouts are unsupported");
    let mut block_vertices: Vec<_> = node_tags
      .iter()
      .map(|(tag, index)| {
        let node = &block.nodes[*index];
        (*tag, na::dvector![node.x, node.y, node.z])
      })
      .collect();
    tagged_vertices.append(&mut block_vertices);
  }

  tagged_vertices.sort_by_key(|(tag, _)| *tag);
  let mut mesh_vertices = Vec::with_capacity(tagged_vertices.len());
  let mut tag_to_index = HashMap::with_capacity(tagged_vertices.len());
  for (index, (tag, coord)) in tagged_vertices.into_iter().enumerate() {
    tag_to_index.insert(tag, index);
    mesh_vertices.push(coord);
  }

  (mesh_vertices, tag_to_index)
}

fn parse_next<T: FromStr>(tokens: &mut std::str::SplitWhitespace<'_>) -> Option<T> {
  tokens.next()?.parse().ok()
}

#[cfg(test)]
mod tests {
  use super::gmsh2coord_cells;

  #[test]
  fn parses_contiguous_node_tags_that_are_shuffled_across_blocks() {
    let msh = br#"$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
2 4 1 4
2 1 0 2
1
4
0 0 0
1 1 0
2 2 0 2
2
3
1 0 0
0 1 0
$EndNodes
$Elements
1 2 1 2
2 1 2 2
1 1 2 3
2 1 3 4
$EndElements
"#;

    let (skeleton, coords) = gmsh2coord_cells(msh);
    let simplices: Vec<_> = skeleton.iter().cloned().collect();

    assert_eq!(coords.dim(), 2);
    assert_eq!(coords.nvertices(), 4);
    assert_eq!(simplices.len(), 2);
    assert_eq!(simplices[0].vertices, vec![0, 1, 2]);
    assert_eq!(simplices[1].vertices, vec![0, 2, 3]);
    assert_eq!(coords.coord(0)[0], 0.0);
    assert_eq!(coords.coord(0)[1], 0.0);
    assert_eq!(coords.coord(1)[0], 1.0);
    assert_eq!(coords.coord(1)[1], 0.0);
    assert_eq!(coords.coord(2)[0], 0.0);
    assert_eq!(coords.coord(2)[1], 1.0);
    assert_eq!(coords.coord(3)[0], 1.0);
    assert_eq!(coords.coord(3)[1], 1.0);
  }

  #[test]
  fn parses_sparse_node_tags_without_assuming_tag_minus_one_indexing() {
    let msh = br#"$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 4 10 40
2 1 0 4
10
40
20
30
0 0 0
1 1 0
1 0 0
0 1 0
$EndNodes
$Elements
1 2 1 2
2 1 2 2
1 10 20 30
2 10 30 40
$EndElements
"#;

    let (skeleton, coords) = gmsh2coord_cells(msh);
    let simplices: Vec<_> = skeleton.iter().cloned().collect();

    assert_eq!(coords.dim(), 2);
    assert_eq!(coords.nvertices(), 4);
    assert_eq!(simplices.len(), 2);
    assert_eq!(simplices[0].vertices, vec![0, 1, 2]);
    assert_eq!(simplices[1].vertices, vec![0, 2, 3]);
    assert_eq!(coords.coord(0)[0], 0.0);
    assert_eq!(coords.coord(0)[1], 0.0);
    assert_eq!(coords.coord(1)[0], 1.0);
    assert_eq!(coords.coord(1)[1], 0.0);
    assert_eq!(coords.coord(2)[0], 0.0);
    assert_eq!(coords.coord(2)[1], 1.0);
    assert_eq!(coords.coord(3)[0], 1.0);
    assert_eq!(coords.coord(3)[1], 1.0);
  }
}
