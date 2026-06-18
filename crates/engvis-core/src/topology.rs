use std::collections::HashMap;
use crate::mesh::Mesh;

/// Result of a topological analysis of a triangle mesh.
#[derive(Debug, Clone)]
pub struct MeshTopology {
    /// Number of vertices (all entries in the vertex buffer).
    pub vertices: usize,
    /// Number of unique edges.
    pub edges: usize,
    /// Number of triangle faces.
    pub faces: usize,
    /// Euler characteristic  χ = V − E + F.
    ///
    /// | χ | Surface (closed, orientable) |
    /// |---|------------------------------|
    /// | 2 | Sphere (genus 0)           |
    /// | 0 | Torus  (genus 1)           |
    /// |−2 | Double torus (genus 2)     |
    pub euler: i64,
    /// Boundary edges — shared by exactly one triangle.
    /// Non-zero count means the mesh has holes / open boundaries.
    pub boundary_edges: usize,
    /// Non-manifold edges — shared by three or more triangles.
    pub non_manifold_edges: usize,
    /// Connected components (via face adjacency across shared edges).
    pub connected_components: usize,
    /// Whether every edge is shared by exactly two triangles
    /// and there is a single connected component.
    pub is_watertight: bool,
}

impl MeshTopology {
    /// Pretty-print the topology to stderr.
    pub fn print(&self) {
        eprintln!(
            "  topology: V={} E={} F={}  χ={}  boundary_edges={}  non_manifold={}  components={}  watertight={}",
            self.vertices, self.edges, self.faces, self.euler,
            self.boundary_edges, self.non_manifold_edges,
            self.connected_components, self.is_watertight,
        );
    }
}

/// Compute the topology of a triangle mesh.
///
/// Uses a half-edge–style adjacency map (HashMap on sorted index pairs)
/// and a union-find to count connected components.
pub fn compute_topology(mesh: &Mesh) -> MeshTopology {
    let v = mesh.vertices.len();
    let f = mesh.indices.len() / 3;

    // edge → list of face indices sharing this edge
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::with_capacity(f * 3);
    for (fi, tri) in mesh.indices.chunks_exact(3).enumerate() {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        for &(i0, i1) in &[(a, b), (b, c), (c, a)] {
            let key = if i0 <= i1 { (i0, i1) } else { (i1, i0) };
            edge_faces.entry(key).or_default().push(fi);
        }
    }

    let e = edge_faces.len();

    // Union-Find for connected components
    let mut parent: Vec<usize> = (0..f).collect();
    let mut rank: Vec<u32> = vec![0; f];

    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut Vec<usize>, rank: &mut Vec<u32>, mut a: usize, mut b: usize) {
        a = find(parent, a);
        b = find(parent, b);
        if a == b { return; }
        match rank[a].cmp(&rank[b]) {
            std::cmp::Ordering::Less => { parent[a] = b; }
            std::cmp::Ordering::Greater => { parent[b] = a; }
            std::cmp::Ordering::Equal => { parent[b] = a; rank[a] += 1; }
        }
    }

    let mut boundary_edges = 0usize;
    let mut non_manifold_edges = 0usize;

    for faces in edge_faces.values() {
        match faces.len() {
            1 => boundary_edges += 1,
            2 => union(&mut parent, &mut rank, faces[0], faces[1]),
            _ => {
                non_manifold_edges += 1;
                // still union all pairs for component count
                for i in 1..faces.len() {
                    union(&mut parent, &mut rank, faces[0], faces[i]);
                }
            }
        }
    }

    let connected_components = (0..f).filter(|&i| find(&mut parent, i) == i).count();

    let euler = v as i64 - e as i64 + f as i64;

    MeshTopology {
        vertices: v,
        edges: e,
        faces: f,
        euler,
        boundary_edges,
        non_manifold_edges,
        connected_components,
        is_watertight: boundary_edges == 0 && non_manifold_edges == 0 && connected_components == 1,
    }
}
