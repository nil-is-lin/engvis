use crate::aabb::Aabb;

/// GPU-ready vertex for PBR rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

impl MeshVertex {
    pub const SIZE: usize = std::mem::size_of::<Self>();
    pub const ATTRIB_COUNT: usize = 4;
}

/// A sub-mesh references a material index
#[derive(Debug, Clone)]
pub struct SubMesh {
    pub material_index: usize,
    pub index_offset: u32,
    pub index_count: u32,
}

/// A complete mesh with vertices and indices
#[derive(Debug, Clone)]
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub sub_meshes: Vec<SubMesh>,
    pub aabb: Aabb,
}

/// Merge duplicate vertices that share the same position (within `eps`).
///
/// Adaptive dual-contouring may emit the same geometric point multiple
/// times with different indices, creating T-junctions and visual gaps at
/// octree resolution boundaries.  This function remaps the index buffer
/// so that co-located vertices collapse to a single index.
pub fn dedup_vertices(positions: &mut Vec<[f32; 3]>, indices: &mut Vec<u32>, eps: f32) {
    use std::collections::HashMap;

    let inv = 1.0 / eps;
    // Spatial hash: quantise each coordinate to an integer grid
    let hash_pos = |p: [f32; 3]| -> (i64, i64, i64) {
        (
            (p[0] * inv).round() as i64,
            (p[1] * inv).round() as i64,
            (p[2] * inv).round() as i64,
        )
    };

    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut remap = vec![0u32; positions.len()];

    let mut new_positions: Vec<[f32; 3]> = Vec::new();

    for (i, &pos) in positions.iter().enumerate() {
        let key = hash_pos(pos);
        let mut found = None;

        // Check the 3×3×3 neighbourhood for an existing vertex
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nk = (key.0 + dx, key.1 + dy, key.2 + dz);
                    if let Some(bucket) = grid.get(&nk) {
                        for &j in bucket {
                            let op = new_positions[j];
                            let d = (pos[0] - op[0]).powi(2)
                                + (pos[1] - op[1]).powi(2)
                                + (pos[2] - op[2]).powi(2);
                            if d.sqrt() < eps {
                                found = Some(j as u32);
                                break;
                            }
                        }
                    }
                    if found.is_some() { break; }
                }
                if found.is_some() { break; }
            }
            if found.is_some() { break; }
        }

        match found {
            Some(j) => { remap[i] = j; }
            None => {
                let j = new_positions.len() as u32;
                remap[i] = j;
                new_positions.push(pos);
                grid.entry(key).or_default().push(j as usize);
            }
        }
    }

    let old_len = positions.len();
    *positions = new_positions;
    for idx in indices.iter_mut() {
        *idx = remap[*idx as usize];
    }

    // Remove degenerate triangles (any two indices identical → zero area).
    // dedup can collapse two vertices of the same triangle into one,
    // producing sliver triangles that pass topology checks but render
    // as visual artifacts (lines/spikes).
    let old_tris = indices.len() / 3;
    let mut write = 0;
    for read in (0..indices.len()).step_by(3) {
        let a = indices[read];
        let b = indices[read + 1];
        let c = indices[read + 2];
        if a != b && b != c && a != c {
            indices[write] = a;
            indices[write + 1] = b;
            indices[write + 2] = c;
            write += 3;
        }
    }
    indices.truncate(write);
    let removed_tris = old_tris - indices.len() / 3;

    if old_len != positions.len() || removed_tris > 0 {
        eprintln!(
            "  dedup: {} → {} verts ({} removed), {} → {} tris ({} degenerate removed)",
            old_len, positions.len(), old_len - positions.len(),
            old_tris, indices.len() / 3, removed_tris
        );
    }
}

/// Fix winding-order inconsistencies in a triangle index buffer.
///
/// Three passes:
/// 1. **BFS local consistency** — propagates consistent winding across
///    adjacent faces so every shared edge has opposite orientations in
///    its two triangles.  Fixes scattered per-triangle flips from
///    dual-contouring that cause back-face culling holes.
/// 2. **Union-Find component detection** — identifies connected components
///    via face adjacency across shared edges (robust across adaptive
///    octree resolution boundaries).
/// 3. **Per-component signed volume** — flips each component whose volume
///    contribution is negative (inward-facing normals).
///
/// Call this on the raw `(positions, indices)` pair before constructing
/// a [`Mesh`], or after any operation that may introduce winding defects.
pub fn fix_winding(positions: &[[f32; 3]], indices: &mut [u32]) {
    use std::collections::{HashMap, VecDeque};

    let n_faces = indices.len() / 3;
    if n_faces == 0 { return; }

    fn tri_edges(idx: &[u32], fi: usize) -> [(u32, u32); 3] {
        let t = &idx[fi * 3..fi * 3 + 3];
        [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
    }

    // ── Build edge → face adjacency ──────────────────────────
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> =
        HashMap::with_capacity(n_faces * 3);
    for (fi, tri) in indices.chunks_exact(3).enumerate() {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = (a.min(b), a.max(b));
            edge_faces.entry(key).or_default().push(fi);
        }
    }

    // ── 1. BFS local winding consistency ────────────────────
    let mut visited = vec![false; n_faces];
    let mut bfs_flipped = 0usize;
    for seed in 0..n_faces {
        if visited[seed] { continue; }
        visited[seed] = true;
        let mut queue = VecDeque::from([seed]);
        while let Some(fi) = queue.pop_front() {
            for &(src, dst) in &tri_edges(indices, fi) {
                let key = (src.min(dst), src.max(dst));
                if let Some(neighbors) = edge_faces.get(&key) {
                    for &ni in neighbors {
                        if ni == fi || visited[ni] { continue; }
                        let ni_edges = tri_edges(indices, ni);
                        if ni_edges.iter().any(|e| *e == (src, dst)) {
                            indices.swap(ni * 3 + 1, ni * 3 + 2);
                            bfs_flipped += 1;
                        }
                        visited[ni] = true;
                        queue.push_back(ni);
                    }
                }
            }
        }
    }

    // ── 2. Union-Find component detection ───────────────────
    let mut parent: Vec<usize> = (0..n_faces).collect();
    let mut rank = vec![0u32; n_faces];
    fn find(par: &mut [usize], mut x: usize) -> usize {
        while par[x] != x { par[x] = par[par[x]]; x = par[x]; }
        x
    }
    fn union(par: &mut [usize], rnk: &mut [u32], mut a: usize, mut b: usize) {
        a = find(par, a); b = find(par, b);
        if a == b { return; }
        match rnk[a].cmp(&rnk[b]) {
            std::cmp::Ordering::Less    => { par[a] = b; }
            std::cmp::Ordering::Greater => { par[b] = a; }
            std::cmp::Ordering::Equal   => { par[b] = a; rnk[a] += 1; }
        }
    }
    for faces in edge_faces.values() {
        for i in 1..faces.len() {
            union(&mut parent, &mut rank, faces[0], faces[i]);
        }
    }

    // ── 3. Per-component signed volume → flip inward comps ──
    let mut comp_volume: HashMap<usize, f64> = HashMap::new();
    for (fi, tri) in indices.chunks_exact(3).enumerate() {
        let root = find(&mut parent, fi);
        let p0 = glam::DVec3::from(glam::Vec3::from(positions[tri[0] as usize]));
        let p1 = glam::DVec3::from(glam::Vec3::from(positions[tri[1] as usize]));
        let p2 = glam::DVec3::from(glam::Vec3::from(positions[tri[2] as usize]));
        *comp_volume.entry(root).or_default() += p0.dot(p1.cross(p2));
    }

    let mut comps_flipped = 0usize;
    let n_comps = comp_volume.len();
    for (&root, &vol) in &comp_volume {
        if vol < 0.0 {
            for fi in 0..n_faces {
                if find(&mut parent, fi) == root {
                    indices.swap(fi * 3 + 1, fi * 3 + 2);
                }
            }
            comps_flipped += 1;
        }
    }
    eprintln!(
        "  winding: bfs_flipped={} components={} comps_flipped={}",
        bfs_flipped, n_comps, comps_flipped
    );
}

impl Mesh {
    /// Extract edge indices from triangle indices for edge (line) rendering.
    /// For each triangle (i0, i1, i2), generates 3 edges: (i0, i1), (i1, i2), (i2, i0).
    pub fn extract_edge_indices(&self) -> Vec<u32> {
        let mut edge_indices = Vec::with_capacity(self.indices.len() * 2);
        for tri in self.indices.chunks(3) {
            if tri.len() == 3 {
                edge_indices.extend_from_slice(&[tri[0], tri[1], tri[1], tri[2], tri[2], tri[0]]);
            }
        }
        edge_indices
    }

    /// Fix winding-order inconsistencies and recompute smooth normals.
    ///
    /// Delegates to the standalone [`fix_winding`] function on this mesh's
    /// positions and indices, then recalculates area-weighted smooth normals.
    /// Useful for meshes imported from GLTF or other sources that may have
    /// inconsistent triangle winding.
    pub fn fix_winding(&mut self) {
        let positions: Vec<[f32; 3]> = self.vertices.iter().map(|v| v.position).collect();
        fix_winding(&positions, &mut self.indices);
        // Recompute smooth normals with the corrected winding
        self.recompute_smooth_normals(&positions);
    }

    /// Recompute area-weighted smooth normals from current indices.
    fn recompute_smooth_normals(&mut self, positions: &[[f32; 3]]) {
        let n_verts = positions.len();
        let mut normals = vec![[0.0_f32; 3]; n_verts];
        for tri in self.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            let p0 = glam::Vec3::from(positions[i0]);
            let p1 = glam::Vec3::from(positions[i1]);
            let p2 = glam::Vec3::from(positions[i2]);
            let n = (p1 - p0).cross(p2 - p0);
            for &i in &[i0, i1, i2] {
                normals[i][0] += n.x;
                normals[i][1] += n.y;
                normals[i][2] += n.z;
            }
        }
        for (vert, norm) in self.vertices.iter_mut().zip(normals.iter()) {
            let len = (norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]).sqrt();
            vert.normal = if len > 1e-10 {
                let inv = 1.0 / len;
                [norm[0] * inv, norm[1] * inv, norm[2] * inv]
            } else {
                [0.0, 1.0, 0.0]
            };
        }
    }

    /// Create a smooth-shaded mesh from flat position and index arrays.
    ///
    /// Pipeline: vertex dedup → winding fix (BFS + per-component signed
    /// volume) → area-weighted smooth normals.
    pub fn from_triangles(
        name: impl Into<String>,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Self {
        Self::from_triangles_impl(name, positions, indices, true)
    }

    /// Create a smooth-shaded mesh from flat position and index arrays
    /// **without** running the global winding-fix passes.
    ///
    /// Used by callers that have already established consistent face
    /// winding (e.g. open implicit-surface patches whose boundary fan
    /// triangles are constructed to match adjacent DC triangles).  The
    /// BFS / signed-volume / geometric-flip passes assume a closed
    /// manifold and can incorrectly flip faces on open patches, so we
    /// skip them here.
    ///
    /// Pipeline: vertex dedup → area-weighted smooth normals.
    pub fn from_triangles_open(
        name: impl Into<String>,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Self {
        Self::from_triangles_impl(name, positions, indices, false)
    }

    fn from_triangles_impl(
        name: impl Into<String>,
        positions: &[[f32; 3]],
        indices: &[u32],
        run_winding_fix: bool,
    ) -> Self {
        let mut positions = positions.to_vec();
        let mut indices = indices.to_vec();

        // Dedup tolerance scaled to mesh size: 1e-6 of the AABB diagonal.
        // A fixed eps (e.g. 1e-4) over-merges vertices on high-resolution
        // meshes, creating non-manifold edges and winding inconsistencies
        // that produce holes under back-face culling.
        let diag = {
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for p in &positions {
                for i in 0..3 {
                    min[i] = min[i].min(p[i]);
                    max[i] = max[i].max(p[i]);
                }
            }
            let d = [
                max[0] - min[0],
                max[1] - min[1],
                max[2] - min[2],
            ];
            (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]).sqrt()
        };
        let eps = diag * 1e-6;

        // Merge duplicate vertices at octree resolution boundaries
        dedup_vertices(&mut positions, &mut indices, eps);

        // Fix winding inconsistencies
        if run_winding_fix {
            fix_winding(&positions, &mut indices);
        }

        let n_verts = positions.len();

        // ── Fix geometric flips from dedup ────────────────────
        // dedup_vertices can move merged vertices, causing individual
        // faces' geometric normals to flip while their topological
        // winding stays consistent with neighbours.  Such faces are
        // culled by back-face culling, producing holes.  Detect them
        // by comparing each face normal against the average of its
        // vertices' neighbours' normals, and flip the winding of
        // faces that disagree.
        if run_winding_fix {
            // First pass: compute per-face normals.
            let nf = indices.len() / 3;
            let mut face_normals: Vec<glam::Vec3> = Vec::with_capacity(nf);
            for tri in indices.chunks_exact(3) {
                let p0 = glam::Vec3::from(positions[tri[0] as usize]);
                let p1 = glam::Vec3::from(positions[tri[1] as usize]);
                let p2 = glam::Vec3::from(positions[tri[2] as usize]);
                face_normals.push((p1 - p0).cross(p2 - p0));
            }

            // Build vertex → face adjacency.
            let mut vert_faces: Vec<Vec<usize>> = vec![Vec::new(); n_verts];
            for (fi, tri) in indices.chunks_exact(3).enumerate() {
                for &idx in tri {
                    vert_faces[idx as usize].push(fi);
                }
            }

            // For each face, compare its normal against the average of
            // adjacent face normals.  If anti-parallel, flip winding.
            let mut to_flip: Vec<usize> = Vec::new();
            for (fi, tri) in indices.chunks_exact(3).enumerate() {
                let n = face_normals[fi];
                if n.length_squared() < 1e-16 { continue; }
                let n = n.normalize();

                // Collect neighbour face normals.
                let mut ref_sum = glam::Vec3::ZERO;
                for &idx in tri {
                    for &nf in &vert_faces[idx as usize] {
                        if nf != fi {
                            ref_sum += face_normals[nf];
                        }
                    }
                }
                if ref_sum.length_squared() < 1e-12 { continue; }

                if n.dot(ref_sum) < 0.0 {
                    to_flip.push(fi);
                }
            }
            for &fi in &to_flip {
                indices.swap(fi * 3 + 1, fi * 3 + 2);
            }
        }

        // ── Smooth normals (area-weighted) ─────────────────────
        let mut smooth_normals = vec![[0.0_f32; 3]; n_verts];
        for tri in indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            let p0 = glam::Vec3::from(positions[i0]);
            let p1 = glam::Vec3::from(positions[i1]);
            let p2 = glam::Vec3::from(positions[i2]);
            let n = (p1 - p0).cross(p2 - p0);
            for &i in &[i0, i1, i2] {
                smooth_normals[i][0] += n.x;
                smooth_normals[i][1] += n.y;
                smooth_normals[i][2] += n.z;
            }
        }
        for n in &mut smooth_normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-10 {
                let inv = 1.0 / len;
                n[0] *= inv; n[1] *= inv; n[2] *= inv;
            } else { *n = [0.0, 1.0, 0.0]; }
        }

        let vertices: Vec<MeshVertex> = positions.iter().zip(smooth_normals.iter())
            .map(|(pos, norm)| MeshVertex {
                position: *pos, normal: *norm,
                uv: [0.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0],
            }).collect();

        let mut aabb = crate::Aabb::empty();
        for p in &positions { aabb.expand(glam::Vec3::from(*p)); }

        let index_count = indices.len() as u32;
        Mesh {
            name: name.into(),
            vertices,
            indices,
            sub_meshes: vec![SubMesh { material_index: 0, index_offset: 0, index_count }],
            aabb,
        }
    }
}

/// Create a unit cube mesh with normals, UVs, and tangents
pub fn create_cube_mesh() -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut aabb = Aabb::empty();

    // Each face: 4 vertices with position, normal, uv, tangent
    let faces: [([f32; 3], [f32; 3], [f32; 4]); 6] = [
        // +Y (top)
        ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
        // -Y (bottom)
        ([0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]),
        // +X (right)
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0]),
        // -X (left)
        ([-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]),
        // +Z (front)
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
        // -Z (back)
        ([0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 1.0]),
    ];

    let face_vertices: [[([f32; 3], [f32; 2]); 4]; 6] = [
        // +Y top
        [
            ([-0.5, 0.5, -0.5], [0.0, 0.0]),
            ([0.5, 0.5, -0.5], [1.0, 0.0]),
            ([0.5, 0.5, 0.5], [1.0, 1.0]),
            ([-0.5, 0.5, 0.5], [0.0, 1.0]),
        ],
        // -Y bottom
        [
            ([-0.5, -0.5, 0.5], [0.0, 0.0]),
            ([0.5, -0.5, 0.5], [1.0, 0.0]),
            ([0.5, -0.5, -0.5], [1.0, 1.0]),
            ([-0.5, -0.5, -0.5], [0.0, 1.0]),
        ],
        // +X right
        [
            ([0.5, -0.5, -0.5], [0.0, 0.0]),
            ([0.5, -0.5, 0.5], [1.0, 0.0]),
            ([0.5, 0.5, 0.5], [1.0, 1.0]),
            ([0.5, 0.5, -0.5], [0.0, 1.0]),
        ],
        // -X left
        [
            ([-0.5, -0.5, 0.5], [0.0, 0.0]),
            ([-0.5, -0.5, -0.5], [1.0, 0.0]),
            ([-0.5, 0.5, -0.5], [1.0, 1.0]),
            ([-0.5, 0.5, 0.5], [0.0, 1.0]),
        ],
        // +Z front
        [
            ([0.5, -0.5, 0.5], [0.0, 0.0]),
            ([-0.5, -0.5, 0.5], [1.0, 0.0]),
            ([-0.5, 0.5, 0.5], [1.0, 1.0]),
            ([0.5, 0.5, 0.5], [0.0, 1.0]),
        ],
        // -Z back
        [
            ([-0.5, -0.5, -0.5], [0.0, 0.0]),
            ([0.5, -0.5, -0.5], [1.0, 0.0]),
            ([0.5, 0.5, -0.5], [1.0, 1.0]),
            ([-0.5, 0.5, -0.5], [0.0, 1.0]),
        ],
    ];

    for (face_idx, (normal, _, tangent)) in faces.iter().enumerate() {
        let base = vertices.len() as u32;
        for (pos, uv) in &face_vertices[face_idx] {
            vertices.push(MeshVertex {
                position: *pos,
                normal: *normal,
                uv: *uv,
                tangent: *tangent,
            });
            aabb.expand(glam::Vec3::from(*pos));
        }
        indices.extend_from_slice(&[
            base, base + 2, base + 1,
            base, base + 3, base + 2,
        ]);
    }

    let index_count = indices.len() as u32;
    Mesh {
        name: "Cube".to_string(),
        vertices,
        indices,
        sub_meshes: vec![SubMesh {
            material_index: 0,
            index_offset: 0,
            index_count,
        }],
        aabb,
    }
}
