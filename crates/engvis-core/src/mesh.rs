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
    use rayon::prelude::*;

    let inv = 1.0 / eps;
    // Pack quantised (x, y, z) into a single u64 key for faster hashing.
    // Each coordinate is offset to non-negative and packed into 21 bits
    // (range 0..2_097_151), which is ample for MC grids up to ~1000³.
    let hash_pos = |p: [f32; 3]| -> u64 {
        let x = (p[0] * inv).round() as i64;
        let y = (p[1] * inv).round() as i64;
        let z = (p[2] * inv).round() as i64;
        // Offset to non-negative (add 1<<20 to handle ±524287 range).
        ((x + (1 << 20)) as u64)
            | (((y + (1 << 20)) as u64) << 21)
            | (((z + (1 << 20)) as u64) << 42)
    };

    let n = positions.len();
    // Sort-based dedup: build (key, orig_idx) pairs, sort by key,
    // then group consecutive same-key entries.  This avoids HashMap
    // overhead and is more cache-friendly for large meshes.
    let mut keyed: Vec<(u64, u32)> = (0..n)
        .into_par_iter()
        .map(|i| (hash_pos(positions[i]), i as u32))
        .collect();
    keyed.par_sort_unstable_by_key(|e| e.0);

    let mut remap = vec![0u32; n];
    let mut new_positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    // For each group of same-key vertices, check distance to the
    // group's first vertex.  If within eps, remap to it; otherwise
    // create a new vertex.  Groups are typically 1-2 entries for MC.
    let mut i = 0;
    while i < n {
        let key = keyed[i].0;
        let group_start = i;
        while i < n && keyed[i].0 == key { i += 1; }
        // First vertex in group → always new.
        let first_orig = keyed[group_start].1 as usize;
        let new_idx = new_positions.len() as u32;
        new_positions.push(positions[first_orig]);
        remap[first_orig] = new_idx;
        // Remaining vertices in group: check distance to first.
        for j in (group_start + 1)..i {
            let orig_idx = keyed[j].1 as usize;
            let pos = positions[orig_idx];
            let op = new_positions[new_idx as usize];
            let d2 = (pos[0] - op[0]).powi(2)
                + (pos[1] - op[1]).powi(2)
                + (pos[2] - op[2]).powi(2);
            if d2 < eps * eps {
                remap[orig_idx] = new_idx;
            } else {
                let ni = new_positions.len() as u32;
                new_positions.push(pos);
                remap[orig_idx] = ni;
            }
        }
    }

    let old_len = positions.len();
    *positions = new_positions;
    // Parallel index remapping.
    indices.par_iter_mut().for_each(|idx| {
        *idx = remap[*idx as usize];
    });

    // Remove degenerate triangles (any two indices identical → zero area).
    // dedup can collapse two vertices of the same triangle into one,
    // producing sliver triangles that pass topology checks but render
    // as visual artifacts (lines/spikes).
    // Parallel: mark valid triangles, then compact.
    let old_tris = indices.len() / 3;
    let keep: Vec<bool> = (0..old_tris)
        .into_par_iter()
        .map(|t| {
            let base = t * 3;
            let a = indices[base];
            let b = indices[base + 1];
            let c = indices[base + 2];
            a != b && b != c && a != c
        })
        .collect();
    let mut write = 0;
    for t in 0..old_tris {
        if keep[t] {
            let base = t * 3;
            indices[write] = indices[base];
            indices[write + 1] = indices[base + 1];
            indices[write + 2] = indices[base + 2];
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
    use rustc_hash::FxHashMap;
    use rayon::prelude::*;

    let n_faces = indices.len() / 3;
    if n_faces == 0 { return; }

    // Pack edge (a, b) into a u64 key for faster hashing.
    let edge_key = |a: u32, b: u32| -> u64 {
        let lo = a.min(b) as u64;
        let hi = a.max(b) as u64;
        (hi << 32) | lo
    };

    let _t_adj = std::time::Instant::now();
    // ── Build edge → [(face, direction)] adjacency ──────────
    // For each edge, store (face_idx, is_forward) where is_forward = 1
    // if the face traverses the edge as (lo→hi).  Two faces sharing an
    // edge are consistent iff their is_forward values differ.
    let mut edge_faces: FxHashMap<u64, Vec<(usize, u8)>> =
        FxHashMap::with_capacity_and_hasher(n_faces * 3, Default::default());
    for (fi, tri) in indices.chunks_exact(3).enumerate() {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = edge_key(a, b);
            let is_forward = (a < b) as u8;
            edge_faces.entry(key).or_insert_with(|| Vec::with_capacity(2)).push((fi, is_forward));
        }
    }
    let _dt_adj = _t_adj.elapsed();

    let _t_uf = std::time::Instant::now();
    // ── Parity Union-Find: winding consistency + components ──
    // Each face has a parity (0 or 1) relative to its root.  parity 1
    // means the face needs to be flipped to match the root's orientation.
    // This replaces BFS traversal with a single union-find pass.
    let mut parent: Vec<usize> = (0..n_faces).collect();
    let mut parity: Vec<u8> = vec![0; n_faces];
    let mut rank = vec![0u32; n_faces];

    // find with path compression: returns (root, parity_from_x_to_root).
    fn find(par: &mut [usize], par_par: &mut [u8], x: usize) -> (usize, u8) {
        // Pass 1: find root and total parity from x to root.
        let mut root = x;
        let mut total_parity = 0u8;
        while par[root] != root {
            total_parity ^= par_par[root];
            root = par[root];
        }
        // Pass 2: path compression — rewire each node to point directly
        // to root, updating its parity accordingly.
        let mut cur = x;
        let mut acc = 0u8; // parity from original x to cur
        while par[cur] != root {
            let next = par[cur];
            let next_acc = acc ^ par_par[cur];
            par_par[cur] = total_parity ^ acc; // parity(cur → root)
            par[cur] = root;
            acc = next_acc;
            cur = next;
        }
        (root, total_parity)
    }

    for faces in edge_faces.values() {
        if faces.len() < 2 { continue; }
        let (fa, fa_fwd) = faces[0];
        for &(fb, fb_fwd) in &faces[1..] {
            // edge_parity = 1 if faces are inconsistent (same direction)
            let edge_parity = (fa_fwd == fb_fwd) as u8;

            let (ra, pa) = find(&mut parent, &mut parity, fa);
            let (rb, pb) = find(&mut parent, &mut parity, fb);
            if ra == rb { continue; }

            // Union by rank; set parity between roots so that
            // parity(fa→fb) = pa ^ parity[rb] ^ pb = edge_parity.
            let link_parity = pa ^ pb ^ edge_parity;
            if rank[ra] < rank[rb] {
                parent[ra] = rb;
                parity[ra] = link_parity;
            } else if rank[ra] > rank[rb] {
                parent[rb] = ra;
                parity[rb] = link_parity;
            } else {
                parent[rb] = ra;
                parity[rb] = link_parity;
                rank[ra] += 1;
            }
        }
    }
    let _dt_uf = _t_uf.elapsed();

    // ── Flip faces with parity 1 (relative to root) ─────────
    let _t_flip = std::time::Instant::now();
    let mut bfs_flipped = 0usize;
    for fi in 0..n_faces {
        let (_, p) = find(&mut parent, &mut parity, fi);
        if p == 1 {
            indices.swap(fi * 3 + 1, fi * 3 + 2);
            bfs_flipped += 1;
        }
    }
    let _dt_flip = _t_flip.elapsed();

    let _t_vol = std::time::Instant::now();
    // ── Per-component signed volume → flip inward comps ──────
    // Flatten roots to contiguous indices [0, n_comps).
    let mut root_id: Vec<usize> = vec![usize::MAX; n_faces];
    let mut n_comps = 0usize;
    for fi in 0..n_faces {
        let (r, _) = find(&mut parent, &mut parity, fi);
        if root_id[r] == usize::MAX {
            root_id[r] = n_comps;
            n_comps += 1;
        }
        root_id[fi] = root_id[r];
    }

    // Parallel per-face signed volume, accumulated per component.
    let comp_vol: Vec<f64> = {
        let partials: Vec<Vec<f64>> = (0..n_faces)
            .into_par_iter()
            .fold(
                || vec![0.0_f64; n_comps],
                |mut local, fi| {
                    let tri = &indices[fi * 3..fi * 3 + 3];
                    let p0 = glam::DVec3::from(glam::Vec3::from(positions[tri[0] as usize]));
                    let p1 = glam::DVec3::from(glam::Vec3::from(positions[tri[1] as usize]));
                    let p2 = glam::DVec3::from(glam::Vec3::from(positions[tri[2] as usize]));
                    local[root_id[fi]] += p0.dot(p1.cross(p2));
                    local
                },
            )
            .collect();
        let mut total = vec![0.0_f64; n_comps];
        for p in &partials {
            for (c, &v) in p.iter().enumerate() {
                total[c] += v;
            }
        }
        total
    };

    // Build comp → faces list for flipping.
    let mut comp_faces: Vec<Vec<usize>> = vec![Vec::new(); n_comps];
    for fi in 0..n_faces {
        comp_faces[root_id[fi]].push(fi);
    }

    let mut comps_flipped = 0usize;
    for (c, &vol) in comp_vol.iter().enumerate() {
        if vol < 0.0 {
            for &fi in &comp_faces[c] {
                indices.swap(fi * 3 + 1, fi * 3 + 2);
            }
            comps_flipped += 1;
        }
    }
    let _dt_vol = _t_vol.elapsed();
    eprintln!(
        "  winding: bfs_flipped={} components={} comps_flipped={} | adj={:.0}ms uf={:.0}ms flip={:.0}ms vol={:.0}ms",
        bfs_flipped, n_comps, comps_flipped,
        _dt_adj.as_secs_f64()*1e3, _dt_uf.as_secs_f64()*1e3,
        _dt_flip.as_secs_f64()*1e3, _dt_vol.as_secs_f64()*1e3,
    );
}

impl Mesh {
    /// Extract unique edge indices from triangle indices for edge (line) rendering.
    ///
    /// For each triangle (i0, i1, i2), generates 3 edges: (i0, i1), (i1, i2),
    /// (i2, i0).  Edges shared between adjacent triangles are stored only
    /// once (deduplicated by sorted vertex pair), which halves the buffer
    /// size for closed manifolds and avoids exceeding GPU buffer limits on
    /// high-resolution meshes.
    pub fn extract_edge_indices(&self) -> Vec<u32> {
        use std::collections::HashSet;
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let mut edge_indices = Vec::with_capacity(self.indices.len());
        for tri in self.indices.chunks(3) {
            if tri.len() == 3 {
                for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    let key = if a < b { (a, b) } else { (b, a) };
                    if seen.insert(key) {
                        edge_indices.extend_from_slice(&[a, b]);
                    }
                }
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

        // Defensive: drop triangles referencing any non-finite vertex.
        // A single inf/NaN position (e.g. from a degenerate marching-cubes
        // edge interpolation) would otherwise make the AABB diagonal inf,
        // inflate the dedup epsilon to inf, and collapse the entire mesh.
        let finite = |p: &[f32; 3]| p[0].is_finite() && p[1].is_finite() && p[2].is_finite();
        if positions.iter().any(|p| !finite(p)) {
            let before = indices.len() / 3;
            indices = indices
                .chunks_exact(3)
                .filter(|t| t.iter().all(|&i| finite(&positions[i as usize])))
                .flatten()
                .copied()
                .collect();
            eprintln!(
                "  sanitize: dropped {} triangles with non-finite vertices",
                before - indices.len() / 3
            );
        }

        // Dedup tolerance scaled to mesh size: 1e-6 of the AABB diagonal.
        // A fixed eps (e.g. 1e-4) over-merges vertices on high-resolution
        // meshes, creating non-manifold edges and winding inconsistencies
        // that produce holes under back-face culling.
        let diag = {
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for p in &positions {
                if !finite(p) { continue; }
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
        let _t_dedup = std::time::Instant::now();
        dedup_vertices(&mut positions, &mut indices, eps);
        let _dt_dedup = _t_dedup.elapsed();

        // Fix winding inconsistencies
        let _t_wind = std::time::Instant::now();
        if run_winding_fix {
            fix_winding(&positions, &mut indices);
        }
        let _dt_wind = _t_wind.elapsed();
        eprintln!("    [from_tri timing] dedup={:.0}ms fix_winding={:.0}ms",
            _dt_dedup.as_secs_f64()*1e3, _dt_wind.as_secs_f64()*1e3);

        let n_verts = positions.len();

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
