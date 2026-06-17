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

    /// Create a smooth-shaded mesh from flat position and index arrays.
    ///
    /// Computes smooth vertex normals (area-weighted), zero UVs and a default tangent.
    pub fn from_triangles(
        name: impl Into<String>,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Self {
        let n_verts = positions.len();
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
        for p in positions { aabb.expand(glam::Vec3::from(*p)); }

        let index_count = indices.len() as u32;
        Mesh {
            name: name.into(),
            vertices,
            indices: indices.to_vec(),
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
