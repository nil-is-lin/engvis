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
            aabb.extend_pos(*pos);
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

impl Aabb {
    fn extend_pos(&mut self, pos: [f32; 3]) {
        let v = glam::Vec3::from(pos);
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }
}
