use glam::Affine3A;
use crate::light::LightingEnvironment;
use crate::material::PbrMaterial;
use crate::mesh::Mesh;
use crate::aabb::Aabb;

/// A scene node with transform
#[derive(Debug)]
pub struct SceneNode {
    pub name: String,
    pub local_transform: Affine3A,
    pub mesh_index: Option<usize>,
    pub children: Vec<SceneNode>,
    pub visible: bool,
    /// Whether to draw this node's mesh through the solid surface pipeline.
    /// Set to `false` for wireframe-only nodes (bounding box, clip sphere)
    /// whose degenerate triangles would produce aliased 1px lines that
    /// interfere with the smooth edge-overlay rendering.
    pub render_surface: bool,
    /// Whether to draw this node's edges in the edge-overlay pass.
    /// Used for wireframe bounding boxes / outlines that should not
    /// light up every triangle edge of the surface mesh.
    pub render_edges: bool,
    /// Optional per-node override for the edge-overlay color.
    /// `None` falls back to the global `RenderState::edge_opts.color`.
    pub edge_color_override: Option<[f32; 3]>,
    /// Optional per-node override for the edge-overlay line width.
    /// `None` falls back to the global `RenderState::edge_opts.line_width`.
    pub edge_width_override: Option<f32>,
}

/// The top-level scene containing all data
#[derive(Debug)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<PbrMaterial>,
    pub nodes: Vec<SceneNode>,
    pub lighting: LightingEnvironment,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
            materials: vec![PbrMaterial::default()],
            nodes: Vec::new(),
            lighting: LightingEnvironment::default(),
        }
    }
}

// ── Convenience constructors ─────────────────────────────────
impl Scene {
    /// Create a scene with a single mesh, one default material, and one node.
    pub fn single_mesh(name: impl Into<String>, mesh: Mesh, material: PbrMaterial) -> Self {
        let material_index = 0;
        let mut mesh = mesh;
        for sub in &mut mesh.sub_meshes {
            sub.material_index = material_index;
        }
        Self {
            meshes: vec![mesh],
            materials: vec![material],
            nodes: vec![SceneNode {
                name: name.into(),
                local_transform: Affine3A::IDENTITY,
                mesh_index: Some(0),
                children: Vec::new(),
                visible: true,
                render_surface: true,
                render_edges: false,
                edge_color_override: None,
                edge_width_override: None,
            }],
            lighting: LightingEnvironment::default(),
        }
    }

    /// Compute the world-space bounding box of the scene by traversing all nodes.
    pub fn compute_aabb(&self) -> Aabb {
        let mut out = Aabb::empty();
        for node in &self.nodes {
            out = out.union(&self.compute_node_aabb(node, Affine3A::IDENTITY));
        }
        out
    }

    fn compute_node_aabb(&self, node: &SceneNode, parent_transform: Affine3A) -> Aabb {
        if !node.visible {
            return Aabb::empty();
        }
        let world = parent_transform * node.local_transform;
        let mut out = Aabb::empty();
        if let Some(mesh_idx) = node.mesh_index
            && let Some(mesh) = self.meshes.get(mesh_idx) {
                out = Aabb::from_transformed_aabb(&mesh.aabb, &world);
            }
        for child in &node.children {
            out = out.union(&self.compute_node_aabb(child, world));
        }
        out
    }

    /// Use default lighting (two directional + ambient).
    pub fn with_default_lighting(self) -> Self {
        Self {
            lighting: LightingEnvironment::default(),
            ..self
        }
    }

    /// Replace the lighting environment.
    pub fn with_lighting(mut self, lighting: LightingEnvironment) -> Self {
        self.lighting = lighting;
        self
    }
}
