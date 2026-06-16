use glam::Affine3A;
use crate::light::LightingEnvironment;
use crate::material::PbrMaterial;
use crate::mesh::Mesh;

/// A scene node with transform
#[derive(Debug)]
pub struct SceneNode {
    pub name: String,
    pub local_transform: Affine3A,
    pub mesh_index: Option<usize>,
    pub children: Vec<SceneNode>,
    pub visible: bool,
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
