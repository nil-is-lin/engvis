#![doc = "Core data types for engineering visualization.\n\nProvides `Scene`, `Mesh`, `PbrMaterial`, `OrbitCamera`, and lighting types."]

pub mod math;
pub mod aabb;
pub mod camera;
pub mod mesh;
pub mod material;
pub mod light;
pub mod scene;
pub mod input;
pub mod annotation;
pub mod topology;
pub mod bourke_table;
pub mod marching_cubes;

pub use camera::OrbitCamera;
pub use input::{InputState, ViewportRect};
pub use scene::{Scene, SceneNode};
pub use mesh::{Mesh, MeshVertex, SubMesh, fix_winding, dedup_vertices};
pub use material::{PbrMaterial, VertexRenderOptions, EdgeRenderOptions, RenderState};
pub use light::{AmbientLight, DirectionalLight, PointLight, LightingEnvironment};
pub use aabb::Aabb;
pub use topology::{MeshTopology, compute_topology};
