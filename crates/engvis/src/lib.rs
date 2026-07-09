//! # engvis — engineering visualization framework
//!
//! Convenience facade that re-exports the functional crates so you can
//! compose them without listing each dependency:
//!
//! ```toml
//! [dependencies]
//! engvis = "0.1.5"
//! ```
//!
//! ```rust
//! use engvis::prelude::*;
//! ```
//!
//! For fine-grained control (e.g. to avoid pulling the GPU stack), depend on
//! the individual crates directly: `engvis-core`, `engvis-surface`,
//! `engvis-tpms`, `engvis-mesher`, `engvis-renderer`.
//!
//! ## Crate roles
//! - `engvis-core` — data model: `Mesh`, `Scene`, `PbrMaterial`, `RenderState`, `OrbitCamera`.
//! - `engvis-surface` — pure-math primitive implicit surfaces (`build_tree`).
//! - `engvis-tpms` — TPMS families, formulas, volume-fraction solver, `Morphology`.
//! - `engvis-mesher` — mesh generation (DC, MC33, shell, MS-loops) from a field.
//! - `engvis-renderer` — wgpu/egui viewer (`EngvisApp`, `run`).

pub use engvis_core;
pub use engvis_mesher;
pub use engvis_renderer;
pub use engvis_surface;
pub use engvis_tpms;

/// Commonly used items from all crates.
pub mod prelude {
    pub use engvis_core::{
        camera::OrbitCamera,
        material::{EdgeRenderOptions, PbrMaterial, RenderState, VertexRenderOptions},
        mesh::Mesh,
        scene::{Scene, SceneNode},
        selection::{project_to_pixel, Selection, DEFAULT_PICK_THRESHOLD_PX},
    };
    pub use engvis_mesher::{build_mesh, MeshBackend};
    pub use engvis_renderer::{run, EngvisApp, FrameCtx};
    pub use engvis_surface::{build_tree, SurfaceType, TreeParams};
    pub use engvis_tpms::{Morphology, TpmsFamily, TpmsSurface};
}
