// engvis-mesher: trait-based multi-stage mesh pipeline.
//
// This module applies the "Rust 多阶段可插拔算法管线" pattern:
//   - One trait per pluggable stage (`Polygonizer`, `MeshPostProcessor`).
//   - Each algorithm is a parameter-bearing struct implementing its stage trait.
//   - Two dispatch modes:
//       * static generic `MeshPipeline<P, Q>`  — zero-cost, for batch builds
//       * dynamic `DynamicMeshPipeline`         — `Box<dyn Trait>`, runtime switch
//
// `MeshBackend` (the legacy enum) is retained ONLY as a config/UI bridge that
// produces a `Box<dyn Polygonizer>`. Adding a new backend now means adding a
// struct + trait impl + one enum arm — no `match` sprawl in `build_mesh`.

use engvis_core::mesh::Mesh;
use engvis_surface::Morphology;
use serde::{Deserialize, Serialize};

use crate::MeshBackend;

// ── Global data carrier ────────────────────────────────
//
// Stable, shared I/O for every polygonisation stage. Kept intentionally small;
// per-algorithm hyperparameters live inside the algorithm structs, not here.

pub struct PolygonizeRequest<'a> {
    pub tree: fidget_core::context::Tree,
    pub name: &'a str,
    pub domain_extent: [f32; 3],
}

// ── Stage 1: Polygoniser trait ─────────────────────────

/// Pluggable isosurface polygonisation algorithm.
/// Implemented by `DualContouring` and `MarchingCubes33`.
pub trait Polygonizer: std::fmt::Debug + Send + Sync + 'static {
    /// Human-readable algorithm name (for stats / UI).
    fn name(&self) -> &str;
    /// Produce a raw mesh from an (already morphology-transformed) implicit field.
    fn polygonize(&self, req: &PolygonizeRequest) -> (Mesh, String);
}

// ── Stage 2: Post-processor trait ──────────────────────

/// Pluggable in-place mesh post-processing step.
/// Implemented by `BallClip`, `GradientNormalRecompute`, etc.
pub trait MeshPostProcessor: std::fmt::Debug + Send + Sync + 'static {
    fn name(&self) -> &str;
    /// Mutate `mesh` in place; return a short stats suffix (or empty).
    fn apply(&self, mesh: &mut Mesh) -> String;
    /// Allow steps to be toggled without removing them from the pipeline.
    fn enabled(&self) -> bool {
        true
    }
}

// ── Algorithm structs (Stage 1) ────────────────────────

/// Dual Contouring: sharp features, then boundary smoothing onto C = 0.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DualContouring {
    pub depth: u8,
}

impl Polygonizer for DualContouring {
    fn name(&self) -> &str {
        "Dual Contouring"
    }
    fn polygonize(&self, req: &PolygonizeRequest) -> (Mesh, String) {
        crate::build_dc_mesh(req.tree.clone(), req.name, self.depth)
    }
}

/// Marching Cubes 33: smooth boundary, grid-resolved.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MarchingCubes33 {
    pub res: usize,
}

impl Polygonizer for MarchingCubes33 {
    fn name(&self) -> &str {
        "Marching Cubes 33"
    }
    fn polygonize(&self, req: &PolygonizeRequest) -> (Mesh, String) {
        let pad = 4.0 / self.res as f32;
        let mc_extent = [
            req.domain_extent[0] + pad,
            req.domain_extent[1] + pad,
            req.domain_extent[2] + pad,
        ];
        crate::build_mc33_mesh_domain(req.tree.clone(), req.name, self.res, mc_extent)
    }
}

// ── Algorithm structs (Stage 2) ────────────────────────

/// Clip the mesh to the interior of a ball (centre `center`, radius `radius`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BallClip {
    pub center: [f32; 3],
    pub radius: f32,
}

impl MeshPostProcessor for BallClip {
    fn name(&self) -> &str {
        "Ball Clip"
    }
    fn apply(&self, mesh: &mut Mesh) -> String {
        crate::clip_mesh_to_ball(mesh, self.center, self.radius);
        format!(
            " | clip({}v/{}t)",
            mesh.vertices.len(),
            mesh.indices.len() / 3
        )
    }
}

/// Recompute smooth per-vertex normals after a topology-changing edit.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct GradientNormalRecompute;

impl MeshPostProcessor for GradientNormalRecompute {
    fn name(&self) -> &str {
        "Gradient Normal Recompute"
    }
    fn apply(&self, mesh: &mut Mesh) -> String {
        crate::recompute_smooth_normals(mesh);
        String::new()
    }
}

/// No-op post-processor — lets the static pipeline run with zero post steps.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPost;

impl MeshPostProcessor for NoPost {
    fn name(&self) -> &str {
        "none"
    }
    fn apply(&self, _mesh: &mut Mesh) -> String {
        String::new()
    }
}

// ── Morphology transform (built-in, config-driven) ────
//
// Morphology is a fixed domain mode (Minimal / Shell / Skeletal), not a
// competing algorithm, so it stays a built-in transform rather than a trait.
// Only `Skeletal` rewrites the implicit field (wraps it in a box SDF).

fn apply_morphology(
    tree: fidget_core::context::Tree,
    morphology: Morphology,
    domain_extent: [f32; 3],
    mc_res: usize,
) -> fidget_core::context::Tree {
    match morphology {
        Morphology::Skeletal => {
            use fidget_core::context::Tree as T;
            let [sx, sy, sz] = domain_extent;
            let pad = 4.0 / mc_res as f32;
            let max_extent = sx.max(sy).max(sz).max(1.0);
            let half = max_extent + pad;
            let cell = 2.0 * half / (mc_res as f32 * half).ceil();
            let (cx, cy, cz) = (sx - 0.5 * cell, sy - 0.5 * cell, sz - 0.5 * cell);
            let box_sdf =
                (T::x().abs() - cx).max(T::y().abs() - cy).max(T::z().abs() - cz);
            tree.max(box_sdf)
        }
        _ => tree,
    }
}

// ── Static generic pipeline (zero-cost, batch) ────────

/// Compile-time-fixed algorithm combination. No vtable, no heap for stages.
pub struct MeshPipeline<P, Q>
where
    P: Polygonizer,
    Q: MeshPostProcessor,
{
    pub polygonizer: P,
    pub post: Q,
    pub morphology: Morphology,
    /// Resolution used only for the `Skeletal` box-SDF sizing.
    pub mc_res: usize,
}

impl<P, Q> MeshPipeline<P, Q>
where
    P: Polygonizer,
    Q: MeshPostProcessor,
{
    pub fn run(
        &self,
        tree: fidget_core::context::Tree,
        name: &str,
        domain_extent: [f32; 3],
    ) -> (Mesh, String) {
        let tree2 = apply_morphology(tree, self.morphology, domain_extent, self.mc_res);
        let req = PolygonizeRequest {
            tree: tree2,
            name,
            domain_extent,
        };
        let (mut mesh, mut stats) = self.polygonizer.polygonize(&req);
        if self.post.enabled() {
            let s = self.post.apply(&mut mesh);
            if !s.is_empty() {
                stats.push_str(&s);
            }
        }
        (mesh, stats)
    }
}

/// Pre-defined standard pipeline alias for quick instantiation.
pub type StandardPipeline = MeshPipeline<MarchingCubes33, NoPost>;

// ── Dynamic pipeline (runtime switchable) ─────────────

/// Runtime-configurable pipeline: polygoniser + ordered post-processing chain.
pub struct DynamicMeshPipeline {
    pub polygonizer: Box<dyn Polygonizer>,
    pub posts: Vec<Box<dyn MeshPostProcessor>>,
    pub morphology: Morphology,
    pub mc_res: usize,
}

impl DynamicMeshPipeline {
    pub fn run(
        &self,
        tree: fidget_core::context::Tree,
        name: &str,
        domain_extent: [f32; 3],
    ) -> (Mesh, String) {
        let tree2 = apply_morphology(tree, self.morphology, domain_extent, self.mc_res);
        let req = PolygonizeRequest {
            tree: tree2,
            name,
            domain_extent,
        };
        let (mut mesh, mut stats) = self.polygonizer.polygonize(&req);
        for post in &self.posts {
            if post.enabled() {
                let s = post.apply(&mut mesh);
                if !s.is_empty() {
                    stats.push_str(&s);
                }
            }
        }
        (mesh, stats)
    }
}

// ── Enum → trait-object bridge (config / UI / serde) ──

impl MeshBackend {
    /// Build a `Box<dyn Polygonizer>` for the chosen backend.
    pub fn make_polygonizer(&self, depth: u8, mc_res: usize) -> Box<dyn Polygonizer> {
        match self {
            MeshBackend::DualContouring => Box::new(DualContouring { depth }),
            MeshBackend::MarchingCubes33 => Box::new(MarchingCubes33 { res: mc_res }),
        }
    }
}

/// Serializable polygoniser selection (TOML/JSON config → trait object).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PolygonizerConfig {
    DualContouring { depth: u8 },
    MarchingCubes33 { res: usize },
}

impl PolygonizerConfig {
    pub fn build(&self) -> Box<dyn Polygonizer> {
        match self {
            PolygonizerConfig::DualContouring { depth } => Box::new(DualContouring { depth: *depth }),
            PolygonizerConfig::MarchingCubes33 { res } => Box::new(MarchingCubes33 { res: *res }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fidget_core::context::Tree as T;

    fn sphere_tree() -> T {
        (T::x().square() + T::y().square() + T::z().square()).sqrt() - 1.0
    }

    #[test]
    fn polygonizer_trait_dispatch() {
        let tree = sphere_tree();
        let dc = DualContouring { depth: 5 };
        let (mesh, stats) = dc.polygonize(&PolygonizeRequest {
            tree: tree.clone(),
            name: "t",
            domain_extent: [1.0, 1.0, 1.0],
        });
        assert!(!stats.is_empty());
        assert!(mesh.indices.len() >= 3);
    }

    #[test]
    fn dynamic_pipeline_runtime_switch() {
        let tree = sphere_tree();
        // Runtime switching between backends — no recompile, no match arms.
        let backends: Vec<Box<dyn Polygonizer>> = vec![
            Box::new(DualContouring { depth: 5 }),
            Box::new(MarchingCubes33 { res: 32 }),
        ];
        for poly in backends {
            let pipe = DynamicMeshPipeline {
                polygonizer: poly,
                posts: vec![],
                morphology: Morphology::MinimalSurface,
                mc_res: 32,
            };
            let (mesh, _stats) = pipe.run(tree.clone(), "t", [1.0, 1.0, 1.0]);
            assert!(mesh.indices.len() >= 3);
        }
    }

    #[test]
    fn static_pipeline_zero_cost() {
        let tree = sphere_tree();
        let pipe: MeshPipeline<MarchingCubes33, NoPost> = MeshPipeline {
            polygonizer: MarchingCubes33 { res: 32 },
            post: NoPost,
            morphology: Morphology::MinimalSurface,
            mc_res: 32,
        };
        let (mesh, _stats) = pipe.run(tree, "t", [1.0, 1.0, 1.0]);
        assert!(mesh.indices.len() >= 3);
    }

    #[test]
    fn dynamic_pipeline_with_post_chain() {
        let tree = sphere_tree();
        let pipe = DynamicMeshPipeline {
            polygonizer: Box::new(MarchingCubes33 { res: 32 }),
            posts: vec![
                Box::new(BallClip { center: [0.0, 0.0, 0.0], radius: 1.0 }),
                Box::new(GradientNormalRecompute),
            ],
            morphology: Morphology::MinimalSurface,
            mc_res: 32,
        };
        let (mesh, stats) = pipe.run(tree, "t", [1.0, 1.0, 1.0]);
        assert!(mesh.indices.len() >= 3);
        assert!(stats.contains("clip"), "expected clip stats, got: {stats}");
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = PolygonizerConfig::MarchingCubes33 { res: 64 };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: PolygonizerConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            PolygonizerConfig::MarchingCubes33 { res: 64 }
        ));
        // The deserialized config must build a working polygoniser.
        let poly = back.build();
        let (mesh, _) = poly.polygonize(&PolygonizeRequest {
            tree: sphere_tree(),
            name: "t",
            domain_extent: [1.0, 1.0, 1.0],
        });
        assert!(mesh.indices.len() >= 3);
    }
}
