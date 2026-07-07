# ADR-001: engvis crate decomposition for independent use and crates.io publishability

## Status
Accepted

## Context
The user wants the project split so a consumer can use a single functional part
independently, or compose several parts together. They are also evaluating
publishing to crates.io. We audited the actual `Cargo.toml` and `lib.rs` of every
crate (not assumptions).

Verified dependency facts:

- `engvis-core`: deps `glam`, `bytemuck`, `rayon`, `meshopt`, `rustc-hash`. No GPU,
  no fidget. Owns `Mesh`, `Scene`, `PbrMaterial`, `RenderState`, `OrbitCamera`,
  lighting, `Aabb`, topology, and the marching-cubes tables (`marching_cubes`,
  `bourke_table`).
- `engvis-surface`: deps `fidget-core`, `fidget-rhai` only. Does **not** depend on
  `engvis-core`. Pure math: `SurfaceType`, TPMS formulas, `TreeParams`,
  `build_tree` -> `fidget_core::context::Tree`.
- `engvis-mesher`: deps `engvis-surface`, `engvis-core`, `fidget-*`, `glam`, `serde`.
  Returns `engvis_core::Mesh`. Does **not** depend on `engvis-renderer`. Also
  accepts a raw `fidget Tree` directly, so `engvis-surface` is optional for meshing.
- `engvis-renderer`: deps `engvis-core` + `wgpu`, `winit`, `egui`, `egui-wgpu`,
  `egui-winit`, `pollster`, `gltf`, `image`. Does **not** depend on `surface`/`mesher`.
- `engvis` (binary): composition root depending on all four.

The dependency graph is already a clean DAG: `engvis-surface` is independent of
`engvis-core`; `engvis-renderer` is independent of `engvis-mesher`/`engvis-surface`.
So the project is **already splittable** — the open question is publishability and
ergonomics, not whether it can be split.

## Decision
Keep the four-crate split. Improve it along two axes:

1. **Composition ergonomics** — add a thin facade library crate `engvis` that
   re-exports the four crates (e.g. `engvis::prelude`), and rename the current
   binary to `engvis-cli` so the `engvis` name becomes the library facade.
2. **Publish mechanics** — replace intra-workspace `path` deps with `version` deps
   and publish in topological order `core -> surface -> mesher -> renderer`
   (-> facade), completing each crate's publish metadata (`readme`, `description`,
   `repository`, `license` — most already present).

Rejected alternative: collapsing everything into one crate behind `#[cfg(feature)]`
flags. That avoids multi-crate publishing but forces every consumer (even a
render-only or data-model-only user) to compile `wgpu`/`egui`/`fidget`, which
directly contradicts the independent-use goal.

## Consequences
- Easier: a render-only user pulls `engvis-core` + `engvis-renderer` (no fidget,
  no meshing). A mesh-only user pulls `engvis-mesher` + `engvis-core` (no GPU).
  A math-only user pulls just `engvis-surface`. Full-stack users do
  `use engvis::prelude::*`.
- Easier: clean `cargo publish` story once deps are version-based and ordered.
- Harder: one extra crate (the facade) to version and maintain.
- Optional refinement (Option B, not required): move `marching_cubes` /
  `bourke_table` (mesher algorithm data) out of `engvis-core` into `engvis-mesher`,
  so `engvis-core` becomes a pure data-model crate. Improves conceptual purity but
  adds churn; not a publish blocker.
- Trade-off accepted: a handful of crates is the right granularity for this domain;
  further splitting risks architecture-astronautics.

## Implementation (2026-07-08)

Decision accepted and implemented. Concrete changes:

- **Facade crate** `engvis` created at `crates/engvis/` — thin library that
  `pub use`s the four crates and exposes a `prelude` module
  (`OrbitCamera`, `PbrMaterial`/`RenderState`/`EdgeRenderOptions`/
  `VertexRenderOptions`, `Mesh`, `Scene`/`SceneNode`, `build_mesh`/`MeshBackend`,
  `run`/`EngvisApp`/`FrameCtx`, `build_tree`/`Morphology`/`SurfaceType`/`TreeParams`).
- **Binary renamed** `engvis` → `engvis-cli` (root `Cargo.toml` `[package] name`).
  The `engvis` name now belongs to the library facade.
- **Publish mechanics**: internal `path` deps upgraded to `version` deps
  (`version = "0.1.5"`) so the crates can be `cargo publish`ed independently.
- **Publish metadata**: `repository` added to `[workspace.package]`;
  `engvis-surface` gained `keywords`/`categories`/`homepage`/`repository`.
- Verified: `cargo check --workspace` (both `engvis` and `engvis-cli`),
  `cargo test -p engvis --doc` (1 doctest passes — re-exports resolve),
  `cargo build -p engvis-cli` (links).

Topological publish order remains `core -> surface -> mesher -> renderer -> facade`.
`cargo publish` was not yet executed (awaiting explicit go-ahead + network).

Optional refinement (Option B) — moving `marching_cubes`/`bourke_table` from
`engvis-core` into `engvis-mesher` — remains deferred; not a publish blocker.
