# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-06-18

### Added

- `engvis-core`: Core data types — `Scene`, `Mesh`, `MeshVertex`, `SubMesh`, `PbrMaterial`, `OrbitCamera`, `InputState`, `Aabb`, lighting types.
- `engvis-renderer`: wgpu-based GPU rendering pipeline with PBR shading, grid/overlay rendering, egui integration, glTF loader via `load_gltf`.
- `EngvisApp` trait for declarative app definition (`config`, `on_setup`, `on_ready`, `ui`, `on_frame`, `on_event`).
- `FrameCtx` per-frame context with mutable scene, camera, render state, and viewport access.
- Orbit camera with orbit/zoom/pan input, view presets (front, top, right, iso), and `fit_to_scene` / `fit_to_aabb`.
- Surface opacity slider, vertex/edge overlay rendering with adjustable color and size.
- MSAA support (configurable sample count via `RunConfig`).
- `hello_viewer` example demonstrating minimal `EngvisApp` usage.
- `fidget-demo`: implicit surface viewer using `fidget-core` + `fidget-mesh` with interactive shape/material selection.

### Fixed

- Edge and vertex overlays not displaying due to incorrect viewport uniform.
- Grid not visible through transparent surface due to rendering order.
- Edge overlay using hardcoded color instead of user-selected color.

## [0.1.2] - 2026-06-19

### Added

- **UI shell**: four-panel egui layout with top `MenuBar` (File/View/Help),
  left numbered workflow panel (1.Source 2.Mesh 3.Display 4.Topology),
  right detail panel, and bottom status bar showing
  $V,E,F,\chi=V-E+F$, boundary edges, components, watertightness, FPS.
- **Mesh I/O** (`src/mesh_io.rs`): import/export OBJ (via `tobj` + inline writer),
  STL (via `stl_io`), PLY (via `ply-rs-bw` + inline ASCII writer).
  Native file dialogs via `rfd`.
- **Custom implicit expressions**: `fidget-rhai` integration allows
  users to type arbitrary Rhai scripts (e.g. `sin(4*x)*cos(4*y)+...`)
  that compile directly to `fidget::Tree`.
- **Extended TPMS catalog**: Schwarz P, Schwarz D, Schoen IWP, Neovius
  added to the built-in surface list, alongside sphere, torus, and gyroid.
- **Marching Cubes 33** (`engvis-core/src/marching_cubes.rs`):
  full MC33 implementation with Bourke lookup table, boundary smoothing,
  and winding-number fix.
- **Topology analysis** (`engvis-core/src/topology.rs`): half-edge
  algorithm computing $V,E,F$, Euler characteristic $\chi$, boundary edges,
  non-manifold edges, connected components, and watertightness.
- **Annotation primitives** (`engvis-core/src/annotation.rs`): sphere
  wireframe, box wireframe builders for bounding visualisation.
- **Post-process pipeline** (`engvis-renderer/src/postprocess.rs`):
  screen-space post-processing support.
- **Colorbar 3D example** (`examples/colorbar3d/`): stand-alone demo
  application.
- **LaTeX documentation** for each code module (gyroid clipping, boundary
  smoothing, mesh winding fix, camera quaternion orbit, UI shell & mesh I/O).

### Changed

- `build_mesh` / `build_dc_mesh` / `build_mc33_mesh` now accept a
  `fidget::Tree` directly instead of a name string; the tree source
  (built-in or Rhai) is resolved in the caller.
- `App` UI fully rewritten from ad-hoc `egui::Window`s to the four-panel
  layout with `MenuBar`.
- `PbrMaterial` extended with `roughness` and smoothed rendering parameters.
- `SceneNode` gained `render_edges` flag for per-node edge overlay control.

### Fixed

- Dependency conflict: `facet-path 0.44.5` incorrectly depends on
  `facet-core ^0.45`, conflicting with `facet 0.44.x`.  Pinned to
  `facet-path 0.44.4` in lockfile.

## [0.1.1] - 2026-06-18

### Added

- `keywords`, `categories`, `homepage`, `repository` metadata for crates.io publishing.
- Crate-level `#![doc]` for `engvis-core` and `engvis-renderer`.
- SSH-based GitHub push configuration.

### Changed

- Workspace version bumped to 0.1.1.
- Repo URL in `Cargo.toml` and `README` updated to `github.com/nil-is-lin/engvis`.
- Grid rendering moved before surface render pass so transparent surfaces show the grid through.
- All clippy warnings resolved across workspace.

### Removed

- Dead `scene_callback.rs` referencing non-existent `RenderMode`.
- README personal email address.
- DEBUG overlay test block in render pass.

### Published

- `engvis-core` v0.1.1 on crates.io.
- `engvis-renderer` v0.1.1 on crates.io.
