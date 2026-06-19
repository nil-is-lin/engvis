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

## [0.1.5] - 2026-06-20

### Added

- **Parameterised primitives**: sphere radius (0.1–3.0), torus major/minor radius
  (0.1–3.0 / 0.02–1.5) controllable from the Source panel, directly below the
  selected primitive.
- **Parameterised TPMS**: period slider $k \in [0.5, 10]$ and human-readable
  implicit equation shown below the dropdown.
- **Clip radius control**: `clip_radius` slider (0.2–5.0) in the Source panel;
  sphere wireframe radius and camera adapt to it.
- **Surface color picker** in the Display panel (Color under Show triangle surface).
- **3 new Fischer-Koch TPMS**: Fischer-Koch S, Y, CP (10 total).
- **Adaptive MC33 resolution**: auto-bumps grid resolution when torus tube or
  TPMS half-period drops below ~3 grid cells (caps at 512). User slider unchanged.
- **Boundary vertex clamping**: post-MC33 pass clamps coordinates to $[-1,1]^3$,
  preventing floating-point overshoot from visually overflowing the wireframe cube.
- **Empty-mesh guard** in all three rendering passes (surface, points, edges):
  nodes with zero vertex/index/edge-instance counts are silently skipped.

### Changed

- Source panel restructured: primitive shapes are radio buttons, TPMS uses
  `egui::ComboBox` dropdown.
- `build_tree` takes a `TreeParams` struct instead of a bare `&str`.
- `build_mesh` / `build_mc33_mesh` / `build_box_wireframe`: sampling domain
  remains fixed at $[-1,1]^3$ (adaptive-domain approach explicitly rejected
  to preserve user feedback on the period slider).
- Camera always fits to $[-1,1]^3$ regardless of `clip_radius`.
- Edge-overlay traversal split into a dedicated `render_overlay_nodes_edges`
  path with per-node color/width overrides (`SceneNode::edge_color_override`,
  `edge_width_override`).

### Fixed

- Empty mesh on sphere/torus extreme parameters no longer panics (buffer
  size guards in renderer).
- `facet-path 0.44.5` dependency conflict pinned in lockfile.
- Release workflow Windows runner `Connection was reset` mitigated with
  sparse registry protocol, `cargo fetch` retry loop, and `rust-cache`.

### Documentation

- `doc/mc33-domain-auto-scale.tex`: fixed-unit-domain convention, boundary
  clamping, and adaptive resolution design.

## [0.1.4] - 2026-06-19

### Added

- **Per-node edge appearance overrides** (`SceneNode::edge_color_override`,
  `edge_width_override`).  Lets the bounding wireframe / annotations
  carry their own color and line width, independent of the global
  triangle-edge overlay.
- **Display panel UX overhaul**: surface, edges, points, and bounding
  wireframe are now toggle-driven — controls only appear once the
  corresponding "Show …" checkbox is enabled (matching the existing
  *Show points* style).
- **Bounding wireframe color and line width** controls in the UI.

### Changed

- Edge overlay traversal split into a dedicated path that builds one
  uniform bind group per node, allowing per-node overrides without
  duplicating the line pipeline.
- `OverlayDrawMode` enum removed (was reduced to a single variant
  after edge handling moved out).

### Fixed

- Release workflow on Windows runners would occasionally fail with
  `Connection was reset` while pulling crates.io.  Mitigated by:
  - `CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse`
  - `CARGO_NET_RETRY=10`, `CARGO_HTTP_MULTIPLEXING=false`,
    `CARGO_NET_GIT_FETCH_WITH_CLI=true`
  - explicit `cargo fetch` step with up-to-5-time retry loop
  - `Swatinem/rust-cache@v2` for registry / target caching
  - `fail-fast: false` so one platform's transient failure does not
    cancel the others

## [0.1.3] - 2026-06-19

### Changed

- Release assets: each platform now uploads the raw binary alongside the
  compressed archive (`.tar.gz`/`.zip`).

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
