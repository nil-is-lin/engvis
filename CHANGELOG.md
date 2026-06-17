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
