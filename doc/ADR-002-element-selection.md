# ADR-002: Element selection (point / edge / face) with visual feedback

## Status
Accepted

## Context
Users need to click an element in the 3D view and see which point / edge / face
was selected.  This is an **interaction / view concern**, not part of the mesh
model — selecting must not mutate `Mesh`, `Scene`, or `RenderState`'s geometry.

Two design questions had to be settled:

1. **Where does the picking math live?**
   `FrameCtx` exposes `camera`, `scene`, `viewport`, and `cursor`, but *not* the
   CPU mesh geometry (the renderer only holds GPU buffers).  So picking must
   run where the CPU `Mesh` is owned — i.e. in the host app.

2. **Where does the visual feedback render?**
   The overlay pipeline expands lines/points into quads in a shader and is fed
   by GPU instance buffers; to highlight one element through it the renderer
   would need the CPU mesh re-exposed.  That couples the renderer to picking
   state and adds a read-back surface.

## Decision
- **Picking is a CPU primitive in `engvis-core`**: `Mesh::pick(camera, cursor,
  viewport, threshold_px) -> Selection` projects each vertex / edge / face to
  screen pixels and returns the nearest within a radius.  `Selection` is a new
  `#[derive(Copy)]` enum (`None | Vertex(u32) | Edge(u32) | Face(u32)`).
  Camera projection helper `project_to_pixel` is exposed for callers.
- **Visual feedback is drawn by the app with egui**, not by the wgpu renderer.
  The app retains the CPU `Mesh` (already built for the scene) and re-projects
  the selected element every frame, drawing an orange marker + label on an
  egui background layer.  The renderer is untouched.
- **Selection state lives in the app** (`selection: Selection`, plus
  `pick_mesh` / `pick_edges` retained from the build result).  Click vs. drag is
  distinguished in `on_event` (press/release cursor delta < 4 px) so orbiting
  never triggers a pick; `egui_wants_pointer` suppresses picks on UI widgets.

## Consequences
- Good: selection adds **zero** changes to the renderer / wgpu pipelines and
  keeps the model data pure.  Picking is unit-testable and reversible.
- Good: egui feedback is always visible (drawn on top) and tracks the camera
  since it re-projects each frame.
- Trade-off accepted: feedback is 2D / screen-space (no depth test, always on
  top) rather than a true 3D highlight in the overlay pass.  For a selection
  indicator this is preferable (you always want to see what's selected) and
  avoids exposing CPU geometry to the renderer.
- Trade-off accepted: `Mesh::pick` is O(V + E + F) and only suitable for
  click-time use, not per-frame.  A BVH is a future optimization if meshes grow
  large; it would slot in behind the same `pick` signature.
- Cost: the app must retain the CPU `Mesh` (a `Clone` at build time).  Memory is
  one extra mesh copy — acceptable for inspection-scale geometry.
