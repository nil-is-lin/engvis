/// Wireframe renderer using PolygonMode::Line
/// The pipeline is shared from MaterialPipeline (wireframe_pipeline field).
/// This module just provides the render helper.

pub struct WireframeRenderer;

impl WireframeRenderer {
    /// Render is done by setting the wireframe pipeline on the mesh renderer's render pass.
    /// No additional state needed since we reuse the same vertex format and bind groups.
    pub fn new() -> Self {
        Self
    }
}
