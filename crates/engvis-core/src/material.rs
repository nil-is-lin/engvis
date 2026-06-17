/// Describes a PBR material
#[derive(Debug, Clone)]
pub struct PbrMaterial {
    pub name: String,
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub normal_scale: f32,
    pub alpha_cutoff: f32,

    /// Texture indices (into a TextureCache)
    pub albedo_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub emissive_texture: Option<usize>,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            albedo: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0, 0.0, 0.0],
            normal_scale: 1.0,
            alpha_cutoff: 0.5,
            albedo_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            emissive_texture: None,
        }
    }
}

/// Vertex (point) overlay rendering options
#[derive(Debug, Clone, Copy)]
pub struct VertexRenderOptions {
    pub enabled: bool,
    pub color: [f32; 3],
    pub point_size: f32,
}

impl Default for VertexRenderOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            color: [1.0, 0.85, 0.3],
            point_size: 4.0,
        }
    }
}

/// Edge (line) overlay rendering options
#[derive(Debug, Clone, Copy)]
pub struct EdgeRenderOptions {
    pub enabled: bool,
    pub color: [f32; 3],
    pub line_width: f32,
}

impl Default for EdgeRenderOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            color: [0.2, 0.8, 1.0],
            line_width: 5.0,
        }
    }
}

// ── RenderState ──────────────────────────────────────────────

/// Aggregated rendering state that can be applied atomically.
///
/// Use `FrameCtx::set_render_state()` to update all rendering options in one call,
/// or modulate individual fields directly via `frame.render_state.field`.
///
/// Clipping planes (near/far) are not part of this struct; use
/// `FrameCtx::set_clip_planes()` to set them explicitly.
#[derive(Debug, Clone, Copy)]
pub struct RenderState {
    /// Whether to render the surface (solid PBR mesh).
    pub show_surface: bool,
    /// Whether to render the ground/axis grid.
    pub show_grid: bool,
    /// Global surface opacity (0.0 = fully transparent, 1.0 = fully opaque).
    pub opacity: f32,
    /// Vertex (point) overlay options.
    pub vertex_opts: VertexRenderOptions,
    /// Edge (wireframe) overlay options.
    pub edge_opts: EdgeRenderOptions,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            show_surface: true,
            show_grid: true,
            opacity: 1.0,
            vertex_opts: VertexRenderOptions::default(),
            edge_opts: EdgeRenderOptions::default(),
        }
    }
}
