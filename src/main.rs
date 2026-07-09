// engvis viewer — Dual Contouring + Marching Cubes 33 meshing of implicit surfaces
// with boundary smoothing and MS-loop visualization.

mod mesh_io;
mod formula_cache;

use engvis_core::{
    material::{PbrMaterial, RenderState, EdgeRenderOptions},
    scene::{Scene, SceneNode},
    camera::OrbitCamera,
    topology::compute_topology,
    selection::{Selection, project_to_pixel},
    Mesh,
    Aabb,
};
use engvis_renderer::{
    EngvisApp, AppCtx, FrameCtx, RunConfig, EventHandling, load_gltf,
};
use engvis_surface::{
    TreeParams, SurfaceType, build_tree, build_tree_from_rhai,
};
use engvis_tpms::{
    GradientField, GradientMode, Morphology, TpmsFamily, TpmsSurface, tpms_formula,
};
use engvis_mesher::{
    MeshBackend, build_box_wireframe, build_sphere_wireframe,
    build_shell_mesh, build_mesh, build_ms_loops_mesh,
};

use std::sync::{Arc, Mutex};
use std::thread;

/// 渲染 GradientField 的 UI 控件（mode 下拉 + 参数滑块）。
fn gradient_field_ui(
    ui: &mut egui::Ui,
    g: &mut GradientField,
    needs_remesh: &mut bool,
    id_salt: &str,
) {
    let mode_id = egui::Id::new(("grad_mode", id_salt));
    egui::ComboBox::from_id_salt(mode_id)
        .selected_text(match g.mode {
            GradientMode::None => "None",
            GradientMode::Linear => "Linear",
            GradientMode::Sigmoid => "Sigmoid",
            GradientMode::BoundaryDecay => "BoundaryDecay",
        })
        .show_ui(ui, |ui| {
            for (m, label) in [
                (GradientMode::None, "None"),
                (GradientMode::Linear, "Linear"),
                (GradientMode::Sigmoid, "Sigmoid"),
                (GradientMode::BoundaryDecay, "BoundaryDecay"),
            ] {
                if ui.selectable_label(g.mode == m, label).clicked() {
                    g.mode = m;
                    *needs_remesh = true;
                }
            }
        });
    if matches!(g.mode, GradientMode::None) {
        return;
    }
    egui::Grid::new(("grad_grid", id_salt))
        .num_columns(2).spacing([8.0, 4.0])
        .show(ui, |ui| {
            ui.label("Axis x");
            if ui.add(egui::Slider::new(&mut g.axis[0], -1.0..=1.0).text("")).changed() {
                *needs_remesh = true;
            }
            ui.end_row();
            ui.label("Axis y");
            if ui.add(egui::Slider::new(&mut g.axis[1], -1.0..=1.0).text("")).changed() {
                *needs_remesh = true;
            }
            ui.end_row();
            ui.label("Axis z");
            if ui.add(egui::Slider::new(&mut g.axis[2], -1.0..=1.0).text("")).changed() {
                *needs_remesh = true;
            }
            ui.end_row();
            ui.label("Base");
            if ui.add(egui::Slider::new(&mut g.base, -2.0..=2.0).text("")).changed() {
                *needs_remesh = true;
            }
            ui.end_row();
            ui.label("Delta");
            if ui.add(egui::Slider::new(&mut g.delta, -2.0..=2.0).text("")).changed() {
                *needs_remesh = true;
            }
            ui.end_row();
            let center_label = match g.mode {
                GradientMode::Linear => "Span",
                GradientMode::Sigmoid => "Center x0",
                GradientMode::BoundaryDecay => "Radius R",
                GradientMode::None => "Center",
            };
            ui.label(center_label);
            if ui.add(egui::Slider::new(&mut g.center, 0.01..=3.0).text("")).changed() {
                *needs_remesh = true;
            }
            ui.end_row();
            if matches!(g.mode, GradientMode::Sigmoid | GradientMode::BoundaryDecay) {
                ui.label("Sharpness");
                if ui.add(egui::Slider::new(&mut g.sharpness, 0.1..=20.0).text("")).changed() {
                    *needs_remesh = true;
                }
                ui.end_row();
            }
        });
}

// =====================================================================
// 5. App
// =====================================================================

/// Tabs shown in the right panel.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Tab {
    Surface, // TPMS type + formula + primitives + custom expr
    Cell,    // period, cells, Lx/Ly/Lz, amplitude, offset C
    Deform,  // rotation, blend, gradient fields
    Morph,   // minimal/shell/skeletal + params
    Mesh,    // backend + simplification
    Display, // visual settings
    Topo,    // topology stats
}

/// Meshes with more vertices than this are treated as "dense": instanced
/// point/edge overlays are capped by the renderer (above `MAX_OVERLAY_ELEMENTS`)
/// to avoid GPU stalls, and the UI shows a hint. 250k is a conservative point
/// where overlay instancing starts costing more than it informs.
const DENSE_MESH_VERTS: usize = 250_000;

struct App {
    // ── implicit surface source ──
    source: SurfaceType,
    /// Active TPMS family, if the current surface is a TPMS. None for
    /// primitives / custom. This is the "TPMS is its own type" selector
    /// (ADR-003); the raw `tpms_*` fields below are assembled into a
    /// `TpmsSurface` when building the tree.
    tpms_family: Option<TpmsFamily>,
    custom_expr: String,
    custom_error: Option<String>,
    clip_to_unit_ball: bool,
    clip_radius: f32,

    // ── shape parameters (persisted across surface switches) ──
    sphere_radius: f32,
    torus_major_r: f32,
    torus_minor_r: f32,
    /// TPMS 内在周期 k（单个晶胞内的空间频率）。
    tpms_period: f32,
    /// 各方向晶胞堆叠数 [nx, ny, nz]；总晶胞数 = nx×ny×nz。
    tpms_cells: [u32; 3],
    tpms_thickness: f32,
    /// 体积分数 φ ∈ [0,1]：0.5=对称极小曲面，→0/1=完全填充。
    tpms_vol_frac: f32,
    /// 缓存的 C 值（由 tpms_vol_frac 求解得到），用于 UI 显示。
    /// 在 remesh 时更新；初始值 0.0。
    cached_c_value: f32,
    /// 三轴单胞边长 [Lx, Ly, Lz]，默认 [1,1,1]。
    tpms_cell_size: [f32; 3],
    /// 三轴幅值缩放系数 [a, b, c]，默认 [1,1,1]。
    tpms_amplitude: [f32; 3],
    /// 全局等值面偏移 C，默认 0（高级模式）。
    tpms_offset: f32,
    /// 旋转轴（单位向量），用于全局单胞旋转。
    rotation_axis: [f32; 3],
    /// 旋转角（度数，UI 友好；构造 Tree 时转弧度）。
    rotation_angle_deg: f32,
    /// 双 TPMS 混合：副曲面名（None 表示禁用）。
    blend_secondary: Option<String>,
    /// 混合权重空间场（值域被裁剪到 [0,1]）。
    blend_weight_field: GradientField,
    /// 过渡方向轴索引：0=x, 1=y, 2=z。
    blend_axis: u32,
    /// f1 (Primary) 占据的 cell 范围 [start, end]（沿 blend_axis 方向）。
    blend_f1_cells: [u32; 2],
    /// f2 (Secondary) 占据的 cell 范围 [start, end]。
    blend_f2_cells: [u32; 2],
    /// 等值面偏移 C 的空间梯度场（None 模式时回落到 tpms_offset 标量）。
    offset_field: GradientField,
    morphology: Morphology,

    // ── meshing ──
    mesh_backend: MeshBackend,
    surf_depth: u8,
    mc_res: usize,
    show_ms_loops: bool,
    show_bounding: bool,

    // ── mesh simplification ──
    simplify_enabled: bool,
    simplify_ratio: f32,
    simplify_stats: String,

    // ── render state (single source of truth for display/overlay opts) ──
    // Owned by the App and pushed to the renderer each frame via
    // `FrameCtx::set_render_state`.  Per-node overrides that require a
    // remesh (below) are kept as separate fields.
    render_state: RenderState,
    // Triangle-mesh edges of the *surface* node — toggling this requires
    // reapplying `render_edges` on the node, which happens on remesh.
    show_surface_edges: bool,
    // Bounding-wireframe appearance, independent of the surface-edge
    // overlay above.  Applied as a per-node override in `build_scene`.
    wireframe_color: [f32; 3],
    wireframe_line_width: f32,

    // ── surface appearance (PBR material; requires remesh) ──
    surface_color: [f32; 3],
    /// PBR metallic factor (0 = dielectric, 1 = metal).
    surface_metallic: f32,
    /// PBR roughness factor (0 = mirror, 1 = fully rough).
    surface_roughness: f32,

    // ── workflow / UI ──
    current_tab: Tab,

    // ── load / save ──
    pending_load: Option<std::path::PathBuf>,
    pending_save: Option<std::path::PathBuf>,

    // ── runtime state ──
    needs_remesh: bool,
    camera_fitted: bool,
    last_topology: Option<engvis_core::topology::MeshTopology>,
    last_build_ok: bool,
    /// True when the current mesh is too dense for instanced point/edge
    /// overlays. The renderer caps overlays per node above
    /// `MAX_OVERLAY_ELEMENTS`; this flag drives the UI hint so the user
    /// understands why points/edges are not drawn.
    dense_mesh: bool,

    // ── element selection (view/interaction state, not model) ──
    /// Currently selected element (vertex / edge / face / none).
    selection: Selection,
    /// CPU mesh retained for picking + highlight projection.
    pick_mesh: Option<Mesh>,
    /// Deduplicated edge list (vertex-index pairs) for the pick mesh.
    pick_edges: Option<Vec<u32>>,
    /// Cursor position at last `CursorMoved` (physical px), for click/drag detection.
    last_cursor: [f64; 2],
    /// Cursor position when the left button was pressed.
    press_cursor: Option<[f64; 2]>,
    /// Set on a left-button release that was a click (not a drag); consumed in `on_frame`.
    pending_pick: bool,

    // ── formula SVG texture cache ──
    formula_cache: formula_cache::FormulaCache,

    // ── async mesh building ──
    /// Background mesh build result (None = idle or building).
    mesh_build_result: Option<Arc<Mutex<Option<MeshBuildResult>>>>,
    /// Human-readable build status for the status bar.
    build_status: String,
    /// Detailed build statistics (verts/tris/timing) for the status bar.
    build_stats: String,
}

/// Cloneable snapshot of the App state needed for mesh building.
/// This is sent to a background thread to avoid blocking the UI.
#[derive(Clone)]
struct AppBuildSnapshot {
    source: SurfaceType,
    tpms_family: Option<TpmsFamily>,
    clip_to_unit_ball: bool,
    clip_radius: f32,
    sphere_radius: f32,
    torus_major_r: f32,
    torus_minor_r: f32,
    tpms_period: f32,
    tpms_cells: [u32; 3],
    tpms_thickness: f32,
    /// 体积分数 φ ∈ [0,1]：0.5=对称极小曲面，→0/1=完全填充。
    /// 内部通过二分查找求解对应的 C 值（f = C 的等值面）。
    tpms_vol_frac: f32,
    /// 三轴单胞边长 [Lx, Ly, Lz]，默认 [1,1,1]。
    tpms_cell_size: [f32; 3],
    /// 三轴幅值缩放系数 [a, b, c]，默认 [1,1,1]。
    tpms_amplitude: [f32; 3],
    /// 全局等值面偏移 C，默认 0（高级模式）。
    tpms_offset: f32,
    rotation_axis: [f32; 3],
    rotation_angle_deg: f32,
    blend_secondary: Option<String>,
    blend_weight_field: GradientField,
    offset_field: GradientField,
    morphology: Morphology,
    mesh_backend: MeshBackend,
    surf_depth: u8,
    mc_res: usize,
    show_ms_loops: bool,
    show_bounding: bool,
    simplify_enabled: bool,
    simplify_ratio: f32,
    show_surface_edges: bool,
    surface_color: [f32; 3],
    surface_metallic: f32,
    surface_roughness: f32,
    wireframe_color: [f32; 3],
    wireframe_line_width: f32,
}

impl AppBuildSnapshot {
    fn from_app(app: &App) -> Self {
        Self {
            source: app.source.clone(),
            tpms_family: app.tpms_family,
            clip_to_unit_ball: app.clip_to_unit_ball,
            clip_radius: app.clip_radius,
            sphere_radius: app.sphere_radius,
            torus_major_r: app.torus_major_r,
            torus_minor_r: app.torus_minor_r,
            tpms_period: app.tpms_period,
            tpms_cells: app.tpms_cells,
            tpms_thickness: app.tpms_thickness,
            tpms_vol_frac: app.tpms_vol_frac,
            tpms_cell_size: app.tpms_cell_size,
            tpms_amplitude: app.tpms_amplitude,
            tpms_offset: app.tpms_offset,
            rotation_axis: app.rotation_axis,
            rotation_angle_deg: app.rotation_angle_deg,
            blend_secondary: app.blend_secondary.clone(),
            blend_weight_field: app.blend_weight_field,
            offset_field: app.offset_field,
            morphology: app.morphology,
            mesh_backend: app.mesh_backend,
            surf_depth: app.surf_depth,
            mc_res: app.mc_res,
            show_ms_loops: app.show_ms_loops,
            show_bounding: app.show_bounding,
            simplify_enabled: app.simplify_enabled,
            simplify_ratio: app.simplify_ratio,
            show_surface_edges: app.show_surface_edges,
            surface_color: app.surface_color,
            surface_metallic: app.surface_metallic,
            surface_roughness: app.surface_roughness,
            wireframe_color: app.wireframe_color,
            wireframe_line_width: app.wireframe_line_width,
        }
    }

    fn surface_label(&self) -> String {
        if let Some(f) = self.tpms_family {
            f.label().to_string()
        } else {
            self.source.label().to_string()
        }
    }

    /// Assemble a `TpmsSurface` from the raw UI fields for the active family.
    fn make_tpms_surface(&self, family: TpmsFamily) -> TpmsSurface {
        TpmsSurface {
            family,
            period: self.tpms_period,
            cells: self.tpms_cells,
            cell_size: self.tpms_cell_size,
            amplitude: self.tpms_amplitude,
            offset: self.tpms_offset,
            vol_frac: self.tpms_vol_frac,
            thickness: self.tpms_thickness,
            blend_secondary: self.blend_secondary.clone(),
            blend_weight_field: self.blend_weight_field,
            offset_field: self.offset_field,
        }
    }

    /// Primitive-only tree params (TPMS is built via `TpmsSurface`).
    fn tree_params(&self) -> TreeParams<'_> {
        TreeParams {
            name: self.source.name(),
            sphere_radius: self.sphere_radius,
            torus_major_r: self.torus_major_r,
            torus_minor_r: self.torus_minor_r,
        }
    }

    fn current_tree(&self) -> Result<fidget_core::context::Tree, String> {
        if let Some(family) = self.tpms_family {
            let tpms = self.make_tpms_surface(family);
            Ok(tpms.build_tree(self.rotation_axis, self.rotation_angle_deg.to_radians()))
        } else if let SurfaceType::Custom(expr) = &self.source {
            build_tree_from_rhai(expr)
        } else {
            let p = self.tree_params();
            Ok(build_tree(&p))
        }
    }

    /// Compute the scene, topology, and build status as a detached result.
    fn build_scene_result(&self) -> MeshBuildResult {
        use fidget_core::shape::Shape;
        use fidget_jit::JitFunction;

        let tree = match self.current_tree() {
            Ok(t) => t,
            Err(e) => {
                return MeshBuildResult {
                    scene: Scene::default(),
                    topology: None,
                    mesh: None,
                    build_ok: false,
                    error: Some(e),
                    c_value: 0.0,
                    build_stats: String::new(),
                    simplify_stats: String::new(),
                };
            }
        };

        let label = self.surface_label();

        let tpms = self.tpms_family.map(|f| self.make_tpms_surface(f));

        let shell_grad = tpms.as_ref().map(|t| t.shell_grad()).unwrap_or(1.0);
        let shell_half_t = if matches!(self.morphology, Morphology::Shell) {
            0.5 * tpms.as_ref().map(|t| t.thickness).unwrap_or(0.0) * shell_grad
        } else {
            0.0
        };

        // ── 体积分数 → C 值求解 ─────────────────────────────
        // 用户设置体积分数 φ，内部二分查找 C 使 vol_frac(C) = φ。
        // vol_frac(C) = |{p : f(p) < C}| / |domain|，是 C 的单调递增函数。
        // 仅 Skeletal 模式使用；MinimalSurface 固定 f=0，与 C 无关。
        let c_value = if matches!(self.morphology, Morphology::Skeletal) {
            tpms.as_ref().map(|t| t.solve_c(&tree) as f32).unwrap_or(0.0)
        } else {
            0.0
        };

        let tree = match self.morphology {
            // Minimal surface: the classic zero level-set f = 0.
            Morphology::MinimalSurface => tree,
            Morphology::Shell => tree,
            Morphology::Skeletal => {
                tree.clone() - c_value
            }
        };
        let effective_res = {
            let mut min_feature = match &tpms {
                Some(t) => t.min_feature(),
                None => if let SurfaceType::Torus = self.source {
                    2.0 * self.torus_minor_r
                } else {
                    0.5
                },
            };
            if matches!(self.morphology, Morphology::Shell) {
                let wall = tpms.as_ref().map(|t| t.thickness).unwrap_or(0.0);
                if wall < min_feature {
                    min_feature = wall;
                }
            }
            let coeff = if matches!(self.morphology, Morphology::Shell) { 10.0 } else { 6.0 };
            let mut needed = ((coeff / min_feature) as usize).max(self.mc_res).min(512);
            if matches!(self.morphology, Morphology::Shell) {
                needed = needed.max(96);
            } else if matches!(self.morphology, Morphology::Skeletal) {
                needed = needed.max(64);
            }
            needed
        };
        // Domain extent: for TPMS, per-axis cell counts; otherwise unit cube.
        let domain_extent = tpms
            .as_ref()
            .map(|t| t.domain_extent())
            .unwrap_or([1.0, 1.0, 1.0]);
        let (mut mesh, mut build_stats) = if matches!(self.morphology, Morphology::Shell) {
            build_shell_mesh(
                tree.clone(), shell_half_t, &label, effective_res,
                self.clip_to_unit_ball, self.clip_radius, domain_extent,
            )
        } else {
            build_mesh(
                tree.clone(), &label,
                self.mesh_backend, self.surf_depth, effective_res,
                self.clip_to_unit_ball, self.clip_radius,
                self.morphology, domain_extent,
            )
        };

        // ── Optional mesh simplification (QEM decimation) ──
        let mut simplify_stats = String::new();
        if self.simplify_enabled && self.simplify_ratio < 1.0 {
            let result = mesh.simplify(self.simplify_ratio, 0.01);
            let stats_line = format!(
                "{} -> {} tris (error {:.4})",
                result.triangles_before, result.triangles_after, result.error
            );
            simplify_stats = stats_line.clone();
            build_stats.push_str(&format!("  |  simplified: {}", stats_line));
        }

        let topology = Some(compute_topology(&mesh));

        let mat = PbrMaterial {
            name: "Surface".into(),
            albedo: [self.surface_color[0], self.surface_color[1], self.surface_color[2], 1.0],
            metallic: self.surface_metallic,
            roughness: self.surface_roughness,
            ..Default::default()
        };
        let pick_mesh = mesh.clone();
        let mut scene = Scene::single_mesh(&label, mesh, mat);
        if let Some(n) = scene.nodes.first_mut() {
            n.render_edges = self.show_surface_edges;
        }

        if self.show_ms_loops {
            let shape = Shape::<JitFunction>::from(tree);
            let ms_mesh = build_ms_loops_mesh(&shape, 64);
            let ms_mat = PbrMaterial {
                name: "MS-loops".into(),
                albedo: [1.0, 0.85, 0.2, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                ..Default::default()
            };
            let mi = scene.meshes.len();
            scene.meshes.push(ms_mesh);
            scene.materials.push(ms_mat);
            scene.nodes.push(SceneNode {
                name: "ms-loops".into(),
                local_transform: glam::Affine3A::IDENTITY,
                mesh_index: Some(mi),
                children: Vec::new(),
                visible: true,
                render_surface: true,
                render_edges: false,
                edge_color_override: None,
                edge_width_override: None,
            });
        }

        if self.show_bounding {
            let wf_mesh = if self.clip_to_unit_ball {
                build_sphere_wireframe(self.clip_radius, 12, 24)
                } else {
                let extent = tpms
                    .as_ref()
                    .map(|t| t.cells_f32())
                    .unwrap_or([1.0, 1.0, 1.0]);
                build_box_wireframe(extent)
            };
            let wf_mat = PbrMaterial { name: "wireframe".into(), ..Default::default() };
            let wi = scene.meshes.len();
            scene.meshes.push(wf_mesh);
            scene.materials.push(wf_mat);
            scene.nodes.push(SceneNode {
                name: "wireframe".into(),
                local_transform: glam::Affine3A::IDENTITY,
                mesh_index: Some(wi),
                children: Vec::new(),
                visible: true,
                render_surface: false,
                render_edges: true,
                edge_color_override: Some(self.wireframe_color),
                edge_width_override: Some(self.wireframe_line_width),
            });
        }

        MeshBuildResult {
            scene,
            topology,
            mesh: Some(pick_mesh),
            build_ok: true,
            error: None,
            c_value,
            build_stats,
            simplify_stats,
        }
    }
}

/// Result of an async mesh build.
struct MeshBuildResult {
    scene: Scene,
    topology: Option<engvis_core::topology::MeshTopology>,
    /// Retained CPU mesh for picking / highlight projection (if available).
    mesh: Option<Mesh>,
    build_ok: bool,
    error: Option<String>,
    /// 由体积分数 φ 求解得到的 C 值（仅 MinimalSurface/Skeletal 有意义）。
    c_value: f32,
    /// Human-readable build statistics for the status bar.
    build_stats: String,
    /// Simplification stats string for the mesh panel.
    simplify_stats: String,
}

impl App {
    fn surface_label(&self) -> String {
        self.source.label().to_string()
    }

    /// Build the scene synchronously (used for initial setup and fallback).
    fn build_scene_sync(&mut self) -> Scene {
        let snapshot = AppBuildSnapshot::from_app(self);
        let result = snapshot.build_scene_result();
        self.last_topology = result.topology;
        self.last_build_ok = result.build_ok;
        self.cached_c_value = result.c_value;
        self.pick_mesh = result.mesh.clone();
        self.pick_edges = result.mesh.as_ref().map(|m| m.extract_edge_indices());
        if let Some(e) = &result.error {
            self.custom_error = Some(e.clone());
        }
        result.scene
    }

    /// Replace the current scene with a single imported mesh.
    fn load_external_mesh(&mut self, path: &std::path::Path) -> Result<Scene, String> {
        let mesh = mesh_io::load_mesh(path)?;
        // Retain the CPU mesh so element picking works on imported geometry.
        let pick_mesh = mesh.clone();
        self.pick_edges = Some(pick_mesh.extract_edge_indices());
        self.pick_mesh = Some(pick_mesh);
        self.last_topology = Some(compute_topology(&mesh));
        self.last_build_ok = true;
        let aabb = mesh.aabb;
        let mat = PbrMaterial {
            name: "Imported".into(),
            albedo: [0.7, 0.7, 0.75, 1.0],
            metallic: 0.0,
            roughness: 0.6,
            ..Default::default()
        };
        let scene = Scene::single_mesh(
            path.file_name().and_then(|s| s.to_str()).unwrap_or("imported"),
            mesh, mat,
        );
        let _ = aabb; // (camera fit happens in caller via scene_dirty path)
        let mut scene = scene;
        if let Some(n) = scene.nodes.first_mut() {
            n.render_edges = self.show_surface_edges;
        }
        Ok(scene)
    }
}

impl EngvisApp for App {
    fn config(&self) -> RunConfig {
        RunConfig {
            title: "engvis — Engineering Visualization".into(),
            width: 1280, height: 800,
            ..Default::default()
        }
    }

    fn on_setup(&mut self, _ctx: &mut AppCtx) -> Scene {
        // Initial build is synchronous (fast for default low resolution).
        self.build_scene_sync()
    }

    fn on_ready(&mut self, scene: &Scene, camera: &mut OrbitCamera) {
        camera.fit_to_scene(scene);
        self.camera_fitted = true;
    }

    fn ui(&mut self, egui_ctx: &egui::Context, frame: &mut FrameCtx) {
        // ── Check for completed async mesh build ───────────────────
        if let Some(result_arc) = &self.mesh_build_result {
            let done = {
                let guard = result_arc.lock().unwrap();
                guard.is_some()
            };
            if done {
                let result = {
                    let mut guard = result_arc.lock().unwrap();
                    guard.take().unwrap()
                };
                self.mesh_build_result = None;
                self.last_topology = result.topology;
                self.last_build_ok = result.build_ok;
                self.cached_c_value = result.c_value;
                if let Some(e) = &result.error {
                    self.custom_error = Some(e.clone());
                    self.build_status = format!("build failed: {e}");
                } else {
                    self.custom_error = None;
                    self.build_status = "ready".into();
                self.build_stats = result.build_stats;
                self.simplify_stats = result.simplify_stats;
            }
            self.pick_mesh = result.mesh.clone();
            self.pick_edges = result.mesh.as_ref().map(|m| m.extract_edge_indices());
            // Dense-mesh guard: instanced point/edge overlays over a very large
            // mesh (hundreds of thousands of elements) combined with 4x MSAA and
            // FXAA overdraw can stall the GPU long enough to freeze the window.
            // The renderer caps overlays per node above `MAX_OVERLAY_ELEMENTS`;
            // we record the state here so the UI can explain why points/edges
            // are not shown, and surface a status hint.
            self.dense_mesh = result
                .mesh
                .as_ref()
                .map(|m| m.vertices.len() > DENSE_MESH_VERTS)
                .unwrap_or(false);
            if self.dense_mesh {
                let nv = result.mesh.as_ref().map(|m| m.vertices.len()).unwrap_or(0);
                self.build_status =
                    format!("ready · overlays capped ({nv} verts too dense)");
            }
            *frame.scene = result.scene;
                // Fit camera to new scene bounds.
                frame.camera.fit_to_scene(frame.scene);
                *frame.scene_dirty = true;
            }
        }

        // ── Launch async mesh build when requested ─────────────────
        if self.needs_remesh && self.mesh_build_result.is_none() {
            self.needs_remesh = false;
            self.build_status = "building...".into();
            // Clone the app state needed for the build.
            let app_snapshot = AppBuildSnapshot::from_app(self);
            let result_slot = Arc::new(Mutex::new(None));
            self.mesh_build_result = Some(result_slot.clone());
            let slot = result_slot.clone();
            thread::spawn(move || {
                let result = app_snapshot.build_scene_result();
                let mut guard = slot.lock().unwrap();
                *guard = Some(result);
            });
            // Request continuous repaint while building.
            egui_ctx.request_repaint();
        }

        // While building, keep requesting repaints to check completion.
        if self.mesh_build_result.is_some() {
            egui_ctx.request_repaint();
        }
        if let Some(path) = self.pending_load.take() {
            // glTF goes through the renderer's loader (textures, nodes);
            // OBJ/STL/PLY go through mesh_io.
            let ext = path.extension().and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase()).unwrap_or_default();
            match ext.as_str() {
                "gltf" | "glb" => {
                    match load_gltf(path.to_string_lossy().as_ref(),
                        frame.device, frame.queue, frame.texture_cache) {
                        Ok((scene, aabb)) => {
                            *frame.scene = scene;
                            frame.camera.fit_to_aabb(aabb);
                            *frame.scene_dirty = true;
                            // Glb may contain multiple meshes; topology stats are
                            // not aggregated here.  Clear pick mesh (no single
                            // CPU mesh to pick against).
                            self.last_topology = None;
                            self.pick_mesh = None;
                            self.pick_edges = None;
                            self.selection = Selection::None;
                        }
                        Err(e) => eprintln!("glTF load failed: {e}"),
                    }
                }
                _ => {
                    match self.load_external_mesh(&path) {
                        Ok(scene) => {
                            let aabb = scene.meshes.iter().fold(
                                    Aabb::empty(),
                                    |mut a, m| { a.expand(glam::Vec3::from(m.aabb.min));
                                                 a.expand(glam::Vec3::from(m.aabb.max)); a }
                                );
                            *frame.scene = scene;
                            frame.camera.fit_to_aabb(aabb);
                            *frame.scene_dirty = true;
                        }
                        Err(e) => eprintln!("mesh load failed: {e}"),
                    }
                }
            }
        }
        if let Some(path) = self.pending_save.take() {
            // Save the first non-wireframe mesh in the scene.
            if let Some(mesh) = frame.scene.meshes.first() {
                if let Err(e) = mesh_io::save_mesh(mesh, &path) {
                    eprintln!("mesh save failed: {e}");
                } else {
                    eprintln!("saved {} ({} verts, {} tris)",
                        path.display(), mesh.vertices.len(), mesh.indices.len()/3);
                }
            }
        }

        // ── Edge overlay enabled so the bounding box / sphere shows up;
        //    only nodes with `render_edges=true` are affected.  Edge color
        //    and line width come from the App (controlled in the
        //    Display panel) so the user's choice persists across remeshes.
        // Edge overlay enabled so the bounding box / sphere shows up;
        // only nodes with `render_edges=true` are affected.  The App owns
        // the `RenderState` as the single source of truth and pushes it to
        // the renderer each frame via `set_render_state`.
        self.render_state.edge_opts.enabled = true;
        frame.set_render_state(self.render_state);
        if !self.camera_fitted {
            frame.camera.fit_to_scene(frame.scene);
            self.camera_fitted = true;
        }

        // ── Top menu bar ─────────────────────────────────────────
        egui::TopBottomPanel::top("menu_bar").show(egui_ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open mesh... (OBJ/STL/PLY/glTF)").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Mesh", &["obj", "stl", "ply", "gltf", "glb"])
                            .pick_file() { self.pending_load = Some(p); }
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Save current mesh as OBJ...").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("OBJ", &["obj"])
                            .set_file_name("mesh.obj").save_file()
                        { self.pending_save = Some(p); }
                        ui.close();
                    }
                    if ui.button("Save current mesh as STL...").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("STL", &["stl"])
                            .set_file_name("mesh.stl").save_file()
                        { self.pending_save = Some(p); }
                        ui.close();
                    }
                    if ui.button("Save current mesh as PLY...").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("PLY", &["ply"])
                            .set_file_name("mesh.ply").save_file()
                        { self.pending_save = Some(p); }
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        egui_ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.render_state.show_surface, "Surface");
                    ui.checkbox(&mut self.render_state.show_grid, "Grid");
                    ui.checkbox(&mut self.render_state.vertex_opts.enabled, "Points");
                    ui.checkbox(&mut self.show_bounding, "Bounding wireframe")
                        .on_hover_text("Re-mesh to apply");
                    ui.checkbox(&mut self.show_ms_loops, "MS boundary loops")
                        .on_hover_text("Re-mesh to apply");
                });
                ui.menu_button("Help", |ui| {
                    ui.label("engvis - Engineering Visualization");
                    ui.label("Implicit surfaces (Fidget) + DC / MC33 meshing.");
                    ui.label("Custom expression syntax: Rhai (x, y, z, sin, cos, ...).");
                });
            });
        });

        // ── Bottom status bar ─────────────────────────────────────
        egui::TopBottomPanel::bottom("status_bar").show(egui_ctx, |ui| {
            ui.horizontal(|ui| {
                // Build status indicator
                if self.mesh_build_result.is_some() {
                    ui.colored_label(egui::Color32::from_rgb(255, 200, 0),
                        "Building mesh...");
                } else if self.last_build_ok {
                    if let Some(t) = &self.last_topology {
                        ui.label(format!(
                            "V={}  E={}  F={}  chi={}  dE={}  comps={}  watertight={}",
                            t.vertices, t.edges, t.faces, t.euler,
                            t.boundary_edges, t.connected_components, t.is_watertight,
                        ));
                    } else {
                        ui.label("no topology");
                    }
                } else {
                    ui.colored_label(egui::Color32::from_rgb(220, 80, 80),
                        "build failed (see expression panel)");
                }
                if !self.build_stats.is_empty() {
                    ui.separator();
                    ui.label(&self.build_stats);
                }
                ui.separator();
                ui.label(format!("FPS {:.0}", frame.fps));
                ui.separator();
                ui.label(format!("backend: {}", match self.mesh_backend {
                    MeshBackend::DualContouring  => "DC",
                    MeshBackend::MarchingCubes33 => "MC33",
                }));
                ui.separator();
                ui.label(format!("surface: {}", self.surface_label()));
                ui.separator();
                ui.label(format!("mode: {}", match self.morphology {
                    Morphology::MinimalSurface => "minimal",
                    Morphology::Shell => "shell",
                    Morphology::Skeletal => "skeletal",
                }));
            });
        });

        // ── Left "tabs" panel (vertical navigation) ────────────────
        egui::SidePanel::left("tabs")
            .resizable(true).default_width(170.0)
            .show(egui_ctx, |ui| {
                ui.heading("engvis");
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Parameters").strong());
                ui.add_space(4.0);
                let tabs = [
                    (Tab::Surface, "Surface",   "TPMS type & formula"),
                    (Tab::Cell,    "Cell",      "Period, cells, L, amp"),
                    (Tab::Deform,  "Deform",    "Rotation, blend, grad"),
                    (Tab::Morph,   "Morph",     "Shell / skeletal"),
                    (Tab::Mesh,    "Mesh",      "Backend & simplify"),
                    (Tab::Display, "Display",   "Colors, PBR, edges"),
                    (Tab::Topo,    "Topo",      "Euler & manifold"),
                ];
                for (t, label, hint) in tabs {
                    let sel = self.current_tab == t;
                    let resp = ui.selectable_label(sel, label);
                    if sel {
                        ui.indent((t, "hint"), |ui| {
                            ui.label(egui::RichText::new(hint)
                                        .small().color(ui.visuals().weak_text_color()));
                        });
                    }
                    if resp.clicked() {
                        self.current_tab = t;
                    }
                }
            });

        // ── Right "details" panel ──────────────────────────────────
        egui::SidePanel::right("details")
            .resizable(true).default_width(340.0)
            .show(egui_ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        match self.current_tab {
                            Tab::Surface => self.ui_surface(ui, egui_ctx),
                            Tab::Cell    => self.ui_cell(ui, egui_ctx),
                            Tab::Deform  => self.ui_deform(ui, egui_ctx),
                            Tab::Morph   => self.ui_morph(ui, egui_ctx),
                            Tab::Mesh    => self.ui_mesh(ui),
                            Tab::Display => self.ui_display(ui),
                            Tab::Topo    => self.ui_topology(ui),
                        }
                    });
            });

        // ── Selection visual feedback (egui overlay) ──
        self.draw_selection_overlay(egui_ctx, frame);
    }

    fn on_frame(&mut self, frame: &mut FrameCtx) {
        if self.pending_pick {
            self.pending_pick = false;
            // Don't pick when the click landed on an egui widget (slider, etc.).
            if !frame.egui_wants_pointer {
                if let Some(mesh) = &self.pick_mesh {
                    let cursor = [frame.cursor_x, frame.cursor_y];
                    self.selection =
                        mesh.pick(frame.camera, cursor, frame.viewport, 8.0);
                }
            }
        }
    }

    fn on_event(&mut self, event: &winit::event::WindowEvent) -> EventHandling {
        use winit::event::{ElementState, MouseButton};
        match event {
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                self.last_cursor = [position.x, position.y];
            }
            winit::event::WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: ElementState::Pressed,
                ..
            } => {
                self.press_cursor = Some(self.last_cursor);
            }
            winit::event::WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: ElementState::Released,
                ..
            } => {
                // Treat as a click (not a drag/orbit) only if the cursor
                // barely moved between press and release.
                if let Some(press) = self.press_cursor {
                    let dx = self.last_cursor[0] - press[0];
                    let dy = self.last_cursor[1] - press[1];
                    if (dx * dx + dy * dy).sqrt() < 4.0 {
                        self.pending_pick = true;
                    }
                }
                self.press_cursor = None;
            }
            _ => {}
        }
        EventHandling::Default
    }
}

impl App {
    /// Draw selection visual feedback as an egui overlay anchored to the 3D
    /// elements.  The marker is re-projected every frame so it tracks the
    /// model as the camera orbits.  Drawn on the background layer so the side
    /// panels stay on top of it.
    fn draw_selection_overlay(&self, egui_ctx: &egui::Context, frame: &FrameCtx) {
        let mesh = match &self.pick_mesh {
            Some(m) => m,
            None => return,
        };
        let edges = self.pick_edges.as_deref();
        let ppp = egui_ctx.pixels_per_point();
        let painter = egui_ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("engvis_selection"),
        ));
        let proj = |w: [f32; 3]| -> Option<egui::Pos2> {
            project_to_pixel(w, &frame.camera.view_projection(), frame.viewport)
                .map(|p| egui::pos2((p[0] / ppp as f64) as f32, (p[1] / ppp as f64) as f32))
        };
        let accent = egui::Color32::from_rgb(255, 153, 0);
        match self.selection {
            Selection::None => {}
            Selection::Vertex(i) => {
                let i = i as usize;
                if i >= mesh.vertices.len() {
                    return;
                }
                let p = match proj(mesh.vertices[i].position) {
                    Some(p) => p,
                    None => return,
                };
                painter.circle_filled(p, 9.0, accent);
                painter.circle_filled(p, 4.0, egui::Color32::WHITE);
                painter.text(
                    p + egui::vec2(12.0, -10.0),
                    egui::Align2::LEFT_CENTER,
                    format!("vertex #{}", i),
                    egui::FontId::default(),
                    accent,
                );
            }
            Selection::Edge(e) => {
                let edges = match edges {
                    Some(e) => e,
                    None => return,
                };
                let k = e as usize * 2;
                if k + 1 >= edges.len() {
                    return;
                }
                let a = edges[k] as usize;
                let b = edges[k + 1] as usize;
                if a >= mesh.vertices.len() || b >= mesh.vertices.len() {
                    return;
                }
                let pa = match proj(mesh.vertices[a].position) {
                    Some(p) => p,
                    None => return,
                };
                let pb = match proj(mesh.vertices[b].position) {
                    Some(p) => p,
                    None => return,
                };
                painter.line_segment([pa, pb], (3.0, accent));
                let mid = egui::pos2((pa.x + pb.x) * 0.5, (pa.y + pb.y) * 0.5);
                painter.text(
                    mid + egui::vec2(0.0, -10.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("edge #{}", e),
                    egui::FontId::default(),
                    accent,
                );
            }
            Selection::Face(f) => {
                let f = f as usize * 3;
                if f + 2 >= mesh.indices.len() {
                    return;
                }
                let p0 = match proj(mesh.vertices[mesh.indices[f] as usize].position) {
                    Some(p) => p,
                    None => return,
                };
                let p1 =
                    match proj(mesh.vertices[mesh.indices[f + 1] as usize].position) {
                        Some(p) => p,
                        None => return,
                    };
                let p2 =
                    match proj(mesh.vertices[mesh.indices[f + 2] as usize].position) {
                        Some(p) => p,
                        None => return,
                    };
                painter.add(egui::Shape::convex_polygon(
                    vec![p0, p1, p2],
                    egui::Color32::from_rgba_unmultiplied(255, 153, 0, 40),
                    (2.0, accent),
                ));
                let cx = (p0.x + p1.x + p2.x) / 3.0;
                let cy = (p0.y + p1.y + p2.y) / 3.0;
                painter.text(
                    egui::pos2(cx, cy),
                    egui::Align2::CENTER_CENTER,
                    format!("face #{}", f / 3),
                    egui::FontId::default(),
                    accent,
                );
            }
        }
    }

    /// True when the active source is a built-in TPMS surface.
    fn tpms_active(&self) -> bool {
        self.tpms_family.is_some()
    }

    /// Internal name of the active surface (TPMS family if active, else the
    /// primitive/custom name) — used for formula cards / cache keys.
    fn active_name(&self) -> String {
        self.tpms_family
            .map(|f| f.name().to_string())
            .unwrap_or_else(|| self.source.name().to_string())
    }

    /// 根据 cell 范围自动计算 blend 权重场的 Sigmoid 参数。
    /// 坐标域为 [-extent, +extent]，其中 extent = cells * L。
    /// f1_cells = [a, b] 表示 cell a..=b 区域用 f1，
    /// f2_cells = [c, d] 表示 cell c..=d 区域用 f2，
    /// 中间自动 Sigmoid 过渡。
    fn sync_blend_weight_from_cells(&mut self) {
        let axis_idx = self.blend_axis as usize;
        let n_cells = self.tpms_cells[axis_idx].max(1);
        let cell_size = self.tpms_cell_size[axis_idx].max(1e-6);
        let extent = n_cells as f32 * cell_size;

        // Cell 编号 0-based，映射到坐标空间 [-extent, +extent]
        // cell i 的中心坐标 = -extent + (i + 0.5) * cell_size
        let cell_center = |i: u32| -> f32 {
            -extent + (i as f32 + 0.5) * cell_size
        };

        let f1_end = (self.blend_f1_cells[1]).min(n_cells - 1);
        let f2_start = (self.blend_f2_cells[0]).min(n_cells - 1);

        // Sigmoid 中心 = f1 末尾和 f2 开头的中间位置
        let x0 = (cell_center(f1_end) + cell_center(f2_start)) / 2.0;
        // Sigmoid 跨度 = 两个区域之间的距离（归一化到 cell_size）
        let span = (cell_center(f2_start) - cell_center(f1_end)).abs().max(cell_size);
        // sharpness 使过渡在约 1 个 cell 宽度内完成
        let sharpness = 4.0 * cell_size / span;

        let mut axis = [0.0f32; 3];
        axis[axis_idx] = 1.0;

        self.blend_weight_field = GradientField {
            mode: GradientMode::Sigmoid,
            axis,
            base: 0.0,  // w → 0 at f2 side
            delta: 1.0,  // w → 1 at f1 side
            sharpness: sharpness.max(0.5).min(20.0),
            center: x0,
        };
        self.needs_remesh = true;
    }

    fn ui_surface(&mut self, ui: &mut egui::Ui, egui_ctx: &egui::Context) {
        ui.heading("Surface");
        ui.label("Implicit surface f(x,y,z) = 0.");
        ui.add_space(6.0);

        // ── Primitive shapes ──────────────────────────────
        ui.label("Primitive shapes:");
        for surface in SurfaceType::primitive_surfaces() {
            let selected = self.source == surface;
            if ui.selectable_label(selected, surface.label()).clicked() {
                self.source = surface.clone();
                self.tpms_family = None;
                self.needs_remesh = true;
            }
            if selected {
                ui.indent((surface.name(), "params"), |ui| {
                    match self.source {
                        SurfaceType::Sphere => {
                            if ui.add(egui::Slider::new(&mut self.sphere_radius, 0.1..=3.0)
                                            .text("Radius")).changed() {
                                self.needs_remesh = true;
                            }
                        }
                        SurfaceType::Torus => {
                            if ui.add(egui::Slider::new(&mut self.torus_major_r, 0.1..=3.0)
                                            .text("Major R")).changed() {
                                self.needs_remesh = true;
                            }
                            if ui.add(egui::Slider::new(&mut self.torus_minor_r, 0.02..=1.5)
                                            .text("Minor r")).changed() {
                                self.needs_remesh = true;
                            }
                        }
                        _ => {}
                    }
                });
            }
        }

        ui.add_space(10.0);
        // ── TPMS (dropdown) ───────────────────────────────
        ui.label("Triply Periodic Minimal Surfaces:");
        // ── TPMS (dropdown) ───────────────────────────────
        ui.label("Triply Periodic Minimal Surfaces:");
        let tpms_families = TpmsFamily::all();
        let current_idx = tpms_families
            .iter()
            .position(|f| Some(*f) == self.tpms_family)
            .unwrap_or(0);
        let mut tpms_idx = current_idx;
        egui::ComboBox::from_id_salt("tpms_combo")
            .width(200.0)
            .selected_text(tpms_families[current_idx].label())
            .show_ui(ui, |ui| {
                for (i, family) in tpms_families.iter().enumerate() {
                    ui.selectable_value(&mut tpms_idx, i, family.label());
                }
            });
        if tpms_idx != current_idx {
            let family = tpms_families[tpms_idx];
            self.tpms_family = Some(family);
            // Adopt this family's default period/cell parameters (matches the
            // prior behaviour of `default_params`).  Volume fraction,
            // thickness, and blend settings are preserved.
            let d = TpmsSurface::new(family);
            self.tpms_period = d.period;
            self.tpms_cells = d.cells;
            self.tpms_cell_size = d.cell_size;
            self.tpms_amplitude = d.amplitude;
            self.tpms_offset = d.offset;
            self.needs_remesh = true;
        }

        // ── Formula card ─────────────────────────────────
        if self.tpms_active() {
            let name = self.active_name();
            ui.add_space(4.0);
            let frame_fill = ui.visuals().widgets.noninteractive.bg_fill;
            egui::Frame::new()
                .fill(frame_fill)
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    if let Some(tex) = self.formula_cache.get(egui_ctx, &name) {
                        let size = tex.size_vec2();
                        let max_w = ui.available_width().max(240.0);
                        let scale = (max_w / size.x).min(1.0);
                        ui.image(egui::load::SizedTexture::new(
                            tex.id(),
                            egui::vec2(size.x * scale, size.y * scale),
                        ));
                    } else {
                        ui.code(tpms_formula(&name));
                    }
                });
        }

        ui.add_space(8.0);
        ui.separator();
        // ── Custom expression ─────────────────────────────────
        ui.label(egui::RichText::new("Custom expression (Rhai)").strong());
        ui.horizontal(|ui| {
            let resp = ui.add(egui::TextEdit::singleline(&mut self.custom_expr)
                .code_editor());
            if resp.lost_focus() && resp.ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.source = SurfaceType::Custom(self.custom_expr.clone());
                self.tpms_family = None;
                self.needs_remesh = true;
            }
            if ui.button("Apply").clicked() {
                self.source = SurfaceType::Custom(self.custom_expr.clone());
                self.tpms_family = None;
                self.needs_remesh = true;
            }
        });
        if let Some(err) = &self.custom_error {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
        }
        ui.add_space(2.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("e.g.").small().color(ui.visuals().weak_text_color()));
            if ui.link("sphere").clicked() {
                self.custom_expr = "x*x + y*y + z*z - 0.64".into();
                self.source = SurfaceType::Custom(self.custom_expr.clone());
                self.tpms_family = None;
                self.needs_remesh = true;
            }
            ui.label(egui::RichText::new("|").small());
            if ui.link("gyroid").clicked() {
                self.custom_expr = "sin(4*x)*cos(4*y) + sin(4*y)*cos(4*z) + sin(4*z)*cos(4*x)".into();
                self.source = SurfaceType::Custom(self.custom_expr.clone());
                self.tpms_family = None;
                self.needs_remesh = true;
            }
        });

        ui.add_space(8.0);
        ui.separator();
        // ── Clip ──────────────────────────────────────────
        ui.label("Bounding region:");
        if ui.checkbox(&mut self.clip_to_unit_ball, "Clip to ball").changed() {
            self.needs_remesh = true;
        }
        ui.indent("clip_opts", |ui| {
            if ui.add(egui::Slider::new(&mut self.clip_radius, 0.2..=5.0)
                    .text("Clip radius")).changed() {
                self.needs_remesh = true;
            }
        });
    }

    fn ui_cell(&mut self, ui: &mut egui::Ui, _egui_ctx: &egui::Context) {
        ui.heading("Cell");
        if !self.tpms_active() {
            ui.label("Select a TPMS surface first (Surface tab).");
            return;
        }
        ui.label("Unit-cell period, array, size & amplitude.");
        ui.add_space(6.0);

            // ── Parameters (Grid-aligned) ─────────────────────
            egui::Grid::new("tpms_params")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Period k");
                    if ui.add(egui::Slider::new(&mut self.tpms_period, 1.0..=10.0)).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();

                    ui.label("Cells x");
                    if ui.add(egui::Slider::new(&mut self.tpms_cells[0], 1..=10)
                        .text("")).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();

                    ui.label("Cells y");
                    if ui.add(egui::Slider::new(&mut self.tpms_cells[1], 1..=10)
                        .text("")).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();

                    ui.label("Cells z");
                    if ui.add(egui::Slider::new(&mut self.tpms_cells[2], 1..=10)
                        .text("")).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();
                });

            // ── Cell size Lx/Ly/Lz ──
            ui.add_space(4.0);
            egui::Grid::new("cell_size_params")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Lx");
                    if ui.add(egui::Slider::new(&mut self.tpms_cell_size[0], 0.25..=4.0)
                        .text("")).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();

                    ui.label("Ly");
                    if ui.add(egui::Slider::new(&mut self.tpms_cell_size[1], 0.25..=4.0)
                        .text("")).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();

                    ui.label("Lz");
                    if ui.add(egui::Slider::new(&mut self.tpms_cell_size[2], 0.25..=4.0)
                        .text("")).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();
                });

            // ── Amplitude a/b/c ──
            ui.add_space(2.0);
            egui::Grid::new("amplitude_params")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Amp a");
                    if ui.add(egui::Slider::new(&mut self.tpms_amplitude[0], 0.1..=3.0)
                        .text("")).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();

                    ui.label("Amp b");
                    if ui.add(egui::Slider::new(&mut self.tpms_amplitude[1], 0.1..=3.0)
                        .text("")).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();

                    ui.label("Amp c");
                    if ui.add(egui::Slider::new(&mut self.tpms_amplitude[2], 0.1..=3.0)
                        .text("")).changed() {
                        self.needs_remesh = true;
                    }
                    ui.end_row();
                });

            // ── Offset C ──
            ui.add_space(2.0);
            if ui.add(egui::Slider::new(&mut self.tpms_offset, -2.0..=2.0)
                .text("Offset C")).changed() {
                self.needs_remesh = true;
            }
            let total = self.tpms_cells[0] * self.tpms_cells[1] * self.tpms_cells[2];
            ui.label(egui::RichText::new(
                format!("{} x {} x {} = {} cells",
                    self.tpms_cells[0], self.tpms_cells[1],
                    self.tpms_cells[2], total))
                .small().color(ui.visuals().weak_text_color()));
    }

    fn ui_deform(&mut self, ui: &mut egui::Ui, egui_ctx: &egui::Context) {
        ui.heading("Deform");
        if !self.tpms_active() {
            ui.label("Select a TPMS surface first (Surface tab).");
            return;
        }
        ui.label("Rotation, dual-TPMS blend & gradient fields.");
        ui.add_space(6.0);

            // ── Topology deformation: rotation ──
            ui.collapsing("Rotation", |ui| {
                egui::Grid::new("rotation_grid")
                    .num_columns(2).spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Axis x");
                        if ui.add(egui::Slider::new(&mut self.rotation_axis[0], -1.0..=1.0)
                            .text("")).changed() { self.needs_remesh = true; }
                        ui.end_row();
                        ui.label("Axis y");
                        if ui.add(egui::Slider::new(&mut self.rotation_axis[1], -1.0..=1.0)
                            .text("")).changed() { self.needs_remesh = true; }
                        ui.end_row();
                        ui.label("Axis z");
                        if ui.add(egui::Slider::new(&mut self.rotation_axis[2], -1.0..=1.0)
                            .text("")).changed() { self.needs_remesh = true; }
                        ui.end_row();
                        ui.label("Angle (deg)");
                        if ui.add(egui::Slider::new(&mut self.rotation_angle_deg, -180.0..=180.0)
                            .text("")).changed() { self.needs_remesh = true; }
                        ui.end_row();
                    });
            });

            // ── Dual-TPMS weighted blend ──
            ui.add_space(4.0);
            ui.collapsing("Dual-TPMS Blend", |ui| {
                let mut enabled = self.blend_secondary.is_some();
                if ui.checkbox(&mut enabled, "Enable blend f = w*f1 + (1-w)*f2").changed() {
                    self.blend_secondary = if enabled {
                        // Default: f1 on left half, f2 on right half, along x
                        let nx = self.tpms_cells[0].max(1);
                        self.blend_axis = 0;
                        self.blend_f1_cells = [0, nx / 2 - 1];
                        self.blend_f2_cells = [nx / 2 + 1, nx - 1];
                        self.sync_blend_weight_from_cells();
                        Some("schwarz-p".to_string())
                    } else {
                        self.blend_weight_field = GradientField::default();
                        None
                    };
                    self.needs_remesh = true;
                }
                    let active = self.active_name();
                    if let Some(ref mut sec) = self.blend_secondary {
                        // ── Show both formulas ──
                        let frame_fill = ui.visuals().widgets.noninteractive.bg_fill;
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("f1 (Primary):").strong());
                        egui::Frame::new()
                            .fill(frame_fill)
                            .corner_radius(egui::CornerRadius::same(4))
                            .inner_margin(egui::Margin::same(6))
                            .show(ui, |ui| {
                                if let Some(tex) = self.formula_cache.get(egui_ctx, &active) {
                                    let size = tex.size_vec2();
                                    let max_w = ui.available_width().max(200.0);
                                    let scale = (max_w / size.x).min(0.8);
                                    ui.image(egui::load::SizedTexture::new(
                                        tex.id(),
                                        egui::vec2(size.x * scale, size.y * scale),
                                    ));
                                } else {
                                    ui.code(tpms_formula(&active));
                                }
                            });

                    ui.add_space(4.0);
                    let current = sec.clone();
                    ui.label(egui::RichText::new("f2 (Secondary):").strong());
                    egui::ComboBox::from_id_salt("blend_secondary_combo")
                        .selected_text(
                            TpmsFamily::all().iter()
                                .find(|s| s.name() == current.as_str())
                                .map(|s| s.label()).unwrap_or("Gyroid"))
                        .show_ui(ui, |ui| {
                            for s in TpmsFamily::all() {
                                if ui.selectable_label(current == s.name(), s.label()).clicked() {
                                    *sec = s.name().to_string();
                                    self.needs_remesh = true;
                                }
                            }
                        });
                    egui::Frame::new()
                        .fill(frame_fill)
                        .corner_radius(egui::CornerRadius::same(4))
                        .inner_margin(egui::Margin::same(6))
                        .show(ui, |ui| {
                            if let Some(tex) = self.formula_cache.get(egui_ctx, sec.as_str()) {
                                let size = tex.size_vec2();
                                let max_w = ui.available_width().max(200.0);
                                let scale = (max_w / size.x).min(0.8);
                                ui.image(egui::load::SizedTexture::new(
                                    tex.id(),
                                    egui::vec2(size.x * scale, size.y * scale),
                                ));
                            } else {
                                ui.code(tpms_formula(sec.as_str()));
                            }
                        });

                    // ── Spatial transition by cell range ──
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Spatial transition:").strong());
                    ui.label(egui::RichText::new(
                        "w=1 ⟶ f1 only  |  w=0 ⟶ f2 only")
                        .small().color(ui.visuals().weak_text_color()));
                    ui.add_space(4.0);

                    let axis_labels = ["X", "Y", "Z"];
                    egui::Grid::new("blend_cell_grid")
                        .num_columns(2).spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Direction");
                            let mut ax = self.blend_axis;
                            egui::ComboBox::from_id_salt("blend_axis_combo")
                                .selected_text(axis_labels[ax as usize])
                                .show_ui(ui, |ui| {
                                    for (i, lbl) in axis_labels.iter().enumerate() {
                                        ui.selectable_value(&mut ax, i as u32, *lbl);
                                    }
                                });
                            if ax != self.blend_axis {
                                self.blend_axis = ax;
                                self.sync_blend_weight_from_cells();
                            }
                            ui.end_row();

                            let n = self.tpms_cells[self.blend_axis as usize].max(1);
                            ui.label("f1 cells");
                            let mut f1s = self.blend_f1_cells[0].min(n - 1);
                            let mut f1e = self.blend_f1_cells[1].min(n - 1);
                            let resp1 = ui.horizontal(|ui| {
                                ui.label("[");
                                let r1 = ui.add(egui::DragValue::new(&mut f1s).range(0..=n-1));
                                ui.label(",");
                                let r2 = ui.add(egui::DragValue::new(&mut f1e).range(0..=n-1));
                                ui.label("]");
                                r1.union(r2)
                            }).inner;
                            if resp1.changed() {
                                self.blend_f1_cells = [f1s, f1e];
                                self.sync_blend_weight_from_cells();
                            }
                            ui.end_row();

                            ui.label("f2 cells");
                            let mut f2s = self.blend_f2_cells[0].min(n - 1);
                            let mut f2e = self.blend_f2_cells[1].min(n - 1);
                            let resp2 = ui.horizontal(|ui| {
                                ui.label("[");
                                let r1 = ui.add(egui::DragValue::new(&mut f2s).range(0..=n-1));
                                ui.label(",");
                                let r2 = ui.add(egui::DragValue::new(&mut f2e).range(0..=n-1));
                                ui.label("]");
                                r1.union(r2)
                            }).inner;
                            if resp2.changed() {
                                self.blend_f2_cells = [f2s, f2e];
                                self.sync_blend_weight_from_cells();
                            }
                            ui.end_row();
                        });

                    ui.add_space(4.0);
                    ui.collapsing("Advanced: manual weight field", |ui| {
                        gradient_field_ui(ui, &mut self.blend_weight_field,
                                          &mut self.needs_remesh, "blend_w");
                    });
                }
            });

            // ── Offset C as spatial gradient field ──
            ui.add_space(4.0);
            ui.collapsing("Offset C(x,y,z) gradient", |ui| {
                ui.label("Replaces scalar Offset C (Cell tab) when mode != None.");
                gradient_field_ui(ui, &mut self.offset_field,
                                  &mut self.needs_remesh, "offset_c");
            });
    }

    fn ui_morph(&mut self, ui: &mut egui::Ui, egui_ctx: &egui::Context) {
        ui.heading("Morph");
        if !self.tpms_active() {
            ui.label("Select a TPMS surface first (Surface tab).");
            return;
        }
        ui.label("Minimal surface / Shell / Skeletal.");
        ui.add_space(6.0);

            let mut morph = self.morphology;
            ui.horizontal(|ui| {
                if ui.radio_value(&mut morph, Morphology::MinimalSurface, "Minimal").changed() {
                    self.morphology = morph; self.needs_remesh = true;
                }
                if ui.radio_value(&mut morph, Morphology::Shell, "Shell").changed() {
                    self.morphology = morph; self.needs_remesh = true;
                }
                if ui.radio_value(&mut morph, Morphology::Skeletal, "Skeletal").changed() {
                    self.morphology = morph; self.needs_remesh = true;
                }
            });
            ui.add_space(2.0);

            // Morphology formula + parameters
            ui.indent("morph_detail", |ui| {
                let morph_key = match self.morphology {
                    Morphology::MinimalSurface => "morph-minimal",
                    Morphology::Shell => "morph-shell",
                    Morphology::Skeletal => "morph-skeletal",
                };
                if let Some(tex) = self.formula_cache.get(egui_ctx, morph_key) {
                    let size = tex.size_vec2();
                    let scale = 0.6;
                    ui.image(egui::load::SizedTexture::new(
                        tex.id(),
                        egui::vec2(size.x * scale, size.y * scale),
                    ));
                }
                ui.add_space(2.0);
                match self.morphology {
                    Morphology::Shell => {
                        if ui.add(egui::Slider::new(&mut self.tpms_thickness,
                                    0.02..=0.5).text("Wall t")).changed() {
                            self.needs_remesh = true;
                        }
                    }
                    Morphology::Skeletal => {
                        if ui.add(egui::Slider::new(&mut self.tpms_vol_frac,
                                    0.01..=0.99).text("Vol frac")).changed() {
                            self.needs_remesh = true;
                        }
                        ui.label(egui::RichText::new(
                            format!("C = {:+.4}", self.cached_c_value))
                            .small().color(ui.visuals().weak_text_color()));
                    }
                    Morphology::MinimalSurface => {}
                }
            });
    }

    fn ui_mesh(&mut self, ui: &mut egui::Ui) {
        ui.heading("Mesh");
        ui.label("Polygonisation backend:");
        if ui.selectable_label(self.mesh_backend == MeshBackend::MarchingCubes33,
            "Marching Cubes 33 (smooth boundary)").clicked() {
            self.mesh_backend = MeshBackend::MarchingCubes33;
            self.needs_remesh = true;
        }
        if ui.selectable_label(self.mesh_backend == MeshBackend::DualContouring,
            "Dual Contouring + boundary smoothing").clicked() {
            self.mesh_backend = MeshBackend::DualContouring;
            self.needs_remesh = true;
        }
        ui.add_space(8.0);
        match self.mesh_backend {
            MeshBackend::DualContouring => {
                if ui.add(egui::Slider::new(&mut self.surf_depth, 3..=10)
                    .text("Octree depth")).changed() {
                    self.needs_remesh = true;
                }
            }
            MeshBackend::MarchingCubes33 => {
                if ui.add(egui::Slider::new(&mut self.mc_res, 16..=256)
                    .step_by(8.0).text("Grid resolution")).changed() {
                    self.needs_remesh = true;
                }
            }
        }
        ui.add_space(8.0);
        if ui.button("Re-mesh").clicked() {
            self.needs_remesh = true;
        }

        // ── Mesh simplification ──
        ui.add_space(12.0);
        ui.separator();
        ui.label("Simplification (QEM decimation):");
        if ui.checkbox(&mut self.simplify_enabled, "Enable simplification").changed() {
            self.needs_remesh = true;
        }
        if self.simplify_enabled {
            if ui.add(egui::Slider::new(&mut self.simplify_ratio, 0.01..=1.0)
                .text("Target ratio")
                .show_value(true)).changed() {
                self.needs_remesh = true;
            }
            if !self.simplify_stats.is_empty() {
                ui.label(
                    egui::RichText::new(&self.simplify_stats)
                        .size(11.0)
                        .color(egui::Color32::GRAY),
                );
            }
        }
    }

    fn ui_display(&mut self, ui: &mut egui::Ui) {
        ui.heading("Display");

        // ── Background ─────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Background");
            ui.color_edit_button_rgb(&mut self.render_state.background_color);
        });

        // ── Surface ────────────────────────────────────────
        ui.checkbox(&mut self.render_state.show_surface, "Show triangle surface");
        if self.render_state.show_surface {
            ui.indent("surface_opts", |ui| {
                if ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut self.surface_color)
                }).inner.changed() {
                    self.needs_remesh = true;
                }
                ui.add(egui::Slider::new(&mut self.render_state.opacity, 0.0..=1.0)
                    .text("Opacity"));

                // ── PBR Material ──────────────────────────
                ui.add_space(4.0);
                ui.label("PBR Material:");
                if ui.add(egui::Slider::new(&mut self.surface_metallic, 0.0..=1.0)
                    .text("Metallic"))
                    .on_hover_text("0 = dielectric (plastic/ceramic), 1 = metal")
                    .changed()
                {
                    self.needs_remesh = true;
                }
                if ui.add(egui::Slider::new(&mut self.surface_roughness, 0.0..=1.0)
                    .text("Roughness"))
                    .on_hover_text("0 = mirror-smooth, 1 = fully rough/matte")
                    .changed()
                {
                    self.needs_remesh = true;
                }
                ui.add(egui::Slider::new(&mut self.render_state.env_intensity, 0.0..=3.0)
                    .text("Env intensity"))
                    .on_hover_text("IBL environment light multiplier");
            });
        }

        // ── Edges ──────────────────────────────────────────
        // Triangle-mesh edges of the surface node — toggling this flag
        // requires reapplying `render_edges` on the node, which happens
        // on the next remesh.
        if ui.checkbox(&mut self.show_surface_edges, "Show triangle edges").changed() {
            self.needs_remesh = true;
        }
        if self.show_surface_edges {
            ui.indent("edge_opts", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut self.render_state.edge_opts.color);
                });
                ui.add(egui::Slider::new(&mut self.render_state.edge_opts.line_width, 0.5..=10.0)
                    .text("Line width (px)"));
            });
        }

        // ── Points ──────────────────────────────────────────
        ui.checkbox(&mut self.render_state.vertex_opts.enabled, "Show points");
        if self.dense_mesh {
            ui.label(
                egui::RichText::new(
                    "Mesh is very dense — point/edge overlays are auto-capped by the \
                     renderer to avoid GPU stalls. Lower Grid resolution / Octree depth \
                     to re-enable them.",
                )
                .size(11.0)
                .color(egui::Color32::GOLD),
            );
        }
        if self.render_state.vertex_opts.enabled {
            ui.indent("point_opts", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut self.render_state.vertex_opts.color);
                });
                ui.add(egui::Slider::new(&mut self.render_state.vertex_opts.point_size, 1.0..=12.0)
                    .text("Point size"));
            });
        }

        // ── Other overlays ─────────────────────────────────
        ui.add_space(6.0);
        ui.separator();
        ui.checkbox(&mut self.render_state.show_grid, "World grid");
        if ui.checkbox(&mut self.show_bounding, "Show bounding wireframe").changed() {
            self.needs_remesh = true;
        }
        if self.show_bounding {
            ui.indent("wireframe_opts", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    if ui.color_edit_button_rgb(&mut self.wireframe_color).changed() {
                        self.needs_remesh = true;
                    }
                });
                if ui.add(egui::Slider::new(&mut self.wireframe_line_width, 0.5..=10.0)
                    .text("Line width (px)")).changed() {
                    self.needs_remesh = true;
                }
            });
        }
        if ui.checkbox(&mut self.show_ms_loops, "MS boundary loops").changed() {
            self.needs_remesh = true;
        }
    }

    fn ui_topology(&mut self, ui: &mut egui::Ui) {
        ui.heading("Topology");
        match &self.last_topology {
            None => { ui.label("(no mesh loaded)"); }
            Some(t) => {
                egui::Grid::new("topo_grid").num_columns(2).striped(true).show(ui, |ui| {
                    ui.label("Vertices  V"); ui.label(format!("{}", t.vertices)); ui.end_row();
                    ui.label("Edges     E"); ui.label(format!("{}", t.edges)); ui.end_row();
                    ui.label("Faces     F"); ui.label(format!("{}", t.faces)); ui.end_row();
                    ui.label("Euler     chi = V-E+F"); ui.label(format!("{}", t.euler)); ui.end_row();
                    ui.label("Boundary edges"); ui.label(format!("{}", t.boundary_edges)); ui.end_row();
                    ui.label("Non-manifold edges"); ui.label(format!("{}", t.non_manifold_edges)); ui.end_row();
                    ui.label("Connected components"); ui.label(format!("{}", t.connected_components)); ui.end_row();
                    ui.label("Watertight"); ui.label(format!("{}", t.is_watertight)); ui.end_row();
                });
                ui.add_space(6.0);
                ui.label("chi legend: 2=sphere, 0=torus, -2=double torus, ...");
            }
        }
    }
}

fn main() {
    env_logger::init();

    if std::env::args().any(|a| a == "--selftest") {
        unsafe { std::env::set_var("ENGVIS_TOPO", "1"); }
        // ── TPMS 值域采样 ─────────────────────────────────────
        // 在单位立方体内均匀采样，统计各 TPMS 函数的 [min, max] 值域，
        // 用于判断 UI 中 C value slider 范围是否合理。
        eprintln!("\n[tpms range] sampling value ranges (period=4.0, N=64³):");
        for family in TpmsFamily::all() {
            let name = family.name();
            let mut surf = TpmsSurface::new(family);
            surf.period = 4.0;
            let tree2 = surf.build_tree([0.0, 0.0, 1.0], 0.0);
            use fidget_core::shape::Shape;
            use fidget_jit::JitFunction;
            let shape2 = Shape::<JitFunction>::from(tree2);
            let tape2 = shape2.float_slice_tape(Default::default());
            let mut ev2 = Shape::<JitFunction>::new_float_slice_eval();
            let n = 64;
            let mut xs = Vec::with_capacity(n*n*n);
            let mut ys = Vec::with_capacity(n*n*n);
            let mut zs = Vec::with_capacity(n*n*n);
            for ix in 0..n {
                let x = -1.0 + 2.0 * ix as f32 / (n - 1) as f32;
                for iy in 0..n {
                    let y = -1.0 + 2.0 * iy as f32 / (n - 1) as f32;
                    for iz in 0..n {
                        xs.push(x); ys.push(y); zs.push(-1.0 + 2.0 * iz as f32 / (n - 1) as f32);
                    }
                }
            }
            let vals = ev2.eval(&tape2, &xs, &ys, &zs).unwrap_or_default();
            let (mn, mx) = vals.iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
                    (mn.min(v), mx.max(v))
                });
            // 体积分数：f < C 的比例（C=0 时为负相占比）
            let neg_frac = vals.iter().filter(|&&v| v < 0.0).count() as f32 / vals.len() as f32;
            eprintln!("  {:<16} range=[{:+.3}, {:+.3}]  vol_frac(C=0)={:.3}",
                name, mn, mx, neg_frac);
        }

        // Validate the volume-fraction → C solver: varying φ must shift
        // the zero-set, producing a different mesh.
        let mut surf = TpmsSurface::new(TpmsFamily::Gyroid);
        surf.period = 4.0;
        let tree = surf.build_tree([0.0, 0.0, 1.0], 0.0);
        for phi in [0.3_f32, 0.5, 0.7] {
            let c_val = engvis_tpms::solve_c_for_vol_frac(&tree, phi);
            let field = tree.clone() - c_val;
            let (mesh, _stats) = build_mesh(
                field, "iso-test",
                MeshBackend::MarchingCubes33, 6, 96,
                false, 1.0, Morphology::MinimalSurface, [1.0, 1.0, 1.0],
            );
            // Mean |f - C| over the mesh: should be ≈0 since vertices
            // lie on f = C ⇔ (f − C) = 0.
            use fidget_core::shape::Shape;
            use fidget_jit::JitFunction;
            let shifted = Shape::<JitFunction>::from(tree.clone() - c_val);
            let tape = shifted.float_slice_tape(Default::default());
            let mut ev = Shape::<JitFunction>::new_float_slice_eval();
            let (mut sum, mut n) = (0.0_f64, 0usize);
            for v in &mesh.vertices {
                let pp = v.position;
                let f = ev.eval(&tape, &[pp[0]], &[pp[1]], &[pp[2]])
                    .map(|r| r[0]).unwrap_or(9.9);
                sum += (f as f64).abs(); n += 1;
            }
            eprintln!("φ={:.2} → C={:+.3}: verts={} mean|f-C|={:.5}",
                phi, c_val, mesh.vertices.len(), sum / n.max(1) as f64);
        }

        // Skeletal mesh topology test: MC33 TPMS surface + box CSG cap.
        eprintln!("\n[skeletal selftest] building skeletal mesh (gyroid, res=64)...");
        let t0 = std::time::Instant::now();
        let (sk_mesh, sk_stats) = build_mesh(
            tree.clone(), "skeletal-test",
            MeshBackend::MarchingCubes33, 5, 64,
            false, 1.0, Morphology::Skeletal, [1.0, 1.0, 1.0],
        );
        let sk_dt = t0.elapsed();
        eprintln!("  {sk_stats}");
        let sk_topo = engvis_core::topology::compute_topology(&sk_mesh);
        eprintln!(
            "[skeletal selftest] V={} F={} | χ={} boundary_edges={} components={} watertight={} | {:.0}ms",
            sk_mesh.vertices.len(), sk_mesh.indices.len() / 3,
            sk_topo.euler, sk_topo.boundary_edges,
            sk_topo.connected_components, sk_topo.is_watertight,
            sk_dt.as_secs_f64() * 1e3,
        );

        // Shell mesh topology test: MC33 TPMS shell + box CSG cap.
        eprintln!("\n[shell selftest] building shell mesh (gyroid, res=96)...");
        let shell_half_t = 0.5 * 0.1 * surf.period.max(1.0);
        let t0 = std::time::Instant::now();
        let (sh_mesh, sh_stats) = build_shell_mesh(
            tree.clone(), shell_half_t, "shell-test", 96,
            false, 1.0, [1.0, 1.0, 1.0],
        );
        let sh_dt = t0.elapsed();
        eprintln!("  {sh_stats}");
        eprintln!(
            "[shell selftest] V={} F={} | {:.0}ms",
            sh_mesh.vertices.len(), sh_mesh.indices.len() / 3,
            sh_dt.as_secs_f64() * 1e3,
        );
        return;
    }

    engvis_renderer::run(App {
        source: SurfaceType::Sphere,
        tpms_family: Some(TpmsFamily::Gyroid),
        custom_expr: "sin(4*x)*cos(4*y) + sin(4*y)*cos(4*z) + sin(4*z)*cos(4*x)".to_string(),
        custom_error: None,
        clip_to_unit_ball: false,
        clip_radius: 1.0,
        sphere_radius: 0.8,
        torus_major_r: 0.6,
        torus_minor_r: 0.2,
        tpms_period: 4.0,
        tpms_cells: [1, 1, 1],
        tpms_thickness: 0.1,
        tpms_vol_frac: 0.5,
        cached_c_value: 0.0,
        tpms_cell_size: [1.0, 1.0, 1.0],
        tpms_amplitude: [1.0, 1.0, 1.0],
        tpms_offset: 0.0,
        rotation_axis: [0.0, 0.0, 1.0],
        rotation_angle_deg: 0.0,
        blend_secondary: None,
        blend_weight_field: GradientField::default(),
        blend_axis: 0,
        blend_f1_cells: [0, 0],
        blend_f2_cells: [1, 1],
        offset_field: GradientField::default(),
        morphology: Morphology::MinimalSurface,
        mesh_backend: MeshBackend::MarchingCubes33,
        surf_depth: 7,
        mc_res: 64,
        show_ms_loops: false,
        show_bounding: true,
        simplify_enabled: false,
        simplify_ratio: 0.25,
        simplify_stats: String::new(),
        show_surface_edges: false,
        render_state: RenderState {
            edge_opts: EdgeRenderOptions {
                enabled: true,
                color: [0.35, 0.35, 0.35],
                line_width: 2.0,
            },
            background_color: [1.0, 1.0, 1.0],
            env_intensity: 1.0,
            ..Default::default()
        },
        wireframe_color: [0.3, 0.3, 0.3],
        wireframe_line_width: 2.0,
        surface_color: [0.30, 0.65, 0.90],
        surface_metallic: 0.9,
        surface_roughness: 0.18,
        current_tab: Tab::Surface,
        pending_load: None,
        pending_save: None,
        needs_remesh: false,
        camera_fitted: false,
        last_topology: None,
        last_build_ok: true,
        selection: Selection::None,
        pick_mesh: None,
        pick_edges: None,
        last_cursor: [0.0, 0.0],
        press_cursor: None,
        pending_pick: false,
        formula_cache: formula_cache::FormulaCache::new(),
        mesh_build_result: None,
        build_status: "ready".into(),
        dense_mesh: false,
        build_stats: String::new(),
    });
}
