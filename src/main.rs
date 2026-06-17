// ── engvis viewer ──────────────────────────────────────────
// General-purpose 3D viewer with model loading, scene tree,
// material editing, and lighting controls.

use engvis_core::{
    mesh::create_cube_mesh,
    material::PbrMaterial,
    scene::Scene,
    camera::OrbitCamera,
    Aabb,
};
use engvis_renderer::{
    EngvisApp, AppCtx, FrameCtx, RunConfig, EventHandling, load_gltf,
};

struct App {
    model_path: String,
    pending_load: Option<String>,
    selected_node: Option<usize>,
    scene_aabb: Aabb,
}

impl EngvisApp for App {
    fn config(&self) -> RunConfig {
        RunConfig {
            title: "engvis - Engineering Visualization".into(),
            width: 1600,
            height: 1000,
            ..Default::default()
        }
    }

    fn on_setup(&mut self, _ctx: &mut AppCtx) -> Scene {
        let cube = create_cube_mesh();
        let aabb = cube.aabb;
        self.scene_aabb = aabb;
        self.selected_node = Some(0);

        Scene::single_mesh(
            "Cube",
            cube,
            PbrMaterial {
                name: "Default Cube".into(),
                albedo: [0.7, 0.4, 0.3, 1.0],
                metallic: 0.1,
                roughness: 0.6,
                ..Default::default()
            },
        )
    }

    fn on_ready(&mut self, _scene: &Scene, camera: &mut OrbitCamera) {
        camera.fit_to_aabb(self.scene_aabb);
        camera.target.y = 0.5;
        camera.distance *= 1.4;
    }

    fn ui(&mut self, egui_ctx: &egui::Context, frame: &mut FrameCtx) {
        let scene = &mut *frame.scene;
        let camera = &mut *frame.camera;
        let model_path = &mut self.model_path;
        let pending_load = &mut self.pending_load;
        let selected_node = &mut self.selected_node;
        let scene_aabb = &mut self.scene_aabb;
        let rs = &mut *frame.render_state;
        let fps = frame.fps;

        // Top menu bar
        egui::TopBottomPanel::top("menu_bar").show(egui_ctx, |ui| {
            ui.horizontal(|ui| {
                ui.menu_button("File", |ui| {
                    ui.label("Model path:");
                    ui.text_edit_singleline(model_path);
                    if ui.button("Load glTF").clicked() {
                        if !model_path.is_empty() {
                            *pending_load = Some(model_path.clone());
                        }
                        ui.close();
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Fit All").clicked() {
                        camera.fit_to_aabb(*scene_aabb);
                    }
                    ui.separator();
                    ui.checkbox(&mut rs.show_surface, "Show Surface");
                    ui.checkbox(&mut rs.vertex_opts.enabled, "Show Vertices");
                    ui.checkbox(&mut rs.edge_opts.enabled, "Show Edges");
                    ui.separator();
                    ui.checkbox(&mut rs.show_grid, "Show Grid");
                    ui.separator();
                    ui.add(egui::Slider::new(&mut rs.opacity, 0.05..=1.0).text("Opacity"));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("FPS: {:.0}", fps));
                });
            });
        });

        // Left: Scene panel
        egui::SidePanel::left("scene_panel")
            .default_width(200.0)
            .min_width(150.0)
            .show(egui_ctx, |ui| {
                ui.heading("Scene");
                ui.separator();
                ui.label(format!("Meshes: {}", scene.meshes.len()));
                ui.label(format!("Materials: {}", scene.materials.len()));
                ui.label(format!("Nodes: {}", scene.nodes.len()));
                ui.separator();
                ui.heading("Nodes");
                for (i, node) in scene.nodes.iter().enumerate() {
                    let sel = *selected_node == Some(i);
                    if ui.selectable_label(sel, &node.name).clicked() {
                        *selected_node = Some(i);
                    }
                }
            });

        // Right: Properties panel (scrollable)
        egui::SidePanel::right("properties_panel")
            .default_width(280.0)
            .min_width(220.0)
            .show(egui_ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Properties");
                    ui.separator();

                    egui::CollapsingHeader::new("Surface")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.checkbox(&mut rs.show_surface, "Show Surface");
                            ui.add(
                                egui::Slider::new(&mut rs.opacity, 0.05..=1.0).text("Opacity"),
                            );
                        });

                    egui::CollapsingHeader::new("Edges")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.checkbox(&mut rs.edge_opts.enabled, "Show Edges");
                            ui.label("Color:");
                            egui::color_picker::color_edit_button_rgb(
                                ui, &mut rs.edge_opts.color,
                            );
                            ui.add(
                                egui::Slider::new(&mut rs.edge_opts.line_width, 0.5..=5.0)
                                    .text("Line Width"),
                            );
                        });

                    egui::CollapsingHeader::new("Vertices")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.checkbox(&mut rs.vertex_opts.enabled, "Show Vertices");
                            ui.label("Color:");
                            egui::color_picker::color_edit_button_rgb(
                                ui, &mut rs.vertex_opts.color,
                            );
                            ui.add(
                                egui::Slider::new(&mut rs.vertex_opts.point_size, 1.0..=15.0)
                                    .text("Point Size"),
                            );
                        });

                    ui.separator();
                    ui.checkbox(&mut rs.show_grid, "Show Grid");

                    // Material editing
                    if !scene.materials.is_empty() {
                        let mat = &mut scene.materials[0];
                        ui.separator();
                        ui.heading("Material");
                        ui.label(&mat.name);

                        ui.label("Albedo Color:");
                        {
                            let mut rgb = [mat.albedo[0], mat.albedo[1], mat.albedo[2]];
                            if egui::color_picker::color_edit_button_rgb(ui, &mut rgb).changed() {
                                mat.albedo = [rgb[0], rgb[1], rgb[2], 1.0];
                                *frame.scene_dirty = true;
                            }
                        }
                        let dirty = &mut *frame.scene_dirty;
                        if ui.add(egui::Slider::new(&mut mat.albedo[0], 0.0..=1.0).text("R"))
                            .changed() { *dirty = true; }
                        if ui.add(egui::Slider::new(&mut mat.albedo[1], 0.0..=1.0).text("G"))
                            .changed() { *dirty = true; }
                        if ui.add(egui::Slider::new(&mut mat.albedo[2], 0.0..=1.0).text("B"))
                            .changed() { *dirty = true; }

                        ui.separator();
                        if ui.add(egui::Slider::new(&mut mat.metallic, 0.0..=1.0).text("Metallic"))
                            .changed() { *dirty = true; }
                        if ui.add(egui::Slider::new(&mut mat.roughness, 0.0..=1.0).text("Roughness"))
                            .changed() { *dirty = true; }

                        ui.separator();
                        ui.label("Emissive:");
                        if ui.add(egui::Slider::new(&mut mat.emissive[0], 0.0..=2.0).text("R"))
                            .changed() { *dirty = true; }
                        if ui.add(egui::Slider::new(&mut mat.emissive[1], 0.0..=2.0).text("G"))
                            .changed() { *dirty = true; }
                        if ui.add(egui::Slider::new(&mut mat.emissive[2], 0.0..=2.0).text("B"))
                            .changed() { *dirty = true; }
                    }

                    // Lighting
                    ui.separator();
                    ui.heading("Lighting");

                    ui.label("Ambient:");
                    egui::color_picker::color_edit_button_rgb(
                        ui, &mut scene.lighting.ambient.color,
                    );
                    ui.add(
                        egui::Slider::new(&mut scene.lighting.ambient.color[0], 0.0..=1.0)
                            .text("  R"),
                    );
                    ui.add(
                        egui::Slider::new(&mut scene.lighting.ambient.color[1], 0.0..=1.0)
                            .text("  G"),
                    );
                    ui.add(
                        egui::Slider::new(&mut scene.lighting.ambient.color[2], 0.0..=1.0)
                            .text("  B"),
                    );
                    ui.add(
                        egui::Slider::new(&mut scene.lighting.ambient.intensity, 0.0..=5.0)
                            .text("Intensity"),
                    );

                    for (i, dl) in scene.lighting.directional_lights.iter_mut().enumerate() {
                        ui.separator();
                        ui.heading(format!("Dir Light {}", i));
                        egui::color_picker::color_edit_button_rgb(ui, &mut dl.color);
                        ui.add(egui::Slider::new(&mut dl.color[0], 0.0..=1.0).text("R"));
                        ui.add(egui::Slider::new(&mut dl.color[1], 0.0..=1.0).text("G"));
                        ui.add(egui::Slider::new(&mut dl.color[2], 0.0..=1.0).text("B"));
                        ui.add(egui::Slider::new(&mut dl.intensity, 0.0..=10.0).text("Intensity"));
                        ui.label("Direction:");
                        ui.add(egui::Slider::new(&mut dl.direction.x, -1.0..=1.0).text("X"));
                        ui.add(egui::Slider::new(&mut dl.direction.y, -1.0..=1.0).text("Y"));
                        ui.add(egui::Slider::new(&mut dl.direction.z, -1.0..=1.0).text("Z"));
                    }

                    ui.separator();
                    ui.label(format!("Point lights: {}", scene.lighting.point_lights.len()));
                });
            });

        // Bottom: camera controls
        egui::TopBottomPanel::bottom("camera_bar").show(egui_ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut camera.distance, 0.1..=500.0)
                        .logarithmic(true)
                        .text("Dist"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut camera.fov_y,
                        0.1..=std::f32::consts::FRAC_PI_2 * 1.5,
                    )
                    .logarithmic(true)
                    .text("FOV"),
                );
                ui.separator();
                if ui.button("Front").clicked() { camera.view_front(); }
                if ui.button("Top").clicked() { camera.view_top(); }
                if ui.button("Right").clicked() { camera.view_right(); }
                if ui.button("Iso").clicked() { camera.view_iso(); }
                if ui.button("Fit").clicked() { camera.fit_to_aabb(*scene_aabb); }
            });
        });
    }

    fn on_frame(&mut self, frame: &mut FrameCtx) {
        if let Some(path) = self.pending_load.take() {
            log::info!("Loading model: {}", path);
            match load_gltf(&path, frame.device, frame.queue, frame.texture_cache) {
                Ok((new_scene, aabb)) => {
                    *frame.scene = new_scene;
                    self.scene_aabb = aabb;
                    frame.camera.fit_to_aabb(self.scene_aabb);
                    *frame.scene_dirty = true;
                    log::info!("Loaded model: {}", path);
                }
                Err(e) => {
                    log::error!("Failed to load {}: {}", path, e);
                }
            }
        }
    }

    fn on_event(&mut self, _event: &winit::event::WindowEvent) -> EventHandling {
        EventHandling::Default
    }
}

fn main() {
    env_logger::init();
    engvis_renderer::run(App {
        model_path: String::new(),
        pending_load: None,
        selected_node: None,
        scene_aabb: Aabb::empty(),
    });
}
