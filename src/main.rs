use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowId;

use engvis_core::{
    InputState, OrbitCamera, Scene, ViewportRect, VertexRenderOptions, EdgeRenderOptions,
    mesh::create_cube_mesh,
    Aabb,
};
use engvis_renderer::{
    create_window_and_gpu, render_egui,
    EguiContext, EventContext, EventResult, GpuResources,
    Renderer, handle_window_event, load_gltf,
};
use glam::{Affine3A, Vec3};

struct App {
    window: Option<Arc<winit::window::Window>>,
    gpu: Option<GpuResources>,
    egui: Option<EguiContext>,
    renderer: Option<Renderer>,
    camera: OrbitCamera,
    input: InputState,
    scene: Scene,
    show_surface: bool,
    show_grid: bool,
    vertex_opts: VertexRenderOptions,
    edge_opts: EdgeRenderOptions,
    opacity: f32,
    selected_node: Option<usize>,
    model_path: String,
    pending_load: Option<String>,
    scene_aabb: Aabb,
    fps_counter: FpsCounter,
}

struct FpsCounter {
    frame_count: u32,
    last_time: std::time::Instant,
    fps: f32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frame_count: 0,
            last_time: std::time::Instant::now(),
            fps: 0.0,
        }
    }

    fn tick(&mut self) {
        self.frame_count += 1;
        let elapsed = self.last_time.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_time = std::time::Instant::now();
        }
    }
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            egui: None,
            renderer: None,
            camera: OrbitCamera::new(Vec3::ZERO, 5.0),
            input: InputState::default(),
            scene: Scene::default(),
            show_surface: true,
            show_grid: true,
            vertex_opts: VertexRenderOptions::default(),
            edge_opts: EdgeRenderOptions::default(),
            opacity: 1.0,
            selected_node: None,
            model_path: String::new(),
            pending_load: None,
            scene_aabb: Aabb::empty(),
            fps_counter: FpsCounter::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: StartCause) {}

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            pollster::block_on(async {
                let (window, gpu) =
                    create_window_and_gpu(event_loop, "engvis - Engineering Visualization", 1600, 1000)
                        .await;
                let size = window.inner_size();
                let egui = EguiContext::new(&window, &gpu.context.device, gpu.surface_format);

                let scene = Scene::default();
                let renderer = Renderer::new(
                    &gpu.context.device,
                    &gpu.context.queue,
                    gpu.surface_format,
                    &scene,
                    size.width,
                    size.height,
                );

                self.camera.aspect_ratio = size.width as f32 / size.height.max(1) as f32;

                self.window = Some(window);
                self.gpu = Some(gpu);
                self.egui = Some(egui);
                self.scene = scene;
                self.renderer = Some(renderer);
            });

            // Set up default cube scene
            {
                let cube = create_cube_mesh();
                let aabb = cube.aabb;

                let mut scene = Scene::default();
                scene.meshes = vec![cube];
                scene.materials = vec![engvis_core::PbrMaterial {
                    name: "Default Cube".to_string(),
                    albedo: [0.7, 0.4, 0.3, 1.0],
                    metallic: 0.1,
                    roughness: 0.6,
                    ..Default::default()
                }];
                scene.nodes = vec![engvis_core::SceneNode {
                    name: "Cube".to_string(),
                    local_transform: Affine3A::from_translation(Vec3::new(0.0, 0.5, 0.0)),
                    mesh_index: Some(0),
                    children: Vec::new(),
                    visible: true,
                }];

                self.scene_aabb = aabb;
                self.scene = scene;

                if let (Some(gpu), Some(renderer)) = (&self.gpu, &mut self.renderer) {
                    renderer.upload_scene(&gpu.context.device, &gpu.context.queue, &self.scene);
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = &self.window else { return };
        let size = window.inner_size();

        let mut ctx = EventContext {
            input: &mut self.input,
            camera: &mut self.camera,
            window_width: size.width,
            window_height: size.height,
        };

        match handle_window_event(event_loop, &event, &mut ctx) {
            EventResult::Exit => return,
            _ => {}
        }

        // Feed event to egui
        if let Some(egui) = &mut self.egui {
            let response = egui.state.on_window_event(window, &event);
            self.input.egui_wants_pointer = response.consumed;
        }

        if let WindowEvent::Resized(new_size) = event {
            if let Some(gpu) = &mut self.gpu {
                gpu.resize(new_size.width, new_size.height);
            }
            if let Some(renderer) = &mut self.renderer {
                if let Some(gpu) = &self.gpu {
                    renderer.resize(&gpu.context.device, new_size.width, new_size.height);
                }
            }
            self.camera.aspect_ratio =
                new_size.width as f32 / new_size.height.max(1) as f32;
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(window) = &self.window else { return };
        let Some(gpu) = &self.gpu else { return };
        let Some(renderer) = &mut self.renderer else { return };
        let Some(egui) = &mut self.egui else { return };

        let size = window.inner_size();

        // Handle pending model load
        if let Some(path) = self.pending_load.take() {
            match load_gltf(
                &path,
                &gpu.context.device,
                &gpu.context.queue,
                &mut renderer.texture_cache,
            ) {
                Ok(scene) => {
                    let mut aabb = Aabb::empty();
                    for mesh in &scene.meshes {
                        if mesh.aabb.is_valid() {
                            aabb.min = aabb.min.min(mesh.aabb.min);
                            aabb.max = aabb.max.max(mesh.aabb.max);
                        }
                    }
                    self.scene = scene;
                    self.scene_aabb = aabb;
                    renderer.upload_scene(
                        &gpu.context.device,
                        &gpu.context.queue,
                        &self.scene,
                    );
                    self.camera.fit_to_aabb(self.scene_aabb);
                    log::info!("Loaded model: {}", path);
                }
                Err(e) => {
                    log::error!("Failed to load {}: {}", path, e);
                }
            }
        }

        // Run egui UI
        self.fps_counter.tick();
        let raw_input = egui.take_input(window);
        let show_surface = &mut self.show_surface;
        let show_grid = &mut self.show_grid;
        let vertex_opts = &mut self.vertex_opts;
        let edge_opts = &mut self.edge_opts;
        let opacity = &mut self.opacity;
        let camera = &mut self.camera;
        let scene = &mut self.scene;
        let selected_node = &mut self.selected_node;
        let pending_load = &mut self.pending_load;
        let model_path = &mut self.model_path;
        let fps = self.fps_counter.fps;
        let scene_aabb = self.scene_aabb;
        let mut viewport_rect = ViewportRect::default();

        let full_output = egui.context.run(raw_input, |ctx| {
            // Top menu bar
            egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
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
                            camera.fit_to_aabb(scene_aabb);
                        }
                        ui.separator();
                        ui.checkbox(show_surface, "Show Surface");
                        ui.checkbox(&mut vertex_opts.enabled, "Show Vertices");
                        ui.checkbox(&mut edge_opts.enabled, "Show Edges");
                        ui.separator();
                        ui.checkbox(show_grid, "Show Grid");
                        ui.separator();
                        ui.add(
                            egui::Slider::new(opacity, 0.05..=1.0)
                                .text("Opacity"),
                        );
                    });
                });
            });

            // Left: Scene panel
            egui::SidePanel::left("scene_panel")
                .default_width(200.0)
                .min_width(150.0)
                .show(ctx, |ui| {
                    ui.heading("Scene");
                    ui.separator();
                    ui.label(format!("Meshes: {}", scene.meshes.len()));
                    ui.label(format!("Materials: {}", scene.materials.len()));
                    ui.label(format!("Nodes: {}", scene.nodes.len()));
                    ui.separator();
                    ui.heading("Nodes");
                    for (i, node) in scene.nodes.iter().enumerate() {
                        let selected = *selected_node == Some(i);
                        if ui.selectable_label(selected, &node.name).clicked() {
                            *selected_node = Some(i);
                        }
                    }
                });

            // Right: Properties panel
            egui::SidePanel::right("properties_panel")
                .default_width(280.0)
                .min_width(220.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Properties");
                        ui.separator();

                        // --- Surface ---
                        egui::CollapsingHeader::new("Surface")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.checkbox(show_surface, "Show Surface");
                                ui.add(
                                    egui::Slider::new(opacity, 0.05..=1.0)
                                        .text("Opacity"),
                                );
                                ui.label(format!("{:.0}%", *opacity * 100.0));
                            });

                        // --- Edges ---
                        egui::CollapsingHeader::new("Edges")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.checkbox(&mut edge_opts.enabled, "Show Edges");
                                ui.label("Color:");
                                egui::color_picker::color_edit_button_rgb(ui, &mut edge_opts.color);
                                ui.add(
                                    egui::Slider::new(&mut edge_opts.line_width, 0.5..=5.0)
                                        .text("Line Width"),
                                );
                            });

                        // --- Vertices ---
                        egui::CollapsingHeader::new("Vertices")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.checkbox(&mut vertex_opts.enabled, "Show Vertices");
                                ui.label("Color:");
                                egui::color_picker::color_edit_button_rgb(ui, &mut vertex_opts.color);
                                ui.add(
                                    egui::Slider::new(&mut vertex_opts.point_size, 1.0..=15.0)
                                        .text("Point Size"),
                                );
                            });

                        // --- Material (first material) ---
                        if !scene.materials.is_empty() {
                            let mat = &mut scene.materials[0];
                            ui.separator();
                            ui.heading("Material");
                            ui.label(&mat.name);

                            ui.label("Albedo Color:");
                            {
                                let mut rgb = [mat.albedo[0], mat.albedo[1], mat.albedo[2]];
                                egui::color_picker::color_edit_button_rgb(ui, &mut rgb);
                                mat.albedo[0] = rgb[0];
                                mat.albedo[1] = rgb[1];
                                mat.albedo[2] = rgb[2];
                            }
                            ui.add(egui::Slider::new(&mut mat.albedo[0], 0.0..=1.0).text("R"));
                            ui.add(egui::Slider::new(&mut mat.albedo[1], 0.0..=1.0).text("G"));
                            ui.add(egui::Slider::new(&mut mat.albedo[2], 0.0..=1.0).text("B"));

                            ui.separator();
                            ui.add(egui::Slider::new(&mut mat.metallic, 0.0..=1.0).text("Metallic"));
                            ui.add(egui::Slider::new(&mut mat.roughness, 0.0..=1.0).text("Roughness"));

                            ui.separator();
                            ui.label("Emissive:");
                            ui.add(egui::Slider::new(&mut mat.emissive[0], 0.0..=2.0).text("R"));
                            ui.add(egui::Slider::new(&mut mat.emissive[1], 0.0..=2.0).text("G"));
                            ui.add(egui::Slider::new(&mut mat.emissive[2], 0.0..=2.0).text("B"));
                        }

                        // --- Lighting ---
                        ui.separator();
                        ui.heading("Lighting");

                        // Ambient
                        ui.label("Ambient:");
                        egui::color_picker::color_edit_button_rgb(
                            ui, &mut scene.lighting.ambient.color,
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut scene.lighting.ambient.color[0], 0.0..=1.0,
                            ).text("  R"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut scene.lighting.ambient.color[1], 0.0..=1.0,
                            ).text("  G"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut scene.lighting.ambient.color[2], 0.0..=1.0,
                            ).text("  B"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut scene.lighting.ambient.intensity, 0.0..=5.0,
                            ).text("Intensity"),
                        );

                        // Directional lights
                        for (i, dl) in scene.lighting.directional_lights.iter_mut().enumerate() {
                            ui.separator();
                            ui.heading(format!("Dir Light {}", i));

                            egui::color_picker::color_edit_button_rgb(
                                ui, &mut dl.color,
                            );
                            ui.add(
                                egui::Slider::new(&mut dl.color[0], 0.0..=1.0).text("R"),
                            );
                            ui.add(
                                egui::Slider::new(&mut dl.color[1], 0.0..=1.0).text("G"),
                            );
                            ui.add(
                                egui::Slider::new(&mut dl.color[2], 0.0..=1.0).text("B"),
                            );
                            ui.add(
                                egui::Slider::new(&mut dl.intensity, 0.0..=10.0)
                                    .text("Intensity"),
                            );

                            ui.label("Direction:");
                            ui.add(
                                egui::Slider::new(&mut dl.direction.x, -1.0..=1.0).text("X"),
                            );
                            ui.add(
                                egui::Slider::new(&mut dl.direction.y, -1.0..=1.0).text("Y"),
                            );
                            ui.add(
                                egui::Slider::new(&mut dl.direction.z, -1.0..=1.0).text("Z"),
                            );
                        }

                        ui.separator();
                        ui.label(format!(
                            "Point lights: {}",
                            scene.lighting.point_lights.len()
                        ));
                    });
                });

            // Bottom: Camera panel
            egui::TopBottomPanel::bottom("camera_panel").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Dist:");
                    ui.add(egui::Slider::new(&mut camera.distance, 0.1..=500.0).logarithmic(true));
                    ui.label("FOV:");
                    ui.add(
                        egui::Slider::new(
                            &mut camera.fov_y,
                            0.1..=std::f32::consts::FRAC_PI_2 * 1.5,
                        )
                        .logarithmic(true),
                    );
                    ui.separator();
                    if ui.button("Front").clicked() {
                        camera.view_front();
                    }
                    if ui.button("Top").clicked() {
                        camera.view_top();
                    }
                    if ui.button("Right").clicked() {
                        camera.view_right();
                    }
                    if ui.button("Iso").clicked() {
                        camera.view_iso();
                    }
                    if ui.button("Fit").clicked() {
                        camera.fit_to_aabb(scene_aabb);
                    }
                    ui.separator();
                    ui.label(format!("FPS: {:.0}", fps));
                });
            });

            // Central panel: transparent so the 3D viewport shows through
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    // Capture the central viewport rect for input hit-testing
                    let rect = ui.max_rect();
                    let ppp = ui.ctx().pixels_per_point();
                    // egui uses logical points, convert to physical pixels for winit cursor coords
                    viewport_rect = ViewportRect {
                        min_x: (rect.min.x * ppp) as f64,
                        min_y: (rect.min.y * ppp) as f64,
                        max_x: (rect.max.x * ppp) as f64,
                        max_y: (rect.max.y * ppp) as f64,
                    };
                });
        });

        // Apply input to camera (after egui run so viewport_rect is available)
        self.input.viewport_rect = viewport_rect;
        self.input
            .apply_to_camera(&mut self.camera, [size.width, size.height]);

        // Get surface texture
        let Some(output) = gpu.get_current_texture() else {
            return;
        };

        // Set renderer state
        renderer.show_surface = self.show_surface;
        renderer.show_grid = self.show_grid;
        renderer.vertex_opts = self.vertex_opts;
        renderer.edge_opts = self.edge_opts;
        renderer.opacity = self.opacity;

        // Render 3D scene + egui onto the same surface texture via closure
        let device = &gpu.context.device;
        let queue = &gpu.context.queue;
        let scene = &self.scene;
        let camera = &self.camera;

        let (cmd, output) = render_egui(
            egui,
            device,
            queue,
            window,
            output,
            full_output,
            |view, encoder| {
                renderer.render_scene_pass(device, queue, view, encoder, scene, camera);
            },
        );

        queue.submit(std::iter::once(cmd));
        output.present();
        window.request_redraw();
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
