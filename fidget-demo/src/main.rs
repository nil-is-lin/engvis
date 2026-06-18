// ── Fidget → engvis viewer ────────────────────────────────
// Uses the EngvisApp runner — just implement ui() and on_setup().

use winit::event::WindowEvent;

use engvis_core::{
    material::PbrMaterial,
    mesh::Mesh, scene::Scene,
    camera::OrbitCamera,
};
use engvis_renderer::{
    EngvisApp, AppCtx, FrameCtx, RunConfig, EventHandling,
};

// ── Surface builder ─────────────────────────────────────────
fn build_fidget_mesh(name: &str, depth: u8) -> Mesh {
    use fidget_core::context::Tree as T;
    use fidget_core::shape::Shape;
    use fidget_core::vm::VmFunction;
    use fidget_mesh::{Octree, Settings};

    let tree = match name {
        "sphere" => (T::x().square()+T::y().square()+T::z().square()).sqrt() - 0.8,
        "torus" => {
            let d = (T::x().square()+T::z().square()).sqrt() - 0.65;
            d.square() + T::y().square() - 0.0324
        }
        "gyroid" => {
            let (x,y,z) = (T::x(), T::y(), T::z());
            x.sin()*y.cos() + y.sin()*z.cos() + z.sin()*x.cos()
        }
        "gyroid-sphere" | _ => {
            let (x,y,z) = (T::x()*3.0, T::y()*3.0, T::z()*3.0);
            let g = x.sin()*y.cos() + y.sin()*z.cos() + z.sin()*x.cos();
            let s = (T::x().square()+T::y().square()+T::z().square()).sqrt() - 0.92;
            g.max(s)
        }
    };

    let shape = Shape::<VmFunction>::from(tree);
    let settings = Settings {
        depth,
        threads: Some(&fidget_core::render::ThreadPool::Global),
        ..Default::default()
    };
    let octree = Octree::build(&shape, &settings).expect("octree");
    let m = octree.walk_dual();

    let pos: Vec<[f32;3]> = m.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
    let idx: Vec<u32> = m.triangles.iter().flat_map(|t| [t.x as u32, t.y as u32, t.z as u32]).collect();
    eprintln!("  {} verts, {} tris (depth={})", pos.len(), idx.len()/3, depth);
    Mesh::from_triangles(name, &pos, &idx)
}

// ── App ──────────────────────────────────────────────────────

struct App {
    surf_name: String,
    surf_depth: u8,
    needs_remesh: bool,
    topo: Option<engvis_core::MeshTopology>,
    albedo:   [f32; 3],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    material_changed: bool,
    camera_fitted: bool,
}

impl EngvisApp for App {
    fn config(&self) -> RunConfig {
        RunConfig {
            title: "Fidget Viewer".into(),
            width: 1200,
            height: 800,
            sample_count: 4,
            ..Default::default()
        }
    }

    fn on_setup(&mut self, _ctx: &mut AppCtx) -> Scene {
        let mesh = build_fidget_mesh(&self.surf_name, self.surf_depth);
        self.topo = Some(engvis_core::compute_topology(&mesh));
        let material = PbrMaterial {
            name: "Surface".into(),
            albedo: [self.albedo[0], self.albedo[1], self.albedo[2], 1.0],
            metallic: self.metallic,
            roughness: self.roughness,
            emissive: self.emissive,
            ..Default::default()
        };
        Scene::single_mesh("Surface", mesh, material)
    }

    fn on_ready(&mut self, scene: &Scene, camera: &mut OrbitCamera) {
        if !self.camera_fitted {
            camera.fit_to_scene(scene);
            self.camera_fitted = true;
        }
    }

    fn ui(&mut self, egui_ctx: &egui::Context, frame: &mut FrameCtx) {
        let surf_name = &mut self.surf_name;
        let surf_depth = &mut self.surf_depth;
        let needs_remesh = &mut self.needs_remesh;
        let topo = &mut self.topo;
        let albedo = &mut self.albedo;
        let metallic = &mut self.metallic;
        let roughness = &mut self.roughness;
        let emissive = &mut self.emissive;
        let material_changed = &mut self.material_changed;

        let scene = &mut *frame.scene;
        let rs = &mut *frame.render_state;
        let fps = frame.fps;

        egui::SidePanel::left("controls")
            .resizable(true).default_width(220.0)
            .show(egui_ctx, |ui| {
                ui.heading("Surface");
                ui.add_space(4.0);

                let prev_name = surf_name.clone();
                let prev_depth = *surf_depth;
                egui::ComboBox::from_label("Type")
                    .selected_text(surf_name.as_str())
                    .show_ui(ui, |ui| {
                        for n in &["sphere", "torus", "gyroid", "gyroid-sphere"] {
                            ui.selectable_value(surf_name, (*n).to_string(), *n);
                        }
                    });
                ui.add(egui::Slider::new(surf_depth, 4..=8).text("Depth"));
                if prev_name != *surf_name || prev_depth != *surf_depth {
                    *needs_remesh = true;
                }

                ui.separator();
                ui.heading("Display");
                ui.add_space(4.0);
                ui.checkbox(&mut rs.show_surface, "Surface");
                if rs.show_surface {
                    ui.add(egui::Slider::new(&mut rs.opacity, 0.0..=1.0).text("Opacity"));
                    ui.horizontal(|ui| {
                        ui.label("Albedo");
                        if ui.color_edit_button_rgb(albedo).changed() { *material_changed = true; }
                    });
                    ui.add(egui::Slider::new(metallic, 0.0..=1.0).text("Metallic")
                        .suffix("%").custom_formatter(|v, _| format!("{:.0}", v*100.0)));
                    ui.add(egui::Slider::new(roughness, 0.0..=1.0).text("Roughness")
                        .suffix("%").custom_formatter(|v, _| format!("{:.0}", v*100.0)));
                    ui.horizontal(|ui| {
                        ui.label("Emit");
                        if ui.color_edit_button_rgb(emissive).changed() { *material_changed = true; }
                    });
                }

                ui.separator();
                ui.checkbox(&mut rs.edge_opts.enabled, "Edges");
                if rs.edge_opts.enabled {
                    ui.horizontal(|ui| {
                        ui.label("Color");
                        ui.color_edit_button_rgb(&mut rs.edge_opts.color);
                    });
                    ui.add(egui::Slider::new(&mut rs.edge_opts.line_width, 1.0..=10.0)
                        .text("Width"));
                }

                ui.separator();
                ui.checkbox(&mut rs.vertex_opts.enabled, "Vertices");
                if rs.vertex_opts.enabled {
                    ui.horizontal(|ui| {
                        ui.label("Color");
                        ui.color_edit_button_rgb(&mut rs.vertex_opts.color);
                    });
                    ui.add(egui::Slider::new(&mut rs.vertex_opts.point_size, 1.0..=10.0)
                        .text("Size"));
                }

                ui.separator();
                ui.checkbox(&mut rs.show_grid, "Grid");
            });

        // ── Bottom status bar ──
        egui::TopBottomPanel::bottom("status_bar").show(egui_ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(t) = topo {
                    let chi_color = if t.is_watertight {
                        egui::Color32::from_rgb(100, 220, 100)
                    } else {
                        egui::Color32::from_rgb(220, 160, 60)
                    };
                    ui.label(format!("V: {}  E: {}  F: {}", t.vertices, t.edges, t.faces));
                    ui.separator();
                    ui.colored_label(chi_color, format!("χ={}", t.euler));
                    ui.separator();
                    ui.label(format!("boundary: {}", t.boundary_edges));
                    ui.separator();
                    ui.label(format!("non-manifold: {}", t.non_manifold_edges));
                    ui.separator();
                    ui.label(format!("components: {}", t.connected_components));
                    ui.separator();
                    ui.colored_label(
                        chi_color,
                        if t.is_watertight { "watertight" } else { "open" },
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("FPS: {fps:.0}"));
                });
            });
        });

        // ── Remesh / material update ──
        if *needs_remesh {
            *needs_remesh = false;
            let mesh = build_fidget_mesh(surf_name, *surf_depth);
            *topo = Some(engvis_core::compute_topology(&mesh));
            scene.meshes = vec![mesh];
            scene.materials[0].name = "Surface".into();
            frame.camera.fit_to_scene(scene);
            *frame.scene_dirty = true;
        }
        if *material_changed {
            *material_changed = false;
            let mat = &mut scene.materials[0];
            mat.albedo = [albedo[0], albedo[1], albedo[2], 1.0];
            mat.metallic = *metallic;
            mat.roughness = *roughness;
            mat.emissive = *emissive;
            *frame.scene_dirty = true;
        }
    }

    fn on_event(&mut self, _event: &WindowEvent) -> EventHandling {
        EventHandling::Default
    }
}

fn main() {
    env_logger::init();
    engvis_renderer::run(App {
        surf_name: "gyroid-sphere".into(),
        surf_depth: 6,
        needs_remesh: true,
        topo: None,
        albedo: [0.25, 0.55, 0.95],
        metallic: 0.2,
        roughness: 0.35,
        emissive: [0.0, 0.0, 0.0],
        material_changed: false,
        camera_fitted: false,
    });
}
