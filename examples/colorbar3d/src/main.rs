// ── 3D Colorbar Viewer ───────────────────────────────────────
// Demonstrates coordinate axes, tick marks, and a colorbar in 3D
// using the engvis framework.

use engvis_core::annotation::{
    create_coordinate_axes, create_all_ticks, create_colorbar,
    JET_COLORMAP, VIRIDIS_COLORMAP,
};
use engvis_core::{
    mesh::create_cube_mesh,
    material::PbrMaterial,
    camera::OrbitCamera,
    scene::{Scene, SceneNode},
};
use engvis_renderer::{
    EngvisApp, AppCtx, FrameCtx, RunConfig, EventHandling,
};
use glam::{Affine3A, Vec3};

struct App {
    show_axes: bool,
    show_ticks: bool,
    show_colorbar: bool,
    axis_length: f32,
    colorbar_colormap: usize,
    jet_visible: bool,
    viridis_visible: bool,
}

impl EngvisApp for App {
    fn config(&self) -> RunConfig {
        RunConfig {
            title: "3D Colorbar — Axes, Ticks & Colormap".into(),
            width: 1400,
            height: 900,
            ..Default::default()
        }
    }

    fn on_setup(&mut self, _ctx: &mut AppCtx) -> Scene {
        // ── Build the full scene ──
        let mut meshes: Vec<engvis_core::Mesh> = Vec::new();
        let mut materials: Vec<PbrMaterial> = Vec::new();
        let mut nodes: Vec<SceneNode> = Vec::new();

        // 1. Reference cube at origin
        let cube = create_cube_mesh();
        meshes.push(cube);
        materials.push(PbrMaterial {
            name: "Cube".into(),
            albedo: [0.5, 0.5, 0.5, 1.0],
            roughness: 0.5,
            metallic: 0.1,
            ..Default::default()
        });
        nodes.push(SceneNode {
            name: "Cube".into(),
            local_transform: Affine3A::IDENTITY,
            mesh_index: Some(meshes.len() - 1),
            children: Vec::new(),
            visible: true,
        });

        // 2. Coordinate axes
        let axis_len = 5.0;
        let body_r = 0.04;
        let head_r = 0.10;
        let head_l = 0.4;

        let (axis_meshes, axis_mats) = create_coordinate_axes(
            axis_len, body_r, head_r, head_l,
        );
        for (i, (mesh, mat)) in axis_meshes.into_iter().zip(axis_mats).enumerate() {
            meshes.push(mesh);
            materials.push(mat);
            nodes.push(SceneNode {
                name: format!("Axis{}", i),
                local_transform: Affine3A::IDENTITY,
                mesh_index: Some(meshes.len() - 1),
                children: Vec::new(),
                visible: true,
            });
        }

        // 3. Tick marks
        let (tick_meshes, tick_mats) = create_all_ticks(
            axis_len, head_l, 0.3, 0.15, 1.0, 0.25,
        );
        for (i, (mesh, mat)) in tick_meshes.into_iter().zip(tick_mats).enumerate() {
            meshes.push(mesh);
            materials.push(mat);
            nodes.push(SceneNode {
                name: format!("Ticks{}", i),
                local_transform: Affine3A::IDENTITY,
                mesh_index: Some(meshes.len() - 1),
                children: Vec::new(),
                visible: true,
            });
        }

        // 4. Colorbar — Jet colormap (visible by default)
        let jet_colorbar = create_colorbar(
            Vec3::new(4.5, -2.5, -0.5),
            0.25,
            5.0,
            64,
            JET_COLORMAP,
        );
        let jet_mesh_idx = meshes.len();
        meshes.push(jet_colorbar.mesh);
        let jet_mat_start = materials.len();
        for mat in jet_colorbar.materials {
            materials.push(mat);
        }
        let jet_mesh = &mut meshes[jet_mesh_idx];
        for sub in &mut jet_mesh.sub_meshes {
            sub.material_index += jet_mat_start;
        }
        nodes.push(SceneNode {
            name: "ColorbarJet".into(),
            local_transform: Affine3A::IDENTITY,
            mesh_index: Some(jet_mesh_idx),
            children: Vec::new(),
            visible: true,
        });

        // 5. Colorbar — Viridis colormap (hidden by default, shifted right)
        let viridis_colorbar = create_colorbar(
            Vec3::new(6.5, -2.5, -0.5),
            0.25,
            5.0,
            64,
            VIRIDIS_COLORMAP,
        );
        let vir_mesh_idx = meshes.len();
        meshes.push(viridis_colorbar.mesh);
        let vir_mat_start = materials.len();
        for mat in viridis_colorbar.materials {
            materials.push(mat);
        }
        let vir_mesh = &mut meshes[vir_mesh_idx];
        for sub in &mut vir_mesh.sub_meshes {
            sub.material_index += vir_mat_start;
        }
        nodes.push(SceneNode {
            name: "ColorbarViridis".into(),
            local_transform: Affine3A::IDENTITY,
            mesh_index: Some(vir_mesh_idx),
            children: Vec::new(),
            visible: false,
        });

        Scene {
            meshes,
            materials,
            nodes,
            lighting: engvis_core::LightingEnvironment::default(),
        }
    }

    fn on_ready(&mut self, _scene: &Scene, camera: &mut OrbitCamera) {
        camera.target = Vec3::new(0.0, 1.0, 0.0);
        camera.distance = 10.0;
        camera.set_orientation_yaw_pitch(-0.785, 0.5);
    }

    fn ui(&mut self, egui_ctx: &egui::Context, frame: &mut FrameCtx) {
        let _rs = &mut *frame.render_state;

        // Top panel
        egui::TopBottomPanel::top("top_bar").show(egui_ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("3D Colorbar");
                ui.separator();
                ui.toggle_value(&mut self.show_axes, "Axes");
                ui.toggle_value(&mut self.show_ticks, "Ticks");
                ui.toggle_value(&mut self.show_colorbar, "Colorbar");
                ui.separator();
                if ui.button("Reset View").clicked() {
                    frame.camera.target = Vec3::new(0.0, 1.0, 0.0);
                    frame.camera.distance = 10.0;
                    frame.camera.set_orientation_yaw_pitch(-0.785, 0.5);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("FPS: {:.0}", frame.fps));
                });
            });
        });

        // Side panel: colormap selection
        egui::SidePanel::right("colormap_panel")
            .default_width(200.0)
            .min_width(160.0)
            .show(egui_ctx, |ui| {
                ui.heading("Colormap");
                ui.separator();

                let prev = self.colorbar_colormap;
                ui.radio_value(&mut self.colorbar_colormap, 0, "Jet");
                ui.radio_value(&mut self.colorbar_colormap, 1, "Viridis");
                if self.colorbar_colormap != prev {
                    self.jet_visible = self.colorbar_colormap == 0;
                    self.viridis_visible = self.colorbar_colormap == 1;
                }

                ui.separator();
                ui.heading("Axis Length");
                ui.add(egui::Slider::new(&mut self.axis_length, 1.0..=10.0));
                ui.separator();
                ui.heading("Camera");
                if ui.button("Front").clicked() { frame.camera.view_front(); }
                if ui.button("Top").clicked() { frame.camera.view_top(); }
                if ui.button("Right").clicked() { frame.camera.view_right(); }
                if ui.button("Iso").clicked() { frame.camera.view_iso(); }
            });

        // Apply visibility toggles to scene nodes
        for node in frame.scene.nodes.iter_mut() {
            if node.name.starts_with("Axis") {
                node.visible = self.show_axes;
            } else if node.name.starts_with("Ticks") {
                node.visible = self.show_ticks;
            } else if node.name == "ColorbarJet" {
                node.visible = self.show_colorbar && self.jet_visible;
            } else if node.name == "ColorbarViridis" {
                node.visible = self.show_colorbar && self.viridis_visible;
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
        show_axes: true,
        show_ticks: true,
        show_colorbar: true,
        axis_length: 5.0,
        colorbar_colormap: 0,
        jet_visible: true,
        viridis_visible: false,
    });
}
