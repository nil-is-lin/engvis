// ── Minimal engvis viewer example ──────────────────────────
// Demonstrates the simplest EngvisApp implementation.
//
// Run:  cargo run --example hello_viewer

use engvis_core::{
    material::PbrMaterial,
    mesh::create_cube_mesh,
    camera::OrbitCamera,
    scene::Scene,
};
use engvis_renderer::{
    EngvisApp, AppCtx, FrameCtx, RunConfig, EventHandling,
};

struct App;

impl EngvisApp for App {
    fn config(&self) -> RunConfig {
        RunConfig {
            title: "Hello engvis".into(),
            width: 1024,
            height: 768,
            sample_count: 1, // disable MSAA for crisp pixels
            ..Default::default()
        }
    }

    fn on_setup(&mut self, _ctx: &mut AppCtx) -> Scene {
        let cube = create_cube_mesh();
        Scene::single_mesh(
            "Cube",
            cube,
            PbrMaterial {
                name: "Default".into(),
                albedo: [0.2, 0.6, 0.9, 1.0],
                ..Default::default()
            },
        )
    }

    fn on_ready(&mut self, scene: &Scene, camera: &mut OrbitCamera) {
        camera.fit_to_scene(scene);
    }

    fn ui(&mut self, _egui_ctx: &egui::Context, _frame: &mut FrameCtx) {
        // No UI panels — just the 3D viewport and default input handling.
    }

    fn on_event(&mut self, _event: &winit::event::WindowEvent) -> EventHandling {
        EventHandling::Default
    }
}

fn main() {
    env_logger::init();
    engvis_renderer::run(App);
}
