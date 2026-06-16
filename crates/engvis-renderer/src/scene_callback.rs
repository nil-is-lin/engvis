use std::sync::{Arc, Mutex};

use egui_wgpu::CallbackResources;
use glam::Affine3A;

use engvis_core::{OrbitCamera, RenderMode, Scene};

use crate::renderer::{Renderer, SceneUniforms};

/// Shared reference to the surface texture view for the current frame.
/// Stored in `CallbackResources` so the scene callback can access it.
pub type SurfaceViewSlot = Arc<Mutex<Option<wgpu::TextureView>>>;

pub fn store_surface_view(resources: &mut CallbackResources, view: wgpu::TextureView) {
    resources
        .get_mut::<SurfaceViewSlot>()
        .expect("SurfaceViewSlot not registered")
        .lock()
        .unwrap()
        .replace(view);
}

fn take_surface_view(resources: &CallbackResources) -> Option<wgpu::TextureView> {
    resources
        .get::<SurfaceViewSlot>()
        .expect("SurfaceViewSlot not registered")
        .lock()
        .unwrap()
        .take()
}

/// egui PaintCallback that renders the 3D scene inside egui's render pass.
///
/// The 3D render pass is recorded in `prepare()` onto egui's command encoder
/// (with a depth attachment). egui's own render pass runs afterwards with
/// `LoadOp::Load`, compositing UI panels on top of the 3D content.
pub struct SceneCallback {
    pub renderer: Arc<Renderer>,
    pub scene: Arc<Mutex<Scene>>,
    pub camera: Arc<Mutex<OrbitCamera>>,
    pub render_mode: RenderMode,
    pub show_grid: bool,
}

// SAFETY: wgpu types and our data types are all Send + Sync.
unsafe impl Send for SceneCallback {}
unsafe impl Sync for SceneCallback {}

impl egui_wgpu::CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Take the surface view that was stored before egui::Context::run()
        let Some(view) = take_surface_view(callback_resources) else {
            return Vec::new();
        };

        let scene = self.scene.lock().unwrap();
        let camera = self.camera.lock().unwrap();

        // Update scene uniforms
        let vp = camera.view_projection();
        let pos = camera.position();
        let uniforms = SceneUniforms {
            view_proj: vp.to_cols_array_2d(),
            camera_pos: [pos.x, pos.y, pos.z, 0.0],
            viewport: [
                self.renderer.depth.texture.width() as f32,
                self.renderer.depth.texture.height() as f32,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(
            &self.renderer.scene_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
        self.renderer.lighting.update(queue, &scene.lighting);

        // Record 3D render pass onto egui's encoder
        {
            let mut render_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Render Pass (via PaintCallback)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.18,
                            g: 0.20,
                            b: 0.24,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.renderer.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_bind_group(0, &self.renderer.scene_bind_group, &[]);
            render_pass.set_bind_group(1, &self.renderer.lighting.bind_group, &[]);

            // Choose pipeline
            match self.render_mode {
                RenderMode::Solid => {
                    render_pass.set_pipeline(&self.renderer.material_pipeline.solid_pipeline);
                }
                RenderMode::Wireframe => {
                    render_pass.set_pipeline(&self.renderer.material_pipeline.wireframe_pipeline);
                }
                RenderMode::SolidWireframe => {
                    render_pass.set_pipeline(&self.renderer.material_pipeline.solid_pipeline);
                }
            }

            self.renderer
                .render_scene_nodes(&mut render_pass, &scene, device, Affine3A::IDENTITY);

            // Wireframe overlay
            if self.render_mode == RenderMode::SolidWireframe {
                render_pass
                    .set_pipeline(&self.renderer.material_pipeline.wireframe_pipeline);
                self.renderer
                    .render_scene_nodes(&mut render_pass, &scene, device, Affine3A::IDENTITY);
            }

            // Grid + axes
            if self.show_grid {
                self.renderer.grid_renderer.render(&mut render_pass);
            }
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        _render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &CallbackResources,
    ) {
        // All 3D rendering done in prepare(); nothing to draw here.
    }
}
