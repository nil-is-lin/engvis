use wgpu::util::DeviceExt;
use engvis_core::{OrbitCamera, Scene, VertexRenderOptions, EdgeRenderOptions};
use glam::Affine3A;

use crate::depth::DepthTexture;
use crate::grid_renderer::GridRenderer;
use crate::lighting::LightingBuffer;
use crate::material_pipeline::MaterialPipeline;
use crate::mesh_renderer::MeshRenderer;
use crate::overlay_renderer::OverlayRenderer;
use crate::texture_cache::TextureCache;

const MSAA_SAMPLE_COUNT: u32 = 4;

#[derive(Debug, Clone, Copy)]
enum OverlayDrawMode {
    Vertices,
    Edges,
}

/// Scene uniform data (group 0)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub viewport: [f32; 4],
    pub global_opacity: [f32; 4],
}

pub struct Renderer {
    pub depth: DepthTexture,
    pub msaa_texture: wgpu::Texture,
    pub msaa_view: wgpu::TextureView,
    pub surface_format: wgpu::TextureFormat,
    pub scene_uniform_buffer: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
    pub scene_bind_group_layout: wgpu::BindGroupLayout,
    pub lighting: LightingBuffer,
    pub material_pipeline: MaterialPipeline,
    pub mesh_renderer: MeshRenderer,
    pub grid_renderer: GridRenderer,
    pub overlay_renderer: OverlayRenderer,
    pub texture_cache: TextureCache,
    pub show_surface: bool,
    pub show_grid: bool,
    pub vertex_opts: VertexRenderOptions,
    pub edge_opts: EdgeRenderOptions,
    pub opacity: f32,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        scene: &Scene,
        width: u32,
        height: u32,
    ) -> Self {
        let depth = DepthTexture::new(device, width.max(1), height.max(1), MSAA_SAMPLE_COUNT);

        // MSAA color target: render into this multisampled texture, then resolve to surface
        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Color Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Scene uniform bind group (group 0)
        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scene Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let initial_uniforms = SceneUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
            viewport: [width as f32, height as f32, 0.0, 0.0],
            global_opacity: [1.0, 0.0, 0.0, 0.0],
        };

        let scene_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Scene Uniform Buffer"),
                contents: bytemuck::cast_slice(&[initial_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene Bind Group"),
            layout: &scene_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_uniform_buffer.as_entire_binding(),
            }],
        });

        // Lighting (group 1)
        let lighting = LightingBuffer::new(device, &scene.lighting);

        // Material pipeline
        // Grid renderer (needs its own reference to scene layout)
        let scene_layout_for_grid = scene_bind_group_layout.clone();
        let grid_renderer = GridRenderer::new(device, surface_format, &scene_layout_for_grid);

        let mesh_renderer = MeshRenderer::new(device);
        let material_pipeline = MaterialPipeline::new(
            device,
            surface_format,
            &scene_bind_group_layout,
            &lighting.bind_group_layout,
            &mesh_renderer.object_bind_group_layout,
        );

        // Overlay renderer (needs scene layout + object layout)
        let scene_layout_for_overlay = scene_bind_group_layout.clone();
        let overlay_renderer = OverlayRenderer::new(
            device,
            surface_format,
            crate::depth::DepthTexture::FORMAT,
            &scene_layout_for_overlay,
            &mesh_renderer.object_bind_group_layout,
        );

        // Texture cache
        let texture_cache = TextureCache::new(device, queue);

        // Upload meshes
        let mut renderer = Self {
            depth,
            msaa_texture,
            msaa_view,
            surface_format,
            scene_uniform_buffer,
            scene_bind_group,
            scene_bind_group_layout,
            lighting,
            material_pipeline,
            mesh_renderer,
            grid_renderer,
            overlay_renderer,
            texture_cache,
            show_surface: true,
            show_grid: true,
            vertex_opts: VertexRenderOptions::default(),
            edge_opts: EdgeRenderOptions::default(),
            opacity: 1.0,
        };

        // Upload scene meshes
        for mesh in &scene.meshes {
            renderer.mesh_renderer.upload_mesh(device, mesh);
        }

        renderer
    }

    pub fn upload_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
    ) {
        // Re-upload meshes
        self.mesh_renderer.mesh_buffers.clear();
        for mesh in &scene.meshes {
            self.mesh_renderer.upload_mesh(device, mesh);
        }

        // Create material bind groups and store uniform buffers for runtime updates
        self.mesh_renderer.material_bind_groups.clear();
        self.mesh_renderer.material_uniform_buffers.clear();
        for material in &scene.materials {
            let (bg, buf) = self
                .material_pipeline
                .create_material_bind_group(device, material, &self.texture_cache);
            self.mesh_renderer.material_bind_groups.push(bg);
            self.mesh_renderer.material_uniform_buffers.push(buf);
        }

        // Update lighting
        self.lighting.update(queue, &scene.lighting);
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.depth.resize(device, width, height);

            // Recreate MSAA color texture
            self.msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("MSAA Color Texture"),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: MSAA_SAMPLE_COUNT,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.msaa_view = self.msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    pub fn render_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        scene: &Scene,
        camera: &OrbitCamera,
    ) -> wgpu::CommandBuffer {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Scene Encoder"),
        });
        self.render_scene_pass(device, queue, view, &mut encoder, scene, camera);
        encoder.finish()
    }

    /// Record the 3D scene pass onto an existing encoder (shared with egui).
    /// The view is the same surface texture view that egui will use with LoadOp::Load.
    pub fn render_scene_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
        camera: &OrbitCamera,
    ) {
        // Update scene uniforms
        let vp = camera.view_projection();
        let pos = camera.position();
        let uniforms = SceneUniforms {
            view_proj: vp.to_cols_array_2d(),
            camera_pos: [pos.x, pos.y, pos.z, 0.0],
            viewport: [
                self.depth.texture.width() as f32,
                self.depth.texture.height() as f32,
                0.0,
                0.0,
            ],
            global_opacity: [self.opacity, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.scene_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
        self.lighting.update(queue, &scene.lighting);

        // Sync material uniforms from scene (allows real-time UI editing)
        for (i, material) in scene.materials.iter().enumerate() {
            if i < self.mesh_renderer.material_uniform_buffers.len() {
                let mat_uniforms = crate::material_pipeline::MaterialUniforms {
                    albedo: material.albedo,
                    emissive: [material.emissive[0], material.emissive[1], material.emissive[2], 0.0],
                    metallic: material.metallic,
                    roughness: material.roughness,
                    normal_scale: material.normal_scale,
                    alpha_cutoff: material.alpha_cutoff,
                };
                queue.write_buffer(
                    &self.mesh_renderer.material_uniform_buffers[i],
                    0,
                    bytemuck::cast_slice(&[mat_uniforms]),
                );
            }
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Scene Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                resolve_target: Some(view),
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
                view: &self.depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
        render_pass.set_bind_group(1, &self.lighting.bind_group, &[]);

        if self.show_surface {
            render_pass.set_pipeline(&self.material_pipeline.solid_pipeline);
            self.render_scene_nodes(&mut render_pass, scene, device, Affine3A::IDENTITY);
        }

        if self.show_grid {
            self.grid_renderer.render(&mut render_pass);
        }

        // --- Edge overlay ---
        if self.edge_opts.enabled {
            let (_buf, edge_overlay_bg) = self
                .overlay_renderer
                .create_uniform_bind_group(device, self.edge_opts.color, 0.0, self.edge_opts.line_width);

            render_pass.set_pipeline(&self.overlay_renderer.line_pipeline);
            self.render_overlay_nodes(&mut render_pass, scene, device, Affine3A::IDENTITY, &edge_overlay_bg, OverlayDrawMode::Edges);
        }

        // Vertex overlay renders in the same MSAA pass
        if self.vertex_opts.enabled {
            render_pass.set_pipeline(&self.overlay_renderer.point_pipeline);
            let (_buf, point_overlay_bg) = self
                .overlay_renderer
                .create_uniform_bind_group(device, self.vertex_opts.color, self.vertex_opts.point_size, 0.0);
            self.render_overlay_nodes(&mut render_pass, scene, device, Affine3A::IDENTITY, &point_overlay_bg, OverlayDrawMode::Vertices);
        }
    }

    fn render_overlay_nodes(
        &self,
        render_pass: &mut wgpu::RenderPass,
        scene: &Scene,
        device: &wgpu::Device,
        parent_transform: Affine3A,
        overlay_bind_group: &wgpu::BindGroup,
        mode: OverlayDrawMode,
    ) {
        for node in &scene.nodes {
            self.render_overlay_node(render_pass, scene, device, node, parent_transform, overlay_bind_group, mode);
        }
    }

    fn render_overlay_node(
        &self,
        render_pass: &mut wgpu::RenderPass,
        scene: &Scene,
        device: &wgpu::Device,
        node: &engvis_core::SceneNode,
        parent_transform: Affine3A,
        overlay_bind_group: &wgpu::BindGroup,
        mode: OverlayDrawMode,
    ) {
        if !node.visible {
            return;
        }

        let world_transform = parent_transform * node.local_transform;

        if let Some(mesh_idx) = node.mesh_index {
            if mesh_idx < self.mesh_renderer.mesh_buffers.len() {
                let mesh_buf = &self.mesh_renderer.mesh_buffers[mesh_idx];

                render_pass.set_bind_group(0, &self.scene_bind_group, &[]);

                let (_obj_buf, obj_bg) = self
                    .mesh_renderer
                    .create_object_bind_group(device, world_transform);
                render_pass.set_bind_group(1, &obj_bg, &[]);
                render_pass.set_bind_group(2, overlay_bind_group, &[]);

                match mode {
                    OverlayDrawMode::Vertices => {
                        // Point pipeline: slot 0 = mesh vertices (per-instance), slot 1 = point quad
                        render_pass.set_vertex_buffer(0, mesh_buf.vertex_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, self.overlay_renderer.point_quad_buffer.slice(..));
                        render_pass.draw(0..6, 0..mesh_buf.vertex_count);
                    }
                    OverlayDrawMode::Edges => {
                        // Line pipeline: slot 0 = edge endpoints (per-instance), slot 1 = line quad
                        render_pass.set_vertex_buffer(0, mesh_buf.edge_endpoint_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, self.overlay_renderer.line_quad_buffer.slice(..));
                        render_pass.draw(0..6, 0..mesh_buf.edge_instance_count);
                    }
                }
            }
        }

        for child in &node.children {
            self.render_overlay_node(render_pass, scene, device, child, world_transform, overlay_bind_group, mode);
        }
    }

    pub fn render_scene_nodes(
        &self,
        render_pass: &mut wgpu::RenderPass,
        scene: &Scene,
        device: &wgpu::Device,
        parent_transform: Affine3A,
    ) {
        for node in &scene.nodes {
            self.render_node(render_pass, scene, device, node, parent_transform);
        }
    }

    fn render_node(
        &self,
        render_pass: &mut wgpu::RenderPass,
        scene: &Scene,
        device: &wgpu::Device,
        node: &engvis_core::SceneNode,
        parent_transform: Affine3A,
    ) {
        if !node.visible {
            return;
        }

        let world_transform = parent_transform * node.local_transform;

        if let Some(mesh_idx) = node.mesh_index {
            if mesh_idx < self.mesh_renderer.mesh_buffers.len() {
                let mesh_buf = &self.mesh_renderer.mesh_buffers[mesh_idx];
                let mesh = &scene.meshes[mesh_idx];

                render_pass.set_vertex_buffer(0, mesh_buf.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    mesh_buf.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                // Create object bind group for this node
                let (_obj_buf, obj_bg) = self
                    .mesh_renderer
                    .create_object_bind_group(device, world_transform);

                render_pass.set_bind_group(3, &obj_bg, &[]);

                for sub_mesh in &mesh.sub_meshes {
                    let mat_idx = sub_mesh.material_index;
                    if mat_idx < self.mesh_renderer.material_bind_groups.len() {
                        render_pass.set_bind_group(
                            2,
                            &self.mesh_renderer.material_bind_groups[mat_idx],
                            &[],
                        );
                    }
                    render_pass.draw_indexed(
                        sub_mesh.index_offset..sub_mesh.index_offset + sub_mesh.index_count,
                        0,
                        0..1,
                    );
                }
            }
        }

        for child in &node.children {
            self.render_node(render_pass, scene, device, child, world_transform);
        }
    }
}
