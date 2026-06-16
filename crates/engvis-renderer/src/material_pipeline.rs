use wgpu::util::DeviceExt;
use engvis_core::PbrMaterial;
use crate::texture_cache::TextureCache;

/// GPU material parameters uniform
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniforms {
    pub albedo: [f32; 4],
    pub emissive: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub normal_scale: f32,
    pub alpha_cutoff: f32,
}

pub struct MaterialPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub solid_pipeline: wgpu::RenderPipeline,
}

impl MaterialPipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        scene_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        object_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Material Bind Group Layout"),
                entries: &[
                    // albedo texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // metallic-roughness texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // normal texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // material params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR Pipeline Layout"),
            bind_group_layouts: &[
                scene_layout,
                lighting_layout,
                &bind_group_layout,
                object_layout,
            ],
            push_constant_ranges: &[],
        });

        let shader_source = Self::build_shader_source();
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PBR Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<engvis_core::MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // normal
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                // uv
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 24,
                    shader_location: 2,
                },
                // tangent
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 3,
                },
            ],
        };

        let solid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PBR Solid Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout.clone()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::depth::DepthTexture::FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 4,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            bind_group_layout,
            solid_pipeline,
        }
    }

    pub fn create_material_bind_group(
        &self,
        device: &wgpu::Device,
        material: &PbrMaterial,
        texture_cache: &TextureCache,
    ) -> (wgpu::BindGroup, wgpu::Buffer) {
        let uniforms = MaterialUniforms {
            albedo: material.albedo,
            emissive: [
                material.emissive[0],
                material.emissive[1],
                material.emissive[2],
                0.0,
            ],
            metallic: material.metallic,
            roughness: material.roughness,
            normal_scale: material.normal_scale,
            alpha_cutoff: material.alpha_cutoff,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        texture_cache.get_view(material.albedo_texture),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        texture_cache.get_view(material.metallic_roughness_texture),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        texture_cache.get_normal_view(material.normal_texture),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&texture_cache.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        (bg, uniform_buffer)
    }

    fn build_shader_source() -> String {
        // Common struct definitions shared across shaders
        let common = r#"
struct SceneUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    viewport: vec4<f32>,
    global_opacity: vec4<f32>,
}

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct DirectionalLightData {
    direction: vec4<f32>,
    color: vec4<f32>,
}

struct PointLightData {
    position: vec4<f32>,
    color: vec4<f32>,
}

struct LightingUniforms {
    ambient_color: vec4<f32>,
    dir_light_count: u32,
    point_light_count: u32,
    _pad0: u32,
    _pad1: u32,
}

struct MaterialUniforms {
    albedo: vec4<f32>,
    emissive: vec4<f32>,
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    alpha_cutoff: f32,
}
"#;

        let pbr = r#"
@group(0) @binding(0) var<uniform> scene: SceneUniforms;

@group(1) @binding(0) var<uniform> lighting: LightingUniforms;
@group(1) @binding(1) var<storage, read> dir_lights: array<DirectionalLightData>;
@group(1) @binding(2) var<storage, read> point_lights: array<PointLightData>;

@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var mr_tex: texture_2d<f32>;
@group(2) @binding(2) var normal_tex: texture_2d<f32>;
@group(2) @binding(3) var mat_sampler: sampler;
@group(2) @binding(4) var<uniform> material: MaterialUniforms;

@group(3) @binding(0) var<uniform> object: ObjectUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = (object.model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_pos = scene.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = normalize((object.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.uv;
    out.world_tangent = normalize((object.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
    out.world_bitangent = cross(out.world_normal, out.world_tangent) * in.tangent.w;
    return out;
}

const PI: f32 = 3.14159265359;

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn cook_torrance_brdf(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
    roughness: f32, metallic: f32, albedo: vec3<f32>,
) -> vec3<f32> {
    let H = normalize(V + L);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    let NDF = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    return numerator / denominator;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample textures
    let base_color = textureSample(albedo_tex, mat_sampler, in.uv) * material.albedo;
    let mr = textureSample(mr_tex, mat_sampler, in.uv);
    let roughness = mr.g * material.roughness;
    let metallic = mr.b * material.metallic;

    // Normal mapping
    let sampled_normal = textureSample(normal_tex, mat_sampler, in.uv).xyz * 2.0 - 1.0;
    let T = normalize(in.world_tangent);
    let B = normalize(in.world_bitangent);
    let Ng = normalize(in.world_normal);
    let TBN = mat3x3<f32>(T, B, Ng);
    let N = normalize(TBN * (sampled_normal * vec3<f32>(material.normal_scale, material.normal_scale, 1.0)));

    let V = normalize(scene.camera_pos.xyz - in.world_pos);
    let F0 = mix(vec3<f32>(0.04), base_color.rgb, metallic);

    var Lo = vec3<f32>(0.0);

    // Directional lights
    for (var i = 0u; i < lighting.dir_light_count; i = i + 1u) {
        let light = dir_lights[i];
        let L = normalize(-light.direction.xyz);
        let radiance = light.color.rgb * light.direction.w;
        let NdotL = max(dot(N, L), 0.0);

        let specular = cook_torrance_brdf(N, V, L, max(roughness, 0.02), metallic, base_color.rgb);
        let kD = (vec3<f32>(1.0) - fresnel_schlick(max(dot(N, V), 0.0), F0)) * (1.0 - metallic);
        let diffuse = kD * base_color.rgb / PI;

        Lo = Lo + (diffuse + specular) * radiance * NdotL;
    }

    // Point lights
    for (var i = 0u; i < lighting.point_light_count; i = i + 1u) {
        let light = point_lights[i];
        let light_vec = light.position.xyz - in.world_pos;
        let distance = length(light_vec);
        let L = normalize(light_vec);
        let attenuation = light.color.w / (distance * distance + 0.0001);
        let radiance = light.color.rgb * attenuation;
        let NdotL = max(dot(N, L), 0.0);

        let specular = cook_torrance_brdf(N, V, L, max(roughness, 0.02), metallic, base_color.rgb);
        let kD = (vec3<f32>(1.0) - fresnel_schlick(max(dot(N, V), 0.0), F0)) * (1.0 - metallic);
        let diffuse = kD * base_color.rgb / PI;

        Lo = Lo + (diffuse + specular) * radiance * NdotL;
    }

    // Ambient
    let ambient = lighting.ambient_color.rgb * lighting.ambient_color.a * base_color.rgb;
    let emissive = material.emissive.rgb;

    var color = ambient + Lo + emissive;

    // Simple tonemap (Reinhard) - HDR to [0,1)
    color = color / (color + vec3<f32>(1.0));
    // Gamma correction is handled by GPU automatically: we write linear color to
    // a Bgra8UnormSrgb surface, and the GPU converts linear -> sRGB.
    // Do NOT apply pow(color, 1/2.2) here - that would cause double gamma!

    return vec4<f32>(color, base_color.a * scene.global_opacity.x);
}
"#;

        format!("{}\n{}", common, pbr)
    }
}
