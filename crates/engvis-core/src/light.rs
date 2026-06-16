use glam::Vec3;

#[derive(Debug, Clone)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub cast_shadows: bool,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(-0.5, -1.0, -0.3).normalize(),
            color: [1.0, 0.98, 0.95],
            intensity: 4.0,
            cast_shadows: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PointLight {
    pub position: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

#[derive(Debug, Clone)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: [0.4, 0.42, 0.45],
            intensity: 0.3,
        }
    }
}

/// Aggregated scene lighting
#[derive(Debug, Clone)]
pub struct LightingEnvironment {
    pub ambient: AmbientLight,
    pub directional_lights: Vec<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
}

impl Default for LightingEnvironment {
    fn default() -> Self {
        Self {
            ambient: AmbientLight::default(),
            directional_lights: vec![
                DirectionalLight::default(),
                // Fill light from opposite side
                DirectionalLight {
                    direction: Vec3::new(0.6, -0.3, 0.7).normalize(),
                    color: [0.9, 0.92, 1.0],
                    intensity: 1.5,
                    cast_shadows: false,
                },
            ],
            point_lights: Vec::new(),
        }
    }
}
