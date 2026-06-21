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
            // Key light: from upper-right-front (Z-up convention)
            direction: Vec3::new(-0.6, -0.8, -1.2).normalize(),
            color: [1.0, 0.97, 0.92],
            intensity: 7.0,
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
            color: [0.45, 0.47, 0.50],
            intensity: 0.5,
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
                // Fill light from lower-left (Z-up)
                DirectionalLight {
                    direction: Vec3::new(0.4, -0.6, -0.3).normalize(),
                    color: [0.85, 0.88, 0.95],
                    intensity: 3.0,
                    cast_shadows: false,
                },
                // Rim/back light for edge definition
                DirectionalLight {
                    direction: Vec3::new(0.1, 0.6, 0.4).normalize(),
                    color: [1.0, 1.0, 1.0],
                    intensity: 2.0,
                    cast_shadows: false,
                },
            ],
            point_lights: Vec::new(),
        }
    }
}
