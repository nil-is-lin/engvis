use glam::{Mat4, Vec3};
use crate::scene::Scene;
use crate::aabb::Aabb;

#[derive(Debug, Clone)]
pub struct OrbitCamera {
    /// Point the camera orbits around
    pub target: Vec3,
    /// Azimuth angle (radians, around Y axis)
    pub yaw: f32,
    /// Elevation angle (radians, clamped)
    pub pitch: f32,
    /// Distance from target
    pub distance: f32,
    /// Vertical field of view (radians)
    pub fov_y: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
    /// Aspect ratio (width / height)
    pub aspect_ratio: f32,
}

impl OrbitCamera {
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            yaw: -0.785, // -45 degrees
            pitch: 0.615, // ~35 degrees
            distance,
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            near: 0.01,
            far: 1000.0,
            aspect_ratio: 16.0 / 9.0,
        }
    }

    /// Target + distance variant (no pitch/yaw offset).
    pub fn looking_at(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            yaw: 0.0,
            pitch: 0.0,
            distance,
            ..Default::default()
        }
    }

    /// Camera world-space position
    pub fn position(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        let offset = Vec3::new(
            self.distance * cos_pitch * self.yaw.sin(),
            self.distance * self.pitch.sin(),
            self.distance * cos_pitch * self.yaw.cos(),
        );
        self.target + offset
    }

    /// Right direction vector
    pub fn right(&self) -> Vec3 {
        let forward = (self.target - self.position()).normalize();
        forward.cross(Vec3::Y).normalize()
    }

    /// Up direction vector
    pub fn up(&self) -> Vec3 {
        let forward = (self.target - self.position()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        right.cross(forward).normalize()
    }

    /// View matrix (world -> camera)
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Y)
    }

    /// Projection matrix
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect_ratio, self.near, self.far)
    }

    /// Combined view-projection matrix
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Orbit by delta yaw/pitch
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }

    /// Pan the target point in camera-local XY
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let right = self.right();
        let up = self.up();
        self.target += right * delta_x + up * delta_y;
    }

    /// Zoom (change distance)
    pub fn zoom(&mut self, delta: f32) {
        self.distance *= 1.0 - delta;
        self.distance = self.distance.clamp(0.1, 500.0);
    }

    /// Fit camera to show a bounding box, adjusting near/far automatically.
    pub fn fit_to_aabb(&mut self, aabb: Aabb) {
        if !aabb.is_valid() {
            return;
        }
        self.target = aabb.center();
        let radius = aabb.diagonal() * 0.5;
        let min_dist = (radius / (self.fov_y * 0.5).sin()).max(0.5);
        self.distance = min_dist * 1.4; // +40% margin
        self.near = (min_dist - radius * 2.0).max(0.01);
        self.far = (min_dist + radius * 4.0).max(10.0);
    }

    /// Fit camera to show an entire scene.
    pub fn fit_to_scene(&mut self, scene: &Scene) {
        self.fit_to_aabb(scene.compute_aabb());
    }

    /// Preset: front view
    pub fn view_front(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
    }

    /// Preset: top view
    pub fn view_top(&mut self) {
        self.yaw = 0.0;
        self.pitch = std::f32::consts::FRAC_PI_2 - 0.01;
    }

    /// Preset: right view
    pub fn view_right(&mut self) {
        self.yaw = -std::f32::consts::FRAC_PI_2;
        self.pitch = 0.0;
    }

    /// Preset: isometric view
    pub fn view_iso(&mut self) {
        self.yaw = -0.785;
        self.pitch = 0.615;
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new(Vec3::ZERO, 5.0)
    }
}
