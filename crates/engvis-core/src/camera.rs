use glam::{Mat4, Quat, Vec3};
use crate::scene::Scene;
use crate::aabb::Aabb;

#[derive(Debug, Clone)]
pub struct OrbitCamera {
    /// Point the camera orbits around
    pub target: Vec3,
    /// Camera orientation as a quaternion (camera-space → world-space).
    /// Replaces yaw/pitch Euler angles — no gimbal lock, no pitch clamp.
    pub orientation: Quat,
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
            orientation: Quat::from_rotation_y(-0.785) * Quat::from_rotation_x(-0.615),
            distance,
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            near: 0.01,
            far: 1000.0,
            aspect_ratio: 16.0 / 9.0,
        }
    }

    /// Target + distance variant (identity orientation).
    pub fn looking_at(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            orientation: Quat::IDENTITY,
            distance,
            ..Default::default()
        }
    }

    /// Camera world-space position.
    /// Camera sits at +Z in camera space relative to target, so the
    /// target→camera direction is `orientation * Vec3::Z`.
    pub fn position(&self) -> Vec3 {
        self.target + self.orientation * (Vec3::Z * self.distance)
    }

    /// Right direction vector (camera local +X in world space)
    pub fn right(&self) -> Vec3 {
        self.orientation * Vec3::X
    }

    /// Up direction vector (camera local +Y in world space)
    pub fn up(&self) -> Vec3 {
        self.orientation * Vec3::Y
    }

    /// View matrix (world → camera).
    /// Constructed directly from the orientation quaternion to avoid
    /// degeneracies that `look_at_rh` suffers when the view direction
    /// aligns with the up vector (poles).
    pub fn view_matrix(&self) -> Mat4 {
        let rot = Mat4::from_quat(self.orientation.inverse());
        let trans = Mat4::from_translation(-self.position());
        rot * trans
    }

    /// Projection matrix
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect_ratio, self.near, self.far)
    }

    /// Combined view-projection matrix
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Orbit by incremental rotation.
    ///
    /// `delta_yaw`   — rotation around world up (Y), keeps the horizon level.
    /// `delta_pitch` — rotation around the camera's local right axis (tilt).
    ///
    /// Both are applied as incremental quaternion multiplications, so there
    /// is no gimbal lock and no pitch clamp — the camera can rotate freely.
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        // Yaw: pre-multiply (world-space rotation around Y)
        let yaw_rot = Quat::from_rotation_y(delta_yaw);
        // Pitch: post-multiply (local-space rotation around camera X).
        // Negate so that positive delta_pitch raises the camera, matching
        // the old Euler convention where pitch > 0 = camera above target.
        let pitch_rot = Quat::from_rotation_x(-delta_pitch);
        self.orientation = (yaw_rot * self.orientation * pitch_rot).normalize();
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

    /// Set orientation from yaw/pitch angles (compatibility helper).
    pub fn set_orientation_yaw_pitch(&mut self, yaw: f32, pitch: f32) {
        self.orientation = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(-pitch);
    }

    /// Preset: front view
    pub fn view_front(&mut self) {
        self.orientation = Quat::IDENTITY;
    }

    /// Preset: top view
    pub fn view_top(&mut self) {
        self.orientation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    }

    /// Preset: right view
    pub fn view_right(&mut self) {
        self.orientation = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
    }

    /// Preset: isometric view
    pub fn view_iso(&mut self) {
        self.orientation = Quat::from_rotation_y(-0.785) * Quat::from_rotation_x(-0.615);
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new(Vec3::ZERO, 5.0)
    }
}
