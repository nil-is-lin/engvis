use crate::camera::OrbitCamera;

/// Represents a rectangular area in screen coordinates (pixels)
#[derive(Debug, Clone, Default)]
pub struct ViewportRect {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl ViewportRect {
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }
}

/// Tracks raw input state for orbit camera control
#[derive(Debug, Clone)]
pub struct InputState {
    pub left_mouse_down: bool,
    pub right_mouse_down: bool,
    pub middle_mouse_down: bool,
    pub last_cursor_x: f64,
    pub last_cursor_y: f64,
    pub cursor_x: f64,
    pub cursor_y: f64,
    pub scroll_delta: f32,
    pub egui_wants_pointer: bool,
    /// The central 3D viewport rect in screen pixels; only mouse events inside this area affect the camera
    pub viewport_rect: ViewportRect,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            left_mouse_down: false,
            right_mouse_down: false,
            middle_mouse_down: false,
            last_cursor_x: 0.0,
            last_cursor_y: 0.0,
            cursor_x: 0.0,
            cursor_y: 0.0,
            scroll_delta: 0.0,
            egui_wants_pointer: false,
            viewport_rect: ViewportRect::default(),
        }
    }
}

impl InputState {
    /// Check if the cursor is currently inside the 3D viewport area
    fn cursor_in_viewport(&self) -> bool {
        self.viewport_rect.contains(self.cursor_x, self.cursor_y)
    }

    /// Apply accumulated input to camera, then reset deltas.
    /// Only applies camera operations when the cursor is inside the 3D viewport area
    /// and egui does not want the pointer.
    pub fn apply_to_camera(&mut self, camera: &mut OrbitCamera, _window_size: [u32; 2]) {
        let in_viewport = self.cursor_in_viewport();

        if self.left_mouse_down && !self.egui_wants_pointer && in_viewport {
            let dx = (self.cursor_x - self.last_cursor_x) as f32;
            let dy = (self.cursor_y - self.last_cursor_y) as f32;
            let sensitivity = 0.005;
            camera.orbit(-dx * sensitivity, -dy * sensitivity);
        }
        if self.right_mouse_down && !self.egui_wants_pointer && in_viewport {
            let dx = (self.cursor_x - self.last_cursor_x) as f32;
            let dy = (self.cursor_y - self.last_cursor_y) as f32;
            let pan_speed = camera.distance * 0.002;
            camera.pan(-dx * pan_speed, dy * pan_speed);
        }
        if self.scroll_delta.abs() > f32::EPSILON && !self.egui_wants_pointer && in_viewport {
            camera.zoom(self.scroll_delta * 0.1);
        }
        self.scroll_delta = 0.0;
        self.last_cursor_x = self.cursor_x;
        self.last_cursor_y = self.cursor_y;
    }
}
