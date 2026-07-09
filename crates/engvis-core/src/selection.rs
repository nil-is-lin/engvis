//! Element selection primitives (pick by screen cursor).
//!
//! Selection is a *view / interaction* concern, not part of the mesh model.
//! Picking is done on the CPU by projecting each element to screen pixels and
//! finding the nearest one within a threshold — no GPU read-back, fully
//! reversible, and unit-testable.  The renderer never needs to know about it;
//! the host app stores the [`Selection`] and draws its own visual feedback
//! (e.g. an egui overlay).

use crate::camera::OrbitCamera;
use crate::input::ViewportRect;
use crate::mesh::Mesh;
use glam::{Mat4, Vec4};

/// Which kind of element is currently selected, and its index into the mesh.
///
/// Indices refer to the *primary* mesh the host app picked against.  `None`
/// means nothing is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Selection {
    #[default]
    None,
    /// A single vertex; index into `Mesh::vertices`.
    Vertex(u32),
    /// A single edge; index into the deduplicated edge list
    /// (`Mesh::extract_edge_indices`, pairs of vertex indices).
    Edge(u32),
    /// A single triangle; index into `Mesh::indices` chunks of 3.
    Face(u32),
}

impl Selection {
    /// Human-readable kind name (for UI labels).
    pub fn kind(&self) -> &'static str {
        match self {
            Selection::None => "none",
            Selection::Vertex(_) => "vertex",
            Selection::Edge(_) => "edge",
            Selection::Face(_) => "face",
        }
    }

    /// The element index, if any is selected.
    pub fn index(&self) -> Option<u32> {
        match self {
            Selection::None => None,
            Selection::Vertex(i) | Selection::Edge(i) | Selection::Face(i) => Some(*i),
        }
    }
}

/// Default screen-space pick radius, in physical pixels.
pub const DEFAULT_PICK_THRESHOLD_PX: f32 = 8.0;

impl Mesh {
    /// Pick the nearest selectable element to a screen-space cursor.
    ///
    /// * `cursor`  — cursor position in **physical pixels** (same unit as
    ///   [`ViewportRect`]).
    /// * `viewport` — the 3D viewport rectangle in physical pixels.
    /// * `threshold_px` — max distance (px) within which an element is
    ///   selectable.
    ///
    /// Resolution priority on ties (so the most specific element wins when
    /// several are equally close): `vertex` > `edge` > `face`.  Cost is
    /// O(V + E + F) and only intended for click-time use, not per-frame.
    pub fn pick(
        &self,
        camera: &OrbitCamera,
        cursor: [f64; 2],
        viewport: &ViewportRect,
        threshold_px: f32,
    ) -> Selection {
        let vp = camera.view_projection();

        // (distance, candidate) with a per-kind bias so specifics win ties.
        let mut best: Option<(f32, Selection)> = None;
        let mut consider = |dist: f32, candidate: Selection, bias: f32| {
            if dist > threshold_px {
                return;
            }
            match best {
                None => best = Some((dist, candidate)),
                Some((bd, _)) => {
                    if dist < bd - bias {
                        best = Some((dist, candidate));
                    }
                }
            }
        };

        // ── Vertices ──
        for (i, v) in self.vertices.iter().enumerate() {
            if let Some(p) = project_to_pixel(v.position, &vp, viewport) {
                let d = ((p[0] - cursor[0]).powi(2) + (p[1] - cursor[1]).powi(2)).sqrt() as f32;
                consider(d, Selection::Vertex(i as u32), 0.0);
            }
        }

        // ── Edges ──
        let edges = self.extract_edge_indices();
        for e in (0..edges.len()).step_by(2) {
            let a = edges[e] as usize;
            let b = edges[e + 1] as usize;
            if a >= self.vertices.len() || b >= self.vertices.len() {
                continue;
            }
            let pa = match project_to_pixel(self.vertices[a].position, &vp, viewport) {
                Some(p) => p,
                None => continue,
            };
            let pb = match project_to_pixel(self.vertices[b].position, &vp, viewport) {
                Some(p) => p,
                None => continue,
            };
            let d = point_to_segment_px(cursor, pa, pb);
            consider(d, Selection::Edge((e / 2) as u32), 1.0);
        }

        // ── Faces ──
        for (f, tri) in self.indices.chunks(3).enumerate() {
            if tri.len() != 3 {
                continue;
            }
            let p0 = match project_to_pixel(self.vertices[tri[0] as usize].position, &vp, viewport) {
                Some(p) => p,
                None => continue,
            };
            let p1 = match project_to_pixel(self.vertices[tri[1] as usize].position, &vp, viewport) {
                Some(p) => p,
                None => continue,
            };
            let p2 = match project_to_pixel(self.vertices[tri[2] as usize].position, &vp, viewport) {
                Some(p) => p,
                None => continue,
            };
            let d = point_to_triangle_px(cursor, p0, p1, p2);
            consider(d, Selection::Face(f as u32), 2.0);
        }

        best.map(|(_, s)| s).unwrap_or(Selection::None)
    }
}

/// Project a world position to screen pixels (physical).  Returns `None` when
/// the point is behind the camera.  Exposed for callers that draw their own
/// selection feedback.
pub fn project_to_pixel(world: [f32; 3], vp: &Mat4, viewport: &ViewportRect) -> Option<[f64; 2]> {
    let clip = vp * Vec4::new(world[0], world[1], world[2], 1.0);
    if clip.w <= 1e-6 {
        return None;
    }
    let ndc_x = (clip.x / clip.w) as f64;
    let ndc_y = (clip.y / clip.w) as f64;
    let px = viewport.min_x + (ndc_x * 0.5 + 0.5) * (viewport.max_x - viewport.min_x);
    let py = viewport.min_y + (1.0 - (ndc_y * 0.5 + 0.5)) * (viewport.max_y - viewport.min_y);
    Some([px, py])
}

// ── 2D distance helpers (screen space) ──

fn point_to_segment_px(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f32 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let apx = p[0] - a[0];
    let apy = p[1] - a[1];
    let len2 = abx * abx + aby * aby;
    let mut t = if len2 > 0.0 {
        (apx * abx + apy * aby) / len2
    } else {
        0.0
    };
    t = t.clamp(0.0, 1.0);
    let cx = a[0] + t * abx;
    let cy = a[1] + t * aby;
    ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt() as f32
}

fn point_to_triangle_px(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f32 {
    if point_in_triangle(p, a, b, c) {
        return 0.0;
    }
    point_to_segment_px(p, a, b)
        .min(point_to_segment_px(p, b, c))
        .min(point_to_segment_px(p, c, a))
}

fn point_in_triangle(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let d1 = sign(p, a, b);
    let d2 = sign(p, b, c);
    let d3 = sign(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

fn sign(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (p[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (p[1] - a[1])
}
