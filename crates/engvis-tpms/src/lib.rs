// engvis-tpms: Triply Periodic Minimal Surface (TPMS) domain for engvis.
//
// This crate is **pure math** — it depends only on `fidget-core`,
// `fidget-rhai`, and `fidget-jit`.  No GPU, no egui, no winit, and crucially
// **no dependency on `engvis-surface`** (primitives live there).  That keeps
// the TPMS math fully decoupled and reusable on its own.
//
// Extracted from `engvis-surface` (ADR-003) so TPMS-related data — the
// formula library, the per-family parameter bundle, the volume-fraction (C)
// solver, and the build `Morphology` — live in one bounded crate.

// ── Morphology ──────────────────────────────────────
//
// NOTE: `Morphology` (MinimalSurface / Shell / Skeletal) is a *general* mesh
// build concept used by `engvis-mesher` for every surface type, not just
// TPMS.  It is colocated here per ADR-003 so that `engvis-surface` (which
// defines the primitives) does not have to depend back on this crate; mesher
// and the CLI import `Morphology` from `engvis-tpms` directly.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Morphology {
    MinimalSurface,
    Shell,
    Skeletal,
}

// ── Gradient-field DSL ──────────────────────────────────

/// Spatial gradient-field mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientMode {
    None,
    Linear,
    Sigmoid,
    BoundaryDecay,
}

/// Spatial gradient-field parameters.
#[derive(Clone, Copy, Debug)]
pub struct GradientField {
    pub mode: GradientMode,
    pub axis: [f32; 3],
    pub base: f32,
    pub delta: f32,
    pub sharpness: f32,
    pub center: f32,
}

impl Default for GradientField {
    fn default() -> Self {
        Self {
            mode: GradientMode::None,
            axis: [1.0, 0.0, 0.0],
            base: 0.0,
            delta: 0.0,
            sharpness: 4.0,
            center: 0.0,
        }
    }
}

impl GradientField {
    pub fn to_tree(&self) -> fidget_core::context::Tree {
        use fidget_core::context::Tree as T;
        let nrm = (self.axis[0] * self.axis[0]
            + self.axis[1] * self.axis[1]
            + self.axis[2] * self.axis[2])
            .sqrt()
            .max(1e-6);
        let ax = self.axis[0] / nrm;
        let ay = self.axis[1] / nrm;
        let az = self.axis[2] / nrm;
        let u = T::x() * ax + T::y() * ay + T::z() * az;
        match self.mode {
            GradientMode::None => T::constant(self.base as f64),
            GradientMode::Linear => {
                let span = self.center.abs().max(1e-3);
                u * (self.delta / span) + self.base
            }
            GradientMode::Sigmoid => {
                let k = self.sharpness;
                let shifted = (u - self.center) * (-k);
                let denom = shifted.exp() + 1.0;
                denom.recip() * self.delta + self.base
            }
            GradientMode::BoundaryDecay => {
                let r = self.center.max(1e-3);
                let k = self.sharpness;
                let abs_u = (u.square() + 1e-6).sqrt();
                let dist = T::constant(r as f64) - abs_u;
                (dist * (-k)).exp() * self.delta + self.base
            }
        }
    }
}

// ── TPMS family ─────────────────────────────────────

/// The 11 built-in triply periodic minimal surface families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TpmsFamily {
    Gyroid,
    SchwarzP,
    SchwarzD,
    SchoenIwp,
    Neovius,
    FRD,
    Lidinoid,
    SplitP,
    FischerKochSY,
    FischerKochSCP,
    FischerKochSC,
}

impl TpmsFamily {
    /// Internal name used by the formula library.
    pub fn name(&self) -> &'static str {
        match self {
            TpmsFamily::Gyroid => "gyroid",
            TpmsFamily::SchwarzP => "schwarz-p",
            TpmsFamily::SchwarzD => "schwarz-d",
            TpmsFamily::SchoenIwp => "schoen-iwp",
            TpmsFamily::Neovius => "neovius",
            TpmsFamily::FRD => "f-rd",
            TpmsFamily::Lidinoid => "lidinoid",
            TpmsFamily::SplitP => "split-p",
            TpmsFamily::FischerKochSY => "fischer-koch-s",
            TpmsFamily::FischerKochSCP => "fischer-koch-cp",
            TpmsFamily::FischerKochSC => "fischer-koch-y",
        }
    }

    /// Display label for UI.
    pub fn label(&self) -> &'static str {
        match self {
            TpmsFamily::Gyroid => "Gyroid",
            TpmsFamily::SchwarzP => "Schwarz P",
            TpmsFamily::SchwarzD => "Schwarz D",
            TpmsFamily::SchoenIwp => "Schoen IWP",
            TpmsFamily::Neovius => "Neovius",
            TpmsFamily::FRD => "F-RD",
            TpmsFamily::Lidinoid => "Lidinoid",
            TpmsFamily::SplitP => "Split-P",
            TpmsFamily::FischerKochSY => "Fischer-Koch S",
            TpmsFamily::FischerKochSCP => "Fischer-Koch CP",
            TpmsFamily::FischerKochSC => "Fischer-Koch Y",
        }
    }

    /// Parse from internal name string.
    pub fn from_name(name: &str) -> Option<TpmsFamily> {
        match name {
            "gyroid" => Some(TpmsFamily::Gyroid),
            "schwarz-p" => Some(TpmsFamily::SchwarzP),
            "schwarz-d" => Some(TpmsFamily::SchwarzD),
            "schoen-iwp" => Some(TpmsFamily::SchoenIwp),
            "neovius" => Some(TpmsFamily::Neovius),
            "f-rd" => Some(TpmsFamily::FRD),
            "lidinoid" => Some(TpmsFamily::Lidinoid),
            "split-p" => Some(TpmsFamily::SplitP),
            "fischer-koch-s" => Some(TpmsFamily::FischerKochSY),
            "fischer-koch-cp" => Some(TpmsFamily::FischerKochSCP),
            "fischer-koch-y" => Some(TpmsFamily::FischerKochSC),
            _ => None,
        }
    }

    /// All built-in TPMS families.
    pub fn all() -> Vec<TpmsFamily> {
        vec![
            TpmsFamily::Gyroid,
            TpmsFamily::SchwarzP,
            TpmsFamily::SchwarzD,
            TpmsFamily::SchoenIwp,
            TpmsFamily::Neovius,
            TpmsFamily::FRD,
            TpmsFamily::Lidinoid,
            TpmsFamily::SplitP,
            TpmsFamily::FischerKochSY,
            TpmsFamily::FischerKochSCP,
            TpmsFamily::FischerKochSC,
        ]
    }
}

// ── TPMS surface (parameter bundle + tree construction) ──

/// A TPMS surface: a family plus all of its parameters.
///
/// This is the first-class "TPMS type" (ADR-003).  The binary keeps raw UI
/// state and assembles a `TpmsSurface` when it needs the fidget `Tree`.
#[derive(Clone, Debug)]
pub struct TpmsSurface {
    pub family: TpmsFamily,
    /// Intrinsic period k (spatial frequency within one unit cell).
    pub period: f32,
    /// Per-axis cell stacking counts [nx, ny, nz].
    pub cells: [u32; 3],
    /// Per-axis unit-cell edge lengths [Lx, Ly, Lz], default [1,1,1].
    pub cell_size: [f32; 3],
    /// Per-axis amplitude scaling [a, b, c], default [1,1,1].
    pub amplitude: [f32; 3],
    /// Global iso-value offset C, default 0 (advanced mode).
    pub offset: f32,
    /// Volume fraction φ ∈ [0,1]; 0.5 = symmetric minimal surface.
    pub vol_frac: f32,
    /// Shell wall thickness (used by the Shell morphology).
    pub thickness: f32,
    /// Secondary surface name for two-TPMS blending (None disables).
    pub blend_secondary: Option<String>,
    /// Blend weight spatial field (clipped to [0,1]).
    pub blend_weight_field: GradientField,
    /// Iso-value offset spatial gradient field (None mode falls back to `offset`).
    pub offset_field: GradientField,
}

impl TpmsSurface {
    /// Construct with family defaults.
    pub fn new(family: TpmsFamily) -> Self {
        let (period, cell_size, amplitude, offset, cells) = match family.name() {
            "gyroid" => (4.0, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0, [1, 1, 1]),
            "fischer-koch-s" | "fischer-koch-y" => {
                (2.0, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0, [1, 1, 1])
            }
            _ => (3.0, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0, [1, 1, 1]),
        };
        Self {
            family,
            period,
            cells,
            cell_size,
            amplitude,
            offset,
            vol_frac: 0.5,
            thickness: 0.1,
            blend_secondary: None,
            blend_weight_field: GradientField::default(),
            offset_field: GradientField::default(),
        }
    }

    /// Human-readable formula description for the active family.
    pub fn formula_desc(&self) -> &'static str {
        tpms_formula(self.family.name())
    }

    /// Build the fidget `Tree` for this surface.
    ///
    /// `rotation_axis` / `rotation_angle_rad` are general (apply to any
    /// surface) and passed in from the caller; TPMS does not own them.
    pub fn build_tree(
        &self,
        rotation_axis: [f32; 3],
        rotation_angle_rad: f32,
    ) -> fidget_core::context::Tree {
        use fidget_core::context::Tree as T;
        let k = self.period;
        let kx = k / self.cell_size[0].max(1e-6);
        let ky = k / self.cell_size[1].max(1e-6);
        let kz = k / self.cell_size[2].max(1e-6);

        let (xr, yr, zr) = apply_rotation(T::x(), T::y(), T::z(), rotation_axis, rotation_angle_rad);

        let x = xr * kx;
        let y = yr * ky;
        let z = zr * kz;
        let (a, b, c) = (self.amplitude[0], self.amplitude[1], self.amplitude[2]);

        let f1 = eval_tpms_formula(self.family.name(), x.clone(), y.clone(), z.clone(), a, b, c);

        let f_mixed = if let Some(secondary) = &self.blend_secondary {
            let f2 = eval_tpms_formula(secondary, x, y, z, a, b, c);
            let w_field = self.blend_weight_field.to_tree();
            let w_clip = w_field.max(T::constant(0.0)).min(T::constant(1.0));
            f1 * w_clip.clone() + f2 * (T::constant(1.0) - w_clip)
        } else {
            f1
        };

        let c_tree = self.offset_field.to_tree();
        f_mixed - c_tree
    }

    /// |grad f| estimate within a unit cell ≈ k (max gradient), amplified by
    /// amplitude.  Used to size the shell wall for the Shell morphology.
    pub fn shell_grad(&self) -> f32 {
        let k = self.period.max(1.0);
        let amp_max = self.amplitude[0].max(self.amplitude[1]).max(self.amplitude[2]);
        k * amp_max
    }

    /// Minimum feature size along the highest-frequency direction (period k).
    pub fn min_feature(&self) -> f32 {
        std::f32::consts::PI / self.period
    }

    /// Domain extent = cells × cell_size.
    pub fn domain_extent(&self) -> [f32; 3] {
        [
            self.cells[0] as f32 * self.cell_size[0],
            self.cells[1] as f32 * self.cell_size[1],
            self.cells[2] as f32 * self.cell_size[2],
        ]
    }

    /// Per-axis cell counts as floats (for the bounding-box wireframe).
    pub fn cells_f32(&self) -> [f32; 3] {
        [self.cells[0] as f32, self.cells[1] as f32, self.cells[2] as f32]
    }

    /// Solve the C offset such that the enclosed volume fraction equals
    /// `self.vol_frac`.  Only meaningful for the Skeletal morphology.
    pub fn solve_c(&self, tree: &fidget_core::context::Tree) -> f64 {
        solve_c_for_vol_frac(tree, self.vol_frac) as f64
    }
}

// ── TPMS formula evaluation ─────────────────────────────

pub fn eval_tpms_formula(
    name: &str,
    x: fidget_core::context::Tree,
    y: fidget_core::context::Tree,
    z: fidget_core::context::Tree,
    a: f32,
    b: f32,
    c: f32,
) -> fidget_core::context::Tree {
    match name {
        "gyroid" => {
            x.clone().sin() * y.clone().cos() * a
                + y.clone().sin() * z.clone().cos() * b
                + z.clone().sin() * x.clone().cos() * c
        }
        "schwarz-p" => x.cos() * a + y.cos() * b + z.cos() * c,
        "schwarz-d" => {
            let (sx, sy, sz) = (x.clone().sin(), y.clone().sin(), z.clone().sin());
            let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
            sx.clone() * sy.clone() * sz.clone() * 4.0
                + sx * cy.clone() * cz.clone() * a
                + cx.clone() * sy * cz.clone() * b
                + cx * cy * sz * c
        }
        "schoen-iwp" => {
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (c2x, c2y, c2z) = ((x * 2.0).cos(), (y * 2.0).cos(), (z * 2.0).cos());
            (cx.clone() * cy.clone() * a + cy.clone() * cz.clone() * b + cz.clone() * cx * c) * 2.0
                - (c2x.clone() * c2y.clone() * a + c2y.clone() * c2z.clone() * b + c2z.clone() * c2x * c)
        }
        "neovius" => {
            let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
            (cx.clone() * a + cy.clone() * b + cz.clone() * c) * 3.0 + cx * cy * cz * 4.0
        }
        "f-rd" => {
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (c2x, c2y, c2z) = ((x * 2.0).cos(), (y * 2.0).cos(), (z * 2.0).cos());
            cx * cy * cz * 4.0
                - (c2x.clone() * c2y.clone() * a + c2y * c2z.clone() * b + c2z * c2x * c)
        }
        "lidinoid" => {
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (s2x, s2y, s2z) =
                ((x.clone() * 2.0).sin(), (y.clone() * 2.0).sin(), (z.clone() * 2.0).sin());
            let (c2x, c2y, c2z) = ((x * 2.0).cos(), (y * 2.0).cos(), (z * 2.0).cos());
            (s2x.clone() * cy.clone() * s2z.clone() * a
                + s2y.clone() * cz.clone() * s2x.clone() * b
                + s2z * cx.clone() * s2y * c)
                * 0.5
                - (c2x.clone() * c2y.clone() * a + c2y.clone() * c2z.clone() * b + c2z.clone() * c2x * c)
                    * 0.5
                + 0.15
        }
        "split-p" => {
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (sx, sy, sz) = (x.clone().sin(), y.clone().sin(), z.clone().sin());
            let (s2x, s2y, s2z) =
                ((x.clone() * 2.0).sin(), (y.clone() * 2.0).sin(), (z.clone() * 2.0).sin());
            let (c2x, c2y, c2z) = ((x * 2.0).cos(), (y * 2.0).cos(), (z * 2.0).cos());
            (s2x.clone() * cy.clone() * sz.clone() * a
                + sx.clone() * s2y.clone() * cz.clone() * b
                + cx.clone() * sy.clone() * s2z * c)
                * 1.1
                - (c2x.clone() * c2y.clone() * a + c2y.clone() * c2z.clone() * b + c2z.clone() * c2x.clone() * c)
                    * 0.2
                - (c2x * a + c2y * b + c2z * c) * 0.4
        }
        "fischer-koch-s" => {
            let (sx, sy, sz) = (x.clone().sin(), y.clone().sin(), z.clone().sin());
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (c2x, c2y, c2z) = ((x * 2.0).cos(), (y * 2.0).cos(), (z * 2.0).cos());
            c2x * sy.clone() * cz.clone() * a
                + c2y * sz.clone() * cx.clone() * b
                + c2z * sx * cy * c
        }
        "fischer-koch-y" => {
            let sx = x.clone().sin();
            let sy = y.clone().sin();
            let cx = x.clone().cos();
            let cy = y.clone().cos();
            let cz = z.clone().cos();
            let s2x = (x * 2.0).sin();
            let s2y = (y * 2.0).sin();
            let s2z = (z.clone() * 2.0).sin();
            cx * cy * cz * 2.0 + s2x * sy * a + s2y * z.sin() * b + s2z * sx * c
        }
        "fischer-koch-cp" => {
            let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
            cx.clone() * a + cy.clone() * b + cz.clone() * c + cx * cy * cz * 4.0
        }
        _ => {
            x.clone().sin() * y.clone().cos() * a
                + y.clone().sin() * z.clone().cos() * b
                + z.clone().sin() * x.clone().cos() * c
        }
    }
}

pub fn tpms_formula(name: &str) -> &'static str {
    match name {
        "gyroid" => "sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = 0",
        "schwarz-p" => "cos(kx) + cos(ky) + cos(kz) = 0",
        "schwarz-d" => "sin(kx)sin(ky)sin(kz) + ... = 0",
        "schoen-iwp" => "2[cos(kx)cos(ky)+...] - [cos(2kx)+...] = 0",
        "neovius" => "3[cos(kx)+...] + 4cos(kx)cos(ky)cos(kz) = 0",
        "f-rd" => "4cos(kx)cos(ky)cos(kz) - [...] = 0",
        "lidinoid" => "(1/2)[...] - (1/2)[...] + 0.15 = 0",
        "split-p" => "1.1[...] - 0.2[...] - 0.4[...] = 0",
        "fischer-koch-s" => "cos(2kx)sin(ky)cos(kz) + ... = 0",
        "fischer-koch-y" => "2cos(kx)cos(ky)cos(kz) + ... = 0",
        "fischer-koch-cp" => "cos(kx)+cos(ky)+cos(kz) + 4cos(kx)cos(ky)cos(kz) = 0",
        _ => "(unknown)",
    }
}

// ── Rotation helper (moved from engvis-surface; TPMS-only consumer) ──

pub fn apply_rotation(
    x: fidget_core::context::Tree,
    y: fidget_core::context::Tree,
    z: fidget_core::context::Tree,
    axis: [f32; 3],
    angle: f32,
) -> (fidget_core::context::Tree, fidget_core::context::Tree, fidget_core::context::Tree) {
    if angle.abs() < 1e-6 {
        return (x, y, z);
    }
    let nrm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt().max(1e-6);
    let (ux, uy, uz) = (axis[0] / nrm, axis[1] / nrm, axis[2] / nrm);
    let ca = angle.cos();
    let sa = angle.sin();
    let one_ca = 1.0 - ca;
    let r00 = ca + ux * ux * one_ca;
    let r01 = ux * uy * one_ca - uz * sa;
    let r02 = ux * uz * one_ca + uy * sa;
    let r10 = uy * ux * one_ca + uz * sa;
    let r11 = ca + uy * uy * one_ca;
    let r12 = uy * uz * one_ca - ux * sa;
    let r20 = uz * ux * one_ca - uy * sa;
    let r21 = uz * uy * one_ca + ux * sa;
    let r22 = ca + uz * uz * one_ca;
    let xr = x.clone() * r00 + y.clone() * r01 + z.clone() * r02;
    let yr = x.clone() * r10 + y.clone() * r11 + z.clone() * r12;
    let zr = x * r20 + y * r21 + z * r22;
    (xr, yr, zr)
}

// ── Volume-fraction → C solver ─────────────────────────
//
// Bisection search for C such that vol_frac(C) = target_phi, where
// vol_frac(C) = |{p : f(p) < C}| / |domain| is monotone increasing in C.
// Sampling is on [-1,1]³ at N=48³ (~110k points; JIT eval ~1ms).

pub fn solve_c_for_vol_frac(
    tree: &fidget_core::context::Tree,
    target_phi: f32,
) -> f32 {
    use fidget_core::shape::Shape;
    use fidget_jit::JitFunction;

    let phi = target_phi.clamp(0.001, 0.999);

    let shape = Shape::<JitFunction>::from(tree.clone());
    let tape = shape.float_slice_tape(Default::default());
    let mut eval = Shape::<JitFunction>::new_float_slice_eval();

    let n = 48i32;
    let total = (n * n * n) as usize;
    let mut xs = Vec::with_capacity(total);
    let mut ys = Vec::with_capacity(total);
    let mut zs = Vec::with_capacity(total);
    for ix in 0..n {
        let x = -1.0 + 2.0 * ix as f32 / (n - 1) as f32;
        for iy in 0..n {
            let y = -1.0 + 2.0 * iy as f32 / (n - 1) as f32;
            for iz in 0..n {
                xs.push(x);
                ys.push(y);
                zs.push(-1.0 + 2.0 * iz as f32 / (n - 1) as f32);
            }
        }
    }

    let vals = match eval.eval(&tape, &xs, &ys, &zs) {
        Ok(r) => r.to_vec(),
        Err(_) => return 0.0,
    };

    let mut sorted: Vec<f32> = vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let target_count = (phi * total as f32).round() as usize;
    let idx = target_count.min(total - 1);
    sorted[idx]
}

// ── Tests ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_name_label_roundtrip() {
        assert_eq!(TpmsFamily::Gyroid.name(), "gyroid");
        assert_eq!(TpmsFamily::Gyroid.label(), "Gyroid");
        assert_eq!(TpmsFamily::from_name("gyroid"), Some(TpmsFamily::Gyroid));
        assert_eq!(TpmsFamily::FischerKochSY.name(), "fischer-koch-s");
        assert_eq!(TpmsFamily::FischerKochSCP.name(), "fischer-koch-cp");
        assert_eq!(TpmsFamily::FischerKochSC.name(), "fischer-koch-y");
    }

    #[test]
    fn family_from_name_invalid() {
        assert_eq!(TpmsFamily::from_name("nope"), None);
    }

    /// Every family `name()` must resolve to a dedicated formula description.
    /// Catches `name()` drifting from the `tpms_formula` arms.
    #[test]
    fn family_names_resolve_to_real_formula() {
        for f in TpmsFamily::all() {
            let desc = tpms_formula(f.name());
            assert_ne!(
                desc, "(unknown)",
                "{} has no formula description — name() likely drifted from tpms_formula arms",
                f.name()
            );
        }
    }

    #[test]
    fn surface_new_defaults() {
        let s = TpmsSurface::new(TpmsFamily::Gyroid);
        assert_eq!(s.period, 4.0);
        assert_eq!(s.vol_frac, 0.5);
        assert_eq!(s.family, TpmsFamily::Gyroid);
        // domain_extent uses cells × cell_size.
        assert_eq!(s.domain_extent(), [4.0, 4.0, 4.0]);
    }

    #[test]
    fn surface_min_feature_monotone() {
        let mut s = TpmsSurface::new(TpmsFamily::Gyroid);
        let f_lo = s.min_feature();
        s.period = 8.0;
        let f_hi = s.min_feature();
        assert!(f_hi < f_lo, "larger period → smaller min feature");
    }
}
