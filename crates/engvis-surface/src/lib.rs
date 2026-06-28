// engvis-surface: implicit surface definitions and TPMS formula library for engvis
//
// This crate is **pure math** — it depends only on `fidget-core` and
// `fidget-rhai`.  No GPU, no egui, no winit.

// ── SurfaceType enum ─────────────────────────────────
//
// Replaces string-based surface identification with a type-safe enum.
// This eliminates string matching and makes the code more maintainable.

/// Type-safe representation of all supported implicit surfaces.
#[derive(Clone, Debug, PartialEq)]
pub enum SurfaceType {
    // Primitive shapes
    Sphere,
    Torus,
    
    // TPMS (Triply Periodic Minimal Surfaces)
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
    
    // User-defined surface (Rhai script)
    Custom(String),
}

impl SurfaceType {
    /// Returns the internal name used by `build_tree` and `TreeParams`.
    pub fn name(&self) -> &str {
        match self {
            SurfaceType::Sphere => "sphere",
            SurfaceType::Torus => "torus",
            SurfaceType::Gyroid => "gyroid",
            SurfaceType::SchwarzP => "schwarz-p",
            SurfaceType::SchwarzD => "schwarz-d",
            SurfaceType::SchoenIwp => "schoen-iwp",
            SurfaceType::Neovius => "neovius",
            SurfaceType::FRD => "f-rd",
            SurfaceType::Lidinoid => "lidinoid",
            SurfaceType::SplitP => "split-p",
            SurfaceType::FischerKochSY => "fischer-koch-s-y",
            SurfaceType::FischerKochSCP => "fischer-koch-s-cp",
            SurfaceType::FischerKochSC => "fischer-koch-s-c",
            SurfaceType::Custom(_) => "custom",
        }
    }
    
    /// Returns the display label for UI.
    pub fn label(&self) -> &str {
        match self {
            SurfaceType::Sphere => "Sphere",
            SurfaceType::Torus => "Torus",
            SurfaceType::Gyroid => "Gyroid",
            SurfaceType::SchwarzP => "Schwarz P",
            SurfaceType::SchwarzD => "Schwarz D",
            SurfaceType::SchoenIwp => "Schoen IWP",
            SurfaceType::Neovius => "Neovius",
            SurfaceType::FRD => "F-RD",
            SurfaceType::Lidinoid => "Lidinoid",
            SurfaceType::SplitP => "Split-P",
            SurfaceType::FischerKochSY => "Fischer-Koch S-Y",
            SurfaceType::FischerKochSCP => "Fischer-Koch S-CP",
            SurfaceType::FischerKochSC => "Fischer-Koch S-C",
            SurfaceType::Custom(_) => "Custom",
        }
    }
    
    /// Returns true if this is a TPMS surface (needs special mesh parameters).
    pub fn is_tpms(&self) -> bool {
        matches!(self, 
            SurfaceType::Gyroid |
            SurfaceType::SchwarzP |
            SurfaceType::SchwarzD |
            SurfaceType::SchoenIwp |
            SurfaceType::Neovius |
            SurfaceType::FRD |
            SurfaceType::Lidinoid |
            SurfaceType::SplitP |
            SurfaceType::FischerKochSY |
            SurfaceType::FischerKochSCP |
            SurfaceType::FischerKochSC
        )
    }
    
    /// Returns true if this is a primitive shape (sphere, torus).
    pub fn is_primitive(&self) -> bool {
        matches!(self, SurfaceType::Sphere | SurfaceType::Torus)
    }
    
    /// Returns the default TreeParams for this surface type.
    pub fn default_params(&self) -> TreeParams<'_> {
        let mut params = TreeParams {
            name: self.name(),
            sphere_radius: 0.8,
            torus_major_r: 0.6,
            torus_minor_r: 0.2,
            tpms_period: 4.0,
            tpms_cell_size: [1.0, 1.0, 1.0],
            tpms_amplitude: [1.0, 1.0, 1.0],
            tpms_offset: 0.0,
            tpms_cells: [1, 1, 1],
            rotation_axis: [0.0, 0.0, 1.0],
            rotation_angle: 0.0,
            blend_secondary: None,
            blend_weight_field: GradientField::default(),
            offset_field: GradientField::default(),
        };
        
        if self.is_tpms() {
            params.set_tpms_defaults(self.name());
        }
        
        params
    }
    
    /// Returns all built-in surfaces (excluding Custom).
    pub fn builtin_surfaces() -> Vec<SurfaceType> {
        vec![
            SurfaceType::Sphere,
            SurfaceType::Torus,
            SurfaceType::Gyroid,
            SurfaceType::SchwarzP,
            SurfaceType::SchwarzD,
            SurfaceType::SchoenIwp,
            SurfaceType::Neovius,
            SurfaceType::FRD,
            SurfaceType::Lidinoid,
            SurfaceType::SplitP,
            SurfaceType::FischerKochSY,
            SurfaceType::FischerKochSCP,
            SurfaceType::FischerKochSC,
        ]
    }
    
    /// Returns all primitive surfaces.
    pub fn primitive_surfaces() -> Vec<SurfaceType> {
        vec![SurfaceType::Sphere, SurfaceType::Torus]
    }
    
    /// Returns all TPMS surfaces.
    pub fn tpms_surfaces() -> Vec<SurfaceType> {
        vec![
            SurfaceType::Gyroid,
            SurfaceType::SchwarzP,
            SurfaceType::SchwarzD,
            SurfaceType::SchoenIwp,
            SurfaceType::Neovius,
            SurfaceType::FRD,
            SurfaceType::Lidinoid,
            SurfaceType::SplitP,
            SurfaceType::FischerKochSY,
            SurfaceType::FischerKochSCP,
            SurfaceType::FischerKochSC,
        ]
    }
    
    /// Parse from internal name string.
    pub fn from_name(name: &str) -> Option<SurfaceType> {
        match name {
            "sphere" => Some(SurfaceType::Sphere),
            "torus" => Some(SurfaceType::Torus),
            "gyroid" => Some(SurfaceType::Gyroid),
            "schwarz-p" => Some(SurfaceType::SchwarzP),
            "schwarz-d" => Some(SurfaceType::SchwarzD),
            "schoen-iwp" => Some(SurfaceType::SchoenIwp),
            "neovius" => Some(SurfaceType::Neovius),
            "f-rd" => Some(SurfaceType::FRD),
            "lidinoid" => Some(SurfaceType::Lidinoid),
            "split-p" => Some(SurfaceType::SplitP),
            "fischer-koch-s-y" => Some(SurfaceType::FischerKochSY),
            "fischer-koch-s-cp" => Some(SurfaceType::FischerKochSCP),
            "fischer-koch-s-c" => Some(SurfaceType::FischerKochSC),
            _ => None,
        }
    }
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
        let nrm = (self.axis[0]*self.axis[0]
                    + self.axis[1]*self.axis[1]
                    + self.axis[2]*self.axis[2]).sqrt().max(1e-6);
        let ax = self.axis[0]/nrm;
        let ay = self.axis[1]/nrm;
        let az = self.axis[2]/nrm;
        let u = T::x()*ax + T::y()*ay + T::z()*az;
        match self.mode {
            GradientMode::None => T::constant(self.base as f64),
            GradientMode::Linear => {
                let span = self.center.abs().max(1e-3);
                u*(self.delta / span) + self.base
            }
            GradientMode::Sigmoid => {
                let k = self.sharpness;
                let shifted = (u - self.center)*(-k);
                let denom = shifted.exp() + 1.0;
                denom.recip()*self.delta + self.base
            }
            GradientMode::BoundaryDecay => {
                let r = self.center.max(1e-3);
                let k = self.sharpness;
                let abs_u = (u.square() + 1e-6).sqrt();
                let dist = T::constant(r as f64) - abs_u;
                (dist*(-k)).exp()*self.delta + self.base
            }
        }
    }
}

// ── Morphology ──────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Morphology {
    MinimalSurface,
    Shell,
    Skeletal,
}

// ── Tree parameters ───────────────────────────────────

#[derive(Clone, Debug)]
pub struct TreeParams<'a> {
    pub name: &'a str,
    pub sphere_radius: f32,
    pub torus_major_r: f32,
    pub torus_minor_r: f32,
    pub tpms_period: f32,
    pub tpms_cell_size: [f32; 3],
    pub tpms_amplitude: [f32; 3],
    pub tpms_offset: f32,
    pub tpms_cells: [u32; 3],
    pub rotation_axis: [f32; 3],
    pub rotation_angle: f32,
    pub blend_secondary: Option<&'a str>,
    pub blend_weight_field: GradientField,
    pub offset_field: GradientField,
}

impl<'a> TreeParams<'a> {
    pub fn set_tpms_defaults(&mut self, name: &str) {
        self.tpms_period = match name {
            "gyroid" => 4.0,
            "fischer-koch-s" | "fischer-koch-y" => 2.0,
            _ => 3.0,
        };
        self.tpms_cell_size = [1.0, 1.0, 1.0];
        self.tpms_amplitude = [1.0, 1.0, 1.0];
        self.tpms_offset = 0.0;
        self.tpms_cells = [1, 1, 1];
        self.rotation_axis = [0.0, 0.0, 1.0];
        self.rotation_angle = 0.0;
        self.blend_secondary = None;
        self.blend_weight_field = GradientField::default();
        self.offset_field = GradientField::default();
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
        "schwarz-p" => x.cos()*a + y.cos()*b + z.cos()*c,
        "schwarz-d" => {
            let (sx, sy, sz) = (x.clone().sin(), y.clone().sin(), z.clone().sin());
            let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
            sx.clone()*sy.clone()*sz.clone()*4.0
                + sx*cy.clone()*cz.clone()*a
                + cx.clone()*sy*cz.clone()*b
                + cx*cy*sz*c
        }
        "schoen-iwp" => {
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (c2x, c2y, c2z) = ((x*2.0).cos(), (y*2.0).cos(), (z*2.0).cos());
            (cx.clone()*cy.clone()*a + cy.clone()*cz.clone()*b + cz.clone()*cx*c)*2.0
                - (c2x.clone()*c2y.clone()*a + c2y.clone()*c2z.clone()*b + c2z.clone()*c2x*c)
        }
        "neovius" => {
            let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
            (cx.clone()*a + cy.clone()*b + cz.clone()*c)*3.0 + cx*cy*cz*4.0
        }
        "f-rd" => {
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (c2x, c2y, c2z) = ((x*2.0).cos(), (y*2.0).cos(), (z*2.0).cos());
            cx*cy*cz*4.0
                - (c2x.clone()*c2y.clone()*a + c2y*c2z.clone()*b + c2z*c2x*c)
        }
        "lidinoid" => {
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (s2x, s2y, s2z) =
                ((x.clone()*2.0).sin(), (y.clone()*2.0).sin(), (z.clone()*2.0).sin());
            let (c2x, c2y, c2z) = ((x*2.0).cos(), (y*2.0).cos(), (z*2.0).cos());
            (s2x.clone()*cy.clone()*s2z.clone()*a
                + s2y.clone()*cz.clone()*s2x.clone()*b
                + s2z*cx.clone()*s2y*c)*0.5
                - (c2x.clone()*c2y.clone()*a + c2y.clone()*c2z.clone()*b + c2z.clone()*c2x*c)*0.5
                + 0.15
        }
        "split-p" => {
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (sx, sy, sz) = (x.clone().sin(), y.clone().sin(), z.clone().sin());
            let (s2x, s2y, s2z) =
                ((x.clone()*2.0).sin(), (y.clone()*2.0).sin(), (z.clone()*2.0).sin());
            let (c2x, c2y, c2z) = ((x*2.0).cos(), (y*2.0).cos(), (z*2.0).cos());
            (s2x.clone()*cy.clone()*sz.clone()*a
                + sx.clone()*s2y.clone()*cz.clone()*b
                + cx.clone()*sy.clone()*s2z*c)*1.1
                - (c2x.clone()*c2y.clone()*a + c2y.clone()*c2z.clone()*b + c2z.clone()*c2x.clone()*c)*0.2
                - (c2x*a + c2y*b + c2z*c)*0.4
        }
        "fischer-koch-s" => {
            let (sx, sy, sz) = (x.clone().sin(), y.clone().sin(), z.clone().sin());
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (c2x, c2y, c2z) = ((x*2.0).cos(), (y*2.0).cos(), (z*2.0).cos());
            c2x*sy.clone()*cz.clone()*a
                + c2y*sz.clone()*cx.clone()*b
                + c2z*sx*cy*c
        }
        "fischer-koch-y" => {
            let sx = x.clone().sin();
            let sy = y.clone().sin();
            let cx = x.clone().cos();
            let cy = y.clone().cos();
            let cz = z.clone().cos();
            let s2x = (x*2.0).sin();
            let s2y = (y*2.0).sin();
            let s2z = (z.clone()*2.0).sin();
            cx*cy*cz*2.0 + s2x*sy*a + s2y*z.sin()*b + s2z*sx*c
        }
        "fischer-koch-cp" => {
            let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
            cx.clone()*a + cy.clone()*b + cz.clone()*c + cx*cy*cz*4.0
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
        "gyroid"          => "sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = 0",
        "schwarz-p"       => "cos(kx) + cos(ky) + cos(kz) = 0",
        "schwarz-d"       => "sin(kx)sin(ky)sin(kz) + ... = 0",
        "schoen-iwp"      => "2[cos(kx)cos(ky)+...] - [cos(2kx)+...] = 0",
        "neovius"         => "3[cos(kx)+...] + 4cos(kx)cos(ky)cos(kz) = 0",
        "f-rd"            => "4cos(kx)cos(ky)cos(kz) - [...] = 0",
        "lidinoid"        => "(1/2)[...] - (1/2)[...] + 0.15 = 0",
        "split-p"         => "1.1[...] - 0.2[...] - 0.4[...] = 0",
        "fischer-koch-s"  => "cos(2kx)sin(ky)cos(kz) + ... = 0",
        "fischer-koch-y"  => "2cos(kx)cos(ky)cos(kz) + ... = 0",
        "fischer-koch-cp" => "cos(kx)+cos(ky)+cos(kz) + 4cos(kx)cos(ky)cos(kz) = 0",
        _ => "(unknown)",
    }
}

// ── Tree construction ─────────────────────────────────

pub fn build_tree(p: &TreeParams) -> fidget_core::context::Tree {
    use fidget_core::context::Tree as T;
    let k = p.tpms_period;
    let kx = k / p.tpms_cell_size[0].max(1e-6);
    let ky = k / p.tpms_cell_size[1].max(1e-6);
    let kz = k / p.tpms_cell_size[2].max(1e-6);

    let (xr, yr, zr) =
        apply_rotation(T::x(), T::y(), T::z(), p.rotation_axis, p.rotation_angle);

    match p.name {
        "sphere" => {
            return (T::x().square() + T::y().square() + T::z().square()).sqrt() - p.sphere_radius;
        }
        "torus" => {
            let major = T::x().square() + T::y().square();
            return (major.sqrt() - p.torus_major_r).square()
                + T::z().square()
                - p.torus_minor_r*p.torus_minor_r;
        }
        _ => {}
    }

    let x = xr * kx;
    let y = yr * ky;
    let z = zr * kz;
    let (a, b, c) = (p.tpms_amplitude[0], p.tpms_amplitude[1], p.tpms_amplitude[2]);

    let f1 = eval_tpms_formula(p.name, x.clone(), y.clone(), z.clone(), a, b, c);

    let f_mixed = if let Some(secondary) = p.blend_secondary {
        let f2 = eval_tpms_formula(secondary, x, y, z, a, b, c);
        let w_field = p.blend_weight_field.to_tree();
        let w_clip = w_field.max(T::constant(0.0)).min(T::constant(1.0));
        f1*w_clip.clone() + f2*(T::constant(1.0)-w_clip)
    } else {
        f1
    };

    let c_tree = p.offset_field.to_tree();
    f_mixed - c_tree
}

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
    let nrm = (axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]).sqrt().max(1e-6);
    let (ux, uy, uz) = (axis[0]/nrm, axis[1]/nrm, axis[2]/nrm);
    let ca = angle.cos();
    let sa = angle.sin();
    let one_ca = 1.0 - ca;
    let r00 = ca + ux*ux*one_ca;
    let r01 = ux*uy*one_ca - uz*sa;
    let r02 = ux*uz*one_ca + uy*sa;
    let r10 = uy*ux*one_ca + uz*sa;
    let r11 = ca + uy*uy*one_ca;
    let r12 = uy*uz*one_ca - ux*sa;
    let r20 = uz*ux*one_ca - uy*sa;
    let r21 = uz*uy*one_ca + ux*sa;
    let r22 = ca + uz*uz*one_ca;
    let xr = x.clone()*r00 + y.clone()*r01 + z.clone()*r02;
    let yr = x.clone()*r10 + y.clone()*r11 + z.clone()*r12;
    let zr = x*r20 + y*r21 + z*r22;
    (xr, yr, zr)
}

pub fn build_tree_from_rhai(src: &str) -> Result<fidget_core::context::Tree, String> {
    let engine = fidget_rhai::engine();
    let tree: fidget_core::context::Tree = engine
        .eval(src)
        .map_err(|e| format!("{e}"))?;
    Ok(tree)
}

// ── Tests ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_type_name() {
        assert_eq!(SurfaceType::Gyroid.name(), "gyroid");
        assert_eq!(SurfaceType::Sphere.name(), "sphere");
        assert_eq!(SurfaceType::Custom("test".to_string()).name(), "custom");
    }

    #[test]
    fn surface_type_label() {
        assert_eq!(SurfaceType::Gyroid.label(), "Gyroid");
        assert_eq!(SurfaceType::SchwarzP.label(), "Schwarz P");
        assert_eq!(SurfaceType::FRD.label(), "F-RD");
    }

    #[test]
    fn surface_type_is_tpms() {
        assert!(SurfaceType::Gyroid.is_tpms());
        assert!(SurfaceType::SchwarzP.is_tpms());
        assert!(!SurfaceType::Sphere.is_tpms());
        assert!(!SurfaceType::Torus.is_tpms());
    }

    #[test]
    fn surface_type_from_name() {
        assert_eq!(SurfaceType::from_name("gyroid"), Some(SurfaceType::Gyroid));
        assert_eq!(SurfaceType::from_name("sphere"), Some(SurfaceType::Sphere));
        assert_eq!(SurfaceType::from_name("invalid"), None);
    }

    #[test]
    fn surface_type_builtin_surfaces() {
        let surfaces = SurfaceType::builtin_surfaces();
        assert!(surfaces.contains(&SurfaceType::Gyroid));
        assert!(surfaces.contains(&SurfaceType::Sphere));
        assert!(!surfaces.iter().any(|s| matches!(s, SurfaceType::Custom(_))));
    }

    #[test]
    fn surface_type_tpms_surfaces() {
        let tpms = SurfaceType::tpms_surfaces();
        assert!(tpms.contains(&SurfaceType::Gyroid));
        assert!(!tpms.contains(&SurfaceType::Sphere));
    }
}
