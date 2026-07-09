// engvis-surface: primitive implicit surface definitions for engvis.
//
// This crate is **pure math** — it depends only on `fidget-core` and
// `fidget-rhai`.  TPMS surfaces, their formula library, the volume-fraction
// (C) solver, and the build `Morphology` live in the separate `engvis-tpms`
// crate (ADR-003), so this crate stays small and TPMS-free.

// ── SurfaceType enum ─────────────────────────────────
//
// Primitive shapes + user-defined (Rhai) surfaces.  TPMS families moved to
// `engvis_tpms::TpmsFamily`.

/// Type-safe representation of the supported primitive / custom surfaces.
#[derive(Clone, Debug, PartialEq)]
pub enum SurfaceType {
    // Primitive shapes
    Sphere,
    Torus,

    // User-defined surface (Rhai script)
    Custom(String),
}

impl SurfaceType {
    /// Returns the internal name used by `build_tree`.
    pub fn name(&self) -> &str {
        match self {
            SurfaceType::Sphere => "sphere",
            SurfaceType::Torus => "torus",
            SurfaceType::Custom(_) => "custom",
        }
    }

    /// Returns the display label for UI.
    pub fn label(&self) -> &str {
        match self {
            SurfaceType::Sphere => "Sphere",
            SurfaceType::Torus => "Torus",
            SurfaceType::Custom(_) => "Custom",
        }
    }

    /// True if this is a primitive shape (sphere, torus).
    pub fn is_primitive(&self) -> bool {
        matches!(self, SurfaceType::Sphere | SurfaceType::Torus)
    }

    /// Returns all built-in surfaces (excluding Custom).
    pub fn builtin_surfaces() -> Vec<SurfaceType> {
        vec![SurfaceType::Sphere, SurfaceType::Torus]
    }

    /// Returns all primitive surfaces.
    pub fn primitive_surfaces() -> Vec<SurfaceType> {
        vec![SurfaceType::Sphere, SurfaceType::Torus]
    }

    /// Parse from internal name string.
    pub fn from_name(name: &str) -> Option<SurfaceType> {
        match name {
            "sphere" => Some(SurfaceType::Sphere),
            "torus" => Some(SurfaceType::Torus),
            _ => None,
        }
    }
}

// ── Tree parameters (primitives only) ─────────────────

#[derive(Clone, Debug, Default)]
pub struct TreeParams<'a> {
    pub name: &'a str,
    pub sphere_radius: f32,
    pub torus_major_r: f32,
    pub torus_minor_r: f32,
}

// ── Tree construction ─────────────────────────────────

pub fn build_tree(p: &TreeParams) -> fidget_core::context::Tree {
    use fidget_core::context::Tree as T;
    match p.name {
        "sphere" => (T::x().square() + T::y().square() + T::z().square()).sqrt() - p.sphere_radius,
        "torus" => {
            let major = T::x().square() + T::y().square();
            (major.sqrt() - p.torus_major_r).square() + T::z().square()
                - p.torus_minor_r * p.torus_minor_r
        }
        // Custom surfaces are built via `build_tree_from_rhai`, never here.
        // Fall back to a unit sphere so a bad name can't panic.
        _ => (T::x().square() + T::y().square() + T::z().square()).sqrt() - 0.8,
    }
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
        assert_eq!(SurfaceType::Sphere.name(), "sphere");
        assert_eq!(SurfaceType::Torus.name(), "torus");
        assert_eq!(SurfaceType::Custom("test".to_string()).name(), "custom");
    }

    #[test]
    fn surface_type_label() {
        assert_eq!(SurfaceType::Sphere.label(), "Sphere");
        assert_eq!(SurfaceType::Torus.label(), "Torus");
        assert_eq!(SurfaceType::Custom("x".into()).label(), "Custom");
    }

    #[test]
    fn surface_type_is_primitive() {
        assert!(SurfaceType::Sphere.is_primitive());
        assert!(SurfaceType::Torus.is_primitive());
        assert!(!SurfaceType::Custom("x".into()).is_primitive());
    }

    #[test]
    fn surface_type_from_name() {
        assert_eq!(SurfaceType::from_name("sphere"), Some(SurfaceType::Sphere));
        assert_eq!(SurfaceType::from_name("torus"), Some(SurfaceType::Torus));
        assert_eq!(SurfaceType::from_name("invalid"), None);
    }

    #[test]
    fn surface_type_builtin_surfaces() {
        let surfaces = SurfaceType::builtin_surfaces();
        assert!(surfaces.contains(&SurfaceType::Sphere));
        assert!(surfaces.contains(&SurfaceType::Torus));
        assert!(!surfaces.iter().any(|s| matches!(s, SurfaceType::Custom(_))));
    }

    #[test]
    fn build_tree_sphere_and_torus() {
        let sphere = build_tree(&TreeParams {
            name: "sphere",
            sphere_radius: 0.8,
            ..Default::default()
        });
        let torus = build_tree(&TreeParams {
            name: "torus",
            torus_major_r: 0.6,
            torus_minor_r: 0.2,
            ..Default::default()
        });
        // Trees build without panicking; just assert they are non-empty ops.
        assert!(format!("{:?}", sphere).len() > 0);
        assert!(format!("{:?}", torus).len() > 0);
    }
}
