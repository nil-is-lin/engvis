//! Tests for TPMS (Triply Periodic Minimal Surfaces) formulas
//!
//! These tests verify that each formula can be created correctly.

use engvis_surface::{
    tpms_formula, build_tree, TreeParams,
    GradientField, GradientMode, Morphology,
};

/// All valid TPMS surface names
const VALID_TPMS_NAMES: &[&str] = &[
    "gyroid", "schwarz-p", "schwarz-d", "schoen-iwp",
    "neovius", "f-rd", "lidinoid", "split-p",
    "fischer-koch-s", "fischer-koch-y", "fischer-koch-cp",
];

/// Primitive surface names
const PRIMITIVE_NAMES: &[&str] = &["sphere", "torus"];

#[test]
fn tpms_formula_valid_names() {
    // All valid TPMS names should return a non-empty description string
    for name in VALID_TPMS_NAMES {
        let desc = tpms_formula(name);
        assert!(!desc.is_empty(), "{} should have a description", name);
        // The description should contain "= 0" (it's an implicit surface)
        assert!(desc.contains("= 0"), "{} description should contain '= 0'", name);
    }
}

#[test]
fn tpms_formula_invalid_name() {
    // Invalid names should return "(unknown)"
    let result = tpms_formula("invalid-surface-name");
    assert_eq!(result, "(unknown)", "invalid surface name should return '(unknown)'");
}

#[test]
fn tpms_formula_all_distinct() {
    // Each formula should have a distinct description
    let descs: Vec<&str> = VALID_TPMS_NAMES.iter()
        .map(|name| tpms_formula(name))
        .collect();
    
    for i in 0..descs.len() {
        for j in (i+1)..descs.len() {
            assert_ne!(descs[i], descs[j], 
                "{} and {} should have different descriptions", 
                VALID_TPMS_NAMES[i], VALID_TPMS_NAMES[j]);
        }
    }
}

#[test]
fn build_tree_primitives() {
    // Test that primitive surfaces build correctly
    for name in PRIMITIVE_NAMES {
        let params = TreeParams {
            name,
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
        
        // Should not panic
        let tree = build_tree(&params);
        // Tree should be non-empty (we can't easily check this, but it's a valid Tree)
        let _ = tree; // Just verify it compiles
    }
}

#[test]
fn build_tree_tpms() {
    // Test that TPMS surfaces build correctly with TreeParams
    for name in VALID_TPMS_NAMES {
        let params = TreeParams {
            name,
            sphere_radius: 0.8,
            torus_major_r: 0.6,
            torus_minor_r: 0.2,
            tpms_period: 4.0,
            tpms_cell_size: [1.0, 1.0, 1.0],
            tpms_amplitude: [1.0, 1.0, 1.0],
            tpms_offset: 0.0,
            tpms_cells: [4, 4, 4],
            rotation_axis: [0.0, 0.0, 1.0],
            rotation_angle: 0.0,
            blend_secondary: None,
            blend_weight_field: GradientField::default(),
            offset_field: GradientField::default(),
        };
        
        // Should not panic
        let tree = build_tree(&params);
        let _ = tree; // Just verify it compiles
    }
}

#[test]
fn build_tree_with_rotation() {
    // Test that rotation parameters work
    let params = TreeParams {
        name: "gyroid",
        sphere_radius: 0.8,
        torus_major_r: 0.6,
        torus_minor_r: 0.2,
        tpms_period: 4.0,
        tpms_cell_size: [1.0, 1.0, 1.0],
        tpms_amplitude: [1.0, 1.0, 1.0],
        tpms_offset: 0.0,
        tpms_cells: [4, 4, 4],
        rotation_axis: [1.0, 0.0, 0.0],  // Rotate around X axis
        rotation_angle: std::f32::consts::PI / 4.0,  // 45 degrees
        blend_secondary: None,
        blend_weight_field: GradientField::default(),
        offset_field: GradientField::default(),
    };
    
    let tree = build_tree(&params);
    let _ = tree;
}

#[test]
fn build_tree_with_blend() {
    // Test that blending two TPMS surfaces works
    let params = TreeParams {
        name: "gyroid",
        sphere_radius: 0.8,
        torus_major_r: 0.6,
        torus_minor_r: 0.2,
        tpms_period: 4.0,
        tpms_cell_size: [1.0, 1.0, 1.0],
        tpms_amplitude: [1.0, 1.0, 1.0],
        tpms_offset: 0.0,
        tpms_cells: [4, 4, 4],
        rotation_axis: [0.0, 0.0, 1.0],
        rotation_angle: 0.0,
        blend_secondary: Some("schwarz-p"),
        blend_weight_field: GradientField {
            mode: GradientMode::Linear,
            axis: [1.0, 0.0, 0.0],
            base: 0.0,
            delta: 1.0,
            sharpness: 4.0,
            center: 1.0,
        },
        offset_field: GradientField::default(),
    };
    
    let tree = build_tree(&params);
    let _ = tree;
}

#[test]
fn gradient_field_default() {
    let g = GradientField::default();
    assert!(matches!(g.mode, GradientMode::None));
    assert_eq!(g.axis, [1.0, 0.0, 0.0]);
    assert_eq!(g.base, 0.0);
    assert_eq!(g.delta, 0.0);  // Note: default delta is 0.0, not 1.0
    assert_eq!(g.sharpness, 4.0);
    assert_eq!(g.center, 0.0);
}

#[test]
fn gradient_field_to_tree_none() {
    let g = GradientField {
        mode: GradientMode::None,
        axis: [1.0, 0.0, 0.0],
        base: 0.5,
        delta: 0.0,
        sharpness: 4.0,
        center: 0.0,
    };
    
    let tree = g.to_tree();
    let _ = tree; // Should compile and produce a constant tree
}

#[test]
fn gradient_field_to_tree_linear() {
    let g = GradientField {
        mode: GradientMode::Linear,
        axis: [1.0, 0.0, 0.0],
        base: 0.0,
        delta: 1.0,
        sharpness: 4.0,
        center: 1.0,
    };
    
    let tree = g.to_tree();
    let _ = tree; // Should compile and produce a linear gradient tree
}

#[test]
fn morphology_variants() {
    // Just verify all morphology variants exist and can be matched
    let morphologies = [
        Morphology::MinimalSurface,
        Morphology::Shell,
        Morphology::Skeletal,
    ];
    
    for m in morphologies {
        match m {
            Morphology::MinimalSurface => {},
            Morphology::Shell => {},
            Morphology::Skeletal => {},
        }
    }
}

#[test]
fn tree_params_set_tpms_defaults() {
    // Test set_tpms_defaults method
    let mut params = TreeParams {
        name: "gyroid",
        sphere_radius: 0.8,
        torus_major_r: 0.6,
        torus_minor_r: 0.2,
        tpms_period: 99.0,  // This should be overridden
        tpms_cell_size: [99.0, 99.0, 99.0],  // This should be overridden
        tpms_amplitude: [99.0, 99.0, 99.0],  // This should be overridden
        tpms_offset: 99.0,  // This should be overridden
        tpms_cells: [99, 99, 99],  // This should be overridden
        rotation_axis: [99.0, 99.0, 99.0],  // This should be overridden
        rotation_angle: 99.0,  // This should be overridden
        blend_secondary: Some("test"),  // This should be overridden to None
        blend_weight_field: GradientField {
            mode: GradientMode::Linear,
            ..Default::default()
        },  // This should be overridden
        offset_field: GradientField {
            mode: GradientMode::Linear,
            ..Default::default()
        },  // This should be overridden
    };
    
    params.set_tpms_defaults("gyroid");
    
    // Verify defaults were set correctly
    assert_eq!(params.tpms_period, 4.0, "gyroid period should be 4.0");
    assert_eq!(params.tpms_cell_size, [1.0, 1.0, 1.0]);
    assert_eq!(params.tpms_amplitude, [1.0, 1.0, 1.0]);
    assert_eq!(params.tpms_offset, 0.0);
    assert_eq!(params.tpms_cells, [1, 1, 1]);
    assert_eq!(params.rotation_axis, [0.0, 0.0, 1.0]);
    assert_eq!(params.rotation_angle, 0.0);
    assert!(params.blend_secondary.is_none());
    assert!(matches!(params.blend_weight_field.mode, GradientMode::None));
    assert!(matches!(params.offset_field.mode, GradientMode::None));
}

#[test]
fn tree_params_set_tpms_defaults_different_surfaces() {
    // Test that different surfaces get different defaults
    let mut params = TreeParams {
        name: "test",
        sphere_radius: 0.8,
        torus_major_r: 0.6,
        torus_minor_r: 0.2,
        tpms_period: 0.0,
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
    
    params.set_tpms_defaults("gyroid");
    assert_eq!(params.tpms_period, 4.0);
    
    params.set_tpms_defaults("fischer-koch-s");
    assert_eq!(params.tpms_period, 2.0);
    
    params.set_tpms_defaults("fischer-koch-y");
    assert_eq!(params.tpms_period, 2.0);
    
    params.set_tpms_defaults("schwarz-p");
    assert_eq!(params.tpms_period, 3.0);  // Default for most surfaces
}

#[test]
fn apply_rotation_no_rotation() {
    use engvis_surface::apply_rotation;
    use fidget_core::context::Tree as T;
    
    let x = T::x();
    let y = T::y();
    let z = T::z();
    
    let (xr, yr, zr) = apply_rotation(x.clone(), y.clone(), z.clone(), [0.0, 0.0, 1.0], 0.0);
    
    // With zero rotation, should return the same trees
    // (We can't easily compare trees, but this should at least compile and not panic)
    let _ = (xr, yr, zr);
}

#[test]
fn eval_tpms_formula_returns_tree() {
    use engvis_surface::eval_tpms_formula;
    use fidget_core::context::Tree as T;
    
    let x = T::x();
    let y = T::y();
    let z = T::z();
    
    // Test a few formulas - they should return valid Trees
    let tree1 = eval_tpms_formula("gyroid", x.clone(), y.clone(), z.clone(), 1.0, 1.0, 1.0);
    let _ = tree1;
    
    let tree2 = eval_tpms_formula("schwarz-p", x.clone(), y.clone(), z.clone(), 1.0, 1.0, 1.0);
    let _ = tree2;
    
    let tree3 = eval_tpms_formula("neovius", x, y, z, 1.0, 1.0, 1.0);
    let _ = tree3;
}
