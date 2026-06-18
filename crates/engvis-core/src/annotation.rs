/// Annotation elements for 3D scientific visualization:
/// coordinate axes (colored arrows), tick marks, and a colorbar.
///
/// All elements are built as `crate::Mesh` values suitable for
/// adding to a `Scene` via `Scene::single_mesh()` or manual node insertion.
use crate::{
    Mesh, MeshVertex, SubMesh, PbrMaterial, Aabb,
};
use glam::Vec3;

// ── Coordinate axes ───────────────────────────────────────────

/// One labeled axis: body cylinder + cone arrowhead.
/// Returns (mesh, material) — use Scene::single_mesh() to add to scene.
pub fn create_axis_arrow(
    axis: Vec3,
    length: f32,
    body_radius: f32,
    head_radius: f32,
    head_length: f32,
    color: [f32; 3],
) -> (Mesh, PbrMaterial) {
    let dir = axis.normalize();
    let up = if axis.x.abs() < 0.99 { Vec3::X } else { Vec3::Y };
    let p1 = dir.cross(up).normalize();
    let p2 = dir.cross(p1).normalize();

    let body_end = dir * (length - head_length);
    let segments = 8usize;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // ── Body: n-sided prism from origin to body_end ──
    let ring0_start = positions.len() as u32;
    for i in 0..segments {
        let angle = (i as f32) * std::f32::consts::TAU / segments as f32;
        let offset = (p1 * angle.cos() + p2 * angle.sin()) * body_radius;
        positions.push(offset.into());
    }
    let ring1_start = positions.len() as u32;
    for i in 0..segments {
        let angle = (i as f32) * std::f32::consts::TAU / segments as f32;
        let offset = (p1 * angle.cos() + p2 * angle.sin()) * body_radius;
        positions.push((body_end + offset).into());
    }
    for i in 0..segments {
        let j = (i + 1) % segments;
        let a = ring0_start + i as u32;
        let b = ring0_start + j as u32;
        let c = ring1_start + i as u32;
        let d = ring1_start + j as u32;
        indices.extend_from_slice(&[a, b, c, b, d, c]);
    }

    // ── Head: cone from body_end to tip ──
    let cone_base_start = positions.len() as u32;
    for i in 0..segments {
        let angle = (i as f32) * std::f32::consts::TAU / segments as f32;
        let offset = (p1 * angle.cos() + p2 * angle.sin()) * head_radius;
        positions.push((body_end + offset).into());
    }
    let cone_tip = positions.len() as u32;
    positions.push((dir * length).into());
    for i in 0..segments {
        let j = (i + 1) % segments;
        let a = cone_base_start + i as u32;
        let b = cone_base_start + j as u32;
        indices.extend_from_slice(&[a, b, cone_tip]);
    }

    let mesh = Mesh::from_triangles(
        format!("Axis_{:?}", axis.to_array()),
        &positions,
        &indices,
    );

    let material = PbrMaterial {
        name: format!("AxisMat_{:?}", color),
        albedo: [color[0], color[1], color[2], 1.0],
        roughness: 0.5,
        metallic: 0.0,
        ..Default::default()
    };

    (mesh, material)
}

/// Create all three coordinate axes (X: red, Y: green, Z: blue).
/// Returns (Vec<Mesh>, Vec<PbrMaterial>) with 3 entries each.
pub fn create_coordinate_axes(
    axis_length: f32,
    body_radius: f32,
    head_radius: f32,
    head_length: f32,
) -> (Vec<Mesh>, Vec<PbrMaterial>) {
    let mut meshes = Vec::new();
    let mut materials = Vec::new();

    let (x_mesh, x_mat) = create_axis_arrow(
        Vec3::X, axis_length, body_radius, head_radius, head_length,
        [0.9, 0.15, 0.15],
    );
    meshes.push(x_mesh);
    materials.push(x_mat);

    let (y_mesh, y_mat) = create_axis_arrow(
        Vec3::Y, axis_length, body_radius, head_radius, head_length,
        [0.15, 0.8, 0.15],
    );
    meshes.push(y_mesh);
    materials.push(y_mat);

    let (z_mesh, z_mat) = create_axis_arrow(
        Vec3::Z, axis_length, body_radius, head_radius, head_length,
        [0.15, 0.25, 0.9],
    );
    meshes.push(z_mesh);
    materials.push(z_mat);

    (meshes, materials)
}

// ── Tick marks ────────────────────────────────────────────────

/// Create tick marks along an axis direction.
/// Returns (mesh, material) for thin quad ticks perpendicular to the axis.
pub fn create_axis_ticks(
    axis: Vec3,
    total_length: f32,
    tick_size: f32,
    minor_tick_size: f32,
    major_step: f32,
    minor_step: f32,
    color: [f32; 3],
) -> (Mesh, PbrMaterial) {
    let dir = axis.normalize();
    let up = if axis.x.abs() < 0.99 { Vec3::X } else { Vec3::Y };
    let perp1 = dir.cross(up).normalize();
    let perp2 = dir.cross(perp1).normalize();

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let n = (total_length / minor_step) as i32;
    for i in 1..=n {
        let t = i as f32 * minor_step;
        if t > total_length + 0.001 { break; }
        let is_major = (i * minor_step as i32) % (major_step as i32) == 0;
        let size = if is_major { tick_size } else { minor_tick_size };
        let base = dir * t;
        let h = size * 0.5;
        // tiny thickness along perp2 so the tick renders on both sides
        let thick = tick_size * 0.3;
        let v0: [f32; 3] = (base - perp1 * h - perp2 * thick).into();
        let v1: [f32; 3] = (base + perp1 * h - perp2 * thick).into();
        let v2: [f32; 3] = (base + perp1 * h + perp2 * thick).into();
        let v3: [f32; 3] = (base - perp1 * h + perp2 * thick).into();

        let idx = positions.len() as u32;
        positions.extend_from_slice(&[v0, v1, v2, v3]);
        indices.extend_from_slice(&[idx, idx + 1, idx + 2, idx, idx + 2, idx + 3]);
    }

    let mesh = Mesh::from_triangles(
        format!("Ticks_{:?}", axis.to_array()),
        &positions,
        &indices,
    );

    let material = PbrMaterial {
        name: format!("TickMat_{:?}", color),
        albedo: [color[0], color[1], color[2], 1.0],
        roughness: 0.6,
        metallic: 0.0,
        ..Default::default()
    };

    (mesh, material)
}

/// Create tick marks for all three axes.
pub fn create_all_ticks(
    axis_length: f32,
    head_length: f32,
    tick_size: f32,
    minor_tick_size: f32,
    major_step: f32,
    minor_step: f32,
) -> (Vec<Mesh>, Vec<PbrMaterial>) {
    let body_len = axis_length - head_length;
    let mut meshes = Vec::new();
    let mut materials = Vec::new();

    for (axis, color) in [
        (Vec3::X, [0.7, 0.1, 0.1_f32]),
        (Vec3::Y, [0.1, 0.6, 0.1]),
        (Vec3::Z, [0.1, 0.2, 0.7]),
    ] {
        let (m, mat) = create_axis_ticks(
            axis, body_len, tick_size, minor_tick_size,
            major_step, minor_step, color,
        );
        meshes.push(m);
        materials.push(mat);
    }

    (meshes, materials)
}

// ── Colorbar ──────────────────────────────────────────────────

/// Result of building a colorbar.
pub struct ColorbarGeometry {
    pub mesh: Mesh,
    pub materials: Vec<PbrMaterial>,
}

/// Build a colorbar as a flat vertical rectangle with `segments` sub-meshes,
/// each colored by looking up the colormap at the segment's height.
///
/// * `bottom` — world position of the bottom-center of the bar
/// * `width`, `height` — dimensions
/// * `segments` — number of color steps
/// * `stops` — slice of (value 0..1, [r, g, b]) colormap stops
pub fn create_colorbar(
    bottom: Vec3,
    width: f32,
    height: f32,
    segments: usize,
    stops: &[(f32, [f32; 3])],
) -> ColorbarGeometry {
    let hw = width * 0.5;
    let seg_h = height / segments as f32;

    // 2 vertices per row (left + right edge), segments+1 rows
    let mut positions = Vec::with_capacity((segments + 1) * 2);
    let mut indices = Vec::with_capacity(segments * 6);
    let mut sub_meshes = Vec::with_capacity(segments);
    let mut materials = Vec::with_capacity(segments);

    for i in 0..=segments {
        let y = bottom.y + i as f32 * seg_h;
        positions.push([bottom.x - hw, y, bottom.z]);
        positions.push([bottom.x + hw, y, bottom.z]);
    }

    for i in 0..segments {
        let bl = (i * 2) as u32;
        let br = bl + 1;
        let tl = ((i + 1) * 2) as u32;
        let tr = tl + 1;

        let offset = (i * 6) as u32;
        indices.extend_from_slice(&[bl, br, tl, br, tr, tl]);

        sub_meshes.push(SubMesh {
            material_index: i,
            index_offset: offset,
            index_count: 6,
        });

        let t = i as f32 / segments.max(1) as f32;
        let c = lookup_color(stops, t);
        materials.push(PbrMaterial {
            name: format!("CBar_{:.3}", t),
            albedo: [c[0], c[1], c[2], 1.0],
            roughness: 0.6,
            metallic: 0.0,
            ..Default::default()
        });
    }

    // Build mesh vertices with auto-computed normals (flat quad, normals point +Z)
    let vertices: Vec<MeshVertex> = positions.iter().map(|p| MeshVertex {
        position: *p,
        normal: [0.0, 0.0, -1.0], // face toward -Z for visibility from default angle
        uv: [0.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
    }).collect();

    let mut aabb = Aabb { min: Vec3::ZERO, max: Vec3::ZERO };
    for v in &vertices {
        aabb.expand(Vec3::from(v.position));
    }

    let _index_count = indices.len() as u32;
    ColorbarGeometry {
        mesh: Mesh {
            name: "Colorbar".into(),
            vertices,
            indices,
            sub_meshes,
            aabb,
        },
        materials,
    }
}

/// Linear interpolation between colormap stops.
fn lookup_color(stops: &[(f32, [f32; 3])], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    if stops.is_empty() {
        return [t, t, t];
    }
    if stops.len() == 1 {
        return stops[0].1;
    }
    for w in stops.windows(2) {
        let (t0, c0) = (w[0].0, w[0].1);
        let (t1, c1) = (w[1].0, w[1].1);
        if t >= t0 && t <= t1 {
            let u = if (t1 - t0).abs() > 1e-6 {
                (t - t0) / (t1 - t0)
            } else {
                0.0
            };
            return [
                c0[0] + (c1[0] - c0[0]) * u,
                c0[1] + (c1[1] - c0[1]) * u,
                c0[2] + (c1[2] - c0[2]) * u,
            ];
        }
    }
    stops.last().unwrap().1
}

/// Jet colormap: blue → cyan → green → yellow → red.
pub const JET_COLORMAP: &[(f32, [f32; 3])] = &[
    (0.0,   [0.0, 0.0, 0.56]),
    (0.125, [0.0, 0.0, 1.0]),
    (0.375, [0.0, 1.0, 1.0]),
    (0.625, [1.0, 1.0, 0.0]),
    (0.875, [1.0, 0.0, 0.0]),
    (1.0,   [0.5, 0.0, 0.0]),
];

/// Viridis-like colormap: purple → blue → green → yellow.
#[allow(dead_code)]
pub const VIRIDIS_COLORMAP: &[(f32, [f32; 3])] = &[
    (0.0,   [0.267, 0.004, 0.329]),
    (0.25,  [0.282, 0.141, 0.458]),
    (0.50,  [0.128, 0.567, 0.551]),
    (0.75,  [0.466, 0.742, 0.323]),
    (1.0,   [0.993, 0.906, 0.144]),
];
