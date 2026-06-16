pub use glam::{Vec2, Vec3, Vec4, Mat3, Mat4, Quat, Affine3A};

/// Compute tangent vectors from positions, normals, UVs and indices.
/// Returns a Vec of [f32; 4] where xyz = tangent direction, w = handedness.
pub fn compute_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let vertex_count = positions.len();
    let mut tangents = vec![Vec3::ZERO; vertex_count];
    let mut bitangents = vec![Vec3::ZERO; vertex_count];

    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

        let p0 = Vec3::from(positions[i0]);
        let p1 = Vec3::from(positions[i1]);
        let p2 = Vec3::from(positions[i2]);

        let uv0 = Vec2::from(uvs[i0]);
        let uv1 = Vec2::from(uvs[i1]);
        let uv2 = Vec2::from(uvs[i2]);

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let duv1 = uv1 - uv0;
        let duv2 = uv2 - uv0;

        let f = 1.0 / (duv1.x * duv2.y - duv2.x * duv1.y + 1e-12);
        let tangent = Vec3::new(
            f * (duv2.y * e1.x - duv1.y * e2.x),
            f * (duv2.y * e1.y - duv1.y * e2.y),
            f * (duv2.y * e1.z - duv1.y * e2.z),
        );
        let bitangent = Vec3::new(
            f * (-duv2.x * e1.x + duv1.x * e2.x),
            f * (-duv2.x * e1.y + duv1.x * e2.y),
            f * (-duv2.x * e1.z + duv1.x * e2.z),
        );

        for &i in &[i0, i1, i2] {
            tangents[i] += tangent;
            bitangents[i] += bitangent;
        }
    }

    (0..vertex_count)
        .map(|i| {
            let n = Vec3::from(normals[i]);
            let t = tangents[i];
            let b = bitangents[i];

            // Gram-Schmidt orthogonalize
            let t_ortho = (t - n * n.dot(t)).normalize_or_zero();
            let handedness = if n.dot(t_ortho.cross(b)) < 0.0 {
                -1.0
            } else {
                1.0
            };
            [t_ortho.x, t_ortho.y, t_ortho.z, handedness]
        })
        .collect()
}
