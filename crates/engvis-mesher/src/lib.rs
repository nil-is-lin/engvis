// engvis-mesher: mesh generation algorithms (DC, MC33, shell, MS-loops).
//
// Depends on `engvis-surface` (for `TreeParams`, `Morphology`, `build_tree`)
// and `engvis-core` (for `Mesh`, `SubMesh`, `Aabb`, `compute_topology`).

use engvis_core::{
    aabb::Aabb,
    mesh::{Mesh, MeshVertex, SubMesh},
    topology::compute_topology,
};
use engvis_surface::Morphology;
use glam::Vec3;

// ── Mesh backend enum ─────────────────────────────────

/// Polygonisation backend.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MeshBackend {
    MarchingCubes33,
    DualContouring,
}

// ── Dual Contouring ─────────────────────────────────────

/// Build a mesh using Dual Contouring (sharp features, jagged boundary).
pub fn build_dc_mesh(
    tree: fidget_core::context::Tree,
    name: &str,
    depth: u8,
) -> (Mesh, String) {
    use fidget_core::shape::Shape;
    use fidget_jit::JitFunction;
    use fidget_mesh::{Octree, Settings};

    let shape = Shape::<JitFunction>::from(tree);
    let settings = Settings {
        depth,
        threads: Some(&fidget_core::render::ThreadPool::Global),
        ..Default::default()
    };
    let octree = Octree::build(&shape, &settings).expect("octree");
    let m = octree.walk_dual();

    let pos: Vec<[f32; 3]> = m.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
    let idx: Vec<u32> = m
        .triangles
        .iter()
        .flat_map(|t| [t.x as u32, t.y as u32, t.z as u32])
        .collect();
    let stats = format!(
        "[DC] {} verts, {} tris (depth={})",
        pos.len(),
        idx.len() / 3,
        depth
    );

    // Step 1: fix DC winding holes via full pipeline.
    let tmp = Mesh::from_triangles("tmp", &pos, &idx);
    let mut pos: Vec<[f32; 3]> = tmp.vertices.iter().map(|v| v.position).collect();
    let mut idx: Vec<u32> = tmp.indices.clone();

    // Step 2: smooth open-boundary silhouette.
    smooth_boundary_ms(&shape, &mut pos, &mut idx);

    // Step 3: build final mesh.
    let mut mesh = Mesh::from_triangles_open(name, &pos, &idx);
    overwrite_normals_with_gradient(&shape, &mut mesh);
    (mesh, stats)
}

// ── Marching Cubes 33 ────────────────────────────────

/// Sample the axis-aligned box `[-sx,sx] × [-sy,sy] × [-sz,sz]`.
pub fn build_mc33_mesh_domain(
    tree: fidget_core::context::Tree,
    name: &str,
    res: usize,
    extent: [f32; 3],
) -> (Mesh, String) {
    use fidget_core::shape::Shape;
    use fidget_jit::JitFunction;

    let t_shape = std::time::Instant::now();
    let shape = Shape::<JitFunction>::from(tree);

    let max_extent = extent.iter().cloned().fold(1.0_f32, f32::max);
    let base_res = ((res as f32) * max_extent).ceil() as usize;

    let float_tape = shape.float_slice_tape(Default::default());
    let dt_shape = t_shape.elapsed();

    let (x0, x1, y0, y1, z0, z1) = (-extent[0], extent[0], -extent[1], extent[1], -extent[2], extent[2]);
    let inv_max = 1.0 / max_extent;
    let nx = ((base_res as f32) * extent[0] * inv_max).ceil() as usize;
    let ny = ((base_res as f32) * extent[1] * inv_max).ceil() as usize;
    let nz = ((base_res as f32) * extent[2] * inv_max).ceil() as usize;
    let dx = (x1 - x0) / nx as f32;
    let dy = (y1 - y0) / ny as f32;
    let dz = (z1 - z0) / nz as f32;
    let stride_y = ny + 1;
    let stride_z = nz + 1;
    let total = (nx + 1) * stride_y * stride_z;

    let t_grid = std::time::Instant::now();
    let mut xs = Vec::with_capacity(total);
    let mut ys = Vec::with_capacity(total);
    let mut zs = Vec::with_capacity(total);
    for ix in 0..=nx {
        let x = x0 + ix as f32 * dx;
        for iy in 0..=ny {
            let y = y0 + iy as f32 * dy;
            for iz in 0..=nz {
                xs.push(x);
                ys.push(y);
                zs.push(z0 + iz as f32 * dz);
            }
        }
    }
    let dt_build = t_grid.elapsed();

    let t_eval = std::time::Instant::now();
    let grid = {
        let mut eval = Shape::<JitFunction>::new_float_slice_eval();
        match eval.eval(&float_tape, &xs, &ys, &zs) {
            Ok(r) => r.to_vec(),
            Err(_) => vec![1e9_f32; total],
        }
    };
    let dt_eval = t_eval.elapsed();

    let t_extract = std::time::Instant::now();
    let (mut pos, idx) = engvis_core::marching_cubes::extract_par_with_grid(
        |x: f32, y: f32, z: f32| -> f32 {
            let mut eval = Shape::<JitFunction>::new_float_slice_eval();
            match eval.eval(&float_tape, &[x], &[y], &[z]) {
                Ok(r) => r[0],
                Err(_) => 1e9,
            }
        },
        &grid,
        (x0, x1, nx),
        (y0, y1, ny),
        (z0, z1, nz),
    );
    let dt_extract = t_extract.elapsed();

    let is_unit = (extent[0] - 1.0).abs() < 1e-6
        && (extent[1] - 1.0).abs() < 1e-6
        && (extent[2] - 1.0).abs() < 1e-6;
    let mut out_of_bounds = 0usize;
    if is_unit {
        for p in &mut pos {
            for v in p {
                if *v < -1.0 {
                    *v = -1.0;
                    out_of_bounds += 1;
                }
                if *v > 1.0 {
                    *v = 1.0;
                    out_of_bounds += 1;
                }
            }
        }
    }

    let t_mesh = std::time::Instant::now();
    let mut mesh = Mesh::from_triangles(name, &pos, &idx);
    let dt_mesh = t_mesh.elapsed();
    let t_norm = std::time::Instant::now();
    overwrite_normals_with_gradient(&shape, &mut mesh);
    let dt_norm = t_norm.elapsed();
    let clamped_msg = if is_unit && out_of_bounds > 0 {
        format!(" | clamped {} coords", out_of_bounds)
    } else {
        String::new()
    };
    let stats = format!(
        "[MC33] {} verts, {} tris (res={}){} | shape={:.0}ms grid={:.0}ms eval={:.0}ms extract={:.0}ms mesh={:.0}ms normals={:.0}ms",
        pos.len(),
        idx.len() / 3,
        res,
        clamped_msg,
        dt_shape.as_secs_f64() * 1e3,
        dt_build.as_secs_f64() * 1e3,
        dt_eval.as_secs_f64() * 1e3,
        dt_extract.as_secs_f64() * 1e3,
        dt_mesh.as_secs_f64() * 1e3,
        dt_norm.as_secs_f64() * 1e3,
    );
    (mesh, stats)
}

/// Recompute smooth normals (used after `clip_mesh_to_ball`).
pub fn recompute_smooth_normals(mesh: &mut Mesh) {
    let n = mesh.vertices.len();
    let mut normals = vec![[0.0_f32; 3]; n];
    for tri in mesh.indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let p0 = Vec3::from(mesh.vertices[i0].position);
        let p1 = Vec3::from(mesh.vertices[i1].position);
        let p2 = Vec3::from(mesh.vertices[i2].position);
        let nrm = (p1 - p0).cross(p2 - p0);
        for &i in &[i0, i1, i2] {
            normals[i][0] += nrm.x;
            normals[i][1] += nrm.y;
            normals[i][2] += nrm.z;
        }
    }
    for (vert, norm) in mesh.vertices.iter_mut().zip(normals.iter()) {
        let len = (norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]).sqrt();
        vert.normal = if len > 1e-10 {
            let inv = 1.0 / len;
            [norm[0] * inv, norm[1] * inv, norm[2] * inv]
        } else {
            [0.0, 1.0, 0.0]
        };
    }
}

// ── Wireframe helpers ────────────────────────────────────

/// Build a wireframe mesh: outer box + internal cell-boundary grid lines.
pub fn build_box_wireframe(extent: [f32; 3]) -> Mesh {
    let [sx, sy, sz] = extent;
    let pts: [[f32; 3]; 8] = [
        [-sx, -sy, -sz],
        [sx, -sy, -sz],
        [sx, sy, -sz],
        [-sx, sy, -sz],
        [-sx, -sy, sz],
        [sx, -sy, sz],
        [sx, sy, sz],
        [-sx, sy, sz],
    ];
    let outer_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let push_edge = |positions: &mut Vec<[f32; 3]>,
                     indices: &mut Vec<u32>,
                     p0: [f32; 3],
                     p1: [f32; 3]| {
        let base = positions.len() as u32;
        positions.push(p0);
        positions.push(p1);
        indices.extend_from_slice(&[base, base + 1, base]);
    };

    for &(a, b) in &outer_edges {
        push_edge(&mut positions, &mut indices, pts[a], pts[b]);
    }

    let nx = extent[0] as usize;
    let ny = extent[1] as usize;
    let nz = extent[2] as usize;
    for i in 1..nx {
        let x = -sx + 2.0 * sx * i as f32 / nx as f32;
        push_edge(&mut positions, &mut indices, [x, -sy, -sz], [x, sy, -sz]);
        push_edge(&mut positions, &mut indices, [x, sy, -sz], [x, sy, sz]);
        push_edge(&mut positions, &mut indices, [x, sy, sz], [x, -sy, sz]);
        push_edge(&mut positions, &mut indices, [x, -sy, sz], [x, -sy, -sz]);
    }
    for j in 1..ny {
        let y = -sy + 2.0 * sy * j as f32 / ny as f32;
        push_edge(&mut positions, &mut indices, [-sx, y, -sz], [sx, y, -sz]);
        push_edge(&mut positions, &mut indices, [sx, y, -sz], [sx, y, sz]);
        push_edge(&mut positions, &mut indices, [sx, y, sz], [-sx, y, sz]);
        push_edge(&mut positions, &mut indices, [-sx, y, sz], [-sx, y, -sz]);
    }
    for k in 1..nz {
        let z = -sz + 2.0 * sz * k as f32 / nz as f32;
        push_edge(&mut positions, &mut indices, [-sx, -sy, z], [sx, -sy, z]);
        push_edge(&mut positions, &mut indices, [sx, -sy, z], [sx, sy, z]);
        push_edge(&mut positions, &mut indices, [sx, sy, z], [-sx, sy, z]);
        push_edge(&mut positions, &mut indices, [-sx, sy, z], [-sx, -sy, z]);
    }
    wireframe_mesh_from_segments("box-wireframe", positions, indices)
}

/// Build a sphere wireframe (latitude/longitude lines).
pub fn build_sphere_wireframe(r: f32, n_lat: usize, n_lon: usize) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let push_seg = |positions: &mut Vec<[f32; 3]>,
                     indices: &mut Vec<u32>,
                     p0: [f32; 3],
                     p1: [f32; 3]| {
        let base = positions.len() as u32;
        positions.push(p0);
        positions.push(p1);
        indices.extend_from_slice(&[base, base + 1, base]);
    };
    for i in 0..n_lon {
        let az = (i as f32) / (n_lon as f32) * std::f32::consts::TAU;
        let ca = az.cos();
        let sa = az.sin();
        for j in 0..(2 * n_lat) {
            let t0 = (j as f32) / ((2 * n_lat) as f32) * std::f32::consts::PI;
            let t1 = ((j + 1) as f32) / ((2 * n_lat) as f32) * std::f32::consts::PI;
            push_seg(
                &mut positions,
                &mut indices,
                [r * t0.sin() * ca, r * t0.cos(), r * t0.sin() * sa],
                [r * t1.sin() * ca, r * t1.cos(), r * t1.sin() * sa],
            );
        }
    }
    for j in 1..n_lat {
        let pol = (j as f32) / (n_lat as f32) * std::f32::consts::PI;
        let y = r * pol.cos();
        let rr = r * pol.sin();
        for i in 0..(2 * n_lat) {
            let a0 = (i as f32) / ((2 * n_lat) as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32) / ((2 * n_lat) as f32) * std::f32::consts::TAU;
            push_seg(
                &mut positions,
                &mut indices,
                [rr * a0.cos(), y, rr * a0.sin()],
                [rr * a1.cos(), y, rr * a1.sin()],
            );
        }
    }
    wireframe_mesh_from_segments("sphere-wireframe", positions, indices)
}

/// Construct a `Mesh` of degenerate triangles (wireframe segments).
pub fn wireframe_mesh_from_segments(
    name: &str,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
) -> Mesh {
    let mut aabb = Aabb::empty();
    for p in &positions {
        aabb.expand(Vec3::from(*p));
    }
    let vertices: Vec<MeshVertex> = positions
        .into_iter()
        .map(|p| MeshVertex {
            position: p,
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        })
        .collect();
    let index_count = indices.len() as u32;
    Mesh {
        name: name.to_string(),
        vertices,
        indices,
        sub_meshes: vec![SubMesh {
            material_index: 0,
            index_offset: 0,
            index_count,
        }],
        aabb,
    }
}

// ── Ball clip ────────────────────────────────────────────

/// Clip a mesh to the interior of a ball (centre `c`, radius `r`).
///
/// Each triangle straddling the sphere is split at exact intersection
/// points so the boundary lies on the sphere.
pub fn clip_mesh_to_ball(mesh: &mut Mesh, c: [f32; 3], r: f32) {
    // (Full implementation is ~90 lines; see original `main.rs` lines 891-1010.)
    // ── Implementation ─────────────────────────────────────
    let r2 = r * r;
    let d2 = |p: [f32; 3]| -> f32 {
        let dx = p[0] - c[0];
        let dy = p[1] - c[1];
        let dz = p[2] - c[2];
        dx * dx + dy * dy + dz * dz - r2
    };
    let hit = |a: [f32; 3], b: [f32; 3], da: f32, db: f32| -> [f32; 3] {
        let t = da / (da - db);
        [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
    };

    fn add_orig(
        positions: &[[f32; 3]],
        new_pos: &mut Vec<[f32; 3]>,
        vert_map: &mut [u32],
        i: usize,
    ) -> u32 {
        if vert_map[i] == u32::MAX {
            let ni = new_pos.len() as u32;
            new_pos.push(positions[i]);
            vert_map[i] = ni;
        }
        vert_map[i]
    }
    fn add_new(new_pos: &mut Vec<[f32; 3]>, p: [f32; 3]) -> u32 {
        let ni = new_pos.len() as u32;
        new_pos.push(p);
        ni
    }

    let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
    let mut new_pos: Vec<[f32; 3]> = Vec::with_capacity(positions.len());
    let mut new_idx: Vec<u32> = Vec::with_capacity(mesh.indices.len());
    let mut vert_map: Vec<u32> = vec![u32::MAX; positions.len()];

    for tri in mesh.indices.chunks_exact(3) {
        let i = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let v = [positions[i[0]], positions[i[1]], positions[i[2]]];
        let d = [d2(v[0]), d2(v[1]), d2(v[2])];
        let inside = [d[0] <= 0.0, d[1] <= 0.0, d[2] <= 0.0];
        let n_in = inside.iter().filter(|&&x| x).count();

        match n_in {
            3 => {
                let a = add_orig(&positions, &mut new_pos, &mut vert_map, i[0]);
                let b = add_orig(&positions, &mut new_pos, &mut vert_map, i[1]);
                let c = add_orig(&positions, &mut new_pos, &mut vert_map, i[2]);
                new_idx.extend_from_slice(&[a, b, c]);
            }
            0 => { /* discard */ }
            1 => {
                let i0 = inside.iter().position(|&x| x).unwrap();
                let o1 = (i0 + 1) % 3;
                let o2 = (i0 + 2) % 3;
                let a = hit(v[i0], v[o1], d[i0], d[o1]);
                let b = hit(v[i0], v[o2], d[i0], d[o2]);
                let ia = add_orig(&positions, &mut new_pos, &mut vert_map, i[i0]);
                let iaa = add_new(&mut new_pos, a);
                let ibb = add_new(&mut new_pos, b);
                new_idx.extend_from_slice(&[ia, iaa, ibb]);
            }
            2 => {
                let o = inside.iter().position(|&x| !x).unwrap();
                let prev = (o + 2) % 3;
                let next = (o + 1) % 3;
                let a = hit(v[prev], v[o], d[prev], d[o]);
                let b = hit(v[next], v[o], d[next], d[o]);
                let inext = add_orig(&positions, &mut new_pos, &mut vert_map, i[next]);
                let iprev = add_orig(&positions, &mut new_pos, &mut vert_map, i[prev]);
                let iaa = add_new(&mut new_pos, a);
                let ibb = add_new(&mut new_pos, b);
                new_idx.extend_from_slice(&[inext, iprev, iaa]);
                new_idx.extend_from_slice(&[inext, iaa, ibb]);
            }
            _ => unreachable!(),
        }
    }

    let mut new_verts: Vec<MeshVertex> = Vec::with_capacity(new_pos.len());
    for &p in &new_pos {
        new_verts.push(MeshVertex {
            position: p,
            normal: [0.0, 0.0, 0.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
    }
    mesh.vertices = new_verts;
    mesh.indices = new_idx;
    mesh.sub_meshes.clear();
    mesh.sub_meshes.push(SubMesh {
        material_index: 0,
        index_offset: 0,
        index_count: mesh.indices.len() as u32,
    });

    mesh.aabb = Aabb::empty();
    for v in &mesh.vertices {
        mesh.aabb.min[0] = mesh.aabb.min[0].min(v.position[0]);
        mesh.aabb.min[1] = mesh.aabb.min[1].min(v.position[1]);
        mesh.aabb.min[2] = mesh.aabb.min[2].min(v.position[2]);
        mesh.aabb.max[0] = mesh.aabb.max[0].max(v.position[0]);
        mesh.aabb.max[1] = mesh.aabb.max[1].max(v.position[1]);
        mesh.aabb.max[2] = mesh.aabb.max[2].max(v.position[2]);
    }
}

// ── Shell mesh ────────────────────────────────────────────

/// Build a TPMS shell mesh: thin solid wall `{ |f| ≤ half_t }`.
pub fn build_shell_mesh(
    tree: fidget_core::context::Tree,
    half_t: f32,
    name: &str,
    res: usize,
    clip_to_unit_ball: bool,
    clip_radius: f32,
    domain_extent: [f32; 3],
) -> (Mesh, String) {
    use fidget_core::context::Tree as T;

    let [sx, sy, sz] = domain_extent;
    let pad = 4.0 / res as f32;
    let mc_extent = [sx + pad, sy + pad, sz + pad];
    let max_extent = sx.max(sy).max(sz).max(1.0);
    let half = max_extent + pad;
    let cell = 2.0 * half / (res as f32 * half).ceil();
    let (cx, cy, cz) = (sx - 0.5 * cell, sy - 0.5 * cell, sz - 0.5 * cell);
    let box_sdf = (T::x().abs() - cx).max(T::y().abs() - cy).max(T::z().abs() - cz);

    let wall = tree.square() - half_t * half_t;
    let field = wall.max(box_sdf);
    let (mut mesh, mc33_stats) = build_mc33_mesh_domain(field, name, res, mc_extent);

    if clip_to_unit_ball {
        clip_mesh_to_ball(&mut mesh, [0.0, 0.0, 0.0], clip_radius);
        let clip_msg = format!(" | clip({}v/{}t)", mesh.vertices.len(), mesh.indices.len() / 3);
        recompute_smooth_normals(&mut mesh);
        (mesh, format!("{mc33_stats}{clip_msg}"))
    } else {
        (mesh, mc33_stats)
    }
}

// ── Unified mesh builder ─────────────────────────────────

/// Build a mesh for the given `tree` with the specified backend and morphology.
pub fn build_mesh(
    tree: fidget_core::context::Tree,
    name: &str,
    backend: MeshBackend,
    depth: u8,
    mc_res: usize,
    clip_to_unit_ball: bool,
    clip_radius: f32,
    morphology: Morphology,
    domain_extent: [f32; 3],
) -> (Mesh, String) {
    let [sx, sy, sz] = domain_extent;
    let max_extent = sx.max(sy).max(sz).max(1.0);
    let pad = 4.0 / mc_res as f32;

    let is_solid = matches!(morphology, Morphology::Skeletal);
    let (mut mesh, gen_stats) = if is_solid {
        use fidget_core::context::Tree as T;
        let half = max_extent + pad;
        let cell = 2.0 * half / (mc_res as f32 * half).ceil();
        let (cx, cy, cz) = (sx - 0.5 * cell, sy - 0.5 * cell, sz - 0.5 * cell);
        let box_sdf = (T::x().abs() - cx).max(T::y().abs() - cy).max(T::z().abs() - cz);
        let clipped = tree.clone().max(box_sdf);
        let mc_extent = [sx + pad, sy + pad, sz + pad];
        match backend {
            MeshBackend::DualContouring => build_dc_mesh(clipped, name, depth),
            MeshBackend::MarchingCubes33 => build_mc33_mesh_domain(clipped, name, mc_res, mc_extent),
        }
    } else {
        let mc_extent = [sx + pad, sy + pad, sz + pad];
        match backend {
            MeshBackend::DualContouring => build_dc_mesh(tree.clone(), name, depth),
            MeshBackend::MarchingCubes33 => build_mc33_mesh_domain(tree.clone(), name, mc_res, mc_extent),
        }
    };
    if clip_to_unit_ball {
        clip_mesh_to_ball(&mut mesh, [0.0, 0.0, 0.0], clip_radius);
        let clip_msg = format!(" | clip({}v/{}t)", mesh.vertices.len(), mesh.indices.len() / 3);
        recompute_smooth_normals(&mut mesh);
        (mesh, format!("{gen_stats}{clip_msg}"))
    } else {
        (mesh, gen_stats)
    }
}

// ── Marching Squares (MS-loop visualisation) ─────────────

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
enum Face {
    Xp,
    Xm,
    Yp,
    Ym,
    Zp,
    Zm ,
}

impl Face {
    fn lock(self) -> (usize, f32) {
        match self {
            Face::Xp => (0, 1.0),
            Face::Xm => (0, -1.0),
            Face::Yp => (1, 1.0),
            Face::Ym => (1, -1.0),
            Face::Zp => (2, 1.0),
            Face::Zm => (2, -1.0),
        }
    }
    fn free_axes(self) -> (usize, usize) {
        match self {
            Face::Xp | Face::Xm => (1, 2),
            Face::Yp | Face::Ym => (0, 2),
            Face::Zp | Face::Zm => (0, 1),
        }
    }
    fn eval_uv<F: fidget_core::eval::Function + Clone>(
        &self,
        shape: &fidget_core::shape::Shape<F, ()>,
        u: f32,
        v: f32,
    ) -> f32 {
        let (ax, sign) = self.lock();
        let (ua, va) = self.free_axes();
        let mut p = [0.0_f32; 3];
        p[ax] = sign;
        p[ua] = u;
        p[va] = v;
        let mut eval = fidget_core::shape::Shape::<F>::new_float_slice_eval();
        let tape = shape.float_slice_tape(Default::default());
        let xs = [p[0]];
        let ys = [p[1]];
        let zs = [p[2]];
        match eval.eval(&tape, &xs, &ys, &zs) {
            Ok(r) => r[0],
            Err(_) => 1e9,
        }
    }
}

fn marching_squares_face<F: fidget_core::eval::Function + Clone>(
    face: Face,
    shape: &fidget_core::shape::Shape<F, ()>,
    res: usize,
) -> Vec<([f32; 2], [f32; 2])> {
    let step = 2.0 / res as f32;
    let mut grid: Vec<Vec<f32>> = Vec::with_capacity(res + 1);
    for j in 0..=res {
        let mut row = Vec::with_capacity(res + 1);
        for i in 0..=res {
            let u = -1.0 + i as f32 * step;
            let v = -1.0 + j as f32 * step;
            row.push(face.eval_uv(shape, u, v));
        }
        grid.push(row);
    }
    let mut segs: Vec<([f32; 2], [f32; 2])> = Vec::new();
    for j in 0..res {
        for i in 0..res {
            let tl = grid[j][i];
            let tr = grid[j][i + 1];
            let bl = grid[j + 1][i];
            let br = grid[j + 1][i + 1];
            let case = ((if tl < 0.0 { 1 } else { 0 })
                | (if tr < 0.0 { 2 } else { 0 })
                | (if br < 0.0 { 4 } else { 0 })
                | (if bl < 0.0 { 8 } else { 0 })) as u8;
            if case == 0 || case == 15 {
                continue;
            }
            let lerp_a = |a: f32, b: f32| {
                let d = b - a;
                if d.abs() < 1e-12 { 0.5 } else { -a / d }
            };
            let top = [-1.0 + (i as f32 + lerp_a(tl, tr)) * step, -1.0 + j as f32 * step];
            let bottom = [-1.0 + (i as f32 + lerp_a(bl, br)) * step, -1.0 + (j + 1) as f32 * step];
            let left = [-1.0 + i as f32 * step, -1.0 + (j as f32 + lerp_a(tl, bl)) * step];
            let right = [-1.0 + (i + 1) as f32 * step, -1.0 + (j as f32 + lerp_a(tr, br)) * step];
            match case {
                1 | 14 => {
                    segs.push((left, bottom));
                }
                2 | 13 => {
                    segs.push((bottom, right));
                }
                3 | 12 => {
                    segs.push((left, right));
                }
                4 | 11 => {
                    segs.push((top, right));
                }
                5 => {
                    segs.push((left, top));
                    segs.push((bottom, right));
                }
                6 | 9 => {
                    segs.push((top, bottom));
                }
                7 | 8 => {
                    segs.push((left, right));
                }
                10 => {
                    segs.push((top, left));
                    segs.push((bottom, right));
                }
                _ => {}
            }
        }
    }
    segs
}

fn dist3_2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Extract 3-D boundary loops (f = 0) on all 6 box faces via Marching Squares.
pub fn extract_ms_loops_3d<F: fidget_core::eval::Function + Clone>(
    shape: &fidget_core::shape::Shape<F, ()>,
    res: usize,
) -> Vec<Vec<[f32; 3]>> {
    let mut all_segs: Vec<([f32; 3], [f32; 3])> = Vec::new();
    for face in [Face::Xp, Face::Xm, Face::Yp, Face::Ym, Face::Zp, Face::Zm] {
        let segs_2d = marching_squares_face(face, shape, res);
        let (ax, sign) = face.lock();
        let (ua, va) = face.free_axes();
        for &(a, b) in &segs_2d {
            let to3d = |p: [f32; 2]| {
                let mut q = [0.0_f32; 3];
                q[ax] = sign;
                q[ua] = p[0];
                q[va] = p[1];
                q
            };
            all_segs.push((to3d(a), to3d(b)));
        }
    }
    // Chain segments into closed loops.
    let mut loops: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut rem = all_segs.clone();
    while !rem.is_empty() {
        let mut lp = vec![rem[0].0, rem[0].1];
        rem.remove(0);
        for _ in 0..100000 {
            let last = *lp.last().unwrap();
            let mut found: Option<(usize, bool)> = None;
            for (si, &(a, b)) in rem.iter().enumerate() {
                if dist3_2(a, last) < 1e-4 {
                    found = Some((si, true));
                    break;
                }
                if dist3_2(b, last) < 1e-4 {
                    found = Some((si, false));
                    break;
                }
            }
            match found {
                Some((si, fwd)) => {
                    let (a, b) = rem.remove(si);
                    lp.push(if fwd { b } else { a });
                }
                None => break,
            }
        }
        if lp.len() >= 3 && dist3_2(lp[0], *lp.last().unwrap()) < 1e-4 {
            lp.pop();
        }
        if lp.len() >= 3 {
            loops.push(lp);
        }
    }
    loops
}

/// Build a thin triangle-strip mesh representing MS boundary loops.
pub fn build_ms_loops_mesh<F: fidget_core::eval::Function + Clone>(
    shape: &fidget_core::shape::Shape<F, ()>,
    res: usize,
) -> Mesh {
    let loops = extract_ms_loops_3d(shape, res);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let half_w = 0.008_f32;
    for lp in &loops {
        let n = lp.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let p0 = lp[i];
            let p1 = lp[(i + 1) % n];
            let dir = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            if len < 1e-10 {
                continue;
            }
            let dn = [dir[0] / len, dir[1] / len, dir[2] / len];
            let axis = if dn[0].abs() < dn[1].abs() && dn[0].abs() < dn[2].abs() {
                [1.0_f32, 0.0, 0.0]
            } else if dn[1].abs() < dn[2].abs() {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let perp = [
                dn[1] * axis[2] - dn[2] * axis[1],
                dn[2] * axis[0] - dn[0] * axis[2],
                dn[0] * axis[1] - dn[1] * axis[0],
            ];
            let pl = (perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]).sqrt();
            let perp = if pl > 1e-10 {
                [perp[0] / pl * half_w, perp[1] / pl * half_w, perp[2] / pl * half_w]
            } else {
                [half_w, 0.0, 0.0]
            };
            let base = positions.len() as u32;
            positions.push([p0[0] + perp[0], p0[1] + perp[1], p0[2] + perp[2]]);
            positions.push([p0[0] - perp[0], p0[1] - perp[1], p0[2] - perp[2]]);
            positions.push([p1[0] + perp[0], p1[1] + perp[1], p1[2] + perp[2]]);
            positions.push([p1[0] - perp[0], p1[1] - perp[1], p1[2] - perp[2]]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
        }
    }
    eprintln!(
        "  [MS-loops] {} loops, {} verts, {} tris",
        loops.len(),
        positions.len(),
        indices.len() / 3
    );
    let mut mesh = Mesh::from_triangles_open("ms-loops", &positions, &indices);
    for v in &mut mesh.vertices {
        v.normal = [0.0, 1.0, 0.0];
    }
    mesh
}

// ── Boundary smoothing ────────────────────────────────────

/// Move DC boundary vertices onto curve C = (box face) ∩ { f = 0 }.
pub fn smooth_boundary_ms<F: fidget_core::eval::Function + Clone>(
    shape: &fidget_core::shape::Shape<F, ()>,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    use std::collections::{HashMap, HashSet};

    let mut edge_cnt: HashMap<(u32, u32), u32> = HashMap::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        for &(i0, i1) in &[(a, b), (b, c), (c, a)] {
            let key = if i0 <= i1 { (i0, i1) } else { (i1, i0) };
            *edge_cnt.entry(key).or_default() += 1;
        }
    }
    let mut bnd_verts: HashSet<u32> = HashSet::new();
    for (&(a, b), &cnt) in &edge_cnt {
        if cnt == 1 {
            bnd_verts.insert(a);
            bnd_verts.insert(b);
        }
    }
    if bnd_verts.is_empty() {
        return;
    }

    let project_to_c = |p: [f32; 3]| -> Option<[f32; 3]> {
        let ax = [p[0].abs(), p[1].abs(), p[2].abs()];
        let max_ax = ax.iter().cloned().fold(0.0_f32, f32::max);
        if max_ax < 0.9 {
            return None;
        }
        let lock_ax = if ax[0] >= ax[1] && ax[0] >= ax[2] {
            0
        } else if ax[1] >= ax[2] {
            1
        } else {
            2
        };
        let sign = if p[lock_ax] >= 0.0 { 1.0_f32 } else { -1.0 };
        let face = match (lock_ax, sign) {
            (0, 1.0) => Face::Xp,
            (0, -1.0) => Face::Xm,
            (1, 1.0) => Face::Yp,
            (1, -1.0) => Face::Ym,
            (2, 1.0) => Face::Zp,
            (2, -1.0) => Face::Zm,
            _ => return None,
        };
        let (ua, va) = face.free_axes();
        let mut u = p[ua];
        let mut v = p[va];
        for _ in 0..24 {
            let fval = face.eval_uv(shape, u, v);
            if fval.abs() < 1e-8 {
                break;
            }
            let eps = 1e-6;
            let gx = (face.eval_uv(shape, u + eps, v) - fval) / eps;
            let gy = (face.eval_uv(shape, u, v + eps) - fval) / eps;
            let m = gx * gx + gy * gy;
            if m < 1e-20 {
                break;
            }
            let s = fval / m;
            u -= s * gx;
            v -= s * gy;
        }
        let mut q = [0.0_f32; 3];
        q[lock_ax] = sign;
        q[ua] = u;
        q[va] = v;
        Some(q)
    };

    let mut moved = 0u32;
    for &vi in &bnd_verts {
        let p = positions[vi as usize];
        if let Some(pc) = project_to_c(p) {
            positions[vi as usize] = pc;
            moved += 1;
        }
    }
    eprintln!(
        "  smooth_boundary_ms: moved {} / {} boundary vertices to curve C",
        moved,
        bnd_verts.len()
    );
}

/// Overwrite per-vertex normals with the analytic surface gradient ∇f.
pub fn overwrite_normals_with_gradient<F: fidget_core::eval::Function + Clone>(
    shape: &fidget_core::shape::Shape<F, ()>,
    mesh: &mut Mesh,
) {
    use fidget_core::types::Grad;
    let mut grad_eval = fidget_core::shape::Shape::<F>::new_grad_slice_eval();
    let tape = shape.grad_slice_tape(Default::default());
    let n = mesh.vertices.len();
    let chunk = 4096;
    for start in (0..n).step_by(chunk) {
        let end = (start + chunk).min(n);
        let xs: Vec<Grad> = mesh.vertices[start..end].iter().map(|v| Grad::from(v.position[0])).collect();
        let ys: Vec<Grad> = mesh.vertices[start..end].iter().map(|v| Grad::from(v.position[1])).collect();
        let zs: Vec<Grad> = mesh.vertices[start..end].iter().map(|v| Grad::from(v.position[2])).collect();
        let res = match grad_eval.eval(&tape, &xs, &ys, &zs) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for (i, g) in res.iter().enumerate() {
            let len = (g.dx * g.dx + g.dy * g.dy + g.dz * g.dz).sqrt();
            if len > 1e-10 {
                let inv = 1.0 / len;
                mesh.vertices[start + i].normal = [g.dx * inv, g.dy * inv, g.dz * inv];
            }
        }
    }
}

// ── Diagnostic ────────────────────────────────────────────

/// Classify boundary edges as on-box-face vs interior (diagnostic).
#[allow(dead_code)]
pub fn classify_boundary_edges(mesh: &Mesh, tag: &str) {
    use std::collections::HashMap;
    let v = mesh.vertices.len();
    let f = mesh.indices.len() / 3;
    let mut edge_faces: HashMap<(u32, u32), usize> = HashMap::with_capacity(f * 3);
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        for &(i0, i1) in &[(a, b), (b, c), (c, a)] {
            let key = if i0 <= i1 { (i0, i1) } else { (i1, i0) };
            *edge_faces.entry(key).or_default() += 1;
        }
    }
    let tol = 0.02;
    let mut on_box_face = 0usize;
    let mut interior = 0usize;
    for &(a, b) in edge_faces.keys() {
        if edge_faces[&(a, b)] != 1 {
            continue;
        }
        let pa = mesh.vertices[a as usize].position;
        let pb = mesh.vertices[b as usize].position;
        let on_face = |p: [f32; 3]| -> bool { p[0].abs() > 1.0 - tol || p[1].abs() > 1.0 - tol || p[2].abs() > 1.0 - tol };
        if on_face(pa) && on_face(pb) {
            on_box_face += 1;
        } else {
            interior += 1;
        }
    }
    let topo = compute_topology(mesh);
    eprintln!(
        "  [diag:{}] V={} F={} χ={} components={} | boundary_edges: {} on_box_face, {} interior | non_manifold={}",
        tag,
        v,
        f,
        topo.euler,
        topo.connected_components,
        on_box_face,
        interior,
        topo.non_manifold_edges
    );
}
