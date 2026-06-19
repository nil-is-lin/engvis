// ── engvis viewer ──────────────────────────────────────────
// Dual Contouring + Marching Cubes 33 meshing of implicit surfaces
// with boundary smoothing and MS-loop visualization.

mod mesh_io;

use engvis_core::{
    material::PbrMaterial,
    scene::{Scene, SceneNode},
    camera::OrbitCamera,
    mesh::Mesh,
    marching_cubes,
    topology::compute_topology,
    Aabb, SubMesh, MeshVertex,
};
use engvis_renderer::{
    EngvisApp, AppCtx, FrameCtx, RunConfig, EventHandling, load_gltf,
};

// =====================================================================
// 1. Implicit surface definition (shared by DC and MC33)
// =====================================================================

/// Parameters for building an implicit surface tree.
#[derive(Clone, Debug)]
struct TreeParams<'a> {
    name: &'a str,
    sphere_radius: f32,
    torus_major_r: f32,
    torus_minor_r: f32,
    tpms_period: f32,
}

impl<'a> TreeParams<'a> {
    /// Set a sensible default period when switching to a TPMS surface.
    fn set_tpms_defaults(&mut self, name: &str) {
        self.tpms_period = match name {
            "gyroid" => 4.0,
            "fischer-koch-s" | "fischer-koch-y" => 2.0,
            _ => 3.0,
        };
    }
}

/// Built-in implicit surfaces. Each name maps to a Fidget `Tree`.
fn build_tree(p: &TreeParams) -> fidget_core::context::Tree {
    use fidget_core::context::Tree as T;
    let k = p.tpms_period;
    let s = move |f: f32| -> (T, T, T) { (T::x() * f, T::y() * f, T::z() * f) };
    match p.name {
        "sphere" => (T::x().square() + T::y().square() + T::z().square()).sqrt()
          - p.sphere_radius,
        "torus" => {
            let major = T::x().square() + T::y().square();
            (major.sqrt() - p.torus_major_r).square() + T::z().square()
          - p.torus_minor_r * p.torus_minor_r
        }
        // Gyroid:        sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = 0
        "gyroid" => {
            let (x, y, z) = s(k);
            x.clone().sin() * y.clone().cos()
          + y.clone().sin() * z.clone().cos()
          + z.clone().sin() * x.clone().cos()
        }
        // Schwarz P:     cos(kx) + cos(ky) + cos(kz) = 0
        "schwarz-p" => {
            let (x, y, z) = s(k);
            x.cos() + y.cos() + z.cos()
        }
        // Schwarz D (Diamond):
        "schwarz-d" => {
            let (x, y, z) = s(k);
            let (sx, sy, sz) = (x.clone().sin(), y.clone().sin(), z.clone().sin());
            let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
            sx.clone()*sy.clone()*sz.clone()
          + sx.clone()*cy.clone()*cz.clone()
          + cx.clone()*sy.clone()*cz.clone()
          + cx*cy*sz
        }
        // Schoen IWP:
        "schoen-iwp" => {
            let (x, y, z) = s(k);
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (c2x, c2y, c2z) = ((x*2.0).cos(), (y*2.0).cos(), (z*2.0).cos());
            (cx.clone()*cy.clone() + cy*cz.clone() + cz*cx) * 2.0
          - (c2x + c2y + c2z)
        }
        // Neovius:
        "neovius" => {
            let (x, y, z) = s(k);
            let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
            (cx.clone() + cy.clone() + cz.clone()) * 3.0
          + cx*cy*cz * 4.0
        }
        // Fischer-Koch F-RD (Schoen FRD):
        "f-rd" => {
            let (x, y, z) = s(k);
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (c2x, c2y, c2z) = ((x*2.0).cos(), (y*2.0).cos(), (z*2.0).cos());
            cx*cy*cz * 4.0
          - (c2x.clone()*c2y.clone() + c2y*c2z.clone() + c2z*c2x)
        }
        // Lidinoid — the only TPMS with an offset constant:
        "lidinoid" => {
            let (x, y, z) = s(k);
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (s2x, s2y, s2z) = ((x.clone()*2.0).sin(), (y.clone()*2.0).sin(), (z.clone()*2.0).sin());
            let (c2x, c2y, c2z) = ((x*2.0).cos(), (y*2.0).cos(), (z*2.0).cos());
            (s2x.clone()*cy.clone()*s2z.clone()
           + s2y.clone()*cz.clone()*s2x.clone()
           + s2z.clone()*cx.clone()*s2y.clone()) * 0.5
          - (c2x.clone()*c2y.clone() + c2y*c2z.clone() + c2z*c2x) * 0.5
          + 0.15
        }
        // Fischer-Koch S:
        "fischer-koch-s" => {
            let (x, y, z) = s(k);
            let (sx, sy, sz) = (x.clone().sin(), y.clone().sin(), z.clone().sin());
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (c2x, c2y, c2z) = ((x*2.0).cos(), (y*2.0).cos(), (z*2.0).cos());
            c2x*sy.clone()*cz.clone()
          + c2y*sz.clone()*cx.clone()
          + c2z*sx*cy
        }
        // Fischer-Koch Y:
        "fischer-koch-y" => {
            let (x, y, z) = s(k);
            let (sx, sy, sz) = (x.clone().sin(), y.clone().sin(), z.clone().sin());
            let (cx, cy, cz) = (x.clone().cos(), y.clone().cos(), z.clone().cos());
            let (s2x, s2y, s2z) = ((x*2.0).sin(), (y*2.0).sin(), (z*2.0).sin());
            cx*cy*cz * 2.0
          + s2x*sy.clone() + s2y*sz.clone() + s2z*sx
        }
        // Fischer-Koch CP:
        "fischer-koch-cp" => {
            let (x, y, z) = s(k);
            let (cx, cy, cz) = (x.cos(), y.cos(), z.cos());
            cx.clone() + cy.clone() + cz.clone() + cx*cy*cz * 4.0
        }
        _ => {
            // Fallback: gyroid at k=4.
            let s = move |f| -> (T, T, T) { (T::x() * f, T::y() * f, T::z() * f) };
            let (x, y, z) = s(4.0);
            x.clone().sin() * y.clone().cos()
          + y.clone().sin() * z.clone().cos()
          + z.clone().sin() * x.clone().cos()
        }
    }
}

/// Human-readable formula for a TPMS surface (shown in the UI).
fn tpms_formula(name: &str) -> &str {
    match name {
        "gyroid"          => "sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = 0",
        "schwarz-p"       => "cos(kx) + cos(ky) + cos(kz) = 0",
        "schwarz-d"       => "sin(kx)sin(ky)sin(kz) + sin(kx)cos(ky)cos(kz)\n  + cos(kx)sin(ky)cos(kz) + cos(kx)cos(ky)sin(kz) = 0",
        "schoen-iwp"      => "2[cos(kx)cos(ky)+cos(ky)cos(kz)+cos(kz)cos(kx)]\n  − [cos(2kx)+cos(2ky)+cos(2kz)] = 0",
        "neovius"         => "3[cos(kx)+cos(ky)+cos(kz)] + 4cos(kx)cos(ky)cos(kz) = 0",
        "f-rd"            => "4cos(kx)cos(ky)cos(kz) − [cos(2kx)cos(2ky)\n  + cos(2ky)cos(2kz) + cos(2kz)cos(2kx)] = 0",
        "lidinoid"        => "½[sin(2kx)cos(ky)sin(kz) + …]\n  − ½[cos(2kx)cos(2ky) + …] + 0.15 = 0",
        "fischer-koch-s"  => "cos(2kx)sin(ky)cos(kz) + cos(2ky)sin(kz)cos(kx)\n  + cos(2kz)sin(kx)cos(ky) = 0",
        "fischer-koch-y"  => "2cos(kx)cos(ky)cos(kz) + sin(2kx)sin(ky)\n  + sin(2ky)sin(kz) + sin(2kz)sin(kx) = 0",
        "fischer-koch-cp" => "cos(kx)+cos(ky)+cos(kz) + 4cos(kx)cos(ky)cos(kz) = 0",
        _ => "(unknown)",
    }
}

/// Compile a Rhai script that returns a `Tree` (the user-typed implicit
/// expression).  Powered by `fidget-rhai`, which already exposes `x, y, z`
/// and full math (sin/cos/sqrt/...) as overloaded operators on Tree.
fn build_tree_from_rhai(src: &str) -> Result<fidget_core::context::Tree, String> {
    let engine = fidget_rhai::engine();
    let tree: fidget_core::context::Tree = engine
        .eval(src)
        .map_err(|e| format!("{e}"))?;
    Ok(tree)
}

// =====================================================================
// 2a. Dual Contouring — sharp features, jagged boundary
// =====================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum MeshBackend { MarchingCubes33, DualContouring }

fn build_dc_mesh(tree: fidget_core::context::Tree, name: &str, depth: u8) -> Mesh {
    use fidget_core::shape::Shape;
    use fidget_core::vm::VmFunction;
    use fidget_mesh::{Octree, Settings};

    let shape = Shape::<VmFunction>::from(tree);
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
    eprintln!(
        "  [DC] {} verts, {} tris (depth={})",
        pos.len(),
        idx.len() / 3,
        depth
    );

    // Step 1: Fix DC winding holes via full pipeline (dedup + BFS +
    // geometric flip fix).  This must happen BEFORE smooth_boundary_ms,
    // because BFS would flip the boundary fan triangles added later.
    let tmp = Mesh::from_triangles("tmp", &pos, &idx);
    let mut pos: Vec<[f32; 3]> = tmp.vertices.iter().map(|v| v.position).collect();
    let mut idx: Vec<u32> = tmp.indices.clone();

    // Step 2: Smooth the open-boundary silhouette by moving boundary
    // vertices onto the true curve C = (box face) ∩ {f=0}.
    smooth_boundary_ms(&shape, &mut pos, &mut idx);

    // Step 3: Build final mesh WITHOUT winding fix (boundary fan winding
    // is already correct from step 2).
    let mut mesh = Mesh::from_triangles_open(name, &pos, &idx);
    overwrite_normals_with_gradient(&shape, &mut mesh);
    mesh
}

// =====================================================================
// 2b. Marching Cubes 33 — boundary naturally smooth
// =====================================================================

fn build_mc33_mesh(tree: fidget_core::context::Tree, name: &str, res: usize) -> Mesh {
    use fidget_core::shape::Shape;
    use fidget_core::vm::VmFunction;

    let shape = Shape::<VmFunction>::from(tree);

    // Float evaluator for MC33 grid sampling.
    let mut float_eval = Shape::<VmFunction>::new_float_slice_eval();
    let float_tape = shape.float_slice_tape(Default::default());
    let f = |x: f32, y: f32, z: f32| -> f32 {
        let xs = [x];
        let ys = [y];
        let zs = [z];
        match float_eval.eval(&float_tape, &xs, &ys, &zs) {
            Ok(r) => r[0],
            Err(_) => 1e9,
        }
    };

    let (mut pos, idx) = marching_cubes::extract(
        f,
        (-1.0, 1.0, res),
        (-1.0, 1.0, res),
        (-1.0, 1.0, res),
    );
    // Clamp all vertices to the sampling domain boundary.
    // MC33 linearly interpolates across cell edges; when a boundary
    // face carries a zero-crossing, the lerp factor can place the
    // vertex fractionally outside [-1, 1]³ due to floating-point
    // noise.  Without clamping, those vertices visually "overflow"
    // the wireframe cube.
    let mut out_of_bounds = 0usize;
    for p in &mut pos {
        for v in &mut *p {
            if *v < -1.0 { *v = -1.0; out_of_bounds += 1; }
            if *v >  1.0 { *v =  1.0; out_of_bounds += 1; }
        }
    }
    if out_of_bounds > 0 {
        eprintln!("  [MC33] clamped {} vertex coordinates to [-1,1]", out_of_bounds);
    }
    eprintln!("  [MC33] {} verts, {} tris (res={})",
        pos.len(), idx.len() / 3, res);

    // MC33 winding fix via full pipeline.
    let mut tmp = Mesh::from_triangles("tmp", &pos, &idx);
    tmp.fix_winding();
    eprintln!("  [MC33] winding fix applied");

    let mut mesh = Mesh::from_triangles_open(name,
        &tmp.vertices.iter().map(|v| v.position).collect::<Vec<_>>(),
        &tmp.indices);
    overwrite_normals_with_gradient(&shape, &mut mesh);
    mesh
}

/// Recompute area-weighted smooth normals from triangle geometry.
/// Used after `clip_mesh_to_ball` which creates new vertices with
/// zero normals.
fn recompute_smooth_normals(mesh: &mut Mesh) {
    let n = mesh.vertices.len();
    let mut normals = vec![[0.0_f32; 3]; n];
    for tri in mesh.indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let p0 = glam::Vec3::from(mesh.vertices[i0].position);
        let p1 = glam::Vec3::from(mesh.vertices[i1].position);
        let p2 = glam::Vec3::from(mesh.vertices[i2].position);
        let nrm = (p1 - p0).cross(p2 - p0);
        for &i in &[i0, i1, i2] {
            normals[i][0] += nrm.x;
            normals[i][1] += nrm.y;
            normals[i][2] += nrm.z;
        }
    }
    for (vert, norm) in mesh.vertices.iter_mut().zip(normals.iter()) {
        let len = (norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]).sqrt();
        vert.normal = if len > 1e-10 {
            let inv = 1.0 / len;
            [norm[0]*inv, norm[1]*inv, norm[2]*inv]
        } else {
            [0.0, 1.0, 0.0]
        };
    }
}

/// Build a wireframe mesh of the cube [-1,1]³ (12 edges).
/// Each edge is a degenerate triangle (A,B,A) so that PBR rasterisation
/// produces no fragments (zero area) but `extract_edge_indices` yields
/// the line segment (A,B) for the edge-overlay pass.
fn build_box_wireframe() -> Mesh {
    let c = 1.0_f32;
    let pts: [[f32;3]; 8] = [
        [-c,-c,-c], [ c,-c,-c], [ c, c,-c], [-c, c,-c],
        [-c,-c, c], [ c,-c, c], [ c, c, c], [-c, c, c],
    ];
    let edges = [
        (0,1),(1,2),(2,3),(3,0), // bottom
        (4,5),(5,6),(6,7),(7,4), // top
        (0,4),(1,5),(2,6),(3,7), // verticals
    ];
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for &(a,b) in &edges {
        let base = positions.len() as u32;
        positions.push(pts[a]);
        positions.push(pts[b]);
        // degenerate triangle (A, B, A): zero area, yields edge (A,B)
        indices.extend_from_slice(&[base, base+1, base]);
    }
    wireframe_mesh_from_segments("box-wireframe", positions, indices)
}

/// Build a wireframe mesh of the sphere r (latitude/longitude lines).
/// Each segment is a degenerate triangle (A,B,A).
fn build_sphere_wireframe(r: f32, n_lat: usize, n_lon: usize) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let push_seg = |positions: &mut Vec<[f32;3]>, indices: &mut Vec<u32>, p0: [f32;3], p1: [f32;3]| {
        let base = positions.len() as u32;
        positions.push(p0);
        positions.push(p1);
        indices.extend_from_slice(&[base, base+1, base]);
    };
    // Meridians (fixed azimuth, vary polar angle)
    for i in 0..n_lon {
        let az = (i as f32) / (n_lon as f32) * std::f32::consts::TAU;
        let ca = az.cos(); let sa = az.sin();
        for j in 0..(2*n_lat) {
            let t0 = (j as f32) / ((2*n_lat) as f32) * std::f32::consts::PI;
            let t1 = ((j+1) as f32) / ((2*n_lat) as f32) * std::f32::consts::PI;
            push_seg(&mut positions, &mut indices,
                [r*t0.sin()*ca, r*t0.cos(), r*t0.sin()*sa],
                [r*t1.sin()*ca, r*t1.cos(), r*t1.sin()*sa]);
        }
    }
    // Parallels (fixed polar angle, vary azimuth)
    for j in 1..n_lat {
        let pol = (j as f32) / (n_lat as f32) * std::f32::consts::PI;
        let y = r * pol.cos();
        let rr = r * pol.sin();
        for i in 0..(2*n_lat) {
            let a0 = (i as f32) / ((2*n_lat) as f32) * std::f32::consts::TAU;
            let a1 = ((i+1) as f32) / ((2*n_lat) as f32) * std::f32::consts::TAU;
            push_seg(&mut positions, &mut indices,
                [rr*a0.cos(), y, rr*a0.sin()],
                [rr*a1.cos(), y, rr*a1.sin()]);
        }
    }
    wireframe_mesh_from_segments("sphere-wireframe", positions, indices)
}

/// Construct a Mesh of degenerate triangles directly, bypassing the
/// `dedup_vertices` / `fix_winding` pipeline used by
/// `Mesh::from_triangles_open`.  Those passes drop zero-area triangles
/// (which is exactly what every segment (A,B,A) is), erasing the entire
/// wireframe.  We keep all vertices and indices verbatim.
fn wireframe_mesh_from_segments(
    name: &str,
    positions: Vec<[f32;3]>,
    indices: Vec<u32>,
) -> Mesh {
    use engvis_core::mesh::SubMesh;
    let mut aabb = engvis_core::aabb::Aabb::empty();
    for p in &positions {
        aabb.expand(glam::Vec3::from(*p));
    }
    let vertices: Vec<MeshVertex> = positions.into_iter().map(|p| MeshVertex {
        position: p,
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
    }).collect();
    let index_count = indices.len() as u32;
    Mesh {
        name: name.to_string(),
        vertices,
        indices,
        sub_meshes: vec![SubMesh { material_index: 0, index_offset: 0, index_count }],
        aabb,
    }
}

/// Clip a mesh to the interior of a ball (center `c`, radius `r`) by
/// **exact spherical cutting**: each triangle straddling the sphere is
/// split at the exact sphere-edge intersection points, so the resulting
/// boundary vertices lie on the sphere and the boundary is a discrete
/// approximation of the curve (sphere) ∩ (surface).
///
/// Cases per triangle (d_i = |V_i-c|² - r², inside ⇔ d_i ≤ 0):
///   • 3 inside : keep
///   • 0 inside : discard
///   • 1 inside : split into 1 triangle (V_in, A, B)
///   • 2 inside : split into 2 triangles (V_next, V_prev, A) + (V_next, A, B)
/// where A,B are sphere-edge intersection points.  Winding is preserved
/// from the original triangle.  New vertices get zero normals; the caller
/// should re-run `overwrite_normals_with_gradient` to fix them.
fn clip_mesh_to_ball(mesh: &mut Mesh, c: [f32; 3], r: f32) {
    let r2 = r * r;
    let d2 = |p: [f32; 3]| -> f32 {
        let dx = p[0] - c[0];
        let dy = p[1] - c[1];
        let dz = p[2] - c[2];
        dx * dx + dy * dy + dz * dz - r2
    };
    // Sphere-edge intersection: t = dA / (dA - dB), P = A + t*(B-A).
    // Valid when dA, dB have opposite signs (one inside, one outside).
    let hit = |a: [f32; 3], b: [f32; 3], da: f32, db: f32| -> [f32; 3] {
        let t = da / (da - db);
        [a[0] + t * (b[0] - a[0]),
         a[1] + t * (b[1] - a[1]),
         a[2] + t * (b[2] - a[2])]
    };

    // Nested helpers avoid closure borrow conflicts over new_pos/vert_map.
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
                let cc = add_orig(&positions, &mut new_pos, &mut vert_map, i[2]);
                new_idx.extend_from_slice(&[a, b, cc]);
            }
            0 => { /* fully outside: discard */ }
            1 => {
                // Single inside vertex i0; two outside o1=(i0+1)%3, o2=(i0+2)%3.
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
                // Single outside vertex o; prev=(o+2)%3, next=(o+1)%3 inside.
                let o = inside.iter().position(|&x| !x).unwrap();
                let prev = (o + 2) % 3;
                let next = (o + 1) % 3;
                let a = hit(v[prev], v[o], d[prev], d[o]); // edge prev-o
                let b = hit(v[next], v[o], d[next], d[o]); // edge next-o
                let inext = add_orig(&positions, &mut new_pos, &mut vert_map, i[next]);
                let iprev = add_orig(&positions, &mut new_pos, &mut vert_map, i[prev]);
                let iaa = add_new(&mut new_pos, a);
                let ibb = add_new(&mut new_pos, b);
                // Quad (next, prev, A, B) → two triangles, winding preserved.
                new_idx.extend_from_slice(&[inext, iprev, iaa]);
                new_idx.extend_from_slice(&[inext, iaa, ibb]);
            }
            _ => unreachable!(),
        }
    }

    // Rebuild vertices (normals zeroed; caller re-runs overwrite_normals_with_gradient).
    // tangent must be non-zero (e.g. [1,0,0,1]); a zero tangent makes
    // `normalize(world_tangent)` NaN in the shader, which propagates
    // through the TBN matrix and turns the fragment black.
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
    // Recompute AABB.
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

fn build_mesh(
    tree: fidget_core::context::Tree,
    name: &str,
    backend: MeshBackend,
    depth: u8,
    mc_res: usize,
    clip_to_unit_ball: bool,
    clip_radius: f32,
) -> Mesh {
    let mut mesh = match backend {
        MeshBackend::DualContouring => build_dc_mesh(tree, name, depth),
        MeshBackend::MarchingCubes33 => build_mc33_mesh(tree, name, mc_res),
    };
    // Clip the open implicit surface to the ball when requested.
    if clip_to_unit_ball {
        clip_mesh_to_ball(&mut mesh, [0.0, 0.0, 0.0], clip_radius);
        eprintln!("  [clip] {} verts, {} tris after ball clip (r={})",
            mesh.vertices.len(), mesh.indices.len() / 3, clip_radius);
        recompute_smooth_normals(&mut mesh);
    }
    mesh
}

// =====================================================================
// 3. Box-face helpers and Marching Squares (for MS-loop visualization)
// =====================================================================

#[derive(Clone, Copy)]
enum Face { Xp, Xm, Yp, Ym, Zp, Zm }

impl Face {
    fn lock(self) -> (usize, f32) {
        match self {
            Face::Xp => (0,  1.0),  Face::Xm => (0, -1.0),
            Face::Yp => (1,  1.0),  Face::Ym => (1, -1.0),
            Face::Zp => (2,  1.0),  Face::Zm => (2, -1.0),
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
        &self, shape: &fidget_core::shape::Shape<F, ()>, u: f32, v: f32,
    ) -> f32 {
        let (ax, sign) = self.lock();
        let (ua, va) = self.free_axes();
        let mut p = [0.0_f32; 3];
        p[ax] = sign; p[ua] = u; p[va] = v;
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
    face: Face, shape: &fidget_core::shape::Shape<F, ()>, res: usize,
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
            let tl = grid[j][i];   let tr = grid[j][i+1];
            let bl = grid[j+1][i]; let br = grid[j+1][i+1];
            let case = ((if tl < 0.0 {1}else{0})
                      | (if tr < 0.0 {2}else{0})
                      | (if br < 0.0 {4}else{0})
                      | (if bl < 0.0 {8}else{0})) as u8;
            if case == 0 || case == 15 { continue; }
            let lerp_a = |a: f32, b: f32| {
                let d = b - a;
                if d.abs() < 1e-12 { 0.5 } else { -a / d }
            };
            let top    = [-1.0+(i as f32+lerp_a(tl,tr))*step, -1.0+j as f32*step];
            let bottom = [-1.0+(i as f32+lerp_a(bl,br))*step, -1.0+(j+1) as f32*step];
            let left   = [-1.0+i as f32*step, -1.0+(j as f32+lerp_a(tl,bl))*step];
            let right  = [-1.0+(i+1) as f32*step, -1.0+(j as f32+lerp_a(tr,br))*step];
            match case {
                1|14  => { segs.push((left,  bottom)); }
                2|13  => { segs.push((bottom,right )); }
                3|12  => { segs.push((left,  right )); }
                4|11  => { segs.push((top,   right )); }
                5     => { segs.push((left,top)); segs.push((bottom,right)); }
                6|9   => { segs.push((top,bottom)); }
                7|8   => { segs.push((left,right)); }
                10    => { segs.push((top,left)); segs.push((bottom,right)); }
                _ => {}
            }
        }
    }
    segs
}

fn dist3_2(a:[f32;3],b:[f32;3])->f32{
    (a[0]-b[0]).powi(2)+(a[1]-b[1]).powi(2)+(a[2]-b[2]).powi(2)
}

/// Extract 3D boundary loops (f=0) on all 6 box faces via Marching Squares.
fn extract_ms_loops_3d<F:fidget_core::eval::Function+Clone>(
    shape:&fidget_core::shape::Shape<F,()>, res:usize,
)->Vec<Vec<[f32;3]>>{
    let mut all_segs:Vec<([f32;3],[f32;3])>=Vec::new();
    for face in [Face::Xp,Face::Xm,Face::Yp,Face::Ym,Face::Zp,Face::Zm]{
        let segs_2d=marching_squares_face(face,shape,res);
        let (ax,sign)=face.lock();
        let (ua,va)=face.free_axes();
        for &(a,b) in &segs_2d{
            let to3d=|p:[f32;2]|{
                let mut q=[0.0_f32;3];
                q[ax]=sign; q[ua]=p[0]; q[va]=p[1];
                q
            };
            all_segs.push((to3d(a),to3d(b)));
        }
    }
    // Chain segments into closed loops.
    let mut loops:Vec<Vec<[f32;3]>>=Vec::new();
    let mut rem=all_segs.clone();
    while !rem.is_empty(){
        let mut lp=vec![rem[0].0,rem[0].1];
        rem.remove(0);
        for _ in 0..100000{
            let last=*lp.last().unwrap();
            let mut found:Option<(usize,bool)>=None;
            for (si,&(a,b)) in rem.iter().enumerate(){
                if dist3_2(a,last)<1e-4{found=Some((si,true));break;}
                if dist3_2(b,last)<1e-4{found=Some((si,false));break;}
            }
            match found{
                Some((si,fwd))=>{
                    let (a,b)=rem.remove(si);
                    lp.push(if fwd{b}else{a});
                }
                None=>break,
            }
        }
        if lp.len()>=3 && dist3_2(lp[0],*lp.last().unwrap())<1e-4{
            lp.pop();
        }
        if lp.len()>=3{loops.push(lp);}
    }
    loops
}

/// Build a thin triangle strip mesh representing MS boundary loops.
fn build_ms_loops_mesh<F:fidget_core::eval::Function+Clone>(
    shape:&fidget_core::shape::Shape<F,()>, res:usize,
)->Mesh{
    let loops=extract_ms_loops_3d(shape,res);
    let mut positions:Vec<[f32;3]>=Vec::new();
    let mut indices:Vec<u32>=Vec::new();
    let half_w=0.008_f32;
    for lp in &loops{
        let n=lp.len();
        if n<2{continue;}
        for i in 0..n{
            let p0=lp[i];
            let p1=lp[(i+1)%n];
            let dir=[p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]];
            let len=(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]).sqrt();
            if len<1e-10{continue;}
            let dn=[dir[0]/len,dir[1]/len,dir[2]/len];
            let axis=if dn[0].abs()<dn[1].abs()&&dn[0].abs()<dn[2].abs(){[1.0_f32,0.0,0.0]}
                     else if dn[1].abs()<dn[2].abs(){[0.0,1.0,0.0]}
                     else{[0.0,0.0,1.0]};
            let perp=[
                dn[1]*axis[2]-dn[2]*axis[1],
                dn[2]*axis[0]-dn[0]*axis[2],
                dn[0]*axis[1]-dn[1]*axis[0],
            ];
            let pl=(perp[0]*perp[0]+perp[1]*perp[1]+perp[2]*perp[2]).sqrt();
            let perp=if pl>1e-10{[perp[0]/pl*half_w,perp[1]/pl*half_w,perp[2]/pl*half_w]}else{[half_w,0.0,0.0]};
            let base=positions.len() as u32;
            positions.push([p0[0]+perp[0],p0[1]+perp[1],p0[2]+perp[2]]);
            positions.push([p0[0]-perp[0],p0[1]-perp[1],p0[2]-perp[2]]);
            positions.push([p1[0]+perp[0],p1[1]+perp[1],p1[2]+perp[2]]);
            positions.push([p1[0]-perp[0],p1[1]-perp[1],p1[2]-perp[2]]);
            indices.extend_from_slice(&[base,base+1,base+2, base+1,base+3,base+2]);
        }
    }
    eprintln!("  [MS-loops] {} loops, {} verts, {} tris",
              loops.len(),positions.len(),indices.len()/3);
    let mut mesh=Mesh::from_triangles_open("ms-loops",&positions,&indices);
    for v in &mut mesh.vertices{ v.normal=[0.0,1.0,0.0]; }
    mesh
}

// =====================================================================
// 4. Boundary smoothing: move DC boundary vertices onto curve C
// =====================================================================

fn smooth_boundary_ms<F:fidget_core::eval::Function+Clone>(
    shape:&fidget_core::shape::Shape<F,()>,
    positions:&mut Vec<[f32;3]>, indices:&mut Vec<u32>,
){
    use std::collections::{HashMap,HashSet};

    // Identify DC boundary edges (edges belonging to exactly one triangle).
    let mut edge_cnt:HashMap<(u32,u32),u32>=HashMap::new();
    for tri in indices.chunks_exact(3){
        let (a,b,c)=(tri[0],tri[1],tri[2]);
        for &(i0,i1) in &[(a,b),(b,c),(c,a)]{
            let key=if i0<=i1{(i0,i1)}else{(i1,i0)};
            *edge_cnt.entry(key).or_default()+=1;
        }
    }
    let mut bnd_verts:HashSet<u32>=HashSet::new();
    for (&(a,b),&cnt) in &edge_cnt{
        if cnt==1{
            bnd_verts.insert(a);
            bnd_verts.insert(b);
        }
    }
    if bnd_verts.is_empty(){return;}

    // Project a 3D point onto curve C = (box face) ∩ {f=0}.
    // Only project if the point is close to a box face (|coord| > 0.9).
    let project_to_c=|p:[f32;3]|->Option<[f32;3]>{
        let ax=[p[0].abs(),p[1].abs(),p[2].abs()];
        let max_ax=ax.iter().cloned().fold(0.0_f32, f32::max);
        if max_ax < 0.9 { return None; }

        let lock_ax=if ax[0]>=ax[1]&&ax[0]>=ax[2]{0}
                    else if ax[1]>=ax[2]{1}else{2};
        let sign=if p[lock_ax]>=0.0{1.0_f32}else{-1.0};
        let face=match (lock_ax,sign){
            (0, 1.0)=>Face::Xp,(0,-1.0)=>Face::Xm,
            (1, 1.0)=>Face::Yp,(1,-1.0)=>Face::Ym,
            (2, 1.0)=>Face::Zp,(2,-1.0)=>Face::Zm,
            _=>return None,
        };
        let (ua,va)=face.free_axes();
        let mut u=p[ua]; let mut v=p[va];
        for _ in 0..24{
            let fval=face.eval_uv(shape,u,v);
            if fval.abs()<1e-8{break;}
            let eps=1e-6;
            let gx=(face.eval_uv(shape,u+eps,v)-fval)/eps;
            let gy=(face.eval_uv(shape,u,v+eps)-fval)/eps;
            let m=gx*gx+gy*gy;
            if m<1e-20{break;}
            let s=fval/m;
            u-=s*gx; v-=s*gy;
        }
        let mut q=[0.0_f32;3]; q[lock_ax]=sign; q[ua]=u; q[va]=v;
        Some(q)
    };

    // Move each boundary vertex onto curve C.
    let mut moved=0u32;
    for &vi in &bnd_verts{
        let p=positions[vi as usize];
        if let Some(pc)=project_to_c(p){
            positions[vi as usize]=pc;
            moved+=1;
        }
    }
    eprintln!("  smooth_boundary_ms: moved {} / {} boundary vertices to curve C",
              moved, bnd_verts.len());
}

/// Overwrite per-vertex normals with the analytic surface gradient ∇f.
fn overwrite_normals_with_gradient<F:fidget_core::eval::Function+Clone>(
    shape: &fidget_core::shape::Shape<F, ()>, mesh: &mut Mesh,
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
            let len = (g.dx*g.dx + g.dy*g.dy + g.dz*g.dz).sqrt();
            if len > 1e-10 {
                let inv = 1.0 / len;
                mesh.vertices[start + i].normal = [g.dx*inv, g.dy*inv, g.dz*inv];
            }
        }
    }
}

// =====================================================================
// 5. App
// =====================================================================

/// Source of the implicit expression: a built-in name, or a user-typed
/// Rhai script (compiled by `fidget-rhai`).
#[derive(Clone, PartialEq, Eq)]
enum SurfaceSource {
    BuiltIn(&'static str),
    Custom, // Rhai expression in `App::custom_expr`
}

/// Workflow steps shown in the left "trait"-style panel.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step { Source, Mesh, Display, Topology }

const PRIMITIVE_SURFACES: &[(&str, &str)] = &[
    ("sphere", "Sphere"),
    ("torus",  "Torus"),
];

const TPMS_SURFACES: &[(&str, &str)] = &[
    ("gyroid",        "Gyroid"),
    ("schwarz-p",     "Schwarz P"),
    ("schwarz-d",     "Schwarz D"),
    ("schoen-iwp",    "Schoen IWP"),
    ("neovius",       "Neovius"),
    ("f-rd",          "F-RD (Schoen FRD)"),
    ("lidinoid",      "Lidinoid"),
    ("fischer-koch-s", "Fischer-Koch S"),
    ("fischer-koch-y", "Fischer-Koch Y"),
    ("fischer-koch-cp","Fischer-Koch CP"),
];

struct App {
    // ── implicit surface source ──
    source: SurfaceSource,
    /// The name currently shown in the TPMS combo-box.
    selected_tpms: &'static str,
    custom_expr: String,
    custom_error: Option<String>,
    clip_to_unit_ball: bool,
    clip_radius: f32,

    // ── shape parameters (persisted across surface switches) ──
    sphere_radius: f32,
    torus_major_r: f32,
    torus_minor_r: f32,
    tpms_period: f32,

    // ── meshing ──
    mesh_backend: MeshBackend,
    surf_depth: u8,
    mc_res: usize,
    show_ms_loops: bool,
    show_bounding: bool,

    // ── edge / vertex overlay (persisted across rebuilds) ──
    show_surface_edges: bool,
    edge_color: [f32; 3],
    edge_line_width: f32,
    // Bounding-wireframe appearance, independent of the surface-edge
    // overlay above.  Applied as a per-node override in `build_scene`.
    wireframe_color: [f32; 3],
    wireframe_line_width: f32,

    // ── surface appearance ──
    surface_color: [f32; 3],

    // ── workflow / UI ──
    current_step: Step,

    // ── load / save ──
    pending_load: Option<std::path::PathBuf>,
    pending_save: Option<std::path::PathBuf>,

    // ── runtime state ──
    needs_remesh: bool,
    camera_fitted: bool,
    last_topology: Option<engvis_core::topology::MeshTopology>,
    last_build_ok: bool,
}

impl App {
    fn surface_label(&self) -> String {
        match &self.source {
            SurfaceSource::BuiltIn(n) => (*n).to_string(),
            SurfaceSource::Custom => "custom".to_string(),
        }
    }

    fn tree_params(&self) -> TreeParams<'_> {
        let name = match &self.source {
            SurfaceSource::BuiltIn(n) => *n,
            SurfaceSource::Custom => "custom",
        };
        TreeParams {
            name,
            sphere_radius: self.sphere_radius,
            torus_major_r: self.torus_major_r,
            torus_minor_r: self.torus_minor_r,
            tpms_period: self.tpms_period,
        }
    }

    fn current_tree(&self) -> Result<fidget_core::context::Tree, String> {
        match &self.source {
            SurfaceSource::BuiltIn(_) => {
                let p = self.tree_params();
                Ok(build_tree(&p))
            }
            SurfaceSource::Custom => build_tree_from_rhai(&self.custom_expr),
        }
    }

    /// Build the scene with the surface mesh and optional MS-loop overlay.
    /// Updates `last_topology` and `last_build_ok` for the status bar.
    fn build_scene(&mut self) -> Scene {
        use fidget_core::shape::Shape;
        use fidget_core::vm::VmFunction;

        let tree = match self.current_tree() {
            Ok(t) => { self.custom_error = None; t }
            Err(e) => {
                eprintln!("expression error: {e}");
                self.custom_error = Some(e);
                self.last_build_ok = false;
                // Empty scene
                return Scene::default();
            }
        };

        let label = self.surface_label();
        // Adaptive resolution: when the surface features are thinner
        // than ~3 MC33 grid cells the mesh becomes degenerate.
        // Bump the resolution automatically, capped at 512.
        let effective_res = {
            let min_feature = match &self.source {
                // Torus tube diameter ≈ 2 * minor_r
                SurfaceSource::BuiltIn("torus") => 2.0 * self.torus_minor_r,
                // TPMS half-period ≈ π/k
                SurfaceSource::BuiltIn(n)
                    if TPMS_SURFACES.iter().any(|(k,_)| k == n) => std::f32::consts::PI / self.tpms_period,
                _ => 0.5, // sphere & others are well-resolved even at res=16
            };
            let needed = ((6.0 / min_feature) as usize).max(self.mc_res).min(512);
            if needed > self.mc_res {
                eprintln!("  [MC33] auto-bump res {} -> {} (feature {:.3})",
                    self.mc_res, needed, min_feature);
            }
            needed
        };
        let mesh = build_mesh(
            tree.clone(), &label,
            self.mesh_backend, self.surf_depth, effective_res,
            self.clip_to_unit_ball, self.clip_radius,
        );
        // Topology stats for the status bar.
        self.last_topology = Some(compute_topology(&mesh));
        self.last_build_ok = true;

        let mat = PbrMaterial {
            name: "Surface".into(),
            albedo: [self.surface_color[0], self.surface_color[1], self.surface_color[2], 1.0],
            metallic: 0.6,
            roughness: 0.3,
            ..Default::default()
        };
        let mut scene = Scene::single_mesh(&label, mesh, mat);
        // The surface node is index 0; toggle its triangle-edge overlay.
        if let Some(n) = scene.nodes.first_mut() {
            n.render_edges = self.show_surface_edges;
        }

        if self.show_ms_loops {
            let shape = Shape::<VmFunction>::from(tree);
            let ms_mesh = build_ms_loops_mesh(&shape, 64);
            let ms_mat = PbrMaterial {
                name: "MS-loops".into(),
                albedo: [1.0, 0.85, 0.2, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                ..Default::default()
            };
            let mi = scene.meshes.len();
            scene.meshes.push(ms_mesh);
            scene.materials.push(ms_mat);
            scene.nodes.push(SceneNode {
                name: "ms-loops".into(),
                local_transform: glam::Affine3A::IDENTITY,
                mesh_index: Some(mi),
                children: Vec::new(),
                visible: true,
                render_edges: false,
                edge_color_override: None,
                edge_width_override: None,
            });
        }

        if self.show_bounding {
            let wf_mesh = if self.clip_to_unit_ball {
                build_sphere_wireframe(self.clip_radius, 12, 24)
            } else {
                build_box_wireframe()
            };
            let wf_mat = PbrMaterial { name: "wireframe".into(), ..Default::default() };
            let wi = scene.meshes.len();
            scene.meshes.push(wf_mesh);
            scene.materials.push(wf_mat);
            scene.nodes.push(SceneNode {
                name: "wireframe".into(),
                local_transform: glam::Affine3A::IDENTITY,
                mesh_index: Some(wi),
                children: Vec::new(),
                visible: true,
                render_edges: true,
                edge_color_override: Some(self.wireframe_color),
                edge_width_override: Some(self.wireframe_line_width),
            });
        }

        scene
    }

    /// Replace the current scene with a single imported mesh.
    fn load_external_mesh(&mut self, path: &std::path::Path) -> Result<Scene, String> {
        let mesh = mesh_io::load_mesh(path)?;
        self.last_topology = Some(compute_topology(&mesh));
        self.last_build_ok = true;
        let aabb = mesh.aabb;
        let mat = PbrMaterial {
            name: "Imported".into(),
            albedo: [0.7, 0.7, 0.75, 1.0],
            metallic: 0.0,
            roughness: 0.6,
            ..Default::default()
        };
        let scene = Scene::single_mesh(
            path.file_name().and_then(|s| s.to_str()).unwrap_or("imported"),
            mesh, mat,
        );
        let _ = aabb; // (camera fit happens in caller via scene_dirty path)
        let mut scene = scene;
        if let Some(n) = scene.nodes.first_mut() {
            n.render_edges = self.show_surface_edges;
        }
        Ok(scene)
    }
}

impl EngvisApp for App {
    fn config(&self) -> RunConfig {
        RunConfig {
            title: "engvis — Engineering Visualization".into(),
            width: 1280, height: 800,
            ..Default::default()
        }
    }

    fn on_setup(&mut self, _ctx: &mut AppCtx) -> Scene {
        self.build_scene()
    }

    fn on_ready(&mut self, _scene: &Scene, camera: &mut OrbitCamera) {
        camera.fit_to_aabb(Aabb {
            min: glam::Vec3::new(-1.0, -1.0, -1.0),
            max: glam::Vec3::new(1.0, 1.0, 1.0),
        });
        self.camera_fitted = true;
    }

    fn ui(&mut self, egui_ctx: &egui::Context, frame: &mut FrameCtx) {
        // ── Apply pending file actions and remesh ─────────────────
        if self.needs_remesh {
            *frame.scene = self.build_scene();
            *frame.scene_dirty = true;
            self.needs_remesh = false;
            self.camera_fitted = false;
        }
        if let Some(path) = self.pending_load.take() {
            // glTF goes through the renderer's loader (textures, nodes);
            // OBJ/STL/PLY go through mesh_io.
            let ext = path.extension().and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase()).unwrap_or_default();
            match ext.as_str() {
                "gltf" | "glb" => {
                    match load_gltf(path.to_string_lossy().as_ref(),
                        frame.device, frame.queue, frame.texture_cache) {
                        Ok((scene, aabb)) => {
                            *frame.scene = scene;
                            frame.camera.fit_to_aabb(aabb);
                            *frame.scene_dirty = true;
                            // Glb may contain multiple meshes; topology stats are
                            // not aggregated here.
                            self.last_topology = None;
                        }
                        Err(e) => eprintln!("glTF load failed: {e}"),
                    }
                }
                _ => {
                    match self.load_external_mesh(&path) {
                        Ok(scene) => {
                            let aabb = scene.meshes.iter().fold(
                                Aabb::empty(),
                                |mut a, m| { a.expand(glam::Vec3::from(m.aabb.min));
                                             a.expand(glam::Vec3::from(m.aabb.max)); a }
                            );
                            *frame.scene = scene;
                            frame.camera.fit_to_aabb(aabb);
                            *frame.scene_dirty = true;
                        }
                        Err(e) => eprintln!("mesh load failed: {e}"),
                    }
                }
            }
        }
        if let Some(path) = self.pending_save.take() {
            // Save the first non-wireframe mesh in the scene.
            if let Some(mesh) = frame.scene.meshes.first() {
                if let Err(e) = mesh_io::save_mesh(mesh, &path) {
                    eprintln!("mesh save failed: {e}");
                } else {
                    eprintln!("saved {} ({} verts, {} tris)",
                        path.display(), mesh.vertices.len(), mesh.indices.len()/3);
                }
            }
        }

        // ── Edge overlay enabled so the bounding box / sphere shows up;
        //    only nodes with `render_edges=true` are affected.  Edge color
        //    and line width come from the App (controlled in the
        //    Display panel) so the user's choice persists across remeshes.
        frame.render_state.edge_opts.enabled = true;
        frame.render_state.edge_opts.color = self.edge_color;
        frame.render_state.edge_opts.line_width = self.edge_line_width;
        if !self.camera_fitted {
            frame.camera.fit_to_aabb(Aabb {
                min: glam::Vec3::new(-1.0, -1.0, -1.0),
                max: glam::Vec3::new(1.0, 1.0, 1.0),
            });
            self.camera_fitted = true;
        }

        // ── Top menu bar ─────────────────────────────────────────
        egui::TopBottomPanel::top("menu_bar").show(egui_ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open mesh… (OBJ/STL/PLY/glTF)").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Mesh", &["obj", "stl", "ply", "gltf", "glb"])
                            .pick_file() { self.pending_load = Some(p); }
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Save current mesh as OBJ…").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("OBJ", &["obj"])
                            .set_file_name("mesh.obj").save_file()
                        { self.pending_save = Some(p); }
                        ui.close();
                    }
                    if ui.button("Save current mesh as STL…").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("STL", &["stl"])
                            .set_file_name("mesh.stl").save_file()
                        { self.pending_save = Some(p); }
                        ui.close();
                    }
                    if ui.button("Save current mesh as PLY…").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("PLY", &["ply"])
                            .set_file_name("mesh.ply").save_file()
                        { self.pending_save = Some(p); }
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        egui_ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut frame.render_state.show_surface, "Surface");
                    ui.checkbox(&mut frame.render_state.show_grid, "Grid");
                    ui.checkbox(&mut frame.render_state.vertex_opts.enabled, "Points");
                    ui.checkbox(&mut self.show_bounding, "Bounding wireframe")
                        .on_hover_text("Re-mesh to apply");
                    ui.checkbox(&mut self.show_ms_loops, "MS boundary loops")
                        .on_hover_text("Re-mesh to apply");
                });
                ui.menu_button("Help", |ui| {
                    ui.label("engvis — Engineering Visualization");
                    ui.label("Implicit surfaces (Fidget) + DC / MC33 meshing.");
                    ui.label("Custom expression syntax: Rhai (x, y, z, sin, cos, ...).");
                });
            });
        });

        // ── Bottom status bar ─────────────────────────────────────
        egui::TopBottomPanel::bottom("status_bar").show(egui_ctx, |ui| {
            ui.horizontal(|ui| {
                if self.last_build_ok {
                    if let Some(t) = &self.last_topology {
                        ui.label(format!(
                            "V={}  E={}  F={}  χ={}  ∂E={}  comps={}  watertight={}",
                            t.vertices, t.edges, t.faces, t.euler,
                            t.boundary_edges, t.connected_components, t.is_watertight,
                        ));
                    } else {
                        ui.label("no topology");
                    }
                } else {
                    ui.colored_label(egui::Color32::from_rgb(220, 80, 80),
                        "build failed (see expression panel)");
                }
                ui.separator();
                ui.label(format!("FPS {:.0}", frame.fps));
                ui.separator();
                ui.label(format!("backend: {}", match self.mesh_backend {
                    MeshBackend::DualContouring  => "DC",
                    MeshBackend::MarchingCubes33 => "MC33",
                }));
                ui.separator();
                ui.label(format!("surface: {}", self.surface_label()));
            });
        });

        // ── Left "workflow" panel (numbered, trait-style) ──────────
        egui::SidePanel::left("workflow")
            .resizable(true).default_width(180.0)
            .show(egui_ctx, |ui| {
                ui.heading("Workflow");
                ui.add_space(6.0);
                let steps = [
                    (Step::Source,   "1. Source"),
                    (Step::Mesh,     "2. Mesh"),
                    (Step::Display,  "3. Display"),
                    (Step::Topology, "4. Topology"),
                ];
                for (s, label) in steps {
                    if ui.selectable_label(self.current_step == s, label).clicked() {
                        self.current_step = s;
                    }
                }
                ui.add_space(12.0);
                ui.separator();
                ui.label("Click a step → edit details on the right.");
            });

        // ── Right "details" panel ──────────────────────────────────
        egui::SidePanel::right("details")
            .resizable(true).default_width(320.0)
            .show(egui_ctx, |ui| {
                match self.current_step {
                    Step::Source    => self.ui_source(ui),
                    Step::Mesh      => self.ui_mesh(ui),
                    Step::Display   => self.ui_display(ui, frame.render_state),
                    Step::Topology  => self.ui_topology(ui),
                }
            });
    }

    fn on_frame(&mut self, _frame: &mut FrameCtx) {}

    fn on_event(&mut self, _event: &winit::event::WindowEvent) -> EventHandling {
        EventHandling::Default
    }
}

impl App {
    fn ui_source(&mut self, ui: &mut egui::Ui) {
        ui.heading("1. Source");
        ui.label("Implicit surface f(x,y,z) = 0.");
        ui.add_space(6.0);

        // ── Primitive shapes ──────────────────────────────
        ui.label("Primitive shapes:");
        for (key, label) in PRIMITIVE_SURFACES {
            let selected = matches!(&self.source, SurfaceSource::BuiltIn(n) if n == key);
            if ui.selectable_label(selected, *label).clicked() {
                self.source = SurfaceSource::BuiltIn(*key);
                self.needs_remesh = true;
            }
            // Show parameters directly below the selected primitive
            if selected {
                ui.indent((*key, "params"), |ui| {
                    match *key {
                        "sphere" => {
                            if ui.add(egui::Slider::new(&mut self.sphere_radius, 0.1..=3.0)
                                .text("Radius")).changed() {
                                self.needs_remesh = true;
                            }
                        }
                        "torus" => {
                            if ui.add(egui::Slider::new(&mut self.torus_major_r, 0.1..=3.0)
                                .text("Major R")).changed() {
                                self.needs_remesh = true;
                            }
                            if ui.add(egui::Slider::new(&mut self.torus_minor_r, 0.02..=1.5)
                                .text("Minor r")).changed() {
                                self.needs_remesh = true;
                            }
                        }
                        _ => {}
                    }
                });
            }
        }

        ui.add_space(10.0);
        // ── TPMS (dropdown) ───────────────────────────────
        ui.label("Triply Periodic Minimal Surfaces:");
        let prev_idx = TPMS_SURFACES.iter()
            .position(|(k, _)| *k == self.selected_tpms)
            .unwrap_or(0);
        let mut tpms_idx = prev_idx;
        egui::ComboBox::from_id_salt("tpms_combo")
            .width(200.0)
            .selected_text(TPMS_SURFACES[tpms_idx].1)
            .show_ui(ui, |ui| {
                for (i, (_key, label)) in TPMS_SURFACES.iter().enumerate() {
                    ui.selectable_value(&mut tpms_idx, i, *label);
                }
            });
        if tpms_idx != prev_idx {
            self.selected_tpms = TPMS_SURFACES[tpms_idx].0;
            self.source = SurfaceSource::BuiltIn(self.selected_tpms);
            // Reset period to the surface's default
            let mut p = self.tree_params();
            p.set_tpms_defaults(self.selected_tpms);
            self.tpms_period = p.tpms_period;
            self.needs_remesh = true;
        }
        // Show formula + period slider when a TPMS is the active source
        if matches!(&self.source, SurfaceSource::BuiltIn(n)
            if TPMS_SURFACES.iter().any(|(k,_)| k == n))
        {
            ui.indent("tpms_opts", |ui| {
                ui.label("Implicit equation:");
                let formula = tpms_formula(self.selected_tpms);
                ui.code(formula);
                if ui.add(egui::Slider::new(&mut self.tpms_period, 0.5..=10.0)
                    .text("Period k")).changed() {
                    self.needs_remesh = true;
                }
            });
        }

        ui.add_space(10.0);
        ui.separator();
        ui.label("Custom expression (Rhai):");
        let resp = ui.add(egui::TextEdit::multiline(&mut self.custom_expr)
            .desired_rows(3).code_editor());
        if resp.lost_focus() && resp.ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
            self.source = SurfaceSource::Custom;
            self.needs_remesh = true;
        }
        if ui.button("Use custom expression").clicked() {
            self.source = SurfaceSource::Custom;
            self.needs_remesh = true;
        }
        if let Some(err) = &self.custom_error {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
        }
        ui.label("Examples:");
        ui.code("x*x + y*y + z*z - 0.64");
        ui.code("sin(4*x)*cos(4*y) + sin(4*y)*cos(4*z) + sin(4*z)*cos(4*x)");

        ui.add_space(8.0);
        ui.separator();
        // ── Clip ──────────────────────────────────────────
        ui.label("Bounding region:");
        if ui.checkbox(&mut self.clip_to_unit_ball, "Clip to ball").changed() {
            self.needs_remesh = true;
        }
        ui.indent("clip_opts", |ui| {
            if ui.add(egui::Slider::new(&mut self.clip_radius, 0.2..=5.0)
                .text("Clip radius")).changed() {
                self.needs_remesh = true;
            }
        });
    }

    fn ui_mesh(&mut self, ui: &mut egui::Ui) {
        ui.heading("2. Mesh");
        ui.label("Polygonisation backend:");
        if ui.selectable_label(self.mesh_backend == MeshBackend::MarchingCubes33,
            "Marching Cubes 33 (smooth boundary)").clicked() {
            self.mesh_backend = MeshBackend::MarchingCubes33;
            self.needs_remesh = true;
        }
        if ui.selectable_label(self.mesh_backend == MeshBackend::DualContouring,
            "Dual Contouring + boundary smoothing").clicked() {
            self.mesh_backend = MeshBackend::DualContouring;
            self.needs_remesh = true;
        }
        ui.add_space(8.0);
        match self.mesh_backend {
            MeshBackend::DualContouring => {
                if ui.add(egui::Slider::new(&mut self.surf_depth, 3..=10)
                    .text("Octree depth")).changed() {
                    self.needs_remesh = true;
                }
            }
            MeshBackend::MarchingCubes33 => {
                if ui.add(egui::Slider::new(&mut self.mc_res, 16..=256)
                    .step_by(8.0).text("Grid resolution")).changed() {
                    self.needs_remesh = true;
                }
            }
        }
        ui.add_space(8.0);
        if ui.button("Re-mesh").clicked() {
            self.needs_remesh = true;
        }
    }

    fn ui_display(&mut self, ui: &mut egui::Ui,
        render_state: &mut engvis_core::material::RenderState)
    {
        ui.heading("3. Display");

        // ── Surface ────────────────────────────────────────
        ui.checkbox(&mut render_state.show_surface, "Show triangle surface");
        if render_state.show_surface {
            ui.indent("surface_opts", |ui| {
                if ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut self.surface_color)
                }).inner.changed() {
                    self.needs_remesh = true;
                }
                ui.add(egui::Slider::new(&mut render_state.opacity, 0.0..=1.0)
                    .text("Opacity"));
            });
        }

        // ── Edges ──────────────────────────────────────────
        // Triangle-mesh edges of the surface node — toggling this flag
        // requires reapplying `render_edges` on the node, which happens
        // on the next remesh.
        if ui.checkbox(&mut self.show_surface_edges, "Show triangle edges").changed() {
            self.needs_remesh = true;
        }
        if self.show_surface_edges {
            ui.indent("edge_opts", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut self.edge_color);
                });
                ui.add(egui::Slider::new(&mut self.edge_line_width, 0.5..=10.0)
                    .text("Line width (px)"));
            });
        }

        // ── Points ─────────────────────────────────────────
        ui.checkbox(&mut render_state.vertex_opts.enabled, "Show points");
        if render_state.vertex_opts.enabled {
            ui.indent("point_opts", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut render_state.vertex_opts.color);
                });
                ui.add(egui::Slider::new(&mut render_state.vertex_opts.point_size, 1.0..=12.0)
                    .text("Point size"));
            });
        }

        // ── Other overlays ─────────────────────────────────
        ui.add_space(6.0);
        ui.separator();
        ui.checkbox(&mut render_state.show_grid, "World grid");
        if ui.checkbox(&mut self.show_bounding, "Show bounding wireframe").changed() {
            self.needs_remesh = true;
        }
        if self.show_bounding {
            ui.indent("wireframe_opts", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    if ui.color_edit_button_rgb(&mut self.wireframe_color).changed() {
                        self.needs_remesh = true;
                    }
                });
                if ui.add(egui::Slider::new(&mut self.wireframe_line_width, 0.5..=10.0)
                    .text("Line width (px)")).changed() {
                    self.needs_remesh = true;
                }
            });
        }
        if ui.checkbox(&mut self.show_ms_loops, "MS boundary loops").changed() {
            self.needs_remesh = true;
        }
    }

    fn ui_topology(&mut self, ui: &mut egui::Ui) {
        ui.heading("4. Topology");
        match &self.last_topology {
            None => { ui.label("(no mesh loaded)"); }
            Some(t) => {
                egui::Grid::new("topo_grid").num_columns(2).striped(true).show(ui, |ui| {
                    ui.label("Vertices  V"); ui.label(format!("{}", t.vertices)); ui.end_row();
                    ui.label("Edges     E"); ui.label(format!("{}", t.edges)); ui.end_row();
                    ui.label("Faces     F"); ui.label(format!("{}", t.faces)); ui.end_row();
                    ui.label("Euler     χ = V−E+F"); ui.label(format!("{}", t.euler)); ui.end_row();
                    ui.label("Boundary edges"); ui.label(format!("{}", t.boundary_edges)); ui.end_row();
                    ui.label("Non-manifold edges"); ui.label(format!("{}", t.non_manifold_edges)); ui.end_row();
                    ui.label("Connected components"); ui.label(format!("{}", t.connected_components)); ui.end_row();
                    ui.label("Watertight"); ui.label(format!("{}", t.is_watertight)); ui.end_row();
                });
                ui.add_space(6.0);
                ui.label("χ legend: 2=sphere, 0=torus, −2=double torus, …");
            }
        }
    }
}

fn main() {
    env_logger::init();
    engvis_renderer::run(App {
        source: SurfaceSource::BuiltIn("gyroid"),
        selected_tpms: "gyroid",
        custom_expr: "sin(4*x)*cos(4*y) + sin(4*y)*cos(4*z) + sin(4*z)*cos(4*x)".to_string(),
        custom_error: None,
        clip_to_unit_ball: false,
        clip_radius: 1.0,
        sphere_radius: 0.8,
        torus_major_r: 0.6,
        torus_minor_r: 0.2,
        tpms_period: 4.0,
        mesh_backend: MeshBackend::MarchingCubes33,
        surf_depth: 7,
        mc_res: 64,
        show_ms_loops: false,
        show_bounding: true,
        show_surface_edges: false,
        edge_color: [0.9, 0.9, 0.9],
        edge_line_width: 2.0,
        wireframe_color: [0.9, 0.9, 0.9],
        wireframe_line_width: 2.0,
        surface_color: [0.2, 0.6, 0.9],
        current_step: Step::Source,
        pending_load: None,
        pending_save: None,
        needs_remesh: false,
        camera_fitted: false,
        last_topology: None,
        last_build_ok: true,
    });
}
