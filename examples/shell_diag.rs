// ── Headless diagnostic for Shell mesh topology ──────────
// Evaluates the Gyroid shell CSG formula at various MC33 resolutions
// and reports detailed topology (components, boundary classification, etc.)
//
// Run:  cargo run --release --example shell_diag

use std::collections::HashMap;

fn build_gyroid_shell_tree(half_t: f32) -> fidget_core::context::Tree {
    use fidget_core::context::Tree as T;
    let k = 4.0_f32; // default TPMS period
    let (x, y, z) = (T::x() * k, T::y() * k, T::z() * k);
    let f = x.clone().sin() * y.clone().cos()
          + y.clone().sin() * z.clone().cos()
          + z.clone().sin() * x.clone().cos();
    // Shell = |f| − t'/2 = max(f − t'/2, −f − t'/2)
    let ft = f.clone() - half_t;
    let nft = f * -1.0 - half_t;
    ft.max(nft)
}

fn build_gyroid_skeletal_tree(c: f32) -> fidget_core::context::Tree {
    use fidget_core::context::Tree as T;
    let k = 4.0_f32;
    let (x, y, z) = (T::x() * k, T::y() * k, T::z() * k);
    let f = x.clone().sin() * y.clone().cos()
          + y.clone().sin() * z.clone().cos()
          + z.clone().sin() * x.clone().cos();
    f - c
}

fn run_mc33(tree: &fidget_core::context::Tree, res: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
    use fidget_core::shape::Shape;
    use fidget_core::vm::VmFunction;

    let shape = Shape::<VmFunction>::from(tree.clone());
    let mut float_eval = Shape::<VmFunction>::new_float_slice_eval();
    let tape = shape.float_slice_tape(Default::default());
    let eval = |x: f32, y: f32, z: f32| -> f32 {
        match float_eval.eval(&tape, &[x], &[y], &[z]) {
            Ok(r) => r[0],
            Err(_) => 1e9,
        }
    };

    let (mut pos, idx) = engvis_core::marching_cubes::extract(
        eval,
        (-1.0, 1.0, res),
        (-1.0, 1.0, res),
        (-1.0, 1.0, res),
    );
    // Clamp to [-1,1]
    for p in &mut pos {
        for v in p.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
    }
    (pos, idx)
}

fn analyse(pos: &[[f32; 3]], idx: &[u32], tag: &str) {
    // Dedup (same eps as from_triangles_open)
    let aabb_diag = {
        let mut mn = [f32::MAX; 3];
        let mut mx = [f32::MIN; 3];
        for p in pos {
            for i in 0..3 { mn[i] = mn[i].min(p[i]); mx[i] = mx[i].max(p[i]); }
        }
        let d = [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]];
        (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt()
    };
    let eps = aabb_diag * 1e-6;

    let mut dpos = pos.to_vec();
    let mut didx = idx.to_vec();
    engvis_core::mesh::dedup_vertices(&mut dpos, &mut didx, eps);

    let nv = dpos.len();
    let nf = didx.len() / 3;

    // Edge → face count
    let mut edge_cnt: HashMap<(u32, u32), usize> = HashMap::with_capacity(nf * 3);
    for tri in didx.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        for &(i0, i1) in &[(a, b), (b, c), (c, a)] {
            let key = if i0 <= i1 { (i0, i1) } else { (i1, i0) };
            *edge_cnt.entry(key).or_default() += 1;
        }
    }
    let ne = edge_cnt.len();

    // Union-Find for components
    let mut parent: Vec<usize> = (0..nf).collect();
    let mut rank_uf: Vec<u32> = vec![0; nf];
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x { p[x] = p[p[x]]; x = p[x]; } x
    }
    fn union(p: &mut [usize], r: &mut [u32], mut a: usize, mut b: usize) {
        a = find(p, a); b = find(p, b);
        if a == b { return; }
        match r[a].cmp(&r[b]) {
            std::cmp::Ordering::Less => { p[a] = b; }
            std::cmp::Ordering::Greater => { p[b] = a; }
            std::cmp::Ordering::Equal => { p[b] = a; r[a] += 1; }
        }
    }

    let mut boundary_edges = 0usize;
    let mut non_manifold = 0usize;
    let mut on_box_face = 0usize;
    let mut interior_bnd = 0usize;

    let tol = 0.02;
    for (&(a, b), &cnt) in &edge_cnt {
        match cnt {
            1 => {
                boundary_edges += 1;
                let pa = dpos[a as usize];
                let pb = dpos[b as usize];
                let on_f = |p: [f32; 3]| -> bool {
                    p[0].abs() > 1.0 - tol || p[1].abs() > 1.0 - tol || p[2].abs() > 1.0 - tol
                };
                if on_f(pa) && on_f(pb) { on_box_face += 1; } else { interior_bnd += 1; }
            }
            2 => {
                // Find the two faces sharing this edge and union them.
                // We need to re-derive which faces share each edge.
            }
            _ => { non_manifold += 1; }
        }
    }

    // Re-derive face adjacency for union-find (we skipped the 2-case above)
    for tri_idx in 0..nf {
        let base = tri_idx * 3;
        let a = didx[base]; let b = didx[base+1]; let c = didx[base+2];
        let edges = [(a,b),(b,c),(c,a)];
        for &(i0, i1) in &edges {
            let key = if i0 <= i1 { (i0, i1) } else { (i1, i0) };
            // For component counting, union all faces sharing this edge
            // (This is O(F*E) but fine for diagnostics)
            let _ = key; // handled below
        }
    }
    // More efficient: build edge → [face_idx] map
    let mut edge_faces_map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, tri) in didx.chunks_exact(3).enumerate() {
        let a = tri[0]; let b = tri[1]; let c = tri[2];
        for &(i0, i1) in &[(a,b),(b,c),(c,a)] {
            let key = if i0 <= i1 { (i0, i1) } else { (i1, i0) };
            edge_faces_map.entry(key).or_default().push(fi);
        }
    }
    for faces in edge_faces_map.values() {
        if faces.len() >= 2 {
            for i in 1..faces.len() {
                union(&mut parent, &mut rank_uf, faces[0], faces[i]);
            }
        }
    }
    let mut roots: Vec<usize> = (0..nf).map(|i| find(&mut parent, i)).collect();
    roots.sort();
    roots.dedup();
    let components = roots.len();

    let euler = nv as i64 - ne as i64 + nf as i64;
    let watertight = boundary_edges == 0 && components == 1;

    eprintln!(
        "[{tag}] res={} → V={} E={} F={} | χ={} components={} | boundary: {} total ({} on_box_face, {} interior) | non_manifold={} | watertight={}",
        0, nv, ne, nf, euler, components, boundary_edges, on_box_face, interior_bnd, non_manifold, watertight,
    );

    // Component size distribution
    let mut comp_sizes: HashMap<usize, usize> = HashMap::new();
    for fi in 0..nf {
        let r = find(&mut parent, fi);
        *comp_sizes.entry(r).or_default() += 1;
    }
    let mut sizes: Vec<usize> = comp_sizes.values().copied().collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let show = sizes.len().min(10);
    eprintln!("  component sizes (top {}): {:?}", show, &sizes[..show]);
    if sizes.len() > show {
        eprintln!("  ... and {} more components (smallest: {} faces)", sizes.len() - show, sizes.last().unwrap_or(&0));
    }
}

/// Apply face-grid-fill capping (2D marching squares on each box face),
/// then report topology.
fn cap_and_analyse(
    pos_in: &[[f32; 3]],
    idx_in: &[u32],
    tag: &str,
    tree: &fidget_core::context::Tree,
    mc_res: usize,
) {
    use fidget_core::shape::Shape;
    use fidget_core::vm::VmFunction;

    // Dedup first — use larger eps to merge MC33 boundary vertices
    // that differ by floating-point interpolation error (up to ~3e-5).
    // Must be ≤ 0.5% of grid spacing: dx=2/res, 0.5%*dx = 0.01/res.
    let aabb_diag = {
        let mut mn = [f32::MAX; 3];
        let mut mx = [f32::MIN; 3];
        for p in pos_in {
            for i in 0..3 { mn[i] = mn[i].min(p[i]); mx[i] = mx[i].max(p[i]); }
        }
        let d = [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]];
        (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt()
    };
    let eps = aabb_diag * 1e-6;
    let max_safe_eps = 0.005 / mc_res as f32; // 0.25% of grid spacing
    let first_dedup_eps = (5.0e-5_f32).min(max_safe_eps).max(eps);
    let mut pos = pos_in.to_vec();
    let mut idx = idx_in.to_vec();
    engvis_core::mesh::dedup_vertices(&mut pos, &mut idx, first_dedup_eps);
    let nv = pos.len();
    eprintln!("[{tag}-capped] after dedup: {} verts, {} tris", nv, idx.len()/3);

    // Compile shape for face grid evaluation
    let shape = Shape::<VmFunction>::from(tree.clone());
    let mut float_eval = Shape::<VmFunction>::new_float_slice_eval();
    let tape = shape.float_slice_tape(Default::default());
    let mut eval_g = |x: f32, y: f32, z: f32| -> f32 {
        match float_eval.eval(&tape, &[x], &[y], &[z]) { Ok(r) => r[0], Err(_) => 1e9 }
    };

    // Quick boundary loop diagnostic (for logging only)
    let bnd_verts = {
        let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (fi, tri) in idx.chunks_exact(3).enumerate() {
            let (a, b, c) = (tri[0], tri[1], tri[2]);
            for &(i0, i1) in &[(a,b),(b,c),(c,a)] {
                let key = if i0 <= i1 { (i0, i1) } else { (i1, i0) };
                edge_faces.entry(key).or_default().push(fi);
            }
        }
        let tol = 0.02_f32;
        let on_face = |p: [f32; 3]| -> bool {
            p[0].abs() > 1.0-tol || p[1].abs() > 1.0-tol || p[2].abs() > 1.0-tol
        };
        let mut bnd_count = 0usize;
        let mut verts = Vec::new();
        for (&(a, b), faces) in &edge_faces {
            if faces.len() == 1 {
                bnd_count += 1;
                if on_face(pos[a as usize]) { verts.push(a); }
                if on_face(pos[b as usize]) { verts.push(b); }
            }
        }
        verts.sort(); verts.dedup();
        eprintln!("[{tag}-capped] {bnd_count} boundary edges, {} boundary verts before capping", verts.len());
        verts
    };

    // Build spatial hash for snapping face fill vertices to MC33 boundary
    let snap_eps = first_dedup_eps;
    let bnd_snap: HashMap<(i64,i64,i64), Vec<u32>> = {
        let inv = 1.0 / snap_eps;
        let mut grid: HashMap<(i64,i64,i64), Vec<u32>> = HashMap::new();
        for &vi in &bnd_verts {
            let p = pos[vi as usize];
            let key = ((p[0]*inv).round() as i64, (p[1]*inv).round() as i64, (p[2]*inv).round() as i64);
            grid.entry(key).or_default().push(vi);
        }
        grid
    };

    // Face grid fill with vertex snapping
    let (av, at) = face_grid_fill(&mut pos, &mut idx, &mut eval_g, mc_res, &bnd_snap, snap_eps);
    eprintln!("[{tag}-capped] face_grid_fill: +{av} verts, +{at} tris");

    // Second dedup: merge face grid vertices with MC33 boundary vertices
    engvis_core::mesh::dedup_vertices(&mut pos, &mut idx, eps);
    eprintln!("[{tag}-capped] after 2nd dedup: {} verts, {} tris", pos.len(), idx.len()/3);

    // Signed volume check
    let sv: f64 = idx.chunks_exact(3).map(|tri| {
        let p0 = pos[tri[0] as usize]; let p1 = pos[tri[1] as usize]; let p2 = pos[tri[2] as usize];
        p0[0] as f64 * (p1[1] as f64 * p2[2] as f64 - p1[2] as f64 * p2[1] as f64)
      + p0[1] as f64 * (p1[2] as f64 * p2[0] as f64 - p1[0] as f64 * p2[2] as f64)
      + p0[2] as f64 * (p1[0] as f64 * p2[1] as f64 - p1[1] as f64 * p2[0] as f64)
    }).sum();
    if sv < 0.0 { for tri in idx.chunks_exact_mut(3) { tri.swap(1, 2); } }
    eprintln!("[{tag}-capped] signed_vol={sv:.1}");

    // Full topology analysis
    let nv2 = pos.len();
    let nf2 = idx.len() / 3;
    let mut ec2: HashMap<(u32,u32), usize> = HashMap::new();
    for tri in idx.chunks_exact(3) {
        let (a,b,c) = (tri[0],tri[1],tri[2]);
        for &(i0,i1) in &[(a,b),(b,c),(c,a)] {
            let k = if i0<=i1{(i0,i1)}else{(i1,i0)};
            *ec2.entry(k).or_default() += 1;
        }
    }
    let ne2 = ec2.len();
    let bnd2 = ec2.values().filter(|&&c| c==1).count();
    let nm2 = ec2.values().filter(|&&c| c>2).count();

    // Print non-manifold edge positions for diagnosis
    if nm2 > 0 {
        eprintln!("  non-manifold edges (first {}):", nm2.min(10));
        let mut count = 0;
        for (&(a, b), &cnt) in &ec2 {
            if cnt > 2 && count < 10 {
                let pa = pos[a as usize];
                let pb = pos[b as usize];
                eprintln!("    edge ({},{}) count={}: ({:.4},{:.4},{:.4}) - ({:.4},{:.4},{:.4})",
                    a, b, cnt, pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]);
                count += 1;
            }
        }
    }

    // Print boundary edge positions for diagnosis
    if bnd2 > 0 {
        let inv_check = 1.0 / snap_eps;
        eprintln!("  boundary edges (first {}):", bnd2.min(20));
        let mut count = 0;
        for (&(a, b), &cnt) in &ec2 {
            if cnt == 1 && count < 20 {
                let pa = pos[a as usize];
                let pb = pos[b as usize];
                // Check if these vertices would be in bnd_snap
                let in_snap = |p: [f32;3]| -> bool {
                    let key = ((p[0]*inv_check).round() as i64, (p[1]*inv_check).round() as i64, (p[2]*inv_check).round() as i64);
                    for dz in -1..=1_i64 {
                        for dy in -1..=1_i64 {
                            for dx_n in -1..=1_i64 {
                                let nk = (key.0+dx_n, key.1+dy, key.2+dz);
                                if let Some(bucket) = bnd_snap.get(&nk) {
                                    for &vi in bucket {
                                        let op = pos[vi as usize];
                                        let d = ((p[0]-op[0]).powi(2)+(p[1]-op[1]).powi(2)+(p[2]-op[2]).powi(2)).sqrt();
                                        if d < snap_eps { return true; }
                                    }
                                }
                            }
                        }
                    }
                    false
                };
                let on_box = |p: [f32;3]| -> bool {
                    (p[0].abs()-1.0).abs() < 0.01 || (p[1].abs()-1.0).abs() < 0.01 || (p[2].abs()-1.0).abs() < 0.01
                };
                let sa = in_snap(pa);
                let sb = in_snap(pb);
                eprintln!("    edge ({},{}) : ({:.6},{:.6},{:.6}) - ({:.6},{:.6},{:.6}) snap_a={} snap_b={} box={}",
                    a, b, pa[0], pa[1], pa[2], pb[0], pb[1], pb[2], sa, sb, on_box(pa) && on_box(pb));
                count += 1;
            }
        }
    }

    let mut parent: Vec<usize> = (0..nf2).collect();
    let mut rank_uf: Vec<u32> = vec![0; nf2];
    fn find(p: &mut [usize], mut x: usize) -> usize { while p[x]!=x { p[x]=p[p[x]]; x=p[x]; } x }
    fn union(p: &mut [usize], r: &mut [u32], mut a: usize, mut b: usize) {
        a=find(p,a); b=find(p,b); if a==b{return;}
        match r[a].cmp(&r[b]) {
            std::cmp::Ordering::Less => {p[a]=b;}
            std::cmp::Ordering::Greater => {p[b]=a;}
            std::cmp::Ordering::Equal => {p[b]=a; r[a]+=1;}
        }
    }
    let mut efm: HashMap<(u32,u32),Vec<usize>> = HashMap::new();
    for (fi,tri) in idx.chunks_exact(3).enumerate() {
        let (a,b,c) = (tri[0],tri[1],tri[2]);
        for &(i0,i1) in &[(a,b),(b,c),(c,a)] {
            let k = if i0<=i1{(i0,i1)}else{(i1,i0)};
            efm.entry(k).or_default().push(fi);
        }
    }
    for faces in efm.values() {
        if faces.len()>=2 { for i in 1..faces.len() { union(&mut parent,&mut rank_uf,faces[0],faces[i]); } }
    }
    let mut roots: Vec<usize> = (0..nf2).map(|i| find(&mut parent,i)).collect();
    roots.sort(); roots.dedup();
    let comps = roots.len();
    let euler = nv2 as i64 - ne2 as i64 + nf2 as i64;
    let wt = bnd2 == 0 && comps == 1;
    eprintln!("  → V={nv2} E={ne2} F={nf2} χ={euler} components={comps} boundary_edges={bnd2} non_manifold={nm2} watertight={wt}");

    let mut cs: HashMap<usize,usize> = HashMap::new();
    for fi in 0..nf2 { let r = find(&mut parent,fi); *cs.entry(r).or_default() += 1; }
    let mut sizes: Vec<usize> = cs.values().copied().collect();
    sizes.sort_unstable_by(|a,b| b.cmp(a));
    eprintln!("  component sizes (top {}): {:?}", sizes.len().min(5), &sizes[..sizes.len().min(5)]);
}

/// Face grid fill: 2D marching squares on each box face to cap the mesh.
/// Evaluates g on a 2D grid matching MC33 resolution, triangulates g≤0 regions.
fn face_grid_fill(
    pos: &mut Vec<[f32; 3]>,
    idx: &mut Vec<u32>,
    eval_g: &mut dyn FnMut(f32, f32, f32) -> f32,
    res: usize,
    bnd_snap: &HashMap<(i64,i64,i64), Vec<u32>>,
    snap_eps: f32,
) -> (usize, usize) {
    let dx = 2.0 / res as f32;
    let n = res + 1; // grid points per axis
    let mut added_verts = 0usize;
    let mut added_tris = 0usize;
    let inv_snap = 1.0 / snap_eps;
    let mut snap_hits = 0usize;
    let mut snap_miss = 0usize;

    // Snap a position to an existing boundary vertex if within snap_eps,
    // otherwise push a new vertex. Returns the vertex index.
    let mut push_or_snap = |pos: &mut Vec<[f32; 3]>, p: [f32; 3]| -> u32 {
        let key = ((p[0]*inv_snap).round() as i64, (p[1]*inv_snap).round() as i64, (p[2]*inv_snap).round() as i64);
        for dz in -1..=1_i64 {
            for dy in -1..=1_i64 {
                for dx_n in -1..=1_i64 {
                    let nk = (key.0+dx_n, key.1+dy, key.2+dz);
                    if let Some(bucket) = bnd_snap.get(&nk) {
                        for &vi in bucket {
                            let op = pos[vi as usize];
                            let d = ((p[0]-op[0]).powi(2) + (p[1]-op[1]).powi(2) + (p[2]-op[2]).powi(2)).sqrt();
                            if d < snap_eps { snap_hits += 1; return vi; }
                        }
                    }
                }
            }
        }
        snap_miss += 1;
        let vi = pos.len() as u32;
        pos.push(p);
        vi
    };

    // Face definitions: (axis, sign, u_axis, v_axis)
    // sign: +1 for positive face (coord=+1), -1 for negative (coord=-1)
    let faces: [(usize, f32, usize, usize); 6] = [
        (0,  1.0, 1, 2), // +X: u=Y, v=Z
        (0, -1.0, 1, 2), // -X: u=Y, v=Z
        (1,  1.0, 0, 2), // +Y: u=X, v=Z
        (1, -1.0, 0, 2), // -Y: u=X, v=Z
        (2,  1.0, 0, 1), // +Z: u=X, v=Y
        (2, -1.0, 0, 1), // -Z: u=X, v=Y
    ];

    for &(face_axis, face_val, u_axis, v_axis) in &faces {
        // Evaluate g on 2D grid
        let mut vals = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut p = [0.0_f32; 3];
                p[face_axis] = face_val;
                p[u_axis] = -1.0 + i as f32 * dx;
                p[v_axis] = -1.0 + j as f32 * dx;
                vals[i][j] = eval_g(p[0], p[1], p[2]);
            }
        }

        let sign = if face_val > 0.0 { 1.0_f32 } else { -1.0 };

        // Helper: 3D position from (u, v) on this face
        let pos3d = |u: f32, v: f32| -> [f32; 3] {
            let mut p = [0.0_f32; 3];
            p[face_axis] = face_val;
            p[u_axis] = u;
            p[v_axis] = v;
            p
        };

        // Helper: edge zero-crossing position (u coordinate)
        let edge_u = |v_a: f32, v_b: f32, va: f32, vb: f32| -> f32 {
            let t = va / (va - vb);
            v_a + t * (v_b - v_a)
        };

        for i in 0..res {
            for j in 0..res {
                let u0 = -1.0 + i as f32 * dx;
                let u1 = u0 + dx;
                let v0 = -1.0 + j as f32 * dx;
                let v1 = v0 + dx;

                let g00 = vals[i][j];
                let g10 = vals[i+1][j];
                let g11 = vals[i+1][j+1];
                let g01 = vals[i][j+1];

                let c0 = g00 <= 0.0;
                let c1 = g10 <= 0.0;
                let c2 = g11 <= 0.0;
                let c3 = g01 <= 0.0;

                let mask = (c0 as u8)
                    | ((c1 as u8) << 1)
                    | ((c2 as u8) << 2)
                    | ((c3 as u8) << 3);

                if mask == 0 { continue; }

                // Corner 3D positions
                let p00 = pos3d(u0, v0);
                let p10 = pos3d(u1, v0);
                let p11 = pos3d(u1, v1);
                let p01 = pos3d(u0, v1);

                // Edge zero-crossing 3D positions
                // edge 0 = bottom (c0-c1), edge 1 = right (c1-c2),
                // edge 2 = top (c2-c3), edge 3 = left (c3-c0)
                let e01 = pos3d(edge_u(u0, u1, g00, g10), v0);
                let e12 = pos3d(u1, edge_u(v0, v1, g10, g11));
                let e23 = pos3d(edge_u(u0, u1, g01, g11), v1);
                let e30 = pos3d(u0, edge_u(v0, v1, g00, g01));

                // Build CCW-ordered polygon of the solid region
                let poly: Vec<[f32; 3]> = match mask {
                    // Single corner solid
                    1  => vec![p00, e01, e30],
                    2  => vec![p10, e12, e01],
                    4  => vec![p11, e23, e12],
                    8  => vec![p01, e30, e23],
                    // Two adjacent corners solid (trapezoid)
                    3  => vec![p00, p10, e12, e30],
                    6  => vec![p10, p11, e23, e01],
                    9  => vec![p00, e01, e23, p01],
                    12 => vec![p01, p11, e12, e30],
                    // Three corners solid (pentagon) — one corner empty
                    7  => vec![p00, p10, p11, e23, e30],  // c3 empty
                    11 => vec![p00, p10, e12, e23, p01],  // c2 empty
                    13 => vec![p00, e01, e12, p11, p01],  // c1 empty
                    14 => vec![e01, p10, p11, p01, e30],  // c0 empty
                    // All four corners solid
                    15 => vec![p00, p10, p11, p01],
                    // Ambiguous: opposite corners solid — handle as two separate polygons
                    5 => {
                        let cu = (u0 + u1) * 0.5;
                        let cv = (v0 + v1) * 0.5;
                        let mut cp = [0.0_f32; 3];
                        cp[face_axis] = face_val;
                        cp[u_axis] = cu;
                        cp[v_axis] = cv;
                        let gc = eval_g(cp[0], cp[1], cp[2]);
                        let (tri_a, tri_b): (Vec<[f32; 3]>, Vec<[f32; 3]>) = if gc <= 0.0 {
                            // Center solid: c0 and c2 connected through center
                            (vec![p00, e01, e30], vec![p11, e23, e12])
                        } else {
                            // Center empty: c1 and c3 as separate triangles
                            (vec![p10, e12, e01], vec![p01, e30, e23])
                        };
                        // Triangulate both polygons
                        for tri in [&tri_a, &tri_b] {
                            let tn = tri.len();
                            if tn < 3 { continue; }
                            let sa: f32 = (0..tn).map(|k| {
                                let l = (k+1)%tn;
                                tri[k][u_axis]*tri[l][v_axis] - tri[l][u_axis]*tri[k][v_axis]
                            }).sum::<f32>() * 0.5;
                            let tri = if sign * sa < 0.0 {
                                let mut r = tri.clone(); r.reverse(); r
                            } else { tri.clone() };
                            let cx: f32 = tri.iter().map(|p| p[0]).sum::<f32>() / tn as f32;
                            let cy: f32 = tri.iter().map(|p| p[1]).sum::<f32>() / tn as f32;
                            let cz: f32 = tri.iter().map(|p| p[2]).sum::<f32>() / tn as f32;
                            let mut vi: Vec<u32> = Vec::with_capacity(tn);
                            for &p in &tri { vi.push(push_or_snap(pos, p)); }
                            added_verts += tn;
                            let centroid = push_or_snap(pos, [cx, cy, cz]);
                            added_verts += 1;
                            for k in 0..tn {
                                let l = (k+1)%tn;
                                idx.push(centroid); idx.push(vi[k]); idx.push(vi[l]);
                                added_tris += 1;
                            }
                        }
                        continue; // skip the normal polygon processing below
                    }
                    10 => {
                        let cu = (u0 + u1) * 0.5;
                        let cv = (v0 + v1) * 0.5;
                        let mut cp = [0.0_f32; 3];
                        cp[face_axis] = face_val;
                        cp[u_axis] = cu;
                        cp[v_axis] = cv;
                        let gc = eval_g(cp[0], cp[1], cp[2]);
                        let (tri_a, tri_b): (Vec<[f32; 3]>, Vec<[f32; 3]>) = if gc <= 0.0 {
                            (vec![p10, e12, e01], vec![p01, e30, e23])
                        } else {
                            (vec![p00, e01, e30], vec![p11, e23, e12])
                        };
                        for tri in [&tri_a, &tri_b] {
                            let tn = tri.len();
                            if tn < 3 { continue; }
                            let sa: f32 = (0..tn).map(|k| {
                                let l = (k+1)%tn;
                                tri[k][u_axis]*tri[l][v_axis] - tri[l][u_axis]*tri[k][v_axis]
                            }).sum::<f32>() * 0.5;
                            let tri = if sign * sa < 0.0 {
                                let mut r = tri.clone(); r.reverse(); r
                            } else { tri.clone() };
                            let cx: f32 = tri.iter().map(|p| p[0]).sum::<f32>() / tn as f32;
                            let cy: f32 = tri.iter().map(|p| p[1]).sum::<f32>() / tn as f32;
                            let cz: f32 = tri.iter().map(|p| p[2]).sum::<f32>() / tn as f32;
                            let mut vi: Vec<u32> = Vec::with_capacity(tn);
                            for &p in &tri { vi.push(push_or_snap(pos, p)); }
                            added_verts += tn;
                            let centroid = push_or_snap(pos, [cx, cy, cz]);
                            added_verts += 1;
                            for k in 0..tn {
                                let l = (k+1)%tn;
                                idx.push(centroid); idx.push(vi[k]); idx.push(vi[l]);
                                added_tris += 1;
                            }
                        }
                        continue;
                    }
                    _ => continue,
                };

                let pn = poly.len();
                if pn < 3 { continue; }

                // Check winding: project to 2D, compute signed area
                let signed_area: f32 = (0..pn).map(|k| {
                    let l = (k+1) % pn;
                    poly[k][u_axis] * poly[l][v_axis]
                      - poly[l][u_axis] * poly[k][v_axis]
                }).sum::<f32>() * 0.5;

                // For positive faces (outward normal = +axis), want CCW (signed_area > 0)
                // For negative faces (outward normal = -axis), want CW (signed_area < 0)
                let poly = if sign * signed_area < 0.0 {
                    let mut r = poly; r.reverse(); r
                } else {
                    poly
                };

                // Push polygon vertices (snapping to MC33 boundary) + centroid, then fan
                let cx: f32 = poly.iter().map(|p| p[0]).sum::<f32>() / pn as f32;
                let cy: f32 = poly.iter().map(|p| p[1]).sum::<f32>() / pn as f32;
                let cz: f32 = poly.iter().map(|p| p[2]).sum::<f32>() / pn as f32;

                let mut vert_indices: Vec<u32> = Vec::with_capacity(pn);
                for &p in &poly {
                    vert_indices.push(push_or_snap(pos, p));
                    added_verts += 1;
                }
                let centroid = push_or_snap(pos, [cx, cy, cz]);
                added_verts += 1;

                for k in 0..pn {
                    let l = (k + 1) % pn;
                    idx.push(centroid);
                    idx.push(vert_indices[k]);
                    idx.push(vert_indices[l]);
                    added_tris += 1;
                }
            }
        }
    }
    eprintln!("  face_grid_fill snap: hits={snap_hits} miss={snap_miss}");
    (added_verts, added_tris)
}

/// Diagnostic: analyze how loops span faces and count box-edge solid segments.
fn diag_loop_faces(
    pos_in: &[[f32; 3]],
    idx_in: &[u32],
    tag: &str,
    tree: &fidget_core::context::Tree,
    res: usize,
) {
    use fidget_core::shape::Shape;
    use fidget_core::vm::VmFunction;

    // Dedup
    let aabb_diag = {
        let mut mn = [f32::MAX; 3];
        let mut mx = [f32::MIN; 3];
        for p in pos_in {
            for i in 0..3 { mn[i] = mn[i].min(p[i]); mx[i] = mx[i].max(p[i]); }
        }
        let d = [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]];
        (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt()
    };
    let eps = aabb_diag * 1e-6;
    let mut pos = pos_in.to_vec();
    let mut idx = idx_in.to_vec();
    engvis_core::mesh::dedup_vertices(&mut pos, &mut idx, eps);
    let _nf = idx.len() / 3;

    // Build edge → face map
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, tri) in idx.chunks_exact(3).enumerate() {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        for &(i0, i1) in &[(a,b),(b,c),(c,a)] {
            let key = if i0 <= i1 { (i0, i1) } else { (i1, i0) };
            edge_faces.entry(key).or_default().push(fi);
        }
    }

    let tol = 0.02_f32;
    let on_surface = |p: [f32; 3]| -> bool {
        p[0].abs() > 1.0 - tol || p[1].abs() > 1.0 - tol || p[2].abs() > 1.0 - tol
    };

    let mut bnd_edges: Vec<(u32, u32)> = Vec::new();
    for (&(a, b), faces) in &edge_faces {
        if faces.len() != 1 { continue; }
        let pa = pos[a as usize]; let pb = pos[b as usize];
        if on_surface(pa) && on_surface(pb) {
            bnd_edges.push((a, b));
        }
    }

    // Build adjacency and chain loops (same as cap_faces_from_boundary)
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &bnd_edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    let bad = adj.values().filter(|ns| ns.len() != 2).count();
    if bad > 0 {
        eprintln!("[{tag}] SKIP: {bad} vertices with degree ≠ 2");
        return;
    }

    let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut loops: Vec<Vec<u32>> = Vec::new();
    for &start in adj.keys() {
        if visited.contains(&start) { continue; }
        let mut lp = vec![start];
        visited.insert(start);
        let mut cur = start;
        loop {
            let nbrs = match adj.get(&cur) { Some(n) => n, None => break };
            let mut next = None;
            for &nb in nbrs {
                if !visited.contains(&nb) { next = Some(nb); break; }
            }
            match next {
                Some(nb) => { visited.insert(nb); lp.push(nb); cur = nb; }
                None => break,
            }
        }
        if lp.len() >= 3 { loops.push(lp); }
    }

    // Classify each vertex to a face
    let face_of = |p: [f32; 3]| -> i8 {
        if p[0] >  1.0 - tol { return 0; }
        if p[0] < -1.0 + tol { return 1; }
        if p[1] >  1.0 - tol { return 2; }
        if p[1] < -1.0 + tol { return 3; }
        if p[2] >  1.0 - tol { return 4; }
        if p[2] < -1.0 + tol { return 5; }
        -1
    };
    let face_name = |f: i8| -> &'static str {
        match f { 0=>"x+1", 1=>"x-1", 2=>"y+1", 3=>"y-1", 4=>"z+1", 5=>"z-1", _=>"?" }
    };

    let mut single_face = 0usize;
    let mut multi_face = 0usize;
    for lp in &loops {
        let mut faces: Vec<i8> = lp.iter().map(|&i| face_of(pos[i as usize])).collect();
        faces.retain(|&f| f >= 0);
        faces.sort();
        faces.dedup();
        if faces.len() <= 1 {
            single_face += 1;
        } else {
            multi_face += 1;
            let face_names: Vec<&str> = faces.iter().map(|&f| face_name(f)).collect();
            eprintln!("  [{tag}] multi-face loop: {} verts, faces={:?}", lp.len(), face_names);
        }
    }
    eprintln!("[{tag}] loops: {single_face} single-face, {multi_face} multi-face");

    // Count solid grid points on box edges
    let shape = Shape::<VmFunction>::from(tree.clone());
    let mut float_eval = Shape::<VmFunction>::new_float_slice_eval();
    let tape = shape.float_slice_tape(Default::default());
    let mut eval_g = |x: f32, y: f32, z: f32| -> f32 {
        match float_eval.eval(&tape, &[x], &[y], &[z]) {
            Ok(r) => r[0], Err(_) => 1e9,
        }
    };

    let dx = 2.0 / res as f32;
    let mut total_box_edge_solid = 0usize;
    let mut total_box_edge_pts = 0usize;

    // Check each of the 12 box edges
    let box_edges: [([f32;3], [f32;3], usize); 12] = [
        // along x: y,z ∈ {-1,1}
        ([-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], 0),
        ([-1.0, -1.0,  1.0], [1.0, -1.0,  1.0], 0),
        ([-1.0,  1.0, -1.0], [1.0,  1.0, -1.0], 0),
        ([-1.0,  1.0,  1.0], [1.0,  1.0,  1.0], 0),
        // along y: x,z ∈ {-1,1}
        ([-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], 1),
        ([-1.0, -1.0,  1.0], [-1.0, 1.0,  1.0], 1),
        ([ 1.0, -1.0, -1.0], [ 1.0, 1.0, -1.0], 1),
        ([ 1.0, -1.0,  1.0], [ 1.0, 1.0,  1.0], 1),
        // along z: x,y ∈ {-1,1}
        ([-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], 2),
        ([-1.0,  1.0, -1.0], [-1.0,  1.0, 1.0], 2),
        ([ 1.0, -1.0, -1.0], [ 1.0, -1.0, 1.0], 2),
        ([ 1.0,  1.0, -1.0], [ 1.0,  1.0, 1.0], 2),
    ];

    let mut solid_segments = 0usize;
    for &(start, _end, axis) in &box_edges {
        let mut prev_solid = false;
        for i in 0..=res {
            let t = -1.0 + i as f32 * dx;
            let mut p = start;
            p[axis] = t;
            let g = eval_g(p[0], p[1], p[2]);
            let solid = g <= 0.0;
            total_box_edge_pts += 1;
            if solid { total_box_edge_solid += 1; }
            if solid && prev_solid {
                solid_segments += 1;
            }
            prev_solid = solid;
        }
    }
    eprintln!(
        "[{tag}] box-edge: {total_box_edge_solid}/{total_box_edge_pts} solid grid pts, {solid_segments} solid segments"
    );
}

fn main() {
    let half_t = 0.125_f32;
    let c_val = 0.5_f32;
    for &res in &[48, 64, 96] {
        eprintln!("=== Shell capping res={res} ===");
        let tree = build_gyroid_shell_tree(half_t);
        let (pos, idx) = run_mc33(&tree, res);
        cap_and_analyse(&pos, &idx, &format!("shell-res{res}"), &tree, res);
    }
    for &res in &[48, 64, 96] {
        eprintln!("=== Skeletal capping res={res} ===");
        let tree = build_gyroid_skeletal_tree(c_val);
        let (pos, idx) = run_mc33(&tree, res);
        cap_and_analyse(&pos, &idx, &format!("skel-res{res}"), &tree, res);
    }
}
