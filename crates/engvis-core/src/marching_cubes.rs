//! Marching Cubes 33 isosurface extraction.
//!
//! Implements the MC33 algorithm with asymptotic decider for ambiguous
//! face configurations, producing a watertight triangle mesh from an
//! implicit function f(x, y, z) = 0.
//!
//! Boundary vertices lie exactly on grid edges via linear interpolation,
//! so open-surface boundaries are naturally smooth (no DC-style QEF
//! jaggedness).

use crate::bourke_table::TRI_TABLE;

fn lerp(a: [f32; 3], b: [f32; 3], va: f32, vb: f32) -> [f32; 3] {
    // Guard against a zero denominator: when the two endpoint field
    // values are (nearly) equal, va/(va-vb) blows up to ±inf/NaN and
    // poisons the whole mesh.  This happens on creases of CSG fields
    // such as the TPMS shell |f|−t/2, where adjacent samples can be
    // numerically identical.  Fall back to the edge midpoint.
    let denom = va - vb;
    let t = if denom.abs() < 1e-12 { 0.5 } else { va / denom };
    // Clamp t to [0,1].  MC33 alternative triangulation tables can
    // reference an edge whose endpoints are same-sign (the edge is not
    // in EDGE_TABLE for this case); the raw t then falls outside [0,1]
    // and the vertex escapes the cell, producing triangles that span
    // outside the sampling box.  Clamping keeps the vertex on the edge
    // segment, which is the closest valid point.
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

pub(crate) const EDGE_VERTS: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

pub static EDGE_TABLE: [u16; 256] = [
    0x000,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,
    0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
    0x190,0x099,0x393,0x29a,0x596,0x49f,0x795,0x69c,
    0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
    0x230,0x339,0x033,0x13a,0x636,0x73f,0x435,0x53c,
    0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
    0x3a0,0x2a9,0x1a3,0x0aa,0x7a6,0x6af,0x5a5,0x4ac,
    0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
    0x460,0x569,0x663,0x76a,0x066,0x16f,0x265,0x36c,
    0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
    0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0x0ff,0x3f5,0x2fc,
    0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
    0x650,0x759,0x453,0x55a,0x256,0x35f,0x055,0x15c,
    0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
    0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0x0cc,
    0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
    0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,
    0x0cc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
    0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,
    0x15c,0x055,0x35f,0x256,0x55a,0x453,0x759,0x650,
    0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,
    0x2fc,0x3f5,0x0ff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
    0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,
    0x36c,0x265,0x16f,0x066,0x76a,0x663,0x569,0x460,
    0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,
    0x4ac,0x5a5,0x6af,0x7a6,0x0aa,0x1a3,0x2a9,0x3a0,
    0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,
    0x53c,0x435,0x73f,0x636,0x13a,0x033,0x339,0x230,
    0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,
    0x69c,0x795,0x49f,0x596,0x29a,0x393,0x099,0x190,
    0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,
    0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x000,
];

// ---------------------------------------------------------------------------
// MC33: asymptotic decider for ambiguous face configurations.
// Activates only on cells with alternating-sign faces (saddle ambiguity).
// ---------------------------------------------------------------------------

const MC33_CASE3_ALT: [i8; 7] = [3, 7, 8, 2, 3, 11, -1];
const MC33_CASE6_ALT: [i8; 16] = [3,7,8,0,9,4,2,3,11,-1,-1,-1,-1,-1,-1,-1];
const MC33_CASE6_ALT2: [i8; 16] = [4,5,9,0,2,10,0,10,9,-1,-1,-1,-1,-1,-1,-1];
const MC33_CASE6_ALT3: [i8; 16] = [2,10,5,2,5,4,2,4,7,-1,-1,-1,-1,-1,-1,-1];
const MC33_CASE7_B: [i8; 16] = [3,7,8,0,1,9,0,2,10,0,10,9,-1,-1,-1,-1];
const MC33_CASE7_C: [i8; 16] = [3,7,8,0,1,9,2,3,11,4,5,9,-1,-1,-1,-1];
const MC33_CASE7_D: [i8; 16] = [0,1,9,2,3,11,4,5,9,4,7,11,-1,-1,-1,-1];
const MC33_CASE7_E: [i8; 16] = [2,3,11,4,7,8,4,5,9,8,9,11,-1,-1,-1,-1];
const MC33_CASE7_F: [i8; 16] = [4,5,9,4,7,8,0,2,10,0,10,9,-1,-1,-1,-1];
const MC33_CASE7_G: [i8; 16] = [0,2,10,4,7,8,4,5,9,8,9,10,-1,-1,-1,-1];
const MC33_CASE7_H: [i8; 16] = [2,3,11,4,5,9,8,9,11,10,5,11,-1,-1,-1,-1];
const MC33_CASE12_ALT: [i8; 16] = [1,2,10,1,10,11,1,11,8,-1,-1,-1,-1,-1,-1,-1];

const FACE_VERTICES: [[usize; 4]; 6] = [
    [0,1,2,3],  [4,5,6,7],  [0,1,5,4],
    [2,3,7,6],  [0,3,7,4],  [1,2,6,5],
];

fn face_alternating(vals: &[f32; 8], fi: usize) -> bool {
    let fv = FACE_VERTICES[fi];
    let s = [vals[fv[0]]>=0.0, vals[fv[1]]>=0.0, vals[fv[2]]>=0.0, vals[fv[3]]>=0.0];
    s[0]==s[2] && s[1]==s[3] && s[0]!=s[1]
}

fn compute_mc33_subcase<F: FnMut(f32,f32,f32)->f32>(mut f: F, v: &[[f32;3];8], vals: &[f32;8]) -> u8 {
    let mut bits = 0u8;
    let neg = vals.iter().filter(|x|**x<0.0).count();
    let max_bits = match neg {2|6=>1, 3|5=>2, 4=>3, _=>0};
    let mut bi = 0u8;
    for fi in 0..6 {
        if bi>=max_bits {break;}
        if face_alternating(vals, fi) {
            let fv = FACE_VERTICES[fi];
            let cx = (v[fv[0]][0]+v[fv[1]][0]+v[fv[2]][0]+v[fv[3]][0])/4.0;
            let cy = (v[fv[0]][1]+v[fv[1]][1]+v[fv[2]][1]+v[fv[3]][1])/4.0;
            let cz = (v[fv[0]][2]+v[fv[1]][2]+v[fv[2]][2]+v[fv[3]][2])/4.0;
            if (vals[fv[0]]<0.0) != (f(cx,cy,cz)<0.0) {bits |= 1<<bi;}
            bi += 1;
        }
    }
    bits
}

fn count_amb_faces_from_case(case_idx: u8) -> u8 {
    let b = |i:usize| (case_idx>>i)&1;
    let mut c = 0u8;
    if b(0)==b(2)&&b(1)==b(3)&&b(0)!=b(1) {c+=1;}
    if b(4)==b(6)&&b(5)==b(7)&&b(4)!=b(5) {c+=1;}
    if b(0)==b(5)&&b(1)==b(4)&&b(0)!=b(1) {c+=1;}
    if b(2)==b(7)&&b(3)==b(6)&&b(2)!=b(3) {c+=1;}
    if b(0)==b(7)&&b(3)==b(4)&&b(0)!=b(3) {c+=1;}
    if b(1)==b(6)&&b(2)==b(5)&&b(1)!=b(2) {c+=1;}
    c
}

fn mc33_tris(case_idx: u8, subcase: u8, amb_faces: u8) -> &'static [i8] {
    if amb_faces==0 {return &TRI_TABLE[case_idx as usize];}
    let s=subcase;
    match case_idx.count_ones() {
        2 if s&1!=0 => &MC33_CASE3_ALT,
        3 if amb_faces>=2 => match s&3 {1=>&MC33_CASE6_ALT,2=>&MC33_CASE6_ALT2,3=>&MC33_CASE6_ALT3,_=>&TRI_TABLE[case_idx as usize]},
        4 => match s&7 {1=>&MC33_CASE7_B,2=>&MC33_CASE7_C,3=>&MC33_CASE7_D,4=>&MC33_CASE7_E,5=>&MC33_CASE7_F,6=>&MC33_CASE7_G,7=>&MC33_CASE7_H,_=>&TRI_TABLE[case_idx as usize]},
        5 if amb_faces>=2 => match s&3 {1=>&MC33_CASE6_ALT,2=>&MC33_CASE6_ALT2,3=>&MC33_CASE6_ALT3,_=>&TRI_TABLE[case_idx as usize]},
        6 if s&1!=0 => &MC33_CASE12_ALT,
        _ => &TRI_TABLE[case_idx as usize],
    }
}

/// Extract an isosurface mesh from a scalar field sampler.
///
/// `f` returns f(x, y, z); the isosurface is f = 0.
/// The domain is [x0, x1] × [y0, y1] × [z0, z1] sampled at
/// (nx+1) × (ny+1) × (nz+1) grid points.
///
/// Process a single grid cell: returns triangles as position triples.
/// Shared between the sequential and parallel extractors.
fn process_cell<F: FnMut(f32, f32, f32) -> f32>(
    f: &mut F,
    v: &[[f32; 3]; 8],
    vals: &[f32; 8],
) -> Vec<([f32; 3], [f32; 3], [f32; 3])> {
    let pos0 = vals[0] >= 0.0;
    if vals.iter().all(|&v| (v >= 0.0) == pos0) {
        return Vec::new();
    }
    let mut case_idx = 0u8;
    for i in 0..8 {
        if vals[i] < 0.0 {
            case_idx |= 1 << i;
        }
    }
    let edges = EDGE_TABLE[case_idx as usize];
    if edges == 0 {
        return Vec::new();
    }
    let mut edge_pts: [[f32; 3]; 12] = [[0.0; 3]; 12];
    for e in 0..12 {
        let (i0, i1) = EDGE_VERTS[e];
        edge_pts[e] = lerp(v[i0], v[i1], vals[i0], vals[i1]);
    }
    let amb_faces = count_amb_faces_from_case(case_idx);
    let tri_row: &[i8] = if amb_faces == 0 {
        &TRI_TABLE[case_idx as usize]
    } else {
        let subcase = compute_mc33_subcase(f, v, vals);
        mc33_tris(case_idx, subcase, amb_faces)
    };
    let mut out = Vec::new();
    for chunk in tri_row.chunks(3) {
        if chunk.len() < 3 || chunk[0] < 0 || chunk[1] < 0 || chunk[2] < 0 {
            break;
        }
        out.push((
            edge_pts[chunk[0] as usize],
            edge_pts[chunk[1] as usize],
            edge_pts[chunk[2] as usize],
        ));
    }
    out
}
/// Returns flat positions and triangle indices (u32).
pub fn extract<F: FnMut(f32, f32, f32) -> f32>(
    mut f: F,
    x_range: (f32, f32, usize),
    y_range: (f32, f32, usize),
    z_range: (f32, f32, usize),
) -> (Vec<[f32; 3]>, Vec<u32>) {
    let (x0, x1, nx) = x_range;
    let (y0, y1, ny) = y_range;
    let (z0, z1, nz) = z_range;
    let dx = (x1 - x0) / nx as f32;
    let dy = (y1 - y0) / ny as f32;
    let dz = (z1 - z0) / nz as f32;

    // Sample the scalar field on a flat grid (cache-friendly, single
    // allocation).  Indexing: grid[(ix*(ny+1) + iy)*(nz+1) + iz].
    let sx = ny + 1;
    let sy = nz + 1;
    let mut grid = vec![0.0_f32; (nx + 1) * sx * sy];
    for ix in 0..=nx {
        let x = x0 + ix as f32 * dx;
        for iy in 0..=ny {
            let y = y0 + iy as f32 * dy;
            for iz in 0..=nz {
                let z = z0 + iz as f32 * dz;
                grid[(ix * sx + iy) * sy + iz] = f(x, y, z);
            }
        }
    }

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let v: [[f32; 3]; 8] = [
                    [x0 + (ix    ) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz    ) as f32 * dz],
                    [x0 + (ix + 1) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz    ) as f32 * dz],
                    [x0 + (ix + 1) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz    ) as f32 * dz],
                    [x0 + (ix    ) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz    ) as f32 * dz],
                    [x0 + (ix    ) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                    [x0 + (ix + 1) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                    [x0 + (ix + 1) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                    [x0 + (ix    ) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                ];
                let vals = [
                    grid[((ix    ) * sx + (iy    )) * sy + (iz    )],
                    grid[((ix + 1) * sx + (iy    )) * sy + (iz    )],
                    grid[((ix + 1) * sx + (iy + 1)) * sy + (iz    )],
                    grid[((ix    ) * sx + (iy + 1)) * sy + (iz    )],
                    grid[((ix    ) * sx + (iy    )) * sy + (iz + 1)],
                    grid[((ix + 1) * sx + (iy    )) * sy + (iz + 1)],
                    grid[((ix + 1) * sx + (iy + 1)) * sy + (iz + 1)],
                    grid[((ix    ) * sx + (iy + 1)) * sy + (iz + 1)],
                ];

                for (p0, p1, p2) in process_cell(&mut f, &v, &vals) {
                    let base = positions.len() as u32;
                    positions.push(p0);
                    positions.push(p1);
                    positions.push(p2);
                    indices.push(base);
                    indices.push(base + 1);
                    indices.push(base + 2);
                }
            }
        }
    }

    // Fix face winding: some Bourke cases produce CW winding instead of CCW.
    // Compare each triangle's face normal against the gradient of f, flip if
    // they disagree. This prevents holes from back-face-culled triangles.
    let h = 1e-4_f32;
    let mut fixed = 0usize;
    for tri in indices.chunks_exact_mut(3) {
        let a = positions[tri[0] as usize];
        let b = positions[tri[1] as usize];
        let c = positions[tri[2] as usize];
        let e1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let e2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let fnx = e1[1]*e2[2] - e1[2]*e2[1];
        let fny = e1[2]*e2[0] - e1[0]*e2[2];
        let fnz = e1[0]*e2[1] - e1[1]*e2[0];
        let flen = (fnx*fnx + fny*fny + fnz*fnz).sqrt();
        if flen < 1e-30 { continue; }
        let cx = (a[0]+b[0]+c[0])/3.0;
        let cy = (a[1]+b[1]+c[1])/3.0;
        let cz = (a[2]+b[2]+c[2])/3.0;
        let gx = f(cx+h,cy,cz) - f(cx-h,cy,cz);
        let gy = f(cx,cy+h,cz) - f(cx,cy-h,cz);
        let gz = f(cx,cy,cz+h) - f(cx,cy,cz-h);
        let glen = (gx*gx+gy*gy+gz*gz).sqrt();
        if glen < 1e-30 { continue; }
        let dot = fnx*gx + fny*gy + fnz*gz;
        if dot < 0.0 {
            tri.swap(1, 2);
            fixed += 1;
        }
    }
    if fixed > 0 {
        eprintln!("  [MC33] winding fix: {} tris flipped", fixed);
    }

    (positions, indices)
}

/// Parallel MC33 extraction with a **pre-computed grid**.
///
/// This variant skips the internal grid sampling (the caller must provide
/// the grid computed via batch evaluation) but still performs per-cell
/// triangulation and the gradient-based winding fix.
///
/// The grid must be laid out as `grid[(ix * (ny+1) + iy) * (nz+1) + iz]`,
/// matching the indexing used by [`extract_par`].
///
/// The closure `f` is used only for:
/// 1. MC33 ambiguous-face resolution (`compute_mc33_subcase`) — rare.
/// 2. Winding fix gradient (6 calls per triangle, parallelised).
pub fn extract_par_with_grid<F: Fn(f32, f32, f32) -> f32 + Sync + Send>(
    f: F,
    grid: &[f32],
    x_range: (f32, f32, usize),
    y_range: (f32, f32, usize),
    z_range: (f32, f32, usize),
) -> (Vec<[f32; 3]>, Vec<u32>) {
    use rayon::prelude::*;
    let (x0, _x1, nx) = x_range;
    let (y0, _y1, ny) = y_range;
    let (z0, _z1, nz) = z_range;
    let dx = (x_range.1 - x0) / nx as f32;
    let dy = (y_range.1 - y0) / ny as f32;
    let dz = (z_range.1 - z0) / nz as f32;

    let sx = ny + 1;
    let sy = nz + 1;

    // Parallel per-cell triangulation.
    let tris: Vec<[f32; 3]> = (0..nx)
        .into_par_iter()
        .flat_map_iter(|ix| {
            let mut slice = Vec::new();
            for iy in 0..ny {
                for iz in 0..nz {
                    let v: [[f32; 3]; 8] = [
                        [x0 + (ix    ) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz    ) as f32 * dz],
                        [x0 + (ix + 1) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz    ) as f32 * dz],
                        [x0 + (ix + 1) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz    ) as f32 * dz],
                        [x0 + (ix    ) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz    ) as f32 * dz],
                        [x0 + (ix    ) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                        [x0 + (ix + 1) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                        [x0 + (ix + 1) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                        [x0 + (ix    ) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                    ];
                    let vals = [
                        grid[((ix    ) * sx + (iy    )) * sy + (iz    )],
                        grid[((ix + 1) * sx + (iy    )) * sy + (iz    )],
                        grid[((ix + 1) * sx + (iy + 1)) * sy + (iz    )],
                        grid[((ix    ) * sx + (iy + 1)) * sy + (iz    )],
                        grid[((ix    ) * sx + (iy    )) * sy + (iz + 1)],
                        grid[((ix + 1) * sx + (iy    )) * sy + (iz + 1)],
                        grid[((ix + 1) * sx + (iy + 1)) * sy + (iz + 1)],
                        grid[((ix    ) * sx + (iy + 1)) * sy + (iz + 1)],
                    ];
                    process_cell(&mut |x, y, z| f(x, y, z), &v, &vals)
                        .into_iter()
                        .for_each(|(a, b, c)| {
                            slice.push(a);
                            slice.push(b);
                            slice.push(c);
                        });
                }
            }
            slice
        })
        .collect();

    let n_tri = tris.len() / 3;
    let positions = tris;
    let indices: Vec<u32> = (0..n_tri as u32)
        .flat_map(|t| [t * 3, t * 3 + 1, t * 3 + 2])
        .collect();

    // NOTE: no winding fix here.  The caller runs Mesh::from_triangles,
    // whose fix_winding does BFS local-consistency propagation plus a
    // per-component signed-volume orientation fix.  That is robust even
    // where the field gradient vanishes (e.g. CSG creases f²−t² at f=0),
    // and avoids 6 expensive per-triangle field evaluations here.

    (positions, indices)
}

/// Parallel variant of [`extract`] for field samplers that are `Fn + Sync`.
/// Grid sampling and per-cell triangulation are parallelised with rayon;
/// the winding fix remains sequential (it is cheap relative to meshing).
pub fn extract_par<F: Fn(f32, f32, f32) -> f32 + Sync + Send>(
    f: F,
    x_range: (f32, f32, usize),
    y_range: (f32, f32, usize),
    z_range: (f32, f32, usize),
) -> (Vec<[f32; 3]>, Vec<u32>) {
    use rayon::prelude::*;
    let (x0, x1, nx) = x_range;
    let (y0, y1, ny) = y_range;
    let (z0, z1, nz) = z_range;
    let dx = (x1 - x0) / nx as f32;
    let dy = (y1 - y0) / ny as f32;
    let dz = (z1 - z0) / nz as f32;

    let sx = ny + 1;
    let sy = nz + 1;
    // Parallel grid sampling: each (ix,iy) row is independent.
    let grid: Vec<f32> = (0..(nx + 1) * sx)
        .into_par_iter()
        .map(|idx| {
            let ix = idx / sx;
            let iy = idx % sx;
            let x = x0 + ix as f32 * dx;
            let y = y0 + iy as f32 * dy;
            (0..=nz)
                .map(|iz| {
                    let z = z0 + iz as f32 * dz;
                    f(x, y, z)
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect();

    // Parallel per-cell triangulation.  Each ix-slice produces a flat list
    // of triangle positions; rayon concatenates the slices.
    let tris: Vec<[f32; 3]> = (0..nx)
        .into_par_iter()
        .flat_map_iter(|ix| {
            let mut slice = Vec::new();
            for iy in 0..ny {
                for iz in 0..nz {
                    let v: [[f32; 3]; 8] = [
                        [x0 + (ix    ) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz    ) as f32 * dz],
                        [x0 + (ix + 1) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz    ) as f32 * dz],
                        [x0 + (ix + 1) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz    ) as f32 * dz],
                        [x0 + (ix    ) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz    ) as f32 * dz],
                        [x0 + (ix    ) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                        [x0 + (ix + 1) as f32 * dx, y0 + (iy    ) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                        [x0 + (ix + 1) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                        [x0 + (ix    ) as f32 * dx, y0 + (iy + 1) as f32 * dy, z0 + (iz + 1) as f32 * dz],
                    ];
                    let vals = [
                        grid[((ix    ) * sx + (iy    )) * sy + (iz    )],
                        grid[((ix + 1) * sx + (iy    )) * sy + (iz    )],
                        grid[((ix + 1) * sx + (iy + 1)) * sy + (iz    )],
                        grid[((ix    ) * sx + (iy + 1)) * sy + (iz    )],
                        grid[((ix    ) * sx + (iy    )) * sy + (iz + 1)],
                        grid[((ix + 1) * sx + (iy    )) * sy + (iz + 1)],
                        grid[((ix + 1) * sx + (iy + 1)) * sy + (iz + 1)],
                        grid[((ix    ) * sx + (iy + 1)) * sy + (iz + 1)],
                    ];
                    process_cell(&mut |x, y, z| f(x, y, z), &v, &vals)
                        .into_iter()
                        .for_each(|(a, b, c)| {
                            slice.push(a);
                            slice.push(b);
                            slice.push(c);
                        });
                }
            }
            slice
        })
        .collect();

    let n_tri = tris.len() / 3;
    let positions = tris;
    let mut indices: Vec<u32> = (0..n_tri as u32)
        .flat_map(|t| [t * 3, t * 3 + 1, t * 3 + 2])
        .collect();

    // Sequential winding fix (cheap relative to meshing).
    let h = 1e-4_f32;
    let mut fixed = 0usize;
    for tri in indices.chunks_exact_mut(3) {
        let a = positions[tri[0] as usize];
        let b = positions[tri[1] as usize];
        let c = positions[tri[2] as usize];
        let e1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let e2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let fnx = e1[1]*e2[2] - e1[2]*e2[1];
        let fny = e1[2]*e2[0] - e1[0]*e2[2];
        let fnz = e1[0]*e2[1] - e1[1]*e2[0];
        let flen = (fnx*fnx + fny*fny + fnz*fnz).sqrt();
        if flen < 1e-30 { continue; }
        let cx = (a[0]+b[0]+c[0])/3.0;
        let cy = (a[1]+b[1]+c[1])/3.0;
        let cz = (a[2]+b[2]+c[2])/3.0;
        let gx = f(cx+h,cy,cz) - f(cx-h,cy,cz);
        let gy = f(cx,cy+h,cz) - f(cx,cy-h,cz);
        let gz = f(cx,cy,cz+h) - f(cx,cy,cz-h);
        let glen = (gx*gx+gy*gy+gz*gz).sqrt();
        if glen < 1e-30 { continue; }
        let dot = fnx*gx + fny*gy + fnz*gz;
        if dot < 0.0 {
            tri.swap(1, 2);
            fixed += 1;
        }
    }
    if fixed > 0 {
        eprintln!("  [MC33] winding fix: {} tris flipped", fixed);
    }

    (positions, indices)
}
