// ── mesh_io ──────────────────────────────────────────────────
// Triangle-mesh import / export for OBJ, STL, PLY.
//
// We use mature crates rather than reimplementing parsers:
//   • `tobj`        — Wavefront OBJ reader
//   • `stl_io`      — binary/ASCII STL reader, binary STL writer
//   • `ply_rs_bw`   — PLY reader (ASCII / LE / BE)
// OBJ and PLY writers are tiny inline functions because the formats
// are trivial to emit and pulling extra crates for one-shot writers
// is overkill.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use engvis_core::mesh::{Mesh, MeshVertex, SubMesh};
use engvis_core::aabb::Aabb;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshFormat { Obj, Stl, Ply }

impl MeshFormat {
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "obj" => Some(Self::Obj),
            "stl" => Some(Self::Stl),
            "ply" => Some(Self::Ply),
            _ => None,
        }
    }
}

pub fn load_mesh(path: &Path) -> Result<Mesh, String> {
    let fmt = MeshFormat::from_path(path)
        .ok_or_else(|| format!("unsupported extension: {}", path.display()))?;
    match fmt {
        MeshFormat::Obj => load_obj(path),
        MeshFormat::Stl => load_stl(path),
        MeshFormat::Ply => load_ply(path),
    }
}

pub fn save_mesh(mesh: &Mesh, path: &Path) -> Result<(), String> {
    let fmt = MeshFormat::from_path(path)
        .ok_or_else(|| format!("unsupported extension: {}", path.display()))?;
    match fmt {
        MeshFormat::Obj => save_obj(mesh, path),
        MeshFormat::Stl => save_stl(mesh, path),
        MeshFormat::Ply => save_ply(mesh, path),
    }
}

// ── OBJ ─────────────────────────────────────────────────────────
fn load_obj(path: &Path) -> Result<Mesh, String> {
    let (models, _mats) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)
        .map_err(|e| format!("tobj: {e}"))?;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for m in &models {
        let base = positions.len() as u32;
        for chunk in m.mesh.positions.chunks_exact(3) {
            positions.push([chunk[0], chunk[1], chunk[2]]);
        }
        for &i in &m.mesh.indices {
            indices.push(base + i);
        }
    }
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh").to_string();
    Ok(build_mesh_with_normals(&name, positions, indices))
}

fn save_obj(mesh: &Mesh, path: &Path) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(w, "# engvis OBJ export — {} vertices, {} triangles",
        mesh.vertices.len(), mesh.indices.len() / 3).map_err(|e| e.to_string())?;
    for v in &mesh.vertices {
        writeln!(w, "v {} {} {}", v.position[0], v.position[1], v.position[2])
            .map_err(|e| e.to_string())?;
    }
    for v in &mesh.vertices {
        writeln!(w, "vn {} {} {}", v.normal[0], v.normal[1], v.normal[2])
            .map_err(|e| e.to_string())?;
    }
    for tri in mesh.indices.chunks_exact(3) {
        // OBJ indices are 1-based; same index for vn as for v.
        let (a, b, c) = (tri[0] + 1, tri[1] + 1, tri[2] + 1);
        writeln!(w, "f {a}//{a} {b}//{b} {c}//{c}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ── STL ─────────────────────────────────────────────────────────
fn load_stl(path: &Path) -> Result<Mesh, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let stl = stl_io::read_stl(&mut f).map_err(|e| format!("stl_io: {e}"))?;
    let positions: Vec<[f32; 3]> = stl.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();
    let mut indices: Vec<u32> = Vec::with_capacity(stl.faces.len() * 3);
    for face in &stl.faces {
        indices.push(face.vertices[0] as u32);
        indices.push(face.vertices[1] as u32);
        indices.push(face.vertices[2] as u32);
    }
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh").to_string();
    Ok(build_mesh_with_normals(&name, positions, indices))
}

fn save_stl(mesh: &Mesh, path: &Path) -> Result<(), String> {
    use stl_io::{Triangle, Normal, Vertex};
    let tris: Vec<Triangle> = mesh.indices.chunks_exact(3).map(|tri| {
        let p0 = mesh.vertices[tri[0] as usize].position;
        let p1 = mesh.vertices[tri[1] as usize].position;
        let p2 = mesh.vertices[tri[2] as usize].position;
        let e1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
        let e2 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
        let n = [
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0],
        ];
        let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt().max(1e-20);
        Triangle {
            normal:   Normal::new([n[0]/len, n[1]/len, n[2]/len]),
            vertices: [Vertex::new(p0), Vertex::new(p1), Vertex::new(p2)],
        }
    }).collect();
    let mut f = File::create(path).map_err(|e| e.to_string())?;
    stl_io::write_stl(&mut f, tris.iter()).map_err(|e| format!("stl_io: {e}"))
}

// ── PLY ─────────────────────────────────────────────────────────
fn load_ply(path: &Path) -> Result<Mesh, String> {
    use ply_rs_bw::parser::Parser;
    use ply_rs_bw::ply::{DefaultElement, Property};
    let mut f = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    let parser = Parser::<DefaultElement>::new();
    let ply = parser.read_ply(&mut f).map_err(|e| format!("ply_rs_bw: {e}"))?;

    let get_f = |p: Option<&Property>| -> f32 {
        match p {
            Some(Property::Float(v)) => *v,
            Some(Property::Double(v)) => *v as f32,
            _ => 0.0,
        }
    };

    let mut positions: Vec<[f32; 3]> = Vec::new();
    if let Some(verts) = ply.payload.get("vertex") {
        for el in verts {
            positions.push([get_f(el.get("x")), get_f(el.get("y")), get_f(el.get("z"))]);
        }
    }
    let mut indices: Vec<u32> = Vec::new();
    if let Some(faces) = ply.payload.get("face") {
        for el in faces {
            // face index property is variously named "vertex_indices",
            // "vertex_index" or similar; pick whichever list we find.
            let list = el.iter().find_map(|(_, p)| match p {
                Property::ListInt(v) => Some(v.iter().map(|x| *x as u32).collect::<Vec<_>>()),
                Property::ListUInt(v) => Some(v.iter().copied().collect::<Vec<_>>()),
                Property::ListShort(v) => Some(v.iter().map(|x| *x as u32).collect::<Vec<_>>()),
                Property::ListUShort(v) => Some(v.iter().map(|x| *x as u32).collect::<Vec<_>>()),
                Property::ListChar(v) => Some(v.iter().map(|x| *x as u32).collect::<Vec<_>>()),
                Property::ListUChar(v) => Some(v.iter().map(|x| *x as u32).collect::<Vec<_>>()),
                _ => None,
            });
            if let Some(idx) = list {
                // Triangulate fan if needed.
                if idx.len() >= 3 {
                    for k in 1..idx.len() - 1 {
                        indices.push(idx[0]);
                        indices.push(idx[k]);
                        indices.push(idx[k + 1]);
                    }
                }
            }
        }
    }
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh").to_string();
    Ok(build_mesh_with_normals(&name, positions, indices))
}

fn save_ply(mesh: &Mesh, path: &Path) -> Result<(), String> {
    // Minimal ASCII PLY writer — straightforward and avoids the
    // complexity of ply-rs-bw's element/property API for a one-shot
    // export. The format is well-defined enough that hand-writing it
    // is shorter than wiring up the typed writer.
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    writeln!(w, "ply").map_err(|e| e.to_string())?;
    writeln!(w, "format ascii 1.0").map_err(|e| e.to_string())?;
    writeln!(w, "comment engvis PLY export").map_err(|e| e.to_string())?;
    writeln!(w, "element vertex {}", mesh.vertices.len()).map_err(|e| e.to_string())?;
    writeln!(w, "property float x").map_err(|e| e.to_string())?;
    writeln!(w, "property float y").map_err(|e| e.to_string())?;
    writeln!(w, "property float z").map_err(|e| e.to_string())?;
    writeln!(w, "property float nx").map_err(|e| e.to_string())?;
    writeln!(w, "property float ny").map_err(|e| e.to_string())?;
    writeln!(w, "property float nz").map_err(|e| e.to_string())?;
    writeln!(w, "element face {}", mesh.indices.len() / 3).map_err(|e| e.to_string())?;
    writeln!(w, "property list uchar int vertex_indices").map_err(|e| e.to_string())?;
    writeln!(w, "end_header").map_err(|e| e.to_string())?;
    for v in &mesh.vertices {
        writeln!(w, "{} {} {} {} {} {}",
            v.position[0], v.position[1], v.position[2],
            v.normal[0], v.normal[1], v.normal[2]).map_err(|e| e.to_string())?;
    }
    for tri in mesh.indices.chunks_exact(3) {
        writeln!(w, "3 {} {} {}", tri[0], tri[1], tri[2]).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ── Build helper: positions+indices → Mesh with smooth normals ──
fn build_mesh_with_normals(name: &str, positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Mesh {
    let n = positions.len();
    let mut normals = vec![[0.0_f32; 3]; n];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= n || i1 >= n || i2 >= n { continue; }
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let e1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
        let e2 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
        let nrm = [
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0],
        ];
        for &i in &[i0, i1, i2] {
            normals[i][0] += nrm[0];
            normals[i][1] += nrm[1];
            normals[i][2] += nrm[2];
        }
    }
    let mut aabb = Aabb::empty();
    let vertices: Vec<MeshVertex> = positions.iter().zip(normals.iter()).map(|(p, n)| {
        aabb.expand(glam::Vec3::from(*p));
        let l = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
        let nrm = if l > 1e-10 { [n[0]/l, n[1]/l, n[2]/l] } else { [0.0, 1.0, 0.0] };
        MeshVertex {
            position: *p,
            normal: nrm,
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        }
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
