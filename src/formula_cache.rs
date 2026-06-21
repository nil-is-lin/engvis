// ── Formula SVG texture cache ─────────────────────────────────────
//
// Embeds pre-generated Typst SVGs (from `formulas/svg/`) and rasterizes
// them to egui textures on demand.  Each surface name maps to one SVG.

use std::collections::HashMap;

/// Embedded SVG bytes keyed by surface name.
macro_rules! svg_bytes {
    ($($name:literal => $path:literal),* $(,)?) => {
        [$(($name, include_bytes!($path) as &[u8])),*]
    };
}

const EMBEDDED_SVGS: &[(&str, &[u8])] = &svg_bytes!(
    "sphere"           => "../formulas/svg/sphere.svg",
    "torus"            => "../formulas/svg/torus.svg",
    "gyroid"           => "../formulas/svg/gyroid.svg",
    "schwarz-p"        => "../formulas/svg/schwarz-p.svg",
    "schwarz-d"        => "../formulas/svg/schwarz-d.svg",
    "schoen-iwp"       => "../formulas/svg/schoen-iwp.svg",
    "neovius"          => "../formulas/svg/neovius.svg",
    "f-rd"             => "../formulas/svg/f-rd.svg",
    "lidinoid"         => "../formulas/svg/lidinoid.svg",
    "split-p"          => "../formulas/svg/split-p.svg",
    "fischer-koch-s"   => "../formulas/svg/fischer-koch-s.svg",
    "fischer-koch-y"   => "../formulas/svg/fischer-koch-y.svg",
    "fischer-koch-cp"  => "../formulas/svg/fischer-koch-cp.svg",
    // Morphology formulas
    "morph-minimal"    => "../formulas/svg/morph-minimal.svg",
    "morph-shell"      => "../formulas/svg/morph-shell.svg",
    "morph-skeletal"   => "../formulas/svg/morph-skeletal.svg",
);

/// Cache of rasterized formula textures.
///
/// SVGs are rasterized lazily on first access and cached for the
/// lifetime of the application.
pub struct FormulaCache {
    textures: HashMap<String, egui::TextureHandle>,
}

impl FormulaCache {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
        }
    }

    /// Get (or lazily create) the egui texture for a surface formula.
    ///
    /// Returns `None` if the surface name has no embedded SVG, or if
    /// rasterization fails.
    pub fn get(&mut self, ctx: &egui::Context, name: &str) -> Option<&egui::TextureHandle> {
        if self.textures.contains_key(name) {
            return self.textures.get(name);
        }

        let svg_data = EMBEDDED_SVGS.iter()
            .find(|(n, _)| *n == name)?
            .1;

        let color_image = rasterize_svg(svg_data)?;
        let tex = ctx.load_texture(
            format!("formula-{name}"),
            color_image,
            egui::TextureOptions::LINEAR,
        );
        self.textures.insert(name.to_string(), tex);
        self.textures.get(name)
    }
}

/// Rasterize an SVG byte slice to an `egui::ColorImage` using resvg.
fn rasterize_svg(svg_bytes: &[u8]) -> Option<egui::ColorImage> {
    let tree = resvg::usvg::Tree::from_data(
        svg_bytes,
        &resvg::usvg::Options::default(),
    ).ok()?;

    // Scale to a reasonable display size (3x for HiDPI clarity).
    let scale = 3.0_f32;
    let size = tree.size();
    let w = (size.width() * scale).ceil() as u32;
    let h = (size.height() * scale).ceil() as u32;
    if w == 0 || h == 0 { return None; }

    let mut pixmap = resvg::tiny_skia::Pixmap::new(w, h)?;
    let transform = resvg::tiny_skia::Transform::from_scale(scale, scale);
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    // Convert RGBA pixmap to egui ColorImage.
    let pixels: Vec<egui::Color32> = pixmap.pixels().iter().map(|p| {
        egui::Color32::from_rgba_premultiplied(p.red(), p.green(), p.blue(), p.alpha())
    }).collect();

    Some(egui::ColorImage {
        size: [w as usize, h as usize],
        pixels,
        source_size: egui::Vec2::new(w as f32, h as f32),
    })
}
