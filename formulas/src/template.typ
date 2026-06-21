// TPMS formula template for typst -> SVG
// Usage: typst compile --input name="Gyroid" --input formula="..." template.typ output.svg

#set page(width: auto, height: auto, margin: (x: 12pt, y: 10pt), fill: none)
#set text(font: "Linux Libertine", size: 16pt, fill: rgb("#e0e0e0"))

#let name = sys.inputs.at("name", default: "Surface")
#let formula = sys.inputs.at("formula", default: "f = 0")
#let morph = sys.inputs.at("morph", default: "f = 0")

#table(
  columns: (auto, auto),
  align: (left, left),
  inset: (x: 8pt, y: 3pt),
  stroke: none,
  table.header(
    [*#name*], [],
  ),
  $ #formula $, [],
  text(size: 12pt, fill: rgb("#999999"), morph), [],
)
