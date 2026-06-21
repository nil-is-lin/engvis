// Morphology formula template for typst -> SVG
// Usage: typst compile --input formula="f = 0" morph-template.typ output.svg

#set page(width: auto, height: auto, margin: (x: 8pt, y: 6pt), fill: none)
#set text(font: "Linux Libertine", size: 14pt, fill: rgb("#999999"))

#let formula = sys.inputs.at("formula", default: "f = 0")

$ #formula $
