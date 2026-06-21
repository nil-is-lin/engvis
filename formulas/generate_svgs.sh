#!/bin/bash
# Generate SVG formulas from typst template
# Usage: bash formulas/generate_svgs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/src/template.typ"
OUT_DIR="$SCRIPT_DIR/svg"

mkdir -p "$OUT_DIR"

# Define formulas: name|formula|morphology
FORMULAS=(
  "sphere|x^2 + y^2 + z^2 = r^2|Primitive"
  "torus|(sqrt(x^2 + y^2) - R)^2 + z^2 = r^2|Primitive"
  "gyroid|sin(kx) cos(ky) + sin(ky) cos(kz) + sin(kz) cos(kx) = 0|Gyroid (G)"
  "schwarz-p|cos(kx) + cos(ky) + cos(kz) = 0|Schwarz P (Primitive)"
  "schwarz-d|sin(kx) sin(ky) sin(kz) + sin(kx) cos(ky) cos(kz) + cos(kx) sin(ky) cos(kz) + cos(kx) cos(ky) sin(kz) = 0|Schwarz D (Diamond)"
  "schoen-iwp|2[cos(kx) cos(ky) + cos(ky) cos(kz) + cos(kz) cos(kx)] - [cos(2kx) + cos(2ky) + cos(2kz)] = 0|Schoen IWP"
  "neovius|3[cos(kx) + cos(ky) + cos(kz)] + 4 cos(kx) cos(ky) cos(kz) = 0|Neovius"
  "f-rd|4 cos(kx) cos(ky) cos(kz) - [cos(2kx) cos(2ky) + cos(2ky) cos(2kz) + cos(2kz) cos(2kx)] = 0|Schoen F-RD"
  "lidinoid|1/2 [sin(2kx) cos(ky) sin(kz) + sin(2ky) cos(kz) sin(kx) + sin(2kz) cos(kx) sin(ky)] - 1/2 [cos(2kx) cos(2ky) + cos(2ky) cos(2kz) + cos(2kz) cos(2kx)] + 0.15 = 0|Lidinoid"
  "split-p|1.1 [sin(2kx) cos(ky) sin(kz) + sin(kx) cos(2ky) sin(kz) + cos(kx) sin(ky) cos(2kz)] - 0.2 [cos(2kx) cos(2ky) + cos(2ky) cos(2kz) + cos(2kz) cos(2kx)] - 0.4 [cos(2kx) + cos(2ky) + cos(2kz)] = 0|Split-P"
  "fischer-koch-s|cos(2kx) sin(ky) cos(kz) + cos(2ky) sin(kz) cos(kx) + cos(2kz) sin(kx) cos(ky) = 0|Fischer-Koch S"
  "fischer-koch-y|2 cos(kx) cos(ky) cos(kz) + sin(2kx) sin(ky) + sin(2ky) sin(kz) + sin(2kz) sin(kx) = 0|Fischer-Koch Y"
  "fischer-koch-cp|cos(kx) + cos(ky) + cos(kz) + 4 cos(kx) cos(ky) cos(kz) = 0|Fischer-Koch CP"
)

echo "Generating SVG formulas..."
for entry in "${FORMULAS[@]}"; do
  IFS='|' read -r name formula morph <<< "$entry"
  outfile="$OUT_DIR/${name}.svg"

  typst compile \
    --format svg \
    --input "name=$name" \
    --input "formula=$formula" \
    --input "morph=$morph" \
    "$TEMPLATE" \
    "$outfile" 2>/dev/null

  if [ -f "$outfile" ]; then
    echo "  [OK] $name -> $outfile"
  else
    echo "  [FAIL] $name"
  fi
done

echo ""
echo "Done. SVGs are in: $OUT_DIR/"
echo "Total: $(ls "$OUT_DIR"/*.svg 2>/dev/null | wc -l | tr -d ' ') files"
