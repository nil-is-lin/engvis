# ADR-003: Extract TPMS into a standalone `engvis-tpms` crate

## Status
Accepted

## Context
TPMS (Triply Periodic Minimal Surfaces) support had grown into a large, special
corner of the codebase: 11 surface families, a volume-fraction → C solver, the
`GradientField` spatial-blend DSL, and the `Morphology` build mode (Minimal /
Shell / Skeletal).  It depended only on `fidget` (the implicit-surface library)
and on no `engvis-*` crate, yet it was tangled inside `engvis-surface` — which
is meant to model *generic* implicit surfaces (sphere, torus, custom Rhai
expressions).

Problems this caused:

1. **False coupling.** `engvis-surface` carried TPMS-specific concerns
   (`tpms_period`, `tpms_cells`, `tpms_vol_frac`, `blend_secondary`, …) that had
   nothing to do with a generic surface type.  Consumers pulling in
   `engvis-surface` for a plain sphere also dragged in the TPMS formula table
   and the volume-fraction solver.
2. **Muddied domain boundary.** "TPMS" is a distinct *kind* of surface with its
   own parameters and invariants — it deserved to be a first-class type, not a
   set of variants bolted onto `SurfaceType`.
3. **Publish blast radius.** Every TPMS tweak forced a version bump of
   `engvis-surface` (and everything downstream), even when the generic-surface
   code was untouched.

The goal: make TPMS a separate, decoupled crate that owns *only* TPMS data and
math, and expose it as a first-class `TpmsSurface` type.

## Decision
- **New crate `engvis-tpms`** depends *only* on `fidget` (`fidget-core`,
  `fidget-rhai`, `fidget-jit`).  It has **no dependency** on `engvis-core`,
  `engvis-surface`, `engvis-mesher`, or `engvis-renderer`.  This is the key
  invariant: TPMS math is portable and reusable on its own.
- **`TpmsSurface` is the first-class "TPMS type".** A struct owning
  `family, period, cells, cell_size, amplitude, offset, vol_frac, thickness,
  blend_secondary, blend_weight_field, offset_field`, with methods:
  `new(family)` (family-default period/cell/amplitude),
  `formula_desc()`, `build_tree(rotation_axis, rotation_angle_rad)`,
  `shell_grad()`, `min_feature()`, `domain_extent()`, `cells_f32()`,
  `solve_c(&tree)`.  The host assembles a `TpmsSurface` from UI fields and asks
  it for everything it needs — no TPMS bookkeeping leaks into `engvis-surface`.
- **`TpmsFamily`** is an 11-variant enum (`Gyroid`, `SchwarzP`, `SchwarzD`,
  `SchoenIwp`, `Neovius`, `FRD`, `Lidinoid`, `SplitP`, `FischerKochSY`,
  `FischerKochSCP`, `FischerKochSC`) with `name()` / `label()` / `from_name()` /
  `all()`.  It is `Copy` and replaces the old `SurfaceType::Tpms*` variants.
- **`Morphology` (Minimal / Shell / Skeletal) is colocated in `engvis-tpms`.**
  The user correctly pointed out `Morphology` is TPMS-specific (it only makes
  sense for an iso-surface of an implicit function), so it belongs with the
  TPMS crate rather than `engvis-surface`.
- **`GradientField` / `GradientMode`** (the spatial-blend DSL) moved to
  `engvis-tpms` too, since it is purely a TPMS blend-weight concern.
- **`engvis-surface` slimmed to primitives:** `SurfaceType` is now only
  `Sphere | Torus | Custom(String)`.  All TPMS formula / morphology / gradient
  code removed.  `build_tree` handles sphere/torus (fallback unit sphere) and
  `build_tree_from_rhai` is retained for custom expressions.
- **`engvis-surface` does NOT depend on `engvis-tpms`.**  Instead, the binary
  (`engvis-cli`) and `engvis-mesher` import `Morphology` / `TpmsFamily` /
  `TpmsSurface` directly from `engvis-tpms`.  This keeps the dependency graph a
  clean DAG and avoids a `surface → tpms → surface` back-edge.
- **Binary treats TPMS as a first-class selector:** `App` holds
  `tpms_family: Option<TpmsFamily>`; `current_tree()` builds via
  `TpmsSurface::build_tree` when active and via `engvis-surface` otherwise.
  The raw `tpms_*` UI fields are assembled on demand into a `TpmsSurface` in
  `make_tpms_surface()` rather than stored as a bespoke `TreeParams` blob.
- **Publish order** in `publish.yml` updated to
  `engvis-core → engvis-surface → engvis-tpms → engvis-mesher →
  engvis-renderer → engvis`.  (`engvis-tpms` sits after `engvis-surface` but
  before `engvis-mesher`, which depends on it.)

## Consequences
- Good: `engvis-tpms` is **fully decoupled** — it can be used, tested, and
  published independently of the viewer.  A future CLI/headless TPMS tool, or a
  different front-end, can depend on just this crate.
- Good: `engvis-surface` is now lean and honest about its domain (generic
  implicit surfaces).  TPMS changes no longer bump `engvis-surface`.
- Good: the volume-fraction → C solver, family table, and morphological modes
  are unit-tested inside `engvis-tpms` with no viewer dependency.
- Trade-off **accepted**: `Morphology` is a *generic* mesh-build concept (the
  mesher's `build_mesh`/`build_shell_mesh` use it for primitives too), but it
  now lives in the TPMS-named crate.  This is a minor semantic awkwardness: a
  non-TPMS caller (e.g. a primitive skeletal mesh) imports `Morphology` from
  `engvis_tpms`.  We chose this over (a) making `engvis-surface` depend on
  `engvis-tpms` (would create a cycle / false coupling) or (b) a separate
  `engvis-morphology` crate (over-splitting for one small enum).  Documented
  here so the next reader understands *why*.
- Trade-off **accepted**: the binary still carries a handful of raw `tpms_*`
  UI fields (period / cells / thickness / vol_frac / …) rather than a single
  `TpmsSurface` value, because the ~50 existing slider bindings mutate them
  individually.  They are assembled into a `TpmsSurface` only at build time.
  This keeps the UI refactor local and reversible; a future cleanup could
  collapse them into one editable `TpmsSurface` value.
- Cost: one more crate to publish and version.  Acceptable given the coupling
  and publish-blast-radius wins.
