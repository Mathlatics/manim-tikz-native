# Architecture

`manim-tikz-native` keeps source parsing, geometry, rendering, animation, and
application integration separate.

```text
restricted TikZ source
        │
        ▼
TikzNativeCompiler ──► DocumentSpec / PictureSpec
        │                       │
        │                       ├──► native 2D/3D Manim renderers
        │                       ├──► geometry-rig analysis
        │                       ├──► readable Manim source generation
        │                       └──► visibility adapters
        │                                      │
        ▼                                      ▼
versioned JSON bridges                 polyhedron_visibility
                                               │
                                               ├── closed convex hull solver
                                               ├── open-face / hinge solver
                                               ├── convex-section solver
                                               ├── source/copy identity handoff
                                               └── derived-dihedral compositor
```

## Packages

### `tikz_native`

The restricted compiler, semantic model, native Manim renderers, 2D/3D motion
runtime, code generators, schemas, and command-line bridges.

The compiler reports unsupported syntax instead of returning an opaque fallback
object. `PictureSpec.objects` contains stable semantic IDs, and named geometry
relations are retained for later animation.

### `polyhedron_visibility`

A TikZ-independent automatic-occlusion module for ordinary Manim scenes. It
operates on explicit topology and a parallel projection:

- `VisibilityModel` / `OcclusionScene3D` for closed convex polyhedra;
- `OpenFaceVisibilityModel` / `OpenFaceScene3D` for finite convex panels and
  articulated hinges.
- `CopyIdentityHandoffMap` for explicit source/copy vertex, face, and stroke
  lineage plus a projection-aware continuous paint-ownership handoff.
- `DerivedDihedralModel` / `ExtractedDihedralScene3D` for one closed solid and
  one rigid two-face teaching copy derived from adjacent source faces; this
  controller consumes the generic copy handoff instead of maintaining a
  separate coincidence heuristic.
- `BasePlaneRotation3D` for a center-relative rotation template that turns one
  validated source face into the horizontal bottom plane; each entity's local
  placement moves that center with the entity.

The solver is NumPy-based. The Cairo binding uses fixed-capacity Manim line
slots so object identity stays stable across animation frames. Exact
transparent section and derived-dihedral paths also use conservative,
preallocated triangle pools; they never allocate fragments inside a frame
updater. The derived-dihedral path gates partitions by finite polygon
crossings, then submits consecutive same-source triangles as one compound fill
path so solver triangulation does not become a visible Cairo seam. Its unified
compositing stage further splits semantic stroke spans only at local depth
events, builds face/line and line/line ordering constraints, rejects cycles,
and maps the resulting deterministic far-to-near order onto the existing fixed
triangle and line slots.

## Component revisions

The package publishes one build revision and independent component revisions.
Changing an occlusion solver should not invalidate an unrelated 2D asset unless
its dependency graph actually includes that solver. Persisted integrations can
also compare contract revisions, which only change when saved author data can
no longer be read safely.

Run `tikz-native health` to inspect these identities.

## Application boundary

The repository does not define a browser editor, presentation model, timeline,
or persistence layer. Applications should treat compiler output, editable
author data, preview media, and final rendered media as separate layers.
