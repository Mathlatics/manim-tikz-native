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
                                               └── open-face / hinge solver
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

The solver is NumPy-based. The Cairo binding uses fixed-capacity Manim line
slots so object identity stays stable across animation frames.

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
