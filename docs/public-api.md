# Public API

The project exposes a Python API for Manim authors and versioned JSON bridges
for process isolation.

## Compile TikZ

```python
from pathlib import Path
from tikz_native import compile_document

document = compile_document(Path("figure.tex"))
picture = document.pictures[0]
if picture.unsupported:
    raise ValueError(picture.unsupported)
```

Use `NativeManimRenderer` for 2D and `NativeManim3DRenderer` for a fixed 3D
projection:

```python
from tikz_native.manim_renderer import NativeManimRenderer
from tikz_native.manim_renderer_3d import NativeManim3DRenderer

figure_2d = NativeManimRenderer(scene_unit_per_cm=1.0).render(picture)
figure_3d = NativeManim3DRenderer(scene_unit_per_cm=1.0).render(picture)
```

Both figures expose an object mapping keyed by stable semantic IDs. Pass a
custom Manim `TexTemplate` to the renderer when the portable Fandol/Latin
Modern defaults are not suitable.

## Ordinary Manim automatic occlusion

Closed convex polyhedron:

```python
from polyhedron_visibility import (
    OcclusionScene3D,
    OcclusionStyle,
    ParallelProjection,
)

visibility = OcclusionScene3D("solid")
visibility.vertex("A", lambda: point_a())
visibility.vertex("B", lambda: point_b())
visibility.vertex("C", lambda: point_c())
visibility.face("front", ("A", "B", "C"))
visibility.stroke("AB", "A", "B", line_ab)

controller = visibility.controller(
    scene,
    projection=ParallelProjection.identity(),
    style=OcclusionStyle(max_projected_length=8.0),
)
with controller.session():
    scene.play(...)
```

Open panels and articulated hinges:

```python
from polyhedron_visibility.open_faces import OpenFaceScene3D

visibility = OpenFaceScene3D("dihedral")
# Register vertices first.
visibility.face(
    "alpha",
    ("A", "B", "C", "D"),
    logical_surface_id="surface-alpha",
    source_mobject=alpha_polygon,
)
visibility.face(
    "beta",
    ("B", "A", "E", "F"),
    logical_surface_id="surface-beta",
    source_mobject=beta_polygon,
)
visibility.articulated_hinge("AB", "alpha", "beta", "A", "B")
visibility.stroke("probe", "P", "Q", probe_line)
```

`source_mobject` on every face enables automatic face fill ordering. Omitting
all face source objects keeps authored face order while still solving line
visibility.

## TikZ visibility adapters

- `tikz_native.polyhedron_visibility_3d_adapter.adapt_picture_visibility_3d`
- `tikz_native.polyhedron_visibility_3d_manim.bind_picture_visibility_3d`
- `tikz_native.open_face_visibility_3d_adapter.adapt_picture_open_face_visibility_3d`
- `tikz_native.open_face_visibility_3d_manim.bind_picture_open_face_visibility_3d`
- `tikz_native.open_face_static_asset_3d.bake_open_face_static_entry_3d`

The open-face adapter understands compiler-proven legacy relation fragments and
normalizes them into complete semantic strokes. It does not accept arbitrary
compound or curved source objects.

## JSON bridges

The installed console commands are:

```text
tikz-native
tikz-native-rig-2d
tikz-native-rig-3d
tikz-native-source-v3
```

Each command supports `health`. The run operations accept strict versioned JSON
requests; refer to `tikz_native/schemas/` and the bridge module constants for
the exact request and response contracts.

## Lifecycle rules

- Construct and add the source geometry to the Scene before attaching an
  occlusion controller.
- Use one controller per source figure.
- Put geometry restoration inside the visibility session. Restore visibility
  first, then the geometry rig.
- A failed frame preserves the last good overlay and never silently displays
  all lines as visible.
- The current real-time binding targets Manim's Cairo renderer.
