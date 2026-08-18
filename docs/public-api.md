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

Add stable classroom-oriented face depth cues without changing the hidden-line
solver:

```python
from polyhedron_visibility.depth_cue import (
    DepthCuedAutoOcclusion3D,
    FaceDepthCueStyle,
)

controller = DepthCuedAutoOcclusion3D(
    scene,
    model,
    position_provider=current_positions,
    stroke_bindings=edge_lines,
    face_fill_bindings=face_polygons,
    projection=ParallelProjection.identity(),
    style=OcclusionStyle(max_projected_length=8.0),
    face_style=FaceDepthCueStyle(),
)
```

Every managed face must be one fill-only native `Polygon` with one solid fill
and a distinct authored z-index; visible boundaries belong in the registered
semantic `Line` objects. The controller keeps the base face color, then
derives a nearby palette from face orientation and depth: distant or back-facing
faces fade toward the configurable fog color, near faces retain stronger color
and opacity, and a continuous warm/cool shift prevents different facets from
collapsing into one flat shade. Only visible silhouette spans are thickened;
hidden dashed spans keep their authored hidden style. All proxy identities stay
fixed and the source faces are restored on normal or exceptional exit.

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

## Convex-solid line intersections and moving sections

`ConvexSectionScene3D` reuses the strict closed-convex-solid contract, then
adds independent straight lines and exactly one moving mathematical plane:

```python
from polyhedron_visibility import OcclusionStyle, ParallelProjection
from polyhedron_visibility.sections import (
    ConvexSectionScene3D,
    ConvexSectionStyle,
    FaceDepthCueStyle,
    SectionPlane3D,
)

section = ConvexSectionScene3D("cube-section")

# Register the cube's live vertices and ordered maximal faces first.
for vertex_id, provider in vertex_providers.items():
    section.vertex(vertex_id, provider)
for face_id, vertex_ids in cube_faces.items():
    section.face(
        face_id,
        vertex_ids,
        source_mobject=face_polygons[face_id],
    )

# Surface edges name their incident faces. A free line uses an empty list, so
# every solid face may hide it.
section.stroke(
    "edge.A.B", "A", "B", edge_ab,
    incident_face_ids=("back", "bottom"),
)
section.vertex("X", lambda: free_line_start())
section.vertex("Y", lambda: free_line_end())
section.stroke("probe.X.Y", "X", "Y", free_line, incident_face_ids=())

section.cutting_plane(
    "moving-cut",
    lambda: SectionPlane3D(
        "moving-cut",
        point=current_plane_point(),
        normal=current_plane_normal(),
        # Minimum display dimensions; auto mode expands them as needed.
        half_width=0.10,
        half_height=0.10,
        u_axis=(1.0, -1.0, 0.0),
    ),
)

controller = section.controller(
    scene,
    projection=ParallelProjection.identity(),
    source_style=OcclusionStyle(max_projected_length=8.0),
    section_style=ConvexSectionStyle(max_boundary_projected_length=8.0),
    face_depth_style=FaceDepthCueStyle(),
    accurate_transparency=True,
)
with controller.session():
    scene.play(...)
```

The pure queries are available before rendering:

```python
cross_section = section.current_section()
line_hits = section.current_stroke_intersections()["probe.X.Y"]
```

`cross_section.kind` is `empty`, `point`, `segment`, or `polygon`.
`line_hits` includes the parameter interval inside the solid and exact
entry/exit point evidence. The real-time controller preallocates all Manim
objects, so topology changes do not replace Scene objects during animation.
When every face supplies `source_mobject`, the same controller also enables
orientation shading, depth-dependent opacity, and silhouette emphasis. The
cutting plane and derived section keep their separate explanatory colors.

The plane is infinite for section geometry. By default
`plane_patch_mode="auto"`, so `half_width` and `half_height` are only minimum
display sizes. Every frame derives the required rectangle from all solid-face
vertices, adds `plane_patch_margin=0.15`, and keeps the largest dimensions seen
during the attached session. This guarantees that the complete section stays
inside the visible patch without frame-to-frame shrinking. Use
`plane_patch_mode="strict"` only for an intentionally finite panel; an
undersized strict panel fails before drawing an incomplete section.

`accurate_transparency=True` additionally replaces each authored whole-face
fill with a stable pool of local triangles. The solver separates the fitted
display patch into `plane_outside` and `section_inside`, splits every crossed
solid face along the same intersection evidence, and orders only overlapping
fragments from far to near. This fixes the case where one part of the plane is
behind a face while another part is in front of it. The source face binding
must be one native, fill-only, non-gradient `Polygon` per face with a distinct
authored `z_index`; failures occur before any source object is hidden.

The default coplanar policy draws the highlighted section over a coincident
solid face. Set `transparent_coplanar_policy="solid_over_section"` to reverse
that teaching convention, or `"fail"` to reject the coincident state.

## TikZ visibility adapters

- `tikz_native.polyhedron_visibility_3d_adapter.adapt_picture_visibility_3d`
- `tikz_native.polyhedron_visibility_3d_manim.bind_picture_visibility_3d`
- `tikz_native.open_face_visibility_3d_adapter.adapt_picture_open_face_visibility_3d`
- `tikz_native.open_face_visibility_3d_manim.bind_picture_open_face_visibility_3d`
- `tikz_native.open_face_static_asset_3d.bake_open_face_static_entry_3d`
- `tikz_native.convex_section_3d_manim.bind_picture_convex_section_3d`

The TikZ convex-section binding accepts the same
`accurate_transparency=True`, `plane_patch_mode`, and `plane_patch_margin`
options. It only accepts compiler-proven semantic faces that map to one native
`Polygon` each; it never reconstructs a face from an arbitrary path or
`VGroup`.

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
