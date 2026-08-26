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

## Source-authoritative project builds

Keep the TikZ file, optional motion and Bridge template, render intent, and any
hooks paired with that Bridge template as the authored project. Rebuild
ShapeAssets, compositing plans, and generated Manim source as disposable
outputs with:

```bash
tikz-native-project build project.json
tikz-native-project status project.json
tikz-native-project rebuild project.json
tikz-native-project clean project.json
```

The equivalent module entry is `python -m tikz_native.source_project`. Both
entry points load the same single module. The manifest never stores a legacy or
unified implementation choice: new generated open-face output uses the current
unified adapter and fails closed instead of automatically falling back to
legacy behavior.

The CLI keeps stdout machine-readable: every non-error result emits one
`tikz-native-project-command-result/v1` JSON document and sends build logs to
stderr. `status` exits with 0 for a fresh project and 1 for missing or stale
derived output; both return JSON. Invalid input or an execution failure exits
with 2, leaves stdout empty, and writes the explanation to stderr.

See [Source-authoritative projects](source-authoritative-projects.md) for the
manifest, derived-output ownership, and component-revision rules.

## Ordinary Manim automatic occlusion

Closed convex polyhedron:

```python
from polyhedron_visibility import (
    OcclusionScene3D,
    OcclusionStyle,
    ParallelProjection,
)
from tikz_native.camera_3d import OBLIQUE_MATRIX

visibility = OcclusionScene3D("solid")
visibility.vertex("A", lambda: point_a())
visibility.vertex("B", lambda: point_b())
visibility.vertex("C", lambda: point_c())
visibility.face("front", ("A", "B", "C"))
visibility.stroke("AB", "A", "B", line_ab)

controller = visibility.controller(
    scene,
    projection=ParallelProjection(OBLIQUE_MATRIX),
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
    projection=ParallelProjection(OBLIQUE_MATRIX),
    style=OcclusionStyle(max_projected_length=8.0),
    face_style=FaceDepthCueStyle(),
)
```

`OBLIQUE_MATRIX` is the classroom `斜二测` view with a 45-degree,
half-scale receding axis and matches the default `MultiProjectionCamera`.
The occlusion binding keeps its `ParallelProjection` argument explicit so
its depth calculation cannot silently differ from the camera.  Pass any other
explicit 3-by-3 parallel-projection source when needed.  TikZ adapters are
source-authoritative and keep the projection compiled from the TikZ picture.

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

## Reuse one identity handoff for any copied geometry

When a second entity is copied from a registered first entity, freeze that
lineage instead of trying to rediscover it from pixels:

```python
from polyhedron_visibility.copy_handoff import (
    CopyIdentityHandoffMap,
    CopyIdentityHandoffPolicy,
    compute_copy_identity_handoff,
)

handoff = CopyIdentityHandoffMap.from_visibility_model(
    "solid-to-analysis-copy",
    solid_visibility_model,
    source_entity_id="solid",
    copy_entity_id="analysis-copy",
    # Omit these two arguments to pair the complete solid.
    face_ids=("front", "top"),
    stroke_ids=("edge.E.F", "edge.F.G", "edge.G.H"),
    policy=CopyIdentityHandoffPolicy(activation_distance=0.12),
)

frame = compute_copy_identity_handoff(
    handoff,
    source_positions=source_positions_by_runtime_vertex_id,
    copy_positions=copy_positions_by_runtime_vertex_id,
    final_point_provider=project_to_overlay_coordinates,
)

# Apply these factors to the corresponding source paint only. The copy stays
# at full opacity throughout the handoff.
source_face_scales = frame.source_face_opacity_scales
source_stroke_scales = frame.source_stroke_opacity_scales
```

Every source/copy vertex identity must be explicit and one-to-one. At zero
separation all paired source factors are zero; at half the configured distance
the cubic smoothstep factor is `0.5`; at or beyond the full distance it is
`1.0`. Each face and stroke uses only its own corresponding vertices, so a
partially copied or articulated structure does not unnecessarily fade unrelated
source primitives. `final_point_provider` determines whether separation is
measured in world, projected, or fixed-frame coordinates.

`ExtractedDihedralOcclusion3D` constructs and consumes this map automatically;
its existing `identity_handoff_distance` argument remains the convenient
author-facing control.

## Extract and move one dihedral from a closed solid

`ExtractedDihedralScene3D` uses the same closed-convex registration contract,
then freezes two adjacent source faces as one independent teaching copy:

```python
import numpy as np
from manim import GOLD, ValueTracker

from polyhedron_visibility import OcclusionStyle, ParallelProjection
from polyhedron_visibility.dihedral_extraction import (
    ExtractedDihedralScene3D,
    RigidTransform3D,
)

visibility = ExtractedDihedralScene3D("cube-analysis")

# Register every live solid vertex, maximal face, semantic boundary Line, and
# optional fill-only face Polygon just as for OcclusionScene3D.
for vertex_id, provider in vertex_providers.items():
    visibility.vertex(vertex_id, provider)
for face_id, vertex_ids in cube_faces.items():
    visibility.face(
        face_id,
        vertex_ids,
        source_mobject=face_polygons[face_id],
    )
for edge_id, (start, end, incident_faces) in cube_edges.items():
    visibility.stroke(
        edge_id,
        start,
        end,
        edge_lines[edge_id],
        incident_face_ids=incident_faces,
    )

progress = ValueTracker(0.0)
separation = np.asarray((2.4, -0.8, 0.9))
entity = visibility.extract_dihedral(
    "analysis-copy",
    ("front", "top"),
    transform_provider=lambda: RigidTransform3D.translation_by(
        separation * progress.get_value()
    ),
    edge_color=GOLD,
    face_color=GOLD,
    face_opacity=0.40,
)
scene.add(entity.mobject)

# Turn one source face into the horizontal bottom face. The source solid's
# geometric center remains fixed and the face's validated outward normal
# finishes at world -Z.
base_progress = ValueTracker(0.0)
base_rotation = visibility.base_plane_rotation("right")

def source_transform():
    source_shift = RigidTransform3D.translation_by(
        -0.5 * separation * progress.get_value()
    )
    return source_shift.compose(
        base_rotation.transform(base_progress.get_value())
    )

controller = visibility.controller(
    scene,
    projection=ParallelProjection(projection_matrix),
    style=OcclusionStyle(max_projected_length=10.0),
    accurate_transparency=True,
    # None is the default: exact transparency automatically enables the
    # shared face/stroke painter graph. Set False only for legacy comparison.
    unified_compositing=None,
    unified_fragment_slots_per_style=12,
    global_transform_provider=source_transform,
    # Final overlay-coordinate distance over which coincident source faces and
    # edges smoothly regain paint ownership. The default is 0.12.
    identity_handoff_distance=0.12,
)
with controller.session():
    scene.play(progress.animate.set_value(1.0))
    scene.play(base_progress.animate.set_value(1.0))
```

The selected faces must share exactly one edge. Every unique boundary edge of
the two-face union must map to exactly one registered semantic `Line`.
`RigidTransform3D` only accepts proper right-handed rotations and finite
translations; it rejects scaling, reflection, and shear.

`base_plane_rotation(face_id)` reads the validated outward orientation of one
source face. By default it rotates that normal to world `(0, 0, -1)`, so the
solid lies above a horizontal bottom face. The source solid's geometric center
(the centroid of all registered solid vertices) defines the rotation pivot.
The extracted dihedral inherits that authored center; its local placement moves
the center with the copy, so the separated copy rotates about its own current
center. The shortest normal-alignment rotation is used. Pass another
`target_outward_normal` or an explicit `anchor` when the lesson needs a
different base direction or pivot.

When `global_transform_provider` is present, the controller owns the shared
center-relative motion for that session. The source solid receives transform
`G`; the extracted dihedral receives `L.compose(G)`, where `L` is its local
placement. Thus a translation inside `L` moves the copy and its rotation center
together instead of making the copy orbit the source center. The example uses
`-T/2` for the source and relative placement `T` for the copy, so the two finish
at opposite sides. Source `Line` and fill-only `Polygon` geometry, hidden-line
slots, and exact transparent fragments are all updated from the same
once-per-frame samples.

If the authored solid is already drawn in final parallel-projected Scene
coordinates, pass the same callable as `display_point_provider` and set
`source_coordinate_mode="display"`. The extracted face and edge source objects
then follow that display mapping before each validation and frame solve.

At the identity transform, selected source fills and edges are suppressed while
the highlighted copy is shown once. When the copy separates, the corresponding
source face and edge slots regain opacity through a geometry-driven smoothstep
instead of switching on in the first non-identity frame. The same handoff runs
in reverse when the copy returns. `identity_handoff_distance` is measured in
the final overlay coordinate space and must be finite and positive. After that
distance both entities are fully active. `accurate_transparency=True` requires
one fill-only,
non-gradient native `Polygon` for every solid and extracted face, with distinct
authored face z-indices and no unrelated drawable inside that z band.
Finite polygon intersection evidence gates every transparent split; crossing
infinite supporting planes alone is not enough. Triangles that remain
consecutive in the valid depth order and share one source face are rendered as
one compound fill batch. Their identities remain available in the trace, but
their internal antialiasing boundaries are absent from the image.

Unless explicitly disabled, `accurate_transparency=True` also enables
`unified_compositing`. The controller refines every authoritative visible or
hidden span at projected line/line crossings and line/face depth exchanges,
then orders the resulting stroke fragments together with exact transparent
face batches. The fixed slot pool contains
`unified_fragment_slots_per_style` visible and hidden fragments per semantic
stroke; exceeding that author-time capacity keeps the last-good frame and
raises an error. All managed face and stroke source objects must have distinct
finite authored `z_index` values, and no unrelated drawable may occupy the
combined managed band.

## Quadratic surfaces and conic sections

`polyhedron_visibility.quadrics` is a lazy public namespace.  Importing its
renderer-neutral contracts and solvers does not import Manim.  Its main entry
points are:

- `SphereSpec`, `CylinderSpec`, `ConeSpec`, `SectionPlane`;
- `compute_quadric_section()` and `section_trace_curves()`;
- `compute_quadric_visibility()` and
  `compute_projected_curve_crossings()`;
- `compute_quadric_compositing()` and
  `compute_global_quadric_frame()`;
- `compute_quadric_section_compositing()` and
  `quadric_plane_fragment_contours()`;
- `fit_plane_display_patch()`;
- `compute_plane_motion_schedule()`,
  `track_scheduled_plane_section()`, and `track_moving_section_point()`;
- `build_section_transition_plan()` and
  `QuadricSectionTransition3D`;
- `QuadricOcclusion3D`, `QuadricManimStyle`, `QuadricBoundaryStyle`, and
  `QuadricManimLimits`.

The Manim binding accepts immutable surface/curve sequences or callbacks that
return a new sequence for the current frame.  IDs and counts remain fixed for
the lifetime of an attached controller.  `paint_policy="physical"` omits
hidden spans; `paint_policy="diagrammatic"` draws them with the hidden dash
style as a global teaching overlay;
`paint_policy="depth_aware_diagrammatic"` keeps the dash but brackets it after
every surface certified farther than its occluders and before the occluding
surfaces themselves.  Section compositing places it between the back/front
projection sheets.  The runtime targets Cairo, preallocates all
Mobjects, and commits each prepared painter frame transactionally.  Its default
`surface_order_mode="automatic"` recomputes the certified global frame on each
update and exposes the committed evidence through `last_global_frame`;
`surface_order_mode="explicit"` retains the legacy manual-constraint path.
In unified mode, `boundary_styles={style_id: QuadricBoundaryStyle(...)}` is an
immutable renderer-level registry. A `GeneratorBoundarySpec.style_id` must
resolve in that registry; unknown identities and fixed dash-slot overflow fail
before a frame is committed. Built-in IDs preserve the historical curve,
surface-boundary, silhouette, and section-outline appearance.
Pass `section_plane=plane` to place one finite display patch, the two projected
surface sheets, the section curves, and every visible/hidden curve fragment in
one painter graph.  The patch is adaptively split into regions behind the
solid, between its far and near sheets, in front of the solid, or outside its
projection.  The Manim layer merges those cells back into continuous compound
contours, so the geometric subdivision does not leave triangle seams.  No
Mobject is created during an update, and `last_section_frame` exposes the
committed renderer-neutral split.
When `projection` is omitted, both `QuadricOcclusion3D` and
`QuadricSectionTransition3D` use a true orthographic isometric view.  Its
screen basis is orthonormal, all three world axes have equal projected scale,
and world-z is vertical on screen.  Pass `ParallelView.from_matrix(...)` to
override it for a deliberate general parallel view.

`QuadricSectionTransition3D` is the topology-changing companion to
`QuadricOcclusion3D`.  It consumes a `ScheduledSectionAnimation` and a
normalized progress source, reserves two banks of curve slots once, and uses
the exact analytic event frames to hand off ellipse, parabola, and hyperbola
families.  Cross-fading curves from both banks stay inside one ordinary
quadric visibility solve and one painter graph.  The updater therefore neither
creates nor removes Manim objects, and sampling the same progress gives the
same result in forward or reverse playback.
Its cutting plane is shown and unified by default; use `show_plane=False` only
when a scene intentionally wants the section curve without a displayed plane.

Global ordering accepts a bounded set of pairwise-strictly-separated convex
spheres, capped finite cylinders, and one-nappe cones/frusta.  Intersecting
entities and a real cyclic surface order fail explicitly because
quadric-to-quadric surface-cell splitting is outside the current contract.
That restriction does not apply to the supported one-quadric/one-cutting-plane
compositor described above.  Full details and a minimal example are in
[quadric-occlusion.md](quadric-occlusion.md).

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
tikz-native-project
```

The four Bridge commands support `health`; `tikz-native-project` instead exposes
`build`, `status`, `rebuild`, and `clean`. Bridge run operations accept strict
versioned JSON requests; refer to `tikz_native/schemas/` and the bridge module
constants for the exact request and response contracts.

## Lifecycle rules

- Construct and add the source geometry to the Scene before attaching an
  occlusion controller.
- Use one controller per source figure.
- Put geometry restoration inside the visibility session. Restore visibility
  first, then the geometry rig.
- A failed frame preserves the last good overlay and never silently displays
  all lines as visible.
- The current real-time binding targets Manim's Cairo renderer.

### Unified quadric semantic boundaries

The renderer-neutral boundary sidecar is exported from
`polyhedron_visibility.quadrics`:

- `QuadricBoundarySource`, `QuadricBoundaryVisibilitySpan`,
  `QuadricBoundaryPaintFragment`, and `QuadricBoundaryCompositingFrame`;
- `BoundarySourceKind`, `BoundarySemanticKind`, `BoundaryOcclusionScope`, and
  `BoundaryRenderIntent`;
- `GeneratorBoundarySpec`, `build_surface_boundary_sources`,
  `compute_boundary_visibility`, and `compute_quadric_boundary_compositing`;
- `BoundaryPlaneRelation`, `QuadricBoundarySectionSpan`,
  `QuadricBoundarySectionLimits`, `QUADRIC_BOUNDARY_SECTION_LIMITS`, and
  `compute_boundary_section_spans`.

The boundary painter frame uses
`manim-quadric-boundary-compositing/v2`. The short-lived v1 boundary frame is
superseded rather than maintained as a second runtime path: generated boundary
frames and caches must be rebuilt. Other quadric v1 surface, visibility, and
section schemas remain unchanged. The Manim controller selects the unified path
only when `boundary_visibility_mode="unified"` is supplied.
When a cutting plane is active, semantic boundaries are partitioned at the
actual `PlaneDepthRole` contours before midpoint classification. Exact section
curves use their analytic surface/plane identity and visibility events instead
of inheriting the display mesh's chord count. `boundary_section_limits` places
explicit fixed bounds on role-contour segments and split parameters; exceeding
either bound raises rather than guessing or allocating more objects mid-frame.

`QuadricBoundaryPaintFragment.surface_visibility_kind` records visibility
against the selected quadratic surfaces only.
`effective_visibility_kind` is the result after also considering the finite
section-plane patch. Canonical v2 JSON names these fields
`surfaceVisibilityKind` and `effectiveVisibilityKind`; the ambiguous v1
`visibilityKind` field is not emitted. `plane_occluded` and
`plane_occluder_item_ids` preserve the renderer-neutral evidence when a
fragment projects inside that patch and lies behind it;
`occluder_surface_ids` continues to contain quadratic-surface identities only.
Consequently a true silhouette can remain visible against its owning cone or
cylinder while still becoming hidden behind a cutting plane.
