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

Use `NativeManimRenderer` for 2D, `NativeFixedViewRenderer` when a 3D TikZ
picture should become ordinary 2D Mobjects in its authored projection, and
`NativeManim3DRenderer` when the objects must remain in world space for a
Manim 3D camera:

```python
from tikz_native.manim_renderer import NativeManimRenderer
from tikz_native.fixed_view_renderer import NativeFixedViewRenderer
from tikz_native.manim_renderer_3d import NativeManim3DRenderer

# Compile `picture_2d` and `picture_3d` from the corresponding source files.
figure_2d = NativeManimRenderer(scene_unit_per_cm=1.0).render(picture_2d)
figure_fixed_3d = NativeFixedViewRenderer(scene_unit_per_cm=1.0).render(picture_3d)
figure_world_3d = NativeManim3DRenderer(scene_unit_per_cm=1.0).render(picture_3d)
```

These figures expose an object mapping keyed by stable semantic IDs. Pass a
custom Manim `TexTemplate` to the renderer when the portable Fandol/Latin
Modern defaults are not suitable.

## Semantic parallel camera

`ParallelCameraState` is a renderer-neutral parallel-camera contract. Its
matrix rows are screen-right, screen-up, and positive depth towards the
observer. `target` is the world point placed at `screen_anchor`; `zoom` changes
the scale around that anchor instead of moving the anchor itself.

```python
from tikz_native import CameraPlane, ParallelCameraState

cut_plane = CameraPlane(
    point=(1.0, -0.5, 0.75),
    normal=(1.0, 1.0, 1.0),
    u_axis=(1.0, -1.0, 0.0),
)

front = ParallelCameraState.normal_to_plane(
    cut_plane,
    target=cut_plane.point,
    screen_anchor=(-2.0, 0.5),
    zoom=1.2,
)
oblique = ParallelCameraState.relative_to_plane(
    cut_plane,
    inclination_degrees=55,
    azimuth_degrees=25,
    target=cut_plane.point,
)
edge_on = ParallelCameraState.along_plane(
    cut_plane,
    azimuth_degrees=25,
    target=cut_plane.point,
)
```

`inclination_degrees=0` is normal to the plane; `90` is exactly along the
plane. The latter is still a valid invertible 3D camera state even though that
particular plane has a rank-one screen projection. `side="negative"` selects
the opposite half-space, and positive `roll_degrees` rotates the rendered
screen coordinates counter-clockwise.
`SectionPlane` can be passed directly because the constructors use only its
`point`, `normal`, and `u_axis` attributes; the camera module does not import
the quadric package.

Use the state with `MultiProjectionCamera`:

```python
from tikz_native.camera_3d import MultiProjectionCamera

camera = MultiProjectionCamera()
camera.register_parallel_state("cut-front", front)
camera.set_parallel_state("cut-front")

# In a Scene:
scene.play(
    camera.animate_to_parallel_state(oblique, transition="orbit"),
    run_time=1.5,
)
```

The new transition path uses rotation interpolation plus positive-definite
stretch interpolation, so every intermediate 3-by-3 matrix stays finite,
right-handed, and invertible. `transition="shortest"` is available when the
two orientations are not exactly 180 degrees apart; an exact half-turn must
use the explicit orbit path. Existing `ProjectionPreset`, `set_mode`,
`animate_to`, and `animate_orbit_to` behavior remains available unchanged.

For a repeatable teaching sequence, author named static shots instead of
calling the camera tracker directly:

```python
from tikz_native import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
    play_parallel_camera_shot_sequence,
)

front_shot = ParallelCameraShot.normal_to_plane(
    "front",
    cut_plane,
    target=cut_plane.point,
    screen_anchor=(-1.2, 0.2),
    duration=1.0,
    hold=0.3,
)
side_shot = ParallelCameraShot.along_plane(
    "side",
    cut_plane,
    azimuth_degrees=25,
    target=cut_plane.point,
    duration=1.2,
    cue="exact edge-on",
)
sequence = ParallelCameraShotSequence((front_shot, side_shot))
play_parallel_camera_shot_sequence(scene, sequence)
```

`look_at`, `normal_to_plane`, `relative_to_plane`, and `along_plane` all create
the same immutable `ParallelCameraShot` contract. A shot records its complete
camera state, duration, optional hold/cue, transition mode, and orbit height.
Sequences have unique IDs and serialize as `parallel-shot-sequence/v1`; the
public schema is
`tikz_native/schemas/parallel-shot-sequence-v1.schema.json`.

`fit_points_to_parallel_camera_state()` computes the largest positive zoom that
keeps a fixed point envelope inside an explicit `ParallelCameraSafeFrame`
without moving the authored target or screen anchor. It supports both area and
line-valued projections. A point set whose screen image collapses completely
must provide an explicit fallback zoom instead of receiving a guessed scale.

For a genuinely moving target, first play a static shot to its exact endpoint,
then start `ParallelCameraTargetFollowController`. Its one preallocated updater
changes only the target; `stop()` keeps the latest followed view and
`restore()` returns to the authored endpoint. Playback or provider failure
removes the updater and rolls the camera back. At startup the binding preserves
already-earlier target producers, then places the follow driver immediately
before explicitly marked quadric/composite camera-state consumers. This gives
the deterministic same-frame order `producer -> follow -> occlusion`. A target
provider that depends on a later, unmarked Mobject updater remains outside this
contract and must be reordered by the author.

Manim's inherited camera zoom composes multiplicatively with the state's
`zoom`, while `screen_anchor` remains a final viewport-relative coordinate even
when the inherited `frame_center` is non-zero. The semantic state's `target`
remains an absolute world point.

The Manim quadric controllers accept the same complete state through their
existing `projection=` argument. Pass the camera itself from a callback when
the view is animated, so direction, target, viewport anchor, and both zoom
layers are sampled together exactly once per frame:

```python
section = QuadricOcclusion3D(
    scene,
    surfaces=(cone,),
    curves=section_curves,
    section_id=section_id,
    section_plane=cut_plane,
    projection=lambda active_scene: active_scene.camera,
)
```

Passing a static `ParallelCameraState` is also supported. Existing 3-by-3
matrices and `ParallelView` values keep their original raw screen-coordinate
semantics; they do not inherit target, anchor, frame-center, or Manim zoom.
`display_offset=(x, y)` remains an extra final screen translation and composes
with, rather than replaces, the semantic camera anchor. Geometry and occlusion
still consume only the resolved linear view; the affine target/anchor shift is
applied at the fixed-frame Manim boundary and therefore cannot change depth
evidence.

For a finite `PlaneDisplayPatchSpec`, an exact edge-on view now has an explicit
one-dimensional display contract. Its fill disappears, it no longer occludes
other boundaries as an area, and the compositor retains one certified finite
near-side outline chain without duplicate strokes. The existing fixed section
slots are reused through `AREA -> LINE -> AREA`. Section sources are certified
from their complete analytic geometry rather than their IDs. Circle, ellipse,
parabola, and hyperbola section members which retain a line are repainted with
hidden intervals before visible intervals; a finite cap chord which becomes one
screen point keeps its source identity and fixed slot but emits no fake stroke.
This exception is deliberately narrow: external curves, other surfaces, and
independently coincident or otherwise uncertifiable arrangements still fail
explicitly.

The low-level controller only grants that section-family provenance when
`section_id=` and `section_plane=` are both supplied. Passing a plane without a
section ID remains backward compatible for ordinary free curves and the finite
plane patch, but those curves are not silently reclassified as a section.

See the runnable
[parallel-camera view example](../examples/parallel_camera_views/README.md) and
the three
[semantic shot acceptance scenes](../examples/parallel_camera_shots/README.md).
For one source-authoritative sequence that coordinates the camera, moving
section, topology banks, finite plane patch, semantic display, and real
`QuadricOcclusion3D` painter order, see the
[parallel camera/section sequence contract](parallel-camera-section-sequence.md)
and its three
[Cairo acceptance scenes](../examples/parallel_camera_section_rig_demo.py).

The Rig binding uses certified surface silhouettes and finite cap rims by
default.  Optional cone generators can be reserved at compile time.  An
isolated tangent section uses a preallocated point Mobject; point activation
does not allocate during playback and never substitutes a short line.  The
older `surface_boundary_mode="legacy"` path remains an explicit display-only
fallback and does not become occlusion evidence.

Renderer-level affine terms are authored with `ParallelScreenTransform` and
passed as `screen_transforms=`.  The semantic camera, positive inherited zoom,
XY frame center, and final display offset are committed by one
`ParallelViewportState` transaction.  This supports a non-identity first frame
and later motion without double-applying the anchor or offset.

Dynamic semantic compositing has three independent axes:

```python
from polyhedron_visibility.quadrics import (
    SectionCompositingAxes,
    SectionCompositingInstruction,
    SectionCompositingOverride,
    compile_section_compositing,
)

instruction = SectionCompositingInstruction.for_catalog(
    catalog,
    defaults=SectionCompositingAxes(
        display_opacity=1.0,
        occlusion_participation="certified",
        depth_presentation="diagrammatic",
    ),
    overrides=(
        SectionCompositingOverride.for_slot(
            surface_fill_slot_id,
            display_opacity=0.0,  # invisible but still an occluder
        ),
    ),
)
compositing_frame = compile_section_compositing(catalog, instruction)
```

`display_opacity` changes only painted alpha;
`occlusion_participation="paint-only"` explicitly excludes a supported
surface fill from visibility work; and `depth_presentation` selects physical,
diagrammatic, or depth-aware diagrammatic painting.  No axis is inferred from
another.  Opacity-only frames use the draw-only path, while participation and
depth-policy changes force a fresh certified geometry frame.

For several solids, compile each local Rig against the same Scene, time grid,
camera, and viewport without attaching it, then aggregate them:

```python
from tikz_native import compile_global_parallel_rig

global_binding = compile_global_parallel_rig((left_rig, right_rig))
global_binding.attach()
coordinator = global_binding.build_coordinator(scene.camera)
for frame in global_binding.sequence.frames:
    coordinator.update(frame)
global_binding.restore()
```

The aggregate owns one global `QuadricOcclusion3D` and one painter band; it is
not a z-ordering wrapper around separately painted Rigs.  Curves, isolated
points, silhouettes, rims, and generators from one Rig can be occluded by any
certified surface in another Rig.  Global v1 rejects visible plane patches,
unequal frame grids/viewports, identity collisions, and intersecting or
otherwise uncertifiable solids before Scene ownership.

The three short Cairo scenes in
[parallel_camera_advanced_compositor_demo.py](../examples/parallel_camera_advanced_compositor_demo.py)
exercise a non-identity viewport with a tangent point, the three compositing
axes, and cross-Rig global occlusion.

## Explicit TikZ planar curves in 3D

The restricted compiler does not infer a spatial supporting plane from a
screen-space circle or ellipse. Declare one static plane from named 3D points,
then author the curve in its local coordinates:

```tex
\begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{base-plane}{O/U/V};
  \DrawSpaceCircle[draw=red,line width=1pt]
    {circle-a}{base-plane}{0,0}{1.5};
  \DrawSpaceEllipse[draw=blue]
    {ellipse-a}{base-plane}{0.5,-0.25}{2}{1};
\end{tikzpicture}
```

`O` is the origin of `base-plane`; `O -> U` fixes its local-u direction and
the curve's parameter phase; non-collinear `V` fixes the positive local-v
side. The lengths of `O -> U` and `O -> V` are orientation evidence, not scale
factors: local centers and semi-axis lengths use world-coordinate units. A
circle takes `(curve ID, plane ID, local center, radius)`. An ellipse
takes `(curve ID, plane ID, local center, semi-u, semi-v)`. The compiler stores
the canonical `PlanarFrame3D` and `Circle3DSpec` / `Ellipse3DSpec` evidence in
the object geometry rather than reducing the curve to sampled screen points.

This v1 frontend is static-safe and supports one complete revolution with one
visible solid stroke. It accepts draw color, positive line width,
draw/overall opacity, line cap, and line join. It explicitly rejects fill,
fill opacity, dashes, arrow tips, additional canvas transforms, partial arcs,
and animated O/U/V authorship. The present embedded geometry-driver runtime
also rejects a picture containing these curve kinds, even if the registered
plane is fixed and unrelated to the active hinge; this rejection happens in
rig analysis instead of failing later during playback. Ordinary
two-dimensional circle and ellipse paths remain supported. An ordinary
three-dimensional circle/ellipse path or named path fails closed because it
has no explicit supporting plane; the physical `circle (1pt)` point-marker
form remains a dot.

`NativeFixedViewRenderer` directly projects both authored semi-axis vectors.
For rank two it applies their full affine screen basis to a unit circle; for an
exact edge-on rank-one projection it emits exactly one finite `Line` between
the curve extrema. It neither inverts a nearly singular ellipse basis nor
extends the segment to an infinite chord. `NativeManim3DRenderer` instead
applies the authored world-space center and two semi-axis vectors to a unit
circle, retaining the actual supporting plane and the same Mobject while the
camera changes. Neither path turns static O/U/V evidence into animated curve
geometry. The result is a standalone curve object; it is not a planar disk and
is not automatically registered with quadric visibility or unified
compositing.

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

- `SphereSpec`, `CylinderSpec`, `ConeSpec`, `ConeModel`,
  `CircularTrimRimSpec`, and `SectionPlane`;
- `PlanarFrame3D`, `PlanarPoint3D`, `Circle3DSpec`, `Ellipse3DSpec`,
  and `PlanarCurveScene3D`;
- `build_cone_projection_layers()`, `ConeProjectionLayers`, and
  `ConeProjectionSheet`;
- `compute_quadric_section()`, `section_trace_curves()`,
  `compute_quadric_section_boundary_curves()`, and
  `section_cap_chord_curve_ids()`;
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
- `QuadricSection3D`, the low-level finite-section authoring facade;
- `QuadricSectionRig`, `SectionState`, and `QuadricSectionAction`, the natural
  fixed-topology mathematical-action layer;
- `compute_dandelin_construction()`, `DandelinConstruction3D`, and the static
  diagrammatic `DandelinSection3D` teaching facade;
- `compute_dandelin_visibility_frame()`, `DandelinVisibilityFrame`, and
  `DandelinTangentContactEvidence` for fixed-camera Dandelin hidden lines;
- `compute_dandelin_surface_layer_frame()`, `DandelinSurfaceLayerFrame`, and
  `DandelinEqualDepthContact` for certified teaching-transparent cone, sphere,
  cutting-plane, and tangent-seam painter order;
- `QUADRIC_PREVIEW_PROFILE`, `QUADRIC_FINAL_PROFILE`,
  and `QuadricCapacityPlanner`;
- `QuadricOcclusion3D`, `QuadricManimStyle`, `QuadricBoundaryStyle`, and
  `QuadricManimLimits`.

### Authored planar circles and ellipses in 3D

`PlanarFrame3D(frame_id, point, normal, u_axis=None)` is the
renderer-neutral supporting-plane contract for an authored 3D circle or
ellipse. It normalizes a stable right-handed in-plane basis; an explicit
`u_axis` fixes both the curve orientation and parameter phase instead of
leaving those choices to a display helper. The automatic basis is deterministic
for one fixed normal, but it is not promised to vary continuously while the
normal moves across a world-axis tie. Animated frames that need continuous
curve phase must provide a continuous `u_axis` explicitly.
An authored `u_axis` whose in-plane projection is no larger than
`sqrt(machine epsilon)` relative to the supplied direction is rejected: at
that scale its phase is no longer numerically trustworthy.

`frame.certified_point((u, v))` returns a `PlanarPoint3D` carrying both the
exact supporting-frame contract and its local coordinates. Use this form when
the center is produced in plane coordinates, especially at large world or
local scales. It avoids trying to infer lost authorship from a rounded world
point. A certified point from a different frame is rejected.

`Circle3DSpec(curve_id, frame, center, radius, domain=...)` and
`Ellipse3DSpec(curve_id, frame, center, semi_u, semi_v, domain=...)` require
positive radii or semi-axis lengths and accept at most one revolution of a
`ParameterInterval`. `center` may be a certified `PlanarPoint3D`. A raw
world-space center instead goes through strict, exact-residual plane-membership
certification based only on the coordinate quantization bound; curve size does
not silently widen that bound. An accepted raw center remains the exact world
point supplied by the caller; it is never snapped to a reconstructed point on
the plane. `from_plane_coordinates(...)` is the compact constructor for the
certified path. Both forms record canonical
`centerCoordinates` in the serialized contract. They do not introduce another
analytic-curve runtime:
`lower_to_analytic_curve()` returns the existing `CircleArcCurve` or
`EllipseArcCurve`, preserving the authored curve ID, supporting frame, and
parameter domain.

Finite input alone is not sufficient at the extreme limits of floating-point
arithmetic. Construction fails explicitly when the existing analytic runtime
cannot certify finite axes, normals, points, and tangents at the requested
scale. It also verifies that the four cardinal semi-axis displacements remain
representable after translation to the world-space center. A radius-one circle
at a center whose floating-point spacing is already much larger than one fails
instead of silently collapsing into a point or line.
For both center-authoring channels, the forward plane-local-to-world embedding
error must be no greater than `sqrt(machine epsilon)` times the circle radius,
or the ellipse's smaller semi-axis. This relative error budget permits a large
curve to remain usable at a large translation while requiring a small feature
at the same translation to fail explicitly.

`PlanarFrame3D.to_dict()` records canonical `normal` / `uAxis` / `vAxis`
values together with scale-independent `normalSeed` / `uAxisSeed` direction
evidence. `PlanarFrame3D.from_dict()` recomputes the basis from those seeds and
requires an exact match; it never decides that arbitrary nearly orthogonal
payload values are already certified. The point and curve contracts provide
matching `from_dict()` methods. `PlanarCurveScene3D.from_dict()` and
`from_json()` rebuild a complete registry, and a canonical scene JSON payload
must round-trip byte-for-byte. This evidence also survives standard immutable
`dataclasses.replace(...)` reconstruction. Replacing `normal` or `u_axis`
reauthors fresh direction seeds; replacing a curve center with a certified
`PlanarPoint3D` makes that new point authoritative over the old derived local
coordinate field. A replacement that supplies a raw center must also reset
`center_coordinates=None` so the coordinates are inferred and certified again.

`PlanarCurveScene3D(frames, curves)` is a deterministic renderer-neutral
registry. Frame IDs and curve IDs must be unique and globally distinct, and
every curve must reference the exact registered frame with the same ID.
`lower_to_analytic_curves()` returns the lowered curves in stable `curve_id`
order; `to_dict()` and `canonical_json()` expose the corresponding serializable
contract.

Finite cylinder/cone terminal circles use the same contract. A
`PlanarCapSpec` or `CircularTrimRimSpec` exposes its normalized
`planar_frame`, and `boundary_circle(curve_id)` returns a `Circle3DSpec` with
the existing cap/rim identity and parameter phase. Surface-boundary
compositing immediately lowers that value to the existing `CircleArcCurve`, so
visibility, painter ordering, and Manim fixed slots still have one analytic
curve runtime. A trim rim remains only a boundary circle; this adapter does not
invent a closing disk for an open shell.

These five types remain renderer-neutral geometry contracts: they do not
themselves parse, style, attach, or animate Manim objects. The controlled TikZ
frontend described in
[Explicit TikZ planar curves in 3D](#explicit-tikz-planar-curves-in-3d) now
maps its three static commands into those contracts and consumes them through
the fixed-view or world-space renderer. This is a deliberately narrow adapter,
not a general `PlanarCurveScene3D` Manim authoring facade and not support for
arbitrary TikZ circle, ellipse, fill, dash, or arc syntax. Advanced callers may
still pass lowered `CircleArcCurve` / `EllipseArcCurve` values to the existing
visibility or Manim layers, but that manual composition is separate from the
static TikZ adapter.

### Certified Dandelin spheres and teaching overlays

`compute_dandelin_construction(construction_id, cone, plane)` derives the
finite Dandelin spheres, their focus points on the section plane, their real
cone-contact `Circle3DSpec` values, and their directrices. It first reuses the
ordinary analytic section solver, then requires every complete sphere extent
to fit strictly inside the authored `ConeSpec.axial_range`. A sphere that
touches or crosses a terminal plane, a cap-chord section, a plane through the
apex, or another degenerate/incomplete construction fails explicitly. The
result is a renderer-neutral `DandelinConstruction3D` with deterministic
canonical JSON.

```python
from math import pi
from polyhedron_visibility.quadrics import (
    ConeModel, ConeSpec, SectionPlane, compute_dandelin_construction,
)

cone = ConeSpec(
    "cone", (0, 0, 0), (0, 0, 1), pi / 6, (0, 9),
    model=ConeModel.OPEN_SINGLE,
)
plane = SectionPlane(
    "cut", (0, 0, 2), (0.6, 0, 0.8), u_axis=(0, 1, 0),
)
construction = compute_dandelin_construction("ellipse-proof", cone, plane)
```

The v1 support rows are deliberately finite and narrow: circle/ellipse use one
single nappe and require two finite spheres; an exact-parabola `OPEN_SINGLE`
uses its one finite sphere and does not create an infinity placeholder; a
complete hyperbola requires one finite sphere on each nappe of an
`OPEN_DOUBLE`. `ANALYTIC_DOUBLE`, a single-nappe hyperbola, an ellipse or
parabola authored on `OPEN_DOUBLE`, and any section with a real cap chord are
rejected. `CLOSED_SINGLE` circle/ellipse works only when it remains a complete
pure lateral section and both spheres fit before the real terminal cap.

`build_dandelin_meridian_diagram(construction)` derives the genuine axial
section: the sphere circles are true great-circle sections and their contacts
with the finite cone generators and section line are certified.
`build_dandelin_section_plane_diagram(construction)` derives the cutting-plane
conic, foci, directrices, and sphere-plane tangencies. It intentionally
contains no sphere-circle field because the sphere centres generally do not
lie in the cutting plane. Both diagrams retain the authoritative construction,
rederive all fields during validation, and use view-local object IDs plus a
shared `sourceRef` for cross-view identity.

`DandelinSection3D(scene, cone=..., plane=...,
construction_id=...).attach()` is the matching static Cairo authoring facade.
It retains the existing cone/section compositor, then draws the certified
spheres, contact circles, clipped directrices, and focus dots in a separate
top teaching band. This is intentionally a diagrammatic overlay:
`visibility_authoritative` is `False`, `overlay_mode` is `"diagrammatic"`, and
the tangent cone/spheres have **not** been physically multi-surface
composited. `build_dandelin_teaching_overlay()` likewise rejects `physical`
and `depth_aware_diagrammatic` mode requests.

The facade accepts immutable cone/plane geometry and one complete immutable
parallel-camera frame. A `ParallelCameraState` freezes matrix, target,
screen-anchor, zoom, and current viewport translation for the section,
overlay, and focus dots together; projection callbacks are rejected without
execution. On `attach()` the facade lazily reserves one non-overlapping Scene
painter band, derives separate section/overlay/focus sub-bands, and releases
the aggregate on `restore()`. If cleanup cannot prove that Scene and
fixed-frame ownership are gone, it retains both the runtime references and
reservation for an explicit retry instead of creating orphan display objects.
Display-slot identities therefore exist only while attached. Perspective,
OpenGL, and dynamic family transitions remain outside v1. Full formulas, the
support/rejection matrix, and the finite-fit rule are in
[Dandelin spheres v1](dandelin-spheres-v1.md).

The static TikZ-facing path declares the source relation rather than inferring
it from arbitrary circles:

```tex
\DeclareSpaceRightCone{cone}{A/Z/R}{30}{0/9}{open_single};
\DeclareSpacePlane{cut}{O/U/V};
\DeclareDandelinConstruction{dan}{cone}{cut};
\DrawDandelinDiagram[
  view=spatial,
  preset=classroom,
  mode=depth_aware_teaching_transparent
]{dan};
```

`view` may be `spatial`, `meridian`, or `section-plane`. The resulting
`dandelin_diagram` is a fixed-view object whose nested semantic items retain
view-local IDs and cross-view `sourceRef` values. The default
`mode="diagrammatic"` preserves the original authored ordering.
`mode="depth_aware_diagrammatic"` is valid only for the spatial view and calls
`compute_dandelin_visibility_frame()` with the same frozen parallel camera.
The returned analytic partitions drive solid visible strokes and dashed hidden
strokes for cone boundaries, sphere silhouettes, contact circles, the section
curve, and optional directrices. Tangent-contact evidence identifies each
sphere, its exact cone nappe, and its contact circle before any fragments are
painted.

`mode="depth_aware_teaching_transparent"` is also spatial-only. It keeps the
same certified curve partition and additionally certifies the painter order of
the teaching-transparent cone sheets, sphere fills, and cutting-plane
fragments. Accordingly, both depth-aware modes have
`curveVisibilityAuthoritative=true`, while only the latter has
`surfaceLayeringAuthoritative=true`. Neither mode claims opaque physical
surface visibility: `surfaceVisibilityAuthoritative`,
`physicalSurfaceVisibilityAuthoritative`, and the aggregate
`visibilityAuthoritative` all remain false. Full physical mode, mixed drawable
objects, a fake section-plane sphere circle, or a teaching-transparent request
with `show-contact-circles=false` are rejected explicitly; the contact strokes
own the certified equal-depth seams. Motion, camera shots, and source-v3
generation are also rejected. The complete example is in
[`examples/tikz_dandelin_views`](../examples/tikz_dandelin_views/README.md).

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
For ordinary section authoring, prefer `QuadricSection3D(surface=...,
section_id=..., plane=..., render_profile="preview")`. It computes the
complete finite section boundary,
reserves every potential cap-chord identity, and passes the same current plane
and active curves to the existing unified `QuadricOcclusion3D`. A plane
callback supports fixed-topology motion. For a scheduled ellipse/parabola/
hyperbola change, pass `scheduled=track_scheduled_plane_section(...)` and
`progress=...`; the facade delegates to the existing
`QuadricSectionTransition3D`. `draw_section_boundary=False` is the explicit
opt-out that retains plane/surface partitioning without adding true section
ink. This ordinary facade still rejects `OPEN_DOUBLE` and direct
`ANALYTIC_DOUBLE` rendering at construction rather than after Manim slot
allocation. Use the dedicated composite facade for the former. Neither mode
creates a second renderer or a second rollback path.

When the scene should read as a mathematical storyboard, construct
`QuadricSectionRig` with the same `surface`, `section_id`, and initial `plane`,
enter `session()`, then pass `animate_plane_shift()`,
`animate_plane_rotation()`, or an unambiguous parallel `animate_plane_to()` to
`Scene.play`. The rig compiles exact critical positions before returning the
animation, maps incidental raw conic labels to stable tracked slots, and joins
its immutable state to the controller's frame transaction. Numeric painter
bands are allocated automatically per Scene; an exact explicit override is
still available. A path needing topology-transition banks fails before
playback rather than relying on finite samples. Phase 1 resolves and freezes
one complete non-callable parallel projection when the rig is constructed,
including a semantic state's target, screen anchor, and zoom; projection
callbacks remain unsupported. Whenever either the plane or its section boundary
is visible, it analytically certifies the complete
axis-angle path as an AREA projection and rejects an unsafe action before
playback without changing existing Scene ownership or its painter-band
reservation. The lower-level facade and semantic-camera APIs still support
certified edge-on LINE frames; this narrower Rig phase does not animate through
that display-rank handoff. Only setting both `show_plane=False` and
`draw_section_boundary=False` opts out because no rank-sensitive section ink is
then displayed.
`reverse_rate_function=True` is explicitly unsupported, whether supplied to
the action or through `Scene.play`. Endpoint-preserving non-monotone rate
functions are allowed because the whole progress interval is certified; each
evaluated value must still be finite and remain in `[0, 1]`.

`render_profile="preview"` and `"final"` expand to the matching style,
approximation, and fixed-capacity defaults inside this facade. Explicit
capacity, error, and semantic-boundary values override profile defaults.
Output resolution remains an
explicit Manim CLI or `tempconfig` setting. For offline planning,
`QuadricCapacityPlanner.scan(scene_factory, frames=...)` calls the factory once,
drives one normalized tracker through the same fixed slots, and returns a
human-readable `summary()` plus immutable `recommended_limits`. The legacy
instance-level `planner.scan(progresses)` and analytic `scan_schedule()` forms
remain available.

Scheduled rigid motion may opt into `use_plane_patch_envelope=True`. This
precomputes one fixed, display-only patch from a certified finite-solid radius
about the authored rotation pivot and reuses it at every progress value. The
option is rejected in static/callback mode, where no complete motion range is
available. `fit_plane_motion_display_patch_envelope()` exposes the same
renderer-neutral evidence to advanced callers.

`CompositeQuadricSection3D(surface=..., section_id=..., plane=...)` accepts one
finite `OPEN_DOUBLE`. It expands the shell into its canonical negative and
positive nappes, calls the existing local section solver for each, paints the
shared plane once, and merges both local painter frames. Two child slot banks
remain fixed, while `branch_lineage` links nappe-owned physical curve IDs to
the common mathematical conic branches. A plane callback may move within a
fixed curve-identity topology. A scheduled composite ellipse/parabola/
hyperbola handoff is not yet available. The current coordinator also requires
the complete projected contact set to be zero-dimensional and contained inside
the shared-apex tolerance. It records `contact_dimension`, `contact_extent`,
the maximum distance from the apex, and the retained contact points. A remote
point, nonzero coincident segment, or positive-area overlap fails
transactionally instead of guessing an interleaved order.

The composite frame exposes the same explicit `projection_kind` and
`patch_projection` evidence as one local frame. In an exact side view both
children must certify the same finite `LINE` endpoints. The coordinator then
omits every plane fill, merges their one-dimensional depth partitions into one
complete near-side outline, and gives each nappe its own authenticated section
source group. `AREA -> LINE -> AREA` reuses the original plane, surface, and
curve slots; a failed critical frame leaves the previous display untouched.

A plane callback may move a section only while its lateral conic topology and
curve identities stay fixed. Cap chords may activate or disappear because all
authored cap identities are reserved independently. An empty/non-empty change,
a component-count change, or an ellipse/parabola/hyperbola family change must
use a scheduled transition. If such a change is sent through the callback
mode, the newly named curve is rejected as unallocated and the last committed
frame is restored.

At the lower level, pass `section_plane=plane` to place one finite display
patch, the two projected surface sheets, the section curves, and every
visible/hidden curve fragment in one painter graph.  The patch is adaptively
split into regions behind the
solid, between its far and near sheets, in front of the solid, or outside its
projection.  The Manim layer merges those cells back into continuous compound
contours, so the geometric subdivision does not leave triangle seams.  No
Mobject is created during an update, and `last_section_frame` exposes the
committed renderer-neutral split.
`section_plane` partitions and paints the finite display patch but deliberately
does not invent section ink. Pass explicit curves from
`compute_quadric_section_boundary_curves()` when the complete finite section
boundary should be drawn. Filled end caps contribute stable `SegmentCurve`
chords; open cone trim rims do not. For a moving cut, add the potential IDs
from `section_cap_chord_curve_ids()` to `allocated_curve_ids` so a chord can
appear and disappear without changing Mobject identity. A cap chord is only
accepted when both endpoints match open endpoints of the lateral trace; an
unresolved near-parallel cut or a lateral/cap tolerance mismatch raises an
explicit geometry error rather than guessing a boundary.
When `projection` is omitted, both `QuadricOcclusion3D` and
`QuadricSectionTransition3D` use a true orthographic isometric view.  Its
screen basis is orthonormal, all three world axes have equal projected scale,
and world-z is vertical on screen.  Pass `ParallelView.from_matrix(...)` to
override it for a deliberate general parallel view.

`ConeSpec(model=...)` distinguishes the finite teaching object instead of
inferring a cap from a silhouette. `CLOSED_SINGLE` contains one lateral
surface and one planar base; `OPEN_SINGLE` contains the lateral surface and
one trim rim; `OPEN_DOUBLE` contains two finite open nappes, two trim rims,
and one shared apex. The double shell is expanded once into stable component
IDs, so updates do not replace Mobjects. `ANALYTIC_DOUBLE` is a compatibility
support for exact section mathematics and is not directly renderable. No
public model silently represents an infinite cone.

Set `QuadricManimStyle.cone_lateral_fill_colors` and/or
`cone_cap_fill_colors` to opt into fixed component-aware cone shading. The
renderer-neutral projection layer keeps the cap and lateral masks separate;
an open mouth therefore receives one translucent lateral sheet instead of a
fake base disk. `cone_lateral_sheen_direction` and
`cone_cap_sheen_direction` are independent because a side highlight and a
planar-base highlight need not point the same way. All component slots are
preallocated and included in transactional rollback. A two-terminal frustum
keeps the lateral sheet and both real cap disks as separate depth components;
the two cap components use the authored cap palette and may exchange far/near
depth slots as the view crosses a side view. An exact edge-on cap has no
certifiable display area, so its fill path is empty while its real terminal-rim
boundary remains available. The implementation never invents a positive-area
cap to hide that degeneracy.

`QuadricSectionTransition3D` is the topology-changing companion to
`QuadricOcclusion3D`.  It consumes a `ScheduledSectionAnimation` and a
normalized progress source, reserves two banks of curve slots once, and uses
the exact analytic event frames to hand off ellipse, parabola, and hyperbola
families.  Cross-fading curves from both banks stay inside one ordinary
quadric visibility solve and one painter graph.  The updater therefore neither
creates nor removes Manim objects, and sampling the same progress gives the
same result in forward or reverse playback. Filled end-cap chords do not need a
topology bank: each cap role has one stable slot, is recomputed from the actual
current plane, and remains a certified part of the complete finite boundary
while the lateral conics cross-fade. The first fixed-topology Rig slice still
rejects cap-chord activation changes; they remain authored through this
scheduled transition layer.

Its cutting plane is shown and unified by default. `show_plane=False` means
more than hiding the rectangle: it removes the plane fill, outline, depth
partition, and plane/curve occlusion from the painter graph, so
`last_section_frame` is `None`. Use it only for an intentionally curve-only
presentation.

Global ordering accepts a bounded set of pairwise-strictly-separated convex
spheres, capped finite cylinders, and one-nappe cones/frusta. The two stable
components of one `OPEN_DOUBLE` may share their apex when their projected
interiors do not overlap; a view requiring interleaved multi-sheet order fails
explicitly. Intersecting entities and a real cyclic surface order fail because
quadric-to-quadric surface-cell splitting is outside the current contract.
That restriction does not apply to the supported one-quadric/one-cutting-plane
compositor described above. It supports `OPEN_SINGLE` directly. A whole
`OPEN_DOUBLE` remains outside that one-surface compositor, but its two
canonical nappes may be coordinated by `CompositeQuadricSection3D` under the
certified apex-only contact constraint.
Full details and a minimal example are in
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
- `BoundaryScreenProjectionDimension`,
  `QuadricBoundarySectionSourceProjection`,
  `QuadricRankOneSectionSourceGroup`, and
  `certify_rank_one_section_boundary_sources`;
- `GeneratorBoundarySpec`, `build_surface_boundary_sources`,
  `compute_boundary_visibility`, and `compute_quadric_boundary_compositing`;
- `BoundaryPlaneRelation`, `QuadricBoundarySectionSpan`,
  `QuadricBoundarySectionLimits`, `QUADRIC_BOUNDARY_SECTION_LIMITS`, and
  `compute_boundary_section_spans`.

The boundary painter frame uses
`manim-quadric-boundary-compositing/v3`. The v3 payload adds explicit
surface/plane provenance on section sources and serializes the optional
rank-one source-group certificate, including its finite line, projection
dimension, screen covector, and tolerance. Earlier boundary frames are
superseded rather than maintained as parallel runtime paths: generated boundary
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
