# Quadratic surfaces and conic-section occlusion

This document defines the implementation boundary for analytic quadratic
surfaces (quadrics) under orthographic or general parallel projection.  The
feature is intentionally a sibling of the existing closed-polyhedron,
open-face, and convex-section solvers.  It reuses their renderer-neutral
kernel layers, but it does not add curved special cases to those models.

## Scope

The public contract targets finite, opaque teaching solids built from:

- spheres;
- right circular cylinders, with explicit axial bounds and optional caps;
- right circular cones, with explicit axial bounds, nappe selection, and
  optional caps;
- infinite mathematical cutting planes whose display patches are described
  separately.

It computes static and animated plane sections, including circles, ellipses,
parabolas, hyperbolas, and degenerate cases.  Semantic segments, circular
arcs, elliptical arcs, and solver-produced conic branches can be partitioned
into visible and hidden parameter intervals.  A physical paint policy removes
hidden fragments; a diagrammatic policy draws them with the configured hidden
stroke style.

Perspective projection, general free-form surfaces, reflective or refractive
materials, and physically accurate transparency are outside this contract.

## Layering

The dependency direction is fixed:

```text
quadrics.contract / algebra / roots / curves
                     |
quadrics.sections / critical / visibility / trace
                     |
shared GeometryContext + topology + VisibilitySpan + ParallelView
                     |
quadrics.compositing
                     |
shared stable painter graph
                     |
quadrics.manim / authoring
                     |
shared ManagedPainterBand + OcclusionStyle
```

Pure geometry modules must not import Manim.  A Manim object is a display and
style target, never the source of analytic geometry.

## Geometry contract

An infinite support surface is represented by a symmetric homogeneous
quadratic form

```text
F(x) = [x, 1]^T Q [x, 1] = 0.
```

Finite teaching solids add explicit trim rules.  For cylinders and cones the
axial interval, nappe selection, and cap policy are therefore part of the
solid contract and are not encoded by `Q`.  This prevents a non-existent
infinite extension from hiding a curve.

A mathematical section plane is also independent of its Manim display patch.
Changing how large the translucent rectangle is drawn must never change the
computed section.

Every frame resolves one `GeometryContext` from the participating surface
scales, trim bounds, and paths.  Root validation, topology partitioning, and
depth comparisons use that same resolved context.

## Section solving

For an orthonormal plane frame

```text
x = p + u*s + v*t,
```

the support conic is the homogeneous two-dimensional form

```text
C = H^T Q H.
```

Classification uses both the rank and signs of the quadratic block and the
rank of the complete conic matrix.  The result distinguishes:

- circle and ellipse;
- parabola;
- hyperbola;
- a point;
- intersecting or parallel lines;
- a repeated line;
- the empty set.

The support-conic type and the finite-section topology are recorded
separately.  A finite cylinder section can, for example, retain only one or
more lateral conic arcs after the axial trim is applied.  End caps participate
in solid containment and ray occlusion; the section trace itself describes the
intersection with the lateral quadric and does not invent cap-boundary
segments.

Branch identifiers are semantic and stable.  Frame-local paint-fragment
identifiers may change when a new critical point appears, but the Manim layer
must not use them as cross-frame object identity.

## Visibility solving

For a curve point `q(t)` and the parallel view direction `d`, each candidate
solid is queried along

```text
x = q(t) + lambda*d.
```

Visibility can change only at analytic events:

- the authored parameter-domain endpoints;
- a tangent ray (a repeated depth root);
- a curve/surface crossing;
- an intersection entering or leaving a finite trim or cap;
- a rational parameter-chart boundary;
- a depth exchange between candidate occluders.

The solver constructs the corresponding low-degree equations, isolates and
validates all real roots, clusters numerically identical events without
discarding their evidence, and then calls the shared topology partitioner.
Only after that partition is complete may one midpoint per open interval be
used to classify depth.  Dense sampling is not a substitute for critical-root
solving.

For a curve authored on an occluding surface, the zero-depth self-hit is
ignored, but another positive-depth hit on the same solid is retained.  This
is what makes the rear half of a sphere section hidden by the front half of
the same sphere.

## Compositing and Manim

Objective visibility and paint policy remain separate.  The visibility frame
always records visible and hidden spans.  In physical mode a hidden fragment
is not painted.  In diagrammatic mode it is painted with the hidden dashed
style above the opaque proxy that would physically hide it.

`depth_aware_diagrammatic` is a third, opt-in policy.  The hidden fragment
remains a dashed teaching aid, but its named occluding surface is painted over
it.  With section compositing, the hidden dash is placed after the between-role
plane and outline and before the front projection sheet:

```text
plane-behind
outline-behind
surface-back
plane-outside
outline-outside
plane-between
outline-between
hidden dashed
surface-front
plane-front
outline-front
visible solid curve
```

The front sheet therefore attenuates the hidden dash when the authored surface
is translucent, so it reads as lying inside or behind the solid instead of
being attached to the screen.  A fully opaque front sheet can cover it
completely; use ordinary `diagrammatic` when the dash must remain equally
strong regardless of surface opacity.

That diagrammatic dashed stroke is a teaching overlay.  It deliberately sits
above the surface and plane painter layers so the learner can see the hidden
construction.  It is not a simulation of light passing through a transparent
material.  Select the physical policy when the hidden curve must contribute no
pixels at all.

All active surface proxies and curve fragments participate in one complete
far-to-near painter graph and one managed z band.  Projected curve/curve
crossings are solved analytically and split the painter relations at the true
crossing parameters.  Painter cycles and missing items are errors; the runtime
does not guess.

Analytic conics are displayed with bounded-error adaptive polylines.
Opaque convex quadrics initially use analytic projected-silhouette fill
proxies rather than a visible mesh, avoiding triangle seams.  Display
approximation never feeds back into geometric visibility.

The runtime preallocates the maximum curve, fragment, polyline-segment, and
dash slots.
An updater prepares and validates a complete frame, snapshots the current
display state, applies it transactionally, and restores the previous frame if
anything fails.  It never creates a Mobject inside an updater.

### Certified boundary-error plane fragments

Projection-outside pieces are physically cut away from the
projection-interior polygon.  Inside that polygon, the finite-surface ray
solver remains authoritative for the `behind`, `between`, and `front`
boundaries, including finite cylinder and cone caps.

Curved role boundaries are represented by a circumscribed tangent envelope.
For every neighboring pair of analytic conic samples, the compositor combines
the endpoint-tangent triangle distance with an analytic second-derivative
interpolation remainder.  This is a conservative upper bound for the
screen-space separation between the emitted envelope and the true boundary;
finite cap boundaries remain exact.

`section_max_screen_error` bounds that separation.  Away from its certified
boundary band, every fragment agrees with the true `PlaneDepthRole`, and no
fragment may span the stable interiors of two different roles.  A fragment
touching the approximated curve may enter either adjacent true region only
inside the certified band; the contract does not claim exact curved geometry.
If the configured subdivision limit cannot prove this bound, frame preparation
fails before any Manim object is changed instead of assigning a mixed polygon
from its centre point.

A tangent-neighborhood repair may leave one role run as several disjoint
positive-winding regions.  The compositor triangulates and certifies each
region independently, then verifies that their combined area still equals the
source run.  Negative-winding contours denote holes and continue to fail
closed rather than being bridged by a guessed triangle.

### Coincident front/back projection sheets

The curved solid is represented in the section painter graph by two complete,
coincident copies of the same projected silhouette: a back alpha sheet and a
front alpha sheet.  They are painter-order surrogates that let the four plane
roles be placed before, between, or after the solid.  They are not tessellated
samples of the physical rear and front curved surfaces and do not encode a
per-pixel curved-surface depth.

For a requested silhouette opacity `a`, each sheet uses

```text
sheet_alpha = 1 - sqrt(1 - a),
```

so Cairo `OVER` compositing of the two coincident sheets restores exactly `a`.
This model preserves the existing opaque-solid visibility contract; it does
not claim physically accurate transparent-surface rendering.

## Dynamic continuity

The stateless solver owns geometric truth.  A separate continuity tracker
matches semantic branches between frames and records topology events.  An
ellipse-to-parabola-to-hyperbola transition is an explicit event, not an
attempt to mutate one arbitrary curve object into another.

A moving point can follow a stable parameter or arc-length fraction while the
topology is unchanged.  Across a topology change it must be defined by an
additional geometric construction, such as an auxiliary plane or line.  If
two successor branches are equally valid and the author did not choose one,
the operation fails explicitly.

The Provider exposes two separate promises.  The existing
`quadric_section_animation_trace_v1` capability covers renderer-neutral event,
lineage, and moving-point evidence.  The new
`quadric_section_topology_transition_manim_v1` capability covers automatic
Manim handoff at those scheduled events.

### Cairo continuity regression

The release regression renders `mainly_behind`, `intersects`, `near_tangent`,
`exact_parabola`, and `mainly_front` with both opaque and translucent fills.
Its fill-only fixtures suppress the authored section-curve and plane-outline
ink.  Eroded role masks then remove the silhouette, true depth-role boundaries,
patch edge, and frame border before any interior pixel is judged.  Within the
remaining safe role interiors, pixels must match the corresponding Cairo
`OVER` stack and no background-coloured seam may remain.  A separate
surface-only exact-parabola frame verifies that the two coincident sheets still
restore the authored silhouette opacity.

The continuous rotating-plane regression samples the Cairo frames on both
sides of the exact parabolic event.  It requires fixed Manim identities,
unchanged relative section-layer order, bounded fragment/ray capacity, no seam
flash, and continuous role masks without a one-frame near-tangent block jump.
An additional real movie test exercises the complete ellipse/parabola/
hyperbola transition through Manim's normal render lifecycle.

## Public workflow

The shortest static section workflow is:

```python
from polyhedron_visibility.quadrics import (
    QuadricOcclusion3D,
    SectionPlane,
    SphereSpec,
    compute_quadric_section,
    section_trace_curves,
)

sphere = SphereSpec("sphere", (0, 0, 0), 2)
plane = SectionPlane("cut", (0.4, 0, 0), (1, 0.3, 0.2))
trace = compute_quadric_section("section", sphere, plane)
curves = section_trace_curves(trace)

controller = QuadricOcclusion3D(
    self,
    surfaces=(sphere,),
    curves=curves,
    section_plane=plane,
    paint_policy="diagrammatic",
).attach()
```

Change `paint_policy` to `"depth_aware_diagrammatic"` to use front-sheet
attenuated hidden dashes.  This option changes painter order only; visibility
intervals, dash geometry, dash pattern, fixed slot identities, and the two
existing policies remain unchanged.

The omitted projection is the true orthographic isometric preset.  It is the
quadric/conic classroom default: equal projected axis scales, no screen shear,
and a vertical world-z cone axis.  Supply a `ParallelView` or a 3-by-3
matrix only when the scene deliberately needs another parallel view.

With `section_plane=plane`, the finite display patch no longer sits at one
manually chosen z-index.  The renderer-neutral section compositor replaces the
smooth projected solid by a far and a near projection sheet, then adaptively
partitions the patch into four depth roles: outside the silhouette, behind the
solid, between the two sheets, and in front of the solid.  These roles, the
plane outline, and all analytic curve fragments enter one deterministic
far-to-near painter graph.  The Cairo binding merges adjacent cells into
continuous compound contours before drawing, so adaptive calculation does not
produce a visible triangle mesh.

The support-quadric restriction supplies analytic boundary candidates and
near-tangent feature detection, but the finite-surface ray solver is the final
role authority.  A small section close to tangency therefore still causes
refinement.  Its curved boundary may be approximated within
`section_max_screen_error`, but emitted fragments are cut along that
approximation and may not span both sides.  The mode currently supports exactly
one finite convex sphere, capped cylinder, or single-nappe cone/frustum and one
non-edge-on cutting plane.  Multiple intersecting quadrics remain a separate
unsupported problem and fail closed.

For a fixed-topology moving plane, pass a callable as `curves`; it may build a
fresh immutable section trace from the current `ValueTracker` value.  Surface
and curve IDs must remain unchanged while one `QuadricOcclusion3D` controller
is attached.

For a rotating plane that changes conic family, use the scheduled transition
controller:

```python
from manim import ValueTracker
from polyhedron_visibility.quadrics import (
    QuadricSectionTransition3D,
    track_scheduled_plane_section,
)

progress = ValueTracker(0.0)
scheduled = track_scheduled_plane_section("section", cone, plane_motion)
controller = QuadricSectionTransition3D(
    self,
    scheduled=scheduled,
    progress=progress,
    transition_fraction=0.055,
).attach()

self.play(progress.animate.set_value(1.0), run_time=6.0)
controller.restore()
```

The controller allocates two bounded banks before playback.  An ordinary
frame uses one bank; near a conic-family event, the live section and the exact
critical section share the two banks and cross-fade with smoothstep weights.
Both banks still enter the same surface-visibility calculation and painter
graph.  Finite trim tangencies that do not change the conic family use an
instantaneous bank handoff instead of holding a numerically repeated tangency
root across several display frames.  The analytic schedule itself always
retains the exact event.

The transition controller displays and unifies the moving cutting plane by
default.  Its plane patch is fitted again from the same immutable surface and
current plane, while the renderer identities remain fixed.  Pass
`show_plane=False` only for a curve-only presentation.

If the schedule was created with an explicit `GeometryContext` or
`coefficient_tolerance`, pass the same values to
`QuadricSectionTransition3D`; this keeps live in-between frames on exactly the
same numerical contract as the scheduled reference frames.

Use `fit_plane_display_patch()` to size the drawn rectangle for an infinite
mathematical plane.  The returned patch is display metadata only: changing its
margin never changes the section or visibility result.

Use `compute_plane_motion_schedule()` and
`track_scheduled_plane_section()` before rendering a rotating-plane motion.
The schedule inserts analytic tangencies, axis-parallel positions, cone
parabolic positions, cone-apex degeneracies, and contact with the finite trim
circles at the ends of a cylinder, cone, or frustum.  It then records stable
branch lineage and explicit topology events.  A moving point may follow a stable
parameter or arc-length fraction; an ambiguous branch handoff requires an
authored auxiliary rule.

For several pairwise-disjoint convex quadrics, call
`compute_global_quadric_frame()`.  It verifies strict 3D separation, determines
whether projected silhouettes overlap, certifies the ray-depth order for every
overlapping pair, and passes the resulting constraints into the same painter
graph.  The supplied scene may contain a bounded mix of spheres, capped finite
cylinders, and one-nappe finite cones/frusta.

`QuadricOcclusion3D` uses this certified global path automatically on every
prepared frame, including frames produced from surface callbacks:

```python
controller = QuadricOcclusion3D(
    self,
    surfaces=surfaces,
    curves=curves,
    projection=view,
).attach()
```

The committed evidence is available as `controller.last_global_frame`.
Additional `surface_constraints` are checked together with the automatic
relations and cannot replace or contradict them.  Set
`surface_order_mode="explicit"` only when deliberately using the legacy manual
constraint path; that mode does not recompute or recertify supplied relations.

## Implemented acceptance boundary

- finite sphere, cylinder, cone/frustum contracts and homogeneous forms;
- circle, ellipse, parabola, hyperbola, and degenerate plane sections;
- exact event equations followed by isolated-root validation and finite trim;
- semantic segments, circular/elliptical arcs, and conic branches;
- opaque-solid visibility in physical and diagrammatic policies;
- analytic projected curve/curve crossings and depth order;
- adaptive polyline display with no display samples fed back into geometry;
- boundary-conforming outside/behind/between/front cutting with
  `section_max_screen_error` limited to boundary-approximation error;
- transaction-safe, fixed-capacity Cairo Manim binding;
- analytic rotating-plane schedules, topology records, and moving-point traces;
- automatic fixed-capacity Manim handoff across ellipse, exact parabola, and
  hyperbola families;
- automatic plane-display-patch fitting;
- global ordering for a bounded set of strictly separated convex quadrics.

The regression suite exercises scales from `1e-6` through `1e6`, large common
world translations, equivalent scaled projection rows, repeated updates, Fade
lifecycle, ten masked five-state Cairo keyframes, a surface-only opacity frame,
and a continuously moving Cairo section near the parabolic event.  Release
validation additionally builds the wheel and sdist, checks their metadata, and
installs the wheel in an isolated environment.

## Deliberate limitations

- The global multi-surface solver requires pairwise strict 3D separation.  It
  rejects touching/intersecting solids and true surface painter cycles rather
  than guessing.  Quadratic surface-cell splitting is not implemented yet.
- `QuadricOcclusion3D` itself has fixed topology while attached.  Use
  `QuadricSectionTransition3D` for scheduled ellipse/parabola/hyperbola family
  changes.  Unscheduled or ambiguous topology changes still fail explicitly.
- Visibility still treats each surface proxy as an opaque convex silhouette.
  The coincident front/back alpha sheets are a painter-order display model;
  transparent curved-surface refraction, reflection, and physically accurate
  alpha blending are outside this module.
- General free-form surfaces and perspective cameras are outside this
  contract.

## Honest numerical boundary

"Exact critical points" means that the correct analytic event equations are
constructed and their real roots are isolated and residual-checked.  Floating
point arithmetic is still used.  If a root cluster, projected detail, or depth
order cannot be separated reliably, the solver raises an explicit error rather
than returning a guessed frame.

Surface filling is necessarily a finite display approximation in Manim.  The
geometric visibility result remains analytic within the resolved numerical
tolerance.
