# Quadratic surfaces and conic-section occlusion

This document defines the implementation boundary for analytic quadratic
surfaces (quadrics) under orthographic or general parallel projection.  The
feature is intentionally a sibling of the existing closed-polyhedron,
open-face, and convex-section solvers.  It reuses their renderer-neutral
kernel layers, but it does not add curved special cases to those models.

The release boundary for finite-cone sections is frozen separately in the
[finite-cone section v1 support contract](quadric-section-v1-contract.md).
That semantic matrix and its versioned
[release manifest](../release/quadric-section-v1-release-manifest.json) are
authoritative when a broad description in this implementation guide could be
read as supporting more than the tested v1 combinations.

## Scope

The public contract targets finite, opaque teaching solids built from:

- spheres;
- right circular cylinders, with explicit axial bounds and optional caps;
- right circular cones, with explicit axial bounds and one of three finite
  teaching models: a closed single cone, an open single shell, or an open
  double shell;
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

The local cutting-plane compositor has a narrower v1 release boundary: one
finite convex surface, one cutting plane whose screen projection has
two-dimensional area, and parallel projection. A separate
`CompositeQuadricSection3D` coordinator may apply that same local solver to the
two canonical nappes of one `OPEN_DOUBLE`, then merge their disjoint projected
interiors around the certified shared apex and paint the common plane once.
Certification retains point, segment, and area intersections instead of
discarding degenerate contact: only a zero-dimensional contact set contained
inside the shared-apex tolerance succeeds.
This does not create a general multi-surface plane arrangement. Multiple
strictly disjoint quadrics may still use the global occlusion graph; multiple
intersecting quadrics do not gain a shared local plane arrangement. The Manim
production binding explicitly accepts Cairo and rejects OpenGL.

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

### Finite cone models

`ConeSpec.model` makes the authored object explicit:

- `ConeModel.CLOSED_SINGLE` is one finite nappe with its lateral surface and
  one non-degenerate planar base. The base circle is a cap rim.
- `ConeModel.OPEN_SINGLE` is one finite nappe with no planar base. Its
  non-degenerate terminal circle is a trim rim: it is real boundary ink, but
  it contributes no disk to volume membership, ray hits, or section depth.
- `ConeModel.OPEN_DOUBLE` is two finite open nappes sharing one apex. The
  renderer expands it once into stable `:nappe:negative` and
  `:nappe:positive` components; each has one trim rim and no planar cap.

There is no infinite renderable cone model. The compatibility-only
`ConeModel.ANALYTIC_DOUBLE` retains the historical finite cross-apex support
used by exact conic-section calculations, but it fails if passed directly to
the renderer. Omitting `model` preserves old construction rules: a one-sided
axial range means `CLOSED_SINGLE`, while a range crossing the apex means
`ANALYTIC_DOUBLE`. New authoring code should always state the model.

Open shells deliberately have no `contains()` volume relation. Calling it is
an error rather than an implicit claim that the shell is a solid. Their
lateral ray intersections still participate in curve and boundary occlusion.
The open double model is not a closed double-cone solid and does not invent an
apex cap or two terminal disks.

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
segments.  Use `compute_quadric_section_boundary_curves()` when the displayed
ink must be the complete one-dimensional boundary of the finite entity's
section.  It adapts the lateral trace and adds every non-degenerate
plane/end-cap chord as a `SegmentCurve`.  A closed cone can therefore change
from one closed lateral conic to a lateral arc plus a stable cap chord, while
an open cone shell keeps only the lateral arc because it has no filled cap.
Tangency does not create a zero-length placeholder, and a plane coincident
with a cap does not duplicate the lateral rim. The compatibility-only
`ANALYTIC_DOUBLE` remains available through `compute_quadric_section()` but
fails explicitly in the finite display-boundary helper because it has no
directly renderable cap model. Every active chord must join two certified open
endpoints of the clipped lateral trace. If cap and lateral clipping disagree
inside the configured numerical resolution, the helper fails explicitly
instead of returning a closed conic with a spurious interior chord.

For an open single shell, local cutting-plane compositing classifies the plane
against the lateral ray intersections only. It uses adaptive certified cells
because the region inside the projected mouth may have a different depth role
from a closed cone's filled base. A closed single cone retains the existing
exact boundary-conforming solid partition.

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
remains a dashed teaching aid.  Certified farther surfaces are painted first,
then the dash, then every named occluding surface.  Disjoint surfaces receive
no invented relation, and the deterministic identity tie break is never used
as geometric depth evidence.  With section compositing, the hidden dash is
placed after the between-role plane and outline and before the front projection
sheet:

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

The runtime preallocates the maximum curve and fragment slots. Each fragment
owns one solid `VMobject` and one dashed `VMobject`; every active dash is an
independent open subpath inside the latter. `max_dashes_per_fragment` remains a
hard numerical capacity checked before display mutation, but it no longer
allocates one hidden Mobject per possible dash. One boundary source therefore
owns `1 + 3 * max_fragments_per_curve` family members, independent of the dash
capacity.
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
    QuadricSection3D,
    SectionPlane,
    SphereSpec,
)

sphere = SphereSpec("sphere", (0, 0, 0), 2)
plane = SectionPlane("cut", (0.4, 0, 0), (1, 0.3, 0.2))

controller = QuadricSection3D(
    self,
    surface=sphere,
    section_id="section",
    plane=plane,
    paint_policy="diagrammatic",
).attach()
```

`QuadricSection3D` computes the complete finite boundary by default and keeps
it synchronized with the same current plane used by the unified compositor.
For a fixed-topology moving cut, pass a plane callback.  Potential finite-cone
or cylinder cap-chord IDs are reserved automatically, so a chord can activate
or disappear without creating, removing, or replacing a Manim object.  Set
`draw_section_boundary=False` only when the intended lesson needs the plane
partition and surface boundaries but deliberately omits the true section ink.
The callback may not change the lateral conic family, branch count, component
count, or empty/non-empty state. Such a change introduces new curve identities,
fails before commit, and rolls back; use the scheduled form below instead.

For an ellipse/parabola/hyperbola topology change, pass the existing analytic
schedule instead of `surface`, `section_id`, and `plane`:

```python
controller = QuadricSection3D(
    self,
    scheduled=track_scheduled_plane_section("section", cone, motion),
    progress=progress,
).attach()
```

This mode delegates to `QuadricSectionTransition3D`; the static mode delegates
to `QuadricOcclusion3D`.  The facade does not contain a second section solver,
painter graph, Manim slot pool, or rollback implementation.  Advanced callers
may still use the lower-level controllers directly.  When doing so, a moving
finite cone or cylinder must reserve `section_cap_chord_curve_ids()` together
with the initially active lateral IDs through `allocated_curve_ids`.
`compute_quadric_section()` plus `section_trace_curves()` remains available
when a caller intentionally wants only the lateral support-quadric trace.
`compute_quadric_section_boundary()` returns both that exact trace and the
complete finite boundary from one solve; its existing
`compute_quadric_section_boundary_curves()` adapter returns only the curves.

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
retains the exact event. Filled cap chords are different: their semantic role
does not change with the lateral conic family, so each `cap_min` or `cap_max`
uses one fixed slot computed from the actual current plane. A critical-bank
plane is never submitted as if it were the current plane's authoritative cap
chord.

The transition controller displays and unifies the moving cutting plane by
default.  Its plane patch is fitted again from the same immutable surface and
current plane, while the renderer identities remain fixed.  Passing
`show_plane=False` disables the complete section-plane compositor, including
its fill, outline, depth partition, and curve attenuation; it does not merely
make the rectangle transparent. Use it only for a curve-only presentation.

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

- finite sphere, cylinder, closed-single-cone, open-single-shell, and
  open-double-shell contracts and homogeneous forms;
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
- global ordering for a bounded set of strictly separated convex quadrics;
- component-aware cone projection layers that distinguish lateral paint from
  one or two real caps, keep both frustum terminal disks separate, and leave
  an open mouth as a one-sheet region;
- fixed-capacity Manim component slots and independent lateral/cap color
  gradients without updater-time Mobject creation.

The regression suite exercises scales from `1e-6` through `1e6`, large common
world translations, equivalent scaled projection rows, repeated updates, Fade
lifecycle, ten masked five-state Cairo keyframes, a surface-only opacity frame,
and a continuously moving Cairo section near the parabolic event.  Release
validation additionally builds the wheel and sdist, checks their metadata, and
installs the wheel in an isolated environment.

## Deliberate limitations

- The global multi-surface solver requires pairwise strict 3D separation,
  except for the certified shared-apex contact between the two components of
  one `OPEN_DOUBLE` shell. Those siblings are accepted only when their
  projected interiors do not overlap. An oblique view that needs interleaved
  multi-sheet ordering fails explicitly. Other touching/intersecting entities
  and true surface painter cycles are still rejected; quadratic surface-cell
  splitting is not implemented yet.
- One local cutting-plane compositor accepts exactly one finite convex surface.
  `OPEN_SINGLE` is supported directly. `CompositeQuadricSection3D` is a narrow
  coordinator for the two canonical components of one `OPEN_DOUBLE`: it keeps
  two local surface-sheet pairs but one common plane partition and painter
  order. If their contact is not one certified shared-apex point, or if a
  callback changes the curve identity/topology family, it fails and retains
  the last committed frame. It does not guess an interleaved surface order.
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

## Opt-in per-frame performance evidence

Performance measurements are diagnostic evidence, not part of the geometry or
painter contract. They are disabled by default, so normal controller updates do
not read a wall clock or scan the display tree solely for metrics. Enable them
before constructing a controller:

```bash
export MANIM_TIKZ_NATIVE_QUADRIC_PERFORMANCE_TRACE=1
```

After a successful update or a rolled-back failure,
`controller.performance_snapshot()` returns one immutable, JSON-safe snapshot.
It separates input resolution, surface/global-frame work, section compositing,
contour union, boundary visibility and crossings, painter-graph construction,
adaptive projection, dash generation, painter-band preparation, transaction
snapshot, and Manim application. Counts include plane fragments, classified
rays, total/active/modified Mobjects, and cache hit/miss evidence. A failed
attempt records its exception type and whether transactional rollback ran; it
does not replace the controller's last-good frame.

The Cairo commit itself is incremental. Each prepared fixed slot has a display
digest derived from its certified numeric paths, style, intent, and effective
opacity. Identical active slots are not rewritten, only formerly active slots
are hidden, and an unchanged painter-band signature is not reapplied. The
transaction snapshots only those fixed Mobject families which can change in
that commit. Trace counts expose active, changed, unchanged, hidden, mutation-
target, and snapshot sizes; this optimization does not weaken capacity checks
or create Mobjects during an updater.

Before preparing an updater frame, both the single-surface and open-double
Cairo controllers resolve every dynamic input exactly once and compute exact
geometry, draw, and root-opacity signatures. An unchanged signature reuses the
last certified prepared frame and skips both renderer-neutral geometry and the
Manim display transaction. A curve-opacity-only or root-opacity-only change
reuses the numeric frame but still updates the affected fixed slots. Any
surface, curve, view, section-plane, patch, tolerance, style, or policy change
continues through the complete fail-closed preparation path. Signatures are
not rounded, topology changes are still validated before reuse, and even a
clean frame rechecks that no unrelated Scene drawable has entered the reserved
painter band. Restore clears the cache; a failed attempt never replaces its
last-good contents.

Full geometry frames also use a fixed-size `SurfaceViewCache`. Its keys contain
the immutable surface data, full parallel-view matrix, geometry context,
ordering inputs, and relevant subdivision limits; Python object identity and
rounded animation values are never used. For a fixed surface and camera, a
moving section plane can therefore reuse the certified surface/global frame,
cone component-fill paths, and intrinsic surface-boundary sources together
with their self-visibility spans and mutual crossings. The current plane,
section curves, plane partition, boundary/plane placement, and every crossing
involving a plane-dependent source are still recomputed. Trace snapshots
publish `surface_view_base`, `surface_component_fill`, and
`static_surface_boundaries` cache evidence. Each named cache holds only its
latest exact entry and is cleared by `restore()`, so an animated camera or
surface cannot create an unbounded history.

Boundary display preparation performs one adaptive projection for each source
which has painted fragments. All current fragment endpoints are mandatory
samples in that certified source polyline, so slicing preserves exact projected
endpoints while inheriting the source error bound. Solid and dashed fragments
then reuse parameter slices of that one polyline; they do not run another
adaptive projection. Source-distance dash anchoring still uses the complete
polyline. Performance traces distinguish `projected_boundary_source_count`
from `projected_fragment_slice_count` and time the lightweight slices under
`projection_slicing`.

Finite closed-section arrangement also memoizes exact ray/polygon
intersections while coalescing narrow cells between nested convex boundaries.
The key contains the inner/outer boundary identity and the unrounded IEEE-754
angle, lives only for that one partition call, and returns the same canonical
vertex from the existing registry. This removes repeated edge scans without
changing subdivision, tolerances, fragment capacity, or any certified frame.

Several controllers which differ only in paint policy may share one
`QuadricGeometryPrototype`. The first exact surface/plane/view signature
computes the section partition and merged contours; later variants reuse that
renderer-neutral geometry and rebuild only their own physical, diagrammatic,
or depth-aware boundary painter graph. The exact boundary/section placement
spans are also reused when their complete sources, visibility spans, projected
crossings, section geometry, context, and limits match; physical-only crossing
filters therefore remain a separate certified cache miss while diagrammatic
and depth-aware variants can share the same placement result.
`display_offset=(x, y)` translates only the prepared Cairo paths, so
side-by-side columns do not alter view depth or the cached geometry. Every
variant still owns its own fixed Mobject slots, painter band, identity map, and
transaction. Inputs are never rounded: a real surface, plane, view, tolerance,
or limit change is a cache miss. Shared caches are bounded to the latest exact
product per named stage and are cleared explicitly with `prototype.clear()`
rather than by restoring one variant. Trace evidence publishes
`shared_section_geometry` and `shared_boundary_section_spans` hits and misses.

The extended Cairo acceptance generator enables this trace automatically. Its
keyframe JSON and fragment/ray CSV contain the controller measurements. Video
subprocesses additionally publish a per-rendered-frame trace with the Cairo
render duration. Timing values are machine-dependent and are therefore used for
profiling and performance budgets only; they never select geometry, change
visibility, or enter deterministic frame hashes.

## Unified semantic boundary visibility

`QuadricOcclusion3D` keeps its historical pixel contract by default. Opt into
fragment-level boundary compositing with `boundary_visibility_mode="unified"`.
The unified sidecar preserves the existing v1 surface, visibility, and section
frames while adding one deterministic painter frame for ordinary analytic
curves, the four finite display-patch edges, cap rims, true silhouettes, and
explicit teaching generators. That painter frame is the separately versioned
`manim-quadric-boundary-compositing/v2` contract; it has one runtime path and
does not generate the superseded boundary-compositing v1 payload.

```python
from polyhedron_visibility.quadrics import (
    GeneratorBoundarySpec,
    QuadricBoundaryStyle,
    QuadricOcclusion3D,
)

controller = QuadricOcclusion3D(
    self,
    surfaces=(cone,),
    curves=section_curves,
    section_plane=plane,
    paint_policy="depth_aware_diagrammatic",
    boundary_visibility_mode="unified",
    boundary_styles={
        "style:emphasis": QuadricBoundaryStyle(
            visible_color="#E53935",
            visible_width=4.5,
            hidden_color="#B71C1C",
            hidden_width=3.0,
            dash_length=0.10,
            dash_gap=0.07,
        ),
    },
    generator_boundaries=(
        GeneratorBoundarySpec(
            "teaching-generator",
            cone.surface_id,
            0.42,
            style_id="style:emphasis",
        ),
    ),
).attach()
```

The controller snapshots this registry during construction. Frame updates only
resolve an existing `style_id` and mutate preallocated solid/dash slots; they
never add a style or create a Mobject. Unknown IDs, too many registered styles,
and a dash pattern that exceeds `max_dashes_per_fragment` raise explicitly.

The three painter policies consume one effective visibility result for every
semantic boundary. A fragment is effectively hidden when either a selected
quadratic surface hides it or it projects inside the finite section-plane patch
and lies behind that plane:

- `physical`: visible fragments are solid and hidden fragments are omitted;
- `diagrammatic`: visible fragments are solid and hidden fragments are dashed
  teaching overlays above their occluders;
- `depth_aware_diagrammatic`: hidden fragments remain dashed, but every
  certified farther object is painted first and every actual occluder is
  painted afterward. A translucent front sheet or section-plane role fill
  attenuates the dash, while an opaque occluder can cover it completely.

A true projection silhouette is not the same object as a cap rim or a display
frame. Sphere silhouettes and the lateral silhouette generators of finite
cylinders/cones use external-only occlusion, so their owning surface never
turns them into hidden dashes. Circular cap rims and explicitly authored
surface generators are ordinary owner-aware semantic boundaries: their front
parts are solid and their rear parts follow the selected hidden-line policy.
Open-shell trim rims follow the same owner-aware rule, but never create a
planar occluder.
The rectangular plane-patch outline reuses its existing exact
`PlaneDepthRole` partition instead of solving visibility again.

External-only applies to quadratic-surface selection, not to every possible
occluder. A finite cutting plane may still hide part of a true cone or cylinder
silhouette. In depth-aware mode that case is bracketed as
`surface_front -> silhouette dash -> plane role fill`; outside the patch, or
where the silhouette is in front of the plane, the same source remains solid.
The fragment contract records the original surface result separately from the
effective result as `surfaceVisibilityKind` and `effectiveVisibilityKind`, and
names plane painter items without pretending that the plane is a quadratic
surface. The ambiguous v1 `visibilityKind` field is not part of the v2 payload.

At a projected crossing with the finite plane outline, diagrammatic hidden ink
keeps its documented top-overlay precedence. Crossings between ordinary entity
boundaries continue to use their certified far-to-near depth order.

Every other semantic boundary is also split where it crosses a plane-role
contour. A midpoint labels only an already partitioned open interval; three
interior probes must agree or preparation fails closed. Curves analytically
certified as the surface/plane section are partitioned by their exact
visibility events, so increasing the plane triangulation density does not
consume additional fixed Manim fragment slots. Use
`boundary_section_limits=QuadricBoundarySectionLimits(...)` to set explicit
role-contour and per-source split capacities.

Unified boundary painter fragments use fixed preallocated solid/dashed pairs.
Dash phase is anchored to the complete semantic source, so a moving visibility
boundary clips only the first or last dash instead of making the pattern crawl.
No updater creates a `VMobject`, `VGroup`, or `DashedVMobject`; object identity,
the ten established section-layer slots, managed painter-band ownership, and
transactional last-good-frame rollback remain stable.
