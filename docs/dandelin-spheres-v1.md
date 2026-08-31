# Dandelin spheres v1

This document freezes the first public contract for constructing Dandelin
spheres for a finite right circular cone and one static cutting plane. The
renderer-neutral solver derives analytic sphere, focus, cone-contact-circle,
directrix, and finite-fit evidence. It does not resize, clip, or visually fit a
sphere when the authored finite cone cannot contain the mathematical result.

The accompanying scene-owning Manim facade is deliberately narrower: it is a
static Cairo **diagrammatic teaching overlay**. The ordinary cone/section
controller still computes the supported cone, cutting-plane, and section-curve
relationships, but the auxiliary spheres are painted in a separate teaching
band. Their front/behind relationship with the tangent cone is not physically
composited. The public evidence makes this explicit with
`visibility_authoritative=False` and `overlay_mode="diagrammatic"`.

The static TikZ spatial renderer has two opt-in depth-aware paths. With
`mode=depth_aware_diagrammatic`, it reuses the analytic quadric visibility
kernel to split cone boundaries, sphere silhouettes, contact circles, the
section curve, and optional directrices into exact visible and hidden parameter
intervals. `mode=depth_aware_teaching_transparent` keeps that curve result and
also certifies a camera-dependent classroom painter order for cone sheets,
sphere fills, cutting-plane fragments, and the equal-depth contact seam. This
second mode is authoritative for **teaching layers**, not for optical material
or opaque physical surface visibility.

## Certified geometry

Let

- `A` be the cone apex;
- `a` be its unit axis;
- `alpha` be its half angle;
- `[L, U]` be its authored axial range;
- `n` be the section plane's unit normal; and
- `h = n . (P - A)` for any authored point `P` on the plane.

Write `q = n . a`, `s = sin(alpha)`, and `c = cos(alpha)`. A sphere tangent to
the cone has its centre on the cone axis. If its signed axial centre is `z`,
its centre and radius are

```text
C = A + z a
r = |z| s
```

Tangency to the cutting plane requires

```text
|q z - h| = r.
```

For one authored nappe sign `tau in {-1, +1}`, write the signed axial centre
as `z = tau d` with `d > 0`. The two analytic candidates then come from the
stable plane-side signs `k in {-1, +1}`:

```text
d_(tau,k) = h / (tau q - k s)
z_(tau,k) = tau d_(tau,k).
```

Only candidates with finite `d_(tau,k) > 0` belong to that nappe. This nappe
factor matters for the two sides of an `OPEN_DOUBLE`; omitting it would produce
the wrong negative-nappe sphere.

A denominator at the exactly certified parabolic angle represents the
classical sphere at infinity. V1 records only the remaining finite sphere; it
does not create a huge or synthetic infinity placeholder. A merely
near-parabolic input is not snapped to this event.

For every finite candidate the solver derives

```text
focus F                 = C - (n . (C - P)) n
contact-circle centre K = A + z c^2 a
contact-circle radius   = |z| s c
eccentricity e          = sqrt(max(0, 1 - q^2)) / c.
```

The contact circle is an explicit `Circle3DSpec` in its real plane, not an end
cap or an invented disk. A non-circular section also receives the directrix
obtained by intersecting the section plane with that contact-circle plane.
The mathematical directrix remains infinite; a teaching view clips it to a
finite `PlaneDisplayPatchSpec` and records the resulting `SegmentCurve`.

Construction certifies more than the formulas alone. The focus must lie on the
cutting plane and on its sphere; cardinal points of every contact circle must
lie on both support quadrics; cone and sphere normals must be parallel there;
and all values must remain representable at the authored scale. The canonical
JSON records `finiteFitCertified: true` only after these checks pass. It also
records the resolved `certificationContext` and optional coefficient tolerance.
The frozen public construction re-derives one canonical sphere-record set from
the cone and plane during validation: changing a derived centre, radius,
extent, focus, contact circle, directrix, family, or eccentricity—even by less
than the geometric residual tolerance—cannot retain the certification flag.
Geometric tolerances still govern analytic residuals and finite-boundary
decisions; they are not used as permission to mutate canonical evidence.
Signed zero is normalized during canonical JSON encoding, so numerically
equivalent `0.0` and `-0.0` inputs cannot create distinct evidence bytes.

## Finite-cone rule

For a candidate with axial centre `z` and radius `r`, its complete axial extent
is `[z - r, z + r]`. V1 accepts it only when the sphere fits strictly inside
the authored range:

```text
L + epsilon < z - r
z + r < U - epsilon
```

`epsilon` comes from the same resolved `GeometryContext` used by the section
solver. Exact or tolerance-level contact with a terminal plane is rejected:
that would introduce an additional cap/trim contact whose ownership is not
part of this contract. An open cone shell is tested with this analytic trim
condition; the solver does not pretend that the shell is a filled volume.

The finite section itself must be non-degenerate. Empty and point-only
sections, a plane through the apex, intersecting-line degeneracies, and any
section containing a real filled-end-cap chord fail explicitly.

## Support matrix

| Authored cone | Section family | V1 result |
| --- | --- | --- |
| `CLOSED_SINGLE` | Circle or ellipse | Supported only when the complete section is a pure closed lateral curve, has no cap chord, and both spheres fit strictly inside the finite range |
| `CLOSED_SINGLE` | Parabola or hyperbola | Rejected; a finite closed cone cannot provide this contract without an incomplete family or a real cap boundary |
| `OPEN_SINGLE` | Circle or ellipse | Supported when the section is a complete closed lateral curve and both same-nappe spheres fit strictly inside the trim range |
| `OPEN_SINGLE` | Exact parabola | Supported with its one finite sphere; the sphere-at-infinity branch is omitted explicitly |
| `OPEN_SINGLE` | Hyperbola | Rejected because one nappe cannot supply the complete two-focus construction |
| `OPEN_DOUBLE` | Hyperbola | Supported when one finite sphere fits strictly in each nappe and the existing composite-section constraints also pass |
| `OPEN_DOUBLE` | Circle, ellipse, or parabola | Rejected by v1; these families use the single-nappe contract |
| `ANALYTIC_DOUBLE` | Any family | Rejected because it is an infinite support object, not a finite directly renderable teaching object |

Every supported row is still conditional on numerical certification and the
finite-range rule. “Supported” never means that an invalid or under-resolved
frame will be guessed.

## Renderer-neutral API

Use `compute_dandelin_construction()` when only the mathematical evidence is
needed. Importing this path does not require Manim.

```python
from math import pi

from polyhedron_visibility.quadrics import (
    ConeModel,
    ConeSpec,
    SectionPlane,
    compute_dandelin_construction,
)

cone = ConeSpec(
    "cone",
    apex=(0, 0, 0),
    axis=(0, 0, 1),
    half_angle=pi / 6,
    axial_range=(0, 9),
    model=ConeModel.OPEN_SINGLE,
)
plane = SectionPlane(
    "cut",
    point=(0, 0, 2),
    normal=(0.6, 0, 0.8),
    u_axis=(0, 1, 0),
)

construction = compute_dandelin_construction(
    "ellipse-proof",
    cone,
    plane,
)

for record in construction.spheres:
    print(record.sphere.center, record.sphere.radius)
    print(record.focus.world_point)
    print(record.cone_contact_circle)
```

`DandelinConstruction3D` exposes the certified `sphere_surfaces`,
`focus_points`, `cone_contact_circles`, and `directrices`. Its
`canonical_json()` form preserves stable semantic identities and the finite-fit
evidence. `canonical_dandelin_construction_json()` is the corresponding strict
helper.

For a custom teaching renderer,
`build_dandelin_teaching_overlay(construction, patch)` bundles the sphere
surfaces, contact curves, clipped directrices, focus identities, and a stable
diagrammatic draw order. It accepts only `mode="diagrammatic"`; requests for
`physical` or `depth_aware_diagrammatic` fail instead of making an unsupported
visibility claim. This restriction belongs to the Manim teaching-overlay
contract; it does not disable the separate fixed-view TikZ hidden-line path.

## Certified two-dimensional views

The same construction can be lowered into two mathematically different 2D
contracts:

```python
from polyhedron_visibility.quadrics import (
    build_dandelin_meridian_diagram,
    build_dandelin_section_plane_diagram,
)

meridian = build_dandelin_meridian_diagram(construction)
section_view = build_dandelin_section_plane_diagram(construction)
```

`DandelinMeridianDiagram2D` is the true plane through the cone axis and the
projected section-plane normal. Its sphere circles are genuine great-circle
sections, and every contact with the section line and finite cone generators
is re-certified. `DandelinSectionPlaneDiagram2D` instead lives in the authored
cutting plane: it contains the exact conic, foci, directrices, and sphere-plane
contact evidence, but deliberately has no sphere-circle field. A circle drawn
around a focus in that view would be invented geometry.

Both views retain the authoritative `DandelinConstruction3D` and rederive all
of their fields during validation. A view-local `pointId`, `lineId`, or
`circleId` is unique to that diagram, while `sourceRef` points back to the same
focus, sphere, contact circle, plane, cone, or directrix across views. This
keeps multi-view teaching assets collision-free without losing semantic
identity. The canonical JSON helpers are
`canonical_dandelin_meridian_diagram_json()` and
`canonical_dandelin_section_plane_diagram_json()`.
`sourceRef` is a source relation rather than a unique object key: for example,
both visible generator sides legitimately refer to the same authored cone.

## Static Manim teaching facade

Inside a Cairo `Scene`, `DandelinSection3D` builds the supported cone section
and the separate auxiliary teaching overlay:

```python
from polyhedron_visibility.quadrics import DandelinSection3D

DandelinSection3D(
    self,
    cone=cone,
    plane=plane,
    construction_id="ellipse-proof",
    show_contact_circles=True,
    show_directrices=True,
    show_foci=True,
).attach()
```

The cone, plane, and complete parallel-camera frame are immutable for the
lifetime of this v1 facade. A `ParallelCameraState` freezes its matrix,
`target`, `screen_anchor`, `zoom`, and the current Manim viewport translation
once; a projection callback is rejected without being called. The section,
overlay, and focus dots then consume that same affine frame.

`attach()` lazily reserves one Scene painter band, builds the fixed-capacity
controllers, and commits all Scene, fixed-frame, display, cache, and author
state as one transaction. The default preferred aggregate band is `(10, 32)`;
it is moved upward when another owner already occupies it, then split into
section `(10, 20)`, overlay `(21, 31)`, and focus `32` in the unshifted case.
`restore()` / `detach()` and `session()` release the reservation even when a
later cleanup layer reports an error, provided Scene and fixed-frame ownership
are actually gone. If a child still owns display objects, the facade retains
its controller references and painter band so a later `restore()` can retry;
another `attach()` is refused in that state. Consequently `slot_identities()`
is available only while the facade is attached. The older explicit
`section_painter_z_band` and `overlay_painter_z_band` arguments remain a paired
exact-band override. A band that cannot represent every active painter item
with a distinct finite float fails before any z-index mutation.

The ordinary cone/section part remains subject to its own certified painter
graph. The auxiliary sphere surfaces, contact circles, finite directrix
segments, and focus dots are then drawn in a separate top teaching band. Thus
the facade is suitable for explaining the focus construction, but it does not
prove which parts of a Dandelin sphere are physically hidden by the cone or
vice versa. In particular, it does not pass the tangent cone and spheres to the
ordinary global multi-quadric compositor, whose contract requires strictly
separated entities.

## Projection and renderer boundary

- The mathematical solver is renderer-neutral.
- The current facade accepts one immutable orthographic or general parallel
  projection, including full `ParallelCameraState` target, anchor, and zoom
  semantics. Perspective projection is unsupported.
- The production Manim path targets Cairo. OpenGL is unsupported and fails
  through the existing quadric binding.
- Camera/projection animation, moving cone or plane callbacks, and scheduled
  ellipse/parabola/hyperbola transitions are outside this static v1 facade.
- The section plane must satisfy the existing finite-section compositor
  requirements. A projection in which that plane has no certifiable display
  area still fails explicitly.
- `OPEN_DOUBLE` hyperbola display also retains the existing shared-apex-only
  projected-contact requirement of `CompositeQuadricSection3D`.

Full optical or opaque physical surface visibility remains outside this
contract. The ordinary global multi-quadric compositor cannot supply the
teaching order either because its surface-order contract requires strictly
separated convex entities, whereas each Dandelin sphere is nested inside one
cone component and has one-dimensional equal-depth contact with it. The TikZ
fixed-view path therefore uses the dedicated
`compute_dandelin_surface_layer_frame()` coordinator: existing cone projection
sheets and cutting-plane partitions remain the geometric evidence, while the
sphere is inserted analytically between its authenticated nappe's back/front
sheets.

Exact hidden-line visibility is narrower and is now supported without inventing
a second solver. `compute_dandelin_visibility_frame(construction, view, ...)`
records the tangent sphere/nappe/contact-circle evidence, lowers every semantic
stroke to the existing analytic boundary contract, and delegates the interval
classification to `compute_boundary_visibility()`. Its result therefore fixes
these two independent facts:

- `curve_visibility_authoritative=True`: visible/hidden curve intervals are
  certified for the frozen parallel camera;
- `surface_visibility_authoritative=False`: translucent fill order is still a
  teaching presentation, not a physical transparency proof.

The optional surface-layer frame adds two more explicit claims:

- `surface_layering_authoritative=True`: the far-to-near order of the
  classroom-transparent cone, spheres, and plane fragments is certified for
  this frozen camera;
- `physical_surface_visibility_authoritative=False`: the result is not an
  optical transparency simulation or an opaque hidden-surface claim.

For every authenticated sphere the coordinator proves

```text
cone back sheet -> sphere fill -> cone front sheet
```

The section compositor supplies the plane's behind/outside/between/front
regions. The sign of the exact sphere-centre-to-plane ray parameter inserts the
sphere on the correct side of those regions. Along each cone-contact circle,
the analytic equation `normal(theta) . view = 0` gives the front/back sheet
transition parameters; the semantic contact stroke owns those equal-depth
pixels so two translucent fills cannot fight for the seam. The two-sphere
circle special case additionally records the zero-dimensional common-focus
tangency. Because that semantic stroke is the certified equal-depth owner,
`depth_aware_teaching_transparent` requires `show-contact-circles=true` and
fails closed before registration when an author requests `false`.

Unsupported contact ownership, a missing finite directrix patch, or any
uncertifiable visibility partition fails closed instead of reverting to an
uncertified draw order.

## TikZ semantic boundary

V1 does not infer a Dandelin relation from an arbitrary authored circle or
sphere-like drawing. The source must name all three relationships explicitly:

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

The diagram view may be `spatial`, `meridian`, or `section-plane`. Each payload
recomputes the cone, section plane, construction, and selected view from the
authoritative named coordinates before rendering. Visible objects use
view-local IDs and retain their shared geometry through `sourceRef`.
Meridian sphere circles are unconditional true great-circle sections;
`show-contact-circles` controls only their certified generator-contact points,
and meridian directrices are rejected as cutting-plane geometry.

This TikZ path is static fixed-view. It permits one Dandelin diagram and no
other drawable object in the picture. Every view accepts the legacy
`mode=diagrammatic`; only `view=spatial` additionally accepts
`mode=depth_aware_diagrammatic` and
`mode=depth_aware_teaching_transparent`. Both depth-aware modes record
`curveVisibilityAuthoritative=true`. Only the teaching-transparent mode records
`surfaceLayeringAuthoritative=true`; both keep
`surfaceVisibilityAuthoritative=false`,
`physicalSurfaceVisibilityAuthoritative=false`, and the conservative aggregate
`visibilityAuthoritative=false`. The first surface-layer release supports the
single-nappe ellipse, circle, and parabola configurations. An open-double
hyperbola whose exact plane partition cannot be certified fails closed and can
still use `depth_aware_diagrammatic`. Teaching-transparent diagrams must keep
`show-contact-circles=true` because those strokes own the equal-depth seams.
Full `physical` mode remains unsupported.
The section-plane view also rejects a request for sphere/contact circles instead
of drawing circles around the foci. Motion, camera shots, geometry Bridge
requests, and source-v3 generation are outside this contract and must fail
before derived output is published. See
[the complete three-view source](../examples/tikz_dandelin_views/README.md).
