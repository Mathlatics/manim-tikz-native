# Supported TikZ subset

The authoritative machine-readable registry is
[`tikz_native/subset_v0_1.json`](../tikz_native/subset_v0_1.json). Every feature
is classified as dynamic-safe (A), static-safe (B), or unsupported (C).

## Dynamic-safe highlights

- named 2D and 3D coordinates;
- numeric, side-effect-free macros in the controlled frontend;
- interpolation, translation, projection, and explicit 3D hinge relations;
- named paths for lines and ordinary 2D circles and ellipses;
- line-line and line-ellipse intersections with explicit path ordering;
- lines, Stealth arrows, polygons, ordinary 2D circles and ellipses, dots,
  labels, angle arcs, and right-angle markers;
- explicit line widths, dash patterns, xcolor mixes, opacity, and controlled
  font-size commands;
- `3d view` and explicit TikZ x/y/z parallel-projection bases;
- billboard labels and path labels with controlled anchors.

## Static-safe only

- TikZ baseline and external trim metadata;
- `dashed` and `densely dashed` keywords, represented with fixed approximate
  pitch instead of pixel-exact TikZ rhythm;
- explicit 3D supporting planes and full planar circles/ellipses through
  `\DeclareSpacePlane`, `\DrawSpaceCircle`, and `\DrawSpaceEllipse`, with the
  narrower solid-stroke v1 contract below;
- explicit finite right cones, Dandelin constructions, and one fixed
  spatial/meridian/section-plane view per picture; the spatial view may opt in
  to certified hidden-line intervals with `mode=depth_aware_diagrammatic`, or
  additionally certify the teaching-transparent surface order with
  `mode=depth_aware_teaching_transparent`;
- one proven redundant `scope` case with `draw=none`.

## Explicit planar 3D curves (static-safe v1)

An ordinary 3D center and radius do not determine a supporting plane. Author
the plane from three named, static 3D coordinates before drawing the curve:

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

`O` is the plane origin. `O -> U` fixes the positive local-u axis and the
parameter phase. `V` must be non-collinear and fixes the positive local-v
side. Their authored lengths certify orientation only and do not scale local
coordinates. The `0,0` and `0.5,-0.25` arguments are plane-local centers in
world-coordinate units; the final circle argument is its radius, and the final
two ellipse arguments are its local-u and local-v semi-axis lengths in the same
units. Plane and curve IDs are stable portable identities, share the global
semantic namespace, and must match `[A-Za-z][A-Za-z0-9_.:-]{0,127}`.

This first version supports exactly one complete revolution with a visible,
solid stroke. It accepts the existing draw color, positive line width,
draw/overall opacity, line cap, and line join fields. Fill, fill opacity,
dashes, arrow tips, additional canvas transforms, partial arcs, and animated
O/U/V geometry fail closed. The general static-safe dash statement above does
not extend to these explicit 3D planar curves. The current embedded
geometry-driver runtime also fails early whenever such a curve is present,
including the case where its plane is fixed and unrelated to the selected
hinge. Camera-only motion in the world-space renderer is a separate supported
path.

`NativeFixedViewRenderer` uses the authored parallel projection. A rank-two
image is the direct affine ellipse; an exact edge-on rank-one image is the
curve's finite segment, never the infinite support line. It does not silently
collapse a merely thin but numerically resolved ellipse. `NativeManim3DRenderer`
keeps the affine curve in its true world plane, so later camera changes act on
the same world-space Manim object. This is camera motion, not animated O/U/V
authorship. Neither renderer invents a filled disk or automatically registers
the curve with quadric visibility/compositing.

Ordinary 2D `circle` / `ellipse` paths and named paths keep their existing
meaning. In a 3D picture, an ordinary circle or ellipse path (including a
named path) is rejected and points the author to the explicit plane syntax.
The existing physical-size circle used as a point marker, such as
`\fill (P) circle (1pt)`, remains a dot rather than a spatial circle.

## Explicit Dandelin diagrams (static-safe)

The compiler never infers a Dandelin sphere from a visually plausible circle.
Declare a finite right cone, reuse one certified `DeclareSpacePlane`, and name
the derived relation explicitly:

```tex
\coordinate (A) at (0,0,0);
\coordinate (Z) at (0,0,1);
\coordinate (R) at (1,0,0);
\coordinate (O) at (0,0,2);
\coordinate (U) at (0,1,2);
\coordinate (V) at (-0.8,0,2.6);
\DeclareSpacePlane{cut}{O/U/V};
\DeclareSpaceRightCone{cone}{A/Z/R}{30}{0/9}{open_single};
\DeclareDandelinConstruction{dan}{cone}{cut};
\DrawDandelinDiagram[
  view=spatial,
  preset=classroom,
  mode=depth_aware_teaching_transparent,
  show-contact-circles=true,
  show-foci=true,
  show-directrices=false
]{dan};
```

`A` is the apex, `Z-A` fixes the positive axis, and the component of `R-A`
orthogonal to that axis fixes the radial phase. The half angle is expressed in
degrees; `0/9` is the finite axial range. Supported models are
`closed_single`, `open_single`, and `open_double`; `analytic_double` is not a
finite display object and is rejected.

`view` is `spatial`, `meridian`, or `section-plane`. The meridian view contains
true great-circle sections of the spheres, and those sphere circles are always
present. In that view `show-contact-circles` controls only the certified
generator-contact points; requesting directrices fails because they belong to
the cutting plane. The section-plane view contains the finite conic, foci, and
optional directrices but no sphere circles; explicitly requesting contact
circles there fails. Every visible item has a view-local ID
and a `sourceRef` back to the shared cone, plane, sphere, focus, contact circle,
or directrix.

The current contract allows one Dandelin diagram and no other drawable object
in the picture. It is always `static=true` and uses the fixed-view Provider
path only. All three views accept `mode=diagrammatic`; only the spatial view
accepts
`mode=depth_aware_diagrammatic` and
`mode=depth_aware_teaching_transparent`. Both depth-aware modes use the frozen
`space view` camera to classify analytic cone boundaries, sphere silhouettes,
contact circles, the section curve, the cutting-plane outline, and optional
clipped directrices, then order the resulting pieces with the shared fragment
painter graph, so both record `curveVisibilityAuthoritative=true`. The
teaching-transparent mode also
records `surfaceLayeringAuthoritative=true`: cone sheets, sphere fills, and
cutting-plane fragments have a certified painter order for this classroom
display. This is not an opaque physical visibility claim, so
`surfaceVisibilityAuthoritative` and
`physicalSurfaceVisibilityAuthoritative` remain false, as does the aggregate
`visibilityAuthoritative`. The first teaching-transparent release covers
single-nappe circle, ellipse, and parabola constructions. An open-double
hyperbola whose plane partition cannot be certified is rejected before the
diagram is registered; use `depth_aware_diagrammatic` for its automatic curve
visibility. This mode also requires `show-contact-circles=true`, because the
contact strokes own the certified equal-depth seams. Full `physical` mode is
rejected. Motion, camera shots, Bridge
geometry rigs, and source-v3 generation also fail closed. A complete
three-view source is in
[`examples/tikz_dandelin_views`](../examples/tikz_dandelin_views/README.md).

## Unsupported examples

- arbitrary Bézier paths, general arcs, plots, or smooth sampling;
- clipping paths;
- decorations, patterns, shading, or gradients;
- complex node shapes, automatic text wrapping, or TikZ matrices;
- nested transformed scopes with inherited geometry and style;
- arbitrary macros, conditionals, file access, or complex `pgfkeys`;
- topology-changing intersections such as tangency merge/disappearance.
- plane-less ordinary 3D circle/ellipse paths, and fill, dash, or arc semantics
  for explicit planar 3D curves.

Unsupported syntax is reported with its source statement. Strict conversion
stops; it never silently substitutes SVG, raster output, or an uneditable
generic path.

## Authoring guidance

- Name every point that will be animated or referenced later.
- Separate polygon fill from edge drawing when the edge needs its own semantic
  identity.
- Prefer explicit dash patterns when exact rhythm matters.
- Declare 3D projection and hinge relationships instead of relying on visual
  coincidence.
- Declare the supporting plane of every spatial circle or ellipse explicitly;
  do not use screen-space radii as a substitute for world geometry.
- Keep macros numeric and side-effect-free.
