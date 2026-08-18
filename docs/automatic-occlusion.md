# Automatic 3D occlusion

The automatic-occlusion module solves hidden **semantic line segments** under
parallel projection. Every registered convex face participates; each line is
split into visible solid spans and hidden dashed spans.

## Why explicit topology is required

A generic `VGroup` does not tell the module whether two coincident points are
the same vertex, whether a polygon is an occluding face, or which faces share an
edge. Authors therefore register:

- stable vertex IDs with live 3D position providers;
- maximal planar convex faces with ordered vertex loops;
- semantic straight `Line` objects;
- incident faces that must not hide their own boundary;
- for open systems, articulated hinge seams and optional excluded occluders.

The topology stays fixed while the coordinates and camera projection may change
on every frame.

## Closed convex polyhedra

`OcclusionScene3D` validates a closed, connected, convex two-manifold. For each
semantic edge, the solver tests every non-incident occluding face, combines all
hidden parameter intervals, and emits the visible complement.

This is designed for teaching-scale solids such as tetrahedra, prisms, pyramids
and cubes. It is not a general mesh renderer.

## Open convex faces

`OpenFaceScene3D` represents a finite set of zero-thickness convex panels. This
covers a dihedral angle or folding construction that is not a closed solid.

An `articulated_hinge` explicitly states that two logical faces share a stable
boundary. Exact coplanar states at 0 and π are valid; the seam-aware solver does
not create a false visible crack along the declared hinge.

Lines inside a face can use `excluded_occluder_face_ids` to avoid numerical
self-occlusion without pretending that they are boundary edges.

## Independent lines and convex sections

`ConvexSectionScene3D` starts from one validated closed convex solid and adds:

- any registered free straight semantic line;
- exactly one infinite mathematical cutting plane with a finite display patch;
- a derived cross-section with stable source-edge/source-vertex evidence;
- optional entry and exit markers where a free line crosses the solid.

A free line has no incident solid faces. Every solid face may therefore hide
part of it, and the interval inside the solid is reported independently of
what is visible from the current camera. This distinction is useful in a
teaching scene: geometry answers “where does the line enter the cube?”, while
visibility answers “which portions should be solid or dashed from this view?”.

The cutting plane may move and rotate while keeping its identity. Its
intersection with the solid changes among `empty`, `point`, `segment`, and a
convex `polygon`. The default `plane_patch_mode="auto"` treats authored patch
dimensions as minimums, expands the visible rectangle around the complete
solid with 15% margin, and never shrinks it during one attached session.
`plane_patch_mode="strict"` preserves literal finite-panel behavior; partial
strict patches fail closed instead of drawing an incomplete result.

The plane can also participate as one additional line occluder. Solid edges,
free lines, and the derived section boundary are all updated through stable,
preallocated slots.

For intersecting transparent fills, `accurate_transparency=True` enables a
separate exact-order path for one closed convex solid and one fitted plane patch
under parallel projection:

1. the existing section solver produces the authoritative empty, point,
   segment, or convex-polygon result;
2. every solid face crossed by the plane is split along the intersection;
3. every section edge's full supporting line partitions the fitted finite
   display patch, so the exterior and the highlighted section are disjoint;
4. all resulting cells are triangulated, projected, and compared only where
   their screen areas overlap;
5. a deterministic dependency graph produces the far-to-near draw order;
6. a fixed Cairo triangle pool changes only points, fill, opacity, and
   `z_index` during animation.

This is exact alpha-compositing order for the supported geometry. It is not a
general depth buffer, refraction model, or order-independent-transparency
renderer. The binding requires one native fill-only, non-gradient `Polygon`
per solid face, distinct face `z_index` values, a display patch containing the
full section with positive margin, and no unrelated drawable inside the
managed face z-band. Ambiguous depth cycles and unsupported source objects fail
closed; an invalid dynamic frame keeps the last good triangle state.

## Didactic face depth cues

`DepthCuedAutoOcclusion3D` adds a reusable visual layer on top of the same
closed-solid visibility frame. It does not replace or weaken hidden-line
removal. For every frame it:

- orients each face normal outward from the solid centroid;
- derives a soft camera-relative light from screen-right, screen-up, and view
  directions;
- scales the authored face opacity using both front/back orientation and
  normalized face depth;
- keeps every face in the authored hue family while applying a small,
  continuous orientation-dependent warm/cool shift;
- fades distant and back-facing faces toward a configurable fog color so
  stacked transparent faces do not collapse into one dark patch;
- identifies silhouette edges from adjacent face orientations and thickens
  only their visible spans.

The Manim binding uses stable fill-only `Polygon` proxies. Original face fills
are hidden transactionally and restored exactly when the session ends. Hidden
dashes, free lines, the cutting plane, and the gold section boundary keep their
own styles. This is a teaching-oriented depth cue, not a physical lighting or
order-independent-transparency renderer.

## Face ordering

When every open face has a source `Polygon`, the Cairo binding also computes an
advisory far-to-near draw order for overlapping translucent fills. It updates a
separate stable proxy layer and restores the original face objects at the end
of the session.

This ordering is for visual composition under parallel projection. It is not a
replacement for physically correct transparency or a depth buffer.

## Stable rendering

The binding preallocates a fixed number of visible and hidden line slots. During
animation it only changes endpoints, opacity, and fill data; it does not replace
the registered source objects every frame. Dashed spans are anchored to the
source line's parameter origin, so the dash phase does not crawl as an
occlusion boundary moves.

`controller.session()` is transactional:

- attach hides the managed source strokes only after all validation succeeds;
- an invalid dynamic frame keeps the last good frame;
- normal or exceptional exit removes every overlay family member;
- source stroke, background stroke, fill, z-order, and Scene ownership are
  restored.

## Supported boundary

Version 0.1 supports:

- parallel projection only;
- straight semantic lines;
- finite, planar, convex faces;
- closed convex polyhedra or explicitly modeled open panels;
- Cairo rendering;
- fixed topology with live coordinates;
- one closed convex solid with one infinite moving section plane and registered
  free straight lines;
- optional face-orientation shading, depth-aware opacity, and silhouette
  emphasis for registered native `Polygon` face fills;
- exact local transparent-fragment ordering for one closed convex solid and
  one intersecting automatically fitted plane patch under parallel projection;
- at most 24 solid faces, 768 preallocated transparent triangles, and 589824
  transparent-fragment pair checks for the real-time exact section binding;
- at most 64 open faces, 128 open strokes, 64 seams, 4096 candidate pairs, and
  65536 preallocated overlay line slots for the real-time open-face binding.

It intentionally rejects or does not claim support for:

- perspective projection;
- non-convex or self-intersecting faces;
- arbitrary curves, compound paths, gradients, or mixed-style source families;
- undeclared seams, non-manifold closed meshes, or topology changes;
- general order-independent transparency, refraction, or arbitrary surface
  intersections;
- multiple solids intersecting or mutually occluding one another;
- non-convex solids, curved cutting surfaces, or several simultaneous cutting
  planes;
- OpenGL binding parity.

The solver fails closed on ambiguous or unsupported input rather than falling
back to authored z-index guesses.
