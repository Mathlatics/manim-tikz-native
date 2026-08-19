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

## General source-to-copy identity handoff

`polyhedron_visibility.copy_handoff` handles the short interval in which a
copied geometric entity still occupies the same pixels as its source. The
copy operation freezes semantic vertex pairs and the corresponding face/stroke
pairs. Every frame then measures those paired vertices in the same final
coordinate space used by the overlay. At exact identity the copy owns the
coincident paint; as it separates, only the paired source faces and strokes
return through a cubic smoothstep. The copy stays at full authored opacity.

The contract works for a complete copied solid or a selected registered
subset. It does not infer similarity from rendered pixels and it does not
match unrelated objects by proximity. Callers can therefore move, rotate, or
return the copy without changing object identity or creating frame-local
Mobjects. The extracted-dihedral controller below is the first built-in
consumer of this generic policy.

## One closed solid plus one extracted dihedral

`ExtractedDihedralScene3D` freezes a teaching copy of exactly two adjacent
source faces from one validated closed convex solid. The two faces must share
one complete source edge, and every boundary edge of their union must already
be registered as one straight semantic `Line`.

The extracted copy has its own stable vertex, face, and stroke identities and
is driven by a proper rigid transform: one right-handed rotation plus one
translation. It does not continue reading the solid's live vertices after the
copy is frozen. This makes “highlight the angle inside the solid, then move it
out for analysis” explicit and reproducible.

The same assembly can then make any declared source face the base plane.
`base_plane_rotation(face_id)` builds a rigid rotation about the source solid's
geometric center (the centroid of all registered solid vertices) and maps the
face's validated outward normal to world `-Z`. Supplying that motion as the
controller's `global_transform_provider` applies that center-relative motion
before each entity's independent placement. The source solid therefore rotates
about its own moved center. The copied dihedral inherits the same authored
center, but its local transform moves that center with the copy before display;
it rotates in place at its separated location instead of orbiting a fixed
world-space pivot. A balanced layout can translate the source by `-T/2` and use
the copy's relative transform `T`, leaving the copy at `+T/2`.

At the identity transform, coincident source faces and boundary edges hand off
their display to the highlighted copy. They are not drawn twice. As the copy
starts moving, each paired source face and boundary edge is reactivated with a
cubic smoothstep derived from its current separation in the final overlay
coordinate space. The default handoff reaches full strength at `0.12` overlay
units, and `identity_handoff_distance` may be adjusted for another scene scale.
This avoids a binary color/layer change on the first animated frame and works
symmetrically when the copy returns. After the handoff, all solid faces and
both extracted faces become one occluder set; all solid and extracted semantic
lines are cut into visible and hidden spans by that same frame calculation.

With `accurate_transparency=True`, the Cairo binding also handles translucent
face intersections. A face is split by another face only when their finite
convex polygons share a positive-length 3D crossing; disjoint polygons are not
partitioned merely because their infinite supporting planes cross. The local
cells are still triangulated for exact depth constraints, but consecutive
triangles from one source face at one valid draw position are submitted as one
compound Cairo path. A stable, conservatively preallocated pool updates points,
opacity, color, and `z_index` in place. Invalid frames keep the last good line
and fill state.

The same accurate mode enables unified Cairo compositing by default. The
authoritative visible/hidden spans are refined at line/face depth-exchange
roots and projected line/line crossings. Exact transparent face batches,
visible stroke fragments, and dashed stroke fragments then enter one local
dependency graph. A deterministic topological order assigns the already
preallocated Manim slots their far-to-near `z_index` values. No updater creates
or replaces a `Mobject`. Ambiguous equal-depth crossings, contradictory local
orders, dependency cycles, or fragment-capacity overflow fail before mutation
and retain the complete last-good frame.

This is a technical-drawing compositor under parallel projection, not a depth
buffer. An incident boundary or a coplanar semantic construction line is ink
on its face and paints above that fill. A line strictly behind a translucent
face paints first and is therefore tinted by the face; a line in front paints
after it. At a projected line crossing the nearer fragment paints last. The
binding requires distinct authored `z_index` values for every managed face and
stroke and rejects unrelated drawables inside the combined managed z band.

This v1 feature deliberately supports one closed convex solid and one copied
two-face dihedral. It does not yet accept an arbitrary second solid, several
copies, non-rigid deformation, curved faces, labels, dots, arrows, or arbitrary
compound source paths as managed occlusion geometry.

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
- one closed convex solid plus one rigidly transformed two-face dihedral copied
  from adjacent source faces, including global semantic-line occlusion,
  duplicate-free identity handoff, and exact local transparent-face splitting;
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
- arbitrary multiple solids intersecting or mutually occluding one another
  (the one-solid-plus-one-derived-dihedral workflow above is the only supported
  multi-entity case);
- non-convex solids, curved cutting surfaces, or several simultaneous cutting
  planes;
- OpenGL binding parity.

The solver fails closed on ambiguous or unsupported input rather than falling
back to authored z-index guesses.
