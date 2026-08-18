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
- at most 64 open faces, 128 open strokes, 64 seams, 4096 candidate pairs, and
  65536 preallocated overlay line slots for the real-time open-face binding.

It intentionally rejects or does not claim support for:

- perspective projection;
- non-convex or self-intersecting faces;
- arbitrary curves, compound paths, gradients, or mixed-style source families;
- undeclared seams, non-manifold closed meshes, or topology changes;
- physically accurate transparency and general solid intersections;
- OpenGL binding parity.

The solver fails closed on ambiguous or unsupported input rather than falling
back to authored z-index guesses.
