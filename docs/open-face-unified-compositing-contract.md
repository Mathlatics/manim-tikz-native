# Open-face unified compositing contract

The open-face unified compositor is a renderer-neutral computation layer. It
combines the frozen open-face visibility trace with local painter events and
returns face items, path fragments, pairwise ordering relations, and one
canonical far-to-near draw order. It does not own Manim objects, render slots,
or animation lifecycle state.

## Frame invariants

`OpenFaceUnifiedCompositingFrame` is a validated boundary, not a loose transport
object. A frame is accepted only when all of the following conditions hold:

- its face identities and logical-surface identities exactly match the embedded
  `OpenFaceVisibilityFrame`, in canonical face order;
- every visibility path has one or more painter fragments, so a renderer cannot
  silently lose a source path;
- fragments are ordered by source path and parameter interval, form a complete
  non-overlapping partition of each source path, and have positive length;
- every fragment's visible/hidden kind and occluder provenance agree with the
  visibility span containing its midpoint;
- paint relations are in canonical identity order, reference known items, have
  no duplicate or contradictory direction, and form an acyclic graph;
- the draw order covers every item exactly once and equals the deterministic
  topological order of the validated relation graph.

A source path whose projection collapses below the painter-event tolerance now
fails closed with `OpenFaceUnifiedCompositingError` instead of disappearing from
the frame. A future contract may represent such paths explicitly as suppressed
items, but omission is not an accepted implicit state.

## Fragment identity

`PaintPathFragment.fragment_id` is deterministic only within one computed
frame. It is deliberately **frame-local**, not a lineage identifier.

When a camera or geometry update introduces a new painter event, later fragments
on the same source path may be renumbered even though much of their geometry is
continuous. Renderer bindings must therefore map each frame into their own
preallocated stable slots. They must not use `fragment_id` alone to decide that
a Manim `Mobject` is the same object across topology-changing frames.

The first generic Manim binding may use source-path order and a fixed-capacity
slot pool. A later curved-path implementation may add explicit event-lineage
identities without changing the meaning of the current frame-local IDs.

## Complexity limits

`OpenFaceUnifiedCompositingLimits` bounds all authored combinatorics before the
renderer allocates objects. In addition to source face/path pairs, total
fragments, fragment-pair candidates, and relations, it includes
`max_fragment_face_candidates`.

The current straight-path implementation conservatively counts

```text
number of path fragments × number of faces
```

before generating path/face relations. Payloads above the configured limit fail
closed. This keeps the exact relation stage bounded even before a future
projected-AABB broad phase is introduced.

## Consumer guidance for the Manim binding

A renderer consuming the frame should:

1. validate or construct the frame before mutating any displayed object;
2. allocate from a fixed-capacity face/fragment pool;
3. preserve source-path dash phase independently of fragment numbering;
4. assign z-order only after the complete item mapping succeeds;
5. commit points, styles, opacity, and z-index transactionally;
6. retain the last good frame if preparation or application fails.

These responsibilities belong to the follow-up product-binding work. The
renderer-neutral computation layer intentionally remains free of Manim imports
and renderer object identity.
