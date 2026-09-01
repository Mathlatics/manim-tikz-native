# Geometry-kernel layers

The geometry kernel is being separated incrementally so existing TikZ and
Manim authoring APIs remain compatible while curved geometry is added.

The dependency direction is deliberately one way:

```text
Geometry -> Topology -> Visibility -> Compositor -> Manim bindings
```

The four kernel modules do not import Manim. The package root also exposes its
legacy renderer-facing API lazily, so importing
`polyhedron_visibility.kernel` does not load Manim as a side effect.

## 1. Geometry

`polyhedron_visibility.geometry` owns the numerical interpretation shared by
all later layers.

The existing `TolerancePolicy.resolve(...)` method remains the source of
truth. `GeometryContext` does **not** approximate or reinterpret that policy.
Instead, it resolves the same positions and optional edge length exactly once
and returns an immutable `ResolvedGeometryContext`.

The resolved context names distinct comparison quantities:

- world length;
- face boundary distance;
- positive view depth;
- line or curve parameter;
- angular comparison;
- projected screen distance.

Length, depth, parameter, and angular values are taken directly from the
legacy `ResolvedTolerance`. Screen tolerance defaults to the resolved world
tolerance until a renderer supplies an explicit projected-space value.
Explicit overrides are exact local values; they are not hidden scale factors.

This preserves the old numerical contract even for very small geometry. A
model with extent below one unit is never silently promoted to unit scale.

Later layers should normally receive a `ResolvedGeometryContext`. The
unresolved `GeometryContext` must first be resolved with the current frame's
world positions and any relevant edge length. Resolving against an empty
geometry is only the library default; it is not a substitute for the scene's
actual scale.

## 2. Topology

`polyhedron_visibility.topology` owns identities and connectivity, not depth.

- `ParameterInterval` represents one finite closed parameter interval.
- `partition_parameter_domain(...)` creates deterministic consecutive cells.
  Its default keeps exact authored-domain boundaries; the explicit upper-cluster
  mode exists only for frozen trace compatibility.
- `TaggedInterval` carries semantic identity through a moving configuration.
- `coalesce_tagged_intervals(...)` merges adjacent cells only when their
  semantic identity agrees.

This distinction matters at tangency and handoff positions. Two coincident
roots may occupy two animation slots even when they represent one distinct
geometric point; similarly, adjacent hidden intervals owned by different
occluders must not silently become one slot.

## 3. Visibility

`polyhedron_visibility.visibility` classifies topology cells.

- `OcclusionInterval` attributes a hidden range to one semantic occluder.
- `VisibilitySpan` rejects inconsistent visible/hidden states at construction.
- `partition_visibility(...)` returns stable visible and hidden spans.
- Its default `EXACT` mode keeps authored parameter boundaries. The explicit
  `TOLERANCE_EXPANDED` mode preserves the historical closed-polyhedron v1 trace
  convention while that production path is migrated.
- Compatibility modes are internal adapters for frozen persisted contracts;
  new geometry should use the exact defaults unless it must reproduce an
  already-versioned trace byte for byte.
- Boundary-only contact has no positive parameter length and remains visible.
- Occluders use first-authored order as the deterministic tie breaker, even
  when a caller-provided semantic sort key is equal.
- Adjacent hidden cells merge only when the complete ordered occluder identity
  agrees.

The visibility layer does not create dashed lines, polygons, or Manim
mobjects. It returns renderer-independent data.

## 4. Compositor

`polyhedron_visibility.compositor` turns pairwise far/near facts into a stable
painter order.

- `PainterConstraint(farther, nearer)` records one depth relation.
- `stable_topological_sort(...)` preserves authored order for unrelated or
  equal-key fragments.
- Hash containers are used only for membership, never as an output-order
  source.
- Cycles raise `CompositorCycleError`; they are not resolved by an unstable
  dictionary or set iteration order.
- `painter_ranks(...)` converts the final order into draw indices.

The compositor does not decide whether a line is visible. It only orders the
fragments supplied by the visibility layer.

## Production adoption and current boundary

The closed-polyhedron production chain was the first migrated slice:

```text
AutoOcclusion3D
    -> ManimOcclusionBinding
    -> parallel_solver.compute_frame_visibility
    -> GeometryContext + partition_visibility
    -> frozen VisibilityFrame / Manim slots
```

The open-face and section production paths now reuse the same topology and
compositor contracts:

```text
OpenFaceOcclusion3D / TikZ open-face binding
    -> open_faces.compute_open_face_visibility
    -> partition_visibility + stable_topological_sort
    -> frozen OpenFaceVisibilityFrame

ConvexSection3D
    -> sections.compute_sectioned_visibility
    -> partition_visibility
    -> frozen VisibilityFrame

TransparentSectionLayer
    -> compute_transparent_section_compositing
    -> domain-specific fragment relations
    -> stable_topological_sort
    -> frozen transparent-compositing trace

DerivedDihedralUnifiedLayer
    -> compute_derived_dihedral_unified_compositing
    -> domain-specific face/line and line/line relations
    -> stable_topological_sort
    -> frozen unified-compositing trace / existing Cairo z slots
```

The line/face intersection formulas, overlap tests, depth relation generation,
and persisted v1 trace objects remain unchanged. Open faces carry a structured
`(face_id, logical_surface_id)` identity through the shared visibility layer,
then adapt it back to the existing face-level and logical-surface-level trace.
The compatibility boundary mode reproduces the old interval splitter byte for
byte, while new geometry keeps exact authored-domain defaults. Open-face whole
faces and transparent section fragments preserve the historical lexicographic
identity tie breaker when no depth constraint distinguishes them.

Differential tests cover legacy breakpoint behavior, public canonical traces,
and painter orders. Production-path tests prove that the public open-face and
section entry points reach the shared partitioner and compositor rather than a
parallel private implementation.

The derived-dihedral compositor also keeps its specialized face/stroke
fragmentation and depth-relation generation, but its final graph ordering now
uses the same shared compositor. Its domain adapter continues to reject unknown
item identities and self-relations before calling the generic sorter, and maps
shared cycle errors back to the existing derived-dihedral error contract. The
fail-closed exception type is preserved for malformed self-relations; the exact
residual-node list in that diagnostic is not part of the compatibility contract.

The renderer-neutral open-face unified-compositing stage is now a separate
Provider component.  It preserves the frozen v1 visibility frame, then splits
straight semantic paths at visibility boundaries, finite-face overlap/depth
events, projected path crossings, and direction-preserving collinear overlap
events.  The result contains only face identities, path-fragment parameter
intervals, visibility provenance, pairwise paint relations, and a validated
deterministic draw order; it imports neither Manim nor render-slot concepts.

Its two paint policies deliberately answer different questions.  The
`physical` policy orders every positive-area face/path overlap by actual view
depth.  The `diagrammatic` policy treats semantic path ink as the foreground
over every overlapping face fill; visibility still decides whether that ink
is solid or dashed.  Explicit coplanar declarations keep the same ink-over-fill
rule, while an undeclared depth tie remains a fail-closed authoring error.

Projected path/path intersections are computed once per source-path pair and
reused both for fragmentation and for painter-relation generation.  Point
events are localized only to the adjacent fragments, and collinear overlaps
are matched with a parameter-ordered sweep instead of comparing the Cartesian
product of all fragments.  A separate fragment-pair-candidate limit fails
closed before an unexpectedly dense arrangement can consume unbounded work.

The production migration described above is complete for the supported
open-face path:

1. renderer-neutral fragments map to preallocated generic open-face Cairo
   slots inside one managed painter z-band;
2. frame application is transactional and covers fade, restore, detach, and
   reattach lifecycles;
3. source-authoritative projects generate the unified binding and fail closed
   instead of selecting a legacy fallback; and
4. quadric semantic boundaries reuse the shared analytic-fragment visibility
   and deterministic painter-graph approach rather than treating every curve
   as an unrelated overlay.

This does not make the kernel a general mesh renderer or depth buffer. General
non-convex faces, arbitrary curved faces, unconstrained intersecting surfaces,
and several simultaneous cutting planes still require new geometry/topology
contracts before they can enter these shared visibility and compositor layers.

## Invariants for future curved geometry

- geometry computes exact or controlled-approximation intersections;
- topology records multiplicity, branch identity, and interval continuity;
- visibility chooses the nearest positive-depth hit and splits parameter
  domains;
- the compositor orders opaque and transparent fragments deterministically;
- Manim bindings update preallocated slots instead of changing object topology
  every frame.
