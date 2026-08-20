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

## Production adoption and migration plan

The closed-polyhedron production chain is now the first migrated slice:

```text
AutoOcclusion3D
    -> ManimOcclusionBinding
    -> parallel_solver.compute_frame_visibility
    -> GeometryContext + partition_visibility
    -> frozen VisibilityFrame / Manim slots
```

The line/face intersection formula and persisted v1 trace objects remain
unchanged. `parallel_solver` resolves frame-, face-, and edge-local contexts at
the same boundaries formerly passed to `TolerancePolicy.resolve(...)`, then
adapts shared kernel spans back to the frozen trace schema. Differential tests
cover legacy breakpoint behavior and canonical traces, and a production-path
test proves that `AutoOcclusion3D` reaches the shared visibility layer.

The remaining migration should stay small and reviewable:

1. migrate open-face and section interval partitioning while preserving their
   richer logical-surface trace identities;
2. replace duplicated painter-graph sorts in section, open-face, and derived
   dihedral compositors with the shared compositor;
3. promote the existing derived-dihedral face/stroke relation generator into a
   reusable open-face compositing stage;
4. route both generic and TikZ-native open-face Manim bindings through that
   shared stage;
5. use the same layers for conics and quadrics rather than adding a parallel
   curved-geometry visibility stack.

## Invariants for future curved geometry

- geometry computes exact or controlled-approximation intersections;
- topology records multiplicity, branch identity, and interval continuity;
- visibility chooses the nearest positive-depth hit and splits parameter
  domains;
- the compositor orders opaque and transparent fragments deterministically;
- Manim bindings update preallocated slots instead of changing object topology
  every frame.
