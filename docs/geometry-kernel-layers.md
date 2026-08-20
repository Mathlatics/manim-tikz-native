# Geometry-kernel layers

The geometry kernel is being separated incrementally so existing TikZ and
Manim authoring APIs remain compatible while curved geometry is added.

The dependency direction is deliberately one way:

```text
Geometry -> Topology -> Visibility -> Compositor -> Manim bindings
```

A lower layer must not import a higher layer. In particular, none of the four
kernel layers imports Manim.

## 1. Geometry

`polyhedron_visibility.geometry` owns the numerical interpretation shared by
all later layers.

- `GeometryContext` carries one `TolerancePolicy` through a complete solve.
- `GeometryScale` separates length, depth, parameter, angular, and projected
  screen scales instead of treating them as the same unit.
- `GeometryContext.epsilon(...)` is the only compatibility adapter for legacy
  policy attributes and methods.
- `resolve_geometry_context(...)` lets existing functions migrate from a
  `tolerance=` argument to `context=` without a breaking API change.

The context is immutable. A scene that needs a local scale or an explicit
exception creates a derived context with `with_scale(...)` or
`with_overrides(...)`; it does not mutate module constants.

## 2. Topology

`polyhedron_visibility.topology` owns identities and connectivity, not depth.

- `ParameterInterval` represents one exact parameter interval.
- `partition_parameter_domain(...)` creates consecutive cells whose first and
  last endpoints are exactly the authored domain boundaries.
- `TaggedInterval` carries semantic identity through a moving configuration.
- `coalesce_tagged_intervals(...)` merges adjacent cells only when their
  semantic identity agrees.

This distinction matters at tangency and handoff positions. Two coincident
roots may occupy two animation slots even when they represent one distinct
geometric point; similarly, adjacent hidden intervals owned by different
occluders must not silently become one slot.

## 3. Visibility

`polyhedron_visibility.visibility` classifies topology cells.

- `OcclusionInterval` attributes a hidden range to a semantic occluder.
- `partition_visibility(...)` returns stable visible and hidden spans.
- Boundary-only contact has zero positive parameter length and remains
  visible.
- Adjacent hidden cells merge only when the complete occluder identity agrees.

The visibility layer does not create dashed lines, polygons, or Manim
mobjects. It returns renderer-independent data.

## 4. Compositor

`polyhedron_visibility.compositor` turns pairwise far/near facts into a
stable painter order.

- `PainterConstraint(farther, nearer)` records one depth relation.
- `stable_topological_sort(...)` preserves authored order for unrelated
  fragments and is deterministic across frames.
- Cycles raise `CompositorCycleError`; they are not resolved by an unstable
  dictionary or set iteration order.
- `painter_ranks(...)` converts the final order into draw indices.

The compositor does not decide whether a line is visible. It only orders the
fragments supplied by the visibility layer.

## Migration plan

This first slice establishes contracts and regression tests without moving all
existing solvers at once. Subsequent changes should be small and reviewable:

1. pass one `GeometryContext` through the legacy line/face and polyhedron
   solvers;
2. replace local breakpoint and interval-merging helpers with the topology
   layer;
3. return `VisibilitySpan` data before constructing stable Manim slots;
4. replace duplicated painter-graph sorts in section, open-face, and derived
   dihedral compositors with the shared compositor;
5. use the same layers for conics and quadrics rather than adding a parallel
   curved-geometry visibility stack.

## Invariants for future curved geometry

The same contracts apply when conics and quadrics are introduced:

- geometry computes exact or controlled-approximation intersections;
- topology records multiplicity, branch identity, and interval continuity;
- visibility chooses the nearest positive-depth hit and splits parameter
  domains;
- the compositor orders opaque and transparent fragments deterministically;
- Manim bindings update preallocated slots instead of changing object topology
  every frame.
