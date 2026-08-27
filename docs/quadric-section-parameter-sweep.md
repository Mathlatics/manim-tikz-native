# Deterministic finite-cone section parameter sweep

The finite-cone section v1 regressions include a fixed parameter-space sweep.
Its purpose is to detect failures before they become a hand-authored screenshot
regression.  The sweep is renderer-neutral except for a separate fixed-capacity
Manim identity test.

Machine-readable matrix:
[`tests/fixtures/quadric-section-parameter-sweep-v1.json`](../tests/fixtures/quadric-section-parameter-sweep-v1.json)

Executable checks:

- [`tests/test_quadric_section_parameter_sweep.py`](../tests/test_quadric_section_parameter_sweep.py)
  covers the analytic section, visibility, plane partition, and painter graph;
- [`tests/test_quadric_section_parameter_sweep_manim.py`](../tests/test_quadric_section_parameter_sweep_manim.py)
  covers fixed Manim object identity, fixed capacity, allocation-free updates,
  and transactional rollback.

## Fixed matrix

The analytic solver tier contains 252 cases:

- four finite surfaces: closed/open single cones and closed/open frusta;
- seven cutting-plane normal angles: transverse, ordinary oblique,
  just below the parabolic threshold, exactly parabolic, just above it,
  near vertical, and vertical;
- nine plane positions: outside the entity, through or near the apex, at or
  near each finite terminal, through the lateral interior, and at an oblique
  terminal tangency.

The compositor tier contains 32 reviewed pairwise cases.  It combines the same
critical section families with seven deterministic parallel views: isometric,
cabinet, exact side, near side, axial, and two general views.  This is a
deliberate pairwise matrix, not a claim that every Cartesian product is within
the frozen v1 support contract.

Two scheduled Manim sequences cross the ellipse/parabola/hyperbola threshold:
one closed cone and one open frustum.  Each is sampled on both sides of the
critical frame and at the exact handoff.

## Certified invariants

The renderer-neutral tier checks:

- repeated runs produce identical traces, fragment IDs, painter relations,
  and draw order;
- finite parameter intervals form an exact partition without gaps;
- every open section endpoint lies on a real trim or cap boundary;
- open shells never acquire a cap chord, while closed surfaces acquire one
  only where the plane crosses a real cap disk;
- all plane fragments stay inside the fitted patch, preserve its area by
  `PlaneDepthRole`, and have no positive-area pairwise overlap;
- the painter graph contains exactly the active paint items, every relation is
  ordered far-to-near, and all fragment/ray counts remain within the published
  limits.

The Manim tier additionally checks that updates do not construct a `Mobject`,
replace a slot, change `scene.mobjects`, or change allocated curve IDs.  An
invalid update must preserve the complete last-good frame and must be followed
by a successful allocation-free recovery.

## Determinism and baselines

The fixture freezes one aggregate analytic semantic digest and one semantic
digest for every compositor case.  These hashes intentionally describe
geometry and ordering, not platform-sensitive pixels.  A digest is updated
only after reviewing the corresponding semantic change.

Signed zero is canonicalized before hashing.  Diagnostic ray-classification
work counts remain subject to the hard capacity assertion, but are not part of
the semantic digest: supported NumPy builds may take a different number of
certified subdivision steps while producing identical fragments, roles,
relations, and draw order.

Ordinary pull-request CI contains no random sampling.  A future randomized
sweep belongs in a separately reported nightly job with a fixed seed recorded
in its output; it must not make normal CI nondeterministic.
