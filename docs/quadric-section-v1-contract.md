# Finite-cone section v1 support contract

- Status: **frozen semantic support contract**
- Contract ID: `quadric-section-v1`

Machine-readable contract:
[`tests/fixtures/quadric-section-v1-contract.json`](../tests/fixtures/quadric-section-v1-contract.json)

Version-specific implementation evidence:
[`release/quadric-section-v1-release-manifest.json`](../release/quadric-section-v1-release-manifest.json)

Deterministic parameter-space regression matrix:
[`docs/quadric-section-parameter-sweep.md`](quadric-section-parameter-sweep.md)

This document fixes the supported boundary of finite-cone sections and their
automatic occlusion. It is deliberately narrower than “all cone geometry”. A
supported row means that the renderer-neutral geometry path and, where
applicable, the Cairo Manim binding have release evidence for the stated
combination. An unsupported row must not be inferred from a nearby supported
case.

## Support matrix

| Scene or capability | v1 status | Boundary |
| --- | --- | --- |
| Closed finite single cone | Supported | Lateral surface, one real cap, cap rim, finite section boundary, unified plane and boundary compositing |
| Open finite single-cone shell | Supported | Lateral surface and real trim rim; no cap disk, cap chord, or volume membership is invented |
| Finite cone frustum: section, clipping, ordinary occlusion | Supported | Uniform styling, automatic boundary visibility, and up to two real cap chords are supported |
| Frustum: component-aware lateral and two-terminal-cap shading | Supported | Both terminal disks remain separate depth components; exact edge-on zero-area cap fills are omitted without inventing area |
| Finite open double shell: display and general occlusion | Supported with constraints | It expands once into two stable open single-nappe components |
| Finite open double shell: unified cutting-plane compositing | Supported with constraints | `CompositeQuadricSection3D` coordinates the two canonical nappes only when their complete projected contact set is one zero-dimensional point inside the shared-apex tolerance; remote points, nonzero segments, area overlap, and topology-family changes fail explicitly |
| One finite convex quadric and one cutting plane | Supported | This is the complete local section-compositor scope |
| Multiple intersecting quadrics and one cutting plane | Unsupported; explicit failure | No local multi-surface plane arrangement is guessed |
| Parallel projection | Supported | Orthographic and general affine-free parallel views use `ParallelView` |
| Perspective projection | Unsupported; explicit failure | The binding accepts only a three-dimensional `ParallelView`; a perspective/projective matrix is rejected before Scene ownership changes |
| Manim Cairo | Supported | The fixed-capacity production binding and pixel regressions target Cairo |
| Manim OpenGL | Unsupported; explicit failure | `QuadricOcclusion3D.attach()` rejects a non-Cairo renderer before Scene ownership changes |
| Cutting plane with a two-dimensional screen projection | Supported | The plane display patch must retain certifiable display area |
| Cutting plane projected completely edge-on | Unsupported; explicit failure | Frame preparation fails and an animated controller rolls back to its last committed frame |

“Multiple intersecting quadrics + one cutting plane” is a local section-plane
limitation. It does not remove the separate ability to place multiple strictly
disjoint quadrics in one global occlusion graph.

The open-double row is a narrow coordination exception, not a general
multi-surface arrangement. Each nappe is still solved by the ordinary
one-surface compositor. The coordinator certifies their projected interiors,
retains both local rear/front sheet pairs, replaces the two local outside
partitions with one area-conserving common plane partition, and publishes one
draw order. Degenerate polygon intersection is not discarded during this
certificate: the evidence records `contact_dimension`, `contact_extent`, the
maximum distance from the apex, and all retained contact points. Only
dimension zero inside the apex tolerance succeeds. A nonzero contact segment,
a remote point, or positive-area contact is an explicit failure.

An individual cone trim rim may project rank one, meaning it becomes a finite
line segment in an exact side view. That case is supported. The compositor
either certifies that the finite segment belongs to the outer projection proxy
or treats it as a finite arrangement boundary. This is different from the
cutting plane itself projecting edge-on, where no two-dimensional display
patch exists and v1 fails explicitly.

## Layer ownership

The renderer-neutral layer owns mathematical truth:

- analytic finite cone, cap, trim-rim, plane, and conic-section contracts;
- parallel projection proxies;
- visible and hidden curve intervals;
- semantic boundary visibility and solid/dashed conversion;
- local plane/surface partitioning and the far-to-near painter graph;
- topology events, capacity checks, and explicit numerical failures.

The recommended authoring entry point is `QuadricSection3D`. It derives the
real section boundary and the active cutting plane from one authority, then
delegates to the same renderer-neutral kernel and fixed-capacity Manim binding.
Scheduled ellipse/parabola/hyperbola handoff uses two stable lateral banks;
real cap chords keep separate semantic slots and always come from the current
plane, not from a neighboring critical plane.

For one authored `OPEN_DOUBLE`, the corresponding entry point is
`CompositeQuadricSection3D`. It preserves two child slot banks, paints the
shared plane once, and records the lineage from each nappe-owned physical curve
ID to its mathematical conic branch. Its current callback mode requires a
fixed curve-identity topology; a scheduled composite topology transition is
not part of this contract.

The Manim binding consumes those results. It owns preallocated Mobjects,
styles, Cairo paint order, in-place updates, and transactional rollback. It
does not recompute geometry from rendered pixels. Renderer-neutral support
therefore does not imply that every Manim renderer is supported: the v1
production binding is Cairo-only.

## Stable component contracts and versioned implementation evidence

The semantic contract freezes only public Provider contract revisions. The word
“v1” in `quadric-section-v1` names this support contract; it does not force every
underlying component ABI to have the same number.

| Component | Persisted contract revision |
| --- | --- |
| `quadric_geometry` | `tikz-native-contract:quadric_geometry/v1` |
| `quadric_visibility` | `tikz-native-contract:quadric_visibility/v2` |
| `quadric_manim` | `tikz-native-contract:quadric_manim/v1` |

The separate release manifest records the exact base commit, live
`source-sha256` render/cache revisions, reproducible wheel and sdist checksums,
the Cairo environment, and executable evidence names. A behavior-preserving
implementation change may update that manifest without pretending the semantic
support promise changed. The manifest is an external checksum sidecar and is
therefore excluded from the archives whose checksums it records. A persisted
author-data ABI revision changes only when existing saved data can no longer be
read safely.

## Canonical release fixtures

The release manifest fixes geometry, view matrices, expected semantic counts,
and evidence test names for these cases:

1. closed finite cone versus open finite shell semantics;
2. the oblique open-shell cutting-plane frame at offset `0.48`, including the
   former false-chord corridor;
3. exact side views of an open cone and an open frustum, where one or two trim
   rims project to finite rank-one segments;
4. a near-side-view rank switch that crosses the two-dimensional/rank-one
   threshold without replacing Manim objects or freezing a frame;
5. a closed frustum with separate lateral/two-cap masks, two real terminal rims,
   and two simultaneously active cap chords;
6. an open double shell whose two local section frames share one plane, one
   zero-dimensional apex-only contact certificate, two fixed slot banks, and
   one Cairo painter order;
7. a scheduled ellipse/parabola/hyperbola handoff that reaches a real end cap
   and keeps the current-plane cap chord in its stable semantic slot.

The Cairo acceptance baseline is
[`tests/baselines/quadric-section-v1-cairo.json`](../tests/baselines/quadric-section-v1-cairo.json).
It fixes two render profiles, stable screen probes, semantic color masks, and
minimum/maximum acceptance thresholds. It intentionally avoids a whole-image
hash: font, anti-aliasing, and Cairo build differences can change irrelevant
edge pixels. The regression instead checks the geometry-sensitive interiors,
the real yellow section ink, the cyan silhouette/trim-rim ink, component-aware
shading, the absence of an open-shell cap chord, and the complete `960x540`
offset-`0.48` failure corridor.

These files are release fixtures, not generated screenshots to be hand-edited
until a test passes. A deliberate baseline change requires a review that names
the changed support promise and provides a freshly inspected Cairo frame.

The release fixtures are complemented by a deterministic parameter-space
regression: 252 analytic section/visibility cases, 32 reviewed compositor
cases across critical section families and parallel views, and two scheduled
Manim topology sequences. The matrix freezes semantic digests and structural
invariants; it does not use random fuzzing in ordinary pull-request CI.

The complete `960x540` release frames, motion scans, videos, painter traces,
RGB probes, and fragment/ray CSV are generated by the separately scheduled
[extended Cairo acceptance workflow](extended-quadric-ci.md). Ordinary PR CI
retains the renderer-neutral matrix, low-resolution Cairo pixels, distribution
builds, and a fixed small animation sample.

## Explicit failure policy

V1 never guesses an image outside the matrix:

- the ordinary local section controller with zero or more than one expanded
  surface fails with `QuadricManimError`; an `OPEN_DOUBLE` must use the
  dedicated composite coordinator;
- the composite coordinator rejects non-canonical children, contact outside
  the shared apex, nonzero coincident contact segments, positive-area nappe
  projection overlap, and unscheduled curve-identity topology changes;
- a completely edge-on cutting plane fails during frame preparation; an
  attached animation retains its prior committed state;
- OpenGL attachment fails before the controller adds display slots to the
  Scene;
- perspective/projective input fails as an invalid parallel projection before
  the controller adds display slots to the Scene;
- uncertifiable error or capacity bounds remain hard failures.

## Change control

Bug fixes that preserve this matrix may keep the public support-contract ID;
their release manifest, Provider render/cache revisions, build checksums, and
inspected Cairo evidence are updated independently. Expanding a row from
unsupported to supported requires new
canonical geometry, renderer-neutral tests, and—if exposed through Manim—a
real Cairo regression. Perspective rendering, OpenGL production binding, and
multi-surface local section arrangements are follow-up projects, not implicit
v1 work.
