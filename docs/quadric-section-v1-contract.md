# Finite-cone section v1 support contract

- Status: **frozen release baseline**
- Contract ID: `quadric-section-v1`
- Frozen from `main`: `34f50e9b2ab66a969ae6a0648669be238ff943ae`

Machine-readable contract:
[`tests/fixtures/quadric-section-v1-contract.json`](../tests/fixtures/quadric-section-v1-contract.json)

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
| Finite cone frustum: section, clipping, ordinary occlusion | Supported | Uniform lateral/cap styling and automatic boundary visibility are supported |
| Frustum: independent component-aware shading for both terminal caps | Unsupported; explicit failure | The current component mask builder accepts only an apex-to-one-rim cone |
| Finite open double shell: display and general occlusion | Supported with constraints | It expands once into two stable open single-nappe components |
| Finite open double shell: unified cutting-plane compositing | Unsupported; explicit failure | The local compositor accepts exactly one finite convex surface, while the double shell expands to two |
| One finite convex quadric and one cutting plane | Supported | This is the complete local section-compositor scope |
| Multiple intersecting quadrics and one cutting plane | Unsupported; explicit failure | No local multi-surface plane arrangement is guessed |
| Parallel projection | Supported | Orthographic and general affine-free parallel views use `ParallelView` |
| Perspective projection | Unsupported | No perspective visibility or compositing contract exists in v1 |
| Manim Cairo | Supported | The fixed-capacity production binding and pixel regressions target Cairo |
| Manim OpenGL | Unsupported; explicit failure | `QuadricOcclusion3D.attach()` rejects a non-Cairo renderer before Scene ownership changes |
| Cutting plane with a two-dimensional screen projection | Supported | The plane display patch must retain certifiable display area |
| Cutting plane projected completely edge-on | Unsupported; explicit failure | Frame preparation fails and an animated controller rolls back to its last committed frame |

“Multiple intersecting quadrics + one cutting plane” is a local section-plane
limitation. It does not remove the separate ability to place multiple strictly
disjoint quadrics in one global occlusion graph.

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

The Manim binding consumes those results. It owns preallocated Mobjects,
styles, Cairo paint order, in-place updates, and transactional rollback. It
does not recompute geometry from rendered pixels. Renderer-neutral support
therefore does not imply that every Manim renderer is supported: the v1
production binding is Cairo-only.

## Frozen component identities

The v1 release bundle freezes the following existing Provider identities. The
word “v1” in `quadric-section-v1` names this release support contract; it does
not force every underlying component ABI to have the same number.

| Component | Persisted contract revision | Render/cache revision |
| --- | --- | --- |
| `quadric_geometry` | `tikz-native-contract:quadric_geometry/v1` | `source-sha256:830e9665b05c37e7d9182684f31896c3bfe294fc5b9a602486974cadd1a38450` |
| `quadric_visibility` | `tikz-native-contract:quadric_visibility/v2` | `source-sha256:1493b65088b72b28a3a6f763bf5655a0bb35d7f39e2b72adda9da541c061522e` |
| `quadric_manim` | `tikz-native-contract:quadric_manim/v1` | `source-sha256:31049bdcf16468c977874d2505db9188ffb5320849aabe4ddb361b9ea0058c3d` |

The executable contract compares these values with the live Provider. A
production change to an owned implementation file changes its render/cache
identity and therefore cannot silently continue to claim this exact baseline.
A persisted author-data ABI revision changes only when existing saved data can
no longer be read safely.

## Canonical release fixtures

The JSON contract fixes the geometry, view matrices, expected semantic counts,
and evidence test names for these cases:

1. closed finite cone versus open finite shell semantics;
2. the oblique open-shell cutting-plane frame at offset `0.48`, including the
   former false-chord corridor;
3. exact side views of an open cone and an open frustum, where one or two trim
   rims project to finite rank-one segments;
4. a near-side-view rank switch that crosses the two-dimensional/rank-one
   threshold without replacing Manim objects or freezing a frame.

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

## Explicit failure policy

V1 never guesses an image outside the matrix:

- a local section controller with zero or more than one expanded surface fails
  with `QuadricManimError`;
- a completely edge-on cutting plane fails during frame preparation; an
  attached animation retains its prior committed state;
- OpenGL attachment fails before the controller adds display slots to the
  Scene;
- a two-terminal frustum request for component-aware cone masks fails with
  `ProjectionProxyError`; callers may use the supported uniform surface style;
- uncertifiable error or capacity bounds remain hard failures.

## Change control

Bug fixes that preserve this matrix may keep the public support-contract ID,
but their Provider render/cache revisions and inspected Cairo evidence must be
updated normally. Expanding a row from unsupported to supported requires new
canonical geometry, renderer-neutral tests, and—if exposed through Manim—a
real Cairo regression. Perspective rendering, OpenGL production binding,
multi-surface local section arrangements, and independent two-terminal
frustum component shading are follow-up projects, not implicit v1 work.
