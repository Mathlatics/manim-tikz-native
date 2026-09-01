# Documentation

`manim-tikz-native` has three related, but independently usable, parts:

1. a restricted semantic TikZ compiler and source-project builder;
2. a TikZ-independent analytic geometry and automatic-occlusion kernel; and
3. parallel-camera, section-timeline, and fixed-capacity Cairo bindings for
   Manim.

Start with the workflow that matches what you are trying to build. The deep
contract documents are reference material; they are not required reading for
the first render.

[中文操作指南](user-guide.zh-CN.md) · [English user guide](user-guide.md) ·
[Public API](public-api.md) · [Contributing](../CONTRIBUTING.md) ·
[Maintainer guide](maintainer-guide.md)

## Start here

| Goal | Recommended starting point | Complete reference |
| --- | --- | --- |
| Install the project and render a first scene | [User guide](user-guide.md) | [Supported TikZ subset](supported-tikz.md) |
| Compile TikZ into native Manim objects | [First TikZ scene](user-guide.md#workflow-1-compile-tikz-into-native-manim-objects) | [Public API](public-api.md#compile-tikz) |
| Keep TikZ as the durable source of truth | [Source-project workflow](user-guide.md#workflow-2-use-a-source-authoritative-project) | [Source-authoritative projects](source-authoritative-projects.md) |
| Add hidden lines to a convex solid or open hinge | [Ordinary occlusion workflow](user-guide.md#workflow-3-add-automatic-occlusion-to-an-ordinary-manim-scene) | [Automatic 3D occlusion](automatic-occlusion.md) |
| Animate a sphere, cylinder, cone, or conic section | [Quadric workflow](user-guide.md#workflow-4-author-quadrics-and-conic-sections) | [Finite-quadric authoring workflow](quadric-authoring-workflow.md) |
| Build a Dandelin teaching scene | [Dandelin workflow](user-guide.md#workflow-5-choose-the-right-dandelin-path) | [Dandelin spheres v1](dandelin-spheres-v1.md) |
| Coordinate a moving camera and changing section | [Parallel-camera workflow](user-guide.md#workflow-6-use-semantic-parallel-camera-shots) | [Parallel camera and section sequences](parallel-camera-section-sequence.md) |
| Change the implementation or prepare evidence | [Maintainer guide](maintainer-guide.md) | [Fast and extended Cairo acceptance](extended-quadric-ci.md) |

## Concepts and architecture

- [Architecture](architecture.md) — package boundaries and the source-project
  data flow.
- [Geometry-kernel layers](geometry-kernel-layers.md) — the stable dependency
  direction: Geometry → Topology → Visibility → Compositor → Manim bindings.
- [Automatic 3D occlusion](automatic-occlusion.md) — closed convex solids,
  open panels, moving sections, copy handoff, and exact local transparency.
- [Public API](public-api.md) — Python entry points and versioned JSON bridges.

## TikZ and source projects

- [Supported TikZ subset](supported-tikz.md) — the accepted language and every
  explicit failure boundary.
- [Source-authoritative projects](source-authoritative-projects.md) — authored
  inputs, disposable derived output, cache keys, locking, and rollback.
- [Open-face unified compositing contract](open-face-unified-compositing-contract.md)
  and [Manim binding](open-face-unified-manim-binding.md).

## Quadrics, conic sections, and Dandelin geometry

- [Finite-quadric authoring workflow](quadric-authoring-workflow.md) — the
  shortest product path for scene authors.
- [Quadratic surfaces and conic-section occlusion](quadric-occlusion.md) —
  detailed geometry, visibility, compositor, and performance contracts.
- [Finite-cone section v1 support contract](quadric-section-v1-contract.md) —
  the authoritative support and explicit-rejection matrix.
- [Deterministic parameter sweep](quadric-section-parameter-sweep.md).
- [Dandelin spheres v1](dandelin-spheres-v1.md) — certified construction,
  curve visibility, teaching-layer authority, and physical-visibility limits.

## Cameras and coordinated playback

- [Parallel camera and section render sequences](parallel-camera-section-sequence.md)
  — semantic camera states, frame grids, topology banks, preflight, and
  transactional playback.
- [Parallel-camera examples](../examples/parallel_camera_views/README.md) and
  [shot-sequence examples](../examples/parallel_camera_shots/README.md).

## Testing, evidence, and release maintenance

- [Contributing](../CONTRIBUTING.md) — the short contributor entry point.
- [Maintainer guide](maintainer-guide.md) — test tiers, component revisions,
  pull-request evidence, and current-main release sidecars.
- [Fast and extended quadric acceptance](extended-quadric-ci.md) — CI tier
  ownership and the evidence bundle.
- [Release evidence sidecars](../release/README.md) — frozen release evidence
  versus the rolling current-main manifest.

## Examples

The [examples directory](../examples/) contains ordinary Manim scenes, source
projects, teaching lessons, and acceptance scenes. Good first choices are:

- [analytic TikZ ellipse animation](../examples/analytic_geometry_ellipse_demo/README.md);
- [convex sections](../examples/convex_sections/README.md);
- [quadric and conic-section demos](../examples/quadrics/README.md);
- [five classroom cone-section lessons](../examples/classroom_cone_sections/README.md);
- [Dandelin classroom scenes](../examples/classroom_dandelin_spheres/README.md);
- [Dandelin cone-to-cylinder switch](../examples/dandelin_cone_cylinder_switch/README.md);
- [source-authoritative camera shots](../examples/source_project_camera_shots/README.md).

## Authority and status

The latest packaged release is `v0.1.1`. The `main` branch also contains
reviewed but unreleased work listed under `Unreleased` in
[`CHANGELOG.md`](../CHANGELOG.md). Do not treat every `main` feature as part of
the published `v0.1.1` package.

For finite-cone claims, the support contract is authoritative. The frozen
historical release manifest and the rolling current-main manifest have
different lifetimes; see [the release sidecar guide](../release/README.md).
