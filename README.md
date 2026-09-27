# manim-tikz-native

[![CI](https://github.com/Mathlatics/manim-tikz-native/actions/workflows/ci.yml/badge.svg)](https://github.com/Mathlatics/manim-tikz-native/actions/workflows/ci.yml)
[![Python 3.11 / 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Manim 0.20.1](https://img.shields.io/badge/Manim-0.20.1-6c55a3.svg)](https://www.manim.community/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A semantic geometry toolkit for mathematical Manim scenes. It compiles a
documented TikZ subset into editable native Manim objects, and supplies
analytic parallel-projection occlusion for explicitly registered polyhedra,
quadrics, conic sections, and teaching constructions.

The project preserves stable object identities and fails closed when geometry
or syntax cannot be certified. It does **not** flatten unsupported input to an
SVG, bitmap, or guessed `z_index` result.

[中文说明](README.zh-CN.md) · [Documentation](docs/README.md) ·
[User guide](docs/user-guide.md) · [Public API](docs/public-api.md) ·
[Examples](examples/) · [Contributing](CONTRIBUTING.md)

> **Release status:** the latest packaged release is
> [`v0.1.1`](https://github.com/Mathlatics/manim-tikz-native/releases/tag/v0.1.1).
> The GitHub `main` branch also contains reviewed but unreleased work listed in
> [`CHANGELOG.md`](CHANGELOG.md). Do not assume that every `main` capability is
> present in the published `v0.1.1` package.

![Ellipse, parabola, and hyperbola transition](https://raw.githubusercontent.com/Mathlatics/manim-tikz-native/main/examples/classroom_cone_sections/gallery/contact-sheets/conic_family_transition.png)

## What the project contains

### 1. Semantic TikZ compilation

- Compile a controlled 2D/3D TikZ subset into native `Line`, `Polygon`,
  `Circle`, `Ellipse`, `Dot`, `Tex`, `MathTex`, arrows, and angle markers.
- Preserve named coordinates, paths, semantic IDs, and authored relationships
  for later animation.
- Render fixed-view 3D TikZ or keep supported objects in world space for a
  Manim camera.
- Build disposable ShapeAsset, compositing, camera-shot, and generated-source
  outputs from a source-authoritative project.

### 2. Analytic geometry and automatic occlusion

- Closed convex polyhedra, finite open panels, articulated hinges, free lines,
  moving plane sections, source/copy handoff, and extracted dihedrals.
- Finite spheres, closed/open cylinders, closed cones, open single cone shells,
  and finite open double shells.
- Analytic circle, ellipse, parabola, hyperbola, tangent-point, silhouette,
  trim-rim, contact-circle, and section-boundary visibility.
- Deterministic far-to-near compositor graphs for supported transparent
  teaching geometry under parallel projection.

### 3. Parallel-camera and fixed-capacity Manim runtime

- `ParallelCameraState` with direction, world target, screen anchor, zoom, and
  semantic views normal to, oblique to, or exactly along a plane.
- Named camera shots, safe transitions, target following, section timelines,
  topology banks, viewport transactions, and constrained global multi-Rig
  coordination.
- Preallocated Cairo slots, stable dash phase, last-good-frame retention, and
  transactional restoration instead of updater-time Mobject replacement.

The stable architectural direction is:

```text
Geometry -> Topology -> Visibility -> Compositor -> Manim bindings
```

The geometry, topology, visibility, and compositor layers are renderer-neutral.
See [Architecture](docs/architecture.md) and
[Geometry-kernel layers](docs/geometry-kernel-layers.md).

## Choose the right entry point

| Goal | Recommended entry | Start here |
| --- | --- | --- |
| Compile TikZ into native Manim objects | `compile_document()` + a native renderer | [First TikZ workflow](docs/user-guide.md#workflow-1-compile-tikz-into-native-manim-objects) |
| Rebuild disposable outputs from authored TikZ | `tikz-native-project` | [Source projects](docs/source-authoritative-projects.md) |
| Hide lines on a closed convex solid | `OcclusionScene3D` | [Automatic occlusion](docs/automatic-occlusion.md) |
| Model open panels or a hinge | `OpenFaceScene3D` | [Open-face example](examples/open_face_visibility/README.md) |
| Add one moving plane, optionally with free lines, to a convex solid | `ConvexSectionScene3D` | [Convex-section examples](examples/convex_sections/README.md) |
| Animate one finite quadric section | `QuadricSectionRig` | [Quadric quick start](docs/quadric-authoring-workflow.md) |
| Occlude separated quadrics and curves without a cutting plane | `QuadricOcclusion3D` | [Quadric occlusion](docs/quadric-occlusion.md) |
| Drive a lower-level or topology-changing quadric section | `QuadricSection3D` / `QuadricSectionTransition3D` | [Quadric authoring](docs/quadric-authoring-workflow.md) |
| Coordinate an `OPEN_DOUBLE` section | `CompositeQuadricSection3D` | [Quadric demos](examples/quadrics/README.md) |
| Build Dandelin geometry | `compute_dandelin_construction()` | [Dandelin contract](docs/dandelin-spheres-v1.md) |
| Add live Dandelin curve visibility | `DandelinOcclusion3D` | [Dandelin guide](docs/user-guide.md#workflow-5-choose-the-right-dandelin-path) |
| Play semantic camera motion in Manim | `MultiProjectionCamera` + `play_parallel_camera_shot_sequence()` | [Camera examples](examples/parallel_camera_shots/README.md) |

The complete Python and JSON surface is documented in
[Public API](docs/public-api.md).

## Five-minute start

Project CI verifies Python 3.11 and 3.12; the release evidence environment is
Python 3.12.13. The package pins Manim Community 0.20.1.

```bash
git clone https://github.com/Mathlatics/manim-tikz-native.git
cd manim-tikz-native
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The machine also needs Cairo/Pango, FFmpeg, XeLaTeX, TikZ, `dvisvgm`, Fandol
and Latin Modern fonts. See the [complete installation guide](docs/user-guide.md#1-choose-a-supported-environment).

Check the versioned frontends:

```bash
tikz-native health
tikz-native-rig-2d health
tikz-native-rig-3d health
tikz-native-source-v3 health
```

Render the checked-in moving-cone quick start:

```bash
python -m manim --renderer cairo -r 480,270 --fps 15 \
  --media_dir media/quadric-quickstart \
  examples/quadrics/quadric_section_rig_quick_start.py \
  ConeSectionRigQuickStart
```

To compile a first TikZ file instead, follow the
[copy-ready two-file example](docs/user-guide.md#workflow-1-compile-tikz-into-native-manim-objects).

## Quadric section quick start

Ordinary authors describe a mathematical plane action. The high-level Rig
derives the complete finite section, visible/hidden strokes, plane/surface
order, fixed slots, and rollback state:

```python
from math import pi

from manim import Scene
from polyhedron_visibility.quadrics import ConeSpec, QuadricSectionRig, SectionPlane


class ConeSectionLesson(Scene):
    def construct(self):
        cone = ConeSpec("cone", (0, 0, -1.5), (0, 0, 1), pi / 6, (0, 4))
        plane = SectionPlane(
            "cut", (0, 0, -0.4), (0.45, 0, 1), u_axis=(0, 1, 0)
        )
        with QuadricSectionRig(
            self,
            surface=cone,
            section_id="cone-section",
            plane=plane,
            paint_policy="depth_aware_diagrammatic",
        ).session() as section:
            self.play(section.animate_plane_shift(0.6), run_time=2)
            self.play(
                section.animate_plane_rotation(
                    axis=(0, 0, 1), angle=pi / 3, pivot=cone.apex
                ),
                run_time=2,
            )
```

Use `render_profile="preview"` with 480×270 at 15 fps for composition and
`render_profile="final"` with 960×540 at 30 fps for the classroom master.
The full Preview / Final / Release-Evidence workflow is in
[Finite-quadric authoring](docs/quadric-authoring-workflow.md).

## Dandelin authority boundaries

Dandelin support has several deliberately different presentation contracts:

| Path | Authoritative result | Not claimed |
| --- | --- | --- |
| `DandelinSection3D` | Analytic construction and ordinary cone section | Physical sphere/cone occlusion; helpers are a top teaching overlay. |
| `DandelinOcclusion3D` | Live callable-camera visible and hidden curve fragments in fixed Cairo slots | Physical surface visibility or cutting-plane fill. |
| Depth-aware spatial TikZ | Frozen-view visible and hidden boundary fragments | Motion, camera shots, or physical surface visibility. |
| `depth_aware_teaching_transparent` | Static spatial, single-nappe circle/ellipse/parabola curve fragments plus certified classroom ordering; contact circles are required | Motion, other views, optical transparency, or opaque hidden surfaces. |
| Nested-tangent scene coordinator | One cone/cylinder mother, one tangent plane, exactly two tangent spheres, and registered boundaries | Arbitrary multi-object occlusion. |

The [cone-to-cylinder switch](examples/dandelin_cone_cylinder_switch/README.md)
uses the narrow nested-tangent path. It registers plane edges, mother-surface
boundaries, sphere silhouettes, contact circles, the true section, and the
teaching axis in one analytic fragment graph. Its translucent fills remain a
teaching-layer model, and its final Manim assembly is an example rather than a
general `DandelinConeCylinderSwitch3D` facade.

## Important boundaries

- This is not an arbitrary TikZ converter. Unsupported syntax is reported and
  never replaced with SVG or raster output.
- Automatic occlusion requires explicit stable topology; it cannot infer a
  trustworthy model from an arbitrary `VGroup` or mesh.
- The production automatic-occlusion and quadric bindings target Cairo and
  parallel projection. Perspective and OpenGL parity are not v1 claims.
- Ordinary global multi-quadric ordering requires strict separation. The
  supported open-double and Dandelin nested-tangent arrangements use explicit
  narrow coordinators; they are not a general intersecting-surface solver.
- Transparent teaching-layer order is not a physical lighting, refraction, or
  order-independent-transparency simulation.
- The repository contains renderer-neutral motion/section timeline contracts,
  but no browser/PPT editor, application-level timeline database, ShapeAsset
  database, or preview-cache service.

See [Supported TikZ](docs/supported-tikz.md),
[Automatic occlusion](docs/automatic-occlusion.md), and the
[finite-cone v1 contract](docs/quadric-section-v1-contract.md) for exact
support and rejection rules.

## Examples

| Example | What it demonstrates |
| --- | --- |
| [Analytic ellipse](examples/analytic_geometry_ellipse_demo/README.md) | Semantic TikZ objects driven by a geometry rig. |
| [Convex sections](examples/convex_sections/README.md) | Free lines, moving sections, and exact plane/solid transparency. |
| [Derived dihedrals](examples/derived_dihedral_extraction/README.md) | Copy handoff, separation, unified stroke/fill order, and round trip. |
| [Quadric demos](examples/quadrics/README.md) | Spheres, cylinders, cones, conic families, and topology transitions. |
| [Classroom cone sections](examples/classroom_cone_sections/README.md) | Five paced teaching scenes with reviewed keyframes. |
| [Dandelin classroom lesson](examples/classroom_dandelin_spheres/README.md) | Static ellipse/parabola/hyperbola teaching overlays. |
| [Dandelin cone/cylinder switch](examples/dandelin_cone_cylinder_switch/README.md) | Two tangent spheres under a continuously changing mother surface. |
| [Semantic camera shots](examples/parallel_camera_shots/README.md) | Named parallel views, exact side view, target following, and rollback. |
| [Source-project camera shots](examples/source_project_camera_shots/README.md) | Authored camera JSON and disposable derived output. |

## Test and contribute

Install `.[dev]` for package-building tools. During development, use the
reviewed test tier that matches the change; the three tier commands below are
alternatives, not a mandatory sequence:

```bash
python scripts/run_ci_test_tier.py core
python scripts/run_ci_test_tier.py cairo-smoke
python scripts/run_ci_test_tier.py all
python -m build
python -m twine check dist/*
```

High-resolution frames, full motion scans, reproducible packages, and MP4
evidence run separately. Read [Contributing](CONTRIBUTING.md), the
[Maintainer guide](docs/maintainer-guide.md), and
[Extended Cairo acceptance](docs/extended-quadric-ci.md) before changing
runtime behavior or release evidence.

## License and security

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

TikZ compilation invokes external TeX and Manim tools. Do not process untrusted
source with unrestricted shell escape or in an environment containing secrets;
see [SECURITY.md](SECURITY.md).
