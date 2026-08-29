# manim-tikz-native

[![CI](https://github.com/Mathlatics/manim-tikz-native/actions/workflows/ci.yml/badge.svg)](https://github.com/Mathlatics/manim-tikz-native/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Manim 0.20.1](https://img.shields.io/badge/Manim-0.20.1-6c55a3.svg)](https://www.manim.community/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Compile a documented, restricted TikZ subset into semantic native Manim
objects. Keep named geometry editable, drive it with `ValueTracker`, and add
projection-aware hidden-line removal for closed convex polyhedra or articulated
open faces such as a dihedral angle.

The ordinary multi-projection camera defaults to the classroom cabinet-oblique
(`斜二测`) preset.  Its occlusion binding still takes the matching projection
explicitly, so calculation and display cannot silently diverge.  Quadrics and
conic-section Manim controllers default to a true orthographic isometric view,
which keeps a world-z cone axis vertical and does not add screen shear.
Compiled TikZ keeps its authored projection.

This project does **not** convert arbitrary TikZ and does not silently fall back
to SVG. Unsupported syntax is reported explicitly.

[中文说明](README.zh-CN.md) · [Public API](docs/public-api.md) ·
[Automatic occlusion](docs/automatic-occlusion.md) ·
[Quadrics and conic sections](docs/quadric-occlusion.md) ·
[Quadric quick start](docs/quadric-authoring-workflow.md) ·
[Finite-cone section v1 contract](docs/quadric-section-v1-contract.md) ·
[Fast and extended Cairo acceptance](docs/extended-quadric-ci.md) ·
[Classroom cone-section gallery](examples/classroom_cone_sections/README.md) ·
[Supported TikZ subset](docs/supported-tikz.md) ·
[Source-authoritative projects](docs/source-authoritative-projects.md)

## Cone-section quick start

One high-level controller derives the complete moving section, hidden dashes,
surface/plane order, and fixed-capacity slots. Ordinary authors do not need to
manage curve IDs, cap-chord IDs, fragments, or painter bands.

```python
from math import pi
from manim import Scene, ValueTracker, linear
from polyhedron_visibility.quadrics import (
    ConeSpec, QuadricManimStyle, QuadricSection3D, SectionPlane,
)

class ConeSectionQuickStart(Scene):
    def construct(self):
        progress = ValueTracker(0)
        cone = ConeSpec("cone", (0, 0, -1.5), (0, 0, 1), pi / 6, (0, 4))
        def plane():
            return SectionPlane(
                "cut", (0, 0, -1 + 2.7 * progress.get_value()),
                (0.65, 0, 1), u_axis=(0, 1, 0),
            )
        QuadricSection3D(
            self, surface=cone, section_id="cone-section", plane=plane,
            paint_policy="depth_aware_diagrammatic", render_profile="preview",
            style=QuadricManimStyle(surface_fill_opacity=.62),
        ).attach()
        self.play(progress.animate.set_value(1), run_time=4, rate_func=linear)
```

Render the [checked-in scene](examples/quadrics/quadric_section_quick_start.py)
at the matching Preview output size:

```bash
manim -r 480,270 --fps 15 \
  examples/quadrics/quadric_section_quick_start.py ConeSectionQuickStart
```

Change `render_profile` to `"final"` and render at 960x540, 30 fps for the
classroom master. The separate
[Preview / Final / Release-Evidence workflow](docs/quadric-authoring-workflow.md)
also gives the one-call capacity-planning path.

## What it provides

- restricted TikZ → native `Line`, `Polygon`, `Circle`, `Ellipse`, `Dot`,
  `Tex`, `MathTex`, arrow and angle-marker objects;
- stable semantic object IDs and named geometric relationships;
- 2D and 3D geometry rigs driven by ordinary Manim trackers;
- fixed-view and local-camera 3D projection;
- automatic solid/hidden line splitting for closed convex polyhedra;
- automatic intersections between a closed convex solid and independent
  straight lines, including stable entry/exit markers;
- a moving mathematical plane that derives point, segment, or polygon
  cross-sections; its finite display patch auto-fits the complete solid by
  default and joins the same global hidden-line calculation;
- opt-in exact local far-to-near compositing for one transparent cutting plane
  intersecting one closed convex solid: the plane, section, and crossed solid
  faces are split before sorting, so one whole plane never has to sit entirely
  in front of or behind the solid;
- optional didactic face shading with near-color retention, distant-face fog,
  continuous orientation tinting, and visible silhouette emphasis for closed
  convex solids, while hidden dashes remain unchanged;
- automatic line occlusion and face ordering for finite open convex faces and
  articulated hinges;
- analytic finite spheres, capped cylinders, closed single cones, open single
  cone shells, and finite open double shells under parallel projection; cone
  caps and open trim rims remain different geometry, while fixed lateral/cap
  paint slots give an open mouth a real one-sheet appearance; including plane
  sections, front-solid/hidden-dashed
  conic arcs, projected curve/curve depth ordering, rotating-plane topology
  schedules, automatic ellipse/parabola/hyperbola Manim handoff, strictly
  separated multi-solid painter graphs, and seam-free local compositing of one
  cutting plane through one quadric using rear/between/front plane regions and
  two smooth surface sheets;
- a high-level `QuadricSection3D` authoring facade that derives the complete
  finite section from the same live plane used for compositing, automatically
  reserves potential cap-chord slots, and delegates topology changes to the
  existing fixed-capacity transition controller; during a conic-family
  cross-fade, the two lateral traces use separate banks while each real end-cap
  chord keeps one stable semantic identity on the current cutting plane;
  scheduled rigid motion may opt into one certified full-motion display-patch
  envelope instead of refitting the visible rectangle at every frame;
  `show_plane=False` deliberately disables the complete plane compositor, not
  only the visible patch;
- explicit `QUADRIC_PREVIEW_PROFILE` and `QUADRIC_FINAL_PROFILE` recipes keep
  480x270 composition work separate from 960x540 Cairo acceptance, while
  `QuadricCapacityPlanner` scans the declared animation frames and analytic
  schedule knots through the real controller, publishes fragment/dash/plane/
  ray peaks, and generates a compact `QuadricManimLimits` without claiming a
  continuous bound for unscanned progress values;
- a sibling `CompositeQuadricSection3D` coordinator for one finite
  `OPEN_DOUBLE`: it expands the authored double shell into its two canonical
  nappes, reuses the ordinary one-surface section solver twice, paints the
  shared plane once, preserves two fixed slot banks, and records physical-to-
  mathematical branch lineage; this coordinated path requires the projected
  nappe contact set to be one zero-dimensional point inside the shared-apex
  tolerance; a remote point, a nonzero contact segment, or area overlap fails
  explicitly;
- a reusable source-to-copy identity handoff: a whole solid or any registered
  face/edge subset keeps explicit semantic lineage, lets the copy own exactly
  coincident pixels, and fades only the paired source primitives back in as
  the two geometries separate;
- one extracted dihedral copied from two adjacent faces of a closed convex
  solid: the source solid and the independently translated/rotated copy share
  one global line-occlusion solve; it uses that generic copy handoff before
  intersecting translucent faces are split and ordered locally;
- synchronized base-plane rotation for that solid/copy assembly: one selected
  source face can be turned into a horizontal bottom face after extraction,
  while each independently placed entity rotates about its own translated
  geometric center;
- readable native Manim source generation and versioned JSON bridges;
- strict, component-level compatibility identities for cached integrations.

## Finite-cone section v1 boundary

The frozen [finite-cone section v1 contract](docs/quadric-section-v1-contract.md)
is the authority for release claims. In short, v1 supports closed finite single
cones, open finite single shells, frustum sections with component-aware lateral
and two-cap shading, rank-one trim rims in exact side views, parallel
projection, one finite convex quadric with one non-edge-on cutting plane, and
the constrained two-nappe coordination described above. The fixed-capacity
Manim production binding is Cairo-only.

V1 explicitly rejects perspective projection and an OpenGL production binding,
general local compositing of multiple intersecting quadrics, or an open-double
view in which the two nappe projections meet anywhere beyond one certified
shared-apex point. Scheduled topology-family changes for the composite
controller are also not yet supported. The documented explicit-failure paths
are part of the contract; unsupported combinations are not approximated
silently.
The stable support promise lives in
[`tests/fixtures/quadric-section-v1-contract.json`](tests/fixtures/quadric-section-v1-contract.json).
The exact implementation commit, component digests, build checksums, and
evidence mapping live in
[`release/quadric-section-v1-release-manifest.json`](release/quadric-section-v1-release-manifest.json),
while Cairo acceptance thresholds live in
[`tests/baselines/quadric-section-v1-cairo.json`](tests/baselines/quadric-section-v1-cairo.json).
The [layered CI guide](docs/extended-quadric-ci.md) explains which checks stay
on every pull request and which 960x540 frames, motion scans, videos, and
machine-readable painter evidence run nightly or for releases.

## Requirements

- Python 3.11 or newer;
- Manim Community 0.20.1;
- XeLaTeX with `standalone`, `fontspec`, `xeCJK`, `unicode-math`, and TikZ;
- the TeX Live Fandol and Latin Modern fonts for the default template;
- FFmpeg for video rendering.

On macOS, a full TeX Live installation already includes the default fonts. On
Debian or Ubuntu, the CI setup installs `texlive-xetex`,
`texlive-latex-extra`, `texlive-lang-chinese`, and
`texlive-fonts-recommended`, plus the separate `dvisvgm` package used by
Manim's TeX renderer and `poppler-utils` for PDF measurement.

## Install

```bash
git clone https://github.com/Mathlatics/manim-tikz-native.git
cd manim-tikz-native
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Check the compiler and its component identities:

```bash
tikz-native health
tikz-native-rig-2d health
tikz-native-rig-3d health
tikz-native-source-v3 health
```

To keep TikZ as the durable source of truth while regenerating disposable
ShapeAsset, compositing, and Manim-source outputs, use the source-project CLI:

```bash
tikz-native-project build project.json
tikz-native-project status project.json
tikz-native-project rebuild project.json
tikz-native-project clean project.json
```

See [source-authoritative-projects.md](docs/source-authoritative-projects.md)
for the manifest, cache, transaction, and fail-closed rules.

## First TikZ scene

```python
from pathlib import Path

from manim import Scene

from tikz_native import compile_document
from tikz_native.manim_renderer import NativeManimRenderer


class NativeTikzScene(Scene):
    def construct(self):
        document = compile_document(Path("figure.tex"))
        picture = document.pictures[0]
        if picture.unsupported:
            raise ValueError("Unsupported TikZ: " + "; ".join(picture.unsupported))

        figure = NativeManimRenderer(scene_unit_per_cm=1.0).render(picture)
        self.add(figure.group)
        self.wait()
```

Render it with normal Manim:

```bash
manim -pql scene.py NativeTikzScene
```

The compiler keeps named coordinates and relationships. It does not embed the
original TikZ as one opaque SVG object.

## Automatic occlusion demos

Closed convex polyhedron:

```bash
manim -pql examples/polyhedron_visibility/cube_auto_occlusion.py \
  CubeAutoOcclusionDemo
```

Articulated open faces / dihedral angle:

```bash
manim -pql examples/open_face_visibility/dihedral_auto_occlusion.py \
  DihedralAutoOcclusionDemo
```

Quadratic surfaces and conic sections:

```bash
manim -pql examples/quadrics/quadric_occlusion_demo.py \
  MovingSphereSectionDemo ObliqueCylinderSectionDemo \
  ConeSectionFamiliesDemo ConeSectionTopologyTransitionDemo \
  GlobalQuadricOcclusionDemo
```

Five paced high-school cone-section lessons, with parameter notes, teacher
prompts, and reviewed keyframes:

```bash
manim -ql --fps 8 \
  examples/classroom_cone_sections/classroom_cone_sections.py \
  ConicFamilyTransitionLesson ClosedVsOpenConeLesson \
  HiddenCurvePoliciesLesson ProjectionDegenerationLesson \
  CapChordTopologyLesson
```

See the [classroom gallery guide](examples/classroom_cone_sections/README.md)
for individual preview and release commands.

Independent line and moving plane sections:

```bash
manim -pql examples/convex_sections/convex_sections_demo.py \
  LineThroughCubeDemo
manim -pql examples/convex_sections/convex_sections_demo.py \
  MovingPlaneSectionDemo
manim -pql examples/convex_sections/convex_sections_demo.py \
  CombinedSectionAndLineDemo
manim -pql examples/convex_sections/convex_sections_demo.py \
  AccurateTransparentSectionDemo

manim -pql examples/convex_sections/other_convex_solids_demo.py \
  TetrahedronSectionDemo TriangularPrismSectionDemo \
  SquarePyramidSectionDemo OctahedronSectionDemo

manim -pql \
  examples/derived_dihedral_extraction/derived_dihedral_extraction_demo.py \
  RectangularBoxDihedralDemo TetrahedronDihedralDemo \
  SquarePyramidDihedralDemo RectangularBoxDihedralRoundTripDemo
```

The round-trip scene separates the highlighted copy, rotates the shared
assembly, and then returns the copy to exact coincidence. It exercises the
same semantic identity handoff in reverse, so the final frame again contains
one visible representation rather than two alpha-blended copies. The
tetrahedron scene isolates the extraction handoff; the rectangular-box and
square-pyramid scenes also demonstrate synchronized base-plane rotation.

For ordinary Manim scenes, register stable vertices, maximal convex faces, and
semantic `Line` objects through `OcclusionScene3D` or `OpenFaceScene3D`. The
module updates preallocated visible and dashed slots in place during
`scene.play()` and restores the original source objects when the session ends.
Use `ConvexSectionScene3D` when one closed convex solid also needs independent
straight-line intersections or one moving infinite cutting plane. Its authored
half-width and half-height are minimum display sizes; the visible patch expands
automatically with 15% margin and never shrinks during one attached session.
Pass `accurate_transparency=True` and bind one native fill-only `Polygon` for
every solid face when the cutting plane and solid faces must be split and
composited in exact local depth order under parallel projection.
Use `ExtractedDihedralScene3D` when two adjacent faces of one closed solid must
be highlighted in place and then moved out as one independent teaching
object. The initial coincident source faces/edges are handed off to the
highlighted copy without double alpha blending; after motion begins, both
entities take part in global hidden-line removal. With
`accurate_transparency=True`, crossings between their translucent faces are
split only where the finite polygons truly intersect before far-to-near
sorting. Consecutive triangles from the same authored face and the same valid
depth position share one compound fill pass, so internal triangulation edges
do not appear in the rendered face. This option now also enables unified
compositing by default: visible and dashed stroke spans are split at local
line/face depth exchanges and projected line/line crossings, then face batches
and stroke fragments share one deterministic far-to-near Cairo paint order.
Thus the hidden-line calculation and the pixels painted by Manim use the same
depth evidence instead of placing every line above every transparent face.
Call `base_plane_rotation(face_id)` and pass its tracker-driven transform as
`global_transform_provider` to rotate the solid and already separated copy as
one assembly without losing automatic line or transparent-face occlusion.
Use `DepthCuedAutoOcclusion3D`, or provide a native `Polygon` for every face to
`ConvexSectionScene3D`, when the scene also needs face-orientation shading,
depth opacity, nearby warm/cool tinting, and automatic silhouette emphasis.

See [automatic-occlusion.md](docs/automatic-occlusion.md) for the supported
geometry and fail-closed rules.

## Test

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m build
python -m twine check dist/*
```

Two expensive motion-render tests are opt-in by design; the ordinary test suite
still exercises real Cairo renders for the automatic-occlusion bindings.

## Project boundaries

This repository contains the reusable compiler, Manim runtime, algorithms,
bridges, schemas, and examples. It deliberately does not contain a PowerPoint
editor, browser UI, timeline model, ShapeAsset/ShapeState storage, or preview
cache. It may generate disposable ShapeAsset JSON inside a source project's
owned output directory, but it does not provide an application ShapeAsset
database. Applications can consume the package through its Python API, CLI, or
JSON bridges.

## Status

Version `0.1.1` is an alpha release. The public contracts are versioned and the
compiler fails closed, but the accepted TikZ language is intentionally smaller
than TikZ itself. Please report a minimal `.tex` example when requesting new
syntax.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
