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

The renderer-neutral `ParallelCameraState` additionally supports arbitrary
view directions, world-space targets, fixed screen anchors, zoom, and semantic
views normal to, oblique to, or exactly along an authored plane. Safe state
transitions keep every intermediate parallel camera invertible.

`ParallelCameraShot` and `ParallelCameraShotSequence` add named timing, holds,
cues, safe orbit transitions, and fixed-envelope zoom fitting on top of that
state. After a static shot reaches its exact endpoint,
`ParallelCameraTargetFollowController` can follow a moving world-space target
without allocating Mobjects per frame. Already-earlier target producers remain
before the follow driver, while explicitly marked quadric/composite consumers
are kept after it, giving deterministic same-frame camera/occlusion updates.
See the [semantic parallel-camera shots](examples/parallel_camera_shots/README.md).

For moving conic sections, one source-authoritative sequence can now coordinate
those camera shots with the cutting plane, two fixed topology banks, a finite
plane patch, semantic display state, and one real automatic-occlusion painter
graph.  See the
[parallel camera/section contract](docs/parallel-camera-section-sequence.md)
and its three
[Cairo acceptance scenes](examples/parallel_camera_section_rig_demo.py).

This project does **not** convert arbitrary TikZ and does not silently fall back
to SVG. Unsupported syntax is reported explicitly.

[中文说明](README.zh-CN.md) · [Public API](docs/public-api.md) ·
[Automatic occlusion](docs/automatic-occlusion.md) ·
[Quadrics and conic sections](docs/quadric-occlusion.md) ·
[Quadric quick start](docs/quadric-authoring-workflow.md) ·
[Finite-cone section v1 contract](docs/quadric-section-v1-contract.md) ·
[Fast and extended Cairo acceptance](docs/extended-quadric-ci.md) ·
[Classroom cone-section gallery](examples/classroom_cone_sections/README.md) ·
[Parallel camera example](examples/parallel_camera_views/README.md) ·
[Parallel camera/section sequences](docs/parallel-camera-section-sequence.md) ·
[Supported TikZ subset](docs/supported-tikz.md) ·
[Source-authoritative projects](docs/source-authoritative-projects.md)

## Cone-section quick start

One high-level rig derives the complete moving section, hidden dashes,
surface/plane order, and fixed-capacity slots. Ordinary authors describe a
mathematical plane action instead of wiring a tracker or painter band.

```python
from math import pi
from manim import Scene
from polyhedron_visibility.quadrics import (
    ConeSpec, QuadricSectionRig, SectionPlane,
)

class ConeSectionRigQuickStart(Scene):
    def construct(self):
        cone = ConeSpec("cone", (0, 0, -1.5), (0, 0, 1), pi / 6, (0, 4))
        plane = SectionPlane(
            "cut", (0, 0, -0.4), (0.45, 0, 1), u_axis=(0, 1, 0),
        )
        with QuadricSectionRig(
            self, surface=cone, section_id="cone-section", plane=plane,
            paint_policy="depth_aware_diagrammatic",
        ).session() as section:
            self.play(section.animate_plane_shift(0.6), run_time=2)
            self.play(section.animate_plane_rotation(
                axis=(0, 0, 1), angle=pi / 3, pivot=cone.apex,
            ), run_time=2)
```

Render the [checked-in scene](examples/quadrics/quadric_section_rig_quick_start.py)
at the matching Preview output size:

```bash
manim -r 480,270 --fps 15 \
  examples/quadrics/quadric_section_rig_quick_start.py \
  ConeSectionRigQuickStart
```

Pass `render_profile="preview"` for composition work or `"final"` and render
at 960x540, 30 fps for the classroom master. The separate
[Preview / Final / Release-Evidence workflow](docs/quadric-authoring-workflow.md)
also gives the one-call capacity-planning path.

## Explicit planar circles and ellipses in 3D

Ordinary TikZ `circle` and `ellipse` paths do not identify a world-space
supporting plane. The controlled 3D frontend therefore requires one explicit
static plane declaration before it accepts a spatial circle or ellipse:

```tex
\begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{base-plane}{O/U/V};
  \DrawSpaceCircle[draw=red,line width=1pt]
    {circle-a}{base-plane}{0,0}{1.5};
  \DrawSpaceEllipse[draw=blue]
    {ellipse-a}{base-plane}{0.5,-0.25}{2}{1};
\end{tikzpicture}
```

`O` is the plane origin, `O -> U` fixes the positive local-u direction and
curve phase, and `V` fixes the positive local-v side. Curve centers are the
two plane-local coordinates shown above. The distances `O -> U` and `O -> V`
certify orientation only; local centers and semi-axis lengths remain in world
coordinate units. This v1 syntax is static-safe and solid-stroke only: fill,
dashes, arcs, and animated O/U/V geometry fail
explicitly. The current embedded geometry-driver runtime also rejects a
picture containing these curves, even when their plane is unrelated to the
active driver; camera-only motion through `NativeManim3DRenderer` remains
supported. `NativeFixedViewRenderer` draws a rank-two projection as the true
affine ellipse and an exact edge-on projection as one finite segment.
`NativeManim3DRenderer` instead keeps the curve in its authored world plane so
the Manim camera can change without flattening the source geometry.
This syntax creates a standalone static curve; it does not invent a planar
disk or automatically enroll the curve in quadric occlusion/compositing.

Ordinary two-dimensional TikZ circles and ellipses remain unchanged. A 3D
circle or ellipse without `DeclareSpacePlane` is rejected rather than guessed.
See [the supported TikZ subset](docs/supported-tikz.md) and
[the public API](docs/public-api.md#explicit-tikz-planar-curves-in-3d).

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
- a `QuadricSectionRig` mathematical-action layer above `QuadricSection3D`:
  immutable plane states, exact pre-play critical-path certification, stable
  tracked curve slots, atomic author/display rollback, and automatic
  non-overlapping Scene painter bands support shift, axis-angle rotation, and
  unambiguous parallel `plane_to` actions without updater allocation; this
  first slice rejects cap-chord activation changes and all other paths needing
  topology-transition banks before playback;
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
and two-cap shading, rank-one trim rims and one-surface finite cutting-plane
patches in exact side views, parallel projection, one finite convex quadric with one
cutting plane, and the constrained two-nappe coordination described above. The fixed-capacity
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
