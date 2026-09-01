# User guide

This guide takes a new checkout from installation to the main supported
workflows. It uses checked-in examples wherever possible so every command has a
known file and Scene class behind it.

[中文版本](user-guide.zh-CN.md) · [Documentation index](README.md) ·
[Public API](public-api.md)

## 1. Choose a supported environment

The package metadata requires Python 3.11 or newer, but project CI currently
verifies Python 3.11 and 3.12. Release evidence is produced with Python
3.12.13. Use Python 3.12 for the most reproducible local setup; a newer system
`python3` is not automatically a tested environment.

The project pins Manim Community `0.20.1`. It also needs:

- Cairo and Pango dependencies required by Manim;
- FFmpeg and `ffprobe` for video output and verification;
- XeLaTeX, TikZ, `dvisvgm`, Fandol fonts, and Latin Modern fonts for text;
- Poppler tools for the full test and evidence workflow.

The [Manim installation guide](https://docs.manim.community/en/stable/installation.html)
explains the platform-specific Cairo/Pango setup. On macOS, a typical base is:

```bash
brew install cairo pango pkg-config ffmpeg poppler
```

Install a TeX distribution that provides XeLaTeX, TikZ, Fandol, and Latin
Modern; a full [MacTeX](https://tug.org/mactex/) installation includes them.

On Debian or Ubuntu, the project CI uses:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  dvisvgm ffmpeg libcairo2-dev libpango1.0-dev pkg-config poppler-utils \
  texlive-fonts-recommended texlive-lang-chinese texlive-latex-extra \
  texlive-xetex
```

## 2. Install from a checkout

```bash
git clone https://github.com/Mathlatics/manim-tikz-native.git
cd manim-tikz-native
export TIKZ_NATIVE_REPO="$PWD"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

If `python3.12` is not the command used by your installation, create the
environment with a Python whose `python --version` reports 3.11 or 3.12.
Contributors who also need package-building tools should install `.[dev]`
instead of `.[test]`.

Check the external toolchain:

```bash
python -m manim --version
printf 'n\n' | python -m manim checkhealth
command -v xelatex
command -v dvisvgm
command -v ffmpeg
command -v ffprobe
command -v pdftoppm
kpsewhich FandolSong-Regular.otf
kpsewhich latinmodern-math.otf
```

Then check the four versioned Bridge frontends. Each command writes a JSON
health document:

```bash
tikz-native health
tikz-native-rig-2d health
tikz-native-rig-3d health
tikz-native-source-v3 health
```

Finally, verify imports from outside the repository. This catches a broken
editable installation that appears to work only because the checkout is the
current directory:

```bash
cd "$TIKZ_NATIVE_REPO"
(cd /tmp && "$TIKZ_NATIVE_REPO/.venv/bin/python" -c \
  'import tikz_native, polyhedron_visibility; print("imports ok")')
```

The workflows below reuse `TIKZ_NATIVE_REPO`. If you open a new terminal,
activate the environment and export that variable to the checkout's absolute
path before continuing.

## Workflow 1: compile TikZ into native Manim objects

Create a separate working directory and copy the packaged starter figure:

```bash
quickstart_directory="$(mktemp -d /tmp/manim-tikz-quickstart.XXXXXX)"
cp "$TIKZ_NATIVE_REPO/tikz_native/examples/native_friendly_figure.tex" \
  "$quickstart_directory/figure.tex"
cd "$quickstart_directory"
```

Create `scene.py`:

```python
from pathlib import Path

from manim import Scene
from tikz_native import NativeManimRenderer, compile_document


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

Render it with Cairo. `-p` is intentionally omitted so the command also works
on a headless machine; add it locally if you want Manim to open the result.

```bash
python -m manim --renderer cairo -ql \
  --media_dir media/quickstart scene.py NativeTikzScene
cd "$TIKZ_NATIVE_REPO"
```

The figure contains native Manim objects and an ID-to-object mapping. It is not
one imported SVG. For a 3D TikZ picture, choose between
`NativeFixedViewRenderer` (ordinary 2D Mobjects in the authored projection) and
`NativeManim3DRenderer` (world-space Mobjects for a Manim 3D camera).

The accepted language is deliberately smaller than TikZ. Read
[Supported TikZ subset](supported-tikz.md) before introducing clips, arbitrary
paths, decorations, or custom macros.

## Workflow 2: use a source-authoritative project

A source project keeps TikZ, optional motion/camera authoring, and render intent
as durable inputs. ShapeAsset JSON, compositing plans, generated source, and the
build manifest are disposable derived output.

Try the checked-in camera-shot project outside the repository:

```bash
source_project_directory="$(mktemp -d /tmp/tikz-native-camera-demo.XXXXXX)"
cp -R "$TIKZ_NATIVE_REPO/examples/source_project_camera_shots/." \
  "$source_project_directory/"
(
  cd "$source_project_directory"
  tikz-native-project build project.json
  tikz-native-project status project.json
  tikz-native-project rebuild project.json
  tikz-native-project clean project.json
)
```

The command lifecycle is:

| Command | Effect |
| --- | --- |
| `build` | Build missing/stale nodes and reuse fresh nodes. |
| `status` | Read only; report fresh, stale, or missing nodes. |
| `rebuild` | Rebuild every derived node without using cache hits. |
| `clean` | Remove only the validated build-owned derived directory. |

`status` exits with code 0 when fresh and code 1 when output is missing or
stale; both are valid JSON results. Invalid input or a safety/build failure is
code 2. The ownership marker and transactional publication rules are described
in [Source-authoritative projects](source-authoritative-projects.md).

`tikz-native-source-v3` is a strict machine-facing source-generation Bridge.
Its `run` operation requires a versioned JSON request, source hash, and current
Provider revision. It returns definitions and helpers, not a self-contained
Manim `Scene`. Ordinary authors should use a source project or a checked-in
host scene instead of treating Source v3 output as directly renderable.

## Workflow 3: add automatic occlusion to an ordinary Manim scene

Choose the authoring facade that matches the geometry:

| Geometry | Recommended facade | Example |
| --- | --- | --- |
| Closed convex polyhedron | `OcclusionScene3D` | `examples/polyhedron_visibility/cube_auto_occlusion.py` |
| Finite open panels or a hinge | `OpenFaceScene3D` | `examples/open_face_visibility/dihedral_auto_occlusion.py` |
| Closed solid plus one cutting plane and optional free lines | `ConvexSectionScene3D` | `examples/convex_sections/convex_sections_demo.py` |
| One copied two-face teaching dihedral | `ExtractedDihedralScene3D` | `examples/derived_dihedral_extraction/` |

Render representative scenes:

```bash
cd "$TIKZ_NATIVE_REPO"
python -m manim --renderer cairo -ql \
  examples/polyhedron_visibility/cube_auto_occlusion.py \
  CubeAutoOcclusionDemo

python -m manim --renderer cairo -ql \
  examples/open_face_visibility/dihedral_auto_occlusion.py \
  DihedralAutoOcclusionDemo

python -m manim --renderer cairo -ql \
  examples/convex_sections/convex_sections_demo.py \
  CombinedSectionAndLineDemo
```

Automatic occlusion is topology-driven. It does not inspect an arbitrary
`VGroup` and guess which polygons are faces or which coincident points are the
same vertex. Register stable vertices, maximal convex faces, semantic strokes,
and incident-face relationships. The solver targets parallel projection and
fails closed on unsupported or ambiguous input.

`ConvexSectionScene3D` requires exactly one cutting plane before it can freeze
or bind; it is not a free-line-only facade. For direct low-level face shading
on an already frozen visibility model, see `DepthCuedAutoOcclusion3D` in the
[public API](public-api.md#automatic-face-depth-cues).

## Workflow 4: author quadrics and conic sections

For a fixed-topology plane action, start with `QuadricSectionRig`. It derives
the section, visible/hidden curve spans, cutting-plane depth regions, surface
order, fixed Manim slots, and rollback state.

```bash
cd "$TIKZ_NATIVE_REPO"
python -m manim --renderer cairo -r 480,270 --fps 15 \
  --media_dir media/quadric-quickstart \
  examples/quadrics/quadric_section_rig_quick_start.py \
  ConeSectionRigQuickStart
```

Use `render_profile="preview"` with 480×270 at 15 fps while composing. Use
`render_profile="final"` with 960×540 at 30 fps for the classroom master.
Resolution and frame rate remain Manim command options; a render profile does
not change a renderer that has already been constructed.

Use `QuadricSection3D` for lower-level callbacks, and the scheduled transition
path through `QuadricSectionTransition3D` for an ellipse → parabola →
hyperbola family change. Use `QuadricOcclusion3D` for certified separated
surfaces and curves when there is no cutting plane. `CompositeQuadricSection3D`
is the constrained two-nappe facade for `OPEN_DOUBLE`.

Phase-1 `QuadricSectionRig` freezes one static parallel projection. When plane
or section ink is enabled, it rejects an edge-on view before playback. A
supported AREA → LINE → AREA camera sequence instead uses the lower-level
`QuadricSection3D` with a live camera projection. For a precompiled sequence
that coordinates authored shots, the section timeline, topology banks, and
preflight, use the separate `compile_parallel_section_rig_from_shots()` path.

The supported analytic surfaces include finite spheres, closed or open
cylinders, closed single cones, open single cone shells, and finite open double
shells. Production bindings are Cairo and parallel-projection only. General
intersecting quadrics, arbitrary multiple visible cutting planes, perspective,
and OpenGL production parity are outside the v1 contract.

See [Finite-quadric authoring workflow](quadric-authoring-workflow.md) before
using the lower-level compositor APIs.

## Workflow 5: choose the right Dandelin path

The project has several Dandelin paths with different authority. They are not
interchangeable:

| Path | What is certified | Important limit |
| --- | --- | --- |
| `DandelinSection3D` classroom facade | Analytic construction and ordinary cone section | Spheres and helpers are a static top teaching overlay. |
| `DandelinOcclusion3D` | Live callable-camera visible/hidden curve fragments in fixed Cairo slots | Surface fills are teaching layers; the binding owns no cutting-plane fill. |
| Depth-aware spatial TikZ | Frozen-view visible/hidden boundary fragments | Motion and camera shots are rejected; fills are not physical hidden surfaces. |
| `depth_aware_teaching_transparent` | Static spatial single-nappe circle/ellipse/parabola curve fragments plus a certified classroom painter order | Contact circles must be shown; motion and other views are rejected, and teaching transparency is not optical transparency. |
| Nested-tangent `SceneOcclusionCoordinator` path | One cone/cylinder mother, one tangent plane, exactly two tangent spheres, and registered analytic boundaries | It is a narrow registered scene contract, not arbitrary-object occlusion. |

Render the static three-act classroom facade:

```bash
cd "$TIKZ_NATIVE_REPO"
python -m manim --renderer cairo --disable_caching -ql --fps 12 \
  --media_dir media/classroom-dandelin \
  examples/classroom_dandelin_spheres/classroom_dandelin_spheres.py \
  DandelinThreeConicsLesson
```

Render the complete cone-to-cylinder switching example:

```bash
python -m manim --renderer cairo --disable_caching -ql --fps 12 \
  --media_dir media/dandelin-cone-cylinder-switch \
  examples/dandelin_cone_cylinder_switch/dandelin_cone_cylinder_switch.py \
  DandelinConeCylinderSwitch
```

In the switching scene, plane edges, mother-surface boundaries, sphere
silhouettes, contact circles, the true section, and the teaching axis are all
registered as analytic boundary sources. Visible and hidden strokes are
recomputed per frame. The translucent fills still express a teaching-layer
order and do not claim pixel-accurate physical transparency. The final Manim
display assembly in this example is not a general-purpose
`DandelinConeCylinderSwitch3D` public facade.

The fixed-view TikZ three-view route is documented in
[TikZ-native Dandelin views](../examples/tikz_dandelin_views/README.md).

## Workflow 6: use semantic parallel-camera shots

`ParallelCameraState` represents an invertible parallel view together with its
world target, screen anchor, and zoom. `ParallelCameraShotSequence` adds names,
durations, holds, cues, and safe transitions. Those two types are immutable
authoring data. Actual Manim playback uses a `MultiProjectionCamera` on the
Scene plus `play_parallel_camera_shot()` or
`play_parallel_camera_shot_sequence()`.

Render the focused camera scenes:

```bash
cd "$TIKZ_NATIVE_REPO"
python -m manim --renderer cairo -ql \
  examples/parallel_camera_views/scene.py \
  TargetOrbitCameraDemo PlaneViewReductionDemo AnchorZoomCameraDemo

python -m manim --renderer cairo -ql -r 480,270 --fps 6 \
  examples/parallel_camera_shots/semantic_parallel_camera_demo.py \
  SingleConeSectionShotDemo
```

An exact view along a plane is a valid 3D camera state even though that plane
projects to a finite line. The coordinated section runtime explicitly handles
supported AREA → LINE → AREA transitions. Perspective states are not accepted
by the quadric/occlusion pipeline.

## Output and working-directory rules

- Relative source paths are resolved from the current working directory or the
  source-project manifest directory, depending on the API. Run example commands
  from the repository root unless the guide says otherwise.
- Put local Manim output under `media/` or outside the checkout. `media/` is
  ignored by Git; `artifacts/` is not a general scratch directory.
- A source-project build owns only its declared `derivedOutput`. Do not place
  authored files in that directory.
- Manim may create a `media/Tex` cache in the command working directory. That is
  normal generated state, not author data.

## Troubleshooting

### An import works at the repository root but fails in a scene

Confirm that the virtual environment is active, then run the outside-checkout
import shown in the installation section. On macOS, an editable `.pth` file can
inherit Finder's hidden flag and be skipped by Python 3.12. Repair only the
virtual environment and reinstall if necessary:

```bash
chflags -R nohidden "$TIKZ_NATIVE_REPO/.venv"
"$TIKZ_NATIVE_REPO/.venv/bin/python" -m pip install \
  --force-reinstall --no-deps -e "$TIKZ_NATIVE_REPO"
```

`PYTHONPATH=$PWD` is useful as a temporary diagnosis; it should not replace a
healthy isolated installation.

### TeX labels or rendering fail

Check `xelatex`, `dvisvgm`, FFmpeg, and the two fonts listed above. Use the
project-pinned Manim 0.20.1 environment. The newest Manim release is not
automatically compatible with this checkout.

### The compiler rejects valid general TikZ

This compiler accepts a documented semantic subset, not arbitrary TikZ. The
error is expected when a feature is outside that subset. Reduce the source to a
minimal fixture and consult [Supported TikZ subset](supported-tikz.md); there is
no SVG or bitmap fallback.

### `tikz-native-project status` returns 1

Read its JSON. Exit 1 means the derived output is missing or stale, not that the
command crashed. Run `build` after reviewing the reported actions. Exit 2 is an
invalid input, ownership, safety, or build failure.

### A quadric or Dandelin scene rejects OpenGL or perspective

Use the Cairo renderer and a supported parallel view. These constraints are
part of the production contract rather than optional visual preferences.

### A Bridge `run` request reports a hash or revision mismatch

Recompute the source hash and query the current `health` response. Bridge
requests are snapshot contracts; the Provider will not silently execute a
request against different source bytes or a different implementation revision.

## Next steps

- Browse the [documentation index](README.md).
- Use the [public API reference](public-api.md) for exact signatures.
- Read [Contributing](../CONTRIBUTING.md) and the
  [maintainer guide](maintainer-guide.md) before changing implementation or
  release evidence.
