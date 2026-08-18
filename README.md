# manim-tikz-native

[![CI](https://github.com/Mathlatics/manim-tikz-native/actions/workflows/ci.yml/badge.svg)](https://github.com/Mathlatics/manim-tikz-native/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Manim 0.20.1](https://img.shields.io/badge/Manim-0.20.1-6c55a3.svg)](https://www.manim.community/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Compile a documented, restricted TikZ subset into semantic native Manim
objects. Keep named geometry editable, drive it with `ValueTracker`, and add
projection-aware hidden-line removal for closed convex polyhedra or articulated
open faces such as a dihedral angle.

This project does **not** convert arbitrary TikZ and does not silently fall back
to SVG. Unsupported syntax is reported explicitly.

[中文说明](README.zh-CN.md) · [Public API](docs/public-api.md) ·
[Automatic occlusion](docs/automatic-occlusion.md) ·
[Supported TikZ subset](docs/supported-tikz.md)

## What it provides

- restricted TikZ → native `Line`, `Polygon`, `Circle`, `Ellipse`, `Dot`,
  `Tex`, `MathTex`, arrow and angle-marker objects;
- stable semantic object IDs and named geometric relationships;
- 2D and 3D geometry rigs driven by ordinary Manim trackers;
- fixed-view and local-camera 3D projection;
- automatic solid/hidden line splitting for closed convex polyhedra;
- automatic line occlusion and face ordering for finite open convex faces and
  articulated hinges;
- readable native Manim source generation and versioned JSON bridges;
- strict, component-level compatibility identities for cached integrations.

## Requirements

- Python 3.11 or newer;
- Manim Community 0.20.1;
- XeLaTeX with `standalone`, `fontspec`, `xeCJK`, `unicode-math`, and TikZ;
- the TeX Live Fandol and Latin Modern fonts for the default template;
- FFmpeg for video rendering.

On macOS, a full TeX Live installation already includes the default fonts. On
Debian or Ubuntu, the CI setup installs `texlive-xetex`,
`texlive-latex-extra`, `texlive-lang-chinese`, and
`texlive-fonts-recommended`.

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

For ordinary Manim scenes, register stable vertices, maximal convex faces, and
semantic `Line` objects through `OcclusionScene3D` or `OpenFaceScene3D`. The
module updates preallocated visible and dashed slots in place during
`scene.play()` and restores the original source objects when the session ends.

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
cache. Those are application concerns and can consume this package through its
Python API or JSON bridges.

## Status

Version `0.1.0` is an alpha release. The public contracts are versioned and the
compiler fails closed, but the accepted TikZ language is intentionally smaller
than TikZ itself. Please report a minimal `.tex` example when requesting new
syntax.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
