# TikZ-native Dandelin views

`tikz_dandelin_views.tex` declares one finite right cone, one explicit cutting
plane, and their certified Dandelin construction. The three pictures then show
the same source geometry as:

1. a fixed spatial teaching diagram;
2. the true cone-axis meridian, where sphere circles are genuine great-circle
   sections; and
3. the cutting-plane conic with foci and directrices, but no invented sphere
   circles.

Compile picture indices 1, 2, and 3 with the ordinary Provider fixed-view
path. From the repository root, this copy-ready command writes three PNG files
under the Git-ignored `media/` directory:

```bash
python - <<'PY'
from pathlib import Path

from tikz_native import compile_asset, render_static_png

source = Path("examples/tikz_dandelin_views/tikz_dandelin_views.tex")
output = Path("media/tikz-dandelin-views")
for picture_index, name in enumerate(("spatial", "meridian", "section"), 1):
    compiled = compile_asset(source, picture_index=picture_index)
    render_static_png(
        compiled,
        output / f"{picture_index}-{name}.png",
        pixel_width=960,
        pixel_height=540,
        media_dir=output / "manim-cache",
    )
PY
```

This is a static fixed-view Provider path. It is not a Source v3 request and
does not produce a camera-motion Scene.

The source pins a painter mode on every picture instead of relying on a
default. The spatial picture opts into
`depth_aware_teaching_transparent`. Existing quadric ray tests split cone
boundaries, sphere silhouettes, contact circles, the conic, the cutting-plane
outline, and directrices into solid visible fragments and dashed hidden
fragments for the authored parallel view, then order them with the shared
fragment painter graph. The same fixed camera also drives a certified teaching painter
order: each cone component is split into a far and near sheet, each Dandelin
sphere is inserted between the sheets of its authenticated nappe, and the
cutting-plane patch is split into behind/outside/between/front regions. The
`behind` and `between` plane regions, which have a nearer cone sheet in front
of them, use a muted grey-blue fill and dashed finite-outline fragments;
`outside` and `front` retain the normal teal fill and solid outline. The
contact-circle stroke owns the equal-depth seam. This is a classroom
transparency model, not a claim about optical materials; physical surface
visibility remains explicitly non-authoritative. Consequently this mode keeps
`show-contact-circles=true`; requesting `false` fails before registration.
Motion, `cameraShots`, and
source-v3 generation are deliberately unsupported.

The meridian and cutting-plane pictures explicitly use `mode=diagrammatic`.
They are true planar reductions rather than projected 3D surface composites:
the meridian keeps the sphere circles above the cutting line, while the
cutting-plane view contains only its conic, foci, and directrices.

The semantic source commands are:

```tex
\DeclareSpaceRightCone{cone}{A/Z/R}{30}{0/9}{open_single};
\DeclareSpacePlane{cut}{O/U/V};
\DeclareDandelinConstruction{dan}{cone}{cut};
\DrawDandelinDiagram[
  view=spatial,
  mode=depth_aware_teaching_transparent,
  preset=classroom
]{dan};
```

`A` is the apex, `Z-A` fixes the positive cone axis, and the component of
`R-A` perpendicular to that axis fixes the radial phase. The half angle is in
degrees; the next argument is the finite axial range.
