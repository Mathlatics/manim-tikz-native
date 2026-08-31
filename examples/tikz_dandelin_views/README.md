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
path. The spatial picture opts into
`depth_aware_teaching_transparent`. Existing quadric ray tests split cone
boundaries, sphere silhouettes, contact circles, the conic, the cutting-plane
outline, and directrices into solid visible fragments and dashed hidden
fragments for the authored parallel view, then order them with the shared
fragment painter graph. The same fixed camera also drives a certified teaching painter
order: each cone component is split into a far and near sheet, each Dandelin
sphere is inserted between the sheets of its authenticated nappe, and the
cutting-plane patch is split into behind/outside/between/front regions. The
contact-circle stroke owns the equal-depth seam. This is a classroom
transparency model, not a claim about optical materials; physical surface
visibility remains explicitly non-authoritative. Consequently this mode keeps
`show-contact-circles=true`; requesting `false` fails before registration.
Motion, `cameraShots`, and
source-v3 generation are deliberately unsupported.

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
