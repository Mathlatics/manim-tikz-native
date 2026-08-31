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
path. These v1 assets are static and `diagrammatic`; their sphere/cone painter
order is not a claim of physical multi-surface occlusion. Motion,
`cameraShots`, and source-v3 generation are deliberately unsupported.

The semantic source commands are:

```tex
\DeclareSpaceRightCone{cone}{A/Z/R}{30}{0/9}{open_single};
\DeclareSpacePlane{cut}{O/U/V};
\DeclareDandelinConstruction{dan}{cone}{cut};
\DrawDandelinDiagram[view=spatial,preset=classroom]{dan};
```

`A` is the apex, `Z-A` fixes the positive cone axis, and the component of
`R-A` perpendicular to that axis fixes the radial phase. The half angle is in
degrees; the next argument is the finite axial range.
