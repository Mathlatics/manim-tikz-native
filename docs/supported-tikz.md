# Supported TikZ subset

The authoritative machine-readable registry is
[`tikz_native/subset_v0_1.json`](../tikz_native/subset_v0_1.json). Every feature
is classified as dynamic-safe (A), static-safe (B), or unsupported (C).

## Dynamic-safe highlights

- named 2D and 3D coordinates;
- numeric, side-effect-free macros in the controlled frontend;
- interpolation, translation, projection, and explicit 3D hinge relations;
- named paths for lines, circles, and ellipses;
- line-line and line-ellipse intersections with explicit path ordering;
- lines, Stealth arrows, polygons, circles, ellipses, dots, labels, angle arcs,
  and right-angle markers;
- explicit line widths, dash patterns, xcolor mixes, opacity, and controlled
  font-size commands;
- `3d view` and explicit TikZ x/y/z parallel-projection bases;
- billboard labels and path labels with controlled anchors.

## Static-safe only

- TikZ baseline and external trim metadata;
- `dashed` and `densely dashed` keywords, represented with fixed approximate
  pitch instead of pixel-exact TikZ rhythm;
- one proven redundant `scope` case with `draw=none`.

## Unsupported examples

- arbitrary Bézier paths, general arcs, plots, or smooth sampling;
- clipping paths;
- decorations, patterns, shading, or gradients;
- complex node shapes, automatic text wrapping, or TikZ matrices;
- nested transformed scopes with inherited geometry and style;
- arbitrary macros, conditionals, file access, or complex `pgfkeys`;
- topology-changing intersections such as tangency merge/disappearance.

Unsupported syntax is reported with its source statement. Strict conversion
stops; it never silently substitutes SVG, raster output, or an uneditable
generic path.

## Authoring guidance

- Name every point that will be animated or referenced later.
- Separate polygon fill from edge drawing when the edge needs its own semantic
  identity.
- Prefer explicit dash patterns when exact rhythm matters.
- Declare 3D projection and hinge relationships instead of relying on visual
  coincidence.
- Keep macros numeric and side-effect-free.
