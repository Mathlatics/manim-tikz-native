# Convex sections and free-line intersections

Render the ordinary-Manim demos with Manim Community 0.20.1:

```bash
manim -pql examples/convex_sections/convex_sections_demo.py LineThroughCubeDemo
manim -pql examples/convex_sections/convex_sections_demo.py MovingPlaneSectionDemo
manim -pql examples/convex_sections/convex_sections_demo.py CombinedSectionAndLineDemo
manim -pql examples/convex_sections/convex_sections_demo.py AccurateTransparentSectionDemo
```

The examples register one closed convex cube, its twelve semantic edges, one
optional free semantic line, and one moving infinite plane. Its visible finite
patch auto-fits the complete solid with margin, so the examples can provide
only tiny minimum half-extents. The module derives:

- the free line's entry/exit points and inside interval;
- the moving plane's empty/point/segment/polygon section;
- stable visible and hidden slots for the original lines and the derived
  section boundary;
- global line occlusion by all solid faces and the fitted cutting-plane patch.

`AccurateTransparentSectionDemo` also binds every source face as a native
fill-only `Polygon` and enables `accurate_transparency=True`. It splits the
full plane, highlighted section, and crossed solid faces into local triangles,
then recomputes their far-to-near order every frame without replacing Manim
objects.

No per-line/per-face occlusion relation is authored in the example.

Four additional closed solids exercise different face/edge topology and
different section polygons:

```bash
manim -pql examples/convex_sections/other_convex_solids_demo.py TetrahedronSectionDemo
manim -pql examples/convex_sections/other_convex_solids_demo.py TriangularPrismSectionDemo
manim -pql examples/convex_sections/other_convex_solids_demo.py SquarePyramidSectionDemo
manim -pql examples/convex_sections/other_convex_solids_demo.py OctahedronSectionDemo
```

Each scene registers the whole closed solid, all of its semantic edges, one
free semantic line, and one moving infinite plane. The same global solver then
derives all visible/hidden line spans and every section boundary; the demos do
not contain handwritten occlusion relations.
