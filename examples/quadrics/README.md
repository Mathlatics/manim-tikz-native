# Quadric occlusion demos

These scenes exercise the public quadratic-surface API with the Cairo
renderer.  They are intentionally ordinary Manim source files: no private
test hook or sampled mesh is required.

The quadric Manim controllers use a true orthographic isometric projection by
default.  The cone demos intentionally rely on that default: their world-z
axes remain vertical on screen and the view contains no affine shear.  The
global-overlap demo passes a custom general parallel view because its entities
are positioned specifically for that overlap test.

```bash
manim -pql examples/quadrics/quadric_occlusion_demo.py \
  MovingSphereSectionDemo ObliqueCylinderSectionDemo \
  ConeSectionFamiliesDemo ConeSectionTopologyTransitionDemo \
  GlobalQuadricOcclusionDemo
```

- `MovingSphereSectionDemo` recomputes a moving circular section and its
  front/hidden spans on every frame.
- `ObliqueCylinderSectionDemo` rotates an infinite mathematical plane while
  keeping a stable finite ellipse branch.
- `ConeSectionFamiliesDemo` shows finite ellipse, parabola, and hyperbola
  sections side by side.
- `ConeSectionTopologyTransitionDemo` rotates one cutting plane continuously
  through ellipse, the exact parabolic critical position, and hyperbola.  The
  controller performs the topology handoff automatically with two
  preallocated banks.  The plane is split against the cone into rear,
  between-sheet, front, and outside regions, then those regions, both smooth
  cone sheets, and both curve banks share one painter graph.  Adjacent display
  cells are merged before Cairo draws them, so the plane has no triangle seams.
- `GlobalQuadricOcclusionDemo` lets the Manim controller certify two disjoint
  convex solids automatically, derive their far-to-near relation, and put both
  surfaces and crossing semantic strokes in one painter graph.

`QuadricOcclusion3D` remains the fixed-topology controller for ordinary scenes.
For a topology-changing rotating-plane section, first build the analytic
schedule with `track_scheduled_plane_section()`, then pass that schedule and a
normalized `ValueTracker` to `QuadricSectionTransition3D`.  It reserves both
render banks before attachment, displays the exact critical conic, and performs
the handoff without silently reusing one Manim object as a different
mathematical branch.
