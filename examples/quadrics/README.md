# Quadric occlusion demos

These scenes exercise the public quadratic-surface API with the Cairo
renderer.  They are intentionally ordinary Manim source files: no private
test hook or sampled mesh is required.

```bash
manim -pql examples/quadrics/quadric_occlusion_demo.py \
  MovingSphereSectionDemo ObliqueCylinderSectionDemo \
  ConeSectionFamiliesDemo GlobalQuadricOcclusionDemo
```

- `MovingSphereSectionDemo` recomputes a moving circular section and its
  front/hidden spans on every frame.
- `ObliqueCylinderSectionDemo` rotates an infinite mathematical plane while
  keeping a stable finite ellipse branch.
- `ConeSectionFamiliesDemo` shows finite ellipse, parabola, and hyperbola
  sections side by side.
- `GlobalQuadricOcclusionDemo` lets the Manim controller certify two disjoint
  convex solids automatically, derive their far-to-near relation, and put both
  surfaces and crossing semantic strokes in one painter graph.

The runtime is fixed-topology while attached.  A topology-changing transition
such as one cone section changing from ellipse to two hyperbola branches is
first authored with `compute_plane_motion_schedule()` and
`track_scheduled_plane_section()`.  At the exact event, the scene must perform
an explicit branch handoff/cross-fade; it must not silently reuse one Manim
object for a different mathematical branch.
