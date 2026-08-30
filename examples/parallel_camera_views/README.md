# Semantic parallel-camera test animations

These examples are independent of the quadric and conic-section packages. The
file contains three focused acceptance animations plus one compact combined
sequence:

1. `TargetOrbitCameraDemo` moves between authored world targets while the view
   direction also rotates;
2. `PlaneViewReductionDemo` moves a finite plane through normal, oblique,
   exact edge-on, and opposite-normal views;
3. `AnchorZoomCameraDemo` isolates world target, viewport anchor, state zoom,
   and inherited Manim zoom while `frame_center` is non-zero;
4. `FrameCenterCompatibilityDemo` checks semantic-to-legacy orbit and snapshot
   restore from a scaled semantic state;
5. `ParallelCameraViewsDemo` is a compact sequence covering all plane-view
   constructors.

The combined sequence moves through:

1. the existing cabinet-oblique view;
2. a view normal to the authored plane;
3. a plane-relative oblique view;
4. an exact view along the plane, where the finite patch projects to a line;
5. a second normal view whose target is placed at a non-central screen anchor.

Render the four focused low-quality Cairo acceptance videos with:

```bash
python -m manim --renderer=cairo -ql \
  examples/parallel_camera_views/scene.py \
  TargetOrbitCameraDemo PlaneViewReductionDemo AnchorZoomCameraDemo \
  FrameCenterCompatibilityDemo
```

The exact edge-on frame demonstrates camera validity and ordinary Manim
projection only. It does not claim that the separate `QuadricSection3D`
cutting-plane compositor supports rank-one plane occlusion.
