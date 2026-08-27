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

manim -pql examples/quadrics/unified_boundary_visibility_demo.py \
  UnifiedBoundaryVisibilityComparison

manim -pql examples/quadrics/section_plane_cone_boundary_demo.py \
  SectionPlaneConeBoundaryDemo

manim -pql examples/quadrics/cone_model_comparison_demo.py \
  ConeModelComparisonDemo ConeModelPlaneComparisonDemo

manim -pql examples/quadrics/extended_acceptance_demo.py \
  ClosedOpenComparisonAcceptance SectionTopologyAcceptance \
  CurvePolicyComparisonAcceptance SideViewTrimRimAcceptance \
  CapChordActivationAcceptance
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
- `UnifiedBoundaryVisibilityComparison` runs the complete
  ellipse-to-parabola-to-hyperbola transition side by side. The unified half
  includes cap rims, true silhouettes, the section outline, and a red teaching
  generator whose `style_id` resolves through the fixed boundary-style
  registry.
- `SectionPlaneConeBoundaryDemo` compares top-overlay and depth-aware hidden
  ink for the yellow true section together with the cyan cone silhouette and
  finite base rim. It keeps the tessellated surface stroke transparent while
  assigning explicit visible and hidden styles to the semantic boundaries.
- `ConeModelComparisonDemo` compares a closed single cone, an open single cone
  shell, and a finite open double shell while their common axis tilts. The
  closed base, open trim rims, and two stable double-shell components all use
  the public contract and fixed component paint slots.
- `ConeModelPlaneComparisonDemo` moves the same translucent plane through a
  closed single cone and an open single shell. The yellow lateral conic is
  explicit in both models; after the plane reaches the terminal disk, only the
  closed solid activates a stable yellow cap chord. The open shell retains one
  open lateral arc. Cyan plane-hidden generator/trim-rim dashes remain beneath
  the plane in the depth-aware painter order.
- The five `*Acceptance` scenes are the video sources for the nightly/release
  evidence bundle. Their shared builders also produce the 960x540 semantic
  keyframes, so the videos and JSON painter evidence consume the same public
  controller path.

`QuadricOcclusion3D` remains the fixed-topology controller for ordinary scenes.
For a topology-changing rotating-plane section, first build the analytic
schedule with `track_scheduled_plane_section()`, then pass that schedule and a
normalized `ValueTracker` to `QuadricSectionTransition3D`.  It reserves both
render banks before attachment, displays the exact critical conic, and performs
the handoff without silently reusing one Manim object as a different
mathematical branch.
