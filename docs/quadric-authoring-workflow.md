# Finite-quadric authoring workflow

This guide is the short product path for ordinary scene authors. Geometry,
visibility, painter ordering, fixed Manim slots, and rollback stay inside the
existing production controllers; the public facade only collects the inputs.

## Natural fixed-topology actions

[`quadric_section_rig_quick_start.py`](../examples/quadrics/quadric_section_rig_quick_start.py)
is the recommended entry point when the storyboard says “move the plane” or
“rotate the plane”. `QuadricSectionRig` owns the immutable current plane,
reserves a non-overlapping painter band for the Scene, and returns ordinary
Manim animations:

```python
from math import pi
from manim import Scene
from polyhedron_visibility.quadrics import (
    ConeSpec, QuadricSectionRig, SectionPlane,
)

class ConeSectionLesson(Scene):
    def construct(self):
        cone = ConeSpec("cone", (0, 0, -1.5), (0, 0, 1), pi / 6, (0, 4))
        plane = SectionPlane(
            "cut", (0, 0, -0.4), (0.45, 0, 1), u_axis=(0, 1, 0),
        )
        with QuadricSectionRig(
            self, surface=cone, section_id="section", plane=plane,
            paint_policy="depth_aware_diagrammatic",
        ).session() as section:
            self.play(section.animate_plane_shift(0.6), run_time=2)
            self.play(
                section.animate_plane_rotation(
                    axis=(0, 0, 1), angle=pi / 3, pivot=cone.apex,
                ),
                run_time=2,
            )
```

The action is compiled before it is returned. Axis-angle rotation uses the
analytic critical schedule; parallel translation uses the exact critical
heights of the finite sphere, cylinder, or cone. Stable tracked curve slots
separate renderer identity from incidental `circle`/`ellipse`, hyperbola-label,
and periodic-seam IDs. No Manim object or curve slot is created by an updater.

Phase 1 also resolves and freezes one complete non-callable parallel
`projection` when the rig is constructed. A semantic camera state's target,
screen anchor, and zoom are retained; a callable projection is rejected instead
of being sampled once.
Whenever either `show_plane` or `draw_section_boundary` is enabled, the initial
plane and the complete axis-angle rotation path are analytically certified as
AREA projections; an interior edge-on or numerically rank-deficient view is
rejected before playback with its analytic candidate progress. The lower-level
facade and semantic-camera APIs can render certified edge-on LINE frames, but
this narrower Phase 1 Rig does not animate through that rank handoff. Set both
options to `False` only when no rank-sensitive plane or section ink should be
displayed.

This first layer deliberately fails before `Scene.play` when a path enters an
empty or degenerate section, changes conic family, branch count, or component
count. `animate_plane_to()` currently accepts only a target with the same
`plane_id` and exactly the same normalized normal and `u_axis`; a normal-changing
point-lerp path needs the topology-aware timeline compiler. Cap-chord
activation/deactivation also remains in that compiler even though the facade
has already reserved its display slots.

Pass `painter_z_band=(low, high)` only for an advanced exact override. The
default reserves the first available Scene band at or above `(20, 30)` and
releases it on `restore()` or session exit.

## Manual callback quick start

[`quadric_section_quick_start.py`](../examples/quadrics/quadric_section_quick_start.py)
contains the equivalent lower-level moving closed cone section. The scene declares one cone and one
live plane; `QuadricSection3D` derives the complete finite section, cap-chord
capacity, solid/hidden spans, and unified painter order automatically.

```python
from math import pi
from manim import Scene, ValueTracker, linear
from polyhedron_visibility.quadrics import (
    ConeSpec, QuadricManimStyle, QuadricSection3D, SectionPlane,
)

class ConeSectionQuickStart(Scene):
    def construct(self):
        progress = ValueTracker(0)
        cone = ConeSpec("cone", (0, 0, -1.5), (0, 0, 1), pi / 6, (0, 4))
        def plane():
            return SectionPlane(
                "cut", (0, 0, -1 + 2.7 * progress.get_value()),
                (0.65, 0, 1), u_axis=(0, 1, 0),
            )
        QuadricSection3D(
            self, surface=cone, section_id="cone-section", plane=plane,
            paint_policy="depth_aware_diagrammatic", render_profile="preview",
            style=QuadricManimStyle(surface_fill_opacity=.62),
        ).attach()
        self.play(progress.animate.set_value(1), run_time=4, rate_func=linear)
```

Render the checked-in scene with the matching output settings:

```bash
# Fast composition work
manim -r 480,270 --fps 15 \
  examples/quadrics/quadric_section_quick_start.py ConeSectionQuickStart

# After changing RENDER_PROFILE to "final"
manim -r 960,540 --fps 30 \
  examples/quadrics/quadric_section_quick_start.py ConeSectionQuickStart
```

`render_profile` selects controller approximation, fixed-capacity, semantic
boundary, and optional component-shading defaults. Resolution and frame rate
remain explicit command/config values because Manim creates the renderer before
the scene controller. Supplying `limits`, `max_chord_error`,
`section_max_screen_error`, or `include_surface_boundaries` explicitly overrides
that profile default. Advanced IDs, painter bands, and compositor limits are
not required for this path.

## Three workflow tiers

| tier | use it for | controller/output | acceptance meaning |
| --- | --- | --- | --- |
| Preview | camera, plane position, text, colour, and timing | `render_profile="preview"`; 480x270 at 15 fps | fast bounded display approximation; true sections, visibility, and painter graph remain enabled |
| Final | classroom video | `render_profile="final"`; 960x540 at 30 fps | full component shading and release-quality approximation |
| Release / Evidence | project release or geometry/runtime change | Final profile plus pinned environment and Extended Quadric Acceptance | keyframes, RGB probes, painter traces, full-motion scans, CSV, video decode, and reproducible packages |

Release / Evidence is deliberately not a third `render_profile`. It is a
repeatable acceptance process around the Final profile. This prevents an
ordinary title or colour edit from accidentally paying for, or claiming, a
full release certification.

The extended workflow and evidence bundle are documented in
[`extended-quadric-ci.md`](extended-quadric-ci.md).

## Capacity planning in one call

For a custom motion, write a factory that accepts one normalized
`ValueTracker` and returns an unattached high-level controller. The planner
calls the factory once and drives the same fixed Manim slots over every listed
frame:

```python
from manim import Scene
from polyhedron_visibility.quadrics import QuadricCapacityPlanner, QuadricSection3D

def scene_factory(progress):
    return QuadricSection3D(
        Scene(), surface=cone, section_id="lesson-section",
        plane=lambda: plane_at(progress.get_value()),
        paint_policy="depth_aware_diagrammatic", render_profile="final",
    )

plan = QuadricCapacityPlanner.scan(scene_factory, frames=range(0, 121))
print(plan.summary())
print(plan.summary(locale="zh-CN"))
limits = plan.recommended_limits
```

Example Chinese output:

```text
扫描样本：121 帧（只认证这些帧）
边界源峰值：8
每源 fragment 峰值：11
每 fragment dash 峰值：27
平面 fragment 峰值：4280
射线分类峰值：21734
预计固定 Mobject：392
建议 profile：final
```

`frames` are mapped linearly from the first index to progress `0` and the last
index to progress `1`. The recommendation certifies only those listed frames.
For an analytic topology schedule, keep using `scan_schedule()` so every
tangency and topology knot is added even when it lies between rendered frames.
For explicitly chosen progress values, the existing
`QuadricCapacityPlanner(controller, progress=tracker).scan(progresses)` form
remains supported.

The recommended limits stay an immutable value (`plan.recommended_limits`),
not a method, so existing scenes and saved planning code remain compatible.
Any overflow, uncertified geometry, changed `scene.mobjects`, replaced slot, or
failed rollback aborts the scan instead of returning a guessed plan.

## Advanced controls

Use the detailed [quadric occlusion guide](quadric-occlusion.md) only when a
scene genuinely needs custom boundary styles, generator boundaries, shared
geometry prototypes, explicit capacity headroom, or compositor limits. The
finite-cone [v1 support contract](quadric-section-v1-contract.md) remains the
authority for supported and explicit-failure geometry.
