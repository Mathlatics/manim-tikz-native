# Finite-quadric authoring workflow

This guide is the short product path for ordinary scene authors. Geometry,
visibility, painter ordering, fixed Manim slots, and rollback stay inside the
existing production controllers; the public facade only collects the inputs.

## Quick start

[`quadric_section_quick_start.py`](../examples/quadrics/quadric_section_quick_start.py)
contains one moving closed cone section. The scene declares one cone and one
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
