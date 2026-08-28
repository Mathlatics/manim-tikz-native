"""Extended Cairo acceptance scenes for the frozen finite-cone section contract.

The scene builders in this module are also consumed by the evidence generator.
That keeps the rendered videos, 960x540 keyframes, and renderer-neutral painter
traces on one production call path instead of maintaining a second test-only
renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import cos, pi, sin
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, Sequence

import numpy as np
from manim import DOWN, Scene, Text, UP, ValueTracker, linear, smooth

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.authoring import QuadricSection3D
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    DEFAULT_QUADRIC_VIEW,
    QuadricBoundaryStyle,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.performance import (
    QUADRIC_CAIRO_FRAME_TRACE_ENV,
    QUADRIC_CAIRO_FRAME_TRACE_SCHEMA,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
    section_cap_chord_curve_ids,
)


VIEW = DEFAULT_QUADRIC_VIEW
BACKGROUND_COLOR = "#101820"


def _scaled_view(view: ParallelView, factor: float) -> ParallelView:
    matrix = view.matrix
    matrix[:2] *= factor
    return ParallelView.from_matrix(matrix)


TOPOLOGY_VIEW = _scaled_view(VIEW, 0.85)


STYLE = QuadricManimStyle(
    surface_fill_color="#315A8A",
    surface_fill_opacity=0.76,
    surface_stroke_opacity=0.0,
    cone_lateral_fill_colors=("#173753", "#4F84B3", "#1D4368"),
    cone_cap_fill_colors=("#557A99", "#294B6B"),
    cone_lateral_sheen_direction=(1.0, 0.0, 0.0),
    cone_cap_sheen_direction=(-1.0, 1.0, 0.0),
    visible_curve_color="#FFD166",
    visible_curve_width=4.0,
    hidden_curve_color="#F59E0B",
    hidden_curve_width=3.0,
    hidden_curve_opacity=0.66,
    section_plane_fill_color="#43D9C0",
    section_plane_fill_opacity=0.34,
    section_plane_stroke_color="#B39DDB",
    section_plane_stroke_width=1.8,
    section_plane_stroke_opacity=0.9,
    dash_length=0.12,
    dash_gap=0.09,
)


BOUNDARY_STYLE = QuadricBoundaryStyle(
    visible_color="#5CE1E6",
    visible_width=4.4,
    visible_opacity=1.0,
    hidden_color="#5CE1E6",
    hidden_width=3.0,
    hidden_opacity=0.24,
    dash_length=0.12,
    dash_gap=0.09,
)


BOUNDARY_STYLES = {
    "style:surface-silhouette": BOUNDARY_STYLE,
    "style:surface-boundary": BOUNDARY_STYLE,
}


def acceptance_limits(*, transition: bool = False) -> QuadricManimLimits:
    """Return a reviewed fixed capacity for the extended acceptance scenes."""

    return QuadricManimLimits(
        max_surfaces=2,
        max_curves=32 if transition else 4,
        max_fragments_per_curve=32 if transition else 18,
        max_segments_per_fragment=384,
        max_surface_segments=768,
        max_dashes_per_fragment=100 if transition else 72,
        max_projected_length=18.0,
        max_total_mobjects=100000 if transition else 24000,
        max_boundary_sources=64 if transition else 28,
        max_boundary_styles=12,
    )


def near_side_view(angle: float) -> ParallelView:
    """Parallel view whose circular rim becomes rank one at ``angle == 0``."""

    return ParallelView.from_matrix(
        (
            (1.0, 0.0, 0.0),
            (0.0, -sin(angle), cos(angle)),
            (0.0, -cos(angle), -sin(angle)),
        )
    )


@dataclass(slots=True)
class AcceptanceState:
    """One reusable authoring state shared by videos and keyframe capture."""

    scenario_id: str
    tracker: ValueTracker
    start_value: float
    end_value: float
    authorings: tuple[object, ...]
    controllers: tuple[tuple[str, QuadricOcclusion3D], ...]
    projections: tuple[Callable[[], ParallelView], ...]

    def set_progress(self, progress: float) -> None:
        value = float(progress)
        if not 0.0 <= value <= 1.0:
            raise ValueError("acceptance progress must lie in [0, 1]")
        current = self.start_value + value * (self.end_value - self.start_value)
        self.tracker.set_value(current)
        for authoring in self.authorings:
            authoring.update(0.0)

    def restore(self) -> None:
        for authoring in reversed(self.authorings):
            authoring.restore()


def _title(scene: Scene, heading: str, note: str) -> tuple[Text, Text]:
    title = Text(heading, font_size=27, color="#F4F7FB").to_edge(UP, buff=0.24)
    subtitle = Text(note, font_size=15, color="#B8C5D6").next_to(
        title, DOWN, buff=0.10
    )
    title.set_z_index(100)
    subtitle.set_z_index(100)
    scene.add(title, subtitle)
    return title, subtitle


def _caption(scene: Scene, label: str, x: float) -> Text:
    value = Text(label, font_size=17, color="#DCE6F2").move_to(
        (x, -3.25, 0.0)
    )
    value.set_z_index(100)
    scene.add(value)
    return value


def _plane_interaction_controller(
    scene: Scene,
    *,
    prefix: str,
    model: ConeModel,
    horizontal: float,
    tracker: ValueTracker,
    policy: QuadricPaintPolicy,
    painter_band: tuple[float, float],
) -> tuple[QuadricOcclusion3D, Callable[[], ParallelView]]:
    normal = np.asarray((0.82, 0.0, 1.0), dtype=float)
    normal /= np.linalg.norm(normal)
    vertical_shift = -0.55 * np.asarray(VIEW.matrix[1], dtype=float)
    shift = horizontal * np.asarray(VIEW.matrix[0], dtype=float) + vertical_shift
    cone = ConeSpec(
        f"{prefix}:cone",
        tuple(shift + np.asarray((0.0, 0.0, -2.4))),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )

    def current_plane() -> SectionPlane:
        point = (
            shift
            + np.asarray((0.0, 0.0, -0.35))
            + tracker.get_value() * normal
        )
        return SectionPlane(
            f"{prefix}:plane",
            tuple(float(item) for item in point),
            (0.82, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )

    section_id = f"{prefix}:section"

    def current_curves():
        return compute_quadric_section_boundary_curves(
            section_id,
            cone,
            current_plane(),
        )

    allocated_curve_ids = tuple(
        sorted(
            {
                *(item.curve_id for item in current_curves()),
                *section_cap_chord_curve_ids(section_id, cone),
            }
        )
    )
    controller = QuadricOcclusion3D(
        scene,
        surfaces=(cone,),
        curves=current_curves,
        projection=TOPOLOGY_VIEW,
        paint_policy=policy,
        style=STYLE,
        boundary_styles=BOUNDARY_STYLES,
        limits=acceptance_limits(),
        max_chord_error=0.008,
        section_plane=current_plane,
        boundary_visibility_mode="unified",
        include_surface_boundaries=True,
        allocated_curve_ids=allocated_curve_ids,
        painter_z_band=painter_band,
    ).attach()
    return controller, lambda: VIEW


def _build_closed_open(
    scene: Scene, progress: float, with_labels: bool
) -> AcceptanceState:
    tracker = ValueTracker(-0.48 + 0.96 * float(progress))
    controllers = []
    projections = []
    rows = (
        ("closed", "CLOSED SOLID", ConeModel.CLOSED_SINGLE, -3.35),
        ("open", "OPEN SHELL", ConeModel.OPEN_SINGLE, 3.35),
    )
    labels: list[Text] = []
    for index, (name, label, model, horizontal) in enumerate(rows):
        controller, projection = _plane_interaction_controller(
            scene,
            prefix=f"acceptance:closed-open:{name}",
            model=model,
            horizontal=horizontal,
            tracker=tracker,
            policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            painter_band=(20.0 + 20.0 * index, 30.0 + 20.0 * index),
        )
        controllers.append((name, controller))
        projections.append(projection)
        if with_labels:
            labels.append(_caption(scene, label, horizontal))
    if with_labels:
        foreground = _title(
            scene,
            "Plane interaction: closed cone vs open shell",
            "yellow = true section  •  cyan = silhouette / rim",
        )
        scene.add_foreground_mobjects(*foreground, *labels)
    return AcceptanceState(
        "closed_open_comparison",
        tracker,
        -0.48,
        0.48,
        tuple(controller for _name, controller in controllers),
        tuple(controllers),
        tuple(projections),
    )


def _build_policies(
    scene: Scene, progress: float, with_labels: bool
) -> AcceptanceState:
    tracker = ValueTracker(-0.48 + 0.96 * float(progress))
    controllers = []
    projections = []
    labels: list[Text] = []
    rows = (
        ("physical", "PHYSICAL", QuadricPaintPolicy.PHYSICAL, -4.25),
        ("diagrammatic", "TOP DASH", QuadricPaintPolicy.DIAGRAMMATIC, 0.0),
        (
            "depth-aware",
            "DEPTH-AWARE",
            QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            4.25,
        ),
    )
    for index, (name, label, policy, horizontal) in enumerate(rows):
        controller, projection = _plane_interaction_controller(
            scene,
            prefix=f"acceptance:policies:{name}",
            model=ConeModel.CLOSED_SINGLE,
            horizontal=horizontal,
            tracker=tracker,
            policy=policy,
            painter_band=(20.0 + 18.0 * index, 30.0 + 18.0 * index),
        )
        controllers.append((name, controller))
        projections.append(projection)
        if with_labels:
            labels.append(_caption(scene, label, horizontal))
    if with_labels:
        foreground = _title(
            scene,
            "Hidden-curve paint policies",
            "physical omission  •  top teaching dash  •  depth-aware dash",
        )
        scene.add_foreground_mobjects(*foreground, *labels)
    return AcceptanceState(
        "hidden_curve_policies",
        tracker,
        -0.48,
        0.48,
        tuple(controller for _name, controller in controllers),
        tuple(controllers),
        tuple(projections),
    )


def _build_side_view(
    scene: Scene, progress: float, with_labels: bool
) -> AcceptanceState:
    tracker = ValueTracker(0.015 - 0.03 * float(progress))
    cone = ConeSpec(
        "acceptance:side-view:cone",
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (0.0, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.OPEN_SINGLE,
    )
    plane = SectionPlane(
        "acceptance:side-view:plane",
        (0.0, 0.5, -1.0),
        (0.0, 1.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    patch = PlaneDisplayPatchSpec(
        "acceptance:side-view:patch",
        plane.plane_id,
        3.0,
        2.6,
        center_coordinates=(0.0, -0.1),
    )
    curves = compute_quadric_section_boundary_curves(
        "acceptance:side-view:section", cone, plane
    )

    def projection() -> ParallelView:
        return near_side_view(tracker.get_value())

    controller = QuadricOcclusion3D(
        scene,
        surfaces=(cone,),
        curves=curves,
        projection=lambda _scene: projection(),
        paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        style=STYLE,
        boundary_styles=BOUNDARY_STYLES,
        limits=acceptance_limits(),
        max_chord_error=0.008,
        section_plane=plane,
        section_patch=patch,
        section_max_screen_error=0.08,
        boundary_visibility_mode="unified",
        include_surface_boundaries=True,
    ).attach()
    if with_labels:
        foreground = _title(
            scene,
            "Open cone side view: ellipse → segment → ellipse",
            "the trim rim remains finite through the exact rank-one frame",
        )
        scene.add_foreground_mobjects(*foreground)
    return AcceptanceState(
        "side_view_trim_rim",
        tracker,
        0.015,
        -0.015,
        (controller,),
        (("open-shell", controller),),
        (projection,),
    )


def _build_cap_chord(
    scene: Scene, progress: float, with_labels: bool
) -> AcceptanceState:
    tracker = ValueTracker(-0.48 + 0.96 * float(progress))
    controller, projection = _plane_interaction_controller(
        scene,
        prefix="acceptance:cap-chord",
        model=ConeModel.CLOSED_SINGLE,
        horizontal=0.0,
        tracker=tracker,
        policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        painter_band=(20.0, 30.0),
    )
    if with_labels:
        foreground = _title(
            scene,
            "Finite closed cone: cap-chord activation",
            "the yellow chord appears only while the plane crosses the real base disk",
        )
        scene.add_foreground_mobjects(*foreground)
    return AcceptanceState(
        "cap_chord_activation",
        tracker,
        -0.48,
        0.48,
        (controller,),
        (("closed", controller),),
        (projection,),
    )


def _build_topology(
    scene: Scene, progress: float, with_labels: bool
) -> AcceptanceState:
    # The transition owns two preallocated banks whose initial attachment is
    # certified at progress zero.  Static evidence then advances through the
    # same public update path used by the rendered animation.
    tracker = ValueTracker(0.0)
    cone = ConeSpec(
        "acceptance:topology:cone",
        (0.0, 0.0, -1.5),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    motion = AxisAnglePlaneMotion(
        "acceptance:topology:motion",
        SectionPlane(
            "acceptance:topology:plane",
            (0.0, 0.0, 0.2),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        (0.0, 0.0, 0.2),
        (0.0, 1.0, 0.0),
        0.0,
        1.2,
    )
    scheduled = track_scheduled_plane_section(
        "acceptance:topology:section", cone, motion
    )
    facade = QuadricSection3D(
        scene,
        scheduled=scheduled,
        progress=tracker,
        projection=VIEW,
        transition_fraction=0.055,
        paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        style=STYLE,
        boundary_styles=BOUNDARY_STYLES,
        limits=acceptance_limits(transition=True),
        max_chord_error=0.02,
        include_surface_boundaries=True,
    ).attach()
    labels: list[Text] = []
    if with_labels:
        for label, x in (
            ("ELLIPSE", -2.7),
            ("PARABOLA", 0.0),
            ("HYPERBOLA", 2.9),
        ):
            labels.append(_caption(scene, label, x))
        foreground = _title(
            scene,
            "Automatic conic-family handoff",
            "ellipse → exact parabola → hyperbola with fixed Manim identities",
        )
        scene.add_foreground_mobjects(*foreground, *labels)
    state = AcceptanceState(
        "section_topology",
        tracker,
        0.0,
        1.0,
        (facade,),
        (("transition", facade.controller),),
        (lambda: TOPOLOGY_VIEW,),
    )
    state.set_progress(progress)
    return state


_BUILDERS = {
    "closed_open_comparison": _build_closed_open,
    "section_topology": _build_topology,
    "hidden_curve_policies": _build_policies,
    "side_view_trim_rim": _build_side_view,
    "cap_chord_activation": _build_cap_chord,
}


def acceptance_scenario_ids() -> tuple[str, ...]:
    return tuple(_BUILDERS)


def build_acceptance_state(
    scene: Scene,
    scenario_id: str,
    *,
    progress: float = 0.0,
    with_labels: bool = False,
) -> AcceptanceState:
    try:
        builder = _BUILDERS[scenario_id]
    except KeyError as exc:
        raise ValueError(f"unknown acceptance scenario {scenario_id!r}") from exc
    scene.camera.background_color = BACKGROUND_COLOR
    return builder(scene, float(progress), bool(with_labels))


class _AcceptanceVideoScene(Scene):
    scenario_id = ""
    run_time = 4.2
    # Store rate functions as static methods.  A plain function-valued class
    # attribute becomes a bound method when accessed through ``self``, which
    # would pass the Scene instance to Manim as the interpolation parameter.
    rate_function = staticmethod(linear)

    def construct(self) -> None:
        state = build_acceptance_state(
            self,
            self.scenario_id,
            progress=0.0,
            with_labels=True,
        )
        trace_path = os.environ.get(QUADRIC_CAIRO_FRAME_TRACE_ENV, "").strip()
        frames: list[dict[str, object]] = []
        original_render = self.renderer.render

        if trace_path:
            def traced_render(
                scene: Scene,
                scene_time: float,
                moving_mobjects=None,
            ) -> None:
                started_ns = perf_counter_ns()
                original_render(scene, scene_time, moving_mobjects)
                elapsed_ns = max(0, perf_counter_ns() - started_ns)
                controllers = []
                for label, controller in state.controllers:
                    snapshot = controller.performance_snapshot()
                    controllers.append(
                        {
                            "controllerId": label,
                            "performance": (
                                None if snapshot is None else snapshot.to_dict()
                            ),
                        }
                    )
                frames.append(
                    {
                        "frameIndex": len(frames),
                        "sceneTime": float(scene_time),
                        "cairoRenderNanoseconds": elapsed_ns,
                        "cairoRenderSeconds": elapsed_ns / 1_000_000_000.0,
                        "controllers": controllers,
                    }
                )

            self.renderer.render = traced_render  # type: ignore[method-assign]

        try:
            self.wait(0.30)
            self.play(
                state.tracker.animate.set_value(state.end_value),
                run_time=self.run_time,
                rate_func=self.rate_function,
            )
            self.wait(0.30)
        finally:
            if trace_path:
                self.renderer.render = original_render  # type: ignore[method-assign]
            try:
                if trace_path:
                    destination = Path(trace_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "schema": QUADRIC_CAIRO_FRAME_TRACE_SCHEMA,
                        "scenarioId": self.scenario_id,
                        "frameCount": len(frames),
                        "cairoRenderNanoseconds": sum(
                            int(item["cairoRenderNanoseconds"])
                            for item in frames
                        ),
                        "frames": frames,
                    }
                    temporary = destination.with_suffix(
                        destination.suffix + ".tmp"
                    )
                    temporary.write_text(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    temporary.replace(destination)
            finally:
                state.restore()


class ClosedOpenComparisonAcceptance(_AcceptanceVideoScene):
    scenario_id = "closed_open_comparison"


class SectionTopologyAcceptance(_AcceptanceVideoScene):
    scenario_id = "section_topology"
    run_time = 6.0
    rate_function = staticmethod(smooth)


class CurvePolicyComparisonAcceptance(_AcceptanceVideoScene):
    scenario_id = "hidden_curve_policies"


class SideViewTrimRimAcceptance(_AcceptanceVideoScene):
    scenario_id = "side_view_trim_rim"
    run_time = 4.0


class CapChordActivationAcceptance(_AcceptanceVideoScene):
    scenario_id = "cap_chord_activation"


__all__: Sequence[str] = (
    "AcceptanceState",
    "CapChordActivationAcceptance",
    "ClosedOpenComparisonAcceptance",
    "CurvePolicyComparisonAcceptance",
    "SectionTopologyAcceptance",
    "SideViewTrimRimAcceptance",
    "acceptance_scenario_ids",
    "build_acceptance_state",
    "near_side_view",
)
