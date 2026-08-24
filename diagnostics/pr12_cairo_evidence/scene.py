"""Minimal Cairo evidence scenes for the merged PR #12 quadric compositor.

This module is diagnostic-only.  It deliberately imports and drives the public
production ``QuadricOcclusion3D`` binding without patching any implementation
function.  Environment variables select the still-frame state and opacity mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, floor, pi, sin
import os
from typing import Callable

import numpy as np
from manim import Scene, ValueTracker, smooth

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import ConeSpec, SectionPlane
from polyhedron_visibility.quadrics.curves import ParametricConicBranch
from polyhedron_visibility.quadrics.manim import (
    DEFAULT_QUADRIC_VIEW,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.sections import compute_quadric_section
from polyhedron_visibility.quadrics.trace import section_trace_curves


BACKGROUND_COLOR = "#F7F9FC"
SURFACE_COLOR = "#3478C8"
SURFACE_STROKE_COLOR = "#102A43"
PLANE_COLOR = "#FF7A2F"
PLANE_OUTLINE_COLOR = "#7C3AED"
VISIBLE_CURVE_COLOR = "#00A86B"
HIDDEN_CURVE_COLOR = "#D81B60"

PIXEL_WIDTH = 960
PIXEL_HEIGHT = 540
FPS = 24


def _screen_zoom(view: ParallelView, factor: float) -> ParallelView:
    matrix = view.matrix
    matrix[:2] *= factor
    return ParallelView.from_matrix(matrix)


DIAGNOSTIC_VIEW = _screen_zoom(DEFAULT_QUADRIC_VIEW, 0.78)
VIEW_DIRECTION = np.asarray(DIAGNOSTIC_VIEW.view_direction, dtype=float)
CONE_APEX = np.asarray((0.0, 0.0, -2.25), dtype=float)
CONE = ConeSpec(
    "diagnostic-cone",
    tuple(float(value) for value in CONE_APEX),
    (0.0, 0.0, 1.0),
    pi / 6.0,
    (0.0, 4.1),
    radial_axis=(1.0, 0.0, 0.0),
)


@dataclass(frozen=True, slots=True)
class DiagnosticState:
    name: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    description: str


def _unit(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("diagnostic direction must be finite and non-zero")
    return value / length


def _normal(theta: float) -> np.ndarray:
    return _unit(np.asarray((sin(theta), 0.0, cos(theta)), dtype=float))


def _state(
    name: str,
    *,
    theta: float,
    point: np.ndarray,
    description: str,
) -> DiagnosticState:
    normal = _normal(theta)
    return DiagnosticState(
        name,
        tuple(float(value) for value in point),
        tuple(float(value) for value in normal),
        description,
    )


PARABOLA_ANGLE = pi / 3.0
PARABOLA_NORMAL = _normal(PARABOLA_ANGLE)
AXIS = np.asarray((0.0, 0.0, 1.0), dtype=float)

# The states form one continuous diagnostic path.  The first and last planes
# are translated along the production view direction so most of their in-cone
# patch area lies respectively behind and in front of the two projection
# sheets.  The third plane is parallel to a cone tangent plane and only a small
# positive offset from the apex.  The fourth uses the exact generator-parallel
# (parabolic) angle with a clearly visible finite section.
STATES = (
    _state(
        "mainly_behind",
        theta=0.80,
        point=CONE_APEX + 1.45 * AXIS - 1.05 * VIEW_DIRECTION,
        description="plane patch predominantly behind the finite cone",
    ),
    _state(
        "intersects",
        theta=0.94,
        point=CONE_APEX + 1.55 * AXIS - 0.30 * VIEW_DIRECTION,
        description="ordinary oblique intersection with all local depth roles",
    ),
    _state(
        "near_tangent",
        theta=PARABOLA_ANGLE,
        point=CONE_APEX + 0.10 * PARABOLA_NORMAL,
        description="generator-parallel plane close to the tangent-plane limit",
    ),
    _state(
        "exact_parabola",
        theta=PARABOLA_ANGLE,
        point=CONE_APEX + 0.82 * PARABOLA_NORMAL + 0.18 * VIEW_DIRECTION,
        description="regular exact parabolic section",
    ),
    _state(
        "mainly_front",
        theta=1.20,
        point=CONE_APEX + 1.45 * AXIS + 1.05 * VIEW_DIRECTION,
        description="plane patch predominantly in front of the finite cone",
    ),
)
STATE_INDEX = {state.name: index for index, state in enumerate(STATES)}
ALLOCATED_CURVE_IDS = tuple(
    f"diagnostic-section:curve-slot:{index}" for index in range(4)
)


def diagnostic_mode() -> str:
    value = os.environ.get("PR12_DIAGNOSTIC_MODE", "translucent").strip().lower()
    if value not in {"translucent", "opaque", "surface_only"}:
        raise ValueError(
            "PR12_DIAGNOSTIC_MODE must be translucent, opaque, or surface_only"
        )
    return value


def diagnostic_state_name() -> str:
    value = os.environ.get("PR12_DIAGNOSTIC_STATE", "exact_parabola").strip()
    if value not in STATE_INDEX:
        raise ValueError(
            "PR12_DIAGNOSTIC_STATE must be one of " + ", ".join(STATE_INDEX)
        )
    return value


def plane_at_progress(progress: float) -> SectionPlane:
    value = min(float(len(STATES) - 1), max(0.0, float(progress)))
    left_index = min(len(STATES) - 1, int(floor(value)))
    right_index = min(len(STATES) - 1, left_index + 1)
    alpha = value - left_index
    left = STATES[left_index]
    right = STATES[right_index]
    point = (1.0 - alpha) * np.asarray(left.point) + alpha * np.asarray(right.point)
    normal = _unit(
        (1.0 - alpha) * np.asarray(left.normal)
        + alpha * np.asarray(right.normal)
    )
    return SectionPlane(
        "diagnostic-plane",
        tuple(float(component) for component in point),
        tuple(float(component) for component in normal),
        u_axis=(0.0, 1.0, 0.0),
    )


def style_for_mode(mode: str) -> QuadricManimStyle:
    if mode == "opaque":
        surface_opacity = 1.0
        plane_opacity = 1.0
        outline_opacity = 1.0
        visible_opacity = 1.0
        hidden_opacity = 1.0
    elif mode == "surface_only":
        surface_opacity = 0.58
        plane_opacity = 0.0
        outline_opacity = 0.0
        visible_opacity = 0.0
        hidden_opacity = 0.0
    else:
        surface_opacity = 0.58
        plane_opacity = 0.36
        outline_opacity = 0.95
        visible_opacity = 0.98
        hidden_opacity = 0.92
    return QuadricManimStyle(
        surface_fill_color=SURFACE_COLOR,
        surface_fill_opacity=surface_opacity,
        surface_stroke_color=SURFACE_STROKE_COLOR,
        surface_stroke_width=2.8,
        surface_stroke_opacity=1.0,
        visible_curve_color=VISIBLE_CURVE_COLOR,
        visible_curve_width=6.2,
        visible_curve_opacity=visible_opacity,
        hidden_curve_color=HIDDEN_CURVE_COLOR,
        hidden_curve_width=5.2,
        hidden_curve_opacity=hidden_opacity,
        dash_length=0.13,
        dash_gap=0.10,
        background_width=0.0,
        background_opacity=0.0,
        section_plane_fill_color=PLANE_COLOR,
        section_plane_fill_opacity=plane_opacity,
        section_plane_stroke_color=PLANE_OUTLINE_COLOR,
        section_plane_stroke_width=4.0,
        section_plane_stroke_opacity=outline_opacity,
    )


def diagnostic_limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=1,
        max_curves=len(ALLOCATED_CURVE_IDS),
        max_fragments_per_curve=24,
        max_segments_per_fragment=1024,
        max_surface_segments=2048,
        max_dashes_per_fragment=192,
        max_projected_length=24.0,
        max_total_mobjects=20000,
    )


def build_controller(
    scene: Scene,
    progress_provider: Callable[[], float],
    mode: str,
) -> QuadricOcclusion3D:
    cache: dict[float, tuple[ParametricConicBranch, ...]] = {}

    def active_curves() -> tuple[ParametricConicBranch, ...]:
        progress = float(progress_provider())
        key = round(progress, 12)
        cached = cache.get(key)
        if cached is not None:
            return cached
        plane = plane_at_progress(progress)
        trace = compute_quadric_section(
            "diagnostic-section",
            CONE,
            plane,
        )
        source = tuple(
            sorted(
                section_trace_curves(trace),
                key=lambda item: (
                    item.domain.start,
                    item.domain.end,
                    item.curve_id,
                ),
            )
        )
        if len(source) > len(ALLOCATED_CURVE_IDS):
            raise RuntimeError(
                f"diagnostic section needs {len(source)} curve slots; "
                f"capacity={len(ALLOCATED_CURVE_IDS)}"
            )
        result = tuple(
            ParametricConicBranch(
                ALLOCATED_CURVE_IDS[index],
                curve.parameterization,
                curve.plane_embedding,
                curve.domain,
            )
            for index, curve in enumerate(source)
        )
        cache[key] = result
        return result

    def curve_opacities() -> dict[str, float]:
        return {curve.curve_id: 1.0 for curve in active_curves()}

    return QuadricOcclusion3D(
        scene,
        surfaces=(CONE,),
        curves=active_curves,
        curve_opacities=curve_opacities,
        allocated_curve_ids=ALLOCATED_CURVE_IDS,
        projection=DIAGNOSTIC_VIEW,
        paint_policy="diagrammatic",
        style=style_for_mode(mode),
        limits=diagnostic_limits(),
        max_chord_error=0.006,
        painter_z_band=(20.0, 30.0),
        surface_order_mode="automatic",
        section_plane=lambda: plane_at_progress(progress_provider()),
        section_patch_margin=0.22,
        section_max_screen_error=0.035,
    )


class CairoConeDiagnosticStill(Scene):
    """One exact diagnostic state selected by environment variables."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND_COLOR
        index = float(STATE_INDEX[diagnostic_state_name()])
        build_controller(self, lambda: index, diagnostic_mode()).attach()
        self.wait(0.12)


class CairoConeDiagnosticVideo(Scene):
    """Move the production cutting plane through all five evidence states."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND_COLOR
        progress = ValueTracker(0.0)
        build_controller(
            self,
            progress.get_value,
            diagnostic_mode(),
        ).attach()
        self.wait(0.55)
        for index in range(1, len(STATES)):
            self.play(
                progress.animate.set_value(float(index)),
                run_time=1.35,
                rate_func=smooth,
            )
            self.wait(0.55)
        self.wait(0.35)


__all__ = [
    "ALLOCATED_CURVE_IDS",
    "BACKGROUND_COLOR",
    "CONE",
    "CairoConeDiagnosticStill",
    "CairoConeDiagnosticVideo",
    "DIAGNOSTIC_VIEW",
    "FPS",
    "HIDDEN_CURVE_COLOR",
    "PIXEL_HEIGHT",
    "PIXEL_WIDTH",
    "PLANE_COLOR",
    "PLANE_OUTLINE_COLOR",
    "STATE_INDEX",
    "STATES",
    "SURFACE_COLOR",
    "VISIBLE_CURVE_COLOR",
    "build_controller",
    "diagnostic_limits",
    "plane_at_progress",
    "style_for_mode",
]
