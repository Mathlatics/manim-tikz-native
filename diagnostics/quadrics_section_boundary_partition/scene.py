"""Five-state Cairo scene for the quadric section-boundary repair baseline.

This module is diagnostic-only.  It drives the public production
``QuadricOcclusion3D`` controller without replacing or patching any production
function.  The fill-only styles deliberately remove curves and strokes so an
interior pixel deviation measures fill geometry rather than authored line art.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin, cos
import os
from typing import Callable

import numpy as np
from manim import Scene

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
PLANE_COLOR = "#FF7A2F"
PIXEL_WIDTH = 960
PIXEL_HEIGHT = 540
FPS = 24


def _unit(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("diagnostic direction must be finite and non-zero")
    return value / length


def _normal(theta: float) -> np.ndarray:
    return _unit(np.asarray((sin(theta), 0.0, cos(theta)), dtype=float))


def _screen_zoom(view: ParallelView, factor: float) -> ParallelView:
    matrix = view.matrix
    matrix[:2] *= factor
    return ParallelView.from_matrix(matrix)


DIAGNOSTIC_VIEW = _screen_zoom(DEFAULT_QUADRIC_VIEW, 0.78)
VIEW_DIRECTION = np.asarray(DIAGNOSTIC_VIEW.view_direction, dtype=float)
CONE_APEX = np.asarray((0.0, 0.0, -2.25), dtype=float)
WORLD_Z = np.asarray((0.0, 0.0, 1.0), dtype=float)
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


def _state(
    name: str,
    *,
    theta: float,
    point: np.ndarray,
    description: str,
) -> DiagnosticState:
    normal = _normal(theta)
    return DiagnosticState(
        name=name,
        point=tuple(float(value) for value in point),
        normal=tuple(float(value) for value in normal),
        description=description,
    )


PARABOLA_ANGLE = pi / 3.0
PARABOLA_NORMAL = _normal(PARABOLA_ANGLE)
STATES = (
    _state(
        "mainly_behind",
        theta=0.80,
        point=CONE_APEX + 1.45 * WORLD_Z - 1.05 * VIEW_DIRECTION,
        description="plane patch predominantly behind the finite cone",
    ),
    _state(
        "intersects",
        theta=0.94,
        point=CONE_APEX + 1.55 * WORLD_Z - 0.30 * VIEW_DIRECTION,
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
        point=(
            CONE_APEX
            + 0.82 * PARABOLA_NORMAL
            + 0.18 * VIEW_DIRECTION
        ),
        description="regular exact parabolic section",
    ),
    _state(
        "mainly_front",
        theta=1.20,
        point=CONE_APEX + 1.45 * WORLD_Z + 1.05 * VIEW_DIRECTION,
        description="plane patch predominantly in front of the finite cone",
    ),
)
STATE_BY_NAME = {state.name: state for state in STATES}
ALLOCATED_CURVE_IDS = tuple(
    f"diagnostic-section:curve-slot:{index}" for index in range(4)
)


def plane_for_state(name: str) -> SectionPlane:
    try:
        state = STATE_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            "unknown diagnostic state; expected one of "
            + ", ".join(STATE_BY_NAME)
        ) from exc
    return SectionPlane(
        "diagnostic-plane",
        state.point,
        state.normal,
        u_axis=(0.0, 1.0, 0.0),
    )


def style_for_mode(mode: str) -> QuadricManimStyle:
    if mode == "opaque_fill":
        surface_opacity = 1.0
        plane_opacity = 1.0
    elif mode == "translucent_fill":
        surface_opacity = 0.58
        plane_opacity = 0.36
    else:
        raise ValueError("mode must be opaque_fill or translucent_fill")
    return QuadricManimStyle(
        surface_fill_color=SURFACE_COLOR,
        surface_fill_opacity=surface_opacity,
        surface_stroke_color=SURFACE_COLOR,
        surface_stroke_width=0.0,
        surface_stroke_opacity=0.0,
        visible_curve_color=SURFACE_COLOR,
        visible_curve_width=0.0,
        visible_curve_opacity=0.0,
        hidden_curve_color=SURFACE_COLOR,
        hidden_curve_width=0.0,
        hidden_curve_opacity=0.0,
        dash_length=0.13,
        dash_gap=0.10,
        background_width=0.0,
        background_opacity=0.0,
        section_plane_fill_color=PLANE_COLOR,
        section_plane_fill_opacity=plane_opacity,
        section_plane_stroke_color=PLANE_COLOR,
        section_plane_stroke_width=0.0,
        section_plane_stroke_opacity=0.0,
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
    state_provider: Callable[[], str],
    mode: str,
) -> QuadricOcclusion3D:
    cache: dict[str, tuple[ParametricConicBranch, ...]] = {}

    def active_curves() -> tuple[ParametricConicBranch, ...]:
        state_name = state_provider()
        cached = cache.get(state_name)
        if cached is not None:
            return cached
        trace = compute_quadric_section(
            "diagnostic-section",
            CONE,
            plane_for_state(state_name),
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
        cache[state_name] = result
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
        section_plane=lambda: plane_for_state(state_provider()),
        section_patch_margin=0.22,
        section_max_screen_error=0.035,
    )


class BoundaryBaselineStill(Scene):
    """Render one fill-only diagnostic state selected by environment."""

    def construct(self) -> None:
        state_name = os.environ.get(
            "QUADRIC_BASELINE_STATE",
            "exact_parabola",
        )
        mode = os.environ.get(
            "QUADRIC_BASELINE_MODE",
            "translucent_fill",
        )
        self.camera.background_color = BACKGROUND_COLOR
        build_controller(self, lambda: state_name, mode).attach()
        self.wait(0.12)


__all__ = [
    "ALLOCATED_CURVE_IDS",
    "BACKGROUND_COLOR",
    "BoundaryBaselineStill",
    "CONE",
    "DIAGNOSTIC_VIEW",
    "DiagnosticState",
    "FPS",
    "PIXEL_HEIGHT",
    "PIXEL_WIDTH",
    "PLANE_COLOR",
    "STATES",
    "STATE_BY_NAME",
    "SURFACE_COLOR",
    "build_controller",
    "diagnostic_limits",
    "plane_for_state",
    "style_for_mode",
]
