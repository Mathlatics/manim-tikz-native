"""Side-by-side Cairo demo for legacy and unified semantic boundaries.

Render with::

    manim -pql examples/quadrics/unified_boundary_visibility_demo.py \
      UnifiedBoundaryVisibilityComparison
"""

from __future__ import annotations

from math import pi

import numpy as np
from manim import DOWN, Scene, Text, UP, ValueTracker, smooth

from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import ConeSpec, SectionPlane
from polyhedron_visibility.quadrics.manim import (
    DEFAULT_QUADRIC_VIEW,
    QuadricManimLimits,
    QuadricManimStyle,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.surface_boundaries import (
    GeneratorBoundarySpec,
)
from polyhedron_visibility.quadrics.transition_manim import (
    QuadricSectionTransition3D,
)


VIEW = DEFAULT_QUADRIC_VIEW


def _schedule(prefix: str, horizontal: float):
    shift = horizontal * np.asarray(VIEW.matrix[0], dtype=float)
    apex = shift + np.asarray((0.0, 0.0, -2.45), dtype=float)
    plane_center = shift + np.asarray((0.0, 0.0, 0.20), dtype=float)
    cone = ConeSpec(
        f"{prefix}-cone",
        tuple(float(value) for value in apex),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    motion = AxisAnglePlaneMotion(
        f"{prefix}-motion",
        SectionPlane(
            f"{prefix}-plane",
            tuple(float(value) for value in plane_center),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        tuple(float(value) for value in plane_center),
        (0.0, 1.0, 0.0),
        0.72,
        1.35,
    )
    return cone, track_scheduled_plane_section(
        f"{prefix}-section", cone, motion
    )


def _style() -> QuadricManimStyle:
    return QuadricManimStyle(
        surface_fill_color="#5275A8",
        surface_fill_opacity=0.72,
        surface_stroke_color="#17324F",
        surface_stroke_width=1.7,
        visible_curve_color="#F6C344",
        visible_curve_width=4.0,
        hidden_curve_color="#F6C344",
        hidden_curve_width=2.8,
        hidden_curve_opacity=0.92,
        dash_length=0.10,
        dash_gap=0.09,
        section_plane_fill_color="#63C7B2",
        section_plane_fill_opacity=0.15,
        section_plane_stroke_color="#7E57C2",
        section_plane_stroke_width=1.7,
        section_plane_stroke_opacity=0.95,
    )


def _limits() -> QuadricManimLimits:
    # A fragment shorter than max_projected_length can intersect at most 100
    # dash periods with the style above.  Sixteen possible semantic sources
    # therefore fit below this explicit total fixed-capacity ceiling.
    return QuadricManimLimits(
        max_surfaces=2,
        max_curves=8,
        max_fragments_per_curve=32,
        max_segments_per_fragment=384,
        max_surface_segments=768,
        max_dashes_per_fragment=100,
        max_projected_length=18.0,
        max_total_mobjects=60000,
        max_boundary_sources=48,
    )


class UnifiedBoundaryVisibilityComparison(Scene):
    """Legacy section ink versus unified semantic boundaries."""

    def construct(self) -> None:
        self.camera.background_color = "#F7FAFC"
        heading = Text(
            "Unified boundary visibility",
            font_size=28,
            color="#183153",
        ).to_edge(UP, buff=0.25)
        note = Text(
            "ellipse → parabola → hyperbola",
            font_size=17,
            color="#52606D",
        ).next_to(heading, DOWN, buff=0.10)
        left_label = Text(
            "LEGACY", font_size=21, color="#6B7280"
        ).move_to((-3.45, -3.22, 0.0))
        right_label = Text(
            "UNIFIED", font_size=21, color="#365C91"
        ).move_to((3.45, -3.22, 0.0))
        self.add(heading, note, left_label, right_label)

        progress = ValueTracker(0.0)
        _left_cone, left_schedule = _schedule("legacy", -3.25)
        right_cone, right_schedule = _schedule("unified", 3.25)

        legacy = QuadricSectionTransition3D(
            self,
            scheduled=left_schedule,
            progress=progress,
            projection=VIEW,
            transition_fraction=0.055,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            style=_style(),
            limits=_limits(),
            max_chord_error=0.008,
            painter_z_band=(20.0, 30.0),
            boundary_visibility_mode="legacy",
        ).attach()
        unified = QuadricSectionTransition3D(
            self,
            scheduled=right_schedule,
            progress=progress,
            projection=VIEW,
            transition_fraction=0.055,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            style=_style(),
            limits=_limits(),
            max_chord_error=0.008,
            painter_z_band=(40.0, 50.0),
            boundary_visibility_mode="unified",
            include_surface_boundaries=True,
            generator_boundaries=(
                GeneratorBoundarySpec(
                    "unified:teaching-generator",
                    right_cone.surface_id,
                    0.42,
                ),
            ),
        ).attach()

        self.wait(0.35)
        self.play(
            progress.animate.set_value(1.0),
            run_time=5.5,
            rate_func=smooth,
        )
        self.wait(0.45)
        unified.restore()
        legacy.restore()
