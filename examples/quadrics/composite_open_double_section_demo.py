"""Classroom comparison for the open-double section coordinator.

Quick preview::

    manim -ql examples/quadrics/composite_open_double_section_demo.py \
        CompositeOpenDoubleSectionDemo

Publication render::

    manim -qh examples/quadrics/composite_open_double_section_demo.py \
        CompositeOpenDoubleSectionDemo
"""

from __future__ import annotations

from math import pi

from manim import Scene, Text, UP, ValueTracker, linear

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricBoundaryStyle,
    QuadricManimLimits,
    QuadricManimStyle,
)


SIDE_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)


class CompositeOpenDoubleSectionDemo(Scene):
    """Move one plane through both finite nappes without replacing Mobjects."""

    def construct(self) -> None:
        self.camera.background_color = "#0B1723"
        title = Text(
            "Open double cone: one plane, two certified nappes",
            font_size=34,
            color="#F4F7FA",
        ).to_edge(UP, buff=0.18)
        title.set_z_index(40)
        subtitle = Text(
            "yellow = one mathematical hyperbola  •  cyan = real shell boundaries",
            font_size=21,
            color="#B9C7D5",
        ).next_to(title, direction=(0.0, -1.0, 0.0), buff=0.10)
        subtitle.set_z_index(40)
        self.add(title, subtitle)

        cone = ConeSpec(
            "classroom-double-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (-2.15, 2.15),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_DOUBLE,
        )
        offset = ValueTracker(0.34)

        def current_plane() -> SectionPlane:
            return SectionPlane(
                "classroom-double-cut",
                (0.0, offset.get_value(), 0.0),
                (0.0, 1.0, 0.16),
                u_axis=(1.0, 0.0, 0.0),
            )

        limits = QuadricManimLimits(
            max_surfaces=2,
            max_curves=8,
            max_fragments_per_curve=28,
            max_segments_per_fragment=384,
            max_surface_segments=512,
            max_dashes_per_fragment=96,
            max_projected_length=24.0,
            max_total_mobjects=50000,
            max_boundary_sources=32,
        )
        controller = CompositeQuadricSection3D(
            self,
            surface=cone,
            section_id="classroom-double-section",
            plane=current_plane,
            projection=SIDE_VIEW,
            paint_policy="depth_aware_diagrammatic",
            style=QuadricManimStyle(
                surface_fill_color="#2B6F9F",
                surface_fill_opacity=0.62,
                surface_stroke_opacity=0.0,
                section_plane_fill_color="#2CB9A4",
                section_plane_fill_opacity=0.18,
                section_plane_stroke_opacity=0.0,
                cone_lateral_fill_colors=(
                    "#173753",
                    "#4F9AC1",
                    "#1D4368",
                ),
                cone_lateral_sheen_direction=(1.0, 0.25, 0.0),
            ),
            boundary_styles={
                "style:curve": QuadricBoundaryStyle(
                    visible_color="#FFD866",
                    visible_width=4.0,
                    visible_opacity=1.0,
                    hidden_color="#FFD866",
                    hidden_width=3.0,
                    hidden_opacity=0.46,
                    dash_length=0.09,
                    dash_gap=0.07,
                ),
                "style:surface-boundary": QuadricBoundaryStyle(
                    visible_color="#61DDF2",
                    visible_width=3.2,
                    visible_opacity=0.95,
                    hidden_color="#61DDF2",
                    hidden_width=2.4,
                    hidden_opacity=0.34,
                    dash_length=0.09,
                    dash_gap=0.07,
                ),
                "style:surface-silhouette": QuadricBoundaryStyle(
                    visible_color="#61DDF2",
                    visible_width=3.2,
                    visible_opacity=0.95,
                    hidden_color="#61DDF2",
                    hidden_width=2.4,
                    hidden_opacity=0.34,
                    dash_length=0.09,
                    dash_gap=0.07,
                ),
                "style:section-outline": QuadricBoundaryStyle(
                    visible_color="#72D7C9",
                    visible_width=1.6,
                    visible_opacity=0.72,
                    hidden_color="#72D7C9",
                    hidden_width=1.4,
                    hidden_opacity=0.28,
                    dash_length=0.08,
                    dash_gap=0.06,
                ),
            },
            limits=limits,
            max_chord_error=0.025,
            section_max_screen_error=0.13,
            plane_patch_margin=0.17,
        ).attach()

        self.wait(0.35)
        self.play(offset.animate.set_value(0.82), run_time=2.8, rate_func=linear)
        self.play(offset.animate.set_value(0.42), run_time=2.4, rate_func=linear)
        self.wait(0.45)
