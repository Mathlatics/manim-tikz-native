"""Cairo comparison for two-terminal frustum component shading.

Preview with::

    manim -ql --fps 8 \
      examples/quadrics/frustum_component_shading_demo.py \
      FrustumComponentShadingComparison
"""

from __future__ import annotations

from math import pi

import numpy as np
from manim import Scene, Text, ValueTracker, smooth

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.authoring import QuadricSection3D
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import QuadricManimStyle


BACKGROUND = "#101820"
BASE_VIEW = np.asarray(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    ),
    dtype=float,
)
DISPLAY_VIEW_MATRIX = BASE_VIEW.copy()
DISPLAY_VIEW_MATRIX[:2] *= 0.76
DISPLAY_VIEW = ParallelView.from_matrix(DISPLAY_VIEW_MATRIX)


def _style(*, component: bool) -> QuadricManimStyle:
    return QuadricManimStyle(
        surface_fill_color="#315A8A",
        surface_fill_opacity=0.76,
        surface_stroke_color="#5EE7F2",
        surface_stroke_width=2.5,
        surface_stroke_opacity=0.9,
        cone_lateral_fill_colors=(
            ("#173753", "#4F84B3", "#1D4368") if component else None
        ),
        cone_cap_fill_colors=(
            ("#8A6A3D", "#D4A85F") if component else None
        ),
        cone_lateral_sheen_direction=(1.0, 0.0, 0.0),
        cone_cap_sheen_direction=(-1.0, 1.0, 0.0),
        visible_curve_color="#FFD166",
        visible_curve_width=4.0,
        hidden_curve_color="#F59E0B",
        hidden_curve_width=3.2,
        hidden_curve_opacity=0.65,
        section_plane_fill_color="#43D9C0",
        section_plane_fill_opacity=0.27,
        section_plane_stroke_color="#A8FFF0",
        section_plane_stroke_width=1.6,
        section_plane_stroke_opacity=0.75,
    )


def _authoring(
    scene: Scene,
    *,
    prefix: str,
    horizontal: float,
    offset: ValueTracker,
    component: bool,
    painter_band: tuple[float, float],
) -> QuadricSection3D:
    screen_horizontal = np.asarray(BASE_VIEW[0], dtype=float)
    translation = horizontal / 0.76 * screen_horizontal
    frustum = ConeSpec(
        f"{prefix}:frustum",
        tuple(float(item) for item in translation),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (0.75, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.CLOSED_SINGLE,
    )

    def current_plane() -> SectionPlane:
        point = translation + np.asarray((offset.get_value(), 0.0, 1.3))
        return SectionPlane(
            f"{prefix}:plane",
            tuple(float(item) for item in point),
            (1.0, 0.0, 0.0),
            u_axis=(0.0, 1.0, 0.0),
        )

    return QuadricSection3D(
        scene,
        surface=frustum,
        section_id=f"{prefix}:section",
        plane=current_plane,
        projection=DISPLAY_VIEW,
        paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        style=_style(component=component),
        max_chord_error=0.012,
        section_max_screen_error=0.07,
        painter_z_band=painter_band,
        include_surface_boundaries=True,
    ).attach()


class FrustumComponentShadingComparison(Scene):
    """Compare uniform fill with side/two-cap component shading."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        offset = ValueTracker(0.18)
        uniform = _authoring(
            self,
            prefix="uniform",
            horizontal=-3.25,
            offset=offset,
            component=False,
            painter_band=(20.0, 30.0),
        )
        component = _authoring(
            self,
            prefix="component",
            horizontal=3.25,
            offset=offset,
            component=True,
            painter_band=(40.0, 50.0),
        )

        title = Text(
            "Frustum: lateral sheet + two real caps",
            font_size=31,
            color="#F3F7FB",
        ).move_to((0.0, 3.35, 0.0))
        subtitle = Text(
            "yellow = two cap chords   cyan = terminal rims",
            font_size=18,
            color="#C9D7E5",
        ).move_to((0.0, 2.88, 0.0))
        left = Text("UNIFORM FILL", font_size=20, color="#DCE6F2").move_to(
            (-3.25, -3.2, 0.0)
        )
        right = Text(
            "COMPONENT SHADING", font_size=20, color="#FFD166"
        ).move_to((3.25, -3.2, 0.0))
        for label in (title, subtitle, left, right):
            label.set_z_index(100)
            self.add_foreground_mobjects(label)

        self.wait(0.6)
        self.play(offset.animate.set_value(0.52), run_time=2.6, rate_func=smooth)
        self.play(offset.animate.set_value(0.18), run_time=2.6, rate_func=smooth)
        self.wait(0.8)

        component.restore()
        uniform.restore()
