"""Compare top-overlay and depth-aware hidden cone boundaries."""

from __future__ import annotations

from math import pi

import numpy as np
from manim import DOWN, Scene, Text, UP, ValueTracker, linear

from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    DEFAULT_QUADRIC_VIEW,
    QuadricBoundaryStyle,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)


VIEW = DEFAULT_QUADRIC_VIEW


def limits() -> QuadricManimLimits:
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


STYLE = QuadricManimStyle(
    surface_fill_color="#315A8A",
    surface_fill_opacity=0.78,
    surface_stroke_opacity=0.0,
    cone_lateral_fill_colors=("#173753", "#4F84B3", "#1D4368"),
    cone_cap_fill_colors=("#557A99", "#294B6B"),
    cone_lateral_sheen_direction=(1.0, 0.0, 0.0),
    cone_cap_sheen_direction=(-1.0, 1.0, 0.0),
    section_plane_fill_color="#43D9C0",
    section_plane_fill_opacity=0.34,
    section_plane_stroke_color="#B39DDB",
    section_plane_stroke_width=1.8,
    section_plane_stroke_opacity=0.9,
    dash_length=0.12,
    dash_gap=0.09,
)


CONE_BOUNDARY_STYLE = QuadricBoundaryStyle(
    visible_color="#5CE1E6",
    visible_width=4.4,
    visible_opacity=1.0,
    hidden_color="#5CE1E6",
    hidden_width=3.0,
    hidden_opacity=0.24,
    dash_length=0.12,
    dash_gap=0.09,
)


class SectionPlaneConeBoundaryDemo(Scene):
    def construct(self) -> None:
        self.camera.background_color = "#101820"
        heading = Text(
            "Section plane occludes cone boundaries",
            font_size=27,
            color="#F4F7FB",
        ).to_edge(UP, buff=0.24)
        note = Text(
            "same cyan boundary  •  hidden parts become faint dashes",
            font_size=15,
            color="#B8C5D6",
        ).next_to(heading, DOWN, buff=0.10)
        heading.set_z_index(100)
        note.set_z_index(100)
        self.add(heading, note)

        offset = ValueTracker(-0.48)
        normal = np.asarray((0.82, 0.0, 1.0), dtype=float)
        normal /= np.linalg.norm(normal)
        controllers = []
        captions = []
        vertical_shift = -0.55 * np.asarray(VIEW.matrix[1], dtype=float)
        for index, (label, policy, horizontal) in enumerate(
            (
                ("TOP OVERLAY DASH", QuadricPaintPolicy.DIAGRAMMATIC, -3.35),
                (
                    "DEPTH-AWARE DASH",
                    QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                    3.35,
                ),
            )
        ):
            shift = (
                horizontal * np.asarray(VIEW.matrix[0], dtype=float)
                + vertical_shift
            )
            cone = ConeSpec(
                f"demo-cone-{index}",
                tuple(shift + np.asarray((0.0, 0.0, -2.4))),
                (0.0, 0.0, 1.0),
                pi / 6.0,
                (0.0, 4.0),
                radial_axis=(1.0, 0.0, 0.0),
                model=ConeModel.CLOSED_SINGLE,
            )

            def current_plane(
                plane_id=f"demo-plane-{index}",
                world_shift=shift.copy(),
            ) -> SectionPlane:
                point = (
                    world_shift
                    + np.asarray((0.0, 0.0, -0.35))
                    + offset.get_value() * normal
                )
                return SectionPlane(
                    plane_id,
                    tuple(float(value) for value in point),
                    (0.82, 0.0, 1.0),
                    u_axis=(0.0, 1.0, 0.0),
                )

            controller = QuadricOcclusion3D(
                self,
                surfaces=(cone,),
                curves=(),
                projection=VIEW,
                paint_policy=policy,
                style=STYLE,
                boundary_styles={
                    "style:surface-silhouette": CONE_BOUNDARY_STYLE,
                    "style:surface-boundary": CONE_BOUNDARY_STYLE,
                },
                limits=limits(),
                max_chord_error=0.008,
                section_plane=current_plane,
                boundary_visibility_mode="unified",
                include_surface_boundaries=True,
                painter_z_band=(20.0 + 20.0 * index, 30.0 + 20.0 * index),
            ).attach()
            controllers.append(controller)
            caption = Text(label, font_size=19, color="#DCE6F2").move_to(
                (horizontal, -3.25, 0.0)
            )
            caption.set_z_index(100)
            self.add(caption)
            captions.append(caption)

        self.add_foreground_mobjects(heading, note, *captions)
        self.wait(0.35)
        self.play(offset.animate.set_value(0.48), run_time=4.2, rate_func=linear)
        self.wait(0.35)
        for controller in reversed(controllers):
            controller.restore()
