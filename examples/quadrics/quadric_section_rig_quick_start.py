"""Natural fixed-topology actions for one finite cone section."""

from math import pi

from manim import Scene, linear

from polyhedron_visibility.quadrics import (
    ConeSpec,
    QuadricManimStyle,
    QuadricSectionRig,
    SectionPlane,
)


class ConeSectionRigQuickStart(Scene):
    def construct(self) -> None:
        cone = ConeSpec(
            "cone",
            (0, 0, -1.5),
            (0, 0, 1),
            pi / 6,
            (0, 4),
        )
        initial_plane = SectionPlane(
            "cut",
            (0, 0, -0.4),
            (0.45, 0, 1),
            u_axis=(0, 1, 0),
        )

        with QuadricSectionRig(
            self,
            surface=cone,
            section_id="cone-section",
            plane=initial_plane,
            paint_policy="depth_aware_diagrammatic",
            render_profile="preview",
            style=QuadricManimStyle(
                surface_fill_opacity=0.62,
                visible_curve_color="#FFD166",
                hidden_curve_color="#FFD166",
            ),
        ).session() as section:
            self.play(
                section.animate_plane_shift(0.6),
                run_time=2.0,
                rate_func=linear,
            )
            self.play(
                section.animate_plane_rotation(
                    axis=(0, 0, 1),
                    angle=pi / 3,
                    pivot=cone.apex,
                ),
                run_time=2.0,
                rate_func=linear,
            )
            self.wait(0.25)
