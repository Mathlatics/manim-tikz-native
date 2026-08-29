"""Small public-API example for one moving finite cone section."""

from math import pi

from manim import Scene, ValueTracker, linear

from polyhedron_visibility.quadrics import (
    ConeSpec,
    QuadricManimStyle,
    QuadricSection3D,
    SectionPlane,
)


RENDER_PROFILE = "preview"  # Change to "final" for classroom output.


class ConeSectionQuickStart(Scene):
    def construct(self) -> None:
        progress = ValueTracker(0.0)
        cone = ConeSpec(
            "cone", (0, 0, -1.5), (0, 0, 1), pi / 6, (0, 4)
        )

        def plane() -> SectionPlane:
            return SectionPlane(
                "cut", (0, 0, -1.0 + 2.7 * progress.get_value()),
                (0.65, 0, 1), u_axis=(0, 1, 0),
            )

        QuadricSection3D(
            self, surface=cone, section_id="cone-section", plane=plane,
            paint_policy="depth_aware_diagrammatic",
            style=QuadricManimStyle(
                surface_fill_opacity=0.62,
                visible_curve_color="#FFD166", hidden_curve_color="#FFD166",
            ),
            render_profile=RENDER_PROFILE,
        ).attach()
        self.play(progress.animate.set_value(1), run_time=4, rate_func=linear)
        self.wait(0.25)
