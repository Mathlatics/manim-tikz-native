"""Compare the supported finite cone models and their plane interaction."""

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
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
    section_cap_chord_curve_ids,
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
        max_total_mobjects=70000,
        max_boundary_sources=48,
    )


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


class ConeModelComparisonDemo(Scene):
    """Closed single, open single, and finite open double cone shells."""

    def construct(self) -> None:
        self.camera.background_color = "#101820"
        heading = Text(
            "Finite cone models",
            font_size=29,
            color="#F4F7FB",
        ).to_edge(UP, buff=0.24)
        note = Text(
            "closed solid = side + base  •  open shells = side + trim rim",
            font_size=15,
            color="#B8C5D6",
        ).next_to(heading, DOWN, buff=0.10)
        heading.set_z_index(100)
        note.set_z_index(100)
        self.add(heading, note)

        tilt = ValueTracker(-0.10)
        controllers = []
        captions = []
        rows = (
            ("CLOSED SINGLE", ConeModel.CLOSED_SINGLE, (0.0, 3.5), -4.3),
            ("OPEN SINGLE", ConeModel.OPEN_SINGLE, (0.0, 3.5), 0.0),
            ("OPEN DOUBLE", ConeModel.OPEN_DOUBLE, (-1.75, 1.75), 4.3),
        )
        vertical_shift = -0.45 * np.asarray(VIEW.matrix[1], dtype=float)
        for index, (label, model, axial_range, horizontal) in enumerate(rows):
            shift = (
                horizontal * np.asarray(VIEW.matrix[0], dtype=float)
                + vertical_shift
            )
            apex_offset = (
                np.asarray((0.0, 0.0, -1.75))
                if model is not ConeModel.OPEN_DOUBLE
                else np.zeros(3)
            )

            def current_surface(
                surface_id=f"model-cone-{index}",
                world_shift=shift.copy(),
                local_apex=apex_offset.copy(),
                cone_model=model,
                cone_range=axial_range,
            ) -> tuple[ConeSpec, ...]:
                return (
                    ConeSpec(
                        surface_id,
                        tuple(world_shift + local_apex),
                        (0.0, tilt.get_value(), 1.0),
                        pi / 6.0,
                        cone_range,
                        radial_axis=(1.0, 0.0, 0.0),
                        model=cone_model,
                    ),
                )

            controller = QuadricOcclusion3D(
                self,
                surfaces=current_surface,
                curves=(),
                projection=VIEW,
                paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                style=STYLE,
                boundary_styles=BOUNDARY_STYLES,
                limits=limits(),
                max_chord_error=0.01,
                boundary_visibility_mode="unified",
                include_surface_boundaries=True,
                painter_z_band=(20.0 + 16.0 * index, 30.0 + 16.0 * index),
            ).attach()
            controllers.append(controller)
            caption = Text(label, font_size=18, color="#DCE6F2").move_to(
                (horizontal, -3.05, 0.0)
            )
            caption.set_z_index(100)
            captions.append(caption)
            self.add(caption)

        self.add_foreground_mobjects(heading, note, *captions)
        self.wait(0.4)
        self.play(tilt.animate.set_value(0.16), run_time=4.0, rate_func=linear)
        self.wait(0.4)
        for controller in reversed(controllers):
            controller.restore()


class ConeModelPlaneComparisonDemo(Scene):
    """The same plane crossing a closed solid and an open single shell."""

    def construct(self) -> None:
        self.camera.background_color = "#101820"
        heading = Text(
            "Plane interaction: closed cone vs open shell",
            font_size=27,
            color="#F4F7FB",
        ).to_edge(UP, buff=0.24)
        note = Text(
            "yellow = true section  •  cyan = silhouette / rim",
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
        for index, (label, model, horizontal) in enumerate(
            (
                ("CLOSED SOLID", ConeModel.CLOSED_SINGLE, -3.35),
                ("OPEN SHELL", ConeModel.OPEN_SINGLE, 3.35),
            )
        ):
            shift = (
                horizontal * np.asarray(VIEW.matrix[0], dtype=float)
                + vertical_shift
            )
            cone = ConeSpec(
                f"plane-model-cone-{index}",
                tuple(shift + np.asarray((0.0, 0.0, -2.4))),
                (0.0, 0.0, 1.0),
                pi / 6.0,
                (0.0, 4.0),
                radial_axis=(1.0, 0.0, 0.0),
                model=model,
            )

            def current_plane(
                plane_id=f"plane-model-cut-{index}",
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

            section_id = f"plane-model-section-{index}"

            def current_curves(
                current_surface=cone,
                plane_callback=current_plane,
                current_section_id=section_id,
            ):
                return compute_quadric_section_boundary_curves(
                    current_section_id,
                    current_surface,
                    plane_callback(),
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
                self,
                surfaces=(cone,),
                curves=current_curves,
                projection=VIEW,
                paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                style=STYLE,
                boundary_styles=BOUNDARY_STYLES,
                limits=limits(),
                max_chord_error=0.008,
                section_plane=current_plane,
                boundary_visibility_mode="unified",
                include_surface_boundaries=True,
                allocated_curve_ids=allocated_curve_ids,
                painter_z_band=(20.0 + 20.0 * index, 30.0 + 20.0 * index),
            ).attach()
            controllers.append(controller)
            caption = Text(label, font_size=18, color="#DCE6F2").move_to(
                (horizontal, -3.25, 0.0)
            )
            caption.set_z_index(100)
            captions.append(caption)
            self.add(caption)

        self.add_foreground_mobjects(heading, note, *captions)
        self.wait(0.35)
        self.play(offset.animate.set_value(0.48), run_time=4.2, rate_func=linear)
        self.wait(0.35)
        for controller in reversed(controllers):
            controller.restore()
