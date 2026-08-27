"""Runnable Cairo demos for analytic quadric sections and occlusion.

Render all scenes at preview quality with::

    manim -pql examples/quadrics/quadric_occlusion_demo.py \
      MovingSphereSectionDemo ObliqueCylinderSectionDemo \
      ConeSectionFamiliesDemo ConeSectionTopologyTransitionDemo \
      GlobalQuadricOcclusionDemo

The mathematical surface and curve records are immutable Python data.  During
an animation the callbacks create a fresh record for the current tracker
value; :class:`QuadricOcclusion3D` mutates only its preallocated Manim slots.
"""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
from manim import DOWN, UP, Scene, Text, ValueTracker, smooth

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.authoring import QuadricSection3D
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.manim import (
    DEFAULT_QUADRIC_VIEW,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.sections import compute_quadric_section
from polyhedron_visibility.quadrics.trace import section_trace_curves


GLOBAL_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.32),
        (0.0, 1.0, 0.18),
        (0.0, 0.0, 1.0),
    )
)

QUADRIC_VIEW = DEFAULT_QUADRIC_VIEW


def _screen_zoom(view: ParallelView, factor: float) -> ParallelView:
    matrix = view.matrix
    matrix[:2] *= factor
    return ParallelView.from_matrix(matrix)


TRANSITION_VIEW = _screen_zoom(QUADRIC_VIEW, 0.85)


def _style(
    *, fill_color: str = "#3B82F6", fill_opacity: float = 1.0
) -> QuadricManimStyle:
    return QuadricManimStyle(
        surface_fill_color=fill_color,
        surface_fill_opacity=fill_opacity,
        surface_stroke_color="#17324F",
        surface_stroke_width=1.6,
        visible_curve_color="#F6C344",
        visible_curve_width=4.4,
        hidden_curve_color="#F6C344",
        hidden_curve_width=3.0,
        hidden_curve_opacity=0.9,
        dash_length=0.10,
        dash_gap=0.10,
    )


def _title(scene: Scene, text: str, subtitle: str) -> None:
    scene.camera.background_color = "#F7FAFC"
    heading = Text(text, font_size=28, color="#183153").to_edge(UP, buff=0.28)
    note = Text(subtitle, font_size=17, color="#52606D").next_to(
        heading, DOWN, buff=0.12
    )
    scene.add(heading, note)


class MovingSphereSectionDemo(Scene):
    """A translated infinite plane keeps one stable circular section branch."""

    def construct(self) -> None:
        _title(
            self,
            "Moving sphere section",
            "rear arc is recomputed as a hidden dashed span on every frame",
        )
        sphere = SphereSpec("sphere", (0.0, -0.25, 0.0), 2.0)
        offset = ValueTracker(-1.25)
        normal = np.asarray((1.0, 0.35, 0.25), dtype=float)
        normal /= np.linalg.norm(normal)

        def curves():
            distance = offset.get_value()
            plane = SectionPlane(
                "moving-plane",
                tuple(float(value) for value in distance * normal),
                tuple(float(value) for value in normal),
                u_axis=(0.0, 1.0, 0.0),
            )
            trace = compute_quadric_section("sphere-section", sphere, plane)
            return section_trace_curves(trace)

        controller = QuadricOcclusion3D(
            self,
            surfaces=(sphere,),
            curves=curves,
            paint_policy="diagrammatic",
            style=_style(fill_color="#3976B8"),
            max_chord_error=0.008,
        ).attach()
        self.wait(0.5)
        self.play(offset.animate.set_value(1.25), run_time=4.0, rate_func=smooth)
        self.play(offset.animate.set_value(-0.55), run_time=2.0, rate_func=smooth)
        self.wait(0.5)
        controller.restore()


class ObliqueCylinderSectionDemo(Scene):
    """A rotating plane remains in the finite cylinder's ellipse family."""

    def construct(self) -> None:
        _title(
            self,
            "Oblique finite-cylinder section",
            "analytic ellipse, finite axial clipping, and automatic hidden arc",
        )
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.45,
            (-2.8, 2.8),
            radial_axis=(1.0, 0.0, 0.0),
        )
        tilt = ValueTracker(0.0)

        def curves():
            progress = tilt.get_value()
            plane = SectionPlane(
                "rotating-plane",
                (0.0, 0.0, 0.0),
                (0.18 + 0.42 * progress, 0.12 + 0.18 * progress, 1.0),
                u_axis=(1.0, 0.0, 0.0),
            )
            trace = compute_quadric_section("cylinder-section", cylinder, plane)
            return section_trace_curves(trace)

        controller = QuadricOcclusion3D(
            self,
            surfaces=(cylinder,),
            curves=curves,
            paint_policy="diagrammatic",
            style=_style(fill_color="#277D6A"),
            max_chord_error=0.008,
        ).attach()
        self.wait(0.5)
        self.play(tilt.animate.set_value(1.0), run_time=4.0, rate_func=smooth)
        self.play(tilt.animate.set_value(0.25), run_time=2.0, rate_func=smooth)
        self.wait(0.5)
        controller.restore()


class ConeSectionFamiliesDemo(Scene):
    """Three finite cones show ellipse, parabola, and hyperbola support types."""

    def construct(self) -> None:
        _title(
            self,
            "Finite cone section families",
            "ellipse / parabola / hyperbola are solved analytically, then clipped",
        )
        centers = (-4.25, 0.0, 4.25)
        screen_right = QUADRIC_VIEW.matrix[0]
        surfaces = tuple(
            ConeSpec(
                f"cone-{index}",
                tuple(
                    float(value)
                    for value in (
                        center * screen_right + np.asarray((0.0, 0.0, -1.45))
                    )
                ),
                (0.0, 0.0, 1.0),
                pi / 6.0,
                (0.0, 2.9),
                radial_axis=(1.0, 0.0, 0.0),
            )
            for index, center in enumerate(centers)
        )
        normals = (
            (0.30, 0.0, 1.0),
            (sqrt(3.0) / 2.0, 0.0, 0.5),
            (sqrt(1.0 - 0.2**2), 0.0, 0.2),
        )
        labels = ("ELLIPSE", "PARABOLA", "HYPERBOLA")
        curves = []
        for label, surface, normal in zip(labels, surfaces, normals):
            apex = np.asarray(surface.apex, dtype=float)
            plane = SectionPlane(
                f"plane-{label.lower()}",
                tuple(float(value) for value in apex + (0.0, 0.0, 1.25)),
                normal,
                u_axis=(0.0, 1.0, 0.0),
            )
            trace = compute_quadric_section(
                f"section-{label.lower()}", surface, plane
            )
            curves.extend(section_trace_curves(trace))

        controller = QuadricOcclusion3D(
            self,
            surfaces=surfaces,
            curves=tuple(curves),
            paint_policy="diagrammatic",
            style=_style(fill_color="#7756A8"),
            max_chord_error=0.01,
        ).attach()
        for center, label in zip(centers, labels):
            self.add(
                Text(label, font_size=17, color="#4A335F").move_to(
                    (center + 0.10, -3.05, 0.0)
                )
            )
        self.wait(2.5)
        controller.restore()


class ConeSectionTopologyTransitionDemo(Scene):
    """One rotating plane automatically hands off ellipse/parabola/hyperbola."""

    def construct(self) -> None:
        _title(
            self,
            "Automatic conic-family handoff",
            "ellipse → exact parabola → hyperbola, with continuous occlusion",
        )
        vertical_shift = -0.90
        cone = ConeSpec(
            "transition-cone",
            (0.0, 0.0, -1.5 + vertical_shift),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        motion = AxisAnglePlaneMotion(
            "transition-plane-motion",
            SectionPlane(
                "transition-plane",
                (0.0, 0.0, 0.2 + vertical_shift),
                (0.0, 0.0, 1.0),
                u_axis=(1.0, 0.0, 0.0),
            ),
            (0.0, 0.0, 0.2 + vertical_shift),
            (0.0, 1.0, 0.0),
            0.72,
            1.35,
        )
        scheduled = track_scheduled_plane_section(
            "transition-section", cone, motion
        )
        progress = ValueTracker(0.0)

        labels = tuple(
            Text(name, font_size=20, color="#6B4F1D").move_to((x, -3.25, 0.0))
            for name, x in (("ELLIPSE", -2.7), ("PARABOLA", 0.0), ("HYPERBOLA", 2.9))
        )
        self.add(*labels)

        controller = QuadricSection3D(
            self,
            scheduled=scheduled,
            progress=progress,
            projection=TRANSITION_VIEW,
            transition_fraction=0.055,
            paint_policy="diagrammatic",
            style=_style(fill_color="#5275A8", fill_opacity=0.76),
            max_chord_error=0.008,
        ).attach()

        family_names = ("oval", "parabola", "hyperbola")
        for label, family in zip(labels, family_names):
            def update_label(value: Text, dt: float, family_name: str = family) -> None:
                del dt
                weight = sum(
                    layer.opacity
                    for layer, signature in zip(
                        controller.transition_frame.layers,
                        controller.active_signatures,
                    )
                    if signature.conic_family.value == family_name
                )
                value.set_opacity(0.28 + 0.72 * weight)

            label.add_updater(update_label)

        self.wait(0.5)
        self.play(progress.animate.set_value(1.0), run_time=6.0, rate_func=smooth)
        self.wait(0.7)
        for label in labels:
            label.clear_updaters()
        controller.restore()


class GlobalQuadricOcclusionDemo(Scene):
    """Disjoint solids and semantic lines share one certified painter graph."""

    def construct(self) -> None:
        _title(
            self,
            "Global quadric painter graph",
            "two disjoint solids overlap on screen; curves cross and pass behind them",
        )
        surfaces = (
            SphereSpec("far-sphere", (0.768, 0.432, -2.4), 1.15),
            CylinderSpec(
                "near-cylinder",
                (-0.768, -0.432, 2.4),
                (0.0, 1.0, 0.0),
                0.88,
                (-1.35, 1.35),
                radial_axis=(1.0, 0.0, 0.0),
            ),
        )
        curves = (
            SegmentCurve("far-horizontal", (-3.0, 0.54, -3.0), (3.0, 0.54, -3.0)),
            SegmentCurve("near-vertical", (-0.96, -2.54, 3.0), (-0.96, 1.46, 3.0)),
        )
        controller = QuadricOcclusion3D(
            self,
            surfaces=surfaces,
            curves=curves,
            projection=GLOBAL_VIEW,
            paint_policy="diagrammatic",
            style=_style(fill_color="#3E79A8"),
            max_chord_error=0.008,
        ).attach()
        self.wait(2.5)
        controller.restore()
