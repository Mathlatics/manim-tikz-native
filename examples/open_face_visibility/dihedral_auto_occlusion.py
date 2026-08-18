"""Ordinary-Manim demo for articulated open-face automatic occlusion."""

from __future__ import annotations

from math import cos, pi, sin

import numpy as np
from manim import BLUE_D, DEGREES, GREEN_D, Line, Polygon, Scene, ValueTracker, VGroup

from polyhedron_visibility import OcclusionStyle, ParallelProjection
from polyhedron_visibility.open_faces import OpenFaceScene3D


class DihedralAutoOcclusionDemo(Scene):
    def construct(self) -> None:
        self.camera.background_color = "#F6F8FC"
        fold = ValueTracker(20 * DEGREES)
        alpha_points = {
            "A": np.array((-2.0, 0.0, 0.0)),
            "B": np.array((2.0, 0.0, 0.0)),
            "C": np.array((2.0, 1.6, 0.0)),
            "D": np.array((-2.0, 1.6, 0.0)),
        }

        def beta_far(x: float) -> np.ndarray:
            angle = fold.get_value()
            return np.array((x, -1.6 * cos(angle), -1.6 * sin(angle)))

        providers = {
            **{name: (lambda point=point: point.copy()) for name, point in alpha_points.items()},
            "E": lambda: beta_far(2.0),
            "F": lambda: beta_far(-2.0),
            "P": lambda: np.array((-2.6, -0.35, -0.9)),
            "Q": lambda: np.array((2.6, 0.75, -0.9)),
        }

        alpha = Polygon(
            *(providers[name]() for name in ("A", "B", "C", "D")),
            color=BLUE_D,
            fill_color=BLUE_D,
            fill_opacity=0.18,
            stroke_opacity=0,
        ).set_z_index(1)
        beta = Polygon(
            *(providers[name]() for name in ("B", "A", "F", "E")),
            color=GREEN_D,
            fill_color=GREEN_D,
            fill_opacity=0.18,
            stroke_opacity=0,
        ).set_z_index(2)
        beta.add_updater(
            lambda polygon: polygon.become(
                Polygon(
                    *(providers[name]() for name in ("B", "A", "F", "E")),
                    color=GREEN_D,
                    fill_color=GREEN_D,
                    fill_opacity=0.18,
                    stroke_opacity=0,
                ).set_z_index(2)
            )
        )

        edge_pairs = (
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "A"),
            ("A", "F"),
            ("F", "E"),
            ("E", "B"),
            ("P", "Q"),
        )
        source_lines: dict[str, Line] = {}
        for index, (start, end) in enumerate(edge_pairs):
            edge_id = f"{start}{end}"
            line = Line(
                providers[start](),
                providers[end](),
                buff=0,
                color="#20242A",
                stroke_width=4,
            ).set_z_index(10 + index)
            line.add_updater(
                lambda item, start=start, end=end: item.put_start_and_end_on(
                    providers[start](), providers[end]()
                )
            )
            source_lines[edge_id] = line

        geometry = VGroup(alpha, beta, *source_lines.values())
        self.add(geometry)

        visibility = OpenFaceScene3D("dihedral-demo")
        for vertex_id, provider in providers.items():
            visibility.vertex(vertex_id, provider)
        visibility.face(
            "alpha",
            ("A", "B", "C", "D"),
            logical_surface_id="surface-alpha",
            source_mobject=alpha,
        )
        visibility.face(
            "beta",
            ("B", "A", "F", "E"),
            logical_surface_id="surface-beta",
            source_mobject=beta,
        )
        visibility.articulated_hinge("hinge-AB", "alpha", "beta", "A", "B")
        for start, end in edge_pairs:
            edge_id = f"{start}{end}"
            visibility.stroke(edge_id, start, end, source_lines[edge_id])

        controller = visibility.controller(
            self,
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(
                max_projected_length=7.0,
                dash_length=0.12,
                dash_gap=0.08,
                hidden_opacity_scale=0.60,
            ),
        )
        with controller.session():
            self.wait(0.3)
            self.play(fold.animate.set_value(112 * DEGREES), run_time=2.0)
            self.play(fold.animate.set_value(pi), run_time=1.2)
            self.wait(0.3)


__all__ = ["DihedralAutoOcclusionDemo"]
