"""Minimal ordinary-Manim demo for automatic convex-polyhedron hidden lines."""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE_E,
    DEGREES,
    Line,
    Polygon,
    RIGHT,
    Rotate,
    Scene,
    VGroup,
    VectorizedPoint,
)

from polyhedron_visibility import (
    OcclusionScene3D,
    OcclusionStyle,
    ParallelProjection,
)


class CubeAutoOcclusionDemo(Scene):
    def construct(self) -> None:
        coordinates = {
            "A": (-1.0, -1.0, -1.0),
            "B": (1.0, -1.0, -1.0),
            "C": (1.0, 1.0, -1.0),
            "D": (-1.0, 1.0, -1.0),
            "E": (-1.0, -1.0, 1.0),
            "F": (1.0, -1.0, 1.0),
            "G": (1.0, 1.0, 1.0),
            "H": (-1.0, 1.0, 1.0),
        }
        anchors = {
            name: VectorizedPoint(point) for name, point in coordinates.items()
        }
        face_cycles = {
            "back": ("A", "D", "C", "B"),
            "front": ("E", "F", "G", "H"),
            "bottom": ("A", "B", "F", "E"),
            "right": ("B", "C", "G", "F"),
            "top": ("D", "H", "G", "C"),
            "left": ("A", "E", "H", "D"),
        }
        face_mobjects = VGroup(*(
            Polygon(
                *(coordinates[name] for name in cycle),
                color=BLUE_E,
                fill_color=BLUE_E,
                fill_opacity=0.12,
                stroke_opacity=0,
            )
            for cycle in face_cycles.values()
        ))
        edge_pairs = sorted({
            tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
            for cycle in face_cycles.values()
            for index in range(len(cycle))
        })
        edges = {}
        for z_offset, pair in enumerate(edge_pairs, start=1):
            edge_id = "".join(pair)
            edges[edge_id] = Line(
                coordinates[pair[0]],
                coordinates[pair[1]],
                buff=0,
                stroke_width=3,
                z_index=10 + z_offset,
            )
        geometry = VGroup(*anchors.values(), face_mobjects, *edges.values())
        geometry.rotate(24 * DEGREES, axis=np.array((0.35, 1.0, 0.20)))
        geometry.shift(RIGHT * 0.25)
        self.add(geometry)

        visibility = OcclusionScene3D("cube")
        for vertex_id, anchor in anchors.items():
            visibility.vertex(vertex_id, anchor.get_center)
        for face_id, cycle in face_cycles.items():
            visibility.face(face_id, cycle)
        for start, end in edge_pairs:
            edge_id = start + end
            visibility.stroke(edge_id, start, end, edges[edge_id])

        controller = visibility.controller(
            self,
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(
                max_projected_length=4.0,
                dash_length=0.10,
                dash_gap=0.07,
                hidden_opacity_scale=0.60,
            ),
        )
        with controller.session():
            self.wait(0.4)
            self.play(
                Rotate(geometry, angle=55 * DEGREES, axis=np.array((0.3, 1.0, 0.2))),
                run_time=2.0,
            )
            self.wait(0.4)


__all__ = ["CubeAutoOcclusionDemo"]
