"""Three ordinary-Manim demos for convex sections and free-line intersections."""

from __future__ import annotations

from math import cos, radians, sin

import numpy as np
from manim import (
    BLUE_E,
    GOLD_D,
    Line,
    Polygon,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    linear,
)

from polyhedron_visibility import OcclusionStyle, ParallelProjection
from polyhedron_visibility.sections import (
    ConvexSectionScene3D,
    ConvexSectionStyle,
    SectionPlane3D,
)


_CUBE = {
    "A": (-1.0, -1.0, -1.0),
    "B": (1.0, -1.0, -1.0),
    "C": (1.0, 1.0, -1.0),
    "D": (-1.0, 1.0, -1.0),
    "E": (-1.0, -1.0, 1.0),
    "F": (1.0, -1.0, 1.0),
    "G": (1.0, 1.0, 1.0),
    "H": (-1.0, 1.0, 1.0),
}

_FACES = {
    "back": ("A", "D", "C", "B"),
    "front": ("E", "F", "G", "H"),
    "bottom": ("A", "B", "F", "E"),
    "right": ("B", "C", "G", "F"),
    "top": ("D", "H", "G", "C"),
    "left": ("A", "E", "H", "D"),
}


def _rotation() -> np.ndarray:
    # Keep the diagonal section plane visibly face-on while the cube still
    # reads as a three-dimensional solid in an ordinary 2D Cairo Scene.
    x_angle = radians(-15)
    y_angle = radians(-35)
    z_angle = radians(8)
    rotate_x = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, cos(x_angle), -sin(x_angle)),
            (0.0, sin(x_angle), cos(x_angle)),
        )
    )
    rotate_y = np.asarray(
        (
            (cos(y_angle), 0.0, sin(y_angle)),
            (0.0, 1.0, 0.0),
            (-sin(y_angle), 0.0, cos(y_angle)),
        )
    )
    rotate_z = np.asarray(
        (
            (cos(z_angle), -sin(z_angle), 0.0),
            (sin(z_angle), cos(z_angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    return rotate_z @ rotate_x @ rotate_y


def _surface_edges() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                tuple(sorted((start, cycle[(index + 1) % len(cycle)])))
                for cycle in _FACES.values()
                for index, start in enumerate(cycle)
            }
        )
    )


class _ConvexSectionFixture:
    def __init__(
        self,
        scene: Scene,
        *,
        plane_offset: float,
        line_height: float,
        include_probe: bool,
        plane_occludes: bool = True,
        show_plane: bool = True,
        show_section_points: bool = True,
        accurate_transparency: bool = False,
    ) -> None:
        self.scene = scene
        self.rotation = _rotation()
        self.plane_offset = ValueTracker(plane_offset)
        self.line_height = ValueTracker(line_height)
        self.include_probe = include_probe

        def transformed(point: tuple[float, float, float]) -> np.ndarray:
            return self.rotation @ np.asarray(point, dtype=float)

        providers = {
            vertex_id: (lambda point=point: transformed(point))
            for vertex_id, point in _CUBE.items()
        }
        if include_probe:
            providers.update(
                {
                    "X": lambda: transformed(
                        (-2.3, self.line_height.get_value(), 0.0)
                    ),
                    "Y": lambda: transformed(
                        (2.3, self.line_height.get_value(), 0.0)
                    ),
                }
            )

        face_polygons: dict[str, Polygon] = {}
        for index, (face_id, cycle) in enumerate(_FACES.items()):
            face = Polygon(
                *(providers[vertex_id]() for vertex_id in cycle),
                color=BLUE_E,
                fill_color=BLUE_E,
                fill_opacity=0.12,
                stroke_opacity=0.0,
            )
            face.set_z_index(2.0 + index if accurate_transparency else 0.0)
            face_polygons[face_id] = face
        face_fill = VGroup(*face_polygons.values())

        edge_lines: dict[str, Line] = {}
        for index, (start, end) in enumerate(_surface_edges()):
            edge_id = f"edge.{start}.{end}"
            edge_lines[edge_id] = Line(
                providers[start](),
                providers[end](),
                buff=0,
                color="#263238",
                stroke_width=3.8,
            ).set_z_index(10 + index)

        if include_probe:
            probe = Line(
                providers["X"](),
                providers["Y"](),
                buff=0,
                color="#D1495B",
                stroke_width=5.0,
            ).set_z_index(30)
            probe.add_updater(
                lambda line: line.put_start_and_end_on(
                    providers["X"](), providers["Y"]()
                )
            )
            edge_lines["probe.X.Y"] = probe

        self.geometry = VGroup(face_fill, *edge_lines.values())
        self.geometry.shift((-0.35, -0.10, 0.0))
        # Providers must include the same final Scene-space placement as the
        # registered source lines.
        shift = np.asarray((-0.35, -0.10, 0.0))
        providers = {
            key: (lambda provider=provider: provider() + shift)
            for key, provider in providers.items()
        }
        scene.add(self.geometry)

        visibility = ConvexSectionScene3D("convex-section-demo")
        for vertex_id, provider in providers.items():
            visibility.vertex(vertex_id, provider)
        for face_id, cycle in _FACES.items():
            visibility.face(
                face_id,
                cycle,
                source_mobject=(
                    face_polygons[face_id] if accurate_transparency else None
                ),
            )
        face_sets = {
            face_id: set(cycle) for face_id, cycle in _FACES.items()
        }
        for start, end in _surface_edges():
            edge_id = f"edge.{start}.{end}"
            visibility.stroke(
                edge_id,
                start,
                end,
                edge_lines[edge_id],
                incident_face_ids=tuple(
                    sorted(
                        face_id
                        for face_id, vertices in face_sets.items()
                        if {start, end}.issubset(vertices)
                    )
                ),
            )
        if include_probe:
            visibility.stroke(
                "probe.X.Y",
                "X",
                "Y",
                edge_lines["probe.X.Y"],
                incident_face_ids=(),
            )

        def current_plane() -> SectionPlane3D:
            offset = self.plane_offset.get_value()
            point = self.rotation @ np.asarray((offset / 3.0,) * 3) + shift
            normal = self.rotation @ np.asarray((1.0, 1.0, 1.0))
            u_axis = self.rotation @ np.asarray((1.0, -1.0, 0.0))
            return SectionPlane3D(
                "moving-plane",
                tuple(float(value) for value in point),
                tuple(float(value) for value in normal),
                # Minimum display dimensions only. The default auto mode
                # expands this infinite mathematical plane around the solid.
                0.10,
                0.10,
                u_axis=tuple(float(value) for value in u_axis),
                occludes_strokes=plane_occludes,
            )

        visibility.cutting_plane("moving-section", current_plane)
        self.visibility = visibility
        self.controller = visibility.controller(
            scene,
            projection=ParallelProjection.identity(),
            source_style=OcclusionStyle(
                max_projected_length=7.0,
                dash_length=0.13,
                dash_gap=0.09,
                hidden_opacity_scale=0.62,
            ),
            section_style=ConvexSectionStyle(
                plane_fill_color="#52B6A8",
                plane_fill_opacity=0.11,
                plane_stroke_color="#2A9D8F",
                section_fill_color=GOLD_D,
                section_fill_opacity=0.48,
                boundary_color="#B86B00",
                boundary_hidden_color="#8B5E00",
                point_color="#B91C1C",
                intersection_point_color="#7C3AED",
                max_boundary_projected_length=6.0,
                dash_length=0.13,
                dash_gap=0.09,
                show_plane=show_plane,
                show_points=show_section_points,
                show_intersection_points=True,
            ),
            accurate_transparency=accurate_transparency,
        )


def _title(scene: Scene, text: str) -> None:
    label = Text(text, font_size=30, color="#263238")
    label.to_edge(np.array((0.0, 1.0, 0.0)), buff=0.35)
    label.set_z_index(100)
    scene.add(label)


class LineThroughCubeDemo(Scene):
    """A free semantic line enters, cuts through, and leaves a cube."""

    def construct(self) -> None:
        self.camera.background_color = "#F7F5EF"
        _title(self, "独立直线 × 凸多面体：自动求交与遮挡")
        fixture = _ConvexSectionFixture(
            self,
            plane_offset=5.0,
            line_height=1.55,
            include_probe=True,
            plane_occludes=False,
            show_plane=False,
            show_section_points=False,
        )
        with fixture.controller.session():
            self.wait(0.4)
            self.play(
                fixture.line_height.animate.set_value(0.0),
                run_time=2.2,
                rate_func=linear,
            )
            self.wait(0.5)
            self.play(
                fixture.line_height.animate.set_value(-1.55),
                run_time=2.2,
                rate_func=linear,
            )
            self.wait(0.4)


class MovingPlaneSectionDemo(Scene):
    """An infinite plane crosses a cube and changes section topology."""

    def construct(self) -> None:
        self.camera.background_color = "#F7F5EF"
        _title(self, "平面 × 凸多面体：截面从空集到六边形")
        fixture = _ConvexSectionFixture(
            self,
            plane_offset=4.0,
            line_height=0.0,
            include_probe=False,
        )
        with fixture.controller.session():
            self.wait(0.4)
            self.play(
                fixture.plane_offset.animate.set_value(0.0),
                run_time=2.8,
                rate_func=linear,
            )
            self.wait(0.6)
            self.play(
                fixture.plane_offset.animate.set_value(-4.0),
                run_time=2.8,
                rate_func=linear,
            )
            self.wait(0.4)


class CombinedSectionAndLineDemo(Scene):
    """Solid faces, a cutting plane, a section boundary, and a free line."""

    def construct(self) -> None:
        self.camera.background_color = "#F7F5EF"
        _title(self, "实体面 + 截平面 + 独立直线：统一全局遮挡")
        fixture = _ConvexSectionFixture(
            self,
            plane_offset=3.2,
            line_height=0.85,
            include_probe=True,
        )
        with fixture.controller.session():
            self.wait(0.4)
            self.play(
                fixture.plane_offset.animate.set_value(0.0),
                fixture.line_height.animate.set_value(0.0),
                run_time=2.8,
                rate_func=linear,
            )
            self.wait(0.7)
            self.play(
                fixture.plane_offset.animate.set_value(-2.2),
                fixture.line_height.animate.set_value(-0.65),
                run_time=2.2,
                rate_func=linear,
            )
            self.wait(0.5)


class AccurateTransparentSectionDemo(Scene):
    """The solid and plane are split wherever their local depth order changes."""

    def construct(self) -> None:
        self.camera.background_color = "#F7F5EF"
        _title(self, "相交半透明面：截面分离与局部准确排序")
        fixture = _ConvexSectionFixture(
            self,
            plane_offset=3.2,
            line_height=0.75,
            include_probe=True,
            accurate_transparency=True,
        )
        with fixture.controller.session():
            self.wait(0.4)
            self.play(
                fixture.plane_offset.animate.set_value(0.0),
                fixture.line_height.animate.set_value(0.0),
                run_time=2.8,
                rate_func=linear,
            )
            self.wait(0.7)
            self.play(
                fixture.plane_offset.animate.set_value(-3.2),
                fixture.line_height.animate.set_value(-0.75),
                run_time=2.8,
                rate_func=linear,
            )
            self.wait(0.5)


__all__ = [
    "AccurateTransparentSectionDemo",
    "CombinedSectionAndLineDemo",
    "LineThroughCubeDemo",
    "MovingPlaneSectionDemo",
]
