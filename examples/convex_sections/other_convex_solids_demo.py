"""Convex-section demos for four closed solids with different topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from manim import GOLD_D, Line, Polygon, Scene, Text, ValueTracker, VGroup, linear

from polyhedron_visibility import OcclusionStyle, ParallelProjection
from polyhedron_visibility.sections import (
    ConvexSectionScene3D,
    ConvexSectionStyle,
    SectionPlane3D,
)


Point3 = tuple[float, float, float]
FaceCycle = tuple[str, ...]


@dataclass(frozen=True)
class ConvexSolidSpec:
    solid_id: str
    title: str
    subtitle: str
    vertices: Mapping[str, Point3]
    faces: Mapping[str, FaceCycle]
    face_color: str
    expected_center_section_vertices: int


CONVEX_SOLIDS: dict[str, ConvexSolidSpec] = {
    "tetrahedron": ConvexSolidSpec(
        solid_id="tetrahedron",
        title="四面体：三角截面与独立直线",
        subtitle="4 个面 · 6 条棱 · 中心斜截面为三角形",
        vertices={
            "A": (1.0, 1.0, 1.0),
            "B": (1.0, -1.0, -1.0),
            "C": (-1.0, 1.0, -1.0),
            "D": (-1.0, -1.0, 1.0),
        },
        faces={
            "ABC": ("A", "B", "C"),
            "ADB": ("A", "D", "B"),
            "ACD": ("A", "C", "D"),
            "BDC": ("B", "D", "C"),
        },
        face_color="#5B8FF9",
        expected_center_section_vertices=3,
    ),
    "triangular-prism": ConvexSolidSpec(
        solid_id="triangular-prism",
        title="三棱柱：四边形截面与独立直线",
        subtitle="5 个面 · 9 条棱 · 截面会在三边形与四边形间变化",
        vertices={
            "A": (-1.1, -0.8, -1.0),
            "B": (1.1, -0.8, -1.0),
            "C": (0.0, 1.1, -1.0),
            "D": (-1.1, -0.8, 1.0),
            "E": (1.1, -0.8, 1.0),
            "F": (0.0, 1.1, 1.0),
        },
        faces={
            "bottom": ("A", "C", "B"),
            "top": ("D", "E", "F"),
            "side-ab": ("A", "B", "E", "D"),
            "side-bc": ("B", "C", "F", "E"),
            "side-ca": ("C", "A", "D", "F"),
        },
        face_color="#61DDAA",
        expected_center_section_vertices=4,
    ),
    "square-pyramid": ConvexSolidSpec(
        solid_id="square-pyramid",
        title="四棱锥：五边形截面与独立直线",
        subtitle="5 个面 · 8 条棱 · 斜截面会经历三角形与五边形",
        vertices={
            "A": (-1.0, -1.0, -1.0),
            "B": (1.0, -1.0, -1.0),
            "C": (1.0, 1.0, -1.0),
            "D": (-1.0, 1.0, -1.0),
            "E": (0.0, 0.0, 1.4),
        },
        faces={
            "base": ("A", "D", "C", "B"),
            "front": ("A", "B", "E"),
            "right": ("B", "C", "E"),
            "back": ("C", "D", "E"),
            "left": ("D", "A", "E"),
        },
        face_color="#65789B",
        expected_center_section_vertices=5,
    ),
    "octahedron": ConvexSolidSpec(
        solid_id="octahedron",
        title="正八面体：六边形截面与独立直线",
        subtitle="8 个面 · 12 条棱 · 多个面同时参与全局遮挡",
        vertices={
            "R": (1.25, 0.0, 0.0),
            "L": (-1.25, 0.0, 0.0),
            "U": (0.0, 1.25, 0.0),
            "D": (0.0, -1.25, 0.0),
            "F": (0.0, 0.0, 1.25),
            "B": (0.0, 0.0, -1.25),
        },
        faces={
            "FRU": ("F", "R", "U"),
            "FDR": ("F", "D", "R"),
            "FLD": ("F", "L", "D"),
            "FUL": ("F", "U", "L"),
            "BUR": ("B", "U", "R"),
            "BRD": ("B", "R", "D"),
            "BDL": ("B", "D", "L"),
            "BLU": ("B", "L", "U"),
        },
        face_color="#8B7EC8",
        expected_center_section_vertices=6,
    ),
}


SECTION_NORMAL = np.asarray((1.0, 0.7, 0.45), dtype=float)
SECTION_NORMAL /= np.linalg.norm(SECTION_NORMAL)


def surface_edges(faces: Mapping[str, FaceCycle]) -> tuple[tuple[str, str], ...]:
    """Return every topological edge exactly once in deterministic order."""

    return tuple(
        sorted(
            {
                tuple(sorted((start, cycle[(index + 1) % len(cycle)])))
                for cycle in faces.values()
                for index, start in enumerate(cycle)
            }
        )
    )


def incident_faces(
    faces: Mapping[str, FaceCycle], start: str, end: str
) -> tuple[str, ...]:
    return tuple(
        sorted(
            face_id
            for face_id, cycle in faces.items()
            if {start, end}.issubset(cycle)
        )
    )


def visibility_payload(
    spec: ConvexSolidSpec, *, include_probe: bool = True
) -> dict[str, object]:
    """Build the exact closed-solid contract exercised by the rendered demo."""

    vertices: dict[str, Point3] = dict(spec.vertices)
    if include_probe:
        vertices.update({"X": (-2.4, 0.0, 0.0), "Y": (2.4, 0.0, 0.0)})
    strokes = [
        {
            "sourceEdgeId": f"edge.{start}.{end}",
            "vertexIds": [start, end],
            "incidentFaceIds": list(incident_faces(spec.faces, start, end)),
        }
        for start, end in surface_edges(spec.faces)
    ]
    if include_probe:
        strokes.append(
            {
                "sourceEdgeId": "probe.X.Y",
                "vertexIds": ["X", "Y"],
                "incidentFaceIds": [],
            }
        )
    return {
        "schema": "manim-convex-polyhedron-visibility/v1",
        "visibilityGroupId": f"other-solid:{spec.solid_id}",
        "vertices": [
            {"vertexId": vertex_id, "entryPosition": list(point)}
            for vertex_id, point in vertices.items()
        ],
        "faces": [
            {"faceId": face_id, "vertexIds": list(cycle)}
            for face_id, cycle in spec.faces.items()
        ],
        "strokes": strokes,
    }


def _rotation() -> np.ndarray:
    """Use a true isometric view so several facets remain simultaneously visible."""

    view = np.asarray((1.0, 1.0, 1.0), dtype=float)
    view /= np.linalg.norm(view)
    screen_right = np.cross(np.asarray((0.0, 0.0, 1.0)), view)
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(view, screen_right)
    screen_up /= np.linalg.norm(screen_up)
    return np.asarray((screen_right, screen_up, view), dtype=float)


class _OtherSolidFixture:
    def __init__(self, scene: Scene, spec: ConvexSolidSpec) -> None:
        self.scene = scene
        self.spec = spec
        self.rotation = _rotation()
        self.scale = 1.30
        self.shift = np.asarray((0.0, -0.25, 0.0), dtype=float)
        support = max(
            abs(float(np.dot(SECTION_NORMAL, np.asarray(point, dtype=float))))
            for point in spec.vertices.values()
        )
        self.plane_extent = support + 0.28
        self.plane_offset = ValueTracker(self.plane_extent)
        self.line_height = ValueTracker(1.55)

        def display(point: Point3 | np.ndarray) -> np.ndarray:
            return self.rotation @ (self.scale * np.asarray(point, dtype=float)) + self.shift

        providers = {
            vertex_id: (lambda point=point: display(point))
            for vertex_id, point in spec.vertices.items()
        }
        providers.update(
            {
                "X": lambda: display((-2.4, self.line_height.get_value(), 0.0)),
                "Y": lambda: display((2.4, self.line_height.get_value(), 0.0)),
            }
        )

        face_polygons: dict[str, Polygon] = {}
        for index, (face_id, cycle) in enumerate(spec.faces.items()):
            face_polygons[face_id] = Polygon(
                *(providers[vertex_id]() for vertex_id in cycle),
                color=spec.face_color,
                fill_color=spec.face_color,
                fill_opacity=0.14,
                stroke_opacity=0.0,
            ).set_z_index(index)
        face_fill = VGroup(*face_polygons.values())
        self.face_polygons = face_polygons

        edge_lines: dict[str, Line] = {}
        for index, (start, end) in enumerate(surface_edges(spec.faces)):
            edge_id = f"edge.{start}.{end}"
            edge_lines[edge_id] = Line(
                providers[start](),
                providers[end](),
                buff=0,
                color="#263238",
                stroke_width=3.8,
            ).set_z_index(20 + index)

        probe = Line(
            providers["X"](),
            providers["Y"](),
            buff=0,
            color="#D1495B",
            stroke_width=5.0,
        ).set_z_index(60)
        probe.add_updater(
            lambda line: line.put_start_and_end_on(providers["X"](), providers["Y"]())
        )
        edge_lines["probe.X.Y"] = probe
        scene.add(VGroup(face_fill, *edge_lines.values()))

        visibility = ConvexSectionScene3D(f"other-solid:{spec.solid_id}")
        for vertex_id, provider in providers.items():
            visibility.vertex(vertex_id, provider)
        for face_id, cycle in spec.faces.items():
            visibility.face(
                face_id,
                cycle,
                source_mobject=face_polygons[face_id],
            )
        for start, end in surface_edges(spec.faces):
            edge_id = f"edge.{start}.{end}"
            visibility.stroke(
                edge_id,
                start,
                end,
                edge_lines[edge_id],
                incident_face_ids=incident_faces(spec.faces, start, end),
            )
        visibility.stroke(
            "probe.X.Y", "X", "Y", probe, incident_face_ids=()
        )

        section_u = np.cross(SECTION_NORMAL, np.asarray((0.0, 0.0, 1.0)))
        section_u /= np.linalg.norm(section_u)

        def current_plane() -> SectionPlane3D:
            point = display(SECTION_NORMAL * self.plane_offset.get_value())
            normal = self.rotation @ SECTION_NORMAL
            u_axis = self.rotation @ section_u
            return SectionPlane3D(
                "moving-plane",
                tuple(float(value) for value in point),
                tuple(float(value) for value in normal),
                # Minimum display dimensions; auto mode expands as needed.
                0.10,
                0.10,
                u_axis=tuple(float(value) for value in u_axis),
                occludes_strokes=True,
            )

        visibility.cutting_plane("moving-section", current_plane)
        self.visibility = visibility
        self.controller = visibility.controller(
            scene,
            projection=ParallelProjection.identity(),
            source_style=OcclusionStyle(
                max_projected_length=8.0,
                dash_length=0.13,
                dash_gap=0.09,
                hidden_opacity_scale=0.62,
            ),
            section_style=ConvexSectionStyle(
                plane_fill_color="#52B6A8",
                plane_fill_opacity=0.11,
                plane_stroke_color="#2A9D8F",
                section_fill_color=GOLD_D,
                section_fill_opacity=0.52,
                boundary_color="#B86B00",
                boundary_hidden_color="#8B5E00",
                point_color="#B91C1C",
                intersection_point_color="#7C3AED",
                max_boundary_projected_length=7.0,
                dash_length=0.13,
                dash_gap=0.09,
                show_plane=True,
                show_points=True,
                show_intersection_points=True,
            ),
        )


def _labels(scene: Scene, spec: ConvexSolidSpec) -> None:
    title = Text(spec.title, font_size=30, color="#263238")
    title.to_edge(np.asarray((0.0, 1.0, 0.0)), buff=0.30)
    title.set_z_index(100)
    subtitle = Text(spec.subtitle, font_size=19, color="#607D8B")
    subtitle.next_to(title, np.asarray((0.0, -1.0, 0.0)), buff=0.10)
    subtitle.set_z_index(100)
    legend = Text(
        "面色阶：朝向·远近·冷暖    金色：自动截面    虚线：被遮挡部分",
        font_size=18,
        color="#455A64",
    )
    legend.to_edge(np.asarray((0.0, -1.0, 0.0)), buff=0.28)
    legend.set_z_index(100)
    scene.add(title, subtitle, legend)


def _run(scene: Scene, spec: ConvexSolidSpec) -> None:
    scene.camera.background_color = "#F7F5EF"
    _labels(scene, spec)
    fixture = _OtherSolidFixture(scene, spec)
    with fixture.controller.session():
        scene.wait(0.4)
        scene.play(
            fixture.plane_offset.animate.set_value(0.0),
            fixture.line_height.animate.set_value(0.0),
            run_time=2.8,
            rate_func=linear,
        )
        scene.wait(0.7)
        scene.play(
            fixture.plane_offset.animate.set_value(-fixture.plane_extent),
            fixture.line_height.animate.set_value(-1.55),
            run_time=2.8,
            rate_func=linear,
        )
        scene.wait(0.5)


class TetrahedronSectionDemo(Scene):
    def construct(self) -> None:
        _run(self, CONVEX_SOLIDS["tetrahedron"])


class TriangularPrismSectionDemo(Scene):
    def construct(self) -> None:
        _run(self, CONVEX_SOLIDS["triangular-prism"])


class SquarePyramidSectionDemo(Scene):
    def construct(self) -> None:
        _run(self, CONVEX_SOLIDS["square-pyramid"])


class OctahedronSectionDemo(Scene):
    def construct(self) -> None:
        _run(self, CONVEX_SOLIDS["octahedron"])


__all__ = [
    "CONVEX_SOLIDS",
    "ConvexSolidSpec",
    "OctahedronSectionDemo",
    "SquarePyramidSectionDemo",
    "TetrahedronSectionDemo",
    "TriangularPrismSectionDemo",
    "incident_faces",
    "surface_edges",
    "visibility_payload",
]
