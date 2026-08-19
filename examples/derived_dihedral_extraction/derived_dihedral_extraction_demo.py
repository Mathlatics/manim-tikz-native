from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (
    BLUE,
    GOLD,
    TEAL,
    Line,
    Polygon,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    smooth,
)

from polyhedron_visibility import OcclusionStyle, ParallelProjection, VisibilityModel
from polyhedron_visibility.dihedral_extraction import (
    ExtractedDihedralScene3D,
    RigidTransform3D,
)


TEX_POINTS_PER_CM = 72.27 / 2.54
DISPLAY_SCENE_UNITS_PER_CM = 1.22
HIDDEN_DASH_PATTERN_PT = (2.0, 2.0)


def hidden_dash_pattern_scene_units() -> tuple[float, float]:
    """Return ``on 2pt off 2pt`` in this demo's display coordinates."""

    return tuple(
        value * DISPLAY_SCENE_UNITS_PER_CM / TEX_POINTS_PER_CM
        for value in HIDDEN_DASH_PATTERN_PT
    )


def _closed_model(
    group_id: str,
    points: dict[str, tuple[float, float, float]],
    raw_faces: dict[str, tuple[str, ...]],
) -> VisibilityModel:
    center = np.mean([points[key] for key in sorted(points)], axis=0)
    faces: dict[str, tuple[str, ...]] = {}
    for face_id, raw_vertex_ids in raw_faces.items():
        vertex_ids = list(raw_vertex_ids)
        vertices = np.asarray([points[item] for item in vertex_ids], dtype=float)
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        if float(np.dot(normal, np.mean(vertices, axis=0) - center)) < 0:
            vertex_ids.reverse()
        faces[face_id] = tuple(vertex_ids)
    edge_faces: dict[tuple[str, str], list[str]] = {}
    for face_id, vertex_ids in faces.items():
        for index, start in enumerate(vertex_ids):
            end = vertex_ids[(index + 1) % len(vertex_ids)]
            edge_faces.setdefault(tuple(sorted((start, end))), []).append(face_id)
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": group_id,
            "vertices": [
                {"vertexId": vertex_id, "entryPosition": points[vertex_id]}
                for vertex_id in sorted(points)
            ],
            "faces": [
                {"faceId": face_id, "vertexIds": list(faces[face_id])}
                for face_id in sorted(faces)
            ],
            "strokes": [
                {
                    "sourceEdgeId": f"edge.{start}.{end}",
                    "vertexIds": [start, end],
                    "incidentFaceIds": sorted(owner_ids),
                }
                for (start, end), owner_ids in sorted(edge_faces.items())
            ],
        }
    )


def rectangular_box() -> VisibilityModel:
    return _closed_model(
        "rectangular-box",
        {
            "A": (-1.6, -1.0, -0.8),
            "B": (1.6, -1.0, -0.8),
            "C": (1.6, 1.0, -0.8),
            "D": (-1.6, 1.0, -0.8),
            "E": (-1.6, -1.0, 0.8),
            "F": (1.6, -1.0, 0.8),
            "G": (1.6, 1.0, 0.8),
            "H": (-1.6, 1.0, 0.8),
        },
        {
            "back": ("A", "B", "C", "D"),
            "front": ("E", "F", "G", "H"),
            "bottom": ("A", "B", "F", "E"),
            "right": ("B", "C", "G", "F"),
            "top": ("D", "C", "G", "H"),
            "left": ("A", "D", "H", "E"),
        },
    )


def tetrahedron() -> VisibilityModel:
    return _closed_model(
        "tetrahedron",
        {
            "A": (1.25, 1.05, 1.0),
            "B": (-1.3, -1.0, 0.95),
            "C": (-1.05, 1.25, -1.0),
            "D": (1.15, -1.15, -0.95),
        },
        {
            "ABC": ("A", "B", "C"),
            "ABD": ("A", "B", "D"),
            "ACD": ("A", "C", "D"),
            "BCD": ("B", "C", "D"),
        },
    )


def square_pyramid() -> VisibilityModel:
    return _closed_model(
        "square-pyramid",
        {
            "A": (-1.25, -1.05, -0.75),
            "B": (1.25, -1.05, -0.75),
            "C": (1.25, 1.05, -0.75),
            "D": (-1.25, 1.05, -0.75),
            "S": (0.0, 0.0, 1.65),
        },
        {
            "base": ("A", "B", "C", "D"),
            "side.AB": ("A", "B", "S"),
            "side.BC": ("B", "C", "S"),
            "side.CD": ("C", "D", "S"),
            "side.DA": ("D", "A", "S"),
        },
    )


def isometric_projection() -> np.ndarray:
    view = np.asarray((1.0, 1.2, 1.0), dtype=float)
    view /= np.linalg.norm(view)
    right = np.cross(np.asarray((0.0, 0.0, 1.0)), view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    return np.asarray((right, up, view), dtype=float)


@dataclass(frozen=True)
class _DemoSpec:
    title: str
    model: VisibilityModel
    source_face_ids: tuple[str, str]
    base_face_id: str
    translation: tuple[float, float, float]
    rotate_to_base: bool = True
    return_to_entry: bool = False


class _DerivedDihedralDemo(Scene):
    SPEC: _DemoSpec

    def construct(self) -> None:
        spec = self.SPEC
        model = spec.model
        world = {
            key: np.asarray(vertex.entry_position, dtype=float)
            for key, vertex in model.vertex_map.items()
        }
        projection_matrix = isometric_projection()
        screen_offset = np.asarray((0.0, -0.2, 0.0), dtype=float)

        def display(point) -> np.ndarray:
            return 1.22 * (projection_matrix @ np.asarray(point, dtype=float)) + screen_offset

        face_colors = ("#8CC8C0", "#7FB7D4", "#A8D5BA", "#90B7D9", "#9EC9B8", "#7FA9C4")
        faces: dict[str, Polygon] = {}
        for index, face in enumerate(model.faces):
            polygon = Polygon(
                *(display(world[item]) for item in face.vertex_ids),
                fill_color=face_colors[index % len(face_colors)],
                fill_opacity=0.23,
                stroke_opacity=0.0,
            ).set_z_index(index)
            faces[face.face_id] = polygon
        strokes: dict[str, Line] = {}
        for index, stroke in enumerate(model.strokes):
            line = Line(
                display(world[stroke.vertex_ids[0]]),
                display(world[stroke.vertex_ids[1]]),
                buff=0,
                color="#24445F",
                stroke_width=4.0,
            ).set_z_index(20 + index)
            strokes[stroke.source_edge_id] = line

        title = Text(spec.title, font_size=32, color=TEAL).to_edge(np.asarray((0, 1, 0)))
        title.set_z_index(100)
        self.add(
            VGroup(*(faces[key] for key in sorted(faces))),
            VGroup(*(strokes[key] for key in sorted(strokes))),
            title,
        )

        builder = ExtractedDihedralScene3D(model.visibility_group_id)
        for vertex_id in sorted(world):
            builder.vertex(vertex_id, lambda vertex_id=vertex_id: world[vertex_id])
        for face in model.faces:
            builder.face(
                face.face_id,
                face.vertex_ids,
                source_mobject=faces[face.face_id],
            )
        for stroke in model.strokes:
            builder.stroke(
                stroke.source_edge_id,
                stroke.vertex_ids[0],
                stroke.vertex_ids[1],
                strokes[stroke.source_edge_id],
                incident_face_ids=stroke.incident_face_ids,
            )

        progress = ValueTracker(0.0)
        base_progress = ValueTracker(0.0)
        translation = np.asarray(spec.translation, dtype=float)
        entity = builder.extract_dihedral(
            "analysis-copy",
            spec.source_face_ids,
            transform_provider=lambda: RigidTransform3D.translation_by(
                translation * progress.get_value()
            ),
            edge_color=GOLD,
            face_color="#F4B942",
            face_opacity=0.42,
        )
        base_rotation = (
            builder.base_plane_rotation(spec.base_face_id)
            if spec.rotate_to_base
            else None
        )
        dash_length, dash_gap = hidden_dash_pattern_scene_units()

        def solid_transform() -> RigidTransform3D:
            # The copy moves by +T/2 and the source solid by -T/2.  The
            # controller applies the same center-relative base rotation before
            # those placements, so each entity rotates about its own translated
            # geometric center while their orientation remains synchronized.
            solid_shift = RigidTransform3D.translation_by(
                -0.5 * translation * progress.get_value()
            )
            if base_rotation is None:
                return solid_shift
            return solid_shift.compose(base_rotation.transform(base_progress.get_value()))

        self.add(entity.mobject)
        controller = builder.controller(
            self,
            projection=ParallelProjection(projection_matrix),
            display_point_provider=display,
            source_coordinate_mode="display",
            style=OcclusionStyle(
                max_projected_length=10.0,
                dash_length=dash_length,
                dash_gap=dash_gap,
                # Keep hidden edges identifiable by their authored color, but
                # darken them enough that a translucent foreground face still
                # reads as the occluder after Cairo alpha compositing.
                hidden_opacity_scale=0.48,
            ),
            accurate_transparency=True,
            global_transform_provider=solid_transform,
        )

        with controller.session():
            self.wait(0.6)
            self.play(
                progress.animate.set_value(1.0),
                run_time=2.8,
                rate_func=smooth,
            )
            self.wait(0.8)
            if spec.rotate_to_base:
                self.play(
                    base_progress.animate.set_value(1.0),
                    run_time=2.4,
                    rate_func=smooth,
                )
                self.wait(0.8)
            self.play(
                progress.animate.set_value(0.55),
                run_time=1.2,
                rate_func=smooth,
            )
            self.wait(0.7)
            if spec.return_to_entry:
                self.play(
                    base_progress.animate.set_value(0.0),
                    run_time=1.8,
                    rate_func=smooth,
                )
                self.play(
                    progress.animate.set_value(0.0),
                    run_time=2.0,
                    rate_func=smooth,
                )
                self.wait(0.8)


class RectangularBoxDihedralDemo(_DerivedDihedralDemo):
    SPEC = _DemoSpec(
        "Rectangular box: extract one dihedral",
        rectangular_box(),
        ("front", "top"),
        "right",
        (2.6, -1.0, 1.0),
    )


class TetrahedronDihedralDemo(_DerivedDihedralDemo):
    SPEC = _DemoSpec(
        "Tetrahedron: pure extraction handoff",
        tetrahedron(),
        ("ABC", "ABD"),
        "ACD",
        (2.25, -0.9, 0.8),
        rotate_to_base=False,
    )


class SquarePyramidDihedralDemo(_DerivedDihedralDemo):
    SPEC = _DemoSpec(
        "Square pyramid: extract one dihedral",
        square_pyramid(),
        ("side.AB", "side.BC"),
        "side.CD",
        (2.5, -0.8, 0.75),
    )


class RectangularBoxDihedralRoundTripDemo(_DerivedDihedralDemo):
    SPEC = _DemoSpec(
        "Rectangular box: extract and return",
        rectangular_box(),
        ("front", "top"),
        "right",
        (2.6, -1.0, 1.0),
        return_to_entry=True,
    )
