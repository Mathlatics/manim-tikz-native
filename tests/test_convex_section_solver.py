from __future__ import annotations

import unittest

import numpy as np

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.sections import (
    ConvexSectionContractError,
    SectionPlane3D,
    canonical_section_trace_json,
    compute_sectioned_visibility,
    fit_plane_patch_to_convex_polyhedron,
    intersect_plane_with_convex_polyhedron,
    intersect_segment_with_convex_polyhedron,
)


def cube_model() -> VisibilityModel:
    vertices = {
        "A": (-1.0, -1.0, -1.0),
        "B": (1.0, -1.0, -1.0),
        "C": (1.0, 1.0, -1.0),
        "D": (-1.0, 1.0, -1.0),
        "E": (-1.0, -1.0, 1.0),
        "F": (1.0, -1.0, 1.0),
        "G": (1.0, 1.0, 1.0),
        "H": (-1.0, 1.0, 1.0),
    }
    faces = {
        "back": ("A", "D", "C", "B"),
        "front": ("E", "F", "G", "H"),
        "bottom": ("A", "B", "F", "E"),
        "right": ("B", "C", "G", "F"),
        "top": ("D", "H", "G", "C"),
        "left": ("A", "E", "H", "D"),
    }
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "cube",
            "vertices": [
                {"vertexId": key, "entryPosition": value}
                for key, value in vertices.items()
            ],
            "faces": [
                {"faceId": key, "vertexIds": list(value)}
                for key, value in faces.items()
            ],
            "strokes": [],
        }
    )


def plane(
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> SectionPlane3D:
    return SectionPlane3D(
        "cut",
        point,
        normal,
        3.0,
        3.0,
        u_axis=(1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0),
    )


class ConvexSectionSolverTests(unittest.TestCase):
    def test_plane_contract_is_canonical_and_strict(self) -> None:
        original = plane((0, 0, 0), (0, 0, 4))
        restored = SectionPlane3D.from_dict(original.to_dict())
        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertAlmostEqual(np.linalg.norm(restored.normal), 1.0)
        self.assertAlmostEqual(float(np.dot(restored.normal, restored.u_axis)), 0.0)
        with self.assertRaisesRegex(ConvexSectionContractError, "normal"):
            SectionPlane3D("cut", (0, 0, 0), (0, 0, 0), 1, 1)

    def test_segment_crossing_cube_reports_two_exact_boundary_hits(self) -> None:
        result = intersect_segment_with_convex_polyhedron(
            cube_model(), (-2, 0, 0), (2, 0, 0)
        )
        self.assertEqual(result.kind, "segment")
        self.assertEqual(result.inside_parameter_interval, (0.25, 0.75))
        self.assertFalse(result.starts_inside)
        self.assertFalse(result.ends_inside)
        self.assertEqual([item.role for item in result.hits], ["entry", "exit"])
        np.testing.assert_allclose(result.hits[0].position, (-1, 0, 0))
        np.testing.assert_allclose(result.hits[1].position, (1, 0, 0))
        self.assertEqual(result.hits[0].face_ids, ("left",))
        self.assertEqual(result.hits[1].face_ids, ("right",))

    def test_segment_inside_miss_and_vertex_touch_are_distinct(self) -> None:
        inside = intersect_segment_with_convex_polyhedron(
            cube_model(), (-0.5, 0, 0), (0.5, 0, 0)
        )
        self.assertEqual(inside.inside_parameter_interval, (0.0, 1.0))
        self.assertTrue(inside.starts_inside)
        self.assertTrue(inside.ends_inside)
        self.assertEqual(inside.hits, ())

        miss = intersect_segment_with_convex_polyhedron(
            cube_model(), (-2, 2, 0), (2, 2, 0)
        )
        self.assertEqual(miss.kind, "none")

        touch = intersect_segment_with_convex_polyhedron(
            cube_model(), (0, 2, 0), (2, 0, 2)
        )
        self.assertEqual(touch.kind, "point")
        self.assertEqual(len(touch.hits), 1)
        np.testing.assert_allclose(touch.hits[0].position, (1, 1, 1))

    def test_axis_plane_makes_square_and_diagonal_plane_makes_hexagon(self) -> None:
        square = intersect_plane_with_convex_polyhedron(
            "square", cube_model(), plane((0, 0, 0), (0, 0, 1))
        )
        self.assertEqual(square.kind, "polygon")
        self.assertEqual(len(square.points), 4)
        self.assertEqual(len(square.boundary_segments), 4)
        self.assertTrue(all(abs(item.position[2]) < 1.0e-12 for item in square.points))

        hexagon = intersect_plane_with_convex_polyhedron(
            "hexagon", cube_model(), plane((0, 0, 0), (1, 1, 1))
        )
        self.assertEqual(hexagon.kind, "polygon")
        self.assertEqual(len(hexagon.points), 6)
        self.assertEqual(len(hexagon.boundary_segments), 6)
        self.assertTrue(
            all(abs(sum(item.position)) < 1.0e-12 for item in hexagon.points)
        )

    def test_infinite_plane_display_patch_auto_fits_the_complete_solid(self) -> None:
        authored = SectionPlane3D(
            "cut",
            (0, 0, 0),
            (0, 0, 1),
            0.05,
            0.10,
            u_axis=(1, 0, 0),
        )
        fitted = fit_plane_patch_to_convex_polyhedron(
            cube_model(), authored, margin_ratio=0.15
        )
        self.assertAlmostEqual(fitted.half_width, 1.15)
        self.assertAlmostEqual(fitted.half_height, 1.15)
        self.assertEqual(fitted.point, authored.point)
        self.assertEqual(fitted.normal, authored.normal)
        section = intersect_plane_with_convex_polyhedron(
            "square", cube_model(), fitted
        )
        self.assertTrue(
            all(
                abs(fitted.coordinates_in_plane(item.position)[0])
                < fitted.half_width
                and abs(fitted.coordinates_in_plane(item.position)[1])
                < fitted.half_height
                for item in section.points
            )
        )

    def test_auto_fit_treats_authored_dimensions_as_minimums(self) -> None:
        authored = SectionPlane3D(
            "cut",
            (0, 0, 0),
            (0, 0, 1),
            4.0,
            5.0,
            u_axis=(1, 0, 0),
        )
        fitted = fit_plane_patch_to_convex_polyhedron(cube_model(), authored)
        self.assertEqual(fitted.half_width, 4.0)
        self.assertEqual(fitted.half_height, 5.0)

    def test_translation_covers_empty_point_triangle_hexagon_and_face(self) -> None:
        model = cube_model()
        normal = (1.0, 1.0, 1.0)
        samples = (
            (4.0, "empty", 0),
            (3.0, "point", 1),
            (2.0, "polygon", 3),
            (0.0, "polygon", 6),
        )
        for offset, kind, count in samples:
            with self.subTest(offset=offset):
                point = (offset / 3.0,) * 3
                frame = intersect_plane_with_convex_polyhedron(
                    f"cut-{offset}", model, plane(point, normal)
                )
                self.assertEqual(frame.kind, kind)
                self.assertEqual(len(frame.points), count)

        face = intersect_plane_with_convex_polyhedron(
            "front", model, plane((0, 0, 1), (0, 0, 1))
        )
        self.assertEqual(face.kind, "polygon")
        self.assertEqual(len(face.points), 4)
        self.assertEqual(
            {vertex for item in face.points for vertex in item.source_vertex_ids},
            {"E", "F", "G", "H"},
        )

    def test_trace_is_deterministic_when_input_arrays_are_reordered(self) -> None:
        first_model = cube_model()
        payload = first_model.to_dict()
        payload["vertices"].reverse()
        payload["faces"].reverse()
        second_model = VisibilityModel.from_dict(payload)
        section_plane = plane((0, 0, 0), (1, 1, 1))
        first = intersect_plane_with_convex_polyhedron(
            "stable", first_model, section_plane
        )
        second = intersect_plane_with_convex_polyhedron(
            "stable", second_model, section_plane
        )
        self.assertEqual(
            canonical_section_trace_json(first), canonical_section_trace_json(second)
        )

    def test_independent_plane_joins_global_stroke_occlusion(self) -> None:
        payload = cube_model().to_dict()
        payload["vertices"].extend(
            (
                {"vertexId": "X", "entryPosition": [-2.0, 1.5, -2.0]},
                {"vertexId": "Y", "entryPosition": [2.0, 1.5, -2.0]},
            )
        )
        payload["strokes"].append(
            {
                "sourceEdgeId": "probe",
                "vertexIds": ["X", "Y"],
                "incidentFaceIds": [],
            }
        )
        model = VisibilityModel.from_dict(payload)
        cutting_plane = SectionPlane3D(
            "cut",
            (0, 0, 0),
            (0, 0, 1),
            half_width=1.0,
            half_height=2.0,
            u_axis=(1, 0, 0),
        )
        frame = compute_sectioned_visibility(
            model,
            cutting_plane,
            projection_matrix=np.eye(3),
        )
        edge = frame.edge_map["probe"]
        plane_intervals = [
            item for item in edge.raw_intervals if item.face_id == "section-plane:cut"
        ]
        self.assertEqual(len(plane_intervals), 1)
        self.assertAlmostEqual(plane_intervals[0].start, 0.25)
        self.assertAlmostEqual(plane_intervals[0].end, 0.75)
        self.assertEqual(
            [(item.kind, round(item.start, 2), round(item.end, 2)) for item in edge.spans],
            [("visible", 0.0, 0.25), ("hidden", 0.25, 0.75), ("visible", 0.75, 1.0)],
        )


if __name__ == "__main__":
    unittest.main()
