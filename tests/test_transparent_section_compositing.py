from __future__ import annotations

import unittest

import numpy as np

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.sections.compositing import (
    TransparentSectionCompositingError,
    canonical_transparent_section_compositing_json,
    compute_transparent_section_compositing,
)
from polyhedron_visibility.sections.contract import SectionPlane3D


_VERTICES = {
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

_ISOMETRIC = np.asarray(
    (
        (0.7071067811865476, -0.7071067811865476, 0.0),
        (0.4082482904638631, 0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, -0.5773502691896258),
    )
)


def _cube_model(scale: float = 1.0) -> VisibilityModel:
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "transparent-cube",
            "vertices": [
                {
                    "vertexId": vertex_id,
                    "entryPosition": [scale * value for value in point],
                }
                for vertex_id, point in _VERTICES.items()
            ],
            "faces": [
                {"faceId": face_id, "vertexIds": list(vertex_ids)}
                for face_id, vertex_ids in _FACES.items()
            ],
            "strokes": [],
        }
    )


def _plane(
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
    *,
    extent: float = 3.0,
) -> SectionPlane3D:
    return SectionPlane3D(
        "cut",
        point,
        normal,
        extent,
        extent,
        u_axis=(1.0, -1.0, 0.0),
    )


def _triangle_area(points: tuple[tuple[float, float, float], ...]) -> float:
    values = np.asarray(points, dtype=float)
    return 0.5 * float(
        np.linalg.norm(np.cross(values[1] - values[0], values[2] - values[0]))
    )


def _screen_depth_at(
    triangle: tuple[tuple[float, float, float], ...],
    screen_point: np.ndarray,
) -> float | None:
    projected = np.asarray(triangle, dtype=float) @ _ISOMETRIC.T
    screen = projected[:, :2]
    first = screen[1] - screen[0]
    second = screen[2] - screen[0]
    denominator = first[0] * second[1] - first[1] * second[0]
    if abs(float(denominator)) <= 1.0e-12:
        return None
    relative = screen_point - screen[0]
    first_weight = (
        relative[0] * second[1] - relative[1] * second[0]
    ) / denominator
    second_weight = (
        first[0] * relative[1] - first[1] * relative[0]
    ) / denominator
    weights = np.asarray(
        (1.0 - first_weight - second_weight, first_weight, second_weight)
    )
    if float(np.min(weights)) <= 2.0e-4:
        return None
    return float(weights @ projected[:, 2])


class TransparentSectionCompositingTests(unittest.TestCase):
    def test_diagonal_cube_partition_preserves_every_surface_area(self) -> None:
        model = _cube_model()
        cutting_plane = _plane((0, 0, 0), (1, 1, 1))
        frame = compute_transparent_section_compositing(
            "section",
            model,
            cutting_plane,
            projection_matrix=_ISOMETRIC,
        )
        self.assertEqual(frame.section.kind, "polygon")
        self.assertEqual(len(frame.section.points), 6)
        self.assertEqual(set(frame.draw_order), set(frame.fragment_map))

        plane_area = sum(
            _triangle_area(item.vertices)
            for item in frame.fragments
            if item.role in {"plane_outside", "section_inside"}
        )
        self.assertAlmostEqual(
            plane_area,
            4.0 * cutting_plane.half_width * cutting_plane.half_height,
            places=9,
        )
        section_area = sum(
            _triangle_area(item.vertices)
            for item in frame.fragments
            if item.role == "section_inside"
        )
        self.assertGreater(section_area, 0.0)
        self.assertLess(section_area, plane_area)

        for face_id in _FACES:
            restored_area = sum(
                _triangle_area(item.vertices)
                for item in frame.fragments
                if item.source_face_id == face_id
            )
            self.assertAlmostEqual(restored_area, 4.0, places=9)

    def test_intersection_creates_both_local_plane_face_orders(self) -> None:
        frame = compute_transparent_section_compositing(
            "section",
            _cube_model(),
            _plane((0, 0, 0), (1, 1, 1)),
            projection_matrix=_ISOMETRIC,
        )
        fragments = frame.fragment_map
        local_orders = {
            (
                fragments[item.far_fragment_id].role,
                fragments[item.near_fragment_id].role,
            )
            for item in frame.order_relations
            if (
                fragments[item.far_fragment_id].role.startswith("solid_face")
                or fragments[item.near_fragment_id].role.startswith("solid_face")
            )
            and (
                fragments[item.far_fragment_id].role
                in {"plane_outside", "section_inside"}
                or fragments[item.near_fragment_id].role
                in {"plane_outside", "section_inside"}
            )
        }
        self.assertTrue(
            any(first.startswith("solid_face") for first, _second in local_orders)
        )
        self.assertTrue(
            any(second.startswith("solid_face") for _first, second in local_orders)
        )

    def test_draw_order_matches_a_dense_per_pixel_depth_oracle(self) -> None:
        frame = compute_transparent_section_compositing(
            "section",
            _cube_model(),
            _plane((0, 0, 0), (1, 1, 1)),
            projection_matrix=_ISOMETRIC,
        )
        projected = np.concatenate(
            [
                np.asarray(item.vertices, dtype=float) @ _ISOMETRIC.T
                for item in frame.fragments
            ],
            axis=0,
        )[:, :2]
        minimum = projected.min(axis=0)
        maximum = projected.max(axis=0)
        order_index = {
            fragment_id: index
            for index, fragment_id in enumerate(frame.draw_order)
        }
        checked_pairs = 0
        for x_value in np.linspace(minimum[0], maximum[0], 43):
            for y_value in np.linspace(minimum[1], maximum[1], 43):
                point = np.asarray((x_value, y_value))
                covered = [
                    (item.fragment_id, depth)
                    for item in frame.fragments
                    if (
                        depth := _screen_depth_at(item.vertices, point)
                    )
                    is not None
                ]
                for first_index, (first_id, first_depth) in enumerate(covered):
                    for second_id, second_depth in covered[first_index + 1 :]:
                        difference = first_depth - second_depth
                        if abs(difference) <= 1.0e-7:
                            continue
                        checked_pairs += 1
                        if difference < 0:
                            self.assertLess(
                                order_index[first_id], order_index[second_id]
                            )
                        else:
                            self.assertLess(
                                order_index[second_id], order_index[first_id]
                            )
        self.assertGreater(checked_pairs, 500)

    def test_empty_point_triangle_hexagon_and_coplanar_face_are_supported(self) -> None:
        model = _cube_model()
        for offset, kind in ((4.0, "empty"), (3.0, "point"), (2.0, "polygon"), (0.0, "polygon")):
            with self.subTest(offset=offset):
                point = (offset / 3.0,) * 3
                frame = compute_transparent_section_compositing(
                    "section",
                    model,
                    _plane(point, (1, 1, 1)),
                    projection_matrix=_ISOMETRIC,
                )
                self.assertEqual(frame.section.kind, kind)
                self.assertEqual(set(frame.draw_order), set(frame.fragment_map))

        segment = compute_transparent_section_compositing(
            "section",
            model,
            _plane((1, 1, 0), (1, 1, 0)),
            projection_matrix=_ISOMETRIC,
        )
        self.assertEqual(segment.section.kind, "segment")
        self.assertEqual(len(segment.section.points), 2)

        coplanar = compute_transparent_section_compositing(
            "section",
            model,
            _plane((0, 0, 1), (0, 0, 1)),
            projection_matrix=_ISOMETRIC,
        )
        self.assertTrue(
            any(item.reason == "coplanar_policy" for item in coplanar.order_relations)
        )
        solid_over = compute_transparent_section_compositing(
            "section",
            model,
            _plane((0, 0, 1), (0, 0, 1)),
            projection_matrix=_ISOMETRIC,
            coplanar_policy="solid_over_section",
        )
        default_pairs = {
            (item.far_fragment_id, item.near_fragment_id)
            for item in coplanar.order_relations
            if item.reason == "coplanar_policy"
        }
        reversed_pairs = {
            (item.near_fragment_id, item.far_fragment_id)
            for item in solid_over.order_relations
            if item.reason == "coplanar_policy"
        }
        self.assertEqual(default_pairs, reversed_pairs)
        with self.assertRaisesRegex(
            TransparentSectionCompositingError, "coplanarly"
        ):
            compute_transparent_section_compositing(
                "section",
                model,
                _plane((0, 0, 1), (0, 0, 1)),
                projection_matrix=_ISOMETRIC,
                coplanar_policy="fail",
            )

    def test_patch_must_cover_section_with_positive_margin(self) -> None:
        with self.assertRaisesRegex(
            TransparentSectionCompositingError, "positive margin"
        ):
            compute_transparent_section_compositing(
                "section",
                _cube_model(),
                _plane((0, 0, 0), (0, 0, 1), extent=1.0),
                projection_matrix=_ISOMETRIC,
            )

    def test_trace_is_deterministic_under_input_reordering(self) -> None:
        first_model = _cube_model()
        payload = first_model.to_dict()
        payload["vertices"].reverse()
        payload["faces"].reverse()
        second_model = VisibilityModel.from_dict(payload)
        cutting_plane = _plane((0, 0, 0), (1, 1, 1))
        first = compute_transparent_section_compositing(
            "section",
            first_model,
            cutting_plane,
            projection_matrix=_ISOMETRIC,
        )
        second = compute_transparent_section_compositing(
            "section",
            second_model,
            cutting_plane,
            projection_matrix=_ISOMETRIC,
        )
        self.assertEqual(
            canonical_transparent_section_compositing_json(first),
            canonical_transparent_section_compositing_json(second),
        )

    def test_uniform_scaling_keeps_roles_and_order_relations(self) -> None:
        signatures = []
        for scale in (1.0e-6, 1.0, 1.0e6):
            frame = compute_transparent_section_compositing(
                "section",
                _cube_model(scale),
                _plane((0, 0, 0), (1, 1, 1), extent=3.0 * scale),
                projection_matrix=_ISOMETRIC,
            )
            roles = tuple(sorted(item.role for item in frame.fragments))
            relations = tuple(
                sorted(
                    (
                        frame.fragment_map[item.far_fragment_id].role,
                        frame.fragment_map[item.near_fragment_id].role,
                        item.reason,
                    )
                    for item in frame.order_relations
                )
            )
            signatures.append((roles, relations))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])

    def test_near_vertex_section_keeps_authoritative_polygon(self) -> None:
        normal = (
            -0.3694768645267391,
            -0.4542421017891346,
            0.8106484808729855,
        )
        offset = -0.725841132084116
        axis = (
            0.9292399295012509,
            -0.18061207033490848,
            0.322323491960869,
        )
        frame = compute_transparent_section_compositing(
            "section",
            _cube_model(),
            SectionPlane3D(
                "cut",
                tuple(offset * item for item in normal),
                normal,
                3.0,
                3.0,
                u_axis=axis,
            ),
            projection_matrix=_ISOMETRIC,
        )
        self.assertEqual(frame.section.kind, "polygon")
        self.assertEqual(len(frame.section.points), 5)
        self.assertEqual(set(frame.draw_order), set(frame.fragment_map))


if __name__ == "__main__":
    unittest.main()
