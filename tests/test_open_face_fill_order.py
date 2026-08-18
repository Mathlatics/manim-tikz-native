from __future__ import annotations

import copy
import unittest

from polyhedron_visibility.open_faces import (
    OPEN_FACE_MODEL_SCHEMA,
    OPEN_FACE_TOPOLOGY,
    OpenFaceSolverError,
    OpenFaceVisibilityModel,
    canonical_open_face_trace_json,
    compute_open_face_visibility,
)


IDENTITY_VIEW = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _model(
    faces: list[tuple[str, tuple[tuple[float, float, float], ...]]],
) -> OpenFaceVisibilityModel:
    vertices: list[dict[str, object]] = []
    face_payload: list[dict[str, object]] = []
    for face_id, points in faces:
        vertex_ids: list[str] = []
        for index, point in enumerate(points):
            vertex_id = f"{face_id}.v{index}"
            vertex_ids.append(vertex_id)
            vertices.append(
                {"vertexId": vertex_id, "entryPosition": list(point)}
            )
        face_payload.append(
            {
                "faceId": face_id,
                "logicalSurfaceId": f"surface-{face_id}",
                "vertexIds": vertex_ids,
                "occludesStrokes": True,
            }
        )
    return OpenFaceVisibilityModel.from_dict(
        {
            "schema": OPEN_FACE_MODEL_SCHEMA,
            "topology": OPEN_FACE_TOPOLOGY,
            "visibilityGroupId": "fill-order-fixture",
            "vertices": vertices,
            "faces": face_payload,
            "seams": [],
            "strokes": [],
        }
    )


def _square(z: float, *, low: float = -1.0, high: float = 1.0):
    return (
        (low, low, z),
        (high, low, z),
        (high, high, z),
        (low, high, z),
    )


class OpenFaceFillOrderTests(unittest.TestCase):
    def test_depth_is_compared_only_inside_the_projected_overlap(self) -> None:
        # The large tilted panel has a much smaller centroid depth, but inside
        # the only region where the two fills overlap it is entirely nearer.
        # Centroid sorting therefore returns the wrong painter order.
        large_near_in_overlap = (
            (-10.0, -1.0, -10.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-10.0, 1.0, -10.0),
        )
        small_far = (
            (0.0, -0.5, -1.0),
            (1.0, -0.5, -1.0),
            (1.0, 0.5, -1.0),
            (0.0, 0.5, -1.0),
        )
        frame = compute_open_face_visibility(
            _model([("large-near", large_near_in_overlap), ("small-far", small_far)]),
            projection_matrix=IDENTITY_VIEW,
        )
        self.assertEqual(
            frame.advisory_face_draw_order,
            ("small-far", "large-near"),
        )

    def test_overlapping_parallel_faces_are_sorted_far_to_near(self) -> None:
        model = _model(
            [
                ("near", _square(2.0)),
                ("middle", _square(0.0)),
                ("far", _square(-3.0)),
            ]
        )
        frame = compute_open_face_visibility(model, projection_matrix=IDENTITY_VIEW)
        self.assertEqual(frame.advisory_face_draw_order, ("far", "middle", "near"))

    def test_non_overlapping_faces_use_deterministic_identity_order(self) -> None:
        left = tuple((x - 5.0, y, 100.0) for x, y, _z in _square(0.0))
        right = tuple((x + 5.0, y, -100.0) for x, y, _z in _square(0.0))
        frame = compute_open_face_visibility(
            _model([("z-face", left), ("a-face", right)]),
            projection_matrix=IDENTITY_VIEW,
        )
        self.assertEqual(frame.advisory_face_draw_order, ("a-face", "z-face"))

    def test_crossing_faces_fail_instead_of_falling_back_to_centroids(self) -> None:
        rising = (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, -1.0),
        )
        with self.assertRaises(OpenFaceSolverError) as caught:
            compute_open_face_visibility(
                _model([("rising", rising), ("flat", _square(0.0))]),
                projection_matrix=IDENTITY_VIEW,
            )
        self.assertEqual(caught.exception.code, "FACE_ORDER_REQUIRES_SPLITTING")

    def test_coplanar_overlap_is_stable_and_input_order_independent(self) -> None:
        payload = _model(
            [("beta", _square(0.0)), ("alpha", _square(0.0))]
        ).to_dict()
        reversed_payload = copy.deepcopy(payload)
        reversed_payload["vertices"].reverse()
        reversed_payload["faces"].reverse()

        first = compute_open_face_visibility(
            OpenFaceVisibilityModel.from_dict(payload),
            projection_matrix=IDENTITY_VIEW,
        )
        second = compute_open_face_visibility(
            OpenFaceVisibilityModel.from_dict(reversed_payload),
            projection_matrix=IDENTITY_VIEW,
        )
        self.assertEqual(first.advisory_face_draw_order, ("alpha", "beta"))
        self.assertEqual(
            canonical_open_face_trace_json(first),
            canonical_open_face_trace_json(second),
        )

    def test_uniform_scale_does_not_change_the_order(self) -> None:
        orders = []
        for scale in (1.0e-9, 1.0, 1.0e9):
            orders.append(
                compute_open_face_visibility(
                    _model(
                        [
                            ("near", _square(2.0 * scale, low=-scale, high=scale)),
                            ("far", _square(-3.0 * scale, low=-scale, high=scale)),
                        ]
                    ),
                    projection_matrix=IDENTITY_VIEW,
                ).advisory_face_draw_order
            )
        self.assertEqual(orders, [("far", "near")] * 3)


if __name__ == "__main__":
    unittest.main()
