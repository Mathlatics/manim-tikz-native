from __future__ import annotations

import json
import unittest

import numpy as np

from polyhedron_visibility import (
    ParallelView,
    SolverError,
    TolerancePolicy,
    VisibilityModel,
    canonical_trace_json,
    compute_frame_visibility,
    segment_face_occlusion_interval,
)


IDENTITY_VIEW = np.eye(3)


def face_model(*, reverse_inputs: bool = False, scale: float = 1.0) -> VisibilityModel:
    vertices = {
        "L": (-2 * scale, 0, 0),
        "R": (2 * scale, 0, 0),
        "A0": (-1.5 * scale, -1 * scale, 1 * scale),
        "A1": (-0.5 * scale, -1 * scale, 1 * scale),
        "A2": (-0.5 * scale, 1 * scale, 1 * scale),
        "A3": (-1.5 * scale, 1 * scale, 1 * scale),
        "B0": (0.5 * scale, -1 * scale, 1 * scale),
        "B1": (1.5 * scale, -1 * scale, 1 * scale),
        "B2": (1.5 * scale, 1 * scale, 1 * scale),
        "B3": (0.5 * scale, 1 * scale, 1 * scale),
    }
    vertex_items = [
        {"vertexId": name, "entryPosition": value}
        for name, value in vertices.items()
    ]
    faces = [
        {"faceId": "left-face", "vertexIds": ["A0", "A1", "A2", "A3"]},
        {"faceId": "right-face", "vertexIds": ["B0", "B1", "B2", "B3"]},
    ]
    if reverse_inputs:
        vertex_items.reverse()
        faces.reverse()
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "two-faces",
            "vertices": vertex_items,
            "faces": faces,
            "strokes": [
                {
                    "sourceEdgeId": "probe",
                    "vertexIds": ["L", "R"],
                    "incidentFaceIds": [],
                }
            ],
        }
    )


class ParallelVisibilitySolverTests(unittest.TestCase):
    def test_parallel_view_and_one_arbitrary_convex_face_interval(self) -> None:
        view = ParallelView.from_matrix(IDENTITY_VIEW)
        self.assertTrue(np.allclose(view.view_direction, (0, 0, 1)))
        interval = segment_face_occlusion_interval(
            (-2, 0, 0),
            (2, 0, 0),
            [(-1, -1, 1), (0.8, -1.2, 1), (1.4, 0, 1), (0.3, 1.3, 1), (-1, 1, 1)],
            view,
        )
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertLess(interval[0], 0.3)
        self.assertGreater(interval[1], 0.7)

    def test_two_disjoint_faces_create_five_ordered_spans(self) -> None:
        frame = compute_frame_visibility(face_model(), projection_matrix=IDENTITY_VIEW)
        edge = frame.edge_map["probe"]

        self.assertEqual(
            [(round(item.start, 3), round(item.end, 3), item.kind) for item in edge.spans],
            [
                (0.0, 0.125, "visible"),
                (0.125, 0.375, "hidden"),
                (0.375, 0.625, "visible"),
                (0.625, 0.875, "hidden"),
                (0.875, 1.0, "visible"),
            ],
        )
        self.assertEqual(edge.spans[1].occluder_face_ids, ("left-face",))
        self.assertEqual(edge.spans[3].occluder_face_ids, ("right-face",))

    def test_overlapping_faces_preserve_occlusion_level_and_provenance(self) -> None:
        model = face_model().to_dict()
        vertices = {item["vertexId"]: item for item in model["vertices"]}
        for name in ("B0", "B3"):
            point = list(vertices[name]["entryPosition"])
            point[0] = -0.75
            vertices[name]["entryPosition"] = point
        for name in ("B1", "B2"):
            point = list(vertices[name]["entryPosition"])
            point[0] = 0.75
            vertices[name]["entryPosition"] = point
        frame = compute_frame_visibility(
            VisibilityModel.from_dict(model),
            projection_matrix=IDENTITY_VIEW,
        )
        hidden = [item for item in frame.edge_map["probe"].spans if item.kind == "hidden"]

        self.assertEqual([item.level for item in hidden], [1, 2, 1])
        self.assertEqual(hidden[1].occluder_face_ids, ("left-face", "right-face"))

    def test_incident_face_coplanar_face_and_edge_on_face_are_skipped(self) -> None:
        payload = face_model().to_dict()
        payload["strokes"].append({
            "sourceEdgeId": "left-edge",
            "vertexIds": ["A0", "A1"],
            "incidentFaceIds": ["left-face"],
        })
        model = VisibilityModel.from_dict(payload)
        frame = compute_frame_visibility(model, projection_matrix=IDENTITY_VIEW)
        edge = frame.edge_map["left-edge"]
        self.assertIn(
            ("left-face", "incident_face"),
            [(item.face_id, item.reason) for item in edge.skipped_faces],
        )

        coplanar_positions = {
            key: np.asarray(value, dtype=float)
            for key, value in model.entry_positions.items()
        }
        coplanar_positions["L"][2] = 1.0
        coplanar_positions["R"][2] = 1.0
        coplanar = compute_frame_visibility(
            model,
            vertex_positions=coplanar_positions,
            projection_matrix=IDENTITY_VIEW,
        )
        self.assertFalse(any(item.kind == "hidden" for item in coplanar.edge_map["probe"].spans))

        edge_on = compute_frame_visibility(
            model,
            projection_matrix=((1, 0, 0), (0, 0, 1), (0, 1, 0)),
        )
        self.assertTrue(any(
            item.reason == "face_edge_on" for item in edge_on.edge_map["probe"].skipped_faces
        ))

    def test_scale_and_input_order_do_not_change_normalized_trace(self) -> None:
        normalized = []
        for scale in (1e-6, 1.0, 1e6):
            frame = compute_frame_visibility(
                face_model(scale=scale),
                projection_matrix=IDENTITY_VIEW,
                tolerance_policy=TolerancePolicy(),
            )
            normalized.append([
                (round(item.start, 6), round(item.end, 6), item.kind)
                for item in frame.edge_map["probe"].spans
            ])
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[1], normalized[2])

        first = compute_frame_visibility(face_model(), projection_matrix=IDENTITY_VIEW)
        second = compute_frame_visibility(
            face_model(reverse_inputs=True), projection_matrix=IDENTITY_VIEW
        )
        self.assertEqual(canonical_trace_json(first), canonical_trace_json(second))
        json.loads(canonical_trace_json(first))

    def test_singular_projection_fails_closed(self) -> None:
        with self.assertRaisesRegex(SolverError, "projection"):
            compute_frame_visibility(
                face_model(),
                projection_matrix=((1, 0, 0), (2, 0, 0), (0, 0, 1)),
            )


if __name__ == "__main__":
    unittest.main()
