from __future__ import annotations

import unittest

from polyhedron_visibility import (
    ContractError,
    VisibilityModel,
)


def cube_payload() -> dict:
    return {
        "schema": "manim-convex-polyhedron-visibility/v1",
        "visibilityGroupId": "cube",
        "vertices": [
            {"vertexId": name, "entryPosition": point}
            for name, point in {
                "A": (-1, -1, -1),
                "B": (1, -1, -1),
                "C": (1, 1, -1),
                "D": (-1, 1, -1),
                "E": (-1, -1, 1),
                "F": (1, -1, 1),
                "G": (1, 1, 1),
                "H": (-1, 1, 1),
            }.items()
        ],
        "faces": [
            {"faceId": "back", "vertexIds": ["A", "D", "C", "B"]},
            {"faceId": "front", "vertexIds": ["E", "F", "G", "H"]},
            {"faceId": "bottom", "vertexIds": ["A", "B", "F", "E"]},
            {"faceId": "right", "vertexIds": ["B", "C", "G", "F"]},
            {"faceId": "top", "vertexIds": ["D", "H", "G", "C"]},
            {"faceId": "left", "vertexIds": ["A", "E", "H", "D"]},
        ],
        "strokes": [
            {
                "sourceEdgeId": "AB",
                "vertexIds": ["A", "B"],
                "incidentFaceIds": ["back", "bottom"],
                "renderBindingId": "line.A.B",
            },
            {
                "sourceEdgeId": "probe",
                "vertexIds": ["A", "G"],
                "incidentFaceIds": [],
                "renderBindingId": "line.probe",
            },
        ],
    }


class VisibilityContractTests(unittest.TestCase):
    def test_round_trip_is_canonical_and_supports_free_semantic_strokes(self) -> None:
        payload = cube_payload()
        payload["vertices"].reverse()
        payload["faces"].reverse()
        payload["strokes"].reverse()

        model = VisibilityModel.from_dict(payload)

        self.assertEqual(model.visibility_group_id, "cube")
        self.assertEqual([item.vertex_id for item in model.vertices], sorted(
            item["vertexId"] for item in payload["vertices"]
        ))
        self.assertEqual(model.stroke_map["probe"].incident_face_ids, ())
        self.assertEqual(
            VisibilityModel.from_dict(model.to_dict()).to_dict(),
            model.to_dict(),
        )

    def test_closed_convex_cube_passes_strict_topology_validation(self) -> None:
        model = VisibilityModel.from_dict(cube_payload())
        model.validate(require_closed_convex_manifold=True)

    def test_open_face_system_is_valid_unless_closed_topology_is_requested(self) -> None:
        payload = cube_payload()
        payload["faces"] = payload["faces"][:2]
        payload["strokes"] = []
        model = VisibilityModel.from_dict(payload)

        model.validate()
        with self.assertRaisesRegex(ContractError, "closed two-manifold"):
            model.validate(require_closed_convex_manifold=True)

    def test_rejects_nonplanar_nonconvex_and_nonfinite_faces(self) -> None:
        nonplanar = cube_payload()
        nonplanar["vertices"][6]["entryPosition"] = (1, 1, 1.2)
        with self.assertRaisesRegex(ContractError, "not planar"):
            VisibilityModel.from_dict(nonplanar).validate()

        nonconvex = {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "concave",
            "vertices": [
                {"vertexId": "A", "entryPosition": (0, 0, 0)},
                {"vertexId": "B", "entryPosition": (2, 0, 0)},
                {"vertexId": "C", "entryPosition": (1, 0.4, 0)},
                {"vertexId": "D", "entryPosition": (2, 2, 0)},
                {"vertexId": "E", "entryPosition": (0, 2, 0)},
            ],
            "faces": [{"faceId": "face", "vertexIds": ["A", "B", "C", "D", "E"]}],
            "strokes": [],
        }
        with self.assertRaisesRegex(ContractError, "not strictly convex"):
            VisibilityModel.from_dict(nonconvex).validate()

        nonfinite = cube_payload()
        nonfinite["vertices"][0]["entryPosition"] = (float("nan"), 0, 0)
        with self.assertRaisesRegex(ContractError, "finite"):
            VisibilityModel.from_dict(nonfinite)

    def test_rejects_invalid_incidence_and_duplicate_identity(self) -> None:
        invalid = cube_payload()
        invalid["strokes"][0]["incidentFaceIds"] = ["front"]
        with self.assertRaisesRegex(ContractError, "does not contain both endpoints"):
            VisibilityModel.from_dict(invalid)

        duplicate = cube_payload()
        duplicate["faces"].append(dict(duplicate["faces"][0]))
        with self.assertRaisesRegex(ContractError, "duplicate faceId"):
            VisibilityModel.from_dict(duplicate)


if __name__ == "__main__":
    unittest.main()
