from __future__ import annotations

import math
import unittest

from polyhedron_visibility import (
    ContractError,
    FaceSpec,
    StrokeSpec,
    VertexSpec,
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

    def test_closed_convex_validation_ignores_free_stroke_only_vertices(self) -> None:
        payload = cube_payload()
        payload["vertices"].extend([
            {"vertexId": "X", "entryPosition": (-4, 0, 0)},
            {"vertexId": "Y", "entryPosition": (4, 0, 0)},
        ])
        payload["strokes"].append({
            "sourceEdgeId": "outside-probe",
            "vertexIds": ["X", "Y"],
            "incidentFaceIds": [],
        })

        VisibilityModel.from_dict(payload).validate(require_closed_convex_manifold=True)

    def test_open_face_system_is_valid_unless_closed_topology_is_requested(self) -> None:
        payload = cube_payload()
        payload["faces"] = payload["faces"][:2]
        payload["strokes"] = []
        model = VisibilityModel.from_dict(payload)

        model.validate()
        with self.assertRaisesRegex(ContractError, "closed two-manifold"):
            model.validate(require_closed_convex_manifold=True)

    def test_coplanar_adjacent_triangulation_must_be_merged_into_one_face(self) -> None:
        payload = {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "triangulated-square",
            "vertices": [
                {"vertexId": "A", "entryPosition": (-1, -1, 1)},
                {"vertexId": "B", "entryPosition": (1, -1, 1)},
                {"vertexId": "C", "entryPosition": (1, 1, 1)},
                {"vertexId": "D", "entryPosition": (-1, 1, 1)},
            ],
            "faces": [
                {"faceId": "ABC", "vertexIds": ["A", "B", "C"]},
                {"faceId": "ACD", "vertexIds": ["A", "C", "D"]},
            ],
            "strokes": [],
        }
        with self.assertRaisesRegex(ContractError, "coplanar adjacent faces must be merged"):
            VisibilityModel.from_dict(payload).validate()

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

    def test_far_free_stroke_does_not_relax_local_face_planarity(self) -> None:
        payload = {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "local-face-tolerance",
            "vertices": [
                {"vertexId": "A", "entryPosition": (-5, -5, 0)},
                {"vertexId": "B", "entryPosition": (5, -5, 0)},
                {"vertexId": "C", "entryPosition": (5, 5, 1)},
                {"vertexId": "D", "entryPosition": (-5, 5, 0)},
                {"vertexId": "X", "entryPosition": (-1e9, 0, 0)},
                {"vertexId": "Y", "entryPosition": (1e9, 0, 0)},
            ],
            "faces": [{"faceId": "warped", "vertexIds": ["A", "B", "C", "D"]}],
            "strokes": [{"sourceEdgeId": "far", "vertexIds": ["X", "Y"]}],
        }
        with self.assertRaisesRegex(ContractError, "not planar"):
            VisibilityModel.from_dict(payload).validate()

    def test_rejects_self_intersecting_star_even_when_local_turns_match(self) -> None:
        angles = [0, 144, 288, 72, 216]
        payload = {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "star",
            "vertices": [
                {
                    "vertexId": f"P{index}",
                    "entryPosition": (
                        math.cos(math.radians(angle)),
                        math.sin(math.radians(angle)),
                        0,
                    ),
                }
                for index, angle in enumerate(angles)
            ],
            "faces": [{"faceId": "star", "vertexIds": [f"P{i}" for i in range(5)]}],
            "strokes": [],
        }
        with self.assertRaisesRegex(ContractError, "not strictly convex"):
            VisibilityModel.from_dict(payload).validate()

    def test_closed_validation_rejects_empty_flat_and_duplicate_shells(self) -> None:
        empty = VisibilityModel.from_dict({
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "empty",
            "vertices": [],
            "faces": [],
            "strokes": [],
        })
        with self.assertRaisesRegex(ContractError, "closed two-manifold"):
            empty.validate(require_closed_convex_manifold=True)

        flat = {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "flat",
            "vertices": [
                {"vertexId": "A", "entryPosition": (-1, -1, 0)},
                {"vertexId": "B", "entryPosition": (1, -1, 0)},
                {"vertexId": "C", "entryPosition": (1, 1, 0)},
                {"vertexId": "D", "entryPosition": (-1, 1, 0)},
            ],
            "faces": [
                {"faceId": "front", "vertexIds": ["A", "B", "C", "D"]},
                {"faceId": "back", "vertexIds": ["D", "C", "B", "A"]},
            ],
            "strokes": [],
        }
        with self.assertRaisesRegex(
            ContractError,
            "closed two-manifold|non-zero volume|duplicate surface|coplanar adjacent faces",
        ):
            VisibilityModel.from_dict(flat).validate(require_closed_convex_manifold=True)

    def test_rejects_invalid_incidence_and_duplicate_identity(self) -> None:
        invalid = cube_payload()
        invalid["strokes"][0]["incidentFaceIds"] = ["front"]
        with self.assertRaisesRegex(ContractError, "does not contain both endpoints"):
            VisibilityModel.from_dict(invalid)

        duplicate = cube_payload()
        duplicate["faces"].append(dict(duplicate["faces"][0]))
        with self.assertRaisesRegex(ContractError, "duplicate faceId"):
            VisibilityModel.from_dict(duplicate)

    def test_direct_dataclass_construction_is_revalidated(self) -> None:
        model = VisibilityModel(
            visibility_group_id="direct",
            vertices=(
                VertexSpec("A", (0, 0, 0)),
                VertexSpec("B", (1, 0, 0)),
                VertexSpec("C", (0, 1, 0)),
            ),
            faces=(
                FaceSpec("same", ("A", "B", "C")),
                FaceSpec("same", ("A", "B", "C")),
            ),
            strokes=(StrokeSpec("edge", ("A", "B"), visibility_mode="mystery"),),
            schema="wrong-schema",
        )
        with self.assertRaisesRegex(ContractError, "schema|duplicate|unsupported"):
            model.validate()


if __name__ == "__main__":
    unittest.main()
