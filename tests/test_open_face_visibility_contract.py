from __future__ import annotations

import copy
from math import cos, pi, sin
import unittest

from polyhedron_visibility.open_faces import (
    OPEN_FACE_MODEL_SCHEMA,
    OPEN_FACE_TOPOLOGY,
    OpenFaceContractError,
    OpenFaceVisibilityModel,
)


def _trig(angle: float) -> tuple[float, float]:
    if angle == 0.0:
        return 1.0, 0.0
    if angle == pi:
        return -1.0, 0.0
    return cos(angle), sin(angle)


def dihedral_payload(*, angle: float = 0.7, scale: float = 1.0) -> dict[str, object]:
    cosine, sine = _trig(angle)
    radial = (2.0 * scale * cosine, 0.0, 2.0 * scale * sine)
    points = {
        "A": (0.0, -scale, 0.0),
        "B": (0.0, scale, 0.0),
        "Alpha0": (2.0 * scale, -scale, 0.0),
        "Alpha1": (2.0 * scale, scale, 0.0),
        "Beta0": (radial[0], -scale, radial[2]),
        "Beta1": (radial[0], scale, radial[2]),
        "M": (0.7 * radial[0], 0.0, 0.7 * radial[2]),
        "N": (0.0, 0.0, 0.0),
        "S": (-scale, 0.0, -scale),
        "E": (3.0 * scale, 0.0, -scale),
    }
    return {
        "schema": OPEN_FACE_MODEL_SCHEMA,
        "topology": OPEN_FACE_TOPOLOGY,
        "visibilityGroupId": "open-dihedral",
        "vertices": [
            {"vertexId": vertex_id, "entryPosition": point}
            for vertex_id, point in points.items()
        ],
        "faces": [
            {
                "faceId": "alpha-face",
                "logicalSurfaceId": "alpha",
                "vertexIds": ["A", "B", "Alpha1", "Alpha0"],
            },
            {
                "faceId": "beta-face",
                "logicalSurfaceId": "beta",
                "vertexIds": ["A", "B", "Beta1", "Beta0"],
            },
        ],
        "seams": [
            {
                "seamId": "fold-angle",
                "policy": "articulated_hinge",
                "faceIds": ["alpha-face", "beta-face"],
                "vertexIds": ["A", "B"],
            }
        ],
        "strokes": [
            {
                "sourceEdgeId": "hinge",
                "vertexIds": ["A", "B"],
                "incidentFaceIds": ["alpha-face", "beta-face"],
            },
            {
                "sourceEdgeId": "probe",
                "vertexIds": ["S", "E"],
            },
            {
                "sourceEdgeId": "helper-MN",
                "vertexIds": ["M", "N"],
                "excludedOccluderFaceIds": ["beta-face"],
            },
        ],
    }


class OpenFaceVisibilityContractTests(unittest.TestCase):
    def test_contract_round_trip_is_canonical_and_explicit(self) -> None:
        payload = dihedral_payload()
        payload["vertices"].reverse()  # type: ignore[union-attr]
        payload["faces"].reverse()  # type: ignore[union-attr]
        payload["strokes"].reverse()  # type: ignore[union-attr]
        payload["seams"][0]["faceIds"].reverse()  # type: ignore[index,union-attr]

        model = OpenFaceVisibilityModel.from_dict(payload)
        model.validate()
        canonical = model.to_dict()

        self.assertEqual(canonical["schema"], OPEN_FACE_MODEL_SCHEMA)
        self.assertEqual(canonical["topology"], OPEN_FACE_TOPOLOGY)
        self.assertEqual(
            model.face_map["alpha-face"].logical_surface_id,
            "alpha",
        )
        self.assertEqual(
            model.stroke_map["helper-MN"].excluded_occluder_face_ids,
            ("beta-face",),
        )
        self.assertEqual(
            OpenFaceVisibilityModel.from_dict(canonical).to_dict(),
            canonical,
        )

    def test_topology_and_unknown_fields_fail_closed(self) -> None:
        wrong = dihedral_payload()
        wrong["topology"] = "closed_convex_polyhedron"
        with self.assertRaises(OpenFaceContractError) as caught:
            OpenFaceVisibilityModel.from_dict(wrong)
        self.assertEqual(caught.exception.code, "INVALID_TOPOLOGY")

        unknown = dihedral_payload()
        unknown["guessFaces"] = True
        with self.assertRaises(OpenFaceContractError) as caught:
            OpenFaceVisibilityModel.from_dict(unknown)
        self.assertEqual(caught.exception.code, "UNKNOWN_FIELD")

    def test_one_maximal_face_per_logical_surface_is_mandatory(self) -> None:
        payload = dihedral_payload()
        payload["faces"][1]["logicalSurfaceId"] = "alpha"  # type: ignore[index]
        with self.assertRaises(OpenFaceContractError) as caught:
            OpenFaceVisibilityModel.from_dict(payload)
        self.assertEqual(
            caught.exception.code,
            "MULTIPLE_FACES_PER_LOGICAL_SURFACE",
        )

    def test_every_shared_boundary_requires_one_articulated_hinge(self) -> None:
        missing = dihedral_payload()
        missing["seams"] = []
        with self.assertRaises(OpenFaceContractError) as caught:
            OpenFaceVisibilityModel.from_dict(missing)
        self.assertEqual(caught.exception.code, "UNDECLARED_SHARED_BOUNDARY")

        wrong_policy = dihedral_payload()
        wrong_policy["seams"][0]["policy"] = "visual_only"  # type: ignore[index]
        with self.assertRaises(OpenFaceContractError) as caught:
            OpenFaceVisibilityModel.from_dict(wrong_policy)
        self.assertEqual(caught.exception.code, "INVALID_SEAM_POLICY")

        wrong_edge = dihedral_payload()
        wrong_edge["seams"][0]["vertexIds"] = ["A", "Beta1"]  # type: ignore[index]
        with self.assertRaises(OpenFaceContractError) as caught:
            OpenFaceVisibilityModel.from_dict(wrong_edge)
        self.assertEqual(caught.exception.code, "SEAM_NOT_SHARED_BOUNDARY")

    def test_incident_faces_remain_topological_and_exclusions_require_coplanarity(self) -> None:
        invalid_incident = dihedral_payload()
        invalid_incident["strokes"][1]["incidentFaceIds"] = ["alpha-face"]  # type: ignore[index]
        with self.assertRaises(OpenFaceContractError) as caught:
            OpenFaceVisibilityModel.from_dict(invalid_incident)
        self.assertEqual(caught.exception.code, "INVALID_INCIDENT_FACE")

        model = OpenFaceVisibilityModel.from_dict(dihedral_payload())
        model.validate()
        displaced = dict(model.entry_positions)
        point = list(displaced["M"])
        point[2] += 0.25
        displaced["M"] = tuple(point)
        with self.assertRaises(OpenFaceContractError) as caught:
            model.validate(vertex_positions=displaced)
        self.assertEqual(caught.exception.code, "UNPROVEN_COPLANAR_EXCLUSION")

    def test_distinct_hinged_surfaces_may_enter_exact_full_overlap(self) -> None:
        model = OpenFaceVisibilityModel.from_dict(dihedral_payload(angle=0.0))
        model.validate()

        self.assertEqual(
            model.entry_positions["Alpha0"],
            model.entry_positions["Beta0"],
        )
        self.assertNotEqual(
            model.vertex_map["Alpha0"].vertex_id,
            model.vertex_map["Beta0"].vertex_id,
        )

    def test_redundant_incident_and_excluded_face_is_rejected(self) -> None:
        payload = dihedral_payload()
        stroke = payload["strokes"][0]  # type: ignore[index]
        stroke["excludedOccluderFaceIds"] = ["alpha-face"]
        with self.assertRaises(OpenFaceContractError) as caught:
            OpenFaceVisibilityModel.from_dict(payload)
        self.assertEqual(caught.exception.code, "REDUNDANT_OCCLUDER_EXCLUSION")

    def test_direct_dataclass_values_are_revalidated(self) -> None:
        model = OpenFaceVisibilityModel.from_dict(dihedral_payload())
        invalid = OpenFaceVisibilityModel(
            visibility_group_id=model.visibility_group_id,
            vertices=model.vertices,
            faces=model.faces,
            seams=tuple(
                copy.copy(seam) if index else seam.__class__(
                    seam.seam_id,
                    "unsupported",
                    seam.face_ids,
                    seam.vertex_ids,
                )
                for index, seam in enumerate(model.seams)
            ),
            strokes=model.strokes,
        )
        with self.assertRaises(OpenFaceContractError) as caught:
            invalid.validate()
        self.assertEqual(caught.exception.code, "INVALID_SEAM_POLICY")


if __name__ == "__main__":
    unittest.main()
