from __future__ import annotations

import copy
import hashlib
from math import cos, pi, sin
import unittest

import numpy as np

from polyhedron_visibility import (
    VisibilityModel as FrozenVisibilityModel,
    canonical_trace_json as frozen_trace_json,
    compute_frame_visibility as compute_frozen_visibility,
)
from polyhedron_visibility.open_faces import (
    OPEN_FACE_MODEL_SCHEMA,
    OPEN_FACE_TOPOLOGY,
    OPEN_FACE_TRACE_SCHEMA,
    OpenFaceSolverError,
    OpenFaceVisibilityModel,
    canonical_open_face_trace_json,
    compute_open_face_visibility,
)


IDENTITY_VIEW = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
FROZEN_V1_SENTINEL_SHA256 = "8e596b3f94608d153b86972c3ad3d0dc44ff541edc81ccdde779c3a951e7a6b7"


def dihedral_payload(*, angle: float = 0.7, scale: float = 1.0) -> dict[str, object]:
    cosine = 1.0 if angle == 0.0 else -1.0 if angle == pi else cos(angle)
    sine = 0.0 if angle in {0.0, pi} else sin(angle)
    points = {
        "A": (0.0, -scale, 0.0),
        "B": (0.0, scale, 0.0),
        "Alpha0": (2.0 * scale, -scale, 0.0),
        "Alpha1": (2.0 * scale, scale, 0.0),
        "Beta0": (2.0 * scale * cosine, -scale, 2.0 * scale * sine),
        "Beta1": (2.0 * scale * cosine, scale, 2.0 * scale * sine),
        "M": (1.4 * scale * cosine, 0.0, 1.4 * scale * sine),
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
            {"sourceEdgeId": "probe", "vertexIds": ["S", "E"]},
            {
                "sourceEdgeId": "helper-MN",
                "vertexIds": ["M", "N"],
                "excludedOccluderFaceIds": ["beta-face"],
            },
        ],
    }


def positions_at(
    model: OpenFaceVisibilityModel,
    angle: float,
    *,
    scale: float = 1.0,
) -> dict[str, tuple[float, float, float]]:
    cosine = 1.0 if angle == 0.0 else -1.0 if angle == pi else cos(angle)
    sine = 0.0 if angle in {0.0, pi} else sin(angle)
    result = dict(model.entry_positions)
    result.update(
        {
            "A": (0.0, -scale, 0.0),
            "B": (0.0, scale, 0.0),
            "Alpha0": (2.0 * scale, -scale, 0.0),
            "Alpha1": (2.0 * scale, scale, 0.0),
            "Beta0": (2.0 * scale * cosine, -scale, 2.0 * scale * sine),
            "Beta1": (2.0 * scale * cosine, scale, 2.0 * scale * sine),
            "M": (1.4 * scale * cosine, 0.0, 1.4 * scale * sine),
            "N": (0.0, 0.0, 0.0),
            "S": (-scale, 0.0, -scale),
            "E": (3.0 * scale, 0.0, -scale),
        }
    )
    return result


def frozen_v1_payload() -> dict[str, object]:
    return {
        "schema": "manim-convex-polyhedron-visibility/v1",
        "visibilityGroupId": "open-face-compat-sentinel",
        "vertices": [
            {"vertexId": "L", "entryPosition": [-2, 0, 0]},
            {"vertexId": "R", "entryPosition": [2, 0, 0]},
            {"vertexId": "A", "entryPosition": [-1, -1, 1]},
            {"vertexId": "B", "entryPosition": [1, -1, 1]},
            {"vertexId": "C", "entryPosition": [1, 1, 1]},
            {"vertexId": "D", "entryPosition": [-1, 1, 1]},
        ],
        "faces": [{"faceId": "panel", "vertexIds": ["A", "B", "C", "D"]}],
        "strokes": [
            {
                "sourceEdgeId": "probe",
                "vertexIds": ["L", "R"],
                "incidentFaceIds": [],
            }
        ],
    }


class OpenFaceVisibilitySolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = OpenFaceVisibilityModel.from_dict(dihedral_payload())

    def solve(self, angle: float, *, scale: float = 1.0):
        return compute_open_face_visibility(
            self.model,
            projection_matrix=IDENTITY_VIEW,
            vertex_positions=positions_at(self.model, angle, scale=scale),
        )

    def test_exact_coplanar_before_middle_after_are_all_successful_frames(self) -> None:
        before = self.solve(-0.2)
        exact = self.solve(0.0)
        after = self.solve(0.2)

        self.assertEqual(before.seam_state_map["fold-angle"].state, "open")
        self.assertEqual(exact.schema, OPEN_FACE_TRACE_SCHEMA)
        self.assertEqual(
            exact.seam_state_map["fold-angle"].state,
            "coplanar_same_normal",
        )
        self.assertEqual(exact.seam_state_map["fold-angle"].dihedral_radians, 0.0)
        self.assertEqual(after.seam_state_map["fold-angle"].state, "open")

    def test_exact_full_overlap_preserves_face_and_surface_provenance(self) -> None:
        frame = self.solve(0.0)
        edge = frame.edge_map["probe"]
        intervals = [item for item in edge.raw_intervals if item.face_id != "__policy__"]

        self.assertEqual(len(intervals), 2)
        self.assertAlmostEqual(intervals[0].start, intervals[1].start, places=12)
        self.assertAlmostEqual(intervals[0].end, intervals[1].end, places=12)
        hidden = [span for span in edge.spans if span.kind == "hidden"]
        self.assertEqual(len(hidden), 1)
        self.assertEqual(hidden[0].face_level, 2)
        self.assertEqual(hidden[0].surface_level, 2)
        self.assertEqual(
            hidden[0].occluder_logical_surface_ids,
            ("alpha", "beta"),
        )

    def test_zero_and_pi_are_both_legal_coplanar_seam_states(self) -> None:
        zero = self.solve(0.0).seam_state_map["fold-angle"]
        opposite = self.solve(pi).seam_state_map["fold-angle"]

        self.assertEqual(zero.state, "coplanar_same_normal")
        self.assertEqual(zero.dihedral_radians, 0.0)
        self.assertEqual(opposite.state, "coplanar_opposite_normal")
        self.assertEqual(opposite.dihedral_radians, pi)

    def test_declared_hinge_is_inclusive_and_never_opens_a_visibility_crack(self) -> None:
        # At pi the panels occupy opposite half-planes.  Just before and after
        # pi they remain a folded pair.  The probe crosses their projected
        # shared edge at t=0.25 in all three frames.
        seam_parameter = 0.25
        for angle in (pi - 0.1, pi, pi + 0.1):
            with self.subTest(angle=angle):
                edge = self.solve(angle).edge_map["probe"]
                visible_at_seam = [
                    span
                    for span in edge.spans
                    if span.kind == "visible"
                    and span.start < seam_parameter + edge.parameter_epsilon
                    and span.end > seam_parameter - edge.parameter_epsilon
                ]
                self.assertEqual(visible_at_seam, [])
                hidden_left = max(
                    span.end
                    for span in edge.spans
                    if span.kind == "hidden" and span.start < seam_parameter
                )
                hidden_right = min(
                    span.start
                    for span in edge.spans
                    if span.kind == "hidden" and span.end > seam_parameter
                )
                self.assertLessEqual(
                    hidden_right - hidden_left,
                    edge.parameter_epsilon,
                )

    def test_excluded_coplanar_helper_is_proven_each_frame_and_traced(self) -> None:
        frame = self.solve(0.63)
        helper = frame.edge_map["helper-MN"]
        self.assertIn(
            ("beta-face", "beta", "excluded_coplanar_stroke"),
            [
                (item.face_id, item.logical_surface_id, item.reason)
                for item in helper.skipped_occluders
            ],
        )

        invalid = positions_at(self.model, 0.63)
        invalid["M"] = tuple(np.asarray(invalid["M"]) + np.asarray((0.0, 0.0, 0.3)))
        with self.assertRaises(OpenFaceSolverError) as caught:
            compute_open_face_visibility(
                self.model,
                projection_matrix=IDENTITY_VIEW,
                vertex_positions=invalid,
            )
        self.assertEqual(caught.exception.code, "UNPROVEN_COPLANAR_EXCLUSION")

    def test_scale_does_not_change_normalized_visibility_or_seam_state(self) -> None:
        normalized = []
        for scale in (1.0e-6, 1.0, 1.0e6):
            model = OpenFaceVisibilityModel.from_dict(
                dihedral_payload(angle=pi, scale=scale)
            )
            frame = compute_open_face_visibility(
                model,
                projection_matrix=IDENTITY_VIEW,
            )
            normalized.append(
                (
                    frame.seam_state_map["fold-angle"].state,
                    [
                        (round(span.start, 7), round(span.end, 7), span.kind)
                        for span in frame.edge_map["probe"].spans
                    ],
                )
            )
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[1], normalized[2])

    def test_input_collection_order_does_not_change_canonical_trace(self) -> None:
        reordered_payload = copy.deepcopy(dihedral_payload())
        for key in ("vertices", "faces", "seams", "strokes"):
            reordered_payload[key].reverse()  # type: ignore[union-attr]
        first = compute_open_face_visibility(
            OpenFaceVisibilityModel.from_dict(dihedral_payload()),
            projection_matrix=IDENTITY_VIEW,
        )
        second = compute_open_face_visibility(
            OpenFaceVisibilityModel.from_dict(reordered_payload),
            projection_matrix=IDENTITY_VIEW,
        )
        self.assertEqual(
            canonical_open_face_trace_json(first),
            canonical_open_face_trace_json(second),
        )

    def test_new_package_does_not_change_the_frozen_v1_trace_hash(self) -> None:
        model = FrozenVisibilityModel.from_dict(frozen_v1_payload())
        frame = compute_frozen_visibility(model, projection_matrix=IDENTITY_VIEW)
        digest = hashlib.sha256(frozen_trace_json(frame).encode("utf-8")).hexdigest()
        self.assertEqual(digest, FROZEN_V1_SENTINEL_SHA256)


if __name__ == "__main__":
    unittest.main()
