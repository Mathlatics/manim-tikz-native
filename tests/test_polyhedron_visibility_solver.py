from __future__ import annotations

import json
import random
import unittest
from unittest.mock import patch

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
from polyhedron_visibility import parallel_solver as solver_module
from polyhedron_visibility.geometry import GeometryContext as SharedGeometryContext
from polyhedron_visibility.parallel_solver import (
    _segment_face_interval_result,
    _spans_from_intervals,
)
from polyhedron_visibility.trace import (
    RawOcclusionInterval,
    VisibilitySpan as TraceVisibilitySpan,
)
from polyhedron_visibility.visibility import (
    partition_visibility as shared_partition_visibility,
)

IDENTITY_VIEW = np.eye(3)


def _legacy_spans_from_intervals(
    intervals: list[RawOcclusionInterval],
    parameter_epsilon: float,
) -> tuple[TraceVisibilitySpan, ...]:
    """Frozen v1 reference copied from the pre-kernel production solver."""

    boundaries = [0.0, 1.0]
    for item in intervals:
        boundaries.extend((item.start, item.end))
    boundaries.sort()
    unique: list[float] = []
    for value in boundaries:
        value = min(1.0, max(0.0, float(value)))
        if not unique or abs(value - unique[-1]) > parameter_epsilon:
            unique.append(value)
        else:
            unique[-1] = max(unique[-1], value)
    if unique[0] > 0.0:
        unique.insert(0, 0.0)
    if unique[-1] < 1.0:
        unique.append(1.0)

    spans: list[TraceVisibilitySpan] = []
    for start, end in zip(unique, unique[1:]):
        if end - start <= parameter_epsilon:
            continue
        midpoint = 0.5 * (start + end)
        active = tuple(
            sorted(
                item.face_id
                for item in intervals
                if item.start - parameter_epsilon
                <= midpoint
                <= item.end + parameter_epsilon
            )
        )
        kind = "hidden" if active else "visible"
        span = TraceVisibilitySpan(start, end, kind, active, len(active))
        if (
            spans
            and spans[-1].kind == span.kind
            and spans[-1].occluder_face_ids == span.occluder_face_ids
            and abs(spans[-1].end - span.start) <= parameter_epsilon
        ):
            previous = spans[-1]
            spans[-1] = TraceVisibilitySpan(
                previous.start,
                span.end,
                previous.kind,
                previous.occluder_face_ids,
                previous.level,
            )
        else:
            spans.append(span)
    if not spans:
        return (TraceVisibilitySpan(0.0, 1.0, "visible", (), 0),)
    first = spans[0]
    spans[0] = TraceVisibilitySpan(
        0.0,
        first.end,
        first.kind,
        first.occluder_face_ids,
        first.level,
    )
    last = spans[-1]
    spans[-1] = TraceVisibilitySpan(
        last.start,
        1.0,
        last.kind,
        last.occluder_face_ids,
        last.level,
    )
    return tuple(spans)


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


def octahedron_model() -> VisibilityModel:
    positions = {
        "Xp": (1.0, 0.0, 0.0),
        "Xn": (-1.0, 0.0, 0.0),
        "Yp": (0.0, 1.0, 0.0),
        "Yn": (0.0, -1.0, 0.0),
        "T": (0.0, 0.0, 1.0),
        "U": (0.0, 0.0, -1.0),
    }
    faces = (
        ("T-Xp-Yp", ("T", "Xp", "Yp")),
        ("T-Yp-Xn", ("T", "Yp", "Xn")),
        ("T-Xn-Yn", ("T", "Xn", "Yn")),
        ("T-Yn-Xp", ("T", "Yn", "Xp")),
        ("U-Yp-Xp", ("U", "Yp", "Xp")),
        ("U-Xp-Yn", ("U", "Xp", "Yn")),
        ("U-Yn-Xn", ("U", "Yn", "Xn")),
        ("U-Xn-Yp", ("U", "Xn", "Yp")),
    )
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "octahedron",
            "vertices": [
                {"vertexId": vertex_id, "entryPosition": point}
                for vertex_id, point in positions.items()
            ],
            "faces": [
                {"faceId": face_id, "vertexIds": vertex_ids}
                for face_id, vertex_ids in faces
            ],
            "strokes": [],
        }
    )


class ParallelVisibilitySolverTests(unittest.TestCase):
    def test_production_solver_reaches_shared_geometry_and_visibility_layers(self) -> None:
        created_contexts: list[SharedGeometryContext] = []

        def tracked_context(*args, **kwargs):
            context = SharedGeometryContext(*args, **kwargs)
            created_contexts.append(context)
            return context

        with (
            patch.object(solver_module, "GeometryContext", side_effect=tracked_context),
            patch.object(
                solver_module,
                "partition_visibility",
                wraps=shared_partition_visibility,
            ) as partition_mock,
        ):
            frame = compute_frame_visibility(
                face_model(),
                projection_matrix=IDENTITY_VIEW,
            )

        self.assertTrue(created_contexts)
        self.assertGreater(partition_mock.call_count, 0)
        self.assertEqual(frame.edge_map["probe"].spans[1].kind, "hidden")

    def test_shared_partition_matches_frozen_v1_trace_exactly(self) -> None:
        targeted = [
            RawOcclusionInterval("face-d", 0.0008, 0.7),
            RawOcclusionInterval("face-b", 0.0025, 0.02),
            RawOcclusionInterval("face-c", 0.5098, 0.7),
            RawOcclusionInterval("face-a", 0.5101, 0.9),
        ]
        for epsilon in (0.0, 1.0e-9, 1.0e-3):
            with self.subTest(kind="targeted", epsilon=epsilon):
                self.assertEqual(
                    _spans_from_intervals(targeted, epsilon),
                    _legacy_spans_from_intervals(targeted, epsilon),
                )

        random_source = random.Random(20260821)
        for epsilon in (0.0, 1.0e-12, 1.0e-9, 1.0e-3, 5.0e-2):
            for sample in range(1000):
                intervals: list[RawOcclusionInterval] = []
                for index in range(random_source.randrange(0, 8)):
                    start = random_source.random()
                    end = random_source.random()
                    if end < start:
                        start, end = end, start
                    if end - start <= epsilon:
                        continue
                    intervals.append(
                        RawOcclusionInterval(f"face-{index}", start, end)
                    )
                intervals.sort(
                    key=lambda item: (item.start, item.end, item.face_id)
                )
                with self.subTest(
                    kind="random",
                    epsilon=epsilon,
                    sample=sample,
                ):
                    self.assertEqual(
                        _spans_from_intervals(intervals, epsilon),
                        _legacy_spans_from_intervals(intervals, epsilon),
                    )

    def test_private_interval_helper_keeps_legacy_tolerance_keyword(self) -> None:
        result = _segment_face_interval_result(
            (-2.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            ),
            ParallelView.from_matrix(IDENTITY_VIEW),
            tolerance_policy=TolerancePolicy(),
        )
        self.assertIsNotNone(result.interval)
        assert result.interval is not None
        self.assertAlmostEqual(result.interval[0], 0.25, places=7)
        self.assertAlmostEqual(result.interval[1], 0.75, places=7)

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

    def test_parallel_view_projection_validation_is_scale_invariant(self) -> None:
        for screen_scale in (1.0e-200, 1.0e-150, 1.0, 1.0e150, 1.0e200):
            with self.subTest(screen_scale=screen_scale):
                view = ParallelView.from_matrix(
                    np.diag((screen_scale, screen_scale, 1.0))
                )
                np.testing.assert_allclose(
                    view.view_direction,
                    (0.0, 0.0, 1.0),
                    rtol=0.0,
                    atol=1.0e-15,
                )

        mixed = ParallelView.from_matrix(
            np.diag((1.0e-200, 1.0e200, 1.0e-150))
        )
        np.testing.assert_allclose(
            mixed.view_direction,
            (0.0, 0.0, 1.0),
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_scaled_parallel_projection_preserves_occlusion_interval(self) -> None:
        start = (-2.0, 0.17, 0.0)
        end = (2.0, -0.23, 0.0)
        face = (
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        )
        expected = segment_face_occlusion_interval(
            start, end, face, ParallelView.from_matrix(IDENTITY_VIEW)
        )
        self.assertIsNotNone(expected)
        for scales in (
            (1.0e-200, 1.0e-150, 1.0e-100),
            (1.0e150, 1.0e200, 1.0e175),
            (1.0e-200, 1.0e200, 1.0e-150),
        ):
            with self.subTest(scales=scales):
                actual = segment_face_occlusion_interval(
                    start,
                    end,
                    face,
                    ParallelView.from_matrix(np.diag(scales)),
                )
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)

    def test_parallel_view_rejects_nonfinite_projection_rows(self) -> None:
        with self.assertRaisesRegex(SolverError, "finite 3x3"):
            ParallelView.from_matrix(
                ((np.nan, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            )

    def test_parallel_view_direct_constructor_cannot_bypass_matrix_contract(self) -> None:
        matrix = tuple(tuple(float(value) for value in row) for row in IDENTITY_VIEW)
        with self.assertRaisesRegex(SolverError, "view direction"):
            ParallelView(matrix, (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(SolverError, "view direction"):
            ParallelView(matrix, (0.0, 0.0, -1.0))
        with self.assertRaisesRegex(SolverError, "unit"):
            ParallelView(matrix, (0.0, 0.0, 2.0))
        with self.assertRaisesRegex(SolverError, "projection"):
            ParallelView(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                (0.0, 0.0, 1.0),
            )

    def test_public_face_solver_rejects_nonconvex_and_nonplanar_faces(self) -> None:
        view = ParallelView.from_matrix(IDENTITY_VIEW)
        with self.assertRaisesRegex(SolverError, "convex"):
            segment_face_occlusion_interval(
                (-2, 0, 0),
                (2, 0, 0),
                [(-1, -1, 1), (1, -1, 1), (0, -0.2, 1), (1, 1, 1), (-1, 1, 1)],
                view,
            )
        with self.assertRaisesRegex(SolverError, "planar"):
            segment_face_occlusion_interval(
                (-2, 0, 0),
                (2, 0, 0),
                [(-1, -1, 1), (1, -1, 1), (1, 1, 1.2), (-1, 1, 1)],
                view,
            )

    def test_tiny_face_still_occludes_a_very_long_semantic_stroke(self) -> None:
        interval = segment_face_occlusion_interval(
            (-5.0e7, 0, 0),
            (5.0e7, 0, 0),
            [(-0.5, -1, 1), (0.5, -1, 1), (0.5, 1, 1), (-0.5, 1, 1)],
            ParallelView.from_matrix(IDENTITY_VIEW),
        )
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertGreater(interval[1] - interval[0], 5.0e-9)
        self.assertAlmostEqual((interval[0] + interval[1]) / 2, 0.5, places=12)

        model = VisibilityModel.from_dict({
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "mixed-scale-trace",
            "vertices": [
                {"vertexId": "L", "entryPosition": (-5.0e7, 0, 0)},
                {"vertexId": "R", "entryPosition": (5.0e7, 0, 0)},
                {"vertexId": "A", "entryPosition": (-0.5, -1, 1)},
                {"vertexId": "B", "entryPosition": (0.5, -1, 1)},
                {"vertexId": "C", "entryPosition": (0.5, 1, 1)},
                {"vertexId": "D", "entryPosition": (-0.5, 1, 1)},
            ],
            "faces": [{"faceId": "small", "vertexIds": ["A", "B", "C", "D"]}],
            "strokes": [{"sourceEdgeId": "long", "vertexIds": ["L", "R"]}],
        })
        edge = compute_frame_visibility(model, projection_matrix=IDENTITY_VIEW).edge_map["long"]
        self.assertLess(edge.parameter_epsilon, 2.0e-9)
        self.assertLess(edge.face_tolerances[0].world, 1.0e-7)

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

        direct_payload = face_model().to_dict()
        direct = VisibilityModel.from_dict(direct_payload)
        reversed_direct = VisibilityModel(
            visibility_group_id=direct.visibility_group_id,
            vertices=direct.vertices,
            faces=tuple(reversed(direct.faces)),
            strokes=direct.strokes,
        )
        self.assertEqual(
            canonical_trace_json(compute_frame_visibility(direct, projection_matrix=IDENTITY_VIEW)),
            canonical_trace_json(
                compute_frame_visibility(reversed_direct, projection_matrix=IDENTITY_VIEW)
            ),
        )

    def test_singular_projection_fails_closed(self) -> None:
        with self.assertRaisesRegex(SolverError, "projection"):
            compute_frame_visibility(
                face_model(),
                projection_matrix=((1, 0, 0), (2, 0, 0), (0, 0, 1)),
            )

    def test_equivalent_uniformly_scaled_projection_is_accepted(self) -> None:
        expected = compute_frame_visibility(face_model(), projection_matrix=IDENTITY_VIEW)
        scaled = compute_frame_visibility(
            face_model(), projection_matrix=np.eye(3) * 1.0e-15
        )
        self.assertEqual(
            [
                (round(item.start, 8), round(item.end, 8), item.kind)
                for item in expected.edge_map["probe"].spans
            ],
            [
                (round(item.start, 8), round(item.end, 8), item.kind)
                for item in scaled.edge_map["probe"].spans
            ],
        )

    def test_strict_closed_convex_mode_revalidates_every_dynamic_frame(self) -> None:
        model = octahedron_model()
        model.validate(require_closed_convex_manifold=True)
        concave_positions = dict(model.entry_positions)
        concave_positions["U"] = (0.0, 0.0, 0.5)

        # Every face is still a valid triangle, so the generic finite-face
        # solver can process this frame when an experimental open-face caller
        # explicitly chooses that contract.
        compute_frame_visibility(
            model,
            projection_matrix=IDENTITY_VIEW,
            vertex_positions=concave_positions,
        )
        with self.assertRaisesRegex(SolverError, "not convex"):
            compute_frame_visibility(
                model,
                projection_matrix=IDENTITY_VIEW,
                vertex_positions=concave_positions,
                require_closed_convex_manifold=True,
            )

    def test_frame_validation_uses_the_callers_tolerance_policy(self) -> None:
        model = face_model()
        warped = dict(model.entry_positions)
        warped["A2"] = (-0.5, 1.0, 1.0001)

        with self.assertRaisesRegex(SolverError, "not planar"):
            compute_frame_visibility(
                model,
                projection_matrix=IDENTITY_VIEW,
                vertex_positions=warped,
            )
        compute_frame_visibility(
            model,
            projection_matrix=IDENTITY_VIEW,
            vertex_positions=warped,
            tolerance_policy=TolerancePolicy(relative=1.0e-3),
        )

    def test_zero_length_semantic_stroke_fails_for_every_visibility_mode(self) -> None:
        for mode in ("auto", "always_visible", "always_hidden"):
            with self.subTest(mode=mode):
                model = VisibilityModel.from_dict(
                    {
                        "schema": "manim-convex-polyhedron-visibility/v1",
                        "visibilityGroupId": f"zero-{mode}",
                        "vertices": [
                            {"vertexId": "A", "entryPosition": (0, 0, 0)},
                            {"vertexId": "B", "entryPosition": (1, 0, 0)},
                        ],
                        "faces": [],
                        "strokes": [
                            {
                                "sourceEdgeId": "AB",
                                "vertexIds": ["A", "B"],
                                "visibilityMode": mode,
                            }
                        ],
                    }
                )
                with self.assertRaisesRegex(SolverError, "zero length"):
                    compute_frame_visibility(
                        model,
                        projection_matrix=IDENTITY_VIEW,
                        vertex_positions={"A": (0, 0, 0), "B": (0, 0, 0)},
                    )


if __name__ == "__main__":
    unittest.main()
