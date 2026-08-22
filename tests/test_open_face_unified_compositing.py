from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from polyhedron_visibility.open_faces.contract import (
    OPEN_FACE_MODEL_SCHEMA,
    OPEN_FACE_TOPOLOGY,
    OpenFaceVisibilityModel,
)
from polyhedron_visibility.open_faces.solver import compute_open_face_visibility
from polyhedron_visibility.open_faces.unified_compositing import (
    OpenFacePaintPolicy,
    OpenFacePaintRelation,
    OpenFaceUnifiedCompositingError,
    canonical_open_face_unified_compositing_json,
    compute_open_face_unified_compositing,
)
from polyhedron_visibility.open_faces.unified_fragments import _cluster_boundaries
from polyhedron_visibility.path_compositing import segment_intersection_parameters


IDENTITY_VIEW = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
ROOT = Path(__file__).resolve().parents[1]


def _panel_probe_model(
    *,
    scale: float = 1.0,
    visibility_mode: str = "auto",
    face_occludes_strokes: bool = True,
    path_depth: float = 0.0,
    path_y: float = 0.0,
    excluded_face: bool = False,
) -> OpenFaceVisibilityModel:
    return OpenFaceVisibilityModel.from_dict(
        {
            "schema": OPEN_FACE_MODEL_SCHEMA,
            "topology": OPEN_FACE_TOPOLOGY,
            "visibilityGroupId": "unified-panel-probe",
            "vertices": [
                {
                    "vertexId": "L",
                    "entryPosition": [
                        -2.0 * scale,
                        path_y * scale,
                        path_depth * scale,
                    ],
                },
                {
                    "vertexId": "R",
                    "entryPosition": [
                        2.0 * scale,
                        path_y * scale,
                        path_depth * scale,
                    ],
                },
                {"vertexId": "A", "entryPosition": [-scale, -scale, scale]},
                {"vertexId": "B", "entryPosition": [scale, -scale, scale]},
                {"vertexId": "C", "entryPosition": [scale, scale, scale]},
                {"vertexId": "D", "entryPosition": [-scale, scale, scale]},
            ],
            "faces": [
                {
                    "faceId": "panel",
                    "logicalSurfaceId": "surface-panel",
                    "vertexIds": ["A", "B", "C", "D"],
                    "occludesStrokes": face_occludes_strokes,
                }
            ],
            "seams": [],
            "strokes": [
                {
                    "sourceEdgeId": "probe",
                    "vertexIds": ["L", "R"],
                    "incidentFaceIds": [],
                    "excludedOccluderFaceIds": (
                        ["panel"] if excluded_face else []
                    ),
                    "visibilityMode": visibility_mode,
                }
            ],
        }
    )


def _reverse_collinear_model() -> OpenFaceVisibilityModel:
    return OpenFaceVisibilityModel.from_dict(
        {
            "schema": OPEN_FACE_MODEL_SCHEMA,
            "topology": OPEN_FACE_TOPOLOGY,
            "visibilityGroupId": "reverse-collinear",
            "vertices": [
                {"vertexId": "F0", "entryPosition": [0.0, 0.0, 0.0]},
                {"vertexId": "F1", "entryPosition": [10.0, 0.0, 1.0]},
                {"vertexId": "S0", "entryPosition": [8.0, 0.0, 0.4]},
                {"vertexId": "S1", "entryPosition": [2.0, 0.0, 1.4]},
                {"vertexId": "A", "entryPosition": [0.0, 9.0, 0.0]},
                {"vertexId": "B", "entryPosition": [1.0, 9.0, 0.0]},
                {"vertexId": "C", "entryPosition": [1.0, 10.0, 0.0]},
                {"vertexId": "D", "entryPosition": [0.0, 10.0, 0.0]},
            ],
            "faces": [
                {
                    "faceId": "remote-panel",
                    "logicalSurfaceId": "remote-surface",
                    "vertexIds": ["A", "B", "C", "D"],
                }
            ],
            "seams": [],
            "strokes": [
                {"sourceEdgeId": "first", "vertexIds": ["F0", "F1"]},
                {"sourceEdgeId": "second", "vertexIds": ["S0", "S1"]},
            ],
        }
    )


def _crossing_paths_model() -> OpenFaceVisibilityModel:
    return OpenFaceVisibilityModel.from_dict(
        {
            "schema": OPEN_FACE_MODEL_SCHEMA,
            "topology": OPEN_FACE_TOPOLOGY,
            "visibilityGroupId": "crossing-paths",
            "vertices": [
                {"vertexId": "H0", "entryPosition": [-2.0, 0.0, 0.0]},
                {"vertexId": "H1", "entryPosition": [2.0, 0.0, 0.0]},
                {"vertexId": "V0", "entryPosition": [0.0, -2.0, 1.0]},
                {"vertexId": "V1", "entryPosition": [0.0, 2.0, 1.0]},
                {"vertexId": "A", "entryPosition": [5.0, 5.0, 0.0]},
                {"vertexId": "B", "entryPosition": [6.0, 5.0, 0.0]},
                {"vertexId": "C", "entryPosition": [6.0, 6.0, 0.0]},
                {"vertexId": "D", "entryPosition": [5.0, 6.0, 0.0]},
            ],
            "faces": [
                {
                    "faceId": "remote-panel",
                    "logicalSurfaceId": "remote-surface",
                    "vertexIds": ["A", "B", "C", "D"],
                }
            ],
            "seams": [],
            "strokes": [
                {"sourceEdgeId": "horizontal", "vertexIds": ["H0", "H1"]},
                {"sourceEdgeId": "vertical", "vertexIds": ["V0", "V1"]},
            ],
        }
    )


def _crossing_arrangement_model(path_count: int) -> OpenFaceVisibilityModel:
    if path_count < 2:
        raise ValueError("path_count must be at least two")
    vertices = [
        {"vertexId": "A", "entryPosition": [-1.0, 100.0, 0.0]},
        {"vertexId": "B", "entryPosition": [1.0, 100.0, 0.0]},
        {"vertexId": "C", "entryPosition": [1.0, 102.0, 0.0]},
        {"vertexId": "D", "entryPosition": [-1.0, 102.0, 0.0]},
    ]
    strokes = []
    denominator = float(path_count - 1)
    for index in range(path_count):
        left = index / denominator
        right = -(index / denominator) ** 2
        vertices.extend(
            (
                {
                    "vertexId": f"L{index}",
                    "entryPosition": [-1.0, left, float(index)],
                },
                {
                    "vertexId": f"R{index}",
                    "entryPosition": [1.0, right, float(index)],
                },
            )
        )
        strokes.append(
            {
                "sourceEdgeId": f"path-{index:03d}",
                "vertexIds": [f"L{index}", f"R{index}"],
            }
        )
    return OpenFaceVisibilityModel.from_dict(
        {
            "schema": OPEN_FACE_MODEL_SCHEMA,
            "topology": OPEN_FACE_TOPOLOGY,
            "visibilityGroupId": "crossing-arrangement",
            "vertices": vertices,
            "faces": [
                {
                    "faceId": "remote-panel",
                    "logicalSurfaceId": "remote-surface",
                    "vertexIds": ["A", "B", "C", "D"],
                }
            ],
            "seams": [],
            "strokes": strokes,
        }
    )


def _parallel_faces_model() -> OpenFaceVisibilityModel:
    vertices = []
    for prefix, depth in (("back", 0.0), ("front", 1.0)):
        for suffix, point in zip(
            ("A", "B", "C", "D"),
            ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
        ):
            vertices.append(
                {
                    "vertexId": f"{prefix}-{suffix}",
                    "entryPosition": [point[0], point[1], depth],
                }
            )
    return OpenFaceVisibilityModel.from_dict(
        {
            "schema": OPEN_FACE_MODEL_SCHEMA,
            "topology": OPEN_FACE_TOPOLOGY,
            "visibilityGroupId": "parallel-faces",
            "vertices": vertices,
            "faces": [
                {
                    "faceId": prefix,
                    "logicalSurfaceId": f"surface-{prefix}",
                    "vertexIds": [f"{prefix}-{suffix}" for suffix in "ABCD"],
                }
                for prefix in ("back", "front")
            ],
            "seams": [],
            "strokes": [],
        }
    )


class ProjectedPathGeometryTests(unittest.TestCase):
    def test_reverse_collinear_parameters_preserve_screen_correspondence(self) -> None:
        result = segment_intersection_parameters(
            np.asarray((0.0, 0.0)),
            np.asarray((10.0, 0.0)),
            np.asarray((8.0, 0.0)),
            np.asarray((2.0, 0.0)),
            1.0e-9,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], "overlap")
        first_a, first_b, second_a, second_b = result[1]
        self.assertAlmostEqual(first_a, 0.2)
        self.assertAlmostEqual(first_b, 0.8)
        self.assertAlmostEqual(second_a, 1.0)
        self.assertAlmostEqual(second_b, 0.0)

        differences = (
            first_a - (0.4 + second_a),
            first_b - (0.4 + second_b),
        )
        ratio = -differences[0] / (differences[1] - differences[0])
        self.assertAlmostEqual(first_a + ratio * (first_b - first_a), 0.65)
        self.assertAlmostEqual(second_a + ratio * (second_b - second_a), 0.25)

    def test_boundary_clustering_is_bounded_by_cluster_span(self) -> None:
        values = (0.0, 0.20, 0.29, 0.38, 0.47, 1.0)
        priorities = {value: (3 if value in {0.0, 1.0} else 0) for value in values}
        clustered = _cluster_boundaries(values, priorities, 0.10)
        self.assertEqual(len(clustered), 4)
        self.assertAlmostEqual(clustered[0], 0.0)
        self.assertAlmostEqual(clustered[1], 0.245)
        self.assertAlmostEqual(clustered[2], 0.425)
        self.assertAlmostEqual(clustered[3], 1.0)

    def test_reverse_collinear_depth_root_splits_both_paths_at_same_point(self) -> None:
        frame = compute_open_face_unified_compositing(
            _reverse_collinear_model(),
            projection_matrix=IDENTITY_VIEW,
        )
        first_boundaries = {
            value
            for fragment in frame.path_fragments
            if fragment.source_path_id == "first"
            for value in (
                fragment.parameter_interval.start,
                fragment.parameter_interval.end,
            )
        }
        second_boundaries = {
            value
            for fragment in frame.path_fragments
            if fragment.source_path_id == "second"
            for value in (
                fragment.parameter_interval.start,
                fragment.parameter_interval.end,
            )
        }
        self.assertTrue(any(abs(value - 0.65) <= 1.0e-9 for value in first_boundaries))
        self.assertTrue(any(abs(value - 0.25) <= 1.0e-9 for value in second_boundaries))


class OpenFaceUnifiedCompositingTests(unittest.TestCase):
    def _middle_probe_fragment(self, frame):
        return next(
            fragment
            for fragment in frame.path_fragments
            if fragment.source_path_id == "probe"
            and fragment.parameter_interval.start < 0.5
            < fragment.parameter_interval.end
        )

    def test_public_pure_api_imports_without_manim(self) -> None:
        script = r'''
import sys
from polyhedron_visibility.open_faces import (
    OpenFacePaintPolicy,
    OpenFaceUnifiedCompositingFrame,
    compute_open_face_unified_compositing,
)
assert OpenFacePaintPolicy.DIAGRAMMATIC.value == "diagrammatic"
assert callable(compute_open_face_unified_compositing)
assert OpenFaceUnifiedCompositingFrame.__name__
assert "manim" not in sys.modules
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(
        importlib.util.find_spec("manim") is not None,
        "full public star import requires the installed Manim dependency",
    )
    def test_public_star_import_includes_unified_api(self) -> None:
        namespace: dict[str, object] = {}
        exec("from polyhedron_visibility.open_faces import *", namespace)
        self.assertIn("OpenFacePaintPolicy", namespace)
        self.assertIn("compute_open_face_unified_compositing", namespace)

    def test_diagrammatic_and_physical_hidden_path_orders_differ_explicitly(self) -> None:
        model = _panel_probe_model()
        visibility = compute_open_face_visibility(
            model,
            projection_matrix=IDENTITY_VIEW,
        )
        diagrammatic = compute_open_face_unified_compositing(
            model,
            projection_matrix=IDENTITY_VIEW,
            paint_policy="diagrammatic",
        )
        physical = compute_open_face_unified_compositing(
            model,
            projection_matrix=IDENTITY_VIEW,
            paint_policy="physical",
        )
        self.assertEqual(diagrammatic.visibility.to_dict(), visibility.to_dict())
        self.assertEqual(physical.visibility.to_dict(), visibility.to_dict())
        hidden = next(
            fragment
            for fragment in diagrammatic.path_fragments
            if fragment.visibility_kind.value == "hidden"
        )
        self.assertIn(
            ("face:panel", hidden.fragment_id, "diagrammatic_hidden_path"),
            [
                (item.far_item_id, item.near_item_id, item.reason)
                for item in diagrammatic.order_relations
            ],
        )
        self.assertIn(
            (hidden.fragment_id, "face:panel", "physical_hidden_path"),
            [
                (item.far_item_id, item.near_item_id, item.reason)
                for item in physical.order_relations
            ],
        )

    def test_diagrammatic_policy_honors_explicit_visibility_overrides(self) -> None:
        for visibility_mode, expected_kind in (
            ("always_visible", "visible"),
            ("always_hidden", "hidden"),
        ):
            for path_depth, physical_direction in (
                (0.0, ("path", "face")),
                (2.0, ("face", "path")),
            ):
                with self.subTest(
                    visibility_mode=visibility_mode,
                    path_depth=path_depth,
                ):
                    model = _panel_probe_model(
                        visibility_mode=visibility_mode,
                        path_depth=path_depth,
                    )
                    diagrammatic = compute_open_face_unified_compositing(
                        model,
                        projection_matrix=IDENTITY_VIEW,
                        paint_policy="diagrammatic",
                    )
                    physical = compute_open_face_unified_compositing(
                        model,
                        projection_matrix=IDENTITY_VIEW,
                        paint_policy="physical",
                    )
                    diagrammatic_fragment = self._middle_probe_fragment(diagrammatic)
                    physical_fragment = self._middle_probe_fragment(physical)
                    self.assertEqual(
                        diagrammatic_fragment.visibility_kind.value,
                        expected_kind,
                    )
                    self.assertIn(
                        (
                            "face:panel",
                            diagrammatic_fragment.fragment_id,
                            "diagrammatic_path_overlay",
                        ),
                        [
                            (item.far_item_id, item.near_item_id, item.reason)
                            for item in diagrammatic.order_relations
                        ],
                    )
                    expected_physical = (
                        physical_fragment.fragment_id,
                        "face:panel",
                    )
                    if physical_direction == ("face", "path"):
                        expected_physical = tuple(reversed(expected_physical))
                    self.assertIn(
                        (*expected_physical, "path_face_depth"),
                        [
                            (item.far_item_id, item.near_item_id, item.reason)
                            for item in physical.order_relations
                        ],
                    )
                    if visibility_mode == "always_hidden":
                        self.assertEqual(
                            diagrammatic_fragment.occluder_face_ids,
                            ("__policy__",),
                        )
                        self.assertNotIn("face:__policy__", diagrammatic.item_ids)

    def test_diagrammatic_policy_keeps_paths_above_non_occluding_faces(self) -> None:
        model = _panel_probe_model(face_occludes_strokes=False)
        diagrammatic = compute_open_face_unified_compositing(
            model,
            projection_matrix=IDENTITY_VIEW,
            paint_policy="diagrammatic",
        )
        physical = compute_open_face_unified_compositing(
            model,
            projection_matrix=IDENTITY_VIEW,
            paint_policy="physical",
        )
        diagrammatic_fragment = self._middle_probe_fragment(diagrammatic)
        physical_fragment = self._middle_probe_fragment(physical)
        self.assertEqual(diagrammatic_fragment.visibility_kind.value, "visible")
        self.assertIn(
            (
                "face:panel",
                diagrammatic_fragment.fragment_id,
                "diagrammatic_path_overlay",
            ),
            [
                (item.far_item_id, item.near_item_id, item.reason)
                for item in diagrammatic.order_relations
            ],
        )
        self.assertIn(
            (physical_fragment.fragment_id, "face:panel", "path_face_depth"),
            [
                (item.far_item_id, item.near_item_id, item.reason)
                for item in physical.order_relations
            ],
        )

    def test_boundary_contact_and_declared_exclusion_keep_readable_ink(self) -> None:
        boundary_model = _panel_probe_model(path_y=1.0)
        boundary_expectations = {
            "diagrammatic": ("face", "path", "diagrammatic_path_overlay"),
            "physical": ("path", "face", "path_face_depth"),
        }
        for paint_policy, expectation in boundary_expectations.items():
            with self.subTest(case="boundary", paint_policy=paint_policy):
                frame = compute_open_face_unified_compositing(
                    boundary_model,
                    projection_matrix=IDENTITY_VIEW,
                    paint_policy=paint_policy,
                )
                fragment = self._middle_probe_fragment(frame)
                self.assertEqual(fragment.visibility_kind.value, "visible")
                item_ids = {
                    "face": "face:panel",
                    "path": fragment.fragment_id,
                }
                self.assertIn(
                    (
                        item_ids[expectation[0]],
                        item_ids[expectation[1]],
                        expectation[2],
                    ),
                    [
                        (item.far_item_id, item.near_item_id, item.reason)
                        for item in frame.order_relations
                    ],
                )

        excluded_model = _panel_probe_model(
            path_depth=1.0,
            excluded_face=True,
        )
        for paint_policy in ("diagrammatic", "physical"):
            with self.subTest(case="excluded", paint_policy=paint_policy):
                frame = compute_open_face_unified_compositing(
                    excluded_model,
                    projection_matrix=IDENTITY_VIEW,
                    paint_policy=paint_policy,
                )
                fragment = self._middle_probe_fragment(frame)
                self.assertEqual(fragment.visibility_kind.value, "visible")
                self.assertIn(
                    (
                        "face:panel",
                        fragment.fragment_id,
                        "declared_coplanar_path",
                    ),
                    [
                        (item.far_item_id, item.near_item_id, item.reason)
                        for item in frame.order_relations
                    ],
                )

    def test_diagrammatic_policy_does_not_hide_undeclared_coplanar_geometry(self) -> None:
        model = _panel_probe_model(path_depth=1.0)
        for paint_policy in ("diagrammatic", "physical"):
            with self.subTest(paint_policy=paint_policy):
                with self.assertRaisesRegex(
                    OpenFaceUnifiedCompositingError,
                    "indistinguishable depth",
                ):
                    compute_open_face_unified_compositing(
                        model,
                        projection_matrix=IDENTITY_VIEW,
                        paint_policy=paint_policy,
                    )

    def test_face_face_and_path_path_relations_are_real_pairwise_constraints(self) -> None:
        faces = compute_open_face_unified_compositing(
            _parallel_faces_model(),
            projection_matrix=IDENTITY_VIEW,
        )
        self.assertIn(
            ("face:back", "face:front"),
            {(item.far_item_id, item.near_item_id) for item in faces.order_relations},
        )

        paths = compute_open_face_unified_compositing(
            _crossing_paths_model(),
            projection_matrix=IDENTITY_VIEW,
        )
        crossing = [
            item
            for item in paths.order_relations
            if item.reason == "path_crossing_depth"
        ]
        self.assertEqual(len(crossing), 4)
        self.assertTrue(
            all("horizontal" in item.far_item_id for item in crossing)
        )
        self.assertTrue(
            all("vertical" in item.near_item_id for item in crossing)
        )

    def test_source_path_pairs_are_intersected_once_before_fragment_relations(self) -> None:
        path_count = 24
        target = (
            "polyhedron_visibility.open_faces.unified_fragments."
            "segment_intersection_parameters"
        )
        with patch(target, wraps=segment_intersection_parameters) as intersection:
            frame = compute_open_face_unified_compositing(
                _crossing_arrangement_model(path_count),
                projection_matrix=IDENTITY_VIEW,
            )
        self.assertEqual(
            intersection.call_count,
            path_count * (path_count - 1) // 2,
        )
        self.assertEqual(len(frame.path_fragments), path_count * path_count)

    def test_fragment_pair_candidate_limit_fails_before_relation_generation(self) -> None:
        from polyhedron_visibility.open_faces.unified_compositing import (
            OpenFaceUnifiedCompositingLimits,
        )

        with self.assertRaisesRegex(
            OpenFaceUnifiedCompositingError,
            "fragment_pair_candidates=4 exceeds limit 3",
        ):
            compute_open_face_unified_compositing(
                _crossing_paths_model(),
                projection_matrix=IDENTITY_VIEW,
                limits=OpenFaceUnifiedCompositingLimits(
                    max_fragment_pair_candidates=3
                ),
            )
        frame = compute_open_face_unified_compositing(
            _crossing_paths_model(),
            projection_matrix=IDENTITY_VIEW,
            limits=OpenFaceUnifiedCompositingLimits(
                max_fragment_pair_candidates=4
            ),
        )
        self.assertEqual(
            len(
                [
                    item
                    for item in frame.order_relations
                    if item.reason == "path_crossing_depth"
                ]
            ),
            4,
        )

    def test_scale_preserves_fragment_topology_order_and_normalized_boundaries(self) -> None:
        frames = [
            compute_open_face_unified_compositing(
                _panel_probe_model(scale=scale),
                projection_matrix=IDENTITY_VIEW,
            )
            for scale in (1.0e-7, 1.0, 1.0e7)
        ]
        baseline = frames[1]
        for frame in frames:
            self.assertEqual(
                tuple(
                    (fragment.source_path_id, fragment.visibility_kind.value)
                    for fragment in frame.path_fragments
                ),
                tuple(
                    (fragment.source_path_id, fragment.visibility_kind.value)
                    for fragment in baseline.path_fragments
                ),
            )
            self.assertEqual(frame.draw_order, baseline.draw_order)
            self.assertEqual(
                tuple(
                    (item.far_item_id, item.near_item_id, item.reason)
                    for item in frame.order_relations
                ),
                tuple(
                    (item.far_item_id, item.near_item_id, item.reason)
                    for item in baseline.order_relations
                ),
            )
            for actual, expected in zip(
                frame.path_fragments, baseline.path_fragments
            ):
                self.assertAlmostEqual(
                    actual.parameter_interval.start,
                    expected.parameter_interval.start,
                    delta=1.0e-6,
                )
                self.assertAlmostEqual(
                    actual.parameter_interval.end,
                    expected.parameter_interval.end,
                    delta=1.0e-6,
                )

    def test_canonical_output_is_hash_seed_independent(self) -> None:
        script = r'''
from tests.test_open_face_unified_compositing import _panel_probe_model, IDENTITY_VIEW
from polyhedron_visibility.open_faces.unified_compositing import (
    canonical_open_face_unified_compositing_json,
    compute_open_face_unified_compositing,
)
print(canonical_open_face_unified_compositing_json(
    compute_open_face_unified_compositing(
        _panel_probe_model(), projection_matrix=IDENTITY_VIEW
    )
))
'''
        outputs = []
        for seed in ("1", "2", "17", "101"):
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT)
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs.append(completed.stdout)
        self.assertTrue(all(value == outputs[0] for value in outputs[1:]))

    def test_result_contract_rejects_unknown_duplicate_cycle_and_bad_order(self) -> None:
        valid = compute_open_face_unified_compositing(
            _panel_probe_model(),
            projection_matrix=IDENTITY_VIEW,
        )
        items = valid.item_ids
        self.assertGreaterEqual(len(items), 3)

        unknown = OpenFacePaintRelation(
            "missing",
            items[0],
            "unknown",
            0.0,
            0.0,
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "unknown item"):
            replace(valid, order_relations=(unknown,))

        forward = OpenFacePaintRelation(
            items[0], items[1], "forward", 0.0, 0.0, 0.0
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            replace(valid, order_relations=(forward, forward))

        reverse = OpenFacePaintRelation(
            items[1], items[0], "reverse", 0.0, 0.0, 0.0
        )
        with self.assertRaisesRegex(ValueError, "contradictory"):
            replace(valid, order_relations=(forward, reverse))

        cycle = (
            OpenFacePaintRelation(items[0], items[1], "a", 0.0, 0.0, 0.0),
            OpenFacePaintRelation(items[1], items[2], "b", 0.0, 0.0, 0.0),
            OpenFacePaintRelation(items[2], items[0], "c", 0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            replace(valid, order_relations=cycle)

        bad_order = tuple(reversed(valid.draw_order))
        if any(
            bad_order.index(item.far_item_id) >= bad_order.index(item.near_item_id)
            for item in valid.order_relations
        ):
            with self.assertRaisesRegex(ValueError, "contradicts"):
                replace(valid, draw_order=bad_order)

    def test_invalid_limits_fail_before_geometry_work(self) -> None:
        from polyhedron_visibility.open_faces.unified_compositing import (
            OpenFaceUnifiedCompositingLimits,
        )

        with self.assertRaisesRegex(OpenFaceUnifiedCompositingError, "paths"):
            compute_open_face_unified_compositing(
                _crossing_paths_model(),
                projection_matrix=IDENTITY_VIEW,
                limits=OpenFaceUnifiedCompositingLimits(max_paths=1),
            )


if __name__ == "__main__":
    unittest.main()
