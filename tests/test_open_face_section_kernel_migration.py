from __future__ import annotations

import heapq
import random
import unittest
from unittest.mock import patch

import numpy as np

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.open_faces.contract import (
    OPEN_FACE_MODEL_SCHEMA,
    OPEN_FACE_TOPOLOGY,
    OpenFaceVisibilityModel,
)
from polyhedron_visibility.open_faces.solver import (
    _spans_from_intervals as open_face_spans_from_intervals,
    compute_open_face_visibility,
)
from polyhedron_visibility.open_faces.trace import (
    OpenFaceRawOcclusionInterval,
    OpenFaceVisibilitySpan,
)
from polyhedron_visibility.sections.compositing import (
    FragmentOrderRelation,
    TransparentSectionCompositingError,
    TransparentTriangle,
    _topological_draw_order,
    compute_transparent_section_compositing,
)
from polyhedron_visibility.sections.contract import SectionPlane3D
from polyhedron_visibility.sections.solver import (
    _visibility_spans as section_visibility_spans,
    compute_sectioned_visibility,
)
from polyhedron_visibility.trace import RawOcclusionInterval, VisibilitySpan


_IDENTITY_VIEW = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_ISOMETRIC_VIEW = (
    (0.7071067811865476, -0.7071067811865476, 0.0),
    (0.4082482904638631, 0.4082482904638631, 0.8164965809277261),
    (0.5773502691896258, 0.5773502691896258, -0.5773502691896258),
)


def _legacy_open_face_spans(
    intervals: list[OpenFaceRawOcclusionInterval],
    parameter_epsilon: float,
) -> tuple[OpenFaceVisibilitySpan, ...]:
    boundaries = [0.0, 1.0]
    for interval in intervals:
        boundaries.extend((interval.start, interval.end))
    boundaries.sort()
    unique: list[float] = []
    for raw in boundaries:
        value = min(1.0, max(0.0, float(raw)))
        if not unique or abs(value - unique[-1]) > parameter_epsilon:
            unique.append(value)
        else:
            unique[-1] = max(unique[-1], value)
    if unique[0] > 0.0:
        unique.insert(0, 0.0)
    if unique[-1] < 1.0:
        unique.append(1.0)

    spans: list[OpenFaceVisibilitySpan] = []
    for start, end in zip(unique, unique[1:]):
        if end - start <= parameter_epsilon:
            continue
        midpoint = 0.5 * (start + end)
        active = tuple(
            sorted(
                (
                    interval
                    for interval in intervals
                    if interval.start - parameter_epsilon
                    <= midpoint
                    <= interval.end + parameter_epsilon
                ),
                key=lambda item: (item.face_id, item.logical_surface_id),
            )
        )
        face_ids = tuple(item.face_id for item in active)
        surface_ids = tuple(sorted({item.logical_surface_id for item in active}))
        span = OpenFaceVisibilitySpan(
            start,
            end,
            "hidden" if active else "visible",
            face_ids,
            surface_ids,
            len(face_ids),
            len(surface_ids),
        )
        if (
            spans
            and spans[-1].kind == span.kind
            and spans[-1].occluder_face_ids == span.occluder_face_ids
            and spans[-1].occluder_logical_surface_ids
            == span.occluder_logical_surface_ids
            and abs(spans[-1].end - span.start) <= parameter_epsilon
        ):
            previous = spans[-1]
            spans[-1] = OpenFaceVisibilitySpan(
                previous.start,
                span.end,
                previous.kind,
                previous.occluder_face_ids,
                previous.occluder_logical_surface_ids,
                previous.face_level,
                previous.surface_level,
            )
        else:
            spans.append(span)
    if not spans:
        return (OpenFaceVisibilitySpan(0.0, 1.0, "visible"),)
    first = spans[0]
    spans[0] = OpenFaceVisibilitySpan(
        0.0,
        first.end,
        first.kind,
        first.occluder_face_ids,
        first.occluder_logical_surface_ids,
        first.face_level,
        first.surface_level,
    )
    last = spans[-1]
    spans[-1] = OpenFaceVisibilitySpan(
        last.start,
        1.0,
        last.kind,
        last.occluder_face_ids,
        last.occluder_logical_surface_ids,
        last.face_level,
        last.surface_level,
    )
    return tuple(spans)


def _legacy_section_spans(
    intervals: list[RawOcclusionInterval],
    parameter_epsilon: float,
) -> tuple[VisibilitySpan, ...]:
    boundaries = [0.0, 1.0]
    for item in intervals:
        boundaries.extend((item.start, item.end))
    boundaries.sort()
    unique: list[float] = []
    for raw in boundaries:
        value = min(1.0, max(0.0, float(raw)))
        if not unique or abs(value - unique[-1]) > parameter_epsilon:
            unique.append(value)
        else:
            unique[-1] = max(unique[-1], value)
    if not unique:
        unique = [0.0, 1.0]
    if unique[0] > 0.0:
        unique.insert(0, 0.0)
    if unique[-1] < 1.0:
        unique.append(1.0)

    spans: list[VisibilitySpan] = []
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
        span = VisibilitySpan(
            start,
            end,
            "hidden" if active else "visible",
            active,
            len(active),
        )
        if (
            spans
            and spans[-1].kind == span.kind
            and spans[-1].occluder_face_ids == span.occluder_face_ids
            and abs(spans[-1].end - span.start) <= parameter_epsilon
        ):
            previous = spans[-1]
            spans[-1] = VisibilitySpan(
                previous.start,
                span.end,
                previous.kind,
                previous.occluder_face_ids,
                previous.level,
            )
        else:
            spans.append(span)
    if not spans:
        return (VisibilitySpan(0.0, 1.0, "visible", (), 0),)
    first = spans[0]
    last = spans[-1]
    spans[0] = VisibilitySpan(
        0.0,
        first.end,
        first.kind,
        first.occluder_face_ids,
        first.level,
    )
    spans[-1] = VisibilitySpan(
        last.start,
        1.0,
        last.kind,
        last.occluder_face_ids,
        last.level,
    )
    return tuple(spans)


def _legacy_draw_order(
    fragment_ids: set[str],
    relations: list[tuple[str, str]],
) -> tuple[str, ...]:
    outgoing = {item: set() for item in fragment_ids}
    indegree = {item: 0 for item in fragment_ids}
    for far_id, near_id in relations:
        if near_id not in outgoing[far_id]:
            outgoing[far_id].add(near_id)
            indegree[near_id] += 1
    ready = [item for item in fragment_ids if indegree[item] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        current = heapq.heappop(ready)
        order.append(current)
        for target in sorted(outgoing[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if len(order) != len(fragment_ids):
        raise ValueError("cycle")
    return tuple(order)


def _open_face_model() -> OpenFaceVisibilityModel:
    return OpenFaceVisibilityModel.from_dict(
        {
            "schema": OPEN_FACE_MODEL_SCHEMA,
            "topology": OPEN_FACE_TOPOLOGY,
            "visibilityGroupId": "kernel-open-face",
            "vertices": [
                {"vertexId": "L", "entryPosition": [-2.0, 0.0, 0.0]},
                {"vertexId": "R", "entryPosition": [2.0, 0.0, 0.0]},
                {"vertexId": "A", "entryPosition": [-1.0, -1.0, 1.0]},
                {"vertexId": "B", "entryPosition": [1.0, -1.0, 1.0]},
                {"vertexId": "C", "entryPosition": [1.0, 1.0, 1.0]},
                {"vertexId": "D", "entryPosition": [-1.0, 1.0, 1.0]},
            ],
            "faces": [
                {
                    "faceId": "panel",
                    "logicalSurfaceId": "surface-panel",
                    "vertexIds": ["A", "B", "C", "D"],
                }
            ],
            "seams": [],
            "strokes": [
                {
                    "sourceEdgeId": "probe",
                    "vertexIds": ["L", "R"],
                    "incidentFaceIds": [],
                }
            ],
        }
    )


def _cube_model(*, with_probe: bool) -> VisibilityModel:
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
    if with_probe:
        vertices.update({"X": (-2.0, 1.5, -2.0), "Y": (2.0, 1.5, -2.0)})
    faces = {
        "back": ("A", "D", "C", "B"),
        "front": ("E", "F", "G", "H"),
        "bottom": ("A", "B", "F", "E"),
        "right": ("B", "C", "G", "F"),
        "top": ("D", "H", "G", "C"),
        "left": ("A", "E", "H", "D"),
    }
    strokes = []
    if with_probe:
        strokes.append(
            {
                "sourceEdgeId": "probe",
                "vertexIds": ["X", "Y"],
                "incidentFaceIds": [],
            }
        )
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "kernel-section",
            "vertices": [
                {"vertexId": key, "entryPosition": value}
                for key, value in vertices.items()
            ],
            "faces": [
                {"faceId": key, "vertexIds": list(value)}
                for key, value in faces.items()
            ],
            "strokes": strokes,
        }
    )


def _plane() -> SectionPlane3D:
    return SectionPlane3D(
        "cut",
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        3.0,
        3.0,
        u_axis=(1.0, -1.0, 0.0),
    )


def _triangle(fragment_id: str) -> TransparentTriangle:
    return TransparentTriangle(
        fragment_id,
        f"surface-{fragment_id}",
        "test",
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (f"{fragment_id}:a", f"{fragment_id}:b", f"{fragment_id}:c"),
    )


class OpenFaceSectionKernelMigrationTests(unittest.TestCase):
    def test_open_face_dual_identity_survives_shared_partitioning(self) -> None:
        spans = open_face_spans_from_intervals(
            [
                OpenFaceRawOcclusionInterval("face-b", "surface", 0.2, 0.8),
                OpenFaceRawOcclusionInterval("face-a", "surface", 0.3, 0.7),
            ],
            1.0e-9,
        )
        deepest = max(spans, key=lambda span: span.face_level)
        self.assertEqual(deepest.occluder_face_ids, ("face-a", "face-b"))
        self.assertEqual(deepest.occluder_logical_surface_ids, ("surface",))
        self.assertEqual(deepest.face_level, 2)
        self.assertEqual(deepest.surface_level, 1)

    def test_open_face_randomized_partitions_match_the_v1_splitter(self) -> None:
        randomizer = random.Random(20260821)
        for case_index in range(2500):
            epsilon = 10.0 ** randomizer.uniform(-12.0, -5.0)
            interval_count = randomizer.randrange(0, 9)
            intervals: list[OpenFaceRawOcclusionInterval] = []
            for index in range(interval_count):
                start = randomizer.uniform(0.0, 0.93)
                minimum_length = max(4.0 * epsilon, 1.0e-8)
                end = randomizer.uniform(start + minimum_length, 1.0)
                intervals.append(
                    OpenFaceRawOcclusionInterval(
                        f"face-{index:02d}",
                        f"surface-{randomizer.randrange(0, 4):02d}",
                        start,
                        end,
                    )
                )
            intervals.sort(
                key=lambda item: (
                    item.start,
                    item.end,
                    item.face_id,
                    item.logical_surface_id,
                )
            )
            with self.subTest(case=case_index):
                self.assertEqual(
                    open_face_spans_from_intervals(intervals, epsilon),
                    _legacy_open_face_spans(intervals, epsilon),
                )

    def test_section_randomized_partitions_match_the_v1_splitter(self) -> None:
        randomizer = random.Random(20260822)
        for case_index in range(2500):
            epsilon = 10.0 ** randomizer.uniform(-12.0, -5.0)
            interval_count = randomizer.randrange(0, 9)
            intervals: list[RawOcclusionInterval] = []
            for index in range(interval_count):
                start = randomizer.uniform(0.0, 0.93)
                minimum_length = max(4.0 * epsilon, 1.0e-8)
                end = randomizer.uniform(start + minimum_length, 1.0)
                intervals.append(
                    RawOcclusionInterval(f"face-{index:02d}", start, end)
                )
            intervals.sort(key=lambda item: (item.start, item.end, item.face_id))
            with self.subTest(case=case_index):
                self.assertEqual(
                    section_visibility_spans(intervals, epsilon),
                    _legacy_section_spans(intervals, epsilon),
                )

    def test_section_sorter_matches_the_v1_heap_order_for_random_dags(self) -> None:
        randomizer = random.Random(20260823)
        for case_index in range(1500):
            count = randomizer.randrange(1, 13)
            ids = [f"fragment-{index:02d}" for index in range(count)]
            authored = ids[:]
            randomizer.shuffle(authored)
            topological = ids[:]
            randomizer.shuffle(topological)
            position = {item: index for index, item in enumerate(topological)}
            pairs: list[tuple[str, str]] = []
            for first in ids:
                for second in ids:
                    if position[first] >= position[second]:
                        continue
                    if randomizer.random() < 0.18:
                        pairs.append((first, second))
            relations = [
                FragmentOrderRelation(far, near, 1.0, -1.0, -1.0, "test")
                for far, near in pairs
            ]
            fragments = [_triangle(item) for item in authored]
            with self.subTest(case=case_index):
                self.assertEqual(
                    _topological_draw_order(fragments, relations),
                    _legacy_draw_order(set(ids), pairs),
                )

    def test_section_sorter_fails_closed_for_unknown_or_cyclic_relations(self) -> None:
        fragments = [_triangle("a"), _triangle("b")]
        with self.assertRaisesRegex(
            TransparentSectionCompositingError,
            "unknown identities: missing",
        ):
            _topological_draw_order(
                fragments,
                [FragmentOrderRelation("a", "missing", 1.0, -1.0, -1.0, "test")],
            )
        with self.assertRaisesRegex(
            TransparentSectionCompositingError,
            "ordering contains a cycle: a",
        ):
            _topological_draw_order(
                fragments,
                [FragmentOrderRelation("a", "a", 1.0, -1.0, -1.0, "test")],
            )
        with self.assertRaisesRegex(
            TransparentSectionCompositingError,
            "ordering contains a cycle: a, b",
        ):
            _topological_draw_order(
                fragments,
                [
                    FragmentOrderRelation("a", "b", 1.0, -1.0, -1.0, "test"),
                    FragmentOrderRelation("b", "a", 1.0, -1.0, -1.0, "test"),
                ],
            )

    def test_open_face_public_path_reaches_shared_partition_and_sorter(self) -> None:
        import polyhedron_visibility.open_faces.solver as solver

        with (
            patch.object(
                solver,
                "partition_visibility",
                wraps=solver.partition_visibility,
            ) as partition,
            patch.object(
                solver,
                "stable_topological_sort",
                wraps=solver.stable_topological_sort,
            ) as sorter,
        ):
            frame = compute_open_face_visibility(
                _open_face_model(),
                projection_matrix=_IDENTITY_VIEW,
            )
        self.assertTrue(partition.called)
        self.assertTrue(sorter.called)
        self.assertEqual(frame.edge_map["probe"].spans[1].kind, "hidden")

    def test_section_public_paths_reach_shared_partition_and_sorter(self) -> None:
        import polyhedron_visibility.sections.compositing as compositing
        import polyhedron_visibility.sections.solver as solver

        with patch.object(
            solver,
            "partition_visibility",
            wraps=solver.partition_visibility,
        ) as partition:
            visibility = compute_sectioned_visibility(
                _cube_model(with_probe=True),
                SectionPlane3D(
                    "cut",
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    1.0,
                    2.0,
                    u_axis=(1.0, 0.0, 0.0),
                ),
                projection_matrix=np.eye(3),
            )
        self.assertTrue(partition.called)
        self.assertTrue(
            any(
                item.face_id == "section-plane:cut"
                for item in visibility.edge_map["probe"].raw_intervals
            )
        )

        with patch.object(
            compositing,
            "stable_topological_sort",
            wraps=compositing.stable_topological_sort,
        ) as sorter:
            frame = compute_transparent_section_compositing(
                "section",
                _cube_model(with_probe=False),
                _plane(),
                projection_matrix=_ISOMETRIC_VIEW,
            )
        self.assertTrue(sorter.called)
        self.assertEqual(set(frame.draw_order), set(frame.fragment_map))


if __name__ == "__main__":
    unittest.main()
