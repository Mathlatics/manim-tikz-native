from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
from math import cos, pi, sin, sqrt
from pathlib import Path
from typing import Mapping, Sequence
import unittest

import numpy as np

from polyhedron_visibility.geometry import GeometryContext, GeometryQuantity
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.topology import assert_exact_partition
from polyhedron_visibility.quadrics.compositing import (
    QuadricPaintPolicy,
    compute_quadric_compositing,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.projection import build_opaque_projection_proxy
from polyhedron_visibility.quadrics.section_compositing import (
    QUADRIC_SECTION_COMPOSITING_LIMITS,
    PlaneDepthRole,
    compute_quadric_section_compositing,
)
from polyhedron_visibility.quadrics.sections import (
    QuadricSectionBoundary,
    compute_quadric_section_boundary,
)
from polyhedron_visibility.quadrics.visibility import compute_quadric_visibility


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "quadric-section-parameter-sweep-v1.json"
)
with FIXTURE_PATH.open(encoding="utf-8") as source:
    SWEEP = json.load(source)


def _unit(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    return result / np.linalg.norm(result)


def _surface(record: Mapping[str, object]) -> ConeSpec:
    return ConeSpec(
        str(record["id"]),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        float(SWEEP["coneHalfAngleDegrees"]) * pi / 180.0,
        tuple(float(value) for value in record["axialRange"]),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel(str(record["model"])),
    )


def _plane(
    plane_id: str,
    *,
    angle: float,
    height: float,
    azimuth: float = 0.0,
) -> SectionPlane:
    theta = float(angle) * pi / 180.0
    phi = float(azimuth) * pi / 180.0
    normal = _unit(
        (
            sin(theta) * cos(phi),
            sin(theta) * sin(phi),
            cos(theta),
        )
    )
    candidate = (
        np.asarray((0.0, 1.0, 0.0), dtype=float)
        if abs(float(normal[1])) < 0.9
        else np.asarray((1.0, 0.0, 0.0), dtype=float)
    )
    u_axis = _unit(candidate - float(np.dot(candidate, normal)) * normal)
    return SectionPlane(
        plane_id,
        (0.0, 0.0, float(height)),
        tuple(float(value) for value in normal),
        u_axis=tuple(float(value) for value in u_axis),
    )


def _view(record: Mapping[str, object]) -> ParallelView:
    matrix = record.get("matrix")
    if matrix is not None:
        return ParallelView.from_matrix(matrix)
    direction = _unit(record["direction"])
    up = np.asarray(record["up"], dtype=float)
    screen_y = up - float(np.dot(up, direction)) * direction
    if float(np.linalg.norm(screen_y)) <= 1.0e-12:
        raise AssertionError(f"view {record['id']!r} has a singular up vector")
    screen_y = _unit(screen_y)
    screen_x = _unit(np.cross(screen_y, direction))
    return ParallelView.from_matrix((screen_x, screen_y, direction))


SURFACE_RECORDS = {str(item["id"]): item for item in SWEEP["surfaces"]}
VIEW_RECORDS = {str(item["id"]): item for item in SWEEP["views"]}
ISOMETRIC_VIEW = _view(VIEW_RECORDS["isometric"])


def _expected_supporting_kind(angle: float, height: float) -> str:
    if angle == 90.0 or (height == 0.0 and angle > 60.0):
        return "intersecting_lines"
    if height == 0.0:
        if angle == 60.0:
            return "coincident_line"
        return "point"
    if angle == 0.0:
        return "circle"
    if angle < 60.0:
        return "ellipse"
    if angle == 60.0:
        return "parabola"
    return "hyperbola"


def _rounded(value: float) -> float:
    result = round(float(value), 10)
    return 0.0 if result == 0.0 else result


def _trace_semantics(boundary: QuadricSectionBoundary) -> dict[str, object]:
    trace = boundary.trace
    branches = trace.branch_map
    return {
        "kind": trace.supporting_kind.value,
        "topology": trace.finite_topology.value,
        "components": [
            {
                "branch": branches[item.branch_id].parameterization.branch_label,
                "closed": item.closed,
                "intervals": [
                    [_rounded(interval.start), _rounded(interval.end)]
                    for interval in item.parameter_intervals
                ],
            }
            for item in trace.components
        ],
        "points": [
            [_rounded(component) for component in point]
            for point in trace.isolated_world_points
        ],
    }


def _solver_signature(
    boundary: QuadricSectionBoundary,
    visibility: object,
) -> dict[str, object]:
    return {
        **_trace_semantics(boundary),
        "curves": [item.curve_id for item in boundary.curves],
        "capChords": [item.curve_id for item in boundary.cap_chords],
        "visibility": [
            {
                "curve": record.curve_id,
                "spans": [
                    [
                        _rounded(span.interval.start),
                        _rounded(span.interval.end),
                        span.kind.value,
                        list(span.occluders),
                    ]
                    for span in record.spans
                ],
            }
            for record in visibility.records
        ],
    }


def _expected_cap_chord_count(surface: ConeSpec, plane: SectionPlane) -> int:
    normal = np.asarray(plane.normal, dtype=float)
    point = np.asarray(plane.point, dtype=float)
    count = 0
    for cap in surface.end_caps:
        cap_normal = np.asarray(cap.normal, dtype=float)
        center = np.asarray(cap.center, dtype=float)
        in_cap_gradient = normal - float(np.dot(normal, cap_normal)) * cap_normal
        gradient_length = float(np.linalg.norm(in_cap_gradient))
        if gradient_length <= 1.0e-12:
            continue
        distance = abs(float(np.dot(normal, center - point))) / gradient_length
        if distance < cap.radius - 1.0e-9:
            count += 1
    return count


def _physical_component_endpoints(
    component: object,
    branch: object,
) -> tuple[float, ...]:
    if component.closed:
        return ()
    intervals = component.parameter_intervals
    natural = branch.parameterization.natural_domain
    if (
        natural is not None
        and branch.parameterization.closed
        and len(intervals) == 2
        and abs(intervals[0].start - natural.start) <= 1.0e-9
        and abs(intervals[-1].end - natural.end) <= 1.0e-9
    ):
        np.testing.assert_allclose(
            branch.world_point(intervals[0].start),
            branch.world_point(intervals[-1].end),
            rtol=0.0,
            atol=1.0e-8,
        )
        return intervals[0].end, intervals[-1].start
    return tuple(
        parameter
        for interval in intervals
        for parameter in (interval.start, interval.end)
    )


def _assert_boundary_geometry(
    test: unittest.TestCase,
    surface: ConeSpec,
    plane: SectionPlane,
    boundary: QuadricSectionBoundary,
) -> None:
    trace = boundary.trace
    branch_ids = tuple(item.branch_id for item in trace.branches)
    component_ids = tuple(item.component_id for item in trace.components)
    test.assertEqual(branch_ids, tuple(sorted(set(branch_ids))))
    test.assertEqual(component_ids, tuple(sorted(set(component_ids))))
    bounds = surface.axial_range
    apex = np.asarray(surface.apex, dtype=float)
    axis = np.asarray(surface.axis, dtype=float)
    context = GeometryContext().resolve(
        (*surface.characteristic_points, plane.point)
    )
    boundary_epsilon = max(
        1.0e-8,
        32.0 * context.epsilon(GeometryQuantity.BOUNDARY),
    )
    for component in trace.components:
        branch = trace.branch_map[component.branch_id]
        intervals = component.parameter_intervals
        test.assertEqual(intervals, tuple(sorted(intervals)))
        for left, right in zip(intervals, intervals[1:]):
            test.assertLessEqual(left.end, right.start)
        natural = branch.parameterization.natural_domain
        if natural is not None:
            for interval in intervals:
                test.assertGreaterEqual(
                    interval.start,
                    natural.start - boundary_epsilon,
                )
                test.assertLessEqual(
                    interval.end,
                    natural.end + boundary_epsilon,
                )
        for parameter in _physical_component_endpoints(component, branch):
            point = branch.world_point(parameter)
            axial = float(np.dot(point - apex, axis))
            test.assertLessEqual(
                min(abs(axial - bound) for bound in bounds),
                boundary_epsilon,
                f"{component.component_id} endpoint is not on a real trim/cap boundary",
            )
    for point in trace.isolated_world_points:
        axial = float(np.dot(np.asarray(point, dtype=float) - apex, axis))
        test.assertGreaterEqual(axial, bounds[0] - boundary_epsilon)
        test.assertLessEqual(axial, bounds[1] + boundary_epsilon)

    test.assertEqual(
        len(boundary.cap_chords),
        _expected_cap_chord_count(surface, plane),
    )
    if surface.model is ConeModel.OPEN_SINGLE:
        test.assertEqual(boundary.cap_chords, ())
    for chord in boundary.cap_chords:
        test.assertIsInstance(chord, SegmentCurve)
        for point in (np.asarray(chord.start), np.asarray(chord.end)):
            test.assertLessEqual(abs(plane.signed_distance(point)), boundary_epsilon)
            axial = float(np.dot(point - apex, axis))
            test.assertLessEqual(
                min(abs(axial - bound) for bound in bounds),
                boundary_epsilon,
            )
            matching_cap = min(
                surface.end_caps,
                key=lambda cap: abs(
                    float(np.dot(point - np.asarray(cap.center), axis))
                ),
            )
            radial = point - np.asarray(matching_cap.center)
            radial -= float(np.dot(radial, axis)) * axis
            test.assertLessEqual(
                abs(float(np.linalg.norm(radial)) - matching_cap.radius),
                boundary_epsilon,
            )


def _triangle_area_3d(points: Sequence[Sequence[float]]) -> float:
    values = np.asarray(points, dtype=float)
    return 0.5 * float(
        np.linalg.norm(np.cross(values[1] - values[0], values[2] - values[0]))
    )


def _signed_area_2d(points: Sequence[Sequence[float]]) -> float:
    values = np.asarray(points, dtype=float)
    shifted = np.roll(values, -1, axis=0)
    return 0.5 * float(
        np.sum(values[:, 0] * shifted[:, 1] - values[:, 1] * shifted[:, 0])
    )


def _counter_clockwise(points: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    return values if _signed_area_2d(values) >= 0.0 else values[::-1].copy()


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _clip_convex_polygon(
    subject: np.ndarray,
    clipper: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    output = list(_counter_clockwise(subject))
    boundary = _counter_clockwise(clipper)
    for edge_index, edge_start in enumerate(boundary):
        edge_end = boundary[(edge_index + 1) % len(boundary)]
        direction = edge_end - edge_start
        previous_values = output
        output = []
        if not previous_values:
            break
        previous = previous_values[-1]
        previous_side = _cross2(direction, previous - edge_start)
        previous_inside = previous_side >= -epsilon
        for current in previous_values:
            current_side = _cross2(direction, current - edge_start)
            current_inside = current_side >= -epsilon
            if current_inside != previous_inside:
                denominator = previous_side - current_side
                if abs(denominator) > np.finfo(float).eps:
                    output.append(
                        previous
                        + (previous_side / denominator) * (current - previous)
                    )
            if current_inside:
                output.append(current)
            previous = current
            previous_side = current_side
            previous_inside = current_inside
    return np.asarray(output, dtype=float)


def _positive_triangle_overlap(
    fragments: Sequence[object],
    *,
    linear_tolerance: float,
    area_tolerance: float,
) -> tuple[str, ...]:
    records = []
    for fragment in fragments:
        triangle = _counter_clockwise(fragment.screen_vertices)
        records.append(
            (
                float(np.min(triangle[:, 0])),
                float(np.max(triangle[:, 0])),
                float(np.min(triangle[:, 1])),
                float(np.max(triangle[:, 1])),
                fragment.fragment_id,
                triangle,
            )
        )
    records.sort(key=lambda item: item[4])
    if not records:
        return ()
    minimum_x = min(item[0] for item in records)
    maximum_x = max(item[1] for item in records)
    minimum_y = min(item[2] for item in records)
    maximum_y = max(item[3] for item in records)
    resolution = min(256, max(16, int(sqrt(len(records)))))
    cell_width = max((maximum_x - minimum_x) / resolution, linear_tolerance)
    cell_height = max((maximum_y - minimum_y) / resolution, linear_tolerance)
    grid: dict[tuple[int, int], list[int]] = {}
    failures: list[str] = []

    def sat_maybe_overlaps(
        subject: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        subject_edges = np.roll(subject, -1, axis=0) - subject
        subject_axes = np.column_stack((-subject_edges[:, 1], subject_edges[:, 0]))
        subject_axes /= np.linalg.norm(subject_axes, axis=1)[:, None]
        subject_projection = subject @ subject_axes.T
        candidate_projection = np.einsum(
            "kvc,ac->kva",
            candidates,
            subject_axes,
        )
        subject_overlap = (
            np.minimum(
                np.max(subject_projection, axis=0)[None, :],
                np.max(candidate_projection, axis=1),
            )
            - np.maximum(
                np.min(subject_projection, axis=0)[None, :],
                np.min(candidate_projection, axis=1),
            )
        )
        possible = np.all(subject_overlap >= -linear_tolerance, axis=1)
        if not np.any(possible):
            return possible
        candidate_edges = np.roll(candidates, -1, axis=1) - candidates
        candidate_axes = np.stack(
            (-candidate_edges[:, :, 1], candidate_edges[:, :, 0]),
            axis=2,
        )
        candidate_axes /= np.linalg.norm(candidate_axes, axis=2)[:, :, None]
        candidate_self_projection = np.einsum(
            "kvc,kac->kva",
            candidates,
            candidate_axes,
        )
        subject_on_candidate_axes = np.einsum(
            "vc,kac->kva",
            subject,
            candidate_axes,
        )
        candidate_overlap = (
            np.minimum(
                np.max(candidate_self_projection, axis=1),
                np.max(subject_on_candidate_axes, axis=1),
            )
            - np.maximum(
                np.min(candidate_self_projection, axis=1),
                np.min(subject_on_candidate_axes, axis=1),
            )
        )
        return possible & np.all(candidate_overlap >= -linear_tolerance, axis=1)

    for record_index, record in enumerate(records):
        first_x = int(
            np.floor((record[0] - linear_tolerance - minimum_x) / cell_width)
        )
        last_x = int(
            np.floor((record[1] + linear_tolerance - minimum_x) / cell_width)
        )
        first_y = int(
            np.floor((record[2] - linear_tolerance - minimum_y) / cell_height)
        )
        last_y = int(
            np.floor((record[3] + linear_tolerance - minimum_y) / cell_height)
        )
        cells = tuple(
            (cell_x, cell_y)
            for cell_x in range(first_x, last_x + 1)
            for cell_y in range(first_y, last_y + 1)
        )
        candidate_indices = sorted(
            {
                candidate_index
                for cell in cells
                for candidate_index in grid.get(cell, ())
            }
        )
        candidate_indices = [
            candidate_index
            for candidate_index in candidate_indices
            if not (
                records[candidate_index][1] < record[0] - linear_tolerance
                or record[1] < records[candidate_index][0] - linear_tolerance
                or records[candidate_index][3] < record[2] - linear_tolerance
                or record[3] < records[candidate_index][2] - linear_tolerance
            )
        ]
        if candidate_indices:
            candidate_polygons = np.asarray(
                [records[index][5] for index in candidate_indices],
                dtype=float,
            )
            mask = sat_maybe_overlaps(record[5], candidate_polygons)
            candidate_indices = [
                candidate_index
                for candidate_index, possible in zip(candidate_indices, mask)
                if possible
            ]
        for candidate_index in candidate_indices:
            candidate = records[candidate_index]
            intersection = _clip_convex_polygon(
                record[5],
                candidate[5],
                linear_tolerance,
            )
            overlap = (
                0.0
                if len(intersection) < 3
                else abs(_signed_area_2d(intersection))
            )
            if overlap > area_tolerance:
                failures.append(
                    f"{candidate[4]} overlaps {record[4]} by {overlap:.12g}"
                )
                if len(failures) >= 8:
                    return tuple(failures)
        for cell in cells:
            grid.setdefault(cell, []).append(record_index)
    return tuple(failures)


def _compositor_signature(frame: object) -> dict[str, object]:
    return {
        "fragments": [
            [item.fragment_id, item.role.value, item.subdivision_depth]
            for item in frame.plane_fragments
        ],
        "outline": [
            [item.fragment_id, item.role.value]
            for item in frame.plane_outline_fragments
        ],
        "relations": [
            [item.far_item_id, item.near_item_id, item.reason]
            for item in frame.order_relations
        ],
        "drawOrder": list(frame.draw_order),
    }


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _compositor_frame(record: Mapping[str, object]) -> object:
    surface = _surface(SURFACE_RECORDS[str(record["surface"])])
    plane = _plane(
        f"sweep:{record['id']}:plane",
        angle=float(record["angle"]),
        azimuth=float(record.get("azimuth", 0.0)),
        height=float(record["height"]),
    )
    view = _view(VIEW_RECORDS[str(record["view"])])
    boundary = compute_quadric_section_boundary(
        f"sweep:{record['id']}:section",
        surface,
        plane,
    )
    proxy = build_opaque_projection_proxy(
        surface,
        view,
        max_chord_error=0.01,
    )
    visibility = compute_quadric_visibility(boundary.curves, (surface,), view)
    base = compute_quadric_compositing(
        visibility,
        (proxy,),
        paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
    )
    patch = fit_plane_display_patch(
        f"sweep:{record['id']}:patch",
        plane,
        (surface,),
        margin_ratio=0.08,
    ).patch
    return compute_quadric_section_compositing(
        base,
        surface,
        plane,
        patch,
        view,
        max_screen_error=0.08,
    )


class DeterministicConeSectionSolverSweepTests(unittest.TestCase):
    def test_fixed_solver_grid_preserves_geometry_and_visibility_invariants(
        self,
    ) -> None:
        signatures: dict[str, object] = {}
        lateral_semantics: dict[tuple[str, float, str], dict[str, object]] = {}
        for surface_record, angle, position in product(
            SWEEP["surfaces"],
            SWEEP["planeNormalAnglesDegrees"],
            SWEEP["planePositions"],
        ):
            surface = _surface(surface_record)
            height = float(position["height"])
            case_id = (
                f"{surface.surface_id}:{float(angle):g}:"
                f"{position['id']}"
            )
            with self.subTest(case=case_id):
                plane = _plane(
                    f"sweep:{case_id}:plane",
                    angle=float(angle),
                    height=height,
                )
                boundary = compute_quadric_section_boundary(
                    f"sweep:{case_id}:section",
                    surface,
                    plane,
                )
                repeated = compute_quadric_section_boundary(
                    f"sweep:{case_id}:section",
                    surface,
                    plane,
                )
                self.assertEqual(boundary, repeated)
                self.assertEqual(
                    boundary.trace.supporting_kind.value,
                    _expected_supporting_kind(float(angle), height),
                )
                _assert_boundary_geometry(self, surface, plane, boundary)

                visibility = compute_quadric_visibility(
                    boundary.curves,
                    (surface,),
                    ISOMETRIC_VIEW,
                )
                self.assertEqual(
                    tuple(record.curve_id for record in visibility.records),
                    tuple(curve.curve_id for curve in boundary.curves),
                )
                for record in visibility.records:
                    assert_exact_partition(
                        record.domain,
                        (span.interval for span in record.spans),
                        tolerance=record.parameter_tolerance,
                    )
                    self.assertEqual(record.spans[0].interval.start, record.domain.start)
                    self.assertEqual(record.spans[-1].interval.end, record.domain.end)
                    for left, right in zip(record.spans, record.spans[1:]):
                        self.assertEqual(left.interval.end, right.interval.start)

                signatures[case_id] = _solver_signature(boundary, visibility)
                range_id = "cone" if surface.axial_range[0] == 0.0 else "frustum"
                lateral_key = (range_id, float(angle), str(position["id"]))
                semantic = _trace_semantics(boundary)
                previous = lateral_semantics.get(lateral_key)
                if previous is None:
                    lateral_semantics[lateral_key] = semantic
                else:
                    self.assertEqual(
                        semantic,
                        previous,
                        "open/closed semantics may change caps, not the lateral trace",
                    )

        self.assertEqual(len(signatures), 252)
        actual_digest = _digest(signatures)
        self.assertEqual(
            actual_digest,
            SWEEP["expectedSolverSemanticDigest"],
            f"update the reviewed solver baseline only if intentional: {actual_digest}",
        )


class DeterministicConeSectionCompositorSweepTests(unittest.TestCase):
    def test_pairwise_compositor_matrix_preserves_partition_and_painter_invariants(
        self,
    ) -> None:
        actual_digests: dict[str, str] = {}
        for record in SWEEP["compositorCases"]:
            case_id = str(record["id"])
            with self.subTest(case=case_id):
                frame = _compositor_frame(record)
                signature = _compositor_signature(frame)
                repeated_signature = _compositor_signature(
                    _compositor_frame(record)
                )
                self.assertEqual(signature, repeated_signature)
                actual_digests[case_id] = _digest(signature)

                fragment_ids = tuple(
                    item.fragment_id for item in frame.plane_fragments
                )
                self.assertEqual(fragment_ids, tuple(sorted(set(fragment_ids))))
                outline_ids = tuple(
                    item.fragment_id for item in frame.plane_outline_fragments
                )
                self.assertEqual(outline_ids, tuple(sorted(set(outline_ids))))

                patch_area = 4.0 * frame.patch.half_width * frame.patch.half_height
                areas_by_role = {
                    role: sum(
                        _triangle_area_3d(item.world_vertices)
                        for item in frame.fragments_by_role[role]
                    )
                    for role in PlaneDepthRole
                }
                restored_area = sum(areas_by_role.values())
                self.assertAlmostEqual(restored_area, patch_area, places=8)
                self.assertTrue(
                    all(
                        _triangle_area_3d(item.world_vertices) > 1.0e-14
                        for item in frame.plane_fragments
                    )
                )

                center_u, center_v = frame.patch.center_coordinates
                for fragment in frame.plane_fragments:
                    for point in fragment.world_vertices:
                        u_value, v_value = frame.plane.coordinates_in_plane(point)
                        self.assertLessEqual(
                            abs(u_value - center_u),
                            frame.patch.half_width + 1.0e-8,
                        )
                        self.assertLessEqual(
                            abs(v_value - center_v),
                            frame.patch.half_height + 1.0e-8,
                        )

                screen_points = np.asarray(
                    [
                        point
                        for fragment in frame.plane_fragments
                        for point in fragment.screen_vertices
                    ],
                    dtype=float,
                )
                screen_scale = max(
                    1.0,
                    float(np.max(np.abs(screen_points))),
                )
                overlap_failures = _positive_triangle_overlap(
                    frame.plane_fragments,
                    linear_tolerance=1.0e-10 * screen_scale,
                    area_tolerance=max(1.0e-11, patch_area * 1.0e-10),
                )
                self.assertEqual(overlap_failures, ())

                active_curve_ids = {
                    item.item_id
                    for item in frame.base_frame.curve_fragments
                    if item.painted
                }
                expected_paint_ids = {
                    *frame.paint_items.ordered,
                    *active_curve_ids,
                }
                self.assertEqual(set(frame.draw_order), expected_paint_ids)
                rank = {
                    item_id: index for index, item_id in enumerate(frame.draw_order)
                }
                self.assertTrue(
                    all(
                        rank[item.far_item_id] < rank[item.near_item_id]
                        for item in frame.order_relations
                    )
                )
                self.assertLessEqual(
                    len(frame.plane_fragments),
                    QUADRIC_SECTION_COMPOSITING_LIMITS.max_plane_fragments,
                )
                self.assertLessEqual(
                    frame.ray_classification_count,
                    QUADRIC_SECTION_COMPOSITING_LIMITS.max_ray_classifications,
                )

        self.assertEqual(len(actual_digests), len(SWEEP["compositorCases"]))
        self.assertEqual(
            actual_digests,
            SWEEP["expectedCompositorDigests"],
            "update reviewed compositor baselines only for an intentional change: "
            + json.dumps(actual_digests, sort_keys=True, indent=2),
        )


if __name__ == "__main__":
    unittest.main()
