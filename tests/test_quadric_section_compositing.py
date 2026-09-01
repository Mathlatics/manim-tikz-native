from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import atan, cos, pi, sin, sqrt
from typing import Sequence
import unittest
from unittest.mock import patch

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.geometry import GeometryContext, GeometryQuantity
from polyhedron_visibility.topology import ParameterInterval
from polyhedron_visibility.quadrics.compositing import (
    QuadricPaintKind,
    QuadricPaintPolicy,
    compute_quadric_compositing,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
    CylinderSpec,
)
from polyhedron_visibility.quadrics.conics import ConicKind, ConicParameterization
from polyhedron_visibility.quadrics.curves import (
    CircleArcCurve,
    ParametricConicBranch,
)
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.plane_motion import AxisAnglePlaneMotion
from polyhedron_visibility.quadrics.projection import build_opaque_projection_proxy
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionKind,
    QUADRIC_SECTION_COMPOSITING_LIMITS,
    QuadricPlaneFragment,
    QuadricSectionCompositingError,
    QuadricSectionCompositingLimits,
    canonical_quadric_section_compositing_json,
    compute_quadric_section_compositing,
    quadric_plane_fragment_contours,
    repaint_quadric_section_compositing,
    _adaptive_section_curve_parameters,
    _CanonicalVertexRegistry,
    _point_segment_distance_2d,
    _plane_patch_projection_evidence,
    _SECTION_BOUNDARY_CHORD_DIVISOR,
    _section_curve_tangent_envelope_error_bound,
    _make_plane_partition_polygon,
    _nested_convex_ring_ray_vertex,
    _nested_convex_ring_polygons,
    _PlanePartitionPolygon,
    _partition_triangle_by_convex_proxy,
    _plane_partition_polygon_contours,
    _split_convex_polygon_by_half_plane,
    _surface_ray_solver,
    _triangulate_plane_partition_polygon,
)
from polyhedron_visibility.quadrics.sections import compute_quadric_section
from polyhedron_visibility.quadrics.visibility import compute_quadric_visibility


IDENTITY_VIEW = ParallelView.from_matrix(np.eye(3))
OBLIQUE_VIEW = ParallelView.from_matrix(
    (
        (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
        (-1.0 / sqrt(6.0), -1.0 / sqrt(6.0), 2.0 / sqrt(6.0)),
        (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    )
)
TRANSITION_TANGENT_VIEW = ParallelView.from_matrix(
    (
        (
            -0.85 / sqrt(2.0),
            0.85 / sqrt(2.0),
            0.0,
        ),
        (
            -0.85 / sqrt(6.0),
            -0.85 / sqrt(6.0),
            1.70 / sqrt(6.0),
        ),
        (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    )
)


QuadricSurface = SphereSpec | CylinderSpec | ConeSpec


@dataclass(frozen=True, slots=True)
class _SectionPartitionCase:
    name: str
    surface: QuadricSurface
    plane: SectionPlane
    expected_kind: str
    expected_topology: str


def _unit(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    return vector / np.linalg.norm(vector)


def _cone_plane_normal(theta: float) -> np.ndarray:
    return _unit((sin(theta), 0.0, cos(theta)))


def _section_partition_cases() -> tuple[_SectionPartitionCase, ...]:
    sphere = SphereSpec("contract-sphere", (0.0, 0.0, 0.0), 1.0)
    cylinder = CylinderSpec(
        "contract-cylinder",
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        0.9,
        (-0.4, 2.1),
        radial_axis=(1.0, 0.0, 0.0),
    )
    apex = np.asarray((0.0, 0.0, -2.25), dtype=float)
    world_z = np.asarray((0.0, 0.0, 1.0), dtype=float)
    cone = ConeSpec(
        "contract-cone",
        tuple(float(value) for value in apex),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.1),
        radial_axis=(1.0, 0.0, 0.0),
    )
    view_direction = np.asarray(OBLIQUE_VIEW.view_direction, dtype=float)
    parabola_normal = _cone_plane_normal(pi / 3.0)
    return (
        _SectionPartitionCase(
            "sphere_oblique",
            sphere,
            SectionPlane(
                "sphere-oblique-plane",
                (0.0, 0.0, 0.0),
                (0.7, 0.0, 1.0),
                u_axis=(0.0, 1.0, 0.0),
            ),
            "circle",
            "closed_curve",
        ),
        _SectionPartitionCase(
            "sphere_near_tangent",
            sphere,
            SectionPlane(
                "sphere-near-tangent-plane",
                (0.0, 0.0, 0.94),
                (0.0, 0.0, 1.0),
                u_axis=(1.0, 0.0, 0.0),
            ),
            "circle",
            "closed_curve",
        ),
        _SectionPartitionCase(
            "cylinder_side_section",
            cylinder,
            SectionPlane(
                "cylinder-side-plane",
                (0.0, 0.0, 0.0),
                (0.55, 0.0, 1.0),
                u_axis=(0.0, 1.0, 0.0),
            ),
            "ellipse",
            "closed_curve",
        ),
        _SectionPartitionCase(
            "cylinder_through_caps",
            cylinder,
            SectionPlane(
                "cylinder-caps-plane",
                (0.0, 0.0, -0.15),
                (1.0, 0.0, 0.12),
                u_axis=(0.0, 1.0, 0.0),
            ),
            "ellipse",
            "multiple_open_curves",
        ),
        _SectionPartitionCase(
            "cone_ellipse",
            cone,
            SectionPlane(
                "cone-ellipse-plane",
                tuple(
                    float(value)
                    for value in apex
                    + 1.45 * world_z
                    - 1.05 * view_direction
                ),
                tuple(float(value) for value in _cone_plane_normal(0.80)),
                u_axis=(0.0, 1.0, 0.0),
            ),
            "ellipse",
            "closed_curve",
        ),
        _SectionPartitionCase(
            "cone_near_tangent",
            cone,
            SectionPlane(
                "cone-near-tangent-plane",
                tuple(float(value) for value in apex + 0.10 * parabola_normal),
                tuple(float(value) for value in parabola_normal),
                u_axis=(0.0, 1.0, 0.0),
            ),
            "parabola",
            "open_curve",
        ),
        _SectionPartitionCase(
            "cone_exact_parabola",
            cone,
            SectionPlane(
                "cone-exact-parabola-plane",
                tuple(
                    float(value)
                    for value in apex
                    + 0.82 * parabola_normal
                    + 0.18 * view_direction
                ),
                tuple(float(value) for value in parabola_normal),
                u_axis=(0.0, 1.0, 0.0),
            ),
            "parabola",
            "open_curve",
        ),
        _SectionPartitionCase(
            "cone_hyperbola_like_finite_branch",
            cone,
            SectionPlane(
                "cone-hyperbola-plane",
                tuple(
                    float(value)
                    for value in apex
                    + 1.45 * world_z
                    + 1.05 * view_direction
                ),
                tuple(float(value) for value in _cone_plane_normal(1.20)),
                u_axis=(0.0, 1.0, 0.0),
            ),
            "hyperbola",
            "open_curve",
        ),
        _SectionPartitionCase(
            "cone_through_finite_cap",
            cone,
            SectionPlane(
                "cone-cap-plane",
                tuple(float(value) for value in apex + 3.70 * world_z),
                tuple(float(value) for value in _unit((0.75, 0.0, 1.0))),
                u_axis=(0.0, 1.0, 0.0),
            ),
            "ellipse",
            "open_curve",
        ),
    )


SECTION_PARTITION_CASES = _section_partition_cases()


_FROZEN_OUTLINE_DIGESTS = {
    "sphere_oblique": "58b888af2387baf7c7e7daad74f460adfa00a46f55fc3e2dea7bcbb76aefca44",
    "sphere_near_tangent": "0ae95e2aea8199c88fd7ce02dc49e67ea053544c8710cb5a40e018e1b1d30032",
    "cylinder_side_section": "092e27177b1eed3d2c8010ee0b8d9af2b8f5d08bf0df1d6800d43747b94d4638",
    "cylinder_through_caps": "ed93dd2280a5c0bb5d5d6c69863c53db6507fa3c9efae9c4efc381858fc8b903",
    "cone_ellipse": "ea54aa779dccb156b384750aba8c37e4d09007c1f026e68412a74fa2d9ea8e83",
    "cone_near_tangent": "3f8d7d1ffe5d0cf3736c76ca5323eb2d1de2c7a778d66c383936c5eed730abaf",
    "cone_exact_parabola": "2c6b4399737d5a1dcf315ab691b46b08ef12725f2594cb57b1eb4d02ebe1e204",
    "cone_hyperbola_like_finite_branch": "fe1ebf3daae5502adaa6b11ac83cf8609c7f7307043316bd451c23cba94bfeda",
    "cone_through_finite_cap": "4ac98ee7d82a8fa31151de2bcea22e4ffc8589960501ed5e90a82e1239e8ced2",
}


def _base_frame(
    surface: QuadricSurface,
    curves: tuple[CircleArcCurve, ...] = (),
    *,
    paint_policy: QuadricPaintPolicy | str = QuadricPaintPolicy.DIAGRAMMATIC,
):
    proxy = build_opaque_projection_proxy(
        surface,
        IDENTITY_VIEW,
        max_chord_error=0.01,
    )
    visibility = compute_quadric_visibility(
        curves,
        (surface,),
        IDENTITY_VIEW,
    )
    return compute_quadric_compositing(
        visibility,
        (proxy,),
        paint_policy=paint_policy,
    )


def _triangle_area(vertices: tuple[tuple[float, float, float], ...]) -> float:
    values = np.asarray(vertices, dtype=float)
    return 0.5 * float(
        np.linalg.norm(np.cross(values[1] - values[0], values[2] - values[0]))
    )


def _screen_signed_area(vertices) -> float:
    values = np.asarray(vertices, dtype=float)
    return 0.5 * sum(
        values[index, 0] * values[(index + 1) % len(values), 1]
        - values[index, 1] * values[(index + 1) % len(values), 0]
        for index in range(len(values))
    )


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _polygon_area(vertices: Sequence[Sequence[float]]) -> float:
    values = tuple(np.asarray(point, dtype=float) for point in vertices)
    if len(values) < 3:
        return 0.0
    return abs(
        0.5
        * sum(
            _cross2(values[index], values[(index + 1) % len(values)])
            for index in range(len(values))
        )
    )


def _counter_clockwise(
    vertices: Sequence[Sequence[float]],
) -> tuple[np.ndarray, ...]:
    result = tuple(np.asarray(point, dtype=float) for point in vertices)
    if _screen_signed_area(result) < 0.0:
        return tuple(reversed(result))
    return result


def _clip_convex_polygon_for_contract(
    subject: Sequence[Sequence[float]],
    clipper: Sequence[Sequence[float]],
    epsilon: float,
) -> tuple[np.ndarray, ...]:
    """Independent Sutherland-Hodgman clipper used only by the contract test."""

    output = list(_counter_clockwise(subject))
    boundary = _counter_clockwise(clipper)
    for edge_index, edge_start in enumerate(boundary):
        edge_end = boundary[(edge_index + 1) % len(boundary)]
        direction = edge_end - edge_start
        side_tolerance = epsilon * max(
            float(np.linalg.norm(direction)),
            np.finfo(float).tiny,
        )
        previous_values = output
        output = []
        if not previous_values:
            break
        previous = previous_values[-1]
        previous_side = _cross2(direction, previous - edge_start)
        previous_inside = previous_side >= -side_tolerance
        for current in previous_values:
            current_side = _cross2(direction, current - edge_start)
            current_inside = current_side >= -side_tolerance
            if current_inside != previous_inside:
                denominator = previous_side - current_side
                if abs(denominator) > np.finfo(float).eps:
                    ratio = previous_side / denominator
                    output.append(previous + ratio * (current - previous))
            if current_inside:
                output.append(current)
            previous = current
            previous_side = current_side
            previous_inside = current_inside
    deduped: list[np.ndarray] = []
    for point in output:
        if not deduped or float(np.linalg.norm(point - deduped[-1])) > epsilon:
            deduped.append(point)
    if (
        len(deduped) > 1
        and float(np.linalg.norm(deduped[0] - deduped[-1])) <= epsilon
    ):
        deduped.pop()
    return tuple(deduped)


def _project_patch(frame) -> tuple[np.ndarray, ...]:
    matrix = np.asarray(frame.base_frame.visibility.projection_matrix, dtype=float)
    return _counter_clockwise(
        tuple(
            matrix[:2] @ np.asarray(point, dtype=float)
            for point in frame.patch.corners(frame.plane)
        )
    )


def _contract_frame(
    case: _SectionPartitionCase,
    *,
    max_screen_error: float = 0.08,
):
    proxy = build_opaque_projection_proxy(
        case.surface,
        OBLIQUE_VIEW,
        max_chord_error=0.01,
    )
    visibility = compute_quadric_visibility(
        (),
        (case.surface,),
        OBLIQUE_VIEW,
    )
    base = compute_quadric_compositing(visibility, (proxy,))
    patch = fit_plane_display_patch(
        f"{case.name}-patch",
        case.plane,
        (case.surface,),
        margin_ratio=0.1,
    ).patch
    return compute_quadric_section_compositing(
        base,
        case.surface,
        case.plane,
        patch,
        OBLIQUE_VIEW,
        max_screen_error=max_screen_error,
    )


def _transition_tangent_regression_frame():
    """Rebuild the exact 30 fps demo frame that once split one role run."""

    vertical_shift = -0.9
    cone = ConeSpec(
        "transition-cone",
        (0.0, 0.0, -1.5 + vertical_shift),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    motion = AxisAnglePlaneMotion(
        "transition-plane-motion",
        SectionPlane(
            "transition-plane",
            (0.0, 0.0, 0.2 + vertical_shift),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        (0.0, 0.0, 0.2 + vertical_shift),
        (0.0, 1.0, 0.0),
        0.72,
        1.35,
    )
    # Manim smooth(115 / 180): the first formal 30 fps render found a
    # microscopic tangent-neighborhood cell at precisely this progress.
    progress = 0.8044906220284254
    plane = motion.plane_at(progress)
    proxy = build_opaque_projection_proxy(
        cone,
        TRANSITION_TANGENT_VIEW,
        max_chord_error=0.008,
    )
    visibility = compute_quadric_visibility(
        (),
        (cone,),
        TRANSITION_TANGENT_VIEW,
    )
    base = compute_quadric_compositing(
        visibility,
        (proxy,),
        paint_policy=QuadricPaintPolicy.DIAGRAMMATIC,
    )
    patch = fit_plane_display_patch(
        "transition-plane:auto-display-patch",
        plane,
        (cone,),
        margin_ratio=0.08,
    ).patch
    return compute_quadric_section_compositing(
        base,
        cone,
        plane,
        patch,
        TRANSITION_TANGENT_VIEW,
    )


def _case_tolerances(
    case: _SectionPartitionCase,
    frame,
) -> tuple[float, float, float]:
    characteristic = tuple(case.surface.characteristic_points) + tuple(
        frame.patch.corners(frame.plane)
    )
    context = GeometryContext().resolve(characteristic)
    screen = 8.0 * context.epsilon(GeometryQuantity.SCREEN)
    boundary = 8.0 * context.epsilon(GeometryQuantity.BOUNDARY)
    patch_area = _polygon_area(_project_patch(frame))
    area = max(patch_area * 1.0e-9, screen * screen, 1.0e-12)
    return screen, boundary, area


def _positive_overlap_issues(
    fragments: Sequence,
    *,
    linear_tolerance: float,
    area_tolerance: float,
) -> tuple[str, ...]:
    records = []
    for fragment in fragments:
        polygon = _counter_clockwise(fragment.screen_vertices)
        values = np.asarray(polygon, dtype=float)
        records.append(
            (
                float(np.min(values[:, 0])),
                float(np.max(values[:, 0])),
                float(np.min(values[:, 1])),
                float(np.max(values[:, 1])),
                fragment.fragment_id,
                polygon,
            )
        )
    records.sort(key=lambda item: item[4])
    minimum_x = min(record[0] for record in records)
    maximum_x = max(record[1] for record in records)
    minimum_y = min(record[2] for record in records)
    maximum_y = max(record[3] for record in records)
    grid_resolution = min(256, max(16, int(sqrt(len(records)))))
    cell_width = max(
        (maximum_x - minimum_x) / grid_resolution,
        linear_tolerance,
    )
    cell_height = max(
        (maximum_y - minimum_y) / grid_resolution,
        linear_tolerance,
    )
    grid: dict[tuple[int, int], list[int]] = {}
    issues: list[str] = []

    def sat_maybe_overlaps(
        subject: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """Vectorized conservative SAT filter before exact polygon clipping."""

        subject_edges = np.roll(subject, -1, axis=0) - subject
        subject_axes = np.column_stack(
            (-subject_edges[:, 1], subject_edges[:, 0])
        )
        subject_lengths = np.linalg.norm(subject_axes, axis=1)
        subject_axes /= subject_lengths[:, None]
        subject_projection = subject @ subject_axes.T
        candidate_projection = np.einsum(
            "kvc,ac->kva",
            candidates,
            subject_axes,
        )
        subject_axis_overlap = (
            np.minimum(
                np.max(subject_projection, axis=0)[None, :],
                np.max(candidate_projection, axis=1),
            )
            - np.maximum(
                np.min(subject_projection, axis=0)[None, :],
                np.min(candidate_projection, axis=1),
            )
        )
        possible = np.all(
            subject_axis_overlap >= -linear_tolerance,
            axis=1,
        )
        if not np.any(possible):
            return possible

        candidate_edges = np.roll(candidates, -1, axis=1) - candidates
        candidate_axes = np.stack(
            (-candidate_edges[:, :, 1], candidate_edges[:, :, 0]),
            axis=2,
        )
        candidate_lengths = np.linalg.norm(candidate_axes, axis=2)
        candidate_axes /= candidate_lengths[:, :, None]
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
        candidate_axis_overlap = (
            np.minimum(
                np.max(candidate_self_projection, axis=1),
                np.max(subject_on_candidate_axes, axis=1),
            )
            - np.maximum(
                np.min(candidate_self_projection, axis=1),
                np.min(subject_on_candidate_axes, axis=1),
            )
        )
        return possible & np.all(
            candidate_axis_overlap >= -linear_tolerance,
            axis=1,
        )

    for record_index, record in enumerate(records):
        (
            record_minimum_x,
            record_maximum_x,
            record_minimum_y,
            record_maximum_y,
            fragment_id,
            polygon,
        ) = record
        first_x = int(
            np.floor(
                (record_minimum_x - linear_tolerance - minimum_x)
                / cell_width
            )
        )
        last_x = int(
            np.floor(
                (record_maximum_x + linear_tolerance - minimum_x)
                / cell_width
            )
        )
        first_y = int(
            np.floor(
                (record_minimum_y - linear_tolerance - minimum_y)
                / cell_height
            )
        )
        last_y = int(
            np.floor(
                (record_maximum_y + linear_tolerance - minimum_y)
                / cell_height
            )
        )
        cells = tuple(
            (x_index, y_index)
            for x_index in range(first_x, last_x + 1)
            for y_index in range(first_y, last_y + 1)
        )
        candidate_indices = sorted(
            {
                candidate_index
                for cell in cells
                for candidate_index in grid.get(cell, ())
            }
        )
        bbox_candidate_indices = [
            candidate_index
            for candidate_index in candidate_indices
            if not (
                records[candidate_index][1]
                < record_minimum_x - linear_tolerance
                or record_maximum_x
                < records[candidate_index][0] - linear_tolerance
                or records[candidate_index][3]
                < record_minimum_y - linear_tolerance
                or record_maximum_y
                < records[candidate_index][2] - linear_tolerance
            )
        ]
        if bbox_candidate_indices:
            candidate_polygons = np.asarray(
                [records[index][5] for index in bbox_candidate_indices],
                dtype=float,
            )
            overlap_mask = sat_maybe_overlaps(
                np.asarray(polygon, dtype=float),
                candidate_polygons,
            )
            exact_candidate_indices = [
                candidate_index
                for candidate_index, possible in zip(
                    bbox_candidate_indices,
                    overlap_mask,
                )
                if possible
            ]
        else:
            exact_candidate_indices = []
        for candidate_index in exact_candidate_indices:
            candidate = records[candidate_index]
            overlap = _clip_convex_polygon_for_contract(
                polygon,
                candidate[5],
                linear_tolerance,
            )
            overlap_area = _polygon_area(overlap)
            if overlap_area > area_tolerance:
                issues.append(
                    f"{candidate[4]} overlaps {fragment_id} by "
                    f"{overlap_area:.12g}"
                )
                if len(issues) >= 8:
                    return tuple(issues)
        for cell in cells:
            grid.setdefault(cell, []).append(record_index)
    return tuple(issues)


def _partition_topology_issues(
    fragments: Sequence,
    patch: Sequence[Sequence[float]],
    *,
    linear_tolerance: float,
    area_tolerance: float,
) -> tuple[str, ...]:
    """Certify one oriented planar triangulation without quadratic pair scans."""

    registry = _CanonicalVertexRegistry(
        plane_origin=(0.0, 0.0, 0.0),
        plane_u=(1.0, 0.0, 0.0),
        plane_v=(0.0, 1.0, 0.0),
        screen_origin=(0.0, 0.0),
        screen_basis=((1.0, 0.0), (0.0, 1.0)),
        coordinate_epsilon=linear_tolerance,
    )
    polygons = []
    for fragment in fragments:
        polygon = _make_plane_partition_polygon(
            fragment.fragment_id,
            tuple(
                registry.register(point)
                for point in fragment.screen_vertices
            ),
            linear_tolerance,
        )
        if polygon is None:
            return (f"{fragment.fragment_id} has no stable topology",)
        polygons.append(polygon)
    try:
        loops = _plane_partition_polygon_contours(
            polygons,
            linear_tolerance,
        )
    except QuadricSectionCompositingError as exc:
        return (f"fragment topology is not a closed planar partition: {exc}",)
    if len(loops) != 1:
        return (
            "fragment topology has "
            f"{len(loops)} residual boundary loops instead of one patch loop",
        )
    loop = tuple(vertex.screen_point for vertex in loops[0])
    loop_area = _polygon_area(loop)
    patch_area = _polygon_area(patch)
    if abs(loop_area - patch_area) > area_tolerance:
        return (
            f"residual boundary area {loop_area:.12g} differs from patch "
            f"area {patch_area:.12g}",
        )
    clipped = _clip_convex_polygon_for_contract(
        loop,
        patch,
        linear_tolerance,
    )
    if abs(_polygon_area(clipped) - patch_area) > area_tolerance:
        return ("residual boundary does not coincide with the patch",)
    return ()


_FRAGMENT_SAMPLE_WEIGHTS = (
    ("vertex-0", (1.0, 0.0, 0.0)),
    ("vertex-1", (0.0, 1.0, 0.0)),
    ("vertex-2", (0.0, 0.0, 1.0)),
    ("edge-01", (0.5, 0.5, 0.0)),
    ("edge-12", (0.0, 0.5, 0.5)),
    ("edge-20", (0.5, 0.0, 0.5)),
    ("centroid", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
    ("interior-near-0", (0.6, 0.2, 0.2)),
    ("interior-near-1", (0.2, 0.6, 0.2)),
    ("interior-near-2", (0.2, 0.2, 0.6)),
)


def _fragment_samples(fragment):
    world = np.asarray(fragment.world_vertices, dtype=float)
    screen = np.asarray(fragment.screen_vertices, dtype=float)
    for label, raw_weights in _FRAGMENT_SAMPLE_WEIGHTS:
        weights = np.asarray(raw_weights, dtype=float)
        yield label, weights @ world, weights @ screen


def _convex_signed_margin(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> float:
    value = np.asarray(point, dtype=float)
    boundary = _counter_clockwise(polygon)
    return min(
        _cross2(
            boundary[(index + 1) % len(boundary)] - boundary[index],
            value - boundary[index],
        )
        / float(
            np.linalg.norm(
                boundary[(index + 1) % len(boundary)] - boundary[index]
            )
        )
        for index in range(len(boundary))
    )


def _case_issue_summary(
    case_name: str,
    total: int,
    examples: Sequence[str],
) -> str:
    joined = "; ".join(examples[:4])
    suffix = "" if total <= len(examples[:4]) else "; ..."
    return f"{case_name}: {total} violation(s): {joined}{suffix}"


def _ray_role(
    solver,
    point: Sequence[float],
    boundary_epsilon: float,
) -> PlaneDepthRole:
    parameters = solver(np.asarray(point, dtype=float))
    if not parameters:
        return PlaneDepthRole.OUTSIDE_PROJECTION
    if min(parameters) > boundary_epsilon:
        return PlaneDepthRole.BEHIND_SURFACE
    if max(parameters) < -boundary_epsilon:
        return PlaneDepthRole.IN_FRONT_OF_SURFACE
    return PlaneDepthRole.BETWEEN_SURFACE_SHEETS


def _stable_ray_role(
    solver,
    point: Sequence[float],
    plane: SectionPlane,
    boundary_epsilon: float,
    *,
    geometric_boundary_tolerance: float = 0.0,
) -> PlaneDepthRole | None:
    value = np.asarray(point, dtype=float)
    plane_u, plane_v, _normal = plane.basis
    probe = max(
        16.0 * boundary_epsilon,
        geometric_boundary_tolerance,
    )
    roles = {
        _ray_role(solver, value + offset, boundary_epsilon)
        for offset in (
            np.zeros(3, dtype=float),
            probe * plane_u,
            -probe * plane_u,
            probe * plane_v,
            -probe * plane_v,
        )
    }
    return next(iter(roles)) if len(roles) == 1 else None


_PARTITION_EPSILON = 1.0e-10


def _partition_registry() -> _CanonicalVertexRegistry:
    return _CanonicalVertexRegistry(
        plane_origin=(0.0, 0.0, 0.0),
        plane_u=(1.0, 0.0, 0.0),
        plane_v=(0.0, 1.0, 0.0),
        screen_origin=(0.0, 0.0),
        screen_basis=((1.0, 0.0), (0.0, 1.0)),
        coordinate_epsilon=_PARTITION_EPSILON,
    )


def _partition_polygon(
    registry: _CanonicalVertexRegistry,
    token: str,
    coordinates: Sequence[Sequence[float]],
):
    polygon = _make_plane_partition_polygon(
        token,
        tuple(registry.register(point) for point in coordinates),
        _PARTITION_EPSILON,
    )
    if polygon is None:
        raise AssertionError(f"test polygon {token!r} is degenerate")
    return polygon


def _partition_polygon_coordinates(polygon) -> tuple[tuple[float, float], ...]:
    return tuple(vertex.plane_coordinates for vertex in polygon.vertices)


def _partition_polygon_area(polygon) -> float:
    return _polygon_area(_partition_polygon_coordinates(polygon))


def _partition_loop_signed_area(loop) -> float:
    return _screen_signed_area(
        tuple(vertex.plane_coordinates for vertex in loop)
    )


class PlanePartitionInfrastructureTests(unittest.TestCase):
    """Batch-2 tests for private renderer-neutral partition primitives."""

    def assert_complete_proxy_partition(
        self,
        triangle,
        proxy,
        inside,
        outside,
    ) -> None:
        pieces = (*inside, *outside)
        source_area = _partition_polygon_area(triangle)
        restored_area = sum(_partition_polygon_area(item) for item in pieces)
        self.assertAlmostEqual(source_area, restored_area, places=8)
        for first_index, first in enumerate(pieces):
            first_coordinates = _partition_polygon_coordinates(first)
            for second in pieces[first_index + 1 :]:
                overlap = _clip_convex_polygon_for_contract(
                    first_coordinates,
                    _partition_polygon_coordinates(second),
                    _PARTITION_EPSILON,
                )
                self.assertLessEqual(_polygon_area(overlap), 1.0e-9)

        proxy_coordinates = _partition_polygon_coordinates(proxy)
        for polygon in inside:
            overlap = _clip_convex_polygon_for_contract(
                _partition_polygon_coordinates(polygon),
                proxy_coordinates,
                _PARTITION_EPSILON,
            )
            self.assertAlmostEqual(
                _partition_polygon_area(polygon),
                _polygon_area(overlap),
                places=8,
            )
        for polygon in outside:
            overlap = _clip_convex_polygon_for_contract(
                _partition_polygon_coordinates(polygon),
                proxy_coordinates,
                _PARTITION_EPSILON,
            )
            self.assertLessEqual(_polygon_area(overlap), 1.0e-9)

        for polygon in pieces:
            triangles = _triangulate_plane_partition_polygon(
                polygon,
                _PARTITION_EPSILON,
            )
            self.assertTrue(triangles)
            self.assertAlmostEqual(
                _partition_polygon_area(polygon),
                sum(_partition_polygon_area(item) for item in triangles),
                places=8,
            )
            self.assertTrue(
                all(
                    _screen_signed_area(
                        _partition_polygon_coordinates(item)
                    ) > 0.0
                    for item in triangles
                )
            )

    def test_half_plane_split_preserves_area_and_intersection_identity(self) -> None:
        registry = _partition_registry()
        subject = _partition_polygon(
            registry,
            "subject",
            ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
        )
        boundary_start = registry.register((0.0, -2.0))
        boundary_end = registry.register((0.0, 2.0))
        inside, outside = _split_convex_polygon_by_half_plane(
            subject,
            boundary_start,
            boundary_end,
            registry,
            _PARTITION_EPSILON,
            boundary_token="vertical",
        )
        self.assertIsNotNone(inside)
        self.assertIsNotNone(outside)
        assert inside is not None and outside is not None
        self.assertAlmostEqual(
            _partition_polygon_area(subject),
            _partition_polygon_area(inside) + _partition_polygon_area(outside),
            places=9,
        )
        overlap = _clip_convex_polygon_for_contract(
            _partition_polygon_coordinates(inside),
            _partition_polygon_coordinates(outside),
            _PARTITION_EPSILON,
        )
        self.assertLessEqual(_polygon_area(overlap), 1.0e-9)
        shared = {
            item.stable_token for item in inside.vertices
        } & {
            item.stable_token for item in outside.vertices
        }
        self.assertEqual(len(shared), 2)
        inside_vertices = {
            item.stable_token: item for item in inside.vertices
        }
        outside_vertices = {
            item.stable_token: item for item in outside.vertices
        }
        for token in shared:
            self.assertIs(inside_vertices[token], outside_vertices[token])
            vertex = inside_vertices[token]
            self.assertAlmostEqual(vertex.plane_coordinates[0], 0.0, places=9)
            self.assertEqual(vertex.world_point[2], 0.0)
            self.assertEqual(vertex.world_point[:2], vertex.screen_point)

    def test_half_plane_split_preserves_one_sided_and_coincident_boundaries(
        self,
    ) -> None:
        cases = (
            (
                "fully-inside",
                (2.0, -2.0),
                (2.0, 2.0),
                True,
                False,
            ),
            (
                "fully-outside",
                (-2.0, -2.0),
                (-2.0, 2.0),
                False,
                True,
            ),
            (
                "coincident-left-edge",
                (-1.0, -2.0),
                (-1.0, 2.0),
                False,
                True,
            ),
        )
        for name, start, end, expects_inside, expects_outside in cases:
            with self.subTest(case=name):
                registry = _partition_registry()
                subject = _partition_polygon(
                    registry,
                    "subject",
                    ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
                )
                inside, outside = _split_convex_polygon_by_half_plane(
                    subject,
                    registry.register(start),
                    registry.register(end),
                    registry,
                    _PARTITION_EPSILON,
                    boundary_token=name,
                )
                self.assertEqual(inside is not None, expects_inside)
                self.assertEqual(outside is not None, expects_outside)
                child = inside if inside is not None else outside
                assert child is not None
                self.assertEqual(
                    tuple(item.stable_token for item in child.vertices),
                    tuple(item.stable_token for item in subject.vertices),
                )
                self.assertAlmostEqual(
                    _partition_polygon_area(child),
                    _partition_polygon_area(subject),
                    places=9,
                )

    def test_triangle_proxy_partition_covers_all_geometric_cases(self) -> None:
        cases = (
            (
                "fully-outside",
                ((2.0, 0.0), (3.0, 0.0), (2.0, 1.0)),
                False,
            ),
            (
                "fully-inside",
                ((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)),
                True,
            ),
            (
                "edge-crossing",
                ((-0.5, -0.5), (0.5, -0.5), (1.8, 0.3)),
                True,
            ),
            (
                "corner-entering",
                ((0.8, 0.8), (2.2, 0.8), (0.8, 2.2)),
                True,
            ),
            (
                "proxy-inside-triangle",
                ((-4.0, -3.0), (4.0, -3.0), (0.0, 5.0)),
                True,
            ),
        )
        for name, coordinates, expects_inside in cases:
            with self.subTest(case=name):
                registry = _partition_registry()
                proxy = _partition_polygon(
                    registry,
                    "proxy",
                    ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
                )
                triangle = _partition_polygon(registry, name, coordinates)
                inside, outside = _partition_triangle_by_convex_proxy(
                    triangle,
                    proxy,
                    registry,
                    _PARTITION_EPSILON,
                )
                self.assertEqual(bool(inside), expects_inside)
                self.assert_complete_proxy_partition(
                    triangle,
                    proxy,
                    inside,
                    outside,
                )
                if name == "fully-inside":
                    self.assertFalse(outside)
                if name == "fully-outside":
                    self.assertFalse(inside)
                if name == "proxy-inside-triangle":
                    self.assertEqual(len(inside), 1)
                    self.assertAlmostEqual(
                        _partition_polygon_area(inside[0]),
                        _partition_polygon_area(proxy),
                        places=8,
                    )
                    self.assertGreaterEqual(len(outside), 3)

                repeated_registry = _partition_registry()
                repeated_proxy = _partition_polygon(
                    repeated_registry,
                    "proxy",
                    ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
                )
                repeated_triangle = _partition_polygon(
                    repeated_registry,
                    name,
                    coordinates,
                )
                self.assertEqual(
                    (inside, outside),
                    _partition_triangle_by_convex_proxy(
                        repeated_triangle,
                        repeated_proxy,
                        repeated_registry,
                        _PARTITION_EPSILON,
                    ),
                )

    def test_polygon_triangulation_is_stable_and_drops_degenerate_fans(self) -> None:
        registry = _partition_registry()
        polygon = _partition_polygon(
            registry,
            "pentagon",
            (
                (-1.2, -0.1),
                (-0.4, -1.1),
                (0.9, -0.7),
                (1.1, 0.6),
                (-0.2, 1.2),
            ),
        )
        first = _triangulate_plane_partition_polygon(
            polygon,
            _PARTITION_EPSILON,
        )
        second = _triangulate_plane_partition_polygon(
            polygon,
            _PARTITION_EPSILON,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(
            tuple(item.stable_token for item in first),
            tuple(
                f"pentagon:triangle:{index:04d}" for index in range(3)
            ),
        )
        self.assertAlmostEqual(
            _partition_polygon_area(polygon),
            sum(_partition_polygon_area(item) for item in first),
            places=8,
        )

        near_collinear = _PlanePartitionPolygon(
            "near-collinear",
            (
                registry.register((0.0, 0.0)),
                registry.register((1.0, 0.0)),
                registry.register((2.0, 1.0e-12)),
                registry.register((2.0, 1.0)),
            ),
        )
        stable_only = _triangulate_plane_partition_polygon(
            near_collinear,
            _PARTITION_EPSILON,
        )
        self.assertEqual(len(stable_only), 1)
        self.assertEqual(
            stable_only[0].stable_token,
            "near-collinear:triangle:0001",
        )

    def test_contour_union_nodes_arbitrary_t_junctions(self) -> None:
        registry = _partition_registry()
        polygons = (
            _partition_polygon(
                registry,
                "left",
                ((0.1, 0.2), (0.73, 0.2), (0.73, 1.91), (0.1, 1.91)),
            ),
            _partition_polygon(
                registry,
                "right-lower",
                ((0.73, 0.2), (2.37, 0.2), (2.37, 0.83), (0.73, 0.83)),
            ),
            _partition_polygon(
                registry,
                "right-upper",
                ((0.73, 0.83), (2.37, 0.83), (2.37, 1.91), (0.73, 1.91)),
            ),
        )
        loops = _plane_partition_polygon_contours(
            polygons,
            _PARTITION_EPSILON,
        )
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 4)
        self.assertAlmostEqual(
            _partition_loop_signed_area(loops[0]),
            (2.37 - 0.1) * (1.91 - 0.2),
            places=8,
        )

    def test_contour_union_preserves_hole_winding(self) -> None:
        registry = _partition_registry()
        ring = (
            _partition_polygon(
                registry,
                "bottom",
                ((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)),
            ),
            _partition_polygon(
                registry,
                "right",
                ((2.0, 1.0), (3.0, 1.0), (3.0, 2.0), (2.0, 2.0)),
            ),
            _partition_polygon(
                registry,
                "top",
                ((0.0, 2.0), (3.0, 2.0), (3.0, 3.0), (0.0, 3.0)),
            ),
            _partition_polygon(
                registry,
                "left",
                ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)),
            ),
        )
        loops = _plane_partition_polygon_contours(ring, _PARTITION_EPSILON)
        self.assertEqual(len(loops), 2)
        signed_areas = sorted(_partition_loop_signed_area(item) for item in loops)
        self.assertAlmostEqual(signed_areas[0], -1.0, places=8)
        self.assertAlmostEqual(signed_areas[1], 9.0, places=8)
        self.assertAlmostEqual(sum(signed_areas), 8.0, places=8)

    def test_contour_union_keeps_disjoint_regions_and_normalizes_winding(self) -> None:
        registry = _partition_registry()
        polygons = (
            _partition_polygon(
                registry,
                "first-clockwise",
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
            ),
            _partition_polygon(
                registry,
                "second",
                ((2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)),
            ),
        )
        first = _plane_partition_polygon_contours(
            polygons,
            _PARTITION_EPSILON,
        )
        second = _plane_partition_polygon_contours(
            polygons,
            _PARTITION_EPSILON,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(tuple(len(item) for item in first), (4, 4))
        self.assertEqual(
            tuple(round(_partition_loop_signed_area(item), 8) for item in first),
            (1.0, 1.0),
        )

    def test_ring_coalesces_near_coincident_opposite_boundary_events(self) -> None:
        registry = _partition_registry()
        angular_offset = 1.0e-7
        inner = _partition_polygon(
            registry,
            "inner",
            ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
        )
        outer = _partition_polygon(
            registry,
            "outer",
            (
                (2.0 * cos(angular_offset), 2.0 * sin(angular_offset)),
                (0.0, 2.0),
                (-2.0, 0.0),
                (0.0, -2.0),
            ),
        )
        raw = _nested_convex_ring_polygons(
            inner,
            outer,
            registry,
            _PARTITION_EPSILON,
        )
        with patch(
            "polyhedron_visibility.quadrics.section_compositing."
            "_nested_convex_ring_ray_vertex",
            wraps=_nested_convex_ring_ray_vertex,
        ) as ray_evaluation:
            stable = _nested_convex_ring_polygons(
                inner,
                outer,
                registry,
                _PARTITION_EPSILON,
                minimum_screen_triangle_altitude=1.0e-6,
            )
        evaluated_keys = tuple(
            (
                call.args[0].stable_token,
                float(call.args[3]).hex(),
            )
            for call in ray_evaluation.call_args_list
        )
        self.assertEqual(len(evaluated_keys), len(set(evaluated_keys)))
        repeated = _nested_convex_ring_polygons(
            inner,
            outer,
            registry,
            _PARTITION_EPSILON,
            minimum_screen_triangle_altitude=1.0e-6,
        )
        self.assertEqual(stable, repeated)
        self.assertEqual(len(stable), len(raw) - 1)
        loops = _plane_partition_polygon_contours(
            stable,
            _PARTITION_EPSILON,
        )
        self.assertEqual(len(loops), 2)
        self.assertAlmostEqual(
            sum(_partition_loop_signed_area(loop) for loop in loops),
            _partition_polygon_area(outer) - _partition_polygon_area(inner),
            places=8,
        )

    def test_partition_topology_certificate_detects_gap_and_overlap(self) -> None:
        def fragment(
            fragment_id: str,
            vertices: tuple[
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
            ],
        ) -> QuadricPlaneFragment:
            return QuadricPlaneFragment(
                fragment_id=fragment_id,
                role=PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                world_vertices=tuple((x, y, 0.0) for x, y in vertices),
                screen_vertices=vertices,
                subdivision_depth=0,
            )

        patch = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
        complete = (
            fragment("lower", ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0))),
            fragment("upper", ((0.0, 0.0), (2.0, 2.0), (0.0, 2.0))),
        )
        kwargs = {
            "linear_tolerance": 1.0e-9,
            "area_tolerance": 1.0e-9,
        }
        self.assertEqual(
            _partition_topology_issues(complete, patch, **kwargs),
            (),
        )
        gap = (
            complete[0],
            fragment("upper-gap", ((0.0, 0.1), (2.0, 2.0), (0.0, 2.0))),
        )
        overlap = (
            *complete,
            fragment("overlap", ((0.2, 0.2), (0.4, 0.2), (0.2, 0.4))),
        )
        self.assertTrue(_partition_topology_issues(gap, patch, **kwargs))
        self.assertTrue(_partition_topology_issues(overlap, patch, **kwargs))

    def test_public_fragment_contract_is_unchanged(self) -> None:
        self.assertEqual(
            tuple(QuadricPlaneFragment.__dataclass_fields__),
            (
                "fragment_id",
                "role",
                "world_vertices",
                "screen_vertices",
                "subdivision_depth",
            ),
        )


class QuadricSectionCompositingTests(unittest.TestCase):
    def test_repaint_reuses_geometry_and_matches_each_cold_policy_frame(self) -> None:
        sphere = SphereSpec("repaint-sphere", (0.0, 0.0, 0.0), 1.0)
        curve = CircleArcCurve(
            "repaint-hidden-circle",
            (0.0, 0.0, -1.2),
            0.6,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "repaint-plane",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "repaint-patch",
            plane,
            (sphere,),
            margin_ratio=0.1,
        ).patch
        proxy = build_opaque_projection_proxy(
            sphere,
            IDENTITY_VIEW,
            max_chord_error=0.01,
        )
        visibility = compute_quadric_visibility(
            (curve,),
            (sphere,),
            IDENTITY_VIEW,
        )
        bases = {
            policy: compute_quadric_compositing(
                visibility,
                (proxy,),
                paint_policy=policy,
            )
            for policy in QuadricPaintPolicy
        }
        geometry = compute_quadric_section_compositing(
            bases[QuadricPaintPolicy.PHYSICAL],
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        for policy, base in bases.items():
            with self.subTest(policy=policy.value):
                repainted = repaint_quadric_section_compositing(geometry, base)
                cold = compute_quadric_section_compositing(
                    base,
                    sphere,
                    plane,
                    patch,
                    IDENTITY_VIEW,
                )
                self.assertEqual(
                    canonical_quadric_section_compositing_json(repainted),
                    canonical_quadric_section_compositing_json(cold),
                )

    def test_adversarial_flat_conic_uses_certified_tangent_bounds(self) -> None:
        start = 2.75084
        end = 5.94173
        domain = ParameterInterval(start, end)
        curve = ParametricConicBranch(
            "adversarial-flat-ellipse",
            ConicParameterization(
                kind=ConicKind.ELLIPSE,
                branch_label="flat",
                origin=(0.0, 0.0),
                first_axis=(2.0, 0.0),
                second_axis=(0.0, 2.0e-6),
                natural_domain=domain,
            ),
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            domain,
        )
        public_error = 0.08
        certified_error = public_error / _SECTION_BOUNDARY_CHORD_DIVISOR
        projection = np.asarray(IDENTITY_VIEW.matrix[:2], dtype=float)
        first = projection @ np.asarray(curve.point(start), dtype=float)
        last = projection @ np.asarray(curve.point(end), dtype=float)

        old_probe_errors = tuple(
            _point_segment_distance_2d(
                projection
                @ np.asarray(
                    curve.point(start + fraction * (end - start)),
                    dtype=float,
                ),
                first,
                last,
            )
            for fraction in (0.25, 0.5, 0.75)
        )
        dense_error = max(
            _point_segment_distance_2d(
                projection @ np.asarray(curve.point(float(parameter)), dtype=float),
                first,
                last,
            )
            for parameter in np.linspace(start, end, 4097)
        )
        self.assertLess(max(old_probe_errors), certified_error)
        self.assertGreater(dense_error, public_error)

        parameters = _adaptive_section_curve_parameters(
            curve,
            IDENTITY_VIEW,
            (start, end),
            max_chord_error=certified_error,
            max_segments=8192,
            parameter_epsilon=1.0e-12,
        )
        self.assertGreater(len(parameters), 2)
        self.assertEqual(
            parameters,
            _adaptive_section_curve_parameters(
                curve,
                IDENTITY_VIEW,
                (start, end),
                max_chord_error=certified_error,
                max_segments=8192,
                parameter_epsilon=1.0e-12,
            ),
        )
        for left, right in zip(parameters, parameters[1:]):
            with self.subTest(left=left, right=right):
                bound = _section_curve_tangent_envelope_error_bound(
                    curve,
                    IDENTITY_VIEW,
                    left,
                    right,
                )
                first = projection @ np.asarray(curve.point(left), dtype=float)
                last = projection @ np.asarray(curve.point(right), dtype=float)
                sampled_error = max(
                    _point_segment_distance_2d(
                        projection
                        @ np.asarray(curve.point(float(parameter)), dtype=float),
                        first,
                        last,
                    )
                    for parameter in np.linspace(left, right, 17)
                )
                self.assertLessEqual(bound, certified_error)
                self.assertLessEqual(sampled_error, bound)

    def test_display_ray_classifier_matches_authoritative_finite_hits(self) -> None:
        surfaces = (
            SphereSpec("sphere", (0.2, -0.1, 0.3), 1.1),
            CylinderSpec(
                "cylinder",
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
                0.9,
                (-0.4, 2.1),
                radial_axis=(1.0, 0.0, 0.0),
            ),
            ConeSpec(
                "cone",
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
                pi / 6.0,
                (0.0, 3.0),
                radial_axis=(1.0, 0.0, 0.0),
            ),
        )
        direction = np.asarray((0.31, -0.27, 0.91), dtype=float)
        direction /= np.linalg.norm(direction)
        generator = np.random.default_rng(20260824)
        for surface in surfaces:
            with self.subTest(surface=surface.surface_id):
                context = GeometryContext().resolve(surface.characteristic_points)
                boundary = context.epsilon(GeometryQuantity.BOUNDARY)
                solver = _surface_ray_solver(
                    surface,
                    direction,
                    boundary_epsilon=boundary,
                    angular_epsilon=context.epsilon(GeometryQuantity.ANGULAR),
                )
                for point in generator.uniform(-2.5, 2.5, size=(40, 3)):
                    expected: list[float] = []
                    for hit in surface.ray_hits(
                        point,
                        direction,
                        context=context,
                        include_caps=True,
                        forward_only=False,
                    ):
                        if (
                            not expected
                            or abs(float(hit.parameter) - expected[-1]) > boundary
                        ):
                            expected.append(float(hit.parameter))
                    np.testing.assert_allclose(
                        solver(point),
                        tuple(expected),
                        rtol=0.0,
                        atol=max(boundary, 1.0e-11),
                    )

    def test_tilted_sphere_plane_splits_all_three_depth_regions(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "cut-patch", plane, (sphere,), margin_ratio=0.1
        ).patch
        frame = compute_quadric_section_compositing(
            _base_frame(sphere),
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        roles = {item.role for item in frame.plane_fragments}
        self.assertTrue(
            {
                PlaneDepthRole.BEHIND_SURFACE,
                PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                PlaneDepthRole.IN_FRONT_OF_SURFACE,
            }.issubset(roles)
        )
        restored_area = sum(
            _triangle_area(item.world_vertices) for item in frame.plane_fragments
        )
        self.assertAlmostEqual(
            restored_area,
            4.0 * patch.half_width * patch.half_height,
            places=9,
        )
        self.assertEqual(set(frame.draw_order), {
            *frame.paint_items.ordered,
        })

    def test_plane_fragments_merge_to_equivalent_renderer_contours(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "cut-patch", plane, (sphere,), margin_ratio=0.1
        ).patch
        frame = compute_quadric_section_compositing(
            _base_frame(sphere),
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        first = quadric_plane_fragment_contours(frame)
        second = quadric_plane_fragment_contours(frame)
        self.assertEqual(first, second)
        for role in PlaneDepthRole:
            source_area = sum(
                _screen_signed_area(item.screen_vertices)
                for item in frame.fragments_by_role[role]
            )
            contour_area = sum(
                _screen_signed_area(item) for item in first[role]
            )
            self.assertAlmostEqual(source_area, contour_area, places=10)
        self.assertLess(
            sum(len(items) for items in first.values()),
            len(frame.plane_fragments) // 10,
        )

    def test_small_near_tangent_section_is_not_missed_by_coarse_samples(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "near-tangent",
            (0.0, 0.0, 0.95),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "near-tangent-patch", plane, (sphere,), margin_ratio=0.1
        ).patch
        frame = compute_quadric_section_compositing(
            _base_frame(sphere),
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        self.assertTrue(frame.fragments_by_role[PlaneDepthRole.BETWEEN_SURFACE_SHEETS])
        between_area = sum(
            abs(_screen_signed_area(item.screen_vertices))
            for item in frame.fragments_by_role[PlaneDepthRole.BETWEEN_SURFACE_SHEETS]
        )
        self.assertGreater(between_area, 0.1)

    def test_surface_sheets_plane_groups_and_curves_share_one_draw_order(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        curve = CircleArcCurve(
            "equator",
            (0.0, 0.0, 0.0),
            1.0,
            (0.0, 1.0, 0.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch("patch", plane, (sphere,)).patch
        base = _base_frame(sphere, (curve,))
        frame = compute_quadric_section_compositing(
            base,
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        active_curves = {
            item.item_id for item in base.curve_fragments if item.painted
        }
        self.assertEqual(
            set(frame.draw_order),
            {*frame.paint_items.ordered, *active_curves},
        )
        outline_rank = frame.draw_order.index(frame.paint_items.plane_outline)
        self.assertTrue(
            all(frame.draw_order.index(item_id) > outline_rank for item_id in active_curves)
        )

    def test_depth_aware_hidden_curve_sits_between_surface_sheets(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        curve = CircleArcCurve(
            "equator",
            (0.0, 0.0, 0.0),
            1.0,
            (0.0, 1.0, 0.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch("patch", plane, (sphere,)).patch
        base = _base_frame(
            sphere,
            (curve,),
            paint_policy="depth_aware_diagrammatic",
        )
        frame = compute_quadric_section_compositing(
            base,
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )
        hidden = tuple(
            item.item_id
            for item in base.curve_fragments
            if item.kind is QuadricPaintKind.HIDDEN_CURVE
        )
        visible = tuple(
            item.item_id
            for item in base.curve_fragments
            if item.kind is QuadricPaintKind.VISIBLE_CURVE
        )
        self.assertTrue(hidden)
        self.assertTrue(visible)

        ranks = {item_id: index for index, item_id in enumerate(frame.draw_order)}
        self.assertLess(
            ranks[frame.paint_items.plane_outline_between],
            min(ranks[item_id] for item_id in hidden),
        )
        self.assertLess(
            max(ranks[item_id] for item_id in hidden),
            ranks[frame.paint_items.surface_front],
        )
        self.assertLess(
            ranks[frame.paint_items.plane_outline],
            min(ranks[item_id] for item_id in visible),
        )

        expected_depth_chain = (
            frame.paint_items.plane_behind,
            frame.paint_items.plane_outline_behind,
            frame.paint_items.surface_back,
            frame.paint_items.plane_outside,
            frame.paint_items.plane_outline_outside,
            frame.paint_items.plane_between,
            frame.paint_items.plane_outline_between,
            *hidden,
            frame.paint_items.surface_front,
            frame.paint_items.plane_front,
            frame.paint_items.plane_outline,
        )
        self.assertEqual(
            tuple(
                item_id
                for item_id in frame.draw_order
                if item_id in set(expected_depth_chain)
            ),
            expected_depth_chain,
        )

    def test_cone_transition_style_frame_is_deterministic(self) -> None:
        cone = ConeSpec(
            "cone",
            (0.0, 0.0, -1.5),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.2),
            (0.5, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "patch", plane, (cone,), margin_ratio=0.08
        ).patch
        first = compute_quadric_section_compositing(
            _base_frame(cone),
            cone,
            plane,
            patch,
            IDENTITY_VIEW,
        )
        second = compute_quadric_section_compositing(
            _base_frame(cone),
            cone,
            plane,
            patch,
            IDENTITY_VIEW,
        )
        self.assertEqual(
            canonical_quadric_section_compositing_json(first),
            canonical_quadric_section_compositing_json(second),
        )

    def test_near_cylinder_frustum_keeps_a_finite_section_scale(self) -> None:
        radius = 1.45
        slope = (radius / 3.0) * 1.0e-6
        apex_z = -radius / slope
        cone = ConeSpec(
            "near-cylinder",
            (0.0, 0.0, apex_z),
            (0.0, 0.0, 1.0),
            atan(slope),
            (-3.0 - apex_z, 5.8 - apex_z),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_SINGLE,
        )
        cylinder = CylinderSpec(
            "near-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            radius,
            (-3.0, 5.8),
            radial_axis=(1.0, 0.0, 0.0),
        )
        depth = _unit((0.75, -1.25, 0.55))
        screen_right = _unit((-depth[1], depth[0], 0.0))
        screen_up = _unit(np.cross(depth, screen_right))
        view = ParallelView.from_matrix((screen_right, screen_up, depth))
        normal = _unit((0.25, 0.0, sqrt(1.0 - 0.25**2)))
        plane = SectionPlane(
            "near-cylinder-plane",
            tuple(float(value) for value in 0.20 * normal),
            tuple(float(value) for value in normal),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = PlaneDisplayPatchSpec(
            "near-cylinder-patch",
            plane.plane_id,
            3.35,
            3.55,
        )

        def section_frame(surface: QuadricSurface):
            proxy = build_opaque_projection_proxy(
                surface,
                view,
                max_chord_error=0.01,
            )
            visibility = compute_quadric_visibility((), (surface,), view)
            base = compute_quadric_compositing(visibility, (proxy,))
            return compute_quadric_section_compositing(
                base,
                surface,
                plane,
                patch,
                view,
                max_screen_error=0.08,
            )

        cone_frame = section_frame(cone)
        cylinder_frame = section_frame(cylinder)
        self.assertIs(cone_frame.projection_kind, PlanePatchProjectionKind.AREA)
        self.assertEqual(
            {fragment.role for fragment in cone_frame.plane_fragments},
            set(PlaneDepthRole),
        )
        restored_area = sum(
            _triangle_area(fragment.world_vertices)
            for fragment in cone_frame.plane_fragments
        )
        self.assertAlmostEqual(
            restored_area,
            4.0 * patch.half_width * patch.half_height,
            places=8,
        )

        def contour_areas(frame) -> tuple[float, ...]:
            contours = quadric_plane_fragment_contours(frame)
            return tuple(
                sum(abs(_screen_signed_area(path)) for path in contours[role])
                for role in PlaneDepthRole
            )

        np.testing.assert_allclose(
            contour_areas(cone_frame),
            contour_areas(cylinder_frame),
            rtol=0.0,
            atol=1.0e-4,
        )

    def test_formal_transition_tangent_frame_preserves_disjoint_role_components(
        self,
    ) -> None:
        first = _transition_tangent_regression_frame()
        second = _transition_tangent_regression_frame()

        first_ids = tuple(item.fragment_id for item in first.plane_fragments)
        second_ids = tuple(item.fragment_id for item in second.plane_fragments)
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertEqual(
            canonical_quadric_section_compositing_json(first),
            canonical_quadric_section_compositing_json(second),
        )
        restored_area = sum(
            _triangle_area(item.world_vertices)
            for item in first.plane_fragments
        )
        self.assertAlmostEqual(
            restored_area,
            4.0 * first.patch.half_width * first.patch.half_height,
            places=8,
        )
        contours = quadric_plane_fragment_contours(first)
        for role in PlaneDepthRole:
            fragment_area = sum(
                _screen_signed_area(item.screen_vertices)
                for item in first.fragments_by_role[role]
            )
            contour_area = sum(
                _screen_signed_area(contour) for contour in contours[role]
            )
            self.assertAlmostEqual(fragment_area, contour_area, places=8)

    def test_edge_on_plane_becomes_one_finite_near_outline_chain(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        edge_on = SectionPlane(
            "edge-on",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            u_axis=(0.0, 0.0, 1.0),
        )
        edge_patch = fit_plane_display_patch("edge", edge_on, (sphere,)).patch
        frame = compute_quadric_section_compositing(
            _base_frame(sphere),
            sphere,
            edge_on,
            edge_patch,
            IDENTITY_VIEW,
        )

        self.assertIs(frame.projection_kind, PlanePatchProjectionKind.LINE)
        self.assertFalse(frame.has_plane_fill)
        self.assertEqual(frame.plane_fragments, ())
        self.assertAlmostEqual(
            frame.patch_projection.singular_values[0],
            edge_patch.half_height,
            places=12,
        )
        self.assertEqual(frame.patch_projection.singular_values[1], 0.0)
        self.assertEqual(frame.patch_projection.rank_ratio, 0.0)
        self.assertEqual(
            frame.to_dict()["patchProjection"]["kind"],  # type: ignore[index]
            "line",
        )
        self.assertTrue(frame.plane_outline_fragments)
        # Positive depth points towards the observer.  Of the two coincident
        # long patch edges, only the z=+half_width edge is retained.
        self.assertEqual(
            {item.edge_index for item in frame.plane_outline_fragments},
            {1},
        )
        self.assertTrue(
            all(
                abs(item.world_start[2] - edge_patch.half_width) <= 1.0e-12
                and abs(item.world_end[2] - edge_patch.half_width) <= 1.0e-12
                for item in frame.plane_outline_fragments
            )
        )

        projected_length = sum(
            float(
                np.linalg.norm(
                    np.asarray(item.screen_end) - np.asarray(item.screen_start)
                )
            )
            for item in frame.plane_outline_fragments
        )
        endpoints = np.asarray(
            (
                frame.patch_projection.line_screen_start,
                frame.patch_projection.line_screen_end,
            ),
            dtype=float,
        )
        self.assertAlmostEqual(
            projected_length,
            float(np.linalg.norm(endpoints[1] - endpoints[0])),
            places=12,
        )
        self.assertTrue(
            all(not loops for loops in quadric_plane_fragment_contours(frame).values())
        )
        repeated = compute_quadric_section_compositing(
            _base_frame(sphere),
            sphere,
            edge_on,
            edge_patch,
            IDENTITY_VIEW,
        )
        self.assertEqual(
            canonical_quadric_section_compositing_json(frame),
            canonical_quadric_section_compositing_json(repeated),
        )
        reverse_view = ParallelView.from_matrix(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
        )
        reverse_proxy = build_opaque_projection_proxy(
            sphere,
            reverse_view,
            max_chord_error=0.01,
        )
        reverse_base = compute_quadric_compositing(
            compute_quadric_visibility((), (sphere,), reverse_view),
            (reverse_proxy,),
        )
        reverse = compute_quadric_section_compositing(
            reverse_base,
            sphere,
            edge_on,
            edge_patch,
            reverse_view,
        )
        self.assertEqual(
            reverse.patch_projection.line_screen_start,
            frame.patch_projection.line_screen_start,
        )
        self.assertEqual(
            reverse.patch_projection.line_screen_end,
            frame.patch_projection.line_screen_end,
        )
        self.assertEqual(
            {item.edge_index for item in reverse.plane_outline_fragments},
            {3},
        )
        self.assertTrue(
            all(
                abs(item.world_start[2] + edge_patch.half_width) <= 1.0e-12
                and abs(item.world_end[2] + edge_patch.half_width) <= 1.0e-12
                for item in reverse.plane_outline_fragments
            )
        )

    def test_projection_rank_evidence_uses_finite_patch_extents(self) -> None:
        plane = SectionPlane(
            "aspect-sensitive",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, -1.0e-13),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = PlaneDisplayPatchSpec(
            "aspect-sensitive-patch",
            plane.plane_id,
            1.0,
            1.0e6,
        )
        plane_u, plane_v, _normal = plane.basis
        unit_screen_basis = np.column_stack(
            (
                IDENTITY_VIEW.matrix[:2] @ plane_u,
                IDENTITY_VIEW.matrix[:2] @ plane_v,
            )
        )
        scaled = unit_screen_basis @ np.diag(
            (patch.half_width, patch.half_height)
        )
        evidence, _axis = _plane_patch_projection_evidence(
            scaled,
            np.asarray(patch.corners(plane), dtype=float),
            IDENTITY_VIEW,
        )
        self.assertIs(evidence.kind, PlanePatchProjectionKind.AREA)
        self.assertGreater(evidence.rank_ratio, evidence.rank_ratio_threshold)
        self.assertAlmostEqual(evidence.rank_ratio, 1.0e-7, delta=1.0e-18)

        # Swapping the two authored plane axes together with their finite
        # extents only permutes the scaled columns and must not change rank.
        swapped, _axis = _plane_patch_projection_evidence(
            scaled[:, ::-1],
            np.asarray(patch.corners(plane), dtype=float),
            IDENTITY_VIEW,
        )
        self.assertIs(swapped.kind, PlanePatchProjectionKind.AREA)
        np.testing.assert_allclose(
            swapped.singular_values,
            evidence.singular_values,
            atol=0.0,
            rtol=1.0e-15,
        )
        self.assertAlmostEqual(swapped.rank_ratio, evidence.rank_ratio)
        sphere = SphereSpec("aspect-sensitive-sphere", (0.0, 0.0, 0.0), 1.0)
        with self.assertRaisesRegex(
            QuadricSectionCompositingError,
            "extreme aspect ratio.*unit plane projection is numerically rank-one",
        ):
            compute_quadric_section_compositing(
                _base_frame(sphere),
                sphere,
                plane,
                patch,
                IDENTITY_VIEW,
            )

        thin_plane = SectionPlane(
            "aspect-thin",
            (0.0, 0.0, 0.3),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        thin_patch = PlaneDisplayPatchSpec(
            "aspect-thin-patch",
            thin_plane.plane_id,
            1.0e-13,
            1.2,
        )
        with self.assertRaisesRegex(
            QuadricSectionCompositingError,
            "extreme thin aspect ratio.*unit plane projection retains area",
        ):
            compute_quadric_section_compositing(
                _base_frame(sphere),
                sphere,
                thin_plane,
                thin_patch,
                IDENTITY_VIEW,
            )

    def test_area_plane_capacity_overflow_still_fails_closed(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        tilted = SectionPlane(
            "tilted",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        tilted_patch = fit_plane_display_patch("tilted", tilted, (sphere,)).patch
        with self.assertRaisesRegex(
            QuadricSectionCompositingError,
            "more than 8 plane fragments",
        ):
            compute_quadric_section_compositing(
                _base_frame(sphere),
                sphere,
                tilted,
                tilted_patch,
                IDENTITY_VIEW,
                limits=QuadricSectionCompositingLimits(
                    minimum_subdivision_depth=0,
                    maximum_subdivision_depth=10,
                    max_plane_fragments=8,
                    max_ray_classifications=4096,
                ),
            )


class QuadricSectionBoundaryPartitionContractTests(unittest.TestCase):
    """Renderer-neutral contract for the boundary-conforming repair."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.frames = {
            case.name: _contract_frame(case)
            for case in SECTION_PARTITION_CASES
        }

    def test_required_case_matrix_has_the_intended_finite_sections(self) -> None:
        for case in SECTION_PARTITION_CASES:
            with self.subTest(case=case.name):
                trace = compute_quadric_section(
                    f"{case.name}-trace",
                    case.surface,
                    case.plane,
                )
                self.assertEqual(trace.supporting_kind.value, case.expected_kind)
                self.assertEqual(
                    trace.finite_topology.value,
                    case.expected_topology,
                )

    def test_common_sections_fit_the_original_fixed_capacity(self) -> None:
        self.assertEqual(
            QUADRIC_SECTION_COMPOSITING_LIMITS.max_plane_fragments,
            8192,
        )
        self.assertEqual(
            QUADRIC_SECTION_COMPOSITING_LIMITS.max_ray_classifications,
            65536,
        )
        for case in SECTION_PARTITION_CASES:
            with self.subTest(case=case.name):
                frame = self.frames[case.name]
                self.assertIs(
                    frame.projection_kind,
                    PlanePatchProjectionKind.AREA,
                )
                self.assertTrue(frame.has_plane_fill)
                self.assertGreater(
                    frame.patch_projection.rank_ratio,
                    frame.patch_projection.rank_ratio_threshold,
                )
                self.assertLessEqual(
                    len(frame.plane_fragments),
                    QUADRIC_SECTION_COMPOSITING_LIMITS.max_plane_fragments,
                )
                self.assertLessEqual(
                    frame.ray_classification_count,
                    QUADRIC_SECTION_COMPOSITING_LIMITS.max_ray_classifications,
                )

    def test_fragments_exactly_partition_the_display_patch(self) -> None:
        failures: list[str] = []
        for case in SECTION_PARTITION_CASES:
            frame = self.frames[case.name]
            linear_tolerance, _boundary, area_tolerance = _case_tolerances(
                case,
                frame,
            )
            patch = _project_patch(frame)
            patch_area = _polygon_area(patch)
            degenerate_area_tolerance = max(
                linear_tolerance * linear_tolerance,
                np.finfo(float).tiny,
            )
            fragment_area = 0.0
            issue_count = 0
            examples: list[str] = []
            for fragment in frame.plane_fragments:
                triangle = _counter_clockwise(fragment.screen_vertices)
                triangle_area = _polygon_area(triangle)
                fragment_area += triangle_area
                if triangle_area <= degenerate_area_tolerance:
                    issue_count += 1
                    examples.append(
                        f"{fragment.fragment_id} has no stable positive area"
                    )
                    continue
                clipped = _clip_convex_polygon_for_contract(
                    triangle,
                    patch,
                    linear_tolerance,
                )
                outside_area = max(
                    0.0,
                    triangle_area - _polygon_area(clipped),
                )
                if outside_area > area_tolerance:
                    issue_count += 1
                    examples.append(
                        f"{fragment.fragment_id} extends outside patch by "
                        f"{outside_area:.12g}"
                    )
            area_error = abs(fragment_area - patch_area)
            if area_error > area_tolerance:
                issue_count += 1
                examples.append(
                    f"fragment area {fragment_area:.12g} differs from patch "
                    f"area {patch_area:.12g} by {area_error:.12g}"
                )
            topology_issues = _partition_topology_issues(
                frame.plane_fragments,
                patch,
                linear_tolerance=linear_tolerance,
                area_tolerance=area_tolerance,
            )
            issue_count += len(topology_issues)
            examples.extend(topology_issues)
            if len(frame.plane_fragments) <= 4096:
                overlap_issues = _positive_overlap_issues(
                    frame.plane_fragments,
                    linear_tolerance=linear_tolerance,
                    area_tolerance=area_tolerance,
                )
                issue_count += len(overlap_issues)
                examples.extend(overlap_issues)
            if issue_count:
                failures.append(
                    _case_issue_summary(
                        case.name,
                        issue_count,
                        examples,
                    )
                )
        self.assertFalse(
            failures,
            "fragments must cover the patch once, without gaps, overlap, or "
            "overflow:\n" + "\n".join(failures),
        )

    def test_fragment_proxy_membership_matches_outside_role(self) -> None:
        failures: list[str] = []
        for case in SECTION_PARTITION_CASES:
            frame = self.frames[case.name]
            linear_tolerance, _boundary, _area = _case_tolerances(case, frame)
            proxy = _counter_clockwise(frame.surface_proxy.vertices)
            issue_count = 0
            examples: list[str] = []
            for fragment in frame.plane_fragments:
                stable_sample_count = 0
                expected_inside = (
                    fragment.role is not PlaneDepthRole.OUTSIDE_PROJECTION
                )
                for label, _world, screen in _fragment_samples(fragment):
                    margin = _convex_signed_margin(screen, proxy)
                    if abs(margin) <= linear_tolerance:
                        continue
                    stable_sample_count += 1
                    observed_inside = margin > 0.0
                    if observed_inside != expected_inside:
                        issue_count += 1
                        examples.append(
                            f"{fragment.fragment_id}/{label} role="
                            f"{fragment.role.value} proxyMargin={margin:.12g}"
                        )
                if stable_sample_count == 0:
                    issue_count += 1
                    examples.append(
                        f"{fragment.fragment_id} has no stable proxy sample"
                    )
            if issue_count:
                failures.append(
                    _case_issue_summary(
                        case.name,
                        issue_count,
                        examples,
                    )
                )
        self.assertFalse(
            failures,
            "non-outside fragments must stay inside the projection proxy, and "
            "outside fragments must stay outside it:\n" + "\n".join(failures),
        )

    def test_fragment_roles_match_authoritative_ray_classifier(self) -> None:
        failures: list[str] = []
        for case in SECTION_PARTITION_CASES:
            frame = self.frames[case.name]
            characteristic = tuple(case.surface.characteristic_points) + tuple(
                frame.patch.corners(frame.plane)
            )
            context = GeometryContext().resolve(characteristic)
            boundary_epsilon = context.epsilon(GeometryQuantity.BOUNDARY)
            solver = _surface_ray_solver(
                case.surface,
                np.asarray(OBLIQUE_VIEW.view_direction, dtype=float),
                boundary_epsilon=boundary_epsilon,
                angular_epsilon=context.epsilon(GeometryQuantity.ANGULAR),
            )
            issue_count = 0
            examples: list[str] = []
            for fragment in frame.plane_fragments:
                if fragment.role is PlaneDepthRole.OUTSIDE_PROJECTION:
                    # Batch 3 makes the finite polygonal proxy authoritative
                    # for outside ownership.  The analytic surface may extend
                    # slightly beyond that chordal proxy near its silhouette.
                    continue
                stable_sample_count = 0
                for label, world, _screen in _fragment_samples(fragment):
                    observed = _stable_ray_role(
                        solver,
                        world,
                        frame.plane,
                        boundary_epsilon,
                        # The emitted boundary is a certified screen-space
                        # tangent envelope.  Samples inside its tiny error band
                        # are legitimate boundary samples, not stable evidence
                        # for either neighboring role.  This remains far below
                        # the public max_screen_error and still catches any
                        # fragment that genuinely crosses a role region.
                        geometric_boundary_tolerance=(
                            2.0
                            * frame.max_screen_error
                            / _SECTION_BOUNDARY_CHORD_DIVISOR
                        ),
                    )
                    if observed is None:
                        continue
                    stable_sample_count += 1
                    if observed is not fragment.role:
                        issue_count += 1
                        examples.append(
                            f"{fragment.fragment_id}/{label} stored="
                            f"{fragment.role.value} classified={observed.value}"
                        )
                if stable_sample_count == 0:
                    # A triangle may lie wholly inside the certified tangent-
                    # envelope band.  Do not ignore that fragment: use
                    # its strict interior centroid as the deterministic
                    # fallback and still require the authoritative solver to
                    # agree with the stored role.
                    centroid = np.mean(
                        np.asarray(fragment.world_vertices, dtype=float),
                        axis=0,
                    )
                    observed = _ray_role(
                        solver,
                        centroid,
                        boundary_epsilon,
                    )
                    if observed is not fragment.role:
                        issue_count += 1
                        examples.append(
                            f"{fragment.fragment_id}/centroid-fallback stored="
                            f"{fragment.role.value} classified={observed.value}"
                        )
            if issue_count:
                failures.append(
                    _case_issue_summary(
                        case.name,
                        issue_count,
                        examples,
                    )
                )
        self.assertFalse(
            failures,
            "every stable vertex, edge midpoint, centroid, and interior sample "
            "must agree with its fragment role:\n" + "\n".join(failures),
        )

    def test_boundary_cell_emits_classified_overlap_not_full_triangle(self) -> None:
        case = next(
            item
            for item in SECTION_PARTITION_CASES
            if item.name == "sphere_oblique"
        )
        def crossing_records(frame) -> list[str]:
            linear_tolerance, _boundary, area_tolerance = _case_tolerances(
                case,
                frame,
            )
            proxy = _counter_clockwise(frame.surface_proxy.vertices)
            records: list[str] = []
            for fragment in frame.plane_fragments:
                triangle = _counter_clockwise(fragment.screen_vertices)
                triangle_area = _polygon_area(triangle)
                overlap_area = _polygon_area(
                    _clip_convex_polygon_for_contract(
                        triangle,
                        proxy,
                        linear_tolerance,
                    )
                )
                if fragment.role is PlaneDepthRole.OUTSIDE_PROJECTION:
                    invalid_area = overlap_area
                else:
                    invalid_area = max(0.0, triangle_area - overlap_area)
                if invalid_area > area_tolerance:
                    records.append(
                        f"{fragment.fragment_id} role={fragment.role.value} "
                        f"triangleArea={triangle_area:.12g} "
                        f"overlapArea={overlap_area:.12g} "
                        f"misownedArea={invalid_area:.12g}"
                    )
            return records

        crossing = crossing_records(
            _contract_frame(case, max_screen_error=0.8)
        )
        repeated = crossing_records(
            _contract_frame(case, max_screen_error=0.8)
        )
        self.assertEqual(crossing, repeated)
        self.assertFalse(
            crossing,
            "boundary classification uses the clipped overlap, but the frame "
            "emits the full source triangle:\n" + "\n".join(crossing[:8]),
        )

    def test_all_role_contours_close_and_preserve_fragment_area(self) -> None:
        for case in SECTION_PARTITION_CASES:
            with self.subTest(case=case.name):
                frame = self.frames[case.name]
                contours = quadric_plane_fragment_contours(frame)
                for role in PlaneDepthRole:
                    source_area = sum(
                        _screen_signed_area(item.screen_vertices)
                        for item in frame.fragments_by_role[role]
                    )
                    contour_area = sum(
                        _screen_signed_area(loop) for loop in contours[role]
                    )
                    self.assertAlmostEqual(source_area, contour_area, places=9)
                    self.assertTrue(
                        all(len(loop) >= 3 for loop in contours[role])
                    )

    def test_outline_four_role_partition_remains_frozen(self) -> None:
        observed: dict[str, str] = {}
        for case in SECTION_PARTITION_CASES:
            # Freeze the semantic edge/role partition at the geometry
            # tolerance scale.  Hashing raw derived world/screen floats makes
            # this contract depend on platform-specific BLAS root noise.
            payload = json.dumps(
                [
                    [
                        item.fragment_id,
                        item.role.value,
                        item.edge_index,
                        round(item.interval.start, 8),
                        round(item.interval.end, 8),
                    ]
                    for item in self.frames[case.name].plane_outline_fragments
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            observed[case.name] = sha256(payload).hexdigest()
        self.assertEqual(observed, _FROZEN_OUTLINE_DIGESTS)

    def test_required_case_partition_is_deterministic(self) -> None:
        for case in SECTION_PARTITION_CASES:
            with self.subTest(case=case.name):
                first = self.frames[case.name]
                second = _contract_frame(case)
                self.assertEqual(first.plane_fragments, second.plane_fragments)
                self.assertEqual(
                    first.plane_outline_fragments,
                    second.plane_outline_fragments,
                )
                self.assertEqual(first.draw_order, second.draw_order)
                self.assertEqual(
                    canonical_quadric_section_compositing_json(first),
                    canonical_quadric_section_compositing_json(second),
                )
                self.assertEqual(
                    quadric_plane_fragment_contours(first),
                    quadric_plane_fragment_contours(second),
                )


if __name__ == "__main__":
    unittest.main()
