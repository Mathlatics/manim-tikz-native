from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Sequence
import unittest

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.geometry import GeometryContext, GeometryQuantity
from polyhedron_visibility.quadrics.compositing import (
    QuadricPaintPolicy,
    compute_quadric_compositing,
)
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    SectionPlane,
    SphereSpec,
    CylinderSpec,
)
from polyhedron_visibility.quadrics.curves import CircleArcCurve
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.projection import build_opaque_projection_proxy
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    QuadricSectionCompositingError,
    QuadricSectionCompositingLimits,
    canonical_quadric_section_compositing_json,
    compute_quadric_section_compositing,
    quadric_plane_fragment_contours,
    _surface_ray_solver,
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
    records.sort(key=lambda item: (item[0], item[4]))
    active: list[tuple] = []
    issues: list[str] = []
    for record in records:
        minimum_x, maximum_x, minimum_y, maximum_y, fragment_id, polygon = record
        active = [
            candidate
            for candidate in active
            if candidate[1] >= minimum_x - linear_tolerance
        ]
        for candidate in active:
            if (
                candidate[3] < minimum_y - linear_tolerance
                or maximum_y < candidate[2] - linear_tolerance
            ):
                continue
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
        active.append(record)
    return tuple(issues)


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
) -> PlaneDepthRole | None:
    value = np.asarray(point, dtype=float)
    plane_u, plane_v, _normal = plane.basis
    probe = 16.0 * boundary_epsilon
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


class QuadricSectionCompositingTests(unittest.TestCase):
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

    def test_edge_on_plane_and_capacity_overflow_fail_closed(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        edge_on = SectionPlane(
            "edge-on",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            u_axis=(0.0, 0.0, 1.0),
        )
        edge_patch = fit_plane_display_patch("edge", edge_on, (sphere,)).patch
        with self.assertRaisesRegex(
            QuadricSectionCompositingError,
            "projects edge-on",
        ):
            compute_quadric_section_compositing(
                _base_frame(sphere),
                sphere,
                edge_on,
                edge_patch,
                IDENTITY_VIEW,
            )

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
            fragment_area = 0.0
            issue_count = 0
            examples: list[str] = []
            for fragment in frame.plane_fragments:
                triangle = _counter_clockwise(fragment.screen_vertices)
                triangle_area = _polygon_area(triangle)
                fragment_area += triangle_area
                if triangle_area <= area_tolerance:
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
                stable_sample_count = 0
                for label, world, _screen in _fragment_samples(fragment):
                    observed = _stable_ray_role(
                        solver,
                        world,
                        frame.plane,
                        boundary_epsilon,
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
                    issue_count += 1
                    examples.append(
                        f"{fragment.fragment_id} has no stable ray sample"
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
