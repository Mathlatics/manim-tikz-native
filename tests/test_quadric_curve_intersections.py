from __future__ import annotations

import json
from math import acosh, cos, pi, sin, tau
import unittest
from unittest.mock import patch

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.geometry import GeometryContext
from polyhedron_visibility.quadrics.conics import ConicKind, ConicParameterization
from polyhedron_visibility.quadrics.contract import ConeModel, ConeSpec
from polyhedron_visibility.quadrics.curve_intersections import (
    ProjectedCurveIntersectionError,
    canonical_projected_curve_crossings_json,
    compute_projected_curve_crossings,
)
from polyhedron_visibility.quadrics.curves import (
    CircleArcCurve,
    EllipseArcCurve,
    ParametricConicBranch,
    SegmentCurve,
)
from polyhedron_visibility.quadrics.surface_boundaries import (
    build_surface_boundary_sources,
)
from polyhedron_visibility.topology import ParameterInterval


IDENTITY_VIEW = ParallelView.from_matrix(
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)
OBLIQUE_VIEW = ParallelView.from_matrix(
    ((1.0, 0.0, 0.35), (0.0, 1.0, 0.2), (0.0, 0.0, 1.0))
)
GENERAL_VIEW_MATRIX = np.asarray(
    ((1.0, 0.2, 0.3), (-0.1, 1.0, 0.15), (0.07, -0.12, 1.0)),
    dtype=float,
)
GENERAL_VIEW = ParallelView.from_matrix(GENERAL_VIEW_MATRIX)
EDGE_ON_CIRCLE_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)
GENERAL_DEPTH_OFFSET = np.linalg.solve(
    GENERAL_VIEW_MATRIX,
    np.asarray((0.0, 0.0, 1.0), dtype=float),
)
PLANE_EMBEDDING = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
)


class ProjectedSegmentCrossingTests(unittest.TestCase):
    def test_disjoint_screen_bounds_skip_the_analytic_pair_solver(self) -> None:
        left = SegmentCurve("left", (-3.0, -1.0, 0.0), (-3.0, 1.0, 0.0))
        right = SegmentCurve("right", (3.0, -1.0, 0.0), (3.0, 1.0, 0.0))

        with patch(
            "polyhedron_visibility.quadrics.curve_intersections."
            "_candidate_source_parameters",
            side_effect=AssertionError("disjoint pair entered the root solver"),
        ):
            crossings = compute_projected_curve_crossings(
                (right, left), IDENTITY_VIEW
            )

        self.assertEqual(crossings, ())

    def test_crossing_segments_record_objective_far_to_near_order(self) -> None:
        farther = SegmentCurve("far", (-1.0, 0.0, -2.0), (1.0, 0.0, -2.0))
        nearer = SegmentCurve("near", (0.0, -1.0, 3.0), (0.0, 1.0, 3.0))

        crossings = compute_projected_curve_crossings(
            (nearer, farther), IDENTITY_VIEW
        )
        self.assertEqual(len(crossings), 1)
        crossing = crossings[0]
        self.assertEqual((crossing.first_curve_id, crossing.second_curve_id), ("far", "near"))
        self.assertEqual((crossing.far_curve_id, crossing.near_curve_id), ("far", "near"))
        self.assertAlmostEqual(crossing.first_parameter, 0.5)
        self.assertAlmostEqual(crossing.second_parameter, 0.5)
        self.assertEqual(crossing.screen_point, (0.0, 0.0))
        self.assertFalse(crossing.tangential)

    def test_true_three_dimensional_intersection_has_no_painter_edge(self) -> None:
        horizontal = SegmentCurve("horizontal", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        vertical = SegmentCurve("vertical", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))
        crossing = compute_projected_curve_crossings(
            (horizontal, vertical), IDENTITY_VIEW
        )[0]
        self.assertTrue(crossing.coincident_depth)
        self.assertIsNone(crossing.far_curve_id)
        self.assertIsNone(crossing.near_curve_id)

    def test_large_common_depth_translation_keeps_resolvable_order(self) -> None:
        farther = SegmentCurve(
            "a",
            (-1.0, 0.0, 1.0e8),
            (1.0, 0.0, 1.0e8),
        )
        nearer = SegmentCurve(
            "b",
            (0.0, -1.0, 1.0e8 + 1.0e-6),
            (0.0, 1.0, 1.0e8 + 1.0e-6),
        )

        crossing = compute_projected_curve_crossings(
            (nearer, farther), IDENTITY_VIEW
        )[0]

        self.assertEqual((crossing.far_curve_id, crossing.near_curve_id), ("a", "b"))
        self.assertGreater(crossing.second_depth - crossing.first_depth, 0.0)

    def test_crossing_contract_rejects_painter_order_opposite_to_depths(self) -> None:
        from polyhedron_visibility.quadrics.curve_intersections import (
            ProjectedCurveCrossing,
        )

        with self.assertRaisesRegex(
            ProjectedCurveIntersectionError,
            "disagree with crossing depths",
        ):
            ProjectedCurveCrossing(
                crossing_id="crossing:a:b:0",
                first_curve_id="a",
                second_curve_id="b",
                first_parameter=0.5,
                second_parameter=0.5,
                screen_point=(0.0, 0.0),
                first_depth=10.0,
                second_depth=0.0,
                far_curve_id="a",
                near_curve_id="b",
            )

    def test_shallow_real_crossing_is_not_scale_dependent_tangency(self) -> None:
        for scale in (1.0e-6, 1.0, 1.0e6):
            with self.subTest(scale=scale):
                first = SegmentCurve(
                    "first",
                    (-scale, 0.0, -scale),
                    (scale, 0.0, -scale),
                )
                second = SegmentCurve(
                    "second",
                    (-scale, -0.01 * scale, scale),
                    (scale, 0.01 * scale, scale),
                )
                crossing = compute_projected_curve_crossings(
                    (first, second), IDENTITY_VIEW
                )[0]
                self.assertFalse(crossing.tangential)

    def test_coincident_projected_segments_fail_closed(self) -> None:
        first = SegmentCurve("first", (-1.0, 0.0, -1.0), (1.0, 0.0, -1.0))
        second = SegmentCurve("second", (-1.0, 0.0, 1.0), (1.0, 0.0, 1.0))
        with self.assertRaisesRegex(
            ProjectedCurveIntersectionError, "infinitely many"
        ):
            compute_projected_curve_crossings((first, second), IDENTITY_VIEW)

    def test_active_paint_intervals_precede_coincident_support_classification(self) -> None:
        first = SegmentCurve("first", (-1.0, 0.0, -1.0), (1.0, 0.0, -1.0))
        second = SegmentCurve("second", (-1.0, 0.0, 1.0), (1.0, 0.0, 1.0))
        self.assertEqual(
            compute_projected_curve_crossings(
                (first, second),
                IDENTITY_VIEW,
                active_intervals={"first": (), "second": ()},
            ),
            (),
        )
        self.assertEqual(
            compute_projected_curve_crossings(
                (first, second),
                IDENTITY_VIEW,
                active_intervals={
                    "first": (ParameterInterval(0.0, 0.4),),
                    "second": (ParameterInterval(0.6, 1.0),),
                },
            ),
            (),
        )
        point = compute_projected_curve_crossings(
            (first, second),
            IDENTITY_VIEW,
            active_intervals={
                "first": (ParameterInterval(0.0, 0.5),),
                "second": (ParameterInterval(0.5, 1.0),),
            },
        )
        self.assertEqual(len(point), 1)
        with self.assertRaisesRegex(
            ProjectedCurveIntersectionError, "infinitely many"
        ):
            compute_projected_curve_crossings(
                (first, second),
                IDENTITY_VIEW,
                active_intervals={
                    "first": (ParameterInterval(0.0, 0.7),),
                    "second": (ParameterInterval(0.3, 1.0),),
                },
            )

    def test_active_interval_uses_local_depth_feature_for_painter_order(self) -> None:
        first = SegmentCurve(
            "first",
            (-1.0, 0.0, -1.0e9),
            (1.0, 0.0, 1.0e9),
        )
        second = SegmentCurve(
            "second",
            (0.0, -1.0, 1.0),
            (0.0, 1.0, 1.0),
        )
        crossings = compute_projected_curve_crossings(
            (first, second),
            IDENTITY_VIEW,
            active_intervals={
                "first": (ParameterInterval(0.499999, 0.500001),),
            },
        )
        self.assertEqual(len(crossings), 1)
        self.assertEqual(
            (crossings[0].far_curve_id, crossings[0].near_curve_id),
            ("first", "second"),
        )

    def test_split_active_intervals_dedupe_a_closed_curve_seam(self) -> None:
        circle = CircleArcCurve(
            "circle",
            (0.0, 0.0, 0.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        tangent = SegmentCurve(
            "tangent", (1.0, -2.0, 1.0), (1.0, 2.0, 1.0)
        )
        crossings = compute_projected_curve_crossings(
            (circle, tangent),
            IDENTITY_VIEW,
            active_intervals={
                "circle": (
                    ParameterInterval(0.0, 0.5),
                    ParameterInterval(5.5, tau),
                )
            },
        )
        self.assertEqual(len(crossings), 1)

    def test_collinear_finite_domains_distinguish_empty_point_and_overlap(self) -> None:
        first = SegmentCurve("first", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        disjoint = SegmentCurve("disjoint", (2.0, 0.0, 1.0), (3.0, 0.0, 1.0))
        touching = SegmentCurve("touching", (1.0, 0.0, 1.0), (2.0, 0.0, 1.0))
        overlapping = SegmentCurve(
            "overlapping", (0.5, 0.0, 1.0), (1.5, 0.0, 1.0)
        )

        self.assertEqual(
            compute_projected_curve_crossings((first, disjoint), IDENTITY_VIEW),
            (),
        )
        point = compute_projected_curve_crossings((first, touching), IDENTITY_VIEW)
        self.assertEqual(len(point), 1)
        self.assertEqual(point[0].screen_point, (1.0, 0.0))
        with self.assertRaisesRegex(
            ProjectedCurveIntersectionError, "positive-length interval"
        ):
            compute_projected_curve_crossings((first, overlapping), IDENTITY_VIEW)

    def test_tiny_collinear_endpoint_contact_survives_large_translation(self) -> None:
        origin = np.asarray((1.0e5, 1.0e5, 0.0))
        step = np.asarray((1.0e-6, 0.0, 0.0))
        first = SegmentCurve("first", tuple(origin), tuple(origin + step))
        second = SegmentCurve(
            "second",
            tuple(origin + step + np.asarray((0.0, 0.0, 1.0))),
            tuple(origin + 2.0 * step + np.asarray((0.0, 0.0, 1.0))),
        )
        crossings = compute_projected_curve_crossings(
            (first, second), IDENTITY_VIEW
        )
        self.assertEqual(len(crossings), 1)

    def test_reversed_target_endpoint_contact_does_not_mutate_projected_model(self) -> None:
        origin = np.asarray((1.0e5, 1.0e5, 0.0))
        step = 1.0e-6
        first = SegmentCurve(
            "first",
            tuple(origin),
            tuple(origin + np.asarray((step, 0.0, 0.0))),
        )
        # Under OBLIQUE_VIEW the z=1 translation contributes (+0.35,+0.2).
        # Compensate that shift and author the second segment in reverse, so
        # the one shared projected point is specifically target.domain.end.
        second = SegmentCurve(
            "second",
            (origin[0] + 2.0 * step - 0.35, origin[1] - 0.2, 1.0),
            (origin[0] + step - 0.35, origin[1] - 0.2, 1.0),
        )
        crossings = compute_projected_curve_crossings(
            (first, second), OBLIQUE_VIEW
        )
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0].second_parameter, 1.0)


class ProjectedConicCrossingTests(unittest.TestCase):
    @staticmethod
    def _rotated_axes(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        angle = 0.7
        first_direction = np.asarray((cos(angle), sin(angle), 0.0), dtype=float)
        second_direction = np.asarray((-sin(angle), cos(angle), 0.0), dtype=float)
        center = np.asarray((12.3, -45.6, 0.7), dtype=float) * scale
        return center, 0.7 * scale * first_direction, 1.3 * scale * second_direction

    def test_circle_and_line_have_two_exact_crossings(self) -> None:
        circle = CircleArcCurve(
            "circle",
            (0.0, 0.0, 0.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        line = SegmentCurve("line", (-2.0, 0.0, 2.0), (2.0, 0.0, 2.0))
        crossings = compute_projected_curve_crossings((line, circle), IDENTITY_VIEW)
        self.assertEqual(len(crossings), 2)
        self.assertAlmostEqual(crossings[0].first_parameter, 0.0)
        self.assertAlmostEqual(crossings[1].first_parameter, pi)
        self.assertEqual(
            {item.near_curve_id for item in crossings},
            {"line"},
        )

    def test_edge_on_circle_and_transverse_segment_keep_both_crossings(self) -> None:
        circle = CircleArcCurve(
            "edge-on-circle",
            (0.0, 0.0, 2.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        segment = SegmentCurve(
            "plane-edge",
            (0.0, 0.5, 1.5),
            (0.0, 0.5, 2.5),
        )

        crossings = compute_projected_curve_crossings(
            (circle, segment), EDGE_ON_CIRCLE_VIEW
        )

        self.assertEqual(len(crossings), 2)
        self.assertAlmostEqual(crossings[0].first_parameter, pi / 2.0)
        self.assertAlmostEqual(crossings[1].first_parameter, 3.0 * pi / 2.0)
        self.assertEqual(
            tuple(item.second_parameter for item in crossings),
            (0.5, 0.5),
        )
        self.assertEqual(
            {(item.far_curve_id, item.near_curve_id) for item in crossings},
            {
                ("edge-on-circle", "plane-edge"),
                ("plane-edge", "edge-on-circle"),
            },
        )

    def test_edge_on_circle_support_endpoint_is_a_certified_turn(self) -> None:
        circle = CircleArcCurve(
            "edge-on-circle",
            (0.0, 0.0, 2.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        segment = SegmentCurve(
            "endpoint-edge",
            (1.0, 0.5, 1.5),
            (1.0, 0.5, 2.5),
        )

        crossings = compute_projected_curve_crossings(
            (circle, segment), EDGE_ON_CIRCLE_VIEW
        )

        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0].first_parameter, 0.0)
        self.assertAlmostEqual(crossings[0].second_parameter, 0.5)
        self.assertTrue(crossings[0].tangential)

    def test_tiny_nondegenerate_circle_projection_is_not_singular(self) -> None:
        scale = 1.0e-7
        circle = CircleArcCurve(
            "tiny-circle",
            (0.0, 0.0, 0.0),
            scale,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        line = SegmentCurve(
            "tiny-line",
            (-2.0 * scale, 0.0, scale),
            (2.0 * scale, 0.0, scale),
        )
        self.assertEqual(
            len(compute_projected_curve_crossings((circle, line), IDENTITY_VIEW)),
            2,
        )

    def test_chart_pole_tangency_is_stable_across_similarity_scale(self) -> None:
        for scale in (1.0e-6, 1.0, 1.0e6, 1.0e12):
            with self.subTest(scale=scale):
                circle = CircleArcCurve(
                    "circle",
                    (3.0 * scale, 5.0 * scale, 0.0),
                    scale,
                    (0.0, 0.0, 1.0),
                    radial_axis=(1.0, 0.0, 0.0),
                )
                line = SegmentCurve(
                    "line",
                    (2.0 * scale, 3.0 * scale, scale),
                    (2.0 * scale, 7.0 * scale, scale),
                )
                crossings = compute_projected_curve_crossings(
                    (circle, line), IDENTITY_VIEW
                )
                self.assertEqual(len(crossings), 1)
                self.assertAlmostEqual(crossings[0].first_parameter, pi)
                self.assertTrue(crossings[0].tangential)

    def test_partial_circle_domain_filters_the_other_intersection(self) -> None:
        circle = CircleArcCurve(
            "arc",
            (0.0, 0.0, 0.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
            domain=ParameterInterval(-1.0, 1.0),
        )
        line = SegmentCurve("line", (-2.0, 0.0, 1.0), (2.0, 0.0, 1.0))
        crossings = compute_projected_curve_crossings((circle, line), IDENTITY_VIEW)
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0].first_parameter, 0.0)

    def test_tangent_line_preserves_even_root_and_tangent_evidence(self) -> None:
        circle = CircleArcCurve(
            "circle",
            (0.0, 0.0, 0.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        tangent = SegmentCurve("tangent", (-2.0, 1.0, 1.0), (2.0, 1.0, 1.0))
        crossings = compute_projected_curve_crossings((circle, tangent), IDENTITY_VIEW)
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0].first_parameter, pi / 2.0)
        self.assertTrue(crossings[0].tangential)

    def test_rotated_oblique_ellipse_tangent_is_one_certified_crossing(self) -> None:
        for scale in (1.0e-3, 1.0, 1.0e3):
            with self.subTest(scale=scale):
                center, first_axis, second_axis = self._rotated_axes(scale)
                ellipse = EllipseArcCurve(
                    "ellipse",
                    tuple(center),
                    tuple(first_axis),
                    tuple(second_axis),
                )
                tangent = SegmentCurve(
                    "line",
                    tuple(
                        center
                        + second_axis
                        - 2.0 * first_axis
                        + scale * GENERAL_DEPTH_OFFSET
                    ),
                    tuple(
                        center
                        + second_axis
                        + 2.0 * first_axis
                        + scale * GENERAL_DEPTH_OFFSET
                    ),
                )

                crossings = compute_projected_curve_crossings(
                    (ellipse, tangent), GENERAL_VIEW
                )

                self.assertEqual(len(crossings), 1)
                self.assertAlmostEqual(crossings[0].first_parameter, pi / 2.0)
                self.assertTrue(crossings[0].tangential)

    def test_translated_small_ellipse_tangent_keeps_tangent_evidence(self) -> None:
        parameter = 1.25
        for center, first_length, second_length, depth_offset in (
            ((10.0, -4.0, 0.0), 0.004, 0.008, 0.0037),
            ((1000.0, -400.0, 0.0), 0.4, 0.8, 0.0037),
            ((1.0e5, -4.0e4, 0.0), 0.4, 0.8, 0.37),
        ):
            with self.subTest(center=center):
                ellipse = EllipseArcCurve(
                    "ellipse",
                    center,
                    (first_length, 0.0, 0.0),
                    (0.0, second_length, 0.0),
                )
                point = np.asarray(ellipse.point(parameter), dtype=float)
                tangent_vector = np.asarray(
                    ellipse.tangent(parameter), dtype=float
                )
                tangent = SegmentCurve(
                    "line",
                    tuple(
                        point
                        - 3.0 * tangent_vector
                        + (0.0, 0.0, depth_offset)
                    ),
                    tuple(
                        point
                        + 3.0 * tangent_vector
                        + (0.0, 0.0, depth_offset)
                    ),
                )
                crossings = compute_projected_curve_crossings(
                    (ellipse, tangent), IDENTITY_VIEW
                )
                self.assertEqual(len(crossings), 1)
                self.assertAlmostEqual(
                    crossings[0].first_parameter,
                    parameter,
                    places=10,
                )
                self.assertTrue(crossings[0].tangential)

    def test_rotated_oblique_ellipse_endpoint_tangent_snaps_to_endpoint(self) -> None:
        scale = 1.0
        center, first_axis, second_axis = self._rotated_axes(scale)
        tangent = SegmentCurve(
            "line",
            tuple(
                center
                + second_axis
                - 2.0 * first_axis
                + GENERAL_DEPTH_OFFSET
            ),
            tuple(
                center
                + second_axis
                + 2.0 * first_axis
                + GENERAL_DEPTH_OFFSET
            ),
        )
        for domain in (
            ParameterInterval(0.0, pi / 2.0),
            ParameterInterval(pi / 2.0, pi),
        ):
            with self.subTest(domain=domain):
                ellipse = EllipseArcCurve(
                    "ellipse",
                    tuple(center),
                    tuple(first_axis),
                    tuple(second_axis),
                    domain=domain,
                )
                crossings = compute_projected_curve_crossings(
                    (ellipse, tangent), GENERAL_VIEW
                )
                self.assertEqual(len(crossings), 1)
                self.assertAlmostEqual(crossings[0].first_parameter, pi / 2.0)
                self.assertTrue(crossings[0].tangential)

    def test_shifted_closed_domain_canonicalizes_tangent_seam(self) -> None:
        scale = 1.0
        center, first_axis, second_axis = self._rotated_axes(scale)
        for start in (10.0, -100.0):
            with self.subTest(start=start):
                domain = ParameterInterval(start, start + tau)
                ellipse = EllipseArcCurve(
                    "ellipse",
                    tuple(center),
                    tuple(first_axis),
                    tuple(second_axis),
                    domain=domain,
                )
                point = np.asarray(ellipse.point(start), dtype=float)
                tangent_vector = np.asarray(ellipse.tangent(start), dtype=float)
                tangent = SegmentCurve(
                    "line",
                    tuple(point - 2.0 * tangent_vector + GENERAL_DEPTH_OFFSET),
                    tuple(point + 2.0 * tangent_vector + GENERAL_DEPTH_OFFSET),
                )
                crossings = compute_projected_curve_crossings(
                    (ellipse, tangent), GENERAL_VIEW
                )
                self.assertEqual(len(crossings), 1)
                self.assertAlmostEqual(crossings[0].first_parameter, start)
                self.assertTrue(crossings[0].tangential)

                split = compute_projected_curve_crossings(
                    (ellipse, tangent),
                    GENERAL_VIEW,
                    active_intervals={
                        "ellipse": (
                            ParameterInterval(start, start + 0.4),
                            ParameterInterval(start + tau - 0.4, start + tau),
                        )
                    },
                )
                self.assertEqual(len(split), 1)
                self.assertTrue(split[0].tangential)

    def test_rotated_oblique_external_ellipses_have_one_seam_tangent(self) -> None:
        for scale in (1.0e-3, 1.0, 1.0e3):
            with self.subTest(scale=scale):
                center, first_axis, second_axis = self._rotated_axes(scale)
                first = EllipseArcCurve(
                    "first",
                    tuple(center),
                    tuple(first_axis),
                    tuple(second_axis),
                )
                second = EllipseArcCurve(
                    "second",
                    tuple(center + 2.0 * first_axis + scale * GENERAL_DEPTH_OFFSET),
                    tuple(first_axis),
                    tuple(second_axis),
                )
                crossings = compute_projected_curve_crossings(
                    (first, second), GENERAL_VIEW
                )
                self.assertEqual(len(crossings), 1)
                self.assertAlmostEqual(crossings[0].first_parameter, 0.0)
                self.assertAlmostEqual(crossings[0].second_parameter, pi)
                self.assertTrue(crossings[0].tangential)

    def test_translated_external_ellipses_deflate_certified_double_root(self) -> None:
        parameter = 1.25
        center = np.asarray((1.0e5, -4.0e4, 0.0), dtype=float)
        first_axis = np.asarray((0.4, 0.0, 0.0), dtype=float)
        second_axis = np.asarray((0.0, 0.8, 0.0), dtype=float)
        radial = cos(parameter) * first_axis + sin(parameter) * second_axis
        first = EllipseArcCurve(
            "first",
            tuple(center),
            tuple(first_axis),
            tuple(second_axis),
        )
        second = EllipseArcCurve(
            "second",
            tuple(center + 2.0 * radial + (0.0, 0.0, 0.37)),
            tuple(first_axis),
            tuple(second_axis),
        )
        crossings = compute_projected_curve_crossings(
            (first, second), IDENTITY_VIEW
        )
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(
            crossings[0].first_parameter,
            parameter,
            places=9,
        )
        self.assertAlmostEqual(
            crossings[0].second_parameter,
            parameter + pi,
            places=9,
        )
        self.assertTrue(crossings[0].tangential)

    def test_translated_near_secant_ellipses_keep_two_real_crossings(self) -> None:
        parameter = 1.25
        center = np.asarray((1.0e5, -4.0e4, 0.0), dtype=float)
        first_axis = np.asarray((0.4, 0.0, 0.0), dtype=float)
        second_axis = np.asarray((0.0, 0.8, 0.0), dtype=float)
        radial = cos(parameter) * first_axis + sin(parameter) * second_axis
        first = EllipseArcCurve(
            "first", tuple(center), tuple(first_axis), tuple(second_axis)
        )
        second = EllipseArcCurve(
            "second",
            tuple(center + 2.0 * (1.0 - 1.0e-10) * radial + (0.0, 0.0, 0.37)),
            tuple(first_axis),
            tuple(second_axis),
        )
        crossings = compute_projected_curve_crossings(
            (first, second), IDENTITY_VIEW
        )
        self.assertEqual(len(crossings), 2)
        self.assertTrue(all(not item.tangential for item in crossings))

    def test_affine_projected_external_ellipse_keeps_tangent_flag(self) -> None:
        matrix = np.asarray(
            (
                (-0.7282077164387947, -0.21021854645858215, -0.7336865016204153),
                (0.9214402424552589, -0.23966565746787047, -0.8458895281206953),
                (0.0012497601457203587, -0.814611027217695, 0.23216491064347194),
            ),
            dtype=float,
        )
        view = ParallelView.from_matrix(matrix)
        center = np.asarray(
            (-16.00612363707383, 0.8895715263178936, -2.1528274545525767),
            dtype=float,
        )
        first_axis = np.asarray(
            (-0.00011379316915934041, 0.0016444638159092864, 0.0014399794809276656),
            dtype=float,
        )
        second_axis = np.asarray(
            (0.0004961580708524393, -0.002033952080022724, 0.002361991988931637),
            dtype=float,
        )
        parameter = 4.102251458679734
        radial = cos(parameter) * first_axis + sin(parameter) * second_axis
        depth_only = np.asarray(
            (7.441132676141835e-7, -0.0004850233665821711, 0.00013823211667923215),
            dtype=float,
        )
        first = EllipseArcCurve(
            "a:b", tuple(center), tuple(first_axis), tuple(second_axis)
        )
        second = EllipseArcCurve(
            "c",
            tuple(center + 2.0 * radial + depth_only),
            tuple(first_axis),
            tuple(second_axis),
        )
        crossings = compute_projected_curve_crossings((first, second), view)
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(
            crossings[0].first_parameter,
            parameter,
            places=9,
        )
        self.assertTrue(crossings[0].tangential)

    def test_certified_affine_tangency_evidence_reaches_final_crossing(self) -> None:
        matrix = np.asarray(
            (
                (-1.2570119970344, 0.5744098693280729, 0.1802476999254147),
                (-0.23131398591634955, -0.27037053998088206, -0.7515266430260766),
                (-0.22141175182656103, -0.5702936163036842, 0.2733186238139441),
            ),
            dtype=float,
        )
        view = ParallelView.from_matrix(matrix)
        center = np.asarray(
            (-60.06783495494622, -40.92270631543523, 59.34654110471347),
            dtype=float,
        )
        first_axis = np.asarray(
            (-0.04818032091169329, -0.004438729395598966, -0.02679874276421147),
            dtype=float,
        )
        second_axis = np.asarray(
            (0.01929991572871493, -3.556398703759745e-05, -0.03469260788252468),
            dtype=float,
        )
        parameter = 2.8327183023518345
        radial = cos(parameter) * first_axis + sin(parameter) * second_axis
        depth_only = np.asarray(
            (-0.005399066991053504, -0.01390645895528471, 0.0066648023318562274),
            dtype=float,
        )
        first = EllipseArcCurve(
            "a:b", tuple(center), tuple(first_axis), tuple(second_axis)
        )
        second = EllipseArcCurve(
            "c",
            tuple(center + 2.0 * radial + depth_only),
            tuple(first_axis),
            tuple(second_axis),
        )
        crossings = compute_projected_curve_crossings((first, second), view)
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(
            crossings[0].first_parameter,
            parameter,
            places=9,
        )
        self.assertTrue(crossings[0].tangential)

    def test_authored_ulp_external_tangency_is_not_split_into_two_roots(self) -> None:
        matrix = np.asarray(
            (
                (-0.6289374141331763, 0.8651276360164915, 0.061490072115455936),
                (-1.2691052412800528, -0.8910883804728512, -0.44369088691372555),
                (-0.24310647374470762, -0.26381872255296185, 1.2252077161601906),
            ),
            dtype=float,
        )
        view = ParallelView.from_matrix(matrix)
        first = EllipseArcCurve(
            "first",
            (-34309.81844632449, -8208.615769201835, -51017.39728878597),
            (-0.3725276652202609, -1.420756801016753, -0.2334020339435005),
            (1.927087688705882, -0.4995299303497364, -0.035059383820449125),
        )
        second = EllipseArcCurve(
            "second",
            (-34307.292118201585, -8206.885146333983, -51016.32116595511),
            first.first_axis,
            first.second_axis,
        )
        parameter = 2.5850648578782454
        crossings = compute_projected_curve_crossings((first, second), view)
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0].first_parameter, parameter, places=9)
        self.assertAlmostEqual(
            crossings[0].second_parameter,
            parameter + pi,
            places=9,
        )
        self.assertTrue(crossings[0].tangential)

    def test_conditioned_projection_ulp_tangency_is_not_split(self) -> None:
        matrix = np.asarray(
            (
                (-0.8885396163527252, -0.08966605913330208, 0.7705896970231526),
                (0.12910432747609357, -0.5064234277646456, 0.08993802649111751),
                (0.4336776699584386, 0.20357344691642884, 0.5237461917462767),
            ),
            dtype=float,
        )
        center = np.asarray(
            (59.42737386667701, -1302.2166849445161, -1417.2714769312724),
            dtype=float,
        )
        first_axis = np.asarray(
            (-0.21262633672747613, 0.08123182156936122, 0.25683832127063994),
            dtype=float,
        )
        second_axis = np.asarray(
            (-0.3650320425331105, -0.12340431401849736, -0.2631658252566616),
            dtype=float,
        )
        parameter = 4.002018907870314
        radial = cos(parameter) * first_axis + sin(parameter) * second_axis
        depth_only = np.asarray(
            (0.22122127445275167, 0.10384389257558015, 0.2671657040561724),
            dtype=float,
        )
        first = EllipseArcCurve(
            "a:b", tuple(center), tuple(first_axis), tuple(second_axis)
        )
        second = EllipseArcCurve(
            "c",
            tuple(center + 2.0 * radial + depth_only),
            tuple(first_axis),
            tuple(second_axis),
        )
        crossings = compute_projected_curve_crossings(
            (first, second), ParallelView.from_matrix(matrix)
        )
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0].first_parameter, parameter, places=9)
        self.assertAlmostEqual(
            crossings[0].second_parameter,
            parameter - pi,
            places=9,
        )
        self.assertTrue(crossings[0].tangential)

    def test_authored_ulp_line_midpoint_tangency_is_not_split(self) -> None:
        matrix = np.asarray(
            (
                (-0.47056199738474697, -0.3269719624184288, -0.16333952119931536),
                (0.7612222915395153, -0.8771005744480356, -0.4372180435614149),
                (-0.000229294375483119, -0.24641213619890254, 0.4939267383889388),
            ),
            dtype=float,
        )
        ellipse = EllipseArcCurve(
            "c",
            (40884.3886062002, 36249.21138628788, -35170.52783470624),
            (-0.042011322105631625, 0.08270942814892197, -0.00010458171732651657),
            (-0.0022861301810360463, -0.0011281497282801913, 0.026149240884587987),
        )
        parameter = 5.686411744945884
        point = np.asarray(ellipse.point(parameter), dtype=float)
        tangent = np.asarray(ellipse.tangent(parameter), dtype=float)
        depth_only = np.asarray(
            (-1.7195725604234003e-05, -0.018479456684025442, 0.03704159181418198),
            dtype=float,
        )
        line = SegmentCurve(
            "a:b",
            tuple(point - 3.0 * tangent + depth_only),
            tuple(point + 3.0 * tangent + depth_only),
        )
        crossings = compute_projected_curve_crossings(
            (line, ellipse), ParallelView.from_matrix(matrix)
        )
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0].first_parameter, 0.5)
        self.assertAlmostEqual(crossings[0].second_parameter, parameter, places=8)
        self.assertTrue(crossings[0].tangential)

    def test_rotated_oblique_external_circles_have_one_seam_tangent(self) -> None:
        angle = 0.7
        radial = np.asarray((cos(angle), sin(angle), 0.0), dtype=float)
        normal = (0.0, 0.0, 1.0)
        for scale in (1.0e-3, 1.0, 1.0e3):
            with self.subTest(scale=scale):
                radius = 0.7 * scale
                center = np.asarray((12.3, -45.6, 0.7), dtype=float) * scale
                first = CircleArcCurve(
                    "first",
                    tuple(center),
                    radius,
                    normal,
                    radial_axis=tuple(radial),
                )
                second = CircleArcCurve(
                    "second",
                    tuple(center + 2.0 * radius * radial + scale * GENERAL_DEPTH_OFFSET),
                    radius,
                    normal,
                    radial_axis=tuple(radial),
                )
                crossings = compute_projected_curve_crossings(
                    (first, second), GENERAL_VIEW
                )
                self.assertEqual(len(crossings), 1)
                self.assertAlmostEqual(crossings[0].first_parameter, 0.0)
                self.assertTrue(crossings[0].tangential)

    def test_genuine_nearby_double_crossing_is_not_collapsed_to_tangent(self) -> None:
        scale = 1.0
        angle = 0.7
        radial = np.asarray((cos(angle), sin(angle), 0.0), dtype=float)
        center = np.asarray((12.3, -45.6, 0.7), dtype=float)
        radius = 0.7
        first = CircleArcCurve(
            "first", tuple(center), radius, (0.0, 0.0, 1.0), radial_axis=tuple(radial)
        )
        # This is close to external tangency but remains objectively secant:
        # the two authored crossings are about 2e-5 radians apart.
        second = CircleArcCurve(
            "second",
            tuple(
                center
                + 2.0 * radius * (1.0 - 1.0e-10) * radial
                + scale * GENERAL_DEPTH_OFFSET
            ),
            radius,
            (0.0, 0.0, 1.0),
            radial_axis=tuple(radial),
        )
        crossings = compute_projected_curve_crossings((first, second), GENERAL_VIEW)
        self.assertEqual(len(crossings), 2)
        self.assertTrue(all(not item.tangential for item in crossings))

    def test_near_side_frustum_rim_joint_is_one_coincident_depth_crossing(
        self,
    ) -> None:
        angle = -0.01
        view = ParallelView.from_matrix(
            (
                (1.0, 0.0, 0.0),
                (0.0, -sin(angle), cos(angle)),
                (0.0, -cos(angle), -sin(angle)),
            )
        )
        frustum = ConeSpec(
            "joint-frustum",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.75, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.CLOSED_SINGLE,
        )
        sources = build_surface_boundary_sources((frustum,), view)
        cap_rim = next(
            item for item in sources if item.source_id.endswith("cap_max:rim")
        )
        silhouette = next(
            item
            for item in sources
            if item.source_id.endswith("silhouette:generator:0")
        )
        crossings = compute_projected_curve_crossings(
            (cap_rim.curve, silhouette.curve),
            view,
        )

        self.assertEqual(len(crossings), 1)
        self.assertTrue(crossings[0].coincident_depth)
        self.assertIsNone(crossings[0].far_curve_id)
        self.assertIsNone(crossings[0].near_curve_id)

    def test_parabola_tangent_is_scale_stable_in_parameter_chart(self) -> None:
        base_embedding = np.asarray(
            (
                (0.8, -0.3, 12.3),
                (0.6, 1.1, -45.6),
                (0.2, 0.4, 0.7),
                (0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        for scale in (1.0e-6, 1.0e-3, 1.0, 1.0e3):
            with self.subTest(scale=scale):
                embedding = base_embedding.copy()
                embedding[:3] *= scale
                parabola = ParametricConicBranch(
                    "parabola",
                    ConicParameterization(
                        ConicKind.PARABOLA,
                        "parabola",
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (0.0, 1.0),
                    ),
                    tuple(tuple(float(item) for item in row) for row in embedding),
                    ParameterInterval(-2.0, 2.0),
                )
                parameter = 0.7
                point = np.asarray(parabola.point(parameter), dtype=float)
                tangent_vector = np.asarray(parabola.tangent(parameter), dtype=float)
                tangent = SegmentCurve(
                    "line",
                    tuple(point - 2.0 * tangent_vector + scale * GENERAL_DEPTH_OFFSET),
                    tuple(point + 2.0 * tangent_vector + scale * GENERAL_DEPTH_OFFSET),
                    domain=ParameterInterval(4.0, 8.0),
                )
                crossing = compute_projected_curve_crossings(
                    (parabola, tangent), GENERAL_VIEW
                )
                self.assertEqual(len(crossing), 1)
                self.assertAlmostEqual(crossing[0].second_parameter, parameter)
                self.assertTrue(crossing[0].tangential)

    def test_hyperbola_tangent_is_scale_stable_in_exp_chart(self) -> None:
        base_embedding = np.asarray(
            (
                (0.8, -0.3, 12.3),
                (0.6, 1.1, -45.6),
                (0.2, 0.4, 0.7),
                (0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        for scale in (1.0e-3, 1.0):
            for parameter in (0.7, 5.0):
                with self.subTest(scale=scale, parameter=parameter):
                    embedding = base_embedding.copy()
                    embedding[:3] *= scale
                    hyperbola = ParametricConicBranch(
                        "hyperbola",
                        ConicParameterization(
                            ConicKind.HYPERBOLA,
                            "hyperbola:positive",
                            (0.0, 0.0),
                            (1.0, 0.0),
                            (0.0, 1.0),
                            branch_sign=1,
                        ),
                        tuple(tuple(float(item) for item in row) for row in embedding),
                        ParameterInterval(-20.0, 20.0),
                    )
                    point = np.asarray(hyperbola.point(parameter), dtype=float)
                    tangent_vector = np.asarray(
                        hyperbola.tangent(parameter), dtype=float
                    )
                    tangent = SegmentCurve(
                        "line",
                        tuple(
                            point
                            - 2.0 * tangent_vector
                            + scale * GENERAL_DEPTH_OFFSET
                        ),
                        tuple(
                            point
                            + 2.0 * tangent_vector
                            + scale * GENERAL_DEPTH_OFFSET
                        ),
                    )
                    crossings = compute_projected_curve_crossings(
                        (hyperbola, tangent), GENERAL_VIEW
                    )
                    self.assertEqual(len(crossings), 1)
                    self.assertAlmostEqual(
                        crossings[0].first_parameter, parameter, places=8
                    )
                    self.assertTrue(crossings[0].tangential)

    def test_distant_hyperbola_midpoint_tangent_keeps_authored_parameter(self) -> None:
        embedding = (
            (0.8, -0.3, 12.3),
            (0.6, 1.1, -45.6),
            (0.2, 0.4, 0.7),
            (0.0, 0.0, 1.0),
        )
        hyperbola = ParametricConicBranch(
            "hyperbola",
            ConicParameterization(
                ConicKind.HYPERBOLA,
                "hyperbola:positive",
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                branch_sign=1,
            ),
            embedding,
            ParameterInterval(-30.0, 30.0),
        )
        for parameter in (8.0, 15.0, 28.0, -8.0, -15.0, -28.0):
            with self.subTest(parameter=parameter):
                point = np.asarray(hyperbola.point(parameter), dtype=float)
                tangent_vector = np.asarray(hyperbola.tangent(parameter), dtype=float)
                tangent = SegmentCurve(
                    "line",
                    tuple(point - 2.0 * tangent_vector + GENERAL_DEPTH_OFFSET),
                    tuple(point + 2.0 * tangent_vector + GENERAL_DEPTH_OFFSET),
                )
                crossings = compute_projected_curve_crossings(
                    (hyperbola, tangent), GENERAL_VIEW
                )
                self.assertEqual(len(crossings), 1)
                self.assertAlmostEqual(
                    crossings[0].first_parameter,
                    parameter,
                    places=10,
                )
                self.assertTrue(crossings[0].tangential)

    def test_same_projected_anisotropic_ellipse_support_is_id_symmetric(self) -> None:
        matrix = np.asarray(
            ((1.0, 0.2, 0.3), (-0.1, 1.0, 0.15), (0.0, 0.0, 1.0)),
            dtype=float,
        )
        view = ParallelView.from_matrix(matrix)
        displacement2 = np.linalg.solve(matrix[:2, :2], -matrix[:2, 2])

        def pair(
            first_id: str,
            second_id: str,
            short_axis: float,
            long_axis: float,
        ) -> tuple[EllipseArcCurve, EllipseArcCurve]:
            first_axis = (short_axis, 0.0, 0.0)
            second_axis = (0.0, long_axis, 0.0)
            first = EllipseArcCurve(
                first_id,
                (100.0, 200.0, 0.0),
                first_axis,
                second_axis,
                domain=ParameterInterval(0.0, 1.0),
            )
            second = EllipseArcCurve(
                second_id,
                (100.0 + displacement2[0], 200.0 + displacement2[1], 1.0),
                second_axis,
                first_axis,
                domain=ParameterInterval(0.0, 1.0),
            )
            return first, second

        for axes in ((1.0e-6, 1.0e6), (1.0e-9, 1.0e9)):
            for ids in (("a", "z"), ("z", "a")):
                with self.subTest(axes=axes, ids=ids):
                    with self.assertRaisesRegex(
                        ProjectedCurveIntersectionError,
                        "infinitely many|cannot be certified symmetrically",
                    ):
                        compute_projected_curve_crossings(
                            pair(*ids, *axes),
                            view,
                        )

    def test_same_projected_tiny_segments_use_pair_screen_tolerance(self) -> None:
        matrix = np.asarray(
            ((1.0, 0.2, 0.3), (-0.1, 1.0, 0.15), (0.0, 0.0, 1.0)),
            dtype=float,
        )
        view = ParallelView.from_matrix(matrix)
        displacement2 = np.linalg.solve(matrix[:2, :2], -matrix[:2, 2])
        scale = 1.0e-6
        first = SegmentCurve("first", (-scale, 0.0, 0.0), (scale, 0.0, 0.0))
        second = SegmentCurve(
            "second",
            (-scale + displacement2[0], displacement2[1], 1.0),
            (scale + displacement2[0], displacement2[1], 1.0),
        )
        with self.assertRaisesRegex(
            ProjectedCurveIntersectionError,
            "infinitely many|cannot be certified symmetrically",
        ):
            compute_projected_curve_crossings((first, second), view)

    def test_explicit_screen_tolerance_controls_support_classification(self) -> None:
        first = SegmentCurve("first", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        second = SegmentCurve(
            "second", (-1.0, 1.0e-4, 1.0), (1.0, 1.0e-4, 1.0)
        )
        with self.assertRaises(ProjectedCurveIntersectionError):
            compute_projected_curve_crossings(
                (first, second),
                IDENTITY_VIEW,
                context=GeometryContext(screen_tolerance=1.0e-3),
            )

    def test_large_unbounded_axes_do_not_merge_opposite_hyperbola_branches(self) -> None:
        for scale in (1.0e9, 1.0e12):
            with self.subTest(scale=scale):
                first_embedding = (
                    (scale, 0.0, 0.0),
                    (0.0, scale, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
                second_embedding = (
                    (scale, 0.0, 0.0),
                    (0.0, scale, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 1.0),
                )
                positive = ParametricConicBranch(
                    "positive",
                    ConicParameterization(
                        ConicKind.HYPERBOLA,
                        "positive",
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (0.0, 1.0),
                        branch_sign=1,
                    ),
                    first_embedding,
                    ParameterInterval(-2.0, 2.0),
                )
                negative = ParametricConicBranch(
                    "negative",
                    ConicParameterization(
                        ConicKind.HYPERBOLA,
                        "negative",
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (0.0, 1.0),
                        branch_sign=-1,
                    ),
                    second_embedding,
                    ParameterInterval(-2.0, 2.0),
                )
                self.assertEqual(
                    compute_projected_curve_crossings(
                        (positive, negative), IDENTITY_VIEW
                    ),
                    (),
                )

    def test_crossing_identity_is_unambiguous_for_colon_curve_ids(self) -> None:
        curves = (
            SegmentCurve("a", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            SegmentCurve("b:c", (0.0, -1.0, 1.0), (0.0, 1.0, 1.0)),
            SegmentCurve("a:b", (9.0, 10.0, 0.0), (11.0, 10.0, 0.0)),
            SegmentCurve("c", (10.0, 9.0, 1.0), (10.0, 11.0, 1.0)),
        )
        crossings = compute_projected_curve_crossings(curves, IDENTITY_VIEW)
        self.assertEqual(len(crossings), 2)
        self.assertEqual(len({item.crossing_id for item in crossings}), 2)
        self.assertEqual(
            json.loads(canonical_projected_curve_crossings_json(crossings)),
            [item.to_dict() for item in crossings],
        )

    def test_closed_parametric_oval_counts_the_seam_only_once(self) -> None:
        circle = ParametricConicBranch(
            "parametric-circle",
            ConicParameterization(
                ConicKind.CIRCLE,
                "circle",
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                natural_domain=ParameterInterval(0.0, tau),
                closed=True,
            ),
            PLANE_EMBEDDING,
            ParameterInterval(0.0, tau),
        )
        secant = SegmentCurve("secant", (-2.0, 0.0, 1.0), (2.0, 0.0, 1.0))
        tangent = SegmentCurve("seam-tangent", (1.0, -2.0, 1.0), (1.0, 2.0, 1.0))

        self.assertEqual(
            len(compute_projected_curve_crossings((circle, secant), IDENTITY_VIEW)),
            2,
        )
        seam = compute_projected_curve_crossings((circle, tangent), IDENTITY_VIEW)
        self.assertEqual(len(seam), 1)
        self.assertAlmostEqual(seam[0].first_parameter, 0.0)

    def test_reparameterized_same_projected_circle_fails_closed(self) -> None:
        angle = 0.3
        first = CircleArcCurve(
            "first-circle",
            (0.0, 0.0, -1.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        second = CircleArcCurve(
            "second-circle",
            (0.0, 0.0, 1.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(cos(angle), sin(angle), 0.0),
        )
        with self.assertRaisesRegex(
            ProjectedCurveIntersectionError, "infinitely many"
        ):
            compute_projected_curve_crossings((first, second), IDENTITY_VIEW)

    def test_large_translation_does_not_merge_concentric_different_radii(self) -> None:
        center = (1.0e5, 1.0e5, 0.0)
        first = CircleArcCurve(
            "radius-one", center, 1.0, (0.0, 0.0, 1.0)
        )
        second = CircleArcCurve(
            "radius-two", center, 2.0, (0.0, 0.0, 1.0)
        )
        self.assertEqual(
            compute_projected_curve_crossings((first, second), IDENTITY_VIEW),
            (),
        )

    def test_line_to_circle_substitution_is_translation_conditioned(self) -> None:
        center = (1.0e8, 1.0e8, 0.0)
        circle = CircleArcCurve(
            "z-circle", center, 1.0, (0.0, 0.0, 1.0)
        )
        line = SegmentCurve(
            "a-line",
            (center[0] - 2.0, center[1], 1.0),
            (center[0] + 2.0, center[1], 1.0),
        )
        self.assertEqual(
            len(compute_projected_curve_crossings((line, circle), IDENTITY_VIEW)),
            2,
        )

    def test_circle_circle_substitution_is_translation_and_scale_conditioned(self) -> None:
        for scale, translation in ((1.0e-6, 100.0), (1.0, 1.0e8)):
            with self.subTest(scale=scale, translation=translation):
                first = CircleArcCurve(
                    "first",
                    (translation, translation, 0.0),
                    scale,
                    (0.0, 0.0, 1.0),
                )
                second = CircleArcCurve(
                    "second",
                    (translation + scale, translation, 1.0),
                    scale,
                    (0.0, 0.0, 1.0),
                )
                self.assertEqual(
                    len(
                        compute_projected_curve_crossings(
                            (first, second), IDENTITY_VIEW
                        )
                    ),
                    2,
                )

    def test_anisotropic_full_ellipse_context_includes_both_axes(self) -> None:
        ellipse = EllipseArcCurve(
            "ellipse",
            (0.0, 0.0, 0.0),
            (1.0e-9, 0.0, 0.0),
            (0.0, 1.0e9, 0.0),
        )
        line = SegmentCurve(
            "line",
            (-2.0e-9, 0.0, 1.0),
            (2.0e-9, 0.0, 1.0),
        )
        self.assertEqual(
            len(compute_projected_curve_crossings((ellipse, line), IDENTITY_VIEW)),
            2,
        )

    def test_anisotropic_ellipse_is_invariant_to_curve_id_source_order(self) -> None:
        ellipse = EllipseArcCurve(
            "z-ellipse",
            (0.0, 0.0, 0.0),
            (1.0e-6, 0.0, 0.0),
            (0.0, 1.0e6, 0.0),
        )
        line = SegmentCurve(
            "a-line",
            (-2.0e-6, 0.0, 1.0),
            (2.0e-6, 0.0, 1.0),
        )
        crossings = compute_projected_curve_crossings(
            (ellipse, line), IDENTITY_VIEW
        )
        self.assertEqual(len(crossings), 2)
        self.assertEqual({item.near_curve_id for item in crossings}, {"a-line"})

    def test_same_circle_disjoint_arcs_do_not_fail_as_infinite_overlap(self) -> None:
        first = CircleArcCurve(
            "first-arc",
            (0.0, 0.0, 0.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
            domain=ParameterInterval(0.0, 0.5),
        )
        second = CircleArcCurve(
            "second-arc",
            (0.0, 0.0, 1.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
            domain=ParameterInterval(1.0, 1.5),
        )
        self.assertEqual(
            compute_projected_curve_crossings((first, second), IDENTITY_VIEW),
            (),
        )

    def test_same_circle_support_sign_is_invariant_to_curve_id_order(self) -> None:
        def pair(first_id: str, second_id: str):
            first = CircleArcCurve(
                first_id,
                (0.0, 0.0, 0.0),
                3.0,
                (0.0, 0.0, 1.0),
                radial_axis=(1.0, 0.0, 0.0),
                domain=ParameterInterval(0.0, 2.0),
            )
            second = CircleArcCurve(
                second_id,
                (0.0, 0.0, 1.0),
                3.0,
                (0.0, 0.0, -1.0),
                radial_axis=(cos(2.0), sin(2.0), 0.0),
                domain=ParameterInterval(0.0, 2.0),
            )
            return first, second

        for ids in (("a", "z"), ("z", "a")):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(
                    ProjectedCurveIntersectionError, "infinitely many"
                ):
                    compute_projected_curve_crossings(pair(*ids), IDENTITY_VIEW)

    def test_unresolvable_tiny_arc_after_huge_translation_fails_closed(self) -> None:
        center = (1.0e8, -2.0e8, 0.0)
        radius = 1.0e-6
        arc = CircleArcCurve(
            "arc",
            center,
            radius,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
            domain=ParameterInterval(pi - 0.4, pi + 0.4),
        )
        point = np.asarray(arc.point(pi), dtype=float)
        tangent = np.asarray(arc.tangent(pi), dtype=float)
        tangent /= np.linalg.norm(tangent)
        line = SegmentCurve(
            "line",
            tuple(point - 2.0 * radius * tangent + np.asarray((0.0, 0.0, 1.0))),
            tuple(point + 2.0 * radius * tangent + np.asarray((0.0, 0.0, 1.0))),
        )
        with self.assertRaisesRegex(
            ProjectedCurveIntersectionError, "numerically indistinguishable"
        ):
            compute_projected_curve_crossings((arc, line), OBLIQUE_VIEW)

    def test_parabola_and_hyperbola_use_their_analytic_parameters(self) -> None:
        parabola = ParametricConicBranch(
            "parabola",
            ConicParameterization(
                ConicKind.PARABOLA,
                "parabola",
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
            PLANE_EMBEDDING,
            ParameterInterval(-2.0, 2.0),
        )
        parabola_line = SegmentCurve(
            "parabola-line", (-2.0, 1.0, 2.0), (2.0, 1.0, 2.0)
        )
        parabola_crossings = compute_projected_curve_crossings(
            (parabola, parabola_line), IDENTITY_VIEW
        )
        self.assertEqual(len(parabola_crossings), 2)
        self.assertEqual(
            tuple(round(item.first_parameter, 12) for item in parabola_crossings),
            (-1.0, 1.0),
        )

        hyperbola = ParametricConicBranch(
            "hyperbola",
            ConicParameterization(
                ConicKind.HYPERBOLA,
                "hyperbola:positive",
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                branch_sign=1,
            ),
            PLANE_EMBEDDING,
            ParameterInterval(-2.0, 2.0),
        )
        hyperbola_line = SegmentCurve(
            "hyperbola-line", (2.0, -3.0, 2.0), (2.0, 3.0, 2.0)
        )
        hyperbola_crossings = compute_projected_curve_crossings(
            (hyperbola, hyperbola_line), IDENTITY_VIEW
        )
        expected = acosh(2.0)
        self.assertEqual(len(hyperbola_crossings), 2)
        self.assertAlmostEqual(hyperbola_crossings[0].first_parameter, -expected)
        self.assertAlmostEqual(hyperbola_crossings[1].first_parameter, expected)

    def test_same_hyperbola_support_separates_branches_and_finite_domains(self) -> None:
        def branch(curve_id: str, sign: int, domain: ParameterInterval) -> ParametricConicBranch:
            return ParametricConicBranch(
                curve_id,
                ConicParameterization(
                    ConicKind.HYPERBOLA,
                    f"hyperbola:{sign}",
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                    branch_sign=sign,
                ),
                PLANE_EMBEDDING,
                domain,
            )

        positive = branch("positive", 1, ParameterInterval(-2.0, 2.0))
        negative = branch("negative", -1, ParameterInterval(-2.0, 2.0))
        disjoint = branch("disjoint", 1, ParameterInterval(3.0, 4.0))
        touching = branch("touching", 1, ParameterInterval(2.0, 3.0))
        overlapping = branch("overlapping", 1, ParameterInterval(1.0, 3.0))

        self.assertEqual(
            compute_projected_curve_crossings((positive, negative), IDENTITY_VIEW),
            (),
        )
        self.assertEqual(
            compute_projected_curve_crossings((positive, disjoint), IDENTITY_VIEW),
            (),
        )
        self.assertEqual(
            len(compute_projected_curve_crossings((positive, touching), IDENTITY_VIEW)),
            1,
        )
        with self.assertRaisesRegex(
            ProjectedCurveIntersectionError, "infinitely many"
        ):
            compute_projected_curve_crossings((positive, overlapping), IDENTITY_VIEW)

    def test_distant_hyperbola_crossings_keep_small_exp_coefficients(self) -> None:
        hyperbola = ParametricConicBranch(
            "a-wide-hyperbola",
            ConicParameterization(
                ConicKind.HYPERBOLA,
                "hyperbola:positive",
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                branch_sign=1,
            ),
            PLANE_EMBEDDING,
            ParameterInterval(-30.0, 30.0),
        )
        target_x = float(np.cosh(28.0))
        line = SegmentCurve(
            "z-distant-line",
            (target_x, -2.0 * target_x, 2.0),
            (target_x, 2.0 * target_x, 2.0),
        )
        crossings = compute_projected_curve_crossings(
            (hyperbola, line), IDENTITY_VIEW
        )
        self.assertEqual(len(crossings), 2)
        self.assertAlmostEqual(crossings[0].first_parameter, -28.0, places=8)
        self.assertAlmostEqual(crossings[1].first_parameter, 28.0, places=8)

    def test_same_parabola_support_respects_finite_parameter_domains(self) -> None:
        def branch(curve_id: str, domain: ParameterInterval) -> ParametricConicBranch:
            return ParametricConicBranch(
                curve_id,
                ConicParameterization(
                    ConicKind.PARABOLA,
                    "parabola",
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                ),
                PLANE_EMBEDDING,
                domain,
            )

        first = branch("first", ParameterInterval(-2.0, 0.0))
        disjoint = branch("disjoint", ParameterInterval(1.0, 2.0))
        touching = branch("touching", ParameterInterval(0.0, 1.0))
        overlapping = branch("overlapping", ParameterInterval(-1.0, 1.0))
        self.assertEqual(
            compute_projected_curve_crossings((first, disjoint), IDENTITY_VIEW),
            (),
        )
        self.assertEqual(
            len(compute_projected_curve_crossings((first, touching), IDENTITY_VIEW)),
            1,
        )
        with self.assertRaisesRegex(
            ProjectedCurveIntersectionError, "infinitely many"
        ):
            compute_projected_curve_crossings((first, overlapping), IDENTITY_VIEW)


class ProjectedCrossingDeterminismTests(unittest.TestCase):
    def test_input_order_json_and_similarity_scale_are_stable(self) -> None:
        first = SegmentCurve("a", (-2.0, 0.0, -1.0), (2.0, 0.0, -1.0))
        second = SegmentCurve("b", (0.0, -2.0, 1.0), (0.0, 2.0, 1.0))
        normal = compute_projected_curve_crossings((first, second), OBLIQUE_VIEW)
        reversed_input = compute_projected_curve_crossings(
            (second, first), OBLIQUE_VIEW
        )
        self.assertEqual(normal, reversed_input)
        payload = canonical_projected_curve_crossings_json(normal)
        self.assertEqual(json.loads(payload), [item.to_dict() for item in normal])

        factor = 1.0e6
        scaled_first = SegmentCurve(
            "a",
            tuple(factor * np.asarray(first.start)),
            tuple(factor * np.asarray(first.end)),
        )
        scaled_second = SegmentCurve(
            "b",
            tuple(factor * np.asarray(second.start)),
            tuple(factor * np.asarray(second.end)),
        )
        scaled = compute_projected_curve_crossings(
            (scaled_first, scaled_second), OBLIQUE_VIEW
        )
        self.assertAlmostEqual(scaled[0].first_parameter, normal[0].first_parameter)
        self.assertAlmostEqual(scaled[0].second_parameter, normal[0].second_parameter)
        np.testing.assert_allclose(
            np.asarray(scaled[0].screen_point) / factor,
            normal[0].screen_point,
            rtol=0.0,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
