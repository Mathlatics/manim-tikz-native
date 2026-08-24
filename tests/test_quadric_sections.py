from __future__ import annotations

from math import pi, tau
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from polyhedron_visibility.quadrics.conics import ConicKind, classify_conic
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section,
    restrict_quadric_to_plane,
)
from polyhedron_visibility.quadrics.trace import (
    FiniteSectionTopology,
    canonical_quadric_section_trace_json,
    section_trace_curves,
)


ROOT = Path(__file__).resolve().parents[1]


def _assert_trace_geometry(
    case: unittest.TestCase,
    trace,
    surface,
    plane: SectionPlane,
) -> None:
    """Check sampled trace points in a scale-independent way."""

    characteristic = np.asarray(
        (*surface.characteristic_points, plane.point),
        dtype=float,
    )
    geometric_scale = max(
        float(np.linalg.norm(np.ptp(characteristic, axis=0))),
        np.finfo(float).tiny,
    )
    quadric = np.asarray(surface.support_quadric.matrix, dtype=float)
    samples: list[np.ndarray] = []
    for component in trace.components:
        branch = trace.branch_map[component.branch_id]
        for interval in component.parameter_intervals:
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                parameter = interval.start + fraction * interval.length
                samples.append(branch.world_point(parameter))
    samples.extend(
        np.asarray(point, dtype=float)
        for point in trace.isolated_world_points
    )

    for point in samples:
        case.assertTrue(np.all(np.isfinite(point)))
        homogeneous = np.append(point, 1.0)
        residual = abs(float(homogeneous @ quadric @ homogeneous))
        residual_scale = float(
            np.sum(
                np.abs(quadric)
                * np.outer(np.abs(homogeneous), np.abs(homogeneous))
            )
        )
        case.assertLessEqual(
            residual,
            2.0e-10 * max(residual_scale, np.finfo(float).tiny),
        )
        case.assertLessEqual(
            abs(plane.signed_distance(point)),
            2.0e-10 * geometric_scale,
        )
        axial_range = getattr(surface, "axial_range", None)
        if axial_range is not None:
            axial = float(surface.frame.to_local_point(point)[2])
            case.assertGreaterEqual(
                axial,
                axial_range[0] - 2.0e-10 * geometric_scale,
            )
            case.assertLessEqual(
                axial,
                axial_range[1] + 2.0e-10 * geometric_scale,
            )


def _interval_signature(trace) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(
        tuple((interval.start, interval.end) for interval in component.parameter_intervals)
        for component in trace.components
    )


class ConicClassificationTests(unittest.TestCase):
    def test_all_real_affine_conic_kinds_are_classified(self) -> None:
        cases = {
            ConicKind.CIRCLE: np.diag((1.0, 1.0, -1.0)),
            ConicKind.ELLIPSE: np.diag((0.25, 1.0, -1.0)),
            ConicKind.PARABOLA: np.asarray(
                ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0))
            ),
            ConicKind.HYPERBOLA: np.diag((1.0, -1.0, -1.0)),
            ConicKind.POINT: np.diag((1.0, 1.0, 0.0)),
            ConicKind.INTERSECTING_LINES: np.diag((1.0, -1.0, 0.0)),
            ConicKind.PARALLEL_LINES: np.diag((1.0, 0.0, -1.0)),
            ConicKind.COINCIDENT_LINE: np.diag((1.0, 0.0, 0.0)),
            ConicKind.EMPTY: np.diag((1.0, 1.0, 1.0)),
        }

        for expected, matrix in cases.items():
            with self.subTest(kind=expected.value):
                classification = classify_conic(matrix)
                self.assertIs(classification.kind, expected)

    def test_every_curve_branch_is_an_exact_analytic_parameterization(self) -> None:
        matrices = (
            np.diag((1.0, 1.0, -4.0)),
            np.diag((0.25, 1.0, -1.0)),
            np.asarray(
                ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0))
            ),
            np.diag((1.0, -1.0, -1.0)),
            np.diag((1.0, -1.0, 0.0)),
            np.diag((1.0, 0.0, -1.0)),
            np.diag((1.0, 0.0, 0.0)),
        )
        for matrix in matrices:
            classification = classify_conic(matrix)
            self.assertTrue(classification.branches)
            for branch in classification.branches:
                for parameter in (-1.25, -0.25, 0.0, 0.75, 1.5):
                    with self.subTest(
                        kind=classification.kind.value,
                        branch=branch.branch_label,
                        parameter=parameter,
                    ):
                        homogeneous = np.append(branch.point(parameter), 1.0)
                        residual = float(homogeneous @ matrix @ homogeneous)
                        self.assertAlmostEqual(residual, 0.0, places=10)
                        self.assertGreater(
                            float(np.linalg.norm(branch.tangent(parameter))), 0.0
                        )

    def test_semantic_branch_labels_are_deterministic(self) -> None:
        matrix = np.diag((1.0, -1.0, -1.0))
        first = classify_conic(matrix)
        second = classify_conic(matrix)
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(branch.branch_label for branch in first.branches),
            ("hyperbola:negative", "hyperbola:positive"),
        )


class SupportingSectionTests(unittest.TestCase):
    def test_plane_restriction_uses_h_transpose_q_h(self) -> None:
        sphere = SphereSpec("sphere", (1.0, -2.0, 0.5), 3.0)
        plane = SectionPlane(
            "oblique",
            (0.25, -0.5, 1.25),
            (1.0, 2.0, 3.0),
            u_axis=(2.0, -1.0, 0.0),
        )
        conic, embedding = restrict_quadric_to_plane(sphere, plane)
        expected = embedding.T @ np.asarray(sphere.support_quadric.matrix) @ embedding
        np.testing.assert_allclose(conic, expected, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            conic,
            plane.restrict(sphere.support_quadric),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_sphere_sections_are_circle_point_and_empty(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 2.0)
        expected = (
            (0.0, ConicKind.CIRCLE, FiniteSectionTopology.CLOSED_CURVE),
            (1.0, ConicKind.CIRCLE, FiniteSectionTopology.CLOSED_CURVE),
            (2.0, ConicKind.POINT, FiniteSectionTopology.POINT),
            (3.0, ConicKind.EMPTY, FiniteSectionTopology.EMPTY),
        )
        for height, kind, topology in expected:
            with self.subTest(height=height):
                trace = compute_quadric_section(
                    f"sphere-{height}",
                    sphere,
                    SectionPlane(f"z-{height}", (0.0, 0.0, height), (0, 0, 1)),
                )
                self.assertIs(trace.supporting_kind, kind)
                self.assertIs(trace.finite_topology, topology)


class FiniteCylinderSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            2.0,
            (-3.0, 3.0),
            radial_axis=(1.0, 0.0, 0.0),
        )

    def test_horizontal_tilted_and_axial_sections(self) -> None:
        cases = (
            (
                "horizontal",
                SectionPlane("horizontal", (0, 0, 0), (0, 0, 1)),
                ConicKind.CIRCLE,
                FiniteSectionTopology.CLOSED_CURVE,
            ),
            (
                "tilted",
                SectionPlane("tilted", (0, 0, 0), (1, 0, 1)),
                ConicKind.ELLIPSE,
                FiniteSectionTopology.CLOSED_CURVE,
            ),
            (
                "axial",
                SectionPlane("axial", (0, 0, 0), (1, 0, 0)),
                ConicKind.PARALLEL_LINES,
                FiniteSectionTopology.MULTIPLE_OPEN_CURVES,
            ),
        )
        for section_id, plane, kind, topology in cases:
            with self.subTest(section=section_id):
                trace = compute_quadric_section(section_id, self.cylinder, plane)
                self.assertIs(trace.supporting_kind, kind)
                self.assertIs(trace.finite_topology, topology)
                self.assert_trace_lies_on_finite_surface(trace, self.cylinder, plane)

    def test_finite_axial_range_can_split_one_supporting_ellipse_into_arcs(self) -> None:
        narrow = CylinderSpec(
            "narrow-cylinder",
            (0, 0, 0),
            (0, 0, 1),
            2.0,
            (-0.5, 0.5),
            radial_axis=(1, 0, 0),
        )
        trace = compute_quadric_section(
            "clipped-ellipse",
            narrow,
            SectionPlane("tilted", (0, 0, 0), (1, 0, 1)),
        )
        self.assertIs(trace.supporting_kind, ConicKind.ELLIPSE)
        self.assertIs(
            trace.finite_topology,
            FiniteSectionTopology.MULTIPLE_OPEN_CURVES,
        )
        self.assertEqual(len(trace.branches), 1)
        self.assertEqual(len(trace.components), 2)
        self.assertTrue(all(not component.closed for component in trace.components))
        self.assert_trace_lies_on_finite_surface(
            trace,
            narrow,
            SectionPlane("tilted", (0, 0, 0), (1, 0, 1)),
        )

    def assert_trace_lies_on_finite_surface(
        self,
        trace,
        surface,
        plane: SectionPlane,
    ) -> None:
        for component in trace.components:
            branch = trace.branch_map[component.branch_id]
            for interval in component.parameter_intervals:
                for parameter in (interval.start, interval.midpoint, interval.end):
                    point = branch.world_point(parameter)
                    self.assertAlmostEqual(
                        surface.support_quadric.evaluate(point), 0.0, places=9
                    )
                    self.assertAlmostEqual(plane.signed_distance(point), 0.0, places=10)
                    axial = float(surface.frame.to_local_point(point)[2])
                    self.assertGreaterEqual(axial, surface.axial_range[0] - 1.0e-9)
                    self.assertLessEqual(axial, surface.axial_range[1] + 1.0e-9)


class FiniteConeSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cone = ConeSpec(
            "cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.0, 5.0),
            radial_axis=(1.0, 0.0, 0.0),
        )

    def test_plane_slope_selects_ellipse_parabola_and_hyperbola(self) -> None:
        expected = (
            (0.5, ConicKind.ELLIPSE, FiniteSectionTopology.CLOSED_CURVE),
            (1.0, ConicKind.PARABOLA, FiniteSectionTopology.OPEN_CURVE),
            (2.0, ConicKind.HYPERBOLA, FiniteSectionTopology.OPEN_CURVE),
        )
        for slope, kind, topology in expected:
            with self.subTest(slope=slope):
                plane = SectionPlane(
                    f"slope-{slope}",
                    (0.0, 0.0, 1.0),
                    (-slope, 0.0, 1.0),
                )
                trace = compute_quadric_section(
                    f"cone-{slope}", self.cone, plane
                )
                self.assertIs(trace.supporting_kind, kind)
                self.assertIs(trace.finite_topology, topology)
                self.assert_trace_lies_on_cone(trace, plane)

    def test_plane_through_apex_returns_two_stable_generator_components(self) -> None:
        plane = SectionPlane("through-apex", (0, 0, 0), (0, 1, 0))
        first = compute_quadric_section("generators", self.cone, plane)
        second = compute_quadric_section("generators", self.cone, plane)
        self.assertEqual(
            canonical_quadric_section_trace_json(first),
            canonical_quadric_section_trace_json(second),
        )
        self.assertIs(first.supporting_kind, ConicKind.INTERSECTING_LINES)
        self.assertIs(
            first.finite_topology,
            FiniteSectionTopology.MULTIPLE_OPEN_CURVES,
        )
        self.assertEqual(
            tuple(branch.branch_id for branch in first.branches),
            (
                "generators:component:intersecting_lines:negative",
                "generators:component:intersecting_lines:positive",
            ),
        )
        self.assert_trace_lies_on_cone(first, plane)

    def test_hyperbola_retains_support_branches_even_when_one_nappe_is_clipped(self) -> None:
        plane = SectionPlane("hyperbola", (0, 0, 1), (-2, 0, 1))
        trace = compute_quadric_section("hyperbola", self.cone, plane)
        self.assertIs(trace.supporting_kind, ConicKind.HYPERBOLA)
        self.assertEqual(
            tuple(branch.branch_id for branch in trace.branches),
            (
                "hyperbola:component:hyperbola:negative",
                "hyperbola:component:hyperbola:positive",
            ),
        )
        self.assertEqual(len(trace.components), 1)
        self.assertIs(trace.finite_topology, FiniteSectionTopology.OPEN_CURVE)

    def assert_trace_lies_on_cone(self, trace, plane: SectionPlane) -> None:
        for component in trace.components:
            branch = trace.branch_map[component.branch_id]
            for interval in component.parameter_intervals:
                for parameter in (interval.start, interval.midpoint, interval.end):
                    point = branch.world_point(parameter)
                    self.assertAlmostEqual(
                        self.cone.support_quadric.evaluate(point), 0.0, places=8
                    )
                    self.assertAlmostEqual(plane.signed_distance(point), 0.0, places=10)
                    axial = float(self.cone.frame.to_local_point(point)[2])
                    self.assertGreaterEqual(axial, -1.0e-8)
                    self.assertLessEqual(axial, 5.0 + 1.0e-8)


class TransformedAndDegenerateSectionTests(unittest.TestCase):
    def test_rotated_translated_cylinder_and_cone_are_scale_invariant(self) -> None:
        axis = np.asarray((1.0, 2.0, 3.0), dtype=float)
        radial_axis = np.asarray((2.0, -1.0, 0.5), dtype=float)
        cylinder_signatures = []
        cone_signatures = []

        for scale in (1.0e-6, 1.0, 1.0e6):
            with self.subTest(scale=scale):
                origin = scale * np.asarray((2.5, -1.75, 0.4), dtype=float)
                cylinder = CylinderSpec(
                    f"scaled-cylinder-{scale}",
                    origin,
                    axis,
                    2.0 * scale,
                    (-3.0 * scale, 3.0 * scale),
                    radial_axis=radial_axis,
                )
                cylinder_plane = SectionPlane(
                    f"scaled-cylinder-plane-{scale}",
                    cylinder.frame.to_world_point((0.0, 0.0, 0.0)),
                    cylinder.frame.to_world_vector((1.0, 0.0, 1.0)),
                    u_axis=cylinder.frame.to_world_vector((0.0, 1.0, 0.0)),
                )
                first_cylinder = compute_quadric_section(
                    "scaled-cylinder-section",
                    cylinder,
                    cylinder_plane,
                )
                second_cylinder = compute_quadric_section(
                    "scaled-cylinder-section",
                    cylinder,
                    cylinder_plane,
                )
                self.assertEqual(
                    canonical_quadric_section_trace_json(first_cylinder),
                    canonical_quadric_section_trace_json(second_cylinder),
                )
                self.assertIs(first_cylinder.supporting_kind, ConicKind.ELLIPSE)
                self.assertIs(
                    first_cylinder.finite_topology,
                    FiniteSectionTopology.CLOSED_CURVE,
                )
                _assert_trace_geometry(
                    self,
                    first_cylinder,
                    cylinder,
                    cylinder_plane,
                )
                cylinder_signatures.append(_interval_signature(first_cylinder))

                cone = ConeSpec(
                    f"scaled-cone-{scale}",
                    origin,
                    axis,
                    pi / 4.0,
                    (-4.0 * scale, 4.0 * scale),
                    radial_axis=radial_axis,
                )
                cone_plane = SectionPlane(
                    f"scaled-cone-plane-{scale}",
                    cone.frame.to_world_point((0.0, 0.0, scale)),
                    cone.frame.to_world_vector((-2.0, 0.0, 1.0)),
                    u_axis=cone.frame.to_world_vector((0.0, 1.0, 0.0)),
                )
                first_cone = compute_quadric_section(
                    "scaled-cone-section",
                    cone,
                    cone_plane,
                )
                second_cone = compute_quadric_section(
                    "scaled-cone-section",
                    cone,
                    cone_plane,
                )
                self.assertEqual(
                    canonical_quadric_section_trace_json(first_cone),
                    canonical_quadric_section_trace_json(second_cone),
                )
                self.assertIs(first_cone.supporting_kind, ConicKind.HYPERBOLA)
                self.assertIs(
                    first_cone.finite_topology,
                    FiniteSectionTopology.MULTIPLE_OPEN_CURVES,
                )
                _assert_trace_geometry(self, first_cone, cone, cone_plane)
                cone_signatures.append(_interval_signature(first_cone))

        for signatures in (cylinder_signatures, cone_signatures):
            for actual in signatures[1:]:
                self.assertEqual(len(actual), len(signatures[0]))
                for actual_component, expected_component in zip(actual, signatures[0]):
                    np.testing.assert_allclose(
                        actual_component,
                        expected_component,
                        rtol=2.0e-9,
                        atol=2.0e-9,
                    )

    def test_double_cone_keeps_both_nappes_in_one_finite_axial_range(self) -> None:
        cone = ConeSpec(
            "double-cone",
            (1.25, -0.5, 2.0),
            (1.0, 2.0, 3.0),
            pi / 4.0,
            (-5.0, 5.0),
            radial_axis=(2.0, -1.0, 0.5),
        )
        plane = SectionPlane(
            "double-cone-plane",
            cone.frame.to_world_point((0.0, 0.0, 1.0)),
            cone.frame.to_world_vector((-2.0, 0.0, 1.0)),
            u_axis=cone.frame.to_world_vector((0.0, 1.0, 0.0)),
        )
        first = compute_quadric_section("double-cone-section", cone, plane)
        second = compute_quadric_section("double-cone-section", cone, plane)
        self.assertEqual(
            canonical_quadric_section_trace_json(first),
            canonical_quadric_section_trace_json(second),
        )
        self.assertIs(first.supporting_kind, ConicKind.HYPERBOLA)
        self.assertIs(
            first.finite_topology,
            FiniteSectionTopology.MULTIPLE_OPEN_CURVES,
        )
        self.assertEqual(len(first.components), 2)
        axial_signs = set()
        for component in first.components:
            interval = component.parameter_intervals[0]
            point = first.branch_map[component.branch_id].world_point(interval.midpoint)
            axial = float(cone.frame.to_local_point(point)[2])
            axial_signs.add(-1 if axial < 0.0 else 1)
        self.assertEqual(axial_signs, {-1, 1})
        _assert_trace_geometry(self, first, cone, plane)

    def test_periodic_component_crossing_zero_and_two_pi_stays_connected(self) -> None:
        cylinder = CylinderSpec(
            "seam-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            2.0,
            (0.5, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "seam-plane",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        trace = compute_quadric_section("seam-section", cylinder, plane)
        repeated = compute_quadric_section("seam-section", cylinder, plane)
        self.assertEqual(
            canonical_quadric_section_trace_json(trace),
            canonical_quadric_section_trace_json(repeated),
        )
        self.assertIs(trace.finite_topology, FiniteSectionTopology.OPEN_CURVE)
        self.assertEqual(len(trace.components), 1)
        intervals = trace.components[0].parameter_intervals
        self.assertEqual(len(intervals), 2)
        self.assertAlmostEqual(intervals[0].start, 0.0, places=14)
        self.assertAlmostEqual(intervals[-1].end, tau, places=14)
        self.assertLess(intervals[0].end, intervals[-1].start)
        branch = trace.branch_map[trace.components[0].branch_id]
        np.testing.assert_allclose(
            branch.world_point(intervals[0].start),
            branch.world_point(intervals[-1].end),
            rtol=0.0,
            atol=1.0e-12,
        )
        _assert_trace_geometry(self, trace, cylinder, plane)

    def test_axial_endpoint_and_cone_apex_touches_are_isolated_points(self) -> None:
        cylinder = CylinderSpec(
            "endpoint-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            2.0,
            (2.0, 3.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        cylinder_plane = SectionPlane(
            "endpoint-plane",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        cylinder_trace = compute_quadric_section(
            "endpoint-touch",
            cylinder,
            cylinder_plane,
        )
        repeated_cylinder_trace = compute_quadric_section(
            "endpoint-touch",
            cylinder,
            cylinder_plane,
        )
        self.assertEqual(
            canonical_quadric_section_trace_json(cylinder_trace),
            canonical_quadric_section_trace_json(repeated_cylinder_trace),
        )
        self.assertIs(cylinder_trace.supporting_kind, ConicKind.ELLIPSE)
        self.assertIs(cylinder_trace.finite_topology, FiniteSectionTopology.POINT)
        self.assertEqual(len(cylinder_trace.components), 0)
        self.assertEqual(len(cylinder_trace.isolated_world_points), 1)
        _assert_trace_geometry(self, cylinder_trace, cylinder, cylinder_plane)

        cone = ConeSpec(
            "apex-cone",
            (2.0, -1.0, 0.5),
            (1.0, 2.0, 3.0),
            pi / 4.0,
            (0.0, 5.0),
            radial_axis=(2.0, -1.0, 0.5),
        )
        apex_plane = SectionPlane(
            "apex-plane",
            cone.apex,
            cone.axis,
            u_axis=cone.radial_axis,
        )
        apex_trace = compute_quadric_section("apex-touch", cone, apex_plane)
        repeated_apex_trace = compute_quadric_section(
            "apex-touch",
            cone,
            apex_plane,
        )
        self.assertEqual(
            canonical_quadric_section_trace_json(apex_trace),
            canonical_quadric_section_trace_json(repeated_apex_trace),
        )
        self.assertIs(apex_trace.supporting_kind, ConicKind.POINT)
        self.assertIs(apex_trace.finite_topology, FiniteSectionTopology.POINT)
        self.assertEqual(len(apex_trace.isolated_world_points), 1)
        np.testing.assert_allclose(
            apex_trace.isolated_world_points[0],
            cone.apex,
            rtol=0.0,
            atol=1.0e-12,
        )
        _assert_trace_geometry(self, apex_trace, cone, apex_plane)

    def test_near_parabolic_planes_do_not_cross_the_numeric_classification_band(self) -> None:
        cone = ConeSpec(
            "near-threshold-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.0, 100.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        cases = (
            (-5.0e-11, ConicKind.PARABOLA),
            (0.0, ConicKind.PARABOLA),
            (5.0e-11, ConicKind.PARABOLA),
        )
        for offset, expected in cases:
            with self.subTest(offset=offset):
                slope = 1.0 + offset
                plane = SectionPlane(
                    f"near-threshold-plane-{offset}",
                    (0.0, 0.0, 1.0),
                    (-slope, 0.0, 1.0),
                    u_axis=(0.0, 1.0, 0.0),
                )
                first = compute_quadric_section(
                    "near-threshold-section",
                    cone,
                    plane,
                )
                second = compute_quadric_section(
                    "near-threshold-section",
                    cone,
                    plane,
                )
                self.assertIs(first.supporting_kind, expected)
                self.assertEqual(
                    canonical_quadric_section_trace_json(first),
                    canonical_quadric_section_trace_json(second),
                )
                _assert_trace_geometry(self, first, cone, plane)


class SectionCurveAdapterTests(unittest.TestCase):
    def test_finite_trace_adapts_directly_to_visibility_curves(self) -> None:
        sphere = SphereSpec("sphere", (0, 0, 0), 2)
        plane = SectionPlane("plane", (0, 0, 0), (0, 0, 1), (1, 0, 0))
        trace = compute_quadric_section("section", sphere, plane)
        curves = section_trace_curves(trace)
        self.assertEqual(len(curves), 1)
        self.assertEqual(curves[0].curve_id, trace.components[0].component_id)
        for parameter in (0.0, pi / 2.0, pi, 3.0 * pi / 2.0):
            point = curves[0].point(parameter)
            np.testing.assert_allclose(
                point,
                trace.world_point(trace.components[0].component_id, parameter),
                rtol=0.0,
                atol=1.0e-12,
            )

    def test_isolated_tangent_point_is_not_invented_as_a_curve(self) -> None:
        sphere = SphereSpec("sphere", (0, 0, 0), 2)
        plane = SectionPlane("tangent", (0, 0, 2), (0, 0, 1), (1, 0, 0))
        trace = compute_quadric_section("section", sphere, plane)
        self.assertEqual(section_trace_curves(trace), ())
        self.assertEqual(len(trace.isolated_world_points), 1)


class RendererNeutralSectionImportTests(unittest.TestCase):
    def test_section_stack_does_not_import_manim(self) -> None:
        script = """
import sys
import polyhedron_visibility.quadrics.conics
import polyhedron_visibility.quadrics.sections
import polyhedron_visibility.quadrics.trace
assert 'manim' not in sys.modules
assert not any(name.startswith('manim.') for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
