from __future__ import annotations

import json
from math import asinh, cos, pi, sin, sqrt, tau
import unittest

import numpy as np

from polyhedron_visibility.geometry import GeometryContext
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.topology import ParameterInterval, assert_exact_partition
from polyhedron_visibility.visibility import VisibilityKind
from polyhedron_visibility.quadrics.conics import ConicKind, ConicParameterization
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.critical import (
    CriticalEventKind,
    compute_curve_critical_events,
)
from polyhedron_visibility.quadrics.curves import (
    CircleArcCurve,
    EllipseArcCurve,
    ParametricConicBranch,
    PointMarker3D,
    SegmentCurve,
)
from polyhedron_visibility.quadrics.visibility import (
    compute_curve_visibility,
    compute_point_visibility,
    compute_quadric_visibility,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary,
)
from polyhedron_visibility.quadrics.surface_boundaries import (
    build_surface_boundary_sources,
)


IDENTITY_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
)
POSITIVE_X_VIEW = ParallelView.from_matrix(
    (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    )
)
ISOMETRIC_VIEW = ParallelView.from_matrix(
    (
        (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
        (-1.0 / sqrt(6.0), -1.0 / sqrt(6.0), 2.0 / sqrt(6.0)),
        (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    )
)


def _assert_complete(test: unittest.TestCase, record: object) -> None:
    domain = record.domain
    spans = record.spans
    assert_exact_partition(
        domain,
        (span.interval for span in spans),
        tolerance=record.parameter_tolerance,
    )
    test.assertEqual(spans[0].interval.start, domain.start)
    test.assertEqual(spans[-1].interval.end, domain.end)


def _circle_branch(curve_id: str = "section-circle") -> ParametricConicBranch:
    parameterization = ConicParameterization(
        kind=ConicKind.CIRCLE,
        branch_label="sphere-section",
        origin=(0.0, 0.0),
        first_axis=(1.0, 0.0),
        second_axis=(0.0, 1.0),
        natural_domain=ParameterInterval(0.0, tau),
        closed=True,
    )
    return ParametricConicBranch(
        curve_id=curve_id,
        parameterization=parameterization,
        plane_embedding=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        domain=ParameterInterval(0.0, tau),
    )


class QuadricCriticalEventTests(unittest.TestCase):
    def test_tangent_even_root_is_preserved_but_does_not_hide(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        tangent = SegmentCurve(
            "tangent",
            (-2.0, 1.0, -2.0),
            (2.0, 1.0, -2.0),
        )

        events = compute_curve_critical_events(tangent, (sphere,), IDENTITY_VIEW)
        middle = min(events, key=lambda event: abs(event.parameter - 0.5))
        evidence = tuple(
            item
            for item in middle.evidence
            if item.kind is CriticalEventKind.SUPPORT_TANGENCY
        )
        self.assertAlmostEqual(middle.parameter, 0.5)
        self.assertTrue(evidence)
        self.assertGreaterEqual(evidence[0].multiplicity, 2)

        record = compute_curve_visibility(tangent, (sphere,), IDENTITY_VIEW)
        self.assertEqual(len(record.spans), 1)
        self.assertIs(record.spans[0].kind, VisibilityKind.VISIBLE)

    def test_circle_on_sphere_keeps_surface_and_self_switch_evidence(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        circle = CircleArcCurve(
            "great-circle",
            (0.0, 0.0, 0.0),
            1.0,
            (1.0, 0.0, 0.0),
            radial_axis=(0.0, 1.0, 0.0),
        )

        events = compute_curve_critical_events(circle, (sphere,), IDENTITY_VIEW)
        kinds = {kind for event in events for kind in event.kinds}
        self.assertIn(CriticalEventKind.CURVE_SURFACE_INTERSECTION, kinds)
        self.assertIn(CriticalEventKind.SELF_OCCLUSION_SWITCH, kinds)
        self.assertTrue(
            any(
                event.parameter == 0.0
                and any(item.identically_zero for item in event.evidence)
                for event in events
            )
        )
        self.assertTrue(any(abs(event.parameter - pi) < 1.0e-12 for event in events))

    def test_axial_caps_and_rims_contribute_events(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
        )
        segment = SegmentCurve("behind-cap", (-2.0, 0.0, -2.0), (2.0, 0.0, -2.0))

        events = compute_curve_critical_events(segment, (cylinder,), IDENTITY_VIEW)
        rim_parameters = tuple(
            event.parameter
            for event in events
            if CriticalEventKind.CAP_RIM in event.kinds
        )
        self.assertEqual(len(rim_parameters), 2)
        self.assertAlmostEqual(rim_parameters[0], 0.25)
        self.assertAlmostEqual(rim_parameters[1], 0.75)


class QuadricCurveVisibilityTests(unittest.TestCase):
    def test_surface_boundaries_reuse_one_resolved_context(self) -> None:
        cone = ConeSpec(
            "resolved-context-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 3.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        context = GeometryContext(screen_tolerance=0.001)
        resolved = context.resolve(positions=cone.characteristic_points)

        unresolved_sources = build_surface_boundary_sources(
            (cone,),
            ISOMETRIC_VIEW,
            context=context,
        )
        resolved_sources = build_surface_boundary_sources(
            (cone,),
            ISOMETRIC_VIEW,
            context=resolved,
        )

        self.assertEqual(
            tuple(item.to_dict() for item in resolved_sources),
            tuple(item.to_dict() for item in unresolved_sources),
        )

    def test_near_parabolic_ellipse_uses_only_its_authored_tan_chart(self) -> None:
        normal_angle = 59.5 * pi / 180.0
        cases = (
            ("cone", (0.0, 3.0), 0.02),
            ("frustum", (1.0, 3.0), 3.0 + sqrt(3.0)),
        )
        for name, axial_range, height in cases:
            with self.subTest(case=name):
                cone = ConeSpec(
                    name,
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    pi / 6.0,
                    axial_range,
                    radial_axis=(1.0, 0.0, 0.0),
                    model=ConeModel.CLOSED_SINGLE,
                )
                plane = SectionPlane(
                    f"{name}-plane",
                    (0.0, 0.0, height),
                    (sin(normal_angle), 0.0, cos(normal_angle)),
                    u_axis=(0.0, 1.0, 0.0),
                )
                boundary = compute_quadric_section_boundary(
                    f"{name}-section",
                    cone,
                    plane,
                )

                frame = compute_quadric_visibility(
                    boundary.curves,
                    (cone,),
                    ISOMETRIC_VIEW,
                )

                self.assertEqual(
                    tuple(record.curve_id for record in frame.records),
                    tuple(curve.curve_id for curve in boundary.curves),
                )
                for record in frame.records:
                    _assert_complete(self, record)

    def test_near_edge_on_trim_rim_factors_its_on_surface_discriminant(
        self,
    ) -> None:
        angle = 0.0024
        view = ParallelView.from_matrix(
            (
                (1.0, 0.0, 0.0),
                (0.0, -sin(angle), cos(angle)),
                (0.0, -cos(angle), -sin(angle)),
            )
        )
        cone = ConeSpec(
            "near-edge-on-open-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_SINGLE,
        )
        rim = next(
            source
            for source in build_surface_boundary_sources((cone,), view)
            if source.source_id.endswith("trim_max:rim")
        )

        record = compute_curve_visibility(rim.curve, (cone,), view)

        _assert_complete(self, record)
        self.assertEqual(
            tuple(span.kind for span in record.spans),
            (
                VisibilityKind.VISIBLE,
                VisibilityKind.HIDDEN,
                VisibilityKind.VISIBLE,
            ),
        )
        factored = tuple(
            evidence
            for event in record.critical_events
            for evidence in event.evidence
            if evidence.equation == "ray_discriminant_on_surface_factor"
        )
        self.assertTrue(factored)
        self.assertTrue(
            all(
                evidence.kind is CriticalEventKind.SUPPORT_TANGENCY
                and evidence.multiplicity % 2 == 0
                for evidence in factored
            )
        )

    def test_translated_cone_cap_merges_one_geometric_switch_event(self) -> None:
        records = []
        for horizontal in (0.0, 3.25):
            shift = horizontal * np.asarray(ISOMETRIC_VIEW.matrix[0])
            cone = ConeSpec(
                f"cone:{horizontal}",
                tuple(shift + np.asarray((0.0, 0.0, -2.45))),
                (0.0, 0.0, 1.0),
                pi / 6.0,
                (0.0, 4.0),
                radial_axis=(1.0, 0.0, 0.0),
            )
            rim = next(
                item
                for item in build_surface_boundary_sources(
                    (cone,), ISOMETRIC_VIEW
                )
                if item.source_id.endswith("cap_max:rim")
            )
            records.append(
                compute_curve_visibility(
                    rim.curve,
                    (cone,),
                    ISOMETRIC_VIEW,
                )
            )

        for record in records:
            self.assertEqual(len(record.spans), 1)
            self.assertIs(record.spans[0].kind, VisibilityKind.VISIBLE)
            self.assertTrue(
                any(
                    {
                        CriticalEventKind.SUPPORT_TANGENCY,
                        CriticalEventKind.SELF_OCCLUSION_SWITCH,
                    }
                    <= set(event.kinds)
                    for event in record.critical_events
                )
            )

    def test_sphere_great_circle_has_front_visible_and_back_hidden(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        circle = CircleArcCurve(
            "great-circle",
            (0.0, 0.0, 0.0),
            1.0,
            (1.0, 0.0, 0.0),
            radial_axis=(0.0, 1.0, 0.0),
        )

        record = compute_curve_visibility(circle, (sphere,), IDENTITY_VIEW)
        _assert_complete(self, record)
        self.assertEqual(len(record.spans), 2)
        self.assertEqual(
            tuple(span.kind for span in record.spans),
            (VisibilityKind.VISIBLE, VisibilityKind.HIDDEN),
        )
        self.assertAlmostEqual(record.spans[0].interval.end, pi)
        self.assertEqual(record.spans[1].occluders, ("sphere",))

    def test_segment_and_ellipse_use_analytic_partitions(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        segment = SegmentCurve("segment", (-2.0, 0.0, -2.0), (2.0, 0.0, -2.0))
        ellipse = EllipseArcCurve(
            "ellipse",
            (0.0, 0.0, -3.0),
            (2.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
        )

        segment_record = compute_curve_visibility(segment, (sphere,), IDENTITY_VIEW)
        _assert_complete(self, segment_record)
        self.assertEqual(
            tuple(span.kind for span in segment_record.spans),
            (
                VisibilityKind.VISIBLE,
                VisibilityKind.HIDDEN,
                VisibilityKind.VISIBLE,
            ),
        )
        self.assertAlmostEqual(segment_record.spans[0].interval.end, 0.25)
        self.assertAlmostEqual(segment_record.spans[1].interval.end, 0.75)

        ellipse_record = compute_curve_visibility(ellipse, (sphere,), IDENTITY_VIEW)
        _assert_complete(self, ellipse_record)
        self.assertEqual(
            sum(span.kind is VisibilityKind.HIDDEN for span in ellipse_record.spans),
            2,
        )
        self.assertTrue(
            all(
                CriticalEventKind.SUPPORT_TANGENCY in event.kinds
                for event in ellipse_record.critical_events[1:-1]
                if CriticalEventKind.CHART_SEAM not in event.kinds
            )
        )

    def test_section_circle_branch_uses_same_self_occlusion_rule(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        record = compute_curve_visibility(
            _circle_branch(),
            (sphere,),
            IDENTITY_VIEW,
        )

        _assert_complete(self, record)
        self.assertEqual(len(record.spans), 2)
        self.assertIs(record.spans[0].kind, VisibilityKind.VISIBLE)
        self.assertIs(record.spans[1].kind, VisibilityKind.HIDDEN)
        self.assertAlmostEqual(record.spans[0].interval.end, pi)

    def test_parabola_and_hyperbola_use_polynomial_and_exp_charts(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        parabola = ParametricConicBranch(
            "parabola",
            ConicParameterization(
                ConicKind.PARABOLA,
                "parabola",
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, -3.0),
                (0.0, 0.0, 1.0),
            ),
            ParameterInterval(-2.0, 2.0),
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
            (
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
                (-1.0, 0.0, -2.0),
                (0.0, 0.0, 1.0),
            ),
            ParameterInterval(-2.0, 2.0),
        )

        parabola_record = compute_curve_visibility(parabola, (sphere,), IDENTITY_VIEW)
        hyperbola_record = compute_curve_visibility(hyperbola, (sphere,), IDENTITY_VIEW)
        for record in (parabola_record, hyperbola_record):
            _assert_complete(self, record)
            self.assertEqual(
                tuple(span.kind for span in record.spans),
                (
                    VisibilityKind.VISIBLE,
                    VisibilityKind.HIDDEN,
                    VisibilityKind.VISIBLE,
                ),
            )
        self.assertAlmostEqual(parabola_record.spans[0].interval.end, -1.0)
        self.assertAlmostEqual(parabola_record.spans[1].interval.end, 1.0)
        expected = asinh(1.0)
        self.assertAlmostEqual(hyperbola_record.spans[0].interval.end, -expected)
        self.assertAlmostEqual(hyperbola_record.spans[1].interval.end, expected)

    def test_finite_cylinder_and_cone_do_not_use_infinite_extensions(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
        )
        cone = ConeSpec(
            "cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            0.25 * pi,
            (0.0, 2.0),
        )
        above_cylinder = SegmentCurve(
            "above-cylinder",
            (-2.0, -0.5, 2.0),
            (-2.0, 0.5, 2.0),
        )
        above_cone = SegmentCurve(
            "above-cone",
            (-4.0, -0.5, 3.0),
            (-4.0, 0.5, 3.0),
        )

        for curve, surface in ((above_cylinder, cylinder), (above_cone, cone)):
            with self.subTest(surface=surface.surface_id):
                record = compute_curve_visibility(curve, (surface,), POSITIVE_X_VIEW)
                _assert_complete(self, record)
                self.assertEqual(len(record.spans), 1)
                self.assertIs(record.spans[0].kind, VisibilityKind.VISIBLE)

    def test_oblique_parallel_view_still_splits_self_occlusion(self) -> None:
        view = ParallelView.from_matrix(
            (
                (1.0, 0.0, -0.5),
                (0.0, 1.0, -0.25),
                (0.0, 0.0, 1.0),
            )
        )
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        circle = CircleArcCurve(
            "oblique-great-circle",
            (0.0, 0.0, 0.0),
            1.0,
            (-1.0, 0.0, 0.5),
            # Perpendicular to both the circle normal and the oblique view,
            # so the two front/back switches remain at 0 and pi.
            radial_axis=(-0.1, 1.0, -0.2),
        )

        record = compute_curve_visibility(circle, (sphere,), view)
        _assert_complete(self, record)
        self.assertEqual(len(record.spans), 2)
        self.assertEqual(
            {span.kind for span in record.spans},
            {VisibilityKind.VISIBLE, VisibilityKind.HIDDEN},
        )
        self.assertAlmostEqual(record.spans[0].interval.end, pi)

    def test_multiple_occluders_are_stable_and_all_retained(self) -> None:
        curve = SegmentCurve("line", (-2.0, 0.0, -3.0), (2.0, 0.0, -3.0))
        back = SphereSpec("z-back", (0.0, 0.0, 2.0), 1.0)
        front = SphereSpec("a-front", (0.0, 0.0, 0.0), 1.0)

        record = compute_curve_visibility(curve, (back, front), IDENTITY_VIEW)
        hidden = tuple(span for span in record.spans if not span.visible)
        self.assertEqual(len(hidden), 1)
        self.assertEqual(hidden[0].occluders, ("a-front", "z-back"))

    def test_classification_is_scale_invariant(self) -> None:
        expected = None
        for scale in (1.0e-6, 1.0, 1.0e6):
            with self.subTest(scale=scale):
                sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), scale)
                circle = CircleArcCurve(
                    "circle",
                    (0.0, 0.0, 0.0),
                    scale,
                    (1.0, 0.0, 0.0),
                    radial_axis=(0.0, 1.0, 0.0),
                )
                record = compute_curve_visibility(circle, (sphere,), IDENTITY_VIEW)
                signature = tuple(
                    (span.interval.start, span.interval.end, span.kind.value)
                    for span in record.spans
                )
                if expected is None:
                    expected = signature
                self.assertEqual(signature, expected)

    def test_frame_output_is_serializable_and_order_independent(self) -> None:
        curves = (
            SegmentCurve("z-line", (-2.0, 0.0, -3.0), (2.0, 0.0, -3.0)),
            _circle_branch("a-section"),
        )
        surfaces = (
            SphereSpec("z-sphere", (0.0, 0.0, 2.0), 1.0),
            SphereSpec("a-sphere", (0.0, 0.0, 0.0), 1.0),
        )

        first = compute_quadric_visibility(curves, surfaces, IDENTITY_VIEW)
        second = compute_quadric_visibility(
            tuple(reversed(curves)),
            tuple(reversed(surfaces)),
            IDENTITY_VIEW,
        )
        first_payload = json.dumps(first.to_dict(), sort_keys=True, allow_nan=False)
        second_payload = json.dumps(second.to_dict(), sort_keys=True, allow_nan=False)
        self.assertEqual(first_payload, second_payload)
        self.assertEqual(tuple(first.record_map), ("a-section", "z-line"))

    def test_isolated_point_visibility_uses_exact_forward_ray_hits(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        visible = compute_point_visibility(
            PointMarker3D("front", (0.0, 0.0, 1.0)),
            (sphere,),
            IDENTITY_VIEW,
        )
        hidden = compute_point_visibility(
            PointMarker3D("back", (0.0, 0.0, -1.0)),
            (sphere,),
            IDENTITY_VIEW,
        )
        self.assertTrue(visible.visible)
        self.assertEqual(visible.occluders, ())
        self.assertFalse(hidden.visible)
        self.assertEqual(hidden.occluders, ("sphere",))
        self.assertEqual(hidden.to_dict()["pointId"], "back")


if __name__ == "__main__":
    unittest.main()
