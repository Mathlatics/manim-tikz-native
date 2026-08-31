from __future__ import annotations

from dataclasses import fields, replace
from math import pi, sqrt
import unittest

import numpy as np

from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.dandelin import compute_dandelin_construction
from polyhedron_visibility.quadrics.dandelin_views import (
    DandelinMeridianDiagram2D,
    DandelinSectionPlaneDiagram2D,
    DandelinView2DError,
    build_dandelin_meridian_diagram,
    build_dandelin_section_plane_diagram,
    canonical_dandelin_meridian_diagram_json,
    canonical_dandelin_section_plane_diagram_json,
)


HALF_ANGLE = pi / 6.0


def _normal_with_axis_dot(value: float) -> tuple[float, float, float]:
    return (sqrt(max(0.0, 1.0 - value * value)), 0.0, value)


def _cone(
    model: ConeModel,
    axial_range: tuple[float, float],
    *,
    surface_id: str = "cone",
) -> ConeSpec:
    return ConeSpec(
        surface_id,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        HALF_ANGLE,
        axial_range,
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )


def _plane(axis_dot: float, *, plane_id: str = "section-plane") -> SectionPlane:
    return SectionPlane(
        plane_id,
        (0.0, 0.0, 2.0),
        _normal_with_axis_dot(axis_dot),
        u_axis=(0.0, 1.0, 0.0),
    )


def _assert_point_round_trip(test: unittest.TestCase, item: object) -> None:
    np.testing.assert_allclose(
        item.frame.point_from_coordinates(item.coordinates),
        item.world_point,
        rtol=0.0,
        atol=1.0e-12,
    )


class DandelinMeridianDiagram2DTests(unittest.TestCase):
    def test_ellipse_uses_axis_and_projected_plane_normal_with_true_sphere_circles(
        self,
    ) -> None:
        construction = compute_dandelin_construction(
            "ellipse-view",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )

        diagram = build_dandelin_meridian_diagram(construction)

        self.assertIsInstance(diagram, DandelinMeridianDiagram2D)
        self.assertEqual(diagram.diagram_id, "ellipse-view:view:meridian")
        self.assertEqual(diagram.radial_source, "projected_plane_normal")
        axis = np.asarray(construction.cone.axis, dtype=float)
        normal = np.asarray(construction.plane.normal, dtype=float)
        projected = normal - float(np.dot(normal, axis)) * axis
        projected /= np.linalg.norm(projected)
        self.assertAlmostEqual(
            abs(float(np.dot(projected, diagram.frame.u_axis))),
            1.0,
            places=12,
        )
        np.testing.assert_allclose(diagram.frame.v_axis, axis, atol=1.0e-12)
        self.assertEqual(len(diagram.generators), 2)
        self.assertEqual(len(diagram.sphere_circles), 2)
        self.assertEqual(len(diagram.focus_points), 2)
        self.assertEqual(len(diagram.tangencies), 6)

        by_sphere = {item.sphere_id: item for item in construction.spheres}
        for circle in diagram.sphere_circles:
            record = by_sphere[circle.sphere_id]
            self.assertEqual(
                circle.circle_id,
                f"{record.sphere_id}:meridian-circle",
            )
            self.assertAlmostEqual(circle.center_coordinates[0], 0.0, places=12)
            self.assertAlmostEqual(circle.radius, record.sphere.radius, places=12)
            np.testing.assert_allclose(
                circle.world_center,
                record.sphere.center,
                rtol=0.0,
                atol=diagram.certification_tolerance,
            )
            np.testing.assert_allclose(
                circle.frame.point_from_coordinates(circle.center_coordinates),
                circle.world_center,
                rtol=0.0,
                atol=1.0e-12,
            )

        focus_by_id = {item.source_ref: item for item in diagram.focus_points}
        for record in construction.spheres:
            focus = focus_by_id[record.focus_id]
            _assert_point_round_trip(self, focus)
            homogeneous = np.asarray((*focus.coordinates, 1.0), dtype=float)
            np.testing.assert_allclose(
                np.asarray(diagram.world_embedding) @ homogeneous,
                (*focus.world_point, 1.0),
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                focus.world_point,
                record.focus.world_point,
                rtol=0.0,
                atol=diagram.certification_tolerance,
            )
            self.assertAlmostEqual(
                construction.plane.signed_distance(focus.world_point),
                0.0,
                delta=diagram.certification_tolerance,
            )

        for evidence in diagram.tangencies:
            self.assertLessEqual(
                evidence.circle_residual,
                diagram.certification_tolerance,
            )
            self.assertLessEqual(
                evidence.carrier_residual,
                diagram.certification_tolerance,
            )
            self.assertLessEqual(
                evidence.orthogonality_residual,
                diagram.angular_tolerance,
            )
            _assert_point_round_trip(self, evidence.contact)

        payload = diagram.to_dict()
        self.assertEqual(
            [item["circleId"] for item in payload["sphereCircles"]],
            sorted(item.circle_id for item in diagram.sphere_circles),
        )
        self.assertEqual(
            [item["tangencyId"] for item in payload["tangencies"]],
            sorted(item.tangency_id for item in diagram.tangencies),
        )
        self.assertEqual(
            canonical_dandelin_meridian_diagram_json(diagram),
            diagram.canonical_json(),
        )

    def test_circular_section_falls_back_to_authored_cone_radial_axis(self) -> None:
        construction = compute_dandelin_construction(
            "circle-view",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 10.0)),
            _plane(1.0),
        )

        diagram = build_dandelin_meridian_diagram(construction)

        self.assertEqual(diagram.radial_source, "cone_radial_axis")
        np.testing.assert_allclose(
            diagram.frame.u_axis,
            construction.cone.radial_axis,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            diagram.frame.v_axis,
            construction.cone.axis,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertEqual(len(diagram.sphere_circles), 2)
        self.assertEqual(len(diagram.tangencies), 6)

    def test_open_double_hyperbola_has_both_nappes_and_four_generators(self) -> None:
        construction = compute_dandelin_construction(
            "hyperbola-view",
            _cone(ConeModel.OPEN_DOUBLE, (-20.0, 20.0)),
            _plane(0.2),
        )

        diagram = build_dandelin_meridian_diagram(construction)

        self.assertEqual(len(diagram.generators), 4)
        self.assertEqual(len(diagram.sphere_circles), 2)
        self.assertEqual(len(diagram.tangencies), 6)
        generator_ids = {item.segment_id for item in diagram.generators}
        self.assertTrue(any(":nappe:negative:" in item for item in generator_ids))
        self.assertTrue(any(":nappe:positive:" in item for item in generator_ids))
        for evidence in diagram.tangencies:
            if evidence.carrier_id == diagram.section_line.line_id:
                continue
            record = next(
                item
                for item in construction.spheres
                if item.sphere_id == evidence.sphere_id
            )
            nappe = "positive" if record.nappe_sign > 0 else "negative"
            self.assertIn(f":nappe:{nappe}:", evidence.carrier_id)

    def test_parabola_has_one_true_circle_and_three_tangencies(self) -> None:
        construction = compute_dandelin_construction(
            "parabola-view",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.5),
        )

        diagram = build_dandelin_meridian_diagram(construction)

        self.assertEqual(construction.family.value, "parabola")
        self.assertEqual(len(diagram.sphere_circles), 1)
        self.assertEqual(len(diagram.focus_points), 1)
        self.assertEqual(len(diagram.tangencies), 3)

    def test_oblique_world_frame_round_trips_all_derived_points(self) -> None:
        axis = np.asarray((1.0, 2.0, 3.0), dtype=float)
        axis /= np.linalg.norm(axis)
        radial = np.asarray((2.0, -1.0, 0.0), dtype=float)
        radial /= np.linalg.norm(radial)
        apex = np.asarray((3.0, -2.0, 1.0), dtype=float)
        normal = 0.8 * axis + 0.6 * radial
        construction = compute_dandelin_construction(
            "oblique-view",
            ConeSpec(
                "oblique-cone",
                apex,
                axis,
                HALF_ANGLE,
                (0.0, 20.0),
                radial_axis=radial,
                model=ConeModel.OPEN_SINGLE,
            ),
            SectionPlane(
                "oblique-plane",
                apex + 2.0 * axis,
                normal,
                u_axis=radial,
            ),
        )

        diagram = build_dandelin_meridian_diagram(construction)

        np.testing.assert_allclose(diagram.frame.v_axis, axis, atol=1.0e-12)
        for focus in diagram.focus_points:
            _assert_point_round_trip(self, focus)
        for circle in diagram.sphere_circles:
            np.testing.assert_allclose(
                circle.frame.point_from_coordinates(circle.center_coordinates),
                circle.world_center,
                atol=1.0e-12,
            )
        for evidence in diagram.tangencies:
            _assert_point_round_trip(self, evidence.contact)

    def test_plane_normal_reversal_preserves_frame_and_stable_ids(self) -> None:
        cone = _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0))
        plane = _plane(0.8)
        reversed_plane = SectionPlane(
            plane.plane_id,
            plane.point,
            tuple(-item for item in plane.normal),
            plane.u_axis,
        )
        forward = build_dandelin_meridian_diagram(
            compute_dandelin_construction("normal-flip-view", cone, plane)
        )
        backward = build_dandelin_meridian_diagram(
            compute_dandelin_construction("normal-flip-view", cone, reversed_plane)
        )

        np.testing.assert_allclose(forward.frame.u_axis, backward.frame.u_axis, atol=1e-12)
        np.testing.assert_allclose(forward.frame.v_axis, backward.frame.v_axis, atol=1e-12)
        self.assertEqual(
            tuple(item.circle_id for item in forward.sphere_circles),
            tuple(item.circle_id for item in backward.sphere_circles),
        )
        self.assertEqual(
            tuple(item.tangency_id for item in forward.tangencies),
            tuple(item.tangency_id for item in backward.tangencies),
        )

    def test_tampered_meridian_tangency_fails_closed(self) -> None:
        construction = compute_dandelin_construction(
            "tampered-meridian-view",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        diagram = build_dandelin_meridian_diagram(construction)
        first = replace(
            diagram.tangencies[0],
            carrier_residual=2.0 * diagram.certification_tolerance,
        )
        tampered = tuple(
            sorted((first, *diagram.tangencies[1:]), key=lambda item: item.tangency_id)
        )

        with self.assertRaisesRegex(
            DandelinView2DError,
            "certification tolerance|stale or forged",
        ):
            replace(diagram, tangencies=tampered)

        forged_point = replace(
            diagram.tangencies[0].contact,
            point=diagram.frame.certified_point((999.0, 999.0)),
        )
        forged_evidence = replace(
            diagram.tangencies[0],
            contact=forged_point,
            circle_residual=0.0,
            carrier_residual=0.0,
            orthogonality_residual=0.0,
        )
        forged_tangencies = tuple(
            sorted(
                (forged_evidence, *diagram.tangencies[1:]),
                key=lambda item: item.tangency_id,
            )
        )
        with self.assertRaisesRegex(
            DandelinView2DError,
            "focus|stale or forged|outside its finite segment",
        ):
            replace(diagram, tangencies=forged_tangencies)

        extra_focus = replace(
            diagram.focus_points[0],
            point_id=f"{diagram.diagram_id}:invented-focus",
        )
        with self.assertRaisesRegex(DandelinView2DError, "exactly"):
            replace(
                diagram,
                focus_points=tuple(
                    sorted(
                        (*diagram.focus_points, extra_focus),
                        key=lambda item: item.point_id,
                    )
                ),
            )

    def test_generator_tangency_must_lie_on_the_finite_authored_segment(self) -> None:
        construction = compute_dandelin_construction(
            "finite-generator-view",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        diagram = build_dandelin_meridian_diagram(construction)
        generator = diagram.generators[0]
        displacement = 100.0 * np.asarray(
            generator.direction_coordinates,
            dtype=float,
        )
        moved = replace(
            generator,
            start=diagram.frame.certified_point(
                np.asarray(generator.start.coordinates) + displacement
            ),
            end=diagram.frame.certified_point(
                np.asarray(generator.end.coordinates) + displacement
            ),
        )
        generators = tuple(
            sorted(
                (moved, *diagram.generators[1:]),
                key=lambda item: item.segment_id,
            )
        )

        with self.assertRaisesRegex(DandelinView2DError, "finite segment"):
            replace(diagram, generators=generators)


class DandelinSectionPlaneDiagram2DTests(unittest.TestCase):
    def test_section_plane_exposes_conic_foci_and_directrices_but_no_sphere_circles(
        self,
    ) -> None:
        construction = compute_dandelin_construction(
            "section-plane-view",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )

        diagram = build_dandelin_section_plane_diagram(construction)

        self.assertIsInstance(diagram, DandelinSectionPlaneDiagram2D)
        self.assertEqual(diagram.frame, construction.section_frame)
        self.assertIs(diagram.supporting_kind, construction.supporting_kind)
        self.assertIs(
            diagram.conic_trace.supporting_kind,
            construction.supporting_kind,
        )
        self.assertEqual(len(diagram.focus_points), 2)
        self.assertEqual(len(diagram.directrices), 2)
        self.assertEqual(len(diagram.sphere_plane_tangencies), 2)

        field_names = {item.name for item in fields(DandelinSectionPlaneDiagram2D)}
        self.assertNotIn("sphere_circles", field_names)
        self.assertNotIn("circles", field_names)
        self.assertFalse(hasattr(diagram, "sphere_circles"))
        self.assertFalse(hasattr(diagram, "circles"))
        payload = diagram.to_dict()
        self.assertNotIn("sphereCircles", payload)
        self.assertNotIn("circles", payload)
        self.assertEqual(
            canonical_dandelin_section_plane_diagram_json(diagram),
            diagram.canonical_json(),
        )

        by_focus = {item.source_ref: item for item in diagram.focus_points}
        for record in construction.spheres:
            focus = by_focus[record.focus_id]
            _assert_point_round_trip(self, focus)
            homogeneous = np.asarray((*focus.coordinates, 1.0), dtype=float)
            np.testing.assert_allclose(
                np.asarray(diagram.world_embedding) @ homogeneous,
                (*focus.world_point, 1.0),
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                focus.world_point,
                record.focus.world_point,
                rtol=0.0,
                atol=1.0e-12,
            )
        self.assertEqual(
            tuple(item.source_ref for item in diagram.directrices),
            tuple(sorted(item.directrix_id for item in construction.directrices)),
        )
        for directrix in diagram.directrices:
            np.testing.assert_allclose(
                directrix.frame.point_from_coordinates(directrix.point_coordinates),
                directrix.world_point,
                rtol=0.0,
                atol=1.0e-12,
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(directrix.world_direction)),
                1.0,
                places=12,
            )

        for evidence in diagram.sphere_plane_tangencies:
            self.assertLessEqual(
                evidence.sphere_residual,
                diagram.certification_tolerance,
            )
            self.assertLessEqual(
                evidence.plane_residual,
                diagram.certification_tolerance,
            )
            self.assertLessEqual(
                evidence.normal_alignment_residual,
                diagram.angular_tolerance,
            )
            self.assertIn(evidence.focus.source_ref, by_focus)
            self.assertNotIn("centerCoordinates", evidence.to_dict())

    def test_circle_section_still_does_not_invent_focus_centered_sphere_circles(
        self,
    ) -> None:
        construction = compute_dandelin_construction(
            "circle-section-plane-view",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 10.0)),
            _plane(1.0),
        )

        diagram = build_dandelin_section_plane_diagram(construction)

        self.assertEqual(diagram.supporting_kind.value, "circle")
        self.assertEqual(len(diagram.focus_points), 2)
        self.assertEqual(len(diagram.directrices), 0)
        self.assertFalse(hasattr(diagram, "sphere_circles"))
        self.assertNotIn("sphereCircles", diagram.to_dict())

    def test_near_parabolic_large_scale_contacts_use_the_authoritative_plane(self) -> None:
        cases = (
            (
                "near-parabolic-ellipse-view",
                _cone(ConeModel.OPEN_SINGLE, (0.0, 1.0e6)),
                0.5 + 1.0e-4,
            ),
            (
                "near-parabolic-hyperbola-view",
                _cone(ConeModel.OPEN_DOUBLE, (-1.0e6, 1.0e6)),
                0.5 - 1.0e-4,
            ),
        )
        for construction_id, cone, axis_dot in cases:
            with self.subTest(construction_id=construction_id):
                construction = compute_dandelin_construction(
                    construction_id,
                    cone,
                    _plane(axis_dot),
                )
                diagram = build_dandelin_section_plane_diagram(construction)
                self.assertTrue(diagram.sphere_plane_tangencies)
                for evidence in diagram.sphere_plane_tangencies:
                    self.assertLessEqual(
                        evidence.plane_residual,
                        diagram.certification_tolerance,
                    )

    def test_tampered_sphere_plane_contact_fails_closed(self) -> None:
        construction = compute_dandelin_construction(
            "tampered-section-plane-view",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        diagram = build_dandelin_section_plane_diagram(construction)
        first = replace(
            diagram.sphere_plane_tangencies[0],
            normal_alignment_residual=2.0 * diagram.angular_tolerance,
        )
        tampered = tuple(
            sorted(
                (first, *diagram.sphere_plane_tangencies[1:]),
                key=lambda item: item.tangency_id,
            )
        )

        with self.assertRaisesRegex(
            DandelinView2DError,
            "certification tolerance|stale or forged",
        ):
            replace(diagram, sphere_plane_tangencies=tampered)

        forged = replace(
            diagram.sphere_plane_tangencies[0],
            sphere_center_world=(999.0, 999.0, 999.0),
            sphere_residual=0.0,
            plane_residual=0.0,
            normal_alignment_residual=0.0,
        )
        forged_records = tuple(
            sorted(
                (forged, *diagram.sphere_plane_tangencies[1:]),
                key=lambda item: item.tangency_id,
            )
        )
        with self.assertRaisesRegex(DandelinView2DError, "stale or forged"):
            replace(diagram, sphere_plane_tangencies=forged_records)

        tampered_embedding = tuple(
            tuple(
                value + (1.0 if row_index == 0 and column_index == 2 else 0.0)
                for column_index, value in enumerate(row)
            )
            for row_index, row in enumerate(diagram.conic_trace.plane_embedding)
        )
        with self.assertRaisesRegex(DandelinView2DError, "embedding"):
            replace(
                diagram,
                conic_trace=replace(
                    diagram.conic_trace,
                    plane_embedding=tampered_embedding,
                ),
            )

    def test_trace_and_source_semantics_are_rederived_from_the_construction(self) -> None:
        construction = compute_dandelin_construction(
            "authoritative-section-view",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        diagram = build_dandelin_section_plane_diagram(construction)
        other_construction = compute_dandelin_construction(
            "authoritative-section-view",
            ConeSpec(
                "other-cone",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                pi / 5.0,
                (0.0, 20.0),
                radial_axis=(1.0, 0.0, 0.0),
                model=ConeModel.OPEN_SINGLE,
            ),
            _plane(0.8),
        )
        other_trace = build_dandelin_section_plane_diagram(
            other_construction
        ).conic_trace

        with self.assertRaisesRegex(
            DandelinView2DError,
            "authoritative construction",
        ):
            replace(diagram, conic_trace=other_trace)
        with self.assertRaisesRegex(
            DandelinView2DError,
            "authoritative construction",
        ):
            replace(diagram, construction=other_construction)

    def test_cross_view_focus_identity_uses_local_ids_and_shared_source_refs(self) -> None:
        construction = compute_dandelin_construction(
            "cross-view-identity",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        meridian = build_dandelin_meridian_diagram(construction)
        section = build_dandelin_section_plane_diagram(construction)

        self.assertEqual(
            {item.source_ref for item in meridian.focus_points},
            {item.source_ref for item in section.focus_points},
        )
        self.assertTrue(
            {item.point_id for item in meridian.focus_points}.isdisjoint(
                item.point_id for item in section.focus_points
            )
        )
        for payload in (
            meridian.to_dict()["focusPoints"],
            section.to_dict()["focusPoints"],
        ):
            self.assertTrue(all(item["sourceRef"] for item in payload))

    def test_public_view_primitives_reject_boolean_coordinates(self) -> None:
        construction = compute_dandelin_construction(
            "boolean-view-coordinate",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        diagram = build_dandelin_section_plane_diagram(construction)

        with self.assertRaisesRegex(DandelinView2DError, "finite"):
            replace(
                diagram.directrices[0],
                direction_coordinates=(True, 0.0),
            )
        with self.assertRaisesRegex(DandelinView2DError, "finite"):
            replace(
                diagram.sphere_plane_tangencies[0],
                sphere_center_world=(True, 0.0, 0.0),
            )

    def test_builders_reject_non_construction_inputs(self) -> None:
        for builder in (
            build_dandelin_meridian_diagram,
            build_dandelin_section_plane_diagram,
        ):
            with self.subTest(builder=builder.__name__):
                with self.assertRaisesRegex(TypeError, "DandelinConstruction3D"):
                    builder(object())


if __name__ == "__main__":
    unittest.main()
