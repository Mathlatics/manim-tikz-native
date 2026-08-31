from __future__ import annotations

from dataclasses import replace
from math import pi, sin, sqrt
import json
import unittest

import numpy as np

from polyhedron_visibility.quadrics import (
    ConeModel,
    ConeSpec,
    ConicKind,
    DandelinConicFamily,
    DandelinConstructionError,
    DandelinPlaneSide,
    PlaneDisplayPatchSpec,
    SectionPlane,
    DandelinTeachingOverlayError,
    build_dandelin_teaching_overlay,
    canonical_dandelin_construction_json,
    compute_dandelin_construction,
    compute_quadric_section,
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


def _distance_to_directrix(point: np.ndarray, directrix: object) -> float:
    line_point = np.asarray(directrix.world_point, dtype=float)
    direction = np.asarray(directrix.world_direction, dtype=float)
    return float(np.linalg.norm(np.cross(point - line_point, direction)))


class DandelinConstructionTests(unittest.TestCase):
    def _assert_focus_directrix_law(self, construction: object) -> None:
        trace = compute_quadric_section(
            "law-section",
            construction.cone,
            construction.plane,
        )
        sampled_points: list[np.ndarray] = []
        by_branch = {item.branch_id: item for item in trace.branches}
        for component in trace.components:
            branch = by_branch[component.branch_id]
            for interval in component.parameter_intervals:
                for ratio in (0.2, 0.5, 0.8):
                    parameter = interval.start + ratio * interval.length
                    sampled_points.append(branch.world_point(parameter))
        self.assertTrue(sampled_points)
        for sphere in construction.spheres:
            self.assertIsNotNone(sphere.directrix)
            focus = np.asarray(sphere.focus.world_point, dtype=float)
            for point in sampled_points:
                focus_distance = float(np.linalg.norm(point - focus))
                directrix_distance = _distance_to_directrix(
                    point,
                    sphere.directrix,
                )
                self.assertGreater(directrix_distance, 0.0)
                self.assertAlmostEqual(
                    focus_distance / directrix_distance,
                    construction.eccentricity,
                    places=8,
                )

    def test_ellipse_has_two_finite_spheres_foci_contact_circles_and_directrices(
        self,
    ) -> None:
        construction = compute_dandelin_construction(
            "ellipse-dandelin",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )

        self.assertIs(construction.family, DandelinConicFamily.ELLIPSE)
        self.assertEqual(len(construction.spheres), 2)
        self.assertEqual(len(construction.focus_points), 2)
        self.assertEqual(len(construction.cone_contact_circles), 2)
        self.assertEqual(len(construction.directrices), 2)
        self.assertEqual({item.nappe_sign for item in construction.spheres}, {1})
        self.assertEqual(
            {item.plane_side for item in construction.spheres},
            {DandelinPlaneSide.APEX, DandelinPlaneSide.OPPOSITE},
        )
        for item in construction.spheres:
            self.assertGreater(item.axial_extent[0], construction.cone.axial_range[0])
            self.assertLess(item.axial_extent[1], construction.cone.axial_range[1])
            self.assertEqual(
                item.cone_contact_circle.frame.normal,
                construction.cone.axis,
            )
        self._assert_focus_directrix_law(construction)

    def test_closed_single_cone_supports_a_pure_lateral_ellipse(self) -> None:
        construction = compute_dandelin_construction(
            "closed-ellipse-dandelin",
            _cone(ConeModel.CLOSED_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )

        self.assertIs(construction.family, DandelinConicFamily.ELLIPSE)
        self.assertIs(construction.cone.model, ConeModel.CLOSED_SINGLE)
        self.assertEqual(len(construction.spheres), 2)
        self.assertEqual({item.nappe_sign for item in construction.spheres}, {1})
        for item in construction.spheres:
            self.assertGreater(item.axial_extent[0], 0.0)
            self.assertLess(item.axial_extent[1], 20.0)
        self._assert_focus_directrix_law(construction)

    def test_negative_nappe_ellipse_has_stable_ids_and_canonical_geometry(
        self,
    ) -> None:
        cone = _cone(ConeModel.OPEN_SINGLE, (-20.0, 0.0))
        plane = SectionPlane(
            "negative-section-plane",
            (0.0, 0.0, -2.0),
            _normal_with_axis_dot(0.8),
            u_axis=(0.0, 1.0, 0.0),
        )
        first = compute_dandelin_construction(
            "negative-ellipse-dandelin",
            cone,
            plane,
        )
        second = compute_dandelin_construction(
            "negative-ellipse-dandelin",
            cone,
            plane,
        )

        self.assertIs(first.family, DandelinConicFamily.ELLIPSE)
        self.assertEqual({item.nappe_sign for item in first.spheres}, {-1})
        self.assertEqual(
            tuple(item.sphere_id for item in first.spheres),
            (
                "negative-ellipse-dandelin:sphere:nappe:negative:side:apex",
                "negative-ellipse-dandelin:sphere:nappe:negative:side:opposite",
            ),
        )
        self.assertEqual(
            canonical_dandelin_construction_json(first),
            canonical_dandelin_construction_json(second),
        )
        for item in first.spheres:
            self.assertGreater(item.axial_extent[0], -20.0)
            self.assertLess(item.axial_extent[1], 0.0)
        self._assert_focus_directrix_law(first)

    def test_exact_parabola_has_one_finite_sphere_and_no_infinity_placeholder(
        self,
    ) -> None:
        construction = compute_dandelin_construction(
            "parabola-dandelin",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(sin(HALF_ANGLE)),
        )

        self.assertIs(construction.family, DandelinConicFamily.PARABOLA)
        self.assertEqual(len(construction.spheres), 1)
        self.assertAlmostEqual(construction.eccentricity, 1.0, places=12)
        self.assertIn(":side:apex", construction.spheres[0].sphere_id)
        self._assert_focus_directrix_law(construction)

    def test_negative_nappe_exact_parabola_keeps_one_finite_sphere(self) -> None:
        construction = compute_dandelin_construction(
            "negative-parabola-dandelin",
            _cone(ConeModel.OPEN_SINGLE, (-20.0, 0.0)),
            SectionPlane(
                "negative-parabola-plane",
                (0.0, 0.0, -2.0),
                _normal_with_axis_dot(sin(HALF_ANGLE)),
                u_axis=(0.0, 1.0, 0.0),
            ),
        )

        self.assertIs(construction.family, DandelinConicFamily.PARABOLA)
        self.assertEqual(len(construction.spheres), 1)
        self.assertEqual(construction.spheres[0].nappe_sign, -1)
        self.assertIn(":nappe:negative:", construction.spheres[0].sphere_id)
        self._assert_focus_directrix_law(construction)

    def test_open_double_hyperbola_has_one_sphere_on_each_nappe(self) -> None:
        construction = compute_dandelin_construction(
            "hyperbola-dandelin",
            _cone(ConeModel.OPEN_DOUBLE, (-20.0, 20.0)),
            _plane(0.2),
        )

        self.assertIs(construction.family, DandelinConicFamily.HYPERBOLA)
        self.assertEqual(len(construction.spheres), 2)
        self.assertEqual({item.nappe_sign for item in construction.spheres}, {-1, 1})
        self.assertGreater(construction.eccentricity, 1.0)
        self._assert_focus_directrix_law(construction)

    def test_circular_section_has_two_coincident_foci_and_no_finite_directrix(
        self,
    ) -> None:
        construction = compute_dandelin_construction(
            "circle-dandelin",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 10.0)),
            _plane(1.0),
        )

        self.assertEqual(construction.supporting_kind.value, "circle")
        self.assertEqual(construction.eccentricity, 0.0)
        self.assertEqual(len(construction.spheres), 2)
        self.assertEqual(len(construction.directrices), 0)
        np.testing.assert_allclose(
            construction.focus_points[0].world_point,
            construction.focus_points[1].world_point,
            atol=1.0e-12,
        )

    def test_directrices_clip_to_finite_segments_on_the_section_patch(self) -> None:
        construction = compute_dandelin_construction(
            "clipped-directrices",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        patch = PlaneDisplayPatchSpec(
            "directrix-patch",
            construction.plane.plane_id,
            20.0,
            20.0,
        )

        segments = construction.directrix_segments(patch)

        self.assertEqual(len(segments), 2)
        for directrix, segment in zip(construction.directrices, segments):
            self.assertEqual(segment.curve_id, directrix.directrix_id)
            for endpoint in (segment.start, segment.end):
                coordinates = construction.plane.coordinates_in_plane(endpoint)
                self.assertLessEqual(abs(coordinates[0]), patch.half_width + 1.0e-10)
                self.assertLessEqual(abs(coordinates[1]), patch.half_height + 1.0e-10)

    def test_normal_reversal_preserves_spheres_foci_and_stable_semantic_ids(self) -> None:
        cone = _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0))
        plane = _plane(0.8)
        reversed_plane = SectionPlane(
            plane.plane_id,
            plane.point,
            tuple(-item for item in plane.normal),
            plane.u_axis,
        )
        forward = compute_dandelin_construction("normal-flip", cone, plane)
        backward = compute_dandelin_construction(
            "normal-flip",
            cone,
            reversed_plane,
        )

        self.assertEqual(
            tuple(item.sphere_id for item in forward.spheres),
            tuple(item.sphere_id for item in backward.spheres),
        )
        for left, right in zip(forward.spheres, backward.spheres):
            np.testing.assert_allclose(left.sphere.center, right.sphere.center, atol=1.0e-12)
            self.assertAlmostEqual(left.sphere.radius, right.sphere.radius, places=12)
            np.testing.assert_allclose(
                left.focus.world_point,
                right.focus.world_point,
                atol=1.0e-12,
            )

    def test_similarity_scales_and_rigid_oblique_axis_preserve_the_construction(
        self,
    ) -> None:
        axis = np.asarray((1.0, 2.0, 3.0), dtype=float)
        axis /= np.linalg.norm(axis)
        radial = np.asarray((2.0, -1.0, 0.0), dtype=float)
        radial /= np.linalg.norm(radial)
        axis_dot = 0.8
        normal = axis_dot * axis + sqrt(1.0 - axis_dot * axis_dot) * radial
        for scale in (1.0e-6, 1.0, 1.0e6):
            with self.subTest(scale=scale):
                apex = scale * np.asarray((3.0, -2.0, 1.0))
                cone = ConeSpec(
                    "scaled-cone",
                    apex,
                    axis,
                    HALF_ANGLE,
                    (0.0, 20.0 * scale),
                    radial_axis=radial,
                    model=ConeModel.OPEN_SINGLE,
                )
                plane = SectionPlane(
                    "scaled-plane",
                    apex + 2.0 * scale * axis,
                    normal,
                    u_axis=radial,
                )
                construction = compute_dandelin_construction(
                    "scaled-dandelin",
                    cone,
                    plane,
                )
                self.assertEqual(len(construction.spheres), 2)
                normalized = sorted(
                    (
                        item.axial_center / scale,
                        item.sphere.radius / scale,
                    )
                    for item in construction.spheres
                )
                np.testing.assert_allclose(
                    normalized,
                    ((1.23076923076923, 0.615384615384615),
                     (5.33333333333333, 2.66666666666667)),
                    rtol=1.0e-8,
                    atol=1.0e-8,
                )

    def test_near_parabolic_angles_are_not_snapped_to_the_critical_family(self) -> None:
        for delta, model, bounds, expected in (
            (1.0e-4, ConeModel.OPEN_SINGLE, (0.0, 1.0e6), DandelinConicFamily.ELLIPSE),
            (-1.0e-4, ConeModel.OPEN_DOUBLE, (-1.0e6, 1.0e6), DandelinConicFamily.HYPERBOLA),
        ):
            with self.subTest(delta=delta):
                construction = compute_dandelin_construction(
                    "near-parabola",
                    _cone(model, bounds),
                    _plane(sin(HALF_ANGLE) + delta),
                )
                self.assertIs(construction.family, expected)

    def test_finite_trim_contact_or_missing_room_fails_explicitly(self) -> None:
        plane = _plane(0.8)
        roomy = compute_dandelin_construction(
            "roomy",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            plane,
        )
        upper_contact = max(item.axial_extent[1] for item in roomy.spheres)

        for upper in (4.0, upper_contact):
            with self.subTest(upper=upper):
                with self.assertRaisesRegex(
                    DandelinConstructionError,
                    "cannot contain|rejected axial extents",
                ):
                    compute_dandelin_construction(
                        "too-short",
                        _cone(ConeModel.OPEN_SINGLE, (0.0, upper)),
                        plane,
                    )

    def test_closed_cone_cap_chord_is_not_mislabeled_as_a_pure_conic(self) -> None:
        with self.assertRaisesRegex(DandelinConstructionError, "cap chord"):
            compute_dandelin_construction(
                "closed-parabola",
                _cone(ConeModel.CLOSED_SINGLE, (0.0, 20.0)),
                _plane(sin(HALF_ANGLE)),
            )

    def test_incomplete_or_non_renderable_cone_models_fail_closed(self) -> None:
        with self.assertRaisesRegex(DandelinConstructionError, "OPEN_DOUBLE"):
            compute_dandelin_construction(
                "single-hyperbola",
                _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
                _plane(0.2),
            )
        with self.assertRaisesRegex(DandelinConstructionError, "ANALYTIC_DOUBLE"):
            compute_dandelin_construction(
                "analytic-hyperbola",
                _cone(ConeModel.ANALYTIC_DOUBLE, (-20.0, 20.0)),
                _plane(0.2),
            )

    def test_plane_through_apex_and_degenerate_sections_fail_closed(self) -> None:
        cone = _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0))
        through_apex = SectionPlane(
            "through-apex",
            cone.apex,
            _normal_with_axis_dot(0.8),
            u_axis=(0.0, 1.0, 0.0),
        )
        with self.assertRaisesRegex(DandelinConstructionError, "non-degenerate|degenerate"):
            compute_dandelin_construction(
                "degenerate-dandelin",
                cone,
                through_apex,
            )

    def test_canonical_json_is_deterministic_and_records_finite_fit_evidence(self) -> None:
        construction = compute_dandelin_construction(
            "canonical-dandelin",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )

        first = canonical_dandelin_construction_json(construction)
        second = construction.canonical_json()
        payload = json.loads(first)

        self.assertEqual(first, second)
        self.assertTrue(payload["finiteFitCertified"])
        self.assertEqual(payload["family"], "ellipse")
        self.assertIn("certificationContext", payload)
        self.assertIsNone(payload["coefficientTolerance"])
        self.assertEqual(len(payload["spheres"]), 2)
        self.assertEqual(
            [item["sphere"]["surfaceId"] for item in payload["spheres"]],
            sorted(item["sphere"]["surfaceId"] for item in payload["spheres"]),
        )

    def test_public_dataclass_replacement_cannot_forge_certified_evidence(
        self,
    ) -> None:
        construction = compute_dandelin_construction(
            "tamper-proof-dandelin",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        first, second = construction.spheres

        def with_first(record: object) -> tuple[object, ...]:
            return tuple(
                sorted((record, second), key=lambda item: item.sphere_id)
            )

        tampered_records = (
            replace(
                first,
                nappe_sign=-1,
                axial_center=999.0,
                axial_extent=(998.0, 1000.0),
            ),
            replace(first, focus_id="forged-focus"),
            replace(
                first,
                cone_contact_circle=replace(
                    first.cone_contact_circle,
                    curve_id="forged-contact-circle",
                ),
            ),
            replace(
                first,
                directrix=replace(
                    first.directrix,
                    directrix_id="forged-directrix",
                ),
            ),
        )
        for record in tampered_records:
            with self.subTest(record=record.sphere_id, focus=record.focus_id):
                with self.assertRaises(DandelinConstructionError):
                    replace(
                        construction,
                        spheres=with_first(record),
                    )

        boundary_epsilon = construction.certification_context.epsilon(
            "boundary"
        )
        tiny_drift_records = (
            replace(
                first,
                axial_center=first.axial_center + boundary_epsilon,
            ),
            replace(
                first,
                axial_extent=(
                    first.axial_extent[0] + boundary_epsilon,
                    first.axial_extent[1],
                ),
            ),
            replace(
                first,
                sphere=replace(
                    first.sphere,
                    center=(
                        first.sphere.center[0],
                        first.sphere.center[1],
                        first.sphere.center[2] + boundary_epsilon,
                    ),
                ),
            ),
        )
        for record in tiny_drift_records:
            with self.subTest(tiny_drift=record.sphere_id):
                with self.assertRaises(DandelinConstructionError):
                    replace(construction, spheres=with_first(record))

        widened_context = construction.certification_context.with_overrides(
            boundary=1.0e-3,
        )
        with self.assertRaises(DandelinConstructionError):
            replace(
                construction,
                certification_context=widened_context,
                spheres=with_first(
                    replace(
                        first,
                        axial_center=first.axial_center + 1.0e-2,
                    )
                ),
            )

        negative_zero_record = replace(
            first,
            sphere=replace(
                first.sphere,
                center=(
                    -0.0,
                    first.sphere.center[1],
                    first.sphere.center[2],
                ),
            ),
        )
        negative_zero_construction = replace(
            construction,
            spheres=with_first(negative_zero_record),
        )
        self.assertEqual(
            negative_zero_construction.canonical_json(),
            construction.canonical_json(),
        )

        for changes in (
            {"eccentricity": 42.0},
            {"eccentricity": construction.eccentricity + 1.0e-10},
            {"family": DandelinConicFamily.HYPERBOLA},
            {"supporting_kind": ConicKind.HYPERBOLA},
            {
                "section_frame": replace(
                    construction.section_frame,
                    frame_id="forged-section-frame",
                )
            },
        ):
            with self.subTest(changes=tuple(changes)):
                with self.assertRaises(DandelinConstructionError):
                    replace(construction, **changes)

    def test_nappe_sign_rejects_boolean_float_and_numpy_integer_values(
        self,
    ) -> None:
        construction = compute_dandelin_construction(
            "strict-nappe-sign",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        record = construction.spheres[0]

        for value in (True, 1.0, np.int64(1)):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(
                    DandelinConstructionError,
                    "nappe_sign",
                ):
                    replace(record, nappe_sign=value)

    def test_teaching_overlay_is_explicitly_non_authoritative_and_canonical(self) -> None:
        construction = compute_dandelin_construction(
            "overlay-dandelin",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        overlay = build_dandelin_teaching_overlay(
            construction,
            PlaneDisplayPatchSpec(
                "overlay-patch",
                construction.plane.plane_id,
                20.0,
                20.0,
            ),
        )

        self.assertEqual(overlay.mode, "diagrammatic")
        self.assertFalse(overlay.visibility_authoritative)
        self.assertEqual(len(overlay.sphere_surfaces), 2)
        self.assertEqual(len(overlay.contact_curves), 2)
        self.assertEqual(len(overlay.directrix_curves), 2)
        self.assertEqual(len(overlay.focus_points), 2)
        self.assertEqual(
            json.loads(overlay.canonical_json())["drawOrder"],
            list(overlay.draw_order),
        )

    def test_teaching_overlay_rejects_physical_or_depth_aware_claims(self) -> None:
        construction = compute_dandelin_construction(
            "overlay-policy",
            _cone(ConeModel.OPEN_SINGLE, (0.0, 20.0)),
            _plane(0.8),
        )
        patch = PlaneDisplayPatchSpec(
            "overlay-policy-patch",
            construction.plane.plane_id,
            20.0,
            20.0,
        )
        for mode in ("physical", "depth_aware_diagrammatic"):
            with self.subTest(mode=mode):
                with self.assertRaises(DandelinTeachingOverlayError):
                    build_dandelin_teaching_overlay(
                        construction,
                        patch,
                        mode=mode,
                    )


if __name__ == "__main__":
    unittest.main()
