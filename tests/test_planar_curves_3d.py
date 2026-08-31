from __future__ import annotations

from dataclasses import replace
import json
from math import pi, tau
import unittest

import numpy as np

from polyhedron_visibility.quadrics import (
    Circle3DSpec,
    Ellipse3DSpec,
    PlanarCurve3DContractError,
    PlanarCurveScene3D,
    PlanarFrame3D,
    PlanarPoint3D,
)
from polyhedron_visibility.quadrics.curves import CircleArcCurve, EllipseArcCurve
from polyhedron_visibility.topology import ParameterInterval


class PlanarFrame3DTests(unittest.TestCase):
    def test_frame_canonicalizes_identity_and_builds_a_right_handed_basis(
        self,
    ) -> None:
        frame = PlanarFrame3D(
            "  oblique-plane  ",
            (1, 2, 3),
            (0, 0, 4),
            (2, 0, 3),
        )

        self.assertEqual(frame.frame_id, "oblique-plane")
        self.assertEqual(frame.point, (1.0, 2.0, 3.0))
        np.testing.assert_allclose(frame.normal, (0.0, 0.0, 1.0), atol=1.0e-12)
        np.testing.assert_allclose(frame.u_axis, (1.0, 0.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(frame.v_axis, (0.0, 1.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(
            np.cross(frame.u_axis, frame.v_axis),
            frame.normal,
            atol=1.0e-12,
        )

    def test_boolean_coordinates_are_not_silently_coerced_to_numbers(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "three finite numbers",
        ):
            PlanarFrame3D(
                "boolean-normal",
                (0.0, 0.0, 0.0),
                (False, 0.0, 1.0),
                (1.0, 0.0, 0.0),
            )

        frame = PlanarFrame3D(
            "numeric-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "two finite numbers",
        ):
            frame.certified_point((True, 0.0))
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "three finite numbers",
        ):
            Circle3DSpec(
                "boolean-center",
                frame,
                (True, 0.0, 0.0),
                1.0,
            )

    def test_canonical_frame_and_point_payloads_round_trip_exactly(self) -> None:
        frame = PlanarFrame3D(
            "round-trip-plane",
            (
                0.2928276418206841,
                -0.9698281416274352,
                -0.30029473129904827,
            ),
            (
                1.4132667908068457,
                0.03110221354462144,
                -0.04297320205816883,
            ),
            (
                -1.7089423051639878,
                -0.29089177871842153,
                -0.42087323476286687,
            ),
        )
        rebuilt = PlanarFrame3D.from_dict(frame.to_dict())

        self.assertEqual(rebuilt, frame)
        self.assertEqual(rebuilt.to_dict(), frame.to_dict())
        self.assertEqual(replace(frame), frame)
        self.assertEqual(frame.affine_frame.x_axis, frame.u_axis)
        self.assertEqual(frame.affine_frame.y_axis, frame.v_axis)
        self.assertEqual(frame.affine_frame.z_axis, frame.normal)
        self.assertTrue(
            all(
                np.array_equal(actual, expected)
                for actual, expected in zip(frame.basis, rebuilt.basis)
            )
        )

        point = frame.certified_point(
            (-806385650.80464, 345678239.24749994)
        )
        rebuilt_point = PlanarPoint3D.from_dict(point.to_dict(), rebuilt)
        self.assertEqual(rebuilt_point, point)
        self.assertEqual(rebuilt_point.to_dict(), point.to_dict())

        invalid_frame = frame.to_dict()
        invalid_frame["vAxis"] = [1.0, 0.0, 0.0]
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "canonical basis",
        ):
            PlanarFrame3D.from_dict(invalid_frame)

        non_planar_payload = {
            "schema": "manim-planar-frame-3d/v1",
            "frameId": "malformed-basis",
            "point": [0.0, 0.0, 0.0],
            "normal": [9.0e-11, 0.0, 1.0],
            "uAxis": [1.0, 0.0, 0.0],
            "vAxis": [0.0, 1.0, 0.0],
            "normalSeed": [9.0e-11, 0.0, 1.0],
            "uAxisSeed": [1.0, 0.0, -9.0e-11],
        }
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "canonical axes|canonical basis|certifiably orthonormal",
        ):
            PlanarFrame3D.from_dict(non_planar_payload)

        non_string_key = frame.to_dict()
        non_string_key[1] = "not-a-field"
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "fields are invalid",
        ):
            PlanarFrame3D.from_dict(non_string_key)

        null_seed = frame.to_dict()
        null_seed["normalSeed"] = None
        null_seed["uAxisSeed"] = [0.0, 1.0, 0.0]
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "normal seed",
        ):
            PlanarFrame3D.from_dict(null_seed)

    def test_nearly_orthogonal_authorship_does_not_bypass_projection(self) -> None:
        frame = PlanarFrame3D(
            "near-orthogonal-authorship",
            (0.0, 0.0, 0.0),
            (1.0e-14, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        self.assertLessEqual(
            abs(float(np.dot(frame.u_axis, frame.normal))),
            64.0 * np.finfo(float).eps,
        )
        point = frame.certified_point((1.0e12, 0.0))
        circle = Circle3DSpec.from_plane_coordinates(
            "near-orthogonal-circle",
            frame,
            point.coordinates,
            1.0e5,
        )
        ellipse = Ellipse3DSpec.from_plane_coordinates(
            "near-orthogonal-ellipse",
            frame,
            point.coordinates,
            1.0e5,
            2.0e5,
        )
        np.testing.assert_allclose(
            circle.lower_to_analytic_curve().normal,
            ellipse.lower_to_analytic_curve().normal,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        )

    def test_frame_rejects_non_finite_or_degenerate_authorship(self) -> None:
        invalid_values = (
            ("point", (0, 0, float("nan")), (0, 0, 1), (1, 0, 0)),
            ("normal", (0, 0, 0), (0, float("inf"), 1), (1, 0, 0)),
            ("u_axis", (0, 0, 0), (0, 0, 1), (float("nan"), 0, 0)),
            ("zero normal", (0, 0, 0), (0, 0, 0), (1, 0, 0)),
            ("parallel u_axis", (0, 0, 0), (0, 0, 1), (0, 0, 2)),
        )
        for label, point, normal, u_axis in invalid_values:
            with self.subTest(label=label):
                with self.assertRaises(PlanarCurve3DContractError):
                    PlanarFrame3D("frame", point, normal, u_axis)

    def test_frame_rejects_a_numerically_unreliable_near_parallel_u_axis(
        self,
    ) -> None:
        normal = (
            0.8179287313623199,
            0.5672712316971092,
            0.09589546444368773,
        )
        in_plane = np.asarray(
            (
                -0.1341648123621885,
                0.025986613909724746,
                0.9906182408078935,
            )
        )
        near_parallel = tuple(
            np.asarray(normal) + 1.5e-14 * in_plane
        )

        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "numerically indistinguishable",
        ):
            PlanarFrame3D(
                "near-parallel-frame",
                (0.0, 0.0, 0.0),
                normal,
                near_parallel,
            )

    def test_dataclass_replace_reauthors_changed_frame_directions(self) -> None:
        frame = PlanarFrame3D(
            "replace-direction-frame",
            (1.0, 2.0, 3.0),
            (1.0, 2.0, 3.0),
            (2.0, -1.0, 0.0),
        )

        scaled = replace(
            frame,
            normal=tuple(2.0 * item for item in frame.normal),
        )
        self.assertEqual(scaled, frame)
        self.assertEqual(scaled.to_dict(), frame.to_dict())

        reauthored = replace(
            frame,
            normal=(0.0, 1.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        self.assertEqual(reauthored.normal, (0.0, 1.0, 0.0))
        self.assertEqual(reauthored.u_axis, (1.0, 0.0, 0.0))
        self.assertEqual(reauthored.v_axis, (0.0, 0.0, -1.0))
        self.assertEqual(
            PlanarFrame3D.from_dict(reauthored.to_dict()),
            reauthored,
        )

        automatic = replace(frame, normal=(0.0, 0.0, 1.0), u_axis=None)
        self.assertEqual(automatic.normal, (0.0, 0.0, 1.0))
        self.assertEqual(
            PlanarFrame3D.from_dict(automatic.to_dict()),
            automatic,
        )

    def test_frame_to_dict_is_canonical_for_equivalent_authorship(self) -> None:
        scaled = PlanarFrame3D(
            " frame ",
            (1, 2, 3),
            (0, 0, 8),
            (4, 0, 5),
        )
        canonical = PlanarFrame3D(
            "frame",
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )

        self.assertEqual(scaled.to_dict(), canonical.to_dict())
        self.assertEqual(
            canonical.to_dict(),
            {
                "schema": "manim-planar-frame-3d/v1",
                "frameId": "frame",
                "point": [1.0, 2.0, 3.0],
                "normal": [0.0, 0.0, 1.0],
                "uAxis": [1.0, 0.0, 0.0],
                "vAxis": [0.0, 1.0, 0.0],
                "normalSeed": [0.0, 0.0, 1.0],
                "uAxisSeed": [1.0, 0.0, 0.0],
            },
        )

        for scale in (1.0e-300, 1.0e308):
            with self.subTest(radial_scale=scale):
                extreme = PlanarFrame3D(
                    "frame",
                    (1.0, 2.0, 3.0),
                    (0.0, 0.0, 1.0),
                    (scale, 0.0, 0.0),
                )
                self.assertEqual(extreme.to_dict(), canonical.to_dict())

    def test_large_world_coordinates_do_not_make_plane_membership_overly_loose(
        self,
    ) -> None:
        frame = PlanarFrame3D(
            "large-plane",
            (1.0e12, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        circle = Circle3DSpec(
            "large-circle",
            frame,
            (1.0e12, 1.0e12, -1.0e12),
            1.0e6,
        )
        self.assertEqual(circle.center, (1.0e12, 1.0e12, -1.0e12))

        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "does not lie",
        ):
            Circle3DSpec(
                "off-plane",
                frame,
                (1.0e12 + 1.0, 1.0e12, -1.0e12),
                1.0e6,
            )

        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "does not lie",
        ):
            Circle3DSpec(
                "ulp-sized-off-plane",
                frame,
                (1.0e12 + 0.001, 1.0e12, -1.0e12),
                1.0e6,
            )

    def test_large_oblique_frame_accepts_its_own_generated_points(self) -> None:
        frame = PlanarFrame3D(
            "large-oblique-plane",
            (1.0e12, -1.0e12, 1.0e12),
            (1.0, 1.0, 1.0),
            (1.0, -1.0, 0.0),
        )
        generated_center = frame.certified_point((1.0, 2.0))

        circle = Circle3DSpec(
            "round-trip-circle",
            frame,
            generated_center,
            1.0e6,
        )

        self.assertEqual(circle.center_coordinates, (1.0, 2.0))
        self.assertEqual(circle.center, generated_center.world_point)

    def test_large_parameter_oblique_frames_accept_generated_points(self) -> None:
        cases = (
            (
                (
                    0.2928276418206841,
                    -0.9698281416274352,
                    -0.30029473129904827,
                ),
                (
                    1.4132667908068457,
                    0.03110221354462144,
                    -0.04297320205816883,
                ),
                (
                    -1.7089423051639878,
                    -0.29089177871842153,
                    -0.42087323476286687,
                ),
                (-806385650.80464, 345678239.24749994),
                1.0e-4,
            ),
            (
                (
                    999999974545.5985,
                    -1000000013149.2682,
                    1000000065638.9359,
                ),
                (
                    -0.16339442802267765,
                    -0.017823951555097507,
                    -2.0393041821364486,
                ),
                (
                    0.050700283512341654,
                    0.23215715911284332,
                    0.7554161502498049,
                ),
                (980466899147.6327, -83336070685.51752),
                1.0e-2,
            ),
        )
        for index, (point, normal, u_axis, uv, off_plane) in enumerate(cases):
            with self.subTest(index=index):
                frame = PlanarFrame3D(
                    f"large-parameter-plane-{index}",
                    point,
                    normal,
                    u_axis,
                )
                generated = frame.certified_point(uv)
                world_spacing = float(
                    np.max(np.abs(np.spacing(generated.world_point)))
                )
                safe_radius = max(1.0, 1.0e9 * world_spacing)

                circle = Circle3DSpec(
                    f"large-parameter-circle-{index}",
                    frame,
                    generated,
                    safe_radius,
                )
                self.assertEqual(circle.radius, safe_radius)
                self.assertEqual(circle.center_coordinates, uv)

                displaced = (
                    np.asarray(generated.world_point)
                    + off_plane * np.asarray(frame.normal)
                )
                with self.assertRaisesRegex(
                    PlanarCurve3DContractError,
                    "does not lie",
                ):
                    Circle3DSpec(
                        f"off-plane-circle-{index}",
                        frame,
                        tuple(displaced),
                        safe_radius,
                    )

    def test_world_coordinate_membership_has_no_feature_size_epsilon(
        self,
    ) -> None:
        frame = PlanarFrame3D(
            "tiny-feature-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )

        with self.assertRaisesRegex(PlanarCurve3DContractError, "does not lie"):
            Circle3DSpec(
                "tiny-off-plane-circle",
                frame,
                (0.0, 0.0, 1.0e-14),
                1.0e-15,
            )

        with self.assertRaisesRegex(PlanarCurve3DContractError, "does not lie"):
            Circle3DSpec(
                "large-radius-does-not-relax-membership",
                frame,
                (0.0, 0.0, 1.0e-6),
                1.0e8,
            )

    def test_certified_point_preserves_large_local_coordinates(self) -> None:
        frame = PlanarFrame3D(
            "large-certified-plane",
            (1.0e12, -1.0e12, 1.0e12),
            (1.0, 1.0, 1.0),
            (1.0, -1.0, 0.0),
        )
        coordinates = (980_466_899_147.6327, -83_336_070_685.51752)

        center = frame.certified_point(coordinates)
        circle = Circle3DSpec("large-circle", frame, center, 2.0e8)
        ellipse = Ellipse3DSpec("large-ellipse", frame, center, 3.0e8, 1.0e8)

        self.assertIsInstance(center, PlanarPoint3D)
        self.assertEqual(center.frame, frame)
        self.assertEqual(center.coordinates, coordinates)
        self.assertEqual(circle.center, center.world_point)
        self.assertEqual(circle.center_coordinates, coordinates)
        self.assertEqual(ellipse.center, center.world_point)
        self.assertEqual(ellipse.center_coordinates, coordinates)
        self.assertEqual(
            circle.to_dict()["centerCoordinates"],
            list(coordinates),
        )
        self.assertEqual(
            ellipse.to_dict()["centerCoordinates"],
            list(coordinates),
        )

    def test_certified_point_payload_is_canonical_and_normalizes_signed_zero(
        self,
    ) -> None:
        frame = PlanarFrame3D(
            "zero-point-plane",
            (-0.0, 0.0, -0.0),
            (-0.0, 0.0, 1.0),
            (1.0, -0.0, 0.0),
        )

        point = frame.certified_point((-0.0, 2.0))
        payload = point.to_dict()

        self.assertEqual(
            payload,
            {
                "schema": "manim-planar-point-3d/v1",
                "frameId": "zero-point-plane",
                "coordinates": [0.0, 2.0],
                "worldPoint": [0.0, 2.0, 0.0],
            },
        )
        self.assertNotIn("-0.0", json.dumps(payload, sort_keys=True))

    def test_certified_point_rejects_a_different_supporting_frame(self) -> None:
        source = PlanarFrame3D(
            "source-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        other = PlanarFrame3D(
            "other-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        center = source.certified_point((2.0, -3.0))

        with self.assertRaisesRegex(PlanarCurve3DContractError, "frame"):
            Circle3DSpec("mismatched-circle", other, center, 1.0)
        with self.assertRaisesRegex(PlanarCurve3DContractError, "frame"):
            Ellipse3DSpec("mismatched-ellipse", other, center, 2.0, 1.0)

    def test_certified_point_rejects_non_finite_or_overflowing_coordinates(
        self,
    ) -> None:
        frame = PlanarFrame3D(
            "overflow-plane",
            (1.0e308, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )

        for coordinates in (
            (float("nan"), 0.0),
            (float("inf"), 0.0),
            (1.0e308, 0.0),
        ):
            with self.subTest(coordinates=coordinates):
                with np.errstate(all="ignore"):
                    with self.assertRaises(PlanarCurve3DContractError):
                        frame.certified_point(coordinates)

    def test_large_oblique_world_point_with_normal_offset_is_not_certified(
        self,
    ) -> None:
        frame = PlanarFrame3D(
            "strict-oblique-plane",
            (1.0e12, -1.0e12, 1.0e12),
            (1.0, 1.0, 1.0),
            (1.0, -1.0, 0.0),
        )
        on_plane = frame.certified_point((8.0e8, -3.0e8)).world_point
        displaced = (
            np.asarray(on_plane)
            + 1.0e-3 * np.asarray(frame.normal)
        )

        with self.assertRaisesRegex(PlanarCurve3DContractError, "does not lie"):
            Circle3DSpec(
                "off-oblique-plane",
                frame,
                tuple(displaced),
                1.0e8,
            )

    def test_fixed_seed_certified_point_scan_is_deterministic(self) -> None:
        rng = np.random.default_rng(20260829)

        for index in range(32):
            normal = rng.normal(size=3)
            authored_u = rng.normal(size=3)
            authored_u -= (
                np.dot(authored_u, normal)
                / np.dot(normal, normal)
            ) * normal
            frame = PlanarFrame3D(
                f"seeded-frame-{index:02d}",
                tuple(rng.uniform(-1.0e12, 1.0e12, size=3)),
                tuple(normal),
                tuple(authored_u),
            )
            coordinates = tuple(
                rng.uniform(-1.0e12, 1.0e12, size=2)
            )
            center = frame.certified_point(coordinates)
            world_spacing = float(
                np.max(np.abs(np.spacing(center.world_point)))
            )
            safe_radius = max(1.0, 1.0e9 * world_spacing)
            curve = Circle3DSpec(
                f"seeded-circle-{index:02d}",
                frame,
                center,
                safe_radius * (1.0 + index / 32.0),
            )

            self.assertEqual(curve.center_coordinates, coordinates)
            self.assertTrue(
                np.all(np.isfinite(curve.lower_to_analytic_curve().point(0.0)))
            )

            world_point = np.asarray(center.world_point)
            normal_unit = np.asarray(frame.normal)
            ulp = float(np.max(np.abs(np.spacing(world_point))))
            perturbation = 128.0 * ulp / float(np.max(np.abs(normal_unit)))
            displaced = world_point + perturbation * normal_unit
            self.assertFalse(np.array_equal(displaced, world_point))
            with self.assertRaisesRegex(
                PlanarCurve3DContractError,
                "does not lie",
            ):
                Circle3DSpec(
                    f"seeded-off-plane-{index:02d}",
                    frame,
                    tuple(displaced),
                    1.0e12,
                )

    def test_large_translation_cannot_swallow_circle_or_ellipse_axes(
        self,
    ) -> None:
        frame = PlanarFrame3D(
            "unrepresentable-axis-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        center = frame.certified_point((1.0e18, 1.0e18))

        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "semi-axis is not representable",
        ):
            Circle3DSpec("collapsed-circle", frame, center, 1.0)
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "semi-axis is not representable",
        ):
            Ellipse3DSpec("collapsed-ellipse", frame, center, 1.0, 2.0)

    def test_certified_center_embedding_has_a_curve_scale_error_budget(
        self,
    ) -> None:
        frame = PlanarFrame3D(
            "rounded-center-plane",
            (1.0e16, 1.0e16, 1.0e16),
            (1.0, 1.0, 1.0),
            (1.0, -1.0, 0.0),
        )
        center = frame.certified_point((1.0, 2.0))

        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "curve-scale error budget",
        ):
            Circle3DSpec("small-rounded-circle", frame, center, 1.0e5)
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "curve-scale error budget",
        ):
            Ellipse3DSpec(
                "small-rounded-ellipse",
                frame,
                center,
                2.0e5,
                1.0e5,
            )

        circle = Circle3DSpec(
            "large-rounded-circle",
            frame,
            center,
            1.0e10,
        )
        ellipse = Ellipse3DSpec(
            "large-rounded-ellipse",
            frame,
            center,
            1.0e10,
            5.0e9,
        )
        self.assertLessEqual(
            abs(frame.signed_distance(circle.center)),
            np.sqrt(np.finfo(float).eps) * circle.radius,
        )
        self.assertLessEqual(
            abs(frame.signed_distance(ellipse.center)),
            np.sqrt(np.finfo(float).eps) * min(
                ellipse.semi_u,
                ellipse.semi_v,
            ),
        )

    def test_raw_world_center_is_never_silently_snapped(self) -> None:
        frame = PlanarFrame3D(
            "raw-center-plane",
            (1.0e16, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        authored_center = (1.0e16 + 2.0, 0.0, 0.0)

        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "curve-scale error budget",
        ):
            Circle3DSpec(
                "small-off-plane-circle",
                frame,
                authored_center,
                1.0,
            )

        accepted = Circle3DSpec(
            "large-off-plane-circle",
            frame,
            authored_center,
            1.0e10,
        )
        self.assertEqual(accepted.center, authored_center)
        self.assertNotEqual(accepted.center, frame.point)
        self.assertEqual(
            Circle3DSpec.from_dict(accepted.to_dict(), frame),
            accepted,
        )


class PlanarCurve3DSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = PlanarFrame3D(
            "support",
            (1.0, 2.0, 3.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

    def test_circle_has_stable_ids_and_canonical_payload(self) -> None:
        circle = Circle3DSpec(
            "  circle-a  ",
            self.frame,
            (1, 4, 5),
            2,
            domain=ParameterInterval(-pi, pi),
        )

        self.assertEqual(circle.curve_id, "circle-a")
        self.assertEqual(circle.frame.frame_id, "support")
        self.assertEqual(
            circle.to_dict(),
            {
                "schema": "manim-planar-curve-3d/v1",
                "kind": "circle",
                "curveId": "circle-a",
                "frameId": "support",
                "center": [1.0, 4.0, 5.0],
                "centerCoordinates": [2.0, 2.0],
                "radius": 2.0,
                "domain": [-pi, pi],
            },
        )

    def test_ellipse_has_stable_ids_and_canonical_payload(self) -> None:
        ellipse = Ellipse3DSpec(
            "  ellipse-a  ",
            self.frame,
            (1, -2, 0),
            3,
            1.5,
        )

        self.assertEqual(ellipse.curve_id, "ellipse-a")
        self.assertEqual(ellipse.frame.frame_id, "support")
        self.assertEqual(
            ellipse.to_dict(),
            {
                "schema": "manim-planar-curve-3d/v1",
                "kind": "ellipse",
                "curveId": "ellipse-a",
                "frameId": "support",
                "center": [1.0, -2.0, 0.0],
                "centerCoordinates": [-4.0, -3.0],
                "semiU": 3.0,
                "semiV": 1.5,
                "domain": [0.0, tau],
            },
        )

    def test_compact_coordinate_factories_match_certified_point_payloads(
        self,
    ) -> None:
        coordinates = (2.5, -1.25)
        domain = ParameterInterval(-pi / 3.0, pi / 2.0)

        direct_circle = Circle3DSpec(
            "coordinate-circle",
            self.frame,
            self.frame.certified_point(coordinates),
            2.25,
            domain=domain,
        )
        compact_circle = Circle3DSpec.from_plane_coordinates(
            "coordinate-circle",
            self.frame,
            coordinates,
            2.25,
            domain=domain,
        )
        direct_ellipse = Ellipse3DSpec(
            "coordinate-ellipse",
            self.frame,
            self.frame.certified_point(coordinates),
            3.5,
            1.25,
            domain=domain,
        )
        compact_ellipse = Ellipse3DSpec.from_plane_coordinates(
            "coordinate-ellipse",
            self.frame,
            coordinates,
            3.5,
            1.25,
            domain=domain,
        )

        self.assertEqual(compact_circle.to_dict(), direct_circle.to_dict())
        self.assertEqual(compact_ellipse.to_dict(), direct_ellipse.to_dict())

    def test_dataclass_replace_preserves_certified_center_evidence(self) -> None:
        frame = PlanarFrame3D(
            "replace-plane",
            (
                0.2928276418206841,
                -0.9698281416274352,
                -0.30029473129904827,
            ),
            (
                1.4132667908068457,
                0.03110221354462144,
                -0.04297320205816883,
            ),
            (
                -1.7089423051639878,
                -0.29089177871842153,
                -0.42087323476286687,
            ),
        )
        coordinates = (-806385650.80464, 345678239.24749994)
        circle = Circle3DSpec.from_plane_coordinates(
            "replace-circle",
            frame,
            coordinates,
            1.0e4,
        )
        ellipse = Ellipse3DSpec.from_plane_coordinates(
            "replace-ellipse",
            frame,
            coordinates,
            2.0e4,
            1.0e4,
        )

        self.assertEqual(replace(circle), circle)
        self.assertEqual(replace(ellipse), ellipse)
        canonical_circle = json.dumps(
            circle.to_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        canonical_ellipse = json.dumps(
            ellipse.to_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(
            json.dumps(
                replace(circle).to_dict(),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            canonical_circle,
        )
        self.assertEqual(
            json.dumps(
                replace(ellipse).to_dict(),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            canonical_ellipse,
        )
        resized_circle = replace(circle, radius=2.0e4)
        resized_ellipse = replace(ellipse, semi_v=1.5e4)
        self.assertEqual(resized_circle.center_coordinates, coordinates)
        self.assertEqual(resized_ellipse.center_coordinates, coordinates)
        self.assertEqual(resized_circle.to_dict()["center"], circle.to_dict()["center"])
        self.assertEqual(resized_ellipse.to_dict()["center"], ellipse.to_dict()["center"])

        moved_point = replace(
            frame.certified_point((0.0, 0.0)),
            coordinates=(3.0, -2.0),
        )
        moved_circle = replace(circle, center=moved_point)
        moved_ellipse = replace(ellipse, center=moved_point)
        self.assertEqual(moved_circle.center_coordinates, (3.0, -2.0))
        self.assertEqual(moved_ellipse.center_coordinates, (3.0, -2.0))
        self.assertEqual(moved_circle.center, moved_point.world_point)
        self.assertEqual(moved_ellipse.center, moved_point.world_point)

        raw_moved_circle = replace(
            circle,
            center=moved_point.world_point,
            center_coordinates=None,
        )
        self.assertEqual(raw_moved_circle.center, moved_point.world_point)
        self.assertEqual(
            raw_moved_circle.center_coordinates,
            frame.coordinates_in_plane(moved_point.world_point),
        )

        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "does not lie|disagrees with center_coordinates",
        ):
            replace(circle, center=(0.0, 0.0, 0.0))

    def test_curve_centers_must_be_finite_and_lie_on_the_supporting_plane(
        self,
    ) -> None:
        invalid_centers = (
            (1.25, 0.0, 0.0),
            (float("nan"), 0.0, 0.0),
            (float("inf"), 0.0, 0.0),
        )
        for center in invalid_centers:
            with self.subTest(kind="circle", center=center):
                with self.assertRaises(PlanarCurve3DContractError):
                    Circle3DSpec("circle", self.frame, center, 1.0)
            with self.subTest(kind="ellipse", center=center):
                with self.assertRaises(PlanarCurve3DContractError):
                    Ellipse3DSpec("ellipse", self.frame, center, 2.0, 1.0)

        tiny_frame = PlanarFrame3D(
            "tiny-support",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(PlanarCurve3DContractError, "does not lie"):
            Circle3DSpec(
                "tiny-circle",
                tiny_frame,
                (0.0, 0.0, 5.0e-11),
                1.0e-12,
            )
        with self.assertRaisesRegex(PlanarCurve3DContractError, "does not lie"):
            Ellipse3DSpec(
                "tiny-ellipse",
                tiny_frame,
                (0.0, 0.0, 5.0e-11),
                2.0e-12,
                1.0e-12,
            )

    def test_radius_and_semi_axes_must_be_finite_and_positive(self) -> None:
        for radius in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(radius=radius):
                with self.assertRaises(PlanarCurve3DContractError):
                    Circle3DSpec("circle", self.frame, (1, 0, 0), radius)

        for semi_u, semi_v in (
            (0.0, 1.0),
            (-1.0, 1.0),
            (float("nan"), 1.0),
            (2.0, 0.0),
            (2.0, float("inf")),
            (2.0, True),
        ):
            with self.subTest(semi_u=semi_u, semi_v=semi_v):
                with self.assertRaises(PlanarCurve3DContractError):
                    Ellipse3DSpec(
                        "ellipse",
                        self.frame,
                        (1, 0, 0),
                        semi_u,
                        semi_v,
                    )

    def test_extreme_scales_fail_before_an_unsafe_analytic_lowering(self) -> None:
        for radius in (1.0e-200, 1.0e-160, 1.0e100, 1.0e155):
            with self.subTest(kind="circle", radius=radius):
                with self.assertRaisesRegex(
                    PlanarCurve3DContractError,
                    "certifiable numeric range|cannot be lowered",
                ):
                    Circle3DSpec("circle", self.frame, (1, 0, 0), radius)

        for semi_u, semi_v in (
            (1.0e-200, 1.0),
            (1.0e-160, 1.0),
            (1.0e-160, 1.0e-160),
            (1.0e100, 1.0e100),
            (1.0e155, 1.0e155),
        ):
            with self.subTest(kind="ellipse", semi_u=semi_u, semi_v=semi_v):
                with self.assertRaisesRegex(
                    PlanarCurve3DContractError,
                    "certifiable numeric range|cannot be lowered",
                ):
                    Ellipse3DSpec(
                        "ellipse",
                        self.frame,
                        (1, 0, 0),
                        semi_u,
                        semi_v,
                    )

    def test_curve_domain_is_finite_positive_and_at_most_one_revolution(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            Circle3DSpec("circle", self.frame, (1, 0, 0), 1, domain=(0, tau))
        with self.assertRaises(PlanarCurve3DContractError):
            Ellipse3DSpec(
                "ellipse",
                self.frame,
                (1, 0, 0),
                2,
                1,
                domain=ParameterInterval(0.0, tau + 1.0e-6),
            )

    def test_circle_lowers_to_the_existing_analytic_circle_contract(self) -> None:
        circle = Circle3DSpec(
            "circle",
            self.frame,
            (1, 4, 5),
            2.5,
            domain=ParameterInterval(0.0, pi),
        )

        lowered = circle.lower_to_analytic_curve()

        self.assertIsInstance(lowered, CircleArcCurve)
        self.assertEqual(lowered.curve_id, circle.curve_id)
        self.assertEqual(lowered.center, circle.center)
        self.assertEqual(lowered.domain, circle.domain)
        self.assertEqual(lowered.radius, circle.radius)
        np.testing.assert_allclose(lowered.normal, self.frame.normal, atol=1.0e-12)
        np.testing.assert_allclose(
            lowered.first_axis,
            np.asarray(self.frame.u_axis) * circle.radius,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            lowered.second_axis,
            np.asarray(self.frame.v_axis) * circle.radius,
            atol=1.0e-12,
        )

    def test_ellipse_lowers_to_the_existing_analytic_ellipse_contract(self) -> None:
        ellipse = Ellipse3DSpec(
            "ellipse",
            self.frame,
            (1, -2, 4),
            3.0,
            1.25,
            domain=ParameterInterval(-pi / 2.0, pi / 2.0),
        )

        lowered = ellipse.lower_to_analytic_curve()

        self.assertIsInstance(lowered, EllipseArcCurve)
        self.assertNotIsInstance(lowered, CircleArcCurve)
        self.assertEqual(lowered.curve_id, ellipse.curve_id)
        self.assertEqual(lowered.center, ellipse.center)
        self.assertEqual(lowered.domain, ellipse.domain)
        np.testing.assert_allclose(
            lowered.first_axis,
            np.asarray(self.frame.u_axis) * ellipse.semi_u,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            lowered.second_axis,
            np.asarray(self.frame.v_axis) * ellipse.semi_v,
            atol=1.0e-12,
        )

    def test_equal_ellipse_axes_preserve_authored_ellipse_kind(self) -> None:
        ellipse = Ellipse3DSpec(
            "authored-ellipse",
            self.frame,
            (1, 0, 0),
            2.0,
            2.0,
        )

        lowered = ellipse.lower_to_analytic_curve()

        self.assertIs(type(lowered), EllipseArcCurve)
        self.assertTrue(lowered.circular)
        self.assertEqual(ellipse.to_dict()["kind"], "ellipse")


class PlanarCurveScene3DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame_a = PlanarFrame3D(
            "frame-a",
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 0),
        )
        self.frame_z = PlanarFrame3D(
            "frame-z",
            (2, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
        )

    def test_scene_canonicalizes_registry_order_and_json(self) -> None:
        circle = Circle3DSpec("curve-z", self.frame_z, (2, 0, 0), 1)
        ellipse = Ellipse3DSpec("curve-a", self.frame_a, (0, 0, 0), 2, 1)
        first = PlanarCurveScene3D(
            frames=(self.frame_z, self.frame_a),
            curves=(circle, ellipse),
        )
        second = PlanarCurveScene3D(
            frames=(self.frame_a, self.frame_z),
            curves=(ellipse, circle),
        )

        self.assertEqual(
            tuple(item.frame_id for item in first.frames),
            ("frame-a", "frame-z"),
        )
        self.assertEqual(
            tuple(item.curve_id for item in first.curves),
            ("curve-a", "curve-z"),
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(
            json.loads(first.canonical_json()),
            first.to_dict(),
        )

    def test_large_local_coordinate_scene_round_trips_without_drift(self) -> None:
        frame = PlanarFrame3D(
            "oblique-round-trip",
            (
                0.2928276418206841,
                -0.9698281416274352,
                -0.30029473129904827,
            ),
            (
                1.4132667908068457,
                0.03110221354462144,
                -0.04297320205816883,
            ),
            (
                -1.7089423051639878,
                -0.29089177871842153,
                -0.42087323476286687,
            ),
        )
        coordinates = (-806385650.80464, 345678239.24749994)
        circle = Circle3DSpec.from_plane_coordinates(
            "round-trip-circle",
            frame,
            coordinates,
            1.0e4,
            domain=ParameterInterval(-pi, pi),
        )
        ellipse = Ellipse3DSpec.from_plane_coordinates(
            "round-trip-ellipse",
            frame,
            coordinates,
            2.0e4,
            1.0e4,
            domain=ParameterInterval(-pi / 2.0, pi / 2.0),
        )
        scene = PlanarCurveScene3D((frame,), (circle, ellipse))

        rebuilt_dict = PlanarCurveScene3D.from_dict(scene.to_dict())
        rebuilt_json = PlanarCurveScene3D.from_json(scene.canonical_json())
        self.assertEqual(rebuilt_dict, scene)
        self.assertEqual(rebuilt_json, scene)
        self.assertEqual(rebuilt_dict.canonical_json(), scene.canonical_json())
        self.assertEqual(rebuilt_json.canonical_json(), scene.canonical_json())
        rebuilt_frame = rebuilt_dict.frames[0]
        self.assertEqual(
            Circle3DSpec.from_dict(circle.to_dict(), rebuilt_frame).to_dict(),
            circle.to_dict(),
        )
        self.assertEqual(
            Ellipse3DSpec.from_dict(ellipse.to_dict(), rebuilt_frame).to_dict(),
            ellipse.to_dict(),
        )

        invalid_circle = circle.to_dict()
        invalid_circle["center"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "does not lie|disagrees with center_coordinates",
        ):
            Circle3DSpec.from_dict(invalid_circle, frame)

    def test_scene_rejects_duplicate_or_colliding_stable_ids(self) -> None:
        duplicate_frame = PlanarFrame3D(
            "frame-a",
            (0, 0, 1),
            (0, 0, 1),
            (1, 0, 0),
        )
        with self.assertRaisesRegex(PlanarCurve3DContractError, "frame identities"):
            PlanarCurveScene3D(
                frames=(self.frame_a, duplicate_frame),
                curves=(),
            )

        circle = Circle3DSpec("curve", self.frame_a, (0, 0, 0), 1)
        ellipse = Ellipse3DSpec("curve", self.frame_a, (0, 0, 0), 2, 1)
        with self.assertRaisesRegex(PlanarCurve3DContractError, "curve identities"):
            PlanarCurveScene3D(frames=(self.frame_a,), curves=(circle, ellipse))

        colliding = Circle3DSpec("frame-a", self.frame_a, (0, 0, 0), 1)
        with self.assertRaisesRegex(PlanarCurve3DContractError, "globally distinct"):
            PlanarCurveScene3D(frames=(self.frame_a,), curves=(colliding,))

    def test_scene_rejects_unregistered_or_inconsistent_frame_references(self) -> None:
        unregistered = Circle3DSpec("circle", self.frame_z, (2, 0, 0), 1)
        with self.assertRaisesRegex(PlanarCurve3DContractError, "unregistered"):
            PlanarCurveScene3D(frames=(self.frame_a,), curves=(unregistered,))

        conflicting_frame = PlanarFrame3D(
            "frame-a",
            (0, 0, 1),
            (0, 0, 1),
            (1, 0, 0),
        )
        inconsistent = Circle3DSpec(
            "circle",
            conflicting_frame,
            (0, 0, 1),
            1,
        )
        with self.assertRaisesRegex(PlanarCurve3DContractError, "disagrees"):
            PlanarCurveScene3D(frames=(self.frame_a,), curves=(inconsistent,))

    def test_scene_lowers_curves_in_stable_identity_order(self) -> None:
        circle = Circle3DSpec("curve-z", self.frame_z, (2, 0, 0), 1)
        ellipse = Ellipse3DSpec("curve-a", self.frame_a, (0, 0, 0), 2, 1)
        scene = PlanarCurveScene3D(
            frames=(self.frame_z, self.frame_a),
            curves=(circle, ellipse),
        )

        lowered = scene.lower_to_analytic_curves()

        self.assertEqual(
            tuple(item.curve_id for item in lowered),
            ("curve-a", "curve-z"),
        )
        self.assertIsInstance(lowered[0], EllipseArcCurve)
        self.assertIsInstance(lowered[1], CircleArcCurve)

    def test_canonical_json_normalizes_signed_zero(self) -> None:
        frame = PlanarFrame3D(
            "zero-frame",
            (-0.0, 0.0, -0.0),
            (-0.0, 0.0, 1.0),
            (1.0, -0.0, 0.0),
        )
        circle = Circle3DSpec(
            "zero-circle",
            frame,
            (-0.0, 0.0, -0.0),
            1.0,
            domain=ParameterInterval(-0.0, tau),
        )

        payload = PlanarCurveScene3D((frame,), (circle,)).canonical_json()

        self.assertNotIn("-0.0", payload)
        self.assertEqual(json.loads(payload)["frames"][0]["point"], [0.0] * 3)


if __name__ == "__main__":
    unittest.main()
