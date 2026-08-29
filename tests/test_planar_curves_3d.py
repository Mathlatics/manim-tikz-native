from __future__ import annotations

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
            },
        )

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
            2.0,
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
                2.0,
            )

        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "does not lie",
        ):
            Circle3DSpec(
                "ulp-sized-off-plane",
                frame,
                (1.0e12 + 0.001, 1.0e12, -1.0e12),
                2.0,
            )

    def test_large_oblique_frame_accepts_its_own_generated_points(self) -> None:
        frame = PlanarFrame3D(
            "large-oblique-plane",
            (1.0e12, -1.0e12, 1.0e12),
            (1.0, 1.0, 1.0),
            (1.0, -1.0, 0.0),
        )
        generated_center = frame.point_from_coordinates((1.0, 2.0))

        circle = Circle3DSpec(
            "round-trip-circle",
            frame,
            tuple(generated_center),
            2.0,
        )

        np.testing.assert_allclose(
            frame.coordinates_in_plane(circle.center, feature_scale=2.0),
            (1.0, 2.0),
            atol=2.0e-4,
        )

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
                generated = frame.point_from_coordinates(uv)

                circle = Circle3DSpec(
                    f"large-parameter-circle-{index}",
                    frame,
                    tuple(generated),
                    1.0,
                )
                self.assertEqual(circle.radius, 1.0)

                displaced = generated + off_plane * np.asarray(frame.normal)
                with self.assertRaisesRegex(
                    PlanarCurve3DContractError,
                    "does not lie",
                ):
                    Circle3DSpec(
                        f"off-plane-circle-{index}",
                        frame,
                        tuple(displaced),
                        1.0,
                    )

    def test_feature_relative_membership_has_no_absolute_epsilon_floor(
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

        canonicalized = Circle3DSpec(
            "within-relative-tolerance",
            frame,
            (0.0, 0.0, 0.5e-10),
            1.0,
        )
        self.assertEqual(canonicalized.center, (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(PlanarCurve3DContractError, "does not lie"):
            Circle3DSpec(
                "outside-relative-tolerance",
                frame,
                (0.0, 0.0, 2.0e-10),
                1.0,
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
                "semiU": 3.0,
                "semiV": 1.5,
                "domain": [0.0, tau],
            },
        )

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
