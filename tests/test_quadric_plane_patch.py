from __future__ import annotations

import json
from math import pi, sqrt
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.plane_patch import (
    DEFAULT_PLANE_PATCH_MARGIN_RATIO,
    PLANE_MOTION_PATCH_ENVELOPE_SCHEMA,
    PLANE_PATCH_FIT_SCHEMA,
    PlanePatchFitError,
    canonical_fitted_plane_display_patch_json,
    canonical_plane_motion_patch_envelope_json,
    finite_surface_support_interval,
    fit_plane_display_patch,
    fit_plane_motion_display_patch_envelope,
)
from polyhedron_visibility.quadrics.plane_motion import AxisAnglePlaneMotion


ROOT = Path(__file__).resolve().parents[1]


def _tuple3(value: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _rotation() -> np.ndarray:
    axis = np.asarray((1.0, -2.0, 3.0), dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


class RendererNeutralPlanePatchTests(unittest.TestCase):
    def test_import_does_not_import_manim(self) -> None:
        script = """
import sys
import polyhedron_visibility.quadrics.plane_patch
assert 'manim' not in sys.modules
assert not any(name.startswith('manim.') for name in sys.modules)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


class AnalyticSupportIntervalTests(unittest.TestCase):
    def test_sphere_interval_is_exact_for_non_unit_direction(self) -> None:
        sphere = SphereSpec("sphere", (1, -2, 3), 2)
        direction = np.asarray((2.0, -1.0, 2.0))
        center_value = float(np.dot(direction, sphere.center))
        radius_value = sphere.radius * float(np.linalg.norm(direction))
        self.assertEqual(
            finite_surface_support_interval(sphere, direction),
            (center_value - radius_value, center_value + radius_value),
        )

    def test_rotated_cylinder_interval_uses_axial_and_radial_support(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (1, 2, 3),
            (1, 0, 0),
            2,
            (-1, 3),
            radial_axis=(0, 1, 0),
        )
        direction = np.asarray((1, 0, 1), dtype=float) / sqrt(2.0)
        base = float(np.dot(direction, cylinder.origin))
        axial = float(np.dot(direction, cylinder.axis))
        radial = sqrt(max(0.0, 1.0 - axial * axial))
        expected = (
            base - axial - cylinder.radius * radial,
            base + 3.0 * axial + cylinder.radius * radial,
        )
        np.testing.assert_allclose(
            finite_surface_support_interval(cylinder, direction),
            expected,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_cone_interval_checks_both_finite_endpoint_disks(self) -> None:
        cone = ConeSpec(
            "cone",
            (1, -2, 0),
            (0, 0, 1),
            pi / 4,
            (1, 3),
            radial_axis=(1, 0, 0),
        )
        np.testing.assert_allclose(
            finite_surface_support_interval(cone, (1, 0, 0)),
            (-2, 4),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            finite_surface_support_interval(cone, (0, 0, 1)),
            (1, 3),
            rtol=0.0,
            atol=1.0e-12,
        )


class PlanePatchFitTests(unittest.TestCase):
    def test_plane_and_display_patch_remain_distinct(self) -> None:
        plane = SectionPlane("plane", (10, -4, 2), (0, 0, 1), (1, 0, 0))
        sphere = SphereSpec("sphere", (12, -1, 100), 2)
        fitted = fit_plane_display_patch(
            "display", plane, (sphere,), margin_ratio=0.25
        )

        # Moving the infinite mathematical plane along its normal does not
        # alter the orthogonal u/v bounds or mutate the plane contract.
        self.assertIs(fitted.plane, plane)
        self.assertEqual(fitted.patch.plane_id, plane.plane_id)
        self.assertEqual(fitted.unpadded_bounds, ((0.0, 4.0), (1.0, 5.0)))
        self.assertEqual(fitted.patch.center_coordinates, (2.0, 3.0))
        self.assertEqual(fitted.patch.half_width, 2.5)
        self.assertEqual(fitted.patch.half_height, 2.5)
        self.assertFalse(fitted.visibility_authoritative)

    def test_default_margin_and_multiple_entities_fit_one_union(self) -> None:
        plane = SectionPlane("plane", (0, 0, 0), (0, 0, 1), (1, 0, 0))
        sphere = SphereSpec("z-sphere", (-5, 0, 0), 1)
        cylinder = CylinderSpec(
            "a-cylinder",
            (4, 2, -2),
            (0, 0, 1),
            2,
            (-1, 3),
            radial_axis=(1, 0, 0),
        )
        fitted = fit_plane_display_patch("display", plane, (sphere, cylinder))

        self.assertEqual(
            tuple(item.surface_id for item in fitted.surface_extents),
            ("a-cylinder", "z-sphere"),
        )
        self.assertEqual(fitted.unpadded_bounds, ((-6.0, 6.0), (-1.0, 4.0)))
        self.assertEqual(
            fitted.patch.center_coordinates,
            (0.0, 1.5),
        )
        self.assertAlmostEqual(
            fitted.patch.half_width,
            6.0 * (1.0 + DEFAULT_PLANE_PATCH_MARGIN_RATIO),
        )
        self.assertAlmostEqual(
            fitted.patch.half_height,
            2.5 * (1.0 + DEFAULT_PLANE_PATCH_MARGIN_RATIO),
        )
        self.assertEqual(fitted.support_evaluation_count, 8)

    def test_frustum_and_full_single_nappe_cones_are_supported(self) -> None:
        plane = SectionPlane("plane", (0, 0, 0), (0, 0, 1), (1, 0, 0))
        for surface in (
            ConeSpec("frustum", (1, -2, 0), (0, 0, 1), pi / 4, (1, 3)),
            ConeSpec("full-positive", (1, -2, 0), (0, 0, 1), pi / 4, (0, 3)),
            ConeSpec("full-negative", (1, -2, 0), (0, 0, 1), pi / 4, (-3, 0)),
        ):
            with self.subTest(surface=surface.surface_id):
                fitted = fit_plane_display_patch(
                    "display", plane, (surface,), margin_ratio=0.0
                )
                np.testing.assert_allclose(
                    fitted.unpadded_bounds,
                    ((-2.0, 4.0), (-5.0, 1.0)),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                self.assertEqual(fitted.patch.center_coordinates, (1.0, -2.0))
                self.assertAlmostEqual(fitted.patch.half_width, 3.0)
                self.assertAlmostEqual(fitted.patch.half_height, 3.0)

    def test_rigid_rotation_preserves_authored_plane_coordinates(self) -> None:
        rotation = _rotation()
        plane = SectionPlane("plane", (1, -2, 0.5), (0, 0, 1), (1, 0, 0))
        surfaces = (
            SphereSpec("sphere", (-2, 1, 0), 1.5),
            CylinderSpec(
                "cylinder",
                (2, -1, 0),
                (1, 1, 2),
                0.75,
                (-2, 3),
                radial_axis=(1, -1, 0),
            ),
            ConeSpec(
                "cone",
                (0.5, 1, -2),
                (-1, 2, 1),
                0.4,
                (0, 4),
                radial_axis=(2, 1, 0),
            ),
        )
        rotated_plane = SectionPlane(
            "plane",
            _tuple3(rotation @ np.asarray(plane.point)),
            _tuple3(rotation @ np.asarray(plane.normal)),
            _tuple3(rotation @ np.asarray(plane.u_axis)),
        )
        rotated_surfaces = (
            SphereSpec(
                "sphere",
                _tuple3(rotation @ np.asarray(surfaces[0].center)),
                surfaces[0].radius,
            ),
            CylinderSpec(
                "cylinder",
                _tuple3(rotation @ np.asarray(surfaces[1].origin)),
                _tuple3(rotation @ np.asarray(surfaces[1].axis)),
                surfaces[1].radius,
                surfaces[1].axial_range,
                radial_axis=_tuple3(rotation @ np.asarray(surfaces[1].radial_axis)),
            ),
            ConeSpec(
                "cone",
                _tuple3(rotation @ np.asarray(surfaces[2].apex)),
                _tuple3(rotation @ np.asarray(surfaces[2].axis)),
                surfaces[2].half_angle,
                surfaces[2].axial_range,
                radial_axis=_tuple3(rotation @ np.asarray(surfaces[2].radial_axis)),
            ),
        )
        original = fit_plane_display_patch(
            "display", plane, surfaces, margin_ratio=0.17
        )
        rotated = fit_plane_display_patch(
            "display", rotated_plane, rotated_surfaces, margin_ratio=0.17
        )
        np.testing.assert_allclose(
            rotated.unpadded_bounds,
            original.unpadded_bounds,
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            rotated.patch.center_coordinates,
            original.patch.center_coordinates,
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            (rotated.patch.half_width, rotated.patch.half_height),
            (original.patch.half_width, original.patch.half_height),
            rtol=0.0,
            atol=2.0e-12,
        )

    def test_scale_covariance_from_micro_to_large_scenes(self) -> None:
        def build(scale: float):
            plane = SectionPlane(
                "plane",
                (scale, -2 * scale, 0.5 * scale),
                (0, 0, 1),
                (1, 0, 0),
            )
            surfaces = (
                SphereSpec("sphere", (-2 * scale, scale, 0), 1.5 * scale),
                CylinderSpec(
                    "cylinder",
                    (2 * scale, -scale, 0),
                    (1, 1, 2),
                    0.75 * scale,
                    (-2 * scale, 3 * scale),
                    radial_axis=(1, -1, 0),
                ),
                ConeSpec(
                    "cone",
                    (0.5 * scale, scale, -2 * scale),
                    (-1, 2, 1),
                    0.4,
                    (0, 4 * scale),
                    radial_axis=(2, 1, 0),
                ),
            )
            return fit_plane_display_patch(
                "display", plane, surfaces, margin_ratio=0.2
            )

        unit = build(1.0)
        for scale in (1.0e-6, 1.0e6):
            with self.subTest(scale=scale):
                fitted = build(scale)
                np.testing.assert_allclose(
                    np.asarray(fitted.unpadded_bounds),
                    scale * np.asarray(unit.unpadded_bounds),
                    rtol=2.0e-12,
                    atol=1.0e-15 * max(1.0, scale),
                )
                np.testing.assert_allclose(
                    fitted.patch.center_coordinates,
                    scale * np.asarray(unit.patch.center_coordinates),
                    rtol=2.0e-12,
                    atol=1.0e-15 * max(1.0, scale),
                )
                np.testing.assert_allclose(
                    (fitted.patch.half_width, fitted.patch.half_height),
                    scale
                    * np.asarray(
                        (unit.patch.half_width, unit.patch.half_height)
                    ),
                    rtol=2.0e-12,
                    atol=1.0e-15 * max(1.0, scale),
                )

    def test_canonical_serialization_is_input_order_independent(self) -> None:
        plane = SectionPlane("plane", (0, 0, 0), (0, 0, 1), (1, 0, 0))
        surfaces = (
            SphereSpec("sphere", (-2, 0, 0), 1),
            CylinderSpec("cylinder", (2, 0, 0), (0, 0, 1), 1, (-1, 1)),
            ConeSpec("cone", (0, 2, 0), (0, 0, 1), 0.4, (0, 2)),
        )
        first = fit_plane_display_patch("display", plane, surfaces)
        second = fit_plane_display_patch("display", plane, tuple(reversed(surfaces)))
        first_json = canonical_fitted_plane_display_patch_json(first)
        self.assertEqual(first_json, canonical_fitted_plane_display_patch_json(second))
        parsed = json.loads(first_json)
        self.assertEqual(parsed["schema"], PLANE_PATCH_FIT_SCHEMA)
        self.assertEqual(parsed["supportEvaluationCount"], 12)
        self.assertFalse(parsed["visibilityAuthoritative"])
        self.assertEqual(
            [item["surfaceId"] for item in parsed["surfaceExtents"]],
            ["cone", "cylinder", "sphere"],
        )
        self.assertEqual(first_json, canonical_fitted_plane_display_patch_json(first))

    def test_invalid_or_uncertifiable_inputs_fail_closed(self) -> None:
        plane = SectionPlane("plane", (0, 0, 0), (0, 0, 1), (1, 0, 0))
        sphere = SphereSpec("sphere", (0, 0, 0), 1)
        with self.assertRaises(PlanePatchFitError):
            fit_plane_display_patch("display", plane, ())
        with self.assertRaises(PlanePatchFitError):
            fit_plane_display_patch("display", plane, (sphere, sphere))
        with self.assertRaises(TypeError):
            fit_plane_display_patch("display", plane, (object(),))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            fit_plane_display_patch("display", object(), (sphere,))  # type: ignore[arg-type]
        for margin in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(margin=margin):
                with self.assertRaises(PlanePatchFitError):
                    fit_plane_display_patch(
                        "display", plane, (sphere,), margin_ratio=margin
                    )

        double_nappe = ConeSpec(
            "double", (0, 0, 0), (0, 0, 1), 0.4, (-1, 1)
        )
        with self.assertRaisesRegex(PlanePatchFitError, "two nappes"):
            fit_plane_display_patch("display", plane, (double_nappe,))

        # Finite authored numbers can still overflow their support operation;
        # returning an undersized rectangle would be unsafe, so this is an
        # explicit failure rather than a sampled fallback.
        enormous = SphereSpec("enormous", (1.0e308, 0, 0), 1.0e308)
        with self.assertRaises(PlanePatchFitError):
            fit_plane_display_patch("display", plane, (enormous,))


class PlaneMotionPatchEnvelopeTests(unittest.TestCase):
    def test_one_fixed_patch_contains_every_dynamic_fit_in_the_motion(self) -> None:
        plane = SectionPlane("plane", (1.0, -0.5, 0.25), (0, 0, 1), (1, 0, 0))
        motion = AxisAnglePlaneMotion(
            "motion",
            plane,
            axis_point=(0.25, -0.25, 0.5),
            axis_direction=(1.0, 2.0, -1.0),
            start_angle=-0.8,
            end_angle=1.1,
        )
        surfaces = (
            SphereSpec("sphere", (-1.0, 0.5, 0.0), 0.75),
            CylinderSpec(
                "cylinder",
                (1.5, -0.5, -1.0),
                (0.0, 0.0, 1.0),
                0.4,
                (-1.0, 2.0),
            ),
            ConeSpec(
                "cone",
                (0.0, 1.0, -0.5),
                (0.0, 0.0, 1.0),
                0.35,
                (0.0, 2.5),
            ),
        )
        margin = 0.13
        envelope = fit_plane_motion_display_patch_envelope(
            "motion-patch",
            motion,
            surfaces,
            margin_ratio=margin,
        )
        envelope_u = (
            envelope.patch.center_coordinates[0] - envelope.patch.half_width,
            envelope.patch.center_coordinates[0] + envelope.patch.half_width,
        )
        envelope_v = (
            envelope.patch.center_coordinates[1] - envelope.patch.half_height,
            envelope.patch.center_coordinates[1] + envelope.patch.half_height,
        )

        for progress in np.linspace(0.0, 1.0, 101):
            fitted = fit_plane_display_patch(
                "dynamic",
                motion.plane_at(float(progress)),
                surfaces,
                margin_ratio=margin,
            ).patch
            dynamic_u = (
                fitted.center_coordinates[0] - fitted.half_width,
                fitted.center_coordinates[0] + fitted.half_width,
            )
            dynamic_v = (
                fitted.center_coordinates[1] - fitted.half_height,
                fitted.center_coordinates[1] + fitted.half_height,
            )
            self.assertLessEqual(envelope_u[0], dynamic_u[0] + 1.0e-12)
            self.assertGreaterEqual(envelope_u[1], dynamic_u[1] - 1.0e-12)
            self.assertLessEqual(envelope_v[0], dynamic_v[0] + 1.0e-12)
            self.assertGreaterEqual(envelope_v[1], dynamic_v[1] - 1.0e-12)

    def test_motion_envelope_is_deterministic_and_not_visibility_truth(self) -> None:
        plane = SectionPlane("plane", (0, 0, 0), (0, 0, 1), (1, 0, 0))
        motion = AxisAnglePlaneMotion(
            "motion",
            plane,
            axis_point=(0, 0, 0),
            axis_direction=(0, 1, 0),
            start_angle=0.0,
            end_angle=pi / 2,
        )
        surfaces = (
            SphereSpec("z", (2, 0, 0), 1),
            SphereSpec("a", (-1, 0, 0), 0.5),
        )
        first = fit_plane_motion_display_patch_envelope(
            "motion-patch", motion, surfaces, margin_ratio=0.1
        )
        second = fit_plane_motion_display_patch_envelope(
            "motion-patch", motion, tuple(reversed(surfaces)), margin_ratio=0.1
        )
        first_json = canonical_plane_motion_patch_envelope_json(first)
        self.assertEqual(
            first_json,
            canonical_plane_motion_patch_envelope_json(second),
        )
        parsed = json.loads(first_json)
        self.assertEqual(parsed["schema"], PLANE_MOTION_PATCH_ENVELOPE_SCHEMA)
        self.assertEqual(parsed["boundingRadius"], 3.0)
        self.assertFalse(parsed["visibilityAuthoritative"])
        self.assertEqual(
            [item["surfaceId"] for item in parsed["surfaceRadii"]],
            ["a", "z"],
        )


if __name__ == "__main__":
    unittest.main()
