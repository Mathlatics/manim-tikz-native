from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from math import atan, pi

import numpy as np

from polyhedron_visibility.geometry import GeometryContext, GeometryQuantity
from polyhedron_visibility.quadrics.algebra import (
    AffineFrame3D,
    CoincidentRayError,
    HomogeneousQuadric,
    QuadricAlgebraError,
)
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    PlanarCapSpec,
    PlaneDisplayPatchSpec,
    QuadricContractError,
    SectionPlane,
    SphereSpec,
)


ROOT = Path(__file__).resolve().parents[1]


class RendererNeutralImportTests(unittest.TestCase):
    def test_quadric_contract_and_algebra_do_not_import_manim(self) -> None:
        script = """
import sys
import polyhedron_visibility.quadrics.algebra
import polyhedron_visibility.quadrics.contract
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


class AffineFrameTests(unittest.TestCase):
    def test_default_frame_is_stable_right_handed_and_round_trips(self) -> None:
        first = AffineFrame3D.from_axis((1, 2, 3), (0, 0, 5))
        second = AffineFrame3D.from_axis((1, 2, 3), (0, 0, 5))
        self.assertEqual(first, second)
        self.assertEqual(first.x_axis, (1.0, 0.0, 0.0))
        self.assertEqual(first.y_axis, (0.0, 1.0, 0.0))
        self.assertEqual(first.z_axis, (0.0, 0.0, 1.0))
        local = np.asarray((2.5, -1.25, 4.0))
        world = first.to_world_point(local)
        np.testing.assert_allclose(first.to_local_point(world), local, atol=1.0e-12)
        np.testing.assert_allclose(
            first.world_to_local_matrix @ first.local_to_world_matrix,
            np.eye(4),
            atol=1.0e-12,
        )

    def test_authored_radial_axis_is_projected_and_preserves_sign(self) -> None:
        frame = AffineFrame3D.from_axis(
            (0, 0, 0),
            (1, 1, 1),
            radial_axis=(1, -1, 0),
        )
        np.testing.assert_allclose(
            np.dot(frame.x_axis, frame.z_axis), 0.0, atol=1.0e-12
        )
        self.assertGreater(float(np.dot(frame.x_axis, (1, -1, 0))), 0.0)
        np.testing.assert_allclose(
            np.cross(frame.x_axis, frame.y_axis), frame.z_axis, atol=1.0e-12
        )

    def test_invalid_frames_fail_closed(self) -> None:
        with self.assertRaises(QuadricAlgebraError):
            AffineFrame3D.from_axis((0, 0, 0), (0, 0, 0))
        with self.assertRaises(QuadricAlgebraError):
            AffineFrame3D.from_axis(
                (0, 0, 0), (0, 0, 1), radial_axis=(0, 0, 5)
            )
        with self.assertRaises(QuadricAlgebraError):
            AffineFrame3D(
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 0),
                (0, 0, 1),
            )


class HomogeneousQuadricTests(unittest.TestCase):
    def test_evaluate_gradient_and_ray_coefficients_are_consistent(self) -> None:
        quadric = HomogeneousQuadric(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, -4),
            )
        )
        self.assertEqual(quadric.evaluate((2, 0, 0)), 0.0)
        np.testing.assert_allclose(quadric.gradient((2, 0, 0)), (4, 0, 0))
        origin = np.asarray((3.0, 1.0, 0.0))
        direction = np.asarray((-2.0, 0.5, 0.0))
        a, b, c = quadric.ray_coefficients(origin, direction)
        for parameter in (-2.0, -0.25, 0.0, 0.75, 3.0):
            expected = quadric.evaluate(origin + parameter * direction)
            self.assertAlmostEqual(
                a * parameter * parameter + b * parameter + c,
                expected,
                places=11,
            )

    def test_affine_transform_preserves_the_zero_set(self) -> None:
        local = HomogeneousQuadric(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, -1),
            )
        )
        transform = np.asarray(
            (
                (2.0, 0.0, 0.0, 4.0),
                (0.0, 3.0, 0.0, -2.0),
                (0.0, 0.0, 4.0, 1.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        world = local.affine_transform(transform)
        for point in ((1, 0, 0), (0, -1, 0), (0, 0, 1)):
            world_point = (transform @ np.append(point, 1.0))[:3]
            self.assertAlmostEqual(world.evaluate(world_point), 0.0, places=12)
        self.assertLess(world.evaluate((4, -2, 1)), 0.0)

    def test_real_ray_roots_handle_secant_tangent_miss_and_coincidence(self) -> None:
        sphere = HomogeneousQuadric(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, -1),
            )
        )
        np.testing.assert_allclose(
            sphere.real_ray_parameters((2, 0, 0), (-1, 0, 0)), (1, 3)
        )
        np.testing.assert_allclose(
            sphere.real_ray_parameters((2, 1, 0), (-1, 0, 0)), (2,)
        )
        self.assertEqual(
            sphere.real_ray_parameters((2, 2, 0), (-1, 0, 0)), ()
        )
        cylinder = HomogeneousQuadric(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, -1),
            )
        )
        with self.assertRaises(CoincidentRayError):
            cylinder.real_ray_parameters((1, 0, 0), (0, 0, 1))

    def test_plane_restriction_returns_the_expected_conic(self) -> None:
        sphere = SphereSpec("sphere", (0, 0, 0), 2)
        plane = SectionPlane("equator", (0, 0, 0), (0, 0, 1))
        np.testing.assert_allclose(
            plane.restrict(sphere.support_quadric),
            np.diag((1.0, 1.0, -4.0)),
            atol=1.0e-12,
        )

    def test_invalid_matrices_and_affine_transforms_are_rejected(self) -> None:
        with self.assertRaises(QuadricAlgebraError):
            HomogeneousQuadric(tuple(tuple(0.0 for _ in range(4)) for _ in range(4)))
        with self.assertRaises(QuadricAlgebraError):
            HomogeneousQuadric(
                (
                    (1, 1, 0, 0),
                    (0, 1, 0, 0),
                    (0, 0, 1, 0),
                    (0, 0, 0, -1),
                )
            )
        sphere = SphereSpec("sphere", (0, 0, 0), 1).support_quadric
        with self.assertRaises(QuadricAlgebraError):
            sphere.affine_transform(np.zeros((4, 4)))
        with self.assertRaises(QuadricAlgebraError):
            sphere.affine_transform(
                (
                    (0, 0, 0, 0),
                    (0, 1, 0, 0),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1),
                )
            )


class StrictContractTests(unittest.TestCase):
    def test_invalid_identities_and_non_finite_values_are_rejected(self) -> None:
        with self.assertRaises(QuadricContractError):
            SphereSpec("", (0, 0, 0), 1)
        with self.assertRaises(QuadricContractError):
            SphereSpec("sphere", (float("nan"), 0, 0), 1)
        with self.assertRaises(QuadricContractError):
            SphereSpec("sphere", (0, 0, 0), float("inf"))
        with self.assertRaises(QuadricContractError):
            CylinderSpec("cylinder", (0, 0, 0), (0, 0, 0), 1, (-1, 1))
        with self.assertRaises(QuadricContractError):
            CylinderSpec("cylinder", (0, 0, 0), (0, 0, 1), 0, (-1, 1))
        with self.assertRaises(QuadricContractError):
            CylinderSpec("cylinder", (0, 0, 0), (0, 0, 1), 1, (1, 1))
        with self.assertRaises(QuadricContractError):
            CylinderSpec(
                "cylinder", (0, 0, 0), (0, 0, 1), 1, (-1, float("inf"))
            )
        with self.assertRaises(QuadricContractError):
            ConeSpec("cone", (0, 0, 0), (0, 0, 1), 0, (0, 1))
        with self.assertRaises(QuadricContractError):
            ConeSpec("cone", (0, 0, 0), (0, 0, 1), pi / 2, (0, 1))
        with self.assertRaises(QuadricContractError):
            SectionPlane("plane", (0, 0, 0), (0, 0, 0))
        with self.assertRaises(QuadricContractError):
            PlaneDisplayPatchSpec("patch", "plane", -1, 1)

    def test_ids_are_canonicalized_without_changing_semantic_text(self) -> None:
        sphere = SphereSpec("  sphere  ", (0, 0, 0), 1)
        self.assertEqual(sphere.surface_id, "sphere")


class SphereContractTests(unittest.TestCase):
    def test_support_quadric_contains_and_ray_hits(self) -> None:
        sphere = SphereSpec("sphere", (1, 2, 3), 2)
        self.assertAlmostEqual(sphere.support_quadric.evaluate((3, 2, 3)), 0.0)
        self.assertLess(sphere.support_quadric.evaluate((1, 2, 3)), 0.0)
        self.assertTrue(sphere.contains((1, 2, 3)))
        self.assertFalse(sphere.contains((3.1, 2, 3)))
        hits = sphere.ray_hits((5, 2, 3), (-2, 0, 0))
        np.testing.assert_allclose([item.parameter for item in hits], (2, 6))
        np.testing.assert_allclose(hits[0].point, (3, 2, 3))
        np.testing.assert_allclose(hits[0].normal, (1, 0, 0))
        self.assertEqual(sphere.end_caps, ())

    def test_contains_uses_explicit_geometry_context_boundary_override(self) -> None:
        sphere = SphereSpec("sphere", (0, 0, 0), 1)
        context = GeometryContext(
            overrides={GeometryQuantity.BOUNDARY: 0.01}
        )
        self.assertFalse(sphere.contains((1.005, 0, 0)))
        self.assertTrue(sphere.contains((1.005, 0, 0), context=context))


class CylinderContractTests(unittest.TestCase):
    def test_infinite_support_is_separate_from_range_and_caps(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0, 0, 0),
            (0, 0, 4),
            1,
            (-1, 1),
        )
        self.assertEqual(cylinder.axis, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(cylinder.support_quadric.evaluate((1, 0, 20)), 0.0)
        self.assertFalse(cylinder.contains((1, 0, 20)))
        self.assertTrue(cylinder.contains((0.5, 0, 0.5)))
        self.assertFalse(cylinder.contains((1.5, 0, 0.5)))

        np.testing.assert_allclose(
            cylinder.support_ray_parameters((2, 0, 2), (-1, 0, 0)),
            (1, 3),
        )
        self.assertEqual(
            cylinder.lateral_ray_hits((2, 0, 2), (-1, 0, 0)), ()
        )
        self.assertEqual(len(cylinder.end_caps), 2)
        # A cap center lies inside the infinite cylinder, not on its support.
        self.assertLess(
            cylinder.support_quadric.evaluate(cylinder.end_caps[0].center), 0.0
        )

    def test_entity_ray_hits_caps_independently(self) -> None:
        cylinder = CylinderSpec(
            "cylinder", (0, 0, 0), (0, 0, 1), 1, (-1, 1)
        )
        hits = cylinder.ray_hits((0, 0, 2), (0, 0, -7))
        self.assertEqual([item.role for item in hits], ["cap_max", "cap_min"])
        np.testing.assert_allclose([item.parameter for item in hits], (1, 3))
        self.assertEqual(
            cylinder.ray_hits((0, 0, 2), (0, 0, -1), include_caps=False),
            (),
        )

    def test_generator_ray_remains_finite_through_cap_contracts(self) -> None:
        cylinder = CylinderSpec(
            "cylinder", (0, 0, 0), (0, 0, 1), 1, (-1, 1)
        )
        hits = cylinder.ray_hits((1, 0, 2), (0, 0, -1))
        self.assertEqual([item.role for item in hits], ["cap_max", "cap_min"])
        np.testing.assert_allclose([item.parameter for item in hits], (1, 3))


class ConeContractTests(unittest.TestCase):
    def test_support_is_a_double_cone_while_entity_can_use_one_nappe(self) -> None:
        cone = ConeSpec(
            "cone", (0, 0, 0), (0, 0, 1), pi / 4, (0, 2)
        )
        self.assertAlmostEqual(cone.support_quadric.evaluate((1, 0, 1)), 0.0)
        self.assertAlmostEqual(cone.support_quadric.evaluate((-1, 0, -1)), 0.0)
        self.assertTrue(cone.contains((0.5, 0, 1)))
        self.assertFalse(cone.contains((1.1, 0, 1)))
        self.assertFalse(cone.contains((0.5, 0, -1)))
        self.assertEqual(len(cone.end_caps), 1)
        self.assertEqual(cone.end_caps[0].role, "cap_max")
        self.assertAlmostEqual(cone.end_caps[0].radius, 2.0)

    def test_lateral_and_cap_ray_hits_are_both_reported(self) -> None:
        cone = ConeSpec(
            "cone", (0, 0, 0), (0, 0, 1), pi / 4, (0, 2)
        )
        side_hits = cone.lateral_ray_hits((2, 0, 1), (-1, 0, 0))
        np.testing.assert_allclose([item.parameter for item in side_hits], (1, 3))
        axial_hits = cone.ray_hits((0, 0, 3), (0, 0, -1))
        self.assertEqual([item.role for item in axial_hits], ["cap_max", "support"])
        np.testing.assert_allclose([item.parameter for item in axial_hits], (1, 3))
        self.assertTrue(axial_hits[-1].tangential)
        self.assertIsNone(axial_hits[-1].normal)

    def test_axial_range_may_retain_both_nappes_and_two_caps(self) -> None:
        cone = ConeSpec(
            "double", (0, 0, 0), (0, 0, 1), pi / 6, (-2, 2)
        )
        self.assertEqual(len(cone.end_caps), 2)
        self.assertTrue(cone.contains((0, 0, -1)))

    def test_finite_frustum_scale_excludes_a_remote_support_apex(self) -> None:
        apex_z = -1.0e6
        slope = 1.0e-6
        frustum = ConeSpec(
            "near-cylinder-frustum",
            (0.0, 0.0, apex_z),
            (0.0, 0.0, 1.0),
            atan(slope),
            (-3.0 - apex_z, 5.8 - apex_z),
            radial_axis=(1.0, 0.0, 0.0),
        )

        points = np.asarray(frustum.characteristic_points, dtype=float)
        self.assertEqual(points.shape, (8, 3))
        self.assertLess(float(np.max(np.abs(points))), 6.0)
        self.assertFalse(
            np.any(np.all(points == np.asarray(frustum.apex), axis=1))
        )

        apex_cone = ConeSpec(
            "apex-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 2.0),
        )
        self.assertEqual(apex_cone.characteristic_points[0], apex_cone.apex)


class PlaneContractTests(unittest.TestCase):
    def test_infinite_plane_and_display_patch_are_separate(self) -> None:
        plane = SectionPlane("cut", (1, 2, 3), (0, 0, 5))
        self.assertEqual(plane.u_axis, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(plane.signed_distance((10, -8, 5)), 2.0)
        point = plane.point_from_coordinates((2, -1))
        np.testing.assert_allclose(point, (3, 1, 3))
        np.testing.assert_allclose(plane.coordinates_in_plane(point), (2, -1))

        patch = PlaneDisplayPatchSpec("patch", "cut", 2, 1, (0.5, -0.5))
        self.assertEqual(
            patch.corners(plane),
            (
                (-0.5, 0.5, 3.0),
                (3.5, 0.5, 3.0),
                (3.5, 2.5, 3.0),
                (-0.5, 2.5, 3.0),
            ),
        )
        self.assertTrue(patch.contains_coordinates((2.5, -0.5)))
        self.assertFalse(patch.contains_coordinates((2.6, -0.5)))

    def test_patch_rejects_a_different_plane_identity(self) -> None:
        patch = PlaneDisplayPatchSpec("patch", "cut", 2, 1)
        with self.assertRaises(QuadricContractError):
            patch.corners(SectionPlane("other", (0, 0, 0), (0, 0, 1)))

    def test_planar_cap_disk_membership_and_ray_hit(self) -> None:
        cap = PlanarCapSpec(
            "cap",
            "cylinder",
            (0, 0, 1),
            (0, 0, 1),
            2,
            role="cap_max",
        )
        self.assertTrue(cap.contains_point((1.5, 0, 1)))
        self.assertFalse(cap.contains_point((2.5, 0, 1)))
        hit = cap.ray_hits((0, 0, 3), (0, 0, -2))[0]
        self.assertAlmostEqual(hit.parameter, 2.0)
        np.testing.assert_allclose(hit.point, (0, 0, 1))
        self.assertEqual(cap.ray_hits((0, 0, 3), (1, 0, 0)), ())


if __name__ == "__main__":
    unittest.main()
