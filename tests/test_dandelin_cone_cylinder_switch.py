from __future__ import annotations

from math import cos, pi, sin
import unittest

import numpy as np
from manim import Scene, tempconfig

from examples.dandelin_cone_cylinder_switch.dandelin_cone_cylinder_switch import (
    AXIAL_RANGE,
    BACKGROUND_COLOR,
    CONE_SLOPE,
    DandelinConeCylinderSwitch,
    PLANE_NORMAL,
    PLANE_OFFSET,
    SURFACE_RADIUS,
    build_switch_diagram,
    compute_switch_frame,
    section_point,
)


class DandelinConeCylinderSwitchGeometryTests(unittest.TestCase):
    def test_endpoints_are_one_true_cone_and_one_true_cylinder(self) -> None:
        cone = compute_switch_frame(0.0)
        cylinder = compute_switch_frame(1.0)

        self.assertEqual(DandelinConeCylinderSwitch.__name__, "DandelinConeCylinderSwitch")
        self.assertEqual(cone.surface_kind, "cone")
        self.assertAlmostEqual(cone.slope, CONE_SLOPE)
        self.assertAlmostEqual(cone.apex_z, AXIAL_RANGE[0])
        self.assertAlmostEqual(cone.radius_at(AXIAL_RANGE[0]), 0.0)

        self.assertEqual(cylinder.surface_kind, "cylinder")
        self.assertEqual(cylinder.slope, 0.0)
        self.assertIsNone(cylinder.apex_z)
        for z in AXIAL_RANGE:
            self.assertAlmostEqual(cylinder.radius_at(z), SURFACE_RADIUS)
        for sphere in cylinder.spheres:
            self.assertAlmostEqual(sphere.radius, SURFACE_RADIUS)
            self.assertAlmostEqual(sphere.surface_contact_radius, SURFACE_RADIUS)
            self.assertAlmostEqual(sphere.surface_contact_z, sphere.center[2])

    def test_two_spheres_remain_tangent_to_surface_and_plane(self) -> None:
        normal = np.asarray(PLANE_NORMAL, dtype=float)
        for progress in np.linspace(0.0, 1.0, 9):
            with self.subTest(progress=float(progress)):
                frame = compute_switch_frame(float(progress))
                self.assertEqual(tuple(item.plane_side for item in frame.spheres), (-1, 1))
                for sphere in frame.spheres:
                    center = np.asarray(sphere.center, dtype=float)
                    plane_contact = np.asarray(sphere.plane_contact, dtype=float)
                    signed_distance = float(np.dot(normal, center) - PLANE_OFFSET)
                    self.assertAlmostEqual(
                        signed_distance,
                        sphere.plane_side * sphere.radius,
                        places=11,
                    )
                    self.assertAlmostEqual(
                        float(np.dot(normal, plane_contact)),
                        PLANE_OFFSET,
                        places=11,
                    )
                    self.assertAlmostEqual(
                        float(np.linalg.norm(plane_contact - center)),
                        sphere.radius,
                        places=11,
                    )

                    contact_point = np.asarray(
                        (
                            sphere.surface_contact_radius,
                            0.0,
                            sphere.surface_contact_z,
                        ),
                        dtype=float,
                    )
                    self.assertAlmostEqual(
                        sphere.surface_contact_radius,
                        frame.radius_at(sphere.surface_contact_z),
                        places=11,
                    )
                    self.assertAlmostEqual(
                        float(np.linalg.norm(contact_point - center)),
                        sphere.radius,
                        places=11,
                    )
                    self.assertGreaterEqual(sphere.center[2] - sphere.radius, AXIAL_RANGE[0])
                    self.assertLessEqual(sphere.center[2] + sphere.radius, AXIAL_RANGE[1])

    def test_section_parameterization_lies_on_both_constraints(self) -> None:
        normal = np.asarray(PLANE_NORMAL, dtype=float)
        for progress in (0.0, 0.2, 0.5, 0.8, 1.0):
            frame = compute_switch_frame(progress)
            for theta in np.linspace(0.0, 2.0 * pi, 25)[:-1]:
                point = np.asarray(section_point(frame, float(theta)), dtype=float)
                radial = float(np.hypot(point[0], point[1]))
                self.assertAlmostEqual(radial, frame.radius_at(float(point[2])), places=11)
                self.assertAlmostEqual(float(np.dot(normal, point)), PLANE_OFFSET, places=11)
                expected = np.asarray(
                    (
                        radial * cos(float(theta)),
                        radial * sin(float(theta)),
                    )
                )
                np.testing.assert_allclose(point[:2], expected, atol=1.0e-11)

    def test_invalid_progress_fails_closed(self) -> None:
        for value in (-0.01, 1.01, float("nan"), True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "progress must lie"):
                    compute_switch_frame(value)


class DandelinConeCylinderSwitchCairoTests(unittest.TestCase):
    def test_keyframes_have_sphere_surface_and_section_pixels(self) -> None:
        frames: list[np.ndarray] = []
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 6,
                "disable_caching": True,
                "write_to_movie": False,
                "save_last_frame": False,
            }
        ):
            for progress in (0.0, 0.5, 1.0):
                scene = Scene()
                scene.camera.background_color = BACKGROUND_COLOR
                scene.add(build_switch_diagram(progress))
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                pixels = scene.camera.pixel_array[:, :, :3].copy()
                frames.append(pixels)

                background = np.asarray((11, 22, 34), dtype=int)
                non_background = np.linalg.norm(
                    pixels.astype(int) - background,
                    axis=2,
                ) > 4.0
                red = pixels[:, :, 0].astype(int)
                green = pixels[:, :, 1].astype(int)
                blue = pixels[:, :, 2].astype(int)
                sphere_orange = (
                    (red > 135)
                    & (green > 60)
                    & (green < 190)
                    & ((red - blue) > 25)
                )
                section_yellow = (
                    (red > 170)
                    & (green > 130)
                    & (blue < 165)
                    & ((red - blue) > 35)
                )
                self.assertGreater(int(np.count_nonzero(non_background)), 1300)
                self.assertGreater(int(np.count_nonzero(sphere_orange)), 35)
                self.assertGreater(int(np.count_nonzero(section_yellow)), 8)

        self.assertGreater(int(np.count_nonzero(frames[0] != frames[1])), 1500)
        self.assertGreater(int(np.count_nonzero(frames[1] != frames[2])), 1500)
        self.assertGreater(int(np.count_nonzero(frames[0] != frames[2])), 2500)


if __name__ == "__main__":
    unittest.main()
