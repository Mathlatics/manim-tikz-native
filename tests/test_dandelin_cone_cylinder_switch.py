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
    PlaneDepthRole,
    SURFACE_RADIUS,
    build_switch_diagram,
    compute_switch_occlusion_frame,
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


class DandelinConeCylinderSwitchOcclusionTests(unittest.TestCase):
    def test_five_keyframes_have_certified_plane_roles_and_sphere_order(self) -> None:
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(progress=progress):
                frame = compute_switch_occlusion_frame(progress)
                self.assertTrue(frame.surface_layering_authoritative)
                self.assertFalse(frame.physical_surface_visibility_authoritative)
                self.assertEqual(
                    {role for role, _paths in frame.plane_contours},
                    set(PlaneDepthRole),
                )
                self.assertTrue(
                    all(frame.contours_for(role) for role in PlaneDepthRole)
                )

                far = next(
                    item for item in frame.sphere_layers if item.plane_is_in_front
                )
                near = next(
                    item
                    for item in frame.sphere_layers
                    if not item.plane_is_in_front
                )
                self.assertEqual(far.plane_side, -1)
                self.assertEqual(near.plane_side, 1)
                rank = {
                    item_id: index
                    for index, item_id in enumerate(frame.draw_order)
                }
                plane_ids = {
                    role: (fill_id, outline_id)
                    for role, fill_id, outline_id in frame.plane_item_ids
                }
                for item_id in plane_ids[PlaneDepthRole.BEHIND_SURFACE]:
                    self.assertLess(rank[item_id], rank[far.item_id])
                for role in (
                    PlaneDepthRole.OUTSIDE_PROJECTION,
                    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                    PlaneDepthRole.IN_FRONT_OF_SURFACE,
                ):
                    for item_id in plane_ids[role]:
                        self.assertLess(rank[far.item_id], rank[item_id])
                for role in (
                    PlaneDepthRole.BEHIND_SURFACE,
                    PlaneDepthRole.OUTSIDE_PROJECTION,
                    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                ):
                    for item_id in plane_ids[role]:
                        self.assertLess(rank[item_id], rank[near.item_id])
                self.assertLess(
                    rank[frame.hidden_section_item_id],
                    rank[near.item_id],
                )
                self.assertLess(
                    rank[far.item_id],
                    rank[frame.hidden_section_item_id],
                )
                self.assertLess(
                    rank[near.item_id],
                    rank[frame.visible_section_item_id],
                )

    def test_near_cylinder_plane_partition_converges_without_role_loss(self) -> None:
        near = compute_switch_occlusion_frame(0.9999)
        cylinder = compute_switch_occlusion_frame(1.0)

        def signed_area(
            contours: tuple[tuple[tuple[float, float], ...], ...],
        ) -> float:
            result = 0.0
            for contour in contours:
                points = np.asarray(contour, dtype=float)
                result += 0.5 * float(
                    np.sum(
                        points[:, 0] * np.roll(points[:, 1], -1)
                        - points[:, 1] * np.roll(points[:, 0], -1)
                    )
                )
            return result

        for role in PlaneDepthRole:
            with self.subTest(role=role.value):
                self.assertAlmostEqual(
                    signed_area(near.contours_for(role)),
                    signed_area(cylinder.contours_for(role)),
                    delta=1.0e-3,
                )


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
            for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
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
        self.assertGreater(int(np.count_nonzero(frames[1] != frames[2])), 1200)
        self.assertGreater(int(np.count_nonzero(frames[2] != frames[3])), 1000)
        self.assertGreater(int(np.count_nonzero(frames[3] != frames[4])), 800)
        self.assertGreater(int(np.count_nonzero(frames[0] != frames[4])), 2500)

    def test_cylinder_plane_outline_switches_order_across_the_two_spheres(self) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 960,
                "pixel_height": 540,
                "frame_rate": 12,
                "disable_caching": True,
                "write_to_movie": False,
                "save_last_frame": False,
            }
        ):
            scene = Scene()
            scene.camera.background_color = BACKGROUND_COLOR
            scene.add(build_switch_diagram(1.0))
            scene.camera.reset()
            scene.camera.capture_mobjects(scene.mobjects)
            pixels = scene.camera.pixel_array[:, :, :3].astype(int)

        far_sphere_boundary = pixels[382:387, 448:453]
        near_sphere_boundary = pixels[233:238, 478:483]
        far_cyan = float(
            np.mean(
                far_sphere_boundary[:, :, 1]
                - far_sphere_boundary[:, :, 0]
            )
        )
        near_cyan = float(
            np.mean(
                near_sphere_boundary[:, :, 1]
                - near_sphere_boundary[:, :, 0]
            )
        )
        self.assertGreater(far_cyan - near_cyan, 12.0)


if __name__ == "__main__":
    unittest.main()
