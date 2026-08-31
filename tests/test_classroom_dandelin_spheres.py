from __future__ import annotations

import unittest

import numpy as np
from manim import Scene, config, tempconfig

from examples.classroom_dandelin_spheres.classroom_dandelin_spheres import (
    ACTS,
    BACKGROUND_COLOR,
    DandelinThreeConicsLesson,
    build_dandelin_act,
)
from polyhedron_visibility.quadrics import (
    DandelinConicFamily,
    DandelinSectionAuthoringError,
)


class ClassroomDandelinMetadataTests(unittest.TestCase):
    def test_three_acts_cover_the_three_non_degenerate_conic_families(self) -> None:
        self.assertEqual(
            DandelinThreeConicsLesson.__name__,
            "DandelinThreeConicsLesson",
        )
        self.assertEqual(
            tuple(item.act_id for item in ACTS),
            ("ellipse", "parabola", "hyperbola"),
        )
        self.assertEqual(len({item.heading for item in ACTS}), 3)
        self.assertTrue(all(item.explanation.strip() for item in ACTS))


class ClassroomDandelinFacadeTests(unittest.TestCase):
    def test_every_act_uses_fixed_identity_and_restores_the_scene(self) -> None:
        expected = {
            "ellipse": (DandelinConicFamily.ELLIPSE, 2),
            "parabola": (DandelinConicFamily.PARABOLA, 1),
            "hyperbola": (DandelinConicFamily.HYPERBOLA, 2),
        }
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 6,
                "disable_caching": True,
            }
        ):
            for act in ACTS:
                with self.subTest(act=act.act_id):
                    scene = Scene()
                    facade = build_dandelin_act(scene, act)
                    display_id = id(facade.display_mobject)
                    with self.assertRaisesRegex(
                        DandelinSectionAuthoringError,
                        "slot_identities.*only while attached",
                    ):
                        facade.slot_identities()

                    facade.attach()
                    slot_ids = facade.slot_identities()

                    family, sphere_count = expected[act.act_id]
                    self.assertIs(facade.construction.family, family)
                    self.assertEqual(len(facade.construction.spheres), sphere_count)
                    self.assertFalse(facade.visibility_authoritative)
                    self.assertEqual(facade.overlay_mode, "diagrammatic")
                    self.assertEqual(facade.slot_identities(), slot_ids)
                    self.assertEqual(id(facade.display_mobject), display_id)
                    self.assertTrue(scene.mobjects)

                    facade.restore()

                    self.assertEqual(scene.mobjects, [])
                    with self.assertRaisesRegex(
                        DandelinSectionAuthoringError,
                        "slot_identities.*only while attached",
                    ):
                        facade.slot_identities()

                    facade.attach()
                    try:
                        self.assertTrue(facade.slot_identities())
                        self.assertEqual(id(facade.display_mobject), display_id)
                    finally:
                        facade.restore()
                    self.assertEqual(scene.mobjects, [])

    def test_restore_purges_focus_family_from_cairo_runtime_caches(self) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 6,
                "disable_caching": True,
            }
        ):
            scene = Scene()
            facade = build_dandelin_act(scene, ACTS[0]).attach()
            focus_family = tuple(facade.focus_group.get_family())
            focus_ids = {id(item) for item in focus_family}
            scene.moving_mobjects = [*scene.moving_mobjects, *focus_family]
            scene.static_mobjects = [*scene.static_mobjects, *focus_family]

            facade.restore()

            self.assertFalse(facade.attached)
            for container in (
                scene.mobjects,
                scene.moving_mobjects,
                scene.static_mobjects,
            ):
                self.assertTrue(
                    focus_ids.isdisjoint(id(item) for item in container)
                )


class ClassroomDandelinCairoTests(unittest.TestCase):
    def test_three_certified_acts_produce_section_overlay_and_focus_pixels(
        self,
    ) -> None:
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
            for act in ACTS:
                with self.subTest(act=act.act_id):
                    scene = Scene()
                    scene.camera.background_color = BACKGROUND_COLOR
                    facade = build_dandelin_act(scene, act)
                    scene.camera.reset()
                    scene.camera.capture_mobjects(scene.mobjects)
                    pristine = scene.camera.pixel_array.copy()
                    facade.attach()
                    try:
                        scene.camera.reset()
                        scene.camera.capture_mobjects(scene.mobjects)
                        pixels = scene.camera.pixel_array[:, :, :3]
                        red = pixels[:, :, 0].astype(int)
                        green = pixels[:, :, 1].astype(int)
                        blue = pixels[:, :, 2].astype(int)
                        background = np.asarray((13, 23, 34), dtype=int)
                        non_background = np.linalg.norm(
                            pixels.astype(int) - background,
                            axis=2,
                        ) > 4.0
                        section_yellow = (
                            (red > 170)
                            & (green > 135)
                            & (blue < 150)
                            & ((red - blue) > 45)
                        )
                        overlay_orange = (
                            (red > 135)
                            & (green > 65)
                            & (green < 175)
                            & (blue < 160)
                            & ((red - green) > 25)
                        )
                        focus_ink = (
                            (red > 230)
                            & (green > 220)
                            & (blue > 130)
                        )
                        self.assertGreater(
                            int(np.count_nonzero(non_background)),
                            1000,
                        )
                        self.assertGreater(
                            int(np.count_nonzero(section_yellow)),
                            10,
                        )
                        self.assertGreater(
                            int(np.count_nonzero(overlay_orange)),
                            20,
                        )
                        self.assertGreater(
                            int(np.count_nonzero(focus_ink)),
                            0,
                        )
                        for record in facade.construction.spheres:
                            screen = facade.view.matrix[:2] @ np.asarray(
                                record.focus.world_point,
                                dtype=float,
                            )
                            column = int(
                                round(
                                    (
                                        float(screen[0])
                                        / float(config.frame_width)
                                        + 0.5
                                    )
                                    * (int(config.pixel_width) - 1)
                                )
                            )
                            row = int(
                                round(
                                    (
                                        0.5
                                        - float(screen[1])
                                        / float(config.frame_height)
                                    )
                                    * (int(config.pixel_height) - 1)
                                )
                            )
                            focus_patch = focus_ink[
                                max(0, row - 3) : row + 4,
                                max(0, column - 3) : column + 4,
                            ]
                            self.assertGreater(
                                int(np.count_nonzero(focus_patch)),
                                0,
                            )
                    finally:
                        facade.restore()
                    self.assertEqual(scene.mobjects, [])
                    scene.camera.reset()
                    scene.camera.capture_mobjects(scene.mobjects)
                    np.testing.assert_array_equal(
                        scene.camera.pixel_array,
                        pristine,
                    )


if __name__ == "__main__":
    unittest.main()
