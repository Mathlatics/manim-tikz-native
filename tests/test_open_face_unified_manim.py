from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import (
    BLACK,
    RED,
    Line,
    Polygon,
    Scene,
    ThreeDScene,
    ValueTracker,
    tempconfig,
)

from polyhedron_visibility.api import ParallelProjection
from polyhedron_visibility.binding import OcclusionBindingError
from polyhedron_visibility.open_faces import OpenFaceOcclusion3D, OpenFaceScene3D
from polyhedron_visibility.style import OcclusionStyle


class _UnifiedFixture:
    def __init__(
        self,
        scene: Scene,
        *,
        paint_policy: str = "diagrammatic",
        shared_source_z: bool = True,
        fixed_display: bool = False,
    ) -> None:
        self.scene = scene
        self.positions = {
            "A": np.array((-1.0, -1.0, 1.0)),
            "B": np.array((1.0, -1.0, 1.0)),
            "C": np.array((1.0, 1.0, 1.0)),
            "D": np.array((-1.0, 1.0, 1.0)),
            "P": np.array((-2.0, 0.0, 0.0)),
            "Q": np.array((2.0, 0.0, 0.0)),
        }
        self.face = Polygon(
            *(self.positions[key] for key in ("A", "B", "C", "D")),
            color=RED,
            fill_opacity=1.0,
            stroke_opacity=0.0,
        ).set_z_index(0)
        self.path = Line(
            self.positions["P"],
            self.positions["Q"],
            buff=0,
            color=BLACK,
            stroke_width=8,
            stroke_opacity=1.0,
        ).set_z_index(0 if shared_source_z else 1)
        scene.add(self.face, self.path)

        builder = OpenFaceScene3D("unified-open-face-manim")
        for vertex_id in sorted(self.positions):
            builder.vertex(
                vertex_id,
                lambda key=vertex_id: self.positions[key].copy(),
            )
        builder.face(
            "panel",
            ("A", "B", "C", "D"),
            logical_surface_id="panel-surface",
            source_mobject=self.face,
        )
        builder.stroke("probe", "P", "Q", self.path)
        self.controller = builder.controller(
            scene,
            projection=ParallelProjection.identity(),
            display_point_provider=(
                (lambda point: np.asarray(point, dtype=float))
                if fixed_display
                else None
            ),
            style=OcclusionStyle(
                max_projected_length=8.0,
                dash_length=0.20,
                dash_gap=0.10,
            ),
            compositing_mode="unified",
            paint_policy=paint_policy,
            painter_z_band=(20.0, 30.0),
        )

    def sync_sources(self) -> None:
        self.face.become(
            Polygon(
                *(self.positions[key] for key in ("A", "B", "C", "D")),
                color=RED,
                fill_opacity=1.0,
                stroke_opacity=0.0,
            ).set_z_index(0)
        )
        self.path.become(
            Line(
                self.positions["P"],
                self.positions["Q"],
                buff=0,
                color=BLACK,
                stroke_width=8,
                stroke_opacity=1.0,
            ).set_z_index(0)
        )


class OpenFaceUnifiedManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "frame_rate": 8,
                "pixel_width": 320,
                "pixel_height": 180,
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_public_entry_reaches_unified_runtime_and_allows_shared_source_z(self) -> None:
        fixture = _UnifiedFixture(Scene())
        controller = fixture.controller.attach()
        self.assertIsInstance(controller, OpenFaceOcclusion3D)
        self.assertEqual(controller.compositing_mode, "unified")
        self.assertIsNotNone(controller.last_unified_frame)
        self.assertEqual(
            set(controller.active_painter_z_indices),
            set(controller.last_unified_frame.draw_order),
        )
        self.assertEqual(fixture.face.get_fill_opacity(), 0.0)
        self.assertEqual(fixture.path.get_stroke_opacity(), 0.0)
        self.assertIn(controller.overlay_root, fixture.scene.mobjects)
        controller.restore()
        self.assertEqual(fixture.face.get_fill_opacity(), 1.0)
        self.assertEqual(fixture.path.get_stroke_opacity(), 1.0)

    def test_unrelated_drawable_in_explicit_band_fails_before_hiding(self) -> None:
        scene = Scene()
        fixture = _UnifiedFixture(scene)
        scene.add(Line((-1, 2, 0), (1, 2, 0)).set_z_index(25))
        with self.assertRaisesRegex(OcclusionBindingError, "managed painter z band"):
            fixture.controller.attach()
        self.assertEqual(fixture.face.get_fill_opacity(), 1.0)
        self.assertEqual(fixture.path.get_stroke_opacity(), 1.0)
        self.assertNotIn(fixture.controller.overlay_root, scene.mobjects)

    def test_invalid_frame_and_apply_failure_preserve_last_good_slots(self) -> None:
        fixture = _UnifiedFixture(Scene())
        controller = fixture.controller.attach()
        identities = controller.slot_identities()
        snapshot = controller.slot_snapshot()
        last_good = controller.last_unified_frame
        previous_z = controller.active_painter_z_indices

        fixture.positions["P"][2] = -0.5
        fixture.positions["Q"][2] = -0.5
        fixture.sync_sources()
        runtime = controller._unified_runtime
        assert runtime is not None
        original_apply = runtime._band.apply

        def fail_after_band(prepared) -> None:
            original_apply(prepared)
            raise RuntimeError("commit failure")

        with patch.object(runtime._band, "apply", side_effect=fail_after_band):
            with self.assertRaisesRegex(RuntimeError, "commit failure"):
                controller.update()
        self.assertEqual(controller.slot_snapshot(), snapshot)
        self.assertEqual(controller.active_painter_z_indices, previous_z)
        self.assertIs(controller.last_unified_frame, last_good)
        self.assertEqual(controller.slot_identities(), identities)
        self.assertEqual(fixture.face.get_fill_opacity(), 0.0)
        self.assertEqual(fixture.path.get_stroke_opacity(), 0.0)

        fixture.positions["Q"][0] = float("nan")
        snapshot = controller.slot_snapshot()
        with self.assertRaises(Exception):
            controller.update()
        self.assertEqual(controller.slot_snapshot(), snapshot)
        self.assertIs(controller.last_unified_frame, last_good)
        fixture.positions["Q"] = np.array((2.0, 0.0, -0.5))
        controller.restore()

    def test_display_opacity_survives_geometry_updates_and_reattach(self) -> None:
        fixture = _UnifiedFixture(Scene())
        controller = fixture.controller.attach()
        identities = controller.slot_identities()
        controller.display_mobject.set_opacity(0.25)
        fixture.positions["P"][1] = -0.25
        fixture.positions["Q"][1] = -0.25
        fixture.sync_sources()
        controller.update()
        runtime = controller._unified_runtime
        assert runtime is not None
        self.assertAlmostEqual(runtime.root.opacity_multiplier, 0.25)
        active = controller.last_unified_frame.path_fragments
        visible = next(item for item in active if item.visibility_kind.value == "visible")
        visible_index = [
            item for item in active if item.source_path_id == visible.source_path_id
        ].index(visible)
        solid = runtime.path_slots[visible.source_path_id].fragments[visible_index].solid
        self.assertLessEqual(float(solid.get_stroke_opacity()), 0.2500001)
        proxy = controller._face_fill_layer.proxies["panel"]
        self.assertLessEqual(float(proxy.get_fill_opacity()), 0.2500001)

        controller.detach()
        self.assertEqual(fixture.face.get_fill_opacity(), 1.0)
        self.assertEqual(fixture.path.get_stroke_opacity(), 1.0)
        controller.attach()
        self.assertEqual(controller.slot_identities(), identities)
        self.assertAlmostEqual(runtime.root.opacity_multiplier, 1.0)
        controller.restore()

    def test_reattach_revalidates_reserved_band(self) -> None:
        scene = Scene()
        fixture = _UnifiedFixture(scene)
        controller = fixture.controller.attach()
        controller.detach()
        scene.add(Line((-1, 2, 0), (1, 2, 0)).set_z_index(25))
        with self.assertRaisesRegex(OcclusionBindingError, "managed painter z band"):
            controller.attach()
        self.assertEqual(fixture.face.get_fill_opacity(), 1.0)
        self.assertEqual(fixture.path.get_stroke_opacity(), 1.0)

    def test_update_revalidates_reserved_band_and_excludes_managed_slots(self) -> None:
        scene = Scene()
        fixture = _UnifiedFixture(scene)
        controller = fixture.controller.attach()
        controller.update()
        last_good = controller.last_unified_frame
        snapshot = controller.slot_snapshot()

        intruder = Line((-1, 2, 0), (1, 2, 0)).set_z_index(25)
        scene.add(intruder)
        with self.assertRaisesRegex(OcclusionBindingError, "managed painter z band"):
            controller.update()
        self.assertIs(controller.last_unified_frame, last_good)
        self.assertEqual(controller.slot_snapshot(), snapshot)
        self.assertEqual(fixture.face.get_fill_opacity(), 0.0)
        self.assertEqual(fixture.path.get_stroke_opacity(), 0.0)

        scene.remove(intruder)
        controller.update()
        self.assertIsNot(controller.last_unified_frame, last_good)
        controller.restore()

    def test_diagrammatic_and_physical_modes_change_real_cairo_pixels(self) -> None:
        arrays = []
        for policy in ("diagrammatic", "physical"):
            scene = Scene()
            fixture = _UnifiedFixture(scene, paint_policy=policy)
            with fixture.controller.session():
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                arrays.append(scene.camera.pixel_array.copy())
        self.assertFalse(np.array_equal(arrays[0], arrays[1]))
        difference = np.abs(arrays[0].astype(int) - arrays[1].astype(int))
        self.assertGreater(int(np.count_nonzero(difference)), 10)

    def test_real_cairo_fade_lifecycle_uses_display_mobject(self) -> None:
        from manim import FadeIn, FadeOut

        class UnifiedScene(Scene):
            def construct(inner_self) -> None:
                fixture = _UnifiedFixture(inner_self)
                controller = fixture.controller.attach()
                runtime = controller._unified_runtime
                assert runtime is not None
                update_driver = controller._unified_update_driver
                assert update_driver is not None
                identities = controller.slot_identities()
                inner_self.display_is_runtime_root = (
                    controller.display_mobject is runtime.root
                )
                fade_samples = []
                motion = ValueTracker(0.0)

                def move_source(line) -> None:
                    y = motion.get_value()
                    fixture.positions["P"][1] = y
                    fixture.positions["Q"][1] = y
                    line.put_start_and_end_on(
                        fixture.positions["P"],
                        fixture.positions["Q"],
                    )

                fixture.path.add_updater(move_source)
                frame_before_fade = controller.last_unified_frame

                def capture_opacity(_mobject, dt) -> None:
                    del dt
                    fade_samples.append(runtime.root.opacity_multiplier)

                update_driver.add_updater(capture_opacity)
                inner_self.play(
                    FadeOut(controller.display_mobject),
                    motion.animate.set_value(0.5),
                    run_time=0.4,
                )
                fixture.path.remove_updater(move_source)
                inner_self.fade_out_minimum = min(fade_samples)
                inner_self.updated_during_fade_out = (
                    controller.last_unified_frame is not frame_before_fade
                    and np.isclose(fixture.positions["P"][1], 0.5)
                )
                inner_self.fade_out_ownership = (
                    controller.display_mobject not in inner_self.mobjects
                    and update_driver in inner_self.mobjects
                )
                fade_samples.clear()
                inner_self.play(
                    FadeIn(controller.display_mobject),
                    run_time=0.4,
                )
                inner_self.fade_in_minimum = min(fade_samples)
                inner_self.after_fade_in = runtime.root.opacity_multiplier
                inner_self.fade_in_ownership = (
                    inner_self.mobjects.count(controller.display_mobject) == 1
                    and inner_self.mobjects.count(update_driver) == 1
                )
                previous_frame = controller.last_unified_frame
                fixture.positions["P"][1] = 0.75
                fixture.positions["Q"][1] = 0.75
                fixture.sync_sources()
                inner_self.wait(0.4)
                inner_self.updated_after_fade_in = (
                    controller.last_unified_frame is not previous_frame
                )
                controller.detach()
                controller.attach()
                inner_self.same_identities = identities == controller.slot_identities()
                controller.restore()
                inner_self.restored = (
                    fixture.face.get_fill_opacity(),
                    fixture.path.get_stroke_opacity(),
                )

        with TemporaryDirectory() as media_dir, tempconfig(
            {
                "renderer": "cairo",
                "media_dir": media_dir,
                "pixel_width": 160,
                "pixel_height": 90,
                "frame_rate": 5,
                "disable_caching": True,
                "write_to_movie": True,
                "save_last_frame": False,
            }
        ):
            scene = UnifiedScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertTrue(scene.display_is_runtime_root)
            self.assertLess(scene.fade_out_minimum, 1.0)
            self.assertTrue(scene.updated_during_fade_out)
            self.assertTrue(scene.fade_out_ownership)
            self.assertLessEqual(scene.fade_in_minimum, 1.0e-6)
            self.assertGreaterEqual(scene.after_fade_in, 1.0 - 1.0e-6)
            self.assertTrue(scene.fade_in_ownership)
            self.assertTrue(scene.updated_after_fade_in)
            self.assertTrue(scene.same_identities)
            self.assertEqual(scene.restored, (1.0, 1.0))

    def test_fixed_frame_fade_detach_and_reattach_do_not_leak(self) -> None:
        from manim import FadeIn, FadeOut

        class FixedFrameScene(ThreeDScene):
            def construct(inner_self) -> None:
                fixture = _UnifiedFixture(inner_self, fixed_display=True)
                controller = fixture.controller.attach()
                runtime = controller._unified_runtime
                update_driver = controller._unified_update_driver
                assert runtime is not None and update_driver is not None
                family = set(controller.overlay_root.get_family())
                inner_self.fixed_after_attach = family.issubset(
                    inner_self.camera.fixed_in_frame_mobjects
                )

                inner_self.play(FadeOut(controller.display_mobject), run_time=0.2)
                inner_self.play(FadeIn(controller.display_mobject), run_time=0.2)
                inner_self.single_scene_ownership = (
                    inner_self.mobjects.count(runtime.root) == 1
                    and inner_self.mobjects.count(update_driver) == 1
                )

                controller.detach()
                inner_self.clear_after_detach = not (
                    family & inner_self.camera.fixed_in_frame_mobjects
                )
                controller.attach()
                inner_self.fixed_after_reattach = family.issubset(
                    inner_self.camera.fixed_in_frame_mobjects
                )
                controller.restore()
                inner_self.clear_after_restore = not (
                    family & inner_self.camera.fixed_in_frame_mobjects
                )

        with TemporaryDirectory() as media_dir, tempconfig(
            {
                "renderer": "cairo",
                "media_dir": media_dir,
                "pixel_width": 160,
                "pixel_height": 90,
                "frame_rate": 5,
                "disable_caching": True,
                "write_to_movie": True,
                "save_last_frame": False,
            }
        ):
            scene = FixedFrameScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertTrue(scene.fixed_after_attach)
            self.assertTrue(scene.single_scene_ownership)
            self.assertTrue(scene.clear_after_detach)
            self.assertTrue(scene.fixed_after_reattach)
            self.assertTrue(scene.clear_after_restore)


if __name__ == "__main__":
    unittest.main()
