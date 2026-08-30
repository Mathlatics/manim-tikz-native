"""Manim/Cairo acceptance for semantic parallel-camera shot playback."""

from __future__ import annotations

from math import pi
import unittest
from unittest.mock import patch

import numpy as np
from manim import Dot, Mobject, ThreeDScene, tempconfig

from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricOcclusion3D,
)
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_shots import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
    parallel_camera_shot_progress,
)
from tikz_native.parallel_shots_manim import (
    ParallelCameraShotManimError,
    ParallelCameraTargetFollowController,
    play_parallel_camera_shot,
    play_parallel_camera_shot_sequence,
)


try:
    import cairo as _cairo  # noqa: F401
    from manim.renderer.cairo_renderer import (  # noqa: F401
        CairoRenderer as _CairoRenderer,
    )
except (ImportError, OSError):
    CAIRO_AVAILABLE = False
else:
    CAIRO_AVAILABLE = True


class _ParallelScene(ThreeDScene):
    def __init__(self) -> None:
        super().__init__(camera_class=MultiProjectionCamera)


class _RecordingParallelScene(_ParallelScene):
    def __init__(self) -> None:
        super().__init__()
        self.play_run_times: list[float] = []
        self.play_rate_funcs: list[object] = []
        self.wait_durations: list[float] = []

    def play(self, *animations, **kwargs):
        if "run_time" in kwargs:
            self.play_run_times.append(float(kwargs["run_time"]))
        if "rate_func" in kwargs:
            self.play_rate_funcs.append(kwargs["rate_func"])
        return super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs):
        self.wait_durations.append(float(duration))
        return super().wait(duration, *args, **kwargs)


def _state(index: int) -> ParallelCameraState:
    directions = (
        (1.0, 1.2, 0.8),
        (-0.7, 1.0, 1.3),
        (1.1, -0.6, 1.0),
    )
    anchors = ((-0.25, 0.12), (0.18, -0.21), (0.31, 0.16))
    targets = ((0.2, -0.1, 0.3), (0.8, 0.4, -0.2), (-0.5, 0.7, 0.6))
    return ParallelCameraState.from_view_direction(
        directions[index],
        target=targets[index],
        screen_anchor=anchors[index],
        zoom=(0.92, 1.08, 0.84)[index],
    )


def _shot(
    index: int,
    *,
    duration: float = 0.12,
    hold: float = 0.1,
    transition: str = "orbit",
) -> ParallelCameraShot:
    return ParallelCameraShot(
        f"manim-shot-{index}",
        _state(index),
        duration=duration,
        hold=hold,
        transition=transition,
        arc_height=0.6,
    )


def _quadric_limits(*, max_surfaces: int) -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=max_surfaces,
        max_curves=8,
        max_fragments_per_curve=24,
        max_segments_per_fragment=192,
        max_surface_segments=320,
        max_dashes_per_fragment=64,
        max_projected_length=24.0,
        max_total_mobjects=30000,
        max_boundary_sources=32,
    )


def _assert_state_exact(
    testcase: unittest.TestCase,
    actual: ParallelCameraState,
    expected: ParallelCameraState,
) -> None:
    testcase.assertTrue(np.array_equal(actual.matrix, expected.matrix))
    testcase.assertTrue(np.array_equal(actual.target, expected.target))
    testcase.assertTrue(
        np.array_equal(actual.screen_anchor, expected.screen_anchor)
    )
    testcase.assertEqual(actual.zoom, expected.zoom)


@unittest.skipUnless(CAIRO_AVAILABLE, "Cairo is required for real playback")
class ParallelCameraShotPlaybackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "frame_rate": 12,
                "pixel_width": 240,
                "pixel_height": 135,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
                "progress_bar": "none",
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_real_cairo_play_uses_authored_timing_and_exact_endpoint(self) -> None:
        scene = _RecordingParallelScene()
        scene.add(Dot(point=(0.8, 0.4, -0.2), radius=0.12))
        shot = _shot(1)
        camera = scene.camera
        self.assertIsInstance(camera, MultiProjectionCamera)
        with patch.object(
            camera,
            "animate_to_parallel_state",
            wraps=camera.animate_to_parallel_state,
        ) as animate:
            endpoint = play_parallel_camera_shot(scene, shot)
        animate.assert_called_once_with(
            shot.state,
            transition=shot.transition,
            arc_height=shot.arc_height,
        )
        self.assertEqual(scene.play_run_times, [shot.duration])
        self.assertEqual(scene.play_rate_funcs, [parallel_camera_shot_progress])
        self.assertEqual(scene.wait_durations, [shot.hold])
        self.assertIs(endpoint, shot.state)
        self.assertIs(camera.snapshot_parallel_state(), shot.state)
        _assert_state_exact(self, camera.snapshot_parallel_state(), shot.state)

        scene.camera.reset()
        scene.camera.capture_mobjects(scene.mobjects)
        background = scene.camera.background_color.to_rgb()
        pixels = scene.camera.pixel_array[:, :, :3].astype(float) / 255.0
        self.assertGreater(
            float(np.max(np.linalg.norm(pixels - background, axis=2))),
            0.2,
        )

    def test_sequence_plays_in_order_and_each_next_shot_sees_prior_endpoint(
        self,
    ) -> None:
        scene = _RecordingParallelScene()
        first = _shot(0, duration=0.1, hold=0.1)
        second = _shot(2, duration=0.12, hold=0.0)
        sequence = ParallelCameraShotSequence((first, second))
        starts: list[ParallelCameraState] = []
        targets: list[ParallelCameraState] = []
        original = scene.camera.animate_to_parallel_state

        def capture_start(target, **kwargs):
            starts.append(scene.camera.snapshot_parallel_state())
            targets.append(target)
            return original(target, **kwargs)

        with patch.object(
            scene.camera,
            "animate_to_parallel_state",
            side_effect=capture_start,
        ):
            endpoint = play_parallel_camera_shot_sequence(scene, sequence)

        self.assertEqual(targets, [first.state, second.state])
        self.assertEqual(scene.play_run_times, [first.duration, second.duration])
        self.assertEqual(scene.wait_durations, [first.hold])
        _assert_state_exact(self, starts[1], first.state)
        self.assertIs(endpoint, second.state)
        self.assertIs(scene.camera.snapshot_parallel_state(), second.state)

    def test_play_failure_restores_source_instead_of_intermediate_state(self) -> None:
        scene = _ParallelScene()
        source = _state(0)
        scene.camera.set_parallel_state(source)
        shot = _shot(1)
        with patch.object(
            scene,
            "play",
            side_effect=RuntimeError("synthetic play failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic play failure"):
                play_parallel_camera_shot(scene, shot)
        self.assertIs(scene.camera.snapshot_parallel_state(), source)

    def test_hold_failure_also_restores_the_pre_shot_state(self) -> None:
        scene = _ParallelScene()
        source = _state(0)
        scene.camera.set_parallel_state(source)
        shot = _shot(2)

        def finish_animation(*_animations, **_kwargs) -> None:
            scene.camera.transition_tracker.set_value(1.0)

        with (
            patch.object(scene, "play", side_effect=finish_animation),
            patch.object(
                scene,
                "wait",
                side_effect=RuntimeError("synthetic hold failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic hold failure"):
                play_parallel_camera_shot(scene, shot)
        self.assertIs(scene.camera.snapshot_parallel_state(), source)

    def test_wrong_camera_and_wrong_authoring_types_fail_before_play(self) -> None:
        ordinary_scene = ThreeDScene()
        with self.assertRaisesRegex(TypeError, "MultiProjectionCamera"):
            play_parallel_camera_shot(ordinary_scene, _shot(0))
        scene = _ParallelScene()
        with self.assertRaisesRegex(TypeError, "ParallelCameraShot"):
            play_parallel_camera_shot(scene, object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "ParallelCameraShotSequence"):
            play_parallel_camera_shot_sequence(
                scene,
                object(),  # type: ignore[arg-type]
            )


class ParallelCameraTargetFollowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "frame_rate": 12,
                "pixel_width": 240,
                "pixel_height": 135,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
                "progress_bar": "none",
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def _assert_same_frame_camera_sample(
        self,
        scene: _ParallelScene,
        controller: object,
        shot: ParallelCameraShot,
    ) -> None:
        scene.camera.set_parallel_state(shot.state)
        dynamic_target = np.asarray((1.4, -0.8, 0.6), dtype=float)
        next_target = np.asarray((-1.1, 0.7, 1.5), dtype=float)
        unrelated = Mobject()

        def produce_target(_mobject: Mobject, dt: float) -> None:
            del dt
            dynamic_target[:] = next_target

        unrelated.add_updater(produce_target)
        scene.add(unrelated)
        controller.attach()

        follow = ParallelCameraTargetFollowController(
            scene,
            lambda: dynamic_target,
        )
        follow.start(shot)
        driver = follow.driver_mobject
        consumer = controller._update_driver
        self.assertLess(scene.mobjects.index(unrelated), scene.mobjects.index(driver))
        self.assertLess(scene.mobjects.index(driver), scene.mobjects.index(consumer))
        self.assertEqual(scene.mobjects.count(driver), 1)

        sampled: list[ParallelCameraState] = []
        original_update = controller.update

        def capture_update(dt: float) -> None:
            sampled.append(scene.camera.snapshot_parallel_state())
            original_update(dt)

        try:
            with (
                patch.object(controller, "update", side_effect=capture_update),
                patch.object(
                    Mobject,
                    "__init__",
                    side_effect=AssertionError("follow frame allocated a Mobject"),
                ),
            ):
                scene.update_mobjects(1.0 / 12.0)
            self.assertEqual(len(sampled), 1)
            np.testing.assert_array_equal(dynamic_target, next_target)
            np.testing.assert_array_equal(sampled[0].target, next_target)
            np.testing.assert_array_equal(sampled[0].matrix, shot.state.matrix)
            np.testing.assert_array_equal(
                sampled[0].screen_anchor,
                shot.state.screen_anchor,
            )
            self.assertEqual(sampled[0].zoom, shot.state.zoom)
            self.assertIs(follow.driver_mobject, driver)
        finally:
            follow.restore()
            controller.restore()

    def test_single_quadric_samples_followed_target_in_the_same_frame(self) -> None:
        scene = _ParallelScene()
        shot = _shot(0, hold=0.0)
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(SphereSpec("follow-sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=(),
            projection=lambda active_scene: active_scene.camera,
            include_surface_boundaries=False,
            limits=_quadric_limits(max_surfaces=1),
        )
        self._assert_same_frame_camera_sample(scene, controller, shot)

    def test_composite_quadric_samples_followed_target_in_the_same_frame(
        self,
    ) -> None:
        scene = _ParallelScene()
        plane = SectionPlane(
            "follow-double-plane",
            (0.0, 0.48, 0.0),
            (0.0, 1.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        state = ParallelCameraState.relative_to_plane(
            plane,
            inclination_degrees=14.0,
            azimuth_degrees=0.0,
            target=plane.point,
        )
        shot = ParallelCameraShot(
            "follow-double-shot",
            state,
            duration=0.1,
            hold=0.0,
        )
        controller = CompositeQuadricSection3D(
            scene,
            surface=ConeSpec(
                "follow-double",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                pi / 4.0,
                (-2.0, 2.0),
                radial_axis=(1.0, 0.0, 0.0),
                model=ConeModel.OPEN_DOUBLE,
            ),
            section_id="follow-double-section",
            plane=plane,
            projection=lambda active_scene: active_scene.camera,
            draw_section_boundary=True,
            include_surface_boundaries=False,
            limits=_quadric_limits(max_surfaces=2),
            max_chord_error=0.04,
            section_max_screen_error=0.16,
            plane_patch_margin=0.16,
        )
        self._assert_same_frame_camera_sample(scene, controller, shot)

    def test_follow_starts_only_at_endpoint_and_changes_only_target(self) -> None:
        scene = _ParallelScene()
        shot = _shot(1)
        dynamic_target = np.asarray((1.1, -0.4, 0.7), dtype=float)
        controller = ParallelCameraTargetFollowController(
            scene,
            lambda: dynamic_target,
        )
        driver_id = id(controller.driver_mobject)
        with self.assertRaisesRegex(
            ParallelCameraShotManimError,
            "only after the shot endpoint",
        ):
            controller.start(shot)
        self.assertFalse(controller.active)
        self.assertNotIn(controller.driver_mobject, scene.mobjects)
        self.assertFalse(controller.driver_mobject.get_updaters())

        play_parallel_camera_shot(scene, shot)
        unrelated_producer = Mobject()

        def update_unrelated(_mobject: Mobject, dt: float) -> None:
            del dt

        unrelated_producer.add_updater(update_unrelated)
        first_consumer = Mobject()
        second_consumer = Mobject()
        for consumer in (first_consumer, second_consumer):
            consumer._tikz_native_parallel_camera_state_consumer = True
        scene.add(unrelated_producer, first_consumer, second_consumer)
        scene.renderer.static_image = np.ones((2, 2, 4), dtype=np.uint8)
        controller.start(shot)
        self.assertTrue(controller.active)
        self.assertIs(controller.endpoint_state, shot.state)
        self.assertEqual(controller.shot_id, shot.id)
        self.assertIn(controller.driver_mobject, scene.mobjects)
        self.assertTrue(controller.driver_mobject.get_updaters())
        self.assertEqual(id(controller.driver_mobject), driver_id)
        self.assertLess(
            scene.mobjects.index(unrelated_producer),
            scene.mobjects.index(controller.driver_mobject),
        )
        self.assertLess(
            scene.mobjects.index(controller.driver_mobject),
            scene.mobjects.index(first_consumer),
        )
        self.assertLess(
            scene.mobjects.index(first_consumer),
            scene.mobjects.index(second_consumer),
        )
        self.assertIsNone(scene.renderer.static_image)
        followed = scene.camera.snapshot_parallel_state()
        np.testing.assert_array_equal(followed.target, dynamic_target)
        np.testing.assert_array_equal(followed.matrix, shot.state.matrix)
        np.testing.assert_array_equal(
            followed.screen_anchor,
            shot.state.screen_anchor,
        )
        self.assertEqual(followed.zoom, shot.state.zoom)

        dynamic_target[:] = (-0.6, 0.9, 1.3)
        with patch.object(
            Mobject,
            "__init__",
            side_effect=AssertionError("target follow allocated a Mobject"),
        ):
            scene.update_mobjects(1.0 / 12.0)
        self.assertEqual(id(controller.driver_mobject), driver_id)
        followed = scene.camera.snapshot_parallel_state()
        np.testing.assert_array_equal(followed.target, dynamic_target)
        np.testing.assert_array_equal(followed.matrix, shot.state.matrix)
        np.testing.assert_array_equal(
            followed.screen_anchor,
            shot.state.screen_anchor,
        )
        self.assertEqual(followed.zoom, shot.state.zoom)

        scene.renderer.static_image = np.ones((2, 2, 4), dtype=np.uint8)
        controller.stop()
        self.assertFalse(controller.active)
        self.assertNotIn(controller.driver_mobject, scene.mobjects)
        self.assertFalse(controller.driver_mobject.get_updaters())
        self.assertIsNone(scene.renderer.static_image)
        stopped = scene.camera.snapshot_parallel_state()
        dynamic_target[:] = (2.0, 2.0, 2.0)
        controller.driver_mobject.update(1.0 / 12.0)
        _assert_state_exact(
            self,
            scene.camera.snapshot_parallel_state(),
            stopped,
        )

        scene.renderer.static_image = np.ones((2, 2, 4), dtype=np.uint8)
        controller.restore()
        self.assertFalse(controller.active)
        self.assertIsNone(controller.endpoint_state)
        self.assertIsNone(controller.shot_id)
        self.assertIs(scene.camera.snapshot_parallel_state(), shot.state)
        self.assertIsNone(scene.renderer.static_image)

        # Reusing the same preallocated driver must not allocate a new Mobject.
        with patch.object(
            Mobject,
            "__init__",
            side_effect=AssertionError("follow restart allocated a Mobject"),
        ):
            controller.start(shot).stop().restore()
        self.assertEqual(id(controller.driver_mobject), driver_id)

    def test_start_and_runtime_provider_failures_leave_no_attached_updater(
        self,
    ) -> None:
        scene = _ParallelScene()
        shot = _shot(0)
        scene.camera.set_parallel_state(shot.state)
        invalid = ParallelCameraTargetFollowController(
            scene,
            lambda: (float("nan"), 0.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "three finite values"):
            invalid.start(shot)
        self.assertFalse(invalid.active)
        self.assertNotIn(invalid.driver_mobject, scene.mobjects)
        self.assertFalse(invalid.driver_mobject.get_updaters())
        self.assertIs(scene.camera.snapshot_parallel_state(), shot.state)

        target_state: dict[str, object] = {
            "value": np.asarray((0.4, 0.2, -0.1), dtype=float)
        }
        controller = ParallelCameraTargetFollowController(
            scene,
            lambda: target_state["value"],
        )
        controller.start(shot)
        target_state["value"] = "invalid"
        with self.assertRaisesRegex(ValueError, "three finite values"):
            controller.driver_mobject.update(1.0 / 12.0)
        self.assertFalse(controller.active)
        self.assertNotIn(controller.driver_mobject, scene.mobjects)
        self.assertFalse(controller.driver_mobject.get_updaters())
        self.assertIs(scene.camera.snapshot_parallel_state(), shot.state)
        controller.restore()

    def test_scene_add_failure_rolls_back_endpoint_and_updater(self) -> None:
        scene = _ParallelScene()
        shot = _shot(2)
        scene.camera.set_parallel_state(shot.state)
        with patch.object(
            scene,
            "add",
            side_effect=RuntimeError("synthetic add failure"),
        ):
            controller = ParallelCameraTargetFollowController(
                scene,
                lambda: (0.2, 0.3, 0.4),
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic add failure"):
                controller.start(shot)
        self.assertFalse(controller.active)
        self.assertFalse(controller.driver_mobject.get_updaters())
        self.assertNotIn(controller.driver_mobject, scene.mobjects)
        self.assertIs(scene.camera.snapshot_parallel_state(), shot.state)

    def test_partial_start_failure_cleans_all_scene_lists_and_static_cache(
        self,
    ) -> None:
        scene = _ParallelScene()
        shot = _shot(1)
        scene.camera.set_parallel_state(shot.state)

        def add_twice(driver: Mobject) -> None:
            scene.mobjects.extend((driver, driver))
            scene.foreground_mobjects.append(driver)
            scene.moving_mobjects = [driver]
            scene.static_mobjects = [driver]

        with patch.object(scene, "add", side_effect=add_twice):
            controller = ParallelCameraTargetFollowController(
                scene,
                lambda: (0.2, 0.3, 0.4),
            )
            scene.renderer.static_image = np.ones((2, 2, 4), dtype=np.uint8)
            with self.assertRaisesRegex(
                ParallelCameraShotManimError,
                "Scene-owned exactly once",
            ):
                controller.start(shot)

        self.assertFalse(controller.active)
        self.assertFalse(controller.driver_mobject.get_updaters())
        for name in (
            "mobjects",
            "foreground_mobjects",
            "moving_mobjects",
            "static_mobjects",
        ):
            self.assertNotIn(controller.driver_mobject, getattr(scene, name))
        self.assertIsNone(scene.renderer.static_image)
        self.assertIsNone(controller.endpoint_state)
        self.assertIsNone(controller.shot_id)
        self.assertIs(scene.camera.snapshot_parallel_state(), shot.state)

    def test_restore_cleans_every_scene_list_when_scene_remove_raises(self) -> None:
        scene = _ParallelScene()
        shot = _shot(2)
        scene.camera.set_parallel_state(shot.state)
        with patch.object(
            scene,
            "remove",
            side_effect=RuntimeError("synthetic remove failure"),
        ):
            controller = ParallelCameraTargetFollowController(
                scene,
                lambda: (0.9, -0.4, 1.2),
            )
            controller.start(shot)
            driver = controller.driver_mobject
            scene.foreground_mobjects.append(driver)
            scene.moving_mobjects = [driver]
            scene.static_mobjects = [driver]
            scene.renderer.static_image = np.ones((2, 2, 4), dtype=np.uint8)
            with self.assertRaisesRegex(RuntimeError, "synthetic remove failure"):
                controller.restore()

        self.assertFalse(controller.active)
        self.assertFalse(driver.get_updaters())
        for name in (
            "mobjects",
            "foreground_mobjects",
            "moving_mobjects",
            "static_mobjects",
        ):
            self.assertNotIn(driver, getattr(scene, name))
        self.assertIsNone(scene.renderer.static_image)
        self.assertIsNone(controller.endpoint_state)
        self.assertIsNone(controller.shot_id)
        self.assertIs(scene.camera.snapshot_parallel_state(), shot.state)

    def test_restore_cleans_scene_when_remove_updater_raises(self) -> None:
        scene = _ParallelScene()
        shot = _shot(0)
        scene.camera.set_parallel_state(shot.state)
        controller = ParallelCameraTargetFollowController(
            scene,
            lambda: (0.6, -0.2, 0.8),
        )
        controller.start(shot)
        driver = controller.driver_mobject
        scene.renderer.static_image = np.ones((2, 2, 4), dtype=np.uint8)

        with patch.object(
            driver,
            "remove_updater",
            side_effect=RuntimeError("synthetic updater cleanup failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "synthetic updater cleanup failure",
            ):
                controller.restore()

        self.assertFalse(controller.active)
        self.assertNotIn(driver, scene.mobjects)
        self.assertIsNone(scene.renderer.static_image)
        self.assertIsNone(controller.endpoint_state)
        self.assertIsNone(controller.shot_id)
        self.assertIs(scene.camera.snapshot_parallel_state(), shot.state)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
