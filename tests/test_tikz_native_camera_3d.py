from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
from manim import Mobject

from tikz_native.camera_3d import (
    FRONT_MATRIX,
    MultiProjectionCamera,
    ParallelCameraTransactionSnapshot,
)
from tikz_native.parallel_camera import CameraPlane, ParallelCameraState, ProjectionRank
from tikz_native.parallel_frame import (
    ParallelFrameCoordinator,
    ParallelFrameParticipant,
    ParallelFramePhase,
    ParallelFrameState,
    parallel_camera_frame_participant,
)
from tikz_native.parallel_viewport import (
    ParallelViewportState,
    parallel_viewport_frame_participant,
)


class MultiProjectionParallelStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = CameraPlane(
            point=np.array((1.25, -0.5, 0.75)),
            normal=np.array((1.0, 1.0, 1.0)),
            u_axis=np.array((1.0, -1.0, 0.0)),
        )

    def assert_transaction_equal(
        self,
        actual: ParallelCameraTransactionSnapshot,
        expected: ParallelCameraTransactionSnapshot,
    ) -> None:
        for name in (
            "source_matrix",
            "target_matrix",
            "control_matrix",
            "source_view_center",
            "target_view_center",
            "source_principal_point",
            "target_principal_point",
            "rotation_matrix",
            "frame_center",
        ):
            np.testing.assert_array_equal(
                getattr(actual, name),
                getattr(expected, name),
                err_msg=name,
            )
        if expected.parallel_control_matrix is None:
            self.assertIsNone(actual.parallel_control_matrix)
        else:
            self.assertIsNotNone(actual.parallel_control_matrix)
            np.testing.assert_array_equal(
                actual.parallel_control_matrix,
                expected.parallel_control_matrix,
                err_msg="parallel_control_matrix",
            )
        for name in (
            "source_perspective",
            "target_perspective",
            "source_focal_distance",
            "target_focal_distance",
            "transition_style",
            "parallel_state_active",
            "parallel_state_cache_alpha",
            "transition_progress",
            "current_mode",
            "target_mode",
            "phi",
            "theta",
            "manim_focal_distance",
            "gamma",
            "manim_zoom",
        ):
            self.assertEqual(getattr(actual, name), getattr(expected, name), name)
        self.assertIs(actual.source_parallel_state, expected.source_parallel_state)
        self.assertIs(actual.target_parallel_state, expected.target_parallel_state)
        self.assertIs(actual.parallel_state_cache, expected.parallel_state_cache)

    def test_semantic_state_keeps_target_on_final_anchor_under_manim_zoom(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        camera.set_zoom(1.6)
        state = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=48.0,
            azimuth_degrees=32.0,
            target=(2.0, -1.0, 0.5),
            screen_anchor=(-2.4, 1.2),
            zoom=1.35,
        )
        camera.set_parallel_state(state)
        points = np.vstack(
            (
                state.target,
                state.target + state.matrix[0],
                state.target + state.view_direction * 2.0,
            )
        )
        projected = camera.project_points(points)
        np.testing.assert_allclose(projected[0, :2], state.screen_anchor, atol=1e-12)
        np.testing.assert_allclose(projected[2, :2], state.screen_anchor, atol=1e-12)
        self.assertAlmostEqual(float(projected[2, 2]), 2.0, places=12)
        self.assertAlmostEqual(
            float(projected[1, 0] - state.screen_anchor[0]),
            1.6 * state.zoom,
            places=12,
        )

    def test_final_anchor_survives_nonzero_manim_frame_center(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        camera.frame_center = np.array((2.0, -1.0, 0.25))
        camera.set_zoom(1.3)
        state = ParallelCameraState.normal_to_plane(
            self.plane,
            target=(0.6, -0.4, 1.1),
            screen_anchor=(-1.5, 0.75),
            zoom=1.1,
        )
        camera.set_parallel_state(state)
        intermediate = camera.project_points(np.asarray((state.target,)))
        np.testing.assert_allclose(
            intermediate[0, :2],
            camera.frame_center[:2] + state.screen_anchor,
            atol=1e-12,
        )
        pixels = camera.points_to_pixel_coords(Mobject(), np.asarray((state.target,)))
        expected = np.array(
            (
                state.screen_anchor[0] * camera.pixel_width / camera.frame_width
                + camera.pixel_width / 2,
                -state.screen_anchor[1] * camera.pixel_height / camera.frame_height
                + camera.pixel_height / 2,
            )
        ).astype(int)
        np.testing.assert_array_equal(pixels[0], expected)

    def test_parallel_frame_center_xy_setter_is_bit_exact_and_preserves_z(
        self,
    ) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        camera._frame_center.points = np.asarray(
            ((-5.0, 0.125, 7.25),),
            dtype=float,
        )

        for target in (
            (-1.8, 0.3),
            (3.78112217016424, -2.625),
            (-4.293676746462265, 1.1),
        ):
            camera.set_parallel_frame_center_xy(target)
            np.testing.assert_array_equal(
                camera.frame_center,
                np.asarray((target[0], target[1], 7.25), dtype=float),
            )

        for invalid in (
            (0.0,),
            (0.0, 1.0, 2.0),
            (float("nan"), 0.0),
            (0.0, float("inf")),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "two finite values"):
                    camera.set_parallel_frame_center_xy(invalid)

    def test_viewport_participant_uses_exact_real_camera_center_capability(
        self,
    ) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        camera._frame_center.points = np.asarray(((-5.0, 0.125, 7.25),))
        display_offset = [(0.0, 0.0)]
        coordinator: ParallelFrameCoordinator[ParallelViewportState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(
            parallel_viewport_frame_participant(
                camera,
                display_offset_getter=lambda: display_offset[0],
                display_offset_setter=lambda value: display_offset.__setitem__(
                    0,
                    tuple(value),
                ),
            )
        )
        target = ParallelViewportState.from_components(
            ParallelCameraState.normal_to_plane(self.plane),
            inherited_zoom=1.4,
            frame_center=(-1.8, 0.3),
            display_offset=(0.6, -0.2),
        )

        coordinator.update(target)

        np.testing.assert_array_equal(
            camera.frame_center,
            np.asarray((-1.8, 0.3, 7.25)),
        )
        self.assertEqual(camera.get_zoom(), 1.4)
        self.assertEqual(display_offset[0], (0.6, -0.2))
        coordinator.restore()
        np.testing.assert_array_equal(
            camera.frame_center,
            np.asarray((-5.0, 0.125, 7.25)),
        )
        self.assertEqual(display_offset[0], (0.0, 0.0))

    def test_snapshot_restore_bridges_semantic_and_legacy_states_exactly(self) -> None:
        camera = MultiProjectionCamera(initial_mode="oblique")
        camera.frame_center = np.array((0.7, -0.35, 0.2))
        camera.set_zoom(1.4)
        state = ParallelCameraState.normal_to_plane(
            self.plane,
            target=(0.5, 0.25, -0.75),
            screen_anchor=(1.75, -0.6),
            zoom=0.8,
        )
        camera.set_parallel_state(state)
        points = np.array(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-2.0, 0.5, 1.0)))
        expected = camera.project_points(points)
        semantic_snapshot = camera.snapshot_parallel_state()
        legacy_snapshot = camera.snapshot()
        camera.set_mode("front")
        camera.restore(legacy_snapshot)
        np.testing.assert_allclose(camera.project_points(points), expected, atol=1e-12)
        np.testing.assert_allclose(semantic_snapshot.matrix, state.matrix, atol=1e-12)
        np.testing.assert_allclose(semantic_snapshot.target, state.target, atol=1e-12)
        np.testing.assert_allclose(
            semantic_snapshot.screen_anchor, state.screen_anchor, atol=1e-12
        )
        self.assertAlmostEqual(semantic_snapshot.zoom, state.zoom, places=12)

    def test_legacy_snapshot_converts_to_final_anchor_semantics(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        camera.frame_center = np.array((1.2, -0.8, 0.4))
        camera.set_zoom(1.5)
        camera.register_mode(
            "legacy-offset",
            FRONT_MATRIX,
            view_center=(0.5, 0.25, -0.1),
            principal_point=(-0.4, 0.3),
        )
        camera.set_mode("legacy-offset")
        state = camera.snapshot_parallel_state()
        np.testing.assert_allclose(
            state.target,
            camera.frame_center + np.array((0.5, 0.25, -0.1)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            state.screen_anchor,
            1.5 * np.array((-0.4, 0.3)) - camera.frame_center[:2],
            atol=1e-12,
        )

    def test_safe_transition_from_oblique_stays_invertible_and_hits_endpoint(
        self,
    ) -> None:
        camera = MultiProjectionCamera(initial_mode="oblique")
        target = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=72.0,
            azimuth_degrees=-18.0,
            target=self.plane.point,
            screen_anchor=(-1.0, 0.4),
            zoom=1.2,
        )
        camera.animate_to_parallel_state(target, transition="orbit", arc_height=0.55)
        for alpha in np.linspace(0.0, 1.0, 41):
            camera.transition_tracker.set_value(float(alpha))
            matrix = camera.get_projection_matrix()
            self.assertTrue(np.all(np.isfinite(matrix)))
            self.assertGreater(float(np.linalg.det(matrix)), 1.0e-9)
        final = camera.snapshot_parallel_state()
        np.testing.assert_allclose(final.matrix, target.matrix, atol=1e-12)
        np.testing.assert_allclose(final.target, target.target, atol=1e-12)
        np.testing.assert_allclose(
            final.screen_anchor, target.screen_anchor, atol=1e-12
        )
        self.assertAlmostEqual(final.zoom, target.zoom, places=12)

    def test_exact_along_plane_state_is_valid_for_camera_and_rank_one_for_plane(
        self,
    ) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        state = ParallelCameraState.along_plane(
            self.plane,
            azimuth_degrees=26.0,
        )
        self.assertIs(state.plane_projection_rank(self.plane), ProjectionRank.LINE)
        camera.set_parallel_state(state)
        self.assertGreater(float(np.linalg.det(camera.get_projection_matrix())), 0.99)
        np.testing.assert_allclose(
            camera.snapshot_parallel_state().view_direction,
            state.view_direction,
            atol=1e-12,
        )

    def test_registered_states_and_direct_states_use_the_same_path(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        state = ParallelCameraState.normal_to_plane(self.plane)
        camera.register_parallel_state("plane-front", state)
        camera.set_parallel_state("plane-front")
        np.testing.assert_allclose(camera.get_projection_matrix(), state.matrix)
        with self.assertRaisesRegex(KeyError, "already exists"):
            camera.register_parallel_state("front", state)

    def test_full_transaction_restores_legacy_interpolation_and_manim_trackers(
        self,
    ) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        camera.frame_center = np.array((0.7, -0.3, 0.2))
        camera.phi_tracker.set_value(0.25)
        camera.theta_tracker.set_value(-0.8)
        camera.focal_distance_tracker.set_value(11.5)
        camera.gamma_tracker.set_value(-0.15)
        camera.zoom_tracker.set_value(1.45)
        camera.register_mode(
            "perspective-target",
            np.identity(3),
            perspective_strength=0.6,
            focal_distance=13.0,
            view_center=(1.0, -0.5, 0.25),
            principal_point=(-0.4, 0.3),
        )
        camera.animate_to("perspective-target")
        camera.transition_tracker.set_value(0.37)
        camera.rotation_matrix = np.array(
            ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
        )
        expected = camera.snapshot_parallel_transaction()

        camera.set_mode("top")
        camera.frame_center = np.array((-2.0, 1.0, 4.0))
        camera.phi_tracker.set_value(-1.0)
        camera.theta_tracker.set_value(1.0)
        camera.focal_distance_tracker.set_value(3.0)
        camera.gamma_tracker.set_value(0.75)
        camera.zoom_tracker.set_value(0.6)
        camera.rotation_matrix = np.zeros((3, 3))
        camera.restore_parallel_transaction(expected)

        self.assert_transaction_equal(
            camera.snapshot_parallel_transaction(),
            expected,
        )
        self.assertEqual(camera.current_mode, "perspective-target")
        self.assertEqual(camera.target_mode, "perspective-target")
        self.assertNotEqual(camera.current_mode, camera.DIRECT_MODE_NAME)

    def test_full_transaction_restores_semantic_orbit_control_and_cache(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        source = ParallelCameraState.normal_to_plane(
            self.plane,
            target=(0.5, -0.25, 1.0),
            screen_anchor=(-1.5, 0.75),
            zoom=0.9,
        )
        target = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=68.0,
            azimuth_degrees=21.0,
            target=(-0.25, 0.5, 0.75),
            screen_anchor=(0.4, -0.2),
            zoom=1.2,
        )
        camera.register_parallel_state("source-plane", source)
        camera.register_parallel_state("target-plane", target)
        camera.set_parallel_state("source-plane")
        camera.animate_to_parallel_state(
            "target-plane",
            transition="orbit",
            arc_height=0.55,
        )
        camera.transition_tracker.set_value(0.41)
        camera.get_projection_matrix()
        expected = camera.snapshot_parallel_transaction()
        self.assertIsNotNone(expected.parallel_control_matrix)
        self.assertIsNotNone(expected.parallel_state_cache)

        camera.set_parallel_state(ParallelCameraState(np.identity(3)))
        camera.restore_parallel_transaction(expected)

        self.assert_transaction_equal(
            camera.snapshot_parallel_transaction(),
            expected,
        )
        self.assertEqual(camera.current_mode, "target-plane")
        self.assertEqual(camera.target_mode, "target-plane")
        self.assertNotEqual(camera.current_mode, camera.DIRECT_MODE_NAME)

    def test_coordinator_failure_rolls_back_registered_camera_mode(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        source = ParallelCameraState.normal_to_plane(self.plane)
        target = ParallelCameraState.along_plane(
            self.plane,
            azimuth_degrees=24.0,
        )
        camera.register_parallel_state("source-plane", source)
        camera.set_parallel_state("source-plane")
        expected = camera.snapshot_parallel_transaction()

        def fail_commit(_prepared: object) -> None:
            raise RuntimeError("injected finalize failure")

        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(parallel_camera_frame_participant(camera))
        coordinator.add(
            ParallelFrameParticipant(
                participant_id="failing-finalizer",
                phase=ParallelFramePhase.FINALIZE,
                prepare=lambda _frame: None,
                snapshot=lambda: None,
                commit=fail_commit,
                rollback=lambda _snapshot: None,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "injected finalize failure"):
            coordinator.update(ParallelFrameState(target))

        self.assert_transaction_equal(
            camera.snapshot_parallel_transaction(),
            expected,
        )
        self.assertEqual(camera.current_mode, "source-plane")
        self.assertEqual(camera.target_mode, "source-plane")
        self.assertNotEqual(camera.current_mode, camera.DIRECT_MODE_NAME)

    def test_coordinator_restore_returns_to_registered_camera_mode(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        source = ParallelCameraState.normal_to_plane(self.plane)
        target = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=55.0,
            azimuth_degrees=-18.0,
        )
        camera.register_parallel_state("source-plane", source)
        camera.set_parallel_state("source-plane")
        expected = camera.snapshot_parallel_transaction()
        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(parallel_camera_frame_participant(camera))

        coordinator.update(ParallelFrameState(target))
        self.assertEqual(camera.current_mode, camera.DIRECT_MODE_NAME)
        self.assertEqual(camera.target_mode, camera.DIRECT_MODE_NAME)
        coordinator.restore()

        self.assert_transaction_equal(
            camera.snapshot_parallel_transaction(),
            expected,
        )
        self.assertEqual(camera.current_mode, "source-plane")
        self.assertEqual(camera.target_mode, "source-plane")
        self.assertNotEqual(camera.current_mode, camera.DIRECT_MODE_NAME)

    def test_transaction_rejects_foreign_owner_before_any_write(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        state = ParallelCameraState.normal_to_plane(self.plane)
        camera.register_parallel_state("owned-plane", state)
        camera.set_parallel_state("owned-plane")
        expected = camera.snapshot_parallel_transaction()
        foreign = MultiProjectionCamera(
            initial_mode="top"
        ).snapshot_parallel_transaction()

        with self.assertRaisesRegex(ValueError, "foreign owner"):
            camera.restore_parallel_transaction(foreign)

        self.assert_transaction_equal(
            camera.snapshot_parallel_transaction(),
            expected,
        )

    def test_malformed_transaction_snapshots_fail_before_any_write(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        state = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=38.0,
            azimuth_degrees=17.0,
        )
        camera.register_parallel_state("owned-plane", state)
        camera.set_parallel_state("owned-plane")
        expected = camera.snapshot_parallel_transaction()
        nan_rotation = expected.rotation_matrix.copy()
        nan_rotation[1, 2] = np.nan
        singular = np.diag((1.0, 1.0, 0.0))
        wrong_cache = ParallelCameraState(
            np.identity(3),
            target=(99.0, -20.0, 7.0),
        )

        malformed = (
            replace(expected, source_matrix=np.zeros((2, 2))),
            replace(expected, rotation_matrix=nan_rotation),
            replace(expected, source_matrix=singular),
            replace(expected, target_matrix=singular),
            replace(expected, control_matrix=singular),
            replace(expected, rotation_matrix=singular),
            replace(expected, parallel_control_matrix=np.zeros((2, 3))),
            replace(expected, parallel_control_matrix=singular),
            replace(expected, source_view_center=np.zeros(2)),
            replace(expected, target_principal_point=np.array((0.0, np.inf))),
            replace(expected, source_perspective=1.1),
            replace(expected, target_focal_distance=0.0),
            replace(expected, manim_focal_distance=np.inf),
            replace(expected, manim_zoom=-1.0),
            replace(expected, transition_progress=np.nan),
            replace(expected, transition_style="unsafe"),
            replace(expected, source_parallel_state=None),
            replace(expected, parallel_state_cache=None),
            replace(expected, parallel_state_cache_alpha=1.1),
            replace(expected, parallel_state_cache=wrong_cache),
            replace(expected, current_mode="unregistered-mode"),
        )
        for index, forged in enumerate(malformed):
            with self.subTest(index=index):
                with self.assertRaises((TypeError, ValueError)):
                    camera.restore_parallel_transaction(forged)
                self.assert_transaction_equal(
                    camera.snapshot_parallel_transaction(),
                    expected,
                )

    def test_valid_stale_cache_is_checked_at_its_own_alpha(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        source = ParallelCameraState.normal_to_plane(self.plane)
        target = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=61.0,
            azimuth_degrees=-27.0,
            target=(0.5, -0.25, 1.0),
            screen_anchor=(-0.4, 0.3),
            zoom=1.15,
        )
        camera.register_parallel_state("source-plane", source)
        camera.register_parallel_state("target-plane", target)
        camera.set_parallel_state("source-plane")
        camera.animate_to_parallel_state("target-plane", transition="orbit")
        camera.transition_tracker.set_value(0.23)
        camera.get_projection_matrix()
        camera.transition_tracker.set_value(0.67)
        stale = camera.snapshot_parallel_transaction()
        self.assertEqual(stale.parallel_state_cache_alpha, 0.23)
        self.assertEqual(stale.transition_progress, 0.67)

        camera.set_mode("top")
        camera.restore_parallel_transaction(stale)

        self.assert_transaction_equal(
            camera.snapshot_parallel_transaction(),
            stale,
        )

    def test_unexpected_apply_failure_atomically_restores_previous_state(
        self,
    ) -> None:
        class FailingApplyCamera(MultiProjectionCamera):
            fail_next_apply = False

            def _apply_parallel_transaction_snapshot_unchecked(
                self,
                snapshot: ParallelCameraTransactionSnapshot,
            ) -> None:
                super()._apply_parallel_transaction_snapshot_unchecked(snapshot)
                if self.fail_next_apply:
                    self.fail_next_apply = False
                    raise RuntimeError("injected transaction apply failure")

        camera = FailingApplyCamera(initial_mode="front")
        state = ParallelCameraState.normal_to_plane(self.plane)
        camera.register_parallel_state("owned-plane", state)
        camera.set_parallel_state("owned-plane")
        expected = camera.snapshot_parallel_transaction()
        camera.set_mode("top")
        target = camera.snapshot_parallel_transaction()
        camera.restore_parallel_transaction(expected)

        camera.fail_next_apply = True
        with self.assertRaisesRegex(RuntimeError, "injected transaction apply failure"):
            camera.restore_parallel_transaction(target)

        self.assert_transaction_equal(
            camera.snapshot_parallel_transaction(),
            expected,
        )
        self.assertEqual(camera.current_mode, "owned-plane")
        self.assertEqual(camera.target_mode, "owned-plane")

    def test_shortest_opposite_transition_fails_before_animation(self) -> None:
        camera = MultiProjectionCamera(initial_mode="top")
        opposite = ParallelCameraState.from_view_direction(
            (0.0, 0.0, -1.0), up_hint=(0.0, 1.0, 0.0)
        )
        with self.assertRaisesRegex(ValueError, "180-degree"):
            camera.animate_to_parallel_state(opposite, transition="shortest")
        camera.animate_to_parallel_state(opposite, transition="orbit")
        camera.transition_tracker.set_value(0.5)
        self.assertGreater(float(np.linalg.det(camera.get_projection_matrix())), 0.99)

    def test_legacy_linear_transition_can_start_without_a_visual_jump(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        camera.set_zoom(1.25)
        state = ParallelCameraState.normal_to_plane(
            self.plane,
            target=(0.5, -0.25, 1.0),
            screen_anchor=(-1.5, 0.75),
            zoom=0.9,
        )
        camera.set_parallel_state(state)
        points = np.array(((0.0, 0.0, 0.0), (2.0, 1.0, -1.0)))
        before = camera.project_points(points)
        camera.animate_to("front")
        np.testing.assert_allclose(camera.project_points(points), before, atol=1e-12)
        camera.transition_tracker.set_value(1.0)
        np.testing.assert_allclose(
            camera.get_projection_matrix(), FRONT_MATRIX, atol=1e-12
        )

    def test_legacy_orbit_can_leave_a_scaled_semantic_state_safely(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        camera.frame_center = np.array((0.6, -0.2, 0.1))
        camera.set_zoom(1.3)
        state = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=42.0,
            azimuth_degrees=17.0,
            target=(0.5, -0.25, 1.0),
            screen_anchor=(-1.5, 0.75),
            zoom=0.75,
        )
        camera.set_parallel_state(state)
        camera.animate_orbit_to("front", arc_height=0.6)
        for alpha in np.linspace(0.0, 1.0, 31):
            camera.transition_tracker.set_value(float(alpha))
            self.assertGreater(
                float(np.linalg.det(camera.get_projection_matrix())), 1.0e-9
            )
        points = np.array(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))
        expected = MultiProjectionCamera(initial_mode="front")
        expected.frame_center = camera.frame_center.copy()
        expected.set_zoom(1.3)
        np.testing.assert_allclose(
            camera.project_points(points),
            expected.project_points(points),
            atol=1e-12,
        )

    def test_scale_invariant_legacy_presets_bridge_in_both_directions(self) -> None:
        semantic = ParallelCameraState.from_view_direction(
            (1.0, -1.0, 0.8),
            up_hint=(0.0, 0.0, 1.0),
        )
        for scale in (1.0e-20, 1.0e150):
            with self.subTest(scale=scale):
                matrix = np.diag((scale, scale, 1.0))
                camera = MultiProjectionCamera(initial_mode="front")
                camera.register_mode("scaled", matrix)
                camera.set_mode("scaled")
                bridged = camera.snapshot_parallel_state()
                np.testing.assert_allclose(bridged.matrix, np.identity(3))
                self.assertTrue(np.isclose(bridged.zoom, scale, rtol=1.0e-12))

                camera.animate_to_parallel_state(semantic, transition="orbit")
                camera.transition_tracker.set_value(0.5)
                self.assertTrue(np.all(np.isfinite(camera.get_projection_matrix())))
                camera.transition_tracker.set_value(1.0)
                np.testing.assert_allclose(
                    camera.get_projection_matrix(), semantic.matrix
                )

                camera.animate_orbit_to("scaled")
                camera.transition_tracker.set_value(0.5)
                self.assertTrue(np.all(np.isfinite(camera.get_projection_matrix())))
                camera.transition_tracker.set_value(1.0)
                np.testing.assert_allclose(
                    camera.snapshot().matrix,
                    matrix,
                    rtol=1.0e-12,
                    atol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
