from __future__ import annotations

import unittest

import numpy as np
from manim import Mobject

from tikz_native.camera_3d import FRONT_MATRIX, MultiProjectionCamera
from tikz_native.parallel_camera import CameraPlane, ParallelCameraState, ProjectionRank


class MultiProjectionParallelStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = CameraPlane(
            point=np.array((1.25, -0.5, 0.75)),
            normal=np.array((1.0, 1.0, 1.0)),
            u_axis=np.array((1.0, -1.0, 0.0)),
        )

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
