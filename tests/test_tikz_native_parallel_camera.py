from __future__ import annotations

import unittest

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import SectionPlane
from tikz_native.parallel_camera import (
    CameraPlane,
    ParallelCameraState,
    ProjectionRank,
    frame_from_view_direction,
    interpolate_parallel_camera_states,
    orbit_control_matrix,
)

FRONT_MATRIX = np.array(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
SIDE_MATRIX = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
TOP_MATRIX = np.identity(3)
ISOMETRIC_MATRIX = np.array(
    (
        (-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0),
        (-1.0 / np.sqrt(6.0), -1.0 / np.sqrt(6.0), 2.0 / np.sqrt(6.0)),
        (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
    )
)


class ParallelCameraStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = CameraPlane(
            point=np.array((1.0, -2.0, 0.5)),
            normal=np.array((0.0, 0.0, 1.0)),
            u_axis=np.array((1.0, 0.0, 0.0)),
        )

    def assert_camera_frame(self, matrix: np.ndarray) -> None:
        np.testing.assert_allclose(matrix @ matrix.T, np.identity(3), atol=1e-12)
        np.testing.assert_allclose(
            np.cross(matrix[0], matrix[1]), matrix[2], atol=1e-12
        )
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=12)

    def test_view_direction_constructor_reproduces_orthographic_presets(self) -> None:
        cases = (
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), FRONT_MATRIX),
            ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), SIDE_MATRIX),
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), TOP_MATRIX),
            ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0), ISOMETRIC_MATRIX),
        )
        for direction, up_hint, expected in cases:
            with self.subTest(direction=direction):
                state = ParallelCameraState.from_view_direction(
                    direction, up_hint=up_hint
                )
                np.testing.assert_allclose(state.matrix, expected, atol=1e-12)
                self.assert_camera_frame(state.matrix)
                np.testing.assert_allclose(
                    ParallelView.from_matrix(state.matrix).view_direction,
                    state.view_direction,
                    atol=1e-12,
                )

    def test_section_plane_is_accepted_structurally(self) -> None:
        plane = SectionPlane(
            "cut",
            tuple(self.plane.point),
            tuple(self.plane.normal),
            u_axis=tuple(self.plane.u_axis),
        )
        state = ParallelCameraState.along_plane(
            plane,
            azimuth_degrees=15.0,
        )
        self.assertIs(state.plane_projection_rank(plane), ProjectionRank.LINE)

    def test_target_maps_to_fixed_screen_anchor_independently_of_zoom(self) -> None:
        state = ParallelCameraState.from_view_direction(
            (1.0, 1.0, 1.0),
            target=(2.0, -1.5, 0.75),
            screen_anchor=(-2.25, 1.1),
            zoom=3.5,
        )
        projected = state.project_point(state.target)
        np.testing.assert_allclose(projected, (-2.25, 1.1, 0.0), atol=1e-12)
        moved = state.project_point(state.target + state.view_direction * 2.75)
        np.testing.assert_allclose(moved[:2], state.screen_anchor, atol=1e-12)
        self.assertAlmostEqual(float(moved[2]), 2.75, places=12)

    def test_normal_relative_and_along_plane_share_exact_endpoints(self) -> None:
        normal = ParallelCameraState.normal_to_plane(self.plane)
        relative_start = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=0.0,
            azimuth_degrees=37.0,
        )
        along = ParallelCameraState.along_plane(
            self.plane,
            azimuth_degrees=37.0,
        )
        relative_end = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=90.0,
            azimuth_degrees=37.0,
        )
        np.testing.assert_allclose(relative_start.matrix, normal.matrix, atol=1e-12)
        np.testing.assert_allclose(relative_end.matrix, along.matrix, atol=1e-12)
        np.testing.assert_allclose(normal.target, self.plane.point, atol=1e-12)
        self.assert_camera_frame(normal.matrix)
        self.assert_camera_frame(along.matrix)

    def test_explicit_along_plane_direction_matches_azimuth(self) -> None:
        azimuth = 28.0
        angle = np.radians(azimuth)
        direction = (
            np.cos(angle) * self.plane.u_axis + np.sin(angle) * self.plane.v_axis
        )
        explicit = ParallelCameraState.along_plane(
            self.plane,
            direction=direction,
        )
        angular = ParallelCameraState.along_plane(
            self.plane,
            azimuth_degrees=azimuth,
        )
        np.testing.assert_allclose(explicit.matrix, angular.matrix, atol=1e-12)

    def test_plane_projection_rank_is_area_then_exact_line(self) -> None:
        oblique = ParallelCameraState.relative_to_plane(
            self.plane,
            inclination_degrees=63.0,
            azimuth_degrees=21.0,
        )
        edge_on = ParallelCameraState.along_plane(
            self.plane,
            azimuth_degrees=21.0,
        )
        self.assertIs(oblique.plane_projection_rank(self.plane), ProjectionRank.AREA)
        self.assertIs(edge_on.plane_projection_rank(self.plane), ProjectionRank.LINE)
        determinant = float(np.linalg.det(oblique.plane_screen_basis(self.plane)))
        self.assertAlmostEqual(determinant, np.cos(np.radians(63.0)), places=12)
        viewing_delta = (
            np.cos(np.radians(21.0)) * self.plane.u_axis
            + np.sin(np.radians(21.0)) * self.plane.v_axis
        )
        np.testing.assert_allclose(
            edge_on.project_point(self.plane.point + viewing_delta)[:2],
            edge_on.screen_anchor,
            atol=1e-12,
        )

    def test_negative_side_and_roll_keep_right_handed_depth(self) -> None:
        negative = ParallelCameraState.normal_to_plane(
            self.plane,
            side="negative",
            roll_degrees=90.0,
        )
        np.testing.assert_allclose(negative.view_direction, -self.plane.normal)
        self.assert_camera_frame(negative.matrix)
        unrolled = ParallelCameraState.normal_to_plane(self.plane, side="negative")
        np.testing.assert_allclose(negative.matrix[0], -unrolled.matrix[1], atol=1e-12)
        np.testing.assert_allclose(negative.matrix[1], unrolled.matrix[0], atol=1e-12)

    def test_camera_state_owns_read_only_copies(self) -> None:
        matrix = np.identity(3)
        target = np.ones(3)
        state = ParallelCameraState(matrix, target)
        matrix[0, 0] = 4.0
        target[0] = 8.0
        np.testing.assert_allclose(state.matrix, np.identity(3))
        np.testing.assert_allclose(state.target, np.ones(3))
        with self.assertRaises(ValueError):
            state.matrix[0, 0] = 2.0

    def test_camera_matrix_validation_is_scale_invariant(self) -> None:
        for scale in (1.0e-20, 1.0e150):
            with self.subTest(scale=scale):
                matrix = np.diag((scale, scale, 1.0))
                state = ParallelCameraState(matrix)
                np.testing.assert_allclose(state.matrix, matrix)
                np.testing.assert_allclose(state.view_direction, (0.0, 0.0, 1.0))
                self.assertTrue(
                    np.all(np.isfinite(state.project_point((1.0, 1.0, 1.0))))
                )

    def test_invalid_camera_and_plane_inputs_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "view_direction must be non-zero"):
            ParallelCameraState.from_view_direction((0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "up_hint must not be parallel"):
            ParallelCameraState.from_view_direction(
                (0.0, 0.0, 1.0), up_hint=(0.0, 0.0, 2.0)
            )
        with self.assertRaisesRegex(ValueError, "right-handed"):
            ParallelCameraState(np.diag((1.0, 1.0, -1.0)))
        with self.assertRaisesRegex(ValueError, "inclination_degrees"):
            ParallelCameraState.relative_to_plane(self.plane, inclination_degrees=90.1)
        with self.assertRaisesRegex(ValueError, "must lie"):
            ParallelCameraState.along_plane(self.plane, direction=(1.0, 0.0, 0.01))

    def test_safe_interpolation_preserves_invertibility_and_endpoints(self) -> None:
        source = ParallelCameraState(
            np.array(
                (
                    (-0.35, 1.0, 0.0),
                    (-0.35, 0.0, 1.0),
                    (0.9428090416, 0.3333333333, 0.3333333333),
                )
            ),
            target=np.array((0.0, 0.0, 0.0)),
            screen_anchor=np.array((0.0, 0.0)),
            zoom=1.0,
        )
        target = ParallelCameraState.normal_to_plane(
            self.plane,
            target=(2.0, 3.0, 4.0),
            screen_anchor=(-1.0, 0.5),
            zoom=2.0,
        )
        control = orbit_control_matrix(source.matrix, target.matrix)
        self.assertIs(
            interpolate_parallel_camera_states(
                source, target, 0.0, control_matrix=control
            ),
            source,
        )
        self.assertIs(
            interpolate_parallel_camera_states(
                source, target, 1.0, control_matrix=control
            ),
            target,
        )
        for alpha in np.linspace(0.05, 0.95, 19):
            state = interpolate_parallel_camera_states(
                source, target, float(alpha), control_matrix=control
            )
            self.assertGreater(float(np.linalg.det(state.matrix)), 1.0e-9)
            self.assertTrue(np.all(np.isfinite(state.matrix)))
            self.assertGreater(state.zoom, 0.0)

    def test_identical_view_interpolates_a_stable_middle_frame(self) -> None:
        source = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
        middle = interpolate_parallel_camera_states(source, source, 0.5)

        np.testing.assert_allclose(middle.matrix, source.matrix, atol=1.0e-12)
        self.assert_camera_frame(middle.matrix)

    def test_opposite_views_require_an_explicit_orbit_control(self) -> None:
        source = ParallelCameraState.from_view_direction(
            (0.0, 0.0, 1.0), up_hint=(0.0, 1.0, 0.0)
        )
        target = ParallelCameraState.from_view_direction(
            (0.0, 0.0, -1.0), up_hint=(0.0, 1.0, 0.0)
        )
        with self.assertRaisesRegex(ValueError, "180-degree"):
            interpolate_parallel_camera_states(source, target, 0.5)
        control = orbit_control_matrix(source.matrix, target.matrix)
        middle = interpolate_parallel_camera_states(
            source, target, 0.5, control_matrix=control
        )
        self.assertGreater(float(np.linalg.det(middle.matrix)), 0.99)

    def test_frame_constructor_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            frame_from_view_direction((np.nan, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
