from __future__ import annotations

from dataclasses import dataclass
from math import pi
import unittest
from unittest.mock import patch

import numpy as np
from manim import Mobject, ThreeDScene, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
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
    QuadricManimError,
    QuadricManimLimits,
    QuadricOcclusion3D,
)
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import ParallelCameraState


IDENTITY_MATRIX = np.identity(3)
SIDE_MATRIX = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=float,
)


class _ParallelCameraScene(ThreeDScene):
    def __init__(self) -> None:
        super().__init__(camera_class=MultiProjectionCamera)


@dataclass
class _InvalidSemanticState:
    matrix: np.ndarray
    target: np.ndarray
    screen_anchor: np.ndarray
    zoom: float


def _limits(**overrides: object) -> QuadricManimLimits:
    values: dict[str, object] = {
        "max_surfaces": 2,
        "max_curves": 8,
        "max_fragments_per_curve": 12,
        "max_segments_per_fragment": 256,
        "max_surface_segments": 512,
        "max_dashes_per_fragment": 48,
        "max_projected_length": 24.0,
        "max_total_mobjects": 12000,
        "max_boundary_sources": 32,
    }
    values.update(overrides)
    return QuadricManimLimits(**values)  # type: ignore[arg-type]


def _sphere_controller(
    scene: object,
    *,
    projection: object,
    display_offset: tuple[float, float] = (0.0, 0.0),
) -> QuadricOcclusion3D:
    return QuadricOcclusion3D(
        scene,
        surfaces=(SphereSpec("camera-sphere", (1.4, -0.8, 0.6), 1.1),),
        curves=(),
        projection=projection,
        limits=_limits(max_curves=1),
        max_chord_error=0.015,
        display_offset=display_offset,
    )


def _composite_controller(
    scene: object,
    *,
    projection: object,
    display_offset: tuple[float, float] = (0.0, 0.0),
) -> CompositeQuadricSection3D:
    cone = ConeSpec(
        "camera-double-cone",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (-2.0, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.OPEN_DOUBLE,
    )
    plane = SectionPlane(
        "camera-cut",
        (0.0, 0.5, 0.0),
        (0.0, 1.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    return CompositeQuadricSection3D(
        scene,
        surface=cone,
        section_id="camera-section",
        plane=plane,
        projection=projection,
        limits=_limits(max_total_mobjects=20000),
        max_chord_error=0.015,
        display_offset=display_offset,
    )


def _configure_camera(
    scene: _ParallelCameraScene,
    *,
    frame_center: tuple[float, float, float],
    zoom: float,
) -> MultiProjectionCamera:
    camera = scene.camera
    assert isinstance(camera, MultiProjectionCamera)
    camera.frame_center = np.asarray(frame_center, dtype=float)
    camera.set_zoom(zoom)
    return camera


def _effective_legacy_inputs(
    state: ParallelCameraState,
    *,
    inherited_zoom: float,
    frame_center: tuple[float, float, float],
    display_offset: tuple[float, float],
) -> tuple[ParallelView, tuple[float, float]]:
    matrix = np.array(state.matrix, dtype=float, copy=True)
    matrix[:2] *= state.zoom * inherited_zoom
    offset = (
        np.asarray(state.screen_anchor, dtype=float)
        - matrix[:2] @ np.asarray(state.target, dtype=float)
        + np.asarray(frame_center[:2], dtype=float)
        + np.asarray(display_offset, dtype=float)
    )
    return ParallelView.from_matrix(matrix), (float(offset[0]), float(offset[1]))


def _quadric_surface_points(controller: QuadricOcclusion3D) -> np.ndarray:
    prepared = controller._last_prepared_frame
    assert prepared is not None
    return prepared.numeric.surfaces[0].points.copy()


class QuadricParallelCameraStateTests(unittest.TestCase):
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

    def test_legacy_matrix_and_parallel_view_remain_equivalent(self) -> None:
        matrix = np.asarray(
            (
                (0.8, -0.6, 0.0),
                (0.3, 0.4, -0.8660254037844386),
                (0.5196152422706632, 0.6928203230275509, 0.5),
            ),
            dtype=float,
        )
        authored_offset = (0.35, -0.2)
        matrix_controller = _sphere_controller(
            _ParallelCameraScene(),
            projection=matrix,
            display_offset=authored_offset,
        ).attach()
        view_controller = _sphere_controller(
            _ParallelCameraScene(),
            projection=ParallelView.from_matrix(matrix),
            display_offset=authored_offset,
        ).attach()
        try:
            matrix_inputs = matrix_controller._resolve_frame_inputs()
            view_inputs = view_controller._resolve_frame_inputs()
            np.testing.assert_array_equal(matrix_inputs.view.matrix, matrix)
            np.testing.assert_array_equal(view_inputs.view.matrix, matrix)
            self.assertEqual(matrix_inputs.display_offset, authored_offset)
            self.assertEqual(view_inputs.display_offset, authored_offset)
            np.testing.assert_array_equal(
                _quadric_surface_points(matrix_controller),
                _quadric_surface_points(view_controller),
            )
        finally:
            matrix_controller.restore()
            view_controller.restore()

    def test_semantic_state_matches_complete_affine_camera_formula(self) -> None:
        frame_center = (1.2, -0.7, 0.25)
        inherited_zoom = 1.6
        authored_offset = (0.2, -0.1)
        state = ParallelCameraState.from_view_direction(
            (1.0, -2.0, 3.0),
            up_hint=(0.0, 0.0, 1.0),
            target=(1.4, -0.8, 0.6),
            screen_anchor=(-0.9, 0.55),
            zoom=1.25,
        )
        expected_view, expected_offset = _effective_legacy_inputs(
            state,
            inherited_zoom=inherited_zoom,
            frame_center=frame_center,
            display_offset=authored_offset,
        )

        semantic_scene = _ParallelCameraScene()
        _configure_camera(
            semantic_scene,
            frame_center=frame_center,
            zoom=inherited_zoom,
        )
        semantic = _sphere_controller(
            semantic_scene,
            projection=state,
            display_offset=authored_offset,
        ).attach()

        legacy_scene = _ParallelCameraScene()
        _configure_camera(
            legacy_scene,
            frame_center=frame_center,
            zoom=inherited_zoom,
        )
        legacy = _sphere_controller(
            legacy_scene,
            projection=expected_view,
            display_offset=expected_offset,
        ).attach()
        try:
            resolved = semantic._resolve_frame_inputs()
            np.testing.assert_allclose(
                resolved.view.matrix,
                expected_view.matrix,
                atol=1.0e-12,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                resolved.display_offset,
                expected_offset,
                atol=1.0e-12,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                _quadric_surface_points(semantic),
                _quadric_surface_points(legacy),
                atol=1.0e-12,
                rtol=0.0,
            )
        finally:
            semantic.restore()
            legacy.restore()

    def test_dynamic_camera_callback_updates_without_reallocating_slots(self) -> None:
        scene = _ParallelCameraScene()
        camera = _configure_camera(
            scene,
            frame_center=(0.65, -0.4, 0.1),
            zoom=1.35,
        )
        first = ParallelCameraState.from_view_direction(
            (0.6, -0.8, 1.0),
            up_hint=(0.0, 0.0, 1.0),
            target=(1.4, -0.8, 0.6),
            screen_anchor=(-1.0, 0.4),
            zoom=0.9,
        )
        second = ParallelCameraState.from_view_direction(
            (-0.4, -1.0, 0.8),
            up_hint=(0.0, 0.0, 1.0),
            target=(0.25, 0.5, -0.3),
            screen_anchor=(1.1, -0.65),
            zoom=1.15,
        )
        camera.set_parallel_state(first)
        callback_scenes: list[object] = []

        def projection(active_scene: object) -> object:
            callback_scenes.append(active_scene)
            return active_scene.camera

        controller = _sphere_controller(
            scene,
            projection=projection,
            display_offset=(0.15, -0.05),
        ).attach()
        identities = controller.slot_identities()
        scene_mobjects = tuple(scene.mobjects)
        before_points = _quadric_surface_points(controller)
        previous_frame = controller.last_frame
        camera.set_parallel_state(second)

        with patch.object(
            Mobject,
            "__init__",
            side_effect=AssertionError("camera update allocated a Mobject"),
        ):
            controller.update()

        try:
            self.assertEqual(callback_scenes, [scene, scene])
            self.assertEqual(controller.slot_identities(), identities)
            self.assertEqual(tuple(scene.mobjects), scene_mobjects)
            self.assertIsNot(controller.last_frame, previous_frame)
            self.assertFalse(
                np.array_equal(_quadric_surface_points(controller), before_points)
            )
            expected_view, expected_offset = _effective_legacy_inputs(
                second,
                inherited_zoom=1.35,
                frame_center=(0.65, -0.4, 0.1),
                display_offset=(0.15, -0.05),
            )
            resolved = controller._resolve_frame_inputs()
            np.testing.assert_allclose(
                resolved.view.matrix,
                expected_view.matrix,
                atol=1.0e-12,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                resolved.display_offset,
                expected_offset,
                atol=1.0e-12,
                rtol=0.0,
            )
        finally:
            controller.restore()

    def test_invalid_dynamic_camera_state_preserves_committed_frame(self) -> None:
        scene = _ParallelCameraScene()
        _configure_camera(
            scene,
            frame_center=(0.4, -0.25, 0.0),
            zoom=1.2,
        )
        current: dict[str, object] = {
            "projection": ParallelCameraState(
                IDENTITY_MATRIX,
                target=(1.4, -0.8, 0.6),
                screen_anchor=(-0.5, 0.25),
                zoom=1.1,
            )
        }
        controller = _sphere_controller(
            scene,
            projection=lambda _scene: current["projection"],
        ).attach()
        identities = controller.slot_identities()
        snapshot = controller.slot_snapshot()
        previous_frame = controller.last_frame
        previous_prepared = controller._last_prepared_frame
        scene_mobjects = tuple(scene.mobjects)
        current["projection"] = _InvalidSemanticState(
            np.zeros((3, 3)),
            np.zeros(3),
            np.zeros(2),
            1.0,
        )

        with self.assertRaisesRegex(
            QuadricManimError,
            "invalid semantic parallel camera",
        ):
            controller.update()

        try:
            self.assertTrue(controller.attached)
            self.assertEqual(controller.slot_identities(), identities)
            self.assertEqual(controller.slot_snapshot(), snapshot)
            self.assertEqual(tuple(scene.mobjects), scene_mobjects)
            self.assertIs(controller.last_frame, previous_frame)
            self.assertIs(controller._last_prepared_frame, previous_prepared)
        finally:
            controller.restore()

    def test_composite_state_matches_equivalent_legacy_projection(self) -> None:
        frame_center = (0.8, -0.45, 0.2)
        inherited_zoom = 1.3
        authored_offset = (-0.15, 0.2)
        state = ParallelCameraState(
            SIDE_MATRIX,
            target=(0.25, 0.5, -0.35),
            screen_anchor=(-0.7, 0.45),
            zoom=1.1,
        )
        expected_view, expected_offset = _effective_legacy_inputs(
            state,
            inherited_zoom=inherited_zoom,
            frame_center=frame_center,
            display_offset=authored_offset,
        )
        semantic_scene = _ParallelCameraScene()
        _configure_camera(
            semantic_scene,
            frame_center=frame_center,
            zoom=inherited_zoom,
        )
        semantic = _composite_controller(
            semantic_scene,
            projection=state,
            display_offset=authored_offset,
        ).attach()

        legacy_scene = _ParallelCameraScene()
        _configure_camera(
            legacy_scene,
            frame_center=frame_center,
            zoom=inherited_zoom,
        )
        legacy = _composite_controller(
            legacy_scene,
            projection=expected_view,
            display_offset=expected_offset,
        ).attach()
        try:
            semantic_numeric = semantic._last_prepared_frame
            legacy_numeric = legacy._last_prepared_frame
            assert semantic_numeric is not None and legacy_numeric is not None
            np.testing.assert_allclose(
                semantic._resolve_frame_inputs().view.matrix,
                expected_view.matrix,
                atol=1.0e-12,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                semantic._resolve_frame_inputs().display_offset,
                expected_offset,
                atol=1.0e-12,
                rtol=0.0,
            )
            self.assertEqual(
                len(semantic_numeric.numeric.surfaces),
                len(legacy_numeric.numeric.surfaces),
            )
            for observed, expected in zip(
                semantic_numeric.numeric.surfaces,
                legacy_numeric.numeric.surfaces,
                strict=True,
            ):
                self.assertEqual(observed.child_surface_id, expected.child_surface_id)
                np.testing.assert_allclose(
                    observed.surface_points,
                    expected.surface_points,
                    atol=1.0e-12,
                    rtol=0.0,
                )
            for role in semantic_numeric.numeric.plane_polygons:
                observed_paths = semantic_numeric.numeric.plane_polygons[role]
                expected_paths = legacy_numeric.numeric.plane_polygons[role]
                self.assertEqual(len(observed_paths), len(expected_paths))
                for observed, expected in zip(
                    observed_paths,
                    expected_paths,
                    strict=True,
                ):
                    np.testing.assert_allclose(
                        observed,
                        expected,
                        atol=1.0e-12,
                        rtol=0.0,
                    )
        finally:
            semantic.restore()
            legacy.restore()


if __name__ == "__main__":
    unittest.main()
