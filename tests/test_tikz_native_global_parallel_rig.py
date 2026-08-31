from __future__ import annotations

import unittest
from unittest.mock import patch

from manim import ThreeDScene, tempconfig
import numpy as np

from polyhedron_visibility.quadrics.contract import SectionPlane, SphereSpec
from polyhedron_visibility.quadrics.parallel_plane_motion import (
    ParallelPlaneTranslation,
)
from polyhedron_visibility.quadrics.section_timeline import (
    compile_section_timeline,
)
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayInstruction,
    SectionDisplayRole,
    compile_section_display,
)
from polyhedron_visibility.quadrics.visibility import VisibilityKind
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.global_parallel_rig import (
    GlobalParallelRigError,
    compile_global_parallel_rig,
)
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_frame import ParallelFrameState
from tikz_native.parallel_preflight import (
    ParallelPreflightLimits,
    ParallelSafeFrame,
    ParallelScreenTransform,
)
from tikz_native.parallel_viewport import PARALLEL_VIEWPORT_TRANSFORM_CHANNEL
from tikz_native.parallel_shots import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
)
from tikz_native.quadric_section_parallel_rig import (
    build_parallel_section_rig_display_catalog,
    compile_parallel_section_rig_from_shots,
)


def _preflight_limits() -> ParallelPreflightLimits:
    return ParallelPreflightLimits(
        ParallelSafeFrame(-10.0, 10.0, -10.0, 10.0),
        0.2,
        2.0,
    )


def _sphere_binding(
    scene: ThreeDScene,
    prefix: str,
    *,
    center_z: float,
    radius: float,
    tangent_end: bool,
    include_plane: bool = False,
    camera_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    render_times: tuple[float, ...] | None = None,
):
    surface = SphereSpec(
        f"{prefix}-sphere",
        (0.0, 0.0, center_z),
        radius,
    )
    # Keep the non-tangent circle away from the sphere's projected silhouette:
    # at the equator those two semantic curves have coincident support and the
    # certified crossing solver correctly fails closed.
    initial_height = center_z + 0.20 * radius
    final_height = center_z + (radius if tangent_end else 0.40 * radius)
    plane = SectionPlane(
        f"{prefix}-plane",
        (0.0, 0.0, initial_height),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    timeline = compile_section_timeline(
        f"{prefix}-section",
        surface,
        (
            ParallelPlaneTranslation(
                f"{prefix}-plane-motion",
                plane,
                (0.0, 0.0, final_height - initial_height),
                start_time=0.0,
                end_time=2.0,
            ),
        ),
    )
    banks = (f"{prefix}-bank-a", f"{prefix}-bank-b")
    catalog = build_parallel_section_rig_display_catalog(
        timeline,
        banks,
        include_plane=include_plane,
        surface_boundary_mode="certified",
    )
    display = compile_section_display(
        catalog,
        SectionDisplayInstruction.for_mode("painted"),
    )
    camera = ParallelCameraState.from_view_direction(
        camera_direction,
        zoom=0.8,
    )
    shots = ParallelCameraShotSequence(
        (ParallelCameraShot(f"{prefix}-shot", camera, duration=2.0),)
    )
    binding = compile_parallel_section_rig_from_shots(
        scene,
        timeline,
        shots,
        camera,
        tuple(display for _ in timeline.samples),
        limits=_preflight_limits(),
        semantic_bank_ids=banks,
        render_times=render_times,
        plane_patch_margin=(0.1 if include_plane else None),
    )
    return binding


class GlobalParallelRigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "frame_rate": 4,
                "pixel_width": 160,
                "pixel_height": 90,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
                "progress_bar": "none",
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def _bindings(self, scene: ThreeDScene):
        far = _sphere_binding(
            scene,
            "global-far",
            center_z=0.0,
            radius=0.8,
            tangent_end=True,
        )
        near = _sphere_binding(
            scene,
            "global-near",
            center_z=3.0,
            radius=1.4,
            tangent_end=False,
            render_times=far.sequence.evaluation_times,
        )
        return far, near

    def test_one_global_controller_hides_other_rig_curve_point_and_boundary(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        far, near = self._bindings(scene)
        global_binding = compile_global_parallel_rig((far, near))

        self.assertEqual(scene.mobjects, [])
        self.assertFalse(far.attached)
        self.assertFalse(near.attached)
        self.assertEqual(
            len(global_binding.sequence.evidences),
            len(far.sequence.frames),
        )
        self.assertTrue(
            all(
                len(item.rig_state_digests) == 2 and item.painter_order.draw_order
                for item in global_binding.sequence.evidences
            )
        )

        global_binding.attach()
        self.assertTrue(global_binding.attached)
        self.assertFalse(far.attached)
        self.assertFalse(near.attached)
        self.assertEqual(len(scene.mobjects), 2)
        identities = global_binding.controller.slot_identities()

        initial_boundary = global_binding.controller.last_boundary_frame
        self.assertIsNotNone(initial_boundary)
        assert initial_boundary is not None
        active_far_curve_ids = {item.curve_id for item in far._curves}
        far_curve_fragments = tuple(
            item
            for item in initial_boundary.fragments
            if item.source_id in active_far_curve_ids
        )
        self.assertTrue(far_curve_fragments)
        self.assertTrue(
            all(
                item.effective_visibility_kind is VisibilityKind.HIDDEN
                and "global-near-sphere" in item.occluder_surface_ids
                for item in far_curve_fragments
            )
        )
        far_surface_boundaries = tuple(
            item
            for item in initial_boundary.fragments
            if item.source_id.startswith("boundary:global-far-sphere:")
        )
        self.assertTrue(far_surface_boundaries)
        self.assertTrue(
            all(
                item.effective_visibility_kind is VisibilityKind.HIDDEN
                and "global-near-sphere" in item.occluder_surface_ids
                for item in far_surface_boundaries
            )
        )

        coordinator = global_binding.build_coordinator(scene.camera)
        coordinator.update(global_binding.sequence.frames[-1])
        prepared = global_binding.controller._last_prepared_frame
        self.assertIsNotNone(prepared)
        assert prepared is not None
        far_points = tuple(
            item
            for item in prepared.numeric.points
            if item.point_id.startswith("global-far-")
        )
        self.assertTrue(far_points)
        self.assertTrue(
            all(
                not item.visible
                and "global-near-sphere" in item.occluders
                for item in far_points
            )
        )
        self.assertEqual(global_binding.controller.slot_identities(), identities)
        self.assertEqual(
            coordinator.participant_ids,
            (
                "global-parallel-preflight",
                "parallel-viewport",
                "global-parallel-provider-stage",
                "global-parallel-paint",
            ),
        )
        self.assertFalse(far.attached)
        self.assertFalse(near.attached)

        coordinator.restore()
        global_binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_failed_global_update_rolls_back_camera_providers_and_display(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        far, near = self._bindings(scene)
        global_binding = compile_global_parallel_rig((far, near)).attach()
        coordinator = global_binding.build_coordinator(scene.camera)
        coordinator.update(global_binding.sequence.frames[0])

        old_far_curves = far._curves
        old_near_curves = near._curves
        old_far_bank = far._bank_frame
        old_near_bank = near._bank_frame
        old_frame = global_binding.controller.last_frame
        old_boundary = global_binding.controller.last_boundary_frame
        old_order = dict(global_binding.controller.active_painter_z_indices)
        old_slots = global_binding.controller.slot_snapshot()
        old_camera = scene.camera.snapshot_parallel_state()
        original_update = global_binding.controller.update

        def fail_after_commit():
            original_update()
            raise RuntimeError("injected global paint failure")

        with patch.object(
            global_binding.controller,
            "update",
            side_effect=fail_after_commit,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected global paint failure",
            ):
                coordinator.update(global_binding.sequence.frames[-1])

        self.assertIs(far._curves, old_far_curves)
        self.assertIs(near._curves, old_near_curves)
        self.assertIs(far._bank_frame, old_far_bank)
        self.assertIs(near._bank_frame, old_near_bank)
        self.assertIs(global_binding.controller.last_frame, old_frame)
        self.assertIs(global_binding.controller.last_boundary_frame, old_boundary)
        self.assertEqual(
            global_binding.controller.active_painter_z_indices,
            old_order,
        )
        self.assertEqual(global_binding.controller.slot_snapshot(), old_slots)
        restored_camera = scene.camera.snapshot_parallel_state()
        np.testing.assert_array_equal(restored_camera.matrix, old_camera.matrix)
        np.testing.assert_array_equal(restored_camera.target, old_camera.target)
        np.testing.assert_array_equal(
            restored_camera.screen_anchor,
            old_camera.screen_anchor,
        )
        self.assertEqual(restored_camera.zoom, old_camera.zoom)
        self.assertFalse(coordinator.poisoned)
        self.assertFalse(far.attached)
        self.assertFalse(near.attached)

        coordinator.restore()
        global_binding.restore()

    def test_forged_viewport_fails_before_any_state_change(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        far, near = self._bindings(scene)
        global_binding = compile_global_parallel_rig((far, near)).attach()
        coordinator = global_binding.build_coordinator(scene.camera)
        expected = global_binding.sequence.frames[-1]
        expected_transform = expected.channel(
            PARALLEL_VIEWPORT_TRANSFORM_CHANNEL
        )
        self.assertIsInstance(expected_transform, ParallelScreenTransform)
        assert isinstance(expected_transform, ParallelScreenTransform)

        old_far_curves = far._curves
        old_near_curves = near._curves
        old_far_bank = far._bank_frame
        old_near_bank = near._bank_frame
        old_far_display = far._display_frame
        old_near_display = near._display_frame
        old_far_compositing = far._compositing_frame
        old_near_compositing = near._compositing_frame
        old_frame = global_binding.controller.last_frame
        old_boundary = global_binding.controller.last_boundary_frame
        old_order = dict(global_binding.controller.active_painter_z_indices)
        old_slots = global_binding.controller.slot_snapshot()
        old_display_offset = global_binding.controller.display_offset
        old_camera = scene.camera.snapshot_parallel_state()
        old_zoom = scene.camera.get_zoom()
        old_frame_center = np.array(scene.camera.frame_center, copy=True)

        forged_transforms = (
            ParallelScreenTransform(
                inherited_zoom=expected_transform.inherited_zoom,
                frame_center=expected_transform.frame_center,
                display_offset=(5.0, -2.0),
            ),
            ParallelScreenTransform(
                inherited_zoom=1.25 * expected_transform.inherited_zoom,
                frame_center=expected_transform.frame_center,
                display_offset=expected_transform.display_offset,
            ),
        )
        for forged_transform in forged_transforms:
            with self.subTest(transform=forged_transform.to_dict()):
                forged_channels = dict(expected.channels)
                forged_channels[
                    PARALLEL_VIEWPORT_TRANSFORM_CHANNEL
                ] = forged_transform
                forged = ParallelFrameState(
                    expected.camera,
                    forged_channels,
                    frame_id=expected.frame_id,
                    preflight_input_digest=expected.preflight_input_digest,
                )
                with self.assertRaisesRegex(
                    GlobalParallelRigError,
                    "viewport transform differs from preflight",
                ):
                    coordinator.update(forged)

                self.assertIs(far._curves, old_far_curves)
                self.assertIs(near._curves, old_near_curves)
                self.assertIs(far._bank_frame, old_far_bank)
                self.assertIs(near._bank_frame, old_near_bank)
                self.assertIs(far._display_frame, old_far_display)
                self.assertIs(near._display_frame, old_near_display)
                self.assertIs(far._compositing_frame, old_far_compositing)
                self.assertIs(near._compositing_frame, old_near_compositing)
                self.assertIs(global_binding.controller.last_frame, old_frame)
                self.assertIs(
                    global_binding.controller.last_boundary_frame,
                    old_boundary,
                )
                self.assertEqual(
                    global_binding.controller.active_painter_z_indices,
                    old_order,
                )
                self.assertEqual(
                    global_binding.controller.slot_snapshot(),
                    old_slots,
                )
                self.assertEqual(
                    global_binding.controller.display_offset,
                    old_display_offset,
                )
                restored_camera = scene.camera.snapshot_parallel_state()
                np.testing.assert_array_equal(
                    restored_camera.matrix,
                    old_camera.matrix,
                )
                np.testing.assert_array_equal(
                    restored_camera.target,
                    old_camera.target,
                )
                np.testing.assert_array_equal(
                    restored_camera.screen_anchor,
                    old_camera.screen_anchor,
                )
                self.assertEqual(restored_camera.zoom, old_camera.zoom)
                self.assertEqual(scene.camera.get_zoom(), old_zoom)
                np.testing.assert_array_equal(
                    scene.camera.frame_center,
                    old_frame_center,
                )
                self.assertIsNone(coordinator.last_committed_frame)
                self.assertFalse(coordinator.active)
                self.assertFalse(coordinator.poisoned)

        global_binding.restore()

    def test_plane_roles_fail_closed_before_any_scene_ownership(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        far = _sphere_binding(
            scene,
            "global-plane-far",
            center_z=0.0,
            radius=0.8,
            tangent_end=False,
            include_plane=True,
            # This test targets the global plane-role preflight, not the
            # projected curve-crossing solver.  An axis-aligned view keeps the
            # section and silhouette concentric on every supported BLAS/LAPACK
            # implementation, so unrelated high-degree root conditioning
            # cannot prevent the intended preflight assertion from running.
            camera_direction=(0.0, 0.0, 1.0),
        )
        near = _sphere_binding(
            scene,
            "global-plane-near",
            center_z=3.0,
            radius=1.4,
            tangent_end=False,
            camera_direction=(0.0, 0.0, 1.0),
        )
        self.assertIn(
            SectionDisplayRole.PLANE_FILL,
            {item.role for item in far.sequence.display_frames[0].slots},
        )
        with self.assertRaisesRegex(
            GlobalParallelRigError,
            "plane patches|plane-fill",
        ):
            compile_global_parallel_rig((far, near))
        self.assertEqual(scene.mobjects, [])
        self.assertFalse(far.attached)
        self.assertFalse(near.attached)

    def test_attach_revalidates_late_camera_drift_before_scene_ownership(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        far, near = self._bindings(scene)
        binding = compile_global_parallel_rig((far, near))
        camera = scene.camera
        baseline = camera.snapshot_parallel_transaction()

        def reject(message: str) -> None:
            with self.assertRaisesRegex(GlobalParallelRigError, message):
                binding.attach()
            self.assertEqual(scene.mobjects, [])
            self.assertFalse(binding.attached)
            self.assertFalse(far.attached)
            self.assertFalse(near.attached)

        camera.register_mode(
            "late-global-perspective",
            np.eye(3),
            perspective_strength=0.5,
        )
        camera.set_mode("late-global-perspective")
        reject("not in a parallel snapshot state")
        camera.restore_parallel_transaction(baseline)

        camera.zoom_tracker.set_value(float("nan"))
        reject("zoom must be finite and positive")
        camera.restore_parallel_transaction(baseline)

        camera._frame_center.points = np.asarray(
            ((float("nan"), 0.0, 0.0),)
        )
        reject("frame_center must contain three finite values")
        camera.restore_parallel_transaction(baseline)

        zoom_setter = camera.set_zoom
        camera.set_zoom = None
        reject(r"get_zoom\(\) and set_zoom\(\)")
        camera.set_zoom = zoom_setter

        center_setter = camera.set_parallel_frame_center_xy
        camera.set_parallel_frame_center_xy = None
        reject("set_parallel_frame_center_xy")
        camera.set_parallel_frame_center_xy = center_setter

        transaction_restore = camera.restore_parallel_transaction
        camera.restore_parallel_transaction = None
        reject("must be provided together")
        camera.restore_parallel_transaction = transaction_restore
        camera.restore_parallel_transaction(baseline)


if __name__ == "__main__":
    unittest.main()
