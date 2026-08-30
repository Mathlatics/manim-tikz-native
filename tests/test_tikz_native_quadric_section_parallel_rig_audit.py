from __future__ import annotations

from dataclasses import replace
import unittest

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
    compile_section_display,
)
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_preflight import (
    ParallelPreflightLimits,
    ParallelSafeFrame,
)
from tikz_native.parallel_shots import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
)
from tikz_native.quadric_section_parallel_rig import (
    ParallelSectionRigBindingError,
    build_parallel_section_rig_display_catalog,
    compile_parallel_section_rig_from_shots,
)


def _limits() -> ParallelPreflightLimits:
    return ParallelPreflightLimits(
        ParallelSafeFrame(-10.0, 10.0, -10.0, 10.0),
        min_zoom=0.2,
        max_zoom=2.0,
    )


def _compile_binding(scene: ThreeDScene):
    surface = SphereSpec("audit-sphere", (0.0, 0.0, 0.0), 1.0)
    plane = SectionPlane(
        "audit-plane",
        (0.0, 0.0, -0.4),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    timeline = compile_section_timeline(
        "audit-section",
        surface,
        (
            ParallelPlaneTranslation(
                "audit-plane-shift",
                plane,
                (0.0, 0.0, 0.8),
                start_time=0.0,
                end_time=1.0,
            ),
        ),
    )
    banks = ("audit-bank-a", "audit-bank-b")
    catalog = build_parallel_section_rig_display_catalog(
        timeline,
        banks,
        include_plane=True,
    )
    display = compile_section_display(
        catalog,
        SectionDisplayInstruction.for_mode("painted"),
    )
    camera = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
    shots = ParallelCameraShotSequence(
        (ParallelCameraShot("audit-shot", camera, duration=1.0),)
    )
    return compile_parallel_section_rig_from_shots(
        scene,
        timeline,
        shots,
        camera,
        tuple(display for _ in timeline.samples),
        limits=_limits(),
        semantic_bank_ids=banks,
        plane_patch_margin=0.1,
    )


class ParallelSectionRigBindingAuditTests(unittest.TestCase):
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

    def test_plain_camera_is_rejected_before_scene_ownership(self) -> None:
        scene = ThreeDScene()

        with self.assertRaisesRegex(
            ParallelSectionRigBindingError,
            "semantic parallel camera",
        ):
            _compile_binding(scene)

        self.assertEqual(scene.mobjects, [])

    def test_perspective_camera_is_rejected_before_scene_ownership(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        scene.camera.register_mode(
            "audit-perspective",
            np.eye(3),
            perspective_strength=0.5,
        )
        scene.camera.set_mode("audit-perspective")

        with self.assertRaisesRegex(
            ParallelSectionRigBindingError,
            "not in a parallel snapshot state",
        ):
            _compile_binding(scene)

        self.assertEqual(scene.mobjects, [])

    def test_invalid_live_viewport_capabilities_fail_before_scene_ownership(
        self,
    ) -> None:
        cases = (
            (
                "missing-set-zoom",
                lambda camera: setattr(camera, "set_zoom", None),
                r"get_zoom\(\) and set_zoom\(\)",
            ),
            (
                "missing-exact-frame-center",
                lambda camera: setattr(
                    camera,
                    "set_parallel_frame_center_xy",
                    None,
                ),
                "set_parallel_frame_center_xy",
            ),
            (
                "unpaired-transaction",
                lambda camera: setattr(
                    camera,
                    "restore_parallel_transaction",
                    None,
                ),
                "must be provided together",
            ),
            (
                "nonfinite-zoom",
                lambda camera: camera.zoom_tracker.set_value(float("nan")),
                "zoom must be finite and positive",
            ),
            (
                "nonfinite-frame-center",
                lambda camera: setattr(
                    camera._frame_center,
                    "points",
                    np.asarray(((float("nan"), 0.0, 0.0),)),
                ),
                "frame_center must contain three finite values",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label):
                scene = ThreeDScene(camera_class=MultiProjectionCamera)
                mutate(scene.camera)
                with self.assertRaisesRegex(
                    ParallelSectionRigBindingError,
                    message,
                ):
                    _compile_binding(scene)
                self.assertEqual(scene.mobjects, [])

    def test_attach_revalidates_late_camera_drift_before_scene_ownership(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        binding = _compile_binding(scene)
        camera = scene.camera
        baseline = camera.snapshot_parallel_transaction()

        def reject(message: str) -> None:
            with self.assertRaisesRegex(ParallelSectionRigBindingError, message):
                binding.attach()
            self.assertEqual(scene.mobjects, [])
            self.assertFalse(binding.attached)

        camera.register_mode(
            "late-perspective",
            np.eye(3),
            perspective_strength=0.5,
        )
        camera.set_mode("late-perspective")
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

    def test_preattach_live_viewport_does_not_pollute_compiled_first_frame(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        binding = _compile_binding(scene)

        scene.camera.set_zoom(1.25)
        scene.camera.frame_center[:] = (2.0, 0.0, 0.0)
        binding.controller.display_offset = (2.0, 0.0)
        binding.attach()
        self.assertEqual(scene.camera.get_zoom(), 1.25)
        self.assertEqual(tuple(scene.camera.frame_center), (2.0, 0.0, 0.0))
        self.assertEqual(binding.controller.display_offset, (0.0, 0.0))
        binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_frame_viewport_atomically_replaces_live_drift(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        binding = _compile_binding(scene)
        binding.attach()
        coordinator = binding.build_coordinator(scene.camera)

        scene.camera.set_zoom(1.25)
        scene.camera.frame_center[:] = (2.0, 0.0, 0.0)
        binding.controller.display_offset = (2.0, 0.0)
        coordinator.update(binding.sequence.frames[0])
        expected = binding.sequence.screen_transforms[0]
        self.assertEqual(scene.camera.get_zoom(), expected.inherited_zoom)
        self.assertEqual(
            tuple(float(item) for item in scene.camera.frame_center[:2]),
            expected.frame_center,
        )
        self.assertEqual(
            binding.controller.display_offset,
            expected.display_offset,
        )
        coordinator.restore()
        binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_live_painter_root_z_drift_is_not_hidden_by_band_cache(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        binding = _compile_binding(scene)
        binding.attach()
        coordinator = binding.build_coordinator(scene.camera)
        _item_id, root_id, expected_z = (
            binding.controller._last_painter_band_signature[0]
        )
        root = next(
            item
            for item in binding.controller.root.get_family()
            if id(item) == root_id
        )
        root.set_z_index(expected_z + 100.0, family=True)

        with self.assertRaisesRegex(
            ParallelSectionRigBindingError,
            "live painter z-index differs",
        ):
            coordinator.update(binding.sequence.frames[0])

        self.assertFalse(coordinator.active)
        binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_binding_reuses_one_coordinator_and_restore_cannot_go_stale(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        binding = _compile_binding(scene)
        binding.attach()
        first = binding.build_coordinator(scene.camera)
        second = binding.build_coordinator(scene.camera)
        self.assertIs(second, first)

        first.update(binding.sequence.frames[0])
        second.update(binding.sequence.frames[1])
        self.assertIsNotNone(first.last_committed_frame)
        second.restore()

        self.assertIsNone(first.last_committed_frame)
        self.assertFalse(first.active)
        self.assertEqual(
            binding.snapshot_section_bank_render_state().frame,
            binding.sequence.bank_render_frames[0],
        )
        binding.restore()
        self.assertEqual(scene.mobjects, [])

        binding.attach()
        reused = binding.build_coordinator(scene.camera)
        self.assertIs(reused, first)
        reused.update(binding.sequence.frames[0])
        binding.restore()
        self.assertFalse(reused.active)
        self.assertIsNone(reused.last_committed_frame)
        self.assertEqual(scene.mobjects, [])

    def test_forged_bank_geometry_digest_is_rejected(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        binding = _compile_binding(scene)
        frame = binding.sequence.bank_render_frames[0]
        forged_layer = replace(
            frame.layers[0],
            geometry_digest="sha256:" + "0" * 64,
        )
        forged = replace(frame, layers=(forged_layer,))

        with self.assertRaisesRegex(
            ParallelSectionRigBindingError,
            "geometry digest differs from the source timeline",
        ):
            binding.apply_section_bank_render_frame(forged)

        self.assertEqual(scene.mobjects, [])


if __name__ == "__main__":
    unittest.main()
