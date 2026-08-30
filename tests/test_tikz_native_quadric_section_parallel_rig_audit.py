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
from tikz_native.quadric_section_parallel import ParallelSectionSequenceError
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

    def test_preattach_screen_transform_drift_is_rejected_without_ownership(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        binding = _compile_binding(scene)

        scene.camera.set_zoom(1.25)
        with self.subTest(term="inherited zoom"):
            with self.assertRaisesRegex(
                ParallelSectionRigBindingError,
                "live renderer screen transform",
            ):
                binding.attach()
            self.assertEqual(scene.mobjects, [])
        scene.camera.set_zoom(1.0)

        scene.camera.frame_center[:] = (2.0, 0.0, 0.0)
        with self.subTest(term="frame center"):
            with self.assertRaisesRegex(
                ParallelSectionRigBindingError,
                "live renderer screen transform",
            ):
                binding.attach()
            self.assertEqual(scene.mobjects, [])
        scene.camera.frame_center[:] = (0.0, 0.0, 0.0)

        binding.controller.display_offset = (2.0, 0.0)
        with self.subTest(term="display offset"):
            with self.assertRaisesRegex(
                ParallelSectionRigBindingError,
                "live renderer screen transform",
            ):
                binding.attach()
            self.assertEqual(scene.mobjects, [])
        binding.controller.display_offset = (0.0, 0.0)

        binding.attach()
        binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_postattach_live_screen_transform_guard_fails_before_commit(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        binding = _compile_binding(scene)
        binding.attach()
        coordinator = binding.build_coordinator(scene.camera)

        mutations = (
            (
                "inherited zoom",
                lambda: scene.camera.set_zoom(1.25),
                lambda: scene.camera.set_zoom(1.0),
            ),
            (
                "frame center",
                lambda: scene.camera.frame_center.__setitem__(
                    slice(None),
                    (2.0, 0.0, 0.0),
                ),
                lambda: scene.camera.frame_center.__setitem__(
                    slice(None),
                    (0.0, 0.0, 0.0),
                ),
            ),
            (
                "display offset",
                lambda: setattr(binding.controller, "display_offset", (2.0, 0.0)),
                lambda: setattr(binding.controller, "display_offset", (0.0, 0.0)),
            ),
        )
        for label, mutate, reset in mutations:
            with self.subTest(term=label):
                mutate()
                with self.assertRaisesRegex(
                    ParallelSectionSequenceError,
                    "live renderer screen transform",
                ):
                    coordinator.update(binding.sequence.frames[0])
                self.assertFalse(coordinator.active)
                reset()

        coordinator.update(binding.sequence.frames[0])
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
