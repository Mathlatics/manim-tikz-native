from __future__ import annotations

import unittest
from unittest.mock import patch
from math import pi

from manim import ThreeDScene, tempconfig
import numpy as np

from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
)
from polyhedron_visibility.quadrics.plane_motion import AxisAnglePlaneMotion
from polyhedron_visibility.quadrics.parallel_plane_motion import (
    ParallelPlaneTranslation,
)
from polyhedron_visibility.quadrics.section_timeline import compile_section_timeline
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayInstruction,
    compile_section_display,
)
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_frame import (
    ParallelFrameParticipant,
    ParallelFramePhase,
)
from tikz_native.parallel_preflight import ParallelPreflightLimits, ParallelSafeFrame
from tikz_native.parallel_shots import ParallelCameraShot, ParallelCameraShotSequence
from tikz_native.quadric_section_parallel_rig import (
    ParallelSectionRigBindingError,
    build_parallel_section_rig_display_catalog,
    compile_parallel_section_rig_from_shots,
)
from tikz_native.quadric_section_parallel_manim import (
    play_parallel_section_sequence,
)


def _limits() -> ParallelPreflightLimits:
    return ParallelPreflightLimits(
        ParallelSafeFrame(-10.0, 10.0, -10.0, 10.0),
        0.2,
        2.0,
    )


def _inside_sphere_source():
    surface = SphereSpec("binding-sphere", (0.0, 0.0, 0.0), 1.2)
    plane = SectionPlane(
        "binding-plane",
        (0.0, 0.0, -0.55),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    timeline = compile_section_timeline(
        "binding-section",
        surface,
        (
            ParallelPlaneTranslation(
                "binding-plane-shift",
                plane,
                (0.0, 0.0, 1.1),
                start_time=0.0,
                end_time=2.0,
            ),
        ),
    )
    banks = ("binding-bank-a", "binding-bank-b")
    catalog = build_parallel_section_rig_display_catalog(
        timeline,
        banks,
        include_plane=True,
    )
    initial = ParallelCameraState.from_view_direction(
        (1.0, 1.0, 1.0),
        zoom=0.8,
    )
    endpoint = ParallelCameraState.along_plane(
        timeline.samples[-1].plane,
        direction=(1.0, 0.0, 0.0),
        zoom=0.8,
    )
    shots = ParallelCameraShotSequence(
        (
            ParallelCameraShot(
                "binding-side-view",
                endpoint,
                duration=2.0,
                transition="orbit",
                arc_height=0.5,
            ),
        )
    )
    return timeline, banks, catalog, initial, shots


def _inside_sphere_fixture(
    scene: ThreeDScene,
    *,
    display_mode: str = "painted",
    emphasize_section: bool = False,
    dim_unemphasized: float = 0.25,
    controller_options: dict[str, object] | None = None,
):
    timeline, banks, catalog, initial, shots = _inside_sphere_source()
    display = compile_section_display(
        catalog,
        SectionDisplayInstruction.for_mode(
            display_mode,
            emphasized_handles=(
                (catalog.section_curve.handle_id,) if emphasize_section else ()
            ),
            dim_unemphasized=dim_unemphasized,
        ),
    )
    binding = compile_parallel_section_rig_from_shots(
        scene,
        timeline,
        shots,
        initial,
        tuple(display for _ in timeline.samples),
        limits=_limits(),
        semantic_bank_ids=banks,
        frame_rate=4.0,
        plane_patch_margin=0.1,
        controller_options=controller_options,
    )
    return timeline, initial, binding


class ParallelSectionRigBindingTests(unittest.TestCase):
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

    def test_outline_only_constant_opacity_multiplies_caller_style(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        authored_style = QuadricManimStyle(
            surface_fill_opacity=0.7,
            surface_stroke_width=3.25,
            surface_stroke_opacity=0.8,
            section_plane_fill_opacity=0.3,
            section_plane_stroke_width=2.75,
            section_plane_stroke_opacity=0.6,
        )
        _timeline, _initial, binding = _inside_sphere_fixture(
            scene,
            display_mode="outline-only",
            emphasize_section=True,
            dim_unemphasized=0.4,
            controller_options={"style": authored_style},
        )

        compiled_style = binding.controller.style
        self.assertTrue(binding.controller.legacy_surface_stroke_fallback)
        self.assertEqual(compiled_style.surface_fill_opacity, 0.0)
        self.assertAlmostEqual(compiled_style.surface_stroke_opacity, 0.32)
        self.assertEqual(compiled_style.surface_stroke_width, 3.25)
        self.assertEqual(compiled_style.section_plane_fill_opacity, 0.0)
        self.assertAlmostEqual(compiled_style.section_plane_stroke_opacity, 0.24)
        self.assertEqual(compiled_style.section_plane_stroke_width, 2.75)
        self.assertEqual(authored_style.surface_fill_opacity, 0.7)
        self.assertEqual(authored_style.surface_stroke_opacity, 0.8)
        self.assertEqual(authored_style.section_plane_fill_opacity, 0.3)
        self.assertEqual(authored_style.section_plane_stroke_opacity, 0.6)
        self.assertEqual(scene.mobjects, [])

    def test_static_display_role_opacity_change_is_rejected(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        timeline, banks, catalog, initial, shots = _inside_sphere_source()
        painted = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        outline = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("outline-only"),
        )
        displays = [painted for _ in timeline.samples]
        displays[-1] = outline

        with self.assertRaisesRegex(
            ParallelSectionRigBindingError,
            "static display role .* must remain constant",
        ):
            compile_parallel_section_rig_from_shots(
                scene,
                timeline,
                shots,
                initial,
                tuple(displays),
                limits=_limits(),
                semantic_bank_ids=banks,
                frame_rate=4.0,
                plane_patch_margin=0.1,
            )
        self.assertEqual(scene.mobjects, [])

    def test_real_painter_preflight_uses_one_controller_and_reaches_line(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        _timeline, _initial, binding = _inside_sphere_fixture(scene)
        self.assertEqual(scene.mobjects, [])
        self.assertTrue(
            all(
                "pending-painter" not in item
                for frame in binding.sequence.painter_orders
                for item in frame.draw_order
            )
        )
        self.assertEqual(binding.controller.allocated_curve_ids, binding.allocated_curve_ids)

        identities = binding.controller.slot_identities()
        binding.attach()
        self.assertFalse(binding.controller.automatic_updates)
        automatic_calls = []
        original_update = binding.controller.update

        def record_automatic_update(dt: float = 0.0):
            automatic_calls.append(dt)
            return original_update(dt)

        binding.controller.update = record_automatic_update
        binding.controller._update_driver.update(0.25)
        self.assertEqual(automatic_calls, [])
        binding.controller.update = original_update
        coordinator = binding.build_coordinator(scene.camera)
        for frame in binding.sequence.frames:
            coordinator.update(frame)

        self.assertEqual(binding.controller.slot_identities(), identities)
        self.assertEqual(
            binding.controller.last_section_frame.projection_kind.value,
            "line",
        )
        actual_order = tuple(
            item_id
            for item_id, _z in sorted(
                binding.controller.active_painter_z_indices.items(),
                key=lambda item: item[1],
            )
        )
        self.assertEqual(actual_order, binding.sequence.painter_orders[-1].draw_order)
        coordinator.restore()
        binding.restore()
        self.assertEqual(scene.mobjects, [])

        # A second session must start from frame zero rather than retain the
        # prior session's final bank, plane patch, or semantic display state.
        binding.attach()
        second = binding.build_coordinator(scene.camera)
        second.update(binding.sequence.frames[0])
        second_order = tuple(
            item_id
            for item_id, _z in sorted(
                binding.controller.active_painter_z_indices.items(),
                key=lambda item: item[1],
            )
        )
        self.assertEqual(second_order, binding.sequence.painter_orders[0].draw_order)
        self.assertEqual(
            binding.controller.last_section_frame.plane,
            binding.sequence.plane_patch_fits[0].plane,
        )
        second.restore()
        binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_late_participant_failure_restores_controller_and_camera(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        _timeline, _initial, binding = _inside_sphere_fixture(scene)
        binding.attach()
        coordinator = binding.build_coordinator(scene.camera)
        fail = {"enabled": False}

        def commit(_value: object) -> None:
            if fail["enabled"]:
                raise RuntimeError("injected finalize failure")

        coordinator.add(
            ParallelFrameParticipant(
                "injected-finalizer",
                ParallelFramePhase.FINALIZE,
                prepare=lambda _frame: None,
                snapshot=lambda: None,
                commit=commit,
                rollback=lambda _value: None,
            )
        )
        coordinator.update(binding.sequence.frames[0])
        slots = binding.controller.slot_snapshot()
        camera = scene.camera.snapshot_parallel_state()
        painter = dict(binding.controller.active_painter_z_indices)
        fail["enabled"] = True
        with self.assertRaisesRegex(RuntimeError, "injected finalize failure"):
            coordinator.update(binding.sequence.frames[1])
        self.assertEqual(binding.controller.slot_snapshot(), slots)
        self.assertEqual(binding.controller.active_painter_z_indices, painter)
        restored = scene.camera.snapshot_parallel_state()
        np.testing.assert_array_equal(restored.matrix, camera.matrix)
        np.testing.assert_array_equal(restored.target, camera.target)
        np.testing.assert_array_equal(restored.screen_anchor, camera.screen_anchor)
        self.assertEqual(restored.zoom, camera.zoom)
        coordinator.restore()
        binding.restore()

    def test_real_player_calls_external_controller_once_per_output_frame(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        _timeline, initial, binding = _inside_sphere_fixture(scene)
        scene.camera.set_parallel_state(initial)
        binding.attach()
        coordinator = binding.build_coordinator(scene.camera)
        shots = binding.sequence.camera_provenance.shot_sequence
        with patch.object(
            binding.controller,
            "update",
            wraps=binding.controller.update,
        ) as update:
            final = play_parallel_section_sequence(
                scene,
                binding.sequence,
                shots,
                coordinator,
            )
        self.assertIs(final, binding.sequence.frames[-1])
        self.assertEqual(update.call_count, len(binding.sequence.frames))
        coordinator.restore()
        binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_isolated_point_activation_fails_before_scene_ownership(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        surface = SphereSpec("point-sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "point-plane",
            (0.0, 0.0, -2.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        timeline = compile_section_timeline(
            "point-section",
            surface,
            (
                ParallelPlaneTranslation(
                    "point-motion",
                    plane,
                    (0.0, 0.0, 4.0),
                    start_time=0.0,
                    end_time=2.0,
                ),
            ),
        )
        banks = ("point-bank-a", "point-bank-b")
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
            (ParallelCameraShot("point-shot", camera, duration=2.0),)
        )
        with self.assertRaisesRegex(
            ParallelSectionRigBindingError,
            "true fixed Manim point slot",
        ):
            compile_parallel_section_rig_from_shots(
                scene,
                timeline,
                shots,
                camera,
                tuple(display for _ in timeline.samples),
                limits=_limits(),
                semantic_bank_ids=banks,
                plane_patch_margin=0.1,
            )
        self.assertEqual(scene.mobjects, [])

    def test_cone_topology_banks_cap_chords_and_exact_side_view(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        cone = ConeSpec(
            "binding-cone",
            (0.0, 0.0, -1.5),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.CLOSED_SINGLE,
        )
        plane = SectionPlane(
            "binding-cone-plane",
            (0.0, 0.0, 0.2),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        motion = AxisAnglePlaneMotion(
            "binding-cone-rotation",
            plane,
            (0.0, 0.0, 0.2),
            (0.0, 1.0, 0.0),
            0.0,
            1.2,
            start_time=0.0,
            end_time=6.0,
        )
        timeline = compile_section_timeline(
            "binding-cone-section",
            cone,
            (motion,),
        )
        banks = ("binding-cone-bank-a", "binding-cone-bank-b")
        catalog = build_parallel_section_rig_display_catalog(
            timeline,
            banks,
            include_plane=True,
        )
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        initial = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
        final_plane = motion.plane_at(1.0)
        endpoint = ParallelCameraState.along_plane(
            final_plane,
            direction=final_plane.u_axis,
            target=final_plane.point,
        )
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "binding-cone-side-view",
                    endpoint,
                    duration=6.0,
                    transition="orbit",
                    arc_height=0.5,
                ),
            )
        )
        binding = compile_parallel_section_rig_from_shots(
            scene,
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=_limits(),
            semantic_bank_ids=banks,
            plane_patch_margin=0.1,
            controller_options={
                "limits": QuadricManimLimits(
                    max_surfaces=2,
                    max_curves=16,
                    max_fragments_per_curve=16,
                    max_segments_per_fragment=256,
                    max_surface_segments=512,
                    max_dashes_per_fragment=128,
                    max_projected_length=30.0,
                    max_total_mobjects=20000,
                    max_boundary_sources=16,
                    max_boundary_styles=16,
                )
            },
        )
        families = {
            frame.signature.conic_family.value
            for frame in timeline.animation.frames
        }
        self.assertTrue({"oval", "parabola", "hyperbola"}.issubset(families))
        crossfades = tuple(
            frame for frame in binding.sequence.bank_render_frames
            if len(frame.layers) == 2
        )
        self.assertTrue(crossfades)
        self.assertTrue(
            any(
                all(layer.active_cap_chord_ids for layer in frame.layers)
                for frame in crossfades
            )
        )

        binding.attach()
        identities = binding.controller.slot_identities()
        coordinator = binding.build_coordinator(scene.camera)
        fail = {"enabled": False}

        def fail_after_display(_value: object) -> None:
            if fail["enabled"]:
                raise RuntimeError("injected crossfade finalize failure")

        coordinator.add(
            ParallelFrameParticipant(
                "binding-cone-finalizer",
                ParallelFramePhase.FINALIZE,
                prepare=lambda _frame: None,
                snapshot=lambda: None,
                commit=fail_after_display,
                rollback=lambda _value: None,
            )
        )
        crossfade_index = next(
            index
            for index, frame in enumerate(binding.sequence.bank_render_frames)
            if len(frame.layers) == 2
        )
        for frame in binding.sequence.frames[:crossfade_index]:
            coordinator.update(frame)
        expected_section_id = binding.controller.section_id
        expected_tolerance = binding.controller.section_coefficient_tolerance
        fail["enabled"] = True
        with self.assertRaisesRegex(
            RuntimeError,
            "injected crossfade finalize failure",
        ):
            coordinator.update(binding.sequence.frames[crossfade_index])
        self.assertTrue(binding._section_sources_authoritative)
        self.assertEqual(binding.controller.section_id, expected_section_id)
        self.assertEqual(
            binding.controller.section_coefficient_tolerance,
            expected_tolerance,
        )
        coordinator.restore()

        fail["enabled"] = False
        coordinator = binding.build_coordinator(scene.camera)
        for frame in binding.sequence.frames:
            coordinator.update(frame)
        self.assertEqual(binding.controller.slot_identities(), identities)
        self.assertEqual(
            binding.controller.last_section_frame.projection_kind.value,
            "line",
        )
        coordinator.restore()
        binding.restore()
        self.assertEqual(scene.mobjects, [])


if __name__ == "__main__":
    unittest.main()
