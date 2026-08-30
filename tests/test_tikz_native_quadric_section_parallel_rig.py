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
    SectionDisplayRole,
    SectionDisplayInstruction,
    compile_section_display,
)
from polyhedron_visibility.quadrics.semantic_compositing import (
    SectionCompositingAxes,
    SectionCompositingInstruction,
    SectionCompositingOverride,
    compile_section_compositing,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_frame import (
    ParallelFrameParticipant,
    ParallelFramePhase,
)
from tikz_native.parallel_preflight import (
    ParallelPreflightLimits,
    ParallelSafeFrame,
    ParallelScreenTransform,
)
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


def _inside_sphere_source(*, surface_boundary_mode: str = "certified"):
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
        surface_boundary_mode=surface_boundary_mode,
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
    surface_boundary_mode: str = "certified",
    controller_options: dict[str, object] | None = None,
):
    timeline, banks, catalog, initial, shots = _inside_sphere_source(
        surface_boundary_mode=surface_boundary_mode,
    )
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
            surface_boundary_mode="legacy",
            controller_options={"style": authored_style},
        )

        compiled_style = binding.controller.style
        self.assertTrue(binding.controller.legacy_surface_stroke_fallback)
        self.assertEqual(compiled_style.surface_fill_opacity, 0.7)
        self.assertAlmostEqual(compiled_style.surface_stroke_opacity, 0.8)
        self.assertEqual(compiled_style.surface_stroke_width, 3.25)
        self.assertEqual(compiled_style.section_plane_fill_opacity, 0.3)
        self.assertAlmostEqual(compiled_style.section_plane_stroke_opacity, 0.6)
        self.assertEqual(compiled_style.section_plane_stroke_width, 2.75)
        self.assertEqual(
            binding._resolve_surface_opacities()["binding-sphere"],
            0.0,
        )
        self.assertEqual(binding._resolve_section_plane_fill_opacity(), 0.0)
        self.assertAlmostEqual(
            binding._resolve_section_plane_stroke_opacity(),
            0.4,
        )
        self.assertAlmostEqual(
            binding._resolve_surface_stroke_opacities()["binding-sphere"],
            0.4,
        )
        self.assertEqual(authored_style.surface_fill_opacity, 0.7)
        self.assertEqual(authored_style.surface_stroke_opacity, 0.8)
        self.assertEqual(authored_style.section_plane_fill_opacity, 0.3)
        self.assertEqual(authored_style.section_plane_stroke_opacity, 0.6)

        binding.attach()
        front = binding.controller._section_surface_paint_slots[4].base
        self.assertEqual(float(front.get_fill_opacity()), 0.0)
        self.assertAlmostEqual(float(front.get_stroke_opacity()), 0.32)
        binding.controller.display_mobject.set_opacity(0.5)
        binding.controller.update()
        self.assertEqual(float(front.get_fill_opacity()), 0.0)
        self.assertAlmostEqual(float(front.get_stroke_opacity()), 0.16)
        binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_certified_surface_boundaries_replace_legacy_outline(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        timeline, banks, catalog, initial, shots = _inside_sphere_source()
        roles = {slot.role for slot in catalog.slots}
        self.assertIn(SectionDisplayRole.CONTOUR, roles)
        self.assertNotIn(SectionDisplayRole.SURFACE_OUTLINE, roles)
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
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
        )

        self.assertTrue(binding.controller.include_surface_boundaries)
        self.assertFalse(binding.controller.legacy_surface_stroke_fallback)
        contour_sources = {
            slot.source_id
            for slot in catalog.slots
            if slot.role is SectionDisplayRole.CONTOUR
        }
        self.assertTrue(contour_sources)
        self.assertTrue(
            contour_sources.issubset(
                set(binding.controller.allocated_boundary_ids)
            )
        )
        self.assertEqual(scene.mobjects, [])

    def test_legacy_surface_outline_compositing_opacity_updates_without_geometry(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        timeline, banks, catalog, initial, shots = _inside_sphere_source(
            surface_boundary_mode="legacy"
        )
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("outline-only"),
        )
        outline_slot = next(
            item.slot_id
            for item in catalog.slots
            if item.role is SectionDisplayRole.SURFACE_OUTLINE
        )
        dimmed_outline = compile_section_compositing(
            catalog,
            SectionCompositingInstruction.for_catalog(
                catalog,
                defaults=SectionCompositingAxes(
                    depth_presentation="diagrammatic"
                ),
                overrides=(
                    SectionCompositingOverride.for_slot(
                        outline_slot,
                        display_opacity=0.25,
                    ),
                ),
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
            controller_options={
                "style": QuadricManimStyle(surface_stroke_opacity=0.8)
            },
        )
        scene.camera.set_parallel_state(initial)
        binding.attach()
        front = binding.controller._section_surface_paint_slots[4].base
        self.assertEqual(float(front.get_fill_opacity()), 0.0)
        self.assertAlmostEqual(float(front.get_stroke_opacity()), 0.8)

        binding.controller.display_mobject.set_opacity(0.5)
        binding.apply_section_compositing_frame(dimmed_outline)
        with patch.object(
            binding.controller,
            "_prepare_numeric",
            wraps=binding.controller._prepare_numeric,
        ) as prepare_numeric:
            binding.controller.update()
        prepare_numeric.assert_not_called()
        self.assertEqual(float(front.get_fill_opacity()), 0.0)
        self.assertAlmostEqual(float(front.get_stroke_opacity()), 0.1)

        binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_dynamic_compositing_axes_reach_the_real_controller_independently(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        timeline, banks, catalog, initial, shots = _inside_sphere_source()
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        surface_fill_slot = next(
            slot.slot_id
            for slot in catalog.slots
            if slot.role is SectionDisplayRole.SURFACE_FILL
        )
        invisible_occluder = compile_section_compositing(
            catalog,
            SectionCompositingInstruction.for_catalog(
                catalog,
                defaults=SectionCompositingAxes(
                    depth_presentation="diagrammatic",
                ),
                overrides=(
                    SectionCompositingOverride.for_slot(
                        surface_fill_slot,
                        display_opacity=0.0,
                    ),
                ),
            ),
        )
        opaque_paint_only = compile_section_compositing(
            catalog,
            SectionCompositingInstruction.for_catalog(
                catalog,
                defaults=SectionCompositingAxes(
                    depth_presentation="physical",
                ),
                overrides=(
                    SectionCompositingOverride.for_slot(
                        surface_fill_slot,
                        occlusion_participation="paint-only",
                    ),
                ),
            ),
        )
        compositing = tuple(
            invisible_occluder if index == 0 else opaque_paint_only
            for index, _sample in enumerate(timeline.samples)
        )
        binding = compile_parallel_section_rig_from_shots(
            scene,
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            compositing_frames=compositing,
            limits=_limits(),
            semantic_bank_ids=banks,
            frame_rate=4.0,
            plane_patch_margin=0.1,
        )

        binding._compositing_frame = binding.sequence.compositing_frames[0]
        self.assertEqual(
            binding._resolve_surface_opacities()["binding-sphere"],
            0.0,
        )
        self.assertEqual(
            binding._resolve_occluding_surface_ids(),
            ("binding-sphere",),
        )
        self.assertIs(
            binding._resolve_paint_policy(),
            QuadricPaintPolicy.DIAGRAMMATIC,
        )
        binding._compositing_frame = binding.sequence.compositing_frames[-1]
        self.assertEqual(
            binding._resolve_surface_opacities()["binding-sphere"],
            1.0,
        )
        self.assertEqual(binding._resolve_occluding_surface_ids(), ())
        self.assertIs(
            binding._resolve_paint_policy(),
            QuadricPaintPolicy.PHYSICAL,
        )
        binding._reset_to_first_frame()
        self.assertEqual(scene.mobjects, [])

    def test_direct_compositing_apply_rejects_unsupported_roles_and_mixed_policy(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        timeline, banks, catalog, initial, shots = _inside_sphere_source()
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
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
        )
        original = binding.snapshot_section_compositing_state()
        curve_slot = next(
            item.slot_id
            for item in catalog.slots
            if item.role is SectionDisplayRole.SECTION_CURVE
        )
        unsupported = compile_section_compositing(
            catalog,
            SectionCompositingInstruction.for_catalog(
                catalog,
                defaults=SectionCompositingAxes(
                    depth_presentation="diagrammatic"
                ),
                overrides=(
                    SectionCompositingOverride.for_slot(
                        curve_slot,
                        occlusion_participation="paint-only",
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(
            ParallelSectionRigBindingError,
            "supported only by the surface-fill slot",
        ):
            binding.apply_section_compositing_frame(unsupported)
        self.assertEqual(binding.snapshot_section_compositing_state(), original)

        mixed_policy = compile_section_compositing(
            catalog,
            SectionCompositingInstruction.for_catalog(
                catalog,
                defaults=SectionCompositingAxes(
                    depth_presentation="diagrammatic"
                ),
                overrides=(
                    SectionCompositingOverride.for_slot(
                        curve_slot,
                        depth_presentation="physical",
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(
            ParallelSectionRigBindingError,
            "one depth presentation policy",
        ):
            binding.apply_section_compositing_frame(mixed_policy)
        self.assertEqual(binding.snapshot_section_compositing_state(), original)
        self.assertEqual(scene.mobjects, [])

    def test_surface_and_plane_display_opacity_can_change_per_frame(self) -> None:
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

        binding = compile_parallel_section_rig_from_shots(
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
        binding.attach()
        identities = binding.controller.slot_identities()
        coordinator = binding.build_coordinator(scene.camera)
        coordinator.update(binding.sequence.frames[0])
        self.assertEqual(
            binding._resolve_surface_opacities()["binding-sphere"],
            1.0,
        )
        for frame in binding.sequence.frames[1:]:
            coordinator.update(frame)
        self.assertEqual(
            binding._resolve_surface_opacities()["binding-sphere"],
            0.0,
        )
        self.assertEqual(binding._resolve_section_plane_fill_opacity(), 0.0)
        self.assertEqual(binding.controller.slot_identities(), identities)
        coordinator.restore()
        binding.restore()
        self.assertEqual(scene.mobjects, [])

    def test_real_painter_preflight_uses_one_controller_and_reaches_line(self) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        _timeline, _initial, binding = _inside_sphere_fixture(scene)
        self.assertEqual(scene.mobjects, [])

    def test_nonidentity_viewport_moves_camera_and_fixed_slots_atomically(
        self,
    ) -> None:
        scene = ThreeDScene(camera_class=MultiProjectionCamera)
        timeline, banks, catalog, initial, shots = _inside_sphere_source()
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        transforms = tuple(
            ParallelScreenTransform(
                inherited_zoom=1.0 + 0.4 * index / (len(timeline.samples) - 1),
                frame_center=(0.5 * index, -0.25 * index),
                display_offset=(0.2 * index, 0.1 * index),
            )
            for index, _sample in enumerate(timeline.samples)
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
            screen_transforms=transforms,
        )
        scene.camera.set_zoom(0.75)
        scene.camera.frame_center[:] = (1.5, -1.0, 3.0)
        binding.controller.display_offset = (9.0, 9.0)
        binding.attach()
        identities = binding.controller.slot_identities()
        self.assertEqual(binding.controller.display_offset, (0.0, 0.0))

        coordinator = binding.build_coordinator(scene.camera)
        for frame, expected in zip(
            binding.sequence.frames,
            binding.sequence.screen_transforms,
        ):
            coordinator.update(frame)
            self.assertEqual(scene.camera.get_zoom(), expected.inherited_zoom)
            self.assertEqual(
                tuple(float(item) for item in scene.camera.frame_center[:2]),
                expected.frame_center,
            )
            self.assertEqual(
                binding.controller.display_offset,
                expected.display_offset,
            )
            self.assertEqual(binding.controller.slot_identities(), identities)

        coordinator.restore()
        self.assertEqual(scene.camera.get_zoom(), 0.75)
        self.assertEqual(
            tuple(float(item) for item in scene.camera.frame_center),
            (1.5, -1.0, 3.0),
        )
        self.assertEqual(binding.controller.display_offset, (0.0, 0.0))
        binding.restore()
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

    def test_isolated_point_activation_uses_fixed_point_slots(self) -> None:
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
        binding = compile_parallel_section_rig_from_shots(
            scene,
            timeline,
            shots,
            camera,
            tuple(display for _ in timeline.samples),
            limits=_limits(),
            semantic_bank_ids=banks,
            plane_patch_margin=0.1,
        )
        self.assertEqual(len(binding.allocated_point_ids), 2)
        self.assertEqual(scene.mobjects, [])
        point_frame_indices = tuple(
            index
            for index, frame in enumerate(binding.sequence.bank_render_frames)
            if any(layer.isolated_point_count for layer in frame.layers)
        )
        self.assertTrue(point_frame_indices)
        self.assertTrue(
            all(
                any(item.startswith("point:") for item in binding.sequence.painter_orders[index].draw_order)
                for index in point_frame_indices
            )
        )

        scene.camera.set_parallel_state(camera)
        binding.attach()
        identities = binding.controller.slot_identities()
        coordinator = binding.build_coordinator(scene.camera)
        observed_point_counts: list[int] = []
        for index, frame in enumerate(binding.sequence.frames):
            coordinator.update(frame)
            prepared = binding.controller._last_prepared_frame
            assert prepared is not None
            observed_point_counts.append(len(prepared.numeric.points))
            self.assertEqual(binding.controller.slot_identities(), identities)
            if index in point_frame_indices:
                self.assertGreater(len(prepared.numeric.points), 0)
                self.assertFalse(
                    any(
                        curve.curve_id in binding.allocated_point_ids
                        for curve in binding._curves
                    )
                )
        self.assertTrue(any(observed_point_counts))
        coordinator.restore()
        binding.restore()
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
