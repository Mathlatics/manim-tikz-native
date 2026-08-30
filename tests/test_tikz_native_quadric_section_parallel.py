from __future__ import annotations

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from polyhedron_visibility.quadrics.contract import (
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.parallel_plane_motion import (
    ParallelPlaneTranslation,
)
from polyhedron_visibility.quadrics.section_timeline import (
    compile_section_timeline,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary,
)
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayCatalog,
    SectionDisplayInstruction,
    SectionDisplayRole,
    SectionSemanticSlot,
    compile_section_display,
)
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_frame import (
    ParallelFrameCoordinator,
    ParallelFrameCoordinatorError,
    parallel_camera_frame_participant,
)
from tikz_native.parallel_preflight import (
    PainterOrderEvidence,
    ParallelPreflightLimits,
    ParallelPreflightRejectedError,
    ParallelSafeFrame,
)
from tikz_native.parallel_shots import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
    parallel_camera_shot_progress,
)
from tikz_native.quadric_section_parallel import (
    PARALLEL_SECTION_SEQUENCE_SCHEMA,
    PARALLEL_SCREEN_TRANSFORM_CHANNEL,
    SECTION_BANK_RENDER_CHANNEL,
    SECTION_PAINTER_ORDER_CHANNEL,
    SECTION_PLANE_PATCH_CHANNEL,
    SECTION_DISPLAY_CHANNEL,
    SECTION_PLANE_CHANNEL,
    SECTION_TIMELINE_FRAME_CHANNEL,
    SECTION_TOPOLOGY_BANK_CHANNEL,
    SECTION_TRANSITION_STATE_CHANNEL,
    ParallelCameraShotSamplePhase,
    ParallelSectionSequenceError,
    compile_parallel_section_preflight_frames,
    compile_parallel_section_sequence_from_shots,
    parallel_screen_transform_guard,
    parallel_camera_shot_frame_times,
    parallel_section_frame_grid,
    parallel_section_preflight_gate,
    sample_parallel_camera_shot_sequence,
    section_bank_frame_participant,
    section_display_frame_participant,
    section_painter_order_participant,
)
from tikz_native.section_bank_render import SectionBankRenderFrame


def _timeline():
    sphere = SphereSpec("joint-sphere", (0.0, 0.0, 0.0), 1.0)
    plane = SectionPlane(
        "joint-plane",
        (0.0, 0.0, -2.0),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    return compile_section_timeline(
        "joint-section",
        sphere,
        (
            ParallelPlaneTranslation(
                "joint-translation",
                plane,
                (0.0, 0.0, 4.0),
                start_time=0.0,
                end_time=2.0,
            ),
        ),
    )


def _display_catalog() -> SectionDisplayCatalog:
    return SectionDisplayCatalog(
        "joint-section",
        (
            SectionSemanticSlot(
                "joint:surface-outline",
                SectionDisplayRole.SURFACE_OUTLINE,
            ),
            SectionSemanticSlot(
                "joint:section-bank-a:0",
                SectionDisplayRole.SECTION_CURVE,
                topology_bank="semantic-a",
            ),
            SectionSemanticSlot(
                "joint:section-bank-a:1",
                SectionDisplayRole.SECTION_CURVE,
                topology_bank="semantic-a",
            ),
            SectionSemanticSlot(
                "joint:section-bank-b:0",
                SectionDisplayRole.SECTION_CURVE,
                topology_bank="semantic-b",
            ),
            SectionSemanticSlot(
                "joint:section-bank-b:1",
                SectionDisplayRole.SECTION_CURVE,
                topology_bank="semantic-b",
            ),
            SectionSemanticSlot(
                "joint:point-bank-a:0",
                SectionDisplayRole.SECTION_POINT,
                topology_bank="semantic-a",
            ),
            SectionSemanticSlot(
                "joint:point-bank-b:0",
                SectionDisplayRole.SECTION_POINT,
                topology_bank="semantic-b",
            ),
        ),
    )


def _camera_samples(timeline):
    initial = ParallelCameraState.from_view_direction(
        (1.0, 1.0, 1.0),
        target=(0.0, 0.0, 0.0),
        zoom=0.8,
    )
    endpoint = ParallelCameraState.along_plane(
        timeline.samples[0].plane,
        direction=(1.0, 0.0, 0.0),
        target=(0.0, 0.0, 0.0),
        zoom=0.8,
    )
    sequence = ParallelCameraShotSequence(
        (
            ParallelCameraShot(
                "joint-side-view",
                endpoint,
                duration=2.0,
                transition="orbit",
                arc_height=0.6,
            ),
        )
    )
    samples = sample_parallel_camera_shot_sequence(
        sequence,
        initial,
        tuple(item.time for item in timeline.samples),
    )
    return initial, endpoint, sequence, samples


def _limits(**overrides: object) -> ParallelPreflightLimits:
    values: dict[str, object] = {
        "safe_frame": ParallelSafeFrame(-10.0, 10.0, -10.0, 10.0),
        "min_zoom": 0.25,
        "max_zoom": 2.0,
        "require_framing_points": True,
    }
    values.update(overrides)
    return ParallelPreflightLimits(**values)  # type: ignore[arg-type]


def _painter() -> PainterOrderEvidence:
    return PainterOrderEvidence(
        item_ids=("surface", "plane", "section"),
        relations=(("surface", "plane"), ("plane", "section")),
        draw_order=("surface", "plane", "section"),
    )


def _painter_provider(
    _time: float,
    _camera: ParallelCameraState,
    _plane: SectionPlane,
) -> PainterOrderEvidence:
    return _painter()


class _CameraTarget:
    def __init__(self, state: ParallelCameraState) -> None:
        self.state = state

    def snapshot_parallel_state(self) -> ParallelCameraState:
        return self.state

    def set_parallel_state(self, state: ParallelCameraState) -> None:
        self.state = state


class _DisplayTarget:
    def __init__(self) -> None:
        self.state: object = "baseline"
        self.applied = []

    def snapshot_section_display_state(self) -> object:
        return self.state

    def apply_section_display_frame(self, frame: object) -> None:
        self.state = frame
        self.applied.append(frame)

    def restore_section_display_state(self, state: object) -> None:
        self.state = state


class _BankTarget:
    def __init__(self) -> None:
        self.state: object = "bank-baseline"
        self.applied: list[SectionBankRenderFrame] = []

    def snapshot_section_bank_render_state(self) -> object:
        return self.state

    def apply_section_bank_render_frame(self, frame: SectionBankRenderFrame) -> None:
        self.state = frame
        self.applied.append(frame)

    def restore_section_bank_render_state(self, state: object) -> None:
        self.state = state


class _PainterTarget:
    def __init__(self) -> None:
        self.state: object = "painter-baseline"
        self.applied: list[PainterOrderEvidence] = []

    def snapshot_section_painter_order_state(self) -> object:
        return self.state

    def apply_section_painter_order(self, value: PainterOrderEvidence) -> None:
        self.state = value
        self.applied.append(value)

    def restore_section_painter_order_state(self, state: object) -> None:
        self.state = state


class QuadricSectionParallelIntegrationTests(unittest.TestCase):
    def test_shot_sampler_matches_endpoints_holds_and_strict_time_range(self) -> None:
        initial = ParallelCameraState(np.identity(3))
        first_state = ParallelCameraState.from_view_direction(
            (1.0, 0.0, 1.0),
            target=(1.0, 0.0, 0.0),
        )
        second_state = ParallelCameraState.from_view_direction(
            (0.0, 1.0, 1.0),
            target=(0.0, 1.0, 0.0),
        )
        sequence = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "first",
                    first_state,
                    duration=1.0,
                    hold=0.5,
                    transition="orbit",
                ),
                ParallelCameraShot(
                    "second",
                    second_state,
                    duration=0.5,
                    transition="shortest",
                ),
            )
        )
        samples = sample_parallel_camera_shot_sequence(
            sequence,
            initial,
            (0.0, 0.5, 1.0, 1.25, 1.5, 2.0),
        )
        self.assertEqual(
            tuple(item.phase for item in samples),
            (
                ParallelCameraShotSamplePhase.TRANSITION,
                ParallelCameraShotSamplePhase.TRANSITION,
                ParallelCameraShotSamplePhase.ENDPOINT,
                ParallelCameraShotSamplePhase.HOLD,
                ParallelCameraShotSamplePhase.HOLD,
                ParallelCameraShotSamplePhase.ENDPOINT,
            ),
        )
        self.assertIs(samples[2].state, first_state)
        self.assertIs(samples[3].state, first_state)
        self.assertIs(samples[-1].state, second_state)
        self.assertAlmostEqual(
            parallel_camera_shot_progress(0.25),
            0.07010371654510815,
            places=15,
        )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "inside the authored camera sequence",
        ):
            sample_parallel_camera_shot_sequence(
                sequence,
                initial,
                (-0.1, 0.0),
            )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "below absolute-time numeric resolution",
        ):
            sample_parallel_camera_shot_sequence(
                sequence,
                initial,
                (1.0e20,),
                start_time=1.0e20,
            )

    def test_joint_sequence_preflights_then_commits_camera_and_display(self) -> None:
        timeline = _timeline()
        initial, endpoint, shots, _camera_keyframes = _camera_samples(timeline)
        catalog = _display_catalog()
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        display_frames = tuple(display for _ in timeline.samples)
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            display_frames,
            limits=_limits(),
            painter_orders=_painter_provider,
            semantic_bank_ids=("semantic-a", "semantic-b"),
            frame_rate=30.0,
        )

        self.assertEqual(sequence.schema, PARALLEL_SECTION_SEQUENCE_SCHEMA)
        self.assertTrue(sequence.preflight_report.accepted)
        self.assertEqual(
            sequence.preflight_report.frame_ids,
            tuple(item.frame_id for item in sequence.preflight_frames),
        )
        self.assertGreater(len(sequence.frames), 20)
        self.assertEqual(
            sequence.evaluation_times,
            tuple(item.time for item in sequence.camera_samples),
        )
        self.assertTrue(
            any(len(item.layers) == 2 for item in sequence.bank_render_frames)
        )
        self.assertTrue(
            any(
                layer.isolated_point_count == 1
                for item in sequence.bank_render_frames
                for layer in item.layers
            )
        )
        self.assertEqual(
            sequence.evaluation_times,
            parallel_section_frame_grid(
                sequence.transition_plan,
                30.0,
                shot_sequence=shots,
                start_time=0.0,
            ),
        )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "frame_rate is too low",
        ):
            parallel_section_frame_grid(
                sequence.transition_plan,
                12.0,
                shot_sequence=shots,
                start_time=0.0,
            )

        event_frames = tuple(
            item
            for item in sequence.preflight_frames
            if item.topology_events
        )
        self.assertEqual(
            sum(len(item.topology_events) for item in event_frames),
            len(timeline.animation.topology_events),
        )
        for frame in event_frames:
            capacity_ids = {item.resource_id for item in frame.capacities}
            for event in frame.topology_events:
                if event.requires_slot_bank:
                    self.assertIn(event.slot_bank_id, capacity_ids)
        resource_sets = {
            tuple(item.resource_id for item in frame.capacities)
            for frame in sequence.preflight_frames
        }
        self.assertEqual(len(resource_sets), 1)

        camera = _CameraTarget(initial)
        display_target = _DisplayTarget()
        bank_target = _BankTarget()
        painter_target = _PainterTarget()
        gate = parallel_section_preflight_gate(sequence)
        coordinator = ParallelFrameCoordinator()
        coordinator.add(gate.participant())
        coordinator.add(
            parallel_screen_transform_guard(
                lambda: sequence.screen_transforms[gate.next_frame_index]
            )
        )
        coordinator.add(parallel_camera_frame_participant(camera))
        coordinator.add(section_bank_frame_participant(bank_target))
        coordinator.add(section_painter_order_participant(painter_target))
        coordinator.add(section_display_frame_participant(display_target))
        for index, frame in enumerate(sequence.frames):
            coordinator.update(frame)
            self.assertIsNotNone(
                frame.channel(SECTION_TIMELINE_FRAME_CHANNEL),
            )
            self.assertEqual(
                frame.channel(  # type: ignore[union-attr]
                    SECTION_PLANE_CHANNEL
                ).plane_id,
                timeline.plane_id,
            )
            self.assertIs(frame.channel(SECTION_DISPLAY_CHANNEL), display)
            self.assertEqual(
                set(frame.channel(SECTION_TOPOLOGY_BANK_CHANNEL)),
                {
                    item.bank_index
                    for item in frame.channel(  # type: ignore[union-attr]
                        SECTION_BANK_RENDER_CHANNEL
                    ).layers
                },
            )
            transition_state = frame.channel(SECTION_TRANSITION_STATE_CHANNEL)
            self.assertEqual(
                transition_state.time,  # type: ignore[attr-defined]
                sequence.evaluation_times[index],
            )
            self.assertIs(
                frame.channel(PARALLEL_SCREEN_TRANSFORM_CHANNEL),
                sequence.screen_transforms[index],
            )
            self.assertIs(
                frame.channel(SECTION_PAINTER_ORDER_CHANNEL),
                sequence.painter_orders[index],
            )
        self.assertEqual(gate.next_frame_index, len(sequence.frames))
        self.assertIs(camera.state, endpoint)
        self.assertIs(display_target.state, display)
        self.assertEqual(len(display_target.applied), len(sequence.frames))
        self.assertIs(bank_target.state, sequence.bank_render_frames[-1])
        self.assertEqual(len(bank_target.applied), len(sequence.frames))
        self.assertIs(painter_target.state, sequence.painter_orders[-1])
        self.assertEqual(len(painter_target.applied), len(sequence.frames))

        coordinator.restore()
        self.assertEqual(gate.next_frame_index, 0)
        self.assertIs(camera.state, initial)
        self.assertEqual(display_target.state, "baseline")
        self.assertEqual(bank_target.state, "bank-baseline")
        self.assertEqual(painter_target.state, "painter-baseline")
        payload = json.loads(sequence.to_json())
        self.assertEqual(payload["schema"], PARALLEL_SECTION_SEQUENCE_SCHEMA)
        self.assertTrue(sequence.digest.startswith("sha256:"))

    def test_shot_frame_grid_restarts_local_clock_at_each_play(self) -> None:
        state = ParallelCameraState(np.identity(3))
        sequence = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "short-a",
                    state,
                    duration=0.25,
                    transition="shortest",
                ),
                ParallelCameraShot(
                    "short-b",
                    state.with_target((1.0, 0.0, 0.0)),
                    duration=0.25,
                    transition="shortest",
                ),
            )
        )
        self.assertEqual(
            parallel_camera_shot_frame_times(
                sequence,
                start_time=0.0,
                frame_rate=10.0,
            ),
            (0.0, 0.1, 0.2, 0.25, 0.35, 0.45),
        )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "absolute times are too large",
        ):
            parallel_camera_shot_frame_times(
                sequence,
                start_time=1.0e14,
                frame_rate=30.0,
            )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "absolute times are too large",
        ):
            sample_parallel_camera_shot_sequence(
                sequence,
                state,
                (1.0e14, 1.0e14 + 0.25, 1.0e14 + 0.5),
                start_time=1.0e14,
            )

    def test_joint_compiler_fails_before_frames_escape_rejected_preflight(self) -> None:
        timeline = _timeline()
        initial, _endpoint, shots, _camera_keyframes = _camera_samples(timeline)
        display = compile_section_display(
            _display_catalog(),
            SectionDisplayInstruction.for_mode("painted"),
        )
        with self.assertRaises(ParallelPreflightRejectedError):
            compile_parallel_section_sequence_from_shots(
                timeline,
                shots,
                initial,
                tuple(display for _ in timeline.samples),
                limits=_limits(
                    safe_frame=ParallelSafeFrame(-0.01, 0.01, -0.01, 0.01)
                ),
                painter_orders=_painter_provider,
                semantic_bank_ids=("semantic-a", "semantic-b"),
            )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "framing point must contain three finite values",
        ):
            compile_parallel_section_sequence_from_shots(
                timeline,
                shots,
                initial,
                tuple(display for _ in timeline.samples),
                limits=_limits(),
                painter_orders=_painter_provider,
                semantic_bank_ids=("semantic-a", "semantic-b"),
                framing_points_by_frame=tuple(
                    (("not-a-number", 0.0, 0.0),)
                    for _ in timeline.samples
                ),
            )

    def test_runtime_channel_tampering_and_live_transform_drift_fail_closed(
        self,
    ) -> None:
        timeline = _timeline()
        initial, _endpoint, shots, _samples = _camera_samples(timeline)
        display = compile_section_display(
            _display_catalog(),
            SectionDisplayInstruction.for_mode("painted"),
        )
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=_limits(),
            painter_orders=_painter_provider,
            semantic_bank_ids=("semantic-a", "semantic-b"),
            frame_rate=60.0,
        )
        first = sequence.frames[0]
        tampered = replace(
            first,
            channels={**first.channels, SECTION_PLANE_CHANNEL: "forged-plane"},
        )
        gate = parallel_section_preflight_gate(sequence)
        coordinator = ParallelFrameCoordinator()
        coordinator.add(gate.participant())
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "section-evaluation-plane.*cannot be canonically verified",
        ):
            coordinator.update(tampered)
        self.assertEqual(gate.next_frame_index, 0)

        live_transform = replace(
            sequence.screen_transforms[0],
            inherited_zoom=1.1,
        )
        transform_gate = parallel_section_preflight_gate(sequence)
        transform_coordinator = ParallelFrameCoordinator()
        transform_coordinator.add(transform_gate.participant())
        transform_coordinator.add(
            parallel_screen_transform_guard(lambda: live_transform)
        )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "live renderer screen transform differs",
        ):
            transform_coordinator.update(first)
        self.assertEqual(transform_gate.next_frame_index, 0)

    def test_sequence_freezes_inputs_and_rejects_top_level_tampering(self) -> None:
        timeline = _timeline()
        initial, _endpoint, shots, _samples = _camera_samples(timeline)
        catalog = _display_catalog()
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=_limits(),
            painter_orders=_painter_provider,
            semantic_bank_ids=("semantic-a", "semantic-b"),
            frame_rate=30.0,
        )
        self.assertIsNotNone(sequence.camera_provenance)
        self.assertEqual(sequence.camera_provenance.coverage, "exact")
        self.assertEqual(
            sequence.camera_provenance.easing,
            "manim-smooth-v1",
        )

        mutable_displays = list(sequence.display_frames)
        frozen_copy = replace(sequence, display_frames=mutable_displays)
        mutable_displays.append(display)
        self.assertEqual(
            len(frozen_copy.display_frames),
            len(sequence.display_frames),
        )
        with self.assertRaises(ParallelSectionSequenceError):
            replace(
                sequence,
                evaluation_times=tuple(
                    item + 100.0 for item in sequence.evaluation_times
                ),
            )
        shifted_provenance = replace(
            sequence.camera_provenance,
            nominal_frame_times=tuple(
                item + 0.1 / sequence.camera_provenance.frame_rate
                for item in sequence.camera_provenance.nominal_frame_times
            ),
        )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "nominal camera grid was not derived",
        ):
            replace(sequence, camera_provenance=shifted_provenance)
        with self.assertRaises(ParallelSectionSequenceError):
            replace(
                sequence,
                evaluation_times=(
                    float("nan"),
                    *sequence.evaluation_times[1:],
                ),
            )
        outline = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("outline-only"),
        )
        with self.assertRaises(ParallelSectionSequenceError):
            replace(
                sequence,
                display_frames=tuple(outline for _ in sequence.display_frames),
            )
        changed_camera = replace(sequence.camera_samples[0], time=0.001)
        with self.assertRaises(ParallelSectionSequenceError):
            replace(
                sequence,
                camera_samples=(changed_camera, *sequence.camera_samples[1:]),
            )
        for changes in (
            {"sample_id": sequence.camera_samples[1].sample_id},
            {"shot_id": "forged-shot"},
            {"phase": ParallelCameraShotSamplePhase.HOLD},
        ):
            with self.subTest(changes=changes):
                forged = replace(sequence.camera_samples[0], **changes)
                with self.assertRaises(ParallelSectionSequenceError):
                    replace(
                        sequence,
                        camera_samples=(forged, *sequence.camera_samples[1:]),
                    )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "does not describe shot_sequence",
        ):
            replace(
                sequence.camera_provenance,
                sequence_digest="sha256:" + "0" * 64,
            )
        changed_initial = sequence.camera_provenance.initial_camera.with_zoom(1.1)
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "not derived from source provenance",
        ):
            replace(
                sequence,
                camera_provenance=replace(
                    sequence.camera_provenance,
                    initial_camera=changed_initial,
                ),
            )

    def test_analytic_time_near_shot_boundary_keeps_nominal_mapping(self) -> None:
        end_time = 2.0000000000000004
        surface = SphereSpec("joint-sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "joint-plane",
            (0.0, 0.0, -2.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        timeline = compile_section_timeline(
            "joint-section",
            surface,
            (
                ParallelPlaneTranslation(
                    "boundary-motion",
                    plane,
                    (0.0, 0.0, 4.0),
                    start_time=0.0,
                    end_time=end_time,
                ),
            ),
        )
        initial = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
        midpoint = initial.with_target((0.2, 0.0, 0.0))
        endpoint = initial.with_target((0.4, 0.0, 0.0))
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "boundary-first",
                    midpoint,
                    duration=1.0,
                    transition="shortest",
                ),
                ParallelCameraShot(
                    "boundary-second",
                    endpoint,
                    duration=end_time - 1.0,
                    transition="shortest",
                ),
            )
        )
        display = compile_section_display(
            _display_catalog(),
            SectionDisplayInstruction.for_mode("painted"),
        )
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=_limits(),
            painter_orders=_painter_provider,
            semantic_bank_ids=("semantic-a", "semantic-b"),
            frame_rate=30.0,
        )
        analytic_boundary = 1.0000000000000002
        index = sequence.evaluation_times.index(analytic_boundary)
        self.assertEqual(sequence.camera_provenance.nominal_frame_times[index], 1.0)
        self.assertEqual(
            sequence.camera_samples[index].phase,
            ParallelCameraShotSamplePhase.ENDPOINT,
        )
        self.assertEqual(sequence.camera_samples[index].shot_id, "boundary-first")
        self.assertIs(sequence.camera_samples[index].state, midpoint)

    def test_camera_source_coverage_is_explicit(self) -> None:
        timeline = _timeline()
        initial = ParallelCameraState(np.identity(3))
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "long-shot",
                    initial.with_target((0.5, 0.0, 0.0)),
                    duration=3.0,
                    transition="shortest",
                ),
            )
        )
        display = compile_section_display(
            _display_catalog(),
            SectionDisplayInstruction.for_mode("painted"),
        )
        arguments = {
            "limits": _limits(),
            "painter_orders": _painter_provider,
            "semantic_bank_ids": ("semantic-a", "semantic-b"),
        }
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "exact camera coverage",
        ):
            compile_parallel_section_sequence_from_shots(
                timeline,
                shots,
                initial,
                tuple(display for _ in timeline.samples),
                **arguments,
            )
        window = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            coverage="window",
            **arguments,
        )
        self.assertEqual(window.camera_provenance.coverage, "window")
        self.assertEqual(window.camera_provenance.end_time, 3.0)

    def test_finite_plane_patch_is_source_bound_and_enters_framing(self) -> None:
        timeline = _timeline()
        initial, _endpoint, shots, _samples = _camera_samples(timeline)
        plane_catalog = SectionDisplayCatalog(
            timeline.section_id,
            (
                *_display_catalog().slots,
                SectionSemanticSlot(
                    "joint:plane-outline",
                    SectionDisplayRole.PLANE_OUTLINE,
                ),
            ),
        )
        visible_plane = compile_section_display(
            plane_catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "visible plane display slots require",
        ):
            compile_parallel_section_sequence_from_shots(
                timeline,
                shots,
                initial,
                tuple(visible_plane for _ in timeline.samples),
                limits=_limits(),
                painter_orders=_painter_provider,
                semantic_bank_ids=("semantic-a", "semantic-b"),
            )
        display = compile_section_display(
            plane_catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=_limits(),
            painter_orders=_painter_provider,
            semantic_bank_ids=("semantic-a", "semantic-b"),
            plane_patch_margin=0.08,
        )
        self.assertTrue(all(item is not None for item in sequence.plane_patch_fits))
        for fit, evidence, frame in zip(
            sequence.plane_patch_fits,
            sequence.preflight_frames,
            sequence.frames,
        ):
            self.assertIsNotNone(fit)
            self.assertEqual(fit.margin_ratio, 0.08)
            self.assertEqual(
                set(fit.patch.corners(fit.plane)),
                set(fit.patch.corners(fit.plane)) & set(evidence.framing_points),
            )
            self.assertIs(frame.channel(SECTION_PLANE_PATCH_CHANNEL), fit)

        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "stored plane patch fits differ",
        ):
            replace(sequence, plane_patch_margin=0.1)
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "stored plane patch fits differ",
        ):
            replace(
                sequence,
                plane_patch_fits=(None, *sequence.plane_patch_fits[1:]),
            )

    def test_plane_patch_overflow_and_exact_side_view_are_certified(self) -> None:
        timeline = _timeline()
        display = compile_section_display(
            _display_catalog(),
            SectionDisplayInstruction.for_mode("painted"),
        )
        normal = ParallelCameraState.normal_to_plane(timeline.samples[0].plane)
        normal_shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "normal-static",
                    normal,
                    duration=2.0,
                    transition="shortest",
                ),
            )
        )
        tight_limits = _limits(
            safe_frame=ParallelSafeFrame(-1.05, 1.05, -1.05, 1.05)
        )
        without_patch = compile_parallel_section_sequence_from_shots(
            timeline,
            normal_shots,
            normal,
            tuple(display for _ in timeline.samples),
            limits=tight_limits,
            painter_orders=_painter_provider,
            semantic_bank_ids=("semantic-a", "semantic-b"),
        )
        self.assertTrue(all(item is None for item in without_patch.plane_patch_fits))
        with self.assertRaises(ParallelPreflightRejectedError):
            compile_parallel_section_sequence_from_shots(
                timeline,
                normal_shots,
                normal,
                tuple(display for _ in timeline.samples),
                limits=tight_limits,
                painter_orders=_painter_provider,
                semantic_bank_ids=("semantic-a", "semantic-b"),
                plane_patch_margin=0.08,
            )

        side = ParallelCameraState.along_plane(
            timeline.samples[0].plane,
            direction=(1.0, 0.0, 0.0),
        )
        side_shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "side-static",
                    side,
                    duration=2.0,
                    transition="shortest",
                ),
            )
        )
        side_sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            side_shots,
            side,
            tuple(display for _ in timeline.samples),
            limits=_limits(),
            painter_orders=_painter_provider,
            semantic_bank_ids=("semantic-a", "semantic-b"),
            plane_patch_margin=0.08,
        )
        fit = side_sequence.plane_patch_fits[0]
        projected = side.project_points(fit.patch.corners(fit.plane))[:, :2]
        centered = projected - np.mean(projected, axis=0)
        self.assertEqual(np.linalg.matrix_rank(centered, tol=1.0e-12), 1)

    def test_semantic_bank_capacity_is_bound_to_real_catalog_slots(self) -> None:
        timeline = _timeline()
        initial, _endpoint, shots, _samples = _camera_samples(timeline)
        undersized = SectionDisplayCatalog(
            "joint-section",
            (
                SectionSemanticSlot(
                    "joint:only-a",
                    SectionDisplayRole.SECTION_CURVE,
                    topology_bank="semantic-a",
                ),
                SectionSemanticSlot(
                    "joint:only-b",
                    SectionDisplayRole.SECTION_CURVE,
                    topology_bank="semantic-b",
                ),
            ),
        )
        display = compile_section_display(
            undersized,
            SectionDisplayInstruction.for_mode("painted"),
        )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "fixed contract requires 2",
        ):
            compile_parallel_section_sequence_from_shots(
                timeline,
                shots,
                initial,
                tuple(display for _ in timeline.samples),
                limits=_limits(),
                painter_orders=_painter_provider,
                semantic_bank_ids=("semantic-a", "semantic-b"),
            )

    def test_public_preflight_compiler_rejects_forged_bank_payload(self) -> None:
        timeline = _timeline()
        initial, _endpoint, shots, _samples = _camera_samples(timeline)
        display = compile_section_display(
            _display_catalog(),
            SectionDisplayInstruction.for_mode("painted"),
        )
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=_limits(),
            painter_orders=_painter_provider,
            semantic_bank_ids=("semantic-a", "semantic-b"),
        )
        first_bank = sequence.bank_render_frames[0]
        forged_layer = replace(
            first_bank.layers[0],
            semantic_bank_id="undeclared-bank",
            active_cap_chord_ids=("undeclared-cap",),
        )
        forged_bank = replace(first_bank, layers=(forged_layer,))
        bank_frames = (forged_bank, *sequence.bank_render_frames[1:])
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "bank render frame differs",
        ):
            compile_parallel_section_preflight_frames(
                timeline,
                sequence.camera_samples,
                sequence.display_frames,
                bank_frames,
                transition_plan=sequence.transition_plan,
                plane_patch_margin=sequence.plane_patch_margin,
                plane_patch_fits=sequence.plane_patch_fits,
                transition_states=tuple(
                    frame.channel(SECTION_TRANSITION_STATE_CHANNEL)
                    for frame in sequence.frames
                ),
                planes=tuple(
                    frame.channel(SECTION_PLANE_CHANNEL)
                    for frame in sequence.frames
                ),
                timeline_frames=tuple(
                    frame.channel(SECTION_TIMELINE_FRAME_CHANNEL)
                    for frame in sequence.frames
                ),
                topology_banks=tuple(
                    frame.channel(SECTION_TOPOLOGY_BANK_CHANNEL)
                    for frame in sequence.frames
                ),
                painter_orders=sequence.painter_orders,
                screen_transforms=sequence.screen_transforms,
                framing_points_by_frame=tuple(
                    item.framing_points for item in sequence.preflight_frames
                ),
                semantic_bank_ids=sequence.semantic_bank_ids,
            )
        transitions = tuple(
            frame.channel(SECTION_TRANSITION_STATE_CHANNEL)
            for frame in sequence.frames
        )
        forged_transition = replace(
            transitions[1],
            layers=tuple(
                replace(layer, geometry_time=timeline.samples[0].time)
                for layer in transitions[1].layers
            ),
        )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "canonical transition plan",
        ):
            compile_parallel_section_preflight_frames(
                timeline,
                sequence.camera_samples,
                sequence.display_frames,
                sequence.bank_render_frames,
                transition_plan=sequence.transition_plan,
                plane_patch_margin=sequence.plane_patch_margin,
                plane_patch_fits=sequence.plane_patch_fits,
                transition_states=(
                    transitions[0],
                    forged_transition,
                    *transitions[2:],
                ),
                planes=tuple(
                    frame.channel(SECTION_PLANE_CHANNEL)
                    for frame in sequence.frames
                ),
                timeline_frames=tuple(
                    frame.channel(SECTION_TIMELINE_FRAME_CHANNEL)
                    for frame in sequence.frames
                ),
                topology_banks=tuple(
                    frame.channel(SECTION_TOPOLOGY_BANK_CHANNEL)
                    for frame in sequence.frames
                ),
                painter_orders=sequence.painter_orders,
                screen_transforms=sequence.screen_transforms,
                framing_points_by_frame=tuple(
                    item.framing_points for item in sequence.preflight_frames
                ),
                semantic_bank_ids=sequence.semantic_bank_ids,
            )

    def test_cap_chord_slots_activation_and_events_enter_joint_preflight(self) -> None:
        cylinder = CylinderSpec(
            "joint-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "joint-cylinder-plane",
            (0.0, 0.0, -3.0),
            (1.0, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        timeline = compile_section_timeline(
            "joint-cylinder-section",
            cylinder,
            (
                ParallelPlaneTranslation(
                    "joint-cylinder-motion",
                    plane,
                    (0.0, 0.0, 6.0),
                    start_time=0.0,
                    end_time=6.0,
                ),
            ),
        )
        slots = [
            SectionSemanticSlot(
                f"joint-cylinder:{bank}:{slot}",
                SectionDisplayRole.SECTION_CURVE,
                topology_bank=bank,
            )
            for bank in ("semantic-a", "semantic-b")
            for slot in range(2)
        ]
        slots.extend(
            SectionSemanticSlot(
                f"joint-cylinder:cap:{bank}:{index}",
                SectionDisplayRole.CAP_CHORD,
                source_id=source_id,
                topology_bank=bank,
            )
            for bank in ("semantic-a", "semantic-b")
            for index, source_id in enumerate(timeline.cap_chord_ids)
        )
        slots.extend(
            SectionSemanticSlot(
                f"joint-cylinder:point:{bank}",
                SectionDisplayRole.SECTION_POINT,
                topology_bank=bank,
            )
            for bank in ("semantic-a", "semantic-b")
        )
        catalog = SectionDisplayCatalog(timeline.section_id, tuple(slots))
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        camera = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "joint-cylinder-shot",
                    camera,
                    duration=6.0,
                    transition="shortest",
                ),
            )
        )
        unbanked_catalog = SectionDisplayCatalog(
            timeline.section_id,
            tuple(
                replace(slot, topology_bank=None)
                if slot.role is SectionDisplayRole.CAP_CHORD
                else slot
                for slot in slots
            ),
        )
        unbanked_display = compile_section_display(
            unbanked_catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        with self.assertRaisesRegex(
            ParallelSectionSequenceError,
            "every cap-chord slot must belong",
        ):
            compile_parallel_section_sequence_from_shots(
                timeline,
                shots,
                camera,
                tuple(unbanked_display for _ in timeline.samples),
                limits=_limits(),
                painter_orders=_painter_provider,
                semantic_bank_ids=("semantic-a", "semantic-b"),
            )
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            camera,
            tuple(display for _ in timeline.samples),
            limits=_limits(),
            painter_orders=_painter_provider,
            semantic_bank_ids=("semantic-a", "semantic-b"),
        )
        self.assertEqual(
            {
                event.event_id
                for frame in sequence.preflight_frames
                for event in frame.topology_events
                if event.event_id
                in {item.event_id for item in timeline.cap_chord_events}
            },
            {item.event_id for item in timeline.cap_chord_events},
        )
        for bank_frame, evidence in zip(
            sequence.bank_render_frames,
            sequence.preflight_frames,
        ):
            capacities = {item.resource_id: item for item in evidence.capacities}
            layer_by_bank = {
                layer.semantic_bank_id: layer for layer in bank_frame.layers
            }
            for bank in ("semantic-a", "semantic-b"):
                for source_id in timeline.cap_chord_ids:
                    resource_id = f"{bank}:cap-chord:{source_id}"
                    active = (
                        source_id in layer_by_bank[bank].active_cap_chord_ids
                        if bank in layer_by_bank
                        else False
                    )
                    self.assertEqual(
                        capacities[resource_id].used,
                        int(active),
                    )
            for layer in bank_frame.layers:
                progress = layer.geometry_time / 6.0
                geometry_plane = timeline.segment_schedules[0].motion.plane_at(
                    progress
                )
                boundary = compute_quadric_section_boundary(
                    timeline.section_id,
                    cylinder,
                    geometry_plane,
                )
                self.assertEqual(
                    layer.active_cap_chord_ids,
                    tuple(sorted(item.curve_id for item in boundary.cap_chords)),
                )
        self.assertTrue(
            all(
                right - left > 1.0e-12
                for left, right in zip(
                    sequence.evaluation_times,
                    sequence.evaluation_times[1:],
                )
            )
        )

    def test_bank_geometry_reuses_timeline_coefficient_policy(self) -> None:
        cylinder = CylinderSpec(
            "policy-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-10.0, 10.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "policy-plane",
            (0.0, 0.0, -0.2),
            (0.01, 0.0, 1.0),
            u_axis=(1.0, 0.0, -0.01),
        )
        timeline = compile_section_timeline(
            "policy-section",
            cylinder,
            (
                ParallelPlaneTranslation(
                    "policy-motion",
                    plane,
                    (0.0, 0.0, 0.4),
                    start_time=0.0,
                    end_time=1.0,
                ),
            ),
            coefficient_tolerance=0.001,
        )
        custom = compute_quadric_section_boundary(
            timeline.section_id,
            cylinder,
            plane,
            coefficient_tolerance=0.001,
        )
        default = compute_quadric_section_boundary(
            timeline.section_id,
            cylinder,
            plane,
        )
        self.assertEqual(custom.trace.supporting_kind.value, "circle")
        self.assertEqual(default.trace.supporting_kind.value, "ellipse")
        self.assertEqual(timeline.coefficient_tolerance, 0.001)
        self.assertEqual(timeline.to_dict()["coefficientTolerance"], 0.001)
        with self.assertRaisesRegex(
            ValueError,
            "geometry_policy_digest",
        ):
            replace(timeline, coefficient_tolerance=None)

        slots = [
            SectionSemanticSlot(
                f"policy:{bank}:curve:{index}",
                SectionDisplayRole.SECTION_CURVE,
                topology_bank=bank,
            )
            for bank in ("semantic-a", "semantic-b")
            for index in range(2)
        ]
        slots.extend(
            SectionSemanticSlot(
                f"policy:{bank}:point",
                SectionDisplayRole.SECTION_POINT,
                topology_bank=bank,
            )
            for bank in ("semantic-a", "semantic-b")
        )
        slots.extend(
            SectionSemanticSlot(
                f"policy:{bank}:cap:{index}",
                SectionDisplayRole.CAP_CHORD,
                source_id=source_id,
                topology_bank=bank,
            )
            for bank in ("semantic-a", "semantic-b")
            for index, source_id in enumerate(timeline.cap_chord_ids)
        )
        display = compile_section_display(
            SectionDisplayCatalog(timeline.section_id, tuple(slots)),
            SectionDisplayInstruction.for_mode("painted"),
        )
        camera = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "policy-static",
                    camera,
                    duration=1.0,
                    transition="shortest",
                ),
            )
        )
        with patch(
            "tikz_native.quadric_section_parallel."
            "compute_quadric_section_boundary",
            wraps=compute_quadric_section_boundary,
        ) as solver:
            sequence = compile_parallel_section_sequence_from_shots(
                timeline,
                shots,
                camera,
                tuple(display for _ in timeline.samples),
                limits=_limits(),
                painter_orders=_painter_provider,
                semantic_bank_ids=("semantic-a", "semantic-b"),
            )
        self.assertTrue(sequence.bank_render_frames)
        self.assertTrue(solver.call_args_list)
        self.assertTrue(
            all(
                call.kwargs["coefficient_tolerance"] == 0.001
                for call in solver.call_args_list
            )
        )

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
