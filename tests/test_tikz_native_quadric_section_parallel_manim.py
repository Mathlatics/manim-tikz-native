from __future__ import annotations

import unittest
from unittest.mock import patch

from manim import ThreeDScene, tempconfig
from manim.utils.caching import get_hash_from_play_call

from polyhedron_visibility.quadrics.contract import SectionPlane, SphereSpec
from polyhedron_visibility.quadrics.parallel_plane_motion import (
    ParallelPlaneTranslation,
)
from polyhedron_visibility.quadrics.section_timeline import (
    compile_section_timeline,
)
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayCatalog,
    SectionDisplayInstruction,
    SectionDisplayRole,
    SectionSemanticSlot,
    compile_section_display,
)
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_frame import (
    ParallelFrameCoordinator,
    ParallelFrameParticipant,
    ParallelFramePhase,
    parallel_camera_frame_participant,
)
from tikz_native.parallel_preflight import (
    PainterOrderEvidence,
    ParallelPreflightLimits,
    ParallelSafeFrame,
)
from tikz_native.parallel_shots import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
)
from tikz_native.quadric_section_parallel import (
    compile_parallel_section_sequence_from_shots,
    parallel_screen_transform_guard,
    parallel_section_preflight_gate,
    section_bank_frame_participant,
    section_display_frame_participant,
    section_painter_order_participant,
)
from tikz_native.quadric_section_parallel_manim import (
    ParallelSectionPlaybackError,
    _ParallelSectionSegmentAnimation,
    _PlaybackCursor,
    play_parallel_section_sequence,
)


def _timeline():
    surface = SphereSpec("play-sphere", (0.0, 0.0, 0.0), 1.0)
    plane = SectionPlane(
        "play-plane",
        (0.0, 0.0, -2.0),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    return compile_section_timeline(
        "play-section",
        surface,
        (
            ParallelPlaneTranslation(
                "play-motion",
                plane,
                (0.0, 0.0, 4.0),
                start_time=0.0,
                end_time=2.0,
            ),
        ),
    )


def _display(timeline):
    slots = [
        SectionSemanticSlot(
            f"play:{bank}:curve:{slot}",
            SectionDisplayRole.SECTION_CURVE,
            topology_bank=bank,
        )
        for bank in ("bank-a", "bank-b")
        for slot in range(2)
    ]
    slots.extend(
        SectionSemanticSlot(
            f"play:{bank}:point",
            SectionDisplayRole.SECTION_POINT,
            topology_bank=bank,
        )
        for bank in ("bank-a", "bank-b")
    )
    catalog = SectionDisplayCatalog(timeline.section_id, tuple(slots))
    return compile_section_display(
        catalog,
        SectionDisplayInstruction.for_mode("painted"),
    )


def _painter(
    _time: float,
    _camera: ParallelCameraState,
    _plane: SectionPlane,
) -> PainterOrderEvidence:
    return PainterOrderEvidence(
        item_ids=("surface", "plane", "section"),
        relations=(("surface", "plane"), ("plane", "section")),
        draw_order=("surface", "plane", "section"),
    )


class _StateTarget:
    def __init__(self, baseline: object) -> None:
        self.state = baseline
        self.applied: list[object] = []

    def snapshot_section_bank_render_state(self) -> object:
        return self.state

    def apply_section_bank_render_frame(self, value: object) -> None:
        self.state = value
        self.applied.append(value)

    def restore_section_bank_render_state(self, value: object) -> None:
        self.state = value

    def snapshot_section_display_state(self) -> object:
        return self.state

    def apply_section_display_frame(self, value: object) -> None:
        self.state = value
        self.applied.append(value)

    def restore_section_display_state(self, value: object) -> None:
        self.state = value

    def snapshot_section_painter_order_state(self) -> object:
        return self.state

    def apply_section_painter_order(self, value: object) -> None:
        self.state = value
        self.applied.append(value)

    def restore_section_painter_order_state(self, value: object) -> None:
        self.state = value


class _RecordingScene(ThreeDScene):
    def __init__(self) -> None:
        super().__init__(camera_class=MultiProjectionCamera)
        self.play_run_times: list[float] = []

    def play(self, *animations, **kwargs):
        self.play_run_times.append(float(kwargs["run_time"]))
        return super().play(*animations, **kwargs)


def _single_shot_fixture():
    timeline = _timeline()
    display = _display(timeline)
    initial = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
    endpoint = initial.with_target((0.5, 0.0, 0.0))
    shots = ParallelCameraShotSequence(
        (
            ParallelCameraShot(
                "playback-guard-shot",
                endpoint,
                duration=2.0,
                transition="shortest",
            ),
        )
    )
    sequence = compile_parallel_section_sequence_from_shots(
        timeline,
        shots,
        initial,
        tuple(display for _ in timeline.samples),
        limits=ParallelPreflightLimits(
            ParallelSafeFrame(-10.0, 10.0, -10.0, 10.0),
            0.25,
            2.0,
        ),
        painter_orders=_painter,
        semantic_bank_ids=("bank-a", "bank-b"),
        frame_rate=30.0,
    )
    return initial, shots, sequence


def _complete_coordinator(scene, sequence):
    bank = _StateTarget("bank-baseline")
    painter = _StateTarget("painter-baseline")
    semantic = _StateTarget("display-baseline")
    gate = parallel_section_preflight_gate(sequence)
    coordinator = ParallelFrameCoordinator()
    coordinator.add(gate.participant())
    coordinator.add(
        parallel_screen_transform_guard(
            lambda: sequence.screen_transforms[gate.next_frame_index]
        )
    )
    coordinator.add(parallel_camera_frame_participant(scene.camera))
    coordinator.add(section_bank_frame_participant(bank))
    coordinator.add(section_painter_order_participant(painter))
    coordinator.add(section_display_frame_participant(semantic))
    return coordinator, gate, bank, painter, semantic


class ParallelSectionManimPlaybackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "frame_rate": 30,
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

    def test_real_scene_consumes_one_compiled_frame_grid_without_time_drift(
        self,
    ) -> None:
        timeline = _timeline()
        display = _display(timeline)
        initial = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
        midpoint = ParallelCameraState.from_view_direction((1.0, -1.0, 1.0))
        endpoint = ParallelCameraState.along_plane(
            timeline.samples[-1].plane,
            direction=(1.0, 0.0, 0.0),
        )
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "play-first",
                    midpoint,
                    duration=1.0,
                    transition="orbit",
                    arc_height=0.6,
                ),
                ParallelCameraShot(
                    "play-second",
                    endpoint,
                    duration=1.0,
                    transition="orbit",
                    arc_height=0.6,
                ),
            )
        )
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=ParallelPreflightLimits(
                ParallelSafeFrame(-10.0, 10.0, -10.0, 10.0),
                0.25,
                2.0,
            ),
            painter_orders=_painter,
            semantic_bank_ids=("bank-a", "bank-b"),
            frame_rate=30.0,
        )
        scene = _RecordingScene()
        scene.camera.set_parallel_state(initial)
        bank = _StateTarget("bank-baseline")
        painter = _StateTarget("painter-baseline")
        semantic = _StateTarget("display-baseline")
        gate = parallel_section_preflight_gate(sequence)
        coordinator = ParallelFrameCoordinator()
        coordinator.add(gate.participant())
        coordinator.add(
            parallel_screen_transform_guard(
                lambda: sequence.screen_transforms[gate.next_frame_index]
            )
        )
        coordinator.add(parallel_camera_frame_participant(scene.camera))
        coordinator.add(section_bank_frame_participant(bank))
        coordinator.add(section_painter_order_participant(painter))
        coordinator.add(section_display_frame_participant(semantic))

        rendered_gate_positions: list[int] = []
        live_render = scene.renderer.render

        def record_render(*args, **kwargs):
            rendered_gate_positions.append(gate.next_frame_index)
            return live_render(*args, **kwargs)

        with patch.object(
            scene.renderer,
            "render",
            side_effect=record_render,
        ) as render:
            final = play_parallel_section_sequence(
                scene,
                sequence,
                shots,
                coordinator,
            )

        self.assertIs(final, sequence.frames[-1])
        self.assertEqual(render.call_count, len(sequence.frames))
        self.assertEqual(
            rendered_gate_positions,
            list(range(1, len(sequence.frames) + 1)),
        )
        self.assertEqual(scene.play_run_times, [1.0, 1.0])
        self.assertAlmostEqual(scene.time, 2.0, places=12)
        self.assertEqual(gate.next_frame_index, len(sequence.frames))
        self.assertEqual(len(bank.applied), len(sequence.frames))
        self.assertEqual(len(painter.applied), len(sequence.frames))
        self.assertEqual(len(semantic.applied), len(sequence.frames))
        self.assertIs(scene.camera.snapshot_parallel_state(), endpoint)

        coordinator.restore()
        self.assertIs(scene.camera.snapshot_parallel_state(), initial)
        self.assertEqual(bank.state, "bank-baseline")
        self.assertEqual(painter.state, "painter-baseline")
        self.assertEqual(semantic.state, "display-baseline")

    def test_non_integral_shot_segments_do_not_duplicate_rendered_frames(
        self,
    ) -> None:
        timeline = _timeline()
        display = _display(timeline)
        initial = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
        midpoint = ParallelCameraState.from_view_direction((1.0, -1.0, 1.0))
        endpoint = initial.with_target((0.5, 0.2, 0.0))
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "fractional-first",
                    midpoint,
                    duration=0.7,
                    hold=0.3,
                    transition="orbit",
                    arc_height=0.3,
                ),
                ParallelCameraShot(
                    "fractional-second",
                    endpoint,
                    duration=0.6,
                    hold=0.4,
                    transition="shortest",
                ),
            )
        )
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=ParallelPreflightLimits(
                ParallelSafeFrame(-10.0, 10.0, -10.0, 10.0),
                0.25,
                2.0,
            ),
            painter_orders=_painter,
            semantic_bank_ids=("bank-a", "bank-b"),
            frame_rate=30.0,
        )
        scene = _RecordingScene()
        scene.camera.set_parallel_state(initial)
        coordinator, gate, bank, painter, semantic = _complete_coordinator(
            scene,
            sequence,
        )
        rendered_gate_positions: list[int] = []
        live_render = scene.renderer.render

        def record_render(*args, **kwargs):
            rendered_gate_positions.append(gate.next_frame_index)
            return live_render(*args, **kwargs)

        with patch.object(
            scene.renderer,
            "render",
            side_effect=record_render,
        ) as render:
            play_parallel_section_sequence(
                scene,
                sequence,
                shots,
                coordinator,
            )

        self.assertEqual(render.call_count, len(sequence.frames))
        self.assertEqual(
            rendered_gate_positions,
            list(range(1, len(sequence.frames) + 1)),
        )
        self.assertEqual(scene.play_run_times, [0.7, 0.3, 0.6, 0.4])
        self.assertAlmostEqual(scene.time, shots.total_duration, places=12)
        self.assertEqual(len(bank.applied), len(sequence.frames))
        self.assertEqual(len(painter.applied), len(sequence.frames))
        self.assertEqual(len(semantic.applied), len(sequence.frames))
        self.assertIs(scene.camera.snapshot_parallel_state(), endpoint)

    def test_manim_cache_hash_binds_full_sequence_and_segment_identity(
        self,
    ) -> None:
        initial, shots, sequence = _single_shot_fixture()
        timeline = _timeline()
        display = _display(timeline)

        def alternate_painter(
            _time: float,
            _camera: ParallelCameraState,
            _plane: SectionPlane,
        ) -> PainterOrderEvidence:
            return PainterOrderEvidence(
                item_ids=("surface", "plane", "section"),
                relations=(("plane", "surface"), ("surface", "section")),
                draw_order=("plane", "surface", "section"),
            )

        changed_sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=ParallelPreflightLimits(
                ParallelSafeFrame(-10.0, 10.0, -10.0, 10.0),
                0.25,
                2.0,
            ),
            painter_orders=alternate_painter,
            semantic_bank_ids=("bank-a", "bank-b"),
            frame_rate=30.0,
        )
        self.assertEqual(
            sequence.camera_provenance.sequence_digest,
            changed_sequence.camera_provenance.sequence_digest,
        )
        self.assertNotEqual(sequence.digest, changed_sequence.digest)
        scene = _RecordingScene()
        scene.camera.set_parallel_state(initial)

        def animation_hash(
            source_sequence,
            segment_identity: str,
        ) -> str:
            cursor = _PlaybackCursor(
                source_sequence,
                ParallelFrameCoordinator(),
                source_sequence.camera_provenance.nominal_frame_times,
            )
            animation = _ParallelSectionSegmentAnimation(
                cursor,
                0.0,
                2.0,
                segment_identity,
            )
            animation.run_time = 2.0
            return get_hash_from_play_call(
                scene,
                scene.camera,
                (animation,),
                scene.mobjects,
            )

        base_hash = animation_hash(sequence, "shot:0:source:transition")
        changed_sequence_hash = animation_hash(
            changed_sequence,
            "shot:0:source:transition",
        )
        changed_segment_hash = animation_hash(
            sequence,
            "shot:0:source:hold",
        )
        self.assertNotEqual(base_hash, changed_sequence_hash)
        self.assertNotEqual(base_hash, changed_segment_hash)

    def test_noop_coordinator_fails_before_scene_play(self) -> None:
        initial, shots, sequence = _single_shot_fixture()
        scene = _RecordingScene()
        scene.camera.set_parallel_state(initial)
        coordinator = ParallelFrameCoordinator()
        coordinator.add(
            ParallelFrameParticipant(
                "unrelated-noop",
                ParallelFramePhase.FINALIZE,
                lambda _frame: None,
                lambda: None,
                lambda _prepared: None,
                lambda _snapshot: None,
            )
        )

        with self.assertRaisesRegex(
            ParallelSectionPlaybackError,
            "missing required participants",
        ):
            play_parallel_section_sequence(
                scene,
                sequence,
                shots,
                coordinator,
            )

        self.assertEqual(scene.play_run_times, [])
        self.assertIsNone(coordinator.last_committed_frame)
        self.assertFalse(coordinator.active)

    def test_live_frame_rate_drift_fails_before_scene_play(self) -> None:
        initial, shots, sequence = _single_shot_fixture()
        with tempconfig({"frame_rate": 12}):
            scene = _RecordingScene()
            scene.camera.set_parallel_state(initial)
            coordinator, gate, _bank, _painter_target, _semantic = (
                _complete_coordinator(scene, sequence)
            )

            with self.assertRaisesRegex(
                ParallelSectionPlaybackError,
                "live Manim frame_rate 12.*compiled frame_rate 30",
            ):
                play_parallel_section_sequence(
                    scene,
                    sequence,
                    shots,
                    coordinator,
                )

            self.assertEqual(scene.play_run_times, [])
            self.assertEqual(gate.next_frame_index, 0)
            self.assertIsNone(coordinator.last_committed_frame)
            self.assertFalse(coordinator.active)

    def test_same_name_noop_participants_do_not_fake_renderer_bindings(self) -> None:
        initial, shots, sequence = _single_shot_fixture()
        scene = _RecordingScene()
        scene.camera.set_parallel_state(initial)
        coordinator = ParallelFrameCoordinator()
        for participant_id in (
            "parallel-preflight-gate",
            "parallel-screen-transform-guard",
            "parallel-camera",
            "section-bank-render",
            "section-painter-order",
            "section-semantic-display",
        ):
            coordinator.add(
                ParallelFrameParticipant(
                    participant_id,
                    ParallelFramePhase.FINALIZE,
                    lambda _frame: None,
                    lambda: None,
                    lambda _prepared: None,
                    lambda _snapshot: None,
                )
            )
        with self.assertRaisesRegex(
            ParallelSectionPlaybackError,
            "invalid participant bindings",
        ):
            play_parallel_section_sequence(
                scene,
                sequence,
                shots,
                coordinator,
            )
        self.assertEqual(scene.play_run_times, [])
        self.assertIs(scene.camera.snapshot_parallel_state(), initial)

    def test_live_shot_digest_mismatch_fails_before_scene_play(self) -> None:
        timeline = _timeline()
        display = _display(timeline)
        initial = ParallelCameraState.from_view_direction((1.0, 1.0, 1.0))
        endpoint = initial.with_target((0.5, 0.0, 0.0))
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "digest-source",
                    endpoint,
                    duration=2.0,
                    transition="shortest",
                ),
            )
        )
        sequence = compile_parallel_section_sequence_from_shots(
            timeline,
            shots,
            initial,
            tuple(display for _ in timeline.samples),
            limits=ParallelPreflightLimits(
                ParallelSafeFrame(-10.0, 10.0, -10.0, 10.0),
                0.25,
                2.0,
            ),
            painter_orders=_painter,
            semantic_bank_ids=("bank-a", "bank-b"),
            frame_rate=30.0,
        )
        changed = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "digest-source",
                    endpoint.with_zoom(1.1),
                    duration=2.0,
                    transition="shortest",
                ),
            )
        )
        scene = _RecordingScene()
        coordinator = ParallelFrameCoordinator()
        with self.assertRaisesRegex(
            ParallelSectionPlaybackError,
            "source digest",
        ):
            play_parallel_section_sequence(
                scene,
                sequence,
                changed,
                coordinator,
            )
        self.assertEqual(scene.play_run_times, [])


if __name__ == "__main__":
    unittest.main()
