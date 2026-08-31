"""Manim playback for one fully preflighted parallel-section sequence.

The renderer-neutral compiler may replace a nominal video-frame time with a
nearby analytic critical time, but it never adds a physical frame.  This
adapter consumes the same shot-local frame clock used by Manim: every camera
transition and hold is one local ``Scene.play`` segment, and compiled logical
times are paired one-to-one with those nominal output frames.
"""

from __future__ import annotations

from bisect import bisect_right
import hashlib
from math import isfinite

from manim import Animation, Mobject, config, linear

from .parallel_frame import (
    ParallelFrameBindingKind,
    ParallelFrameCoordinator,
    ParallelFramePhase,
    ParallelFrameState,
)
from .parallel_shots import (
    ParallelCameraShotSequence,
    canonical_parallel_camera_shot_sequence_json,
)
from .quadric_section_parallel import (
    ParallelSectionSequence,
    parallel_camera_shot_frame_times,
)


class ParallelSectionPlaybackError(RuntimeError):
    """A compiled sequence and its live Manim playback source disagree."""


_REQUIRED_PARTICIPANTS = {
    "parallel-preflight-gate": (
        ParallelFramePhase.PREFLIGHT,
        ParallelFrameBindingKind.PREFLIGHT_GATE,
    ),
    "parallel-viewport": (
        ParallelFramePhase.CAMERA,
        ParallelFrameBindingKind.CAMERA,
    ),
    "section-bank-render": (
        ParallelFramePhase.GEOMETRY,
        ParallelFrameBindingKind.SECTION_BANK,
    ),
    "section-painter-order": (
        ParallelFramePhase.VISIBILITY,
        ParallelFrameBindingKind.SECTION_PAINTER,
    ),
    "section-semantic-display": (
        ParallelFramePhase.PAINT,
        ParallelFrameBindingKind.SECTION_DISPLAY,
    ),
    "section-semantic-compositing": (
        ParallelFramePhase.PAINT,
        ParallelFrameBindingKind.SECTION_COMPOSITING,
    ),
}


def _require_live_frame_rate(expected: float) -> None:
    live = float(config.frame_rate)
    if live != expected:
        raise ParallelSectionPlaybackError(
            f"live Manim frame_rate {live:.12g} differs from compiled "
            f"frame_rate {expected:.12g}"
        )


def _require_complete_coordinator(
    coordinator: ParallelFrameCoordinator[ParallelFrameState],
    sequence: ParallelSectionSequence,
) -> None:
    required = dict(_REQUIRED_PARTICIPANTS)
    if any(item is not None for item in sequence.plane_patch_fits):
        required["section-plane-patch"] = (
            ParallelFramePhase.GEOMETRY,
            ParallelFrameBindingKind.SECTION_PLANE_PATCH,
        )
    actual = {
        participant_id: (phase, binding_kind)
        for participant_id, phase, binding_kind
        in coordinator.participant_bindings
    }
    missing = tuple(sorted(set(required) - set(actual)))
    if missing:
        raise ParallelSectionPlaybackError(
            "parallel-section coordinator is missing required participants: "
            + ", ".join(missing)
        )
    mismatched = tuple(
        sorted(
            participant_id
            for participant_id, contract in required.items()
            if actual[participant_id] != contract
        )
    )
    if mismatched:
        raise ParallelSectionPlaybackError(
            "parallel-section coordinator has invalid participant bindings: "
            + ", ".join(mismatched)
        )


class _PlaybackCursor:
    def __init__(
        self,
        sequence: ParallelSectionSequence,
        coordinator: ParallelFrameCoordinator[ParallelFrameState],
        nominal_times: tuple[float, ...],
    ) -> None:
        self.sequence = sequence
        self.coordinator = coordinator
        self.nominal_times = nominal_times
        self.sequence_digest = sequence.digest
        self.next_index = 0

    def commit_through_time(self, nominal_time: float) -> None:
        tolerance = (
            64.0
            * 2.220446049250313e-16
            * max(1.0, abs(nominal_time))
        )
        target = bisect_right(
            self.nominal_times,
            nominal_time + tolerance,
        ) - 1
        while self.next_index <= target:
            self.coordinator.update(
                self.sequence.frames[self.next_index]
            )
            self.next_index += 1


class _ParallelSectionSegmentAnimation(Animation):
    def __init__(
        self,
        cursor: _PlaybackCursor,
        start_time: float,
        end_time: float,
        segment_identity: str,
    ) -> None:
        self.cursor = cursor
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        if not isinstance(segment_identity, str) or not segment_identity.strip():
            raise ParallelSectionPlaybackError(
                "playback segment identity must be a non-empty string"
            )
        # Manim's cache encoder cannot serialize the slotted sequence itself.
        # Keep these primitive fields directly on the Animation so two authored
        # sequences or two semantic segments can never share a cached movie.
        self.parallel_section_sequence_digest = cursor.sequence_digest
        self.parallel_section_segment_identity = segment_identity.strip()
        if (
            not isfinite(self.start_time)
            or not isfinite(self.end_time)
            or self.end_time <= self.start_time
        ):
            raise ParallelSectionPlaybackError(
                "playback segment times must be finite and increasing"
            )
        super().__init__(
            Mobject(),
            introducer=False,
            remover=True,
            rate_func=linear,
        )

    def interpolate_mobject(self, alpha: float) -> None:
        progress = min(1.0, max(0.0, float(alpha)))
        nominal_time = self.start_time + progress * (
            self.end_time - self.start_time
        )
        self.cursor.commit_through_time(nominal_time)

    def finish(self) -> None:
        self.cursor.commit_through_time(self.end_time)
        super().finish()


def _sequence_digest(sequence: ParallelCameraShotSequence) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_parallel_camera_shot_sequence_json(sequence).encode("utf-8")
    ).hexdigest()


def play_parallel_section_sequence(
    scene: object,
    sequence: ParallelSectionSequence,
    shot_sequence: ParallelCameraShotSequence,
    coordinator: ParallelFrameCoordinator[ParallelFrameState],
) -> ParallelFrameState:
    """Play every certified frame without adding frames or drifting duration."""

    if not isinstance(sequence, ParallelSectionSequence):
        raise TypeError("sequence must be a ParallelSectionSequence")
    if not isinstance(shot_sequence, ParallelCameraShotSequence):
        raise TypeError("shot_sequence must be a ParallelCameraShotSequence")
    if not isinstance(coordinator, ParallelFrameCoordinator):
        raise TypeError("coordinator must be a ParallelFrameCoordinator")
    play = getattr(scene, "play", None)
    if not callable(play):
        raise TypeError("scene must provide a callable play() method")
    provenance = sequence.camera_provenance
    if provenance is None or provenance.frame_rate is None:
        raise ParallelSectionPlaybackError(
            "sequence must be compiled from shots with an explicit frame_rate"
        )
    if provenance.coverage != "exact":
        raise ParallelSectionPlaybackError(
            "Manim playback currently requires exact camera source coverage"
        )
    _require_live_frame_rate(provenance.frame_rate)
    if _sequence_digest(shot_sequence) != provenance.sequence_digest:
        raise ParallelSectionPlaybackError(
            "live camera shots differ from the compiled source digest"
        )
    _require_complete_coordinator(coordinator, sequence)
    if coordinator.last_committed_frame is not None or coordinator.active:
        raise ParallelSectionPlaybackError(
            "coordinator must be at a restored sequence boundary"
        )
    live_nominal_times = parallel_camera_shot_frame_times(
        shot_sequence,
        start_time=provenance.start_time,
        frame_rate=provenance.frame_rate,
    )
    nominal_times = provenance.nominal_frame_times
    if nominal_times != live_nominal_times:
        raise ParallelSectionPlaybackError(
            "live camera shot frame clock differs from compiled provenance"
        )
    if len(nominal_times) != len(sequence.evaluation_times):
        raise ParallelSectionPlaybackError(
            "compiled analytic replacements changed the physical frame count"
        )
    if nominal_times[0] != provenance.start_time:
        raise ParallelSectionPlaybackError(
            "nominal playback grid disagrees with source provenance"
        )

    cursor = _PlaybackCursor(sequence, coordinator, nominal_times)
    segments: list[tuple[str, float, float, float]] = []
    segment_start = provenance.start_time
    for shot_index, shot in enumerate(shot_sequence.shots):
        transition_end = segment_start + shot.duration
        segments.append(
            (
                f"shot:{shot_index}:{shot.id}:transition",
                segment_start,
                transition_end,
                shot.duration,
            )
        )
        segment_start = transition_end
        if shot.hold > 0.0:
            hold_end = segment_start + shot.hold
            segments.append(
                (
                    f"shot:{shot_index}:{shot.id}:hold",
                    segment_start,
                    hold_end,
                    shot.hold,
                )
            )
            segment_start = hold_end
    try:
        for segment_identity, start, end, run_time in segments:
            play(
                _ParallelSectionSegmentAnimation(
                    cursor,
                    start,
                    end,
                    segment_identity,
                ),
                run_time=run_time,
                rate_func=linear,
            )
        cursor.commit_through_time(provenance.end_time)
        if cursor.next_index != len(sequence.frames):
            raise ParallelSectionPlaybackError(
                "Manim playback did not consume every compiled frame"
            )
    except BaseException:
        if coordinator.active:
            coordinator.restore()
        raise
    return sequence.frames[-1]


__all__ = [
    "ParallelSectionPlaybackError",
    "play_parallel_section_sequence",
]
