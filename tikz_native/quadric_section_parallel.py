"""Renderer-neutral composition for camera shots and quadric section timelines.

This module is the reviewed seam between otherwise independent contracts:

* semantic camera shots are sampled at the exact SectionTimeline times;
* topology events become fixed-bank preflight evidence;
* semantic display frames retain their catalog and bank identities;
* only an accepted joint preflight can produce coordinated frame states.

There is deliberately no Manim or QuadricSectionRig import here.  A future Rig
adapter only needs to implement the semantic display transaction methods used
by :func:`section_display_frame_participant`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Callable, Mapping, Sequence

import numpy as np

from polyhedron_visibility.quadrics.animation import TrackedSectionFrame
from polyhedron_visibility.quadrics.contract import SectionPlane
from polyhedron_visibility.quadrics.plane_patch import (
    FittedPlaneDisplayPatch,
    PlanePatchFitError,
    fit_plane_display_patch,
)
from polyhedron_visibility.quadrics.section_timeline import SectionTimeline
from polyhedron_visibility.quadrics.section_timeline_transition import (
    SectionTimelineTransitionState,
    SectionTimelineTransitionMode,
    SectionTimelineTransitionPlan,
    build_section_timeline_transition_plan,
    section_timeline_transition_state_at,
)
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayFrame,
    SectionDisplayRole,
)
from polyhedron_visibility.quadrics.sections import (
    QuadricSectionError,
    compute_quadric_section_boundary,
)

from .parallel_camera import (
    ParallelCameraState,
    interpolate_parallel_camera_states,
    orbit_control_matrix,
)
from .parallel_frame import (
    ParallelFrameBindingKind,
    ParallelFrameParticipant,
    ParallelFramePhase,
    ParallelFrameState,
)
from .parallel_preflight import (
    CapacityEvidence,
    PainterOrderEvidence,
    PARALLEL_PREFLIGHT_FRAME_CHANNEL,
    ParallelPreflightFrame,
    ParallelPreflightGate,
    ParallelPreflightLimits,
    ParallelPreflightReport,
    ParallelScreenTransform,
    TopologyEventEvidence,
    preflight_parallel_frames,
)
from .parallel_shots import (
    PARALLEL_CAMERA_SHOT_EASING,
    ParallelCameraShotSequence,
    canonical_parallel_camera_shot_sequence_json,
    parallel_camera_shot_progress,
)
from .section_bank_render import (
    SectionBankRenderFrame,
    SectionBankRenderLayer,
    section_bank_frame_participant,
)


PARALLEL_SECTION_SEQUENCE_SCHEMA = "parallel-quadric-section-sequence/v1"
SECTION_PRIMARY_REFERENCE_FRAME_CHANNEL = "section-primary-reference-frame"
SECTION_EVALUATION_PLANE_CHANNEL = "section-evaluation-plane"
# Backward-compatible names for the unpublished integration prototype.  The
# channel strings are explicit about which time each value describes.
SECTION_TIMELINE_FRAME_CHANNEL = SECTION_PRIMARY_REFERENCE_FRAME_CHANNEL
SECTION_PLANE_CHANNEL = SECTION_EVALUATION_PLANE_CHANNEL
SECTION_DISPLAY_CHANNEL = "section-display"
SECTION_TOPOLOGY_BANK_CHANNEL = "section-topology-bank"
SECTION_TRANSITION_STATE_CHANNEL = "section-transition-state"
SECTION_BANK_RENDER_CHANNEL = "section-bank-render-frame"
SECTION_PAINTER_ORDER_CHANNEL = "section-painter-order"
SECTION_PLANE_PATCH_CHANNEL = "section-plane-display-patch"
PARALLEL_SCREEN_TRANSFORM_CHANNEL = "parallel-screen-transform"


class ParallelSectionSequenceError(ValueError):
    """Camera, timeline, display, or preflight evidence cannot be aligned."""


class ParallelCameraShotSamplePhase(str, Enum):
    TRANSITION = "transition"
    ENDPOINT = "endpoint"
    HOLD = "hold"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParallelSectionSequenceError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ParallelSectionSequenceError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelSectionSequenceError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise ParallelSectionSequenceError(f"{label} must be finite")
    return result


def _time_tolerance(*values: float) -> float:
    return (
        64.0
        * float(np.finfo(float).eps)
        * max(1.0, *(abs(float(item)) for item in values))
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_json(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _sha256_identity(value: object, label: str) -> str:
    digest = _identity(value, label)
    payload = digest[7:] if digest.startswith("sha256:") else ""
    if len(payload) != 64 or any(
        item not in "0123456789abcdef" for item in payload
    ):
        raise ParallelSectionSequenceError(
            f"{label} must be a lowercase sha256 digest"
        )
    return digest


def _camera_states_equal(
    left: ParallelCameraState,
    right: ParallelCameraState,
) -> bool:
    return bool(
        np.array_equal(left.matrix, right.matrix)
        and np.array_equal(left.target, right.target)
        and np.array_equal(left.screen_anchor, right.screen_anchor)
        and left.zoom == right.zoom
    )


def _camera_samples_equal(
    left: ParallelCameraShotSample,
    right: ParallelCameraShotSample,
) -> bool:
    return bool(
        left.sample_id == right.sample_id
        and left.time == right.time
        and left.shot_id == right.shot_id
        and left.phase is right.phase
        and _camera_states_equal(left.state, right.state)
    )


def _plane_to_dict(plane: SectionPlane) -> dict[str, object]:
    if not isinstance(plane, SectionPlane):
        raise TypeError("section plane channel must contain SectionPlane")
    return {
        "planeId": plane.plane_id,
        "point": list(plane.point),
        "normal": list(plane.normal),
        "uAxis": list(plane.u_axis),
    }


def _digest_tracked_frame(value: object) -> str:
    if not isinstance(value, TrackedSectionFrame):
        raise TypeError(
            "section timeline frame channel must contain TrackedSectionFrame"
        )
    return _digest_json(value.to_dict())


def _digest_plane(value: object) -> str:
    if not isinstance(value, SectionPlane):
        raise TypeError("section plane channel must contain SectionPlane")
    return _digest_json(_plane_to_dict(value))


def _digest_display(value: object) -> str:
    if not isinstance(value, SectionDisplayFrame):
        raise TypeError("section display channel must contain SectionDisplayFrame")
    return value.digest


def _digest_topology_banks(value: object) -> str:
    if not isinstance(value, tuple) or not value or any(
        isinstance(item, bool) or item not in {0, 1} for item in value
    ):
        raise TypeError(
            "section topology bank channel must contain a non-empty tuple of 0/1"
        )
    return _digest_json(list(value))


def _digest_transition_state(value: object) -> str:
    if not isinstance(value, SectionTimelineTransitionState):
        raise TypeError(
            "section transition channel must contain SectionTimelineTransitionState"
        )
    return _digest_json(value.to_dict())


def _digest_bank_render_frame(value: object) -> str:
    if not isinstance(value, SectionBankRenderFrame):
        raise TypeError(
            "section bank channel must contain SectionBankRenderFrame"
        )
    return value.digest


def _digest_screen_transform(value: object) -> str:
    if not isinstance(value, ParallelScreenTransform):
        raise TypeError(
            "screen-transform channel must contain ParallelScreenTransform"
        )
    return _digest_json(value.to_dict())


def _digest_painter_order(value: object) -> str:
    if not isinstance(value, PainterOrderEvidence):
        raise TypeError(
            "painter-order channel must contain PainterOrderEvidence"
        )
    return _digest_json(value.to_dict())


def _digest_plane_patch(value: object) -> str:
    if value is None:
        return _digest_json(None)
    if not isinstance(value, FittedPlaneDisplayPatch):
        raise TypeError(
            "plane-patch channel must contain FittedPlaneDisplayPatch or None"
        )
    return _digest_json(value.to_dict())


def parallel_section_channel_digesters(
) -> Mapping[str, Callable[[object], str]]:
    """Return the fixed digest contract used by the section preflight gate."""

    return {
        SECTION_TIMELINE_FRAME_CHANNEL: _digest_tracked_frame,
        SECTION_PLANE_CHANNEL: _digest_plane,
        SECTION_DISPLAY_CHANNEL: _digest_display,
        SECTION_TOPOLOGY_BANK_CHANNEL: _digest_topology_banks,
        SECTION_TRANSITION_STATE_CHANNEL: _digest_transition_state,
        SECTION_BANK_RENDER_CHANNEL: _digest_bank_render_frame,
        SECTION_PAINTER_ORDER_CHANNEL: _digest_painter_order,
        SECTION_PLANE_PATCH_CHANNEL: _digest_plane_patch,
        PARALLEL_SCREEN_TRANSFORM_CHANNEL: _digest_screen_transform,
    }


@dataclass(frozen=True, slots=True)
class ParallelCameraShotSample:
    sample_id: str
    time: float
    shot_id: str
    phase: ParallelCameraShotSamplePhase
    state: ParallelCameraState

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _identity(self.sample_id, "sample_id"))
        object.__setattr__(self, "time", _finite(self.time, "sample time"))
        object.__setattr__(self, "shot_id", _identity(self.shot_id, "shot_id"))
        try:
            phase = ParallelCameraShotSamplePhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise ParallelSectionSequenceError(
                "phase must be a ParallelCameraShotSamplePhase"
            ) from exc
        object.__setattr__(self, "phase", phase)
        if not isinstance(self.state, ParallelCameraState):
            raise TypeError("state must be a ParallelCameraState")

    def to_dict(self) -> dict[str, object]:
        return {
            "sampleId": self.sample_id,
            "time": self.time,
            "shotId": self.shot_id,
            "phase": self.phase.value,
            "state": {
                "matrix": self.state.matrix.tolist(),
                "target": self.state.target.tolist(),
                "screenAnchor": self.state.screen_anchor.tolist(),
                "zoom": self.state.zoom,
            },
        }


@dataclass(frozen=True, slots=True)
class ParallelCameraSamplingProvenance:
    sequence_digest: str
    shot_sequence: ParallelCameraShotSequence
    initial_camera: ParallelCameraState
    start_time: float
    end_time: float
    coverage: str
    frame_rate: float | None = None
    nominal_frame_times: tuple[float, ...] = ()
    easing: str = PARALLEL_CAMERA_SHOT_EASING

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_digest",
            _sha256_identity(self.sequence_digest, "camera sequence digest"),
        )
        if not isinstance(self.shot_sequence, ParallelCameraShotSequence):
            raise TypeError("shot_sequence must be a ParallelCameraShotSequence")
        expected_digest = "sha256:" + hashlib.sha256(
            canonical_parallel_camera_shot_sequence_json(
                self.shot_sequence
            ).encode("utf-8")
        ).hexdigest()
        if self.sequence_digest != expected_digest:
            raise ParallelSectionSequenceError(
                "camera sequence digest does not describe shot_sequence"
            )
        if not isinstance(self.initial_camera, ParallelCameraState):
            raise TypeError("initial_camera must be a ParallelCameraState")
        start = _finite(self.start_time, "camera sequence start_time")
        end = _finite(self.end_time, "camera sequence end_time")
        if end <= start:
            raise ParallelSectionSequenceError(
                "camera sequence provenance must have positive duration"
            )
        if self.coverage not in {"exact", "window"}:
            raise ParallelSectionSequenceError(
                "camera sequence coverage must be 'exact' or 'window'"
            )
        if self.easing != PARALLEL_CAMERA_SHOT_EASING:
            raise ParallelSectionSequenceError(
                "camera sequence easing identity is unsupported"
            )
        if self.frame_rate is not None:
            rate = _finite(self.frame_rate, "camera sequence frame_rate")
            if rate <= 0.0:
                raise ParallelSectionSequenceError(
                    "camera sequence frame_rate must be positive"
                )
            object.__setattr__(self, "frame_rate", rate)
        nominal = tuple(
            _finite(item, "nominal camera frame time")
            for item in self.nominal_frame_times
        )
        if any(right <= left for left, right in zip(nominal, nominal[1:])):
            raise ParallelSectionSequenceError(
                "nominal camera frame times must increase strictly"
            )
        if self.frame_rate is None and nominal:
            raise ParallelSectionSequenceError(
                "nominal camera frame times require an explicit frame_rate"
            )
        if self.frame_rate is not None and not nominal:
            raise ParallelSectionSequenceError(
                "frame-rate camera provenance requires nominal frame times"
            )
        if nominal and (nominal[0] < start or nominal[-1] >= end):
            raise ParallelSectionSequenceError(
                "nominal camera frame times must lie in [start_time, end_time)"
            )
        object.__setattr__(self, "nominal_frame_times", nominal)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)

    def to_dict(self) -> dict[str, object]:
        return {
            "sequenceDigest": self.sequence_digest,
            "shotSequence": self.shot_sequence.to_dict(),
            "initialCamera": {
                "matrix": self.initial_camera.matrix.tolist(),
                "target": self.initial_camera.target.tolist(),
                "screenAnchor": self.initial_camera.screen_anchor.tolist(),
                "zoom": self.initial_camera.zoom,
            },
            "startTime": self.start_time,
            "endTime": self.end_time,
            "coverage": self.coverage,
            "frameRate": self.frame_rate,
            "nominalFrameTimes": list(self.nominal_frame_times),
            "easing": self.easing,
        }


def sample_parallel_camera_shot_sequence(
    sequence: ParallelCameraShotSequence,
    initial_state: ParallelCameraState,
    sample_times: Sequence[float],
    *,
    start_time: float = 0.0,
) -> tuple[ParallelCameraShotSample, ...]:
    """Sample the same interpolation family used by semantic Manim playback."""

    if not isinstance(sequence, ParallelCameraShotSequence):
        raise TypeError("sequence must be a ParallelCameraShotSequence")
    if not isinstance(initial_state, ParallelCameraState):
        raise TypeError("initial_state must be a ParallelCameraState")
    start = _finite(start_time, "start_time")
    times = tuple(_finite(item, "sample time") for item in sample_times)
    if not times:
        raise ParallelSectionSequenceError("sample_times must be non-empty")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ParallelSectionSequenceError(
            "sample_times must increase strictly"
        )
    segments: list[
        tuple[
            object,
            ParallelCameraState,
            float,
            float,
            float,
            np.ndarray | None,
        ]
    ] = []
    cursor = start
    source = initial_state
    for shot in sequence.shots:
        transition_end = cursor + shot.duration
        hold_end = transition_end + shot.hold
        if transition_end <= cursor or (
            shot.hold > 0.0 and hold_end <= transition_end
        ):
            raise ParallelSectionSequenceError(
                "camera shot timing is below absolute-time numeric resolution"
            )
        transition_tolerance = _time_tolerance(cursor, transition_end)
        if transition_tolerance > shot.duration * 1.0e-6:
            raise ParallelSectionSequenceError(
                "camera shot absolute times are too large for its local duration"
            )
        if shot.hold > 0.0 and _time_tolerance(
            transition_end,
            hold_end,
        ) > shot.hold * 1.0e-6:
            raise ParallelSectionSequenceError(
                "camera hold absolute times are too large for its local duration"
            )
        control = None
        if shot.transition == "orbit":
            control = (
                source.matrix
                if np.allclose(
                    source.matrix,
                    shot.state.matrix,
                    atol=1.0e-12,
                    rtol=0.0,
                )
                else orbit_control_matrix(
                    source.matrix,
                    shot.state.matrix,
                    arc_height=shot.arc_height,
                )
            )
        segments.append(
            (shot, source, cursor, transition_end, hold_end, control)
        )
        cursor = hold_end
        source = shot.state
    boundary_tolerance = _time_tolerance(
        start,
        cursor,
        times[0],
        times[-1],
    )
    if (
        times[0] < start - boundary_tolerance
        or times[-1] > cursor + boundary_tolerance
    ):
        raise ParallelSectionSequenceError(
            "sample_times must lie inside the authored camera sequence"
        )

    result: list[ParallelCameraShotSample] = []
    for sample_index, sample_time in enumerate(times):
        selected: ParallelCameraShotSample | None = None
        for (
            shot,
            source,
            transition_start,
            transition_end,
            hold_end,
            control,
        ) in segments:
            tolerance = _time_tolerance(
                sample_time,
                transition_start,
                transition_end,
                hold_end,
            )
            if abs(sample_time - transition_end) <= tolerance:
                selected = ParallelCameraShotSample(
                    f"camera-shot-sample:{sample_index:04d}",
                    sample_time,
                    shot.id,
                    ParallelCameraShotSamplePhase.ENDPOINT,
                    shot.state,
                )
                break
            if (
                shot.hold > 0.0
                and abs(sample_time - hold_end) <= tolerance
            ):
                selected = ParallelCameraShotSample(
                    f"camera-shot-sample:{sample_index:04d}",
                    sample_time,
                    shot.id,
                    ParallelCameraShotSamplePhase.HOLD,
                    shot.state,
                )
                break
            if sample_time <= transition_end:
                progress = (sample_time - transition_start) / (
                    transition_end - transition_start
                )
                progress = min(1.0, max(0.0, progress))
                eased_progress = parallel_camera_shot_progress(progress)
                state = interpolate_parallel_camera_states(
                    source,
                    shot.state,
                    eased_progress,
                    control_matrix=control,
                )
                phase = (
                    ParallelCameraShotSamplePhase.ENDPOINT
                    if progress == 1.0
                    else ParallelCameraShotSamplePhase.TRANSITION
                )
                selected = ParallelCameraShotSample(
                    f"camera-shot-sample:{sample_index:04d}",
                    sample_time,
                    shot.id,
                    phase,
                    state,
                )
                break
            if sample_time <= hold_end:
                selected = ParallelCameraShotSample(
                    f"camera-shot-sample:{sample_index:04d}",
                    sample_time,
                    shot.id,
                    ParallelCameraShotSamplePhase.HOLD,
                    shot.state,
                )
                break
        if selected is None:  # Defensive against inconsistent float accumulation.
            raise ParallelSectionSequenceError(
                "sample time could not be assigned to a camera shot"
            )
        result.append(selected)
    return tuple(result)


def parallel_section_render_times(
    plan: SectionTimelineTransitionPlan,
    render_times: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Return a certified render grid containing every analytic key time.

    When no external frame grid is supplied, one midpoint is inserted into
    each side of every non-empty crossfade window.  That makes both topology
    banks enter a real coordinated frame.  A caller supplying a renderer grid
    must include every timeline sample and at least one strict interior sample
    for each such window; missing evidence is rejected rather than guessed.
    """

    if not isinstance(plan, SectionTimelineTransitionPlan):
        raise TypeError("plan must be a SectionTimelineTransitionPlan")
    key_times = tuple(item.time for item in plan.timeline.samples)
    if render_times is None:
        values = set(key_times)
        for knot in plan.knots:
            if knot.left_crossfade:
                values.add(0.5 * (knot.left_start + knot.critical_time))
            if knot.right_crossfade:
                values.add(0.5 * (knot.critical_time + knot.right_end))
        result = tuple(sorted(values))
    else:
        result = tuple(_finite(item, "render time") for item in render_times)
        if not result or any(
            right <= left for left, right in zip(result, result[1:])
        ):
            raise ParallelSectionSequenceError(
                "render_times must be non-empty and strictly increasing"
            )
        if not set(key_times).issubset(result):
            raise ParallelSectionSequenceError(
                "render_times must contain every analytic SectionTimeline time"
            )
    if result[0] != key_times[0] or result[-1] != key_times[-1]:
        raise ParallelSectionSequenceError(
            "render_times must exactly span the SectionTimeline"
        )
    for left, right in zip(result, result[1:]):
        resolution = (
            64.0
            * float(np.finfo(float).eps)
            * max(1.0, abs(left), abs(right))
        )
        if right - left <= resolution:
            raise ParallelSectionSequenceError(
                "render_times contain two numerically indistinguishable frames"
            )
    for knot in plan.knots:
        for label, left, right, enabled in (
            (
                "left",
                knot.left_start,
                knot.critical_time,
                knot.left_crossfade,
            ),
            (
                "right",
                knot.critical_time,
                knot.right_end,
                knot.right_crossfade,
            ),
        ):
            if enabled and not any(left < item < right for item in result):
                raise ParallelSectionSequenceError(
                    f"render_times omit the {label} interior of topology "
                    f"crossfade {knot.knot_id!r}"
                )
    return result


def _local_segment_frame_times(
    start: float,
    end: float,
    frame_rate: float,
) -> tuple[float, ...]:
    duration = end - start
    if duration <= 0.0:
        raise ParallelSectionSequenceError(
            "playback segment duration must be positive"
        )
    if _time_tolerance(start, end) > min(
        duration,
        1.0 / frame_rate,
    ) * 1.0e-6:
        raise ParallelSectionSequenceError(
            "playback absolute times are too large for the local frame clock"
        )
    count = int(duration * frame_rate)
    result = [
        start + index / frame_rate
        for index in range(count + 1)
        if index / frame_rate < duration
    ]
    result.append(end)
    return tuple(result)


def parallel_camera_shot_frame_times(
    sequence: ParallelCameraShotSequence,
    *,
    start_time: float = 0.0,
    frame_rate: float,
) -> tuple[float, ...]:
    """Return the physical frame times Manim renders for every play/hold.

    Manim renders local times in ``[0, run_time)``.  A segment endpoint is
    therefore represented by the next segment's first frame, when there is
    one; the final authored endpoint is not an additional output frame.
    """

    if not isinstance(sequence, ParallelCameraShotSequence):
        raise TypeError("sequence must be a ParallelCameraShotSequence")
    cursor = _finite(start_time, "start_time")
    rate = _finite(frame_rate, "frame_rate")
    if rate <= 0.0:
        raise ParallelSectionSequenceError("frame_rate must be positive")
    values: set[float] = set()
    for shot in sequence.shots:
        transition_end = cursor + shot.duration
        values.update(
            _local_segment_frame_times(cursor, transition_end, rate)[:-1]
        )
        cursor = transition_end
        if shot.hold > 0.0:
            hold_end = cursor + shot.hold
            values.update(
                _local_segment_frame_times(cursor, hold_end, rate)[:-1]
            )
            cursor = hold_end
    return tuple(sorted(values))


def _replace_generated_times_with_analytic(
    generated: Sequence[float],
    analytic: Sequence[float],
    *,
    frame_rate: float,
    protected: Sequence[float],
) -> set[float]:
    result = list(sorted(set(float(item) for item in generated)))
    protected_values = tuple(float(item) for item in protected)
    used_indices: set[int] = set()
    for exact in analytic:
        tolerance = (
            64.0
            * float(np.finfo(float).eps)
            * max(1.0, abs(exact))
        )
        near = tuple(
            index
            for index, item in enumerate(result)
            if index not in used_indices and abs(item - exact) <= tolerance
        )
        if near:
            selected = min(near, key=lambda index: abs(result[index] - exact))
        else:
            candidates = tuple(
                index
                for index, item in enumerate(result)
                if index not in used_indices
                and not any(
                    abs(item - boundary) <= tolerance
                    for boundary in protected_values
                )
            )
            if not candidates:
                raise ParallelSectionSequenceError(
                    "frame_rate leaves no replaceable frame for an analytic time"
                )
            selected = min(
                candidates,
                key=lambda index: abs(result[index] - exact),
            )
            analytic_is_boundary = any(
                abs(exact - boundary) <= tolerance
                for boundary in protected_values
            )
            maximum_distance = (
                1.0 / frame_rate
                if analytic_is_boundary
                else 0.5 / frame_rate
            )
            if abs(result[selected] - exact) > maximum_distance + tolerance:
                raise ParallelSectionSequenceError(
                    "analytic time is farther than half a frame from the "
                    "renderer grid"
                )
        result[selected] = exact
        used_indices.add(selected)
    if len(set(result)) != len(result):
        raise ParallelSectionSequenceError(
            "analytic frame replacement collided; increase frame_rate"
        )
    return set(result)


def parallel_section_frame_grid(
    plan: SectionTimelineTransitionPlan,
    frame_rate: float,
    *,
    shot_sequence: ParallelCameraShotSequence | None = None,
    start_time: float | None = None,
) -> tuple[float, ...]:
    """Build a renderer frame grid split at every certified analytic time."""

    if not isinstance(plan, SectionTimelineTransitionPlan):
        raise TypeError("plan must be a SectionTimelineTransitionPlan")
    rate = _finite(frame_rate, "frame_rate")
    if rate <= 0.0:
        raise ParallelSectionSequenceError("frame_rate must be positive")
    start = plan.timeline.samples[0].time
    end = plan.timeline.samples[-1].time
    if shot_sequence is not None:
        camera_start = start if start_time is None else _finite(
            start_time,
            "start_time",
        )
        generated = parallel_camera_shot_frame_times(
            shot_sequence,
            start_time=camera_start,
            frame_rate=rate,
        )
        coverage_tolerance = _time_tolerance(
            generated[0],
            generated[-1],
            start,
            end,
        )
        if (
            generated[0] > start + coverage_tolerance
            or generated[-1] < end - 1.0 / rate - coverage_tolerance
        ):
            raise ParallelSectionSequenceError(
                "camera shot frame grid does not cover the SectionTimeline"
            )
        generated = tuple(item for item in generated if start <= item <= end)
        cursor = camera_start
        protected_values = [cursor, start, end]
        for shot in shot_sequence.shots:
            cursor += shot.duration
            protected_values.append(cursor)
            if shot.hold > 0.0:
                cursor += shot.hold
                protected_values.append(cursor)
    else:
        generated = tuple(
            time
            for schedule in plan.timeline.segment_schedules
            for time in _local_segment_frame_times(
                schedule.motion.time_at(0.0),
                schedule.motion.time_at(1.0),
                rate,
            )
        )
        protected_values = [
            schedule.motion.time_at(endpoint)
            for schedule in plan.timeline.segment_schedules
            for endpoint in (0.0, 1.0)
        ]
    analytic_times = tuple(item.time for item in plan.timeline.samples)
    values = _replace_generated_times_with_analytic(
        generated,
        analytic_times,
        frame_rate=rate,
        protected=protected_values,
    )
    for knot in plan.knots:
        if knot.left_crossfade and not any(
            knot.left_start < item < knot.critical_time for item in values
        ):
            raise ParallelSectionSequenceError(
                "frame_rate is too low to sample a left topology crossfade"
            )
        if knot.right_crossfade and not any(
            knot.critical_time < item < knot.right_end for item in values
        ):
            raise ParallelSectionSequenceError(
                "frame_rate is too low to sample a right topology crossfade"
            )
    return parallel_section_render_times(plan, tuple(sorted(values)))


def _timeline_plane_at_time(timeline: SectionTimeline, time: float) -> SectionPlane:
    exact = {
        sample.time: sample.plane for sample in timeline.samples
    }.get(time)
    if exact is not None:
        return exact
    for schedule in timeline.segment_schedules:
        motion = schedule.motion
        start = motion.time_at(0.0)
        end = motion.time_at(1.0)
        if start < time < end:
            progress = (time - start) / (end - start)
            return motion.plane_at(progress)
    raise ParallelSectionSequenceError(
        "render time cannot be resolved to a SectionTimeline plane"
    )


def _primary_reference_index(state: SectionTimelineTransitionState) -> int:
    # Stable tie-break: transition-state layer order is canonical.
    return max(
        enumerate(state.layers),
        key=lambda item: (item[1].opacity, -item[0]),
    )[1].reference_frame_index


def _resolve_keyframed_values(
    values: Sequence[object],
    *,
    timeline: SectionTimeline,
    evaluation_times: tuple[float, ...],
    transition_states: tuple[SectionTimelineTransitionState, ...],
    label: str,
    equivalent: Callable[[object, object], bool],
) -> tuple[object, ...]:
    """Resolve exact render values without inventing an in-between policy."""

    frozen = tuple(values)
    if len(frozen) == len(evaluation_times):
        return frozen
    if len(frozen) != len(timeline.samples):
        raise ParallelSectionSequenceError(
            f"{label} must cover either timeline keyframes or all render frames"
        )
    sample_index_by_time = {
        item.time: index for index, item in enumerate(timeline.samples)
    }
    result: list[object] = []
    for time, transition in zip(evaluation_times, transition_states):
        exact_index = sample_index_by_time.get(time)
        if exact_index is not None:
            result.append(frozen[exact_index])
            continue
        references = tuple(
            frozen[item.reference_frame_index] for item in transition.layers
        )
        if any(not equivalent(references[0], item) for item in references[1:]):
            raise ParallelSectionSequenceError(
                f"{label} changes inside a topology crossfade; provide explicit "
                "render-frame values"
            )
        result.append(references[0])
    return tuple(result)


def _resolve_plane_patch_fits(
    timeline: SectionTimeline,
    planes: Sequence[SectionPlane],
    margin: float | None,
) -> tuple[FittedPlaneDisplayPatch | None, ...]:
    if margin is None:
        return tuple(None for _ in planes)
    resolved_margin = _finite(margin, "plane_patch_margin")
    if resolved_margin < 0.0:
        raise ParallelSectionSequenceError(
            "plane_patch_margin must be non-negative"
        )
    surface = timeline.samples[0].surface
    result: list[FittedPlaneDisplayPatch] = []
    for plane in planes:
        try:
            result.append(
                fit_plane_display_patch(
                    f"{plane.plane_id}:auto-display-patch",
                    plane,
                    (surface,),
                    margin_ratio=resolved_margin,
                )
            )
        except (PlanePatchFitError, TypeError, ValueError) as exc:
            raise ParallelSectionSequenceError(
                f"finite section-plane patch fitting failed: {exc}"
            ) from exc
    return tuple(result)


def _resolve_framing_points(
    timeline: SectionTimeline,
    evaluation_times: tuple[float, ...],
    transition_states: tuple[SectionTimelineTransitionState, ...],
    authored: Sequence[Sequence[Sequence[float]]] | None,
    plane_patch_fits: Sequence[FittedPlaneDisplayPatch | None],
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    def point3(value: object) -> tuple[float, float, float]:
        try:
            result = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ParallelSectionSequenceError(
                "framing point must contain three finite values"
            ) from exc
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ParallelSectionSequenceError(
                "framing point must contain three finite values"
            )
        return tuple(float(item) for item in result)  # type: ignore[return-value]

    render_authored: tuple[tuple[tuple[float, float, float], ...], ...] | None
    render_authored = None
    if authored is None:
        keyframes: tuple[tuple[tuple[float, float, float], ...], ...] = tuple(
            tuple(
                point3(point)
                for point in (
                    *sample.surface.characteristic_points,
                    sample.plane.point,
                )
            )
            for sample in timeline.samples
        )
    else:
        normalized = tuple(
            tuple(point3(point) for point in frame)
            for frame in authored
        )
        if len(normalized) == len(evaluation_times):
            render_authored = normalized
            keyframes = ()
        elif len(normalized) == len(timeline.samples):
            keyframes = normalized
        else:
            raise ParallelSectionSequenceError(
                "framing_points_by_frame must cover keyframes or render frames"
            )
    sample_index_by_time = {
        item.time: index for index, item in enumerate(timeline.samples)
    }
    result: list[tuple[tuple[float, float, float], ...]] = []
    if len(plane_patch_fits) != len(evaluation_times):
        raise ParallelSectionSequenceError(
            "plane patch fits must cover every evaluation frame"
        )
    for time, transition, patch_fit in zip(
        evaluation_times,
        transition_states,
        plane_patch_fits,
    ):
        exact = sample_index_by_time.get(time)
        indices = (
            (exact,)
            if exact is not None
            else tuple(item.reference_frame_index for item in transition.layers)
        )
        points = (
            set(render_authored[len(result)])
            if render_authored is not None
            else {
                point
                for index in indices
                for point in keyframes[index]
            }
        )
        points.add(point3(_timeline_plane_at_time(timeline, time).point))
        if patch_fit is not None:
            points.update(
                point3(point)
                for point in patch_fit.patch.corners(patch_fit.plane)
            )
        points.update(
            point3(
                _timeline_plane_at_time(timeline, layer.geometry_time).point
            )
            for layer in transition.layers
        )
        points.update(
            point3(point)
            for layer in transition.layers
            for point in timeline.animation.frames[
                layer.reference_frame_index
            ].section.isolated_world_points
        )
        points.update(point3(item) for item in timeline.samples[0].surface.characteristic_points)
        result.append(tuple(sorted(points)))
    return tuple(result)


def _display_signature(frame: SectionDisplayFrame) -> tuple[object, ...]:
    return tuple(
        (
            item.slot_id,
            item.role.value,
            item.source_id,
            item.topology_bank,
        )
        for item in frame.slots
    )


def _display_requires_plane_patch(frame: SectionDisplayFrame) -> bool:
    return any(
        item.role
        in {
            SectionDisplayRole.PLANE_FILL,
            SectionDisplayRole.PLANE_OUTLINE,
        }
        and item.opacity_multiplier > 0.0
        for item in frame.slots
    )


def _semantic_slot_capacities(
    timeline: SectionTimeline,
    display_frames: tuple[SectionDisplayFrame, ...],
    semantic_bank_ids: tuple[str, str],
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[tuple[str, str], int],
]:
    if len(semantic_bank_ids) != 2:
        raise ParallelSectionSequenceError(
            "semantic_bank_ids must contain exactly two ids"
        )
    bank_ids = tuple(
        _identity(item, "semantic topology bank id")
        for item in semantic_bank_ids
    )
    if len(set(bank_ids)) != 2:
        raise ParallelSectionSequenceError(
            "semantic topology bank ids must be unique"
        )
    if not display_frames:
        raise ParallelSectionSequenceError("display_frames must be non-empty")
    first = display_frames[0]
    signature = _display_signature(first)
    for frame in display_frames:
        if not isinstance(frame, SectionDisplayFrame):
            raise TypeError(
                "display_frames must contain SectionDisplayFrame values"
            )
        if frame.section_id != timeline.section_id:
            raise ParallelSectionSequenceError(
                "display frame changed section identity"
            )
        if (
            frame.catalog_digest != first.catalog_digest
            or _display_signature(frame) != signature
        ):
            raise ParallelSectionSequenceError(
                "display frames must use one immutable semantic slot catalog"
            )

    bank_counts = {item: 0 for item in bank_ids}
    point_counts = {item: 0 for item in bank_ids}
    cap_counts: dict[tuple[str, str], int] = {}
    for slot in first.slots:
        if slot.role is SectionDisplayRole.SECTION_CURVE:
            if slot.topology_bank not in bank_counts:
                raise ParallelSectionSequenceError(
                    "every section-curve slot must belong to one declared "
                    "semantic topology bank"
                )
            bank_counts[slot.topology_bank] += 1
        elif slot.role is SectionDisplayRole.SECTION_POINT:
            if slot.topology_bank not in point_counts:
                raise ParallelSectionSequenceError(
                    "every section-point slot must belong to one declared "
                    "semantic topology bank"
                )
            point_counts[slot.topology_bank] += 1
        elif slot.role is SectionDisplayRole.CAP_CHORD:
            assert slot.source_id is not None
            if slot.topology_bank not in bank_counts:
                raise ParallelSectionSequenceError(
                    "every cap-chord slot must belong to one declared semantic "
                    "topology bank"
                )
            key = (slot.topology_bank, slot.source_id)
            cap_counts[key] = cap_counts.get(key, 0) + 1
    required = timeline.animation.capacity_plan.maximum_slots
    for bank_id, count in bank_counts.items():
        if count < required:
            raise ParallelSectionSequenceError(
                f"semantic topology bank {bank_id!r} reserves {count} section "
                f"slot(s), but the fixed contract requires {required}"
            )
    required_points = max(
        item.signature.isolated_point_count
        for item in timeline.animation.frames
    )
    for bank_id, count in point_counts.items():
        if count < required_points:
            raise ParallelSectionSequenceError(
                f"semantic topology bank {bank_id!r} reserves {count} isolated "
                f"point slot(s), but the timeline requires {required_points}"
            )
    required_caps = {
        (bank_id, source_id)
        for bank_id in bank_ids
        for source_id in timeline.cap_chord_ids
    }
    if set(cap_counts) != required_caps:
        missing = sorted(required_caps - set(cap_counts))
        extra = sorted(set(cap_counts) - required_caps)
        raise ParallelSectionSequenceError(
            "cap-chord semantic slots disagree with the timeline reservation: "
            f"missing={missing!r}, extra={extra!r}"
        )
    duplicates = sorted(
        key for key, count in cap_counts.items() if count != 1
    )
    if duplicates:
        raise ParallelSectionSequenceError(
            "each semantic bank must reserve exactly one cap-chord slot per "
            f"source; invalid={duplicates!r}"
        )
    return bank_counts, point_counts, cap_counts


def _cap_chord_ids_at_geometry_time(
    timeline: SectionTimeline,
    geometry_time: float,
) -> tuple[str, ...]:
    value = _finite(geometry_time, "bank geometry time")
    exact = {
        item.time: item.active_curve_ids for item in timeline.cap_chord_states
    }.get(value)
    if exact is not None:
        return exact
    start = timeline.samples[0].time
    end = timeline.samples[-1].time
    if value < start or value > end:
        raise ParallelSectionSequenceError(
            "bank geometry time lies outside the SectionTimeline"
        )
    critical_by_id = {item.event_id: item for item in timeline.critical_events}
    active = timeline.cap_chord_states[0].active_curve_ids
    timed_events: list[tuple[float, object]] = []
    for event in timeline.cap_chord_events:
        times = {critical_by_id[item].time for item in event.critical_event_ids}
        if len(times) != 1:
            raise ParallelSectionSequenceError(
                "cap-chord event does not resolve to one analytic time"
            )
        timed_events.append((next(iter(times)), event))
    for critical_time, event in sorted(
        timed_events,
        key=lambda item: (item[0], item[1].event_id),
    ):
        if value <= critical_time:
            break
        active = timeline.cap_chord_states[
            event.right_frame_index
        ].active_curve_ids
    return active


def _bank_render_frame(
    timeline: SectionTimeline,
    state: SectionTimelineTransitionState,
    semantic_bank_ids: tuple[str, str],
) -> SectionBankRenderFrame:
    for item in state.layers:
        expected_bank = timeline.topology_frame_banks[
            item.reference_frame_index
        ]
        if item.bank_index != expected_bank:
            raise ParallelSectionSequenceError(
                "transition layer bank disagrees with its certified frame"
            )
    layers: list[SectionBankRenderLayer] = []
    for item in state.layers:
        reference = timeline.animation.frames[item.reference_frame_index]
        geometry_plane = _timeline_plane_at_time(timeline, item.geometry_time)
        try:
            boundary = compute_quadric_section_boundary(
                timeline.section_id,
                timeline.samples[0].surface,
                geometry_plane,
                context=timeline.geometry_context,
                coefficient_tolerance=timeline.coefficient_tolerance,
            )
        except (QuadricSectionError, TypeError, ValueError) as exc:
            raise ParallelSectionSequenceError(
                f"bank geometry solve failed at time {item.geometry_time!r}: {exc}"
            ) from exc
        active_cap_chords = tuple(
            sorted(curve.curve_id for curve in boundary.cap_chords)
        )
        if active_cap_chords != _cap_chord_ids_at_geometry_time(
            timeline,
            item.geometry_time,
        ):
            raise ParallelSectionSequenceError(
                "bank geometry cap chords differ from the certified timeline"
            )
        geometry_digest = _digest_json(
            {
                "sectionId": timeline.section_id,
                "surfaceId": timeline.surface_id,
                "geometryPolicyDigest": timeline.geometry_policy_digest,
                "referenceFrame": reference.to_dict(),
                "geometryTime": item.geometry_time,
                "geometryPlane": _plane_to_dict(geometry_plane),
                "resolvedBoundary": {
                    "trace": boundary.trace.to_dict(),
                    "curves": [curve.to_dict() for curve in boundary.curves],
                    "capChordIds": list(active_cap_chords),
                },
                "activeCapChordIds": list(active_cap_chords),
            }
        )
        layers.append(
            SectionBankRenderLayer(
                bank_index=item.bank_index,
                semantic_bank_id=semantic_bank_ids[item.bank_index],
                reference_frame_index=item.reference_frame_index,
                geometry_time=item.geometry_time,
                opacity=item.opacity,
                branch_count=len(boundary.trace.components),
                isolated_point_count=len(boundary.trace.isolated_world_points),
                role=item.role.value,
                geometry_digest=geometry_digest,
                active_cap_chord_ids=active_cap_chords,
            )
        )
    return SectionBankRenderFrame(
        time=state.time,
        layers=tuple(layers),
    )


def compile_parallel_section_preflight_frames(
    timeline: SectionTimeline,
    camera_samples: Sequence[ParallelCameraShotSample],
    display_frames: Sequence[SectionDisplayFrame],
    bank_render_frames: Sequence[SectionBankRenderFrame],
    *,
    transition_plan: SectionTimelineTransitionPlan,
    plane_patch_margin: float | None,
    plane_patch_fits: Sequence[FittedPlaneDisplayPatch | None],
    transition_states: Sequence[SectionTimelineTransitionState],
    planes: Sequence[SectionPlane],
    timeline_frames: Sequence[TrackedSectionFrame],
    topology_banks: Sequence[tuple[int, ...]],
    painter_orders: Sequence[PainterOrderEvidence],
    screen_transforms: Sequence[ParallelScreenTransform],
    framing_points_by_frame: Sequence[Sequence[Sequence[float]]],
    semantic_bank_ids: tuple[str, str],
) -> tuple[ParallelPreflightFrame, ...]:
    """Bind every runtime channel to one auditable preflight frame."""

    if not isinstance(timeline, SectionTimeline):
        raise TypeError("timeline must be a SectionTimeline")
    if (
        not isinstance(transition_plan, SectionTimelineTransitionPlan)
        or transition_plan.timeline is not timeline
    ):
        raise ParallelSectionSequenceError(
            "transition_plan must describe the preflight SectionTimeline"
        )
    if callable(painter_orders) or not isinstance(painter_orders, Sequence):
        raise TypeError("painter_orders must be a resolved sequence")
    values = {
        "camera_samples": tuple(camera_samples),
        "display_frames": tuple(display_frames),
        "bank_render_frames": tuple(bank_render_frames),
        "plane_patch_fits": tuple(plane_patch_fits),
        "transition_states": tuple(transition_states),
        "planes": tuple(planes),
        "timeline_frames": tuple(timeline_frames),
        "topology_banks": tuple(topology_banks),
        "painter_orders": tuple(painter_orders),
        "screen_transforms": tuple(screen_transforms),
        "framing_points_by_frame": tuple(framing_points_by_frame),
    }
    count = len(values["camera_samples"])
    if not count or any(len(items) != count for items in values.values()):
        raise ParallelSectionSequenceError(
            "all render evidence must cover the same non-empty frame grid"
        )
    cameras = values["camera_samples"]
    displays = values["display_frames"]
    bank_frames = values["bank_render_frames"]
    patch_fits = values["plane_patch_fits"]
    transitions = values["transition_states"]
    resolved_planes = values["planes"]
    tracked_frames = values["timeline_frames"]
    active_banks = values["topology_banks"]
    painters = values["painter_orders"]
    transforms = values["screen_transforms"]
    framing = values["framing_points_by_frame"]
    if not all(isinstance(item, ParallelCameraShotSample) for item in cameras):
        raise TypeError("camera_samples must contain ParallelCameraShotSample values")
    if not all(isinstance(item, SectionDisplayFrame) for item in displays):
        raise TypeError("display_frames must contain SectionDisplayFrame values")
    if not all(isinstance(item, SectionBankRenderFrame) for item in bank_frames):
        raise TypeError(
            "bank_render_frames must contain SectionBankRenderFrame values"
        )
    expected_patch_fits = _resolve_plane_patch_fits(
        timeline,
        resolved_planes,
        plane_patch_margin,
    )
    if patch_fits != expected_patch_fits:
        raise ParallelSectionSequenceError(
            "plane patch fits differ from the source-authoritative fit"
        )
    if any(
        _display_requires_plane_patch(display) and fit is None
        for display, fit in zip(displays, patch_fits)
    ):
        raise ParallelSectionSequenceError(
            "visible plane display slots require a finite plane patch"
        )
    if not all(
        isinstance(item, SectionTimelineTransitionState) for item in transitions
    ):
        raise TypeError(
            "transition_states must contain SectionTimelineTransitionState values"
        )
    if not all(isinstance(item, SectionPlane) for item in resolved_planes):
        raise TypeError("planes must contain SectionPlane values")
    if not all(isinstance(item, TrackedSectionFrame) for item in tracked_frames):
        raise TypeError("timeline_frames must contain TrackedSectionFrame values")
    if not all(isinstance(item, PainterOrderEvidence) for item in painters):
        raise TypeError("painter_orders must contain PainterOrderEvidence values")
    if any(not item.item_ids or not item.draw_order for item in painters):
        raise ParallelSectionSequenceError(
            "parallel section painter evidence must not be empty"
        )
    if not all(isinstance(item, ParallelScreenTransform) for item in transforms):
        raise TypeError(
            "screen_transforms must contain ParallelScreenTransform values"
        )

    if len(semantic_bank_ids) != 2:
        raise ParallelSectionSequenceError(
            "semantic_bank_ids must contain exactly two ids"
        )
    resolved_bank_ids = tuple(
        _identity(item, "semantic topology bank id")
        for item in semantic_bank_ids
    )
    if len(set(resolved_bank_ids)) != 2:
        raise ParallelSectionSequenceError(
            "semantic topology bank ids must be unique"
        )
    bank_counts, point_counts, cap_counts = _semantic_slot_capacities(
        timeline,
        displays,  # type: ignore[arg-type]
        resolved_bank_ids,
    )
    certifications = {
        item.topology_event_id: item
        for item in timeline.topology_certifications
    }
    topology_by_id = {
        item.event_id: item for item in timeline.animation.topology_events
    }
    critical_by_id = {item.event_id: item for item in timeline.critical_events}
    topology_by_time: dict[float, list[object]] = {}
    for event_id, event in topology_by_id.items():
        certification = certifications[event_id]
        times = {
            critical_by_id[item].time for item in certification.critical_event_ids
        }
        if len(times) != 1:
            raise ParallelSectionSequenceError(
                "topology certification does not resolve to one time"
            )
        topology_by_time.setdefault(next(iter(times)), []).append(event)
    cap_events_by_time: dict[float, list[object]] = {}
    for event in timeline.cap_chord_events:
        times = {critical_by_id[item].time for item in event.critical_event_ids}
        if len(times) != 1:
            raise ParallelSectionSequenceError(
                "cap-chord certification does not resolve to one time"
            )
        cap_events_by_time.setdefault(next(iter(times)), []).append(event)

    digesters = parallel_section_channel_digesters()
    result: list[ParallelPreflightFrame] = []
    for index in range(count):
        camera = cameras[index]
        display = displays[index]
        bank_frame = bank_frames[index]
        patch_fit = patch_fits[index]
        transition = transitions[index]
        plane = resolved_planes[index]
        tracked = tracked_frames[index]
        banks = active_banks[index]
        painter = painters[index]
        transform = transforms[index]
        points = framing[index]
        assert isinstance(camera, ParallelCameraShotSample)
        assert isinstance(display, SectionDisplayFrame)
        assert isinstance(bank_frame, SectionBankRenderFrame)
        assert isinstance(transition, SectionTimelineTransitionState)
        assert isinstance(plane, SectionPlane)
        assert isinstance(tracked, TrackedSectionFrame)
        assert isinstance(painter, PainterOrderEvidence)
        assert isinstance(transform, ParallelScreenTransform)
        expected_transition = section_timeline_transition_state_at(
            transition_plan,
            camera.time,
        )
        if transition != expected_transition:
            raise ParallelSectionSequenceError(
                "transition state differs from the canonical transition plan"
            )
        if not (
            camera.time
            == bank_frame.time
            == transition.time
        ):
            raise ParallelSectionSequenceError(
                "camera, transition, and bank render times must match exactly"
            )
        expected_plane = _timeline_plane_at_time(timeline, camera.time)
        if plane != expected_plane:
            raise ParallelSectionSequenceError(
                "evaluation plane differs from the SectionTimeline at this time"
            )
        expected_primary = timeline.animation.frames[
            _primary_reference_index(transition)
        ]
        if tracked != expected_primary:
            raise ParallelSectionSequenceError(
                "primary reference frame differs from the transition state"
            )
        expected_banks = tuple(item.bank_index for item in transition.layers)
        if banks != expected_banks:
            raise ParallelSectionSequenceError(
                "topology bank channel differs from the transition state"
            )
        expected_bank_frame = _bank_render_frame(
            timeline,
            transition,
            resolved_bank_ids,
        )
        if bank_frame != expected_bank_frame:
            raise ParallelSectionSequenceError(
                "bank render frame differs from certified transition geometry"
            )
        runtime_channels: dict[str, object] = {
            SECTION_TIMELINE_FRAME_CHANNEL: tracked,
            SECTION_PLANE_CHANNEL: plane,
            SECTION_DISPLAY_CHANNEL: display,
            SECTION_TOPOLOGY_BANK_CHANNEL: banks,
            SECTION_TRANSITION_STATE_CHANNEL: transition,
            SECTION_BANK_RENDER_CHANNEL: bank_frame,
            SECTION_PAINTER_ORDER_CHANNEL: painter,
            SECTION_PLANE_PATCH_CHANNEL: patch_fit,
            PARALLEL_SCREEN_TRANSFORM_CHANNEL: transform,
        }
        channel_digests = tuple(
            (name, digesters[name](value))
            for name, value in sorted(runtime_channels.items())
        )

        event_evidence: list[TopologyEventEvidence] = []
        for event in sorted(
            topology_by_time.get(camera.time, ()),
            key=lambda item: item.event_id,
        ):
            certification = certifications[event.event_id]
            destination_bank = timeline.topology_frame_banks[
                event.right_frame_index
            ]
            event_evidence.append(
                TopologyEventEvidence(
                    event.event_id,
                    "+".join(item.value for item in event.reasons)
                    + ":"
                    + "+".join(certification.critical_event_ids),
                    True,
                    requires_slot_bank=True,
                    slot_bank_id=resolved_bank_ids[destination_bank],
                )
            )
        for event in sorted(
            cap_events_by_time.get(camera.time, ()),
            key=lambda item: item.event_id,
        ):
            event_evidence.append(
                TopologyEventEvidence(
                    event.event_id,
                    "cap-chord:"
                    + "+".join(event.activated_curve_ids)
                    + "->"
                    + "+".join(event.deactivated_curve_ids)
                    + ":"
                    + "+".join(event.critical_event_ids),
                    True,
                )
            )

        used_by_bank = {item: 0 for item in resolved_bank_ids}
        used_points_by_bank = {item: 0 for item in resolved_bank_ids}
        for layer in bank_frame.layers:
            used_by_bank[layer.semantic_bank_id] = layer.branch_count
            used_points_by_bank[layer.semantic_bank_id] = (
                layer.isolated_point_count
            )
        capacities = [
            CapacityEvidence(bank_id, used_by_bank[bank_id], bank_counts[bank_id])
            for bank_id in resolved_bank_ids
        ]
        capacities.extend(
            CapacityEvidence(
                f"{bank_id}:isolated-points",
                used_points_by_bank[bank_id],
                point_counts[bank_id],
            )
            for bank_id in resolved_bank_ids
        )
        active_cap_ids_by_bank = {
            bank_id: set() for bank_id in resolved_bank_ids
        }
        for layer in bank_frame.layers:
            active_cap_ids_by_bank[layer.semantic_bank_id] = set(
                layer.active_cap_chord_ids
            )
        capacities.extend(
            CapacityEvidence(
                f"{bank_id}:cap-chord:{source_id}",
                1 if source_id in active_cap_ids_by_bank[bank_id] else 0,
                slot_count,
            )
            for (bank_id, source_id), slot_count in sorted(cap_counts.items())
        )
        result.append(
            ParallelPreflightFrame(
                frame_id=f"{timeline.section_id}:render-frame:{index:04d}",
                time=camera.time,
                camera=camera.state,
                screen_transform=transform,
                framing_points=tuple(tuple(point) for point in points),
                topology_events=tuple(event_evidence),
                capacities=tuple(capacities),
                painter_order=painter,
                channel_digests=channel_digests,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ParallelSectionSequence:
    timeline: SectionTimeline
    transition_plan: SectionTimelineTransitionPlan
    evaluation_times: tuple[float, ...]
    semantic_bank_ids: tuple[str, str]
    camera_samples: tuple[ParallelCameraShotSample, ...]
    display_frames: tuple[SectionDisplayFrame, ...]
    bank_render_frames: tuple[SectionBankRenderFrame, ...]
    plane_patch_margin: float | None
    plane_patch_fits: tuple[FittedPlaneDisplayPatch | None, ...]
    painter_orders: tuple[PainterOrderEvidence, ...]
    screen_transforms: tuple[ParallelScreenTransform, ...]
    preflight_limits: ParallelPreflightLimits
    preflight_frames: tuple[ParallelPreflightFrame, ...]
    preflight_report: ParallelPreflightReport
    frames: tuple[ParallelFrameState, ...]
    camera_provenance: ParallelCameraSamplingProvenance | None = None
    schema: str = PARALLEL_SECTION_SEQUENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PARALLEL_SECTION_SEQUENCE_SCHEMA:
            raise ParallelSectionSequenceError("invalid parallel section schema")
        if not isinstance(self.timeline, SectionTimeline):
            raise TypeError("timeline must be a SectionTimeline")
        if (
            not isinstance(self.transition_plan, SectionTimelineTransitionPlan)
            or self.transition_plan.timeline is not self.timeline
        ):
            raise ParallelSectionSequenceError(
                "transition_plan must describe the compiled SectionTimeline"
            )
        for field_name in (
            "evaluation_times",
            "semantic_bank_ids",
            "camera_samples",
            "display_frames",
            "bank_render_frames",
            "plane_patch_fits",
            "painter_orders",
            "screen_transforms",
            "preflight_frames",
            "frames",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        evaluation_times = tuple(
            _finite(item, "evaluation time") for item in self.evaluation_times
        )
        object.__setattr__(self, "evaluation_times", evaluation_times)
        semantic_bank_ids = tuple(
            _identity(item, "semantic topology bank id")
            for item in self.semantic_bank_ids
        )
        object.__setattr__(self, "semantic_bank_ids", semantic_bank_ids)
        count = len(self.evaluation_times)
        if not count or any(
            right <= left
            for left, right in zip(
                self.evaluation_times,
                self.evaluation_times[1:],
            )
        ):
            raise ParallelSectionSequenceError(
                "evaluation_times must be non-empty and strictly increasing"
            )
        if len(self.semantic_bank_ids) != 2 or len(set(self.semantic_bank_ids)) != 2:
            raise ParallelSectionSequenceError(
                "semantic_bank_ids must contain two unique ids"
            )
        if parallel_section_render_times(
            self.transition_plan,
            self.evaluation_times,
        ) != self.evaluation_times:
            raise ParallelSectionSequenceError(
                "evaluation_times do not form a certified transition grid"
            )
        for label, values in (
            ("camera_samples", self.camera_samples),
            ("display_frames", self.display_frames),
            ("bank_render_frames", self.bank_render_frames),
            ("plane_patch_fits", self.plane_patch_fits),
            ("painter_orders", self.painter_orders),
            ("screen_transforms", self.screen_transforms),
            ("preflight_frames", self.preflight_frames),
            ("frames", self.frames),
        ):
            if len(values) != count:
                raise ParallelSectionSequenceError(
                    f"{label} must cover every timeline frame"
                )
        for label, values, expected_type in (
            ("camera_samples", self.camera_samples, ParallelCameraShotSample),
            ("display_frames", self.display_frames, SectionDisplayFrame),
            ("bank_render_frames", self.bank_render_frames, SectionBankRenderFrame),
            ("painter_orders", self.painter_orders, PainterOrderEvidence),
            ("screen_transforms", self.screen_transforms, ParallelScreenTransform),
            ("preflight_frames", self.preflight_frames, ParallelPreflightFrame),
            ("frames", self.frames, ParallelFrameState),
        ):
            if not all(isinstance(item, expected_type) for item in values):
                raise TypeError(f"{label} contains an invalid value")
        if not all(
            item is None or isinstance(item, FittedPlaneDisplayPatch)
            for item in self.plane_patch_fits
        ):
            raise TypeError(
                "plane_patch_fits must contain FittedPlaneDisplayPatch or None"
            )
        if self.plane_patch_margin is None:
            margin = None
        else:
            margin = _finite(self.plane_patch_margin, "plane_patch_margin")
            if margin < 0.0:
                raise ParallelSectionSequenceError(
                    "plane_patch_margin must be non-negative"
                )
        object.__setattr__(self, "plane_patch_margin", margin)
        expected_planes = tuple(
            _timeline_plane_at_time(self.timeline, item)
            for item in self.evaluation_times
        )
        if self.plane_patch_fits != _resolve_plane_patch_fits(
            self.timeline,
            expected_planes,
            margin,
        ):
            raise ParallelSectionSequenceError(
                "stored plane patch fits differ from source-authoritative inputs"
            )
        if any(
            _display_requires_plane_patch(display) and fit is None
            for display, fit in zip(
                self.display_frames,
                self.plane_patch_fits,
            )
        ):
            raise ParallelSectionSequenceError(
                "visible plane display slots require a finite plane patch"
            )
        if not isinstance(self.preflight_limits, ParallelPreflightLimits):
            raise TypeError("preflight_limits must be ParallelPreflightLimits")
        if not isinstance(self.preflight_report, ParallelPreflightReport):
            raise TypeError("preflight_report must be a ParallelPreflightReport")
        sample_ids = tuple(item.sample_id for item in self.camera_samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ParallelSectionSequenceError(
                "camera sample ids must be unique"
            )
        if self.camera_provenance is not None:
            if not isinstance(
                self.camera_provenance,
                ParallelCameraSamplingProvenance,
            ):
                raise TypeError(
                    "camera_provenance must be ParallelCameraSamplingProvenance"
                )
            provenance = self.camera_provenance
            if (
                self.evaluation_times[0] < provenance.start_time
                or self.evaluation_times[-1] > provenance.end_time
            ):
                raise ParallelSectionSequenceError(
                    "evaluation grid lies outside camera source provenance"
                )
            if provenance.coverage == "exact" and (
                self.evaluation_times[0] != provenance.start_time
                or self.evaluation_times[-1] != provenance.end_time
            ):
                raise ParallelSectionSequenceError(
                    "exact camera coverage must match the evaluation span"
                )
            expected_end = (
                provenance.start_time
                + provenance.shot_sequence.total_duration
            )
            if expected_end != provenance.end_time:
                raise ParallelSectionSequenceError(
                    "camera provenance duration differs from shot_sequence"
                )
            if provenance.frame_rate is not None:
                expected_nominal = tuple(
                    item
                    for item in parallel_camera_shot_frame_times(
                        provenance.shot_sequence,
                        start_time=provenance.start_time,
                        frame_rate=provenance.frame_rate,
                    )
                    if self.evaluation_times[0]
                    <= item
                    <= self.evaluation_times[-1]
                )
                if provenance.nominal_frame_times != expected_nominal:
                    raise ParallelSectionSequenceError(
                        "nominal camera grid was not derived from source provenance"
                    )
                if len(provenance.nominal_frame_times) != count:
                    raise ParallelSectionSequenceError(
                        "nominal camera grid must map one-to-one to evaluation frames"
                    )
                for nominal, analytic in zip(
                    provenance.nominal_frame_times,
                    self.evaluation_times,
                ):
                    tolerance = _time_tolerance(nominal, analytic)
                    if abs(nominal - analytic) > (
                        1.0 / provenance.frame_rate + tolerance
                    ):
                        raise ParallelSectionSequenceError(
                            "analytic camera time is more than one physical frame "
                            "from its nominal output time"
                        )
            expected_samples = sample_parallel_camera_shot_sequence(
                provenance.shot_sequence,
                provenance.initial_camera,
                self.evaluation_times,
                start_time=provenance.start_time,
            )
            if any(
                not _camera_samples_equal(actual, expected)
                for actual, expected in zip(
                    self.camera_samples,
                    expected_samples,
                )
            ):
                raise ParallelSectionSequenceError(
                    "camera samples were not derived from source provenance"
                )
        self.preflight_report.require_accepted()
        if preflight_parallel_frames(
            self.preflight_frames,
            self.preflight_limits,
        ) != self.preflight_report:
            raise ParallelSectionSequenceError(
                "preflight report was not derived from the serialized inputs"
            )
        if tuple(item.frame_id for item in self.preflight_frames) != (
            self.preflight_report.frame_ids
        ) or tuple(item.digest for item in self.preflight_frames) != (
            self.preflight_report.frame_digests
        ):
            raise ParallelSectionSequenceError(
                "preflight report does not describe the compiled frames"
            )
        digesters = parallel_section_channel_digesters()
        for index, (state, evidence) in enumerate(
            zip(self.frames, self.preflight_frames)
        ):
            evaluation_time = self.evaluation_times[index]
            camera_sample = self.camera_samples[index]
            display = self.display_frames[index]
            bank_frame = self.bank_render_frames[index]
            plane_patch_fit = self.plane_patch_fits[index]
            painter = self.painter_orders[index]
            screen_transform = self.screen_transforms[index]
            if not (
                camera_sample.time
                == bank_frame.time
                == evidence.time
                == evaluation_time
            ):
                raise ParallelSectionSequenceError(
                    "stored camera, bank, preflight, and evaluation times differ"
                )
            if not _camera_states_equal(state.camera, camera_sample.state):
                raise ParallelSectionSequenceError(
                    "stored camera sample differs from its coordinated frame"
                )
            expected_transition = section_timeline_transition_state_at(
                self.transition_plan,
                evaluation_time,
            )
            expected_bank_frame = _bank_render_frame(
                self.timeline,
                expected_transition,
                self.semantic_bank_ids,
            )
            if bank_frame != expected_bank_frame:
                raise ParallelSectionSequenceError(
                    "stored bank render frame differs from the certified timeline"
                )
            expected_plane = _timeline_plane_at_time(
                self.timeline,
                evaluation_time,
            )
            expected_primary = self.timeline.animation.frames[
                _primary_reference_index(expected_transition)
            ]
            expected_banks = tuple(
                item.bank_index for item in expected_transition.layers
            )
            stored_channels = {
                SECTION_TIMELINE_FRAME_CHANNEL: expected_primary,
                SECTION_PLANE_CHANNEL: expected_plane,
                SECTION_DISPLAY_CHANNEL: display,
                SECTION_TOPOLOGY_BANK_CHANNEL: expected_banks,
                SECTION_TRANSITION_STATE_CHANNEL: expected_transition,
                SECTION_BANK_RENDER_CHANNEL: bank_frame,
                SECTION_PAINTER_ORDER_CHANNEL: painter,
                SECTION_PLANE_PATCH_CHANNEL: plane_patch_fit,
                PARALLEL_SCREEN_TRANSFORM_CHANNEL: screen_transform,
            }
            if (
                state.frame_id != evidence.frame_id
                or state.preflight_input_digest
                != self.preflight_report.input_digest
            ):
                raise ParallelSectionSequenceError(
                    "coordinated frame is not bound to its preflight evidence"
                )
            if state.channel(PARALLEL_PREFLIGHT_FRAME_CHANNEL) is not evidence:
                raise ParallelSectionSequenceError(
                    "coordinated frame lost its exact preflight evidence"
                )
            if not _camera_states_equal(evidence.camera, camera_sample.state):
                raise ParallelSectionSequenceError(
                    "preflight camera differs from the stored camera sample"
                )
            if evidence.screen_transform != screen_transform:
                raise ParallelSectionSequenceError(
                    "preflight screen transform differs from the stored value"
                )
            if evidence.painter_order != painter:
                raise ParallelSectionSequenceError(
                    "preflight painter order differs from the stored value"
                )
            if set(state.channels) != {
                PARALLEL_PREFLIGHT_FRAME_CHANNEL,
                *stored_channels,
            }:
                raise ParallelSectionSequenceError(
                    "coordinated frame contains unbound runtime channels"
                )
            for channel_name, expected_value in stored_channels.items():
                actual_value = state.channel(channel_name)
                if digesters[channel_name](actual_value) != digesters[
                    channel_name
                ](expected_value):
                    raise ParallelSectionSequenceError(
                        f"stored channel {channel_name!r} differs from its "
                        "coordinated frame"
                    )
            for channel_name, expected_digest in evidence.channel_digests:
                actual_digest = digesters[channel_name](
                    state.channel(channel_name)
                )
                if actual_digest != expected_digest:
                    raise ParallelSectionSequenceError(
                        f"coordinated channel {channel_name!r} lost preflight binding"
                    )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "timeline": self.timeline.to_dict(),
            "transitionPlan": self.transition_plan.to_dict(),
            "evaluationTimes": list(self.evaluation_times),
            "semanticBankIds": list(self.semantic_bank_ids),
            "cameraSamples": [item.to_dict() for item in self.camera_samples],
            "displayFrames": [item.to_dict() for item in self.display_frames],
            "bankRenderFrames": [
                item.to_dict() for item in self.bank_render_frames
            ],
            "planePatchMargin": self.plane_patch_margin,
            "planePatchFits": [
                None if item is None else item.to_dict()
                for item in self.plane_patch_fits
            ],
            "painterOrders": [item.to_dict() for item in self.painter_orders],
            "screenTransforms": [
                item.to_dict() for item in self.screen_transforms
            ],
            "preflightLimits": self.preflight_limits.to_dict(),
            "preflightFrames": [item.to_dict() for item in self.preflight_frames],
            "preflight": self.preflight_report.to_dict(),
            "cameraProvenance": (
                None
                if self.camera_provenance is None
                else self.camera_provenance.to_dict()
            ),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def compile_parallel_section_sequence(
    timeline: SectionTimeline,
    camera_samples: Sequence[ParallelCameraShotSample],
    display_frames: Sequence[SectionDisplayFrame],
    *,
    limits: ParallelPreflightLimits,
    painter_orders: Sequence[PainterOrderEvidence],
    semantic_bank_ids: tuple[str, str],
    render_times: Sequence[float] | None = None,
    plane_patch_margin: float | None = None,
    screen_transforms: Sequence[ParallelScreenTransform] | None = None,
    framing_points_by_frame: Sequence[Sequence[Sequence[float]]] | None = None,
    transition_fraction: float = 0.25,
    transition_mode: SectionTimelineTransitionMode | str = (
        SectionTimelineTransitionMode.CROSSFADE
    ),
    camera_provenance: ParallelCameraSamplingProvenance | None = None,
) -> ParallelSectionSequence:
    """Compile, require joint preflight, then emit coordinated frame states."""

    if not isinstance(timeline, SectionTimeline):
        raise TypeError("timeline must be a SectionTimeline")
    if not isinstance(limits, ParallelPreflightLimits):
        raise TypeError("limits must be ParallelPreflightLimits")
    if not callable(painter_orders) and not isinstance(painter_orders, Sequence):
        raise TypeError("painter_orders must be a sequence or callable provider")
    if len(semantic_bank_ids) != 2:
        raise ParallelSectionSequenceError(
            "semantic_bank_ids must contain exactly two ids"
        )
    resolved_bank_ids = tuple(
        _identity(item, "semantic topology bank id")
        for item in semantic_bank_ids
    )
    if len(set(resolved_bank_ids)) != 2:
        raise ParallelSectionSequenceError(
            "semantic topology bank ids must be unique"
        )
    transition_plan = build_section_timeline_transition_plan(
        timeline,
        transition_fraction=transition_fraction,
        mode=transition_mode,
    )
    evaluation_times = parallel_section_render_times(
        transition_plan,
        render_times,
    )
    cameras = tuple(camera_samples)
    if len(cameras) != len(evaluation_times) or not all(
        isinstance(item, ParallelCameraShotSample) for item in cameras
    ):
        raise ParallelSectionSequenceError(
            "camera_samples must contain one sample for every render time"
        )
    if tuple(item.time for item in cameras) != evaluation_times:
        raise ParallelSectionSequenceError(
            "camera sample times must exactly match the certified render grid"
        )
    planes = tuple(
        _timeline_plane_at_time(timeline, item) for item in evaluation_times
    )
    plane_patch_fits = _resolve_plane_patch_fits(
        timeline,
        planes,
        plane_patch_margin,
    )
    transition_states = tuple(
        section_timeline_transition_state_at(transition_plan, item)
        for item in evaluation_times
    )
    authored_displays = tuple(display_frames)
    resolved_displays = _resolve_keyframed_values(
        authored_displays,
        timeline=timeline,
        evaluation_times=evaluation_times,
        transition_states=transition_states,
        label="display_frames",
        equivalent=lambda left, right: (
            isinstance(left, SectionDisplayFrame)
            and isinstance(right, SectionDisplayFrame)
            and left.digest == right.digest
        ),
    )
    if not all(isinstance(item, SectionDisplayFrame) for item in resolved_displays):
        raise TypeError("display_frames must contain SectionDisplayFrame values")
    if callable(painter_orders):
        try:
            resolved_painters = tuple(
                painter_orders(time, camera.state, plane)
                for time, camera, plane in zip(
                    evaluation_times,
                    cameras,
                    planes,
                )
            )
        except Exception as exc:
            raise ParallelSectionSequenceError(
                f"painter_order provider failed: {exc}"
            ) from exc
    else:
        resolved_painters = tuple(painter_orders)
        if len(resolved_painters) != len(evaluation_times):
            raise ParallelSectionSequenceError(
                "painter_orders must cover every render frame; use a provider "
                "for view-dependent automatic occlusion"
            )
    if not all(isinstance(item, PainterOrderEvidence) for item in resolved_painters):
        raise TypeError("painter_orders must contain PainterOrderEvidence values")
    authored_transforms: tuple[object, ...] = (
        tuple(ParallelScreenTransform() for _ in timeline.samples)
        if screen_transforms is None
        else tuple(screen_transforms)
    )
    resolved_transforms = _resolve_keyframed_values(
        authored_transforms,
        timeline=timeline,
        evaluation_times=evaluation_times,
        transition_states=transition_states,
        label="screen_transforms",
        equivalent=lambda left, right: left == right,
    )
    if not all(
        isinstance(item, ParallelScreenTransform) for item in resolved_transforms
    ):
        raise TypeError(
            "screen_transforms must contain ParallelScreenTransform values"
        )
    tracked_frames = tuple(
        timeline.animation.frames[_primary_reference_index(item)]
        for item in transition_states
    )
    topology_banks = tuple(
        tuple(layer.bank_index for layer in item.layers)
        for item in transition_states
    )
    bank_frames = tuple(
        _bank_render_frame(timeline, item, resolved_bank_ids)
        for item in transition_states
    )
    framing = _resolve_framing_points(
        timeline,
        evaluation_times,
        transition_states,
        framing_points_by_frame,
        plane_patch_fits,
    )
    preflight_frames = compile_parallel_section_preflight_frames(
        timeline,
        cameras,
        resolved_displays,  # type: ignore[arg-type]
        bank_frames,
        transition_plan=transition_plan,
        plane_patch_margin=plane_patch_margin,
        plane_patch_fits=plane_patch_fits,
        transition_states=transition_states,
        planes=planes,
        timeline_frames=tracked_frames,
        topology_banks=topology_banks,
        painter_orders=resolved_painters,  # type: ignore[arg-type]
        screen_transforms=resolved_transforms,  # type: ignore[arg-type]
        framing_points_by_frame=framing,
        semantic_bank_ids=resolved_bank_ids,
    )
    report = preflight_parallel_frames(preflight_frames, limits)
    report.require_accepted()
    frames = tuple(
        ParallelFrameState(
            camera=camera.state,
            channels={
                PARALLEL_PREFLIGHT_FRAME_CHANNEL: evidence,
                SECTION_TIMELINE_FRAME_CHANNEL: tracked_frames[index],
                SECTION_PLANE_CHANNEL: planes[index],
                SECTION_DISPLAY_CHANNEL: resolved_displays[index],
                SECTION_TOPOLOGY_BANK_CHANNEL: topology_banks[index],
                SECTION_TRANSITION_STATE_CHANNEL: transition_states[index],
                SECTION_BANK_RENDER_CHANNEL: bank_frames[index],
                SECTION_PAINTER_ORDER_CHANNEL: resolved_painters[index],
                SECTION_PLANE_PATCH_CHANNEL: plane_patch_fits[index],
                PARALLEL_SCREEN_TRANSFORM_CHANNEL: resolved_transforms[index],
            },
            frame_id=evidence.frame_id,
            preflight_input_digest=report.input_digest,
        )
        for index, (camera, evidence) in enumerate(
            zip(cameras, preflight_frames)
        )
    )
    return ParallelSectionSequence(
        timeline=timeline,
        transition_plan=transition_plan,
        evaluation_times=evaluation_times,
        semantic_bank_ids=resolved_bank_ids,
        camera_samples=cameras,
        display_frames=tuple(resolved_displays),  # type: ignore[arg-type]
        bank_render_frames=bank_frames,
        plane_patch_margin=plane_patch_margin,
        plane_patch_fits=plane_patch_fits,
        painter_orders=tuple(resolved_painters),
        screen_transforms=tuple(resolved_transforms),  # type: ignore[arg-type]
        preflight_limits=limits,
        preflight_frames=preflight_frames,
        preflight_report=report,
        frames=frames,
        camera_provenance=camera_provenance,
    )


def compile_parallel_section_sequence_from_shots(
    timeline: SectionTimeline,
    shot_sequence: ParallelCameraShotSequence,
    initial_camera: ParallelCameraState,
    display_frames: Sequence[SectionDisplayFrame],
    *,
    limits: ParallelPreflightLimits,
    painter_orders: Sequence[PainterOrderEvidence]
    | Callable[[float, ParallelCameraState, SectionPlane], PainterOrderEvidence],
    semantic_bank_ids: tuple[str, str],
    start_time: float = 0.0,
    coverage: str = "exact",
    render_times: Sequence[float] | None = None,
    frame_rate: float | None = None,
    plane_patch_margin: float | None = None,
    screen_transforms: Sequence[ParallelScreenTransform] | None = None,
    framing_points_by_frame: Sequence[Sequence[Sequence[float]]] | None = None,
    transition_fraction: float = 0.25,
    transition_mode: SectionTimelineTransitionMode | str = (
        SectionTimelineTransitionMode.CROSSFADE
    ),
) -> ParallelSectionSequence:
    """Compile a full render grid directly from source-authoritative shots."""

    if not isinstance(timeline, SectionTimeline):
        raise TypeError("timeline must be a SectionTimeline")
    if not isinstance(shot_sequence, ParallelCameraShotSequence):
        raise TypeError("shot_sequence must be a ParallelCameraShotSequence")
    if not isinstance(initial_camera, ParallelCameraState):
        raise TypeError("initial_camera must be a ParallelCameraState")
    if not callable(painter_orders) and not isinstance(painter_orders, Sequence):
        raise TypeError("painter_orders must be a sequence or callable provider")
    plan = build_section_timeline_transition_plan(
        timeline,
        transition_fraction=transition_fraction,
        mode=transition_mode,
    )
    authored_start = _finite(start_time, "start_time")
    authored_end = authored_start + shot_sequence.total_duration
    timeline_start = timeline.samples[0].time
    timeline_end = timeline.samples[-1].time
    if coverage not in {"exact", "window"}:
        raise ParallelSectionSequenceError(
            "coverage must be 'exact' or 'window'"
        )
    if coverage == "exact" and (
        authored_start != timeline_start or authored_end != timeline_end
    ):
        raise ParallelSectionSequenceError(
            "exact camera coverage must match the SectionTimeline span"
        )
    if coverage == "window" and (
        authored_start > timeline_start or authored_end < timeline_end
    ):
        raise ParallelSectionSequenceError(
            "window camera coverage must contain the SectionTimeline span"
        )
    if render_times is not None and frame_rate is not None:
        raise ParallelSectionSequenceError(
            "provide either render_times or frame_rate, not both"
        )
    evaluation_times = (
        parallel_section_frame_grid(
            plan,
            frame_rate,
            shot_sequence=shot_sequence,
            start_time=start_time,
        )
        if frame_rate is not None
        else parallel_section_render_times(plan, render_times)
    )
    nominal_frame_times = (
        tuple(
            item
            for item in parallel_camera_shot_frame_times(
                shot_sequence,
                start_time=authored_start,
                frame_rate=frame_rate,
            )
            if timeline_start <= item <= timeline_end
        )
        if frame_rate is not None
        else ()
    )
    if frame_rate is not None and len(nominal_frame_times) != len(
        evaluation_times
    ):
        raise ParallelSectionSequenceError(
            "analytic replacements changed the physical camera frame count"
        )
    provenance = ParallelCameraSamplingProvenance(
        sequence_digest="sha256:"
        + hashlib.sha256(
            canonical_parallel_camera_shot_sequence_json(
                shot_sequence
            ).encode("utf-8")
        ).hexdigest(),
        shot_sequence=shot_sequence,
        initial_camera=initial_camera,
        start_time=authored_start,
        end_time=authored_end,
        coverage=coverage,
        frame_rate=frame_rate,
        nominal_frame_times=nominal_frame_times,
    )
    cameras = sample_parallel_camera_shot_sequence(
        shot_sequence,
        initial_camera,
        evaluation_times,
        start_time=start_time,
    )
    return compile_parallel_section_sequence(
        timeline,
        cameras,
        tuple(display_frames),
        limits=limits,
        painter_orders=(
            painter_orders if callable(painter_orders) else tuple(painter_orders)
        ),
        semantic_bank_ids=semantic_bank_ids,
        render_times=evaluation_times,
        plane_patch_margin=plane_patch_margin,
        screen_transforms=(
            None if screen_transforms is None else tuple(screen_transforms)
        ),
        framing_points_by_frame=(
            None
            if framing_points_by_frame is None
            else tuple(framing_points_by_frame)
        ),
        transition_fraction=transition_fraction,
        transition_mode=transition_mode,
        camera_provenance=provenance,
    )


def parallel_section_preflight_gate(
    sequence: ParallelSectionSequence,
) -> ParallelPreflightGate:
    if not isinstance(sequence, ParallelSectionSequence):
        raise TypeError("sequence must be a ParallelSectionSequence")
    return ParallelPreflightGate(
        sequence.preflight_report,
        channel_digesters=parallel_section_channel_digesters(),
    )


def parallel_screen_transform_guard(
    provider: Callable[[], ParallelScreenTransform],
    *,
    participant_id: str = "parallel-screen-transform-guard",
) -> ParallelFrameParticipant[ParallelFrameState]:
    """Fail a frame if live renderer affine terms differ from preflight."""

    if not callable(provider):
        raise TypeError("provider must be callable")

    def prepare(frame: ParallelFrameState) -> None:
        expected = frame.channel(PARALLEL_SCREEN_TRANSFORM_CHANNEL)
        if not isinstance(expected, ParallelScreenTransform):
            raise TypeError(
                "screen-transform channel must contain ParallelScreenTransform"
            )
        actual = provider()
        if not isinstance(actual, ParallelScreenTransform):
            raise TypeError("provider must return ParallelScreenTransform")
        if actual != expected:
            raise ParallelSectionSequenceError(
                "live renderer screen transform differs from preflight evidence"
            )
        return None

    return ParallelFrameParticipant(
        participant_id=participant_id,
        phase=ParallelFramePhase.PREFLIGHT,
        prepare=prepare,
        snapshot=lambda: None,
        commit=lambda _value: None,
        rollback=lambda _value: None,
        binding_kind=ParallelFrameBindingKind.SCREEN_TRANSFORM_GUARD,
    )


def section_painter_order_participant(
    target: object,
    *,
    participant_id: str = "section-painter-order",
) -> ParallelFrameParticipant[ParallelFrameState]:
    """Atomically apply the exact painter plan certified for one render frame."""

    snapshot = getattr(target, "snapshot_section_painter_order_state", None)
    apply = getattr(target, "apply_section_painter_order", None)
    restore = getattr(target, "restore_section_painter_order_state", None)
    if not all(callable(item) for item in (snapshot, apply, restore)):
        raise TypeError(
            "painter target must provide snapshot, apply, and restore methods"
        )

    def prepare(frame: ParallelFrameState) -> PainterOrderEvidence:
        value = frame.channel(SECTION_PAINTER_ORDER_CHANNEL)
        if not isinstance(value, PainterOrderEvidence):
            raise TypeError(
                "painter-order channel must contain PainterOrderEvidence"
            )
        return value

    def commit(value: object) -> None:
        if not isinstance(value, PainterOrderEvidence):
            raise TypeError("prepared painter order must be PainterOrderEvidence")
        apply(value)

    return ParallelFrameParticipant(
        participant_id=participant_id,
        phase=ParallelFramePhase.VISIBILITY,
        prepare=prepare,
        snapshot=snapshot,
        commit=commit,
        rollback=restore,
        binding_kind=ParallelFrameBindingKind.SECTION_PAINTER,
    )


def section_plane_patch_participant(
    target: object,
    *,
    participant_id: str = "section-plane-patch",
) -> ParallelFrameParticipant[ParallelFrameState]:
    """Apply the exact finite display patch used by framing preflight."""

    snapshot = getattr(target, "snapshot_section_plane_patch_state", None)
    apply = getattr(target, "apply_section_plane_patch_fit", None)
    restore = getattr(target, "restore_section_plane_patch_state", None)
    if not all(callable(item) for item in (snapshot, apply, restore)):
        raise TypeError(
            "plane patch target must provide snapshot, apply, and restore methods"
        )

    def prepare(
        frame: ParallelFrameState,
    ) -> FittedPlaneDisplayPatch | None:
        value = frame.channel(SECTION_PLANE_PATCH_CHANNEL)
        if value is not None and not isinstance(value, FittedPlaneDisplayPatch):
            raise TypeError(
                "plane-patch channel must contain FittedPlaneDisplayPatch or None"
            )
        return value

    def commit(value: object) -> None:
        if value is not None and not isinstance(value, FittedPlaneDisplayPatch):
            raise TypeError(
                "prepared plane patch must be FittedPlaneDisplayPatch or None"
            )
        apply(value)

    return ParallelFrameParticipant(
        participant_id=participant_id,
        phase=ParallelFramePhase.GEOMETRY,
        prepare=prepare,
        snapshot=snapshot,
        commit=commit,
        rollback=restore,
        binding_kind=ParallelFrameBindingKind.SECTION_PLANE_PATCH,
    )


def section_display_frame_participant(
    target: object,
    *,
    channel_name: str = SECTION_DISPLAY_CHANNEL,
    participant_id: str = "section-semantic-display",
) -> ParallelFrameParticipant[ParallelFrameState]:
    """Adapt a future Rig/display binding through an explicit full transaction."""

    channel = _identity(channel_name, "channel_name")
    snapshot = getattr(target, "snapshot_section_display_state", None)
    apply = getattr(target, "apply_section_display_frame", None)
    restore = getattr(target, "restore_section_display_state", None)
    if not all(callable(item) for item in (snapshot, apply, restore)):
        raise TypeError(
            "display target must provide snapshot, apply, and restore methods"
        )

    def prepare(frame: ParallelFrameState) -> SectionDisplayFrame:
        if not isinstance(frame, ParallelFrameState):
            raise TypeError("display participant requires ParallelFrameState")
        value = frame.channel(channel)
        if not isinstance(value, SectionDisplayFrame):
            raise TypeError("display channel must contain SectionDisplayFrame")
        return value

    def commit(value: object) -> None:
        if not isinstance(value, SectionDisplayFrame):
            raise TypeError("prepared display value must be SectionDisplayFrame")
        apply(value)

    return ParallelFrameParticipant(
        participant_id=participant_id,
        phase=ParallelFramePhase.PAINT,
        prepare=prepare,
        snapshot=snapshot,
        commit=commit,
        rollback=restore,
        binding_kind=ParallelFrameBindingKind.SECTION_DISPLAY,
    )


__all__ = [
    "PARALLEL_SECTION_SEQUENCE_SCHEMA",
    "PARALLEL_SCREEN_TRANSFORM_CHANNEL",
    "SECTION_BANK_RENDER_CHANNEL",
    "SECTION_DISPLAY_CHANNEL",
    "SECTION_EVALUATION_PLANE_CHANNEL",
    "SECTION_PAINTER_ORDER_CHANNEL",
    "SECTION_PLANE_PATCH_CHANNEL",
    "SECTION_PLANE_CHANNEL",
    "SECTION_PRIMARY_REFERENCE_FRAME_CHANNEL",
    "SECTION_TIMELINE_FRAME_CHANNEL",
    "SECTION_TOPOLOGY_BANK_CHANNEL",
    "SECTION_TRANSITION_STATE_CHANNEL",
    "ParallelCameraShotSample",
    "ParallelCameraShotSamplePhase",
    "ParallelCameraSamplingProvenance",
    "ParallelSectionSequence",
    "ParallelSectionSequenceError",
    "compile_parallel_section_preflight_frames",
    "compile_parallel_section_sequence",
    "compile_parallel_section_sequence_from_shots",
    "parallel_screen_transform_guard",
    "parallel_camera_shot_frame_times",
    "parallel_section_channel_digesters",
    "parallel_section_frame_grid",
    "parallel_section_preflight_gate",
    "parallel_section_render_times",
    "sample_parallel_camera_shot_sequence",
    "section_bank_frame_participant",
    "section_display_frame_participant",
    "section_plane_patch_participant",
    "section_painter_order_participant",
]
