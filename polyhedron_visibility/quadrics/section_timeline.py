"""Renderer-neutral compilation of analytic quadric-section timelines.

A timeline is an ordered sequence of explicit, unambiguous plane motions:
rigid axis-angle rotation and parallel translation.  Every segment contributes
its analytic critical events.  Their samples are stitched into one strictly
increasing global sequence and passed through branch tracking exactly once, so
lineage and capacity are authoritative for the complete authored timeline.

This module deliberately has no Manim, camera, painter-band, or Scene binding.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from typing import Mapping, Sequence, TypeAlias

from ..geometry import GeometryContext, ResolvedGeometryContext
from .animation import (
    SectionAnimationSample,
    SectionAnimationTrace,
    TopologyEventKind,
    track_quadric_section_animation,
)
from .contract import ConeModel, ConeSpec, CylinderSpec, SectionPlane, SphereSpec
from .parallel_plane_motion import (
    ParallelPlaneMotionSchedule,
    ParallelPlaneTranslation,
    compute_parallel_plane_motion_schedule,
)
from .plane_motion import (
    AxisAnglePlaneMotion,
    PlaneMotionSchedule,
    compute_plane_motion_schedule,
)
from .sections import (
    QuadricSurfaceSpec,
    compute_quadric_section_boundary,
    section_cap_chord_curve_ids,
)


SECTION_TIMELINE_SCHEMA = "manim-quadric-section-timeline/v1"
_CAP_CHORD_EVENT_KINDS = frozenset(
    {
        "cylinder_trim_tangency",
        "cone_trim_tangency",
    }
)
_SURFACE_CRITICAL_KINDS = {
    SphereSpec: frozenset({"sphere_tangency"}),
    CylinderSpec: frozenset(
        {"cylinder_axis_parallel", "cylinder_trim_tangency"}
    ),
    ConeSpec: frozenset(
        {"cone_parabolic", "cone_apex_degeneracy", "cone_trim_tangency"}
    ),
}
_SUPPORTING_CONIC_EVENT_KINDS = {
    SphereSpec: frozenset({"sphere_tangency"}),
    CylinderSpec: frozenset({"cylinder_axis_parallel"}),
    ConeSpec: frozenset({"cone_parabolic", "cone_apex_degeneracy"}),
}
_FINITE_BOUNDARY_EVENT_KINDS = {
    SphereSpec: frozenset({"sphere_tangency"}),
    CylinderSpec: frozenset({"cylinder_trim_tangency"}),
    ConeSpec: frozenset({"cone_trim_tangency"}),
}
_SUPPORTING_CONIC_REASONS = frozenset(
    {
        TopologyEventKind.CONIC_FAMILY_CHANGED,
        TopologyEventKind.ENTERED_DEGENERACY,
        TopologyEventKind.EXITED_DEGENERACY,
    }
)


SectionTimelineMotion: TypeAlias = AxisAnglePlaneMotion | ParallelPlaneTranslation
SectionTimelineSegmentSchedule: TypeAlias = (
    PlaneMotionSchedule | ParallelPlaneMotionSchedule
)


class SectionTimelineError(ValueError):
    """A complete analytic section timeline cannot be certified safely."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SectionTimelineError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SectionTimelineError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SectionTimelineError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise SectionTimelineError(f"{label} must be finite")
    return result


def _geometry_context_to_dict(
    context: GeometryContext | ResolvedGeometryContext | None,
) -> dict[str, object] | None:
    if context is None:
        return None
    if isinstance(context, ResolvedGeometryContext):
        return {"kind": "resolved", **context.to_dict()}
    if not isinstance(context, GeometryContext):
        raise TypeError(
            "geometry_context must be GeometryContext or ResolvedGeometryContext"
        )
    policy = context.tolerance
    return {
        "kind": "unresolved",
        "tolerance": {
            "relative": policy.relative,
            "absoluteFloor": policy.absolute_floor,
            "angular": policy.angular,
            "boundaryFactor": policy.boundary_factor,
            "depthFactor": policy.depth_factor,
        },
        "screenTolerance": context.screen_tolerance,
        "overrides": {
            quantity.value: value
            for quantity, value in context.overrides.items()
        },
    }


def _geometry_policy_digest(
    context: GeometryContext | ResolvedGeometryContext | None,
    coefficient_tolerance: float | None,
) -> str:
    payload = json.dumps(
        {
            "geometryContext": _geometry_context_to_dict(context),
            "coefficientTolerance": coefficient_tolerance,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _surface(value: object) -> QuadricSurfaceSpec:
    if not isinstance(value, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    if isinstance(value, ConeSpec) and value.model in {
        ConeModel.OPEN_DOUBLE,
        ConeModel.ANALYTIC_DOUBLE,
    }:
        raise SectionTimelineError(
            "SectionTimeline requires one directly renderable cone nappe"
        )
    return value


def _motion_id(motion: SectionTimelineMotion) -> str:
    return motion.motion_id


def _motion_start_time(motion: SectionTimelineMotion) -> float:
    return motion.time_at(0.0)


def _motion_end_time(motion: SectionTimelineMotion) -> float:
    return motion.time_at(1.0)


def _motion_start_plane(motion: SectionTimelineMotion) -> SectionPlane:
    return motion.plane_at(0.0)


def _motion_end_plane(motion: SectionTimelineMotion) -> SectionPlane:
    return motion.plane_at(1.0)


def _schedule_kind(schedule: SectionTimelineSegmentSchedule) -> str:
    return (
        "axis_angle"
        if isinstance(schedule, PlaneMotionSchedule)
        else "parallel_translation"
    )


@dataclass(frozen=True, slots=True)
class SectionTimelineCriticalEvent:
    event_id: str
    source_event_id: str
    segment_id: str
    segment_index: int
    segment_kind: str
    progress: float
    time: float
    kinds: tuple[str, ...]
    equations: tuple[str, ...]
    persistent: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identity(self.event_id, "event_id"))
        object.__setattr__(
            self,
            "source_event_id",
            _identity(self.source_event_id, "source_event_id"),
        )
        object.__setattr__(self, "segment_id", _identity(self.segment_id, "segment_id"))
        if isinstance(self.segment_index, bool) or self.segment_index < 0:
            raise SectionTimelineError("segment_index must be non-negative")
        if self.segment_kind not in {"axis_angle", "parallel_translation"}:
            raise SectionTimelineError("invalid segment_kind")
        progress = _finite(self.progress, "critical progress")
        if progress < 0.0 or progress > 1.0:
            raise SectionTimelineError("critical progress must lie in [0, 1]")
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "time", _finite(self.time, "critical time"))
        kinds = tuple(
            sorted(set(_identity(item, "critical kind") for item in self.kinds))
        )
        if not kinds or kinds != self.kinds:
            raise SectionTimelineError("critical kinds must be unique and canonical")
        equations = tuple(
            sorted(set(_identity(item, "equation") for item in self.equations))
        )
        if not equations or equations != self.equations:
            raise SectionTimelineError(
                "critical equations must be unique and canonical"
            )
        if not isinstance(self.persistent, bool):
            raise TypeError("persistent must be a bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "eventId": self.event_id,
            "sourceEventId": self.source_event_id,
            "segmentId": self.segment_id,
            "segmentIndex": self.segment_index,
            "segmentKind": self.segment_kind,
            "progress": self.progress,
            "time": self.time,
            "kinds": list(self.kinds),
            "equations": list(self.equations),
            "persistent": self.persistent,
        }


@dataclass(frozen=True, slots=True)
class SectionTimelineTopologyCertification:
    topology_event_id: str
    critical_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "topology_event_id",
            _identity(self.topology_event_id, "topology_event_id"),
        )
        identities = tuple(
            sorted(
                set(
                    _identity(item, "critical_event_id")
                    for item in self.critical_event_ids
                )
            )
        )
        if not identities or identities != self.critical_event_ids:
            raise SectionTimelineError(
                "critical_event_ids must be non-empty, unique, and canonical"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "topologyEventId": self.topology_event_id,
            "criticalEventIds": list(self.critical_event_ids),
        }


@dataclass(frozen=True, slots=True)
class SectionTimelineCapChordState:
    frame_index: int
    time: float
    active_curve_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise SectionTimelineError("frame_index must be non-negative")
        object.__setattr__(self, "time", _finite(self.time, "cap-chord state time"))
        identities = tuple(
            sorted(
                set(
                    _identity(item, "active cap-chord id")
                    for item in self.active_curve_ids
                )
            )
        )
        if identities != self.active_curve_ids:
            raise SectionTimelineError(
                "active_curve_ids must be unique and canonical"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "frameIndex": self.frame_index,
            "time": self.time,
            "activeCurveIds": list(self.active_curve_ids),
        }


@dataclass(frozen=True, slots=True)
class SectionTimelineCapChordEvent:
    event_id: str
    left_frame_index: int
    right_frame_index: int
    left_time: float
    right_time: float
    activated_curve_ids: tuple[str, ...]
    deactivated_curve_ids: tuple[str, ...]
    critical_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identity(self.event_id, "event_id"))
        if (
            isinstance(self.left_frame_index, bool)
            or self.left_frame_index < 0
            or self.right_frame_index != self.left_frame_index + 1
        ):
            raise SectionTimelineError(
                "cap-chord event must bracket adjacent frames"
            )
        left = _finite(self.left_time, "left_time")
        right = _finite(self.right_time, "right_time")
        if right <= left:
            raise SectionTimelineError("cap-chord event times must increase")
        object.__setattr__(self, "left_time", left)
        object.__setattr__(self, "right_time", right)
        activated = tuple(
            sorted(
                set(
                    _identity(item, "activated cap-chord id")
                    for item in self.activated_curve_ids
                )
            )
        )
        deactivated = tuple(
            sorted(
                set(
                    _identity(item, "deactivated cap-chord id")
                    for item in self.deactivated_curve_ids
                )
            )
        )
        if activated != self.activated_curve_ids:
            raise SectionTimelineError(
                "activated_curve_ids must be unique and canonical"
            )
        if deactivated != self.deactivated_curve_ids:
            raise SectionTimelineError(
                "deactivated_curve_ids must be unique and canonical"
            )
        if not activated and not deactivated:
            raise SectionTimelineError("cap-chord event must change activation")
        if set(activated) & set(deactivated):
            raise SectionTimelineError(
                "one cap chord cannot activate and deactivate in the same event"
            )
        critical = tuple(
            sorted(
                set(
                    _identity(item, "critical_event_id")
                    for item in self.critical_event_ids
                )
            )
        )
        if not critical or critical != self.critical_event_ids:
            raise SectionTimelineError(
                "critical_event_ids must be non-empty, unique, and canonical"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "eventId": self.event_id,
            "leftFrameIndex": self.left_frame_index,
            "rightFrameIndex": self.right_frame_index,
            "leftTime": self.left_time,
            "rightTime": self.right_time,
            "activatedCurveIds": list(self.activated_curve_ids),
            "deactivatedCurveIds": list(self.deactivated_curve_ids),
            "criticalEventIds": list(self.critical_event_ids),
        }


@dataclass(frozen=True, slots=True)
class SectionTimeline:
    section_id: str
    surface_id: str
    plane_id: str
    segment_schedules: tuple[SectionTimelineSegmentSchedule, ...]
    samples: tuple[SectionAnimationSample, ...]
    critical_events: tuple[SectionTimelineCriticalEvent, ...]
    animation: SectionAnimationTrace
    topology_certifications: tuple[SectionTimelineTopologyCertification, ...]
    topology_frame_banks: tuple[int, ...]
    cap_chord_ids: tuple[str, ...]
    cap_chord_states: tuple[SectionTimelineCapChordState, ...]
    cap_chord_events: tuple[SectionTimelineCapChordEvent, ...]
    geometry_context: GeometryContext | ResolvedGeometryContext | None = None
    coefficient_tolerance: float | None = None
    geometry_policy_digest: str = ""
    schema: str = SECTION_TIMELINE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SECTION_TIMELINE_SCHEMA:
            raise SectionTimelineError("invalid SectionTimeline schema")
        object.__setattr__(
            self,
            "section_id",
            _identity(self.section_id, "section_id"),
        )
        object.__setattr__(
            self,
            "surface_id",
            _identity(self.surface_id, "surface_id"),
        )
        object.__setattr__(self, "plane_id", _identity(self.plane_id, "plane_id"))
        if self.geometry_context is not None and not isinstance(
            self.geometry_context,
            (GeometryContext, ResolvedGeometryContext),
        ):
            raise TypeError(
                "geometry_context must be GeometryContext or ResolvedGeometryContext"
            )
        coefficient_tolerance = self.coefficient_tolerance
        if coefficient_tolerance is not None:
            coefficient_tolerance = _finite(
                coefficient_tolerance,
                "coefficient_tolerance",
            )
            if coefficient_tolerance <= 0.0:
                raise SectionTimelineError(
                    "coefficient_tolerance must be finite and positive"
                )
        object.__setattr__(
            self,
            "coefficient_tolerance",
            coefficient_tolerance,
        )
        expected_policy_digest = _geometry_policy_digest(
            self.geometry_context,
            coefficient_tolerance,
        )
        if self.geometry_policy_digest:
            if self.geometry_policy_digest != expected_policy_digest:
                raise SectionTimelineError(
                    "geometry_policy_digest does not match the stored solve policy"
                )
        else:
            object.__setattr__(
                self,
                "geometry_policy_digest",
                expected_policy_digest,
            )
        if not self.segment_schedules:
            raise SectionTimelineError("timeline requires at least one segment")
        if not all(
            isinstance(item, (PlaneMotionSchedule, ParallelPlaneMotionSchedule))
            for item in self.segment_schedules
        ):
            raise TypeError(
                "segment_schedules must contain supported plane-motion schedules"
            )
        motion_ids = tuple(
            schedule.motion.motion_id for schedule in self.segment_schedules
        )
        if len(set(motion_ids)) != len(motion_ids):
            raise SectionTimelineError("timeline motion ids must be unique")
        expected_samples: list[SectionAnimationSample] = []
        for index, schedule in enumerate(self.segment_schedules):
            if schedule.surface_id != self.surface_id:
                raise SectionTimelineError(
                    "segment schedule changed surface identity"
                )
            if schedule.motion.base_plane.plane_id != self.plane_id:
                raise SectionTimelineError(
                    "segment schedule changed plane identity"
                )
            if index:
                previous = self.segment_schedules[index - 1].motion
                current = schedule.motion
                if _motion_end_time(previous) != _motion_start_time(current):
                    raise SectionTimelineError(
                        "timeline motion times must join exactly without gaps or overlap"
                    )
                if _motion_end_plane(previous) != _motion_start_plane(current):
                    raise SectionTimelineError(
                        "timeline motion planes must join at one exact endpoint"
                    )
            for sample in schedule.samples:
                if expected_samples and sample.time == expected_samples[-1].time:
                    if sample != expected_samples[-1]:
                        raise SectionTimelineError(
                            "shared timeline time contains different section states"
                        )
                    continue
                expected_samples.append(sample)
        if tuple(expected_samples) != self.samples:
            raise SectionTimelineError(
                "timeline samples do not match the stitched segment schedules"
            )
        if not self.samples or any(
            right.time <= left.time
            for left, right in zip(self.samples, self.samples[1:])
        ):
            raise SectionTimelineError("timeline sample times must increase strictly")
        if self.animation.section_id != self.section_id:
            raise SectionTimelineError("animation section identity changed")
        if self.animation.surface_id != self.surface_id:
            raise SectionTimelineError("animation surface identity changed")
        if self.animation.plane_id != self.plane_id:
            raise SectionTimelineError("animation plane identity changed")
        if len(self.animation.frames) != len(self.samples):
            raise SectionTimelineError("animation must cover every timeline sample")
        for sample, frame in zip(self.samples, self.animation.frames):
            if frame.time != sample.time:
                raise SectionTimelineError(
                    "animation frame does not match its timeline sample"
                )
            if (
                frame.section.surface_id != sample.surface.surface_id
                or sample.plane.plane_id != self.plane_id
            ):
                raise SectionTimelineError(
                    "animation frame does not match its timeline sample"
                )
        event_keys = tuple(
            (item.time, item.segment_index, item.persistent, item.event_id)
            for item in self.critical_events
        )
        if event_keys != tuple(sorted(event_keys)):
            raise SectionTimelineError("critical events must use canonical order")
        if self.critical_events != _timeline_critical_events(
            self.section_id,
            self.segment_schedules,
        ):
            raise SectionTimelineError(
                "timeline critical events do not match segment evidence"
            )
        critical_ids = tuple(item.event_id for item in self.critical_events)
        if len(set(critical_ids)) != len(critical_ids):
            raise SectionTimelineError("critical event identities must be unique")
        critical_id_set = set(critical_ids)
        topology_ids = tuple(item.event_id for item in self.animation.topology_events)
        certified_ids = tuple(
            item.topology_event_id for item in self.topology_certifications
        )
        if topology_ids != certified_ids:
            raise SectionTimelineError(
                "every topology event must have one canonical certification"
            )
        if self.topology_certifications != _topology_certifications(
            self.animation,
            self.critical_events,
            self.samples[0].surface,
        ):
            raise SectionTimelineError(
                "topology certifications do not match analytic evidence"
            )
        if any(
            not set(item.critical_event_ids).issubset(critical_id_set)
            for item in self.topology_certifications
        ):
            raise SectionTimelineError(
                "topology certification references an unknown critical event"
            )
        expected_banks = _topology_frame_banks(self.animation)
        if self.topology_frame_banks != expected_banks:
            raise SectionTimelineError(
                "topology_frame_banks must change exactly at topology events"
            )
        cap_ids = tuple(
            sorted(
                set(_identity(item, "cap_chord_id") for item in self.cap_chord_ids)
            )
        )
        if cap_ids != self.cap_chord_ids:
            raise SectionTimelineError("cap_chord_ids must be unique and canonical")
        expected_cap_ids = section_cap_chord_curve_ids(
            self.section_id,
            self.samples[0].surface,
        )
        if cap_ids != expected_cap_ids:
            raise SectionTimelineError(
                "cap_chord_ids do not match the authored finite surface"
            )
        if len(self.cap_chord_states) != len(self.samples):
            raise SectionTimelineError(
                "cap-chord states must cover every timeline sample"
            )
        for index, (sample, state) in enumerate(
            zip(self.samples, self.cap_chord_states)
        ):
            if state.frame_index != index or state.time != sample.time:
                raise SectionTimelineError(
                    "cap-chord state does not match its timeline sample"
                )
            if not set(state.active_curve_ids).issubset(cap_ids):
                raise SectionTimelineError(
                    "cap-chord state references an unreserved identity"
                )
        cap_event_ids = tuple(item.event_id for item in self.cap_chord_events)
        if len(set(cap_event_ids)) != len(cap_event_ids):
            raise SectionTimelineError("cap-chord event identities must be unique")
        cap_event_right_frames = tuple(
            item.right_frame_index for item in self.cap_chord_events
        )
        if cap_event_right_frames != tuple(sorted(cap_event_right_frames)):
            raise SectionTimelineError(
                "cap-chord events must use canonical frame order"
            )
        cap_event_by_right_frame: dict[int, SectionTimelineCapChordEvent] = {}
        for event in self.cap_chord_events:
            if event.right_frame_index >= len(self.cap_chord_states):
                raise SectionTimelineError(
                    "cap-chord event references a missing timeline frame"
                )
            if event.right_frame_index in cap_event_by_right_frame:
                raise SectionTimelineError(
                    "cap-chord events must use canonical frame order"
                )
            left = self.cap_chord_states[event.left_frame_index]
            right = self.cap_chord_states[event.right_frame_index]
            if event.left_time != left.time or event.right_time != right.time:
                raise SectionTimelineError(
                    "cap-chord event times disagree with their states"
                )
            left_ids = set(left.active_curve_ids)
            right_ids = set(right.active_curve_ids)
            if (
                event.activated_curve_ids != tuple(sorted(right_ids - left_ids))
                or event.deactivated_curve_ids
                != tuple(sorted(left_ids - right_ids))
            ):
                raise SectionTimelineError(
                    "cap-chord event does not describe its state transition"
                )
            if not set(event.critical_event_ids).issubset(critical_id_set):
                raise SectionTimelineError(
                    "cap-chord event references an unknown critical event"
                )
            expected_evidence = _critical_ids_for_bracket(
                self.critical_events,
                left.time,
                right.time,
                accepted_kinds=_CAP_CHORD_EVENT_KINDS,
            )
            if event.critical_event_ids != expected_evidence:
                raise SectionTimelineError(
                    "cap-chord event does not use its complete trim evidence"
                )
            cap_event_by_right_frame[event.right_frame_index] = event
        changed_right_frames = {
            right.frame_index
            for left, right in zip(
                self.cap_chord_states,
                self.cap_chord_states[1:],
            )
            if left.active_curve_ids != right.active_curve_ids
        }
        if set(cap_event_by_right_frame) != changed_right_frames:
            raise SectionTimelineError(
                "every cap-chord state change must have one canonical event"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sectionId": self.section_id,
            "surfaceId": self.surface_id,
            "planeId": self.plane_id,
            "segments": [
                {
                    "segmentIndex": index,
                    "segmentKind": _schedule_kind(schedule),
                    "schedule": schedule.to_dict(),
                }
                for index, schedule in enumerate(self.segment_schedules)
            ],
            "samples": [
                {
                    "time": sample.time,
                    "plane": {
                        "point": list(sample.plane.point),
                        "normal": list(sample.plane.normal),
                        "uAxis": list(sample.plane.u_axis),
                    },
                }
                for sample in self.samples
            ],
            "criticalEvents": [item.to_dict() for item in self.critical_events],
            "animation": self.animation.to_dict(),
            "topologyCertifications": [
                item.to_dict() for item in self.topology_certifications
            ],
            "topologyFrameBanks": list(self.topology_frame_banks),
            "capChordIds": list(self.cap_chord_ids),
            "capChordStates": [item.to_dict() for item in self.cap_chord_states],
            "capChordEvents": [item.to_dict() for item in self.cap_chord_events],
            "geometryContext": _geometry_context_to_dict(self.geometry_context),
            "coefficientTolerance": self.coefficient_tolerance,
            "geometryPolicyDigest": self.geometry_policy_digest,
        }


def _compile_segment(
    surface: QuadricSurfaceSpec,
    motion: SectionTimelineMotion,
    *,
    authored_progresses: Sequence[float],
    include_interval_midpoints: bool,
    context: GeometryContext | ResolvedGeometryContext | None,
) -> SectionTimelineSegmentSchedule:
    if isinstance(motion, AxisAnglePlaneMotion):
        return compute_plane_motion_schedule(
            surface,
            motion,
            authored_progresses=authored_progresses,
            include_interval_midpoints=include_interval_midpoints,
            context=context,
        )
    if isinstance(motion, ParallelPlaneTranslation):
        return compute_parallel_plane_motion_schedule(
            surface,
            motion,
            authored_progresses=authored_progresses,
            include_interval_midpoints=include_interval_midpoints,
            context=context,
        )
    raise TypeError(
        "timeline motions must be AxisAnglePlaneMotion or ParallelPlaneTranslation"
    )


def _timeline_critical_events(
    section_id: str,
    schedules: Sequence[SectionTimelineSegmentSchedule],
) -> tuple[SectionTimelineCriticalEvent, ...]:
    identity = _identity(section_id, "section_id")
    result: list[SectionTimelineCriticalEvent] = []
    for segment_index, schedule in enumerate(schedules):
        segment_id = schedule.motion.motion_id
        segment_kind = _schedule_kind(schedule)
        for source in schedule.critical_events:
            result.append(
                SectionTimelineCriticalEvent(
                    event_id=(
                        f"{identity}:timeline:segment:{segment_index:04d}:"
                        f"{source.event_id}"
                    ),
                    source_event_id=source.event_id,
                    segment_id=segment_id,
                    segment_index=segment_index,
                    segment_kind=segment_kind,
                    progress=source.progress,
                    time=source.time,
                    kinds=tuple(item.value for item in source.kinds),
                    equations=source.equations,
                    persistent=source.persistent,
                )
            )
    result.sort(
        key=lambda item: (
            item.time,
            item.segment_index,
            item.persistent,
            item.event_id,
        )
    )
    identities = tuple(item.event_id for item in result)
    if len(set(identities)) != len(identities):
        raise SectionTimelineError("timeline critical event identities collided")
    return tuple(result)


def _critical_ids_for_bracket(
    events: Sequence[SectionTimelineCriticalEvent],
    left_time: float,
    right_time: float,
    *,
    accepted_kinds: frozenset[str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            event.event_id
            for event in events
            if not event.persistent
            and (
                accepted_kinds is None
                or not accepted_kinds.isdisjoint(event.kinds)
            )
            and left_time <= event.time <= right_time
            and event.time in {left_time, right_time}
        )
    )


def _topology_certifications(
    animation: SectionAnimationTrace,
    events: Sequence[SectionTimelineCriticalEvent],
    surface: QuadricSurfaceSpec,
) -> tuple[SectionTimelineTopologyCertification, ...]:
    surface_kinds = _SURFACE_CRITICAL_KINDS[type(surface)]
    supporting_kinds = _SUPPORTING_CONIC_EVENT_KINDS[type(surface)]
    finite_boundary_kinds = _FINITE_BOUNDARY_EVENT_KINDS[type(surface)]
    result: list[SectionTimelineTopologyCertification] = []
    for topology_event in animation.topology_events:
        bracket_ids = _critical_ids_for_bracket(
            events,
            topology_event.left_time,
            topology_event.right_time,
        )
        bracket_events = tuple(
            event for event in events if event.event_id in set(bracket_ids)
        )
        allowed_kinds = (
            supporting_kinds
            if not _SUPPORTING_CONIC_REASONS.isdisjoint(topology_event.reasons)
            else finite_boundary_kinds
        )
        critical_ids = tuple(
            event.event_id
            for event in bracket_events
            if set(event.kinds).issubset(surface_kinds)
            and not set(event.kinds).isdisjoint(allowed_kinds)
        )
        if not critical_ids:
            raise SectionTimelineError(
                f"topology event {topology_event.event_id!r} is not bracketed "
                "by analytic critical evidence compatible with its surface "
                "and topology reason"
            )
        result.append(
            SectionTimelineTopologyCertification(
                topology_event.event_id,
                critical_ids,
            )
        )
    return tuple(result)


def _topology_frame_banks(animation: SectionAnimationTrace) -> tuple[int, ...]:
    """Assign every topology epoch to one of two preallocated render banks."""

    if not animation.frames:
        raise SectionTimelineError("animation requires at least one frame")
    banks = [0]
    for left, right in zip(animation.frames, animation.frames[1:]):
        if left.signature.topologically_equivalent(right.signature):
            banks.append(banks[-1])
        else:
            banks.append(1 - banks[-1])
    return tuple(banks)


def _cap_chord_evidence(
    section_id: str,
    surface: QuadricSurfaceSpec,
    samples: Sequence[SectionAnimationSample],
    animation: SectionAnimationTrace,
    events: Sequence[SectionTimelineCriticalEvent],
    *,
    context: GeometryContext | ResolvedGeometryContext | None,
    coefficient_tolerance: float | None,
) -> tuple[
    tuple[str, ...],
    tuple[SectionTimelineCapChordState, ...],
    tuple[SectionTimelineCapChordEvent, ...],
]:
    reserved = section_cap_chord_curve_ids(section_id, surface)
    states: list[SectionTimelineCapChordState] = []
    for index, (sample, tracked) in enumerate(zip(samples, animation.frames)):
        boundary = compute_quadric_section_boundary(
            section_id,
            surface,
            sample.plane,
            context=context,
            coefficient_tolerance=coefficient_tolerance,
        )
        if boundary.trace != tracked.section:
            raise SectionTimelineError(
                "cap-chord evidence and branch tracking solved different sections"
            )
        active = tuple(sorted(item.curve_id for item in boundary.cap_chords))
        if not set(active).issubset(reserved):
            raise SectionTimelineError(
                "active cap chord was not present in the reserved identity set"
            )
        states.append(SectionTimelineCapChordState(index, sample.time, active))

    cap_events: list[SectionTimelineCapChordEvent] = []
    for left, right in zip(states, states[1:]):
        left_ids = set(left.active_curve_ids)
        right_ids = set(right.active_curve_ids)
        if left_ids == right_ids:
            continue
        critical_ids = _critical_ids_for_bracket(
            events,
            left.time,
            right.time,
            accepted_kinds=_CAP_CHORD_EVENT_KINDS,
        )
        if not critical_ids:
            raise SectionTimelineError(
                "cap-chord activation changed without analytic critical evidence"
            )
        cap_events.append(
            SectionTimelineCapChordEvent(
                event_id=f"{section_id}:cap-chord-event:{len(cap_events):04d}",
                left_frame_index=left.frame_index,
                right_frame_index=right.frame_index,
                left_time=left.time,
                right_time=right.time,
                activated_curve_ids=tuple(sorted(right_ids - left_ids)),
                deactivated_curve_ids=tuple(sorted(left_ids - right_ids)),
                critical_event_ids=critical_ids,
            )
        )
    return reserved, tuple(states), tuple(cap_events)


def compile_section_timeline(
    section_id: str,
    surface: QuadricSurfaceSpec,
    motions: Sequence[SectionTimelineMotion],
    *,
    authored_progresses: Mapping[str, Sequence[float]] | None = None,
    include_interval_midpoints: bool = True,
    context: GeometryContext | ResolvedGeometryContext | None = None,
    coefficient_tolerance: float | None = None,
) -> SectionTimeline:
    """Compile explicit plane motions into one certified topology timeline."""

    identity = _identity(section_id, "section_id")
    selected_surface = _surface(surface)
    if authored_progresses is None:
        authored: dict[str, tuple[float, ...]] = {}
    elif not isinstance(authored_progresses, Mapping):
        raise TypeError("authored_progresses must be a mapping")
    else:
        authored = {}
        for raw_motion_id, raw_progresses in authored_progresses.items():
            motion_id = _identity(raw_motion_id, "authored motion id")
            if motion_id in authored:
                raise SectionTimelineError(
                    "authored_progresses contains duplicate normalized motion ids"
                )
            try:
                authored[motion_id] = tuple(raw_progresses)
            except TypeError as exc:
                raise TypeError(
                    "authored progress values must be a finite sequence"
                ) from exc
    if not isinstance(include_interval_midpoints, bool):
        raise TypeError("include_interval_midpoints must be a bool")
    if not include_interval_midpoints:
        raise SectionTimelineError(
            "a complete SectionTimeline requires every analytic interval midpoint"
        )
    if context is not None and not isinstance(
        context,
        (GeometryContext, ResolvedGeometryContext),
    ):
        raise TypeError(
            "context must be a GeometryContext or ResolvedGeometryContext"
        )
    if coefficient_tolerance is not None:
        if _finite(coefficient_tolerance, "coefficient_tolerance") <= 0.0:
            raise SectionTimelineError(
                "coefficient_tolerance must be finite and positive"
            )

    sequence = tuple(motions)
    if not sequence:
        raise SectionTimelineError("timeline requires at least one motion")
    if not all(
        isinstance(item, (AxisAnglePlaneMotion, ParallelPlaneTranslation))
        for item in sequence
    ):
        raise TypeError(
            "motions must contain AxisAnglePlaneMotion or ParallelPlaneTranslation"
        )
    motion_ids = tuple(_motion_id(item) for item in sequence)
    if len(set(motion_ids)) != len(motion_ids):
        raise SectionTimelineError("timeline motion ids must be unique")
    unknown_authored = tuple(sorted(set(authored) - set(motion_ids)))
    if unknown_authored:
        raise SectionTimelineError(
            "authored_progresses contains unknown motion ids: "
            + ", ".join(unknown_authored)
        )

    plane_id = _motion_start_plane(sequence[0]).plane_id
    for index, motion in enumerate(sequence):
        start_plane = _motion_start_plane(motion)
        end_plane = _motion_end_plane(motion)
        if start_plane.plane_id != plane_id or end_plane.plane_id != plane_id:
            raise SectionTimelineError("timeline motion changed plane identity")
        if index:
            previous = sequence[index - 1]
            if _motion_end_time(previous) != _motion_start_time(motion):
                raise SectionTimelineError(
                    "timeline motion times must join exactly without gaps or overlap"
                )
            if _motion_end_plane(previous) != start_plane:
                raise SectionTimelineError(
                    "timeline motion planes must join at one exact endpoint"
                )

    schedules = tuple(
        _compile_segment(
            selected_surface,
            motion,
            authored_progresses=authored.get(motion.motion_id, ()),
            include_interval_midpoints=include_interval_midpoints,
            context=context,
        )
        for motion in sequence
    )

    global_samples: list[SectionAnimationSample] = []
    for schedule in schedules:
        for sample in schedule.samples:
            if global_samples and sample.time == global_samples[-1].time:
                if sample != global_samples[-1]:
                    raise SectionTimelineError(
                        "shared timeline time contains different section states"
                    )
                continue
            if global_samples and sample.time <= global_samples[-1].time:
                raise SectionTimelineError(
                    "timeline sample times must increase strictly"
                )
            global_samples.append(sample)

    samples = tuple(global_samples)
    animation = track_quadric_section_animation(
        identity,
        samples,
        context=context,
        coefficient_tolerance=coefficient_tolerance,
    )
    critical_events = _timeline_critical_events(identity, schedules)
    topology_certifications = _topology_certifications(
        animation,
        critical_events,
        selected_surface,
    )
    topology_frame_banks = _topology_frame_banks(animation)
    cap_ids, cap_states, cap_events = _cap_chord_evidence(
        identity,
        selected_surface,
        samples,
        animation,
        critical_events,
        context=context,
        coefficient_tolerance=coefficient_tolerance,
    )
    return SectionTimeline(
        section_id=identity,
        surface_id=selected_surface.surface_id,
        plane_id=plane_id,
        segment_schedules=schedules,
        samples=samples,
        critical_events=critical_events,
        animation=animation,
        topology_certifications=topology_certifications,
        topology_frame_banks=topology_frame_banks,
        cap_chord_ids=cap_ids,
        cap_chord_states=cap_states,
        cap_chord_events=cap_events,
        geometry_context=context,
        coefficient_tolerance=coefficient_tolerance,
    )


def canonical_section_timeline_json(timeline: SectionTimeline) -> str:
    if not isinstance(timeline, SectionTimeline):
        raise TypeError("timeline must be a SectionTimeline")
    return json.dumps(
        timeline.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "SECTION_TIMELINE_SCHEMA",
    "SectionTimeline",
    "SectionTimelineCapChordEvent",
    "SectionTimelineCapChordState",
    "SectionTimelineCriticalEvent",
    "SectionTimelineError",
    "SectionTimelineMotion",
    "SectionTimelineSegmentSchedule",
    "SectionTimelineTopologyCertification",
    "canonical_section_timeline_json",
    "compile_section_timeline",
]
