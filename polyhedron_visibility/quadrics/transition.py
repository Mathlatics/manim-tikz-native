"""Renderer-neutral plans for animated section-topology handoff.

The exact plane-motion schedule already records every analytic critical
progress.  This module turns those critical samples into a deterministic
two-bank render plan:

* ordinary motion uses one live bank;
* immediately before a topology event the live section fades into the exact
  critical section;
* immediately after the event the critical section fades into the new live
  topology; and
* a cut mode remains available for proofs that require an instantaneous
  change.

The plan contains no Manim objects.  It is safe to serialize, cache, test, and
sample in either time direction.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite

from .animation import TrackedSectionFrame
from .plane_motion import ScheduledSectionAnimation


SECTION_TRANSITION_PLAN_SCHEMA = "manim-quadric-section-transition-plan/v1"
_PROGRESS_TOLERANCE = 1.0e-12
_OPACITY_TOLERANCE = 1.0e-15


class SectionTransitionError(ValueError):
    """An automatic topology handoff cannot be planned without guessing."""


class SectionTransitionMode(str, Enum):
    """Supported deterministic visual policies at a topology event."""

    CROSSFADE = "crossfade"
    CUT = "cut"


class SectionTransitionRole(str, Enum):
    """Meaning of one active render-bank layer."""

    LIVE = "live"
    CRITICAL = "critical"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SectionTransitionError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SectionTransitionError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SectionTransitionError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise SectionTransitionError(f"{label} must be finite")
    return result


def _progress(value: object, label: str = "progress") -> float:
    result = _finite(value, label)
    if result < 0.0 or result > 1.0:
        raise SectionTransitionError(f"{label} must lie in [0, 1]")
    return result


def _frame_index(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SectionTransitionError(f"{label} must be a non-negative integer")
    return value


def _smoothstep(value: float) -> float:
    clamped = min(1.0, max(0.0, float(value)))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _equivalent(left: TrackedSectionFrame, right: TrackedSectionFrame) -> bool:
    return left.signature.topologically_equivalent(right.signature)


@dataclass(frozen=True, slots=True)
class TopologyTransitionKnot:
    """One exact critical progress and its bounded one-sided fade windows."""

    event_id: str
    progress: float
    time: float
    critical_frame_index: int
    before_frame_index: int | None
    after_frame_index: int | None
    left_start: float
    right_end: float
    left_changes: bool
    right_changes: bool
    left_crossfade: bool
    right_crossfade: bool
    critical_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identity(self.event_id, "event_id"))
        progress = _progress(self.progress, "knot progress")
        left_start = _progress(self.left_start, "left_start")
        right_end = _progress(self.right_end, "right_end")
        if left_start > progress or right_end < progress:
            raise SectionTransitionError("transition window does not contain its knot")
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "time", _finite(self.time, "knot time"))
        object.__setattr__(self, "left_start", left_start)
        object.__setattr__(self, "right_end", right_end)
        _frame_index(self.critical_frame_index, "critical_frame_index")
        if self.before_frame_index is not None:
            _frame_index(self.before_frame_index, "before_frame_index")
        if self.after_frame_index is not None:
            _frame_index(self.after_frame_index, "after_frame_index")
        if not all(
            isinstance(item, bool)
            for item in (
                self.left_changes,
                self.right_changes,
                self.left_crossfade,
                self.right_crossfade,
            )
        ):
            raise TypeError("transition side flags must be boolean")
        if not self.left_changes and not self.right_changes:
            raise SectionTransitionError("a transition knot must change topology")
        if self.left_crossfade:
            if not self.left_changes:
                raise SectionTransitionError(
                    "a left crossfade requires a left topology change"
                )
            if self.before_frame_index is None or progress - left_start <= 0.0:
                raise SectionTransitionError(
                    "a left topology change requires a non-empty left window"
                )
        elif left_start != progress:
            raise SectionTransitionError(
                "an unchanged left side must not allocate a fade window"
            )
        if self.right_crossfade:
            if not self.right_changes:
                raise SectionTransitionError(
                    "a right crossfade requires a right topology change"
                )
            if self.after_frame_index is None or right_end - progress <= 0.0:
                raise SectionTransitionError(
                    "a right topology change requires a non-empty right window"
                )
        elif right_end != progress:
            raise SectionTransitionError(
                "an unchanged right side must not allocate a fade window"
            )
        kinds = tuple(sorted({_identity(item, "critical kind") for item in self.critical_kinds}))
        if not kinds or kinds != self.critical_kinds:
            raise SectionTransitionError("critical kinds must be unique and canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "eventId": self.event_id,
            "progress": self.progress,
            "time": self.time,
            "criticalFrameIndex": self.critical_frame_index,
            "beforeFrameIndex": self.before_frame_index,
            "afterFrameIndex": self.after_frame_index,
            "leftStart": self.left_start,
            "rightEnd": self.right_end,
            "leftChanges": self.left_changes,
            "rightChanges": self.right_changes,
            "leftCrossfade": self.left_crossfade,
            "rightCrossfade": self.right_crossfade,
            "criticalKinds": list(self.critical_kinds),
        }


@dataclass(frozen=True, slots=True)
class SectionTransitionLayer:
    """One live or exact-critical section assigned to a preallocated bank."""

    bank_index: int
    geometry_progress: float
    opacity: float
    reference_frame_index: int
    role: SectionTransitionRole

    def __post_init__(self) -> None:
        if self.bank_index not in {0, 1}:
            raise SectionTransitionError("transition bank_index must be 0 or 1")
        object.__setattr__(
            self, "geometry_progress", _progress(self.geometry_progress)
        )
        opacity = _finite(self.opacity, "layer opacity")
        if opacity < 0.0 or opacity > 1.0:
            raise SectionTransitionError("layer opacity must lie in [0, 1]")
        object.__setattr__(self, "opacity", opacity)
        _frame_index(self.reference_frame_index, "reference_frame_index")
        if not isinstance(self.role, SectionTransitionRole):
            raise TypeError("role must be a SectionTransitionRole")

    def to_dict(self) -> dict[str, object]:
        return {
            "bankIndex": self.bank_index,
            "geometryProgress": self.geometry_progress,
            "opacity": self.opacity,
            "referenceFrameIndex": self.reference_frame_index,
            "role": self.role.value,
        }


@dataclass(frozen=True, slots=True)
class SectionTransitionFrame:
    """Deterministic render-bank state at one normalized motion progress."""

    progress: float
    layers: tuple[SectionTransitionLayer, ...]
    active_event_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "progress", _progress(self.progress))
        if not self.layers or len(self.layers) > 2:
            raise SectionTransitionError("a transition frame requires one or two layers")
        if not all(isinstance(item, SectionTransitionLayer) for item in self.layers):
            raise TypeError("layers must contain SectionTransitionLayer objects")
        banks = tuple(item.bank_index for item in self.layers)
        if len(set(banks)) != len(banks):
            raise SectionTransitionError("active transition layers must use unique banks")
        total = sum(item.opacity for item in self.layers)
        if abs(total - 1.0) > 1.0e-12:
            raise SectionTransitionError("transition layer opacities must sum to one")
        if self.active_event_id is not None:
            object.__setattr__(
                self,
                "active_event_id",
                _identity(self.active_event_id, "active_event_id"),
            )

    @property
    def transitioning(self) -> bool:
        return len(self.layers) == 2

    def to_dict(self) -> dict[str, object]:
        return {
            "progress": self.progress,
            "activeEventId": self.active_event_id,
            "transitioning": self.transitioning,
            "layers": [item.to_dict() for item in self.layers],
        }


@dataclass(frozen=True, slots=True)
class SectionTransitionPlan:
    """A complete two-bank handoff plan for one scheduled plane motion."""

    scheduled: ScheduledSectionAnimation
    mode: SectionTransitionMode
    transition_fraction: float
    frame_banks: tuple[int, ...]
    knots: tuple[TopologyTransitionKnot, ...]
    schema: str = SECTION_TRANSITION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SECTION_TRANSITION_PLAN_SCHEMA:
            raise SectionTransitionError("invalid section-transition plan schema")
        if not isinstance(self.scheduled, ScheduledSectionAnimation):
            raise TypeError("scheduled must be a ScheduledSectionAnimation")
        if not isinstance(self.mode, SectionTransitionMode):
            raise TypeError("mode must be a SectionTransitionMode")
        fraction = _finite(self.transition_fraction, "transition_fraction")
        if fraction < 0.0 or fraction > 0.5:
            raise SectionTransitionError("transition_fraction must lie in [0, 0.5]")
        if self.mode is SectionTransitionMode.CROSSFADE and fraction <= 0.0:
            raise SectionTransitionError(
                "crossfade mode requires a positive transition_fraction"
            )
        object.__setattr__(self, "transition_fraction", fraction)
        frames = self.scheduled.animation.frames
        if len(self.frame_banks) != len(frames) or any(
            item not in {0, 1} for item in self.frame_banks
        ):
            raise SectionTransitionError("frame_banks must assign every frame")
        for index, (left, right) in enumerate(zip(frames, frames[1:])):
            same = _equivalent(left, right)
            if same != (self.frame_banks[index] == self.frame_banks[index + 1]):
                raise SectionTransitionError(
                    "frame bank changes must exactly match topology changes"
                )
        progresses = tuple(item.progress for item in self.knots)
        if progresses != tuple(sorted(set(progresses))):
            raise SectionTransitionError("transition knots must use canonical order")
        for left, right in zip(self.knots, self.knots[1:]):
            if left.right_end > right.left_start + _PROGRESS_TOLERANCE:
                raise SectionTransitionError("transition windows must not overlap")

    @property
    def section_id(self) -> str:
        return self.scheduled.animation.section_id

    def sample(self, progress: float) -> SectionTransitionFrame:
        return sample_section_transition(self, progress)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sectionId": self.section_id,
            "surfaceId": self.scheduled.animation.surface_id,
            "planeId": self.scheduled.animation.plane_id,
            "mode": self.mode.value,
            "transitionFraction": self.transition_fraction,
            "frameBanks": list(self.frame_banks),
            "knots": [item.to_dict() for item in self.knots],
        }


def _find_exact_progress(progresses: tuple[float, ...], value: float) -> int | None:
    for index, progress in enumerate(progresses):
        if abs(progress - value) <= _PROGRESS_TOLERANCE:
            return index
    return None


def _topology_frame_banks(frames: tuple[TrackedSectionFrame, ...]) -> tuple[int, ...]:
    banks = [0]
    for left, right in zip(frames, frames[1:]):
        banks.append(banks[-1] if _equivalent(left, right) else 1 - banks[-1])
    return tuple(banks)


def build_section_transition_plan(
    scheduled: ScheduledSectionAnimation,
    *,
    transition_fraction: float = 0.04,
    mode: SectionTransitionMode | str = SectionTransitionMode.CROSSFADE,
) -> SectionTransitionPlan:
    """Build a deterministic handoff plan from an analytic motion schedule."""

    if not isinstance(scheduled, ScheduledSectionAnimation):
        raise TypeError("scheduled must be a ScheduledSectionAnimation")
    try:
        selected_mode = SectionTransitionMode(mode)
    except (TypeError, ValueError) as exc:
        raise SectionTransitionError("mode must be 'crossfade' or 'cut'") from exc
    fraction = _finite(transition_fraction, "transition_fraction")
    if fraction < 0.0 or fraction > 0.5:
        raise SectionTransitionError("transition_fraction must lie in [0, 0.5]")
    if selected_mode is SectionTransitionMode.CROSSFADE and fraction <= 0.0:
        raise SectionTransitionError(
            "crossfade mode requires a positive transition_fraction"
        )

    schedule = scheduled.schedule
    animation = scheduled.animation
    progresses = schedule.progresses
    frames = animation.frames
    critical_times = tuple(
        item.time for item in schedule.critical_events if not item.persistent
    )
    for event in animation.topology_events:
        if not any(
            abs(event.left_time - critical) <= _PROGRESS_TOLERANCE
            or abs(event.right_time - critical) <= _PROGRESS_TOLERANCE
            for critical in critical_times
        ):
            raise SectionTransitionError(
                "topology change is not bracketed by an analytic critical event"
            )

    candidates: list[
        tuple[object, int, int | None, int | None, bool, bool, bool, bool]
    ] = []
    for event in schedule.critical_events:
        if event.persistent:
            continue
        critical_index = _find_exact_progress(progresses, event.progress)
        if critical_index is None:
            raise SectionTransitionError(
                f"critical event {event.event_id!r} has no exact schedule frame"
            )
        before_index = critical_index - 1 if critical_index > 0 else None
        after_index = (
            critical_index + 1 if critical_index + 1 < len(frames) else None
        )
        critical_frame = frames[critical_index]
        left_changes = bool(
            before_index is not None
            and not _equivalent(frames[before_index], critical_frame)
        )
        right_changes = bool(
            after_index is not None
            and not _equivalent(critical_frame, frames[after_index])
        )
        left_crossfade = bool(
            selected_mode is SectionTransitionMode.CROSSFADE
            and left_changes
            and before_index is not None
            and frames[before_index].signature.conic_family
            is not critical_frame.signature.conic_family
        )
        right_crossfade = bool(
            selected_mode is SectionTransitionMode.CROSSFADE
            and right_changes
            and after_index is not None
            and critical_frame.signature.conic_family
            is not frames[after_index].signature.conic_family
        )
        if left_changes or right_changes:
            candidates.append(
                (
                    event,
                    critical_index,
                    before_index,
                    after_index,
                    left_changes,
                    right_changes,
                    left_crossfade,
                    right_crossfade,
                )
            )

    event_progresses = tuple(float(item[0].progress) for item in candidates)
    knots: list[TopologyTransitionKnot] = []
    for index, candidate in enumerate(candidates):
        (
            event,
            critical_index,
            before_index,
            after_index,
            left_changes,
            right_changes,
            left_crossfade,
            right_crossfade,
        ) = candidate
        previous_progress = event_progresses[index - 1] if index else 0.0
        next_progress = (
            event_progresses[index + 1] if index + 1 < len(candidates) else 1.0
        )
        left_width = (
            min(fraction, 0.45 * (event.progress - previous_progress))
            if left_crossfade
            else 0.0
        )
        right_width = (
            min(fraction, 0.45 * (next_progress - event.progress))
            if right_crossfade
            else 0.0
        )
        if left_crossfade and left_width <= 0.0:
            raise SectionTransitionError(
                f"critical event {event.event_id!r} has no left fade interval"
            )
        if right_crossfade and right_width <= 0.0:
            raise SectionTransitionError(
                f"critical event {event.event_id!r} has no right fade interval"
            )
        knots.append(
            TopologyTransitionKnot(
                event_id=event.event_id,
                progress=event.progress,
                time=event.time,
                critical_frame_index=critical_index,
                before_frame_index=before_index,
                after_frame_index=after_index,
                left_start=event.progress - left_width,
                right_end=event.progress + right_width,
                left_changes=left_changes,
                right_changes=right_changes,
                left_crossfade=left_crossfade,
                right_crossfade=right_crossfade,
                critical_kinds=tuple(item.value for item in event.kinds),
            )
        )

    return SectionTransitionPlan(
        scheduled=scheduled,
        mode=selected_mode,
        transition_fraction=fraction,
        frame_banks=_topology_frame_banks(frames),
        knots=tuple(knots),
    )


def _knot_at_progress(
    plan: SectionTransitionPlan, progress: float
) -> TopologyTransitionKnot | None:
    for knot in plan.knots:
        if abs(knot.progress - progress) <= _PROGRESS_TOLERANCE:
            return knot
    return None


def _reference_frame_index(plan: SectionTransitionPlan, progress: float) -> int:
    progresses = plan.scheduled.schedule.progresses
    exact = _find_exact_progress(progresses, progress)
    if exact is not None:
        return exact
    right_index = bisect_right(progresses, progress)
    if right_index <= 0 or right_index >= len(progresses):
        raise SectionTransitionError("progress lies outside the scheduled interval")
    left_index = right_index - 1
    frames = plan.scheduled.animation.frames
    if _equivalent(frames[left_index], frames[right_index]):
        left_distance = progress - progresses[left_index]
        right_distance = progresses[right_index] - progress
        return left_index if left_distance <= right_distance else right_index
    right_knot = _knot_at_progress(plan, progresses[right_index])
    if right_knot is not None and right_knot.left_changes:
        return left_index
    left_knot = _knot_at_progress(plan, progresses[left_index])
    if left_knot is not None and left_knot.right_changes:
        return right_index
    raise SectionTransitionError(
        "scheduled topology changed away from an automatic transition knot"
    )


def _one_layer(
    plan: SectionTransitionPlan,
    progress: float,
    reference_index: int,
    *,
    role: SectionTransitionRole = SectionTransitionRole.LIVE,
    active_event_id: str | None = None,
    geometry_progress: float | None = None,
) -> SectionTransitionFrame:
    return SectionTransitionFrame(
        progress,
        (
            SectionTransitionLayer(
                bank_index=plan.frame_banks[reference_index],
                geometry_progress=(
                    progress if geometry_progress is None else geometry_progress
                ),
                opacity=1.0,
                reference_frame_index=reference_index,
                role=role,
            ),
        ),
        active_event_id,
    )


def _blend_layers(
    plan: SectionTransitionPlan,
    progress: float,
    knot: TopologyTransitionKnot,
    *,
    live_reference_index: int,
    live_opacity: float,
    critical_opacity: float,
) -> SectionTransitionFrame:
    layers = (
        SectionTransitionLayer(
            bank_index=plan.frame_banks[live_reference_index],
            geometry_progress=progress,
            opacity=live_opacity,
            reference_frame_index=live_reference_index,
            role=SectionTransitionRole.LIVE,
        ),
        SectionTransitionLayer(
            bank_index=plan.frame_banks[knot.critical_frame_index],
            geometry_progress=knot.progress,
            opacity=critical_opacity,
            reference_frame_index=knot.critical_frame_index,
            role=SectionTransitionRole.CRITICAL,
        ),
    )
    active = tuple(item for item in layers if item.opacity > _OPACITY_TOLERANCE)
    if len(active) == 1:
        only = active[0]
        active = (
            SectionTransitionLayer(
                bank_index=only.bank_index,
                geometry_progress=only.geometry_progress,
                opacity=1.0,
                reference_frame_index=only.reference_frame_index,
                role=only.role,
            ),
        )
    return SectionTransitionFrame(progress, active, knot.event_id)


def sample_section_transition(
    plan: SectionTransitionPlan,
    progress: float,
) -> SectionTransitionFrame:
    """Return the exact one- or two-bank state at ``progress``."""

    if not isinstance(plan, SectionTransitionPlan):
        raise TypeError("plan must be a SectionTransitionPlan")
    value = _progress(progress)
    exact_knot = _knot_at_progress(plan, value)
    if exact_knot is not None:
        pure_trim_tangency = all(
            item in {"cone_trim_tangency", "cylinder_trim_tangency"}
            for item in exact_knot.critical_kinds
        )
        if pure_trim_tangency:
            progresses = plan.scheduled.schedule.progresses
            if exact_knot.before_frame_index is not None:
                reference = exact_knot.before_frame_index
                gap = exact_knot.progress - progresses[reference]
                geometry_progress = exact_knot.progress - min(1.0e-6, gap * 1.0e-3)
            elif exact_knot.after_frame_index is not None:
                reference = exact_knot.after_frame_index
                gap = progresses[reference] - exact_knot.progress
                geometry_progress = exact_knot.progress + min(1.0e-6, gap * 1.0e-3)
            else:
                raise SectionTransitionError(
                    "an instantaneous transition has no neighboring frame"
                )
            return _one_layer(
                plan,
                value,
                reference,
                role=SectionTransitionRole.LIVE,
                active_event_id=exact_knot.event_id,
                geometry_progress=geometry_progress,
            )
        return _one_layer(
            plan,
            exact_knot.progress,
            exact_knot.critical_frame_index,
            role=SectionTransitionRole.CRITICAL,
            active_event_id=exact_knot.event_id,
        )

    if plan.mode is SectionTransitionMode.CROSSFADE:
        for knot in plan.knots:
            if knot.left_crossfade and knot.left_start < value < knot.progress:
                if knot.before_frame_index is None:
                    raise SectionTransitionError("left transition has no reference frame")
                ratio = (value - knot.left_start) / (
                    knot.progress - knot.left_start
                )
                critical_opacity = _smoothstep(ratio)
                return _blend_layers(
                    plan,
                    value,
                    knot,
                    live_reference_index=knot.before_frame_index,
                    live_opacity=1.0 - critical_opacity,
                    critical_opacity=critical_opacity,
                )
            if knot.right_crossfade and knot.progress < value < knot.right_end:
                if knot.after_frame_index is None:
                    raise SectionTransitionError("right transition has no reference frame")
                ratio = (value - knot.progress) / (
                    knot.right_end - knot.progress
                )
                live_opacity = _smoothstep(ratio)
                return _blend_layers(
                    plan,
                    value,
                    knot,
                    live_reference_index=knot.after_frame_index,
                    live_opacity=live_opacity,
                    critical_opacity=1.0 - live_opacity,
                )

    reference = _reference_frame_index(plan, value)
    return _one_layer(plan, value, reference)


def canonical_section_transition_plan_json(plan: SectionTransitionPlan) -> str:
    if not isinstance(plan, SectionTransitionPlan):
        raise TypeError("plan must be a SectionTransitionPlan")
    return json.dumps(
        plan.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "SECTION_TRANSITION_PLAN_SCHEMA",
    "SectionTransitionError",
    "SectionTransitionFrame",
    "SectionTransitionLayer",
    "SectionTransitionMode",
    "SectionTransitionPlan",
    "SectionTransitionRole",
    "TopologyTransitionKnot",
    "build_section_transition_plan",
    "canonical_section_transition_plan_json",
    "sample_section_transition",
]
