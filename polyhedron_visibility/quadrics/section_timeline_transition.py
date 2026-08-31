"""Global-time two-bank handoffs for a certified :mod:`SectionTimeline`.

The older transition contract plans one motion segment in normalized progress.
This module groups topology events across the complete multi-segment timeline by
their analytic critical time.  Supporting-conic family changes may crossfade;
finite trim changes remain exact cuts.  No renderer objects are allocated here.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite, nextafter

from .section_timeline import SectionTimeline


SECTION_TIMELINE_TRANSITION_SCHEMA = "quadric-section-timeline-transition/v1"
_OPACITY_TOLERANCE = 1.0e-15
_TRIM_CRITICAL_KINDS = frozenset(
    {"cone_trim_tangency", "cylinder_trim_tangency"}
)


class SectionTimelineTransitionError(ValueError):
    """A global topology handoff cannot be planned without guessing."""


class SectionTimelineTransitionMode(str, Enum):
    CROSSFADE = "crossfade"
    CUT = "cut"


class SectionTimelineLayerRole(str, Enum):
    LIVE_BEFORE = "live-before"
    EXACT_CRITICAL = "exact-critical"
    LIVE_AFTER = "live-after"
    LIVE = "live"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SectionTimelineTransitionError(f"{label} must be non-empty")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SectionTimelineTransitionError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SectionTimelineTransitionError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise SectionTimelineTransitionError(f"{label} must be finite")
    return result


def _fraction(value: object) -> float:
    result = _finite(value, "transition_fraction")
    if result < 0.0 or result > 0.5:
        raise SectionTimelineTransitionError(
            "transition_fraction must lie in [0, 0.5]"
        )
    return result


def _frame_index(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SectionTimelineTransitionError(
            f"{label} must be a non-negative integer"
        )
    return value


def _bank(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in {0, 1}
    ):
        raise SectionTimelineTransitionError(f"{label} must be integer 0 or 1")
    return value


def _canonical_ids(
    values: object,
    label: str,
    *,
    require_non_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise SectionTimelineTransitionError(f"{label} must be a tuple")
    normalized = tuple(sorted({_identity(item, label) for item in values}))
    if normalized != values or (require_non_empty and not normalized):
        qualifier = "non-empty, " if require_non_empty else ""
        raise SectionTimelineTransitionError(
            f"{label} must be {qualifier}unique, and canonical"
        )
    return normalized


def _smoothstep(value: float) -> float:
    clamped = min(1.0, max(0.0, float(value)))
    return clamped * clamped * (3.0 - 2.0 * clamped)


@dataclass(frozen=True, slots=True)
class SectionTimelineTransitionKnot:
    knot_id: str
    critical_time: float
    critical_frame_index: int
    before_frame_index: int | None
    after_frame_index: int | None
    before_bank: int | None
    critical_bank: int
    after_bank: int | None
    left_start: float
    right_end: float
    left_crossfade: bool
    right_crossfade: bool
    topology_event_ids: tuple[str, ...]
    critical_event_ids: tuple[str, ...]
    left_topology_event_ids: tuple[str, ...]
    right_topology_event_ids: tuple[str, ...]
    left_critical_event_ids: tuple[str, ...]
    right_critical_event_ids: tuple[str, ...]
    critical_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "knot_id", _identity(self.knot_id, "knot_id"))
        critical = _finite(self.critical_time, "critical_time")
        left = _finite(self.left_start, "left_start")
        right = _finite(self.right_end, "right_end")
        if left > critical or right < critical:
            raise SectionTimelineTransitionError(
                "transition window must contain its critical time"
            )
        _frame_index(self.critical_frame_index, "critical_frame_index")
        for label, value in (
            ("before_frame_index", self.before_frame_index),
            ("after_frame_index", self.after_frame_index),
        ):
            if value is not None:
                _frame_index(value, label)
        _bank(self.critical_bank, "critical_bank")
        for label, value in (
            ("before_bank", self.before_bank),
            ("after_bank", self.after_bank),
        ):
            if value is not None:
                _bank(value, label)
        if (self.before_frame_index is None) != (self.before_bank is None):
            raise SectionTimelineTransitionError(
                "before frame and bank must either both exist or both be absent"
            )
        if (self.after_frame_index is None) != (self.after_bank is None):
            raise SectionTimelineTransitionError(
                "after frame and bank must either both exist or both be absent"
            )
        if not isinstance(self.left_crossfade, bool) or not isinstance(
            self.right_crossfade, bool
        ):
            raise TypeError("crossfade flags must be bool")
        if self.left_crossfade and (
            self.before_frame_index is None or left == critical
        ):
            raise SectionTimelineTransitionError(
                "left crossfade requires a non-empty before window"
            )
        if self.left_crossfade and self.before_bank == self.critical_bank:
            raise SectionTimelineTransitionError(
                "left crossfade requires two distinct banks"
            )
        if not self.left_crossfade and left != critical:
            raise SectionTimelineTransitionError(
                "a non-crossfade left side must not allocate a window"
            )
        if self.right_crossfade and (
            self.after_frame_index is None or right == critical
        ):
            raise SectionTimelineTransitionError(
                "right crossfade requires a non-empty after window"
            )
        if self.right_crossfade and self.after_bank == self.critical_bank:
            raise SectionTimelineTransitionError(
                "right crossfade requires two distinct banks"
            )
        if not self.right_crossfade and right != critical:
            raise SectionTimelineTransitionError(
                "a non-crossfade right side must not allocate a window"
            )
        object.__setattr__(self, "critical_time", critical)
        object.__setattr__(self, "left_start", left)
        object.__setattr__(self, "right_end", right)

        topology_ids = _canonical_ids(
            self.topology_event_ids,
            "topology_event_ids",
            require_non_empty=True,
        )
        critical_ids = _canonical_ids(
            self.critical_event_ids,
            "critical_event_ids",
            require_non_empty=True,
        )
        left_topology = _canonical_ids(
            self.left_topology_event_ids,
            "left_topology_event_ids",
            require_non_empty=False,
        )
        right_topology = _canonical_ids(
            self.right_topology_event_ids,
            "right_topology_event_ids",
            require_non_empty=False,
        )
        left_critical = _canonical_ids(
            self.left_critical_event_ids,
            "left_critical_event_ids",
            require_non_empty=False,
        )
        right_critical = _canonical_ids(
            self.right_critical_event_ids,
            "right_critical_event_ids",
            require_non_empty=False,
        )
        critical_kinds = _canonical_ids(
            self.critical_kinds,
            "critical_kinds",
            require_non_empty=True,
        )
        if topology_ids != tuple(
            sorted(set(left_topology) | set(right_topology))
        ):
            raise SectionTimelineTransitionError(
                "left/right topology evidence must exactly cover topology_event_ids"
            )
        if critical_ids != tuple(
            sorted(set(left_critical) | set(right_critical))
        ):
            raise SectionTimelineTransitionError(
                "left/right critical evidence must exactly cover critical_event_ids"
            )
        for side, frame_index, topology, evidence in (
            ("left", self.before_frame_index, left_topology, left_critical),
            ("right", self.after_frame_index, right_topology, right_critical),
        ):
            if bool(topology) != (frame_index is not None):
                raise SectionTimelineTransitionError(
                    f"{side} topology evidence disagrees with its neighboring frame"
                )
            if bool(topology) != bool(evidence):
                raise SectionTimelineTransitionError(
                    f"{side} topology and critical evidence must occur together"
                )
        object.__setattr__(self, "critical_kinds", critical_kinds)

    @property
    def pure_trim_tangency(self) -> bool:
        return bool(self.critical_kinds) and set(self.critical_kinds).issubset(
            _TRIM_CRITICAL_KINDS
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "knotId": self.knot_id,
            "criticalTime": self.critical_time,
            "criticalFrameIndex": self.critical_frame_index,
            "beforeFrameIndex": self.before_frame_index,
            "afterFrameIndex": self.after_frame_index,
            "beforeBank": self.before_bank,
            "criticalBank": self.critical_bank,
            "afterBank": self.after_bank,
            "leftStart": self.left_start,
            "rightEnd": self.right_end,
            "leftCrossfade": self.left_crossfade,
            "rightCrossfade": self.right_crossfade,
            "topologyEventIds": list(self.topology_event_ids),
            "criticalEventIds": list(self.critical_event_ids),
            "leftTopologyEventIds": list(self.left_topology_event_ids),
            "rightTopologyEventIds": list(self.right_topology_event_ids),
            "leftCriticalEventIds": list(self.left_critical_event_ids),
            "rightCriticalEventIds": list(self.right_critical_event_ids),
            "criticalKinds": list(self.critical_kinds),
            "pureTrimTangency": self.pure_trim_tangency,
        }


@dataclass(frozen=True, slots=True)
class SectionTimelineTransitionLayer:
    bank_index: int
    geometry_time: float
    opacity: float
    role: SectionTimelineLayerRole
    reference_frame_index: int

    def __post_init__(self) -> None:
        _bank(self.bank_index, "bank_index")
        object.__setattr__(
            self,
            "geometry_time",
            _finite(self.geometry_time, "geometry_time"),
        )
        opacity = _finite(self.opacity, "layer opacity")
        if opacity < 0.0 or opacity > 1.0:
            raise SectionTimelineTransitionError(
                "layer opacity must lie in [0, 1]"
            )
        try:
            role = SectionTimelineLayerRole(self.role)
        except (TypeError, ValueError) as exc:
            raise SectionTimelineTransitionError(
                "role must be a SectionTimelineLayerRole"
            ) from exc
        _frame_index(self.reference_frame_index, "reference_frame_index")
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "role", role)

    def to_dict(self) -> dict[str, object]:
        return {
            "bankIndex": self.bank_index,
            "geometryTime": self.geometry_time,
            "opacity": self.opacity,
            "role": self.role.value,
            "referenceFrameIndex": self.reference_frame_index,
        }


@dataclass(frozen=True, slots=True)
class SectionTimelineTransitionState:
    time: float
    layers: tuple[SectionTimelineTransitionLayer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _finite(self.time, "state time"))
        if (
            not isinstance(self.layers, tuple)
            or not self.layers
            or len(self.layers) > 2
        ):
            raise SectionTimelineTransitionError(
                "transition state requires one or two layers"
            )
        if not all(
            isinstance(item, SectionTimelineTransitionLayer)
            for item in self.layers
        ):
            raise TypeError(
                "layers must contain SectionTimelineTransitionLayer values"
            )
        if any(item.opacity <= _OPACITY_TOLERANCE for item in self.layers):
            raise SectionTimelineTransitionError(
                "transition layers must have positive active opacity"
            )
        banks = tuple(item.bank_index for item in self.layers)
        if len(set(banks)) != len(banks):
            raise SectionTimelineTransitionError(
                "transition layers must use unique banks"
            )
        if abs(sum(item.opacity for item in self.layers) - 1.0) > 1.0e-12:
            raise SectionTimelineTransitionError(
                "transition layer opacities must sum to one"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "time": self.time,
            "layers": [item.to_dict() for item in self.layers],
        }


@dataclass(frozen=True, slots=True)
class SectionTimelineTransitionPlan:
    timeline: SectionTimeline
    mode: SectionTimelineTransitionMode
    transition_fraction: float
    knots: tuple[SectionTimelineTransitionKnot, ...]
    schema: str = SECTION_TIMELINE_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SECTION_TIMELINE_TRANSITION_SCHEMA:
            raise SectionTimelineTransitionError("invalid transition schema")
        if not isinstance(self.timeline, SectionTimeline):
            raise TypeError("timeline must be a SectionTimeline")
        try:
            mode = SectionTimelineTransitionMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise SectionTimelineTransitionError(
                "mode must be 'crossfade' or 'cut'"
            ) from exc
        fraction = _fraction(self.transition_fraction)
        if mode is SectionTimelineTransitionMode.CROSSFADE and fraction <= 0.0:
            raise SectionTimelineTransitionError(
                "crossfade mode requires a positive transition_fraction"
            )
        if not isinstance(self.knots, tuple) or not all(
            isinstance(item, SectionTimelineTransitionKnot)
            for item in self.knots
        ):
            raise TypeError(
                "knots must be a tuple of SectionTimelineTransitionKnot values"
            )
        times = tuple(item.critical_time for item in self.knots)
        if times != tuple(sorted(set(times))):
            raise SectionTimelineTransitionError(
                "transition knots must use unique increasing critical times"
            )
        for left, right in zip(self.knots, self.knots[1:]):
            if left.right_end > right.left_start:
                raise SectionTimelineTransitionError(
                    "global transition windows must not overlap"
                )
        expected = _canonical_transition_knots(self.timeline, mode, fraction)
        if self.knots != expected:
            raise SectionTimelineTransitionError(
                "transition knots do not match the canonical SectionTimeline evidence"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "transition_fraction", fraction)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sectionId": self.timeline.section_id,
            "mode": self.mode.value,
            "transitionFraction": self.transition_fraction,
            "frameBanks": list(self.timeline.topology_frame_banks),
            "knots": [item.to_dict() for item in self.knots],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _family_changed(left: object, right: object) -> bool:
    return left.signature.conic_family is not right.signature.conic_family


def _canonical_transition_knots(
    timeline: SectionTimeline,
    selected_mode: SectionTimelineTransitionMode,
    fraction: float,
) -> tuple[SectionTimelineTransitionKnot, ...]:
    """Rebuild the only canonical knot sequence for one timeline and policy."""

    critical_by_id = {item.event_id: item for item in timeline.critical_events}
    certification_by_topology = {
        item.topology_event_id: item
        for item in timeline.topology_certifications
    }
    topology_by_time: dict[float, list[object]] = {}
    for event in timeline.animation.topology_events:
        certification = certification_by_topology[event.event_id]
        critical_times = {
            critical_by_id[item].time for item in certification.critical_event_ids
        }
        if len(critical_times) != 1:
            raise SectionTimelineTransitionError(
                "one topology event must resolve to one analytic critical time"
            )
        critical_time = next(iter(critical_times))
        if critical_time not in {event.left_time, event.right_time}:
            raise SectionTimelineTransitionError(
                "topology critical time must be an adjacent sampled endpoint"
            )
        topology_by_time.setdefault(critical_time, []).append(event)

    frame_index_by_time = {
        sample.time: index for index, sample in enumerate(timeline.samples)
    }
    knots: list[SectionTimelineTransitionKnot] = []
    frames = timeline.animation.frames
    banks = timeline.topology_frame_banks
    for knot_index, critical_time in enumerate(sorted(topology_by_time)):
        try:
            critical_index = frame_index_by_time[critical_time]
        except KeyError as exc:
            raise SectionTimelineTransitionError(
                "analytic critical time has no exact timeline frame"
            ) from exc
        topology_events = tuple(
            sorted(
                topology_by_time[critical_time],
                key=lambda item: item.event_id,
            )
        )
        left_events = tuple(
            item
            for item in topology_events
            if item.right_frame_index == critical_index
        )
        right_events = tuple(
            item
            for item in topology_events
            if item.left_frame_index == critical_index
        )
        if len(left_events) > 1 or len(right_events) > 1:
            raise SectionTimelineTransitionError(
                "one critical frame may have at most one topology event per side"
            )
        if len(left_events) + len(right_events) != len(topology_events):
            raise SectionTimelineTransitionError(
                "topology event is not adjacent to its analytic critical frame"
            )
        left_event = left_events[0] if left_events else None
        right_event = right_events[0] if right_events else None
        before_index = None if left_event is None else critical_index - 1
        after_index = None if right_event is None else critical_index + 1
        left_topology_ids = tuple(item.event_id for item in left_events)
        right_topology_ids = tuple(item.event_id for item in right_events)
        left_critical_ids = tuple(
            sorted(
                {
                    critical_id
                    for item in left_events
                    for critical_id in certification_by_topology[
                        item.event_id
                    ].critical_event_ids
                }
            )
        )
        right_critical_ids = tuple(
            sorted(
                {
                    critical_id
                    for item in right_events
                    for critical_id in certification_by_topology[
                        item.event_id
                    ].critical_event_ids
                }
            )
        )
        critical_ids = tuple(
            sorted(set(left_critical_ids) | set(right_critical_ids))
        )
        critical_kinds = tuple(
            sorted(
                {
                    kind
                    for critical_id in critical_ids
                    for kind in critical_by_id[critical_id].kinds
                }
            )
        )
        left_family_change = bool(
            before_index is not None
            and _family_changed(frames[before_index], frames[critical_index])
        )
        right_family_change = bool(
            after_index is not None
            and _family_changed(frames[critical_index], frames[after_index])
        )
        left_crossfade = bool(
            selected_mode is SectionTimelineTransitionMode.CROSSFADE
            and left_family_change
        )
        right_crossfade = bool(
            selected_mode is SectionTimelineTransitionMode.CROSSFADE
            and right_family_change
        )
        left_start = critical_time
        if left_crossfade:
            assert before_index is not None
            left_start = critical_time - fraction * (
                critical_time - timeline.samples[before_index].time
            )
        right_end = critical_time
        if right_crossfade:
            assert after_index is not None
            right_end = critical_time + fraction * (
                timeline.samples[after_index].time - critical_time
            )
        knots.append(
            SectionTimelineTransitionKnot(
                knot_id=f"{timeline.section_id}:transition:{knot_index:04d}",
                critical_time=critical_time,
                critical_frame_index=critical_index,
                before_frame_index=before_index,
                after_frame_index=after_index,
                before_bank=None if before_index is None else banks[before_index],
                critical_bank=banks[critical_index],
                after_bank=None if after_index is None else banks[after_index],
                left_start=left_start,
                right_end=right_end,
                left_crossfade=left_crossfade,
                right_crossfade=right_crossfade,
                topology_event_ids=tuple(
                    sorted(item.event_id for item in topology_events)
                ),
                critical_event_ids=critical_ids,
                left_topology_event_ids=left_topology_ids,
                right_topology_event_ids=right_topology_ids,
                left_critical_event_ids=left_critical_ids,
                right_critical_event_ids=right_critical_ids,
                critical_kinds=critical_kinds,
            )
        )
    for left, right in zip(knots, knots[1:]):
        if left.right_end > right.left_start:
            raise SectionTimelineTransitionError(
                "global transition windows must not overlap"
            )
    return tuple(knots)


def build_section_timeline_transition_plan(
    timeline: SectionTimeline,
    *,
    transition_fraction: float = 0.25,
    mode: SectionTimelineTransitionMode | str = (
        SectionTimelineTransitionMode.CROSSFADE
    ),
) -> SectionTimelineTransitionPlan:
    """Build one non-overlapping global-time handoff plan."""

    if not isinstance(timeline, SectionTimeline):
        raise TypeError("timeline must be a SectionTimeline")
    try:
        selected_mode = SectionTimelineTransitionMode(mode)
    except (TypeError, ValueError) as exc:
        raise SectionTimelineTransitionError(
            "mode must be 'crossfade' or 'cut'"
        ) from exc
    fraction = _fraction(transition_fraction)
    if (
        selected_mode is SectionTimelineTransitionMode.CROSSFADE
        and fraction <= 0.0
    ):
        raise SectionTimelineTransitionError(
            "crossfade mode requires a positive transition_fraction"
        )
    return SectionTimelineTransitionPlan(
        timeline=timeline,
        mode=selected_mode,
        transition_fraction=fraction,
        knots=_canonical_transition_knots(timeline, selected_mode, fraction),
    )


def _single_layer(
    plan: SectionTimelineTransitionPlan,
    frame_index: int,
    time: float,
    role: SectionTimelineLayerRole = SectionTimelineLayerRole.LIVE,
    *,
    geometry_time: float | None = None,
) -> SectionTimelineTransitionState:
    return SectionTimelineTransitionState(
        time,
        (
            SectionTimelineTransitionLayer(
                bank_index=plan.timeline.topology_frame_banks[frame_index],
                geometry_time=time if geometry_time is None else geometry_time,
                opacity=1.0,
                role=role,
                reference_frame_index=frame_index,
            ),
        ),
    )


def _active_state(
    time: float,
    layers: tuple[SectionTimelineTransitionLayer, ...],
) -> SectionTimelineTransitionState:
    active = tuple(item for item in layers if item.opacity > _OPACITY_TOLERANCE)
    if not active:
        raise SectionTimelineTransitionError(
            "transition opacity filtering removed every active layer"
        )
    if len(active) == 1 and active[0].opacity != 1.0:
        only = active[0]
        active = (
            SectionTimelineTransitionLayer(
                bank_index=only.bank_index,
                geometry_time=only.geometry_time,
                opacity=1.0,
                role=only.role,
                reference_frame_index=only.reference_frame_index,
            ),
        )
    return SectionTimelineTransitionState(time, active)


def _trim_geometry_time(
    plan: SectionTimelineTransitionPlan,
    knot: SectionTimelineTransitionKnot,
) -> tuple[int, float]:
    """Select a certified one-sided live geometry for a pure trim cut."""

    if knot.before_frame_index is not None:
        reference = knot.before_frame_index
        neighbor = plan.timeline.samples[reference].time
    elif knot.after_frame_index is not None:
        reference = knot.after_frame_index
        neighbor = plan.timeline.samples[reference].time
    else:  # Canonical topology knots always have at least one changed side.
        raise SectionTimelineTransitionError(
            "a pure trim cut has no neighboring live frame"
        )
    gap = abs(knot.critical_time - neighbor)
    if gap <= 0.0:
        raise SectionTimelineTransitionError(
            "a pure trim cut has no non-empty neighboring interval"
        )
    offset = min(1.0e-6, gap * 1.0e-3)
    direction = -1.0 if neighbor < knot.critical_time else 1.0
    candidate = knot.critical_time + direction * offset
    low, high = sorted((neighbor, knot.critical_time))
    inside = (
        low <= candidate < high
        if direction < 0.0
        else low < candidate <= high
    )
    if not inside:
        candidate = nextafter(knot.critical_time, neighbor)
    if candidate == knot.critical_time or not low <= candidate <= high:
        candidate = neighbor
    if candidate == knot.critical_time:
        raise SectionTimelineTransitionError(
            "a pure trim cut has no representable one-sided geometry time"
        )
    return reference, candidate


def section_timeline_transition_state_at(
    plan: SectionTimelineTransitionPlan,
    time: float,
) -> SectionTimelineTransitionState:
    """Return exact one- or two-bank opacity state at a global timeline time."""

    if not isinstance(plan, SectionTimelineTransitionPlan):
        raise TypeError("plan must be a SectionTimelineTransitionPlan")
    value = _finite(time, "transition time")
    sample_times = tuple(item.time for item in plan.timeline.samples)
    if value < sample_times[0] or value > sample_times[-1]:
        raise SectionTimelineTransitionError(
            "transition time lies outside the SectionTimeline"
        )
    for knot in plan.knots:
        if value == knot.critical_time:
            if knot.pure_trim_tangency:
                reference, geometry_time = _trim_geometry_time(plan, knot)
                return _single_layer(
                    plan,
                    reference,
                    value,
                    SectionTimelineLayerRole.LIVE,
                    geometry_time=geometry_time,
                )
            return _single_layer(
                plan,
                knot.critical_frame_index,
                value,
                SectionTimelineLayerRole.EXACT_CRITICAL,
                geometry_time=knot.critical_time,
            )
        if knot.left_crossfade and knot.left_start < value < knot.critical_time:
            assert knot.before_frame_index is not None
            ratio = (value - knot.left_start) / (
                knot.critical_time - knot.left_start
            )
            critical_opacity = _smoothstep(ratio)
            return _active_state(
                value,
                (
                    SectionTimelineTransitionLayer(
                        bank_index=plan.timeline.topology_frame_banks[
                            knot.before_frame_index
                        ],
                        geometry_time=value,
                        opacity=1.0 - critical_opacity,
                        role=SectionTimelineLayerRole.LIVE_BEFORE,
                        reference_frame_index=knot.before_frame_index,
                    ),
                    SectionTimelineTransitionLayer(
                        bank_index=knot.critical_bank,
                        geometry_time=knot.critical_time,
                        opacity=critical_opacity,
                        role=SectionTimelineLayerRole.EXACT_CRITICAL,
                        reference_frame_index=knot.critical_frame_index,
                    ),
                ),
            )
        if knot.right_crossfade and knot.critical_time < value < knot.right_end:
            assert knot.after_frame_index is not None
            ratio = (value - knot.critical_time) / (
                knot.right_end - knot.critical_time
            )
            live_opacity = _smoothstep(ratio)
            return _active_state(
                value,
                (
                    SectionTimelineTransitionLayer(
                        bank_index=plan.timeline.topology_frame_banks[
                            knot.after_frame_index
                        ],
                        geometry_time=value,
                        opacity=live_opacity,
                        role=SectionTimelineLayerRole.LIVE_AFTER,
                        reference_frame_index=knot.after_frame_index,
                    ),
                    SectionTimelineTransitionLayer(
                        bank_index=knot.critical_bank,
                        geometry_time=knot.critical_time,
                        opacity=1.0 - live_opacity,
                        role=SectionTimelineLayerRole.EXACT_CRITICAL,
                        reference_frame_index=knot.critical_frame_index,
                    ),
                ),
            )

    exact = next(
        (
            index
            for index, sample_time in enumerate(sample_times)
            if sample_time == value
        ),
        None,
    )
    if exact is not None:
        return _single_layer(plan, exact, value)
    right = bisect_right(sample_times, value)
    left = right - 1
    if right >= len(sample_times):
        return _single_layer(plan, left, value)
    left_bank = plan.timeline.topology_frame_banks[left]
    right_bank = plan.timeline.topology_frame_banks[right]
    if left_bank == right_bank:
        return _single_layer(plan, left, value)
    right_is_critical = any(
        knot.critical_frame_index == right for knot in plan.knots
    )
    return _single_layer(plan, left if right_is_critical else right, value)


__all__ = [
    "SECTION_TIMELINE_TRANSITION_SCHEMA",
    "SectionTimelineLayerRole",
    "SectionTimelineTransitionError",
    "SectionTimelineTransitionKnot",
    "SectionTimelineTransitionLayer",
    "SectionTimelineTransitionMode",
    "SectionTimelineTransitionPlan",
    "SectionTimelineTransitionState",
    "build_section_timeline_transition_plan",
    "section_timeline_transition_state_at",
]
