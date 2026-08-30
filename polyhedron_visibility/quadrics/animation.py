"""Renderer-neutral continuity tracking for animated quadric sections.

This module deliberately keeps three decisions separate:

* :func:`compute_quadric_section` decides the exact conic and finite topology
  of every authored frame;
* this module matches connected curve components only while that topology is
  unchanged;
* a later renderer may use the fixed two-slot capacity plan without creating
  or deleting objects inside an updater.

No dense point cloud is used to classify topology.  A bounded set of analytic
curve probes is used only to disambiguate component identity and orientation
inside an already-proven topology interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import permutations
import json
from math import atan2, cos, isfinite, sin
from typing import Callable, Sequence

import numpy as np

from ..geometry import GeometryContext, ResolvedGeometryContext
from ..topology import ParameterInterval
from .conics import ConicKind
from .contract import ConeSpec, CylinderSpec, SectionPlane, SphereSpec
from .curves import ParametricConicBranch
from .sections import QuadricSurfaceSpec, compute_quadric_section
from .trace import (
    FiniteSectionTopology,
    QuadricSectionTrace,
    SectionBranchTrace,
    SectionComponentTrace,
)


QUADRIC_SECTION_ANIMATION_SCHEMA = "manim-quadric-section-animation/v1"
MAX_SECTION_BRANCH_SLOTS = 2
_PERIODIC_EPSILON = 1.0e-10
_MATCH_AMBIGUITY = 1.0e-10
_PROBES = (0.0, 0.25, 0.5, 0.75, 1.0)
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(16)


class SectionAnimationError(ValueError):
    """An animated section cannot be continued without guessing."""


class BranchContinuityError(SectionAnimationError):
    """Two same-topology components have no unambiguous correspondence."""


class MovingPointContinuityError(SectionAnimationError):
    """A moving point crossed a topology event without an authored rule."""


class SectionConicFamily(str, Enum):
    """Topological family of an affine supporting conic."""

    OVAL = "oval"
    PARABOLA = "parabola"
    HYPERBOLA = "hyperbola"
    POINT = "point"
    INTERSECTING_LINES = "intersecting_lines"
    PARALLEL_LINES = "parallel_lines"
    COINCIDENT_LINE = "coincident_line"
    EMPTY = "empty"


class TopologyEventKind(str, Enum):
    """Reasons why branch identity must not be silently carried forward."""

    CONIC_FAMILY_CHANGED = "conic_family_changed"
    FINITE_TOPOLOGY_CHANGED = "finite_topology_changed"
    BRANCH_COUNT_CHANGED = "branch_count_changed"
    COMPONENT_COUNT_CHANGED = "component_count_changed"
    CLOSEDNESS_CHANGED = "closedness_changed"
    ISOLATED_POINT_COUNT_CHANGED = "isolated_point_count_changed"
    ENTERED_DEGENERACY = "entered_degeneracy"
    EXITED_DEGENERACY = "exited_degeneracy"


class PointParameterMode(str, Enum):
    """How a value in ``[0, 1]`` selects a point on one finite component."""

    NORMALIZED_PARAMETER = "normalized_parameter"
    ARC_LENGTH_FRACTION = "arc_length_fraction"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SectionAnimationError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SectionAnimationError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SectionAnimationError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise SectionAnimationError(f"{label} must be finite")
    return result


def _fraction(value: object, label: str = "fraction") -> float:
    result = _finite(value, label)
    if result < 0.0 or result > 1.0:
        raise SectionAnimationError(f"{label} must lie in [0, 1]")
    return result


def _tuple3(value: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise SectionAnimationError("world point must contain three finite numbers")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def _conic_family(kind: ConicKind) -> SectionConicFamily:
    if kind in {ConicKind.CIRCLE, ConicKind.ELLIPSE}:
        return SectionConicFamily.OVAL
    return SectionConicFamily(kind.value)


def _is_degenerate(family: SectionConicFamily) -> bool:
    return family not in {
        SectionConicFamily.OVAL,
        SectionConicFamily.PARABOLA,
        SectionConicFamily.HYPERBOLA,
    }


@dataclass(frozen=True, slots=True)
class SectionTopologySignature:
    """Serializable exact-topology summary for one section frame.

    Circle and ellipse retain their exact ``supporting_kind`` but share the
    ``OVAL`` family, so a harmless circle-to-ellipse deformation does not
    force a branch reset.
    """

    supporting_kind: ConicKind
    conic_family: SectionConicFamily
    finite_topology: FiniteSectionTopology
    branch_count: int
    component_count: int
    component_closedness: tuple[bool, ...]
    isolated_point_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.supporting_kind, ConicKind):
            raise TypeError("supporting_kind must be a ConicKind")
        if not isinstance(self.conic_family, SectionConicFamily):
            raise TypeError("conic_family must be a SectionConicFamily")
        if self.conic_family is not _conic_family(self.supporting_kind):
            raise SectionAnimationError("supporting kind and conic family disagree")
        if not isinstance(self.finite_topology, FiniteSectionTopology):
            raise TypeError("finite_topology must be a FiniteSectionTopology")
        for label, value in (
            ("branch_count", self.branch_count),
            ("component_count", self.component_count),
            ("isolated_point_count", self.isolated_point_count),
        ):
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise SectionAnimationError(f"{label} must be a non-negative integer")
        if len(self.component_closedness) != self.component_count:
            raise SectionAnimationError(
                "component_closedness must describe every finite component"
            )
        if tuple(sorted(self.component_closedness)) != self.component_closedness:
            raise SectionAnimationError(
                "component_closedness must use canonical false-before-true order"
            )

    @classmethod
    def from_trace(cls, trace: QuadricSectionTrace) -> "SectionTopologySignature":
        if not isinstance(trace, QuadricSectionTrace):
            raise TypeError("trace must be a QuadricSectionTrace")
        return cls(
            supporting_kind=trace.supporting_kind,
            conic_family=_conic_family(trace.supporting_kind),
            finite_topology=trace.finite_topology,
            branch_count=len(trace.branches),
            component_count=len(trace.components),
            component_closedness=tuple(sorted(item.closed for item in trace.components)),
            isolated_point_count=len(trace.isolated_world_points),
        )

    @property
    def degenerate(self) -> bool:
        return _is_degenerate(self.conic_family)

    def topologically_equivalent(self, other: "SectionTopologySignature") -> bool:
        if not isinstance(other, SectionTopologySignature):
            return False
        return (
            self.conic_family,
            self.finite_topology,
            self.branch_count,
            self.component_count,
            self.component_closedness,
            self.isolated_point_count,
        ) == (
            other.conic_family,
            other.finite_topology,
            other.branch_count,
            other.component_count,
            other.component_closedness,
            other.isolated_point_count,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "supportingKind": self.supporting_kind.value,
            "conicFamily": self.conic_family.value,
            "finiteTopology": self.finite_topology.value,
            "branchCount": self.branch_count,
            "componentCount": self.component_count,
            "componentClosedness": list(self.component_closedness),
            "isolatedPointCount": self.isolated_point_count,
            "degenerate": self.degenerate,
        }


@dataclass(frozen=True, slots=True)
class SectionAnimationSample:
    """One authored time, finite surface, and infinite mathematical plane."""

    time: float
    surface: QuadricSurfaceSpec
    plane: SectionPlane

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _finite(self.time, "sample time"))
        if not isinstance(self.surface, (SphereSpec, CylinderSpec, ConeSpec)):
            raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
        if not isinstance(self.plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")


@dataclass(frozen=True, slots=True)
class BranchCapacityPlan:
    """Fixed renderer allocation metadata for at most two curve branches."""

    slot_ids: tuple[str, str]
    required_slots: int
    maximum_slots: int = MAX_SECTION_BRANCH_SLOTS

    def __post_init__(self) -> None:
        if self.maximum_slots != MAX_SECTION_BRANCH_SLOTS:
            raise SectionAnimationError("quadric sections use exactly two branch slots")
        if len(self.slot_ids) != self.maximum_slots or len(set(self.slot_ids)) != 2:
            raise SectionAnimationError("capacity plan requires two unique slot ids")
        if not 0 <= self.required_slots <= self.maximum_slots:
            raise SectionAnimationError("required branch slots exceed fixed capacity")

    def to_dict(self) -> dict[str, object]:
        return {
            "maximumSlots": self.maximum_slots,
            "requiredSlots": self.required_slots,
            "slotIds": list(self.slot_ids),
            "preallocate": True,
        }


@dataclass(frozen=True, slots=True)
class TrackedSectionBranch:
    """One stable finite-component identity and its current raw parameter map."""

    stable_branch_id: str
    source_branch_id: str
    source_component_id: str
    capacity_slot: int
    orientation: int
    phase_offset: float
    closed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stable_branch_id", _identity(self.stable_branch_id, "stable_branch_id")
        )
        object.__setattr__(
            self, "source_branch_id", _identity(self.source_branch_id, "source_branch_id")
        )
        object.__setattr__(
            self,
            "source_component_id",
            _identity(self.source_component_id, "source_component_id"),
        )
        if self.capacity_slot not in {0, 1}:
            raise SectionAnimationError("capacity_slot must be 0 or 1")
        if self.orientation not in {-1, 1}:
            raise SectionAnimationError("orientation must be -1 or 1")
        object.__setattr__(
            self, "phase_offset", _finite(self.phase_offset, "phase_offset")
        )
        if not isinstance(self.closed, bool):
            raise SectionAnimationError("closed must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "stableBranchId": self.stable_branch_id,
            "sourceBranchId": self.source_branch_id,
            "sourceComponentId": self.source_component_id,
            "capacitySlot": self.capacity_slot,
            "orientation": self.orientation,
            "phaseOffset": self.phase_offset,
            "closed": self.closed,
        }


@dataclass(frozen=True, slots=True)
class TopologyEvent:
    """A sampled bracket across which branch identity is intentionally reset."""

    event_id: str
    left_frame_index: int
    right_frame_index: int
    left_time: float
    right_time: float
    before: SectionTopologySignature
    after: SectionTopologySignature
    reasons: tuple[TopologyEventKind, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identity(self.event_id, "event_id"))
        if self.left_frame_index < 0 or self.right_frame_index != self.left_frame_index + 1:
            raise SectionAnimationError("topology event must bracket adjacent frames")
        left = _finite(self.left_time, "left_time")
        right = _finite(self.right_time, "right_time")
        if right <= left:
            raise SectionAnimationError("topology event times must increase")
        object.__setattr__(self, "left_time", left)
        object.__setattr__(self, "right_time", right)
        if not self.reasons or tuple(sorted(set(self.reasons), key=lambda item: item.value)) != self.reasons:
            raise SectionAnimationError("topology event reasons must be unique and canonical")
        if self.before.topologically_equivalent(self.after):
            raise SectionAnimationError("equivalent signatures do not form a topology event")

    def to_dict(self) -> dict[str, object]:
        return {
            "eventId": self.event_id,
            "leftFrameIndex": self.left_frame_index,
            "rightFrameIndex": self.right_frame_index,
            "leftTime": self.left_time,
            "rightTime": self.right_time,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "reasons": [item.value for item in self.reasons],
        }


def _branch_and_component(
    trace: QuadricSectionTrace,
    tracked: TrackedSectionBranch,
) -> tuple[SectionBranchTrace, SectionComponentTrace]:
    try:
        branch = trace.branch_map[tracked.source_branch_id]
        component = trace.component_map[tracked.source_component_id]
    except KeyError as exc:
        raise SectionAnimationError("tracked branch references missing source geometry") from exc
    if component.branch_id != branch.branch_id:
        raise SectionAnimationError("tracked branch and component disagree")
    return branch, component


def _periodic_domain(branch: SectionBranchTrace) -> ParameterInterval | None:
    parameterization = branch.parameterization
    if not parameterization.closed or parameterization.natural_domain is None:
        return None
    return parameterization.natural_domain


def _component_segments(
    branch: SectionBranchTrace,
    component: SectionComponentTrace,
    *,
    orientation: int,
    phase_offset: float,
) -> tuple[tuple[float, float], ...]:
    """Return raw-parameter segments in stable traversal order."""

    periodic = _periodic_domain(branch)
    intervals = component.parameter_intervals
    if component.closed:
        if periodic is None:
            raise SectionAnimationError("a closed component requires a periodic branch")
        start, end = periodic.start, periodic.end
        period = periodic.length
        phase = start + ((phase_offset - start) % period)
        if abs(phase - end) <= _PERIODIC_EPSILON:
            phase = start
        if orientation > 0:
            result = ((phase, end), (start, phase))
        else:
            result = ((phase, start), (end, phase))
        return tuple((a, b) for a, b in result if abs(b - a) > _PERIODIC_EPSILON)

    ordered = list(intervals)
    if (
        periodic is not None
        and len(ordered) >= 2
        and abs(ordered[0].start - periodic.start) <= _PERIODIC_EPSILON
        and abs(ordered[-1].end - periodic.end) <= _PERIODIC_EPSILON
    ):
        # One connected arc crosses the 2*pi seam.  Traverse the upper piece
        # first and then wrap to the lower piece instead of drawing a jump.
        ordered = [ordered[-1], *ordered[1:-1], ordered[0]]
    result = tuple((item.start, item.end) for item in ordered)
    if orientation < 0:
        result = tuple((end, start) for start, end in reversed(result))
    return result


def _parameter_at_fraction(
    segments: Sequence[tuple[float, float]],
    fraction: float,
) -> float:
    value = _fraction(fraction)
    lengths = tuple(abs(end - start) for start, end in segments)
    total = sum(lengths)
    if total <= 0.0:
        raise SectionAnimationError("component has no positive parameter length")
    if value == 1.0:
        return float(segments[-1][1])
    target = value * total
    consumed = 0.0
    for (start, end), length in zip(segments, lengths):
        if target <= consumed + length:
            local = 0.0 if length == 0.0 else (target - consumed) / length
            return float(start + local * (end - start))
        consumed += length
    return float(segments[-1][1])


def _arc_length(
    branch: SectionBranchTrace,
    start: float,
    end: float,
) -> float:
    if start == end:
        return 0.0
    midpoint = 0.5 * (start + end)
    half = 0.5 * (end - start)
    parameters = midpoint + half * _GAUSS_NODES
    speeds = np.asarray(
        [float(np.linalg.norm(branch.world_tangent(float(item)))) for item in parameters],
        dtype=float,
    )
    if not np.all(np.isfinite(speeds)):
        raise SectionAnimationError("curve tangent produced a non-finite arc length")
    return abs(half) * float(np.dot(_GAUSS_WEIGHTS, speeds))


def _parameter_at_arc_fraction(
    branch: SectionBranchTrace,
    segments: Sequence[tuple[float, float]],
    fraction: float,
) -> float:
    value = _fraction(fraction)
    lengths = tuple(_arc_length(branch, start, end) for start, end in segments)
    total = sum(lengths)
    if total <= 0.0:
        raise SectionAnimationError("component has zero arc length")
    if value == 0.0:
        return float(segments[0][0])
    if value == 1.0:
        return float(segments[-1][1])
    target = value * total
    consumed = 0.0
    for (start, end), length in zip(segments, lengths):
        if target > consumed + length:
            consumed += length
            continue
        local_target = target - consumed
        low = 0.0
        high = 1.0
        for _ in range(48):
            ratio = 0.5 * (low + high)
            parameter = start + ratio * (end - start)
            partial = _arc_length(branch, start, parameter)
            if partial < local_target:
                low = ratio
            else:
                high = ratio
        ratio = 0.5 * (low + high)
        return float(start + ratio * (end - start))
    return float(segments[-1][1])


@dataclass(frozen=True, slots=True)
class TrackedSectionFrame:
    """One exact section plus stable component identities for a topology epoch."""

    frame_index: int
    time: float
    topology_epoch: int
    signature: SectionTopologySignature
    section: QuadricSectionTrace
    branches: tuple[TrackedSectionBranch, ...]

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.topology_epoch < 0:
            raise SectionAnimationError("frame index and topology epoch must be non-negative")
        object.__setattr__(self, "time", _finite(self.time, "frame time"))
        if self.signature != SectionTopologySignature.from_trace(self.section):
            raise SectionAnimationError("frame signature does not describe its section")
        if len(self.branches) != len(self.section.components):
            raise SectionAnimationError("every finite component must have one tracked branch")
        ids = tuple(item.stable_branch_id for item in self.branches)
        slots = tuple(item.capacity_slot for item in self.branches)
        if len(set(ids)) != len(ids) or len(set(slots)) != len(slots):
            raise SectionAnimationError("tracked branch identities and slots must be unique")
        if slots != tuple(sorted(slots)):
            raise SectionAnimationError("tracked branches must use canonical slot order")
        for tracked in self.branches:
            branch, component = _branch_and_component(self.section, tracked)
            if component.closed != tracked.closed:
                raise SectionAnimationError("tracked closedness disagrees with source component")
            if tracked.closed and _periodic_domain(branch) is None:
                raise SectionAnimationError("closed tracked branch is not periodic")

    @property
    def branch_map(self) -> dict[str, TrackedSectionBranch]:
        return {item.stable_branch_id: item for item in self.branches}

    @property
    def slot_map(self) -> dict[int, TrackedSectionBranch]:
        return {item.capacity_slot: item for item in self.branches}

    def source_parameter(
        self,
        stable_branch_id: str,
        fraction: float,
        *,
        mode: PointParameterMode | str = PointParameterMode.NORMALIZED_PARAMETER,
    ) -> float:
        try:
            tracked = self.branch_map[stable_branch_id]
        except KeyError as exc:
            raise SectionAnimationError(f"unknown stable branch: {stable_branch_id}") from exc
        branch, component = _branch_and_component(self.section, tracked)
        segments = _component_segments(
            branch,
            component,
            orientation=tracked.orientation,
            phase_offset=tracked.phase_offset,
        )
        selected = PointParameterMode(mode)
        if selected is PointParameterMode.NORMALIZED_PARAMETER:
            return _parameter_at_fraction(segments, fraction)
        return _parameter_at_arc_fraction(branch, segments, fraction)

    def world_point(
        self,
        stable_branch_id: str,
        fraction: float,
        *,
        mode: PointParameterMode | str = PointParameterMode.NORMALIZED_PARAMETER,
    ) -> tuple[float, float, float]:
        tracked = self.branch_map[stable_branch_id]
        branch, _ = _branch_and_component(self.section, tracked)
        parameter = self.source_parameter(stable_branch_id, fraction, mode=mode)
        return _tuple3(branch.world_point(parameter))

    def to_dict(self) -> dict[str, object]:
        return {
            "frameIndex": self.frame_index,
            "time": self.time,
            "topologyEpoch": self.topology_epoch,
            "signature": self.signature.to_dict(),
            "section": self.section.to_dict(),
            "branches": [item.to_dict() for item in self.branches],
        }


@dataclass(frozen=True, slots=True)
class SectionAnimationTrace:
    """Stable, serializable renderer-neutral result for an authored timeline."""

    section_id: str
    surface_id: str
    plane_id: str
    frames: tuple[TrackedSectionFrame, ...]
    topology_events: tuple[TopologyEvent, ...]
    capacity_plan: BranchCapacityPlan
    schema: str = QUADRIC_SECTION_ANIMATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUADRIC_SECTION_ANIMATION_SCHEMA:
            raise SectionAnimationError("invalid quadric-section animation schema")
        object.__setattr__(self, "section_id", _identity(self.section_id, "section_id"))
        object.__setattr__(self, "surface_id", _identity(self.surface_id, "surface_id"))
        object.__setattr__(self, "plane_id", _identity(self.plane_id, "plane_id"))
        if not self.frames:
            raise SectionAnimationError("animation requires at least one frame")
        if tuple(item.frame_index for item in self.frames) != tuple(range(len(self.frames))):
            raise SectionAnimationError("animation frame indices must be consecutive")
        if any(right.time <= left.time for left, right in zip(self.frames, self.frames[1:])):
            raise SectionAnimationError("animation frame times must increase strictly")
        if any(frame.section.section_id != self.section_id for frame in self.frames):
            raise SectionAnimationError("animation frames changed section identity")
        if any(frame.section.surface_id != self.surface_id for frame in self.frames):
            raise SectionAnimationError("animation frames changed surface identity")
        event_right_indices = tuple(item.right_frame_index for item in self.topology_events)
        if event_right_indices != tuple(sorted(set(event_right_indices))):
            raise SectionAnimationError("topology events must use canonical frame order")
        required = max((len(frame.branches) for frame in self.frames), default=0)
        if required != self.capacity_plan.required_slots:
            raise SectionAnimationError("capacity plan does not match animation frames")

    @property
    def event_by_right_frame(self) -> dict[int, TopologyEvent]:
        return {item.right_frame_index: item for item in self.topology_events}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sectionId": self.section_id,
            "surfaceId": self.surface_id,
            "planeId": self.plane_id,
            "capacityPlan": self.capacity_plan.to_dict(),
            "topologyEvents": [item.to_dict() for item in self.topology_events],
            "frames": [item.to_dict() for item in self.frames],
        }


def _unit(value: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not isfinite(length) or length <= 0.0:
        raise BranchContinuityError(f"{label} must be a finite non-zero vector")
    return vector / length


def _plane_normal(trace: QuadricSectionTrace) -> np.ndarray:
    embedding = np.asarray(trace.plane_embedding, dtype=float)
    return _unit(np.cross(embedding[:3, 0], embedding[:3, 1]), "plane normal")


def _plane_origin(trace: QuadricSectionTrace) -> np.ndarray:
    return np.asarray(trace.plane_embedding, dtype=float)[:3, 2]


def _minimal_rotation(source_normal: np.ndarray, target_normal: np.ndarray) -> np.ndarray:
    """Return the unique minimal rotation between non-opposite unit normals."""

    source = _unit(source_normal, "source plane normal")
    target = _unit(target_normal, "target plane normal")
    cosine = min(1.0, max(-1.0, float(np.dot(source, target))))
    if cosine < -1.0 + 1.0e-10:
        raise BranchContinuityError(
            "successive plane normals are opposite; continuity is ambiguous"
        )
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine <= 1.0e-14:
        return np.eye(3)
    axis = cross / sine
    skew = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=float,
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def _world_center(branch: SectionBranchTrace) -> np.ndarray:
    embedding = np.asarray(branch.plane_embedding, dtype=float)
    origin = np.asarray(branch.parameterization.origin, dtype=float)
    return embedding[:3, :2] @ origin + embedding[:3, 2]


def _world_axes(branch: SectionBranchTrace) -> tuple[np.ndarray, np.ndarray]:
    embedding = np.asarray(branch.plane_embedding, dtype=float)
    linear = embedding[:3, :2]
    return (
        linear @ np.asarray(branch.parameterization.first_axis, dtype=float),
        linear @ np.asarray(branch.parameterization.second_axis, dtype=float),
    )


def _raw_parameter_for_mapping(
    trace: QuadricSectionTrace,
    tracked: TrackedSectionBranch,
    fraction: float,
) -> float:
    branch, component = _branch_and_component(trace, tracked)
    segments = _component_segments(
        branch,
        component,
        orientation=tracked.orientation,
        phase_offset=tracked.phase_offset,
    )
    return _parameter_at_fraction(segments, fraction)


def _relative_probe_points(
    trace: QuadricSectionTrace,
    tracked: TrackedSectionBranch,
) -> tuple[np.ndarray, ...]:
    branch, _ = _branch_and_component(trace, tracked)
    origin = _plane_origin(trace)
    return tuple(
        branch.world_point(_raw_parameter_for_mapping(trace, tracked, fraction)) - origin
        for fraction in _PROBES
    )


def _open_candidate(
    previous_frame: TrackedSectionFrame,
    previous: TrackedSectionBranch,
    current_trace: QuadricSectionTrace,
    current_component: SectionComponentTrace,
) -> tuple[float, int, float]:
    current_branch = current_trace.branch_map[current_component.branch_id]
    rotation = _minimal_rotation(
        _plane_normal(previous_frame.section), _plane_normal(current_trace)
    )
    previous_points = tuple(
        rotation @ point
        for point in _relative_probe_points(previous_frame.section, previous)
    )
    candidates: list[tuple[float, int]] = []
    current_origin = _plane_origin(current_trace)
    for orientation in (1, -1):
        candidate = TrackedSectionBranch(
            stable_branch_id=previous.stable_branch_id,
            source_branch_id=current_branch.branch_id,
            source_component_id=current_component.component_id,
            capacity_slot=previous.capacity_slot,
            orientation=orientation,
            phase_offset=0.0,
            closed=False,
        )
        current_points = tuple(
            current_branch.world_point(
                _raw_parameter_for_mapping(current_trace, candidate, fraction)
            )
            - current_origin
            for fraction in _PROBES
        )
        scale = max(
            1.0e-15,
            *(float(np.linalg.norm(item)) for item in previous_points),
            *(float(np.linalg.norm(item)) for item in current_points),
        )
        cost = sum(
            float(np.dot(left - right, left - right)) / (scale * scale)
            for left, right in zip(previous_points, current_points)
        )
        candidates.append((cost, orientation))
    cost, orientation = min(candidates, key=lambda item: (item[0], -item[1]))
    return cost, orientation, 0.0


def _closed_candidate(
    previous_frame: TrackedSectionFrame,
    previous: TrackedSectionBranch,
    current_trace: QuadricSectionTrace,
    current_component: SectionComponentTrace,
) -> tuple[float, int, float]:
    previous_branch, _ = _branch_and_component(previous_frame.section, previous)
    current_branch = current_trace.branch_map[current_component.branch_id]
    previous_parameter = _raw_parameter_for_mapping(previous_frame.section, previous, 0.0)
    previous_radial = previous_branch.world_point(previous_parameter) - _world_center(previous_branch)
    previous_tangent = previous.orientation * previous_branch.world_tangent(previous_parameter)
    rotation = _minimal_rotation(
        _plane_normal(previous_frame.section), _plane_normal(current_trace)
    )
    target_radial = _unit(rotation @ previous_radial, "transported radial direction")
    target_tangent = _unit(rotation @ previous_tangent, "transported tangent direction")
    first, second = _world_axes(current_branch)
    first_unit = _unit(first, "current first ellipse axis")
    second_unit = _unit(second, "current second ellipse axis")
    # For an ellipse the radial direction is
    # ``a*cos(t)*first_unit + b*sin(t)*second_unit``.  Using the raw direction
    # components as ``atan2`` inputs is correct only for a circle and causes a
    # visible phase jump when the semi-axis lengths change.  Divide by the
    # authored radii before recovering the analytic parameter.
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    phase = atan2(
        float(np.dot(target_radial, second_unit)) / second_length,
        float(np.dot(target_radial, first_unit)) / first_length,
    )
    periodic = _periodic_domain(current_branch)
    if periodic is None:
        raise BranchContinuityError("closed continuity requires a periodic branch")
    phase = periodic.start + ((phase - periodic.start) % periodic.length)
    # Retain an unwrapped phase in the continuity trace.  The component
    # evaluator reduces it back into the raw conic domain, while this value
    # itself stays numerically close to the previous frame across the 2*pi
    # seam (for example -0.01 instead of 2*pi-0.01).
    phase += round((previous.phase_offset - phase) / periodic.length) * periodic.length
    tangent = current_branch.world_tangent(phase)
    orientation = 1 if float(np.dot(_unit(tangent, "current tangent"), target_tangent)) >= 0.0 else -1
    current_radial = _unit(
        current_branch.world_point(phase) - _world_center(current_branch),
        "current radial direction",
    )
    center_delta = _world_center(current_branch) - _plane_origin(current_trace)
    previous_center_delta = rotation @ (
        _world_center(previous_branch) - _plane_origin(previous_frame.section)
    )
    scale = max(
        1.0e-15,
        float(np.linalg.norm(center_delta)),
        float(np.linalg.norm(previous_center_delta)),
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
    )
    cost = (
        float(np.dot(current_radial - target_radial, current_radial - target_radial))
        + float(np.dot(center_delta - previous_center_delta, center_delta - previous_center_delta))
        / (scale * scale)
    )
    return cost, orientation, phase


def _candidate(
    previous_frame: TrackedSectionFrame,
    previous: TrackedSectionBranch,
    current_trace: QuadricSectionTrace,
    current_component: SectionComponentTrace,
) -> tuple[float, int, float]:
    if previous.closed != current_component.closed:
        return float("inf"), 1, 0.0
    if current_component.closed:
        return _closed_candidate(
            previous_frame, previous, current_trace, current_component
        )
    return _open_candidate(previous_frame, previous, current_trace, current_component)


def _initial_mappings(
    section_id: str,
    epoch: int,
    trace: QuadricSectionTrace,
) -> tuple[TrackedSectionBranch, ...]:
    if len(trace.components) > MAX_SECTION_BRANCH_SLOTS:
        raise SectionAnimationError(
            "finite quadric section exceeds the fixed two-branch render capacity"
        )
    result: list[TrackedSectionBranch] = []
    for slot, component in enumerate(trace.components):
        branch = trace.branch_map[component.branch_id]
        periodic = _periodic_domain(branch)
        phase = periodic.start if component.closed and periodic is not None else 0.0
        result.append(
            TrackedSectionBranch(
                stable_branch_id=f"{section_id}:epoch:{epoch:04d}:branch:{slot:02d}",
                source_branch_id=branch.branch_id,
                source_component_id=component.component_id,
                capacity_slot=slot,
                orientation=1,
                phase_offset=phase,
                closed=component.closed,
            )
        )
    return tuple(result)


def _continued_mappings(
    previous_frame: TrackedSectionFrame,
    current_trace: QuadricSectionTrace,
) -> tuple[TrackedSectionBranch, ...]:
    previous = previous_frame.branches
    current = current_trace.components
    if len(previous) != len(current):
        raise BranchContinuityError("same-topology frames changed component count")
    if not current:
        return ()
    ranked: list[
        tuple[
            float,
            tuple[str, ...],
            tuple[tuple[SectionComponentTrace, int, float], ...],
        ]
    ] = []
    for ordering in permutations(current):
        choices: list[tuple[SectionComponentTrace, int, float]] = []
        cost = 0.0
        for old, component in zip(previous, ordering):
            pair_cost, orientation, phase = _candidate(
                previous_frame, old, current_trace, component
            )
            cost += pair_cost
            choices.append((component, orientation, phase))
        ranked.append(
            (
                cost,
                tuple(component.component_id for component in ordering),
                tuple(choices),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]))
    best = ranked[0]
    if not isfinite(best[0]):
        raise BranchContinuityError("no compatible branch correspondence exists")
    if len(ranked) > 1:
        second = ranked[1]
        tolerance = _MATCH_AMBIGUITY * max(1.0, abs(best[0]), abs(second[0]))
        if abs(second[0] - best[0]) <= tolerance:
            raise BranchContinuityError(
                "same-topology branch matching is symmetric; an explicit rule is required"
            )
    result = []
    for old, (component, orientation, phase) in zip(previous, best[2]):
        result.append(
            TrackedSectionBranch(
                stable_branch_id=old.stable_branch_id,
                source_branch_id=component.branch_id,
                source_component_id=component.component_id,
                capacity_slot=old.capacity_slot,
                orientation=orientation,
                phase_offset=phase,
                closed=component.closed,
            )
        )
    result.sort(key=lambda item: item.capacity_slot)
    return tuple(result)


def _event_reasons(
    before: SectionTopologySignature,
    after: SectionTopologySignature,
) -> tuple[TopologyEventKind, ...]:
    reasons: set[TopologyEventKind] = set()
    if before.conic_family is not after.conic_family:
        reasons.add(TopologyEventKind.CONIC_FAMILY_CHANGED)
    if before.finite_topology is not after.finite_topology:
        reasons.add(TopologyEventKind.FINITE_TOPOLOGY_CHANGED)
    if before.branch_count != after.branch_count:
        reasons.add(TopologyEventKind.BRANCH_COUNT_CHANGED)
    if before.component_count != after.component_count:
        reasons.add(TopologyEventKind.COMPONENT_COUNT_CHANGED)
    if before.component_closedness != after.component_closedness:
        reasons.add(TopologyEventKind.CLOSEDNESS_CHANGED)
    if before.isolated_point_count != after.isolated_point_count:
        reasons.add(TopologyEventKind.ISOLATED_POINT_COUNT_CHANGED)
    if not before.degenerate and after.degenerate:
        reasons.add(TopologyEventKind.ENTERED_DEGENERACY)
    if before.degenerate and not after.degenerate:
        reasons.add(TopologyEventKind.EXITED_DEGENERACY)
    if not reasons:
        raise SectionAnimationError("non-equivalent signatures require an event reason")
    return tuple(sorted(reasons, key=lambda item: item.value))


def track_quadric_section_animation(
    section_id: str,
    samples: Sequence[SectionAnimationSample],
    *,
    context: GeometryContext | ResolvedGeometryContext | None = None,
    coefficient_tolerance: float | None = None,
) -> SectionAnimationTrace:
    """Compute exact frames and preserve branch identity within topology epochs.

    Callers choose the sample times.  The function does not insert hidden
    samples, so a critical plane position that must appear as a distinct
    parabola or degenerate conic must be included explicitly by the authoring
    layer.
    """

    identity = _identity(section_id, "section_id")
    authored = tuple(samples)
    if not authored:
        raise SectionAnimationError("animation requires at least one sample")
    if not all(isinstance(item, SectionAnimationSample) for item in authored):
        raise TypeError("samples must contain SectionAnimationSample objects")
    if any(right.time <= left.time for left, right in zip(authored, authored[1:])):
        raise SectionAnimationError("sample times must increase strictly")
    surface_id = authored[0].surface.surface_id
    plane_id = authored[0].plane.plane_id
    if any(item.surface.surface_id != surface_id for item in authored):
        raise SectionAnimationError("time-varying surface must retain its surface_id")
    if any(item.plane.plane_id != plane_id for item in authored):
        raise SectionAnimationError("time-varying plane must retain its plane_id")

    frames: list[TrackedSectionFrame] = []
    events: list[TopologyEvent] = []
    epoch = 0
    for index, sample in enumerate(authored):
        trace = compute_quadric_section(
            identity,
            sample.surface,
            sample.plane,
            context=context,
            coefficient_tolerance=coefficient_tolerance,
        )
        signature = SectionTopologySignature.from_trace(trace)
        if len(trace.components) > MAX_SECTION_BRANCH_SLOTS:
            raise SectionAnimationError(
                "finite quadric section exceeds the fixed two-branch render capacity"
            )
        if not frames:
            mappings = _initial_mappings(identity, epoch, trace)
        else:
            previous = frames[-1]
            if previous.signature.topologically_equivalent(signature):
                mappings = _continued_mappings(previous, trace)
            else:
                event = TopologyEvent(
                    event_id=f"{identity}:topology-event:{len(events):04d}",
                    left_frame_index=index - 1,
                    right_frame_index=index,
                    left_time=previous.time,
                    right_time=sample.time,
                    before=previous.signature,
                    after=signature,
                    reasons=_event_reasons(previous.signature, signature),
                )
                events.append(event)
                epoch += 1
                mappings = _initial_mappings(identity, epoch, trace)
        frames.append(
            TrackedSectionFrame(
                frame_index=index,
                time=sample.time,
                topology_epoch=epoch,
                signature=signature,
                section=trace,
                branches=mappings,
            )
        )

    required = max((len(frame.branches) for frame in frames), default=0)
    capacity = BranchCapacityPlan(
        slot_ids=(f"{identity}:slot:0", f"{identity}:slot:1"),
        required_slots=required,
    )
    return SectionAnimationTrace(
        section_id=identity,
        surface_id=surface_id,
        plane_id=plane_id,
        frames=tuple(frames),
        topology_events=tuple(events),
        capacity_plan=capacity,
    )


def match_tracked_section_frame(
    reference: TrackedSectionFrame,
    section: QuadricSectionTrace,
    *,
    frame_index: int | None = None,
    time: float | None = None,
) -> TrackedSectionFrame:
    """Match one exact section against a proven same-topology reference.

    The renderer-facing transition layer frequently evaluates a section at a
    progress value that was not one of the authored schedule samples.  It must
    still use the schedule's stable component slots rather than trusting the
    incidental order of freshly classified branches.  This helper exposes the
    same analytic correspondence used by :func:`track_quadric_section_animation`
    without recomputing either section.

    A topology change is deliberately rejected here.  The caller must select
    a reference frame from the correct topology epoch (or use an explicit
    transition between two epochs).
    """

    if not isinstance(reference, TrackedSectionFrame):
        raise TypeError("reference must be a TrackedSectionFrame")
    if not isinstance(section, QuadricSectionTrace):
        raise TypeError("section must be a QuadricSectionTrace")
    signature = SectionTopologySignature.from_trace(section)
    if not reference.signature.topologically_equivalent(signature):
        raise BranchContinuityError(
            "section topology differs from the selected reference frame"
        )
    resolved_index = reference.frame_index if frame_index is None else frame_index
    resolved_time = reference.time if time is None else time
    if isinstance(resolved_index, bool) or not isinstance(resolved_index, int):
        raise SectionAnimationError("frame_index must be a non-negative integer")
    if resolved_index < 0:
        raise SectionAnimationError("frame_index must be a non-negative integer")
    return TrackedSectionFrame(
        frame_index=resolved_index,
        time=_finite(resolved_time, "frame time"),
        topology_epoch=reference.topology_epoch,
        signature=signature,
        section=section,
        branches=_continued_mappings(reference, section),
    )


def _materialize_tracked_section_curves(
    frame: TrackedSectionFrame,
    curve_id: Callable[[TrackedSectionBranch, int], str],
    *,
    max_intervals_per_component: int = 2,
) -> tuple[ParametricConicBranch, ...]:
    """Adapt tracked capacity slots to renderer-facing analytic curve IDs.

    Raw conic labels and periodic seam splits are representation details, not
    persistent renderer capacity identities.  Both the scheduled transition
    controller and the high-level fixed-topology rig use this one adapter so a
    circle/ellipse label change, a hyperbola label flip, or a one/two-interval
    seam crossing cannot allocate a new slot during playback.
    """

    if not isinstance(frame, TrackedSectionFrame):
        raise TypeError("frame must be a TrackedSectionFrame")
    if not callable(curve_id):
        raise TypeError("curve_id must be callable")
    if (
        isinstance(max_intervals_per_component, bool)
        or not isinstance(max_intervals_per_component, int)
        or max_intervals_per_component <= 0
    ):
        raise SectionAnimationError(
            "max_intervals_per_component must be a positive integer"
        )
    result: list[ParametricConicBranch] = []
    for mapping in frame.branches:
        component = frame.section.component_map[mapping.source_component_id]
        branch = frame.section.branch_map[mapping.source_branch_id]
        if len(component.parameter_intervals) > max_intervals_per_component:
            raise SectionAnimationError(
                "one tracked section component exceeds its fixed interval capacity"
            )
        for interval_index, interval in enumerate(component.parameter_intervals):
            result.append(
                ParametricConicBranch(
                    _identity(curve_id(mapping, interval_index), "curve_id"),
                    branch.parameterization,
                    branch.plane_embedding,
                    interval,
                )
            )
    identities = tuple(item.curve_id for item in result)
    if len(set(identities)) != len(identities):
        raise SectionAnimationError(
            "tracked section curve adapter produced duplicate identities"
        )
    return tuple(sorted(result, key=lambda item: item.curve_id))


def canonical_quadric_section_animation_json(trace: SectionAnimationTrace) -> str:
    """Return deterministic, strict JSON for caches and regression fixtures."""

    if not isinstance(trace, SectionAnimationTrace):
        raise TypeError("trace must be a SectionAnimationTrace")
    return json.dumps(
        trace.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class PointTrackSelection:
    """Authored selection used initially or after one topology transition."""

    capacity_slot: int
    fraction: float
    mode: PointParameterMode = PointParameterMode.NORMALIZED_PARAMETER

    def __post_init__(self) -> None:
        if self.capacity_slot not in {0, 1}:
            raise MovingPointContinuityError("point capacity_slot must be 0 or 1")
        object.__setattr__(self, "fraction", _fraction(self.fraction, "point fraction"))
        object.__setattr__(self, "mode", PointParameterMode(self.mode))

    def to_dict(self) -> dict[str, object]:
        return {
            "capacitySlot": self.capacity_slot,
            "fraction": self.fraction,
            "mode": self.mode.value,
        }


@dataclass(frozen=True, slots=True)
class MovingPointSample:
    """One resolved analytic point on one tracked finite component."""

    frame_index: int
    time: float
    stable_branch_id: str
    source_parameter: float
    world_point: tuple[float, float, float]
    selection: PointTrackSelection

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise MovingPointContinuityError("point frame index must be non-negative")
        object.__setattr__(self, "time", _finite(self.time, "point sample time"))
        object.__setattr__(
            self, "stable_branch_id", _identity(self.stable_branch_id, "stable_branch_id")
        )
        object.__setattr__(
            self,
            "source_parameter",
            _finite(self.source_parameter, "point source parameter"),
        )
        object.__setattr__(self, "world_point", _tuple3(self.world_point))

    def to_dict(self) -> dict[str, object]:
        return {
            "frameIndex": self.frame_index,
            "time": self.time,
            "stableBranchId": self.stable_branch_id,
            "sourceParameter": self.source_parameter,
            "worldPoint": list(self.world_point),
            "selection": self.selection.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PointTransitionContext:
    """Information supplied to an explicit cross-topology remapping rule."""

    event: TopologyEvent
    previous_sample: MovingPointSample
    current_frame: TrackedSectionFrame


PointAuxiliaryRule = Callable[[PointTransitionContext], PointTrackSelection]


@dataclass(frozen=True, slots=True)
class MovingPointTrace:
    """Serializable moving-point result derived from one section animation."""

    samples: tuple[MovingPointSample, ...]
    crossed_topology_events: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise MovingPointContinuityError("moving point trace requires samples")
        if tuple(item.frame_index for item in self.samples) != tuple(range(len(self.samples))):
            raise MovingPointContinuityError("moving point samples must cover every frame")

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": [item.to_dict() for item in self.samples],
            "crossedTopologyEvents": list(self.crossed_topology_events),
        }


def track_moving_section_point(
    animation: SectionAnimationTrace,
    selection: PointTrackSelection,
    *,
    auxiliary_rule: PointAuxiliaryRule | None = None,
) -> MovingPointTrace:
    """Keep one point continuous inside epochs and fail closed across events.

    An ``auxiliary_rule`` is called only at a topology event.  It must
    explicitly choose a slot and parameter/arc-length fraction on the new
    topology.  The kernel never guesses which branch of a hyperbola should
    inherit a point from an ellipse or parabola.
    """

    if not isinstance(animation, SectionAnimationTrace):
        raise TypeError("animation must be a SectionAnimationTrace")
    if not isinstance(selection, PointTrackSelection):
        raise TypeError("selection must be a PointTrackSelection")
    current_selection = selection
    samples: list[MovingPointSample] = []
    crossed: list[str] = []
    events = animation.event_by_right_frame
    for frame in animation.frames:
        event = events.get(frame.frame_index)
        if event is not None:
            if auxiliary_rule is None:
                raise MovingPointContinuityError(
                    f"moving point crossed {event.event_id} without an auxiliary rule"
                )
            directive = auxiliary_rule(
                PointTransitionContext(event, samples[-1], frame)
            )
            if not isinstance(directive, PointTrackSelection):
                raise MovingPointContinuityError(
                    "auxiliary rule must return PointTrackSelection"
                )
            current_selection = directive
            crossed.append(event.event_id)
        try:
            tracked = frame.slot_map[current_selection.capacity_slot]
        except KeyError as exc:
            raise MovingPointContinuityError(
                "selected branch slot is empty in the current topology"
            ) from exc
        parameter = frame.source_parameter(
            tracked.stable_branch_id,
            current_selection.fraction,
            mode=current_selection.mode,
        )
        branch = frame.section.branch_map[tracked.source_branch_id]
        samples.append(
            MovingPointSample(
                frame_index=frame.frame_index,
                time=frame.time,
                stable_branch_id=tracked.stable_branch_id,
                source_parameter=parameter,
                world_point=_tuple3(branch.world_point(parameter)),
                selection=current_selection,
            )
        )
    return MovingPointTrace(tuple(samples), tuple(crossed))


__all__ = [
    "BranchCapacityPlan",
    "BranchContinuityError",
    "MAX_SECTION_BRANCH_SLOTS",
    "MovingPointContinuityError",
    "MovingPointSample",
    "MovingPointTrace",
    "PointAuxiliaryRule",
    "PointParameterMode",
    "PointTrackSelection",
    "PointTransitionContext",
    "QUADRIC_SECTION_ANIMATION_SCHEMA",
    "SectionAnimationError",
    "SectionAnimationSample",
    "SectionAnimationTrace",
    "SectionConicFamily",
    "SectionTopologySignature",
    "TopologyEvent",
    "TopologyEventKind",
    "TrackedSectionBranch",
    "TrackedSectionFrame",
    "canonical_quadric_section_animation_json",
    "match_tracked_section_frame",
    "track_moving_section_point",
    "track_quadric_section_animation",
]
