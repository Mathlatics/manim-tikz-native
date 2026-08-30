"""Renderer-neutral joint preflight for authored parallel-camera frames.

Rendering failures are expensive when a scene combines camera framing,
topology handoffs, fixed-capacity slots, and painter ordering.  This module
accepts immutable evidence from those independent compilers and checks it as
one sequence before any Scene or Mobject is created.

The preflight does not solve geometry and does not guess missing evidence.  It
reports every certifiable problem, produces deterministic JSON evidence, and
fails closed when ``require_accepted()`` is called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Sequence

import numpy as np

from .parallel_camera import ParallelCameraState


PARALLEL_PREFLIGHT_REPORT_SCHEMA = "parallel-scene-preflight-report/v1"


class ParallelPreflightError(ValueError):
    """Preflight evidence is malformed and cannot be audited."""


class ParallelPreflightRejectedError(ParallelPreflightError):
    """A well-formed preflight report contains one or more errors."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParallelPreflightError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ParallelPreflightError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelPreflightError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise ParallelPreflightError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ParallelPreflightError(f"{label} must be positive")
    return result


def _non_negative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ParallelPreflightError(f"{label} must be non-negative")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParallelPreflightError(f"{label} must be an integer")
    return value


def _point3(value: object, label: str) -> tuple[float, float, float]:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelPreflightError(
            f"{label} must contain three finite values"
        ) from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ParallelPreflightError(
            f"{label} must contain three finite values"
        )
    return tuple(float(item) for item in point)  # type: ignore[return-value]


def _point2(value: object, label: str) -> tuple[float, float]:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelPreflightError(
            f"{label} must contain two finite values"
        ) from exc
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ParallelPreflightError(
            f"{label} must contain two finite values"
        )
    return tuple(float(item) for item in point)  # type: ignore[return-value]


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ParallelPreflightError(
            f"preflight evidence is not canonical JSON: {exc}"
        ) from exc


class ParallelPreflightSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ParallelSafeFrame:
    """Allowed final screen-coordinate rectangle for authored framing points."""

    left: float
    right: float
    bottom: float
    top: float

    def __post_init__(self) -> None:
        left = _finite(self.left, "safe frame left")
        right = _finite(self.right, "safe frame right")
        bottom = _finite(self.bottom, "safe frame bottom")
        top = _finite(self.top, "safe frame top")
        if not left < right or not bottom < top:
            raise ParallelPreflightError(
                "safe frame must have positive width and height"
            )
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "bottom", bottom)
        object.__setattr__(self, "top", top)

    def to_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "right": self.right,
            "bottom": self.bottom,
            "top": self.top,
        }


@dataclass(frozen=True, slots=True)
class ParallelPreflightLimits:
    """Author-approved framing and zoom limits for one sequence."""

    safe_frame: ParallelSafeFrame
    min_zoom: float
    max_zoom: float
    tolerance: float = 1.0e-9
    require_framing_points: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.safe_frame, ParallelSafeFrame):
            raise TypeError("safe_frame must be a ParallelSafeFrame")
        minimum = _positive(self.min_zoom, "min_zoom")
        maximum = _positive(self.max_zoom, "max_zoom")
        if minimum > maximum:
            raise ParallelPreflightError("min_zoom must not exceed max_zoom")
        tolerance = _non_negative(self.tolerance, "tolerance")
        if not isinstance(self.require_framing_points, bool):
            raise TypeError("require_framing_points must be a bool")
        object.__setattr__(self, "min_zoom", minimum)
        object.__setattr__(self, "max_zoom", maximum)
        object.__setattr__(self, "tolerance", tolerance)

    def to_dict(self) -> dict[str, object]:
        return {
            "safeFrame": self.safe_frame.to_dict(),
            "minZoom": self.min_zoom,
            "maxZoom": self.max_zoom,
            "tolerance": self.tolerance,
            "requireFramingPoints": self.require_framing_points,
        }


@dataclass(frozen=True, slots=True)
class ParallelScreenTransform:
    """Renderer-level affine terms outside ``ParallelCameraState``.

    ``MultiProjectionCamera`` keeps one inherited Manim zoom and frame center
    for legacy compatibility, while individual fixed-frame controllers may
    add a display offset.  The semantic camera anchor itself is not scaled.
    """

    inherited_zoom: float = 1.0
    frame_center: tuple[float, float] = (0.0, 0.0)
    display_offset: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        zoom = _positive(self.inherited_zoom, "inherited_zoom")
        center = _point2(self.frame_center, "frame_center")
        offset = _point2(self.display_offset, "display_offset")
        object.__setattr__(self, "inherited_zoom", zoom)
        object.__setattr__(self, "frame_center", center)
        object.__setattr__(self, "display_offset", offset)

    def apply(
        self,
        projected: np.ndarray,
        screen_anchor: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(projected, dtype=float)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ParallelPreflightError(
                "projected points must be a two-dimensional point array"
            )
        anchor = np.asarray(screen_anchor, dtype=float)
        if anchor.shape != (2,) or not np.all(np.isfinite(anchor)):
            raise ParallelPreflightError(
                "screen_anchor must contain two finite values"
            )
        return (
            anchor
            + self.inherited_zoom * (values[:, :2] - anchor)
            + np.asarray(self.frame_center)
            + np.asarray(self.display_offset)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "inheritedZoom": self.inherited_zoom,
            "frameCenter": list(self.frame_center),
            "displayOffset": list(self.display_offset),
        }


@dataclass(frozen=True, slots=True)
class TopologyEventEvidence:
    """One analytic timeline event and its fixed-bank certification."""

    event_id: str
    kind: str
    certified: bool
    requires_slot_bank: bool = False
    slot_bank_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identity(self.event_id, "event_id"))
        object.__setattr__(self, "kind", _identity(self.kind, "event kind"))
        if not isinstance(self.certified, bool):
            raise TypeError("certified must be a bool")
        if not isinstance(self.requires_slot_bank, bool):
            raise TypeError("requires_slot_bank must be a bool")
        if self.slot_bank_id is not None:
            object.__setattr__(
                self,
                "slot_bank_id",
                _identity(self.slot_bank_id, "slot_bank_id"),
            )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "eventId": self.event_id,
            "kind": self.kind,
            "certified": self.certified,
            "requiresSlotBank": self.requires_slot_bank,
        }
        if self.slot_bank_id is not None:
            result["slotBankId"] = self.slot_bank_id
        return result


@dataclass(frozen=True, slots=True)
class CapacityEvidence:
    """Used and preallocated capacity for one fixed renderer resource."""

    resource_id: str
    used: int
    limit: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_id",
            _identity(self.resource_id, "resource_id"),
        )
        object.__setattr__(self, "used", _integer(self.used, "used capacity"))
        object.__setattr__(self, "limit", _integer(self.limit, "capacity limit"))

    def to_dict(self) -> dict[str, object]:
        return {
            "resourceId": self.resource_id,
            "used": self.used,
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True)
class PainterOrderEvidence:
    """Authored painter items, precedence relations, and proposed draw order."""

    item_ids: tuple[str, ...] = ()
    relations: tuple[tuple[str, str], ...] = ()
    draw_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        items = tuple(_identity(item, "painter item") for item in self.item_ids)
        order = tuple(_identity(item, "draw-order item") for item in self.draw_order)
        relations: list[tuple[str, str]] = []
        for index, relation in enumerate(self.relations):
            if not isinstance(relation, (tuple, list)) or len(relation) != 2:
                raise ParallelPreflightError(
                    f"painter relation {index} must contain two item ids"
                )
            relations.append(
                (
                    _identity(relation[0], "painter relation source"),
                    _identity(relation[1], "painter relation target"),
                )
            )
        object.__setattr__(self, "item_ids", items)
        object.__setattr__(self, "draw_order", order)
        object.__setattr__(self, "relations", tuple(relations))

    def to_dict(self) -> dict[str, object]:
        return {
            "itemIds": sorted(self.item_ids),
            "relations": [list(item) for item in sorted(self.relations)],
            "drawOrder": list(self.draw_order),
        }


@dataclass(frozen=True, slots=True)
class ParallelPreflightFrame:
    """All renderer-neutral evidence for one authored sequence time."""

    frame_id: str
    time: float
    camera: ParallelCameraState
    screen_transform: ParallelScreenTransform = field(
        default_factory=ParallelScreenTransform
    )
    framing_points: tuple[tuple[float, float, float], ...] = ()
    topology_events: tuple[TopologyEventEvidence, ...] = ()
    capacities: tuple[CapacityEvidence, ...] = ()
    painter_order: PainterOrderEvidence = field(default_factory=PainterOrderEvidence)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _identity(self.frame_id, "frame_id"))
        object.__setattr__(self, "time", _finite(self.time, "frame time"))
        if not isinstance(self.camera, ParallelCameraState):
            raise TypeError("camera must be a ParallelCameraState")
        if not isinstance(self.screen_transform, ParallelScreenTransform):
            raise TypeError("screen_transform must be a ParallelScreenTransform")
        points = tuple(
            sorted(
                _point3(item, f"framing_points[{index}]")
                for index, item in enumerate(self.framing_points)
            )
        )
        events = tuple(self.topology_events)
        if not all(isinstance(item, TopologyEventEvidence) for item in events):
            raise TypeError(
                "topology_events must contain TopologyEventEvidence values"
            )
        capacities = tuple(self.capacities)
        if not all(isinstance(item, CapacityEvidence) for item in capacities):
            raise TypeError("capacities must contain CapacityEvidence values")
        if not isinstance(self.painter_order, PainterOrderEvidence):
            raise TypeError("painter_order must be PainterOrderEvidence")
        object.__setattr__(self, "framing_points", points)
        object.__setattr__(
            self,
            "topology_events",
            tuple(sorted(events, key=lambda item: (item.event_id, item.kind))),
        )
        object.__setattr__(
            self,
            "capacities",
            tuple(sorted(capacities, key=lambda item: item.resource_id)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frameId": self.frame_id,
            "time": self.time,
            "camera": {
                "matrix": self.camera.matrix.tolist(),
                "target": self.camera.target.tolist(),
                "screenAnchor": self.camera.screen_anchor.tolist(),
                "zoom": self.camera.zoom,
            },
            "screenTransform": self.screen_transform.to_dict(),
            "framingPoints": [list(item) for item in self.framing_points],
            "topologyEvents": [item.to_dict() for item in self.topology_events],
            "capacities": [item.to_dict() for item in self.capacities],
            "painterOrder": self.painter_order.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ParallelPreflightIssue:
    severity: ParallelPreflightSeverity
    code: str
    message: str
    frame_id: str | None = None
    subject_id: str | None = None

    def __post_init__(self) -> None:
        try:
            severity = ParallelPreflightSeverity(self.severity)
        except (TypeError, ValueError) as exc:
            raise ParallelPreflightError(
                "severity must be a ParallelPreflightSeverity"
            ) from exc
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "code", _identity(self.code, "issue code"))
        object.__setattr__(self, "message", _identity(self.message, "issue message"))
        if self.frame_id is not None:
            object.__setattr__(
                self,
                "frame_id",
                _identity(self.frame_id, "frame_id"),
            )
        if self.subject_id is not None:
            object.__setattr__(
                self,
                "subject_id",
                _identity(self.subject_id, "subject_id"),
            )

    def sort_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.frame_id or "",
            self.severity.value,
            self.code,
            self.subject_id or "",
            self.message,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
        }
        if self.frame_id is not None:
            result["frameId"] = self.frame_id
        if self.subject_id is not None:
            result["subjectId"] = self.subject_id
        return result


@dataclass(frozen=True, slots=True)
class ParallelPreflightReport:
    """Deterministic audit evidence for a complete authored sequence."""

    input_digest: str
    frame_count: int
    framing_point_count: int
    topology_event_count: int
    capacity_count: int
    painter_relation_count: int
    issues: tuple[ParallelPreflightIssue, ...]

    @property
    def accepted(self) -> bool:
        return not any(
            item.severity is ParallelPreflightSeverity.ERROR
            for item in self.issues
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PARALLEL_PREFLIGHT_REPORT_SCHEMA,
            "accepted": self.accepted,
            "inputDigest": self.input_digest,
            "counts": {
                "frames": self.frame_count,
                "framingPoints": self.framing_point_count,
                "topologyEvents": self.topology_event_count,
                "capacities": self.capacity_count,
                "painterRelations": self.painter_relation_count,
                "errors": sum(
                    item.severity is ParallelPreflightSeverity.ERROR
                    for item in self.issues
                ),
                "warnings": sum(
                    item.severity is ParallelPreflightSeverity.WARNING
                    for item in self.issues
                ),
            },
            "issues": [item.to_dict() for item in self.issues],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def require_accepted(self) -> "ParallelPreflightReport":
        if self.accepted:
            return self
        errors = tuple(
            item for item in self.issues
            if item.severity is ParallelPreflightSeverity.ERROR
        )
        preview = "; ".join(
            f"{item.frame_id or 'sequence'}:{item.code}"
            for item in errors[:5]
        )
        suffix = "" if len(errors) <= 5 else f"; +{len(errors) - 5} more"
        raise ParallelPreflightRejectedError(
            f"parallel scene preflight rejected {len(errors)} error(s): "
            f"{preview}{suffix}"
        )


def _issue(
    issues: list[ParallelPreflightIssue],
    code: str,
    message: str,
    *,
    frame: ParallelPreflightFrame | None = None,
    subject_id: str | None = None,
) -> None:
    issues.append(
        ParallelPreflightIssue(
            ParallelPreflightSeverity.ERROR,
            code,
            message,
            None if frame is None else frame.frame_id,
            subject_id,
        )
    )


def _painter_has_cycle(
    items: set[str],
    relations: Sequence[tuple[str, str]],
) -> bool:
    adjacency: dict[str, set[str]] = {item: set() for item in items}
    indegree = {item: 0 for item in items}
    for source, target in relations:
        if source not in items or target not in items or source == target:
            continue
        if target in adjacency[source]:
            continue
        adjacency[source].add(target)
        indegree[target] += 1
    ready = sorted(item for item, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        source = ready.pop(0)
        visited += 1
        for target in sorted(adjacency[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    return visited != len(items)


def _check_painter_order(
    frame: ParallelPreflightFrame,
    issues: list[ParallelPreflightIssue],
) -> None:
    evidence = frame.painter_order
    items = set(evidence.item_ids)
    order = set(evidence.draw_order)
    if len(items) != len(evidence.item_ids):
        _issue(
            issues,
            "duplicate-painter-item",
            "painter item_ids contain duplicates",
            frame=frame,
        )
    if len(order) != len(evidence.draw_order):
        _issue(
            issues,
            "duplicate-draw-order-item",
            "draw_order contains duplicate item ids",
            frame=frame,
        )
    missing = sorted(items - order)
    extra = sorted(order - items)
    if missing or extra:
        _issue(
            issues,
            "draw-order-item-mismatch",
            f"draw_order missing={missing!r}, extra={extra!r}",
            frame=frame,
        )
    if len(set(evidence.relations)) != len(evidence.relations):
        _issue(
            issues,
            "duplicate-painter-relation",
            "painter relations contain duplicates",
            frame=frame,
        )
    valid_relations = []
    for source, target in evidence.relations:
        relation_id = f"{source}->{target}"
        if source == target:
            _issue(
                issues,
                "self-painter-relation",
                "a painter item cannot precede itself",
                frame=frame,
                subject_id=relation_id,
            )
            continue
        unknown = sorted({source, target} - items)
        if unknown:
            _issue(
                issues,
                "unknown-painter-relation-item",
                f"painter relation references unknown items {unknown!r}",
                frame=frame,
                subject_id=relation_id,
            )
            continue
        valid_relations.append((source, target))
    if _painter_has_cycle(items, valid_relations):
        _issue(
            issues,
            "painter-relation-cycle",
            "painter precedence relations contain a cycle",
            frame=frame,
        )
    if not missing and not extra and len(order) == len(evidence.draw_order):
        positions = {
            item: index for index, item in enumerate(evidence.draw_order)
        }
        for source, target in valid_relations:
            if positions[source] >= positions[target]:
                _issue(
                    issues,
                    "draw-order-violates-relation",
                    "proposed draw_order violates painter precedence",
                    frame=frame,
                    subject_id=f"{source}->{target}",
                )


def preflight_parallel_frames(
    frames: Sequence[ParallelPreflightFrame],
    limits: ParallelPreflightLimits,
) -> ParallelPreflightReport:
    """Audit a complete sequence before any renderer state is mutated."""

    if not isinstance(limits, ParallelPreflightLimits):
        raise TypeError("limits must be ParallelPreflightLimits")
    authored = tuple(frames)
    if not all(isinstance(item, ParallelPreflightFrame) for item in authored):
        raise TypeError("frames must contain ParallelPreflightFrame values")
    issues: list[ParallelPreflightIssue] = []
    if not authored:
        _issue(
            issues,
            "empty-sequence",
            "preflight requires at least one authored frame",
        )
    seen_frame_ids: set[str] = set()
    previous_time: float | None = None
    seen_event_ids: set[str] = set()
    for frame in authored:
        if frame.frame_id in seen_frame_ids:
            _issue(
                issues,
                "duplicate-frame-id",
                "frame ids must be unique across the sequence",
                frame=frame,
            )
        seen_frame_ids.add(frame.frame_id)
        if previous_time is not None and frame.time <= previous_time:
            _issue(
                issues,
                "frame-time-not-increasing",
                "frame times must be strictly increasing",
                frame=frame,
            )
        previous_time = frame.time

        tolerance = limits.tolerance
        effective_zoom = (
            frame.camera.zoom * frame.screen_transform.inherited_zoom
        )
        if effective_zoom < limits.min_zoom - tolerance:
            _issue(
                issues,
                "zoom-below-minimum",
                f"effective zoom {effective_zoom:.12g} is below "
                f"{limits.min_zoom:.12g}",
                frame=frame,
            )
        if effective_zoom > limits.max_zoom + tolerance:
            _issue(
                issues,
                "zoom-above-maximum",
                f"effective zoom {effective_zoom:.12g} exceeds "
                f"{limits.max_zoom:.12g}",
                frame=frame,
            )
        if not frame.framing_points and limits.require_framing_points:
            _issue(
                issues,
                "missing-framing-points",
                "frame has no points proving the authored safe framing",
                frame=frame,
            )
        if frame.framing_points:
            projected = frame.camera.project_points(frame.framing_points)
            screen_points = frame.screen_transform.apply(
                projected,
                frame.camera.screen_anchor,
            )
            outside = (
                (screen_points[:, 0] < limits.safe_frame.left - tolerance)
                | (screen_points[:, 0] > limits.safe_frame.right + tolerance)
                | (screen_points[:, 1] < limits.safe_frame.bottom - tolerance)
                | (screen_points[:, 1] > limits.safe_frame.top + tolerance)
            )
            count = int(np.count_nonzero(outside))
            if count:
                _issue(
                    issues,
                    "safe-frame-overflow",
                    f"{count} framing point(s) project outside the safe frame",
                    frame=frame,
                )

        for event in frame.topology_events:
            if event.event_id in seen_event_ids:
                _issue(
                    issues,
                    "duplicate-topology-event",
                    "topology event ids must be unique across the sequence",
                    frame=frame,
                    subject_id=event.event_id,
                )
            seen_event_ids.add(event.event_id)
            if not event.certified:
                _issue(
                    issues,
                    "uncertified-topology-event",
                    f"topology event {event.kind!r} lacks analytic certification",
                    frame=frame,
                    subject_id=event.event_id,
                )
            if event.requires_slot_bank and event.slot_bank_id is None:
                _issue(
                    issues,
                    "missing-topology-slot-bank",
                    "topology event requires a preallocated slot bank",
                    frame=frame,
                    subject_id=event.event_id,
                )

        seen_resources: set[str] = set()
        for capacity in frame.capacities:
            if capacity.resource_id in seen_resources:
                _issue(
                    issues,
                    "duplicate-capacity-resource",
                    "capacity resource ids must be unique within one frame",
                    frame=frame,
                    subject_id=capacity.resource_id,
                )
            seen_resources.add(capacity.resource_id)
            if capacity.used < 0 or capacity.limit < 0:
                _issue(
                    issues,
                    "negative-capacity",
                    "capacity used and limit must be non-negative",
                    frame=frame,
                    subject_id=capacity.resource_id,
                )
            elif capacity.used > capacity.limit:
                _issue(
                    issues,
                    "capacity-overflow",
                    f"capacity uses {capacity.used} of {capacity.limit}",
                    frame=frame,
                    subject_id=capacity.resource_id,
                )
        _check_painter_order(frame, issues)

    input_value = {
        "limits": limits.to_dict(),
        "frames": [item.to_dict() for item in authored],
    }
    input_digest = "sha256:" + hashlib.sha256(
        _canonical_json(input_value).encode("utf-8")
    ).hexdigest()
    return ParallelPreflightReport(
        input_digest=input_digest,
        frame_count=len(authored),
        framing_point_count=sum(len(item.framing_points) for item in authored),
        topology_event_count=sum(len(item.topology_events) for item in authored),
        capacity_count=sum(len(item.capacities) for item in authored),
        painter_relation_count=sum(
            len(item.painter_order.relations) for item in authored
        ),
        issues=tuple(sorted(issues, key=ParallelPreflightIssue.sort_key)),
    )


__all__ = [
    "PARALLEL_PREFLIGHT_REPORT_SCHEMA",
    "CapacityEvidence",
    "PainterOrderEvidence",
    "ParallelPreflightError",
    "ParallelPreflightFrame",
    "ParallelPreflightIssue",
    "ParallelPreflightLimits",
    "ParallelPreflightRejectedError",
    "ParallelPreflightReport",
    "ParallelPreflightSeverity",
    "ParallelSafeFrame",
    "ParallelScreenTransform",
    "TopologyEventEvidence",
    "preflight_parallel_frames",
]
