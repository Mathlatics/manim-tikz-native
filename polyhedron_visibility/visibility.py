"""Visibility contracts built on geometry and topology primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Generic, Hashable, Iterable, TypeVar

from .geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from .topology import (
    ParameterInterval,
    TaggedInterval,
    coalesce_tagged_intervals,
    partition_parameter_domain,
)


OccluderT = TypeVar("OccluderT", bound=Hashable)


class VisibilityKind(str, Enum):
    VISIBLE = "visible"
    HIDDEN = "hidden"


def _require_hashable(name: str, value: object) -> None:
    if value is None:
        raise ValueError(f"{name} must not be None")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be hashable") from exc


@dataclass(frozen=True, slots=True)
class VisibilitySpan(Generic[OccluderT]):
    """One valid visible or hidden parameter span."""

    interval: ParameterInterval
    kind: VisibilityKind
    occluders: tuple[OccluderT, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.interval, ParameterInterval):
            raise TypeError("interval must be a ParameterInterval")
        if not isinstance(self.kind, VisibilityKind):
            raise TypeError("kind must be a VisibilityKind")
        if not isinstance(self.occluders, tuple):
            raise TypeError("occluders must be a tuple")
        seen: set[OccluderT] = set()
        for index, occluder in enumerate(self.occluders):
            _require_hashable(f"occluders[{index}]", occluder)
            if occluder in seen:
                raise ValueError("occluders must not contain duplicates")
            seen.add(occluder)
        if self.kind is VisibilityKind.VISIBLE and self.occluders:
            raise ValueError("visible spans must not name an occluder")
        if self.kind is VisibilityKind.HIDDEN and not self.occluders:
            raise ValueError("hidden spans must name at least one occluder")

    @property
    def visible(self) -> bool:
        return self.kind is VisibilityKind.VISIBLE


@dataclass(frozen=True, slots=True)
class OcclusionInterval(Generic[OccluderT]):
    """A hidden interval attributed to one semantic occluder."""

    interval: ParameterInterval
    occluder: OccluderT

    def __post_init__(self) -> None:
        if not isinstance(self.interval, ParameterInterval):
            raise TypeError("interval must be a ParameterInterval")
        _require_hashable("occluder", self.occluder)


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def partition_visibility(
    domain: ParameterInterval,
    hidden: Iterable[OcclusionInterval[OccluderT]],
    *,
    context: GeometryContext | ResolvedGeometryContext | None = None,
    parameter_tolerance: float | None = None,
    occluder_key: Callable[[OccluderT], object] | None = None,
) -> tuple[VisibilitySpan[OccluderT], ...]:
    """Classify a parameter domain into stable visible and hidden spans.

    Occluders are kept in first-authored order.  When ``occluder_key`` is
    supplied, it is the primary key and first-authored order is the explicit
    secondary key.  Equal keys therefore never fall back to set/hash order.
    """

    if not isinstance(domain, ParameterInterval):
        raise TypeError("domain must be a ParameterInterval")
    resolved_context = resolve_geometry_context(context)
    epsilon = (
        resolved_context.epsilon(GeometryQuantity.PARAMETER)
        if parameter_tolerance is None
        else _finite_nonnegative("parameter_tolerance", parameter_tolerance)
    )

    clipped: list[OcclusionInterval[OccluderT]] = []
    breakpoints: list[float] = []
    first_seen: dict[OccluderT, int] = {}
    for candidate in hidden:
        if not isinstance(candidate, OcclusionInterval):
            raise TypeError("hidden entries must be OcclusionInterval objects")
        interval = candidate.interval.intersection(domain, tolerance=epsilon)
        if interval is None or interval.length <= epsilon:
            continue
        first_seen.setdefault(candidate.occluder, len(first_seen))
        clipped.append(OcclusionInterval(interval, candidate.occluder))
        breakpoints.extend((interval.start, interval.end))

    cells = partition_parameter_domain(domain, breakpoints, tolerance=epsilon)
    tagged: list[
        TaggedInterval[tuple[VisibilityKind, tuple[OccluderT, ...]]]
    ] = []

    for cell in cells:
        active: list[OccluderT] = []
        active_seen: set[OccluderT] = set()
        for candidate in clipped:
            owner = candidate.occluder
            if owner in active_seen or not candidate.interval.contains(cell.midpoint):
                continue
            active_seen.add(owner)
            active.append(owner)

        if active:
            if occluder_key is not None:
                try:
                    active.sort(
                        key=lambda owner: (
                            occluder_key(owner),
                            first_seen[owner],
                        )
                    )
                except TypeError as exc:
                    raise ValueError(
                        "occluder_key must return mutually orderable values"
                    ) from exc
            owners = tuple(active)
            tag = (VisibilityKind.HIDDEN, owners)
        else:
            tag = (VisibilityKind.VISIBLE, ())
        tagged.append(TaggedInterval(cell, tag))

    merged = coalesce_tagged_intervals(tagged, tolerance=epsilon)
    return tuple(
        VisibilitySpan(span.interval, span.tag[0], span.tag[1])
        for span in merged
    )


def visible_intervals(
    spans: Iterable[VisibilitySpan[OccluderT]],
) -> tuple[ParameterInterval, ...]:
    return tuple(span.interval for span in spans if span.visible)


def hidden_intervals(
    spans: Iterable[VisibilitySpan[OccluderT]],
) -> tuple[ParameterInterval, ...]:
    return tuple(span.interval for span in spans if not span.visible)
