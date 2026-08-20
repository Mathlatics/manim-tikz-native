"""Topology-layer primitives for stable parameter-domain partitioning.

The topology layer deliberately knows nothing about cameras, depth, Manim, or
stroke styles. It only preserves connectivity, interval identity, and exact
domain coverage while geometry values move through degenerate positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Generic, Iterable, TypeVar


TagT = TypeVar("TagT")


class BreakpointCluster(str, Enum):
    """Representative retained when nearby breakpoints form one cluster."""

    LOWER = "lower"
    UPPER = "upper"


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tolerance(value: float) -> float:
    result = _finite("tolerance", value)
    if result < 0.0:
        raise ValueError("tolerance must be non-negative")
    return result


@dataclass(frozen=True, order=True, slots=True)
class ParameterInterval:
    """Closed interval in one parameter domain."""

    start: float
    end: float

    def __post_init__(self) -> None:
        start = _finite("start", self.start)
        end = _finite("end", self.end)
        if end < start:
            raise ValueError("interval end must not precede interval start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.start + self.end)

    def contains(self, value: float, *, tolerance: float = 0.0) -> bool:
        epsilon = _tolerance(tolerance)
        scalar = _finite("value", value)
        return self.start - epsilon <= scalar <= self.end + epsilon

    def intersection(
        self,
        other: "ParameterInterval",
        *,
        tolerance: float = 0.0,
    ) -> "ParameterInterval | None":
        epsilon = _tolerance(tolerance)
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if end < start - epsilon:
            return None
        if end < start:
            end = start
        return ParameterInterval(start, end)

    def clamp(self, domain: "ParameterInterval") -> "ParameterInterval | None":
        return self.intersection(domain)


@dataclass(frozen=True, slots=True)
class TaggedInterval(Generic[TagT]):
    """A parameter interval whose identity must survive animation updates."""

    interval: ParameterInterval
    tag: TagT


def partition_parameter_domain(
    domain: ParameterInterval,
    breakpoints: Iterable[float],
    *,
    tolerance: float = 0.0,
    cluster: BreakpointCluster | str = BreakpointCluster.LOWER,
) -> tuple[ParameterInterval, ...]:
    """Split ``domain`` into deterministic consecutive cells.

    ``LOWER`` is the exact-domain default introduced with the shared kernel:
    near-duplicate interior breakpoints keep their lower representative and
    the first/last endpoints remain exactly the authored domain boundaries.

    ``UPPER`` retains the upper representative of each tolerance cluster.  It
    exists for frozen trace contracts whose historical interval splitter used
    that convention.  A tolerance-sized edge cell may then be returned; the
    visibility layer decides whether that cell is semantically paintable.
    """

    epsilon = _tolerance(tolerance)
    try:
        cluster_mode = BreakpointCluster(cluster)
    except ValueError as exc:
        raise ValueError("cluster must be 'lower' or 'upper'") from exc

    points = [domain.start, domain.end]
    for raw in breakpoints:
        value = _finite("breakpoint", raw)
        if value < domain.start - epsilon or value > domain.end + epsilon:
            continue
        points.append(min(domain.end, max(domain.start, value)))
    points.sort()

    if cluster_mode is BreakpointCluster.LOWER:
        unique = [domain.start]
        for value in points[1:-1]:
            if value - unique[-1] <= epsilon:
                continue
            if domain.end - value <= epsilon:
                continue
            unique.append(value)
        unique.append(domain.end)
    else:
        unique: list[float] = []
        for value in points:
            if not unique or value - unique[-1] > epsilon:
                unique.append(value)
            else:
                unique[-1] = max(unique[-1], value)
        if unique[0] > domain.start:
            unique.insert(0, domain.start)
        if unique[-1] < domain.end:
            unique.append(domain.end)

    if len(unique) == 2 and unique[0] == domain.start and unique[1] == domain.end:
        return (domain,)

    cells = tuple(
        ParameterInterval(start, end)
        for start, end in zip(unique, unique[1:])
        if end > start
    )
    return cells or (domain,)


def coalesce_tagged_intervals(
    spans: Iterable[TaggedInterval[TagT]],
    *,
    tolerance: float = 0.0,
) -> tuple[TaggedInterval[TagT], ...]:
    """Merge adjacent or overlapping spans only when their identity matches."""

    epsilon = _tolerance(tolerance)
    ordered = sorted(
        spans,
        key=lambda span: (span.interval.start, span.interval.end),
    )
    if not ordered:
        return ()

    merged: list[TaggedInterval[TagT]] = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        same_identity = previous.tag == span.tag
        touching = span.interval.start <= previous.interval.end + epsilon
        if same_identity and touching:
            merged[-1] = TaggedInterval(
                ParameterInterval(
                    previous.interval.start,
                    max(previous.interval.end, span.interval.end),
                ),
                previous.tag,
            )
        else:
            merged.append(span)
    return tuple(merged)


def assert_exact_partition(
    domain: ParameterInterval,
    cells: Iterable[ParameterInterval],
    *,
    tolerance: float = 0.0,
) -> tuple[ParameterInterval, ...]:
    """Validate that cells cover one domain without gaps or overlaps."""

    epsilon = _tolerance(tolerance)
    ordered = tuple(cells)
    if not ordered:
        raise ValueError("a partition must contain at least one cell")
    if abs(ordered[0].start - domain.start) > epsilon:
        raise ValueError("partition does not start at the domain boundary")
    if abs(ordered[-1].end - domain.end) > epsilon:
        raise ValueError("partition does not end at the domain boundary")
    for left, right in zip(ordered, ordered[1:]):
        if abs(left.end - right.start) > epsilon:
            raise ValueError("partition contains a gap or overlap")
    return ordered
