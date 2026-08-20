"""Visibility-layer contracts built on geometry and topology primitives.

This layer answers *which parameter spans are visible* and *which occluder
owns a hidden span*. It does not decide Manim draw order or construct any
renderer object; those responsibilities belong to the compositor and binding
layers respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Generic, Hashable, Iterable, TypeVar

from .geometry import GeometryContext, GeometryQuantity, resolve_geometry_context
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


@dataclass(frozen=True, slots=True)
class VisibilitySpan(Generic[OccluderT]):
    """One stable visible or hidden parameter span."""

    interval: ParameterInterval
    kind: VisibilityKind
    occluders: tuple[OccluderT, ...] = ()

    def __post_init__(self) -> None:
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


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def partition_visibility(
    domain: ParameterInterval,
    hidden: Iterable[OcclusionInterval[OccluderT]],
    *,
    context: GeometryContext | None = None,
    parameter_tolerance: float | None = None,
    occluder_key: Callable[[OccluderT], object] | None = None,
) -> tuple[VisibilitySpan[OccluderT], ...]:
    """Classify a parameter domain into stable visible and hidden spans.

    Boundary-only contact is discarded because it has no positive parameter
    length. Adjacent cells are merged only when both their visibility state
    and semantic occluder identity agree. This prevents a handoff between two
    different occluders from being mistaken for one animation slot, while two
    adjacent intervals owned by the same occluder remain a single slot.
    """

    resolved_context = resolve_geometry_context(context)
    if parameter_tolerance is None:
        epsilon = resolved_context.epsilon(
            GeometryQuantity.PARAMETER,
            scale=max(1.0, domain.length),
        )
    else:
        epsilon = _finite_nonnegative(
            "parameter_tolerance",
            parameter_tolerance,
        )

    clipped: list[OcclusionInterval[OccluderT]] = []
    breakpoints: list[float] = []
    for candidate in hidden:
        interval = candidate.interval.intersection(domain, tolerance=epsilon)
        if interval is None or interval.length <= epsilon:
            continue
        clipped.append(OcclusionInterval(interval, candidate.occluder))
        breakpoints.extend((interval.start, interval.end))

    cells = partition_parameter_domain(
        domain,
        breakpoints,
        tolerance=epsilon,
    )

    if occluder_key is None:
        occluder_key = lambda value: repr(value)

    tagged: list[
        TaggedInterval[tuple[VisibilityKind, tuple[OccluderT, ...]]]
    ] = []
    for cell in cells:
        midpoint = cell.midpoint
        active = {
            candidate.occluder
            for candidate in clipped
            if candidate.interval.contains(midpoint)
        }
        if active:
            owners = tuple(sorted(active, key=occluder_key))
            tag = (VisibilityKind.HIDDEN, owners)
        else:
            tag = (VisibilityKind.VISIBLE, ())
        tagged.append(TaggedInterval(cell, tag))

    merged = coalesce_tagged_intervals(tagged, tolerance=epsilon)
    return tuple(
        VisibilitySpan(
            span.interval,
            span.tag[0],
            span.tag[1],
        )
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
