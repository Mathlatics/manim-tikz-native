"""Shared, renderer-independent geometry tolerance context.

The existing :class:`TolerancePolicy` remains the numerical source of truth.
This module does not reinterpret its coefficients.  Instead, one unresolved
``GeometryContext`` delegates to ``TolerancePolicy.resolve`` for the concrete
geometry being solved and returns an immutable ``ResolvedGeometryContext``
that can be passed through topology, visibility, and compositing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .contract import ResolvedTolerance, TolerancePolicy


PositionInput = Mapping[str, Sequence[float]] | Sequence[Sequence[float]]


class GeometryQuantity(str, Enum):
    """Numerical comparisons that must not silently share one unit."""

    LENGTH = "length"
    BOUNDARY = "boundary"
    DEPTH = "depth"
    PARAMETER = "parameter"
    ANGULAR = "angular"
    SCREEN = "screen"


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _normalize_overrides(
    overrides: Mapping[GeometryQuantity | str, float],
) -> Mapping[GeometryQuantity, float]:
    normalized: dict[GeometryQuantity, float] = {}
    for quantity, value in dict(overrides).items():
        resolved = GeometryQuantity(quantity)
        normalized[resolved] = _finite_nonnegative(
            f"override[{resolved.value}]",
            value,
        )
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ResolvedGeometryContext:
    """One concrete tolerance interpretation for one geometry solve.

    ``resolved`` is exactly the object produced by the legacy
    ``TolerancePolicy.resolve`` method.  Quantity lookup only names those
    already-resolved values; it never applies an additional scale floor or
    relative multiplier.
    """

    policy: TolerancePolicy
    resolved: ResolvedTolerance
    screen: float
    overrides: Mapping[GeometryQuantity | str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, TolerancePolicy):
            raise TypeError("policy must be a TolerancePolicy")
        if not isinstance(self.resolved, ResolvedTolerance):
            raise TypeError("resolved must be a ResolvedTolerance")
        for name in ("scale", "world", "parameter", "angular", "boundary", "depth"):
            _finite_nonnegative(f"resolved.{name}", getattr(self.resolved, name))
        object.__setattr__(self, "screen", _finite_nonnegative("screen", self.screen))
        object.__setattr__(self, "overrides", _normalize_overrides(self.overrides))

    def epsilon(self, quantity: GeometryQuantity | str) -> float:
        """Return the already-resolved tolerance for one quantity."""

        resolved_quantity = GeometryQuantity(quantity)
        override = self.overrides.get(resolved_quantity)
        if override is not None:
            return float(override)
        values = {
            GeometryQuantity.LENGTH: self.resolved.world,
            GeometryQuantity.BOUNDARY: self.resolved.boundary,
            GeometryQuantity.DEPTH: self.resolved.depth,
            GeometryQuantity.PARAMETER: self.resolved.parameter,
            GeometryQuantity.ANGULAR: self.resolved.angular,
            GeometryQuantity.SCREEN: self.screen,
        }
        return float(values[resolved_quantity])

    def with_overrides(self, **overrides: float) -> "ResolvedGeometryContext":
        merged = dict(self.overrides)
        merged.update(overrides)
        return replace(self, overrides=merged)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": {
                "relative": self.policy.relative,
                "absolute_floor": self.policy.absolute_floor,
                "angular": self.policy.angular,
                "boundary_factor": self.policy.boundary_factor,
                "depth_factor": self.policy.depth_factor,
            },
            "resolved": self.resolved.to_dict(),
            "screen": self.screen,
            "overrides": {
                quantity.value: value
                for quantity, value in self.overrides.items()
            },
        }


@dataclass(frozen=True, slots=True)
class GeometryContext:
    """Unresolved numerical policy shared by geometry-kernel layers.

    Call :meth:`resolve` with the same positions and optional edge length that
    a legacy solver would pass to ``TolerancePolicy.resolve``.  This explicit
    resolution step prevents a later layer from inventing a different local
    scale for the same frame.
    """

    tolerance: TolerancePolicy = field(default_factory=TolerancePolicy)
    screen_tolerance: float | None = None
    overrides: Mapping[GeometryQuantity | str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tolerance, TolerancePolicy):
            raise TypeError("tolerance must be a TolerancePolicy")
        screen = self.screen_tolerance
        if screen is not None:
            screen = _finite_nonnegative("screen_tolerance", screen)
        object.__setattr__(self, "screen_tolerance", screen)
        object.__setattr__(self, "overrides", _normalize_overrides(self.overrides))

    def resolve(
        self,
        positions: PositionInput = (),
        *,
        edge_length: float | None = None,
    ) -> ResolvedGeometryContext:
        """Resolve by delegating exactly once to the existing policy."""

        resolved = self.tolerance.resolve(positions, edge_length=edge_length)
        screen = (
            resolved.world
            if self.screen_tolerance is None
            else self.screen_tolerance
        )
        return ResolvedGeometryContext(
            policy=self.tolerance,
            resolved=resolved,
            screen=screen,
            overrides=self.overrides,
        )

    def with_tolerance(self, tolerance: TolerancePolicy) -> "GeometryContext":
        return replace(self, tolerance=tolerance)

    def with_screen_tolerance(self, value: float | None) -> "GeometryContext":
        return replace(self, screen_tolerance=value)

    def with_overrides(self, **overrides: float) -> "GeometryContext":
        merged = dict(self.overrides)
        merged.update(overrides)
        return replace(self, overrides=merged)


DEFAULT_GEOMETRY_CONTEXT = GeometryContext()
DEFAULT_RESOLVED_GEOMETRY_CONTEXT = DEFAULT_GEOMETRY_CONTEXT.resolve()


def resolve_geometry_context(
    context: GeometryContext | ResolvedGeometryContext | None = None,
    *,
    tolerance: TolerancePolicy | None = None,
    positions: PositionInput = (),
    edge_length: float | None = None,
) -> ResolvedGeometryContext:
    """Normalize legacy ``tolerance=`` and new context call styles.

    A resolved context is returned unchanged.  An unresolved context is
    resolved with the supplied geometry.  Supplying contradictory policies is
    rejected instead of choosing one silently.
    """

    if isinstance(context, ResolvedGeometryContext):
        if tolerance is not None and context.policy != tolerance:
            raise ValueError("context and tolerance specify different policies")
        if edge_length is not None or tuple(positions):
            raise ValueError("a resolved context cannot be resolved a second time")
        return context

    if context is not None and not isinstance(context, GeometryContext):
        raise TypeError("context must be a GeometryContext or ResolvedGeometryContext")

    if context is not None and tolerance is not None and context.tolerance != tolerance:
        raise ValueError("context and tolerance specify different policies")

    unresolved = context or GeometryContext(tolerance=tolerance or TolerancePolicy())
    return unresolved.resolve(positions, edge_length=edge_length)


def coordinate_scale(values: Iterable[object], *, floor: float = 0.0) -> float:
    """Return the largest finite coordinate magnitude.

    The default floor is zero.  A hidden unit-scale floor would reproduce the
    very small-geometry mismatch this context is designed to prevent.
    """

    result = _finite_nonnegative("floor", floor)
    stack = list(values)
    while stack:
        value = stack.pop()
        if isinstance(value, (str, bytes)):
            raise TypeError("coordinate values must be numeric")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError):
            try:
                stack.extend(value)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError("coordinate values must be numeric") from exc
            continue
        if not isfinite(scalar):
            raise ValueError("coordinate values must be finite")
        result = max(result, abs(scalar))
    return result
