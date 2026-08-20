"""Shared geometry-layer context and scale-aware tolerance access.

This module is intentionally independent from Manim. It is the canonical
entry point for numerical interpretation shared by geometry, topology,
visibility, and compositing code. Existing solvers can migrate incrementally
by accepting :class:`GeometryContext` while keeping their public APIs stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .contract import TolerancePolicy


class GeometryQuantity(str, Enum):
    """Kinds of numerical comparison used by the geometry kernel."""

    LENGTH = "length"
    DEPTH = "depth"
    PARAMETER = "parameter"
    ANGULAR = "angular"
    SCREEN = "screen"


_DEFAULT_FLOORS: Mapping[GeometryQuantity, float] = MappingProxyType(
    {
        GeometryQuantity.LENGTH: 1.0e-12,
        GeometryQuantity.DEPTH: 1.0e-12,
        GeometryQuantity.PARAMETER: 1.0e-12,
        GeometryQuantity.ANGULAR: 1.0e-12,
        GeometryQuantity.SCREEN: 1.0e-12,
    }
)

_POLICY_ALIASES: Mapping[GeometryQuantity, tuple[str, ...]] = MappingProxyType(
    {
        GeometryQuantity.LENGTH: (
            "length_epsilon",
            "length_tolerance",
            "point_epsilon",
            "point_tolerance",
        ),
        GeometryQuantity.DEPTH: (
            "depth_epsilon",
            "depth_tolerance",
            "visibility_epsilon",
        ),
        GeometryQuantity.PARAMETER: (
            "parameter_epsilon",
            "parameter_tolerance",
            "param_epsilon",
            "param_tolerance",
        ),
        GeometryQuantity.ANGULAR: (
            "angular_epsilon",
            "angular_tolerance",
            "angle_epsilon",
            "angle_tolerance",
        ),
        GeometryQuantity.SCREEN: (
            "screen_epsilon",
            "screen_tolerance",
            "projection_epsilon",
            "projection_tolerance",
        ),
    }
)

_ABSOLUTE_ALIASES = (
    "absolute_epsilon",
    "absolute_tolerance",
    "absolute",
    "abs_epsilon",
    "abs_tolerance",
    "abs_tol",
)
_RELATIVE_ALIASES = (
    "relative_epsilon",
    "relative_tolerance",
    "relative",
    "rel_epsilon",
    "rel_tolerance",
    "rel_tol",
)


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _candidate_value(candidate: Any, scale: float) -> float | None:
    """Return one finite non-negative policy value, or ``None``.

    A legacy policy may expose either a scalar attribute or a method accepting
    the local scale. Keeping that compatibility logic here prevents every
    solver from inventing another interpretation of the same policy.
    """

    try:
        value = candidate(scale) if callable(candidate) else candidate
    except TypeError:
        try:
            value = candidate() if callable(candidate) else candidate
        except (TypeError, ValueError, OverflowError):
            return None
    except (ValueError, OverflowError):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(result) or result < 0.0:
        return None
    return result


def _policy_value(policy: Any, names: Iterable[str], scale: float) -> float | None:
    for name in names:
        if not hasattr(policy, name):
            continue
        result = _candidate_value(getattr(policy, name), scale)
        if result is not None:
            return result
    return None


@dataclass(frozen=True, slots=True)
class GeometryScale:
    """Local scales for quantities that do not share the same unit."""

    length: float = 1.0
    depth: float = 1.0
    parameter: float = 1.0
    angular: float = 1.0
    screen: float = 1.0

    def __post_init__(self) -> None:
        for name in ("length", "depth", "parameter", "angular", "screen"):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(name, getattr(self, name)),
            )

    def for_quantity(self, quantity: GeometryQuantity | str) -> float:
        resolved = GeometryQuantity(quantity)
        return float(getattr(self, resolved.value))


@dataclass(frozen=True, slots=True)
class GeometryContext:
    """One numerical contract shared by all geometry-kernel layers.

    ``TolerancePolicy`` remains the source of authored defaults. This adapter
    gives every layer the same lookup rules, separates quantity-specific
    scales, and supports explicit per-scene overrides without mutating global
    state.
    """

    tolerance: TolerancePolicy = field(default_factory=TolerancePolicy)
    scale: GeometryScale = field(default_factory=GeometryScale)
    overrides: Mapping[GeometryQuantity | str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tolerance is None:
            raise ValueError("tolerance policy must not be None")
        normalized: dict[GeometryQuantity, float] = {}
        for quantity, value in dict(self.overrides).items():
            resolved = GeometryQuantity(quantity)
            normalized[resolved] = _finite_nonnegative(
                f"override[{resolved.value}]",
                value,
            )
        object.__setattr__(self, "overrides", MappingProxyType(normalized))

    def with_scale(self, **changes: float) -> "GeometryContext":
        """Return a context with selected local scales replaced."""

        return replace(self, scale=replace(self.scale, **changes))

    def with_tolerance(self, tolerance: TolerancePolicy) -> "GeometryContext":
        return replace(self, tolerance=tolerance)

    def with_overrides(self, **overrides: float) -> "GeometryContext":
        merged = dict(self.overrides)
        merged.update(overrides)
        return replace(self, overrides=merged)

    def epsilon(
        self,
        quantity: GeometryQuantity | str,
        *,
        scale: float | None = None,
    ) -> float:
        """Resolve one scale-aware tolerance using a single policy contract."""

        resolved = GeometryQuantity(quantity)
        if resolved in self.overrides:
            return float(self.overrides[resolved])

        effective_scale = self.scale.for_quantity(resolved)
        if scale is not None:
            effective_scale = _finite_nonnegative("scale", scale)

        for generic_name in ("epsilon_for", "tolerance_for"):
            generic = getattr(self.tolerance, generic_name, None)
            if not callable(generic):
                continue
            for args in (
                (resolved.value, effective_scale),
                (resolved, effective_scale),
                (resolved.value,),
                (resolved,),
            ):
                try:
                    candidate = generic(*args)
                except (TypeError, ValueError, OverflowError):
                    continue
                value = _candidate_value(candidate, effective_scale)
                if value is not None:
                    return max(_DEFAULT_FLOORS[resolved], value)

        specific = _policy_value(
            self.tolerance,
            _POLICY_ALIASES[resolved],
            effective_scale,
        )
        absolute = _policy_value(
            self.tolerance,
            _ABSOLUTE_ALIASES,
            effective_scale,
        )
        relative = _policy_value(
            self.tolerance,
            _RELATIVE_ALIASES,
            effective_scale,
        )

        candidates = [_DEFAULT_FLOORS[resolved]]
        if specific is not None:
            candidates.append(specific)
        if absolute is not None:
            candidates.append(absolute)
        if relative is not None:
            candidates.append(relative * max(1.0, effective_scale))
        return max(candidates)


DEFAULT_GEOMETRY_CONTEXT = GeometryContext()


def resolve_geometry_context(
    context: GeometryContext | None = None,
    *,
    tolerance: TolerancePolicy | None = None,
) -> GeometryContext:
    """Normalize legacy ``tolerance=`` and new ``context=`` call styles."""

    if context is not None and tolerance is not None:
        if context.tolerance is not tolerance and context.tolerance != tolerance:
            raise ValueError("context and tolerance specify different policies")
    if context is not None:
        return context
    if tolerance is not None:
        return GeometryContext(tolerance=tolerance)
    return DEFAULT_GEOMETRY_CONTEXT


def coordinate_scale(values: Iterable[Any], *, floor: float = 1.0) -> float:
    """Return a finite coordinate magnitude without depending on NumPy."""

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
                stack.extend(value)
            except TypeError as exc:
                raise TypeError("coordinate values must be numeric") from exc
            continue
        if not isfinite(scalar):
            raise ValueError("coordinate values must be finite")
        result = max(result, abs(scalar))
    return result
