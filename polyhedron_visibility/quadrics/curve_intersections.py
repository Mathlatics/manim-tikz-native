"""Analytic projected crossings and depth order for finite quadric curves.

The visibility kernel answers whether a curve is hidden by an opaque solid.
This module handles the separate stroke/stroke painter question: when two
analytic curves project to the same screen point, which stroke is farther
from the viewer there?

No dense screen sampling is used.  One curve is written in the same rational
chart used by :mod:`.critical`; substituting that chart into the other
curve's projected implicit line or conic produces a polynomial of degree at
most four.  Every real root is residual checked by :mod:`.roots`, mapped back
to both finite authored domains, and compared with the projection depth row.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import acos, asinh, atanh, atan2, ceil, floor, isfinite, sqrt, tau
from typing import Mapping, Sequence

import numpy as np
from numpy.polynomial import Polynomial

from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from ..parallel_solver import ParallelView
from ..topology import ParameterInterval
from .conics import ConicKind
from .critical import AnalyticCurve3D, _curve_chart
from .curves import EllipseArcCurve, ParametricConicBranch, SegmentCurve
from .roots import (
    PolynomialRootError,
    solve_real_polynomial,
    solve_real_polynomial_exp_chart,
)


PROJECTED_CURVE_CROSSING_SCHEMA = "manim-projected-curve-crossing/v1"
_FLOAT_EPSILON = float(np.finfo(float).eps)
_STATIONARY_RESIDUAL_FACTOR = 256.0
_DIRECT_TANGENCY_FACTOR = 32768.0


class ProjectedCurveIntersectionError(ValueError):
    """Projected crossings cannot be isolated without guessing."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectedCurveIntersectionError(f"{label} must be a non-empty string")
    return value.strip()


def _point2(value: object, label: str) -> tuple[float, float]:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectedCurveIntersectionError(
            f"{label} must contain two finite values"
        ) from exc
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ProjectedCurveIntersectionError(
            f"{label} must contain two finite values"
        )
    return float(point[0]), float(point[1])


@dataclass(frozen=True, slots=True)
class ProjectedCurveCrossing:
    """One isolated projected crossing with objective depth evidence."""

    crossing_id: str
    first_curve_id: str
    second_curve_id: str
    first_parameter: float
    second_parameter: float
    screen_point: tuple[float, float]
    first_depth: float
    second_depth: float
    far_curve_id: str | None
    near_curve_id: str | None
    tangential: bool = False
    schema: str = PROJECTED_CURVE_CROSSING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROJECTED_CURVE_CROSSING_SCHEMA:
            raise ProjectedCurveIntersectionError(
                "invalid projected-curve-crossing schema"
            )
        crossing_id = _identity(self.crossing_id, "crossing_id")
        first = _identity(self.first_curve_id, "first_curve_id")
        second = _identity(self.second_curve_id, "second_curve_id")
        if first >= second:
            raise ProjectedCurveIntersectionError(
                "curve identities must use strict canonical order"
            )
        parameters = (float(self.first_parameter), float(self.second_parameter))
        depths = (float(self.first_depth), float(self.second_depth))
        if not all(isfinite(item) for item in (*parameters, *depths)):
            raise ProjectedCurveIntersectionError(
                "crossing parameters and depths must be finite"
            )
        screen = _point2(self.screen_point, "screen_point")
        if not isinstance(self.tangential, bool):
            raise TypeError("tangential must be a bool")
        if self.far_curve_id is None or self.near_curve_id is None:
            if self.far_curve_id is not None or self.near_curve_id is not None:
                raise ProjectedCurveIntersectionError(
                    "coincident-depth crossings must omit both painter identities"
                )
        else:
            far = _identity(self.far_curve_id, "far_curve_id")
            near = _identity(self.near_curve_id, "near_curve_id")
            if {far, near} != {first, second} or far == near:
                raise ProjectedCurveIntersectionError(
                    "painter identities must be the two crossing curves"
                )
            far_depth = depths[0] if far == first else depths[1]
            near_depth = depths[0] if near == first else depths[1]
            if not far_depth < near_depth:
                raise ProjectedCurveIntersectionError(
                    "painter identities disagree with crossing depths"
                )
            object.__setattr__(self, "far_curve_id", far)
            object.__setattr__(self, "near_curve_id", near)
        object.__setattr__(self, "crossing_id", crossing_id)
        object.__setattr__(self, "first_curve_id", first)
        object.__setattr__(self, "second_curve_id", second)
        object.__setattr__(self, "first_parameter", parameters[0])
        object.__setattr__(self, "second_parameter", parameters[1])
        object.__setattr__(self, "screen_point", screen)
        object.__setattr__(self, "first_depth", depths[0])
        object.__setattr__(self, "second_depth", depths[1])

    @property
    def coincident_depth(self) -> bool:
        return self.far_curve_id is None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "crossingId": self.crossing_id,
            "firstCurveId": self.first_curve_id,
            "secondCurveId": self.second_curve_id,
            "firstParameter": self.first_parameter,
            "secondParameter": self.second_parameter,
            "screenPoint": list(self.screen_point),
            "firstDepth": self.first_depth,
            "secondDepth": self.second_depth,
            "farCurveId": self.far_curve_id,
            "nearCurveId": self.near_curve_id,
            "coincidentDepth": self.coincident_depth,
            "tangential": self.tangential,
        }


@dataclass(frozen=True, slots=True)
class _ProjectedModel:
    curve: AnalyticCurve3D
    screen_origin: np.ndarray
    screen_first: np.ndarray
    screen_second: np.ndarray | None
    canonical_matrix: np.ndarray | None
    line: np.ndarray | None
    parameter_kind: str
    branch_sign: int = 1
    rank_one_coefficients: tuple[float, float] | None = None

    def evaluate_homogeneous(self, x: Polynomial, y: Polynomial, w: Polynomial) -> Polynomial:
        if self.line is not None:
            return float(self.line[0]) * x + float(self.line[1]) * y + float(
                self.line[2]
            ) * w
        if self.canonical_matrix is None or self.screen_second is None:  # pragma: no cover
            raise ProjectedCurveIntersectionError("projected model has no equation")
        linear = np.column_stack((self.screen_first, self.screen_second))
        inverse = np.linalg.inv(linear)
        dx = x - float(self.screen_origin[0]) * w
        dy = y - float(self.screen_origin[1]) * w
        u = float(inverse[0, 0]) * dx + float(inverse[0, 1]) * dy
        v = float(inverse[1, 0]) * dx + float(inverse[1, 1]) * dy
        homogeneous = (u, v, w)
        result = Polynomial((0.0,))
        for row in range(3):
            for column in range(3):
                coefficient = float(self.canonical_matrix[row, column])
                if coefficient:
                    result = result + coefficient * homogeneous[row] * homogeneous[column]
        return result


@dataclass(frozen=True, slots=True)
class _SourceParameterCandidate:
    """One source root plus any already-certified geometric evidence."""

    parameter: float
    tangential_certified: bool = False


def _world_branch_geometry(
    curve: ParametricConicBranch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch = curve.parameterization
    embedding = np.asarray(curve.plane_embedding, dtype=float)
    linear = embedding[:3, :2]
    origin2 = np.asarray(branch.origin, dtype=float)
    origin = embedding[:3, 2] + linear @ origin2
    first = linear @ np.asarray(branch.first_axis, dtype=float)
    second = linear @ np.asarray(branch.second_axis, dtype=float)
    return origin, first, second


def _projected_line(origin: np.ndarray, direction: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ProjectedCurveIntersectionError(
            f"curve {label!r} collapses to one screen point"
        )
    # Construct the normal from the known displacement.  Taking a homogeneous
    # cross product of ``origin`` and ``origin + direction`` loses the small
    # direction completely when a tiny segment is far from the screen origin.
    normal = np.asarray((-direction[1], direction[0]), dtype=float) / norm
    line = np.asarray(
        (normal[0], normal[1], -float(np.dot(normal, origin))),
        dtype=float,
    )
    index = int(np.argmax(np.abs(line)))
    if line[index] < 0.0:
        line = -line
    return line


def _conic_matrix(
    origin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    canonical: np.ndarray,
    label: str,
) -> np.ndarray:
    linear = np.column_stack((first, second))
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    area_scale = first_length * second_length
    if area_scale <= 0.0 or abs(float(np.linalg.det(linear))) <= (
        1024.0 * _FLOAT_EPSILON * area_scale
    ):
        raise ProjectedCurveIntersectionError(
            f"curve {label!r} has an edge-on or singular conic projection"
        )
    # Keep the equation in this conic's own well-conditioned coordinates.
    # Expanding it into global screen coordinates would create O(origin**2)
    # constants and lose unit-radius detail after a large translation.
    return 0.5 * (canonical + canonical.T)


def _rank_one_ellipse_model(
    curve: AnalyticCurve3D,
    origin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> _ProjectedModel | None:
    """Represent an edge-on ellipse by its finite line support.

    A circle or ellipse viewed exactly edge-on is not an invalid projection:
    it is a line segment that the authored parameter traverses twice.  Keep a
    canonical support direction here and recover both parameters later from
    the original trigonometric axes.
    """

    linear = np.column_stack((first, second))
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    area_scale = first_length * second_length
    determinant = abs(float(np.linalg.det(linear)))
    if area_scale > 0.0 and determinant > 1024.0 * _FLOAT_EPSILON * area_scale:
        return None
    left, singular, _right = np.linalg.svd(linear, full_matrices=False)
    amplitude = float(singular[0]) if len(singular) else 0.0
    if not isfinite(amplitude) or amplitude <= 0.0:
        raise ProjectedCurveIntersectionError(
            f"curve {curve.curve_id!r} collapses to one screen point"
        )
    unit_direction = np.asarray(left[:, 0], dtype=float)
    direction = unit_direction * amplitude
    coefficients = (
        float(np.dot(first, unit_direction)),
        float(np.dot(second, unit_direction)),
    )
    return _ProjectedModel(
        curve,
        origin,
        direction,
        None,
        None,
        _projected_line(origin, direction, curve.curve_id),
        "ellipse_rank_one",
        rank_one_coefficients=coefficients,
    )


def _projected_model(curve: AnalyticCurve3D, view: ParallelView) -> _ProjectedModel:
    screen = view.matrix[:2]
    if isinstance(curve, SegmentCurve):
        origin = screen @ np.asarray(curve.start, dtype=float)
        direction = screen @ np.asarray(curve.displacement, dtype=float)
        return _ProjectedModel(
            curve,
            origin,
            direction,
            None,
            None,
            _projected_line(origin, direction, curve.curve_id),
            "line",
        )

    if isinstance(curve, EllipseArcCurve):
        origin = screen @ np.asarray(curve.center, dtype=float)
        first = screen @ np.asarray(curve.first_axis, dtype=float)
        second = screen @ np.asarray(curve.second_axis, dtype=float)
        rank_one = _rank_one_ellipse_model(curve, origin, first, second)
        if rank_one is not None:
            return rank_one
        matrix = _conic_matrix(
            origin,
            first,
            second,
            np.diag((1.0, 1.0, -1.0)),
            curve.curve_id,
        )
        return _ProjectedModel(
            curve, origin, first, second, matrix, None, "ellipse"
        )

    if not isinstance(curve, ParametricConicBranch):  # pragma: no cover
        raise TypeError("curve must be a supported analytic curve")
    world_origin, world_first, world_second = _world_branch_geometry(curve)
    origin = screen @ world_origin
    first = screen @ world_first
    second = screen @ world_second
    kind = curve.parameterization.kind
    if kind in {ConicKind.CIRCLE, ConicKind.ELLIPSE}:
        canonical = np.diag((1.0, 1.0, -1.0))
        parameter_kind = "ellipse"
        rank_one = _rank_one_ellipse_model(curve, origin, first, second)
        if rank_one is not None:
            return rank_one
    elif kind is ConicKind.HYPERBOLA:
        canonical = np.diag((1.0, -1.0, -1.0))
        parameter_kind = "hyperbola"
    elif kind is ConicKind.PARABOLA:
        canonical = np.asarray(
            ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.5, 0.0)),
            dtype=float,
        )
        parameter_kind = "parabola"
    elif kind in {
        ConicKind.INTERSECTING_LINES,
        ConicKind.PARALLEL_LINES,
        ConicKind.COINCIDENT_LINE,
    }:
        return _ProjectedModel(
            curve,
            origin,
            first,
            None,
            None,
            _projected_line(origin, first, curve.curve_id),
            "line",
        )
    else:
        raise ProjectedCurveIntersectionError(
            f"curve {curve.curve_id!r} has unsupported conic kind {kind.value!r}"
        )
    matrix = _conic_matrix(origin, first, second, canonical, curve.curve_id)
    return _ProjectedModel(
        curve,
        origin,
        first,
        second,
        matrix,
        None,
        parameter_kind,
        curve.parameterization.branch_sign,
    )


def _parameters_in_angular_domain(base: float, curve: AnalyticCurve3D, epsilon: float) -> tuple[float, ...]:
    lower = floor((curve.domain.start - base) / tau) - 1
    upper = ceil((curve.domain.end - base) / tau) + 1
    values: list[float] = []
    for index in range(lower, upper + 1):
        candidate = base + index * tau
        if candidate < curve.domain.start - epsilon or candidate > curve.domain.end + epsilon:
            continue
        candidate = min(curve.domain.end, max(curve.domain.start, candidate))
        if not values or candidate - values[-1] > epsilon:
            values.append(float(candidate))
    if (
        getattr(curve, "closed", False)
        and len(values) == 2
        and abs(values[0] - curve.domain.start) <= epsilon
        and abs(values[1] - curve.domain.end) <= epsilon
    ):
        return (values[0],)
    return tuple(values)


def _model_world_origin(model: _ProjectedModel) -> np.ndarray:
    curve = model.curve
    if isinstance(curve, SegmentCurve):
        return np.asarray(curve.start, dtype=float)
    if isinstance(curve, EllipseArcCurve):
        return np.asarray(curve.center, dtype=float)
    origin, _first, _second = _world_branch_geometry(curve)
    return origin


def _target_parameters(
    model: _ProjectedModel,
    point: np.ndarray,
    *,
    parameter_epsilon: float,
    screen_epsilon: float,
    world_point: np.ndarray | None = None,
    view: ParallelView | None = None,
) -> tuple[float, ...]:
    curve = model.curve
    if (world_point is None) != (view is None):
        raise TypeError("world_point and view must be provided together")
    screen_delta = point - model.screen_origin
    if world_point is not None and view is not None:
        # Project the local world displacement instead of subtracting two
        # large projected positions.  This makes parameter recovery stable
        # across BLAS/NumPy matmul implementations for translated geometry.
        screen_delta = view.matrix[:2] @ (
            np.asarray(world_point, dtype=float) - _model_world_origin(model)
        )
    if model.parameter_kind == "ellipse_rank_one":
        axis = np.asarray(model.screen_first, dtype=float)
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 0.0:  # pragma: no cover - construction rejects this
            return ()
        axis /= axis_length
        if model.rank_one_coefficients is None:  # pragma: no cover
            return ()
        first_coefficient, second_coefficient = model.rank_one_coefficients
        amplitude = float(np.hypot(first_coefficient, second_coefficient))
        if amplitude <= 0.0:  # pragma: no cover - construction rejects this
            return ()
        ratio = float(np.dot(screen_delta, axis)) / amplitude
        coordinate_epsilon = screen_epsilon / amplitude
        if ratio < -1.0 - coordinate_epsilon or ratio > 1.0 + coordinate_epsilon:
            return ()
        ratio = min(1.0, max(-1.0, ratio))
        phase = atan2(second_coefficient, first_coefficient)
        offset = acos(ratio)
        candidates = tuple(
            parameter
            for base in (phase - offset, phase + offset)
            for parameter in _parameters_in_angular_domain(
                base, curve, parameter_epsilon
            )
        )
    elif model.parameter_kind == "line":
        displacement = model.screen_first
        denominator = float(np.dot(displacement, displacement))
        if denominator <= 0.0:  # pragma: no cover - construction rejects this
            return ()
        ratio = float(np.dot(screen_delta, displacement) / denominator)
        if isinstance(curve, SegmentCurve):
            candidate = curve.domain.start + ratio * curve.domain.length
        else:
            candidate = ratio
        candidates = (candidate,)
    else:
        if model.screen_second is None:  # pragma: no cover
            return ()
        linear = np.column_stack((model.screen_first, model.screen_second))
        coordinates = np.linalg.solve(linear, screen_delta)
        minimum_scale = float(np.min(np.linalg.svd(linear, compute_uv=False)))
        if minimum_scale <= 0.0:
            return ()
        coordinate_epsilon = screen_epsilon / minimum_scale
        first, second = float(coordinates[0]), float(coordinates[1])
        if model.parameter_kind == "ellipse":
            candidates = _parameters_in_angular_domain(
                atan2(second, first), curve, parameter_epsilon
            )
        elif model.parameter_kind == "hyperbola":
            candidate = asinh(second)
            expected = model.branch_sign * np.cosh(candidate)
            if abs(first - expected) > coordinate_epsilon * max(1.0, abs(expected)):
                return ()
            candidates = (candidate,)
        elif model.parameter_kind == "parabola":
            if abs(second - first * first) > coordinate_epsilon * max(
                1.0, abs(second), first * first
            ):
                return ()
            candidates = (first,)
        else:  # pragma: no cover
            return ()

    result: list[float] = []
    for candidate in sorted(float(item) for item in candidates):
        if world_point is not None and view is not None:
            for endpoint in (curve.domain.start, curve.domain.end):
                endpoint_delta = view.matrix[:2] @ (
                    np.asarray(world_point, dtype=float)
                    - np.asarray(curve.point(endpoint), dtype=float)
                )
                if float(np.linalg.norm(endpoint_delta)) <= screen_epsilon:
                    candidate = endpoint
                    break
        if candidate < curve.domain.start - parameter_epsilon:
            continue
        if candidate > curve.domain.end + parameter_epsilon:
            continue
        candidate = min(curve.domain.end, max(curve.domain.start, candidate))
        if not result or candidate - result[-1] > parameter_epsilon:
            result.append(candidate)
    return tuple(result)


def _curve_points_for_context(curve: AnalyticCurve3D) -> tuple[tuple[float, float, float], ...]:
    values = (curve.domain.start, curve.domain.midpoint, curve.domain.end)
    result = [curve.point(value) for value in values]
    if isinstance(curve, EllipseArcCurve):
        center = np.asarray(curve.center, dtype=float)
        for axis in (curve.first_axis, curve.second_axis):
            vector = np.asarray(axis, dtype=float)
            result.extend(
                tuple(float(item) for item in point)
                for point in (center - vector, center + vector)
            )
    elif isinstance(curve, ParametricConicBranch) and curve.parameterization.kind in {
        ConicKind.CIRCLE,
        ConicKind.ELLIPSE,
    }:
        origin, first, second = _world_branch_geometry(curve)
        for axis in (first, second):
            result.extend(
                tuple(float(item) for item in point)
                for point in (origin - axis, origin + axis)
            )
    return tuple(result)


def _resolve_context(
    curves: Sequence[AnalyticCurve3D],
    context: GeometryContext | ResolvedGeometryContext | None,
) -> ResolvedGeometryContext:
    if isinstance(context, ResolvedGeometryContext):
        return resolve_geometry_context(context)
    return resolve_geometry_context(
        context,
        positions=tuple(
            point for curve in curves for point in _curve_points_for_context(curve)
        ),
    )


def _model_screen_feature_scale(model: _ProjectedModel) -> float:
    if model.line is not None:
        return float(np.linalg.norm(model.screen_first))
    if model.screen_second is None:
        return 0.0
    linear = np.column_stack((model.screen_first, model.screen_second))
    return float(np.min(np.linalg.svd(linear, compute_uv=False)))


def _pair_screen_epsilon(
    first: _ProjectedModel,
    second: _ProjectedModel,
    view: ParallelView,
    context: ResolvedGeometryContext,
) -> float:
    explicit_screen = (
        context.epsilon(GeometryQuantity.SCREEN)
        if (
            GeometryQuantity.SCREEN in context.overrides
            or context.screen != context.resolved.world
        )
        else 0.0
    )
    if GeometryQuantity.BOUNDARY in context.overrides:
        return max(
            explicit_screen,
            context.epsilon(GeometryQuantity.BOUNDARY)
            * max(1.0, float(np.linalg.norm(view.matrix[:2], ord=2))),
        )
    feature = min(
        value
        for value in (
            _model_screen_feature_scale(first),
            _model_screen_feature_scale(second),
        )
        if value > 0.0
    )
    view_scale = max(1.0, float(np.linalg.norm(view.matrix[:2], ord=2)))
    policy = context.policy
    relative = policy.relative * policy.boundary_factor * feature
    absolute = policy.absolute_floor * policy.boundary_factor * view_scale
    coordinate_scale = max(
        1.0,
        float(np.linalg.norm(first.screen_origin)),
        float(np.linalg.norm(second.screen_origin)),
    )
    ulp = abs(float(np.spacing(coordinate_scale)))
    if feature <= 128.0 * ulp:
        raise ProjectedCurveIntersectionError(
            "projected curve detail is numerically indistinguishable after "
            "the authored screen translation"
        )
    machine = 4.0 * ulp
    return max(relative, absolute, machine, explicit_screen)


def _pair_tangency_epsilon(
    first: _ProjectedModel,
    second: _ProjectedModel,
) -> float:
    """Return the baseline dimensionless angular arithmetic uncertainty.

    Translation does not change a tangent direction.  In particular, using
    ``eps * |origin| / feature`` here would turn any sufficiently translated
    near-secant into a tangent.  The positional roundoff of an authored
    contact is handled separately by :func:`_projected_contact_roundoff`.
    """

    del first, second
    return 1024.0 * _FLOAT_EPSILON


def _curve_depth_feature_scale(
    curve: AnalyticCurve3D,
    depth_row: np.ndarray,
) -> float:
    if isinstance(curve, SegmentCurve):
        return abs(float(depth_row @ np.asarray(curve.displacement, dtype=float)))
    if isinstance(curve, EllipseArcCurve):
        first = float(depth_row @ np.asarray(curve.first_axis, dtype=float))
        second = float(depth_row @ np.asarray(curve.second_axis, dtype=float))
        return float(np.hypot(first, second))
    values = tuple(
        float(depth_row @ np.asarray(curve.point(parameter), dtype=float))
        for parameter in (curve.domain.start, curve.domain.midpoint, curve.domain.end)
    )
    return max(values) - min(values)


def _pair_depth_epsilon(
    first: AnalyticCurve3D,
    second: AnalyticCurve3D,
    depth_row: np.ndarray,
    context: ResolvedGeometryContext,
) -> float:
    if GeometryQuantity.DEPTH in context.overrides:
        return context.epsilon(GeometryQuantity.DEPTH) * max(
            1.0,
            float(np.linalg.norm(depth_row)),
        )
    feature = max(
        _curve_depth_feature_scale(first, depth_row),
        _curve_depth_feature_scale(second, depth_row),
    )
    policy = context.policy
    relative = policy.relative * policy.depth_factor * feature
    absolute = policy.absolute_floor * policy.depth_factor * max(
        1.0,
        float(np.linalg.norm(depth_row)),
    )
    return max(relative, absolute)


def _chart_polynomial(
    source: AnalyticCurve3D,
    target: _ProjectedModel,
    view: ParallelView,
) -> tuple[object, Polynomial]:
    chart = _curve_chart(source)
    screen = view.matrix[:2]
    x = sum(
        (float(screen[0, axis]) * chart.numerator[axis] for axis in range(3)),
        Polynomial((0.0,)),
    )
    y = sum(
        (float(screen[1, axis]) * chart.numerator[axis] for axis in range(3)),
        Polynomial((0.0,)),
    )
    return chart, target.evaluate_homogeneous(x, y, chart.denominator)


def _direct_equation_value(model: _ProjectedModel, point: np.ndarray) -> float:
    if model.line is not None:
        # Evaluate against a known point on the line.  The expanded constant
        # term cancels two large translated coordinates and can turn an exact
        # tangent into a pair of fake nearby crossings.
        return float(
            np.dot(
                model.line[:2],
                np.asarray(point, dtype=float) - model.screen_origin,
            )
        )
    if model.canonical_matrix is None or model.screen_second is None:  # pragma: no cover
        return float("nan")
    linear = np.column_stack((model.screen_first, model.screen_second))
    local = np.linalg.solve(linear, point - model.screen_origin)
    homogeneous = np.asarray((local[0], local[1], 1.0), dtype=float)
    return float(homogeneous @ model.canonical_matrix @ homogeneous)


def _direct_equation_matches(
    model: _ProjectedModel,
    point: np.ndarray,
    *,
    screen_epsilon: float,
) -> bool:
    value = abs(_direct_equation_value(model, point))
    if model.line is not None:
        return value <= screen_epsilon
    if model.screen_second is None:
        return False
    linear = np.column_stack((model.screen_first, model.screen_second))
    singular = np.linalg.svd(linear, compute_uv=False)
    minimum_scale = float(np.min(singular))
    if minimum_scale <= 0.0:
        return False
    local = np.linalg.solve(linear, point - model.screen_origin)
    local_epsilon = screen_epsilon / minimum_scale
    return value <= 8.0 * local_epsilon * max(1.0, float(np.linalg.norm(local)))


def _normalized_polynomial_residual(
    coefficients: Sequence[float],
    value: float,
) -> float:
    """Return a chart-scale-independent residual using long-double Horner."""

    argument = np.longdouble(value)
    absolute_argument = max(np.longdouble(1.0), abs(argument))
    polynomial = np.longdouble(0.0)
    scale = np.longdouble(0.0)
    power = np.longdouble(1.0)
    for coefficient in reversed(coefficients):
        polynomial = polynomial * argument + np.longdouble(coefficient)
    for coefficient in coefficients:
        scale += abs(np.longdouble(coefficient)) * power
        power *= absolute_argument
    if scale == 0.0:
        return 0.0 if polynomial == 0.0 else float("inf")
    return float(abs(polynomial) / scale)


def _polynomial_derivative(coefficients: Sequence[float]) -> tuple[float, ...]:
    result = tuple(
        index * float(coefficients[index])
        for index in range(1, len(coefficients))
    )
    while len(result) > 1 and result[-1] == 0.0:
        result = result[:-1]
    return result


def _deflate_polynomial_root(
    coefficients: Sequence[float],
    root: float,
) -> tuple[float, ...]:
    """Synthetic-divide one certified ``(x - root)`` factor."""

    if len(coefficients) <= 1:
        return tuple(float(item) for item in coefficients)
    source = tuple(np.longdouble(item) for item in coefficients)
    value = np.longdouble(root)
    quotient = [np.longdouble(0.0)] * (len(source) - 1)
    quotient[-1] = source[-1]
    for index in range(len(quotient) - 2, -1, -1):
        quotient[index] = source[index + 1] + value * quotient[index + 1]
    return tuple(float(item) for item in quotient)


def _stationary_chart_values(
    coefficients: Sequence[float],
    chart: object,
    *,
    context: ResolvedGeometryContext,
    parameter_epsilon: float,
) -> tuple[float, ...]:
    """Return derivative roots in the source curve's authoritative chart."""

    derivative = _polynomial_derivative(coefficients)
    if len(derivative) <= 1:
        return ()
    # Avoid normalizing a linear equation across an enormous exp-chart domain:
    # that transformation can lose the small positive stationary value before
    # the general solver sees it.
    if len(derivative) == 2:
        candidate = -float(derivative[0]) / float(derivative[1])
        root_domain = chart.root_domain
        if root_domain is not None and not root_domain.contains(
            candidate,
            tolerance=parameter_epsilon * max(1.0, abs(candidate)),
        ):
            return ()
        return (float(candidate),)
    try:
        if chart.name == "exp":
            roots = solve_real_polynomial_exp_chart(
                derivative,
                parameter_domain=chart.curve_domain,
                context=context,
                parameter_tolerance=parameter_epsilon,
            )
            return tuple(item.chart_root.value for item in roots)
        roots = solve_real_polynomial(
            derivative,
            domain=chart.root_domain,
            context=context,
            parameter_tolerance=(
                parameter_epsilon
                if chart.name == "parameter"
                else max(4096.0 * _FLOAT_EPSILON, parameter_epsilon)
            ),
        )
        return tuple(item.value for item in roots)
    except (OverflowError, PolynomialRootError) as exc:
        raise ProjectedCurveIntersectionError(
            f"projected crossing stationary equation is ambiguous: {exc}"
        ) from exc


def _normalized_direct_equation_residual(
    model: _ProjectedModel,
    point: np.ndarray,
) -> float:
    value = abs(_direct_equation_value(model, point))
    if model.line is not None:
        feature = _model_screen_feature_scale(model)
        return float("inf") if feature <= 0.0 else value / feature
    if model.canonical_matrix is None or model.screen_second is None:
        return float("inf")
    linear = np.column_stack((model.screen_first, model.screen_second))
    local = np.linalg.solve(linear, point - model.screen_origin)
    homogeneous = np.asarray((local[0], local[1], 1.0), dtype=float)
    scale = float(
        np.sum(
            np.abs(model.canonical_matrix)
            * np.outer(np.abs(homogeneous), np.abs(homogeneous))
        )
    )
    if scale <= 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return value / scale


def _component_ulp(values: np.ndarray) -> np.ndarray:
    """Return a per-coordinate one-ULP enclosure for finite float data."""

    data = np.asarray(values, dtype=float)
    upward = np.abs(np.nextafter(data, np.inf) - data)
    downward = np.abs(data - np.nextafter(data, -np.inf))
    return np.maximum(upward, downward)


def _curve_point_roundoff_bound(
    curve: AnalyticCurve3D,
    parameter: float,
) -> np.ndarray:
    """Propagate authored-coordinate ULPs through one curve point evaluation.

    This is intentionally local to the actual contact.  A global relative
    tolerance based on a remote world origin would hide resolvable nearby
    crossings.  Ellipse points receive a componentwise bound for their
    center, axes, trigonometric products and two additions.  Other curve
    families retain the final evaluation ULP as a conservative supplemental
    bound; their existing analytic residual tests remain authoritative.
    """

    point = np.asarray(curve.point(parameter), dtype=float)
    if isinstance(curve, EllipseArcCurve):
        center = np.asarray(curve.center, dtype=float)
        first_axis = np.asarray(curve.first_axis, dtype=float)
        second_axis = np.asarray(curve.second_axis, dtype=float)
        sine = float(np.sin(parameter))
        cosine = float(np.cos(parameter))
        first_term = cosine * first_axis
        second_term = sine * second_axis
        radial = first_term + second_term
        return (
            _component_ulp(center)
            + abs(cosine) * _component_ulp(first_axis)
            + np.abs(first_axis) * _component_ulp(np.asarray(cosine))
            + _component_ulp(first_term)
            + abs(sine) * _component_ulp(second_axis)
            + np.abs(second_axis) * _component_ulp(np.asarray(sine))
            + _component_ulp(second_term)
            + _component_ulp(radial)
            + _component_ulp(point)
        )
    return _component_ulp(point)


def _projected_contact_roundoff(
    first: AnalyticCurve3D,
    first_parameter: float,
    second: _ProjectedModel,
    second_parameter: float,
    view: ParallelView,
) -> tuple[float, float]:
    """Return contact separation and its local componentwise ULP enclosure.

    Besides projecting each authored coordinate ULP with ``abs(M)``, the
    enclosure propagates that screen uncertainty through the target conic's
    local 2x2 solve.  This is the narrowly scoped condition-number term needed
    when two projected axes are nearly parallel; it is not a global tolerance
    based on the curve's distance from the world origin.
    """

    first_point = np.asarray(first.point(first_parameter), dtype=float)
    second_curve = second.curve
    second_point = np.asarray(
        second_curve.point(second_parameter), dtype=float
    )
    world_delta = first_point - second_point
    world_uncertainty = (
        _curve_point_roundoff_bound(first, first_parameter)
        + _curve_point_roundoff_bound(second_curve, second_parameter)
        + _component_ulp(world_delta)
    )
    screen = view.matrix[:2]
    products = screen * world_delta[np.newaxis, :]
    screen_delta = screen @ world_delta
    screen_uncertainty = (
        np.abs(screen) @ world_uncertainty
        + np.sum(_component_ulp(products), axis=1)
        + _component_ulp(screen_delta)
    )
    if second.screen_second is not None:
        linear = np.column_stack((second.screen_first, second.screen_second))
        try:
            inverse = np.linalg.inv(linear)
            local = np.linalg.solve(
                linear,
                screen @ (first_point - _model_world_origin(second)),
            )
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(screen_delta)), float("inf")

        def projected_vector_uncertainty(vector: np.ndarray) -> np.ndarray:
            projected_products = screen * vector[np.newaxis, :]
            projected = screen @ vector
            return (
                np.abs(screen) @ _component_ulp(vector)
                + np.sum(_component_ulp(projected_products), axis=1)
                + _component_ulp(projected)
            )

        if isinstance(second_curve, EllipseArcCurve):
            world_first = np.asarray(second_curve.first_axis, dtype=float)
            world_second = np.asarray(second_curve.second_axis, dtype=float)
        else:
            _origin, world_first, world_second = _world_branch_geometry(
                second_curve
            )
        linear_uncertainty = np.column_stack(
            (
                projected_vector_uncertainty(world_first),
                projected_vector_uncertainty(world_second),
            )
        )
        local_uncertainty = np.abs(inverse) @ (
            screen_uncertainty + linear_uncertainty @ np.abs(local)
        )
        if second.parameter_kind == "ellipse":
            denominator = float(np.dot(local, local))
            parameter_uncertainty = (
                (
                    abs(float(local[1])) * float(local_uncertainty[0])
                    + abs(float(local[0])) * float(local_uncertainty[1])
                )
                / denominator
                if denominator > 0.0
                else 0.0
            )
        elif second.parameter_kind == "hyperbola":
            parameter_uncertainty = float(local_uncertainty[1]) / max(
                1.0, abs(float(local[0]))
            )
        elif second.parameter_kind == "parabola":
            parameter_uncertainty = float(local_uncertainty[0])
        else:  # pragma: no cover - projected conic kinds are exhaustive
            parameter_uncertainty = 0.0
        target_tangent = screen @ np.asarray(
            second_curve.tangent(second_parameter), dtype=float
        )
        screen_uncertainty = screen_uncertainty + (
            np.abs(target_tangent) * parameter_uncertainty
        )
    return float(np.linalg.norm(screen_delta)), float(
        np.linalg.norm(screen_uncertainty)
    )


def _stationary_contact_ulp_residual_limit(
    source: AnalyticCurve3D,
    source_model: _ProjectedModel,
    target: _ProjectedModel,
    view: ParallelView,
    parameter: float,
    *,
    parameter_epsilon: float,
    screen_epsilon: float,
) -> float:
    """Convert a certified local contact ULP enclosure to residual units."""

    source_point = view.matrix[:2] @ np.asarray(
        source.point(parameter), dtype=float
    )
    feature = min(
        value
        for value in (
            _model_screen_feature_scale(source_model),
            _model_screen_feature_scale(target),
        )
        if value > 0.0
    )
    result = 0.0
    for target_parameter in _target_parameters(
        target,
        source_point,
        parameter_epsilon=parameter_epsilon,
        screen_epsilon=screen_epsilon,
        world_point=np.asarray(source.point(parameter), dtype=float),
        view=view,
    ):
        separation, uncertainty = _projected_contact_roundoff(
            source,
            parameter,
            target,
            target_parameter,
            view,
        )
        if separation <= uncertainty:
            result = max(result, uncertainty / feature)
    return result


def _stationary_is_projected_tangency(
    source: AnalyticCurve3D,
    source_model: _ProjectedModel,
    target: _ProjectedModel,
    view: ParallelView,
    parameter: float,
    *,
    parameter_epsilon: float,
    screen_epsilon: float,
) -> bool:
    screen = view.matrix[:2]
    source_world = np.asarray(source.point(parameter), dtype=float)
    source_point = screen @ source_world
    direct_residual = _normalized_direct_equation_residual(target, source_point)
    direct_tolerance = max(
        _DIRECT_TANGENCY_FACTOR * _FLOAT_EPSILON,
        2.0 * _pair_tangency_epsilon(source_model, target),
    )
    target_parameters = _target_parameters(
        target,
        source_point,
        parameter_epsilon=parameter_epsilon,
        screen_epsilon=screen_epsilon,
        world_point=source_world,
        view=view,
    )
    source_tangent = screen @ np.asarray(source.tangent(parameter), dtype=float)
    for target_parameter in target_parameters:
        separation, roundoff = _projected_contact_roundoff(
            source,
            parameter,
            target,
            target_parameter,
            view,
        )
        ulp_contact = separation <= roundoff
        if direct_residual > direct_tolerance and not ulp_contact:
            continue
        target_tangent = screen @ np.asarray(
            target.curve.tangent(target_parameter), dtype=float
        )
        tangent_scale = float(np.linalg.norm(source_tangent)) * float(
            np.linalg.norm(target_tangent)
        )
        if tangent_scale <= 0.0:
            continue
        cross = abs(
            float(
                source_tangent[0] * target_tangent[1]
                - source_tangent[1] * target_tangent[0]
            )
        )
        angular_tolerance = direct_tolerance
        if ulp_contact:
            feature = min(
                value
                for value in (
                    _model_screen_feature_scale(source_model),
                    _model_screen_feature_scale(target),
                )
                if value > 0.0
            )
            # Both recovered parameters contribute first-order angular error.
            # The contact test above already proved their positional mismatch
            # lies inside the local ULP enclosure, so this factor cannot turn
            # a resolvable near-secant into a tangent.
            angular_tolerance = max(angular_tolerance, 2.0 * roundoff / feature)
        if cross <= angular_tolerance * tangent_scale:
            return True
    return False


def _canonical_curve_parameter(
    curve: AnalyticCurve3D,
    parameter: float,
    parameter_epsilon: float,
) -> float:
    """Use the authored start as the canonical seam of a closed curve."""

    value = min(curve.domain.end, max(curve.domain.start, float(parameter)))
    if getattr(curve, "closed", False):
        if (
            abs(value - curve.domain.start) <= parameter_epsilon
            or abs(value - curve.domain.end) <= parameter_epsilon
        ):
            return float(curve.domain.start)
    return value


def _crossing_identity(first_curve_id: str, second_curve_id: str, index: int) -> str:
    """Encode curve identities without delimiter collisions."""

    return (
        f"crossing:{len(first_curve_id)}:{first_curve_id}:"
        f"{len(second_curve_id)}:{second_curve_id}:{index}"
    )


def _support_probe_points(model: _ProjectedModel) -> tuple[np.ndarray, ...]:
    """Return exact characteristic screen points for one implicit support."""

    if model.line is not None:
        return (
            model.screen_origin - model.screen_first,
            model.screen_origin,
            model.screen_origin + model.screen_first,
        )
    if model.screen_second is None:
        return ()
    if model.parameter_kind == "ellipse":
        root_half = sqrt(0.5)
        local_points = (
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
            (root_half, root_half),
            (-root_half, root_half),
        )
    elif model.parameter_kind == "hyperbola":
        cosine = float(np.cosh(1.0))
        sine = float(np.sinh(1.0))
        local_points = (
            (1.0, 0.0),
            (-1.0, 0.0),
            (cosine, sine),
            (cosine, -sine),
            (-cosine, sine),
            (-cosine, -sine),
        )
    elif model.parameter_kind == "parabola":
        local_points = tuple((value, value * value) for value in (-2.0, -1.0, 0.0, 1.0, 2.0))
    else:
        return ()
    return tuple(
        model.screen_origin
        + first * model.screen_first
        + second * model.screen_second
        for first, second in local_points
    )


def _screen_distance_to_support(
    model: _ProjectedModel,
    point: np.ndarray,
    *,
    screen_epsilon: float,
) -> tuple[float, bool]:
    """Return first-order screen distance and whether it was certified."""

    if model.line is not None:
        return abs(_direct_equation_value(model, point)), True
    if model.canonical_matrix is None or model.screen_second is None:
        return float("inf"), False
    linear = np.column_stack((model.screen_first, model.screen_second))
    try:
        local = np.linalg.solve(linear, point - model.screen_origin)
    except np.linalg.LinAlgError:
        return float("inf"), False
    reconstructed = model.screen_origin + linear @ local
    reconstruction_error = float(np.linalg.norm(reconstructed - point))
    if not np.all(np.isfinite(local)) or reconstruction_error > 8.0 * screen_epsilon:
        return float("inf"), False
    homogeneous = np.asarray((local[0], local[1], 1.0), dtype=float)
    value = abs(float(homogeneous @ model.canonical_matrix @ homogeneous))
    local_gradient = 2.0 * (model.canonical_matrix @ homogeneous)[:2]
    try:
        screen_gradient = np.linalg.solve(linear.T, local_gradient)
    except np.linalg.LinAlgError:
        return float("inf"), False
    gradient_norm = float(np.linalg.norm(screen_gradient))
    if not isfinite(gradient_norm) or gradient_norm <= 0.0:
        # The center of a non-degenerate ellipse has zero implicit gradient
        # but is still objectively away from its boundary.  The square-root
        # fallback converts its quadratic residual to a conservative screen
        # distance; a zero residual at a singular point remains ambiguous.
        if value == 0.0:
            return float("inf"), False
        minimum_scale = float(np.min(np.linalg.svd(linear, compute_uv=False)))
        return sqrt(value) * minimum_scale, True
    return value / gradient_norm, True


def _support_condition_uncertainty(model: _ProjectedModel) -> float:
    if model.line is not None:
        return 0.0
    if model.screen_second is None:
        return float("inf")
    linear = np.column_stack((model.screen_first, model.screen_second))
    singular = np.linalg.svd(linear, compute_uv=False)
    minimum = float(np.min(singular))
    maximum = float(np.max(singular))
    if minimum <= 0.0 or not isfinite(maximum):
        return float("inf")
    return _FLOAT_EPSILON * (maximum / minimum) * maximum


def _support_direction_relation(
    source: _ProjectedModel,
    target: _ProjectedModel,
    *,
    screen_epsilon: float,
) -> str:
    if max(
        _support_condition_uncertainty(source),
        _support_condition_uncertainty(target),
    ) > screen_epsilon:
        return "ambiguous"
    probes = _support_probe_points(source)
    if not probes:
        return "ambiguous"
    distances: list[float] = []
    for point in probes:
        distance, certified = _screen_distance_to_support(
            target,
            point,
            screen_epsilon=screen_epsilon,
        )
        if not certified or not isfinite(distance):
            return "ambiguous"
        distances.append(distance)
    maximum = max(distances, default=float("inf"))
    if maximum <= screen_epsilon:
        return "same"
    if maximum > 4.0 * screen_epsilon:
        return "distinct"
    return "ambiguous"


def _same_projected_support(
    first: _ProjectedModel,
    second: _ProjectedModel,
    *,
    screen_epsilon: float,
) -> bool:
    """Recognize support symmetrically in a shared screen-distance metric."""

    if (first.line is None) != (second.line is None):
        return False
    forward = _support_direction_relation(
        first,
        second,
        screen_epsilon=screen_epsilon,
    )
    reverse = _support_direction_relation(
        second,
        first,
        screen_epsilon=screen_epsilon,
    )
    if forward == reverse == "same":
        return True
    if forward == reverse == "distinct":
        return False
    raise ProjectedCurveIntersectionError(
        f"projected support relation for {first.curve.curve_id!r} and "
        f"{second.curve.curve_id!r} cannot be certified symmetrically"
    )


def _same_line_support_parameters(
    source: AnalyticCurve3D,
    target: _ProjectedModel,
    view: ParallelView,
    *,
    parameter_epsilon: float,
    screen_epsilon: float,
) -> tuple[float, ...]:
    """Classify finite overlap on one shared projected line support."""

    screen = view.matrix[:2]
    # ``screen_first`` is stored inside a frozen model, but NumPy arrays are
    # still mutable.  Copy before normalization so one pair calculation cannot
    # silently change the model used by the later parameter recovery step.
    axis = np.array(target.screen_first, dtype=float, copy=True)
    axis /= float(np.linalg.norm(axis))

    def coordinate(curve: AnalyticCurve3D, parameter: float) -> float:
        point = screen @ np.asarray(curve.point(parameter), dtype=float)
        return float(np.dot(point - target.screen_origin, axis))

    source_start = coordinate(source, source.domain.start)
    source_end = coordinate(source, source.domain.end)
    target_curve = target.curve
    target_start = coordinate(target_curve, target_curve.domain.start)
    target_end = coordinate(target_curve, target_curve.domain.end)
    lower = max(min(source_start, source_end), min(target_start, target_end))
    upper = min(max(source_start, source_end), max(target_start, target_end))
    if upper < lower - screen_epsilon:
        return ()
    if upper - lower > screen_epsilon:
        raise ProjectedCurveIntersectionError(
            f"curves {source.curve_id!r} and {target_curve.curve_id!r} have "
            "coincident projected support over a positive-length interval "
            "and therefore infinitely many crossings"
        )
    denominator = source_end - source_start
    if abs(denominator) <= screen_epsilon:
        raise ProjectedCurveIntersectionError(
            f"curve {source.curve_id!r} collapses on its projected line"
        )
    scalar = 0.5 * (lower + upper)
    ratio = (scalar - source_start) / denominator
    parameter = source.domain.start + ratio * source.domain.length
    parameter = min(source.domain.end, max(source.domain.start, parameter))
    return (float(parameter),)


def _same_ellipse_support_parameters(
    source: AnalyticCurve3D,
    source_model: _ProjectedModel,
    target: _ProjectedModel,
    view: ParallelView,
    *,
    parameter_epsilon: float,
) -> tuple[float, ...]:
    """Classify finite arc overlap on one shared projected ellipse."""

    if target.screen_second is None:
        raise ProjectedCurveIntersectionError("ellipse support has no second axis")
    screen = view.matrix[:2]
    source_start = source.domain.start
    start_point = screen @ np.asarray(source.point(source_start), dtype=float)
    target_linear = np.column_stack((target.screen_first, target.screen_second))
    target_coordinates = np.linalg.solve(
        target_linear,
        start_point - target.screen_origin,
    )
    target_angle = atan2(
        float(target_coordinates[1]),
        float(target_coordinates[0]),
    )
    target_tangent = (
        -np.sin(target_angle) * target.screen_first
        + np.cos(target_angle) * target.screen_second
    )
    source_tangent = screen @ np.asarray(source.tangent(source_start), dtype=float)
    orientation_dot = float(np.dot(source_tangent, target_tangent))
    orientation_scale = float(np.linalg.norm(source_tangent)) * float(
        np.linalg.norm(target_tangent)
    )
    if orientation_scale <= 0.0 or abs(orientation_dot) <= (
        8192.0 * _FLOAT_EPSILON * orientation_scale
    ):
        raise ProjectedCurveIntersectionError(
            "coincident ellipse parameter orientation is numerically ambiguous"
        )
    orientation = 1.0 if orientation_dot > 0.0 else -1.0
    source_end_angle = target_angle + orientation * source.domain.length
    source_lower = min(target_angle, source_end_angle)
    source_upper = max(target_angle, source_end_angle)

    target_curve = target.curve
    first_shift = floor((source_lower - target_curve.domain.end) / tau) - 1
    last_shift = ceil((source_upper - target_curve.domain.start) / tau) + 1
    point_angles: list[float] = []
    for shift in range(first_shift, last_shift + 1):
        target_lower = target_curve.domain.start + shift * tau
        target_upper = target_curve.domain.end + shift * tau
        lower = max(source_lower, target_lower)
        upper = min(source_upper, target_upper)
        if upper < lower - parameter_epsilon:
            continue
        if upper - lower > parameter_epsilon:
            raise ProjectedCurveIntersectionError(
                f"curves {source.curve_id!r} and {target_curve.curve_id!r} have "
                "coincident projected support over a positive-length arc and "
                "therefore infinitely many crossings"
            )
        point_angles.append(0.5 * (lower + upper))

    parameters: list[float] = []
    for angle in sorted(point_angles):
        parameter = source_start + (angle - target_angle) / orientation
        if parameter < source.domain.start - parameter_epsilon:
            continue
        if parameter > source.domain.end + parameter_epsilon:
            continue
        parameter = min(source.domain.end, max(source.domain.start, parameter))
        if not parameters or parameter - parameters[-1] > parameter_epsilon:
            parameters.append(float(parameter))
    if (
        getattr(source, "closed", False)
        and len(parameters) >= 2
        and abs(parameters[0] - source.domain.start) <= parameter_epsilon
        and abs(parameters[-1] - source.domain.end) <= parameter_epsilon
    ):
        parameters.pop()
    return tuple(parameters)


def _unbounded_parameter_on_model(
    model: _ProjectedModel,
    point: np.ndarray,
    *,
    screen_epsilon: float,
) -> float | None:
    if model.screen_second is None:
        raise ProjectedCurveIntersectionError("conic support has no second axis")
    linear = np.column_stack((model.screen_first, model.screen_second))
    minimum_scale = float(np.min(np.linalg.svd(linear, compute_uv=False)))
    if minimum_scale <= 0.0:
        raise ProjectedCurveIntersectionError(
            "unbounded conic support has a singular projection"
        )
    coordinate_epsilon = screen_epsilon / minimum_scale
    first, second = (
        float(item)
        for item in np.linalg.solve(linear, point - model.screen_origin)
    )
    if model.parameter_kind == "hyperbola":
        parameter = asinh(second)
        expected = model.branch_sign * np.cosh(parameter)
        if abs(first - expected) > coordinate_epsilon * max(1.0, abs(expected)):
            return None
        return float(parameter)
    if model.parameter_kind == "parabola":
        expected = first * first
        if abs(second - expected) > coordinate_epsilon * max(
            1.0, abs(second), expected
        ):
            return None
        return float(first)
    raise ProjectedCurveIntersectionError(
        "unbounded support parameter requested for an unsupported conic"
    )


def _same_unbounded_conic_support_parameters(
    source: AnalyticCurve3D,
    target: _ProjectedModel,
    view: ParallelView,
    *,
    parameter_epsilon: float,
    screen_epsilon: float,
) -> tuple[float, ...]:
    """Classify finite overlap on one parabola or one hyperbola branch."""

    screen = view.matrix[:2]
    mapped: list[float | None] = []
    for parameter in (source.domain.start, source.domain.end):
        point = screen @ np.asarray(source.point(parameter), dtype=float)
        mapped.append(
            _unbounded_parameter_on_model(
                target,
                point,
                screen_epsilon=screen_epsilon,
            )
        )
    if mapped[0] is None and mapped[1] is None:
        # The two finite branches of a non-degenerate hyperbola are disjoint.
        if target.parameter_kind == "hyperbola":
            return ()
        raise ProjectedCurveIntersectionError(
            "coincident parabola parameter mapping is numerically ambiguous"
        )
    if mapped[0] is None or mapped[1] is None:
        raise ProjectedCurveIntersectionError(
            "coincident conic branch mapping changes support branch"
        )
    mapped_start = float(mapped[0])
    mapped_end = float(mapped[1])
    mapped_lower = min(mapped_start, mapped_end)
    mapped_upper = max(mapped_start, mapped_end)
    target_curve = target.curve
    lower = max(mapped_lower, target_curve.domain.start)
    upper = min(mapped_upper, target_curve.domain.end)
    if upper < lower - parameter_epsilon:
        return ()
    if upper - lower > parameter_epsilon:
        raise ProjectedCurveIntersectionError(
            f"curves {source.curve_id!r} and {target_curve.curve_id!r} have "
            "coincident projected support over a positive-length conic branch "
            "and therefore infinitely many crossings"
        )
    denominator = mapped_end - mapped_start
    if abs(denominator) <= parameter_epsilon:
        raise ProjectedCurveIntersectionError(
            "coincident conic parameter mapping collapsed"
        )
    target_parameter = 0.5 * (lower + upper)
    ratio = (target_parameter - mapped_start) / denominator
    source_parameter = source.domain.start + ratio * source.domain.length
    source_parameter = min(
        source.domain.end,
        max(source.domain.start, source_parameter),
    )
    return (float(source_parameter),)


def _same_support_source_parameters(
    source: AnalyticCurve3D,
    source_model: _ProjectedModel,
    target: _ProjectedModel,
    view: ParallelView,
    *,
    parameter_epsilon: float,
    screen_epsilon: float,
) -> tuple[float, ...]:
    if source_model.parameter_kind == target.parameter_kind == "line":
        return _same_line_support_parameters(
            source,
            target,
            view,
            parameter_epsilon=parameter_epsilon,
            screen_epsilon=screen_epsilon,
        )
    if source_model.parameter_kind == target.parameter_kind == "ellipse":
        return _same_ellipse_support_parameters(
            source,
            source_model,
            target,
            view,
            parameter_epsilon=parameter_epsilon,
        )
    if source_model.parameter_kind == target.parameter_kind and target.parameter_kind in {
        "hyperbola",
        "parabola",
    }:
        return _same_unbounded_conic_support_parameters(
            source,
            target,
            view,
            parameter_epsilon=parameter_epsilon,
            screen_epsilon=screen_epsilon,
        )
    raise ProjectedCurveIntersectionError(
        f"curves {source.curve_id!r} and {target.curve.curve_id!r} have "
        "coincident projected support that cannot be finitely isolated"
    )


def _direct_line_midpoint_tangency_parameters(
    source: AnalyticCurve3D,
    source_model: _ProjectedModel,
    target: _ProjectedModel,
    view: ParallelView,
    *,
    parameter_epsilon: float,
    screen_epsilon: float,
) -> tuple[float, ...]:
    """Recover an authored conic/line tangent from its exact segment midpoint.

    Very distant hyperbola tangents lose the exponentially small coefficient
    that distinguishes ``cosh(t)`` from ``sinh(t)`` when expanded into one
    global polynomial.  The finite segment still stores its midpoint exactly;
    mapping that screen point through the conic's own local coordinates keeps
    the authored parameter well conditioned.
    """

    if (source_model.line is None) == (target.line is None):
        return ()
    candidates: tuple[float, ...]
    if isinstance(source, SegmentCurve):
        candidates = (source.domain.midpoint,)
    elif isinstance(target.curve, SegmentCurve):
        midpoint_world = np.asarray(
            target.curve.point(target.curve.domain.midpoint), dtype=float
        )
        midpoint = view.matrix[:2] @ midpoint_world
        candidates = _target_parameters(
            source_model,
            midpoint,
            parameter_epsilon=parameter_epsilon,
            screen_epsilon=screen_epsilon,
            world_point=midpoint_world,
            view=view,
        )
    else:
        return ()
    certified = [
        _canonical_curve_parameter(source, parameter, parameter_epsilon)
        for parameter in candidates
        if _stationary_is_projected_tangency(
            source,
            source_model,
            target,
            view,
            parameter,
            parameter_epsilon=parameter_epsilon,
            screen_epsilon=screen_epsilon,
        )
    ]
    result: list[float] = []
    for parameter in sorted(certified):
        if not result or parameter - result[-1] > parameter_epsilon:
            result.append(parameter)
    return tuple(result)


def _candidate_source_parameters(
    source: AnalyticCurve3D,
    source_model: _ProjectedModel,
    target: _ProjectedModel,
    view: ParallelView,
    *,
    context: ResolvedGeometryContext,
    parameter_epsilon: float,
    screen_epsilon: float,
) -> tuple[_SourceParameterCandidate, ...]:
    if _same_projected_support(
        source_model,
        target,
        screen_epsilon=screen_epsilon,
    ):
        return tuple(
            _SourceParameterCandidate(parameter)
            for parameter in _same_support_source_parameters(
                source,
                source_model,
                target,
                view,
                parameter_epsilon=parameter_epsilon,
                screen_epsilon=screen_epsilon,
            )
        )
    direct_tangency = _direct_line_midpoint_tangency_parameters(
        source,
        source_model,
        target,
        view,
        parameter_epsilon=parameter_epsilon,
        screen_epsilon=screen_epsilon,
    )
    if direct_tangency:
        return tuple(
            _SourceParameterCandidate(parameter, tangential_certified=True)
            for parameter in direct_tangency
        )
    chart, polynomial = _chart_polynomial(source, target, view)
    coefficients = [float(item) for item in polynomial.coef]
    coefficient_scale = max((abs(item) for item in coefficients), default=0.0)
    if not isfinite(coefficient_scale):
        raise ProjectedCurveIntersectionError(
            "projected crossing equation contains non-finite coefficients"
        )
    # A tan-half-angle chart represents t=pi at infinity.  If that authored
    # seam is objectively on the target support, the polynomial's formal
    # leading coefficient is mathematically zero.  Clear only that one
    # cancellation-prone coefficient.  A blanket relative trim would destroy
    # legitimate small coefficients in an exp chart such as
    # 0.5*y**2-cosh(T)*y+0.5 for a distant hyperbola crossing.
    if chart.name == "tan_half_angle" and chart.chart_poles:
        expected_degree = 2 if target.line is not None else 4
        while len(coefficients) <= expected_degree:
            coefficients.append(0.0)
        screen = view.matrix[:2]
        for seam in chart.chart_poles:
            seam_point = screen @ np.asarray(source.point(seam), dtype=float)
            if _direct_equation_matches(
                target,
                seam_point,
                screen_epsilon=screen_epsilon,
            ):
                coefficients[expected_degree] = 0.0
                break
    coefficients = tuple(coefficients)
    if not any(item != 0.0 for item in coefficients):
        raise ProjectedCurveIntersectionError(
            f"curves {source.curve_id!r} and {target.curve.curve_id!r} have "
            "a numerically indistinguishable projected crossing equation"
        )
    # A mathematically repeated crossing can be perturbed into two nearby
    # simple polynomial roots by the finite-precision chart coefficients.
    # Certify the derivative stationary point against both the normalized
    # crossing residual and the original projected geometry.  Once certified,
    # deflate its double algebraic factor before solving the remaining roots;
    # this avoids both a fake split pair and loss of unrelated intersections.
    # A genuine nearby secant has a non-negligible stationary residual and is
    # retained as two crossings.
    degree = len(coefficients) - 1
    stationary_residual_limit = max(
        (
            _STATIONARY_RESIDUAL_FACTOR
            * _FLOAT_EPSILON
            * max(1, degree + 1)
        ),
        _pair_tangency_epsilon(source_model, target),
    )
    certified_stationaries: list[tuple[float, tuple[float, ...]]] = []
    for stationary in _stationary_chart_values(
        coefficients,
        chart,
        context=context,
        parameter_epsilon=parameter_epsilon,
    ):
        residual = _normalized_polynomial_residual(coefficients, stationary)
        stationary_parameters = chart.parameters(stationary, parameter_epsilon)
        local_residual_limit = max(
            (
                _stationary_contact_ulp_residual_limit(
                    source,
                    source_model,
                    target,
                    view,
                    parameter,
                    parameter_epsilon=parameter_epsilon,
                    screen_epsilon=screen_epsilon,
                )
                for parameter in stationary_parameters
            ),
            default=0.0,
        )
        if residual > max(stationary_residual_limit, local_residual_limit):
            continue
        certified_parameters = tuple(
            parameter
            for parameter in stationary_parameters
            if _stationary_is_projected_tangency(
                source,
                source_model,
                target,
                view,
                parameter,
                parameter_epsilon=parameter_epsilon,
                screen_epsilon=screen_epsilon,
            )
        )
        if not certified_parameters:
            continue
        if any(
            abs(stationary - previous[0])
            <= parameter_epsilon * max(1.0, abs(stationary))
            for previous in certified_stationaries
        ):
            continue
        certified_stationaries.append((float(stationary), certified_parameters))

    solve_coefficients = coefficients
    for stationary, _parameters in certified_stationaries:
        solve_coefficients = _deflate_polynomial_root(
            _deflate_polynomial_root(solve_coefficients, stationary),
            stationary,
        )

    root_entries: list[tuple[float, float, bool]] = []
    try:
        if len(solve_coefficients) > 1:
            if chart.name == "exp":
                roots = solve_real_polynomial_exp_chart(
                    solve_coefficients,
                    parameter_domain=source.domain,
                    context=context,
                    parameter_tolerance=parameter_epsilon,
                )
                root_entries.extend(
                    (item.chart_root.value, item.parameter, False) for item in roots
                )
            else:
                roots = solve_real_polynomial(
                    solve_coefficients,
                    domain=chart.root_domain,
                    context=context,
                    parameter_tolerance=(
                        parameter_epsilon
                        if chart.name == "parameter"
                        else max(4096.0 * _FLOAT_EPSILON, parameter_epsilon)
                    ),
                )
                root_entries.extend(
                    (root.value, parameter, False)
                    for root in roots
                    for parameter in chart.parameters(
                        root.value,
                        parameter_epsilon,
                    )
                )
    except (OverflowError, PolynomialRootError) as exc:
        raise ProjectedCurveIntersectionError(
            f"projected crossings for {source.curve_id!r} and "
            f"{target.curve.curve_id!r} are ambiguous: {exc}"
        ) from exc
    root_entries.extend(
        (stationary, float(parameter), True)
        for stationary, stationary_parameters in certified_stationaries
        for parameter in stationary_parameters
    )

    parameter_entries = [
        (parameter, tangential_certified)
        for _root, parameter, tangential_certified in root_entries
    ]
    # Rational chart poles and authored endpoints are checked directly.  This
    # retains a legitimate root at tan(t/2)=infinity and avoids seam loss.
    # Ordinary authored endpoints are already part of the finite root domain
    # and are validated by the polynomial solver.  Only rational chart poles
    # are absent from that finite domain and require a direct limit check.
    candidates = chart.chart_poles
    screen = view.matrix[:2]
    for parameter in candidates:
        if not source.domain.contains(parameter, tolerance=parameter_epsilon):
            continue
        point = screen @ np.asarray(source.point(parameter), dtype=float)
        if _direct_equation_matches(
            target,
            point,
            screen_epsilon=screen_epsilon,
        ):
            parameter_entries.append((float(parameter), False))

    normalized: list[_SourceParameterCandidate] = []
    for parameter, tangential_certified in parameter_entries:
        if parameter < source.domain.start - parameter_epsilon:
            continue
        if parameter > source.domain.end + parameter_epsilon:
            continue
        parameter = _canonical_curve_parameter(
            source,
            parameter,
            parameter_epsilon,
        )
        normalized.append(
            _SourceParameterCandidate(parameter, tangential_certified)
        )
    normalized.sort(key=lambda item: item.parameter)
    ordered: list[_SourceParameterCandidate] = []
    for candidate in normalized:
        if (
            ordered
            and candidate.parameter - ordered[-1].parameter <= parameter_epsilon
        ):
            if candidate.tangential_certified and not ordered[-1].tangential_certified:
                ordered[-1] = _SourceParameterCandidate(
                    ordered[-1].parameter,
                    tangential_certified=True,
                )
            continue
        ordered.append(candidate)
    return tuple(ordered)


def _active_domains(
    curves: Sequence[AnalyticCurve3D],
    active_intervals: Mapping[str, Sequence[ParameterInterval]] | None,
) -> dict[str, tuple[ParameterInterval, ...]]:
    curve_map = {curve.curve_id: curve for curve in curves}
    if active_intervals is None:
        return {curve.curve_id: (curve.domain,) for curve in curves}
    unknown = set(active_intervals) - set(curve_map)
    if unknown:
        raise ProjectedCurveIntersectionError(
            "active_intervals references unknown curves: "
            + ", ".join(sorted(unknown))
        )
    result: dict[str, tuple[ParameterInterval, ...]] = {}
    for curve in curves:
        values = tuple(active_intervals.get(curve.curve_id, (curve.domain,)))
        if not all(isinstance(item, ParameterInterval) for item in values):
            raise TypeError("active_intervals must contain ParameterInterval objects")
        ordered = tuple(sorted(values, key=lambda item: (item.start, item.end)))
        previous: ParameterInterval | None = None
        for interval in ordered:
            if interval.length <= 0.0:
                raise ProjectedCurveIntersectionError(
                    "active curve intervals must have positive length"
                )
            if (
                interval.start < curve.domain.start
                or interval.end > curve.domain.end
            ):
                raise ProjectedCurveIntersectionError(
                    f"active interval lies outside curve {curve.curve_id!r} domain"
                )
            if previous is not None and interval.start < previous.end:
                raise ProjectedCurveIntersectionError(
                    f"active intervals for curve {curve.curve_id!r} overlap"
                )
            previous = interval
        result[curve.curve_id] = ordered
    return result


def _curve_interval(
    curve: AnalyticCurve3D,
    interval: ParameterInterval,
) -> AnalyticCurve3D:
    if interval == curve.domain:
        return curve
    if isinstance(curve, SegmentCurve):
        return SegmentCurve(
            curve.curve_id,
            curve.point(interval.start),
            curve.point(interval.end),
            domain=interval,
        )
    if isinstance(curve, EllipseArcCurve):
        return EllipseArcCurve(
            curve.curve_id,
            curve.center,
            curve.first_axis,
            curve.second_axis,
            domain=interval,
        )
    return ParametricConicBranch(
        curve.curve_id,
        curve.parameterization,
        curve.plane_embedding,
        interval,
    )


def _projected_interval_bounds(
    curve: AnalyticCurve3D,
    view: ParallelView,
) -> tuple[float, float, float, float]:
    """Return an outward-rounded exact AABB for one finite curve interval.

    The supported parameterizations have closed-form coordinate extrema.  No
    sampled polyline is used here: a broad-phase rejection is allowed only
    when these certified bounds are separated by more than the pair's screen
    tolerance.
    """

    candidates = {float(curve.domain.start), float(curve.domain.end)}
    screen = view.matrix[:2]

    if isinstance(curve, SegmentCurve):
        pass
    else:
        if isinstance(curve, EllipseArcCurve):
            first = screen @ np.asarray(curve.first_axis, dtype=float)
            second = screen @ np.asarray(curve.second_axis, dtype=float)
            kind = ConicKind.ELLIPSE
            branch_sign = 1
        else:
            _origin, world_first, world_second = _world_branch_geometry(curve)
            first = screen @ world_first
            second = screen @ world_second
            kind = curve.parameterization.kind
            branch_sign = curve.parameterization.branch_sign

        if kind in {ConicKind.CIRCLE, ConicKind.ELLIPSE}:
            for coordinate in range(2):
                first_value = float(first[coordinate])
                second_value = float(second[coordinate])
                if first_value == 0.0 and second_value == 0.0:
                    continue
                base = atan2(second_value, first_value)
                candidates.update(
                    _parameters_in_angular_domain(base, curve, 0.0)
                )
                candidates.update(
                    _parameters_in_angular_domain(base + 0.5 * tau, curve, 0.0)
                )
        elif kind is ConicKind.HYPERBOLA:
            for coordinate in range(2):
                denominator = branch_sign * float(first[coordinate])
                numerator = -float(second[coordinate])
                if denominator == 0.0:
                    continue
                ratio = numerator / denominator
                if abs(ratio) < 1.0:
                    parameter = atanh(ratio)
                    if curve.domain.start <= parameter <= curve.domain.end:
                        candidates.add(float(parameter))
        elif kind is ConicKind.PARABOLA:
            for coordinate in range(2):
                quadratic = float(second[coordinate])
                if quadratic == 0.0:
                    continue
                parameter = -float(first[coordinate]) / (2.0 * quadratic)
                if curve.domain.start <= parameter <= curve.domain.end:
                    candidates.add(float(parameter))

    points = np.asarray(
        [
            screen @ np.asarray(curve.point(parameter), dtype=float)
            for parameter in sorted(candidates)
        ],
        dtype=float,
    )
    minimum = np.nextafter(np.min(points, axis=0), -np.inf)
    maximum = np.nextafter(np.max(points, axis=0), np.inf)
    return (
        float(minimum[0]),
        float(maximum[0]),
        float(minimum[1]),
        float(maximum[1]),
    )


def _projected_bounds_are_disjoint(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    *,
    screen_epsilon: float,
) -> bool:
    return bool(
        first[1] + screen_epsilon < second[0]
        or second[1] + screen_epsilon < first[0]
        or first[3] + screen_epsilon < second[2]
        or second[3] + screen_epsilon < first[2]
    )


def compute_projected_curve_crossings(
    curves: Sequence[AnalyticCurve3D],
    view: ParallelView,
    *,
    context: GeometryContext | ResolvedGeometryContext | None = None,
    active_intervals: Mapping[str, Sequence[ParameterInterval]] | None = None,
) -> tuple[ProjectedCurveCrossing, ...]:
    """Return isolated projected crossings on the requested painted domains.

    ``active_intervals`` lets a paint policy exclude mathematically present
    but unpainted spans before coincident-support classification.  Missing
    mapping keys default to the complete authored domain; an explicit empty
    tuple means that curve has no active paint in this solve.
    """

    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    items = tuple(curves)
    if not all(
        isinstance(item, (SegmentCurve, EllipseArcCurve, ParametricConicBranch))
        for item in items
    ):
        raise TypeError("curves must contain supported analytic curves")
    ids = tuple(item.curve_id for item in items)
    if len(set(ids)) != len(ids):
        raise ProjectedCurveIntersectionError("curve identities must be unique")
    items = tuple(sorted(items, key=lambda item: item.curve_id))
    domains = _active_domains(items, active_intervals)
    if len(items) < 2:
        return ()
    resolved = _resolve_context(items, context)
    parameter_epsilon = resolved.epsilon(GeometryQuantity.PARAMETER)
    screen_matrix = view.matrix[:2]
    depth_row = view.matrix[2]
    boundary_epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    world_joint_screen_epsilon = boundary_epsilon * max(
        1.0,
        float(np.linalg.norm(screen_matrix, ord=2)),
    )
    result: list[ProjectedCurveCrossing] = []
    projected_interval_cache: dict[
        tuple[str, float, float],
        tuple[
            AnalyticCurve3D,
            _ProjectedModel,
            tuple[float, float, float, float],
        ],
    ] = {}

    def projected_interval(
        curve: AnalyticCurve3D,
        interval: ParameterInterval,
    ) -> tuple[
        AnalyticCurve3D,
        _ProjectedModel,
        tuple[float, float, float, float],
    ]:
        key = (curve.curve_id, float(interval.start), float(interval.end))
        cached = projected_interval_cache.get(key)
        if cached is not None:
            return cached
        active = _curve_interval(curve, interval)
        prepared = (
            active,
            _projected_model(active, view),
            _projected_interval_bounds(active, view),
        )
        projected_interval_cache[key] = prepared
        return prepared

    for first_index, first_curve in enumerate(items):
        for second_curve in items[first_index + 1 :]:
            pair_entries: list[
                tuple[
                    float,
                    float,
                    np.ndarray,
                    float,
                    float,
                    bool,
                    float,
                    bool,
                    float,
                ]
            ] = []

            def authored_open_endpoint(
                curve: AnalyticCurve3D,
                parameter: float,
            ) -> bool:
                if bool(getattr(curve, "closed", False)):
                    return False
                return (
                    abs(parameter - curve.domain.start) <= parameter_epsilon
                    or abs(parameter - curve.domain.end) <= parameter_epsilon
                )

            for first_interval in domains[first_curve.curve_id]:
                first_active, first_model, first_bounds = projected_interval(
                    first_curve,
                    first_interval,
                )
                for second_interval in domains[second_curve.curve_id]:
                    second_active, second_model, second_bounds = projected_interval(
                        second_curve,
                        second_interval,
                    )
                    screen_epsilon = _pair_screen_epsilon(
                        first_model,
                        second_model,
                        view,
                        resolved,
                    )
                    if _projected_bounds_are_disjoint(
                        first_bounds,
                        second_bounds,
                        screen_epsilon=screen_epsilon,
                    ):
                        continue
                    local_depth_epsilon = _pair_depth_epsilon(
                        first_active,
                        second_active,
                        depth_row,
                        resolved,
                    )
                    tangency_epsilon = _pair_tangency_epsilon(
                        first_model,
                        second_model,
                    )
                    first_parameters = _candidate_source_parameters(
                        first_active,
                        first_model,
                        second_model,
                        view,
                        context=resolved,
                        parameter_epsilon=parameter_epsilon,
                        screen_epsilon=screen_epsilon,
                    )
                    for first_candidate in first_parameters:
                        first_parameter = first_candidate.parameter
                        first_world = np.asarray(
                            first_active.point(first_parameter), dtype=float
                        )
                        screen_point = screen_matrix @ first_world
                        if not _direct_equation_matches(
                            second_model,
                            screen_point,
                            screen_epsilon=screen_epsilon,
                        ):
                            continue
                        second_parameters = _target_parameters(
                            second_model,
                            screen_point,
                            parameter_epsilon=parameter_epsilon,
                            screen_epsilon=screen_epsilon,
                            world_point=first_world,
                            view=view,
                        )
                        for second_parameter in second_parameters:
                            second_world = np.asarray(
                                second_active.point(second_parameter), dtype=float
                            )
                            first_tangent = screen_matrix @ np.asarray(
                                first_active.tangent(first_parameter), dtype=float
                            )
                            second_tangent = screen_matrix @ np.asarray(
                                second_active.tangent(second_parameter), dtype=float
                            )
                            tangent_scale = float(
                                np.linalg.norm(first_tangent)
                            ) * float(np.linalg.norm(second_tangent))
                            if tangent_scale <= 0.0:
                                rank_one_turn = (
                                    first_model.parameter_kind
                                    == "ellipse_rank_one"
                                    and float(np.linalg.norm(first_tangent)) == 0.0
                                    and float(np.linalg.norm(second_tangent)) > 0.0
                                ) or (
                                    second_model.parameter_kind
                                    == "ellipse_rank_one"
                                    and float(np.linalg.norm(second_tangent)) == 0.0
                                    and float(np.linalg.norm(first_tangent)) > 0.0
                                )
                                if not rank_one_turn:
                                    raise ProjectedCurveIntersectionError(
                                        "a projected curve tangent collapsed at a crossing"
                                    )
                                tangential = True
                            else:
                                tangent_cross = float(
                                    first_tangent[0] * second_tangent[1]
                                    - first_tangent[1] * second_tangent[0]
                                )
                                tangential = (
                                    first_candidate.tangential_certified
                                    or abs(tangent_cross)
                                    <= tangency_epsilon * tangent_scale
                                )
                            pair_entries.append(
                                (
                                    first_parameter,
                                    second_parameter,
                                    screen_point,
                                    float(depth_row @ first_world),
                                    float(depth_row @ second_world),
                                    tangential,
                                    local_depth_epsilon,
                                    float(
                                        np.linalg.norm(first_world - second_world)
                                    )
                                    <= boundary_epsilon
                                    and (
                                        authored_open_endpoint(
                                            first_curve, first_parameter
                                        )
                                        or authored_open_endpoint(
                                            second_curve, second_parameter
                                        )
                                    ),
                                    screen_epsilon,
                                )
                            )

            pair_entries.sort(key=lambda item: (item[0], item[1]))
            deduped: list[
                tuple[
                    float,
                    float,
                    np.ndarray,
                    float,
                    float,
                    bool,
                    float,
                    bool,
                    float,
                ]
            ] = []

            def equivalent_parameter(
                curve: AnalyticCurve3D,
                first: float,
                second: float,
            ) -> bool:
                if abs(first - second) <= parameter_epsilon:
                    return True
                return bool(
                    getattr(curve, "closed", False)
                    and (
                        (
                            abs(first - curve.domain.start) <= parameter_epsilon
                            and abs(second - curve.domain.end) <= parameter_epsilon
                        )
                        or (
                            abs(second - curve.domain.start) <= parameter_epsilon
                            and abs(first - curve.domain.end) <= parameter_epsilon
                        )
                    )
                )

            for entry in pair_entries:
                if any(
                    (
                        equivalent_parameter(first_curve, entry[0], previous[0])
                        and equivalent_parameter(
                            second_curve, entry[1], previous[1]
                        )
                    )
                    or (
                        entry[7]
                        and previous[7]
                        and float(np.linalg.norm(entry[2] - previous[2]))
                        <= max(
                            entry[8],
                            previous[8],
                            world_joint_screen_epsilon,
                        )
                    )
                    for previous in deduped
                ):
                    continue
                deduped.append(entry)
            for crossing_index, entry in enumerate(deduped):
                (
                    first_parameter,
                    second_parameter,
                    point,
                    first_depth,
                    second_depth,
                    tangential,
                    depth_epsilon_base,
                    world_coincident,
                    _screen_epsilon,
                ) = entry
                difference = first_depth - second_depth
                # A large common world/depth translation must not turn a
                # still-resolvable separation into an invented 3D
                # intersection.  Use the actual local floating-point spacing
                # instead of a relative tolerance proportional to the
                # absolute depth coordinate.
                machine_epsilon = 4.0 * max(
                    abs(float(np.spacing(first_depth))),
                    abs(float(np.spacing(second_depth))),
                    abs(float(np.spacing(max(abs(first_depth), abs(second_depth))))),
                )
                if world_coincident or abs(difference) <= depth_epsilon_base:
                    far_curve_id = near_curve_id = None
                elif abs(difference) <= machine_epsilon:
                    raise ProjectedCurveIntersectionError(
                        "crossing depth order is numerically indistinguishable "
                        "after the authored depth translation"
                    )
                elif difference < 0.0:
                    far_curve_id = first_curve.curve_id
                    near_curve_id = second_curve.curve_id
                else:
                    far_curve_id = second_curve.curve_id
                    near_curve_id = first_curve.curve_id
                result.append(
                    ProjectedCurveCrossing(
                        crossing_id=_crossing_identity(
                            first_curve.curve_id,
                            second_curve.curve_id,
                            crossing_index,
                        ),
                        first_curve_id=first_curve.curve_id,
                        second_curve_id=second_curve.curve_id,
                        first_parameter=first_parameter,
                        second_parameter=second_parameter,
                        screen_point=(float(point[0]), float(point[1])),
                        first_depth=first_depth,
                        second_depth=second_depth,
                        far_curve_id=far_curve_id,
                        near_curve_id=near_curve_id,
                        tangential=tangential,
                    )
                )
    return tuple(sorted(result, key=lambda item: item.crossing_id))


def canonical_projected_curve_crossings_json(
    crossings: Sequence[ProjectedCurveCrossing],
) -> str:
    items = tuple(crossings)
    if not all(isinstance(item, ProjectedCurveCrossing) for item in items):
        raise TypeError("crossings must contain ProjectedCurveCrossing objects")
    ids = tuple(item.crossing_id for item in items)
    if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
        raise ProjectedCurveIntersectionError(
            "crossings must have unique canonical identities"
        )
    return json.dumps(
        [item.to_dict() for item in items],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "PROJECTED_CURVE_CROSSING_SCHEMA",
    "ProjectedCurveCrossing",
    "ProjectedCurveIntersectionError",
    "canonical_projected_curve_crossings_json",
    "compute_projected_curve_crossings",
]
