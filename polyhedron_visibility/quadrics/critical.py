"""Analytic critical events for curve/quadric visibility.

No visibility boundary is found by dense sampling.  Segment, line, and
parabola branches produce ordinary low-degree polynomials.  Ellipse and circle
arcs use ``z = tan(t / 2)`` and hyperbola branches use ``y = exp(t)``; after
clearing denominators every supported event is at most quartic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan, ceil, exp, floor, isfinite, log, pi, tan, tau
from typing import Sequence

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
from .contract import ConeSpec, CylinderSpec, SphereSpec
from .curves import (
    EllipseArcCurve,
    ParametricConicBranch,
    SegmentCurve,
)
from .roots import (
    PolynomialRootError,
    solve_real_polynomial,
    solve_real_polynomial_exp_chart,
)


CRITICAL_EVENT_SCHEMA = "manim-quadric-critical-event/v1"
_FLOAT_EPSILON = float(np.finfo(float).eps)
_IDENTITY_FACTOR = 2048.0


QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec
AnalyticCurve3D = SegmentCurve | EllipseArcCurve | ParametricConicBranch
ContextInput = GeometryContext | ResolvedGeometryContext | None


class CriticalEventError(ValueError):
    """Analytic critical events cannot be constructed without guessing."""


class CriticalEventKind(str, Enum):
    DOMAIN_ENDPOINT = "domain_endpoint"
    CHART_SEAM = "chart_seam"
    SUPPORT_TANGENCY = "support_tangency"
    CURVE_SURFACE_INTERSECTION = "curve_surface_intersection"
    SELF_OCCLUSION_SWITCH = "self_occlusion_switch"
    AXIAL_BOUNDARY = "axial_boundary"
    CAP_RIM = "cap_rim"


@dataclass(frozen=True, slots=True)
class CriticalEvidence:
    """One exact equation responsible for one critical parameter."""

    kind: CriticalEventKind
    equation: str
    surface_id: str | None = None
    chart: str = "parameter"
    coefficients: tuple[float, ...] = ()
    root_value: float | None = None
    multiplicity: int = 1
    residual: float = 0.0
    identically_zero: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CriticalEventKind):
            raise TypeError("kind must be a CriticalEventKind")
        if not isinstance(self.equation, str) or not self.equation:
            raise ValueError("critical equation must be a non-empty string")
        if self.surface_id is not None and (
            not isinstance(self.surface_id, str) or not self.surface_id
        ):
            raise ValueError("surface_id must be a non-empty string when present")
        if not isinstance(self.chart, str) or not self.chart:
            raise ValueError("critical chart must be a non-empty string")
        coefficients = tuple(float(item) for item in self.coefficients)
        if not all(isfinite(item) for item in coefficients):
            raise ValueError("critical coefficients must be finite")
        root = None if self.root_value is None else float(self.root_value)
        if root is not None and not isfinite(root):
            raise ValueError("critical root_value must be finite")
        if isinstance(self.multiplicity, bool) or not isinstance(self.multiplicity, int):
            raise TypeError("critical multiplicity must be an integer")
        if self.multiplicity <= 0:
            raise ValueError("critical multiplicity must be positive")
        residual = float(self.residual)
        if not isfinite(residual) or residual < 0.0:
            raise ValueError("critical residual must be finite and non-negative")
        if not isinstance(self.identically_zero, bool):
            raise TypeError("identically_zero must be boolean")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "root_value", root)
        object.__setattr__(self, "residual", residual)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "equation": self.equation,
            "surfaceId": self.surface_id,
            "chart": self.chart,
            "coefficients": list(self.coefficients),
            "rootValue": self.root_value,
            "multiplicity": self.multiplicity,
            "residual": self.residual,
            "identicallyZero": self.identically_zero,
        }


@dataclass(frozen=True, slots=True)
class CriticalEvent:
    """All analytic evidence clustered at one curve parameter."""

    parameter: float
    evidence: tuple[CriticalEvidence, ...]
    schema: str = CRITICAL_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CRITICAL_EVENT_SCHEMA:
            raise ValueError("invalid critical-event schema")
        parameter = float(self.parameter)
        if not isfinite(parameter):
            raise ValueError("critical parameter must be finite")
        if not self.evidence or not all(
            isinstance(item, CriticalEvidence) for item in self.evidence
        ):
            raise ValueError("a critical event requires evidence")
        ordered = tuple(sorted(self.evidence, key=_evidence_key))
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "evidence", ordered)

    @property
    def kinds(self) -> tuple[CriticalEventKind, ...]:
        return tuple(dict.fromkeys(item.kind for item in self.evidence))

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item.surface_id
                for item in self.evidence
                if item.surface_id is not None
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "parameter": self.parameter,
            "kinds": [item.value for item in self.kinds],
            "surfaceIds": list(self.surface_ids),
            "evidence": [item.to_dict() for item in self.evidence],
        }


def _evidence_key(item: CriticalEvidence) -> tuple[object, ...]:
    return (
        "" if item.surface_id is None else item.surface_id,
        item.kind.value,
        item.equation,
        item.chart,
        item.root_value if item.root_value is not None else float("-inf"),
    )


@dataclass(frozen=True, slots=True)
class _CurveChart:
    name: str
    numerator: tuple[Polynomial, Polynomial, Polynomial]
    denominator: Polynomial
    curve_domain: ParameterInterval
    root_domain: ParameterInterval | None

    @property
    def homogeneous_scale(self) -> float:
        values = [
            *(float(abs(item)) for polynomial in self.numerator for item in polynomial.coef),
            *(float(abs(item)) for item in self.denominator.coef),
        ]
        return max(values, default=1.0)

    @property
    def chart_poles(self) -> tuple[float, ...]:
        if self.name != "tan_half_angle":
            return ()
        start, end = self.curve_domain.start, self.curve_domain.end
        first = ceil((start - pi) / tau)
        last = floor((end - pi) / tau)
        return tuple(
            pi + index * tau
            for index in range(first, last + 1)
            if start <= pi + index * tau <= end
        )

    @property
    def chart_seams(self) -> tuple[float, ...]:
        return tuple(
            parameter
            for parameter in self.chart_poles
            if self.curve_domain.start < parameter < self.curve_domain.end
        )

    def parameters(self, root: float, epsilon: float) -> tuple[float, ...]:
        if self.name == "parameter":
            candidates = (float(root),)
        elif self.name == "exp":
            if root <= 0.0:
                return ()
            candidates = (log(root),)
        elif self.name == "tan_half_angle":
            base = 2.0 * atan(root)
            lower = floor((self.curve_domain.start - base) / tau) - 1
            upper = ceil((self.curve_domain.end - base) / tau) + 1
            candidates = tuple(base + index * tau for index in range(lower, upper + 1))
        else:  # pragma: no cover - private construction is exhaustive
            raise CriticalEventError(f"unsupported curve chart {self.name!r}")

        result: list[float] = []
        for candidate in sorted(candidates):
            if candidate < self.curve_domain.start - epsilon:
                continue
            if candidate > self.curve_domain.end + epsilon:
                continue
            if candidate < self.curve_domain.start:
                candidate = self.curve_domain.start
            elif candidate > self.curve_domain.end:
                candidate = self.curve_domain.end
            if not result or candidate - result[-1] > epsilon:
                result.append(float(candidate))
        return tuple(result)


def _poly_vector(
    coefficients_by_power: Sequence[Sequence[float]],
) -> tuple[Polynomial, Polynomial, Polynomial]:
    powers = [np.asarray(item, dtype=float) for item in coefficients_by_power]
    if not powers or any(item.shape != (3,) or not np.all(np.isfinite(item)) for item in powers):
        raise CriticalEventError("curve polynomial coefficients must be finite 3D vectors")
    return tuple(
        Polynomial(tuple(float(power[axis]) for power in powers))
        for axis in range(3)
    )  # type: ignore[return-value]


def _embedded_branch_geometry(
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


def _curve_chart(curve: AnalyticCurve3D) -> _CurveChart:
    if isinstance(curve, SegmentCurve):
        start = np.asarray(curve.start, dtype=float)
        displacement = np.asarray(curve.displacement, dtype=float)
        slope = displacement / curve.domain.length
        constant = start - curve.domain.start * slope
        return _CurveChart(
            "parameter",
            _poly_vector((constant, slope)),
            Polynomial((1.0,)),
            curve.domain,
            curve.domain,
        )

    if isinstance(curve, EllipseArcCurve):
        center = np.asarray(curve.center, dtype=float)
        first = np.asarray(curve.first_axis, dtype=float)
        second = np.asarray(curve.second_axis, dtype=float)
        numerator = _poly_vector((center + first, 2.0 * second, center - first))
        return _CurveChart(
            "tan_half_angle",
            numerator,
            Polynomial((1.0, 0.0, 1.0)),
            curve.domain,
            None,
        )

    if not isinstance(curve, ParametricConicBranch):
        raise TypeError("curve must be a supported analytic 3D curve")
    branch = curve.parameterization
    origin, first, second = _embedded_branch_geometry(curve)
    if branch.kind in {ConicKind.CIRCLE, ConicKind.ELLIPSE}:
        numerator = _poly_vector((origin + first, 2.0 * second, origin - first))
        return _CurveChart(
            "tan_half_angle",
            numerator,
            Polynomial((1.0, 0.0, 1.0)),
            curve.domain,
            None,
        )
    if branch.kind is ConicKind.HYPERBOLA:
        first = branch.branch_sign * first
        try:
            lower, upper = exp(curve.domain.start), exp(curve.domain.end)
        except OverflowError as exc:
            raise CriticalEventError("hyperbola exp chart overflowed") from exc
        if not isfinite(lower) or not isfinite(upper) or lower <= 0.0:
            raise CriticalEventError("hyperbola exp chart must remain finite and positive")
        numerator = _poly_vector(
            (0.5 * (first - second), origin, 0.5 * (first + second))
        )
        return _CurveChart(
            "exp",
            numerator,
            Polynomial((0.0, 1.0)),
            curve.domain,
            ParameterInterval(lower, upper),
        )
    if branch.kind is ConicKind.PARABOLA:
        return _CurveChart(
            "parameter",
            _poly_vector((origin, first, second)),
            Polynomial((1.0,)),
            curve.domain,
            curve.domain,
        )
    if branch.kind in {
        ConicKind.INTERSECTING_LINES,
        ConicKind.PARALLEL_LINES,
        ConicKind.COINCIDENT_LINE,
    }:
        return _CurveChart(
            "parameter",
            _poly_vector((origin, first)),
            Polynomial((1.0,)),
            curve.domain,
            curve.domain,
        )
    raise CriticalEventError(f"unsupported conic branch kind {branch.kind.value!r}")


def _poly_dot(left: Sequence[Polynomial], right: Sequence[float]) -> Polynomial:
    result = Polynomial((0.0,))
    for polynomial, scalar in zip(left, right):
        result = result + float(scalar) * polynomial
    return result


def _quadric_form(
    numerator: Sequence[Polynomial],
    denominator: Polynomial,
    matrix: np.ndarray,
) -> Polynomial:
    homogeneous = (*numerator, denominator)
    result = Polynomial((0.0,))
    for row in range(4):
        for column in range(4):
            coefficient = float(matrix[row, column])
            if coefficient != 0.0:
                result = result + coefficient * homogeneous[row] * homogeneous[column]
    return result


def _ray_equations(
    chart: _CurveChart,
    matrix: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, Polynomial, Polynomial, Polynomial, float, float, float]:
    a_matrix = matrix[:3, :3]
    affine = matrix[:3, 3]
    a = float(direction @ a_matrix @ direction)
    transformed = []
    for row in range(3):
        value = Polynomial((0.0,))
        for column in range(3):
            value = value + float(a_matrix[row, column]) * chart.numerator[column]
        value = value + float(affine[row]) * chart.denominator
        transformed.append(value)
    b = 2.0 * _poly_dot(transformed, direction)
    c = _quadric_form(chart.numerator, chart.denominator, matrix)
    discriminant = b * b - 4.0 * a * c

    q_scale = float(np.sum(np.abs(matrix)))
    h_scale = max(chart.homogeneous_scale, np.finfo(float).tiny)
    b_scale = max(2.0 * q_scale * h_scale, np.finfo(float).tiny)
    c_scale = max(q_scale * h_scale * h_scale, np.finfo(float).tiny)
    discriminant_scale = max(
        b_scale * b_scale + 4.0 * abs(a) * c_scale,
        np.finfo(float).tiny,
    )
    return a, b, c, discriminant, b_scale, c_scale, discriminant_scale


def _expected_degree(chart: _CurveChart, equation: CriticalEventKind) -> int:
    if chart.name == "tan_half_angle":
        return 2 if equation in {
            CriticalEventKind.SELF_OCCLUSION_SWITCH,
            CriticalEventKind.AXIAL_BOUNDARY,
        } else 4
    if chart.name == "exp":
        return 2 if equation in {
            CriticalEventKind.SELF_OCCLUSION_SWITCH,
            CriticalEventKind.AXIAL_BOUNDARY,
        } else 4
    return 8


def _normalized_coefficients(polynomial: Polynomial) -> tuple[float, ...]:
    return tuple(float(item) for item in polynomial.coef)


def _equation_identity(coefficients: Sequence[float], scale: float) -> bool:
    maximum = max((abs(float(item)) for item in coefficients), default=0.0)
    return maximum <= _IDENTITY_FACTOR * _FLOAT_EPSILON * max(
        abs(float(scale)), np.finfo(float).tiny
    )


def _root_tolerance(chart: _CurveChart, parameter_epsilon: float) -> float:
    if chart.name == "parameter":
        return parameter_epsilon
    if chart.name == "exp" and chart.root_domain is not None:
        return max(
            _FLOAT_EPSILON * 4096.0,
            parameter_epsilon * max(1.0, chart.root_domain.end),
        )
    return _FLOAT_EPSILON * 4096.0


def _tan_half_angle_root_domains(
    chart: _CurveChart,
    coefficients: Sequence[float],
    parameter_epsilon: float,
    scale: float,
) -> tuple[ParameterInterval, ...]:
    """Map one finite angle domain to the real tan-half-angle chart.

    A clipped ellipse arc often occupies only the two tails on either side of
    a ``tan(t / 2)`` pole.  Solving the event polynomial over its entire
    Cauchy interval also inspects the excluded middle of the chart; for a
    nearly parabolic ellipse that unnecessary interval can manufacture an
    unvalidated stationary candidate.  Split at every chart pole and solve
    only the root intervals that map back to the authored curve domain.
    """

    canonical = [float(value) for value in coefficients]
    chart_pole_epsilon = _IDENTITY_FACTOR * _FLOAT_EPSILON * max(
        abs(float(scale)),
        np.finfo(float).tiny,
    )
    while len(canonical) > 1 and abs(canonical[-1]) <= chart_pole_epsilon:
        canonical.pop()
    if len(canonical) <= 1:
        return ()
    degree = len(canonical) - 1
    leading = abs(canonical[-1])
    # A Fujiwara-style bound is substantially tighter than the elementary
    # Cauchy ``1 + max(abs(a_i / a_n))`` bound for the quartics produced by a
    # nearly parabolic ellipse.  The factor-two form below is conservative for
    # every coefficient and keeps tail-domain normalization well conditioned.
    root_bound = 2.0 * max(
        (abs(value) / leading) ** (1.0 / (degree - index))
        for index, value in enumerate(canonical[:-1])
    )
    if not isfinite(root_bound):
        raise CriticalEventError(
            "tan-half-angle root bound overflowed for a finite curve domain"
        )

    poles = tuple(
        value
        for value in chart.chart_poles
        if chart.curve_domain.start < value < chart.curve_domain.end
    )
    boundaries = (chart.curve_domain.start, *poles, chart.curve_domain.end)
    pole_values = chart.chart_poles

    def is_pole(value: float) -> bool:
        return any(abs(value - pole) <= parameter_epsilon for pole in pole_values)

    result: list[ParameterInterval] = []
    for start, end in zip(boundaries, boundaries[1:]):
        raw_lower = -root_bound if is_pole(start) else tan(0.5 * start)
        raw_upper = root_bound if is_pole(end) else tan(0.5 * end)
        if raw_lower > raw_upper + parameter_epsilon:
            raise CriticalEventError(
                "tan-half-angle chart interval is not monotone between poles"
            )
        lower = max(-root_bound, float(raw_lower))
        upper = min(root_bound, float(raw_upper))
        if lower > upper + parameter_epsilon:
            continue
        if lower > upper:
            lower = upper
        interval = ParameterInterval(lower, upper)
        if interval not in result:
            result.append(interval)
    return tuple(result)


def _equation_events(
    chart: _CurveChart,
    *,
    kind: CriticalEventKind,
    equation: str,
    surface_id: str,
    polynomial: Polynomial,
    scale: float,
    context: ResolvedGeometryContext,
) -> tuple[tuple[float, CriticalEvidence], ...]:
    coefficients = _normalized_coefficients(polynomial)
    parameter_epsilon = context.epsilon(GeometryQuantity.PARAMETER)
    if _equation_identity(coefficients, scale):
        return (
            (
                chart.curve_domain.start,
                CriticalEvidence(
                    kind,
                    equation,
                    surface_id,
                    chart.name,
                    coefficients,
                    None,
                    1,
                    0.0,
                    True,
                ),
            ),
        )
    try:
        if chart.name == "exp":
            parameter_roots = tuple(
                (item.parameter, item.chart_root)
                for item in solve_real_polynomial_exp_chart(
                    coefficients,
                    parameter_domain=chart.curve_domain,
                    context=context,
                    parameter_tolerance=parameter_epsilon,
                )
            )
        else:
            root_domains = (
                _tan_half_angle_root_domains(
                    chart,
                    coefficients,
                    parameter_epsilon,
                    scale,
                )
                if chart.name == "tan_half_angle"
                else (chart.root_domain,)
            )
            roots = tuple(
                root
                for domain in root_domains
                for root in solve_real_polynomial(
                    coefficients,
                    domain=domain,
                    context=context,
                    parameter_tolerance=_root_tolerance(
                        chart,
                        parameter_epsilon,
                    ),
                )
            )
            parameter_roots = tuple(
                (parameter, root)
                for root in roots
                for parameter in chart.parameters(root.value, parameter_epsilon)
            )
    except PolynomialRootError as exc:
        raise CriticalEventError(
            f"critical equation {equation!r} is ambiguous: {exc}"
        ) from exc

    result: list[tuple[float, CriticalEvidence]] = []
    for parameter, root in parameter_roots:
        result.append(
            (
                parameter,
                CriticalEvidence(
                    kind,
                    equation,
                    surface_id,
                    chart.name,
                    coefficients,
                    root.value,
                    root.multiplicity,
                    root.residual,
                    False,
                ),
            )
        )

    # ``tan(t/2)`` has one point at infinity per revolution.  A missing
    # highest coefficient is an analytic root at that chart pole, not a
    # sampling artefact.  Preserve the original equation as evidence there.
    if chart.name == "tan_half_angle":
        expected = _expected_degree(chart, kind)
        padded = (*coefficients, *((0.0,) * max(0, expected + 1 - len(coefficients))))
        multiplicity = 0
        for index in range(expected, -1, -1):
            if abs(padded[index]) <= _IDENTITY_FACTOR * _FLOAT_EPSILON * max(
                abs(float(scale)), np.finfo(float).tiny
            ):
                multiplicity += 1
            else:
                break
        if multiplicity:
            for seam in chart.chart_poles:
                result.append(
                    (
                        seam,
                        CriticalEvidence(
                            kind,
                            equation,
                            surface_id,
                            chart.name,
                            coefficients,
                            None,
                            multiplicity,
                            0.0,
                            False,
                        ),
                    )
                )
    return tuple(result)


def _axial_events(
    chart: _CurveChart,
    surface: CylinderSpec | ConeSpec,
    direction: np.ndarray,
    matrix: np.ndarray,
    *,
    context: ResolvedGeometryContext,
) -> tuple[tuple[float, CriticalEvidence], ...]:
    axis = np.asarray(surface.axis, dtype=float)
    reference = np.asarray(
        surface.origin if isinstance(surface, CylinderSpec) else surface.apex,
        dtype=float,
    )
    denominator = chart.denominator
    axial_numerator = _poly_dot(chart.numerator, axis) - float(
        np.dot(axis, reference)
    ) * denominator
    direction_axial = float(np.dot(axis, direction))
    result: list[tuple[float, CriticalEvidence]] = []
    h_scale = chart.homogeneous_scale
    matrix_scale = float(np.sum(np.abs(matrix)))
    for index, boundary in enumerate(surface.axial_range):
        label = "min" if index == 0 else "max"
        source_boundary = axial_numerator - float(boundary) * denominator
        result.extend(
            _equation_events(
                chart,
                kind=CriticalEventKind.AXIAL_BOUNDARY,
                equation=f"axial_{label}",
                surface_id=surface.surface_id,
                polynomial=source_boundary,
                scale=max(h_scale * (1.0 + abs(boundary)), np.finfo(float).tiny),
                context=context,
            )
        )
        # A cone endpoint at its apex has zero radius and therefore is not a
        # circular cap/trim rim.  Its genuine event is the axial boundary
        # above; support tangency already owns a projected generator through
        # the apex.  Expanding a fictitious zero-radius rim into a quartic is
        # both redundant and ill-conditioned near the parabolic section angle.
        if isinstance(surface, ConeSpec) and boundary == 0.0:
            continue
        if abs(direction_axial) <= context.epsilon(GeometryQuantity.ANGULAR):
            continue
        # At the axial boundary, lambda=(boundary-z(point))/(axis.direction).
        # Multiplying the hit by direction_axial*W clears that denominator;
        # substituting it into Q yields the exact projected cap-rim equation.
        lambda_numerator = float(boundary) * denominator - axial_numerator
        hit_numerator = tuple(
            direction_axial * item + float(component) * lambda_numerator
            for item, component in zip(chart.numerator, direction)
        )
        hit_denominator = direction_axial * denominator
        cap_rim = _quadric_form(hit_numerator, hit_denominator, matrix)
        result.extend(
            _equation_events(
                chart,
                kind=CriticalEventKind.CAP_RIM,
                equation=f"cap_rim_{label}",
                surface_id=surface.surface_id,
                polynomial=cap_rim,
                scale=max(
                    matrix_scale
                    * h_scale
                    * h_scale
                    * (1.0 + abs(boundary)) ** 2,
                    np.finfo(float).tiny,
                ),
                context=context,
            )
        )
    return tuple(result)


def _curve_characteristic_points(curve: AnalyticCurve3D) -> tuple[tuple[float, float, float], ...]:
    if isinstance(curve, SegmentCurve):
        return (curve.start, curve.end)
    if isinstance(curve, EllipseArcCurve):
        center = np.asarray(curve.center, dtype=float)
        first = np.asarray(curve.first_axis, dtype=float)
        second = np.asarray(curve.second_axis, dtype=float)
        return tuple(
            tuple(float(item) for item in point)
            for point in (
                center + first,
                center - first,
                center + second,
                center - second,
            )
        )
    return tuple(
        curve.point(value)
        for value in (curve.domain.start, curve.domain.midpoint, curve.domain.end)
    )


def _resolved_context(
    curve: AnalyticCurve3D,
    surfaces: Sequence[QuadricSurfaceSpec],
    context: ContextInput,
) -> ResolvedGeometryContext:
    if isinstance(context, ResolvedGeometryContext):
        return resolve_geometry_context(context)
    points = list(_curve_characteristic_points(curve))
    for surface in surfaces:
        points.extend(surface.characteristic_points)
    return resolve_geometry_context(context, positions=points)


def _merge_events(
    entries: Sequence[tuple[float, CriticalEvidence]],
    *,
    curve: AnalyticCurve3D,
    domain: ParameterInterval,
    parameter_epsilon: float,
    point_epsilon: float,
) -> tuple[CriticalEvent, ...]:
    ordered = sorted(entries, key=lambda item: (item[0], _evidence_key(item[1])))
    groups: list[list[tuple[float, CriticalEvidence]]] = []
    for entry in ordered:
        parameter = entry[0]
        if parameter < domain.start - parameter_epsilon or parameter > domain.end + parameter_epsilon:
            continue
        if parameter < domain.start:
            entry = (domain.start, entry[1])
        elif parameter > domain.end:
            entry = (domain.end, entry[1])
        parameter_close = bool(
            groups
            and entry[0] - groups[-1][-1][0] <= parameter_epsilon
        )
        endpoint_evidence = (
            entry[1].kind is CriticalEventKind.DOMAIN_ENDPOINT
            or (
                groups
                and any(
                    evidence.kind is CriticalEventKind.DOMAIN_ENDPOINT
                    for _value, evidence in groups[-1]
                )
            )
        )
        point_close = False
        if groups and not endpoint_evidence and not parameter_close:
            # Different analytic equations may describe the same geometric
            # visibility switch.  Their independently solved roots can be
            # farther apart than the parameter tolerance after a rigid
            # translation, even though the evaluated curve points still
            # coincide.  Keep real domain seams distinct, then cluster the
            # remaining evidence in world space as well as parameter space.
            previous = groups[-1][-1][0]
            point_close = (
                float(
                    np.linalg.norm(
                        np.asarray(curve.point(entry[0]), dtype=float)
                        - np.asarray(curve.point(previous), dtype=float)
                    )
                )
                <= point_epsilon
            )
        if groups and (parameter_close or point_close):
            groups[-1].append(entry)
        else:
            groups.append([entry])
    result: list[CriticalEvent] = []
    for group in groups:
        parameter = group[0][0]
        evidence = tuple(dict.fromkeys(item[1] for item in group))
        result.append(CriticalEvent(parameter, evidence))
    return tuple(result)


def compute_curve_critical_events(
    curve: AnalyticCurve3D,
    surfaces: Sequence[QuadricSurfaceSpec],
    view: ParallelView,
    *,
    context: ContextInput = None,
) -> tuple[CriticalEvent, ...]:
    """Return every analytic parameter where visibility can change."""

    if not isinstance(curve, (SegmentCurve, EllipseArcCurve, ParametricConicBranch)):
        raise TypeError("curve must be a supported analytic 3D curve")
    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    surface_items = tuple(surfaces)
    if not all(isinstance(item, (SphereSpec, CylinderSpec, ConeSpec)) for item in surface_items):
        raise TypeError("surfaces must contain sphere, cylinder, or cone specs")
    surface_ids = tuple(item.surface_id for item in surface_items)
    if len(set(surface_ids)) != len(surface_ids):
        raise CriticalEventError("surface identities must be unique")
    surface_items = tuple(sorted(surface_items, key=lambda item: item.surface_id))
    resolved = _resolved_context(curve, surface_items, context)
    chart = _curve_chart(curve)
    direction = np.asarray(view.view_direction, dtype=float)
    entries: list[tuple[float, CriticalEvidence]] = [
        (
            curve.domain.start,
            CriticalEvidence(
                CriticalEventKind.DOMAIN_ENDPOINT,
                "domain_start",
            ),
        ),
        (
            curve.domain.end,
            CriticalEvidence(
                CriticalEventKind.DOMAIN_ENDPOINT,
                "domain_end",
            ),
        ),
    ]
    entries.extend(
        (
            seam,
            CriticalEvidence(
                CriticalEventKind.CHART_SEAM,
                "tan_half_angle_chart_seam",
                chart="tan_half_angle",
            ),
        )
        for seam in chart.chart_seams
    )
    for surface in surface_items:
        matrix = np.asarray(surface.support_quadric.matrix, dtype=float)
        _a, b, c, discriminant, b_scale, c_scale, discriminant_scale = _ray_equations(
            chart,
            matrix,
            direction,
        )
        curve_on_support = _equation_identity(
            _normalized_coefficients(c),
            c_scale,
        )
        if curve_on_support:
            # Along a curve already contained in the quadric support, c=0 and
            # the ray discriminant is exactly b**2.  Solving that expanded
            # repeated-root polynomial is needlessly ill-conditioned when a
            # projected circle approaches rank one.  Solve its analytic
            # factor instead, while retaining support-tangency evidence and
            # the doubled multiplicity of the original discriminant root.
            factored_tangencies = _equation_events(
                chart,
                # The factor has the same chart degree as the linear ray
                # coefficient.  Rewrite its evidence kind below after the
                # chart-pole accounting has used that correct degree.
                kind=CriticalEventKind.SELF_OCCLUSION_SWITCH,
                equation="ray_discriminant_on_surface_factor",
                surface_id=surface.surface_id,
                polynomial=b,
                scale=b_scale,
                context=resolved,
            )
            entries.extend(
                (
                    parameter,
                    CriticalEvidence(
                        CriticalEventKind.SUPPORT_TANGENCY,
                        evidence.equation,
                        evidence.surface_id,
                        evidence.chart,
                        evidence.coefficients,
                        evidence.root_value,
                        2 * evidence.multiplicity,
                        evidence.residual,
                        evidence.identically_zero,
                    ),
                )
                for parameter, evidence in factored_tangencies
            )
        else:
            entries.extend(
                _equation_events(
                    chart,
                    kind=CriticalEventKind.SUPPORT_TANGENCY,
                    equation="ray_discriminant",
                    surface_id=surface.surface_id,
                    polynomial=discriminant,
                    scale=discriminant_scale,
                    context=resolved,
                )
            )
        entries.extend(
            _equation_events(
                chart,
                kind=CriticalEventKind.CURVE_SURFACE_INTERSECTION,
                equation="curve_on_surface",
                surface_id=surface.surface_id,
                polynomial=c,
                scale=c_scale,
                context=resolved,
            )
        )
        entries.extend(
            _equation_events(
                chart,
                kind=CriticalEventKind.SELF_OCCLUSION_SWITCH,
                equation="ray_linear_coefficient",
                surface_id=surface.surface_id,
                polynomial=b,
                scale=b_scale,
                context=resolved,
            )
        )
        if isinstance(surface, (CylinderSpec, ConeSpec)):
            entries.extend(
                _axial_events(
                    chart,
                    surface,
                    direction,
                    matrix,
                    context=resolved,
                )
            )
    return _merge_events(
        entries,
        curve=curve,
        domain=curve.domain,
        parameter_epsilon=resolved.epsilon(GeometryQuantity.PARAMETER),
        point_epsilon=resolved.epsilon(GeometryQuantity.BOUNDARY),
    )


__all__ = [
    "CRITICAL_EVENT_SCHEMA",
    "AnalyticCurve3D",
    "CriticalEvidence",
    "CriticalEvent",
    "CriticalEventError",
    "CriticalEventKind",
    "QuadricSurfaceSpec",
    "compute_curve_critical_events",
]
