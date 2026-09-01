"""Deterministic real roots for the low-degree quadric kernel.

The coefficient convention is the NumPy ``Polynomial`` convention:
``coefficients[index]`` multiplies ``x ** index``.  The implementation is
renderer-neutral and intentionally limited to the degree-eight equations
produced by conic/quadric critical-event resultants.

Roots are isolated on finite monotonic intervals obtained recursively from
the derivative.  Consequently an even-multiplicity root is inspected at the
corresponding derivative root instead of being lost because its polynomial
does not change sign.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np
from numpy.polynomial import Polynomial

from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from ..topology import ParameterInterval


MAX_POLYNOMIAL_DEGREE = 8
_FLOAT_EPSILON = float(np.finfo(float).eps)
_DEFAULT_RESIDUAL_FACTOR = 256.0
_INTERNAL_CLUSTER_FACTOR = 4096.0


class PolynomialRootError(ValueError):
    """Raised when a polynomial cannot be solved without guessing."""


@dataclass(frozen=True, slots=True)
class RealRoot:
    """One validated real root.

    ``residual`` is the scale-independent backward residual

    ``abs(p(value)) / sum(abs(c_i) * max(1, abs(value)) ** i)``.

    A clustered root keeps the lowest parameter value as its deterministic
    representative and sums the multiplicities of the clustered roots.
    """

    value: float
    multiplicity: int
    residual: float

    def __post_init__(self) -> None:
        value = float(self.value)
        residual = float(self.residual)
        if not isfinite(value):
            raise ValueError("root value must be finite")
        if isinstance(self.multiplicity, bool) or not isinstance(
            self.multiplicity, int
        ):
            raise TypeError("root multiplicity must be an integer")
        if self.multiplicity <= 0:
            raise ValueError("root multiplicity must be positive")
        if not isfinite(residual) or residual < 0.0:
            raise ValueError("root residual must be finite and non-negative")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "residual", residual)


@dataclass(frozen=True, slots=True)
class ExpChartRoot:
    """One root of ``p(exp(t))`` with both parameter and chart evidence."""

    parameter: float
    chart_root: RealRoot

    def __post_init__(self) -> None:
        parameter = float(self.parameter)
        if not isfinite(parameter):
            raise ValueError("exp-chart parameter must be finite")
        if not isinstance(self.chart_root, RealRoot):
            raise TypeError("chart_root must be a RealRoot")
        if self.chart_root.value <= 0.0:
            raise ValueError("an exp-chart root must be positive")
        object.__setattr__(self, "parameter", parameter)


def _parameter_epsilon(
    context: GeometryContext | ResolvedGeometryContext | None,
    parameter_tolerance: float | None,
) -> float:
    if parameter_tolerance is None:
        return resolve_geometry_context(context).epsilon(GeometryQuantity.PARAMETER)
    value = float(parameter_tolerance)
    if not isfinite(value) or value < 0.0:
        raise ValueError("parameter_tolerance must be finite and non-negative")
    return value


def _residual_epsilon(degree: int, residual_tolerance: float | None) -> float:
    if residual_tolerance is None:
        return _DEFAULT_RESIDUAL_FACTOR * _FLOAT_EPSILON * max(1, degree + 1)
    value = float(residual_tolerance)
    if not isfinite(value) or value < 0.0:
        raise ValueError("residual_tolerance must be finite and non-negative")
    return value


def _canonical_coefficients(coefficients: Sequence[float]) -> tuple[float, ...]:
    if isinstance(coefficients, (str, bytes)):
        raise TypeError("coefficients must be a numeric sequence")
    try:
        values = [float(value) for value in coefficients]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("coefficients must be a numeric sequence") from exc
    if not values:
        raise PolynomialRootError("a polynomial must contain at least one coefficient")
    if not all(isfinite(value) for value in values):
        raise PolynomialRootError("polynomial coefficients must be finite")
    while len(values) > 1 and values[-1] == 0.0:
        values.pop()
    if all(value == 0.0 for value in values):
        raise PolynomialRootError("the zero polynomial has infinitely many roots")
    degree = len(values) - 1
    if degree > MAX_POLYNOMIAL_DEGREE:
        raise PolynomialRootError(
            f"polynomial degree {degree} exceeds supported degree "
            f"{MAX_POLYNOMIAL_DEGREE}"
        )
    return tuple(values)


def _coerce_domain(
    domain: ParameterInterval | Sequence[float] | None,
    coefficients: Sequence[float],
) -> ParameterInterval:
    if domain is not None:
        if isinstance(domain, ParameterInterval):
            return domain
        if isinstance(domain, (str, bytes)) or len(domain) != 2:
            raise TypeError("domain must be a ParameterInterval or two finite values")
        return ParameterInterval(float(domain[0]), float(domain[1]))

    # Every root of a_n*x^n + ... + a_0 lies inside the Cauchy bound below.
    leading = abs(float(coefficients[-1]))
    ratio = max((abs(float(value)) for value in coefficients[:-1]), default=0.0)
    bound = 1.0 + ratio / leading
    if not isfinite(bound):
        raise PolynomialRootError(
            "automatic root bound overflowed; provide an explicit finite domain"
        )
    return ParameterInterval(-bound, bound)


def _sign_variations(coefficients: Sequence[float]) -> int:
    """Return the exact sign changes after zero coefficients are removed."""

    signs = tuple(1 if value > 0.0 else -1 for value in coefficients if value != 0.0)
    return sum(left != right for left, right in zip(signs, signs[1:]))


def _descartes_excludes_domain_roots(
    coefficients: Sequence[float],
    domain: ParameterInterval,
) -> bool:
    """Certify that one authored half-axis contains no real polynomial root.

    Descartes' rule of signs says a polynomial with no coefficient sign
    changes has no strictly positive roots.  Applying the same test to
    ``p(-x)`` covers the negative half-axis.  This exact pre-check avoids
    mapping an obviously root-free tan-half-angle tail to an enormous
    normalized interval, where floating-point cancellation can otherwise
    manufacture a false endpoint candidate.
    """

    # Keep a possible root at x=0 in the ordinary validated path.
    if coefficients[0] == 0.0:
        return False
    if domain.start >= 0.0:
        return _sign_variations(coefficients) == 0
    if domain.end <= 0.0:
        reflected = tuple(
            value if index % 2 == 0 else -value
            for index, value in enumerate(coefficients)
        )
        return _sign_variations(reflected) == 0
    return False


def _polyval(coefficients: Sequence[float], value: float) -> np.longdouble:
    argument = np.longdouble(value)
    result = np.longdouble(0.0)
    for coefficient in reversed(coefficients):
        result = result * argument + np.longdouble(coefficient)
    return result


def _normalized_residual(coefficients: Sequence[float], value: float) -> float:
    argument = np.longdouble(value)
    absolute_argument = max(np.longdouble(1.0), abs(argument))
    scale = np.longdouble(0.0)
    power = np.longdouble(1.0)
    for coefficient in coefficients:
        scale += abs(np.longdouble(coefficient)) * power
        power *= absolute_argument
    if scale == 0.0:
        return 0.0 if _polyval(coefficients, value) == 0.0 else float("inf")
    return float(abs(_polyval(coefficients, value)) / scale)


def _derivative(coefficients: Sequence[float]) -> tuple[float, ...]:
    return tuple(index * float(coefficients[index]) for index in range(1, len(coefficients)))


def _cluster_values(
    values: Iterable[float],
    *,
    tolerance: float,
) -> tuple[tuple[float, ...], ...]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return ()
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return tuple(tuple(group) for group in groups)


def _bisect_sign_change(
    coefficients: Sequence[float],
    lower: float,
    upper: float,
) -> float:
    left = float(lower)
    right = float(upper)
    left_value = _polyval(coefficients, left)
    right_value = _polyval(coefficients, right)
    if left_value == 0.0:
        return left
    if right_value == 0.0:
        return right
    if (left_value < 0.0) == (right_value < 0.0):
        raise PolynomialRootError("root interval does not contain a sign change")

    # Bisection supplies a deterministic bracket.  Continuing until the
    # midpoint can no longer move also gives a substantially stronger
    # residual than stopping at the geometry parameter tolerance.
    for _ in range(256):
        midpoint = left + 0.5 * (right - left)
        if midpoint == left or midpoint == right:
            break
        middle_value = _polyval(coefficients, midpoint)
        if middle_value == 0.0:
            return midpoint
        if (middle_value < 0.0) == (left_value < 0.0):
            left = midpoint
            left_value = middle_value
        else:
            right = midpoint
            right_value = middle_value

    candidates = (left, left + 0.5 * (right - left), right)
    return min(
        candidates,
        key=lambda value: (_normalized_residual(coefficients, value), value),
    )


def _isolate_candidates(
    coefficients: Sequence[float],
    lower: float,
    upper: float,
    *,
    residual_epsilon: float,
    numerical_cluster_epsilon: float,
) -> tuple[float, ...]:
    degree = len(coefficients) - 1
    if degree <= 0:
        return ()
    if degree == 1:
        root = -float(coefficients[0]) / float(coefficients[1])
        if lower <= root <= upper:
            return (root,)
        return ()

    derivative_roots = _isolate_candidates(
        _derivative(coefficients),
        lower,
        upper,
        residual_epsilon=residual_epsilon,
        numerical_cluster_epsilon=numerical_cluster_epsilon,
    )
    derivative_roots = tuple(
        group[0]
        for group in _cluster_values(
            derivative_roots,
            tolerance=numerical_cluster_epsilon,
        )
    )
    breakpoints = (lower, *derivative_roots, upper)
    candidates: list[float] = []

    # Endpoints cover authored-domain roots.  At an interior stationary point
    # a small residual alone is not sufficient: two very close simple roots
    # have a tiny non-zero extremum between them.  If that extremum has the
    # opposite sign from an adjacent monotonic endpoint, the two surrounding
    # sign changes own the roots and the extremum itself must not be emitted as
    # a spurious even root.
    for value in (lower, upper):
        if _normalized_residual(coefficients, value) <= residual_epsilon:
            candidates.append(float(value))
    for index, value in enumerate(derivative_roots, start=1):
        if _normalized_residual(coefficients, value) > residual_epsilon:
            continue
        center_value = _polyval(coefficients, value)
        if center_value == 0.0:
            candidates.append(float(value))
            continue
        center_negative = center_value < 0.0
        neighbor_values = (
            _polyval(coefficients, breakpoints[index - 1]),
            _polyval(coefficients, breakpoints[index + 1]),
        )
        has_opposite_neighbor = any(
            neighbor != 0.0 and (neighbor < 0.0) != center_negative
            for neighbor in neighbor_values
        )
        if not has_opposite_neighbor:
            candidates.append(float(value))

    # Between consecutive derivative roots the polynomial is monotonic, so a
    # strict sign change contains exactly one distinct real root.
    for left, right in zip(breakpoints, breakpoints[1:]):
        if right <= left:
            continue
        left_value = _polyval(coefficients, left)
        right_value = _polyval(coefficients, right)
        if left_value == 0.0 or right_value == 0.0:
            continue
        if (left_value < 0.0) != (right_value < 0.0):
            candidates.append(_bisect_sign_change(coefficients, left, right))

    groups = _cluster_values(candidates, tolerance=numerical_cluster_epsilon)
    return tuple(
        min(
            group,
            key=lambda value: (_normalized_residual(coefficients, value), value),
        )
        for group in groups
    )


def _multiplicity(
    coefficients: Sequence[float],
    value: float,
    *,
    residual_epsilon: float,
) -> int:
    current = tuple(float(item) for item in coefficients)
    degree = len(current) - 1
    for order in range(1, degree + 1):
        current = _derivative(current)
        if _normalized_residual(current, value) > residual_epsilon:
            return order
    return degree


def _polish_mapped_candidate(
    coefficients: Sequence[float],
    value: float,
    domain: ParameterInterval,
) -> float:
    """Refine a normalized-domain candidate in the original polynomial.

    Mapping a well-isolated root from ``[-1, 1]`` back to a one-sided chart
    interval can lose several bits through ``center + half_width * y``
    cancellation.  A deterministic long-double Newton step restores the
    best representable original-domain value before residual validation.  The
    step is strictly bounded by the authored root interval and is never used
    to accept a candidate whose original polynomial still fails validation.
    """

    current = np.longdouble(value)
    lower = np.longdouble(domain.start)
    upper = np.longdouble(domain.end)
    best = float(value)
    best_residual = _normalized_residual(coefficients, best)
    for _ in range(16):
        polynomial = np.longdouble(0.0)
        derivative = np.longdouble(0.0)
        for coefficient in reversed(coefficients):
            derivative = derivative * current + polynomial
            polynomial = polynomial * current + np.longdouble(coefficient)
        if derivative == 0.0:
            break
        candidate = current - polynomial / derivative
        if not np.isfinite(candidate) or candidate < lower or candidate > upper:
            break
        candidate_float = float(candidate)
        residual = _normalized_residual(coefficients, candidate_float)
        if residual < best_residual:
            best = candidate_float
            best_residual = residual
        if candidate_float == float(current):
            break
        current = candidate

    neighborhood = (
        float(np.nextafter(best, float("-inf"))),
        best,
        float(np.nextafter(best, float("inf"))),
    )
    return min(
        (item for item in neighborhood if domain.start <= item <= domain.end),
        key=lambda item: (_normalized_residual(coefficients, item), item),
    )


def cluster_real_roots(
    roots: Iterable[RealRoot],
    *,
    context: GeometryContext | ResolvedGeometryContext | None = None,
    parameter_tolerance: float | None = None,
) -> tuple[RealRoot, ...]:
    """Cluster nearby validated roots in deterministic parameter order.

    The lowest value in each tolerance-connected cluster is retained, matching
    the exact-boundary convention used by ``partition_parameter_domain``.
    Multiplicities are summed because the inputs represent algebraic roots,
    not duplicate numerical candidates.
    """

    epsilon = _parameter_epsilon(context, parameter_tolerance)
    indexed: list[tuple[RealRoot, int]] = []
    for index, root in enumerate(roots):
        if not isinstance(root, RealRoot):
            raise TypeError("roots must contain RealRoot objects")
        indexed.append((root, index))
    indexed.sort(key=lambda item: (item[0].value, item[1]))
    if not indexed:
        return ()

    groups: list[list[RealRoot]] = [[indexed[0][0]]]
    for root, _index in indexed[1:]:
        if root.value - groups[-1][-1].value <= epsilon:
            groups[-1].append(root)
        else:
            groups.append([root])
    return tuple(
        RealRoot(
            value=group[0].value,
            multiplicity=sum(root.multiplicity for root in group),
            residual=group[0].residual,
        )
        for group in groups
    )


def solve_real_polynomial(
    coefficients: Sequence[float],
    *,
    domain: ParameterInterval | Sequence[float] | None = None,
    context: GeometryContext | ResolvedGeometryContext | None = None,
    parameter_tolerance: float | None = None,
    residual_tolerance: float | None = None,
) -> tuple[RealRoot, ...]:
    """Return all validated real roots in one finite closed domain.

    Coefficients use ascending power order.  When ``domain`` is omitted, a
    finite Cauchy bound containing every root is derived.  Geometry parameter
    tolerance controls near-root clustering and boundary snapping; it does not
    turn a merely small polynomial value into a root.
    """

    canonical = _canonical_coefficients(coefficients)
    degree = len(canonical) - 1
    if degree == 0:
        return ()
    interval = _coerce_domain(domain, canonical)
    parameter_epsilon = _parameter_epsilon(context, parameter_tolerance)
    residual_epsilon = _residual_epsilon(degree, residual_tolerance)

    if interval.length == 0.0:
        residual = _normalized_residual(canonical, interval.start)
        if residual > residual_epsilon:
            return ()
        return (
            RealRoot(
                interval.start,
                _multiplicity(
                    canonical,
                    interval.start,
                    residual_epsilon=residual_epsilon,
                ),
                residual,
            ),
        )

    if _descartes_excludes_domain_roots(canonical, interval):
        return ()

    center = interval.midpoint
    half_width = 0.5 * interval.length
    try:
        transformed = Polynomial(canonical)(Polynomial((center, half_width))).coef
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise PolynomialRootError("polynomial domain normalization failed") from exc
    if not np.all(np.isfinite(transformed)):
        raise PolynomialRootError("polynomial domain normalization overflowed")
    transformed_scale = float(np.max(np.abs(transformed)))
    if transformed_scale == 0.0 or not isfinite(transformed_scale):
        raise PolynomialRootError("polynomial domain normalization is singular")
    normalized = tuple(float(value / transformed_scale) for value in transformed)
    while len(normalized) > 1 and normalized[-1] == 0.0:
        normalized = normalized[:-1]

    parameter_epsilon_y = parameter_epsilon / half_width
    numerical_cluster_epsilon = max(
        _INTERNAL_CLUSTER_FACTOR * _FLOAT_EPSILON,
        min(parameter_epsilon_y, 1.0) * _FLOAT_EPSILON,
    )
    candidates_y = _isolate_candidates(
        normalized,
        -1.0,
        1.0,
        residual_epsilon=residual_epsilon,
        numerical_cluster_epsilon=numerical_cluster_epsilon,
    )

    raw_roots: list[RealRoot] = []
    for candidate_y in candidates_y:
        value = center + half_width * candidate_y
        if value < interval.start - parameter_epsilon:
            continue
        if value > interval.end + parameter_epsilon:
            continue
        if value < interval.start:
            boundary_residual = _normalized_residual(canonical, interval.start)
            if boundary_residual > residual_epsilon:
                continue
            value = interval.start
        elif value > interval.end:
            boundary_residual = _normalized_residual(canonical, interval.end)
            if boundary_residual > residual_epsilon:
                continue
            value = interval.end

        residual = _normalized_residual(canonical, value)
        if residual > residual_epsilon:
            value = _polish_mapped_candidate(canonical, value, interval)
            residual = _normalized_residual(canonical, value)
        if residual > residual_epsilon:
            raise PolynomialRootError(
                "isolated root failed residual validation; coefficients are "
                "numerically ambiguous"
            )
        raw_roots.append(
            RealRoot(
                value=float(value),
                multiplicity=_multiplicity(
                    normalized,
                    candidate_y,
                    residual_epsilon=residual_epsilon,
                ),
                residual=residual,
            )
        )

    return cluster_real_roots(
        raw_roots,
        context=context,
        parameter_tolerance=parameter_epsilon,
    )


def solve_real_polynomial_exp_chart(
    coefficients: Sequence[float],
    *,
    parameter_domain: ParameterInterval | Sequence[float],
    context: GeometryContext | ResolvedGeometryContext | None = None,
    parameter_tolerance: float | None = None,
    residual_tolerance: float | None = None,
) -> tuple[ExpChartRoot, ...]:
    """Solve ``p(exp(t)) = 0`` without mixing tiny and huge chart roots.

    The non-negative half uses ``y = exp(t)``.  The non-positive half uses the
    exact reciprocal polynomial in ``z = exp(-t)``.  Each half is additionally
    scaled to ``(0, 1]`` before calling :func:`solve_real_polynomial`.  This is
    essential for wide hyperbola domains such as ``t in [-30, 30]``, where the
    two valid chart roots differ by more than 24 orders of magnitude.
    """

    canonical = _canonical_coefficients(coefficients)
    domain = (
        parameter_domain
        if isinstance(parameter_domain, ParameterInterval)
        else ParameterInterval(float(parameter_domain[0]), float(parameter_domain[1]))
    )
    parameter_epsilon = _parameter_epsilon(context, parameter_tolerance)
    residual_epsilon = _residual_epsilon(len(canonical) - 1, residual_tolerance)

    def solve_half(
        half_coefficients: Sequence[float],
        lower_chart: float,
        upper_chart: float,
    ) -> tuple[RealRoot, ...]:
        scaled: list[float] = []
        power = np.longdouble(1.0)
        upper_long = np.longdouble(upper_chart)
        for coefficient in half_coefficients:
            value = np.longdouble(coefficient) * power
            if not np.isfinite(value):
                raise PolynomialRootError("exp-chart coefficient scaling overflowed")
            scaled.append(float(value))
            power *= upper_long
        scale = max((abs(item) for item in scaled), default=0.0)
        if not isfinite(scale) or scale <= 0.0:
            raise PolynomialRootError("exp-chart coefficient scaling is singular")
        normalized = tuple(item / scale for item in scaled)
        return solve_real_polynomial(
            normalized,
            domain=(lower_chart / upper_chart, 1.0),
            context=context,
            parameter_tolerance=max(4096.0 * _FLOAT_EPSILON, parameter_epsilon),
            residual_tolerance=residual_tolerance,
        )

    try:
        candidates: list[tuple[float, int]] = []
        if domain.end >= 0.0:
            lower_t = max(0.0, domain.start)
            lower_chart = exp(lower_t)
            upper_chart = exp(domain.end)
            for root in solve_half(canonical, lower_chart, upper_chart):
                chart_root = root.value * upper_chart
                candidates.append((log(chart_root), root.multiplicity))
        if domain.start <= 0.0:
            upper_t = min(0.0, domain.end)
            lower_chart = exp(-upper_t)
            upper_chart = exp(-domain.start)
            reciprocal = tuple(reversed(canonical))
            for root in solve_half(reciprocal, lower_chart, upper_chart):
                reciprocal_root = root.value * upper_chart
                candidates.append((-log(reciprocal_root), root.multiplicity))
    except OverflowError as exc:
        raise PolynomialRootError("exp-chart parameter domain overflowed") from exc

    validated: list[ExpChartRoot] = []
    for parameter, multiplicity in sorted(candidates):
        if parameter < domain.start - parameter_epsilon:
            continue
        if parameter > domain.end + parameter_epsilon:
            continue
        parameter = min(domain.end, max(domain.start, parameter))
        chart_value = exp(parameter)
        residual = _normalized_residual(canonical, chart_value)
        if residual > residual_epsilon:
            # Domain normalization can expose a non-root endpoint as a tiny
            # value in the scaled polynomial.  The authoritative original
            # equation removes it here without turning ambiguity into truth.
            continue
        candidate = ExpChartRoot(
            parameter,
            RealRoot(chart_value, multiplicity, residual),
        )
        if validated and abs(parameter - validated[-1].parameter) <= parameter_epsilon:
            if residual < validated[-1].chart_root.residual:
                validated[-1] = candidate
            continue
        validated.append(candidate)
    return tuple(validated)


__all__ = [
    "ExpChartRoot",
    "MAX_POLYNOMIAL_DEGREE",
    "PolynomialRootError",
    "RealRoot",
    "cluster_real_roots",
    "solve_real_polynomial",
    "solve_real_polynomial_exp_chart",
]
