"""Renderer-independent affine conic classification and parameterization.

The module works with one symmetric homogeneous matrix ``C`` whose affine
zero set is

``[u, v, 1] @ C @ [u, v, 1].T == 0``.

It deliberately does not know about Manim, quadric surface contracts, or
cutting-plane display patches.  The section solver supplies ``C = H.T @ Q @ H``
and later maps the returned two-dimensional parameterizations back to world
space.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import cos, cosh, isfinite, sin, sinh, tau
from typing import Sequence

import numpy as np

from ..topology import ParameterInterval


_TRIG_SNAP_TOLERANCE = 64.0 * float(np.finfo(float).eps)


class ConicError(ValueError):
    """A homogeneous conic cannot be classified without guessing."""


class ConicKind(str, Enum):
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    PARABOLA = "parabola"
    HYPERBOLA = "hyperbola"
    POINT = "point"
    INTERSECTING_LINES = "intersecting_lines"
    PARALLEL_LINES = "parallel_lines"
    COINCIDENT_LINE = "coincident_line"
    EMPTY = "empty"


_NONDEGENERATE_KINDS = frozenset(
    {
        ConicKind.CIRCLE,
        ConicKind.ELLIPSE,
        ConicKind.PARABOLA,
        ConicKind.HYPERBOLA,
    }
)


def _point2(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ConicError(f"{label} must be a finite two-component point")
    return result


def _matrix3(value: Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ConicError("conic matrix must be a finite 3x3 matrix")
    asymmetry = float(np.max(np.abs(result - result.T)))
    scale = max(1.0, float(np.max(np.abs(result))))
    if asymmetry > 1.0e-10 * scale:
        raise ConicError("conic matrix must be symmetric")
    return 0.5 * (result + result.T)


def _canonical_unit(value: Sequence[float]) -> np.ndarray:
    vector = _point2(value, "axis")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise ConicError("conic axis must be non-zero")
    vector = vector / length
    index = int(np.argmax(np.abs(vector)))
    if vector[index] < 0.0:
        vector = -vector
    return vector


def _right_handed(first: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = _canonical_unit(first)
    second = np.asarray((-first[1], first[0]), dtype=float)
    return first, second


def _stable_sin_cos(value: float) -> tuple[float, float]:
    sine = sin(value)
    cosine = cos(value)
    if abs(sine) <= _TRIG_SNAP_TOLERANCE:
        sine = 0.0
    elif abs(abs(sine) - 1.0) <= _TRIG_SNAP_TOLERANCE:
        sine = 1.0 if sine > 0.0 else -1.0
    if abs(cosine) <= _TRIG_SNAP_TOLERANCE:
        cosine = 0.0
    elif abs(abs(cosine) - 1.0) <= _TRIG_SNAP_TOLERANCE:
        cosine = 1.0 if cosine > 0.0 else -1.0
    return sine, cosine


@dataclass(frozen=True, slots=True)
class ConicParameterization:
    """One analytic affine branch in a canonical two-dimensional frame.

    ``first_axis`` and ``second_axis`` already contain their geometric scale:

    - ellipse/circle: ``origin + first*cos(t) + second*sin(t)``;
    - hyperbola: ``origin + branch_sign*first*cosh(t) + second*sinh(t)``;
    - parabola: ``origin + first*t + second*t**2``;
    - line: ``origin + first*t``.

    The label is semantic and deterministic for one conic matrix; the section
    layer prefixes it with the authored section identity.
    """

    kind: ConicKind
    branch_label: str
    origin: tuple[float, float]
    first_axis: tuple[float, float]
    second_axis: tuple[float, float] = (0.0, 0.0)
    branch_sign: int = 1
    natural_domain: ParameterInterval | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ConicKind):
            raise TypeError("kind must be a ConicKind")
        if not isinstance(self.branch_label, str) or not self.branch_label:
            raise ValueError("branch_label must be a non-empty string")
        _point2(self.origin, "origin")
        first = _point2(self.first_axis, "first_axis")
        _point2(self.second_axis, "second_axis")
        if float(np.linalg.norm(first)) <= 0.0:
            raise ValueError("first_axis must be non-zero")
        if self.branch_sign not in {-1, 1}:
            raise ValueError("branch_sign must be -1 or 1")
        if self.closed and self.natural_domain is None:
            raise ValueError("a closed parameterization requires a finite domain")

    def point(self, parameter: float) -> np.ndarray:
        value = float(parameter)
        if not isfinite(value):
            raise ConicError("conic parameter must be finite")
        origin = np.asarray(self.origin, dtype=float)
        first = np.asarray(self.first_axis, dtype=float)
        second = np.asarray(self.second_axis, dtype=float)
        if self.kind in {ConicKind.CIRCLE, ConicKind.ELLIPSE}:
            sine, cosine = _stable_sin_cos(value)
            return origin + cosine * first + sine * second
        if self.kind is ConicKind.HYPERBOLA:
            return (
                origin
                + self.branch_sign * cosh(value) * first
                + sinh(value) * second
            )
        if self.kind is ConicKind.PARABOLA:
            return origin + value * first + value * value * second
        if self.kind in {
            ConicKind.INTERSECTING_LINES,
            ConicKind.PARALLEL_LINES,
            ConicKind.COINCIDENT_LINE,
        }:
            return origin + value * first
        raise ConicError(f"{self.kind.value} has no curve parameterization")

    def tangent(self, parameter: float) -> np.ndarray:
        value = float(parameter)
        if not isfinite(value):
            raise ConicError("conic parameter must be finite")
        first = np.asarray(self.first_axis, dtype=float)
        second = np.asarray(self.second_axis, dtype=float)
        if self.kind in {ConicKind.CIRCLE, ConicKind.ELLIPSE}:
            sine, cosine = _stable_sin_cos(value)
            return -sine * first + cosine * second
        if self.kind is ConicKind.HYPERBOLA:
            return (
                self.branch_sign * sinh(value) * first
                + cosh(value) * second
            )
        if self.kind is ConicKind.PARABOLA:
            return first + 2.0 * value * second
        if self.kind in {
            ConicKind.INTERSECTING_LINES,
            ConicKind.PARALLEL_LINES,
            ConicKind.COINCIDENT_LINE,
        }:
            return first.copy()
        raise ConicError(f"{self.kind.value} has no curve tangent")


@dataclass(frozen=True, slots=True)
class ConicClassification:
    matrix: tuple[tuple[float, float, float], ...]
    kind: ConicKind
    matrix_rank: int
    quadratic_rank: int
    branches: tuple[ConicParameterization, ...] = ()
    isolated_points: tuple[tuple[float, float], ...] = ()
    coefficient_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        _matrix3(self.matrix)
        if not isinstance(self.kind, ConicKind):
            raise TypeError("kind must be a ConicKind")
        if self.matrix_rank not in {0, 1, 2, 3}:
            raise ValueError("matrix_rank must lie between zero and three")
        if self.quadratic_rank not in {0, 1, 2}:
            raise ValueError("quadratic_rank must lie between zero and two")
        if not isfinite(self.coefficient_tolerance) or self.coefficient_tolerance <= 0:
            raise ValueError("coefficient_tolerance must be finite and positive")
        if self.kind in _NONDEGENERATE_KINDS and not self.branches:
            raise ValueError("a non-degenerate conic requires an analytic branch")
        for point in self.isolated_points:
            _point2(point, "isolated point")

    @property
    def nondegenerate(self) -> bool:
        return self.kind in _NONDEGENERATE_KINDS

    def evaluate(self, point: Sequence[float]) -> float:
        uv = _point2(point, "conic point")
        homogeneous = np.asarray((uv[0], uv[1], 1.0), dtype=float)
        return float(homogeneous @ np.asarray(self.matrix) @ homogeneous)


def _branch(
    kind: ConicKind,
    label: str,
    origin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray | None = None,
    *,
    sign: int = 1,
    domain: ParameterInterval | None = None,
    closed: bool = False,
) -> ConicParameterization:
    return ConicParameterization(
        kind=kind,
        branch_label=label,
        origin=tuple(float(item) for item in origin),
        first_axis=tuple(float(item) for item in first),
        second_axis=tuple(
            float(item) for item in (
                np.zeros(2, dtype=float) if second is None else second
            )
        ),
        branch_sign=sign,
        natural_domain=domain,
        closed=closed,
    )


def classify_conic(
    matrix: Sequence[Sequence[float]],
    *,
    coefficient_tolerance: float = 1.0e-10,
) -> ConicClassification:
    """Classify and analytically parameterize one real affine conic.

    The input is normalized by its largest coefficient before rank and sign
    tests.  ``coefficient_tolerance`` is therefore dimensionless and local to
    this matrix; world/parameter tolerances remain the section solver's job.
    """

    tolerance = float(coefficient_tolerance)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ConicError("coefficient_tolerance must be finite and positive")
    raw = _matrix3(matrix)
    scale = float(np.max(np.abs(raw)))
    if scale <= 0.0:
        raise ConicError("the zero polynomial does not define one conic")
    value = raw / scale
    singular = np.linalg.svd(value, compute_uv=False)
    rank_threshold = tolerance * max(1.0, float(singular[0]))
    matrix_rank = int(np.count_nonzero(singular > rank_threshold))

    quadratic = value[:2, :2]
    linear = value[:2, 2]
    constant = float(value[2, 2])
    eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    quadratic_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    eigen_threshold = tolerance * quadratic_scale
    quadratic_rank = int(np.count_nonzero(np.abs(eigenvalues) > eigen_threshold))

    def result(
        kind: ConicKind,
        *,
        branches: tuple[ConicParameterization, ...] = (),
        points: tuple[tuple[float, float], ...] = (),
    ) -> ConicClassification:
        return ConicClassification(
            matrix=tuple(tuple(float(item) for item in row) for row in value),
            kind=kind,
            matrix_rank=matrix_rank,
            quadratic_rank=quadratic_rank,
            branches=branches,
            isolated_points=points,
            coefficient_tolerance=tolerance,
        )

    if quadratic_rank == 2:
        center = -np.linalg.solve(quadratic, linear)
        level = float(linear @ np.linalg.solve(quadratic, linear) - constant)
        determinant = float(np.linalg.det(quadratic))
        level_tolerance = tolerance * max(
            1.0,
            abs(constant),
            abs(float(linear @ np.linalg.solve(quadratic, linear))),
        )
        if determinant > eigen_threshold * eigen_threshold:
            if abs(level) <= level_tolerance:
                return result(
                    ConicKind.POINT,
                    points=(tuple(float(item) for item in center),),
                )
            ratios = level / eigenvalues
            if np.any(ratios <= level_tolerance):
                return result(ConicKind.EMPTY)

            # Stable major/minor ordering avoids eigenvector swaps when the
            # authored ellipse rotates.  A mathematical circle uses the
            # cutting-plane axes directly because its eigenspace is arbitrary.
            circular = abs(float(eigenvalues[1] - eigenvalues[0])) <= (
                32.0 * eigen_threshold
            )
            kind = ConicKind.CIRCLE if circular else ConicKind.ELLIPSE
            if circular:
                first_axis = np.asarray((1.0, 0.0), dtype=float)
                second_axis = np.asarray((0.0, 1.0), dtype=float)
                radius = float(np.sqrt(np.mean(ratios)))
                first_axis *= radius
                second_axis *= radius
            else:
                radii = np.sqrt(ratios)
                major_index = int(np.argmax(radii))
                minor_index = 1 - major_index
                major, minor = _right_handed(eigenvectors[:, major_index])
                if float(np.dot(minor, eigenvectors[:, minor_index])) < 0.0:
                    # The second canonical axis is defined by handedness, not
                    # by the arbitrary sign returned from eigh.
                    pass
                first_axis = major * float(radii[major_index])
                second_axis = minor * float(radii[minor_index])
            return result(
                kind,
                branches=(
                    _branch(
                        kind,
                        kind.value,
                        center,
                        first_axis,
                        second_axis,
                        domain=ParameterInterval(0.0, tau),
                        closed=True,
                    ),
                ),
            )

        if determinant < -(eigen_threshold * eigen_threshold):
            if abs(level) <= level_tolerance:
                positive_index = int(np.argmax(eigenvalues))
                negative_index = 1 - positive_index
                positive_axis, perpendicular = _right_handed(
                    eigenvectors[:, positive_index]
                )
                if float(np.dot(perpendicular, eigenvectors[:, negative_index])) < 0.0:
                    negative_axis = -perpendicular
                else:
                    negative_axis = perpendicular
                slope = float(
                    np.sqrt(
                        eigenvalues[positive_index]
                        / -eigenvalues[negative_index]
                    )
                )
                first_direction = _canonical_unit(
                    positive_axis - slope * negative_axis
                )
                second_direction = _canonical_unit(
                    positive_axis + slope * negative_axis
                )
                return result(
                    ConicKind.INTERSECTING_LINES,
                    branches=(
                        _branch(
                            ConicKind.INTERSECTING_LINES,
                            "intersecting_lines:negative",
                            center,
                            first_direction,
                        ),
                        _branch(
                            ConicKind.INTERSECTING_LINES,
                            "intersecting_lines:positive",
                            center,
                            second_direction,
                        ),
                    ),
                )

            transverse_index = next(
                index
                for index, eigenvalue in enumerate(eigenvalues)
                if eigenvalue * level > 0.0
            )
            conjugate_index = 1 - transverse_index
            transverse, conjugate = _right_handed(
                eigenvectors[:, transverse_index]
            )
            if float(np.dot(conjugate, eigenvectors[:, conjugate_index])) < 0.0:
                conjugate = -conjugate
            transverse *= float(
                np.sqrt(level / eigenvalues[transverse_index])
            )
            conjugate *= float(
                np.sqrt(-level / eigenvalues[conjugate_index])
            )
            return result(
                ConicKind.HYPERBOLA,
                branches=(
                    _branch(
                        ConicKind.HYPERBOLA,
                        "hyperbola:negative",
                        center,
                        transverse,
                        conjugate,
                        sign=-1,
                    ),
                    _branch(
                        ConicKind.HYPERBOLA,
                        "hyperbola:positive",
                        center,
                        transverse,
                        conjugate,
                        sign=1,
                    ),
                ),
            )

        raise ConicError("quadratic rank and determinant classification disagree")

    if quadratic_rank == 1:
        nonzero_index = int(np.argmax(np.abs(eigenvalues)))
        eigenvalue = float(eigenvalues[nonzero_index])
        axis, null_axis = _right_handed(eigenvectors[:, nonzero_index])
        linear_axis = float(np.dot(linear, axis))
        linear_null = float(np.dot(linear, null_axis))
        x_center = -linear_axis / eigenvalue
        remainder = constant - linear_axis * linear_axis / eigenvalue
        linear_tolerance = tolerance * max(
            1.0,
            float(np.linalg.norm(linear)),
            abs(eigenvalue),
            abs(constant),
        )
        if abs(linear_null) > linear_tolerance:
            y_vertex = -remainder / (2.0 * linear_null)
            vertex = x_center * axis + y_vertex * null_axis
            quadratic_axis = -eigenvalue / (2.0 * linear_null) * null_axis
            return result(
                ConicKind.PARABOLA,
                branches=(
                    _branch(
                        ConicKind.PARABOLA,
                        "parabola",
                        vertex,
                        axis,
                        quadratic_axis,
                    ),
                ),
            )

        ratio = -remainder / eigenvalue
        ratio_tolerance = tolerance * max(1.0, abs(ratio), abs(remainder))
        if ratio < -ratio_tolerance:
            return result(ConicKind.EMPTY)
        if abs(ratio) <= ratio_tolerance:
            origin = x_center * axis
            return result(
                ConicKind.COINCIDENT_LINE,
                branches=(
                    _branch(
                        ConicKind.COINCIDENT_LINE,
                        "coincident_line",
                        origin,
                        null_axis,
                    ),
                ),
            )
        offset = float(np.sqrt(ratio))
        return result(
            ConicKind.PARALLEL_LINES,
            branches=(
                _branch(
                    ConicKind.PARALLEL_LINES,
                    "parallel_lines:negative",
                    (x_center - offset) * axis,
                    null_axis,
                ),
                _branch(
                    ConicKind.PARALLEL_LINES,
                    "parallel_lines:positive",
                    (x_center + offset) * axis,
                    null_axis,
                ),
            ),
        )

    linear_norm = float(np.linalg.norm(linear))
    linear_tolerance = tolerance * max(1.0, linear_norm, abs(constant))
    if linear_norm > linear_tolerance:
        normal = linear / linear_norm
        direction = _canonical_unit((-normal[1], normal[0]))
        origin = -constant / (2.0 * linear_norm) * normal
        return result(
            ConicKind.COINCIDENT_LINE,
            branches=(
                _branch(
                    ConicKind.COINCIDENT_LINE,
                    "coincident_line",
                    origin,
                    direction,
                ),
            ),
        )
    if abs(constant) > linear_tolerance:
        return result(ConicKind.EMPTY)
    raise ConicError("the zero polynomial does not define a unique conic")


__all__ = [
    "ConicClassification",
    "ConicError",
    "ConicKind",
    "ConicParameterization",
    "classify_conic",
]
