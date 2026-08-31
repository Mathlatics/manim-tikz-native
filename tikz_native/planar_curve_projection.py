"""Pure parallel-projection helpers for authored planar 3D curves.

The semantic circle and ellipse contracts live in the renderer-neutral
quadric package.  This module supplies the small adapter needed by TikZ's
fixed-view renderer without importing Manim or the TikZ compiler.

An affine parallel projection maps

``center + first_axis*cos(t) + second_axis*sin(t)``

to the same expression in screen coordinates.  A rank-two screen basis is an
ellipse.  A rank-one basis is a finite segment (possibly only a subsegment for
an authored arc) and must not be sent through an ill-conditioned ellipse
inverse or silently expanded to the basis' infinite support line.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, floor, isfinite, pi, tau
from typing import Sequence

import numpy as np

from polyhedron_visibility.quadrics.planar_curves import (
    Circle3DSpec,
    Ellipse3DSpec,
    PlanarCurve3DSpec,
)


_DEFAULT_RELATIVE_RANK_TOLERANCE = 64.0 * float(np.finfo(float).eps)


class PlanarCurveProjectionError(ValueError):
    """Raised when a planar curve projection cannot be certified."""


def _nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise PlanarCurveProjectionError(f"{label} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurveProjectionError(
            f"{label} must be finite and non-negative"
        ) from exc
    if not isfinite(result) or result < 0.0:
        raise PlanarCurveProjectionError(
            f"{label} must be finite and non-negative"
        )
    return result


def _screen_matrix(value: object) -> np.ndarray:
    try:
        authored = np.asarray(value, dtype=object)
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurveProjectionError(
            "parallel projection must be a finite 2x3 or 3x3 matrix"
        ) from exc
    if (
        matrix.shape not in {(2, 3), (3, 3)}
        or authored.shape != matrix.shape
        or any(isinstance(item, (bool, np.bool_)) for item in authored.flat)
        or not np.all(np.isfinite(matrix))
    ):
        raise PlanarCurveProjectionError(
            "parallel projection must be a finite 2x3 or 3x3 matrix"
        )
    screen = matrix[:2]
    scale = float(np.max(np.abs(screen)))
    if not isfinite(scale) or scale <= 0.0:
        raise PlanarCurveProjectionError(
            "parallel projection screen rows must be linearly independent"
        )
    try:
        singular = np.linalg.svd(screen / scale, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise PlanarCurveProjectionError(
            "parallel projection screen rows could not be certified"
        ) from exc
    if (
        not np.all(np.isfinite(singular))
        or float(singular[1])
        <= _DEFAULT_RELATIVE_RANK_TOLERANCE * float(singular[0])
    ):
        raise PlanarCurveProjectionError(
            "parallel projection screen rows must be linearly independent"
        )
    return screen


def _tuple2(value: Sequence[float] | np.ndarray) -> tuple[float, float]:
    result = tuple(0.0 if float(item) == 0.0 else float(item) for item in value)
    assert len(result) == 2
    return result  # type: ignore[return-value]


def _canonical_direction(value: np.ndarray) -> np.ndarray:
    direction = np.asarray(value, dtype=float)
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    return direction


def _critical_parameters(
    phase: float,
    start: float,
    end: float,
) -> tuple[float, ...]:
    """Return extrema of ``cos(t - phase)`` inside one finite interval."""

    parameters: list[float] = [start, end]
    for base in (phase, phase + pi):
        first = int(ceil((start - base) / tau))
        last = int(floor((end - base) / tau))
        parameters.extend(base + multiple * tau for multiple in range(first, last + 1))
    return tuple(sorted(set(parameters)))


@dataclass(frozen=True, slots=True)
class ProjectedPlanarCurve2D:
    """Certified affine screen image of one planar circle or ellipse.

    ``first_axis`` and ``second_axis`` always retain the direct projected
    affine basis.  For ``rank == 2`` a renderer can apply those columns to a
    unit circle.  For ``rank == 1`` it must draw exactly ``segment_start`` to
    ``segment_end``.  Rank zero is rejected by :func:`project_planar_curve_2d`.
    """

    curve_id: str
    center: tuple[float, float]
    first_axis: tuple[float, float]
    second_axis: tuple[float, float]
    singular_values: tuple[float, float]
    rank: int
    segment_start: tuple[float, float] | None = None
    segment_end: tuple[float, float] | None = None
    segment_start_offset: tuple[float, float] | None = None
    segment_end_offset: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.rank not in {1, 2}:
            raise PlanarCurveProjectionError("projected planar curve rank must be 1 or 2")
        segment_values = (
            self.segment_start,
            self.segment_end,
            self.segment_start_offset,
            self.segment_end_offset,
        )
        has_segment = all(value is not None for value in segment_values)
        if any(value is not None for value in segment_values) and not has_segment:
            raise PlanarCurveProjectionError(
                "a rank-one projected planar curve requires complete finite-segment evidence"
            )
        if (self.rank == 1) != has_segment:
            raise PlanarCurveProjectionError(
                "only a rank-one projected planar curve owns a finite segment"
            )

    @property
    def screen_basis(self) -> np.ndarray:
        return np.column_stack((self.first_axis, self.second_axis))


def project_planar_curve_2d(
    curve: PlanarCurve3DSpec,
    projection: Sequence[Sequence[float]] | np.ndarray,
    *,
    relative_rank_tolerance: float = _DEFAULT_RELATIVE_RANK_TOLERANCE,
    absolute_rank_tolerance: float = 0.0,
) -> ProjectedPlanarCurve2D:
    """Project an authored planar curve through one parallel screen chart.

    The relative threshold only decides whether the smaller singular direction
    is numerically distinguishable from zero.  No matrix inverse is used.  A
    caller that wants to approximate a merely *thin* ellipse by a segment must
    opt into that approximation by supplying an explicit absolute display
    tolerance; the default does not silently lower rendering precision.
    """

    relative = _nonnegative_finite(
        relative_rank_tolerance,
        "relative_rank_tolerance",
    )
    if relative >= 1.0:
        raise PlanarCurveProjectionError(
            "relative_rank_tolerance must be smaller than one"
        )
    absolute = _nonnegative_finite(
        absolute_rank_tolerance,
        "absolute_rank_tolerance",
    )
    if not isinstance(curve, (Circle3DSpec, Ellipse3DSpec)):
        raise TypeError("curve must be a Circle3DSpec or Ellipse3DSpec")

    matrix = _screen_matrix(projection)
    analytic = curve.lower_to_analytic_curve()
    with np.errstate(all="ignore"):
        center = matrix @ np.asarray(analytic.center, dtype=float)
        basis = matrix @ np.column_stack(
            (
                np.asarray(analytic.first_axis, dtype=float),
                np.asarray(analytic.second_axis, dtype=float),
            )
        )
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(basis)):
        raise PlanarCurveProjectionError(
            "projected planar curve lies outside the certifiable finite range"
        )

    scale = float(np.max(np.abs(basis)))
    if not isfinite(scale) or scale <= absolute:
        raise PlanarCurveProjectionError(
            "planar curve projects to a point within the configured tolerance"
        )
    try:
        left, normalized_singular_values, _right_transpose = np.linalg.svd(
            basis / scale,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as exc:
        raise PlanarCurveProjectionError(
            "projected planar curve singular values could not be certified"
        ) from exc
    singular = scale * normalized_singular_values
    if not np.all(np.isfinite(singular)):
        raise PlanarCurveProjectionError(
            "projected planar curve singular values are not finite"
        )
    sigma_max, sigma_min = (float(singular[0]), float(singular[1]))
    threshold = max(absolute, relative * sigma_max)

    common = {
        "curve_id": analytic.curve_id,
        "center": _tuple2(center),
        "first_axis": _tuple2(basis[:, 0]),
        "second_axis": _tuple2(basis[:, 1]),
        "singular_values": (sigma_max, sigma_min),
    }
    if sigma_min > threshold:
        return ProjectedPlanarCurve2D(rank=2, **common)

    direction = _canonical_direction(left[:, 0])
    first_coefficient = float(np.dot(direction, basis[:, 0]))
    second_coefficient = float(np.dot(direction, basis[:, 1]))
    amplitude = float(np.hypot(first_coefficient, second_coefficient))
    if not isfinite(amplitude) or amplitude <= absolute:
        raise PlanarCurveProjectionError(
            "rank-one planar curve has no certifiable segment extent"
        )
    phase = atan2(second_coefficient, first_coefficient)
    values = [
        first_coefficient * np.cos(parameter)
        + second_coefficient * np.sin(parameter)
        for parameter in _critical_parameters(
            phase,
            analytic.domain.start,
            analytic.domain.end,
        )
    ]
    minimum = float(min(values))
    maximum = float(max(values))
    segment_start_offset = minimum * direction
    segment_end_offset = maximum * direction
    segment_start = center + segment_start_offset
    segment_end = center + segment_end_offset
    segment_extent = segment_end_offset - segment_start_offset
    if (
        not np.all(np.isfinite(segment_start))
        or not np.all(np.isfinite(segment_end))
        or not np.all(np.isfinite(segment_extent))
        or float(np.linalg.norm(segment_extent)) <= absolute
    ):
        raise PlanarCurveProjectionError(
            "rank-one planar curve has no certifiable segment extent"
        )
    return ProjectedPlanarCurve2D(
        rank=1,
        segment_start=_tuple2(segment_start),
        segment_end=_tuple2(segment_end),
        segment_start_offset=_tuple2(segment_start_offset),
        segment_end_offset=_tuple2(segment_end_offset),
        **common,
    )


__all__ = [
    "PlanarCurveProjectionError",
    "ProjectedPlanarCurve2D",
    "project_planar_curve_2d",
]
