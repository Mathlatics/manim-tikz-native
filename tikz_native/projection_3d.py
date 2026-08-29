from __future__ import annotations

from math import cos, hypot, isfinite, radians, sin
from typing import Iterable


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Basis2 = tuple[float, float]


# The determinant of the normalized two-row Gram matrix is sin(theta)^2.
# Comparing this dimensionless value keeps projection validation independent of
# the authored TikZ unit scale while rejecting an inverse whose condition is too
# poor for stable camera motion and screen-offset recovery.
_RELATIVE_GRAM_DETERMINANT_TOLERANCE = 1.0e-12


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _unit_vector(
    vector: tuple[float, float, float],
    *,
    error_message: str,
) -> tuple[tuple[float, float, float], float]:
    """Return a finite unit vector without overflow or a hidden scale floor."""

    length = hypot(*vector)
    if not isfinite(length) or length == 0.0:
        raise ValueError(error_message)
    unit = tuple(component / length for component in vector)
    if not all(isfinite(component) for component in unit):
        raise ValueError(error_message)
    return unit, length  # type: ignore[return-value]


def _relative_gram_determinant(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
    float,
]:
    """Normalize two screen rows and return their scale-free Gram data."""

    error = "TikZ 三维投影的屏幕方向必须是有限非零向量"
    unit_first, first_length = _unit_vector(first, error_message=error)
    unit_second, second_length = _unit_vector(second, error_message=error)
    cross = _cross(unit_first, unit_second)
    cross_length = hypot(*cross)
    determinant = cross_length * cross_length
    if (
        not isfinite(determinant)
        or determinant <= _RELATIVE_GRAM_DETERMINANT_TOLERANCE
    ):
        raise ValueError("TikZ 三维投影的两个屏幕方向线性相关")
    return (
        unit_first,
        unit_second,
        cross,
        cross_length,
        first_length,
        second_length,
    )


def matrix_from_tikz_basis(
    x_basis: Basis2,
    y_basis: Basis2,
    z_basis: Basis2,
) -> Matrix3:
    """Convert TikZ ``x/y/z={(u,v)}`` vectors into a 3x3 camera matrix.

    The first two rows reproduce TikZ's screen coordinates exactly. TikZ does
    not retain depth; the normalized cross product supplies a deterministic
    third row solely for Manim depth sorting and later camera motion. Linear
    independence is tested on normalized rows, so changing only the TikZ unit
    scale cannot change whether the projection is accepted.
    """

    screen_u = (x_basis[0], y_basis[0], z_basis[0])
    screen_v = (x_basis[1], y_basis[1], z_basis[1])
    _, _, cross, cross_length, _, _ = _relative_gram_determinant(
        screen_u,
        screen_v,
    )
    depth = tuple(component / cross_length for component in cross)
    return (screen_u, screen_v, depth)  # type: ignore[return-value]


def tikz_three_d_view_basis(
    azimuth_degrees: float,
    elevation_degrees: float,
) -> tuple[Basis2, Basis2, Basis2]:
    """Return the basis used by TikZ's perspective-library ``3d view`` key.

    This follows ``tikzlibraryperspective.code.tex`` exactly:

    ``x=(cos az, -sin az sin el)``,
    ``y=(sin az,  cos az sin el)``,
    ``z=(0,       cos el)``.
    """

    azimuth = radians(azimuth_degrees)
    elevation = radians(elevation_degrees)
    return (
        (cos(azimuth), -sin(azimuth) * sin(elevation)),
        (sin(azimuth), cos(azimuth) * sin(elevation)),
        (0.0, cos(elevation)),
    )


def matrix_from_tikz_three_d_view(
    azimuth_degrees: float,
    elevation_degrees: float,
) -> Matrix3:
    return matrix_from_tikz_basis(
        *tikz_three_d_view_basis(azimuth_degrees, elevation_degrees)
    )


def project_point(
    matrix: Matrix3,
    point: Iterable[float],
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in point)
    if len(values) != 3:
        raise ValueError("三维投影点必须有三个分量")
    return tuple(
        sum(row[index] * values[index] for index in range(3))
        for row in matrix
    )  # type: ignore[return-value]


def screen_delta_to_world(
    matrix: Matrix3,
    delta_u: float,
    delta_v: float,
) -> tuple[float, float, float]:
    """Return the minimum-norm world displacement for a screen displacement.

    The pseudoinverse is evaluated in a normalized row basis. This avoids
    overflow for very large TikZ units, underflow for very small units, and an
    absolute determinant threshold whose answer changes under uniform scaling.
    """

    first, second = matrix[0], matrix[1]
    try:
        (
            unit_first,
            unit_second,
            _,
            cross_length,
            first_length,
            second_length,
        ) = _relative_gram_determinant(first, second)
    except ValueError as exc:
        raise ValueError("TikZ 三维投影无法反解屏幕偏移") from exc

    first_delta = float(delta_u)
    second_delta = float(delta_v)
    if not isfinite(first_delta) or not isfinite(second_delta):
        raise ValueError("TikZ 三维投影的屏幕偏移必须是有限数")
    normalized_u = first_delta / first_length
    normalized_v = second_delta / second_length
    if not isfinite(normalized_u) or not isfinite(normalized_v):
        raise ValueError("TikZ 三维投影无法反解屏幕偏移")

    cosine = sum(
        left * right
        for left, right in zip(unit_first, unit_second, strict=True)
    )
    cosine = max(-1.0, min(1.0, cosine))
    perpendicular_second = tuple(
        (unit_second[index] - cosine * unit_first[index]) / cross_length
        for index in range(3)
    )
    perpendicular_amount = (
        normalized_v - cosine * normalized_u
    ) / cross_length
    result = tuple(
        normalized_u * unit_first[index]
        + perpendicular_amount * perpendicular_second[index]
        for index in range(3)
    )
    if not all(isfinite(component) for component in result):
        raise ValueError("TikZ 三维投影无法反解屏幕偏移")
    return result  # type: ignore[return-value]


__all__ = [
    "Basis2",
    "Matrix3",
    "matrix_from_tikz_basis",
    "matrix_from_tikz_three_d_view",
    "project_point",
    "screen_delta_to_world",
    "tikz_three_d_view_basis",
]
