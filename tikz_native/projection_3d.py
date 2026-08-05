from __future__ import annotations

from math import cos, radians, sin, sqrt
from typing import Iterable


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Basis2 = tuple[float, float]


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _normalized(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    length = sqrt(sum(component * component for component in vector))
    if length <= 1e-12:
        raise ValueError("TikZ 三维投影的两个屏幕方向线性相关")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def matrix_from_tikz_basis(
    x_basis: Basis2,
    y_basis: Basis2,
    z_basis: Basis2,
) -> Matrix3:
    """Convert TikZ ``x/y/z={(u,v)}`` vectors into a 3x3 camera matrix.

    The first two rows reproduce TikZ's screen coordinates exactly.  TikZ does
    not retain depth; the normalized cross product supplies a deterministic
    third row solely for Manim depth sorting and later camera motion.
    """

    screen_u = (x_basis[0], y_basis[0], z_basis[0])
    screen_v = (x_basis[1], y_basis[1], z_basis[1])
    depth = _normalized(_cross(screen_u, screen_v))
    return (screen_u, screen_v, depth)


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
    """Return the minimum-norm world displacement for a screen displacement."""

    first, second = matrix[0], matrix[1]
    aa = sum(value * value for value in first)
    ab = sum(a * b for a, b in zip(first, second, strict=True))
    bb = sum(value * value for value in second)
    determinant = aa * bb - ab * ab
    if abs(determinant) <= 1e-12:
        raise ValueError("TikZ 三维投影无法反解屏幕偏移")
    coefficient_a = (bb * delta_u - ab * delta_v) / determinant
    coefficient_b = (aa * delta_v - ab * delta_u) / determinant
    return tuple(
        coefficient_a * first[index] + coefficient_b * second[index]
        for index in range(3)
    )  # type: ignore[return-value]


__all__ = [
    "Basis2",
    "Matrix3",
    "matrix_from_tikz_basis",
    "matrix_from_tikz_three_d_view",
    "project_point",
    "screen_delta_to_world",
    "tikz_three_d_view_basis",
]
