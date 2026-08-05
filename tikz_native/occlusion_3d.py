"""Reusable line-versus-face occlusion geometry for native 3D TikZ figures.

The TikZ handout helpers use parallel projection: a point on a segment is
hidden when its ray toward the viewer meets a finite triangular or
parallelogram face first.  Both the compiler's authored-view split and the
Manim camera updater call the functions in this module so that a static frame
and the first frame of a dynamic scene use exactly the same geometry.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _clip_affine_interval(
    low: float,
    high: float,
    constant: float,
    slope: float,
    minimum: float | None = None,
    maximum: float | None = None,
    *,
    tolerance: float = 1e-10,
) -> tuple[float, float] | None:
    """Clip ``[low, high]`` by bounds on ``constant + slope * t``."""

    if abs(slope) <= tolerance:
        if minimum is not None and constant < minimum - tolerance:
            return None
        if maximum is not None and constant > maximum + tolerance:
            return None
        return low, high
    if minimum is not None:
        boundary = (minimum - constant) / slope
        if slope > 0:
            low = max(low, boundary)
        else:
            high = min(high, boundary)
    if maximum is not None:
        boundary = (maximum - constant) / slope
        if slope > 0:
            high = min(high, boundary)
        else:
            low = max(low, boundary)
    if low >= high - tolerance:
        return None
    return max(0.0, low), min(1.0, high)


def parallel_view_direction(
    projection_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return the world-space ray direction toward a parallel camera.

    The first two rows of the camera matrix define screen coordinates.  Their
    cross product is therefore the direction that leaves a projected point
    unchanged.  Its sign is aligned with the matrix depth row so that positive
    ray parameters point toward the viewer, matching the TikZ helpers.
    """

    matrix = np.asarray(projection_matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("projection_matrix must have shape (3,3)")
    direction = np.cross(matrix[0], matrix[1])
    length = float(np.linalg.norm(direction))
    if length <= tolerance:
        raise ValueError("projection screen rows are linearly dependent")
    direction /= length
    if float(np.dot(direction, matrix[2])) < 0.0:
        direction = -direction
    return direction


def parallel_occlusion_interval(
    start: Sequence[float] | np.ndarray,
    end: Sequence[float] | np.ndarray,
    face: Sequence[Sequence[float]] | np.ndarray,
    view_direction: Sequence[float] | np.ndarray,
) -> tuple[float, float] | None:
    """Return the part of a segment hidden by a finite convex face.

    ``face`` must contain either three vertices or four vertices ordered as a
    triangle/parallelogram boundary.  The returned values are parameters of
    ``start + t * (end - start)``.  ``None`` means that no non-zero part of the
    segment is hidden.
    """

    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    face_array = np.asarray(face, dtype=float)
    view = np.asarray(view_direction, dtype=float)
    if start_array.shape != (3,) or end_array.shape != (3,):
        return None
    if face_array.shape not in {(3, 3), (4, 3)} or view.shape != (3,):
        return None

    segment = end_array - start_array
    anchor = face_array[0]
    basis_u = face_array[1] - anchor
    basis_v = face_array[2 if len(face_array) == 3 else 3] - anchor
    normal = np.cross(basis_u, basis_v)
    denominator = float(np.dot(view, normal))
    if abs(denominator) <= 1e-10:
        return None

    lambda_zero = float(np.dot(anchor - start_array, normal)) / denominator
    lambda_slope = -float(np.dot(segment, normal)) / denominator
    w_zero = start_array + view * lambda_zero - anchor
    w_slope = segment + view * lambda_slope
    uu = float(np.dot(basis_u, basis_u))
    uv = float(np.dot(basis_u, basis_v))
    vv = float(np.dot(basis_v, basis_v))
    gram = uu * vv - uv * uv
    if abs(gram) <= 1e-10:
        return None

    def coefficients(vector: np.ndarray) -> tuple[float, float]:
        wu = float(np.dot(vector, basis_u))
        wv = float(np.dot(vector, basis_v))
        return (
            (wu * vv - wv * uv) / gram,
            (wv * uu - wu * uv) / gram,
        )

    alpha_zero, beta_zero = coefficients(w_zero)
    alpha_slope, beta_slope = coefficients(w_slope)
    interval: tuple[float, float] | None = (0.0, 1.0)
    constraints = [
        (lambda_zero, lambda_slope, 0.0, None),
        (alpha_zero, alpha_slope, 0.0, 1.0),
        (beta_zero, beta_slope, 0.0, 1.0),
    ]
    if len(face_array) == 3:
        constraints.append(
            (
                alpha_zero + beta_zero,
                alpha_slope + beta_slope,
                None,
                1.0,
            )
        )
    for constant, slope, minimum, maximum in constraints:
        if interval is None:
            return None
        interval = _clip_affine_interval(
            interval[0],
            interval[1],
            constant,
            slope,
            minimum,
            maximum,
        )
    return interval


__all__ = ["parallel_occlusion_interval", "parallel_view_direction"]
