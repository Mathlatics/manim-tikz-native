"""Reusable line-versus-face occlusion geometry for native 3D TikZ figures.

The TikZ handout helpers use parallel projection: a point on a segment is
hidden when its ray toward the viewer meets a finite convex face first. Both
the compiler's authored-view split and the Manim camera updater call the
functions in this module so that a static frame and the first frame of a
dynamic scene use exactly the same geometry.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


_RELATIVE_TOLERANCE = 1.0e-9
_ABSOLUTE_FLOOR = 1.0e-14
_ANGULAR_TOLERANCE = 1.0e-10
_BOUNDARY_FACTOR = 8.0
_DEPTH_FACTOR = 8.0


def _stable_norm(vector: np.ndarray) -> float | None:
    """Return a finite Euclidean norm without avoidable overflow/underflow."""

    scale = float(np.max(np.abs(vector)))
    if not np.isfinite(scale):
        return None
    if scale == 0.0:
        return 0.0
    length = scale * float(np.linalg.norm(vector / scale))
    return length if np.isfinite(length) else None


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    """Return a scale-invariant unit vector, or ``None`` for invalid input."""

    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return None
    scaled = vector / scale
    length = float(np.linalg.norm(scaled))
    if not np.isfinite(length) or length == 0.0:
        return None
    return scaled / length


def _clip_greater_equal(
    lower: float,
    upper: float,
    value_at_zero: float,
    slope: float,
    threshold: float,
    parameter_tolerance: float,
) -> tuple[float, float] | None:
    """Clip ``[lower, upper]`` by ``value_at_zero + slope*t >= threshold``."""

    slope_tolerance = threshold * 1.0e-6 + np.finfo(float).tiny
    if abs(slope) <= slope_tolerance:
        if value_at_zero < threshold:
            return None
        return lower, upper

    crossing = (threshold - value_at_zero) / slope
    if slope > 0.0:
        lower = max(lower, crossing)
    else:
        upper = min(upper, crossing)

    lower = max(0.0, lower)
    upper = min(1.0, upper)
    if upper - lower <= parameter_tolerance:
        return None
    return lower, upper


def parallel_view_direction(
    projection_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return the world-space ray direction toward a parallel camera.

    The first two matrix rows define screen coordinates. Their cross product
    therefore leaves projected screen coordinates unchanged. The calculation
    normalizes each screen row before crossing, so uniformly scaling a valid
    projection matrix cannot make it appear singular.
    """

    try:
        matrix = np.asarray(projection_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("projection_matrix must be a finite 3x3 matrix") from exc
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("projection_matrix must be a finite 3x3 matrix")
    if (
        isinstance(tolerance, bool)
        or not np.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise ValueError("tolerance must be finite and positive")

    first_scale = float(np.max(np.abs(matrix[0])))
    second_scale = float(np.max(np.abs(matrix[1])))
    if first_scale == 0.0 or second_scale == 0.0:
        raise ValueError("projection screen rows are linearly dependent")

    first = matrix[0] / first_scale
    second = matrix[1] / second_scale
    direction = np.cross(first, second)
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= float(tolerance):
        raise ValueError("projection screen rows are linearly dependent")
    direction /= length

    depth = _unit_vector(matrix[2])
    if depth is None:
        raise ValueError("projection matrix has no usable depth direction")
    alignment = float(np.dot(direction, depth))
    if abs(alignment) <= float(tolerance):
        raise ValueError("projection matrix has no usable depth direction")
    if alignment < 0.0:
        direction = -direction
    return direction


def parallel_occlusion_interval(
    start: Sequence[float] | np.ndarray,
    end: Sequence[float] | np.ndarray,
    face: Sequence[Sequence[float]] | np.ndarray,
    view_direction: Sequence[float] | np.ndarray,
) -> tuple[float, float] | None:
    """Return the non-zero part of a segment hidden by a finite convex face.

    The returned values parameterize ``start + t * (end - start)``. A face only
    occludes when it lies a positive, tolerance-aware distance toward the
    viewer. Coplanar contacts, boundary-only contacts, invalid geometry, and
    zero-length intersections return ``None``.

    Calculations are performed in coordinates local to the face scale. This
    keeps the result stable when the whole model is uniformly scaled and avoids
    losing a small face merely because the semantic stroke is very long.
    """

    try:
        start_array = np.asarray(start, dtype=float)
        end_array = np.asarray(end, dtype=float)
        face_array = np.asarray(face, dtype=float)
        view = _unit_vector(np.asarray(view_direction, dtype=float))
    except (TypeError, ValueError):
        return None
    if start_array.shape != (3,) or end_array.shape != (3,):
        return None
    if (
        face_array.ndim != 2
        or face_array.shape[1:] != (3,)
        or len(face_array) < 3
        or view is None
    ):
        return None
    if not (
        np.all(np.isfinite(start_array))
        and np.all(np.isfinite(end_array))
        and np.all(np.isfinite(face_array))
    ):
        return None

    segment = end_array - start_array
    if not np.all(np.isfinite(segment)):
        return None
    segment_length = _stable_norm(segment)
    if segment_length is None or segment_length <= _ABSOLUTE_FLOOR:
        return None
    segment_world_tolerance = max(
        _ABSOLUTE_FLOOR,
        _RELATIVE_TOLERANCE * segment_length,
    )
    parameter_tolerance = segment_world_tolerance / max(
        segment_length,
        segment_world_tolerance,
    )

    extent = np.max(face_array, axis=0) - np.min(face_array, axis=0)
    if not np.all(np.isfinite(extent)):
        return None
    face_scale = _stable_norm(extent)
    if face_scale is None or face_scale <= _ABSOLUTE_FLOOR:
        return None

    world_tolerance = max(
        _ABSOLUTE_FLOOR,
        _RELATIVE_TOLERANCE * face_scale,
    )
    world_local = world_tolerance / face_scale
    boundary_local = _BOUNDARY_FACTOR * world_local
    depth_local = _DEPTH_FACTOR * world_local

    anchor = face_array[0]
    face_local = (face_array - anchor) / face_scale
    start_local = (start_array - anchor) / face_scale
    end_local = (end_array - anchor) / face_scale
    if not (
        np.all(np.isfinite(face_local))
        and np.all(np.isfinite(start_local))
        and np.all(np.isfinite(end_local))
    ):
        return None

    normal: np.ndarray | None = None
    for index in range(1, len(face_local) - 1):
        candidate = np.cross(face_local[index], face_local[index + 1])
        length = _stable_norm(candidate)
        if length is not None and length > world_local * world_local:
            normal = candidate / length
            break
    if normal is None:
        return None

    distances = face_local @ normal
    if float(np.max(np.abs(distances))) > boundary_local:
        return None

    turns: list[float] = []
    for index in range(len(face_local)):
        before = face_local[index - 1]
        current = face_local[index]
        after = face_local[(index + 1) % len(face_local)]
        signed = float(
            np.dot(np.cross(current - before, after - current), normal)
        )
        if abs(signed) <= world_local * world_local:
            return None
        turns.append(signed)
    if min(turns) < 0.0 < max(turns):
        return None
    orientation = 1.0 if turns[0] > 0.0 else -1.0

    # Locally consistent turns are not enough to reject every self-crossing
    # polygon. Require every other vertex to stay in the same open half-plane
    # of every directed boundary edge.
    for index, edge_start in enumerate(face_local):
        next_index = (index + 1) % len(face_local)
        edge = face_local[next_index] - edge_start
        edge_length = _stable_norm(edge)
        if edge_length is None:
            return None
        threshold = boundary_local * max(edge_length, world_local)
        for point_index, point in enumerate(face_local):
            if point_index in {index, next_index}:
                continue
            signed = orientation * float(
                np.dot(np.cross(edge, point - edge_start), normal)
            )
            if signed <= threshold:
                return None

    denominator = float(np.dot(view, normal))
    if abs(denominator) <= _ANGULAR_TOLERANCE:
        return None

    segment_local = end_local - start_local
    lambda_zero = float(-np.dot(start_local, normal) / denominator)
    lambda_slope = float(-np.dot(segment_local, normal) / denominator)
    interval = _clip_greater_equal(
        0.0,
        1.0,
        lambda_zero,
        lambda_slope,
        depth_local,
        parameter_tolerance,
    )
    if interval is None:
        return None

    projected_zero = start_local + lambda_zero * view
    projected_slope = segment_local + lambda_slope * view
    lower, upper = interval
    for index, edge_start in enumerate(face_local):
        edge = face_local[(index + 1) % len(face_local)] - edge_start
        edge_length = _stable_norm(edge)
        if edge_length is None:
            return None
        threshold = boundary_local * max(edge_length, world_local)
        value_zero = orientation * float(
            np.dot(np.cross(edge, projected_zero - edge_start), normal)
        )
        value_slope = orientation * float(
            np.dot(np.cross(edge, projected_slope), normal)
        )
        interval = _clip_greater_equal(
            lower,
            upper,
            value_zero,
            value_slope,
            threshold,
            parameter_tolerance,
        )
        if interval is None:
            return None
        lower, upper = interval

    lower = min(1.0, max(0.0, float(lower)))
    upper = min(1.0, max(0.0, float(upper)))
    if upper - lower <= parameter_tolerance:
        return None
    return lower, upper


__all__ = ["parallel_occlusion_interval", "parallel_view_direction"]
