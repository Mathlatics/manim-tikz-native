"""Renderer-neutral helpers for projected path painter events."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def line_convex_polygon_interval(
    start: np.ndarray,
    end: np.ndarray,
    polygon: Sequence[np.ndarray],
    epsilon: float,
) -> tuple[float, float] | None:
    """Clip one projected segment against a convex polygon."""

    if len(polygon) < 3:
        return None
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= epsilon:
        return None
    area = 0.5 * sum(
        float(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[index][1] * polygon[(index + 1) % len(polygon)][0]
        )
        for index in range(len(polygon))
    )
    if abs(area) <= epsilon * epsilon:
        return None
    orientation = 1.0 if area > 0.0 else -1.0
    low, high = 0.0, 1.0
    for index, edge_start in enumerate(polygon):
        edge_end = polygon[(index + 1) % len(polygon)]
        edge = edge_end - edge_start
        value = orientation * float(
            edge[0] * (start[1] - edge_start[1])
            - edge[1] * (start[0] - edge_start[0])
        )
        slope = orientation * float(
            edge[0] * direction[1] - edge[1] * direction[0]
        )
        edge_length = float(np.linalg.norm(edge))
        threshold = -epsilon * max(edge_length, epsilon)
        # ``slope`` is a 2D cross product with units of length squared.
        # Compare it with the half-space boundary tolerance in the same units;
        # a hard unit-length floor would classify every small-scale crossing as
        # parallel and can turn a tiny endpoint touch into full-segment overlap.
        slope_tolerance = epsilon * max(edge_length, epsilon)
        if abs(slope) <= slope_tolerance:
            if value < threshold:
                return None
            continue
        crossing = (threshold - value) / slope
        if slope > 0.0:
            low = max(low, crossing)
        else:
            high = min(high, crossing)
        if high < low - epsilon / length:
            return None
    low = min(1.0, max(0.0, float(low)))
    high = min(1.0, max(0.0, float(high)))
    return None if high - low <= epsilon / length else (low, high)


def segment_intersection_parameters(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    epsilon: float,
) -> tuple[str, tuple[float, ...]] | None:
    """Return projected point or collinear-overlap parameters."""

    first = first_end - first_start
    second = second_end - second_start
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    if first_length <= epsilon or second_length <= epsilon:
        return None
    cross = float(first[0] * second[1] - first[1] * second[0])
    offset = second_start - first_start
    tolerance = epsilon * max(first_length, second_length, epsilon)
    if abs(cross) > tolerance:
        first_t = float((offset[0] * second[1] - offset[1] * second[0]) / cross)
        second_t = float((offset[0] * first[1] - offset[1] * first[0]) / cross)
        if (
            -epsilon / first_length <= first_t <= 1.0 + epsilon / first_length
            and -epsilon / second_length
            <= second_t
            <= 1.0 + epsilon / second_length
        ):
            return (
                "point",
                (
                    min(1.0, max(0.0, first_t)),
                    min(1.0, max(0.0, second_t)),
                ),
            )
        return None
    distance = abs(float(offset[0] * first[1] - offset[1] * first[0])) / first_length
    if distance > epsilon:
        return None
    axis = first / first_length
    second_low = float(np.dot(second_start - first_start, axis))
    second_high = float(np.dot(second_end - first_start, axis))
    overlap_low = max(0.0, min(second_low, second_high))
    overlap_high = min(first_length, max(second_low, second_high))
    scalar_delta = second_high - second_low
    if overlap_high - overlap_low <= epsilon or abs(scalar_delta) <= epsilon:
        return None
    first_a, first_b = overlap_low / first_length, overlap_high / first_length
    # Preserve correspondence between the two segments.  ``first_a`` and
    # ``second_a`` describe the same projected point, as do ``first_b`` and
    # ``second_b``.  The second parameters are intentionally allowed to run
    # from high to low when the second segment is oriented opposite to the
    # first; sorting them independently would move a depth-exchange root to a
    # different screen point.
    second_a = min(
        1.0,
        max(0.0, (overlap_low - second_low) / scalar_delta),
    )
    second_b = min(
        1.0,
        max(0.0, (overlap_high - second_low) / scalar_delta),
    )
    if abs(second_b - second_a) <= epsilon / second_length:
        return None
    return "overlap", (first_a, first_b, second_a, second_b)


__all__ = ["line_convex_polygon_interval", "segment_intersection_parameters"]
