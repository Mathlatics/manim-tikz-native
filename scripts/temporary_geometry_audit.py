"""Temporary property probes for the repository audit.

This file is deleted before the audit PR is finalized.
"""

from __future__ import annotations

import math
import random

import numpy as np

from polyhedron_visibility.parallel_solver import (
    ParallelView,
    segment_face_occlusion_interval,
)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * skew
    )


def _interval(start: np.ndarray, end: np.ndarray, face: np.ndarray, matrix: np.ndarray):
    return segment_face_occlusion_interval(start, end, face, ParallelView.from_matrix(matrix))


def _assert_close(actual, expected, label: str, *, atol: float = 2.0e-8) -> None:
    if actual is None or expected is None:
        if actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def main() -> None:
    random.seed(20260824)
    base_start = np.array([-2.0, 0.17, 0.0])
    base_end = np.array([2.0, -0.23, 0.0])
    base_face = np.array(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    base_matrix = np.eye(3)
    baseline = _interval(base_start, base_end, base_face, base_matrix)
    if baseline is None:
        raise AssertionError("baseline square must occlude the semantic stroke")

    reversed_face = _interval(base_start, base_end, base_face[::-1], base_matrix)
    _assert_close(reversed_face, baseline, "face winding invariance")

    reversed_stroke = _interval(base_end, base_start, base_face, base_matrix)
    _assert_close(
        reversed_stroke,
        (1.0 - baseline[1], 1.0 - baseline[0]),
        "stroke reversal",
    )

    scaled_projection = np.diag([7.5, 0.125, 31.0])
    _assert_close(
        _interval(base_start, base_end, base_face, scaled_projection),
        baseline,
        "projection row scaling",
    )

    for index in range(250):
        axis = np.array([random.uniform(-1.0, 1.0) for _ in range(3)])
        if np.linalg.norm(axis) < 1.0e-6:
            axis[0] = 1.0
        rotation = _rotation(axis, random.uniform(-math.pi, math.pi))
        scale = 10.0 ** random.uniform(-4.0, 4.0)
        translation = np.array(
            [random.uniform(-1.0e4, 1.0e4) for _ in range(3)], dtype=float
        )
        transformed_start = scale * (rotation @ base_start) + translation
        transformed_end = scale * (rotation @ base_end) + translation
        transformed_face = np.array(
            [scale * (rotation @ point) + translation for point in base_face]
        )
        transformed_matrix = base_matrix @ rotation.T
        transformed = _interval(
            transformed_start,
            transformed_end,
            transformed_face,
            transformed_matrix,
        )
        _assert_close(
            transformed,
            baseline,
            f"similarity invariance sample {index}",
            atol=3.0e-8,
        )

    print("temporary geometry audit passed: 250 similarity samples")


if __name__ == "__main__":
    main()
