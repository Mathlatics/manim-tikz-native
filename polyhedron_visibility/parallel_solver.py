from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from .contract import ContractError, ResolvedTolerance, TolerancePolicy, VisibilityModel
from .trace import (
    EdgeVisibility,
    RawOcclusionInterval,
    SkippedFace,
    VisibilityFrame,
    VisibilitySpan,
)


class SolverError(ValueError):
    """Raised when a frame cannot be solved without guessing."""


@dataclass(frozen=True)
class ParallelView:
    projection_matrix: tuple[tuple[float, float, float], ...]
    view_direction: tuple[float, float, float]

    @classmethod
    def from_matrix(cls, matrix: Sequence[Sequence[float]]) -> "ParallelView":
        value = np.asarray(matrix, dtype=float)
        if value.shape != (3, 3) or not np.all(np.isfinite(value)):
            raise SolverError("parallel projection matrix must be a finite 3x3 matrix")
        screen_x = value[0]
        screen_y = value[1]
        direction = np.cross(screen_x, screen_y)
        length = float(np.linalg.norm(direction))
        row_scale = max(float(np.linalg.norm(screen_x) * np.linalg.norm(screen_y)), 1.0e-300)
        if length <= 1.0e-12 * row_scale:
            raise SolverError("parallel projection screen axes are singular")
        direction /= length
        depth_alignment = float(np.dot(direction, value[2]))
        if abs(depth_alignment) <= 1.0e-12 * max(float(np.linalg.norm(value[2])), 1.0):
            raise SolverError("parallel projection has no usable depth direction")
        if depth_alignment < 0:
            direction *= -1.0
        canonical = tuple(tuple(float(component) for component in row) for row in value)
        return cls(canonical, tuple(float(component) for component in direction))

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.projection_matrix, dtype=float)


@dataclass(frozen=True)
class _IntervalResult:
    interval: tuple[float, float] | None
    reason: str | None = None


def _normal_for_convex_face(points: np.ndarray, tolerance: ResolvedTolerance) -> np.ndarray:
    origin = points[0]
    for index in range(1, len(points) - 1):
        normal = np.cross(points[index] - origin, points[index + 1] - origin)
        length = float(np.linalg.norm(normal))
        if length > tolerance.world * tolerance.world:
            return normal / length
    raise SolverError("occluder face is degenerate")


def _clip_greater_equal(
    lower: float,
    upper: float,
    value_at_zero: float,
    slope: float,
    threshold: float,
    parameter_epsilon: float,
) -> tuple[float, float] | None:
    if abs(slope) <= threshold * 1.0e-6 + 1.0e-300:
        if value_at_zero < threshold:
            return None
        return lower, upper
    crossing = (threshold - value_at_zero) / slope
    if slope > 0:
        lower = max(lower, crossing)
    else:
        upper = min(upper, crossing)
    if upper - lower <= parameter_epsilon:
        return None
    return lower, upper


def _segment_face_interval_result(
    segment_start: Sequence[float],
    segment_end: Sequence[float],
    face_points: Sequence[Sequence[float]],
    view: ParallelView,
    *,
    tolerance_policy: TolerancePolicy,
) -> _IntervalResult:
    start = np.asarray(segment_start, dtype=float)
    end = np.asarray(segment_end, dtype=float)
    points = np.asarray(face_points, dtype=float)
    if start.shape != (3,) or end.shape != (3,) or points.ndim != 2 or points.shape[1:] != (3,):
        raise SolverError("segment and face coordinates must be three-dimensional")
    if len(points) < 3 or not np.all(np.isfinite(points)) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise SolverError("segment and face coordinates must be finite and non-degenerate")
    segment = end - start
    segment_length = float(np.linalg.norm(segment))
    tolerance = tolerance_policy.resolve(
        np.vstack((start[None, :], end[None, :], points)), edge_length=segment_length
    )
    if segment_length <= tolerance.world:
        raise SolverError("semantic stroke has zero length")
    normal = _normal_for_convex_face(points, tolerance)
    direction = np.asarray(view.view_direction, dtype=float)
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= tolerance.angular:
        return _IntervalResult(None, "face_edge_on")

    lambda_zero = float(np.dot(points[0] - start, normal) / denominator)
    lambda_slope = float(-np.dot(segment, normal) / denominator)
    lower, upper = 0.0, 1.0
    clipped = _clip_greater_equal(
        lower,
        upper,
        lambda_zero,
        lambda_slope,
        tolerance.depth,
        tolerance.parameter,
    )
    if clipped is None:
        if abs(lambda_zero) <= tolerance.depth and abs(lambda_slope) <= tolerance.depth:
            return _IntervalResult(None, "coplanar_touch")
        return _IntervalResult(None, "face_not_in_front")
    lower, upper = clipped

    projected_zero = start + lambda_zero * direction
    projected_slope = segment + lambda_slope * direction
    for index, edge_start in enumerate(points):
        edge_end = points[(index + 1) % len(points)]
        face_edge = edge_end - edge_start
        value_zero = float(np.dot(np.cross(face_edge, projected_zero - edge_start), normal))
        value_slope = float(np.dot(np.cross(face_edge, projected_slope), normal))
        # The half-plane expression has units of length squared.  Scale the
        # boundary clearance by this particular face-edge length so that a
        # geometrically identical model behaves identically at every scale.
        boundary_threshold = tolerance.boundary * max(
            float(np.linalg.norm(face_edge)), tolerance.world
        )
        clipped = _clip_greater_equal(
            lower,
            upper,
            value_zero,
            value_slope,
            boundary_threshold,
            tolerance.parameter,
        )
        if clipped is None:
            return _IntervalResult(None, "outside_face_or_boundary_touch")
        lower, upper = clipped
    lower = min(1.0, max(0.0, float(lower)))
    upper = min(1.0, max(0.0, float(upper)))
    if upper - lower <= tolerance.parameter:
        return _IntervalResult(None, "zero_length_touch")
    return _IntervalResult((lower, upper), None)


def segment_face_occlusion_interval(
    segment_start: Sequence[float],
    segment_end: Sequence[float],
    face_points: Sequence[Sequence[float]],
    view: ParallelView,
    *,
    tolerance_policy: TolerancePolicy | None = None,
) -> tuple[float, float] | None:
    return _segment_face_interval_result(
        segment_start,
        segment_end,
        face_points,
        view,
        tolerance_policy=tolerance_policy or TolerancePolicy(),
    ).interval


def _spans_from_intervals(
    intervals: Sequence[RawOcclusionInterval], parameter_epsilon: float
) -> tuple[VisibilitySpan, ...]:
    boundaries = [0.0, 1.0]
    for item in intervals:
        boundaries.extend((item.start, item.end))
    boundaries.sort()
    unique: list[float] = []
    for value in boundaries:
        value = min(1.0, max(0.0, float(value)))
        if not unique or abs(value - unique[-1]) > parameter_epsilon:
            unique.append(value)
        else:
            unique[-1] = max(unique[-1], value)
    if unique[0] > 0:
        unique.insert(0, 0.0)
    if unique[-1] < 1:
        unique.append(1.0)

    spans: list[VisibilitySpan] = []
    for start, end in zip(unique, unique[1:]):
        if end - start <= parameter_epsilon:
            continue
        midpoint = 0.5 * (start + end)
        active = tuple(sorted(
            item.face_id
            for item in intervals
            if item.start - parameter_epsilon <= midpoint <= item.end + parameter_epsilon
        ))
        kind = "hidden" if active else "visible"
        span = VisibilitySpan(start, end, kind, active, len(active))
        if (
            spans
            and spans[-1].kind == span.kind
            and spans[-1].occluder_face_ids == span.occluder_face_ids
            and abs(spans[-1].end - span.start) <= parameter_epsilon
        ):
            previous = spans[-1]
            spans[-1] = VisibilitySpan(
                previous.start,
                span.end,
                previous.kind,
                previous.occluder_face_ids,
                previous.level,
            )
        else:
            spans.append(span)
    if not spans:
        return (VisibilitySpan(0.0, 1.0, "visible", (), 0),)
    spans[0] = VisibilitySpan(
        0.0, spans[0].end, spans[0].kind, spans[0].occluder_face_ids, spans[0].level
    )
    spans[-1] = VisibilitySpan(
        spans[-1].start, 1.0, spans[-1].kind, spans[-1].occluder_face_ids, spans[-1].level
    )
    return tuple(spans)


def _validated_positions(
    model: VisibilityModel,
    vertex_positions: Mapping[str, Sequence[float]] | None,
) -> dict[str, np.ndarray]:
    raw = model.entry_positions if vertex_positions is None else vertex_positions
    try:
        model.validate(vertex_positions=raw)
    except ContractError as exc:
        raise SolverError(f"invalid visibility frame: {exc}") from exc
    return {key: np.asarray(raw[key], dtype=float) for key in sorted(raw)}


def compute_frame_visibility(
    model: VisibilityModel,
    *,
    projection_matrix: Sequence[Sequence[float]],
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
) -> VisibilityFrame:
    policy = tolerance_policy or TolerancePolicy()
    view = ParallelView.from_matrix(projection_matrix)
    positions = _validated_positions(model, vertex_positions)
    tolerance = policy.resolve(positions)
    face_depths: list[tuple[float, str]] = []
    for face in model.faces:
        centroid = np.mean([positions[item] for item in face.vertex_ids], axis=0)
        face_depths.append((float(np.dot(centroid, view.view_direction)), face.face_id))
    face_draw_order = tuple(item[1] for item in sorted(face_depths, key=lambda item: (item[0], item[1])))

    edges: list[EdgeVisibility] = []
    for stroke in model.strokes:
        start = positions[stroke.vertex_ids[0]]
        end = positions[stroke.vertex_ids[1]]
        length = float(np.linalg.norm(end - start))
        edge_tolerance = policy.resolve(positions, edge_length=length)
        raw_intervals: list[RawOcclusionInterval] = []
        skipped: list[SkippedFace] = []
        if stroke.visibility_mode == "always_visible":
            skipped.extend(SkippedFace(face.face_id, "stroke_always_visible") for face in model.faces)
        elif stroke.visibility_mode == "always_hidden":
            raw_intervals.append(RawOcclusionInterval("__policy__", 0.0, 1.0))
        else:
            for face in model.faces:
                if face.face_id in stroke.incident_face_ids:
                    skipped.append(SkippedFace(face.face_id, "incident_face"))
                    continue
                if not face.occludes_strokes:
                    skipped.append(SkippedFace(face.face_id, "occlusion_disabled"))
                    continue
                result = _segment_face_interval_result(
                    start,
                    end,
                    [positions[item] for item in face.vertex_ids],
                    view,
                    tolerance_policy=policy,
                )
                if result.interval is None:
                    skipped.append(SkippedFace(face.face_id, result.reason or "no_occlusion"))
                    continue
                raw_intervals.append(
                    RawOcclusionInterval(face.face_id, result.interval[0], result.interval[1])
                )
        raw_intervals.sort(key=lambda item: (item.start, item.end, item.face_id))
        skipped.sort(key=lambda item: (item.face_id, item.reason))
        edges.append(
            EdgeVisibility(
                source_edge_id=stroke.source_edge_id,
                raw_intervals=tuple(raw_intervals),
                skipped_faces=tuple(skipped),
                spans=_spans_from_intervals(raw_intervals, edge_tolerance.parameter),
            )
        )
    return VisibilityFrame(
        visibility_group_id=model.visibility_group_id,
        projection_matrix=view.projection_matrix,
        view_direction=view.view_direction,
        tolerance=tolerance,
        edges=tuple(sorted(edges, key=lambda item: item.source_edge_id)),
        face_draw_order=face_draw_order,
    )


__all__ = [
    "ParallelView",
    "SolverError",
    "compute_frame_visibility",
    "segment_face_occlusion_interval",
]
