from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .contract import (
    ContractError,
    TolerancePolicy,
    VisibilityModel,
    _validate_convex_face_points,
)
from .geometry import GeometryContext, ResolvedGeometryContext
from .topology import ParameterInterval
from .trace import (
    EdgeVisibility,
    FaceToleranceTrace,
    RawOcclusionInterval,
    SkippedFace,
    VisibilityFrame,
    VisibilitySpan as TraceVisibilitySpan,
)
from .visibility import (
    OcclusionInterval as KernelOcclusionInterval,
    VisibilityBoundaryMode,
    partition_visibility,
)


class SolverError(ValueError):
    """Raised when a frame cannot be solved without guessing."""


def _validated_projection(
    matrix: Sequence[Sequence[float]],
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[float, float, float],
]:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise SolverError("parallel projection matrix must be a finite 3x3 matrix")
    screen_x = value[0]
    screen_y = value[1]
    direction = np.cross(screen_x, screen_y)
    length = float(np.linalg.norm(direction))
    row_scale = max(
        float(np.linalg.norm(screen_x) * np.linalg.norm(screen_y)), 1.0e-300
    )
    if length <= 1.0e-12 * row_scale:
        raise SolverError("parallel projection screen axes are singular")
    direction /= length
    depth_norm = float(np.linalg.norm(value[2]))
    depth_alignment = float(np.dot(direction, value[2]))
    if depth_norm == 0.0 or abs(depth_alignment) <= 1.0e-12 * depth_norm:
        raise SolverError("parallel projection has no usable depth direction")
    if depth_alignment < 0:
        direction *= -1.0
    canonical = tuple(tuple(float(component) for component in row) for row in value)
    return canonical, tuple(float(component) for component in direction)


@dataclass(frozen=True)
class ParallelView:
    projection_matrix: tuple[tuple[float, float, float], ...]
    view_direction: tuple[float, float, float]

    def __post_init__(self) -> None:
        canonical_matrix, expected_direction = _validated_projection(self.projection_matrix)
        direction = np.asarray(self.view_direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise SolverError("parallel view direction must be a finite three-component vector")
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0:
            raise SolverError("parallel view direction must be non-zero")
        if abs(direction_norm - 1.0) > 1.0e-9:
            raise SolverError("parallel view direction must be a unit vector")
        expected = np.asarray(expected_direction, dtype=float)
        if float(np.linalg.norm(expected - direction)) > 1.0e-9:
            raise SolverError("parallel view direction disagrees with projection matrix")
        object.__setattr__(self, "projection_matrix", canonical_matrix)
        object.__setattr__(self, "view_direction", expected_direction)

    @classmethod
    def from_matrix(cls, matrix: Sequence[Sequence[float]]) -> "ParallelView":
        canonical, direction = _validated_projection(matrix)
        return cls(canonical, direction)

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.projection_matrix, dtype=float)


@dataclass(frozen=True)
class _IntervalResult:
    interval: tuple[float, float] | None
    reason: str | None = None


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
    geometry_context: GeometryContext | None = None,
    tolerance_policy: TolerancePolicy | None = None,
) -> _IntervalResult:
    if geometry_context is not None and tolerance_policy is not None:
        if geometry_context.tolerance != tolerance_policy:
            raise SolverError(
                "geometry_context and tolerance_policy specify different policies"
            )
    context = geometry_context or GeometryContext(
        tolerance=tolerance_policy or TolerancePolicy()
    )

    start = np.asarray(segment_start, dtype=float)
    end = np.asarray(segment_end, dtype=float)
    points = np.asarray(face_points, dtype=float)
    if start.shape != (3,) or end.shape != (3,) or points.ndim != 2 or points.shape[1:] != (3,):
        raise SolverError("segment and face coordinates must be three-dimensional")
    if len(points) < 3 or not np.all(np.isfinite(points)) or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise SolverError("segment and face coordinates must be finite and non-degenerate")
    segment = end - start
    segment_length = float(np.linalg.norm(segment))
    edge_tolerance = context.resolve(
        (start, end), edge_length=segment_length
    ).resolved
    face_tolerance = context.resolve(points).resolved
    if segment_length <= context.tolerance.absolute_floor:
        raise SolverError("semantic stroke has zero length")
    try:
        normal = _validate_convex_face_points(points, face_tolerance, "occluder")
    except ContractError as exc:
        raise SolverError(str(exc)) from exc
    direction = np.asarray(view.view_direction, dtype=float)
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= face_tolerance.angular:
        return _IntervalResult(None, "face_edge_on")

    lambda_zero = float(np.dot(points[0] - start, normal) / denominator)
    lambda_slope = float(-np.dot(segment, normal) / denominator)
    lower, upper = 0.0, 1.0
    clipped = _clip_greater_equal(
        lower,
        upper,
        lambda_zero,
        lambda_slope,
        face_tolerance.depth,
        edge_tolerance.parameter,
    )
    if clipped is None:
        if (
            abs(lambda_zero) <= face_tolerance.depth
            and abs(lambda_slope) <= face_tolerance.depth
        ):
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
        boundary_threshold = face_tolerance.boundary * max(
            float(np.linalg.norm(face_edge)), face_tolerance.world
        )
        clipped = _clip_greater_equal(
            lower,
            upper,
            value_zero,
            value_slope,
            boundary_threshold,
            edge_tolerance.parameter,
        )
        if clipped is None:
            return _IntervalResult(None, "outside_face_or_boundary_touch")
        lower, upper = clipped
    lower = min(1.0, max(0.0, float(lower)))
    upper = min(1.0, max(0.0, float(upper)))
    if upper - lower <= edge_tolerance.parameter:
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
    policy = tolerance_policy or TolerancePolicy()
    return _segment_face_interval_result(
        segment_start,
        segment_end,
        face_points,
        view,
        geometry_context=GeometryContext(tolerance=policy),
    ).interval


def _spans_from_intervals(
    intervals: Sequence[RawOcclusionInterval],
    parameter_epsilon: float | None = None,
    *,
    context: ResolvedGeometryContext | None = None,
) -> tuple[TraceVisibilitySpan, ...]:
    """Adapt shared kernel spans to the frozen v1 trace schema.

    ``TOLERANCE_EXPANDED`` intentionally preserves the historical v1
    breakpoint and membership convention, including upper-clustered
    near-duplicate endpoints and tolerance-expanded ownership.  The shared
    kernel owns partitioning and semantic coalescing; this adapter only maps
    its fail-closed objects back into the persisted trace shape.
    """

    try:
        kernel_spans = partition_visibility(
            ParameterInterval(0.0, 1.0),
            (
                KernelOcclusionInterval(
                    ParameterInterval(item.start, item.end),
                    item.face_id,
                )
                for item in intervals
            ),
            context=context,
            parameter_tolerance=parameter_epsilon,
            occluder_key=lambda face_id: face_id,
            boundary_mode=VisibilityBoundaryMode.TOLERANCE_EXPANDED,
        )
    except (TypeError, ValueError) as exc:
        raise SolverError(f"invalid occlusion interval partition: {exc}") from exc
    return tuple(
        TraceVisibilitySpan(
            span.interval.start,
            span.interval.end,
            span.kind.value,
            span.occluders,
            len(span.occluders),
        )
        for span in kernel_spans
    )


def _validated_positions(
    model: VisibilityModel,
    vertex_positions: Mapping[str, Sequence[float]] | None,
    *,
    tolerance_policy: TolerancePolicy,
    require_closed_convex_manifold: bool,
) -> dict[str, np.ndarray]:
    raw = model.entry_positions if vertex_positions is None else vertex_positions
    try:
        model.validate(
            vertex_positions=raw,
            tolerance_policy=tolerance_policy,
            require_closed_convex_manifold=require_closed_convex_manifold,
        )
    except ContractError as exc:
        raise SolverError(f"invalid visibility frame: {exc}") from exc
    return {key: np.asarray(raw[key], dtype=float) for key in sorted(raw)}


def compute_frame_visibility(
    model: VisibilityModel,
    *,
    projection_matrix: Sequence[Sequence[float]],
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
    require_closed_convex_manifold: bool = False,
) -> VisibilityFrame:
    policy = tolerance_policy or TolerancePolicy()
    geometry_context = GeometryContext(tolerance=policy)
    view = ParallelView.from_matrix(projection_matrix)
    positions = _validated_positions(
        model,
        vertex_positions,
        tolerance_policy=policy,
        require_closed_convex_manifold=require_closed_convex_manifold,
    )
    surface_vertex_ids = sorted({item for face in model.faces for item in face.vertex_ids})
    tolerance_positions = (
        {item: positions[item] for item in surface_vertex_ids}
        if surface_vertex_ids
        else positions
    )
    tolerance = geometry_context.resolve(tolerance_positions).resolved
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
        edge_context = geometry_context.resolve(
            (start, end), edge_length=length
        )
        edge_tolerance = edge_context.resolved
        if length <= edge_tolerance.world:
            raise SolverError(
                f"semantic stroke {stroke.source_edge_id} has zero length"
            )
        raw_intervals: list[RawOcclusionInterval] = []
        skipped: list[SkippedFace] = []
        face_tolerances = tuple(
            FaceToleranceTrace(
                face_id=face.face_id,
                world=(resolved := geometry_context.resolve(
                    [positions[item] for item in face.vertex_ids]
                ).resolved).world,
                boundary=resolved.boundary,
                depth=resolved.depth,
                angular=resolved.angular,
            )
            for face in sorted(model.faces, key=lambda item: item.face_id)
        )
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
                    geometry_context=geometry_context,
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
                spans=_spans_from_intervals(raw_intervals, context=edge_context),
                parameter_epsilon=edge_tolerance.parameter,
                face_tolerances=face_tolerances,
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
