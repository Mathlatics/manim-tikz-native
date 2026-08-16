from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi
from typing import Mapping, Sequence

import numpy as np

from ..contract import TolerancePolicy
from ..parallel_solver import (
    ParallelView,
    SolverError as FrozenSolverError,
    _clip_greater_equal,
)
from .contract import OpenFaceContractError, OpenFaceVisibilityModel
from .trace import (
    OpenFaceEdgeVisibility,
    OpenFaceRawOcclusionInterval,
    OpenFaceSeamState,
    OpenFaceSkippedOccluder,
    OpenFaceToleranceTrace,
    OpenFaceVisibilityFrame,
    OpenFaceVisibilitySpan,
)


class OpenFaceSolverError(ValueError):
    """A stable fail-closed error from one open-face frame attempt."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class _OpenFaceIntervalResult:
    interval: tuple[float, float] | None
    reason: str | None = None


def _segment_open_face_interval_result(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    face_vertex_ids: Sequence[str],
    positions: Mapping[str, np.ndarray],
    inclusive_boundary_edges: set[tuple[str, str]],
    view: ParallelView,
    *,
    tolerance_policy: TolerancePolicy,
) -> _OpenFaceIntervalResult:
    """Clip one line against one face without opening cracks at hinge seams.

    The frozen closed-polyhedron core deliberately requires positive clearance
    from every polygon edge.  That is correct for an isolated maximal face, but
    two articulated panels which meet at a declared hinge would both retreat
    from their common edge and leave a visible epsilon-sized crack.  Only the
    declared hinge edge is therefore inclusive here; every external boundary
    keeps the frozen strict-clearance rule.
    """

    start = np.asarray(segment_start, dtype=float)
    end = np.asarray(segment_end, dtype=float)
    points = np.asarray([positions[item] for item in face_vertex_ids], dtype=float)
    segment = end - start
    segment_length = float(np.linalg.norm(segment))
    edge_tolerance = tolerance_policy.resolve((start, end), edge_length=segment_length)
    face_tolerance = tolerance_policy.resolve(points)
    if segment_length <= edge_tolerance.world:
        raise OpenFaceSolverError("DEGENERATE_STROKE", "semantic stroke has zero length")

    origin = points[0]
    normal: np.ndarray | None = None
    for index in range(1, len(points) - 1):
        candidate = np.cross(points[index] - origin, points[index + 1] - origin)
        candidate_length = float(np.linalg.norm(candidate))
        if candidate_length > face_tolerance.world * face_tolerance.world:
            normal = candidate / candidate_length
            break
    if normal is None:
        raise OpenFaceSolverError("INVALID_FACE_GEOMETRY", "occluding face is degenerate")

    direction = np.asarray(view.view_direction, dtype=float)
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= face_tolerance.angular:
        return _OpenFaceIntervalResult(None, "face_edge_on")

    lambda_zero = float(np.dot(points[0] - start, normal) / denominator)
    lambda_slope = float(-np.dot(segment, normal) / denominator)
    clipped = _clip_greater_equal(
        0.0,
        1.0,
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
            return _OpenFaceIntervalResult(None, "coplanar_touch")
        return _OpenFaceIntervalResult(None, "face_not_in_front")
    lower, upper = clipped

    projected_zero = start + lambda_zero * direction
    projected_slope = segment + lambda_slope * direction
    for index, edge_start in enumerate(points):
        next_index = (index + 1) % len(points)
        edge_end = points[next_index]
        face_edge = edge_end - edge_start
        value_zero = float(np.dot(np.cross(face_edge, projected_zero - edge_start), normal))
        value_slope = float(np.dot(np.cross(face_edge, projected_slope), normal))
        edge_key = tuple(
            sorted((face_vertex_ids[index], face_vertex_ids[next_index]))
        )
        if edge_key in inclusive_boundary_edges:
            # Zero is inclusive.  Passing a negative clearance into the frozen
            # helper would invalidate its near-zero-slope assumption.
            boundary_threshold = 0.0
        else:
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
            return _OpenFaceIntervalResult(None, "outside_face_or_boundary_touch")
        lower, upper = clipped
    lower = min(1.0, max(0.0, float(lower)))
    upper = min(1.0, max(0.0, float(upper)))
    if upper - lower <= edge_tolerance.parameter:
        return _OpenFaceIntervalResult(None, "zero_length_touch")
    return _OpenFaceIntervalResult((lower, upper))


def _spans_from_intervals(
    intervals: Sequence[OpenFaceRawOcclusionInterval],
    parameter_epsilon: float,
) -> tuple[OpenFaceVisibilitySpan, ...]:
    boundaries = [0.0, 1.0]
    for interval in intervals:
        boundaries.extend((interval.start, interval.end))
    boundaries.sort()
    unique: list[float] = []
    for raw in boundaries:
        value = min(1.0, max(0.0, float(raw)))
        if not unique or abs(value - unique[-1]) > parameter_epsilon:
            unique.append(value)
        else:
            unique[-1] = max(unique[-1], value)
    if unique[0] > 0.0:
        unique.insert(0, 0.0)
    if unique[-1] < 1.0:
        unique.append(1.0)

    spans: list[OpenFaceVisibilitySpan] = []
    for start, end in zip(unique, unique[1:]):
        if end - start <= parameter_epsilon:
            continue
        midpoint = 0.5 * (start + end)
        active = tuple(
            sorted(
                (
                    interval
                    for interval in intervals
                    if interval.start - parameter_epsilon
                    <= midpoint
                    <= interval.end + parameter_epsilon
                ),
                key=lambda item: (item.face_id, item.logical_surface_id),
            )
        )
        face_ids = tuple(item.face_id for item in active)
        surface_ids = tuple(sorted({item.logical_surface_id for item in active}))
        kind = "hidden" if active else "visible"
        span = OpenFaceVisibilitySpan(
            start=start,
            end=end,
            kind=kind,
            occluder_face_ids=face_ids,
            occluder_logical_surface_ids=surface_ids,
            face_level=len(face_ids),
            surface_level=len(surface_ids),
        )
        if (
            spans
            and spans[-1].kind == span.kind
            and spans[-1].occluder_face_ids == span.occluder_face_ids
            and spans[-1].occluder_logical_surface_ids
            == span.occluder_logical_surface_ids
            and abs(spans[-1].end - span.start) <= parameter_epsilon
        ):
            previous = spans[-1]
            spans[-1] = OpenFaceVisibilitySpan(
                previous.start,
                span.end,
                previous.kind,
                previous.occluder_face_ids,
                previous.occluder_logical_surface_ids,
                previous.face_level,
                previous.surface_level,
            )
        else:
            spans.append(span)
    if not spans:
        return (OpenFaceVisibilitySpan(0.0, 1.0, "visible"),)
    first = spans[0]
    spans[0] = OpenFaceVisibilitySpan(
        0.0,
        first.end,
        first.kind,
        first.occluder_face_ids,
        first.occluder_logical_surface_ids,
        first.face_level,
        first.surface_level,
    )
    last = spans[-1]
    spans[-1] = OpenFaceVisibilitySpan(
        last.start,
        1.0,
        last.kind,
        last.occluder_face_ids,
        last.occluder_logical_surface_ids,
        last.face_level,
        last.surface_level,
    )
    return tuple(spans)


def _seam_states(
    model: OpenFaceVisibilityModel,
    positions: Mapping[str, np.ndarray],
    normals: Mapping[str, np.ndarray],
    policy: TolerancePolicy,
) -> tuple[OpenFaceSeamState, ...]:
    result: list[OpenFaceSeamState] = []
    for seam in sorted(model.seams, key=lambda item: item.seam_id):
        first_id, second_id = sorted(seam.face_ids)
        first_face = model.face_map[first_id]
        second_face = model.face_map[second_id]
        first_normal = normals[first_id]
        second_normal = normals[second_id]
        axis_start, axis_end = (positions[item] for item in seam.vertex_ids)
        axis = axis_end - axis_start
        axis_length = float(np.linalg.norm(axis))
        axis /= axis_length
        cosine = float(np.clip(np.dot(first_normal, second_normal), -1.0, 1.0))
        signed_sine = float(np.dot(np.cross(first_normal, second_normal), axis))
        angle = float(acos(cosine))
        if angle <= policy.angular:
            state = "coplanar_same_normal"
            angle = 0.0
            cosine = 1.0
            signed_sine = 0.0
        elif pi - angle <= policy.angular:
            state = "coplanar_opposite_normal"
            angle = float(pi)
            cosine = -1.0
            signed_sine = 0.0
        else:
            state = "open"
            if abs(signed_sine) <= policy.angular:
                signed_sine = 0.0
        pair_points = [
            positions[vertex_id]
            for face in (first_face, second_face)
            for vertex_id in face.vertex_ids
        ]
        seam_tolerance = policy.resolve(pair_points, edge_length=axis_length)
        result.append(
            OpenFaceSeamState(
                seam_id=seam.seam_id,
                policy=seam.policy,
                face_ids=(first_id, second_id),
                logical_surface_ids=(
                    first_face.logical_surface_id,
                    second_face.logical_surface_id,
                ),
                vertex_ids=seam.vertex_ids,
                state=state,
                dihedral_radians=angle,
                cosine=cosine,
                signed_sine=signed_sine,
                world_tolerance=seam_tolerance.world,
                angular_tolerance=seam_tolerance.angular,
            )
        )
    return tuple(result)


def compute_open_face_visibility(
    model: OpenFaceVisibilityModel,
    *,
    projection_matrix: Sequence[Sequence[float]],
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
) -> OpenFaceVisibilityFrame:
    if not isinstance(model, OpenFaceVisibilityModel):
        raise OpenFaceSolverError(
            "INVALID_MODEL", "model must be an OpenFaceVisibilityModel"
        )
    policy = tolerance_policy or TolerancePolicy()
    try:
        validated = model._validated_frame(
            vertex_positions=vertex_positions,
            tolerance_policy=policy,
        )
    except OpenFaceContractError as exc:
        raise OpenFaceSolverError(exc.code, f"invalid open-face frame: {exc}") from exc
    try:
        view = ParallelView.from_matrix(projection_matrix)
    except FrozenSolverError as exc:
        raise OpenFaceSolverError("INVALID_PROJECTION", str(exc)) from exc

    positions = validated.positions
    surface_vertex_ids = sorted(
        {vertex_id for face in model.faces for vertex_id in face.vertex_ids}
    )
    tolerance = policy.resolve({key: positions[key] for key in surface_vertex_ids})
    face_tolerances = tuple(
        OpenFaceToleranceTrace(
            face_id=face.face_id,
            logical_surface_id=face.logical_surface_id,
            world=validated.face_tolerances[face.face_id].world,
            boundary=validated.face_tolerances[face.face_id].boundary,
            depth=validated.face_tolerances[face.face_id].depth,
            angular=validated.face_tolerances[face.face_id].angular,
        )
        for face in sorted(model.faces, key=lambda item: item.face_id)
    )
    seam_states = _seam_states(model, positions, validated.face_normals, policy)

    face_depths: list[tuple[float, str]] = []
    for face in model.faces:
        centroid = np.mean([positions[item] for item in face.vertex_ids], axis=0)
        face_depths.append((float(np.dot(centroid, view.view_direction)), face.face_id))
    face_draw_order = tuple(
        item[1] for item in sorted(face_depths, key=lambda item: (item[0], item[1]))
    )
    inclusive_edges_by_face: dict[str, set[tuple[str, str]]] = {
        face.face_id: set() for face in model.faces
    }
    for seam in model.seams:
        edge_key = tuple(sorted(seam.vertex_ids))
        for face_id in seam.face_ids:
            inclusive_edges_by_face[face_id].add(edge_key)

    edges: list[OpenFaceEdgeVisibility] = []
    for stroke in model.strokes:
        start = positions[stroke.vertex_ids[0]]
        end = positions[stroke.vertex_ids[1]]
        length = float(np.linalg.norm(end - start))
        edge_tolerance = policy.resolve((start, end), edge_length=length)
        raw_intervals: list[OpenFaceRawOcclusionInterval] = []
        skipped: list[OpenFaceSkippedOccluder] = []
        if stroke.visibility_mode == "always_visible":
            skipped.extend(
                OpenFaceSkippedOccluder(
                    face.face_id,
                    face.logical_surface_id,
                    "stroke_always_visible",
                )
                for face in model.faces
            )
        elif stroke.visibility_mode == "always_hidden":
            raw_intervals.append(
                OpenFaceRawOcclusionInterval("__policy__", "__policy__", 0.0, 1.0)
            )
        else:
            for face in model.faces:
                if face.face_id in stroke.incident_face_ids:
                    skipped.append(
                        OpenFaceSkippedOccluder(
                            face.face_id, face.logical_surface_id, "incident_face"
                        )
                    )
                    continue
                if face.face_id in stroke.excluded_occluder_face_ids:
                    skipped.append(
                        OpenFaceSkippedOccluder(
                            face.face_id,
                            face.logical_surface_id,
                            "excluded_coplanar_stroke",
                        )
                    )
                    continue
                if not face.occludes_strokes:
                    skipped.append(
                        OpenFaceSkippedOccluder(
                            face.face_id,
                            face.logical_surface_id,
                            "occlusion_disabled",
                        )
                    )
                    continue
                try:
                    interval_result = _segment_open_face_interval_result(
                        start,
                        end,
                        face.vertex_ids,
                        positions,
                        inclusive_edges_by_face[face.face_id],
                        view,
                        tolerance_policy=policy,
                    )
                except (FrozenSolverError, OpenFaceSolverError) as exc:
                    raise OpenFaceSolverError("OCCLUSION_INTERVAL_FAILED", str(exc)) from exc
                if interval_result.interval is None:
                    skipped.append(
                        OpenFaceSkippedOccluder(
                            face.face_id,
                            face.logical_surface_id,
                            interval_result.reason or "no_occlusion",
                        )
                    )
                    continue
                raw_intervals.append(
                    OpenFaceRawOcclusionInterval(
                        face.face_id,
                        face.logical_surface_id,
                        interval_result.interval[0],
                        interval_result.interval[1],
                    )
                )
        raw_intervals.sort(
            key=lambda item: (
                item.start,
                item.end,
                item.face_id,
                item.logical_surface_id,
            )
        )
        skipped.sort(key=lambda item: (item.face_id, item.logical_surface_id, item.reason))
        edges.append(
            OpenFaceEdgeVisibility(
                source_edge_id=stroke.source_edge_id,
                raw_intervals=tuple(raw_intervals),
                skipped_occluders=tuple(skipped),
                spans=_spans_from_intervals(raw_intervals, edge_tolerance.parameter),
                parameter_epsilon=edge_tolerance.parameter,
            )
        )

    return OpenFaceVisibilityFrame(
        visibility_group_id=model.visibility_group_id,
        model_schema=model.schema,
        topology=model.topology,
        projection_matrix=view.projection_matrix,
        view_direction=view.view_direction,
        tolerance=tolerance,
        face_tolerances=face_tolerances,
        seam_states=seam_states,
        edges=tuple(sorted(edges, key=lambda item: item.source_edge_id)),
        advisory_face_draw_order=face_draw_order,
    )


__all__ = ["OpenFaceSolverError", "compute_open_face_visibility"]
