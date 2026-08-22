from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi
from typing import Mapping, Sequence

import numpy as np

from ..compositor import (
    CompositorCycleError,
    PainterConstraint,
    stable_topological_sort,
)
from ..contract import TolerancePolicy
from ..parallel_solver import (
    ParallelView,
    SolverError as FrozenSolverError,
    _clip_greater_equal,
)
from ..topology import ParameterInterval
from ..visibility import (
    OcclusionInterval,
    VisibilityBoundaryMode,
    partition_visibility,
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


@dataclass(frozen=True, order=True, slots=True)
class _OpenFaceOccluderIdentity:
    """Stable dual identity carried through the shared visibility layer."""

    face_id: str
    logical_surface_id: str


@dataclass(frozen=True, slots=True)
class _OpenFaceFaceRelation:
    """One pairwise whole-face painter relation produced by geometry."""

    far_face_id: str
    near_face_id: str
    reason: str
    minimum_depth_difference: float
    maximum_depth_difference: float
    overlap_measure: float


@dataclass(frozen=True, slots=True)
class _OpenFaceFacePainterSolution:
    """Pairwise face relations plus the frozen v1 total order."""

    draw_order: tuple[str, ...]
    relations: tuple[_OpenFaceFaceRelation, ...]
    projected_faces: Mapping[str, tuple[np.ndarray, ...]]
    screen_epsilon: float
    area_epsilon: float


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _polygon_signed_area(points: Sequence[np.ndarray]) -> float:
    if len(points) < 3:
        return 0.0
    return 0.5 * sum(
        _cross2(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def _canonical_ccw_polygon(
    points: Sequence[Sequence[float]],
    *,
    point_epsilon: float,
) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for raw in points:
        point = np.asarray(raw, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise OpenFaceSolverError(
                "INVALID_FACE_PROJECTION",
                "face projection must contain finite two-component points",
            )
        if result and float(np.linalg.norm(point - result[-1])) <= point_epsilon:
            continue
        result.append(point)
    if len(result) > 1 and float(np.linalg.norm(result[0] - result[-1])) <= point_epsilon:
        result.pop()
    if len(result) < 3:
        return ()
    area = _polygon_signed_area(result)
    if area < 0.0:
        result.reverse()
    return tuple(result)


def _convex_polygon_intersection(
    subject: Sequence[np.ndarray],
    clip: Sequence[np.ndarray],
    *,
    boundary_epsilon: float,
) -> tuple[np.ndarray, ...]:
    """Return the convex 2D overlap using deterministic Sutherland-Hodgman."""

    output = list(subject)
    for edge_index, clip_start in enumerate(clip):
        if not output:
            break
        clip_end = clip[(edge_index + 1) % len(clip)]
        clip_edge = clip_end - clip_start
        input_points = output
        output = []
        previous = input_points[-1]
        previous_value = _cross2(clip_edge, previous - clip_start)
        previous_inside = previous_value >= -boundary_epsilon
        for current in input_points:
            current_value = _cross2(clip_edge, current - clip_start)
            current_inside = current_value >= -boundary_epsilon
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                if abs(denominator) > 1.0e-300:
                    amount = previous_value / denominator
                    output.append(previous + amount * (current - previous))
            if current_inside:
                output.append(current)
            previous = current
            previous_value = current_value
            previous_inside = current_inside
        canonical: list[np.ndarray] = []
        for point in output:
            if (
                not canonical
                or float(np.linalg.norm(point - canonical[-1])) > boundary_epsilon
            ):
                canonical.append(point)
        if (
            len(canonical) > 1
            and float(np.linalg.norm(canonical[0] - canonical[-1]))
            <= boundary_epsilon
        ):
            canonical.pop()
        output = canonical
    return tuple(output)


def _face_depth_at_screen_point(
    screen_point: np.ndarray,
    face_vertex_ids: Sequence[str],
    positions: Mapping[str, np.ndarray],
    normal: np.ndarray,
    view: ParallelView,
) -> float:
    projection = np.asarray(view.projection_matrix, dtype=float)
    anchor = positions[face_vertex_ids[0]]
    system = np.asarray((projection[0], projection[1], normal), dtype=float)
    target = np.asarray(
        (screen_point[0], screen_point[1], float(np.dot(normal, anchor))),
        dtype=float,
    )
    try:
        world = np.linalg.solve(system, target)
    except np.linalg.LinAlgError as exc:
        raise OpenFaceSolverError(
            "INVALID_FACE_PROJECTION",
            "projected face has no stable parallel-view depth function",
        ) from exc
    if not np.all(np.isfinite(world)):
        raise OpenFaceSolverError(
            "INVALID_FACE_PROJECTION",
            "projected face depth is not finite",
        )
    return float(np.dot(world, np.asarray(view.view_direction, dtype=float)))


def _solve_face_painter(
    model: OpenFaceVisibilityModel,
    positions: Mapping[str, np.ndarray],
    normals: Mapping[str, np.ndarray],
    view: ParallelView,
    policy: TolerancePolicy,
) -> _OpenFaceFacePainterSolution:
    """Return exact pairwise whole-face relations and one stable total order.

    Depth is compared only inside the actual projected overlap polygon.  For
    planar faces under parallel projection the depth difference is affine, so
    its extrema occur at overlap vertices.  A sign change therefore proves
    that no single whole-face painter order can represent the frame.

    The pairwise relations are retained for the unified face/path compositor;
    the existing visibility trace continues to expose only ``draw_order``.
    """

    projection = np.asarray(view.projection_matrix, dtype=float)
    raw_projected: dict[str, tuple[np.ndarray, ...]] = {}
    all_screen_points: list[np.ndarray] = []
    ordered_faces = tuple(sorted(model.faces, key=lambda item: item.face_id))
    for face in ordered_faces:
        projected = tuple(
            np.asarray((projection @ positions[vertex_id])[:2], dtype=float)
            for vertex_id in face.vertex_ids
        )
        raw_projected[face.face_id] = projected
        all_screen_points.extend(projected)

    screen_values = np.asarray(all_screen_points, dtype=float)
    extent = np.max(screen_values, axis=0) - np.min(screen_values, axis=0)
    screen_scale = max(float(np.linalg.norm(extent)), policy.absolute_floor)
    screen_world = max(policy.absolute_floor, policy.relative * screen_scale)
    boundary_epsilon = policy.boundary_factor * screen_world
    area_epsilon = boundary_epsilon * screen_scale
    projected_by_face = {
        face.face_id: _canonical_ccw_polygon(
            raw_projected[face.face_id],
            point_epsilon=boundary_epsilon,
        )
        for face in ordered_faces
    }

    constraints: list[PainterConstraint[str]] = []
    relations: list[_OpenFaceFaceRelation] = []
    for first_index, first in enumerate(ordered_faces):
        first_polygon = projected_by_face[first.face_id]
        if (
            not first_polygon
            or abs(_polygon_signed_area(first_polygon)) <= area_epsilon
        ):
            continue
        for second in ordered_faces[first_index + 1 :]:
            second_polygon = projected_by_face[second.face_id]
            if (
                not second_polygon
                or abs(_polygon_signed_area(second_polygon)) <= area_epsilon
            ):
                continue
            overlap = _convex_polygon_intersection(
                first_polygon,
                second_polygon,
                boundary_epsilon=boundary_epsilon,
            )
            overlap_measure = abs(_polygon_signed_area(overlap))
            if len(overlap) < 3 or overlap_measure <= area_epsilon:
                continue
            pair_points = tuple(
                positions[vertex_id]
                for face in (first, second)
                for vertex_id in face.vertex_ids
            )
            depth_epsilon = policy.resolve(pair_points).depth
            differences = tuple(
                _face_depth_at_screen_point(
                    point,
                    first.vertex_ids,
                    positions,
                    normals[first.face_id],
                    view,
                )
                - _face_depth_at_screen_point(
                    point,
                    second.vertex_ids,
                    positions,
                    normals[second.face_id],
                    view,
                )
                for point in overlap
            )
            minimum = min(differences)
            maximum = max(differences)
            if minimum < -depth_epsilon and maximum > depth_epsilon:
                raise OpenFaceSolverError(
                    "FACE_ORDER_REQUIRES_SPLITTING",
                    f"faces {first.face_id!r} and {second.face_id!r} cross inside "
                    "their projected overlap",
                )
            if maximum <= depth_epsilon and minimum < -depth_epsilon:
                far_id, near_id = first.face_id, second.face_id
                reason = "face_depth"
            elif minimum >= -depth_epsilon and maximum > depth_epsilon:
                far_id, near_id = second.face_id, first.face_id
                reason = "face_depth"
            else:
                # Coplanar/touching overlap has no geometric near side.  The
                # frozen identity tie-break keeps traces byte deterministic.
                far_id, near_id = sorted((first.face_id, second.face_id))
                reason = "face_coplanar_tie"
            constraints.append(PainterConstraint(far_id, near_id))
            relations.append(
                _OpenFaceFaceRelation(
                    far_id,
                    near_id,
                    reason,
                    float(minimum),
                    float(maximum),
                    float(overlap_measure),
                )
            )

    face_ids = tuple(face.face_id for face in ordered_faces)
    try:
        draw_order = stable_topological_sort(
            face_ids,
            constraints,
            key=lambda face_id: face_id,
        )
    except CompositorCycleError as exc:
        cyclic = sorted(str(face_id) for face_id in exc.unresolved)
        raise OpenFaceSolverError(
            "FACE_ORDER_CYCLE",
            "whole-face painter order contains a cycle: " + ", ".join(cyclic),
        ) from exc
    return _OpenFaceFacePainterSolution(
        draw_order=draw_order,
        relations=tuple(relations),
        projected_faces=projected_by_face,
        screen_epsilon=boundary_epsilon,
        area_epsilon=area_epsilon,
    )


def _authoritative_face_draw_order(
    model: OpenFaceVisibilityModel,
    positions: Mapping[str, np.ndarray],
    normals: Mapping[str, np.ndarray],
    view: ParallelView,
    policy: TolerancePolicy,
) -> tuple[str, ...]:
    """Compatibility wrapper returning the frozen v1 whole-face order."""

    return _solve_face_painter(
        model,
        positions,
        normals,
        view,
        policy,
    ).draw_order


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
    """Adapt shared visibility spans back to the frozen open-face v1 trace."""

    hidden = tuple(
        OcclusionInterval(
            ParameterInterval(interval.start, interval.end),
            _OpenFaceOccluderIdentity(
                interval.face_id,
                interval.logical_surface_id,
            ),
        )
        for interval in intervals
    )
    kernel_spans = partition_visibility(
        ParameterInterval(0.0, 1.0),
        hidden,
        parameter_tolerance=parameter_epsilon,
        occluder_key=lambda owner: (
            owner.face_id,
            owner.logical_surface_id,
        ),
        boundary_mode=VisibilityBoundaryMode.TOLERANCE_EXPANDED,
    )
    result: list[OpenFaceVisibilitySpan] = []
    for span in kernel_spans:
        face_ids = tuple(owner.face_id for owner in span.occluders)
        surface_ids = tuple(
            sorted({owner.logical_surface_id for owner in span.occluders})
        )
        result.append(
            OpenFaceVisibilitySpan(
                start=span.interval.start,
                end=span.interval.end,
                kind=span.kind.value,
                occluder_face_ids=face_ids,
                occluder_logical_surface_ids=surface_ids,
                face_level=len(face_ids),
                surface_level=len(surface_ids),
            )
        )
    return tuple(result)


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

    face_draw_order = _authoritative_face_draw_order(
        model,
        positions,
        validated.face_normals,
        view,
        policy,
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
