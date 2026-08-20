from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from ..contract import ContractError, TolerancePolicy, VisibilityModel
from ..parallel_solver import (
    ParallelView,
    compute_frame_visibility,
    segment_face_occlusion_interval,
)
from ..topology import ParameterInterval
from ..visibility import (
    OcclusionInterval,
    VisibilityBoundaryMode,
    partition_visibility,
)
from ..trace import (
    EdgeVisibility,
    FaceToleranceTrace,
    RawOcclusionInterval,
    SkippedFace,
    VisibilityFrame,
    VisibilitySpan,
)
from .contract import SectionPlane3D
from .trace import (
    ConvexSectionFrame,
    SectionBoundarySegment,
    SectionPoint,
    SegmentSolidIntersection,
    SolidBoundaryHit,
)


class ConvexSectionSolverError(ValueError):
    """Raised when a requested section cannot be solved unambiguously."""


@dataclass(frozen=True)
class _FacePlane:
    face_id: str
    point: np.ndarray
    outward_normal: np.ndarray
    boundary_epsilon: float


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ConvexSectionSolverError(
            f"{label} must be a finite three-component point"
        )
    return point


def _surface_vertex_ids(model: VisibilityModel) -> tuple[str, ...]:
    return tuple(sorted({item for face in model.faces for item in face.vertex_ids}))


def _validated_positions(
    model: VisibilityModel,
    vertex_positions: Mapping[str, Sequence[float]] | None,
    policy: TolerancePolicy,
) -> dict[str, np.ndarray]:
    raw = model.entry_positions if vertex_positions is None else vertex_positions
    try:
        model.validate(
            vertex_positions=raw,
            require_closed_convex_manifold=True,
            tolerance_policy=policy,
        )
    except ContractError as exc:
        raise ConvexSectionSolverError(
            f"invalid closed convex polyhedron: {exc}"
        ) from exc
    return {
        vertex_id: _point3(raw[vertex_id], f"vertex {vertex_id}")
        for vertex_id in sorted(raw)
    }


def fit_plane_patch_to_convex_polyhedron(
    model: VisibilityModel,
    plane: SectionPlane3D,
    *,
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    margin_ratio: float = 0.15,
    tolerance_policy: TolerancePolicy | None = None,
) -> SectionPlane3D:
    """Return a display patch large enough for the complete infinite-plane cut.

    ``SectionPlane3D.point`` and ``normal`` define the mathematical plane.  Its
    authored half-width and half-height are only minimum display dimensions in
    this helper.  The returned rectangle preserves the authored plane anchor
    and basis, but expands both in-plane axes far enough to cover the complete
    closed solid with a positive margin.  Free semantic-line endpoints are
    deliberately ignored so a long helper line cannot make the panel huge.
    """

    try:
        margin = float(margin_ratio)
    except (TypeError, ValueError) as exc:
        raise ConvexSectionSolverError(
            "plane patch margin_ratio must be a finite non-negative number"
        ) from exc
    if not isfinite(margin) or margin < 0:
        raise ConvexSectionSolverError(
            "plane patch margin_ratio must be a finite non-negative number"
        )
    policy = tolerance_policy or TolerancePolicy()
    positions = _validated_positions(model, vertex_positions, policy)
    surface_points = [
        positions[vertex_id] for vertex_id in _surface_vertex_ids(model)
    ]
    if not surface_points:
        raise ConvexSectionSolverError(
            "cannot fit a cutting-plane patch without solid surface vertices"
        )
    u_axis, v_axis, _normal = plane.basis
    anchor = np.asarray(plane.point, dtype=float)
    u_values = np.asarray(
        [float(np.dot(point - anchor, u_axis)) for point in surface_points],
        dtype=float,
    )
    v_values = np.asarray(
        [float(np.dot(point - anchor, v_axis)) for point in surface_points],
        dtype=float,
    )
    tolerance = policy.resolve(surface_points)

    def fitted_half_extent(values: np.ndarray, authored: float) -> float:
        span = float(np.max(values) - np.min(values))
        required = float(np.max(np.abs(values)))
        padding = max(
            0.5 * span * margin,
            tolerance.boundary * 4.0,
        )
        return max(float(authored), required + padding)

    return SectionPlane3D(
        plane.plane_id,
        plane.point,
        plane.normal,
        fitted_half_extent(u_values, plane.half_width),
        fitted_half_extent(v_values, plane.half_height),
        u_axis=plane.u_axis,
        occludes_strokes=plane.occludes_strokes,
    )


def _face_normal(points: Sequence[np.ndarray], epsilon: float, face_id: str) -> np.ndarray:
    origin = points[0]
    for index in range(1, len(points) - 1):
        normal = np.cross(points[index] - origin, points[index + 1] - origin)
        length = float(np.linalg.norm(normal))
        if length > epsilon * epsilon:
            return normal / length
    raise ConvexSectionSolverError(f"face {face_id} is degenerate")


def _outward_face_planes(
    model: VisibilityModel,
    positions: Mapping[str, np.ndarray],
    policy: TolerancePolicy,
) -> tuple[_FacePlane, ...]:
    surface_ids = _surface_vertex_ids(model)
    center = np.mean([positions[item] for item in surface_ids], axis=0)
    planes: list[_FacePlane] = []
    for face in model.faces:
        points = [positions[item] for item in face.vertex_ids]
        tolerance = policy.resolve(points)
        normal = _face_normal(points, tolerance.world, face.face_id)
        point = points[0]
        if float(np.dot(center - point, normal)) > 0:
            normal = -normal
        planes.append(
            _FacePlane(face.face_id, point, normal, tolerance.boundary)
        )
    return tuple(sorted(planes, key=lambda item: item.face_id))


def _inside(
    point: np.ndarray,
    planes: Sequence[_FacePlane],
) -> bool:
    return all(
        float(np.dot(point - face.point, face.outward_normal))
        <= face.boundary_epsilon
        for face in planes
    )


def _boundary_faces(
    point: np.ndarray,
    planes: Sequence[_FacePlane],
) -> tuple[str, ...]:
    return tuple(
        face.face_id
        for face in planes
        if abs(float(np.dot(point - face.point, face.outward_normal)))
        <= face.boundary_epsilon
    )


def intersect_segment_with_convex_polyhedron(
    model: VisibilityModel,
    start: Sequence[float],
    end: Sequence[float],
    *,
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
) -> SegmentSolidIntersection:
    """Clip one finite semantic segment against a closed convex polyhedron."""

    policy = tolerance_policy or TolerancePolicy()
    positions = _validated_positions(model, vertex_positions, policy)
    planes = _outward_face_planes(model, positions, policy)
    first = _point3(start, "segment start")
    last = _point3(end, "segment end")
    delta = last - first
    length = float(np.linalg.norm(delta))
    tolerance = policy.resolve((first, last), edge_length=length)
    if length <= tolerance.world:
        raise ConvexSectionSolverError("segment endpoints must be distinct")

    lower = 0.0
    upper = 1.0
    for face in planes:
        value = float(np.dot(first - face.point, face.outward_normal))
        slope = float(np.dot(delta, face.outward_normal))
        epsilon = max(face.boundary_epsilon, tolerance.boundary)
        if abs(slope) <= epsilon:
            if value > epsilon:
                return SegmentSolidIntersection("none", None, False, False, ())
            continue
        crossing = -value / slope
        if slope > 0:
            upper = min(upper, crossing)
        else:
            lower = max(lower, crossing)
        if lower > upper + tolerance.parameter:
            return SegmentSolidIntersection("none", None, False, False, ())

    lower = min(1.0, max(0.0, lower))
    upper = min(1.0, max(0.0, upper))
    if lower > upper + tolerance.parameter:
        return SegmentSolidIntersection("none", None, False, False, ())
    starts_inside = _inside(first, planes)
    ends_inside = _inside(last, planes)
    if upper - lower <= tolerance.parameter:
        parameter = 0.5 * (lower + upper)
        point = first + parameter * delta
        hit = SolidBoundaryHit(
            "touch",
            float(parameter),
            tuple(float(item) for item in point),
            _boundary_faces(point, planes),
        )
        return SegmentSolidIntersection(
            "point", (float(parameter), float(parameter)), starts_inside, ends_inside, (hit,)
        )

    hits: list[SolidBoundaryHit] = []
    entry_point = first + lower * delta
    entry_faces = _boundary_faces(entry_point, planes)
    if entry_faces:
        hits.append(
            SolidBoundaryHit(
                "entry",
                float(lower),
                tuple(float(item) for item in entry_point),
                entry_faces,
            )
        )
    exit_point = first + upper * delta
    exit_faces = _boundary_faces(exit_point, planes)
    if exit_faces and (
        not hits or abs(upper - hits[-1].parameter) > tolerance.parameter
    ):
        hits.append(
            SolidBoundaryHit(
                "exit",
                float(upper),
                tuple(float(item) for item in exit_point),
                exit_faces,
            )
        )
    return SegmentSolidIntersection(
        "segment",
        (float(lower), float(upper)),
        starts_inside,
        ends_inside,
        tuple(hits),
    )


def _surface_edges(model: VisibilityModel) -> tuple[tuple[str, str], ...]:
    result = {
        tuple(sorted((start, face.vertex_ids[(index + 1) % len(face.vertex_ids)])))
        for face in model.faces
        for index, start in enumerate(face.vertex_ids)
    }
    return tuple(sorted(result))


@dataclass
class _PointEvidence:
    position: np.ndarray
    source_edges: set[str]
    source_vertices: set[str]


def _edge_id(start: str, end: str) -> str:
    first, second = sorted((start, end))
    return f"{first}--{second}"


def _merge_candidate(
    candidates: list[_PointEvidence],
    point: np.ndarray,
    *,
    source_edge: str | None,
    source_vertex: str | None,
    epsilon: float,
) -> None:
    target = next(
        (
            item
            for item in candidates
            if float(np.linalg.norm(item.position - point)) <= epsilon
        ),
        None,
    )
    if target is None:
        target = _PointEvidence(point.copy(), set(), set())
        candidates.append(target)
    if source_edge is not None:
        target.source_edges.add(source_edge)
    if source_vertex is not None:
        target.source_vertices.add(source_vertex)


def _evidence_id(item: _PointEvidence) -> str:
    if item.source_vertices:
        return "vertex:" + "+".join(sorted(item.source_vertices))
    if item.source_edges:
        return "edge:" + "+".join(sorted(item.source_edges))
    raise ConvexSectionSolverError("section point has no source evidence")


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _convex_hull_indices(
    coordinates: Sequence[np.ndarray],
    identities: Sequence[str],
    epsilon: float,
) -> tuple[int, ...]:
    if len(coordinates) <= 2:
        return tuple(range(len(coordinates)))
    ordered = sorted(
        range(len(coordinates)),
        key=lambda index: (
            float(coordinates[index][0]),
            float(coordinates[index][1]),
            identities[index],
        ),
    )

    def build(indices: Sequence[int]) -> list[int]:
        hull: list[int] = []
        for index in indices:
            while len(hull) >= 2:
                first = coordinates[hull[-1]] - coordinates[hull[-2]]
                second = coordinates[index] - coordinates[hull[-1]]
                if _cross2(first, second) > epsilon:
                    break
                hull.pop()
            hull.append(index)
        return hull

    lower = build(ordered)
    upper = build(tuple(reversed(ordered)))
    hull = lower[:-1] + upper[:-1]
    if len(hull) == 2 and hull[0] == hull[1]:
        hull = hull[:1]
    return tuple(hull)


def intersect_plane_with_convex_polyhedron(
    section_id: str,
    model: VisibilityModel,
    plane: SectionPlane3D,
    *,
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
) -> ConvexSectionFrame:
    """Return the ordered convex cross-section cut by ``plane``.

    The result deliberately retains source-edge and source-vertex evidence so
    applications can expose stable teaching semantics without guessing which
    polyhedron feature produced an intersection point.
    """

    if not isinstance(section_id, str) or not section_id.strip():
        raise ConvexSectionSolverError("section_id must be a non-empty string")
    policy = tolerance_policy or TolerancePolicy()
    positions = _validated_positions(model, vertex_positions, policy)
    surface_ids = _surface_vertex_ids(model)
    surface_points = [positions[item] for item in surface_ids]
    tolerance = policy.resolve(surface_points)
    normal = np.asarray(plane.normal, dtype=float)
    plane_point = np.asarray(plane.point, dtype=float)
    distances = {
        vertex_id: float(np.dot(positions[vertex_id] - plane_point, normal))
        for vertex_id in surface_ids
    }
    candidates: list[_PointEvidence] = []
    for vertex_id in surface_ids:
        if abs(distances[vertex_id]) <= tolerance.boundary:
            _merge_candidate(
                candidates,
                positions[vertex_id],
                source_edge=None,
                source_vertex=vertex_id,
                epsilon=tolerance.boundary,
            )
    for start_id, end_id in _surface_edges(model):
        start_distance = distances[start_id]
        end_distance = distances[end_id]
        source_edge = _edge_id(start_id, end_id)
        if (
            abs(start_distance) <= tolerance.boundary
            or abs(end_distance) <= tolerance.boundary
        ):
            for vertex_id, distance in (
                (start_id, start_distance),
                (end_id, end_distance),
            ):
                if abs(distance) <= tolerance.boundary:
                    _merge_candidate(
                        candidates,
                        positions[vertex_id],
                        source_edge=source_edge,
                        source_vertex=vertex_id,
                        epsilon=tolerance.boundary,
                    )
            continue
        if start_distance * end_distance >= 0:
            continue
        parameter = start_distance / (start_distance - end_distance)
        point = positions[start_id] + parameter * (
            positions[end_id] - positions[start_id]
        )
        _merge_candidate(
            candidates,
            point,
            source_edge=source_edge,
            source_vertex=None,
            epsilon=tolerance.boundary,
        )

    if not candidates:
        return ConvexSectionFrame(section_id, plane, "empty", (), ())

    identities = [_evidence_id(item) for item in candidates]
    u_axis, v_axis, _normal = plane.basis
    plane_coordinates = [
        np.asarray(
            (
                float(np.dot(item.position - plane_point, u_axis)),
                float(np.dot(item.position - plane_point, v_axis)),
            )
        )
        for item in candidates
    ]
    hull_indices = _convex_hull_indices(
        plane_coordinates,
        identities,
        max(tolerance.boundary * tolerance.boundary, 1.0e-18),
    )
    ordered = [candidates[index] for index in hull_indices]
    points = tuple(
        SectionPoint(
            _evidence_id(item),
            tuple(float(value) for value in item.position),
            tuple(sorted(item.source_edges)),
            tuple(sorted(item.source_vertices)),
        )
        for item in ordered
    )
    if len(points) == 1:
        kind = "point"
        boundaries: tuple[SectionBoundarySegment, ...] = ()
    elif len(points) == 2:
        kind = "segment"
        boundaries = (
            SectionBoundarySegment(
                f"{section_id}:boundary:0",
                points[0].point_id,
                points[1].point_id,
            ),
        )
    else:
        kind = "polygon"
        boundaries = tuple(
            SectionBoundarySegment(
                f"{section_id}:boundary:{index}",
                point.point_id,
                points[(index + 1) % len(points)].point_id,
            )
            for index, point in enumerate(points)
        )
    return ConvexSectionFrame(section_id, plane, kind, points, boundaries)


def _visibility_spans(
    intervals: Sequence[RawOcclusionInterval],
    parameter_epsilon: float,
) -> tuple[VisibilitySpan, ...]:
    """Adapt shared visibility spans back to the frozen section v1 trace."""

    kernel_spans = partition_visibility(
        ParameterInterval(0.0, 1.0),
        tuple(
            OcclusionInterval(
                ParameterInterval(item.start, item.end),
                item.face_id,
            )
            for item in intervals
        ),
        parameter_tolerance=parameter_epsilon,
        occluder_key=lambda face_id: face_id,
        boundary_mode=VisibilityBoundaryMode.TOLERANCE_EXPANDED,
    )
    return tuple(
        VisibilitySpan(
            span.interval.start,
            span.interval.end,
            span.kind.value,
            tuple(span.occluders),
            len(span.occluders),
        )
        for span in kernel_spans
    )


def compute_sectioned_visibility(
    model: VisibilityModel,
    plane: SectionPlane3D,
    *,
    projection_matrix: Sequence[Sequence[float]],
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
) -> VisibilityFrame:
    """Solve solid strokes against both the solid and one finite panel.

    The closed-polyhedron contract remains authoritative.  The independent
    plane is evaluated as one additional occluding face, so a cutting plane
    never weakens or rewrites the solid topology.
    """

    policy = tolerance_policy or TolerancePolicy()
    positions = _validated_positions(model, vertex_positions, policy)
    base = compute_frame_visibility(
        model,
        projection_matrix=projection_matrix,
        vertex_positions=positions,
        tolerance_policy=policy,
        require_closed_convex_manifold=True,
    )
    patch = plane.patch_corners()
    patch_tolerance = policy.resolve(patch)
    plane_face_id = f"section-plane:{plane.plane_id}"
    view = ParallelView.from_matrix(base.projection_matrix)
    strokes = model.stroke_map
    edges: list[EdgeVisibility] = []
    for edge in base.edges:
        stroke = strokes[edge.source_edge_id]
        raw = list(edge.raw_intervals)
        skipped = list(edge.skipped_faces)
        if not plane.occludes_strokes:
            skipped.append(SkippedFace(plane_face_id, "occlusion_disabled"))
        elif stroke.visibility_mode == "always_visible":
            skipped.append(SkippedFace(plane_face_id, "stroke_always_visible"))
        elif stroke.visibility_mode != "always_hidden":
            start = positions[stroke.vertex_ids[0]]
            end = positions[stroke.vertex_ids[1]]
            interval = segment_face_occlusion_interval(
                start,
                end,
                patch,
                view,
                tolerance_policy=policy,
            )
            if interval is None:
                skipped.append(SkippedFace(plane_face_id, "no_occlusion"))
            else:
                raw.append(
                    RawOcclusionInterval(
                        plane_face_id,
                        float(interval[0]),
                        float(interval[1]),
                    )
                )
        raw.sort(key=lambda item: (item.start, item.end, item.face_id))
        skipped.sort(key=lambda item: (item.face_id, item.reason))
        face_tolerances = tuple(
            sorted(
                (
                    *edge.face_tolerances,
                    FaceToleranceTrace(
                        plane_face_id,
                        patch_tolerance.world,
                        patch_tolerance.boundary,
                        patch_tolerance.depth,
                        patch_tolerance.angular,
                    ),
                ),
                key=lambda item: item.face_id,
            )
        )
        edges.append(
            EdgeVisibility(
                edge.source_edge_id,
                tuple(raw),
                tuple(skipped),
                _visibility_spans(raw, edge.parameter_epsilon),
                edge.parameter_epsilon,
                face_tolerances,
            )
        )

    view_direction = np.asarray(base.view_direction, dtype=float)
    face_depths = []
    for face in model.faces:
        centroid = np.mean([positions[item] for item in face.vertex_ids], axis=0)
        face_depths.append((float(np.dot(centroid, view_direction)), face.face_id))
    plane_centroid = np.mean(np.asarray(patch, dtype=float), axis=0)
    face_depths.append(
        (float(np.dot(plane_centroid, view_direction)), plane_face_id)
    )
    return VisibilityFrame(
        base.visibility_group_id,
        base.projection_matrix,
        base.view_direction,
        base.tolerance,
        tuple(sorted(edges, key=lambda item: item.source_edge_id)),
        tuple(item[1] for item in sorted(face_depths)),
    )


__all__ = [
    "ConvexSectionSolverError",
    "compute_sectioned_visibility",
    "fit_plane_patch_to_convex_polyhedron",
    "intersect_plane_with_convex_polyhedron",
    "intersect_segment_with_convex_polyhedron",
]
