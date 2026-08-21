from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from ..compositor import (
    CompositorCycleError,
    PainterConstraint,
    stable_topological_sort,
)
from ..contract import TolerancePolicy
from ..parallel_solver import ParallelView
from ..sections.compositing import (
    _depth_coefficients,
    _projected_triangle,
    _signed_area2,
)
from .compositing import (
    DerivedDihedralTransparentCompositingFrame,
    compute_derived_dihedral_transparent_compositing,
)
from .contract import DerivedDihedralModel, RigidTransform3D


DERIVED_DIHEDRAL_UNIFIED_COMPOSITING_SCHEMA = (
    "manim-derived-dihedral-unified-compositing/v1"
)
# The geometric solver works far below raster precision.  Intersections shorter
# than roughly 1e-5 of the projected scene extent are numerical endpoint events,
# not paintable fragments; retaining them creates zero-pixel painter cycles at
# shared vertices.  This remains orders of magnitude below one Cairo pixel in
# the supported teaching renders while staying scale-aware.
_PAINTER_EVENT_EPSILON_FACTOR = 1024.0


class DerivedDihedralUnifiedCompositingError(ValueError):
    """Raised when faces and strokes cannot share one exact painter order."""


@dataclass(frozen=True)
class UnifiedFaceBatch:
    item_id: str
    source_face_id: str
    fragment_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "itemId": self.item_id,
            "sourceFaceId": self.source_face_id,
            "fragmentIds": list(self.fragment_ids),
        }


@dataclass(frozen=True)
class UnifiedStrokeFragment:
    item_id: str
    source_edge_id: str
    span_index: int
    slot_kind: str
    slot_index: int
    style_kind: str
    start_parameter: float
    end_parameter: float
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    vertex_ids: tuple[str, str]
    incident_face_ids: tuple[str, ...]
    occluder_face_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "itemId": self.item_id,
            "sourceEdgeId": self.source_edge_id,
            "spanIndex": self.span_index,
            "slotKind": self.slot_kind,
            "slotIndex": self.slot_index,
            "styleKind": self.style_kind,
            "startParameter": self.start_parameter,
            "endParameter": self.end_parameter,
            "start": list(self.start),
            "end": list(self.end),
            "vertexIds": list(self.vertex_ids),
            "incidentFaceIds": list(self.incident_face_ids),
            "occluderFaceIds": list(self.occluder_face_ids),
        }


@dataclass(frozen=True)
class UnifiedPaintRelation:
    far_item_id: str
    near_item_id: str
    reason: str
    minimum_depth_difference: float
    maximum_depth_difference: float
    overlap_measure: float

    def to_dict(self) -> dict[str, object]:
        return {
            "farItemId": self.far_item_id,
            "nearItemId": self.near_item_id,
            "reason": self.reason,
            "minimumDepthDifference": self.minimum_depth_difference,
            "maximumDepthDifference": self.maximum_depth_difference,
            "overlapMeasure": self.overlap_measure,
        }


@dataclass(frozen=True)
class DerivedDihedralUnifiedCompositingFrame:
    transparent: DerivedDihedralTransparentCompositingFrame
    face_batches: tuple[UnifiedFaceBatch, ...]
    stroke_fragments: tuple[UnifiedStrokeFragment, ...]
    draw_order: tuple[str, ...]
    order_relations: tuple[UnifiedPaintRelation, ...]
    schema: str = DERIVED_DIHEDRAL_UNIFIED_COMPOSITING_SCHEMA

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    *(item.item_id for item in self.face_batches),
                    *(item.item_id for item in self.stroke_fragments),
                )
            )
        )

    @property
    def face_batch_map(self) -> dict[str, UnifiedFaceBatch]:
        return {item.item_id: item for item in self.face_batches}

    @property
    def stroke_fragment_map(self) -> dict[str, UnifiedStrokeFragment]:
        return {item.item_id: item for item in self.stroke_fragments}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "transparent": self.transparent.to_dict(),
            "faceBatches": [item.to_dict() for item in self.face_batches],
            "strokeFragments": [item.to_dict() for item in self.stroke_fragments],
            "drawOrder": list(self.draw_order),
            "orderRelations": [item.to_dict() for item in self.order_relations],
        }


def _batch_id(fragment_ids: Sequence[str]) -> str:
    return "batch:" + hashlib.sha256(
        "|".join(fragment_ids).encode("utf-8")
    ).hexdigest()[:20]


def _world_positions(
    model: DerivedDihedralModel,
    transform: RigidTransform3D,
    solid_vertex_positions: Mapping[str, Sequence[float]] | None,
) -> dict[str, np.ndarray]:
    raw = (
        model.solid.entry_positions
        if solid_vertex_positions is None
        else solid_vertex_positions
    )
    solid = {
        vertex_id: np.asarray(raw[vertex_id], dtype=float)
        for vertex_id in model.solid.vertex_map
    }
    extracted = {
        vertex_id: transform.apply(model.solid.vertex_map[vertex_id].entry_position)
        for vertex_id in model.extracted_vertex_ids
    }
    return {
        **{
            model.solid_vertex_id(vertex_id): point
            for vertex_id, point in solid.items()
        },
        **{
            model.extracted_vertex_id(vertex_id): point
            for vertex_id, point in extracted.items()
        },
    }


def _face_batches(
    frame: DerivedDihedralTransparentCompositingFrame,
) -> tuple[UnifiedFaceBatch, ...]:
    result: list[UnifiedFaceBatch] = []
    for batch in frame.draw_batches:
        faces = {
            frame.fragment_map[fragment_id].source_face_id
            for fragment_id in batch
        }
        if len(faces) != 1 or None in faces:
            raise DerivedDihedralUnifiedCompositingError(
                "one transparent draw batch must contain exactly one source face"
            )
        result.append(
            UnifiedFaceBatch(
                _batch_id(batch),
                next(iter(faces)),  # type: ignore[arg-type]
                tuple(batch),
            )
        )
    return tuple(result)


def _stroke_fragments(
    model: DerivedDihedralModel,
    frame: DerivedDihedralTransparentCompositingFrame,
    positions: Mapping[str, np.ndarray],
    view: ParallelView,
    policy: TolerancePolicy,
) -> tuple[UnifiedStrokeFragment, ...]:
    overlay = model.overlay_model()
    suppressed = set(frame.visibility.suppressed_source_stroke_ids)
    matrix = np.asarray(view.projection_matrix, dtype=float)
    active_strokes = tuple(
        stroke
        for stroke in overlay.strokes
        if stroke.source_edge_id not in suppressed
    )
    all_world_points = [
        positions[vertex_id]
        for stroke in active_strokes
        for vertex_id in stroke.vertex_ids
    ] + [
        np.asarray(point, dtype=float)
        for triangle in frame.fragments
        for point in triangle.vertices
    ]
    resolved = policy.resolve(all_world_points)
    projected_points = [point @ matrix[:2].T for point in all_world_points]
    if projected_points:
        screen = np.vstack(projected_points)
        extent = np.max(screen, axis=0) - np.min(screen, axis=0)
        screen_scale = max(float(np.linalg.norm(extent)), policy.absolute_floor)
    else:
        screen_scale = 1.0
    screen_epsilon = max(
        policy.absolute_floor,
        policy.relative * screen_scale,
    ) * policy.boundary_factor
    depth_epsilon = resolved.depth * max(
        float(np.linalg.norm(matrix[2])), 1.0e-300
    )
    boundaries: dict[str, list[float]] = {}
    boundary_priorities: dict[str, dict[float, int]] = {}
    for stroke in active_strokes:
        edge_id = stroke.source_edge_id
        values = [
            float(value)
            for span in frame.visibility.line_visibility.edge_map[edge_id].spans
            for value in (span.start, span.end)
        ]
        boundaries[edge_id] = values
        boundary_priorities[edge_id] = {
            value: (3 if value in {0.0, 1.0} else 1) for value in values
        }

    def add_boundary(edge_id: str, value: float, priority: int) -> None:
        normalized = min(1.0, max(0.0, float(value)))
        boundaries[edge_id].append(normalized)
        if priority > 0:
            boundary_priorities[edge_id][normalized] = max(
                priority,
                boundary_priorities[edge_id].get(normalized, 0),
            )

    # A visibility interval only records where a finite face is in front of a
    # stroke.  A line can still pierce that face at a zero-width boundary and
    # exchange painter order there.  Freeze every such root before creating
    # render fragments so no fragment can be both in front of and behind one
    # face batch.
    for stroke in active_strokes:
        world = np.asarray(
            (
                positions[stroke.vertex_ids[0]],
                positions[stroke.vertex_ids[1]],
            ),
            dtype=float,
        )
        projected_line = world @ matrix.T
        for triangle in frame.fragments:
            triangle_screen, triangle_depth = _projected_triangle(
                triangle, matrix
            )
            interval = _line_triangle_interval(
                projected_line[0, :2],
                projected_line[1, :2],
                triangle_screen,
                screen_epsilon,
            )
            if interval is None:
                continue
            # A single long stroke can meet several spatially disjoint face
            # cells.  Even when its depth does not exchange inside either
            # cell, keeping both contacts in one painter item can close an
            # otherwise artificial A -> line -> B -> A cycle.  Localize the
            # stroke to every finite face-overlap boundary before building the
            # dependency graph.
            add_boundary(stroke.source_edge_id, interval[0], 0)
            add_boundary(stroke.source_edge_id, interval[1], 0)
            coefficients = _depth_coefficients(
                triangle_screen, triangle_depth
            )

            def difference(parameter: float) -> float:
                screen_point = projected_line[0, :2] + parameter * (
                    projected_line[1, :2] - projected_line[0, :2]
                )
                line_depth = projected_line[0, 2] + parameter * (
                    projected_line[1, 2] - projected_line[0, 2]
                )
                face_depth = float(
                    coefficients[0] * screen_point[0]
                    + coefficients[1] * screen_point[1]
                    + coefficients[2]
                )
                return float(line_depth - face_depth)

            first_difference = difference(interval[0])
            last_difference = difference(interval[1])
            slope = last_difference - first_difference
            if (
                first_difference < -depth_epsilon
                and last_difference > depth_epsilon
            ) or (
                first_difference > depth_epsilon
                and last_difference < -depth_epsilon
            ):
                root = interval[0] - first_difference * (
                    interval[1] - interval[0]
                ) / slope
                add_boundary(stroke.source_edge_id, root, 2)

    # Localize line/line crossings as well.  The later ordering graph may then
    # order only the adjacent pieces instead of moving an entire long edge in
    # front of (or behind) every other edge it meets.
    for first_index, first in enumerate(active_strokes):
        first_world = np.asarray(
            (
                positions[first.vertex_ids[0]],
                positions[first.vertex_ids[1]],
            ),
            dtype=float,
        )
        first_projected = first_world @ matrix.T
        for second in active_strokes[first_index + 1 :]:
            second_world = np.asarray(
                (
                    positions[second.vertex_ids[0]],
                    positions[second.vertex_ids[1]],
                ),
                dtype=float,
            )
            second_projected = second_world @ matrix.T
            intersection = _segment_intersection_parameters(
                first_projected[0, :2],
                first_projected[1, :2],
                second_projected[0, :2],
                second_projected[1, :2],
                screen_epsilon,
            )
            if intersection is None:
                continue
            kind, parameters = intersection
            if kind == "point":
                first_parameter, second_parameter = parameters  # type: ignore[misc]
                add_boundary(first.source_edge_id, first_parameter, 2)
                add_boundary(second.source_edge_id, second_parameter, 2)
                continue
            first_lower, first_upper, second_lower, second_upper = parameters  # type: ignore[misc]
            add_boundary(first.source_edge_id, first_lower, 2)
            add_boundary(first.source_edge_id, first_upper, 2)
            add_boundary(second.source_edge_id, second_lower, 2)
            add_boundary(second.source_edge_id, second_upper, 2)
            first_depths = (
                first_projected[0, 2]
                + first_lower
                * (first_projected[1, 2] - first_projected[0, 2]),
                first_projected[0, 2]
                + first_upper
                * (first_projected[1, 2] - first_projected[0, 2]),
            )
            second_depths = (
                second_projected[0, 2]
                + second_lower
                * (second_projected[1, 2] - second_projected[0, 2]),
                second_projected[0, 2]
                + second_upper
                * (second_projected[1, 2] - second_projected[0, 2]),
            )
            differences = (
                float(first_depths[0] - second_depths[0]),
                float(first_depths[1] - second_depths[1]),
            )
            if differences[0] * differences[1] < -(depth_epsilon * depth_epsilon):
                ratio = -differences[0] / (differences[1] - differences[0])
                first_root = first_lower + ratio * (first_upper - first_lower)
                second_root = second_lower + ratio * (
                    second_upper - second_lower
                )
                add_boundary(first.source_edge_id, first_root, 2)
                add_boundary(second.source_edge_id, second_root, 2)

    result: list[UnifiedStrokeFragment] = []
    for stroke in active_strokes:
        edge = frame.visibility.line_visibility.edge_map[stroke.source_edge_id]
        world_start = positions[stroke.vertex_ids[0]]
        world_delta = positions[stroke.vertex_ids[1]] - world_start
        slot_counts = {"visible": 0, "hidden": 0}
        values = sorted(boundaries[stroke.source_edge_id])
        projected_endpoints = np.asarray(
            (
                positions[stroke.vertex_ids[0]],
                positions[stroke.vertex_ids[1]],
            ),
            dtype=float,
        ) @ matrix[:2].T
        projected_length = float(
            np.linalg.norm(projected_endpoints[1] - projected_endpoints[0])
        )
        parameter_tolerance = max(
            edge.parameter_epsilon,
            _PAINTER_EVENT_EPSILON_FACTOR
            * screen_epsilon
            / max(projected_length, screen_epsilon),
        )
        clusters: list[list[float]] = []
        for value in values:
            if not clusters or abs(value - clusters[-1][-1]) > parameter_tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        priorities = boundary_priorities[stroke.source_edge_id]
        unique: list[float] = []
        for cluster in clusters:
            highest = max(priorities.get(value, 0) for value in cluster)
            preferred = [
                value
                for value in cluster
                if priorities.get(value, 0) == highest
            ]
            # Exact endpoints and algebraic depth/crossing roots take
            # precedence over tolerance-expanded face-overlap boundaries.
            # Averaging the latter into a root can move a fragment slightly
            # across the face and create a false depth exchange.
            unique.append(float(np.median(preferred)))
        if unique:
            unique[0] = 0.0
            unique[-1] = 1.0
        for span_index, (start_parameter, end_parameter) in enumerate(
            zip(unique, unique[1:])
        ):
            if end_parameter - start_parameter <= parameter_tolerance:
                continue
            midpoint = 0.5 * (start_parameter + end_parameter)
            matches = [
                span
                for span in edge.spans
                if span.start - edge.parameter_epsilon
                <= midpoint
                <= span.end + edge.parameter_epsilon
            ]
            if len(matches) != 1:
                raise DerivedDihedralUnifiedCompositingError(
                    f"stroke {stroke.source_edge_id!r} lost its visibility span"
                )
            span = matches[0]
            if span.kind not in slot_counts:
                raise DerivedDihedralUnifiedCompositingError(
                    f"stroke {stroke.source_edge_id!r} has unsupported span kind {span.kind!r}"
                )
            slot_index = slot_counts[span.kind]
            slot_counts[span.kind] += 1
            start = world_start + start_parameter * world_delta
            end = world_start + end_parameter * world_delta
            item_id = (
                f"stroke:{stroke.source_edge_id}:{span.kind}:{slot_index}"
            )
            result.append(
                UnifiedStrokeFragment(
                    item_id,
                    stroke.source_edge_id,
                    span_index,
                    span.kind,
                    slot_index,
                    span.kind,
                    float(start_parameter),
                    float(end_parameter),
                    tuple(float(item) for item in start),
                    tuple(float(item) for item in end),
                    tuple(stroke.vertex_ids),
                    tuple(stroke.incident_face_ids),
                    tuple(span.occluder_face_ids),
                )
            )
    return tuple(sorted(result, key=lambda item: item.item_id))


def _line_triangle_interval(
    line_start: np.ndarray,
    line_end: np.ndarray,
    triangle: np.ndarray,
    epsilon: float,
) -> tuple[float, float] | None:
    direction = line_end - line_start
    lower = 0.0
    upper = 1.0
    area = _signed_area2(triangle)
    if abs(area) <= epsilon * epsilon:
        return None
    orientation = 1.0 if area > 0 else -1.0
    for index, edge_start in enumerate(triangle):
        edge_end = triangle[(index + 1) % len(triangle)]
        edge = edge_end - edge_start
        value_zero = orientation * float(
            edge[0] * (line_start[1] - edge_start[1])
            - edge[1] * (line_start[0] - edge_start[0])
        )
        value_slope = orientation * float(
            edge[0] * direction[1] - edge[1] * direction[0]
        )
        threshold = -epsilon * max(float(np.linalg.norm(edge)), epsilon)
        if abs(value_slope) <= epsilon * max(float(np.linalg.norm(direction)), 1.0):
            if value_zero < threshold:
                return None
            continue
        crossing = (threshold - value_zero) / value_slope
        if value_slope > 0:
            lower = max(lower, crossing)
        else:
            upper = min(upper, crossing)
        if upper < lower - epsilon:
            return None
    lower = min(1.0, max(0.0, lower))
    upper = min(1.0, max(0.0, upper))
    if upper - lower <= epsilon:
        return None
    return lower, upper


def _relation(
    far_item_id: str,
    near_item_id: str,
    reason: str,
    minimum: float,
    maximum: float,
    measure: float,
) -> UnifiedPaintRelation:
    return UnifiedPaintRelation(
        far_item_id,
        near_item_id,
        reason,
        float(minimum),
        float(maximum),
        float(measure),
    )


def _line_face_relations(
    stroke_fragments: Sequence[UnifiedStrokeFragment],
    face_batches: Sequence[UnifiedFaceBatch],
    transparent: DerivedDihedralTransparentCompositingFrame,
    view: ParallelView,
    policy: TolerancePolicy,
) -> list[UnifiedPaintRelation]:
    matrix = np.asarray(view.projection_matrix, dtype=float)
    fragment_map = transparent.fragment_map
    all_points = [
        point
        for face in transparent.fragments
        for point in face.vertices
    ] + [
        point
        for stroke in stroke_fragments
        for point in (stroke.start, stroke.end)
    ]
    resolved = policy.resolve(all_points)
    projected_points = [
        np.asarray(point, dtype=float) @ matrix[:2].T
        for point in all_points
    ]
    if projected_points:
        values = np.vstack(projected_points)
        extent = np.max(values, axis=0) - np.min(values, axis=0)
        screen_scale = max(float(np.linalg.norm(extent)), policy.absolute_floor)
    else:
        screen_scale = 1.0
    screen_epsilon = max(
        policy.absolute_floor,
        policy.relative * screen_scale,
    ) * policy.boundary_factor
    depth_epsilon = resolved.depth * max(
        float(np.linalg.norm(matrix[2])), 1.0e-300
    )
    result: list[UnifiedPaintRelation] = []
    for stroke in stroke_fragments:
        line_world = np.asarray((stroke.start, stroke.end), dtype=float)
        projected_line = line_world @ matrix.T
        line_screen = projected_line[:, :2]
        line_depth = projected_line[:, 2]
        if float(np.linalg.norm(line_screen[1] - line_screen[0])) <= screen_epsilon:
            raise DerivedDihedralUnifiedCompositingError(
                f"stroke fragment {stroke.item_id!r} projects to one point"
            )
        for batch in face_batches:
            directions: set[tuple[str, str]] = set()
            minimums: list[float] = []
            maximums: list[float] = []
            overlap = 0.0
            reasons: set[str] = set()
            for fragment_id in batch.fragment_ids:
                triangle = fragment_map[fragment_id]
                triangle_screen, triangle_depth = _projected_triangle(
                    triangle, matrix
                )
                interval = _line_triangle_interval(
                    line_screen[0],
                    line_screen[1],
                    triangle_screen,
                    screen_epsilon,
                )
                if interval is None:
                    continue
                measure = (interval[1] - interval[0]) * float(
                    np.linalg.norm(line_screen[1] - line_screen[0])
                )
                # A non-incident stroke may merely touch a face at one
                # endpoint.  The half-space tolerance turns that zero-width
                # contact into a microscopic interval; it has no raster area
                # and must not invent an arbitrary painter relation.
                if measure <= screen_epsilon * _PAINTER_EVENT_EPSILON_FACTOR:
                    continue
                coefficients = _depth_coefficients(
                    triangle_screen, triangle_depth
                )
                differences: list[float] = []
                for parameter in interval:
                    screen = line_screen[0] + parameter * (
                        line_screen[1] - line_screen[0]
                    )
                    depth = line_depth[0] + parameter * (
                        line_depth[1] - line_depth[0]
                    )
                    face_depth = float(
                        coefficients[0] * screen[0]
                        + coefficients[1] * screen[1]
                        + coefficients[2]
                    )
                    differences.append(float(depth - face_depth))
                minimum = min(differences)
                maximum = max(differences)
                if minimum < -depth_epsilon and maximum <= depth_epsilon:
                    directions.add((stroke.item_id, batch.item_id))
                    reason = "stroke_behind_face"
                elif maximum > depth_epsilon and minimum >= -depth_epsilon:
                    directions.add((batch.item_id, stroke.item_id))
                    reason = "stroke_in_front_of_face"
                elif batch.source_face_id in stroke.incident_face_ids:
                    directions.add((batch.item_id, stroke.item_id))
                    reason = "incident_boundary_over_face"
                elif minimum < -depth_epsilon and maximum > depth_epsilon:
                    raise DerivedDihedralUnifiedCompositingError(
                        "one stroke fragment exchanges depth with one face batch; "
                        f"split {stroke.item_id!r} again at {batch.source_face_id!r}"
                    )
                else:
                    # A semantic construction line can lie in a face without
                    # being one of that face's topological boundary edges.
                    # In technical diagrams the line is the foreground ink on
                    # that surface, so equal-depth linework deterministically
                    # paints over the fill.
                    directions.add((batch.item_id, stroke.item_id))
                    reason = "coplanar_stroke_over_face"
                minimums.append(minimum)
                maximums.append(maximum)
                overlap += measure
                reasons.add(reason)
            if not directions:
                continue
            if len(directions) != 1:
                raise DerivedDihedralUnifiedCompositingError(
                    "one face batch lies on both sides of one stroke fragment: "
                    f"{stroke.item_id!r}, {batch.source_face_id!r}"
                )
            far, near = next(iter(directions))
            result.append(
                _relation(
                    far,
                    near,
                    "+".join(sorted(reasons)),
                    min(minimums),
                    max(maximums),
                    overlap,
                )
            )
    return result


def _segment_intersection_parameters(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    epsilon: float,
) -> tuple[str, tuple[float, float] | tuple[float, float, float, float]] | None:
    first = first_end - first_start
    second = second_end - second_start
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    if first_length <= epsilon or second_length <= epsilon:
        return None
    cross = float(first[0] * second[1] - first[1] * second[0])
    offset = second_start - first_start
    cross_tolerance = epsilon * max(first_length, second_length, epsilon)
    if abs(cross) > cross_tolerance:
        first_parameter = float(
            (offset[0] * second[1] - offset[1] * second[0]) / cross
        )
        second_parameter = float(
            (offset[0] * first[1] - offset[1] * first[0]) / cross
        )
        first_parameter_tolerance = epsilon / first_length
        second_parameter_tolerance = epsilon / second_length
        if (
            -first_parameter_tolerance
            <= first_parameter
            <= 1.0 + first_parameter_tolerance
            and -second_parameter_tolerance
            <= second_parameter
            <= 1.0 + second_parameter_tolerance
        ):
            return (
                "point",
                (
                    min(1.0, max(0.0, first_parameter)),
                    min(1.0, max(0.0, second_parameter)),
                ),
            )
        return None
    perpendicular_distance = abs(
        float(offset[0] * first[1] - offset[1] * first[0])
    ) / first_length
    if perpendicular_distance > epsilon:
        return None
    direction = first / first_length
    second_scalar_start = float(np.dot(second_start - first_start, direction))
    second_scalar_end = float(np.dot(second_end - first_start, direction))
    overlap_start = max(0.0, min(second_scalar_start, second_scalar_end))
    overlap_end = min(
        first_length,
        max(second_scalar_start, second_scalar_end),
    )
    if overlap_end - overlap_start <= epsilon:
        return None
    second_scalar_delta = second_scalar_end - second_scalar_start
    if abs(second_scalar_delta) <= epsilon:
        return None
    first_lower = overlap_start / first_length
    first_upper = overlap_end / first_length
    second_at_start = (overlap_start - second_scalar_start) / second_scalar_delta
    second_at_end = (overlap_end - second_scalar_start) / second_scalar_delta
    second_lower = min(1.0, max(0.0, min(second_at_start, second_at_end)))
    second_upper = min(1.0, max(0.0, max(second_at_start, second_at_end)))
    if second_upper - second_lower <= epsilon / second_length:
        return None
    return (
        "overlap",
        (
            first_lower,
            first_upper,
            min(second_lower, second_upper),
            max(second_lower, second_upper),
        ),
    )


def _line_line_relations(
    fragments: Sequence[UnifiedStrokeFragment],
    view: ParallelView,
    policy: TolerancePolicy,
) -> list[UnifiedPaintRelation]:
    matrix = np.asarray(view.projection_matrix, dtype=float)
    points = [
        np.asarray(point, dtype=float)
        for fragment in fragments
        for point in (fragment.start, fragment.end)
    ]
    resolved = policy.resolve(points)
    projected = [point @ matrix.T for point in points]
    if projected:
        screen = np.asarray([item[:2] for item in projected])
        extent = np.max(screen, axis=0) - np.min(screen, axis=0)
        scale = max(float(np.linalg.norm(extent)), policy.absolute_floor)
    else:
        scale = 1.0
    screen_epsilon = max(policy.absolute_floor, policy.relative * scale) * (
        policy.boundary_factor
    )
    depth_epsilon = resolved.depth * max(
        float(np.linalg.norm(matrix[2])), 1.0e-300
    )
    result: list[UnifiedPaintRelation] = []
    for first_index, first in enumerate(fragments):
        first_world = np.asarray((first.start, first.end), dtype=float)
        first_projected = first_world @ matrix.T
        for second in fragments[first_index + 1 :]:
            if first.source_edge_id == second.source_edge_id:
                continue
            second_world = np.asarray((second.start, second.end), dtype=float)
            second_projected = second_world @ matrix.T
            intersection = _segment_intersection_parameters(
                first_projected[0, :2],
                first_projected[1, :2],
                second_projected[0, :2],
                second_projected[1, :2],
                screen_epsilon,
            )
            if intersection is None:
                continue
            kind, parameters = intersection
            if kind == "point":
                first_parameter, second_parameter = parameters  # type: ignore[misc]
                if (
                    set(first.vertex_ids) & set(second.vertex_ids)
                    and min(first_parameter, 1.0 - first_parameter)
                    <= screen_epsilon
                    and min(second_parameter, 1.0 - second_parameter)
                    <= screen_epsilon
                ):
                    continue
                first_depth = first_projected[0, 2] + first_parameter * (
                    first_projected[1, 2] - first_projected[0, 2]
                )
                second_depth = second_projected[0, 2] + second_parameter * (
                    second_projected[1, 2] - second_projected[0, 2]
                )
                first_point = first_world[0] + first_parameter * (
                    first_world[1] - first_world[0]
                )
                second_point = second_world[0] + second_parameter * (
                    second_world[1] - second_world[0]
                )
                difference = float(first_depth - second_depth)
                if abs(difference) <= depth_epsilon:
                    if float(np.linalg.norm(first_point - second_point)) <= resolved.boundary:
                        continue
                    raise DerivedDihedralUnifiedCompositingError(
                        "two projected strokes overlap at indistinguishable depth: "
                        f"{first.item_id!r}, {second.item_id!r}"
                    )
                far, near = (
                    (first.item_id, second.item_id)
                    if difference < 0
                    else (second.item_id, first.item_id)
                )
                result.append(
                    _relation(
                        far,
                        near,
                        "stroke_crossing_depth",
                        difference,
                        difference,
                        0.0,
                    )
                )
                continue
            first_lower, first_upper, second_lower, second_upper = parameters  # type: ignore[misc]
            overlap_measure = (first_upper - first_lower) * float(
                np.linalg.norm(
                    first_projected[1, :2] - first_projected[0, :2]
                )
            )
            if overlap_measure <= screen_epsilon * _PAINTER_EVENT_EPSILON_FACTOR:
                continue
            differences = []
            for first_parameter, second_parameter in (
                (first_lower, second_lower),
                (first_upper, second_upper),
            ):
                first_depth = first_projected[0, 2] + first_parameter * (
                    first_projected[1, 2] - first_projected[0, 2]
                )
                second_depth = second_projected[0, 2] + second_parameter * (
                    second_projected[1, 2] - second_projected[0, 2]
                )
                differences.append(float(first_depth - second_depth))
            minimum = min(differences)
            maximum = max(differences)
            if minimum < -depth_epsilon and maximum > depth_epsilon:
                raise DerivedDihedralUnifiedCompositingError(
                    "two collinear stroke fragments exchange depth; split them again: "
                    f"{first.item_id!r}, {second.item_id!r}"
                )
            if maximum < -depth_epsilon:
                far, near = first.item_id, second.item_id
            elif minimum > depth_epsilon:
                far, near = second.item_id, first.item_id
            else:
                first_overlap = (
                    first_world[0]
                    + first_lower * (first_world[1] - first_world[0]),
                    first_world[0]
                    + first_upper * (first_world[1] - first_world[0]),
                )
                second_overlap = (
                    second_world[0]
                    + second_lower * (second_world[1] - second_world[0]),
                    second_world[0]
                    + second_upper * (second_world[1] - second_world[0]),
                )
                if max(
                    float(np.linalg.norm(first_overlap[index] - second_overlap[index]))
                    for index in range(2)
                ) <= resolved.boundary:
                    continue
                raise DerivedDihedralUnifiedCompositingError(
                    "collinear projected strokes overlap at indistinguishable depth: "
                    f"{first.item_id!r}, {second.item_id!r}"
                )
            result.append(
                _relation(
                    far,
                    near,
                    "collinear_stroke_depth",
                    minimum,
                    maximum,
                    overlap_measure,
                )
            )
    return result


def _face_relations(
    transparent: DerivedDihedralTransparentCompositingFrame,
    face_batches: Sequence[UnifiedFaceBatch],
) -> list[UnifiedPaintRelation]:
    fragment_to_batch = {
        fragment_id: batch.item_id
        for batch in face_batches
        for fragment_id in batch.fragment_ids
    }
    result: list[UnifiedPaintRelation] = []
    for item in transparent.order_relations:
        far = fragment_to_batch[item.far_fragment_id]
        near = fragment_to_batch[item.near_fragment_id]
        if far == near:
            continue
        result.append(
            _relation(
                far,
                near,
                "face_" + item.reason,
                item.minimum_depth_difference,
                item.maximum_depth_difference,
                item.overlap_area,
            )
        )
    return result


def _dedupe_relations(
    relations: Sequence[UnifiedPaintRelation],
) -> tuple[UnifiedPaintRelation, ...]:
    directions: dict[tuple[str, str], list[UnifiedPaintRelation]] = {}
    for item in relations:
        if item.far_item_id == item.near_item_id:
            continue
        reverse = (item.near_item_id, item.far_item_id)
        if reverse in directions:
            raise DerivedDihedralUnifiedCompositingError(
                "paint items require contradictory local depth orders: "
                f"{item.far_item_id!r}, {item.near_item_id!r}"
            )
        directions.setdefault((item.far_item_id, item.near_item_id), []).append(item)
    result: list[UnifiedPaintRelation] = []
    for key in sorted(directions):
        items = directions[key]
        result.append(
            _relation(
                key[0],
                key[1],
                "+".join(sorted({item.reason for item in items})),
                min(item.minimum_depth_difference for item in items),
                max(item.maximum_depth_difference for item in items),
                sum(item.overlap_measure for item in items),
            )
        )
    return tuple(result)


def _draw_order(
    item_ids: Sequence[str],
    relations: Sequence[UnifiedPaintRelation],
) -> tuple[str, ...]:
    """Adapt derived-dihedral relations to the shared stable compositor.

    The domain layer remains fail-closed: relation endpoints must already be
    registered paint items and self-relations retain the historical cycle
    error instead of being ignored by the generic compositor.  Lexicographic
    item identity remains the v1 tie-break for otherwise unrelated nodes.
    """

    identities = set(item_ids)
    constraints: list[PainterConstraint[str]] = []
    self_relations: set[str] = set()
    for relation in relations:
        if (
            relation.far_item_id not in identities
            or relation.near_item_id not in identities
        ):
            raise DerivedDihedralUnifiedCompositingError(
                "unified paint relation references an unknown item"
            )
        if relation.far_item_id == relation.near_item_id:
            self_relations.add(relation.far_item_id)
            continue
        constraints.append(
            PainterConstraint(
                relation.far_item_id,
                relation.near_item_id,
            )
        )

    if self_relations:
        raise DerivedDihedralUnifiedCompositingError(
            "unified face/stroke painter order contains a cycle: "
            + ", ".join(sorted(self_relations))
        )

    try:
        return stable_topological_sort(
            tuple(sorted(identities)),
            constraints,
            key=lambda item_id: item_id,
        )
    except CompositorCycleError as exc:
        cyclic = sorted(str(item_id) for item_id in exc.unresolved)
        raise DerivedDihedralUnifiedCompositingError(
            "unified face/stroke painter order contains a cycle: "
            + ", ".join(cyclic)
        ) from exc


def compute_derived_dihedral_unified_compositing(
    model: DerivedDihedralModel,
    *,
    transform: RigidTransform3D,
    projection_matrix: Sequence[Sequence[float]],
    solid_vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
) -> DerivedDihedralUnifiedCompositingFrame:
    """Return one far-to-near order for transparent faces and line spans.

    The visibility solver still owns solid/dashed classification.  This stage
    only makes the already-correct result paint correctly in Cairo: every
    visible or hidden span is interleaved with the exact transparent face
    batches and with every other crossing stroke by projected depth.
    """

    policy = tolerance_policy or TolerancePolicy()
    transparent = compute_derived_dihedral_transparent_compositing(
        model,
        transform=transform,
        projection_matrix=projection_matrix,
        solid_vertex_positions=solid_vertex_positions,
        tolerance_policy=policy,
    )
    positions = _world_positions(model, transform, solid_vertex_positions)
    view = ParallelView.from_matrix(projection_matrix)
    face_batches = _face_batches(transparent)
    stroke_fragments = _stroke_fragments(
        model,
        transparent,
        positions,
        view,
        policy,
    )
    relations = _dedupe_relations(
        (
            *_face_relations(transparent, face_batches),
            *_line_face_relations(
                stroke_fragments,
                face_batches,
                transparent,
                view,
                policy,
            ),
            *_line_line_relations(stroke_fragments, view, policy),
        )
    )
    item_ids = tuple(
        (
            *(item.item_id for item in face_batches),
            *(item.item_id for item in stroke_fragments),
        )
    )
    return DerivedDihedralUnifiedCompositingFrame(
        transparent,
        face_batches,
        stroke_fragments,
        _draw_order(item_ids, relations),
        relations,
    )


def canonical_derived_dihedral_unified_compositing_json(
    frame: DerivedDihedralUnifiedCompositingFrame,
) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "DERIVED_DIHEDRAL_UNIFIED_COMPOSITING_SCHEMA",
    "DerivedDihedralUnifiedCompositingError",
    "DerivedDihedralUnifiedCompositingFrame",
    "UnifiedFaceBatch",
    "UnifiedPaintRelation",
    "UnifiedStrokeFragment",
    "canonical_derived_dihedral_unified_compositing_json",
    "compute_derived_dihedral_unified_compositing",
]
