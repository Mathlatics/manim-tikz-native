"""Renderer-neutral unified painter graph for finite convex open faces.

The visibility solver owns hidden/visible classification.  This module adds
painter-event path fragmentation and deterministic face/path ordering without
importing Manim or assigning render slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..compositor import (
    CompositorCycleError,
    PainterConstraint,
    stable_topological_sort,
)
from ..contract import TolerancePolicy
from ..parallel_solver import ParallelView, SolverError as ParallelSolverError
from ..path_compositing import (
    line_convex_polygon_interval,
)
from .contract import OpenFaceContractError, OpenFaceVisibilityModel
from .solver import (
    OpenFaceSolverError,
    _face_depth_at_screen_point,
    _solve_face_painter,
    compute_open_face_visibility,
)
from .unified_contract import (
    OPEN_FACE_UNIFIED_COMPOSITING_LIMITS,
    OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA,
    OpenFacePaintFace,
    OpenFacePaintPolicy,
    OpenFacePaintRelation,
    OpenFaceUnifiedCompositingError,
    OpenFaceUnifiedCompositingFrame,
    OpenFaceUnifiedCompositingLimits,
    PaintPathFragment,
    canonical_open_face_unified_compositing_json,
)
from .unified_fragments import (
    _ProjectedPathPairEvent,
    _compute_projected_path_pair_events,
    compute_path_fragments,
)


_PAINTER_EVENT_EPSILON_FACTOR = 1024.0


@dataclass(frozen=True, slots=True)
class _FragmentGeometry:
    fragment: PaintPathFragment
    vertex_ids: tuple[str, str]
    incident_face_ids: frozenset[str]
    excluded_face_ids: frozenset[str]
    world: np.ndarray
    screen: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True, slots=True)
class _FragmentPairCandidate:
    """One source event localized to the fragments which touch it."""

    first_fragment_id: str
    second_fragment_id: str
    kind: str
    first_source_parameters: tuple[float, float]
    second_source_parameters: tuple[float, float]


def _relation(
    far_item_id: str,
    near_item_id: str,
    reason: str,
    minimum: float,
    maximum: float,
    overlap: float,
) -> OpenFacePaintRelation:
    return OpenFacePaintRelation(
        far_item_id=far_item_id,
        near_item_id=near_item_id,
        reason=reason,
        minimum_depth_difference=float(minimum),
        maximum_depth_difference=float(maximum),
        overlap_measure=float(overlap),
    )


def _guard_scale(
    model: OpenFaceVisibilityModel,
    limits: OpenFaceUnifiedCompositingLimits,
) -> None:
    face_count = len(model.faces)
    path_count = len(model.strokes)
    line_face_pairs = face_count * path_count
    line_line_pairs = path_count * (path_count - 1) // 2
    values = (
        ("faces", face_count, limits.max_faces),
        ("paths", path_count, limits.max_paths),
        ("line_face_pairs", line_face_pairs, limits.max_line_face_pairs),
        ("line_line_pairs", line_line_pairs, limits.max_line_line_pairs),
    )
    for label, value, maximum in values:
        if value > maximum:
            raise OpenFaceUnifiedCompositingError(
                f"unified open-face {label}={value} exceeds limit {maximum}"
            )


def _fragment_geometries(
    model: OpenFaceVisibilityModel,
    fragments: Sequence[PaintPathFragment],
    positions: Mapping[str, np.ndarray],
    view: ParallelView,
) -> dict[str, _FragmentGeometry]:
    direction = np.asarray(view.view_direction, dtype=float)
    matrix = np.asarray(view.projection_matrix, dtype=float)
    result: dict[str, _FragmentGeometry] = {}
    for fragment in fragments:
        stroke = model.stroke_map[fragment.source_path_id]
        source = np.asarray(
            (positions[stroke.vertex_ids[0]], positions[stroke.vertex_ids[1]]),
            dtype=float,
        )
        delta = source[1] - source[0]
        interval = fragment.parameter_interval
        world = np.asarray(
            (
                source[0] + interval.start * delta,
                source[0] + interval.end * delta,
            ),
            dtype=float,
        )
        screen = world @ matrix[:2].T
        result[fragment.fragment_id] = _FragmentGeometry(
            fragment=fragment,
            vertex_ids=stroke.vertex_ids,
            incident_face_ids=frozenset(stroke.incident_face_ids),
            excluded_face_ids=frozenset(stroke.excluded_occluder_face_ids),
            world=world,
            screen=screen,
            depth=world @ direction,
        )
    return result


def _face_relations(face_solution: object) -> list[OpenFacePaintRelation]:
    return [
        _relation(
            f"face:{item.far_face_id}",
            f"face:{item.near_face_id}",
            item.reason,
            item.minimum_depth_difference,
            item.maximum_depth_difference,
            item.overlap_measure,
        )
        for item in face_solution.relations
    ]


def _path_face_relations(
    model: OpenFaceVisibilityModel,
    geometries: Mapping[str, _FragmentGeometry],
    positions: Mapping[str, np.ndarray],
    normals: Mapping[str, np.ndarray],
    projected_faces: Mapping[str, tuple[np.ndarray, ...]],
    view: ParallelView,
    policy: TolerancePolicy,
    screen_epsilon: float,
    paint_policy: OpenFacePaintPolicy,
) -> list[OpenFacePaintRelation]:
    result: list[OpenFacePaintRelation] = []
    screen_measure_epsilon = screen_epsilon * _PAINTER_EVENT_EPSILON_FACTOR
    for fragment_id in sorted(geometries):
        geometry = geometries[fragment_id]
        projected_length = float(
            np.linalg.norm(geometry.screen[1] - geometry.screen[0])
        )
        if projected_length <= screen_epsilon:
            continue
        for face in sorted(model.faces, key=lambda item: item.face_id):
            polygon = projected_faces[face.face_id]
            interval = line_convex_polygon_interval(
                geometry.screen[0],
                geometry.screen[1],
                polygon,
                screen_epsilon,
            )
            if interval is None:
                continue
            overlap_measure = (interval[1] - interval[0]) * projected_length
            if overlap_measure <= screen_measure_epsilon:
                continue

            def difference(parameter: float) -> float:
                world = geometry.world[0] + parameter * (
                    geometry.world[1] - geometry.world[0]
                )
                screen = geometry.screen[0] + parameter * (
                    geometry.screen[1] - geometry.screen[0]
                )
                return float(np.dot(world, view.view_direction)) - (
                    _face_depth_at_screen_point(
                        screen,
                        face.vertex_ids,
                        positions,
                        normals[face.face_id],
                        view,
                    )
                )

            differences = (difference(interval[0]), difference(interval[1]))
            minimum = min(differences)
            maximum = max(differences)
            depth_epsilon = policy.resolve(
                (
                    geometry.world[0],
                    geometry.world[1],
                    *(positions[key] for key in face.vertex_ids),
                )
            ).depth
            if minimum < -depth_epsilon and maximum > depth_epsilon:
                raise OpenFaceUnifiedCompositingError(
                    "path/face depth changes inside one painter fragment: "
                    f"{fragment_id!r}, {face.face_id!r}"
                )

            face_item_id = f"face:{face.face_id}"
            is_declared_coplanar = (
                face.face_id in geometry.incident_face_ids
                or face.face_id in geometry.excluded_face_ids
            )
            is_occluder = (
                face.face_id in geometry.fragment.occluder_face_ids
            )

            if is_occluder:
                # The visibility solver has already proved that this face is in
                # front of the path.  Diagrammatic mode deliberately reverses
                # only the paint order so pedagogical hidden dashes remain
                # visible; physical mode retains the true depth order.
                if minimum > depth_epsilon:
                    raise OpenFaceUnifiedCompositingError(
                        "visibility and painter depth disagree for occluder "
                        f"{face.face_id!r} and path {fragment_id!r}"
                    )
                if paint_policy is OpenFacePaintPolicy.DIAGRAMMATIC:
                    far, near = face_item_id, fragment_id
                    reason = "diagrammatic_hidden_path"
                else:
                    far, near = fragment_id, face_item_id
                    reason = "physical_hidden_path"
            elif is_declared_coplanar:
                far, near = face_item_id, fragment_id
                reason = "declared_coplanar_path"
            elif maximum <= depth_epsilon and minimum < -depth_epsilon:
                if paint_policy is OpenFacePaintPolicy.DIAGRAMMATIC:
                    # Diagrammatic mode treats every semantic path as readable
                    # foreground ink.  Visibility still decides solid versus
                    # dashed style; physical mode alone lets a face cover a
                    # path which lies behind it.
                    far, near = face_item_id, fragment_id
                    reason = "diagrammatic_path_overlay"
                else:
                    far, near = fragment_id, face_item_id
                    reason = "path_face_depth"
            elif minimum >= -depth_epsilon and maximum > depth_epsilon:
                far, near = face_item_id, fragment_id
                reason = (
                    "diagrammatic_path_overlay"
                    if paint_policy is OpenFacePaintPolicy.DIAGRAMMATIC
                    else "path_face_depth"
                )
            else:
                raise OpenFaceUnifiedCompositingError(
                    "path and face overlap at indistinguishable depth without "
                    "a declared incidence or exclusion: "
                    f"{fragment_id!r}, {face.face_id!r}"
                )
            result.append(
                _relation(
                    far,
                    near,
                    reason,
                    minimum,
                    maximum,
                    overlap_measure,
                )
            )
    return result


def _fragment_pair_candidates(
    model: OpenFaceVisibilityModel,
    geometries: Mapping[str, _FragmentGeometry],
    positions: Mapping[str, np.ndarray],
    view: ParallelView,
    screen_epsilon: float,
    pair_events: Sequence[_ProjectedPathPairEvent],
    limits: OpenFaceUnifiedCompositingLimits,
) -> tuple[_FragmentPairCandidate, ...]:
    """Map each source-path event only to fragments which touch that event."""

    groups: dict[str, list[_FragmentGeometry]] = {}
    for geometry in geometries.values():
        groups.setdefault(geometry.fragment.source_path_id, []).append(geometry)
    for values in groups.values():
        values.sort(
            key=lambda item: (
                item.fragment.parameter_interval.start,
                item.fragment.parameter_interval.end,
                item.fragment.fragment_id,
            )
        )

    matrix = np.asarray(view.projection_matrix, dtype=float)
    projected_lengths: dict[str, float] = {}
    for stroke in model.strokes:
        world = np.asarray(
            (positions[stroke.vertex_ids[0]], positions[stroke.vertex_ids[1]]),
            dtype=float,
        )
        screen = world @ matrix[:2].T
        projected_lengths[stroke.source_edge_id] = float(
            np.linalg.norm(screen[1] - screen[0])
        )

    result: dict[tuple[str, str], _FragmentPairCandidate] = {}

    def add(candidate: _FragmentPairCandidate) -> None:
        key = (candidate.first_fragment_id, candidate.second_fragment_id)
        previous = result.get(key)
        if previous is not None:
            if previous != candidate:
                raise OpenFaceUnifiedCompositingError(
                    "one fragment pair received incompatible projected events: "
                    f"{key[0]!r}, {key[1]!r}"
                )
            return
        result[key] = candidate
        if len(result) > limits.max_fragment_pair_candidates:
            raise OpenFaceUnifiedCompositingError(
                "unified painter fragment_pair_candidates="
                f"{len(result)} exceeds limit "
                f"{limits.max_fragment_pair_candidates}"
            )

    for event in pair_events:
        first_group = groups.get(event.first_path_id, ())
        second_group = groups.get(event.second_path_id, ())
        if not first_group or not second_group:
            continue
        first_length = projected_lengths[event.first_path_id]
        second_length = projected_lengths[event.second_path_id]
        if event.kind == "point":
            first_parameter, second_parameter = event.parameters
            first_tolerance = screen_epsilon / max(first_length, screen_epsilon)
            second_tolerance = screen_epsilon / max(second_length, screen_epsilon)
            first_matches = [
                item
                for item in first_group
                if item.fragment.parameter_interval.start - first_tolerance
                <= first_parameter
                <= item.fragment.parameter_interval.end + first_tolerance
            ]
            second_matches = [
                item
                for item in second_group
                if item.fragment.parameter_interval.start - second_tolerance
                <= second_parameter
                <= item.fragment.parameter_interval.end + second_tolerance
            ]
            for first in first_matches:
                for second in second_matches:
                    add(
                        _FragmentPairCandidate(
                            first.fragment.fragment_id,
                            second.fragment.fragment_id,
                            "point",
                            (first_parameter, first_parameter),
                            (second_parameter, second_parameter),
                        )
                    )
            continue

        first_a, first_b, second_a, second_b = event.parameters
        correspondence_slope = (second_b - second_a) / (first_b - first_a)
        first_spans: list[tuple[float, float, _FragmentGeometry]] = []
        for item in first_group:
            interval = item.fragment.parameter_interval
            low = max(first_a, interval.start)
            high = min(first_b, interval.end)
            if high >= low:
                first_spans.append((low, high, item))

        second_spans: list[tuple[float, float, _FragmentGeometry]] = []
        for item in second_group:
            interval = item.fragment.parameter_interval
            mapped = (
                first_a + (interval.start - second_a) / correspondence_slope,
                first_a + (interval.end - second_a) / correspondence_slope,
            )
            low = max(first_a, min(mapped))
            high = min(first_b, max(mapped))
            if high >= low:
                second_spans.append((low, high, item))
        first_spans.sort(
            key=lambda value: (
                value[0],
                value[1],
                value[2].fragment.fragment_id,
            )
        )
        second_spans.sort(
            key=lambda value: (
                value[0],
                value[1],
                value[2].fragment.fragment_id,
            )
        )

        first_index = 0
        second_index = 0
        while first_index < len(first_spans) and second_index < len(second_spans):
            first_low, first_high, first = first_spans[first_index]
            second_low, second_high, second = second_spans[second_index]
            overlap_low = max(first_low, second_low)
            overlap_high = min(first_high, second_high)
            overlap_measure = (overlap_high - overlap_low) * first_length
            if overlap_measure > screen_epsilon * _PAINTER_EVENT_EPSILON_FACTOR:
                second_low_parameter = second_a + correspondence_slope * (
                    overlap_low - first_a
                )
                second_high_parameter = second_a + correspondence_slope * (
                    overlap_high - first_a
                )
                add(
                    _FragmentPairCandidate(
                        first.fragment.fragment_id,
                        second.fragment.fragment_id,
                        "overlap",
                        (overlap_low, overlap_high),
                        (second_low_parameter, second_high_parameter),
                    )
                )
            if first_high <= second_high:
                first_index += 1
            if second_high <= first_high:
                second_index += 1

    return tuple(result[key] for key in sorted(result))


def _local_fragment_parameter(
    geometry: _FragmentGeometry,
    source_parameter: float,
) -> float:
    interval = geometry.fragment.parameter_interval
    value = (source_parameter - interval.start) / (interval.end - interval.start)
    return min(1.0, max(0.0, float(value)))


def _path_path_relations(
    model: OpenFaceVisibilityModel,
    geometries: Mapping[str, _FragmentGeometry],
    positions: Mapping[str, np.ndarray],
    view: ParallelView,
    policy: TolerancePolicy,
    screen_epsilon: float,
    pair_events: Sequence[_ProjectedPathPairEvent],
    limits: OpenFaceUnifiedCompositingLimits,
) -> list[OpenFacePaintRelation]:
    result: list[OpenFacePaintRelation] = []
    candidates = _fragment_pair_candidates(
        model,
        geometries,
        positions,
        view,
        screen_epsilon,
        pair_events,
        limits,
    )
    for candidate in candidates:
        first = geometries[candidate.first_fragment_id]
        second = geometries[candidate.second_fragment_id]
        resolved = policy.resolve(
            (
                first.world[0],
                first.world[1],
                second.world[0],
                second.world[1],
            )
        )
        depth_epsilon = resolved.depth
        if candidate.kind == "point":
            first_t = _local_fragment_parameter(
                first,
                candidate.first_source_parameters[0],
            )
            second_t = _local_fragment_parameter(
                second,
                candidate.second_source_parameters[0],
            )
            first_world = first.world[0] + first_t * (
                first.world[1] - first.world[0]
            )
            second_world = second.world[0] + second_t * (
                second.world[1] - second.world[0]
            )
            difference = float(
                first.depth[0]
                + first_t * (first.depth[1] - first.depth[0])
                - second.depth[0]
                - second_t * (second.depth[1] - second.depth[0])
            )
            if abs(difference) <= depth_epsilon:
                if float(np.linalg.norm(first_world - second_world)) <= resolved.boundary:
                    continue
                raise OpenFaceUnifiedCompositingError(
                    "projected paths cross at indistinguishable depth: "
                    f"{first.fragment.fragment_id!r}, "
                    f"{second.fragment.fragment_id!r}"
                )
            far, near = (
                (first.fragment.fragment_id, second.fragment.fragment_id)
                if difference < 0.0
                else (second.fragment.fragment_id, first.fragment.fragment_id)
            )
            result.append(
                _relation(
                    far,
                    near,
                    "path_crossing_depth",
                    difference,
                    difference,
                    0.0,
                )
            )
            continue

        first_a = _local_fragment_parameter(
            first,
            candidate.first_source_parameters[0],
        )
        first_b = _local_fragment_parameter(
            first,
            candidate.first_source_parameters[1],
        )
        second_a = _local_fragment_parameter(
            second,
            candidate.second_source_parameters[0],
        )
        second_b = _local_fragment_parameter(
            second,
            candidate.second_source_parameters[1],
        )
        first_length = float(np.linalg.norm(first.screen[1] - first.screen[0]))
        overlap_measure = (first_b - first_a) * first_length
        if overlap_measure <= screen_epsilon * _PAINTER_EVENT_EPSILON_FACTOR:
            continue
        differences = []
        paired_world = []
        for first_t, second_t in (
            (first_a, second_a),
            (first_b, second_b),
        ):
            differences.append(
                float(
                    first.depth[0]
                    + first_t * (first.depth[1] - first.depth[0])
                    - second.depth[0]
                    - second_t * (second.depth[1] - second.depth[0])
                )
            )
            paired_world.append(
                (
                    first.world[0]
                    + first_t * (first.world[1] - first.world[0]),
                    second.world[0]
                    + second_t * (second.world[1] - second.world[0]),
                )
            )
        minimum = min(differences)
        maximum = max(differences)
        if minimum < -depth_epsilon and maximum > depth_epsilon:
            raise OpenFaceUnifiedCompositingError(
                "collinear path fragments exchange depth and require another split: "
                f"{first.fragment.fragment_id!r}, "
                f"{second.fragment.fragment_id!r}"
            )
        if maximum <= depth_epsilon and minimum < -depth_epsilon:
            far, near = first.fragment.fragment_id, second.fragment.fragment_id
        elif minimum >= -depth_epsilon and maximum > depth_epsilon:
            far, near = second.fragment.fragment_id, first.fragment.fragment_id
        elif max(
            float(np.linalg.norm(first_point - second_point))
            for first_point, second_point in paired_world
        ) <= resolved.boundary:
            continue
        else:
            raise OpenFaceUnifiedCompositingError(
                "collinear projected path fragments overlap at indistinguishable depth: "
                f"{first.fragment.fragment_id!r}, "
                f"{second.fragment.fragment_id!r}"
            )
        result.append(
            _relation(
                far,
                near,
                "collinear_path_depth",
                minimum,
                maximum,
                overlap_measure,
            )
        )
    return result


def _dedupe_relations(
    relations: Sequence[OpenFacePaintRelation],
    limits: OpenFaceUnifiedCompositingLimits,
) -> tuple[OpenFacePaintRelation, ...]:
    grouped: dict[tuple[str, str], list[OpenFacePaintRelation]] = {}
    for relation in relations:
        key = (relation.far_item_id, relation.near_item_id)
        reverse = (key[1], key[0])
        if reverse in grouped:
            raise OpenFaceUnifiedCompositingError(
                "paint items require contradictory local orders: "
                f"{key[0]!r}, {key[1]!r}"
            )
        grouped.setdefault(key, []).append(relation)
    if len(grouped) > limits.max_relations:
        raise OpenFaceUnifiedCompositingError(
            f"unified painter produced {len(grouped)} relations; "
            f"limit is {limits.max_relations}"
        )
    result: list[OpenFacePaintRelation] = []
    for key in sorted(grouped):
        values = grouped[key]
        result.append(
            _relation(
                key[0],
                key[1],
                "+".join(sorted({value.reason for value in values})),
                min(value.minimum_depth_difference for value in values),
                max(value.maximum_depth_difference for value in values),
                sum(value.overlap_measure for value in values),
            )
        )
    return tuple(result)


def _draw_order(
    item_ids: Sequence[str],
    relations: Sequence[OpenFacePaintRelation],
) -> tuple[str, ...]:
    identities = set(item_ids)
    constraints: list[PainterConstraint[str]] = []
    for relation in relations:
        if (
            relation.far_item_id not in identities
            or relation.near_item_id not in identities
        ):
            raise OpenFaceUnifiedCompositingError(
                "paint relation references an unknown item"
            )
        constraints.append(
            PainterConstraint(relation.far_item_id, relation.near_item_id)
        )
    try:
        return stable_topological_sort(
            tuple(sorted(identities)),
            constraints,
            key=lambda item_id: item_id,
        )
    except CompositorCycleError as exc:
        unresolved = ", ".join(sorted(str(item) for item in exc.unresolved))
        raise OpenFaceUnifiedCompositingError(
            "open-face face/path painter graph contains a cycle: " + unresolved
        ) from exc


def compute_open_face_unified_compositing(
    model: OpenFaceVisibilityModel,
    *,
    projection_matrix: Sequence[Sequence[float]],
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
    paint_policy: OpenFacePaintPolicy | str = OpenFacePaintPolicy.DIAGRAMMATIC,
    limits: OpenFaceUnifiedCompositingLimits = OPEN_FACE_UNIFIED_COMPOSITING_LIMITS,
) -> OpenFaceUnifiedCompositingFrame:
    """Return one deterministic face/path painter graph for an open-face frame."""

    if not isinstance(model, OpenFaceVisibilityModel):
        raise OpenFaceUnifiedCompositingError(
            "model must be an OpenFaceVisibilityModel"
        )
    if not isinstance(limits, OpenFaceUnifiedCompositingLimits):
        raise TypeError("limits must be an OpenFaceUnifiedCompositingLimits")
    selected_policy = OpenFacePaintPolicy.parse(paint_policy)
    policy = tolerance_policy or TolerancePolicy()
    _guard_scale(model, limits)
    try:
        validated = model._validated_frame(
            vertex_positions=vertex_positions,
            tolerance_policy=policy,
        )
        view = ParallelView.from_matrix(projection_matrix)
        visibility = compute_open_face_visibility(
            model,
            projection_matrix=projection_matrix,
            vertex_positions=validated.positions,
            tolerance_policy=policy,
        )
        face_solution = _solve_face_painter(
            model,
            validated.positions,
            validated.face_normals,
            view,
            policy,
        )
    except (OpenFaceContractError, OpenFaceSolverError, ParallelSolverError) as exc:
        raise OpenFaceUnifiedCompositingError(str(exc)) from exc

    faces = tuple(
        OpenFacePaintFace(
            item_id=f"face:{face.face_id}",
            face_id=face.face_id,
            logical_surface_id=face.logical_surface_id,
        )
        for face in sorted(model.faces, key=lambda item: item.face_id)
    )
    pair_events = _compute_projected_path_pair_events(
        model,
        validated.positions,
        view,
        face_solution.screen_epsilon,
    )
    fragments = compute_path_fragments(
        model,
        visibility,
        validated.positions,
        validated.face_normals,
        view,
        policy,
        face_solution.projected_faces,
        face_solution.screen_epsilon,
        limits,
        pair_events,
    )
    geometries = _fragment_geometries(
        model,
        fragments,
        validated.positions,
        view,
    )
    relations = _dedupe_relations(
        (
            *_face_relations(face_solution),
            *_path_face_relations(
                model,
                geometries,
                validated.positions,
                validated.face_normals,
                face_solution.projected_faces,
                view,
                policy,
                face_solution.screen_epsilon,
                selected_policy,
            ),
            *_path_path_relations(
                model,
                geometries,
                validated.positions,
                view,
                policy,
                face_solution.screen_epsilon,
                pair_events,
                limits,
            ),
        ),
        limits,
    )
    item_ids = tuple(
        (
            *(item.item_id for item in faces),
            *(item.fragment_id for item in fragments),
        )
    )
    return OpenFaceUnifiedCompositingFrame(
        visibility=visibility,
        paint_policy=selected_policy,
        faces=faces,
        path_fragments=fragments,
        order_relations=relations,
        draw_order=_draw_order(item_ids, relations),
    )


__all__ = [
    "OPEN_FACE_UNIFIED_COMPOSITING_LIMITS",
    "OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA",
    "OpenFacePaintFace",
    "OpenFacePaintPolicy",
    "OpenFacePaintRelation",
    "OpenFaceUnifiedCompositingError",
    "OpenFaceUnifiedCompositingFrame",
    "OpenFaceUnifiedCompositingLimits",
    "PaintPathFragment",
    "canonical_open_face_unified_compositing_json",
    "compute_open_face_unified_compositing",
]
