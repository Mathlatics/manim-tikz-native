"""Path-fragment extraction for open-face unified compositing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..contract import TolerancePolicy
from ..parallel_solver import ParallelView
from ..path_compositing import (
    line_convex_polygon_interval,
    segment_intersection_parameters,
)
from ..topology import ParameterInterval
from ..visibility import VisibilityKind
from .contract import OpenFaceVisibilityModel
from .solver import _face_depth_at_screen_point
from .trace import OpenFaceVisibilityFrame
from .unified_contract import (
    OpenFaceUnifiedCompositingError,
    OpenFaceUnifiedCompositingLimits,
    PaintPathFragment,
)


_PAINTER_EVENT_EPSILON_FACTOR = 1024.0


@dataclass(frozen=True, slots=True)
class _ProjectedPathPairEvent:
    """One projected event expressed in complete source-path parameters."""

    first_path_id: str
    second_path_id: str
    kind: str
    parameters: tuple[float, ...]


def _compute_projected_path_pair_events(
    model: OpenFaceVisibilityModel,
    positions: Mapping[str, np.ndarray],
    view: ParallelView,
    screen_epsilon: float,
) -> tuple[_ProjectedPathPairEvent, ...]:
    """Intersect every source-path pair once in stable identity order."""

    matrix = np.asarray(view.projection_matrix, dtype=float)
    strokes = tuple(sorted(model.strokes, key=lambda item: item.source_edge_id))
    screens: dict[str, np.ndarray] = {}
    for stroke in strokes:
        world = np.asarray(
            (positions[stroke.vertex_ids[0]], positions[stroke.vertex_ids[1]]),
            dtype=float,
        )
        screens[stroke.source_edge_id] = world @ matrix[:2].T

    result: list[_ProjectedPathPairEvent] = []
    for first_index, first in enumerate(strokes):
        first_screen = screens[first.source_edge_id]
        for second in strokes[first_index + 1 :]:
            second_screen = screens[second.source_edge_id]
            hit = segment_intersection_parameters(
                first_screen[0],
                first_screen[1],
                second_screen[0],
                second_screen[1],
                screen_epsilon,
            )
            if hit is None:
                continue
            kind, parameters = hit
            result.append(
                _ProjectedPathPairEvent(
                    first.source_edge_id,
                    second.source_edge_id,
                    kind,
                    tuple(float(value) for value in parameters),
                )
            )
    return tuple(result)


def _add_boundary(
    values: dict[str, list[float]],
    priorities: dict[str, dict[float, int]],
    path_id: str,
    value: float,
    priority: int,
) -> None:
    normalized = min(1.0, max(0.0, float(value)))
    values[path_id].append(normalized)
    priorities[path_id][normalized] = max(
        priority,
        priorities[path_id].get(normalized, 0),
    )


def _cluster_boundaries(
    values: Sequence[float],
    priorities: Mapping[float, int],
    tolerance: float,
) -> list[float]:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("boundary-cluster tolerance must be finite and non-negative")
    ordered = [float(value) for value in sorted(values)]
    if any(not np.isfinite(value) for value in ordered):
        raise ValueError("painter boundaries must be finite")

    # A cluster is bounded by its first member, not by a chain of adjacent
    # near-neighbours.  The latter can swallow arbitrarily distant real events
    # when every consecutive gap happens to lie inside the tolerance.
    clusters: list[list[float]] = []
    for value in ordered:
        if not clusters or value - clusters[-1][0] > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    unique: list[float] = []
    for cluster in clusters:
        highest = max(priorities.get(value, 0) for value in cluster)
        preferred = [value for value in cluster if priorities.get(value, 0) == highest]
        unique.append(float(np.median(preferred)))
    if not unique:
        return [0.0, 1.0]
    unique[0] = 0.0
    unique[-1] = 1.0
    return unique


def compute_path_fragments(
    model: OpenFaceVisibilityModel,
    visibility: OpenFaceVisibilityFrame,
    positions: Mapping[str, np.ndarray],
    normals: Mapping[str, np.ndarray],
    view: ParallelView,
    policy: TolerancePolicy,
    projected_faces: Mapping[str, tuple[np.ndarray, ...]],
    screen_epsilon: float,
    limits: OpenFaceUnifiedCompositingLimits,
    pair_events: Sequence[_ProjectedPathPairEvent] | None = None,
) -> tuple[PaintPathFragment, ...]:
    """Split semantic paths at visibility and painter-order events."""

    matrix = np.asarray(view.projection_matrix, dtype=float)
    direction = np.asarray(view.view_direction, dtype=float)
    boundaries: dict[str, list[float]] = {}
    priorities: dict[str, dict[float, int]] = {}
    for stroke in model.strokes:
        edge = visibility.edge_map[stroke.source_edge_id]
        raw = [
            float(value)
            for span in edge.spans
            for value in (span.start, span.end)
        ]
        boundaries[stroke.source_edge_id] = raw
        priorities[stroke.source_edge_id] = {
            value: (3 if value in {0.0, 1.0} else 1) for value in raw
        }

    # A visibility interval does not capture every painter event.  Add finite
    # face-overlap boundaries and line/face depth roots.
    for stroke in model.strokes:
        path_id = stroke.source_edge_id
        world = np.asarray(
            (positions[stroke.vertex_ids[0]], positions[stroke.vertex_ids[1]]),
            dtype=float,
        )
        screen = world @ matrix[:2].T
        delta = world[1] - world[0]
        for face in model.faces:
            interval = line_convex_polygon_interval(
                screen[0], screen[1], projected_faces[face.face_id], screen_epsilon
            )
            if interval is None:
                continue
            _add_boundary(boundaries, priorities, path_id, interval[0], 0)
            _add_boundary(boundaries, priorities, path_id, interval[1], 0)

            def difference(parameter: float) -> float:
                point = world[0] + parameter * delta
                screen_point = screen[0] + parameter * (screen[1] - screen[0])
                return float(np.dot(point, direction)) - _face_depth_at_screen_point(
                    screen_point,
                    face.vertex_ids,
                    positions,
                    normals[face.face_id],
                    view,
                )

            first, last = difference(interval[0]), difference(interval[1])
            epsilon = policy.resolve(
                (world[0], world[1], *(positions[key] for key in face.vertex_ids))
            ).depth
            if (
                first < -epsilon and last > epsilon
            ) or (last < -epsilon and first > epsilon):
                root = interval[0] - first * (interval[1] - interval[0]) / (
                    last - first
                )
                _add_boundary(boundaries, priorities, path_id, root, 2)

    strokes = tuple(sorted(model.strokes, key=lambda item: item.source_edge_id))
    events = (
        tuple(pair_events)
        if pair_events is not None
        else _compute_projected_path_pair_events(
            model,
            positions,
            view,
            screen_epsilon,
        )
    )

    # Localize projected path crossings and collinear overlap/depth events.
    for event in events:
        first = model.stroke_map[event.first_path_id]
        second = model.stroke_map[event.second_path_id]
        first_world = np.asarray(
            (positions[first.vertex_ids[0]], positions[first.vertex_ids[1]]),
            dtype=float,
        )
        first_depth = first_world @ direction
        second_world = np.asarray(
            (positions[second.vertex_ids[0]], positions[second.vertex_ids[1]]),
            dtype=float,
        )
        second_depth = second_world @ direction
        values = event.parameters
        if event.kind == "point":
            first_t, second_t = values
            _add_boundary(boundaries, priorities, first.source_edge_id, first_t, 2)
            _add_boundary(boundaries, priorities, second.source_edge_id, second_t, 2)
            continue
        first_a, first_b, second_a, second_b = values
        for path_id, low, high in (
            (first.source_edge_id, first_a, first_b),
            (second.source_edge_id, second_a, second_b),
        ):
            _add_boundary(boundaries, priorities, path_id, low, 2)
            _add_boundary(boundaries, priorities, path_id, high, 2)
        differences = (
            float(
                first_depth[0] + first_a * (first_depth[1] - first_depth[0])
                - second_depth[0]
                - second_a * (second_depth[1] - second_depth[0])
            ),
            float(
                first_depth[0] + first_b * (first_depth[1] - first_depth[0])
                - second_depth[0]
                - second_b * (second_depth[1] - second_depth[0])
            ),
        )
        depth_epsilon = policy.resolve(
            (first_world[0], first_world[1], second_world[0], second_world[1])
        ).depth
        if differences[0] * differences[1] < -(depth_epsilon * depth_epsilon):
            ratio = -differences[0] / (differences[1] - differences[0])
            _add_boundary(
                boundaries,
                priorities,
                first.source_edge_id,
                first_a + ratio * (first_b - first_a),
                2,
            )
            _add_boundary(
                boundaries,
                priorities,
                second.source_edge_id,
                second_a + ratio * (second_b - second_a),
                2,
            )

    result: list[PaintPathFragment] = []
    for stroke in strokes:
        path_id = stroke.source_edge_id
        edge = visibility.edge_map[path_id]
        world = np.asarray(
            (positions[stroke.vertex_ids[0]], positions[stroke.vertex_ids[1]]),
            dtype=float,
        )
        screen = world @ matrix[:2].T
        projected_length = float(np.linalg.norm(screen[1] - screen[0]))
        parameter_tolerance = max(
            edge.parameter_epsilon,
            _PAINTER_EVENT_EPSILON_FACTOR
            * screen_epsilon
            / max(projected_length, screen_epsilon),
        )
        unique = _cluster_boundaries(
            boundaries[path_id], priorities[path_id], parameter_tolerance
        )
        path_result: list[PaintPathFragment] = []
        for start, end in zip(unique, unique[1:]):
            if end - start <= parameter_tolerance:
                continue
            midpoint = 0.5 * (start + end)
            matches = [
                span
                for span in edge.spans
                if span.start - edge.parameter_epsilon
                <= midpoint
                <= span.end + edge.parameter_epsilon
            ]
            if len(matches) != 1:
                raise OpenFaceUnifiedCompositingError(
                    f"path {path_id!r} lost its visibility span"
                )
            span = matches[0]
            try:
                kind = VisibilityKind(span.kind)
            except ValueError as exc:
                raise OpenFaceUnifiedCompositingError(
                    f"path {path_id!r} has unsupported visibility kind {span.kind!r}"
                ) from exc
            path_result.append(
                PaintPathFragment(
                    f"path:{path_id}:fragment:{len(path_result)}",
                    path_id,
                    ParameterInterval(float(start), float(end)),
                    kind,
                    tuple(sorted(set(span.occluder_face_ids))),
                    tuple(sorted(set(span.occluder_logical_surface_ids))),
                )
            )
        if len(path_result) > limits.max_fragments_per_path:
            raise OpenFaceUnifiedCompositingError(
                f"path {path_id!r} produced {len(path_result)} fragments; "
                f"limit is {limits.max_fragments_per_path}"
            )
        result.extend(path_result)
    if len(result) > limits.max_total_fragments:
        raise OpenFaceUnifiedCompositingError(
            f"unified painter produced {len(result)} fragments; "
            f"limit is {limits.max_total_fragments}"
        )
    return tuple(result)


__all__ = ["compute_path_fragments"]
