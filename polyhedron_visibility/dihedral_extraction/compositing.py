from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from ..contract import TolerancePolicy
from ..parallel_solver import ParallelView, SolverError
from ..sections.compositing import (
    FragmentOrderRelation,
    TransparentSectionCompositingError,
    TransparentTriangle,
    _fragment_relations,
)
from .contract import DerivedDihedralModel, RigidTransform3D
from .solver import DerivedDihedralSolverError, compute_derived_dihedral_visibility
from .trace import DerivedDihedralVisibilityFrame


DERIVED_DIHEDRAL_TRANSPARENT_COMPOSITING_SCHEMA = (
    "manim-derived-dihedral-transparent-compositing/v1"
)


class DerivedDihedralTransparentCompositingError(ValueError):
    """Raised when intersecting transparent faces cannot be ordered exactly."""


@dataclass(frozen=True)
class DerivedDihedralTransparentCompositingFrame:
    visibility: DerivedDihedralVisibilityFrame
    projection_matrix: tuple[tuple[float, float, float], ...]
    fragments: tuple[TransparentTriangle, ...]
    draw_order: tuple[str, ...]
    order_relations: tuple[FragmentOrderRelation, ...]
    schema: str = DERIVED_DIHEDRAL_TRANSPARENT_COMPOSITING_SCHEMA

    @property
    def fragment_map(self) -> dict[str, TransparentTriangle]:
        return {item.fragment_id: item for item in self.fragments}

    @property
    def draw_batches(self) -> tuple[tuple[str, ...], ...]:
        """Consecutive same-source fragments that may share one fill pass."""

        fragment_map = self.fragment_map
        batches: list[list[str]] = []
        for fragment_id in self.draw_order:
            source_face_id = fragment_map[fragment_id].source_face_id
            if (
                batches
                and fragment_map[batches[-1][0]].source_face_id == source_face_id
            ):
                batches[-1].append(fragment_id)
            else:
                batches.append([fragment_id])
        return tuple(tuple(batch) for batch in batches)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "visibility": self.visibility.to_dict(),
            "projectionMatrix": [list(row) for row in self.projection_matrix],
            "fragments": [item.to_dict() for item in self.fragments],
            "drawOrder": list(self.draw_order),
            "drawBatches": [list(batch) for batch in self.draw_batches],
            "orderRelations": [item.to_dict() for item in self.order_relations],
        }


@dataclass(frozen=True)
class _Vertex:
    token: str
    point: np.ndarray


@dataclass(frozen=True)
class _Cell:
    token: str
    vertices: tuple[_Vertex, ...]


def _face_normal(points: Sequence[np.ndarray], epsilon: float, label: str) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    for index in range(1, len(values) - 1):
        normal = np.cross(values[index] - values[0], values[index + 1] - values[0])
        length = float(np.linalg.norm(normal))
        if length > epsilon:
            return normal / length
    raise DerivedDihedralTransparentCompositingError(
        f"transparent face {label} is degenerate"
    )


def _dedupe(vertices: Sequence[_Vertex], epsilon: float) -> tuple[_Vertex, ...]:
    result: list[_Vertex] = []
    for vertex in vertices:
        if not result or float(np.linalg.norm(result[-1].point - vertex.point)) > epsilon:
            result.append(vertex)
    if len(result) > 1 and float(np.linalg.norm(result[0].point - result[-1].point)) <= epsilon:
        result.pop()
    return tuple(result)


def _clip_cell(
    cell: _Cell,
    distances: Mapping[str, float],
    *,
    keep_positive: bool,
    epsilon: float,
    splitter_id: str,
) -> _Cell | None:
    vertices = cell.vertices
    if not vertices:
        return None

    def signed(vertex: _Vertex) -> float:
        value = distances[vertex.token]
        return value if keep_positive else -value

    output: list[_Vertex] = []
    previous = vertices[-1]
    previous_value = signed(previous)
    previous_inside = previous_value >= -epsilon
    for current in vertices:
        current_value = signed(current)
        current_inside = current_value >= -epsilon
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > epsilon * 1.0e-6:
                parameter = previous_value / denominator
                point = previous.point + parameter * (current.point - previous.point)
                edge_token = "|".join(sorted((previous.token, current.token)))
                token = "ix:" + hashlib.sha256(
                    f"{splitter_id}|{edge_token}".encode("utf-8")
                ).hexdigest()[:16]
                output.append(_Vertex(token, point))
                # Later splitters need signed distances for generated points.
                if isinstance(distances, dict):
                    distances[token] = 0.0
            elif abs(previous_value) <= epsilon:
                output.append(previous)
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    deduped = _dedupe(output, epsilon)
    if len(deduped) < 3:
        return None
    suffix = "+" if keep_positive else "-"
    return _Cell(f"{cell.token}|{splitter_id}:{suffix}", deduped)


def _split_cells(
    cells: Sequence[_Cell],
    splitter_id: str,
    splitter_points: Sequence[np.ndarray],
    policy: TolerancePolicy,
) -> tuple[_Cell, ...]:
    tolerance = policy.resolve(splitter_points)
    normal = _face_normal(splitter_points, tolerance.world, splitter_id)
    plane_point = np.asarray(splitter_points[0], dtype=float)
    result: list[_Cell] = []
    for cell in cells:
        distances: dict[str, float] = {
            vertex.token: float(np.dot(vertex.point - plane_point, normal))
            for vertex in cell.vertices
        }
        positive = any(value > tolerance.boundary for value in distances.values())
        negative = any(value < -tolerance.boundary for value in distances.values())
        if not positive or not negative:
            result.append(cell)
            continue
        first = _clip_cell(
            cell,
            distances,
            keep_positive=True,
            epsilon=tolerance.boundary,
            splitter_id=splitter_id,
        )
        second = _clip_cell(
            cell,
            distances,
            keep_positive=False,
            epsilon=tolerance.boundary,
            splitter_id=splitter_id,
        )
        if first is not None:
            result.append(first)
        if second is not None:
            result.append(second)
    return tuple(result)


def _polygon_plane_interval(
    polygon_points: Sequence[np.ndarray],
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    line_direction: np.ndarray,
    epsilon: float,
) -> tuple[float, float] | None:
    """Return the finite polygon's interval on one plane-intersection line."""

    intersections: list[np.ndarray] = []

    def append_unique(point: np.ndarray) -> None:
        if not any(
            float(np.linalg.norm(point - existing)) <= epsilon
            for existing in intersections
        ):
            intersections.append(point)

    values = tuple(np.asarray(point, dtype=float) for point in polygon_points)
    distances = tuple(
        float(np.dot(point - plane_point, plane_normal)) for point in values
    )
    for index, start in enumerate(values):
        end = values[(index + 1) % len(values)]
        start_distance = distances[index]
        end_distance = distances[(index + 1) % len(values)]
        if abs(start_distance) <= epsilon:
            append_unique(start)
        if start_distance > epsilon and end_distance < -epsilon or (
            start_distance < -epsilon and end_distance > epsilon
        ):
            denominator = start_distance - end_distance
            parameter = start_distance / denominator
            append_unique(start + parameter * (end - start))

    if len(intersections) < 2:
        return None
    parameters = [float(np.dot(point, line_direction)) for point in intersections]
    lower = min(parameters)
    upper = max(parameters)
    if upper - lower <= epsilon:
        return None
    return lower, upper


def _finite_faces_intersect_transversely(
    first_points: Sequence[np.ndarray],
    second_points: Sequence[np.ndarray],
    policy: TolerancePolicy,
) -> bool:
    """Whether two finite convex faces share a positive-length 3D crossing.

    The old compositor split a face whenever the *infinite supporting plane*
    of another face crossed it.  That was sufficient for depth ordering, but
    it left many unnecessary cuts after the actual finite polygons had moved
    apart.  Under a parallel view, two projected faces can exchange depth only
    where the finite faces themselves meet, so a positive shared interval on
    their plane-intersection line is the authoritative split gate.
    """

    first = tuple(np.asarray(point, dtype=float) for point in first_points)
    second = tuple(np.asarray(point, dtype=float) for point in second_points)
    first_tolerance = policy.resolve(first)
    second_tolerance = policy.resolve(second)
    epsilon = max(first_tolerance.boundary, second_tolerance.boundary)
    first_normal = _face_normal(first, first_tolerance.world, "first finite face")
    second_normal = _face_normal(
        second,
        second_tolerance.world,
        "second finite face",
    )
    direction = np.cross(first_normal, second_normal)
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= max(
        first_tolerance.angular,
        second_tolerance.angular,
    ):
        return False
    direction /= direction_length

    first_interval = _polygon_plane_interval(
        first,
        plane_point=second[0],
        plane_normal=second_normal,
        line_direction=direction,
        epsilon=epsilon,
    )
    second_interval = _polygon_plane_interval(
        second,
        plane_point=first[0],
        plane_normal=first_normal,
        line_direction=direction,
        epsilon=epsilon,
    )
    if first_interval is None or second_interval is None:
        return False
    overlap_start = max(first_interval[0], second_interval[0])
    overlap_end = min(first_interval[1], second_interval[1])
    return overlap_end - overlap_start > epsilon


def _triangulate(
    face_id: str,
    role: str,
    cells: Sequence[_Cell],
    policy: TolerancePolicy,
) -> list[TransparentTriangle]:
    result: list[TransparentTriangle] = []
    for cell in cells:
        points = [item.point for item in cell.vertices]
        tolerance = policy.resolve(points)
        for index in range(1, len(cell.vertices) - 1):
            vertices = (cell.vertices[0], cell.vertices[index], cell.vertices[index + 1])
            area = float(
                np.linalg.norm(
                    np.cross(
                        vertices[1].point - vertices[0].point,
                        vertices[2].point - vertices[0].point,
                    )
                )
            )
            if area <= tolerance.world * tolerance.world:
                continue
            digest = hashlib.sha256(
                f"{face_id}|{cell.token}|{index}".encode("utf-8")
            ).hexdigest()[:20]
            result.append(
                TransparentTriangle(
                    fragment_id=f"fragment:{digest}",
                    surface_id=face_id,
                    role=role,
                    vertices=tuple(
                        tuple(float(value) for value in item.point)
                        for item in vertices
                    ),  # type: ignore[arg-type]
                    vertex_tokens=tuple(item.token for item in vertices),  # type: ignore[arg-type]
                    source_face_id=face_id,
                )
            )
    return result


def _surface_aware_draw_order(
    fragments: Sequence[TransparentTriangle],
    relations: Sequence[FragmentOrderRelation],
) -> tuple[str, ...]:
    """Return a valid far-to-near order while keeping equal fills together.

    A source face is often triangulated only so the depth solver can reason
    about it.  Whenever several of its triangles are simultaneously ready in
    the ordering DAG, drawing them consecutively is equally correct and lets
    the Manim layer composite them as one compound fill without internal
    antialiasing seams.
    """

    fragment_map = {item.fragment_id: item for item in fragments}
    identities = set(fragment_map)
    outgoing = {fragment_id: set() for fragment_id in identities}
    indegree = {fragment_id: 0 for fragment_id in identities}
    for relation in relations:
        if (
            relation.far_fragment_id not in identities
            or relation.near_fragment_id not in identities
        ):
            raise DerivedDihedralTransparentCompositingError(
                "transparent ordering relation references an unknown fragment"
            )
        if relation.near_fragment_id not in outgoing[relation.far_fragment_id]:
            outgoing[relation.far_fragment_id].add(relation.near_fragment_id)
            indegree[relation.near_fragment_id] += 1

    ready = {fragment_id for fragment_id, degree in indegree.items() if degree == 0}
    order: list[str] = []
    current_surface: str | None = None
    while ready:
        same_surface = [
            fragment_id
            for fragment_id in ready
            if fragment_map[fragment_id].source_face_id == current_surface
        ]
        if same_surface:
            current = min(same_surface)
        else:
            current = min(
                ready,
                key=lambda fragment_id: (
                    fragment_map[fragment_id].source_face_id or "",
                    fragment_id,
                ),
            )
        ready.remove(current)
        order.append(current)
        current_surface = fragment_map[current].source_face_id
        for target in sorted(outgoing[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.add(target)

    if len(order) != len(identities):
        cyclic = sorted(fragment_id for fragment_id, degree in indegree.items() if degree > 0)
        raise DerivedDihedralTransparentCompositingError(
            "transparent fragment ordering contains a cycle: " + ", ".join(cyclic)
        )
    return tuple(order)


def transparent_dihedral_triangle_capacity(model: DerivedDihedralModel) -> int:
    """Stable upper bound for all line arrangements on every transparent face.

    ``m`` general-position plane cuts induce at most ``m`` lines on a convex
    n-gon.  Triangulating every resulting cell needs at most
    ``n + m(m+1) - 2`` triangles.  The deliberately conservative bound keeps
    the Cairo slot pool fixed even when several cut lines cross in one face.
    """

    def arrangement_limit(vertex_count: int, splitter_count: int) -> int:
        return max(
            1,
            vertex_count + splitter_count * (splitter_count + 1) - 2,
        )

    solid_count = sum(
        arrangement_limit(len(face.vertex_ids), 2)
        for face in model.solid.faces
    )
    solid_face_count = len(model.solid.faces)
    extracted_count = sum(
        arrangement_limit(
            len(model.solid.face_map[face_id].vertex_ids),
            solid_face_count,
        )
        for face_id in model.extraction.source_face_ids
    )
    return solid_count + extracted_count


def compute_derived_dihedral_transparent_compositing(
    model: DerivedDihedralModel,
    *,
    transform: RigidTransform3D,
    projection_matrix: Sequence[Sequence[float]],
    solid_vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
) -> DerivedDihedralTransparentCompositingFrame:
    policy = tolerance_policy or TolerancePolicy()
    try:
        visibility = compute_derived_dihedral_visibility(
            model,
            transform=transform,
            projection_matrix=projection_matrix,
            solid_vertex_positions=solid_vertex_positions,
            tolerance_policy=policy,
        )
        view = ParallelView.from_matrix(projection_matrix)
    except (DerivedDihedralSolverError, SolverError) as exc:
        raise DerivedDihedralTransparentCompositingError(str(exc)) from exc
    raw_solid = (
        model.solid.entry_positions
        if solid_vertex_positions is None
        else solid_vertex_positions
    )
    solid = {key: np.asarray(raw_solid[key], dtype=float) for key in raw_solid}
    extracted = {
        key: transform.apply(model.solid.vertex_map[key].entry_position)
        for key in model.extracted_vertex_ids
    }
    coincident = set(visibility.coincident_source_face_ids)
    solid_faces = [
        face for face in model.solid.faces if face.face_id not in coincident
    ]
    extracted_faces = [model.solid.face_map[item] for item in model.extraction.source_face_ids]
    fragments: list[TransparentTriangle] = []

    for face in solid_faces:
        face_id = model.solid_face_id(face.face_id)
        face_points = [solid[item] for item in face.vertex_ids]
        cells: tuple[_Cell, ...] = (
            _Cell(
                face_id,
                tuple(
                    _Vertex(model.solid_vertex_id(item), solid[item])
                    for item in face.vertex_ids
                ),
            ),
        )
        for splitter in extracted_faces:
            splitter_id = model.extracted_face_id(splitter.face_id)
            splitter_points = [
                extracted[item] for item in splitter.vertex_ids
            ]
            if not _finite_faces_intersect_transversely(
                face_points,
                splitter_points,
                policy,
            ):
                continue
            cells = _split_cells(
                cells,
                splitter_id,
                splitter_points,
                policy,
            )
        fragments.extend(_triangulate(face_id, "solid_face", cells, policy))

    for face in extracted_faces:
        face_id = model.extracted_face_id(face.face_id)
        face_points = [extracted[item] for item in face.vertex_ids]
        cells = (
            _Cell(
                face_id,
                tuple(
                    _Vertex(model.extracted_vertex_id(item), extracted[item])
                    for item in face.vertex_ids
                ),
            ),
        )
        for splitter in solid_faces:
            splitter_id = model.solid_face_id(splitter.face_id)
            splitter_points = [solid[item] for item in splitter.vertex_ids]
            if not _finite_faces_intersect_transversely(
                face_points,
                splitter_points,
                policy,
            ):
                continue
            cells = _split_cells(
                cells,
                splitter_id,
                splitter_points,
                policy,
            )
        fragments.extend(_triangulate(face_id, "section_inside", cells, policy))

    fragments.sort(key=lambda item: item.fragment_id)
    if len(fragments) > transparent_dihedral_triangle_capacity(model):
        raise DerivedDihedralTransparentCompositingError(
            "transparent dihedral frame exceeds its precomputed triangle capacity"
        )
    try:
        relations = _fragment_relations(
            fragments,
            view,
            policy,
            coplanar_policy="section_over_solid",
        )
        order = _surface_aware_draw_order(fragments, relations)
    except TransparentSectionCompositingError as exc:
        raise DerivedDihedralTransparentCompositingError(str(exc)) from exc
    return DerivedDihedralTransparentCompositingFrame(
        visibility,
        view.projection_matrix,
        tuple(fragments),
        order,
        relations,
    )


def canonical_derived_dihedral_compositing_json(
    frame: DerivedDihedralTransparentCompositingFrame,
) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "DERIVED_DIHEDRAL_TRANSPARENT_COMPOSITING_SCHEMA",
    "DerivedDihedralTransparentCompositingError",
    "DerivedDihedralTransparentCompositingFrame",
    "canonical_derived_dihedral_compositing_json",
    "compute_derived_dihedral_transparent_compositing",
    "transparent_dihedral_triangle_capacity",
]
