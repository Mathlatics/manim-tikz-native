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
from ..contract import ContractError, TolerancePolicy, VisibilityModel
from ..parallel_solver import ParallelView, SolverError
from .contract import SectionPlane3D
from .solver import ConvexSectionSolverError, intersect_plane_with_convex_polyhedron
from .trace import ConvexSectionFrame


TRANSPARENT_SECTION_COMPOSITING_SCHEMA = (
    "manim-convex-section-transparent-compositing/v1"
)


class TransparentSectionCompositingError(ValueError):
    """Raised when transparent fragments cannot be ordered without guessing."""


@dataclass(frozen=True)
class TransparentTriangle:
    """One stable, independently sortable transparent triangle."""

    fragment_id: str
    surface_id: str
    role: str
    vertices: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    vertex_tokens: tuple[str, str, str]
    source_face_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "fragmentId": self.fragment_id,
            "surfaceId": self.surface_id,
            "role": self.role,
            "vertices": [list(item) for item in self.vertices],
            "vertexTokens": list(self.vertex_tokens),
        }
        if self.source_face_id is not None:
            result["sourceFaceId"] = self.source_face_id
        return result


@dataclass(frozen=True)
class FragmentOrderRelation:
    far_fragment_id: str
    near_fragment_id: str
    overlap_area: float
    minimum_depth_difference: float
    maximum_depth_difference: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "farFragmentId": self.far_fragment_id,
            "nearFragmentId": self.near_fragment_id,
            "overlapArea": self.overlap_area,
            "minimumDepthDifference": self.minimum_depth_difference,
            "maximumDepthDifference": self.maximum_depth_difference,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TransparentSectionCompositingFrame:
    section: ConvexSectionFrame
    projection_matrix: tuple[tuple[float, float, float], ...]
    fragments: tuple[TransparentTriangle, ...]
    draw_order: tuple[str, ...]
    order_relations: tuple[FragmentOrderRelation, ...]
    schema: str = TRANSPARENT_SECTION_COMPOSITING_SCHEMA

    @property
    def fragment_map(self) -> dict[str, TransparentTriangle]:
        return {item.fragment_id: item for item in self.fragments}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "section": self.section.to_dict(),
            "projectionMatrix": [list(row) for row in self.projection_matrix],
            "fragments": [item.to_dict() for item in self.fragments],
            "drawOrder": list(self.draw_order),
            "orderRelations": [item.to_dict() for item in self.order_relations],
        }


@dataclass(frozen=True)
class _Vertex3:
    token: str
    point: np.ndarray


@dataclass(frozen=True)
class _Vertex2:
    token: str
    uv: np.ndarray
    point: np.ndarray


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise TransparentSectionCompositingError(
            f"{label} must be a finite three-component point"
        )
    return point


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
        raise TransparentSectionCompositingError(
            f"invalid closed convex polyhedron: {exc}"
        ) from exc
    return {
        vertex_id: _point3(raw[vertex_id], f"vertex {vertex_id}")
        for vertex_id in sorted(raw)
    }


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _signed_area2(points: Sequence[np.ndarray]) -> float:
    return 0.5 * sum(
        _cross2(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def _dedupe_vertices3(
    values: Sequence[_Vertex3], epsilon: float
) -> list[_Vertex3]:
    result: list[_Vertex3] = []
    for item in values:
        if result and float(np.linalg.norm(result[-1].point - item.point)) <= epsilon:
            if item.token < result[-1].token:
                result[-1] = item
            continue
        result.append(item)
    if (
        len(result) >= 2
        and float(np.linalg.norm(result[0].point - result[-1].point)) <= epsilon
    ):
        if result[-1].token < result[0].token:
            result[0] = result[-1]
        result.pop()
    return result


def _dedupe_vertices2(
    values: Sequence[_Vertex2], epsilon: float
) -> list[_Vertex2]:
    result: list[_Vertex2] = []
    for item in values:
        if result and float(np.linalg.norm(result[-1].uv - item.uv)) <= epsilon:
            if item.token < result[-1].token:
                result[-1] = item
            continue
        result.append(item)
    if (
        len(result) >= 2
        and float(np.linalg.norm(result[0].uv - result[-1].uv)) <= epsilon
    ):
        if result[-1].token < result[0].token:
            result[0] = result[-1]
        result.pop()
    return result


def _edge_key(start: str, end: str) -> str:
    return "--".join(sorted((start, end)))


def _section_edge_points(
    section: ConvexSectionFrame,
) -> dict[str, _Vertex3]:
    result: dict[str, _Vertex3] = {}
    for item in section.points:
        vertex = _Vertex3(
            f"section-point:{item.point_id}", np.asarray(item.position, dtype=float)
        )
        for source_edge_id in item.source_edge_ids:
            result[source_edge_id] = vertex
    return result


def _clip_face_halfspace(
    vertices: Sequence[_Vertex3],
    distances: Mapping[str, float],
    *,
    keep_positive: bool,
    epsilon: float,
    snapped_edges: Mapping[str, _Vertex3],
) -> list[_Vertex3]:
    if not vertices:
        return []

    def signed(item: _Vertex3) -> float:
        value = distances[item.token]
        return value if keep_positive else -value

    result: list[_Vertex3] = []
    previous = vertices[-1]
    previous_value = signed(previous)
    previous_inside = previous_value >= -epsilon
    for current in vertices:
        current_value = signed(current)
        current_inside = current_value >= -epsilon
        if current_inside != previous_inside:
            original_start = previous.token.removeprefix("vertex:")
            original_end = current.token.removeprefix("vertex:")
            snapped = snapped_edges.get(_edge_key(original_start, original_end))
            if snapped is None:
                denominator = previous_value - current_value
                if abs(denominator) <= epsilon:
                    raise TransparentSectionCompositingError(
                        "face-plane crossing is numerically ambiguous"
                    )
                parameter = previous_value / denominator
                point = previous.point + parameter * (current.point - previous.point)
                token = "face-cut:" + _edge_key(original_start, original_end)
                snapped = _Vertex3(token, point)
            result.append(snapped)
        if current_inside:
            result.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return _dedupe_vertices3(result, epsilon)


def _triangle_id(
    surface_id: str,
    role: str,
    vertex_tokens: Sequence[str],
) -> str:
    evidence = "|".join(sorted(vertex_tokens))
    digest = hashlib.sha256(evidence.encode("utf-8")).hexdigest()[:14]
    return f"{surface_id}:{role}:triangle:{digest}"


def _triangulate3(
    values: Sequence[_Vertex3],
    *,
    surface_id: str,
    role: str,
    source_face_id: str | None,
    area_epsilon: float,
) -> list[TransparentTriangle]:
    if len(values) < 3:
        return []
    first_index = min(range(len(values)), key=lambda index: values[index].token)
    ordered = list(values[first_index:]) + list(values[:first_index])
    result: list[TransparentTriangle] = []
    for index in range(1, len(ordered) - 1):
        triangle = (ordered[0], ordered[index], ordered[index + 1])
        area = 0.5 * float(
            np.linalg.norm(
                np.cross(
                    triangle[1].point - triangle[0].point,
                    triangle[2].point - triangle[0].point,
                )
            )
        )
        if area <= area_epsilon:
            continue
        tokens = tuple(item.token for item in triangle)
        result.append(
            TransparentTriangle(
                _triangle_id(surface_id, role, tokens),
                surface_id,
                role,
                tuple(
                    tuple(float(component) for component in item.point)
                    for item in triangle
                ),
                tokens,
                source_face_id,
            )
        )
    return result


def _solid_face_fragments(
    model: VisibilityModel,
    positions: Mapping[str, np.ndarray],
    plane: SectionPlane3D,
    section: ConvexSectionFrame,
    policy: TolerancePolicy,
) -> list[TransparentTriangle]:
    point = np.asarray(plane.point, dtype=float)
    normal = np.asarray(plane.normal, dtype=float)
    snapped_edges = _section_edge_points(section)
    result: list[TransparentTriangle] = []
    for face in sorted(model.faces, key=lambda item: item.face_id):
        original = [
            _Vertex3(f"vertex:{vertex_id}", positions[vertex_id])
            for vertex_id in face.vertex_ids
        ]
        tolerance = policy.resolve([item.point for item in original])
        distances = {
            item.token: float(np.dot(item.point - point, normal))
            for item in original
        }
        positive = any(value > tolerance.boundary for value in distances.values())
        negative = any(value < -tolerance.boundary for value in distances.values())
        surface_id = f"solid-face:{face.face_id}"
        area_epsilon = tolerance.world * tolerance.world
        if not positive and not negative:
            result.extend(
                _triangulate3(
                    original,
                    surface_id=surface_id,
                    role="solid_face_coplanar",
                    source_face_id=face.face_id,
                    area_epsilon=area_epsilon,
                )
            )
            continue
        if positive and negative:
            for keep_positive, side in ((False, "negative"), (True, "positive")):
                clipped = _clip_face_halfspace(
                    original,
                    distances,
                    keep_positive=keep_positive,
                    epsilon=tolerance.boundary,
                    snapped_edges=snapped_edges,
                )
                result.extend(
                    _triangulate3(
                        clipped,
                        surface_id=surface_id,
                        role=f"solid_face_{side}",
                        source_face_id=face.face_id,
                        area_epsilon=area_epsilon,
                    )
                )
            continue
        role = "solid_face_positive" if positive else "solid_face_negative"
        result.extend(
            _triangulate3(
                original,
                surface_id=surface_id,
                role=role,
                source_face_id=face.face_id,
                area_epsilon=area_epsilon,
            )
        )
    return result


def _plane_vertex(
    token: str,
    uv: Sequence[float],
    plane: SectionPlane3D,
) -> _Vertex2:
    uv_point = np.asarray(uv, dtype=float)
    u_axis, v_axis, _normal = plane.basis
    world = (
        np.asarray(plane.point, dtype=float)
        + uv_point[0] * u_axis
        + uv_point[1] * v_axis
    )
    return _Vertex2(token, uv_point, world)


def _clip_uv_halfspace(
    vertices: Sequence[_Vertex2],
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    *,
    keep_inside: bool,
    epsilon: float,
    boundary_token: str,
) -> list[_Vertex2]:
    if not vertices:
        return []
    direction = edge_end - edge_start
    boundary_threshold = epsilon * max(float(np.linalg.norm(direction)), epsilon)

    def signed(item: _Vertex2) -> float:
        value = _cross2(direction, item.uv - edge_start)
        return value if keep_inside else -value

    result: list[_Vertex2] = []
    previous = vertices[-1]
    previous_value = signed(previous)
    previous_inside = previous_value >= -boundary_threshold
    for current in vertices:
        current_value = signed(current)
        current_inside = current_value >= -boundary_threshold
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) <= max(
                epsilon * epsilon,
                boundary_threshold * 1.0e-6,
            ):
                raise TransparentSectionCompositingError(
                    "section boundary and plane partition edge are ambiguous"
                )
            parameter = previous_value / denominator
            uv = previous.uv + parameter * (current.uv - previous.uv)
            point = previous.point + parameter * (current.point - previous.point)
            evidence = "|".join(sorted((previous.token, current.token)))
            token = f"plane-cut:{boundary_token}:{evidence}"
            result.append(_Vertex2(token, uv, point))
        if current_inside:
            result.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return _dedupe_vertices2(result, epsilon)


def _plane_fragments(
    plane: SectionPlane3D,
    section: ConvexSectionFrame,
    policy: TolerancePolicy,
) -> list[TransparentTriangle]:
    patch = [
        _plane_vertex(
            f"plane-corner:{index}",
            uv,
            plane,
        )
        for index, uv in enumerate(
            (
                (-plane.half_width, -plane.half_height),
                (plane.half_width, -plane.half_height),
                (plane.half_width, plane.half_height),
                (-plane.half_width, plane.half_height),
            )
        )
    ]
    tolerance = policy.resolve([item.point for item in patch])
    area_epsilon = tolerance.world * tolerance.world
    if section.kind != "polygon":
        return _triangulate3(
            [_Vertex3(item.token, item.point) for item in patch],
            surface_id=f"cutting-plane:{plane.plane_id}",
            role="plane_outside",
            source_face_id=None,
            area_epsilon=area_epsilon,
        )

    section_vertices = [
        _Vertex2(
            f"section-point:{item.point_id}",
            np.asarray(plane.coordinates_in_plane(item.position), dtype=float),
            np.asarray(item.position, dtype=float),
        )
        for item in section.points
    ]
    section_area = _signed_area2([item.uv for item in section_vertices])
    if abs(section_area) <= area_epsilon:
        raise TransparentSectionCompositingError(
            "polygon section has no stable in-plane area"
        )
    if section_area < 0:
        section_vertices.reverse()
    for item in section_vertices:
        if (
            abs(float(item.uv[0])) >= plane.half_width - tolerance.boundary
            or abs(float(item.uv[1])) >= plane.half_height - tolerance.boundary
        ):
            raise TransparentSectionCompositingError(
                "cutting-plane patch must contain the complete section with a positive margin"
            )

    result: list[TransparentTriangle] = []
    uv_epsilon = max(policy.absolute_floor, policy.relative * max(
        2.0 * plane.half_width,
        2.0 * plane.half_height,
    )) * policy.boundary_factor
    boundaries: list[tuple[_Vertex2, _Vertex2, str]] = []
    for index, start in enumerate(section_vertices):
        end = section_vertices[(index + 1) % len(section_vertices)]
        boundary_token = _edge_key(start.token, end.token)
        boundaries.append((start, end, boundary_token))

    # A ring-only decomposition is insufficient: one outside piece can still
    # cross another solid-face/plane intersection line and therefore exchange
    # depth with that face.  Split the complete finite patch by every section
    # boundary's supporting line.  Each resulting cell then lies on one fixed
    # side of every intersected solid face.
    cells: list[list[_Vertex2]] = [patch]
    for start, end, boundary_token in boundaries:
        next_cells: list[list[_Vertex2]] = []
        for cell in cells:
            for keep_inside in (False, True):
                clipped = _clip_uv_halfspace(
                    cell,
                    start.uv,
                    end.uv,
                    keep_inside=keep_inside,
                    epsilon=uv_epsilon,
                    boundary_token=boundary_token,
                )
                if (
                    len(clipped) >= 3
                    and abs(_signed_area2([item.uv for item in clipped]))
                    > uv_epsilon * uv_epsilon
                ):
                    next_cells.append(clipped)
        cells = next_cells
    inside_cells: list[list[_Vertex2]] = []
    outside_cells: list[tuple[str, list[_Vertex2]]] = []
    for cell in cells:
        centroid = np.mean([item.uv for item in cell], axis=0)
        signature = "".join(
            "1" if _cross2(end.uv - start.uv, centroid - start.uv) >= -uv_epsilon else "0"
            for start, end, _token in boundaries
        )
        if signature == "1" * len(boundaries):
            inside_cells.append(cell)
        else:
            outside_cells.append((signature, cell))
    if not inside_cells:
        raise TransparentSectionCompositingError(
            "plane line arrangement did not preserve the authoritative section"
        )
    # Near a solid vertex two supporting lines can be closer than the resolved
    # tolerance and numerically split the same convex inside region into more
    # than one cell.  The authoritative section polygon is already available,
    # so only compare the union area here; never replace it with arrangement
    # output or require one particular cell topology.
    remaining_area = sum(
        abs(_signed_area2([item.uv for item in cell]))
        for cell in inside_cells
    )
    if abs(remaining_area - abs(section_area)) > max(
        uv_epsilon * uv_epsilon,
        abs(section_area) * policy.relative * policy.boundary_factor,
    ):
        raise TransparentSectionCompositingError(
            "plane partition does not reproduce the authoritative section area"
        )
    for signature, outside in sorted(
        outside_cells,
        key=lambda item: (
            item[0],
            tuple(
                float(value)
                for value in np.mean(
                    [vertex.uv for vertex in item[1]], axis=0
                )
            ),
        ),
    ):
        result.extend(
            _triangulate3(
                [_Vertex3(item.token, item.point) for item in outside],
                surface_id=f"cutting-plane:{plane.plane_id}:outside:{signature}",
                role="plane_outside",
                source_face_id=None,
                area_epsilon=area_epsilon,
            )
        )
    result.extend(
        _triangulate3(
            [_Vertex3(item.token, item.point) for item in section_vertices],
            surface_id=f"section:{section.section_id}",
            role="section_inside",
            source_face_id=None,
            area_epsilon=area_epsilon,
        )
    )
    return result


def _projected_triangle(
    triangle: TransparentTriangle,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(triangle.vertices, dtype=float)
    projected = points @ matrix.T
    screen = projected[:, :2]
    depth = projected[:, 2]
    if _signed_area2(screen) < 0:
        screen = screen[::-1]
        depth = depth[::-1]
    return screen, depth


def _clip_convex_polygon_2d(
    subject: Sequence[np.ndarray],
    clipper: Sequence[np.ndarray],
    epsilon: float,
) -> list[np.ndarray]:
    output = [np.asarray(item, dtype=float) for item in subject]
    for index, edge_start in enumerate(clipper):
        edge_end = clipper[(index + 1) % len(clipper)]
        direction = edge_end - edge_start
        boundary_threshold = epsilon * max(
            float(np.linalg.norm(direction)), epsilon
        )
        if not output:
            break
        input_values = output
        output = []
        previous = input_values[-1]
        previous_value = _cross2(direction, previous - edge_start)
        previous_inside = previous_value >= -boundary_threshold
        for current in input_values:
            current_value = _cross2(direction, current - edge_start)
            current_inside = current_value >= -boundary_threshold
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                if abs(denominator) > max(
                    epsilon * epsilon,
                    boundary_threshold * 1.0e-6,
                ):
                    parameter = previous_value / denominator
                    output.append(previous + parameter * (current - previous))
            if current_inside:
                output.append(current)
            previous = current
            previous_value = current_value
            previous_inside = current_inside
    deduped: list[np.ndarray] = []
    for item in output:
        if not deduped or float(np.linalg.norm(deduped[-1] - item)) > epsilon:
            deduped.append(item)
    if (
        len(deduped) > 1
        and float(np.linalg.norm(deduped[0] - deduped[-1])) <= epsilon
    ):
        deduped.pop()
    return deduped


def _depth_coefficients(screen: np.ndarray, depth: np.ndarray) -> np.ndarray:
    matrix = np.column_stack((screen, np.ones(3)))
    try:
        return np.linalg.solve(matrix, depth)
    except np.linalg.LinAlgError as exc:
        raise TransparentSectionCompositingError(
            "a transparent fragment projects edge-on"
        ) from exc


def _fragment_relations(
    fragments: Sequence[TransparentTriangle],
    view: ParallelView,
    policy: TolerancePolicy,
    *,
    coplanar_policy: str,
) -> tuple[FragmentOrderRelation, ...]:
    if coplanar_policy not in {"section_over_solid", "solid_over_section", "fail"}:
        raise TransparentSectionCompositingError(
            "coplanar_policy must be 'section_over_solid', 'solid_over_section', or 'fail'"
        )
    matrix = np.asarray(view.projection_matrix, dtype=float)
    all_points = [point for item in fragments for point in item.vertices]
    tolerance = policy.resolve(all_points)
    projected_values = [
        np.asarray(item.vertices, dtype=float) @ matrix[:2].T for item in fragments
    ]
    if projected_values:
        stacked = np.vstack(projected_values)
        screen_extent = np.max(stacked, axis=0) - np.min(stacked, axis=0)
        screen_scale = max(float(np.linalg.norm(screen_extent)), policy.absolute_floor)
    else:
        screen_scale = 1.0
    screen_epsilon = (
        max(policy.absolute_floor, policy.relative * screen_scale)
        * policy.boundary_factor
    )
    area_epsilon = screen_epsilon * screen_epsilon
    depth_epsilon = tolerance.depth * max(float(np.linalg.norm(matrix[2])), 1.0e-300)
    projected = [_projected_triangle(item, matrix) for item in fragments]
    relations: list[FragmentOrderRelation] = []
    for first_index, first in enumerate(fragments):
        first_screen, first_depth = projected[first_index]
        if abs(_signed_area2(first_screen)) <= area_epsilon:
            continue
        first_coefficients = _depth_coefficients(first_screen, first_depth)
        for second_index in range(first_index + 1, len(fragments)):
            second = fragments[second_index]
            second_screen, second_depth = projected[second_index]
            if abs(_signed_area2(second_screen)) <= area_epsilon:
                continue
            overlap = _clip_convex_polygon_2d(
                first_screen, second_screen, screen_epsilon
            )
            if len(overlap) < 3:
                continue
            overlap_area = abs(_signed_area2(overlap))
            if overlap_area <= area_epsilon:
                continue
            second_coefficients = _depth_coefficients(second_screen, second_depth)
            difference_coefficients = first_coefficients - second_coefficients
            differences = [
                float(
                    difference_coefficients[0] * item[0]
                    + difference_coefficients[1] * item[1]
                    + difference_coefficients[2]
                )
                for item in overlap
            ]
            minimum = min(differences)
            maximum = max(differences)
            if minimum < -depth_epsilon and maximum > depth_epsilon:
                raise TransparentSectionCompositingError(
                    "transparent fragments still exchange depth inside one overlap; "
                    f"split {first.fragment_id!r} and {second.fragment_id!r} again"
                )
            if minimum < -depth_epsilon and maximum <= depth_epsilon:
                far, near = first, second
                reason = "depth"
            elif maximum > depth_epsilon and minimum >= -depth_epsilon:
                far, near = second, first
                reason = "depth"
            else:
                roles = {first.role, second.role}
                if "section_inside" in roles and any(
                    role.startswith("solid_face") for role in roles
                ):
                    section_fragment = first if first.role == "section_inside" else second
                    solid_fragment = second if section_fragment is first else first
                    if coplanar_policy == "section_over_solid":
                        far, near = solid_fragment, section_fragment
                    elif coplanar_policy == "solid_over_section":
                        far, near = section_fragment, solid_fragment
                    else:
                        raise TransparentSectionCompositingError(
                            "section and solid face overlap coplanarly"
                        )
                    reason = "coplanar_policy"
                else:
                    raise TransparentSectionCompositingError(
                        "distinct transparent fragments overlap at the same depth: "
                        f"{first.fragment_id!r}, {second.fragment_id!r}; "
                        f"area={overlap_area!r}, depth=({minimum!r}, {maximum!r}), "
                        f"epsilon=({area_epsilon!r}, {depth_epsilon!r})"
                    )
            relations.append(
                FragmentOrderRelation(
                    far.fragment_id,
                    near.fragment_id,
                    float(overlap_area),
                    float(minimum),
                    float(maximum),
                    reason,
                )
            )
    unique = {
        (item.far_fragment_id, item.near_fragment_id): item for item in relations
    }
    return tuple(
        unique[key]
        for key in sorted(unique)
    )


def _topological_draw_order(
    fragments: Sequence[TransparentTriangle],
    relations: Sequence[FragmentOrderRelation],
) -> tuple[str, ...]:
    """Adapt section fragment relations to the shared stable compositor."""

    identities = {item.fragment_id for item in fragments}
    for relation in relations:
        missing = {
            relation.far_fragment_id,
            relation.near_fragment_id,
        } - identities
        if missing:
            raise TransparentSectionCompositingError(
                "transparent fragment relation references unknown identities: "
                + ", ".join(sorted(missing))
            )
        if relation.far_fragment_id == relation.near_fragment_id:
            raise TransparentSectionCompositingError(
                "transparent fragment ordering contains a cycle: "
                + relation.far_fragment_id
            )
    constraints = tuple(
        PainterConstraint(
            relation.far_fragment_id,
            relation.near_fragment_id,
        )
        for relation in relations
    )
    try:
        # The v1 trace used a heap of fragment IDs, so lexicographic identity
        # remains the authored-order tie breaker for otherwise unrelated items.
        return stable_topological_sort(
            sorted(identities),
            constraints,
            key=lambda fragment_id: fragment_id,
        )
    except CompositorCycleError as exc:
        cyclic = sorted(str(fragment_id) for fragment_id in exc.unresolved)
        raise TransparentSectionCompositingError(
            "transparent fragment ordering contains a cycle: "
            + ", ".join(cyclic)
        ) from exc


def compute_transparent_section_compositing(
    section_id: str,
    model: VisibilityModel,
    plane: SectionPlane3D,
    *,
    projection_matrix: Sequence[Sequence[float]],
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
    coplanar_policy: str = "section_over_solid",
) -> TransparentSectionCompositingFrame:
    """Split and order a closed convex solid and one finite cutting plane.

    The section is solved in world space first.  The finite plane patch is then
    partitioned into disjoint outside pieces plus the authoritative section,
    every intersected solid face is split along the same source-edge evidence,
    and all transparent triangles receive an exact far-to-near order under the
    supplied parallel projection.
    """

    policy = tolerance_policy or TolerancePolicy()
    positions = _validated_positions(model, vertex_positions, policy)
    try:
        view = ParallelView.from_matrix(projection_matrix)
        section = intersect_plane_with_convex_polyhedron(
            section_id,
            model,
            plane,
            vertex_positions=positions,
            tolerance_policy=policy,
        )
    except (SolverError, ConvexSectionSolverError) as exc:
        raise TransparentSectionCompositingError(str(exc)) from exc
    fragments = [
        *_solid_face_fragments(model, positions, plane, section, policy),
        *_plane_fragments(plane, section, policy),
    ]
    fragment_ids = [item.fragment_id for item in fragments]
    if len(fragment_ids) != len(set(fragment_ids)):
        raise TransparentSectionCompositingError(
            "transparent fragment identities are not unique"
        )
    fragments.sort(key=lambda item: item.fragment_id)
    relations = _fragment_relations(
        fragments,
        view,
        policy,
        coplanar_policy=coplanar_policy,
    )
    draw_order = _topological_draw_order(fragments, relations)
    return TransparentSectionCompositingFrame(
        section,
        view.projection_matrix,
        tuple(fragments),
        draw_order,
        relations,
    )


def canonical_transparent_section_compositing_json(
    frame: TransparentSectionCompositingFrame,
) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "FragmentOrderRelation",
    "TRANSPARENT_SECTION_COMPOSITING_SCHEMA",
    "TransparentSectionCompositingError",
    "TransparentSectionCompositingFrame",
    "TransparentTriangle",
    "canonical_transparent_section_compositing_json",
    "compute_transparent_section_compositing",
]
