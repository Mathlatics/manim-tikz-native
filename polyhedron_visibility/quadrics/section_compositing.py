"""Unified painter graph for one convex quadric and one cutting plane.

The ordinary quadric compositor intentionally treats every finite solid as one
closed two-dimensional silhouette.  That representation is sufficient for
opaque hidden-line removal, but one silhouette cannot interleave with a plane
which passes through the solid.

This module supplies the missing local-compositing stage.  A convex solid is
represented by two coincident projection sheets (far and near).  The finite
display patch is adaptively partitioned into cells which lie behind both
sheets, between them, or in front of both.  Those cells, the two smooth sheets,
depth-split plane-outline fragments, and every analytic curve fragment then
share one stable far-to-near painter graph.

The support quadric and its finite ``ray_hits`` remain geometric truth.  The
adaptive triangles are display fragments only; ambiguous boundary cells are
refined until their projected diameter is within an explicit error bound.
No renderer object is created here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import copysign, gcd, isfinite, sqrt
from typing import Callable, Sequence

import numpy as np

from ..compositor import (
    CompositorCycleError,
    PainterConstraint,
    stable_topological_sort,
)
from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from ..parallel_solver import ParallelView
from ..topology import (
    ParameterInterval,
    assert_exact_partition,
    partition_parameter_domain,
)
from .compositing import (
    QuadricCompositingFrame,
    QuadricPaintRelation,
)
from .contract import (
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from .critical import compute_curve_critical_events
from .curves import SegmentCurve
from .projection import OpaqueProjectionProxy


QUADRIC_SECTION_COMPOSITING_SCHEMA = "manim-quadric-section-compositing/v1"


QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec
ContextInput = GeometryContext | ResolvedGeometryContext | None


class QuadricSectionCompositingError(ValueError):
    """A quadric/plane painter frame cannot be certified without guessing."""


class PlaneDepthRole(str, Enum):
    """Depth class of one finite plane cell relative to a convex solid."""

    OUTSIDE_PROJECTION = "outside_projection"
    BEHIND_SURFACE = "behind_surface"
    BETWEEN_SURFACE_SHEETS = "between_surface_sheets"
    IN_FRONT_OF_SURFACE = "in_front_of_surface"


@dataclass(frozen=True, slots=True)
class QuadricSectionCompositingLimits:
    """Hard display-subdivision bounds for one section painter frame."""

    minimum_subdivision_depth: int = 0
    maximum_subdivision_depth: int = 10
    max_plane_fragments: int = 8192
    max_outline_fragments: int = 256
    max_ray_classifications: int = 65536

    def __post_init__(self) -> None:
        for name in (
            "minimum_subdivision_depth",
            "maximum_subdivision_depth",
            "max_plane_fragments",
            "max_outline_fragments",
            "max_ray_classifications",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.minimum_subdivision_depth < 0:
            raise ValueError("minimum_subdivision_depth must be non-negative")
        if self.maximum_subdivision_depth < self.minimum_subdivision_depth:
            raise ValueError(
                "maximum_subdivision_depth must not be smaller than "
                "minimum_subdivision_depth"
            )
        if (
            self.max_plane_fragments <= 0
            or self.max_outline_fragments <= 0
            or self.max_ray_classifications <= 0
        ):
            raise ValueError("section compositing capacities must be positive")


QUADRIC_SECTION_COMPOSITING_LIMITS = QuadricSectionCompositingLimits()


@dataclass(frozen=True, slots=True)
class QuadricPlaneFragment:
    """One independently classified triangle of the finite display patch."""

    fragment_id: str
    role: PlaneDepthRole
    world_vertices: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    screen_vertices: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    subdivision_depth: int

    def __post_init__(self) -> None:
        if not isinstance(self.fragment_id, str) or not self.fragment_id.strip():
            raise QuadricSectionCompositingError(
                "plane fragment_id must be a non-empty string"
            )
        if not isinstance(self.role, PlaneDepthRole):
            raise TypeError("plane fragment role must be a PlaneDepthRole")
        world = np.asarray(self.world_vertices, dtype=float)
        screen = np.asarray(self.screen_vertices, dtype=float)
        if world.shape != (3, 3) or not np.all(np.isfinite(world)):
            raise QuadricSectionCompositingError(
                "plane fragment world vertices must be a finite triangle"
            )
        if screen.shape != (3, 2) or not np.all(np.isfinite(screen)):
            raise QuadricSectionCompositingError(
                "plane fragment screen vertices must be a finite triangle"
            )
        if (
            isinstance(self.subdivision_depth, bool)
            or not isinstance(self.subdivision_depth, int)
            or self.subdivision_depth < 0
        ):
            raise QuadricSectionCompositingError(
                "plane fragment subdivision_depth must be non-negative"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "fragmentId": self.fragment_id,
            "role": self.role.value,
            "worldVertices": [list(item) for item in self.world_vertices],
            "screenVertices": [list(item) for item in self.screen_vertices],
            "subdivisionDepth": self.subdivision_depth,
        }


@dataclass(frozen=True, slots=True)
class QuadricPlaneOutlineFragment:
    """One exact depth-owned interval of a finite plane-patch edge."""

    fragment_id: str
    role: PlaneDepthRole
    edge_index: int
    interval: ParameterInterval
    world_start: tuple[float, float, float]
    world_end: tuple[float, float, float]
    screen_start: tuple[float, float]
    screen_end: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.fragment_id, str) or not self.fragment_id.strip():
            raise QuadricSectionCompositingError(
                "plane outline fragment_id must be a non-empty string"
            )
        if not isinstance(self.role, PlaneDepthRole):
            raise TypeError("plane outline fragment role must be a PlaneDepthRole")
        if (
            isinstance(self.edge_index, bool)
            or not isinstance(self.edge_index, int)
            or not 0 <= self.edge_index < 4
        ):
            raise QuadricSectionCompositingError(
                "plane outline edge_index must lie in [0, 3]"
            )
        if (
            not isinstance(self.interval, ParameterInterval)
            or self.interval.length <= 0.0
        ):
            raise QuadricSectionCompositingError(
                "plane outline interval must have positive length"
            )
        world = np.asarray((self.world_start, self.world_end), dtype=float)
        screen = np.asarray((self.screen_start, self.screen_end), dtype=float)
        if world.shape != (2, 3) or not np.all(np.isfinite(world)):
            raise QuadricSectionCompositingError(
                "plane outline world endpoints must be finite 3D points"
            )
        if screen.shape != (2, 2) or not np.all(np.isfinite(screen)):
            raise QuadricSectionCompositingError(
                "plane outline screen endpoints must be finite 2D points"
            )
        if float(np.linalg.norm(world[1] - world[0])) <= 0.0:
            raise QuadricSectionCompositingError(
                "plane outline fragment must have positive world length"
            )
        if float(np.linalg.norm(screen[1] - screen[0])) <= 0.0:
            raise QuadricSectionCompositingError(
                "plane outline fragment must have positive screen length"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "fragmentId": self.fragment_id,
            "role": self.role.value,
            "edgeIndex": self.edge_index,
            "interval": [self.interval.start, self.interval.end],
            "worldStart": list(self.world_start),
            "worldEnd": list(self.world_end),
            "screenStart": list(self.screen_start),
            "screenEnd": list(self.screen_end),
        }


@dataclass(frozen=True, slots=True)
class QuadricSectionPaintItems:
    """Stable fill, sheet, and depth-split outline painter identities."""

    plane_behind: str
    surface_back: str
    plane_outside: str
    plane_between: str
    surface_front: str
    plane_front: str
    plane_outline: str
    plane_outline_behind: str
    plane_outline_outside: str
    plane_outline_between: str

    @property
    def ordered(self) -> tuple[str, ...]:
        return (
            self.plane_behind,
            self.surface_back,
            self.plane_outside,
            self.plane_between,
            self.surface_front,
            self.plane_front,
            self.plane_outline,
            self.plane_outline_behind,
            self.plane_outline_outside,
            self.plane_outline_between,
        )

    @property
    def depth_chain(self) -> tuple[str, ...]:
        return (
            self.plane_behind,
            self.plane_outline_behind,
            self.surface_back,
            self.plane_outside,
            self.plane_outline_outside,
            self.plane_between,
            self.plane_outline_between,
            self.surface_front,
            self.plane_front,
            self.plane_outline,
        )

    @property
    def outline_by_role(self) -> dict[PlaneDepthRole, str]:
        return {
            PlaneDepthRole.BEHIND_SURFACE: self.plane_outline_behind,
            PlaneDepthRole.OUTSIDE_PROJECTION: self.plane_outline_outside,
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS: self.plane_outline_between,
            PlaneDepthRole.IN_FRONT_OF_SURFACE: self.plane_outline,
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "planeBehind": self.plane_behind,
            "surfaceBack": self.surface_back,
            "planeOutside": self.plane_outside,
            "planeBetween": self.plane_between,
            "surfaceFront": self.surface_front,
            "planeFront": self.plane_front,
            "planeOutline": self.plane_outline,
            "planeOutlineBehind": self.plane_outline_behind,
            "planeOutlineOutside": self.plane_outline_outside,
            "planeOutlineBetween": self.plane_outline_between,
        }


@dataclass(frozen=True, slots=True)
class QuadricSectionCompositingFrame:
    """Complete renderer-neutral section geometry and painter order."""

    base_frame: QuadricCompositingFrame
    surface_id: str
    plane: SectionPlane
    patch: PlaneDisplayPatchSpec
    surface_proxy: OpaqueProjectionProxy
    plane_fragments: tuple[QuadricPlaneFragment, ...]
    plane_outline_fragments: tuple[QuadricPlaneOutlineFragment, ...]
    paint_items: QuadricSectionPaintItems
    order_relations: tuple[QuadricPaintRelation, ...]
    draw_order: tuple[str, ...]
    max_screen_error: float
    ray_classification_count: int
    schema: str = QUADRIC_SECTION_COMPOSITING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUADRIC_SECTION_COMPOSITING_SCHEMA:
            raise QuadricSectionCompositingError(
                "invalid quadric-section compositing schema"
            )
        if not isinstance(self.base_frame, QuadricCompositingFrame):
            raise TypeError("base_frame must be a QuadricCompositingFrame")
        if not isinstance(self.plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")
        if not isinstance(self.patch, PlaneDisplayPatchSpec):
            raise TypeError("patch must be a PlaneDisplayPatchSpec")
        if self.patch.plane_id != self.plane.plane_id:
            raise QuadricSectionCompositingError(
                "display patch does not belong to the supplied plane"
            )
        if not isinstance(self.surface_proxy, OpaqueProjectionProxy):
            raise TypeError("surface_proxy must be an OpaqueProjectionProxy")
        if self.surface_proxy.surface_id != self.surface_id:
            raise QuadricSectionCompositingError(
                "surface proxy does not belong to the supplied surface"
            )
        fragment_ids = tuple(item.fragment_id for item in self.plane_fragments)
        if len(fragment_ids) != len(set(fragment_ids)):
            raise QuadricSectionCompositingError(
                "plane fragment identities must be unique"
            )
        if fragment_ids != tuple(sorted(fragment_ids)):
            raise QuadricSectionCompositingError(
                "plane fragments must be sorted by identity"
            )
        if not all(
            isinstance(item, QuadricPlaneOutlineFragment)
            for item in self.plane_outline_fragments
        ):
            raise TypeError(
                "plane_outline_fragments must contain QuadricPlaneOutlineFragment"
            )
        outline_ids = tuple(
            item.fragment_id for item in self.plane_outline_fragments
        )
        if outline_ids != tuple(sorted(outline_ids)) or len(set(outline_ids)) != len(
            outline_ids
        ):
            raise QuadricSectionCompositingError(
                "plane outline fragments must have unique sorted identities"
            )
        for edge_index in range(4):
            edge = tuple(
                sorted(
                    (
                        item
                        for item in self.plane_outline_fragments
                        if item.edge_index == edge_index
                    ),
                    key=lambda item: (
                        item.interval.start,
                        item.interval.end,
                        item.fragment_id,
                    ),
                )
            )
            if not edge:
                raise QuadricSectionCompositingError(
                    "every plane outline edge must retain at least one fragment"
                )
            try:
                assert_exact_partition(
                    ParameterInterval(0.0, 1.0),
                    (item.interval for item in edge),
                )
            except ValueError as exc:
                raise QuadricSectionCompositingError(
                    "plane outline fragments must exactly cover every edge"
                ) from exc
        item_ids = {
            *self.paint_items.ordered,
            *(
                item.item_id
                for item in self.base_frame.curve_fragments
                if item.painted
            ),
        }
        if len(self.draw_order) != len(set(self.draw_order)) or set(
            self.draw_order
        ) != item_ids:
            raise QuadricSectionCompositingError(
                "draw_order must cover every section painter item exactly once"
            )
        relation_keys = tuple(
            (item.far_item_id, item.near_item_id, item.reason)
            for item in self.order_relations
        )
        if relation_keys != tuple(sorted(relation_keys)):
            raise QuadricSectionCompositingError(
                "section painter relations must be sorted"
            )
        if not isfinite(self.max_screen_error) or self.max_screen_error <= 0.0:
            raise QuadricSectionCompositingError(
                "max_screen_error must be finite and positive"
            )
        if (
            isinstance(self.ray_classification_count, bool)
            or not isinstance(self.ray_classification_count, int)
            or self.ray_classification_count < 0
        ):
            raise QuadricSectionCompositingError(
                "ray_classification_count must be non-negative"
            )

    @property
    def fragments_by_role(self) -> dict[PlaneDepthRole, tuple[QuadricPlaneFragment, ...]]:
        return {
            role: tuple(item for item in self.plane_fragments if item.role is role)
            for role in PlaneDepthRole
        }

    @property
    def outline_fragments_by_role(
        self,
    ) -> dict[PlaneDepthRole, tuple[QuadricPlaneOutlineFragment, ...]]:
        return {
            role: tuple(
                item for item in self.plane_outline_fragments if item.role is role
            )
            for role in PlaneDepthRole
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "baseFrame": self.base_frame.to_dict(),
            "surfaceId": self.surface_id,
            "plane": {
                "planeId": self.plane.plane_id,
                "point": list(self.plane.point),
                "normal": list(self.plane.normal),
                "uAxis": list(self.plane.u_axis or ()),
            },
            "patch": {
                "patchId": self.patch.patch_id,
                "planeId": self.patch.plane_id,
                "halfWidth": self.patch.half_width,
                "halfHeight": self.patch.half_height,
                "centerCoordinates": list(self.patch.center_coordinates),
            },
            "surfaceProxy": self.surface_proxy.to_dict(),
            "planeFragments": [item.to_dict() for item in self.plane_fragments],
            "planeOutlineFragments": [
                item.to_dict() for item in self.plane_outline_fragments
            ],
            "paintItems": self.paint_items.to_dict(),
            "orderRelations": [item.to_dict() for item in self.order_relations],
            "drawOrder": list(self.draw_order),
            "maxScreenError": self.max_screen_error,
            "rayClassificationCount": self.ray_classification_count,
        }


def quadric_plane_fragment_contours(
    frame: QuadricSectionCompositingFrame,
) -> dict[PlaneDepthRole, tuple[tuple[tuple[float, float], ...], ...]]:
    """Merge adaptive triangles into a few renderer-friendly boundary loops.

    Every subdivision vertex lies on one dyadic lattice in the plane patch.
    Splitting coarse edges into primitive lattice segments lets shared internal
    edges cancel exactly, including T-junctions between different refinement
    depths.  The returned loop winding is preserved so Cairo can represent
    holes without drawing thousands of individual triangles.
    """

    if not isinstance(frame, QuadricSectionCompositingFrame):
        raise TypeError("frame must be a QuadricSectionCompositingFrame")
    maximum_depth = max(
        (item.subdivision_depth for item in frame.plane_fragments),
        default=0,
    )
    denominator = 1 << maximum_depth
    minimum_u = frame.patch.center_coordinates[0] - frame.patch.half_width
    minimum_v = frame.patch.center_coordinates[1] - frame.patch.half_height
    width = 2.0 * frame.patch.half_width
    height = 2.0 * frame.patch.half_height
    plane_u, plane_v, _normal = frame.plane.basis
    plane_origin = np.asarray(frame.plane.point, dtype=float)
    projection_matrix = np.asarray(
        frame.base_frame.visibility.projection_matrix,
        dtype=float,
    )
    screen_origin = projection_matrix[:2] @ np.asarray(
        frame.plane.point,
        dtype=float,
    )
    screen_basis = np.column_stack(
        (
            projection_matrix[:2] @ plane_u,
            projection_matrix[:2] @ plane_v,
        )
    )

    def lattice(point: Sequence[float]) -> tuple[int, int]:
        delta = np.asarray(point, dtype=float) - plane_origin
        u = float(np.dot(delta, plane_u))
        v = float(np.dot(delta, plane_v))
        raw_i = (u - minimum_u) * denominator / width
        raw_j = (v - minimum_v) * denominator / height
        i, j = int(round(raw_i)), int(round(raw_j))
        if abs(raw_i - i) > 1.0e-6 or abs(raw_j - j) > 1.0e-6:
            raise QuadricSectionCompositingError(
                "adaptive plane fragment escaped its dyadic patch lattice"
            )
        return i, j

    def screen(point: tuple[int, int]) -> tuple[float, float]:
        u = minimum_u + width * point[0] / denominator
        v = minimum_v + height * point[1] / denominator
        result = screen_origin + screen_basis @ np.asarray((u, v), dtype=float)
        return float(result[0]), float(result[1])

    result: dict[PlaneDepthRole, tuple[tuple[tuple[float, float], ...], ...]] = {}
    for role in PlaneDepthRole:
        edges: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for fragment in frame.plane_fragments:
            if fragment.role is not role:
                continue
            vertices = tuple(lattice(point) for point in fragment.world_vertices)
            for start, end in zip(vertices, (*vertices[1:], vertices[0])):
                delta_i = end[0] - start[0]
                delta_j = end[1] - start[1]
                steps = gcd(abs(delta_i), abs(delta_j))
                if steps == 0:
                    raise QuadricSectionCompositingError(
                        "plane fragment contains a zero-length lattice edge"
                    )
                step = (delta_i // steps, delta_j // steps)
                current = start
                for _index in range(steps):
                    following = (current[0] + step[0], current[1] + step[1])
                    reverse = (following, current)
                    if edges.get(reverse, 0):
                        edges[reverse] -= 1
                        if edges[reverse] == 0:
                            del edges[reverse]
                    else:
                        key = (current, following)
                        edges[key] = edges.get(key, 0) + 1
                    current = following
        if any(count != 1 for count in edges.values()):
            raise QuadricSectionCompositingError(
                "plane fragment boundary contains duplicate directed edges"
            )
        outgoing: dict[tuple[int, int], set[tuple[int, int]]] = {}
        incoming: dict[tuple[int, int], set[tuple[int, int]]] = {}
        remaining = set(edges)
        for start, end in remaining:
            outgoing.setdefault(start, set()).add(end)
            incoming.setdefault(end, set()).add(start)
        if set(outgoing) != set(incoming):
            raise QuadricSectionCompositingError(
                "plane fragment union produced an open boundary"
            )
        loops: list[tuple[tuple[int, int], ...]] = []
        while remaining:
            first_edge = min(remaining)
            start, current = first_edge
            remaining.remove(first_edge)
            values = [start, current]
            while current != start:
                candidates = sorted(
                    end for end in outgoing.get(current, ())
                    if (current, end) in remaining
                )
                if not candidates:
                    raise QuadricSectionCompositingError(
                        "plane fragment union produced an open contour"
                    )
                following = candidates[0]
                remaining.remove((current, following))
                values.append(following)
                current = following
                if len(values) > len(edges) + 2:
                    raise QuadricSectionCompositingError(
                        "plane fragment contour traversal did not close"
                    )
            values.pop()
            simplified: list[tuple[int, int]] = []
            for value in values:
                simplified.append(value)
                while len(simplified) >= 3:
                    first, middle, last = simplified[-3:]
                    if (
                        (middle[0] - first[0]) * (last[1] - middle[1])
                        - (middle[1] - first[1]) * (last[0] - middle[0])
                    ) != 0:
                        break
                    simplified.pop(-2)
            changed = True
            while changed and len(simplified) >= 3:
                changed = False
                first, middle, last = simplified[-2], simplified[-1], simplified[0]
                if (
                    (middle[0] - first[0]) * (last[1] - middle[1])
                    - (middle[1] - first[1]) * (last[0] - middle[0])
                ) == 0:
                    simplified.pop()
                    changed = True
                if len(simplified) < 3:
                    break
                first, middle, last = simplified[-1], simplified[0], simplified[1]
                if (
                    (middle[0] - first[0]) * (last[1] - middle[1])
                    - (middle[1] - first[1]) * (last[0] - middle[0])
                ) == 0:
                    simplified.pop(0)
                    changed = True
            if len(simplified) < 3:
                raise QuadricSectionCompositingError(
                    "plane fragment contour has no stable area"
                )
            loops.append(tuple(simplified))
        loops.sort()
        result[role] = tuple(
            tuple(screen(point) for point in loop) for loop in loops
        )
    return result


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise QuadricSectionCompositingError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricSectionCompositingError(
            f"{label} must be finite and positive"
        ) from exc
    if not isfinite(result) or result <= 0.0:
        raise QuadricSectionCompositingError(f"{label} must be finite and positive")
    return result


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _signed_area(points: Sequence[np.ndarray]) -> float:
    return 0.5 * sum(
        _cross2(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def _prepare_convex_clipper(
    clipper: Sequence[np.ndarray],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clip = [np.asarray(item, dtype=float) for item in clipper]
    if _signed_area(clip) < 0.0:
        clip.reverse()
    starts = np.asarray(clip, dtype=float)
    directions = np.asarray(
        [clip[(index + 1) % len(clip)] - clip[index] for index in range(len(clip))],
        dtype=float,
    )
    thresholds = epsilon * np.maximum(
        np.linalg.norm(directions, axis=1),
        epsilon,
    )
    return starts, directions, thresholds


def _clip_convex_polygon(
    subject: Sequence[np.ndarray],
    clipper: tuple[np.ndarray, np.ndarray, np.ndarray],
    epsilon: float,
) -> list[np.ndarray]:
    output = [np.asarray(item, dtype=float) for item in subject]
    starts, directions, thresholds = clipper
    points = np.asarray(output, dtype=float)
    values = (
        directions[:, 0, None] * (points[None, :, 1] - starts[:, None, 1])
        - directions[:, 1, None] * (points[None, :, 0] - starts[:, None, 0])
    )
    if np.any(np.max(values, axis=1) < -thresholds):
        return []
    candidate_indices = np.flatnonzero(
        np.any(values < -thresholds[:, None], axis=1)
    )
    if not len(candidate_indices):
        return output
    for index in candidate_indices:
        edge_start = starts[index]
        direction = directions[index]
        threshold = float(thresholds[index])
        if not output:
            break
        values = output
        output = []
        previous = values[-1]
        previous_value = _cross2(direction, previous - edge_start)
        previous_inside = previous_value >= -threshold
        for current in values:
            current_value = _cross2(direction, current - edge_start)
            current_inside = current_value >= -threshold
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                if abs(denominator) > max(epsilon * epsilon, threshold * 1.0e-9):
                    ratio = previous_value / denominator
                    output.append(previous + ratio * (current - previous))
            if current_inside:
                output.append(current)
            previous = current
            previous_value = current_value
            previous_inside = current_inside
    deduped: list[np.ndarray] = []
    for point in output:
        if not deduped or float(np.linalg.norm(deduped[-1] - point)) > epsilon:
            deduped.append(point)
    if (
        len(deduped) > 1
        and float(np.linalg.norm(deduped[0] - deduped[-1])) <= epsilon
    ):
        deduped.pop()
    return deduped


def _clip_polygon_by_linear_range(
    values: Sequence[np.ndarray],
    coefficients: np.ndarray,
    offset: float,
    lower: float,
    upper: float,
    epsilon: float,
) -> list[np.ndarray]:
    """Clip one plane cell to the finite entity's two cap half-spaces."""

    def axial(point: np.ndarray) -> float:
        return float(coefficients @ point + offset)

    result = [np.asarray(point, dtype=float) for point in values]
    for signed in (
        lambda point: axial(point) - lower,
        lambda point: upper - axial(point),
    ):
        if not result:
            break
        clipped: list[np.ndarray] = []
        previous = result[-1]
        previous_value = signed(previous)
        previous_inside = previous_value >= -epsilon
        for current in result:
            current_value = signed(current)
            current_inside = current_value >= -epsilon
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                if abs(denominator) > np.finfo(float).eps:
                    ratio = previous_value / denominator
                    clipped.append(previous + ratio * (current - previous))
            if current_inside:
                clipped.append(current)
            previous = current
            previous_value = current_value
            previous_inside = current_inside
        result = clipped
    deduped: list[np.ndarray] = []
    for point in result:
        if not deduped or float(np.linalg.norm(deduped[-1] - point)) > epsilon:
            deduped.append(point)
    if (
        len(deduped) > 1
        and float(np.linalg.norm(deduped[0] - deduped[-1])) <= epsilon
    ):
        deduped.pop()
    return deduped


def _point_in_convex_polygon(
    point: np.ndarray,
    polygon: Sequence[np.ndarray],
    epsilon: float,
) -> bool:
    orientation = _signed_area(polygon)
    if abs(orientation) <= epsilon * epsilon:
        return False
    sign = 1.0 if orientation > 0.0 else -1.0
    return all(
        sign
        * _cross2(
            polygon[(index + 1) % len(polygon)] - polygon[index],
            point - polygon[index],
        )
        >= -epsilon
        for index in range(len(polygon))
    )


def _minimum_quadratic_on_convex_polygon(
    conic: np.ndarray,
    polygon: Sequence[np.ndarray],
    epsilon: float,
    stationary: np.ndarray | None,
) -> float:
    """Return the exact candidate minimum of a quadratic on a convex polygon."""

    matrix = np.asarray(conic[:2, :2], dtype=float)
    linear = np.asarray(conic[:2, 2], dtype=float)
    constant = float(conic[2, 2])

    def evaluate(point: np.ndarray) -> float:
        return float(point @ matrix @ point + 2.0 * linear @ point + constant)

    candidates = [np.asarray(point, dtype=float) for point in polygon]
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        direction = end - start
        quadratic = float(direction @ matrix @ direction)
        slope = float(2.0 * (start @ matrix @ direction + linear @ direction))
        scale = max(
            abs(quadratic),
            abs(slope),
            float(np.linalg.norm(matrix, ord=2)),
            np.finfo(float).tiny,
        )
        if abs(quadratic) > np.finfo(float).eps * 128.0 * scale:
            parameter = -slope / (2.0 * quadratic)
            if 0.0 < parameter < 1.0:
                candidates.append(start + parameter * direction)

    if stationary is not None and _point_in_convex_polygon(
        stationary,
        polygon,
        epsilon,
    ):
        candidates.append(stationary)
    return min(evaluate(point) for point in candidates)


def _plane_cell_may_meet_solid(
    screen_polygon: Sequence[np.ndarray],
    inverse_screen_basis: np.ndarray,
    screen_origin: np.ndarray,
    restricted_surface: np.ndarray,
    stationary: np.ndarray | None,
    axial_mapping: tuple[np.ndarray, float, float, float] | None,
    coordinate_epsilon: float,
    implicit_epsilon: float,
) -> bool:
    """Conservatively detect a finite-solid section inside one display cell.

    This is the feature gate that prevents a small near-tangent conic from
    being missed when every coarse sample happens to lie outside it.
    """

    uv = [
        inverse_screen_basis
        @ (np.asarray(point, dtype=float) - screen_origin)
        for point in screen_polygon
    ]
    if axial_mapping is not None:
        coefficients, offset, lower, upper = axial_mapping
        uv = _clip_polygon_by_linear_range(
            uv,
            coefficients,
            offset,
            lower,
            upper,
            coordinate_epsilon,
        )
    if len(uv) < 3:
        return False
    minimum = _minimum_quadratic_on_convex_polygon(
        restricted_surface,
        uv,
        coordinate_epsilon,
        stationary,
    )
    return minimum <= implicit_epsilon


def _subdivide_triangle(world: np.ndarray) -> tuple[np.ndarray, ...]:
    first, second, third = world
    first_second = 0.5 * (first + second)
    second_third = 0.5 * (second + third)
    third_first = 0.5 * (third + first)
    return (
        np.asarray((first, first_second, third_first), dtype=float),
        np.asarray((first_second, second, second_third), dtype=float),
        np.asarray((third_first, second_third, third), dtype=float),
        np.asarray((first_second, second_third, third_first), dtype=float),
    )


def _stable_quadratic_roots(
    first: float,
    second: float,
    third: float,
    *,
    coefficient_epsilon: float,
) -> tuple[float, ...]:
    """Return display-classification roots without allocating trace objects."""

    scale = max(abs(first), abs(second), abs(third), 1.0)
    epsilon = coefficient_epsilon * scale
    if abs(first) <= epsilon:
        if abs(second) <= epsilon:
            return ()
        return (-third / second,)
    discriminant = second * second - 4.0 * first * third
    discriminant_scale = max(
        second * second,
        abs(4.0 * first * third),
        scale * scale,
        1.0,
    )
    discriminant_epsilon = coefficient_epsilon * discriminant_scale
    if discriminant < -discriminant_epsilon:
        return ()
    if abs(discriminant) <= discriminant_epsilon:
        return (-second / (2.0 * first),)
    root = sqrt(max(0.0, discriminant))
    stable = -0.5 * (second + copysign(root, second))
    if stable == 0.0:
        values = (
            (-second - root) / (2.0 * first),
            (-second + root) / (2.0 * first),
        )
    else:
        values = (stable / first, third / stable)
    return tuple(sorted(float(item) for item in values))


def _surface_ray_solver(
    surface: QuadricSurfaceSpec,
    direction: np.ndarray,
    *,
    boundary_epsilon: float,
    angular_epsilon: float,
) -> Callable[[np.ndarray], tuple[float, ...]]:
    """Precompute immutable surface data for many display-ray queries."""

    coefficient_epsilon = max(angular_epsilon, np.finfo(float).eps * 64.0)
    if isinstance(surface, SphereSpec):
        center = np.asarray(surface.center, dtype=float)
        first = float(np.dot(direction, direction))
        radius_squared = surface.radius**2

        def sphere_solver(point: np.ndarray) -> tuple[float, ...]:
            local = point - center
            return _stable_quadratic_roots(
                first,
                2.0 * float(np.dot(local, direction)),
                float(np.dot(local, local)) - radius_squared,
                coefficient_epsilon=coefficient_epsilon,
            )

        return sphere_solver

    frame = surface.frame
    origin = np.asarray(
        surface.origin if isinstance(surface, CylinderSpec) else surface.apex,
        dtype=float,
    )
    axes = np.asarray((frame.x_axis, frame.y_axis, frame.z_axis), dtype=float)
    local_direction = axes @ direction
    lower, upper = surface.axial_range
    is_cylinder = isinstance(surface, CylinderSpec)
    radius = surface.radius if is_cylinder else 0.0
    slope = 0.0 if is_cylinder else surface.slope
    if is_cylinder:
        first = float(np.dot(local_direction[:2], local_direction[:2]))
    else:
        first = float(np.dot(local_direction[:2], local_direction[:2])) - (
            slope**2 * float(local_direction[2] ** 2)
        )

    def axial_solver(point: np.ndarray) -> tuple[float, ...]:
        local_point = axes @ (point - origin)
        if is_cylinder:
            second = 2.0 * float(
                np.dot(local_point[:2], local_direction[:2])
            )
            third = float(np.dot(local_point[:2], local_point[:2])) - radius**2
        else:
            slope_squared = slope**2
            second = 2.0 * (
                float(np.dot(local_point[:2], local_direction[:2]))
                - slope_squared * float(local_point[2] * local_direction[2])
            )
            third = float(np.dot(local_point[:2], local_point[:2])) - (
                slope_squared * float(local_point[2] ** 2)
            )
        candidates = list(
            _stable_quadratic_roots(
                first,
                second,
                third,
                coefficient_epsilon=coefficient_epsilon,
            )
        )
        candidates = [
            value
            for value in candidates
            if lower - boundary_epsilon
            <= float(local_point[2] + value * local_direction[2])
            <= upper + boundary_epsilon
        ]
        if abs(float(local_direction[2])) > angular_epsilon:
            for axial in (lower, upper):
                parameter = float((axial - local_point[2]) / local_direction[2])
                radial = local_point[:2] + parameter * local_direction[:2]
                cap_radius = radius if is_cylinder else abs(axial) * slope
                if float(np.linalg.norm(radial)) <= cap_radius + boundary_epsilon:
                    candidates.append(parameter)
        result: list[float] = []
        for value in sorted(candidates):
            if not result or abs(value - result[-1]) > boundary_epsilon:
                result.append(float(value))
        return tuple(result)

    return axial_solver


def _canonical_triangle(
    world: np.ndarray,
    screen: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if _signed_area(tuple(screen)) < 0.0:
        return world[::-1].copy(), screen[::-1].copy()
    return world, screen


def _paint_items(surface_id: str, plane_id: str) -> QuadricSectionPaintItems:
    prefix = f"section-compositor:{plane_id}"
    return QuadricSectionPaintItems(
        plane_behind=f"{prefix}:plane:behind",
        surface_back=f"surface:{surface_id}:projection-sheet:back",
        plane_outside=f"{prefix}:plane:outside",
        plane_between=f"{prefix}:plane:between",
        surface_front=f"surface:{surface_id}:projection-sheet:front",
        plane_front=f"{prefix}:plane:front",
        plane_outline=f"{prefix}:plane:outline:front",
        plane_outline_behind=f"{prefix}:plane:outline:behind",
        plane_outline_outside=f"{prefix}:plane:outline:outside",
        plane_outline_between=f"{prefix}:plane:outline:between",
    )


def _classify_outline_world_point(
    point: np.ndarray,
    ray_parameters: Callable[[np.ndarray], tuple[float, ...]],
    boundary_epsilon: float,
) -> PlaneDepthRole:
    parameters = ray_parameters(point)
    if not parameters:
        return PlaneDepthRole.OUTSIDE_PROJECTION
    if min(parameters) > boundary_epsilon:
        return PlaneDepthRole.BEHIND_SURFACE
    if max(parameters) < -boundary_epsilon:
        return PlaneDepthRole.IN_FRONT_OF_SURFACE
    if min(parameters) <= boundary_epsilon and max(parameters) >= -boundary_epsilon:
        return PlaneDepthRole.BETWEEN_SURFACE_SHEETS
    raise QuadricSectionCompositingError(  # pragma: no cover
        "surface ray endpoints cannot classify a plane outline point"
    )


def _compute_outline_fragments(
    surface: QuadricSurfaceSpec,
    plane: SectionPlane,
    patch: PlaneDisplayPatchSpec,
    view: ParallelView,
    *,
    context: ContextInput,
    limits: QuadricSectionCompositingLimits,
) -> tuple[QuadricPlaneOutlineFragment, ...]:
    corners = np.asarray(patch.corners(plane), dtype=float)
    characteristic = tuple(surface.characteristic_points) + tuple(
        tuple(float(value) for value in point) for point in corners
    )
    resolved = (
        resolve_geometry_context(context)
        if isinstance(context, ResolvedGeometryContext)
        else resolve_geometry_context(context, positions=characteristic)
    )
    boundary_epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    angular_epsilon = resolved.epsilon(GeometryQuantity.ANGULAR)
    parameter_epsilon = resolved.epsilon(GeometryQuantity.PARAMETER)
    ray_parameters = _surface_ray_solver(
        surface,
        np.asarray(view.view_direction, dtype=float),
        boundary_epsilon=boundary_epsilon,
        angular_epsilon=angular_epsilon,
    )

    result: list[QuadricPlaneOutlineFragment] = []
    ends = (*corners[1:], corners[0])
    for edge_index, (start, end) in enumerate(zip(corners, ends)):
        curve = SegmentCurve(
            f"{plane.plane_id}:outline-edge:{edge_index}",
            tuple(float(value) for value in start),
            tuple(float(value) for value in end),
        )
        events = compute_curve_critical_events(
            curve,
            (surface,),
            view,
            context=resolved,
        )
        cells = partition_parameter_domain(
            curve.domain,
            (event.parameter for event in events),
            tolerance=parameter_epsilon,
        )
        classified = tuple(
            (
                cell,
                _classify_outline_world_point(
                    np.asarray(curve.point(cell.midpoint), dtype=float),
                    ray_parameters,
                    boundary_epsilon,
                ),
            )
            for cell in cells
        )
        merged: list[tuple[ParameterInterval, PlaneDepthRole]] = []
        for interval, role in classified:
            if (
                merged
                and merged[-1][1] is role
                and abs(merged[-1][0].end - interval.start) <= parameter_epsilon
            ):
                merged[-1] = (
                    ParameterInterval(merged[-1][0].start, interval.end),
                    role,
                )
            else:
                merged.append((interval, role))
        for fragment_index, (interval, role) in enumerate(merged):
            world_start = np.asarray(curve.point(interval.start), dtype=float)
            world_end = np.asarray(curve.point(interval.end), dtype=float)
            screen_start = np.asarray(view.matrix[:2] @ world_start, dtype=float)
            screen_end = np.asarray(view.matrix[:2] @ world_end, dtype=float)
            result.append(
                QuadricPlaneOutlineFragment(
                    fragment_id=(
                        f"plane:{plane.plane_id}:outline:"
                        f"edge:{edge_index:02d}:fragment:{fragment_index:03d}:"
                        f"{role.value}"
                    ),
                    role=role,
                    edge_index=edge_index,
                    interval=interval,
                    world_start=tuple(float(value) for value in world_start),
                    world_end=tuple(float(value) for value in world_end),
                    screen_start=tuple(float(value) for value in screen_start),
                    screen_end=tuple(float(value) for value in screen_end),
                )
            )
            if len(result) > limits.max_outline_fragments:
                raise QuadricSectionCompositingError(
                    "quadric section outline exceeds max_outline_fragments="
                    f"{limits.max_outline_fragments}"
                )
    return tuple(sorted(result, key=lambda item: item.fragment_id))


def _relation_chain(items: Sequence[str]) -> list[QuadricPaintRelation]:
    return [
        QuadricPaintRelation(far, near, "quadric_section_depth_layer")
        for far, near in zip(items, items[1:])
    ]


def _dedupe_relations(
    relations: Sequence[QuadricPaintRelation],
) -> tuple[QuadricPaintRelation, ...]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for relation in relations:
        key = (relation.far_item_id, relation.near_item_id)
        reverse = (key[1], key[0])
        if reverse in grouped:
            raise QuadricSectionCompositingError(
                "section painter graph contains contradictory relations: "
                f"{key[0]!r}, {key[1]!r}"
            )
        grouped.setdefault(key, set()).add(relation.reason)
    return tuple(
        QuadricPaintRelation(far, near, "+".join(sorted(reasons)))
        for (far, near), reasons in sorted(grouped.items())
    )


def compute_quadric_section_compositing(
    base_frame: QuadricCompositingFrame,
    surface: QuadricSurfaceSpec,
    plane: SectionPlane,
    patch: PlaneDisplayPatchSpec,
    view: ParallelView,
    *,
    context: ContextInput = None,
    max_screen_error: float = 0.08,
    limits: QuadricSectionCompositingLimits = QUADRIC_SECTION_COMPOSITING_LIMITS,
) -> QuadricSectionCompositingFrame:
    """Split one display patch and merge it with one quadric painter frame.

    The function supports one convex finite sphere, cylinder, or single-nappe
    cone/frustum.  Curve visibility remains owned by ``base_frame``.  This
    stage replaces its one whole-surface paint item by two smooth projection
    sheets and inserts the locally classified plane regions between them.
    """

    if not isinstance(base_frame, QuadricCompositingFrame):
        raise TypeError("base_frame must be a QuadricCompositingFrame")
    if not isinstance(surface, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    if isinstance(surface, ConeSpec):
        lower, upper = surface.axial_range
        if lower < 0.0 < upper:
            raise QuadricSectionCompositingError(
                "a double-nappe cone must be split before section compositing"
            )
    if not isinstance(plane, SectionPlane):
        raise TypeError("plane must be a SectionPlane")
    if not isinstance(patch, PlaneDisplayPatchSpec):
        raise TypeError("patch must be a PlaneDisplayPatchSpec")
    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    if not isinstance(limits, QuadricSectionCompositingLimits):
        raise TypeError("limits must be a QuadricSectionCompositingLimits")
    error = _positive(max_screen_error, "max_screen_error")
    if patch.plane_id != plane.plane_id:
        raise QuadricSectionCompositingError(
            "display patch plane_id does not match the supplied plane"
        )
    if tuple(item.surface_id for item in base_frame.surface_items) != (
        surface.surface_id,
    ):
        raise QuadricSectionCompositingError(
            "section compositing requires one matching base-frame surface"
        )
    proxy = base_frame.surface_items[0].proxy

    patch_corners = np.asarray(patch.corners(plane), dtype=float)
    characteristic = tuple(surface.characteristic_points) + tuple(
        tuple(float(value) for value in point) for point in patch_corners
    )
    resolved = (
        resolve_geometry_context(context)
        if isinstance(context, ResolvedGeometryContext)
        else resolve_geometry_context(context, positions=characteristic)
    )
    screen_epsilon = max(
        resolved.epsilon(GeometryQuantity.SCREEN),
        np.finfo(float).eps * max(1.0, float(np.max(np.abs(view.matrix[:2])))),
    )

    plane_u, plane_v, _normal = plane.basis
    screen_origin = view.matrix[:2] @ np.asarray(plane.point, dtype=float)
    screen_basis = np.column_stack(
        (view.matrix[:2] @ plane_u, view.matrix[:2] @ plane_v)
    )
    determinant = float(np.linalg.det(screen_basis))
    basis_scale = max(float(np.linalg.norm(screen_basis, ord=2)), 1.0e-300)
    if abs(determinant) <= 1.0e-12 * basis_scale * basis_scale:
        raise QuadricSectionCompositingError(
            "cutting plane projects edge-on and has no sortable display area"
        )
    inverse_screen_basis = np.linalg.inv(screen_basis)

    proxy_polygon = [
        np.asarray(point, dtype=float) for point in proxy.boundary_points
    ]
    if (
        len(proxy_polygon) > 1
        and float(np.linalg.norm(proxy_polygon[0] - proxy_polygon[-1]))
        <= screen_epsilon
    ):
        proxy_polygon.pop()
    if len(proxy_polygon) < 3 or abs(_signed_area(proxy_polygon)) <= screen_epsilon**2:
        raise QuadricSectionCompositingError(
            "quadric projection proxy has no stable display area"
        )
    if _signed_area(proxy_polygon) < 0.0:
        proxy_polygon.reverse()
    proxy_clipper = _prepare_convex_clipper(proxy_polygon, screen_epsilon)
    classification_cache: dict[tuple[float, float], PlaneDepthRole] = {}
    classification_count = 0
    ray_direction = np.asarray(view.view_direction, dtype=float)
    boundary_epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    angular_epsilon = resolved.epsilon(GeometryQuantity.ANGULAR)
    ray_parameters = _surface_ray_solver(
        surface,
        ray_direction,
        boundary_epsilon=boundary_epsilon,
        angular_epsilon=angular_epsilon,
    )
    restricted_surface = plane.restrict(surface.support_quadric)
    restricted_matrix = np.asarray(restricted_surface[:2, :2], dtype=float)
    restricted_linear = np.asarray(restricted_surface[:2, 2], dtype=float)
    restricted_singular_values = np.linalg.svd(
        restricted_matrix,
        compute_uv=False,
    )
    stationary: np.ndarray | None = None
    if (
        restricted_singular_values[0] > 0.0
        and restricted_singular_values[-1]
        > np.finfo(float).eps
        * 128.0
        * restricted_singular_values[0]
    ):
        stationary = np.linalg.solve(restricted_matrix, -restricted_linear)
    screen_singular_values = np.linalg.svd(screen_basis, compute_uv=False)
    coordinate_epsilon = screen_epsilon / max(
        float(screen_singular_values[-1]),
        np.finfo(float).tiny,
    )
    axial_mapping: tuple[np.ndarray, float, float, float] | None = None
    if isinstance(surface, (CylinderSpec, ConeSpec)):
        axial_origin = np.asarray(
            surface.origin if isinstance(surface, CylinderSpec) else surface.apex,
            dtype=float,
        )
        axial_axis = np.asarray(surface.axis, dtype=float)
        axial_mapping = (
            np.asarray(
                (
                    float(np.dot(plane_u, axial_axis)),
                    float(np.dot(plane_v, axial_axis)),
                ),
                dtype=float,
            ),
            float(
                np.dot(
                    np.asarray(plane.point, dtype=float) - axial_origin,
                    axial_axis,
                )
            ),
            surface.axial_range[0],
            surface.axial_range[1],
        )
    if isinstance(surface, (SphereSpec, CylinderSpec)):
        maximum_radius = surface.radius
    else:
        maximum_radius = (
            max(abs(item) for item in surface.axial_range) * surface.slope
        )
    implicit_epsilon = (
        2.0 * maximum_radius * boundary_epsilon
        + boundary_epsilon * boundary_epsilon
    )

    def world_at_screen(screen_point: np.ndarray) -> np.ndarray:
        coordinates = inverse_screen_basis @ (screen_point - screen_origin)
        return (
            np.asarray(plane.point, dtype=float)
            + coordinates[0] * plane_u
            + coordinates[1] * plane_v
        )

    def classify(screen_point: np.ndarray) -> PlaneDepthRole:
        nonlocal classification_count
        key = (float(screen_point[0]), float(screen_point[1]))
        cached = classification_cache.get(key)
        if cached is not None:
            return cached
        classification_count += 1
        if classification_count > limits.max_ray_classifications:
            raise QuadricSectionCompositingError(
                "quadric section needs more than "
                f"{limits.max_ray_classifications} ray classifications"
            )
        world = world_at_screen(screen_point)
        parameters = ray_parameters(world)
        if not parameters:
            role = PlaneDepthRole.OUTSIDE_PROJECTION
        elif min(parameters) > boundary_epsilon:
            role = PlaneDepthRole.BEHIND_SURFACE
        elif max(parameters) < -boundary_epsilon:
            role = PlaneDepthRole.IN_FRONT_OF_SURFACE
        elif min(parameters) <= boundary_epsilon and max(parameters) >= -boundary_epsilon:
            role = PlaneDepthRole.BETWEEN_SURFACE_SHEETS
        else:
            raise QuadricSectionCompositingError(  # pragma: no cover
                "surface ray endpoints cannot be classified relative to the plane"
            )
        classification_cache[key] = role
        return role

    fragments: list[QuadricPlaneFragment] = []

    def append_leaf(world: np.ndarray, path: str, depth: int, role: PlaneDepthRole) -> None:
        screen = np.asarray(world @ view.matrix[:2].T, dtype=float)
        world_ordered, screen_ordered = _canonical_triangle(world, screen)
        fragments.append(
            QuadricPlaneFragment(
                fragment_id=f"plane:{plane.plane_id}:cell:{path}",
                role=role,
                world_vertices=tuple(
                    tuple(float(value) for value in point)
                    for point in world_ordered
                ),  # type: ignore[arg-type]
                screen_vertices=tuple(
                    tuple(float(value) for value in point)
                    for point in screen_ordered
                ),  # type: ignore[arg-type]
                subdivision_depth=depth,
            )
        )
        if len(fragments) > limits.max_plane_fragments:
            raise QuadricSectionCompositingError(
                "quadric section needs more than "
                f"{limits.max_plane_fragments} plane fragments"
            )

    def visit(world: np.ndarray, path: str, depth: int) -> None:
        screen = np.asarray(world @ view.matrix[:2].T, dtype=float)
        overlap = _clip_convex_polygon(screen, proxy_clipper, screen_epsilon)
        if len(overlap) < 3 or abs(_signed_area(overlap)) <= screen_epsilon**2:
            if depth < limits.minimum_subdivision_depth:
                for index, child in enumerate(_subdivide_triangle(world)):
                    visit(child, f"{path}.{index}", depth + 1)
            else:
                append_leaf(
                    world,
                    path,
                    depth,
                    PlaneDepthRole.OUTSIDE_PROJECTION,
                )
            return

        samples = [*overlap]
        samples.append(np.mean(overlap, axis=0))
        roles = {classify(np.asarray(point, dtype=float)) for point in samples}
        projected_diameter = max(
            float(np.linalg.norm(screen[first] - screen[second]))
            for first in range(3)
            for second in range(first + 1, 3)
        )
        uniform = len(roles) == 1
        if uniform and next(iter(roles)) is not PlaneDepthRole.BETWEEN_SURFACE_SHEETS:
            uniform = not _plane_cell_may_meet_solid(
                overlap,
                inverse_screen_basis,
                screen_origin,
                restricted_surface,
                stationary,
                axial_mapping,
                coordinate_epsilon,
                implicit_epsilon,
            )
        if depth >= limits.minimum_subdivision_depth and uniform:
            append_leaf(world, path, depth, next(iter(roles)))
            return
        if projected_diameter <= error:
            append_leaf(world, path, depth, classify(np.mean(overlap, axis=0)))
            return
        if depth >= limits.maximum_subdivision_depth:
            raise QuadricSectionCompositingError(
                "quadric section boundary needs more than "
                f"{limits.maximum_subdivision_depth} subdivision levels to meet "
                f"max_screen_error={error:.9g}"
            )
        for index, child in enumerate(_subdivide_triangle(world)):
            visit(child, f"{path}.{index}", depth + 1)

    for root_index, indices in enumerate(((0, 1, 2), (0, 2, 3))):
        visit(patch_corners[list(indices)], str(root_index), 0)

    fragments.sort(key=lambda item: item.fragment_id)
    outline_fragments = _compute_outline_fragments(
        surface,
        plane,
        patch,
        view,
        context=resolved,
        limits=limits,
    )
    items = _paint_items(surface.surface_id, plane.plane_id)
    relations = _relation_chain(items.depth_chain)
    active_curve_ids = {
        item.item_id for item in base_frame.curve_fragments if item.painted
    }
    relations.extend(
        relation
        for relation in base_frame.order_relations
        if relation.far_item_id in active_curve_ids
        and relation.near_item_id in active_curve_ids
    )
    relations.extend(
        QuadricPaintRelation(
            items.plane_outline,
            curve_id,
            "section_curve_overlay",
        )
        for curve_id in sorted(active_curve_ids)
    )
    normalized = _dedupe_relations(relations)
    active_ids = (*items.ordered, *sorted(active_curve_ids))
    try:
        draw_order = stable_topological_sort(
            active_ids,
            (
                PainterConstraint(item.far_item_id, item.near_item_id)
                for item in normalized
            ),
            key=lambda item_id: item_id,
        )
    except CompositorCycleError as exc:
        raise QuadricSectionCompositingError(
            "quadric section painter graph contains a cycle: "
            + ", ".join(sorted(str(item) for item in exc.unresolved))
        ) from exc
    return QuadricSectionCompositingFrame(
        base_frame=base_frame,
        surface_id=surface.surface_id,
        plane=plane,
        patch=patch,
        surface_proxy=proxy,
        plane_fragments=tuple(fragments),
        plane_outline_fragments=outline_fragments,
        paint_items=items,
        order_relations=normalized,
        draw_order=draw_order,
        max_screen_error=error,
        ray_classification_count=classification_count,
    )


def canonical_quadric_section_compositing_json(
    frame: QuadricSectionCompositingFrame,
) -> str:
    if not isinstance(frame, QuadricSectionCompositingFrame):
        raise TypeError("frame must be a QuadricSectionCompositingFrame")
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "PlaneDepthRole",
    "QUADRIC_SECTION_COMPOSITING_LIMITS",
    "QUADRIC_SECTION_COMPOSITING_SCHEMA",
    "QuadricPlaneFragment",
    "QuadricPlaneOutlineFragment",
    "QuadricSectionCompositingError",
    "QuadricSectionCompositingFrame",
    "QuadricSectionCompositingLimits",
    "QuadricSectionPaintItems",
    "canonical_quadric_section_compositing_json",
    "compute_quadric_section_compositing",
    "quadric_plane_fragment_contours",
]
