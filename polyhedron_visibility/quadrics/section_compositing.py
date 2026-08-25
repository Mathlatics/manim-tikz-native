"""Unified painter graph for one convex quadric and one cutting plane.

The ordinary quadric compositor intentionally treats every finite solid as one
closed two-dimensional silhouette.  That representation is sufficient for
opaque hidden-line removal, but one silhouette cannot interleave with a plane
which passes through the solid.

This module supplies the missing local-compositing stage.  A convex solid is
represented by two coincident projection sheets (far and near).  Its finite
section and projection silhouette split the display patch into polygons which
lie behind both sheets, between them, in front of both, or outside the
projection.  Those polygons, the two smooth sheets, depth-split plane-outline
fragments, and every analytic curve fragment then share one stable far-to-near
painter graph.

The finite surface ray solver remains geometric truth.  Analytic section curves
and cap boundaries locate role transitions; canonical polygon clipping and
triangulation turn them into deterministic display fragments.  Any unresolved
mixed region fails closed instead of inheriting a centre-point role.  No
renderer object is created here.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import atan2, copysign, cos, floor, isfinite, sin, sqrt, tau
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
from .critical import CriticalEventError, compute_curve_critical_events
from .curves import ParametricConicBranch, SegmentCurve
from .projection import OpaqueProjectionProxy
from .sections import QuadricSectionError, compute_quadric_section
from .trace import section_trace_curves


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
    max_plane_fragments: int = 65536
    max_outline_fragments: int = 256
    max_ray_classifications: int = 2097152

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
    """Merge plane fragments into deterministic renderer-friendly loops.

    Fragment vertices are canonicalized in plane coordinates before directed
    edges are cancelled.  Residual edges are noded at arbitrary collinear
    vertices, so clipped points and T-junctions no longer need to belong to the
    old dyadic refinement lattice.  Winding is preserved for holes.
    """

    if not isinstance(frame, QuadricSectionCompositingFrame):
        raise TypeError("frame must be a QuadricSectionCompositingFrame")
    plane_u, plane_v, _normal = frame.plane.basis
    plane_origin = np.asarray(frame.plane.point, dtype=float)
    projection_matrix = np.asarray(
        frame.base_frame.visibility.projection_matrix,
        dtype=float,
    )
    screen_origin = projection_matrix[:2] @ plane_origin
    screen_basis = np.column_stack(
        (
            projection_matrix[:2] @ plane_u,
            projection_matrix[:2] @ plane_v,
        )
    )
    coordinate_scale = max(
        abs(frame.patch.center_coordinates[0]) + frame.patch.half_width,
        abs(frame.patch.center_coordinates[1]) + frame.patch.half_height,
        np.finfo(float).tiny,
    )
    coordinate_epsilon = max(
        np.finfo(float).eps * 4096.0 * coordinate_scale,
        np.finfo(float).tiny,
    )
    registry = _CanonicalVertexRegistry(
        plane_origin=plane_origin,
        plane_u=plane_u,
        plane_v=plane_v,
        screen_origin=screen_origin,
        screen_basis=screen_basis,
        coordinate_epsilon=coordinate_epsilon,
    )
    polygons_by_role: dict[PlaneDepthRole, list[_PlanePartitionPolygon]] = {
        role: [] for role in PlaneDepthRole
    }
    for fragment in frame.plane_fragments:
        vertices = []
        for point in fragment.world_vertices:
            delta = np.asarray(point, dtype=float) - plane_origin
            vertices.append(
                registry.register(
                    (
                        float(np.dot(delta, plane_u)),
                        float(np.dot(delta, plane_v)),
                    )
                )
            )
        polygon = _make_plane_partition_polygon(
            fragment.fragment_id,
            vertices,
            coordinate_epsilon,
        )
        if polygon is None:
            raise QuadricSectionCompositingError(
                f"plane fragment {fragment.fragment_id!r} has no stable area"
            )
        polygons_by_role[fragment.role].append(polygon)

    result: dict[
        PlaneDepthRole,
        tuple[tuple[tuple[float, float], ...], ...],
    ] = {}
    for role in PlaneDepthRole:
        loops = _plane_partition_polygon_contours(
            polygons_by_role[role],
            coordinate_epsilon,
        )
        result[role] = tuple(
            tuple(vertex.screen_point for vertex in loop) for loop in loops
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


@dataclass(frozen=True, slots=True)
class _PlanePartitionVertex:
    """One private, canonical vertex shared by all plane partition pieces."""

    stable_token: str
    plane_coordinates: tuple[float, float]
    world_point: tuple[float, float, float]
    screen_point: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.stable_token, str) or not self.stable_token:
            raise QuadricSectionCompositingError(
                "plane partition vertex token must be non-empty"
            )
        if len(self.plane_coordinates) != 2 or not all(
            isfinite(value) for value in self.plane_coordinates
        ):
            raise QuadricSectionCompositingError(
                "plane partition coordinates must contain two finite values"
            )
        if len(self.world_point) != 3 or not all(
            isfinite(value) for value in self.world_point
        ):
            raise QuadricSectionCompositingError(
                "plane partition world point must contain three finite values"
            )
        if len(self.screen_point) != 2 or not all(
            isfinite(value) for value in self.screen_point
        ):
            raise QuadricSectionCompositingError(
                "plane partition screen point must contain two finite values"
            )


@dataclass(frozen=True, slots=True)
class _PlanePartitionPolygon:
    """One private convex polygon with stable vertex and polygon identity."""

    stable_token: str
    vertices: tuple[_PlanePartitionVertex, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stable_token, str) or not self.stable_token:
            raise QuadricSectionCompositingError(
                "plane partition polygon token must be non-empty"
            )
        if len(self.vertices) < 3 or not all(
            isinstance(item, _PlanePartitionVertex) for item in self.vertices
        ):
            raise QuadricSectionCompositingError(
                "plane partition polygon must contain at least three vertices"
            )
        tokens = tuple(item.stable_token for item in self.vertices)
        if len(tokens) != len(set(tokens)):
            raise QuadricSectionCompositingError(
                "plane partition polygon vertices must be unique"
            )


@dataclass(frozen=True, slots=True)
class _PlanePartitionHalfPlane:
    """One normalized supporting inequality ``normal dot uv <= offset``."""

    stable_token: str
    normal: tuple[float, float]
    offset: float


class _CanonicalVertexRegistry:
    """Create deterministic plane/world/screen vertices on one tolerance grid."""

    __slots__ = (
        "_coordinate_epsilon",
        "_grid_step",
        "_plane_origin",
        "_plane_u",
        "_plane_v",
        "_screen_origin",
        "_screen_basis",
        "_vertices",
    )

    def __init__(
        self,
        *,
        plane_origin: Sequence[float],
        plane_u: Sequence[float],
        plane_v: Sequence[float],
        screen_origin: Sequence[float],
        screen_basis: Sequence[Sequence[float]],
        coordinate_epsilon: float,
    ) -> None:
        epsilon = float(coordinate_epsilon)
        if not isfinite(epsilon) or epsilon <= 0.0:
            raise QuadricSectionCompositingError(
                "canonical vertex epsilon must be finite and positive"
            )
        origin = np.asarray(plane_origin, dtype=float)
        plane_first = np.asarray(plane_u, dtype=float)
        plane_second = np.asarray(plane_v, dtype=float)
        projected_origin = np.asarray(screen_origin, dtype=float)
        projected_basis = np.asarray(screen_basis, dtype=float)
        if (
            origin.shape != (3,)
            or plane_first.shape != (3,)
            or plane_second.shape != (3,)
            or projected_origin.shape != (2,)
            or projected_basis.shape != (2, 2)
            or not all(
                np.all(np.isfinite(value))
                for value in (
                    origin,
                    plane_first,
                    plane_second,
                    projected_origin,
                    projected_basis,
                )
            )
        ):
            raise QuadricSectionCompositingError(
                "canonical vertex registry requires finite plane projection data"
            )
        if (
            float(np.linalg.norm(plane_first)) <= epsilon
            or float(np.linalg.norm(plane_second)) <= epsilon
        ):
            raise QuadricSectionCompositingError(
                "canonical vertex plane basis must be non-degenerate"
            )
        self._coordinate_epsilon = epsilon
        self._grid_step = epsilon / sqrt(2.0)
        self._plane_origin = origin
        self._plane_u = plane_first
        self._plane_v = plane_second
        self._screen_origin = projected_origin
        self._screen_basis = projected_basis
        self._vertices: dict[tuple[int, int], _PlanePartitionVertex] = {}

    @property
    def coordinate_epsilon(self) -> float:
        return self._coordinate_epsilon

    def register(
        self,
        plane_coordinates: Sequence[float],
    ) -> _PlanePartitionVertex:
        try:
            coordinates = tuple(float(value) for value in plane_coordinates)
        except (TypeError, ValueError, OverflowError) as exc:
            raise QuadricSectionCompositingError(
                "canonical plane vertex must contain two finite values"
            ) from exc
        if len(coordinates) != 2 or not all(isfinite(value) for value in coordinates):
            raise QuadricSectionCompositingError(
                "canonical plane vertex must contain two finite values"
            )
        key = tuple(
            int(round(float(value) / self._grid_step)) for value in coordinates
        )
        candidates: list[_PlanePartitionVertex] = []
        for first_offset in (-1, 0, 1):
            for second_offset in (-1, 0, 1):
                candidate = self._vertices.get(
                    (key[0] + first_offset, key[1] + second_offset)
                )
                if candidate is None:
                    continue
                delta_u = candidate.plane_coordinates[0] - coordinates[0]
                delta_v = candidate.plane_coordinates[1] - coordinates[1]
                if (
                    delta_u * delta_u + delta_v * delta_v
                    <= self._coordinate_epsilon * self._coordinate_epsilon
                ):
                    candidates.append(candidate)
        if candidates:
            return min(candidates, key=lambda item: item.stable_token)

        canonical = np.asarray(coordinates, dtype=float)
        world = (
            self._plane_origin
            + canonical[0] * self._plane_u
            + canonical[1] * self._plane_v
        )
        screen = self._screen_origin + self._screen_basis @ canonical
        vertex = _PlanePartitionVertex(
            stable_token=f"vertex:{key[0]}:{key[1]}",
            plane_coordinates=(float(canonical[0]), float(canonical[1])),
            world_point=tuple(float(value) for value in world),  # type: ignore[arg-type]
            screen_point=tuple(float(value) for value in screen),  # type: ignore[arg-type]
        )
        self._vertices[key] = vertex
        return vertex

    def interpolate(
        self,
        first: _PlanePartitionVertex,
        second: _PlanePartitionVertex,
        ratio: float,
    ) -> _PlanePartitionVertex:
        if not isinstance(first, _PlanePartitionVertex) or not isinstance(
            second, _PlanePartitionVertex
        ):
            raise TypeError("canonical interpolation requires partition vertices")
        value = float(ratio)
        if not isfinite(value) or not -self._coordinate_epsilon <= value <= (
            1.0 + self._coordinate_epsilon
        ):
            raise QuadricSectionCompositingError(
                "canonical intersection ratio lies outside its source edge"
            )
        value = min(1.0, max(0.0, value))
        first_coordinates = np.asarray(first.plane_coordinates, dtype=float)
        second_coordinates = np.asarray(second.plane_coordinates, dtype=float)
        return self.register(
            first_coordinates + value * (second_coordinates - first_coordinates)
        )


def _partition_signed_area(vertices: Sequence[_PlanePartitionVertex]) -> float:
    return 0.5 * sum(
        vertices[index].plane_coordinates[0]
        * vertices[(index + 1) % len(vertices)].plane_coordinates[1]
        - vertices[index].plane_coordinates[1]
        * vertices[(index + 1) % len(vertices)].plane_coordinates[0]
        for index in range(len(vertices))
    )


def _partition_vertex_is_collinear(
    first: _PlanePartitionVertex,
    middle: _PlanePartitionVertex,
    last: _PlanePartitionVertex,
    epsilon: float,
) -> bool:
    first_u, first_v = first.plane_coordinates
    middle_u, middle_v = middle.plane_coordinates
    last_u, last_v = last.plane_coordinates
    span_u = last_u - first_u
    span_v = last_v - first_v
    middle_delta_u = middle_u - first_u
    middle_delta_v = middle_v - first_v
    length_squared = span_u * span_u + span_v * span_v
    length = sqrt(length_squared)
    if length <= epsilon:
        return True
    distance = abs(span_u * middle_delta_v - span_v * middle_delta_u) / length
    if distance > epsilon:
        return False
    parameter = (
        middle_delta_u * span_u + middle_delta_v * span_v
    ) / length_squared
    return -epsilon / length <= parameter <= 1.0 + epsilon / length


def _make_plane_partition_polygon(
    stable_token: str,
    vertices: Sequence[_PlanePartitionVertex],
    coordinate_epsilon: float,
) -> _PlanePartitionPolygon | None:
    """Normalize one convex polygon to fixed winding and starting vertex."""

    values: list[_PlanePartitionVertex] = []
    for vertex in vertices:
        if not isinstance(vertex, _PlanePartitionVertex):
            raise TypeError("plane partition polygons require canonical vertices")
        if not values or values[-1].stable_token != vertex.stable_token:
            values.append(vertex)
    if len(values) > 1 and values[0].stable_token == values[-1].stable_token:
        values.pop()
    changed = True
    while changed and len(values) >= 3:
        changed = False
        for index in range(len(values)):
            if _partition_vertex_is_collinear(
                values[index - 1],
                values[index],
                values[(index + 1) % len(values)],
                coordinate_epsilon,
            ):
                values.pop(index)
                changed = True
                break
    if len(values) < 3:
        return None
    area = _partition_signed_area(values)
    area_epsilon = coordinate_epsilon * coordinate_epsilon
    if abs(area) <= area_epsilon:
        return None
    if area < 0.0:
        values.reverse()
    start_index = min(
        range(len(values)),
        key=lambda index: (
            values[index].stable_token,
            values[index].plane_coordinates,
        ),
    )
    ordered = values[start_index:] + values[:start_index]
    return _PlanePartitionPolygon(stable_token, tuple(ordered))


def _intersect_convex_support_half_planes(
    stable_token: str,
    constraints: Sequence[_PlanePartitionHalfPlane],
    registry: _CanonicalVertexRegistry,
    coordinate_epsilon: float,
) -> _PlanePartitionPolygon | None:
    """Build a deterministic convex polygon from ordered support inequalities."""

    normalized: list[tuple[float, _PlanePartitionHalfPlane]] = []
    for constraint in constraints:
        normal = np.asarray(constraint.normal, dtype=float)
        length = float(np.linalg.norm(normal))
        if not isfinite(length) or length <= coordinate_epsilon:
            raise QuadricSectionCompositingError(
                "plane partition support normal must be finite and non-zero"
            )
        unit = normal / length
        offset = float(constraint.offset) / length
        if not isfinite(offset):
            raise QuadricSectionCompositingError(
                "plane partition support offset must be finite"
            )
        angle = atan2(float(unit[1]), float(unit[0]))
        normalized.append(
            (
                angle,
                _PlanePartitionHalfPlane(
                    constraint.stable_token,
                    (float(unit[0]), float(unit[1])),
                    offset,
                ),
            )
        )
    normalized.sort(key=lambda item: (item[0], item[1].stable_token))
    if len(normalized) < 3:
        return None

    angular_threshold = max(coordinate_epsilon * 8.0, 1.0e-14)
    ordered: list[_PlanePartitionHalfPlane] = []
    for _angle, constraint in normalized:
        if ordered:
            previous_normal = np.asarray(ordered[-1].normal, dtype=float)
            current_normal = np.asarray(constraint.normal, dtype=float)
            if (
                abs(_cross2(previous_normal, current_normal))
                <= angular_threshold
                and float(np.dot(previous_normal, current_normal)) > 0.0
            ):
                if (constraint.offset, constraint.stable_token) < (
                    ordered[-1].offset,
                    ordered[-1].stable_token,
                ):
                    ordered[-1] = constraint
                continue
        ordered.append(constraint)
    if len(ordered) > 1:
        first_normal = np.asarray(ordered[0].normal, dtype=float)
        last_normal = np.asarray(ordered[-1].normal, dtype=float)
        if (
            abs(_cross2(last_normal, first_normal)) <= angular_threshold
            and float(np.dot(last_normal, first_normal)) > 0.0
        ):
            preferred = min(
                (ordered[-1], ordered[0]),
                key=lambda item: (item.offset, item.stable_token),
            )
            ordered[0] = preferred
            ordered.pop()
    if len(ordered) < 3:
        return None

    def intersection(
        first: _PlanePartitionHalfPlane,
        second: _PlanePartitionHalfPlane,
    ) -> np.ndarray:
        matrix = np.asarray((first.normal, second.normal), dtype=float)
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= angular_threshold:
            raise QuadricSectionCompositingError(
                "convex section support lines do not form a bounded corner"
            )
        return np.linalg.solve(
            matrix,
            np.asarray((first.offset, second.offset), dtype=float),
        )

    def excludes(
        constraint: _PlanePartitionHalfPlane,
        point: np.ndarray,
    ) -> bool:
        return float(np.dot(constraint.normal, point)) > (
            constraint.offset + coordinate_epsilon * 8.0
        )

    active: deque[_PlanePartitionHalfPlane] = deque()
    for constraint in ordered:
        while len(active) >= 2 and excludes(
            constraint,
            intersection(active[-2], active[-1]),
        ):
            active.pop()
        while len(active) >= 2 and excludes(
            constraint,
            intersection(active[0], active[1]),
        ):
            active.popleft()
        active.append(constraint)
    while len(active) >= 3 and excludes(
        active[0],
        intersection(active[-2], active[-1]),
    ):
        active.pop()
    while len(active) >= 3 and excludes(
        active[-1],
        intersection(active[0], active[1]),
    ):
        active.popleft()
    if len(active) < 3:
        return None

    lines = tuple(active)
    vertices = tuple(
        registry.register(intersection(lines[index - 1], lines[index]))
        for index in range(len(lines))
    )
    polygon = _make_plane_partition_polygon(
        stable_token,
        vertices,
        coordinate_epsilon,
    )
    if polygon is None:
        return None
    centroid = np.mean(
        np.asarray(
            tuple(vertex.plane_coordinates for vertex in polygon.vertices),
            dtype=float,
        ),
        axis=0,
    )
    for constraint in lines:
        if excludes(constraint, centroid):
            raise QuadricSectionCompositingError(
                "convex section support intersection is inconsistent"
            )
    return polygon


def _split_convex_polygon_by_half_plane(
    polygon: _PlanePartitionPolygon,
    boundary_start: _PlanePartitionVertex,
    boundary_end: _PlanePartitionVertex,
    registry: _CanonicalVertexRegistry,
    coordinate_epsilon: float,
    *,
    boundary_token: str | None = None,
) -> tuple[_PlanePartitionPolygon | None, _PlanePartitionPolygon | None]:
    """Split one convex polygon into the left and right of a directed line."""

    if not isinstance(polygon, _PlanePartitionPolygon):
        raise TypeError("polygon must be a _PlanePartitionPolygon")
    if not isinstance(registry, _CanonicalVertexRegistry):
        raise TypeError("registry must be a _CanonicalVertexRegistry")
    start = np.asarray(boundary_start.plane_coordinates, dtype=float)
    end = np.asarray(boundary_end.plane_coordinates, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= coordinate_epsilon:
        raise QuadricSectionCompositingError(
            "plane partition boundary must have positive length"
        )
    threshold = coordinate_epsilon * length

    def signed(vertex: _PlanePartitionVertex) -> float:
        return _cross2(
            direction,
            np.asarray(vertex.plane_coordinates, dtype=float) - start,
        )

    def side(value: float) -> int:
        if value > threshold:
            return 1
        if value < -threshold:
            return -1
        return 0

    inside: list[_PlanePartitionVertex] = []
    outside: list[_PlanePartitionVertex] = []
    vertices = polygon.vertices
    for index, current in enumerate(vertices):
        following = vertices[(index + 1) % len(vertices)]
        current_value = signed(current)
        following_value = signed(following)
        current_side = side(current_value)
        following_side = side(following_value)
        if current_side >= 0:
            inside.append(current)
        if current_side <= 0:
            outside.append(current)
        if current_side * following_side < 0:
            denominator = current_value - following_value
            if abs(denominator) <= np.finfo(float).tiny:
                raise QuadricSectionCompositingError(
                    "plane partition crossing has no stable intersection"
                )
            intersection = registry.interpolate(
                current,
                following,
                current_value / denominator,
            )
            inside.append(intersection)
            outside.append(intersection)

    token = boundary_token or (
        f"{boundary_start.stable_token}->{boundary_end.stable_token}"
    )

    def child_token(side_name: str) -> str:
        digest = sha256(
            f"{polygon.stable_token}|{token}:{side_name}".encode("utf-8")
        ).hexdigest()[:24]
        return f"partition:{digest}:{side_name}"

    return (
        _make_plane_partition_polygon(
            child_token("inside"),
            inside,
            coordinate_epsilon,
        ),
        _make_plane_partition_polygon(
            child_token("outside"),
            outside,
            coordinate_epsilon,
        ),
    )


def _partition_triangle_by_convex_proxy(
    triangle: _PlanePartitionPolygon,
    proxy: _PlanePartitionPolygon,
    registry: _CanonicalVertexRegistry,
    coordinate_epsilon: float,
) -> tuple[
    tuple[_PlanePartitionPolygon, ...],
    tuple[_PlanePartitionPolygon, ...],
]:
    """Return the complete inside and outside partition of one triangle."""

    if len(triangle.vertices) != 3:
        raise QuadricSectionCompositingError(
            "convex proxy partition input must be a triangle"
        )
    return _partition_convex_polygon_by_convex_boundary(
        triangle,
        proxy,
        registry,
        coordinate_epsilon,
    )


def _partition_convex_polygon_by_convex_boundary(
    polygon: _PlanePartitionPolygon,
    boundary: _PlanePartitionPolygon,
    registry: _CanonicalVertexRegistry,
    coordinate_epsilon: float,
) -> tuple[
    tuple[_PlanePartitionPolygon, ...],
    tuple[_PlanePartitionPolygon, ...],
]:
    """Return a complete inside/outside partition for two convex polygons."""

    current: _PlanePartitionPolygon | None = polygon
    outside: list[_PlanePartitionPolygon] = []
    for edge_index, boundary_start in enumerate(boundary.vertices):
        if current is None:
            break
        boundary_end = boundary.vertices[
            (edge_index + 1) % len(boundary.vertices)
        ]
        current, rejected = _split_convex_polygon_by_half_plane(
            current,
            boundary_start,
            boundary_end,
            registry,
            coordinate_epsilon,
            boundary_token=f"proxy-edge:{edge_index:04d}",
        )
        if rejected is not None:
            outside.append(rejected)
    inside = () if current is None else (current,)
    return inside, tuple(outside)


def _nested_convex_ring_polygons(
    inner: _PlanePartitionPolygon,
    outer: _PlanePartitionPolygon,
    registry: _CanonicalVertexRegistry,
    coordinate_epsilon: float,
    *,
    minimum_screen_triangle_altitude: float | None = None,
) -> tuple[_PlanePartitionPolygon, ...]:
    """Partition the complete region between two nested convex polygons.

    Nearly coincident angular events from opposite boundaries are represented
    by one shared radial edge.  Both original boundary vertices are retained,
    so this removes an unstable microscopic ring cell without changing either
    polygon's boundary.
    """

    center = np.mean(
        np.asarray(
            tuple(vertex.plane_coordinates for vertex in inner.vertices),
            dtype=float,
        ),
        axis=0,
    )
    maximum_radius = max(
        float(
            np.linalg.norm(
                np.asarray(vertex.plane_coordinates, dtype=float) - center
            )
        )
        for vertex in (*inner.vertices, *outer.vertices)
    )
    if maximum_radius <= coordinate_epsilon:
        return ()
    angular_epsilon = coordinate_epsilon / maximum_radius

    raw_angles = sorted(
        (
            atan2(
                vertex.plane_coordinates[1] - center[1],
                vertex.plane_coordinates[0] - center[0],
            )
            % tau,
            kind,
            vertex,
        )
        for kind, polygon in (("inner", inner), ("outer", outer))
        for vertex in polygon.vertices
    )
    angle_groups: list[
        tuple[float, dict[str, _PlanePartitionVertex]]
    ] = []
    for angle, kind, vertex in raw_angles:
        if not angle_groups or angle - angle_groups[-1][0] > angular_epsilon:
            angle_groups.append((angle, {kind: vertex}))
        else:
            angle_groups[-1][1].setdefault(kind, vertex)
    if (
        len(angle_groups) > 1
        and angle_groups[0][0] + tau - angle_groups[-1][0]
        <= angular_epsilon
    ):
        _last_angle, last_vertices = angle_groups.pop()
        for kind, vertex in last_vertices.items():
            angle_groups[0][1].setdefault(kind, vertex)
    if len(angle_groups) < 3:
        return ()

    def ray_vertex(
        polygon: _PlanePartitionPolygon,
        angle: float,
    ) -> _PlanePartitionVertex:
        direction = np.asarray((cos(angle), sin(angle)), dtype=float)
        candidates: list[tuple[float, str, np.ndarray]] = []
        for start_vertex, end_vertex in zip(
            polygon.vertices,
            (*polygon.vertices[1:], polygon.vertices[0]),
        ):
            start = np.asarray(start_vertex.plane_coordinates, dtype=float)
            end = np.asarray(end_vertex.plane_coordinates, dtype=float)
            edge = end - start
            denominator = _cross2(direction, edge)
            if abs(denominator) <= angular_epsilon:
                continue
            delta = start - center
            distance = _cross2(delta, edge) / denominator
            edge_parameter = _cross2(delta, direction) / denominator
            if (
                distance >= -coordinate_epsilon
                and -coordinate_epsilon <= edge_parameter <= 1.0 + coordinate_epsilon
            ):
                point = center + max(0.0, distance) * direction
                candidates.append(
                    (
                        max(0.0, distance),
                        f"{start_vertex.stable_token}->{end_vertex.stable_token}",
                        point,
                    )
                )
        if not candidates:
            raise QuadricSectionCompositingError(
                "nested convex ring ray did not meet its polygon boundary"
            )
        _distance, _token, point = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return registry.register(point)

    def boundary_vertices() -> tuple[
        tuple[_PlanePartitionVertex, ...],
        tuple[_PlanePartitionVertex, ...],
    ]:
        return (
            tuple(
                vertices.get("inner") or ray_vertex(inner, angle)
                for angle, vertices in angle_groups
            ),
            tuple(
                vertices.get("outer") or ray_vertex(outer, angle)
                for angle, vertices in angle_groups
            ),
        )

    if minimum_screen_triangle_altitude is not None:
        screen_threshold = float(minimum_screen_triangle_altitude)
        if not isfinite(screen_threshold) or screen_threshold <= 0.0:
            raise QuadricSectionCompositingError(
                "minimum ring triangle altitude must be finite and positive"
            )

        def triangle_altitude(
            triangle: _PlanePartitionPolygon,
        ) -> float:
            points = np.asarray(
                tuple(vertex.screen_point for vertex in triangle.vertices),
                dtype=float,
            )
            edges = np.roll(points, -1, axis=0) - points
            lengths = np.linalg.norm(edges, axis=1)
            if np.any(lengths <= np.finfo(float).tiny):
                return 0.0
            double_area = abs(_cross2(points[1] - points[0], points[2] - points[0]))
            return float(double_area / np.max(lengths))

        while len(angle_groups) > 3:
            inner_vertices, outer_vertices = boundary_vertices()
            merge_candidates: list[
                tuple[float, float, str, int, int]
            ] = []
            for index in range(len(angle_groups)):
                following = (index + 1) % len(angle_groups)
                first_kinds = angle_groups[index][1]
                second_kinds = angle_groups[following][1]
                if set(first_kinds) & set(second_kinds):
                    continue
                polygon = _make_plane_partition_polygon(
                    f"ring-probe:{index:06d}",
                    (
                        inner_vertices[index],
                        outer_vertices[index],
                        outer_vertices[following],
                        inner_vertices[following],
                    ),
                    coordinate_epsilon,
                )
                if polygon is None:
                    unstable = True
                else:
                    triangles = _triangulate_plane_partition_polygon(
                        polygon,
                        coordinate_epsilon,
                    )
                    unstable = (
                        len(triangles) != len(polygon.vertices) - 2
                        or any(
                            triangle_altitude(triangle) <= screen_threshold
                            for triangle in triangles
                        )
                    )
                if not unstable:
                    continue
                inner_span = float(
                    np.linalg.norm(
                        np.asarray(
                            inner_vertices[following].screen_point,
                            dtype=float,
                        )
                        - np.asarray(
                            inner_vertices[index].screen_point,
                            dtype=float,
                        )
                    )
                )
                outer_span = float(
                    np.linalg.norm(
                        np.asarray(
                            outer_vertices[following].screen_point,
                            dtype=float,
                        )
                        - np.asarray(
                            outer_vertices[index].screen_point,
                            dtype=float,
                        )
                    )
                )
                maximum_span = max(inner_span, outer_span)
                if maximum_span > 2.0 * screen_threshold:
                    continue
                angular_gap = (
                    angle_groups[following][0] - angle_groups[index][0]
                ) % tau
                token = "|".join(
                    sorted(
                        vertex.stable_token
                        for vertex in (*first_kinds.values(), *second_kinds.values())
                    )
                )
                merge_candidates.append(
                    (maximum_span, angular_gap, token, index, following)
                )
            if not merge_candidates:
                break
            _span, _gap, _token, index, following = min(merge_candidates)
            if following == 0:
                merged = dict(angle_groups[0][1])
                merged.update(angle_groups[index][1])
                angle_groups[0] = (angle_groups[0][0], merged)
                angle_groups.pop(index)
            else:
                angle, merged = angle_groups[index]
                combined = dict(merged)
                combined.update(angle_groups[following][1])
                angle_groups[index] = (angle, combined)
                angle_groups.pop(following)

    inner_vertices, outer_vertices = boundary_vertices()
    polygons: list[_PlanePartitionPolygon] = []
    for index in range(len(angle_groups)):
        following = (index + 1) % len(angle_groups)
        polygon = _make_plane_partition_polygon(
            f"ring:{index:06d}",
            (
                inner_vertices[index],
                outer_vertices[index],
                outer_vertices[following],
                inner_vertices[following],
            ),
            coordinate_epsilon,
        )
        if polygon is not None:
            polygons.append(polygon)
    return tuple(polygons)


def _point_segment_distance_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    delta = end - start
    length_squared = float(np.dot(delta, delta))
    if length_squared <= 0.0:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, delta) / length_squared)
    ratio = min(1.0, max(0.0, ratio))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def _adaptive_section_curve_parameters(
    curve: SegmentCurve | ParametricConicBranch,
    view: ParallelView,
    breakpoints: Sequence[float],
    *,
    max_chord_error: float,
    max_segments: int,
    parameter_epsilon: float,
) -> tuple[float, ...]:
    """Approximate exact finite-section arcs with certified screen chords."""

    values: list[float] = []
    for raw in sorted(float(item) for item in breakpoints):
        value = min(curve.domain.end, max(curve.domain.start, raw))
        if not values or value - values[-1] > parameter_epsilon:
            values.append(value)
    if not values or values[0] > curve.domain.start + parameter_epsilon:
        values.insert(0, curve.domain.start)
    else:
        values[0] = curve.domain.start
    if values[-1] < curve.domain.end - parameter_epsilon:
        values.append(curve.domain.end)
    else:
        values[-1] = curve.domain.end
    intervals = [
        (left, right)
        for left, right in zip(values, values[1:])
        if right - left > parameter_epsilon
    ]
    if not intervals:
        raise QuadricSectionCompositingError(
            f"section curve {curve.curve_id!r} has no stable parameter interval"
        )

    projection = np.asarray(view.matrix[:2], dtype=float)
    cache: dict[float, np.ndarray] = {}

    def project(parameter: float) -> np.ndarray:
        key = float(parameter)
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = projection @ np.asarray(curve.point(key), dtype=float)
        if result.shape != (2,) or not np.all(np.isfinite(result)):
            raise QuadricSectionCompositingError(
                f"section curve {curve.curve_id!r} has a non-finite projection"
            )
        cache[key] = result
        return result

    probe_fractions = (0.25, 0.5, 0.75)
    while True:
        split: list[int] = []
        for index, (left, right) in enumerate(intervals):
            first = project(left)
            last = project(right)
            observed = max(
                _point_segment_distance_2d(
                    project(left + fraction * (right - left)),
                    first,
                    last,
                )
                for fraction in probe_fractions
            )
            if observed > max_chord_error:
                split.append(index)
        if not split:
            break
        if len(intervals) + len(split) > max_segments:
            raise QuadricSectionCompositingError(
                f"section curve {curve.curve_id!r} needs more than "
                f"{max_segments} plane fragments for its boundary"
            )
        marked = set(split)
        refined: list[tuple[float, float]] = []
        for index, (left, right) in enumerate(intervals):
            if index not in marked:
                refined.append((left, right))
                continue
            middle = left + 0.5 * (right - left)
            if middle == left or middle == right:
                raise QuadricSectionCompositingError(
                    f"section curve {curve.curve_id!r} cannot refine at "
                    "floating-point resolution"
                )
            refined.extend(((left, middle), (middle, right)))
        intervals = refined
    result = [intervals[0][0]]
    result.extend(right for _left, right in intervals)
    return tuple(result)


def _triangulate_plane_partition_polygon(
    polygon: _PlanePartitionPolygon,
    coordinate_epsilon: float,
) -> tuple[_PlanePartitionPolygon, ...]:
    """Triangulate one canonical convex polygon with stable fan identities."""

    result: list[_PlanePartitionPolygon] = []
    anchor = polygon.vertices[0]
    for index in range(1, len(polygon.vertices) - 1):
        triangle = _make_plane_partition_polygon(
            f"{polygon.stable_token}:triangle:{index - 1:04d}",
            (anchor, polygon.vertices[index], polygon.vertices[index + 1]),
            coordinate_epsilon,
        )
        if triangle is not None:
            result.append(triangle)
    return tuple(result)


def _triangulate_convex_partition_polygon_from_center(
    polygon: _PlanePartitionPolygon,
    registry: _CanonicalVertexRegistry,
    coordinate_epsilon: float,
) -> tuple[_PlanePartitionPolygon, ...]:
    """Triangulate a fine convex boundary without boundary-only fan slivers."""

    center = registry.register(
        np.mean(
            np.asarray(
                tuple(
                    vertex.plane_coordinates for vertex in polygon.vertices
                ),
                dtype=float,
            ),
            axis=0,
        )
    )
    result: list[_PlanePartitionPolygon] = []
    for index, start in enumerate(polygon.vertices):
        end = polygon.vertices[(index + 1) % len(polygon.vertices)]
        triangle = _make_plane_partition_polygon(
            f"{polygon.stable_token}:center-triangle:{index:06d}",
            (center, start, end),
            coordinate_epsilon,
        )
        if triangle is not None:
            result.append(triangle)
    return tuple(result)


def _cancel_opposite_partition_edges(
    edges: Counter[tuple[str, str]],
) -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    pairs = {tuple(sorted(edge)) for edge in edges}
    for first, second in sorted(pairs):
        balance = edges[(first, second)] - edges[(second, first)]
        if balance > 0:
            result[(first, second)] = balance
        elif balance < 0:
            result[(second, first)] = -balance
    return result


def _plane_partition_polygon_contours(
    polygons: Sequence[_PlanePartitionPolygon],
    coordinate_epsilon: float,
) -> tuple[tuple[_PlanePartitionVertex, ...], ...]:
    """Union canonical polygons by noding and cancelling directed edges."""

    if not polygons:
        return ()
    vertices: dict[str, _PlanePartitionVertex] = {}
    raw_edges: Counter[tuple[str, str]] = Counter()
    for polygon in sorted(polygons, key=lambda item: item.stable_token):
        if not isinstance(polygon, _PlanePartitionPolygon):
            raise TypeError("contour union requires plane partition polygons")
        for vertex in polygon.vertices:
            existing = vertices.get(vertex.stable_token)
            if existing is not None:
                delta_u = (
                    existing.plane_coordinates[0] - vertex.plane_coordinates[0]
                )
                delta_v = (
                    existing.plane_coordinates[1] - vertex.plane_coordinates[1]
                )
                if (
                    delta_u * delta_u + delta_v * delta_v
                    > coordinate_epsilon * coordinate_epsilon
                ):
                    raise QuadricSectionCompositingError(
                        "canonical plane vertex token refers to two positions"
                    )
            vertices[vertex.stable_token] = vertex
        for start, end in zip(
            polygon.vertices,
            (*polygon.vertices[1:], polygon.vertices[0]),
        ):
            if start.stable_token == end.stable_token:
                raise QuadricSectionCompositingError(
                    "plane partition polygon contains a zero-length edge"
                )
            raw_edges[(start.stable_token, end.stable_token)] += 1

    residual = _cancel_opposite_partition_edges(raw_edges)
    residual_vertices = {
        token
        for edge in residual
        for token in edge
    }
    residual_positions = {
        token: vertices[token].plane_coordinates for token in residual_vertices
    }
    minimum_corner = (
        min(point[0] for point in residual_positions.values()),
        min(point[1] for point in residual_positions.values()),
    )
    maximum_corner = (
        max(point[0] for point in residual_positions.values()),
        max(point[1] for point in residual_positions.values()),
    )
    maximum_extent = max(
        maximum_corner[0] - minimum_corner[0],
        maximum_corner[1] - minimum_corner[1],
        coordinate_epsilon,
    )
    grid_cell_size = max(
        coordinate_epsilon * 32.0,
        maximum_extent / max(sqrt(len(residual_vertices)), 1.0),
    )

    def grid_cell(point: tuple[float, float]) -> tuple[int, int]:
        return (
            floor((point[0] - minimum_corner[0]) / grid_cell_size),
            floor((point[1] - minimum_corner[1]) / grid_cell_size),
        )

    vertex_grid: dict[tuple[int, int], set[str]] = {}
    for token, point in residual_positions.items():
        vertex_grid.setdefault(grid_cell(point), set()).add(token)

    def segment_grid_cells(
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[tuple[int, int], ...]:
        """Traverse the uniform vertex grid along one residual segment."""

        current_x, current_y = grid_cell(start)
        target_x, target_y = grid_cell(end)
        cells = [(current_x, current_y)]
        direction_x = end[0] - start[0]
        direction_y = end[1] - start[1]
        if current_x == target_x and current_y == target_y:
            return tuple(cells)

        if direction_x > 0.0:
            step_x = 1
            next_x = minimum_corner[0] + (current_x + 1) * grid_cell_size
            maximum_x = (next_x - start[0]) / direction_x
            delta_x = grid_cell_size / direction_x
        elif direction_x < 0.0:
            step_x = -1
            next_x = minimum_corner[0] + current_x * grid_cell_size
            maximum_x = (next_x - start[0]) / direction_x
            delta_x = -grid_cell_size / direction_x
        else:
            step_x = 0
            maximum_x = float("inf")
            delta_x = float("inf")

        if direction_y > 0.0:
            step_y = 1
            next_y = minimum_corner[1] + (current_y + 1) * grid_cell_size
            maximum_y = (next_y - start[1]) / direction_y
            delta_y = grid_cell_size / direction_y
        elif direction_y < 0.0:
            step_y = -1
            next_y = minimum_corner[1] + current_y * grid_cell_size
            maximum_y = (next_y - start[1]) / direction_y
            delta_y = -grid_cell_size / direction_y
        else:
            step_y = 0
            maximum_y = float("inf")
            delta_y = float("inf")

        maximum_steps = abs(target_x - current_x) + abs(target_y - current_y) + 2
        while (current_x, current_y) != (target_x, target_y):
            if len(cells) > maximum_steps:
                raise QuadricSectionCompositingError(
                    "plane partition spatial traversal did not reach its endpoint"
                )
            needs_x_step = current_x != target_x
            needs_y_step = current_y != target_y
            comparison_epsilon = (
                np.finfo(float).eps
                * max(1.0, abs(maximum_x), abs(maximum_y))
                if isfinite(maximum_x) and isfinite(maximum_y)
                else 0.0
            )
            if not needs_y_step:
                current_x += step_x
                maximum_x += delta_x
            elif not needs_x_step:
                current_y += step_y
                maximum_y += delta_y
            elif maximum_x < maximum_y - comparison_epsilon:
                current_x += step_x
                maximum_x += delta_x
            elif maximum_y < maximum_x - comparison_epsilon:
                current_y += step_y
                maximum_y += delta_y
            else:
                current_x += step_x
                current_y += step_y
                maximum_x += delta_x
                maximum_y += delta_y
            cells.append((current_x, current_y))
        return tuple(cells)

    primitive_edges: Counter[tuple[str, str]] = Counter()
    for (start_token, end_token), multiplicity in sorted(residual.items()):
        start = residual_positions[start_token]
        end = residual_positions[end_token]
        direction_u = end[0] - start[0]
        direction_v = end[1] - start[1]
        length_squared = direction_u * direction_u + direction_v * direction_v
        length = sqrt(max(0.0, length_squared))
        if length <= coordinate_epsilon:
            raise QuadricSectionCompositingError(
                "plane partition contour contains a zero-length edge"
            )
        points: list[tuple[float, str]] = [(0.0, start_token), (1.0, end_token)]
        parameter_epsilon = coordinate_epsilon / length
        candidate_tokens: set[str] = set()
        for cell_x, cell_y in segment_grid_cells(start, end):
            for x_offset in (-1, 0, 1):
                for y_offset in (-1, 0, 1):
                    candidate_tokens.update(
                        vertex_grid.get(
                            (cell_x + x_offset, cell_y + y_offset),
                            (),
                        )
                    )
        for token in sorted(candidate_tokens):
            if token in (start_token, end_token):
                continue
            point = residual_positions[token]
            point_u = point[0] - start[0]
            point_v = point[1] - start[1]
            parameter = (
                point_u * direction_u + point_v * direction_v
            ) / length_squared
            if not parameter_epsilon < parameter < 1.0 - parameter_epsilon:
                continue
            distance = abs(direction_u * point_v - direction_v * point_u) / length
            if distance <= coordinate_epsilon:
                points.append((parameter, token))
        points.sort(key=lambda item: (item[0], item[1]))
        node_tokens: list[str] = []
        for _parameter, token in points:
            if not node_tokens or token != node_tokens[-1]:
                node_tokens.append(token)
        for first, second in zip(node_tokens, node_tokens[1:]):
            primitive_edges[(first, second)] += multiplicity

    boundary_edges = _cancel_opposite_partition_edges(primitive_edges)
    if any(count != 1 for count in boundary_edges.values()):
        raise QuadricSectionCompositingError(
            "plane fragment boundary contains duplicate directed edges"
        )
    remaining = set(boundary_edges)
    outgoing: dict[str, set[str]] = {}
    incoming: dict[str, set[str]] = {}
    for start, end in remaining:
        outgoing.setdefault(start, set()).add(end)
        incoming.setdefault(end, set()).add(start)
    if set(outgoing) != set(incoming) or any(
        len(outgoing[token]) != len(incoming[token]) for token in set(outgoing)
    ):
        raise QuadricSectionCompositingError(
            "plane fragment union produced an open boundary"
        )

    loops: list[tuple[_PlanePartitionVertex, ...]] = []
    while remaining:
        start, current = min(remaining)
        previous = start
        remaining.remove((start, current))
        tokens = [start, current]
        while current != start:
            candidates = sorted(
                candidate
                for candidate in outgoing.get(current, ())
                if (current, candidate) in remaining
            )
            if not candidates:
                raise QuadricSectionCompositingError(
                    "plane fragment union produced an open contour"
                )
            previous_point = vertices[previous].plane_coordinates
            current_point = vertices[current].plane_coordinates
            incoming_direction = (
                current_point[0] - previous_point[0],
                current_point[1] - previous_point[1],
            )

            def turn_key(candidate: str) -> tuple[float, str]:
                candidate_point = vertices[candidate].plane_coordinates
                outgoing_direction = (
                    candidate_point[0] - current_point[0],
                    candidate_point[1] - current_point[1],
                )
                angle = atan2(
                    incoming_direction[0] * outgoing_direction[1]
                    - incoming_direction[1] * outgoing_direction[0],
                    incoming_direction[0] * outgoing_direction[0]
                    + incoming_direction[1] * outgoing_direction[1],
                )
                if angle < 0.0:
                    angle += tau
                return angle, candidate

            following = min(candidates, key=turn_key)
            remaining.remove((current, following))
            previous, current = current, following
            tokens.append(current)
            if len(tokens) > len(boundary_edges) + 2:
                raise QuadricSectionCompositingError(
                    "plane fragment contour traversal did not close"
                )
        tokens.pop()
        changed = True
        while changed and len(tokens) >= 3:
            changed = False
            for index in range(len(tokens)):
                if _partition_vertex_is_collinear(
                    vertices[tokens[index - 1]],
                    vertices[tokens[index]],
                    vertices[tokens[(index + 1) % len(tokens)]],
                    coordinate_epsilon,
                ):
                    tokens.pop(index)
                    changed = True
                    break
        if len(tokens) < 3:
            raise QuadricSectionCompositingError(
                "plane fragment contour has no stable area"
            )
        start_index = min(range(len(tokens)), key=tokens.__getitem__)
        tokens = tokens[start_index:] + tokens[:start_index]
        loops.append(tuple(vertices[token] for token in tokens))
    loops.sort(key=lambda loop: tuple(vertex.stable_token for vertex in loop))
    return tuple(loops)


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
    plane_origin = np.asarray(plane.point, dtype=float)
    partition_registry = _CanonicalVertexRegistry(
        plane_origin=plane_origin,
        plane_u=plane_u,
        plane_v=plane_v,
        screen_origin=screen_origin,
        screen_basis=screen_basis,
        coordinate_epsilon=coordinate_epsilon,
    )
    partition_proxy = _make_plane_partition_polygon(
        f"proxy:{surface.surface_id}",
        tuple(
            partition_registry.register(
                inverse_screen_basis
                @ (np.asarray(point, dtype=float) - screen_origin)
            )
            for point in proxy_polygon
        ),
        coordinate_epsilon,
    )
    if partition_proxy is None:
        raise QuadricSectionCompositingError(
            "quadric projection proxy has no stable plane partition"
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

    # Build one finite-solid section boundary in plane coordinates.  Analytic
    # side curves contribute tangent support half-planes, while finite cylinder
    # and cone caps contribute exact axial half-planes.  Critical parameters
    # are retained before adaptive chord sampling so a front/back switch cannot
    # hide inside one boundary edge.  Keep the chord error well below the
    # requested display error: the resulting circumscribed polygon may differ
    # from the true curve only within that explicit boundary tolerance.
    sphere_needs_tight_tangent_boundary = False
    if isinstance(surface, SphereSpec):
        plane_offset = abs(
            float(
                np.dot(
                    np.asarray(surface.center, dtype=float) - plane_origin,
                    np.asarray(plane.normal, dtype=float),
                )
            )
        )
        section_radius = sqrt(
            max(
                0.0,
                surface.radius * surface.radius - plane_offset * plane_offset,
            )
        )
        sphere_needs_tight_tangent_boundary = (
            section_radius < 0.5 * surface.radius
        )
    if sphere_needs_tight_tangent_boundary:
        section_chord_error = max(
            screen_epsilon * 16.0,
            error / 1048576.0,
        )
    else:
        section_chord_error = max(
            screen_epsilon * 32.0,
            error / 262144.0,
        )
    if section_chord_error > error:
        raise QuadricSectionCompositingError(
            "floating-point screen tolerance cannot certify the requested "
            f"max_screen_error={error:.9g}"
        )
    try:
        section_trace = compute_quadric_section(
            f"{plane.plane_id}:finite-solid-boundary",
            surface,
            plane,
            context=resolved,
        )
        section_curves = section_trace_curves(section_trace)
    except (QuadricSectionError, ValueError) as exc:
        raise QuadricSectionCompositingError(
            f"finite quadric section boundary is ambiguous: {exc}"
        ) from exc
    section_tangent_constraints: list[
        tuple[_PlanePartitionVertex, np.ndarray]
    ] = []
    section_segment_count = 0
    parameter_epsilon = resolved.epsilon(GeometryQuantity.PARAMETER)
    for curve in section_curves:
        try:
            events = compute_curve_critical_events(
                curve,
                (surface,),
                view,
                context=resolved,
            )
        except CriticalEventError as exc:
            raise QuadricSectionCompositingError(
                f"finite section curve {curve.curve_id!r} has ambiguous "
                f"boundary events: {exc}"
            ) from exc
        parameters = _adaptive_section_curve_parameters(
            curve,
            view,
            tuple(event.parameter for event in events),
            max_chord_error=section_chord_error,
            max_segments=limits.max_plane_fragments,
            parameter_epsilon=parameter_epsilon,
        )
        section_segment_count += len(parameters) - 1
        if section_segment_count > limits.max_plane_fragments:
            raise QuadricSectionCompositingError(
                "finite section boundary needs more than "
                f"{limits.max_plane_fragments} plane fragments for its boundary"
            )
        for parameter in parameters:
            world = np.asarray(curve.point(parameter), dtype=float)
            vertex = partition_registry.register(
                (
                    float(np.dot(world - plane_origin, plane_u)),
                    float(np.dot(world - plane_origin, plane_v)),
                )
            )
            coordinates = np.asarray(vertex.plane_coordinates, dtype=float)
            gradient = (
                2.0 * restricted_matrix @ coordinates
                + 2.0 * restricted_linear
            )
            if float(np.linalg.norm(gradient)) > coordinate_epsilon:
                section_tangent_constraints.append((vertex, gradient))
    section_seed = _make_plane_partition_polygon(
        f"section-seed:{surface.surface_id}:{plane.plane_id}",
        tuple(
            partition_registry.register(
                (
                    float(np.dot(point - plane_origin, plane_u)),
                    float(np.dot(point - plane_origin, plane_v)),
                )
            )
            for point in patch_corners
        ),
        coordinate_epsilon,
    )
    if section_seed is None:
        raise QuadricSectionCompositingError(
            "display patch has no stable plane partition"
        )
    display_proxy_inside, _proxy_outside_patch = (
        _partition_convex_polygon_by_convex_boundary(
            partition_proxy,
            section_seed,
            partition_registry,
            coordinate_epsilon,
        )
    )
    if not display_proxy_inside:
        raise QuadricSectionCompositingError(
            "quadric projection does not overlap the display patch"
        )
    display_proxy_partition = display_proxy_inside[0]

    # The sampled curve chords form an inscribed polygon and would leave a
    # thin, genuinely-inside crescent in an exterior role.  Instead intersect
    # the supporting tangent half-planes.  The result is a deterministic
    # circumscribed approximation: its small geometric error lies inside the
    # explicit boundary tolerance, while every exterior piece is truly
    # outside the finite section.  Axial cap half-planes are exact and close
    # finite cylinder/cone sections independently of the support quadric.
    support_constraints = [
        _PlanePartitionHalfPlane(
            stable_token=f"section-tangent:{constraint_index:06d}",
            normal=(float(gradient[0]), float(gradient[1])),
            offset=float(
                np.dot(
                    gradient,
                    np.asarray(vertex.plane_coordinates, dtype=float),
                )
            ),
        )
        for constraint_index, (vertex, gradient) in enumerate(
            sorted(
                section_tangent_constraints,
                key=lambda item: item[0].stable_token,
            )
        )
    ]

    if section_tangent_constraints and axial_mapping is not None:
        axial_direction, axial_offset, lower, upper = axial_mapping
        axial_norm_squared = float(np.dot(axial_direction, axial_direction))
        if axial_norm_squared > coordinate_epsilon * coordinate_epsilon:
            for cap_name, target, normal in (
                ("lower", lower, -axial_direction),
                ("upper", upper, axial_direction),
            ):
                support_constraints.append(
                    _PlanePartitionHalfPlane(
                        stable_token=f"section-cap:{cap_name}",
                        normal=(float(normal[0]), float(normal[1])),
                        offset=float(
                            np.dot(
                                normal,
                                (target - axial_offset)
                                * axial_direction
                                / axial_norm_squared,
                            )
                        ),
                    )
                )

    section_partition = _intersect_convex_support_half_planes(
        f"section:{surface.surface_id}:{plane.plane_id}",
        support_constraints,
        partition_registry,
        coordinate_epsilon,
    )
    if section_partition is not None:
        clipped_section, _outside_proxy = (
            _partition_convex_polygon_by_convex_boundary(
                section_partition,
                display_proxy_partition,
                partition_registry,
                coordinate_epsilon,
            )
        )
        section_partition = clipped_section[0] if clipped_section else None
    if section_tangent_constraints and section_partition is None:
        raise QuadricSectionCompositingError(
            "finite section tangent partition unexpectedly became empty"
        )
    if section_partition is not None:
        stable_interior = np.mean(
            np.asarray(
                tuple(
                    vertex.plane_coordinates
                    for vertex in section_partition.vertices
                ),
                dtype=float,
            ),
            axis=0,
        )
        interior_world = (
            plane_origin
            + stable_interior[0] * plane_u
            + stable_interior[1] * plane_v
        )
        interior_parameters = ray_parameters(interior_world)
        if not interior_parameters or not (
            min(interior_parameters) <= boundary_epsilon
            and max(interior_parameters) >= -boundary_epsilon
        ):
            raise QuadricSectionCompositingError(
                "finite section tangent partition has no certified interior"
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
        elif (
            min(parameters) <= boundary_epsilon
            and max(parameters) >= -boundary_epsilon
        ):
            role = PlaneDepthRole.BETWEEN_SURFACE_SHEETS
        else:
            raise QuadricSectionCompositingError(  # pragma: no cover
                "surface ray endpoints cannot be classified relative to the plane"
            )
        classification_cache[key] = role
        return role

    fragments: list[QuadricPlaneFragment] = []

    def append_leaf(
        world: np.ndarray,
        path: str,
        depth: int,
        role: PlaneDepthRole,
    ) -> None:
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

    def partition_triangle(
        world: np.ndarray,
        path: str,
    ) -> tuple[
        _PlanePartitionPolygon,
        tuple[_PlanePartitionPolygon, ...],
        tuple[_PlanePartitionPolygon, ...],
    ]:
        source = _make_plane_partition_polygon(
            f"plane:{plane.plane_id}:cell:{path}",
            tuple(
                partition_registry.register(
                    (
                        float(np.dot(point - plane_origin, plane_u)),
                        float(np.dot(point - plane_origin, plane_v)),
                    )
                )
                for point in np.asarray(world, dtype=float)
            ),
            coordinate_epsilon,
        )
        if source is None:
            raise QuadricSectionCompositingError(
                f"plane partition cell {path!r} has no stable area"
            )
        inside, outside = _partition_triangle_by_convex_proxy(
            source,
            partition_proxy,
            partition_registry,
            coordinate_epsilon,
        )
        return source, inside, outside

    probe_distance = 16.0 * boundary_epsilon
    stable_probe_offsets = (
        np.zeros(2, dtype=float),
        probe_distance * screen_basis[:, 0],
        -probe_distance * screen_basis[:, 0],
        probe_distance * screen_basis[:, 1],
        -probe_distance * screen_basis[:, 1],
    )

    def stable_role(screen_point: np.ndarray) -> PlaneDepthRole | None:
        roles = {
            classify(np.asarray(screen_point, dtype=float) + offset)
            for offset in stable_probe_offsets
        }
        return next(iter(roles)) if len(roles) == 1 else None

    def polygon_role(
        polygon: _PlanePartitionPolygon,
    ) -> PlaneDepthRole | None:
        vertices = tuple(
            np.asarray(vertex.screen_point, dtype=float)
            for vertex in polygon.vertices
        )
        centroid = np.mean(vertices, axis=0)
        # Boundary vertices intentionally lie on the role transition and are
        # therefore poor certification probes.  Validate a bounded sequence
        # of deterministic points strictly inside the already-cut convex
        # polygon.  Stop after two agreeing stable probes; the independent
        # renderer-neutral contract tests every emitted triangle more densely.
        samples = [centroid]
        samples.extend(
            0.2 * vertices[index] + 0.8 * centroid
            for index in range(min(4, len(vertices)))
        )
        roles: set[PlaneDepthRole] = set()
        stable_count = 0
        for sample in samples:
            role = stable_role(np.asarray(sample, dtype=float))
            if role is None:
                continue
            stable_count += 1
            roles.add(role)
            if len(roles) > 1:
                return None
            if stable_count >= 2:
                break
        if stable_count == 0 or len(roles) != 1:
            return None
        role = next(iter(roles))
        if role is PlaneDepthRole.OUTSIDE_PROJECTION:
            return None
        return role

    def polygon_role_diagnostics(
        polygon: _PlanePartitionPolygon,
    ) -> str:
        vertices = tuple(
            np.asarray(vertex.screen_point, dtype=float)
            for vertex in polygon.vertices
        )
        centroid = np.mean(vertices, axis=0)
        samples = [centroid]
        samples.extend(
            0.2 * vertices[index] + 0.8 * centroid
            for index in range(min(4, len(vertices)))
        )
        stable_roles = [
            role.value
            for sample in samples
            if (role := stable_role(np.asarray(sample, dtype=float))) is not None
        ]
        direct_roles = sorted(
            {classify(np.asarray(sample, dtype=float)).value for sample in samples}
        )
        return (
            f"{polygon.stable_token}:stable={sorted(set(stable_roles))}:"
            f"stable_count={len(stable_roles)}:direct={direct_roles}"
        )

    def partition_inside_depth_roles(
        polygon: _PlanePartitionPolygon,
    ) -> tuple[tuple[PlaneDepthRole, _PlanePartitionPolygon], ...] | None:
        if section_partition is None:
            between: tuple[_PlanePartitionPolygon, ...] = ()
            exterior = (polygon,)
        else:
            between, exterior = _partition_convex_polygon_by_convex_boundary(
                polygon,
                section_partition,
                partition_registry,
                coordinate_epsilon,
            )
        result: list[tuple[PlaneDepthRole, _PlanePartitionPolygon]] = []
        for candidate in between:
            role = polygon_role(candidate)
            if role is not PlaneDepthRole.BETWEEN_SURFACE_SHEETS:
                return None
            result.append((role, candidate))
        for candidate in exterior:
            role = polygon_role(candidate)
            if role is None:
                return None
            result.append((role, candidate))
        if not result:
            return None
        return tuple(
            sorted(result, key=lambda item: (item[0].value, item[1].stable_token))
        )

    def append_partitioned_leaf(
        source: _PlanePartitionPolygon,
        polygon_emissions: Sequence[
            tuple[PlaneDepthRole, _PlanePartitionPolygon]
        ],
        path: str,
        depth: int,
    ) -> None:
        triangle_emissions: list[
            tuple[PlaneDepthRole, _PlanePartitionPolygon]
        ] = []
        for role, polygon in polygon_emissions:
            triangle_emissions.extend(
                (role, triangle)
                for triangle in _triangulate_plane_partition_polygon(
                    polygon,
                    coordinate_epsilon,
                )
            )
        if not triangle_emissions:
            raise QuadricSectionCompositingError(
                f"plane partition cell {path!r} emitted no stable triangle"
            )
        triangle_emissions.sort(
            key=lambda item: (item[0].value, item[1].stable_token)
        )
        source_tokens = tuple(
            vertex.stable_token for vertex in source.vertices
        )
        if (
            len(triangle_emissions) == 1
            and tuple(
                vertex.stable_token
                for vertex in triangle_emissions[0][1].vertices
            )
            == source_tokens
        ):
            fragment_path = path
            role, triangle = triangle_emissions[0]
            append_leaf(
                np.asarray(
                    tuple(vertex.world_point for vertex in triangle.vertices),
                    dtype=float,
                ),
                fragment_path,
                depth,
                role,
            )
            return
        for index, (role, triangle) in enumerate(triangle_emissions):
            append_leaf(
                np.asarray(
                    tuple(vertex.world_point for vertex in triangle.vertices),
                    dtype=float,
                ),
                f"{path}:piece:{index:04d}:{role.value}",
                depth,
                role,
            )

    def emit_finite_section_arrangement() -> None:
        if section_partition is None:  # pragma: no cover - caller guards
            raise QuadricSectionCompositingError(
                "finite section arrangement requires a positive-area section"
            )

        # Batch 3 already established the exact patch/proxy split.  Preserve
        # its outside pieces, then replace the repeatedly clipped adaptive
        # cells inside the proxy by one global nested-convex arrangement.
        for root_index, indices in enumerate(((0, 1, 2), (0, 2, 3))):
            source, _inside, outside = partition_triangle(
                patch_corners[list(indices)],
                str(root_index),
            )
            if outside:
                append_partitioned_leaf(
                    source,
                    tuple(
                        (PlaneDepthRole.OUTSIDE_PROJECTION, polygon)
                        for polygon in outside
                    ),
                    str(root_index),
                    0,
                )

        section_role = polygon_role(section_partition)
        if section_role is not PlaneDepthRole.BETWEEN_SURFACE_SHEETS:
            raise QuadricSectionCompositingError(
                "finite section interior is not certified between the "
                "surface sheets"
            )
        section_center = np.mean(
            np.asarray(
                tuple(
                    vertex.plane_coordinates
                    for vertex in section_partition.vertices
                ),
                dtype=float,
            ),
            axis=0,
        )
        core_vertex_count = min(64, len(section_partition.vertices))
        core_indices = tuple(
            sorted(
                {
                    (index * len(section_partition.vertices))
                    // core_vertex_count
                    for index in range(core_vertex_count)
                }
            )
        )
        section_core = _make_plane_partition_polygon(
            f"section-core:{surface.surface_id}:{plane.plane_id}",
            tuple(
                partition_registry.register(
                    section_center
                    + 0.5
                    * (
                        np.asarray(
                            section_partition.vertices[index].plane_coordinates,
                            dtype=float,
                        )
                        - section_center
                    )
                )
                for index in core_indices
            ),
            coordinate_epsilon,
        )
        if section_core is None or polygon_role(section_core) is not (
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS
        ):
            raise QuadricSectionCompositingError(
                "finite section core is not stably between the surface sheets"
            )
        section_triangles = _triangulate_convex_partition_polygon_from_center(
            section_core,
            partition_registry,
            coordinate_epsilon,
        )
        for triangle_index, triangle in enumerate(section_triangles):
            append_leaf(
                np.asarray(
                    tuple(vertex.world_point for vertex in triangle.vertices),
                    dtype=float,
                ),
                "section:piece:"
                f"{triangle_index:06d}:"
                f"{PlaneDepthRole.BETWEEN_SURFACE_SHEETS.value}",
                0,
                PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
            )
        section_boundary_band = _nested_convex_ring_polygons(
            section_core,
            section_partition,
            partition_registry,
            coordinate_epsilon,
            minimum_screen_triangle_altitude=8.0 * screen_epsilon,
        )
        if not section_boundary_band:
            raise QuadricSectionCompositingError(
                "finite section boundary band has no stable area"
            )
        boundary_triangle_index = len(section_triangles)
        for polygon in section_boundary_band:
            for triangle in _triangulate_plane_partition_polygon(
                polygon,
                coordinate_epsilon,
            ):
                append_leaf(
                    np.asarray(
                        tuple(
                            vertex.world_point for vertex in triangle.vertices
                        ),
                        dtype=float,
                    ),
                    "section:piece:"
                    f"{boundary_triangle_index:06d}:"
                    f"{PlaneDepthRole.BETWEEN_SURFACE_SHEETS.value}",
                    0,
                    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                )
                boundary_triangle_index += 1

        ring_polygons = _nested_convex_ring_polygons(
            section_partition,
            display_proxy_partition,
            partition_registry,
            coordinate_epsilon,
            minimum_screen_triangle_altitude=8.0 * screen_epsilon,
        )
        if not ring_polygons:
            raise QuadricSectionCompositingError(
                "finite section/projection ring has no stable area"
            )

        def ring_role(
            polygon: _PlanePartitionPolygon,
        ) -> tuple[PlaneDepthRole, bool]:
            role = polygon_role(polygon)
            stably_certified = role is not None
            if role is None:
                centroid = np.mean(
                    np.asarray(
                        tuple(
                            vertex.screen_point for vertex in polygon.vertices
                        ),
                        dtype=float,
                    ),
                    axis=0,
                )
                direct_role = classify(centroid)
                if direct_role in (
                    PlaneDepthRole.BEHIND_SURFACE,
                    PlaneDepthRole.IN_FRONT_OF_SURFACE,
                ):
                    role = direct_role
            if role not in (
                PlaneDepthRole.BEHIND_SURFACE,
                PlaneDepthRole.IN_FRONT_OF_SURFACE,
            ):
                raise QuadricSectionCompositingError(
                    "finite section ring piece remains mixed after critical "
                    f"boundary insertion: {polygon_role_diagnostics(polygon)}"
                )
            return role, stably_certified

        role_records = []
        for index, polygon in enumerate(ring_polygons):
            role, stably_certified = ring_role(polygon)
            role_records.append((index, polygon, role, stably_certified))

        triangulation_cache: dict[
            tuple[str, tuple[str, ...]],
            tuple[_PlanePartitionPolygon, ...] | None,
        ] = {}
        triangulation_diagnostics: dict[
            tuple[str, tuple[str, ...]],
            tuple[str, ...],
        ] = {}

        def triangle_role_certified(
            triangle: _PlanePartitionPolygon,
            role: PlaneDepthRole,
        ) -> bool:
            vertices = tuple(
                np.asarray(vertex.screen_point, dtype=float)
                for vertex in triangle.vertices
            )
            centroid = np.mean(vertices, axis=0)
            samples = [*vertices, centroid]
            samples.extend(
                0.5 * (start + end)
                for start, end in zip(
                    vertices,
                    (*vertices[1:], vertices[0]),
                )
            )
            samples.extend(
                0.4 * vertex + 0.6 * centroid for vertex in vertices
            )
            stable_sample_count = 0
            for sample in samples:
                observed = stable_role(sample)
                if observed is None:
                    continue
                stable_sample_count += 1
                if observed is not role:
                    return False
            return stable_sample_count > 0

        def role_triangulation(
            polygons: Sequence[_PlanePartitionPolygon],
            role: PlaneDepthRole,
        ) -> tuple[_PlanePartitionPolygon, ...] | None:
            cache_key = (
                role.value,
                tuple(polygon.stable_token for polygon in polygons),
            )
            if cache_key in triangulation_cache:
                return triangulation_cache[cache_key]
            failure_reasons: list[str] = []
            if len(polygons) == 1:
                contours = (polygons[0].vertices,)
            else:
                contours = _plane_partition_polygon_contours(
                    polygons,
                    coordinate_epsilon,
                )
            if len(contours) != 1:
                triangulation_cache[cache_key] = None
                return None
            simple = _make_plane_partition_polygon(
                "ring-union:"
                + sha256("|".join(cache_key[1]).encode("utf-8")).hexdigest()[:24],
                contours[0],
                coordinate_epsilon,
            )
            if simple is None:
                triangulation_cache[cache_key] = None
                return None

            source_area = abs(_partition_signed_area(simple.vertices))
            source_perimeter = sum(
                float(
                    np.linalg.norm(
                        np.asarray(end.plane_coordinates, dtype=float)
                        - np.asarray(start.plane_coordinates, dtype=float)
                    )
                )
                for start, end in zip(
                    simple.vertices,
                    (*simple.vertices[1:], simple.vertices[0]),
                )
            )
            topology_area_tolerance = max(
                coordinate_epsilon * coordinate_epsilon,
                coordinate_epsilon * source_perimeter,
                32.0 * boundary_epsilon * source_perimeter,
                source_area * 1.0e-10,
            )
            weighted_center_numerator = np.zeros(2, dtype=float)
            weighted_center_denominator = 0.0
            center_candidates: list[np.ndarray] = []
            kernel: _PlanePartitionPolygon | None = display_proxy_partition
            for edge_index, start_vertex in enumerate(simple.vertices):
                if kernel is None:
                    break
                end_vertex = simple.vertices[
                    (edge_index + 1) % len(simple.vertices)
                ]
                kernel, _outside_kernel = _split_convex_polygon_by_half_plane(
                    kernel,
                    start_vertex,
                    end_vertex,
                    partition_registry,
                    coordinate_epsilon,
                    boundary_token=f"kernel-edge:{edge_index:06d}",
                )
            if kernel is not None:
                center_candidates.append(
                    np.mean(
                        np.asarray(
                            tuple(
                                vertex.plane_coordinates
                                for vertex in kernel.vertices
                            ),
                            dtype=float,
                        ),
                        axis=0,
                    )
                )
            center_candidates.append(
                np.mean(
                    np.asarray(
                        tuple(
                            vertex.plane_coordinates
                            for vertex in simple.vertices
                        ),
                        dtype=float,
                    ),
                    axis=0,
                )
            )
            for polygon in polygons:
                area = abs(_partition_signed_area(polygon.vertices))
                polygon_center = np.mean(
                    np.asarray(
                        tuple(
                            vertex.plane_coordinates
                            for vertex in polygon.vertices
                        ),
                        dtype=float,
                    ),
                    axis=0,
                )
                weighted_center_numerator += area * polygon_center
                weighted_center_denominator += area
            if weighted_center_denominator > 0.0:
                center_candidates.insert(
                    0,
                    weighted_center_numerator / weighted_center_denominator,
                )
            center_candidates.extend(
                np.mean(
                    np.asarray(
                        tuple(
                            vertex.plane_coordinates
                            for vertex in polygon.vertices
                        ),
                        dtype=float,
                    ),
                    axis=0,
                )
                for polygon in (polygons[0], polygons[-1])
            )
            center_candidates.extend(
                0.5
                * (
                    np.asarray(start.plane_coordinates, dtype=float)
                    + np.asarray(end.plane_coordinates, dtype=float)
                )
                for start, end in zip(
                    simple.vertices,
                    (*simple.vertices[1:], simple.vertices[0]),
                )
            )
            for center_index, center_coordinates in enumerate(center_candidates):
                center_vertex = partition_registry.register(center_coordinates)
                center_role = stable_role(
                    np.asarray(center_vertex.screen_point, dtype=float)
                )
                if center_role is not role:
                    failure_reasons.append(
                        f"center-{center_index}:role-"
                        f"{None if center_role is None else center_role.value}"
                    )
                center_fan: list[_PlanePartitionPolygon] = []
                valid_center = True
                for edge_index, start in enumerate(simple.vertices):
                    end = simple.vertices[(edge_index + 1) % len(simple.vertices)]
                    first = np.asarray(start.plane_coordinates, dtype=float)
                    second = np.asarray(end.plane_coordinates, dtype=float)
                    center = np.asarray(
                        center_vertex.plane_coordinates,
                        dtype=float,
                    )
                    signed_area = _cross2(first - center, second - center)
                    signed_tolerance = coordinate_epsilon * max(
                        float(np.linalg.norm(second - first)),
                        coordinate_epsilon,
                    )
                    if signed_area < -signed_tolerance:
                        valid_center = False
                        failure_reasons.append(
                            f"center-{center_index}:outside-kernel-edge-{edge_index}"
                        )
                        break
                    if abs(signed_area) <= signed_tolerance:
                        continue
                    triangle = _make_plane_partition_polygon(
                        f"{simple.stable_token}:center:{center_index:02d}:"
                        f"{edge_index:04d}",
                        (center_vertex, start, end),
                        coordinate_epsilon,
                    )
                    if triangle is None or not triangle_role_certified(
                        triangle,
                        role,
                    ):
                        valid_center = False
                        failure_reasons.append(
                            f"center-{center_index}:uncertified-edge-{edge_index}"
                        )
                        break
                    center_fan.append(triangle)
                center_fan_area = sum(
                    abs(_partition_signed_area(triangle.vertices))
                    for triangle in center_fan
                )
                if (
                    valid_center
                    and center_fan
                    and abs(source_area - center_fan_area)
                    <= topology_area_tolerance
                ):
                    result = tuple(center_fan)
                    triangulation_cache[cache_key] = result
                    return result
                if valid_center and center_fan:
                    failure_reasons.append(
                        f"center-{center_index}:area-delta-"
                        f"{center_fan_area - source_area:.9g}"
                    )

            for anchor_index in sorted(
                range(len(simple.vertices)),
                key=lambda index: simple.vertices[index].stable_token,
            ):
                ordered = (
                    simple.vertices[anchor_index:]
                    + simple.vertices[:anchor_index]
                )
                fan = tuple(
                    triangle
                    for triangle_index in range(1, len(ordered) - 1)
                    if (
                        triangle := _make_plane_partition_polygon(
                            f"{simple.stable_token}:fan:{anchor_index:04d}:"
                            f"{triangle_index:04d}",
                            (
                                ordered[0],
                                ordered[triangle_index],
                                ordered[triangle_index + 1],
                            ),
                            coordinate_epsilon,
                        )
                    )
                    is not None
                )
                fan_area = sum(
                    abs(_partition_signed_area(triangle.vertices))
                    for triangle in fan
                )
                if (
                    len(fan) == len(simple.vertices) - 2
                    and abs(source_area - fan_area)
                    <= topology_area_tolerance
                    and all(
                        triangle_role_certified(triangle, role)
                        for triangle in fan
                    )
                ):
                    triangulation_cache[cache_key] = fan
                    return fan

            failed_states: set[tuple[str, ...]] = set()

            def solve(
                vertices: tuple[_PlanePartitionVertex, ...],
            ) -> tuple[_PlanePartitionPolygon, ...] | None:
                state = tuple(vertex.stable_token for vertex in vertices)
                if state in failed_states:
                    return None
                if len(vertices) == 3:
                    triangle = _make_plane_partition_polygon(
                        f"{simple.stable_token}:ear:final",
                        vertices,
                        coordinate_epsilon,
                    )
                    if triangle is not None and triangle_role_certified(
                        triangle,
                        role,
                    ):
                        return (triangle,)
                    failed_states.add(state)
                    return None

                ear_candidates: list[
                    tuple[float, str, int, _PlanePartitionPolygon]
                ] = []
                for index, middle in enumerate(vertices):
                    previous = vertices[index - 1]
                    following = vertices[(index + 1) % len(vertices)]
                    first = np.asarray(previous.plane_coordinates, dtype=float)
                    second = np.asarray(middle.plane_coordinates, dtype=float)
                    third = np.asarray(following.plane_coordinates, dtype=float)
                    signed_double_area = _cross2(second - first, third - first)
                    scale = max(
                        float(np.linalg.norm(second - first)),
                        float(np.linalg.norm(third - second)),
                        coordinate_epsilon,
                    )
                    if signed_double_area <= coordinate_epsilon * scale:
                        continue

                    # These loops are unions of consecutive radial quads, so
                    # they are angular-monotone: every positive local corner
                    # is an ear.  A generic all-vertices containment scan would
                    # make the rare tangent merge cubic in boundary samples.
                    triangle = _make_plane_partition_polygon(
                        f"{simple.stable_token}:ear:{middle.stable_token}",
                        (previous, middle, following),
                        coordinate_epsilon,
                    )
                    if triangle is None:
                        continue
                    ear_candidates.append(
                        (
                            -signed_double_area,
                            middle.stable_token,
                            index,
                            triangle,
                        )
                    )
                for _area, _token, index, triangle in sorted(ear_candidates):
                    if not triangle_role_certified(triangle, role):
                        continue
                    remainder = solve(vertices[:index] + vertices[index + 1 :])
                    if remainder is not None:
                        return (triangle, *remainder)
                failed_states.add(state)
                return None

            result = solve(simple.vertices)
            if result is not None:
                triangle_area = sum(
                    abs(_partition_signed_area(triangle.vertices))
                    for triangle in result
                )
                if (
                    abs(source_area - triangle_area)
                    > topology_area_tolerance
                ):
                    result = None
            if result is None:
                failure_reasons.append(
                    f"ear-search-failed-states:{len(failed_states)}"
                )
                triangulation_diagnostics[cache_key] = tuple(failure_reasons)
            triangulation_cache[cache_key] = result
            return result

        if any(
            role_records[index][2] is not role_records[index - 1][2]
            for index in range(len(role_records))
        ):
            rotation = next(
                index
                for index in range(len(role_records))
                if role_records[index][2] is not role_records[index - 1][2]
            )
            role_records = role_records[rotation:] + role_records[:rotation]

        runs: list[
            tuple[
                PlaneDepthRole,
                list[tuple[int, _PlanePartitionPolygon, bool]],
            ]
        ] = []
        for index, polygon, role, stably_certified in role_records:
            if not runs or runs[-1][0] is not role:
                runs.append((role, []))
            runs[-1][1].append((index, polygon, stably_certified))

        output_groups: list[
            tuple[
                PlaneDepthRole,
                list[tuple[int, _PlanePartitionPolygon, bool]],
                tuple[_PlanePartitionPolygon, ...],
            ]
        ] = []
        for role, run in runs:
            groups: list[
                list[tuple[int, _PlanePartitionPolygon, bool]]
            ] = []
            unstable_positions = [
                index for index, record in enumerate(run) if not record[2]
            ]
            padded_intervals: list[tuple[int, int]] = []
            for position in unstable_positions:
                start = max(0, position - 16)
                end = min(len(run) - 1, position + 16)
                if padded_intervals and start <= padded_intervals[-1][1] + 1:
                    padded_intervals[-1] = (
                        padded_intervals[-1][0],
                        max(padded_intervals[-1][1], end),
                    )
                else:
                    padded_intervals.append((start, end))
            cursor = 0
            for start, end in padded_intervals:
                groups.extend([run[index]] for index in range(cursor, start))
                groups.append(run[start : end + 1])
                cursor = end + 1
            groups.extend([run[index]] for index in range(cursor, len(run)))

            for group in groups:
                if len(group) == 1 and group[0][2]:
                    triangulation = _triangulate_plane_partition_polygon(
                        group[0][1],
                        coordinate_epsilon,
                    )
                else:
                    triangulation = role_triangulation(
                        tuple(polygon for _index, polygon, _stable in group),
                        role,
                    )
                if triangulation is None:
                    key = (
                        role.value,
                        tuple(polygon.stable_token for _i, polygon, _s in group),
                    )
                    raise QuadricSectionCompositingError(
                        "finite section tangent neighborhood cannot be "
                        "triangulated without an unstable fragment "
                        f"({role.value}, ring indices "
                        f"{group[0][0]}..{group[-1][0]}); "
                        f"diagnostics={triangulation_diagnostics.get(key, ())}"
                    )
                output_groups.append((role, group, triangulation))

        output_groups.sort(
            key=lambda item: min(index for index, _polygon, _stable in item[1])
        )
        for group_index, (role, _records, triangles) in enumerate(output_groups):
            for triangle_index, triangle in enumerate(triangles):
                append_leaf(
                    np.asarray(
                        tuple(
                            vertex.world_point for vertex in triangle.vertices
                        ),
                        dtype=float,
                    ),
                    f"ring:{group_index:06d}:piece:{triangle_index:04d}:"
                    f"{role.value}",
                    0,
                    role,
                )

    def visit(world: np.ndarray, path: str, depth: int) -> None:
        screen = np.asarray(world @ view.matrix[:2].T, dtype=float)
        source, inside, outside = partition_triangle(world, path)
        if not inside:
            if depth < limits.minimum_subdivision_depth:
                for index, child in enumerate(_subdivide_triangle(world)):
                    visit(child, f"{path}.{index}", depth + 1)
            else:
                append_partitioned_leaf(
                    source,
                    tuple(
                        (PlaneDepthRole.OUTSIDE_PROJECTION, polygon)
                        for polygon in outside
                    ),
                    path,
                    depth,
                )
            return

        inside_screen = [
            np.asarray(vertex.screen_point, dtype=float)
            for vertex in inside[0].vertices
        ]
        projected_diameter = max(
            float(np.linalg.norm(screen[first] - screen[second]))
            for first in range(3)
            for second in range(first + 1, 3)
        )
        role_partition = partition_inside_depth_roles(inside[0])
        unresolved_tangent_feature = False
        if section_partition is None and role_partition is not None:
            unresolved_tangent_feature = (
                all(
                    role is not PlaneDepthRole.BETWEEN_SURFACE_SHEETS
                    for role, _polygon in role_partition
                )
                and _plane_cell_may_meet_solid(
                    inside_screen,
                    inverse_screen_basis,
                    screen_origin,
                    restricted_surface,
                    stationary,
                    axial_mapping,
                    coordinate_epsilon,
                    implicit_epsilon,
                )
                and projected_diameter > error
            )
        if (
            depth >= limits.minimum_subdivision_depth
            and role_partition is not None
            and not unresolved_tangent_feature
        ):
            append_partitioned_leaf(
                source,
                (
                    *(
                        (PlaneDepthRole.OUTSIDE_PROJECTION, polygon)
                        for polygon in outside
                    ),
                    *role_partition,
                ),
                path,
                depth,
            )
            return
        if depth >= limits.maximum_subdivision_depth:
            if section_partition is None:
                diagnostic_candidates = (inside[0],)
            else:
                diagnostic_between, diagnostic_exterior = (
                    _partition_convex_polygon_by_convex_boundary(
                        inside[0],
                        section_partition,
                        partition_registry,
                        coordinate_epsilon,
                    )
                )
                diagnostic_candidates = (
                    *diagnostic_between,
                    *diagnostic_exterior,
                )
            diagnostics = "; ".join(
                polygon_role_diagnostics(polygon)
                for polygon in diagnostic_candidates
            )
            raise QuadricSectionCompositingError(
                f"finite-surface depth boundary in cell {path!r} remains "
                "mixed after "
                f"{limits.maximum_subdivision_depth} subdivision levels; "
                f"refusing to guess a role at max_screen_error={error:.9g}; "
                f"candidates: {diagnostics}"
            )
        for index, child in enumerate(_subdivide_triangle(world)):
            visit(child, f"{path}.{index}", depth + 1)

    if section_partition is None:
        for root_index, indices in enumerate(((0, 1, 2), (0, 2, 3))):
            visit(patch_corners[list(indices)], str(root_index), 0)
    else:
        emit_finite_section_arrangement()

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
