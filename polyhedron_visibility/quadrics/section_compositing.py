"""Depth-aware public facade for quadric/section compositing.

The implementation module retains the established adaptive fill partition. This
facade adds exact, independently painted outline fragments so a cutting-plane
border cannot remain in front of a solid while the corresponding plane point is
geometrically behind or between the two projection sheets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence

import numpy as np

from ..compositor import CompositorCycleError, PainterConstraint, stable_topological_sort
from ..geometry import GeometryQuantity, resolve_geometry_context
from ..topology import ParameterInterval, assert_exact_partition, partition_parameter_domain
from . import _section_compositing_impl as _impl
from ._section_compositing_impl import *  # noqa: F401,F403
from .critical import compute_curve_critical_events
from .curves import SegmentCurve


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
        if not isinstance(self.interval, ParameterInterval) or self.interval.length <= 0.0:
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

    # Keep the released seven slots first so existing fixed-slot consumers keep
    # the two surface-sheet indices (1 and 4). Painter relations, not tuple
    # position, place the three additional outline groups at their true depths.
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


class QuadricSectionCompositingFrame(_impl.QuadricSectionCompositingFrame):
    """Released section frame plus exact plane-outline depth fragments."""

    __slots__ = ("plane_outline_fragments",)

    def __init__(
        self,
        *,
        base_frame: QuadricCompositingFrame,
        surface_id: str,
        plane: object,
        patch: object,
        surface_proxy: object,
        plane_fragments: tuple[QuadricPlaneFragment, ...],
        plane_outline_fragments: tuple[QuadricPlaneOutlineFragment, ...],
        paint_items: QuadricSectionPaintItems,
        order_relations: tuple[object, ...],
        draw_order: tuple[str, ...],
        max_screen_error: float,
        ray_classification_count: int,
        schema: str = QUADRIC_SECTION_COMPOSITING_SCHEMA,
    ) -> None:
        super().__init__(
            base_frame=base_frame,
            surface_id=surface_id,
            plane=plane,
            patch=patch,
            surface_proxy=surface_proxy,
            plane_fragments=plane_fragments,
            paint_items=paint_items,
            order_relations=order_relations,
            draw_order=draw_order,
            max_screen_error=max_screen_error,
            ray_classification_count=ray_classification_count,
            schema=schema,
        )
        values = tuple(plane_outline_fragments)
        if not all(isinstance(item, QuadricPlaneOutlineFragment) for item in values):
            raise TypeError(
                "plane_outline_fragments must contain QuadricPlaneOutlineFragment"
            )
        identities = tuple(item.fragment_id for item in values)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(
            identities
        ):
            raise QuadricSectionCompositingError(
                "plane outline fragments must have unique sorted identities"
            )
        for edge_index in range(4):
            edge = tuple(
                sorted(
                    (item for item in values if item.edge_index == edge_index),
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
        object.__setattr__(self, "plane_outline_fragments", values)

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
        result = super().to_dict()
        result["planeOutlineFragments"] = [
            item.to_dict() for item in self.plane_outline_fragments
        ]
        return result


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
    ray_parameters: object,
    boundary_epsilon: float,
) -> PlaneDepthRole:
    parameters = ray_parameters(point)  # type: ignore[operator]
    if not parameters:
        return PlaneDepthRole.OUTSIDE_PROJECTION
    if min(parameters) > boundary_epsilon:
        return PlaneDepthRole.BEHIND_SURFACE
    if max(parameters) < -boundary_epsilon:
        return PlaneDepthRole.IN_FRONT_OF_SURFACE
    if min(parameters) <= boundary_epsilon and max(parameters) >= -boundary_epsilon:
        return PlaneDepthRole.BETWEEN_SURFACE_SHEETS
    raise QuadricSectionCompositingError(
        "surface ray endpoints cannot classify a plane outline point"
    )


def _compute_outline_fragments(
    surface: object,
    plane: object,
    patch: object,
    view: object,
    *,
    context: object,
    limits: QuadricSectionCompositingLimits,
) -> tuple[QuadricPlaneOutlineFragment, ...]:
    corners = np.asarray(patch.corners(plane), dtype=float)  # type: ignore[attr-defined]
    characteristic = tuple(surface.characteristic_points) + tuple(  # type: ignore[attr-defined]
        tuple(float(value) for value in point) for point in corners
    )
    resolved = resolve_geometry_context(context, positions=characteristic)
    boundary_epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    angular_epsilon = resolved.epsilon(GeometryQuantity.ANGULAR)
    parameter_epsilon = resolved.epsilon(GeometryQuantity.PARAMETER)
    ray_parameters = _impl._surface_ray_solver(
        surface,
        np.asarray(view.view_direction, dtype=float),  # type: ignore[attr-defined]
        boundary_epsilon=boundary_epsilon,
        angular_epsilon=angular_epsilon,
    )

    result: list[QuadricPlaneOutlineFragment] = []
    ends = (*corners[1:], corners[0])
    for edge_index, (start, end) in enumerate(zip(corners, ends)):
        curve = SegmentCurve(
            f"{plane.plane_id}:outline-edge:{edge_index}",  # type: ignore[attr-defined]
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
            screen_start = np.asarray(view.matrix[:2] @ world_start, dtype=float)  # type: ignore[attr-defined]
            screen_end = np.asarray(view.matrix[:2] @ world_end, dtype=float)  # type: ignore[attr-defined]
            result.append(
                QuadricPlaneOutlineFragment(
                    fragment_id=(
                        f"plane:{plane.plane_id}:outline:"  # type: ignore[attr-defined]
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
            if len(result) > limits.max_plane_fragments:
                raise QuadricSectionCompositingError(
                    "quadric section outline exceeds max_plane_fragments="
                    f"{limits.max_plane_fragments}"
                )
    return tuple(sorted(result, key=lambda item: item.fragment_id))


def compute_quadric_section_compositing(
    base_frame: QuadricCompositingFrame,
    surface: object,
    plane: object,
    patch: object,
    view: object,
    *,
    context: object = None,
    max_screen_error: float = 0.08,
    limits: QuadricSectionCompositingLimits = QUADRIC_SECTION_COMPOSITING_LIMITS,
) -> QuadricSectionCompositingFrame:
    """Build fill regions and exact depth-split plane-outline fragments."""

    base = _impl.compute_quadric_section_compositing(
        base_frame,
        surface,
        plane,
        patch,
        view,
        context=context,
        max_screen_error=max_screen_error,
        limits=limits,
    )
    outline_fragments = _compute_outline_fragments(
        surface,
        plane,
        patch,
        view,
        context=context,
        limits=limits,
    )
    items = _paint_items(base.surface_id, base.plane.plane_id)
    relations = [
        _impl.QuadricPaintRelation(far, near, "quadric_section_depth_layer")
        for far, near in zip(items.depth_chain, items.depth_chain[1:])
    ]
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
        _impl.QuadricPaintRelation(
            items.plane_outline,
            curve_id,
            "section_curve_overlay",
        )
        for curve_id in sorted(active_curve_ids)
    )
    normalized = _impl._dedupe_relations(relations)
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
        base_frame=base.base_frame,
        surface_id=base.surface_id,
        plane=base.plane,
        patch=base.patch,
        surface_proxy=base.surface_proxy,
        plane_fragments=base.plane_fragments,
        plane_outline_fragments=outline_fragments,
        paint_items=items,
        order_relations=normalized,
        draw_order=draw_order,
        max_screen_error=base.max_screen_error,
        ray_classification_count=base.ray_classification_count,
        schema=base.schema,
    )


def quadric_plane_fragment_contours(
    frame: QuadricSectionCompositingFrame,
) -> dict[PlaneDepthRole, tuple[tuple[tuple[float, float], ...], ...]]:
    return _impl.quadric_plane_fragment_contours(frame)


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


def __getattr__(name: str) -> object:
    return getattr(_impl, name)


__all__ = tuple(dict.fromkeys((*_impl.__all__, "QuadricPlaneOutlineFragment")))
