"""Renderer-neutral coordination of one finite open double-cone section.

The ordinary section compositor deliberately owns one convex surface.  An
``OPEN_DOUBLE`` teaching shell is already represented by two stable
``OPEN_SINGLE`` siblings, so this module combines two independently certified
local frames instead of teaching that local solver about multiple surfaces.

The complete projected contact set must be one point at the authored shared
apex.  Under that contract, local non-outside plane cells can be retained
verbatim and the common outside region is the first local outside partition
minus the second convex projection proxy.  The result paints the cutting plane
once, keeps both pairs of surface sheets, and supplies one deterministic
far-to-near painter graph.  Remote point contact, a nonzero coincident segment,
or positive-area nappe overlap fails explicitly.  When the finite cutting patch
projects rank-one, the same contact certificate is retained while the two local
near-side outline chains are merged as one finite scalar interval partition;
no plane fill is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from ..compositor import (
    CompositorCycleError,
    PainterConstraint,
    stable_topological_sort,
)
from ..path_compositing import segment_intersection_parameters
from ..topology import ParameterInterval, assert_exact_partition
from .compositing import QuadricPaintRelation
from .contract import ConeModel, ConeSpec, PlaneDisplayPatchSpec, SectionPlane
from .section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionEvidence,
    PlanePatchProjectionKind,
    QUADRIC_SECTION_COMPOSITING_LIMITS,
    QuadricPlaneFragment,
    QuadricPlaneOutlineFragment,
    QuadricSectionCompositingError,
    QuadricSectionCompositingFrame,
)


COMPOSITE_QUADRIC_SECTION_COMPOSITING_SCHEMA = (
    "manim-composite-quadric-section-compositing/v3"
)


class CompositeQuadricSectionCompositingError(ValueError):
    """Two local open-double section frames cannot be merged safely."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompositeQuadricSectionCompositingError(
            f"{label} must be a non-empty string"
        )
    return value.strip()


@dataclass(frozen=True, slots=True)
class CompositeSectionBranchLineage:
    """Link one physical nappe curve to its mathematical conic branch."""

    physical_curve_id: str
    mathematical_branch_id: str
    child_surface_id: str
    nappe_role: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "physical_curve_id",
            _identity(self.physical_curve_id, "physical_curve_id"),
        )
        object.__setattr__(
            self,
            "mathematical_branch_id",
            _identity(self.mathematical_branch_id, "mathematical_branch_id"),
        )
        object.__setattr__(
            self,
            "child_surface_id",
            _identity(self.child_surface_id, "child_surface_id"),
        )
        if self.nappe_role not in {"negative", "positive"}:
            raise CompositeQuadricSectionCompositingError(
                "nappe_role must be 'negative' or 'positive'"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "physicalCurveId": self.physical_curve_id,
            "mathematicalBranchId": self.mathematical_branch_id,
            "childSurfaceId": self.child_surface_id,
            "nappeRole": self.nappe_role,
        }


@dataclass(frozen=True, slots=True)
class CompositeSharedApexEvidence:
    """Certificate for the only contact allowed between the two siblings."""

    world_point: tuple[float, float, float]
    screen_point: tuple[float, float]
    projected_overlap_area: float
    boundary_tolerance: float
    contact_dimension: int = 0
    contact_extent: float = 0.0
    max_contact_distance_from_apex: float = 0.0
    contact_points: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        world = np.asarray(self.world_point, dtype=float)
        screen = np.asarray(self.screen_point, dtype=float)
        if world.shape != (3,) or not np.all(np.isfinite(world)):
            raise CompositeQuadricSectionCompositingError(
                "shared apex world_point must be finite three-dimensional"
            )
        if screen.shape != (2,) or not np.all(np.isfinite(screen)):
            raise CompositeQuadricSectionCompositingError(
                "shared apex screen_point must be finite two-dimensional"
            )
        if (
            isinstance(self.contact_dimension, bool)
            or not isinstance(self.contact_dimension, int)
            or self.contact_dimension not in {0, 1, 2}
        ):
            raise CompositeQuadricSectionCompositingError(
                "shared-apex contact_dimension must be 0, 1, or 2"
            )
        points = tuple(
            tuple(float(value) for value in point)
            for point in self.contact_points
        )
        if not points:
            points = (tuple(float(value) for value in screen),)
        point_array = np.asarray(points, dtype=float)
        if (
            not points
            or point_array.ndim != 2
            or point_array.shape[1] != 2
            or not np.all(np.isfinite(point_array))
        ):
            raise CompositeQuadricSectionCompositingError(
                "shared-apex contact_points must contain finite 2D points"
            )
        if (
            not isfinite(self.projected_overlap_area)
            or self.projected_overlap_area < 0.0
            or not isfinite(self.contact_extent)
            or self.contact_extent < 0.0
            or not isfinite(self.max_contact_distance_from_apex)
            or self.max_contact_distance_from_apex < 0.0
            or not isfinite(self.boundary_tolerance)
            or self.boundary_tolerance <= 0.0
        ):
            raise CompositeQuadricSectionCompositingError(
                "shared-apex evidence tolerances must be finite and valid"
            )
        computed_extent = max(
            (
                float(np.linalg.norm(right - left))
                for index, left in enumerate(point_array)
                for right in point_array[index + 1 :]
            ),
            default=0.0,
        )
        computed_max_distance = max(
            float(np.linalg.norm(point - screen)) for point in point_array
        )
        consistency_tolerance = max(
            np.finfo(float).eps
            * 256.0
            * max(computed_extent, computed_max_distance, 1.0),
            self.boundary_tolerance * 1.0e-9,
        )
        if (
            abs(self.contact_extent - computed_extent) > consistency_tolerance
            or abs(
                self.max_contact_distance_from_apex - computed_max_distance
            )
            > consistency_tolerance
        ):
            raise CompositeQuadricSectionCompositingError(
                "shared-apex contact metrics disagree with contact_points"
            )
        if self.contact_dimension != 0:
            raise CompositeQuadricSectionCompositingError(
                "certified shared-apex contact must be zero-dimensional"
            )
        if (
            self.contact_extent > self.boundary_tolerance
            or self.max_contact_distance_from_apex > self.boundary_tolerance
        ):
            raise CompositeQuadricSectionCompositingError(
                "certified contact must lie inside the shared-apex tolerance"
            )
        object.__setattr__(
            self,
            "world_point",
            tuple(float(value) for value in world),
        )
        object.__setattr__(
            self,
            "screen_point",
            tuple(float(value) for value in screen),
        )
        object.__setattr__(self, "contact_points", points)

    def to_dict(self) -> dict[str, object]:
        return {
            "worldPoint": list(self.world_point),
            "screenPoint": list(self.screen_point),
            "projectedOverlapArea": self.projected_overlap_area,
            "contactDimension": self.contact_dimension,
            "contactExtent": self.contact_extent,
            "maxContactDistanceFromApex": self.max_contact_distance_from_apex,
            "contactPoints": [list(point) for point in self.contact_points],
            "boundaryTolerance": self.boundary_tolerance,
        }


@dataclass(frozen=True, slots=True)
class CompositeSurfaceSheetItems:
    child_surface_id: str
    nappe_role: str
    surface_back: str
    surface_front: str

    def __post_init__(self) -> None:
        for name in ("child_surface_id", "surface_back", "surface_front"):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        if self.nappe_role not in {"negative", "positive"}:
            raise CompositeQuadricSectionCompositingError(
                "surface-sheet nappe_role must be 'negative' or 'positive'"
            )
        if self.surface_back == self.surface_front:
            raise CompositeQuadricSectionCompositingError(
                "surface back/front sheet identities must differ"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "childSurfaceId": self.child_surface_id,
            "nappeRole": self.nappe_role,
            "surfaceBack": self.surface_back,
            "surfaceFront": self.surface_front,
        }


@dataclass(frozen=True, slots=True)
class CompositeQuadricSectionPaintItems:
    plane_behind: str
    plane_outside: str
    plane_between: str
    plane_front: str
    plane_outline_behind: str
    plane_outline_outside: str
    plane_outline_between: str
    plane_outline_front: str
    surface_sheets: tuple[CompositeSurfaceSheetItems, ...]

    def __post_init__(self) -> None:
        for name in (
            "plane_behind",
            "plane_outside",
            "plane_between",
            "plane_front",
            "plane_outline_behind",
            "plane_outline_outside",
            "plane_outline_between",
            "plane_outline_front",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        if len(self.surface_sheets) != 2 or not all(
            isinstance(item, CompositeSurfaceSheetItems)
            for item in self.surface_sheets
        ):
            raise CompositeQuadricSectionCompositingError(
                "composite paint items require exactly two surface-sheet pairs"
            )
        keys = tuple(
            (item.nappe_role, item.child_surface_id) for item in self.surface_sheets
        )
        if keys != tuple(sorted(keys)) or {item.nappe_role for item in self.surface_sheets} != {
            "negative",
            "positive",
        }:
            raise CompositeQuadricSectionCompositingError(
                "surface-sheet pairs must be canonical negative/positive siblings"
            )
        if len(set(self.ordered)) != len(self.ordered):
            raise CompositeQuadricSectionCompositingError(
                "composite painter identities must be unique"
            )

    @property
    def fill_by_role(self) -> dict[PlaneDepthRole, str]:
        return {
            PlaneDepthRole.BEHIND_SURFACE: self.plane_behind,
            PlaneDepthRole.OUTSIDE_PROJECTION: self.plane_outside,
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS: self.plane_between,
            PlaneDepthRole.IN_FRONT_OF_SURFACE: self.plane_front,
        }

    @property
    def outline_by_role(self) -> dict[PlaneDepthRole, str]:
        return {
            PlaneDepthRole.BEHIND_SURFACE: self.plane_outline_behind,
            PlaneDepthRole.OUTSIDE_PROJECTION: self.plane_outline_outside,
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS: self.plane_outline_between,
            PlaneDepthRole.IN_FRONT_OF_SURFACE: self.plane_outline_front,
        }

    @property
    def ordered(self) -> tuple[str, ...]:
        backs = tuple(item.surface_back for item in self.surface_sheets)
        fronts = tuple(item.surface_front for item in self.surface_sheets)
        return (
            self.plane_behind,
            self.plane_outline_behind,
            *backs,
            self.plane_outside,
            self.plane_outline_outside,
            self.plane_between,
            self.plane_outline_between,
            *fronts,
            self.plane_front,
            self.plane_outline_front,
        )

    def sheet_for_surface(self, surface_id: str) -> CompositeSurfaceSheetItems:
        matches = tuple(
            item for item in self.surface_sheets if item.child_surface_id == surface_id
        )
        if len(matches) != 1:
            raise CompositeQuadricSectionCompositingError(
                f"no unique surface-sheet pair for {surface_id!r}"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "planeBehind": self.plane_behind,
            "planeOutside": self.plane_outside,
            "planeBetween": self.plane_between,
            "planeFront": self.plane_front,
            "planeOutlineBehind": self.plane_outline_behind,
            "planeOutlineOutside": self.plane_outline_outside,
            "planeOutlineBetween": self.plane_outline_between,
            "planeOutlineFront": self.plane_outline_front,
            "surfaceSheets": [item.to_dict() for item in self.surface_sheets],
        }


@dataclass(frozen=True, slots=True)
class CompositeQuadricSectionCompositingFrame:
    parent_surface_id: str
    section_id: str
    plane: SectionPlane
    patch: PlaneDisplayPatchSpec
    child_frames: tuple[QuadricSectionCompositingFrame, ...]
    patch_projection: PlanePatchProjectionEvidence
    plane_fragments: tuple[QuadricPlaneFragment, ...]
    plane_outline_fragments: tuple[QuadricPlaneOutlineFragment, ...]
    paint_items: CompositeQuadricSectionPaintItems
    branch_lineage: tuple[CompositeSectionBranchLineage, ...]
    shared_apex: CompositeSharedApexEvidence
    order_relations: tuple[QuadricPaintRelation, ...]
    draw_order: tuple[str, ...]
    max_screen_error: float
    schema: str = COMPOSITE_QUADRIC_SECTION_COMPOSITING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COMPOSITE_QUADRIC_SECTION_COMPOSITING_SCHEMA:
            raise CompositeQuadricSectionCompositingError(
                "invalid composite quadric-section schema"
            )
        object.__setattr__(
            self,
            "parent_surface_id",
            _identity(self.parent_surface_id, "parent_surface_id"),
        )
        object.__setattr__(self, "section_id", _identity(self.section_id, "section_id"))
        if not isinstance(self.plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")
        if not isinstance(self.patch, PlaneDisplayPatchSpec):
            raise TypeError("patch must be a PlaneDisplayPatchSpec")
        if self.patch.plane_id != self.plane.plane_id:
            raise CompositeQuadricSectionCompositingError(
                "composite patch plane_id does not match the cutting plane"
            )
        if not isfinite(self.max_screen_error) or self.max_screen_error <= 0.0:
            raise CompositeQuadricSectionCompositingError(
                "max_screen_error must be finite and positive"
            )
        if len(self.child_frames) != 2 or not all(
            isinstance(item, QuadricSectionCompositingFrame)
            for item in self.child_frames
        ):
            raise CompositeQuadricSectionCompositingError(
                "composite section frame requires exactly two local child frames"
            )
        child_ids = tuple(item.surface_id for item in self.child_frames)
        if child_ids != tuple(sorted(child_ids)) or len(set(child_ids)) != 2:
            raise CompositeQuadricSectionCompositingError(
                "local child frames must have unique sorted surface identities"
            )
        if any(item.plane != self.plane or item.patch != self.patch for item in self.child_frames):
            raise CompositeQuadricSectionCompositingError(
                "local child frames must share the exact plane and display patch"
            )
        if not isinstance(self.patch_projection, PlanePatchProjectionEvidence):
            raise TypeError(
                "patch_projection must be a PlanePatchProjectionEvidence"
            )
        child_projection_kinds = tuple(
            item.projection_kind for item in self.child_frames
        )
        if len(set(child_projection_kinds)) != 1:
            raise CompositeQuadricSectionCompositingError(
                "local child frames must share one projection topology"
            )
        if child_projection_kinds[0] is not self.patch_projection.kind:
            raise CompositeQuadricSectionCompositingError(
                "composite patch projection disagrees with its child frames"
            )
        if self.projection_kind is PlanePatchProjectionKind.LINE:
            certified_projection = _certify_matching_line_projection_evidence(
                self.child_frames,
                max(
                    np.finfo(float).eps * 8192.0,
                    self.max_screen_error * 1.0e-9,
                ),
            )
            if self.patch_projection != certified_projection:
                raise CompositeQuadricSectionCompositingError(
                    "composite LINE projection evidence is not the canonical "
                    "child evidence"
                )
        elif any(
            item.patch_projection != self.patch_projection
            for item in self.child_frames
        ):
            raise CompositeQuadricSectionCompositingError(
                "AREA child frames must share exact patch-projection evidence"
            )
        fragment_ids = tuple(item.fragment_id for item in self.plane_fragments)
        if fragment_ids != tuple(sorted(fragment_ids)) or len(set(fragment_ids)) != len(
            fragment_ids
        ):
            raise CompositeQuadricSectionCompositingError(
                "composite plane fragments must have unique sorted identities"
            )
        outline_ids = tuple(item.fragment_id for item in self.plane_outline_fragments)
        if outline_ids != tuple(sorted(outline_ids)) or len(set(outline_ids)) != len(
            outline_ids
        ):
            raise CompositeQuadricSectionCompositingError(
                "composite plane-outline fragments must have unique sorted identities"
            )
        if self.projection_kind is PlanePatchProjectionKind.LINE:
            if self.plane_fragments:
                raise CompositeQuadricSectionCompositingError(
                    "LINE composite plane projection cannot contain fill fragments"
                )
            _certify_line_outline_chain(
                self.patch_projection,
                self.plane_outline_fragments,
                max(
                    np.finfo(float).eps * 8192.0,
                    self.max_screen_error * 1.0e-9,
                ),
            )
        else:
            if not self.plane_fragments:
                raise CompositeQuadricSectionCompositingError(
                    "AREA composite plane projection requires fill fragments"
                )
            for edge_index in range(4):
                edge = tuple(
                    item
                    for item in self.plane_outline_fragments
                    if item.edge_index == edge_index
                )
                try:
                    assert_exact_partition(
                        ParameterInterval(0.0, 1.0),
                        (item.interval for item in edge),
                    )
                except ValueError as exc:
                    raise CompositeQuadricSectionCompositingError(
                        "composite outline fragments must cover every patch edge"
                    ) from exc
        lineage_keys = tuple(item.physical_curve_id for item in self.branch_lineage)
        if lineage_keys != tuple(sorted(lineage_keys)) or len(set(lineage_keys)) != len(
            lineage_keys
        ):
            raise CompositeQuadricSectionCompositingError(
                "branch lineage must use unique sorted physical curve identities"
            )
        active = {
            *self.paint_items.ordered,
            *(
                fragment.item_id
                for child in self.child_frames
                for fragment in child.base_frame.curve_fragments
                if fragment.painted
            ),
        }
        if len(self.draw_order) != len(active) or set(self.draw_order) != active:
            raise CompositeQuadricSectionCompositingError(
                "composite draw_order must cover every active painter item exactly once"
            )
        relation_keys = tuple(
            (item.far_item_id, item.near_item_id, item.reason)
            for item in self.order_relations
        )
        if relation_keys != tuple(sorted(relation_keys)):
            raise CompositeQuadricSectionCompositingError(
                "composite painter relations must be sorted"
            )
    @property
    def fragments_by_role(self) -> dict[PlaneDepthRole, tuple[QuadricPlaneFragment, ...]]:
        return {
            role: tuple(item for item in self.plane_fragments if item.role is role)
            for role in PlaneDepthRole
        }

    @property
    def projection_kind(self) -> PlanePatchProjectionKind:
        """Explicit AREA/LINE topology shared by both child frames."""

        return self.patch_projection.kind

    @property
    def has_plane_fill(self) -> bool:
        """Whether this composite frame owns two-dimensional plane fill."""

        return self.projection_kind is PlanePatchProjectionKind.AREA

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

    def child_frame(self, surface_id: str) -> QuadricSectionCompositingFrame:
        matches = tuple(item for item in self.child_frames if item.surface_id == surface_id)
        if len(matches) != 1:
            raise CompositeQuadricSectionCompositingError(
                f"no unique child frame for {surface_id!r}"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "parentSurfaceId": self.parent_surface_id,
            "sectionId": self.section_id,
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
            "childFrames": [item.to_dict() for item in self.child_frames],
            "patchProjection": self.patch_projection.to_dict(),
            "projectionKind": self.projection_kind.value,
            "hasPlaneFill": self.has_plane_fill,
            "planeFragments": [item.to_dict() for item in self.plane_fragments],
            "planeOutlineFragments": [
                item.to_dict() for item in self.plane_outline_fragments
            ],
            "paintItems": self.paint_items.to_dict(),
            "branchLineage": [item.to_dict() for item in self.branch_lineage],
            "sharedApex": self.shared_apex.to_dict(),
            "orderRelations": [item.to_dict() for item in self.order_relations],
            "drawOrder": list(self.draw_order),
            "maxScreenError": self.max_screen_error,
        }


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _signed_area(points: Sequence[np.ndarray]) -> float:
    return 0.5 * sum(
        _cross2(start, end)
        for start, end in zip(points, (*points[1:], points[0]))
    )


def _canonical_polygon(
    points: Sequence[Sequence[float]],
    epsilon: float,
) -> tuple[np.ndarray, ...]:
    values: list[np.ndarray] = []
    for raw in points:
        point = np.asarray(raw, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise CompositeQuadricSectionCompositingError(
                "projected partition vertices must be finite 2D points"
            )
        if not values or float(np.linalg.norm(point - values[-1])) > epsilon:
            values.append(point)
    if len(values) > 1 and float(np.linalg.norm(values[0] - values[-1])) <= epsilon:
        values.pop()
    changed = True
    while changed and len(values) >= 3:
        changed = False
        cleaned: list[np.ndarray] = []
        count = len(values)
        for index, point in enumerate(values):
            previous = values[(index - 1) % count]
            following = values[(index + 1) % count]
            baseline = following - previous
            scale = max(float(np.linalg.norm(baseline)), epsilon)
            if abs(_cross2(point - previous, baseline)) <= epsilon * scale:
                changed = True
                continue
            cleaned.append(point)
        values = cleaned
    if len(values) < 3:
        return ()
    area = _signed_area(values)
    if abs(area) <= epsilon * epsilon:
        return ()
    if area < 0.0:
        values.reverse()
    start = min(
        range(len(values)),
        key=lambda index: (
            round(float(values[index][0]) / epsilon),
            round(float(values[index][1]) / epsilon),
            float(values[index][0]),
            float(values[index][1]),
        ),
    )
    return tuple((*values[start:], *values[:start]))


def _clip_half_plane(
    polygon: Sequence[np.ndarray],
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    *,
    keep_inside: bool,
    epsilon: float,
) -> tuple[np.ndarray, ...]:
    if not polygon:
        return ()
    edge = edge_end - edge_start

    def signed(point: np.ndarray) -> float:
        value = _cross2(edge, point - edge_start)
        return value if keep_inside else -value

    result: list[np.ndarray] = []
    previous = np.asarray(polygon[-1], dtype=float)
    previous_value = signed(previous)
    previous_inside = previous_value >= -epsilon
    for raw_current in polygon:
        current = np.asarray(raw_current, dtype=float)
        current_value = signed(current)
        current_inside = current_value >= -epsilon
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > np.finfo(float).tiny:
                parameter = previous_value / denominator
                result.append(previous + parameter * (current - previous))
        if current_inside:
            result.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return _canonical_polygon(result, epsilon)


def _convex_intersection(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    epsilon: float,
) -> tuple[np.ndarray, ...]:
    result = _canonical_polygon(first, epsilon)
    clipper = _canonical_polygon(second, epsilon)
    for start, end in zip(clipper, (*clipper[1:], clipper[0])):
        result = _clip_half_plane(
            result,
            start,
            end,
            keep_inside=True,
            epsilon=epsilon,
        )
        if not result:
            break
    return result


@dataclass(frozen=True, slots=True)
class _ConvexContactSet:
    """Closed convex-polygon intersection, including rank-zero/one contact."""

    points: tuple[np.ndarray, ...]
    dimension: int
    extent: float
    area: float


def _point_in_convex_polygon(
    point: np.ndarray,
    polygon: Sequence[np.ndarray],
    epsilon: float,
) -> bool:
    return all(
        _cross2(end - start, point - start)
        >= -epsilon * max(float(np.linalg.norm(end - start)), epsilon)
        for start, end in zip(polygon, (*polygon[1:], polygon[0]))
    )


def _segment_contact_points(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, ...]:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    intersection = segment_intersection_parameters(
        first_start,
        first_end,
        second_start,
        second_end,
        epsilon,
    )
    if intersection is None:
        return ()
    kind, parameters = intersection
    if kind == "point":
        first_parameter, second_parameter = parameters
        first_point = first_start + first_parameter * first_direction
        second_point = second_start + second_parameter * second_direction
        return (0.5 * (first_point + second_point),)
    first_start_parameter, first_end_parameter, _, _ = parameters
    return (
        first_start + first_start_parameter * first_direction,
        first_start + first_end_parameter * first_direction,
    )


def _dedupe_contact_points(
    points: Sequence[np.ndarray],
    epsilon: float,
) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for point in sorted(
        (np.asarray(value, dtype=float) for value in points),
        key=lambda value: (float(value[0]), float(value[1])),
    ):
        if not any(
            float(np.linalg.norm(point - existing)) <= epsilon
            for existing in result
        ):
            result.append(point)
    return tuple(result)


def _contact_hull(
    points: Sequence[np.ndarray],
    epsilon: float,
) -> tuple[np.ndarray, ...]:
    values = _dedupe_contact_points(points, epsilon)
    if len(values) <= 1:
        return values

    def append_point(chain: list[np.ndarray], point: np.ndarray) -> None:
        while len(chain) >= 2:
            first, second = chain[-2], chain[-1]
            first_edge = second - first
            second_edge = point - second
            tolerance = epsilon * max(
                float(np.linalg.norm(first_edge)),
                float(np.linalg.norm(second_edge)),
                epsilon,
            )
            if _cross2(first_edge, second_edge) > tolerance:
                break
            chain.pop()
        chain.append(point)

    lower: list[np.ndarray] = []
    for point in values:
        append_point(lower, point)
    upper: list[np.ndarray] = []
    for point in reversed(values):
        append_point(upper, point)
    return tuple((*lower[:-1], *upper[:-1]))


def _convex_contact_set(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    *,
    epsilon: float,
    point_tolerance: float,
) -> _ConvexContactSet:
    candidates: list[np.ndarray] = []
    candidates.extend(
        point
        for point in first
        if _point_in_convex_polygon(point, second, epsilon)
    )
    candidates.extend(
        point
        for point in second
        if _point_in_convex_polygon(point, first, epsilon)
    )
    for first_start, first_end in zip(first, (*first[1:], first[0])):
        for second_start, second_end in zip(second, (*second[1:], second[0])):
            candidates.extend(
                _segment_contact_points(
                    first_start,
                    first_end,
                    second_start,
                    second_end,
                    epsilon,
                )
            )
    hull = _contact_hull(candidates, epsilon)
    if not hull:
        return _ConvexContactSet((), -1, 0.0, 0.0)
    farthest: tuple[np.ndarray, np.ndarray] | None = None
    extent = 0.0
    for index, first_point in enumerate(hull):
        for second_point in hull[index + 1 :]:
            distance = float(np.linalg.norm(second_point - first_point))
            if distance > extent:
                extent = distance
                farthest = (first_point, second_point)
    area = abs(_signed_area(hull)) if len(hull) >= 3 else 0.0
    if extent <= point_tolerance or farthest is None:
        dimension = 0
    else:
        start, end = farthest
        direction = end - start
        maximum_normal_distance = max(
            abs(_cross2(direction, point - start)) / extent for point in hull
        )
        dimension = 1 if maximum_normal_distance <= point_tolerance else 2
    return _ConvexContactSet(hull, dimension, extent, area)


def _point_to_polygon_boundary_distance(
    point: np.ndarray,
    polygon: Sequence[np.ndarray],
) -> float:
    distances = []
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        direction = end - start
        squared_length = float(np.dot(direction, direction))
        if squared_length <= np.finfo(float).tiny:
            distances.append(float(np.linalg.norm(point - start)))
            continue
        parameter = float(np.dot(point - start, direction) / squared_length)
        parameter = min(1.0, max(0.0, parameter))
        distances.append(
            float(np.linalg.norm(point - (start + parameter * direction)))
        )
    return min(distances, default=float("inf"))


def _subtract_convex(
    subject: Sequence[np.ndarray],
    clipper: Sequence[np.ndarray],
    epsilon: float,
) -> tuple[tuple[np.ndarray, ...], ...]:
    candidates = (_canonical_polygon(subject, epsilon),)
    outside: list[tuple[np.ndarray, ...]] = []
    boundary = _canonical_polygon(clipper, epsilon)
    for start, end in zip(boundary, (*boundary[1:], boundary[0])):
        next_candidates: list[tuple[np.ndarray, ...]] = []
        for candidate in candidates:
            exterior = _clip_half_plane(
                candidate,
                start,
                end,
                keep_inside=False,
                epsilon=epsilon,
            )
            interior = _clip_half_plane(
                candidate,
                start,
                end,
                keep_inside=True,
                epsilon=epsilon,
            )
            if exterior:
                outside.append(exterior)
            if interior:
                next_candidates.append(interior)
        candidates = tuple(next_candidates)
        if not candidates:
            break
    return tuple(outside)


def _role_at_outline_parameter(
    frame: QuadricSectionCompositingFrame,
    edge_index: int,
    parameter: float,
) -> PlaneDepthRole:
    matches = tuple(
        item.role
        for item in frame.plane_outline_fragments
        if item.edge_index == edge_index
        and item.interval.contains(parameter, tolerance=0.0)
    )
    if len(set(matches)) != 1:
        raise CompositeQuadricSectionCompositingError(
            "a local outline partition has no unique role at an interval midpoint"
        )
    return matches[0]


def _combined_role(
    first: PlaneDepthRole,
    second: PlaneDepthRole,
    *,
    label: str,
) -> PlaneDepthRole:
    active = tuple(
        role
        for role in (first, second)
        if role is not PlaneDepthRole.OUTSIDE_PROJECTION
    )
    if len(active) > 1:
        raise CompositeQuadricSectionCompositingError(
            f"open-double nappe projections overlap at {label}; "
            "interleaved multi-sheet ordering is outside this coordinator"
        )
    return active[0] if active else PlaneDepthRole.OUTSIDE_PROJECTION


def _certify_matching_line_projection_evidence(
    frames: Sequence[QuadricSectionCompositingFrame],
    tolerance: float,
) -> PlanePatchProjectionEvidence:
    """Require both local LINE frames to describe exactly one finite segment."""

    if len(frames) != 2 or any(
        item.projection_kind is not PlanePatchProjectionKind.LINE
        for item in frames
    ):
        raise CompositeQuadricSectionCompositingError(
            "LINE composite coordination requires two LINE child frames"
        )
    first = frames[0].patch_projection
    first_endpoints = np.asarray(
        (first.line_screen_start, first.line_screen_end),
        dtype=float,
    )
    scale = max(
        1.0,
        float(np.max(np.abs(first_endpoints))),
        *first.singular_values,
    )
    epsilon = max(
        float(tolerance),
        np.finfo(float).eps * 8192.0 * scale,
    )
    for frame in frames[1:]:
        evidence = frame.patch_projection
        endpoints = np.asarray(
            (evidence.line_screen_start, evidence.line_screen_end),
            dtype=float,
        )
        if (
            not np.allclose(
                np.asarray(evidence.singular_values, dtype=float),
                np.asarray(first.singular_values, dtype=float),
                rtol=np.finfo(float).eps * 8192.0,
                atol=epsilon,
            )
            or abs(evidence.rank_ratio - first.rank_ratio)
            > np.finfo(float).eps * 8192.0
            or evidence.rank_ratio_threshold != first.rank_ratio_threshold
        ):
            raise CompositeQuadricSectionCompositingError(
                "LINE child frames disagree on rank-one projection evidence"
            )
        if not np.allclose(
            endpoints,
            first_endpoints,
            rtol=np.finfo(float).eps * 8192.0,
            atol=epsilon,
        ):
            raise CompositeQuadricSectionCompositingError(
                "LINE child frames disagree on finite projection endpoints"
            )
    return first


def _line_projection_axis(
    evidence: PlanePatchProjectionEvidence,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    start = np.asarray(evidence.line_screen_start, dtype=float)
    end = np.asarray(evidence.line_screen_end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    line_tolerance = max(
        float(tolerance),
        np.finfo(float).eps * 8192.0 * max(1.0, length),
    )
    if length <= line_tolerance:
        raise CompositeQuadricSectionCompositingError(
            "LINE composite projection has no finite extent"
        )
    return start, direction / length, length, line_tolerance


@dataclass(frozen=True, slots=True)
class _LineOutlineRecord:
    fragment: QuadricPlaneOutlineFragment
    scalar_start: float
    scalar_end: float
    lower: float
    upper: float


def _certify_line_outline_chain(
    evidence: PlanePatchProjectionEvidence,
    fragments: Sequence[QuadricPlaneOutlineFragment],
    tolerance: float,
) -> tuple[_LineOutlineRecord, ...]:
    """Certify one finite, gap-free, non-overlapping rank-one outline chain."""

    start, axis, length, line_tolerance = _line_projection_axis(
        evidence,
        tolerance,
    )
    normal = np.asarray((-axis[1], axis[0]), dtype=float)
    records: list[_LineOutlineRecord] = []
    for fragment in fragments:
        if not isinstance(fragment, QuadricPlaneOutlineFragment):
            raise TypeError(
                "plane_outline_fragments must contain "
                "QuadricPlaneOutlineFragment"
            )
        screen_start = np.asarray(fragment.screen_start, dtype=float)
        screen_end = np.asarray(fragment.screen_end, dtype=float)
        if max(
            abs(float(np.dot(screen_start - start, normal))),
            abs(float(np.dot(screen_end - start, normal))),
        ) > line_tolerance:
            raise CompositeQuadricSectionCompositingError(
                "LINE composite outline fragments must share one screen line"
            )
        scalar_start = float(np.dot(screen_start - start, axis))
        scalar_end = float(np.dot(screen_end - start, axis))
        lower, upper = sorted((scalar_start, scalar_end))
        if (
            upper - lower <= line_tolerance
            or lower < -line_tolerance
            or upper > length + line_tolerance
        ):
            raise CompositeQuadricSectionCompositingError(
                "LINE composite outline fragment has invalid finite extent"
            )
        records.append(
            _LineOutlineRecord(
                fragment,
                scalar_start,
                scalar_end,
                max(0.0, lower),
                min(length, upper),
            )
        )
    if not records:
        raise CompositeQuadricSectionCompositingError(
            "LINE composite projection requires a finite outline chain"
        )
    records.sort(
        key=lambda item: (
            item.lower,
            item.upper,
            item.fragment.fragment_id,
        )
    )
    cursor = 0.0
    for record in records:
        if record.lower > cursor + line_tolerance:
            raise CompositeQuadricSectionCompositingError(
                "LINE composite outline chain has a finite gap"
            )
        if record.lower < cursor - line_tolerance:
            raise CompositeQuadricSectionCompositingError(
                "LINE composite outline chain contains positive-length "
                "duplicate drawing"
            )
        cursor = max(cursor, record.upper)
    if cursor < length - line_tolerance:
        raise CompositeQuadricSectionCompositingError(
            "LINE composite outline chain does not reach its finite endpoint"
        )
    return tuple(records)


def _record_at_scalar(
    records: Sequence[_LineOutlineRecord],
    scalar: float,
) -> _LineOutlineRecord:
    matches = tuple(
        item for item in records if item.lower < scalar < item.upper
    )
    if len(matches) != 1:
        raise CompositeQuadricSectionCompositingError(
            "LINE child outline has no unique interval at a scalar midpoint"
        )
    return matches[0]


def _sample_line_record(
    record: _LineOutlineRecord,
    scalar: float,
    tolerance: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    denominator = record.scalar_end - record.scalar_start
    if abs(denominator) <= tolerance:
        raise CompositeQuadricSectionCompositingError(
            "LINE outline interval cannot be lifted back to its finite edge"
        )
    parameter = (float(scalar) - record.scalar_start) / denominator
    if parameter < -tolerance or parameter > 1.0 + tolerance:
        raise CompositeQuadricSectionCompositingError(
            "LINE outline scalar lies outside its certified source interval"
        )
    parameter = min(1.0, max(0.0, parameter))
    fragment = record.fragment
    world_start = np.asarray(fragment.world_start, dtype=float)
    world_end = np.asarray(fragment.world_end, dtype=float)
    screen_start = np.asarray(fragment.screen_start, dtype=float)
    screen_end = np.asarray(fragment.screen_end, dtype=float)
    world = world_start + parameter * (world_end - world_start)
    screen = screen_start + parameter * (screen_end - screen_start)
    edge_parameter = fragment.interval.start + parameter * fragment.interval.length
    return edge_parameter, world, screen


def _combined_line_outline_fragments(
    frames: tuple[QuadricSectionCompositingFrame, ...],
    epsilon: float,
) -> tuple[QuadricPlaneOutlineFragment, ...]:
    """Merge two child near-envelope chains as one scalar line partition."""

    evidence = _certify_matching_line_projection_evidence(frames, epsilon)
    line_tolerance = max(
        epsilon * 64.0,
        max(item.max_screen_error for item in frames) * 1.0e-9,
    )
    records_by_child = tuple(
        _certify_line_outline_chain(
            frame.patch_projection,
            frame.plane_outline_fragments,
            line_tolerance,
        )
        for frame in frames
    )
    _start, _axis, length, line_tolerance = _line_projection_axis(
        evidence,
        line_tolerance,
    )
    raw_breaks = [
        0.0,
        length,
        *(
            value
            for records in records_by_child
            for record in records
            for value in (record.lower, record.upper)
        ),
    ]
    breaks: list[float] = []
    for value in sorted(min(length, max(0.0, float(item))) for item in raw_breaks):
        if not breaks or value - breaks[-1] > line_tolerance:
            breaks.append(value)
        else:
            breaks[-1] = 0.5 * (breaks[-1] + value)
    if breaks:
        breaks[0] = 0.0
        breaks[-1] = length

    plane = frames[0].plane
    world_tolerance = max(
        epsilon * 64.0,
        np.finfo(float).eps
        * 8192.0
        * max(
            1.0,
            *(abs(value) for value in plane.point),
            *(frames[0].patch.half_width, frames[0].patch.half_height),
        ),
    )
    result: list[QuadricPlaneOutlineFragment] = []
    for scalar_start, scalar_end in zip(breaks, breaks[1:]):
        if scalar_end - scalar_start <= line_tolerance:
            continue
        midpoint = 0.5 * (scalar_start + scalar_end)
        first_record = _record_at_scalar(records_by_child[0], midpoint)
        second_record = _record_at_scalar(records_by_child[1], midpoint)
        role = _combined_role(
            first_record.fragment.role,
            second_record.fragment.role,
            label=(
                "the rank-one cutting-plane interval "
                f"[{scalar_start:.9g}, {scalar_end:.9g}]"
            ),
        )
        if first_record.fragment.edge_index != second_record.fragment.edge_index:
            raise CompositeQuadricSectionCompositingError(
                "LINE child outline chains select different finite patch edges"
            )
        first_samples = (
            _sample_line_record(first_record, scalar_start, line_tolerance),
            _sample_line_record(first_record, scalar_end, line_tolerance),
        )
        second_samples = (
            _sample_line_record(second_record, scalar_start, line_tolerance),
            _sample_line_record(second_record, scalar_end, line_tolerance),
        )
        if any(
            float(np.linalg.norm(first_value[1] - second_value[1]))
            > world_tolerance
            for first_value, second_value in zip(first_samples, second_samples)
        ):
            raise CompositeQuadricSectionCompositingError(
                "LINE child outline chains disagree in finite world geometry"
            )
        first_parameter, first_world, first_screen = first_samples[0]
        second_parameter, second_world, second_screen = first_samples[1]
        if first_parameter > second_parameter:
            first_parameter, second_parameter = second_parameter, first_parameter
            first_world, second_world = second_world, first_world
            first_screen, second_screen = second_screen, first_screen
        result.append(
            QuadricPlaneOutlineFragment(
                fragment_id=(
                    f"composite-plane:{plane.plane_id}:line:"
                    f"span:{len(result):04d}:{role.value}"
                ),
                role=role,
                edge_index=first_record.fragment.edge_index,
                interval=ParameterInterval(first_parameter, second_parameter),
                world_start=tuple(float(item) for item in first_world),
                world_end=tuple(float(item) for item in second_world),
                screen_start=tuple(float(item) for item in first_screen),
                screen_end=tuple(float(item) for item in second_screen),
            )
        )
    result_tuple = tuple(sorted(result, key=lambda item: item.fragment_id))
    _certify_line_outline_chain(evidence, result_tuple, line_tolerance)
    return result_tuple


def _combined_outline_fragments(
    frames: tuple[QuadricSectionCompositingFrame, ...],
    projection: np.ndarray,
    epsilon: float,
) -> tuple[QuadricPlaneOutlineFragment, ...]:
    plane = frames[0].plane
    patch = frames[0].patch
    corners = tuple(np.asarray(item, dtype=float) for item in patch.corners(plane))
    result: list[QuadricPlaneOutlineFragment] = []
    for edge_index, (world_start, world_end) in enumerate(
        zip(corners, (*corners[1:], corners[0]))
    ):
        values = [0.0, 1.0]
        for frame in frames:
            values.extend(
                value
                for item in frame.plane_outline_fragments
                if item.edge_index == edge_index
                for value in (item.interval.start, item.interval.end)
            )
        canonical: list[float] = []
        for value in sorted(values):
            value = min(1.0, max(0.0, float(value)))
            if not canonical or value - canonical[-1] > epsilon:
                canonical.append(value)
        runs: list[tuple[float, float, PlaneDepthRole]] = []
        for start, end in zip(canonical, canonical[1:]):
            if end - start <= epsilon:
                continue
            midpoint = 0.5 * (start + end)
            role = _combined_role(
                _role_at_outline_parameter(frames[0], edge_index, midpoint),
                _role_at_outline_parameter(frames[1], edge_index, midpoint),
                label=f"plane edge {edge_index}",
            )
            if runs and runs[-1][2] is role and start - runs[-1][1] <= epsilon:
                runs[-1] = (runs[-1][0], end, role)
            else:
                runs.append((start, end, role))
        for run_index, (start, end, role) in enumerate(runs):
            first = world_start + start * (world_end - world_start)
            second = world_start + end * (world_end - world_start)
            screen_first = projection[:2] @ first
            screen_second = projection[:2] @ second
            result.append(
                QuadricPlaneOutlineFragment(
                    fragment_id=(
                        f"composite-plane:{plane.plane_id}:edge:{edge_index}:"
                        f"span:{run_index:04d}:{role.value}"
                    ),
                    role=role,
                    edge_index=edge_index,
                    interval=ParameterInterval(start, end),
                    world_start=tuple(float(item) for item in first),
                    world_end=tuple(float(item) for item in second),
                    screen_start=tuple(float(item) for item in screen_first),
                    screen_end=tuple(float(item) for item in screen_second),
                )
            )
    return tuple(sorted(result, key=lambda item: item.fragment_id))


def _combined_area_plane_geometry(
    frames: tuple[QuadricSectionCompositingFrame, ...],
    proxies: tuple[tuple[np.ndarray, ...], ...],
    projection: np.ndarray,
    epsilon: float,
    area_tolerance: float,
    max_plane_fragments: int,
) -> tuple[
    tuple[QuadricPlaneFragment, ...],
    tuple[QuadricPlaneOutlineFragment, ...],
]:
    """Preserve the v2 positive-area partition path byte-for-byte in intent."""

    polygon_records: list[
        tuple[PlaneDepthRole, str, int, tuple[np.ndarray, ...]]
    ] = []
    first, second = frames
    for fragment in first.plane_fragments:
        polygon = _canonical_polygon(fragment.screen_vertices, epsilon)
        if not polygon:
            continue
        if fragment.role is PlaneDepthRole.OUTSIDE_PROJECTION:
            pieces = _subtract_convex(polygon, proxies[1], epsilon)
            polygon_records.extend(
                (
                    fragment.role,
                    f"{fragment.fragment_id}:minus-second:{index:04d}",
                    fragment.subdivision_depth,
                    piece,
                )
                for index, piece in enumerate(pieces)
            )
        else:
            intersection = _convex_intersection(
                polygon,
                proxies[1],
                epsilon,
            )
            if (
                intersection
                and abs(_signed_area(intersection)) > area_tolerance
            ):
                raise CompositeQuadricSectionCompositingError(
                    "positive-area nappe role overlap survived proxy certification"
                )
            polygon_records.append(
                (
                    fragment.role,
                    fragment.fragment_id,
                    fragment.subdivision_depth,
                    polygon,
                )
            )
    polygon_records.extend(
        (
            fragment.role,
            fragment.fragment_id,
            fragment.subdivision_depth,
            _canonical_polygon(fragment.screen_vertices, epsilon),
        )
        for fragment in second.plane_fragments
        if fragment.role is not PlaneDepthRole.OUTSIDE_PROJECTION
    )
    polygon_records = [item for item in polygon_records if item[3]]
    polygon_records.sort(
        key=lambda item: (
            item[0].value,
            item[1],
            tuple(
                (
                    round(float(point[0]) / epsilon),
                    round(float(point[1]) / epsilon),
                )
                for point in item[3]
            ),
        )
    )

    plane = frames[0].plane
    patch = frames[0].patch
    plane_u, plane_v, _normal = plane.basis
    plane_axes = np.column_stack((plane_u, plane_v))
    screen_origin = projection[:2] @ np.asarray(plane.point, dtype=float)
    screen_basis = projection[:2] @ plane_axes
    determinant = float(np.linalg.det(screen_basis))
    basis_scale = max(
        float(np.linalg.norm(screen_basis, ord=2)),
        np.finfo(float).tiny,
    )
    if abs(determinant) <= 1.0e-12 * basis_scale * basis_scale:
        raise CompositeQuadricSectionCompositingError(
            "cutting plane projects edge-on and has no sortable display area"
        )
    inverse = np.linalg.inv(screen_basis)

    fragments: list[QuadricPlaneFragment] = []
    triangle_index = 0
    for role, token, depth, polygon in polygon_records:
        del token
        for index in range(1, len(polygon) - 1):
            screen_triangle = (polygon[0], polygon[index], polygon[index + 1])
            if abs(_signed_area(screen_triangle)) <= area_tolerance:
                continue
            world_triangle = tuple(
                np.asarray(plane.point, dtype=float)
                + plane_axes @ (inverse @ (point - screen_origin))
                for point in screen_triangle
            )
            fragments.append(
                QuadricPlaneFragment(
                    fragment_id=(
                        f"composite-plane:{plane.plane_id}:cell:"
                        f"{triangle_index:06d}:{role.value}"
                    ),
                    role=role,
                    world_vertices=tuple(
                        tuple(float(value) for value in point)
                        for point in world_triangle
                    ),  # type: ignore[arg-type]
                    screen_vertices=tuple(
                        tuple(float(value) for value in point)
                        for point in screen_triangle
                    ),  # type: ignore[arg-type]
                    subdivision_depth=depth,
                )
            )
            triangle_index += 1
    fragments.sort(key=lambda item: item.fragment_id)
    if not fragments:
        raise CompositeQuadricSectionCompositingError(
            "composite plane partition has no positive-area fragments"
        )
    if len(fragments) > max_plane_fragments:
        raise CompositeQuadricSectionCompositingError(
            f"composite plane fragment count {len(fragments)} exceeds "
            f"capacity {max_plane_fragments}"
        )

    projected_patch = _canonical_polygon(
        tuple(
            projection[:2] @ np.asarray(point, dtype=float)
            for point in patch.corners(plane)
        ),
        epsilon,
    )
    expected_patch_area = abs(_signed_area(projected_patch))
    fragment_area = sum(
        abs(
            _signed_area(
                tuple(
                    np.asarray(point, dtype=float)
                    for point in fragment.screen_vertices
                )
            )
        )
        for fragment in fragments
    )
    partition_tolerance = max(
        area_tolerance * max(16, 4 * len(fragments)),
        expected_patch_area * 5.0e-9,
    )
    if (
        expected_patch_area <= area_tolerance
        or abs(fragment_area - expected_patch_area) > partition_tolerance
    ):
        raise CompositeQuadricSectionCompositingError(
            "composite plane partition does not conserve the projected patch area"
        )
    return (
        tuple(fragments),
        _combined_outline_fragments(frames, projection, epsilon),
    )


def _dedupe_relations(
    values: Sequence[QuadricPaintRelation],
) -> tuple[QuadricPaintRelation, ...]:
    by_key = {
        (item.far_item_id, item.near_item_id, item.reason): item for item in values
    }
    return tuple(by_key[key] for key in sorted(by_key))


def compute_composite_quadric_section_compositing(
    parent_surface: ConeSpec,
    section_id: str,
    child_frames: Sequence[QuadricSectionCompositingFrame],
    branch_lineage: Sequence[CompositeSectionBranchLineage] = (),
    *,
    max_plane_fragments: int = (
        QUADRIC_SECTION_COMPOSITING_LIMITS.max_plane_fragments
    ),
) -> CompositeQuadricSectionCompositingFrame:
    """Merge two certified local section frames for one ``OPEN_DOUBLE`` shell."""

    if not isinstance(parent_surface, ConeSpec):
        raise TypeError("parent_surface must be a ConeSpec")
    if parent_surface.model is not ConeModel.OPEN_DOUBLE:
        raise CompositeQuadricSectionCompositingError(
            "composite section coordination requires ConeModel.OPEN_DOUBLE"
        )
    if (
        isinstance(max_plane_fragments, bool)
        or not isinstance(max_plane_fragments, int)
        or max_plane_fragments <= 0
    ):
        raise CompositeQuadricSectionCompositingError(
            "max_plane_fragments must be a positive integer"
        )
    identity = _identity(section_id, "section_id")
    frames = tuple(sorted(child_frames, key=lambda item: item.surface_id))
    if len(frames) != 2 or not all(
        isinstance(item, QuadricSectionCompositingFrame) for item in frames
    ):
        raise CompositeQuadricSectionCompositingError(
            "exactly two local section frames are required"
        )
    expected_children = parent_surface.render_components
    expected_ids = tuple(item.surface_id for item in expected_children)
    if tuple(item.surface_id for item in frames) != expected_ids:
        raise CompositeQuadricSectionCompositingError(
            "local section frames are not the canonical siblings of the parent"
        )
    plane = frames[0].plane
    patch = frames[0].patch
    if any(item.plane != plane or item.patch != patch for item in frames):
        raise CompositeQuadricSectionCompositingError(
            "both local frames must use one identical plane and display patch"
        )
    projection_kinds = tuple(item.projection_kind for item in frames)
    if len(set(projection_kinds)) != 1:
        raise CompositeQuadricSectionCompositingError(
            "both local frames must share one projection topology"
        )
    projection_kind = projection_kinds[0]
    projection = np.asarray(
        frames[0].base_frame.visibility.projection_matrix,
        dtype=float,
    )
    if any(
        not np.array_equal(
            projection,
            np.asarray(item.base_frame.visibility.projection_matrix, dtype=float),
        )
        for item in frames[1:]
    ):
        raise CompositeQuadricSectionCompositingError(
            "both local frames must use one identical parallel projection"
        )
    scale = max(
        patch.half_width,
        patch.half_height,
        *(abs(value) for value in patch.center_coordinates),
        1.0,
    )
    epsilon = max(np.finfo(float).eps * 8192.0 * scale, 1.0e-12 * scale)
    area_tolerance = max(
        epsilon * epsilon,
        1.0e-12 * patch.half_width * patch.half_height,
    )
    patch_projection = frames[0].patch_projection
    if projection_kind is PlanePatchProjectionKind.LINE:
        patch_projection = _certify_matching_line_projection_evidence(
            frames,
            epsilon,
        )

    proxies = tuple(
        _canonical_polygon(item.surface_proxy.boundary_points, epsilon)
        for item in frames
    )
    if any(len(item) < 3 for item in proxies):
        raise CompositeQuadricSectionCompositingError(
            "each local surface proxy must retain positive display area"
        )
    apex = np.asarray(parent_surface.apex, dtype=float)
    screen_apex = projection[:2] @ apex
    apex_tolerance = max(epsilon * 64.0, 1.0e-9 * scale)
    contact = _convex_contact_set(
        proxies[0],
        proxies[1],
        epsilon=epsilon,
        point_tolerance=epsilon,
    )
    if contact.area > area_tolerance:
        raise CompositeQuadricSectionCompositingError(
            "open-double nappe projections have positive-area overlap; "
            "interleaved multi-sheet ordering is outside this coordinator"
        )
    for frame, proxy in zip(frames, proxies):
        if (
            _point_to_polygon_boundary_distance(screen_apex, proxy)
            > apex_tolerance
        ):
            raise CompositeQuadricSectionCompositingError(
                f"local proxy {frame.surface_id!r} does not own the shared apex"
            )
    if not contact.points:
        raise CompositeQuadricSectionCompositingError(
            "open-double nappe projections have no certifiable shared-apex contact"
        )
    if contact.dimension == 1:
        raise CompositeQuadricSectionCompositingError(
            "open-double nappe projections share a nonzero-length contact "
            f"segment (extent {contact.extent:.9g}); only point contact at the "
            "shared apex is supported"
        )
    if contact.dimension == 2:
        raise CompositeQuadricSectionCompositingError(
            "open-double nappe projection contact is two-dimensional even "
            "though its area lies below the positive-overlap tolerance"
        )
    max_contact_distance = max(
        float(np.linalg.norm(point - screen_apex)) for point in contact.points
    )
    if max_contact_distance > apex_tolerance:
        raise CompositeQuadricSectionCompositingError(
            "open-double nappe projection contact lies away from the shared "
            f"apex (maximum distance {max_contact_distance:.9g})"
        )
    shared_apex = CompositeSharedApexEvidence(
        world_point=tuple(float(item) for item in apex),
        screen_point=tuple(float(item) for item in screen_apex),
        projected_overlap_area=contact.area,
        contact_dimension=contact.dimension,
        contact_extent=contact.extent,
        max_contact_distance_from_apex=max_contact_distance,
        contact_points=tuple(
            tuple(float(value) for value in point) for point in contact.points
        ),
        boundary_tolerance=apex_tolerance,
    )

    if projection_kind is PlanePatchProjectionKind.LINE:
        fragments: tuple[QuadricPlaneFragment, ...] = ()
        outline = _combined_line_outline_fragments(frames, epsilon)
    else:
        fragments, outline = _combined_area_plane_geometry(
            frames,
            proxies,
            projection,
            epsilon,
            area_tolerance,
            max_plane_fragments,
        )
    local_items = frames[0].paint_items
    surface_sheets = tuple(
        CompositeSurfaceSheetItems(
            child.surface_id,
            child.surface_id.rsplit(":", 1)[-1],
            frame.paint_items.surface_back,
            frame.paint_items.surface_front,
        )
        for child, frame in zip(expected_children, frames)
    )
    paint_items = CompositeQuadricSectionPaintItems(
        plane_behind=local_items.plane_behind,
        plane_outside=local_items.plane_outside,
        plane_between=local_items.plane_between,
        plane_front=local_items.plane_front,
        plane_outline_behind=local_items.plane_outline_behind,
        plane_outline_outside=local_items.plane_outline_outside,
        plane_outline_between=local_items.plane_outline_between,
        plane_outline_front=local_items.plane_outline,
        surface_sheets=surface_sheets,
    )
    relations = _dedupe_relations(
        tuple(
            relation
            for frame in frames
            for relation in frame.order_relations
        )
    )
    active = {
        *paint_items.ordered,
        *(
            fragment.item_id
            for frame in frames
            for fragment in frame.base_frame.curve_fragments
            if fragment.painted
        ),
    }
    try:
        draw_order = stable_topological_sort(
            sorted(active),
            (
                PainterConstraint(item.far_item_id, item.near_item_id)
                for item in relations
            ),
            key=lambda item_id: item_id,
        )
    except CompositorCycleError as exc:
        raise CompositeQuadricSectionCompositingError(
            "composite open-double painter graph contains a cycle: "
            + ", ".join(sorted(str(item) for item in exc.unresolved))
        ) from exc
    raw_lineages = tuple(branch_lineage)
    if not all(
        isinstance(item, CompositeSectionBranchLineage) for item in raw_lineages
    ):
        raise TypeError("branch_lineage must contain CompositeSectionBranchLineage values")
    lineages = tuple(
        sorted(raw_lineages, key=lambda item: item.physical_curve_id)
    )
    expected_role_by_child = {
        child.surface_id: child.surface_id.rsplit(":", 1)[-1]
        for child in expected_children
    }
    for lineage in lineages:
        expected_role = expected_role_by_child.get(lineage.child_surface_id)
        if expected_role is None or lineage.nappe_role != expected_role:
            raise CompositeQuadricSectionCompositingError(
                "branch lineage must reference one canonical child nappe"
            )
        if not lineage.physical_curve_id.startswith(
            f"{identity}:nappe:{lineage.nappe_role}:"
        ):
            raise CompositeQuadricSectionCompositingError(
                "physical branch lineage does not belong to the composite section"
            )
    return CompositeQuadricSectionCompositingFrame(
        parent_surface_id=parent_surface.surface_id,
        section_id=identity,
        plane=plane,
        patch=patch,
        child_frames=frames,
        patch_projection=patch_projection,
        plane_fragments=tuple(fragments),
        plane_outline_fragments=outline,
        paint_items=paint_items,
        branch_lineage=lineages,
        shared_apex=shared_apex,
        order_relations=relations,
        draw_order=draw_order,
        max_screen_error=max(item.max_screen_error for item in frames),
    )


def canonical_composite_quadric_section_compositing_json(
    frame: CompositeQuadricSectionCompositingFrame,
) -> str:
    if not isinstance(frame, CompositeQuadricSectionCompositingFrame):
        raise TypeError("frame must be a CompositeQuadricSectionCompositingFrame")
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "COMPOSITE_QUADRIC_SECTION_COMPOSITING_SCHEMA",
    "CompositeQuadricSectionCompositingError",
    "CompositeQuadricSectionCompositingFrame",
    "CompositeQuadricSectionPaintItems",
    "CompositeSectionBranchLineage",
    "CompositeSharedApexEvidence",
    "CompositeSurfaceSheetItems",
    "canonical_composite_quadric_section_compositing_json",
    "compute_composite_quadric_section_compositing",
]
