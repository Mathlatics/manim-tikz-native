"""Certified teaching-transparent surface layers for Dandelin diagrams.

The ordinary quadric compositor cannot sort a cone and its Dandelin sphere as
two whole surfaces: they are nested and share a one-dimensional tangent set.
This module coordinates existing certified pieces instead of weakening that
contract.  Cone back/front sheets come from ``build_cone_projection_layers``;
the cutting plane comes from the section compositor; and each sphere is
inserted between the two sheets of its uniquely authenticated cone component.

The result is authoritative for a transparent *teaching* painter order.  It is
not an optical model and deliberately does not claim physical surface
visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import acos, atan2, ceil, floor, isfinite, tau
from typing import Sequence

import numpy as np

from ..compositor import (
    CompositorCycleError,
    PainterConstraint,
    stable_topological_sort,
)
from ..geometry import GeometryQuantity
from ..parallel_solver import ParallelView
from ..topology import (
    ParameterInterval,
    assert_exact_partition,
    partition_parameter_domain,
)
from .composite_section import (
    CompositeQuadricSectionCompositingError,
    CompositeQuadricSectionCompositingFrame,
    compute_composite_quadric_section_compositing,
)
from .compositing import QuadricPaintPolicy, QuadricPaintRelation
from .contract import ConeModel, PlaneDisplayPatchSpec
from .dandelin import DandelinConstruction3D
from .dandelin_visibility import certify_dandelin_tangent_contacts
from .global_occlusion import (
    GlobalQuadricOcclusionError,
    compute_global_quadric_frame,
)
from .projection import (
    ConeProjectionLayers,
    OpaqueProjectionProxy,
    ProjectionProxyError,
    build_cone_projection_layers,
    build_opaque_projection_proxy,
)
from .section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionKind,
    QuadricPlaneOutlineFragment,
    QuadricSectionCompositingError,
    QuadricSectionCompositingFrame,
    compute_quadric_section_compositing,
    merge_quadric_plane_fragment_contours,
)


DANDELIN_SURFACE_LAYER_FRAME_SCHEMA = "manim-dandelin-surface-layer-frame/v1"
_DEFAULT_MAX_SCREEN_ERROR = 0.16
_DEFAULT_MAX_SEGMENTS = 8192


class DandelinSurfaceCompositingError(ValueError):
    """A teaching-transparent Dandelin painter frame cannot be certified."""


class DandelinPlanePosition(str, Enum):
    """Position of the cutting plane relative to a tangent sphere."""

    IN_FRONT_OF_SPHERE = "in_front_of_sphere"
    BEHIND_SPHERE = "behind_sphere"


class DandelinContactSheet(str, Enum):
    """Cone/sphere sheet owning one open interval of the tangent circle."""

    BACK = "back"
    FRONT = "front"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DandelinSurfaceCompositingError(
            f"{label} must be a non-empty string"
        )
    return value.strip()


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise DandelinSurfaceCompositingError(
            f"{label} must be finite and positive"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinSurfaceCompositingError(
            f"{label} must be finite and positive"
        ) from exc
    if not isfinite(result) or result <= 0.0:
        raise DandelinSurfaceCompositingError(
            f"{label} must be finite and positive"
        )
    return result


def _path2(
    value: Sequence[Sequence[float]],
    label: str,
    *,
    minimum: int,
) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for raw in value:
        point = np.asarray(raw, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise DandelinSurfaceCompositingError(
                f"{label} must contain finite 2D points"
            )
        result.append((float(point[0]), float(point[1])))
    if len(result) < minimum:
        raise DandelinSurfaceCompositingError(
            f"{label} requires at least {minimum} points"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DandelinConeLayer:
    """One finite cone component represented by certified far/near sheets."""

    surface_id: str
    back_item_id: str
    front_item_id: str
    projection_layers: ConeProjectionLayers

    def __post_init__(self) -> None:
        surface_id = _identity(self.surface_id, "cone surface_id")
        back = _identity(self.back_item_id, "cone back_item_id")
        front = _identity(self.front_item_id, "cone front_item_id")
        if back == front:
            raise DandelinSurfaceCompositingError(
                "cone back/front painter identities must differ"
            )
        if not isinstance(self.projection_layers, ConeProjectionLayers):
            raise TypeError("projection_layers must be ConeProjectionLayers")
        if self.projection_layers.surface_id != surface_id:
            raise DandelinSurfaceCompositingError(
                "cone layer surface identity disagrees with projection evidence"
            )
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "back_item_id", back)
        object.__setattr__(self, "front_item_id", front)

    def to_dict(self) -> dict[str, object]:
        layers = self.projection_layers

        def paths(values: Sequence[Sequence[Sequence[float]]]) -> list[object]:
            return [[list(point) for point in path] for path in values]

        return {
            "surfaceId": self.surface_id,
            "backItemId": self.back_item_id,
            "frontItemId": self.front_item_id,
            "proxy": layers.proxy.to_dict(),
            "back": {
                "lateralPaths": paths(layers.back.lateral_paths),
                "capPaths": paths(layers.back.cap_paths),
            },
            "front": {
                "lateralPaths": paths(layers.front.lateral_paths),
                "capPaths": paths(layers.front.cap_paths),
            },
        }


@dataclass(frozen=True, slots=True)
class DandelinSphereLayer:
    """One sphere fill inserted inside its uniquely certified cone component."""

    sphere_id: str
    owner_cone_surface_id: str
    item_id: str
    proxy: OpaqueProjectionProxy
    plane_position: DandelinPlanePosition
    plane_ray_parameter: float

    def __post_init__(self) -> None:
        sphere_id = _identity(self.sphere_id, "sphere_id")
        owner = _identity(self.owner_cone_surface_id, "owner_cone_surface_id")
        item_id = _identity(self.item_id, "sphere item_id")
        if not isinstance(self.proxy, OpaqueProjectionProxy):
            raise TypeError("sphere proxy must be an OpaqueProjectionProxy")
        if self.proxy.surface_id != sphere_id:
            raise DandelinSurfaceCompositingError(
                "sphere layer identity disagrees with its projection proxy"
            )
        if not isinstance(self.plane_position, DandelinPlanePosition):
            raise TypeError("plane_position must be a DandelinPlanePosition")
        parameter = float(self.plane_ray_parameter)
        if not isfinite(parameter) or parameter == 0.0:
            raise DandelinSurfaceCompositingError(
                "sphere/plane ray parameter must be finite and non-zero"
            )
        if (
            parameter > 0.0
        ) != (
            self.plane_position is DandelinPlanePosition.IN_FRONT_OF_SPHERE
        ):
            raise DandelinSurfaceCompositingError(
                "sphere plane-position label disagrees with its ray parameter"
            )
        object.__setattr__(self, "sphere_id", sphere_id)
        object.__setattr__(self, "owner_cone_surface_id", owner)
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "plane_ray_parameter", parameter)

    def to_dict(self) -> dict[str, object]:
        return {
            "sphereId": self.sphere_id,
            "ownerConeSurfaceId": self.owner_cone_surface_id,
            "itemId": self.item_id,
            "proxy": self.proxy.to_dict(),
            "planePosition": self.plane_position.value,
            "planeRayParameter": self.plane_ray_parameter,
        }


@dataclass(frozen=True, slots=True)
class DandelinPlaneLayer:
    """Merged fill contours for one certified cutting-plane depth role."""

    role: PlaneDepthRole
    item_id: str
    contours: tuple[tuple[tuple[float, float], ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, PlaneDepthRole):
            raise TypeError("plane role must be a PlaneDepthRole")
        item_id = _identity(self.item_id, "plane item_id")
        contours = tuple(
            _path2(path, "plane contour", minimum=3) for path in self.contours
        )
        if not contours:
            raise DandelinSurfaceCompositingError(
                "a plane layer requires at least one positive-area contour"
            )
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "contours", contours)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "itemId": self.item_id,
            "contours": [
                [list(point) for point in contour] for contour in self.contours
            ],
        }


@dataclass(frozen=True, slots=True)
class DandelinPlaneOutlineLayer:
    """Depth-classified finite outline segments for one cutting-plane role."""

    role: PlaneDepthRole
    item_id: str
    paths: tuple[tuple[tuple[float, float], ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, PlaneDepthRole):
            raise TypeError("plane outline role must be a PlaneDepthRole")
        item_id = _identity(self.item_id, "plane outline item_id")
        paths = tuple(
            _path2(path, "plane outline path", minimum=2) for path in self.paths
        )
        if not paths:
            raise DandelinSurfaceCompositingError(
                "a plane outline layer requires at least one finite path"
            )
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "paths", paths)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "itemId": self.item_id,
            "paths": [[list(point) for point in path] for path in self.paths],
        }


@dataclass(frozen=True, slots=True)
class DandelinContactSheetSpan:
    """One exact parameter cell on a cone/sphere equal-depth contact circle."""

    interval: ParameterInterval
    sheet: DandelinContactSheet

    def __post_init__(self) -> None:
        if not isinstance(self.interval, ParameterInterval):
            raise TypeError("contact interval must be a ParameterInterval")
        if not isinstance(self.sheet, DandelinContactSheet):
            raise TypeError("contact sheet must be a DandelinContactSheet")

    def to_dict(self) -> dict[str, object]:
        return {
            "interval": [self.interval.start, self.interval.end],
            "sheet": self.sheet.value,
        }


@dataclass(frozen=True, slots=True)
class DandelinEqualDepthContact:
    """Certified sheet ownership along one cone/sphere tangent circle."""

    contact_curve_id: str
    sphere_id: str
    cone_surface_id: str
    domain: ParameterInterval
    transition_parameters: tuple[float, ...]
    spans: tuple[DandelinContactSheetSpan, ...]
    feature_stroke_owns_equal_depth: bool = True

    def __post_init__(self) -> None:
        curve_id = _identity(self.contact_curve_id, "contact_curve_id")
        sphere_id = _identity(self.sphere_id, "contact sphere_id")
        cone_id = _identity(self.cone_surface_id, "contact cone_surface_id")
        if not isinstance(self.domain, ParameterInterval):
            raise TypeError("contact domain must be a ParameterInterval")
        transitions = tuple(float(item) for item in self.transition_parameters)
        if (
            any(not isfinite(item) for item in transitions)
            or transitions != tuple(sorted(set(transitions)))
            or any(
                item < self.domain.start or item >= self.domain.end
                for item in transitions
            )
        ):
            raise DandelinSurfaceCompositingError(
                "contact transitions must be unique sorted domain parameters"
            )
        spans = tuple(self.spans)
        if not spans or not all(
            isinstance(item, DandelinContactSheetSpan) for item in spans
        ):
            raise DandelinSurfaceCompositingError(
                "equal-depth contact requires certified sheet spans"
            )
        try:
            assert_exact_partition(
                self.domain,
                (item.interval for item in spans),
            )
        except ValueError as exc:
            raise DandelinSurfaceCompositingError(
                "contact sheet spans must exactly cover the tangent circle"
            ) from exc
        if self.feature_stroke_owns_equal_depth is not True:
            raise DandelinSurfaceCompositingError(
                "the semantic contact stroke must own equal-depth pixels"
            )
        object.__setattr__(self, "contact_curve_id", curve_id)
        object.__setattr__(self, "sphere_id", sphere_id)
        object.__setattr__(self, "cone_surface_id", cone_id)
        object.__setattr__(self, "transition_parameters", transitions)
        object.__setattr__(self, "spans", spans)

    def to_dict(self) -> dict[str, object]:
        return {
            "contactCurveId": self.contact_curve_id,
            "sphereId": self.sphere_id,
            "coneSurfaceId": self.cone_surface_id,
            "domain": [self.domain.start, self.domain.end],
            "transitionParameters": list(self.transition_parameters),
            "spans": [item.to_dict() for item in self.spans],
            "featureStrokeOwnsEqualDepth": self.feature_stroke_owns_equal_depth,
        }


@dataclass(frozen=True, slots=True)
class DandelinSpherePairEvidence:
    """Separation or external-tangency evidence for the optional sphere pair."""

    first_sphere_id: str
    second_sphere_id: str
    relation: str
    surface_gap: float
    farther_sphere_id: str
    nearer_sphere_id: str
    tangent_point: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        first = _identity(self.first_sphere_id, "first_sphere_id")
        second = _identity(self.second_sphere_id, "second_sphere_id")
        farther = _identity(self.farther_sphere_id, "farther_sphere_id")
        nearer = _identity(self.nearer_sphere_id, "nearer_sphere_id")
        if first >= second:
            raise DandelinSurfaceCompositingError(
                "sphere-pair identities must use canonical order"
            )
        if {farther, nearer} != {first, second}:
            raise DandelinSurfaceCompositingError(
                "sphere-pair far/near identities must cover the pair"
            )
        if self.relation not in {"strictly_separated", "external_tangent"}:
            raise DandelinSurfaceCompositingError(
                "unsupported Dandelin sphere-pair relation"
            )
        gap = float(self.surface_gap)
        if not isfinite(gap):
            raise DandelinSurfaceCompositingError(
                "sphere-pair surface gap must be finite"
            )
        point = self.tangent_point
        if self.relation == "external_tangent":
            array = np.asarray(point, dtype=float)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise DandelinSurfaceCompositingError(
                    "external sphere tangency requires one finite point"
                )
            point = tuple(float(value) for value in array)
        elif point is not None:
            raise DandelinSurfaceCompositingError(
                "strictly separated spheres cannot carry a tangent point"
            )
        object.__setattr__(self, "first_sphere_id", first)
        object.__setattr__(self, "second_sphere_id", second)
        object.__setattr__(self, "farther_sphere_id", farther)
        object.__setattr__(self, "nearer_sphere_id", nearer)
        object.__setattr__(self, "surface_gap", gap)
        object.__setattr__(self, "tangent_point", point)

    def to_dict(self) -> dict[str, object]:
        return {
            "firstSphereId": self.first_sphere_id,
            "secondSphereId": self.second_sphere_id,
            "relation": self.relation,
            "surfaceGap": self.surface_gap,
            "fartherSphereId": self.farther_sphere_id,
            "nearerSphereId": self.nearer_sphere_id,
            "tangentPoint": (
                None if self.tangent_point is None else list(self.tangent_point)
            ),
        }


SectionFrame = (
    QuadricSectionCompositingFrame | CompositeQuadricSectionCompositingFrame
)


@dataclass(frozen=True, slots=True)
class DandelinSurfaceLayerFrame:
    """Renderer-neutral, far-to-near teaching-transparent surface trace."""

    construction_id: str
    construction: DandelinConstruction3D
    projection_matrix: tuple[tuple[float, float, float], ...]
    view_direction: tuple[float, float, float]
    patch: PlaneDisplayPatchSpec
    section_frame: SectionFrame
    cone_layers: tuple[DandelinConeLayer, ...]
    sphere_layers: tuple[DandelinSphereLayer, ...]
    plane_layers: tuple[DandelinPlaneLayer, ...]
    plane_outline_layers: tuple[DandelinPlaneOutlineLayer, ...]
    equal_depth_contacts: tuple[DandelinEqualDepthContact, ...]
    sphere_pair_evidence: tuple[DandelinSpherePairEvidence, ...]
    order_relations: tuple[QuadricPaintRelation, ...]
    draw_order: tuple[str, ...]
    max_screen_error: float
    max_segments: int
    surface_layering_authoritative: bool = True
    physical_surface_visibility_authoritative: bool = False
    schema: str = DANDELIN_SURFACE_LAYER_FRAME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DANDELIN_SURFACE_LAYER_FRAME_SCHEMA:
            raise DandelinSurfaceCompositingError(
                "invalid Dandelin surface-layer schema"
            )
        construction_id = _identity(self.construction_id, "construction_id")
        if not isinstance(self.construction, DandelinConstruction3D):
            raise TypeError("construction must be a DandelinConstruction3D")
        if self.construction.construction_id != construction_id:
            raise DandelinSurfaceCompositingError(
                "surface-layer construction identity disagrees with its evidence"
            )
        view = ParallelView(self.projection_matrix, self.view_direction)
        error = _positive(self.max_screen_error, "max_screen_error")
        if (
            isinstance(self.max_segments, bool)
            or not isinstance(self.max_segments, int)
            or self.max_segments < 8
        ):
            raise DandelinSurfaceCompositingError(
                "max_segments must be an integer of at least eight"
            )
        segment_limit = self.max_segments
        if not isinstance(self.patch, PlaneDisplayPatchSpec):
            raise TypeError("patch must be a PlaneDisplayPatchSpec")
        if not isinstance(
            self.section_frame,
            (QuadricSectionCompositingFrame, CompositeQuadricSectionCompositingFrame),
        ):
            raise TypeError("section_frame must be a certified section compositor frame")
        if self.section_frame.patch != self.patch:
            raise DandelinSurfaceCompositingError(
                "surface-layer patch disagrees with its section frame"
            )
        if (
            self.section_frame.plane != self.construction.plane
            or self.patch.plane_id != self.construction.plane.plane_id
        ):
            raise DandelinSurfaceCompositingError(
                "surface-layer section evidence belongs to a different construction plane"
            )
        if self.section_frame.projection_kind is not PlanePatchProjectionKind.AREA:
            raise DandelinSurfaceCompositingError(
                "surface-layer frame requires an AREA cutting-plane projection"
            )
        expected_section_frame, _ = _section_frame(
            self.construction,
            view,
            self.patch,
            max_screen_error=error,
            max_segments=segment_limit,
        )
        if self.section_frame != expected_section_frame:
            raise DandelinSurfaceCompositingError(
                "section compositor evidence disagrees with construction-derived geometry"
            )
        section_projection = (
            self.section_frame.base_frame.visibility.projection_matrix
            if isinstance(self.section_frame, QuadricSectionCompositingFrame)
            else self.section_frame.child_frames[
                0
            ].base_frame.visibility.projection_matrix
        )
        if section_projection != view.projection_matrix:
            raise DandelinSurfaceCompositingError(
                "surface-layer view disagrees with section compositor evidence"
            )
        cones = tuple(self.cone_layers)
        spheres = tuple(self.sphere_layers)
        planes = tuple(self.plane_layers)
        outlines = tuple(self.plane_outline_layers)
        contacts = tuple(self.equal_depth_contacts)
        pairs = tuple(self.sphere_pair_evidence)
        typed = (
            (cones, DandelinConeLayer, "cone_layers"),
            (spheres, DandelinSphereLayer, "sphere_layers"),
            (planes, DandelinPlaneLayer, "plane_layers"),
            (outlines, DandelinPlaneOutlineLayer, "plane_outline_layers"),
            (contacts, DandelinEqualDepthContact, "equal_depth_contacts"),
            (pairs, DandelinSpherePairEvidence, "sphere_pair_evidence"),
        )
        for values, expected, label in typed:
            if not all(isinstance(item, expected) for item in values):
                raise TypeError(f"{label} contains an invalid value")
        if tuple(item.surface_id for item in cones) != tuple(
            sorted({item.surface_id for item in cones})
        ):
            raise DandelinSurfaceCompositingError(
                "cone layers must have unique canonical surface identities"
            )
        components = tuple(
            sorted(
                self.construction.cone.render_components,
                key=lambda item: item.surface_id,
            )
        )
        if tuple(item.surface_id for item in components) != tuple(
            item.surface_id for item in cones
        ):
            raise DandelinSurfaceCompositingError(
                "cone layers do not cover the certified construction components"
            )
        expected_projection_layers = tuple(
            build_cone_projection_layers(
                component,
                view,
                max_chord_error=error,
                max_segments=segment_limit,
            )
            for component in components
        )
        if tuple(item.projection_layers for item in cones) != expected_projection_layers:
            raise DandelinSurfaceCompositingError(
                "cone layers disagree with construction-derived projection evidence"
            )
        if tuple(item.sphere_id for item in spheres) != tuple(
            sorted({item.sphere_id for item in spheres})
        ):
            raise DandelinSurfaceCompositingError(
                "sphere layers must have unique canonical identities"
            )
        cone_ids = {item.surface_id for item in cones}
        if any(item.owner_cone_surface_id not in cone_ids for item in spheres):
            raise DandelinSurfaceCompositingError(
                "a sphere layer references an unknown cone component"
            )
        sheet_map, _fill_ids, _outline_ids = _sheet_item_maps(self.section_frame)
        if set(sheet_map) != cone_ids:
            raise DandelinSurfaceCompositingError(
                "cone layers do not cover the section compositor surfaces"
            )
        child_frames = (
            (self.section_frame,)
            if isinstance(self.section_frame, QuadricSectionCompositingFrame)
            else self.section_frame.child_frames
        )
        child_by_id = {item.surface_id: item for item in child_frames}
        for item in cones:
            if (item.back_item_id, item.front_item_id) != sheet_map[item.surface_id]:
                raise DandelinSurfaceCompositingError(
                    "cone painter identities disagree with section evidence"
                )
            if (
                item.projection_layers.proxy
                != child_by_id[item.surface_id].surface_proxy
            ):
                raise DandelinSurfaceCompositingError(
                    "cone projection layers disagree with section evidence"
                )
        expected_planes, expected_outlines = _plane_layers(
            self.section_frame,
            view.projection_matrix,
        )
        if planes != expected_planes or outlines != expected_outlines:
            raise DandelinSurfaceCompositingError(
                "plane render layers disagree with certified section geometry"
            )
        contact_keys = tuple(item.contact_curve_id for item in contacts)
        if contact_keys != tuple(sorted(set(contact_keys))):
            raise DandelinSurfaceCompositingError(
                "equal-depth contacts must use unique canonical curve identities"
            )
        sphere_ids = {item.sphere_id for item in spheres}
        if len(contacts) != len(spheres) or {
            item.sphere_id for item in contacts
        } != sphere_ids or any(
            item.cone_surface_id
            != next(
                sphere.owner_cone_surface_id
                for sphere in spheres
                if sphere.sphere_id == item.sphere_id
            )
            for item in contacts
        ):
            raise DandelinSurfaceCompositingError(
                "equal-depth contacts do not cover the certified sphere layers"
            )
        if any(
            item.item_id != f"surface:{item.sphere_id}:teaching-fill"
            for item in spheres
        ):
            raise DandelinSurfaceCompositingError(
                "sphere painter identities are not canonical"
            )
        expected_pair_count = 1 if len(spheres) == 2 else 0
        if len(pairs) != expected_pair_count or any(
            {item.first_sphere_id, item.second_sphere_id} != sphere_ids
            for item in pairs
        ):
            raise DandelinSurfaceCompositingError(
                "sphere-pair evidence does not cover the certified sphere set"
            )
        expected_spheres, expected_contacts = _sphere_layers(
            self.construction,
            view,
            max_screen_error=error,
            max_segments=segment_limit,
        )
        if spheres != expected_spheres or contacts != expected_contacts:
            raise DandelinSurfaceCompositingError(
                "sphere layers or equal-depth seams disagree with construction evidence"
            )
        expected_pairs = _sphere_pair_evidence(
            self.construction,
            expected_spheres,
        )
        if pairs != expected_pairs:
            raise DandelinSurfaceCompositingError(
                "sphere-pair evidence disagrees with construction geometry"
            )
        item_ids = (
            *(value for item in cones for value in (item.back_item_id, item.front_item_id)),
            *(item.item_id for item in spheres),
            *(item.item_id for item in planes),
            *(item.item_id for item in outlines),
        )
        if len(item_ids) != len(set(item_ids)):
            raise DandelinSurfaceCompositingError(
                "surface painter identities must be globally unique"
            )
        relations = tuple(self.order_relations)
        if not all(isinstance(item, QuadricPaintRelation) for item in relations):
            raise TypeError("order_relations must contain QuadricPaintRelation")
        relation_keys = tuple(
            (item.far_item_id, item.near_item_id, item.reason)
            for item in relations
        )
        if relation_keys != tuple(sorted(relation_keys)):
            raise DandelinSurfaceCompositingError(
                "surface painter relations must use canonical order"
            )
        unknown = sorted(
            {
                value
                for item in relations
                for value in (item.far_item_id, item.near_item_id)
                if value not in set(item_ids)
            }
        )
        if unknown:
            raise DandelinSurfaceCompositingError(
                "surface painter graph references unknown items: "
                + ", ".join(unknown)
            )
        expected_relations, expected_painter_order = _painter_graph(
            cones,
            spheres,
            planes,
            outlines,
            pairs,
        )
        if relations != expected_relations or tuple(self.draw_order) != expected_painter_order:
            raise DandelinSurfaceCompositingError(
                "surface painter graph disagrees with construction-derived layers"
            )
        try:
            expected_order = stable_topological_sort(
                item_ids,
                (
                    PainterConstraint(item.far_item_id, item.near_item_id)
                    for item in relations
                ),
                key=lambda item_id: item_id,
            )
        except CompositorCycleError as exc:
            raise DandelinSurfaceCompositingError(
                "Dandelin surface painter graph contains a cycle: "
                + ", ".join(sorted(str(item) for item in exc.unresolved))
            ) from exc
        if tuple(self.draw_order) != expected_order:
            raise DandelinSurfaceCompositingError(
                "draw_order is not the canonical certified surface order"
            )
        if error != self.section_frame.max_screen_error:
            raise DandelinSurfaceCompositingError(
                "surface-layer error budget disagrees with section evidence"
            )
        if self.surface_layering_authoritative is not True:
            raise DandelinSurfaceCompositingError(
                "teaching-transparent frames must certify surface layering"
            )
        if self.physical_surface_visibility_authoritative is not False:
            raise DandelinSurfaceCompositingError(
                "teaching-transparent frames cannot claim physical visibility"
            )
        object.__setattr__(self, "construction_id", construction_id)
        object.__setattr__(self, "projection_matrix", view.projection_matrix)
        object.__setattr__(self, "view_direction", view.view_direction)
        object.__setattr__(self, "cone_layers", cones)
        object.__setattr__(self, "sphere_layers", spheres)
        object.__setattr__(self, "plane_layers", planes)
        object.__setattr__(self, "plane_outline_layers", outlines)
        object.__setattr__(self, "equal_depth_contacts", contacts)
        object.__setattr__(self, "sphere_pair_evidence", pairs)
        object.__setattr__(self, "order_relations", relations)
        object.__setattr__(self, "draw_order", expected_order)
        object.__setattr__(self, "max_screen_error", error)
        object.__setattr__(self, "max_segments", segment_limit)

    @property
    def paint_item_ids(self) -> tuple[str, ...]:
        return self.draw_order

    def __deepcopy__(self, memo: dict[int, object]) -> "DandelinSurfaceLayerFrame":
        """Share this immutable proof frame when a Manim tree is copied."""

        memo[id(self)] = self
        return self

    @property
    def plane_fragment_count(self) -> int:
        return len(self.section_frame.plane_fragments)

    def to_dict(self) -> dict[str, object]:
        section_frame_json = json.dumps(
            self.section_frame.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "schema": self.schema,
            "constructionId": self.construction_id,
            "constructionSha256": sha256(
                self.construction.canonical_json().encode("utf-8")
            ).hexdigest(),
            "projectionMatrix": [list(row) for row in self.projection_matrix],
            "viewDirection": list(self.view_direction),
            "patch": {
                "patchId": self.patch.patch_id,
                "planeId": self.patch.plane_id,
                "halfWidth": self.patch.half_width,
                "halfHeight": self.patch.half_height,
                "centerCoordinates": list(self.patch.center_coordinates),
            },
            "sectionFrameSchema": self.section_frame.schema,
            "sectionFrameSha256": sha256(
                section_frame_json.encode("utf-8")
            ).hexdigest(),
            "projectionKind": self.section_frame.projection_kind.value,
            "planeFragmentCount": self.plane_fragment_count,
            "coneLayers": [item.to_dict() for item in self.cone_layers],
            "sphereLayers": [item.to_dict() for item in self.sphere_layers],
            "planeLayers": [item.to_dict() for item in self.plane_layers],
            "planeOutlineLayers": [
                item.to_dict() for item in self.plane_outline_layers
            ],
            "equalDepthContacts": [
                item.to_dict() for item in self.equal_depth_contacts
            ],
            "spherePairEvidence": [
                item.to_dict() for item in self.sphere_pair_evidence
            ],
            "orderRelations": [item.to_dict() for item in self.order_relations],
            "drawOrder": list(self.draw_order),
            "maxScreenError": self.max_screen_error,
            "maxSegments": self.max_segments,
            "surfaceLayeringAuthoritative": self.surface_layering_authoritative,
            "physicalSurfaceVisibilityAuthoritative": (
                self.physical_surface_visibility_authoritative
            ),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _sheet_item_maps(
    section_frame: SectionFrame,
) -> tuple[
    dict[str, tuple[str, str]],
    dict[PlaneDepthRole, str],
    dict[PlaneDepthRole, str],
]:
    if isinstance(section_frame, QuadricSectionCompositingFrame):
        items = section_frame.paint_items
        return (
            {section_frame.surface_id: (items.surface_back, items.surface_front)},
            {
                PlaneDepthRole.BEHIND_SURFACE: items.plane_behind,
                PlaneDepthRole.OUTSIDE_PROJECTION: items.plane_outside,
                PlaneDepthRole.BETWEEN_SURFACE_SHEETS: items.plane_between,
                PlaneDepthRole.IN_FRONT_OF_SURFACE: items.plane_front,
            },
            items.outline_by_role,
        )
    items = section_frame.paint_items
    return (
        {
            item.child_surface_id: (item.surface_back, item.surface_front)
            for item in items.surface_sheets
        },
        items.fill_by_role,
        items.outline_by_role,
    )


def _section_frame(
    construction: DandelinConstruction3D,
    view: ParallelView,
    patch: PlaneDisplayPatchSpec,
    *,
    max_screen_error: float,
    max_segments: int,
) -> tuple[SectionFrame, tuple[ConeProjectionLayers, ...]]:
    local_frames: list[QuadricSectionCompositingFrame] = []
    cone_layers: list[ConeProjectionLayers] = []
    for component in construction.cone.render_components:
        layers = build_cone_projection_layers(
            component,
            view,
            max_chord_error=max_screen_error,
            max_segments=max_segments,
        )
        base = compute_global_quadric_frame(
            (),
            (component,),
            view,
            context=construction.certification_context,
            paint_policy=QuadricPaintPolicy.PHYSICAL,
            max_chord_error=max_screen_error,
            max_segments=max_segments,
        ).frame
        if base.surface_items[0].proxy != layers.proxy:
            raise DandelinSurfaceCompositingError(
                "cone sheet and section compositor projection evidence drifted"
            )
        local_frames.append(
            compute_quadric_section_compositing(
                base,
                component,
                construction.plane,
                patch,
                view,
                context=construction.certification_context,
                max_screen_error=max_screen_error,
            )
        )
        cone_layers.append(layers)
    if len(local_frames) == 1:
        result: SectionFrame = local_frames[0]
    elif construction.cone.model is ConeModel.OPEN_DOUBLE and len(local_frames) == 2:
        result = compute_composite_quadric_section_compositing(
            construction.cone,
            f"{construction.construction_id}:surface-layer-section",
            local_frames,
            (),
        )
    else:
        raise DandelinSurfaceCompositingError(
            "Dandelin surface compositing requires one nappe or canonical open-double siblings"
        )
    if result.projection_kind is not PlanePatchProjectionKind.AREA:
        raise DandelinSurfaceCompositingError(
            "teaching-transparent Dandelin mode requires an AREA cutting-plane projection"
        )
    return result, tuple(cone_layers)


def _plane_layers(
    frame: SectionFrame,
    projection_matrix: Sequence[Sequence[float]],
) -> tuple[
    tuple[DandelinPlaneLayer, ...],
    tuple[DandelinPlaneOutlineLayer, ...],
]:
    _sheet_map, fill_ids, outline_ids = _sheet_item_maps(frame)
    contours = merge_quadric_plane_fragment_contours(
        frame.plane,
        frame.patch,
        projection_matrix,
        frame.plane_fragments,
    )
    fills = tuple(
        DandelinPlaneLayer(role, fill_ids[role], contours[role])
        for role in PlaneDepthRole
        if contours[role]
    )
    outline_by_role: dict[PlaneDepthRole, list[tuple[tuple[float, float], ...]]] = {
        role: [] for role in PlaneDepthRole
    }
    for fragment in frame.plane_outline_fragments:
        if not isinstance(fragment, QuadricPlaneOutlineFragment):
            raise DandelinSurfaceCompositingError(
                "section frame contains an invalid plane-outline fragment"
            )
        outline_by_role[fragment.role].append(
            (fragment.screen_start, fragment.screen_end)
        )
    outlines = tuple(
        DandelinPlaneOutlineLayer(
            role,
            outline_ids[role],
            tuple(outline_by_role[role]),
        )
        for role in PlaneDepthRole
        if outline_by_role[role]
    )
    return fills, outlines


def _harmonic_contact(
    construction: DandelinConstruction3D,
    sphere_id: str,
    cone_surface_id: str,
    view: ParallelView,
) -> DandelinEqualDepthContact:
    matches = tuple(
        item for item in construction.spheres if item.sphere_id == sphere_id
    )
    if len(matches) != 1:
        raise DandelinSurfaceCompositingError(
            f"no unique Dandelin sphere record for {sphere_id!r}"
        )
    record = matches[0]
    curve = record.cone_contact_circle.lower_to_analytic_curve()
    center_delta = (
        np.asarray(curve.center, dtype=float)
        - np.asarray(record.sphere.center, dtype=float)
    ) / record.sphere.radius
    first = np.asarray(curve.first_axis, dtype=float) / record.sphere.radius
    second = np.asarray(curve.second_axis, dtype=float) / record.sphere.radius
    direction = np.asarray(view.view_direction, dtype=float)
    coefficients = (
        float(np.dot(first, direction)),
        float(np.dot(second, direction)),
        float(np.dot(center_delta, direction)),
    )
    cosine, sine_coefficient, constant = coefficients
    amplitude = float(np.hypot(cosine, sine_coefficient))
    scale = max(1.0, abs(cosine), abs(sine_coefficient), abs(constant))
    tolerance = 4096.0 * np.finfo(float).eps * scale
    if amplitude <= tolerance:
        if abs(constant) <= tolerance:
            raise DandelinSurfaceCompositingError(
                f"contact circle {curve.curve_id!r} is persistently edge-on"
            )
        roots: tuple[float, ...] = ()
    else:
        ratio = -constant / amplitude
        if ratio < -1.0 - tolerance or ratio > 1.0 + tolerance:
            roots = ()
        else:
            ratio = min(1.0, max(-1.0, ratio))
            phase = atan2(sine_coefficient, cosine)
            offset = acos(ratio)
            values: list[float] = []
            for base in (phase - offset, phase + offset):
                first_index = floor((curve.domain.start - base) / tau) - 1
                last_index = ceil((curve.domain.end - base) / tau) + 1
                for index in range(first_index, last_index + 1):
                    value = base + index * tau
                    if (
                        value < curve.domain.start - tolerance
                        or value > curve.domain.end + tolerance
                    ):
                        continue
                    value = min(curve.domain.end, max(curve.domain.start, value))
                    if value == curve.domain.end and curve.closed:
                        value = curve.domain.start
                    values.append(float(value))
            normalized: list[float] = []
            for value in sorted(values):
                if not normalized or value - normalized[-1] > 1.0e-12:
                    normalized.append(value)
            roots = tuple(normalized)
    interior = tuple(
        value
        for value in roots
        if curve.domain.start < value < curve.domain.end
    )
    cells = partition_parameter_domain(
        curve.domain,
        interior,
        tolerance=max(tolerance, np.finfo(float).eps * 64.0),
    )
    spans: list[DandelinContactSheetSpan] = []
    for cell in cells:
        value = (
            cosine * np.cos(cell.midpoint)
            + sine_coefficient * np.sin(cell.midpoint)
            + constant
        )
        if abs(value) <= tolerance:
            raise DandelinSurfaceCompositingError(
                f"contact circle {curve.curve_id!r} sheet side is unresolved"
            )
        spans.append(
            DandelinContactSheetSpan(
                cell,
                (
                    DandelinContactSheet.FRONT
                    if value > 0.0
                    else DandelinContactSheet.BACK
                ),
            )
        )
    for root in roots:
        residual = (
            cosine * np.cos(root)
            + sine_coefficient * np.sin(root)
            + constant
        )
        if abs(residual) > tolerance * 4.0:
            raise DandelinSurfaceCompositingError(
                f"contact circle {curve.curve_id!r} transition residual is too large"
            )
    return DandelinEqualDepthContact(
        curve.curve_id,
        sphere_id,
        cone_surface_id,
        curve.domain,
        roots,
        tuple(spans),
    )


def _sphere_layers(
    construction: DandelinConstruction3D,
    view: ParallelView,
    *,
    max_screen_error: float,
    max_segments: int,
) -> tuple[
    tuple[DandelinSphereLayer, ...],
    tuple[DandelinEqualDepthContact, ...],
]:
    contacts = certify_dandelin_tangent_contacts(construction)
    owner_by_sphere = {item.sphere_id: item for item in contacts}
    direction = np.asarray(view.view_direction, dtype=float)
    normal = np.asarray(construction.plane.normal, dtype=float)
    denominator = float(np.dot(normal, direction))
    denominator_tolerance = 4096.0 * np.finfo(float).eps
    if abs(denominator) <= denominator_tolerance:
        raise DandelinSurfaceCompositingError(
            "cutting plane is edge-on to the teaching-transparent view"
        )
    boundary_epsilon = construction.certification_context.epsilon(
        GeometryQuantity.BOUNDARY
    )
    depth_epsilon = construction.certification_context.epsilon(
        GeometryQuantity.DEPTH
    )
    components = {
        item.surface_id: item for item in construction.cone.render_components
    }
    layers: list[DandelinSphereLayer] = []
    seams: list[DandelinEqualDepthContact] = []
    for record in construction.spheres:
        contact = owner_by_sphere.get(record.sphere_id)
        if contact is None or contact.cone_surface_id not in components:
            raise DandelinSurfaceCompositingError(
                f"sphere {record.sphere_id!r} has no certified cone component"
            )
        component = components[contact.cone_surface_id]
        if (
            record.axial_extent[0]
            < component.axial_range[0] - boundary_epsilon
            or record.axial_extent[1]
            > component.axial_range[1] + boundary_epsilon
        ):
            raise DandelinSurfaceCompositingError(
                f"sphere {record.sphere_id!r} exceeds its finite cone component"
            )
        signed_distance = construction.plane.signed_distance(record.sphere.center)
        tangency_error = abs(abs(signed_distance) - record.sphere.radius)
        if tangency_error > max(boundary_epsilon * 8.0, 1.0e-10 * record.sphere.radius):
            raise DandelinSurfaceCompositingError(
                f"sphere {record.sphere_id!r} is not certified tangent to the cutting plane"
            )
        parameter = -signed_distance / denominator
        if abs(parameter) <= depth_epsilon:
            raise DandelinSurfaceCompositingError(
                f"sphere {record.sphere_id!r} has unresolved plane depth"
            )
        layers.append(
            DandelinSphereLayer(
                record.sphere_id,
                contact.cone_surface_id,
                f"surface:{record.sphere_id}:teaching-fill",
                build_opaque_projection_proxy(
                    record.sphere,
                    view,
                    max_chord_error=max_screen_error,
                    max_segments=max_segments,
                ),
                (
                    DandelinPlanePosition.IN_FRONT_OF_SPHERE
                    if parameter > 0.0
                    else DandelinPlanePosition.BEHIND_SPHERE
                ),
                parameter,
            )
        )
        seams.append(
            _harmonic_contact(
                construction,
                record.sphere_id,
                contact.cone_surface_id,
                view,
            )
        )
    return (
        tuple(sorted(layers, key=lambda item: item.sphere_id)),
        tuple(sorted(seams, key=lambda item: item.contact_curve_id)),
    )


def _sphere_pair_evidence(
    construction: DandelinConstruction3D,
    layers: Sequence[DandelinSphereLayer],
) -> tuple[DandelinSpherePairEvidence, ...]:
    if len(layers) < 2:
        return ()
    if len(layers) != 2 or len(construction.spheres) != 2:
        raise DandelinSurfaceCompositingError(
            "the first Dandelin surface compositor supports at most two spheres"
        )
    records = {item.sphere_id: item for item in construction.spheres}
    first_layer, second_layer = tuple(sorted(layers, key=lambda item: item.sphere_id))
    first = records[first_layer.sphere_id]
    second = records[second_layer.sphere_id]
    first_center = np.asarray(first.sphere.center, dtype=float)
    second_center = np.asarray(second.sphere.center, dtype=float)
    displacement = second_center - first_center
    distance = float(np.linalg.norm(displacement))
    if not isfinite(distance) or distance <= 0.0:
        raise DandelinSurfaceCompositingError(
            "Dandelin sphere centers must be distinct"
        )
    gap = distance - first.sphere.radius - second.sphere.radius
    tolerance = max(
        construction.certification_context.epsilon(GeometryQuantity.BOUNDARY) * 8.0,
        1.0e-10 * max(distance, first.sphere.radius, second.sphere.radius),
    )
    if gap < -tolerance:
        raise DandelinSurfaceCompositingError(
            "Dandelin spheres overlap and have no certified painter relation"
        )
    if first_layer.plane_position is second_layer.plane_position:
        raise DandelinSurfaceCompositingError(
            "two Dandelin spheres on the same cutting-plane side need an additional depth certificate"
        )
    farther = (
        first_layer
        if first_layer.plane_position is DandelinPlanePosition.IN_FRONT_OF_SPHERE
        else second_layer
    )
    nearer = second_layer if farther is first_layer else first_layer
    if abs(gap) <= tolerance:
        tangent = first_center + (
            first.sphere.radius / distance
        ) * displacement
        if abs(construction.plane.signed_distance(tangent)) > tolerance:
            raise DandelinSurfaceCompositingError(
                "externally tangent Dandelin spheres do not meet on the cutting plane"
            )
        for record in (first, second):
            focus = np.asarray(record.focus.world_point, dtype=float)
            if float(np.linalg.norm(tangent - focus)) > tolerance:
                raise DandelinSurfaceCompositingError(
                    "sphere tangent point disagrees with the certified common focus"
                )
        relation = "external_tangent"
        tangent_point: tuple[float, float, float] | None = tuple(
            float(value) for value in tangent
        )
    else:
        relation = "strictly_separated"
        tangent_point = None
    return (
        DandelinSpherePairEvidence(
            first_layer.sphere_id,
            second_layer.sphere_id,
            relation,
            gap,
            farther.sphere_id,
            nearer.sphere_id,
            tangent_point,
        ),
    )


def _dedupe_relations(
    values: Sequence[QuadricPaintRelation],
) -> tuple[QuadricPaintRelation, ...]:
    reasons: dict[tuple[str, str], set[str]] = {}
    for item in values:
        key = (item.far_item_id, item.near_item_id)
        reverse = (key[1], key[0])
        if reverse in reasons:
            raise DandelinSurfaceCompositingError(
                "Dandelin surface painter graph contains contradictory relations"
            )
        reasons.setdefault(key, set()).add(item.reason)
    return tuple(
        QuadricPaintRelation(far, near, "+".join(sorted(labels)))
        for (far, near), labels in sorted(reasons.items())
    )


def _painter_graph(
    cones: Sequence[DandelinConeLayer],
    spheres: Sequence[DandelinSphereLayer],
    planes: Sequence[DandelinPlaneLayer],
    outlines: Sequence[DandelinPlaneOutlineLayer],
    pairs: Sequence[DandelinSpherePairEvidence],
) -> tuple[tuple[QuadricPaintRelation, ...], tuple[str, ...]]:
    plane_nodes: dict[PlaneDepthRole, tuple[str, ...]] = {
        role: tuple(
            (
                *(item.item_id for item in planes if item.role is role),
                *(item.item_id for item in outlines if item.role is role),
            )
        )
        for role in PlaneDepthRole
    }
    backs = tuple(item.back_item_id for item in cones)
    fronts = tuple(item.front_item_id for item in cones)
    groups = (
        plane_nodes[PlaneDepthRole.BEHIND_SURFACE],
        backs,
        plane_nodes[PlaneDepthRole.OUTSIDE_PROJECTION],
        plane_nodes[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
        fronts,
        plane_nodes[PlaneDepthRole.IN_FRONT_OF_SURFACE],
    )
    relations: list[QuadricPaintRelation] = []
    previous: str | None = None
    for group in groups:
        for item_id in group:
            if previous is not None:
                relations.append(
                    QuadricPaintRelation(
                        previous,
                        item_id,
                        "dandelin_section_depth_chain",
                    )
                )
            previous = item_id
    cone_by_id = {item.surface_id: item for item in cones}
    behind_nodes = (
        *plane_nodes[PlaneDepthRole.BEHIND_SURFACE],
        *plane_nodes[PlaneDepthRole.OUTSIDE_PROJECTION],
        *plane_nodes[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
    )
    front_nodes = (
        *plane_nodes[PlaneDepthRole.OUTSIDE_PROJECTION],
        *plane_nodes[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
        *plane_nodes[PlaneDepthRole.IN_FRONT_OF_SURFACE],
    )
    for sphere in spheres:
        owner = cone_by_id[sphere.owner_cone_surface_id]
        relations.extend(
            (
                QuadricPaintRelation(
                    owner.back_item_id,
                    sphere.item_id,
                    "sphere_inside_cone_after_back_sheet",
                ),
                QuadricPaintRelation(
                    sphere.item_id,
                    owner.front_item_id,
                    "sphere_inside_cone_before_front_sheet",
                ),
            )
        )
        if sphere.plane_position is DandelinPlanePosition.IN_FRONT_OF_SPHERE:
            relations.extend(
                QuadricPaintRelation(
                    sphere.item_id,
                    item_id,
                    "cutting_plane_in_front_of_tangent_sphere",
                )
                for item_id in front_nodes
            )
            relations.extend(
                QuadricPaintRelation(
                    item_id,
                    sphere.item_id,
                    "plane_behind_region_precedes_far_side_sphere",
                )
                for item_id in plane_nodes[PlaneDepthRole.BEHIND_SURFACE]
            )
        else:
            relations.extend(
                QuadricPaintRelation(
                    item_id,
                    sphere.item_id,
                    "cutting_plane_behind_tangent_sphere",
                )
                for item_id in behind_nodes
            )
            relations.extend(
                QuadricPaintRelation(
                    sphere.item_id,
                    item_id,
                    "near_side_sphere_precedes_plane_front_region",
                )
                for item_id in plane_nodes[PlaneDepthRole.IN_FRONT_OF_SURFACE]
            )
    sphere_item_by_id = {item.sphere_id: item.item_id for item in spheres}
    relations.extend(
        QuadricPaintRelation(
            sphere_item_by_id[item.farther_sphere_id],
            sphere_item_by_id[item.nearer_sphere_id],
            "dandelin_sphere_pair_depth",
        )
        for item in pairs
    )
    normalized = _dedupe_relations(relations)
    item_ids = (
        *(value for item in cones for value in (item.back_item_id, item.front_item_id)),
        *(item.item_id for item in spheres),
        *(item.item_id for item in planes),
        *(item.item_id for item in outlines),
    )
    try:
        order = stable_topological_sort(
            item_ids,
            (
                PainterConstraint(item.far_item_id, item.near_item_id)
                for item in normalized
            ),
            key=lambda item_id: item_id,
        )
    except CompositorCycleError as exc:
        raise DandelinSurfaceCompositingError(
            "Dandelin surface painter graph contains a cycle: "
            + ", ".join(sorted(str(item) for item in exc.unresolved))
        ) from exc
    return normalized, order


def compute_dandelin_surface_layer_frame(
    construction: DandelinConstruction3D,
    view: ParallelView,
    patch: PlaneDisplayPatchSpec,
    *,
    max_screen_error: float = _DEFAULT_MAX_SCREEN_ERROR,
    max_segments: int = _DEFAULT_MAX_SEGMENTS,
) -> DandelinSurfaceLayerFrame:
    """Build a fail-closed teaching-transparent surface painter frame.

    ``patch`` is the finite cutting-plane fill to partition.  Feature lines may
    use a larger display patch; they do not alter this certified fill geometry.
    """

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    if not isinstance(patch, PlaneDisplayPatchSpec):
        raise TypeError("patch must be a PlaneDisplayPatchSpec")
    if patch.plane_id != construction.plane.plane_id:
        raise DandelinSurfaceCompositingError(
            "surface-layer patch belongs to a different cutting plane"
        )
    error = _positive(max_screen_error, "max_screen_error")
    if (
        isinstance(max_segments, bool)
        or not isinstance(max_segments, int)
        or max_segments < 8
    ):
        raise DandelinSurfaceCompositingError(
            "max_segments must be an integer of at least eight"
        )
    try:
        section_frame, projection_layers = _section_frame(
            construction,
            view,
            patch,
            max_screen_error=error,
            max_segments=max_segments,
        )
        sheet_map, _fill_ids, _outline_ids = _sheet_item_maps(section_frame)
        cones = tuple(
            DandelinConeLayer(
                item.surface_id,
                sheet_map[item.surface_id][0],
                sheet_map[item.surface_id][1],
                item,
            )
            for item in sorted(
                projection_layers,
                key=lambda value: value.surface_id,
            )
        )
        planes, outlines = _plane_layers(
            section_frame,
            view.projection_matrix,
        )
        spheres, seams = _sphere_layers(
            construction,
            view,
            max_screen_error=error,
            max_segments=max_segments,
        )
        pairs = _sphere_pair_evidence(construction, spheres)
        relations, draw_order = _painter_graph(
            cones,
            spheres,
            planes,
            outlines,
            pairs,
        )
    except DandelinSurfaceCompositingError:
        raise
    except (
        ProjectionProxyError,
        GlobalQuadricOcclusionError,
        QuadricSectionCompositingError,
        CompositeQuadricSectionCompositingError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ) as exc:
        raise DandelinSurfaceCompositingError(
            f"Dandelin surface layering cannot be certified: {exc}"
        ) from exc
    return DandelinSurfaceLayerFrame(
        construction_id=construction.construction_id,
        construction=construction,
        projection_matrix=view.projection_matrix,
        view_direction=view.view_direction,
        patch=patch,
        section_frame=section_frame,
        cone_layers=cones,
        sphere_layers=spheres,
        plane_layers=planes,
        plane_outline_layers=outlines,
        equal_depth_contacts=seams,
        sphere_pair_evidence=pairs,
        order_relations=relations,
        draw_order=draw_order,
        max_screen_error=error,
        max_segments=max_segments,
    )


def canonical_dandelin_surface_layer_json(
    frame: DandelinSurfaceLayerFrame,
) -> str:
    if not isinstance(frame, DandelinSurfaceLayerFrame):
        raise TypeError("frame must be a DandelinSurfaceLayerFrame")
    return frame.canonical_json()


__all__ = [
    "DANDELIN_SURFACE_LAYER_FRAME_SCHEMA",
    "DandelinConeLayer",
    "DandelinContactSheet",
    "DandelinContactSheetSpan",
    "DandelinEqualDepthContact",
    "DandelinPlaneLayer",
    "DandelinPlaneOutlineLayer",
    "DandelinPlanePosition",
    "DandelinSphereLayer",
    "DandelinSpherePairEvidence",
    "DandelinSurfaceCompositingError",
    "DandelinSurfaceLayerFrame",
    "canonical_dandelin_surface_layer_json",
    "compute_dandelin_surface_layer_frame",
]
