"""Unified renderer-neutral contracts for semantic boundary strokes.

The existing quadric stack intentionally keeps exact curve visibility, finite
section geometry, and Manim allocation separate.  This module is the sidecar
which turns all semantic stroke sources into one fragment-level painter graph
without changing any existing v1 frame or importing Manim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from typing import Mapping, Sequence

from ..compositor import (
    CompositorCycleError,
    PainterConstraint,
    stable_topological_sort,
)
from ..geometry import GeometryContext, ResolvedGeometryContext
from ..parallel_solver import ParallelView
from ..topology import ParameterInterval
from ..visibility import VisibilityKind
from .compositing import (
    QuadricCompositingError,
    QuadricPaintPolicy,
    QuadricPaintRelation,
    _depth_aware_farther_items,
    _paint_predecessors,
)
from .contract import ConeSpec, CylinderSpec, SphereSpec
from .critical import AnalyticCurve3D
from .curve_intersections import ProjectedCurveCrossing
from .visibility import CurveVisibilityRecord, compute_curve_visibility


QUADRIC_BOUNDARY_COMPOSITING_SCHEMA = "manim-quadric-boundary-compositing/v1"
QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec
ContextInput = GeometryContext | ResolvedGeometryContext | None


class QuadricBoundaryCompositingError(ValueError):
    """A deterministic semantic-boundary painter frame cannot be formed."""


class BoundarySourceKind(str, Enum):
    ANALYTIC_CURVE = "analytic_curve"
    PLANE_PATCH_EDGE = "plane_patch_edge"
    SURFACE_CAP_RIM = "surface_cap_rim"
    SURFACE_GENERATOR = "surface_generator"
    SURFACE_SILHOUETTE = "surface_silhouette"
    POLYHEDRON_EDGE = "polyhedron_edge"
    FEATURE_LINE = "feature_line"


class BoundarySemanticKind(str, Enum):
    FREE_CURVE = "free_curve"
    DISPLAY_FRAME = "display_frame"
    SURFACE_BOUNDARY = "surface_boundary"
    TEACHING_FEATURE = "teaching_feature"
    TRUE_SILHOUETTE = "true_silhouette"


class BoundaryOcclusionScope(str, Enum):
    ALL_SURFACES = "all_surfaces"
    OWNER_AND_EXTERNAL = "owner_and_external"
    EXTERNAL_ONLY = "external_only"
    NONE = "none"


class BoundaryRenderIntent(str, Enum):
    SOLID = "solid"
    DASHED = "dashed"
    OMIT = "omit"


_DEPTH_ROLES = {
    "outside_projection",
    "behind_surface",
    "between_surface_sheets",
    "in_front_of_surface",
}


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuadricBoundaryCompositingError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class QuadricBoundarySource:
    """One cross-frame semantic stroke source with exact world geometry."""

    source_id: str
    curve: AnalyticCurve3D
    source_kind: BoundarySourceKind
    semantic_kind: BoundarySemanticKind
    occlusion_scope: BoundaryOcclusionScope
    owner_id: str
    owner_surface_id: str | None = None
    style_id: str | None = None
    stable_sort_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        source_id = _identity(self.source_id, "boundary source_id")
        if self.curve.curve_id != source_id:
            raise QuadricBoundaryCompositingError(
                "boundary curve_id must equal its stable source_id"
            )
        if not isinstance(self.source_kind, BoundarySourceKind):
            raise TypeError("source_kind must be a BoundarySourceKind")
        if not isinstance(self.semantic_kind, BoundarySemanticKind):
            raise TypeError("semantic_kind must be a BoundarySemanticKind")
        if not isinstance(self.occlusion_scope, BoundaryOcclusionScope):
            raise TypeError("occlusion_scope must be a BoundaryOcclusionScope")
        owner = _identity(self.owner_id, "boundary owner_id")
        owner_surface = (
            None
            if self.owner_surface_id is None
            else _identity(self.owner_surface_id, "boundary owner_surface_id")
        )
        if self.occlusion_scope in {
            BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
            BoundaryOcclusionScope.EXTERNAL_ONLY,
        } and owner_surface is None:
            raise QuadricBoundaryCompositingError(
                "owner-aware boundary scopes require owner_surface_id"
            )
        style_id = (
            None
            if self.style_id is None
            else _identity(self.style_id, "style_id")
        )
        sort_key = tuple(
            _identity(item, "stable_sort_key item")
            for item in self.stable_sort_key
        )
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "owner_surface_id", owner_surface)
        object.__setattr__(self, "style_id", style_id)
        object.__setattr__(self, "stable_sort_key", sort_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceId": self.source_id,
            "sourceKind": self.source_kind.value,
            "semanticKind": self.semantic_kind.value,
            "occlusionScope": self.occlusion_scope.value,
            "ownerId": self.owner_id,
            "ownerSurfaceId": self.owner_surface_id,
            "styleId": self.style_id,
            "stableSortKey": list(self.stable_sort_key),
            "curve": self.curve.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class QuadricBoundaryVisibilitySpan:
    """One exact source interval plus optional plane-depth ownership."""

    interval: ParameterInterval
    kind: VisibilityKind
    occluder_surface_ids: tuple[str, ...] = ()
    depth_role: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.interval, ParameterInterval)
            or self.interval.length <= 0.0
        ):
            raise QuadricBoundaryCompositingError(
                "boundary visibility interval must have positive length"
            )
        if not isinstance(self.kind, VisibilityKind):
            raise TypeError("kind must be a VisibilityKind")
        occluders = tuple(
            _identity(item, "occluder surface identity")
            for item in self.occluder_surface_ids
        )
        if occluders != tuple(sorted(set(occluders))):
            raise QuadricBoundaryCompositingError(
                "boundary occluder identities must be unique and sorted"
            )
        if self.kind is VisibilityKind.VISIBLE and occluders:
            raise QuadricBoundaryCompositingError(
                "visible boundary spans cannot name occluders"
            )
        if self.kind is VisibilityKind.HIDDEN and not occluders:
            raise QuadricBoundaryCompositingError(
                "hidden boundary spans must name an occluder"
            )
        role = self.depth_role
        if role is not None and role not in _DEPTH_ROLES:
            raise QuadricBoundaryCompositingError(
                f"unsupported boundary depth role {role!r}"
            )
        object.__setattr__(self, "occluder_surface_ids", occluders)

    def to_dict(self) -> dict[str, object]:
        return {
            "interval": [self.interval.start, self.interval.end],
            "kind": self.kind.value,
            "occluderSurfaceIds": list(self.occluder_surface_ids),
            "depthRole": self.depth_role,
        }


@dataclass(frozen=True, slots=True)
class QuadricBoundaryPaintFragment:
    """One fragment-level painter item with stable semantic lineage."""

    item_id: str
    source_id: str
    interval: ParameterInterval
    visibility_kind: VisibilityKind
    occluder_surface_ids: tuple[str, ...]
    render_intent: BoundaryRenderIntent
    painted: bool
    semantic_kind: BoundarySemanticKind
    depth_role: str | None
    plane_relation: str | None
    plane_depth_roles: tuple[str, ...]
    style_id: str | None
    stable_sort_key: tuple[str, ...]

    def __post_init__(self) -> None:
        item_id = _identity(self.item_id, "boundary fragment item_id")
        source_id = _identity(self.source_id, "boundary fragment source_id")
        if (
            not isinstance(self.interval, ParameterInterval)
            or self.interval.length <= 0.0
        ):
            raise QuadricBoundaryCompositingError(
                "boundary fragment interval must have positive length"
            )
        if not isinstance(self.visibility_kind, VisibilityKind):
            raise TypeError("visibility_kind must be a VisibilityKind")
        if not isinstance(self.render_intent, BoundaryRenderIntent):
            raise TypeError("render_intent must be a BoundaryRenderIntent")
        if not isinstance(self.painted, bool):
            raise TypeError("painted must be a bool")
        if not isinstance(self.semantic_kind, BoundarySemanticKind):
            raise TypeError("semantic_kind must be a BoundarySemanticKind")
        if self.painted != (self.render_intent is not BoundaryRenderIntent.OMIT):
            raise QuadricBoundaryCompositingError(
                "painted flag must agree with boundary render_intent"
            )
        if (
            self.visibility_kind is VisibilityKind.VISIBLE
            and self.render_intent is not BoundaryRenderIntent.SOLID
        ):
            raise QuadricBoundaryCompositingError(
                "visible boundary fragments must render solid"
            )
        if (
            self.visibility_kind is VisibilityKind.HIDDEN
            and self.render_intent is BoundaryRenderIntent.SOLID
        ):
            raise QuadricBoundaryCompositingError(
                "hidden boundary fragments cannot render solid"
            )
        roles = tuple(str(item) for item in self.plane_depth_roles)
        if roles != tuple(sorted(set(roles))) or any(
            role not in _DEPTH_ROLES for role in roles
        ):
            raise QuadricBoundaryCompositingError(
                "fragment plane-depth roles must be unique, sorted, and valid"
            )
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "plane_depth_roles", roles)

    def to_dict(self) -> dict[str, object]:
        return {
            "itemId": self.item_id,
            "sourceId": self.source_id,
            "interval": [self.interval.start, self.interval.end],
            "visibilityKind": self.visibility_kind.value,
            "occluderSurfaceIds": list(self.occluder_surface_ids),
            "renderIntent": self.render_intent.value,
            "painted": self.painted,
            "semanticKind": self.semantic_kind.value,
            "depthRole": self.depth_role,
            "planeRelation": self.plane_relation,
            "planeDepthRoles": list(self.plane_depth_roles),
            "styleId": self.style_id,
            "stableSortKey": list(self.stable_sort_key),
        }


@dataclass(frozen=True, slots=True)
class BoundarySectionAnchors:
    plane_behind: str
    outline_behind: str
    surface_back: str
    plane_outside: str
    outline_outside: str
    plane_between: str
    outline_between: str
    surface_front: str
    plane_front: str
    outline_front: str

    def __post_init__(self) -> None:
        values = tuple(
            _identity(getattr(self, name), name)
            for name in self.__dataclass_fields__
        )
        if len(set(values)) != len(values):
            raise QuadricBoundaryCompositingError(
                "section boundary anchors must have unique identities"
            )


@dataclass(frozen=True, slots=True)
class QuadricBoundaryCompositingFrame:
    sources: tuple[QuadricBoundarySource, ...]
    fragments: tuple[QuadricBoundaryPaintFragment, ...]
    parent_item_ids: tuple[str, ...]
    order_relations: tuple[QuadricPaintRelation, ...]
    draw_order: tuple[str, ...]
    crossings: tuple[ProjectedCurveCrossing, ...] = ()
    schema: str = QUADRIC_BOUNDARY_COMPOSITING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUADRIC_BOUNDARY_COMPOSITING_SCHEMA:
            raise QuadricBoundaryCompositingError(
                "invalid quadric boundary compositing schema"
            )
        source_ids = tuple(item.source_id for item in self.sources)
        if (
            source_ids != tuple(sorted(source_ids))
            or len(set(source_ids)) != len(source_ids)
        ):
            raise QuadricBoundaryCompositingError(
                "boundary sources must have unique sorted identities"
            )
        fragment_ids = tuple(item.item_id for item in self.fragments)
        if (
            fragment_ids != tuple(sorted(fragment_ids))
            or len(set(fragment_ids)) != len(fragment_ids)
        ):
            raise QuadricBoundaryCompositingError(
                "boundary fragments must have unique sorted identities"
            )
        if len(set(self.parent_item_ids)) != len(self.parent_item_ids):
            raise QuadricBoundaryCompositingError(
                "parent painter identities must be unique"
            )
        active = {
            *self.parent_item_ids,
            *(item.item_id for item in self.fragments if item.painted),
        }
        if len(self.draw_order) != len(active) or set(self.draw_order) != active:
            raise QuadricBoundaryCompositingError(
                "boundary draw_order must cover every active item exactly once"
            )
        relation_keys = tuple(
            (item.far_item_id, item.near_item_id, item.reason)
            for item in self.order_relations
        )
        if relation_keys != tuple(sorted(relation_keys)):
            raise QuadricBoundaryCompositingError(
                "boundary painter relations must be sorted"
            )
        try:
            expected = stable_topological_sort(
                sorted(active),
                (
                    PainterConstraint(item.far_item_id, item.near_item_id)
                    for item in self.order_relations
                ),
                key=lambda item_id: item_id,
            )
        except CompositorCycleError as exc:
            raise QuadricBoundaryCompositingError(
                "boundary painter graph contains a cycle: "
                + ", ".join(sorted(str(item) for item in exc.unresolved))
            ) from exc
        if self.draw_order != expected:
            raise QuadricBoundaryCompositingError(
                "boundary draw_order is not canonical"
            )

    @property
    def painted_fragments(self) -> tuple[QuadricBoundaryPaintFragment, ...]:
        return tuple(item for item in self.fragments if item.painted)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sources": [item.to_dict() for item in self.sources],
            "fragments": [item.to_dict() for item in self.fragments],
            "parentItemIds": list(self.parent_item_ids),
            "orderRelations": [item.to_dict() for item in self.order_relations],
            "drawOrder": list(self.draw_order),
            "crossings": [item.to_dict() for item in self.crossings],
        }


def _selected_surfaces(
    source: QuadricBoundarySource,
    surfaces: tuple[QuadricSurfaceSpec, ...],
) -> tuple[QuadricSurfaceSpec, ...]:
    if source.occlusion_scope is BoundaryOcclusionScope.NONE:
        return ()
    if source.occlusion_scope in {
        BoundaryOcclusionScope.ALL_SURFACES,
        BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
    }:
        return surfaces
    return tuple(
        surface
        for surface in surfaces
        if surface.surface_id != source.owner_surface_id
    )


def compute_boundary_visibility(
    sources: Sequence[QuadricBoundarySource],
    surfaces: Sequence[QuadricSurfaceSpec],
    view: ParallelView,
    *,
    context: ContextInput = None,
) -> dict[str, tuple[QuadricBoundaryVisibilitySpan, ...]]:
    """Run the exact analytic visibility kernel with source-specific scope."""

    source_items = tuple(sorted(sources, key=lambda item: item.source_id))
    surface_items = tuple(sorted(surfaces, key=lambda item: item.surface_id))
    result: dict[str, tuple[QuadricBoundaryVisibilitySpan, ...]] = {}
    for source in source_items:
        record: CurveVisibilityRecord = compute_curve_visibility(
            source.curve,
            _selected_surfaces(source, surface_items),
            view,
            context=context,
        )
        result[source.source_id] = tuple(
            QuadricBoundaryVisibilitySpan(
                span.interval,
                span.kind,
                tuple(span.occluders),
            )
            for span in record.spans
        )
    return result


def _dedupe_relations(
    relations: Sequence[QuadricPaintRelation],
) -> tuple[QuadricPaintRelation, ...]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for relation in relations:
        pair = (relation.far_item_id, relation.near_item_id)
        reverse = (pair[1], pair[0])
        if reverse in grouped:
            raise QuadricBoundaryCompositingError(
                "boundary painter relations contain contradictory evidence: "
                f"{pair[0]!r}, {pair[1]!r}"
            )
        grouped.setdefault(pair, set()).add(relation.reason)
    return tuple(
        QuadricPaintRelation(far, near, "+".join(sorted(reasons)))
        for (far, near), reasons in sorted(grouped.items())
    )


def _render_intent(
    policy: QuadricPaintPolicy,
    kind: VisibilityKind,
) -> BoundaryRenderIntent:
    if kind is VisibilityKind.VISIBLE:
        return BoundaryRenderIntent.SOLID
    if policy is QuadricPaintPolicy.PHYSICAL:
        return BoundaryRenderIntent.OMIT
    return BoundaryRenderIntent.DASHED


def _crossing_parameters(
    crossings: Sequence[ProjectedCurveCrossing],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for crossing in crossings:
        result.setdefault(crossing.first_curve_id, []).append(crossing.first_parameter)
        result.setdefault(crossing.second_curve_id, []).append(
            crossing.second_parameter
        )
    return result


def _split_interval(
    interval: ParameterInterval,
    values: Sequence[float],
    tolerance: float,
) -> tuple[ParameterInterval, ...]:
    points = [interval.start]
    for value in sorted(float(item) for item in values):
        if interval.start + tolerance < value < interval.end - tolerance:
            if value - points[-1] > tolerance:
                points.append(value)
    points.append(interval.end)
    return tuple(
        ParameterInterval(start, end)
        for start, end in zip(points, points[1:])
        if end - start > tolerance
    )


def _fragments_at_parameter(
    fragments: Sequence[QuadricBoundaryPaintFragment],
    source_id: str,
    parameter: float,
    tolerance: float,
) -> tuple[QuadricBoundaryPaintFragment, ...]:
    return tuple(
        item
        for item in fragments
        if item.painted
        and item.source_id == source_id
        and item.interval.contains(parameter, tolerance=tolerance)
    )


def _add_bracket(
    relations: list[QuadricPaintRelation],
    fragment_id: str,
    far_anchor: str,
    near_anchor: str,
    reason: str,
) -> None:
    relations.append(QuadricPaintRelation(far_anchor, fragment_id, reason + ":after"))
    relations.append(QuadricPaintRelation(fragment_id, near_anchor, reason + ":before"))


def _plane_item_for_role(
    role: str | None,
    anchors: BoundarySectionAnchors,
) -> str | None:
    return {
        "behind_surface": anchors.plane_behind,
        "outside_projection": anchors.plane_outside,
        "between_surface_sheets": anchors.plane_between,
        "in_front_of_surface": anchors.plane_front,
    }.get(role)


def _add_section_plane_relation(
    relations: list[QuadricPaintRelation],
    fragment: QuadricBoundaryPaintFragment,
    anchors: BoundarySectionAnchors,
) -> None:
    relation = fragment.plane_relation
    if relation in {None, "outside_patch", "coincident"}:
        return
    item_ids = tuple(
        item_id
        for role in fragment.plane_depth_roles
        if (item_id := _plane_item_for_role(role, anchors)) is not None
    )
    if not item_ids:
        raise QuadricBoundaryCompositingError(
            "boundary span inside the patch has no adjacent plane painter item"
        )
    if relation == "boundary_behind_plane":
        relations.extend(
            QuadricPaintRelation(
                fragment.item_id,
                item_id,
                "boundary_behind_section_plane",
            )
            for item_id in item_ids
        )
    elif relation == "boundary_in_front_of_plane":
        relations.extend(
            QuadricPaintRelation(
                item_id,
                fragment.item_id,
                "boundary_in_front_of_section_plane",
            )
            for item_id in item_ids
        )
    else:
        raise QuadricBoundaryCompositingError(
            f"unsupported boundary/plane relation {relation!r}"
        )


def compute_quadric_boundary_compositing(
    sources: Sequence[QuadricBoundarySource],
    spans_by_source: Mapping[str, Sequence[QuadricBoundaryVisibilitySpan]],
    *,
    paint_policy: QuadricPaintPolicy | str,
    parent_item_ids: Sequence[str],
    parent_relations: Sequence[QuadricPaintRelation],
    surface_item_by_id: Mapping[str, str],
    crossings: Sequence[ProjectedCurveCrossing] = (),
    section_anchors: BoundarySectionAnchors | None = None,
    section_spans_by_source: Mapping[str, Sequence[object]] | None = None,
    parameter_tolerance: float = 1.0e-12,
) -> QuadricBoundaryCompositingFrame:
    """Build one fragment-level painter graph for all semantic boundaries."""

    try:
        policy = QuadricPaintPolicy(paint_policy)
    except (TypeError, ValueError) as exc:
        raise QuadricBoundaryCompositingError("invalid boundary paint policy") from exc
    tolerance = float(parameter_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise QuadricBoundaryCompositingError(
            "parameter_tolerance must be finite and non-negative"
        )
    source_items = tuple(sorted(sources, key=lambda item: item.source_id))
    source_ids = tuple(item.source_id for item in source_items)
    if len(set(source_ids)) != len(source_ids):
        raise QuadricBoundaryCompositingError(
            "boundary source identities must be unique"
        )
    if set(spans_by_source) != set(source_ids):
        raise QuadricBoundaryCompositingError(
            "spans_by_source must cover every boundary source exactly"
        )
    crossings_tuple = tuple(sorted(crossings, key=lambda item: item.crossing_id))
    split_values = _crossing_parameters(crossings_tuple)
    section_spans = (
        {} if section_spans_by_source is None else dict(section_spans_by_source)
    )
    unknown_section_sources = sorted(set(section_spans) - set(source_ids))
    if unknown_section_sources:
        raise QuadricBoundaryCompositingError(
            "section spans reference unknown boundary sources: "
            + ", ".join(unknown_section_sources)
        )
    for source_id, spans in section_spans.items():
        split_values.setdefault(source_id, []).extend(
            value
            for span in spans
            for value in (span.interval.start, span.interval.end)
        )
    fragments: list[QuadricBoundaryPaintFragment] = []
    for source in source_items:
        spans = tuple(spans_by_source[source.source_id])
        for span_index, span in enumerate(spans):
            if not isinstance(span, QuadricBoundaryVisibilitySpan):
                raise TypeError(
                    "spans_by_source must contain QuadricBoundaryVisibilitySpan"
                )
            pieces = _split_interval(
                span.interval,
                split_values.get(source.source_id, ()),
                tolerance,
            )
            for piece_index, interval in enumerate(pieces):
                placement = None
                placement_matches = [
                    item
                    for item in section_spans.get(source.source_id, ())
                    if item.interval.contains(
                        interval.midpoint, tolerance=tolerance
                    )
                ]
                if len(placement_matches) > 1:
                    raise QuadricBoundaryCompositingError(
                        f"boundary {source.source_id!r} has overlapping section spans"
                    )
                if placement_matches:
                    placement = placement_matches[0]
                intent = _render_intent(policy, span.kind)
                item_id = (
                    f"boundary:{source.source_id}:span:{span_index:04d}:"
                    f"piece:{piece_index:04d}:{span.kind.value}"
                )
                fragments.append(
                    QuadricBoundaryPaintFragment(
                        item_id=item_id,
                        source_id=source.source_id,
                        interval=interval,
                        visibility_kind=span.kind,
                        occluder_surface_ids=span.occluder_surface_ids,
                        render_intent=intent,
                        painted=intent is not BoundaryRenderIntent.OMIT,
                        semantic_kind=source.semantic_kind,
                        # ``depth_role`` is the source-owned role used by
                        # plane-patch edges.  Surface boundaries keep all
                        # adjacent plane regions in ``plane_depth_roles``.
                        depth_role=span.depth_role,
                        plane_relation=(
                            None if placement is None else placement.relation.value
                        ),
                        plane_depth_roles=(
                            () if placement is None else placement.plane_depth_roles
                        ),
                        style_id=source.style_id,
                        stable_sort_key=(
                            *source.stable_sort_key,
                            f"{span_index:04d}",
                            f"{piece_index:04d}",
                        ),
                    )
                )
    fragments.sort(key=lambda item: item.item_id)

    parent_ids = tuple(parent_item_ids)
    parent_set = set(parent_ids)
    normalized_surface_items = {
        _identity(surface_id, "surface_item_by_id key"): _identity(
            item_id, "surface_item_by_id value"
        )
        for surface_id, item_id in surface_item_by_id.items()
    }
    if len(normalized_surface_items) != len(surface_item_by_id):
        raise QuadricBoundaryCompositingError(
            "surface_item_by_id keys must be unique after normalization"
        )
    surface_item_values = tuple(normalized_surface_items.values())
    if len(set(surface_item_values)) != len(surface_item_values):
        raise QuadricBoundaryCompositingError(
            "surface_item_by_id values must be unique"
        )
    unknown_surface_items = sorted(set(surface_item_values) - parent_set)
    if unknown_surface_items:
        raise QuadricBoundaryCompositingError(
            "surface_item_by_id references non-parent items: "
            + ", ".join(unknown_surface_items)
        )
    surface_item_by_id = normalized_surface_items
    relaxed_parent_pairs: set[tuple[str, str]] = set()
    if section_anchors is not None:
        # The outside patch region is screen-disjoint from both projection
        # sheets and the inside depth roles.  The legacy compositor gives the
        # ten anchors a convenient total chain, but retaining those two
        # non-overlap edges in a fragment-level graph creates false cycles
        # when an actual silhouette or cap rim orders itself against the
        # outside fill.  Keep the outside fill/outline pair ordered locally;
        # direct boundary evidence supplies every relation that can paint the
        # same pixels.
        relaxed_parent_pairs = {
            (section_anchors.surface_back, section_anchors.plane_outside),
            (section_anchors.outline_outside, section_anchors.plane_between),
        }
    relations = [
        item
        for item in parent_relations
        if item.far_item_id in parent_set
        and item.near_item_id in parent_set
        and (item.far_item_id, item.near_item_id) not in relaxed_parent_pairs
    ]
    surface_predecessors: dict[str, frozenset[str]] = {}
    if section_anchors is None:
        surface_item_ids = tuple(sorted(surface_item_by_id.values()))
        surface_item_set = set(surface_item_ids)
        try:
            surface_predecessors = _paint_predecessors(
                surface_item_ids,
                tuple(
                    item
                    for item in relations
                    if item.far_item_id in surface_item_set
                    and item.near_item_id in surface_item_set
                ),
            )
        except QuadricCompositingError as exc:
            raise QuadricBoundaryCompositingError(
                f"invalid parent surface ordering: {exc}"
            ) from exc

    for fragment in fragments:
        if not fragment.painted:
            continue
        source = next(
            item
            for item in source_items
            if item.source_id == fragment.source_id
        )
        is_plane_edge = source.source_kind is BoundarySourceKind.PLANE_PATCH_EDGE
        if is_plane_edge and fragment.visibility_kind is VisibilityKind.VISIBLE:
            if section_anchors is None:
                raise QuadricBoundaryCompositingError(
                    "plane-edge fragments require section anchors"
                )
            role = fragment.depth_role
            if role == "outside_projection":
                _add_bracket(
                    relations,
                    fragment.item_id,
                    section_anchors.plane_outside,
                    section_anchors.outline_outside,
                    "plane_outline_outside",
                )
            elif role == "in_front_of_surface":
                _add_bracket(
                    relations,
                    fragment.item_id,
                    section_anchors.plane_front,
                    section_anchors.outline_front,
                    "plane_outline_front",
                )
            else:
                raise QuadricBoundaryCompositingError(
                    "visible plane outline fragment has a hidden depth role"
                )
            continue

        if fragment.visibility_kind is VisibilityKind.VISIBLE:
            if section_anchors is not None:
                if source.owner_surface_id is not None:
                    relations.append(
                        QuadricPaintRelation(
                            section_anchors.surface_front,
                            fragment.item_id,
                            "visible_owner_surface_boundary",
                        )
                    )
                elif (
                    fragment.plane_relation == "boundary_behind_plane"
                    and "in_front_of_surface" in fragment.plane_depth_roles
                ):
                    # The boundary is visible because it lies in front of the
                    # finite surface, but the current section plane is nearer.
                    # Give the fragment both sides of that certified bracket:
                    # surface_front -> boundary -> plane_front.  Without the
                    # lower edge, deterministic tie-breaking can place the
                    # boundary behind the opaque front sheet.
                    relations.append(
                        QuadricPaintRelation(
                            section_anchors.surface_front,
                            fragment.item_id,
                            "visible_boundary_in_front_of_surface",
                        )
                    )
                elif fragment.plane_relation != "boundary_behind_plane":
                    # A transition bank can carry a visible section curve
                    # whose geometry lies behind the currently displayed
                    # plane.  Its certified plane relation must then place it
                    # before the front fill and outline; forcing the generic
                    # visible overlay here would create the reverse path.
                    relations.append(
                        QuadricPaintRelation(
                            section_anchors.outline_front,
                            fragment.item_id,
                            "visible_boundary_overlay",
                        )
                    )
            else:
                relations.extend(
                    QuadricPaintRelation(
                        item_id,
                        fragment.item_id,
                        "visible_boundary_overlay",
                    )
                    for item_id in sorted(surface_item_by_id.values())
                )
            if section_anchors is not None:
                _add_section_plane_relation(
                    relations, fragment, section_anchors
                )
            continue

        if policy is QuadricPaintPolicy.DIAGRAMMATIC:
            if section_anchors is not None:
                relations.append(
                    QuadricPaintRelation(
                        section_anchors.outline_front,
                        fragment.item_id,
                        "diagrammatic_hidden_boundary_overlay",
                    )
                )
            else:
                relations.extend(
                    QuadricPaintRelation(
                        item_id,
                        fragment.item_id,
                        "diagrammatic_hidden_boundary_overlay",
                    )
                    for item_id in sorted(surface_item_by_id.values())
                )
            continue

        if policy is not QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC:
            raise QuadricBoundaryCompositingError(
                "painted hidden fragment requires a diagrammatic policy"
            )

        if is_plane_edge:
            if section_anchors is None:
                raise QuadricBoundaryCompositingError(
                    "plane-edge fragments require section anchors"
                )
            if fragment.depth_role == "behind_surface":
                _add_bracket(
                    relations,
                    fragment.item_id,
                    section_anchors.plane_behind,
                    section_anchors.outline_behind,
                    "plane_outline_behind",
                )
            elif fragment.depth_role == "between_surface_sheets":
                _add_bracket(
                    relations,
                    fragment.item_id,
                    section_anchors.plane_between,
                    section_anchors.outline_between,
                    "plane_outline_between",
                )
            else:
                raise QuadricBoundaryCompositingError(
                    "hidden plane outline fragment has a visible depth role"
                )
        elif section_anchors is not None:
            if source.owner_surface_id is not None:
                _add_bracket(
                    relations,
                    fragment.item_id,
                    section_anchors.surface_back,
                    section_anchors.surface_front,
                    "depth_aware_hidden_owner_boundary",
                )
            else:
                far_anchor = (
                    section_anchors.surface_back
                    if fragment.plane_relation == "boundary_behind_plane"
                    else section_anchors.outline_between
                )
                _add_bracket(
                    relations,
                    fragment.item_id,
                    far_anchor,
                    section_anchors.surface_front,
                    "depth_aware_hidden_boundary",
                )
            _add_section_plane_relation(
                relations, fragment, section_anchors
            )
        else:
            occluder_items: set[str] = set()
            for surface_id in fragment.occluder_surface_ids:
                occluder_item = surface_item_by_id.get(surface_id)
                if occluder_item is None:
                    raise QuadricBoundaryCompositingError(
                        f"unknown boundary occluder surface {surface_id!r}"
                    )
                occluder_items.add(occluder_item)
            try:
                farther_surface_items = _depth_aware_farther_items(
                    occluder_items, surface_predecessors
                )
            except QuadricCompositingError as exc:
                raise QuadricBoundaryCompositingError(
                    f"cannot bracket hidden boundary {fragment.item_id!r}: {exc}"
                ) from exc
            relations.extend(
                QuadricPaintRelation(
                    item_id,
                    fragment.item_id,
                    "depth_aware_hidden_boundary_after_farther_surface",
                )
                for item_id in sorted(farther_surface_items)
            )
            relations.extend(
                QuadricPaintRelation(
                    fragment.item_id,
                    item_id,
                    "depth_aware_hidden_boundary_occlusion",
                )
                for item_id in sorted(occluder_items)
            )

    source_map = {item.source_id: item for item in source_items}
    for crossing in crossings_tuple:
        if crossing.far_curve_id is None or crossing.near_curve_id is None:
            continue
        crossing_sources = (
            source_map[crossing.first_curve_id],
            source_map[crossing.second_curve_id],
        )
        del crossing_sources
        far = _fragments_at_parameter(
            fragments,
            crossing.far_curve_id,
            (
                crossing.first_parameter
                if crossing.far_curve_id == crossing.first_curve_id
                else crossing.second_parameter
            ),
            tolerance,
        )
        near = _fragments_at_parameter(
            fragments,
            crossing.near_curve_id,
            (
                crossing.first_parameter
                if crossing.near_curve_id == crossing.first_curve_id
                else crossing.second_parameter
            ),
            tolerance,
        )
        for farther in far:
            for nearer in near:
                relations.append(
                    QuadricPaintRelation(
                        farther.item_id,
                        nearer.item_id,
                        f"boundary_crossing:{crossing.crossing_id}",
                    )
                )

    normalized = _dedupe_relations(relations)
    active = tuple(
        sorted(
            (
                *parent_ids,
                *(item.item_id for item in fragments if item.painted),
            )
        )
    )
    try:
        draw_order = stable_topological_sort(
            active,
            (
                PainterConstraint(item.far_item_id, item.near_item_id)
                for item in normalized
            ),
            key=lambda item_id: item_id,
        )
    except CompositorCycleError as exc:
        raise QuadricBoundaryCompositingError(
            "boundary painter graph contains a cycle: "
            + ", ".join(sorted(str(item) for item in exc.unresolved))
        ) from exc
    return QuadricBoundaryCompositingFrame(
        sources=source_items,
        fragments=tuple(fragments),
        parent_item_ids=parent_ids,
        order_relations=normalized,
        draw_order=draw_order,
        crossings=crossings_tuple,
    )


def canonical_quadric_boundary_compositing_json(
    frame: QuadricBoundaryCompositingFrame,
) -> str:
    if not isinstance(frame, QuadricBoundaryCompositingFrame):
        raise TypeError("frame must be a QuadricBoundaryCompositingFrame")
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "BoundaryOcclusionScope",
    "BoundaryRenderIntent",
    "BoundarySectionAnchors",
    "BoundarySemanticKind",
    "BoundarySourceKind",
    "QUADRIC_BOUNDARY_COMPOSITING_SCHEMA",
    "QuadricBoundaryCompositingError",
    "QuadricBoundaryCompositingFrame",
    "QuadricBoundaryPaintFragment",
    "QuadricBoundarySource",
    "QuadricBoundaryVisibilitySpan",
    "canonical_quadric_boundary_compositing_json",
    "compute_boundary_visibility",
    "compute_quadric_boundary_compositing",
]
