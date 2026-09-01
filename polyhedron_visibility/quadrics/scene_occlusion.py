"""Dispatch and coordinate registered analytic quadric occlusion scenes.

The existing kernels intentionally have narrow contracts.  In particular,
the global surface sorter accepts only pairwise-disjoint finite convex
quadrics, while the section compositor accepts exactly one mother surface and
one plane.  This module keeps both contracts unchanged and chooses between
them explicitly:

* ordinary disjoint surfaces use the global compositor;
* one surface plus one section uses the existing section fast path verbatim;
* a registered mother surface with two tangent inner spheres uses the nested
  tangent parent adapter and then attaches the existing analytic boundary
  visibility/compositing pipeline.

Unregistered mixtures still fail closed.  The coordinator is renderer-neutral
and emits one final painter graph for every registered parent and boundary
fragment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
from math import isfinite
from typing import Sequence

from ..compositor import PainterConstraint, stable_topological_sort
from ..geometry import GeometryContext, GeometryQuantity, ResolvedGeometryContext
from ..parallel_solver import ParallelView
from .boundary_compositing import (
    BoundaryOcclusionScope,
    BoundarySectionAnchors,
    QuadricBoundaryCompositingFrame,
    QuadricBoundarySource,
    QuadricRankOneSectionSourceGroup,
    compute_boundary_visibility,
    compute_quadric_boundary_compositing,
    compute_quadric_boundary_crossings,
)
from .curves import EllipseArcCurve
from .boundary_section import (
    certify_rank_one_section_boundary_sources,
    compute_boundary_section_spans,
)
from .compositing import QuadricPaintPolicy, QuadricPaintRelation
from .contract import (
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from .global_occlusion import GlobalQuadricFrame, compute_global_quadric_frame
from .nested_tangent_compositing import (
    NestedTangentParentFrame,
    NestedTangentSphereSpec,
    _required_nested_parent_relations,
    compute_nested_tangent_parent_frame,
)
from .section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionKind,
    QuadricSectionCompositingFrame,
    compute_quadric_section_compositing,
)


SCENE_OCCLUSION_FRAME_SCHEMA = "manim-quadric-scene-occlusion-frame/v1"
QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec
ContextInput = GeometryContext | ResolvedGeometryContext | None


class SceneOcclusionError(ValueError):
    """A scene request cannot be dispatched or certified without guessing."""


class SceneOcclusionPath(str, Enum):
    """The explicit solver path selected for one registered scene."""

    GLOBAL_DISJOINT = "global_disjoint"
    SINGLE_SECTION = "single_section"
    NESTED_TANGENT_SECTION = "nested_tangent_section"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneOcclusionError(f"{label} must be a non-empty string")
    return value.strip()


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SceneOcclusionError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneOcclusionError(
            f"{label} must be finite and positive"
        ) from exc
    if not isfinite(result) or result <= 0.0:
        raise SceneOcclusionError(f"{label} must be finite and positive")
    return result


@dataclass(frozen=True, slots=True)
class SceneSectionSpec:
    """Register one finite plane patch against one mother surface."""

    mother_surface_id: str
    plane: SectionPlane
    patch: PlaneDisplayPatchSpec

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mother_surface_id",
            _identity(self.mother_surface_id, "mother_surface_id"),
        )
        if not isinstance(self.plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")
        if not isinstance(self.patch, PlaneDisplayPatchSpec):
            raise TypeError("patch must be a PlaneDisplayPatchSpec")
        if self.patch.plane_id != self.plane.plane_id:
            raise SceneOcclusionError(
                "section display patch does not belong to its plane"
            )


@dataclass(frozen=True, slots=True)
class SceneOcclusionRequest:
    """One immutable registered analytic scene request."""

    scene_id: str
    surfaces: tuple[QuadricSurfaceSpec, ...]
    view: ParallelView
    boundary_sources: tuple[QuadricBoundarySource, ...] = ()
    section: SceneSectionSpec | None = None
    tangent_spheres: tuple[NestedTangentSphereSpec, ...] = ()
    paint_policy: QuadricPaintPolicy = (
        QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC
    )
    context: ContextInput = None
    max_chord_error: float = 0.08
    max_surface_segments: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _identity(self.scene_id, "scene_id"))
        surfaces = tuple(self.surfaces)
        if not surfaces or not all(
            isinstance(item, (SphereSpec, CylinderSpec, ConeSpec))
            for item in surfaces
        ):
            raise SceneOcclusionError(
                "surfaces must contain at least one supported finite quadric"
            )
        surface_ids = tuple(item.surface_id for item in surfaces)
        if len(set(surface_ids)) != len(surface_ids):
            raise SceneOcclusionError("surface identities must be unique")
        sources = tuple(self.boundary_sources)
        if not all(isinstance(item, QuadricBoundarySource) for item in sources):
            raise TypeError("boundary_sources must contain QuadricBoundarySource")
        source_ids = tuple(item.source_id for item in sources)
        if len(set(source_ids)) != len(source_ids):
            raise SceneOcclusionError("boundary source identities must be unique")
        bindings = tuple(self.tangent_spheres)
        if not all(isinstance(item, NestedTangentSphereSpec) for item in bindings):
            raise TypeError(
                "tangent_spheres must contain NestedTangentSphereSpec"
            )
        if not isinstance(self.view, ParallelView):
            raise TypeError("view must be a ParallelView")
        if self.section is not None and not isinstance(
            self.section, SceneSectionSpec
        ):
            raise TypeError("section must be a SceneSectionSpec")
        if self.context is not None and not isinstance(
            self.context, (GeometryContext, ResolvedGeometryContext)
        ):
            raise TypeError(
                "context must be a GeometryContext or ResolvedGeometryContext"
            )
        try:
            policy = QuadricPaintPolicy(self.paint_policy)
        except (TypeError, ValueError) as exc:
            raise SceneOcclusionError("invalid scene paint policy") from exc
        max_chord_error = _positive(self.max_chord_error, "max_chord_error")
        if (
            isinstance(self.max_surface_segments, bool)
            or not isinstance(self.max_surface_segments, int)
            or self.max_surface_segments < 8
        ):
            raise SceneOcclusionError(
                "max_surface_segments must be an integer of at least eight"
            )
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "boundary_sources", sources)
        object.__setattr__(self, "tangent_spheres", bindings)
        object.__setattr__(self, "paint_policy", policy)
        object.__setattr__(self, "max_chord_error", max_chord_error)


@dataclass(frozen=True, slots=True)
class SceneOcclusionFrame:
    """One final renderer-neutral painter graph for a registered scene."""

    scene_id: str
    dispatch_path: SceneOcclusionPath
    global_frame: GlobalQuadricFrame
    section_frame: QuadricSectionCompositingFrame | None
    nested_parent_frame: NestedTangentParentFrame | None
    boundary_frame: QuadricBoundaryCompositingFrame | None
    parent_item_ids: tuple[str, ...]
    order_relations: tuple[QuadricPaintRelation, ...]
    draw_order: tuple[str, ...]
    surface_layering_authoritative: bool
    curve_visibility_authoritative: bool
    physical_surface_visibility_authoritative: bool
    schema: str = SCENE_OCCLUSION_FRAME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCENE_OCCLUSION_FRAME_SCHEMA:
            raise SceneOcclusionError("invalid scene occlusion frame schema")
        object.__setattr__(self, "scene_id", _identity(self.scene_id, "scene_id"))
        if not isinstance(self.dispatch_path, SceneOcclusionPath):
            raise TypeError("dispatch_path must be a SceneOcclusionPath")
        if not isinstance(self.global_frame, GlobalQuadricFrame):
            raise TypeError("global_frame must be a GlobalQuadricFrame")
        if self.section_frame is not None and not isinstance(
            self.section_frame, QuadricSectionCompositingFrame
        ):
            raise TypeError(
                "section_frame must be a QuadricSectionCompositingFrame"
            )
        if self.nested_parent_frame is not None and not isinstance(
            self.nested_parent_frame, NestedTangentParentFrame
        ):
            raise TypeError(
                "nested_parent_frame must be a NestedTangentParentFrame"
            )
        if self.boundary_frame is not None and not isinstance(
            self.boundary_frame, QuadricBoundaryCompositingFrame
        ):
            raise TypeError(
                "boundary_frame must be a QuadricBoundaryCompositingFrame"
            )
        parent_ids = tuple(self.parent_item_ids)
        if len(parent_ids) != len(set(parent_ids)):
            raise SceneOcclusionError("parent painter identities must be unique")
        draw_order = tuple(self.draw_order)
        active_ids = (
            set(parent_ids)
            if self.boundary_frame is None
            else set(self.boundary_frame.draw_order)
        )
        if len(draw_order) != len(set(draw_order)) or set(draw_order) != active_ids:
            raise SceneOcclusionError(
                "scene draw order must cover every active item exactly once"
            )
        relations = tuple(self.order_relations)
        if not all(isinstance(item, QuadricPaintRelation) for item in relations):
            raise TypeError("order_relations must contain QuadricPaintRelation")
        relation_keys = tuple(
            (item.far_item_id, item.near_item_id, item.reason)
            for item in relations
        )
        if relation_keys != tuple(sorted(relation_keys)):
            raise SceneOcclusionError("scene painter relations must be sorted")
        if any(
            item.far_item_id not in active_ids
            or item.near_item_id not in active_ids
            for item in relations
        ):
            raise SceneOcclusionError(
                "scene painter relation references an inactive item"
            )
        rank = {item_id: index for index, item_id in enumerate(draw_order)}
        if any(
            rank[item.far_item_id] >= rank[item.near_item_id]
            for item in relations
        ):
            raise SceneOcclusionError(
                "scene draw order violates a painter relation"
            )
        if self.dispatch_path is SceneOcclusionPath.GLOBAL_DISJOINT:
            expected_parent_ids = self.global_frame.frame.draw_order
            expected_parent_relations = self.global_frame.frame.order_relations
            expected_parent_order = self.global_frame.frame.draw_order
        elif self.dispatch_path is SceneOcclusionPath.SINGLE_SECTION:
            if self.section_frame is None:
                raise SceneOcclusionError(
                    "single-section frames require section evidence"
                )
            expected_parent_ids = self.section_frame.draw_order
            expected_parent_relations = self.section_frame.order_relations
            expected_parent_order = self.section_frame.draw_order
        else:
            if self.nested_parent_frame is None:
                raise SceneOcclusionError(
                    "nested frames require parent evidence"
                )
            expected_parent_ids = self.nested_parent_frame.parent_item_ids
            expected_parent_relations = self.nested_parent_frame.order_relations
            expected_parent_order = self.nested_parent_frame.draw_order
        if parent_ids != expected_parent_ids:
            raise SceneOcclusionError(
                "scene parent identities disagree with the selected parent frame"
            )

        if self.boundary_frame is not None:
            if (
                parent_ids != self.boundary_frame.parent_item_ids
                or relations != self.boundary_frame.order_relations
                or draw_order != self.boundary_frame.draw_order
            ):
                raise SceneOcclusionError(
                    "scene fields disagree with the authoritative boundary frame"
                )
            relaxed_parent_pairs: set[tuple[str, str]] = set()
            if self.section_frame is not None:
                anchors = _section_anchors(self.section_frame)
                relaxed_parent_pairs = {
                    (anchors.surface_back, anchors.plane_outside),
                    (anchors.outline_outside, anchors.plane_between),
                }
            boundary_pairs = {
                (item.far_item_id, item.near_item_id)
                for item in self.boundary_frame.order_relations
            }
            required_parent_pairs = {
                (item.far_item_id, item.near_item_id)
                for item in expected_parent_relations
                if (item.far_item_id, item.near_item_id)
                not in relaxed_parent_pairs
            }
            if not required_parent_pairs.issubset(boundary_pairs):
                raise SceneOcclusionError(
                    "boundary frame omits a selected parent painter relation"
                )
            if any(
                rank[item.far_item_id] >= rank[item.near_item_id]
                and (item.far_item_id, item.near_item_id)
                not in relaxed_parent_pairs
                for item in expected_parent_relations
            ):
                raise SceneOcclusionError(
                    "boundary draw order violates a selected parent relation"
                )
        else:
            if (
                relations != expected_parent_relations
                or draw_order != expected_parent_order
            ):
                raise SceneOcclusionError(
                    "scene fields disagree with the selected parent frame"
                )
        if any(
            not isinstance(value, bool)
            for value in (
                self.surface_layering_authoritative,
                self.curve_visibility_authoritative,
                self.physical_surface_visibility_authoritative,
            )
        ):
            raise TypeError("scene authority flags must be boolean")
        if (
            not self.surface_layering_authoritative
            or not self.curve_visibility_authoritative
        ):
            raise SceneOcclusionError(
                "a scene frame cannot disable its certified authority flags"
            )
        if self.dispatch_path is SceneOcclusionPath.GLOBAL_DISJOINT:
            if self.section_frame is not None or self.nested_parent_frame is not None:
                raise SceneOcclusionError(
                    "global-disjoint frames cannot carry section or nested evidence"
                )
            if not self.physical_surface_visibility_authoritative:
                raise SceneOcclusionError(
                    "global-disjoint frames require physical surface authority"
                )
        elif self.dispatch_path is SceneOcclusionPath.SINGLE_SECTION:
            if self.section_frame is None or self.nested_parent_frame is not None:
                raise SceneOcclusionError(
                    "single-section frames require only section evidence"
                )
            if not self.physical_surface_visibility_authoritative:
                raise SceneOcclusionError(
                    "single-section frames require physical surface authority"
                )
            if self.section_frame.base_frame != self.global_frame.frame:
                raise SceneOcclusionError(
                    "section evidence must derive from the selected global frame"
                )
        else:
            if self.nested_parent_frame is None or self.section_frame is None:
                raise SceneOcclusionError(
                    "nested tangent frames require section and parent evidence"
                )
            if self.physical_surface_visibility_authoritative:
                raise SceneOcclusionError(
                    "nested teaching layers cannot claim physical surface authority"
                )
            nested = self.nested_parent_frame
            section = self.section_frame
            boundary = self.boundary_frame
            if boundary is None:
                raise SceneOcclusionError(
                    "nested tangent frames require contact boundary evidence"
                )
            if section.base_frame != self.global_frame.frame:
                raise SceneOcclusionError(
                    "section evidence must derive from the selected global frame"
                )
            pair_visibility = nested.sphere_pair_frame.frame.visibility
            selected_visibility = self.global_frame.frame.visibility
            if (
                pair_visibility.projection_matrix
                != selected_visibility.projection_matrix
                or pair_visibility.view_direction
                != selected_visibility.view_direction
                or nested.sphere_pair_frame.geometry_context
                != self.global_frame.geometry_context
            ):
                raise SceneOcclusionError(
                    "sphere-pair evidence must share the selected view and context"
                )
            global_surface_ids = tuple(
                item.surface_id for item in self.global_frame.frame.surface_items
            )
            if (
                nested.mother_surface_id != section.surface_id
                or global_surface_ids != (section.surface_id,)
            ):
                raise SceneOcclusionError(
                    "nested, section, and global evidence must share one mother"
                )
            if parent_ids != nested.parent_item_ids:
                raise SceneOcclusionError(
                    "scene parent identities disagree with nested parent evidence"
                )
            if (
                nested.surface_items[nested.mother_surface_id]
                != section.paint_items.surface_front
            ):
                raise SceneOcclusionError(
                    "nested mother mapping disagrees with the section front sheet"
                )
            expected_parent_ids = {
                *section.draw_order,
                *(
                    item.sphere_item_id
                    for item in nested.contacts
                ),
            }
            if set(parent_ids) != expected_parent_ids:
                raise SceneOcclusionError(
                    "nested parent items must be the section plus both spheres"
                )
            required_nested_relations = _required_nested_parent_relations(
                section,
                nested.contacts,
                nested.sphere_pair_frame,
                nested.surface_items,
            )
            nested_pairs = {
                (item.far_item_id, item.near_item_id)
                for item in nested.order_relations
            }
            if any(
                (item.far_item_id, item.near_item_id) not in nested_pairs
                for item in required_nested_relations
            ):
                raise SceneOcclusionError(
                    "nested parent frame omits a mandatory painter relation"
                )
            source_map = {item.source_id: item for item in boundary.sources}
            for contact in nested.contacts:
                source = source_map.get(contact.contact_source_id)
                if (
                    source is None
                    or source.owner_surface_id != contact.sphere_surface_id
                    or not isinstance(source.curve, EllipseArcCurve)
                    or source.occlusion_scope
                    not in {
                        BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
                        BoundaryOcclusionScope.ALL_SURFACES,
                    }
                ):
                    raise SceneOcclusionError(
                        "nested contact evidence disagrees with boundary sources"
                    )
        object.__setattr__(self, "parent_item_ids", parent_ids)
        object.__setattr__(self, "order_relations", relations)
        object.__setattr__(self, "draw_order", draw_order)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sceneId": self.scene_id,
            "dispatchPath": self.dispatch_path.value,
            "globalFrame": self.global_frame.to_dict(),
            "sectionFrame": (
                None if self.section_frame is None else self.section_frame.to_dict()
            ),
            "nestedParentFrame": (
                None
                if self.nested_parent_frame is None
                else self.nested_parent_frame.to_dict()
            ),
            "boundaryFrame": (
                None if self.boundary_frame is None else self.boundary_frame.to_dict()
            ),
            "parentItemIds": list(self.parent_item_ids),
            "orderRelations": [item.to_dict() for item in self.order_relations],
            "drawOrder": list(self.draw_order),
            "surfaceLayeringAuthoritative": self.surface_layering_authoritative,
            "curveVisibilityAuthoritative": self.curve_visibility_authoritative,
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


def _section_anchors(
    section: QuadricSectionCompositingFrame,
) -> BoundarySectionAnchors:
    items = section.paint_items
    outlines = items.outline_by_role
    return BoundarySectionAnchors(
        items.plane_behind,
        outlines[PlaneDepthRole.BEHIND_SURFACE],
        items.surface_back,
        items.plane_outside,
        outlines[PlaneDepthRole.OUTSIDE_PROJECTION],
        items.plane_between,
        outlines[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
        items.surface_front,
        items.plane_front,
        outlines[PlaneDepthRole.IN_FRONT_OF_SURFACE],
    )


def _merge_relations(
    relations: Sequence[QuadricPaintRelation],
) -> tuple[QuadricPaintRelation, ...]:
    reasons: dict[tuple[str, str], set[str]] = {}
    for relation in relations:
        if not isinstance(relation, QuadricPaintRelation):
            raise TypeError("relations must contain QuadricPaintRelation")
        pair = (relation.far_item_id, relation.near_item_id)
        reverse = (pair[1], pair[0])
        if reverse in reasons:
            raise SceneOcclusionError(
                "scene painter relations contain contradictory evidence: "
                f"{pair[0]!r}, {pair[1]!r}"
            )
        reasons.setdefault(pair, set()).add(relation.reason)
    return tuple(
        QuadricPaintRelation(far, near, "+".join(sorted(values)))
        for (far, near), values in sorted(reasons.items())
    )


def _attach_boundaries(
    sources: tuple[QuadricBoundarySource, ...],
    surfaces: tuple[QuadricSurfaceSpec, ...],
    view: ParallelView,
    *,
    context: ResolvedGeometryContext,
    paint_policy: QuadricPaintPolicy,
    parent_item_ids: tuple[str, ...],
    parent_relations: tuple[QuadricPaintRelation, ...],
    surface_item_by_id: dict[str, str],
    section: QuadricSectionCompositingFrame | None,
    nested_parent: NestedTangentParentFrame | None,
) -> QuadricBoundaryCompositingFrame | None:
    if not sources:
        return None
    spans = compute_boundary_visibility(
        sources,
        surfaces,
        view,
        context=context,
    )
    rank_one_group: QuadricRankOneSectionSourceGroup | None = None
    mother = None
    if section is not None:
        mother = next(
            item for item in surfaces if item.surface_id == section.surface_id
        )
        if section.projection_kind is PlanePatchProjectionKind.LINE:
            rank_one_group = certify_rank_one_section_boundary_sources(
                sources,
                section,
                view,
                surface=mother,
                context=context,
            )
    crossings = compute_quadric_boundary_crossings(
        sources,
        spans,
        view,
        paint_policy=paint_policy,
        context=context,
        rank_one_section_source_groups=(
            () if rank_one_group is None else (rank_one_group,)
        ),
    )
    section_spans = None
    anchors = None
    if section is not None:
        anchors = _section_anchors(section)
        section_spans = (
            {}
            if section.projection_kind is PlanePatchProjectionKind.LINE
            else compute_boundary_section_spans(
                sources,
                section,
                view,
                crossings,
                surface=mother,
                visibility_spans_by_source=spans,
                context=context,
            )
        )
    provisional = compute_quadric_boundary_compositing(
        sources,
        spans,
        paint_policy=paint_policy,
        parent_item_ids=parent_item_ids,
        parent_relations=parent_relations,
        surface_item_by_id=surface_item_by_id,
        crossings=crossings,
        section_anchors=anchors,
        section_spans_by_source=section_spans,
        rank_one_section_source_group=rank_one_group,
        parameter_tolerance=context.epsilon(GeometryQuantity.PARAMETER),
    )
    if nested_parent is None:
        return provisional

    source_map = {item.source_id: item for item in sources}
    sphere_item_by_id = {
        item.sphere_surface_id: item.sphere_item_id
        for item in nested_parent.contacts
    }
    sphere_by_id = {
        item.surface_id: item
        for item in surfaces
        if isinstance(item, SphereSpec) and item.surface_id in sphere_item_by_id
    }
    relations = list(provisional.order_relations)
    for fragment in provisional.fragments:
        if not fragment.painted:
            continue
        source = source_map[fragment.source_id]
        point = source.curve.point(fragment.interval.midpoint)
        for sphere_id, sphere_item in sorted(sphere_item_by_id.items()):
            owns_fragment = source.owner_surface_id == sphere_id
            sphere_occludes = sphere_id in fragment.occluder_surface_ids
            ray_overlaps_sphere = bool(
                sphere_by_id[sphere_id].ray_hits(
                    point,
                    view.view_direction,
                    context=context,
                    forward_only=False,
                )
            )
            if paint_policy is QuadricPaintPolicy.DIAGRAMMATIC:
                if not owns_fragment and not ray_overlaps_sphere:
                    continue
                relations.append(
                    QuadricPaintRelation(
                        sphere_item,
                        fragment.item_id,
                        "diagrammatic_boundary_after_registered_sphere",
                    )
                )
            elif sphere_occludes:
                relations.append(
                    QuadricPaintRelation(
                        fragment.item_id,
                        sphere_item,
                        "boundary_hidden_by_registered_sphere",
                    )
                )
            elif owns_fragment or ray_overlaps_sphere:
                relations.append(
                    QuadricPaintRelation(
                        sphere_item,
                        fragment.item_id,
                        "boundary_in_front_of_registered_sphere",
                    )
                )
    normalized = _merge_relations(relations)
    draw_order = stable_topological_sort(
        sorted(provisional.draw_order),
        tuple(
            PainterConstraint(item.far_item_id, item.near_item_id)
            for item in normalized
        ),
        key=lambda item_id: item_id,
    )
    return replace(
        provisional,
        order_relations=normalized,
        draw_order=draw_order,
    )


class SceneOcclusionCoordinator:
    """Explicitly dispatch one request and return its unique painter graph."""

    def compute_frame(self, request: SceneOcclusionRequest) -> SceneOcclusionFrame:
        if not isinstance(request, SceneOcclusionRequest):
            raise TypeError("request must be a SceneOcclusionRequest")
        surfaces = tuple(sorted(request.surfaces, key=lambda item: item.surface_id))
        by_surface = {item.surface_id: item for item in surfaces}
        section_spec = request.section

        if section_spec is None:
            if request.tangent_spheres:
                raise SceneOcclusionError(
                    "nested tangent spheres require an explicit section"
                )
            path = SceneOcclusionPath.GLOBAL_DISJOINT
            global_frame = compute_global_quadric_frame(
                (),
                surfaces,
                request.view,
                context=request.context,
                paint_policy=request.paint_policy,
                max_chord_error=request.max_chord_error,
                max_segments=request.max_surface_segments,
            )
            parent_ids = global_frame.frame.draw_order
            parent_relations = global_frame.frame.order_relations
            surface_map = {
                item.surface_id: item.item_id
                for item in global_frame.frame.surface_items
            }
            boundary = _attach_boundaries(
                request.boundary_sources,
                surfaces,
                request.view,
                context=global_frame.geometry_context,
                paint_policy=request.paint_policy,
                parent_item_ids=parent_ids,
                parent_relations=parent_relations,
                surface_item_by_id=surface_map,
                section=None,
                nested_parent=None,
            )
            final_relations = (
                parent_relations if boundary is None else boundary.order_relations
            )
            final_order = parent_ids if boundary is None else boundary.draw_order
            return SceneOcclusionFrame(
                request.scene_id,
                path,
                global_frame,
                None,
                None,
                boundary,
                parent_ids,
                final_relations,
                final_order,
                True,
                True,
                True,
            )

        mother = by_surface.get(section_spec.mother_surface_id)
        if mother is None:
            raise SceneOcclusionError(
                "section mother_surface_id does not name a registered surface"
            )
        mother_global = compute_global_quadric_frame(
            (),
            (mother,),
            request.view,
            context=request.context,
            paint_policy=request.paint_policy,
            max_chord_error=request.max_chord_error,
            max_segments=request.max_surface_segments,
        )
        section_frame = compute_quadric_section_compositing(
            mother_global.frame,
            mother,
            section_spec.plane,
            section_spec.patch,
            request.view,
            context=mother_global.geometry_context,
            max_screen_error=request.max_chord_error,
        )
        other_surfaces = tuple(
            item for item in surfaces if item.surface_id != mother.surface_id
        )
        if not other_surfaces and not request.tangent_spheres:
            path = SceneOcclusionPath.SINGLE_SECTION
            surface_map = {
                mother.surface_id: section_frame.paint_items.surface_front
            }
            boundary = _attach_boundaries(
                request.boundary_sources,
                surfaces,
                request.view,
                context=mother_global.geometry_context,
                paint_policy=request.paint_policy,
                parent_item_ids=section_frame.draw_order,
                parent_relations=section_frame.order_relations,
                surface_item_by_id=surface_map,
                section=section_frame,
                nested_parent=None,
            )
            final_relations = (
                section_frame.order_relations
                if boundary is None
                else boundary.order_relations
            )
            final_order = (
                section_frame.draw_order if boundary is None else boundary.draw_order
            )
            return SceneOcclusionFrame(
                request.scene_id,
                path,
                mother_global,
                section_frame,
                None,
                boundary,
                section_frame.draw_order,
                final_relations,
                final_order,
                True,
                True,
                True,
            )

        binding_ids = {
            item.sphere_surface_id for item in request.tangent_spheres
        }
        other_ids = {item.surface_id for item in other_surfaces}
        nested_shape = (
            isinstance(mother, (ConeSpec, CylinderSpec))
            and len(other_surfaces) == 2
            and all(isinstance(item, SphereSpec) for item in other_surfaces)
            and len(request.tangent_spheres) == 2
            and binding_ids == other_ids
        )
        if not nested_shape:
            raise SceneOcclusionError(
                "section scenes with additional surfaces require exactly two "
                "registered tangent spheres owned by one cone or cylinder mother"
            )
        path = SceneOcclusionPath.NESTED_TANGENT_SECTION
        nested_parent = compute_nested_tangent_parent_frame(
            mother,
            other_surfaces,  # type: ignore[arg-type]
            section_spec.plane,
            section_frame,
            request.boundary_sources,
            request.tangent_spheres,
            request.view,
            context=mother_global.geometry_context,
            max_chord_error=request.max_chord_error,
            max_surface_segments=request.max_surface_segments,
        )
        boundary = _attach_boundaries(
            request.boundary_sources,
            surfaces,
            request.view,
            context=mother_global.geometry_context,
            paint_policy=request.paint_policy,
            parent_item_ids=nested_parent.parent_item_ids,
            parent_relations=nested_parent.order_relations,
            surface_item_by_id=nested_parent.surface_items,
            section=section_frame,
            nested_parent=nested_parent,
        )
        final_relations = (
            nested_parent.order_relations
            if boundary is None
            else boundary.order_relations
        )
        final_order = (
            nested_parent.draw_order if boundary is None else boundary.draw_order
        )
        return SceneOcclusionFrame(
            request.scene_id,
            path,
            mother_global,
            section_frame,
            nested_parent,
            boundary,
            nested_parent.parent_item_ids,
            final_relations,
            final_order,
            True,
            True,
            False,
        )


def compute_scene_occlusion_frame(
    request: SceneOcclusionRequest,
) -> SceneOcclusionFrame:
    """Convenience wrapper for the stateless scene coordinator."""

    return SceneOcclusionCoordinator().compute_frame(request)


def canonical_scene_occlusion_json(frame: SceneOcclusionFrame) -> str:
    if not isinstance(frame, SceneOcclusionFrame):
        raise TypeError("frame must be a SceneOcclusionFrame")
    return frame.canonical_json()


__all__ = [
    "SCENE_OCCLUSION_FRAME_SCHEMA",
    "SceneOcclusionCoordinator",
    "SceneOcclusionError",
    "SceneOcclusionFrame",
    "SceneOcclusionPath",
    "SceneOcclusionRequest",
    "SceneSectionSpec",
    "canonical_scene_occlusion_json",
    "compute_scene_occlusion_frame",
]
