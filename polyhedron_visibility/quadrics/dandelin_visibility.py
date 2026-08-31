"""Certified hidden-line visibility for finite Dandelin constructions.

This module is the narrow adapter between the already-certified Dandelin
geometry and the existing quadric visibility kernel.  It does not invent a
second ray solver or painter graph.  Instead it:

* records the positive-dimensional cone/sphere tangent contacts which make the
  generic *strictly separated* global surface sorter inapplicable;
* lowers cone boundaries, sphere silhouettes, contact circles, the finite
  section, cutting-plane outline, and optional directrices to ordinary analytic
  boundary sources;
* delegates every visible/hidden interval and projected crossing to the shared
  analytic kernels; and
* emits the existing ``QuadricBoundaryCompositingFrame`` painter graph.

The resulting frame is authoritative for curve visibility under one immutable
parallel view.  Translucent surface-fill ordering remains diagrammatic; callers
must not advertise this frame as a physical transparent-surface compositor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from math import isfinite, tau
from typing import Sequence

import numpy as np

from ..geometry import GeometryQuantity, ResolvedGeometryContext
from ..parallel_solver import ParallelView
from ..topology import ParameterInterval, assert_exact_partition
from .boundary_compositing import (
    BoundaryOcclusionScope,
    BoundarySemanticKind,
    BoundarySourceKind,
    QuadricBoundaryCompositingError,
    QuadricBoundaryCompositingFrame,
    QuadricBoundaryPaintFragment,
    QuadricBoundarySource,
    QuadricBoundaryVisibilitySpan,
    compute_boundary_visibility,
    compute_quadric_boundary_crossings,
    compute_quadric_boundary_compositing,
)
from .compositing import (
    QuadricCompositingFrame,
    QuadricCompositingError,
    QuadricPaintPolicy,
    compute_quadric_compositing,
)
from .contract import ConeSpec, PlaneDisplayPatchSpec, SphereSpec
from .dandelin import DandelinConstruction3D
from .dandelin_views import build_dandelin_section_plane_diagram
from .projection import (
    ProjectionProxyError,
    ProjectionSubdivisionError,
    build_opaque_projection_proxy,
)
from .plane_patch import PlanePatchFitError, fit_plane_display_patch
from .surface_boundaries import (
    GeneratorBoundarySpec,
    build_surface_boundary_sources,
    curve_boundary_source,
)
from .curves import SegmentCurve
from .trace import section_trace_curves
from .visibility import compute_quadric_visibility


DANDELIN_VISIBILITY_FRAME_SCHEMA = "manim-dandelin-visibility-frame/v1"
DANDELIN_TANGENT_CONTACT_SCHEMA = "manim-dandelin-tangent-contact/v1"


class DandelinVisibilityError(ValueError):
    """A Dandelin hidden-line frame cannot be certified without guessing."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DandelinVisibilityError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class DandelinTangentContactEvidence:
    """Authenticated one-dimensional contact between one sphere and one nappe."""

    sphere_id: str
    cone_surface_id: str
    contact_curve_id: str
    contact_dimension: int = 1
    equal_depth_contact: bool = True
    schema: str = DANDELIN_TANGENT_CONTACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DANDELIN_TANGENT_CONTACT_SCHEMA:
            raise DandelinVisibilityError("invalid Dandelin tangent-contact schema")
        sphere_id = _identity(self.sphere_id, "sphere_id")
        cone_id = _identity(self.cone_surface_id, "cone_surface_id")
        curve_id = _identity(self.contact_curve_id, "contact_curve_id")
        if isinstance(self.contact_dimension, bool) or self.contact_dimension != 1:
            raise DandelinVisibilityError(
                "Dandelin cone/sphere contact must be one-dimensional"
            )
        if self.equal_depth_contact is not True:
            raise DandelinVisibilityError(
                "Dandelin tangent contact must retain equal-depth evidence"
            )
        object.__setattr__(self, "sphere_id", sphere_id)
        object.__setattr__(self, "cone_surface_id", cone_id)
        object.__setattr__(self, "contact_curve_id", curve_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sphereId": self.sphere_id,
            "coneSurfaceId": self.cone_surface_id,
            "contactCurveId": self.contact_curve_id,
            "contactDimension": self.contact_dimension,
            "equalDepthContact": self.equal_depth_contact,
        }


@dataclass(frozen=True, slots=True)
class DandelinVisibilitySource:
    """One semantic Dandelin role lowered to the shared boundary contract."""

    role: str
    source_ref: str
    source: QuadricBoundarySource

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identity(self.role, "source role"))
        object.__setattr__(
            self,
            "source_ref",
            _identity(self.source_ref, "source_ref"),
        )
        if not isinstance(self.source, QuadricBoundarySource):
            raise TypeError("source must be a QuadricBoundarySource")

    @property
    def source_id(self) -> str:
        return self.source.source_id


@dataclass(frozen=True, slots=True)
class DandelinVisibilityStroke:
    """One semantic analytic stroke and its exact visibility partition."""

    role: str
    source_ref: str
    source: QuadricBoundarySource
    spans: tuple[QuadricBoundaryVisibilitySpan, ...]
    parameter_tolerance: float

    def __post_init__(self) -> None:
        role = _identity(self.role, "stroke role")
        source_ref = _identity(self.source_ref, "stroke source_ref")
        if not isinstance(self.source, QuadricBoundarySource):
            raise TypeError("source must be a QuadricBoundarySource")
        spans = tuple(self.spans)
        if not spans or not all(
            isinstance(item, QuadricBoundaryVisibilitySpan) for item in spans
        ):
            raise DandelinVisibilityError(
                "a Dandelin visibility stroke requires certified spans"
            )
        tolerance = float(self.parameter_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise DandelinVisibilityError(
                "parameter_tolerance must be finite and non-negative"
            )
        assert_exact_partition(
            self.source.curve.domain,
            (item.interval for item in spans),
            tolerance=tolerance,
        )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "parameter_tolerance", tolerance)

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def hidden_span_count(self) -> int:
        return sum(item.kind.value == "hidden" for item in self.spans)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "sourceRef": self.source_ref,
            "source": self.source.to_dict(),
            "parameterTolerance": self.parameter_tolerance,
            "spans": [item.to_dict() for item in self.spans],
        }


def _coalesced_visibility_spans(
    fragments: Sequence[QuadricBoundaryPaintFragment],
    *,
    tolerance: float,
) -> tuple[QuadricBoundaryVisibilitySpan, ...]:
    ordered = tuple(
        sorted(
            fragments,
            key=lambda item: (item.interval.start, item.interval.end, item.item_id),
        )
    )
    if not ordered:
        return ()
    result: list[QuadricBoundaryVisibilitySpan] = []
    for fragment in ordered:
        current = QuadricBoundaryVisibilitySpan(
            fragment.interval,
            fragment.surface_visibility_kind,
            fragment.occluder_surface_ids,
            fragment.depth_role,
        )
        if result:
            previous = result[-1]
            contiguous = abs(
                previous.interval.end - current.interval.start
            ) <= tolerance
            same_evidence = (
                previous.kind is current.kind
                and previous.occluder_surface_ids
                == current.occluder_surface_ids
                and previous.depth_role == current.depth_role
            )
            if contiguous and same_evidence:
                result[-1] = QuadricBoundaryVisibilitySpan(
                    ParameterInterval(
                        previous.interval.start,
                        current.interval.end,
                    ),
                    current.kind,
                    current.occluder_surface_ids,
                    current.depth_role,
                )
                continue
        result.append(current)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DandelinVisibilityFrame:
    """Renderer-neutral automatic hidden-line frame for one parallel view."""

    construction_id: str
    projection_matrix: tuple[tuple[float, float, float], ...]
    view_direction: tuple[float, float, float]
    surface_ids: tuple[str, ...]
    tangent_contacts: tuple[DandelinTangentContactEvidence, ...]
    strokes: tuple[DandelinVisibilityStroke, ...]
    compositing_frame: QuadricBoundaryCompositingFrame
    curve_visibility_authoritative: bool = True
    surface_visibility_authoritative: bool = False
    schema: str = DANDELIN_VISIBILITY_FRAME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DANDELIN_VISIBILITY_FRAME_SCHEMA:
            raise DandelinVisibilityError("invalid Dandelin visibility-frame schema")
        construction_id = _identity(self.construction_id, "construction_id")
        view = ParallelView(self.projection_matrix, self.view_direction)
        surface_ids = tuple(_identity(item, "surface_id") for item in self.surface_ids)
        if surface_ids != tuple(sorted(set(surface_ids))):
            raise DandelinVisibilityError(
                "Dandelin visibility surface_ids must be unique and sorted"
            )
        contacts = tuple(self.tangent_contacts)
        if not all(isinstance(item, DandelinTangentContactEvidence) for item in contacts):
            raise TypeError(
                "tangent_contacts must contain DandelinTangentContactEvidence"
            )
        contact_keys = tuple(
            (item.sphere_id, item.cone_surface_id, item.contact_curve_id)
            for item in contacts
        )
        if contact_keys != tuple(sorted(set(contact_keys))):
            raise DandelinVisibilityError(
                "Dandelin tangent contacts must be unique and sorted"
            )
        known_surfaces = set(surface_ids)
        if any(
            item.sphere_id not in known_surfaces
            or item.cone_surface_id not in known_surfaces
            for item in contacts
        ):
            raise DandelinVisibilityError(
                "Dandelin tangent contact references an unknown surface"
            )
        strokes = tuple(self.strokes)
        if not all(isinstance(item, DandelinVisibilityStroke) for item in strokes):
            raise TypeError("strokes must contain DandelinVisibilityStroke values")
        stroke_ids = tuple(item.source_id for item in strokes)
        if stroke_ids != tuple(sorted(set(stroke_ids))):
            raise DandelinVisibilityError(
                "Dandelin visibility strokes must be unique and sorted"
            )
        if not isinstance(
            self.compositing_frame,
            QuadricBoundaryCompositingFrame,
        ):
            raise TypeError(
                "compositing_frame must be a QuadricBoundaryCompositingFrame"
            )
        if self.compositing_frame.sources != tuple(
            item.source for item in strokes
        ):
            raise DandelinVisibilityError(
                "Dandelin strokes and painter sources disagree"
            )
        fragments_by_source: dict[str, list[QuadricBoundaryPaintFragment]] = {
            source_id: [] for source_id in stroke_ids
        }
        for fragment in self.compositing_frame.fragments:
            try:
                fragments_by_source[fragment.source_id].append(fragment)
            except KeyError as exc:
                raise DandelinVisibilityError(
                    "Dandelin painter fragment references an unknown stroke"
                ) from exc
        for stroke in strokes:
            reconstructed = _coalesced_visibility_spans(
                fragments_by_source[stroke.source_id],
                tolerance=stroke.parameter_tolerance,
            )
            if reconstructed != stroke.spans:
                raise DandelinVisibilityError(
                    f"Dandelin painter fragments disagree with visibility "
                    f"stroke {stroke.source_id!r}"
                )
        unknown_occluders = sorted(
            {
                owner
                for stroke in strokes
                for span in stroke.spans
                for owner in span.occluder_surface_ids
                if owner not in known_surfaces
            }
        )
        if unknown_occluders:
            raise DandelinVisibilityError(
                "Dandelin visibility spans name unknown surfaces: "
                + ", ".join(unknown_occluders)
            )
        if self.curve_visibility_authoritative is not True:
            raise DandelinVisibilityError(
                "Dandelin hidden-line frames must certify curve visibility"
            )
        if self.surface_visibility_authoritative is not False:
            raise DandelinVisibilityError(
                "Dandelin hidden-line frames do not certify translucent fill order"
            )
        object.__setattr__(self, "construction_id", construction_id)
        object.__setattr__(self, "projection_matrix", view.projection_matrix)
        object.__setattr__(self, "view_direction", view.view_direction)
        object.__setattr__(self, "surface_ids", surface_ids)
        object.__setattr__(self, "tangent_contacts", contacts)
        object.__setattr__(self, "strokes", strokes)

    @property
    def stroke_map(self) -> dict[str, DandelinVisibilityStroke]:
        return {item.source_id: item for item in self.strokes}

    @property
    def hidden_span_count(self) -> int:
        return sum(item.hidden_span_count for item in self.strokes)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "constructionId": self.construction_id,
            "projectionMatrix": [list(row) for row in self.projection_matrix],
            "viewDirection": list(self.view_direction),
            "surfaceIds": list(self.surface_ids),
            "curveVisibilityAuthoritative": self.curve_visibility_authoritative,
            "surfaceVisibilityAuthoritative": self.surface_visibility_authoritative,
            "tangentContacts": [item.to_dict() for item in self.tangent_contacts],
            "strokes": [item.to_dict() for item in self.strokes],
            "compositingFrame": self.compositing_frame.to_dict(),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _surface_items(
    construction: DandelinConstruction3D,
) -> tuple[ConeSpec | SphereSpec, ...]:
    surfaces: tuple[ConeSpec | SphereSpec, ...] = (
        *construction.cone.render_components,
        *construction.sphere_surfaces,
    )
    ordered = tuple(sorted(surfaces, key=lambda item: item.surface_id))
    if len({item.surface_id for item in ordered}) != len(ordered):
        raise DandelinVisibilityError(
            "Dandelin visibility surfaces must have unique identities"
        )
    return ordered


def fit_dandelin_visibility_patch(
    construction: DandelinConstruction3D,
    *,
    include_directrices: bool = True,
    margin_ratio: float = 0.14,
) -> PlaneDisplayPatchSpec:
    """Fit one finite cutting-plane patch shared by static and live views."""

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    if type(include_directrices) is not bool:
        raise TypeError("include_directrices must be a bool")
    try:
        base = fit_plane_display_patch(
            f"{construction.plane.plane_id}:dandelin-visibility-base",
            construction.plane,
            construction.cone.render_components,
            margin_ratio=margin_ratio,
        ).patch
    except PlanePatchFitError as exc:
        raise DandelinVisibilityError(
            f"Dandelin visibility patch cannot be fitted: {exc}"
        ) from exc
    if not include_directrices or not construction.directrices:
        return base
    center = np.asarray(base.center_coordinates, dtype=float)
    lower = center - np.asarray((base.half_width, base.half_height), dtype=float)
    upper = center + np.asarray((base.half_width, base.half_height), dtype=float)
    padding = 0.18 * max(base.half_width, base.half_height)
    for directrix in construction.directrices:
        point = np.asarray(directrix.point.coordinates, dtype=float)
        lower = np.minimum(lower, point - padding)
        upper = np.maximum(upper, point + padding)
    expanded_center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    if not np.all(np.isfinite((*expanded_center, *half))) or np.any(half <= 0.0):
        raise DandelinVisibilityError(
            "Dandelin directrix visibility patch has no finite positive extent"
        )
    return PlaneDisplayPatchSpec(
        f"{construction.plane.plane_id}:dandelin-visibility",
        construction.plane.plane_id,
        float(half[0]),
        float(half[1]),
        (float(expanded_center[0]), float(expanded_center[1])),
    )


def _contact_evidence(
    construction: DandelinConstruction3D,
    cone_components: Sequence[ConeSpec],
    context: ResolvedGeometryContext,
) -> tuple[DandelinTangentContactEvidence, ...]:
    epsilon = context.epsilon(GeometryQuantity.BOUNDARY)
    cone = construction.cone
    result: list[DandelinTangentContactEvidence] = []
    for record in construction.spheres:
        local = cone.frame.to_local_point(record.cone_contact_circle.center)
        axial = float(local[2])
        matches = tuple(
            item
            for item in cone_components
            if item.axial_range[0] - epsilon
            <= axial
            <= item.axial_range[1] + epsilon
        )
        if len(matches) != 1:
            raise DandelinVisibilityError(
                f"contact circle {record.cone_contact_circle.curve_id!r} does "
                "not identify exactly one finite cone component"
            )
        result.append(
            DandelinTangentContactEvidence(
                record.sphere_id,
                matches[0].surface_id,
                record.cone_contact_circle.curve_id,
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.sphere_id,
                item.cone_surface_id,
                item.contact_curve_id,
            ),
        )
    )


def certify_dandelin_tangent_contacts(
    construction: DandelinConstruction3D,
) -> tuple[DandelinTangentContactEvidence, ...]:
    """Return the canonical sphere-to-cone-component tangency mapping.

    The same certificate is consumed by hidden-line visibility and by the
    teaching-transparent surface compositor.  Keeping the mapping here avoids
    two independent nappe-selection rules drifting apart.
    """

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    components = tuple(
        sorted(
            construction.cone.render_components,
            key=lambda item: item.surface_id,
        )
    )
    return _contact_evidence(
        construction,
        components,
        construction.certification_context,
    )


def _generator_specs(
    components: Sequence[ConeSpec],
    count: int,
) -> tuple[GeneratorBoundarySpec, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise DandelinVisibilityError(
            "generator_count must be an integer of at least two"
        )
    return tuple(
        GeneratorBoundarySpec(
            f"boundary:{component.surface_id}:dandelin-generator:{index:04d}",
            component.surface_id,
            tau * index / count,
            style_id="style:dandelin-cone-wire",
        )
        for component in components
        for index in range(count)
    )


def _plane_boundary_sources(
    construction: DandelinConstruction3D,
    patch: PlaneDisplayPatchSpec,
) -> tuple[DandelinVisibilitySource, ...]:
    corners = patch.corners(construction.plane)
    ends = (*corners[1:], corners[0])
    return tuple(
        DandelinVisibilitySource(
            "plane_boundary",
            construction.plane.plane_id,
            curve_boundary_source(
                SegmentCurve(
                    f"boundary:plane:{construction.plane.plane_id}:"
                    f"dandelin-edge:{index}",
                    start,
                    end,
                ),
                source_kind=BoundarySourceKind.FEATURE_LINE,
                semantic_kind=BoundarySemanticKind.DISPLAY_FRAME,
                occlusion_scope=BoundaryOcclusionScope.ALL_SURFACES,
                owner_id=patch.patch_id,
                style_id="style:dandelin-plane-outline",
            ),
        )
        for index, (start, end) in enumerate(zip(corners, ends))
    )


def build_dandelin_visibility_sources(
    construction: DandelinConstruction3D,
    view: ParallelView,
    *,
    display_patch: PlaneDisplayPatchSpec | None = None,
    include_contact_circles: bool = True,
    include_directrices: bool = True,
    include_plane_boundary: bool = True,
    generator_count: int = 8,
) -> tuple[DandelinVisibilitySource, ...]:
    """Lower one construction to stable semantic boundary sources."""

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    for name, value in (
        ("include_contact_circles", include_contact_circles),
        ("include_directrices", include_directrices),
        ("include_plane_boundary", include_plane_boundary),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool")
    if (include_directrices or include_plane_boundary) and display_patch is None:
        raise DandelinVisibilityError(
            "Dandelin directrix or plane-boundary visibility requires one "
            "finite display patch"
        )
    if display_patch is not None:
        if not isinstance(display_patch, PlaneDisplayPatchSpec):
            raise TypeError("display_patch must be a PlaneDisplayPatchSpec or None")
        if display_patch.plane_id != construction.plane.plane_id:
            raise DandelinVisibilityError(
                "Dandelin display patch does not belong to the cutting plane"
            )

    context = construction.certification_context
    surfaces = _surface_items(construction)
    cone_components = tuple(
        item for item in surfaces if isinstance(item, ConeSpec)
    )
    sphere_ids = {
        item.surface_id for item in surfaces if isinstance(item, SphereSpec)
    }
    entries: list[DandelinVisibilitySource] = []
    for source in build_surface_boundary_sources(
        surfaces,
        view,
        _generator_specs(cone_components, generator_count),
        include_cap_rims=True,
        include_silhouettes=True,
        context=context,
    ):
        if source.owner_surface_id in sphere_ids:
            role = "sphere_silhouette"
            source_ref = source.owner_surface_id
            source = replace(
                source,
                style_id="style:dandelin-sphere-silhouette",
            )
        else:
            role = "cone_boundary"
            source_ref = construction.cone.surface_id
            source = replace(source, style_id="style:dandelin-cone-wire")
        if source_ref is None:  # pragma: no cover - surface builder invariant
            raise DandelinVisibilityError(
                "Dandelin surface boundary lost its owner"
            )
        entries.append(DandelinVisibilitySource(role, source_ref, source))

    if include_contact_circles:
        for record in construction.spheres:
            curve = record.cone_contact_circle.lower_to_analytic_curve()
            entries.append(
                DandelinVisibilitySource(
                    "contact_circle",
                    record.cone_contact_circle.curve_id,
                    curve_boundary_source(
                        curve,
                        source_kind=BoundarySourceKind.FEATURE_LINE,
                        semantic_kind=BoundarySemanticKind.TEACHING_FEATURE,
                        occlusion_scope=(
                            BoundaryOcclusionScope.OWNER_AND_EXTERNAL
                        ),
                        owner_id=curve.curve_id,
                        owner_surface_id=record.sphere_id,
                        style_id="style:dandelin-contact",
                    ),
                )
            )

    diagram = build_dandelin_section_plane_diagram(construction)
    section_ref = f"{construction.construction_id}:section"
    for curve in section_trace_curves(diagram.conic_trace):
        entries.append(
            DandelinVisibilitySource(
                "section_curve",
                section_ref,
                curve_boundary_source(
                    curve,
                    source_kind=BoundarySourceKind.SECTION_CURVE,
                    semantic_kind=BoundarySemanticKind.FREE_CURVE,
                    occlusion_scope=BoundaryOcclusionScope.ALL_SURFACES,
                    owner_id=section_ref,
                    section_surface_id=construction.cone.surface_id,
                    section_plane_id=construction.plane.plane_id,
                    style_id="style:dandelin-section",
                ),
            )
        )

    if include_directrices:
        assert display_patch is not None
        directrix_by_id = {
            item.directrix_id: item for item in construction.directrices
        }
        for curve in construction.directrix_segments(
            display_patch,
            context=context,
        ):
            directrix = directrix_by_id.get(curve.curve_id)
            if directrix is None:
                raise DandelinVisibilityError(
                    f"directrix curve {curve.curve_id!r} has no construction record"
                )
            entries.append(
                DandelinVisibilitySource(
                    "directrix",
                    directrix.directrix_id,
                    curve_boundary_source(
                        curve,
                        source_kind=BoundarySourceKind.FEATURE_LINE,
                        semantic_kind=BoundarySemanticKind.TEACHING_FEATURE,
                        occlusion_scope=BoundaryOcclusionScope.ALL_SURFACES,
                        owner_id=directrix.directrix_id,
                        style_id="style:dandelin-directrix",
                    ),
                )
            )

    if include_plane_boundary:
        assert display_patch is not None
        entries.extend(_plane_boundary_sources(construction, display_patch))

    ordered = tuple(sorted(entries, key=lambda item: item.source_id))
    source_ids = tuple(item.source_id for item in ordered)
    if len(set(source_ids)) != len(source_ids):
        raise DandelinVisibilityError(
            "Dandelin automatic visibility sources must be unique"
        )
    return ordered


def _diagrammatic_surface_constraints(
    surfaces: Sequence[ConeSpec | SphereSpec],
) -> tuple[tuple[str, str], ...]:
    surface_ids = tuple(item.surface_id for item in surfaces)
    return tuple(zip(surface_ids, surface_ids[1:]))


def _surface_compositing_base(
    surfaces: tuple[ConeSpec | SphereSpec, ...],
    view: ParallelView,
    *,
    context: ResolvedGeometryContext,
    max_chord_error: float,
    max_surface_segments: int,
) -> QuadricCompositingFrame:
    try:
        proxies = tuple(
            build_opaque_projection_proxy(
                surface,
                view,
                patch_id=f"{surface.surface_id}:opaque-projection",
                max_chord_error=max_chord_error,
                max_segments=max_surface_segments,
            )
            for surface in surfaces
        )
        visibility = compute_quadric_visibility(
            (),
            surfaces,
            view,
            context=context,
        )
        return compute_quadric_compositing(
            visibility,
            proxies,
            paint_policy=QuadricPaintPolicy.PHYSICAL,
            surface_constraints=_diagrammatic_surface_constraints(surfaces),
        )
    except (
        ProjectionProxyError,
        ProjectionSubdivisionError,
        QuadricCompositingError,
    ) as exc:
        raise DandelinVisibilityError(
            f"Dandelin diagrammatic surface base cannot be certified: {exc}"
        ) from exc


def _visibility_frame_from_compositing(
    construction: DandelinConstruction3D,
    view: ParallelView,
    sources: Sequence[DandelinVisibilitySource],
    compositing_frame: QuadricBoundaryCompositingFrame,
) -> DandelinVisibilityFrame:
    ordered_sources = tuple(sorted(sources, key=lambda item: item.source_id))
    if compositing_frame.sources != tuple(
        item.source for item in ordered_sources
    ):
        raise DandelinVisibilityError(
            "Dandelin painter frame does not match its semantic sources"
        )
    context = construction.certification_context
    fragments_by_source: dict[str, list[QuadricBoundaryPaintFragment]] = {
        item.source_id: [] for item in ordered_sources
    }
    for fragment in compositing_frame.fragments:
        try:
            fragments_by_source[fragment.source_id].append(fragment)
        except KeyError as exc:
            raise DandelinVisibilityError(
                "Dandelin painter frame contains an unknown source"
            ) from exc
    parameter_tolerance = context.epsilon(GeometryQuantity.PARAMETER)
    strokes = tuple(
        DandelinVisibilityStroke(
            item.role,
            item.source_ref,
            item.source,
            _coalesced_visibility_spans(
                fragments_by_source[item.source_id],
                tolerance=parameter_tolerance,
            ),
            parameter_tolerance,
        )
        for item in ordered_sources
    )
    surfaces = _surface_items(construction)
    contacts = certify_dandelin_tangent_contacts(construction)
    return DandelinVisibilityFrame(
        construction.construction_id,
        view.projection_matrix,
        view.view_direction,
        tuple(item.surface_id for item in surfaces),
        contacts,
        strokes,
        compositing_frame,
    )


def compute_dandelin_visibility_frame(
    construction: DandelinConstruction3D,
    view: ParallelView,
    *,
    directrix_patch: PlaneDisplayPatchSpec | None = None,
    include_contact_circles: bool = True,
    include_directrices: bool = True,
    include_plane_boundary: bool = True,
    generator_count: int = 8,
    max_chord_error: float = 1.0e-3,
    max_surface_segments: int = 2048,
) -> DandelinVisibilityFrame:
    """Build visibility and one shared fragment-level painter graph."""

    entries = build_dandelin_visibility_sources(
        construction,
        view,
        display_patch=directrix_patch,
        include_contact_circles=include_contact_circles,
        include_directrices=include_directrices,
        include_plane_boundary=include_plane_boundary,
        generator_count=generator_count,
    )
    context = construction.certification_context
    surfaces = _surface_items(construction)
    sources = tuple(item.source for item in entries)
    try:
        spans_by_source = compute_boundary_visibility(
            sources,
            surfaces,
            view,
            context=context,
        )
        crossings = compute_quadric_boundary_crossings(
            sources,
            spans_by_source,
            view,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            context=context,
        )
        surface_base = _surface_compositing_base(
            surfaces,
            view,
            context=context,
            max_chord_error=max_chord_error,
            max_surface_segments=max_surface_segments,
        )
        surface_item_by_id = {
            item.surface_id: item.item_id for item in surface_base.surface_items
        }
        compositing = compute_quadric_boundary_compositing(
            sources,
            spans_by_source,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            parent_item_ids=surface_base.draw_order,
            parent_relations=surface_base.order_relations,
            surface_item_by_id=surface_item_by_id,
            crossings=crossings,
            parameter_tolerance=context.epsilon(GeometryQuantity.PARAMETER),
        )
    except QuadricBoundaryCompositingError as exc:
        raise DandelinVisibilityError(
            f"Dandelin boundary painter graph cannot be certified: {exc}"
        ) from exc
    return _visibility_frame_from_compositing(
        construction,
        view,
        entries,
        compositing,
    )


def canonical_dandelin_visibility_json(frame: DandelinVisibilityFrame) -> str:
    if not isinstance(frame, DandelinVisibilityFrame):
        raise TypeError("frame must be a DandelinVisibilityFrame")
    return frame.canonical_json()


__all__ = [
    "DANDELIN_TANGENT_CONTACT_SCHEMA",
    "DANDELIN_VISIBILITY_FRAME_SCHEMA",
    "DandelinTangentContactEvidence",
    "DandelinVisibilityError",
    "DandelinVisibilityFrame",
    "DandelinVisibilitySource",
    "DandelinVisibilityStroke",
    "build_dandelin_visibility_sources",
    "canonical_dandelin_visibility_json",
    "certify_dandelin_tangent_contacts",
    "compute_dandelin_visibility_frame",
    "fit_dandelin_visibility_patch",
]
