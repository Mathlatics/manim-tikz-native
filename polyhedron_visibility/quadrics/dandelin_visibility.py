"""Certified hidden-line visibility for finite Dandelin constructions.

This module is the narrow adapter between the already-certified Dandelin
geometry and the existing quadric visibility kernel.  It does not invent a
second ray solver or painter graph.  Instead it:

* records the positive-dimensional cone/sphere tangent contacts which make the
  generic *strictly separated* global surface sorter inapplicable;
* lowers cone boundaries, sphere silhouettes, contact circles, the finite
  section, and optional directrices to ordinary analytic boundary sources; and
* delegates every visible/hidden interval to ``compute_boundary_visibility``.

The resulting frame is authoritative for curve visibility under one immutable
parallel view.  Translucent surface-fill ordering remains diagrammatic; callers
must not advertise this frame as a physical transparent-surface compositor.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite, tau
from typing import Sequence

import numpy as np

from ..geometry import GeometryQuantity, ResolvedGeometryContext
from ..parallel_solver import ParallelView
from ..topology import assert_exact_partition
from .boundary_compositing import (
    BoundaryOcclusionScope,
    BoundarySemanticKind,
    BoundarySourceKind,
    QuadricBoundarySource,
    QuadricBoundaryVisibilitySpan,
    compute_boundary_visibility,
)
from .contract import ConeSpec, PlaneDisplayPatchSpec, SphereSpec
from .dandelin import DandelinConstruction3D
from .dandelin_views import build_dandelin_section_plane_diagram
from .surface_boundaries import (
    GeneratorBoundarySpec,
    build_surface_boundary_sources,
    curve_boundary_source,
)
from .trace import section_trace_curves


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


@dataclass(frozen=True, slots=True)
class DandelinVisibilityFrame:
    """Renderer-neutral automatic hidden-line frame for one parallel view."""

    construction_id: str
    projection_matrix: tuple[tuple[float, float, float], ...]
    view_direction: tuple[float, float, float]
    surface_ids: tuple[str, ...]
    tangent_contacts: tuple[DandelinTangentContactEvidence, ...]
    strokes: tuple[DandelinVisibilityStroke, ...]
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


def compute_dandelin_visibility_frame(
    construction: DandelinConstruction3D,
    view: ParallelView,
    *,
    directrix_patch: PlaneDisplayPatchSpec | None = None,
    include_contact_circles: bool = True,
    include_directrices: bool = True,
    generator_count: int = 8,
) -> DandelinVisibilityFrame:
    """Reuse the existing quadric kernel for one Dandelin hidden-line frame."""

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    for name, value in (
        ("include_contact_circles", include_contact_circles),
        ("include_directrices", include_directrices),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool")
    if include_directrices and directrix_patch is None:
        raise DandelinVisibilityError(
            "automatic directrix visibility requires one finite display patch"
        )
    if directrix_patch is not None and not isinstance(
        directrix_patch, PlaneDisplayPatchSpec
    ):
        raise TypeError("directrix_patch must be a PlaneDisplayPatchSpec or None")

    context = construction.certification_context
    surfaces = _surface_items(construction)
    cone_components = tuple(
        item for item in surfaces if isinstance(item, ConeSpec)
    )
    sphere_ids = {
        item.surface_id for item in surfaces if isinstance(item, SphereSpec)
    }
    contacts = _contact_evidence(construction, cone_components, context)

    entries: list[tuple[str, str, QuadricBoundarySource]] = []
    boundary_sources = build_surface_boundary_sources(
        surfaces,
        view,
        _generator_specs(cone_components, generator_count),
        include_cap_rims=True,
        include_silhouettes=True,
        context=context,
    )
    for source in boundary_sources:
        if source.owner_surface_id in sphere_ids:
            role = "sphere_silhouette"
            source_ref = source.owner_surface_id
        else:
            role = "cone_boundary"
            source_ref = construction.cone.surface_id
        entries.append((role, source_ref, source))

    if include_contact_circles:
        for record in construction.spheres:
            curve = record.cone_contact_circle.lower_to_analytic_curve()
            entries.append(
                (
                    "contact_circle",
                    record.cone_contact_circle.curve_id,
                    curve_boundary_source(
                        curve,
                        source_kind=BoundarySourceKind.FEATURE_LINE,
                        semantic_kind=BoundarySemanticKind.TEACHING_FEATURE,
                        occlusion_scope=BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
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
            (
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
        assert directrix_patch is not None
        directrix_by_id = {
            item.directrix_id: item for item in construction.directrices
        }
        for curve in construction.directrix_segments(
            directrix_patch,
            context=context,
        ):
            directrix = directrix_by_id.get(curve.curve_id)
            if directrix is None:
                raise DandelinVisibilityError(
                    f"directrix curve {curve.curve_id!r} has no construction record"
                )
            entries.append(
                (
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

    entries.sort(key=lambda item: item[2].source_id)
    source_ids = tuple(item[2].source_id for item in entries)
    if len(set(source_ids)) != len(source_ids):
        raise DandelinVisibilityError(
            "Dandelin automatic visibility sources must be unique"
        )
    sources = tuple(item[2] for item in entries)
    spans_by_source = compute_boundary_visibility(
        sources,
        surfaces,
        view,
        context=context,
    )
    parameter_tolerance = context.epsilon(GeometryQuantity.PARAMETER)
    strokes = tuple(
        DandelinVisibilityStroke(
            role,
            source_ref,
            source,
            tuple(spans_by_source[source.source_id]),
            parameter_tolerance,
        )
        for role, source_ref, source in entries
    )
    return DandelinVisibilityFrame(
        construction.construction_id,
        view.projection_matrix,
        view.view_direction,
        tuple(item.surface_id for item in surfaces),
        contacts,
        strokes,
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
    "DandelinVisibilityStroke",
    "canonical_dandelin_visibility_json",
    "compute_dandelin_visibility_frame",
]
