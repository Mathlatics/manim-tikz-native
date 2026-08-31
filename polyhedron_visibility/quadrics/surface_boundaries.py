"""Semantic boundary-source construction for finite quadrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt, tau
from typing import Sequence

import numpy as np

from ..geometry import (
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from ..parallel_solver import ParallelView
from ..topology import ParameterInterval
from .boundary_compositing import (
    BoundaryOcclusionScope,
    BoundarySemanticKind,
    BoundarySourceKind,
    QuadricBoundaryCompositingError,
    QuadricBoundarySource,
)
from .contract import (
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from .curves import (
    CircleArcCurve,
    EllipseArcCurve,
    ParametricConicBranch,
    SegmentCurve,
)
from .sections import compute_quadric_section_boundary_curves


QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec


@dataclass(frozen=True, slots=True)
class GeneratorBoundarySpec:
    """One explicitly authored cylinder/cone generator in local azimuth form."""

    boundary_id: str
    surface_id: str
    azimuth: float
    axial_interval: tuple[float, float] | None = None
    style_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_id, str) or not self.boundary_id.strip():
            raise ValueError("boundary_id must be a non-empty string")
        if not isinstance(self.surface_id, str) or not self.surface_id.strip():
            raise ValueError("surface_id must be a non-empty string")
        value = float(self.azimuth)
        if not isfinite(value):
            raise ValueError("azimuth must be finite")
        interval = self.axial_interval
        if interval is not None:
            if len(interval) != 2:
                raise ValueError("axial_interval must contain two values")
            lower, upper = (float(item) for item in interval)
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                raise ValueError("axial_interval must be finite and increasing")
            object.__setattr__(self, "axial_interval", (lower, upper))
        style_id = self.style_id
        if style_id is not None:
            if not isinstance(style_id, str) or not style_id.strip():
                raise ValueError("style_id must be a non-empty string")
            object.__setattr__(self, "style_id", style_id.strip())
        object.__setattr__(self, "boundary_id", self.boundary_id.strip())
        object.__setattr__(self, "surface_id", self.surface_id.strip())
        object.__setattr__(self, "azimuth", value % tau)


@dataclass(frozen=True, slots=True)
class SurfaceBoundarySlotDescriptor:
    """View-independent semantic metadata for one preallocated boundary slot."""

    source_id: str
    owner_id: str
    owner_surface_id: str
    source_kind: BoundarySourceKind
    semantic_kind: BoundarySemanticKind
    occlusion_scope: BoundaryOcclusionScope
    style_id: str

    def __post_init__(self) -> None:
        for name in ("source_id", "owner_id", "owner_surface_id", "style_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "source_kind", BoundarySourceKind(self.source_kind))
        object.__setattr__(
            self,
            "semantic_kind",
            BoundarySemanticKind(self.semantic_kind),
        )
        object.__setattr__(
            self,
            "occlusion_scope",
            BoundaryOcclusionScope(self.occlusion_scope),
        )


def curve_boundary_source(
    curve,
    *,
    source_kind: BoundarySourceKind = BoundarySourceKind.ANALYTIC_CURVE,
    semantic_kind: BoundarySemanticKind = BoundarySemanticKind.FREE_CURVE,
    occlusion_scope: BoundaryOcclusionScope = BoundaryOcclusionScope.ALL_SURFACES,
    owner_id: str | None = None,
    owner_surface_id: str | None = None,
    section_surface_id: str | None = None,
    section_plane_id: str | None = None,
    style_id: str | None = None,
) -> QuadricBoundarySource:
    source_id = curve.curve_id
    return QuadricBoundarySource(
        source_id=source_id,
        curve=curve,
        source_kind=source_kind,
        semantic_kind=semantic_kind,
        occlusion_scope=occlusion_scope,
        owner_id=source_id if owner_id is None else owner_id,
        owner_surface_id=owner_surface_id,
        section_surface_id=section_surface_id,
        section_plane_id=section_plane_id,
        style_id=style_id,
        stable_sort_key=(source_kind.value, semantic_kind.value, source_id),
    )


def _same_curve_geometry(
    first,
    second,
    *,
    allow_parametric_subdomain: bool = False,
    parameter_tolerance: float = 0.0,
) -> bool:
    """Compare authored analytic structure without consulting curve identity."""

    if isinstance(first, SegmentCurve) and isinstance(second, SegmentCurve):
        if first.domain != second.domain:
            return False
        return (
            first.start == second.start and first.end == second.end
        ) or (
            first.start == second.end and first.end == second.start
        )
    if isinstance(first, EllipseArcCurve) and isinstance(
        second, EllipseArcCurve
    ):
        return (
            first.center == second.center
            and first.first_axis == second.first_axis
            and first.second_axis == second.second_axis
            and first.domain == second.domain
        )
    if isinstance(first, ParametricConicBranch) and isinstance(
        second, ParametricConicBranch
    ):
        if (
            first.parameterization != second.parameterization
            or first.plane_embedding != second.plane_embedding
        ):
            return False
        if first.domain == second.domain:
            return True
        return bool(
            allow_parametric_subdomain
            and first.domain.start >= second.domain.start - parameter_tolerance
            and first.domain.end <= second.domain.end + parameter_tolerance
        )
    return False


def _segment_cap_owners(
    curve: SegmentCurve,
    surface: QuadricSurfaceSpec,
    context: ResolvedGeometryContext,
):
    """Return caps whose finite rim contains both segment endpoints."""

    validation_epsilon = 8.0 * context.epsilon(GeometryQuantity.BOUNDARY)
    matches = []
    for cap in surface.end_caps:
        center = np.asarray(cap.center, dtype=float)
        normal = np.asarray(cap.normal, dtype=float)
        valid = True
        for endpoint in (curve.start, curve.end):
            offset = np.asarray(endpoint, dtype=float) - center
            axial = float(np.dot(offset, normal))
            radial = offset - axial * normal
            if max(
                abs(axial),
                abs(float(np.linalg.norm(radial)) - cap.radius),
            ) > validation_epsilon:
                valid = False
                break
        if valid:
            matches.append(cap)
    return tuple(matches)


def section_curve_boundary_source(
    curve,
    surface: QuadricSurfaceSpec,
    plane: SectionPlane,
    *,
    section_id: str,
    authoritative_curves: Sequence[object] | None = None,
    context=None,
    style_id: str | None = "style:curve",
) -> QuadricBoundarySource:
    """Authenticate one curve against a complete authoritative section.

    ``section_id`` selects the authoritative finite-boundary solve; it is not
    inferred from ``curve_id``.  Geometry, including reversed cap-chord
    orientation, is compared structurally.  An unrelated curve therefore
    remains a free curve even if its name resembles an internal section name.
    A frame which has already called
    :func:`compute_quadric_section_boundary_curves` may pass that complete
    tuple as ``authoritative_curves`` so every component reuses one solve.
    """

    if not isinstance(surface, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    if not isinstance(plane, SectionPlane):
        raise TypeError("plane must be a SectionPlane")
    if not isinstance(section_id, str) or not section_id.strip():
        raise ValueError("section_id must be a non-empty string")
    identity = section_id.strip()
    resolved = (
        resolve_geometry_context(context)
        if isinstance(context, ResolvedGeometryContext)
        else resolve_geometry_context(
            context,
            positions=(*surface.characteristic_points, plane.point),
        )
    )
    if authoritative_curves is None:
        expected_curves = compute_quadric_section_boundary_curves(
            identity,
            surface,
            plane,
            context=resolved,
        )
    else:
        expected_curves = tuple(authoritative_curves)
        if not all(
            isinstance(
                item,
                (SegmentCurve, EllipseArcCurve, ParametricConicBranch),
            )
            for item in expected_curves
        ):
            raise TypeError(
                "authoritative_curves must contain finite analytic curves"
            )
        expected_ids = tuple(item.curve_id for item in expected_curves)
        if len(set(expected_ids)) != len(expected_ids):
            raise QuadricBoundaryCompositingError(
                "authoritative section curves must have unique identities"
            )
    matching = tuple(
        item
        for item in expected_curves
        if _same_curve_geometry(
            curve,
            item,
            allow_parametric_subdomain=True,
            parameter_tolerance=resolved.epsilon(GeometryQuantity.PARAMETER),
        )
    )
    if len(matching) > 1:
        raise QuadricBoundaryCompositingError(
            f"section curve {curve.curve_id!r} matches multiple authoritative "
            "finite-boundary components"
        )
    if matching:
        expected = matching[0]
        if isinstance(expected, SegmentCurve):
            caps = _segment_cap_owners(expected, surface, resolved)
            if len(caps) != 1:
                raise QuadricBoundaryCompositingError(
                    f"authoritative section cap-chord {curve.curve_id!r} does "
                    "not identify exactly one finite end cap"
                )
            source_kind = BoundarySourceKind.SECTION_CAP_CHORD
            owner_id = caps[0].cap_id
        else:
            source_kind = BoundarySourceKind.SECTION_CURVE
            owner_id = identity
        return curve_boundary_source(
            curve,
            source_kind=source_kind,
            semantic_kind=BoundarySemanticKind.FREE_CURVE,
            occlusion_scope=BoundaryOcclusionScope.ALL_SURFACES,
            owner_id=owner_id,
            # Section provenance is deliberately separate from painter
            # ownership.  AREA frames therefore retain their existing
            # free-curve ordering while LINE frames can authenticate family
            # membership.
            section_surface_id=surface.surface_id,
            section_plane_id=plane.plane_id,
            style_id=style_id,
        )

    return curve_boundary_source(curve, style_id=style_id)


def plane_outline_sources(
    plane: SectionPlane,
    patch: PlaneDisplayPatchSpec,
) -> tuple[QuadricBoundarySource, ...]:
    corners = patch.corners(plane)
    ends = (*corners[1:], corners[0])
    result = []
    for edge_index, (start, end) in enumerate(zip(corners, ends)):
        source_id = f"boundary:plane:{plane.plane_id}:edge:{edge_index}"
        result.append(
            curve_boundary_source(
                SegmentCurve(source_id, start, end),
                source_kind=BoundarySourceKind.PLANE_PATCH_EDGE,
                semantic_kind=BoundarySemanticKind.DISPLAY_FRAME,
                occlusion_scope=BoundaryOcclusionScope.NONE,
                owner_id=patch.patch_id,
                style_id="style:section-outline",
            )
        )
    return tuple(result)


def _surface_rim_sources(
    surface: QuadricSurfaceSpec,
) -> tuple[QuadricBoundarySource, ...]:
    result = []
    for cap in surface.end_caps:
        source_id = f"boundary:{surface.surface_id}:{cap.role}:rim"
        result.append(
            curve_boundary_source(
                cap.boundary_circle(source_id).lower_to_analytic_curve(),
                source_kind=BoundarySourceKind.SURFACE_CAP_RIM,
                semantic_kind=BoundarySemanticKind.SURFACE_BOUNDARY,
                occlusion_scope=BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
                owner_id=cap.cap_id,
                owner_surface_id=surface.surface_id,
                style_id="style:surface-boundary",
            )
        )
    if isinstance(surface, ConeSpec):
        for rim in surface.trim_rims:
            source_id = f"boundary:{surface.surface_id}:{rim.role}:rim"
            result.append(
                curve_boundary_source(
                    rim.boundary_circle(source_id).lower_to_analytic_curve(),
                    source_kind=BoundarySourceKind.SURFACE_TRIM_RIM,
                    semantic_kind=BoundarySemanticKind.SURFACE_BOUNDARY,
                    occlusion_scope=BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
                    owner_id=rim.rim_id,
                    owner_surface_id=surface.surface_id,
                    style_id="style:surface-boundary",
                )
            )
    return tuple(result)


def _sphere_silhouette_source(
    surface: SphereSpec,
    view: ParallelView,
    context: ResolvedGeometryContext,
) -> QuadricBoundarySource:
    normal = np.asarray(view.view_direction, dtype=float)
    radial = np.asarray(view.matrix[0], dtype=float)
    radial = radial - float(np.dot(radial, normal)) * normal
    radial_length = float(np.linalg.norm(radial))
    if radial_length <= context.epsilon(GeometryQuantity.ANGULAR):
        raise QuadricBoundaryCompositingError(
            "sphere silhouette has no stable screen-horizontal phase"
        )
    radial /= radial_length
    source_id = f"boundary:{surface.surface_id}:silhouette"
    return curve_boundary_source(
        CircleArcCurve(
            source_id,
            surface.center,
            surface.radius,
            view.view_direction,
            radial_axis=tuple(float(item) for item in radial),
        ),
        source_kind=BoundarySourceKind.SURFACE_SILHOUETTE,
        semantic_kind=BoundarySemanticKind.TRUE_SILHOUETTE,
        occlusion_scope=BoundaryOcclusionScope.EXTERNAL_ONLY,
        owner_id=surface.surface_id,
        owner_surface_id=surface.surface_id,
        style_id="style:surface-silhouette",
    )


def _axial_generator_source(
    surface: CylinderSpec | ConeSpec,
    radial_local: np.ndarray,
    index: int,
    *,
    semantic_kind: BoundarySemanticKind,
    source_kind: BoundarySourceKind,
    occlusion_scope: BoundaryOcclusionScope,
    source_id: str,
    axial_interval: tuple[float, float] | None = None,
    style_id: str | None = None,
) -> QuadricBoundarySource:
    lower, upper = surface.axial_range if axial_interval is None else axial_interval
    if lower < surface.axial_range[0] or upper > surface.axial_range[1]:
        raise QuadricBoundaryCompositingError(
            f"generator {source_id!r} lies outside its finite surface"
        )
    frame = surface.frame
    axis = np.asarray(frame.z_axis, dtype=float)
    x_axis = np.asarray(frame.x_axis, dtype=float)
    y_axis = np.asarray(frame.y_axis, dtype=float)
    radial = radial_local[0] * x_axis + radial_local[1] * y_axis
    base = np.asarray(
        surface.origin if isinstance(surface, CylinderSpec) else surface.apex,
        dtype=float,
    )

    def point(axial: float) -> np.ndarray:
        radius = (
            surface.radius
            if isinstance(surface, CylinderSpec)
            else abs(axial) * surface.slope
        )
        return base + axial * axis + radius * radial

    return curve_boundary_source(
        SegmentCurve(
            source_id,
            tuple(float(item) for item in point(lower)),
            tuple(float(item) for item in point(upper)),
        ),
        source_kind=source_kind,
        semantic_kind=semantic_kind,
        occlusion_scope=occlusion_scope,
        owner_id=surface.surface_id,
        owner_surface_id=surface.surface_id,
        style_id=style_id,
    )


def _silhouette_generators(
    surface: CylinderSpec | ConeSpec,
    view: ParallelView,
    context: ResolvedGeometryContext,
) -> tuple[QuadricBoundarySource, ...]:
    frame = surface.frame
    direction = np.asarray(view.view_direction, dtype=float)
    dx = float(np.dot(direction, frame.x_axis))
    dy = float(np.dot(direction, frame.y_axis))
    dz = float(np.dot(direction, frame.z_axis))
    radial_norm = float(np.hypot(dx, dy))
    angular_epsilon = context.epsilon(GeometryQuantity.ANGULAR)
    if radial_norm <= angular_epsilon:
        return ()
    radial_direction = np.asarray((dx, dy), dtype=float) / radial_norm
    perpendicular = np.asarray(
        (-radial_direction[1], radial_direction[0]),
        dtype=float,
    )
    radial_branches: tuple[np.ndarray, ...]
    if isinstance(surface, CylinderSpec):
        radial_branches = (perpendicular, -perpendicular)
    else:
        midpoint = 0.5 * (surface.axial_range[0] + surface.axial_range[1])
        nappe_sign = 1.0 if midpoint >= 0.0 else -1.0
        cosine = surface.slope * nappe_sign * dz / radial_norm
        if cosine < -1.0 - angular_epsilon or cosine > 1.0 + angular_epsilon:
            return ()
        cosine = min(1.0, max(-1.0, cosine))
        sine = (
            0.0
            if 1.0 - abs(cosine) <= angular_epsilon
            else sqrt(max(0.0, 1.0 - cosine * cosine))
        )
        positive = cosine * radial_direction + sine * perpendicular
        if sine <= angular_epsilon:
            radial_branches = (positive,)
        else:
            negative = cosine * radial_direction - sine * perpendicular
            radial_branches = (positive, negative)
    return tuple(
        _axial_generator_source(
            surface,
            radial,
            index,
            semantic_kind=BoundarySemanticKind.TRUE_SILHOUETTE,
            source_kind=BoundarySourceKind.SURFACE_SILHOUETTE,
            occlusion_scope=BoundaryOcclusionScope.EXTERNAL_ONLY,
            source_id=f"boundary:{surface.surface_id}:silhouette:generator:{index}",
            style_id="style:surface-silhouette",
        )
        for index, radial in enumerate(radial_branches)
    )


def surface_boundary_slot_descriptors(
    surfaces: Sequence[QuadricSurfaceSpec],
    generators: Sequence[GeneratorBoundarySpec] = (),
    *,
    include_cap_rims: bool = True,
    include_silhouettes: bool = True,
) -> tuple[SurfaceBoundarySlotDescriptor, ...]:
    """Return the complete view-independent fixed boundary-slot catalog."""

    surface_items = tuple(sorted(surfaces, key=lambda item: item.surface_id))
    by_id = {item.surface_id: item for item in surface_items}
    result: list[SurfaceBoundarySlotDescriptor] = []
    for surface in surface_items:
        if include_cap_rims:
            for cap in surface.end_caps:
                result.append(
                    SurfaceBoundarySlotDescriptor(
                        f"boundary:{surface.surface_id}:{cap.role}:rim",
                        cap.cap_id,
                        surface.surface_id,
                        BoundarySourceKind.SURFACE_CAP_RIM,
                        BoundarySemanticKind.SURFACE_BOUNDARY,
                        BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
                        "style:surface-boundary",
                    )
                )
            if isinstance(surface, ConeSpec):
                for rim in surface.trim_rims:
                    result.append(
                        SurfaceBoundarySlotDescriptor(
                            f"boundary:{surface.surface_id}:{rim.role}:rim",
                            rim.rim_id,
                            surface.surface_id,
                            BoundarySourceKind.SURFACE_TRIM_RIM,
                            BoundarySemanticKind.SURFACE_BOUNDARY,
                            BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
                            "style:surface-boundary",
                        )
                    )
        if include_silhouettes:
            count = 1 if isinstance(surface, SphereSpec) else 2
            for index in range(count):
                suffix = "silhouette" if count == 1 else f"silhouette:generator:{index}"
                result.append(
                    SurfaceBoundarySlotDescriptor(
                        f"boundary:{surface.surface_id}:{suffix}",
                        surface.surface_id,
                        surface.surface_id,
                        BoundarySourceKind.SURFACE_SILHOUETTE,
                        BoundarySemanticKind.TRUE_SILHOUETTE,
                        BoundaryOcclusionScope.EXTERNAL_ONLY,
                        "style:surface-silhouette",
                    )
                )
    for spec in sorted(generators, key=lambda item: item.boundary_id):
        surface = by_id.get(spec.surface_id)
        if not isinstance(surface, (CylinderSpec, ConeSpec)):
            raise QuadricBoundaryCompositingError(
                f"generator {spec.boundary_id!r} requires a cylinder or cone"
            )
        result.append(
            SurfaceBoundarySlotDescriptor(
                spec.boundary_id,
                surface.surface_id,
                surface.surface_id,
                BoundarySourceKind.SURFACE_GENERATOR,
                BoundarySemanticKind.TEACHING_FEATURE,
                BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
                spec.style_id or "style:teaching-boundary",
            )
        )
    result.sort(key=lambda item: item.source_id)
    if len({item.source_id for item in result}) != len(result):
        raise QuadricBoundaryCompositingError(
            "surface boundary slot identities must be unique"
        )
    return tuple(result)


def _explicit_generator_source(
    spec: GeneratorBoundarySpec,
    surface: CylinderSpec | ConeSpec,
) -> QuadricBoundarySource:
    return _axial_generator_source(
        surface,
        np.asarray((cos(spec.azimuth), sin(spec.azimuth)), dtype=float),
        0,
        semantic_kind=BoundarySemanticKind.TEACHING_FEATURE,
        source_kind=BoundarySourceKind.SURFACE_GENERATOR,
        occlusion_scope=BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
        source_id=spec.boundary_id,
        axial_interval=spec.axial_interval,
        style_id=spec.style_id or "style:teaching-boundary",
    )


def surface_boundary_source_ids(
    surfaces: Sequence[QuadricSurfaceSpec],
    generators: Sequence[GeneratorBoundarySpec] = (),
    *,
    include_cap_rims: bool = True,
    include_silhouettes: bool = True,
) -> tuple[str, ...]:
    return tuple(
        item.source_id
        for item in surface_boundary_slot_descriptors(
            surfaces,
            generators,
            include_cap_rims=include_cap_rims,
            include_silhouettes=include_silhouettes,
        )
    )


def build_surface_boundary_sources(
    surfaces: Sequence[QuadricSurfaceSpec],
    view: ParallelView,
    generators: Sequence[GeneratorBoundarySpec] = (),
    *,
    include_cap_rims: bool = True,
    include_silhouettes: bool = True,
    context=None,
) -> tuple[QuadricBoundarySource, ...]:
    surface_items = tuple(sorted(surfaces, key=lambda item: item.surface_id))
    by_id = {item.surface_id: item for item in surface_items}
    resolved = None
    if include_silhouettes:
        resolved = (
            resolve_geometry_context(context)
            if isinstance(context, ResolvedGeometryContext)
            else resolve_geometry_context(
                context,
                positions=tuple(
                    point
                    for surface in surface_items
                    for point in surface.characteristic_points
                ),
            )
        )
    result: list[QuadricBoundarySource] = []
    for surface in surface_items:
        if include_cap_rims:
            result.extend(_surface_rim_sources(surface))
        if include_silhouettes:
            assert resolved is not None
            if isinstance(surface, SphereSpec):
                result.append(_sphere_silhouette_source(surface, view, resolved))
            else:
                result.extend(_silhouette_generators(surface, view, resolved))
    for spec in sorted(generators, key=lambda item: item.boundary_id):
        surface = by_id.get(spec.surface_id)
        if not isinstance(surface, (CylinderSpec, ConeSpec)):
            raise QuadricBoundaryCompositingError(
                f"generator {spec.boundary_id!r} requires a cylinder or cone"
            )
        result.append(_explicit_generator_source(spec, surface))
    result.sort(key=lambda item: item.source_id)
    if len({item.source_id for item in result}) != len(result):
        raise QuadricBoundaryCompositingError(
            "surface boundary source identities must be unique"
        )
    return tuple(result)


__all__ = [
    "GeneratorBoundarySpec",
    "SurfaceBoundarySlotDescriptor",
    "build_surface_boundary_sources",
    "curve_boundary_source",
    "plane_outline_sources",
    "section_curve_boundary_source",
    "surface_boundary_slot_descriptors",
    "surface_boundary_source_ids",
]
