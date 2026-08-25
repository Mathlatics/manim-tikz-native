"""Semantic boundary-source construction for finite quadrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, isfinite, sin, tau
from typing import Sequence

import numpy as np

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
from .curves import CircleArcCurve, SegmentCurve


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
        object.__setattr__(self, "boundary_id", self.boundary_id.strip())
        object.__setattr__(self, "surface_id", self.surface_id.strip())
        object.__setattr__(self, "azimuth", value % tau)


def curve_boundary_source(
    curve,
    *,
    source_kind: BoundarySourceKind = BoundarySourceKind.ANALYTIC_CURVE,
    semantic_kind: BoundarySemanticKind = BoundarySemanticKind.FREE_CURVE,
    occlusion_scope: BoundaryOcclusionScope = BoundaryOcclusionScope.ALL_SURFACES,
    owner_id: str | None = None,
    owner_surface_id: str | None = None,
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
        style_id=style_id,
        stable_sort_key=(source_kind.value, semantic_kind.value, source_id),
    )


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


def _cap_rim_sources(surface: QuadricSurfaceSpec) -> tuple[QuadricBoundarySource, ...]:
    result = []
    for cap in surface.end_caps:
        source_id = f"boundary:{surface.surface_id}:{cap.role}:rim"
        result.append(
            curve_boundary_source(
                CircleArcCurve(
                    source_id,
                    cap.center,
                    cap.radius,
                    cap.normal,
                    radial_axis=cap.radial_axis,
                ),
                source_kind=BoundarySourceKind.SURFACE_CAP_RIM,
                semantic_kind=BoundarySemanticKind.SURFACE_BOUNDARY,
                occlusion_scope=BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
                owner_id=cap.cap_id,
                owner_surface_id=surface.surface_id,
                style_id="style:surface-boundary",
            )
        )
    return tuple(result)


def _sphere_silhouette_source(
    surface: SphereSpec,
    view: ParallelView,
) -> QuadricBoundarySource:
    normal = np.asarray(view.view_direction, dtype=float)
    radial = np.asarray(surface.frame.x_axis, dtype=float)
    radial = radial - float(np.dot(radial, normal)) * normal
    if float(np.linalg.norm(radial)) <= 1.0e-12:
        radial = np.asarray(surface.frame.y_axis, dtype=float)
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
) -> tuple[QuadricBoundarySource, ...]:
    frame = surface.frame
    direction = np.asarray(view.view_direction, dtype=float)
    dx = float(np.dot(direction, frame.x_axis))
    dy = float(np.dot(direction, frame.y_axis))
    dz = float(np.dot(direction, frame.z_axis))
    radial_norm = float(np.hypot(dx, dy))
    if radial_norm <= 1.0e-12:
        return ()
    angles: tuple[float, ...]
    if isinstance(surface, CylinderSpec):
        base = atan2(dy, dx) + 0.5 * np.pi
        angles = (base % tau, (base + np.pi) % tau)
    else:
        midpoint = 0.5 * (surface.axial_range[0] + surface.axial_range[1])
        nappe_sign = 1.0 if midpoint >= 0.0 else -1.0
        cosine = surface.slope * nappe_sign * dz / radial_norm
        if cosine < -1.0 - 1.0e-12 or cosine > 1.0 + 1.0e-12:
            return ()
        cosine = min(1.0, max(-1.0, cosine))
        base = atan2(dy, dx)
        offset = acos(cosine)
        angles = ((base - offset) % tau, (base + offset) % tau)
    ordered = tuple(sorted(angles))
    return tuple(
        _axial_generator_source(
            surface,
            np.asarray((cos(angle), sin(angle)), dtype=float),
            index,
            semantic_kind=BoundarySemanticKind.TRUE_SILHOUETTE,
            source_kind=BoundarySourceKind.SURFACE_SILHOUETTE,
            occlusion_scope=BoundaryOcclusionScope.EXTERNAL_ONLY,
            source_id=f"boundary:{surface.surface_id}:silhouette:generator:{index}",
            style_id="style:surface-silhouette",
        )
        for index, angle in enumerate(ordered)
    )


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
    result: set[str] = {item.boundary_id for item in generators}
    for surface in surfaces:
        if include_cap_rims:
            result.update(
                f"boundary:{surface.surface_id}:{cap.role}:rim"
                for cap in surface.end_caps
            )
        if include_silhouettes:
            if isinstance(surface, SphereSpec):
                result.add(f"boundary:{surface.surface_id}:silhouette")
            else:
                result.update(
                    f"boundary:{surface.surface_id}:silhouette:generator:{index}"
                    for index in range(2)
                )
    return tuple(sorted(result))


def build_surface_boundary_sources(
    surfaces: Sequence[QuadricSurfaceSpec],
    view: ParallelView,
    generators: Sequence[GeneratorBoundarySpec] = (),
    *,
    include_cap_rims: bool = True,
    include_silhouettes: bool = True,
) -> tuple[QuadricBoundarySource, ...]:
    surface_items = tuple(sorted(surfaces, key=lambda item: item.surface_id))
    by_id = {item.surface_id: item for item in surface_items}
    result: list[QuadricBoundarySource] = []
    for surface in surface_items:
        if include_cap_rims:
            result.extend(_cap_rim_sources(surface))
        if include_silhouettes:
            if isinstance(surface, SphereSpec):
                result.append(_sphere_silhouette_source(surface, view))
            else:
                result.extend(_silhouette_generators(surface, view))
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
    "build_surface_boundary_sources",
    "curve_boundary_source",
    "plane_outline_sources",
    "surface_boundary_source_ids",
]
