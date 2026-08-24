"""Analytic display-patch fitting for finite quadric entities.

``SectionPlane`` remains the infinite mathematical plane used by the section
solver.  This module derives a separate finite ``PlaneDisplayPatchSpec`` for
rendering only.  The rectangle is fitted in the plane's stable ``(u, v)``
coordinates from exact support intervals of the *finite solids*; no sampled
outline or rendered geometry is fed back into mathematical truth.

The supported solids are spheres, capped finite cylinders, and convex
single-nappe finite cones/frusta.  A finite double-nappe cone is deliberately
rejected: although its coordinate extrema can be bounded, it is not one convex
entity and therefore has no single support contract shared by the global
occlusion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import Sequence

import numpy as np

from .contract import (
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)


PLANE_PATCH_FIT_SCHEMA = "manim-quadric-plane-patch-fit/v1"
DEFAULT_PLANE_PATCH_MARGIN_RATIO = 0.1


QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec


class PlanePatchFitError(ValueError):
    """A finite display rectangle cannot be certified without guessing."""


def _canonical_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise PlanePatchFitError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanePatchFitError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise PlanePatchFitError(f"{label} must be finite")
    # Canonical JSON must not depend on the sign bit of a computed zero.
    return 0.0 if result == 0.0 else result


def _non_negative(value: object, label: str) -> float:
    result = _canonical_float(value, label)
    if result < 0.0:
        raise PlanePatchFitError(f"{label} must be non-negative")
    return result


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanePatchFitError(f"{label} must be a non-empty string")
    return value.strip()


def _direction3(value: object) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise PlanePatchFitError("support direction must contain three finite values")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanePatchFitError(
            "support direction must contain three finite values"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise PlanePatchFitError(
            "support direction must contain three finite values"
        )
    if float(np.linalg.norm(result)) == 0.0:
        raise PlanePatchFitError("support direction must be non-zero")
    return result


def _validate_surface(surface: object) -> QuadricSurfaceSpec:
    if not isinstance(surface, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError(
            "surfaces must contain only SphereSpec, CylinderSpec, or ConeSpec"
        )
    if isinstance(surface, ConeSpec):
        lower, upper = surface.axial_range
        if lower < 0.0 < upper:
            raise PlanePatchFitError(
                f"surface {surface.surface_id!r} crosses the cone apex and has "
                "two nappes; split it into separate single-nappe entities before "
                "fitting a display patch"
            )
    return surface


def _surface_kind(surface: QuadricSurfaceSpec) -> str:
    if isinstance(surface, SphereSpec):
        return "sphere"
    if isinstance(surface, CylinderSpec):
        return "cylinder"
    return "cone"


def _support_value(surface: QuadricSurfaceSpec, direction: np.ndarray) -> float:
    """Return ``max(dot(direction, point))`` over one finite solid."""

    vector = _direction3(direction)
    if isinstance(surface, SphereSpec):
        center = np.asarray(surface.center, dtype=float)
        value = float(np.dot(vector, center)) + surface.radius * float(
            np.linalg.norm(vector)
        )
        return _canonical_float(value, "sphere support value")

    frame = surface.frame
    base = np.asarray(
        surface.origin if isinstance(surface, CylinderSpec) else surface.apex,
        dtype=float,
    )
    axis = np.asarray(frame.z_axis, dtype=float)
    x_axis = np.asarray(frame.x_axis, dtype=float)
    y_axis = np.asarray(frame.y_axis, dtype=float)
    axial_coefficient = float(np.dot(vector, axis))
    radial_coefficient = float(
        np.hypot(np.dot(vector, x_axis), np.dot(vector, y_axis))
    )
    base_value = float(np.dot(vector, base))
    candidates: list[float] = []
    for axial in surface.axial_range:
        radius = (
            surface.radius
            if isinstance(surface, CylinderSpec)
            else abs(axial) * surface.slope
        )
        candidates.append(
            _canonical_float(
                base_value + axial * axial_coefficient + radius * radial_coefficient,
                f"{_surface_kind(surface)} support value",
            )
        )
    return max(candidates)


def finite_surface_support_interval(
    surface: QuadricSurfaceSpec,
    direction: Sequence[float],
) -> tuple[float, float]:
    """Return the exact finite-solid interval of one linear functional.

    The result is ``(minimum, maximum)`` for ``dot(direction, point)``.  Four
    calls of this function (``+u, -u, +v, -v`` through two intervals) are all
    that patch fitting needs per entity; its cost is independent of display
    resolution.
    """

    checked = _validate_surface(surface)
    vector = _direction3(direction)
    maximum = _support_value(checked, vector)
    minimum = -_support_value(checked, -vector)
    minimum = _canonical_float(minimum, "support minimum")
    if minimum > maximum:
        raise PlanePatchFitError("analytic support interval is inverted")
    return minimum, maximum


@dataclass(frozen=True, slots=True)
class SurfacePlaneExtents:
    """Exact ``u/v`` bounds of one finite solid relative to a section plane."""

    surface_id: str
    surface_kind: str
    minimum_u: float
    maximum_u: float
    minimum_v: float
    maximum_v: float

    def __post_init__(self) -> None:
        surface_id = _identity(self.surface_id, "surface_id")
        if self.surface_kind not in {"sphere", "cylinder", "cone"}:
            raise PlanePatchFitError("surface_kind must identify a supported quadric")
        values = tuple(
            _canonical_float(getattr(self, label), label)
            for label in ("minimum_u", "maximum_u", "minimum_v", "maximum_v")
        )
        minimum_u, maximum_u, minimum_v, maximum_v = values
        if minimum_u >= maximum_u or minimum_v >= maximum_v:
            raise PlanePatchFitError(
                "finite surface extents must have positive width and height"
            )
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "minimum_u", minimum_u)
        object.__setattr__(self, "maximum_u", maximum_u)
        object.__setattr__(self, "minimum_v", minimum_v)
        object.__setattr__(self, "maximum_v", maximum_v)

    def to_dict(self) -> dict[str, object]:
        return {
            "surfaceId": self.surface_id,
            "surfaceKind": self.surface_kind,
            "u": [self.minimum_u, self.maximum_u],
            "v": [self.minimum_v, self.maximum_v],
        }


def _plane_dict(plane: SectionPlane) -> dict[str, object]:
    u_axis, v_axis, normal = plane.basis
    return {
        "planeId": plane.plane_id,
        "point": list(plane.point),
        "normal": [float(item) for item in normal],
        "uAxis": [float(item) for item in u_axis],
        "vAxis": [float(item) for item in v_axis],
    }


def _patch_dict(patch: PlaneDisplayPatchSpec) -> dict[str, object]:
    return {
        "patchId": patch.patch_id,
        "planeId": patch.plane_id,
        "centerCoordinates": list(patch.center_coordinates),
        "halfWidth": patch.half_width,
        "halfHeight": patch.half_height,
    }


@dataclass(frozen=True, slots=True)
class FittedPlaneDisplayPatch:
    """A display-only patch plus deterministic analytic-fit evidence."""

    plane: SectionPlane
    patch: PlaneDisplayPatchSpec
    surface_extents: tuple[SurfacePlaneExtents, ...]
    margin_ratio: float = DEFAULT_PLANE_PATCH_MARGIN_RATIO
    support_evaluation_count: int = 0
    visibility_authoritative: bool = False
    schema: str = PLANE_PATCH_FIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLANE_PATCH_FIT_SCHEMA:
            raise PlanePatchFitError("invalid plane-patch fit schema")
        if not isinstance(self.plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")
        if not isinstance(self.patch, PlaneDisplayPatchSpec):
            raise TypeError("patch must be a PlaneDisplayPatchSpec")
        if self.patch.plane_id != self.plane.plane_id:
            raise PlanePatchFitError("patch plane_id does not match fitted plane")
        if not isinstance(self.surface_extents, tuple) or not self.surface_extents:
            raise PlanePatchFitError("surface_extents must be a non-empty tuple")
        if not all(isinstance(item, SurfacePlaneExtents) for item in self.surface_extents):
            raise TypeError("surface_extents must contain SurfacePlaneExtents")
        surface_ids = tuple(item.surface_id for item in self.surface_extents)
        if surface_ids != tuple(sorted(surface_ids)) or len(set(surface_ids)) != len(
            surface_ids
        ):
            raise PlanePatchFitError(
                "surface_extents must have unique, sorted surface identities"
            )
        margin = _non_negative(self.margin_ratio, "margin_ratio")
        expected_evaluations = 4 * len(self.surface_extents)
        if (
            isinstance(self.support_evaluation_count, bool)
            or not isinstance(self.support_evaluation_count, int)
            or self.support_evaluation_count != expected_evaluations
        ):
            raise PlanePatchFitError(
                "support_evaluation_count must equal four per fitted surface"
            )
        if self.visibility_authoritative is not False:
            raise PlanePatchFitError(
                "a display patch cannot be visibility-authoritative"
            )

        minimum_u = min(item.minimum_u for item in self.surface_extents)
        maximum_u = max(item.maximum_u for item in self.surface_extents)
        minimum_v = min(item.minimum_v for item in self.surface_extents)
        maximum_v = max(item.maximum_v for item in self.surface_extents)
        expected_center = (
            0.5 * (minimum_u + maximum_u),
            0.5 * (minimum_v + maximum_v),
        )
        expected_half_width = 0.5 * (maximum_u - minimum_u) * (1.0 + margin)
        expected_half_height = 0.5 * (maximum_v - minimum_v) * (1.0 + margin)
        scale = max(
            1.0,
            abs(expected_center[0]),
            abs(expected_center[1]),
            expected_half_width,
            expected_half_height,
        )
        tolerance = 128.0 * np.finfo(float).eps * scale
        if not np.allclose(
            self.patch.center_coordinates,
            expected_center,
            rtol=0.0,
            atol=tolerance,
        ) or not np.allclose(
            (self.patch.half_width, self.patch.half_height),
            (expected_half_width, expected_half_height),
            rtol=0.0,
            atol=tolerance,
        ):
            raise PlanePatchFitError(
                "patch geometry does not match analytic extents and margin"
            )
        object.__setattr__(self, "margin_ratio", margin)

    @property
    def unpadded_bounds(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (
                min(item.minimum_u for item in self.surface_extents),
                max(item.maximum_u for item in self.surface_extents),
            ),
            (
                min(item.minimum_v for item in self.surface_extents),
                max(item.maximum_v for item in self.surface_extents),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        u_bounds, v_bounds = self.unpadded_bounds
        return {
            "schema": self.schema,
            "plane": _plane_dict(self.plane),
            "surfaceExtents": [item.to_dict() for item in self.surface_extents],
            "unpaddedBounds": {"u": list(u_bounds), "v": list(v_bounds)},
            "marginRatio": self.margin_ratio,
            "supportEvaluationCount": self.support_evaluation_count,
            "visibilityAuthoritative": self.visibility_authoritative,
            "patch": _patch_dict(self.patch),
        }


def fit_plane_display_patch(
    patch_id: str,
    plane: SectionPlane,
    surfaces: Sequence[QuadricSurfaceSpec],
    *,
    margin_ratio: float = DEFAULT_PLANE_PATCH_MARGIN_RATIO,
) -> FittedPlaneDisplayPatch:
    """Fit one finite display rectangle around projected finite entities.

    Projection here means orthogonal coordinates in ``plane``'s authored
    ``u/v`` frame.  Covering the projection of each complete finite solid is a
    stronger guarantee than merely covering its section curve, so every
    finite section is covered as well.
    """

    if not isinstance(plane, SectionPlane):
        raise TypeError("plane must be a SectionPlane")
    margin = _non_negative(margin_ratio, "margin_ratio")
    if isinstance(surfaces, (str, bytes)):
        raise TypeError("surfaces must be a non-empty sequence of finite quadrics")
    try:
        checked = tuple(_validate_surface(surface) for surface in surfaces)
    except TypeError:
        raise
    if not checked:
        raise PlanePatchFitError("surfaces must not be empty")
    identities = tuple(surface.surface_id for surface in checked)
    if len(set(identities)) != len(identities):
        raise PlanePatchFitError("surface identities must be unique")

    u_axis, v_axis, _normal = plane.basis
    plane_point = np.asarray(plane.point, dtype=float)
    offsets = (
        float(np.dot(u_axis, plane_point)),
        float(np.dot(v_axis, plane_point)),
    )
    extents: list[SurfacePlaneExtents] = []
    for surface in sorted(checked, key=lambda item: item.surface_id):
        minimum_u, maximum_u = finite_surface_support_interval(surface, u_axis)
        minimum_v, maximum_v = finite_surface_support_interval(surface, v_axis)
        extents.append(
            SurfacePlaneExtents(
                surface_id=surface.surface_id,
                surface_kind=_surface_kind(surface),
                minimum_u=_canonical_float(
                    minimum_u - offsets[0], "minimum plane-u extent"
                ),
                maximum_u=_canonical_float(
                    maximum_u - offsets[0], "maximum plane-u extent"
                ),
                minimum_v=_canonical_float(
                    minimum_v - offsets[1], "minimum plane-v extent"
                ),
                maximum_v=_canonical_float(
                    maximum_v - offsets[1], "maximum plane-v extent"
                ),
            )
        )

    minimum_u = min(item.minimum_u for item in extents)
    maximum_u = max(item.maximum_u for item in extents)
    minimum_v = min(item.minimum_v for item in extents)
    maximum_v = max(item.maximum_v for item in extents)
    patch = PlaneDisplayPatchSpec(
        patch_id=_identity(patch_id, "patch_id"),
        plane_id=plane.plane_id,
        half_width=0.5 * (maximum_u - minimum_u) * (1.0 + margin),
        half_height=0.5 * (maximum_v - minimum_v) * (1.0 + margin),
        center_coordinates=(
            0.5 * (minimum_u + maximum_u),
            0.5 * (minimum_v + maximum_v),
        ),
    )
    return FittedPlaneDisplayPatch(
        plane=plane,
        patch=patch,
        surface_extents=tuple(extents),
        margin_ratio=margin,
        support_evaluation_count=4 * len(extents),
    )


def canonical_fitted_plane_display_patch_json(
    fitted: FittedPlaneDisplayPatch,
) -> str:
    if not isinstance(fitted, FittedPlaneDisplayPatch):
        raise TypeError("fitted must be a FittedPlaneDisplayPatch")
    return json.dumps(
        fitted.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "DEFAULT_PLANE_PATCH_MARGIN_RATIO",
    "FittedPlaneDisplayPatch",
    "PLANE_PATCH_FIT_SCHEMA",
    "PlanePatchFitError",
    "SurfacePlaneExtents",
    "canonical_fitted_plane_display_patch_json",
    "finite_surface_support_interval",
    "fit_plane_display_patch",
]
