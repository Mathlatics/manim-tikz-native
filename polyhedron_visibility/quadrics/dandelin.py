"""Certified Dandelin-sphere constructions for finite right circular cones.

The construction in this module is renderer-neutral.  It derives the spheres,
their plane-contact points (the conic foci), the cone-contact circles, and the
corresponding directrices from one authored :class:`ConeSpec` and
:class:`SectionPlane`.

The finite cone remains part of the mathematical contract.  A sphere that
would cross an authored terminal plane is rejected instead of being silently
clipped or replaced by a visually plausible smaller sphere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import cos, isfinite, sin, sqrt
from typing import Sequence

import numpy as np

from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from .conics import ConicKind
from .contract import (
    ConeModel,
    ConeSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from .curves import CurveContractError, SegmentCurve
from .planar_curves import Circle3DSpec, PlanarFrame3D, PlanarPoint3D
from .sections import (
    QuadricSectionError,
    compute_quadric_section,
    compute_quadric_section_boundary,
)
from .trace import FiniteSectionTopology


DANDELIN_DIRECTRIX_SCHEMA = "manim-dandelin-directrix-3d/v1"
DANDELIN_SPHERE_SCHEMA = "manim-dandelin-sphere-3d/v1"
DANDELIN_CONSTRUCTION_SCHEMA = "manim-dandelin-construction-3d/v1"

ContextInput = GeometryContext | ResolvedGeometryContext | None


class DandelinConstructionError(ValueError):
    """One requested Dandelin construction cannot be certified."""


class DandelinConicFamily(str, Enum):
    """The three non-degenerate conic families used in the construction."""

    ELLIPSE = "ellipse"
    PARABOLA = "parabola"
    HYPERBOLA = "hyperbola"


class DandelinPlaneSide(str, Enum):
    """Orientation-free location of a sphere centre relative to the plane."""

    APEX = "apex"
    OPPOSITE = "opposite"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DandelinConstructionError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise DandelinConstructionError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinConstructionError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise DandelinConstructionError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _point3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinConstructionError(
            f"{label} must contain three finite numbers"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise DandelinConstructionError(
            f"{label} must contain three finite numbers"
        )
    return tuple(0.0 if item == 0.0 else float(item) for item in result)  # type: ignore[return-value]


def _coordinates2(
    value: Sequence[float],
    label: str,
) -> tuple[float, float]:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinConstructionError(
            f"{label} must contain two finite numbers"
        ) from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise DandelinConstructionError(
            f"{label} must contain two finite numbers"
        )
    return tuple(0.0 if item == 0.0 else float(item) for item in result)  # type: ignore[return-value]


def _normalize_canonical_numbers(value: object) -> object:
    """Normalize signed zero recursively before deterministic JSON encoding."""

    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [_normalize_canonical_numbers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_canonical_numbers(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _normalize_canonical_numbers(item)
            for key, item in value.items()
        }
    return value


def _resolved_context(
    cone: ConeSpec,
    plane: SectionPlane,
    context: ContextInput,
) -> ResolvedGeometryContext:
    if isinstance(context, ResolvedGeometryContext):
        return resolve_geometry_context(context)
    return resolve_geometry_context(
        context,
        positions=(*cone.characteristic_points, plane.point),
    )


def _section_frame(
    construction_id: str,
    plane: SectionPlane,
) -> PlanarFrame3D:
    return PlanarFrame3D(
        f"{construction_id}:section-plane",
        plane.point,
        plane.normal,
        plane.u_axis,
    )


def _canonical_direction2(value: Sequence[float]) -> tuple[float, float]:
    direction = np.asarray(_coordinates2(value, "directrix direction"), dtype=float)
    length = float(np.linalg.norm(direction))
    if not isfinite(length) or length <= 0.0:
        raise DandelinConstructionError("directrix direction must be non-zero")
    direction /= length
    index = int(np.argmax(np.abs(direction)))
    if direction[index] < 0.0:
        direction = -direction
    return _coordinates2(direction, "canonical directrix direction")


@dataclass(frozen=True, slots=True)
class DandelinDirectrix3D:
    """One infinite directrix carried by the authored section-plane frame."""

    directrix_id: str
    sphere_id: str
    plane_id: str
    point: PlanarPoint3D
    direction_coordinates: tuple[float, float]
    schema: str = DANDELIN_DIRECTRIX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DANDELIN_DIRECTRIX_SCHEMA:
            raise DandelinConstructionError("invalid Dandelin-directrix schema")
        object.__setattr__(
            self,
            "directrix_id",
            _identity(self.directrix_id, "directrix_id"),
        )
        object.__setattr__(self, "sphere_id", _identity(self.sphere_id, "sphere_id"))
        object.__setattr__(self, "plane_id", _identity(self.plane_id, "plane_id"))
        if not isinstance(self.point, PlanarPoint3D):
            raise TypeError("directrix point must be a PlanarPoint3D")
        object.__setattr__(
            self,
            "direction_coordinates",
            _canonical_direction2(self.direction_coordinates),
        )

    @property
    def frame(self) -> PlanarFrame3D:
        return self.point.frame

    @property
    def world_point(self) -> tuple[float, float, float]:
        return self.point.world_point

    @property
    def world_direction(self) -> tuple[float, float, float]:
        u_axis = np.asarray(self.frame.u_axis, dtype=float)
        v_axis = np.asarray(self.frame.v_axis, dtype=float)
        direction = (
            self.direction_coordinates[0] * u_axis
            + self.direction_coordinates[1] * v_axis
        )
        return _point3(direction, "directrix world direction")

    def clipped_segment(
        self,
        patch: PlaneDisplayPatchSpec,
        *,
        curve_id: str | None = None,
        context: ContextInput = None,
    ) -> SegmentCurve:
        """Clip this infinite line to one finite section display patch."""

        if not isinstance(patch, PlaneDisplayPatchSpec):
            raise TypeError("patch must be a PlaneDisplayPatchSpec")
        if patch.plane_id != self.plane_id:
            raise DandelinConstructionError(
                "directrix display patch belongs to a different section plane"
            )
        if isinstance(context, ResolvedGeometryContext):
            resolved = resolve_geometry_context(context)
        else:
            center = self.frame.point_from_coordinates(patch.center_coordinates)
            corners = tuple(
                self.frame.point_from_coordinates((u, v))
                for u in (
                    patch.center_coordinates[0] - patch.half_width,
                    patch.center_coordinates[0] + patch.half_width,
                )
                for v in (
                    patch.center_coordinates[1] - patch.half_height,
                    patch.center_coordinates[1] + patch.half_height,
                )
            )
            resolved = resolve_geometry_context(
                context,
                positions=(center, *corners),
            )
        epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
        origin = np.asarray(self.point.coordinates, dtype=float)
        direction = np.asarray(self.direction_coordinates, dtype=float)
        lower = np.asarray(
            (
                patch.center_coordinates[0] - patch.half_width,
                patch.center_coordinates[1] - patch.half_height,
            ),
            dtype=float,
        )
        upper = np.asarray(
            (
                patch.center_coordinates[0] + patch.half_width,
                patch.center_coordinates[1] + patch.half_height,
            ),
            dtype=float,
        )
        parameter_min = -float("inf")
        parameter_max = float("inf")
        for axis in range(2):
            component = float(direction[axis])
            if abs(component) <= np.finfo(float).eps * 64.0:
                if origin[axis] < lower[axis] - epsilon or origin[axis] > upper[axis] + epsilon:
                    raise DandelinConstructionError(
                        "directrix does not intersect the finite display patch"
                    )
                continue
            first = float((lower[axis] - origin[axis]) / component)
            second = float((upper[axis] - origin[axis]) / component)
            parameter_min = max(parameter_min, min(first, second))
            parameter_max = min(parameter_max, max(first, second))
        if (
            not isfinite(parameter_min)
            or not isfinite(parameter_max)
            or parameter_max - parameter_min <= epsilon
        ):
            raise DandelinConstructionError(
                "directrix has no certifiable finite segment in the display patch"
            )
        start_coordinates = origin + parameter_min * direction
        end_coordinates = origin + parameter_max * direction
        start = self.frame.point_from_coordinates(start_coordinates)
        end = self.frame.point_from_coordinates(end_coordinates)
        try:
            return SegmentCurve(
                self.directrix_id if curve_id is None else _identity(curve_id, "curve_id"),
                _point3(start, "directrix segment start"),
                _point3(end, "directrix segment end"),
            )
        except CurveContractError as exc:
            raise DandelinConstructionError(
                "directrix display segment cannot be represented"
            ) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "directrixId": self.directrix_id,
            "sphereId": self.sphere_id,
            "planeId": self.plane_id,
            "point": self.point.to_dict(),
            "directionCoordinates": list(self.direction_coordinates),
            "worldPoint": list(self.world_point),
            "worldDirection": list(self.world_direction),
        }


@dataclass(frozen=True, slots=True)
class DandelinSphere3D:
    """One sphere and all certified contact geometry attached to it."""

    sphere: SphereSpec
    nappe_sign: int
    plane_side: DandelinPlaneSide
    focus_id: str
    focus: PlanarPoint3D
    cone_contact_circle: Circle3DSpec
    directrix: DandelinDirectrix3D | None
    axial_center: float
    axial_extent: tuple[float, float]
    schema: str = DANDELIN_SPHERE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DANDELIN_SPHERE_SCHEMA:
            raise DandelinConstructionError("invalid Dandelin-sphere schema")
        if not isinstance(self.sphere, SphereSpec):
            raise TypeError("sphere must be a SphereSpec")
        if type(self.nappe_sign) is not int or self.nappe_sign not in {-1, 1}:
            raise DandelinConstructionError("nappe_sign must be -1 or 1")
        if not isinstance(self.plane_side, DandelinPlaneSide):
            raise TypeError("plane_side must be a DandelinPlaneSide")
        object.__setattr__(self, "focus_id", _identity(self.focus_id, "focus_id"))
        if not isinstance(self.focus, PlanarPoint3D):
            raise TypeError("focus must be a PlanarPoint3D")
        if not isinstance(self.cone_contact_circle, Circle3DSpec):
            raise TypeError("cone_contact_circle must be a Circle3DSpec")
        if self.directrix is not None:
            if not isinstance(self.directrix, DandelinDirectrix3D):
                raise TypeError("directrix must be a DandelinDirectrix3D or None")
            if self.directrix.sphere_id != self.sphere.surface_id:
                raise DandelinConstructionError(
                    "directrix sphere_id does not match its Dandelin sphere"
                )
        axial_center = _finite(self.axial_center, "axial_center")
        lower, upper = (
            _finite(self.axial_extent[0], "axial_extent lower"),
            _finite(self.axial_extent[1], "axial_extent upper"),
        )
        if lower >= upper:
            raise DandelinConstructionError("axial_extent must be increasing")
        if not lower < axial_center < upper:
            raise DandelinConstructionError(
                "axial_center must lie strictly inside axial_extent"
            )
        object.__setattr__(self, "axial_center", axial_center)
        object.__setattr__(self, "axial_extent", (lower, upper))

    @property
    def sphere_id(self) -> str:
        return self.sphere.surface_id

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sphere": {
                "surfaceId": self.sphere.surface_id,
                "center": list(self.sphere.center),
                "radius": self.sphere.radius,
            },
            "nappeSign": self.nappe_sign,
            "planeSide": self.plane_side.value,
            "focusId": self.focus_id,
            "focus": self.focus.to_dict(),
            "coneContactCircle": self.cone_contact_circle.to_dict(),
            "directrix": None if self.directrix is None else self.directrix.to_dict(),
            "axialCenter": self.axial_center,
            "axialExtent": list(self.axial_extent),
        }


@dataclass(frozen=True, slots=True)
class DandelinConstruction3D:
    """A complete finite Dandelin construction for one non-degenerate section."""

    construction_id: str
    cone: ConeSpec
    plane: SectionPlane
    section_frame: PlanarFrame3D
    supporting_kind: ConicKind
    family: DandelinConicFamily
    eccentricity: float
    spheres: tuple[DandelinSphere3D, ...]
    certification_context: ResolvedGeometryContext
    coefficient_tolerance: float | None = None
    schema: str = DANDELIN_CONSTRUCTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DANDELIN_CONSTRUCTION_SCHEMA:
            raise DandelinConstructionError("invalid Dandelin-construction schema")
        object.__setattr__(
            self,
            "construction_id",
            _identity(self.construction_id, "construction_id"),
        )
        if not isinstance(self.cone, ConeSpec):
            raise TypeError("cone must be a ConeSpec")
        if not isinstance(self.plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")
        if not isinstance(self.section_frame, PlanarFrame3D):
            raise TypeError("section_frame must be a PlanarFrame3D")
        if not isinstance(self.supporting_kind, ConicKind):
            raise TypeError("supporting_kind must be a ConicKind")
        if not isinstance(self.family, DandelinConicFamily):
            raise TypeError("family must be a DandelinConicFamily")
        if not isinstance(self.certification_context, ResolvedGeometryContext):
            raise TypeError(
                "certification_context must be a ResolvedGeometryContext"
            )
        coefficient_tolerance = self.coefficient_tolerance
        if coefficient_tolerance is not None:
            coefficient_tolerance = _finite(
                coefficient_tolerance,
                "coefficient_tolerance",
            )
            if coefficient_tolerance <= 0.0:
                raise DandelinConstructionError(
                    "coefficient_tolerance must be finite and positive"
                )
        eccentricity = _finite(self.eccentricity, "eccentricity")
        if eccentricity < 0.0:
            raise DandelinConstructionError("eccentricity must be non-negative")
        spheres = tuple(self.spheres)
        if not spheres or not all(isinstance(item, DandelinSphere3D) for item in spheres):
            raise DandelinConstructionError(
                "a Dandelin construction requires certified sphere records"
            )
        identities = tuple(item.sphere_id for item in spheres)
        if len(set(identities)) != len(identities):
            raise DandelinConstructionError("Dandelin sphere identities must be unique")
        if tuple(sorted(spheres, key=lambda item: item.sphere_id)) != spheres:
            raise DandelinConstructionError(
                "Dandelin spheres must use canonical identity order"
            )
        expected_count = {
            DandelinConicFamily.ELLIPSE: 2,
            DandelinConicFamily.PARABOLA: 1,
            DandelinConicFamily.HYPERBOLA: 2,
        }[self.family]
        if len(spheres) != expected_count:
            raise DandelinConstructionError(
                f"{self.family.value} requires {expected_count} finite Dandelin sphere(s)"
            )
        if any(item.focus.frame != self.section_frame for item in spheres):
            raise DandelinConstructionError(
                "every Dandelin focus must use the construction section frame"
            )
        object.__setattr__(self, "eccentricity", eccentricity)
        object.__setattr__(self, "spheres", spheres)
        object.__setattr__(self, "coefficient_tolerance", coefficient_tolerance)
        _validate_construction_integrity(self)

    @property
    def sphere_surfaces(self) -> tuple[SphereSpec, ...]:
        return tuple(item.sphere for item in self.spheres)

    @property
    def focus_points(self) -> tuple[PlanarPoint3D, ...]:
        return tuple(item.focus for item in self.spheres)

    @property
    def cone_contact_circles(self) -> tuple[Circle3DSpec, ...]:
        return tuple(item.cone_contact_circle for item in self.spheres)

    @property
    def directrices(self) -> tuple[DandelinDirectrix3D, ...]:
        return tuple(
            item.directrix for item in self.spheres if item.directrix is not None
        )

    def directrix_segments(
        self,
        patch: PlaneDisplayPatchSpec,
        *,
        context: ContextInput = None,
    ) -> tuple[SegmentCurve, ...]:
        return tuple(
            item.clipped_segment(patch, context=context)
            for item in self.directrices
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "constructionId": self.construction_id,
            "cone": {
                "surfaceId": self.cone.surface_id,
                "apex": list(self.cone.apex),
                "axis": list(self.cone.axis),
                "radialAxis": list(self.cone.radial_axis or ()),
                "halfAngle": self.cone.half_angle,
                "axialRange": list(self.cone.axial_range),
                "model": self.cone.model.value,
                "componentParentId": self.cone.component_parent_id,
            },
            "plane": {
                "planeId": self.plane.plane_id,
                "point": list(self.plane.point),
                "normal": list(self.plane.normal),
                "uAxis": list(self.plane.u_axis or ()),
            },
            "sectionFrame": self.section_frame.to_dict(),
            "supportingKind": self.supporting_kind.value,
            "family": self.family.value,
            "eccentricity": self.eccentricity,
            "spheres": [item.to_dict() for item in self.spheres],
            "certificationContext": self.certification_context.to_dict(),
            "coefficientTolerance": self.coefficient_tolerance,
            "finiteFitCertified": True,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            _normalize_canonical_numbers(self.to_dict()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _family(kind: ConicKind) -> DandelinConicFamily:
    if kind in {ConicKind.CIRCLE, ConicKind.ELLIPSE}:
        return DandelinConicFamily.ELLIPSE
    if kind is ConicKind.PARABOLA:
        return DandelinConicFamily.PARABOLA
    if kind is ConicKind.HYPERBOLA:
        return DandelinConicFamily.HYPERBOLA
    raise DandelinConstructionError(
        f"Dandelin spheres require a non-degenerate conic, got {kind.value!r}"
    )


def _build_directrix(
    directrix_id: str,
    sphere_id: str,
    plane: SectionPlane,
    section_frame: PlanarFrame3D,
    cone_axis: np.ndarray,
    contact_center: np.ndarray,
    *,
    angular_epsilon: float,
) -> DandelinDirectrix3D:
    u_axis = np.asarray(section_frame.u_axis, dtype=float)
    v_axis = np.asarray(section_frame.v_axis, dtype=float)
    coefficients = np.asarray(
        (float(np.dot(cone_axis, u_axis)), float(np.dot(cone_axis, v_axis))),
        dtype=float,
    )
    coefficient_norm = float(np.linalg.norm(coefficients))
    if coefficient_norm <= angular_epsilon:
        raise DandelinConstructionError(
            "the contact-circle plane and section plane have no certifiable finite directrix"
        )
    constant = float(
        np.dot(
            cone_axis,
            np.asarray(section_frame.point, dtype=float) - contact_center,
        )
    )
    point_coordinates = -constant * coefficients / (coefficient_norm * coefficient_norm)
    direction = np.asarray((-coefficients[1], coefficients[0]), dtype=float)
    direction /= coefficient_norm
    point = section_frame.certified_point(point_coordinates)
    result = DandelinDirectrix3D(
        directrix_id,
        sphere_id,
        plane.plane_id,
        point,
        _coordinates2(direction, "directrix direction"),
    )
    residual = abs(
        float(
            np.dot(
                cone_axis,
                np.asarray(result.world_point, dtype=float) - contact_center,
            )
        )
    )
    if residual > 64.0 * np.finfo(float).eps * max(
        1.0,
        float(np.linalg.norm(contact_center)),
        abs(constant),
    ):
        raise DandelinConstructionError(
            "derived directrix does not lie in the cone-contact plane"
        )
    return result


def _validate_contact_geometry(
    cone: ConeSpec,
    plane: SectionPlane,
    record: DandelinSphere3D,
    *,
    boundary_epsilon: float,
) -> None:
    tolerance = 32.0 * boundary_epsilon
    center = np.asarray(record.sphere.center, dtype=float)
    focus = np.asarray(record.focus.world_point, dtype=float)
    if abs(plane.signed_distance(focus)) > tolerance:
        raise DandelinConstructionError("derived focus is not on the section plane")
    if abs(float(np.linalg.norm(focus - center)) - record.sphere.radius) > tolerance:
        raise DandelinConstructionError("derived focus is not on its Dandelin sphere")
    analytic_circle = record.cone_contact_circle.lower_to_analytic_curve()
    for parameter in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        point = np.asarray(analytic_circle.point(parameter), dtype=float)
        sphere_residual = abs(
            float(np.linalg.norm(point - center)) - record.sphere.radius
        )
        local = cone.frame.to_local_point(point)
        cone_residual = abs(
            float(np.linalg.norm(local[:2])) - abs(float(local[2])) * cone.slope
        )
        if max(sphere_residual, cone_residual) > tolerance:
            raise DandelinConstructionError(
                "derived contact circle is not jointly tangent to sphere and cone"
            )
        sphere_normal = point - center
        cone_normal = cone.support_quadric.gradient(point)
        sphere_length = float(np.linalg.norm(sphere_normal))
        cone_length = float(np.linalg.norm(cone_normal))
        if sphere_length <= 0.0 or cone_length <= 0.0:
            raise DandelinConstructionError(
                "contact-circle tangency has a zero surface normal"
            )
        normal_cross = float(
            np.linalg.norm(
                np.cross(
                    sphere_normal / sphere_length,
                    cone_normal / cone_length,
                )
            )
        )
        if normal_cross > 64.0 * max(
            np.finfo(float).eps,
            boundary_epsilon / max(record.sphere.radius, boundary_epsilon),
        ):
            raise DandelinConstructionError(
                "sphere and cone normals are not tangent along the contact circle"
            )


def _derive_sphere_records(
    construction_id: str,
    cone: ConeSpec,
    plane: SectionPlane,
    section_frame: PlanarFrame3D,
    supporting_kind: ConicKind,
    family: DandelinConicFamily,
    resolved: ResolvedGeometryContext,
) -> tuple[
    tuple[DandelinSphere3D, ...],
    tuple[tuple[float, float], ...],
]:
    """Derive the one canonical record set shared by solve and validation."""

    boundary_epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    angular_epsilon = resolved.epsilon(GeometryQuantity.ANGULAR)
    apex = np.asarray(cone.apex, dtype=float)
    axis = np.asarray(cone.axis, dtype=float)
    normal = np.asarray(plane.normal, dtype=float)
    plane_at_apex = plane.signed_distance(cone.apex)
    axis_normal_dot = float(np.dot(axis, normal))
    sine = sin(cone.half_angle)
    cosine = cos(cone.half_angle)
    lower, upper = cone.axial_range
    nappe_signs = tuple(
        sign
        for sign, available in (
            (-1, lower < -boundary_epsilon),
            (1, upper > boundary_epsilon),
        )
        if available
    )
    records: list[DandelinSphere3D] = []
    outside_extents: list[tuple[float, float]] = []
    for nappe_sign in nappe_signs:
        for oriented_plane_side in (-1, 1):
            denominator = (
                nappe_sign * axis_normal_dot - oriented_plane_side * sine
            )
            if abs(denominator) <= angular_epsilon:
                # At the exact parabolic angle the second classical sphere is
                # at infinity. It is not represented by a finite placeholder.
                if family is not DandelinConicFamily.PARABOLA:
                    raise DandelinConstructionError(
                        "a Dandelin sphere denominator is numerically singular "
                        "outside an exact parabolic section"
                    )
                continue
            distance_from_apex = -plane_at_apex / denominator
            if not isfinite(distance_from_apex) or distance_from_apex <= 0.0:
                continue
            axial_center = nappe_sign * distance_from_apex
            radius = distance_from_apex * sine
            axial_extent = (axial_center - radius, axial_center + radius)
            if (
                radius <= boundary_epsilon
                or axial_extent[0] <= lower + boundary_epsilon
                or axial_extent[1] >= upper - boundary_epsilon
            ):
                outside_extents.append(axial_extent)
                continue
            center = apex + axial_center * axis
            center_plane_distance = (
                plane_at_apex + axial_center * axis_normal_dot
            )
            side = (
                DandelinPlaneSide.APEX
                if plane_at_apex * center_plane_distance > 0.0
                else DandelinPlaneSide.OPPOSITE
            )
            nappe_label = "positive" if nappe_sign > 0 else "negative"
            sphere_id = (
                f"{construction_id}:sphere:nappe:{nappe_label}:"
                f"side:{side.value}"
            )
            sphere = SphereSpec(
                sphere_id,
                _point3(center, "Dandelin sphere center"),
                radius,
            )
            raw_focus = center - center_plane_distance * normal
            focus = section_frame.certified_point(
                plane.coordinates_in_plane(raw_focus)
            )
            focus_id = f"{sphere_id}:focus"

            contact_axial = axial_center * cosine * cosine
            contact_center = apex + contact_axial * axis
            contact_radius = distance_from_apex * sine * cosine
            if contact_radius <= boundary_epsilon:
                outside_extents.append(axial_extent)
                continue
            contact_frame = PlanarFrame3D(
                f"{sphere_id}:cone-contact-plane",
                _point3(contact_center, "cone-contact center"),
                cone.axis,
                cone.radial_axis,
            )
            contact_circle = Circle3DSpec.from_plane_coordinates(
                f"{sphere_id}:cone-contact-circle",
                contact_frame,
                (0.0, 0.0),
                contact_radius,
            )
            directrix = None
            if supporting_kind is not ConicKind.CIRCLE:
                directrix = _build_directrix(
                    f"{sphere_id}:directrix",
                    sphere_id,
                    plane,
                    section_frame,
                    axis,
                    contact_center,
                    angular_epsilon=angular_epsilon,
                )
            record = DandelinSphere3D(
                sphere=sphere,
                nappe_sign=nappe_sign,
                plane_side=side,
                focus_id=focus_id,
                focus=focus,
                cone_contact_circle=contact_circle,
                directrix=directrix,
                axial_center=axial_center,
                axial_extent=axial_extent,
            )
            _validate_contact_geometry(
                cone,
                plane,
                record,
                boundary_epsilon=boundary_epsilon,
            )
            records.append(record)
    return (
        tuple(sorted(records, key=lambda item: item.sphere_id)),
        tuple(sorted(outside_extents)),
    )


def _validate_construction_integrity(
    construction: DandelinConstruction3D,
) -> None:
    """Re-certify authoritative geometry before advertising finite-fit evidence.

    The public construction records are frozen dataclasses, so callers may use
    ``dataclasses.replace``. Revalidation here prevents a replaced derived
    record from retaining ``finiteFitCertified=true`` after it drifts away
    from the cone/plane geometry that produced it. Authored identities and the
    explicit numerical context remain inputs and are recorded in canonical
    evidence rather than treated as hidden provenance.
    """

    resolved = resolve_geometry_context(construction.certification_context)
    boundary_epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    angular_epsilon = resolved.epsilon(GeometryQuantity.ANGULAR)
    expected_frame = _section_frame(
        construction.construction_id,
        construction.plane,
    )
    if construction.section_frame != expected_frame:
        raise DandelinConstructionError(
            "section_frame is not the canonical frame of the authored plane"
        )

    try:
        trace = compute_quadric_section(
            f"{construction.construction_id}:section",
            construction.cone,
            construction.plane,
            context=resolved,
            coefficient_tolerance=construction.coefficient_tolerance,
        )
        boundary = compute_quadric_section_boundary(
            f"{construction.construction_id}:section",
            construction.cone,
            construction.plane,
            context=resolved,
            coefficient_tolerance=construction.coefficient_tolerance,
        )
    except QuadricSectionError as exc:
        raise DandelinConstructionError(
            f"construction fields no longer define a certified finite section: {exc}"
        ) from exc
    if trace.supporting_kind is not construction.supporting_kind:
        raise DandelinConstructionError(
            "supporting_kind does not match the authored cone/plane section"
        )
    if _family(trace.supporting_kind) is not construction.family:
        raise DandelinConstructionError(
            "family does not match the certified supporting conic"
        )
    if trace.finite_topology in {
        FiniteSectionTopology.EMPTY,
        FiniteSectionTopology.POINT,
        FiniteSectionTopology.MULTIPLE_POINTS,
        FiniteSectionTopology.CURVES_AND_POINTS,
    }:
        raise DandelinConstructionError(
            "the finite cone section is empty or degenerate"
        )
    if (
        construction.family is DandelinConicFamily.ELLIPSE
        and trace.finite_topology is not FiniteSectionTopology.CLOSED_CURVE
    ):
        raise DandelinConstructionError(
            "a finite Dandelin ellipse must be a complete closed lateral section"
        )
    if boundary.cap_chords:
        raise DandelinConstructionError(
            "the certified construction cannot contain a real cone cap chord"
        )

    cone = construction.cone
    plane = construction.plane
    if cone.model is ConeModel.ANALYTIC_DOUBLE:
        raise DandelinConstructionError(
            "ANALYTIC_DOUBLE cannot carry finite-fit Dandelin evidence"
        )
    if construction.family in {
        DandelinConicFamily.ELLIPSE,
        DandelinConicFamily.PARABOLA,
    } and cone.nappe_count != 1:
        raise DandelinConstructionError(
            "ellipse/parabola Dandelin evidence requires one authored nappe"
        )
    if (
        construction.family is DandelinConicFamily.HYPERBOLA
        and cone.nappe_count != 2
    ):
        raise DandelinConstructionError(
            "hyperbola Dandelin evidence requires both authored nappes"
        )

    axis = np.asarray(cone.axis, dtype=float)
    apex = np.asarray(cone.apex, dtype=float)
    normal = np.asarray(plane.normal, dtype=float)
    axis_normal_dot = float(np.dot(axis, normal))
    sine = sin(cone.half_angle)
    cosine = cos(cone.half_angle)
    transverse = sqrt(max(0.0, 1.0 - axis_normal_dot * axis_normal_dot))
    expected_eccentricity = (
        0.0
        if trace.supporting_kind is ConicKind.CIRCLE
        else transverse / cosine
    )
    if construction.eccentricity != expected_eccentricity:
        raise DandelinConstructionError(
            "eccentricity is not the canonical authored cone/plane value"
        )

    expected_nappes = {
        DandelinConicFamily.ELLIPSE: {construction.spheres[0].nappe_sign},
        DandelinConicFamily.PARABOLA: {construction.spheres[0].nappe_sign},
        DandelinConicFamily.HYPERBOLA: {-1, 1},
    }[construction.family]
    if {item.nappe_sign for item in construction.spheres} != expected_nappes:
        raise DandelinConstructionError(
            "Dandelin sphere nappe distribution does not match the conic family"
        )

    tolerance = 32.0 * boundary_epsilon
    lower, upper = cone.axial_range
    plane_at_apex = plane.signed_distance(cone.apex)
    for record in construction.spheres:
        center = np.asarray(record.sphere.center, dtype=float)
        offset = center - apex
        axial_center = float(np.dot(offset, axis))
        radial_offset = offset - axial_center * axis
        if float(np.linalg.norm(radial_offset)) > tolerance:
            raise DandelinConstructionError(
                "a Dandelin sphere centre is no longer on the cone axis"
            )
        if abs(record.axial_center - axial_center) > tolerance:
            raise DandelinConstructionError(
                "axial_center does not match the Dandelin sphere centre"
            )
        expected_nappe = 1 if axial_center > 0.0 else -1
        if abs(axial_center) <= boundary_epsilon or record.nappe_sign != expected_nappe:
            raise DandelinConstructionError(
                "nappe_sign does not match the Dandelin sphere centre"
            )
        expected_radius = abs(axial_center) * sine
        if abs(record.sphere.radius - expected_radius) > tolerance:
            raise DandelinConstructionError(
                "Dandelin sphere radius does not match cone tangency"
            )
        expected_extent = (
            axial_center - expected_radius,
            axial_center + expected_radius,
        )
        if any(
            abs(actual - expected) > tolerance
            for actual, expected in zip(record.axial_extent, expected_extent)
        ):
            raise DandelinConstructionError(
                "axial_extent does not match the sphere centre and radius"
            )
        if (
            expected_extent[0] <= lower + boundary_epsilon
            or expected_extent[1] >= upper - boundary_epsilon
        ):
            raise DandelinConstructionError(
                "a certified Dandelin sphere does not fit strictly inside the cone"
            )

        side = (
            DandelinPlaneSide.APEX
            if plane_at_apex * plane.signed_distance(center) > 0.0
            else DandelinPlaneSide.OPPOSITE
        )
        if record.plane_side is not side:
            raise DandelinConstructionError(
                "plane_side does not match the sphere centre and section plane"
            )
        nappe_label = "positive" if record.nappe_sign > 0 else "negative"
        expected_sphere_id = (
            f"{construction.construction_id}:sphere:nappe:{nappe_label}:"
            f"side:{side.value}"
        )
        if record.sphere_id != expected_sphere_id:
            raise DandelinConstructionError(
                "Dandelin sphere identity is not canonical"
            )
        if record.focus_id != f"{expected_sphere_id}:focus":
            raise DandelinConstructionError("Dandelin focus identity is not canonical")

        raw_focus = center - plane.signed_distance(center) * normal
        expected_focus = construction.section_frame.certified_point(
            plane.coordinates_in_plane(raw_focus)
        )
        if (
            record.focus.frame != expected_focus.frame
            or float(
                np.linalg.norm(
                    np.asarray(record.focus.world_point, dtype=float)
                    - np.asarray(expected_focus.world_point, dtype=float)
                )
            )
            > tolerance
        ):
            raise DandelinConstructionError(
                "focus does not match sphere-plane tangency"
            )

        contact_center = apex + axial_center * cosine * cosine * axis
        contact_radius = abs(axial_center) * sine * cosine
        contact_frame = PlanarFrame3D(
            f"{expected_sphere_id}:cone-contact-plane",
            _point3(contact_center, "cone-contact center"),
            cone.axis,
            cone.radial_axis,
        )
        expected_circle = Circle3DSpec.from_plane_coordinates(
            f"{expected_sphere_id}:cone-contact-circle",
            contact_frame,
            (0.0, 0.0),
            contact_radius,
        )
        actual_circle = record.cone_contact_circle
        actual_frame = actual_circle.frame
        expected_contact_frame = expected_circle.frame
        if (
            actual_circle.curve_id != expected_circle.curve_id
            or actual_circle.schema != expected_circle.schema
            or actual_circle.domain != expected_circle.domain
            or actual_frame.frame_id != expected_contact_frame.frame_id
            or actual_frame.schema != expected_contact_frame.schema
            or float(
                np.linalg.norm(
                    np.asarray(actual_circle.center, dtype=float)
                    - np.asarray(expected_circle.center, dtype=float)
                )
            )
            > tolerance
            or abs(actual_circle.radius - expected_circle.radius) > tolerance
            or float(
                np.linalg.norm(
                    np.asarray(actual_circle.center_coordinates, dtype=float)
                    - np.asarray(expected_circle.center_coordinates, dtype=float)
                )
            )
            > tolerance
            or any(
                float(
                    np.linalg.norm(
                        np.asarray(actual, dtype=float)
                        - np.asarray(expected, dtype=float)
                    )
                )
                > 64.0 * max(np.finfo(float).eps, angular_epsilon)
                for actual, expected in (
                    (actual_frame.normal, expected_contact_frame.normal),
                    (actual_frame.u_axis, expected_contact_frame.u_axis),
                    (actual_frame.v_axis, expected_contact_frame.v_axis),
                )
            )
        ):
            raise DandelinConstructionError(
                "cone_contact_circle is not the canonical tangency circle"
            )

        if trace.supporting_kind is ConicKind.CIRCLE:
            if record.directrix is not None:
                raise DandelinConstructionError(
                    "a circular section must not claim a finite directrix"
                )
        else:
            if record.directrix is None:
                raise DandelinConstructionError(
                    "a non-circular section requires its certified directrix"
                )
            expected_directrix = _build_directrix(
                f"{expected_sphere_id}:directrix",
                expected_sphere_id,
                plane,
                construction.section_frame,
                axis,
                contact_center,
                angular_epsilon=angular_epsilon,
            )
            actual_directrix = record.directrix
            if (
                actual_directrix.directrix_id
                != expected_directrix.directrix_id
                or actual_directrix.sphere_id != expected_directrix.sphere_id
                or actual_directrix.plane_id != expected_directrix.plane_id
                or actual_directrix.schema != expected_directrix.schema
                or actual_directrix.frame != expected_directrix.frame
                or float(
                    np.linalg.norm(
                        np.asarray(actual_directrix.world_point, dtype=float)
                        - np.asarray(expected_directrix.world_point, dtype=float)
                    )
                )
                > tolerance
                or float(
                    np.linalg.norm(
                        np.asarray(actual_directrix.world_direction, dtype=float)
                        - np.asarray(expected_directrix.world_direction, dtype=float)
                    )
                )
                > 64.0 * max(np.finfo(float).eps, angular_epsilon)
            ):
                raise DandelinConstructionError(
                    "directrix is not canonical for the contact-circle plane"
                )
        _validate_contact_geometry(
            cone,
            plane,
            record,
            boundary_epsilon=boundary_epsilon,
        )

    expected_records, _outside_extents = _derive_sphere_records(
        construction.construction_id,
        cone,
        plane,
        construction.section_frame,
        construction.supporting_kind,
        construction.family,
        resolved,
    )
    if construction.spheres != expected_records:
        raise DandelinConstructionError(
            "Dandelin sphere records are not the canonical values derived "
            "from the authored cone and section plane"
        )


def compute_dandelin_construction(
    construction_id: str,
    cone: ConeSpec,
    plane: SectionPlane,
    *,
    context: ContextInput = None,
    coefficient_tolerance: float | None = None,
) -> DandelinConstruction3D:
    """Return the complete finite Dandelin construction for ``cone ∩ plane``.

    Ellipses (including circles) require two finite spheres on one nappe,
    parabolas require the one finite sphere left by the exact critical angle,
    and hyperbolas require one sphere on each nappe.  A single-nappe hyperbola
    is therefore rejected with guidance to author a finite ``OPEN_DOUBLE``
    cone.  ``ANALYTIC_DOUBLE`` remains an infinite support contract and is not
    accepted by this finite teaching construction.
    """

    identity = _identity(construction_id, "construction_id")
    if not isinstance(cone, ConeSpec):
        raise TypeError("cone must be a ConeSpec")
    if not isinstance(plane, SectionPlane):
        raise TypeError("plane must be a SectionPlane")
    if cone.model is ConeModel.ANALYTIC_DOUBLE:
        raise DandelinConstructionError(
            "ANALYTIC_DOUBLE is an infinite support contract; use OPEN_DOUBLE "
            "for a finite renderable Dandelin construction"
        )
    resolved = _resolved_context(cone, plane, context)
    boundary_epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    angular_epsilon = resolved.epsilon(GeometryQuantity.ANGULAR)
    try:
        trace = compute_quadric_section(
            f"{identity}:section",
            cone,
            plane,
            context=resolved,
            coefficient_tolerance=coefficient_tolerance,
        )
    except QuadricSectionError as exc:
        raise DandelinConstructionError(
            f"supporting conic could not be certified: {exc}"
        ) from exc
    family = _family(trace.supporting_kind)
    if family in {
        DandelinConicFamily.ELLIPSE,
        DandelinConicFamily.PARABOLA,
    } and cone.nappe_count != 1:
        raise DandelinConstructionError(
            f"{family.value} Dandelin v1 requires a single-nappe cone model"
        )
    if trace.finite_topology in {
        FiniteSectionTopology.EMPTY,
        FiniteSectionTopology.POINT,
        FiniteSectionTopology.MULTIPLE_POINTS,
        FiniteSectionTopology.CURVES_AND_POINTS,
    }:
        raise DandelinConstructionError(
            "the finite cone section is empty or degenerate"
        )
    if family is DandelinConicFamily.ELLIPSE and (
        trace.finite_topology is not FiniteSectionTopology.CLOSED_CURVE
    ):
        raise DandelinConstructionError(
            "a finite Dandelin ellipse must be a complete closed lateral section"
        )
    if family is DandelinConicFamily.HYPERBOLA and cone.nappe_count != 2:
        raise DandelinConstructionError(
            "a complete hyperbola construction requires an OPEN_DOUBLE cone "
            "so both finite Dandelin spheres are authored"
        )
    try:
        boundary = compute_quadric_section_boundary(
            f"{identity}:section",
            cone,
            plane,
            context=resolved,
            coefficient_tolerance=coefficient_tolerance,
        )
    except QuadricSectionError as exc:
        raise DandelinConstructionError(
            f"finite section boundary could not be certified: {exc}"
        ) from exc
    if boundary.cap_chords:
        raise DandelinConstructionError(
            "the finite section reaches a filled cone end cap; Dandelin v1 "
            "requires a pure lateral conic without a cap chord"
        )

    section_frame = _section_frame(identity, plane)
    axis = np.asarray(cone.axis, dtype=float)
    normal = np.asarray(plane.normal, dtype=float)
    plane_at_apex = plane.signed_distance(cone.apex)
    if abs(plane_at_apex) <= boundary_epsilon:
        raise DandelinConstructionError(
            "a plane through the cone apex gives a degenerate section"
        )
    axis_normal_dot = float(np.dot(axis, normal))
    sine = sin(cone.half_angle)
    cosine = cos(cone.half_angle)
    transverse = sqrt(max(0.0, 1.0 - axis_normal_dot * axis_normal_dot))
    eccentricity = transverse / cosine
    if trace.supporting_kind is ConicKind.CIRCLE:
        eccentricity = 0.0
    angular_family = (
        DandelinConicFamily.PARABOLA
        if abs(abs(axis_normal_dot) - sine) <= angular_epsilon
        else (
            DandelinConicFamily.ELLIPSE
            if abs(axis_normal_dot) > sine
            else DandelinConicFamily.HYPERBOLA
        )
    )
    if angular_family is not family:
        raise DandelinConstructionError(
            "cone/plane angle disagrees with the certified supporting conic family"
        )

    records, outside_extents = _derive_sphere_records(
        identity,
        cone,
        plane,
        section_frame,
        trace.supporting_kind,
        family,
        resolved,
    )

    expected_count = {
        DandelinConicFamily.ELLIPSE: 2,
        DandelinConicFamily.PARABOLA: 1,
        DandelinConicFamily.HYPERBOLA: 2,
    }[family]
    if len(records) != expected_count:
        if outside_extents:
            formatted = ", ".join(
                f"[{item[0]:.12g}, {item[1]:.12g}]"
                for item in sorted(outside_extents)
            )
            raise DandelinConstructionError(
                "the authored finite cone cannot contain every required "
                f"Dandelin sphere; rejected axial extents: {formatted}"
            )
        raise DandelinConstructionError(
            f"{family.value} requires {expected_count} finite Dandelin sphere(s), "
            f"but only {len(records)} could be certified"
        )
    if family is DandelinConicFamily.ELLIPSE and len(
        {item.nappe_sign for item in records}
    ) != 1:
        raise DandelinConstructionError(
            "an ellipse's two Dandelin spheres must belong to one cone nappe"
        )
    if family is DandelinConicFamily.HYPERBOLA and {
        item.nappe_sign for item in records
    } != {-1, 1}:
        raise DandelinConstructionError(
            "a hyperbola requires one Dandelin sphere on each cone nappe"
        )
    return DandelinConstruction3D(
        construction_id=identity,
        cone=cone,
        plane=plane,
        section_frame=section_frame,
        supporting_kind=trace.supporting_kind,
        family=family,
        eccentricity=eccentricity,
        spheres=records,
        certification_context=resolved,
        coefficient_tolerance=coefficient_tolerance,
    )


def canonical_dandelin_construction_json(
    construction: DandelinConstruction3D,
) -> str:
    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    return construction.canonical_json()


__all__ = [
    "DANDELIN_CONSTRUCTION_SCHEMA",
    "DANDELIN_DIRECTRIX_SCHEMA",
    "DANDELIN_SPHERE_SCHEMA",
    "DandelinConicFamily",
    "DandelinConstruction3D",
    "DandelinConstructionError",
    "DandelinDirectrix3D",
    "DandelinPlaneSide",
    "DandelinSphere3D",
    "canonical_dandelin_construction_json",
    "compute_dandelin_construction",
]
