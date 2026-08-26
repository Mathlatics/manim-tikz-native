"""Strict renderer-neutral contracts for common opaque quadrics.

Each finite teaching entity deliberately exposes two different concepts:

* :attr:`support_quadric` is the infinite implicit sphere, cylinder, or
  double-cone used by analytic intersection code;
* :meth:`contains`, :meth:`lateral_ray_hits`, and :meth:`ray_hits` apply the
  finite axial range and, where requested, the separate planar end caps.

Keeping these layers separate prevents a finite display patch or cap from
silently changing the mathematical quadric.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, pi, tan
from typing import Literal, Sequence

import numpy as np

from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from .algebra import (
    AffineFrame3D,
    CoincidentRayError,
    HomogeneousQuadric,
    QuadricAlgebraError,
)


class QuadricContractError(ValueError):
    """Raised when a persisted or authored quadric contract is ambiguous."""


ContextInput = GeometryContext | ResolvedGeometryContext | None


class ConeModel(str, Enum):
    """Finite cone models accepted by the public geometry contract.

    ``ANALYTIC_DOUBLE`` preserves the historical cross-apex ``ConeSpec`` used
    by conic-section solvers.  It is deliberately not a directly renderable
    solid.  Finite double-cone display geometry must use ``OPEN_DOUBLE`` and is
    then expanded into two stable single-nappe shell components.
    """

    CLOSED_SINGLE = "closed_single"
    OPEN_SINGLE = "open_single"
    OPEN_DOUBLE = "open_double"
    ANALYTIC_DOUBLE = "analytic_double"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuadricContractError(f"{label} must be a non-empty string")
    return value.strip()


def _point3(value: object, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise QuadricContractError(f"{label} must contain three finite numbers")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricContractError(
            f"{label} must contain three finite numbers"
        ) from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise QuadricContractError(f"{label} must contain three finite numbers")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def _point2(value: object, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise QuadricContractError(f"{label} must contain two finite numbers")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricContractError(
            f"{label} must contain two finite numbers"
        ) from exc
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise QuadricContractError(f"{label} must contain two finite numbers")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise QuadricContractError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricContractError(f"{label} must be finite and positive") from exc
    if not isfinite(result) or result <= 0.0:
        raise QuadricContractError(f"{label} must be finite and positive")
    return result


def _axial_range(value: object) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise QuadricContractError(
            "axial_range must contain two finite increasing values"
        )
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricContractError(
            "axial_range must contain two finite increasing values"
        ) from exc
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise QuadricContractError(
            "axial_range must contain two finite increasing values"
        )
    return lower, upper


def _frame(
    origin: Sequence[float],
    axis: Sequence[float],
    radial_axis: Sequence[float] | None,
    label: str,
) -> AffineFrame3D:
    try:
        return AffineFrame3D.from_axis(
            origin,
            axis,
            radial_axis=radial_axis,
        )
    except QuadricAlgebraError as exc:
        raise QuadricContractError(f"invalid {label} frame: {exc}") from exc


def _resolve(
    context: ContextInput,
    characteristic_points: Sequence[Sequence[float]],
) -> ResolvedGeometryContext:
    if isinstance(context, ResolvedGeometryContext):
        return resolve_geometry_context(context)
    return resolve_geometry_context(context, positions=characteristic_points)


def _normalized_ray(
    origin: Sequence[float],
    direction: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(_point3(origin, "ray origin"), dtype=float)
    vector = np.asarray(_point3(direction, "ray direction"), dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise QuadricContractError("ray direction must be non-zero")
    return point, vector / length


def _normal(
    value: np.ndarray,
    *,
    epsilon: float = 0.0,
) -> tuple[float, float, float] | None:
    length = float(np.linalg.norm(value))
    if not isfinite(length) or length <= epsilon:
        return None
    return tuple(float(item) for item in value / length)  # type: ignore[return-value]


def _local_quadric(
    diagonal: tuple[float, float, float, float],
    frame: AffineFrame3D,
) -> HomogeneousQuadric:
    matrix = np.diag(np.asarray(diagonal, dtype=float))
    return HomogeneousQuadric.from_local_matrix(
        matrix,
        frame.local_to_world_matrix,
    )


@dataclass(frozen=True, slots=True)
class QuadricRayHit:
    """One finite-entity ray hit, measured along a unit ray direction."""

    parameter: float
    point: tuple[float, float, float]
    normal: tuple[float, float, float] | None
    surface_id: str
    role: str
    tangential: bool = False

    def __post_init__(self) -> None:
        parameter = float(self.parameter)
        if not isfinite(parameter):
            raise QuadricContractError("ray-hit parameter must be finite")
        point = _point3(self.point, "ray-hit point")
        normal = None
        if self.normal is not None:
            value = np.asarray(_point3(self.normal, "ray-hit normal"), dtype=float)
            length = float(np.linalg.norm(value))
            if length <= 0.0:
                raise QuadricContractError("ray-hit normal must be non-zero")
            normal = tuple(float(item) for item in value / length)
        surface_id = _identity(self.surface_id, "ray-hit surface_id")
        role = _identity(self.role, "ray-hit role")
        if not isinstance(self.tangential, bool):
            raise QuadricContractError("ray-hit tangential must be boolean")
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class PlanarCapSpec:
    """A finite circular cap kept separate from an infinite support quadric."""

    cap_id: str
    parent_surface_id: str
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float
    radial_axis: tuple[float, float, float] | None = None
    role: Literal["cap_min", "cap_max"] = "cap_max"

    def __post_init__(self) -> None:
        cap_id = _identity(self.cap_id, "cap_id")
        parent_id = _identity(self.parent_surface_id, "parent_surface_id")
        center = _point3(self.center, "cap center")
        normal = _point3(self.normal, "cap normal")
        radial = (
            None
            if self.radial_axis is None
            else _point3(self.radial_axis, "cap radial_axis")
        )
        if self.role not in {"cap_min", "cap_max"}:
            raise QuadricContractError("cap role must be 'cap_min' or 'cap_max'")
        frame = _frame(center, normal, radial, "cap")
        object.__setattr__(self, "cap_id", cap_id)
        object.__setattr__(self, "parent_surface_id", parent_id)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal", frame.z_axis)
        object.__setattr__(self, "radial_axis", frame.x_axis)
        object.__setattr__(self, "radius", _positive(self.radius, "cap radius"))

    @property
    def frame(self) -> AffineFrame3D:
        return AffineFrame3D.from_axis(
            self.center,
            self.normal,
            radial_axis=self.radial_axis,
        )

    @property
    def characteristic_points(self) -> tuple[tuple[float, float, float], ...]:
        frame = self.frame
        center = np.asarray(self.center, dtype=float)
        x_axis = np.asarray(frame.x_axis, dtype=float)
        y_axis = np.asarray(frame.y_axis, dtype=float)
        return tuple(
            tuple(float(item) for item in point)
            for point in (
                center + self.radius * x_axis,
                center - self.radius * x_axis,
                center + self.radius * y_axis,
                center - self.radius * y_axis,
            )
        )

    def contains_point(self, point: Sequence[float], *, context: ContextInput = None) -> bool:
        value = np.asarray(_point3(point, "cap query point"), dtype=float)
        resolved = _resolve(context, self.characteristic_points)
        local = self.frame.to_local_point(value)
        epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
        return (
            abs(float(local[2])) <= epsilon
            and float(np.linalg.norm(local[:2])) <= self.radius + epsilon
        )

    def ray_hits(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
        forward_only: bool = True,
    ) -> tuple[QuadricRayHit, ...]:
        point, vector = _normalized_ray(origin, direction)
        resolved = _resolve(context, self.characteristic_points)
        normal = np.asarray(self.normal, dtype=float)
        denominator = float(np.dot(vector, normal))
        if abs(denominator) <= resolved.epsilon(GeometryQuantity.ANGULAR):
            return ()
        parameter = float(
            np.dot(np.asarray(self.center, dtype=float) - point, normal)
            / denominator
        )
        boundary = resolved.epsilon(GeometryQuantity.BOUNDARY)
        if forward_only and parameter < -boundary:
            return ()
        if forward_only and parameter < 0.0:
            parameter = 0.0
        hit_point = point + parameter * vector
        local = self.frame.to_local_point(hit_point)
        if float(np.linalg.norm(local[:2])) > self.radius + boundary:
            return ()
        return (
            QuadricRayHit(
                parameter,
                tuple(float(item) for item in hit_point),
                self.normal,
                self.cap_id,
                self.role,
                False,
            ),
        )


@dataclass(frozen=True, slots=True)
class CircularTrimRimSpec:
    """A circular finite-surface boundary without a planar closing disk."""

    rim_id: str
    parent_surface_id: str
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float
    radial_axis: tuple[float, float, float] | None = None
    role: Literal["trim_min", "trim_max"] = "trim_max"

    def __post_init__(self) -> None:
        rim_id = _identity(self.rim_id, "rim_id")
        parent_id = _identity(self.parent_surface_id, "parent_surface_id")
        center = _point3(self.center, "trim-rim center")
        normal = _point3(self.normal, "trim-rim normal")
        radial = (
            None
            if self.radial_axis is None
            else _point3(self.radial_axis, "trim-rim radial_axis")
        )
        if self.role not in {"trim_min", "trim_max"}:
            raise QuadricContractError(
                "trim-rim role must be 'trim_min' or 'trim_max'"
            )
        frame = _frame(center, normal, radial, "trim rim")
        object.__setattr__(self, "rim_id", rim_id)
        object.__setattr__(self, "parent_surface_id", parent_id)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal", frame.z_axis)
        object.__setattr__(self, "radial_axis", frame.x_axis)
        object.__setattr__(self, "radius", _positive(self.radius, "trim-rim radius"))

    @property
    def frame(self) -> AffineFrame3D:
        return AffineFrame3D.from_axis(
            self.center,
            self.normal,
            radial_axis=self.radial_axis,
        )

    @property
    def characteristic_points(self) -> tuple[tuple[float, float, float], ...]:
        frame = self.frame
        center = np.asarray(self.center, dtype=float)
        x_axis = np.asarray(frame.x_axis, dtype=float)
        y_axis = np.asarray(frame.y_axis, dtype=float)
        return tuple(
            tuple(float(item) for item in point)
            for point in (
                center + self.radius * x_axis,
                center - self.radius * x_axis,
                center + self.radius * y_axis,
                center - self.radius * y_axis,
            )
        )


def _support_ray_hits(
    support: HomogeneousQuadric,
    surface_id: str,
    origin: Sequence[float],
    direction: Sequence[float],
    *,
    context: ResolvedGeometryContext,
    frame: AffineFrame3D | None,
    axial_range: tuple[float, float] | None,
    forward_only: bool,
) -> tuple[QuadricRayHit, ...]:
    point, vector = _normalized_ray(origin, direction)
    try:
        roots = support.real_ray_parameters(point, vector, context=context)
    except CoincidentRayError:
        # A generator or cylinder ruling belongs to the complete support
        # surface.  It has no isolated lateral hits; finite end caps remain
        # independently queryable by the entity.
        return ()
    boundary = context.epsilon(GeometryQuantity.BOUNDARY)
    filtered: list[float] = []
    for raw in roots:
        parameter = float(raw)
        if forward_only and parameter < -boundary:
            continue
        if forward_only and parameter < 0.0:
            parameter = 0.0
        hit_point = point + parameter * vector
        if frame is not None and axial_range is not None:
            axial = float(frame.to_local_point(hit_point)[2])
            if axial < axial_range[0] - boundary or axial > axial_range[1] + boundary:
                continue
        filtered.append(parameter)

    result: list[QuadricRayHit] = []
    tangent = len(roots) == 1
    for parameter in sorted(filtered):
        hit_point = point + parameter * vector
        result.append(
            QuadricRayHit(
                parameter,
                tuple(float(item) for item in hit_point),
                _normal(
                    support.gradient(hit_point),
                    epsilon=context.epsilon(GeometryQuantity.LENGTH),
                ),
                surface_id,
                "support",
                tangent,
            )
        )
    return tuple(result)


def _combined_hits(
    lateral: Sequence[QuadricRayHit],
    caps: Sequence[PlanarCapSpec],
    origin: Sequence[float],
    direction: Sequence[float],
    *,
    context: ResolvedGeometryContext,
    include_caps: bool,
    forward_only: bool,
) -> tuple[QuadricRayHit, ...]:
    result = list(lateral)
    if include_caps:
        for cap in caps:
            result.extend(
                cap.ray_hits(
                    origin,
                    direction,
                    context=context,
                    forward_only=forward_only,
                )
            )
    return tuple(sorted(result, key=lambda item: (item.parameter, item.role, item.surface_id)))


@dataclass(frozen=True, slots=True)
class SphereSpec:
    surface_id: str
    center: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface_id", _identity(self.surface_id, "surface_id"))
        object.__setattr__(self, "center", _point3(self.center, "sphere center"))
        object.__setattr__(self, "radius", _positive(self.radius, "sphere radius"))

    @property
    def frame(self) -> AffineFrame3D:
        return AffineFrame3D.from_axis(
            self.center,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )

    @property
    def support_quadric(self) -> HomogeneousQuadric:
        return _local_quadric(
            (1.0, 1.0, 1.0, -(self.radius * self.radius)),
            self.frame,
        )

    @property
    def quadric(self) -> HomogeneousQuadric:
        """Compatibility shorthand for the infinite support surface."""

        return self.support_quadric

    @property
    def end_caps(self) -> tuple[PlanarCapSpec, ...]:
        return ()

    @property
    def characteristic_points(self) -> tuple[tuple[float, float, float], ...]:
        center = np.asarray(self.center, dtype=float)
        return tuple(
            tuple(float(item) for item in center + sign * self.radius * axis)
            for axis in np.eye(3)
            for sign in (-1.0, 1.0)
        )

    def contains(self, point: Sequence[float], *, context: ContextInput = None) -> bool:
        value = np.asarray(_point3(point, "sphere query point"), dtype=float)
        resolved = _resolve(context, self.characteristic_points)
        return float(np.linalg.norm(value - np.asarray(self.center, dtype=float))) <= (
            self.radius + resolved.epsilon(GeometryQuantity.BOUNDARY)
        )

    def support_ray_parameters(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
    ) -> tuple[float, ...]:
        point, vector = _normalized_ray(origin, direction)
        resolved = _resolve(context, self.characteristic_points)
        return self.support_quadric.real_ray_parameters(point, vector, context=resolved)

    def lateral_ray_hits(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
        forward_only: bool = True,
    ) -> tuple[QuadricRayHit, ...]:
        resolved = _resolve(context, self.characteristic_points)
        return _support_ray_hits(
            self.support_quadric,
            self.surface_id,
            origin,
            direction,
            context=resolved,
            frame=None,
            axial_range=None,
            forward_only=forward_only,
        )

    def ray_hits(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
        include_caps: bool = True,
        forward_only: bool = True,
    ) -> tuple[QuadricRayHit, ...]:
        del include_caps
        return self.lateral_ray_hits(
            origin,
            direction,
            context=context,
            forward_only=forward_only,
        )


@dataclass(frozen=True, slots=True)
class CylinderSpec:
    """A finite circular cylinder around an infinite cylindrical support."""

    surface_id: str
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius: float
    axial_range: tuple[float, float]
    radial_axis: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        surface_id = _identity(self.surface_id, "surface_id")
        origin = _point3(self.origin, "cylinder origin")
        axis = _point3(self.axis, "cylinder axis")
        radial = (
            None
            if self.radial_axis is None
            else _point3(self.radial_axis, "cylinder radial_axis")
        )
        frame = _frame(origin, axis, radial, "cylinder")
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "axis", frame.z_axis)
        object.__setattr__(self, "radial_axis", frame.x_axis)
        object.__setattr__(self, "radius", _positive(self.radius, "cylinder radius"))
        object.__setattr__(self, "axial_range", _axial_range(self.axial_range))

    @property
    def frame(self) -> AffineFrame3D:
        return AffineFrame3D.from_axis(
            self.origin,
            self.axis,
            radial_axis=self.radial_axis,
        )

    @property
    def support_quadric(self) -> HomogeneousQuadric:
        return _local_quadric(
            (1.0, 1.0, 0.0, -(self.radius * self.radius)),
            self.frame,
        )

    @property
    def quadric(self) -> HomogeneousQuadric:
        return self.support_quadric

    @property
    def end_caps(self) -> tuple[PlanarCapSpec, PlanarCapSpec]:
        frame = self.frame
        axis = np.asarray(frame.z_axis, dtype=float)
        origin = np.asarray(self.origin, dtype=float)
        lower, upper = self.axial_range
        return (
            PlanarCapSpec(
                f"{self.surface_id}:cap:min",
                self.surface_id,
                tuple(float(item) for item in origin + lower * axis),
                tuple(float(item) for item in -axis),
                self.radius,
                radial_axis=frame.x_axis,
                role="cap_min",
            ),
            PlanarCapSpec(
                f"{self.surface_id}:cap:max",
                self.surface_id,
                tuple(float(item) for item in origin + upper * axis),
                frame.z_axis,
                self.radius,
                radial_axis=frame.x_axis,
                role="cap_max",
            ),
        )

    @property
    def characteristic_points(self) -> tuple[tuple[float, float, float], ...]:
        frame = self.frame
        origin = np.asarray(self.origin, dtype=float)
        axis = np.asarray(frame.z_axis, dtype=float)
        x_axis = np.asarray(frame.x_axis, dtype=float)
        y_axis = np.asarray(frame.y_axis, dtype=float)
        return tuple(
            tuple(float(item) for item in origin + axial * axis + self.radius * radial)
            for axial in self.axial_range
            for radial in (x_axis, -x_axis, y_axis, -y_axis)
        )

    def contains(self, point: Sequence[float], *, context: ContextInput = None) -> bool:
        value = np.asarray(_point3(point, "cylinder query point"), dtype=float)
        resolved = _resolve(context, self.characteristic_points)
        local = self.frame.to_local_point(value)
        epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
        return (
            self.axial_range[0] - epsilon
            <= float(local[2])
            <= self.axial_range[1] + epsilon
            and float(np.linalg.norm(local[:2])) <= self.radius + epsilon
        )

    def support_ray_parameters(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
    ) -> tuple[float, ...]:
        point, vector = _normalized_ray(origin, direction)
        resolved = _resolve(context, self.characteristic_points)
        return self.support_quadric.real_ray_parameters(point, vector, context=resolved)

    def lateral_ray_hits(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
        forward_only: bool = True,
    ) -> tuple[QuadricRayHit, ...]:
        resolved = _resolve(context, self.characteristic_points)
        return _support_ray_hits(
            self.support_quadric,
            self.surface_id,
            origin,
            direction,
            context=resolved,
            frame=self.frame,
            axial_range=self.axial_range,
            forward_only=forward_only,
        )

    def ray_hits(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
        include_caps: bool = True,
        forward_only: bool = True,
    ) -> tuple[QuadricRayHit, ...]:
        resolved = _resolve(context, self.characteristic_points)
        return _combined_hits(
            self.lateral_ray_hits(
                origin,
                direction,
                context=resolved,
                forward_only=forward_only,
            ),
            self.end_caps,
            origin,
            direction,
            context=resolved,
            include_caps=include_caps,
            forward_only=forward_only,
        )


@dataclass(frozen=True, slots=True)
class ConeSpec:
    """A finite cone solid, open shell, or analytic double-cone slice.

    Omitting ``model`` keeps historical construction compatible: a range on
    one side of the apex becomes ``CLOSED_SINGLE`` while a range crossing the
    apex becomes ``ANALYTIC_DOUBLE``.  The latter remains available to exact
    conic-section code but is not silently treated as a renderable solid.
    """

    surface_id: str
    apex: tuple[float, float, float]
    axis: tuple[float, float, float]
    half_angle: float
    axial_range: tuple[float, float]
    radial_axis: tuple[float, float, float] | None = None
    model: ConeModel | str | None = None
    component_parent_id: str | None = None

    def __post_init__(self) -> None:
        surface_id = _identity(self.surface_id, "surface_id")
        apex = _point3(self.apex, "cone apex")
        axis = _point3(self.axis, "cone axis")
        radial = (
            None
            if self.radial_axis is None
            else _point3(self.radial_axis, "cone radial_axis")
        )
        if isinstance(self.half_angle, bool):
            raise QuadricContractError(
                "cone half_angle must lie strictly between 0 and pi/2"
            )
        try:
            half_angle = float(self.half_angle)
        except (TypeError, ValueError, OverflowError) as exc:
            raise QuadricContractError(
                "cone half_angle must lie strictly between 0 and pi/2"
            ) from exc
        if not isfinite(half_angle) or not 0.0 < half_angle < 0.5 * pi:
            raise QuadricContractError(
                "cone half_angle must lie strictly between 0 and pi/2"
            )
        frame = _frame(apex, axis, radial, "cone")
        axial_range = _axial_range(self.axial_range)
        crosses_apex = axial_range[0] < 0.0 < axial_range[1]
        if self.model is None:
            model = (
                ConeModel.ANALYTIC_DOUBLE
                if crosses_apex
                else ConeModel.CLOSED_SINGLE
            )
        else:
            try:
                model = ConeModel(self.model)
            except (TypeError, ValueError) as exc:
                raise QuadricContractError(
                    "cone model must be 'closed_single', 'open_single', "
                    "'open_double', or 'analytic_double'"
                ) from exc
        if model in {ConeModel.CLOSED_SINGLE, ConeModel.OPEN_SINGLE} and crosses_apex:
            raise QuadricContractError(
                f"cone model {model.value!r} requires one nappe; axial_range "
                "must not cross the apex"
            )
        if (
            model in {ConeModel.OPEN_DOUBLE, ConeModel.ANALYTIC_DOUBLE}
            and not crosses_apex
        ):
            raise QuadricContractError(
                f"cone model {model.value!r} requires axial_range to cross "
                "the apex"
            )
        component_parent_id = (
            None
            if self.component_parent_id is None
            else _identity(self.component_parent_id, "cone component_parent_id")
        )
        if component_parent_id is not None and model is not ConeModel.OPEN_SINGLE:
            raise QuadricContractError(
                "cone component_parent_id is reserved for open-double "
                "single-nappe render components"
            )
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "apex", apex)
        object.__setattr__(self, "axis", frame.z_axis)
        object.__setattr__(self, "radial_axis", frame.x_axis)
        object.__setattr__(self, "half_angle", half_angle)
        object.__setattr__(self, "axial_range", axial_range)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "component_parent_id", component_parent_id)

    @property
    def frame(self) -> AffineFrame3D:
        return AffineFrame3D.from_axis(
            self.apex,
            self.axis,
            radial_axis=self.radial_axis,
        )

    @property
    def slope(self) -> float:
        return float(tan(self.half_angle))

    @property
    def support_quadric(self) -> HomogeneousQuadric:
        slope = self.slope
        return _local_quadric(
            (1.0, 1.0, -(slope * slope), 0.0),
            self.frame,
        )

    @property
    def quadric(self) -> HomogeneousQuadric:
        return self.support_quadric

    @property
    def is_open_shell(self) -> bool:
        return self.model in {ConeModel.OPEN_SINGLE, ConeModel.OPEN_DOUBLE}

    @property
    def nappe_count(self) -> int:
        return 2 if self.model in {
            ConeModel.OPEN_DOUBLE,
            ConeModel.ANALYTIC_DOUBLE,
        } else 1

    @property
    def is_directly_renderable(self) -> bool:
        return self.model is not ConeModel.ANALYTIC_DOUBLE

    @property
    def end_caps(self) -> tuple[PlanarCapSpec, ...]:
        if self.is_open_shell:
            return ()
        frame = self.frame
        axis = np.asarray(frame.z_axis, dtype=float)
        apex = np.asarray(self.apex, dtype=float)
        result: list[PlanarCapSpec] = []
        for axial, role, normal in (
            (self.axial_range[0], "cap_min", -axis),
            (self.axial_range[1], "cap_max", axis),
        ):
            radius = abs(axial) * self.slope
            # An endpoint at the apex is a point, not a planar disk.
            if radius == 0.0:
                continue
            result.append(
                PlanarCapSpec(
                    f"{self.surface_id}:cap:{'min' if role == 'cap_min' else 'max'}",
                    self.surface_id,
                    tuple(float(item) for item in apex + axial * axis),
                    tuple(float(item) for item in normal),
                    radius,
                    radial_axis=frame.x_axis,
                    role=role,  # type: ignore[arg-type]
                )
            )
        return tuple(result)

    @property
    def trim_rims(self) -> tuple[CircularTrimRimSpec, ...]:
        if not self.is_open_shell:
            return ()
        frame = self.frame
        axis = np.asarray(frame.z_axis, dtype=float)
        apex = np.asarray(self.apex, dtype=float)
        result: list[CircularTrimRimSpec] = []
        for axial, suffix, role, normal in (
            (self.axial_range[0], "min", "trim_min", -axis),
            (self.axial_range[1], "max", "trim_max", axis),
        ):
            radius = abs(axial) * self.slope
            if radius == 0.0:
                continue
            result.append(
                CircularTrimRimSpec(
                    f"{self.surface_id}:trim:{suffix}",
                    self.surface_id,
                    tuple(float(item) for item in apex + axial * axis),
                    tuple(float(item) for item in normal),
                    radius,
                    radial_axis=frame.x_axis,
                    role=role,  # type: ignore[arg-type]
                )
            )
        return tuple(result)

    @property
    def render_components(self) -> tuple["ConeSpec", ...]:
        """Return stable convex components for a finite display model."""

        if self.model is ConeModel.ANALYTIC_DOUBLE:
            raise QuadricContractError(
                "analytic_double cones are mathematical section supports only; "
                "use model='open_double' for a finite renderable double shell"
            )
        if self.model is not ConeModel.OPEN_DOUBLE:
            return (self,)
        lower, upper = self.axial_range
        common = {
            "apex": self.apex,
            "axis": self.axis,
            "half_angle": self.half_angle,
            "radial_axis": self.radial_axis,
            "model": ConeModel.OPEN_SINGLE,
            "component_parent_id": self.surface_id,
        }
        return (
            ConeSpec(
                f"{self.surface_id}:nappe:negative",
                axial_range=(lower, 0.0),
                **common,
            ),
            ConeSpec(
                f"{self.surface_id}:nappe:positive",
                axial_range=(0.0, upper),
                **common,
            ),
        )

    @property
    def characteristic_points(self) -> tuple[tuple[float, float, float], ...]:
        frame = self.frame
        apex = np.asarray(self.apex, dtype=float)
        axis = np.asarray(frame.z_axis, dtype=float)
        x_axis = np.asarray(frame.x_axis, dtype=float)
        y_axis = np.asarray(frame.y_axis, dtype=float)
        result = [apex]
        for axial in self.axial_range:
            center = apex + axial * axis
            radius = abs(axial) * self.slope
            result.extend(
                center + radius * radial
                for radial in (x_axis, -x_axis, y_axis, -y_axis)
            )
        return tuple(tuple(float(item) for item in point) for point in result)

    def contains(self, point: Sequence[float], *, context: ContextInput = None) -> bool:
        if self.is_open_shell:
            raise QuadricContractError(
                "an open cone shell has no filled-volume contains relation"
            )
        value = np.asarray(_point3(point, "cone query point"), dtype=float)
        resolved = _resolve(context, self.characteristic_points)
        local = self.frame.to_local_point(value)
        epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
        radius = abs(float(local[2])) * self.slope
        return (
            self.axial_range[0] - epsilon
            <= float(local[2])
            <= self.axial_range[1] + epsilon
            and float(np.linalg.norm(local[:2])) <= radius + epsilon
        )

    def support_ray_parameters(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
    ) -> tuple[float, ...]:
        point, vector = _normalized_ray(origin, direction)
        resolved = _resolve(context, self.characteristic_points)
        return self.support_quadric.real_ray_parameters(point, vector, context=resolved)

    def lateral_ray_hits(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
        forward_only: bool = True,
    ) -> tuple[QuadricRayHit, ...]:
        resolved = _resolve(context, self.characteristic_points)
        return _support_ray_hits(
            self.support_quadric,
            self.surface_id,
            origin,
            direction,
            context=resolved,
            frame=self.frame,
            axial_range=self.axial_range,
            forward_only=forward_only,
        )

    def ray_hits(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: ContextInput = None,
        include_caps: bool = True,
        forward_only: bool = True,
    ) -> tuple[QuadricRayHit, ...]:
        resolved = _resolve(context, self.characteristic_points)
        return _combined_hits(
            self.lateral_ray_hits(
                origin,
                direction,
                context=resolved,
                forward_only=forward_only,
            ),
            self.end_caps,
            origin,
            direction,
            context=resolved,
            include_caps=include_caps,
            forward_only=forward_only,
        )


@dataclass(frozen=True, slots=True)
class SectionPlane:
    """An infinite mathematical plane with a stable local coordinate frame."""

    plane_id: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    u_axis: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        plane_id = _identity(self.plane_id, "plane_id")
        point = _point3(self.point, "plane point")
        normal = _point3(self.normal, "plane normal")
        u_axis = None if self.u_axis is None else _point3(self.u_axis, "plane u_axis")
        frame = _frame(point, normal, u_axis, "section plane")
        object.__setattr__(self, "plane_id", plane_id)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "normal", frame.z_axis)
        object.__setattr__(self, "u_axis", frame.x_axis)

    @property
    def frame(self) -> AffineFrame3D:
        return AffineFrame3D.from_axis(
            self.point,
            self.normal,
            radial_axis=self.u_axis,
        )

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = self.frame
        return (
            np.asarray(frame.x_axis, dtype=float),
            np.asarray(frame.y_axis, dtype=float),
            np.asarray(frame.z_axis, dtype=float),
        )

    def signed_distance(self, point: Sequence[float]) -> float:
        value = np.asarray(_point3(point, "plane query point"), dtype=float)
        return float(
            np.dot(
                value - np.asarray(self.point, dtype=float),
                np.asarray(self.normal, dtype=float),
            )
        )

    def coordinates_in_plane(self, point: Sequence[float]) -> tuple[float, float]:
        local = self.frame.to_local_point(_point3(point, "plane query point"))
        return float(local[0]), float(local[1])

    def point_from_coordinates(self, value: Sequence[float]) -> np.ndarray:
        u, v = _point2(value, "plane coordinates")
        return self.frame.to_world_point((u, v, 0.0))

    def restrict(self, quadric: HomogeneousQuadric) -> np.ndarray:
        if not isinstance(quadric, HomogeneousQuadric):
            raise QuadricContractError("plane restriction requires a HomogeneousQuadric")
        u_axis, v_axis, _normal = self.basis
        return quadric.restrict_to_affine_plane(self.point, u_axis, v_axis)


@dataclass(frozen=True, slots=True)
class PlaneDisplayPatchSpec:
    """A finite display rectangle attached to, but not defining, a plane."""

    patch_id: str
    plane_id: str
    half_width: float
    half_height: float
    center_coordinates: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patch_id", _identity(self.patch_id, "patch_id"))
        object.__setattr__(self, "plane_id", _identity(self.plane_id, "plane_id"))
        object.__setattr__(self, "half_width", _positive(self.half_width, "half_width"))
        object.__setattr__(self, "half_height", _positive(self.half_height, "half_height"))
        object.__setattr__(
            self,
            "center_coordinates",
            _point2(self.center_coordinates, "center_coordinates"),
        )

    def corners(self, plane: SectionPlane) -> tuple[tuple[float, float, float], ...]:
        if not isinstance(plane, SectionPlane):
            raise QuadricContractError("display patch requires a SectionPlane")
        if plane.plane_id != self.plane_id:
            raise QuadricContractError(
                "display patch plane_id does not match the supplied plane"
            )
        center_u, center_v = self.center_coordinates
        coordinates = (
            (center_u - self.half_width, center_v - self.half_height),
            (center_u + self.half_width, center_v - self.half_height),
            (center_u + self.half_width, center_v + self.half_height),
            (center_u - self.half_width, center_v + self.half_height),
        )
        return tuple(
            tuple(float(item) for item in plane.point_from_coordinates(value))
            for value in coordinates
        )

    def contains_coordinates(
        self,
        value: Sequence[float],
        *,
        context: ContextInput = None,
    ) -> bool:
        u, v = _point2(value, "display-patch coordinates")
        characteristic = (
            (-self.half_width, -self.half_height, 0.0),
            (self.half_width, self.half_height, 0.0),
        )
        resolved = _resolve(context, characteristic)
        epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
        center_u, center_v = self.center_coordinates
        return (
            abs(u - center_u) <= self.half_width + epsilon
            and abs(v - center_v) <= self.half_height + epsilon
        )


__all__ = [
    "CircularTrimRimSpec",
    "ConeModel",
    "ConeSpec",
    "CylinderSpec",
    "PlanarCapSpec",
    "PlaneDisplayPatchSpec",
    "QuadricContractError",
    "QuadricRayHit",
    "SectionPlane",
    "SphereSpec",
]
