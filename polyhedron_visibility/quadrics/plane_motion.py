"""Exact critical schedules for an axis-angle moving section plane.

The continuity tracker in :mod:`.animation` intentionally consumes authored
frames and never searches between them.  This module is the authoring layer
that supplies the important frames for a common teaching motion: one infinite
cutting plane rotating rigidly about a fixed axis.

For a rotated vector, every dot product with a fixed vector has the form
``A*cos(theta) + B*sin(theta) + C``.  Sphere tangencies, cylinder-parallel
positions, cone parabolic positions, cone-apex degeneracies, and contact with
finite cylinder/cone trim circles therefore have algebraic roots.  Those
roots, not a dense animation sampling, are inserted into the schedule before
the ordinary continuity tracker is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import acos, atan2, ceil, floor, isfinite, pi, sin, tau
from typing import Sequence

import numpy as np

from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from .animation import (
    SectionAnimationSample,
    SectionAnimationTrace,
    track_quadric_section_animation,
)
from .contract import ConeSpec, CylinderSpec, SectionPlane, SphereSpec
from .roots import PolynomialRootError, solve_real_polynomial
from .sections import QuadricSurfaceSpec


PLANE_MOTION_SCHEDULE_SCHEMA = "manim-quadric-plane-motion-schedule/v1"
_FLOAT_EPSILON = float(np.finfo(float).eps)


class PlaneMotionError(ValueError):
    """A rotating-plane schedule cannot be formed without guessing."""


class PlaneMotionCriticalKind(str, Enum):
    SPHERE_TANGENCY = "sphere_tangency"
    CYLINDER_AXIS_PARALLEL = "cylinder_axis_parallel"
    CYLINDER_TRIM_TANGENCY = "cylinder_trim_tangency"
    CONE_PARABOLIC = "cone_parabolic"
    CONE_APEX_DEGENERACY = "cone_apex_degeneracy"
    CONE_TRIM_TANGENCY = "cone_trim_tangency"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlaneMotionError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise PlaneMotionError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlaneMotionError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise PlaneMotionError(f"{label} must be finite")
    return result


def _point3(value: object, label: str) -> tuple[float, float, float]:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlaneMotionError(f"{label} must contain three finite values") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise PlaneMotionError(f"{label} must contain three finite values")
    return tuple(float(item) for item in result)  # type: ignore[return-value]


def _unit3(value: object, label: str) -> tuple[float, float, float]:
    result = np.asarray(_point3(value, label), dtype=float)
    length = float(np.linalg.norm(result))
    if length <= 0.0:
        raise PlaneMotionError(f"{label} must be non-zero")
    return tuple(float(item) for item in result / length)  # type: ignore[return-value]


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    skew = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=float,
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


@dataclass(frozen=True, slots=True)
class AxisAnglePlaneMotion:
    """Rigidly rotate one mathematical plane about a fixed world-space axis."""

    motion_id: str
    base_plane: SectionPlane
    axis_point: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    start_angle: float
    end_angle: float
    start_time: float = 0.0
    end_time: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "motion_id", _identity(self.motion_id, "motion_id"))
        if not isinstance(self.base_plane, SectionPlane):
            raise TypeError("base_plane must be a SectionPlane")
        point = _point3(self.axis_point, "axis_point")
        direction = _unit3(self.axis_direction, "axis_direction")
        start = _finite(self.start_angle, "start_angle")
        end = _finite(self.end_angle, "end_angle")
        if start == end:
            raise PlaneMotionError("plane motion angle interval must be non-zero")
        if abs(end - start) > tau + 1.0e-12:
            raise PlaneMotionError(
                "one plane-motion schedule cannot span more than one revolution"
            )
        start_time = _finite(self.start_time, "start_time")
        end_time = _finite(self.end_time, "end_time")
        if end_time <= start_time:
            raise PlaneMotionError("plane motion times must increase")
        object.__setattr__(self, "axis_point", point)
        object.__setattr__(self, "axis_direction", direction)
        object.__setattr__(self, "start_angle", start)
        object.__setattr__(self, "end_angle", end)
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)

    def angle_at(self, progress: float) -> float:
        value = _finite(progress, "motion progress")
        if value < 0.0 or value > 1.0:
            raise PlaneMotionError("motion progress must lie in [0, 1]")
        return self.start_angle + value * (self.end_angle - self.start_angle)

    def time_at(self, progress: float) -> float:
        value = _finite(progress, "motion progress")
        if value < 0.0 or value > 1.0:
            raise PlaneMotionError("motion progress must lie in [0, 1]")
        return self.start_time + value * (self.end_time - self.start_time)

    def plane_at(self, progress: float) -> SectionPlane:
        rotation = _rotation(
            np.asarray(self.axis_direction, dtype=float), self.angle_at(progress)
        )
        pivot = np.asarray(self.axis_point, dtype=float)
        point = pivot + rotation @ (
            np.asarray(self.base_plane.point, dtype=float) - pivot
        )
        normal = rotation @ np.asarray(self.base_plane.normal, dtype=float)
        u_axis = rotation @ np.asarray(self.base_plane.u_axis, dtype=float)
        return SectionPlane(
            self.base_plane.plane_id,
            tuple(float(item) for item in point),
            tuple(float(item) for item in normal),
            u_axis=tuple(float(item) for item in u_axis),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "motionId": self.motion_id,
            "planeId": self.base_plane.plane_id,
            "axisPoint": list(self.axis_point),
            "axisDirection": list(self.axis_direction),
            "startAngle": self.start_angle,
            "endAngle": self.end_angle,
            "startTime": self.start_time,
            "endTime": self.end_time,
        }


@dataclass(frozen=True, slots=True)
class PlaneMotionCriticalEvent:
    event_id: str
    progress: float
    time: float
    angle: float
    kinds: tuple[PlaneMotionCriticalKind, ...]
    equations: tuple[str, ...]
    persistent: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identity(self.event_id, "event_id"))
        progress = _finite(self.progress, "critical progress")
        if progress < 0.0 or progress > 1.0:
            raise PlaneMotionError("critical progress must lie in [0, 1]")
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "time", _finite(self.time, "critical time"))
        object.__setattr__(self, "angle", _finite(self.angle, "critical angle"))
        kinds = tuple(sorted(set(self.kinds), key=lambda item: item.value))
        if not kinds or kinds != self.kinds:
            raise PlaneMotionError("critical kinds must be unique and canonical")
        equations = tuple(sorted(set(_identity(item, "equation") for item in self.equations)))
        if not equations or equations != self.equations:
            raise PlaneMotionError("critical equations must be unique and canonical")
        if not isinstance(self.persistent, bool):
            raise TypeError("persistent must be a bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "eventId": self.event_id,
            "progress": self.progress,
            "time": self.time,
            "angle": self.angle,
            "kinds": [item.value for item in self.kinds],
            "equations": list(self.equations),
            "persistent": self.persistent,
        }


@dataclass(frozen=True, slots=True)
class PlaneMotionSchedule:
    motion: AxisAnglePlaneMotion
    surface_id: str
    progresses: tuple[float, ...]
    samples: tuple[SectionAnimationSample, ...]
    critical_events: tuple[PlaneMotionCriticalEvent, ...]
    schema: str = PLANE_MOTION_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLANE_MOTION_SCHEDULE_SCHEMA:
            raise PlaneMotionError("invalid plane-motion schedule schema")
        if not isinstance(self.motion, AxisAnglePlaneMotion):
            raise TypeError("motion must be AxisAnglePlaneMotion")
        surface_id = _identity(self.surface_id, "surface_id")
        if not self.progresses or self.progresses[0] != 0.0 or self.progresses[-1] != 1.0:
            raise PlaneMotionError("schedule must include progress endpoints")
        if any(right <= left for left, right in zip(self.progresses, self.progresses[1:])):
            raise PlaneMotionError("schedule progresses must increase strictly")
        if len(self.samples) != len(self.progresses):
            raise PlaneMotionError("schedule samples must cover every progress")
        for progress, sample in zip(self.progresses, self.samples):
            if abs(sample.time - self.motion.time_at(progress)) > 1.0e-12:
                raise PlaneMotionError("sample time disagrees with motion progress")
            if sample.surface.surface_id != surface_id:
                raise PlaneMotionError("sample surface identity changed")
        event_progresses = tuple(item.progress for item in self.critical_events)
        if event_progresses != tuple(sorted(event_progresses)):
            raise PlaneMotionError("critical events must use canonical order")
        object.__setattr__(self, "surface_id", surface_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "surfaceId": self.surface_id,
            "motion": self.motion.to_dict(),
            "progresses": list(self.progresses),
            "criticalEvents": [item.to_dict() for item in self.critical_events],
            "samples": [
                {
                    "time": sample.time,
                    "plane": {
                        "planeId": sample.plane.plane_id,
                        "point": list(sample.plane.point),
                        "normal": list(sample.plane.normal),
                        "uAxis": list(sample.plane.u_axis),
                    },
                }
                for sample in self.samples
            ],
        }


@dataclass(frozen=True, slots=True)
class ScheduledSectionAnimation:
    schedule: PlaneMotionSchedule
    animation: SectionAnimationTrace

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, PlaneMotionSchedule):
            raise TypeError("schedule must be a PlaneMotionSchedule")
        if not isinstance(self.animation, SectionAnimationTrace):
            raise TypeError("animation must be a SectionAnimationTrace")
        if len(self.schedule.samples) != len(self.animation.frames):
            raise PlaneMotionError("schedule and animation frame counts disagree")

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule": self.schedule.to_dict(),
            "animation": self.animation.to_dict(),
        }


def _harmonic_coefficients(
    motion: AxisAnglePlaneMotion,
    fixed_vector: Sequence[float],
    *,
    constant_offset: float = 0.0,
) -> tuple[float, float, float]:
    axis = np.asarray(motion.axis_direction, dtype=float)
    normal = np.asarray(motion.base_plane.normal, dtype=float)
    fixed = np.asarray(fixed_vector, dtype=float)
    parallel = axis * float(np.dot(axis, normal))
    cosine_vector = normal - parallel
    sine_vector = np.cross(axis, normal)
    return (
        float(np.dot(cosine_vector, fixed)),
        float(np.dot(sine_vector, fixed)),
        float(np.dot(parallel, fixed) + constant_offset),
    )


def _harmonic_roots(
    coefficients: tuple[float, float, float],
    target: float,
    motion: AxisAnglePlaneMotion,
) -> tuple[tuple[float, ...], bool]:
    cosine, sine_coefficient, constant = coefficients
    amplitude = float(np.hypot(cosine, sine_coefficient))
    scale = max(1.0, abs(cosine), abs(sine_coefficient), abs(constant), abs(target))
    tolerance = 4096.0 * _FLOAT_EPSILON * scale
    if amplitude <= tolerance:
        return ((), abs(constant - target) <= tolerance)
    ratio = (target - constant) / amplitude
    if ratio < -1.0 - tolerance or ratio > 1.0 + tolerance:
        return (), False
    ratio = min(1.0, max(-1.0, ratio))
    phase = atan2(sine_coefficient, cosine)
    offset = acos(ratio)
    angle_low = min(motion.start_angle, motion.end_angle)
    angle_high = max(motion.start_angle, motion.end_angle)
    angles: list[float] = []
    for base in (phase - offset, phase + offset):
        first = floor((angle_low - base) / tau) - 1
        last = ceil((angle_high - base) / tau) + 1
        for index in range(first, last + 1):
            angle = base + index * tau
            if angle < angle_low - tolerance or angle > angle_high + tolerance:
                continue
            angle = min(angle_high, max(angle_low, angle))
            angles.append(angle)
    progresses = sorted(
        (angle - motion.start_angle) / (motion.end_angle - motion.start_angle)
        for angle in angles
    )
    result: list[float] = []
    for progress in progresses:
        progress = min(1.0, max(0.0, progress))
        if not result or progress - result[-1] > 1.0e-12:
            result.append(float(progress))
    return tuple(result), False


def _trim_circle_polynomial(
    center_coefficients: tuple[float, float, float],
    axis_coefficients: tuple[float, float, float],
    radius: float,
) -> tuple[np.ndarray, float]:
    """Return the half-angle polynomial for plane/rim tangency.

    For a unit plane normal ``n``, rim center ``c``, unit rim axis ``a`` and
    radius ``r``, contact occurs exactly when

    ``plane_value(c)^2 + r^2 * dot(n, a)^2 - r^2 == 0``.

    Substituting ``t = tan(theta / 2)`` turns both harmonic terms into
    quadratics over ``1 + t^2`` and therefore yields one quartic polynomial.
    Coefficients are stored in ascending power order.
    """

    def numerator(
        coefficients: tuple[float, float, float],
    ) -> np.ndarray:
        cosine, sine_coefficient, constant = coefficients
        return np.asarray(
            (
                cosine + constant,
                2.0 * sine_coefficient,
                constant - cosine,
            ),
            dtype=float,
        )

    length_scale = max(
        *(abs(value) for value in center_coefficients),
        radius,
        np.finfo(float).tiny,
    )
    center_numerator = numerator(
        tuple(value / length_scale for value in center_coefficients)
    )
    axis_numerator = numerator(axis_coefficients)
    denominator_squared = np.asarray((1.0, 0.0, 2.0, 0.0, 1.0))
    center_squared = np.convolve(center_numerator, center_numerator)
    axis_squared = np.convolve(axis_numerator, axis_numerator)
    normalized_radius = radius / length_scale
    radius_squared = normalized_radius * normalized_radius
    polynomial = (
        center_squared
        + radius_squared * axis_squared
        - radius_squared * denominator_squared
    )
    construction_scale = max(
        float(np.max(np.abs(center_squared))),
        float(np.max(np.abs(radius_squared * axis_squared))),
        float(np.max(np.abs(radius_squared * denominator_squared))),
        np.finfo(float).tiny,
    )
    return polynomial, construction_scale


def _angles_in_motion(
    principal_angle: float,
    motion: AxisAnglePlaneMotion,
    *,
    tolerance: float,
) -> tuple[float, ...]:
    angle_low = min(motion.start_angle, motion.end_angle)
    angle_high = max(motion.start_angle, motion.end_angle)
    first = floor((angle_low - principal_angle) / tau) - 1
    last = ceil((angle_high - principal_angle) / tau) + 1
    values: list[float] = []
    for index in range(first, last + 1):
        angle = principal_angle + index * tau
        if angle < angle_low - tolerance or angle > angle_high + tolerance:
            continue
        values.append(min(angle_high, max(angle_low, angle)))
    return tuple(values)


def _trim_circle_residual(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    motion: AxisAnglePlaneMotion,
    progress: float,
) -> tuple[float, float]:
    plane = motion.plane_at(progress)
    normal = np.asarray(plane.normal, dtype=float)
    center_value = float(
        np.dot(normal, center - np.asarray(plane.point, dtype=float))
    )
    axis_value = float(np.dot(normal, axis))
    length_scale = max(abs(center_value), radius, np.finfo(float).tiny)
    normalized_center = center_value / length_scale
    normalized_radius = radius / length_scale
    radius_squared = normalized_radius * normalized_radius
    first = normalized_center * normalized_center
    second = radius_squared * axis_value * axis_value
    residual = first + second - radius_squared
    scale = max(abs(first), abs(second), radius_squared, np.finfo(float).tiny)
    return residual, scale


def _trim_circle_tangency_roots(
    *,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    motion: AxisAnglePlaneMotion,
    context: GeometryContext | ResolvedGeometryContext | None,
) -> tuple[tuple[float, ...], bool]:
    pivot = np.asarray(motion.axis_point, dtype=float)
    base_point = np.asarray(motion.base_plane.point, dtype=float)
    base_normal = np.asarray(motion.base_plane.normal, dtype=float)
    point_offset = float(np.dot(base_normal, base_point - pivot))
    center_coefficients = _harmonic_coefficients(
        motion,
        center - pivot,
        constant_offset=-point_offset,
    )
    axis_coefficients = _harmonic_coefficients(motion, axis)
    polynomial, construction_scale = _trim_circle_polynomial(
        center_coefficients,
        axis_coefficients,
        radius,
    )
    if not np.all(np.isfinite(polynomial)) or not isfinite(construction_scale):
        raise PlaneMotionError(
            "finite trim-circle tangency polynomial is non-finite"
        )
    coefficient_tolerance = 16384.0 * _FLOAT_EPSILON * construction_scale
    if float(np.max(np.abs(polynomial))) <= coefficient_tolerance:
        return (), True

    if isinstance(context, ResolvedGeometryContext):
        resolved = context
    else:
        # Resolve from motion-local coordinates so a large common world
        # translation cannot relax root isolation or merge distinct events.
        resolved = resolve_geometry_context(
            context,
            positions=(
                tuple(float(value) for value in center - pivot),
                tuple(float(value) for value in base_point - pivot),
                (0.0, 0.0, 0.0),
            ),
            edge_length=radius,
        )
    normalized = polynomial / float(np.max(np.abs(polynomial)))
    try:
        algebraic_roots = solve_real_polynomial(
            tuple(float(value) for value in normalized),
            context=resolved,
            parameter_tolerance=resolved.epsilon(GeometryQuantity.PARAMETER),
        )
    except PolynomialRootError as exc:
        raise PlaneMotionError(
            f"finite trim-circle tangency roots are ambiguous: {exc}"
        ) from exc

    angular_tolerance = 32768.0 * _FLOAT_EPSILON * max(
        1.0,
        abs(motion.start_angle),
        abs(motion.end_angle),
    )
    candidate_angles: list[tuple[float, bool]] = []
    for root in algebraic_roots:
        principal = 2.0 * atan2(root.value, 1.0)
        candidate_angles.extend(
            (angle, False)
            for angle in _angles_in_motion(
                principal, motion, tolerance=angular_tolerance
            )
        )

    # ``tan(theta / 2)`` does not represent odd multiples of pi.  Validate
    # those algebraic points explicitly so a leading-zero quartic cannot lose
    # a genuine contact event at infinity.
    candidate_angles.extend(
        (angle, True)
        for angle in _angles_in_motion(
            pi, motion, tolerance=angular_tolerance
        )
    )

    validated_progresses: list[float] = []
    denominator = motion.end_angle - motion.start_angle
    for angle, chart_closure_only in sorted(candidate_angles):
        progress = (angle - motion.start_angle) / denominator
        progress = min(1.0, max(0.0, progress))
        residual, scale = _trim_circle_residual(
            center,
            axis,
            radius,
            motion,
            progress,
        )
        validation_tolerance = 131072.0 * _FLOAT_EPSILON * scale
        if abs(residual) > validation_tolerance:
            # Pi candidates are deliberately unconditional because they close
            # the half-angle chart; non-roots are simply irrelevant.
            if chart_closure_only:
                continue
            raise PlaneMotionError(
                "finite trim-circle tangency root failed geometric validation"
            )
        validated_progresses.append(float(progress))
    progresses: list[float] = []
    for progress in sorted(validated_progresses):
        if not progresses or progress - progresses[-1] > 1.0e-10:
            progresses.append(progress)
    return tuple(progresses), False


def _trim_boundary_events(
    surface: CylinderSpec | ConeSpec,
    motion: AxisAnglePlaneMotion,
    context: GeometryContext | ResolvedGeometryContext | None,
) -> tuple[tuple[PlaneMotionCriticalKind, str, float, bool], ...]:
    axis = np.asarray(surface.axis, dtype=float)
    origin = np.asarray(
        surface.origin if isinstance(surface, CylinderSpec) else surface.apex,
        dtype=float,
    )
    kind = (
        PlaneMotionCriticalKind.CYLINDER_TRIM_TANGENCY
        if isinstance(surface, CylinderSpec)
        else PlaneMotionCriticalKind.CONE_TRIM_TANGENCY
    )
    result: list[tuple[PlaneMotionCriticalKind, str, float, bool]] = []
    for boundary_index, axial in enumerate(surface.axial_range):
        radius = (
            surface.radius
            if isinstance(surface, CylinderSpec)
            else abs(axial) * surface.slope
        )
        # A zero-radius cone trim is the apex event already represented by
        # CONE_APEX_DEGENERACY, not a finite circular rim.
        if radius == 0.0:
            continue
        center = origin + axial * axis
        progresses, persistent = _trim_circle_tangency_roots(
            center=center,
            axis=axis,
            radius=radius,
            motion=motion,
            context=context,
        )
        equation = (
            f"plane_tangent_to_trim_circle[{boundary_index}]"
            f"(axial={axial:.17g},radius={radius:.17g})"
        )
        if persistent:
            result.append((kind, equation, 0.0, True))
        else:
            result.extend(
                (kind, equation, progress, False) for progress in progresses
            )
    return tuple(result)


def _critical_equations(
    surface: QuadricSurfaceSpec,
    motion: AxisAnglePlaneMotion,
) -> tuple[
    tuple[PlaneMotionCriticalKind, str, tuple[float, float, float], float], ...
]:
    pivot = np.asarray(motion.axis_point, dtype=float)
    base_point = np.asarray(motion.base_plane.point, dtype=float)
    base_normal = np.asarray(motion.base_plane.normal, dtype=float)
    point_offset = float(np.dot(base_normal, base_point - pivot))
    if isinstance(surface, SphereSpec):
        coefficients = _harmonic_coefficients(
            motion,
            np.asarray(surface.center, dtype=float) - pivot,
            constant_offset=-point_offset,
        )
        return tuple(
            (
                PlaneMotionCriticalKind.SPHERE_TANGENCY,
                f"signed_distance={sign:+g}*radius",
                coefficients,
                sign * surface.radius,
            )
            for sign in (-1.0, 1.0)
        )
    if isinstance(surface, CylinderSpec):
        return (
            (
                PlaneMotionCriticalKind.CYLINDER_AXIS_PARALLEL,
                "dot(plane_normal,cylinder_axis)=0",
                _harmonic_coefficients(motion, surface.axis),
                0.0,
            ),
        )
    if isinstance(surface, ConeSpec):
        axis_coefficients = _harmonic_coefficients(motion, surface.axis)
        apex_coefficients = _harmonic_coefficients(
            motion,
            np.asarray(surface.apex, dtype=float) - pivot,
            constant_offset=-point_offset,
        )
        return (
            (
                PlaneMotionCriticalKind.CONE_PARABOLIC,
                "dot(plane_normal,cone_axis)=-sin(half_angle)",
                axis_coefficients,
                -sin(surface.half_angle),
            ),
            (
                PlaneMotionCriticalKind.CONE_PARABOLIC,
                "dot(plane_normal,cone_axis)=+sin(half_angle)",
                axis_coefficients,
                sin(surface.half_angle),
            ),
            (
                PlaneMotionCriticalKind.CONE_APEX_DEGENERACY,
                "signed_distance(cone_apex)=0",
                apex_coefficients,
                0.0,
            ),
        )
    raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")


def compute_plane_motion_schedule(
    surface: QuadricSurfaceSpec,
    motion: AxisAnglePlaneMotion,
    *,
    authored_progresses: Sequence[float] = (),
    include_interval_midpoints: bool = True,
    context: GeometryContext | ResolvedGeometryContext | None = None,
) -> PlaneMotionSchedule:
    """Insert every analytic support-family event into one motion schedule."""

    if not isinstance(surface, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    if not isinstance(motion, AxisAnglePlaneMotion):
        raise TypeError("motion must be AxisAnglePlaneMotion")
    if not isinstance(include_interval_midpoints, bool):
        raise TypeError("include_interval_midpoints must be a bool")
    if context is not None and not isinstance(
        context, (GeometryContext, ResolvedGeometryContext)
    ):
        raise TypeError("context must be a GeometryContext or ResolvedGeometryContext")

    grouped: dict[
        float, list[tuple[PlaneMotionCriticalKind, str, bool]]
    ] = {}
    for kind, equation, coefficients, target in _critical_equations(surface, motion):
        progresses, persistent = _harmonic_roots(coefficients, target, motion)
        if persistent:
            grouped.setdefault(0.0, []).append((kind, equation, True))
        else:
            for progress in progresses:
                grouped.setdefault(progress, []).append((kind, equation, False))
    if isinstance(surface, (CylinderSpec, ConeSpec)):
        for kind, equation, progress, persistent in _trim_boundary_events(
            surface,
            motion,
            context,
        ):
            grouped.setdefault(progress, []).append(
                (kind, equation, persistent)
            )

    knots = [0.0, 1.0]
    for raw in authored_progresses:
        progress = _finite(raw, "authored progress")
        if progress < 0.0 or progress > 1.0:
            raise PlaneMotionError("authored progress must lie in [0, 1]")
        knots.append(progress)
    knots.extend(grouped)
    knots.sort()
    canonical: list[float] = []
    for value in knots:
        if not canonical or value - canonical[-1] > 1.0e-12:
            canonical.append(float(value))
    if include_interval_midpoints:
        canonical = sorted(
            {
                *canonical,
                *(0.5 * (left + right) for left, right in zip(canonical, canonical[1:])),
            }
        )

    events: list[PlaneMotionCriticalEvent] = []
    for event_index, (progress, evidence) in enumerate(sorted(grouped.items())):
        kinds = tuple(sorted({item[0] for item in evidence}, key=lambda item: item.value))
        equations = tuple(sorted({item[1] for item in evidence}))
        events.append(
            PlaneMotionCriticalEvent(
                event_id=f"{motion.motion_id}:critical:{event_index:04d}",
                progress=progress,
                time=motion.time_at(progress),
                angle=motion.angle_at(progress),
                kinds=kinds,
                equations=equations,
                persistent=all(item[2] for item in evidence),
            )
        )
    samples = tuple(
        SectionAnimationSample(
            motion.time_at(progress),
            surface,
            motion.plane_at(progress),
        )
        for progress in canonical
    )
    return PlaneMotionSchedule(
        motion=motion,
        surface_id=surface.surface_id,
        progresses=tuple(canonical),
        samples=samples,
        critical_events=tuple(events),
    )


def track_scheduled_plane_section(
    section_id: str,
    surface: QuadricSurfaceSpec,
    motion: AxisAnglePlaneMotion,
    *,
    authored_progresses: Sequence[float] = (),
    include_interval_midpoints: bool = True,
    context: GeometryContext | ResolvedGeometryContext | None = None,
    coefficient_tolerance: float | None = None,
) -> ScheduledSectionAnimation:
    schedule = compute_plane_motion_schedule(
        surface,
        motion,
        authored_progresses=authored_progresses,
        include_interval_midpoints=include_interval_midpoints,
        context=context,
    )
    animation = track_quadric_section_animation(
        section_id,
        schedule.samples,
        context=context,
        coefficient_tolerance=coefficient_tolerance,
    )
    return ScheduledSectionAnimation(schedule, animation)


def canonical_plane_motion_schedule_json(schedule: PlaneMotionSchedule) -> str:
    if not isinstance(schedule, PlaneMotionSchedule):
        raise TypeError("schedule must be a PlaneMotionSchedule")
    return json.dumps(
        schedule.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "AxisAnglePlaneMotion",
    "PLANE_MOTION_SCHEDULE_SCHEMA",
    "PlaneMotionCriticalEvent",
    "PlaneMotionCriticalKind",
    "PlaneMotionError",
    "PlaneMotionSchedule",
    "ScheduledSectionAnimation",
    "canonical_plane_motion_schedule_json",
    "compute_plane_motion_schedule",
    "track_scheduled_plane_section",
]
