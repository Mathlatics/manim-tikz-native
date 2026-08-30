"""Analytic schedules for translating one mathematical plane in parallel.

The existing :mod:`.plane_motion` scheduler covers rigid axis-angle rotation.
This sibling module covers the other unambiguous primitive needed by a section
timeline: the plane frame retains its normal and in-plane axis while its
reference point moves linearly.  Critical positions are derived from the exact
support levels of the finite sphere, cylinder, or one-nappe cone; rendered
frames are never used as geometric evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from math import cos, isfinite, sin
from typing import Sequence

import numpy as np

from ..geometry import GeometryContext, ResolvedGeometryContext
from .animation import SectionAnimationSample
from .contract import ConeModel, ConeSpec, CylinderSpec, SectionPlane, SphereSpec
from .plane_motion import PlaneMotionCriticalKind
from .sections import QuadricSurfaceSpec


PARALLEL_PLANE_MOTION_SCHEDULE_SCHEMA = (
    "manim-quadric-parallel-plane-motion-schedule/v1"
)
_FLOAT_EPSILON = float(np.finfo(float).eps)
_PROGRESS_TOLERANCE = 1.0e-12


class ParallelPlaneMotionError(ValueError):
    """A parallel-plane schedule cannot be certified without guessing."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParallelPlaneMotionError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ParallelPlaneMotionError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelPlaneMotionError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise ParallelPlaneMotionError(f"{label} must be finite")
    return result


def _point3(value: object, label: str) -> tuple[float, float, float]:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelPlaneMotionError(
            f"{label} must contain three finite values"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ParallelPlaneMotionError(
            f"{label} must contain three finite values"
        )
    return tuple(float(item) for item in result)  # type: ignore[return-value]


def _progress(value: object, label: str = "progress") -> float:
    result = _finite(value, label)
    if result < 0.0 or result > 1.0:
        raise ParallelPlaneMotionError(f"{label} must lie in [0, 1]")
    return result


def _canonical_progresses(values: Sequence[float]) -> tuple[float, ...]:
    ordered: list[float] = []
    for raw in sorted(_finite(item, "schedule progress") for item in values):
        if raw < 0.0 or raw > 1.0:
            raise ParallelPlaneMotionError("schedule progress must lie in [0, 1]")
        if not ordered or raw != ordered[-1]:
            ordered.append(raw)
    if not ordered or ordered[0] != 0.0:
        ordered.insert(0, 0.0)
    if ordered[-1] != 1.0:
        ordered.append(1.0)
    return tuple(ordered)


def _with_interval_midpoints(values: Sequence[float]) -> tuple[float, ...]:
    canonical = _canonical_progresses(values)
    return _canonical_progresses(
        (
            *canonical,
            *(
                0.5 * (left + right)
                for left, right in zip(canonical, canonical[1:])
            ),
        )
    )


def _surface(value: object) -> QuadricSurfaceSpec:
    if not isinstance(value, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    if isinstance(value, ConeSpec) and value.model in {
        ConeModel.OPEN_DOUBLE,
        ConeModel.ANALYTIC_DOUBLE,
    }:
        raise ParallelPlaneMotionError(
            "parallel section timelines require one directly renderable cone nappe"
        )
    return value


@dataclass(frozen=True, slots=True)
class ParallelPlaneTranslation:
    """Linearly translate a plane frame while retaining normal and ``u_axis``."""

    motion_id: str
    base_plane: SectionPlane
    displacement: tuple[float, float, float]
    start_time: float = 0.0
    end_time: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "motion_id", _identity(self.motion_id, "motion_id"))
        if not isinstance(self.base_plane, SectionPlane):
            raise TypeError("base_plane must be a SectionPlane")
        object.__setattr__(
            self,
            "displacement",
            _point3(self.displacement, "displacement"),
        )
        start = _finite(self.start_time, "start_time")
        end = _finite(self.end_time, "end_time")
        if end <= start:
            raise ParallelPlaneMotionError("motion times must increase")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)

    def time_at(self, progress: float) -> float:
        value = _progress(progress, "motion progress")
        return self.start_time + value * (self.end_time - self.start_time)

    def plane_at(self, progress: float) -> SectionPlane:
        value = _progress(progress, "motion progress")
        if value == 0.0:
            return self.base_plane
        point = np.asarray(self.base_plane.point, dtype=float) + value * np.asarray(
            self.displacement,
            dtype=float,
        )
        return SectionPlane(
            self.base_plane.plane_id,
            tuple(float(item) for item in point),
            self.base_plane.normal,
            u_axis=self.base_plane.u_axis,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "motionId": self.motion_id,
            "planeId": self.base_plane.plane_id,
            "basePlane": {
                "point": list(self.base_plane.point),
                "normal": list(self.base_plane.normal),
                "uAxis": list(self.base_plane.u_axis),
            },
            "displacement": list(self.displacement),
            "startTime": self.start_time,
            "endTime": self.end_time,
        }


@dataclass(frozen=True, slots=True)
class ParallelPlaneMotionCriticalEvent:
    event_id: str
    progress: float
    time: float
    kinds: tuple[PlaneMotionCriticalKind, ...]
    equations: tuple[str, ...]
    normal_offsets: tuple[float, ...] = ()
    persistent: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identity(self.event_id, "event_id"))
        progress = _progress(self.progress, "critical progress")
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "time", _finite(self.time, "critical time"))
        kinds = tuple(sorted(set(self.kinds), key=lambda item: item.value))
        if not kinds or kinds != self.kinds:
            raise ParallelPlaneMotionError(
                "critical kinds must be unique and canonical"
            )
        equations = tuple(
            sorted(set(_identity(item, "equation") for item in self.equations))
        )
        if not equations or equations != self.equations:
            raise ParallelPlaneMotionError(
                "critical equations must be unique and canonical"
            )
        offsets = tuple(
            sorted(
                set(_finite(item, "normal offset") for item in self.normal_offsets)
            )
        )
        if offsets != self.normal_offsets:
            raise ParallelPlaneMotionError(
                "normal_offsets must be unique and canonical"
            )
        if not isinstance(self.persistent, bool):
            raise TypeError("persistent must be a bool")
        if self.persistent and progress != 0.0:
            raise ParallelPlaneMotionError(
                "persistent critical evidence must use progress zero"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "eventId": self.event_id,
            "progress": self.progress,
            "time": self.time,
            "kinds": [item.value for item in self.kinds],
            "equations": list(self.equations),
            "normalOffsets": list(self.normal_offsets),
            "persistent": self.persistent,
        }


@dataclass(frozen=True, slots=True)
class ParallelPlaneMotionSchedule:
    motion: ParallelPlaneTranslation
    surface_id: str
    progresses: tuple[float, ...]
    samples: tuple[SectionAnimationSample, ...]
    critical_events: tuple[ParallelPlaneMotionCriticalEvent, ...]
    schema: str = PARALLEL_PLANE_MOTION_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PARALLEL_PLANE_MOTION_SCHEDULE_SCHEMA:
            raise ParallelPlaneMotionError(
                "invalid parallel-plane-motion schedule schema"
            )
        if not isinstance(self.motion, ParallelPlaneTranslation):
            raise TypeError("motion must be a ParallelPlaneTranslation")
        surface_id = _identity(self.surface_id, "surface_id")
        if (
            not self.progresses
            or self.progresses[0] != 0.0
            or self.progresses[-1] != 1.0
            or any(
                right <= left
                for left, right in zip(self.progresses, self.progresses[1:])
            )
        ):
            raise ParallelPlaneMotionError(
                "schedule progresses must increase strictly from 0 to 1"
            )
        if len(self.samples) != len(self.progresses):
            raise ParallelPlaneMotionError(
                "schedule samples must cover every progress"
            )
        for progress, sample in zip(self.progresses, self.samples):
            if not isinstance(sample, SectionAnimationSample):
                raise TypeError(
                    "schedule samples must contain SectionAnimationSample objects"
                )
            if sample.surface.surface_id != surface_id:
                raise ParallelPlaneMotionError("sample surface identity changed")
            if sample.time != self.motion.time_at(progress):
                raise ParallelPlaneMotionError(
                    "sample time disagrees with motion progress"
                )
            if sample.plane != self.motion.plane_at(progress):
                raise ParallelPlaneMotionError(
                    "sample plane disagrees with motion progress"
                )
        event_keys = tuple(
            (item.progress, item.persistent, item.event_id)
            for item in self.critical_events
        )
        if event_keys != tuple(sorted(event_keys)):
            raise ParallelPlaneMotionError(
                "critical events must use canonical order"
            )
        event_ids: set[str] = set()
        for event in self.critical_events:
            if not isinstance(event, ParallelPlaneMotionCriticalEvent):
                raise TypeError(
                    "critical_events must contain "
                    "ParallelPlaneMotionCriticalEvent objects"
                )
            if event.event_id in event_ids:
                raise ParallelPlaneMotionError(
                    "critical event identities must be unique"
                )
            event_ids.add(event.event_id)
            if event.time != self.motion.time_at(event.progress):
                raise ParallelPlaneMotionError(
                    "critical event time disagrees with motion progress"
                )
            if not event.persistent and event.progress not in self.progresses:
                raise ParallelPlaneMotionError(
                    "every non-persistent critical event must be sampled"
                )
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
class _CriticalLevelEvidence:
    kind: PlaneMotionCriticalKind
    equation: str
    offset: float
    roundoff: float
    local_offset: float
    local_roundoff: float
    exact_local_offset: Fraction
    structural_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PlaneMotionCriticalKind):
            raise TypeError("kind must be a PlaneMotionCriticalKind")
        _identity(self.equation, "equation")
        if (
            not isfinite(self.offset)
            or not isfinite(self.roundoff)
            or not isfinite(self.local_offset)
            or not isfinite(self.local_roundoff)
        ):
            raise ParallelPlaneMotionError(
                "critical level evidence must remain finite"
            )
        if self.roundoff < 0.0 or self.local_roundoff < 0.0:
            raise ParallelPlaneMotionError(
                "critical level roundoff must be non-negative"
            )
        if self.structural_key is not None:
            _identity(self.structural_key, "structural_key")
        if not isinstance(self.exact_local_offset, Fraction):
            raise TypeError("exact_local_offset must be a Fraction")


def _relative_dot(
    normal: np.ndarray,
    point: np.ndarray,
    base_point: np.ndarray,
) -> tuple[float, float]:
    difference = point - base_point
    products = normal * difference
    value = float(np.sum(products))
    subtraction_envelope = float(
        np.sum(np.abs(normal) * (np.abs(point) + np.abs(base_point)))
    )
    product_envelope = float(np.sum(np.abs(products)))
    roundoff = (
        32.0
        * _FLOAT_EPSILON
        * max(subtraction_envelope + product_envelope, 1.0e-300)
    )
    return value, roundoff


def _canonical_direction(value: np.ndarray) -> tuple[float, float, float]:
    length = float(np.linalg.norm(value))
    if not isfinite(length) or length <= 0.0:
        raise ParallelPlaneMotionError(
            "structural direction must be finite and non-zero"
        )
    return tuple(float(item) for item in value / length)  # type: ignore[return-value]


def _exact_dot(first: np.ndarray, second: np.ndarray) -> Fraction:
    return sum(
        (
            Fraction.from_float(float(left))
            * Fraction.from_float(float(right))
            for left, right in zip(first, second)
        ),
        Fraction(0),
    )


def _cone_parabolic_sign(surface: ConeSpec, normal: np.ndarray) -> int | None:
    axis = np.asarray(surface.axis, dtype=float)
    normal_identity = tuple(float(item) for item in normal)
    for sign in (-1, 1):
        for radial in (surface.frame.x_axis, surface.frame.y_axis):
            candidate = (
                cos(surface.half_angle) * np.asarray(radial, dtype=float)
                + sign * sin(surface.half_angle) * axis
            )
            if normal_identity == _canonical_direction(candidate):
                return sign
            if normal_identity == _canonical_direction(-candidate):
                return -sign
    products = normal * axis
    alignment = float(np.sum(products))
    exact_alignment = _exact_dot(normal, axis)
    target = sin(surface.half_angle)
    tolerance = (
        64.0
        * _FLOAT_EPSILON
        * max(float(np.sum(np.abs(products))), abs(target), 1.0e-300)
    )
    for sign in (-1, 1):
        delta = alignment - sign * target
        if exact_alignment == sign * Fraction.from_float(target):
            return sign
        if abs(delta) <= tolerance:
            raise ParallelPlaneMotionError(
                "cone parabolic orientation is numerically ambiguous"
            )
    return None


def _cylinder_axis_parallel(
    surface: CylinderSpec,
    normal: np.ndarray,
) -> bool:
    normal_identity = tuple(float(item) for item in normal)
    for radial in (surface.frame.x_axis, surface.frame.y_axis):
        candidate = np.asarray(radial, dtype=float)
        if normal_identity in {
            _canonical_direction(candidate),
            _canonical_direction(-candidate),
        }:
            return True
    products = normal * np.asarray(surface.axis, dtype=float)
    exact_alignment = _exact_dot(
        normal,
        np.asarray(surface.axis, dtype=float),
    )
    alignment = float(np.sum(products))
    if exact_alignment == 0:
        return True
    tolerance = (
        64.0
        * _FLOAT_EPSILON
        * max(float(np.sum(np.abs(products))), 1.0e-300)
    )
    if abs(alignment) <= tolerance:
        raise ParallelPlaneMotionError(
            "cylinder axis-parallel orientation is numerically ambiguous"
        )
    return False


def _circle_level_evidence(
    surface: CylinderSpec | ConeSpec,
    normal: np.ndarray,
    base_point: np.ndarray,
) -> tuple[_CriticalLevelEvidence, ...]:
    axis = np.asarray(surface.axis, dtype=float)
    raw_alignment = float(np.dot(normal, axis))
    origin = np.asarray(
        surface.origin if isinstance(surface, CylinderSpec) else surface.apex,
        dtype=float,
    )
    kind = (
        PlaneMotionCriticalKind.CYLINDER_TRIM_TANGENCY
        if isinstance(surface, CylinderSpec)
        else PlaneMotionCriticalKind.CONE_TRIM_TANGENCY
    )
    parabolic_sign = (
        _cone_parabolic_sign(surface, normal)
        if isinstance(surface, ConeSpec)
        else None
    )
    axis_parallel = (
        _cylinder_axis_parallel(surface, normal)
        if isinstance(surface, CylinderSpec)
        else False
    )
    if axis_parallel:
        alignment = 0.0
        radial_length = 1.0
    elif parabolic_sign is not None:
        alignment = parabolic_sign * sin(surface.half_angle)
        radial_length = cos(surface.half_angle)
    else:
        alignment = raw_alignment
        radial_length = float(
            np.linalg.norm(np.cross(normal, axis))
        )
    origin_offset, origin_roundoff = _relative_dot(normal, origin, base_point)
    apex_offset: float | None = None
    apex_roundoff = 0.0
    if isinstance(surface, ConeSpec):
        apex_offset, apex_roundoff = _relative_dot(
            normal,
            np.asarray(surface.apex, dtype=float),
            base_point,
        )
    evidence: list[_CriticalLevelEvidence] = []
    for axial, role in zip(surface.axial_range, ("min", "max")):
        radius = (
            surface.radius
            if isinstance(surface, CylinderSpec)
            else abs(axial) * surface.slope
        )
        if radius == 0.0:
            continue
        middle_local = axial * alignment
        middle_local_roundoff = (
            32.0
            * _FLOAT_EPSILON
            * max(abs(axial * alignment), abs(axial * raw_alignment), 1.0e-300)
        )
        middle = origin_offset + middle_local
        middle_roundoff = origin_roundoff + middle_local_roundoff
        radius_term = radius * radial_length
        radius_roundoff = (
            32.0
            * _FLOAT_EPSILON
            * max(abs(radius_term), abs(radius), 1.0e-300)
        )
        for sign in (-1.0, 1.0):
            structural_key: str | None = None
            offset = middle + sign * radius_term
            roundoff = middle_roundoff + radius_roundoff
            local_offset = middle_local + sign * radius_term
            local_roundoff = middle_local_roundoff + radius_roundoff
            exact_local_offset = (
                Fraction.from_float(float(axial))
                * Fraction.from_float(alignment)
                + Fraction.from_float(sign)
                * Fraction.from_float(radius)
                * Fraction.from_float(radial_length)
            )
            if (
                isinstance(surface, ConeSpec)
                and parabolic_sign is not None
                and axial != 0.0
                and int(sign) == -parabolic_sign * (1 if axial > 0.0 else -1)
            ):
                assert apex_offset is not None
                offset = apex_offset
                roundoff = max(roundoff, apex_roundoff)
                local_offset = 0.0
                local_roundoff = 0.0
                exact_local_offset = Fraction(0)
                structural_key = f"{surface.surface_id}:apex-generator-level"
            elif isinstance(surface, CylinderSpec) and axis_parallel:
                structural_key = (
                    f"{surface.surface_id}:axis-parallel-generator:{int(sign):+d}"
                )
            elif radial_length == 0.0:
                structural_key = (
                    f"{surface.surface_id}:trim:{role}:normal-parallel"
                )
            evidence.append(
                _CriticalLevelEvidence(
                    kind=kind,
                    equation=(
                        f"{surface.surface_id}:trim_{role}:signed_distance="
                        f"{sign:+g}*radius"
                    ),
                    offset=offset,
                    roundoff=roundoff,
                    local_offset=local_offset,
                    local_roundoff=local_roundoff,
                    exact_local_offset=exact_local_offset,
                    structural_key=structural_key,
                )
            )
    return tuple(evidence)


def _critical_level_evidence(
    surface: QuadricSurfaceSpec,
    motion: ParallelPlaneTranslation,
) -> tuple[_CriticalLevelEvidence, ...]:
    normal = np.asarray(motion.base_plane.normal, dtype=float)
    base_point = np.asarray(motion.base_plane.point, dtype=float)
    if isinstance(surface, SphereSpec):
        center_offset, center_roundoff = _relative_dot(
            normal,
            np.asarray(surface.center, dtype=float),
            base_point,
        )
        return tuple(
            _CriticalLevelEvidence(
                kind=PlaneMotionCriticalKind.SPHERE_TANGENCY,
                equation=(
                    f"{surface.surface_id}:signed_distance={sign:+g}*radius"
                ),
                offset=center_offset + sign * surface.radius,
                roundoff=(
                    center_roundoff
                    + 16.0 * _FLOAT_EPSILON * surface.radius
                ),
                local_offset=sign * surface.radius,
                local_roundoff=0.0,
                exact_local_offset=(
                    Fraction.from_float(sign)
                    * Fraction.from_float(surface.radius)
                ),
            )
            for sign in (-1.0, 1.0)
        )
    result = list(_circle_level_evidence(surface, normal, base_point))
    if isinstance(surface, ConeSpec):
        apex_offset, apex_roundoff = _relative_dot(
            normal,
            np.asarray(surface.apex, dtype=float),
            base_point,
        )
        result.append(
            _CriticalLevelEvidence(
                kind=PlaneMotionCriticalKind.CONE_APEX_DEGENERACY,
                equation=f"{surface.surface_id}:signed_distance(cone_apex)=0",
                offset=apex_offset,
                roundoff=apex_roundoff,
                local_offset=0.0,
                local_roundoff=0.0,
                exact_local_offset=Fraction(0),
                structural_key=(
                    f"{surface.surface_id}:apex-generator-level"
                    if _cone_parabolic_sign(surface, normal) is not None
                    else None
                ),
            )
        )
    return tuple(result)


def _levels_certifiably_equal(
    first: _CriticalLevelEvidence,
    second: _CriticalLevelEvidence,
) -> bool:
    return bool(
        (
            first.structural_key is not None
            and first.structural_key == second.structural_key
        )
        or first.exact_local_offset == second.exact_local_offset
    )


def _persistent_orientation_evidence(
    surface: QuadricSurfaceSpec,
    motion: ParallelPlaneTranslation,
) -> tuple[tuple[PlaneMotionCriticalKind, str], ...]:
    normal = np.asarray(motion.base_plane.normal, dtype=float)
    if isinstance(surface, SphereSpec):
        return ()
    alignment = float(np.dot(normal, np.asarray(surface.axis, dtype=float)))
    if isinstance(surface, CylinderSpec):
        return (
            (
                PlaneMotionCriticalKind.CYLINDER_AXIS_PARALLEL,
                "dot(plane_normal,cylinder_axis)=0",
            ),
        ) if _cylinder_axis_parallel(surface, normal) else ()
    parabolic_sign = _cone_parabolic_sign(surface, normal)
    if parabolic_sign is None:
        return ()
    target = sin(surface.half_angle)
    return (
        (
            PlaneMotionCriticalKind.CONE_PARABOLIC,
            "dot(plane_normal,cone_axis)="
            f"{parabolic_sign * target:+.17g}",
        ),
    )


def compute_parallel_plane_motion_schedule(
    surface: QuadricSurfaceSpec,
    motion: ParallelPlaneTranslation,
    *,
    authored_progresses: Sequence[float] = (),
    include_interval_midpoints: bool = True,
    context: GeometryContext | ResolvedGeometryContext | None = None,
) -> ParallelPlaneMotionSchedule:
    """Insert every analytic finite-surface event into one translation path."""

    selected_surface = _surface(surface)
    if not isinstance(motion, ParallelPlaneTranslation):
        raise TypeError("motion must be a ParallelPlaneTranslation")
    if not isinstance(include_interval_midpoints, bool):
        raise TypeError("include_interval_midpoints must be a bool")
    if context is not None and not isinstance(
        context,
        (GeometryContext, ResolvedGeometryContext),
    ):
        raise TypeError(
            "context must be a GeometryContext or ResolvedGeometryContext"
        )

    normal = np.asarray(motion.base_plane.normal, dtype=float)
    displacement = np.asarray(motion.displacement, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        normal_delta = _finite(
            float(np.dot(normal, displacement)),
            "translation normal displacement",
        )
        dot_envelope = _finite(
            float(np.dot(np.abs(normal), np.abs(displacement))),
            "translation dot-product envelope",
        )
    unresolved = 128.0 * _FLOAT_EPSILON * max(dot_envelope, 1.0e-300)
    if normal_delta != 0.0 and abs(normal_delta) <= unresolved:
        raise ParallelPlaneMotionError(
            "translation normal displacement is below numeric resolution"
        )

    raw_events: list[
        tuple[float, bool, PlaneMotionCriticalKind, str, float | None]
    ] = []
    for kind, equation in _persistent_orientation_evidence(
        selected_surface,
        motion,
    ):
        raw_events.append((0.0, True, kind, equation, None))

    level_evidence = _critical_level_evidence(selected_surface, motion)
    evidence_by_equation = {
        evidence.equation: evidence for evidence in level_evidence
    }
    if normal_delta == 0.0:
        for evidence in level_evidence:
            if evidence.offset == 0.0:
                raw_events.append(
                    (
                        0.0,
                        True,
                        evidence.kind,
                        evidence.equation,
                        evidence.offset,
                    )
                )
            elif abs(evidence.offset) <= evidence.roundoff:
                raise ParallelPlaneMotionError(
                    "persistent critical height is indistinguishable from "
                    "floating-point roundoff"
                )
    else:
        for evidence in level_evidence:
            progress = evidence.offset / normal_delta
            if 0.0 <= progress <= 1.0:
                raw_events.append(
                    (
                        float(progress),
                        False,
                        evidence.kind,
                        evidence.equation,
                        evidence.offset,
                    )
                )
            elif (
                -_PROGRESS_TOLERANCE <= progress < 0.0
                or 1.0 < progress <= 1.0 + _PROGRESS_TOLERANCE
            ):
                raise ParallelPlaneMotionError(
                    "critical progress is numerically ambiguous at a motion endpoint"
                )

    raw_events.sort(key=lambda item: (item[0], item[1], item[2].value, item[3]))
    clusters: list[
        list[
            tuple[
                float,
                bool,
                PlaneMotionCriticalKind,
                str,
                float | None,
            ]
        ]
    ] = []
    for item in raw_events:
        if (
            not clusters
            or item[1] != clusters[-1][0][1]
            or abs(item[0] - clusters[-1][0][0]) > _PROGRESS_TOLERANCE
        ):
            clusters.append([item])
        else:
            existing_offsets = tuple(
                float(member[4])
                for member in clusters[-1]
                if member[4] is not None
            )
            if item[4] is not None and existing_offsets:
                source_evidence = evidence_by_equation[item[3]]
                same_level = any(
                    _levels_certifiably_equal(
                        source_evidence,
                        evidence_by_equation[member[3]],
                    )
                    for member in clusters[-1]
                    if member[4] is not None
                )
                if not same_level:
                    raise ParallelPlaneMotionError(
                        "distinct critical levels collapse below timeline progress "
                        "resolution, or their equality is uncertified"
                    )
            clusters[-1].append(item)

    events = tuple(
        ParallelPlaneMotionCriticalEvent(
            event_id=f"{motion.motion_id}:critical:{index:04d}",
            progress=cluster[0][0],
            time=motion.time_at(cluster[0][0]),
            kinds=tuple(
                sorted(
                    {item[2] for item in cluster},
                    key=lambda item: item.value,
                )
            ),
            equations=tuple(sorted({item[3] for item in cluster})),
            normal_offsets=tuple(
                sorted({float(item[4]) for item in cluster if item[4] is not None})
            ),
            persistent=cluster[0][1],
        )
        for index, cluster in enumerate(clusters)
    )

    authored = tuple(
        _progress(item, "authored progress") for item in authored_progresses
    )
    knots = _canonical_progresses(
        (
            0.0,
            1.0,
            *authored,
            *(event.progress for event in events if not event.persistent),
        )
    )
    progresses = (
        _with_interval_midpoints(knots)
        if include_interval_midpoints
        else knots
    )
    samples = tuple(
        SectionAnimationSample(
            motion.time_at(progress),
            selected_surface,
            motion.plane_at(progress),
        )
        for progress in progresses
    )
    return ParallelPlaneMotionSchedule(
        motion=motion,
        surface_id=selected_surface.surface_id,
        progresses=progresses,
        samples=samples,
        critical_events=events,
    )


def canonical_parallel_plane_motion_schedule_json(
    schedule: ParallelPlaneMotionSchedule,
) -> str:
    if not isinstance(schedule, ParallelPlaneMotionSchedule):
        raise TypeError("schedule must be a ParallelPlaneMotionSchedule")
    return json.dumps(
        schedule.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "PARALLEL_PLANE_MOTION_SCHEDULE_SCHEMA",
    "ParallelPlaneMotionCriticalEvent",
    "ParallelPlaneMotionError",
    "ParallelPlaneMotionSchedule",
    "ParallelPlaneTranslation",
    "canonical_parallel_plane_motion_schedule_json",
    "compute_parallel_plane_motion_schedule",
]
