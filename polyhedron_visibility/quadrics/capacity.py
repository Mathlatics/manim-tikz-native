"""Deterministic fixed-capacity planning for finite quadric Manim scenes.

The planner drives an existing fixed-capacity controller over an explicit set
of normalized progress values and inspects the already prepared numeric frame.
It does not implement geometry, visibility, dashing, or painter ordering.  A
plan therefore certifies the listed progress values only; it never promotes a
finite scan into a claim about every point of a continuous interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, isfinite
from typing import Callable, Sequence

import numpy as np

from .manim import (
    QuadricManimLimits,
    QuadricOcclusion3D,
    estimate_quadric_mobject_count,
)
from .plane_motion import (
    ScheduledSectionAnimation,
    canonical_plane_motion_schedule_json,
)
from .profiles import QuadricRenderProfile


QUADRIC_CAPACITY_PLAN_SCHEMA = "manim-quadric-capacity-plan/v1"
_PROGRESS_TOLERANCE = 1.0e-12


class QuadricCapacityPlanningError(RuntimeError):
    """A capacity scan could not produce complete, truthful evidence."""


def _finite_progress(value: object, label: str = "progress") -> float:
    if isinstance(value, bool):
        raise QuadricCapacityPlanningError(
            f"{label} must be finite and lie in [0, 1]"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricCapacityPlanningError(
            f"{label} must be finite and lie in [0, 1]"
        ) from exc
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise QuadricCapacityPlanningError(
            f"{label} must be finite and lie in [0, 1]"
        )
    return result


def _canonical_progresses(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(_finite_progress(value) for value in values)
    result: list[float] = []
    for value in ordered:
        if not result or value - result[-1] > _PROGRESS_TOLERANCE:
            result.append(value)
    if not result:
        raise QuadricCapacityPlanningError(
            "capacity planning requires at least one progress value"
        )
    return tuple(result)


def _contains_progress(values: Sequence[float], target: float) -> bool:
    return any(abs(value - target) <= _PROGRESS_TOLERANCE for value in values)


def scheduled_capacity_progresses(
    scheduled: ScheduledSectionAnimation,
    *,
    frame_rate: float,
    rate_function: Callable[[float], float] | None = None,
) -> tuple[float, ...]:
    """Return a deterministic frame grid plus every analytic schedule knot.

    The grid includes both motion endpoints.  ``rate_function`` should match
    the Manim animation's normalized easing function when it is not linear.
    Schedule knots are added even when they fall between output frames so
    tangencies and topology events are always planned explicitly.
    """

    if not isinstance(scheduled, ScheduledSectionAnimation):
        raise TypeError("scheduled must be a ScheduledSectionAnimation")
    if isinstance(frame_rate, bool):
        raise QuadricCapacityPlanningError("frame_rate must be finite and positive")
    try:
        fps = float(frame_rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricCapacityPlanningError(
            "frame_rate must be finite and positive"
        ) from exc
    if not isfinite(fps) or fps <= 0.0:
        raise QuadricCapacityPlanningError("frame_rate must be finite and positive")
    motion = scheduled.schedule.motion
    duration = motion.end_time - motion.start_time
    interval_count = max(1, int(ceil(duration * fps)))
    easing = (lambda value: value) if rate_function is None else rate_function
    grid: list[float] = []
    for index in range(interval_count + 1):
        raw = index / interval_count
        try:
            eased = easing(raw)
        except Exception as exc:
            raise QuadricCapacityPlanningError(
                f"rate_function failed at normalized time {raw:.12g}"
            ) from exc
        grid.append(_finite_progress(eased, "rate_function output"))
    return _canonical_progresses(
        (*grid, *scheduled.schedule.progresses)
    )


@dataclass(frozen=True, slots=True)
class QuadricCapacityHeadroom:
    """Small explicit margins added to observed numeric peaks."""

    fragment_slots: int = 2
    segment_slots: int = 8
    surface_segment_slots: int = 8
    dash_slots: int = 2
    projected_length_scale: float = 1.05
    minimum_projected_length: float = 1.0
    mobject_slots: int = 0

    def __post_init__(self) -> None:
        for name in (
            "fragment_slots",
            "segment_slots",
            "surface_segment_slots",
            "dash_slots",
            "mobject_slots",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("projected_length_scale", "minimum_projected_length"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite and positive")
            numeric = float(value)
            if not isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, numeric)
        if self.projected_length_scale < 1.0:
            raise ValueError("projected_length_scale must not be smaller than 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "fragmentSlots": self.fragment_slots,
            "segmentSlots": self.segment_slots,
            "surfaceSegmentSlots": self.surface_segment_slots,
            "dashSlots": self.dash_slots,
            "projectedLengthScale": self.projected_length_scale,
            "minimumProjectedLength": self.minimum_projected_length,
            "mobjectSlots": self.mobject_slots,
        }


@dataclass(frozen=True, slots=True)
class QuadricCapacitySample:
    """Observed numeric demand at one committed normalized progress."""

    progress: float
    surface_count: int
    active_curve_count: int
    active_source_count: int
    active_fragment_count: int
    max_fragments_per_source: int
    max_segments_per_fragment: int
    max_projected_source_segments: int
    max_surface_segments: int
    max_dashes_per_fragment: int
    max_projected_length: float
    plane_fragment_count: int
    ray_classification_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "progress": self.progress,
            "surfaceCount": self.surface_count,
            "activeCurveCount": self.active_curve_count,
            "activeSourceCount": self.active_source_count,
            "activeFragmentCount": self.active_fragment_count,
            "maxFragmentsPerSource": self.max_fragments_per_source,
            "maxSegmentsPerFragment": self.max_segments_per_fragment,
            "maxProjectedSourceSegments": self.max_projected_source_segments,
            "maxSurfaceSegments": self.max_surface_segments,
            "maxDashesPerFragment": self.max_dashes_per_fragment,
            "maxProjectedLength": self.max_projected_length,
            "planeFragmentCount": self.plane_fragment_count,
            "rayClassificationCount": self.ray_classification_count,
        }


@dataclass(frozen=True, slots=True)
class QuadricCapacityPeaks:
    """Componentwise maxima across every listed sample."""

    surface_count: int
    active_curve_count: int
    active_source_count: int
    active_fragment_count: int
    max_fragments_per_source: int
    max_segments_per_fragment: int
    max_projected_source_segments: int
    max_surface_segments: int
    max_dashes_per_fragment: int
    max_projected_length: float
    plane_fragment_count: int
    ray_classification_count: int

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[QuadricCapacitySample],
    ) -> "QuadricCapacityPeaks":
        if not samples:
            raise QuadricCapacityPlanningError("capacity plan has no samples")
        fields = (
            "surface_count",
            "active_curve_count",
            "active_source_count",
            "active_fragment_count",
            "max_fragments_per_source",
            "max_segments_per_fragment",
            "max_projected_source_segments",
            "max_surface_segments",
            "max_dashes_per_fragment",
            "max_projected_length",
            "plane_fragment_count",
            "ray_classification_count",
        )
        return cls(
            **{
                name: max(getattr(sample, name) for sample in samples)
                for name in fields
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "surfaceCount": self.surface_count,
            "activeCurveCount": self.active_curve_count,
            "activeSourceCount": self.active_source_count,
            "activeFragmentCount": self.active_fragment_count,
            "maxFragmentsPerSource": self.max_fragments_per_source,
            "maxSegmentsPerFragment": self.max_segments_per_fragment,
            "maxProjectedSourceSegments": self.max_projected_source_segments,
            "maxSurfaceSegments": self.max_surface_segments,
            "maxDashesPerFragment": self.max_dashes_per_fragment,
            "maxProjectedLength": self.max_projected_length,
            "planeFragmentCount": self.plane_fragment_count,
            "rayClassificationCount": self.ray_classification_count,
        }


@dataclass(frozen=True, slots=True)
class QuadricCapacityPlan:
    """Deterministic evidence and one compact fixed-capacity recommendation."""

    coverage: str
    samples: tuple[QuadricCapacitySample, ...]
    required_progresses: tuple[float, ...]
    peaks: QuadricCapacityPeaks
    fixed_surface_count: int
    allocated_curve_count: int
    allocated_boundary_source_count: int
    slot_source_count: int
    boundary_style_count: int
    section_enabled: bool
    source_identity_digest: str
    recommended_limits: QuadricManimLimits
    estimated_mobject_total: int
    headroom: QuadricCapacityHeadroom
    schedule_digest: str | None = None
    profile_id: str | None = None
    frame_rate: float | None = None
    continuous_interval_certified: bool = False
    schema: str = QUADRIC_CAPACITY_PLAN_SCHEMA

    def to_dict(self) -> dict[str, object]:
        limits = self.recommended_limits
        return {
            "schema": self.schema,
            "coverage": self.coverage,
            "continuousIntervalCertified": self.continuous_interval_certified,
            "profileId": self.profile_id,
            "frameRate": self.frame_rate,
            "scheduleDigest": self.schedule_digest,
            "sourceIdentityDigest": self.source_identity_digest,
            "requiredProgresses": list(self.required_progresses),
            "sampleCount": len(self.samples),
            "samples": [sample.to_dict() for sample in self.samples],
            "peaks": self.peaks.to_dict(),
            "fixedAllocation": {
                "surfaceCount": self.fixed_surface_count,
                "allocatedCurveCount": self.allocated_curve_count,
                "allocatedBoundarySourceCount": (
                    self.allocated_boundary_source_count
                ),
                "slotSourceCount": self.slot_source_count,
                "boundaryStyleCount": self.boundary_style_count,
                "sectionEnabled": self.section_enabled,
            },
            "headroom": self.headroom.to_dict(),
            "estimatedMobjectTotal": self.estimated_mobject_total,
            "recommendedLimits": {
                name: getattr(limits, name)
                for name in QuadricManimLimits.__dataclass_fields__
            },
        }


def canonical_quadric_capacity_plan_json(plan: QuadricCapacityPlan) -> str:
    if not isinstance(plan, QuadricCapacityPlan):
        raise TypeError("plan must be a QuadricCapacityPlan")
    return json.dumps(
        plan.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _polyline_length(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)))


def _cone_fill_paths(fill: object | None) -> tuple[np.ndarray, ...]:
    if fill is None:
        return ()
    names = (
        "opaque_lateral_paths",
        "opaque_cap_paths",
        "back_lateral_paths",
        "back_cap_paths",
        "front_lateral_paths",
        "front_cap_paths",
    )
    return tuple(
        np.asarray(path, dtype=float)
        for name in names
        for path in getattr(fill, name)
    )


def _capacity_sample(
    controller: QuadricOcclusion3D,
    progress: float,
) -> QuadricCapacitySample:
    prepared = controller._last_prepared_frame
    if prepared is None:
        raise QuadricCapacityPlanningError(
            "controller has no committed prepared frame to inspect"
        )
    numeric = prepared.numeric
    groups = (
        numeric.fragments
        if numeric.boundary_fragments is None
        else numeric.boundary_fragments
    )
    fragments = tuple(
        fragment for values in groups.values() for fragment in values
    )
    surface_paths: list[np.ndarray] = []
    for surface in numeric.surfaces:
        surface_paths.append(np.asarray(surface.points, dtype=float))
        surface_paths.extend(_cone_fill_paths(surface.cone_fill))
    if numeric.section_layers is not None:
        surface_paths.append(
            np.asarray(numeric.section_layers.surface_points, dtype=float)
        )
        surface_paths.extend(_cone_fill_paths(numeric.section_layers.cone_fill))
    section = numeric.section_layers
    boundary = numeric.boundary_frame
    active_source_count = (
        len(groups) if boundary is None else len(boundary.sources)
    )
    return QuadricCapacitySample(
        progress=progress,
        surface_count=len(numeric.frame.surface_items),
        active_curve_count=len(numeric.curve_opacities),
        active_source_count=active_source_count,
        active_fragment_count=len(fragments),
        max_fragments_per_source=max(
            (len(values) for values in groups.values()),
            default=0,
        ),
        max_segments_per_fragment=max(
            (max(0, len(fragment.points) - 1) for fragment in fragments),
            default=0,
        ),
        max_projected_source_segments=max(
            numeric.projected_source_segment_counts.values(),
            default=0,
        ),
        max_surface_segments=max(
            (max(0, len(path) - 1) for path in surface_paths),
            default=0,
        ),
        max_dashes_per_fragment=max(
            (len(fragment.dashes) for fragment in fragments),
            default=0,
        ),
        max_projected_length=max(
            (_polyline_length(fragment.points) for fragment in fragments),
            default=0.0,
        ),
        plane_fragment_count=(
            0 if section is None else len(section.frame.plane_fragments)
        ),
        ray_classification_count=(
            0 if section is None else section.frame.ray_classification_count
        ),
    )


def _resolve_low_level_controller(value: object) -> QuadricOcclusion3D:
    current = value
    visited: set[int] = set()
    while not isinstance(current, QuadricOcclusion3D):
        identity = id(current)
        if identity in visited:
            break
        visited.add(identity)
        current = getattr(current, "controller", None)
        if current is None:
            break
    if not isinstance(current, QuadricOcclusion3D):
        raise QuadricCapacityPlanningError(
            "capacity planning currently requires QuadricOcclusion3D or a "
            "single-surface facade that exposes it as controller"
        )
    return current


def _recommended_limits(
    controller: QuadricOcclusion3D,
    peaks: QuadricCapacityPeaks,
    headroom: QuadricCapacityHeadroom,
) -> tuple[QuadricManimLimits, int]:
    fragment_capacity = max(
        1,
        peaks.max_fragments_per_source + headroom.fragment_slots,
    )
    segment_capacity = max(
        1,
        peaks.max_projected_source_segments + headroom.segment_slots,
    )
    surface_segment_capacity = max(
        1,
        peaks.max_surface_segments + headroom.surface_segment_slots,
    )
    dash_capacity = max(
        1,
        peaks.max_dashes_per_fragment + headroom.dash_slots,
    )
    projected_length = max(
        headroom.minimum_projected_length,
        peaks.max_projected_length * headroom.projected_length_scale,
    )
    estimated = estimate_quadric_mobject_count(
        surface_count=len(controller._surface_ids),
        source_count=len(controller._slot_source_ids),
        max_fragments_per_curve=fragment_capacity,
        section_enabled=controller._section_enabled,
    )
    maximum_mobjects = estimated + headroom.mobject_slots
    return (
        QuadricManimLimits(
            max_surfaces=max(1, len(controller._surface_ids)),
            max_curves=max(1, len(controller._curve_ids)),
            max_fragments_per_curve=fragment_capacity,
            max_segments_per_fragment=segment_capacity,
            max_surface_segments=surface_segment_capacity,
            max_dashes_per_fragment=dash_capacity,
            max_projected_length=projected_length,
            max_total_mobjects=maximum_mobjects,
            max_boundary_sources=max(1, len(controller._boundary_source_ids)),
            max_boundary_styles=max(1, len(controller.boundary_styles)),
        ),
        estimated,
    )


class QuadricCapacityPlanner:
    """Scan a controller without rendering pixels and generate tight limits.

    ``progress`` must expose ``get_value()`` and ``set_value(value)``.  The
    planner preserves its original value, fixed slot identities, and Scene
    ownership.  A controller supplied unattached is attached only for the scan
    and restored before the method returns.
    """

    def __init__(self, controller: object, *, progress: object) -> None:
        self.controller = controller
        self.low_level_controller = _resolve_low_level_controller(controller)
        self.progress = progress
        if not callable(getattr(progress, "get_value", None)) or not callable(
            getattr(progress, "set_value", None)
        ):
            raise TypeError("progress must expose get_value() and set_value(value)")
        for name in ("attach", "update", "restore", "slot_identities"):
            if not callable(getattr(controller, name, None)):
                raise TypeError(f"controller must expose {name}()")
        if not hasattr(controller, "attached"):
            raise TypeError("controller must expose attached")

    def _source_identity_digest(self) -> str:
        controller = self.low_level_controller
        payload = {
            "surfaces": list(controller._surface_ids),
            "curves": list(controller._curve_ids),
            "boundarySources": list(controller._boundary_source_ids),
            "slotSources": list(controller._slot_source_ids),
            "boundaryStyles": sorted(controller.boundary_styles),
            "sectionEnabled": controller._section_enabled,
        }
        return sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def scan(
        self,
        progresses: Sequence[float],
        *,
        required_progresses: Sequence[float] = (),
        headroom: QuadricCapacityHeadroom = QuadricCapacityHeadroom(),
    ) -> QuadricCapacityPlan:
        """Plan exactly the listed progress values."""

        return self._scan(
            progresses,
            required_progresses=required_progresses,
            coverage="listed_progresses",
            headroom=headroom,
            schedule_digest=None,
            profile_id=None,
            frame_rate=None,
        )

    def scan_schedule(
        self,
        scheduled: ScheduledSectionAnimation,
        *,
        profile: QuadricRenderProfile,
        rate_function: Callable[[float], float] | None = None,
        headroom: QuadricCapacityHeadroom = QuadricCapacityHeadroom(),
    ) -> QuadricCapacityPlan:
        """Plan one render grid and every analytic schedule knot."""

        if not isinstance(profile, QuadricRenderProfile):
            raise TypeError("profile must be a QuadricRenderProfile")
        exposed = getattr(self.controller, "scheduled", None)
        if exposed is None:
            transition = getattr(self.controller, "transition_controller", None)
            exposed = None if transition is None else getattr(
                transition,
                "scheduled",
                None,
            )
        if not isinstance(exposed, ScheduledSectionAnimation):
            raise QuadricCapacityPlanningError(
                "scan_schedule requires a controller authored from the supplied "
                "ScheduledSectionAnimation; use scan() for custom low-level motion"
            )
        expected_schedule = canonical_plane_motion_schedule_json(scheduled.schedule)
        exposed_schedule = canonical_plane_motion_schedule_json(exposed.schedule)
        if (
            exposed_schedule != expected_schedule
            or exposed.animation.section_id != scheduled.animation.section_id
        ):
            raise QuadricCapacityPlanningError(
                "controller schedule does not match the supplied schedule"
            )
        progresses = scheduled_capacity_progresses(
            scheduled,
            frame_rate=profile.frame_rate,
            rate_function=rate_function,
        )
        digest = sha256(
            canonical_plane_motion_schedule_json(scheduled.schedule).encode("utf-8")
        ).hexdigest()
        return self._scan(
            progresses,
            required_progresses=scheduled.schedule.progresses,
            coverage="uniform_render_grid_plus_schedule_knots",
            headroom=headroom,
            schedule_digest=digest,
            profile_id=profile.profile_id,
            frame_rate=profile.frame_rate,
        )

    def _scan(
        self,
        progresses: Sequence[float],
        *,
        required_progresses: Sequence[float],
        coverage: str,
        headroom: QuadricCapacityHeadroom,
        schedule_digest: str | None,
        profile_id: str | None,
        frame_rate: float | None,
    ) -> QuadricCapacityPlan:
        if not isinstance(headroom, QuadricCapacityHeadroom):
            raise TypeError("headroom must be a QuadricCapacityHeadroom")
        values = _canonical_progresses(progresses)
        required = _canonical_progresses(required_progresses) if required_progresses else ()
        missing = tuple(
            value for value in required if not _contains_progress(values, value)
        )
        if missing:
            raise QuadricCapacityPlanningError(
                "capacity scan omits required progress values: "
                + ", ".join(f"{value:.12g}" for value in missing)
            )

        original_progress = _finite_progress(
            self.progress.get_value(),
            "original progress",
        )
        attached_before = bool(self.controller.attached)
        attached_here = False
        samples: list[QuadricCapacitySample] = []
        failure: Exception | None = None
        failure_phase: str | None = None
        failed_progress: float | None = None
        phase = "attachment"
        try:
            if not attached_before:
                self.controller.attach()
                attached_here = True
            scene = self.low_level_controller.scene
            scene_identity = tuple(id(item) for item in scene.mobjects)
            slot_identity = tuple(self.controller.slot_identities())
            for progress in values:
                phase = "progress"
                failed_progress = progress
                self.progress.set_value(progress)
                self.controller.update(0.0)
                if tuple(id(item) for item in scene.mobjects) != scene_identity:
                    raise QuadricCapacityPlanningError(
                        "capacity scan changed scene.mobjects"
                    )
                if tuple(self.controller.slot_identities()) != slot_identity:
                    raise QuadricCapacityPlanningError(
                        "capacity scan replaced a fixed Manim slot"
                    )
                samples.append(
                    _capacity_sample(self.low_level_controller, progress)
                )
        except Exception as exc:
            failure = exc
            failure_phase = phase
        finally:
            cleanup_error: Exception | None = None
            try:
                phase = "cleanup"
                self.progress.set_value(original_progress)
                if attached_here:
                    self.controller.restore()
                elif attached_before:
                    self.controller.update(0.0)
            except Exception as exc:
                cleanup_error = exc
            if failure is not None and cleanup_error is not None:
                failure.add_note(
                    "capacity planner also failed to restore the original state: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            elif failure is None and cleanup_error is not None:
                failure = cleanup_error
                failure_phase = "cleanup"

        if failure is not None:
            where = (
                f"during {failure_phase}"
                if failed_progress is None or failure_phase == "cleanup"
                else f"at progress {failed_progress:.12g}"
            )
            raise QuadricCapacityPlanningError(
                f"capacity scan failed {where}: "
                f"{type(failure).__name__}: {failure}"
            ) from failure

        peaks = QuadricCapacityPeaks.from_samples(samples)
        limits, estimated = _recommended_limits(
            self.low_level_controller,
            peaks,
            headroom,
        )
        return QuadricCapacityPlan(
            coverage=coverage,
            samples=tuple(samples),
            required_progresses=required,
            peaks=peaks,
            fixed_surface_count=len(self.low_level_controller._surface_ids),
            allocated_curve_count=len(self.low_level_controller._curve_ids),
            allocated_boundary_source_count=len(
                self.low_level_controller._boundary_source_ids
            ),
            slot_source_count=len(self.low_level_controller._slot_source_ids),
            boundary_style_count=len(self.low_level_controller.boundary_styles),
            section_enabled=self.low_level_controller._section_enabled,
            source_identity_digest=self._source_identity_digest(),
            recommended_limits=limits,
            estimated_mobject_total=estimated,
            headroom=headroom,
            schedule_digest=schedule_digest,
            profile_id=profile_id,
            frame_rate=frame_rate,
        )


__all__ = [
    "QUADRIC_CAPACITY_PLAN_SCHEMA",
    "QuadricCapacityHeadroom",
    "QuadricCapacityPeaks",
    "QuadricCapacityPlan",
    "QuadricCapacityPlanner",
    "QuadricCapacityPlanningError",
    "QuadricCapacitySample",
    "canonical_quadric_capacity_plan_json",
    "scheduled_capacity_progresses",
]
