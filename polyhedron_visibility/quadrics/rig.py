"""Natural fixed-topology Manim actions for one finite quadric section.

``QuadricSectionRig`` is intentionally a thin authoring layer over
:class:`~.authoring.QuadricSection3D`.  It compiles a mathematical plane path
before returning a Manim animation, then feeds exactly one immutable
:class:`SectionState` to the existing facade for every frame.  The rig does
not solve sections, allocate curve slots, or paint geometry itself.

Phase 1 accepts paths whose complete critical set can be certified with the
existing analytic rotation scheduler or the exact height critical values of a
parallel plane translation.  A path which needs topology-transition banks is
rejected before playback; that compilation belongs to the later timeline
layer.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import acos, atan2, ceil, floor, isfinite, pi, tau
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
from manim import Animation, Mobject

from ..geometry import GeometryContext, ResolvedGeometryContext
from ..parallel_solver import ParallelView
from ..painter_band import (
    ScenePainterBandReservation,
    release_scene_painter_band,
    reserve_scene_painter_band,
)
from .animation import (
    SectionAnimationSample,
    SectionAnimationTrace,
    SectionConicFamily,
    SectionTopologySignature,
    TrackedSectionFrame,
    _materialize_tracked_section_curves,
    match_tracked_section_frame,
    track_quadric_section_animation,
)
from .authoring import QuadricSection3D, QuadricSectionAuthoringError
from .contract import ConeModel, ConeSpec, CylinderSpec, SectionPlane, SphereSpec
from .manim import (
    DEFAULT_QUADRIC_VIEW,
    ProjectionValue,
    QuadricManimError,
)
from .manim_runtime import (
    _ResolvedParallelCameraFrame,
    _coerce_projection_frame,
)
from .plane_motion import (
    AxisAnglePlaneMotion,
    PlaneMotionError,
    _harmonic_coefficients,
    track_scheduled_plane_section,
)
from .section_compositing import PLANE_PATCH_RANK_RATIO_THRESHOLD
from .sections import (
    QuadricSectionBoundary,
    QuadricSurfaceSpec,
    compute_quadric_section_boundary,
    section_cap_chord_curve_ids,
)


_DEFAULT_PREFERRED_PAINTER_Z_BAND = (20.0, 30.0)
_PROGRESS_TOLERANCE = 1.0e-12


class QuadricSectionRigError(QuadricSectionAuthoringError):
    """A high-level mathematical action cannot be compiled safely."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuadricSectionRigError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise QuadricSectionRigError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricSectionRigError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise QuadricSectionRigError(f"{label} must be finite")
    return result


def _point3(value: object, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricSectionRigError(
            f"{label} must contain three finite values"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise QuadricSectionRigError(
            f"{label} must contain three finite values"
        )
    return result


def _unit3(value: object, label: str) -> np.ndarray:
    result = _point3(value, label)
    length = float(np.linalg.norm(result))
    if length <= 0.0:
        raise QuadricSectionRigError(f"{label} must be non-zero")
    return result / length


def _surface(value: object) -> QuadricSurfaceSpec:
    if not isinstance(value, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    if isinstance(value, ConeSpec):
        if value.model is ConeModel.OPEN_DOUBLE:
            raise QuadricSectionRigError(
                "OPEN_DOUBLE is outside the Phase 1 Rig contract; use "
                "CompositeQuadricSection3D"
            )
        if value.model is ConeModel.ANALYTIC_DOUBLE:
            raise QuadricSectionRigError(
                "ANALYTIC_DOUBLE is not a directly renderable finite surface"
            )
    return value


def _static_projection_frame(
    scene: object,
    value: object,
) -> _ResolvedParallelCameraFrame:
    """Freeze one complete Phase 1 projection without evaluating callbacks."""

    if callable(value):
        raise QuadricSectionRigError(
            "Phase 1 requires a static projection; callable projection is "
            "unsupported"
        )
    try:
        return _coerce_projection_frame(
            DEFAULT_QUADRIC_VIEW if value is None else value,
            scene=scene,
        )
    except (QuadricManimError, TypeError, ValueError) as exc:
        raise QuadricSectionRigError(
            f"invalid static parallel projection: {exc}"
        ) from exc


def _certify_plane_display_rank(
    plane: SectionPlane,
    view: ParallelView,
    *,
    action: str,
    progress: float,
) -> None:
    """Require the Phase 1 AREA-only subset of the compositor rank contract."""

    plane_u, plane_v, _normal = plane.basis
    screen_basis = view.matrix[:2] @ np.column_stack((plane_u, plane_v))
    determinant = float(np.linalg.det(screen_basis))
    basis_scale = max(
        float(np.linalg.norm(screen_basis, ord=2)),
        1.0e-300,
    )
    if (
        abs(determinant)
        <= PLANE_PATCH_RANK_RATIO_THRESHOLD * basis_scale * basis_scale
    ):
        raise QuadricSectionRigError(
            f"{action} cutting plane projects edge-on and loses display "
            f"rank (progress={progress:.12g})"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SectionState:
    """The one immutable mathematical state consumed by a rig frame.

    Phase 1 deliberately contains only the cutting plane.  View, paint policy,
    style, and display modes remain immutable facade configuration until their
    own reviewed authoring phases are added.
    """

    plane: SectionPlane

    def __post_init__(self) -> None:
        if not isinstance(self.plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")


@dataclass(frozen=True, slots=True)
class _FixedTopologyActionCertificate:
    """Evidence that one action stays inside the facade's fixed slot set."""

    action: str
    certified_progresses: tuple[float, ...]
    conic_family: SectionConicFamily
    component_count: int
    allocated_curve_ids: tuple[str, ...]
    proof: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _identity(self.action, "action"))
        if (
            not self.certified_progresses
            or self.certified_progresses[0] != 0.0
            or self.certified_progresses[-1] != 1.0
            or any(
                right <= left
                for left, right in zip(
                    self.certified_progresses,
                    self.certified_progresses[1:],
                )
            )
        ):
            raise QuadricSectionRigError(
                "certified progresses must increase from 0 to 1"
            )
        if not isinstance(self.conic_family, SectionConicFamily):
            raise TypeError("conic_family must be a SectionConicFamily")
        if self.component_count <= 0:
            raise QuadricSectionRigError(
                "a fixed-topology action requires at least one curve component"
            )
        if (
            not self.allocated_curve_ids
            or tuple(sorted(set(self.allocated_curve_ids)))
            != self.allocated_curve_ids
        ):
            raise QuadricSectionRigError(
                "allocated_curve_ids must be non-empty, unique, and canonical"
            )
        object.__setattr__(self, "proof", _identity(self.proof, "proof"))


@dataclass(frozen=True, slots=True)
class _CompiledPlaneAction:
    action: str
    start_state: SectionState
    target_state: SectionState
    plane_at: Callable[[float], SectionPlane]
    certificate: _FixedTopologyActionCertificate
    tracking: SectionAnimationTrace
    tracking_progresses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _RigFrameToken:
    committed_state: SectionState
    idle_reference: TrackedSectionFrame | None
    staged_reference: TrackedSectionFrame | None


def _rig_lateral_curve_ids(section_id: str) -> tuple[str, ...]:
    return tuple(
        f"{section_id}:rig:slot:{slot}:interval:{interval}"
        for slot in (0, 1)
        for interval in (0, 1)
    )


def _canonical_progresses(values: Sequence[float]) -> tuple[float, ...]:
    ordered: list[float] = []
    for raw in sorted(float(item) for item in values):
        value = min(1.0, max(0.0, raw))
        if not ordered or abs(value - ordered[-1]) > _PROGRESS_TOLERANCE:
            ordered.append(value)
    if not ordered or ordered[0] > _PROGRESS_TOLERANCE:
        ordered.insert(0, 0.0)
    else:
        ordered[0] = 0.0
    if ordered[-1] < 1.0 - _PROGRESS_TOLERANCE:
        ordered.append(1.0)
    else:
        ordered[-1] = 1.0
    return tuple(ordered)


def _periodic_progresses(
    motion: AxisAnglePlaneMotion,
    phases: Sequence[float],
) -> tuple[float, ...]:
    """Return the first in-path occurrence of each periodic angle family."""

    span = motion.end_angle - motion.start_angle
    length = abs(span)
    if length <= 0.0:  # pragma: no cover - AxisAnglePlaneMotion invariant
        return ()
    forward = span > 0.0
    tolerance = (
        128.0
        * float(np.finfo(float).eps)
        * max(
            1.0,
            min(length, tau),
            abs(motion.start_angle % tau),
            abs(motion.end_angle % tau),
            *(abs(float(phase)) for phase in phases),
        )
    )
    result: list[float] = []
    for phase in phases:
        signed_distance = (
            float(phase) - motion.start_angle
            if forward
            else motion.start_angle - float(phase)
        )
        distance = signed_distance % tau
        if distance <= tolerance or tau - distance <= tolerance:
            distance = 0.0
        if distance > length:
            if distance - length > tolerance:
                continue
            distance = length
        progress = min(1.0, max(0.0, distance / length))
        if not any(
            abs(progress - existing) <= _PROGRESS_TOLERANCE
            for existing in result
        ):
            result.append(float(progress))
    return tuple(sorted(result))


def _certify_rotation_display_rank(
    motion: AxisAnglePlaneMotion,
    view: ParallelView,
    *,
    action: str,
) -> None:
    """Conservatively prove the compositor rank predicate over a rotation."""

    cosine, sine_coefficient, constant = _harmonic_coefficients(
        motion,
        view.view_direction,
    )
    amplitude = float(np.hypot(cosine, sine_coefficient))
    coefficient_error = (
        4096.0
        * float(np.finfo(float).eps)
        * max(
            1.0,
            abs(cosine),
            abs(sine_coefficient),
            abs(constant),
        )
    )
    stationary_progresses: tuple[float, ...] = ()
    edge_on_progresses: tuple[float, ...] = ()
    if amplitude > 0.0:
        phase = atan2(sine_coefficient, cosine)
        stationary_progresses = _periodic_progresses(
            motion,
            (phase, phase + pi),
        )
        if abs(constant) <= amplitude + coefficient_error:
            ratio = min(1.0, max(-1.0, -constant / amplitude))
            offset = acos(ratio)
            edge_on_progresses = _periodic_progresses(
                motion,
                (phase - offset, phase + offset),
            )
    candidates = _canonical_progresses(
        (0.0, 1.0, *edge_on_progresses, *stationary_progresses)
    )
    alignments = tuple(
        abs(
            float(
                np.dot(
                    np.asarray(motion.plane_at(progress).normal, dtype=float),
                    np.asarray(view.view_direction, dtype=float),
                )
            )
        )
        for progress in candidates
    )
    minimum_index = min(range(len(candidates)), key=alignments.__getitem__)
    failing_progress = candidates[minimum_index]
    minimum_alignment = (
        0.0 if edge_on_progresses else alignments[minimum_index]
    )
    if edge_on_progresses:
        failing_progress = min(edge_on_progresses)

    screen_projection = view.matrix[:2]
    projection_scale = float(np.max(np.abs(screen_projection)))
    normalized_projection = screen_projection / projection_scale
    normalized_area_scale = float(
        np.linalg.norm(
            np.cross(normalized_projection[0], normalized_projection[1])
        )
    )
    normalized_basis_bound = float(
        np.linalg.norm(normalized_projection, ord=2)
    )
    certified_alignment = max(0.0, minimum_alignment - coefficient_error)
    certified_area = normalized_area_scale * certified_alignment
    required_area = (
        PLANE_PATCH_RANK_RATIO_THRESHOLD
        * normalized_basis_bound
        * normalized_basis_bound
    )
    if certified_area <= required_area:
        raise QuadricSectionRigError(
            f"{action} cutting plane becomes edge-on or numerically "
            "rank-deficient; display rank cannot be certified "
            f"(progress={failing_progress:.12g})"
        )


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


def _animation_tree_contains(root: object, target: object) -> bool:
    """Return whether a Scene play tree contains ``target`` by identity."""

    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if value is target:
            return True
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        children = (
            value
            if isinstance(value, (tuple, list))
            else getattr(value, "animations", ())
        )
        if isinstance(children, (tuple, list)):
            pending.extend(children)
    return False


def _animation_tree_reverses_target(root: object, target: object) -> bool:
    """Return whether ``target`` inherits a reversed animation-tree path."""

    pending: list[tuple[object, bool]] = [(root, False)]
    seen: set[tuple[int, bool]] = set()
    while pending:
        value, inherited_reverse = pending.pop()
        reversed_path = inherited_reverse or bool(
            getattr(value, "reverse_rate_function", False)
        )
        if value is target:
            return reversed_path
        key = (id(value), reversed_path)
        if key in seen:
            continue
        seen.add(key)
        children = (
            value
            if isinstance(value, (tuple, list))
            else getattr(value, "animations", ())
        )
        if isinstance(children, (tuple, list)):
            pending.extend((child, reversed_path) for child in children)
    return False


def _translation_critical_levels(
    surface: QuadricSurfaceSpec,
    normal: np.ndarray,
) -> tuple[float, ...]:
    """Return every height where a translated parallel section may change.

    These are the critical values of ``normal dot point`` on the compact
    lateral surface and its axial boundary circles.  For a one-nappe cone the
    apex height is also included because a generator can become a persistent
    critical level even when the displayed axial range excludes the apex.
    """

    if isinstance(surface, SphereSpec):
        center = np.asarray(surface.center, dtype=float)
        middle = float(np.dot(normal, center))
        return middle - surface.radius, middle + surface.radius

    axis = np.asarray(surface.axis, dtype=float)
    radial_length = float(
        np.linalg.norm(normal - float(np.dot(normal, axis)) * axis)
    )
    if isinstance(surface, CylinderSpec):
        origin = np.asarray(surface.origin, dtype=float)
        result: list[float] = []
        for axial in surface.axial_range:
            center = origin + axial * axis
            middle = float(np.dot(normal, center))
            radius_term = surface.radius * radial_length
            result.extend((middle - radius_term, middle + radius_term))
        return tuple(result)

    apex = np.asarray(surface.apex, dtype=float)
    apex_level = float(np.dot(normal, apex))
    result = [apex_level]
    for axial in surface.axial_range:
        center = apex + axial * axis
        middle = float(np.dot(normal, center))
        radius_term = abs(axial) * surface.slope * radial_length
        result.extend((middle - radius_term, middle + radius_term))
    return tuple(result)


def _translation_proof_progresses(
    surface: QuadricSurfaceSpec,
    start: SectionPlane,
    target: SectionPlane,
) -> tuple[float, ...]:
    normal = np.asarray(start.normal, dtype=float)
    start_level = float(np.dot(normal, np.asarray(start.point, dtype=float)))
    end_level = float(np.dot(normal, np.asarray(target.point, dtype=float)))
    delta = end_level - start_level
    values: list[float] = [0.0, 1.0]
    scale = max(1.0, abs(start_level), abs(end_level))
    if abs(delta) > np.finfo(float).eps * 128.0 * scale:
        for level in _translation_critical_levels(surface, normal):
            progress = (level - start_level) / delta
            if -_PROGRESS_TOLERANCE <= progress <= 1.0 + _PROGRESS_TOLERANCE:
                values.append(progress)
    return _with_interval_midpoints(values)


def _axis_alignment_progresses(
    surface: QuadricSurfaceSpec,
    motion: AxisAnglePlaneMotion,
) -> tuple[float, ...]:
    """Find circle/ellipse identity knots omitted by topology schedules.

    Circle and ellipse intentionally share one topological family, while the
    section solver gives their raw curves different representation IDs. The
    rig maps both to tracked capacity slots, but still authors exact alignment
    as canonical evidence instead of silently skipping the family-internal
    mathematical event.
    """

    if isinstance(surface, SphereSpec):
        return ()
    rotation_axis = np.asarray(motion.axis_direction, dtype=float)
    normal = np.asarray(motion.base_plane.normal, dtype=float)
    surface_axis = np.asarray(surface.axis, dtype=float)
    parallel = rotation_axis * float(np.dot(rotation_axis, normal))
    cosine_vector = normal - parallel
    sine_vector = np.cross(rotation_axis, normal)
    cosine = float(np.dot(cosine_vector, surface_axis))
    sine = float(np.dot(sine_vector, surface_axis))
    constant = float(np.dot(parallel, surface_axis))
    amplitude = float(np.hypot(cosine, sine))
    scale = max(1.0, abs(cosine), abs(sine), abs(constant))
    tolerance = 8192.0 * float(np.finfo(float).eps) * scale
    if amplitude <= tolerance:
        return ()

    low = min(motion.start_angle, motion.end_angle)
    high = max(motion.start_angle, motion.end_angle)
    phase = atan2(sine, cosine)
    angles: list[float] = []
    for target in (-1.0, 1.0):
        ratio = (target - constant) / amplitude
        if ratio < -1.0 - tolerance or ratio > 1.0 + tolerance:
            continue
        ratio = min(1.0, max(-1.0, ratio))
        offset = acos(ratio)
        for base in (phase - offset, phase + offset):
            first = floor((low - base) / tau) - 1
            last = ceil((high - base) / tau) + 1
            for index in range(first, last + 1):
                angle = base + index * tau
                if low - tolerance <= angle <= high + tolerance:
                    angles.append(min(high, max(low, angle)))
    denominator = motion.end_angle - motion.start_angle
    return _canonical_progresses(
        tuple(
            (angle - motion.start_angle) / denominator
            for angle in angles
            if -tolerance
            <= (angle - motion.start_angle) / denominator
            <= 1.0 + tolerance
        )
    ) if angles else ()


def _slot_compatible(
    baseline: SectionTopologySignature,
    candidate: SectionTopologySignature,
) -> bool:
    return (
        not baseline.degenerate
        and not candidate.degenerate
        and baseline.conic_family is candidate.conic_family
        and baseline.branch_count == candidate.branch_count
        and baseline.component_count == candidate.component_count
        and baseline.isolated_point_count == 0
        and candidate.isolated_point_count == 0
    )


def _certify_fixed_slots(
    *,
    action: str,
    proof: str,
    section_id: str,
    surface: QuadricSurfaceSpec,
    plane_at: Callable[[float], SectionPlane],
    progresses: Sequence[float],
    context: GeometryContext | ResolvedGeometryContext | None,
    coefficient_tolerance: float | None,
) -> tuple[_FixedTopologyActionCertificate, SectionAnimationTrace]:
    canonical = _canonical_progresses(progresses)
    boundaries = tuple(
        compute_quadric_section_boundary(
            section_id,
            surface,
            plane_at(progress),
            context=context,
            coefficient_tolerance=coefficient_tolerance,
        )
        for progress in canonical
    )
    baseline = SectionTopologySignature.from_trace(boundaries[0].trace)
    baseline_cap_chords = tuple(
        chord.curve_id for chord in boundaries[0].cap_chords
    )
    allocated = tuple(
        sorted(
            {
                *_rig_lateral_curve_ids(section_id),
                *section_cap_chord_curve_ids(section_id, surface),
            }
        )
    )
    if baseline.component_count <= 0 or baseline.degenerate:
        raise QuadricSectionRigError(
            f"{action} requires a non-degenerate, non-empty initial section"
        )
    for progress, boundary in zip(canonical, boundaries):
        signature = SectionTopologySignature.from_trace(boundary.trace)
        if not _slot_compatible(baseline, signature):
            raise QuadricSectionRigError(
                f"{action} crosses a section topology that requires the "
                f"scheduled transition controller (progress={progress:.12g})"
            )
        if (
            tuple(chord.curve_id for chord in boundary.cap_chords)
            != baseline_cap_chords
        ):
            raise QuadricSectionRigError(
                f"{action} changes cap-chord activation and requires the "
                f"scheduled transition controller (progress={progress:.12g})"
            )
        if any(
            len(component.parameter_intervals) > 2
            for component in boundary.trace.components
        ):
            raise QuadricSectionRigError(
                f"{action} exceeds the two-interval capacity of one fixed "
                f"component (progress={progress:.12g})"
            )
    tracking = track_quadric_section_animation(
        section_id,
        tuple(
            SectionAnimationSample(progress, surface, plane_at(progress))
            for progress in canonical
        ),
        context=context,
        coefficient_tolerance=coefficient_tolerance,
    )
    certificate = _FixedTopologyActionCertificate(
        action=action,
        certified_progresses=canonical,
        conic_family=baseline.conic_family,
        component_count=baseline.component_count,
        allocated_curve_ids=allocated,
        proof=proof,
    )
    return certificate, tracking


class QuadricSectionAction(Animation):
    """A precompiled plane action returned by ``QuadricSectionRig``."""

    def __init__(
        self,
        rig: "QuadricSectionRig",
        compiled: _CompiledPlaneAction,
        **animation_kwargs: object,
    ) -> None:
        self.rig = rig
        self._compiled = compiled
        self._begun_once = False
        self._finished_once = False
        # This non-drawing driver is allocated while authoring, never from an
        # updater.  It remains an ordinary remover so direct playback and
        # AnimationGroup/Succession clean up the same Scene family.  Every
        # exceptional path also removes it explicitly below.
        animation_kwargs["introducer"] = False
        super().__init__(Mobject(), remover=True, **animation_kwargs)

    @property
    def target_state(self) -> SectionState:
        return self._compiled.target_state

    def _validate_rate_func_contract(self) -> None:
        scene_animations = getattr(self.rig.scene, "animations", None)
        if self.reverse_rate_function or (
            scene_animations is not None
            and _animation_tree_reverses_target(scene_animations, self)
        ):
            raise QuadricSectionRigError(
                "reverse_rate_function=True is unsupported by "
                "QuadricSectionAction"
            )
        try:
            start = _finite(self.rate_func(0.0), "rate_func(0)")
            finish = _finite(self.rate_func(1.0), "rate_func(1)")
        except Exception as exc:
            if isinstance(exc, QuadricSectionRigError):
                raise
            raise QuadricSectionRigError(
                "rate_func endpoint evaluation failed"
            ) from exc
        if (
            abs(start) > _PROGRESS_TOLERANCE
            or abs(finish - 1.0) > _PROGRESS_TOLERANCE
        ):
            raise QuadricSectionRigError(
                "rate_func must map animation endpoints to 0 and 1"
            )

    def begin(self) -> None:
        if self._begun_once:
            raise QuadricSectionRigError("a QuadricSectionAction cannot be reused")
        try:
            self._validate_rate_func_contract()
            self.rig._begin_action(self)
            self._begun_once = True
            super().begin()
        except Exception:
            self.rig._abort_action(self)
            raise

    def interpolate_mobject(self, alpha: float) -> None:
        try:
            progress = _finite(
                self.rate_func(float(alpha)),
                "animation progress",
            )
            if (
                progress < -_PROGRESS_TOLERANCE
                or progress > 1.0 + _PROGRESS_TOLERANCE
            ):
                raise QuadricSectionRigError(
                    "animation rate_func left the certified progress "
                    "interval [0, 1]"
                )
            progress = min(1.0, max(0.0, progress))
            self.rig._set_action_progress(self, progress)
        except Exception:
            self.rig._abort_action(self)
            raise

    def finish(self) -> None:
        if self._finished_once:
            return
        try:
            super().finish()
            # Scene normally rendered alpha=1 immediately before ``finish``.
            # An explicit update also makes manual Animation driving obey the
            # same target-state transaction and is a clean-frame shortcut in
            # the ordinary Scene.play path.
            self.rig._set_action_progress(self, 1.0)
            self.rig.update()
            self.rig._finish_action(self)
            self._finished_once = True
        except Exception:
            self.rig._abort_action(self)
            raise


class QuadricSectionRig:
    """Author fixed-topology section motion as mathematical Manim actions.

    The wrapped :class:`QuadricSection3D` is created lazily on ``attach`` so a
    painter band is reserved only while the rig actually owns Scene objects.
    All extra keyword arguments are forwarded unchanged to the facade.
    ``scheduled`` and ``progress`` are intentionally unavailable here: a
    topology-changing timeline has one separate authority in the next phase.
    """

    def __init__(
        self,
        scene: object,
        *,
        surface: QuadricSurfaceSpec,
        plane: SectionPlane,
        section_id: str,
        projection: ProjectionValue | None = None,
        painter_z_band: tuple[float, float] | None = None,
        preferred_painter_z_band: tuple[float, float] = (
            _DEFAULT_PREFERRED_PAINTER_Z_BAND
        ),
        **section_options: object,
    ) -> None:
        if not isinstance(plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")
        forbidden = tuple(
            sorted(
                set(section_options)
                & {"surface", "plane", "section_id", "scheduled", "progress"}
            )
        )
        if forbidden:
            raise QuadricSectionRigError(
                "rig owns " + ", ".join(forbidden) + " and cannot forward them"
            )
        self.scene = scene
        self.surface = _surface(surface)
        self.section_id = _identity(section_id, "section_id")
        self._projection_frame = _static_projection_frame(scene, projection)
        self._view = self._projection_frame.view
        raw_show_plane = section_options.get("show_plane", True)
        if not isinstance(raw_show_plane, bool):
            raise TypeError("show_plane must be a bool")
        self._show_plane = raw_show_plane
        raw_draw_section_boundary = section_options.get(
            "draw_section_boundary",
            True,
        )
        if not isinstance(raw_draw_section_boundary, bool):
            raise TypeError("draw_section_boundary must be a bool")
        self._draw_section_boundary = raw_draw_section_boundary
        self._requires_display_rank = (
            self._show_plane or self._draw_section_boundary
        )
        if self._requires_display_rank:
            _certify_plane_display_rank(
                plane,
                self._view,
                action="initial rig view",
                progress=0.0,
            )
        self._state = SectionState(plane=plane)
        self._frame_state = self._state
        self._attach_state: SectionState | None = None
        self._facade: QuadricSection3D | None = None
        self._active_action: QuadricSectionAction | None = None
        self._active_progress = 0.0
        self._idle_reference: TrackedSectionFrame | None = None
        self._staged_reference: TrackedSectionFrame | None = None
        self._resolved_frame_state: SectionState | None = None
        self._resolved_action: QuadricSectionAction | None = None
        self._resolved_progress = 0.0
        self._frame_token: _RigFrameToken | None = None
        self._committed_frame_token: _RigFrameToken | None = None
        self._action_index = 0
        frozen_section_options = dict(section_options)
        frozen_section_options["projection"] = self._projection_frame
        self._section_options: Mapping[str, object] = frozen_section_options
        self._context = section_options.get("context")
        if self._context is not None and not isinstance(
            self._context,
            (GeometryContext, ResolvedGeometryContext),
        ):
            raise TypeError(
                "context must be a GeometryContext or ResolvedGeometryContext"
            )
        raw_tolerance = section_options.get("coefficient_tolerance")
        self._coefficient_tolerance = (
            None
            if raw_tolerance is None
            else _finite(raw_tolerance, "coefficient_tolerance")
        )
        preferred = (
            painter_z_band
            if painter_z_band is not None
            else preferred_painter_z_band
        )
        self._band_reservation = ScenePainterBandReservation(
            ("quadric-section-rig", self.section_id),
            preferred,
            exact=painter_z_band is not None,
        )
        self._painter_z_band: tuple[float, float] | None = None

    @property
    def state(self) -> SectionState:
        """Return the last completed authoring state."""

        return self._state

    @property
    def frame_state(self) -> SectionState:
        """Return the immutable state currently consumed by the frame."""

        return self._frame_state

    @property
    def plane(self) -> SectionPlane:
        return self._state.plane

    @property
    def view(self) -> ParallelView:
        """Return the immutable static parallel view certified by Phase 1."""

        return self._view

    @property
    def painter_z_band(self) -> tuple[float, float] | None:
        return self._painter_z_band

    @property
    def attached(self) -> bool:
        return self._facade is not None and self._facade.attached

    @property
    def controller(self):
        if self._facade is None:
            raise QuadricSectionRigError("rig is not attached")
        return self._facade.controller

    @property
    def display_mobject(self) -> Mobject:
        if self._facade is None:
            raise QuadricSectionRigError("rig is not attached")
        return self._facade.display_mobject

    @property
    def last_frame(self):
        return None if self._facade is None else self._facade.last_frame

    @property
    def last_global_frame(self):
        return None if self._facade is None else self._facade.last_global_frame

    @property
    def last_section_frame(self):
        return None if self._facade is None else self._facade.last_section_frame

    @property
    def last_boundary_frame(self):
        return None if self._facade is None else self._facade.last_boundary_frame

    @property
    def active_painter_z_indices(self) -> dict[str, float]:
        return (
            {}
            if self._facade is None
            else self._facade.active_painter_z_indices
        )

    @property
    def allocated_curve_ids(self) -> tuple[str, ...]:
        return () if self._facade is None else self._facade.allocated_curve_ids

    @property
    def allocated_boundary_ids(self) -> tuple[str, ...]:
        return () if self._facade is None else self._facade.allocated_boundary_ids

    def _resolve_plane(self) -> SectionPlane:
        state = (
            self._frame_state
            if self._resolved_frame_state is None
            else self._resolved_frame_state
        )
        return state.plane

    def attach(self) -> "QuadricSectionRig":
        if self.attached:
            return self
        band = reserve_scene_painter_band(self.scene, self._band_reservation)
        facade: QuadricSection3D | None = None
        self._attach_state = self._state
        self._frame_state = self._state
        try:
            facade = QuadricSection3D(
                self.scene,
                surface=self.surface,
                plane=self._resolve_plane,
                section_id=self.section_id,
                painter_z_band=band,
                _curve_binding=self,
                **self._section_options,
            )
            facade.controller._set_frame_transaction(self)
            facade.attach()
        except BaseException as error:
            if facade is not None:
                try:
                    facade.restore()
                except BaseException as cleanup_error:
                    error.add_note(
                        "facade cleanup after attach failure also failed: "
                        f"{cleanup_error!r}"
                    )
                    try:
                        facade.controller.restore()
                    except BaseException as controller_cleanup_error:
                        error.add_note(
                            "controller fallback cleanup also failed: "
                            f"{controller_cleanup_error!r}"
                        )
            try:
                release_scene_painter_band(
                    self.scene,
                    self._band_reservation,
                )
            except BaseException as release_error:
                error.add_note(
                    "painter-band cleanup after attach failure also failed: "
                    f"{release_error!r}"
                )
            self._cancel_quadric_frame()
            self._facade = None
            self._painter_z_band = None
            self._attach_state = None
            raise
        self._facade = facade
        self._painter_z_band = band
        return self

    def update(self, dt: float = 0.0) -> "QuadricSectionRig":
        if self._facade is None:
            raise QuadricSectionRigError("rig is not attached")
        try:
            self._facade.update(dt)
        except Exception:
            if self._active_action is not None:
                self._abort_action(self._active_action)
            raise
        return self

    def restore(self) -> "QuadricSectionRig":
        if self._active_action is not None:
            self._abort_action(self._active_action)
        facade = self._facade
        attach_state = self._attach_state
        failure: BaseException | None = None
        if facade is not None:
            try:
                facade.restore()
            except BaseException as error:
                failure = error
                try:
                    facade.controller.restore()
                except BaseException as controller_cleanup_error:
                    error.add_note(
                        "controller fallback cleanup also failed: "
                        f"{controller_cleanup_error!r}"
                    )
        try:
            release_scene_painter_band(self.scene, self._band_reservation)
        except BaseException as release_error:
            if failure is None:
                failure = release_error
            else:
                failure.add_note(
                    "painter-band cleanup also failed: "
                    f"{release_error!r}"
                )
        finally:
            self._cancel_quadric_frame()
            self._facade = None
            self._painter_z_band = None
            if attach_state is not None:
                self._state = attach_state
                self._frame_state = attach_state
            self._attach_state = None
        if failure is not None:
            raise failure.with_traceback(failure.__traceback__)
        return self

    def detach(self) -> "QuadricSectionRig":
        return self.restore()

    @contextmanager
    def session(self) -> Iterator["QuadricSectionRig"]:
        self.attach()
        try:
            yield self
        finally:
            self.restore()

    def slot_identities(self) -> tuple[int, ...]:
        if self._facade is None:
            return ()
        return self._facade.slot_identities()

    def slot_snapshot(self) -> tuple[object, ...]:
        if self._facade is None:
            return ()
        return self._facade.slot_snapshot()

    def _next_action_name(self, kind: str) -> str:
        return f"{self.section_id}:rig:{self._action_index:04d}:{kind}"

    def _compile_translation(
        self,
        *,
        action: str,
        plane_at: Callable[[float], SectionPlane],
    ) -> _CompiledPlaneAction:
        start = self._state
        target = plane_at(1.0)
        progresses = _translation_proof_progresses(
            self.surface,
            start.plane,
            target,
        )
        certificate, tracking = _certify_fixed_slots(
            action=action,
            proof="exact finite-surface height critical values",
            section_id=self.section_id,
            surface=self.surface,
            plane_at=plane_at,
            progresses=progresses,
            context=self._context,
            coefficient_tolerance=self._coefficient_tolerance,
        )
        self._action_index += 1
        return _CompiledPlaneAction(
            action,
            start,
            SectionState(plane=target),
            plane_at,
            certificate,
            tracking,
            progresses,
        )

    def animate_plane_shift(
        self,
        distance: float,
        *,
        direction: str | Sequence[float] = "normal",
        **animation_kwargs: object,
    ) -> QuadricSectionAction:
        """Move the reference point while preserving plane identity and frame."""

        amount = _finite(distance, "distance")
        start = self._state.plane
        if isinstance(direction, str):
            if direction.strip().lower() != "normal":
                raise QuadricSectionRigError(
                    "direction must be 'normal' or a three-component vector"
                )
            vector = np.asarray(start.normal, dtype=float)
        else:
            vector = _unit3(direction, "direction")
        start_point = np.asarray(start.point, dtype=float)

        def plane_at(progress: float) -> SectionPlane:
            point = start_point + float(progress) * amount * vector
            return SectionPlane(
                start.plane_id,
                tuple(float(item) for item in point),
                start.normal,
                u_axis=start.u_axis,
            )

        compiled = self._compile_translation(
            action=self._next_action_name("plane-shift"),
            plane_at=plane_at,
        )
        return QuadricSectionAction(self, compiled, **animation_kwargs)

    def animate_plane_rotation(
        self,
        axis: Sequence[float],
        angle: float,
        pivot: Sequence[float],
        **animation_kwargs: object,
    ) -> QuadricSectionAction:
        """Rigidly rotate point, normal, and in-plane axis by axis-angle."""

        amount = _finite(angle, "angle")
        if abs(amount) <= np.finfo(float).eps:
            # AxisAnglePlaneMotion intentionally rejects zero-span schedules;
            # a zero action is still a valid fixed-state Manim animation.
            _unit3(axis, "axis")
            _point3(pivot, "pivot")
            start_plane = self._state.plane

            def constant_plane(_progress: float) -> SectionPlane:
                return start_plane

            compiled = self._compile_translation(
                action=self._next_action_name("plane-rotation"),
                plane_at=constant_plane,
            )
            return QuadricSectionAction(self, compiled, **animation_kwargs)

        action = self._next_action_name("plane-rotation")
        try:
            motion = AxisAnglePlaneMotion(
                action,
                self._state.plane,
                tuple(float(item) for item in _point3(pivot, "pivot")),
                tuple(float(item) for item in _unit3(axis, "axis")),
                0.0,
                amount,
            )
            if self._requires_display_rank:
                _certify_rotation_display_rank(
                    motion,
                    self._view,
                    action=action,
                )
            scheduled = track_scheduled_plane_section(
                self.section_id,
                self.surface,
                motion,
                authored_progresses=_axis_alignment_progresses(
                    self.surface,
                    motion,
                ),
                context=self._context,
                coefficient_tolerance=self._coefficient_tolerance,
            )
        except PlaneMotionError as exc:
            raise QuadricSectionRigError(str(exc)) from exc
        certificate, tracking = _certify_fixed_slots(
            action=action,
            proof="analytic axis-angle critical schedule",
            section_id=self.section_id,
            surface=self.surface,
            plane_at=motion.plane_at,
            progresses=scheduled.schedule.progresses,
            context=self._context,
            coefficient_tolerance=self._coefficient_tolerance,
        )
        self._action_index += 1
        compiled = _CompiledPlaneAction(
            action,
            self._state,
            SectionState(plane=motion.plane_at(1.0)),
            motion.plane_at,
            certificate,
            tracking,
            certificate.certified_progresses,
        )
        return QuadricSectionAction(self, compiled, **animation_kwargs)

    def animate_plane_to(
        self,
        target_plane: SectionPlane,
        **animation_kwargs: object,
    ) -> QuadricSectionAction:
        """Move explicitly to a parallel target whose frame is unambiguous.

        A normal-changing target needs a mixed translation/shortest-rotation
        critical compiler.  Phase 1 fails that case before playback and points
        callers to ``animate_plane_rotation``; it never substitutes finite
        sampling for a continuous certificate.
        """

        if not isinstance(target_plane, SectionPlane):
            raise TypeError("target_plane must be a SectionPlane")
        start = self._state.plane
        if target_plane.plane_id != start.plane_id:
            raise QuadricSectionRigError("target plane must preserve plane_id")
        if target_plane.normal != start.normal:
            raise QuadricSectionRigError(
                "a normal-changing animate_plane_to path requires the topology-"
                "aware timeline compiler; use animate_plane_rotation for a "
                "certified rigid rotation in Phase 1"
            )
        if target_plane.u_axis != start.u_axis:
            raise QuadricSectionRigError(
                "parallel target changes the in-plane axis; the shortest normal "
                "transport has no such twist"
            )
        start_point = np.asarray(start.point, dtype=float)
        target_point = np.asarray(target_plane.point, dtype=float)

        def plane_at(progress: float) -> SectionPlane:
            if progress == 0.0:
                return start
            if progress == 1.0:
                return target_plane
            point = start_point + float(progress) * (target_point - start_point)
            return SectionPlane(
                start.plane_id,
                tuple(float(item) for item in point),
                start.normal,
                u_axis=start.u_axis,
            )

        compiled = self._compile_translation(
            action=self._next_action_name("plane-to"),
            plane_at=plane_at,
        )
        return QuadricSectionAction(self, compiled, **animation_kwargs)

    def _begin_action(self, action: QuadricSectionAction) -> None:
        if not self.attached:
            raise QuadricSectionRigError(
                "attach the rig or enter rig.session() before playing an action"
            )
        if self._active_action is not None:
            previous = self._active_action
            self._abort_action(previous)
            current_animations = getattr(self.scene, "animations", None)
            if (
                current_animations is None
                or not _animation_tree_contains(current_animations, action)
                or _animation_tree_contains(current_animations, previous)
            ):
                raise QuadricSectionRigError(
                    "only one mathematical action may drive a rig at a time"
                )
            # A sibling animation can make Scene.play exit before Manim calls
            # finish/cleanup on this action.  A later Scene.play has already
            # replaced ``scene.animations``; treat the absent previous action
            # as stale and recover from its last committed state.
        if self._state != action._compiled.start_state:
            raise QuadricSectionRigError(
                "rig state changed after this action was compiled; "
                "create a fresh action"
            )
        self._active_action = action
        self._active_progress = 0.0
        self._frame_state = self._state

    def _set_action_progress(
        self,
        action: QuadricSectionAction,
        progress: float,
    ) -> None:
        if self._active_action is not action:
            raise QuadricSectionRigError("action is not the active rig authority")
        self._active_progress = progress
        self._frame_state = SectionState(
            plane=action._compiled.plane_at(progress)
        )

    def _finish_action(self, action: QuadricSectionAction) -> None:
        if self._active_action is not action:
            raise QuadricSectionRigError("action is not the active rig authority")
        if self._state != action._compiled.target_state:
            raise QuadricSectionRigError(
                "target frame was not committed before the action finished"
            )
        self._frame_state = self._state
        self._active_action = None
        self._active_progress = 0.0

    def _abort_action(self, action: QuadricSectionAction) -> None:
        if self._active_action is action:
            self._frame_state = self._state
            self._active_action = None
            self._active_progress = 0.0
        remove = getattr(self.scene, "remove", None)
        if callable(remove):
            try:
                remove(action.mobject)
            except Exception:
                # Never replace the mathematical/display failure which caused
                # the abort with best-effort driver cleanup diagnostics.
                pass

    def _quadric_section_allocated_curve_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *_rig_lateral_curve_ids(self.section_id),
                    *section_cap_chord_curve_ids(
                        self.section_id,
                        self.surface,
                    ),
                }
            )
        )

    def _reference_for_boundary(
        self,
        boundary: QuadricSectionBoundary,
    ) -> TrackedSectionFrame:
        signature = SectionTopologySignature.from_trace(boundary.trace)
        action = (
            self._active_action
            if self._resolved_action is None
            else self._resolved_action
        )
        progress = (
            self._active_progress
            if self._resolved_action is None
            else self._resolved_progress
        )
        if action is not None:
            compiled = action._compiled
            candidates = tuple(
                (progress, frame)
                for progress, frame in zip(
                    compiled.tracking_progresses,
                    compiled.tracking.frames,
                )
                if frame.signature.topologically_equivalent(signature)
            )
            if not candidates:
                raise QuadricSectionRigError(
                    "compiled action has no reference for the live section topology"
                )
            return min(
                candidates,
                key=lambda item: (
                    abs(item[0] - progress),
                    item[0],
                ),
            )[1]
        if (
            self._idle_reference is not None
            and self._idle_reference.signature.topologically_equivalent(signature)
        ):
            return self._idle_reference
        initial = track_quadric_section_animation(
            self.section_id,
            (
                SectionAnimationSample(
                    0.0,
                    self.surface,
                    self._resolve_plane(),
                ),
            ),
            context=self._context,
            coefficient_tolerance=self._coefficient_tolerance,
        )
        return initial.frames[0]

    def _bind_quadric_section_curves(
        self,
        boundary: QuadricSectionBoundary,
    ) -> tuple[object, ...]:
        progress = (
            self._active_progress
            if self._resolved_action is None
            else self._resolved_progress
        )
        reference = self._reference_for_boundary(boundary)
        tracked = match_tracked_section_frame(
            reference,
            boundary.trace,
            time=progress,
        )
        curves: list[object] = list(
            _materialize_tracked_section_curves(
                tracked,
                lambda mapping, interval_index: (
                    f"{self.section_id}:rig:slot:{mapping.capacity_slot}:"
                    f"interval:{interval_index}"
                ),
                max_intervals_per_component=2,
            )
        )
        curves.extend(boundary.cap_chords)
        self._staged_reference = tracked
        return tuple(sorted(curves, key=lambda item: item.curve_id))

    def _begin_quadric_frame(self) -> _RigFrameToken:
        if (
            self._frame_token is not None
            or self._committed_frame_token is not None
        ):
            raise QuadricSectionRigError(
                "nested or unfinalized rig frame transaction"
            )
        token = _RigFrameToken(
            self._state,
            self._idle_reference,
            self._staged_reference,
        )
        self._frame_token = token
        self._resolved_frame_state = self._frame_state
        self._resolved_action = self._active_action
        self._resolved_progress = self._active_progress
        return token

    def _commit_quadric_frame(self, token: object) -> None:
        """Join the low-level display commit with the staged author state."""

        if token is not self._frame_token or self._resolved_frame_state is None:
            raise QuadricSectionRigError("invalid rig frame commit token")
        self._state = self._resolved_frame_state
        self._frame_state = self._state
        if self._staged_reference is not None:
            self._idle_reference = self._staged_reference
        self._resolved_frame_state = None
        self._resolved_action = None
        self._resolved_progress = 0.0
        self._committed_frame_token = token
        self._frame_token = None

    def _finalize_quadric_frame(self, token: object) -> None:
        """Forget rollback evidence after the joint frame commit succeeds."""

        if token is not self._committed_frame_token or self._frame_token is not None:
            raise QuadricSectionRigError("invalid rig frame finalize token")
        self._committed_frame_token = None

    def _rollback_quadric_frame(self, token: object) -> None:
        """Discard a failed frame without retaining its mathematical plane."""

        if not isinstance(token, _RigFrameToken) or (
            token is not self._frame_token
            and token is not self._committed_frame_token
        ):
            raise QuadricSectionRigError("invalid rig frame rollback token")
        self._state = token.committed_state
        self._frame_state = token.committed_state
        self._idle_reference = token.idle_reference
        self._staged_reference = token.staged_reference
        self._active_action = None
        self._active_progress = 0.0
        self._resolved_frame_state = None
        self._resolved_action = None
        self._resolved_progress = 0.0
        self._frame_token = None
        self._committed_frame_token = None

    def _cancel_quadric_frame(self) -> None:
        self._resolved_frame_state = None
        self._resolved_action = None
        self._resolved_progress = 0.0
        self._frame_token = None
        self._committed_frame_token = None
        self._frame_state = self._state
        self._staged_reference = self._idle_reference
        self._active_action = None
        self._active_progress = 0.0


__all__ = [
    "QuadricSectionAction",
    "QuadricSectionRig",
    "QuadricSectionRigError",
    "SectionState",
]
