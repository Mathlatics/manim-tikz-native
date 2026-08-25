"""Cairo binding for automatic quadric-section topology handoff.

``QuadricSectionTransition3D`` consumes an analytic
:class:`~polyhedron_visibility.quadrics.plane_motion.ScheduledSectionAnimation`
and a normalized Manim progress source.  It reserves two banks of finite
section curves once, then feeds one or two bank subsets into the ordinary
``QuadricOcclusion3D`` compositor.  No Mobject is created by a frame updater,
and both sides of a cross-fade retain full surface occlusion and painter-graph
ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping, Sequence

from manim import Mobject

from ..geometry import GeometryContext, ResolvedGeometryContext
from .animation import (
    SectionAnimationError,
    SectionTopologySignature,
    match_tracked_section_frame,
)
from .compositing import QuadricPaintPolicy, SurfaceConstraintInput
from .contract import SectionPlane
from .curves import ParametricConicBranch
from .manim import (
    QUADRIC_MANIM_LIMITS,
    ProjectionInput,
    QuadricBoundaryStyle,
    QuadricManimCapacityError,
    QuadricManimError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from .boundary_section import (
    QUADRIC_BOUNDARY_SECTION_LIMITS,
    QuadricBoundarySectionLimits,
)
from .section_compositing import (
    QUADRIC_SECTION_COMPOSITING_LIMITS,
    QuadricSectionCompositingLimits,
)
from .surface_boundaries import GeneratorBoundarySpec
from .plane_motion import ScheduledSectionAnimation
from .sections import compute_quadric_section
from .transition import (
    SectionTransitionFrame,
    SectionTransitionMode,
    SectionTransitionPlan,
    build_section_transition_plan,
)


MAX_TRANSITION_INTERVAL_SLOTS = 2


class QuadricSectionTransitionManimError(QuadricManimError):
    """A scheduled section cannot be handed off safely to Manim."""


ProgressInput = float | Callable[[], float] | object


def _progress_value(source: ProgressInput) -> float:
    if callable(source):
        raw = source()
    elif hasattr(source, "get_value") and callable(source.get_value):
        raw = source.get_value()
    else:
        raw = source
    if isinstance(raw, bool):
        raise QuadricSectionTransitionManimError(
            "transition progress must be finite and lie in [0, 1]"
        )
    try:
        result = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricSectionTransitionManimError(
            "transition progress must be finite and lie in [0, 1]"
        ) from exc
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise QuadricSectionTransitionManimError(
            "transition progress must be finite and lie in [0, 1]"
        )
    return result


def _curve_slot_id(
    section_id: str,
    bank_index: int,
    capacity_slot: int,
    interval_slot: int,
) -> str:
    return (
        f"{section_id}:transition:bank:{bank_index}:"
        f"slot:{capacity_slot}:interval:{interval_slot}"
    )


def _allocated_curve_ids(section_id: str) -> tuple[str, ...]:
    return tuple(
        _curve_slot_id(section_id, bank, slot, interval)
        for bank in (0, 1)
        for slot in (0, 1)
        for interval in range(MAX_TRANSITION_INTERVAL_SLOTS)
    )


@dataclass(frozen=True, slots=True)
class PreparedSectionTransitionGeometry:
    """Renderer-neutral geometry selected for one Manim update."""

    transition_frame: SectionTransitionFrame
    curves: tuple[ParametricConicBranch, ...]
    curve_opacities: Mapping[str, float]
    signatures: tuple[SectionTopologySignature, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.transition_frame, SectionTransitionFrame):
            raise TypeError("transition_frame must be a SectionTransitionFrame")
        if not all(isinstance(item, ParametricConicBranch) for item in self.curves):
            raise TypeError("curves must contain ParametricConicBranch objects")
        curve_ids = tuple(item.curve_id for item in self.curves)
        if len(set(curve_ids)) != len(curve_ids):
            raise QuadricSectionTransitionManimError(
                "prepared transition curve identities must be unique"
            )
        if set(curve_ids) != set(self.curve_opacities):
            raise QuadricSectionTransitionManimError(
                "prepared transition opacities must cover every active curve"
            )
        if len(self.signatures) != len(self.transition_frame.layers):
            raise QuadricSectionTransitionManimError(
                "prepared signatures must describe every transition layer"
            )


class QuadricSectionTransition3D:
    """Automatic fixed-capacity handoff for a scheduled moving section.

    The caller animates one normalized progress source from ``0`` to ``1``.
    Analytic schedule events select the correct live and critical sections;
    this controller only manages their preallocated render banks.
    """

    def __init__(
        self,
        scene: object,
        *,
        scheduled: ScheduledSectionAnimation,
        progress: ProgressInput,
        projection: ProjectionInput | None = None,
        transition_fraction: float = 0.04,
        transition_mode: SectionTransitionMode | str = SectionTransitionMode.CROSSFADE,
        paint_policy: QuadricPaintPolicy | str = QuadricPaintPolicy.DIAGRAMMATIC,
        style: QuadricManimStyle = QuadricManimStyle(),
        boundary_styles: Mapping[str, QuadricBoundaryStyle] | None = None,
        limits: QuadricManimLimits = QUADRIC_MANIM_LIMITS,
        max_chord_error: float = 1.0e-3,
        painter_z_band: tuple[float, float] = (20.0, 30.0),
        surface_constraints: Sequence[SurfaceConstraintInput] = (),
        context: GeometryContext | ResolvedGeometryContext | None = None,
        coefficient_tolerance: float | None = None,
        show_plane: bool = True,
        plane_patch_margin: float = 0.08,
        section_max_screen_error: float = 0.08,
        section_compositing_limits: QuadricSectionCompositingLimits = (
            QUADRIC_SECTION_COMPOSITING_LIMITS
        ),
        boundary_section_limits: QuadricBoundarySectionLimits = (
            QUADRIC_BOUNDARY_SECTION_LIMITS
        ),
        boundary_visibility_mode: str = "legacy",
        include_surface_boundaries: bool = True,
        generator_boundaries: Sequence[GeneratorBoundarySpec] = (),
        allocated_boundary_ids: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(scheduled, ScheduledSectionAnimation):
            raise TypeError("scheduled must be a ScheduledSectionAnimation")
        self.scene = scene
        self.scheduled = scheduled
        self.progress_source = progress
        self.plan: SectionTransitionPlan = build_section_transition_plan(
            scheduled,
            transition_fraction=transition_fraction,
            mode=transition_mode,
        )
        if context is not None and not isinstance(
            context, (GeometryContext, ResolvedGeometryContext)
        ):
            raise TypeError(
                "context must be a GeometryContext or ResolvedGeometryContext"
            )
        self.context = context
        self.coefficient_tolerance = coefficient_tolerance
        if not isinstance(show_plane, bool):
            raise TypeError("show_plane must be a bool")
        self.show_plane = show_plane
        self._allocated_curve_ids = _allocated_curve_ids(self.plan.section_id)
        self._cache_progress: float | None = None
        self._cache_geometry: PreparedSectionTransitionGeometry | None = None

        surface = scheduled.schedule.samples[0].surface
        self._controller = QuadricOcclusion3D(
            scene,
            surfaces=(surface,),
            curves=self._active_curves,
            curve_opacities=self._active_curve_opacities,
            allocated_curve_ids=self._allocated_curve_ids,
            projection=projection,
            paint_policy=paint_policy,
            style=style,
            boundary_styles=boundary_styles,
            limits=limits,
            max_chord_error=max_chord_error,
            context=context,
            painter_z_band=painter_z_band,
            surface_constraints=surface_constraints,
            surface_order_mode="automatic",
            section_plane=(self._active_plane if show_plane else None),
            section_patch_margin=plane_patch_margin,
            section_max_screen_error=section_max_screen_error,
            section_compositing_limits=section_compositing_limits,
            boundary_section_limits=boundary_section_limits,
            boundary_visibility_mode=boundary_visibility_mode,
            include_surface_boundaries=include_surface_boundaries,
            generator_boundaries=generator_boundaries,
            allocated_boundary_ids=allocated_boundary_ids,
        )

    def _active_plane(self) -> SectionPlane:
        return self.scheduled.schedule.motion.plane_at(
            _progress_value(self.progress_source)
        )

    def _resolve_geometry(self) -> PreparedSectionTransitionGeometry:
        progress = _progress_value(self.progress_source)
        if self._cache_progress == progress and self._cache_geometry is not None:
            return self._cache_geometry
        transition_frame = self.plan.sample(progress)
        curves: list[ParametricConicBranch] = []
        opacities: dict[str, float] = {}
        signatures: list[SectionTopologySignature] = []
        motion = self.scheduled.schedule.motion
        surface = self.scheduled.schedule.samples[0].surface
        frames = self.scheduled.animation.frames

        for layer in transition_frame.layers:
            plane = motion.plane_at(layer.geometry_progress)
            trace = compute_quadric_section(
                self.plan.section_id,
                surface,
                plane,
                context=self.context,
                coefficient_tolerance=self.coefficient_tolerance,
            )
            signature = SectionTopologySignature.from_trace(trace)
            reference = frames[layer.reference_frame_index]
            try:
                tracked = match_tracked_section_frame(
                    reference,
                    trace,
                    frame_index=reference.frame_index,
                    time=motion.time_at(layer.geometry_progress),
                )
            except SectionAnimationError as exc:
                raise QuadricSectionTransitionManimError(
                    "live section no longer matches its analytic topology schedule"
                ) from exc
            signatures.append(signature)
            for branch_mapping in tracked.branches:
                component = trace.component_map[
                    branch_mapping.source_component_id
                ]
                branch = trace.branch_map[branch_mapping.source_branch_id]
                if len(component.parameter_intervals) > MAX_TRANSITION_INTERVAL_SLOTS:
                    raise QuadricManimCapacityError(
                        "one section component exceeds the two-interval transition "
                        "capacity"
                    )
                for interval_slot, interval in enumerate(
                    component.parameter_intervals
                ):
                    curve_id = _curve_slot_id(
                        self.plan.section_id,
                        layer.bank_index,
                        branch_mapping.capacity_slot,
                        interval_slot,
                    )
                    curves.append(
                        ParametricConicBranch(
                            curve_id,
                            branch.parameterization,
                            branch.plane_embedding,
                            interval,
                        )
                    )
                    opacities[curve_id] = layer.opacity

        prepared = PreparedSectionTransitionGeometry(
            transition_frame=transition_frame,
            curves=tuple(sorted(curves, key=lambda item: item.curve_id)),
            curve_opacities={key: opacities[key] for key in sorted(opacities)},
            signatures=tuple(signatures),
        )
        self._cache_progress = progress
        self._cache_geometry = prepared
        return prepared

    def _active_curves(self) -> tuple[ParametricConicBranch, ...]:
        return self._resolve_geometry().curves

    def _active_curve_opacities(self) -> Mapping[str, float]:
        return self._resolve_geometry().curve_opacities

    @property
    def controller(self) -> QuadricOcclusion3D:
        return self._controller

    @property
    def transition_frame(self) -> SectionTransitionFrame:
        return self._resolve_geometry().transition_frame

    @property
    def active_signatures(self) -> tuple[SectionTopologySignature, ...]:
        return self._resolve_geometry().signatures

    @property
    def attached(self) -> bool:
        return self._controller.attached

    @property
    def display_mobject(self) -> Mobject:
        return self._controller.display_mobject

    def attach(self) -> "QuadricSectionTransition3D":
        self._controller.attach()
        return self

    def update(self, dt: float = 0.0) -> "QuadricSectionTransition3D":
        self._controller.update(dt)
        return self

    def restore(self) -> "QuadricSectionTransition3D":
        self._controller.restore()
        return self

    def detach(self) -> "QuadricSectionTransition3D":
        return self.restore()

    def slot_identities(self) -> tuple[int, ...]:
        return self._controller.slot_identities()


__all__ = [
    "MAX_TRANSITION_INTERVAL_SLOTS",
    "PreparedSectionTransitionGeometry",
    "QuadricSectionTransition3D",
    "QuadricSectionTransitionManimError",
]
