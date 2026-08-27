"""High-level Manim authoring facade for one finite quadric section.

``QuadricSection3D`` keeps the cutting plane, its complete finite boundary,
the fixed curve-slot allocation, and the existing Cairo compositor under one
authoring contract.  It deliberately owns no renderer or geometry solver: the
static path delegates to :class:`QuadricOcclusion3D`, while scheduled topology
changes delegate to :class:`QuadricSectionTransition3D`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Literal, Mapping, Sequence

from manim import Mobject

from ..geometry import GeometryContext, ResolvedGeometryContext
from .animation import SectionTopologySignature
from .boundary_compositing import QuadricBoundaryCompositingFrame
from .boundary_section import (
    QUADRIC_BOUNDARY_SECTION_LIMITS,
    QuadricBoundarySectionLimits,
)
from .compositing import (
    QuadricCompositingFrame,
    QuadricPaintPolicy,
    SurfaceConstraintInput,
)
from .contract import ConeSpec, CylinderSpec, SectionPlane, SphereSpec
from .global_occlusion import GlobalQuadricFrame
from .manim import (
    QUADRIC_MANIM_LIMITS,
    ProjectionInput,
    QuadricBoundaryStyle,
    QuadricManimError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from .plane_motion import ScheduledSectionAnimation
from .section_compositing import (
    QUADRIC_SECTION_COMPOSITING_LIMITS,
    QuadricSectionCompositingLimits,
    QuadricSectionCompositingFrame,
)
from .sections import (
    FiniteSectionBoundaryCurve,
    QuadricSurfaceSpec,
    compute_quadric_section_boundary_curves,
    section_cap_chord_curve_ids,
)
from .surface_boundaries import GeneratorBoundarySpec
from .transition import SectionTransitionFrame, SectionTransitionMode
from .transition_manim import (
    ProgressInput,
    QuadricSectionTransition3D,
)


PlaneInput = SectionPlane | Callable[[], SectionPlane]
_UNSET = object()


class QuadricSectionAuthoringError(QuadricManimError):
    """A high-level section controller was configured ambiguously."""


def _section_identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuadricSectionAuthoringError(
            "section_id must be a non-empty string"
        )
    return value.strip()


class QuadricSection3D:
    """Author one finite quadric section without synchronizing low-level inputs.

    Static and fixed-topology motion use ``surface``, ``section_id``, and
    ``plane``.  ``plane`` may be a callback returning a fresh immutable
    :class:`SectionPlane` for the current frame.  Potential cap-chord slots are
    reserved automatically.

    Topology-changing motion instead uses ``scheduled`` and ``progress`` and
    is delegated to :class:`QuadricSectionTransition3D`.  The two modes are
    intentionally mutually exclusive so there is only one authority for the
    current plane and section identity.
    """

    def __init__(
        self,
        scene: object,
        *,
        surface: QuadricSurfaceSpec | None = None,
        section_id: str | None = None,
        plane: PlaneInput | None = None,
        scheduled: ScheduledSectionAnimation | None = None,
        progress: ProgressInput = _UNSET,
        projection: ProjectionInput | None = None,
        transition_fraction: float = 0.04,
        transition_mode: SectionTransitionMode | str = (
            SectionTransitionMode.CROSSFADE
        ),
        paint_policy: QuadricPaintPolicy | str = QuadricPaintPolicy.DIAGRAMMATIC,
        style: QuadricManimStyle = QuadricManimStyle(),
        boundary_styles: Mapping[str, QuadricBoundaryStyle] | None = None,
        limits: QuadricManimLimits = QUADRIC_MANIM_LIMITS,
        max_chord_error: float = 1.0e-3,
        painter_z_band: tuple[float, float] = (20.0, 30.0),
        surface_constraints: Sequence[SurfaceConstraintInput] = (),
        context: GeometryContext | ResolvedGeometryContext | None = None,
        coefficient_tolerance: float | None = None,
        draw_section_boundary: bool = True,
        show_plane: bool = True,
        plane_patch_margin: float = 0.08,
        section_max_screen_error: float = 0.08,
        section_compositing_limits: QuadricSectionCompositingLimits = (
            QUADRIC_SECTION_COMPOSITING_LIMITS
        ),
        boundary_section_limits: QuadricBoundarySectionLimits = (
            QUADRIC_BOUNDARY_SECTION_LIMITS
        ),
        include_surface_boundaries: bool = True,
        generator_boundaries: Sequence[GeneratorBoundarySpec] = (),
        allocated_boundary_ids: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(draw_section_boundary, bool):
            raise TypeError("draw_section_boundary must be a bool")
        if not isinstance(show_plane, bool):
            raise TypeError("show_plane must be a bool")
        self.scene = scene
        self.draw_section_boundary = draw_section_boundary
        self.show_plane = show_plane
        self.context = context
        self.coefficient_tolerance = coefficient_tolerance
        self._plane_input: PlaneInput | None = None
        self._expected_plane_id: str | None = None
        self._pending_plane: SectionPlane | None = None
        self._initial_plane: SectionPlane | None = None
        self._initial_curves: tuple[FiniteSectionBoundaryCurve, ...] | None = None
        self._transition: QuadricSectionTransition3D | None = None

        common = {
            "projection": projection,
            "paint_policy": paint_policy,
            "style": style,
            "boundary_styles": boundary_styles,
            "limits": limits,
            "max_chord_error": max_chord_error,
            "painter_z_band": painter_z_band,
            "surface_constraints": surface_constraints,
            "context": context,
            "section_max_screen_error": section_max_screen_error,
            "section_compositing_limits": section_compositing_limits,
            "boundary_section_limits": boundary_section_limits,
            "include_surface_boundaries": include_surface_boundaries,
            "generator_boundaries": generator_boundaries,
            "allocated_boundary_ids": allocated_boundary_ids,
        }

        if scheduled is not None:
            if not isinstance(scheduled, ScheduledSectionAnimation):
                raise TypeError("scheduled must be a ScheduledSectionAnimation")
            supplied_static = tuple(
                name
                for name, value in (
                    ("surface", surface),
                    ("section_id", section_id),
                    ("plane", plane),
                )
                if value is not None
            )
            if supplied_static:
                raise QuadricSectionAuthoringError(
                    "scheduled mode cannot also define "
                    + ", ".join(supplied_static)
                )
            if progress is _UNSET:
                raise QuadricSectionAuthoringError(
                    "scheduled mode requires progress"
                )
            self.mode: Literal["static", "scheduled"] = "scheduled"
            self.surface = scheduled.schedule.samples[0].surface
            self.section_id = scheduled.animation.section_id
            self._transition = QuadricSectionTransition3D(
                scene,
                scheduled=scheduled,
                progress=progress,
                transition_fraction=transition_fraction,
                transition_mode=transition_mode,
                coefficient_tolerance=coefficient_tolerance,
                draw_section_boundary=draw_section_boundary,
                show_plane=show_plane,
                plane_patch_margin=plane_patch_margin,
                boundary_visibility_mode="unified",
                **common,
            )
            self._controller = self._transition.controller
            return

        if progress is not _UNSET:
            raise QuadricSectionAuthoringError(
                "progress is only valid together with scheduled"
            )
        missing = tuple(
            name
            for name, value in (
                ("surface", surface),
                ("section_id", section_id),
                ("plane", plane),
            )
            if value is None
        )
        if missing:
            raise QuadricSectionAuthoringError(
                "static mode requires " + ", ".join(missing)
            )
        if not isinstance(surface, (SphereSpec, CylinderSpec, ConeSpec)):
            raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
        assert plane is not None
        self.mode = "static"
        self.surface = surface
        self.section_id = _section_identity(section_id)
        self._plane_input = plane
        initial_plane = self._resolve_plane_source()
        self._expected_plane_id = initial_plane.plane_id

        if draw_section_boundary:
            initial_curves = self._compute_boundary(initial_plane)
            self._initial_plane = initial_plane
            self._initial_curves = initial_curves
            allocated_curve_ids = tuple(
                sorted(
                    {
                        *(item.curve_id for item in initial_curves),
                        *section_cap_chord_curve_ids(self.section_id, surface),
                    }
                )
            )
            curve_input = self._active_curves
        else:
            allocated_curve_ids = ()
            curve_input = ()

        self._controller = QuadricOcclusion3D(
            scene,
            surfaces=(surface,),
            curves=curve_input,
            allocated_curve_ids=allocated_curve_ids,
            section_plane=(self._active_plane if show_plane else None),
            section_patch_margin=plane_patch_margin,
            surface_order_mode="automatic",
            boundary_visibility_mode="unified",
            **common,
        )

    def _resolve_plane_source(self) -> SectionPlane:
        source = self._plane_input
        value = source() if callable(source) else source
        if not isinstance(value, SectionPlane):
            raise QuadricSectionAuthoringError(
                "plane must resolve to a SectionPlane"
            )
        if (
            self._expected_plane_id is not None
            and value.plane_id != self._expected_plane_id
        ):
            raise QuadricSectionAuthoringError(
                "plane identity changed while QuadricSection3D was active"
            )
        return value

    def _compute_boundary(
        self,
        plane: SectionPlane,
    ) -> tuple[FiniteSectionBoundaryCurve, ...]:
        return compute_quadric_section_boundary_curves(
            self.section_id,
            self.surface,
            plane,
            context=self.context,
            coefficient_tolerance=self.coefficient_tolerance,
        )

    def _active_curves(self) -> tuple[FiniteSectionBoundaryCurve, ...]:
        if self._initial_curves is not None:
            curves = self._initial_curves
            plane = self._initial_plane
            self._initial_curves = None
            self._initial_plane = None
            assert plane is not None
            self._pending_plane = plane
            return curves
        plane = self._resolve_plane_source()
        curves = self._compute_boundary(plane)
        self._pending_plane = plane
        return curves

    def _active_plane(self) -> SectionPlane:
        if self._pending_plane is not None:
            plane = self._pending_plane
            self._pending_plane = None
            return plane
        return self._resolve_plane_source()

    @property
    def controller(self) -> QuadricOcclusion3D:
        """Return the single existing low-level renderer used by the facade."""

        return self._controller

    @property
    def transition_controller(self) -> QuadricSectionTransition3D | None:
        """Return the scheduled handoff controller, or ``None`` in static mode."""

        return self._transition

    @property
    def attached(self) -> bool:
        return self._controller.attached

    @property
    def display_mobject(self) -> Mobject:
        return self._controller.display_mobject

    @property
    def last_frame(self) -> QuadricCompositingFrame | None:
        return self._controller.last_frame

    @property
    def last_global_frame(self) -> GlobalQuadricFrame | None:
        return self._controller.last_global_frame

    @property
    def last_section_frame(self) -> QuadricSectionCompositingFrame | None:
        return self._controller.last_section_frame

    @property
    def last_boundary_frame(self) -> QuadricBoundaryCompositingFrame | None:
        return self._controller.last_boundary_frame

    @property
    def transition_frame(self) -> SectionTransitionFrame:
        if self._transition is None:
            raise QuadricSectionAuthoringError(
                "transition_frame is available only in scheduled mode"
            )
        return self._transition.transition_frame

    @property
    def active_signatures(self) -> tuple[SectionTopologySignature, ...]:
        if self._transition is None:
            raise QuadricSectionAuthoringError(
                "active_signatures is available only in scheduled mode"
            )
        return self._transition.active_signatures

    @property
    def active_painter_z_indices(self) -> dict[str, float]:
        return self._controller.active_painter_z_indices

    @property
    def allocated_curve_ids(self) -> tuple[str, ...]:
        return self._controller.allocated_curve_ids

    @property
    def allocated_boundary_ids(self) -> tuple[str, ...]:
        return self._controller.allocated_boundary_ids

    def attach(self) -> "QuadricSection3D":
        if self._transition is None:
            self._controller.attach()
        else:
            self._transition.attach()
        return self

    def update(self, dt: float = 0.0) -> "QuadricSection3D":
        if self._transition is None:
            self._controller.update(dt)
        else:
            self._transition.update(dt)
        return self

    def restore(self) -> "QuadricSection3D":
        if self._transition is None:
            self._controller.restore()
        else:
            self._transition.restore()
        return self

    def detach(self) -> "QuadricSection3D":
        return self.restore()

    @contextmanager
    def session(self) -> Iterator["QuadricSection3D"]:
        self.attach()
        try:
            yield self
        finally:
            self.restore()

    def slot_identities(self) -> tuple[int, ...]:
        return self._controller.slot_identities()

    def slot_snapshot(self) -> tuple[object, ...]:
        return self._controller.slot_snapshot()


__all__ = [
    "QuadricSection3D",
    "QuadricSectionAuthoringError",
    "PlaneInput",
]
