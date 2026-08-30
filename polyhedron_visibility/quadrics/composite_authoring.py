"""Fixed-capacity Manim coordinator for one finite open double-cone section.

``CompositeQuadricSection3D`` is deliberately a sibling coordinator, not a
second section solver.  It expands one ``OPEN_DOUBLE`` into its two canonical
``OPEN_SINGLE`` components, builds the existing local section frame for each,
merges those renderer-neutral products, and commits one shared Cairo painter
band.  Every Mobject is allocated in ``__init__`` and every update is
transactional.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import combinations
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
from manim import (
    Line,
    Mobject,
    RendererType,
    ThreeDCamera,
    VGroup,
    VMobject,
    config,
)

from ..geometry import GeometryContext, ResolvedGeometryContext
from ..painter_band import (
    ManagedPainterBand,
    ManagedPainterBandError,
    PreparedPainterBand,
)
from ..parallel_solver import ParallelView
from ..visibility import VisibilityKind
from .boundary_compositing import (
    BoundarySectionAnchors,
    BoundarySemanticKind,
    BoundarySourceKind,
    QuadricBoundaryCompositingError,
    QuadricBoundaryCompositingFrame,
    QuadricBoundaryPaintFragment,
    QuadricBoundarySource,
    QuadricBoundaryVisibilitySpan,
    compute_boundary_visibility,
    compute_quadric_boundary_compositing,
)
from .boundary_section import (
    QUADRIC_BOUNDARY_SECTION_LIMITS,
    QuadricBoundarySectionLimits,
    compute_boundary_section_spans,
)
from .composite_section import (
    CompositeQuadricSectionCompositingError,
    CompositeQuadricSectionCompositingFrame,
    CompositeSectionBranchLineage,
    compute_composite_quadric_section_compositing,
)
from .compositing import QuadricPaintPolicy
from .contract import ConeModel, ConeSpec, PlaneDisplayPatchSpec, SectionPlane
from .curve_intersections import (
    ProjectedCurveIntersectionError,
    compute_projected_curve_crossings,
)
from .curves import EllipseArcCurve, ParametricConicBranch, SegmentCurve
from .global_occlusion import (
    GlobalQuadricOcclusionError,
    compute_global_quadric_frame,
)
from .manim import (
    DEFAULT_QUADRIC_VIEW,
    QUADRIC_MANIM_LIMITS,
    ProjectionInput,
    QuadricBoundaryStyle,
    QuadricManimCapacityError,
    QuadricManimError,
    QuadricManimLimits,
    QuadricManimStyle,
    _display_offset,
)
from .manim_runtime import (
    _CommittedDisplaySlot,
    _CurveSlots,
    _DirtyFrameKind,
    _ManagedQuadricDisplayGroup,
    _PreparedBoundaryFragment,
    _PreparedConeFill,
    _PreparedDash,
    _PreparedDisplayAction,
    _ResolvedParallelCameraFrame,
    _SurfaceViewCache,
    _SurfacePaintSlot,
    _apply_display_delta,
    _apply_boundary_fragment as _apply_runtime_boundary_fragment,
    _apply_surface_sheet_pair,
    _boundary_style_registry,
    _capture_root,
    _classify_dirty_frame,
    _coerce_projection_frame,
    _curve_slots_family_capacity,
    _display_digest,
    _hide_vmobject,
    _invalidate_cairo_static_image,
    _prepare_boundary_fragments,
    _prepare_display_delta,
    _prepared_cone_fill,
    _register_fixed_frame,
    _remove_fixed_frame,
    _remove_owned_identities,
    _restore_root,
    _rollback_display_transaction,
    _scene_containers,
    _set_closed_subpaths,
    _painter_band_signature,
    _projection_display_offset,
)
from .plane_patch import PlanePatchFitError, fit_plane_display_patch
from .projection import (
    ProjectionProxyError,
    ProjectionSubdivisionError,
    build_cone_projection_layers,
)
from .performance import (
    QuadricPerformanceSnapshot,
    _PerformanceAttempt,
    _performance_stage,
    quadric_performance_tracing_enabled,
)
from .section_compositing import (
    PlaneDepthRole,
    QUADRIC_SECTION_COMPOSITING_LIMITS,
    QuadricSectionCompositingError,
    QuadricSectionCompositingFrame,
    QuadricSectionCompositingLimits,
    compute_quadric_section_compositing,
    merge_quadric_plane_fragment_contours,
)
from .sections import (
    FiniteSectionBoundaryCurve,
    compute_quadric_section_boundary_curves,
)
from .surface_boundaries import (
    GeneratorBoundarySpec,
    build_surface_boundary_sources,
    plane_outline_sources,
    section_curve_boundary_source,
    surface_boundary_source_ids,
)


PlaneInput = SectionPlane | Callable[[], SectionPlane]
AnalyticCurve3D = SegmentCurve | EllipseArcCurve | ParametricConicBranch


class CompositeQuadricSectionAuthoringError(QuadricManimError):
    """The open-double coordinator was configured ambiguously."""


@dataclass(frozen=True, slots=True)
class _PreparedCompositeSurface:
    child_surface_id: str
    surface_points: np.ndarray
    cone_fill: _PreparedConeFill | None


@dataclass(frozen=True, slots=True)
class _PreparedCompositeNumeric:
    frame: CompositeQuadricSectionCompositingFrame
    surfaces: tuple[_PreparedCompositeSurface, ...]
    plane_polygons: Mapping[PlaneDepthRole, tuple[np.ndarray, ...]]
    boundary_frame: QuadricBoundaryCompositingFrame
    boundary_fragments: Mapping[str, tuple[_PreparedBoundaryFragment, ...]]
    fragment_slot_maps: Mapping[str, Mapping[str, int]]
    item_mobjects: Mapping[str, Mobject]
    draw_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ResolvedCompositeFrameInputs:
    plane: SectionPlane
    curves: tuple[AnalyticCurve3D, ...]
    lineage: tuple[CompositeSectionBranchLineage, ...]
    owners: Mapping[str, ConeSpec]
    view: ParallelView
    display_offset: tuple[float, float]
    patch: PlaneDisplayPatchSpec
    surface_view_signature: bytes
    geometry_signature: bytes
    draw_signature: bytes


@dataclass(frozen=True, slots=True)
class PreparedCompositeQuadricSectionFrame:
    numeric: _PreparedCompositeNumeric
    painter_band: PreparedPainterBand
    _performance_attempt: _PerformanceAttempt | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def frame(self) -> CompositeQuadricSectionCompositingFrame:
        return self.numeric.frame

    @property
    def boundary_frame(self) -> QuadricBoundaryCompositingFrame:
        return self.numeric.boundary_frame


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompositeQuadricSectionAuthoringError(
            f"{label} must be a non-empty string"
        )
    return value.strip()


def _nappe_role(surface: ConeSpec) -> str:
    role = surface.surface_id.rsplit(":", 1)[-1]
    if role not in {"negative", "positive"}:
        raise CompositeQuadricSectionAuthoringError(
            f"child surface {surface.surface_id!r} has no canonical nappe role"
        )
    return role


def _physical_curve_id(section_id: str, role: str, mathematical_id: str) -> str:
    prefix = f"{section_id}:"
    if not mathematical_id.startswith(prefix):
        raise CompositeQuadricSectionAuthoringError(
            "local section solver returned an identity outside the authored section"
        )
    return f"{section_id}:nappe:{role}:{mathematical_id[len(prefix):]}"


def _mathematical_branch_id(
    section_id: str,
    curve: FiniteSectionBoundaryCurve,
) -> str:
    if isinstance(curve, ParametricConicBranch):
        return f"{section_id}:component:{curve.parameterization.branch_label}"
    return curve.curve_id


class CompositeQuadricSection3D:
    """Coordinate one cutting plane across both nappes of an ``OPEN_DOUBLE``.

    ``plane`` may be a callback while the physical curve identity set remains
    fixed.  A family/component change fails before display mutation; scheduled
    topology banks remain a separate future extension.  The current contract
    is parallel projection plus Cairo and requires the two nappe projection
    interiors to be disjoint apart from their shared apex.
    """

    def __init__(
        self,
        scene: object,
        *,
        surface: ConeSpec,
        section_id: str,
        plane: PlaneInput,
        projection: ProjectionInput | None = None,
        paint_policy: QuadricPaintPolicy | str = (
            QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC
        ),
        style: QuadricManimStyle = QuadricManimStyle(),
        boundary_styles: Mapping[str, QuadricBoundaryStyle] | None = None,
        limits: QuadricManimLimits = QUADRIC_MANIM_LIMITS,
        max_chord_error: float = 1.0e-3,
        painter_z_band: tuple[float, float] = (20.0, 30.0),
        context: GeometryContext | ResolvedGeometryContext | None = None,
        coefficient_tolerance: float | None = None,
        draw_section_boundary: bool = True,
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
        display_offset: Sequence[float] = (0.0, 0.0),
    ) -> None:
        if not isinstance(surface, ConeSpec):
            raise TypeError("surface must be a ConeSpec")
        if surface.model is not ConeModel.OPEN_DOUBLE:
            raise CompositeQuadricSectionAuthoringError(
                "CompositeQuadricSection3D requires ConeModel.OPEN_DOUBLE"
            )
        if not isinstance(style, QuadricManimStyle):
            raise TypeError("style must be a QuadricManimStyle")
        if not isinstance(limits, QuadricManimLimits):
            raise TypeError("limits must be a QuadricManimLimits")
        if not isinstance(draw_section_boundary, bool):
            raise TypeError("draw_section_boundary must be a bool")
        if not isinstance(include_surface_boundaries, bool):
            raise TypeError("include_surface_boundaries must be a bool")
        if not isinstance(
            section_compositing_limits, QuadricSectionCompositingLimits
        ):
            raise TypeError(
                "section_compositing_limits must be QuadricSectionCompositingLimits"
            )
        if not isinstance(boundary_section_limits, QuadricBoundarySectionLimits):
            raise TypeError(
                "boundary_section_limits must be QuadricBoundarySectionLimits"
            )
        try:
            policy = QuadricPaintPolicy(paint_policy)
        except (TypeError, ValueError) as exc:
            raise CompositeQuadricSectionAuthoringError(
                "invalid quadric paint policy"
            ) from exc
        if (
            isinstance(max_chord_error, bool)
            or not np.isfinite(float(max_chord_error))
            or float(max_chord_error) <= 0.0
        ):
            raise ValueError("max_chord_error must be finite and positive")
        if (
            isinstance(plane_patch_margin, bool)
            or not np.isfinite(float(plane_patch_margin))
            or float(plane_patch_margin) < 0.0
        ):
            raise ValueError("plane_patch_margin must be finite and non-negative")
        if (
            isinstance(section_max_screen_error, bool)
            or not np.isfinite(float(section_max_screen_error))
            or float(section_max_screen_error) <= 0.0
        ):
            raise ValueError("section_max_screen_error must be finite and positive")
        generators = tuple(generator_boundaries)
        if not all(isinstance(item, GeneratorBoundarySpec) for item in generators):
            raise TypeError(
                "generator_boundaries must contain GeneratorBoundarySpec values"
            )

        self.scene = scene
        self.surface = surface
        self.section_id = _identity(section_id, "section_id")
        self._plane_input = plane
        self._projection_input = (
            DEFAULT_QUADRIC_VIEW if projection is None else projection
        )
        self.paint_policy = policy
        self.style = style
        self.limits = limits
        self.max_chord_error = float(max_chord_error)
        self.context = context
        self.coefficient_tolerance = coefficient_tolerance
        self.draw_section_boundary = draw_section_boundary
        self.plane_patch_margin = float(plane_patch_margin)
        self.section_max_screen_error = float(section_max_screen_error)
        self.section_compositing_limits = section_compositing_limits
        self.boundary_section_limits = boundary_section_limits
        self.include_surface_boundaries = include_surface_boundaries
        self._generator_boundaries = generators
        self.display_offset = display_offset
        self.boundary_styles = _boundary_style_registry(
            style,
            boundary_styles,
            limits,
        )
        self.children = surface.render_components
        if len(self.children) != 2:
            raise CompositeQuadricSectionAuthoringError(
                "OPEN_DOUBLE did not expand into exactly two stable children"
            )

        initial_plane = self._resolve_plane(expected_id=None)
        self._plane_id = initial_plane.plane_id
        initial_curves, initial_lineage, initial_owner = self._section_curves(
            initial_plane
        )
        self._pending_plane: SectionPlane | None = initial_plane
        self._pending_curves: tuple[AnalyticCurve3D, ...] | None = initial_curves
        self._pending_lineage: tuple[CompositeSectionBranchLineage, ...] | None = (
            initial_lineage
        )
        self._pending_owner: dict[str, ConeSpec] | None = initial_owner
        self._curve_ids = tuple(item.curve_id for item in initial_curves)
        if len(self._curve_ids) > limits.max_curves:
            raise QuadricManimCapacityError(
                f"curve count exceeds fixed limit {limits.max_curves}"
            )

        boundary_ids = surface_boundary_source_ids(
            self.children,
            generators,
            include_cap_rims=include_surface_boundaries,
            include_silhouettes=include_surface_boundaries,
        )
        plane_ids = tuple(
            f"boundary:plane:{self._plane_id}:edge:{index}" for index in range(4)
        )
        extras: list[str] = []
        if allocated_boundary_ids is not None:
            if isinstance(allocated_boundary_ids, (str, bytes)):
                raise TypeError("allocated_boundary_ids must be a sequence")
            extras = [
                _identity(item, "allocated_boundary_ids item")
                for item in allocated_boundary_ids
            ]
            if len(set(extras)) != len(extras):
                raise CompositeQuadricSectionAuthoringError(
                    "allocated_boundary_ids must be unique"
                )
        self._boundary_source_ids = tuple(
            sorted(set((*self._curve_ids, *boundary_ids, *plane_ids, *extras)))
        )
        if len(self._boundary_source_ids) > limits.max_boundary_sources:
            raise QuadricManimCapacityError(
                "boundary source count exceeds fixed limit "
                f"{limits.max_boundary_sources}"
            )
        self._child_boundary_source_ids = {
            child.surface_id: tuple(
                sorted(
                    {
                        *(
                            curve_id
                            for curve_id, owner in initial_owner.items()
                            if owner.surface_id == child.surface_id
                        ),
                        *(
                            source_id
                            for source_id in boundary_ids
                            if source_id.startswith(
                                f"boundary:{child.surface_id}:"
                            )
                        ),
                        *(
                            generator.boundary_id
                            for generator in generators
                            if generator.surface_id == child.surface_id
                        ),
                    }
                )
            )
            for child in self.children
        }
        estimated_mobjects = (
            4 * 6
            + 8
            + 1
            + len(self._boundary_source_ids)
            * _curve_slots_family_capacity(limits.max_fragments_per_curve)
            + 4
        )
        if estimated_mobjects > limits.max_total_mobjects:
            raise QuadricManimCapacityError(
                f"preallocated Mobject count {estimated_mobjects} exceeds fixed "
                f"limit {limits.max_total_mobjects}"
            )
        self._fragment_slot_maps: dict[str, dict[str, int]] = {
            source_id: {} for source_id in self._boundary_source_ids
        }
        self._curve_slots = {
            source_id: _CurveSlots(
                limits.max_fragments_per_curve,
                limits.max_dashes_per_fragment,
            )
            for source_id in self._boundary_source_ids
        }
        self._surface_sheet_slots: dict[
            str, tuple[_SurfacePaintSlot, _SurfacePaintSlot]
        ] = {
            child.surface_id: (_SurfacePaintSlot(), _SurfacePaintSlot())
            for child in self.children
        }
        prefix = f"section-compositor:{self._plane_id}"
        self._plane_item_ids = {
            PlaneDepthRole.BEHIND_SURFACE: f"{prefix}:plane:behind",
            PlaneDepthRole.OUTSIDE_PROJECTION: f"{prefix}:plane:outside",
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS: f"{prefix}:plane:between",
            PlaneDepthRole.IN_FRONT_OF_SURFACE: f"{prefix}:plane:front",
        }
        self._plane_outline_anchor_ids = {
            PlaneDepthRole.BEHIND_SURFACE: f"{prefix}:plane:outline:behind",
            PlaneDepthRole.OUTSIDE_PROJECTION: f"{prefix}:plane:outline:outside",
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS: f"{prefix}:plane:outline:between",
            PlaneDepthRole.IN_FRONT_OF_SURFACE: f"{prefix}:plane:outline:front",
        }
        self._plane_slots = {
            item_id: VMobject()
            for item_id in (
                *self._plane_item_ids.values(),
                *self._plane_outline_anchor_ids.values(),
            )
        }
        for slot in self._plane_slots.values():
            _hide_vmobject(slot)

        surface_root = VGroup(
            *(
                slot.root
                for child in self.children
                for slot in self._surface_sheet_slots[child.surface_id]
            )
        )
        plane_root = VGroup(*self._plane_slots.values())
        curve_root = VGroup(
            *(self._curve_slots[key].root for key in self._boundary_source_ids)
        )
        self._opacity_sentinel = Line((0, 0, 0), (1.0e-9, 0, 0), buff=0)
        self._opacity_sentinel.set_stroke(width=0.0, opacity=1.0)
        self.root = _ManagedQuadricDisplayGroup(
            surface_root,
            plane_root,
            curve_root,
            opacity_sentinel=self._opacity_sentinel,
        )
        self._update_driver = Mobject()

        def update_display(mobject: Mobject, dt: float) -> None:
            del mobject
            if self._attached:
                self.update(dt)

        self._update_driver.add_updater(update_display)
        self._band = ManagedPainterBand(
            z_band=painter_z_band,
            managed_roots=(self.root,),
        )
        self._attached = False
        self._fixed_frame_camera: ThreeDCamera | None = None
        self._last_frame: CompositeQuadricSectionCompositingFrame | None = None
        self._last_boundary_frame: QuadricBoundaryCompositingFrame | None = None
        self._last_lineage: tuple[CompositeSectionBranchLineage, ...] = ()
        self._performance_enabled = quadric_performance_tracing_enabled()
        self._performance_frame_index = 0
        self._last_performance_snapshot: QuadricPerformanceSnapshot | None = None
        self._display_slot_state: dict[str, _CommittedDisplaySlot] = {}
        self._last_painter_band_signature: tuple[
            tuple[str, int, float], ...
        ] = ()
        self._last_input_geometry_signature: bytes | None = None
        self._last_input_draw_signature: bytes | None = None
        self._last_input_opacity: float | None = None
        self._last_prepared_frame: PreparedCompositeQuadricSectionFrame | None = (
            None
        )
        self._last_prepared_performance_counts: dict[str, int] = {}
        self._surface_view_cache = _SurfaceViewCache()

    def _resolve_plane(self, *, expected_id: str | None = None) -> SectionPlane:
        value = (
            self._plane_input()
            if callable(self._plane_input)
            else self._plane_input
        )
        if not isinstance(value, SectionPlane):
            raise CompositeQuadricSectionAuthoringError(
                "plane must resolve to a SectionPlane"
            )
        expected = (
            getattr(self, "_plane_id", None)
            if expected_id is None
            else expected_id
        )
        if expected is not None and value.plane_id != expected:
            raise CompositeQuadricSectionAuthoringError(
                "plane identity changed while CompositeQuadricSection3D was active"
            )
        return value

    def _resolve_projection_frame(self) -> _ResolvedParallelCameraFrame:
        value = (
            self._projection_input(self.scene)
            if callable(self._projection_input)
            else self._projection_input
        )
        return _coerce_projection_frame(value, scene=self.scene)

    def _resolve_view(
        self,
        projection_frame: _ResolvedParallelCameraFrame | None = None,
    ) -> ParallelView:
        """Resolve the linear kernel view, preserving the legacy helper."""

        if projection_frame is None:
            projection_frame = self._resolve_projection_frame()
        return projection_frame.view

    def _section_curves(
        self,
        plane: SectionPlane,
    ) -> tuple[
        tuple[AnalyticCurve3D, ...],
        tuple[CompositeSectionBranchLineage, ...],
        dict[str, ConeSpec],
    ]:
        if not self.draw_section_boundary:
            return (), (), {}
        curves: list[AnalyticCurve3D] = []
        lineage: list[CompositeSectionBranchLineage] = []
        owner: dict[str, ConeSpec] = {}
        for child in self.children:
            role = _nappe_role(child)
            local = compute_quadric_section_boundary_curves(
                self.section_id,
                child,
                plane,
                context=self.context,
                coefficient_tolerance=self.coefficient_tolerance,
            )
            for curve in local:
                physical_id = _physical_curve_id(
                    self.section_id, role, curve.curve_id
                )
                physical = replace(curve, curve_id=physical_id)
                curves.append(physical)
                owner[physical_id] = child
                lineage.append(
                    CompositeSectionBranchLineage(
                        physical_id,
                        _mathematical_branch_id(self.section_id, curve),
                        child.surface_id,
                        role,
                    )
                )
        curves.sort(key=lambda item: item.curve_id)
        lineage.sort(key=lambda item: item.physical_curve_id)
        return tuple(curves), tuple(lineage), owner

    def _resolve_frame_inputs(
        self,
    ) -> _ResolvedCompositeFrameInputs:
        if self._pending_plane is not None:
            plane = self._pending_plane
            curves = self._pending_curves or ()
            lineage = self._pending_lineage or ()
            owner = self._pending_owner or {}
            self._pending_plane = None
            self._pending_curves = None
            self._pending_lineage = None
            self._pending_owner = None
        else:
            plane = self._resolve_plane(expected_id=self._plane_id)
            curves, lineage, owner = self._section_curves(plane)
        self._validate_curve_topology(curves)
        projection_frame = self._resolve_projection_frame()
        view = self._resolve_view(projection_frame)
        display_offset = _projection_display_offset(
            self.scene,
            projection_frame,
            self.display_offset,
        )
        patch = self._fit_patch(plane)
        surface_view_signature = _display_digest(
            "composite-quadric-surface-view-v1",
            self.surface,
            self.children,
            view.matrix,
            self.context,
            self.max_chord_error,
            self.limits.max_surface_segments,
        )
        geometry_signature = _display_digest(
            "composite-quadric-frame-inputs-v1",
            surface_view_signature,
            plane,
            curves,
            lineage,
            {curve_id: surface.surface_id for curve_id, surface in owner.items()},
            patch,
            self.coefficient_tolerance,
            self.draw_section_boundary,
            self.plane_patch_margin,
            self.paint_policy,
            self.style,
            self.boundary_styles,
            self.limits,
            self.max_chord_error,
            self.section_max_screen_error,
            self.section_compositing_limits,
            self.boundary_section_limits,
            self.include_surface_boundaries,
            self._generator_boundaries,
            display_offset,
        )
        return _ResolvedCompositeFrameInputs(
            plane,
            curves,
            lineage,
            dict(owner),
            view,
            display_offset,
            patch,
            surface_view_signature,
            geometry_signature,
            _display_digest("composite-quadric-frame-draw-v1"),
        )

    @property
    def display_offset(self) -> tuple[float, float]:
        """Return the validated display-only screen translation."""

        return self._display_offset

    @display_offset.setter
    def display_offset(self, value: Sequence[float]) -> None:
        self._display_offset = _display_offset(value)

    def _validate_curve_topology(
        self,
        curves: Sequence[AnalyticCurve3D],
    ) -> None:
        ids = tuple(item.curve_id for item in curves)
        if ids != self._curve_ids:
            raise QuadricManimCapacityError(
                "open-double section curve identities changed after fixed-capacity "
                "allocation; use a future scheduled composite transition for a "
                "topology-family change"
            )

    def _fit_patch(
        self,
        plane: SectionPlane,
    ) -> PlaneDisplayPatchSpec:
        try:
            return fit_plane_display_patch(
                f"{plane.plane_id}:auto-display-patch",
                plane,
                self.children,
                margin_ratio=self.plane_patch_margin,
            ).patch
        except PlanePatchFitError as exc:
            raise CompositeQuadricSectionAuthoringError(
                f"automatic section-plane patch fitting failed: {exc}"
            ) from exc

    def _local_frames(
        self,
        plane: SectionPlane,
        patch: PlaneDisplayPatchSpec,
        view: ParallelView,
        surface_view_signature: bytes,
        performance_attempt: _PerformanceAttempt | None = None,
    ) -> tuple[QuadricSectionCompositingFrame, ...]:
        try:
            with _performance_stage(
                performance_attempt, "surface_proxy_global_frame"
            ):
                hit, cached = self._surface_view_cache.lookup(
                    "composite_surface_bases",
                    surface_view_signature,
                )
                if hit:
                    if performance_attempt is not None:
                        performance_attempt.cache_hit("surface_view_base")
                    bases = cached  # type: ignore[assignment]
                else:
                    if performance_attempt is not None:
                        performance_attempt.cache_miss("surface_view_base")
                    bases = tuple(
                        compute_global_quadric_frame(
                            (),
                            (child,),
                            view,
                            context=self.context,
                            paint_policy=QuadricPaintPolicy.PHYSICAL,
                            max_chord_error=self.max_chord_error,
                            max_segments=self.limits.max_surface_segments,
                        )
                        for child in self.children
                    )
                    self._surface_view_cache.store(
                        "composite_surface_bases",
                        surface_view_signature,
                        bases,
                    )
            result = []
            for child, base in zip(self.children, bases, strict=True):
                with _performance_stage(performance_attempt, "section_compositing"):
                    local = compute_quadric_section_compositing(
                        base.frame,
                        child,
                        plane,
                        patch,
                        view,
                        context=self.context,
                        max_screen_error=self.section_max_screen_error,
                        limits=self.section_compositing_limits,
                    )
                result.append(local)
        except (
            GlobalQuadricOcclusionError,
            ProjectionProxyError,
            ProjectionSubdivisionError,
            QuadricSectionCompositingError,
        ) as exc:
            raise CompositeQuadricSectionAuthoringError(
                f"local nappe section preparation failed: {exc}"
            ) from exc
        return tuple(result)

    @staticmethod
    def _plane_outline_visibility(
        frame: CompositeQuadricSectionCompositingFrame,
    ) -> dict[str, tuple[QuadricBoundaryVisibilitySpan, ...]]:
        grouped: dict[str, list[QuadricBoundaryVisibilitySpan]] = {
            f"boundary:plane:{frame.plane.plane_id}:edge:{index}": []
            for index in range(4)
        }
        for fragment in frame.plane_outline_fragments:
            visible = fragment.role in {
                PlaneDepthRole.OUTSIDE_PROJECTION,
                PlaneDepthRole.IN_FRONT_OF_SURFACE,
            }
            grouped[
                f"boundary:plane:{frame.plane.plane_id}:edge:{fragment.edge_index}"
            ].append(
                QuadricBoundaryVisibilitySpan(
                    fragment.interval,
                    VisibilityKind.VISIBLE if visible else VisibilityKind.HIDDEN,
                    (),
                    fragment.role.value,
                )
            )
        return {
            source_id: tuple(sorted(values, key=lambda item: item.interval.start))
            for source_id, values in grouped.items()
        }

    def _boundary_sources(
        self,
        curves: Sequence[AnalyticCurve3D],
        owners: Mapping[str, ConeSpec],
        plane: SectionPlane,
        patch: PlaneDisplayPatchSpec,
        view: ParallelView,
        *,
        surface_sources: Sequence[QuadricBoundarySource] | None = None,
    ) -> tuple[QuadricBoundarySource, ...]:
        result = [
            section_curve_boundary_source(
                curve,
                owners[curve.curve_id],
                plane,
                context=self.context,
                style_id="style:curve",
            )
            for curve in curves
        ]
        if surface_sources is None:
            surface_sources = self._surface_boundary_sources(view)
        result.extend(surface_sources)
        result.extend(plane_outline_sources(plane, patch))
        result.sort(key=lambda item: item.source_id)
        ids = tuple(item.source_id for item in result)
        if len(set(ids)) != len(ids):
            raise CompositeQuadricSectionAuthoringError(
                "composite semantic boundaries have duplicate identities"
            )
        unknown = sorted(set(ids) - set(self._boundary_source_ids))
        if unknown:
            raise QuadricManimCapacityError(
                "semantic boundary identities were not preallocated: "
                + ", ".join(unknown)
            )
        return tuple(result)

    def _surface_boundary_sources(
        self,
        view: ParallelView,
    ) -> tuple[QuadricBoundarySource, ...]:
        if self.include_surface_boundaries:
            return build_surface_boundary_sources(
                self.children,
                view,
                self._generator_boundaries,
                include_cap_rims=True,
                include_silhouettes=True,
            )
        if self._generator_boundaries:
            return build_surface_boundary_sources(
                self.children,
                view,
                self._generator_boundaries,
                include_cap_rims=False,
                include_silhouettes=False,
            )
        return ()

    def _boundary_crossings(
        self,
        sources: Sequence[QuadricBoundarySource],
        spans: Mapping[str, Sequence[QuadricBoundaryVisibilitySpan]],
        view: ParallelView,
        *,
        cached_source_ids: frozenset[str] = frozenset(),
        cached_crossings: Sequence[object] = (),
    ) -> tuple[object, ...]:
        result = list(cached_crossings)
        child_ids = {child.surface_id for child in self.children}
        for first, second in combinations(sources, 2):
            if (
                first.source_id in cached_source_ids
                and second.source_id in cached_source_ids
            ):
                continue
            active_intervals = None
            if self.paint_policy is QuadricPaintPolicy.PHYSICAL:
                active_intervals = {
                    source.source_id: tuple(
                        span.interval
                        for span in spans[source.source_id]
                        if span.kind is VisibilityKind.VISIBLE
                    )
                    for source in (first, second)
                }
            try:
                result.extend(
                    compute_projected_curve_crossings(
                        (first.curve, second.curve),
                        view,
                        context=self.context,
                        active_intervals=active_intervals,
                    )
                )
            except ProjectedCurveIntersectionError as exc:
                same_owner = (
                    first.owner_surface_id is not None
                    and first.owner_surface_id == second.owner_surface_id
                )
                sibling_owner = {
                    first.owner_surface_id,
                    second.owner_surface_id,
                } == child_ids
                ordinary_surface_ink = {
                    first.semantic_kind,
                    second.semantic_kind,
                } <= {
                    BoundarySemanticKind.SURFACE_BOUNDARY,
                    BoundarySemanticKind.TRUE_SILHOUETTE,
                }
                if (same_owner or sibling_owner) and ordinary_surface_ink:
                    # Same-owner and shared-apex sibling silhouettes can have
                    # coincident analytic supports while their finite domains
                    # merely meet at a certified endpoint. Their local/parent
                    # painter anchors already own the stroke order.
                    continue
                raise CompositeQuadricSectionAuthoringError(
                    "semantic boundary crossings cannot be certified: "
                    f"{first.source_id!r}, {second.source_id!r}: {exc}"
                ) from exc
        by_id = {item.crossing_id: item for item in result}
        return tuple(by_id[key] for key in sorted(by_id))

    def _prepare_static_surface_boundaries(
        self,
        view: ParallelView,
        surface_view_signature: bytes,
        performance_attempt: _PerformanceAttempt | None,
    ) -> tuple[
        tuple[QuadricBoundarySource, ...],
        Mapping[str, tuple[QuadricBoundaryVisibilitySpan, ...]],
        tuple[object, ...],
    ]:
        signature = _display_digest(
            "composite-static-surface-boundaries-v1",
            surface_view_signature,
            self.include_surface_boundaries,
            self._generator_boundaries,
            self.paint_policy,
        )
        hit, cached = self._surface_view_cache.lookup(
            "static_boundaries",
            signature,
        )
        if hit:
            if performance_attempt is not None:
                performance_attempt.cache_hit("static_surface_boundaries")
            return cached  # type: ignore[return-value]
        if performance_attempt is not None:
            performance_attempt.cache_miss("static_surface_boundaries")
        sources = self._surface_boundary_sources(view)
        with _performance_stage(performance_attempt, "boundary_visibility"):
            spans = compute_boundary_visibility(
                sources,
                self.children,
                view,
                context=self.context,
            )
        with _performance_stage(performance_attempt, "curve_crossings"):
            crossings = self._boundary_crossings(sources, spans, view)
        prepared = (sources, dict(spans), crossings)
        self._surface_view_cache.store(
            "static_boundaries",
            signature,
            prepared,
        )
        return prepared

    def _anchors_by_surface(
        self,
        frame: CompositeQuadricSectionCompositingFrame,
    ) -> dict[str, BoundarySectionAnchors]:
        items = frame.paint_items
        return {
            sheet.child_surface_id: BoundarySectionAnchors(
                items.plane_behind,
                items.plane_outline_behind,
                sheet.surface_back,
                items.plane_outside,
                items.plane_outline_outside,
                items.plane_between,
                items.plane_outline_between,
                sheet.surface_front,
                items.plane_front,
                items.plane_outline_front,
            )
            for sheet in items.surface_sheets
        }

    def _boundary_style_for_source(
        self,
        source: QuadricBoundarySource,
    ) -> QuadricBoundaryStyle:
        style_id = source.style_id
        if style_id is None:
            if source.semantic_kind is BoundarySemanticKind.DISPLAY_FRAME:
                style_id = "style:section-outline"
            elif source.semantic_kind is BoundarySemanticKind.TRUE_SILHOUETTE:
                style_id = "style:surface-silhouette"
            elif source.semantic_kind is BoundarySemanticKind.SURFACE_BOUNDARY:
                style_id = "style:surface-boundary"
            else:
                style_id = "style:curve"
        try:
            return self.boundary_styles[style_id]
        except KeyError as exc:
            raise CompositeQuadricSectionAuthoringError(
                f"unknown boundary style {style_id!r}"
            ) from exc

    def _prepare_cone_fill(
        self,
        surface: ConeSpec,
        view: ParallelView,
    ) -> _PreparedConeFill | None:
        if not self.style.cone_component_shading:
            return None
        try:
            return _prepared_cone_fill(
                build_cone_projection_layers(
                    surface,
                    view,
                    max_chord_error=self.max_chord_error,
                    max_segments=self.limits.max_surface_segments,
                )
            )
        except ProjectionSubdivisionError as exc:
            raise QuadricManimCapacityError(str(exc)) from exc
        except ProjectionProxyError as exc:
            raise CompositeQuadricSectionAuthoringError(
                f"cone component shading failed: {exc}"
            ) from exc

    def _prepare_surface_component_fills(
        self,
        view: ParallelView,
        surface_view_signature: bytes,
        performance_attempt: _PerformanceAttempt | None,
    ) -> dict[str, _PreparedConeFill | None]:
        signature = _display_digest(
            "composite-surface-component-fills-v1",
            surface_view_signature,
            self.style.cone_component_shading,
        )
        with _performance_stage(performance_attempt, "surface_component_fill"):
            hit, cached = self._surface_view_cache.lookup(
                "component_fills",
                signature,
            )
            if hit:
                if performance_attempt is not None:
                    performance_attempt.cache_hit("surface_component_fill")
                return dict(cached)  # type: ignore[arg-type]
            if performance_attempt is not None:
                performance_attempt.cache_miss("surface_component_fill")
            prepared = tuple(
                (child.surface_id, self._prepare_cone_fill(child, view))
                for child in self.children
            )
            self._surface_view_cache.store(
                "component_fills",
                signature,
                prepared,
            )
            return dict(prepared)

    def _prepare_numeric(
        self,
        performance_attempt: _PerformanceAttempt | None = None,
        resolved_inputs: _ResolvedCompositeFrameInputs | None = None,
    ) -> _PreparedCompositeNumeric:
        if resolved_inputs is None:
            with _performance_stage(performance_attempt, "resolve_inputs"):
                resolved_inputs = self._resolve_frame_inputs()
        plane = resolved_inputs.plane
        curves = resolved_inputs.curves
        lineage = resolved_inputs.lineage
        owners = resolved_inputs.owners
        view = resolved_inputs.view
        patch = resolved_inputs.patch
        child_frames = self._local_frames(
            plane,
            patch,
            view,
            resolved_inputs.surface_view_signature,
            performance_attempt,
        )
        component_fills = self._prepare_surface_component_fills(
            view,
            resolved_inputs.surface_view_signature,
            performance_attempt,
        )
        try:
            with _performance_stage(performance_attempt, "section_compositing"):
                frame = compute_composite_quadric_section_compositing(
                    self.surface,
                    self.section_id,
                    child_frames,
                    lineage,
                    max_plane_fragments=(
                        self.section_compositing_limits.max_plane_fragments
                    ),
                )
            with _performance_stage(performance_attempt, "contour_union"):
                contours = merge_quadric_plane_fragment_contours(
                    frame.plane,
                    frame.patch,
                    child_frames[0].base_frame.visibility.projection_matrix,
                    frame.plane_fragments,
                )
        except (
            CompositeQuadricSectionCompositingError,
            QuadricSectionCompositingError,
        ) as exc:
            raise CompositeQuadricSectionAuthoringError(
                f"composite open-double section preparation failed: {exc}"
            ) from exc

        prepared_surface_values = []
        for child in self.children:
            prepared_surface_values.append(
                _PreparedCompositeSurface(
                    child.surface_id,
                    np.asarray(
                        [
                            (x, y, 0.0)
                            for x, y in frame.child_frame(
                                child.surface_id
                            ).surface_proxy.boundary_points
                        ],
                        dtype=float,
                    ),
                    component_fills[child.surface_id],
                )
            )
        prepared_surfaces = tuple(prepared_surface_values)
        plane_polygons = {
            role: tuple(
                np.asarray([(x, y, 0.0) for x, y in contour], dtype=float)
                for contour in contours[role]
            )
            for role in PlaneDepthRole
        }

        static_sources, static_spans, static_crossings = (
            self._prepare_static_surface_boundaries(
                view,
                resolved_inputs.surface_view_signature,
                performance_attempt,
            )
        )
        static_source_ids = frozenset(item.source_id for item in static_sources)
        with _performance_stage(performance_attempt, "boundary_visibility"):
            sources = self._boundary_sources(
                curves,
                owners,
                plane,
                patch,
                view,
                surface_sources=static_sources,
            )
            non_plane = tuple(
                item
                for item in sources
                if item.source_kind is not BoundarySourceKind.PLANE_PATCH_EDGE
            )
            dynamic_non_plane = tuple(
                item
                for item in non_plane
                if item.source_id not in static_source_ids
            )
            try:
                spans = dict(static_spans)
                if dynamic_non_plane:
                    spans.update(
                        compute_boundary_visibility(
                            dynamic_non_plane,
                            self.children,
                            view,
                            context=self.context,
                        )
                    )
            except Exception as exc:
                raise CompositeQuadricSectionAuthoringError(
                    f"semantic boundary visibility failed: {exc}"
                ) from exc
            spans.update(self._plane_outline_visibility(frame))
        with _performance_stage(performance_attempt, "curve_crossings"):
            crossings = self._boundary_crossings(
                sources,
                spans,
                view,
                cached_source_ids=static_source_ids,
                cached_crossings=static_crossings,
            )
        section_spans: dict[str, tuple[object, ...]] = {}
        with _performance_stage(performance_attempt, "boundary_section_spans"):
            for child in self.children:
                owned = tuple(
                    source
                    for source in non_plane
                    if source.owner_surface_id == child.surface_id
                )
                if not owned:
                    continue
                try:
                    section_spans.update(
                        compute_boundary_section_spans(
                            owned,
                            frame.child_frame(child.surface_id),
                            view,
                            crossings,
                            surface=child,
                            visibility_spans_by_source=spans,
                            context=self.context,
                            limits=self.boundary_section_limits,
                        )
                    )
                except QuadricBoundaryCompositingError as exc:
                    raise CompositeQuadricSectionAuthoringError(
                        f"boundary/section placement failed: {exc}"
                    ) from exc
        anchors = self._anchors_by_surface(frame)
        surface_item_by_id = {
            item.child_surface_id: item.surface_front
            for item in frame.paint_items.surface_sheets
        }
        try:
            with _performance_stage(
                performance_attempt, "boundary_painter_graph"
            ):
                boundary_frame = compute_quadric_boundary_compositing(
                    sources,
                    spans,
                    paint_policy=self.paint_policy,
                    parent_item_ids=frame.draw_order,
                    parent_relations=frame.order_relations,
                    surface_item_by_id=surface_item_by_id,
                    crossings=crossings,
                    section_anchors_by_surface=anchors,
                    section_spans_by_source=section_spans,
                )
        except QuadricBoundaryCompositingError as exc:
            raise CompositeQuadricSectionAuthoringError(
                f"semantic boundary painter graph failed: {exc}"
            ) from exc

        item_mobjects: dict[str, Mobject] = {}
        item_mobjects.update(self._plane_slots)
        for sheet in frame.paint_items.surface_sheets:
            back, front = self._surface_sheet_slots[sheet.child_surface_id]
            item_mobjects[sheet.surface_back] = back.root
            item_mobjects[sheet.surface_front] = front.root
        boundary_batch = _prepare_boundary_fragments(
            sources=sources,
            frame=boundary_frame,
            view=view,
            style_for_source=self._boundary_style_for_source,
            previous_slot_maps=self._fragment_slot_maps,
            curve_slots=self._curve_slots,
            slot_source_ids=self._boundary_source_ids,
            max_chord_error=self.max_chord_error,
            limits=self.limits,
            performance_attempt=performance_attempt,
        )
        item_mobjects.update(boundary_batch.item_mobjects)
        if set(item_mobjects) != set(boundary_frame.draw_order):
            raise CompositeQuadricSectionAuthoringError(
                "prepared Mobjects do not cover the composite boundary draw order"
            )
        if performance_attempt is not None:
            performance_attempt.set_count("surface_count", len(self.children))
            performance_attempt.set_count("curve_count", len(curves))
            performance_attempt.set_count("boundary_source_count", len(sources))
            performance_attempt.set_count(
                "boundary_fragment_count", len(boundary_frame.fragments)
            )
            performance_attempt.set_count(
                "painted_boundary_fragment_count",
                len(boundary_frame.painted_fragments),
            )
            performance_attempt.set_count(
                "plane_fragment_count", len(frame.plane_fragments)
            )
            performance_attempt.set_count(
                "ray_classification_count",
                sum(item.ray_classification_count for item in child_frames),
            )
        numeric = _PreparedCompositeNumeric(
            frame,
            prepared_surfaces,
            plane_polygons,
            boundary_frame,
            boundary_batch.fragments,
            boundary_batch.fragment_slot_maps,
            item_mobjects,
            boundary_frame.draw_order,
        )
        return self._translate_prepared_numeric(
            numeric,
            resolved_inputs.display_offset,
        )

    def _translate_prepared_numeric(
        self,
        numeric: _PreparedCompositeNumeric,
        display_offset: tuple[float, float],
    ) -> _PreparedCompositeNumeric:
        """Apply the shared affine-camera translation to display paths only."""

        if display_offset == (0.0, 0.0):
            return numeric
        delta = np.asarray((*display_offset, 0.0), dtype=float)

        def points(value: np.ndarray) -> np.ndarray:
            return np.asarray(value, dtype=float) + delta

        def dashes(
            values: tuple[_PreparedDash, ...],
        ) -> tuple[_PreparedDash, ...]:
            return tuple(replace(item, points=points(item.points)) for item in values)

        def cone_fill(value: _PreparedConeFill | None) -> _PreparedConeFill | None:
            if value is None:
                return None
            return replace(
                value,
                opaque_lateral_paths=tuple(
                    points(item) for item in value.opaque_lateral_paths
                ),
                opaque_cap_paths=tuple(
                    points(item) for item in value.opaque_cap_paths
                ),
                back_lateral_paths=tuple(
                    points(item) for item in value.back_lateral_paths
                ),
                back_cap_paths=tuple(points(item) for item in value.back_cap_paths),
                front_lateral_paths=tuple(
                    points(item) for item in value.front_lateral_paths
                ),
                front_cap_paths=tuple(
                    points(item) for item in value.front_cap_paths
                ),
            )

        return replace(
            numeric,
            surfaces=tuple(
                replace(
                    item,
                    surface_points=points(item.surface_points),
                    cone_fill=cone_fill(item.cone_fill),
                )
                for item in numeric.surfaces
            ),
            plane_polygons={
                role: tuple(points(item) for item in values)
                for role, values in numeric.plane_polygons.items()
            },
            boundary_fragments={
                source_id: tuple(
                    replace(
                        item,
                        points=points(item.points),
                        dashes=dashes(item.dashes),
                    )
                    for item in values
                )
                for source_id, values in numeric.boundary_fragments.items()
            },
        )

    def _scene_containers(self) -> tuple[list[object], ...]:
        return _scene_containers(self.scene)

    def _prepare_painter(
        self,
        numeric: _PreparedCompositeNumeric,
        performance_attempt: _PerformanceAttempt | None = None,
    ) -> PreparedCompositeQuadricSectionFrame:
        try:
            with _performance_stage(
                performance_attempt, "painter_band_preparation"
            ):
                self._band.configure(
                    containers=self._scene_containers(),
                    sources={"composite-section:reservation": self._update_driver},
                )
                band = self._band.prepare(
                    draw_order=numeric.draw_order,
                    item_mobjects=numeric.item_mobjects,
                )
        except ManagedPainterBandError as exc:
            raise CompositeQuadricSectionAuthoringError(str(exc)) from exc
        return PreparedCompositeQuadricSectionFrame(
            numeric,
            band,
            performance_attempt,
        )

    def _new_performance_attempt(self) -> _PerformanceAttempt | None:
        if not self._performance_enabled:
            return None
        attempt = _PerformanceAttempt(
            "composite_quadric_section_3d",
            self._performance_frame_index,
        )
        self._performance_frame_index += 1
        return attempt

    def _finish_performance_attempt(
        self,
        attempt: _PerformanceAttempt | None,
        *,
        status: str,
        rollback_performed: bool = False,
        error: BaseException | None = None,
    ) -> None:
        if attempt is None or attempt.finished:
            return
        self._last_performance_snapshot = attempt.finish(
            status=status,
            rollback_performed=rollback_performed,
            error=error,
        )

    def _prepare_with_performance(self) -> PreparedCompositeQuadricSectionFrame:
        attempt = self._new_performance_attempt()
        try:
            with _performance_stage(attempt, "resolve_inputs"):
                resolved = self._resolve_frame_inputs()
            numeric = self._prepare_numeric(attempt, resolved)
            return self._prepare_painter(numeric, attempt)
        except Exception as exc:
            self._finish_performance_attempt(
                attempt,
                status="failed",
                error=exc,
            )
            raise

    def _seed_cached_performance_counts(
        self,
        attempt: _PerformanceAttempt | None,
    ) -> None:
        if attempt is None:
            return
        for name, value in self._last_prepared_performance_counts.items():
            attempt.set_count(name, value)

    def _validate_cached_painter_band(
        self,
        attempt: _PerformanceAttempt | None,
    ) -> None:
        try:
            with _performance_stage(attempt, "painter_band_preparation"):
                self._band.configure(
                    containers=self._scene_containers(),
                    sources={
                        "composite-section:reservation": self._update_driver
                    },
                )
        except ManagedPainterBandError as exc:
            raise CompositeQuadricSectionAuthoringError(str(exc)) from exc

    def _commit_input_cache(
        self,
        resolved: _ResolvedCompositeFrameInputs,
        opacity: float,
        prepared: PreparedCompositeQuadricSectionFrame,
    ) -> None:
        self._last_input_geometry_signature = resolved.geometry_signature
        self._last_input_draw_signature = resolved.draw_signature
        self._last_input_opacity = opacity
        if prepared._performance_attempt is not None:
            self._last_prepared_performance_counts = dict(
                prepared._performance_attempt.counts
            )
        self._last_prepared_frame = replace(
            prepared,
            _performance_attempt=None,
        )

    def prepare(self) -> PreparedCompositeQuadricSectionFrame:
        if not self._attached:
            raise CompositeQuadricSectionAuthoringError(
                "composite section controller is not attached"
            )
        return self._prepare_with_performance()

    def _apply_surface_pair(
        self,
        prepared: _PreparedCompositeSurface,
        opacity: float,
    ) -> None:
        back, front = self._surface_sheet_slots[prepared.child_surface_id]
        _apply_surface_sheet_pair(
            back,
            front,
            prepared.surface_points,
            prepared.cone_fill,
            self.style,
            opacity,
            configure_front_stroke=False,
            draw_front_stroke=False,
        )

    def _apply_boundary_fragment(
        self,
        source_id: str,
        prepared: _PreparedBoundaryFragment,
        opacity: float,
    ) -> None:
        _apply_runtime_boundary_fragment(
            self._curve_slots,
            source_id,
            prepared,
            opacity,
        )

    def _apply_plane_roles(
        self,
        numeric: _PreparedCompositeNumeric,
        opacity: float,
    ) -> None:
        for role, item_id in self._plane_item_ids.items():
            slot = self._plane_slots[item_id]
            _set_closed_subpaths(slot, numeric.plane_polygons[role])
            slot.set_fill(
                color=self.style.section_plane_fill_color,
                opacity=self.style.section_plane_fill_opacity * opacity,
            )
            slot.set_stroke(opacity=0.0)

    def _prepare_display_actions(
        self,
        prepared: PreparedCompositeQuadricSectionFrame,
        opacity: float,
    ) -> tuple[_PreparedDisplayAction, ...]:
        actions: list[_PreparedDisplayAction] = []
        for surface in prepared.numeric.surfaces:
            slots = self._surface_sheet_slots[surface.child_surface_id]
            actions.append(
                _PreparedDisplayAction(
                    f"surface-pair:{surface.child_surface_id}",
                    tuple(slot.root for slot in slots),
                    _display_digest(
                        "composite-surface-pair",
                        surface.surface_points,
                        surface.cone_fill,
                        self.style,
                        opacity,
                    ),
                    partial(self._apply_surface_pair, surface, opacity),
                )
            )

        actions.append(
            _PreparedDisplayAction(
                "section:plane-roles",
                tuple(self._plane_slots.values()),
                _display_digest(
                    "composite-plane-roles",
                    prepared.numeric.plane_polygons,
                    self.style,
                    opacity,
                ),
                partial(self._apply_plane_roles, prepared.numeric, opacity),
            )
        )

        for source_id, fragments in prepared.numeric.boundary_fragments.items():
            for fragment in fragments:
                slot = self._curve_slots[source_id].fragments[fragment.slot_index]
                actions.append(
                    _PreparedDisplayAction(
                        f"path:{source_id}:{fragment.slot_index}",
                        (slot.root,),
                        _display_digest(
                            "boundary",
                            fragment.fragment.render_intent,
                            fragment.points,
                            tuple(item.points for item in fragment.dashes),
                            fragment.style,
                            opacity,
                        ),
                        partial(
                            self._apply_boundary_fragment,
                            source_id,
                            fragment,
                            opacity,
                        ),
                    )
                )
        return tuple(actions)

    def apply(self, prepared: PreparedCompositeQuadricSectionFrame) -> None:
        if not self._attached:
            raise CompositeQuadricSectionAuthoringError(
                "composite section controller is not attached"
            )
        if not isinstance(prepared, PreparedCompositeQuadricSectionFrame):
            raise TypeError(
                "prepared must be a PreparedCompositeQuadricSectionFrame"
            )

        def capture_controller_state() -> tuple[
            dict[str, dict[str, int]],
            CompositeQuadricSectionCompositingFrame | None,
            QuadricBoundaryCompositingFrame | None,
            tuple[CompositeSectionBranchLineage, ...],
            dict[str, _CommittedDisplaySlot],
            tuple[tuple[str, int, float], ...],
        ]:
            return (
                {
                    source_id: dict(values)
                    for source_id, values in self._fragment_slot_maps.items()
                },
                self._last_frame,
                self._last_boundary_frame,
                self._last_lineage,
                dict(self._display_slot_state),
                self._last_painter_band_signature,
            )

        def restore_controller_state(
            state: tuple[
                dict[str, dict[str, int]],
                CompositeQuadricSectionCompositingFrame | None,
                QuadricBoundaryCompositingFrame | None,
                tuple[CompositeSectionBranchLineage, ...],
                dict[str, _CommittedDisplaySlot],
                tuple[tuple[str, int, float], ...],
            ],
        ) -> None:
            (
                self._fragment_slot_maps,
                self._last_frame,
                self._last_boundary_frame,
                self._last_lineage,
                self._display_slot_state,
                self._last_painter_band_signature,
            ) = state

        attempt = prepared._performance_attempt
        if attempt is None or attempt.finished:
            attempt = self._new_performance_attempt()
        opacity = self.root.opacity_multiplier
        try:
            actions = self._prepare_display_actions(prepared, opacity)
            delta = _prepare_display_delta(self._display_slot_state, actions)
            painter_signature = _painter_band_signature(prepared.painter_band)
            painter_changed = (
                painter_signature != self._last_painter_band_signature
            )
            painter_items = (
                self._band.changed_items(prepared.painter_band)
                if painter_changed
                else ()
            )
            painter_roots = (
                tuple(item.mobject for item in painter_items)
            )
            mutation_roots = (*delta.mutation_roots, *painter_roots)
            if attempt is not None:
                attempt.set_count("display_active_slot_count", len(actions))
                attempt.set_count(
                    "display_changed_slot_count", len(delta.changed)
                )
                attempt.set_count(
                    "display_unchanged_slot_count",
                    len(delta.unchanged_slot_ids),
                )
                attempt.set_count("display_hidden_slot_count", len(delta.hidden))
                attempt.set_count(
                    "painter_band_changed_count", int(painter_changed)
                )
                attempt.set_count(
                    "painter_band_modified_item_count", len(painter_roots)
                )
                attempt.set_count(
                    "mutation_target_root_count",
                    len({id(root) for root in mutation_roots}),
                )
            with _rollback_display_transaction(
                self.root,
                self._band,
                capture_controller_state=capture_controller_state,
                restore_controller_state=restore_controller_state,
                mutation_roots=mutation_roots,
                performance_attempt=attempt,
            ):
                with _performance_stage(attempt, "manim_apply"):
                    _apply_display_delta(delta)
                    if painter_changed:
                        self._band.apply(prepared.painter_band)
                    self._fragment_slot_maps = {
                        source_id: dict(values)
                        for source_id, values in (
                            prepared.numeric.fragment_slot_maps.items()
                        )
                    }
                    self._last_frame = prepared.frame
                    self._last_boundary_frame = prepared.boundary_frame
                    self._last_lineage = prepared.frame.branch_lineage
                    self._display_slot_state = dict(delta.next_state)
                    self._last_painter_band_signature = painter_signature
        except Exception as exc:
            self._finish_performance_attempt(
                attempt,
                status="failed",
                rollback_performed=True,
                error=exc,
            )
            raise
        self._finish_performance_attempt(attempt, status="committed")

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def display_mobject(self) -> Mobject:
        return self.root

    def performance_snapshot(self) -> QuadricPerformanceSnapshot | None:
        """Return the latest opt-in frame measurement, if tracing is enabled."""

        return self._last_performance_snapshot

    @property
    def last_frame(self) -> CompositeQuadricSectionCompositingFrame | None:
        return self._last_frame

    @property
    def last_composite_frame(
        self,
    ) -> CompositeQuadricSectionCompositingFrame | None:
        return self._last_frame

    @property
    def last_child_frames(self) -> tuple[QuadricSectionCompositingFrame, ...]:
        return () if self._last_frame is None else self._last_frame.child_frames

    @property
    def last_boundary_frame(self) -> QuadricBoundaryCompositingFrame | None:
        return self._last_boundary_frame

    @property
    def branch_lineage(self) -> tuple[CompositeSectionBranchLineage, ...]:
        return self._last_lineage

    @property
    def active_painter_z_indices(self) -> dict[str, float]:
        return self._band.active_z_indices

    @property
    def allocated_curve_ids(self) -> tuple[str, ...]:
        return self._curve_ids

    @property
    def allocated_boundary_ids(self) -> tuple[str, ...]:
        return self._boundary_source_ids

    def attach(self) -> "CompositeQuadricSection3D":
        if self._attached:
            return self
        if config.renderer != RendererType.CAIRO:
            raise CompositeQuadricSectionAuthoringError(
                "composite quadric section binding supports Cairo only"
            )
        family_ids = {id(item) for item in self.root.get_family()}
        family_ids.update(id(item) for item in self._update_driver.get_family())
        if any(
            id(item) in family_ids
            for container in self._scene_containers()
            for item in container
        ):
            raise CompositeQuadricSectionAuthoringError(
                "composite display slots are already Scene-owned"
            )
        attempt = self._new_performance_attempt()
        try:
            with _performance_stage(attempt, "resolve_inputs"):
                resolved = self._resolve_frame_inputs()
            numeric = self._prepare_numeric(attempt, resolved)
        except Exception as exc:
            self._finish_performance_attempt(
                attempt,
                status="failed",
                error=exc,
            )
            raise
        root_state = _capture_root(self.root)
        previous_band = self._band.capture_active_state()
        self.root.reset_opacity()
        try:
            self.scene.mobjects.append(self._update_driver)
            self.scene.mobjects.append(self.root)
            self._register_fixed_frame()
            prepared = self._prepare_painter(numeric, attempt)
            self._attached = True
            self.apply(prepared)
            self._commit_input_cache(
                resolved,
                self.root.opacity_multiplier,
                prepared,
            )
        except Exception as exc:
            self._attached = False
            _restore_root(root_state)
            self._band.restore_active_state(previous_band)
            self._remove_fixed_frame()
            self._remove_owned_identities()
            self._band.restore()
            self._invalidate_cairo_static_image()
            self._finish_performance_attempt(
                attempt,
                status="failed",
                rollback_performed=True,
                error=exc,
            )
            raise
        return self

    def update(self, dt: float = 0.0) -> "CompositeQuadricSection3D":
        del dt
        if not self._attached:
            raise CompositeQuadricSectionAuthoringError(
                "composite section controller is not attached"
            )
        attempt = self._new_performance_attempt()
        try:
            with _performance_stage(attempt, "resolve_inputs"):
                resolved = self._resolve_frame_inputs()
            opacity = self.root.opacity_multiplier
            dirty_kind = _classify_dirty_frame(
                self._last_input_geometry_signature,
                self._last_input_draw_signature,
                self._last_input_opacity,
                geometry=resolved.geometry_signature,
                draw=resolved.draw_signature,
                opacity=opacity,
            )
            if self._last_prepared_frame is None:
                dirty_kind = _DirtyFrameKind.FULL
            if dirty_kind is _DirtyFrameKind.FULL:
                if attempt is not None:
                    attempt.cache_miss("dirty_frame")
                    attempt.cache_miss("prepared_numeric")
                numeric = self._prepare_numeric(attempt, resolved)
                prepared = self._prepare_painter(numeric, attempt)
            else:
                self._seed_cached_performance_counts(attempt)
                self._validate_cached_painter_band(attempt)
                if attempt is not None:
                    attempt.cache_hit("dirty_frame")
                    attempt.cache_hit("prepared_numeric")
                cached = self._last_prepared_frame
                assert cached is not None
                prepared = replace(
                    cached,
                    _performance_attempt=attempt,
                )
                if dirty_kind is _DirtyFrameKind.CLEAN:
                    with _performance_stage(attempt, "dirty_frame_shortcut"):
                        if attempt is not None:
                            active = len(self._display_slot_state)
                            attempt.set_count("display_active_slot_count", active)
                            attempt.set_count("display_changed_slot_count", 0)
                            attempt.set_count(
                                "display_unchanged_slot_count", active
                            )
                            attempt.set_count("display_hidden_slot_count", 0)
                            attempt.set_count("painter_band_changed_count", 0)
                            attempt.set_count(
                                "painter_band_modified_item_count", 0
                            )
                            attempt.set_count("mutation_target_root_count", 0)
                            attempt.set_count(
                                "transaction_snapshot_mobject_count", 0
                            )
                            attempt.set_count("modified_mobject_count", 0)
                    self._finish_performance_attempt(
                        attempt,
                        status="committed",
                    )
                    return self
            self.apply(prepared)
            self._commit_input_cache(resolved, opacity, prepared)
        except Exception as exc:
            self._finish_performance_attempt(
                attempt,
                status="failed",
                error=exc,
            )
            raise
        return self

    def _register_fixed_frame(self) -> None:
        self._fixed_frame_camera = _register_fixed_frame(self.scene, self.root)

    def _remove_fixed_frame(self) -> None:
        _remove_fixed_frame(self._fixed_frame_camera, self.root)
        self._fixed_frame_camera = None

    def _remove_owned_identities(self) -> None:
        _remove_owned_identities(self.scene, self.root, self._update_driver)

    def _invalidate_cairo_static_image(self) -> None:
        _invalidate_cairo_static_image(self.scene)

    def restore(self) -> "CompositeQuadricSection3D":
        self._attached = False
        self._remove_fixed_frame()
        self._remove_owned_identities()
        for slots in self._surface_sheet_slots.values():
            for slot in slots:
                slot.hide()
        for slot in self._plane_slots.values():
            _hide_vmobject(slot)
        for slots in self._curve_slots.values():
            for slot in slots.fragments:
                slot.hide()
        self._fragment_slot_maps = {
            source_id: {} for source_id in self._boundary_source_ids
        }
        self._last_frame = None
        self._last_boundary_frame = None
        self._last_lineage = ()
        self._display_slot_state = {}
        self._last_painter_band_signature = ()
        self._last_input_geometry_signature = None
        self._last_input_draw_signature = None
        self._last_input_opacity = None
        self._last_prepared_frame = None
        self._last_prepared_performance_counts = {}
        self._surface_view_cache.clear()
        self._band.restore()
        self.root.reset_opacity()
        self._invalidate_cairo_static_image()
        return self

    def detach(self) -> "CompositeQuadricSection3D":
        return self.restore()

    @contextmanager
    def session(self) -> Iterator["CompositeQuadricSection3D"]:
        self.attach()
        try:
            yield self
        finally:
            self.restore()

    def slot_identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())

    def child_slot_identities(self) -> dict[str, tuple[int, ...]]:
        return {
            child.surface_id: tuple(
                (
                    *(
                        id(item)
                        for slot in self._surface_sheet_slots[child.surface_id]
                        for item in slot.root.get_family()
                    ),
                    *(
                        id(item)
                        for source_id in self._child_boundary_source_ids[
                            child.surface_id
                        ]
                        for item in self._curve_slots[source_id].root.get_family()
                    ),
                )
            )
            for child in self.children
        }

    def slot_snapshot(self) -> tuple[object, ...]:
        values: list[object] = []
        for member in self.root.get_family():
            points = np.asarray(
                getattr(member, "points", np.empty((0, 3))), dtype=float
            )
            values.append(tuple(np.round(points.reshape(-1), 12)))
            for name in ("fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"):
                rgba = np.asarray(getattr(member, name, np.empty((0, 4))), dtype=float)
                values.append(tuple(np.round(rgba.reshape(-1), 12)))
            values.append(float(getattr(member, "z_index", 0.0)))
        return tuple(values)


__all__ = [
    "CompositeQuadricSection3D",
    "CompositeQuadricSectionAuthoringError",
    "PlaneInput",
    "PreparedCompositeQuadricSectionFrame",
]
