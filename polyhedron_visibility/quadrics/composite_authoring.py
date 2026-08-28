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
from dataclasses import dataclass, replace
from itertools import combinations
from math import sqrt
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
    BoundaryRenderIntent,
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
    _CurveSlots,
    _ManagedQuadricDisplayGroup,
    _PreparedBoundaryFragment,
    _PreparedConeFill,
    _PreparedDash,
    _SurfacePaintSlot,
    _adaptive_project_curve,
    _adaptive_project_curve_samples,
    _boundary_style_registry,
    _capture_root,
    _coerce_view,
    _dash_polyline_anchored,
    _hide_vmobject,
    _polyline_lengths,
    _prepared_cone_fill,
    _restore_root,
    _set_closed_subpaths,
    _source_distance_at_parameter,
    _style_component_fill,
)
from .plane_patch import PlanePatchFitError, fit_plane_display_patch
from .projection import (
    ProjectionProxyError,
    ProjectionSubdivisionError,
    build_cone_projection_layers,
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
class PreparedCompositeQuadricSectionFrame:
    numeric: _PreparedCompositeNumeric
    painter_band: PreparedPainterBand

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
            * (
                1
                + limits.max_fragments_per_curve
                * (limits.max_dashes_per_fragment + 3)
            )
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

    def _resolve_view(self) -> ParallelView:
        value = (
            self._projection_input(self.scene)
            if callable(self._projection_input)
            else self._projection_input
        )
        return _coerce_view(value)

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
    ) -> tuple[
        SectionPlane,
        tuple[AnalyticCurve3D, ...],
        tuple[CompositeSectionBranchLineage, ...],
        dict[str, ConeSpec],
    ]:
        if self._pending_plane is not None:
            plane = self._pending_plane
            curves = self._pending_curves or ()
            lineage = self._pending_lineage or ()
            owner = self._pending_owner or {}
            self._pending_plane = None
            self._pending_curves = None
            self._pending_lineage = None
            self._pending_owner = None
            return plane, curves, lineage, owner
        plane = self._resolve_plane(expected_id=self._plane_id)
        curves, lineage, owner = self._section_curves(plane)
        return plane, curves, lineage, owner

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
    ) -> tuple[QuadricSectionCompositingFrame, ...]:
        result = []
        for child in self.children:
            try:
                base = compute_global_quadric_frame(
                    (),
                    (child,),
                    view,
                    context=self.context,
                    paint_policy=QuadricPaintPolicy.PHYSICAL,
                    max_chord_error=self.max_chord_error,
                    max_segments=self.limits.max_surface_segments,
                )
                result.append(
                    compute_quadric_section_compositing(
                        base.frame,
                        child,
                        plane,
                        patch,
                        view,
                        context=self.context,
                        max_screen_error=self.section_max_screen_error,
                        limits=self.section_compositing_limits,
                    )
                )
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
        if self.include_surface_boundaries:
            result.extend(
                build_surface_boundary_sources(
                    self.children,
                    view,
                    self._generator_boundaries,
                    include_cap_rims=True,
                    include_silhouettes=True,
                )
            )
        elif self._generator_boundaries:
            result.extend(
                build_surface_boundary_sources(
                    self.children,
                    view,
                    self._generator_boundaries,
                    include_cap_rims=False,
                    include_silhouettes=False,
                )
            )
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

    def _boundary_crossings(
        self,
        sources: Sequence[QuadricBoundarySource],
        spans: Mapping[str, Sequence[QuadricBoundaryVisibilitySpan]],
        view: ParallelView,
    ) -> tuple[object, ...]:
        result = []
        child_ids = {child.surface_id for child in self.children}
        for first, second in combinations(sources, 2):
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

    def _assign_fragment_slots(
        self,
        source_id: str,
        active_ids: Sequence[str],
    ) -> dict[str, int]:
        active = tuple(active_ids)
        if len(active) > self.limits.max_fragments_per_curve:
            raise QuadricManimCapacityError(
                f"boundary {source_id!r} has {len(active)} painted fragments; "
                f"capacity is {self.limits.max_fragments_per_curve}"
            )
        previous = self._fragment_slot_maps[source_id]
        result = {
            item_id: previous[item_id] for item_id in active if item_id in previous
        }
        used = set(result.values())
        free = iter(
            index
            for index in range(self.limits.max_fragments_per_curve)
            if index not in used
        )
        for item_id in active:
            if item_id not in result:
                result[item_id] = next(free)
        return result

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

    def _prepare_numeric(self) -> _PreparedCompositeNumeric:
        plane, curves, lineage, owners = self._resolve_frame_inputs()
        self._validate_curve_topology(curves)
        view = self._resolve_view()
        patch = self._fit_patch(plane)
        child_frames = self._local_frames(plane, patch, view)
        try:
            frame = compute_composite_quadric_section_compositing(
                self.surface,
                self.section_id,
                child_frames,
                lineage,
                max_plane_fragments=(
                    self.section_compositing_limits.max_plane_fragments
                ),
            )
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

        prepared_surfaces = tuple(
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
                self._prepare_cone_fill(child, view),
            )
            for child in self.children
        )
        plane_polygons = {
            role: tuple(
                np.asarray([(x, y, 0.0) for x, y in contour], dtype=float)
                for contour in contours[role]
            )
            for role in PlaneDepthRole
        }

        sources = self._boundary_sources(curves, owners, plane, patch, view)
        non_plane = tuple(
            item
            for item in sources
            if item.source_kind is not BoundarySourceKind.PLANE_PATCH_EDGE
        )
        try:
            spans = compute_boundary_visibility(
                non_plane,
                self.children,
                view,
                context=self.context,
            )
        except Exception as exc:
            raise CompositeQuadricSectionAuthoringError(
                f"semantic boundary visibility failed: {exc}"
            ) from exc
        spans.update(self._plane_outline_visibility(frame))
        crossings = self._boundary_crossings(sources, spans, view)
        section_spans: dict[str, tuple[object, ...]] = {}
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
        source_map = {item.source_id: item for item in sources}
        by_source: dict[str, list[QuadricBoundaryPaintFragment]] = {
            item.source_id: [] for item in sources
        }
        for fragment in boundary_frame.fragments:
            if fragment.painted:
                by_source[fragment.source_id].append(fragment)
        prepared_by_source: dict[
            str, tuple[_PreparedBoundaryFragment, ...]
        ] = {}
        next_maps: dict[str, Mapping[str, int]] = {
            source_id: {} for source_id in self._boundary_source_ids
        }
        for source_id in sorted(by_source):
            source = source_map[source_id]
            style = self._boundary_style_for_source(source)
            fragments = tuple(
                sorted(by_source[source_id], key=lambda item: item.item_id)
            )
            assignment = self._assign_fragment_slots(
                source_id,
                tuple(item.item_id for item in fragments),
            )
            next_maps[source_id] = assignment
            parameters, source_points = _adaptive_project_curve_samples(
                source.curve,
                view,
                max_chord_error=self.max_chord_error,
                max_segments=self.limits.max_segments_per_fragment,
            )
            values: list[_PreparedBoundaryFragment] = []
            for fragment in fragments:
                points = _adaptive_project_curve(
                    source.curve,
                    view,
                    fragment.interval.start,
                    fragment.interval.end,
                    max_chord_error=self.max_chord_error,
                    max_segments=self.limits.max_segments_per_fragment,
                )
                _cumulative, length = _polyline_lengths(points)
                allowance = max(
                    1.0e-12,
                    self.limits.max_projected_length * 1.0e-9,
                )
                if length > self.limits.max_projected_length + allowance:
                    raise QuadricManimCapacityError(
                        f"boundary {source_id!r} fragment length {length:.9g} "
                        "exceeds max_projected_length"
                    )
                dashes = (
                    _dash_polyline_anchored(
                        points,
                        source_distance_start=_source_distance_at_parameter(
                            parameters,
                            source_points,
                            fragment.interval.start,
                        ),
                        dash_length=style.dash_length,
                        dash_gap=style.dash_gap,
                        capacity=self.limits.max_dashes_per_fragment,
                    )
                    if fragment.render_intent is BoundaryRenderIntent.DASHED
                    else ()
                )
                slot_index = assignment[fragment.item_id]
                values.append(
                    _PreparedBoundaryFragment(
                        fragment,
                        source,
                        style,
                        slot_index,
                        points,
                        dashes,
                    )
                )
                item_mobjects[fragment.item_id] = self._curve_slots[
                    source_id
                ].fragments[slot_index].root
            prepared_by_source[source_id] = tuple(values)
        if set(item_mobjects) != set(boundary_frame.draw_order):
            raise CompositeQuadricSectionAuthoringError(
                "prepared Mobjects do not cover the composite boundary draw order"
            )
        return _PreparedCompositeNumeric(
            frame,
            prepared_surfaces,
            plane_polygons,
            boundary_frame,
            prepared_by_source,
            next_maps,
            item_mobjects,
            boundary_frame.draw_order,
        )

    def _scene_containers(self) -> tuple[list[object], ...]:
        result: list[list[object]] = []
        for name in (
            "mobjects",
            "foreground_mobjects",
            "moving_mobjects",
            "static_mobjects",
        ):
            value = getattr(self.scene, name, None)
            if isinstance(value, list) and all(value is not item for item in result):
                result.append(value)
        return tuple(result)

    def _prepare_painter(
        self,
        numeric: _PreparedCompositeNumeric,
    ) -> PreparedCompositeQuadricSectionFrame:
        try:
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
        return PreparedCompositeQuadricSectionFrame(numeric, band)

    def prepare(self) -> PreparedCompositeQuadricSectionFrame:
        if not self._attached:
            raise CompositeQuadricSectionAuthoringError(
                "composite section controller is not attached"
            )
        return self._prepare_painter(self._prepare_numeric())

    def _apply_surface_pair(
        self,
        prepared: _PreparedCompositeSurface,
        opacity: float,
    ) -> None:
        back, front = self._surface_sheet_slots[prepared.child_surface_id]
        for slot in (back, front):
            slot.base.set_points_as_corners(prepared.surface_points)
        combined = min(1.0, self.style.surface_fill_opacity * opacity)
        sheet_opacity = 1.0 - sqrt(max(0.0, 1.0 - combined))
        back.root.set_fill(
            color=self.style.surface_fill_color,
            opacity=sheet_opacity,
            family=False,
        )
        front.root.set_fill(
            color=self.style.surface_fill_color,
            opacity=sheet_opacity,
            family=False,
        )
        back.base.set_stroke(opacity=0.0)
        front.base.set_stroke(opacity=0.0)
        if prepared.cone_fill is None:
            for slot in (back, front):
                slot.base.set_fill(
                    color=self.style.surface_fill_color,
                    opacity=sheet_opacity,
                )
                for component in slot.components:
                    _hide_vmobject(component)
            return
        back.base.set_fill(opacity=0.0)
        front.base.set_fill(opacity=0.0)
        lateral_colors = self.style.cone_lateral_fill_colors or (
            self.style.surface_fill_color,
        )
        cap_colors = self.style.cone_cap_fill_colors or lateral_colors
        for slot, lateral, cap, lateral_paths, cap_paths in (
            (
                back,
                back.back_lateral,
                back.back_cap,
                prepared.cone_fill.back_lateral_paths,
                prepared.cone_fill.back_cap_paths,
            ),
            (
                front,
                front.front_lateral,
                front.front_cap,
                prepared.cone_fill.front_lateral_paths,
                prepared.cone_fill.front_cap_paths,
            ),
        ):
            for component in slot.components:
                _hide_vmobject(component)
            _style_component_fill(
                lateral,
                lateral_paths,
                colors=lateral_colors,
                sheen_direction=self.style.cone_lateral_sheen_direction,
                opacity=sheet_opacity,
            )
            _style_component_fill(
                cap,
                cap_paths,
                colors=cap_colors,
                sheen_direction=self.style.cone_cap_sheen_direction,
                opacity=sheet_opacity,
            )

    def _apply_boundary_fragment(
        self,
        source_id: str,
        prepared: _PreparedBoundaryFragment,
        opacity: float,
    ) -> None:
        slot = self._curve_slots[source_id].fragments[prepared.slot_index]
        hidden = prepared.fragment.render_intent is BoundaryRenderIntent.DASHED
        style = prepared.style
        color = style.hidden_color if hidden else style.visible_color
        width = style.hidden_width if hidden else style.visible_width
        stroke_opacity = (
            style.hidden_opacity if hidden else style.visible_opacity
        ) * opacity
        if prepared.fragment.render_intent is BoundaryRenderIntent.SOLID:
            slot.solid.set_points_as_corners(prepared.points)
            slot.solid.set_fill(opacity=0.0)
            slot.solid.set_stroke(
                color=color,
                width=width,
                opacity=stroke_opacity,
            )
            slot.solid.set_stroke(
                color=style.background_color,
                width=style.background_width,
                opacity=style.background_opacity * opacity,
                background=True,
            )
            if style.cap_style is not None:
                slot.solid.set_cap_style(style.cap_style)
            if style.joint_type is not None:
                slot.solid.joint_type = style.joint_type
            for dash in slot.dashes:
                _hide_vmobject(dash)
            return
        _hide_vmobject(slot.solid)
        for index, dash in enumerate(slot.dashes):
            if index >= len(prepared.dashes):
                _hide_vmobject(dash)
                continue
            dash.set_points_as_corners(prepared.dashes[index].points)
            dash.set_fill(opacity=0.0)
            dash.set_stroke(
                color=color,
                width=width,
                opacity=stroke_opacity,
            )
            dash.set_stroke(
                color=style.background_color,
                width=style.background_width,
                opacity=style.background_opacity * opacity,
                background=True,
            )
            cap = (
                style.cap_style
                if style.hidden_cap_style is None
                else style.hidden_cap_style
            )
            joint = (
                style.joint_type
                if style.hidden_joint_type is None
                else style.hidden_joint_type
            )
            if cap is not None:
                dash.set_cap_style(cap)
            if joint is not None:
                dash.joint_type = joint

    def apply(self, prepared: PreparedCompositeQuadricSectionFrame) -> None:
        if not self._attached:
            raise CompositeQuadricSectionAuthoringError(
                "composite section controller is not attached"
            )
        if not isinstance(prepared, PreparedCompositeQuadricSectionFrame):
            raise TypeError(
                "prepared must be a PreparedCompositeQuadricSectionFrame"
            )
        root_state = _capture_root(self.root)
        band_state = self._band.capture_active_state()
        previous_maps = {
            source_id: dict(values)
            for source_id, values in self._fragment_slot_maps.items()
        }
        previous_frame = self._last_frame
        previous_boundary = self._last_boundary_frame
        previous_lineage = self._last_lineage
        opacity = self.root.opacity_multiplier
        try:
            for slots in self._surface_sheet_slots.values():
                for slot in slots:
                    slot.hide()
            for slot in self._plane_slots.values():
                _hide_vmobject(slot)
            for slots in self._curve_slots.values():
                for slot in slots.fragments:
                    slot.hide()
            for surface in prepared.numeric.surfaces:
                self._apply_surface_pair(surface, opacity)
            for role, item_id in self._plane_item_ids.items():
                slot = self._plane_slots[item_id]
                _set_closed_subpaths(slot, prepared.numeric.plane_polygons[role])
                slot.set_fill(
                    color=self.style.section_plane_fill_color,
                    opacity=self.style.section_plane_fill_opacity * opacity,
                )
                slot.set_stroke(opacity=0.0)
            for source_id, fragments in prepared.numeric.boundary_fragments.items():
                for fragment in fragments:
                    self._apply_boundary_fragment(source_id, fragment, opacity)
            self._band.apply(prepared.painter_band)
            self._fragment_slot_maps = {
                source_id: dict(values)
                for source_id, values in prepared.numeric.fragment_slot_maps.items()
            }
            self._last_frame = prepared.frame
            self._last_boundary_frame = prepared.boundary_frame
            self._last_lineage = prepared.frame.branch_lineage
        except Exception:
            _restore_root(root_state)
            self._band.restore_active_state(band_state)
            self._fragment_slot_maps = previous_maps
            self._last_frame = previous_frame
            self._last_boundary_frame = previous_boundary
            self._last_lineage = previous_lineage
            raise

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def display_mobject(self) -> Mobject:
        return self.root

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
        numeric = self._prepare_numeric()
        root_state = _capture_root(self.root)
        previous_band = self._band.capture_active_state()
        self.root.reset_opacity()
        try:
            self.scene.mobjects.append(self._update_driver)
            self.scene.mobjects.append(self.root)
            self._register_fixed_frame()
            prepared = self._prepare_painter(numeric)
            self._attached = True
            self.apply(prepared)
        except Exception:
            self._attached = False
            _restore_root(root_state)
            self._band.restore_active_state(previous_band)
            self._remove_fixed_frame()
            self._remove_owned_identities()
            self._band.restore()
            self._invalidate_cairo_static_image()
            raise
        return self

    def update(self, dt: float = 0.0) -> "CompositeQuadricSection3D":
        del dt
        if not self._attached:
            raise CompositeQuadricSectionAuthoringError(
                "composite section controller is not attached"
            )
        self.apply(self.prepare())
        return self

    def _register_fixed_frame(self) -> None:
        camera = getattr(self.scene, "camera", None)
        if isinstance(camera, ThreeDCamera):
            self._fixed_frame_camera = camera
            camera.add_fixed_in_frame_mobjects(self.root)

    def _remove_fixed_frame(self) -> None:
        if self._fixed_frame_camera is not None:
            self._fixed_frame_camera.remove_fixed_in_frame_mobjects(self.root)
            self._fixed_frame_camera = None

    def _remove_owned_identities(self) -> None:
        owned = {id(item) for item in self.root.get_family()}
        owned.update(id(item) for item in self._update_driver.get_family())
        for container in self._scene_containers():
            container[:] = [item for item in container if id(item) not in owned]

    def _invalidate_cairo_static_image(self) -> None:
        renderer = getattr(self.scene, "renderer", None)
        if renderer is not None and hasattr(renderer, "static_image"):
            renderer.static_image = None

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
