"""Fixed-capacity Cairo binding for analytic quadric occlusion.

The renderer-neutral quadric stack has two deliberately separate products:

* exact analytic curve visibility, which is the geometric truth; and
* closed two-dimensional projection proxies, which are opaque paint only.

This module joins those products at the Manim boundary.  Every Mobject is
allocated in ``__init__``.  Frame updates only prepare numeric geometry and
then mutate existing slots in one rollback-safe transaction.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from math import sqrt
from typing import Callable, Iterator, Mapping, Protocol, Sequence

import numpy as np
from manim import (
    BLUE_D,
    Dot,
    WHITE,
    Line,
    Mobject,
    RendererType,
    ThreeDCamera,
    VGroup,
    VMobject,
    config,
)

from ..painter_band import (
    ManagedPainterBand,
    ManagedPainterBandError,
    PreparedPainterBand,
)
from ..geometry import GeometryContext, ResolvedGeometryContext
from ..parallel_solver import ParallelView, SolverError
from ..style import OcclusionStyle
from ..visibility import VisibilityKind
from .compositing import (
    QuadricCompositingError,
    QuadricCompositingFrame,
    QuadricCurvePaintFragment,
    QuadricPaintPolicy,
    SurfaceConstraintInput,
    compute_quadric_compositing,
)
from .boundary_compositing import (
    BoundarySectionAnchors,
    BoundarySemanticKind,
    BoundarySourceKind,
    QuadricBoundaryCompositingError,
    QuadricBoundaryCompositingFrame,
    QuadricBoundaryPaintFragment,
    QuadricRankOneSectionSourceGroup,
    QuadricBoundarySource,
    QuadricBoundaryVisibilitySpan,
    compute_boundary_visibility,
    compute_quadric_boundary_compositing,
)
from .contract import (
    ConeModel,
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from .curve_intersections import (
    ProjectedCurveIntersectionError,
    compute_projected_curve_crossings,
)
from .curves import (
    EllipseArcCurve,
    ParametricConicBranch,
    PointMarker3D,
    SegmentCurve,
)
from .global_occlusion import (
    GlobalQuadricFrame,
    GlobalQuadricOcclusionError,
    compute_global_quadric_frame,
)
from .projection import (
    ConeProjectionLayers,
    OpaqueProjectionProxy,
    ProjectionProxyError,
    ProjectionSubdivisionError,
    build_cone_projection_layers,
    build_opaque_projection_proxy,
)
from .performance import (
    QuadricPerformanceSnapshot,
    _PerformanceAttempt,
    _performance_stage,
    quadric_performance_tracing_enabled,
)
from .plane_patch import PlanePatchFitError, fit_plane_display_patch
from .section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionKind,
    QUADRIC_SECTION_COMPOSITING_LIMITS,
    QuadricSectionCompositingError,
    QuadricSectionCompositingFrame,
    QuadricSectionCompositingLimits,
    compute_quadric_section_compositing,
    quadric_plane_fragment_contours,
    repaint_quadric_section_compositing,
)
from .visibility import compute_point_visibility, compute_quadric_visibility
from .boundary_section import (
    QUADRIC_BOUNDARY_SECTION_LIMITS,
    QuadricBoundarySectionLimits,
    _compute_boundary_section_spans_with_contours,
    certify_rank_one_section_boundary_sources,
)
from .surface_boundaries import (
    GeneratorBoundarySpec,
    build_surface_boundary_sources,
    curve_boundary_source,
    plane_outline_sources,
    section_curve_boundary_source,
    surface_boundary_source_ids,
)
from .sections import (
    QuadricSectionError,
    compute_quadric_section_boundary_curves,
)
from .manim_runtime import (
    QuadricBoundaryStyle,
    QuadricManimCapacityError,
    QuadricManimError,
    _CommittedDisplaySlot,
    _CurveSlots,
    _DirtyFrameKind,
    _ManagedQuadricDisplayGroup,
    _MobjectState,
    _PreparedBoundaryFragment,
    _PreparedConeFill,
    _PreparedDash,
    _PreparedDisplayAction,
    _ResolvedParallelCameraFrame,
    _SurfaceViewCache,
    _SurfacePaintSlot,
    _adaptive_project_curve,
    _adaptive_project_curve_samples,
    _apply_boundary_fragment as _apply_runtime_boundary_fragment,
    _apply_opaque_surface_slot,
    _apply_surface_sheet_pair,
    _boundary_style_registry,
    _capture_root,
    _coerce_projection_frame,
    _classify_dirty_frame,
    _curve_slots_family_capacity,
    _dash_polyline,
    _dash_polyline_anchored,
    _hide_vmobject,
    _invalidate_cairo_static_image,
    _non_negative,
    _polyline_lengths,
    _positive,
    _projection_display_offset,
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
    _set_open_subpaths,
    _slice_projected_curve_samples,
    _apply_display_delta,
    _display_digest,
    _display_offset,
    _painter_band_signature,
)

QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec
AnalyticCurve3D = SegmentCurve | EllipseArcCurve | ParametricConicBranch
SurfaceInput = Sequence[QuadricSurfaceSpec] | Callable[[], Sequence[QuadricSurfaceSpec]]
CurveInput = Sequence[AnalyticCurve3D] | Callable[[], Sequence[AnalyticCurve3D]]
CurveOpacityInput = (
    Mapping[str, float] | Callable[[], Mapping[str, float]] | None
)
PointInput = Sequence[PointMarker3D] | Callable[[], Sequence[PointMarker3D]]
PointOpacityInput = (
    Mapping[str, float] | Callable[[], Mapping[str, float]] | None
)
BoundaryOpacityInput = (
    Mapping[str, float] | Callable[[], Mapping[str, float]] | None
)
SurfaceOpacityInput = (
    Mapping[str, float] | Callable[[], Mapping[str, float]] | None
)
SurfaceStrokeOpacityInput = SurfaceOpacityInput
ScalarOpacityInput = float | Callable[[], float] | None
OccludingSurfaceInput = Sequence[str] | Callable[[], Sequence[str]] | None
PaintPolicyInput = (
    QuadricPaintPolicy
    | str
    | Callable[[], QuadricPaintPolicy | str]
)
SectionPlaneInput = SectionPlane | Callable[[], SectionPlane] | None
PlanePatchInput = (
    PlaneDisplayPatchSpec | Callable[[], PlaneDisplayPatchSpec] | None
)

class _SemanticParallelProjection(Protocol):
    matrix: object
    target: object
    screen_anchor: object
    zoom: object


class _ParallelCameraSnapshotSource(Protocol):
    def snapshot_parallel_state(self) -> _SemanticParallelProjection: ...


class _SemanticParallelViewport(Protocol):
    camera: _SemanticParallelProjection
    screen_transform: object


ProjectionValue = (
    ParallelView
    | Sequence[Sequence[float]]
    | _SemanticParallelProjection
    | _SemanticParallelViewport
    | _ParallelCameraSnapshotSource
)
ProjectionInput = ProjectionValue | Callable[[object], ProjectionValue]
BoundaryGeneratorInput = Sequence[GeneratorBoundarySpec]


DEFAULT_QUADRIC_VIEW = ParallelView.from_matrix(
    (
        (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
        (-1.0 / sqrt(6.0), -1.0 / sqrt(6.0), 2.0 / sqrt(6.0)),
        (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    )
)


@dataclass(frozen=True, slots=True)
class QuadricManimStyle:
    """Static display style for one managed quadric painter graph."""

    surface_fill_color: object = BLUE_D
    surface_fill_opacity: float = 1.0
    surface_stroke_color: object = BLUE_D
    surface_stroke_width: float = 1.5
    surface_stroke_opacity: float = 1.0
    visible_curve_color: object = WHITE
    visible_curve_width: float = 3.0
    visible_curve_opacity: float = 1.0
    hidden_curve_color: object = WHITE
    hidden_curve_width: float = 2.4
    hidden_curve_opacity: float = 0.78
    point_color: object = WHITE
    point_radius: float = 0.055
    point_opacity: float = 1.0
    hidden_point_color: object = WHITE
    hidden_point_opacity: float = 0.45
    dash_length: float = 0.08
    dash_gap: float = 0.06
    background_color: object = WHITE
    background_width: float = 0.0
    background_opacity: float = 0.0
    cap_style: object | None = None
    joint_type: object | None = None
    hidden_cap_style: object | None = None
    hidden_joint_type: object | None = None
    section_plane_fill_color: object = "#63C7B2"
    section_plane_fill_opacity: float = 0.15
    section_plane_stroke_color: object = "#2C8C7A"
    section_plane_stroke_width: float = 1.4
    section_plane_stroke_opacity: float = 0.65
    cone_lateral_fill_colors: tuple[object, ...] | None = None
    cone_cap_fill_colors: tuple[object, ...] | None = None
    cone_lateral_sheen_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    cone_cap_sheen_direction: tuple[float, float, float] = (-1.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        for name in (
            "surface_fill_opacity",
            "surface_stroke_width",
            "surface_stroke_opacity",
            "visible_curve_width",
            "visible_curve_opacity",
            "hidden_curve_width",
            "hidden_curve_opacity",
            "point_radius",
            "point_opacity",
            "hidden_point_opacity",
            "background_width",
            "background_opacity",
            "section_plane_fill_opacity",
            "section_plane_stroke_width",
            "section_plane_stroke_opacity",
        ):
            object.__setattr__(self, name, _non_negative(getattr(self, name), name))
        object.__setattr__(
            self, "dash_length", _positive(self.dash_length, "dash_length")
        )
        object.__setattr__(
            self, "point_radius", _positive(self.point_radius, "point_radius")
        )
        object.__setattr__(self, "dash_gap", _non_negative(self.dash_gap, "dash_gap"))
        for name in (
            "surface_fill_opacity",
            "surface_stroke_opacity",
            "visible_curve_opacity",
            "hidden_curve_opacity",
            "point_opacity",
            "hidden_point_opacity",
            "background_opacity",
            "section_plane_fill_opacity",
            "section_plane_stroke_opacity",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed 1")
        for name in ("cone_lateral_fill_colors", "cone_cap_fill_colors"):
            raw = getattr(self, name)
            if raw is None:
                continue
            if isinstance(raw, (str, bytes)):
                colors = (raw,)
            else:
                try:
                    colors = tuple(raw)
                except TypeError as exc:
                    raise ValueError(
                        f"{name} must be a non-empty color sequence"
                    ) from exc
            if not colors:
                raise ValueError(f"{name} must be a non-empty color sequence")
            object.__setattr__(self, name, colors)
        for name in (
            "cone_lateral_sheen_direction",
            "cone_cap_sheen_direction",
        ):
            try:
                direction = np.asarray(getattr(self, name), dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must contain three finite values") from exc
            if (
                direction.shape != (3,)
                or not np.all(np.isfinite(direction))
                or float(np.linalg.norm(direction)) <= 0.0
            ):
                raise ValueError(f"{name} must contain three finite non-zero values")
            object.__setattr__(
                self,
                name,
                tuple(float(item) for item in direction / np.linalg.norm(direction)),
            )

    @property
    def dash_period(self) -> float:
        return self.dash_length + self.dash_gap

    @property
    def cone_component_shading(self) -> bool:
        return (
            self.cone_lateral_fill_colors is not None
            or self.cone_cap_fill_colors is not None
        )

    def compositor_style(self, *, max_projected_length: float) -> OcclusionStyle:
        """Return the renderer-neutral style descriptor used by the graph."""

        return OcclusionStyle(
            max_projected_length=max_projected_length,
            dash_length=self.dash_length,
            dash_gap=self.dash_gap,
            visible_color=self.visible_curve_color,
            hidden_color=self.hidden_curve_color,
            visible_width_scale=1.0,
            hidden_width_scale=1.0,
            visible_opacity_scale=self.visible_curve_opacity,
            hidden_opacity_scale=self.hidden_curve_opacity,
            hidden_cap_style=self.hidden_cap_style,
            hidden_joint_type=self.hidden_joint_type,
        )


@dataclass(frozen=True, slots=True)
class QuadricManimLimits:
    """Hard bounds checked before any Manim state is changed."""

    max_surfaces: int = 16
    max_curves: int = 32
    max_fragments_per_curve: int = 32
    max_segments_per_fragment: int = 1024
    max_surface_segments: int = 2048
    max_dashes_per_fragment: int = 128
    max_projected_length: float = 16.0
    max_total_mobjects: int = 100000
    max_boundary_sources: int = 64
    max_boundary_styles: int = 64
    max_points: int = 32

    def __post_init__(self) -> None:
        for name in (
            "max_surfaces",
            "max_curves",
            "max_fragments_per_curve",
            "max_segments_per_fragment",
            "max_surface_segments",
            "max_dashes_per_fragment",
            "max_total_mobjects",
            "max_boundary_sources",
            "max_boundary_styles",
            "max_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self,
            "max_projected_length",
            _positive(self.max_projected_length, "max_projected_length"),
        )


def estimate_quadric_mobject_count(
    *,
    surface_count: int,
    source_count: int,
    max_fragments_per_curve: int,
    section_enabled: bool,
    point_count: int = 0,
) -> int:
    """Return the exact fixed-family estimate checked by this binding.

    ``source_count`` is the size of the immutable slot-source union, not only
    the number of sources painted in one frame.  Dash capacity is deliberately
    absent because every fragment stores all dashed subpaths in one VMobject.
    """

    for name, value in (
        ("surface_count", surface_count),
        ("source_count", source_count),
        ("max_fragments_per_curve", max_fragments_per_curve),
        ("point_count", point_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(section_enabled, bool):
        raise TypeError("section_enabled must be a bool")
    return (
        6 * surface_count
        + (20 if section_enabled else 0)
        + 1
        + source_count
        * _curve_slots_family_capacity(max_fragments_per_curve)
        + 4
        + (1 + point_count if point_count else 0)
    )


QUADRIC_MANIM_LIMITS = QuadricManimLimits()


class QuadricGeometryPrototype:
    """Bounded renderer-neutral cache shared by coordinated paint variants."""

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache = _SurfaceViewCache()

    def clear(self) -> None:
        """Discard every cached exact-signature geometry product."""

        self._cache.clear()


@dataclass(frozen=True, slots=True)
class _PreparedCurveFragment:
    fragment: QuadricCurvePaintFragment
    slot_index: int
    points: np.ndarray
    dashes: tuple[_PreparedDash, ...]


@dataclass(frozen=True, slots=True)
class _PreparedSurface:
    item_id: str
    surface_id: str
    slot_index: int
    points: np.ndarray
    cone_fill: _PreparedConeFill | None = None


@dataclass(frozen=True, slots=True)
class _PreparedPoint:
    item_id: str
    point_id: str
    screen_point: np.ndarray
    visible: bool
    occluders: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedSectionLayers:
    frame: QuadricSectionCompositingFrame
    surface_points: np.ndarray
    plane_polygons: Mapping[PlaneDepthRole, tuple[np.ndarray, ...]]
    plane_outline_paths: Mapping[PlaneDepthRole, tuple[np.ndarray, ...]]
    cone_fill: _PreparedConeFill | None = None


@dataclass(frozen=True, slots=True)
class _PreparedNumericFrame:
    frame: QuadricCompositingFrame
    global_frame: GlobalQuadricFrame | None
    surfaces: tuple[_PreparedSurface, ...]
    fragments: Mapping[str, tuple[_PreparedCurveFragment, ...]]
    curve_opacities: Mapping[str, float]
    points: tuple[_PreparedPoint, ...]
    point_opacities: Mapping[str, float]
    fragment_slot_maps: Mapping[str, Mapping[str, int]]
    item_mobjects: Mapping[str, Mobject]
    painter_draw_order: tuple[str, ...]
    section_layers: _PreparedSectionLayers | None = None
    boundary_frame: QuadricBoundaryCompositingFrame | None = None
    boundary_fragments: Mapping[
        str, tuple[_PreparedBoundaryFragment, ...]
    ] | None = None
    boundary_opacities: Mapping[str, float] | None = None
    surface_opacities: Mapping[str, float] = field(default_factory=dict)
    surface_stroke_opacities: Mapping[str, float] = field(default_factory=dict)
    section_plane_fill_opacity: float = 1.0
    section_plane_stroke_opacity: float = 1.0
    projected_source_segment_counts: Mapping[str, int] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class _ResolvedQuadricFrameInputs:
    surfaces: tuple[QuadricSurfaceSpec, ...]
    occluding_surfaces: tuple[QuadricSurfaceSpec, ...]
    surface_opacities: Mapping[str, float]
    surface_stroke_opacities: Mapping[str, float]
    curves: tuple[AnalyticCurve3D, ...]
    curve_opacities: Mapping[str, float]
    points: tuple[PointMarker3D, ...]
    point_opacities: Mapping[str, float]
    boundary_opacities: Mapping[str, float]
    section_plane_fill_opacity: float
    section_plane_stroke_opacity: float
    paint_policy: QuadricPaintPolicy
    view: ParallelView
    display_offset: tuple[float, float]
    section_plane: SectionPlane | None
    section_patch: PlaneDisplayPatchSpec | None
    surface_view_signature: bytes
    geometry_signature: bytes
    draw_signature: bytes


@dataclass(frozen=True, slots=True)
class PreparedQuadricManimFrame:
    numeric: _PreparedNumericFrame
    painter_band: PreparedPainterBand
    _performance_attempt: _PerformanceAttempt | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def frame(self) -> QuadricCompositingFrame:
        return self.numeric.frame

    @property
    def global_frame(self) -> GlobalQuadricFrame | None:
        """Return the automatic global evidence prepared with this frame."""

        return self.numeric.global_frame

    @property
    def section_frame(self) -> QuadricSectionCompositingFrame | None:
        """Return the prepared plane/surface split, when section mode is active."""

        layers = self.numeric.section_layers
        return None if layers is None else layers.frame

    @property
    def boundary_frame(self) -> QuadricBoundaryCompositingFrame | None:
        """Return the unified semantic-boundary painter frame, when enabled."""

        return self.numeric.boundary_frame


@dataclass(frozen=True, slots=True)
class _CommittedControllerState:
    fragment_slot_maps: dict[str, dict[str, int]]
    last_frame: QuadricCompositingFrame | None
    last_global_frame: GlobalQuadricFrame | None
    last_section_frame: QuadricSectionCompositingFrame | None
    last_boundary_frame: QuadricBoundaryCompositingFrame | None
    display_slot_state: dict[str, _CommittedDisplaySlot]
    last_painter_band_signature: tuple[tuple[str, int, float], ...]
    last_input_geometry_signature: bytes | None
    last_input_draw_signature: bytes | None
    last_input_opacity: float | None
    last_prepared_frame: PreparedQuadricManimFrame | None
    last_prepared_performance_counts: dict[str, int]


@dataclass(frozen=True, slots=True, eq=False)
class QuadricOcclusionTransactionSnapshot:
    """Opaque, controller-bound snapshot of one attached display frame.

    Instances are created by :meth:`QuadricOcclusion3D.snapshot_transaction_state`.
    They deliberately retain the fixed Mobject identities owned by that
    controller, so a snapshot cannot be applied to another controller even
    when both controllers were authored from equivalent geometry.
    """

    _owner_token: object = field(repr=False)
    _root_state: tuple[_MobjectState, ...] = field(repr=False)
    _painter_band_state: Mapping[str, float] = field(repr=False)
    _fragment_slot_maps: dict[str, dict[str, int]] = field(repr=False)
    _last_frame: QuadricCompositingFrame | None = field(repr=False)
    _last_global_frame: GlobalQuadricFrame | None = field(repr=False)
    _last_section_frame: QuadricSectionCompositingFrame | None = field(
        repr=False
    )
    _last_boundary_frame: QuadricBoundaryCompositingFrame | None = field(
        repr=False
    )
    _display_slot_state: dict[str, _CommittedDisplaySlot] = field(repr=False)
    _last_painter_band_signature: tuple[tuple[str, int, float], ...] = field(
        repr=False
    )
    _last_input_geometry_signature: bytes | None = field(repr=False)
    _last_input_draw_signature: bytes | None = field(repr=False)
    _last_input_opacity: float | None = field(repr=False)
    _last_prepared_frame: PreparedQuadricManimFrame | None = field(repr=False)
    _last_prepared_performance_counts: dict[str, int] = field(repr=False)
    _performance_frame_index: int = field(repr=False)
    _last_performance_snapshot: QuadricPerformanceSnapshot | None = field(
        repr=False
    )


_TRANSACTION_NUMERIC_STYLE_ATTRIBUTES = frozenset(
    {
        "fill_rgbas",
        "stroke_rgbas",
        "background_stroke_rgbas",
        "fill_opacity",
        "stroke_opacity",
        "background_stroke_opacity",
        "stroke_width",
        "background_stroke_width",
        "sheen_direction",
        "sheen_factor",
    }
)
_TRANSACTION_ENUM_STYLE_ATTRIBUTES = frozenset({"cap_style", "joint_type"})


def _capture_transaction_root(root: Mobject) -> tuple[_MobjectState, ...]:
    """Capture every Cairo-visible style field in addition to base state."""

    states = _capture_root(root)
    for state in states:
        for name in (
            "stroke_width",
            "background_stroke_width",
            "cap_style",
            "joint_type",
        ):
            if not hasattr(state.mobject, name):
                continue
            value = getattr(state.mobject, name)
            state.attributes[name] = (
                value.copy() if isinstance(value, np.ndarray) else value
            )
    return states


def _surface_items(
    value: Sequence[QuadricSurfaceSpec],
) -> tuple[QuadricSurfaceSpec, ...]:
    authored = tuple(value)
    if not authored or not all(
        isinstance(item, (SphereSpec, CylinderSpec, ConeSpec)) for item in authored
    ):
        raise TypeError("surfaces must contain at least one supported quadric spec")
    expanded: list[QuadricSurfaceSpec] = []
    for item in authored:
        if isinstance(item, ConeSpec) and item.model is ConeModel.OPEN_DOUBLE:
            expanded.extend(item.render_components)
        else:
            expanded.append(item)
    result = tuple(expanded)
    identities = tuple(item.surface_id for item in result)
    if len(set(identities)) != len(identities):
        raise QuadricManimError("surface identities must be unique")
    return tuple(sorted(result, key=lambda item: item.surface_id))


def _curve_items(value: Sequence[AnalyticCurve3D]) -> tuple[AnalyticCurve3D, ...]:
    result = tuple(value)
    if not all(
        isinstance(item, (SegmentCurve, EllipseArcCurve, ParametricConicBranch))
        for item in result
    ):
        raise TypeError("curves must contain supported analytic curves")
    identities = tuple(item.curve_id for item in result)
    if len(set(identities)) != len(identities):
        raise QuadricManimError("curve identities must be unique")
    return tuple(sorted(result, key=lambda item: item.curve_id))


def _point_items(value: Sequence[PointMarker3D]) -> tuple[PointMarker3D, ...]:
    result = tuple(value)
    if not all(isinstance(item, PointMarker3D) for item in result):
        raise TypeError("points must contain PointMarker3D values")
    identities = tuple(item.point_id for item in result)
    if len(set(identities)) != len(identities):
        raise QuadricManimError("point identities must be unique")
    return tuple(sorted(result, key=lambda item: item.point_id))


class QuadricOcclusion3D:
    """One fixed-topology, fixed-capacity quadric Cairo controller.

    ``surfaces`` and ``curves`` may be callables so an animation can return
    freshly constructed immutable analytic specs each frame.  By default their
    semantic identities must remain unchanged while the controller is alive.
    The advanced ``allocated_curve_ids`` mode reserves a larger immutable
    identity pool and lets each frame activate any subset of that pool;
    ``curve_opacities`` then supplies the opacity for every active identity.
    ``surface_order_mode='automatic'`` recomputes and consumes one complete
    global painter frame on every update.  ``'explicit'`` keeps the legacy
    caller-supplied surface-order path.  ``automatic_updates=False`` retains
    the time-aware Cairo driver that keeps the display out of the static
    background, but leaves every update to an external frame coordinator.
    ``legacy_surface_stroke_fallback=True`` is a narrow opt-in for one
    unoccluded teaching outline when unified intrinsic surface boundaries are
    explicitly excluded; the default remains off.
    """

    def __init__(
        self,
        scene: object,
        *,
        surfaces: SurfaceInput,
        curves: CurveInput,
        points: PointInput = (),
        projection: ProjectionInput | None = None,
        paint_policy: PaintPolicyInput = QuadricPaintPolicy.DIAGRAMMATIC,
        style: QuadricManimStyle = QuadricManimStyle(),
        surface_opacities: SurfaceOpacityInput = None,
        surface_stroke_opacities: SurfaceStrokeOpacityInput = None,
        occluding_surface_ids: OccludingSurfaceInput = None,
        boundary_styles: Mapping[str, QuadricBoundaryStyle] | None = None,
        boundary_opacities: BoundaryOpacityInput = None,
        limits: QuadricManimLimits = QUADRIC_MANIM_LIMITS,
        max_chord_error: float = 1.0e-3,
        context: GeometryContext | ResolvedGeometryContext | None = None,
        painter_z_band: tuple[float, float] = (20.0, 30.0),
        surface_constraints: Sequence[SurfaceConstraintInput] = (),
        surface_order_mode: str = "automatic",
        allocated_curve_ids: Sequence[str] | None = None,
        curve_opacities: CurveOpacityInput = None,
        allocated_point_ids: Sequence[str] | None = None,
        point_opacities: PointOpacityInput = None,
        section_id: str | None = None,
        section_coefficient_tolerance: float | None = None,
        section_plane: SectionPlaneInput = None,
        section_patch: PlanePatchInput = None,
        section_plane_fill_opacity: ScalarOpacityInput = None,
        section_plane_stroke_opacity: ScalarOpacityInput = None,
        section_patch_margin: float = 0.08,
        section_max_screen_error: float = 0.08,
        section_compositing_limits: QuadricSectionCompositingLimits = (
            QUADRIC_SECTION_COMPOSITING_LIMITS
        ),
        boundary_section_limits: QuadricBoundarySectionLimits = (
            QUADRIC_BOUNDARY_SECTION_LIMITS
        ),
        boundary_visibility_mode: str = "legacy",
        include_surface_boundaries: bool = True,
        generator_boundaries: BoundaryGeneratorInput = (),
        allocated_boundary_ids: Sequence[str] | None = None,
        geometry_prototype: QuadricGeometryPrototype | None = None,
        display_offset: Sequence[float] = (0.0, 0.0),
        automatic_updates: bool = True,
        legacy_surface_stroke_fallback: bool = False,
    ) -> None:
        if not isinstance(style, QuadricManimStyle):
            raise TypeError("style must be a QuadricManimStyle")
        if not isinstance(limits, QuadricManimLimits):
            raise TypeError("limits must be a QuadricManimLimits")
        if geometry_prototype is not None and not isinstance(
            geometry_prototype, QuadricGeometryPrototype
        ):
            raise TypeError(
                "geometry_prototype must be a QuadricGeometryPrototype"
            )
        if context is not None and not isinstance(
            context, (GeometryContext, ResolvedGeometryContext)
        ):
            raise TypeError(
                "context must be a GeometryContext or ResolvedGeometryContext"
            )
        self._paint_policy_input = paint_policy
        policy = self._resolve_paint_policy()
        if boundary_visibility_mode not in ("legacy", "unified"):
            raise QuadricManimError(
                "boundary_visibility_mode must be 'legacy' or 'unified'"
            )
        if not isinstance(include_surface_boundaries, bool):
            raise TypeError("include_surface_boundaries must be a bool")
        if not isinstance(automatic_updates, bool):
            raise TypeError("automatic_updates must be a bool")
        if not isinstance(legacy_surface_stroke_fallback, bool):
            raise TypeError("legacy_surface_stroke_fallback must be a bool")
        if legacy_surface_stroke_fallback and (
            boundary_visibility_mode != "unified" or include_surface_boundaries
        ):
            raise QuadricManimError(
                "legacy_surface_stroke_fallback requires unified boundary "
                "visibility with surface boundaries excluded"
            )
        generators = tuple(generator_boundaries)
        if not all(isinstance(item, GeneratorBoundarySpec) for item in generators):
            raise TypeError(
                "generator_boundaries must contain GeneratorBoundarySpec values"
            )
        if surface_order_mode not in ("automatic", "explicit"):
            raise QuadricManimError(
                "surface_order_mode must be 'automatic' or 'explicit'"
            )
        if section_patch is not None and section_plane is None:
            raise QuadricManimError(
                "section_patch requires section_plane"
            )
        if section_id is not None and (
            not isinstance(section_id, str) or not section_id.strip()
        ):
            raise QuadricManimError("section_id must be a non-empty string")
        if section_id is not None and section_plane is None:
            raise QuadricManimError("section_id requires section_plane")
        if section_coefficient_tolerance is not None and section_id is None:
            raise QuadricManimError(
                "section_coefficient_tolerance requires section_id"
            )
        if not isinstance(
            section_compositing_limits, QuadricSectionCompositingLimits
        ):
            raise TypeError(
                "section_compositing_limits must be a "
                "QuadricSectionCompositingLimits"
            )
        if not isinstance(boundary_section_limits, QuadricBoundarySectionLimits):
            raise TypeError(
                "boundary_section_limits must be a "
                "QuadricBoundarySectionLimits"
            )
        self.scene = scene
        self.geometry_prototype = geometry_prototype
        self.display_offset = display_offset
        self._automatic_updates = automatic_updates
        self._legacy_surface_stroke_fallback = legacy_surface_stroke_fallback
        self._surface_input = surfaces
        self._surface_opacity_input = surface_opacities
        self._surface_stroke_opacity_input = surface_stroke_opacities
        self._occluding_surface_input = occluding_surface_ids
        self._curve_input = curves
        self._point_input = points
        self._projection_input = (
            DEFAULT_QUADRIC_VIEW if projection is None else projection
        )
        self.paint_policy = policy
        self.style = style
        self.limits = limits
        self.boundary_styles = _boundary_style_registry(
            style,
            boundary_styles,
            limits,
        )
        self._boundary_opacity_input = boundary_opacities
        if boundary_visibility_mode == "unified":
            unknown_generator_styles = sorted(
                {
                    spec.style_id
                    for spec in generators
                    if spec.style_id is not None
                    and spec.style_id not in self.boundary_styles
                }
            )
            if unknown_generator_styles:
                raise QuadricManimError(
                    "generator boundaries reference unknown boundary styles: "
                    + ", ".join(unknown_generator_styles)
                )
        self.max_chord_error = _positive(max_chord_error, "max_chord_error")
        self.context = context
        self.surface_constraints = tuple(surface_constraints)
        self.surface_order_mode = surface_order_mode
        self.boundary_visibility_mode = boundary_visibility_mode
        self.include_surface_boundaries = include_surface_boundaries
        self._generator_boundaries = generators
        self._allocated_boundary_ids_input = allocated_boundary_ids
        self._curve_opacity_input = curve_opacities
        self._point_opacity_input = point_opacities
        self.section_id = None if section_id is None else section_id.strip()
        self.section_coefficient_tolerance = section_coefficient_tolerance
        self._section_plane_input = section_plane
        self._section_patch_input = section_patch
        self._section_plane_fill_opacity_input = section_plane_fill_opacity
        self._section_plane_stroke_opacity_input = section_plane_stroke_opacity
        self.section_patch_margin = _non_negative(
            section_patch_margin, "section_patch_margin"
        )
        self.section_max_screen_error = _positive(
            section_max_screen_error, "section_max_screen_error"
        )
        self.section_compositing_limits = section_compositing_limits
        self.boundary_section_limits = boundary_section_limits
        self._section_enabled = section_plane is not None
        self._attached = False
        self._fixed_frame_camera: ThreeDCamera | None = None
        self._last_frame: QuadricCompositingFrame | None = None
        self._last_global_frame: GlobalQuadricFrame | None = None
        self._last_section_frame: QuadricSectionCompositingFrame | None = None
        self._last_boundary_frame: QuadricBoundaryCompositingFrame | None = None
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
        self._last_prepared_frame: PreparedQuadricManimFrame | None = None
        self._last_prepared_performance_counts: dict[str, int] = {}
        self._transaction_snapshot_owner_token = object()
        self._owns_surface_view_cache = geometry_prototype is None
        self._surface_view_cache = (
            _SurfaceViewCache()
            if geometry_prototype is None
            else geometry_prototype._cache
        )

        initial_surfaces = self._resolve_surfaces()
        initial_curves = self._resolve_curves()
        initial_points = self._resolve_points()
        self._surface_ids = tuple(item.surface_id for item in initial_surfaces)
        initial_curve_ids = tuple(item.curve_id for item in initial_curves)
        initial_point_ids = tuple(item.point_id for item in initial_points)
        if self._section_enabled and len(initial_surfaces) != 1:
            raise QuadricManimError(
                "section compositing requires exactly one finite convex quadric"
            )
        if self._section_enabled:
            initial_plane = self._resolve_section_plane()
            initial_patch = self._resolve_section_patch(
                initial_surfaces[0], initial_plane
            )
            self._section_plane_id: str | None = initial_plane.plane_id
            self._section_patch_id: str | None = initial_patch.patch_id
        else:
            self._section_plane_id = None
            self._section_patch_id = None
        self._allow_curve_subset = allocated_curve_ids is not None
        if allocated_curve_ids is None:
            self._curve_ids = initial_curve_ids
        else:
            if isinstance(allocated_curve_ids, (str, bytes)):
                raise TypeError("allocated_curve_ids must be a sequence of identities")
            canonical_ids: list[str] = []
            for raw in allocated_curve_ids:
                if not isinstance(raw, str) or not raw.strip():
                    raise QuadricManimError(
                        "allocated_curve_ids must contain non-empty strings"
                    )
                canonical_ids.append(raw.strip())
            if len(set(canonical_ids)) != len(canonical_ids):
                raise QuadricManimError("allocated_curve_ids must be unique")
            self._curve_ids = tuple(sorted(canonical_ids))
            unknown = sorted(set(initial_curve_ids) - set(self._curve_ids))
            if unknown:
                raise QuadricManimCapacityError(
                    "initial curves were not preallocated: " + ", ".join(unknown)
                )
        self._allow_point_subset = allocated_point_ids is not None
        if allocated_point_ids is None:
            self._point_ids = initial_point_ids
        else:
            if isinstance(allocated_point_ids, (str, bytes)):
                raise TypeError("allocated_point_ids must be a sequence of identities")
            canonical_point_ids: list[str] = []
            for raw in allocated_point_ids:
                if not isinstance(raw, str) or not raw.strip():
                    raise QuadricManimError(
                        "allocated_point_ids must contain non-empty strings"
                    )
                canonical_point_ids.append(raw.strip())
            if len(set(canonical_point_ids)) != len(canonical_point_ids):
                raise QuadricManimError("allocated_point_ids must be unique")
            self._point_ids = tuple(sorted(canonical_point_ids))
            unknown_points = sorted(set(initial_point_ids) - set(self._point_ids))
            if unknown_points:
                raise QuadricManimCapacityError(
                    "initial points were not preallocated: "
                    + ", ".join(unknown_points)
                )
        auto_boundary_ids: tuple[str, ...] = ()
        if self.boundary_visibility_mode == "unified":
            auto_boundary_ids = surface_boundary_source_ids(
                initial_surfaces,
                self._generator_boundaries,
                include_cap_rims=self.include_surface_boundaries,
                include_silhouettes=self.include_surface_boundaries,
            )
            if self._section_enabled:
                auto_boundary_ids = (
                    *auto_boundary_ids,
                    *(
                        f"boundary:plane:{self._section_plane_id}:edge:{index}"
                        for index in range(4)
                    ),
                )
        extra_boundary_ids: tuple[str, ...] = ()
        if allocated_boundary_ids is not None:
            if isinstance(allocated_boundary_ids, (str, bytes)):
                raise TypeError(
                    "allocated_boundary_ids must be a sequence of identities"
                )
            values = []
            for raw in allocated_boundary_ids:
                if not isinstance(raw, str) or not raw.strip():
                    raise QuadricManimError(
                        "allocated_boundary_ids must contain non-empty strings"
                    )
                values.append(raw.strip())
            if len(set(values)) != len(values):
                raise QuadricManimError("allocated_boundary_ids must be unique")
            extra_boundary_ids = tuple(values)
        self._boundary_source_ids = tuple(
            sorted(set((*auto_boundary_ids, *extra_boundary_ids)))
        )
        if len(self._boundary_source_ids) > limits.max_boundary_sources:
            raise QuadricManimCapacityError(
                "boundary source count exceeds fixed limit "
                f"{limits.max_boundary_sources}"
            )
        self._slot_source_ids = (
            self._curve_ids
            if self.boundary_visibility_mode == "legacy"
            else tuple(sorted(set((*self._curve_ids, *self._boundary_source_ids))))
        )
        if len(self._surface_ids) > limits.max_surfaces:
            raise QuadricManimCapacityError(
                f"surface count exceeds fixed limit {limits.max_surfaces}"
            )
        if len(self._curve_ids) > limits.max_curves:
            raise QuadricManimCapacityError(
                f"curve count exceeds fixed limit {limits.max_curves}"
            )
        if len(self._point_ids) > limits.max_points:
            raise QuadricManimCapacityError(
                f"point count exceeds fixed limit {limits.max_points}"
            )

        estimated_mobjects = estimate_quadric_mobject_count(
            surface_count=len(self._surface_ids),
            source_count=len(self._slot_source_ids),
            max_fragments_per_curve=limits.max_fragments_per_curve,
            section_enabled=self._section_enabled,
            point_count=len(self._point_ids),
        )
        if estimated_mobjects > limits.max_total_mobjects:
            raise QuadricManimCapacityError(
                f"preallocated Mobject count {estimated_mobjects} exceeds fixed "
                f"limit {limits.max_total_mobjects}"
            )

        self._surface_paint_slots = tuple(
            _SurfacePaintSlot() for _ in self._surface_ids
        )
        self._surface_slots = tuple(
            slot.root for slot in self._surface_paint_slots
        )
        self._surface_slot_by_id = {
            surface_id: index for index, surface_id in enumerate(self._surface_ids)
        }
        self._curve_slots = {
            curve_id: _CurveSlots(
                limits.max_fragments_per_curve,
                limits.max_dashes_per_fragment,
            )
            for curve_id in self._slot_source_ids
        }
        self._point_slots = {
            point_id: Dot(radius=self.style.point_radius)
            for point_id in self._point_ids
        }
        for slot in self._point_slots.values():
            slot.set_fill(opacity=0.0)
            slot.set_stroke(opacity=0.0)
        self._fragment_slot_maps: dict[str, dict[str, int]] = {
            curve_id: {} for curve_id in self._slot_source_ids
        }
        self._section_surface_paint_slots = (
            {1: _SurfacePaintSlot(), 4: _SurfacePaintSlot()}
            if self._section_enabled
            else {}
        )
        self._section_slots = (
            tuple(
                self._section_surface_paint_slots[index].root
                if index in self._section_surface_paint_slots
                else VMobject()
                for index in range(10)
            )
            if self._section_enabled
            else ()
        )
        surface_root = VGroup(*self._surface_slots)
        section_root = VGroup(*self._section_slots)
        curve_root = VGroup(
            *(self._curve_slots[key].root for key in self._slot_source_ids)
        )
        point_root = (
            VGroup(*(self._point_slots[key] for key in self._point_ids))
            if self._point_ids
            else None
        )
        self._opacity_sentinel = Line((0, 0, 0), (1.0e-9, 0, 0), buff=0)
        self._opacity_sentinel.set_stroke(width=0.0, opacity=1.0)
        roots = [surface_root, section_root, curve_root]
        if point_root is not None:
            roots.append(point_root)
        self.root = _ManagedQuadricDisplayGroup(
            *roots,
            opacity_sentinel=self._opacity_sentinel,
        )
        self._update_driver = Mobject()
        self._update_driver._tikz_native_parallel_camera_state_consumer = True

        # Manim recognizes time-aware updaters by the literal ``dt`` name.
        def update_display(mobject: Mobject, dt: float) -> None:
            del mobject
            if self._attached and self.automatic_updates:
                self.update(dt)

        self._update_driver.add_updater(update_display)
        self._band = ManagedPainterBand(
            z_band=painter_z_band,
            managed_roots=(self.root,),
        )
        self._frame_transaction: object | None = None

    def _set_frame_transaction(self, transaction: object) -> None:
        """Bind one author-state participant before Scene ownership begins.

        This package-private hook lets a high-level rig join the existing
        display transaction without changing any renderer-neutral contract.
        The participant must expose begin/commit/rollback/cancel methods.
        A commit failure joins the display and committed-input-cache rollback;
        finalize, rollback, and cancel themselves must not fail. Low-level/
        manual controllers do not bind a participant and retain their
        historical behavior.
        """

        if self._attached:
            raise QuadricManimError(
                "frame transaction must be bound before controller attachment"
            )
        for method_name in (
            "_begin_quadric_frame",
            "_commit_quadric_frame",
            "_finalize_quadric_frame",
            "_rollback_quadric_frame",
            "_cancel_quadric_frame",
        ):
            if not callable(getattr(transaction, method_name, None)):
                raise TypeError(
                    f"frame transaction must provide {method_name}()"
                )
        self._frame_transaction = transaction

    def _begin_bound_frame_transaction(self) -> object | None:
        if self._frame_transaction is None:
            return None
        return self._frame_transaction._begin_quadric_frame()

    def _commit_bound_frame_transaction(self, token: object | None) -> None:
        if self._frame_transaction is not None:
            self._frame_transaction._commit_quadric_frame(token)

    def _finalize_bound_frame_transaction(self, token: object | None) -> None:
        if self._frame_transaction is not None:
            self._frame_transaction._finalize_quadric_frame(token)

    def _rollback_bound_frame_transaction(self, token: object | None) -> None:
        if self._frame_transaction is not None:
            self._frame_transaction._rollback_quadric_frame(token)

    def _cancel_bound_frame_transaction(self) -> None:
        if self._frame_transaction is not None:
            self._frame_transaction._cancel_quadric_frame()

    def _resolve_surfaces(self) -> tuple[QuadricSurfaceSpec, ...]:
        value = (
            self._surface_input()
            if callable(self._surface_input)
            else self._surface_input
        )
        return _surface_items(value)

    def _resolve_paint_policy(self) -> QuadricPaintPolicy:
        source = self._paint_policy_input
        value = source() if callable(source) else source
        try:
            return QuadricPaintPolicy(value)
        except (TypeError, ValueError) as exc:
            raise QuadricManimError(
                "paint_policy must resolve to 'physical', 'diagrammatic', or "
                "'depth_aware_diagrammatic'"
            ) from exc

    def _resolve_occluding_surfaces(
        self,
        surfaces: tuple[QuadricSurfaceSpec, ...],
    ) -> tuple[QuadricSurfaceSpec, ...]:
        source = self._occluding_surface_input
        if source is None:
            return surfaces
        raw = source() if callable(source) else source
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise QuadricManimError(
                "occluding_surface_ids must resolve to a sequence"
            )
        identities: list[str] = []
        for value in raw:
            if not isinstance(value, str) or not value.strip():
                raise QuadricManimError(
                    "occluding_surface_ids must contain non-empty strings"
                )
            identities.append(value.strip())
        if len(set(identities)) != len(identities):
            raise QuadricManimError("occluding_surface_ids must be unique")
        by_id = {item.surface_id: item for item in surfaces}
        unknown = sorted(set(identities) - set(by_id))
        if unknown:
            raise QuadricManimCapacityError(
                "occluding_surface_ids reference unallocated surfaces: "
                + ", ".join(unknown)
            )
        if self.boundary_visibility_mode != "unified" and set(identities) != set(by_id):
            raise QuadricManimError(
                "selective occlusion participation requires unified boundary visibility"
            )
        return tuple(by_id[item] for item in sorted(identities))

    def _resolve_surface_opacity_mapping(
        self,
        source: SurfaceOpacityInput,
        *,
        input_label: str,
        item_label: str,
    ) -> dict[str, float]:
        expected = tuple(self._surface_ids)
        if source is None:
            return {surface_id: 1.0 for surface_id in expected}
        raw = source() if callable(source) else source
        if not isinstance(raw, Mapping):
            raise QuadricManimError(
                f"{input_label} must resolve to a mapping"
            )
        result: dict[str, float] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise QuadricManimError(
                    f"{input_label} keys must be non-empty surface identities"
                )
            surface_id = key.strip()
            if surface_id not in expected:
                raise QuadricManimCapacityError(
                    f"{item_label} references unallocated surface {surface_id!r}"
                )
            opacity = _non_negative(value, f"{item_label} for {surface_id!r}")
            if opacity > 1.0:
                raise QuadricManimError(f"{item_label} must not exceed 1")
            result[surface_id] = opacity
        missing = sorted(set(expected) - set(result))
        if missing:
            raise QuadricManimError(
                f"{input_label} omitted allocated surfaces: " + ", ".join(missing)
            )
        return {surface_id: result[surface_id] for surface_id in expected}

    def _resolve_surface_opacities(self) -> dict[str, float]:
        return self._resolve_surface_opacity_mapping(
            self._surface_opacity_input,
            input_label="surface_opacities",
            item_label="surface opacity",
        )

    def _resolve_surface_stroke_opacities(self) -> dict[str, float]:
        return self._resolve_surface_opacity_mapping(
            self._surface_stroke_opacity_input,
            input_label="surface_stroke_opacities",
            item_label="surface stroke opacity",
        )

    @staticmethod
    def _resolve_scalar_opacity(
        source: ScalarOpacityInput,
        label: str,
    ) -> float:
        if source is None:
            return 1.0
        value = source() if callable(source) else source
        opacity = _non_negative(value, label)
        if opacity > 1.0:
            raise QuadricManimError(f"{label} must not exceed 1")
        return opacity

    def _resolve_curves(self) -> tuple[AnalyticCurve3D, ...]:
        value = (
            self._curve_input() if callable(self._curve_input) else self._curve_input
        )
        return _curve_items(value)

    def _resolve_points(self) -> tuple[PointMarker3D, ...]:
        value = (
            self._point_input() if callable(self._point_input) else self._point_input
        )
        return _point_items(value)

    def _resolve_section_plane(self) -> SectionPlane:
        source = self._section_plane_input
        value = source() if callable(source) else source
        if not isinstance(value, SectionPlane):
            raise QuadricManimError(
                "section_plane must resolve to a SectionPlane"
            )
        expected = getattr(self, "_section_plane_id", None)
        if expected is not None and value.plane_id != expected:
            raise QuadricManimError(
                "section_plane identity changed while the controller was active"
            )
        return value

    def _resolve_section_patch(
        self,
        surface: QuadricSurfaceSpec,
        plane: SectionPlane,
    ) -> PlaneDisplayPatchSpec:
        source = self._section_patch_input
        if source is None:
            try:
                value = fit_plane_display_patch(
                    f"{plane.plane_id}:auto-display-patch",
                    plane,
                    (surface,),
                    margin_ratio=self.section_patch_margin,
                ).patch
            except PlanePatchFitError as exc:
                raise QuadricManimError(
                    f"automatic section-plane patch fitting failed: {exc}"
                ) from exc
        else:
            value = source() if callable(source) else source
        if not isinstance(value, PlaneDisplayPatchSpec):
            raise QuadricManimError(
                "section_patch must resolve to a PlaneDisplayPatchSpec"
            )
        if value.plane_id != plane.plane_id:
            raise QuadricManimError(
                "section_patch plane_id does not match section_plane"
            )
        expected = getattr(self, "_section_patch_id", None)
        if expected is not None and value.patch_id != expected:
            raise QuadricManimError(
                "section_patch identity changed while the controller was active"
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

    def _prepare_cone_fill_for_surface(
        self,
        surface: QuadricSurfaceSpec,
        view: ParallelView,
    ) -> _PreparedConeFill | None:
        if not self.style.cone_component_shading or not isinstance(
            surface, ConeSpec
        ):
            return None
        try:
            layers = build_cone_projection_layers(
                surface,
                view,
                max_chord_error=self.max_chord_error,
                max_segments=self.limits.max_surface_segments,
            )
        except ProjectionSubdivisionError as exc:
            raise QuadricManimCapacityError(str(exc)) from exc
        except ProjectionProxyError as exc:
            raise QuadricManimError(
                f"cone component shading failed: {exc}"
            ) from exc
        return _prepared_cone_fill(layers)

    def _prepare_surface_component_fills(
        self,
        surfaces: Sequence[QuadricSurfaceSpec],
        view: ParallelView,
        surface_view_signature: bytes,
        performance_attempt: _PerformanceAttempt | None,
    ) -> dict[str, _PreparedConeFill | None]:
        signature = _display_digest(
            "quadric-surface-component-fills-v1",
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
                (surface.surface_id, self._prepare_cone_fill_for_surface(surface, view))
                for surface in surfaces
            )
            self._surface_view_cache.store(
                "component_fills",
                signature,
                prepared,
            )
            return dict(prepared)

    def _resolve_curve_opacities(
        self, active_curve_ids: Sequence[str]
    ) -> dict[str, float]:
        active = tuple(active_curve_ids)
        if self._curve_opacity_input is None:
            return {curve_id: 1.0 for curve_id in active}
        raw = (
            self._curve_opacity_input()
            if callable(self._curve_opacity_input)
            else self._curve_opacity_input
        )
        if not isinstance(raw, Mapping):
            raise QuadricManimError("curve_opacities must resolve to a mapping")
        result: dict[str, float] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise QuadricManimError(
                    "curve_opacities keys must be non-empty curve identities"
                )
            curve_id = key.strip()
            if curve_id not in self._curve_ids:
                raise QuadricManimCapacityError(
                    f"curve opacity references unallocated curve {curve_id!r}"
                )
            opacity = _non_negative(value, f"curve opacity for {curve_id!r}")
            if opacity > 1.0:
                raise QuadricManimError("curve opacity must not exceed 1")
            result[curve_id] = opacity
        missing = sorted(set(active) - set(result))
        if missing:
            raise QuadricManimError(
                "curve_opacities omitted active curves: " + ", ".join(missing)
            )
        return {curve_id: result[curve_id] for curve_id in active}

    def _resolve_point_opacities(
        self, active_point_ids: Sequence[str]
    ) -> dict[str, float]:
        active = tuple(active_point_ids)
        if self._point_opacity_input is None:
            return {point_id: 1.0 for point_id in active}
        raw = (
            self._point_opacity_input()
            if callable(self._point_opacity_input)
            else self._point_opacity_input
        )
        if not isinstance(raw, Mapping):
            raise QuadricManimError("point_opacities must resolve to a mapping")
        result: dict[str, float] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise QuadricManimError(
                    "point_opacities keys must be non-empty point identities"
                )
            point_id = key.strip()
            if point_id not in self._point_ids:
                raise QuadricManimCapacityError(
                    f"point opacity references unallocated point {point_id!r}"
                )
            opacity = _non_negative(value, f"point opacity for {point_id!r}")
            if opacity > 1.0:
                raise QuadricManimError("point opacity must not exceed 1")
            result[point_id] = opacity
        missing = sorted(set(active) - set(result))
        if missing:
            raise QuadricManimError(
                "point_opacities omitted active points: " + ", ".join(missing)
            )
        return {point_id: result[point_id] for point_id in active}

    def _resolve_boundary_opacities(self) -> dict[str, float]:
        expected = set(self._boundary_source_ids)
        if self._boundary_opacity_input is None:
            return {source_id: 1.0 for source_id in self._boundary_source_ids}
        raw = (
            self._boundary_opacity_input()
            if callable(self._boundary_opacity_input)
            else self._boundary_opacity_input
        )
        if not isinstance(raw, Mapping):
            raise QuadricManimError("boundary_opacities must resolve to a mapping")
        result: dict[str, float] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise QuadricManimError(
                    "boundary_opacities keys must be non-empty source identities"
                )
            source_id = key.strip()
            if source_id not in expected:
                raise QuadricManimCapacityError(
                    f"boundary opacity references unallocated source {source_id!r}"
                )
            opacity = _non_negative(
                value,
                f"boundary opacity for {source_id!r}",
            )
            if opacity > 1.0:
                raise QuadricManimError("boundary opacity must not exceed 1")
            result[source_id] = opacity
        missing = sorted(expected - set(result))
        if missing:
            raise QuadricManimError(
                "boundary_opacities omitted allocated sources: " + ", ".join(missing)
            )
        return {source_id: result[source_id] for source_id in self._boundary_source_ids}

    def _resolve_frame_inputs(self) -> _ResolvedQuadricFrameInputs:
        surfaces = self._resolve_surfaces()
        occluding_surfaces = self._resolve_occluding_surfaces(surfaces)
        surface_opacities = self._resolve_surface_opacities()
        surface_stroke_opacities = self._resolve_surface_stroke_opacities()
        curves = self._resolve_curves()
        points = self._resolve_points()
        self._validate_fixed_topology(surfaces, curves, points)
        active_curve_ids = tuple(item.curve_id for item in curves)
        curve_opacities = self._resolve_curve_opacities(active_curve_ids)
        active_point_ids = tuple(item.point_id for item in points)
        point_opacities = self._resolve_point_opacities(active_point_ids)
        boundary_opacities = self._resolve_boundary_opacities()
        section_plane_fill_opacity = self._resolve_scalar_opacity(
            self._section_plane_fill_opacity_input,
            "section plane fill opacity",
        )
        section_plane_stroke_opacity = self._resolve_scalar_opacity(
            self._section_plane_stroke_opacity_input,
            "section plane stroke opacity",
        )
        paint_policy = self._resolve_paint_policy()
        projection_frame = self._resolve_projection_frame()
        view = self._resolve_view(projection_frame)
        display_offset = _projection_display_offset(
            self.scene,
            projection_frame,
            self.display_offset,
        )
        if self._section_enabled:
            section_plane = self._resolve_section_plane()
            section_patch = self._resolve_section_patch(
                surfaces[0], section_plane
            )
        else:
            section_plane = None
            section_patch = None
        surface_view_signature = _display_digest(
            "quadric-surface-view-v1",
            surfaces,
            tuple(item.surface_id for item in occluding_surfaces),
            view.matrix,
            self.context,
            self.surface_constraints,
            self.surface_order_mode,
            self.max_chord_error,
            self.limits.max_surface_segments,
        )
        geometry_signature = _display_digest(
            "quadric-frame-inputs-v1",
            surface_view_signature,
            curves,
            points,
            section_plane,
            section_patch,
            paint_policy,
            self.style,
            self.boundary_visibility_mode,
            self.include_surface_boundaries,
            self.legacy_surface_stroke_fallback,
            self._generator_boundaries,
            self.boundary_styles,
            self.section_id,
            self.section_coefficient_tolerance,
            self.max_chord_error,
            self.section_max_screen_error,
            self.section_compositing_limits,
            self.boundary_section_limits,
            self.limits,
            display_offset,
        )
        draw_signature = _display_digest(
            "quadric-frame-draw-v1",
            curve_opacities,
            point_opacities,
            boundary_opacities,
            surface_opacities,
            surface_stroke_opacities,
            section_plane_fill_opacity,
            section_plane_stroke_opacity,
        )
        return _ResolvedQuadricFrameInputs(
            surfaces,
            occluding_surfaces,
            surface_opacities,
            surface_stroke_opacities,
            curves,
            curve_opacities,
            points,
            point_opacities,
            boundary_opacities,
            section_plane_fill_opacity,
            section_plane_stroke_opacity,
            paint_policy,
            view,
            display_offset,
            section_plane,
            section_patch,
            surface_view_signature,
            geometry_signature,
            draw_signature,
        )

    @property
    def display_offset(self) -> tuple[float, float]:
        """Return the validated display-only screen translation."""

        return self._display_offset

    @display_offset.setter
    def display_offset(self, value: Sequence[float]) -> None:
        self._display_offset = _display_offset(value)

    @property
    def automatic_updates(self) -> bool:
        """Whether the time-aware driver owns frame updates.

        The mode is immutable after construction.  Coordinated bindings use a
        no-op time-aware driver so Cairo still treats the display as dynamic
        while exactly one external transaction commits each output frame.
        """

        return self._automatic_updates

    @property
    def legacy_surface_stroke_fallback(self) -> bool:
        """Whether unified rendering draws one uncertified legacy outline.

        The opt-in exists for adapters that deliberately exclude intrinsic
        surface boundaries but still need a static teaching-outline style.
        It is immutable after construction and defaults to ``False`` so
        ``include_surface_boundaries=False`` keeps its historical meaning.
        """

        return self._legacy_surface_stroke_fallback

    def _validate_fixed_topology(
        self,
        surfaces: Sequence[QuadricSurfaceSpec],
        curves: Sequence[AnalyticCurve3D],
        points: Sequence[PointMarker3D],
    ) -> None:
        surface_ids = tuple(item.surface_id for item in surfaces)
        curve_ids = tuple(item.curve_id for item in curves)
        point_ids = tuple(item.point_id for item in points)
        if surface_ids != self._surface_ids:
            raise QuadricManimCapacityError(
                "surface identities changed after fixed-capacity allocation"
            )
        if self._allow_curve_subset:
            unknown = sorted(set(curve_ids) - set(self._curve_ids))
            if unknown:
                raise QuadricManimCapacityError(
                    "curve identities were not preallocated: " + ", ".join(unknown)
                )
        elif curve_ids != self._curve_ids:
            raise QuadricManimCapacityError(
                "curve identities changed after fixed-capacity allocation"
            )
        if self._allow_point_subset:
            unknown_points = sorted(set(point_ids) - set(self._point_ids))
            if unknown_points:
                raise QuadricManimCapacityError(
                    "point identities were not preallocated: "
                    + ", ".join(unknown_points)
                )
        elif point_ids != self._point_ids:
            raise QuadricManimCapacityError(
                "point identities changed after fixed-capacity allocation"
            )

    def _assign_fragment_slots(
        self, curve_id: str, active_ids: Sequence[str]
    ) -> dict[str, int]:
        active = tuple(active_ids)
        if len(active) > self.limits.max_fragments_per_curve:
            raise QuadricManimCapacityError(
                f"curve {curve_id!r} has {len(active)} painted fragments; fixed "
                f"capacity is {self.limits.max_fragments_per_curve}"
            )
        previous = self._fragment_slot_maps[curve_id]
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
            raise QuadricManimError(
                f"boundary source {source.source_id!r} references unknown "
                f"style_id {style_id!r}"
            ) from exc

    def _boundary_sources_for_frame(
        self,
        surfaces: Sequence[QuadricSurfaceSpec],
        curves: Sequence[AnalyticCurve3D],
        view: ParallelView,
        plane: SectionPlane | None,
        patch: PlaneDisplayPatchSpec | None,
        *,
        surface_sources: Sequence[QuadricBoundarySource] | None = None,
        include_plane_outline: bool = True,
    ) -> tuple[QuadricBoundarySource, ...]:
        authoritative_section_curves: tuple[AnalyticCurve3D, ...] | None = None
        if (
            plane is not None
            and len(surfaces) == 1
            and self.section_id is not None
        ):
            try:
                authoritative_section_curves = (
                    compute_quadric_section_boundary_curves(
                        self.section_id,
                        surfaces[0],
                        plane,
                        context=self.context,
                        coefficient_tolerance=(
                            self.section_coefficient_tolerance
                        ),
                    )
                )
            except (QuadricSectionError, ValueError) as exc:
                raise QuadricManimError(
                    "authoritative section-boundary preparation failed: "
                    f"{exc}"
                ) from exc
        result = [
            (
                section_curve_boundary_source(
                    curve,
                    surfaces[0],
                    plane,
                    section_id=self.section_id,
                    authoritative_curves=authoritative_section_curves,
                    context=self.context,
                    style_id="style:curve",
                )
                if (
                    plane is not None
                    and len(surfaces) == 1
                    and self.section_id is not None
                )
                else curve_boundary_source(
                    curve,
                    style_id="style:curve",
                )
            )
            for curve in curves
        ]
        if surface_sources is None:
            surface_sources = self._surface_boundary_sources(surfaces, view)
        result.extend(surface_sources)
        if include_plane_outline and plane is not None and patch is not None:
            result.extend(plane_outline_sources(plane, patch))
        result.sort(key=lambda item: item.source_id)
        ids = tuple(item.source_id for item in result)
        if len(set(ids)) != len(ids):
            raise QuadricManimError(
                "user curves and semantic boundaries have duplicate identities"
            )
        unknown = sorted(set(ids) - set(self._slot_source_ids))
        if unknown:
            raise QuadricManimCapacityError(
                "semantic boundary identities were not preallocated: "
                + ", ".join(unknown)
            )
        return tuple(result)

    def _surface_boundary_sources(
        self,
        surfaces: Sequence[QuadricSurfaceSpec],
        view: ParallelView,
    ) -> tuple[QuadricBoundarySource, ...]:
        if self.include_surface_boundaries:
            return build_surface_boundary_sources(
                surfaces,
                view,
                self._generator_boundaries,
                include_cap_rims=True,
                include_silhouettes=True,
                context=self.context,
            )
        if self._generator_boundaries:
            return build_surface_boundary_sources(
                surfaces,
                view,
                self._generator_boundaries,
                include_cap_rims=False,
                include_silhouettes=False,
                context=self.context,
            )
        return ()

    @staticmethod
    def _section_anchors(frame: QuadricSectionCompositingFrame) -> BoundarySectionAnchors:
        items = frame.paint_items
        return BoundarySectionAnchors(
            items.plane_behind,
            items.plane_outline_behind,
            items.surface_back,
            items.plane_outside,
            items.plane_outline_outside,
            items.plane_between,
            items.plane_outline_between,
            items.surface_front,
            items.plane_front,
            items.plane_outline,
        )

    @staticmethod
    def _plane_outline_visibility(
        frame: QuadricSectionCompositingFrame,
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
                    () if visible else (frame.surface_id,),
                    fragment.role.value,
                )
            )
        return {
            source_id: tuple(
                sorted(values, key=lambda item: item.interval.start)
            )
            for source_id, values in grouped.items()
        }

    def _boundary_crossings(
        self,
        sources: Sequence[QuadricBoundarySource],
        spans: Mapping[str, Sequence[QuadricBoundaryVisibilitySpan]],
        view: ParallelView,
        *,
        paint_policy: QuadricPaintPolicy,
        cached_source_ids: frozenset[str] = frozenset(),
        cached_crossings: Sequence[object] = (),
        rank_one_section_source_group: (
            QuadricRankOneSectionSourceGroup | None
        ) = None,
    ) -> tuple[object, ...]:
        from itertools import combinations

        result = list(cached_crossings)
        rank_one_source_ids = (
            frozenset()
            if rank_one_section_source_group is None
            else rank_one_section_source_group.source_ids
        )
        rank_one_point_source_ids = (
            frozenset()
            if rank_one_section_source_group is None
            else rank_one_section_source_group.point_source_ids
        )

        def pair_is_certified_rank_one_overlap(
            first: QuadricBoundarySource,
            second: QuadricBoundarySource,
        ) -> bool:
            if rank_one_section_source_group is None:
                return False
            if (
                first.source_id in rank_one_point_source_ids
                or second.source_id in rank_one_point_source_ids
            ):
                # A certified POINT member paints no stroke, so it cannot own a
                # fragment-level crossing even when another projected curve
                # happens to pass through the same screen coordinate.
                return True
            first_is_section = first.source_id in rank_one_source_ids
            second_is_section = second.source_id in rank_one_source_ids
            if first_is_section and second_is_section:
                # Both world curves are analytically certified members of the
                # same surface/plane section.  Their rank-one images overlap by
                # construction; asking the generic 2D root finder to rediscover
                # that fact produces duplicate roots and zero-length pieces.
                return True
            if not first_is_section and not second_is_section:
                return False
            other = second if first_is_section else first
            return (
                other.owner_surface_id
                == rank_one_section_source_group.surface_id
                and other.semantic_kind
                in {
                    BoundarySemanticKind.SURFACE_BOUNDARY,
                    BoundarySemanticKind.TRUE_SILHOUETTE,
                }
            )

        for first, second in combinations(sources, 2):
            if (
                first.source_id in cached_source_ids
                and second.source_id in cached_source_ids
            ):
                continue
            if pair_is_certified_rank_one_overlap(first, second):
                continue
            active_intervals = None
            if paint_policy is QuadricPaintPolicy.PHYSICAL:
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
                if same_owner and {
                    first.semantic_kind,
                    second.semantic_kind,
                } <= {
                    BoundarySemanticKind.SURFACE_BOUNDARY,
                    BoundarySemanticKind.TRUE_SILHOUETTE,
                }:
                    continue
                raise QuadricManimError(
                    "semantic boundary crossings cannot be certified: "
                    f"{first.source_id!r}, {second.source_id!r}: {exc}"
                ) from exc
        by_id = {item.crossing_id: item for item in result}
        return tuple(by_id[key] for key in sorted(by_id))

    def _prepare_static_surface_boundaries(
        self,
        surfaces: Sequence[QuadricSurfaceSpec],
        occluding_surfaces: Sequence[QuadricSurfaceSpec],
        view: ParallelView,
        paint_policy: QuadricPaintPolicy,
        surface_view_signature: bytes,
        performance_attempt: _PerformanceAttempt | None,
    ) -> tuple[
        tuple[QuadricBoundarySource, ...],
        Mapping[str, tuple[QuadricBoundaryVisibilitySpan, ...]],
        tuple[object, ...],
    ]:
        signature = _display_digest(
            "quadric-static-surface-boundaries-v1",
            surface_view_signature,
            self.include_surface_boundaries,
            self._generator_boundaries,
            paint_policy,
        )
        cache_name = f"static_boundaries:{paint_policy.value}"
        hit, cached = self._surface_view_cache.lookup(
            cache_name,
            signature,
        )
        if hit:
            if performance_attempt is not None:
                performance_attempt.cache_hit("static_surface_boundaries")
            return cached  # type: ignore[return-value]
        if performance_attempt is not None:
            performance_attempt.cache_miss("static_surface_boundaries")
        sources = self._surface_boundary_sources(surfaces, view)
        with _performance_stage(performance_attempt, "boundary_visibility"):
            spans = compute_boundary_visibility(
                sources,
                occluding_surfaces,
                view,
                context=self.context,
            )
        with _performance_stage(performance_attempt, "curve_crossings"):
            crossings = self._boundary_crossings(
                sources,
                spans,
                view,
                paint_policy=paint_policy,
            )
        prepared = (sources, dict(spans), crossings)
        self._surface_view_cache.store(
            cache_name,
            signature,
            prepared,
        )
        return prepared

    def _prepare_unified_numeric(
        self,
        surfaces: tuple[QuadricSurfaceSpec, ...],
        occluding_surfaces: tuple[QuadricSurfaceSpec, ...],
        curves: tuple[AnalyticCurve3D, ...],
        view: ParallelView,
        paint_policy: QuadricPaintPolicy,
        curve_opacities: Mapping[str, float],
        boundary_opacities: Mapping[str, float],
        section_plane: SectionPlane | None,
        section_patch: PlaneDisplayPatchSpec | None,
        surface_view_signature: bytes,
        performance_attempt: _PerformanceAttempt | None,
    ) -> _PreparedNumericFrame:
        component_fills = self._prepare_surface_component_fills(
            surfaces,
            view,
            surface_view_signature,
            performance_attempt,
        )
        with _performance_stage(
            performance_attempt, "surface_proxy_global_frame"
        ):
            hit, cached = self._surface_view_cache.lookup(
                "unified_surface_base",
                surface_view_signature,
            )
            if hit:
                if performance_attempt is not None:
                    performance_attempt.cache_hit("surface_view_base")
                frame, global_frame = cached  # type: ignore[misc]
            else:
                if performance_attempt is not None:
                    performance_attempt.cache_miss("surface_view_base")
                global_frame: GlobalQuadricFrame | None = None
                if self.surface_order_mode == "automatic":
                    try:
                        global_frame = compute_global_quadric_frame(
                            (),
                            surfaces,
                            view,
                            context=self.context,
                            paint_policy=QuadricPaintPolicy.PHYSICAL,
                            max_chord_error=self.max_chord_error,
                            max_segments=self.limits.max_surface_segments,
                            additional_surface_constraints=self.surface_constraints,
                        )
                    except (
                        ProjectionSubdivisionError,
                        ProjectionProxyError,
                        GlobalQuadricOcclusionError,
                    ) as exc:
                        raise QuadricManimError(
                            f"unified surface preparation failed: {exc}"
                        ) from exc
                    frame = global_frame.frame
                else:
                    try:
                        proxies = tuple(
                            build_opaque_projection_proxy(
                                surface,
                                view,
                                patch_id=(
                                    f"{surface.surface_id}:opaque-projection"
                                ),
                                max_chord_error=self.max_chord_error,
                                max_segments=self.limits.max_surface_segments,
                            )
                            for surface in surfaces
                        )
                        visibility = compute_quadric_visibility(
                            (), surfaces, view, context=self.context
                        )
                        frame = compute_quadric_compositing(
                            visibility,
                            proxies,
                            paint_policy=QuadricPaintPolicy.PHYSICAL,
                            surface_constraints=self.surface_constraints,
                        )
                    except (
                        ProjectionSubdivisionError,
                        ProjectionProxyError,
                        QuadricCompositingError,
                    ) as exc:
                        raise QuadricManimError(
                            "unified explicit surface preparation failed: "
                            f"{exc}"
                        ) from exc
                self._surface_view_cache.store(
                    "unified_surface_base",
                    surface_view_signature,
                    (frame, global_frame),
                )

        surface_plans: list[_PreparedSurface] = []
        item_mobjects: dict[str, Mobject] = {}
        section_layers: _PreparedSectionLayers | None = None
        section_frame: QuadricSectionCompositingFrame | None = None
        section_geometry_signature: bytes | None = None
        plane: SectionPlane | None = None
        patch: PlaneDisplayPatchSpec | None = None
        if self._section_enabled:
            surface = surfaces[0]
            plane = section_plane
            patch = section_patch
            if plane is None or patch is None:
                raise QuadricManimError(
                    "section inputs were not resolved before unified preparation"
                )
            try:
                section_geometry_signature = _display_digest(
                    "quadric-section-geometry-prototype-v1",
                    surface_view_signature,
                    plane,
                    patch,
                    self.context,
                    self.section_max_screen_error,
                    self.section_compositing_limits,
                )
                with _performance_stage(
                    performance_attempt, "section_compositing"
                ):
                    hit, cached = self._surface_view_cache.lookup(
                        "section_geometry",
                        section_geometry_signature,
                    )
                    if hit:
                        if performance_attempt is not None:
                            performance_attempt.cache_hit(
                                "shared_section_geometry"
                            )
                        geometry_frame, contours = cached  # type: ignore[misc]
                        section_frame = repaint_quadric_section_compositing(
                            geometry_frame,
                            frame,
                        )
                    else:
                        if performance_attempt is not None:
                            performance_attempt.cache_miss(
                                "shared_section_geometry"
                            )
                        section_frame = compute_quadric_section_compositing(
                            frame,
                            surface,
                            plane,
                            patch,
                            view,
                            context=self.context,
                            max_screen_error=self.section_max_screen_error,
                            limits=self.section_compositing_limits,
                        )
                with _performance_stage(performance_attempt, "contour_union"):
                    if not hit:
                        contours = quadric_plane_fragment_contours(section_frame)
                        self._surface_view_cache.store(
                            "section_geometry",
                            section_geometry_signature,
                            (section_frame, contours),
                        )
            except QuadricSectionCompositingError as exc:
                raise QuadricManimError(
                    f"unified section preparation failed: {exc}"
                ) from exc
            surface_points = np.asarray(
                [
                    (x, y, 0.0)
                    for x, y in section_frame.surface_proxy.boundary_points
                ],
                dtype=float,
            )
            plane_polygons = {
                role: tuple(
                    np.asarray([(x, y, 0.0) for x, y in contour], dtype=float)
                    for contour in contours[role]
                )
                for role in PlaneDepthRole
            }
            plane_outline_paths = {
                role: (
                    tuple(
                        np.asarray(
                            (
                                (*fragment.screen_start, 0.0),
                                (*fragment.screen_end, 0.0),
                            ),
                            dtype=float,
                        )
                        for fragment in section_frame.outline_fragments_by_role[role]
                    )
                    if section_frame.projection_kind
                    is PlanePatchProjectionKind.LINE
                    else ()
                )
                for role in PlaneDepthRole
            }
            section_layers = _PreparedSectionLayers(
                section_frame,
                surface_points,
                plane_polygons,
                plane_outline_paths,
                component_fills[surface.surface_id],
            )
            item_mobjects.update(
                {
                    item_id: self._section_slots[index]
                    for index, item_id in enumerate(
                        section_frame.paint_items.ordered
                    )
                }
            )
            parent_ids = section_frame.paint_items.ordered
            parent_relations = section_frame.order_relations
            surface_item_by_id = {
                surface.surface_id: section_frame.paint_items.surface_front
            }
        else:
            for item in frame.surface_items:
                slot_index = self._surface_slot_by_id[item.surface_id]
                points = np.asarray(
                    [(x, y, 0.0) for x, y in item.proxy.boundary_points],
                    dtype=float,
                )
                surface_plans.append(
                    _PreparedSurface(
                        item.item_id,
                        item.surface_id,
                        slot_index,
                        points,
                        component_fills[item.surface_id],
                    )
                )
                item_mobjects[item.item_id] = self._surface_slots[slot_index]
            parent_ids = frame.draw_order
            parent_relations = frame.order_relations
            surface_item_by_id = {
                item.surface_id: item.item_id for item in frame.surface_items
            }

        static_sources, static_spans, static_crossings = (
            self._prepare_static_surface_boundaries(
                surfaces,
                occluding_surfaces,
                view,
                paint_policy,
                surface_view_signature,
                performance_attempt,
            )
        )
        static_source_ids = frozenset(item.source_id for item in static_sources)
        with _performance_stage(performance_attempt, "boundary_visibility"):
            sources = self._boundary_sources_for_frame(
                surfaces,
                curves,
                view,
                plane,
                patch,
                surface_sources=static_sources,
                include_plane_outline=(
                    section_frame is None
                    or section_frame.projection_kind
                    is PlanePatchProjectionKind.AREA
                ),
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
                            occluding_surfaces,
                            view,
                            context=self.context,
                        )
                    )
            except Exception as exc:
                raise QuadricManimError(
                    f"semantic boundary visibility failed: {exc}"
                ) from exc
            if (
                section_frame is not None
                and section_frame.projection_kind is PlanePatchProjectionKind.AREA
            ):
                spans.update(self._plane_outline_visibility(section_frame))
        rank_one_section_source_group: (
            QuadricRankOneSectionSourceGroup | None
        ) = None
        if (
            section_frame is not None
            and section_frame.projection_kind is PlanePatchProjectionKind.LINE
        ):
            try:
                with _performance_stage(
                    performance_attempt, "rank_one_section_certification"
                ):
                    rank_one_section_source_group = (
                        certify_rank_one_section_boundary_sources(
                            sources,
                            section_frame,
                            view,
                            surface=surfaces[0],
                            context=self.context,
                        )
                    )
            except QuadricBoundaryCompositingError as exc:
                raise QuadricManimError(
                    f"rank-one section boundary certification failed: {exc}"
                ) from exc
        with _performance_stage(performance_attempt, "curve_crossings"):
            crossings = self._boundary_crossings(
                sources,
                spans,
                view,
                paint_policy=paint_policy,
                cached_source_ids=static_source_ids,
                cached_crossings=static_crossings,
                rank_one_section_source_group=rank_one_section_source_group,
            )
        with _performance_stage(performance_attempt, "boundary_section_spans"):
            if section_frame is None:
                section_spans = {}
            elif section_frame.projection_kind is PlanePatchProjectionKind.LINE:
                # The rank-one certificate above replaces area placement.  A
                # finite LINE patch has no fill that can occlude or bracket a
                # boundary, so there are deliberately no section-plane spans.
                section_spans = {}
            else:
                if section_geometry_signature is None:
                    raise QuadricManimError(
                        "section geometry signature was not prepared"
                    )
                section_span_signature = _display_digest(
                    "quadric-boundary-section-spans-prototype-v1",
                    section_geometry_signature,
                    sources,
                    spans,
                    crossings,
                    self.context,
                    self.boundary_section_limits,
                )
                hit, cached = self._surface_view_cache.lookup(
                    "boundary_section_spans",
                    section_span_signature,
                )
                if hit:
                    if performance_attempt is not None:
                        performance_attempt.cache_hit(
                            "shared_boundary_section_spans"
                        )
                    section_spans = cached  # type: ignore[assignment]
                else:
                    if performance_attempt is not None:
                        performance_attempt.cache_miss(
                            "shared_boundary_section_spans"
                        )
                    section_spans = _compute_boundary_section_spans_with_contours(
                        sources,
                        section_frame,
                        view,
                        crossings,
                        surface=surfaces[0],
                        visibility_spans_by_source=spans,
                        plane_fragment_contours=contours,
                        context=self.context,
                        limits=self.boundary_section_limits,
                    )
                    self._surface_view_cache.store(
                        "boundary_section_spans",
                        section_span_signature,
                        section_spans,
                    )
        try:
            with _performance_stage(
                performance_attempt, "boundary_painter_graph"
            ):
                boundary_frame = compute_quadric_boundary_compositing(
                    sources,
                    spans,
                    paint_policy=paint_policy,
                    parent_item_ids=parent_ids,
                    parent_relations=parent_relations,
                    surface_item_by_id=surface_item_by_id,
                    crossings=crossings,
                    section_anchors=(
                        None
                        if section_frame is None
                        else self._section_anchors(section_frame)
                    ),
                    section_spans_by_source=section_spans,
                    rank_one_section_source_group=(
                        rank_one_section_source_group
                    ),
                )
        except QuadricBoundaryCompositingError as exc:
            raise QuadricManimError(
                f"semantic boundary painter graph failed: {exc}"
            ) from exc

        boundary_batch = _prepare_boundary_fragments(
            sources=sources,
            frame=boundary_frame,
            view=view,
            style_for_source=self._boundary_style_for_source,
            previous_slot_maps=self._fragment_slot_maps,
            curve_slots=self._curve_slots,
            slot_source_ids=self._slot_source_ids,
            max_chord_error=self.max_chord_error,
            limits=self.limits,
            performance_attempt=performance_attempt,
        )
        item_mobjects.update(boundary_batch.item_mobjects)
        resolved_boundary_opacities = {
            item.source_id: (
                curve_opacities[item.source_id]
                if item.source_id in curve_opacities
                else boundary_opacities.get(item.source_id, 1.0)
            )
            for item in sources
        }
        if set(item_mobjects) != set(boundary_frame.draw_order):
            raise QuadricManimError(
                "unified Manim items do not cover boundary draw_order"
            )
        if performance_attempt is not None:
            performance_attempt.set_count("surface_count", len(surfaces))
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
                "plane_fragment_count",
                0 if section_frame is None else len(section_frame.plane_fragments),
            )
            performance_attempt.set_count(
                "ray_classification_count",
                0 if section_frame is None else section_frame.ray_classification_count,
            )
        return _PreparedNumericFrame(
            frame=frame,
            global_frame=global_frame,
            surfaces=tuple(surface_plans),
            fragments={},
            curve_opacities=curve_opacities,
            points=(),
            point_opacities={},
            fragment_slot_maps=boundary_batch.fragment_slot_maps,
            item_mobjects=item_mobjects,
            painter_draw_order=boundary_frame.draw_order,
            section_layers=section_layers,
            boundary_frame=boundary_frame,
            boundary_fragments=boundary_batch.fragments,
            boundary_opacities=resolved_boundary_opacities,
            surface_opacities={},
            surface_stroke_opacities={},
            projected_source_segment_counts=(
                boundary_batch.projected_source_segment_counts
            ),
        )

    def _translate_prepared_numeric(
        self,
        numeric: _PreparedNumericFrame,
        display_offset: tuple[float, float],
    ) -> _PreparedNumericFrame:
        """Apply a display-only screen translation to prepared numeric paths."""

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

        surfaces = tuple(
            replace(
                item,
                points=points(item.points),
                cone_fill=cone_fill(item.cone_fill),
            )
            for item in numeric.surfaces
        )
        fragments = {
            source_id: tuple(
                replace(
                    item,
                    points=points(item.points),
                    dashes=dashes(item.dashes),
                )
                for item in values
            )
            for source_id, values in numeric.fragments.items()
        }
        boundary_fragments = (
            None
            if numeric.boundary_fragments is None
            else {
                source_id: tuple(
                    replace(
                        item,
                        points=points(item.points),
                        dashes=dashes(item.dashes),
                    )
                    for item in values
                )
                for source_id, values in numeric.boundary_fragments.items()
            }
        )
        section_layers = numeric.section_layers
        if section_layers is not None:
            section_layers = replace(
                section_layers,
                surface_points=points(section_layers.surface_points),
                plane_polygons={
                    role: tuple(points(item) for item in values)
                    for role, values in section_layers.plane_polygons.items()
                },
                plane_outline_paths={
                    role: tuple(points(item) for item in values)
                    for role, values in section_layers.plane_outline_paths.items()
                },
                cone_fill=cone_fill(section_layers.cone_fill),
            )
        return replace(
            numeric,
            surfaces=surfaces,
            fragments=fragments,
            section_layers=section_layers,
            boundary_fragments=boundary_fragments,
        )

    def _prepare_numeric(
        self,
        performance_attempt: _PerformanceAttempt | None = None,
        resolved_inputs: _ResolvedQuadricFrameInputs | None = None,
    ) -> _PreparedNumericFrame:
        if resolved_inputs is None:
            with _performance_stage(performance_attempt, "resolve_inputs"):
                resolved_inputs = self._resolve_frame_inputs()
        surfaces = resolved_inputs.surfaces
        curves = resolved_inputs.curves
        active_curve_ids = tuple(item.curve_id for item in curves)
        curve_opacities = resolved_inputs.curve_opacities
        view = resolved_inputs.view
        compositor_style = self.style.compositor_style(
            max_projected_length=self.limits.max_projected_length
        )
        if self.boundary_visibility_mode == "unified":
            numeric = self._prepare_unified_numeric(
                surfaces,
                resolved_inputs.occluding_surfaces,
                curves,
                view,
                resolved_inputs.paint_policy,
                curve_opacities,
                resolved_inputs.boundary_opacities,
                resolved_inputs.section_plane,
                resolved_inputs.section_patch,
                resolved_inputs.surface_view_signature,
                performance_attempt,
            )
            numeric = replace(
                numeric,
                surface_opacities=dict(resolved_inputs.surface_opacities),
                surface_stroke_opacities=dict(
                    resolved_inputs.surface_stroke_opacities
                ),
                section_plane_fill_opacity=(
                    resolved_inputs.section_plane_fill_opacity
                ),
                section_plane_stroke_opacity=(
                    resolved_inputs.section_plane_stroke_opacity
                ),
            )
            translated = self._translate_prepared_numeric(
                numeric,
                resolved_inputs.display_offset,
            )
            return self._prepare_point_markers(translated, resolved_inputs)
        component_fills = self._prepare_surface_component_fills(
            surfaces,
            view,
            resolved_inputs.surface_view_signature,
            performance_attempt,
        )
        global_frame: GlobalQuadricFrame | None = None
        if self.surface_order_mode == "automatic":
            try:
                with _performance_stage(
                    performance_attempt, "surface_proxy_global_frame"
                ):
                    global_frame = compute_global_quadric_frame(
                        curves,
                        surfaces,
                        view,
                        context=self.context,
                        paint_policy=resolved_inputs.paint_policy,
                        curve_styles=(compositor_style if curves else None),
                        max_chord_error=self.max_chord_error,
                        max_segments=self.limits.max_surface_segments,
                        additional_surface_constraints=self.surface_constraints,
                    )
            except ProjectionSubdivisionError as exc:
                raise QuadricManimCapacityError(str(exc)) from exc
            except ProjectionProxyError as exc:
                raise QuadricManimError(str(exc)) from exc
            except GlobalQuadricOcclusionError as exc:
                raise QuadricManimError(
                    f"automatic global quadric ordering failed: {exc}"
                ) from exc
            frame = global_frame.frame
        else:
            try:
                with _performance_stage(
                    performance_attempt, "surface_proxy_global_frame"
                ):
                    proxies = tuple(
                        build_opaque_projection_proxy(
                            surface,
                            view,
                            patch_id=f"{surface.surface_id}:opaque-projection",
                            max_chord_error=self.max_chord_error,
                            max_segments=self.limits.max_surface_segments,
                        )
                        for surface in surfaces
                    )
            except ProjectionSubdivisionError as exc:
                raise QuadricManimCapacityError(str(exc)) from exc
            except ProjectionProxyError as exc:
                raise QuadricManimError(str(exc)) from exc
            with _performance_stage(performance_attempt, "boundary_visibility"):
                visibility = compute_quadric_visibility(
                    curves,
                    surfaces,
                    view,
                    context=self.context,
                )
            active_intervals = None
            if resolved_inputs.paint_policy is QuadricPaintPolicy.PHYSICAL:
                active_intervals = {
                    record.curve_id: tuple(
                        span.interval
                        for span in record.spans
                        if span.kind is VisibilityKind.VISIBLE
                    )
                    for record in visibility.records
                }
            try:
                with _performance_stage(performance_attempt, "curve_crossings"):
                    crossings = compute_projected_curve_crossings(
                        curves,
                        view,
                        active_intervals=active_intervals,
                    )
            except ProjectedCurveIntersectionError as exc:
                raise QuadricManimError(
                    f"projected curve ordering cannot be certified: {exc}"
                ) from exc
            try:
                with _performance_stage(
                    performance_attempt, "boundary_painter_graph"
                ):
                    frame = compute_quadric_compositing(
                        visibility,
                        proxies,
                        paint_policy=resolved_inputs.paint_policy,
                        curve_styles=(compositor_style if curves else None),
                        surface_constraints=self.surface_constraints,
                        curve_crossings=crossings,
                    )
            except QuadricCompositingError as exc:
                raise QuadricManimError(
                    f"explicit quadric painter graph failed: {exc}"
                ) from exc

        surface_plans: list[_PreparedSurface] = []
        item_mobjects: dict[str, Mobject] = {}
        section_layers: _PreparedSectionLayers | None = None
        painter_draw_order = frame.draw_order
        if self._section_enabled:
            surface = surfaces[0]
            plane = resolved_inputs.section_plane
            patch = resolved_inputs.section_patch
            if plane is None or patch is None:
                raise QuadricManimError(
                    "section inputs were not resolved before preparation"
                )
            try:
                with _performance_stage(
                    performance_attempt, "section_compositing"
                ):
                    section_frame = compute_quadric_section_compositing(
                        frame,
                        surface,
                        plane,
                        patch,
                        view,
                        context=self.context,
                        max_screen_error=self.section_max_screen_error,
                        limits=self.section_compositing_limits,
                    )
            except QuadricSectionCompositingError as exc:
                raise QuadricManimError(
                    f"quadric section compositing failed: {exc}"
                ) from exc
            surface_points = np.asarray(
                [
                    (x, y, 0.0)
                    for x, y in section_frame.surface_proxy.boundary_points
                ],
                dtype=float,
            )
            try:
                with _performance_stage(performance_attempt, "contour_union"):
                    plane_contours = quadric_plane_fragment_contours(section_frame)
            except QuadricSectionCompositingError as exc:
                raise QuadricManimError(
                    f"quadric section contour compositing failed: {exc}"
                ) from exc
            plane_polygons = {
                role: tuple(
                    np.asarray([(x, y, 0.0) for x, y in contour], dtype=float)
                    for contour in plane_contours[role]
                )
                for role in PlaneDepthRole
            }
            plane_outline_paths = {
                role: tuple(
                    np.asarray(
                        (
                            (*fragment.screen_start, 0.0),
                            (*fragment.screen_end, 0.0),
                        ),
                        dtype=float,
                    )
                    for fragment in section_frame.outline_fragments_by_role[role]
                )
                for role in PlaneDepthRole
            }
            section_layers = _PreparedSectionLayers(
                section_frame,
                surface_points,
                plane_polygons,
                plane_outline_paths,
                component_fills[surface.surface_id],
            )
            if len(self._section_slots) != len(section_frame.paint_items.ordered):
                raise QuadricManimCapacityError(
                    "section painter slot count changed after allocation"
                )
            item_mobjects.update(
                {
                    item_id: self._section_slots[index]
                    for index, item_id in enumerate(
                        section_frame.paint_items.ordered
                    )
                }
            )
            painter_draw_order = section_frame.draw_order
        else:
            for item in frame.surface_items:
                slot_index = self._surface_slot_by_id[item.surface_id]
                points = np.asarray(
                    [(x, y, 0.0) for x, y in item.proxy.boundary_points],
                    dtype=float,
                )
                surface_plans.append(
                    _PreparedSurface(
                        item.item_id,
                        item.surface_id,
                        slot_index,
                        points,
                        component_fills[item.surface_id],
                    )
                )
                item_mobjects[item.item_id] = self._surface_slots[slot_index]

        curve_map = {curve.curve_id: curve for curve in curves}
        by_curve: dict[str, list[QuadricCurvePaintFragment]] = {
            curve_id: [] for curve_id in active_curve_ids
        }
        for fragment in frame.curve_fragments:
            if fragment.painted:
                by_curve[fragment.curve_id].append(fragment)

        prepared_by_curve: dict[str, tuple[_PreparedCurveFragment, ...]] = {}
        projected_source_segment_counts: dict[str, int] = {}
        next_maps: dict[str, Mapping[str, int]] = {
            curve_id: {} for curve_id in self._slot_source_ids
        }
        for curve_id in active_curve_ids:
            fragments = tuple(sorted(by_curve[curve_id], key=lambda item: item.item_id))
            assignment = self._assign_fragment_slots(
                curve_id, tuple(item.item_id for item in fragments)
            )
            next_maps[curve_id] = assignment
            curve = curve_map[curve_id]
            values: list[_PreparedCurveFragment] = []
            if not fragments:
                prepared_by_curve[curve_id] = ()
                continue
            required_parameters = tuple(
                value
                for fragment in fragments
                for value in (fragment.interval.start, fragment.interval.end)
            )
            with _performance_stage(performance_attempt, "adaptive_projection"):
                parameters, source_points = _adaptive_project_curve_samples(
                    curve,
                    view,
                    required_parameters=required_parameters,
                    max_chord_error=self.max_chord_error,
                    max_segments=self.limits.max_segments_per_fragment,
                )
            projected_source_segment_counts[curve_id] = max(
                0,
                len(source_points) - 1,
            )
            if performance_attempt is not None:
                performance_attempt.increment_count(
                    "projected_curve_source_count"
                )
            for fragment in fragments:
                with _performance_stage(
                    performance_attempt, "projection_slicing"
                ):
                    points = _slice_projected_curve_samples(
                        parameters,
                        source_points,
                        fragment.interval.start,
                        fragment.interval.end,
                        curve_id=curve.curve_id,
                    )
                _cumulative, length = _polyline_lengths(points)
                allowance = max(1.0e-12, self.limits.max_projected_length * 1.0e-9)
                if length > self.limits.max_projected_length + allowance:
                    raise QuadricManimCapacityError(
                        f"curve {curve_id!r} projected fragment length {length:.9g} "
                        "exceeds max_projected_length "
                        f"{self.limits.max_projected_length:.9g}"
                    )
                with _performance_stage(performance_attempt, "dash_generation"):
                    dashes = (
                        _dash_polyline(
                            points,
                            dash_length=self.style.dash_length,
                            dash_gap=self.style.dash_gap,
                            capacity=self.limits.max_dashes_per_fragment,
                        )
                        if fragment.render_intent == "dashed"
                        else ()
                    )
                if performance_attempt is not None:
                    performance_attempt.increment_count(
                        "prepared_curve_fragment_count"
                    )
                    performance_attempt.increment_count(
                        "projected_fragment_slice_count"
                    )
                    performance_attempt.increment_count(
                        "prepared_dash_count", len(dashes)
                    )
                slot_index = assignment[fragment.item_id]
                prepared = _PreparedCurveFragment(
                    fragment,
                    slot_index,
                    points,
                    dashes,
                )
                values.append(prepared)
                slot = self._curve_slots[curve_id].fragments[slot_index]
                item_mobjects[fragment.item_id] = (
                    slot.solid if fragment.render_intent == "solid" else slot.dashed
                )
            prepared_by_curve[curve_id] = tuple(values)

        if set(item_mobjects) != set(painter_draw_order):
            raise QuadricManimError(
                "prepared Manim items do not cover compositor draw_order"
            )
        if performance_attempt is not None:
            performance_attempt.set_count("surface_count", len(surfaces))
            performance_attempt.set_count("curve_count", len(curves))
            performance_attempt.set_count(
                "curve_fragment_count", len(frame.curve_fragments)
            )
            performance_attempt.set_count(
                "plane_fragment_count",
                0
                if section_layers is None
                else len(section_layers.frame.plane_fragments),
            )
            performance_attempt.set_count(
                "ray_classification_count",
                0
                if section_layers is None
                else section_layers.frame.ray_classification_count,
            )
        numeric = _PreparedNumericFrame(
            frame=frame,
            global_frame=global_frame,
            surfaces=tuple(surface_plans),
            fragments=prepared_by_curve,
            curve_opacities=curve_opacities,
            points=(),
            point_opacities={},
            fragment_slot_maps=next_maps,
            item_mobjects=item_mobjects,
            painter_draw_order=tuple(painter_draw_order),
            section_layers=section_layers,
            surface_opacities=dict(resolved_inputs.surface_opacities),
            surface_stroke_opacities=dict(
                resolved_inputs.surface_stroke_opacities
            ),
            section_plane_fill_opacity=(
                resolved_inputs.section_plane_fill_opacity
            ),
            section_plane_stroke_opacity=(
                resolved_inputs.section_plane_stroke_opacity
            ),
            projected_source_segment_counts=projected_source_segment_counts,
        )
        translated = self._translate_prepared_numeric(
            numeric,
            resolved_inputs.display_offset,
        )
        return self._prepare_point_markers(translated, resolved_inputs)

    def _prepare_point_markers(
        self,
        numeric: _PreparedNumericFrame,
        resolved_inputs: _ResolvedQuadricFrameInputs,
    ) -> _PreparedNumericFrame:
        """Add exact point items to the already certified painter order.

        The marker remains a true fixed ``Dot`` slot.  A hidden point is placed
        immediately before the nearest owning surface sheet; a visible point is
        a diagrammatic section marker painted after the certified parent graph.
        No zero-length curve or artificial parameter interval is introduced.
        """

        if not resolved_inputs.points:
            return replace(
                numeric,
                points=(),
                point_opacities=resolved_inputs.point_opacities,
            )
        order = list(numeric.painter_draw_order)
        item_mobjects = dict(numeric.item_mobjects)
        surface_item_by_id = {
            item.surface_id: item.item_id for item in numeric.surfaces
        }
        section_front: str | None = None
        if numeric.section_layers is not None:
            section_front = numeric.section_layers.frame.paint_items.surface_front

        prepared_points: list[_PreparedPoint] = []
        for marker in resolved_inputs.points:
            visibility = compute_point_visibility(
                marker,
                resolved_inputs.occluding_surfaces,
                resolved_inputs.view,
                context=self.context,
            )
            screen = (
                resolved_inputs.view.matrix[:2]
                @ np.asarray(marker.point, dtype=float)
                + np.asarray(resolved_inputs.display_offset, dtype=float)
            )
            item_id = f"point:{marker.point_id}"
            prepared = _PreparedPoint(
                item_id,
                marker.point_id,
                np.asarray((screen[0], screen[1], 0.0), dtype=float),
                visibility.visible,
                visibility.occluders,
            )
            prepared_points.append(prepared)
            item_mobjects[item_id] = self._point_slots[marker.point_id]

            if visibility.visible:
                order.append(item_id)
                continue
            occluder_items = [
                surface_item_by_id[surface_id]
                for surface_id in visibility.occluders
                if surface_id in surface_item_by_id
            ]
            if section_front is not None and visibility.occluders:
                occluder_items.append(section_front)
            positions = [order.index(item) for item in occluder_items if item in order]
            if not positions:
                raise QuadricManimError(
                    f"hidden point {marker.point_id!r} has no painter surface item"
                )
            order.insert(min(positions), item_id)

        if set(item_mobjects) != set(order):
            raise QuadricManimError(
                "prepared point items do not cover the augmented painter order"
            )
        return replace(
            numeric,
            points=tuple(prepared_points),
            point_opacities=resolved_inputs.point_opacities,
            item_mobjects=item_mobjects,
            painter_draw_order=tuple(order),
        )

    def _prepare_painter(
        self,
        numeric: _PreparedNumericFrame,
        performance_attempt: _PerformanceAttempt | None = None,
    ) -> PreparedQuadricManimFrame:
        try:
            with _performance_stage(
                performance_attempt, "painter_band_preparation"
            ):
                self._band.configure(
                    containers=self._scene_containers(),
                    sources={"quadric:reservation": self._update_driver},
                )
                painter = self._band.prepare(
                    draw_order=numeric.painter_draw_order,
                    item_mobjects=numeric.item_mobjects,
                )
        except ManagedPainterBandError as exc:
            raise QuadricManimError(str(exc)) from exc
        return PreparedQuadricManimFrame(
            numeric,
            painter,
            performance_attempt,
        )

    def _new_performance_attempt(self) -> _PerformanceAttempt | None:
        if not self._performance_enabled:
            return None
        attempt = _PerformanceAttempt(
            "quadric_occlusion_3d",
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

    def _prepare_with_performance(self) -> PreparedQuadricManimFrame:
        attempt = self._new_performance_attempt()
        try:
            numeric = self._prepare_numeric(attempt)
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
                    sources={"quadric:reservation": self._update_driver},
                )
        except ManagedPainterBandError as exc:
            raise QuadricManimError(str(exc)) from exc

    def _reuse_prepared_draw_inputs(
        self,
        resolved: _ResolvedQuadricFrameInputs,
        attempt: _PerformanceAttempt | None,
    ) -> PreparedQuadricManimFrame:
        cached = self._last_prepared_frame
        if cached is None:
            raise QuadricManimError("no committed frame is available for reuse")
        numeric = cached.numeric
        boundary_opacities = numeric.boundary_opacities
        if boundary_opacities is not None:
            boundary_opacities = {
                source_id: (
                    resolved.curve_opacities[source_id]
                    if source_id in resolved.curve_opacities
                    else resolved.boundary_opacities.get(source_id, 1.0)
                )
                for source_id in boundary_opacities
            }
        reused_numeric = replace(
            numeric,
            curve_opacities=dict(resolved.curve_opacities),
            point_opacities=dict(resolved.point_opacities),
            boundary_opacities=boundary_opacities,
            surface_opacities=dict(resolved.surface_opacities),
            surface_stroke_opacities=dict(
                resolved.surface_stroke_opacities
            ),
            section_plane_fill_opacity=(
                resolved.section_plane_fill_opacity
            ),
            section_plane_stroke_opacity=(
                resolved.section_plane_stroke_opacity
            ),
        )
        return PreparedQuadricManimFrame(
            reused_numeric,
            cached.painter_band,
            attempt,
        )

    def _commit_input_cache(
        self,
        resolved: _ResolvedQuadricFrameInputs,
        opacity: float,
        prepared: PreparedQuadricManimFrame,
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

    def prepare(self) -> PreparedQuadricManimFrame:
        """Prepare and validate one frame without changing any display slot."""

        if not self._attached:
            raise QuadricManimError("quadric occlusion controller is not attached")
        return self._prepare_with_performance()

    def _apply_surface(
        self,
        prepared: _PreparedSurface,
        fill_opacity: float,
        stroke_opacity: float,
        *,
        draw_stroke: bool = True,
    ) -> None:
        slot = self._surface_paint_slots[prepared.slot_index]
        _apply_opaque_surface_slot(
            slot,
            prepared.points,
            prepared.cone_fill,
            self.style,
            fill_opacity,
            draw_stroke=draw_stroke,
            stroke_opacity=stroke_opacity,
        )

    def _apply_section_layers(
        self,
        prepared: _PreparedSectionLayers,
        surface_fill_opacity: float,
        surface_stroke_opacity: float,
        plane_fill_opacity: float,
        plane_stroke_opacity: float,
        *,
        draw_legacy_strokes: bool = True,
        draw_plane_outline: bool | None = None,
    ) -> None:
        frame = prepared.frame
        if len(self._section_slots) != len(frame.paint_items.ordered):
            raise QuadricManimCapacityError(
                "section painter slots were not allocated"
            )
        slots = dict(zip(frame.paint_items.ordered, self._section_slots))
        surface_back = self._section_surface_paint_slots[1]
        surface_front = self._section_surface_paint_slots[4]
        if slots[frame.paint_items.surface_back] is not surface_back.root:
            raise QuadricManimCapacityError("section back-sheet slot changed identity")
        if slots[frame.paint_items.surface_front] is not surface_front.root:
            raise QuadricManimCapacityError("section front-sheet slot changed identity")

        _apply_surface_sheet_pair(
            surface_back,
            surface_front,
            prepared.surface_points,
            prepared.cone_fill,
            self.style,
            surface_fill_opacity,
            configure_front_stroke=True,
            draw_front_stroke=draw_legacy_strokes,
            stroke_opacity=surface_stroke_opacity,
        )

        if draw_plane_outline is None:
            draw_plane_outline = draw_legacy_strokes

        fill_item_by_role = {
            PlaneDepthRole.BEHIND_SURFACE: frame.paint_items.plane_behind,
            PlaneDepthRole.OUTSIDE_PROJECTION: frame.paint_items.plane_outside,
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS: frame.paint_items.plane_between,
            PlaneDepthRole.IN_FRONT_OF_SURFACE: frame.paint_items.plane_front,
        }
        for role, item_id in fill_item_by_role.items():
            slot = slots[item_id]
            _set_closed_subpaths(slot, prepared.plane_polygons[role])
            slot.set_fill(
                color=self.style.section_plane_fill_color,
                opacity=(
                    self.style.section_plane_fill_opacity
                    * plane_fill_opacity
                ),
            )
            slot.set_stroke(opacity=0.0)

        for role, item_id in frame.paint_items.outline_by_role.items():
            slot = slots[item_id]
            _set_open_subpaths(
                slot,
                prepared.plane_outline_paths[role]
                if draw_plane_outline
                else (),
            )
            slot.set_fill(opacity=0.0)
            slot.set_stroke(
                color=self.style.section_plane_stroke_color,
                width=self.style.section_plane_stroke_width,
                opacity=(
                    self.style.section_plane_stroke_opacity
                    * plane_stroke_opacity
                    if draw_plane_outline
                    else 0.0
                ),
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

    def _apply_curve_fragment(
        self,
        curve_id: str,
        prepared: _PreparedCurveFragment,
        opacity: float,
    ) -> None:
        slot = self._curve_slots[curve_id].fragments[prepared.slot_index]
        fragment = prepared.fragment
        if fragment.render_intent == "solid":
            slot.solid.set_points_as_corners(prepared.points)
            slot.solid.set_fill(opacity=0.0)
            slot.solid.set_stroke(
                color=self.style.visible_curve_color,
                width=self.style.visible_curve_width,
                opacity=self.style.visible_curve_opacity * opacity,
            )
            slot.solid.set_stroke(
                color=self.style.background_color,
                width=self.style.background_width,
                opacity=self.style.background_opacity * opacity,
                background=True,
            )
            if self.style.cap_style is not None:
                slot.solid.set_cap_style(self.style.cap_style)
            if self.style.joint_type is not None:
                slot.solid.joint_type = self.style.joint_type
            _hide_vmobject(slot.dashed)
            return

        _hide_vmobject(slot.solid)
        _set_open_subpaths(
            slot.dashed,
            tuple(item.points for item in prepared.dashes),
        )
        slot.dashed.set_fill(opacity=0.0)
        slot.dashed.set_stroke(
            color=self.style.hidden_curve_color,
            width=self.style.hidden_curve_width,
            opacity=self.style.hidden_curve_opacity * opacity,
        )
        slot.dashed.set_stroke(
            color=self.style.background_color,
            width=self.style.background_width,
            opacity=self.style.background_opacity * opacity,
            background=True,
        )
        cap = (
            self.style.cap_style
            if self.style.hidden_cap_style is None
            else self.style.hidden_cap_style
        )
        joint = (
            self.style.joint_type
            if self.style.hidden_joint_type is None
            else self.style.hidden_joint_type
        )
        if cap is not None:
            slot.dashed.set_cap_style(cap)
        if joint is not None:
            slot.dashed.joint_type = joint

    def _apply_point_marker(
        self,
        prepared: _PreparedPoint,
        opacity: float,
    ) -> None:
        slot = self._point_slots[prepared.point_id]
        slot.move_to(prepared.screen_point)
        color = (
            self.style.point_color
            if prepared.visible
            else self.style.hidden_point_color
        )
        visibility_opacity = (
            self.style.point_opacity
            if prepared.visible
            else self.style.hidden_point_opacity
        )
        slot.set_fill(color=color, opacity=visibility_opacity * opacity)
        slot.set_stroke(opacity=0.0)

    def _prepare_display_actions(
        self,
        prepared: PreparedQuadricManimFrame,
        opacity: float,
    ) -> tuple[_PreparedDisplayAction, ...]:
        """Describe only the fixed slots which should be active this frame."""

        actions: list[_PreparedDisplayAction] = []
        unified = prepared.numeric.boundary_frame is not None
        draw_legacy_surface_stroke = (
            not unified or self.legacy_surface_stroke_fallback
        )
        for surface in prepared.numeric.surfaces:
            slot = self._surface_paint_slots[surface.slot_index]
            surface_fill_opacity = (
                opacity
                * prepared.numeric.surface_opacities.get(
                    surface.surface_id,
                    1.0,
                )
            )
            surface_stroke_opacity = (
                opacity
                * prepared.numeric.surface_stroke_opacities.get(
                    surface.surface_id,
                    1.0,
                )
            )
            actions.append(
                _PreparedDisplayAction(
                    f"surface:{surface.slot_index}",
                    (slot.root,),
                    _display_digest(
                        "surface",
                        surface.points,
                        surface.cone_fill,
                        self.style,
                        surface_fill_opacity,
                        surface_stroke_opacity,
                        draw_legacy_surface_stroke,
                    ),
                    partial(
                        self._apply_surface,
                        surface,
                        surface_fill_opacity,
                        surface_stroke_opacity,
                        draw_stroke=draw_legacy_surface_stroke,
                    ),
                )
            )

        section = prepared.numeric.section_layers
        if section is not None:
            surface_fill_opacity = (
                opacity
                * prepared.numeric.surface_opacities.get(
                    section.frame.surface_id,
                    1.0,
                )
            )
            surface_stroke_opacity = (
                opacity
                * prepared.numeric.surface_stroke_opacities.get(
                    section.frame.surface_id,
                    1.0,
                )
            )
            plane_fill_opacity = (
                opacity * prepared.numeric.section_plane_fill_opacity
            )
            plane_stroke_opacity = (
                opacity * prepared.numeric.section_plane_stroke_opacity
            )
            draw_plane_outline = (
                not unified
                or section.frame.projection_kind is PlanePatchProjectionKind.LINE
            )
            actions.append(
                _PreparedDisplayAction(
                    "section:layers",
                    tuple(self._section_slots),
                    _display_digest(
                        "section",
                        section.surface_points,
                        section.plane_polygons,
                        section.plane_outline_paths,
                        section.cone_fill,
                        section.frame.paint_items.ordered,
                        self.style,
                        surface_fill_opacity,
                        surface_stroke_opacity,
                        plane_fill_opacity,
                        plane_stroke_opacity,
                        draw_legacy_surface_stroke,
                        draw_plane_outline,
                    ),
                    partial(
                        self._apply_section_layers,
                        section,
                        surface_fill_opacity,
                        surface_stroke_opacity,
                        plane_fill_opacity,
                        plane_stroke_opacity,
                        draw_legacy_strokes=draw_legacy_surface_stroke,
                        draw_plane_outline=draw_plane_outline,
                    ),
                )
            )

        for curve_id, fragments in prepared.numeric.fragments.items():
            curve_opacity = opacity * prepared.numeric.curve_opacities[curve_id]
            for fragment in fragments:
                slot = self._curve_slots[curve_id].fragments[fragment.slot_index]
                actions.append(
                    _PreparedDisplayAction(
                        f"path:{curve_id}:{fragment.slot_index}",
                        (slot.root,),
                        _display_digest(
                            "curve",
                            fragment.fragment.render_intent,
                            fragment.points,
                            tuple(item.points for item in fragment.dashes),
                            self.style,
                            curve_opacity,
                        ),
                        partial(
                            self._apply_curve_fragment,
                            curve_id,
                            fragment,
                            curve_opacity,
                        ),
                    )
                )

        for point in prepared.numeric.points:
            point_opacity = opacity * prepared.numeric.point_opacities[point.point_id]
            slot = self._point_slots[point.point_id]
            actions.append(
                _PreparedDisplayAction(
                    f"point:{point.point_id}",
                    (slot,),
                    _display_digest(
                        "point",
                        point.screen_point,
                        point.visible,
                        point.occluders,
                        self.style.point_color,
                        self.style.point_radius,
                        self.style.point_opacity,
                        self.style.hidden_point_color,
                        self.style.hidden_point_opacity,
                        point_opacity,
                    ),
                    partial(self._apply_point_marker, point, point_opacity),
                )
            )

        boundary_opacities = prepared.numeric.boundary_opacities or {}
        for source_id, fragments in (
            prepared.numeric.boundary_fragments or {}
        ).items():
            source_opacity = opacity * boundary_opacities.get(source_id, 1.0)
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
                            source_opacity,
                        ),
                        partial(
                            self._apply_boundary_fragment,
                            source_id,
                            fragment,
                            source_opacity,
                        ),
                    )
                )
        return tuple(actions)

    def apply(
        self,
        prepared: PreparedQuadricManimFrame,
        *,
        _commit_frame: Callable[[], None] | None = None,
        _finalize_frame: Callable[[], None] | None = None,
    ) -> None:
        """Commit one validated frame and its optional author continuation."""

        if not self._attached:
            raise QuadricManimError("quadric occlusion controller is not attached")
        if not isinstance(prepared, PreparedQuadricManimFrame):
            raise TypeError("prepared must be a PreparedQuadricManimFrame")

        def capture_controller_state() -> _CommittedControllerState:
            return _CommittedControllerState(
                fragment_slot_maps={
                    curve_id: dict(values)
                    for curve_id, values in self._fragment_slot_maps.items()
                },
                last_frame=self._last_frame,
                last_global_frame=self._last_global_frame,
                last_section_frame=self._last_section_frame,
                last_boundary_frame=self._last_boundary_frame,
                display_slot_state=dict(self._display_slot_state),
                last_painter_band_signature=self._last_painter_band_signature,
                last_input_geometry_signature=(
                    self._last_input_geometry_signature
                ),
                last_input_draw_signature=self._last_input_draw_signature,
                last_input_opacity=self._last_input_opacity,
                last_prepared_frame=self._last_prepared_frame,
                last_prepared_performance_counts=dict(
                    self._last_prepared_performance_counts
                ),
            )

        def restore_controller_state(
            state: _CommittedControllerState,
        ) -> None:
            self._fragment_slot_maps = state.fragment_slot_maps
            self._last_frame = state.last_frame
            self._last_global_frame = state.last_global_frame
            self._last_section_frame = state.last_section_frame
            self._last_boundary_frame = state.last_boundary_frame
            self._display_slot_state = state.display_slot_state
            self._last_painter_band_signature = (
                state.last_painter_band_signature
            )
            self._last_input_geometry_signature = (
                state.last_input_geometry_signature
            )
            self._last_input_draw_signature = state.last_input_draw_signature
            self._last_input_opacity = state.last_input_opacity
            self._last_prepared_frame = state.last_prepared_frame
            self._last_prepared_performance_counts = (
                state.last_prepared_performance_counts
            )

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
                        curve_id: dict(values)
                        for curve_id, values in (
                            prepared.numeric.fragment_slot_maps.items()
                        )
                    }
                    self._last_frame = prepared.frame
                    self._last_global_frame = prepared.global_frame
                    self._last_section_frame = prepared.section_frame
                    self._last_boundary_frame = prepared.boundary_frame
                    self._display_slot_state = dict(delta.next_state)
                    self._last_painter_band_signature = painter_signature
                if _commit_frame is not None:
                    _commit_frame()
                if _finalize_frame is not None:
                    _finalize_frame()
        except Exception as exc:
            self._finish_performance_attempt(
                attempt,
                status="failed",
                rollback_performed=True,
                error=exc,
            )
            raise
        self._finish_performance_attempt(attempt, status="committed")

    def snapshot_transaction_state(
        self,
    ) -> QuadricOcclusionTransactionSnapshot:
        """Capture the complete committed state of one attached Cairo frame.

        The returned value is intentionally opaque and controller-bound.  It
        is suitable for a higher-level coordinator that must atomically roll
        back several already-attached quadric controllers after one of them
        fails to commit.
        """

        if not self._attached:
            raise QuadricManimError(
                "quadric transaction state requires an attached controller"
            )
        return QuadricOcclusionTransactionSnapshot(
            _owner_token=self._transaction_snapshot_owner_token,
            _root_state=_capture_transaction_root(self.root),
            _painter_band_state=self._band.capture_active_state(),
            _fragment_slot_maps={
                source_id: dict(values)
                for source_id, values in self._fragment_slot_maps.items()
            },
            _last_frame=self._last_frame,
            _last_global_frame=self._last_global_frame,
            _last_section_frame=self._last_section_frame,
            _last_boundary_frame=self._last_boundary_frame,
            _display_slot_state=dict(self._display_slot_state),
            _last_painter_band_signature=self._last_painter_band_signature,
            _last_input_geometry_signature=self._last_input_geometry_signature,
            _last_input_draw_signature=self._last_input_draw_signature,
            _last_input_opacity=self._last_input_opacity,
            _last_prepared_frame=self._last_prepared_frame,
            _last_prepared_performance_counts=dict(
                self._last_prepared_performance_counts
            ),
            _performance_frame_index=self._performance_frame_index,
            _last_performance_snapshot=self._last_performance_snapshot,
        )

    def _validate_transaction_snapshot(
        self,
        snapshot: QuadricOcclusionTransactionSnapshot,
    ) -> None:
        if snapshot._owner_token is not self._transaction_snapshot_owner_token:
            raise QuadricManimError(
                "quadric transaction snapshot belongs to another controller"
            )

        family = tuple(self.root.get_family())
        family_ids = {id(member) for member in family}
        if len(snapshot._root_state) != len(family) or any(
            state.mobject is not member
            for state, member in zip(snapshot._root_state, family)
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has incompatible display slots"
            )
        allowed_attributes = (
            _TRANSACTION_NUMERIC_STYLE_ATTRIBUTES
            | _TRANSACTION_ENUM_STYLE_ATTRIBUTES
        )
        for state in snapshot._root_state:
            if not isinstance(state, _MobjectState):
                raise QuadricManimError(
                    "quadric transaction snapshot has invalid display state"
                )
            if state.points is not None and (
                not isinstance(state.points, np.ndarray)
                or not np.all(np.isfinite(state.points))
            ):
                raise QuadricManimError(
                    "quadric transaction snapshot has invalid display points"
                )
            if state.z_index is not None and not np.isfinite(state.z_index):
                raise QuadricManimError(
                    "quadric transaction snapshot has invalid display z-index"
                )
            if not isinstance(state.attributes, dict) or not set(
                state.attributes
            ).issubset(allowed_attributes):
                raise QuadricManimError(
                    "quadric transaction snapshot has invalid display styles"
                )
            for name, value in state.attributes.items():
                if name in _TRANSACTION_ENUM_STYLE_ATTRIBUTES:
                    current = getattr(state.mobject, name, None)
                    if current is None or not isinstance(value, type(current)):
                        raise QuadricManimError(
                            "quadric transaction snapshot has invalid display styles"
                        )
                    continue
                try:
                    numeric = np.asarray(value, dtype=float)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise QuadricManimError(
                        "quadric transaction snapshot has invalid display styles"
                    ) from exc
                if not np.all(np.isfinite(numeric)):
                    raise QuadricManimError(
                        "quadric transaction snapshot has invalid display styles"
                    )

        band_state = snapshot._painter_band_state
        if not isinstance(band_state, Mapping) or any(
            not isinstance(item_id, str)
            or not item_id
            or not np.isfinite(float(value))
            for item_id, value in band_state.items()
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid painter-band state"
            )
        band_mobject_ids = getattr(band_state, "mobject_ids", None)
        if (
            not isinstance(band_mobject_ids, dict)
            or set(band_mobject_ids) != set(band_state)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in family_ids
                for value in band_mobject_ids.values()
            )
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid painter-band identities"
            )

        signature = snapshot._last_painter_band_signature
        if not isinstance(signature, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 3
            or not isinstance(item[0], str)
            or not item[0]
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] not in family_ids
            or not np.isfinite(float(item[2]))
            for item in signature
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid painter signature"
            )
        expected_band = {item_id: z_index for item_id, _root, z_index in signature}
        expected_band_mobject_ids = {
            item_id: root_id for item_id, root_id, _z_index in signature
        }
        if (
            dict(band_state) != expected_band
            or band_mobject_ids != expected_band_mobject_ids
        ):
            raise QuadricManimError(
                "quadric transaction snapshot painter evidence is inconsistent"
            )

        slot_maps = snapshot._fragment_slot_maps
        if not isinstance(slot_maps, dict) or set(slot_maps) != set(
            self._slot_source_ids
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid fragment-slot maps"
            )
        for source_id, mapping in slot_maps.items():
            if not isinstance(source_id, str) or not isinstance(mapping, dict):
                raise QuadricManimError(
                    "quadric transaction snapshot has invalid fragment-slot maps"
                )
            if any(
                not isinstance(fragment_id, str)
                or not fragment_id
                or isinstance(slot_index, bool)
                or not isinstance(slot_index, int)
                or not 0 <= slot_index < self.limits.max_fragments_per_curve
                for fragment_id, slot_index in mapping.items()
            ) or len(set(mapping.values())) != len(mapping):
                raise QuadricManimError(
                    "quadric transaction snapshot has invalid fragment-slot maps"
                )

        for name, value, expected_type in (
            ("last frame", snapshot._last_frame, QuadricCompositingFrame),
            ("last global frame", snapshot._last_global_frame, GlobalQuadricFrame),
            (
                "last section frame",
                snapshot._last_section_frame,
                QuadricSectionCompositingFrame,
            ),
            (
                "last boundary frame",
                snapshot._last_boundary_frame,
                QuadricBoundaryCompositingFrame,
            ),
        ):
            if value is not None and not isinstance(value, expected_type):
                raise QuadricManimError(
                    f"quadric transaction snapshot has invalid {name}"
                )

        display_state = snapshot._display_slot_state
        if not isinstance(display_state, dict):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid display-slot state"
            )
        for slot_id, state in display_state.items():
            if (
                not isinstance(slot_id, str)
                or not slot_id
                or not isinstance(state, _CommittedDisplaySlot)
                or not isinstance(state.digest, bytes)
                or any(id(root) not in family_ids for root in state.roots)
            ):
                raise QuadricManimError(
                    "quadric transaction snapshot has invalid display-slot state"
                )

        for name, value in (
            ("geometry", snapshot._last_input_geometry_signature),
            ("draw", snapshot._last_input_draw_signature),
        ):
            if value is not None and not isinstance(value, bytes):
                raise QuadricManimError(
                    f"quadric transaction snapshot has invalid {name} signature"
                )
        if snapshot._last_input_opacity is not None and (
            not np.isfinite(snapshot._last_input_opacity)
            or snapshot._last_input_opacity < 0.0
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid input opacity"
            )
        prepared = snapshot._last_prepared_frame
        if prepared is not None and not isinstance(
            prepared, PreparedQuadricManimFrame
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid prepared frame"
            )
        if prepared is not None:
            prepared_maps = {
                source_id: dict(values)
                for source_id, values in prepared.numeric.fragment_slot_maps.items()
            }
            if prepared_maps != slot_maps:
                raise QuadricManimError(
                    "quadric transaction snapshot fragment evidence is inconsistent"
                )
            if (
                prepared.frame is not snapshot._last_frame
                or prepared.global_frame is not snapshot._last_global_frame
                or prepared.section_frame is not snapshot._last_section_frame
                or prepared.boundary_frame is not snapshot._last_boundary_frame
            ):
                raise QuadricManimError(
                    "quadric transaction snapshot frame evidence is inconsistent"
                )
        counts = snapshot._last_prepared_performance_counts
        if not isinstance(counts, dict) or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in counts.items()
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid performance counts"
            )
        if (
            isinstance(snapshot._performance_frame_index, bool)
            or not isinstance(snapshot._performance_frame_index, int)
            or snapshot._performance_frame_index < 0
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid performance frame index"
            )
        if snapshot._last_performance_snapshot is not None and not isinstance(
            snapshot._last_performance_snapshot,
            QuadricPerformanceSnapshot,
        ):
            raise QuadricManimError(
                "quadric transaction snapshot has invalid performance evidence"
            )

    def _restore_transaction_state_unchecked(
        self,
        snapshot: QuadricOcclusionTransactionSnapshot,
    ) -> None:
        _restore_root(snapshot._root_state)
        self._band.restore_active_state(snapshot._painter_band_state)
        self._fragment_slot_maps = {
            source_id: dict(values)
            for source_id, values in snapshot._fragment_slot_maps.items()
        }
        self._last_frame = snapshot._last_frame
        self._last_global_frame = snapshot._last_global_frame
        self._last_section_frame = snapshot._last_section_frame
        self._last_boundary_frame = snapshot._last_boundary_frame
        self._display_slot_state = dict(snapshot._display_slot_state)
        self._last_painter_band_signature = (
            snapshot._last_painter_band_signature
        )
        self._last_input_geometry_signature = (
            snapshot._last_input_geometry_signature
        )
        self._last_input_draw_signature = snapshot._last_input_draw_signature
        self._last_input_opacity = snapshot._last_input_opacity
        self._last_prepared_frame = snapshot._last_prepared_frame
        self._last_prepared_performance_counts = dict(
            snapshot._last_prepared_performance_counts
        )
        self._performance_frame_index = snapshot._performance_frame_index
        self._last_performance_snapshot = snapshot._last_performance_snapshot

    def restore_transaction_state(
        self,
        snapshot: QuadricOcclusionTransactionSnapshot,
    ) -> "QuadricOcclusion3D":
        """Restore one attached frame without changing Scene ownership.

        Validation completes before any Mobject is touched.  If an unexpected
        restore error still occurs, the current frame is restored before the
        error is reported, so invalid input cannot leave a half-applied frame.
        """

        if not self._attached:
            raise QuadricManimError(
                "quadric transaction restore requires an attached controller"
            )
        if not isinstance(snapshot, QuadricOcclusionTransactionSnapshot):
            raise TypeError(
                "snapshot must be a QuadricOcclusionTransactionSnapshot"
            )
        self._validate_transaction_snapshot(snapshot)
        current = self.snapshot_transaction_state()
        try:
            self._restore_transaction_state_unchecked(snapshot)
            self._invalidate_cairo_static_image()
        except Exception as exc:
            try:
                self._restore_transaction_state_unchecked(current)
                self._invalidate_cairo_static_image()
            except Exception as rollback_exc:
                raise QuadricManimError(
                    "quadric transaction restore and rollback both failed"
                ) from rollback_exc
            raise QuadricManimError(
                "quadric transaction snapshot could not be restored"
            ) from exc
        return self

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
    def last_frame(self) -> QuadricCompositingFrame | None:
        return self._last_frame

    @property
    def last_global_frame(self) -> GlobalQuadricFrame | None:
        """Return the last committed automatic global frame and its evidence."""

        return self._last_global_frame

    @property
    def last_section_frame(self) -> QuadricSectionCompositingFrame | None:
        """Return the last committed plane/surface split and painter trace."""

        return self._last_section_frame

    @property
    def last_boundary_frame(self) -> QuadricBoundaryCompositingFrame | None:
        """Return the last committed unified semantic-boundary frame."""

        return self._last_boundary_frame

    @property
    def active_painter_z_indices(self) -> dict[str, float]:
        return self._band.active_z_indices

    @property
    def allocated_curve_ids(self) -> tuple[str, ...]:
        """Return the immutable authored curve identities."""

        return self._curve_ids

    @property
    def allocated_point_ids(self) -> tuple[str, ...]:
        """Return the immutable authored point-marker identities."""

        return self._point_ids

    @property
    def allocated_boundary_ids(self) -> tuple[str, ...]:
        """Return every fixed semantic-boundary source identity."""

        return self._boundary_source_ids

    def attach(self) -> "QuadricOcclusion3D":
        if self._attached:
            return self
        if config.renderer != RendererType.CAIRO:
            raise QuadricManimError(
                "quadric occlusion binding supports the Cairo renderer only"
            )
        family_ids = {id(item) for item in self.root.get_family()}
        family_ids.update(id(item) for item in self._update_driver.get_family())
        if any(
            id(item) in family_ids
            for container in self._scene_containers()
            for item in container
        ):
            raise QuadricManimError("quadric display slots are already Scene-owned")

        # Numerical, geometric, and capacity failures happen before ownership
        # changes.  Painter-band validation temporarily needs the persistent
        # non-rendering driver in the Scene and is fully rolled back on error.
        attempt = self._new_performance_attempt()
        frame_token = self._begin_bound_frame_transaction()
        try:
            with _performance_stage(attempt, "resolve_inputs"):
                resolved = self._resolve_frame_inputs()
            numeric = self._prepare_numeric(attempt, resolved)
        except Exception as exc:
            self._rollback_bound_frame_transaction(frame_token)
            self._finish_performance_attempt(
                attempt,
                status="failed",
                error=exc,
            )
            raise
        root_state = _capture_root(self.root)
        previous_band_state = self._band.capture_active_state()
        previous_maps = {
            curve_id: dict(values)
            for curve_id, values in self._fragment_slot_maps.items()
        }
        previous_frame = self._last_frame
        previous_global_frame = self._last_global_frame
        previous_section_frame = self._last_section_frame
        previous_boundary_frame = self._last_boundary_frame
        self.root.reset_opacity()
        try:
            # Cairo caches every mobject before the first time-aware updater as
            # one static background image.  Keep the invisible driver before
            # the display root so the root is always part of the moving suffix
            # while unrelated objects already present in the Scene stay in the
            # reusable static background.
            self.scene.mobjects.append(self._update_driver)
            self.scene.mobjects.append(self.root)
            self._register_fixed_frame()
            prepared = self._prepare_painter(numeric, attempt)
            self._attached = True

            def commit_attached_frame() -> None:
                self._commit_input_cache(
                    resolved,
                    self.root.opacity_multiplier,
                    prepared,
                )
                self._commit_bound_frame_transaction(frame_token)

            self.apply(
                prepared,
                _commit_frame=commit_attached_frame,
                _finalize_frame=lambda: self._finalize_bound_frame_transaction(
                    frame_token
                ),
            )
        except Exception as exc:
            self._attached = False
            _restore_root(root_state)
            self._band.restore_active_state(previous_band_state)
            self._fragment_slot_maps = previous_maps
            self._last_frame = previous_frame
            self._last_global_frame = previous_global_frame
            self._last_section_frame = previous_section_frame
            self._last_boundary_frame = previous_boundary_frame
            self._remove_fixed_frame()
            self._remove_owned_identities()
            self._band.restore()
            self._invalidate_cairo_static_image()
            self._rollback_bound_frame_transaction(frame_token)
            self._finish_performance_attempt(
                attempt,
                status="failed",
                rollback_performed=True,
                error=exc,
            )
            raise
        return self

    def update(self, dt: float = 0.0) -> "QuadricOcclusion3D":
        del dt
        if not self._attached:
            raise QuadricManimError("quadric occlusion controller is not attached")
        attempt = self._new_performance_attempt()
        frame_token = self._begin_bound_frame_transaction()
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
                if dirty_kind is _DirtyFrameKind.DRAW_ONLY:
                    prepared = self._reuse_prepared_draw_inputs(
                        resolved,
                        attempt,
                    )
                else:
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
                    self._commit_bound_frame_transaction(frame_token)
                    self._finalize_bound_frame_transaction(frame_token)
                    self._finish_performance_attempt(
                        attempt,
                        status="committed",
                    )
                    return self
            def commit_updated_frame() -> None:
                self._commit_input_cache(
                    resolved,
                    opacity,
                    prepared,
                )
                self._commit_bound_frame_transaction(frame_token)

            self.apply(
                prepared,
                _commit_frame=commit_updated_frame,
                _finalize_frame=lambda: self._finalize_bound_frame_transaction(
                    frame_token
                ),
            )
        except Exception as exc:
            self._rollback_bound_frame_transaction(frame_token)
            self._finish_performance_attempt(
                attempt,
                status="failed",
                error=exc,
            )
            raise
        return self

    def _scene_containers(self) -> tuple[list[object], ...]:
        return _scene_containers(self.scene)

    def _register_fixed_frame(self) -> None:
        self._fixed_frame_camera = _register_fixed_frame(self.scene, self.root)

    def _remove_fixed_frame(self) -> None:
        _remove_fixed_frame(self._fixed_frame_camera, self.root)
        self._fixed_frame_camera = None

    def _remove_owned_identities(self) -> None:
        _remove_owned_identities(self.scene, self.root, self._update_driver)

    def _invalidate_cairo_static_image(self) -> None:
        _invalidate_cairo_static_image(self.scene)

    def restore(self) -> "QuadricOcclusion3D":
        self._cancel_bound_frame_transaction()
        self._attached = False
        self._remove_fixed_frame()
        self._remove_owned_identities()
        for slot in self._surface_slots:
            _hide_vmobject(slot)
        for slot in self._section_slots:
            _hide_vmobject(slot)
        for slots in self._curve_slots.values():
            for slot in slots.fragments:
                slot.hide()
        for slot in self._point_slots.values():
            slot.set_fill(opacity=0.0)
            slot.set_stroke(opacity=0.0)
        self._fragment_slot_maps = {
            source_id: {} for source_id in self._slot_source_ids
        }
        self._last_frame = None
        self._last_global_frame = None
        self._last_section_frame = None
        self._last_boundary_frame = None
        self._display_slot_state = {}
        self._last_painter_band_signature = ()
        self._last_input_geometry_signature = None
        self._last_input_draw_signature = None
        self._last_input_opacity = None
        self._last_prepared_frame = None
        self._last_prepared_performance_counts = {}
        if self._owns_surface_view_cache:
            self._surface_view_cache.clear()
        self._band.restore()
        self.root.reset_opacity()
        self._invalidate_cairo_static_image()
        return self

    def detach(self) -> "QuadricOcclusion3D":
        return self.restore()

    @contextmanager
    def session(self) -> Iterator["QuadricOcclusion3D"]:
        self.attach()
        try:
            yield self
        finally:
            self.restore()

    def slot_identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())

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
    "DEFAULT_QUADRIC_VIEW",
    "PreparedQuadricManimFrame",
    "QUADRIC_MANIM_LIMITS",
    "QuadricBoundaryStyle",
    "QuadricGeometryPrototype",
    "QuadricManimCapacityError",
    "QuadricManimError",
    "QuadricManimLimits",
    "QuadricManimStyle",
    "QuadricOcclusion3D",
    "QuadricOcclusionTransactionSnapshot",
    "estimate_quadric_mobject_count",
]
