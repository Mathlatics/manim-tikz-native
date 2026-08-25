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
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
from manim import (
    BLUE_D,
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
from .contract import (
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
from .curves import EllipseArcCurve, ParametricConicBranch, SegmentCurve
from .global_occlusion import (
    GlobalQuadricFrame,
    GlobalQuadricOcclusionError,
    compute_global_quadric_frame,
)
from .projection import (
    OpaqueProjectionProxy,
    ProjectionProxyError,
    ProjectionSubdivisionError,
    build_opaque_projection_proxy,
)
from .plane_patch import PlanePatchFitError, fit_plane_display_patch
from .section_compositing import (
    PlaneDepthRole,
    QUADRIC_SECTION_COMPOSITING_LIMITS,
    QuadricSectionCompositingError,
    QuadricSectionCompositingFrame,
    QuadricSectionCompositingLimits,
    compute_quadric_section_compositing,
    quadric_plane_fragment_contours,
)
from .visibility import compute_quadric_visibility
from .boundary_section import compute_boundary_section_spans
from .surface_boundaries import (
    GeneratorBoundarySpec,
    build_surface_boundary_sources,
    curve_boundary_source,
    plane_outline_sources,
    surface_boundary_source_ids,
)

QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec
AnalyticCurve3D = SegmentCurve | EllipseArcCurve | ParametricConicBranch
SurfaceInput = Sequence[QuadricSurfaceSpec] | Callable[[], Sequence[QuadricSurfaceSpec]]
CurveInput = Sequence[AnalyticCurve3D] | Callable[[], Sequence[AnalyticCurve3D]]
CurveOpacityInput = (
    Mapping[str, float] | Callable[[], Mapping[str, float]] | None
)
SectionPlaneInput = SectionPlane | Callable[[], SectionPlane] | None
PlanePatchInput = (
    PlaneDisplayPatchSpec | Callable[[], PlaneDisplayPatchSpec] | None
)
ProjectionInput = (
    ParallelView
    | Sequence[Sequence[float]]
    | Callable[[object], ParallelView | Sequence[Sequence[float]]]
)
BoundaryGeneratorInput = Sequence[GeneratorBoundarySpec]


DEFAULT_QUADRIC_VIEW = ParallelView.from_matrix(
    (
        (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
        (-1.0 / sqrt(6.0), -1.0 / sqrt(6.0), 2.0 / sqrt(6.0)),
        (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    )
)


class QuadricManimError(RuntimeError):
    """A quadric frame cannot be committed safely to Manim."""


class QuadricManimCapacityError(QuadricManimError):
    """A prepared frame exceeds an explicitly preallocated capacity."""


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _non_negative(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


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

    def __post_init__(self) -> None:
        for name in (
            "surface_fill_opacity",
            "surface_stroke_width",
            "surface_stroke_opacity",
            "visible_curve_width",
            "visible_curve_opacity",
            "hidden_curve_width",
            "hidden_curve_opacity",
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
        object.__setattr__(self, "dash_gap", _non_negative(self.dash_gap, "dash_gap"))
        for name in (
            "surface_fill_opacity",
            "surface_stroke_opacity",
            "visible_curve_opacity",
            "hidden_curve_opacity",
            "background_opacity",
            "section_plane_fill_opacity",
            "section_plane_stroke_opacity",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed 1")

    @property
    def dash_period(self) -> float:
        return self.dash_length + self.dash_gap

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
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self,
            "max_projected_length",
            _positive(self.max_projected_length, "max_projected_length"),
        )


QUADRIC_MANIM_LIMITS = QuadricManimLimits()


@dataclass(frozen=True, slots=True)
class _PreparedDash:
    points: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedCurveFragment:
    fragment: QuadricCurvePaintFragment
    slot_index: int
    points: np.ndarray
    dashes: tuple[_PreparedDash, ...]


@dataclass(frozen=True, slots=True)
class _PreparedBoundaryFragment:
    fragment: QuadricBoundaryPaintFragment
    source: QuadricBoundarySource
    slot_index: int
    points: np.ndarray
    dashes: tuple[_PreparedDash, ...]


@dataclass(frozen=True, slots=True)
class _PreparedSurface:
    item_id: str
    surface_id: str
    slot_index: int
    points: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedSectionLayers:
    frame: QuadricSectionCompositingFrame
    surface_points: np.ndarray
    plane_polygons: Mapping[PlaneDepthRole, tuple[np.ndarray, ...]]
    plane_outline_paths: Mapping[PlaneDepthRole, tuple[np.ndarray, ...]]


@dataclass(frozen=True, slots=True)
class _PreparedNumericFrame:
    frame: QuadricCompositingFrame
    global_frame: GlobalQuadricFrame | None
    surfaces: tuple[_PreparedSurface, ...]
    fragments: Mapping[str, tuple[_PreparedCurveFragment, ...]]
    curve_opacities: Mapping[str, float]
    fragment_slot_maps: Mapping[str, Mapping[str, int]]
    item_mobjects: Mapping[str, Mobject]
    painter_draw_order: tuple[str, ...]
    section_layers: _PreparedSectionLayers | None = None
    boundary_frame: QuadricBoundaryCompositingFrame | None = None
    boundary_fragments: Mapping[
        str, tuple[_PreparedBoundaryFragment, ...]
    ] | None = None
    boundary_opacities: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class PreparedQuadricManimFrame:
    numeric: _PreparedNumericFrame
    painter_band: PreparedPainterBand

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


@dataclass(slots=True)
class _MobjectState:
    mobject: object
    points: np.ndarray | None
    z_index: float | None
    attributes: dict[str, object]


def _copy_value(value: object) -> object:
    return value.copy() if isinstance(value, np.ndarray) else value


def _capture_root(root: Mobject) -> tuple[_MobjectState, ...]:
    result: list[_MobjectState] = []
    seen: set[int] = set()
    for member in root.get_family():
        if id(member) in seen:
            continue
        seen.add(id(member))
        points = None
        if hasattr(member, "points"):
            points = np.asarray(member.points, dtype=float).copy()
        attributes: dict[str, object] = {}
        for name in (
            "fill_rgbas",
            "stroke_rgbas",
            "background_stroke_rgbas",
            "fill_opacity",
            "stroke_opacity",
            "background_stroke_opacity",
        ):
            if hasattr(member, name):
                attributes[name] = _copy_value(getattr(member, name))
        raw_z = getattr(member, "z_index", None)
        z_index = None
        if raw_z is not None:
            value = float(raw_z)
            if np.isfinite(value):
                z_index = value
        result.append(_MobjectState(member, points, z_index, attributes))
    return tuple(result)


def _restore_root(states: Sequence[_MobjectState]) -> None:
    for state in states:
        if state.points is not None and hasattr(state.mobject, "points"):
            state.mobject.points = state.points.copy()
        for name, value in state.attributes.items():
            setattr(state.mobject, name, _copy_value(value))
        if state.z_index is not None:
            state.mobject.z_index = state.z_index


class _ManagedQuadricDisplayGroup(VGroup):
    """Fade proxy whose invisible sentinel owns the lifecycle multiplier."""

    def __init__(self, *mobjects: Mobject, opacity_sentinel: Line) -> None:
        self._opacity_sentinel = opacity_sentinel
        super().__init__(*mobjects, opacity_sentinel)

    @property
    def opacity_multiplier(self) -> float:
        rgba = np.asarray(
            getattr(self._opacity_sentinel, "stroke_rgbas", ()), dtype=float
        )
        if rgba.ndim < 2 or rgba.shape[-1] < 4 or not rgba.size:
            return 1.0
        value = float(rgba[0, 3])
        return value if np.isfinite(value) and value >= 0.0 else 0.0

    def set_opacity(
        self, opacity: float, family: bool = True
    ) -> "_ManagedQuadricDisplayGroup":
        del family
        value = _non_negative(opacity, "display opacity multiplier")
        self._opacity_sentinel.set_stroke(opacity=value)
        return self

    def reset_opacity(self) -> None:
        self._opacity_sentinel.set_stroke(opacity=1.0)


def _hide_vmobject(value: VMobject) -> None:
    value.set_fill(opacity=0.0)
    value.set_stroke(opacity=0.0)
    value.set_stroke(opacity=0.0, background=True)


def _set_closed_subpaths(
    value: VMobject,
    polygons: Sequence[np.ndarray],
) -> None:
    """Replace one fixed VMobject with any number of closed polygon subpaths."""

    value.clear_points()
    for raw in polygons:
        points = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3:
            raise QuadricManimError(
                "section display polygons must contain finite three-dimensional points"
            )
        if not np.all(np.isfinite(points)):
            raise QuadricManimError(
                "section display polygons must contain finite three-dimensional points"
            )
        value.start_new_path(points[0])
        value.add_points_as_corners((*points[1:], points[0]))


def _set_open_subpaths(
    value: VMobject,
    paths: Sequence[np.ndarray],
) -> None:
    """Replace one fixed VMobject with independently open polyline subpaths."""

    value.clear_points()
    for raw in paths:
        points = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
            raise QuadricManimError(
                "section outline paths must contain finite 3D polylines"
            )
        if not np.all(np.isfinite(points)):
            raise QuadricManimError(
                "section outline paths must contain finite 3D polylines"
            )
        value.start_new_path(points[0])
        value.add_points_as_corners(points[1:])


class _CurveFragmentSlot:
    def __init__(self, dash_capacity: int) -> None:
        self.solid = VMobject()
        self.dashes = tuple(VMobject() for _ in range(dash_capacity))
        self.dash_group = VGroup(*self.dashes)
        self.root = VGroup(self.solid, self.dash_group)
        self.hide()

    def hide(self) -> None:
        _hide_vmobject(self.solid)
        for dash in self.dashes:
            _hide_vmobject(dash)

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


class _CurveSlots:
    def __init__(self, fragment_capacity: int, dash_capacity: int) -> None:
        self.fragments = tuple(
            _CurveFragmentSlot(dash_capacity) for _ in range(fragment_capacity)
        )
        self.root = VGroup(*(slot.root for slot in self.fragments))

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


def _coerce_view(value: object) -> ParallelView:
    if isinstance(value, ParallelView):
        return value
    try:
        return ParallelView.from_matrix(value)  # type: ignore[arg-type]
    except (SolverError, TypeError, ValueError) as exc:
        raise QuadricManimError(f"invalid parallel projection: {exc}") from exc


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    delta = end - start
    squared = float(np.dot(delta, delta))
    if squared == 0.0:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, delta) / squared)
    ratio = min(1.0, max(0.0, ratio))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def _adaptive_project_curve(
    curve: AnalyticCurve3D,
    view: ParallelView,
    start: float,
    end: float,
    *,
    max_chord_error: float,
    max_segments: int,
) -> np.ndarray:
    """Approximate one exact analytic interval without renderer allocation."""

    screen = view.matrix[:2]
    cache: dict[float, np.ndarray] = {}

    def project(parameter: float) -> np.ndarray:
        key = float(parameter)
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = np.asarray(curve.point(key), dtype=float)
        projected = np.asarray(screen @ value, dtype=float)
        if projected.shape != (2,) or not np.all(np.isfinite(projected)):
            raise QuadricManimError(
                f"curve {curve.curve_id!r} produced a non-finite projection"
            )
        cache[key] = projected
        return projected

    intervals: list[tuple[float, float]] = [(float(start), float(end))]
    probe_fractions = (0.25, 0.5, 0.75)
    while True:
        split: list[int] = []
        for index, (left, right) in enumerate(intervals):
            first = project(left)
            last = project(right)
            observed = max(
                _point_segment_distance(
                    project(left + fraction * (right - left)), first, last
                )
                for fraction in probe_fractions
            )
            if observed > max_chord_error:
                split.append(index)
        if not split:
            break
        if len(intervals) + len(split) > max_segments:
            raise QuadricManimCapacityError(
                f"curve {curve.curve_id!r} needs more than {max_segments} "
                "display segments for max_chord_error"
            )
        marked = set(split)
        refined: list[tuple[float, float]] = []
        for index, (left, right) in enumerate(intervals):
            if index not in marked:
                refined.append((left, right))
                continue
            middle = left + 0.5 * (right - left)
            if middle == left or middle == right:
                raise QuadricManimCapacityError(
                    f"curve {curve.curve_id!r} cannot refine at floating-point scale"
                )
            refined.extend(((left, middle), (middle, right)))
        intervals = refined

    parameters = [intervals[0][0]]
    parameters.extend(right for _left, right in intervals)
    points = [project(parameter) for parameter in parameters]
    precision_floor = 4.0 * max(
        (
            abs(float(np.spacing(value)))
            for point in points
            for value in point
        ),
        default=0.0,
    )
    if precision_floor >= max_chord_error:
        raise QuadricManimError(
            f"curve {curve.curve_id!r} cannot certify max_chord_error at the "
            "available floating-point screen resolution; requested "
            f"{max_chord_error:.17g}, resolution floor {precision_floor:.17g}"
        )
    anchor = points[0]

    def duplicate_tolerance(left: np.ndarray, right: np.ndarray) -> float:
        local_scale = max(
            float(np.linalg.norm(left - anchor)),
            float(np.linalg.norm(right - anchor)),
            max_chord_error,
            np.finfo(float).tiny,
        )
        local_roundoff = 32.0 * np.finfo(float).eps * local_scale
        ulp_roundoff = 2.0 * max(
            *(abs(float(np.spacing(value))) for value in left),
            *(abs(float(np.spacing(value))) for value in right),
        )
        return min(
            max(local_roundoff, ulp_roundoff),
            0.125 * max_chord_error,
        )

    result: list[np.ndarray] = [points[0]]
    source_to_result = [0]
    for point in points[1:]:
        if float(np.linalg.norm(point - result[-1])) > duplicate_tolerance(
            result[-1], point
        ):
            result.append(point)
        source_to_result.append(len(result) - 1)
    if len(result) < 2:
        raise QuadricManimError(
            f"curve {curve.curve_id!r} interval collapses in the selected projection"
        )

    measured_error = 0.0
    certification_fractions = (0.0, *probe_fractions, 1.0)
    for index, (left, right) in enumerate(intervals):
        chord_start = result[source_to_result[index]]
        chord_end = result[source_to_result[index + 1]]
        for fraction in certification_fractions:
            parameter = left + fraction * (right - left)
            measured_error = max(
                measured_error,
                _point_segment_distance(
                    project(parameter),
                    chord_start,
                    chord_end,
                ),
            )
    certified_error = measured_error + precision_floor
    if certified_error > max_chord_error * (
        1.0 + 64.0 * np.finfo(float).eps
    ):
        raise QuadricManimError(
            f"curve {curve.curve_id!r} cannot certify max_chord_error after "
            "floating-point-stable deduplication; requested "
            f"{max_chord_error:.17g}, observed {certified_error:.17g}"
        )
    return np.asarray([(point[0], point[1], 0.0) for point in result], dtype=float)


def _polyline_lengths(points: np.ndarray) -> tuple[np.ndarray, float]:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate((np.asarray((0.0,)), np.cumsum(segment_lengths)))
    return cumulative, float(cumulative[-1])


def _point_at_distance(
    points: np.ndarray, cumulative: np.ndarray, distance: float
) -> np.ndarray:
    if distance <= 0.0:
        return points[0].copy()
    if distance >= float(cumulative[-1]):
        return points[-1].copy()
    index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    index = min(index, len(points) - 2)
    span = float(cumulative[index + 1] - cumulative[index])
    if span <= 0.0:
        return points[index].copy()
    ratio = (distance - float(cumulative[index])) / span
    return points[index] + ratio * (points[index + 1] - points[index])


def _slice_polyline(
    points: np.ndarray,
    cumulative: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    values = [_point_at_distance(points, cumulative, start)]
    for index in range(1, len(points) - 1):
        distance = float(cumulative[index])
        if start < distance < end:
            values.append(points[index].copy())
    values.append(_point_at_distance(points, cumulative, end))
    return np.asarray(values, dtype=float)


def _dash_polyline(
    points: np.ndarray,
    *,
    dash_length: float,
    dash_gap: float,
    capacity: int,
) -> tuple[_PreparedDash, ...]:
    cumulative, length = _polyline_lengths(points)
    if length <= 0.0:
        return ()
    period = dash_length + dash_gap
    result: list[_PreparedDash] = []
    period_index = 0
    while period_index * period < length - 1.0e-12:
        start = period_index * period
        end = min(length, start + dash_length)
        period_index += 1
        if end - start <= 1.0e-12:
            continue
        result.append(_PreparedDash(_slice_polyline(points, cumulative, start, end)))
        if len(result) > capacity:
            raise QuadricManimCapacityError(
                f"dash count exceeds fixed slot capacity {capacity}"
            )
    return tuple(result)


def _adaptive_project_curve_samples(
    curve: AnalyticCurve3D,
    view: ParallelView,
    *,
    max_chord_error: float,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-source parameters and points for stable dash phase."""

    projection = view.matrix[:2]
    cache: dict[float, np.ndarray] = {}

    def project(parameter: float) -> np.ndarray:
        key = float(parameter)
        if key not in cache:
            value = projection @ np.asarray(curve.point(key), dtype=float)
            cache[key] = np.asarray((value[0], value[1], 0.0), dtype=float)
        return cache[key]

    intervals = [(curve.domain.start, curve.domain.end)]
    probes = (0.25, 0.5, 0.75)
    while True:
        split = []
        for index, (left, right) in enumerate(intervals):
            first = project(left)
            last = project(right)
            observed = max(
                _point_segment_distance(
                    project(left + fraction * (right - left)),
                    first,
                    last,
                )
                for fraction in probes
            )
            if observed > max_chord_error:
                split.append(index)
        if not split:
            break
        if len(intervals) + len(split) > max_segments:
            raise QuadricManimCapacityError(
                f"boundary source {curve.curve_id!r} needs more than "
                f"{max_segments} display segments"
            )
        marked = set(split)
        refined = []
        for index, (left, right) in enumerate(intervals):
            if index not in marked:
                refined.append((left, right))
                continue
            middle = left + 0.5 * (right - left)
            if middle == left or middle == right:
                raise QuadricManimCapacityError(
                    f"boundary source {curve.curve_id!r} cannot refine at "
                    "floating-point resolution"
                )
            refined.extend(((left, middle), (middle, right)))
        intervals = refined
    parameters = np.asarray(
        [intervals[0][0], *(right for _left, right in intervals)],
        dtype=float,
    )
    points = np.asarray([project(float(value)) for value in parameters], dtype=float)
    return parameters, points


def _source_distance_at_parameter(
    parameters: np.ndarray,
    points: np.ndarray,
    parameter: float,
) -> float:
    cumulative, _length = _polyline_lengths(points)
    value = float(parameter)
    if value <= float(parameters[0]):
        return 0.0
    if value >= float(parameters[-1]):
        return float(cumulative[-1])
    index = int(np.searchsorted(parameters, value, side="right") - 1)
    index = min(index, len(parameters) - 2)
    span = float(parameters[index + 1] - parameters[index])
    ratio = 0.0 if span <= 0.0 else (value - float(parameters[index])) / span
    segment = float(np.linalg.norm(points[index + 1] - points[index]))
    return float(cumulative[index]) + ratio * segment


def _dash_polyline_anchored(
    points: np.ndarray,
    *,
    source_distance_start: float,
    dash_length: float,
    dash_gap: float,
    capacity: int,
) -> tuple[_PreparedDash, ...]:
    cumulative, length = _polyline_lengths(points)
    if length <= 0.0:
        return ()
    period = dash_length + dash_gap
    global_start = float(source_distance_start)
    global_end = global_start + length
    first_period = max(
        0,
        int(np.floor((global_start - dash_length) / period)) + 1,
    )
    result: list[_PreparedDash] = []
    period_index = first_period
    while period_index * period < global_end - 1.0e-12:
        dash_start = period_index * period
        dash_end = dash_start + dash_length
        period_index += 1
        clipped_start = max(global_start, dash_start) - global_start
        clipped_end = min(global_end, dash_end) - global_start
        if clipped_end - clipped_start <= 1.0e-12:
            continue
        result.append(
            _PreparedDash(
                _slice_polyline(points, cumulative, clipped_start, clipped_end)
            )
        )
        if len(result) > capacity:
            raise QuadricManimCapacityError(
                f"dash count exceeds fixed slot capacity {capacity}"
            )
    return tuple(result)


def _surface_items(
    value: Sequence[QuadricSurfaceSpec],
) -> tuple[QuadricSurfaceSpec, ...]:
    result = tuple(value)
    if not result or not all(
        isinstance(item, (SphereSpec, CylinderSpec, ConeSpec)) for item in result
    ):
        raise TypeError("surfaces must contain at least one supported quadric spec")
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
    caller-supplied surface-order path.
    """

    def __init__(
        self,
        scene: object,
        *,
        surfaces: SurfaceInput,
        curves: CurveInput,
        projection: ProjectionInput | None = None,
        paint_policy: QuadricPaintPolicy | str = QuadricPaintPolicy.DIAGRAMMATIC,
        style: QuadricManimStyle = QuadricManimStyle(),
        limits: QuadricManimLimits = QUADRIC_MANIM_LIMITS,
        max_chord_error: float = 1.0e-3,
        context: GeometryContext | ResolvedGeometryContext | None = None,
        painter_z_band: tuple[float, float] = (20.0, 30.0),
        surface_constraints: Sequence[SurfaceConstraintInput] = (),
        surface_order_mode: str = "automatic",
        allocated_curve_ids: Sequence[str] | None = None,
        curve_opacities: CurveOpacityInput = None,
        section_plane: SectionPlaneInput = None,
        section_patch: PlanePatchInput = None,
        section_patch_margin: float = 0.08,
        section_max_screen_error: float = 0.08,
        section_compositing_limits: QuadricSectionCompositingLimits = (
            QUADRIC_SECTION_COMPOSITING_LIMITS
        ),
        boundary_visibility_mode: str = "legacy",
        include_surface_boundaries: bool = True,
        generator_boundaries: BoundaryGeneratorInput = (),
        allocated_boundary_ids: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(style, QuadricManimStyle):
            raise TypeError("style must be a QuadricManimStyle")
        if not isinstance(limits, QuadricManimLimits):
            raise TypeError("limits must be a QuadricManimLimits")
        if context is not None and not isinstance(
            context, (GeometryContext, ResolvedGeometryContext)
        ):
            raise TypeError(
                "context must be a GeometryContext or ResolvedGeometryContext"
            )
        try:
            policy = QuadricPaintPolicy(paint_policy)
        except (TypeError, ValueError) as exc:
            raise QuadricManimError(
                "paint_policy must be 'physical', 'diagrammatic', or "
                "'depth_aware_diagrammatic'"
            ) from exc
        if boundary_visibility_mode not in ("legacy", "unified"):
            raise QuadricManimError(
                "boundary_visibility_mode must be 'legacy' or 'unified'"
            )
        if not isinstance(include_surface_boundaries, bool):
            raise TypeError("include_surface_boundaries must be a bool")
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
        if not isinstance(
            section_compositing_limits, QuadricSectionCompositingLimits
        ):
            raise TypeError(
                "section_compositing_limits must be a "
                "QuadricSectionCompositingLimits"
            )
        self.scene = scene
        self._surface_input = surfaces
        self._curve_input = curves
        self._projection_input = (
            DEFAULT_QUADRIC_VIEW if projection is None else projection
        )
        self.paint_policy = policy
        self.style = style
        self.limits = limits
        self.max_chord_error = _positive(max_chord_error, "max_chord_error")
        self.context = context
        self.surface_constraints = tuple(surface_constraints)
        self.surface_order_mode = surface_order_mode
        self.boundary_visibility_mode = boundary_visibility_mode
        self.include_surface_boundaries = include_surface_boundaries
        self._generator_boundaries = generators
        self._allocated_boundary_ids_input = allocated_boundary_ids
        self._curve_opacity_input = curve_opacities
        self._section_plane_input = section_plane
        self._section_patch_input = section_patch
        self.section_patch_margin = _non_negative(
            section_patch_margin, "section_patch_margin"
        )
        self.section_max_screen_error = _positive(
            section_max_screen_error, "section_max_screen_error"
        )
        self.section_compositing_limits = section_compositing_limits
        self._section_enabled = section_plane is not None
        self._attached = False
        self._fixed_frame_camera: ThreeDCamera | None = None
        self._last_frame: QuadricCompositingFrame | None = None
        self._last_global_frame: GlobalQuadricFrame | None = None
        self._last_section_frame: QuadricSectionCompositingFrame | None = None
        self._last_boundary_frame: QuadricBoundaryCompositingFrame | None = None

        initial_surfaces = self._resolve_surfaces()
        initial_curves = self._resolve_curves()
        self._surface_ids = tuple(item.surface_id for item in initial_surfaces)
        initial_curve_ids = tuple(item.curve_id for item in initial_curves)
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

        estimated_mobjects = (
            len(self._surface_ids)
            + (10 if self._section_enabled else 0)
            + 1
            + len(self._slot_source_ids)
            * (
                1
                + limits.max_fragments_per_curve * (limits.max_dashes_per_fragment + 3)
            )
            + 4
        )
        if estimated_mobjects > limits.max_total_mobjects:
            raise QuadricManimCapacityError(
                f"preallocated Mobject count {estimated_mobjects} exceeds fixed "
                f"limit {limits.max_total_mobjects}"
            )

        self._surface_slots = tuple(VMobject() for _ in self._surface_ids)
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
        self._fragment_slot_maps: dict[str, dict[str, int]] = {
            curve_id: {} for curve_id in self._slot_source_ids
        }
        self._section_slots = (
            tuple(VMobject() for _index in range(10))
            if self._section_enabled
            else ()
        )
        surface_root = VGroup(*self._surface_slots)
        section_root = VGroup(*self._section_slots)
        curve_root = VGroup(
            *(self._curve_slots[key].root for key in self._slot_source_ids)
        )
        self._opacity_sentinel = Line((0, 0, 0), (1.0e-9, 0, 0), buff=0)
        self._opacity_sentinel.set_stroke(width=0.0, opacity=1.0)
        self.root = _ManagedQuadricDisplayGroup(
            surface_root,
            section_root,
            curve_root,
            opacity_sentinel=self._opacity_sentinel,
        )
        self._update_driver = Mobject()

        # Manim recognizes time-aware updaters by the literal ``dt`` name.
        def update_display(mobject: Mobject, dt: float) -> None:
            del mobject
            if self._attached:
                self.update(dt)

        self._update_driver.add_updater(update_display)
        self._band = ManagedPainterBand(
            z_band=painter_z_band,
            managed_roots=(self.root,),
        )

    def _resolve_surfaces(self) -> tuple[QuadricSurfaceSpec, ...]:
        value = (
            self._surface_input()
            if callable(self._surface_input)
            else self._surface_input
        )
        return _surface_items(value)

    def _resolve_curves(self) -> tuple[AnalyticCurve3D, ...]:
        value = (
            self._curve_input() if callable(self._curve_input) else self._curve_input
        )
        return _curve_items(value)

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

    def _resolve_view(self) -> ParallelView:
        value = (
            self._projection_input(self.scene)
            if callable(self._projection_input)
            else self._projection_input
        )
        return _coerce_view(value)

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

    def _validate_fixed_topology(
        self,
        surfaces: Sequence[QuadricSurfaceSpec],
        curves: Sequence[AnalyticCurve3D],
    ) -> None:
        surface_ids = tuple(item.surface_id for item in surfaces)
        curve_ids = tuple(item.curve_id for item in curves)
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

    def _boundary_sources_for_frame(
        self,
        surfaces: Sequence[QuadricSurfaceSpec],
        curves: Sequence[AnalyticCurve3D],
        view: ParallelView,
        plane: SectionPlane | None,
        patch: PlaneDisplayPatchSpec | None,
    ) -> tuple[QuadricBoundarySource, ...]:
        result = [
            curve_boundary_source(
                curve,
                style_id="style:curve",
            )
            for curve in curves
        ]
        if self.include_surface_boundaries:
            result.extend(
                build_surface_boundary_sources(
                    surfaces,
                    view,
                    self._generator_boundaries,
                    include_cap_rims=True,
                    include_silhouettes=True,
                )
            )
        elif self._generator_boundaries:
            result.extend(
                build_surface_boundary_sources(
                    surfaces,
                    view,
                    self._generator_boundaries,
                    include_cap_rims=False,
                    include_silhouettes=False,
                )
            )
        if plane is not None and patch is not None:
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
    ) -> tuple[object, ...]:
        from itertools import combinations

        result = []
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

    def _prepare_unified_numeric(
        self,
        surfaces: tuple[QuadricSurfaceSpec, ...],
        curves: tuple[AnalyticCurve3D, ...],
        view: ParallelView,
        curve_opacities: Mapping[str, float],
    ) -> _PreparedNumericFrame:
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
                        patch_id=f"{surface.surface_id}:opaque-projection",
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
                    f"unified explicit surface preparation failed: {exc}"
                ) from exc

        surface_plans: list[_PreparedSurface] = []
        item_mobjects: dict[str, Mobject] = {}
        section_layers: _PreparedSectionLayers | None = None
        section_frame: QuadricSectionCompositingFrame | None = None
        plane: SectionPlane | None = None
        patch: PlaneDisplayPatchSpec | None = None
        if self._section_enabled:
            surface = surfaces[0]
            plane = self._resolve_section_plane()
            patch = self._resolve_section_patch(surface, plane)
            try:
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
                contours = quadric_plane_fragment_contours(section_frame)
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
            section_layers = _PreparedSectionLayers(
                section_frame,
                surface_points,
                plane_polygons,
                {role: () for role in PlaneDepthRole},
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
                    _PreparedSurface(item.item_id, item.surface_id, slot_index, points)
                )
                item_mobjects[item.item_id] = self._surface_slots[slot_index]
            parent_ids = frame.draw_order
            parent_relations = frame.order_relations
            surface_item_by_id = {
                item.surface_id: item.item_id for item in frame.surface_items
            }

        sources = self._boundary_sources_for_frame(
            surfaces, curves, view, plane, patch
        )
        non_plane = tuple(
            item
            for item in sources
            if item.source_kind is not BoundarySourceKind.PLANE_PATCH_EDGE
        )
        try:
            spans = compute_boundary_visibility(
                non_plane,
                surfaces,
                view,
                context=self.context,
            )
        except Exception as exc:
            raise QuadricManimError(
                f"semantic boundary visibility failed: {exc}"
            ) from exc
        if section_frame is not None:
            spans.update(self._plane_outline_visibility(section_frame))
        crossings = self._boundary_crossings(sources, spans, view)
        section_spans = (
            {}
            if section_frame is None
            else compute_boundary_section_spans(
                sources,
                section_frame,
                view,
                crossings,
                context=self.context,
            )
        )
        try:
            boundary_frame = compute_quadric_boundary_compositing(
                sources,
                spans,
                paint_policy=self.paint_policy,
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
            )
        except QuadricBoundaryCompositingError as exc:
            raise QuadricManimError(
                f"semantic boundary painter graph failed: {exc}"
            ) from exc

        source_map = {item.source_id: item for item in sources}
        by_source: dict[str, list[QuadricBoundaryPaintFragment]] = {
            item.source_id: [] for item in sources
        }
        for fragment in boundary_frame.fragments:
            if fragment.painted:
                by_source[fragment.source_id].append(fragment)
        next_maps: dict[str, Mapping[str, int]] = {
            source_id: {} for source_id in self._slot_source_ids
        }
        prepared_by_source: dict[
            str, tuple[_PreparedBoundaryFragment, ...]
        ] = {}
        boundary_opacities = {
            item.source_id: curve_opacities.get(item.source_id, 1.0)
            for item in sources
        }
        for source_id in sorted(by_source):
            source = source_map[source_id]
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
            values = []
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
                        f"boundary source {source_id!r} fragment length "
                        f"{length:.9g} exceeds max_projected_length"
                    )
                dashes = (
                    _dash_polyline_anchored(
                        points,
                        source_distance_start=_source_distance_at_parameter(
                            parameters,
                            source_points,
                            fragment.interval.start,
                        ),
                        dash_length=self.style.dash_length,
                        dash_gap=self.style.dash_gap,
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
            raise QuadricManimError(
                "unified Manim items do not cover boundary draw_order"
            )
        return _PreparedNumericFrame(
            frame=frame,
            global_frame=global_frame,
            surfaces=tuple(surface_plans),
            fragments={},
            curve_opacities=curve_opacities,
            fragment_slot_maps=next_maps,
            item_mobjects=item_mobjects,
            painter_draw_order=boundary_frame.draw_order,
            section_layers=section_layers,
            boundary_frame=boundary_frame,
            boundary_fragments=prepared_by_source,
            boundary_opacities=boundary_opacities,
        )

    def _prepare_numeric(self) -> _PreparedNumericFrame:
        surfaces = self._resolve_surfaces()
        curves = self._resolve_curves()
        self._validate_fixed_topology(surfaces, curves)
        active_curve_ids = tuple(item.curve_id for item in curves)
        curve_opacities = self._resolve_curve_opacities(active_curve_ids)
        view = self._resolve_view()
        compositor_style = self.style.compositor_style(
            max_projected_length=self.limits.max_projected_length
        )
        if self.boundary_visibility_mode == "unified":
            return self._prepare_unified_numeric(
                surfaces,
                curves,
                view,
                curve_opacities,
            )
        global_frame: GlobalQuadricFrame | None = None
        if self.surface_order_mode == "automatic":
            try:
                global_frame = compute_global_quadric_frame(
                    curves,
                    surfaces,
                    view,
                    context=self.context,
                    paint_policy=self.paint_policy,
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
            visibility = compute_quadric_visibility(
                curves,
                surfaces,
                view,
                context=self.context,
            )
            active_intervals = None
            if self.paint_policy is QuadricPaintPolicy.PHYSICAL:
                active_intervals = {
                    record.curve_id: tuple(
                        span.interval
                        for span in record.spans
                        if span.kind is VisibilityKind.VISIBLE
                    )
                    for record in visibility.records
                }
            try:
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
                frame = compute_quadric_compositing(
                    visibility,
                    proxies,
                    paint_policy=self.paint_policy,
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
            plane = self._resolve_section_plane()
            patch = self._resolve_section_patch(surface, plane)
            try:
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
                    _PreparedSurface(item.item_id, item.surface_id, slot_index, points)
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
            for fragment in fragments:
                points = _adaptive_project_curve(
                    curve,
                    view,
                    fragment.interval.start,
                    fragment.interval.end,
                    max_chord_error=self.max_chord_error,
                    max_segments=self.limits.max_segments_per_fragment,
                )
                _cumulative, length = _polyline_lengths(points)
                allowance = max(1.0e-12, self.limits.max_projected_length * 1.0e-9)
                if length > self.limits.max_projected_length + allowance:
                    raise QuadricManimCapacityError(
                        f"curve {curve_id!r} projected fragment length {length:.9g} "
                        "exceeds max_projected_length "
                        f"{self.limits.max_projected_length:.9g}"
                    )
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
                    slot.solid if fragment.render_intent == "solid" else slot.dash_group
                )
            prepared_by_curve[curve_id] = tuple(values)

        if set(item_mobjects) != set(painter_draw_order):
            raise QuadricManimError(
                "prepared Manim items do not cover compositor draw_order"
            )
        return _PreparedNumericFrame(
            frame,
            global_frame,
            tuple(surface_plans),
            prepared_by_curve,
            curve_opacities,
            next_maps,
            item_mobjects,
            tuple(painter_draw_order),
            section_layers,
        )

    def _prepare_painter(
        self, numeric: _PreparedNumericFrame
    ) -> PreparedQuadricManimFrame:
        try:
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
        return PreparedQuadricManimFrame(numeric, painter)

    def prepare(self) -> PreparedQuadricManimFrame:
        """Prepare and validate one frame without changing any display slot."""

        if not self._attached:
            raise QuadricManimError("quadric occlusion controller is not attached")
        numeric = self._prepare_numeric()
        return self._prepare_painter(numeric)

    def _apply_surface(
        self,
        prepared: _PreparedSurface,
        opacity: float,
        *,
        draw_stroke: bool = True,
    ) -> None:
        slot = self._surface_slots[prepared.slot_index]
        slot.set_points_as_corners(prepared.points)
        slot.set_fill(
            color=self.style.surface_fill_color,
            opacity=self.style.surface_fill_opacity * opacity,
        )
        slot.set_stroke(
            color=self.style.surface_stroke_color,
            width=self.style.surface_stroke_width,
            opacity=(
                self.style.surface_stroke_opacity * opacity
                if draw_stroke
                else 0.0
            ),
        )

    def _apply_section_layers(
        self,
        prepared: _PreparedSectionLayers,
        opacity: float,
        *,
        draw_legacy_strokes: bool = True,
    ) -> None:
        frame = prepared.frame
        if len(self._section_slots) != len(frame.paint_items.ordered):
            raise QuadricManimCapacityError(
                "section painter slots were not allocated"
            )
        slots = dict(zip(frame.paint_items.ordered, self._section_slots))
        surface_back = slots[frame.paint_items.surface_back]
        surface_front = slots[frame.paint_items.surface_front]

        surface_back.set_points_as_corners(prepared.surface_points)
        surface_front.set_points_as_corners(prepared.surface_points)
        combined_surface_opacity = min(
            1.0,
            self.style.surface_fill_opacity * opacity,
        )
        sheet_opacity = 1.0 - sqrt(max(0.0, 1.0 - combined_surface_opacity))
        surface_back.set_fill(
            color=self.style.surface_fill_color,
            opacity=sheet_opacity,
        )
        surface_back.set_stroke(opacity=0.0)
        surface_front.set_fill(
            color=self.style.surface_fill_color,
            opacity=sheet_opacity,
        )
        surface_front.set_stroke(
            color=self.style.surface_stroke_color,
            width=self.style.surface_stroke_width,
            opacity=(
                self.style.surface_stroke_opacity * opacity
                if draw_legacy_strokes
                else 0.0
            ),
        )

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
                opacity=self.style.section_plane_fill_opacity * opacity,
            )
            slot.set_stroke(opacity=0.0)

        for role, item_id in frame.paint_items.outline_by_role.items():
            slot = slots[item_id]
            _set_open_subpaths(
                slot,
                prepared.plane_outline_paths[role]
                if draw_legacy_strokes
                else (),
            )
            slot.set_fill(opacity=0.0)
            slot.set_stroke(
                color=self.style.section_plane_stroke_color,
                width=self.style.section_plane_stroke_width,
                opacity=(
                    self.style.section_plane_stroke_opacity * opacity
                    if draw_legacy_strokes
                    else 0.0
                ),
            )

    def _boundary_stroke_style(
        self,
        prepared: _PreparedBoundaryFragment,
        opacity: float,
    ) -> tuple[object, float, float]:
        semantic = prepared.source.semantic_kind
        hidden = prepared.fragment.render_intent is BoundaryRenderIntent.DASHED
        if semantic is BoundarySemanticKind.DISPLAY_FRAME:
            color = self.style.section_plane_stroke_color
            width = self.style.section_plane_stroke_width
            base_opacity = self.style.section_plane_stroke_opacity
        elif semantic in {
            BoundarySemanticKind.SURFACE_BOUNDARY,
            BoundarySemanticKind.TRUE_SILHOUETTE,
        }:
            color = self.style.surface_stroke_color
            width = self.style.surface_stroke_width
            base_opacity = self.style.surface_stroke_opacity
        else:
            color = (
                self.style.hidden_curve_color
                if hidden
                else self.style.visible_curve_color
            )
            width = (
                self.style.hidden_curve_width
                if hidden
                else self.style.visible_curve_width
            )
            base_opacity = (
                self.style.hidden_curve_opacity
                if hidden
                else self.style.visible_curve_opacity
            )
            return color, width, base_opacity * opacity
        if hidden:
            ratio = (
                self.style.hidden_curve_width / self.style.visible_curve_width
                if self.style.visible_curve_width > 0.0
                else 0.82
            )
            width *= ratio
            base_opacity *= self.style.hidden_curve_opacity
        return color, width, base_opacity * opacity

    def _apply_boundary_fragment(
        self,
        source_id: str,
        prepared: _PreparedBoundaryFragment,
        opacity: float,
    ) -> None:
        slot = self._curve_slots[source_id].fragments[prepared.slot_index]
        color, width, stroke_opacity = self._boundary_stroke_style(
            prepared, opacity
        )
        if prepared.fragment.render_intent is BoundaryRenderIntent.SOLID:
            slot.solid.set_points_as_corners(prepared.points)
            slot.solid.set_fill(opacity=0.0)
            slot.solid.set_stroke(
                color=color, width=width, opacity=stroke_opacity
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
                color=color, width=width, opacity=stroke_opacity
            )
            dash.set_stroke(
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
                dash.set_cap_style(cap)
            if joint is not None:
                dash.joint_type = joint

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
                color=self.style.hidden_curve_color,
                width=self.style.hidden_curve_width,
                opacity=self.style.hidden_curve_opacity * opacity,
            )
            dash.set_stroke(
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
                dash.set_cap_style(cap)
            if joint is not None:
                dash.joint_type = joint

    def apply(self, prepared: PreparedQuadricManimFrame) -> None:
        """Commit one already validated frame, rolling back on any exception."""

        if not self._attached:
            raise QuadricManimError("quadric occlusion controller is not attached")
        if not isinstance(prepared, PreparedQuadricManimFrame):
            raise TypeError("prepared must be a PreparedQuadricManimFrame")
        root_state = _capture_root(self.root)
        band_state = self._band.capture_active_state()
        previous_maps = {
            curve_id: dict(values)
            for curve_id, values in self._fragment_slot_maps.items()
        }
        previous_frame = self._last_frame
        previous_global_frame = self._last_global_frame
        previous_section_frame = self._last_section_frame
        previous_boundary_frame = self._last_boundary_frame
        opacity = self.root.opacity_multiplier
        try:
            for slot in self._surface_slots:
                _hide_vmobject(slot)
            for slot in self._section_slots:
                _hide_vmobject(slot)
            for slots in self._curve_slots.values():
                for slot in slots.fragments:
                    slot.hide()
            unified = prepared.numeric.boundary_frame is not None
            for surface in prepared.numeric.surfaces:
                self._apply_surface(
                    surface, opacity, draw_stroke=not unified
                )
            if prepared.numeric.section_layers is not None:
                self._apply_section_layers(
                    prepared.numeric.section_layers,
                    opacity,
                    draw_legacy_strokes=not unified,
                )
            for curve_id, fragments in prepared.numeric.fragments.items():
                for fragment in fragments:
                    self._apply_curve_fragment(
                        curve_id,
                        fragment,
                        opacity * prepared.numeric.curve_opacities[curve_id],
                    )
            if prepared.numeric.boundary_fragments is not None:
                boundary_opacities = prepared.numeric.boundary_opacities or {}
                for source_id, fragments in (
                    prepared.numeric.boundary_fragments.items()
                ):
                    for fragment in fragments:
                        self._apply_boundary_fragment(
                            source_id,
                            fragment,
                            opacity * boundary_opacities.get(source_id, 1.0),
                        )
            self._band.apply(prepared.painter_band)
            self._fragment_slot_maps = {
                curve_id: dict(values)
                for curve_id, values in prepared.numeric.fragment_slot_maps.items()
            }
            self._last_frame = prepared.frame
            self._last_global_frame = prepared.global_frame
            self._last_section_frame = prepared.section_frame
            self._last_boundary_frame = prepared.boundary_frame
        except Exception:
            _restore_root(root_state)
            self._band.restore_active_state(band_state)
            self._fragment_slot_maps = previous_maps
            self._last_frame = previous_frame
            self._last_global_frame = previous_global_frame
            self._last_section_frame = previous_section_frame
            self._last_boundary_frame = previous_boundary_frame
            raise

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def display_mobject(self) -> Mobject:
        return self.root

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
        numeric = self._prepare_numeric()
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
            prepared = self._prepare_painter(numeric)
            self._attached = True
            self.apply(prepared)
        except Exception:
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
            raise
        return self

    def update(self, dt: float = 0.0) -> "QuadricOcclusion3D":
        del dt
        if not self._attached:
            raise QuadricManimError("quadric occlusion controller is not attached")
        prepared = self.prepare()
        self.apply(prepared)
        return self

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

    def restore(self) -> "QuadricOcclusion3D":
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
        self._fragment_slot_maps = {
            source_id: {} for source_id in self._slot_source_ids
        }
        self._last_frame = None
        self._last_global_frame = None
        self._last_section_frame = None
        self._last_boundary_frame = None
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
    "QuadricManimCapacityError",
    "QuadricManimError",
    "QuadricManimLimits",
    "QuadricManimStyle",
    "QuadricOcclusion3D",
]
