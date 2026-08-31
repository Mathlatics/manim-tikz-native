"""Static Manim views of certified Dandelin constructions.

This module is intentionally independent of the TikZ compiler and of the
scene-owning quadric Manim facades.  It converts one already-certified
``DandelinConstruction3D`` into ordinary two-dimensional Manim mobjects in
source-coordinate units.  Picture scaling remains the caller's responsibility.

The spatial view is a fixed parallel projection.  Legacy ``diagrammatic``
mode retains its stable authored order.  ``depth_aware_diagrammatic`` reuses
the analytic quadric kernel to split every semantic stroke into visible and
hidden fragments while keeping translucent surface-fill ordering explicitly
diagrammatic.
"""

from __future__ import annotations

from collections import defaultdict, deque
from math import ceil, isfinite, tau
from typing import Sequence

import numpy as np
from manim import (
    Circle,
    DashedVMobject,
    Dot,
    Line,
    Mobject,
    ORIGIN,
    Polygon,
    VGroup,
    VMobject,
)

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.topology import ParameterInterval
from polyhedron_visibility.visibility import VisibilityKind
from polyhedron_visibility.quadrics.critical import AnalyticCurve3D

from polyhedron_visibility.quadrics.conics import ConicKind
from polyhedron_visibility.quadrics.contract import PlaneDisplayPatchSpec
from polyhedron_visibility.quadrics.curves import (
    CircleArcCurve,
    EllipseArcCurve,
    ParametricConicBranch,
    SegmentCurve,
)
from polyhedron_visibility.quadrics.dandelin import DandelinConstruction3D
from polyhedron_visibility.quadrics.dandelin_visibility import (
    DandelinVisibilityError,
    DandelinVisibilityFrame,
    DandelinVisibilityStroke,
    compute_dandelin_visibility_frame,
)
from polyhedron_visibility.quadrics.dandelin_views import (
    DandelinMeridianDiagram2D,
    DandelinSectionPlaneDiagram2D,
    DandelinView2DError,
    build_dandelin_meridian_diagram,
    build_dandelin_section_plane_diagram,
)
from polyhedron_visibility.quadrics.plane_patch import (
    PlanePatchFitError,
    fit_plane_display_patch,
)
from polyhedron_visibility.quadrics.trace import section_trace_curves

from .planar_curve_projection import (
    PlanarCurveProjectionError,
    ProjectedPlanarCurve2D,
    project_planar_curve_2d,
)
from .dandelin_contract import build_dandelin_semantic_plan


_VIEWS = frozenset({"spatial", "meridian", "section-plane"})
_PRESETS = frozenset({"classroom"})
_MODES = frozenset({"diagrammatic", "depth_aware_diagrammatic"})
_DEFAULT_MODE = "diagrammatic"
_VIEW_FLAG_DEFAULTS = {
    "spatial": (True, True, True),
    "meridian": (True, False, True),
    "section-plane": (False, True, True),
}
_CONE_RIM_SAMPLES = 48
_CONE_GENERATOR_COUNT = 8
_SECTION_SAMPLES = 96
_RELATIVE_RANK_TOLERANCE = 64.0 * float(np.finfo(float).eps)

_CLASSROOM = {
    "cone_fill": "#173753",
    "cone_wire": "#67D8EE",
    "plane_fill": "#2CB9A4",
    "plane_stroke": "#7EE5D5",
    "sphere_fill": "#F59E7A",
    "sphere_stroke": "#FFD0B8",
    "section": "#FFD166",
    "contact": "#FF8A5B",
    "focus": "#FFF4A3",
    "directrix": "#C4B5FD",
}


class DandelinFixedViewError(ValueError):
    """One certified construction cannot produce the requested fixed view."""


def _resolved_flag(value: object, label: str, *, default: bool) -> bool:
    if value is None:
        return default
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool or None")
    return value


def _projection_matrix(value: object) -> np.ndarray:
    try:
        authored = np.asarray(value, dtype=object)
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinFixedViewError(
            "spatial projection_matrix must be a finite 3x3 matrix"
        ) from exc
    if (
        matrix.shape != (3, 3)
        or authored.shape != (3, 3)
        or any(isinstance(item, (bool, np.bool_)) for item in authored.flat)
        or not np.all(np.isfinite(matrix))
    ):
        raise DandelinFixedViewError(
            "spatial projection_matrix must be a finite 3x3 matrix"
        )
    screen = matrix[:2]
    scale = float(np.max(np.abs(screen)))
    if not isfinite(scale) or scale <= 0.0:
        raise DandelinFixedViewError(
            "spatial projection_matrix must have two independent screen rows"
        )
    try:
        singular = np.linalg.svd(screen / scale, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise DandelinFixedViewError(
            "spatial projection_matrix screen rank cannot be certified"
        ) from exc
    if (
        singular.shape != (2,)
        or not np.all(np.isfinite(singular))
        or float(singular[1])
        <= _RELATIVE_RANK_TOLERANCE * float(singular[0])
    ):
        raise DandelinFixedViewError(
            "spatial projection_matrix must have two independent screen rows"
        )
    return matrix.copy()


def _point2(value: Sequence[float] | np.ndarray, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinFixedViewError(
            f"{label} must contain two finite coordinates"
        ) from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise DandelinFixedViewError(
            f"{label} must contain two finite coordinates"
        )
    return result


def _point3(value: Sequence[float] | np.ndarray, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinFixedViewError(
            f"{label} must contain three finite coordinates"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise DandelinFixedViewError(
            f"{label} must contain three finite coordinates"
        )
    return result


def _screen_point(
    matrix: np.ndarray,
    world_point: Sequence[float] | np.ndarray,
) -> np.ndarray:
    world = _point3(world_point, "world point")
    with np.errstate(all="ignore"):
        screen = matrix[:2] @ world
    if screen.shape != (2,) or not np.all(np.isfinite(screen)):
        raise DandelinFixedViewError(
            "parallel projection produced a non-finite screen point"
        )
    return np.asarray((float(screen[0]), float(screen[1]), 0.0), dtype=float)


def _local_point(value: Sequence[float] | np.ndarray) -> np.ndarray:
    point = _point2(value, "diagram point")
    return np.asarray((float(point[0]), float(point[1]), 0.0), dtype=float)


def _source_refs(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(sorted(set(values)))
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise DandelinFixedViewError("semantic source refs must be non-empty strings")
    return result


def _tag(
    mobject: Mobject,
    *,
    semantic_kind: str,
    source_refs: Sequence[str],
    semantic_id: str,
    **extra: object,
) -> Mobject:
    metadata: dict[str, object] = {
        "semanticKind": semantic_kind,
        "semanticId": semantic_id,
        "semanticSourceRefs": _source_refs(source_refs),
        **extra,
    }
    # ``metadata`` is convenient for generic consumers; the namespaced alias
    # avoids ambiguity when a host application adds its own metadata field.
    mobject.metadata = metadata
    mobject.dandelin_metadata = metadata
    return mobject


def _semantic_wrapper(
    role: str,
    source_ref: str,
    *members: Mobject,
) -> VGroup:
    if not members:
        raise DandelinFixedViewError(
            f"canonical semantic object {role!r} has no render primitives"
        )
    result = VGroup(*members)
    result._dandelin_plan_role = role
    result._dandelin_plan_source_ref = source_ref
    return result


def _line_mobject(
    start: Sequence[float] | np.ndarray,
    end: Sequence[float] | np.ndarray,
    *,
    color: str,
    width: float,
    semantic_kind: str,
    semantic_id: str,
    source_refs: Sequence[str],
    **metadata: object,
) -> Mobject:
    first = _point3(start, "line start")
    second = _point3(end, "line end")
    extent = float(np.linalg.norm(second[:2] - first[:2]))
    scale = max(1.0, float(np.max(np.abs(first[:2]))), float(np.max(np.abs(second[:2]))))
    if not isfinite(extent):
        raise DandelinFixedViewError("projected line extent must be finite")
    if extent <= _RELATIVE_RANK_TOLERANCE * scale:
        value: Mobject = Dot(first, radius=0.035, color=color)
        rank = 0
    else:
        value = Line(first, second, color=color, stroke_width=width)
        rank = 1
    return _tag(
        value,
        semantic_kind=semantic_kind,
        semantic_id=semantic_id,
        source_refs=source_refs,
        projectionRank=rank,
        **metadata,
    )


def _path_mobject(
    points: Sequence[Sequence[float] | np.ndarray],
    *,
    color: str,
    width: float,
    semantic_kind: str,
    semantic_id: str,
    source_refs: Sequence[str],
    closed: bool = False,
    **metadata: object,
) -> VMobject:
    values = tuple(_point3(point, "path point") for point in points)
    if len(values) < 2:
        raise DandelinFixedViewError("a finite path requires at least two points")
    planar = np.asarray([point[:2] for point in values], dtype=float)
    if not np.all(np.isfinite(planar)):
        raise DandelinFixedViewError("path points must remain finite")
    extent = float(np.max(np.linalg.norm(planar - planar[0], axis=1)))
    if not isfinite(extent) or extent <= 0.0:
        raise DandelinFixedViewError("a finite path must not collapse to one point")
    authored = [*values]
    if closed and not np.array_equal(values[0], values[-1]):
        authored.append(values[0])
    path = VMobject()
    path.set_points_as_corners(authored)
    path.set_fill(opacity=0.0)
    path.set_stroke(color=color, width=width, opacity=1.0)
    return _tag(
        path,
        semantic_kind=semantic_kind,
        semantic_id=semantic_id,
        source_refs=source_refs,
        projectionRank=1,
        **metadata,
    )  # type: ignore[return-value]


def _convex_hull(points: Sequence[Sequence[float] | np.ndarray]) -> tuple[np.ndarray, ...]:
    values = sorted(
        {
            (float(point[0]), float(point[1]))
            for point in (_point2(item, "hull point") for item in points)
        }
    )
    if len(values) <= 2:
        return tuple(np.asarray(item, dtype=float) for item in values)
    scale = max(1.0, max(abs(value) for point in values for value in point))
    tolerance = 128.0 * np.finfo(float).eps * scale * scale

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    hull = (*lower[:-1], *upper[:-1])
    return tuple(np.asarray(item, dtype=float) for item in hull)


def _filled_hull_mobject(
    points: Sequence[Sequence[float] | np.ndarray],
    *,
    fill_color: str,
    fill_opacity: float,
    stroke_color: str,
    stroke_opacity: float,
    semantic_kind: str,
    semantic_id: str,
    source_refs: Sequence[str],
    **metadata: object,
) -> Mobject:
    hull = _convex_hull(points)
    if len(hull) < 2:
        raise DandelinFixedViewError(
            f"{semantic_kind} projects to a point and has no finite display extent"
        )
    if len(hull) == 2:
        return _line_mobject(
            _local_point(hull[0]),
            _local_point(hull[1]),
            color=stroke_color,
            width=1.4,
            semantic_kind=semantic_kind,
            semantic_id=semantic_id,
            source_refs=source_refs,
            fillOpacity=0.0,
            **metadata,
        )
    polygon = Polygon(
        *(_local_point(point) for point in hull),
        color=stroke_color,
        stroke_width=1.2,
        stroke_opacity=stroke_opacity,
        fill_color=fill_color,
        fill_opacity=fill_opacity,
    )
    return _tag(
        polygon,
        semantic_kind=semantic_kind,
        semantic_id=semantic_id,
        source_refs=source_refs,
        projectionRank=2,
        fillOpacity=fill_opacity,
        **metadata,
    )


def _affine_circle_mobject(
    center: Sequence[float] | np.ndarray,
    basis: Sequence[Sequence[float]] | np.ndarray,
    *,
    fill_color: str,
    fill_opacity: float,
    stroke_color: str,
    stroke_width: float,
    stroke_opacity: float = 1.0,
    semantic_kind: str,
    semantic_id: str,
    source_refs: Sequence[str],
    semi_axes: Sequence[float] | None = None,
    **metadata: object,
) -> Circle:
    screen_center = _point2(center, "affine circle center")
    screen_basis = np.asarray(basis, dtype=float)
    if screen_basis.shape != (2, 2) or not np.all(np.isfinite(screen_basis)):
        raise DandelinFixedViewError("affine circle basis must be a finite 2x2 matrix")
    try:
        singular = np.linalg.svd(screen_basis, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise DandelinFixedViewError(
            "affine circle display rank cannot be certified"
        ) from exc
    if (
        singular.shape != (2,)
        or not np.all(np.isfinite(singular))
        or float(singular[1])
        <= _RELATIVE_RANK_TOLERANCE * float(singular[0])
    ):
        raise DandelinFixedViewError(
            "affine circle requires a certifiable rank-two display"
        )
    transform = np.asarray(
        (
            (screen_basis[0, 0], screen_basis[0, 1], 0.0),
            (screen_basis[1, 0], screen_basis[1, 1], 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
    curve = Circle(radius=1.0)
    curve.set_fill(color=fill_color, opacity=fill_opacity)
    curve.set_stroke(
        color=stroke_color,
        width=stroke_width,
        opacity=stroke_opacity,
    )
    curve.apply_matrix(transform, about_point=ORIGIN)
    curve.shift(_local_point(screen_center))
    axes = (
        tuple(float(item) for item in singular)
        if semi_axes is None
        else tuple(float(item) for item in semi_axes)
    )
    return _tag(
        curve,
        semantic_kind=semantic_kind,
        semantic_id=semantic_id,
        source_refs=source_refs,
        projectionRank=2,
        screenCenter=tuple(float(item) for item in screen_center),
        screenBasis=tuple(tuple(float(item) for item in row) for row in screen_basis),
        semiAxes=axes,
        strokeOpacity=stroke_opacity,
        **metadata,
    )  # type: ignore[return-value]


def _projected_planar_curve_mobject(
    projected: ProjectedPlanarCurve2D,
    *,
    color: str,
    width: float,
    semantic_kind: str,
    semantic_id: str,
    source_refs: Sequence[str],
) -> Mobject:
    if projected.rank == 1:
        assert projected.segment_start is not None
        assert projected.segment_end is not None
        return _line_mobject(
            _local_point(projected.segment_start),
            _local_point(projected.segment_end),
            color=color,
            width=width,
            semantic_kind=semantic_kind,
            semantic_id=semantic_id,
            source_refs=source_refs,
        )
    return _affine_circle_mobject(
        projected.center,
        projected.screen_basis,
        fill_color=color,
        fill_opacity=0.0,
        stroke_color=color,
        stroke_width=width,
        semantic_kind=semantic_kind,
        semantic_id=semantic_id,
        source_refs=source_refs,
        semi_axes=projected.singular_values,
    )


def _sphere_mobject(
    construction: DandelinConstruction3D,
    sphere_index: int,
    matrix: np.ndarray,
    *,
    stroke_opacity: float = 1.0,
) -> Circle:
    record = construction.spheres[sphere_index]
    screen = matrix[:2]
    scale = float(np.max(np.abs(screen)))
    try:
        left, normalized_singular, _right = np.linalg.svd(
            screen / scale,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as exc:
        raise DandelinFixedViewError(
            "sphere affine projection cannot be certified"
        ) from exc
    singular = scale * normalized_singular
    with np.errstate(all="ignore"):
        semi_axes = record.sphere.radius * singular
        basis = left @ np.diag(semi_axes)
        center = screen @ np.asarray(record.sphere.center, dtype=float)
    if (
        singular.shape != (2,)
        or not np.all(np.isfinite(semi_axes))
        or not np.all(np.isfinite(basis))
        or not np.all(np.isfinite(center))
        or float(semi_axes[1]) <= 0.0
    ):
        raise DandelinFixedViewError(
            "sphere projection lies outside the finite affine display range"
        )
    return _affine_circle_mobject(
        center,
        basis,
        fill_color=_CLASSROOM["sphere_fill"],
        fill_opacity=0.22,
        stroke_color=_CLASSROOM["sphere_stroke"],
        stroke_width=1.8,
        stroke_opacity=stroke_opacity,
        semantic_kind="dandelin_sphere",
        semantic_id=record.sphere_id,
        source_refs=(record.sphere_id,),
        semi_axes=semi_axes,
        worldCenter=record.sphere.center,
        worldRadius=record.sphere.radius,
    )


def _display_patch(
    construction: DandelinConstruction3D,
    *,
    include_directrices: bool,
) -> PlaneDisplayPatchSpec:
    try:
        base = fit_plane_display_patch(
            f"{construction.plane.plane_id}:dandelin-fixed-view-base",
            construction.plane,
            construction.cone.render_components,
            margin_ratio=0.14,
        ).patch
    except PlanePatchFitError as exc:
        raise DandelinFixedViewError(
            f"section-plane display patch cannot be fitted: {exc}"
        ) from exc
    if not include_directrices or not construction.directrices:
        return base
    center = np.asarray(base.center_coordinates, dtype=float)
    lower = center - np.asarray((base.half_width, base.half_height), dtype=float)
    upper = center + np.asarray((base.half_width, base.half_height), dtype=float)
    padding = 0.18 * max(base.half_width, base.half_height)
    for directrix in construction.directrices:
        point = np.asarray(directrix.point.coordinates, dtype=float)
        lower = np.minimum(lower, point - padding)
        upper = np.maximum(upper, point + padding)
    expanded_center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    if not np.all(np.isfinite((*expanded_center, *half))) or np.any(half <= 0.0):
        raise DandelinFixedViewError(
            "directrix display patch has no finite positive extent"
        )
    return PlaneDisplayPatchSpec(
        f"{construction.plane.plane_id}:dandelin-fixed-view",
        construction.plane.plane_id,
        float(half[0]),
        float(half[1]),
        (float(expanded_center[0]), float(expanded_center[1])),
    )


def _bounds_from_patch(
    patch: PlaneDisplayPatchSpec,
) -> tuple[float, float, float, float]:
    center_x, center_y = patch.center_coordinates
    return (
        center_x - patch.half_width,
        center_y - patch.half_height,
        center_x + patch.half_width,
        center_y + patch.half_height,
    )


def _clip_infinite_line(
    anchor: Sequence[float] | np.ndarray,
    direction: Sequence[float] | np.ndarray,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    point = _point2(anchor, "line anchor")
    vector = _point2(direction, "line direction")
    length = float(np.linalg.norm(vector))
    if not isfinite(length) or length <= 0.0:
        raise DandelinFixedViewError("line direction must be non-zero")
    vector /= length
    lower = np.asarray((bounds[0], bounds[1]), dtype=float)
    upper = np.asarray((bounds[2], bounds[3]), dtype=float)
    scale = max(1.0, float(np.max(np.abs((*point, *lower, *upper)))))
    epsilon = 128.0 * np.finfo(float).eps * scale
    parameter_min = -float("inf")
    parameter_max = float("inf")
    for axis in range(2):
        component = float(vector[axis])
        if abs(component) <= _RELATIVE_RANK_TOLERANCE:
            if point[axis] < lower[axis] - epsilon or point[axis] > upper[axis] + epsilon:
                raise DandelinFixedViewError(
                    "infinite line does not intersect the finite display bounds"
                )
            continue
        first = float((lower[axis] - point[axis]) / component)
        second = float((upper[axis] - point[axis]) / component)
        parameter_min = max(parameter_min, min(first, second))
        parameter_max = min(parameter_max, max(first, second))
    if (
        not isfinite(parameter_min)
        or not isfinite(parameter_max)
        or parameter_max - parameter_min <= epsilon
    ):
        raise DandelinFixedViewError(
            "infinite line has no finite segment inside the display bounds"
        )
    return point + parameter_min * vector, point + parameter_max * vector


def _sample_curve(
    curve: ParametricConicBranch,
    *,
    spatial_matrix: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    parameters = np.linspace(
        curve.domain.start,
        curve.domain.end,
        _SECTION_SAMPLES + 1,
    )
    result: list[np.ndarray] = []
    for parameter in parameters:
        if spatial_matrix is None:
            point = curve.parameterization.point(float(parameter))
            result.append(_local_point(point))
        else:
            result.append(_screen_point(spatial_matrix, curve.point(float(parameter))))
    if curve.closed and not np.array_equal(result[0], result[-1]):
        result.append(result[0])
    return tuple(result)


def _section_curve_mobjects(
    diagram: DandelinSectionPlaneDiagram2D,
    *,
    spatial_matrix: np.ndarray | None,
) -> tuple[Mobject, ...]:
    result: list[Mobject] = []
    for curve in section_trace_curves(diagram.conic_trace):
        result.append(
            _path_mobject(
                _sample_curve(curve, spatial_matrix=spatial_matrix),
                color=_CLASSROOM["section"],
                width=3.0,
                semantic_kind="section_curve",
                semantic_id=curve.curve_id,
                source_refs=(curve.curve_id,),
                closed=curve.closed,
                supportingKind=diagram.supporting_kind.value,
            )
        )
    if not result:
        raise DandelinFixedViewError(
            "certified Dandelin section has no finite curve to display"
        )
    return tuple(result)


def _focus_groups(
    values: Sequence[tuple[str, np.ndarray]],
    *,
    supporting_kind: ConicKind,
    tolerance: float,
) -> tuple[tuple[tuple[str, ...], np.ndarray], ...]:
    ordered = tuple(sorted(values, key=lambda item: item[0]))
    # A circle has coincident focus positions, but the two certified sphere
    # relations remain distinct semantic objects.  Preserve both identities and
    # allow their dots to overlap visually instead of merging the source refs.
    del supporting_kind, tolerance
    return tuple(((source_ref,), point) for source_ref, point in ordered)


def _focus_mobjects(
    values: Sequence[tuple[str, np.ndarray]],
    *,
    supporting_kind: ConicKind,
    tolerance: float,
    semantic_prefix: str,
) -> tuple[Mobject, ...]:
    result: list[Mobject] = []
    for index, (refs, point) in enumerate(
        _focus_groups(
            values,
            supporting_kind=supporting_kind,
            tolerance=tolerance,
        )
    ):
        dot = Dot(point, radius=0.065, color=_CLASSROOM["focus"])
        result.append(
            _tag(
                dot,
                semantic_kind="focus",
                semantic_id=f"{semantic_prefix}:focus:{index:04d}",
                source_refs=refs,
                coincidentSourceCount=len(refs),
            )
        )
    return tuple(result)


def _cone_nappe_intervals(
    construction: DandelinConstruction3D,
) -> tuple[tuple[str, float, float], ...]:
    lower, upper = construction.cone.axial_range
    result: list[tuple[str, float, float]] = []
    if lower < 0.0:
        result.append(("negative", lower, min(upper, 0.0)))
    if upper > 0.0:
        result.append(("positive", max(lower, 0.0), upper))
    if not result:
        raise DandelinFixedViewError("finite cone has no authored nappe extent")
    return tuple(result)


def _cone_world_point(
    construction: DandelinConstruction3D,
    axial: float,
    angle: float,
) -> np.ndarray:
    cone = construction.cone
    apex = np.asarray(cone.apex, dtype=float)
    axis = np.asarray(cone.axis, dtype=float)
    frame = cone.frame
    radial = np.cos(angle) * np.asarray(frame.x_axis, dtype=float) + np.sin(
        angle
    ) * np.asarray(frame.y_axis, dtype=float)
    point = apex + axial * axis + abs(axial) * cone.slope * radial
    return _point3(point, "cone wireframe point")


def _spatial_cone_mobjects(
    construction: DandelinConstruction3D,
    matrix: np.ndarray,
    *,
    include_wires: bool = True,
) -> tuple[Mobject, ...]:
    cone = construction.cone
    faces: list[Mobject] = []
    wires: list[Mobject] = []
    angles = tuple(
        float(value)
        for value in np.linspace(0.0, tau, _CONE_RIM_SAMPLES, endpoint=False)
    )
    generator_angles = tuple(
        float(value)
        for value in np.linspace(0.0, tau, _CONE_GENERATOR_COUNT, endpoint=False)
    )
    for nappe, start_axial, end_axial in _cone_nappe_intervals(construction):
        projected_rims = {
            axial: tuple(
                _screen_point(matrix, _cone_world_point(construction, axial, angle))
                for angle in angles
            )
            for axial in (start_axial, end_axial)
        }
        face_points = tuple(
            point[:2]
            for axial in (start_axial, end_axial)
            for point in projected_rims[axial]
        )
        faces.append(
            _filled_hull_mobject(
                face_points,
                fill_color=_CLASSROOM["cone_fill"],
                fill_opacity=0.13,
                stroke_color=_CLASSROOM["cone_wire"],
                stroke_opacity=0.35,
                semantic_kind="cone_face",
                semantic_id=f"{cone.surface_id}:fixed-view:nappe:{nappe}:face",
                source_refs=(cone.surface_id,),
                nappe=nappe,
                representation="projected-convex-nappe",
            )
        )
        if not include_wires:
            continue
        for axial, rim in projected_rims.items():
            if abs(axial) <= np.finfo(float).eps:
                continue
            wires.append(
                _path_mobject(
                    (*rim, rim[0]),
                    color=_CLASSROOM["cone_wire"],
                    width=1.2,
                    semantic_kind="cone_rim",
                    semantic_id=f"{cone.surface_id}:fixed-view:rim:{axial:.17g}",
                    source_refs=(cone.surface_id,),
                    closed=True,
                    nappe=nappe,
                    axialCoordinate=axial,
                )
            )
        for index, angle in enumerate(generator_angles):
            start = _screen_point(
                matrix,
                _cone_world_point(construction, start_axial, angle),
            )
            end = _screen_point(
                matrix,
                _cone_world_point(construction, end_axial, angle),
            )
            wires.append(
                _line_mobject(
                    start,
                    end,
                    color=_CLASSROOM["cone_wire"],
                    width=1.0,
                    semantic_kind="cone_generator",
                    semantic_id=(
                        f"{cone.surface_id}:fixed-view:nappe:{nappe}:"
                        f"generator:{index:04d}"
                    ),
                    source_refs=(cone.surface_id,),
                    nappe=nappe,
                )
            )
    return (*faces, *wires)


def _diagrammatic_spatial_view(
    construction: DandelinConstruction3D,
    matrix: np.ndarray,
    *,
    show_contact_circles: bool,
    show_directrices: bool,
    show_foci: bool,
) -> tuple[Mobject, ...]:
    diagram = build_dandelin_section_plane_diagram(construction)
    patch = _display_patch(
        construction,
        include_directrices=show_directrices,
    )
    result: list[Mobject] = [
        _semantic_wrapper(
            "cone_surface",
            construction.cone.surface_id,
            *_spatial_cone_mobjects(construction, matrix),
        )
    ]

    plane_points = tuple(
        _screen_point(matrix, point)
        for point in patch.corners(construction.plane)
    )
    result.append(
        _semantic_wrapper(
            "section_plane",
            construction.plane.plane_id,
            _filled_hull_mobject(
            tuple(point[:2] for point in plane_points),
            fill_color=_CLASSROOM["plane_fill"],
            fill_opacity=0.12,
            stroke_color=_CLASSROOM["plane_stroke"],
            stroke_opacity=0.65,
            semantic_kind="section_plane",
            semantic_id=patch.patch_id,
            source_refs=(construction.plane.plane_id,),
            patchId=patch.patch_id,
            ),
        )
    )

    for index, record in enumerate(construction.spheres):
        result.append(
            _semantic_wrapper(
                "sphere_surface",
                record.sphere_id,
                _sphere_mobject(construction, index, matrix),
            )
        )
    result.append(
        _semantic_wrapper(
            "section_curve",
            f"{construction.construction_id}:section",
            *_section_curve_mobjects(diagram, spatial_matrix=matrix),
        )
    )

    if show_contact_circles:
        for record in construction.spheres:
            try:
                projected = project_planar_curve_2d(
                    record.cone_contact_circle,
                    matrix,
                )
            except PlanarCurveProjectionError as exc:
                raise DandelinFixedViewError(
                    f"contact circle projection failed: {exc}"
                ) from exc
            result.append(
                _semantic_wrapper(
                    "contact_circle",
                    record.cone_contact_circle.curve_id,
                    _projected_planar_curve_mobject(
                    projected,
                    color=_CLASSROOM["contact"],
                    width=2.0,
                    semantic_kind="contact_circle",
                    semantic_id=record.cone_contact_circle.curve_id,
                    source_refs=(record.cone_contact_circle.curve_id,),
                    ),
                )
            )
    if show_directrices:
        try:
            directrices = construction.directrix_segments(
                patch,
                context=construction.certification_context,
            )
        except ValueError as exc:
            raise DandelinFixedViewError(
                f"directrix clipping failed: {exc}"
            ) from exc
        for segment, directrix in zip(directrices, construction.directrices):
            result.append(
                _semantic_wrapper(
                    "directrix",
                    directrix.directrix_id,
                    _line_mobject(
                    _screen_point(matrix, segment.start),
                    _screen_point(matrix, segment.end),
                    color=_CLASSROOM["directrix"],
                    width=1.6,
                    semantic_kind="directrix",
                    semantic_id=directrix.directrix_id,
                    source_refs=(directrix.directrix_id,),
                    sphereId=directrix.sphere_id,
                    ),
                )
            )
    if show_foci:
        focus_values = tuple(
            (
                record.focus_id,
                _screen_point(matrix, record.focus.world_point),
            )
            for record in construction.spheres
        )
        for focus in _focus_mobjects(
            focus_values,
            supporting_kind=construction.supporting_kind,
            tolerance=diagram.certification_tolerance,
            semantic_prefix=f"{construction.construction_id}:spatial",
        ):
            source_ref = focus.dandelin_metadata["semanticSourceRefs"][0]
            result.append(_semantic_wrapper("focus", source_ref, focus))
    return tuple(result)


_AUTOMATIC_STROKE_STYLE = {
    "cone_boundary": (_CLASSROOM["cone_wire"], 1.25),
    "sphere_silhouette": (_CLASSROOM["sphere_stroke"], 1.8),
    "section_curve": (_CLASSROOM["section"], 3.0),
    "contact_circle": (_CLASSROOM["contact"], 2.0),
    "directrix": (_CLASSROOM["directrix"], 1.6),
}


def _visibility_curve_points(
    curve: AnalyticCurve3D,
    interval: ParameterInterval,
    matrix: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if isinstance(curve, SegmentCurve):
        count = 2
    elif isinstance(curve, (CircleArcCurve, EllipseArcCurve)):
        fraction = interval.length / curve.domain.length
        count = max(4, int(ceil(64.0 * fraction)) + 1)
    elif isinstance(curve, ParametricConicBranch):
        fraction = interval.length / curve.domain.length
        count = max(6, int(ceil(_SECTION_SAMPLES * fraction)) + 1)
    else:  # pragma: no cover - guarded by the renderer-neutral contract
        raise DandelinFixedViewError(
            f"unsupported automatic visibility curve {type(curve).__name__}"
        )
    parameters = np.linspace(interval.start, interval.end, count)
    points: list[np.ndarray] = []
    for parameter in parameters:
        point = _screen_point(matrix, curve.point(float(parameter)))
        if not points or not np.array_equal(point, points[-1]):
            points.append(point)
    if len(points) < 2:
        raise DandelinFixedViewError(
            f"visibility fragment on {curve.curve_id!r} collapses in projection"
        )
    return tuple(points)


def _visibility_fragment_mobject(
    stroke: DandelinVisibilityStroke,
    span_index: int,
    matrix: np.ndarray,
) -> Mobject:
    span = stroke.spans[span_index]
    color, width = _AUTOMATIC_STROKE_STYLE[stroke.role]
    points = _visibility_curve_points(stroke.source.curve, span.interval, matrix)
    fragment_id = f"{stroke.source_id}:visibility-span:{span_index:04d}"
    base = _path_mobject(
        points,
        color=color,
        width=width,
        semantic_kind=f"{stroke.role}_fragment",
        semantic_id=fragment_id,
        source_refs=(stroke.source_ref,),
        closed=(
            bool(getattr(stroke.source.curve, "closed", False))
            and span.interval == stroke.source.curve.domain
        ),
        visibilitySourceId=stroke.source_id,
        visibilityKind=span.kind.value,
        occluderSurfaceIds=span.occluder_surface_ids,
        interval=(span.interval.start, span.interval.end),
        curveVisibilityAuthoritative=True,
    )
    if span.kind is VisibilityKind.VISIBLE:
        base.dandelin_metadata["renderIntent"] = "solid"
        base.metadata["renderIntent"] = "solid"
        return base
    length = float(base.get_arc_length())
    if not isfinite(length) or length <= 0.0:
        raise DandelinFixedViewError(
            f"hidden visibility fragment on {stroke.source_id!r} has no length"
        )
    dashed = DashedVMobject(
        base,
        num_dashes=max(2, min(128, int(ceil(length / 0.14)))),
        dashed_ratio=0.56,
    )
    dashed.set_stroke(color=color, width=max(0.8, 0.72 * width), opacity=0.48)
    return _tag(
        dashed,
        semantic_kind=f"{stroke.role}_fragment",
        semantic_id=fragment_id,
        source_refs=(stroke.source_ref,),
        projectionRank=1,
        visibilitySourceId=stroke.source_id,
        visibilityKind=span.kind.value,
        occluderSurfaceIds=span.occluder_surface_ids,
        interval=(span.interval.start, span.interval.end),
        renderIntent="dashed",
        curveVisibilityAuthoritative=True,
    )


def _visibility_fragments(
    frame: DandelinVisibilityFrame,
    matrix: np.ndarray,
    *,
    role: str,
    source_ref: str | None = None,
) -> tuple[Mobject, ...]:
    result = tuple(
        _visibility_fragment_mobject(stroke, span_index, matrix)
        for stroke in frame.strokes
        if stroke.role == role
        and (source_ref is None or stroke.source_ref == source_ref)
        for span_index in range(len(stroke.spans))
    )
    if not result:
        suffix = "" if source_ref is None else f" for {source_ref!r}"
        raise DandelinFixedViewError(
            f"automatic Dandelin visibility has no {role!r} fragments{suffix}"
        )
    return result


def _replace_wrapper_members(
    wrapper: VGroup,
    members: Sequence[Mobject],
) -> None:
    if wrapper.submobjects:
        wrapper.remove(*tuple(wrapper.submobjects))
    wrapper.add(*tuple(members))


def _depth_aware_spatial_view(
    construction: DandelinConstruction3D,
    matrix: np.ndarray,
    *,
    show_contact_circles: bool,
    show_directrices: bool,
    show_foci: bool,
) -> tuple[tuple[Mobject, ...], DandelinVisibilityFrame]:
    objects = list(
        _diagrammatic_spatial_view(
            construction,
            matrix,
            show_contact_circles=show_contact_circles,
            show_directrices=show_directrices,
            show_foci=show_foci,
        )
    )
    patch = _display_patch(
        construction,
        include_directrices=show_directrices,
    )
    try:
        frame = compute_dandelin_visibility_frame(
            construction,
            ParallelView.from_matrix(matrix),
            directrix_patch=patch if show_directrices else None,
            include_contact_circles=show_contact_circles,
            include_directrices=show_directrices,
            generator_count=_CONE_GENERATOR_COUNT,
        )
    except DandelinVisibilityError as exc:
        raise DandelinFixedViewError(
            f"automatic Dandelin visibility cannot be certified: {exc}"
        ) from exc

    for item in objects:
        role = getattr(item, "_dandelin_plan_role", None)
        source_ref = getattr(item, "_dandelin_plan_source_ref", None)
        if role == "cone_surface":
            fills = tuple(
                member
                for member in item.submobjects
                if getattr(member, "dandelin_metadata", {}).get("semanticKind")
                == "cone_face"
            )
            _replace_wrapper_members(
                item,
                (
                    *fills,
                    *_visibility_fragments(
                        frame,
                        matrix,
                        role="cone_boundary",
                    ),
                ),
            )
        elif role == "sphere_surface":
            fills = tuple(item.submobjects)
            for fill in fills:
                fill.set_stroke(opacity=0.0)
                if isinstance(getattr(fill, "dandelin_metadata", None), dict):
                    fill.dandelin_metadata["strokeOpacity"] = 0.0
                    fill.metadata["strokeOpacity"] = 0.0
            _replace_wrapper_members(
                item,
                (
                    *fills,
                    *_visibility_fragments(
                        frame,
                        matrix,
                        role="sphere_silhouette",
                        source_ref=source_ref,
                    ),
                ),
            )
        elif role == "section_curve":
            _replace_wrapper_members(
                item,
                _visibility_fragments(
                    frame,
                    matrix,
                    role="section_curve",
                    source_ref=source_ref,
                ),
            )
        elif role == "contact_circle":
            _replace_wrapper_members(
                item,
                _visibility_fragments(
                    frame,
                    matrix,
                    role="contact_circle",
                    source_ref=source_ref,
                ),
            )
        elif role == "directrix":
            _replace_wrapper_members(
                item,
                _visibility_fragments(
                    frame,
                    matrix,
                    role="directrix",
                    source_ref=source_ref,
                ),
            )
    return tuple(objects), frame


def _meridian_bounds(
    diagram: DandelinMeridianDiagram2D,
) -> tuple[float, float, float, float]:
    points: list[np.ndarray] = []
    for generator in diagram.generators:
        points.extend(
            (
                np.asarray(generator.start.coordinates, dtype=float),
                np.asarray(generator.end.coordinates, dtype=float),
            )
        )
    for circle in diagram.sphere_circles:
        center = np.asarray(circle.center_coordinates, dtype=float)
        radius = circle.radius
        points.extend(
            (
                center + np.asarray((radius, 0.0)),
                center - np.asarray((radius, 0.0)),
                center + np.asarray((0.0, radius)),
                center - np.asarray((0.0, radius)),
            )
        )
    points.extend(np.asarray(item.coordinates, dtype=float) for item in diagram.focus_points)
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise DandelinFixedViewError("meridian geometry has no finite display bounds")
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    span = upper - lower
    scale = max(1.0, float(np.max(np.abs(values))))
    if np.any(span <= _RELATIVE_RANK_TOLERANCE * scale):
        raise DandelinFixedViewError("meridian geometry has degenerate display bounds")
    margin = 0.12 * max(float(span[0]), float(span[1]))
    return (
        float(lower[0] - margin),
        float(lower[1] - margin),
        float(upper[0] + margin),
        float(upper[1] + margin),
    )


def _meridian_view(
    construction: DandelinConstruction3D,
    *,
    show_contact_circles: bool,
    show_foci: bool,
) -> tuple[Mobject, ...]:
    diagram = build_dandelin_meridian_diagram(construction)
    bounds = _meridian_bounds(diagram)
    result: list[Mobject] = []

    for nappe in ("negative", "positive"):
        generators = tuple(
            item for item in diagram.generators if f":nappe:{nappe}:" in item.segment_id
        )
        if not generators:
            continue
        hull_points = tuple(
            point
            for generator in generators
            for point in (generator.start.coordinates, generator.end.coordinates)
        )
        result.append(
            _semantic_wrapper(
                "cone_face",
                construction.cone.surface_id,
                _filled_hull_mobject(
                    hull_points,
                    fill_color=_CLASSROOM["cone_fill"],
                    fill_opacity=0.13,
                    stroke_color=_CLASSROOM["cone_wire"],
                    stroke_opacity=0.35,
                    semantic_kind="cone_face",
                    semantic_id=f"{diagram.diagram_id}:nappe:{nappe}:face",
                    source_refs=(construction.cone.surface_id,),
                    nappe=nappe,
                ),
            )
        )
    for generator in diagram.generators:
        result.append(
            _semantic_wrapper(
                "cone_generator",
                generator.source_ref,
                _line_mobject(
                    _local_point(generator.start.coordinates),
                    _local_point(generator.end.coordinates),
                    color=_CLASSROOM["cone_wire"],
                    width=1.5,
                    semantic_kind="cone_generator",
                    semantic_id=generator.segment_id,
                    source_refs=(generator.source_ref,),
                ),
            )
        )
    section_start, section_end = _clip_infinite_line(
        diagram.section_line.point_coordinates,
        diagram.section_line.direction_coordinates,
        bounds,
    )
    result.append(
        _semantic_wrapper(
            "section_line",
            diagram.section_line.source_ref,
            _line_mobject(
                _local_point(section_start),
                _local_point(section_end),
                color=_CLASSROOM["plane_stroke"],
                width=2.0,
                semantic_kind="section_line",
                semantic_id=diagram.section_line.line_id,
                source_refs=(diagram.section_line.source_ref,),
            ),
        )
    )
    for circle in diagram.sphere_circles:
        basis = np.asarray(((circle.radius, 0.0), (0.0, circle.radius)), dtype=float)
        result.append(
            _semantic_wrapper(
                "sphere_circle_section",
                circle.source_ref,
                _affine_circle_mobject(
                    circle.center_coordinates,
                    basis,
                    fill_color=_CLASSROOM["sphere_fill"],
                    fill_opacity=0.18,
                    stroke_color=_CLASSROOM["sphere_stroke"],
                    stroke_width=1.8,
                    semantic_kind="sphere_circle_section",
                    semantic_id=circle.circle_id,
                    source_refs=(circle.source_ref,),
                    semi_axes=(circle.radius, circle.radius),
                    sphereId=circle.sphere_id,
                ),
            )
        )
    if show_contact_circles:
        contacts = tuple(
            item
            for item in diagram.tangencies
            if item.carrier_id != diagram.section_line.line_id
        )
        for contact in contacts:
            dot = Dot(
                _local_point(contact.contact.coordinates),
                radius=0.045,
                color=_CLASSROOM["contact"],
            )
            result.append(
                _semantic_wrapper(
                    "contact_circle_section_point",
                    contact.source_ref,
                    _tag(
                        dot,
                        semantic_kind="contact_circle_section_point",
                        semantic_id=contact.tangency_id,
                        source_refs=(contact.source_ref,),
                        sphereId=contact.sphere_id,
                        carrierId=contact.carrier_id,
                    ),
                )
            )
    if show_foci:
        for focus in _focus_mobjects(
            tuple(
                (item.source_ref, _local_point(item.coordinates))
                for item in diagram.focus_points
            ),
            supporting_kind=construction.supporting_kind,
            tolerance=diagram.certification_tolerance,
            semantic_prefix=diagram.diagram_id,
        ):
            source_ref = focus.dandelin_metadata["semanticSourceRefs"][0]
            result.append(_semantic_wrapper("focus", source_ref, focus))
    return tuple(result)


def _section_plane_view(
    construction: DandelinConstruction3D,
    *,
    show_directrices: bool,
    show_foci: bool,
) -> tuple[Mobject, ...]:
    diagram = build_dandelin_section_plane_diagram(construction)
    patch = _display_patch(
        construction,
        include_directrices=show_directrices,
    )
    bounds = _bounds_from_patch(patch)
    result: list[Mobject] = [
        _semantic_wrapper(
            "section_curve",
            f"{construction.construction_id}:section",
            *_section_curve_mobjects(diagram, spatial_matrix=None),
        )
    ]
    if show_directrices:
        directrix_by_source = {
            item.directrix_id: item for item in construction.directrices
        }
        for line in diagram.directrices:
            start, end = _clip_infinite_line(
                line.point_coordinates,
                line.direction_coordinates,
                bounds,
            )
            source = directrix_by_source[line.source_ref]
            result.append(
                _semantic_wrapper(
                    "directrix",
                    line.source_ref,
                    _line_mobject(
                        _local_point(start),
                        _local_point(end),
                        color=_CLASSROOM["directrix"],
                        width=1.6,
                        semantic_kind="directrix",
                        semantic_id=line.line_id,
                        source_refs=(line.source_ref,),
                        sphereId=source.sphere_id,
                    ),
                )
            )
    if show_foci:
        for focus in _focus_mobjects(
            tuple(
                (item.source_ref, _local_point(item.coordinates))
                for item in diagram.focus_points
            ),
            supporting_kind=diagram.supporting_kind,
            tolerance=diagram.certification_tolerance,
            semantic_prefix=diagram.diagram_id,
        ):
            source_ref = focus.dandelin_metadata["semanticSourceRefs"][0]
            result.append(_semantic_wrapper("focus", source_ref, focus))
    return tuple(result)


def _construction_source_refs(
    construction: DandelinConstruction3D,
) -> tuple[str, ...]:
    values = [
        construction.construction_id,
        construction.cone.surface_id,
        construction.plane.plane_id,
    ]
    for record in construction.spheres:
        values.extend(
            (
                record.sphere_id,
                record.focus_id,
                record.cone_contact_circle.curve_id,
            )
        )
        if record.directrix is not None:
            values.append(record.directrix.directrix_id)
    return _source_refs(values)


def _bind_semantic_plan(
    construction: DandelinConstruction3D,
    objects: Sequence[Mobject],
    *,
    view: str,
    show_contact_circles: bool,
    show_directrices: bool,
    show_foci: bool,
) -> tuple[Mobject, ...]:
    plan = build_dandelin_semantic_plan(
        construction,
        view=view,
        show_contact_circles=show_contact_circles,
        show_directrices=show_directrices,
        show_foci=show_foci,
    )
    pending: dict[tuple[str, str], deque[object]] = defaultdict(deque)
    for record in plan:
        pending[(record.role, record.source_ref)].append(record)

    result: list[Mobject] = []
    for item in objects:
        role = getattr(item, "_dandelin_plan_role", None)
        source_ref = getattr(item, "_dandelin_plan_source_ref", None)
        if not isinstance(role, str) or not isinstance(source_ref, str):
            raise DandelinFixedViewError(
                "fixed Dandelin renderer produced an unindexed top-level object"
            )
        queue = pending.get((role, source_ref))
        if not queue:
            raise DandelinFixedViewError(
                "fixed Dandelin renderer produced an object absent from the "
                "canonical semantic plan"
            )
        record = queue.popleft()
        _tag(
            item,
            semantic_kind=record.role,
            semantic_id=record.object_id,
            source_refs=(record.source_ref,),
            canonicalSemanticObject=True,
            renderPrimitiveCount=len(item.submobjects),
        )
        item.semantic_role = record.role
        item.source_ref = record.source_ref
        result.append(item)

    leftovers = tuple(
        record.object_id
        for queue in pending.values()
        for record in queue
    )
    if leftovers:
        raise DandelinFixedViewError(
            "fixed Dandelin renderer omitted canonical semantic objects: "
            + ", ".join(leftovers)
        )
    if len(result) != len(plan):
        raise DandelinFixedViewError(
            "fixed Dandelin semantic plan and renderer object counts disagree"
        )
    return tuple(result)


def _assign_depth_aware_z_indices(objects: Sequence[Mobject]) -> None:
    counters = {
        "hidden": 0,
        "surface": 0,
        "visible": 0,
        "focus": 0,
    }
    bases = {
        "hidden": 10.0,
        "surface": 100.0,
        "visible": 200.0,
        "focus": 300.0,
    }

    def assign(item: Mobject) -> None:
        metadata = getattr(item, "dandelin_metadata", None)
        if isinstance(metadata, dict):
            intent = metadata.get("renderIntent")
            kind = metadata.get("semanticKind")
            if intent == "dashed":
                layer = "hidden"
            elif intent == "solid" and metadata.get(
                "curveVisibilityAuthoritative"
            ) is True:
                layer = "visible"
            elif kind == "focus":
                layer = "focus"
            elif metadata.get("canonicalSemanticObject") is not True:
                layer = "surface"
            else:
                layer = None
            if layer is not None:
                item.set_z_index(bases[layer] + counters[layer], family=True)
                counters[layer] += 1
                return
        item.set_z_index(0.0, family=False)
        for child in item.submobjects:
            assign(child)

    for item in objects:
        assign(item)


def _finalize_group(
    construction: DandelinConstruction3D,
    objects: Sequence[Mobject],
    *,
    view: str,
    preset: str,
    show_contact_circles: bool,
    show_directrices: bool,
    show_foci: bool,
    mode: str,
    visibility_frame: DandelinVisibilityFrame | None = None,
) -> VGroup:
    if not objects:
        raise DandelinFixedViewError("fixed Dandelin view contains no display objects")
    objects = _bind_semantic_plan(
        construction,
        objects,
        view=view,
        show_contact_circles=show_contact_circles,
        show_directrices=show_directrices,
        show_foci=show_foci,
    )
    if mode == "depth_aware_diagrammatic":
        if visibility_frame is None:
            raise DandelinFixedViewError(
                "depth-aware Dandelin view has no visibility frame"
            )
        _assign_depth_aware_z_indices(objects)
    else:
        if visibility_frame is not None:
            raise DandelinFixedViewError(
                "legacy diagrammatic Dandelin view cannot retain a visibility frame"
            )
        for index, item in enumerate(objects):
            item.set_z_index(index)
    group = VGroup(*objects)
    points = group.get_all_points()
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise DandelinFixedViewError("fixed Dandelin view has non-finite Manim bounds")
    if not len(points):
        raise DandelinFixedViewError("fixed Dandelin view has no Manim points")
    lower = np.min(points[:, :2], axis=0)
    upper = np.max(points[:, :2], axis=0)
    span = upper - lower
    scale = max(1.0, float(np.max(np.abs(points[:, :2]))))
    if np.any(span <= _RELATIVE_RANK_TOLERANCE * scale):
        raise DandelinFixedViewError("fixed Dandelin view bounds are degenerate")
    source_bounds = (
        float(lower[0]),
        float(lower[1]),
        float(upper[0]),
        float(upper[1]),
    )
    center = 0.5 * (lower + upper)
    translation = np.asarray((-center[0], -center[1], 0.0), dtype=float)
    group.shift(translation)
    centered_points = group.get_all_points()
    centered_lower = np.min(centered_points[:, :2], axis=0)
    centered_upper = np.max(centered_points[:, :2], axis=0)
    bounds = (
        float(centered_lower[0]),
        float(centered_lower[1]),
        float(centered_upper[0]),
        float(centered_upper[1]),
    )
    metadata: dict[str, object] = {
        "semanticKind": "dandelin_fixed_view",
        "semanticId": f"{construction.construction_id}:fixed-view:{view}",
        "semanticSourceRefs": _construction_source_refs(construction),
        "view": view,
        "preset": preset,
        "mode": mode,
        "visibilityAuthoritative": False,
        "curveVisibilityAuthoritative": visibility_frame is not None,
        "surfaceVisibilityAuthoritative": False,
        "family": construction.family.value,
        "supportingKind": construction.supporting_kind.value,
        "showContactCircles": show_contact_circles,
        "showDirectrices": show_directrices,
        "showFoci": show_foci,
        "finiteBounds": bounds,
        "sourceBounds": source_bounds,
        "displayTranslation": (float(translation[0]), float(translation[1])),
        "sourceCoordinateUnits": True,
        "objectCount": len(objects),
        "sectionPlaneSphereCircles": False,
    }
    if visibility_frame is not None:
        metadata.update(
            {
                "visibilityFrameSchema": visibility_frame.schema,
                "hiddenSpanCount": visibility_frame.hidden_span_count,
                "tangentContactCount": len(visibility_frame.tangent_contacts),
            }
        )
    group.metadata = metadata
    group.dandelin_metadata = metadata
    group.view = view
    group.mode = mode
    group.visibility_authoritative = False
    group.curve_visibility_authoritative = visibility_frame is not None
    group.surface_visibility_authoritative = False
    group.visibility_frame = visibility_frame
    group.semantic_source_refs = metadata["semanticSourceRefs"]
    return group


def build_dandelin_fixed_view(
    construction: DandelinConstruction3D,
    *,
    view: str,
    projection_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
    preset: str = "classroom",
    mode: str = _DEFAULT_MODE,
    show_contact_circles: bool | None = None,
    show_directrices: bool | None = None,
    show_foci: bool | None = None,
) -> VGroup:
    """Build one finite static Dandelin teaching view in source units.

    ``projection_matrix`` is required only for ``view="spatial"``.  It is a
    finite 3x3 parallel-view matrix whose first two independent rows define the
    screen chart. ``mode="depth_aware_diagrammatic"`` is available only for
    the spatial view; it certifies hidden-line visibility while keeping
    translucent fill order diagrammatic.  The function never mutates a
    ``Scene``.

    Optional visibility flags use view-specific defaults when left as ``None``:
    spatial enables all three, meridian enables contact circles and foci, and
    section-plane enables directrices and foci.  Asking a meridian view for
    directrices or a section-plane view for contact circles fails closed because
    those objects do not belong to the respective two-dimensional contract.
    """

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    if not isinstance(view, str) or view not in _VIEWS:
        raise DandelinFixedViewError(
            "view must be 'spatial', 'meridian', or 'section-plane'"
        )
    if not isinstance(preset, str) or preset not in _PRESETS:
        raise DandelinFixedViewError("preset must be 'classroom'")
    if not isinstance(mode, str) or mode not in _MODES:
        raise DandelinFixedViewError(
            "mode must be 'diagrammatic' or 'depth_aware_diagrammatic'"
        )
    if mode == "depth_aware_diagrammatic" and view != "spatial":
        raise DandelinFixedViewError(
            "depth_aware_diagrammatic mode is only valid for the spatial view"
        )
    contact_default, directrix_default, focus_default = _VIEW_FLAG_DEFAULTS[view]
    contact_flag = _resolved_flag(
        show_contact_circles,
        "show_contact_circles",
        default=contact_default,
    )
    directrix_flag = _resolved_flag(
        show_directrices,
        "show_directrices",
        default=directrix_default,
    )
    focus_flag = _resolved_flag(
        show_foci,
        "show_foci",
        default=focus_default,
    )
    if view == "section-plane" and contact_flag:
        raise DandelinFixedViewError(
            "section-plane view cannot show cone contact circles because they "
            "do not lie in the authored cutting plane"
        )
    if view == "meridian" and directrix_flag:
        raise DandelinFixedViewError(
            "meridian view cannot show section-plane directrix lines"
        )
    if view == "spatial":
        if projection_matrix is None:
            raise DandelinFixedViewError(
                "spatial view requires projection_matrix"
            )
        matrix = _projection_matrix(projection_matrix)
    else:
        if projection_matrix is not None:
            raise DandelinFixedViewError(
                "projection_matrix is only valid for the spatial view"
            )
        matrix = None

    visibility_frame: DandelinVisibilityFrame | None = None
    try:
        if view == "spatial":
            assert matrix is not None
            if mode == "depth_aware_diagrammatic":
                objects, visibility_frame = _depth_aware_spatial_view(
                    construction,
                    matrix,
                    show_contact_circles=contact_flag,
                    show_directrices=directrix_flag,
                    show_foci=focus_flag,
                )
            else:
                objects = _diagrammatic_spatial_view(
                    construction,
                    matrix,
                    show_contact_circles=contact_flag,
                    show_directrices=directrix_flag,
                    show_foci=focus_flag,
                )
        elif view == "meridian":
            objects = _meridian_view(
                construction,
                show_contact_circles=contact_flag,
                show_foci=focus_flag,
            )
        else:
            objects = _section_plane_view(
                construction,
                show_directrices=directrix_flag,
                show_foci=focus_flag,
            )
    except DandelinFixedViewError:
        raise
    except (
        DandelinView2DError,
        PlanePatchFitError,
        PlanarCurveProjectionError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ) as exc:
        raise DandelinFixedViewError(
            f"{view} Dandelin fixed view cannot be certified: {exc}"
        ) from exc
    return _finalize_group(
        construction,
        objects,
        view=view,
        preset=preset,
        show_contact_circles=contact_flag,
        show_directrices=directrix_flag,
        show_foci=focus_flag,
        mode=mode,
        visibility_frame=visibility_frame,
    )


__all__ = [
    "DandelinFixedViewError",
    "build_dandelin_fixed_view",
]
