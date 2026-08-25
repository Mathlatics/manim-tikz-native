"""Renderer-neutral placement of semantic boundaries against a section plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import acos, atan2, atanh, ceil, floor, isfinite, pi, tau
from typing import Mapping, Sequence

import numpy as np
from numpy.polynomial import Polynomial

from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from ..parallel_solver import ParallelView
from ..topology import (
    ParameterInterval,
    assert_exact_partition,
    partition_parameter_domain,
)
from ..visibility import VisibilityKind
from .boundary_compositing import (
    BoundarySourceKind,
    QuadricBoundaryCompositingError,
    QuadricBoundarySource,
    QuadricBoundaryVisibilitySpan,
)
from .contract import (
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from .conics import ConicKind
from .critical import _curve_chart
from .curve_intersections import ProjectedCurveCrossing
from .curves import EllipseArcCurve, ParametricConicBranch, SegmentCurve
from .roots import (
    PolynomialRootError,
    solve_real_polynomial,
    solve_real_polynomial_exp_chart,
)
from .section_compositing import (
    PlaneDepthRole,
    QuadricSectionCompositingFrame,
    QuadricSectionCompositingError,
    quadric_plane_fragment_contours,
)


ContextInput = GeometryContext | ResolvedGeometryContext | None
QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec


@dataclass(frozen=True, slots=True)
class QuadricBoundarySectionLimits:
    """Explicit complexity bounds for boundary/plane-role partitioning."""

    max_role_boundary_segments: int = 32768
    max_split_parameters_per_source: int = 8192

    def __post_init__(self) -> None:
        for name in (
            "max_role_boundary_segments",
            "max_split_parameters_per_source",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


QUADRIC_BOUNDARY_SECTION_LIMITS = QuadricBoundarySectionLimits()


class BoundaryPlaneRelation(str, Enum):
    OUTSIDE_PATCH = "outside_patch"
    BOUNDARY_BEHIND_PLANE = "boundary_behind_plane"
    COINCIDENT = "coincident"
    BOUNDARY_IN_FRONT_OF_PLANE = "boundary_in_front_of_plane"


@dataclass(frozen=True, slots=True)
class QuadricBoundarySectionSpan:
    interval: ParameterInterval
    relation: BoundaryPlaneRelation
    plane_depth_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.interval, ParameterInterval)
            or self.interval.length <= 0.0
        ):
            raise QuadricBoundaryCompositingError(
                "boundary section interval must have positive length"
            )
        if not isinstance(self.relation, BoundaryPlaneRelation):
            raise TypeError("relation must be a BoundaryPlaneRelation")
        roles = tuple(str(item) for item in self.plane_depth_roles)
        valid = {item.value for item in PlaneDepthRole}
        if roles != tuple(sorted(set(roles))) or any(
            role not in valid for role in roles
        ):
            raise QuadricBoundaryCompositingError(
                "boundary plane-depth roles must be unique, sorted, and valid"
            )
        if self.relation is BoundaryPlaneRelation.OUTSIDE_PATCH:
            if roles:
                raise QuadricBoundaryCompositingError(
                    "outside-patch boundary span cannot carry plane roles"
                )
        elif not roles:
            raise QuadricBoundaryCompositingError(
                "boundary span inside the patch requires adjacent PlaneDepthRoles"
            )
        object.__setattr__(self, "plane_depth_roles", roles)

    def to_dict(self) -> dict[str, object]:
        return {
            "interval": [self.interval.start, self.interval.end],
            "relation": self.relation.value,
            "planeDepthRoles": list(self.plane_depth_roles),
        }


def _resolve_context(
    sources: Sequence[QuadricBoundarySource],
    plane: SectionPlane,
    patch: PlaneDisplayPatchSpec,
    context: ContextInput,
) -> ResolvedGeometryContext:
    if isinstance(context, ResolvedGeometryContext):
        return resolve_geometry_context(context)
    positions = list(patch.corners(plane))
    for source in sources:
        domain = source.curve.domain
        positions.extend(
            source.curve.point(value)
            for value in (domain.start, domain.midpoint, domain.end)
        )
    return resolve_geometry_context(context, positions=positions)


def _plane_intersection_parameters(
    source: QuadricBoundarySource,
    plane: SectionPlane,
    context: ResolvedGeometryContext,
) -> tuple[float, ...]:
    chart = _curve_chart(source.curve)
    normal = np.asarray(plane.normal, dtype=float)
    offset = float(np.dot(normal, np.asarray(plane.point, dtype=float)))
    polynomial = Polynomial((0.0,))
    for component, numerator in zip(normal, chart.numerator):
        polynomial = polynomial + float(component) * numerator
    polynomial = polynomial - offset * chart.denominator
    coefficients = tuple(float(item) for item in polynomial.coef)
    scale = max(
        *(abs(item) for item in coefficients),
        chart.homogeneous_scale * max(1.0, abs(offset)),
        np.finfo(float).tiny,
    )
    identity_tolerance = 2048.0 * np.finfo(float).eps * scale
    if max((abs(item) for item in coefficients), default=0.0) <= identity_tolerance:
        return ()
    parameter_epsilon = context.epsilon(GeometryQuantity.PARAMETER)
    try:
        if chart.name == "exp":
            roots = solve_real_polynomial_exp_chart(
                coefficients,
                parameter_domain=source.curve.domain,
                context=context,
                parameter_tolerance=parameter_epsilon,
            )
            values = tuple(item.parameter for item in roots)
        else:
            roots = solve_real_polynomial(
                coefficients,
                domain=chart.root_domain,
                context=context,
                parameter_tolerance=parameter_epsilon,
            )
            values = tuple(
                parameter
                for root in roots
                for parameter in chart.parameters(root.value, parameter_epsilon)
            )
    except PolynomialRootError as exc:
        raise QuadricBoundaryCompositingError(
            f"boundary {source.source_id!r} cannot isolate section-plane "
            f"intersections: {exc}"
        ) from exc
    result: list[float] = []
    for value in sorted(float(item) for item in values):
        if not source.curve.domain.contains(value, tolerance=parameter_epsilon):
            continue
        value = min(source.curve.domain.end, max(source.curve.domain.start, value))
        if not result or value - result[-1] > parameter_epsilon:
            result.append(value)
    return tuple(result)


def _polynomial_is_identically_zero(
    polynomial: Polynomial,
    reference_scale: float,
) -> bool:
    coefficients = tuple(float(item) for item in polynomial.coef)
    scale = max(
        *(abs(item) for item in coefficients),
        abs(float(reference_scale)),
        np.finfo(float).tiny,
    )
    tolerance = 2048.0 * np.finfo(float).eps * scale
    return max((abs(item) for item in coefficients), default=0.0) <= tolerance


def _curve_lies_on_section_surface(
    source: QuadricBoundarySource,
    surface: QuadricSurfaceSpec,
    plane: SectionPlane,
    context: ResolvedGeometryContext,
) -> bool:
    """Certify a finite source as part of the exact plane/surface section."""

    chart = _curve_chart(source.curve)
    normal = np.asarray(plane.normal, dtype=float)
    offset = float(np.dot(normal, np.asarray(plane.point, dtype=float)))
    plane_polynomial = Polynomial((0.0,))
    for component, numerator in zip(normal, chart.numerator):
        plane_polynomial = plane_polynomial + float(component) * numerator
    plane_polynomial = plane_polynomial - offset * chart.denominator
    if not _polynomial_is_identically_zero(
        plane_polynomial,
        chart.homogeneous_scale * max(1.0, abs(offset)),
    ):
        return False

    homogeneous = (*chart.numerator, chart.denominator)
    matrix = np.asarray(surface.support_quadric.matrix, dtype=float)
    surface_polynomial = Polynomial((0.0,))
    for row in range(4):
        for column in range(4):
            coefficient = float(matrix[row, column])
            if coefficient != 0.0:
                surface_polynomial = (
                    surface_polynomial
                    + coefficient * homogeneous[row] * homogeneous[column]
                )
    if not _polynomial_is_identically_zero(
        surface_polynomial,
        chart.homogeneous_scale
        * chart.homogeneous_scale
        * max(float(np.max(np.abs(matrix))), np.finfo(float).tiny),
    ):
        return False

    if isinstance(surface, SphereSpec):
        return True

    curve = source.curve
    domain = curve.domain
    candidates = [domain.start, domain.end]
    axis = np.asarray(surface.frame.z_axis, dtype=float)
    surface_origin = np.asarray(surface.frame.origin, dtype=float)

    if isinstance(curve, SegmentCurve):
        pass
    elif isinstance(curve, EllipseArcCurve):
        first = np.asarray(curve.first_axis, dtype=float)
        second = np.asarray(curve.second_axis, dtype=float)
        first_axial = float(np.dot(first, axis))
        second_axial = float(np.dot(second, axis))
        if np.hypot(first_axial, second_axial) > 0.0:
            base = atan2(second_axial, first_axial)
            candidates.extend(
                parameter
                for angle in (base, base + pi)
                for parameter in _angular_domain_parameters(
                    source,
                    angle,
                    context.epsilon(GeometryQuantity.PARAMETER),
                    )
                )
    else:
        _origin, first, second = _parametric_world_geometry(curve)
        branch = curve.parameterization
        first_axial = float(np.dot(first, axis))
        second_axial = float(np.dot(second, axis))
        if branch.kind in {ConicKind.CIRCLE, ConicKind.ELLIPSE}:
            if np.hypot(first_axial, second_axial) > 0.0:
                base = atan2(second_axial, first_axial)
                candidates.extend(
                    parameter
                    for angle in (base, base + pi)
                    for parameter in _angular_domain_parameters(
                        source,
                        angle,
                        context.epsilon(GeometryQuantity.PARAMETER),
                    )
                )
        elif branch.kind is ConicKind.HYPERBOLA:
            cosine = branch.branch_sign * first_axial
            if cosine != 0.0:
                ratio = -second_axial / cosine
                if abs(ratio) < 1.0:
                    candidates.append(atanh(ratio))
        elif branch.kind is ConicKind.PARABOLA and second_axial != 0.0:
            candidates.append(-first_axial / (2.0 * second_axial))

    parameter_epsilon = context.epsilon(GeometryQuantity.PARAMETER)
    axial_values = tuple(
        float(
            np.dot(
                np.asarray(curve.point(parameter), dtype=float) - surface_origin,
                axis,
            )
        )
        for parameter in candidates
        if domain.contains(parameter, tolerance=parameter_epsilon)
    )
    boundary_epsilon = context.epsilon(GeometryQuantity.BOUNDARY)
    lower, upper = surface.axial_range
    return (
        bool(axial_values)
        and min(axial_values) >= lower - boundary_epsilon
        and max(axial_values) <= upper + boundary_epsilon
    )


def _visibility_kind_at(
    source_id: str,
    parameter: float,
    spans_by_source: Mapping[
        str, Sequence[QuadricBoundaryVisibilitySpan]
    ],
    parameter_epsilon: float,
) -> VisibilityKind:
    matches = tuple(
        span.kind
        for span in spans_by_source.get(source_id, ())
        if span.interval.contains(parameter, tolerance=parameter_epsilon)
    )
    kinds = set(matches)
    if len(kinds) != 1:
        raise QuadricBoundaryCompositingError(
            f"boundary {source_id!r} has no unique visibility kind at "
            f"parameter {parameter:.17g}"
        )
    return next(iter(kinds))


def _chart_polynomial_parameters(
    source: QuadricBoundarySource,
    chart: object,
    polynomial: Polynomial,
    context: ResolvedGeometryContext,
    *,
    label: str,
) -> tuple[float, ...] | None:
    """Solve one authored-curve chart equation.

    ``None`` means the equation vanishes identically on the projected curve;
    callers must then isolate the finite overlap boundaries instead of
    pretending that there are no critical events.
    """

    coefficients = tuple(float(item) for item in polynomial.coef)
    scale = max(
        *(abs(item) for item in coefficients),
        chart.homogeneous_scale,
        np.finfo(float).tiny,
    )
    identity_tolerance = 2048.0 * np.finfo(float).eps * scale
    if max((abs(item) for item in coefficients), default=0.0) <= identity_tolerance:
        return None
    parameter_epsilon = context.epsilon(GeometryQuantity.PARAMETER)
    try:
        if chart.name == "exp":
            roots = solve_real_polynomial_exp_chart(
                coefficients,
                parameter_domain=source.curve.domain,
                context=context,
                parameter_tolerance=parameter_epsilon,
            )
            values = tuple(item.parameter for item in roots)
        else:
            roots = solve_real_polynomial(
                coefficients,
                domain=chart.root_domain,
                context=context,
                parameter_tolerance=parameter_epsilon,
            )
            values = tuple(
                parameter
                for root in roots
                for parameter in chart.parameters(root.value, parameter_epsilon)
            )
    except PolynomialRootError as exc:
        raise QuadricBoundaryCompositingError(
            f"boundary {source.source_id!r} cannot isolate {label}: {exc}"
        ) from exc
    result: list[float] = []
    for value in sorted(float(item) for item in values):
        if not source.curve.domain.contains(value, tolerance=parameter_epsilon):
            continue
        value = min(source.curve.domain.end, max(source.curve.domain.start, value))
        if not result or value - result[-1] > parameter_epsilon:
            result.append(value)
    return tuple(result)


def _projected_chart_polynomial(
    source: QuadricBoundarySource,
    view: ParallelView,
    direction: np.ndarray,
    offset: float,
) -> tuple[object, Polynomial]:
    chart = _curve_chart(source.curve)
    screen = view.matrix[:2]
    x = sum(
        (
            float(screen[0, axis]) * chart.numerator[axis]
            for axis in range(3)
        ),
        Polynomial((0.0,)),
    )
    y = sum(
        (
            float(screen[1, axis]) * chart.numerator[axis]
            for axis in range(3)
        ),
        Polynomial((0.0,)),
    )
    polynomial = (
        float(direction[0]) * x
        + float(direction[1]) * y
        + float(offset) * chart.denominator
    )
    return chart, polynomial


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _segment_screen_parameters(
    source: QuadricBoundarySource,
    start: np.ndarray,
    end: np.ndarray,
    view: ParallelView,
    screen_epsilon: float,
) -> tuple[float, ...]:
    curve = source.curve
    if not isinstance(curve, SegmentCurve):  # pragma: no cover - private caller
        return ()
    first = view.matrix[:2] @ np.asarray(curve.start, dtype=float)
    last = view.matrix[:2] @ np.asarray(curve.end, dtype=float)
    source_delta = last - first
    target_delta = end - start
    source_length = float(np.linalg.norm(source_delta))
    target_length = float(np.linalg.norm(target_delta))
    if source_length <= screen_epsilon:
        return ()
    determinant = _cross2(source_delta, target_delta)
    ratios: list[float] = []
    offset = start - first
    if abs(determinant) > screen_epsilon * max(source_length, target_length):
        source_ratio = _cross2(offset, target_delta) / determinant
        target_ratio = _cross2(offset, source_delta) / determinant
        source_tolerance = screen_epsilon / source_length
        target_tolerance = screen_epsilon / target_length
        if (
            -source_tolerance <= source_ratio <= 1.0 + source_tolerance
            and -target_tolerance <= target_ratio <= 1.0 + target_tolerance
        ):
            ratios.append(min(1.0, max(0.0, source_ratio)))
    elif abs(_cross2(offset, source_delta)) <= screen_epsilon * source_length:
        denominator = float(np.dot(source_delta, source_delta))
        target_ratios = (
            float(np.dot(start - first, source_delta) / denominator),
            float(np.dot(end - first, source_delta) / denominator),
        )
        lower = max(0.0, min(target_ratios))
        upper = min(1.0, max(target_ratios))
        tolerance = screen_epsilon / source_length
        if upper >= lower - tolerance:
            ratios.extend((lower, upper))
    parameter_epsilon = max(
        np.finfo(float).eps,
        screen_epsilon / source_length * curve.domain.length,
    )
    result: list[float] = []
    for ratio in sorted(ratios):
        parameter = curve.domain.start + ratio * curve.domain.length
        if not result or parameter - result[-1] > parameter_epsilon:
            result.append(float(parameter))
    return tuple(result)


def _parametric_world_geometry(
    curve: ParametricConicBranch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch = curve.parameterization
    embedding = np.asarray(curve.plane_embedding, dtype=float)
    linear = embedding[:3, :2]
    origin = embedding[:3, 2] + linear @ np.asarray(branch.origin, dtype=float)
    first = linear @ np.asarray(branch.first_axis, dtype=float)
    second = linear @ np.asarray(branch.second_axis, dtype=float)
    return origin, first, second


def _ellipse_screen_geometry(
    source: QuadricBoundarySource,
    view: ParallelView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    curve = source.curve
    if isinstance(curve, EllipseArcCurve):
        world = (
            np.asarray(curve.center, dtype=float),
            np.asarray(curve.first_axis, dtype=float),
            np.asarray(curve.second_axis, dtype=float),
        )
    elif (
        isinstance(curve, ParametricConicBranch)
        and curve.parameterization.kind in {ConicKind.CIRCLE, ConicKind.ELLIPSE}
    ):
        world = _parametric_world_geometry(curve)
    else:
        return None
    projection = view.matrix[:2]
    return (
        projection @ world[0],
        projection @ world[1],
        projection @ world[2],
    )


def _angular_domain_parameters(
    source: QuadricBoundarySource,
    base: float,
    parameter_epsilon: float,
) -> tuple[float, ...]:
    curve = source.curve
    lower = floor((curve.domain.start - base) / tau) - 1
    upper = ceil((curve.domain.end - base) / tau) + 1
    values: list[float] = []
    for index in range(lower, upper + 1):
        candidate = base + index * tau
        if candidate < curve.domain.start - parameter_epsilon:
            continue
        if candidate > curve.domain.end + parameter_epsilon:
            continue
        candidate = min(curve.domain.end, max(curve.domain.start, candidate))
        if not values or candidate - values[-1] > parameter_epsilon:
            values.append(float(candidate))
    if (
        getattr(curve, "closed", False)
        and len(values) == 2
        and abs(values[0] - curve.domain.start) <= parameter_epsilon
        and abs(values[1] - curve.domain.end) <= parameter_epsilon
    ):
        values.pop()
    return tuple(values)


def _trigonometric_parameters(
    source: QuadricBoundarySource,
    first: float,
    second: float,
    constant: float,
    screen_epsilon: float,
    parameter_epsilon: float,
) -> tuple[float, ...] | None:
    amplitude = float(np.hypot(first, second))
    if amplitude <= screen_epsilon:
        return None if abs(constant) <= screen_epsilon else ()
    ratio = -constant / amplitude
    tolerance = screen_epsilon / amplitude
    if ratio < -1.0 - tolerance or ratio > 1.0 + tolerance:
        return ()
    ratio = min(1.0, max(-1.0, ratio))
    phase = atan2(second, first)
    offset = acos(ratio)
    values = tuple(
        parameter
        for base in (phase - offset, phase + offset)
        for parameter in _angular_domain_parameters(
            source,
            base,
            parameter_epsilon,
        )
    )
    result: list[float] = []
    for value in sorted(values):
        if not result or value - result[-1] > parameter_epsilon:
            result.append(value)
    return tuple(result)


def _projected_segment_intersection_parameters(
    source: QuadricBoundarySource,
    screen_start: Sequence[float],
    screen_end: Sequence[float],
    view: ParallelView,
    context: ResolvedGeometryContext,
) -> tuple[float, ...]:
    """Return every source parameter on one finite screen-space segment.

    The equation is solved in the source curve's analytic chart.  This remains
    valid when a circle is viewed edge-on and projects to a line segment.  If
    the complete projected support is collinear, intersections with the two
    finite segment endpoints isolate the overlap boundaries.
    """

    start = np.asarray(screen_start, dtype=float)
    end = np.asarray(screen_end, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    screen_epsilon = context.epsilon(GeometryQuantity.SCREEN)
    if length <= screen_epsilon:
        return ()
    if isinstance(source.curve, SegmentCurve):
        return _segment_screen_parameters(
            source,
            start,
            end,
            view,
            screen_epsilon,
        )
    tangent = delta / length
    normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
    parameter_epsilon = context.epsilon(GeometryQuantity.PARAMETER)
    ellipse = _ellipse_screen_geometry(source, view)
    chart = None
    if ellipse is not None:
        origin, first_axis, second_axis = ellipse
        support_roots = _trigonometric_parameters(
            source,
            float(np.dot(normal, first_axis)),
            float(np.dot(normal, second_axis)),
            float(np.dot(normal, origin - start)),
            screen_epsilon,
            parameter_epsilon,
        )
    else:
        chart, support_polynomial = _projected_chart_polynomial(
            source,
            view,
            normal,
            -float(np.dot(normal, start)),
        )
        support_roots = _chart_polynomial_parameters(
            source,
            chart,
            support_polynomial,
            context,
            label="a plane-role boundary crossing",
        )
    candidates: list[float] = []
    if support_roots is None:
        for scalar in (0.0, length):
            if ellipse is not None:
                origin, first_axis, second_axis = ellipse
                endpoint_roots = _trigonometric_parameters(
                    source,
                    float(np.dot(tangent, first_axis)),
                    float(np.dot(tangent, second_axis)),
                    float(np.dot(tangent, origin - start)) - scalar,
                    screen_epsilon,
                    parameter_epsilon,
                )
            else:
                endpoint_chart, endpoint_polynomial = _projected_chart_polynomial(
                    source,
                    view,
                    tangent,
                    -float(np.dot(tangent, start)) - scalar,
                )
                endpoint_roots = _chart_polynomial_parameters(
                    source,
                    endpoint_chart,
                    endpoint_polynomial,
                    context,
                    label="a collinear plane-role boundary endpoint",
                )
            if endpoint_roots is not None:
                candidates.extend(endpoint_roots)
    else:
        candidates.extend(support_roots)

    # The tan-half-angle chart represents its pole at infinity.  Check that
    # authored seam, plus finite authored endpoints, directly in screen space.
    if chart is not None:
        candidates.extend(chart.chart_poles)
    candidates.extend((source.curve.domain.start, source.curve.domain.end))
    projection = view.matrix[:2]
    result: list[float] = []
    for value in sorted(float(item) for item in candidates):
        if not source.curve.domain.contains(value, tolerance=parameter_epsilon):
            continue
        value = min(source.curve.domain.end, max(source.curve.domain.start, value))
        screen = projection @ np.asarray(source.curve.point(value), dtype=float)
        perpendicular = abs(float(np.dot(screen - start, normal)))
        along = float(np.dot(screen - start, tangent))
        if perpendicular > screen_epsilon:
            continue
        if along < -screen_epsilon or along > length + screen_epsilon:
            continue
        if not result or value - result[-1] > parameter_epsilon:
            result.append(value)
    return tuple(result)


def _projected_vertex_parameters(
    source: QuadricBoundarySource,
    screen_point: Sequence[float],
    view: ParallelView,
    context: ResolvedGeometryContext,
) -> tuple[float, ...]:
    """Recover source parameters at one canonical plane-role vertex."""

    target = np.asarray(screen_point, dtype=float)
    projection = view.matrix[:2]
    screen_epsilon = context.epsilon(GeometryQuantity.SCREEN)
    parameter_epsilon = context.epsilon(GeometryQuantity.PARAMETER)
    curve = source.curve
    candidates: list[float] = []
    if isinstance(curve, SegmentCurve):
        first = projection @ np.asarray(curve.start, dtype=float)
        last = projection @ np.asarray(curve.end, dtype=float)
        delta = last - first
        denominator = float(np.dot(delta, delta))
        if denominator > screen_epsilon * screen_epsilon:
            ratio = float(np.dot(target - first, delta) / denominator)
            tolerance = screen_epsilon / float(np.sqrt(denominator))
            if -tolerance <= ratio <= 1.0 + tolerance:
                candidates.append(
                    curve.domain.start
                    + min(1.0, max(0.0, ratio)) * curve.domain.length
                )
    else:
        ellipse = _ellipse_screen_geometry(source, view)
        if ellipse is not None:
            origin, first_axis, second_axis = ellipse
            linear = np.column_stack((first_axis, second_axis))
            singular = np.linalg.svd(linear, compute_uv=False)
            maximum = float(np.max(singular))
            minimum = float(np.min(singular))
            if minimum > 1024.0 * np.finfo(float).eps * maximum:
                coordinates = np.linalg.solve(linear, target - origin)
                reconstruction = origin + linear @ coordinates
                local_epsilon = screen_epsilon / minimum
                if (
                    float(np.linalg.norm(reconstruction - target))
                    <= 8.0 * screen_epsilon
                    and abs(float(np.dot(coordinates, coordinates)) - 1.0)
                    <= 8.0 * local_epsilon
                ):
                    candidates.extend(
                        _angular_domain_parameters(
                            source,
                            atan2(float(coordinates[1]), float(coordinates[0])),
                            parameter_epsilon,
                        )
                    )
            elif maximum > screen_epsilon:
                left, _singular, _right = np.linalg.svd(
                    linear, full_matrices=False
                )
                axis = np.asarray(left[:, 0], dtype=float)
                if abs(_cross2(axis, target - origin)) <= 8.0 * screen_epsilon:
                    roots = _trigonometric_parameters(
                        source,
                        float(np.dot(first_axis, axis)),
                        float(np.dot(second_axis, axis)),
                        float(np.dot(origin - target, axis)),
                        screen_epsilon,
                        parameter_epsilon,
                    )
                    if roots is not None:
                        candidates.extend(roots)
        else:
            chart = _curve_chart(curve)
            for axis in range(2):
                direction = np.zeros(2, dtype=float)
                direction[axis] = 1.0
                point_chart, polynomial = _projected_chart_polynomial(
                    source,
                    view,
                    direction,
                    -float(target[axis]),
                )
                roots = _chart_polynomial_parameters(
                    source,
                    point_chart,
                    polynomial,
                    context,
                    label="a plane-role boundary vertex",
                )
                if roots is not None:
                    candidates.extend(roots)
            candidates.extend(chart.chart_poles)

    candidates.extend((curve.domain.start, curve.domain.end))
    result: list[float] = []
    for value in sorted(float(item) for item in candidates):
        if not curve.domain.contains(value, tolerance=parameter_epsilon):
            continue
        value = min(curve.domain.end, max(curve.domain.start, value))
        point = projection @ np.asarray(curve.point(value), dtype=float)
        if float(np.linalg.norm(point - target)) > 8.0 * screen_epsilon:
            continue
        if not result or value - result[-1] > parameter_epsilon:
            result.append(value)
    return tuple(result)


def _role_boundary_segments(
    frame: QuadricSectionCompositingFrame,
    screen_epsilon: float,
    limits: QuadricBoundarySectionLimits,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    try:
        contours = quadric_plane_fragment_contours(frame)
    except QuadricSectionCompositingError as exc:
        raise QuadricBoundaryCompositingError(
            f"cannot extract plane-role boundaries: {exc}"
        ) from exc
    by_key: dict[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ] = {}
    for role in PlaneDepthRole:
        for loop in contours[role]:
            if len(loop) < 2:
                raise QuadricBoundaryCompositingError(
                    "plane-role contour must contain at least two vertices"
                )
            points = tuple(tuple(float(item) for item in point) for point in loop)
            for start, end in zip(points, (*points[1:], points[0])):
                if float(
                    np.linalg.norm(
                        np.asarray(end, dtype=float)
                        - np.asarray(start, dtype=float)
                    )
                ) <= screen_epsilon:
                    continue
                key = (start, end) if start < end else (end, start)
                by_key.setdefault(key, key)
                if len(by_key) > limits.max_role_boundary_segments:
                    raise QuadricBoundaryCompositingError(
                        "plane-role boundary count exceeds "
                        f"max_role_boundary_segments={limits.max_role_boundary_segments}"
                    )
    return tuple(by_key[key] for key in sorted(by_key))


@dataclass(frozen=True, slots=True)
class _PlaneRoleLocator:
    triangles: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    orientations: np.ndarray
    edge_lengths: np.ndarray
    roles: tuple[PlaneDepthRole, ...]

    @classmethod
    def from_frame(
        cls, frame: QuadricSectionCompositingFrame
    ) -> "_PlaneRoleLocator":
        triangles = np.asarray(
            [fragment.screen_vertices for fragment in frame.plane_fragments],
            dtype=float,
        )
        edges = np.roll(triangles, -1, axis=1) - triangles
        area = (
            (triangles[:, 1, 0] - triangles[:, 0, 0])
            * (triangles[:, 2, 1] - triangles[:, 0, 1])
            - (triangles[:, 1, 1] - triangles[:, 0, 1])
            * (triangles[:, 2, 0] - triangles[:, 0, 0])
        )
        return cls(
            triangles,
            np.min(triangles, axis=1),
            np.max(triangles, axis=1),
            np.where(area >= 0.0, 1.0, -1.0),
            np.linalg.norm(edges, axis=2),
            tuple(fragment.role for fragment in frame.plane_fragments),
        )

    def roles_at(
        self,
        screen_point: np.ndarray,
        screen_epsilon: float,
    ) -> tuple[PlaneDepthRole, ...]:
        point = np.asarray(screen_point, dtype=float)
        candidates = np.flatnonzero(
            np.all(point >= self.minimum - screen_epsilon, axis=1)
            & np.all(point <= self.maximum + screen_epsilon, axis=1)
        )
        if len(candidates):
            triangles = self.triangles[candidates]
            edges = np.roll(triangles, -1, axis=1) - triangles
            offsets = point[np.newaxis, np.newaxis, :] - triangles
            sides = self.orientations[candidates, np.newaxis] * (
                edges[:, :, 0] * offsets[:, :, 1]
                - edges[:, :, 1] * offsets[:, :, 0]
            )
            thresholds = -screen_epsilon * np.maximum(
                self.edge_lengths[candidates], screen_epsilon
            )
            candidates = candidates[np.all(sides >= thresholds, axis=1)]
        roles = {self.roles[index] for index in candidates}
        if not roles:
            raise QuadricBoundaryCompositingError(
                "boundary section midpoint is not covered by the plane partition"
            )
        return tuple(sorted(roles, key=lambda item: item.value))


def _edge_crossing_parameters(
    sources: Sequence[QuadricBoundarySource],
    crossings: Sequence[ProjectedCurveCrossing],
) -> dict[str, list[float]]:
    source_map = {item.source_id: item for item in sources}
    result: dict[str, list[float]] = {item.source_id: [] for item in sources}
    for crossing in crossings:
        first = source_map[crossing.first_curve_id]
        second = source_map[crossing.second_curve_id]
        if first.source_kind is BoundarySourceKind.PLANE_PATCH_EDGE:
            result[second.source_id].append(crossing.second_parameter)
        if second.source_kind is BoundarySourceKind.PLANE_PATCH_EDGE:
            result[first.source_id].append(crossing.first_parameter)
    return result


def _coalesce(
    spans: Sequence[QuadricBoundarySectionSpan],
    tolerance: float,
) -> tuple[QuadricBoundarySectionSpan, ...]:
    result: list[QuadricBoundarySectionSpan] = []
    for span in spans:
        if (
            result
            and result[-1].relation is span.relation
            and result[-1].plane_depth_roles == span.plane_depth_roles
            and abs(result[-1].interval.end - span.interval.start) <= tolerance
        ):
            previous = result[-1]
            result[-1] = QuadricBoundarySectionSpan(
                ParameterInterval(previous.interval.start, span.interval.end),
                span.relation,
                span.plane_depth_roles,
            )
        else:
            result.append(span)
    return tuple(result)


def compute_boundary_section_spans(
    sources: Sequence[QuadricBoundarySource],
    section_frame: QuadricSectionCompositingFrame,
    view: ParallelView,
    crossings: Sequence[ProjectedCurveCrossing] = (),
    *,
    surface: QuadricSurfaceSpec | None = None,
    visibility_spans_by_source: Mapping[
        str, Sequence[QuadricBoundaryVisibilitySpan]
    ] | None = None,
    context: ContextInput = None,
    limits: QuadricBoundarySectionLimits = QUADRIC_BOUNDARY_SECTION_LIMITS,
) -> dict[str, tuple[QuadricBoundarySectionSpan, ...]]:
    """Split semantic curves wherever their order against the finite plane changes."""

    if not isinstance(section_frame, QuadricSectionCompositingFrame):
        raise TypeError("section_frame must be a QuadricSectionCompositingFrame")
    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    if surface is not None and not isinstance(
        surface, (SphereSpec, CylinderSpec, ConeSpec)
    ):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    if not isinstance(limits, QuadricBoundarySectionLimits):
        raise TypeError("limits must be a QuadricBoundarySectionLimits")
    visibility_spans = (
        {} if visibility_spans_by_source is None else visibility_spans_by_source
    )
    if not isinstance(visibility_spans, Mapping):
        raise TypeError("visibility_spans_by_source must be a mapping")
    source_items = tuple(sorted(sources, key=lambda item: item.source_id))
    plane = section_frame.plane
    patch = section_frame.patch
    resolved = _resolve_context(source_items, plane, patch, context)
    parameter_epsilon = resolved.epsilon(GeometryQuantity.PARAMETER)
    depth_epsilon = resolved.epsilon(GeometryQuantity.DEPTH)
    screen_epsilon = resolved.epsilon(GeometryQuantity.SCREEN)
    plane_u, plane_v, _plane_normal = plane.basis
    screen_origin = view.matrix[:2] @ np.asarray(plane.point, dtype=float)
    screen_basis = np.column_stack(
        (view.matrix[:2] @ plane_u, view.matrix[:2] @ plane_v)
    )
    determinant = float(np.linalg.det(screen_basis))
    scale = max(float(np.linalg.norm(screen_basis, ord=2)), np.finfo(float).tiny)
    if abs(determinant) <= 1.0e-12 * scale * scale:
        raise QuadricBoundaryCompositingError(
            "section plane projects edge-on during boundary placement"
        )
    inverse = np.linalg.inv(screen_basis)
    direction = np.asarray(view.view_direction, dtype=float)
    edge_parameters = _edge_crossing_parameters(source_items, crossings)
    role_segments = _role_boundary_segments(section_frame, screen_epsilon, limits)
    role_vertices = tuple(
        sorted({point for segment in role_segments for point in segment})
    )
    role_locator = _PlaneRoleLocator.from_frame(section_frame)
    result: dict[str, tuple[QuadricBoundarySectionSpan, ...]] = {}

    for source in source_items:
        if source.source_kind is BoundarySourceKind.PLANE_PATCH_EDGE:
            continue
        is_section_boundary = (
            surface is not None
            and source.source_id in visibility_spans
            and _curve_lies_on_section_surface(
                source,
                surface,
                plane,
                resolved,
            )
        )
        roots = _plane_intersection_parameters(source, plane, resolved)
        split_parameters = [*roots, *edge_parameters[source.source_id]]
        if is_section_boundary:
            split_parameters.extend(
                endpoint
                for span in visibility_spans[source.source_id]
                for endpoint in (span.interval.start, span.interval.end)
            )
        else:
            for vertex in role_vertices:
                split_parameters.extend(
                    _projected_vertex_parameters(
                        source,
                        vertex,
                        view,
                        resolved,
                    )
                )
            for start, end in role_segments:
                split_parameters.extend(
                    _projected_segment_intersection_parameters(
                        source,
                        start,
                        end,
                        view,
                        resolved,
                    )
                )
        canonical_parameters: list[float] = []
        canonical_screen_points: list[np.ndarray] = []
        for value in sorted(float(item) for item in split_parameters):
            if not source.curve.domain.contains(
                value, tolerance=parameter_epsilon
            ):
                continue
            value = min(
                source.curve.domain.end,
                max(source.curve.domain.start, value),
            )
            screen_point = view.matrix[:2] @ np.asarray(
                source.curve.point(value), dtype=float
            )
            if canonical_parameters and (
                value - canonical_parameters[-1] <= parameter_epsilon
                or float(
                    np.linalg.norm(screen_point - canonical_screen_points[-1])
                )
                <= 8.0 * screen_epsilon
            ):
                continue
            canonical_parameters.append(value)
            canonical_screen_points.append(screen_point)
        if len(canonical_parameters) > limits.max_split_parameters_per_source:
            raise QuadricBoundaryCompositingError(
                f"boundary {source.source_id!r} exceeds "
                "max_split_parameters_per_source="
                f"{limits.max_split_parameters_per_source}"
            )
        cells = partition_parameter_domain(
            source.curve.domain,
            canonical_parameters,
            tolerance=parameter_epsilon,
        )

        def classify(parameter: float) -> tuple[BoundaryPlaneRelation, tuple[str, ...]]:
            world = np.asarray(source.curve.point(parameter), dtype=float)
            screen = view.matrix[:2] @ world
            coordinates = inverse @ (screen - screen_origin)
            if not patch.contains_coordinates(coordinates, context=resolved):
                return BoundaryPlaneRelation.OUTSIDE_PATCH, ()
            plane_world = plane.point_from_coordinates(coordinates)
            depth_difference = float(np.dot(world - plane_world, direction))
            if depth_difference > depth_epsilon:
                relation = BoundaryPlaneRelation.BOUNDARY_IN_FRONT_OF_PLANE
            elif depth_difference < -depth_epsilon:
                relation = BoundaryPlaneRelation.BOUNDARY_BEHIND_PLANE
            else:
                relation = BoundaryPlaneRelation.COINCIDENT
            if is_section_boundary:
                if relation is not BoundaryPlaneRelation.COINCIDENT:
                    raise QuadricBoundaryCompositingError(
                        f"boundary {source.source_id!r} left its certified "
                        "section plane"
                    )
                visibility = _visibility_kind_at(
                    source.source_id,
                    parameter,
                    visibility_spans,
                    parameter_epsilon,
                )
                roles = (
                    (
                        PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                        PlaneDepthRole.IN_FRONT_OF_SURFACE,
                    )
                    if visibility is VisibilityKind.VISIBLE
                    else (
                        PlaneDepthRole.BEHIND_SURFACE,
                        PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                    )
                )
            else:
                roles = role_locator.roles_at(screen, screen_epsilon)
            return relation, tuple(role.value for role in roles)

        spans: list[QuadricBoundarySectionSpan] = []
        for cell in cells:
            states = tuple(
                classify(cell.start + fraction * cell.length)
                for fraction in (0.25, 0.5, 0.75)
            )
            if states[0] != states[1] or states[1] != states[2]:
                raise QuadricBoundaryCompositingError(
                    f"boundary {source.source_id!r} retains mixed section roles "
                    f"inside interval [{cell.start:.17g}, {cell.end:.17g}]: "
                    + ", ".join(
                        f"{relation.value}/{roles}" for relation, roles in states
                    )
                )
            relation, roles = states[1]
            spans.append(
                QuadricBoundarySectionSpan(
                    cell,
                    relation,
                    roles,
                )
            )
        coalesced = _coalesce(spans, parameter_epsilon)
        assert_exact_partition(
            source.curve.domain,
            (item.interval for item in coalesced),
            tolerance=parameter_epsilon,
        )
        result[source.source_id] = coalesced
    return result


__all__ = [
    "BoundaryPlaneRelation",
    "QUADRIC_BOUNDARY_SECTION_LIMITS",
    "QuadricBoundarySectionLimits",
    "QuadricBoundarySectionSpan",
    "compute_boundary_section_spans",
]
