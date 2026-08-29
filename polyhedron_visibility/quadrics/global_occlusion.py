"""Global painter ordering for a finite set of disjoint convex quadrics.

This module joins the existing renderer-neutral quadric layers into one frame
builder:

* analytic support mappings and a bounded GJK separation certificate reject
  touching, intersecting, non-convex, or numerically ambiguous solid pairs;
* exact projected support mappings decide whether two silhouettes are
  definitely disjoint;
* adaptive projection proxies locate a deterministic interior overlap domain;
* finite-solid ``ray_hits`` provide the authoritative front/back depth order;
* the established visibility and compositing modules build the final curve
  spans and far-to-near painter graph.

The projection proxy is therefore never geometric truth.  It only supplies a
bounded display-domain witness location after analytic convex geometry has
validated the entities themselves.  The supported scene boundary is narrow on
purpose: pairwise-disjoint spheres, capped finite cylinders, and one-nappe
finite cones/frusta under a parallel view.  Anything else fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from ..compositor import PainterConstraint
from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from ..parallel_solver import ParallelView
from ..style import OcclusionStyle
from .compositing import (
    QuadricCompositingError,
    QuadricCompositingFrame,
    QuadricPaintPolicy,
    compute_quadric_compositing,
)
from .contract import ConeModel, ConeSpec, CylinderSpec, SphereSpec
from .critical import AnalyticCurve3D
from .curve_intersections import (
    ProjectedCurveIntersectionError,
    compute_projected_curve_crossings,
)
from .projection import (
    OpaqueProjectionProxy,
    build_opaque_projection_proxy,
)
from .visibility import compute_quadric_visibility


GLOBAL_QUADRIC_FRAME_SCHEMA = "manim-quadric-global-occlusion-frame/v1"
_DEFAULT_GJK_ITERATIONS = 64
_DEFAULT_PROXY_RELATIVE_ERROR = 1.0e-3


QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec
ContextInput = GeometryContext | ResolvedGeometryContext | None
StyleInput = Mapping[str, OcclusionStyle] | OcclusionStyle | None
ConstraintInput = PainterConstraint[str] | tuple[str, str]


class GlobalQuadricOcclusionError(ValueError):
    """A global quadric frame cannot be certified without guessing."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GlobalQuadricOcclusionError(f"{label} must be a non-empty string")
    return value.strip()


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise GlobalQuadricOcclusionError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GlobalQuadricOcclusionError(
            f"{label} must be finite and positive"
        ) from exc
    if not isfinite(result) or result <= 0.0:
        raise GlobalQuadricOcclusionError(f"{label} must be finite and positive")
    return result


def _iteration_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 4:
        raise GlobalQuadricOcclusionError(
            "gjk_max_iterations must be an integer of at least four"
        )
    return value


def _surface_center(surface: QuadricSurfaceSpec) -> np.ndarray:
    if isinstance(surface, SphereSpec):
        return np.asarray(surface.center, dtype=float)
    frame = surface.frame
    base = np.asarray(
        surface.origin if isinstance(surface, CylinderSpec) else surface.apex,
        dtype=float,
    )
    midpoint = 0.5 * (surface.axial_range[0] + surface.axial_range[1])
    return base + midpoint * np.asarray(frame.z_axis, dtype=float)


def _surface_local_scale(
    surface: QuadricSurfaceSpec,
    anchor: np.ndarray,
) -> float:
    """Return a translation-independent conditioning scale for one solid."""

    distances = tuple(
        float(np.linalg.norm(np.asarray(point, dtype=float) - anchor))
        for point in surface.characteristic_points
    )
    scale = max(distances, default=0.0)
    if not isfinite(scale) or scale <= 0.0:
        raise GlobalQuadricOcclusionError(
            f"surface {surface.surface_id!r} has no finite local conditioning scale"
        )
    return scale


def _localized_surface(
    surface: QuadricSurfaceSpec,
    anchor: np.ndarray,
    scale: float,
) -> QuadricSurfaceSpec:
    """Express a finite solid in a nearby, unit-scale coordinate system.

    Ray parameters in this local system are multiplied by ``scale`` before
    they are compared.  The operation is a similarity transform, so it does
    not change containment, cap filtering, or front/back order.
    """

    def local_point(value: Sequence[float]) -> tuple[float, float, float]:
        result = (np.asarray(value, dtype=float) - anchor) / scale
        return tuple(float(item) for item in result)  # type: ignore[return-value]

    if isinstance(surface, SphereSpec):
        return SphereSpec(
            surface.surface_id,
            local_point(surface.center),
            surface.radius / scale,
        )
    axial_range = tuple(float(value / scale) for value in surface.axial_range)
    if isinstance(surface, CylinderSpec):
        return CylinderSpec(
            surface.surface_id,
            local_point(surface.origin),
            surface.axis,
            surface.radius / scale,
            axial_range,
            radial_axis=surface.radial_axis,
        )
    return ConeSpec(
        surface.surface_id,
        local_point(surface.apex),
        surface.axis,
        surface.half_angle,
        axial_range,
        radial_axis=surface.radial_axis,
        model=surface.model,
        component_parent_id=surface.component_parent_id,
    )


def _screen_epsilon(
    context: ResolvedGeometryContext,
    screen_matrix: np.ndarray,
) -> float:
    """Resolve a tolerance in the actual units of the two screen rows."""

    projection_norm = float(np.linalg.norm(screen_matrix, ord=2))
    if not isfinite(projection_norm) or projection_norm <= 0.0:
        raise GlobalQuadricOcclusionError(
            "parallel view has no usable screen projection scale"
        )
    projected = max(
        context.epsilon(GeometryQuantity.BOUNDARY) * projection_norm,
        context.policy.absolute_floor * projection_norm,
    )
    if GeometryQuantity.SCREEN in context.overrides:
        return max(projected, context.epsilon(GeometryQuantity.SCREEN))
    if context.screen != context.resolved.world:
        return max(projected, context.screen)
    return projected


def _validate_convex_surface(surface: QuadricSurfaceSpec) -> None:
    if not isinstance(surface, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surfaces must contain sphere, cylinder, or cone specs")
    if isinstance(surface, ConeSpec):
        lower, upper = surface.axial_range
        if lower < 0.0 < upper:
            raise GlobalQuadricOcclusionError(
                f"surface {surface.surface_id!r} crosses the cone apex and has "
                "two nappes; split it into convex one-nappe entities"
            )


def _validated_surfaces(
    surfaces: Sequence[QuadricSurfaceSpec],
) -> tuple[QuadricSurfaceSpec, ...]:
    result = tuple(surfaces)
    for surface in result:
        _validate_convex_surface(surface)
    identities = tuple(surface.surface_id for surface in result)
    if len(set(identities)) != len(identities):
        raise GlobalQuadricOcclusionError("surface identities must be unique")
    return tuple(sorted(result, key=lambda surface: surface.surface_id))


def _shared_open_double_components(
    first: QuadricSurfaceSpec,
    second: QuadricSurfaceSpec,
) -> bool:
    if not isinstance(first, ConeSpec) or not isinstance(second, ConeSpec):
        return False
    parent_id = first.component_parent_id
    if (
        first.model is not ConeModel.OPEN_SINGLE
        or second.model is not ConeModel.OPEN_SINGLE
        or parent_id is None
        or second.component_parent_id != parent_id
    ):
        return False

    def component_role(surface: ConeSpec) -> str | None:
        lower, upper = surface.axial_range
        if (
            surface.surface_id == f"{parent_id}:nappe:negative"
            and lower < 0.0
            and upper == 0.0
        ):
            return "negative"
        if (
            surface.surface_id == f"{parent_id}:nappe:positive"
            and lower == 0.0
            and upper > 0.0
        ):
            return "positive"
        return None

    return (
        {component_role(first), component_role(second)} == {"negative", "positive"}
        and first.apex == second.apex
        and first.axis == second.axis
        and first.radial_axis == second.radial_axis
        and first.half_angle == second.half_angle
    )


def _resolve_context(
    surfaces: Sequence[QuadricSurfaceSpec],
    context: ContextInput,
    *,
    curves: Sequence[AnalyticCurve3D] = (),
) -> ResolvedGeometryContext:
    if isinstance(context, ResolvedGeometryContext):
        return resolve_geometry_context(context)
    points = tuple(
        point
        for surface in surfaces
        for point in surface.characteristic_points
    )
    def curve_probe(curve: AnalyticCurve3D, index: int) -> tuple[float, float, float]:
        # Reconstructing the upper endpoint as ``start + length`` can round one
        # ULP beyond ``domain.end`` for large or asymmetric conic parameters.
        # Preserve the authored endpoints exactly and interpolate only the
        # three interior probes.
        if index == 0:
            parameter = curve.domain.start
        elif index == 4:
            parameter = curve.domain.end
        else:
            ratio = index / 4.0
            parameter = (1.0 - ratio) * curve.domain.start + ratio * curve.domain.end
        return curve.point(parameter)

    curve_points = tuple(
        curve_probe(curve, index) for curve in curves for index in range(5)
    )
    return resolve_geometry_context(context, positions=(*points, *curve_points))


def _support_surface(
    surface: QuadricSurfaceSpec,
    direction: np.ndarray,
) -> np.ndarray:
    """Return an analytic support point of one finite convex solid."""

    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise GlobalQuadricOcclusionError(
            "support direction must contain three finite values"
        )
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise GlobalQuadricOcclusionError("support direction must be non-zero")
    if isinstance(surface, SphereSpec):
        return (
            np.asarray(surface.center, dtype=float)
            + surface.radius * vector / length
        )

    frame = surface.frame
    axis = np.asarray(frame.z_axis, dtype=float)
    x_axis = np.asarray(frame.x_axis, dtype=float)
    y_axis = np.asarray(frame.y_axis, dtype=float)
    first = float(np.dot(vector, x_axis))
    second = float(np.dot(vector, y_axis))
    radial_norm = float(np.hypot(first, second))
    radial_direction = (
        np.zeros(3, dtype=float)
        if radial_norm == 0.0
        else (first * x_axis + second * y_axis) / radial_norm
    )
    base = np.asarray(
        surface.origin if isinstance(surface, CylinderSpec) else surface.apex,
        dtype=float,
    )
    candidates: list[np.ndarray] = []
    for axial in surface.axial_range:
        radius = (
            surface.radius
            if isinstance(surface, CylinderSpec)
            else abs(axial) * surface.slope
        )
        candidates.append(base + axial * axis + radius * radial_direction)
    return max(candidates, key=lambda point: float(np.dot(vector, point)))


@dataclass(frozen=True, slots=True)
class _SupportVertex:
    difference: np.ndarray
    first: np.ndarray
    second: np.ndarray


def _minkowski_support(
    first: QuadricSurfaceSpec,
    second: QuadricSurfaceSpec,
    direction: np.ndarray,
) -> _SupportVertex:
    first_point = _support_surface(first, direction)
    second_point = _support_surface(second, -direction)
    return _SupportVertex(first_point - second_point, first_point, second_point)


def _closest_simplex(
    vertices: Sequence[_SupportVertex],
) -> tuple[np.ndarray, tuple[_SupportVertex, ...], tuple[float, ...]]:
    """Closest point to zero on a simplex, with deterministic active set.

    A GJK simplex contains at most four vertices.  Enumerating its non-empty
    faces avoids dimension-specific line/triangle/tetrahedron case code and is
    still a fixed, tiny amount of work.  Every candidate solves the affine
    barycentric least-squares system exactly once.
    """

    items = tuple(vertices)
    if not items or len(items) > 4:
        raise GlobalQuadricOcclusionError("internal GJK simplex is invalid")
    best: tuple[float, tuple[int, ...], np.ndarray, np.ndarray] | None = None
    barycentric_tolerance = 2048.0 * np.finfo(float).eps
    for count in range(1, len(items) + 1):
        for indices in combinations(range(len(items)), count):
            points = np.asarray([items[index].difference for index in indices])
            gram = points @ points.T
            system = np.block(
                [
                    [gram, np.ones((count, 1), dtype=float)],
                    [np.ones((1, count), dtype=float), np.zeros((1, 1))],
                ]
            )
            target = np.concatenate((np.zeros(count, dtype=float), (1.0,)))
            solution, *_ = np.linalg.lstsq(system, target, rcond=None)
            weights = solution[:count]
            if not np.all(np.isfinite(weights)) or float(np.min(weights)) < (
                -barycentric_tolerance
            ):
                continue
            weights = np.maximum(weights, 0.0)
            total = float(np.sum(weights))
            if total == 0.0:
                continue
            weights /= total
            closest = weights @ points
            squared = float(np.dot(closest, closest))
            candidate = (squared, indices, closest, weights)
            if best is None or (squared, indices) < (best[0], best[1]):
                best = candidate
    if best is None:
        raise GlobalQuadricOcclusionError(
            "GJK simplex became numerically indeterminate"
        )
    _, indices, closest, weights = best
    active = tuple(items[index] for index in indices)
    return closest, active, tuple(float(value) for value in weights)


@dataclass(frozen=True, slots=True)
class StrictSeparationEvidence:
    """One analytic support-plane certificate for a disjoint solid pair."""

    first_surface_id: str
    second_surface_id: str
    separating_direction: tuple[float, float, float]
    support_gap_lower_bound: float
    closest_distance_upper_bound: float
    first_support_point: tuple[float, float, float]
    second_support_point: tuple[float, float, float]
    iterations: int
    support_evaluations: int
    method: str = "bounded_gjk_support_certificate"

    def to_dict(self) -> dict[str, object]:
        return {
            "firstSurfaceId": self.first_surface_id,
            "secondSurfaceId": self.second_surface_id,
            "method": self.method,
            "separatingDirection": list(self.separating_direction),
            "supportGapLowerBound": self.support_gap_lower_bound,
            "closestDistanceUpperBound": self.closest_distance_upper_bound,
            "firstSupportPoint": list(self.first_support_point),
            "secondSupportPoint": list(self.second_support_point),
            "iterations": self.iterations,
            "supportEvaluations": self.support_evaluations,
        }


def _strict_separation_certificate(
    first: QuadricSurfaceSpec,
    second: QuadricSurfaceSpec,
    *,
    epsilon: float,
    max_iterations: int,
) -> StrictSeparationEvidence | None:
    """Return a strict support-plane certificate, or ``None`` fail-closed."""

    direction = _surface_center(first) - _surface_center(second)
    if float(np.linalg.norm(direction)) <= epsilon:
        direction = np.asarray((1.0, 0.0, 0.0), dtype=float)
    simplex = [_minkowski_support(first, second, direction)]
    evaluations = 1
    previous_squared = float("inf")

    for iteration in range(1, max_iterations + 1):
        closest, active, _weights = _closest_simplex(simplex)
        simplex = list(active)
        distance = float(np.linalg.norm(closest))
        if not isfinite(distance) or distance <= epsilon:
            return None

        search = -closest
        vertex = _minkowski_support(first, second, search)
        evaluations += 1
        # ``vertex`` minimizes closest·x over the complete Minkowski
        # difference.  A positive minimum is a global separating plane.
        gap = float(np.dot(closest, vertex.difference) / distance)
        if gap > epsilon:
            unit = closest / distance
            return StrictSeparationEvidence(
                first.surface_id,
                second.surface_id,
                tuple(float(value) for value in unit),
                gap,
                distance,
                tuple(float(value) for value in vertex.first),
                tuple(float(value) for value in vertex.second),
                iteration,
                evaluations,
            )

        duplicate_epsilon = max(epsilon, 64.0 * np.finfo(float).eps * distance)
        if any(
            float(np.linalg.norm(vertex.difference - item.difference))
            <= duplicate_epsilon
            for item in simplex
        ):
            return None
        simplex.append(vertex)
        updated, _, _ = _closest_simplex(simplex)
        squared = float(np.dot(updated, updated))
        progress = previous_squared - squared
        progress_floor = max(
            epsilon * epsilon,
            256.0 * np.finfo(float).eps * max(previous_squared, squared, 1.0e-300),
        )
        if previous_squared < float("inf") and progress <= progress_floor:
            return None
        previous_squared = squared
    return None


def verify_strict_quadric_separation(
    surfaces: Sequence[QuadricSurfaceSpec],
    *,
    context: ContextInput = None,
    gjk_max_iterations: int = _DEFAULT_GJK_ITERATIONS,
) -> tuple[StrictSeparationEvidence, ...]:
    """Certify every pair as strictly disjoint without dense sampling.

    Contact, intersection, a two-nappe cone, lack of GJK convergence, and any
    tolerance-level ambiguity all raise :class:`GlobalQuadricOcclusionError`.
    """

    items = _validated_surfaces(surfaces)
    resolved = _resolve_context(items, context)
    iterations = _iteration_limit(gjk_max_iterations)
    epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    result: list[StrictSeparationEvidence] = []
    for first, second in combinations(items, 2):
        evidence = _strict_separation_certificate(
            first,
            second,
            epsilon=epsilon,
            max_iterations=iterations,
        )
        if evidence is None:
            raise GlobalQuadricOcclusionError(
                "finite convex entities are touching, intersecting, or "
                "numerically inseparable: "
                f"{first.surface_id!r}, {second.surface_id!r}"
            )
        result.append(evidence)
    return tuple(result)


def _verify_render_component_separation(
    surfaces: Sequence[QuadricSurfaceSpec],
    *,
    context: ContextInput,
    gjk_max_iterations: int,
) -> tuple[StrictSeparationEvidence, ...]:
    """Certify unrelated entities while allowing one authored shared apex."""

    items = _validated_surfaces(surfaces)
    resolved = _resolve_context(items, context)
    iterations = _iteration_limit(gjk_max_iterations)
    epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    result: list[StrictSeparationEvidence] = []
    for first, second in combinations(items, 2):
        if _shared_open_double_components(first, second):
            continue
        evidence = _strict_separation_certificate(
            first,
            second,
            epsilon=epsilon,
            max_iterations=iterations,
        )
        if evidence is None:
            raise GlobalQuadricOcclusionError(
                "finite convex entities are touching, intersecting, or "
                "numerically inseparable: "
                f"{first.surface_id!r}, {second.surface_id!r}"
            )
        result.append(evidence)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _ProjectedSeparation:
    gap: float
    direction: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _ProjectedVertex:
    difference: np.ndarray
    first: np.ndarray
    second: np.ndarray


def _projected_support(
    surface: QuadricSurfaceSpec,
    screen_matrix: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    covector = screen_matrix.T @ direction
    return screen_matrix @ _support_surface(surface, covector)


def _projected_separation_certificate(
    first: QuadricSurfaceSpec,
    second: QuadricSurfaceSpec,
    screen_matrix: np.ndarray,
    *,
    epsilon: float,
    max_iterations: int,
) -> _ProjectedSeparation | None:
    """Bounded 2D GJK support-plane certificate for disjoint silhouettes."""

    def support(direction: np.ndarray) -> _ProjectedVertex:
        first_point = _projected_support(first, screen_matrix, direction)
        second_point = _projected_support(second, screen_matrix, -direction)
        return _ProjectedVertex(
            first_point - second_point,
            first_point,
            second_point,
        )

    direction = screen_matrix @ (_surface_center(first) - _surface_center(second))
    if float(np.linalg.norm(direction)) <= epsilon:
        direction = np.asarray((1.0, 0.0), dtype=float)
    simplex: list[_ProjectedVertex] = [support(direction)]
    previous_squared = float("inf")
    barycentric_tolerance = 2048.0 * np.finfo(float).eps

    def closest_simplex(
        vertices: Sequence[_ProjectedVertex],
    ) -> tuple[np.ndarray, tuple[_ProjectedVertex, ...]]:
        items = tuple(vertices)
        best: tuple[float, tuple[int, ...], np.ndarray] | None = None
        for count in range(1, len(items) + 1):
            for indices in combinations(range(len(items)), count):
                points = np.asarray([items[index].difference for index in indices])
                gram = points @ points.T
                system = np.block(
                    [
                        [gram, np.ones((count, 1), dtype=float)],
                        [np.ones((1, count), dtype=float), np.zeros((1, 1))],
                    ]
                )
                target = np.concatenate((np.zeros(count), (1.0,)))
                solution, *_ = np.linalg.lstsq(system, target, rcond=None)
                weights = solution[:count]
                if float(np.min(weights)) < -barycentric_tolerance:
                    continue
                weights = np.maximum(weights, 0.0)
                weights /= float(np.sum(weights))
                point = weights @ points
                squared = float(np.dot(point, point))
                candidate = (squared, indices, point)
                if best is None or (squared, indices) < (best[0], best[1]):
                    best = candidate
        if best is None:
            raise GlobalQuadricOcclusionError(
                "projected GJK simplex became numerically indeterminate"
            )
        return best[2], tuple(items[index] for index in best[1])

    for _ in range(max_iterations):
        closest, active = closest_simplex(simplex)
        simplex = list(active)
        distance = float(np.linalg.norm(closest))
        if distance <= epsilon:
            return None
        vertex = support(-closest)
        gap = float(np.dot(closest, vertex.difference) / distance)
        if gap > epsilon:
            unit = closest / distance
            return _ProjectedSeparation(gap, (float(unit[0]), float(unit[1])))
        duplicate_epsilon = max(epsilon, 64.0 * np.finfo(float).eps * distance)
        if any(
            float(np.linalg.norm(vertex.difference - item.difference))
            <= duplicate_epsilon
            for item in simplex
        ):
            return None
        simplex.append(vertex)
        updated, _ = closest_simplex(simplex)
        squared = float(np.dot(updated, updated))
        progress_floor = max(
            epsilon * epsilon,
            256.0 * np.finfo(float).eps * max(previous_squared, squared, 1.0e-300),
        )
        if previous_squared < float("inf") and previous_squared - squared <= progress_floor:
            return None
        previous_squared = squared
    return None


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _convex_polygon_intersection(
    subject: Sequence[tuple[float, float]],
    clip: Sequence[tuple[float, float]],
    *,
    epsilon: float,
) -> tuple[tuple[float, float], ...]:
    """Sutherland-Hodgman intersection of two counter-clockwise proxies."""

    output = [np.asarray(point, dtype=float) for point in subject]
    clip_points = [np.asarray(point, dtype=float) for point in clip]
    for clip_start, clip_end in zip(
        clip_points,
        (*clip_points[1:], clip_points[0]),
    ):
        if not output:
            break
        edge = clip_end - clip_start
        source = output
        output = []
        boundary_threshold = epsilon * max(
            float(np.linalg.norm(edge)),
            epsilon,
        )
        for start, end in zip(source, (*source[1:], source[0])):
            start_value = _cross2(edge, start - clip_start)
            end_value = _cross2(edge, end - clip_start)
            start_adjusted = start_value + boundary_threshold
            end_adjusted = end_value + boundary_threshold
            start_inside = start_adjusted >= 0.0
            end_inside = end_adjusted >= 0.0
            if start_inside != end_inside:
                denominator = start_adjusted - end_adjusted
                if abs(denominator) <= np.finfo(float).tiny:
                    raise GlobalQuadricOcclusionError(
                        "projection overlap clipping became numerically ambiguous"
                    )
                ratio = start_adjusted / denominator
                output.append(start + ratio * (end - start))
            if end_inside:
                output.append(end)

    deduped: list[np.ndarray] = []
    for point in output:
        if not deduped or float(np.linalg.norm(point - deduped[-1])) > epsilon:
            deduped.append(point)
    if len(deduped) > 1 and float(np.linalg.norm(deduped[-1] - deduped[0])) <= epsilon:
        deduped.pop()
    return tuple((float(point[0]), float(point[1])) for point in deduped)


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    vectors = [np.asarray(point, dtype=float) for point in points]
    # Area is translation invariant.  Local coordinates avoid catastrophic
    # cancellation when the complete scene has a large common offset.
    reference = vectors[0]
    vectors = [point - reference for point in vectors]
    return 0.5 * abs(
        sum(_cross2(first, second) for first, second in zip(vectors, (*vectors[1:], vectors[0])))
    )


def _polygon_centroid(points: Sequence[tuple[float, float]]) -> np.ndarray:
    vectors = [np.asarray(point, dtype=float) for point in points]
    if not vectors:
        raise GlobalQuadricOcclusionError(
            "projection overlap has no stable interior centroid"
        )
    reference = np.mean(vectors, axis=0)
    local = [point - reference for point in vectors]
    signed_twice = sum(
        _cross2(first, second)
        for first, second in zip(local, (*local[1:], local[0]))
    )
    if abs(signed_twice) <= np.finfo(float).tiny:
        raise GlobalQuadricOcclusionError(
            "projection overlap has no stable interior centroid"
        )
    weighted = np.zeros(2, dtype=float)
    for first, second in zip(local, (*local[1:], local[0])):
        cross = _cross2(first, second)
        weighted += (first + second) * cross
    return reference + weighted / (3.0 * signed_twice)


def _interior_witnesses(
    polygon: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    center = _polygon_centroid(polygon)
    points = [np.asarray(point, dtype=float) for point in polygon]
    # Four topology-derived extreme vertices plus the centroid provide stable
    # diagnostic redundancy without dense sampling.
    extreme_indices = {
        min(range(len(points)), key=lambda index: (points[index][0], points[index][1])),
        max(range(len(points)), key=lambda index: (points[index][0], points[index][1])),
        min(range(len(points)), key=lambda index: (points[index][1], points[index][0])),
        max(range(len(points)), key=lambda index: (points[index][1], points[index][0])),
    }
    result = [center]
    result.extend(0.5 * (center + points[index]) for index in sorted(extreme_indices))
    return tuple((float(point[0]), float(point[1])) for point in result)


def _ray_interval(
    surface: QuadricSurfaceSpec,
    screen_point: tuple[float, float],
    view: ParallelView,
    *,
    context: ResolvedGeometryContext,
    depth_epsilon: float,
    scene_anchor: np.ndarray,
) -> tuple[float, float]:
    screen_matrix = view.matrix[:2]
    screen_target = np.asarray(screen_point, dtype=float)
    # A parallel ray only needs the two screen equations.  Solving the full
    # 3x3 view matrix wrongly turns a harmless tiny depth row into a singular
    # reconstruction problem.  Work relative to the scene, then take the
    # minimum-norm solution of the rank-two screen system.
    screen_delta = screen_target - screen_matrix @ scene_anchor
    left, singular_values, right = np.linalg.svd(
        screen_matrix,
        full_matrices=False,
    )
    if len(singular_values) != 2 or float(np.min(singular_values)) <= 0.0:
        raise GlobalQuadricOcclusionError(
            "parallel view screen rows are not independently solvable"
        )
    offset = right.T @ ((left.T @ screen_delta) / singular_values)
    origin = scene_anchor + offset
    residual = float(np.linalg.norm(screen_matrix @ origin - screen_target))
    residual_floor = 4096.0 * np.finfo(float).eps * max(
        float(np.linalg.norm(screen_target)),
        float(np.linalg.norm(screen_matrix @ scene_anchor)),
        float(np.linalg.norm(screen_delta)),
        np.finfo(float).tiny,
    )
    residual_floor = max(
        residual_floor,
        _screen_epsilon(context, screen_matrix),
    )
    if not isfinite(residual) or residual > residual_floor:
        raise GlobalQuadricOcclusionError(
            "parallel view screen ray reconstruction is numerically ambiguous"
        )

    # The finite-solid contract remains authoritative, but it is evaluated in
    # a surface-local unit frame.  This removes both large common translations
    # and 1e9-scale homogeneous coefficients without replacing the contract's
    # cap filtering or containment rules.
    surface_anchor = _surface_center(surface)
    local_scale = _surface_local_scale(surface, surface_anchor)
    local_surface = _localized_surface(surface, surface_anchor, local_scale)
    local_origin = (origin - surface_anchor) / local_scale
    local_overrides: dict[GeometryQuantity, float] = {}
    for quantity, value in context.overrides.items():
        local_overrides[quantity] = (
            value / local_scale
            if quantity
            in {
                GeometryQuantity.LENGTH,
                GeometryQuantity.BOUNDARY,
                GeometryQuantity.DEPTH,
            }
            else value
        )
    local_context = GeometryContext(
        tolerance=context.policy,
        overrides=local_overrides,
    ).resolve(local_surface.characteristic_points)
    # The homogeneous quadric is free to normalize its matrix by a constant.
    # A world-scale angular tolerance must therefore not be reused as a
    # dimensionless polynomial-coefficient cutoff (large-radius spheres would
    # otherwise look spuriously linear).  Keep the shared boundary/depth
    # tolerances, but resolve this low-degree analytic root classification at
    # machine precision.  The finite entity's own ``ray_hits`` remains the
    # authoritative source of intersections and cap filtering.
    ray_context = local_context.with_overrides(
        angular=128.0 * np.finfo(float).eps,
    )
    hits = local_surface.ray_hits(
        local_origin,
        view.view_direction,
        context=ray_context,
        include_caps=True,
        forward_only=False,
    )
    parameters: list[float] = []
    for hit in hits:
        world_parameter = hit.parameter * local_scale
        if not parameters or all(
            abs(world_parameter - existing) > depth_epsilon
            for existing in parameters
        ):
            parameters.append(world_parameter)
    parameters.sort()
    if len(parameters) < 2:
        raise GlobalQuadricOcclusionError(
            f"overlap witness ray did not cross closed surface {surface.surface_id!r} twice"
        )
    interval = (parameters[0], parameters[-1])
    local_midpoint = local_origin + (
        0.5 * (interval[0] + interval[1]) / local_scale
    ) * np.asarray(
        view.view_direction,
        dtype=float,
    )
    if not local_surface.contains(local_midpoint, context=local_context):
        raise GlobalQuadricOcclusionError(
            f"ray-hit endpoints for surface {surface.surface_id!r} do not bound its solid interior"
        )
    return interval


@dataclass(frozen=True, slots=True)
class SurfaceDepthWitness:
    screen_point: tuple[float, float]
    first_depth_interval: tuple[float, float]
    second_depth_interval: tuple[float, float]
    farther_surface_id: str
    nearer_surface_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "screenPoint": list(self.screen_point),
            "firstDepthInterval": list(self.first_depth_interval),
            "secondDepthInterval": list(self.second_depth_interval),
            "fartherSurfaceId": self.farther_surface_id,
            "nearerSurfaceId": self.nearer_surface_id,
        }


@dataclass(frozen=True, slots=True)
class SurfaceDepthEvidence:
    first_surface_id: str
    second_surface_id: str
    projection_relation: str
    projected_separation_gap: float | None
    overlap_polygon: tuple[tuple[float, float], ...]
    witnesses: tuple[SurfaceDepthWitness, ...]
    farther_surface_id: str | None
    nearer_surface_id: str | None
    proxy_visibility_authoritative: bool = False

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "firstSurfaceId": self.first_surface_id,
            "secondSurfaceId": self.second_surface_id,
            "projectionRelation": self.projection_relation,
            "overlapPolygon": [list(point) for point in self.overlap_polygon],
            "witnesses": [item.to_dict() for item in self.witnesses],
            "proxyVisibilityAuthoritative": self.proxy_visibility_authoritative,
        }
        if self.projected_separation_gap is not None:
            result["projectedSeparationGap"] = self.projected_separation_gap
        if self.farther_surface_id is not None:
            result["fartherSurfaceId"] = self.farther_surface_id
            result["nearerSurfaceId"] = self.nearer_surface_id
        return result


@dataclass(frozen=True, slots=True)
class SurfaceOrderConstraint:
    farther_surface_id: str
    nearer_surface_id: str
    reason: str

    def __post_init__(self) -> None:
        farther = _identity(self.farther_surface_id, "farther_surface_id")
        nearer = _identity(self.nearer_surface_id, "nearer_surface_id")
        if farther == nearer:
            raise GlobalQuadricOcclusionError(
                "a surface cannot be ordered against itself"
            )
        object.__setattr__(self, "farther_surface_id", farther)
        object.__setattr__(self, "nearer_surface_id", nearer)
        object.__setattr__(self, "reason", _identity(self.reason, "constraint reason"))

    def to_dict(self) -> dict[str, object]:
        return {
            "fartherSurfaceId": self.farther_surface_id,
            "nearerSurfaceId": self.nearer_surface_id,
            "reason": self.reason,
        }


def _automatic_surface_order(
    surfaces: tuple[QuadricSurfaceSpec, ...],
    proxies: tuple[OpaqueProjectionProxy, ...],
    view: ParallelView,
    *,
    context: ResolvedGeometryContext,
    gjk_max_iterations: int,
) -> tuple[tuple[SurfaceDepthEvidence, ...], tuple[SurfaceOrderConstraint, ...]]:
    by_surface = {surface.surface_id: surface for surface in surfaces}
    by_proxy = {proxy.surface_id: proxy for proxy in proxies}
    screen_matrix = view.matrix[:2]
    screen_epsilon = _screen_epsilon(context, screen_matrix)
    depth_epsilon = context.epsilon(GeometryQuantity.DEPTH)
    scene_anchor = (
        np.mean(
            np.asarray([_surface_center(surface) for surface in surfaces]),
            axis=0,
        )
        if surfaces
        else np.zeros(3, dtype=float)
    )
    evidence_items: list[SurfaceDepthEvidence] = []
    constraints: list[SurfaceOrderConstraint] = []

    for first_id, second_id in combinations(sorted(by_surface), 2):
        first = by_surface[first_id]
        second = by_surface[second_id]
        separated_projection = _projected_separation_certificate(
            first,
            second,
            screen_matrix,
            epsilon=screen_epsilon,
            max_iterations=gjk_max_iterations,
        )
        if separated_projection is not None:
            evidence_items.append(
                SurfaceDepthEvidence(
                    first_id,
                    second_id,
                    "disjoint",
                    separated_projection.gap,
                    (),
                    (),
                    None,
                    None,
                )
            )
            continue

        first_proxy = by_proxy[first_id]
        second_proxy = by_proxy[second_id]
        first_vertices = tuple(
            np.asarray(point, dtype=float) for point in first_proxy.vertices
        )
        second_vertices = tuple(
            np.asarray(point, dtype=float) for point in second_proxy.vertices
        )
        polygon_anchor = np.mean((*first_vertices, *second_vertices), axis=0)
        first_local = tuple(
            tuple(float(value) for value in point - polygon_anchor)
            for point in first_vertices
        )
        second_local = tuple(
            tuple(float(value) for value in point - polygon_anchor)
            for point in second_vertices
        )
        local_extent = max(
            (
                float(np.linalg.norm(point - polygon_anchor))
                for point in (*first_vertices, *second_vertices)
            ),
            default=0.0,
        )
        clipping_epsilon = max(
            screen_epsilon,
            64.0
            * np.finfo(float).eps
            * max(local_extent, np.finfo(float).tiny),
        )
        overlap_local = _convex_polygon_intersection(
            first_local,
            second_local,
            epsilon=clipping_epsilon,
        )
        area = _polygon_area(overlap_local)
        area_floor = max(
            clipping_epsilon * clipping_epsilon,
            (first_proxy.metadata.max_chord_error + second_proxy.metadata.max_chord_error)
            * clipping_epsilon,
        )
        shared_open_double_parent = _shared_open_double_components(first, second)
        if len(overlap_local) < 3 or area <= area_floor:
            if shared_open_double_parent:
                evidence_items.append(
                    SurfaceDepthEvidence(
                        first_id,
                        second_id,
                        "touching_open_double_nappes",
                        0.0,
                        (),
                        (),
                        None,
                        None,
                    )
                )
                continue
            raise GlobalQuadricOcclusionError(
                "projected silhouettes may touch or overlap, but the adaptive "
                "proxies do not contain a stable interior witness: "
                f"{first_id!r}, {second_id!r}"
            )
        if shared_open_double_parent:
            raise GlobalQuadricOcclusionError(
                "the two finite open-cone nappes have overlapping projected "
                "interiors in this view; the current whole-surface painter "
                "cannot certify their multi-sheet order"
            )

        witnesses: list[SurfaceDepthWitness] = []
        relation: tuple[str, str] | None = None
        for local_screen_point in _interior_witnesses(overlap_local):
            screen_point_array = (
                np.asarray(local_screen_point, dtype=float) + polygon_anchor
            )
            screen_point = (
                float(screen_point_array[0]),
                float(screen_point_array[1]),
            )
            first_interval = _ray_interval(
                first,
                screen_point,
                view,
                context=context,
                depth_epsilon=depth_epsilon,
                scene_anchor=scene_anchor,
            )
            second_interval = _ray_interval(
                second,
                screen_point,
                view,
                context=context,
                depth_epsilon=depth_epsilon,
                scene_anchor=scene_anchor,
            )
            if first_interval[1] < second_interval[0] - depth_epsilon:
                current = (first_id, second_id)
            elif second_interval[1] < first_interval[0] - depth_epsilon:
                current = (second_id, first_id)
            else:
                raise GlobalQuadricOcclusionError(
                    "strictly separated entities produced touching or overlapping "
                    "depth intervals at a projection witness: "
                    f"{first_id!r}, {second_id!r}"
                )
            if relation is None:
                relation = current
            elif relation != current:
                raise GlobalQuadricOcclusionError(
                    "surface depth order changes inside one connected projection "
                    "overlap; whole-surface painting would be ambiguous: "
                    f"{first_id!r}, {second_id!r}"
                )
            witnesses.append(
                SurfaceDepthWitness(
                    screen_point,
                    first_interval,
                    second_interval,
                    current[0],
                    current[1],
                )
            )
        assert relation is not None
        constraints.append(
            SurfaceOrderConstraint(relation[0], relation[1], "exact_ray_depth")
        )
        evidence_items.append(
            SurfaceDepthEvidence(
                first_id,
                second_id,
                "overlap",
                None,
                tuple(
                    (
                        float(point[0] + polygon_anchor[0]),
                        float(point[1] + polygon_anchor[1]),
                    )
                    for point in overlap_local
                ),
                tuple(witnesses),
                relation[0],
                relation[1],
            )
        )
    return tuple(evidence_items), tuple(constraints)


def _proxy_error_for_surface(
    surface: QuadricSurfaceSpec,
    view: ParallelView,
    context: ResolvedGeometryContext,
) -> float:
    screen = view.matrix[:2]
    points = np.asarray(
        [screen @ np.asarray(point, dtype=float) for point in surface.characteristic_points]
    )
    center = np.mean(points, axis=0)
    extent = max(float(np.linalg.norm(point - center)) for point in points)
    return max(
        _DEFAULT_PROXY_RELATIVE_ERROR * extent,
        16.0 * _screen_epsilon(context, screen),
    )


def _normalize_injected_constraints(
    constraints: Sequence[ConstraintInput],
    known_surface_ids: set[str],
) -> tuple[SurfaceOrderConstraint, ...]:
    result: list[SurfaceOrderConstraint] = []
    for item in constraints:
        if isinstance(item, PainterConstraint):
            farther, nearer = item.farther, item.nearer
        else:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "additional_surface_constraints must contain PainterConstraint or pairs"
                )
            farther, nearer = item
        farther = _identity(farther, "injected farther surface identity")
        nearer = _identity(nearer, "injected nearer surface identity")
        unknown = sorted({farther, nearer} - known_surface_ids)
        if unknown:
            raise GlobalQuadricOcclusionError(
                "additional surface constraint references unknown surfaces: "
                + ", ".join(unknown)
            )
        result.append(SurfaceOrderConstraint(farther, nearer, "injected"))
    return tuple(sorted(result, key=lambda item: (item.farther_surface_id, item.nearer_surface_id)))


def _merge_constraints(
    automatic: Sequence[SurfaceOrderConstraint],
    injected: Sequence[SurfaceOrderConstraint],
) -> tuple[SurfaceOrderConstraint, ...]:
    reasons: dict[tuple[str, str], set[str]] = {}
    for item in (*automatic, *injected):
        pair = (item.farther_surface_id, item.nearer_surface_id)
        reverse = (pair[1], pair[0])
        if reverse in reasons:
            raise GlobalQuadricOcclusionError(
                "surface constraints contain contradictory direct evidence: "
                f"{pair[0]!r}, {pair[1]!r}"
            )
        reasons.setdefault(pair, set()).add(item.reason)
    return tuple(
        SurfaceOrderConstraint(farther, nearer, "+".join(sorted(items)))
        for (farther, nearer), items in sorted(reasons.items())
    )


@dataclass(frozen=True, slots=True)
class GlobalQuadricFrame:
    """Serializable result of one complete global quadric frame build."""

    geometry_context: ResolvedGeometryContext
    separation_evidence: tuple[StrictSeparationEvidence, ...]
    surface_depth_evidence: tuple[SurfaceDepthEvidence, ...]
    surface_constraints: tuple[SurfaceOrderConstraint, ...]
    frame: QuadricCompositingFrame
    schema: str = GLOBAL_QUADRIC_FRAME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GLOBAL_QUADRIC_FRAME_SCHEMA:
            raise GlobalQuadricOcclusionError("invalid global-quadric frame schema")
        if not isinstance(self.geometry_context, ResolvedGeometryContext):
            raise TypeError("geometry_context must be a ResolvedGeometryContext")
        separation_pairs = tuple(
            (item.first_surface_id, item.second_surface_id)
            for item in self.separation_evidence
        )
        if separation_pairs != tuple(sorted(separation_pairs)):
            raise GlobalQuadricOcclusionError("separation evidence must be sorted")
        depth_pairs = tuple(
            (item.first_surface_id, item.second_surface_id)
            for item in self.surface_depth_evidence
        )
        if depth_pairs != tuple(sorted(depth_pairs)):
            raise GlobalQuadricOcclusionError("surface-depth evidence must be sorted")
        constraint_keys = tuple(
            (item.farther_surface_id, item.nearer_surface_id)
            for item in self.surface_constraints
        )
        if constraint_keys != tuple(sorted(constraint_keys)):
            raise GlobalQuadricOcclusionError("surface constraints must be sorted")
        if not isinstance(self.frame, QuadricCompositingFrame):
            raise TypeError("frame must be a QuadricCompositingFrame")

    @property
    def geometry_evidence(self) -> tuple[StrictSeparationEvidence, ...]:
        """Compatibility name for the pairwise solid-geometry certificates."""

        return self.separation_evidence

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "geometryContext": self.geometry_context.to_dict(),
            "separationEvidence": [item.to_dict() for item in self.separation_evidence],
            "surfaceDepthEvidence": [item.to_dict() for item in self.surface_depth_evidence],
            "surfaceConstraints": [item.to_dict() for item in self.surface_constraints],
            "frame": self.frame.to_dict(),
        }


def compute_global_quadric_frame(
    curves: Sequence[AnalyticCurve3D],
    surfaces: Sequence[QuadricSurfaceSpec],
    view: ParallelView,
    *,
    context: ContextInput = None,
    paint_policy: QuadricPaintPolicy | str = QuadricPaintPolicy.DIAGRAMMATIC,
    curve_styles: StyleInput = None,
    max_chord_error: float | None = None,
    max_segments: int = 4096,
    gjk_max_iterations: int = _DEFAULT_GJK_ITERATIONS,
    additional_surface_constraints: Sequence[ConstraintInput] = (),
) -> GlobalQuadricFrame:
    """Build visibility, proxies, automatic surface order, and painter frame.

    ``additional_surface_constraints`` is an explicit diagnostic/product
    override expressed as ``(farther, nearer)``.  It is primarily useful for
    validating cycle handling; automatic depth evidence is never discarded.
    """

    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    curve_items = tuple(curves)
    surface_items = _validated_surfaces(surfaces)
    resolved = _resolve_context(surface_items, context, curves=curve_items)
    iteration_limit = _iteration_limit(gjk_max_iterations)

    # A single finite surface has no pair to separate or globally depth-sort.
    # Section rendering uses this path for every frame, so avoid re-validating
    # the same surface collection and entering the pairwise ordering machinery.
    # The resulting evidence and constraints are exactly the empty tuples that
    # the general path would have produced.
    single_surface = len(surface_items) == 1
    separation = (
        ()
        if single_surface
        else _verify_render_component_separation(
            surface_items,
            context=resolved,
            gjk_max_iterations=iteration_limit,
        )
    )
    proxies: list[OpaqueProjectionProxy] = []
    if max_chord_error is not None:
        fixed_error = _finite_positive(max_chord_error, "max_chord_error")
    else:
        fixed_error = None
    for surface in surface_items:
        proxies.append(
            build_opaque_projection_proxy(
                surface,
                view,
                max_chord_error=(
                    fixed_error
                    if fixed_error is not None
                    else _proxy_error_for_surface(surface, view, resolved)
                ),
                max_segments=max_segments,
            )
        )
    proxy_items = tuple(sorted(proxies, key=lambda item: item.surface_id))
    if single_surface:
        depth_evidence: tuple[SurfaceDepthEvidence, ...] = ()
        automatic: tuple[SurfaceOrderConstraint, ...] = ()
    else:
        depth_evidence, automatic = _automatic_surface_order(
            surface_items,
            proxy_items,
            view,
            context=resolved,
            gjk_max_iterations=iteration_limit,
        )
    injected = _normalize_injected_constraints(
        additional_surface_constraints,
        {surface.surface_id for surface in surface_items},
    )
    constraints = _merge_constraints(automatic, injected)
    visibility = compute_quadric_visibility(
        curve_items,
        surface_items,
        view,
        context=resolved,
    )
    active_intervals = None
    if paint_policy is QuadricPaintPolicy.PHYSICAL or paint_policy == "physical":
        # In physical mode a hidden span has no paint item.  Restrict the
        # analytic crossing solve to the exact visible intervals so coincident
        # but wholly omitted supports cannot block an otherwise valid frame.
        active_intervals = {
            record.curve_id: record.visible_intervals
            for record in visibility.records
        }
    try:
        crossings = compute_projected_curve_crossings(
            curve_items,
            view,
            context=resolved,
            active_intervals=active_intervals,
        )
    except ProjectedCurveIntersectionError as exc:
        raise GlobalQuadricOcclusionError(
            f"projected curve ordering cannot be certified: {exc}"
        ) from exc
    try:
        frame = compute_quadric_compositing(
            visibility,
            proxy_items,
            paint_policy=paint_policy,
            curve_styles=curve_styles,
            surface_constraints=tuple(
                (item.farther_surface_id, item.nearer_surface_id)
                for item in constraints
            ),
            curve_crossings=crossings,
        )
    except QuadricCompositingError as exc:
        raise GlobalQuadricOcclusionError(
            f"global quadric painter graph failed: {exc}"
        ) from exc
    return GlobalQuadricFrame(
        resolved,
        separation,
        depth_evidence,
        constraints,
        frame,
    )


def canonical_global_quadric_frame_json(frame: GlobalQuadricFrame) -> str:
    if not isinstance(frame, GlobalQuadricFrame):
        raise TypeError("frame must be a GlobalQuadricFrame")
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "GLOBAL_QUADRIC_FRAME_SCHEMA",
    "GlobalQuadricFrame",
    "GlobalQuadricOcclusionError",
    "StrictSeparationEvidence",
    "SurfaceDepthEvidence",
    "SurfaceDepthWitness",
    "SurfaceOrderConstraint",
    "canonical_global_quadric_frame_json",
    "compute_global_quadric_frame",
    "verify_strict_quadric_separation",
]
