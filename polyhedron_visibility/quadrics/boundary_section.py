"""Renderer-neutral placement of semantic boundaries against a section plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
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
from .boundary_compositing import (
    BoundarySourceKind,
    QuadricBoundaryCompositingError,
    QuadricBoundarySource,
)
from .contract import PlaneDisplayPatchSpec, SectionPlane
from .critical import _curve_chart
from .curve_intersections import ProjectedCurveCrossing
from .roots import (
    PolynomialRootError,
    solve_real_polynomial,
    solve_real_polynomial_exp_chart,
)
from .section_compositing import (
    PlaneDepthRole,
    QuadricSectionCompositingFrame,
)


ContextInput = GeometryContext | ResolvedGeometryContext | None


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
        if not isinstance(self.interval, ParameterInterval) or self.interval.length <= 0.0:
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


def _point_in_triangle(
    point: np.ndarray,
    triangle: np.ndarray,
    epsilon: float,
) -> bool:
    first, second, third = triangle
    first_delta = second - first
    second_delta = third - first
    area = float(
        first_delta[0] * second_delta[1]
        - first_delta[1] * second_delta[0]
    )
    if abs(area) <= epsilon * epsilon:
        return False
    orientation = 1.0 if area > 0.0 else -1.0
    for start, end in ((first, second), (second, third), (third, first)):
        edge = end - start
        offset = point - start
        side = orientation * float(
            edge[0] * offset[1] - edge[1] * offset[0]
        )
        if side < -epsilon * max(float(np.linalg.norm(end - start)), epsilon):
            return False
    return True


def _roles_at_screen(
    frame: QuadricSectionCompositingFrame,
    screen_point: np.ndarray,
    screen_epsilon: float,
) -> tuple[PlaneDepthRole, ...]:
    candidates: set[PlaneDepthRole] = set()
    for fragment in frame.plane_fragments:
        triangle = np.asarray(fragment.screen_vertices, dtype=float)
        minimum = np.min(triangle, axis=0) - screen_epsilon
        maximum = np.max(triangle, axis=0) + screen_epsilon
        if np.any(screen_point < minimum) or np.any(screen_point > maximum):
            continue
        if _point_in_triangle(screen_point, triangle, screen_epsilon):
            candidates.add(fragment.role)
    if not candidates:
        raise QuadricBoundaryCompositingError(
            "boundary section midpoint is not covered by the plane partition"
        )
    # A true silhouette is intentionally shared by the projection-outside
    # region and one or more finite-depth regions.  Preserve every adjacent
    # role instead of guessing one side of the certified boundary.
    return tuple(sorted(candidates, key=lambda item: item.value))


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
    context: ContextInput = None,
) -> dict[str, tuple[QuadricBoundarySectionSpan, ...]]:
    """Split semantic curves wherever their order against the finite plane changes."""

    if not isinstance(section_frame, QuadricSectionCompositingFrame):
        raise TypeError("section_frame must be a QuadricSectionCompositingFrame")
    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
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
    result: dict[str, tuple[QuadricBoundarySectionSpan, ...]] = {}

    for source in source_items:
        if source.source_kind is BoundarySourceKind.PLANE_PATCH_EDGE:
            continue
        roots = _plane_intersection_parameters(source, plane, resolved)
        cells = partition_parameter_domain(
            source.curve.domain,
            (*roots, *edge_parameters[source.source_id]),
            tolerance=parameter_epsilon,
        )
        spans: list[QuadricBoundarySectionSpan] = []
        for cell in cells:
            world = np.asarray(source.curve.point(cell.midpoint), dtype=float)
            screen = view.matrix[:2] @ world
            coordinates = inverse @ (screen - screen_origin)
            if not patch.contains_coordinates(coordinates, context=resolved):
                spans.append(
                    QuadricBoundarySectionSpan(
                        cell,
                        BoundaryPlaneRelation.OUTSIDE_PATCH,
                        (),
                    )
                )
                continue
            plane_world = plane.point_from_coordinates(coordinates)
            depth_difference = float(np.dot(world - plane_world, direction))
            if depth_difference > depth_epsilon:
                relation = BoundaryPlaneRelation.BOUNDARY_IN_FRONT_OF_PLANE
            elif depth_difference < -depth_epsilon:
                relation = BoundaryPlaneRelation.BOUNDARY_BEHIND_PLANE
            else:
                relation = BoundaryPlaneRelation.COINCIDENT
            roles = _roles_at_screen(section_frame, screen, screen_epsilon)
            spans.append(
                QuadricBoundarySectionSpan(
                    cell,
                    relation,
                    tuple(role.value for role in roles),
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
    "QuadricBoundarySectionSpan",
    "compute_boundary_section_spans",
]
