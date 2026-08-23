"""Exact supporting-conic sections of finite sphere/cylinder/cone entities.

The mathematical plane is embedded by the homogeneous matrix ``H`` and the
supporting conic is computed as ``C = H.T @ Q @ H``.  Classification is kept
separate from finite-entity clipping: a cone can, for example, have a
supporting hyperbola while its authored axial interval displays two bounded
open arcs (or nothing).
"""

from __future__ import annotations

from math import acos, atan2, cos, cosh, exp, isfinite, log, sin, sinh, tau
from typing import Callable, Sequence

import numpy as np

from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)
from ..topology import ParameterInterval, partition_parameter_domain
from .conics import (
    ConicClassification,
    ConicError,
    ConicKind,
    ConicParameterization,
    classify_conic,
)
from .contract import ConeSpec, CylinderSpec, SectionPlane, SphereSpec
from .roots import PolynomialRootError, solve_real_polynomial
from .trace import (
    FiniteSectionTopology,
    QuadricSectionTrace,
    SectionBranchTrace,
    SectionComponentTrace,
)


QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec
ContextInput = GeometryContext | ResolvedGeometryContext | None


class QuadricSectionError(ValueError):
    """A plane/quadric section cannot be represented without guessing."""


class UnboundedFiniteSectionError(QuadricSectionError):
    """An authored finite entity failed to bound a supporting branch."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuadricSectionError(f"{label} must be a non-empty string")
    return value.strip()


def _plane_embedding(plane: SectionPlane) -> np.ndarray:
    frame = plane.frame
    origin = np.asarray(frame.origin, dtype=float)
    first = np.asarray(frame.x_axis, dtype=float)
    second = np.asarray(frame.y_axis, dtype=float)
    return np.asarray(
        (
            (first[0], second[0], origin[0]),
            (first[1], second[1], origin[1]),
            (first[2], second[2], origin[2]),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )


def restrict_quadric_to_plane(
    surface: QuadricSurfaceSpec,
    plane: SectionPlane,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(C, H)`` with ``C = H.T @ Q @ H``."""

    if not isinstance(surface, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    if not isinstance(plane, SectionPlane):
        raise TypeError("plane must be a SectionPlane")
    embedding = _plane_embedding(plane)
    quadric = np.asarray(surface.support_quadric.matrix, dtype=float)
    conic = embedding.T @ quadric @ embedding
    conic = 0.5 * (conic + conic.T)
    # Keep the explicit H.T@Q@H path authoritative while checking the public
    # algebra helper does not silently interpret the plane differently.
    public_restriction = np.asarray(plane.restrict(surface.support_quadric), dtype=float)
    scale = max(1.0, float(np.max(np.abs(conic))))
    if not np.allclose(conic, public_restriction, rtol=0.0, atol=1.0e-12 * scale):
        raise QuadricSectionError("plane and homogeneous embedding restrictions disagree")
    return conic, embedding


def _resolved_context(
    surface: QuadricSurfaceSpec,
    plane: SectionPlane,
    context: ContextInput,
) -> ResolvedGeometryContext:
    if isinstance(context, ResolvedGeometryContext):
        return resolve_geometry_context(context)
    characteristic = tuple(surface.characteristic_points) + (plane.point,)
    return resolve_geometry_context(context, positions=characteristic)


def _axial_range(surface: QuadricSurfaceSpec) -> tuple[float, float] | None:
    value = getattr(surface, "axial_range", None)
    if value is None:
        return None
    lower, upper = (float(item) for item in value)
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise QuadricSectionError("surface axial_range must be finite and increasing")
    return lower, upper


def _tuple_matrix(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _world_point(embedding: np.ndarray, uv: Sequence[float]) -> np.ndarray:
    point = np.asarray(uv, dtype=float)
    homogeneous = embedding @ np.asarray((point[0], point[1], 1.0), dtype=float)
    return homogeneous[:3]


def _classify_in_scaled_plane_coordinates(
    conic: np.ndarray,
    coordinate_scale: float,
    *,
    coefficient_tolerance: float,
) -> tuple[
    ConicClassification,
    tuple[ConicParameterization, ...],
    tuple[tuple[float, float], ...],
]:
    """Classify after making the plane coordinates dimensionless.

    In an affine conic the quadratic, linear, and constant coefficients have
    different length units.  Merely dividing the whole 3x3 matrix by its
    largest entry therefore changes the apparent rank when geometrically
    similar scenes are authored at very small or very large world scales.

    ``uv = coordinate_scale * uv_scaled`` balances those units before the
    dimensionless rank/sign tests.  Returned branch geometry is mapped back to
    the section plane's original world-length coordinates; the raw
    ``H.T @ Q @ H`` matrix remains authoritative in the public trace.
    """

    scale = float(coordinate_scale)
    if not isfinite(scale) or scale <= 0.0:
        raise QuadricSectionError("plane coordinate scale must be finite and positive")
    coordinate_transform = np.diag((scale, scale, 1.0))
    scaled_conic = coordinate_transform.T @ conic @ coordinate_transform
    classification = classify_conic(
        scaled_conic,
        coefficient_tolerance=coefficient_tolerance,
    )
    parameterizations = tuple(
        ConicParameterization(
            kind=branch.kind,
            branch_label=branch.branch_label,
            origin=tuple(scale * float(item) for item in branch.origin),
            first_axis=tuple(scale * float(item) for item in branch.first_axis),
            second_axis=tuple(scale * float(item) for item in branch.second_axis),
            branch_sign=branch.branch_sign,
            natural_domain=branch.natural_domain,
            closed=branch.closed,
        )
        for branch in classification.branches
    )
    isolated_points = tuple(
        tuple(scale * float(item) for item in point)
        for point in classification.isolated_points
    )
    return classification, parameterizations, isolated_points


def _axial_value(surface: QuadricSurfaceSpec, point: Sequence[float]) -> float:
    return float(surface.frame.to_local_point(point)[2])


def _axial_coefficients(
    surface: QuadricSurfaceSpec,
    embedding: np.ndarray,
    branch: ConicParameterization,
) -> tuple[float, float, float]:
    origin = np.asarray(branch.origin, dtype=float)
    first = np.asarray(branch.first_axis, dtype=float)
    second = np.asarray(branch.second_axis, dtype=float)
    world_origin = _world_point(embedding, origin)
    origin_axial = _axial_value(surface, world_origin)
    first_axial = _axial_value(surface, _world_point(embedding, origin + first)) - origin_axial
    second_axial = _axial_value(surface, _world_point(embedding, origin + second)) - origin_axial
    if branch.kind is ConicKind.HYPERBOLA:
        first_axial *= branch.branch_sign
    return origin_axial, first_axial, second_axial


def _inside(value: float, bounds: tuple[float, float], epsilon: float) -> bool:
    return bounds[0] - epsilon <= value <= bounds[1] + epsilon


def _merge_intervals(
    intervals: Sequence[ParameterInterval],
    epsilon: float,
) -> tuple[ParameterInterval, ...]:
    if not intervals:
        return ()
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    result = [ordered[0]]
    for interval in ordered[1:]:
        previous = result[-1]
        if interval.start <= previous.end + epsilon:
            result[-1] = ParameterInterval(
                previous.start,
                max(previous.end, interval.end),
            )
        else:
            result.append(interval)
    return tuple(result)


def _cluster(values: Sequence[float], epsilon: float) -> tuple[float, ...]:
    result: list[float] = []
    for value in sorted(float(item) for item in values if isfinite(float(item))):
        if not result or value - result[-1] > epsilon:
            result.append(value)
    return tuple(result)


def _ellipse_roots(
    constant: float,
    cosine: float,
    sine: float,
    target: float,
    epsilon: float,
) -> tuple[float, ...]:
    amplitude = float(np.hypot(cosine, sine))
    if amplitude <= epsilon:
        return ()
    ratio = (target - constant) / amplitude
    if ratio < -1.0 - epsilon or ratio > 1.0 + epsilon:
        return ()
    ratio = min(1.0, max(-1.0, ratio))
    phase = atan2(sine, cosine)
    angle = acos(ratio)
    values = []
    for raw in (phase - angle, phase + angle):
        normalized = raw % tau
        if tau - normalized <= epsilon:
            normalized = 0.0
        values.append(normalized)
    return _cluster(values, epsilon)


def _polynomial_roots(
    coefficients: Sequence[float],
    *,
    context: ResolvedGeometryContext,
    parameter_tolerance: float,
) -> tuple[float, ...]:
    scale = max(abs(float(item)) for item in coefficients)
    if scale == 0.0:
        return ()
    try:
        roots = solve_real_polynomial(
            tuple(float(item) / scale for item in coefficients),
            context=context,
            parameter_tolerance=parameter_tolerance,
        )
    except PolynomialRootError as exc:
        raise QuadricSectionError(f"axial clipping roots are ambiguous: {exc}") from exc
    return tuple(item.value for item in roots)


def _periodic_components(
    branch: ConicParameterization,
    coefficients: tuple[float, float, float],
    bounds: tuple[float, float],
    *,
    boundary_epsilon: float,
    parameter_epsilon: float,
) -> tuple[tuple[tuple[ParameterInterval, ...], ...], tuple[float, ...]]:
    constant, cosine, sine = coefficients
    roots = _cluster(
        (
            *_ellipse_roots(
                constant, cosine, sine, bounds[0], parameter_epsilon
            ),
            *_ellipse_roots(
                constant, cosine, sine, bounds[1], parameter_epsilon
            ),
        ),
        parameter_epsilon,
    )
    domain = branch.natural_domain
    if domain is None:
        raise QuadricSectionError("periodic branch has no natural domain")
    cells = partition_parameter_domain(
        domain,
        roots,
        tolerance=parameter_epsilon,
    )

    def axial(parameter: float) -> float:
        return constant + cosine * cos(parameter) + sine * sin(parameter)

    kept = _merge_intervals(
        tuple(
            cell
            for cell in cells
            if _inside(axial(cell.midpoint), bounds, boundary_epsilon)
        ),
        parameter_epsilon,
    )
    if (
        len(kept) == 1
        and abs(kept[0].start - domain.start) <= parameter_epsilon
        and abs(kept[0].end - domain.end) <= parameter_epsilon
    ):
        return ((domain,),), ()

    components: list[tuple[ParameterInterval, ...]] = []
    if (
        len(kept) >= 2
        and abs(kept[0].start - domain.start) <= parameter_epsilon
        and abs(kept[-1].end - domain.end) <= parameter_epsilon
    ):
        # The two canonical intervals touch through the periodic seam and are
        # one connected arc.  Keep interval order canonical in the trace; the
        # renderer can detect the seam from the branch's natural domain.
        components.append((kept[0], kept[-1]))
        components.extend((item,) for item in kept[1:-1])
    else:
        components.extend((item,) for item in kept)

    isolated = tuple(
        root
        for root in roots
        if _inside(axial(root), bounds, boundary_epsilon)
        and not any(
            interval.contains(root, tolerance=parameter_epsilon)
            for component in components
            for interval in component
        )
    )
    return tuple(components), _cluster(isolated, parameter_epsilon)


def _open_axial_function(
    branch: ConicParameterization,
    coefficients: tuple[float, float, float],
) -> Callable[[float], float]:
    constant, first, second = coefficients
    if branch.kind is ConicKind.HYPERBOLA:
        return lambda parameter: (
            constant + first * cosh(parameter) + second * sinh(parameter)
        )
    if branch.kind is ConicKind.PARABOLA:
        return lambda parameter: (
            constant + first * parameter + second * parameter * parameter
        )
    return lambda parameter: constant + first * parameter


def _open_boundary_roots(
    branch: ConicParameterization,
    coefficients: tuple[float, float, float],
    target: float,
    *,
    context: ResolvedGeometryContext,
    parameter_epsilon: float,
) -> tuple[float, ...]:
    constant, first, second = coefficients
    if branch.kind is ConicKind.HYPERBOLA:
        # y=e**t turns a*cosh(t)+b*sinh(t)+c-target into
        # ((a+b)/2)y**2 + (c-target)y + (a-b)/2.
        roots_y = _polynomial_roots(
            (
                0.5 * (first - second),
                constant - target,
                0.5 * (first + second),
            ),
            context=context,
            parameter_tolerance=parameter_epsilon,
        )
        return tuple(log(value) for value in roots_y if value > 0.0)
    if branch.kind is ConicKind.PARABOLA:
        return _polynomial_roots(
            (constant - target, first, second),
            context=context,
            parameter_tolerance=parameter_epsilon,
        )
    return _polynomial_roots(
        (constant - target, first),
        context=context,
        parameter_tolerance=parameter_epsilon,
    )


def _safe_open_value(function: Callable[[float], float], value: float) -> float:
    try:
        result = float(function(value))
    except (OverflowError, FloatingPointError):
        return float("inf")
    if np.isnan(result):
        raise QuadricSectionError("axial clipping produced NaN")
    return result


def _open_components(
    branch: ConicParameterization,
    coefficients: tuple[float, float, float],
    bounds: tuple[float, float],
    *,
    context: ResolvedGeometryContext,
    boundary_epsilon: float,
    parameter_epsilon: float,
) -> tuple[tuple[tuple[ParameterInterval, ...], ...], tuple[float, ...]]:
    roots = _cluster(
        (
            *_open_boundary_roots(
                branch,
                coefficients,
                bounds[0],
                context=context,
                parameter_epsilon=parameter_epsilon,
            ),
            *_open_boundary_roots(
                branch,
                coefficients,
                bounds[1],
                context=context,
                parameter_epsilon=parameter_epsilon,
            ),
        ),
        parameter_epsilon,
    )
    axial = _open_axial_function(branch, coefficients)
    if not roots:
        if _inside(_safe_open_value(axial, 0.0), bounds, boundary_epsilon):
            raise UnboundedFiniteSectionError(
                f"finite axial range does not bound branch {branch.branch_label!r}"
            )
        return (), ()

    # All finite interior cells are classified only after every exact boundary
    # root has been found.  The two unbounded tails are checked separately;
    # retaining either would contradict the finite-entity contract.
    if _inside(
        _safe_open_value(axial, roots[0] - 1.0), bounds, boundary_epsilon
    ) or _inside(
        _safe_open_value(axial, roots[-1] + 1.0), bounds, boundary_epsilon
    ):
        raise UnboundedFiniteSectionError(
            f"finite axial range leaves branch {branch.branch_label!r} unbounded"
        )
    kept = _merge_intervals(
        tuple(
            ParameterInterval(start, end)
            for start, end in zip(roots, roots[1:])
            if end - start > parameter_epsilon
            and _inside(
                _safe_open_value(axial, 0.5 * (start + end)),
                bounds,
                boundary_epsilon,
            )
        ),
        parameter_epsilon,
    )
    components = tuple((item,) for item in kept)
    isolated = tuple(
        root
        for root in roots
        if _inside(_safe_open_value(axial, root), bounds, boundary_epsilon)
        and not any(
            interval.contains(root, tolerance=parameter_epsilon)
            for component in components
            for interval in component
        )
    )
    return components, _cluster(isolated, parameter_epsilon)


def _topology(
    components: Sequence[SectionComponentTrace],
    points: Sequence[Sequence[float]],
) -> FiniteSectionTopology:
    if components and points:
        return FiniteSectionTopology.CURVES_AND_POINTS
    if points:
        return (
            FiniteSectionTopology.POINT
            if len(points) == 1
            else FiniteSectionTopology.MULTIPLE_POINTS
        )
    if not components:
        return FiniteSectionTopology.EMPTY
    if len(components) == 1:
        return (
            FiniteSectionTopology.CLOSED_CURVE
            if components[0].closed
            else FiniteSectionTopology.OPEN_CURVE
        )
    return FiniteSectionTopology.MULTIPLE_OPEN_CURVES


def compute_quadric_section(
    section_id: str,
    surface: QuadricSurfaceSpec,
    plane: SectionPlane,
    *,
    context: ContextInput = None,
    coefficient_tolerance: float | None = None,
) -> QuadricSectionTrace:
    """Return the supporting conic and finite axial clipping trace."""

    identity = _identity(section_id, "section_id")
    if not isinstance(surface, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    if not isinstance(plane, SectionPlane):
        raise TypeError("plane must be a SectionPlane")
    resolved = _resolved_context(surface, plane, context)
    classification_tolerance = (
        max(np.finfo(float).eps * 1024.0, resolved.epsilon(GeometryQuantity.ANGULAR))
        if coefficient_tolerance is None
        else float(coefficient_tolerance)
    )
    if not isfinite(classification_tolerance) or classification_tolerance <= 0.0:
        raise QuadricSectionError(
            "coefficient_tolerance must be finite and positive"
        )
    conic, embedding = restrict_quadric_to_plane(surface, plane)
    try:
        classification, parameterizations, isolated_plane_points = (
            _classify_in_scaled_plane_coordinates(
                conic,
                resolved.resolved.scale,
                coefficient_tolerance=classification_tolerance,
            )
        )
    except ConicError as exc:
        raise QuadricSectionError(str(exc)) from exc

    embedding_tuple = _tuple_matrix(embedding)
    conic_tuple = _tuple_matrix(conic)
    bounds = _axial_range(surface)
    boundary_epsilon = resolved.epsilon(GeometryQuantity.BOUNDARY)
    parameter_epsilon = max(
        np.finfo(float).eps * 1024.0,
        resolved.epsilon(GeometryQuantity.PARAMETER),
    )

    isolated_world: list[tuple[float, float, float]] = []
    if classification.kind is ConicKind.POINT:
        for uv in isolated_plane_points:
            point = _world_point(embedding, uv)
            if bounds is None or _inside(
                _axial_value(surface, point), bounds, boundary_epsilon
            ):
                isolated_world.append(tuple(float(item) for item in point))
    elif classification.kind is ConicKind.EMPTY:
        pass

    branches: list[SectionBranchTrace] = []
    components: list[SectionComponentTrace] = []
    for parameterization in parameterizations:
        branch_id = f"{identity}:component:{parameterization.branch_label}"
        branch_trace = SectionBranchTrace(
            branch_id,
            parameterization,
            embedding_tuple,  # type: ignore[arg-type]
        )
        branches.append(branch_trace)
        if bounds is None:
            if parameterization.natural_domain is None:
                raise UnboundedFiniteSectionError(
                    "an unbounded supporting branch requires a finite axial entity"
                )
            interval_groups = ((parameterization.natural_domain,),)
            isolated_parameters: tuple[float, ...] = ()
        else:
            coefficients = _axial_coefficients(
                surface,
                embedding,
                parameterization,
            )
            if parameterization.closed:
                interval_groups, isolated_parameters = _periodic_components(
                    parameterization,
                    coefficients,
                    bounds,
                    boundary_epsilon=boundary_epsilon,
                    parameter_epsilon=parameter_epsilon,
                )
            else:
                interval_groups, isolated_parameters = _open_components(
                    parameterization,
                    coefficients,
                    bounds,
                    context=resolved,
                    boundary_epsilon=boundary_epsilon,
                    parameter_epsilon=parameter_epsilon,
                )
        for parameter in isolated_parameters:
            point = branch_trace.world_point(parameter)
            isolated_world.append(tuple(float(item) for item in point))

        for index, intervals in enumerate(interval_groups):
            full_closed = (
                parameterization.closed
                and len(interval_groups) == 1
                and len(intervals) == 1
                and parameterization.natural_domain == intervals[0]
            )
            component_id = (
                branch_id
                if len(interval_groups) == 1
                else f"{branch_id}:clip:{index:04d}"
            )
            components.append(
                SectionComponentTrace(
                    component_id,
                    branch_id,
                    tuple(sorted(intervals)),
                    closed=full_closed,
                )
            )

    # Deduplicate tangent points contributed by two algebraic branches (for
    # example two generator lines meeting at the cone apex).
    deduped_points: list[tuple[float, float, float]] = []
    for point in sorted(isolated_world):
        value = np.asarray(point, dtype=float)
        if any(
            float(np.linalg.norm(value - np.asarray(previous, dtype=float)))
            <= boundary_epsilon
            for previous in deduped_points
        ):
            continue
        deduped_points.append(point)

    branches.sort(key=lambda item: item.branch_id)
    components.sort(key=lambda item: item.component_id)
    return QuadricSectionTrace(
        section_id=identity,
        surface_id=surface.surface_id,
        supporting_kind=classification.kind,
        finite_topology=_topology(components, deduped_points),
        conic_matrix=conic_tuple,  # type: ignore[arg-type]
        plane_embedding=embedding_tuple,  # type: ignore[arg-type]
        branches=tuple(branches),
        components=tuple(components),
        isolated_world_points=tuple(deduped_points),
    )


def intersect_plane_with_quadric(
    section_id: str,
    surface: QuadricSurfaceSpec,
    plane: SectionPlane,
    **kwargs: object,
) -> QuadricSectionTrace:
    """Descriptive alias for :func:`compute_quadric_section`."""

    return compute_quadric_section(section_id, surface, plane, **kwargs)


__all__ = [
    "QuadricSectionError",
    "QuadricSurfaceSpec",
    "UnboundedFiniteSectionError",
    "compute_quadric_section",
    "intersect_plane_with_quadric",
    "restrict_quadric_to_plane",
]
