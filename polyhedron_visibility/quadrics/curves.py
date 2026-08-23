"""Finite analytic three-dimensional curves for quadric visibility.

The objects in this module contain geometry, not render samples.  Every curve
has one finite :class:`~polyhedron_visibility.topology.ParameterInterval`, can
evaluate an exact analytic point and tangent, and exposes a JSON-compatible
``to_dict`` representation.  Importing the module never imports Manim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, tau
from typing import Sequence

import numpy as np

from ..topology import ParameterInterval
from .conics import ConicKind, ConicParameterization


ANALYTIC_CURVE_SCHEMA = "manim-analytic-curve-3d/v1"
_ORTHOGONAL_TOLERANCE = 1.0e-10
_ANGULAR_TOLERANCE = 1.0e-12
_TRIG_SNAP_TOLERANCE = 64.0 * float(np.finfo(float).eps)


class CurveContractError(ValueError):
    """Raised when an analytic curve contract is ambiguous or degenerate."""


def _identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CurveContractError("curve_id must be a non-empty string")
    return value.strip()


def _point3(value: object, label: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise CurveContractError(f"{label} must contain three finite numbers")
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CurveContractError(
            f"{label} must contain three finite numbers"
        ) from exc
    if not all(isfinite(component) for component in result):
        raise CurveContractError(f"{label} must contain three finite numbers")
    return result  # type: ignore[return-value]


def _nonzero_vector3(value: object, label: str) -> tuple[float, float, float]:
    result = _point3(value, label)
    if float(np.linalg.norm(result)) <= 0.0:
        raise CurveContractError(f"{label} must be non-zero")
    return result


def _finite_domain(value: object) -> ParameterInterval:
    if not isinstance(value, ParameterInterval):
        raise TypeError("domain must be a ParameterInterval")
    if value.length <= 0.0:
        raise CurveContractError("curve domain must have positive length")
    return value


def _parameter(value: float, domain: ParameterInterval) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CurveContractError("curve parameter must be finite") from exc
    if not isfinite(result):
        raise CurveContractError("curve parameter must be finite")
    if not domain.contains(result):
        raise CurveContractError("curve parameter lies outside the curve domain")
    return result


def _tuple3(value: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(component) for component in value)  # type: ignore[return-value]


def _domain_dict(domain: ParameterInterval) -> list[float]:
    return [domain.start, domain.end]


def _stable_sin_cos(value: float) -> tuple[float, float]:
    sine = sin(value)
    cosine = cos(value)
    if abs(sine) <= _TRIG_SNAP_TOLERANCE:
        sine = 0.0
    elif abs(abs(sine) - 1.0) <= _TRIG_SNAP_TOLERANCE:
        sine = 1.0 if sine > 0.0 else -1.0
    if abs(cosine) <= _TRIG_SNAP_TOLERANCE:
        cosine = 0.0
    elif abs(abs(cosine) - 1.0) <= _TRIG_SNAP_TOLERANCE:
        cosine = 1.0 if cosine > 0.0 else -1.0
    return sine, cosine


@dataclass(frozen=True, slots=True)
class SegmentCurve:
    """One non-degenerate affine segment over an arbitrary finite domain."""

    curve_id: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    domain: ParameterInterval = ParameterInterval(0.0, 1.0)
    schema: str = ANALYTIC_CURVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ANALYTIC_CURVE_SCHEMA:
            raise CurveContractError("invalid analytic-curve schema")
        curve_id = _identity(self.curve_id)
        start = _point3(self.start, "segment start")
        end = _point3(self.end, "segment end")
        domain = _finite_domain(self.domain)
        if float(np.linalg.norm(np.asarray(end) - np.asarray(start))) <= 0.0:
            raise CurveContractError("segment endpoints must be distinct")
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "domain", domain)

    @property
    def displacement(self) -> tuple[float, float, float]:
        return _tuple3(np.asarray(self.end) - np.asarray(self.start))

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.displacement))

    def point(self, parameter: float) -> tuple[float, float, float]:
        value = _parameter(parameter, self.domain)
        ratio = (value - self.domain.start) / self.domain.length
        start = np.asarray(self.start, dtype=float)
        return _tuple3(start + ratio * np.asarray(self.displacement, dtype=float))

    def tangent(self, parameter: float) -> tuple[float, float, float]:
        _parameter(parameter, self.domain)
        return _tuple3(np.asarray(self.displacement, dtype=float) / self.domain.length)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "kind": "segment",
            "curveId": self.curve_id,
            "domain": _domain_dict(self.domain),
            "start": list(self.start),
            "end": list(self.end),
        }


@dataclass(frozen=True, slots=True)
class EllipseArcCurve:
    """A finite circle or ellipse arc in one authored three-dimensional plane.

    ``first_axis`` and ``second_axis`` include their semi-axis lengths.  The
    parameterization is ``center + first*cos(t) + second*sin(t)``.
    """

    curve_id: str
    center: tuple[float, float, float]
    first_axis: tuple[float, float, float]
    second_axis: tuple[float, float, float]
    domain: ParameterInterval = ParameterInterval(0.0, tau)
    schema: str = ANALYTIC_CURVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ANALYTIC_CURVE_SCHEMA:
            raise CurveContractError("invalid analytic-curve schema")
        curve_id = _identity(self.curve_id)
        center = _point3(self.center, "ellipse center")
        first = _nonzero_vector3(self.first_axis, "ellipse first_axis")
        second = _nonzero_vector3(self.second_axis, "ellipse second_axis")
        domain = _finite_domain(self.domain)
        first_array = np.asarray(first, dtype=float)
        second_array = np.asarray(second, dtype=float)
        first_length = float(np.linalg.norm(first_array))
        second_length = float(np.linalg.norm(second_array))
        normalized_dot = abs(float(np.dot(first_array, second_array))) / (
            first_length * second_length
        )
        if normalized_dot > _ORTHOGONAL_TOLERANCE:
            raise CurveContractError("ellipse axes must be orthogonal")
        if domain.length > tau + _ANGULAR_TOLERANCE:
            raise CurveContractError("ellipse arc cannot span more than one revolution")
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "first_axis", first)
        object.__setattr__(self, "second_axis", second)
        object.__setattr__(self, "domain", domain)

    @property
    def semi_axis_lengths(self) -> tuple[float, float]:
        return (
            float(np.linalg.norm(self.first_axis)),
            float(np.linalg.norm(self.second_axis)),
        )

    @property
    def normal(self) -> tuple[float, float, float]:
        normal = np.cross(self.first_axis, self.second_axis)
        return _tuple3(normal / float(np.linalg.norm(normal)))

    @property
    def closed(self) -> bool:
        return abs(self.domain.length - tau) <= _ANGULAR_TOLERANCE

    @property
    def circular(self) -> bool:
        first, second = self.semi_axis_lengths
        return abs(first - second) <= _ORTHOGONAL_TOLERANCE * max(first, second)

    def point(self, parameter: float) -> tuple[float, float, float]:
        value = _parameter(parameter, self.domain)
        sine, cosine = _stable_sin_cos(value)
        result = (
            np.asarray(self.center, dtype=float)
            + cosine * np.asarray(self.first_axis, dtype=float)
            + sine * np.asarray(self.second_axis, dtype=float)
        )
        return _tuple3(result)

    def tangent(self, parameter: float) -> tuple[float, float, float]:
        value = _parameter(parameter, self.domain)
        sine, cosine = _stable_sin_cos(value)
        result = (
            -sine * np.asarray(self.first_axis, dtype=float)
            + cosine * np.asarray(self.second_axis, dtype=float)
        )
        return _tuple3(result)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "kind": "ellipse_arc",
            "curveId": self.curve_id,
            "domain": _domain_dict(self.domain),
            "center": list(self.center),
            "firstAxis": list(self.first_axis),
            "secondAxis": list(self.second_axis),
            "closed": self.closed,
            "circular": self.circular,
        }


@dataclass(frozen=True, slots=True, init=False)
class CircleArcCurve(EllipseArcCurve):
    """Convenience constructor for a deterministically oriented circle arc."""

    def __init__(
        self,
        curve_id: str,
        center: Sequence[float],
        radius: float,
        normal: Sequence[float],
        *,
        radial_axis: Sequence[float] | None = None,
        domain: ParameterInterval = ParameterInterval(0.0, tau),
        schema: str = ANALYTIC_CURVE_SCHEMA,
    ) -> None:
        try:
            scalar_radius = float(radius)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CurveContractError("circle radius must be finite and positive") from exc
        if isinstance(radius, bool) or not isfinite(scalar_radius) or scalar_radius <= 0.0:
            raise CurveContractError("circle radius must be finite and positive")
        unit_normal = np.asarray(_nonzero_vector3(normal, "circle normal"), dtype=float)
        unit_normal /= float(np.linalg.norm(unit_normal))
        if radial_axis is None:
            reference = np.eye(3)[int(np.argmin(np.abs(unit_normal)))]
        else:
            reference = np.asarray(
                _nonzero_vector3(radial_axis, "circle radial_axis"), dtype=float
            )
        first = reference - float(np.dot(reference, unit_normal)) * unit_normal
        first_length = float(np.linalg.norm(first))
        if first_length <= _ORTHOGONAL_TOLERANCE * float(np.linalg.norm(reference)):
            raise CurveContractError("circle radial_axis must not be parallel to normal")
        first /= first_length
        second = np.cross(unit_normal, first)
        EllipseArcCurve.__init__(
            self,
            curve_id=curve_id,
            center=_point3(center, "circle center"),
            first_axis=_tuple3(scalar_radius * first),
            second_axis=_tuple3(scalar_radius * second),
            domain=domain,
            schema=schema,
        )

    @property
    def radius(self) -> float:
        return self.semi_axis_lengths[0]

    def to_dict(self) -> dict[str, object]:
        # ``dataclass(slots=True)`` replaces the class object during
        # decoration.  A zero-argument ``super()`` can therefore retain the
        # pre-decoration ``__class__`` cell and fail at runtime.  Call the
        # known base implementation explicitly; this is also clearer here
        # because the method only changes the serialized semantic kind.
        result = EllipseArcCurve.to_dict(self)
        result["kind"] = "circle_arc"
        result["radius"] = self.radius
        result["normal"] = list(self.normal)
        return result


@dataclass(frozen=True, slots=True)
class ParametricConicBranch:
    """Adapt one analytic 2D conic branch into finite 3D curve geometry.

    ``plane_embedding`` is a 4x3 affine homogeneous matrix mapping
    ``[u, v, 1]`` to ``[x, y, z, 1]``.  The adapter retains the exact conic
    functions; no sampled points are stored or used for evaluation.
    """

    curve_id: str
    parameterization: ConicParameterization
    plane_embedding: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    domain: ParameterInterval
    schema: str = ANALYTIC_CURVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ANALYTIC_CURVE_SCHEMA:
            raise CurveContractError("invalid analytic-curve schema")
        curve_id = _identity(self.curve_id)
        if not isinstance(self.parameterization, ConicParameterization):
            raise TypeError("parameterization must be a ConicParameterization")
        if self.parameterization.kind in {ConicKind.POINT, ConicKind.EMPTY}:
            raise CurveContractError("conic branch must describe a curve")
        domain = _finite_domain(self.domain)
        try:
            embedding = np.asarray(self.plane_embedding, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CurveContractError(
                "plane_embedding must be a finite affine 4x3 matrix"
            ) from exc
        if embedding.shape != (4, 3) or not np.all(np.isfinite(embedding)):
            raise CurveContractError(
                "plane_embedding must be a finite affine 4x3 matrix"
            )
        if not np.allclose(
            embedding[3], np.asarray((0.0, 0.0, 1.0)), rtol=0.0, atol=1.0e-12
        ):
            raise CurveContractError(
                "plane_embedding must map affine points to affine world points"
            )
        linear = embedding[:3, :2]
        if int(np.linalg.matrix_rank(linear, tol=1.0e-12)) != 2:
            raise CurveContractError("plane_embedding axes must be independent")
        natural = self.parameterization.natural_domain
        if natural is not None and (
            domain.start < natural.start - _ANGULAR_TOLERANCE
            or domain.end > natural.end + _ANGULAR_TOLERANCE
        ):
            raise CurveContractError("domain lies outside the conic branch domain")
        canonical = tuple(
            tuple(float(component) for component in row) for row in embedding
        )
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "plane_embedding", canonical)
        object.__setattr__(self, "domain", domain)

    def point(self, parameter: float) -> tuple[float, float, float]:
        value = _parameter(parameter, self.domain)
        plane_point = self.parameterization.point(value)
        homogeneous = np.asarray((plane_point[0], plane_point[1], 1.0), dtype=float)
        world = np.asarray(self.plane_embedding, dtype=float) @ homogeneous
        if abs(float(world[3]) - 1.0) > 1.0e-10:
            raise CurveContractError("plane embedding produced a non-affine point")
        return _tuple3(world[:3])

    def tangent(self, parameter: float) -> tuple[float, float, float]:
        value = _parameter(parameter, self.domain)
        plane_tangent = self.parameterization.tangent(value)
        world = np.asarray(self.plane_embedding, dtype=float)[:3, :2] @ plane_tangent
        if float(np.linalg.norm(world)) <= 0.0:
            raise CurveContractError("conic branch produced a zero tangent")
        return _tuple3(world)

    @property
    def closed(self) -> bool:
        """Whether the authored finite domain contains one closed oval once.

        ``ConicParameterization.closed`` describes the natural analytic
        branch, while this adapter may deliberately expose only an arc of that
        branch.  Both conditions are required before the start/end parameters
        may be treated as one geometric seam.
        """

        return bool(
            self.parameterization.closed
            and abs(self.domain.length - tau) <= _ANGULAR_TOLERANCE
        )

    def to_dict(self) -> dict[str, object]:
        branch = self.parameterization
        return {
            "schema": self.schema,
            "kind": "parametric_conic_branch",
            "curveId": self.curve_id,
            "domain": _domain_dict(self.domain),
            "planeEmbedding": [list(row) for row in self.plane_embedding],
            "conic": {
                "kind": branch.kind.value,
                "branchLabel": branch.branch_label,
                "origin": list(branch.origin),
                "firstAxis": list(branch.first_axis),
                "secondAxis": list(branch.second_axis),
                "branchSign": branch.branch_sign,
                "naturalDomain": (
                    None
                    if branch.natural_domain is None
                    else _domain_dict(branch.natural_domain)
                ),
                "closed": branch.closed,
            },
        }


__all__ = [
    "ANALYTIC_CURVE_SCHEMA",
    "CircleArcCurve",
    "CurveContractError",
    "EllipseArcCurve",
    "ParametricConicBranch",
    "SegmentCurve",
]
