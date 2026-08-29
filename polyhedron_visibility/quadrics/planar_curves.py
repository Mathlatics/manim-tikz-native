"""Renderer-neutral authored planes, circles, and ellipses in three dimensions.

TikZ's ordinary two-dimensional ``circle`` and ``ellipse`` paths do not carry
enough information to identify a plane in world space.  These contracts make
that missing authorship explicit while lowering to the analytic curve objects
already consumed by the quadric visibility and Manim layers.

The supporting frame owns parameter orientation as well as the mathematical
plane.  An animation that needs continuous parameter phase must author a
continuous ``u_axis``; the deterministic automatic basis is intended for
static frames and may change branch as the normal crosses a world-axis tie.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite, tau
from typing import Sequence

import numpy as np

from ..topology import ParameterInterval
from .algebra import AffineFrame3D, QuadricAlgebraError
from .curves import CircleArcCurve, CurveContractError, EllipseArcCurve


PLANAR_FRAME_3D_SCHEMA = "manim-planar-frame-3d/v1"
PLANAR_CURVE_3D_SCHEMA = "manim-planar-curve-3d/v1"
PLANAR_CURVE_SCENE_3D_SCHEMA = "manim-planar-curve-scene-3d/v1"

_PLANE_MEMBERSHIP_RELATIVE_TOLERANCE = 1.0e-10
_ANGULAR_TOLERANCE = 1.0e-12
_ROUND_OFF_MULTIPLIER = 64.0
_NORMAL_UNIT_TOLERANCE = 1.0e-10


class PlanarCurve3DContractError(ValueError):
    """Raised when authored planar geometry is ambiguous or inconsistent."""


def _canonical_float(value: float) -> float:
    result = float(value)
    return 0.0 if result == 0.0 else result


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanarCurve3DContractError(f"{label} must be a non-empty string")
    return value.strip()


def _point3(value: object, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise PlanarCurve3DContractError(
            f"{label} must contain three finite numbers"
        )
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurve3DContractError(
            f"{label} must contain three finite numbers"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise PlanarCurve3DContractError(
            f"{label} must contain three finite numbers"
        )
    return tuple(_canonical_float(item) for item in result)  # type: ignore[return-value]


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise PlanarCurve3DContractError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurve3DContractError(
            f"{label} must be finite and positive"
        ) from exc
    if not isfinite(result) or result <= 0.0:
        raise PlanarCurve3DContractError(f"{label} must be finite and positive")
    return _canonical_float(result)


def _domain(value: object) -> ParameterInterval:
    if not isinstance(value, ParameterInterval):
        raise TypeError("domain must be a ParameterInterval")
    if value.length <= 0.0 or value.length > tau + _ANGULAR_TOLERANCE:
        raise PlanarCurve3DContractError(
            "planar circle/ellipse domain must have positive length no greater than one revolution"
        )
    return value


def _domain_payload(value: ParameterInterval) -> list[float]:
    return [_canonical_float(value.start), _canonical_float(value.end)]


def _certify_analytic_curve(
    curve: CircleArcCurve | EllipseArcCurve,
) -> None:
    """Fail before authored data reaches an unsafe analytic-curve scale.

    The existing analytic runtime deliberately uses ordinary NumPy norms and
    cross products.  At extreme subnormal or near-overflow scales those
    operations can return zero, infinity, or NaN even though each authored
    scalar is finite.  This facade must not accept such a value and then fail
    later during an animation update.
    """

    try:
        center = np.asarray(curve.center, dtype=float)
        first = np.asarray(curve.first_axis, dtype=float)
        second = np.asarray(curve.second_axis, dtype=float)
        with np.errstate(all="ignore"):
            lengths = np.asarray(curve.semi_axis_lengths, dtype=float)
            normal = np.asarray(curve.normal, dtype=float)
            normal_length = float(np.linalg.norm(normal))
            coordinate_bound = np.abs(center) + np.abs(first) + np.abs(second)
            tangent_bound = np.abs(first) + np.abs(second)
    except (CurveContractError, FloatingPointError, OverflowError) as exc:
        raise PlanarCurve3DContractError(
            "planar curve lies outside the analytic runtime's certifiable numeric range"
        ) from exc
    if (
        np.any(lengths <= 0.0)
        or not np.all(np.isfinite(lengths))
        or not np.all(np.isfinite(normal))
        or not isfinite(normal_length)
        or abs(normal_length - 1.0) > _NORMAL_UNIT_TOLERANCE
        or not np.all(np.isfinite(coordinate_bound))
        or not np.all(np.isfinite(tangent_bound))
    ):
        raise PlanarCurve3DContractError(
            "planar curve lies outside the analytic runtime's certifiable numeric range"
        )


def _relative_point(
    value: object,
    origin: tuple[float, float, float],
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(_point3(value, label), dtype=float)
    delta = point - np.asarray(origin, dtype=float)
    if not np.all(np.isfinite(delta)):
        raise PlanarCurve3DContractError(
            f"{label} lies outside the certifiable finite coordinate range"
        )
    return point, delta


def _plane_membership_epsilon(
    point: np.ndarray,
    origin: tuple[float, float, float],
    delta: np.ndarray,
    normal: tuple[float, float, float],
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    normal_array = np.asarray(normal, dtype=float)
    dot_roundoff = (
        _ROUND_OFF_MULTIPLIER
        * float(np.finfo(float).eps)
        * float(np.sum(np.abs(delta * normal_array)))
    )
    origin_array = np.asarray(origin, dtype=float)
    coordinate_quantization = float(
        np.sum(
            0.5
            * (
                np.abs(np.spacing(point))
                + np.abs(np.spacing(origin_array))
            )
            * np.abs(normal_array)
        )
    )
    basis_orthogonality = (
        abs(float(coordinates[0]))
        * abs(float(np.dot(u_axis, normal_array)))
        + abs(float(coordinates[1]))
        * abs(float(np.dot(v_axis, normal_array)))
    )
    epsilon = dot_roundoff + coordinate_quantization + basis_orthogonality
    if not isfinite(epsilon):
        raise PlanarCurve3DContractError(
            "plane membership cannot be certified at this coordinate scale"
        )
    return epsilon


@dataclass(frozen=True, slots=True)
class PlanarFrame3D:
    """One stable right-handed coordinate frame on an infinite 3D plane."""

    frame_id: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    u_axis: tuple[float, float, float] | None = None
    schema: str = PLANAR_FRAME_3D_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLANAR_FRAME_3D_SCHEMA:
            raise PlanarCurve3DContractError("invalid planar-frame schema")
        frame_id = _identity(self.frame_id, "frame_id")
        point = _point3(self.point, "plane point")
        normal = _point3(self.normal, "plane normal")
        u_axis = None if self.u_axis is None else _point3(self.u_axis, "plane u_axis")
        try:
            frame = AffineFrame3D.from_axis(
                point,
                normal,
                radial_axis=u_axis,
            )
        except QuadricAlgebraError as exc:
            raise PlanarCurve3DContractError(
                f"invalid planar frame: {exc}"
            ) from exc
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "point", point)
        object.__setattr__(
            self,
            "normal",
            _point3(frame.z_axis, "normalized plane normal"),
        )
        object.__setattr__(
            self,
            "u_axis",
            _point3(frame.x_axis, "normalized plane u_axis"),
        )

    @property
    def affine_frame(self) -> AffineFrame3D:
        return AffineFrame3D.from_axis(
            self.point,
            self.normal,
            radial_axis=self.u_axis,
        )

    @property
    def v_axis(self) -> tuple[float, float, float]:
        return _point3(self.affine_frame.y_axis, "derived plane v_axis")

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = self.affine_frame
        return (
            np.asarray(frame.x_axis, dtype=float),
            np.asarray(frame.y_axis, dtype=float),
            np.asarray(frame.z_axis, dtype=float),
        )

    def signed_distance(self, point: Sequence[float]) -> float:
        _value, delta = _relative_point(
            point,
            self.point,
            "plane query point",
        )
        result = float(np.dot(delta, np.asarray(self.normal, dtype=float)))
        if not isfinite(result):
            raise PlanarCurve3DContractError(
                "plane signed distance is outside the certifiable finite range"
            )
        return result

    def coordinates_in_plane(
        self,
        point: Sequence[float],
        *,
        feature_scale: float | None = None,
    ) -> tuple[float, float]:
        value, delta = _relative_point(
            point,
            self.point,
            "plane query point",
        )
        normal = np.asarray(self.normal, dtype=float)
        distance = float(np.dot(delta, normal))
        if not isfinite(distance):
            raise PlanarCurve3DContractError(
                "plane membership is outside the certifiable finite range"
            )
        u_axis = np.asarray(self.u_axis, dtype=float)
        v_axis = np.asarray(self.v_axis, dtype=float)
        coordinates = np.asarray(
            (float(np.dot(delta, u_axis)), float(np.dot(delta, v_axis))),
            dtype=float,
        )
        if not np.all(np.isfinite(coordinates)):
            raise PlanarCurve3DContractError(
                "plane coordinates are outside the certifiable finite range"
            )
        epsilon = _plane_membership_epsilon(
            value,
            self.point,
            delta,
            self.normal,
            u_axis,
            v_axis,
            coordinates,
        )
        if feature_scale is not None:
            scale = _positive(feature_scale, "plane feature_scale")
            epsilon = max(
                epsilon,
                _PLANE_MEMBERSHIP_RELATIVE_TOLERANCE * scale,
            )
        if abs(distance) > epsilon:
            raise PlanarCurve3DContractError(
                "point does not lie on its authored supporting plane"
            )
        return (
            _canonical_float(coordinates[0]),
            _canonical_float(coordinates[1]),
        )

    def point_from_coordinates(self, value: Sequence[float]) -> np.ndarray:
        if isinstance(value, (str, bytes)):
            raise PlanarCurve3DContractError(
                "plane coordinates must contain two finite numbers"
            )
        try:
            uv = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PlanarCurve3DContractError(
                "plane coordinates must contain two finite numbers"
            ) from exc
        if uv.shape != (2,) or not np.all(np.isfinite(uv)):
            raise PlanarCurve3DContractError(
                "plane coordinates must contain two finite numbers"
            )
        result = (
            np.asarray(self.point, dtype=float)
            + float(uv[0]) * np.asarray(self.u_axis, dtype=float)
            + float(uv[1]) * np.asarray(self.v_axis, dtype=float)
        )
        if not np.all(np.isfinite(result)):
            raise PlanarCurve3DContractError(
                "world point is outside the certifiable finite coordinate range"
            )
        return result

    def canonical_point_on_plane(
        self,
        point: Sequence[float],
        *,
        feature_scale: float | None = None,
    ) -> tuple[float, float, float]:
        coordinates = self.coordinates_in_plane(
            point,
            feature_scale=feature_scale,
        )
        result = self.point_from_coordinates(coordinates)
        return _point3(result, "canonical plane point")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "frameId": self.frame_id,
            "point": list(self.point),
            "normal": list(self.normal),
            "uAxis": list(self.u_axis or ()),
            "vAxis": list(self.v_axis),
        }


@dataclass(frozen=True, slots=True)
class Circle3DSpec:
    """A circle with a stable supporting frame and parameter phase."""

    curve_id: str
    frame: PlanarFrame3D
    center: tuple[float, float, float]
    radius: float
    domain: ParameterInterval = ParameterInterval(0.0, tau)
    schema: str = PLANAR_CURVE_3D_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLANAR_CURVE_3D_SCHEMA:
            raise PlanarCurve3DContractError("invalid planar-curve schema")
        if not isinstance(self.frame, PlanarFrame3D):
            raise TypeError("frame must be a PlanarFrame3D")
        curve_id = _identity(self.curve_id, "curve_id")
        radius = _positive(self.radius, "circle radius")
        center = self.frame.canonical_point_on_plane(
            _point3(self.center, "circle center"),
            feature_scale=radius,
        )
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "domain", _domain(self.domain))
        try:
            with np.errstate(all="ignore"):
                _certify_analytic_curve(self.lower_to_analytic_curve())
        except CurveContractError as exc:
            raise PlanarCurve3DContractError(
                "circle cannot be lowered to the analytic curve runtime"
            ) from exc

    def lower_to_analytic_curve(self) -> CircleArcCurve:
        return CircleArcCurve(
            self.curve_id,
            self.center,
            self.radius,
            self.frame.normal,
            radial_axis=self.frame.u_axis,
            domain=self.domain,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "kind": "circle",
            "curveId": self.curve_id,
            "frameId": self.frame.frame_id,
            "center": list(self.center),
            "radius": self.radius,
            "domain": _domain_payload(self.domain),
        }


@dataclass(frozen=True, slots=True)
class Ellipse3DSpec:
    """An ellipse whose semi-axes follow one authored supporting frame."""

    curve_id: str
    frame: PlanarFrame3D
    center: tuple[float, float, float]
    semi_u: float
    semi_v: float
    domain: ParameterInterval = ParameterInterval(0.0, tau)
    schema: str = PLANAR_CURVE_3D_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLANAR_CURVE_3D_SCHEMA:
            raise PlanarCurve3DContractError("invalid planar-curve schema")
        if not isinstance(self.frame, PlanarFrame3D):
            raise TypeError("frame must be a PlanarFrame3D")
        curve_id = _identity(self.curve_id, "curve_id")
        semi_u = _positive(self.semi_u, "ellipse semi_u")
        semi_v = _positive(self.semi_v, "ellipse semi_v")
        center = self.frame.canonical_point_on_plane(
            _point3(self.center, "ellipse center"),
            feature_scale=min(semi_u, semi_v),
        )
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "semi_u", semi_u)
        object.__setattr__(self, "semi_v", semi_v)
        object.__setattr__(self, "domain", _domain(self.domain))
        try:
            with np.errstate(all="ignore"):
                _certify_analytic_curve(self.lower_to_analytic_curve())
        except CurveContractError as exc:
            raise PlanarCurve3DContractError(
                "ellipse cannot be lowered to the analytic curve runtime"
            ) from exc

    def lower_to_analytic_curve(self) -> EllipseArcCurve:
        u_axis = self.frame.u_axis
        assert u_axis is not None
        return EllipseArcCurve(
            self.curve_id,
            self.center,
            tuple(self.semi_u * item for item in u_axis),
            tuple(self.semi_v * item for item in self.frame.v_axis),
            domain=self.domain,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "kind": "ellipse",
            "curveId": self.curve_id,
            "frameId": self.frame.frame_id,
            "center": list(self.center),
            "semiU": self.semi_u,
            "semiV": self.semi_v,
            "domain": _domain_payload(self.domain),
        }


PlanarCurve3DSpec = Circle3DSpec | Ellipse3DSpec


@dataclass(frozen=True, slots=True)
class PlanarCurveScene3D:
    """A deterministic registry of shared supporting frames and planar curves."""

    frames: tuple[PlanarFrame3D, ...]
    curves: tuple[PlanarCurve3DSpec, ...]
    schema: str = PLANAR_CURVE_SCENE_3D_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLANAR_CURVE_SCENE_3D_SCHEMA:
            raise PlanarCurve3DContractError("invalid planar-curve-scene schema")
        frames = tuple(self.frames)
        curves = tuple(self.curves)
        if not all(isinstance(item, PlanarFrame3D) for item in frames):
            raise TypeError("frames must contain PlanarFrame3D values")
        if not all(isinstance(item, (Circle3DSpec, Ellipse3DSpec)) for item in curves):
            raise TypeError("curves must contain Circle3DSpec or Ellipse3DSpec values")
        frame_ids = tuple(item.frame_id for item in frames)
        curve_ids = tuple(item.curve_id for item in curves)
        if len(set(frame_ids)) != len(frame_ids):
            raise PlanarCurve3DContractError("supporting frame identities must be unique")
        if len(set(curve_ids)) != len(curve_ids):
            raise PlanarCurve3DContractError("planar curve identities must be unique")
        collisions = sorted(set(frame_ids) & set(curve_ids))
        if collisions:
            raise PlanarCurve3DContractError(
                "frame and curve identities must be globally distinct: "
                + ", ".join(collisions)
            )
        by_id = {item.frame_id: item for item in frames}
        for curve in curves:
            registered = by_id.get(curve.frame.frame_id)
            if registered is None:
                raise PlanarCurve3DContractError(
                    f"curve {curve.curve_id!r} references an unregistered supporting frame"
                )
            if registered != curve.frame:
                raise PlanarCurve3DContractError(
                    f"curve {curve.curve_id!r} disagrees with registered frame "
                    f"{curve.frame.frame_id!r}"
                )
        object.__setattr__(self, "frames", tuple(sorted(frames, key=lambda item: item.frame_id)))
        object.__setattr__(self, "curves", tuple(sorted(curves, key=lambda item: item.curve_id)))

    def lower_to_analytic_curves(
        self,
    ) -> tuple[CircleArcCurve | EllipseArcCurve, ...]:
        return tuple(item.lower_to_analytic_curve() for item in self.curves)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "frames": [item.to_dict() for item in self.frames],
            "curves": [item.to_dict() for item in self.curves],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


__all__ = [
    "Circle3DSpec",
    "Ellipse3DSpec",
    "PLANAR_CURVE_3D_SCHEMA",
    "PLANAR_CURVE_SCENE_3D_SCHEMA",
    "PLANAR_FRAME_3D_SCHEMA",
    "PlanarCurve3DContractError",
    "PlanarCurve3DSpec",
    "PlanarCurveScene3D",
    "PlanarFrame3D",
]
