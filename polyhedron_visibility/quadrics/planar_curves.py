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

from dataclasses import dataclass, field
from fractions import Fraction
import json
from math import isfinite, tau
from typing import Mapping, Sequence

import numpy as np

from ..topology import ParameterInterval
from .algebra import AffineFrame3D, QuadricAlgebraError
from .curves import CircleArcCurve, CurveContractError, EllipseArcCurve


PLANAR_FRAME_3D_SCHEMA = "manim-planar-frame-3d/v1"
PLANAR_POINT_3D_SCHEMA = "manim-planar-point-3d/v1"
PLANAR_CURVE_3D_SCHEMA = "manim-planar-curve-3d/v1"
PLANAR_CURVE_SCENE_3D_SCHEMA = "manim-planar-curve-scene-3d/v1"

_ANGULAR_TOLERANCE = 1.0e-12
_NORMAL_UNIT_TOLERANCE = 1.0e-10
_POINT_EVALUATION_RELATIVE_TOLERANCE = float(np.sqrt(np.finfo(float).eps))
_PLANAR_BASIS_TOLERANCE = 64.0 * float(np.finfo(float).eps)
_RADIAL_DIRECTION_RELATIVE_TOLERANCE = float(np.sqrt(np.finfo(float).eps))


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
        authored = np.asarray(value, dtype=object)
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurve3DContractError(
            f"{label} must contain three finite numbers"
        ) from exc
    if (
        result.shape != (3,)
        or authored.shape != (3,)
        or any(isinstance(item, (bool, np.bool_)) for item in authored)
        or not np.all(np.isfinite(result))
    ):
        raise PlanarCurve3DContractError(
            f"{label} must contain three finite numbers"
        )
    return tuple(_canonical_float(item) for item in result)  # type: ignore[return-value]


def _coordinates2(value: object, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise PlanarCurve3DContractError(
            f"{label} must contain two finite numbers"
        )
    try:
        authored = np.asarray(value, dtype=object)
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurve3DContractError(
            f"{label} must contain two finite numbers"
        ) from exc
    if (
        result.shape != (2,)
        or authored.shape != (2,)
        or any(isinstance(item, (bool, np.bool_)) for item in authored)
        or not np.all(np.isfinite(result))
    ):
        raise PlanarCurve3DContractError(
            f"{label} must contain two finite numbers"
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


def _mapping(
    value: object,
    expected_keys: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PlanarCurve3DContractError(f"{label} must be an object")
    keys = frozenset(value.keys())
    if keys != expected_keys:
        missing = sorted(key for key in expected_keys if key not in keys)
        extra = sorted(repr(key) for key in keys if key not in expected_keys)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise PlanarCurve3DContractError(
            f"{label} fields are invalid: " + "; ".join(details)
        )
    return value


def _certify_planar_basis(
    u_axis: tuple[float, float, float],
    v_axis: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> None:
    basis = np.column_stack(
        (
            np.asarray(u_axis, dtype=float),
            np.asarray(v_axis, dtype=float),
            np.asarray(normal, dtype=float),
        )
    )
    gram = basis.T @ basis
    if not np.allclose(
        gram,
        np.eye(3),
        rtol=0.0,
        atol=_PLANAR_BASIS_TOLERANCE,
    ):
        raise PlanarCurve3DContractError(
            "planar-frame axes are not certifiably orthonormal"
        )
    if float(np.linalg.det(basis)) <= 0.0 or not np.allclose(
        np.cross(basis[:, 0], basis[:, 1]),
        basis[:, 2],
        rtol=0.0,
        atol=_PLANAR_BASIS_TOLERANCE,
    ):
        raise PlanarCurve3DContractError(
            "planar-frame axes are not certifiably right-handed"
        )


def _direction_seed(
    value: tuple[float, float, float],
    label: str,
) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=float)
    scale = float(np.max(np.abs(vector)))
    if not isfinite(scale) or scale <= 0.0:
        raise PlanarCurve3DContractError(f"{label} must be non-zero")
    return _point3(vector / scale, label)


def _radial_direction_seed(
    normal_seed: tuple[float, float, float],
    value: tuple[float, float, float],
) -> tuple[float, float, float]:
    normal = np.asarray(normal_seed, dtype=float)
    normal /= float(np.linalg.norm(normal))
    radial = np.asarray(value, dtype=float)
    radial_scale = float(np.max(np.abs(radial)))
    if not isfinite(radial_scale) or radial_scale <= 0.0:
        raise PlanarCurve3DContractError("plane u_axis must be non-zero")
    radial /= radial_scale
    radial_length = float(np.linalg.norm(radial))
    candidate = radial - float(np.dot(radial, normal)) * normal
    candidate_length = float(np.linalg.norm(candidate))
    if (
        not isfinite(candidate_length)
        or candidate_length
        <= _RADIAL_DIRECTION_RELATIVE_TOLERANCE * radial_length
    ):
        raise PlanarCurve3DContractError(
            "plane u_axis must not be parallel or numerically indistinguishable from the normal"
        )
    return _direction_seed(
        _point3(candidate, "projected plane u_axis"),
        "plane u_axis seed",
    )


def _certify_direction_seed(
    value: tuple[float, float, float],
    label: str,
) -> tuple[float, float, float]:
    seed = _point3(value, label)
    if float(np.max(np.abs(seed))) != 1.0:
        raise PlanarCurve3DContractError(
            f"{label} must have a canonical unit maximum component"
        )
    return seed


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
    for axis, axis_length in zip((first, second), lengths):
        for direction in (-1.0, 1.0):
            endpoint = center + direction * axis
            actual_displacement = endpoint - center
            relative_error = float(
                np.linalg.norm(actual_displacement - direction * axis)
                / axis_length
            )
            if (
                not np.all(np.isfinite(endpoint))
                or not isfinite(relative_error)
                or relative_error > _POINT_EVALUATION_RELATIVE_TOLERANCE
            ):
                raise PlanarCurve3DContractError(
                    "planar curve semi-axis is not representable at its world-space center"
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


def _exact_plane_residual(
    point: np.ndarray,
    origin: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> Fraction:
    return sum(
        (
            Fraction.from_float(float(coordinate))
            - Fraction.from_float(float(reference))
        )
        * Fraction.from_float(float(component))
        for coordinate, reference, component in zip(point, origin, normal)
    )


def _certify_center_embedding(
    frame: PlanarFrame3D,
    center: tuple[float, float, float],
    coordinates: tuple[float, float],
    feature_scale: float,
    label: str,
) -> None:
    point = np.asarray(center, dtype=float)
    exact_u = float(_exact_plane_residual(point, frame.point, frame.u_axis))
    exact_v = float(_exact_plane_residual(point, frame.point, frame.v_axis))
    exact_normal = float(
        _exact_plane_residual(point, frame.point, frame.normal)
    )
    errors = np.asarray(
        (
            exact_u - coordinates[0],
            exact_v - coordinates[1],
            exact_normal,
        ),
        dtype=float,
    )
    maximum = float(np.max(np.abs(errors)))
    if maximum == 0.0:
        embedding_error = 0.0
    elif not isfinite(maximum):
        embedding_error = float("inf")
    else:
        embedding_error = maximum * float(np.linalg.norm(errors / maximum))
    tolerance = _POINT_EVALUATION_RELATIVE_TOLERANCE * feature_scale
    if not isfinite(embedding_error) or embedding_error > tolerance:
        raise PlanarCurve3DContractError(
            f"{label} center cannot be embedded in world space within the curve-scale error budget"
        )


def _plane_membership_epsilon(
    point: np.ndarray,
    origin: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> float:
    normal_array = np.asarray(normal, dtype=float)
    origin_array = np.asarray(origin, dtype=float)
    epsilon = float(
        np.sum(
            0.5
            * (
                np.abs(np.spacing(point))
                + np.abs(np.spacing(origin_array))
            )
            * np.abs(normal_array)
        )
    )
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
    _normal_seed: tuple[float, float, float] | None = field(
        default=None,
        repr=False,
        kw_only=True,
    )
    _u_axis_seed: tuple[float, float, float] | None = field(
        default=None,
        repr=False,
        kw_only=True,
    )
    _require_seed_match: bool = field(
        default=False,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _v_axis: tuple[float, float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema != PLANAR_FRAME_3D_SCHEMA:
            raise PlanarCurve3DContractError("invalid planar-frame schema")
        frame_id = _identity(self.frame_id, "frame_id")
        point = _point3(self.point, "plane point")
        supplied_normal = _point3(self.normal, "plane normal")
        supplied_u_axis = (
            None
            if self.u_axis is None
            else _point3(self.u_axis, "plane u_axis")
        )
        if not isinstance(self._require_seed_match, bool):
            raise TypeError("_require_seed_match must be boolean")
        if self._normal_seed is None and self._u_axis_seed is not None:
            raise PlanarCurve3DContractError(
                "plane u_axis seed requires a plane normal seed"
            )
        restoring = self._normal_seed is not None
        if restoring:
            normal_seed = _certify_direction_seed(
                self._normal_seed,
                "plane normal seed",
            )
            u_axis_seed = (
                None
                if self._u_axis_seed is None
                else _certify_direction_seed(
                    self._u_axis_seed,
                    "plane u_axis seed",
                )
            )
        else:
            normal_seed = _direction_seed(
                supplied_normal,
                "plane normal seed",
            )
            u_axis_seed = (
                None
                if supplied_u_axis is None
                else _radial_direction_seed(normal_seed, supplied_u_axis)
            )

        def build_frame() -> AffineFrame3D:
            try:
                return AffineFrame3D.from_axis(
                    point,
                    normal_seed,
                    radial_axis=u_axis_seed,
                )
            except QuadricAlgebraError as exc:
                raise PlanarCurve3DContractError(
                    f"invalid planar frame: {exc}"
                ) from exc

        frame = build_frame()
        normal = _point3(frame.z_axis, "normalized plane normal")
        u_axis = _point3(frame.x_axis, "normalized plane u_axis")
        v_axis = _point3(frame.y_axis, "normalized plane v_axis")
        _certify_planar_basis(u_axis, v_axis, normal)
        if restoring and (
            supplied_normal != normal or supplied_u_axis != u_axis
        ):
            if self._require_seed_match:
                raise PlanarCurve3DContractError(
                    "planar-frame canonical axes disagree with their direction seeds"
                )
            normal_seed = _direction_seed(
                supplied_normal,
                "plane normal seed",
            )
            u_axis_seed = (
                None
                if supplied_u_axis is None
                else _radial_direction_seed(normal_seed, supplied_u_axis)
            )
            frame = build_frame()
            normal = _point3(frame.z_axis, "normalized plane normal")
            u_axis = _point3(frame.x_axis, "normalized plane u_axis")
            v_axis = _point3(frame.y_axis, "normalized plane v_axis")
            _certify_planar_basis(u_axis, v_axis, normal)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "u_axis", u_axis)
        object.__setattr__(self, "_normal_seed", normal_seed)
        object.__setattr__(self, "_u_axis_seed", u_axis_seed)
        object.__setattr__(self, "_require_seed_match", False)
        object.__setattr__(self, "_v_axis", v_axis)

    @property
    def affine_frame(self) -> AffineFrame3D:
        result = object.__new__(AffineFrame3D)
        object.__setattr__(result, "origin", self.point)
        object.__setattr__(result, "x_axis", self.u_axis)
        object.__setattr__(result, "y_axis", self.v_axis)
        object.__setattr__(result, "z_axis", self.normal)
        return result

    @property
    def v_axis(self) -> tuple[float, float, float]:
        assert self._v_axis is not None
        return self._v_axis

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.u_axis, dtype=float),
            np.asarray(self.v_axis, dtype=float),
            np.asarray(self.normal, dtype=float),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PlanarFrame3D:
        value = _mapping(
            payload,
            frozenset(
                {
                    "schema",
                    "frameId",
                    "point",
                    "normal",
                    "uAxis",
                    "vAxis",
                    "normalSeed",
                    "uAxisSeed",
                }
            ),
            "planar-frame payload",
        )
        result = cls(
            value["frameId"],
            value["point"],
            value["normal"],
            value["uAxis"],
            schema=value["schema"],
            _normal_seed=_certify_direction_seed(
                value["normalSeed"],
                "plane normal seed",
            ),
            _u_axis_seed=(
                None
                if value["uAxisSeed"] is None
                else _certify_direction_seed(
                    value["uAxisSeed"],
                    "plane u_axis seed",
                )
            ),
            _require_seed_match=True,
        )
        if (
            result.normal != _point3(value["normal"], "plane normal")
            or result.u_axis != _point3(value["uAxis"], "plane uAxis")
            or result.v_axis != _point3(value["vAxis"], "plane vAxis")
        ):
            raise PlanarCurve3DContractError(
                "planar-frame payload is not the canonical basis for its plane"
            )
        return result

    def signed_distance(self, point: Sequence[float]) -> float:
        value, _delta = _relative_point(
            point,
            self.point,
            "plane query point",
        )
        result = float(_exact_plane_residual(value, self.point, self.normal))
        if not isfinite(result):
            raise PlanarCurve3DContractError(
                "plane signed distance is outside the certifiable finite range"
            )
        return result

    def coordinates_in_plane(
        self,
        point: Sequence[float],
    ) -> tuple[float, float]:
        value, delta = _relative_point(
            point,
            self.point,
            "plane query point",
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
            self.normal,
        )
        exact_residual = _exact_plane_residual(
            value,
            self.point,
            self.normal,
        )
        if abs(exact_residual) > Fraction.from_float(epsilon):
            raise PlanarCurve3DContractError(
                "point does not lie on its authored supporting plane"
            )
        return (
            _canonical_float(coordinates[0]),
            _canonical_float(coordinates[1]),
        )

    def point_from_coordinates(self, value: Sequence[float]) -> np.ndarray:
        uv = np.asarray(_coordinates2(value, "plane coordinates"), dtype=float)
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

    def certified_point(self, value: Sequence[float]) -> PlanarPoint3D:
        """Author a point by frame-local coordinates without lossy inference."""

        return PlanarPoint3D(self, _coordinates2(value, "plane coordinates"))

    def canonical_point_on_plane(
        self,
        point: Sequence[float],
    ) -> tuple[float, float, float]:
        coordinates = self.coordinates_in_plane(point)
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
            "normalSeed": list(self._normal_seed or ()),
            "uAxisSeed": (
                None
                if self._u_axis_seed is None
                else list(self._u_axis_seed)
            ),
        }


@dataclass(frozen=True, slots=True)
class PlanarPoint3D:
    """A plane-local point carrying the frame evidence that authored it."""

    frame: PlanarFrame3D
    coordinates: tuple[float, float]
    schema: str = PLANAR_POINT_3D_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLANAR_POINT_3D_SCHEMA:
            raise PlanarCurve3DContractError("invalid planar-point schema")
        if not isinstance(self.frame, PlanarFrame3D):
            raise TypeError("frame must be a PlanarFrame3D")
        coordinates = _coordinates2(self.coordinates, "plane coordinates")
        self.frame.point_from_coordinates(coordinates)
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def world_point(self) -> tuple[float, float, float]:
        return _point3(
            self.frame.point_from_coordinates(self.coordinates),
            "certified plane point",
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        frame: PlanarFrame3D,
    ) -> PlanarPoint3D:
        value = _mapping(
            payload,
            frozenset({"schema", "frameId", "coordinates", "worldPoint"}),
            "planar-point payload",
        )
        if value["frameId"] != frame.frame_id:
            raise PlanarCurve3DContractError(
                "planar-point payload references a different supporting frame"
            )
        point = cls(
            frame,
            value["coordinates"],
            schema=value["schema"],
        )
        if _point3(value["worldPoint"], "planar-point worldPoint") != point.world_point:
            raise PlanarCurve3DContractError(
                "planar-point worldPoint disagrees with its frame coordinates"
            )
        return point

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "frameId": self.frame.frame_id,
            "coordinates": list(self.coordinates),
            "worldPoint": list(self.world_point),
        }


def _curve_center(
    frame: PlanarFrame3D,
    center: tuple[float, float, float] | PlanarPoint3D,
    center_coordinates: tuple[float, float] | None,
    label: str,
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    supplied_coordinates = (
        None
        if center_coordinates is None
        else _coordinates2(center_coordinates, f"{label} center_coordinates")
    )
    if isinstance(center, PlanarPoint3D):
        if center.frame != frame:
            raise PlanarCurve3DContractError(
                f"{label} center references a different supporting frame"
            )
        # A certified point carries the stronger authorship channel.  Treat it
        # as authoritative so ``dataclasses.replace(curve, center=point)``
        # cannot be defeated by the old derived ``center_coordinates`` field.
        return center.world_point, center.coordinates

    authored_center = _point3(center, f"{label} center")
    if supplied_coordinates is not None:
        certified = frame.certified_point(supplied_coordinates)
        if certified.world_point == authored_center:
            # Frozen dataclasses expose the normalized world center and the
            # original plane coordinates as separate init fields.  A plain
            # ``dataclasses.replace(curve)`` feeds both back into ``__init__``;
            # retain the already-certified authorship evidence instead of
            # deriving the coordinates a second time and demanding a bitwise
            # inverse round trip from floating-point projection.
            return authored_center, supplied_coordinates
    resolved_coordinates = frame.coordinates_in_plane(authored_center)
    if (
        supplied_coordinates is not None
        and supplied_coordinates != resolved_coordinates
    ):
        raise PlanarCurve3DContractError(
            f"{label} center disagrees with center_coordinates"
        )
    # Preserve a raw world-space center exactly as authored.  Plane membership
    # may use its strict component-ULP quantization bound, but accepting the
    # point must never silently snap it to a different world coordinate.
    return authored_center, resolved_coordinates


def _curve_center_from_payload(
    frame: PlanarFrame3D,
    center: object,
    center_coordinates: object,
    label: str,
) -> tuple[tuple[float, float, float] | PlanarPoint3D, tuple[float, float]]:
    world_center = _point3(center, f"{label} center")
    coordinates = _coordinates2(
        center_coordinates,
        f"{label} centerCoordinates",
    )
    certified = frame.certified_point(coordinates)
    if certified.world_point == world_center:
        return certified, coordinates
    return world_center, coordinates


@dataclass(frozen=True, slots=True, init=False)
class Circle3DSpec:
    """A circle with a stable supporting frame and parameter phase."""

    curve_id: str
    frame: PlanarFrame3D
    center: tuple[float, float, float]
    radius: float
    center_coordinates: tuple[float, float]
    domain: ParameterInterval = ParameterInterval(0.0, tau)
    schema: str = PLANAR_CURVE_3D_SCHEMA

    def __init__(
        self,
        curve_id: str,
        frame: PlanarFrame3D,
        center: Sequence[float] | PlanarPoint3D,
        radius: float,
        domain: ParameterInterval = ParameterInterval(0.0, tau),
        schema: str = PLANAR_CURVE_3D_SCHEMA,
        *,
        center_coordinates: Sequence[float] | None = None,
    ) -> None:
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "center_coordinates", center_coordinates)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema != PLANAR_CURVE_3D_SCHEMA:
            raise PlanarCurve3DContractError("invalid planar-curve schema")
        if not isinstance(self.frame, PlanarFrame3D):
            raise TypeError("frame must be a PlanarFrame3D")
        curve_id = _identity(self.curve_id, "curve_id")
        radius = _positive(self.radius, "circle radius")
        center, center_coordinates = _curve_center(
            self.frame,
            self.center,
            self.center_coordinates,
            "circle",
        )
        _certify_center_embedding(
            self.frame,
            center,
            center_coordinates,
            radius,
            "circle",
        )
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "center_coordinates", center_coordinates)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "domain", _domain(self.domain))
        try:
            with np.errstate(all="ignore"):
                _certify_analytic_curve(self.lower_to_analytic_curve())
        except CurveContractError as exc:
            raise PlanarCurve3DContractError(
                "circle cannot be lowered to the analytic curve runtime"
            ) from exc

    @classmethod
    def from_plane_coordinates(
        cls,
        curve_id: str,
        frame: PlanarFrame3D,
        center_coordinates: Sequence[float],
        radius: float,
        domain: ParameterInterval = ParameterInterval(0.0, tau),
    ) -> Circle3DSpec:
        return cls(
            curve_id,
            frame,
            frame.certified_point(center_coordinates),
            radius,
            domain=domain,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        frame: PlanarFrame3D,
    ) -> Circle3DSpec:
        value = _mapping(
            payload,
            frozenset(
                {
                    "schema",
                    "kind",
                    "curveId",
                    "frameId",
                    "center",
                    "centerCoordinates",
                    "radius",
                    "domain",
                }
            ),
            "circle payload",
        )
        if value["kind"] != "circle":
            raise PlanarCurve3DContractError("circle payload kind must be 'circle'")
        if value["frameId"] != frame.frame_id:
            raise PlanarCurve3DContractError(
                "circle payload references a different supporting frame"
            )
        domain_values = _coordinates2(value["domain"], "circle domain")
        center, center_coordinates = _curve_center_from_payload(
            frame,
            value["center"],
            value["centerCoordinates"],
            "circle",
        )
        return cls(
            value["curveId"],
            frame,
            center,
            value["radius"],
            domain=ParameterInterval(*domain_values),
            schema=value["schema"],
            center_coordinates=center_coordinates,
        )

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
            "centerCoordinates": list(self.center_coordinates),
            "radius": self.radius,
            "domain": _domain_payload(self.domain),
        }


@dataclass(frozen=True, slots=True, init=False)
class Ellipse3DSpec:
    """An ellipse whose semi-axes follow one authored supporting frame."""

    curve_id: str
    frame: PlanarFrame3D
    center: tuple[float, float, float]
    semi_u: float
    semi_v: float
    center_coordinates: tuple[float, float]
    domain: ParameterInterval = ParameterInterval(0.0, tau)
    schema: str = PLANAR_CURVE_3D_SCHEMA

    def __init__(
        self,
        curve_id: str,
        frame: PlanarFrame3D,
        center: Sequence[float] | PlanarPoint3D,
        semi_u: float,
        semi_v: float,
        domain: ParameterInterval = ParameterInterval(0.0, tau),
        schema: str = PLANAR_CURVE_3D_SCHEMA,
        *,
        center_coordinates: Sequence[float] | None = None,
    ) -> None:
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "semi_u", semi_u)
        object.__setattr__(self, "semi_v", semi_v)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "center_coordinates", center_coordinates)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema != PLANAR_CURVE_3D_SCHEMA:
            raise PlanarCurve3DContractError("invalid planar-curve schema")
        if not isinstance(self.frame, PlanarFrame3D):
            raise TypeError("frame must be a PlanarFrame3D")
        curve_id = _identity(self.curve_id, "curve_id")
        semi_u = _positive(self.semi_u, "ellipse semi_u")
        semi_v = _positive(self.semi_v, "ellipse semi_v")
        center, center_coordinates = _curve_center(
            self.frame,
            self.center,
            self.center_coordinates,
            "ellipse",
        )
        _certify_center_embedding(
            self.frame,
            center,
            center_coordinates,
            min(semi_u, semi_v),
            "ellipse",
        )
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "center_coordinates", center_coordinates)
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

    @classmethod
    def from_plane_coordinates(
        cls,
        curve_id: str,
        frame: PlanarFrame3D,
        center_coordinates: Sequence[float],
        semi_u: float,
        semi_v: float,
        domain: ParameterInterval = ParameterInterval(0.0, tau),
    ) -> Ellipse3DSpec:
        return cls(
            curve_id,
            frame,
            frame.certified_point(center_coordinates),
            semi_u,
            semi_v,
            domain=domain,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        frame: PlanarFrame3D,
    ) -> Ellipse3DSpec:
        value = _mapping(
            payload,
            frozenset(
                {
                    "schema",
                    "kind",
                    "curveId",
                    "frameId",
                    "center",
                    "centerCoordinates",
                    "semiU",
                    "semiV",
                    "domain",
                }
            ),
            "ellipse payload",
        )
        if value["kind"] != "ellipse":
            raise PlanarCurve3DContractError(
                "ellipse payload kind must be 'ellipse'"
            )
        if value["frameId"] != frame.frame_id:
            raise PlanarCurve3DContractError(
                "ellipse payload references a different supporting frame"
            )
        domain_values = _coordinates2(value["domain"], "ellipse domain")
        center, center_coordinates = _curve_center_from_payload(
            frame,
            value["center"],
            value["centerCoordinates"],
            "ellipse",
        )
        return cls(
            value["curveId"],
            frame,
            center,
            value["semiU"],
            value["semiV"],
            domain=ParameterInterval(*domain_values),
            schema=value["schema"],
            center_coordinates=center_coordinates,
        )

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
            "centerCoordinates": list(self.center_coordinates),
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PlanarCurveScene3D:
        value = _mapping(
            payload,
            frozenset({"schema", "frames", "curves"}),
            "planar-curve-scene payload",
        )
        raw_frames = value["frames"]
        raw_curves = value["curves"]
        if (
            isinstance(raw_frames, (str, bytes))
            or not isinstance(raw_frames, Sequence)
        ):
            raise PlanarCurve3DContractError(
                "planar-curve-scene frames must be an array"
            )
        if (
            isinstance(raw_curves, (str, bytes))
            or not isinstance(raw_curves, Sequence)
        ):
            raise PlanarCurve3DContractError(
                "planar-curve-scene curves must be an array"
            )
        frames = tuple(PlanarFrame3D.from_dict(item) for item in raw_frames)
        frame_ids = tuple(item.frame_id for item in frames)
        if len(set(frame_ids)) != len(frame_ids):
            raise PlanarCurve3DContractError(
                "supporting frame identities must be unique"
            )
        by_id = {item.frame_id: item for item in frames}
        curves: list[PlanarCurve3DSpec] = []
        for raw_curve in raw_curves:
            if not isinstance(raw_curve, Mapping):
                raise PlanarCurve3DContractError(
                    "planar-curve-scene curves must contain objects"
                )
            frame_id = raw_curve.get("frameId")
            frame = by_id.get(frame_id) if isinstance(frame_id, str) else None
            if frame is None:
                raise PlanarCurve3DContractError(
                    "planar curve references an unregistered supporting frame"
                )
            kind = raw_curve.get("kind")
            if kind == "circle":
                curves.append(Circle3DSpec.from_dict(raw_curve, frame))
            elif kind == "ellipse":
                curves.append(Ellipse3DSpec.from_dict(raw_curve, frame))
            else:
                raise PlanarCurve3DContractError(
                    "planar curve kind must be 'circle' or 'ellipse'"
                )
        return cls(
            frames,
            tuple(curves),
            schema=value["schema"],
        )

    @classmethod
    def from_json(cls, payload: str) -> PlanarCurveScene3D:
        if not isinstance(payload, str):
            raise TypeError("planar-curve-scene JSON must be a string")
        try:
            value = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PlanarCurve3DContractError(
                "planar-curve-scene JSON is invalid"
            ) from exc
        return cls.from_dict(value)

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
    "PLANAR_POINT_3D_SCHEMA",
    "PlanarCurve3DContractError",
    "PlanarCurve3DSpec",
    "PlanarCurveScene3D",
    "PlanarFrame3D",
    "PlanarPoint3D",
]
