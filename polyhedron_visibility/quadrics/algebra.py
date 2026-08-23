"""Renderer-neutral affine and homogeneous-quadric algebra.

The public contracts in :mod:`polyhedron_visibility.quadrics.contract` retain
the semantic difference between a sphere, cylinder, and cone.  This module is
the deliberately smaller numerical layer beneath those contracts: a stable
right-handed local frame and a symmetric homogeneous quadratic form

``[x, y, z, 1] @ Q @ [x, y, z, 1].T == 0``.

No renderer is imported here.  In particular, importing this module must not
import Manim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

from ..geometry import (
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    resolve_geometry_context,
)


class QuadricAlgebraError(ValueError):
    """Raised when affine or quadric algebra is ambiguous or non-finite."""


class CoincidentRayError(QuadricAlgebraError):
    """Raised when every point on a ray belongs to a quadric surface."""


def _vector3(value: Sequence[float], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricAlgebraError(
            f"{label} must be a finite three-component vector"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise QuadricAlgebraError(
            f"{label} must be a finite three-component vector"
        )
    return result


def _unit_vector(value: Sequence[float], label: str) -> np.ndarray:
    result = _vector3(value, label)
    length = float(np.linalg.norm(result))
    if not isfinite(length) or length <= 0.0:
        raise QuadricAlgebraError(f"{label} must be non-zero")
    return result / length


def _matrix4(value: Sequence[Sequence[float]], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricAlgebraError(f"{label} must be a finite 4x4 matrix") from exc
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise QuadricAlgebraError(f"{label} must be a finite 4x4 matrix")
    return result


def _canonical_tuple3(value: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _canonical_matrix(
    value: np.ndarray,
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _resolved_context(
    context: GeometryContext | ResolvedGeometryContext | None,
    positions: Sequence[Sequence[float]],
) -> ResolvedGeometryContext:
    if isinstance(context, ResolvedGeometryContext):
        return resolve_geometry_context(context)
    return resolve_geometry_context(context, positions=positions)


@dataclass(frozen=True, slots=True)
class AffineFrame3D:
    """One immutable right-handed orthonormal local coordinate frame.

    Local points are mapped to world space as
    ``origin + x*x_axis + y*y_axis + z*z_axis``.  :meth:`from_axis` chooses a
    deterministic radial axis when the author does not provide one.
    """

    origin: tuple[float, float, float]
    x_axis: tuple[float, float, float]
    y_axis: tuple[float, float, float]
    z_axis: tuple[float, float, float]

    def __post_init__(self) -> None:
        origin = _vector3(self.origin, "frame origin")
        x_axis = _unit_vector(self.x_axis, "frame x_axis")
        y_axis = _unit_vector(self.y_axis, "frame y_axis")
        z_axis = _unit_vector(self.z_axis, "frame z_axis")
        basis = np.column_stack((x_axis, y_axis, z_axis))
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(3), rtol=0.0, atol=1.0e-10):
            raise QuadricAlgebraError("frame axes must be mutually orthonormal")
        if float(np.linalg.det(basis)) <= 0.0 or not np.allclose(
            np.cross(x_axis, y_axis), z_axis, rtol=0.0, atol=1.0e-10
        ):
            raise QuadricAlgebraError("frame axes must form a right-handed basis")
        object.__setattr__(self, "origin", _canonical_tuple3(origin))
        object.__setattr__(self, "x_axis", _canonical_tuple3(x_axis))
        object.__setattr__(self, "y_axis", _canonical_tuple3(y_axis))
        object.__setattr__(self, "z_axis", _canonical_tuple3(z_axis))

    @classmethod
    def from_axis(
        cls,
        origin: Sequence[float],
        axis: Sequence[float],
        *,
        radial_axis: Sequence[float] | None = None,
    ) -> "AffineFrame3D":
        """Build a stable frame whose local z-axis follows ``axis``.

        Without an authored radial direction, the canonical world basis least
        aligned with the z-axis is projected into the radial plane.  Ties use
        the first world basis vector, so identical input always produces
        byte-identical axes.
        """

        center = _vector3(origin, "frame origin")
        z_axis = _unit_vector(axis, "frame axis")
        if radial_axis is None:
            basis = np.eye(3)[int(np.argmin(np.abs(z_axis)))]
            candidate = basis - float(np.dot(basis, z_axis)) * z_axis
            reference_length = 1.0
        else:
            authored = _vector3(radial_axis, "frame radial_axis")
            reference_length = float(np.linalg.norm(authored))
            if reference_length <= 0.0:
                raise QuadricAlgebraError("frame radial_axis must be non-zero")
            candidate = authored - float(np.dot(authored, z_axis)) * z_axis
        candidate_length = float(np.linalg.norm(candidate))
        if (
            not isfinite(candidate_length)
            or candidate_length
            <= np.finfo(float).eps * 64.0 * reference_length
        ):
            raise QuadricAlgebraError(
                "frame radial_axis must not be parallel to the axis"
            )
        x_axis = candidate / candidate_length
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= float(np.linalg.norm(y_axis))
        return cls(
            _canonical_tuple3(center),
            _canonical_tuple3(x_axis),
            _canonical_tuple3(y_axis),
            _canonical_tuple3(z_axis),
        )

    @property
    def local_to_world_matrix(self) -> np.ndarray:
        result = np.eye(4, dtype=float)
        result[:3, :3] = np.column_stack(
            (
                np.asarray(self.x_axis, dtype=float),
                np.asarray(self.y_axis, dtype=float),
                np.asarray(self.z_axis, dtype=float),
            )
        )
        result[:3, 3] = np.asarray(self.origin, dtype=float)
        return result

    @property
    def world_to_local_matrix(self) -> np.ndarray:
        # The linear part is orthonormal, so the transpose is the exact inverse
        # up to the already-validated floating-point representation.
        rotation = self.local_to_world_matrix[:3, :3]
        result = np.eye(4, dtype=float)
        result[:3, :3] = rotation.T
        result[:3, 3] = -rotation.T @ np.asarray(self.origin, dtype=float)
        return result

    def to_world_point(self, value: Sequence[float]) -> np.ndarray:
        point = _vector3(value, "local point")
        homogeneous = self.local_to_world_matrix @ np.append(point, 1.0)
        return homogeneous[:3]

    def to_local_point(self, value: Sequence[float]) -> np.ndarray:
        point = _vector3(value, "world point")
        homogeneous = self.world_to_local_matrix @ np.append(point, 1.0)
        return homogeneous[:3]

    def to_world_vector(self, value: Sequence[float]) -> np.ndarray:
        vector = _vector3(value, "local vector")
        return self.local_to_world_matrix[:3, :3] @ vector

    def to_local_vector(self, value: Sequence[float]) -> np.ndarray:
        vector = _vector3(value, "world vector")
        return self.world_to_local_matrix[:3, :3] @ vector


@dataclass(frozen=True, slots=True)
class HomogeneousQuadric:
    """A finite, non-zero symmetric homogeneous quadratic form."""

    matrix: tuple[tuple[float, float, float, float], ...]

    def __post_init__(self) -> None:
        matrix = _matrix4(self.matrix, "quadric matrix")
        scale = float(np.max(np.abs(matrix)))
        if scale == 0.0:
            raise QuadricAlgebraError("quadric matrix must not be zero")
        asymmetry = float(np.max(np.abs(matrix - matrix.T)))
        if asymmetry > 1.0e-12 * scale:
            raise QuadricAlgebraError("quadric matrix must be symmetric")
        matrix = 0.5 * (matrix + matrix.T)
        object.__setattr__(self, "matrix", _canonical_matrix(matrix))

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)

    @classmethod
    def from_local_matrix(
        cls,
        local_matrix: Sequence[Sequence[float]],
        local_to_world: Sequence[Sequence[float]],
    ) -> "HomogeneousQuadric":
        return cls(_canonical_matrix(_matrix4(local_matrix, "local quadric matrix"))).affine_transform(
            local_to_world
        )

    def affine_transform(
        self,
        local_to_world: Sequence[Sequence[float]],
    ) -> "HomogeneousQuadric":
        """Return this form after one invertible affine point transform.

        ``local_to_world`` maps points from the current coordinate system to
        the returned coordinate system.  Consequently ``Q_world`` is
        ``inverse(T).T @ Q_local @ inverse(T)``.
        """

        transform = _matrix4(local_to_world, "affine transform")
        scale = float(np.max(np.abs(transform)))
        if not np.allclose(
            transform[3],
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            rtol=0.0,
            atol=max(np.finfo(float).eps * 64.0 * scale, 1.0e-15),
        ):
            raise QuadricAlgebraError(
                "affine transform must have last row [0, 0, 0, 1]"
            )
        singular_values = np.linalg.svd(transform[:3, :3], compute_uv=False)
        if (
            not np.all(np.isfinite(singular_values))
            or singular_values[-1]
            <= np.finfo(float).eps * 64.0 * singular_values[0]
        ):
            raise QuadricAlgebraError("affine transform must be invertible")
        inverse = np.linalg.inv(transform)
        result = inverse.T @ self.array @ inverse
        return HomogeneousQuadric(_canonical_matrix(result))

    def evaluate(self, point: Sequence[float]) -> float:
        value = _vector3(point, "quadric point")
        homogeneous = np.append(value, 1.0)
        return float(homogeneous @ self.array @ homogeneous)

    def gradient(self, point: Sequence[float]) -> np.ndarray:
        value = _vector3(point, "quadric point")
        homogeneous = np.append(value, 1.0)
        return 2.0 * (self.array @ homogeneous)[:3]

    def ray_coefficients(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
    ) -> tuple[float, float, float]:
        """Return ``a, b, c`` for ``evaluate(origin + t*direction)``."""

        point = _vector3(origin, "ray origin")
        vector = _vector3(direction, "ray direction")
        if float(np.linalg.norm(vector)) <= 0.0:
            raise QuadricAlgebraError("ray direction must be non-zero")
        homogeneous_origin = np.append(point, 1.0)
        homogeneous_direction = np.append(vector, 0.0)
        matrix = self.array
        return (
            float(homogeneous_direction @ matrix @ homogeneous_direction),
            float(2.0 * homogeneous_origin @ matrix @ homogeneous_direction),
            float(homogeneous_origin @ matrix @ homogeneous_origin),
        )

    def real_ray_parameters(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        *,
        context: GeometryContext | ResolvedGeometryContext | None = None,
    ) -> tuple[float, ...]:
        """Return sorted real roots of one ray/quadric equation.

        This is the low-degree primitive needed by finite entity contracts.
        The dedicated quadrics root layer may add certified isolation for
        higher-degree critical equations; it must not change this method's
        scale-normalized coincident-ray behavior.
        """

        point = _vector3(origin, "ray origin")
        vector = _vector3(direction, "ray direction")
        if float(np.linalg.norm(vector)) <= 0.0:
            raise QuadricAlgebraError("ray direction must be non-zero")
        resolved = _resolved_context(
            context,
            (point, point + vector),
        )
        coefficients = np.asarray(self.ray_coefficients(point, vector), dtype=float)
        scale = float(np.max(np.abs(coefficients)))
        if scale == 0.0:
            raise CoincidentRayError("ray lies entirely on the quadric surface")
        a, b, c = (float(item) for item in coefficients / scale)
        relative_epsilon = max(
            np.finfo(float).eps * 128.0,
            resolved.epsilon(GeometryQuantity.ANGULAR),
        )
        if abs(a) <= relative_epsilon:
            if abs(b) <= relative_epsilon:
                if abs(c) <= relative_epsilon:
                    raise CoincidentRayError(
                        "ray lies entirely on the quadric surface"
                    )
                return ()
            return (float(-c / b),)

        discriminant = b * b - 4.0 * a * c
        discriminant_scale = max(
            abs(b * b),
            abs(4.0 * a * c),
            relative_epsilon,
        )
        discriminant_epsilon = relative_epsilon * discriminant_scale
        if discriminant < -discriminant_epsilon:
            return ()
        if abs(discriminant) <= discriminant_epsilon:
            return (float(-b / (2.0 * a)),)

        square_root = float(np.sqrt(discriminant))
        q = -0.5 * (b + np.copysign(square_root, b))
        if q == 0.0:
            roots = (-square_root / (2.0 * a), square_root / (2.0 * a))
        else:
            roots = (q / a, c / q)
        return tuple(sorted(float(item) for item in roots))

    def restrict_to_affine_plane(
        self,
        origin: Sequence[float],
        u_axis: Sequence[float],
        v_axis: Sequence[float],
    ) -> np.ndarray:
        """Return the 3x3 conic form in coordinates ``origin + u*u + v*v``."""

        point = _vector3(origin, "plane origin")
        first = _vector3(u_axis, "plane u_axis")
        second = _vector3(v_axis, "plane v_axis")
        if float(np.linalg.norm(np.cross(first, second))) <= 0.0:
            raise QuadricAlgebraError("plane axes must be linearly independent")
        embedding = np.asarray(
            (
                (first[0], second[0], point[0]),
                (first[1], second[1], point[1]),
                (first[2], second[2], point[2]),
                (0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        result = embedding.T @ self.array @ embedding
        return 0.5 * (result + result.T)


__all__ = [
    "AffineFrame3D",
    "CoincidentRayError",
    "HomogeneousQuadric",
    "QuadricAlgebraError",
]
