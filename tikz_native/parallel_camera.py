"""Renderer-neutral parallel-camera states and semantic view constructors.

The camera matrix uses the same convention as the visibility kernel: its rows
are screen-right, screen-up, and positive depth (from the target towards the
observer).  The complete screen transform is

``screen(point) = screen_anchor + zoom * matrix[:2] @ (point - target)``.

No Manim or quadric implementation is imported here.  Plane-aware constructors
accept any object exposing ``point``, ``normal``, and optional ``u_axis``
attributes, so geometry packages can opt in without becoming dependencies of
the camera core.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from math import cos, exp, isfinite, log, radians, sin
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

_VECTOR_TOLERANCE = 1.0e-12
_MATRIX_TOLERANCE = 1.0e-12
_ANGLE_TOLERANCE_DEGREES = 1.0e-12


@runtime_checkable
class PlaneLike(Protocol):
    """Structural input accepted by the plane-relative camera constructors."""

    point: Sequence[float]
    normal: Sequence[float]


class ProjectionRank(IntEnum):
    """Dimension retained by a world-space basis after screen projection."""

    POINT = 0
    LINE = 1
    AREA = 2


def _finite_vector(value: object, size: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain {size} finite values") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite values")
    return result.copy()


def _unit_vector(value: object, label: str) -> np.ndarray:
    result = _finite_vector(value, 3, label)
    length = float(np.linalg.norm(result))
    if length <= _VECTOR_TOLERANCE:
        raise ValueError(f"{label} must be non-zero")
    return result / length


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _frozen_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _canonical_sin_cos(degrees: object, label: str) -> tuple[float, float]:
    angle = _finite_float(degrees, label)
    quarter_turns = round(angle / 90.0)
    if abs(angle - 90.0 * quarter_turns) <= _ANGLE_TOLERANCE_DEGREES:
        sine_cycle = (0.0, 1.0, 0.0, -1.0)
        cosine_cycle = (1.0, 0.0, -1.0, 0.0)
        index = quarter_turns % 4
        return sine_cycle[index], cosine_cycle[index]
    value = radians(angle)
    return sin(value), cos(value)


def _side_sign(side: object) -> float:
    if side in ("positive", "+", 1, 1.0):
        return 1.0
    if side in ("negative", "-", -1, -1.0):
        return -1.0
    raise ValueError("side must be 'positive' or 'negative'")


def _stable_in_plane_axis(normal: np.ndarray) -> np.ndarray:
    candidates = (
        np.array((1.0, 0.0, 0.0)),
        np.array((0.0, 1.0, 0.0)),
        np.array((0.0, 0.0, 1.0)),
    )
    candidate = min(candidates, key=lambda item: abs(float(np.dot(item, normal))))
    projected = candidate - float(np.dot(candidate, normal)) * normal
    return projected / np.linalg.norm(projected)


def _rotation_matrix(axis: object, sine: float, cosine: float) -> np.ndarray:
    direction = _unit_vector(axis, "rotation axis")
    x, y, z = direction
    cross_matrix = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)
    return (
        np.identity(3)
        + sine * cross_matrix
        + (1.0 - cosine) * (cross_matrix @ cross_matrix)
    )


def _apply_roll(matrix: np.ndarray, roll_degrees: object) -> np.ndarray:
    sine, cosine = _canonical_sin_cos(roll_degrees, "roll_degrees")
    screen_rotation = np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )
    return screen_rotation @ matrix


@dataclass(frozen=True, slots=True)
class CameraPlane:
    """A dependency-free point and oriented orthonormal frame for one plane."""

    point: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray | None = None

    def __post_init__(self) -> None:
        point = _finite_vector(self.point, 3, "plane point")
        normal = _unit_vector(self.normal, "plane normal")
        if self.u_axis is None:
            u_axis = _stable_in_plane_axis(normal)
        else:
            authored = _finite_vector(self.u_axis, 3, "plane u_axis")
            projected = authored - float(np.dot(authored, normal)) * normal
            length = float(np.linalg.norm(projected))
            if length <= _VECTOR_TOLERANCE * max(1.0, float(np.linalg.norm(authored))):
                raise ValueError("plane u_axis must not be parallel to plane normal")
            u_axis = projected / length
        object.__setattr__(self, "point", _frozen_array(point))
        object.__setattr__(self, "normal", _frozen_array(normal))
        object.__setattr__(self, "u_axis", _frozen_array(u_axis))

    @classmethod
    def from_plane(cls, plane: PlaneLike | "CameraPlane") -> "CameraPlane":
        if isinstance(plane, cls):
            return plane
        try:
            point = plane.point
            normal = plane.normal
        except AttributeError as exc:
            raise TypeError(
                "plane must expose point, normal, and optional u_axis attributes"
            ) from exc
        return cls(point, normal, getattr(plane, "u_axis", None))

    @property
    def v_axis(self) -> np.ndarray:
        return _frozen_array(np.cross(self.normal, self.u_axis))

    @property
    def basis(self) -> np.ndarray:
        return _frozen_array(np.column_stack((self.u_axis, self.v_axis)))


def _normalized_camera_rows(matrix: np.ndarray) -> np.ndarray:
    row_scales = np.max(np.abs(matrix), axis=1)
    if np.any(row_scales == 0.0) or not np.all(np.isfinite(row_scales)):
        raise ValueError("camera matrix must be invertible and right-handed")
    normalized = matrix / row_scales[:, np.newaxis]
    row_norms = np.linalg.norm(normalized, axis=1)
    if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
        raise ValueError("camera matrix must be invertible and right-handed")
    return normalized / row_norms[:, np.newaxis]


def _validated_camera_matrix(value: object) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("camera matrix must be a finite 3x3 matrix") from exc
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera matrix must be a finite 3x3 matrix")
    normalized = _normalized_camera_rows(matrix)
    determinant = float(np.linalg.det(normalized))
    if not np.isfinite(determinant) or determinant <= _MATRIX_TOLERANCE:
        raise ValueError("camera matrix must be invertible and right-handed")
    return matrix.copy()


def frame_from_view_direction(
    view_direction: Sequence[float],
    *,
    up_hint: Sequence[float] | None = None,
    roll_degrees: float = 0.0,
) -> np.ndarray:
    """Build a right-handed orthonormal screen frame for one viewing direction."""

    depth = _unit_vector(view_direction, "view_direction")
    if up_hint is None:
        candidates = (
            np.array((0.0, 0.0, 1.0)),
            np.array((0.0, 1.0, 0.0)),
            np.array((1.0, 0.0, 0.0)),
        )
        authored_up = next(
            item
            for item in candidates
            if float(np.linalg.norm(item - np.dot(item, depth) * depth))
            > _VECTOR_TOLERANCE
        )
    else:
        authored_up = _finite_vector(up_hint, 3, "up_hint")
    projected_up = authored_up - float(np.dot(authored_up, depth)) * depth
    up_length = float(np.linalg.norm(projected_up))
    if up_length <= _VECTOR_TOLERANCE * max(1.0, float(np.linalg.norm(authored_up))):
        raise ValueError("up_hint must not be parallel to view_direction")
    up = projected_up / up_length
    right = np.cross(up, depth)
    right /= np.linalg.norm(right)
    up = np.cross(depth, right)
    matrix = np.vstack((right, up, depth))
    return _frozen_array(_apply_roll(matrix, roll_degrees))


@dataclass(frozen=True, slots=True)
class ParallelCameraState:
    """One immutable renderer-neutral affine parallel-camera state."""

    matrix: np.ndarray
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    screen_anchor: np.ndarray = field(default_factory=lambda: np.zeros(2))
    zoom: float = 1.0

    def __post_init__(self) -> None:
        matrix = _validated_camera_matrix(self.matrix)
        target = _finite_vector(self.target, 3, "target")
        anchor = _finite_vector(self.screen_anchor, 2, "screen_anchor")
        zoom = _positive_float(self.zoom, "zoom")
        object.__setattr__(self, "matrix", _frozen_array(matrix))
        object.__setattr__(self, "target", _frozen_array(target))
        object.__setattr__(self, "screen_anchor", _frozen_array(anchor))
        object.__setattr__(self, "zoom", zoom)

    @classmethod
    def from_view_direction(
        cls,
        view_direction: Sequence[float],
        *,
        target: Sequence[float] = (0.0, 0.0, 0.0),
        screen_anchor: Sequence[float] = (0.0, 0.0),
        zoom: float = 1.0,
        up_hint: Sequence[float] | None = None,
        roll_degrees: float = 0.0,
    ) -> "ParallelCameraState":
        return cls(
            frame_from_view_direction(
                view_direction,
                up_hint=up_hint,
                roll_degrees=roll_degrees,
            ),
            np.asarray(target, dtype=float),
            np.asarray(screen_anchor, dtype=float),
            zoom,
        )

    @classmethod
    def normal_to_plane(
        cls,
        plane: PlaneLike | CameraPlane,
        *,
        side: str = "positive",
        target: Sequence[float] | None = None,
        screen_anchor: Sequence[float] = (0.0, 0.0),
        zoom: float = 1.0,
        roll_degrees: float = 0.0,
    ) -> "ParallelCameraState":
        reference = CameraPlane.from_plane(plane)
        sign = _side_sign(side)
        matrix = np.vstack(
            (sign * reference.u_axis, reference.v_axis, sign * reference.normal)
        )
        return cls(
            _apply_roll(matrix, roll_degrees),
            reference.point if target is None else np.asarray(target, dtype=float),
            np.asarray(screen_anchor, dtype=float),
            zoom,
        )

    @classmethod
    def relative_to_plane(
        cls,
        plane: PlaneLike | CameraPlane,
        *,
        inclination_degrees: float,
        azimuth_degrees: float = 0.0,
        side: str = "positive",
        target: Sequence[float] | None = None,
        screen_anchor: Sequence[float] = (0.0, 0.0),
        zoom: float = 1.0,
        roll_degrees: float = 0.0,
    ) -> "ParallelCameraState":
        reference = CameraPlane.from_plane(plane)
        inclination = _finite_float(inclination_degrees, "inclination_degrees")
        if not 0.0 <= inclination <= 90.0:
            raise ValueError("inclination_degrees must lie inside [0, 90]")
        azimuth_sine, azimuth_cosine = _canonical_sin_cos(
            azimuth_degrees, "azimuth_degrees"
        )
        in_plane_direction = (
            azimuth_cosine * reference.u_axis + azimuth_sine * reference.v_axis
        )
        sign = _side_sign(side)
        base = np.vstack(
            (sign * reference.u_axis, reference.v_axis, sign * reference.normal)
        )
        inclination_sine, inclination_cosine = _canonical_sin_cos(
            inclination, "inclination_degrees"
        )
        if inclination_sine == 0.0:
            tilted = base
        else:
            axis = np.cross(base[2], in_plane_direction)
            rotation = _rotation_matrix(
                axis,
                inclination_sine,
                inclination_cosine,
            )
            tilted = base @ rotation.T
        return cls(
            _apply_roll(tilted, roll_degrees),
            reference.point if target is None else np.asarray(target, dtype=float),
            np.asarray(screen_anchor, dtype=float),
            zoom,
        )

    @classmethod
    def along_plane(
        cls,
        plane: PlaneLike | CameraPlane,
        *,
        direction: Sequence[float] | None = None,
        azimuth_degrees: float = 0.0,
        side: str = "positive",
        target: Sequence[float] | None = None,
        screen_anchor: Sequence[float] = (0.0, 0.0),
        zoom: float = 1.0,
        roll_degrees: float = 0.0,
    ) -> "ParallelCameraState":
        reference = CameraPlane.from_plane(plane)
        azimuth = _finite_float(azimuth_degrees, "azimuth_degrees")
        if direction is not None:
            if abs(azimuth) > _ANGLE_TOLERANCE_DEGREES:
                raise ValueError(
                    "direction and a non-zero azimuth_degrees cannot both be supplied"
                )
            authored = _unit_vector(direction, "direction")
            normal_component = float(np.dot(authored, reference.normal))
            if abs(normal_component) > _VECTOR_TOLERANCE:
                raise ValueError("direction must lie in the supplied plane")
            azimuth = float(
                np.degrees(
                    np.arctan2(
                        np.dot(authored, reference.v_axis),
                        np.dot(authored, reference.u_axis),
                    )
                )
            )
        return cls.relative_to_plane(
            reference,
            inclination_degrees=90.0,
            azimuth_degrees=azimuth,
            side=side,
            target=target,
            screen_anchor=screen_anchor,
            zoom=zoom,
            roll_degrees=roll_degrees,
        )

    @property
    def view_direction(self) -> np.ndarray:
        normalized = _normalized_camera_rows(self.matrix)
        direction = np.cross(normalized[0], normalized[1])
        direction /= np.linalg.norm(direction)
        return _frozen_array(direction)

    def project_points(
        self, points: Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.ndim < 1 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError(
                "points must have a final dimension of three finite values"
            )
        projected = (values - self.target) @ self.matrix.T
        projected = np.array(projected, dtype=float, copy=True)
        projected[..., :2] *= self.zoom
        projected[..., :2] += self.screen_anchor
        return projected

    def project_point(self, point: Sequence[float]) -> np.ndarray:
        value = _finite_vector(point, 3, "point")
        return self.project_points(value)

    def plane_screen_basis(self, plane: PlaneLike | CameraPlane) -> np.ndarray:
        reference = CameraPlane.from_plane(plane)
        return self.zoom * self.matrix[:2] @ reference.basis

    def plane_projection_rank(
        self,
        plane: PlaneLike | CameraPlane,
        *,
        tolerance: float = _MATRIX_TOLERANCE,
    ) -> ProjectionRank:
        threshold = _positive_float(tolerance, "tolerance")
        basis = self.plane_screen_basis(plane)
        singular_values = np.linalg.svd(basis, compute_uv=False)
        scale = max(float(singular_values[0]), np.finfo(float).tiny)
        rank = int(np.count_nonzero(singular_values > threshold * scale))
        return ProjectionRank(rank)

    def with_target(self, target: Sequence[float]) -> "ParallelCameraState":
        return replace(self, target=np.asarray(target, dtype=float))

    def with_screen_anchor(
        self, screen_anchor: Sequence[float]
    ) -> "ParallelCameraState":
        return replace(self, screen_anchor=np.asarray(screen_anchor, dtype=float))

    def with_zoom(self, zoom: float) -> "ParallelCameraState":
        return replace(self, zoom=zoom)


def _is_rotation_matrix(matrix: np.ndarray, tolerance: float = 1.0e-9) -> bool:
    return bool(
        np.allclose(matrix @ matrix.T, np.identity(3), atol=tolerance, rtol=0.0)
        and np.isclose(np.linalg.det(matrix), 1.0, atol=tolerance, rtol=0.0)
    )


def _rotation_slerp(source: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return source.copy()
    if alpha >= 1.0:
        return target.copy()
    relative = target @ source.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= 1.0e-10:
        blended = (1.0 - alpha) * source + alpha * target
        left, _singular, right = np.linalg.svd(blended)
        return left @ right
    if abs(np.pi - angle) <= 1.0e-8:
        raise ValueError(
            "a 180-degree camera transition needs an explicit orbit control frame"
        )
    sine = float(np.sin(angle))
    axis = np.array(
        (
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ),
        dtype=float,
    ) / (2.0 * sine)
    step = _rotation_matrix(axis, sin(alpha * angle), cos(alpha * angle))
    return step @ source


def _polar_projection_parts(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = _validated_camera_matrix(matrix)
    left, singular_values, right = np.linalg.svd(value)
    rotation = left @ right
    if not _is_rotation_matrix(rotation):
        raise ValueError("camera matrix has no right-handed polar rotation")
    stretch = right.T @ np.diag(singular_values) @ right
    return rotation, stretch


def _symmetric_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    if np.any(values <= 0.0):
        raise ValueError("camera stretch must be positive definite")
    return vectors @ np.diag(np.log(values)) @ vectors.T


def _symmetric_exp(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.exp(values)) @ vectors.T


def orbit_control_matrix(
    source: Sequence[Sequence[float]] | np.ndarray,
    target: Sequence[Sequence[float]] | np.ndarray,
    *,
    arc_height: float = 0.85,
) -> np.ndarray:
    """Return an explicit orthonormal via frame for a safe camera orbit."""

    source_rotation, _source_stretch = _polar_projection_parts(
        np.asarray(source, dtype=float)
    )
    target_rotation, _target_stretch = _polar_projection_parts(
        np.asarray(target, dtype=float)
    )
    height = _finite_float(arc_height, "arc_height")
    if abs(height) <= _VECTOR_TOLERANCE:
        raise ValueError("arc_height must be non-zero")
    bend = -np.cross(source_rotation[2], target_rotation[2])
    if float(np.linalg.norm(bend)) <= _VECTOR_TOLERANCE:
        # Opposite depth directions need an explicit great-circle side.  The
        # source screen-right axis produces a quarter-turn control view instead
        # of a frame which differs from the source by an ambiguous 180 degrees.
        bend = source_rotation[0].copy()
    bend /= np.linalg.norm(bend)
    direction = source_rotation[2] + target_rotation[2] + height * bend
    if float(np.linalg.norm(direction)) <= _VECTOR_TOLERANCE:
        raise ValueError("orbit control direction is ambiguous")
    up_hint = source_rotation[1] + target_rotation[1]
    projected_up = up_hint - np.dot(up_hint, direction) * direction / np.dot(
        direction, direction
    )
    if float(np.linalg.norm(projected_up)) <= _VECTOR_TOLERANCE:
        up_hint = source_rotation[2]
    return frame_from_view_direction(direction, up_hint=up_hint)


def interpolate_parallel_camera_states(
    source: ParallelCameraState,
    target: ParallelCameraState,
    alpha: float,
    *,
    control_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> ParallelCameraState:
    """Interpolate without allowing an intermediate projection to collapse."""

    if not isinstance(source, ParallelCameraState) or not isinstance(
        target, ParallelCameraState
    ):
        raise TypeError("source and target must be ParallelCameraState values")
    progress = _finite_float(alpha, "alpha")
    if not 0.0 <= progress <= 1.0:
        raise ValueError("alpha must lie inside [0, 1]")
    if progress == 0.0:
        return source
    if progress == 1.0:
        return target
    source_rotation, source_stretch = _polar_projection_parts(source.matrix)
    target_rotation, target_stretch = _polar_projection_parts(target.matrix)
    if control_matrix is None:
        rotation = _rotation_slerp(source_rotation, target_rotation, progress)
    else:
        control_rotation, _control_stretch = _polar_projection_parts(
            np.asarray(control_matrix, dtype=float)
        )
        first = _rotation_slerp(source_rotation, control_rotation, progress)
        second = _rotation_slerp(control_rotation, target_rotation, progress)
        rotation = _rotation_slerp(first, second, progress)
    stretch_log = (1.0 - progress) * _symmetric_log(
        source_stretch
    ) + progress * _symmetric_log(target_stretch)
    matrix = rotation @ _symmetric_exp(stretch_log)
    return ParallelCameraState(
        matrix,
        (1.0 - progress) * source.target + progress * target.target,
        (1.0 - progress) * source.screen_anchor + progress * target.screen_anchor,
        exp((1.0 - progress) * log(source.zoom) + progress * log(target.zoom)),
    )


__all__ = [
    "CameraPlane",
    "ParallelCameraState",
    "PlaneLike",
    "ProjectionRank",
    "frame_from_view_direction",
    "interpolate_parallel_camera_states",
    "orbit_control_matrix",
]
