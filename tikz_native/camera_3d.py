"""Portable multi-projection camera for TikZ Native 3D scenes.

This module is the provider-owned version of an earlier Manim projection-camera
prototype.  It intentionally has no dependency on an external checkout:
rendered scenes remain portable with the versioned provider alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from manim import ThreeDCamera, ValueTracker


R = np.sqrt(2.0) / 4.0
DEFAULT_FOCAL_DISTANCE = 8.0


@dataclass(frozen=True)
class ProjectionPreset:
    """One immutable camera projection state."""

    name: str
    matrix: np.ndarray
    perspective_strength: float = 0.0
    focal_distance: float = DEFAULT_FOCAL_DISTANCE
    view_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    principal_point: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=float, copy=True)
        view_center = np.array(self.view_center, dtype=float, copy=True)
        principal_point = np.array(self.principal_point, dtype=float, copy=True)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("ProjectionPreset.matrix must be a finite invertible 3x3 matrix")

        # Invertibility is a directional property.  Normalize every row before
        # checking the determinant so a valid TikZ projection is not rejected
        # merely because its authored screen units are tiny or huge.
        row_scales = np.max(np.abs(matrix), axis=1)
        if np.any(row_scales == 0.0) or not np.all(np.isfinite(row_scales)):
            raise ValueError("ProjectionPreset.matrix must be a finite invertible 3x3 matrix")
        normalized = matrix / row_scales[:, np.newaxis]
        row_norms = np.linalg.norm(normalized, axis=1)
        if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
            raise ValueError("ProjectionPreset.matrix must be a finite invertible 3x3 matrix")
        normalized /= row_norms[:, np.newaxis]
        determinant = float(np.linalg.det(normalized))
        if not np.isfinite(determinant) or abs(determinant) <= 1.0e-12:
            raise ValueError("ProjectionPreset.matrix must be a finite invertible 3x3 matrix")

        try:
            perspective_strength = float(self.perspective_strength)
            focal_distance = float(self.focal_distance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("projection scalar parameters must be finite numbers") from exc
        if not np.isfinite(perspective_strength) or not 0.0 <= perspective_strength <= 1.0:
            raise ValueError("perspective_strength must be a finite number inside [0, 1]")
        if not np.isfinite(focal_distance) or focal_distance <= 0.0:
            raise ValueError("focal_distance must be finite and positive")
        if view_center.shape != (3,) or not np.all(np.isfinite(view_center)):
            raise ValueError("view_center must be a finite 3D vector")
        if principal_point.shape != (2,) or not np.all(np.isfinite(principal_point)):
            raise ValueError("principal_point must be a finite 2D vector")

        # ``frozen=True`` does not make NumPy buffers immutable by itself.
        # Copy and freeze all arrays so caller-owned inputs and public presets
        # cannot silently mutate a camera state after validation.
        matrix.setflags(write=False)
        view_center.setflags(write=False)
        principal_point.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "perspective_strength", perspective_strength)
        object.__setattr__(self, "focal_distance", focal_distance)
        object.__setattr__(self, "view_center", view_center)
        object.__setattr__(self, "principal_point", principal_point)


FRONT_MATRIX = np.array(
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    dtype=float,
)
SIDE_MATRIX = np.array(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    dtype=float,
)
TOP_MATRIX = np.identity(3)
OBLIQUE_DIRECTION = np.array((1.0, R, R), dtype=float)
OBLIQUE_MATRIX = np.vstack(
    (
        np.array((-R, 1.0, 0.0)),
        np.array((-R, 0.0, 1.0)),
        OBLIQUE_DIRECTION / np.linalg.norm(OBLIQUE_DIRECTION),
    )
)
ISOMETRIC_MATRIX = np.array(
    (
        (-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0),
        (-1.0 / np.sqrt(6.0), -1.0 / np.sqrt(6.0), 2.0 / np.sqrt(6.0)),
        (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
    ),
    dtype=float,
)


DEFAULT_PRESETS: dict[str, ProjectionPreset] = {
    "front": ProjectionPreset("front", FRONT_MATRIX),
    "side": ProjectionPreset("side", SIDE_MATRIX),
    "top": ProjectionPreset("top", TOP_MATRIX),
    "oblique": ProjectionPreset("oblique", OBLIQUE_MATRIX),
    "isometric": ProjectionPreset("isometric", ISOMETRIC_MATRIX),
}


def _is_rotation_matrix(matrix: np.ndarray, atol: float = 1e-7) -> bool:
    return bool(
        np.allclose(matrix @ matrix.T, np.identity(3), atol=atol)
        and np.isclose(np.linalg.det(matrix), 1.0, atol=atol)
    )


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    length = float(np.linalg.norm(axis))
    if length <= 1e-12:
        raise ValueError("rotation axis must be nonzero")
    axis /= length
    x, y, z = axis
    cross_matrix = np.array(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=float,
    )
    return (
        np.identity(3)
        + np.sin(angle) * cross_matrix
        + (1.0 - np.cos(angle)) * (cross_matrix @ cross_matrix)
    )


def _rotation_slerp(source: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    relative = target @ source.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return source.copy()
    sine = float(np.sin(angle))
    if abs(sine) < 1e-7:
        values, vectors = np.linalg.eig(relative)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    else:
        axis = np.array(
            (
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            )
        ) / (2.0 * sine)
    return _axis_angle_rotation(axis, alpha * angle) @ source


def _frame_from_view_direction(
    view_direction: np.ndarray,
    horizontal_hint: np.ndarray,
) -> np.ndarray:
    normal = np.asarray(view_direction, dtype=float)
    normal /= np.linalg.norm(normal)
    horizontal = np.cross(np.array((0.0, 0.0, 1.0)), normal)
    if np.linalg.norm(horizontal) < 1e-7:
        hint = np.asarray(horizontal_hint, dtype=float)
        horizontal = hint - np.dot(hint, normal) * normal
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    vertical /= np.linalg.norm(vertical)
    return np.vstack((horizontal, vertical, normal))


def _orbit_control_matrix(
    source: np.ndarray,
    target: np.ndarray,
    arc_height: float,
) -> np.ndarray:
    bend = -np.cross(source[2], target[2])
    if np.linalg.norm(bend) < 1e-7:
        bend = source[1]
    bend /= np.linalg.norm(bend)
    direction = source[2] + target[2] + float(arc_height) * bend
    direction /= np.linalg.norm(direction)
    hint = source[0] + target[0]
    if np.linalg.norm(hint) < 1e-7:
        hint = source[0]
    return _frame_from_view_direction(direction, hint)


def _spherical_bezier_matrix(
    source: np.ndarray,
    control: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    first = _rotation_slerp(source, control, alpha)
    second = _rotation_slerp(control, target, alpha)
    return _rotation_slerp(first, second, alpha)


class MultiProjectionCamera(ThreeDCamera):
    """Cairo camera that interpolates between named engineering projections."""

    DIRECT_MODE_NAME = "__direct_projection__"

    def __init__(
        self,
        initial_mode: str = "oblique",
        presets: Mapping[str, ProjectionPreset] | None = None,
        **kwargs,
    ) -> None:
        self.presets = dict(DEFAULT_PRESETS if presets is None else presets)
        if initial_mode not in self.presets:
            raise KeyError(f"unknown projection mode: {initial_mode!r}")
        initial = self.presets[initial_mode]
        self.transition_tracker = ValueTracker(1.0)
        self._source_matrix = initial.matrix.copy()
        self._target_matrix = initial.matrix.copy()
        self._source_perspective = initial.perspective_strength
        self._target_perspective = initial.perspective_strength
        self._source_focal_distance = initial.focal_distance
        self._target_focal_distance = initial.focal_distance
        self._source_view_center = initial.view_center.copy()
        self._target_view_center = initial.view_center.copy()
        self._source_principal_point = initial.principal_point.copy()
        self._target_principal_point = initial.principal_point.copy()
        self._transition_style = "linear"
        self._control_matrix = initial.matrix.copy()
        self.current_mode = initial_mode
        self.target_mode = initial_mode
        kwargs.setdefault("focal_distance", DEFAULT_FOCAL_DISTANCE)
        kwargs.setdefault("should_apply_shading", False)
        kwargs.setdefault("exponential_projection", False)
        super().__init__(**kwargs)

    def get_value_trackers(self) -> list[ValueTracker]:
        return [*super().get_value_trackers(), self.transition_tracker]

    def _alpha(self) -> float:
        return float(np.clip(self.transition_tracker.get_value(), 0.0, 1.0))

    def get_projection_matrix(self) -> np.ndarray:
        alpha = self._alpha()
        if self._transition_style == "orbit":
            return _spherical_bezier_matrix(
                self._source_matrix,
                self._control_matrix,
                self._target_matrix,
                alpha,
            )
        return (1.0 - alpha) * self._source_matrix + alpha * self._target_matrix

    def get_perspective_strength(self) -> float:
        alpha = self._alpha()
        return (1.0 - alpha) * self._source_perspective + alpha * self._target_perspective

    def get_projection_focal_distance(self) -> float:
        alpha = self._alpha()
        return (
            (1.0 - alpha) * self._source_focal_distance
            + alpha * self._target_focal_distance
        )

    def get_view_center(self) -> np.ndarray:
        alpha = self._alpha()
        return (
            (1.0 - alpha) * self._source_view_center
            + alpha * self._target_view_center
        )

    def get_principal_point(self) -> np.ndarray:
        alpha = self._alpha()
        return (
            (1.0 - alpha) * self._source_principal_point
            + alpha * self._target_principal_point
        )

    def generate_rotation_matrix(self) -> np.ndarray:
        return self.get_projection_matrix()

    def get_rotation_matrix(self) -> np.ndarray:
        return self.get_projection_matrix()

    def reset_rotation_matrix(self) -> None:
        self.rotation_matrix = self.get_projection_matrix()

    def project_points(self, points: np.ndarray) -> np.ndarray:
        projected = np.array(points, dtype=float, copy=True)
        projected -= self.frame_center
        projected -= self.get_view_center()
        projected = projected @ self.get_projection_matrix().T
        perspective = self.get_perspective_strength()
        if perspective > 1e-12:
            focal = self.get_projection_focal_distance()
            denominator = focal - perspective * projected[:, 2]
            near_zero = np.abs(denominator) < 1e-6
            denominator[near_zero] = np.where(
                denominator[near_zero] < 0.0, -1e-6, 1e-6
            )
            projected[:, :2] *= np.clip(
                focal / denominator, -1e4, 1e4
            )[:, np.newaxis]
        projected[:, :2] += self.get_principal_point()
        projected[:, :2] *= self.get_zoom()
        return projected

    def register_mode(
        self,
        name: str,
        matrix: np.ndarray,
        *,
        perspective_strength: float = 0.0,
        focal_distance: float = DEFAULT_FOCAL_DISTANCE,
        view_center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        principal_point: np.ndarray | tuple[float, float] = (0.0, 0.0),
        overwrite: bool = False,
    ) -> ProjectionPreset:
        if name == self.DIRECT_MODE_NAME:
            raise ValueError(f"{name!r} is reserved")
        if name in self.presets and not overwrite:
            raise KeyError(f"projection mode {name!r} already exists")
        preset = ProjectionPreset(
            name=name,
            matrix=matrix,
            perspective_strength=perspective_strength,
            focal_distance=focal_distance,
            view_center=np.asarray(view_center, dtype=float),
            principal_point=np.asarray(principal_point, dtype=float),
        )
        self.presets[name] = preset
        return preset

    def snapshot(self) -> ProjectionPreset:
        """Freeze the exact current interpolated camera state."""

        return ProjectionPreset(
            name="__snapshot__",
            matrix=self.get_projection_matrix().copy(),
            perspective_strength=self.get_perspective_strength(),
            focal_distance=self.get_projection_focal_distance(),
            view_center=self.get_view_center().copy(),
            principal_point=self.get_principal_point().copy(),
        )

    def restore(self, preset: ProjectionPreset) -> None:
        """Restore a snapshot immediately without relying on a public mode name."""

        self.presets[self.DIRECT_MODE_NAME] = ProjectionPreset(
            self.DIRECT_MODE_NAME,
            preset.matrix,
            preset.perspective_strength,
            preset.focal_distance,
            preset.view_center,
            preset.principal_point,
        )
        self.set_mode(self.DIRECT_MODE_NAME)

    def set_mode(self, mode: str) -> None:
        if mode not in self.presets:
            raise KeyError(f"unknown projection mode: {mode!r}")
        preset = self.presets[mode]
        self._source_matrix = preset.matrix.copy()
        self._target_matrix = preset.matrix.copy()
        self._source_perspective = preset.perspective_strength
        self._target_perspective = preset.perspective_strength
        self._source_focal_distance = preset.focal_distance
        self._target_focal_distance = preset.focal_distance
        self._source_view_center = preset.view_center.copy()
        self._target_view_center = preset.view_center.copy()
        self._source_principal_point = preset.principal_point.copy()
        self._target_principal_point = preset.principal_point.copy()
        self._transition_style = "linear"
        self._control_matrix = preset.matrix.copy()
        self.transition_tracker.set_value(1.0)
        self.current_mode = mode
        self.target_mode = mode
        self.reset_rotation_matrix()

    def _prepare_transition(self, mode: str, transition: str, arc_height: float) -> None:
        if mode not in self.presets:
            available = ", ".join(sorted(self.presets))
            raise KeyError(f"unknown projection mode {mode!r}; available: {available}")
        source_matrix = self.get_projection_matrix().copy()
        target = self.presets[mode]
        if transition == "orbit":
            if not _is_rotation_matrix(source_matrix) or not _is_rotation_matrix(
                target.matrix
            ):
                raise ValueError("orbit endpoints must be right-handed orthogonal frames")
            control = _orbit_control_matrix(source_matrix, target.matrix, arc_height)
        elif transition == "linear":
            control = source_matrix.copy()
        else:
            raise ValueError("transition must be 'linear' or 'orbit'")
        self._source_matrix = source_matrix
        self._target_matrix = target.matrix.copy()
        self._control_matrix = control
        self._source_perspective = self.get_perspective_strength()
        self._target_perspective = target.perspective_strength
        self._source_focal_distance = self.get_projection_focal_distance()
        self._target_focal_distance = target.focal_distance
        self._source_view_center = self.get_view_center().copy()
        self._target_view_center = target.view_center.copy()
        self._source_principal_point = self.get_principal_point().copy()
        self._target_principal_point = target.principal_point.copy()
        self._transition_style = transition
        self.transition_tracker.set_value(0.0)
        self.current_mode = mode
        self.target_mode = mode

    def animate_to(self, mode: str):
        self._prepare_transition(mode, "linear", 0.0)
        return self.transition_tracker.animate.set_value(1.0)

    def animate_orbit_to(self, mode: str, arc_height: float = 0.85):
        self._prepare_transition(mode, "orbit", float(arc_height))
        return self.transition_tracker.animate.set_value(1.0)


__all__ = [
    "DEFAULT_PRESETS",
    "FRONT_MATRIX",
    "ISOMETRIC_MATRIX",
    "MultiProjectionCamera",
    "OBLIQUE_DIRECTION",
    "OBLIQUE_MATRIX",
    "ProjectionPreset",
    "R",
    "SIDE_MATRIX",
    "TOP_MATRIX",
]
