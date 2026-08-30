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

from .parallel_camera import (
    ParallelCameraState,
    interpolate_parallel_camera_states,
    orbit_control_matrix,
)

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
            raise ValueError(
                "ProjectionPreset.matrix must be a finite invertible 3x3 matrix"
            )

        # Invertibility is a directional property.  Normalize every row before
        # checking the determinant so a valid TikZ projection is not rejected
        # merely because its authored screen units are tiny or huge.
        row_scales = np.max(np.abs(matrix), axis=1)
        if np.any(row_scales == 0.0) or not np.all(np.isfinite(row_scales)):
            raise ValueError(
                "ProjectionPreset.matrix must be a finite invertible 3x3 matrix"
            )
        normalized = matrix / row_scales[:, np.newaxis]
        row_norms = np.linalg.norm(normalized, axis=1)
        if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
            raise ValueError(
                "ProjectionPreset.matrix must be a finite invertible 3x3 matrix"
            )
        normalized /= row_norms[:, np.newaxis]
        determinant = float(np.linalg.det(normalized))
        if not np.isfinite(determinant) or abs(determinant) <= 1.0e-12:
            raise ValueError(
                "ProjectionPreset.matrix must be a finite invertible 3x3 matrix"
            )

        try:
            perspective_strength = float(self.perspective_strength)
            focal_distance = float(self.focal_distance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "projection scalar parameters must be finite numbers"
            ) from exc
        if (
            not np.isfinite(perspective_strength)
            or not 0.0 <= perspective_strength <= 1.0
        ):
            raise ValueError(
                "perspective_strength must be a finite number inside [0, 1]"
            )
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


@dataclass(frozen=True, slots=True, eq=False)
class ParallelCameraTransactionSnapshot:
    """Opaque, exact rollback state for one ``MultiProjectionCamera``.

    Unlike :meth:`MultiProjectionCamera.snapshot`, this is not a portable
    visual preset.  It retains the in-flight interpolation endpoints, semantic
    parallel-camera sources, cached state, Manim trackers, and public mode
    names so a coordinated frame can be rolled back without being converted
    to ``__direct_projection__``.
    """

    source_matrix: np.ndarray
    target_matrix: np.ndarray
    control_matrix: np.ndarray
    source_perspective: float
    target_perspective: float
    source_focal_distance: float
    target_focal_distance: float
    source_view_center: np.ndarray
    target_view_center: np.ndarray
    source_principal_point: np.ndarray
    target_principal_point: np.ndarray
    transition_style: str
    parallel_state_active: bool
    source_parallel_state: ParallelCameraState | None
    target_parallel_state: ParallelCameraState | None
    parallel_control_matrix: np.ndarray | None
    parallel_state_cache_alpha: float | None
    parallel_state_cache: ParallelCameraState | None
    transition_progress: float
    current_mode: str
    target_mode: str
    rotation_matrix: np.ndarray
    frame_center: np.ndarray
    phi: float
    theta: float
    manim_focal_distance: float
    gamma: float
    manim_zoom: float
    _owner_token: object = field(repr=False)

    def __post_init__(self) -> None:
        array_fields = (
            "source_matrix",
            "target_matrix",
            "control_matrix",
            "source_view_center",
            "target_view_center",
            "source_principal_point",
            "target_principal_point",
            "rotation_matrix",
            "frame_center",
        )
        for name in array_fields:
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.parallel_control_matrix is not None:
            control = np.array(
                self.parallel_control_matrix,
                dtype=float,
                copy=True,
            )
            control.setflags(write=False)
            object.__setattr__(self, "parallel_control_matrix", control)


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


def _parallel_state_from_legacy_values(
    matrix: np.ndarray,
    target: np.ndarray,
    screen_anchor: np.ndarray,
) -> ParallelCameraState:
    """Move a common legacy screen scale into semantic ``zoom``."""

    value = np.asarray(matrix, dtype=float)
    log_norms = []
    for row in value[:2]:
        row_scale = float(np.max(np.abs(row)))
        if not np.isfinite(row_scale) or row_scale <= 0.0:
            raise ValueError("legacy camera screen rows must be finite and non-zero")
        normalized_norm = float(np.linalg.norm(row / row_scale))
        if not np.isfinite(normalized_norm) or normalized_norm <= 0.0:
            raise ValueError("legacy camera screen rows must be finite and non-zero")
        log_norms.append(float(np.log(row_scale) + np.log(normalized_norm)))
    screen_scale = float(np.exp(0.5 * (log_norms[0] + log_norms[1])))
    if not np.isfinite(screen_scale) or screen_scale <= 0.0:
        raise ValueError("legacy camera screen scale must be finite and positive")
    normalized_matrix = np.array(value, dtype=float, copy=True)
    normalized_matrix[:2] /= screen_scale
    return ParallelCameraState(
        normalized_matrix,
        target=np.asarray(target, dtype=float),
        screen_anchor=np.asarray(screen_anchor, dtype=float),
        zoom=screen_scale,
    )


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
        self.parallel_states: dict[str, ParallelCameraState] = {}
        self._parallel_transaction_owner_token = object()
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
        self._parallel_state_active = False
        self._source_parallel_state: ParallelCameraState | None = None
        self._target_parallel_state: ParallelCameraState | None = None
        self._parallel_control_matrix: np.ndarray | None = None
        self._parallel_state_cache_alpha: float | None = None
        self._parallel_state_cache: ParallelCameraState | None = None
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

    def _interpolated_parallel_state(self) -> ParallelCameraState:
        if (
            not self._parallel_state_active
            or self._source_parallel_state is None
            or self._target_parallel_state is None
        ):
            raise RuntimeError("no semantic parallel-camera state is active")
        alpha = self._alpha()
        if (
            self._parallel_state_cache_alpha == alpha
            and self._parallel_state_cache is not None
        ):
            return self._parallel_state_cache
        state = interpolate_parallel_camera_states(
            self._source_parallel_state,
            self._target_parallel_state,
            alpha,
            control_matrix=self._parallel_control_matrix,
        )
        self._parallel_state_cache_alpha = alpha
        self._parallel_state_cache = state
        return state

    def get_projection_matrix(self) -> np.ndarray:
        if self._parallel_state_active:
            return self._interpolated_parallel_state().matrix.copy()
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
        if self._parallel_state_active:
            return 0.0
        alpha = self._alpha()
        return (
            1.0 - alpha
        ) * self._source_perspective + alpha * self._target_perspective

    def get_projection_focal_distance(self) -> float:
        alpha = self._alpha()
        return (
            1.0 - alpha
        ) * self._source_focal_distance + alpha * self._target_focal_distance

    def get_view_center(self) -> np.ndarray:
        if self._parallel_state_active:
            return self._interpolated_parallel_state().target - self.frame_center
        alpha = self._alpha()
        return (
            1.0 - alpha
        ) * self._source_view_center + alpha * self._target_view_center

    def get_principal_point(self) -> np.ndarray:
        if self._parallel_state_active:
            zoom = float(ThreeDCamera.get_zoom(self))
            if zoom <= 0.0:
                raise ValueError("Manim camera zoom must be positive")
            return (
                self._interpolated_parallel_state().screen_anchor
                + self.frame_center[:2]
            ) / zoom
        alpha = self._alpha()
        return (
            1.0 - alpha
        ) * self._source_principal_point + alpha * self._target_principal_point

    def generate_rotation_matrix(self) -> np.ndarray:
        return self.get_projection_matrix()

    def get_rotation_matrix(self) -> np.ndarray:
        return self.get_projection_matrix()

    def reset_rotation_matrix(self) -> None:
        self.rotation_matrix = self.get_projection_matrix()

    def project_points(self, points: np.ndarray) -> np.ndarray:
        if self._parallel_state_active:
            state = self._interpolated_parallel_state()
            # ``ThreeDCamera.transform_points_pre_display`` has already
            # normalized non-finite geometry before this hot path.  Avoid a
            # second full scan of every Mobject's points in the state helper.
            projected = (
                np.asarray(points, dtype=float) - state.target
            ) @ state.matrix.T
            projected = np.array(projected, dtype=float, copy=True)
            projected[:, :2] *= state.zoom
            projected[:, :2] += state.screen_anchor
            zoom = float(ThreeDCamera.get_zoom(self))
            if zoom <= 0.0:
                raise ValueError("Manim camera zoom must be positive")
            projected[:, :2] = (
                self.frame_center[:2]
                + state.screen_anchor
                + zoom * (projected[:, :2] - state.screen_anchor)
            )
            return projected
        projected = np.array(points, dtype=float, copy=True)
        projected -= self.frame_center
        projected -= self.get_view_center()
        projected = projected @ self.get_projection_matrix().T
        perspective = self.get_perspective_strength()
        if perspective > 1e-12:
            focal = self.get_projection_focal_distance()
            denominator = focal - perspective * projected[:, 2]
            near_zero = np.abs(denominator) < 1e-6
            denominator[near_zero] = np.where(denominator[near_zero] < 0.0, -1e-6, 1e-6)
            projected[:, :2] *= np.clip(focal / denominator, -1e4, 1e4)[:, np.newaxis]
        projected[:, :2] += self.get_principal_point()
        projected[:, :2] *= self.get_zoom()
        return projected

    def register_parallel_state(
        self,
        name: str,
        state: ParallelCameraState,
        *,
        overwrite: bool = False,
    ) -> ParallelCameraState:
        """Register one renderer-neutral semantic parallel-camera state."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError("parallel camera state name must be a non-empty string")
        name = name.strip()
        if name == self.DIRECT_MODE_NAME:
            raise ValueError(f"{name!r} is reserved")
        if not isinstance(state, ParallelCameraState):
            raise TypeError("state must be a ParallelCameraState")
        if (name in self.parallel_states or name in self.presets) and not overwrite:
            raise KeyError(f"camera mode {name!r} already exists")
        if name in self.presets:
            del self.presets[name]
        self.parallel_states[name] = state
        return state

    def _resolve_parallel_state(
        self, state: ParallelCameraState | str
    ) -> tuple[str, ParallelCameraState]:
        if isinstance(state, ParallelCameraState):
            return self.DIRECT_MODE_NAME, state
        if not isinstance(state, str):
            raise TypeError("state must be a ParallelCameraState or registered name")
        try:
            return state, self.parallel_states[state]
        except KeyError as exc:
            available = ", ".join(sorted(self.parallel_states)) or "<none>"
            raise KeyError(
                f"unknown parallel camera state {state!r}; available: {available}"
            ) from exc

    def snapshot_parallel_state(self) -> ParallelCameraState:
        """Freeze the current view using final-screen anchor semantics."""

        if self._parallel_state_active:
            return self._interpolated_parallel_state()
        if self.get_perspective_strength() > 1.0e-12:
            raise ValueError(
                "a perspective camera cannot become a parallel camera state"
            )
        zoom = float(ThreeDCamera.get_zoom(self))
        if zoom <= 0.0:
            raise ValueError("Manim camera zoom must be positive")
        return _parallel_state_from_legacy_values(
            self.get_projection_matrix(),
            self.frame_center + self.get_view_center(),
            zoom * self.get_principal_point() - self.frame_center[:2],
        )

    def snapshot_parallel_transaction(self) -> ParallelCameraTransactionSnapshot:
        """Capture the complete mutable camera state used by one frame commit.

        The frame coordinator discovers this method by capability.  Capturing
        raw endpoints instead of a flattened visual preset is what preserves a
        registered ``current_mode``/``target_mode`` and an in-flight semantic
        orbit across either rollback or :meth:`ParallelFrameCoordinator.restore`.
        """

        return ParallelCameraTransactionSnapshot(
            source_matrix=self._source_matrix,
            target_matrix=self._target_matrix,
            control_matrix=self._control_matrix,
            source_perspective=self._source_perspective,
            target_perspective=self._target_perspective,
            source_focal_distance=self._source_focal_distance,
            target_focal_distance=self._target_focal_distance,
            source_view_center=self._source_view_center,
            target_view_center=self._target_view_center,
            source_principal_point=self._source_principal_point,
            target_principal_point=self._target_principal_point,
            transition_style=self._transition_style,
            parallel_state_active=self._parallel_state_active,
            source_parallel_state=self._source_parallel_state,
            target_parallel_state=self._target_parallel_state,
            parallel_control_matrix=self._parallel_control_matrix,
            parallel_state_cache_alpha=self._parallel_state_cache_alpha,
            parallel_state_cache=self._parallel_state_cache,
            transition_progress=float(self.transition_tracker.get_value()),
            current_mode=self.current_mode,
            target_mode=self.target_mode,
            rotation_matrix=self.rotation_matrix,
            frame_center=self.frame_center,
            phi=float(self.phi_tracker.get_value()),
            theta=float(self.theta_tracker.get_value()),
            manim_focal_distance=float(self.focal_distance_tracker.get_value()),
            gamma=float(self.gamma_tracker.get_value()),
            manim_zoom=float(self.zoom_tracker.get_value()),
            _owner_token=self._parallel_transaction_owner_token,
        )

    def restore_parallel_transaction(self, snapshot: object) -> None:
        """Restore an exact transaction snapshot without changing mode identity."""

        self._validate_parallel_transaction_snapshot(snapshot)
        assert isinstance(snapshot, ParallelCameraTransactionSnapshot)
        previous = self.snapshot_parallel_transaction()
        try:
            self._apply_parallel_transaction_snapshot_unchecked(snapshot)
        except BaseException as error:
            try:
                # Bypass an overridden/monkey-patched apply hook while
                # repairing the receiver after a partial write.
                MultiProjectionCamera._apply_parallel_transaction_snapshot_unchecked(
                    self,
                    previous,
                )
            except BaseException as rollback_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        "parallel camera transaction rollback also failed: "
                        f"{rollback_error!r}"
                    )
            raise

    def _validate_parallel_transaction_snapshot(self, snapshot: object) -> None:
        """Validate a transaction token completely without mutating the camera."""

        if not isinstance(snapshot, ParallelCameraTransactionSnapshot):
            raise TypeError("snapshot must be a ParallelCameraTransactionSnapshot")
        if snapshot._owner_token is not self._parallel_transaction_owner_token:
            raise ValueError("parallel camera transaction snapshot has a foreign owner")

        def require_array(name: str, shape: tuple[int, ...]) -> None:
            value = getattr(snapshot, name)
            if not isinstance(value, np.ndarray):
                raise TypeError(f"snapshot {name} must be a NumPy array")
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"snapshot {name} must be a finite array with shape {shape}"
                )

        def require_invertible_matrix(name: str) -> None:
            require_array(name, (3, 3))
            matrix = getattr(snapshot, name)
            # Keep this certification identical to ProjectionPreset: first
            # remove authored row scales, then normalize row norms before the
            # determinant check.  Valid very-small/very-large screen units are
            # accepted, while a directionally singular camera still fails.
            row_scales = np.max(np.abs(matrix), axis=1)
            if np.any(row_scales == 0.0) or not np.all(np.isfinite(row_scales)):
                raise ValueError(f"snapshot {name} must be invertible")
            normalized = matrix / row_scales[:, np.newaxis]
            row_norms = np.linalg.norm(normalized, axis=1)
            if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
                raise ValueError(f"snapshot {name} must be invertible")
            normalized /= row_norms[:, np.newaxis]
            determinant = float(np.linalg.det(normalized))
            if not np.isfinite(determinant) or abs(determinant) <= 1.0e-12:
                raise ValueError(f"snapshot {name} must be invertible")

        def finite_scalar(name: str) -> float:
            value = getattr(snapshot, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"snapshot {name} must be finite")
            try:
                result = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"snapshot {name} must be finite") from exc
            if not np.isfinite(result):
                raise ValueError(f"snapshot {name} must be finite")
            return result

        for name in (
            "source_matrix",
            "target_matrix",
            "control_matrix",
            "rotation_matrix",
        ):
            require_invertible_matrix(name)
        for name in (
            "source_view_center",
            "target_view_center",
            "frame_center",
        ):
            require_array(name, (3,))
        for name in (
            "source_principal_point",
            "target_principal_point",
        ):
            require_array(name, (2,))
        if snapshot.parallel_control_matrix is not None:
            require_invertible_matrix("parallel_control_matrix")

        for name in ("source_perspective", "target_perspective"):
            value = finite_scalar(name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"snapshot {name} must lie inside [0, 1]")
        for name in (
            "source_focal_distance",
            "target_focal_distance",
            "manim_focal_distance",
            "manim_zoom",
        ):
            if finite_scalar(name) <= 0.0:
                raise ValueError(f"snapshot {name} must be positive")
        for name in ("transition_progress", "phi", "theta", "gamma"):
            finite_scalar(name)
        if snapshot.parallel_state_cache_alpha is not None:
            cache_alpha = finite_scalar("parallel_state_cache_alpha")
            if not 0.0 <= cache_alpha <= 1.0:
                raise ValueError(
                    "snapshot parallel_state_cache_alpha must lie inside [0, 1]"
                )

        if snapshot.transition_style not in ("linear", "orbit"):
            raise ValueError(
                "snapshot transition_style must be 'linear' or 'orbit'"
            )
        if not isinstance(snapshot.parallel_state_active, bool):
            raise TypeError("snapshot parallel_state_active must be a bool")
        for name in ("source_parallel_state", "target_parallel_state"):
            value = getattr(snapshot, name)
            if value is not None and not isinstance(value, ParallelCameraState):
                raise TypeError(f"snapshot {name} must be ParallelCameraState or None")
        if snapshot.parallel_state_active:
            if not isinstance(snapshot.source_parallel_state, ParallelCameraState):
                raise ValueError(
                    "active semantic snapshot requires source_parallel_state"
                )
            if not isinstance(snapshot.target_parallel_state, ParallelCameraState):
                raise ValueError(
                    "active semantic snapshot requires target_parallel_state"
                )
        elif any(
            value is not None
            for value in (
                snapshot.source_parallel_state,
                snapshot.target_parallel_state,
                snapshot.parallel_control_matrix,
                snapshot.parallel_state_cache_alpha,
                snapshot.parallel_state_cache,
            )
        ):
            raise ValueError(
                "inactive semantic snapshot cannot retain parallel state or cache"
            )
        cache_pair = (
            snapshot.parallel_state_cache_alpha is not None,
            snapshot.parallel_state_cache is not None,
        )
        if cache_pair[0] != cache_pair[1]:
            raise ValueError(
                "snapshot parallel state cache and cache alpha must be paired"
            )
        if snapshot.parallel_state_cache is not None and not isinstance(
            snapshot.parallel_state_cache,
            ParallelCameraState,
        ):
            raise TypeError(
                "snapshot parallel_state_cache must be ParallelCameraState or None"
            )
        if snapshot.parallel_state_cache is not None:
            assert snapshot.parallel_state_cache_alpha is not None
            assert isinstance(snapshot.source_parallel_state, ParallelCameraState)
            assert isinstance(snapshot.target_parallel_state, ParallelCameraState)
            expected_cache = interpolate_parallel_camera_states(
                snapshot.source_parallel_state,
                snapshot.target_parallel_state,
                float(snapshot.parallel_state_cache_alpha),
                control_matrix=snapshot.parallel_control_matrix,
            )
            cached = snapshot.parallel_state_cache
            if not (
                np.array_equal(cached.matrix, expected_cache.matrix)
                and np.array_equal(cached.target, expected_cache.target)
                and np.array_equal(
                    cached.screen_anchor,
                    expected_cache.screen_anchor,
                )
                and cached.zoom == expected_cache.zoom
            ):
                raise ValueError(
                    "snapshot parallel state cache does not match its sources"
                )

        registered_modes = set(self.presets) | set(self.parallel_states)
        for name in ("current_mode", "target_mode"):
            mode = getattr(snapshot, name)
            if not isinstance(mode, str) or not mode.strip():
                raise ValueError(f"snapshot {name} must be a non-empty mode name")
            if mode != self.DIRECT_MODE_NAME and mode not in registered_modes:
                raise ValueError(
                    f"snapshot {name} refers to unregistered camera mode {mode!r}"
                )

    def _apply_parallel_transaction_snapshot_unchecked(
        self,
        snapshot: ParallelCameraTransactionSnapshot,
    ) -> None:
        """Apply one already-validated snapshot; caller owns atomic rollback."""

        self._source_matrix = snapshot.source_matrix.copy()
        self._target_matrix = snapshot.target_matrix.copy()
        self._control_matrix = snapshot.control_matrix.copy()
        self._source_perspective = snapshot.source_perspective
        self._target_perspective = snapshot.target_perspective
        self._source_focal_distance = snapshot.source_focal_distance
        self._target_focal_distance = snapshot.target_focal_distance
        self._source_view_center = snapshot.source_view_center.copy()
        self._target_view_center = snapshot.target_view_center.copy()
        self._source_principal_point = snapshot.source_principal_point.copy()
        self._target_principal_point = snapshot.target_principal_point.copy()
        self._transition_style = snapshot.transition_style
        self._parallel_state_active = snapshot.parallel_state_active
        self._source_parallel_state = snapshot.source_parallel_state
        self._target_parallel_state = snapshot.target_parallel_state
        self._parallel_control_matrix = (
            None
            if snapshot.parallel_control_matrix is None
            else snapshot.parallel_control_matrix.copy()
        )
        self._parallel_state_cache_alpha = snapshot.parallel_state_cache_alpha
        self._parallel_state_cache = snapshot.parallel_state_cache
        self.transition_tracker.set_value(snapshot.transition_progress)
        self.current_mode = snapshot.current_mode
        self.target_mode = snapshot.target_mode

        # ``frame_center = value`` is implemented as a relative Mobject move
        # and can retain round-off from the temporary state.  A transaction
        # restore must reproduce the captured coordinate bit-for-bit.
        self._frame_center.points = snapshot.frame_center[np.newaxis, :].copy()
        self.phi_tracker.set_value(snapshot.phi)
        self.theta_tracker.set_value(snapshot.theta)
        self.focal_distance_tracker.set_value(snapshot.manim_focal_distance)
        self.gamma_tracker.set_value(snapshot.gamma)
        self.zoom_tracker.set_value(snapshot.manim_zoom)
        self.rotation_matrix = snapshot.rotation_matrix.copy()

    def _parallel_state_from_preset(
        self, preset: ProjectionPreset
    ) -> ParallelCameraState:
        if preset.perspective_strength > 1.0e-12:
            raise ValueError(
                "a semantic parallel-camera orbit cannot target perspective"
            )
        zoom = float(ThreeDCamera.get_zoom(self))
        if zoom <= 0.0:
            raise ValueError("Manim camera zoom must be positive")
        return _parallel_state_from_legacy_values(
            preset.matrix,
            self.frame_center + preset.view_center,
            zoom * preset.principal_point - self.frame_center[:2],
        )

    def set_parallel_state(self, state: ParallelCameraState | str) -> None:
        """Apply one semantic parallel state immediately."""

        name, resolved = self._resolve_parallel_state(state)
        focal = self.get_projection_focal_distance()
        self._source_parallel_state = resolved
        self._target_parallel_state = resolved
        self._parallel_control_matrix = None
        self._parallel_state_active = True
        self._parallel_state_cache_alpha = None
        self._parallel_state_cache = None
        self._source_focal_distance = focal
        self._target_focal_distance = focal
        self.transition_tracker.set_value(1.0)
        self.current_mode = name
        self.target_mode = name
        self.reset_rotation_matrix()

    def animate_to_parallel_state(
        self,
        state: ParallelCameraState | str,
        *,
        transition: str = "orbit",
        arc_height: float = 0.85,
    ):
        """Prepare a safe parallel-state transition and return its animation."""

        name, target = self._resolve_parallel_state(state)
        source = self.snapshot_parallel_state()
        focal = self.get_projection_focal_distance()
        if transition == "orbit":
            control = (
                source.matrix
                if np.allclose(source.matrix, target.matrix, atol=1.0e-12, rtol=0.0)
                else orbit_control_matrix(
                    source.matrix,
                    target.matrix,
                    arc_height=float(arc_height),
                )
            )
        elif transition == "shortest":
            control = None
        else:
            raise ValueError("transition must be 'orbit' or 'shortest'")
        # Validate the complete interpolation family before the Scene starts
        # mutating.  In particular, a shortest 180-degree turn must fail here
        # and ask for an explicit orbit instead of failing halfway through.
        interpolate_parallel_camera_states(
            source,
            target,
            0.5,
            control_matrix=control,
        )
        self._source_parallel_state = source
        self._target_parallel_state = target
        self._parallel_control_matrix = control
        self._parallel_state_active = True
        self._parallel_state_cache_alpha = None
        self._parallel_state_cache = None
        self._source_focal_distance = focal
        self._target_focal_distance = focal
        self.transition_tracker.set_value(0.0)
        self.current_mode = name
        self.target_mode = name
        return self.transition_tracker.animate.set_value(1.0)

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
        if (name in self.presets or name in self.parallel_states) and not overwrite:
            raise KeyError(f"camera mode {name!r} already exists")
        if name in self.parallel_states:
            del self.parallel_states[name]
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

        if self._parallel_state_active:
            state = self._interpolated_parallel_state()
            zoom = float(ThreeDCamera.get_zoom(self))
            if zoom <= 0.0:
                raise ValueError("Manim camera zoom must be positive")
            matrix = state.matrix.copy()
            matrix[:2] *= state.zoom
            return ProjectionPreset(
                name="__snapshot__",
                matrix=matrix,
                perspective_strength=0.0,
                focal_distance=self.get_projection_focal_distance(),
                view_center=state.target - self.frame_center,
                principal_point=(state.screen_anchor + self.frame_center[:2]) / zoom,
            )
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
        self._parallel_state_active = False
        self._source_parallel_state = None
        self._target_parallel_state = None
        self._parallel_control_matrix = None
        self._parallel_state_cache_alpha = None
        self._parallel_state_cache = None
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

    def _prepare_transition(
        self, mode: str, transition: str, arc_height: float
    ) -> None:
        if mode not in self.presets:
            available = ", ".join(sorted(self.presets))
            raise KeyError(f"unknown projection mode {mode!r}; available: {available}")
        if self._parallel_state_active and transition == "orbit":
            source_state = self.snapshot_parallel_state()
            target_state = self._parallel_state_from_preset(self.presets[mode])
            control = (
                source_state.matrix
                if np.allclose(
                    source_state.matrix,
                    target_state.matrix,
                    atol=1.0e-12,
                    rtol=0.0,
                )
                else orbit_control_matrix(
                    source_state.matrix,
                    target_state.matrix,
                    arc_height=float(arc_height),
                )
            )
            interpolate_parallel_camera_states(
                source_state,
                target_state,
                0.5,
                control_matrix=control,
            )
            self._source_parallel_state = source_state
            self._target_parallel_state = target_state
            self._parallel_control_matrix = control
            self._parallel_state_cache_alpha = None
            self._parallel_state_cache = None
            self.transition_tracker.set_value(0.0)
            self.current_mode = mode
            self.target_mode = mode
            return
        source = self.snapshot()
        source_matrix = source.matrix.copy()
        target = self.presets[mode]
        if transition == "orbit":
            if not _is_rotation_matrix(source_matrix) or not _is_rotation_matrix(
                target.matrix
            ):
                raise ValueError(
                    "orbit endpoints must be right-handed orthogonal frames"
                )
            control = _orbit_control_matrix(source_matrix, target.matrix, arc_height)
        elif transition == "linear":
            control = source_matrix.copy()
        else:
            raise ValueError("transition must be 'linear' or 'orbit'")
        self._parallel_state_active = False
        self._source_parallel_state = None
        self._target_parallel_state = None
        self._parallel_control_matrix = None
        self._parallel_state_cache_alpha = None
        self._parallel_state_cache = None
        self._source_matrix = source_matrix
        self._target_matrix = target.matrix.copy()
        self._control_matrix = control
        self._source_perspective = source.perspective_strength
        self._target_perspective = target.perspective_strength
        self._source_focal_distance = source.focal_distance
        self._target_focal_distance = target.focal_distance
        self._source_view_center = source.view_center.copy()
        self._target_view_center = target.view_center.copy()
        self._source_principal_point = source.principal_point.copy()
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
    "ParallelCameraTransactionSnapshot",
    "ParallelCameraState",
    "ProjectionPreset",
    "R",
    "SIDE_MATRIX",
    "TOP_MATRIX",
]
