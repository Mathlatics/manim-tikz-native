"""Renderer-neutral semantic authoring for static parallel-camera shots.

This module intentionally stops at immutable authoring data.  It does not own a
timeline player, Manim camera, geometry registry, or occlusion controller.  A
shot stores one complete :class:`ParallelCameraState`; consumers decide how and
when to interpolate between consecutive shots.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import Sequence

import numpy as np

from .parallel_camera import CameraPlane, ParallelCameraState, PlaneLike


PARALLEL_CAMERA_SHOT_SEQUENCE_SCHEMA = "parallel-shot-sequence/v1"
_FIT_TOLERANCE_FACTOR = 4096.0


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


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


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _nonnegative_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class ParallelCameraSafeFrame:
    """Explicit screen rectangle inside which fitted points must remain."""

    left: float
    right: float
    bottom: float
    top: float

    def __post_init__(self) -> None:
        values = tuple(
            _finite_float(getattr(self, name), f"safe_frame.{name}")
            for name in ("left", "right", "bottom", "top")
        )
        left, right, bottom, top = values
        if left >= right or bottom >= top:
            raise ValueError(
                "safe frame must satisfy left < right and bottom < top"
            )
        for name, value in zip(
            ("left", "right", "bottom", "top"),
            values,
        ):
            object.__setattr__(self, name, value)

    def contains(
        self,
        point: Sequence[float],
        *,
        tolerance: float = 0.0,
    ) -> bool:
        value = np.asarray(point, dtype=float)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("screen point must contain two finite values")
        epsilon = _nonnegative_float(tolerance, "tolerance")
        return bool(
            self.left - epsilon <= value[0] <= self.right + epsilon
            and self.bottom - epsilon <= value[1] <= self.top + epsilon
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "right": self.right,
            "bottom": self.bottom,
            "top": self.top,
        }


def _state_to_dict(state: ParallelCameraState) -> dict[str, object]:
    return {
        "matrix": [
            [float(value) for value in row]
            for row in np.asarray(state.matrix, dtype=float)
        ],
        "target": [float(value) for value in state.target],
        "screenAnchor": [float(value) for value in state.screen_anchor],
        "zoom": state.zoom,
    }


@dataclass(frozen=True, slots=True)
class ParallelCameraShot:
    """One static semantic camera destination plus its transition metadata."""

    id: str
    state: ParallelCameraState
    duration: float = 1.0
    hold: float = 0.0
    transition: str = "orbit"
    arc_height: float = 0.85
    cue: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identity(self.id, "shot id"))
        if not isinstance(self.state, ParallelCameraState):
            raise TypeError("state must be a ParallelCameraState")
        duration = _positive_float(self.duration, "duration")
        hold = _nonnegative_float(self.hold, "hold")
        if self.transition not in {"orbit", "shortest"}:
            raise ValueError("transition must be 'orbit' or 'shortest'")
        arc_height = _finite_float(self.arc_height, "arc_height")
        if self.transition == "orbit" and abs(arc_height) <= 1.0e-12:
            raise ValueError("orbit arc_height must be non-zero")
        cue = None if self.cue is None else _identity(self.cue, "cue")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "hold", hold)
        object.__setattr__(self, "arc_height", arc_height)
        object.__setattr__(self, "cue", cue)

    @classmethod
    def look_at(
        cls,
        id: str,
        target: Sequence[float],
        *,
        view_direction: Sequence[float],
        up_hint: Sequence[float] | None = None,
        screen_anchor: Sequence[float] = (0.0, 0.0),
        zoom: float = 1.0,
        roll_degrees: float = 0.0,
        duration: float = 1.0,
        hold: float = 0.0,
        transition: str = "orbit",
        arc_height: float = 0.85,
        cue: str | None = None,
    ) -> "ParallelCameraShot":
        """Look at ``target`` from a parallel positive-depth direction."""

        state = ParallelCameraState.from_view_direction(
            view_direction,
            target=target,
            screen_anchor=screen_anchor,
            zoom=zoom,
            up_hint=up_hint,
            roll_degrees=roll_degrees,
        )
        return cls(
            id,
            state,
            duration,
            hold,
            transition,
            arc_height,
            cue,
        )

    @classmethod
    def normal_to_plane(
        cls,
        id: str,
        plane: PlaneLike | CameraPlane,
        *,
        side: str = "positive",
        target: Sequence[float] | None = None,
        screen_anchor: Sequence[float] = (0.0, 0.0),
        zoom: float = 1.0,
        roll_degrees: float = 0.0,
        duration: float = 1.0,
        hold: float = 0.0,
        transition: str = "orbit",
        arc_height: float = 0.85,
        cue: str | None = None,
    ) -> "ParallelCameraShot":
        state = ParallelCameraState.normal_to_plane(
            plane,
            side=side,
            target=target,
            screen_anchor=screen_anchor,
            zoom=zoom,
            roll_degrees=roll_degrees,
        )
        return cls(
            id,
            state,
            duration,
            hold,
            transition,
            arc_height,
            cue,
        )

    @classmethod
    def along_plane(
        cls,
        id: str,
        plane: PlaneLike | CameraPlane,
        *,
        direction: Sequence[float] | None = None,
        azimuth_degrees: float = 0.0,
        side: str = "positive",
        target: Sequence[float] | None = None,
        screen_anchor: Sequence[float] = (0.0, 0.0),
        zoom: float = 1.0,
        roll_degrees: float = 0.0,
        duration: float = 1.0,
        hold: float = 0.0,
        transition: str = "orbit",
        arc_height: float = 0.85,
        cue: str | None = None,
    ) -> "ParallelCameraShot":
        state = ParallelCameraState.along_plane(
            plane,
            direction=direction,
            azimuth_degrees=azimuth_degrees,
            side=side,
            target=target,
            screen_anchor=screen_anchor,
            zoom=zoom,
            roll_degrees=roll_degrees,
        )
        return cls(
            id,
            state,
            duration,
            hold,
            transition,
            arc_height,
            cue,
        )

    @classmethod
    def relative_to_plane(
        cls,
        id: str,
        plane: PlaneLike | CameraPlane,
        *,
        inclination_degrees: float,
        azimuth_degrees: float = 0.0,
        side: str = "positive",
        target: Sequence[float] | None = None,
        screen_anchor: Sequence[float] = (0.0, 0.0),
        zoom: float = 1.0,
        roll_degrees: float = 0.0,
        duration: float = 1.0,
        hold: float = 0.0,
        transition: str = "orbit",
        arc_height: float = 0.85,
        cue: str | None = None,
    ) -> "ParallelCameraShot":
        state = ParallelCameraState.relative_to_plane(
            plane,
            inclination_degrees=inclination_degrees,
            azimuth_degrees=azimuth_degrees,
            side=side,
            target=target,
            screen_anchor=screen_anchor,
            zoom=zoom,
            roll_degrees=roll_degrees,
        )
        return cls(
            id,
            state,
            duration,
            hold,
            transition,
            arc_height,
            cue,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "state": _state_to_dict(self.state),
            "duration": self.duration,
            "hold": self.hold,
            "transition": self.transition,
            "arcHeight": self.arc_height,
            "cue": self.cue,
        }


@dataclass(frozen=True, slots=True)
class ParallelCameraShotSequence:
    """Strict ordered sequence of uniquely identified static camera shots."""

    shots: tuple[ParallelCameraShot, ...]
    schema: str = PARALLEL_CAMERA_SHOT_SEQUENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PARALLEL_CAMERA_SHOT_SEQUENCE_SCHEMA:
            raise ValueError(
                "schema must be "
                f"{PARALLEL_CAMERA_SHOT_SEQUENCE_SCHEMA!r}"
            )
        shots = tuple(self.shots)
        if not shots or not all(isinstance(item, ParallelCameraShot) for item in shots):
            raise ValueError(
                "shots must be a non-empty sequence of ParallelCameraShot values"
            )
        ids = tuple(item.id for item in shots)
        if len(set(ids)) != len(ids):
            raise ValueError("parallel camera shot ids must be unique")
        object.__setattr__(self, "shots", shots)

    @property
    def total_duration(self) -> float:
        return sum(item.duration + item.hold for item in self.shots)

    def shot(self, shot_id: str) -> ParallelCameraShot:
        identity = _identity(shot_id, "shot id")
        matches = tuple(item for item in self.shots if item.id == identity)
        if len(matches) != 1:
            raise KeyError(f"unknown parallel camera shot: {identity!r}")
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "shots": [item.to_dict() for item in self.shots],
        }


def _fit_tolerance(values: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(values))))
    return np.finfo(float).eps * _FIT_TOLERANCE_FACTOR * scale


def _assert_points_inside_safe_frame(
    state: ParallelCameraState,
    points: np.ndarray,
    safe_frame: ParallelCameraSafeFrame,
) -> None:
    screen = state.project_points(points)[..., :2]
    tolerance = _fit_tolerance(screen)
    if not all(
        safe_frame.contains(point, tolerance=tolerance) for point in screen
    ):
        raise ValueError("fitted points do not lie inside the explicit safe frame")


def fit_points_to_parallel_camera_state(
    state: ParallelCameraState,
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    safe_frame: ParallelCameraSafeFrame,
    fallback_zoom: float | None = None,
) -> ParallelCameraState:
    """Return the largest zoom which fits points without moving target/anchor.

    The camera orientation, world target, and screen anchor remain unchanged.
    AREA and LINE screen point sets use the same directional-margin equations.
    A coincident screen set has no intrinsic extent and therefore requires an
    explicit ``fallback_zoom``.
    """

    if not isinstance(state, ParallelCameraState):
        raise TypeError("state must be a ParallelCameraState")
    if not isinstance(safe_frame, ParallelCameraSafeFrame):
        raise TypeError("safe_frame must be a ParallelCameraSafeFrame")
    try:
        values = np.asarray(points, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points must be a non-empty finite Nx3 array") from exc
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("points must be a non-empty finite Nx3 array")
    anchor = np.asarray(state.screen_anchor, dtype=float)
    if not (
        safe_frame.left < anchor[0] < safe_frame.right
        and safe_frame.bottom < anchor[1] < safe_frame.top
    ):
        raise ValueError(
            "screen_anchor must lie strictly inside the explicit safe frame"
        )

    unscaled = (values - state.target) @ state.matrix[:2].T
    screen_span = np.ptp(unscaled, axis=0)
    coincidence_tolerance = _fit_tolerance(unscaled)
    if float(np.max(screen_span)) <= coincidence_tolerance:
        if fallback_zoom is None:
            raise ValueError(
                "coincident projected points require an explicit fallback_zoom"
            )
        result = state.with_zoom(_positive_float(fallback_zoom, "fallback_zoom"))
        _assert_points_inside_safe_frame(result, values, safe_frame)
        return result

    margins = (
        float(anchor[0] - safe_frame.left),
        float(safe_frame.right - anchor[0]),
        float(anchor[1] - safe_frame.bottom),
        float(safe_frame.top - anchor[1]),
    )
    constraints: list[float] = []
    for coordinates in unscaled:
        x, y = (float(item) for item in coordinates)
        if x < 0.0:
            constraints.append(margins[0] / -x)
        elif x > 0.0:
            constraints.append(margins[1] / x)
        if y < 0.0:
            constraints.append(margins[2] / -y)
        elif y > 0.0:
            constraints.append(margins[3] / y)
    if not constraints:
        raise ValueError(
            "projected point set has no finite fit extent; provide fallback_zoom"
        )
    zoom = min(constraints)
    if not isfinite(zoom) or zoom <= 0.0:
        raise ValueError("point set cannot be fitted with a finite positive zoom")
    result = state.with_zoom(zoom)
    _assert_points_inside_safe_frame(result, values, safe_frame)
    return result


def canonical_parallel_camera_shot_sequence_json(
    sequence: ParallelCameraShotSequence,
) -> str:
    if not isinstance(sequence, ParallelCameraShotSequence):
        raise TypeError("sequence must be a ParallelCameraShotSequence")
    return json.dumps(
        sequence.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "PARALLEL_CAMERA_SHOT_SEQUENCE_SCHEMA",
    "ParallelCameraSafeFrame",
    "ParallelCameraShot",
    "ParallelCameraShotSequence",
    "canonical_parallel_camera_shot_sequence_json",
    "fit_points_to_parallel_camera_state",
]
