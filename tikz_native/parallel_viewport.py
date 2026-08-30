"""Atomic viewport transactions for renderer-neutral parallel-camera frames.

This module joins the semantic :class:`ParallelCameraState` with the narrow
renderer-level affine terms already described by :class:`ParallelScreenTransform`:

* one positive isotropic inherited zoom;
* an XY camera frame center; and
* one controller-local XY display offset.

It deliberately has no arbitrary 2x2 transform API.  A viewport participant
commits the camera and local display offset as one coordinated transaction, so
a later participant failure restores both to the same previous frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from .parallel_camera import ParallelCameraState
from .parallel_frame import (
    ParallelFrameBindingKind,
    ParallelFrameCoordinatorError,
    ParallelFrameParticipant,
    ParallelFramePhase,
    ParallelFrameState,
)
from .parallel_preflight import ParallelScreenTransform


PARALLEL_VIEWPORT_TRANSFORM_CHANNEL = "parallel-screen-transform"


class ParallelViewportError(ParallelFrameCoordinatorError):
    """A viewport state cannot be captured, committed, or restored safely."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParallelViewportError(f"{label} must be a non-empty string")
    return value.strip()


def _positive(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ParallelViewportError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelViewportError(
            f"{label} must be finite and positive"
        ) from exc
    if not isfinite(result) or result <= 0.0:
        raise ParallelViewportError(f"{label} must be finite and positive")
    return result


def _point2(value: object, label: str) -> tuple[float, float]:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelViewportError(
            f"{label} must contain two finite values"
        ) from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ParallelViewportError(f"{label} must contain two finite values")
    return (float(result[0]), float(result[1]))


def _frame_center(value: object) -> tuple[float, ...]:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelViewportError(
            "camera frame_center must contain two or three finite values"
        ) from exc
    if result.shape not in ((2,), (3,)) or not np.all(np.isfinite(result)):
        raise ParallelViewportError(
            "camera frame_center must contain two or three finite values"
        )
    return tuple(float(item) for item in result)


def _camera_states_equal(
    first: ParallelCameraState,
    second: ParallelCameraState,
) -> bool:
    return bool(
        np.array_equal(first.matrix, second.matrix)
        and np.array_equal(first.target, second.target)
        and np.array_equal(first.screen_anchor, second.screen_anchor)
        and first.zoom == second.zoom
    )


@dataclass(frozen=True, slots=True)
class ParallelViewportState:
    """One immutable semantic camera plus its narrow screen transform."""

    camera: ParallelCameraState
    screen_transform: ParallelScreenTransform = field(
        default_factory=ParallelScreenTransform
    )

    def __post_init__(self) -> None:
        if not isinstance(self.camera, ParallelCameraState):
            raise TypeError("camera must be a ParallelCameraState")
        if not isinstance(self.screen_transform, ParallelScreenTransform):
            raise TypeError(
                "screen_transform must be a ParallelScreenTransform"
            )
        # Reconstruct the narrow transform so even a forged/mutated dataclass
        # cannot smuggle non-finite or non-scalar affine terms into prepare().
        object.__setattr__(
            self,
            "screen_transform",
            ParallelScreenTransform(
                inherited_zoom=self.screen_transform.inherited_zoom,
                frame_center=self.screen_transform.frame_center,
                display_offset=self.screen_transform.display_offset,
            ),
        )

    @classmethod
    def from_components(
        cls,
        camera: ParallelCameraState,
        *,
        inherited_zoom: float = 1.0,
        frame_center: tuple[float, float] = (0.0, 0.0),
        display_offset: tuple[float, float] = (0.0, 0.0),
    ) -> "ParallelViewportState":
        """Build a viewport without exposing a general 2x2 transform."""

        return cls(
            camera,
            ParallelScreenTransform(
                inherited_zoom=inherited_zoom,
                frame_center=frame_center,
                display_offset=display_offset,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "camera": {
                "matrix": [list(row) for row in self.camera.matrix],
                "target": list(self.camera.target),
                "screenAnchor": list(self.camera.screen_anchor),
                "zoom": self.camera.zoom,
            },
            "screenTransform": self.screen_transform.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _FallbackCameraSnapshot:
    state: ParallelCameraState
    inherited_zoom: float
    frame_center: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ParallelViewportTransactionSnapshot:
    """Opaque snapshot owned by one viewport participant factory."""

    camera_snapshot: object
    camera_state: ParallelCameraState
    inherited_zoom: float
    frame_center: tuple[float, ...]
    display_offset: tuple[float, float]
    uses_full_camera_transaction: bool
    _owner_token: object = field(repr=False, compare=False)


def parallel_viewport_frame_participant(
    camera: object,
    *,
    display_offset_getter: Callable[[], object],
    display_offset_setter: Callable[[tuple[float, float]], None],
    participant_id: str = "parallel-viewport",
    phase: ParallelFramePhase = ParallelFramePhase.CAMERA,
    state_getter: Callable[[object], ParallelViewportState] | None = None,
    transform_channel: str = PARALLEL_VIEWPORT_TRANSFORM_CHANNEL,
) -> ParallelFrameParticipant[object]:
    """Adapt one camera and one local display offset as a single participant.

    By default, a :class:`ParallelFrameState` supplies the semantic camera and
    its ``parallel-screen-transform`` channel supplies the renderer terms.  A
    custom ``state_getter`` may instead return a complete
    :class:`ParallelViewportState`.

    Cameras may expose ``snapshot_parallel_transaction`` and
    ``restore_parallel_transaction`` for exact in-flight rollback.  Without
    those paired methods, the participant falls back to a static semantic
    camera snapshot plus the inherited zoom and complete frame-center vector.
    """

    name = _identity(participant_id, "participant_id")
    channel = _identity(transform_channel, "transform_channel")
    if state_getter is not None and not callable(state_getter):
        raise TypeError("state_getter must be callable or None")
    if not callable(display_offset_getter):
        raise TypeError("display_offset_getter must be callable")
    if not callable(display_offset_setter):
        raise TypeError("display_offset_setter must be callable")

    state_snapshot = getattr(camera, "snapshot_parallel_state", None)
    state_setter = getattr(camera, "set_parallel_state", None)
    zoom_getter = getattr(camera, "get_zoom", None)
    zoom_setter = getattr(camera, "set_zoom", None)
    exact_center_setter = getattr(camera, "set_parallel_frame_center_xy", None)
    if not callable(state_snapshot) or not callable(state_setter):
        raise TypeError(
            "camera must provide snapshot_parallel_state() and set_parallel_state()"
        )
    if not callable(zoom_getter) or not callable(zoom_setter):
        raise TypeError("camera must provide get_zoom() and set_zoom()")
    if not hasattr(camera, "frame_center"):
        raise TypeError("camera must expose frame_center")

    transaction_snapshot = getattr(camera, "snapshot_parallel_transaction", None)
    transaction_restore = getattr(camera, "restore_parallel_transaction", None)
    if callable(transaction_snapshot) != callable(transaction_restore):
        raise TypeError(
            "camera transaction snapshot and restore methods must be provided together"
        )
    use_full_transaction = callable(transaction_snapshot)
    owner_token = object()

    def require_static_boundary() -> None:
        if use_full_transaction:
            return
        readiness = getattr(camera, "parallel_transaction_ready", None)
        if callable(readiness):
            if not bool(readiness()):
                raise ParallelViewportError(
                    "viewport participant requires a static camera frame boundary"
                )
            return
        tracker = getattr(camera, "transition_tracker", None)
        get_value = getattr(tracker, "get_value", None)
        if callable(get_value):
            try:
                progress = float(get_value())
            except (TypeError, ValueError, OverflowError) as exc:
                raise ParallelViewportError(
                    "camera transition progress must be finite"
                ) from exc
            if not isfinite(progress):
                raise ParallelViewportError(
                    "camera transition progress must be finite"
                )
            if progress != 1.0:
                raise ParallelViewportError(
                    "viewport participant requires a static camera frame boundary"
                )

    def read_camera_state() -> ParallelCameraState:
        value = state_snapshot()
        if not isinstance(value, ParallelCameraState):
            raise TypeError(
                "snapshot_parallel_state() must return ParallelCameraState"
            )
        return value

    def read_zoom() -> float:
        return _positive(zoom_getter(), "camera inherited zoom")

    def read_frame_center() -> tuple[float, ...]:
        return _frame_center(getattr(camera, "frame_center"))

    def read_display_offset() -> tuple[float, float]:
        return _point2(display_offset_getter(), "display_offset")

    def set_frame_center_xy(value: tuple[float, float]) -> None:
        if callable(exact_center_setter):
            exact_center_setter(value)
            return
        current = np.asarray(read_frame_center(), dtype=float)
        current[:2] = np.asarray(value, dtype=float)
        frame_center_mobject = getattr(camera, "_frame_center", None)
        points = getattr(frame_center_mobject, "points", None)
        if (
            isinstance(points, np.ndarray)
            and points.shape == (1, current.shape[0])
        ):
            points[:] = current[np.newaxis, :]
        else:
            setattr(camera, "frame_center", current)

    def restore_frame_center(value: tuple[float, ...]) -> None:
        current = np.asarray(value, dtype=float)
        frame_center_mobject = getattr(camera, "_frame_center", None)
        points = getattr(frame_center_mobject, "points", None)
        if (
            isinstance(points, np.ndarray)
            and points.shape == (1, current.shape[0])
        ):
            points[:] = current[np.newaxis, :]
        else:
            setattr(camera, "frame_center", current)

    def resolve(frame: object) -> ParallelViewportState:
        require_static_boundary()
        if state_getter is not None:
            value = state_getter(frame)
        elif isinstance(frame, ParallelViewportState):
            value = frame
        elif isinstance(frame, ParallelFrameState):
            transform = frame.channel(channel)
            if not isinstance(transform, ParallelScreenTransform):
                raise TypeError(
                    "viewport transform channel must contain ParallelScreenTransform"
                )
            value = ParallelViewportState(frame.camera, transform)
        else:
            value = None
        if not isinstance(value, ParallelViewportState):
            raise TypeError(
                "viewport participant frame must resolve to ParallelViewportState"
            )
        return value

    def capture() -> ParallelViewportTransactionSnapshot:
        require_static_boundary()
        # Capture the opaque full token before resolving the visual state:
        # MultiProjectionCamera may populate an interpolation cache while
        # snapshot_parallel_state() is read, and rollback must still reproduce
        # the exact pre-read transaction state.
        camera_snapshot = transaction_snapshot() if use_full_transaction else None
        try:
            current_state = read_camera_state()
            current_zoom = read_zoom()
            current_center = read_frame_center()
            current_offset = read_display_offset()
        finally:
            if use_full_transaction:
                transaction_restore(camera_snapshot)
        if not use_full_transaction:
            camera_snapshot = _FallbackCameraSnapshot(
                current_state,
                current_zoom,
                current_center,
            )
        return ParallelViewportTransactionSnapshot(
            camera_snapshot=camera_snapshot,
            camera_state=current_state,
            inherited_zoom=current_zoom,
            frame_center=current_center,
            display_offset=current_offset,
            uses_full_camera_transaction=use_full_transaction,
            _owner_token=owner_token,
        )

    def verify(value: ParallelViewportState) -> None:
        current_state = read_camera_state()
        transform = value.screen_transform
        if not _camera_states_equal(current_state, value.camera):
            raise ParallelViewportError(
                "camera did not commit the prepared parallel state"
            )
        if read_zoom() != transform.inherited_zoom:
            raise ParallelViewportError(
                "camera did not commit the prepared inherited zoom"
            )
        if read_frame_center()[:2] != transform.frame_center:
            raise ParallelViewportError(
                "camera did not commit the prepared XY frame center"
            )
        if read_display_offset() != transform.display_offset:
            raise ParallelViewportError(
                "display target did not commit the prepared offset"
            )

    def apply(value: object) -> None:
        if not isinstance(value, ParallelViewportState):
            raise TypeError("prepared viewport state must be ParallelViewportState")
        transform = value.screen_transform
        state_setter(value.camera)
        zoom_setter(transform.inherited_zoom)
        set_frame_center_xy(transform.frame_center)
        display_offset_setter(transform.display_offset)
        verify(value)

    def validate_snapshot(
        value: object,
    ) -> ParallelViewportTransactionSnapshot:
        if not isinstance(value, ParallelViewportTransactionSnapshot):
            raise TypeError(
                "viewport snapshot must be a ParallelViewportTransactionSnapshot"
            )
        if value._owner_token is not owner_token:
            raise ParallelViewportError("viewport snapshot has a foreign owner")
        if not isinstance(value.uses_full_camera_transaction, bool):
            raise TypeError(
                "viewport snapshot transaction mode must be a bool"
            )
        if value.uses_full_camera_transaction != use_full_transaction:
            raise ParallelViewportError(
                "viewport snapshot camera transaction mode differs from its owner"
            )
        if not isinstance(value.camera_state, ParallelCameraState):
            raise TypeError("viewport snapshot camera_state is invalid")
        inherited_zoom = _positive(
            value.inherited_zoom,
            "snapshot inherited zoom",
        )
        if not isinstance(value.frame_center, tuple):
            raise TypeError("snapshot frame_center must be a canonical tuple")
        if not isinstance(value.display_offset, tuple):
            raise TypeError("snapshot display_offset must be a canonical tuple")
        center = _frame_center(value.frame_center)
        offset = _point2(value.display_offset, "snapshot display_offset")
        if center != value.frame_center:
            raise TypeError("snapshot frame_center must be a canonical tuple")
        if offset != value.display_offset:
            raise TypeError("snapshot display_offset must be a canonical tuple")
        if not use_full_transaction:
            if not isinstance(value.camera_snapshot, _FallbackCameraSnapshot):
                raise TypeError("viewport fallback camera snapshot is invalid")
            fallback = value.camera_snapshot
            if not isinstance(fallback.state, ParallelCameraState):
                raise TypeError("viewport fallback camera state is invalid")
            fallback_zoom = _positive(
                fallback.inherited_zoom,
                "fallback inherited zoom",
            )
            fallback_center = _frame_center(fallback.frame_center)
            if not _camera_states_equal(fallback.state, value.camera_state):
                raise ParallelViewportError(
                    "viewport fallback camera state differs from snapshot evidence"
                )
            if fallback_zoom != inherited_zoom:
                raise ParallelViewportError(
                    "viewport fallback zoom differs from snapshot evidence"
                )
            if fallback_center != center:
                raise ParallelViewportError(
                    "viewport fallback frame_center differs from snapshot evidence"
                )
        return value

    def rollback(value: object) -> None:
        snapshot = validate_snapshot(value)
        if use_full_transaction:
            transaction_restore(snapshot.camera_snapshot)
        else:
            fallback = snapshot.camera_snapshot
            assert isinstance(fallback, _FallbackCameraSnapshot)
            state_setter(fallback.state)
            zoom_setter(fallback.inherited_zoom)
            restore_frame_center(fallback.frame_center)
        display_offset_setter(snapshot.display_offset)

        # A full camera restore is already owner-validated and atomic.  Do not
        # call snapshot_parallel_state() afterward: MultiProjectionCamera may
        # repopulate an interpolation cache and thereby change the exact state
        # which its opaque transaction token just restored.
        if not use_full_transaction and not _camera_states_equal(
            read_camera_state(),
            snapshot.camera_state,
        ):
            raise ParallelViewportError(
                "camera rollback did not restore the captured parallel state"
            )
        if read_zoom() != snapshot.inherited_zoom:
            raise ParallelViewportError(
                "camera rollback did not restore inherited zoom"
            )
        if read_frame_center() != snapshot.frame_center:
            raise ParallelViewportError(
                "camera rollback did not restore frame_center"
            )
        if read_display_offset() != snapshot.display_offset:
            raise ParallelViewportError(
                "display rollback did not restore display_offset"
            )

    return ParallelFrameParticipant(
        participant_id=name,
        phase=phase,
        prepare=resolve,
        snapshot=capture,
        commit=apply,
        rollback=rollback,
        binding_kind=ParallelFrameBindingKind.CAMERA,
    )


__all__ = [
    "PARALLEL_VIEWPORT_TRANSFORM_CHANNEL",
    "ParallelViewportError",
    "ParallelViewportState",
    "ParallelViewportTransactionSnapshot",
    "parallel_viewport_frame_participant",
]
