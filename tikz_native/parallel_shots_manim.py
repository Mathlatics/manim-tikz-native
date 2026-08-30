"""Manim playback binding for renderer-neutral parallel-camera shots.

The authoring objects in :mod:`tikz_native.parallel_shots` remain independent
of Manim.  This module is the deliberately small timeline adapter: it plays a
validated shot through :class:`MultiProjectionCamera` and optionally follows a
dynamic world-space target after the authored endpoint has been reached.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
from manim import Mobject

from .camera_3d import MultiProjectionCamera
from .parallel_camera import ParallelCameraState
from .parallel_shots import ParallelCameraShot, ParallelCameraShotSequence


ParallelCameraTargetProvider: TypeAlias = Callable[[], Sequence[float]]


class ParallelCameraShotManimError(RuntimeError):
    """A shot cannot be played or followed safely in the supplied Scene."""


def _camera_for_scene(scene: object) -> MultiProjectionCamera:
    camera = getattr(scene, "camera", None)
    if not isinstance(camera, MultiProjectionCamera):
        raise TypeError("scene.camera must be a MultiProjectionCamera")
    return camera


def _scene_method(scene: object, name: str) -> Callable[..., object]:
    value = getattr(scene, name, None)
    if not callable(value):
        raise TypeError(f"scene must provide a callable {name}() method")
    return value


def _states_are_exactly_equal(
    first: ParallelCameraState,
    second: ParallelCameraState,
) -> bool:
    return bool(
        np.array_equal(first.matrix, second.matrix)
        and np.array_equal(first.target, second.target)
        and np.array_equal(first.screen_anchor, second.screen_anchor)
        and first.zoom == second.zoom
    )


def _finite_target(value: object) -> np.ndarray:
    try:
        target = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "target_provider must return three finite values"
        ) from exc
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_provider must return three finite values")
    return target.copy()


def _restore_camera_after_failure(
    camera: MultiProjectionCamera,
    state: ParallelCameraState,
) -> None:
    """Best-effort rollback which never hides the original playback error."""

    try:
        camera.set_parallel_state(state)
    except Exception:
        # Both values were already validated ParallelCameraState objects.  A
        # renderer failure must remain the primary exception even if an
        # externally modified camera can no longer accept its saved state.
        pass


def play_parallel_camera_shot(
    scene: object,
    shot: ParallelCameraShot,
) -> ParallelCameraState:
    """Play one shot, including its hold, and commit its exact endpoint.

    The complete target state is re-applied after ``Scene.play`` and again
    after the hold.  If animation or waiting raises, the state present before
    the shot is restored instead of leaving an intermediate camera active.
    """

    if not isinstance(shot, ParallelCameraShot):
        raise TypeError("shot must be a ParallelCameraShot")
    camera = _camera_for_scene(scene)
    play = _scene_method(scene, "play")
    wait = _scene_method(scene, "wait")
    source = camera.snapshot_parallel_state()
    try:
        animation = camera.animate_to_parallel_state(
            shot.state,
            transition=shot.transition,
            arc_height=shot.arc_height,
        )
        play(animation, run_time=shot.duration)
        camera.set_parallel_state(shot.state)
        if shot.hold > 0.0:
            wait(shot.hold)
            camera.set_parallel_state(shot.state)
    except BaseException:
        _restore_camera_after_failure(camera, source)
        raise
    endpoint = camera.snapshot_parallel_state()
    if not _states_are_exactly_equal(endpoint, shot.state):
        _restore_camera_after_failure(camera, source)
        raise ParallelCameraShotManimError(
            f"parallel camera shot {shot.id!r} did not reach its exact endpoint"
        )
    return endpoint


def play_parallel_camera_shot_sequence(
    scene: object,
    sequence: ParallelCameraShotSequence,
) -> ParallelCameraState:
    """Play every shot in authored order and return the final exact state."""

    if not isinstance(sequence, ParallelCameraShotSequence):
        raise TypeError("sequence must be a ParallelCameraShotSequence")
    _camera_for_scene(scene)
    endpoint: ParallelCameraState | None = None
    for shot in sequence.shots:
        endpoint = play_parallel_camera_shot(scene, shot)
    assert endpoint is not None  # A valid sequence is non-empty by contract.
    return endpoint


class ParallelCameraTargetFollowController:
    """Follow one dynamic target with a single preallocated updater Mobject.

    ``start(shot)`` succeeds only while the camera is exactly at
    ``shot.state``.  This makes the lifecycle explicit: play the authored shot
    first, then enable target following.  Every updater frame derives from the
    immutable shot endpoint and changes only ``target``; matrix, screen anchor,
    and zoom therefore cannot drift.
    """

    def __init__(
        self,
        scene: object,
        target_provider: ParallelCameraTargetProvider,
    ) -> None:
        self.scene = scene
        self.camera = _camera_for_scene(scene)
        self._add_to_scene = _scene_method(scene, "add")
        self._remove_from_scene = _scene_method(scene, "remove")
        if not callable(target_provider):
            raise TypeError("target_provider must be callable")
        self.target_provider = target_provider
        self._driver = Mobject()
        self._active = False
        self._endpoint_state: ParallelCameraState | None = None
        self._shot_id: str | None = None

        def update_target(_mobject: Mobject, dt: float) -> None:
            del dt
            if not self._active:
                return
            try:
                self._apply_current_target()
            except BaseException:
                self._fail_closed()
                raise

        self._updater = update_target

    @property
    def active(self) -> bool:
        return self._active

    @property
    def driver_mobject(self) -> Mobject:
        return self._driver

    @property
    def endpoint_state(self) -> ParallelCameraState | None:
        return self._endpoint_state

    @property
    def shot_id(self) -> str | None:
        return self._shot_id

    def _apply_target(self, target: np.ndarray) -> None:
        endpoint = self._endpoint_state
        if endpoint is None:
            raise ParallelCameraShotManimError(
                "target follow has no authored shot endpoint"
            )
        self.camera.set_parallel_state(endpoint.with_target(target))

    def _apply_current_target(self) -> None:
        self._apply_target(_finite_target(self.target_provider()))

    def _remove_driver(self) -> None:
        self._driver.remove_updater(self._updater)
        try:
            self._remove_from_scene(self._driver)
        finally:
            self._active = False

    def _fail_closed(self) -> None:
        endpoint = self._endpoint_state
        try:
            self._remove_driver()
        finally:
            if endpoint is not None:
                _restore_camera_after_failure(self.camera, endpoint)

    def start(
        self,
        shot: ParallelCameraShot,
    ) -> "ParallelCameraTargetFollowController":
        """Attach after ``shot`` has reached its exact static endpoint."""

        if not isinstance(shot, ParallelCameraShot):
            raise TypeError("shot must be a ParallelCameraShot")
        if self._active:
            raise ParallelCameraShotManimError("target follow is already active")
        current = self.camera.snapshot_parallel_state()
        if not _states_are_exactly_equal(current, shot.state):
            raise ParallelCameraShotManimError(
                "target follow can start only after the shot endpoint is reached"
            )
        target = _finite_target(self.target_provider())
        self._endpoint_state = shot.state
        self._shot_id = shot.id
        self._driver.add_updater(self._updater)
        try:
            self._add_to_scene(self._driver)
            self._active = True
            self._apply_target(target)
        except BaseException:
            self._fail_closed()
            self._endpoint_state = None
            self._shot_id = None
            raise
        return self

    def stop(self) -> "ParallelCameraTargetFollowController":
        """Detach the updater while retaining the most recent dynamic target."""

        if self._active:
            self._remove_driver()
        return self

    def restore(self) -> "ParallelCameraTargetFollowController":
        """Detach and restore the exact authored endpoint captured by start."""

        endpoint = self._endpoint_state
        self.stop()
        if endpoint is not None:
            self.camera.set_parallel_state(endpoint)
        self._endpoint_state = None
        self._shot_id = None
        return self


__all__ = [
    "ParallelCameraShotManimError",
    "ParallelCameraTargetFollowController",
    "ParallelCameraTargetProvider",
    "play_parallel_camera_shot",
    "play_parallel_camera_shot_sequence",
]
