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

_PARALLEL_CAMERA_STATE_CONSUMER_MARKER = (
    "_tikz_native_parallel_camera_state_consumer"
)


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


def _scene_mobjects(scene: object) -> list[Mobject]:
    mobjects = getattr(scene, "mobjects", None)
    if not isinstance(mobjects, list):
        raise TypeError("scene must expose its mutable mobjects list")
    return mobjects


def _place_driver_before_camera_state_consumers(
    scene: object,
    driver: Mobject,
) -> None:
    """Order ``driver`` before marked consumers without moving other objects."""

    mobjects = _scene_mobjects(scene)
    driver_indices = [
        index for index, mobject in enumerate(mobjects) if mobject is driver
    ]
    if len(driver_indices) != 1:
        raise ParallelCameraShotManimError(
            "target follow driver must be Scene-owned exactly once"
        )
    consumer_indices = [
        index
        for index, mobject in enumerate(mobjects)
        if mobject is not driver
        and bool(
            getattr(mobject, _PARALLEL_CAMERA_STATE_CONSUMER_MARKER, False)
        )
    ]
    if not consumer_indices:
        return
    driver_index = driver_indices[0]
    first_consumer_index = min(consumer_indices)
    if driver_index < first_consumer_index:
        return
    mobjects.pop(driver_index)
    mobjects.insert(first_consumer_index, driver)


def _remove_driver_identity_from_scene(scene: object, driver: Mobject) -> None:
    """Best-effort identity cleanup after Scene.remove partially fails."""

    for name in (
        "mobjects",
        "foreground_mobjects",
        "moving_mobjects",
        "static_mobjects",
    ):
        mobjects = getattr(scene, name, None)
        if isinstance(mobjects, list):
            mobjects[:] = [mobject for mobject in mobjects if mobject is not driver]


def _invalidate_cairo_static_image(scene: object) -> None:
    renderer = getattr(scene, "renderer", None)
    if renderer is not None and hasattr(renderer, "static_image"):
        renderer.static_image = None


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
    and zoom therefore cannot drift.  On start, the driver is placed after
    already-earlier target-producer Mobjects and immediately before the first
    marked quadric camera-state consumer, so both observe the new target in one
    frame.
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
        removal_error: BaseException | None = None
        try:
            self._driver.remove_updater(self._updater)
        except BaseException as exc:
            removal_error = exc
        try:
            self._remove_from_scene(self._driver)
        except BaseException as exc:
            if removal_error is None:
                removal_error = exc
            else:
                removal_error.add_note(
                    f"Scene.remove cleanup also failed: {exc!r}"
                )
        finally:
            _remove_driver_identity_from_scene(self.scene, self._driver)
            _invalidate_cairo_static_image(self.scene)
            self._active = False
        if removal_error is not None:
            raise removal_error

    def _fail_closed(self) -> None:
        endpoint = self._endpoint_state
        try:
            try:
                self._remove_driver()
            except BaseException:
                # Cleanup already performed an identity-based fallback.  Do not
                # hide the provider/playback exception which triggered rollback.
                pass
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
        try:
            self._driver.add_updater(self._updater)
            self._add_to_scene(self._driver)
            _place_driver_before_camera_state_consumers(
                self.scene,
                self._driver,
            )
            _invalidate_cairo_static_image(self.scene)
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
        cleanup_error: BaseException | None = None
        try:
            self.stop()
        except BaseException as exc:
            cleanup_error = exc
        try:
            if endpoint is not None:
                self.camera.set_parallel_state(endpoint)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        finally:
            _invalidate_cairo_static_image(self.scene)
            self._endpoint_state = None
            self._shot_id = None
        if cleanup_error is not None:
            raise cleanup_error
        return self


__all__ = [
    "ParallelCameraShotManimError",
    "ParallelCameraTargetFollowController",
    "ParallelCameraTargetProvider",
    "play_parallel_camera_shot",
    "play_parallel_camera_shot_sequence",
]
