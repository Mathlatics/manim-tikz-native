from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_frame import (
    ParallelFrameBindingKind,
    ParallelFrameCoordinator,
    ParallelFrameCoordinatorError,
    ParallelFrameParticipant,
    ParallelFramePhase,
    ParallelFrameState,
)
from tikz_native.parallel_preflight import ParallelScreenTransform
from tikz_native.parallel_viewport import (
    PARALLEL_VIEWPORT_TRANSFORM_CHANNEL,
    ParallelViewportError,
    ParallelViewportState,
    parallel_viewport_frame_participant,
)


IDENTITY = np.identity(3)


def _state(
    *,
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    anchor: tuple[float, float] = (0.0, 0.0),
    zoom: float = 1.0,
) -> ParallelCameraState:
    return ParallelCameraState(
        IDENTITY,
        target=target,
        screen_anchor=anchor,
        zoom=zoom,
    )


def _assert_camera_state_equal(
    case: unittest.TestCase,
    actual: ParallelCameraState,
    expected: ParallelCameraState,
) -> None:
    np.testing.assert_array_equal(actual.matrix, expected.matrix)
    np.testing.assert_array_equal(actual.target, expected.target)
    np.testing.assert_array_equal(actual.screen_anchor, expected.screen_anchor)
    case.assertEqual(actual.zoom, expected.zoom)


class _Tracker:
    def __init__(self, value: float) -> None:
        self.value = value

    def get_value(self) -> float:
        return self.value


class _Camera:
    def __init__(
        self,
        state: ParallelCameraState,
        *,
        inherited_zoom: float = 1.0,
        frame_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.state = state
        self.inherited_zoom = inherited_zoom
        self.frame_center = np.asarray(frame_center, dtype=float)
        self.state_set_count = 0
        self.state_snapshot_count = 0
        self.zoom_set_count = 0

    def snapshot_parallel_state(self) -> ParallelCameraState:
        self.state_snapshot_count += 1
        return self.state

    def set_parallel_state(self, state: ParallelCameraState) -> None:
        if not isinstance(state, ParallelCameraState):
            raise TypeError("invalid camera state")
        self.state_set_count += 1
        self.state = state

    def get_zoom(self) -> float:
        return self.inherited_zoom

    def set_zoom(self, value: float) -> None:
        self.zoom_set_count += 1
        self.inherited_zoom = float(value)


class _TransactionalCamera(_Camera):
    def __init__(self, state: ParallelCameraState, **kwargs: object) -> None:
        super().__init__(state, **kwargs)  # type: ignore[arg-type]
        self.transition_tracker = _Tracker(0.4)
        self.mode = "authored-orbit"
        self.semantic_cache: object | None = None

    def snapshot_parallel_state(self) -> ParallelCameraState:
        value = super().snapshot_parallel_state()
        self.semantic_cache = ("resolved", self.state_snapshot_count)
        return value

    def set_parallel_state(self, state: ParallelCameraState) -> None:
        super().set_parallel_state(state)
        self.transition_tracker.value = 1.0
        self.mode = "direct"
        self.semantic_cache = None

    def snapshot_parallel_transaction(self) -> tuple[object, ...]:
        return (
            self.state,
            self.inherited_zoom,
            self.frame_center.copy(),
            self.transition_tracker.value,
            self.mode,
            self.state_snapshot_count,
            self.semantic_cache,
        )

    def restore_parallel_transaction(self, value: object) -> None:
        state, zoom, center, progress, mode, snapshots, cache = value  # type: ignore[misc]
        self.state = state
        self.inherited_zoom = zoom
        self.frame_center = np.asarray(center, dtype=float).copy()
        self.transition_tracker.value = progress
        self.mode = mode
        self.state_snapshot_count = snapshots
        self.semantic_cache = cache


class _DisplayOffset:
    def __init__(self, value: tuple[float, float]) -> None:
        self.value = value
        self.fail_after_next_write = False
        self.set_count = 0

    def get(self) -> tuple[float, float]:
        return self.value

    def set(self, value: tuple[float, float]) -> None:
        self.set_count += 1
        self.value = tuple(value)
        if self.fail_after_next_write:
            self.fail_after_next_write = False
            raise RuntimeError("display offset commit failed")


def _frame(viewport: ParallelViewportState) -> ParallelFrameState:
    return ParallelFrameState(
        viewport.camera,
        {
            PARALLEL_VIEWPORT_TRANSFORM_CHANNEL: viewport.screen_transform,
        },
    )


def _failing_paint_participant() -> ParallelFrameParticipant[object]:
    def fail(_prepared: object) -> None:
        raise RuntimeError("later paint failed")

    return ParallelFrameParticipant(
        participant_id="failing-paint",
        phase=ParallelFramePhase.PAINT,
        prepare=lambda _frame: None,
        snapshot=lambda: None,
        commit=fail,
        rollback=lambda _snapshot: None,
    )


class ParallelViewportStateTests(unittest.TestCase):
    def test_components_expose_only_scalar_positive_isotropic_zoom(self) -> None:
        camera = _state()
        viewport = ParallelViewportState.from_components(
            camera,
            inherited_zoom=1.75,
            frame_center=(2.0, -1.0),
            display_offset=(0.25, 0.5),
        )
        self.assertEqual(viewport.screen_transform.inherited_zoom, 1.75)
        self.assertEqual(viewport.screen_transform.frame_center, (2.0, -1.0))
        self.assertEqual(viewport.screen_transform.display_offset, (0.25, 0.5))
        self.assertNotIn("matrix", viewport.screen_transform.to_dict())

        for invalid in (
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            True,
            (1.0, 2.0),
            np.identity(2),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ParallelViewportState.from_components(
                        camera,
                        inherited_zoom=invalid,  # type: ignore[arg-type]
                    )

    def test_nonfinite_centers_offsets_and_invalid_types_fail(self) -> None:
        camera = _state()
        for kwargs in (
            {"frame_center": (float("nan"), 0.0)},
            {"frame_center": (0.0, float("inf"))},
            {"frame_center": (0.0, 1.0, 2.0)},
            {"display_offset": (0.0, float("nan"))},
            {"display_offset": (0.0,)},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ParallelViewportState.from_components(
                        camera,
                        **kwargs,  # type: ignore[arg-type]
                    )
        with self.assertRaisesRegex(TypeError, "ParallelCameraState"):
            ParallelViewportState(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "ParallelScreenTransform"):
            ParallelViewportState(camera, object())  # type: ignore[arg-type]

    def test_forged_screen_transform_is_revalidated(self) -> None:
        transform = ParallelScreenTransform()
        object.__setattr__(transform, "inherited_zoom", float("nan"))
        with self.assertRaises(ValueError):
            ParallelViewportState(_state(), transform)


class ParallelViewportParticipantTests(unittest.TestCase):
    def test_commit_and_restore_move_one_complete_viewport(self) -> None:
        initial = _state(target=(-1.0, 0.0, 2.0), anchor=(0.1, -0.2))
        target = _state(target=(2.0, 3.0, 4.0), anchor=(-0.4, 0.6), zoom=1.2)
        camera = _Camera(
            initial,
            inherited_zoom=0.75,
            frame_center=(4.0, -3.0, 7.0),
        )
        display = _DisplayOffset((-0.5, 0.25))
        viewport = ParallelViewportState.from_components(
            target,
            inherited_zoom=2.0,
            frame_center=(1.5, -2.0),
            display_offset=(0.4, 0.6),
        )
        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        participant = parallel_viewport_frame_participant(
            camera,
            display_offset_getter=display.get,
            display_offset_setter=display.set,
        )
        coordinator.add(participant)

        committed = coordinator.update(_frame(viewport))

        self.assertEqual(committed.participant_ids, ("parallel-viewport",))
        self.assertIs(participant.binding_kind, ParallelFrameBindingKind.CAMERA)
        _assert_camera_state_equal(self, camera.state, target)
        self.assertEqual(camera.inherited_zoom, 2.0)
        np.testing.assert_array_equal(camera.frame_center, (1.5, -2.0, 7.0))
        self.assertEqual(display.value, (0.4, 0.6))

        coordinator.restore()
        _assert_camera_state_equal(self, camera.state, initial)
        self.assertEqual(camera.inherited_zoom, 0.75)
        np.testing.assert_array_equal(camera.frame_center, (4.0, -3.0, 7.0))
        self.assertEqual(display.value, (-0.5, 0.25))

    def test_partial_display_commit_failure_restores_camera_and_offset(self) -> None:
        initial = _state(target=(1.0, 2.0, 3.0))
        camera = _Camera(
            initial,
            inherited_zoom=1.25,
            frame_center=(0.5, -0.25, 9.0),
        )
        display = _DisplayOffset((0.1, 0.2))
        display.fail_after_next_write = True
        participant = parallel_viewport_frame_participant(
            camera,
            display_offset_getter=display.get,
            display_offset_setter=display.set,
        )
        coordinator: ParallelFrameCoordinator[ParallelViewportState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(participant)
        target = ParallelViewportState.from_components(
            _state(target=(-3.0, 1.0, 0.0)),
            inherited_zoom=2.5,
            frame_center=(4.0, 5.0),
            display_offset=(-0.8, 0.7),
        )

        with self.assertRaisesRegex(RuntimeError, "display offset commit failed"):
            coordinator.update(target)

        _assert_camera_state_equal(self, camera.state, initial)
        self.assertEqual(camera.inherited_zoom, 1.25)
        np.testing.assert_array_equal(camera.frame_center, (0.5, -0.25, 9.0))
        self.assertEqual(display.value, (0.1, 0.2))
        self.assertFalse(coordinator.active)
        self.assertIsNone(coordinator.last_committed_frame)
        self.assertFalse(coordinator.poisoned)

    def test_failed_second_frame_restores_last_committed_viewport(self) -> None:
        initial = _state()
        camera = _Camera(
            initial,
            inherited_zoom=1.0,
            frame_center=(0.0, 0.0, 6.0),
        )
        display = _DisplayOffset((0.0, 0.0))
        participant = parallel_viewport_frame_participant(
            camera,
            display_offset_getter=display.get,
            display_offset_setter=display.set,
        )
        coordinator: ParallelFrameCoordinator[ParallelViewportState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(participant)
        first = ParallelViewportState.from_components(
            _state(target=(1.0, 2.0, 3.0)),
            inherited_zoom=1.5,
            frame_center=(0.5, -0.5),
            display_offset=(0.2, 0.4),
        )
        second = ParallelViewportState.from_components(
            _state(target=(8.0, 9.0, 10.0)),
            inherited_zoom=2.5,
            frame_center=(3.0, 4.0),
            display_offset=(-0.9, 0.8),
        )
        committed = coordinator.update(first)
        display.fail_after_next_write = True

        with self.assertRaisesRegex(RuntimeError, "display offset commit failed"):
            coordinator.update(second)

        self.assertIs(coordinator.last_committed_frame, committed)
        _assert_camera_state_equal(self, camera.state, first.camera)
        self.assertEqual(camera.inherited_zoom, 1.5)
        np.testing.assert_array_equal(camera.frame_center, (0.5, -0.5, 6.0))
        self.assertEqual(display.value, (0.2, 0.4))

    def test_later_participant_failure_rolls_back_complete_viewport(self) -> None:
        initial = _state()
        camera = _Camera(
            initial,
            inherited_zoom=1.0,
            frame_center=(0.0, 0.0, 3.0),
        )
        display = _DisplayOffset((0.0, 0.0))
        coordinator: ParallelFrameCoordinator[ParallelViewportState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(
            parallel_viewport_frame_participant(
                camera,
                display_offset_getter=display.get,
                display_offset_setter=display.set,
            )
        )
        coordinator.add(_failing_paint_participant())
        target = ParallelViewportState.from_components(
            _state(target=(1.0, 1.0, 1.0)),
            inherited_zoom=1.8,
            frame_center=(-2.0, 4.0),
            display_offset=(0.3, -0.6),
        )

        with self.assertRaisesRegex(RuntimeError, "later paint failed"):
            coordinator.update(target)

        _assert_camera_state_equal(self, camera.state, initial)
        self.assertEqual(camera.inherited_zoom, 1.0)
        np.testing.assert_array_equal(camera.frame_center, (0.0, 0.0, 3.0))
        self.assertEqual(display.value, (0.0, 0.0))

    def test_full_camera_transaction_restores_inflight_state(self) -> None:
        initial = _state(target=(0.5, 0.0, 0.0))
        camera = _TransactionalCamera(
            initial,
            inherited_zoom=1.1,
            frame_center=(0.2, 0.3, 0.4),
        )
        display = _DisplayOffset((0.6, -0.7))
        coordinator: ParallelFrameCoordinator[ParallelViewportState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(
            parallel_viewport_frame_participant(
                camera,
                display_offset_getter=display.get,
                display_offset_setter=display.set,
            )
        )
        coordinator.add(_failing_paint_participant())
        target = ParallelViewportState.from_components(
            _state(target=(9.0, 8.0, 7.0)),
            inherited_zoom=2.0,
            frame_center=(3.0, 4.0),
            display_offset=(-1.0, 1.0),
        )

        with self.assertRaisesRegex(RuntimeError, "later paint failed"):
            coordinator.update(target)

        _assert_camera_state_equal(self, camera.state, initial)
        self.assertEqual(camera.transition_tracker.value, 0.4)
        self.assertEqual(camera.mode, "authored-orbit")
        self.assertEqual(camera.state_snapshot_count, 0)
        self.assertIsNone(camera.semantic_cache)
        self.assertEqual(camera.inherited_zoom, 1.1)
        np.testing.assert_array_equal(camera.frame_center, (0.2, 0.3, 0.4))
        self.assertEqual(display.value, (0.6, -0.7))

    def test_later_snapshot_failure_leaves_full_camera_token_unchanged(
        self,
    ) -> None:
        camera = _TransactionalCamera(
            _state(target=(0.25, -0.5, 0.75)),
            inherited_zoom=1.3,
            frame_center=(-2.0, 0.4, 5.0),
        )
        display = _DisplayOffset((0.6, -0.2))
        before = camera.snapshot_parallel_transaction()

        def fail_snapshot() -> object:
            raise RuntimeError("later participant snapshot failed")

        coordinator: ParallelFrameCoordinator[ParallelViewportState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(
            parallel_viewport_frame_participant(
                camera,
                display_offset_getter=display.get,
                display_offset_setter=display.set,
            )
        )
        coordinator.add(
            ParallelFrameParticipant(
                participant_id="snapshot-failure",
                phase=ParallelFramePhase.PAINT,
                prepare=lambda _frame: None,
                snapshot=fail_snapshot,
                commit=lambda _prepared: None,
                rollback=lambda _snapshot: None,
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "later participant snapshot failed",
        ):
            coordinator.update(
                ParallelViewportState.from_components(
                    _state(target=(9.0, 8.0, 7.0)),
                    inherited_zoom=2.0,
                    frame_center=(3.0, 4.0),
                    display_offset=(-1.0, 1.0),
                )
            )

        after = camera.snapshot_parallel_transaction()
        self.assertIs(after[0], before[0])
        self.assertEqual(after[1], before[1])
        np.testing.assert_array_equal(after[2], before[2])
        self.assertEqual(after[3:], before[3:])
        self.assertEqual(camera.state_snapshot_count, 0)
        self.assertIsNone(camera.semantic_cache)
        self.assertEqual(display.value, (0.6, -0.2))
        self.assertEqual(display.set_count, 0)
        self.assertFalse(coordinator.active)
        self.assertFalse(coordinator.poisoned)

    def test_fallback_camera_rejects_nonstatic_boundary_before_mutation(self) -> None:
        camera = _Camera(_state())
        camera.transition_tracker = _Tracker(0.4)  # type: ignore[attr-defined]
        display = _DisplayOffset((0.0, 0.0))
        coordinator: ParallelFrameCoordinator[ParallelViewportState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(
            parallel_viewport_frame_participant(
                camera,
                display_offset_getter=display.get,
                display_offset_setter=display.set,
            )
        )

        with self.assertRaisesRegex(
            ParallelViewportError,
            "static camera frame boundary",
        ):
            coordinator.update(ParallelViewportState(_state(target=(1.0, 0.0, 0.0))))

        self.assertEqual(camera.state_set_count, 0)
        self.assertEqual(camera.zoom_set_count, 0)
        self.assertEqual(display.set_count, 0)

    def test_channel_and_custom_state_getter_are_strict(self) -> None:
        initial = _state()
        target = ParallelViewportState.from_components(
            _state(target=(1.0, 2.0, 3.0)),
            inherited_zoom=1.4,
            display_offset=(0.2, 0.3),
        )
        camera = _Camera(initial)
        display = _DisplayOffset((0.0, 0.0))
        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(
            parallel_viewport_frame_participant(
                camera,
                display_offset_getter=display.get,
                display_offset_setter=display.set,
            )
        )
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "no channel",
        ):
            coordinator.update(ParallelFrameState(target.camera))
        self.assertEqual(camera.state_set_count, 0)

        custom_camera = _Camera(initial)
        custom_display = _DisplayOffset((0.0, 0.0))
        custom: ParallelFrameCoordinator[object] = ParallelFrameCoordinator()
        custom.add(
            parallel_viewport_frame_participant(
                custom_camera,
                display_offset_getter=custom_display.get,
                display_offset_setter=custom_display.set,
                state_getter=lambda _frame: target,
                transform_channel="unused-channel",
            )
        )
        custom.update(object())
        _assert_camera_state_equal(self, custom_camera.state, target.camera)
        self.assertEqual(custom_display.value, (0.2, 0.3))

    def test_foreign_and_malformed_snapshots_fail_before_mutation(self) -> None:
        camera = _Camera(
            _state(),
            inherited_zoom=1.2,
            frame_center=(1.0, 2.0, 3.0),
        )
        display = _DisplayOffset((0.4, 0.5))
        first = parallel_viewport_frame_participant(
            camera,
            display_offset_getter=display.get,
            display_offset_setter=display.set,
        )
        second = parallel_viewport_frame_participant(
            camera,
            display_offset_getter=display.get,
            display_offset_setter=display.set,
            participant_id="second-viewport",
        )
        snapshot = first.snapshot()

        with self.assertRaisesRegex(ParallelViewportError, "foreign owner"):
            second.rollback(snapshot)
        malformed = replace(snapshot, inherited_zoom=float("nan"))
        with self.assertRaisesRegex(ParallelViewportError, "finite and positive"):
            first.rollback(malformed)
        forged_fallback = replace(
            snapshot.camera_snapshot,  # type: ignore[arg-type]
            inherited_zoom=9.0,
        )
        with self.assertRaisesRegex(
            ParallelViewportError,
            "fallback zoom differs",
        ):
            first.rollback(replace(snapshot, camera_snapshot=forged_fallback))

        self.assertEqual(camera.state_set_count, 0)
        self.assertEqual(camera.zoom_set_count, 0)
        np.testing.assert_array_equal(camera.frame_center, (1.0, 2.0, 3.0))
        self.assertEqual(display.value, (0.4, 0.5))

    def test_invalid_live_state_and_configuration_fail_closed(self) -> None:
        camera = _Camera(_state())
        display = _DisplayOffset((0.0, 0.0))
        with self.assertRaisesRegex(TypeError, "provided together"):
            class HalfTransactionCamera(_Camera):
                def snapshot_parallel_transaction(self) -> object:
                    return object()

            half = HalfTransactionCamera(_state())
            parallel_viewport_frame_participant(
                half,
                display_offset_getter=display.get,
                display_offset_setter=display.set,
            )
        with self.assertRaisesRegex(TypeError, "state_getter"):
            parallel_viewport_frame_participant(
                camera,
                display_offset_getter=display.get,
                display_offset_setter=display.set,
                state_getter=object(),  # type: ignore[arg-type]
            )

        participant = parallel_viewport_frame_participant(
            camera,
            display_offset_getter=display.get,
            display_offset_setter=display.set,
        )
        camera.inherited_zoom = float("nan")
        with self.assertRaisesRegex(ParallelViewportError, "finite and positive"):
            participant.snapshot()
        camera.inherited_zoom = 1.0
        camera.frame_center = np.asarray((0.0, float("nan"), 0.0))
        with self.assertRaisesRegex(ParallelViewportError, "frame_center"):
            participant.snapshot()
        camera.frame_center = np.zeros(3)
        display.value = (float("inf"), 0.0)
        with self.assertRaisesRegex(ParallelViewportError, "display_offset"):
            participant.snapshot()


if __name__ == "__main__":
    unittest.main()
