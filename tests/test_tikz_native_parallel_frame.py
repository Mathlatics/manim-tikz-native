from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_frame import (
    ParallelFrameCoordinator,
    ParallelFrameCoordinatorError,
    ParallelFrameCoordinatorPoisonedError,
    ParallelFrameParticipant,
    ParallelFramePhase,
    ParallelFrameState,
    parallel_camera_frame_participant,
)


IDENTITY = np.identity(3)


@dataclass
class _MutableParticipant:
    participant_id: str
    phase: ParallelFramePhase
    value: int
    events: list[str]
    fail_prepare: bool = False
    fail_commit: bool = False
    fail_rollback: bool = False

    def binding(self) -> ParallelFrameParticipant[int]:
        def prepare(frame: int) -> int:
            self.events.append(f"prepare:{self.participant_id}:{frame}")
            if self.fail_prepare:
                raise RuntimeError(f"prepare failed: {self.participant_id}")
            return frame

        def snapshot() -> int:
            self.events.append(f"snapshot:{self.participant_id}:{self.value}")
            return self.value

        def commit(prepared: object) -> None:
            self.events.append(f"commit:{self.participant_id}:{prepared}")
            self.value = int(prepared)
            if self.fail_commit:
                raise RuntimeError(f"commit failed: {self.participant_id}")

        def rollback(snapshot: object) -> None:
            self.events.append(f"rollback:{self.participant_id}:{snapshot}")
            if self.fail_rollback:
                raise RuntimeError(f"rollback failed: {self.participant_id}")
            self.value = int(snapshot)

        return ParallelFrameParticipant(
            self.participant_id,
            self.phase,
            prepare,
            snapshot,
            commit,
            rollback,
        )


class _Camera:
    def __init__(self, state: ParallelCameraState) -> None:
        self.state = state
        self.history: list[ParallelCameraState] = []

    def snapshot_parallel_state(self) -> ParallelCameraState:
        return self.state

    def set_parallel_state(self, state: ParallelCameraState) -> None:
        self.history.append(state)
        self.state = state


class _Tracker:
    def __init__(self, value: float) -> None:
        self.value = value

    def get_value(self) -> float:
        return self.value


class _TransitioningCamera(_Camera):
    def __init__(self, state: ParallelCameraState, progress: float) -> None:
        super().__init__(state)
        self.transition_tracker = _Tracker(progress)


class ParallelFrameCoordinatorTests(unittest.TestCase):
    def test_sources_resolve_once_and_phases_override_registration_order(
        self,
    ) -> None:
        events: list[str] = []
        calls = 0
        coordinator: ParallelFrameCoordinator[int] = ParallelFrameCoordinator()

        def provide_shared() -> int:
            nonlocal calls
            calls += 1
            events.append("source:shared")
            return 7

        shared = coordinator.source("shared", provide_shared)
        paint = _MutableParticipant(
            "paint",
            ParallelFramePhase.PAINT,
            0,
            events,
        )
        camera = _MutableParticipant(
            "camera",
            ParallelFramePhase.CAMERA,
            0,
            events,
        )

        def participant_with_source(
            participant: _MutableParticipant,
        ) -> ParallelFrameParticipant[int]:
            binding = participant.binding()

            def prepare(frame: int) -> int:
                self.assertEqual(shared(), 7)
                return binding.prepare(frame)

            return ParallelFrameParticipant(
                binding.participant_id,
                binding.phase,
                prepare,
                binding.snapshot,
                binding.commit,
                binding.rollback,
            )

        coordinator.add(participant_with_source(paint))
        coordinator.add(participant_with_source(camera))
        result = coordinator.update(3)

        self.assertEqual(calls, 1)
        self.assertEqual(result.participant_ids, ("camera", "paint"))
        self.assertEqual(result.resolved_source_ids, ("shared",))
        self.assertEqual(coordinator.participant_ids, ("camera", "paint"))
        self.assertEqual(camera.value, 3)
        self.assertEqual(paint.value, 3)
        self.assertEqual(
            events,
            [
                "source:shared",
                "prepare:camera:3",
                "prepare:paint:3",
                "snapshot:camera:0",
                "snapshot:paint:0",
                "commit:camera:3",
                "commit:paint:3",
            ],
        )

    def test_frame_provider_is_called_once_inside_source_scope(self) -> None:
        provider_calls = 0
        source_calls = 0
        source_holder = {}

        def frame_provider() -> int:
            nonlocal provider_calls
            provider_calls += 1
            source = source_holder["value"]
            return source() + source()

        coordinator: ParallelFrameCoordinator[int]
        coordinator = ParallelFrameCoordinator(frame_provider)

        def source_provider() -> int:
            nonlocal source_calls
            source_calls += 1
            return 5

        source = coordinator.source("value", source_provider)
        source_holder["value"] = source
        participant = _MutableParticipant(
            "camera",
            ParallelFramePhase.CAMERA,
            0,
            [],
        )
        coordinator.add(participant.binding())

        result = coordinator.update()
        self.assertEqual(result.frame, 10)
        self.assertEqual(provider_calls, 1)
        self.assertEqual(source_calls, 1)

    def test_prepare_failure_mutates_nothing(self) -> None:
        events: list[str] = []
        first = _MutableParticipant(
            "first",
            ParallelFramePhase.GEOMETRY,
            1,
            events,
        )
        second = _MutableParticipant(
            "second",
            ParallelFramePhase.VISIBILITY,
            2,
            events,
            fail_prepare=True,
        )
        coordinator = ParallelFrameCoordinator[int]()
        coordinator.add(first.binding()).add(second.binding())

        with self.assertRaisesRegex(RuntimeError, "prepare failed: second"):
            coordinator.update(9)

        self.assertEqual((first.value, second.value), (1, 2))
        self.assertFalse(any(item.startswith("snapshot:") for item in events))
        self.assertFalse(any(item.startswith("commit:") for item in events))
        self.assertIsNone(coordinator.last_committed_frame)
        self.assertFalse(coordinator.active)
        self.assertFalse(coordinator.poisoned)

    def test_commit_failure_rolls_back_failing_and_prior_participants(self) -> None:
        events: list[str] = []
        first = _MutableParticipant(
            "first",
            ParallelFramePhase.GEOMETRY,
            1,
            events,
        )
        second = _MutableParticipant(
            "second",
            ParallelFramePhase.PAINT,
            2,
            events,
        )
        coordinator = ParallelFrameCoordinator[int]()
        coordinator.add(first.binding()).add(second.binding())
        committed = coordinator.update(4)
        self.assertEqual(committed.index, 0)
        second.fail_commit = True
        events.clear()

        with self.assertRaisesRegex(RuntimeError, "commit failed: second"):
            coordinator.update(8)

        self.assertEqual((first.value, second.value), (4, 4))
        self.assertEqual(
            events[-2:],
            ["rollback:second:4", "rollback:first:4"],
        )
        self.assertIs(coordinator.last_committed_frame, committed)
        self.assertFalse(coordinator.poisoned)

        second.fail_commit = False
        retried = coordinator.update(8)
        self.assertEqual(retried.index, 1)
        self.assertEqual((first.value, second.value), (8, 8))

    def test_rollback_failure_poison_closes_future_updates(self) -> None:
        events: list[str] = []
        first = _MutableParticipant(
            "first",
            ParallelFramePhase.GEOMETRY,
            1,
            events,
            fail_rollback=True,
        )
        second = _MutableParticipant(
            "second",
            ParallelFramePhase.PAINT,
            2,
            events,
            fail_commit=True,
        )
        coordinator = ParallelFrameCoordinator[int]()
        coordinator.add(first.binding()).add(second.binding())

        with self.assertRaisesRegex(RuntimeError, "commit failed: second") as raised:
            coordinator.update(6)

        self.assertTrue(coordinator.poisoned)
        self.assertTrue(
            any("rollback for participant 'first'" in note for note in raised.exception.__notes__)
        )
        with self.assertRaises(ParallelFrameCoordinatorPoisonedError):
            coordinator.update(7)

    def test_restore_returns_to_pre_first_frame_baseline_and_can_restart(self) -> None:
        events: list[str] = []
        first = _MutableParticipant(
            "first",
            ParallelFramePhase.CAMERA,
            1,
            events,
        )
        second = _MutableParticipant(
            "second",
            ParallelFramePhase.PAINT,
            2,
            events,
        )
        coordinator = ParallelFrameCoordinator[int]()
        coordinator.add(first.binding()).add(second.binding())
        coordinator.update(5)
        coordinator.update(9)
        events.clear()

        coordinator.restore()

        self.assertEqual((first.value, second.value), (1, 2))
        self.assertEqual(
            events,
            ["rollback:second:2", "rollback:first:1"],
        )
        self.assertFalse(coordinator.active)
        self.assertIsNone(coordinator.last_committed_frame)
        self.assertEqual(coordinator.update(3).index, 0)

    def test_sources_are_scoped_detect_cycles_and_configuration_is_fixed(
        self,
    ) -> None:
        coordinator: ParallelFrameCoordinator[int] = ParallelFrameCoordinator()
        source = coordinator.source("plain", lambda: 1)
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "only during update",
        ):
            source()
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "duplicate coordinated source",
        ):
            coordinator.source("plain", lambda: 2)

        recursive = None

        def resolve_recursive() -> object:
            assert recursive is not None
            return recursive()

        recursive = coordinator.source("recursive", resolve_recursive)
        participant = _MutableParticipant(
            "participant",
            ParallelFramePhase.CAMERA,
            0,
            [],
        )
        binding = participant.binding()

        def prepare(frame: int) -> int:
            recursive()
            return frame

        coordinator.add(
            ParallelFrameParticipant(
                binding.participant_id,
                binding.phase,
                prepare,
                binding.snapshot,
                binding.commit,
                binding.rollback,
            )
        )
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "source cycle",
        ):
            coordinator.update(1)
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "sealed",
        ):
            coordinator.source("late", lambda: 3)

    def test_reentrant_update_fails_before_any_commit(self) -> None:
        coordinator: ParallelFrameCoordinator[int] = ParallelFrameCoordinator()
        participant = _MutableParticipant(
            "participant",
            ParallelFramePhase.CAMERA,
            0,
            [],
        )
        binding = participant.binding()

        def prepare(frame: int) -> int:
            coordinator.update(frame)
            return frame

        coordinator.add(
            ParallelFrameParticipant(
                binding.participant_id,
                binding.phase,
                prepare,
                binding.snapshot,
                binding.commit,
                binding.rollback,
            )
        )
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "not reentrant",
        ):
            coordinator.update(1)
        self.assertEqual(participant.value, 0)

    def test_parallel_camera_participant_commits_and_restores_exact_states(
        self,
    ) -> None:
        initial = ParallelCameraState(IDENTITY, target=(0.0, 0.0, 0.0))
        target = ParallelCameraState(
            np.asarray(
                (
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 0.0),
                )
            ),
            target=(1.0, 2.0, 3.0),
            screen_anchor=(-0.4, 0.3),
            zoom=1.25,
        )
        camera = _Camera(initial)
        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(parallel_camera_frame_participant(camera))

        result = coordinator.update(ParallelFrameState(target))
        self.assertIs(result.frame.camera, target)
        self.assertIs(camera.state, target)
        coordinator.restore()
        self.assertIs(camera.state, initial)

    def test_camera_participant_rejects_incomplete_transition_snapshot(self) -> None:
        initial = ParallelCameraState(IDENTITY)
        target = ParallelCameraState(IDENTITY, target=(1.0, 0.0, 0.0))
        camera = _TransitioningCamera(initial, 0.4)
        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(parallel_camera_frame_participant(camera))

        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "static frame boundary",
        ):
            coordinator.update(ParallelFrameState(target))

        self.assertIs(camera.state, initial)
        self.assertEqual(camera.transition_tracker.get_value(), 0.4)
        self.assertIsNone(coordinator.last_committed_frame)

    def test_camera_participant_uses_full_transaction_snapshot_when_available(
        self,
    ) -> None:
        initial = ParallelCameraState(IDENTITY)
        target = ParallelCameraState(IDENTITY, target=(1.0, 0.0, 0.0))

        class TransactionCamera(_TransitioningCamera):
            def snapshot_parallel_transaction(self) -> tuple[object, ...]:
                return (self.state, self.transition_tracker.value)

            def restore_parallel_transaction(self, value: object) -> None:
                state, progress = value  # type: ignore[misc]
                self.state = state
                self.transition_tracker.value = progress

        camera = TransactionCamera(initial, 0.4)
        failing = _MutableParticipant(
            "failing",
            ParallelFramePhase.PAINT,
            0,
            [],
            fail_commit=True,
        )
        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(parallel_camera_frame_participant(camera))

        binding = failing.binding()
        coordinator.add(
            ParallelFrameParticipant(
                binding.participant_id,
                binding.phase,
                lambda _frame: 1,
                binding.snapshot,
                binding.commit,
                binding.rollback,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            coordinator.update(ParallelFrameState(target))

        self.assertIs(camera.state, initial)
        self.assertEqual(camera.transition_tracker.get_value(), 0.4)

    def test_invalid_configuration_and_empty_coordinator_fail_early(self) -> None:
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "participant_id",
        ):
            ParallelFrameParticipant(
                " ",
                ParallelFramePhase.CAMERA,
                lambda frame: frame,
                lambda: None,
                lambda _prepared: None,
                lambda _snapshot: None,
            )
        coordinator: ParallelFrameCoordinator[int] = ParallelFrameCoordinator()
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "at least one participant",
        ):
            coordinator.update(1)
        with self.assertRaisesRegex(TypeError, "state_getter"):
            parallel_camera_frame_participant(
                _Camera(ParallelCameraState(IDENTITY)),
                state_getter=object(),  # type: ignore[arg-type]
            )

    def test_parallel_frame_state_freezes_channel_mapping(self) -> None:
        channels = {" cut-plane ": 3}
        state = ParallelFrameState(
            ParallelCameraState(IDENTITY),
            channels,
        )
        channels["cut-plane"] = 4
        self.assertEqual(state.channel("cut-plane"), 3)
        with self.assertRaises(TypeError):
            state.channels["new"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "no channel",
        ):
            state.channel("missing")


if __name__ == "__main__":
    unittest.main()
