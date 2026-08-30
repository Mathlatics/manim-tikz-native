"""Renderer-neutral transaction core for coordinated parallel-camera frames.

The Manim bindings in this project historically own independent updater
transactions.  That is sufficient while one controller changes at a time, but
not when one authored frame changes a camera, a cutting plane, and more than
one display controller together.  This module supplies the small shared core:

* named sources are resolved at most once per frame;
* every participant prepares before any participant mutates state;
* commits run in deterministic phase/registration order;
* a failed commit rolls back the failing participant and every earlier commit;
* a failed rollback poisons the coordinator and closes it to future updates.

The module deliberately imports neither Manim nor the quadric binding.  Those
layers can adapt their existing prepare/apply/snapshot operations to the public
participant protocol without moving geometry or painter logic into this file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import isfinite
from types import MappingProxyType
from typing import Generic, TypeVar, cast

from .parallel_camera import ParallelCameraState


FrameT = TypeVar("FrameT")
SourceT = TypeVar("SourceT")


class ParallelFrameCoordinatorError(RuntimeError):
    """A coordinated frame cannot be prepared or committed safely."""


class ParallelFrameCoordinatorPoisonedError(ParallelFrameCoordinatorError):
    """A prior rollback failed, so the coordinator must not run again."""


class ParallelFramePhase(IntEnum):
    """Stable commit phases for one coordinated frame.

    Participants in the same phase retain registration order.  The gaps are
    intentional: future reviewed adapters can add a phase without renumbering
    the existing public contract.
    """

    PREFLIGHT = 0
    CAMERA = 10
    GEOMETRY = 20
    VISIBILITY = 30
    PAINT = 40
    FINALIZE = 50


class ParallelFrameBindingKind(str, Enum):
    """Audited capability implemented by a standard participant factory."""

    PREFLIGHT_GATE = "preflight-gate"
    SCREEN_TRANSFORM_GUARD = "screen-transform-guard"
    CAMERA = "camera"
    SECTION_BANK = "section-bank"
    SECTION_PLANE_PATCH = "section-plane-patch"
    SECTION_PAINTER = "section-painter"
    SECTION_DISPLAY = "section-display"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParallelFrameCoordinatorError(
            f"{label} must be a non-empty string"
        )
    return value.strip()


@dataclass(frozen=True, slots=True)
class ParallelFrameState:
    """One immutable authored frame shared by every participant.

    ``channels`` is an intentionally narrow extension point.  A future section
    timeline may place its immutable plane/display state there while this core
    continues to own only transaction semantics.  Channel values must already
    be immutable authoring values; the coordinator never copies or mutates
    them.
    """

    camera: ParallelCameraState
    channels: Mapping[str, object] = field(default_factory=dict)
    frame_id: str | None = None
    preflight_input_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera, ParallelCameraState):
            raise TypeError("camera must be a ParallelCameraState")
        if not isinstance(self.channels, Mapping):
            raise TypeError("channels must be a mapping")
        normalized: dict[str, object] = {}
        for raw_name, value in self.channels.items():
            name = _identity(raw_name, "channel name")
            if name in normalized:
                raise ParallelFrameCoordinatorError(
                    f"duplicate channel name {name!r}"
                )
            normalized[name] = value
        object.__setattr__(
            self,
            "channels",
            MappingProxyType(dict(sorted(normalized.items()))),
        )
        if (self.frame_id is None) != (self.preflight_input_digest is None):
            raise ParallelFrameCoordinatorError(
                "frame_id and preflight_input_digest must be provided together"
            )
        if self.frame_id is not None:
            object.__setattr__(
                self,
                "frame_id",
                _identity(self.frame_id, "frame_id"),
            )
            object.__setattr__(
                self,
                "preflight_input_digest",
                _identity(
                    self.preflight_input_digest,
                    "preflight_input_digest",
                ),
            )

    def channel(self, name: str) -> object:
        """Return one required channel or fail with a useful authoring error."""

        key = _identity(name, "channel name")
        try:
            return self.channels[key]
        except KeyError as exc:
            raise ParallelFrameCoordinatorError(
                f"coordinated frame has no channel {key!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ParallelFrameParticipant(Generic[FrameT]):
    """One fixed participant in a coordinated frame transaction.

    ``prepare`` must be read-only.  ``snapshot`` captures exactly the mutable
    state which ``commit`` may change, and ``rollback`` restores that snapshot.
    The coordinator treats the participant whose ``commit`` raises as possibly
    partially mutated and rolls it back too.
    """

    participant_id: str
    phase: ParallelFramePhase
    prepare: Callable[[FrameT], object]
    snapshot: Callable[[], object]
    commit: Callable[[object], None]
    rollback: Callable[[object], None]
    binding_kind: ParallelFrameBindingKind | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "participant_id",
            _identity(self.participant_id, "participant_id"),
        )
        try:
            phase = ParallelFramePhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise ParallelFrameCoordinatorError(
                "phase must be a ParallelFramePhase"
            ) from exc
        object.__setattr__(self, "phase", phase)
        if self.binding_kind is not None:
            try:
                binding_kind = ParallelFrameBindingKind(self.binding_kind)
            except (TypeError, ValueError) as exc:
                raise ParallelFrameCoordinatorError(
                    "binding_kind must be a ParallelFrameBindingKind"
                ) from exc
            object.__setattr__(self, "binding_kind", binding_kind)
        for name in ("prepare", "snapshot", "commit", "rollback"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


@dataclass(frozen=True, slots=True)
class ParallelFrameCommit(Generic[FrameT]):
    """Evidence for the most recent fully committed frame."""

    index: int
    frame: FrameT
    participant_ids: tuple[str, ...]
    resolved_source_ids: tuple[str, ...]


class ParallelFrameSource(Generic[SourceT]):
    """Callable handle whose provider is evaluated once per active frame."""

    __slots__ = ("_coordinator", "source_id")

    def __init__(
        self,
        coordinator: "ParallelFrameCoordinator[object]",
        source_id: str,
    ) -> None:
        self._coordinator = coordinator
        self.source_id = source_id

    def __call__(self) -> SourceT:
        return cast(SourceT, self._coordinator._resolve_source(self.source_id))


class ParallelFrameCoordinator(Generic[FrameT]):
    """Prepare and atomically commit one immutable multi-participant frame."""

    def __init__(
        self,
        frame_provider: Callable[[], FrameT] | None = None,
    ) -> None:
        if frame_provider is not None and not callable(frame_provider):
            raise TypeError("frame_provider must be callable or None")
        self._frame_provider = frame_provider
        self._participants: list[ParallelFrameParticipant[FrameT]] = []
        self._participant_ids: set[str] = set()
        self._source_providers: dict[str, Callable[[], object]] = {}
        self._source_cache: dict[str, object] | None = None
        self._resolving_sources: set[str] = set()
        self._resolved_source_ids: list[str] = []
        self._sealed = False
        self._updating = False
        self._poisoned = False
        self._baseline_snapshots: tuple[object, ...] | None = None
        self._last_commit: ParallelFrameCommit[FrameT] | None = None
        self._next_index = 0

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def active(self) -> bool:
        return self._baseline_snapshots is not None

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def last_committed_frame(self) -> ParallelFrameCommit[FrameT] | None:
        return self._last_commit

    @property
    def participant_ids(self) -> tuple[str, ...]:
        return tuple(
            participant.participant_id
            for _, participant in self._ordered_participants()
        )

    @property
    def participant_bindings(
        self,
    ) -> tuple[
        tuple[str, ParallelFramePhase, ParallelFrameBindingKind | None],
        ...,
    ]:
        return tuple(
            (
                participant.participant_id,
                participant.phase,
                participant.binding_kind,
            )
            for _, participant in self._ordered_participants()
        )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._source_providers))

    def _require_open_configuration(self) -> None:
        if self._sealed:
            raise ParallelFrameCoordinatorError(
                "coordinator participants and sources are sealed"
            )

    def source(
        self,
        source_id: str,
        provider: Callable[[], SourceT],
    ) -> ParallelFrameSource[SourceT]:
        """Register one read-only provider before the first frame is prepared."""

        self._require_open_configuration()
        name = _identity(source_id, "source_id")
        if name in self._source_providers:
            raise ParallelFrameCoordinatorError(
                f"duplicate coordinated source {name!r}"
            )
        if not callable(provider):
            raise TypeError("source provider must be callable")
        self._source_providers[name] = cast(Callable[[], object], provider)
        return ParallelFrameSource(
            cast(ParallelFrameCoordinator[object], self),
            name,
        )

    def add(
        self,
        participant: ParallelFrameParticipant[FrameT],
    ) -> "ParallelFrameCoordinator[FrameT]":
        """Add one participant; the participant set freezes on first update."""

        self._require_open_configuration()
        if not isinstance(participant, ParallelFrameParticipant):
            raise TypeError("participant must be a ParallelFrameParticipant")
        if participant.participant_id in self._participant_ids:
            raise ParallelFrameCoordinatorError(
                f"duplicate participant_id {participant.participant_id!r}"
            )
        self._participant_ids.add(participant.participant_id)
        self._participants.append(participant)
        return self

    def _ordered_participants(
        self,
    ) -> tuple[tuple[int, ParallelFrameParticipant[FrameT]], ...]:
        return tuple(
            sorted(
                enumerate(self._participants),
                key=lambda item: (item[1].phase, item[0]),
            )
        )

    def _require_usable(self) -> None:
        if self._poisoned:
            raise ParallelFrameCoordinatorPoisonedError(
                "coordinator is poisoned after a failed rollback"
            )
        if self._updating:
            raise ParallelFrameCoordinatorError(
                "coordinator update is not reentrant"
            )

    def _resolve_source(self, source_id: str) -> object:
        cache = self._source_cache
        if cache is None:
            raise ParallelFrameCoordinatorError(
                "coordinated sources can be read only during update()"
            )
        if source_id in cache:
            return cache[source_id]
        if source_id in self._resolving_sources:
            raise ParallelFrameCoordinatorError(
                f"coordinated source cycle at {source_id!r}"
            )
        try:
            provider = self._source_providers[source_id]
        except KeyError as exc:  # Defensive: handles are coordinator-owned.
            raise ParallelFrameCoordinatorError(
                f"unknown coordinated source {source_id!r}"
            ) from exc
        self._resolving_sources.add(source_id)
        try:
            value = provider()
        finally:
            self._resolving_sources.remove(source_id)
        cache[source_id] = value
        self._resolved_source_ids.append(source_id)
        return value

    @staticmethod
    def _annotate_rollback_failure(
        primary: BaseException,
        participant_id: str,
        rollback_error: BaseException,
    ) -> None:
        primary.add_note(
            f"rollback for participant {participant_id!r} also failed: "
            f"{rollback_error!r}"
        )

    def update(
        self,
        frame: FrameT | None = None,
    ) -> ParallelFrameCommit[FrameT]:
        """Resolve, prepare, and atomically commit one coordinated frame.

        When ``frame`` is omitted, ``frame_provider`` is called exactly once.
        Passing an explicit frame is useful in deterministic tests or offline
        authoring tools.  A coordinator whose frame type legitimately includes
        ``None`` should always use a provider rather than pass ``None`` here.
        """

        self._require_usable()
        if not self._participants:
            raise ParallelFrameCoordinatorError(
                "coordinator requires at least one participant"
            )
        self._sealed = True
        self._updating = True
        self._source_cache = {}
        self._resolving_sources.clear()
        self._resolved_source_ids.clear()
        try:
            if frame is None:
                if self._frame_provider is None:
                    raise ParallelFrameCoordinatorError(
                        "update() requires an explicit frame or frame_provider"
                    )
                candidate = self._frame_provider()
            else:
                candidate = frame
            ordered = self._ordered_participants()
            prepared = tuple(
                participant.prepare(candidate)
                for _, participant in ordered
            )
            snapshots = tuple(
                participant.snapshot()
                for _, participant in ordered
            )
            pending_baseline = (
                snapshots if self._baseline_snapshots is None else None
            )
            applied_count = 0
            try:
                for index, (_, participant) in enumerate(ordered):
                    # Count before invoking commit: a raising participant may
                    # already have partially mutated its owned state.
                    applied_count = index + 1
                    participant.commit(prepared[index])
            except BaseException as primary:
                rollback_failed = False
                for rollback_index in range(applied_count - 1, -1, -1):
                    participant = ordered[rollback_index][1]
                    try:
                        participant.rollback(snapshots[rollback_index])
                    except BaseException as rollback_error:
                        rollback_failed = True
                        self._annotate_rollback_failure(
                            primary,
                            participant.participant_id,
                            rollback_error,
                        )
                if rollback_failed:
                    self._poisoned = True
                raise
            if pending_baseline is not None:
                self._baseline_snapshots = pending_baseline
            result = ParallelFrameCommit(
                index=self._next_index,
                frame=candidate,
                participant_ids=tuple(
                    participant.participant_id
                    for _, participant in ordered
                ),
                resolved_source_ids=tuple(self._resolved_source_ids),
            )
            self._next_index += 1
            self._last_commit = result
            return result
        finally:
            self._source_cache = None
            self._resolving_sources.clear()
            self._resolved_source_ids.clear()
            self._updating = False

    def restore(self) -> None:
        """Restore every participant to its pre-first-frame baseline."""

        if self._updating:
            raise ParallelFrameCoordinatorError(
                "cannot restore during a coordinator update"
            )
        baseline = self._baseline_snapshots
        if baseline is None:
            return
        ordered = self._ordered_participants()
        primary: BaseException | None = None
        for index in range(len(ordered) - 1, -1, -1):
            participant = ordered[index][1]
            try:
                participant.rollback(baseline[index])
            except BaseException as rollback_error:
                if primary is None:
                    primary = rollback_error
                else:
                    self._annotate_rollback_failure(
                        primary,
                        participant.participant_id,
                        rollback_error,
                    )
        if primary is not None:
            self._poisoned = True
            raise primary
        self._baseline_snapshots = None
        self._last_commit = None
        self._next_index = 0


def parallel_camera_frame_participant(
    camera: object,
    *,
    participant_id: str = "parallel-camera",
    phase: ParallelFramePhase = ParallelFramePhase.CAMERA,
    state_getter: Callable[[object], ParallelCameraState] | None = None,
) -> ParallelFrameParticipant[object]:
    """Adapt a semantic parallel camera without importing its renderer class."""

    if state_getter is not None and not callable(state_getter):
        raise TypeError("state_getter must be callable or None")
    snapshot = getattr(camera, "snapshot_parallel_state", None)
    setter = getattr(camera, "set_parallel_state", None)
    if not callable(snapshot) or not callable(setter):
        raise TypeError(
            "camera must provide snapshot_parallel_state() and set_parallel_state()"
        )
    transaction_snapshot = getattr(camera, "snapshot_parallel_transaction", None)
    transaction_restore = getattr(camera, "restore_parallel_transaction", None)
    if callable(transaction_snapshot) != callable(transaction_restore):
        raise TypeError(
            "camera transaction snapshot and restore methods must be provided together"
        )
    use_full_transaction = callable(transaction_snapshot)

    def require_static_boundary() -> None:
        if use_full_transaction:
            return
        readiness = getattr(camera, "parallel_transaction_ready", None)
        if callable(readiness):
            if not bool(readiness()):
                raise ParallelFrameCoordinatorError(
                    "camera coordinator requires a static frame boundary"
                )
            return
        tracker = getattr(camera, "transition_tracker", None)
        get_value = getattr(tracker, "get_value", None)
        if callable(get_value):
            try:
                progress = float(get_value())
            except (TypeError, ValueError, OverflowError) as exc:
                raise ParallelFrameCoordinatorError(
                    "camera transition progress must be finite"
                ) from exc
            if not isfinite(progress):
                raise ParallelFrameCoordinatorError(
                    "camera transition progress must be finite"
                )
            if progress != 1.0:
                raise ParallelFrameCoordinatorError(
                    "camera coordinator requires a static frame boundary"
                )

    def prepare(frame: object) -> ParallelCameraState:
        require_static_boundary()
        value = (
            state_getter(frame)
            if state_getter is not None
            else frame.camera
            if isinstance(frame, ParallelFrameState)
            else None
        )
        if not isinstance(value, ParallelCameraState):
            raise TypeError(
                "camera participant frame must resolve to ParallelCameraState"
            )
        return value

    def capture() -> object:
        require_static_boundary()
        if use_full_transaction:
            return transaction_snapshot()
        value = snapshot()
        if not isinstance(value, ParallelCameraState):
            raise TypeError(
                "snapshot_parallel_state() must return ParallelCameraState"
            )
        return value

    def apply(value: object) -> None:
        if not isinstance(value, ParallelCameraState):
            raise TypeError("prepared camera state must be ParallelCameraState")
        setter(value)

    def rollback(value: object) -> None:
        if use_full_transaction:
            transaction_restore(value)
            return
        if not isinstance(value, ParallelCameraState):
            raise TypeError("camera snapshot must be ParallelCameraState")
        setter(value)

    return ParallelFrameParticipant(
        participant_id=participant_id,
        phase=phase,
        prepare=prepare,
        snapshot=capture,
        commit=apply,
        rollback=rollback,
        binding_kind=ParallelFrameBindingKind.CAMERA,
    )


__all__ = [
    "ParallelFrameCommit",
    "ParallelFrameCoordinator",
    "ParallelFrameCoordinatorError",
    "ParallelFrameCoordinatorPoisonedError",
    "ParallelFrameBindingKind",
    "ParallelFrameParticipant",
    "ParallelFramePhase",
    "ParallelFrameSource",
    "ParallelFrameState",
    "parallel_camera_frame_participant",
]
