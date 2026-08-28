"""Opt-in structured performance evidence for quadric Manim frames.

The renderer-neutral geometry contract must not depend on wall-clock timing.
This module therefore owns diagnostic-only, process-local measurements used by
the Cairo acceptance suite.  Recording is disabled unless the dedicated
environment flag is set, and snapshots never participate in painter or
geometry decisions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from os import environ
from time import perf_counter_ns
from types import MappingProxyType
from typing import Iterator, Mapping


QUADRIC_PERFORMANCE_TRACE_SCHEMA = "manim-quadric-performance-frame/v1"
QUADRIC_PERFORMANCE_TRACE_ENV = "MANIM_TIKZ_NATIVE_QUADRIC_PERFORMANCE_TRACE"
QUADRIC_CAIRO_FRAME_TRACE_SCHEMA = "manim-quadric-cairo-frame-trace/v1"
QUADRIC_CAIRO_FRAME_TRACE_ENV = (
    "MANIM_TIKZ_NATIVE_QUADRIC_CAIRO_FRAME_TRACE_PATH"
)


_TRUTHY_ENVIRONMENT_VALUES = frozenset({"1", "true", "yes", "on"})


def quadric_performance_tracing_enabled() -> bool:
    """Return whether this process opted into per-frame measurements."""

    return (
        environ.get(QUADRIC_PERFORMANCE_TRACE_ENV, "").strip().lower()
        in _TRUTHY_ENVIRONMENT_VALUES
    )


@dataclass(frozen=True, slots=True)
class QuadricPerformanceSnapshot:
    """One completed prepare/apply attempt with structured stage evidence."""

    controller_kind: str
    frame_index: int
    status: str
    total_ns: int
    stage_durations_ns: Mapping[str, int]
    counts: Mapping[str, int]
    cache_hits: Mapping[str, int]
    cache_misses: Mapping[str, int]
    rollback_performed: bool = False
    error_type: str | None = None
    schema: str = QUADRIC_PERFORMANCE_TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.status not in {"committed", "failed"}:
            raise ValueError("performance snapshot status must be committed or failed")
        if self.frame_index < 0 or self.total_ns < 0:
            raise ValueError("performance frame index and duration must be non-negative")
        for label, values in (
            ("stage durations", self.stage_durations_ns),
            ("counts", self.counts),
            ("cache hits", self.cache_hits),
            ("cache misses", self.cache_misses),
        ):
            if any(not isinstance(key, str) or not key for key in values):
                raise ValueError(f"{label} keys must be non-empty strings")
            if any(isinstance(value, bool) or int(value) < 0 for value in values.values()):
                raise ValueError(f"{label} values must be non-negative integers")
        object.__setattr__(
            self,
            "stage_durations_ns",
            MappingProxyType(dict(sorted(self.stage_durations_ns.items()))),
        )
        object.__setattr__(
            self,
            "counts",
            MappingProxyType(dict(sorted(self.counts.items()))),
        )
        object.__setattr__(
            self,
            "cache_hits",
            MappingProxyType(dict(sorted(self.cache_hits.items()))),
        )
        object.__setattr__(
            self,
            "cache_misses",
            MappingProxyType(dict(sorted(self.cache_misses.items()))),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe evidence without process-specific representations."""

        return {
            "schema": self.schema,
            "controllerKind": self.controller_kind,
            "frameIndex": self.frame_index,
            "status": self.status,
            "totalNanoseconds": self.total_ns,
            "totalSeconds": self.total_ns / 1_000_000_000.0,
            "stageNanoseconds": dict(self.stage_durations_ns),
            "stageSeconds": {
                key: value / 1_000_000_000.0
                for key, value in self.stage_durations_ns.items()
            },
            "counts": dict(self.counts),
            "cache": {
                "hits": dict(self.cache_hits),
                "misses": dict(self.cache_misses),
            },
            "rollbackPerformed": self.rollback_performed,
            "errorType": self.error_type,
        }


class _PerformanceAttempt:
    """Mutable timing accumulator owned by one prepared frame."""

    __slots__ = (
        "controller_kind",
        "frame_index",
        "started_ns",
        "stage_durations_ns",
        "counts",
        "cache_hits",
        "cache_misses",
        "finished",
    )

    def __init__(self, controller_kind: str, frame_index: int) -> None:
        self.controller_kind = controller_kind
        self.frame_index = frame_index
        self.started_ns = perf_counter_ns()
        self.stage_durations_ns: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.cache_hits: dict[str, int] = {}
        self.cache_misses: dict[str, int] = {}
        self.finished = False

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.finished:
            yield
            return
        started = perf_counter_ns()
        try:
            yield
        finally:
            elapsed = max(0, perf_counter_ns() - started)
            self.stage_durations_ns[name] = (
                self.stage_durations_ns.get(name, 0) + elapsed
            )

    def set_count(self, name: str, value: int) -> None:
        if not self.finished:
            self.counts[name] = max(0, int(value))

    def increment_count(self, name: str, amount: int = 1) -> None:
        if not self.finished:
            self.counts[name] = self.counts.get(name, 0) + max(0, int(amount))

    def cache_hit(self, name: str) -> None:
        if not self.finished:
            self.cache_hits[name] = self.cache_hits.get(name, 0) + 1

    def cache_miss(self, name: str) -> None:
        if not self.finished:
            self.cache_misses[name] = self.cache_misses.get(name, 0) + 1

    def finish(
        self,
        *,
        status: str,
        rollback_performed: bool = False,
        error: BaseException | None = None,
    ) -> QuadricPerformanceSnapshot:
        total_ns = max(0, perf_counter_ns() - self.started_ns)
        self.finished = True
        return QuadricPerformanceSnapshot(
            controller_kind=self.controller_kind,
            frame_index=self.frame_index,
            status=status,
            total_ns=total_ns,
            stage_durations_ns=self.stage_durations_ns,
            counts=self.counts,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            rollback_performed=rollback_performed,
            error_type=None if error is None else type(error).__name__,
        )


@contextmanager
def _performance_stage(
    attempt: _PerformanceAttempt | None,
    name: str,
) -> Iterator[None]:
    """Enter one stage without reading clocks when tracing is disabled."""

    if attempt is None:
        yield
        return
    with attempt.stage(name):
        yield


__all__ = (
    "QUADRIC_CAIRO_FRAME_TRACE_ENV",
    "QUADRIC_CAIRO_FRAME_TRACE_SCHEMA",
    "QUADRIC_PERFORMANCE_TRACE_ENV",
    "QUADRIC_PERFORMANCE_TRACE_SCHEMA",
    "QuadricPerformanceSnapshot",
)
