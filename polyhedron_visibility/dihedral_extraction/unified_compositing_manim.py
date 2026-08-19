from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from manim import Mobject

from ..binding import OcclusionBindingError, OverlayPlan
from .compositing_manim import (
    DerivedDihedralTransparentLayer,
    PreparedDerivedDihedralTransparentFrame,
)
from .unified_compositing import DerivedDihedralUnifiedCompositingFrame


class DerivedDihedralUnifiedManimError(OcclusionBindingError):
    """Raised before one unified face/stroke paint order mutates Cairo state."""


@dataclass(frozen=True)
class PreparedUnifiedPaintItem:
    item_id: str
    mobject: Mobject
    z_index: float


@dataclass(frozen=True)
class PreparedDerivedDihedralUnifiedFrame:
    frame: DerivedDihedralUnifiedCompositingFrame
    items: tuple[PreparedUnifiedPaintItem, ...]


def _scene_family(containers: Sequence[list[object]]) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[int] = set()
    for container in containers:
        for root in container:
            for member in root.get_family():
                if id(member) not in seen:
                    seen.add(id(member))
                    result.append(member)
    return tuple(result)


class DerivedDihedralUnifiedLayer:
    """Assign exact far-to-near z slots to face batches and line spans."""

    def __init__(
        self,
        *,
        face_sources: Mapping[str, Mobject],
        stroke_sources: Mapping[str, Mobject],
    ) -> None:
        self.face_sources = dict(face_sources)
        self.stroke_sources = dict(stroke_sources)
        self._z_low = 0.0
        self._z_high = 0.0
        self._configured = False
        self._active_z_indices: dict[str, float] = {}

    def configure(self, containers: Sequence[list[object]]) -> None:
        if self._configured:
            return
        family = _scene_family(containers)
        scene_ids = {id(item) for item in family}
        sources = {**self.face_sources, **self.stroke_sources}
        source_family_ids = {
            id(member)
            for source in sources.values()
            for member in source.get_family()
        }
        z_values: dict[str, float] = {}
        for source_id, source in sources.items():
            if id(source) not in scene_ids:
                raise DerivedDihedralUnifiedManimError(
                    f"unified paint source {source_id!r} is not owned by the Scene"
                )
            value = float(source.z_index)
            if not np.isfinite(value):
                raise DerivedDihedralUnifiedManimError(
                    f"unified paint source {source_id!r} has non-finite z_index"
                )
            z_values[source_id] = value
        if len(set(z_values.values())) != len(z_values):
            raise DerivedDihedralUnifiedManimError(
                "unified face and stroke sources must occupy distinct authored z_index slots"
            )
        if len(z_values) < 2:
            raise DerivedDihedralUnifiedManimError(
                "unified compositing requires at least two authored z_index slots"
            )
        low = min(z_values.values())
        high = max(z_values.values())
        for member in family:
            if id(member) in source_family_ids:
                continue
            if not getattr(member, "has_points", lambda: False)():
                continue
            value = float(getattr(member, "z_index", float("nan")))
            if low <= value <= high:
                raise DerivedDihedralUnifiedManimError(
                    "an unrelated Scene drawable occupies the unified face/stroke z band"
                )
        self._z_low = low
        self._z_high = high
        self._configured = True

    @staticmethod
    def _validate_segment(
        *,
        item_id: str,
        expected_start: float,
        expected_end: float,
        actual_start: float,
        actual_end: float,
    ) -> None:
        if not np.allclose(
            (expected_start, expected_end),
            (actual_start, actual_end),
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise DerivedDihedralUnifiedManimError(
                f"unified stroke item {item_id!r} no longer matches its render plan"
            )

    def prepare(
        self,
        frame: DerivedDihedralUnifiedCompositingFrame,
        *,
        plans: Mapping[str, OverlayPlan],
        stroke_slots: Mapping[str, object],
        transparent_layer: DerivedDihedralTransparentLayer,
        transparent_prepared: PreparedDerivedDihedralTransparentFrame,
        containers: Sequence[list[object]],
    ) -> PreparedDerivedDihedralUnifiedFrame:
        self.configure(containers)
        batch_map = {
            batch.batch_id: batch for batch in transparent_prepared.batches
        }
        item_mobjects: dict[str, Mobject] = {}
        for batch in frame.face_batches:
            prepared = batch_map.get(batch.item_id)
            if prepared is None or prepared.fragment_ids != batch.fragment_ids:
                raise DerivedDihedralUnifiedManimError(
                    f"unified face batch {batch.item_id!r} lost its transparent slot"
                )
            item_mobjects[batch.item_id] = transparent_layer.slots[
                prepared.slot_index
            ]
        for fragment in frame.stroke_fragments:
            if fragment.source_edge_id not in plans:
                raise DerivedDihedralUnifiedManimError(
                    f"unified stroke {fragment.source_edge_id!r} lost its plan"
                )
            plan = plans[fragment.source_edge_id]
            slots = stroke_slots[fragment.source_edge_id]
            if fragment.slot_kind == "visible":
                if fragment.slot_index >= len(plan.visible_segments):
                    raise DerivedDihedralUnifiedManimError(
                        f"unified visible slot overflow for {fragment.item_id!r}"
                    )
                segment = plan.visible_segments[fragment.slot_index]
                mobject = slots.visible[fragment.slot_index]
            elif fragment.slot_kind == "hidden":
                if fragment.slot_index >= len(plan.hidden_segments):
                    raise DerivedDihedralUnifiedManimError(
                        f"unified hidden slot overflow for {fragment.item_id!r}"
                    )
                segment = plan.hidden_segments[fragment.slot_index]
                mobject = slots.hidden_groups[fragment.slot_index]
            else:
                raise DerivedDihedralUnifiedManimError(
                    f"unsupported unified slot kind {fragment.slot_kind!r}"
                )
            self._validate_segment(
                item_id=fragment.item_id,
                expected_start=fragment.start_parameter,
                expected_end=fragment.end_parameter,
                actual_start=segment.start_parameter,
                actual_end=segment.end_parameter,
            )
            item_mobjects[fragment.item_id] = mobject
        if set(item_mobjects) != set(frame.draw_order):
            raise DerivedDihedralUnifiedManimError(
                "unified draw order does not cover every active face and stroke item"
            )
        denominator = max(1, len(frame.draw_order) - 1)
        items = tuple(
            PreparedUnifiedPaintItem(
                item_id,
                item_mobjects[item_id],
                self._z_low
                + (self._z_high - self._z_low) * rank / denominator,
            )
            for rank, item_id in enumerate(frame.draw_order)
        )
        return PreparedDerivedDihedralUnifiedFrame(frame, items)

    def apply(self, prepared: PreparedDerivedDihedralUnifiedFrame) -> None:
        active: dict[str, float] = {}
        for item in prepared.items:
            item.mobject.set_z_index(item.z_index, family=True)
            active[item.item_id] = item.z_index
        self._active_z_indices = active

    @property
    def active_z_indices(self) -> dict[str, float]:
        return dict(self._active_z_indices)

    def restore(self) -> None:
        self._active_z_indices = {}


__all__ = [
    "DerivedDihedralUnifiedLayer",
    "DerivedDihedralUnifiedManimError",
    "PreparedDerivedDihedralUnifiedFrame",
    "PreparedUnifiedPaintItem",
]
