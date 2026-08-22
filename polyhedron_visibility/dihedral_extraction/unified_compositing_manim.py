from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from manim import Mobject

from ..binding import OcclusionBindingError, OverlayPlan
from ..painter_band import (
    ManagedPainterBand,
    ManagedPainterBandError,
    PreparedPainterBand,
    PreparedPainterItem,
)
from .compositing_manim import (
    DerivedDihedralTransparentLayer,
    PreparedDerivedDihedralTransparentFrame,
)
from .unified_compositing import DerivedDihedralUnifiedCompositingFrame


class DerivedDihedralUnifiedManimError(OcclusionBindingError):
    """Raised before one unified face/stroke paint order mutates Cairo state."""


# Compatibility export: existing callers and traces use this domain name.
PreparedUnifiedPaintItem = PreparedPainterItem


@dataclass(frozen=True)
class PreparedDerivedDihedralUnifiedFrame:
    frame: DerivedDihedralUnifiedCompositingFrame
    items: tuple[PreparedUnifiedPaintItem, ...]


class DerivedDihedralUnifiedLayer:
    """Assign exact far-to-near z slots to face batches and line spans."""

    def __init__(
        self,
        *,
        face_sources: Mapping[str, Mobject],
        stroke_sources: Mapping[str, Mobject],
        managed_roots: Sequence[Mobject] = (),
    ) -> None:
        self.face_sources = dict(face_sources)
        self.stroke_sources = dict(stroke_sources)
        # Preserve the released derived-dihedral contract: authored sources
        # define the band and must occupy distinct slots.
        self._band = ManagedPainterBand(
            require_distinct_source_z=True,
            managed_roots=managed_roots,
        )

    def configure(self, containers: Sequence[list[object]]) -> None:
        try:
            self._band.configure(
                containers=containers,
                sources={**self.face_sources, **self.stroke_sources},
            )
        except ManagedPainterBandError as exc:
            raise DerivedDihedralUnifiedManimError(str(exc)) from exc

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
        batch_map = {batch.batch_id: batch for batch in transparent_prepared.batches}
        item_mobjects: dict[str, Mobject] = {}
        for batch in frame.face_batches:
            prepared = batch_map.get(batch.item_id)
            if prepared is None or prepared.fragment_ids != batch.fragment_ids:
                raise DerivedDihedralUnifiedManimError(
                    f"unified face batch {batch.item_id!r} lost its transparent slot"
                )
            item_mobjects[batch.item_id] = transparent_layer.slots[prepared.slot_index]
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
        try:
            prepared = self._band.prepare(
                draw_order=frame.draw_order,
                item_mobjects=item_mobjects,
            )
        except ManagedPainterBandError as exc:
            raise DerivedDihedralUnifiedManimError(str(exc)) from exc
        return PreparedDerivedDihedralUnifiedFrame(frame, prepared.items)

    def apply(self, prepared: PreparedDerivedDihedralUnifiedFrame) -> None:
        try:
            self._band.apply(PreparedPainterBand(prepared.items))
        except ManagedPainterBandError as exc:
            raise DerivedDihedralUnifiedManimError(str(exc)) from exc

    @property
    def active_z_indices(self) -> dict[str, float]:
        return self._band.active_z_indices

    def restore(self) -> None:
        self._band.restore()


__all__ = [
    "DerivedDihedralUnifiedLayer",
    "DerivedDihedralUnifiedManimError",
    "PreparedDerivedDihedralUnifiedFrame",
    "PreparedUnifiedPaintItem",
]
