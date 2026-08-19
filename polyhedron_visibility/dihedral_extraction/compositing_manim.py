from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal, Mapping, Sequence

import numpy as np
from manim import ManimColor, Mobject, Polygon, VGroup

from ..binding import OcclusionBindingError
from ..contract import TolerancePolicy
from .compositing import (
    DerivedDihedralTransparentCompositingFrame,
    TransparentTriangle,
    transparent_dihedral_triangle_capacity,
)
from .contract import DerivedDihedralModel


class DerivedDihedralTransparentManimError(OcclusionBindingError):
    """Raised before exact transparent dihedral rendering mutates a Scene."""


@dataclass(frozen=True)
class DerivedDihedralTransparentBindingScaleLimits:
    max_faces: int = 32
    max_transparent_triangles: int = 768
    max_pair_tests: int = 589_824


DERIVED_DIHEDRAL_TRANSPARENT_BINDING_SCALE_LIMITS = (
    DerivedDihedralTransparentBindingScaleLimits()
)


@dataclass(frozen=True)
class _FaceFillSnapshot:
    source: Polygon
    fill_rgbas: np.ndarray
    fill_opacity: object


@dataclass(frozen=True)
class PreparedDerivedDihedralFillBatch:
    batch_id: str
    fragment_ids: tuple[str, ...]
    slot_index: int
    draw_rank: int
    display_triangles: tuple[np.ndarray, ...]
    fill_color: object
    fill_opacity: float


@dataclass(frozen=True)
class PreparedDerivedDihedralTransparentFrame:
    frame: DerivedDihedralTransparentCompositingFrame
    batches: tuple[PreparedDerivedDihedralFillBatch, ...]
    slot_map: Mapping[str, int]


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


def _face_points_match(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float,
) -> bool:
    if actual.shape != expected.shape or len(actual) < 3:
        return False
    for candidate in (expected, expected[::-1]):
        for offset in range(len(candidate)):
            if float(
                np.max(
                    np.linalg.norm(
                        actual - np.roll(candidate, -offset, axis=0), axis=1
                    )
                )
            ) <= tolerance:
                return True
    return False


def _solid_fill(source: Polygon, face_id: str) -> np.ndarray:
    raw = np.asarray(getattr(source, "fill_rgbas", ()), dtype=float)
    if (
        raw.ndim != 2
        or raw.shape[1:] != (4,)
        or not len(raw)
        or not np.all(np.isfinite(raw))
        or any(
            not np.allclose(item, raw[0], rtol=0.0, atol=1.0e-12)
            for item in raw[1:]
        )
    ):
        raise DerivedDihedralTransparentManimError(
            f"derived-dihedral transparent face {face_id} must use one "
            "solid non-gradient fill"
        )
    foreground_stroke = float(source.get_stroke_width()) * float(
        source.get_stroke_opacity()
    )
    background_stroke = float(source.get_stroke_width(background=True)) * float(
        source.get_stroke_opacity(background=True)
    )
    if (
        not np.isfinite(foreground_stroke)
        or not np.isfinite(background_stroke)
        or foreground_stroke > 1.0e-12
        or background_stroke > 1.0e-12
    ):
        raise DerivedDihedralTransparentManimError(
            f"derived-dihedral transparent face {face_id} must be fill-only; "
            "register its boundaries as semantic Lines"
        )
    return raw[0].copy()


def guard_derived_dihedral_transparent_scale(model: DerivedDihedralModel) -> int:
    limits = DERIVED_DIHEDRAL_TRANSPARENT_BINDING_SCALE_LIMITS
    face_count = len(model.solid.faces) + len(model.extraction.source_face_ids)
    if face_count > limits.max_faces:
        raise DerivedDihedralTransparentManimError(
            f"exact derived-dihedral transparent faces={face_count} exceeds "
            f"fixed limit {limits.max_faces}"
        )
    capacity = transparent_dihedral_triangle_capacity(model)
    if capacity > limits.max_transparent_triangles:
        raise DerivedDihedralTransparentManimError(
            "exact derived-dihedral triangle capacity="
            f"{capacity} exceeds fixed limit {limits.max_transparent_triangles}"
        )
    if capacity * capacity > limits.max_pair_tests:
        raise DerivedDihedralTransparentManimError(
            "exact derived-dihedral pair tests="
            f"{capacity * capacity} exceeds fixed limit {limits.max_pair_tests}"
        )
    return capacity


class DerivedDihedralTransparentLayer:
    """Stable Cairo triangle pool for one solid and one extracted dihedral."""

    def __init__(
        self,
        model: DerivedDihedralModel,
        face_bindings: Mapping[str, Mobject],
        *,
        tolerance_policy: TolerancePolicy,
        source_coordinate_mode: Literal["world", "display"],
        managed_peer_sources: Mapping[str, Mobject] | None = None,
    ) -> None:
        overlay_model = model.overlay_model()
        expected = set(overlay_model.face_map)
        if set(face_bindings) != expected:
            missing = sorted(expected - set(face_bindings))
            extra = sorted(set(face_bindings) - expected)
            raise DerivedDihedralTransparentManimError(
                "derived-dihedral transparent face binding identity mismatch"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
        if source_coordinate_mode not in {"world", "display"}:
            raise DerivedDihedralTransparentManimError(
                "derived-dihedral transparent source_coordinate_mode must be "
                "'world' or 'display'"
            )
        self.model = model
        self.overlay_model = overlay_model
        self.tolerance_policy = tolerance_policy
        self.source_coordinate_mode = source_coordinate_mode
        self.managed_peer_sources = dict(managed_peer_sources or {})
        self.capacity = guard_derived_dihedral_transparent_scale(model)
        self.sources: dict[str, Polygon] = {}
        for face_id in sorted(expected):
            source = face_bindings[face_id]
            if not isinstance(source, Polygon) or tuple(source.get_family()) != (source,):
                raise DerivedDihedralTransparentManimError(
                    f"derived-dihedral transparent face {face_id} must be one "
                    "native Manim Polygon"
                )
            self.sources[face_id] = source

        dummy = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        self.slots = tuple(Polygon(*dummy) for _item in range(self.capacity))
        for slot in self.slots:
            slot.set_fill(opacity=0.0)
            slot.set_stroke(opacity=0.0)
        self.root = VGroup(*self.slots)
        self._snapshots: dict[str, _FaceFillSnapshot] = {}
        self._base_fill_rgba: dict[str, np.ndarray] = {}
        self._source_z: dict[str, float] = {}
        self._z_low = 0.0
        self._z_high = 0.0
        self._slot_map: dict[str, int] = {}
        self._fragment_slot_map: dict[str, int] = {}
        self._active_fragment_ids: tuple[str, ...] = ()

    def configure(self, containers: Sequence[list[object]]) -> None:
        if self._base_fill_rgba:
            return
        family = _scene_family(containers)
        scene_ids = {id(item) for item in family}
        managed_ids = {
            id(member)
            for source in (
                *self.sources.values(),
                *self.managed_peer_sources.values(),
            )
            for member in source.get_family()
        }
        source_z: dict[str, float] = {}
        base_fill: dict[str, np.ndarray] = {}
        for face_id, source in self.sources.items():
            if id(source) not in scene_ids:
                raise DerivedDihedralTransparentManimError(
                    f"derived-dihedral transparent face {face_id} is not owned "
                    "by the current Scene"
                )
            z_index = float(source.z_index)
            if not np.isfinite(z_index):
                raise DerivedDihedralTransparentManimError(
                    f"derived-dihedral transparent face {face_id} has a "
                    "non-finite z_index"
                )
            source_z[face_id] = z_index
            base_fill[face_id] = _solid_fill(source, face_id)
        z_values = tuple(sorted(source_z.values()))
        if len(set(z_values)) != len(z_values):
            raise DerivedDihedralTransparentManimError(
                "managed derived-dihedral transparent faces must occupy "
                "distinct authored z_index slots"
            )
        low, high = min(z_values), max(z_values)
        for member in family:
            if id(member) in managed_ids or not getattr(
                member, "has_points", lambda: False
            )():
                continue
            member_z = float(getattr(member, "z_index", float("nan")))
            if low <= member_z <= high:
                raise DerivedDihedralTransparentManimError(
                    "an unrelated Scene drawable occupies the exact "
                    "derived-dihedral face z band"
                )
        self._source_z = source_z
        self._base_fill_rgba = base_fill
        self._z_low = low
        self._z_high = high

    def _validate_sources(
        self,
        *,
        world_points: Mapping[str, np.ndarray],
        display_points: Mapping[str, np.ndarray],
    ) -> None:
        for face in self.overlay_model.faces:
            expected = np.asarray(
                [
                    (
                        world_points[vertex_id]
                        if self.source_coordinate_mode == "world"
                        else display_points[vertex_id]
                    )
                    for vertex_id in face.vertex_ids
                ],
                dtype=float,
            )
            source = self.sources[face.face_id]
            actual = np.asarray(source.get_vertices(), dtype=float)
            tolerance = self.tolerance_policy.resolve(expected).boundary
            if not _face_points_match(actual, expected, tolerance):
                raise DerivedDihedralTransparentManimError(
                    f"derived-dihedral transparent face {face.face_id} no "
                    "longer matches its registered polygon"
                )
            if float(source.z_index) != self._source_z[face.face_id]:
                raise DerivedDihedralTransparentManimError(
                    f"derived-dihedral transparent face {face.face_id} changed "
                    "its authored z_index"
                )
            current = _solid_fill(source, face.face_id)
            if not np.allclose(
                current[:3],
                self._base_fill_rgba[face.face_id][:3],
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise DerivedDihedralTransparentManimError(
                    f"derived-dihedral transparent face {face.face_id} changed "
                    "its authored fill color"
                )

    @staticmethod
    def _next_slot_map(
        fragment_ids: Sequence[str],
        current: Mapping[str, int],
        capacity: int,
    ) -> dict[str, int]:
        retained = {
            fragment_id: current[fragment_id]
            for fragment_id in fragment_ids
            if fragment_id in current
        }
        available = [
            index for index in range(capacity) if index not in retained.values()
        ]
        for fragment_id in fragment_ids:
            if fragment_id not in retained:
                if not available:
                    raise DerivedDihedralTransparentManimError(
                        "derived-dihedral transparent frame exceeds the "
                        "preallocated triangle pool"
                    )
                retained[fragment_id] = available.pop(0)
        return retained

    def prepare(
        self,
        frame: DerivedDihedralTransparentCompositingFrame,
        *,
        world_points: Mapping[str, np.ndarray],
        display_point_provider: object,
        containers: Sequence[list[object]],
        opacity_scales: Mapping[str, float] | None = None,
    ) -> PreparedDerivedDihedralTransparentFrame:
        self.configure(containers)
        display = display_point_provider
        if display is None:
            display = lambda point: point
        if not callable(display):
            raise DerivedDihedralTransparentManimError(
                "derived-dihedral display_point_provider must be callable"
            )
        display_points = {
            vertex_id: np.asarray(display(point), dtype=float)
            for vertex_id, point in world_points.items()
        }
        self._validate_sources(
            world_points=world_points,
            display_points=display_points,
        )
        if len(frame.fragments) > self.capacity:
            raise DerivedDihedralTransparentManimError(
                f"derived-dihedral transparent frame requires "
                f"{len(frame.fragments)} slots; capacity={self.capacity}"
            )
        fragment_ids = tuple(frame.draw_order)
        if set(fragment_ids) != set(frame.fragment_map):
            raise DerivedDihedralTransparentManimError(
                "derived-dihedral draw order does not cover every fragment"
            )
        batches = frame.draw_batches
        if tuple(fragment_id for batch in batches for fragment_id in batch) != fragment_ids:
            raise DerivedDihedralTransparentManimError(
                "derived-dihedral draw batches do not preserve draw order"
            )
        batch_ids = tuple(
            "batch:"
            + hashlib.sha256("|".join(batch).encode("utf-8")).hexdigest()[:20]
            for batch in batches
        )
        slot_map = self._next_slot_map(
            batch_ids,
            self._slot_map,
            self.capacity,
        )
        scales = opacity_scales or {}
        if any(
            not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in scales.values()
        ):
            raise DerivedDihedralTransparentManimError(
                "derived-dihedral face opacity scales must be finite values from 0 to 1"
            )
        prepared: list[PreparedDerivedDihedralFillBatch] = []
        for rank, (batch_id, batch) in enumerate(zip(batch_ids, batches)):
            triangles = tuple(frame.fragment_map[fragment_id] for fragment_id in batch)
            source_face_ids = {triangle.source_face_id for triangle in triangles}
            if len(source_face_ids) != 1:
                raise DerivedDihedralTransparentManimError(
                    f"derived-dihedral fill batch {batch_id} mixes source faces"
                )
            source_face_id = next(iter(source_face_ids))
            if source_face_id not in self._base_fill_rgba:
                raise DerivedDihedralTransparentManimError(
                    f"derived-dihedral fill batch {batch_id} lost its "
                    "source face identity"
                )
            display_triangles = tuple(
                np.asarray(
                    [display(point) for point in triangle.vertices],
                    dtype=float,
                )
                for triangle in triangles
            )
            if any(
                points.shape != (3, 3) or not np.all(np.isfinite(points))
                for points in display_triangles
            ):
                raise DerivedDihedralTransparentManimError(
                    f"derived-dihedral fill batch {batch_id} has invalid display points"
                )
            rgba = self._base_fill_rgba[source_face_id]
            prepared.append(
                PreparedDerivedDihedralFillBatch(
                    batch_id,
                    tuple(batch),
                    slot_map[batch_id],
                    rank,
                    display_triangles,
                    ManimColor.from_rgb(rgba[:3]),
                    float(rgba[3]) * float(scales.get(source_face_id, 1.0)),
                )
            )
        return PreparedDerivedDihedralTransparentFrame(
            frame,
            tuple(prepared),
            slot_map,
        )

    def apply(self, prepared: PreparedDerivedDihedralTransparentFrame) -> None:
        active_slots: set[int] = set()
        fragment_slot_map: dict[str, int] = {}
        denominator = max(1, len(prepared.batches) - 1)
        for batch in prepared.batches:
            active_slots.add(batch.slot_index)
            slot = self.slots[batch.slot_index]
            slot.clear_points()
            for points in batch.display_triangles:
                slot.start_new_path(points[0])
                slot.add_line_to(points[1])
                slot.add_line_to(points[2])
                slot.close_path()
            slot.set_fill(batch.fill_color, opacity=batch.fill_opacity)
            slot.set_stroke(opacity=0.0)
            z_index = self._z_low + (self._z_high - self._z_low) * (
                batch.draw_rank / denominator
            )
            slot.set_z_index(z_index, family=True)
            for fragment_id in batch.fragment_ids:
                fragment_slot_map[fragment_id] = batch.slot_index
        for index, slot in enumerate(self.slots):
            if index not in active_slots:
                slot.set_fill(opacity=0.0)
                slot.set_stroke(opacity=0.0)
        self._slot_map = dict(prepared.slot_map)
        self._fragment_slot_map = fragment_slot_map
        self._active_fragment_ids = tuple(prepared.frame.draw_order)

    def capture_and_hide(self) -> None:
        self._snapshots = {
            face_id: _FaceFillSnapshot(
                source,
                np.asarray(source.fill_rgbas, dtype=float).copy(),
                getattr(source, "fill_opacity", None),
            )
            for face_id, source in self.sources.items()
        }
        self.hide()

    def hide(self) -> None:
        for snapshot in self._snapshots.values():
            hidden = snapshot.fill_rgbas.copy()
            hidden[..., 3] = 0.0
            snapshot.source.fill_rgbas = hidden
            if hasattr(snapshot.source, "fill_opacity"):
                snapshot.source.fill_opacity = 0.0

    def restore(self) -> None:
        for snapshot in self._snapshots.values():
            snapshot.source.fill_rgbas = snapshot.fill_rgbas.copy()
            if snapshot.fill_opacity is not None and hasattr(
                snapshot.source, "fill_opacity"
            ):
                snapshot.source.fill_opacity = snapshot.fill_opacity
        self._snapshots = {}
        self._slot_map = {}
        self._fragment_slot_map = {}
        self._active_fragment_ids = ()

    @property
    def active_fragment_ids(self) -> tuple[str, ...]:
        return self._active_fragment_ids

    def active_fragment_z_indices(self) -> dict[str, float]:
        return {
            fragment_id: float(
                self.slots[self._fragment_slot_map[fragment_id]].z_index
            )
            for fragment_id in self._active_fragment_ids
        }

    @property
    def active_draw_batch_count(self) -> int:
        return len(set(self._fragment_slot_map.values()))

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


__all__ = [
    "DERIVED_DIHEDRAL_TRANSPARENT_BINDING_SCALE_LIMITS",
    "DerivedDihedralTransparentBindingScaleLimits",
    "DerivedDihedralTransparentLayer",
    "DerivedDihedralTransparentManimError",
    "PreparedDerivedDihedralFillBatch",
    "PreparedDerivedDihedralTransparentFrame",
    "guard_derived_dihedral_transparent_scale",
]
