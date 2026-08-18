from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
from manim import ManimColor, Mobject, Polygon, VGroup

from ..binding import OcclusionBindingError
from ..contract import TolerancePolicy, VisibilityModel
from ..depth_cue.contract import FaceDepthCueFrame
from ..depth_cue.manim import _shaded_rgb
from .compositing import (
    TransparentSectionCompositingFrame,
    TransparentTriangle,
)


class TransparentSectionManimError(OcclusionBindingError):
    """Raised before exact transparent fragment binding mutates a Scene."""


@dataclass(frozen=True)
class TransparentSectionBindingScaleLimits:
    max_faces: int = 24
    max_transparent_triangles: int = 768
    max_pair_tests: int = 589_824


TRANSPARENT_SECTION_BINDING_SCALE_LIMITS = TransparentSectionBindingScaleLimits()


@dataclass(frozen=True)
class _FaceFillSnapshot:
    source: Polygon
    fill_rgbas: np.ndarray
    fill_opacity: object


@dataclass(frozen=True)
class PreparedTransparentFragment:
    fragment_id: str
    slot_index: int
    draw_rank: int
    display_points: np.ndarray
    fill_color: object
    fill_opacity: float


@dataclass(frozen=True)
class PreparedTransparentSectionFrame:
    frame: TransparentSectionCompositingFrame
    fragments: tuple[PreparedTransparentFragment, ...]
    slot_map: Mapping[str, int]


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
        raise TransparentSectionManimError(
            f"transparent face source {face_id} must use one solid non-gradient fill"
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
        raise TransparentSectionManimError(
            f"transparent face source {face_id} must be fill-only; "
            "register boundaries as semantic Lines"
        )
    return raw[0].copy()


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


def transparent_triangle_capacity(model: VisibilityModel) -> int:
    """Return a fixed upper bound for one plane intersecting one convex solid.

    A plane section has at most one boundary edge per solid face.  An
    arrangement of ``k`` full boundary lines inside a rectangular patch needs
    at most ``k² + k + 2`` triangles.  Splitting one convex n-gon by one plane
    needs at most n triangles across both sides.
    """

    section_edge_limit = len(model.faces)
    solid_limit = sum(len(face.vertex_ids) for face in model.faces)
    return solid_limit + section_edge_limit * section_edge_limit + section_edge_limit + 2


def guard_transparent_section_scale(model: VisibilityModel) -> int:
    limits = TRANSPARENT_SECTION_BINDING_SCALE_LIMITS
    if len(model.faces) > limits.max_faces:
        raise TransparentSectionManimError(
            f"exact transparent section faces={len(model.faces)} exceeds "
            f"fixed limit {limits.max_faces}"
        )
    capacity = transparent_triangle_capacity(model)
    if capacity > limits.max_transparent_triangles:
        raise TransparentSectionManimError(
            f"exact transparent section triangle capacity={capacity} exceeds fixed limit "
            f"{limits.max_transparent_triangles}"
        )
    if capacity * capacity > limits.max_pair_tests:
        raise TransparentSectionManimError(
            f"exact transparent section pair tests={capacity * capacity} exceeds fixed limit "
            f"{limits.max_pair_tests}"
        )
    return capacity


class TransparentSectionLayer:
    """Stable Cairo triangle pool for exact local transparent ordering."""

    def __init__(
        self,
        model: VisibilityModel,
        face_bindings: Mapping[str, Mobject],
        *,
        tolerance_policy: TolerancePolicy,
        source_coordinate_mode: Literal["world", "display"],
    ) -> None:
        expected = set(model.face_map)
        if set(face_bindings) != expected:
            missing = sorted(expected - set(face_bindings))
            extra = sorted(set(face_bindings) - expected)
            raise TransparentSectionManimError(
                "transparent face binding identity mismatch"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
        if source_coordinate_mode not in {"world", "display"}:
            raise TransparentSectionManimError(
                "transparent face source_coordinate_mode must be 'world' or 'display'"
            )
        self.model = model
        self.tolerance_policy = tolerance_policy
        self.source_coordinate_mode = source_coordinate_mode
        self.capacity = guard_transparent_section_scale(model)
        self.sources: dict[str, Polygon] = {}
        for face in sorted(model.faces, key=lambda item: item.face_id):
            source = face_bindings[face.face_id]
            if not isinstance(source, Polygon) or tuple(source.get_family()) != (source,):
                raise TransparentSectionManimError(
                    f"transparent face source {face.face_id} must be one native Manim Polygon"
                )
            self.sources[face.face_id] = source
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
        self._active_fragment_ids: tuple[str, ...] = ()

    def configure(self, containers: Sequence[list[object]]) -> None:
        if self._base_fill_rgba:
            return
        family = _scene_family(containers)
        scene_ids = {id(item) for item in family}
        managed_ids = {
            id(member)
            for source in self.sources.values()
            for member in source.get_family()
        }
        source_z: dict[str, float] = {}
        base_fill: dict[str, np.ndarray] = {}
        for face_id, source in self.sources.items():
            if id(source) not in scene_ids:
                raise TransparentSectionManimError(
                    f"transparent face source {face_id} is not owned by the current Scene"
                )
            z_index = float(source.z_index)
            if not np.isfinite(z_index):
                raise TransparentSectionManimError(
                    f"transparent face source {face_id} has a non-finite z_index"
                )
            source_z[face_id] = z_index
            base_fill[face_id] = _solid_fill(source, face_id)
        slots = tuple(sorted(source_z.values()))
        if len(set(slots)) != len(slots):
            raise TransparentSectionManimError(
                "managed transparent faces must occupy distinct authored z_index slots"
            )
        low, high = min(slots), max(slots)
        for member in family:
            if id(member) in managed_ids or not getattr(member, "has_points", lambda: False)():
                continue
            member_z = float(getattr(member, "z_index", float("nan")))
            if low <= member_z <= high:
                raise TransparentSectionManimError(
                    "an unrelated Scene drawable occupies the transparent face z band"
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
        for face in self.model.faces:
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
            actual = np.asarray(self.sources[face.face_id].get_vertices(), dtype=float)
            tolerance = self.tolerance_policy.resolve(expected).boundary
            if not _face_points_match(actual, expected, tolerance):
                raise TransparentSectionManimError(
                    f"transparent face source {face.face_id} no longer "
                    "matches its registered polygon"
                )
            if float(self.sources[face.face_id].z_index) != self._source_z[face.face_id]:
                raise TransparentSectionManimError(
                    f"transparent face source {face.face_id} changed its authored z_index"
                )
            current = _solid_fill(self.sources[face.face_id], face.face_id)
            if not np.allclose(
                current[:3],
                self._base_fill_rgba[face.face_id][:3],
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise TransparentSectionManimError(
                    f"transparent face source {face.face_id} changed its authored fill color"
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
        available = [index for index in range(capacity) if index not in retained.values()]
        for fragment_id in fragment_ids:
            if fragment_id not in retained:
                if not available:
                    raise TransparentSectionManimError(
                        "transparent fragment frame exceeds the preallocated slot pool"
                    )
                retained[fragment_id] = available.pop(0)
        return retained

    def _face_style(
        self,
        triangle: TransparentTriangle,
        face_depth_cue: FaceDepthCueFrame | None,
    ) -> tuple[object, float]:
        if triangle.source_face_id is None:
            raise TransparentSectionManimError(
                "solid transparent fragment lost its source face identity"
            )
        base = self._base_fill_rgba[triangle.source_face_id]
        if face_depth_cue is None:
            return ManimColor.from_rgb(base[:3]), float(base[3])
        cue = face_depth_cue.face_map[triangle.source_face_id]
        return (
            ManimColor.from_rgb(
                _shaded_rgb(
                    base[:3],
                    brightness=cue.brightness,
                    saturation_scale=cue.saturation_scale,
                    hue_shift_turns=cue.hue_shift_turns,
                    fog_strength=cue.fog_strength,
                    fog_color_rgb=face_depth_cue.fog_color_rgb,
                )
            ),
            float(np.clip(base[3] * cue.opacity_scale, 0.0, 1.0)),
        )

    def prepare(
        self,
        frame: TransparentSectionCompositingFrame,
        *,
        world_points: Mapping[str, np.ndarray],
        display_point_provider: object,
        plane_fill_color: object,
        plane_fill_opacity: float,
        section_fill_color: object,
        section_fill_opacity: float,
        face_depth_cue: FaceDepthCueFrame | None,
        containers: Sequence[list[object]],
    ) -> PreparedTransparentSectionFrame:
        self.configure(containers)
        display = display_point_provider
        if not callable(display):
            raise TransparentSectionManimError(
                "transparent section display point provider must be callable"
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
            raise TransparentSectionManimError(
                f"transparent fragment frame requires {len(frame.fragments)} "
                f"slots; capacity={self.capacity}"
            )
        fragment_ids = tuple(frame.draw_order)
        if set(fragment_ids) != set(frame.fragment_map):
            raise TransparentSectionManimError(
                "transparent draw order does not cover every fragment"
            )
        slot_map = self._next_slot_map(fragment_ids, self._slot_map, self.capacity)
        fragments: list[PreparedTransparentFragment] = []
        for rank, fragment_id in enumerate(fragment_ids):
            triangle = frame.fragment_map[fragment_id]
            points = np.asarray([display(point) for point in triangle.vertices], dtype=float)
            if points.shape != (3, 3) or not np.all(np.isfinite(points)):
                raise TransparentSectionManimError(
                    f"transparent fragment {fragment_id} has invalid display points"
                )
            if triangle.role.startswith("solid_face"):
                fill_color, fill_opacity = self._face_style(
                    triangle, face_depth_cue
                )
            elif triangle.role == "plane_outside":
                fill_color, fill_opacity = plane_fill_color, float(plane_fill_opacity)
            elif triangle.role == "section_inside":
                fill_color, fill_opacity = section_fill_color, float(section_fill_opacity)
            else:
                raise TransparentSectionManimError(
                    f"transparent fragment {fragment_id} has unknown role {triangle.role!r}"
                )
            fragments.append(
                PreparedTransparentFragment(
                    fragment_id,
                    slot_map[fragment_id],
                    rank,
                    points,
                    fill_color,
                    fill_opacity,
                )
            )
        return PreparedTransparentSectionFrame(
            frame,
            tuple(fragments),
            slot_map,
        )

    def apply(self, prepared: PreparedTransparentSectionFrame) -> None:
        active_slots: set[int] = set()
        denominator = max(1, len(prepared.fragments) - 1)
        for item in prepared.fragments:
            active_slots.add(item.slot_index)
            slot = self.slots[item.slot_index]
            points = item.display_points
            slot.set_points_as_corners([points[0], points[1], points[2], points[0]])
            slot.set_fill(item.fill_color, opacity=item.fill_opacity)
            slot.set_stroke(opacity=0.0)
            z_index = self._z_low + (self._z_high - self._z_low) * (
                item.draw_rank / denominator
            )
            slot.set_z_index(z_index, family=True)
        for index, slot in enumerate(self.slots):
            if index not in active_slots:
                slot.set_fill(opacity=0.0)
                slot.set_stroke(opacity=0.0)
        self._slot_map = dict(prepared.slot_map)
        self._active_fragment_ids = tuple(
            item.fragment_id for item in prepared.fragments
        )

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
        self._active_fragment_ids = ()

    @property
    def active_fragment_ids(self) -> tuple[str, ...]:
        return self._active_fragment_ids

    def active_fragment_z_indices(self) -> dict[str, float]:
        return {
            fragment_id: float(self.slots[self._slot_map[fragment_id]].z_index)
            for fragment_id in self._active_fragment_ids
        }

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


__all__ = [
    "PreparedTransparentFragment",
    "PreparedTransparentSectionFrame",
    "TRANSPARENT_SECTION_BINDING_SCALE_LIMITS",
    "TransparentSectionBindingScaleLimits",
    "TransparentSectionLayer",
    "TransparentSectionManimError",
    "guard_transparent_section_scale",
    "transparent_triangle_capacity",
]
