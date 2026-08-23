from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable, Mapping, Sequence

import numpy as np
from manim import Line, Mobject, VGroup

from ..binding import OcclusionBindingError, OcclusionCapacityError, PlannedDash
from ..painter_band import (
    ManagedPainterBand,
    ManagedPainterBandError,
    PreparedPainterBand,
)
from ..style import OcclusionStyle, ResolvedOcclusionStyle
from ..visibility import VisibilityKind
from .contract import OpenFaceVisibilityModel
from .unified_compositing import (
    OpenFaceUnifiedCompositingFrame,
    PaintPathFragment,
)


class OpenFaceUnifiedManimError(OcclusionBindingError):
    """Raised before a unified open-face frame mutates Cairo state."""


@dataclass(frozen=True, slots=True)
class OpenFaceUnifiedBindingScaleLimits:
    max_fragments_per_path: int = 512
    max_total_fragments: int = 8192
    max_dashes_per_fragment: int = 1024
    max_total_dash_lines: int = 65536
    max_total_mobjects: int = 100000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


OPEN_FACE_UNIFIED_BINDING_SCALE_LIMITS = OpenFaceUnifiedBindingScaleLimits()


class ManagedDisplayGroup(VGroup):
    """Animation proxy whose sentinel alpha drives all managed drawables.

    Manim fades interpolate RGBA arrays directly.  The fade target changes only
    the invisible sentinel; the geometry updater reads that interpolated alpha
    and reapplies it to the preallocated face/path proxies.  This prevents
    geometry updates from cancelling FadeIn/FadeOut.
    """

    def __init__(self, *mobjects: Mobject, opacity_sentinel: Line) -> None:
        self._opacity_sentinel = opacity_sentinel
        super().__init__(*mobjects, opacity_sentinel)

    @property
    def opacity_multiplier(self) -> float:
        rgba = np.asarray(
            getattr(self._opacity_sentinel, "stroke_rgbas", ()), dtype=float
        )
        if rgba.ndim < 2 or rgba.shape[-1] < 4 or not rgba.size:
            return 1.0
        value = float(rgba[0, 3])
        return value if np.isfinite(value) and value >= 0.0 else 0.0

    def set_opacity(self, opacity: float, family: bool = True) -> "ManagedDisplayGroup":
        del family
        value = float(opacity)
        if not np.isfinite(value) or value < 0.0:
            raise OpenFaceUnifiedManimError(
                "display opacity multiplier must be finite and non-negative"
            )
        self._opacity_sentinel.set_stroke(opacity=value)
        return self

    def reset_opacity(self) -> None:
        self._opacity_sentinel.set_stroke(opacity=1.0)


@dataclass(frozen=True, slots=True)
class _FragmentCapacity:
    fragment_slots: int
    dash_slots_per_fragment: int


@dataclass(frozen=True, slots=True)
class _PreparedFragment:
    fragment: PaintPathFragment
    slot_index: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    dashes: tuple[PlannedDash, ...]


@dataclass(frozen=True, slots=True)
class PreparedOpenFaceUnifiedManimFrame:
    frame: OpenFaceUnifiedCompositingFrame
    face_plans: Mapping[str, np.ndarray]
    path_plans: Mapping[str, tuple[_PreparedFragment, ...]]
    painter_band: PreparedPainterBand


@dataclass(slots=True)
class _MobjectState:
    mobject: object
    points: np.ndarray | None
    z_index: float | None
    attributes: dict[str, object]


def _copy_value(value: object) -> object:
    return value.copy() if isinstance(value, np.ndarray) else value


def _capture_root(root: Mobject) -> tuple[_MobjectState, ...]:
    result: list[_MobjectState] = []
    seen: set[int] = set()
    for member in root.get_family():
        if id(member) in seen:
            continue
        seen.add(id(member))
        points = None
        if hasattr(member, "points"):
            points = np.asarray(member.points, dtype=float).copy()
        attributes: dict[str, object] = {}
        for name in (
            "fill_rgbas",
            "stroke_rgbas",
            "background_stroke_rgbas",
            "fill_opacity",
            "stroke_opacity",
            "background_stroke_opacity",
        ):
            if hasattr(member, name):
                attributes[name] = _copy_value(getattr(member, name))
        z_index = (
            float(member.z_index)
            if hasattr(member, "z_index") and np.isfinite(float(member.z_index))
            else None
        )
        result.append(_MobjectState(member, points, z_index, attributes))
    return tuple(result)


def _restore_root(states: Sequence[_MobjectState]) -> None:
    for state in states:
        if state.points is not None and hasattr(state.mobject, "points"):
            state.mobject.points = state.points.copy()
        for name, value in state.attributes.items():
            setattr(state.mobject, name, _copy_value(value))
        if state.z_index is not None:
            state.mobject.z_index = state.z_index


class _UnifiedFragmentSlot:
    def __init__(self, dash_capacity: int) -> None:
        self.solid = Line((0, 0, 0), (1, 0, 0), buff=0)
        self.dashes = [
            Line((0, 0, 0), (1, 0, 0), buff=0) for _ in range(dash_capacity)
        ]
        self.dash_group = VGroup(*self.dashes)
        self.root = VGroup(self.solid, self.dash_group)
        self.hide()

    @staticmethod
    def _hide_line(line: Line) -> None:
        line.set_stroke(opacity=0.0)
        line.set_stroke(opacity=0.0, background=True)

    def hide(self) -> None:
        self._hide_line(self.solid)
        for line in self.dashes:
            self._hide_line(line)

    def apply_static_style(self, style: ResolvedOcclusionStyle) -> None:
        if style.cap_style is not None:
            self.solid.set_cap_style(style.cap_style)
        if style.joint_type is not None:
            self.solid.joint_type = style.joint_type
        hidden_cap_style = (
            style.cap_style
            if style.hidden_cap_style is None
            else style.hidden_cap_style
        )
        hidden_joint_type = (
            style.joint_type
            if style.hidden_joint_type is None
            else style.hidden_joint_type
        )
        for line in self.dashes:
            if hidden_cap_style is not None:
                line.set_cap_style(hidden_cap_style)
            if hidden_joint_type is not None:
                line.joint_type = hidden_joint_type
        for line in (self.solid, *self.dashes):
            line.set_stroke(
                color=style.background_color,
                width=style.background_width,
                opacity=0.0,
                background=True,
            )

    def apply(
        self,
        prepared: _PreparedFragment,
        style: ResolvedOcclusionStyle,
        opacity_multiplier: float,
    ) -> Mobject:
        if prepared.fragment.visibility_kind is VisibilityKind.VISIBLE:
            self.solid.put_start_and_end_on(prepared.start, prepared.end)
            self.solid.set_stroke(
                color=style.visible_color,
                width=style.visible_width,
                opacity=style.visible_opacity * opacity_multiplier,
            )
            self.solid.set_stroke(
                color=style.background_color,
                width=style.background_width,
                opacity=style.background_opacity * opacity_multiplier,
                background=True,
            )
            for dash in self.dashes:
                self._hide_line(dash)
            return self.solid

        self._hide_line(self.solid)
        for index, line in enumerate(self.dashes):
            if index >= len(prepared.dashes):
                self._hide_line(line)
                continue
            dash = prepared.dashes[index]
            line.put_start_and_end_on(dash.start, dash.end)
            line.set_stroke(
                color=style.hidden_color,
                width=style.hidden_width,
                opacity=style.hidden_opacity * opacity_multiplier,
            )
            line.set_stroke(
                color=style.background_color,
                width=style.background_width,
                opacity=style.background_opacity * opacity_multiplier,
                background=True,
            )
        return self.dash_group

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


class _UnifiedPathSlots:
    def __init__(self, capacity: _FragmentCapacity) -> None:
        self.capacity = capacity
        self.fragments = [
            _UnifiedFragmentSlot(capacity.dash_slots_per_fragment)
            for _ in range(capacity.fragment_slots)
        ]
        self.root = VGroup(*(item.root for item in self.fragments))

    def apply_static_style(self, style: ResolvedOcclusionStyle) -> None:
        for item in self.fragments:
            item.apply_static_style(style)

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


def _capacity_for_model(
    model: OpenFaceVisibilityModel,
    styles: Mapping[str, OcclusionStyle],
    limits: OpenFaceUnifiedBindingScaleLimits,
) -> dict[str, _FragmentCapacity]:
    face_count = len(model.faces)
    path_count = len(model.strokes)
    result: dict[str, _FragmentCapacity] = {}
    total_fragments = 0
    total_dashes = 0
    total_fragment_mobjects = 0
    for stroke in model.strokes:
        style = styles[stroke.source_edge_id]
        dash_capacity = int(ceil(style.max_projected_length / style.dash_period)) + 2
        if dash_capacity > limits.max_dashes_per_fragment:
            raise OpenFaceUnifiedManimError(
                f"path {stroke.source_edge_id!r} dashes_per_fragment="
                f"{dash_capacity} exceeds limit {limits.max_dashes_per_fragment}"
            )
        occluder_count = sum(
            1
            for face in model.faces
            if face.occludes_strokes and face.face_id not in stroke.incident_face_ids
        )
        # Visibility boundaries, finite face entry/exit/depth roots, and the
        # worst point/overlap/depth event contribution from every other path.
        fragment_capacity = max(
            1,
            1 + 2 * occluder_count + 3 * face_count + 3 * max(0, path_count - 1),
        )
        if fragment_capacity > limits.max_fragments_per_path:
            raise OpenFaceUnifiedManimError(
                f"path {stroke.source_edge_id!r} fragment capacity "
                f"{fragment_capacity} exceeds limit {limits.max_fragments_per_path}"
            )
        result[stroke.source_edge_id] = _FragmentCapacity(
            fragment_capacity,
            dash_capacity,
        )
        total_fragments += fragment_capacity
        total_dashes += fragment_capacity * dash_capacity
        total_fragment_mobjects += fragment_capacity * (dash_capacity + 3)
    if total_fragments > limits.max_total_fragments:
        raise OpenFaceUnifiedManimError(
            f"unified open-face fragment slots={total_fragments} exceeds limit "
            f"{limits.max_total_fragments}"
        )
    if total_dashes > limits.max_total_dash_lines:
        raise OpenFaceUnifiedManimError(
            f"unified open-face dash lines={total_dashes} exceeds limit "
            f"{limits.max_total_dash_lines}"
        )
    total_mobjects = (
        face_count
        + 1  # face root
        + path_count  # one root per source path
        + total_fragment_mobjects
        + 2  # display root and opacity sentinel
    )
    if total_mobjects > limits.max_total_mobjects:
        raise OpenFaceUnifiedManimError(
            f"unified open-face Mobjects={total_mobjects} exceeds limit "
            f"{limits.max_total_mobjects}"
        )
    return result


def _point(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise OpenFaceUnifiedManimError(f"{label} must be a finite 3D point")
    return result


def _prepare_fragment(
    fragment: PaintPathFragment,
    *,
    slot_index: int,
    display_start: np.ndarray,
    display_end: np.ndarray,
    capacity: _FragmentCapacity,
    style: OcclusionStyle,
) -> _PreparedFragment:
    delta = display_end - display_start
    full_length = float(np.linalg.norm(delta))
    allowance = max(1.0e-12, style.max_projected_length * 1.0e-9)
    if full_length > style.max_projected_length + allowance:
        raise OcclusionCapacityError(
            f"projected length {full_length:.9g} exceeds fixed maximum "
            f"{style.max_projected_length:.9g}"
        )
    interval = fragment.parameter_interval
    start = display_start + interval.start * delta
    end = display_start + interval.end * delta
    dashes: list[PlannedDash] = []
    if fragment.visibility_kind is VisibilityKind.HIDDEN and full_length > 1.0e-12:
        hidden_start = interval.start * full_length
        hidden_end = interval.end * full_length
        period_index = max(
            0,
            int(np.floor((hidden_start - style.dash_length) / style.dash_period)) + 1,
        )
        while period_index * style.dash_period < hidden_end - 1.0e-12:
            dash_start_distance = period_index * style.dash_period
            dash_end_distance = dash_start_distance + style.dash_length
            period_index += 1
            clipped_start = max(hidden_start, dash_start_distance)
            clipped_end = min(hidden_end, dash_end_distance)
            if clipped_end - clipped_start <= 1.0e-12:
                continue
            first = display_start + clipped_start / full_length * delta
            last = display_start + clipped_end / full_length * delta
            dashes.append(
                PlannedDash(
                    tuple(float(item) for item in first),
                    tuple(float(item) for item in last),
                )
            )
    if len(dashes) > capacity.dash_slots_per_fragment:
        raise OcclusionCapacityError(
            f"dash count {len(dashes)} exceeds unified fragment capacity "
            f"{capacity.dash_slots_per_fragment}"
        )
    return _PreparedFragment(
        fragment,
        slot_index,
        tuple(float(item) for item in start),
        tuple(float(item) for item in end),
        tuple(dashes),
    )


class OpenFaceUnifiedManimRuntime:
    """Preallocated Cairo runtime for renderer-neutral open-face painter frames."""

    def __init__(
        self,
        model: OpenFaceVisibilityModel,
        *,
        face_layer: object,
        stroke_sources: Mapping[str, Mobject],
        face_sources: Mapping[str, Mobject],
        style: OcclusionStyle,
        painter_z_band: tuple[float, float],
        scale_limits: OpenFaceUnifiedBindingScaleLimits = (
            OPEN_FACE_UNIFIED_BINDING_SCALE_LIMITS
        ),
        stroke_styles: Mapping[str, OcclusionStyle] | None = None,
    ) -> None:
        if set(stroke_sources) != set(model.stroke_map):
            raise OpenFaceUnifiedManimError("unified stroke source identities mismatch")
        if set(face_sources) != set(model.face_map):
            raise OpenFaceUnifiedManimError("unified face source identities mismatch")
        self.model = model
        self.face_layer = face_layer
        self.stroke_sources = dict(stroke_sources)
        self.face_sources = dict(face_sources)
        self.style = style
        self.stroke_styles = (
            {stroke.source_edge_id: style for stroke in model.strokes}
            if stroke_styles is None
            else dict(stroke_styles)
        )
        if set(self.stroke_styles) != set(model.stroke_map) or not all(
            isinstance(value, OcclusionStyle) for value in self.stroke_styles.values()
        ):
            raise OpenFaceUnifiedManimError("unified stroke style identities mismatch")
        self.capacities = _capacity_for_model(model, self.stroke_styles, scale_limits)
        self.path_slots = {
            path_id: _UnifiedPathSlots(self.capacities[path_id])
            for path_id in sorted(self.capacities)
        }
        # A tiny non-zero line avoids degenerate-Line constructor behaviour;
        # width zero keeps it absent from the rendered frame.
        self._opacity_sentinel = Line((0, 0, 0), (1.0e-9, 0, 0), buff=0)
        self._opacity_sentinel.set_stroke(width=0.0, opacity=1.0)
        self.root = ManagedDisplayGroup(
            face_layer.root,
            *(self.path_slots[key].root for key in sorted(self.path_slots)),
            opacity_sentinel=self._opacity_sentinel,
        )
        self._band = ManagedPainterBand(
            z_band=painter_z_band,
            managed_roots=(self.root,),
        )
        self._resolved_styles: dict[str, ResolvedOcclusionStyle] = {}
        self._styles_configured = False
        self._last_frame: OpenFaceUnifiedCompositingFrame | None = None

    def set_painter_z_band(self, value: tuple[float, float]) -> None:
        if self._last_frame is not None:
            raise OpenFaceUnifiedManimError(
                "painter z band can only change while the runtime is restored"
            )
        self._band = ManagedPainterBand(z_band=value, managed_roots=(self.root,))

    @property
    def display_mobject(self) -> ManagedDisplayGroup:
        return self.root

    @property
    def last_frame(self) -> OpenFaceUnifiedCompositingFrame | None:
        return self._last_frame

    def configure_styles(self) -> dict[str, ResolvedOcclusionStyle]:
        resolved = {
            path_id: self.stroke_styles[path_id].resolve_for(source)
            for path_id, source in self.stroke_sources.items()
        }
        for path_id, slots in self.path_slots.items():
            slots.apply_static_style(resolved[path_id])
        self._resolved_styles = resolved
        self._styles_configured = True
        return resolved

    def configure_band(self, containers: Sequence[list[object]]) -> None:
        try:
            self._band.configure(
                containers=containers,
                sources={
                    **{f"face:{key}": value for key, value in self.face_sources.items()},
                    **{f"path:{key}": value for key, value in self.stroke_sources.items()},
                },
            )
        except ManagedPainterBandError as exc:
            raise OpenFaceUnifiedManimError(str(exc)) from exc

    def prepare(
        self,
        frame: OpenFaceUnifiedCompositingFrame,
        *,
        face_plans: Mapping[str, np.ndarray],
        display_positions: Mapping[str, np.ndarray],
        containers: Sequence[list[object]],
    ) -> PreparedOpenFaceUnifiedManimFrame:
        if not self._styles_configured:
            raise OpenFaceUnifiedManimError("unified path styles are not configured")
        self.configure_band(containers)
        by_path: dict[str, list[PaintPathFragment]] = {}
        for fragment in frame.path_fragments:
            by_path.setdefault(fragment.source_path_id, []).append(fragment)
        path_plans: dict[str, tuple[_PreparedFragment, ...]] = {}
        item_mobjects: dict[str, Mobject] = {
            face.item_id: self.face_layer.proxies[face.face_id]
            for face in frame.faces
        }
        for stroke in self.model.strokes:
            path_id = stroke.source_edge_id
            fragments = by_path[path_id]
            slots = self.path_slots[path_id]
            if len(fragments) > len(slots.fragments):
                raise OcclusionCapacityError(
                    f"path {path_id!r} painter fragments={len(fragments)} exceeds "
                    f"slot capacity {len(slots.fragments)}"
                )
            display_start = _point(
                display_positions[stroke.vertex_ids[0]],
                f"path {path_id} display start",
            )
            display_end = _point(
                display_positions[stroke.vertex_ids[1]],
                f"path {path_id} display end",
            )
            prepared_values: list[_PreparedFragment] = []
            for index, fragment in enumerate(fragments):
                prepared = _prepare_fragment(
                    fragment,
                    slot_index=index,
                    display_start=display_start,
                    display_end=display_end,
                    capacity=self.capacities[path_id],
                    style=self.stroke_styles[path_id],
                )
                prepared_values.append(prepared)
                slot = slots.fragments[index]
                item_mobjects[fragment.fragment_id] = (
                    slot.solid
                    if fragment.visibility_kind is VisibilityKind.VISIBLE
                    else slot.dash_group
                )
            path_plans[path_id] = tuple(prepared_values)
        try:
            painter = self._band.prepare(
                draw_order=frame.draw_order,
                item_mobjects=item_mobjects,
            )
        except ManagedPainterBandError as exc:
            raise OpenFaceUnifiedManimError(str(exc)) from exc
        return PreparedOpenFaceUnifiedManimFrame(
            frame,
            dict(face_plans),
            path_plans,
            painter,
        )

    def apply(
        self,
        prepared: PreparedOpenFaceUnifiedManimFrame,
        *,
        after_apply: Callable[[], None] | None = None,
    ) -> None:
        snapshots = _capture_root(self.root)
        previous_band_state = self._band.capture_active_state()
        opacity = self.root.opacity_multiplier
        try:
            self.face_layer.apply_geometry(
                prepared.face_plans,
                opacity_multiplier=opacity,
            )
            for path_id, slots in self.path_slots.items():
                plans = prepared.path_plans[path_id]
                for index, slot in enumerate(slots.fragments):
                    if index >= len(plans):
                        slot.hide()
                        continue
                    slot.apply(plans[index], self._resolved_styles[path_id], opacity)
            self._band.apply(prepared.painter_band)
            if after_apply is not None:
                after_apply()
        except Exception:
            _restore_root(snapshots)
            self._band.restore_active_state(previous_band_state)
            raise
        self._last_frame = prepared.frame

    def restore(self) -> None:
        self._band.restore()
        self._last_frame = None
        self.root.reset_opacity()

    @property
    def active_z_indices(self) -> dict[str, float]:
        return self._band.active_z_indices

    def slot_counts(self, path_id: str) -> tuple[int, int]:
        capacity = self.capacities[path_id]
        return capacity.fragment_slots, capacity.dash_slots_per_fragment

    def slot_identities(self) -> tuple[int, ...]:
        return tuple(
            identity
            for path_id in sorted(self.path_slots)
            for identity in self.path_slots[path_id].identities()
        )

    def slot_snapshot(self) -> tuple[object, ...]:
        values: list[object] = []
        for member in self.root.get_family():
            points = np.asarray(getattr(member, "points", np.empty((0, 3))), dtype=float)
            values.append(tuple(np.round(points.reshape(-1), 12)))
            for name in ("fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"):
                rgba = np.asarray(getattr(member, name, np.empty((0, 4))), dtype=float)
                values.append(tuple(np.round(rgba.reshape(-1), 12)))
            values.append(float(getattr(member, "z_index", 0.0)))
        return tuple(values)


__all__ = [
    "ManagedDisplayGroup",
    "OPEN_FACE_UNIFIED_BINDING_SCALE_LIMITS",
    "OpenFaceUnifiedBindingScaleLimits",
    "OpenFaceUnifiedManimError",
    "OpenFaceUnifiedManimRuntime",
    "PreparedOpenFaceUnifiedManimFrame",
]
