"""Shared fixed-capacity Manim runtime for analytic quadric controllers.

This internal module owns the mutable Cairo-facing machinery shared by
``QuadricOcclusion3D`` and ``CompositeQuadricSection3D``.  Renderer-neutral
geometry remains in the existing solver/compositor modules; controllers keep
their authoring policy, while slot allocation, numeric boundary preparation,
stroke application, and rollback semantics live here exactly once.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Protocol, Sequence, TypeVar

import numpy as np
from manim import BLUE_D, WHITE, Line, Mobject, ThreeDCamera, VGroup, VMobject

from ..painter_band import ManagedPainterBand, PreparedPainterBand
from ..parallel_solver import ParallelView, SolverError
from .boundary_compositing import (
    BoundaryRenderIntent,
    QuadricBoundaryCompositingFrame,
    QuadricBoundaryPaintFragment,
    QuadricBoundarySource,
)
from .curves import EllipseArcCurve, ParametricConicBranch, SegmentCurve
from .performance import _PerformanceAttempt, _performance_stage
from .projection import ConeProjectionLayers


AnalyticCurve3D = SegmentCurve | EllipseArcCurve | ParametricConicBranch


class QuadricManimError(RuntimeError):
    """A quadric frame cannot be committed safely to Manim."""


class QuadricManimCapacityError(QuadricManimError):
    """A prepared frame exceeds an explicitly preallocated capacity."""


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _non_negative(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class QuadricBoundaryStyle:
    """One immutable visible/hidden stroke pair in the boundary registry."""

    visible_color: object = WHITE
    visible_width: float = 3.0
    visible_opacity: float = 1.0
    hidden_color: object = WHITE
    hidden_width: float = 2.4
    hidden_opacity: float = 0.78
    dash_length: float = 0.08
    dash_gap: float = 0.06
    background_color: object = WHITE
    background_width: float = 0.0
    background_opacity: float = 0.0
    cap_style: object | None = None
    joint_type: object | None = None
    hidden_cap_style: object | None = None
    hidden_joint_type: object | None = None

    def __post_init__(self) -> None:
        for name in (
            "visible_width",
            "visible_opacity",
            "hidden_width",
            "hidden_opacity",
            "background_width",
            "background_opacity",
        ):
            object.__setattr__(self, name, _non_negative(getattr(self, name), name))
        object.__setattr__(
            self, "dash_length", _positive(self.dash_length, "dash_length")
        )
        object.__setattr__(self, "dash_gap", _non_negative(self.dash_gap, "dash_gap"))
        for name in (
            "visible_opacity",
            "hidden_opacity",
            "background_opacity",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed 1")

    @property
    def dash_period(self) -> float:
        return self.dash_length + self.dash_gap


class _StyleContract(Protocol):
    surface_fill_color: object
    surface_fill_opacity: float
    surface_stroke_color: object
    surface_stroke_width: float
    surface_stroke_opacity: float
    visible_curve_color: object
    visible_curve_width: float
    visible_curve_opacity: float
    hidden_curve_color: object
    hidden_curve_width: float
    hidden_curve_opacity: float
    dash_length: float
    dash_gap: float
    background_color: object
    background_width: float
    background_opacity: float
    cap_style: object | None
    joint_type: object | None
    hidden_cap_style: object | None
    hidden_joint_type: object | None
    section_plane_stroke_color: object
    section_plane_stroke_width: float
    section_plane_stroke_opacity: float
    cone_lateral_fill_colors: tuple[object, ...] | None
    cone_cap_fill_colors: tuple[object, ...] | None
    cone_lateral_sheen_direction: tuple[float, float, float]
    cone_cap_sheen_direction: tuple[float, float, float]


class _LimitsContract(Protocol):
    max_fragments_per_curve: int
    max_segments_per_fragment: int
    max_dashes_per_fragment: int
    max_projected_length: float
    max_boundary_styles: int


def _boundary_style_from_curve_style(
    style: _StyleContract,
) -> QuadricBoundaryStyle:
    return QuadricBoundaryStyle(
        visible_color=style.visible_curve_color,
        visible_width=style.visible_curve_width,
        visible_opacity=style.visible_curve_opacity,
        hidden_color=style.hidden_curve_color,
        hidden_width=style.hidden_curve_width,
        hidden_opacity=style.hidden_curve_opacity,
        dash_length=style.dash_length,
        dash_gap=style.dash_gap,
        background_color=style.background_color,
        background_width=style.background_width,
        background_opacity=style.background_opacity,
        cap_style=style.cap_style,
        joint_type=style.joint_type,
        hidden_cap_style=style.hidden_cap_style,
        hidden_joint_type=style.hidden_joint_type,
    )


def _boundary_style_from_base_stroke(
    style: _StyleContract,
    *,
    color: object,
    width: float,
    opacity: float,
) -> QuadricBoundaryStyle:
    hidden_width_ratio = (
        style.hidden_curve_width / style.visible_curve_width
        if style.visible_curve_width > 0.0
        else 0.82
    )
    return QuadricBoundaryStyle(
        visible_color=color,
        visible_width=width,
        visible_opacity=opacity,
        hidden_color=color,
        hidden_width=width * hidden_width_ratio,
        hidden_opacity=opacity * style.hidden_curve_opacity,
        dash_length=style.dash_length,
        dash_gap=style.dash_gap,
        background_color=style.background_color,
        background_width=style.background_width,
        background_opacity=style.background_opacity,
        cap_style=style.cap_style,
        joint_type=style.joint_type,
        hidden_cap_style=style.hidden_cap_style,
        hidden_joint_type=style.hidden_joint_type,
    )


def _boundary_style_registry(
    base_style: _StyleContract,
    custom_styles: Mapping[str, QuadricBoundaryStyle] | None,
    limits: _LimitsContract,
) -> Mapping[str, QuadricBoundaryStyle]:
    curve_style = _boundary_style_from_curve_style(base_style)
    surface_style = _boundary_style_from_base_stroke(
        base_style,
        color=base_style.surface_stroke_color,
        width=base_style.surface_stroke_width,
        opacity=base_style.surface_stroke_opacity,
    )
    section_style = _boundary_style_from_base_stroke(
        base_style,
        color=base_style.section_plane_stroke_color,
        width=base_style.section_plane_stroke_width,
        opacity=base_style.section_plane_stroke_opacity,
    )
    result: dict[str, QuadricBoundaryStyle] = {
        "style:curve": curve_style,
        "style:section-outline": section_style,
        "style:surface-boundary": surface_style,
        "style:surface-silhouette": surface_style,
        "style:teaching-boundary": curve_style,
    }
    if custom_styles is not None:
        if not isinstance(custom_styles, Mapping):
            raise TypeError("boundary_styles must be a mapping")
        for raw_style_id, value in custom_styles.items():
            if not isinstance(raw_style_id, str) or not raw_style_id.strip():
                raise QuadricManimError(
                    "boundary_styles keys must be non-empty style identities"
                )
            if not isinstance(value, QuadricBoundaryStyle):
                raise TypeError(
                    "boundary_styles values must be QuadricBoundaryStyle objects"
                )
            result[raw_style_id.strip()] = value
    if len(result) > limits.max_boundary_styles:
        raise QuadricManimCapacityError(
            f"boundary style count {len(result)} exceeds fixed limit "
            f"{limits.max_boundary_styles}"
        )
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class _PreparedDash:
    points: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedBoundaryFragment:
    fragment: QuadricBoundaryPaintFragment
    source: QuadricBoundarySource
    style: QuadricBoundaryStyle
    slot_index: int
    points: np.ndarray
    dashes: tuple[_PreparedDash, ...]


@dataclass(frozen=True, slots=True)
class _PreparedConeFill:
    opaque_lateral_paths: tuple[np.ndarray, ...]
    opaque_cap_paths: tuple[np.ndarray, ...]
    back_lateral_paths: tuple[np.ndarray, ...]
    back_cap_paths: tuple[np.ndarray, ...]
    front_lateral_paths: tuple[np.ndarray, ...]
    front_cap_paths: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class _PreparedBoundaryBatch:
    fragments: Mapping[str, tuple[_PreparedBoundaryFragment, ...]]
    fragment_slot_maps: Mapping[str, Mapping[str, int]]
    item_mobjects: Mapping[str, Mobject]


@dataclass(slots=True)
class _MobjectState:
    mobject: object
    points: np.ndarray | None
    z_index: float | None
    attributes: dict[str, object]


@dataclass(frozen=True, slots=True)
class _PreparedDisplayAction:
    """One desired fixed-slot display state with no renderer allocation."""

    slot_id: str
    roots: tuple[Mobject, ...]
    digest: bytes
    apply: Callable[[], None] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _CommittedDisplaySlot:
    roots: tuple[Mobject, ...]
    digest: bytes


@dataclass(frozen=True, slots=True)
class _PreparedDisplayDelta:
    changed: tuple[_PreparedDisplayAction, ...]
    unchanged_slot_ids: tuple[str, ...]
    hidden: tuple[_CommittedDisplaySlot, ...]
    next_state: Mapping[str, _CommittedDisplaySlot]
    mutation_roots: tuple[Mobject, ...]


class _DirtyFrameKind(str, Enum):
    FULL = "full"
    DRAW_ONLY = "draw_only"
    OPACITY_ONLY = "opacity_only"
    CLEAN = "clean"


def _classify_dirty_frame(
    previous_geometry: bytes | None,
    previous_draw: bytes | None,
    previous_opacity: float | None,
    *,
    geometry: bytes,
    draw: bytes,
    opacity: float,
) -> _DirtyFrameKind:
    if previous_geometry is None or previous_geometry != geometry:
        return _DirtyFrameKind.FULL
    if previous_draw is None or previous_draw != draw:
        return _DirtyFrameKind.DRAW_ONLY
    if previous_opacity is None or previous_opacity != opacity:
        return _DirtyFrameKind.OPACITY_ONLY
    return _DirtyFrameKind.CLEAN


class _SurfaceViewCache:
    """Fixed-size exact-signature cache for pure surface/view products."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, tuple[bytes, object]] = {}

    def lookup(self, name: str, signature: bytes) -> tuple[bool, object | None]:
        entry = self._entries.get(name)
        if entry is None or entry[0] != signature:
            return False, None
        return True, entry[1]

    def store(self, name: str, signature: bytes, value: object) -> None:
        self._entries[name] = (bytes(signature), value)

    def clear(self) -> None:
        self._entries.clear()


def _copy_value(value: object) -> object:
    return value.copy() if isinstance(value, np.ndarray) else value


def _unique_family(roots: Sequence[Mobject]) -> tuple[Mobject, ...]:
    result: list[Mobject] = []
    seen: set[int] = set()
    for root in roots:
        for member in root.get_family():
            if id(member) in seen:
                continue
            seen.add(id(member))
            result.append(member)
    return tuple(result)


def _capture_roots(roots: Sequence[Mobject]) -> tuple[_MobjectState, ...]:
    result: list[_MobjectState] = []
    for member in _unique_family(roots):
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
            "sheen_direction",
            "sheen_factor",
        ):
            if hasattr(member, name):
                attributes[name] = _copy_value(getattr(member, name))
        raw_z = getattr(member, "z_index", None)
        z_index = None
        if raw_z is not None:
            value = float(raw_z)
            if np.isfinite(value):
                z_index = value
        result.append(_MobjectState(member, points, z_index, attributes))
    return tuple(result)


def _capture_root(root: Mobject) -> tuple[_MobjectState, ...]:
    return _capture_roots((root,))


def _restore_root(states: Sequence[_MobjectState]) -> None:
    for state in states:
        if state.points is not None and hasattr(state.mobject, "points"):
            state.mobject.points = state.points.copy()
        for name, value in state.attributes.items():
            setattr(state.mobject, name, _copy_value(value))
        if state.z_index is not None:
            state.mobject.z_index = state.z_index


def _update_display_digest(hasher: object, value: object) -> None:
    """Feed one deterministic, process-local display value into ``hasher``."""

    update = getattr(hasher, "update")
    if value is None:
        update(b"N")
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        update(b"A")
        update(array.dtype.str.encode("ascii"))
        update(repr(array.shape).encode("ascii"))
        update(array.tobytes())
        return
    if isinstance(value, Enum):
        update(b"E")
        update(type(value).__qualname__.encode("utf-8"))
        _update_display_digest(hasher, value.value)
        return
    if isinstance(value, bool):
        update(b"B1" if value else b"B0")
        return
    if isinstance(value, int):
        update(b"I")
        update(str(value).encode("ascii"))
        update(b";")
        return
    if isinstance(value, float):
        update(b"F")
        update(np.asarray((value,), dtype=np.float64).tobytes())
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        update(b"S")
        update(str(len(encoded)).encode("ascii"))
        update(b":")
        update(encoded)
        return
    if isinstance(value, bytes):
        update(b"Y")
        update(str(len(value)).encode("ascii"))
        update(b":")
        update(value)
        return
    if isinstance(value, Mapping):
        update(b"M")
        ordered = sorted(
            value.items(),
            key=lambda item: (type(item[0]).__qualname__, repr(item[0])),
        )
        for key, item in ordered:
            _update_display_digest(hasher, key)
            _update_display_digest(hasher, item)
        update(b";")
        return
    if isinstance(value, (tuple, list)):
        update(b"T")
        for item in value:
            _update_display_digest(hasher, item)
        update(b";")
        return
    if is_dataclass(value) and not isinstance(value, type):
        update(b"D")
        update(type(value).__qualname__.encode("utf-8"))
        for descriptor in fields(value):
            _update_display_digest(hasher, descriptor.name)
            _update_display_digest(hasher, getattr(value, descriptor.name))
        update(b";")
        return
    update(b"R")
    update(type(value).__qualname__.encode("utf-8"))
    update(b":")
    update(repr(value).encode("utf-8"))


def _display_digest(*values: object) -> bytes:
    hasher = sha256()
    for value in values:
        _update_display_digest(hasher, value)
    return hasher.digest()


def _unique_roots(roots: Sequence[Mobject]) -> tuple[Mobject, ...]:
    result: list[Mobject] = []
    seen: set[int] = set()
    for root in roots:
        if id(root) in seen:
            continue
        seen.add(id(root))
        result.append(root)
    return tuple(result)


def _prepare_display_delta(
    previous: Mapping[str, _CommittedDisplaySlot],
    actions: Sequence[_PreparedDisplayAction],
) -> _PreparedDisplayDelta:
    """Compare desired fixed slots with the last committed display state."""

    desired: dict[str, _PreparedDisplayAction] = {}
    root_owners: dict[int, str] = {}
    for action in actions:
        if not isinstance(action.slot_id, str) or not action.slot_id:
            raise QuadricManimError("display slot identity must be a non-empty string")
        if action.slot_id in desired:
            raise QuadricManimError(
                f"display slot {action.slot_id!r} was prepared more than once"
            )
        if not action.roots or not all(isinstance(root, Mobject) for root in action.roots):
            raise QuadricManimError(
                f"display slot {action.slot_id!r} must own fixed Mobject roots"
            )
        for root in action.roots:
            owner = root_owners.setdefault(id(root), action.slot_id)
            if owner != action.slot_id:
                raise QuadricManimError(
                    f"fixed display root is shared by slots {owner!r} and "
                    f"{action.slot_id!r}"
                )
        desired[action.slot_id] = action

    changed: list[_PreparedDisplayAction] = []
    unchanged: list[str] = []
    next_state: dict[str, _CommittedDisplaySlot] = {}
    mutation_roots: list[Mobject] = []
    for action in actions:
        committed = previous.get(action.slot_id)
        if committed is not None and tuple(map(id, committed.roots)) != tuple(
            map(id, action.roots)
        ):
            raise QuadricManimError(
                f"fixed display slot {action.slot_id!r} changed Mobject identity"
            )
        record = _CommittedDisplaySlot(action.roots, bytes(action.digest))
        next_state[action.slot_id] = record
        if committed is not None and committed.digest == action.digest:
            unchanged.append(action.slot_id)
            continue
        changed.append(action)
        mutation_roots.extend(action.roots)

    hidden = tuple(
        previous[slot_id]
        for slot_id in sorted(set(previous) - set(desired))
    )
    for record in hidden:
        mutation_roots.extend(record.roots)
    return _PreparedDisplayDelta(
        tuple(changed),
        tuple(unchanged),
        hidden,
        MappingProxyType(next_state),
        _unique_roots(mutation_roots),
    )


def _hide_mobject_family(root: Mobject) -> None:
    for member in root.get_family():
        if isinstance(member, VMobject):
            _hide_vmobject(member)


def _apply_display_delta(delta: _PreparedDisplayDelta) -> None:
    for record in delta.hidden:
        for root in record.roots:
            _hide_mobject_family(root)
    for action in delta.changed:
        action.apply()


def _painter_band_signature(
    prepared: PreparedPainterBand,
) -> tuple[tuple[str, int, float], ...]:
    return tuple(
        (item.item_id, id(item.mobject), float(item.z_index))
        for item in prepared.items
    )


def _state_value_equal(first: object, second: object) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(first), np.asarray(second)))
        except (TypeError, ValueError):
            return False
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _changed_state_count(states: Sequence[_MobjectState]) -> int:
    changed = 0
    for state in states:
        member = state.mobject
        if state.points is not None and hasattr(member, "points"):
            if not np.array_equal(state.points, np.asarray(member.points)):
                changed += 1
                continue
        if any(
            not _state_value_equal(value, getattr(member, name, None))
            for name, value in state.attributes.items()
        ):
            changed += 1
            continue
        raw_z = getattr(member, "z_index", None)
        current_z = None
        if raw_z is not None:
            value = float(raw_z)
            if np.isfinite(value):
                current_z = value
        if current_z != state.z_index:
            changed += 1
    return changed


def _active_renderable_mobject_count(root: Mobject) -> int:
    result = 0
    seen: set[int] = set()
    for member in root.get_family():
        if id(member) in seen:
            continue
        seen.add(id(member))
        points = np.asarray(getattr(member, "points", ()), dtype=float)
        if points.ndim != 2 or not points.size:
            continue
        for name in ("fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"):
            rgba = np.asarray(getattr(member, name, ()), dtype=float)
            if rgba.ndim >= 2 and rgba.shape[-1] >= 4 and rgba.size:
                if bool(np.any(rgba[..., 3] > 0.0)):
                    result += 1
                    break
    return result


class _ManagedQuadricDisplayGroup(VGroup):
    """Fade proxy whose invisible sentinel owns the lifecycle multiplier."""

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

    def set_opacity(
        self, opacity: float, family: bool = True
    ) -> "_ManagedQuadricDisplayGroup":
        del family
        value = _non_negative(opacity, "display opacity multiplier")
        self._opacity_sentinel.set_stroke(opacity=value)
        return self

    def reset_opacity(self) -> None:
        self._opacity_sentinel.set_stroke(opacity=1.0)


def _hide_vmobject(value: VMobject) -> None:
    value.set_fill(opacity=0.0)
    value.set_stroke(opacity=0.0)
    value.set_stroke(opacity=0.0, background=True)


def _set_closed_subpaths(
    value: VMobject,
    polygons: Sequence[np.ndarray],
) -> None:
    """Replace one fixed VMobject with any number of closed polygon subpaths."""

    value.clear_points()
    for raw in polygons:
        points = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3:
            raise QuadricManimError(
                "section display polygons must contain finite three-dimensional points"
            )
        if not np.all(np.isfinite(points)):
            raise QuadricManimError(
                "section display polygons must contain finite three-dimensional points"
            )
        value.start_new_path(points[0])
        value.add_points_as_corners((*points[1:], points[0]))


def _set_open_subpaths(
    value: VMobject,
    paths: Sequence[np.ndarray],
) -> None:
    """Replace one fixed VMobject with independently open polyline subpaths."""

    value.clear_points()
    for raw in paths:
        points = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
            raise QuadricManimError(
                "open display paths must contain finite 3D polylines"
            )
        if not np.all(np.isfinite(points)):
            raise QuadricManimError(
                "open display paths must contain finite 3D polylines"
            )
        value.start_new_path(points[0])
        value.add_points_as_corners(points[1:])


def _projection_paths_3d(
    paths: Sequence[Sequence[Sequence[float]]],
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray([(x, y, 0.0) for x, y in path], dtype=float)
        for path in paths
    )


def _prepared_cone_fill(layers: ConeProjectionLayers) -> _PreparedConeFill:
    if not isinstance(layers, ConeProjectionLayers):
        raise TypeError("layers must be ConeProjectionLayers")
    return _PreparedConeFill(
        _projection_paths_3d(layers.opaque_lateral_paths),
        _projection_paths_3d(layers.opaque_cap_paths),
        _projection_paths_3d(layers.back.lateral_paths),
        _projection_paths_3d(layers.back.cap_paths),
        _projection_paths_3d(layers.front.lateral_paths),
        _projection_paths_3d(layers.front.cap_paths),
    )


def _style_component_fill(
    value: VMobject,
    paths: Sequence[np.ndarray],
    *,
    colors: tuple[object, ...],
    sheen_direction: Sequence[float],
    opacity: float,
) -> None:
    if not paths:
        _hide_vmobject(value)
        value.clear_points()
        return
    _set_closed_subpaths(value, paths)
    color: object = colors[0] if len(colors) == 1 else colors
    value.set_sheen_direction(np.asarray(sheen_direction, dtype=float))
    value.set_fill(color=color, opacity=opacity)
    value.set_stroke(opacity=0.0)


class _SurfacePaintSlot:
    """One fixed painter item with optional cone component children."""

    def __init__(self) -> None:
        self.base = VMobject()
        self.back_lateral = VMobject()
        self.back_cap = VMobject()
        self.front_lateral = VMobject()
        self.front_cap = VMobject()
        self.root = VGroup(
            self.back_lateral,
            self.back_cap,
            self.front_lateral,
            self.front_cap,
            self.base,
        )
        self.hide()

    def hide(self) -> None:
        _hide_vmobject(self.base)
        for component in self.components:
            _hide_vmobject(component)

    @property
    def components(self) -> tuple[VMobject, ...]:
        return (
            self.back_lateral,
            self.back_cap,
            self.front_lateral,
            self.front_cap,
        )

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


def _apply_opaque_surface_slot(
    slot: _SurfacePaintSlot,
    points: np.ndarray,
    cone_fill: _PreparedConeFill | None,
    style: _StyleContract,
    opacity: float,
    *,
    draw_stroke: bool,
) -> None:
    """Apply one opaque projection proxy to a fixed surface slot."""

    representative_opacity = style.surface_fill_opacity * opacity
    slot.root.set_fill(
        color=style.surface_fill_color,
        opacity=representative_opacity,
        family=False,
    )
    slot.base.set_points_as_corners(points)
    slot.base.set_stroke(
        color=style.surface_stroke_color,
        width=style.surface_stroke_width,
        opacity=style.surface_stroke_opacity * opacity if draw_stroke else 0.0,
    )
    if cone_fill is None:
        slot.base.set_fill(
            color=style.surface_fill_color,
            opacity=representative_opacity,
        )
        for component in slot.components:
            _hide_vmobject(component)
        return

    slot.base.set_fill(opacity=0.0)
    lateral_colors = style.cone_lateral_fill_colors or (
        style.surface_fill_color,
    )
    cap_colors = style.cone_cap_fill_colors or lateral_colors
    sheet_opacity = 1.0 - sqrt(
        max(0.0, 1.0 - min(1.0, representative_opacity))
    )
    for component, paths, colors, direction in (
        (
            slot.back_lateral,
            cone_fill.back_lateral_paths,
            lateral_colors,
            style.cone_lateral_sheen_direction,
        ),
        (
            slot.back_cap,
            cone_fill.back_cap_paths,
            cap_colors,
            style.cone_cap_sheen_direction,
        ),
        (
            slot.front_lateral,
            cone_fill.front_lateral_paths,
            lateral_colors,
            style.cone_lateral_sheen_direction,
        ),
        (
            slot.front_cap,
            cone_fill.front_cap_paths,
            cap_colors,
            style.cone_cap_sheen_direction,
        ),
    ):
        _style_component_fill(
            component,
            paths,
            colors=colors,
            sheen_direction=direction,
            opacity=sheet_opacity,
        )


def _apply_surface_sheet_pair(
    back: _SurfacePaintSlot,
    front: _SurfacePaintSlot,
    points: np.ndarray,
    cone_fill: _PreparedConeFill | None,
    style: _StyleContract,
    opacity: float,
    *,
    configure_front_stroke: bool,
    draw_front_stroke: bool,
) -> None:
    """Apply matching back/front sheets without duplicating component styling."""

    back.base.set_points_as_corners(points)
    front.base.set_points_as_corners(points)
    combined = min(1.0, style.surface_fill_opacity * opacity)
    sheet_opacity = 1.0 - sqrt(max(0.0, 1.0 - combined))
    for slot in (back, front):
        slot.root.set_fill(
            color=style.surface_fill_color,
            opacity=sheet_opacity,
            family=False,
        )
    back.base.set_stroke(opacity=0.0)
    if configure_front_stroke:
        front.base.set_stroke(
            color=style.surface_stroke_color,
            width=style.surface_stroke_width,
            opacity=(
                style.surface_stroke_opacity * opacity
                if draw_front_stroke
                else 0.0
            ),
        )
    else:
        front.base.set_stroke(opacity=0.0)

    if cone_fill is None:
        for slot in (back, front):
            slot.base.set_fill(
                color=style.surface_fill_color,
                opacity=sheet_opacity,
            )
            for component in slot.components:
                _hide_vmobject(component)
        return

    back.base.set_fill(opacity=0.0)
    front.base.set_fill(opacity=0.0)
    lateral_colors = style.cone_lateral_fill_colors or (
        style.surface_fill_color,
    )
    cap_colors = style.cone_cap_fill_colors or lateral_colors
    for slot, lateral, cap, lateral_paths, cap_paths in (
        (
            back,
            back.back_lateral,
            back.back_cap,
            cone_fill.back_lateral_paths,
            cone_fill.back_cap_paths,
        ),
        (
            front,
            front.front_lateral,
            front.front_cap,
            cone_fill.front_lateral_paths,
            cone_fill.front_cap_paths,
        ),
    ):
        for component in slot.components:
            _hide_vmobject(component)
        _style_component_fill(
            lateral,
            lateral_paths,
            colors=lateral_colors,
            sheen_direction=style.cone_lateral_sheen_direction,
            opacity=sheet_opacity,
        )
        _style_component_fill(
            cap,
            cap_paths,
            colors=cap_colors,
            sheen_direction=style.cone_cap_sheen_direction,
            opacity=sheet_opacity,
        )


class _CurveFragmentSlot:
    def __init__(self, dash_capacity: int) -> None:
        self.dash_capacity = int(dash_capacity)
        self.solid = VMobject()
        self.dashed = VMobject()
        self.root = VGroup(self.solid, self.dashed)
        self.hide()

    def hide(self) -> None:
        _hide_vmobject(self.solid)
        _hide_vmobject(self.dashed)

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


class _CurveSlots:
    def __init__(self, fragment_capacity: int, dash_capacity: int) -> None:
        self.fragments = tuple(
            _CurveFragmentSlot(dash_capacity) for _ in range(fragment_capacity)
        )
        self.root = VGroup(*(slot.root for slot in self.fragments))

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


def _curve_slots_family_capacity(fragment_capacity: int) -> int:
    """Return the fixed family size for one compact boundary-source slot."""

    return 1 + 3 * int(fragment_capacity)


def _coerce_view(value: object) -> ParallelView:
    if isinstance(value, ParallelView):
        return value
    try:
        return ParallelView.from_matrix(value)  # type: ignore[arg-type]
    except (SolverError, TypeError, ValueError) as exc:
        raise QuadricManimError(f"invalid parallel projection: {exc}") from exc


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    delta = end - start
    squared = float(np.dot(delta, delta))
    if squared == 0.0:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, delta) / squared)
    ratio = min(1.0, max(0.0, ratio))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def _adaptive_project_curve(
    curve: AnalyticCurve3D,
    view: ParallelView,
    start: float,
    end: float,
    *,
    max_chord_error: float,
    max_segments: int,
) -> np.ndarray:
    """Approximate one exact analytic interval without renderer allocation."""

    screen = view.matrix[:2]
    cache: dict[float, np.ndarray] = {}
    domain = curve.domain
    parameter_scale = max(
        1.0,
        abs(float(domain.start)),
        abs(float(domain.end)),
    )
    parameter_roundoff = max(
        16.0 * np.finfo(float).eps * parameter_scale,
        4.0 * abs(float(np.spacing(domain.start))),
        4.0 * abs(float(np.spacing(domain.end))),
    )

    def canonical_parameter(parameter: float) -> float:
        value = float(parameter)
        if (
            value < domain.start - parameter_roundoff
            or value > domain.end + parameter_roundoff
        ):
            raise QuadricManimError(
                f"curve {curve.curve_id!r} display interval lies outside its "
                "authored parameter domain"
            )
        return min(domain.end, max(domain.start, value))

    def project(parameter: float) -> np.ndarray:
        key = canonical_parameter(parameter)
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = np.asarray(curve.point(key), dtype=float)
        projected = np.asarray(screen @ value, dtype=float)
        if projected.shape != (2,) or not np.all(np.isfinite(projected)):
            raise QuadricManimError(
                f"curve {curve.curve_id!r} produced a non-finite projection"
            )
        cache[key] = projected
        return projected

    interval_start = canonical_parameter(start)
    interval_end = canonical_parameter(end)
    if interval_end <= interval_start:
        raise QuadricManimError(
            f"curve {curve.curve_id!r} display interval must have positive length"
        )
    intervals: list[tuple[float, float]] = [(interval_start, interval_end)]
    probe_fractions = (0.25, 0.5, 0.75)
    while True:
        split: list[int] = []
        for index, (left, right) in enumerate(intervals):
            first = project(left)
            last = project(right)
            observed = max(
                _point_segment_distance(
                    project(left + fraction * (right - left)), first, last
                )
                for fraction in probe_fractions
            )
            if observed > max_chord_error:
                split.append(index)
        if not split:
            break
        if len(intervals) + len(split) > max_segments:
            raise QuadricManimCapacityError(
                f"curve {curve.curve_id!r} needs more than {max_segments} "
                "display segments for max_chord_error"
            )
        marked = set(split)
        refined: list[tuple[float, float]] = []
        for index, (left, right) in enumerate(intervals):
            if index not in marked:
                refined.append((left, right))
                continue
            middle = left + 0.5 * (right - left)
            if middle == left or middle == right:
                raise QuadricManimCapacityError(
                    f"curve {curve.curve_id!r} cannot refine at floating-point scale"
                )
            refined.extend(((left, middle), (middle, right)))
        intervals = refined

    parameters = [intervals[0][0]]
    parameters.extend(right for _left, right in intervals)
    points = [project(parameter) for parameter in parameters]
    precision_floor = 4.0 * max(
        (
            abs(float(np.spacing(value)))
            for point in points
            for value in point
        ),
        default=0.0,
    )
    if precision_floor >= max_chord_error:
        raise QuadricManimError(
            f"curve {curve.curve_id!r} cannot certify max_chord_error at the "
            "available floating-point screen resolution; requested "
            f"{max_chord_error:.17g}, resolution floor {precision_floor:.17g}"
        )
    anchor = points[0]

    def duplicate_tolerance(left: np.ndarray, right: np.ndarray) -> float:
        local_scale = max(
            float(np.linalg.norm(left - anchor)),
            float(np.linalg.norm(right - anchor)),
            max_chord_error,
            np.finfo(float).tiny,
        )
        local_roundoff = 32.0 * np.finfo(float).eps * local_scale
        ulp_roundoff = 2.0 * max(
            *(abs(float(np.spacing(value))) for value in left),
            *(abs(float(np.spacing(value))) for value in right),
        )
        return min(
            max(local_roundoff, ulp_roundoff),
            0.125 * max_chord_error,
        )

    result: list[np.ndarray] = [points[0]]
    source_to_result = [0]
    for point in points[1:]:
        if float(np.linalg.norm(point - result[-1])) > duplicate_tolerance(
            result[-1], point
        ):
            result.append(point)
        source_to_result.append(len(result) - 1)
    if len(result) < 2:
        raise QuadricManimError(
            f"curve {curve.curve_id!r} interval collapses in the selected projection"
        )

    measured_error = 0.0
    certification_fractions = (0.0, *probe_fractions, 1.0)
    for index, (left, right) in enumerate(intervals):
        chord_start = result[source_to_result[index]]
        chord_end = result[source_to_result[index + 1]]
        for fraction in certification_fractions:
            parameter = left + fraction * (right - left)
            measured_error = max(
                measured_error,
                _point_segment_distance(
                    project(parameter),
                    chord_start,
                    chord_end,
                ),
            )
    certified_error = measured_error + precision_floor
    if certified_error > max_chord_error * (
        1.0 + 64.0 * np.finfo(float).eps
    ):
        raise QuadricManimError(
            f"curve {curve.curve_id!r} cannot certify max_chord_error after "
            "floating-point-stable deduplication; requested "
            f"{max_chord_error:.17g}, observed {certified_error:.17g}"
        )
    return np.asarray([(point[0], point[1], 0.0) for point in result], dtype=float)


def _polyline_lengths(points: np.ndarray) -> tuple[np.ndarray, float]:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate((np.asarray((0.0,)), np.cumsum(segment_lengths)))
    return cumulative, float(cumulative[-1])


def _point_at_distance(
    points: np.ndarray, cumulative: np.ndarray, distance: float
) -> np.ndarray:
    if distance <= 0.0:
        return points[0].copy()
    if distance >= float(cumulative[-1]):
        return points[-1].copy()
    index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    index = min(index, len(points) - 2)
    span = float(cumulative[index + 1] - cumulative[index])
    if span <= 0.0:
        return points[index].copy()
    ratio = (distance - float(cumulative[index])) / span
    return points[index] + ratio * (points[index + 1] - points[index])


def _slice_polyline(
    points: np.ndarray,
    cumulative: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    values = [_point_at_distance(points, cumulative, start)]
    for index in range(1, len(points) - 1):
        distance = float(cumulative[index])
        if start < distance < end:
            values.append(points[index].copy())
    values.append(_point_at_distance(points, cumulative, end))
    return np.asarray(values, dtype=float)


def _dash_polyline(
    points: np.ndarray,
    *,
    dash_length: float,
    dash_gap: float,
    capacity: int,
) -> tuple[_PreparedDash, ...]:
    cumulative, length = _polyline_lengths(points)
    if length <= 0.0:
        return ()
    period = dash_length + dash_gap
    result: list[_PreparedDash] = []
    period_index = 0
    while period_index * period < length - 1.0e-12:
        start = period_index * period
        end = min(length, start + dash_length)
        period_index += 1
        if end - start <= 1.0e-12:
            continue
        result.append(_PreparedDash(_slice_polyline(points, cumulative, start, end)))
        if len(result) > capacity:
            raise QuadricManimCapacityError(
                f"dash count exceeds fixed slot capacity {capacity}"
            )
    return tuple(result)


def _adaptive_project_curve_samples(
    curve: AnalyticCurve3D,
    view: ParallelView,
    *,
    max_chord_error: float,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-source parameters and points for stable dash phase."""

    projection = view.matrix[:2]
    cache: dict[float, np.ndarray] = {}

    def project(parameter: float) -> np.ndarray:
        key = float(parameter)
        if key not in cache:
            value = projection @ np.asarray(curve.point(key), dtype=float)
            cache[key] = np.asarray((value[0], value[1], 0.0), dtype=float)
        return cache[key]

    intervals = [(curve.domain.start, curve.domain.end)]
    probes = (0.25, 0.5, 0.75)
    while True:
        split = []
        for index, (left, right) in enumerate(intervals):
            first = project(left)
            last = project(right)
            observed = max(
                _point_segment_distance(
                    project(left + fraction * (right - left)),
                    first,
                    last,
                )
                for fraction in probes
            )
            if observed > max_chord_error:
                split.append(index)
        if not split:
            break
        if len(intervals) + len(split) > max_segments:
            raise QuadricManimCapacityError(
                f"boundary source {curve.curve_id!r} needs more than "
                f"{max_segments} display segments"
            )
        marked = set(split)
        refined = []
        for index, (left, right) in enumerate(intervals):
            if index not in marked:
                refined.append((left, right))
                continue
            middle = left + 0.5 * (right - left)
            if middle == left or middle == right:
                raise QuadricManimCapacityError(
                    f"boundary source {curve.curve_id!r} cannot refine at "
                    "floating-point resolution"
                )
            refined.extend(((left, middle), (middle, right)))
        intervals = refined
    parameters = np.asarray(
        [intervals[0][0], *(right for _left, right in intervals)],
        dtype=float,
    )
    points = np.asarray([project(float(value)) for value in parameters], dtype=float)
    return parameters, points


def _source_distance_at_parameter(
    parameters: np.ndarray,
    points: np.ndarray,
    parameter: float,
) -> float:
    cumulative, _length = _polyline_lengths(points)
    value = float(parameter)
    if value <= float(parameters[0]):
        return 0.0
    if value >= float(parameters[-1]):
        return float(cumulative[-1])
    index = int(np.searchsorted(parameters, value, side="right") - 1)
    index = min(index, len(parameters) - 2)
    span = float(parameters[index + 1] - parameters[index])
    ratio = 0.0 if span <= 0.0 else (value - float(parameters[index])) / span
    segment = float(np.linalg.norm(points[index + 1] - points[index]))
    return float(cumulative[index]) + ratio * segment


def _dash_polyline_anchored(
    points: np.ndarray,
    *,
    source_distance_start: float,
    dash_length: float,
    dash_gap: float,
    capacity: int,
) -> tuple[_PreparedDash, ...]:
    cumulative, length = _polyline_lengths(points)
    if length <= 0.0:
        return ()
    period = dash_length + dash_gap
    global_start = float(source_distance_start)
    global_end = global_start + length
    first_period = max(
        0,
        int(np.floor((global_start - dash_length) / period)) + 1,
    )
    result: list[_PreparedDash] = []
    period_index = first_period
    while period_index * period < global_end - 1.0e-12:
        dash_start = period_index * period
        dash_end = dash_start + dash_length
        period_index += 1
        clipped_start = max(global_start, dash_start) - global_start
        clipped_end = min(global_end, dash_end) - global_start
        if clipped_end - clipped_start <= 1.0e-12:
            continue
        result.append(
            _PreparedDash(
                _slice_polyline(points, cumulative, clipped_start, clipped_end)
            )
        )
        if len(result) > capacity:
            raise QuadricManimCapacityError(
                f"dash count exceeds fixed slot capacity {capacity}"
            )
    return tuple(result)


def _assign_fragment_slots(
    source_id: str,
    active_ids: Sequence[str],
    *,
    previous: Mapping[str, int],
    capacity: int,
) -> dict[str, int]:
    active = tuple(active_ids)
    if len(active) > capacity:
        raise QuadricManimCapacityError(
            f"boundary source {source_id!r} has {len(active)} painted fragments; "
            f"fixed capacity is {capacity}"
        )
    result = {
        item_id: previous[item_id] for item_id in active if item_id in previous
    }
    used = set(result.values())
    free = iter(index for index in range(capacity) if index not in used)
    for item_id in active:
        if item_id not in result:
            result[item_id] = next(free)
    return result


def _prepare_boundary_fragments(
    *,
    sources: Sequence[QuadricBoundarySource],
    frame: QuadricBoundaryCompositingFrame,
    view: ParallelView,
    style_for_source: Callable[[QuadricBoundarySource], QuadricBoundaryStyle],
    previous_slot_maps: Mapping[str, Mapping[str, int]],
    curve_slots: Mapping[str, _CurveSlots],
    slot_source_ids: Sequence[str],
    max_chord_error: float,
    limits: _LimitsContract,
    performance_attempt: _PerformanceAttempt | None = None,
) -> _PreparedBoundaryBatch:
    """Prepare every painted semantic boundary without allocating Mobjects."""

    source_map = {item.source_id: item for item in sources}
    by_source: dict[str, list[QuadricBoundaryPaintFragment]] = {
        item.source_id: [] for item in sources
    }
    for fragment in frame.fragments:
        if fragment.painted:
            by_source[fragment.source_id].append(fragment)
    next_maps: dict[str, Mapping[str, int]] = {
        source_id: {} for source_id in slot_source_ids
    }
    prepared_by_source: dict[
        str, tuple[_PreparedBoundaryFragment, ...]
    ] = {}
    item_mobjects: dict[str, Mobject] = {}
    for source_id in sorted(by_source):
        source = source_map[source_id]
        style = style_for_source(source)
        fragments = tuple(
            sorted(by_source[source_id], key=lambda item: item.item_id)
        )
        assignment = _assign_fragment_slots(
            source_id,
            tuple(item.item_id for item in fragments),
            previous=previous_slot_maps[source_id],
            capacity=limits.max_fragments_per_curve,
        )
        next_maps[source_id] = assignment
        with _performance_stage(performance_attempt, "adaptive_projection"):
            parameters, source_points = _adaptive_project_curve_samples(
                source.curve,
                view,
                max_chord_error=max_chord_error,
                max_segments=limits.max_segments_per_fragment,
            )
        values: list[_PreparedBoundaryFragment] = []
        for fragment in fragments:
            with _performance_stage(performance_attempt, "adaptive_projection"):
                points = _adaptive_project_curve(
                    source.curve,
                    view,
                    fragment.interval.start,
                    fragment.interval.end,
                    max_chord_error=max_chord_error,
                    max_segments=limits.max_segments_per_fragment,
                )
            _cumulative, length = _polyline_lengths(points)
            allowance = max(
                1.0e-12,
                limits.max_projected_length * 1.0e-9,
            )
            if length > limits.max_projected_length + allowance:
                raise QuadricManimCapacityError(
                    f"boundary source {source_id!r} fragment length "
                    f"{length:.9g} exceeds max_projected_length"
                )
            with _performance_stage(performance_attempt, "dash_generation"):
                dashes = (
                    _dash_polyline_anchored(
                        points,
                        source_distance_start=_source_distance_at_parameter(
                            parameters,
                            source_points,
                            fragment.interval.start,
                        ),
                        dash_length=style.dash_length,
                        dash_gap=style.dash_gap,
                        capacity=limits.max_dashes_per_fragment,
                    )
                    if fragment.render_intent is BoundaryRenderIntent.DASHED
                    else ()
                )
            if performance_attempt is not None:
                performance_attempt.increment_count(
                    "prepared_boundary_fragment_count"
                )
                performance_attempt.increment_count(
                    "prepared_dash_count", len(dashes)
                )
            slot_index = assignment[fragment.item_id]
            values.append(
                _PreparedBoundaryFragment(
                    fragment,
                    source,
                    style,
                    slot_index,
                    points,
                    dashes,
                )
            )
            item_mobjects[fragment.item_id] = curve_slots[
                source_id
            ].fragments[slot_index].root
        prepared_by_source[source_id] = tuple(values)
    return _PreparedBoundaryBatch(
        fragments=prepared_by_source,
        fragment_slot_maps=next_maps,
        item_mobjects=item_mobjects,
    )


def _apply_boundary_fragment(
    curve_slots: Mapping[str, _CurveSlots],
    source_id: str,
    prepared: _PreparedBoundaryFragment,
    opacity: float,
) -> None:
    """Apply one prepared semantic-boundary fragment to its fixed slot."""

    slot = curve_slots[source_id].fragments[prepared.slot_index]
    hidden = prepared.fragment.render_intent is BoundaryRenderIntent.DASHED
    style = prepared.style
    color = style.hidden_color if hidden else style.visible_color
    width = style.hidden_width if hidden else style.visible_width
    stroke_opacity = (
        style.hidden_opacity if hidden else style.visible_opacity
    ) * opacity
    if prepared.fragment.render_intent is BoundaryRenderIntent.SOLID:
        slot.solid.set_points_as_corners(prepared.points)
        slot.solid.set_fill(opacity=0.0)
        slot.solid.set_stroke(color=color, width=width, opacity=stroke_opacity)
        slot.solid.set_stroke(
            color=style.background_color,
            width=style.background_width,
            opacity=style.background_opacity * opacity,
            background=True,
        )
        if style.cap_style is not None:
            slot.solid.set_cap_style(style.cap_style)
        if style.joint_type is not None:
            slot.solid.joint_type = style.joint_type
        _hide_vmobject(slot.dashed)
        return
    _hide_vmobject(slot.solid)
    _set_open_subpaths(
        slot.dashed,
        tuple(item.points for item in prepared.dashes),
    )
    slot.dashed.set_fill(opacity=0.0)
    slot.dashed.set_stroke(color=color, width=width, opacity=stroke_opacity)
    slot.dashed.set_stroke(
        color=style.background_color,
        width=style.background_width,
        opacity=style.background_opacity * opacity,
        background=True,
    )
    cap = (
        style.cap_style
        if style.hidden_cap_style is None
        else style.hidden_cap_style
    )
    joint = (
        style.joint_type
        if style.hidden_joint_type is None
        else style.hidden_joint_type
    )
    if cap is not None:
        slot.dashed.set_cap_style(cap)
    if joint is not None:
        slot.dashed.joint_type = joint


_ControllerState = TypeVar("_ControllerState")


@contextmanager
def _rollback_display_transaction(
    root: Mobject,
    band: ManagedPainterBand,
    *,
    capture_controller_state: Callable[[], _ControllerState],
    restore_controller_state: Callable[[_ControllerState], None],
    mutation_roots: Sequence[Mobject] | None = None,
    performance_attempt: _PerformanceAttempt | None = None,
) -> Iterator[None]:
    """Rollback changed display roots, painter band, and bookkeeping together."""

    capture_targets = (root,) if mutation_roots is None else tuple(mutation_roots)
    with _performance_stage(performance_attempt, "transaction_snapshot"):
        root_state = _capture_roots(capture_targets)
    if performance_attempt is not None:
        performance_attempt.set_count(
            "mobject_family_count", len(_unique_family((root,)))
        )
        performance_attempt.set_count(
            "transaction_snapshot_mobject_count", len(root_state)
        )
    band_state = band.capture_active_state()
    controller_state = capture_controller_state()
    try:
        yield
    except Exception:
        if performance_attempt is not None:
            performance_attempt.set_count(
                "modified_mobject_count", _changed_state_count(root_state)
            )
        with _performance_stage(performance_attempt, "transaction_rollback"):
            _restore_root(root_state)
            band.restore_active_state(band_state)
            restore_controller_state(controller_state)
        raise
    else:
        if performance_attempt is not None:
            performance_attempt.set_count(
                "modified_mobject_count", _changed_state_count(root_state)
            )
            performance_attempt.set_count(
                "active_mobject_count",
                _active_renderable_mobject_count(root),
            )


def _scene_containers(scene: object) -> tuple[list[object], ...]:
    result: list[list[object]] = []
    for name in (
        "mobjects",
        "foreground_mobjects",
        "moving_mobjects",
        "static_mobjects",
    ):
        value = getattr(scene, name, None)
        if isinstance(value, list) and all(value is not item for item in result):
            result.append(value)
    return tuple(result)


def _register_fixed_frame(scene: object, root: Mobject) -> ThreeDCamera | None:
    camera = getattr(scene, "camera", None)
    if not isinstance(camera, ThreeDCamera):
        return None
    camera.add_fixed_in_frame_mobjects(root)
    return camera


def _remove_fixed_frame(camera: ThreeDCamera | None, root: Mobject) -> None:
    if camera is not None:
        camera.remove_fixed_in_frame_mobjects(root)


def _remove_owned_identities(
    scene: object,
    root: Mobject,
    update_driver: Mobject,
) -> None:
    owned = {id(item) for item in root.get_family()}
    owned.update(id(item) for item in update_driver.get_family())
    for container in _scene_containers(scene):
        container[:] = [item for item in container if id(item) not in owned]


def _invalidate_cairo_static_image(scene: object) -> None:
    renderer = getattr(scene, "renderer", None)
    if renderer is not None and hasattr(renderer, "static_image"):
        renderer.static_image = None
