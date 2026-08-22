from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import Callable, Iterator, Literal, Mapping, Sequence

import numpy as np
from manim import Line, Mobject, RendererType, ThreeDCamera, VGroup, config

from .contract import StrokeSpec, TolerancePolicy, VisibilityModel
from .parallel_solver import compute_frame_visibility
from .style import OcclusionStyle, ResolvedOcclusionStyle
from .trace import EdgeVisibility, VisibilityFrame


class OcclusionBindingError(RuntimeError):
    """Raised when a Manim binding cannot be attached safely."""


class OcclusionCapacityError(OcclusionBindingError):
    """Raised before mutation when a frame exceeds fixed overlay capacity."""


def _using_cairo_renderer() -> bool:
    return config.renderer == RendererType.CAIRO


@dataclass(frozen=True)
class OverlayCapacity:
    visible_slots: int
    hidden_slots: int
    dash_slots_per_hidden: int
    max_projected_length: float

    @classmethod
    def for_stroke(
        cls,
        stroke: StrokeSpec,
        model: VisibilityModel,
        style: OcclusionStyle,
    ) -> "OverlayCapacity":
        candidate_count = sum(
            1
            for face in model.faces
            if face.occludes_strokes and face.face_id not in stroke.incident_face_ids
        )
        hidden_slots = candidate_count
        if stroke.visibility_mode == "always_hidden":
            hidden_slots = max(1, hidden_slots)
        # A union of K one-dimensional intervals has at most K hidden
        # components and K+1 visible components.  Extra dash objects cover
        # interval clipping without any updater-time Mobject allocation.
        dash_capacity = (
            int(ceil(style.max_projected_length / style.dash_period))
            + hidden_slots
            + 1
        )
        return cls(
            visible_slots=candidate_count + 1,
            hidden_slots=hidden_slots,
            dash_slots_per_hidden=dash_capacity,
            max_projected_length=style.max_projected_length,
        )


@dataclass(frozen=True)
class PlannedDash:
    start: tuple[float, float, float]
    end: tuple[float, float, float]


@dataclass(frozen=True)
class PlannedSegment:
    start_parameter: float
    end_parameter: float
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    dashes: tuple[PlannedDash, ...] = ()


@dataclass(frozen=True)
class OverlayPlan:
    visible_segments: tuple[PlannedSegment, ...]
    hidden_segments: tuple[PlannedSegment, ...]


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise OcclusionBindingError(f"{label} must be a finite three-component point")
    return point


def _segment(
    start_parameter: float,
    end_parameter: float,
    start: np.ndarray,
    delta: np.ndarray,
    *,
    dashes: tuple[PlannedDash, ...] = (),
) -> PlannedSegment:
    first = start + float(start_parameter) * delta
    last = start + float(end_parameter) * delta
    return PlannedSegment(
        float(start_parameter),
        float(end_parameter),
        tuple(float(item) for item in first),
        tuple(float(item) for item in last),
        dashes,
    )


def _coalesced_intervals(edge: EdgeVisibility, kind: str) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for span in edge.spans:
        if span.kind != kind:
            continue
        if result and abs(result[-1][1] - span.start) <= 1.0e-12:
            result[-1] = (result[-1][0], span.end)
        else:
            result.append((span.start, span.end))
    return result


def build_overlay_plan(
    edge: EdgeVisibility,
    *,
    display_start: Sequence[float],
    display_end: Sequence[float],
    capacity: OverlayCapacity,
    style: OcclusionStyle,
) -> OverlayPlan:
    """Turn a core visibility trace into a mutation-free render plan."""

    first = _point3(display_start, "display stroke start")
    last = _point3(display_end, "display stroke end")
    delta = last - first
    display_length = float(np.linalg.norm(delta))
    allowance = max(1.0e-12, capacity.max_projected_length * 1.0e-9)
    if display_length > capacity.max_projected_length + allowance:
        raise OcclusionCapacityError(
            "projected length "
            f"{display_length:.9g} exceeds fixed maximum "
            f"{capacity.max_projected_length:.9g}"
        )

    visible_intervals = _coalesced_intervals(edge, "visible")
    hidden_intervals = _coalesced_intervals(edge, "hidden")
    if len(visible_intervals) > capacity.visible_slots:
        raise OcclusionCapacityError(
            f"visible component count {len(visible_intervals)} exceeds slot capacity "
            f"{capacity.visible_slots}"
        )
    if len(hidden_intervals) > capacity.hidden_slots:
        raise OcclusionCapacityError(
            f"hidden component count {len(hidden_intervals)} exceeds slot capacity "
            f"{capacity.hidden_slots}"
        )

    visible = tuple(
        _segment(begin, finish, first, delta)
        for begin, finish in visible_intervals
    )
    hidden: list[PlannedSegment] = []
    for begin, finish in hidden_intervals:
        planned_dashes: list[PlannedDash] = []
        if display_length > 1.0e-12:
            hidden_distance_start = begin * display_length
            hidden_distance_end = finish * display_length
            # Dash phase is anchored to source-edge t=0.  Moving an occlusion
            # boundary only clips the first/last dash; it does not restart the
            # pattern and make all interior dashes crawl.
            first_period = max(
                0,
                int(np.floor((hidden_distance_start - style.dash_length) / style.dash_period))
                + 1,
            )
            period_index = first_period
            while period_index * style.dash_period < hidden_distance_end - 1.0e-12:
                dash_start_distance = period_index * style.dash_period
                dash_end_distance = dash_start_distance + style.dash_length
                clipped_start = max(hidden_distance_start, dash_start_distance)
                clipped_end = min(hidden_distance_end, dash_end_distance)
                period_index += 1
                if clipped_end - clipped_start <= 1.0e-12:
                    continue
                dash_begin_parameter = clipped_start / display_length
                dash_end_parameter = clipped_end / display_length
                dash_start = first + dash_begin_parameter * delta
                dash_finish = first + dash_end_parameter * delta
                planned_dashes.append(
                    PlannedDash(
                        tuple(float(item) for item in dash_start),
                        tuple(float(item) for item in dash_finish),
                    )
                )
        if len(planned_dashes) > capacity.dash_slots_per_hidden:
            raise OcclusionCapacityError(
                f"dash count {len(planned_dashes)} exceeds slot capacity "
                f"{capacity.dash_slots_per_hidden}"
            )
        hidden.append(
            _segment(
                begin,
                finish,
                first,
                delta,
                dashes=tuple(planned_dashes),
            )
        )
    return OverlayPlan(visible, tuple(hidden))


@dataclass
class _AttributeSnapshot:
    mobject: object
    attributes: dict[str, object]


_RGBA_ATTRIBUTES = ("stroke_rgbas", "background_stroke_rgbas")
_OPACITY_ATTRIBUTES = ("stroke_opacity", "background_stroke_opacity")


def _copy_attribute(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _capture_family_style(source: Mobject) -> tuple[_AttributeSnapshot, ...]:
    snapshots: list[_AttributeSnapshot] = []
    for member in source.get_family():
        attributes: dict[str, object] = {}
        for name in (*_RGBA_ATTRIBUTES, *_OPACITY_ATTRIBUTES):
            if hasattr(member, name):
                attributes[name] = _copy_attribute(getattr(member, name))
        snapshots.append(_AttributeSnapshot(member, attributes))
    return tuple(snapshots)


def _hide_snapshots(snapshots: Sequence[_AttributeSnapshot]) -> None:
    for snapshot in snapshots:
        for name in _RGBA_ATTRIBUTES:
            if name not in snapshot.attributes:
                continue
            value = np.asarray(snapshot.attributes[name], dtype=float).copy()
            if value.ndim >= 1 and value.shape[-1] >= 4:
                value[..., 3] = 0.0
            setattr(snapshot.mobject, name, value)
        for name in _OPACITY_ATTRIBUTES:
            if name in snapshot.attributes:
                setattr(snapshot.mobject, name, 0.0)


def _restore_snapshots(snapshots: Sequence[_AttributeSnapshot]) -> None:
    for snapshot in snapshots:
        for name, value in snapshot.attributes.items():
            setattr(snapshot.mobject, name, _copy_attribute(value))


class _StrokeSlots:
    def __init__(self, capacity: OverlayCapacity) -> None:
        self.visible = [Line((0, 0, 0), (1, 0, 0), buff=0) for _ in range(capacity.visible_slots)]
        self.hidden = [
            [Line((0, 0, 0), (1, 0, 0), buff=0) for _ in range(capacity.dash_slots_per_hidden)]
            for _ in range(capacity.hidden_slots)
        ]
        for line in self.visible:
            self._hide_line(line)
        hidden_groups: list[VGroup] = []
        for dashes in self.hidden:
            for line in dashes:
                self._hide_line(line)
            hidden_groups.append(VGroup(*dashes))
        self.hidden_groups = hidden_groups
        self.root = VGroup(*self.visible, *hidden_groups)

    def identities(self) -> tuple[int, ...]:
        return tuple(
            [
                id(self.root),
                *(id(item) for item in self.visible),
                *(id(item) for item in self.hidden_groups),
            ]
            + [id(item) for group in self.hidden for item in group]
        )

    def apply_static_style(self, style: ResolvedOcclusionStyle) -> None:
        lines = [*self.visible, *(line for slot in self.hidden for line in slot)]
        for line in lines:
            if style.cap_style is not None:
                line.set_cap_style(style.cap_style)
            if style.joint_type is not None:
                line.joint_type = style.joint_type
            line.set_stroke(
                color=style.background_color,
                width=style.background_width,
                opacity=0,
                background=True,
            )

    @staticmethod
    def _hide_line(line: Line) -> None:
        line.set_stroke(opacity=0)
        line.set_stroke(opacity=0, background=True)

    def apply(self, plan: OverlayPlan, style: ResolvedOcclusionStyle) -> None:
        for index, line in enumerate(self.visible):
            if index >= len(plan.visible_segments):
                self._hide_line(line)
                continue
            segment = plan.visible_segments[index]
            line.put_start_and_end_on(segment.start, segment.end)
            line.set_stroke(
                color=style.visible_color,
                width=style.visible_width,
                opacity=style.visible_opacity,
            )
            line.set_stroke(
                color=style.background_color,
                width=style.background_width,
                opacity=style.background_opacity,
                background=True,
            )
        for slot_index, dash_slot in enumerate(self.hidden):
            dashes = (
                plan.hidden_segments[slot_index].dashes
                if slot_index < len(plan.hidden_segments)
                else ()
            )
            for dash_index, line in enumerate(dash_slot):
                if dash_index >= len(dashes):
                    self._hide_line(line)
                    continue
                dash = dashes[dash_index]
                line.put_start_and_end_on(dash.start, dash.end)
                line.set_stroke(
                    color=style.hidden_color,
                    width=style.hidden_width,
                    opacity=style.hidden_opacity,
                )
                line.set_stroke(
                    color=style.background_color,
                    width=style.background_width,
                    opacity=style.background_opacity,
                    background=True,
                )


PositionProvider = Callable[[], Mapping[str, Sequence[float]]]
ProjectionProvider = Callable[[object], Sequence[Sequence[float]]]
DisplayPointProvider = Callable[[Sequence[float]], Sequence[float]]


def _drawable_member(mobject: object) -> bool:
    points = np.asarray(getattr(mobject, "points", np.empty((0, 3))))
    if points.size == 0:
        return False
    rgba_found = False
    for attribute in ("fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"):
        if not hasattr(mobject, attribute):
            continue
        rgba_found = True
        rgba = np.asarray(getattr(mobject, attribute), dtype=float)
        if rgba.ndim >= 1 and rgba.shape[-1] >= 4 and np.any(rgba[..., 3] > 0):
            return True
    # Unknown point-bearing Mobjects fail conservatively as drawable.
    return not rgba_found


class ManimOcclusionBinding:
    """Stable Cairo binding for a fixed-topology visibility model."""

    def __init__(
        self,
        scene: object,
        model: VisibilityModel,
        *,
        position_provider: PositionProvider,
        stroke_bindings: Mapping[str, Mobject],
        projection_provider: ProjectionProvider,
        display_point_provider: DisplayPointProvider | None,
        style: OcclusionStyle,
        tolerance_policy: TolerancePolicy | None = None,
        require_closed_convex_manifold: bool = False,
        source_coordinate_mode: Literal["world", "display"] = "world",
        allocate_overlay_slots: bool = True,
    ) -> None:
        self.scene = scene
        self.model = model
        self.position_provider = position_provider
        self.projection_provider = projection_provider
        # None means world-space overlay coordinates: ThreeDCamera projects the
        # overlay exactly as it projects the source.  A callable must return
        # final camera-frame Scene coordinates and is fixed in frame on attach.
        self.display_point_provider = display_point_provider
        self.style = style
        self.tolerance_policy = tolerance_policy or TolerancePolicy()
        self.require_closed_convex_manifold = bool(require_closed_convex_manifold)
        if source_coordinate_mode not in {"world", "display"}:
            raise OcclusionBindingError(
                "source_coordinate_mode must be 'world' or 'display'"
            )
        self.source_coordinate_mode = source_coordinate_mode
        self._attached = False
        self._source_snapshots: dict[str, tuple[_AttributeSnapshot, ...]] = {}
        self._resolved_styles: dict[str, ResolvedOcclusionStyle] = {}
        self.last_frame: VisibilityFrame | None = None
        self._fixed_frame_camera: ThreeDCamera | None = None

        expected = {stroke.source_edge_id for stroke in model.strokes}
        if set(stroke_bindings) != expected:
            missing = sorted(expected - set(stroke_bindings))
            extra = sorted(set(stroke_bindings) - expected)
            raise OcclusionBindingError(
                "stroke binding identity mismatch"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
        self.stroke_bindings = dict(stroke_bindings)
        for edge_id, source in self.stroke_bindings.items():
            drawable = [
                member
                for member in source.get_family()
                if np.asarray(getattr(member, "points", ()), dtype=float).size > 0
            ]
            if not isinstance(source, Line) or drawable != [source]:
                raise OcclusionBindingError(
                    f"source stroke {edge_id} must be one complete straight Manim Line"
                )
        self.capacities = (
            {
                stroke.source_edge_id: OverlayCapacity.for_stroke(stroke, model, style)
                for stroke in model.strokes
            }
            if allocate_overlay_slots
            else {}
        )
        self._slots = (
            {
                stroke.source_edge_id: _StrokeSlots(
                    self.capacities[stroke.source_edge_id]
                )
                for stroke in model.strokes
            }
            if allocate_overlay_slots
            else {}
        )
        self.overlay_root = VGroup(*(self._slots[key].root for key in sorted(self._slots)))

        # Manim detects time-aware Mobject updaters by the literal ``dt``
        # parameter name.  Keep this signature exact.
        def update_overlay(mobject: Mobject, dt: float) -> None:
            del mobject
            if self._attached:
                self.update(dt)

        self.overlay_root.add_updater(update_overlay)

    @property
    def attached(self) -> bool:
        return self._attached

    def _current_inputs(
        self,
    ) -> tuple[dict[str, np.ndarray], tuple[tuple[float, float, float], ...]]:
        raw_positions = self.position_provider()
        positions = {
            key: _point3(value, f"vertex {key}")
            for key, value in raw_positions.items()
        }
        matrix = np.asarray(self.projection_provider(self.scene), dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise OcclusionBindingError("projection provider must return a finite 3x3 matrix")
        projection = tuple(tuple(float(item) for item in row) for row in matrix)
        return positions, projection

    def _prepare_frame(
        self,
    ) -> tuple[VisibilityFrame, dict[str, OverlayPlan], dict[str, np.ndarray]]:
        positions, projection = self._current_inputs()
        frame = compute_frame_visibility(
            self.model,
            projection_matrix=projection,
            vertex_positions=positions,
            tolerance_policy=self.tolerance_policy,
            require_closed_convex_manifold=self.require_closed_convex_manifold,
        )
        plans: dict[str, OverlayPlan] = {}
        for stroke in self.model.strokes:
            if self.display_point_provider is None:
                display_start = positions[stroke.vertex_ids[0]]
                display_end = positions[stroke.vertex_ids[1]]
            else:
                display_start = self.display_point_provider(
                    positions[stroke.vertex_ids[0]]
                )
                display_end = self.display_point_provider(
                    positions[stroke.vertex_ids[1]]
                )
            plans[stroke.source_edge_id] = build_overlay_plan(
                frame.edge_map[stroke.source_edge_id],
                display_start=display_start,
                display_end=display_end,
                capacity=self.capacities[stroke.source_edge_id],
                style=self.style,
            )
        return frame, plans, positions

    def _validate_source_geometry(
        self,
        plans: Mapping[str, OverlayPlan],
        positions: Mapping[str, np.ndarray],
    ) -> None:
        for stroke in self.model.strokes:
            edge_id = stroke.source_edge_id
            source = self.stroke_bindings[edge_id]
            if self.source_coordinate_mode == "world":
                expected_start = positions[stroke.vertex_ids[0]]
                expected_end = positions[stroke.vertex_ids[1]]
            else:
                segments = (
                    *plans[edge_id].visible_segments,
                    *plans[edge_id].hidden_segments,
                )
                if not segments:
                    raise OcclusionBindingError(
                        f"source stroke {edge_id} has no visibility span"
                    )
                expected_start = min(
                    segments, key=lambda item: item.start_parameter
                ).start
                expected_end = max(
                    segments, key=lambda item: item.end_parameter
                ).end
            actual_start = _point3(source.get_start(), f"source stroke {edge_id} start")
            actual_end = _point3(source.get_end(), f"source stroke {edge_id} end")
            expected_start_array = np.asarray(expected_start, dtype=float)
            expected_end_array = np.asarray(expected_end, dtype=float)
            expected_length = float(
                np.linalg.norm(expected_end_array - expected_start_array)
            )
            endpoint_tolerance = self.tolerance_policy.resolve(
                (expected_start_array, expected_end_array),
                edge_length=expected_length,
            ).boundary
            endpoints_match = (
                np.allclose(
                    actual_start,
                    expected_start_array,
                    rtol=0.0,
                    atol=endpoint_tolerance,
                )
                and np.allclose(
                    actual_end,
                    expected_end_array,
                    rtol=0.0,
                    atol=endpoint_tolerance,
                )
            ) or (
                np.allclose(
                    actual_start,
                    expected_end_array,
                    rtol=0.0,
                    atol=endpoint_tolerance,
                )
                and np.allclose(
                    actual_end,
                    expected_start_array,
                    rtol=0.0,
                    atol=endpoint_tolerance,
                )
            )
            chord = actual_end - actual_start
            chord_length = float(np.linalg.norm(chord))
            points = np.asarray(source.points, dtype=float)
            if chord_length <= endpoint_tolerance or points.ndim != 2 or points.shape[1] != 3:
                endpoints_match = False
            else:
                offsets = points - actual_start
                distance = np.linalg.norm(np.cross(offsets, chord), axis=1) / chord_length
                if float(np.max(distance, initial=0.0)) > endpoint_tolerance:
                    endpoints_match = False
            if not endpoints_match:
                raise OcclusionBindingError(
                    f"source stroke {edge_id} must match its registered straight segment"
                )

    def _apply_frame(
        self,
        frame: VisibilityFrame,
        plans: Mapping[str, OverlayPlan],
    ) -> None:
        for edge_id in sorted(self._slots):
            self._slots[edge_id].apply(plans[edge_id], self._resolved_styles[edge_id])
        self.last_frame = frame

    def _validate_unique_source_z_indices(self) -> dict[str, float]:
        source_families = {
            edge_id: tuple(source.get_family())
            for edge_id, source in self.stroke_bindings.items()
        }
        source_family_ids = {
            edge_id: {id(item) for item in family}
            for edge_id, family in source_families.items()
        }
        result: dict[str, float] = {}
        for edge_id, family in source_families.items():
            drawable = [item for item in family if _drawable_member(item)]
            if not drawable:
                raise OcclusionBindingError(
                    f"source stroke {edge_id} has no visible drawable family member"
                )
            values = {float(item.z_index) for item in drawable}
            if len(values) != 1 or not all(np.isfinite(item) for item in values):
                raise OcclusionBindingError(
                    f"source stroke {edge_id} has ambiguous z_index within its family"
                )
            result[edge_id] = next(iter(values))

        inverse: dict[float, str] = {}
        for edge_id in sorted(result):
            value = result[edge_id]
            if value in inverse:
                raise OcclusionBindingError(
                    f"registered source strokes {inverse[value]} and {edge_id} share z_index {value}"
                )
            inverse[value] = edge_id

        scene_family: list[object] = []
        seen: set[int] = set()
        for container in self._scene_containers():
            for root in container:
                for member in root.get_family():
                    if id(member) not in seen:
                        seen.add(id(member))
                        scene_family.append(member)
        for edge_id, source in self.stroke_bindings.items():
            if id(source) not in seen:
                raise OcclusionBindingError(
                    f"source stroke {edge_id} is not owned by the current Scene"
                )
        for edge_id, value in result.items():
            owned = source_family_ids[edge_id]
            for member in scene_family:
                if id(member) in owned or not _drawable_member(member):
                    continue
                if float(member.z_index) == value:
                    raise OcclusionBindingError(
                        f"source stroke {edge_id} z_index {value} is shared by another visible drawable"
                    )
        return result

    def _register_fixed_frame_overlay(self) -> None:
        if self.display_point_provider is None:
            return
        camera = getattr(self.scene, "camera", None)
        if isinstance(camera, ThreeDCamera):
            self._fixed_frame_camera = camera
            camera.add_fixed_in_frame_mobjects(self.overlay_root)

    def _remove_fixed_frame_overlay(self) -> None:
        if self._fixed_frame_camera is not None:
            self._fixed_frame_camera.remove_fixed_in_frame_mobjects(
                self.overlay_root
            )
            self._fixed_frame_camera = None

    def attach(self) -> "ManimOcclusionBinding":
        if self._attached:
            return self
        if not _using_cairo_renderer():
            raise OcclusionBindingError(
                "automatic occlusion binding v1 supports the Cairo renderer only"
            )
        containers = self._scene_containers()
        if any(any(item is self.overlay_root for item in container) for container in containers):
            raise OcclusionBindingError("overlay root is already owned by the Scene")

        # All fallible numerical and capacity work happens before sources are
        # hidden or Scene ownership changes.
        frame, plans, positions = self._prepare_frame()
        self._validate_source_geometry(plans, positions)
        source_z_indices = self._validate_unique_source_z_indices()
        snapshots = {
            edge_id: _capture_family_style(source)
            for edge_id, source in self.stroke_bindings.items()
        }
        resolved = {
            edge_id: self.style.resolve_for(source)
            for edge_id, source in self.stroke_bindings.items()
        }
        previous_resolved = self._resolved_styles
        self._resolved_styles = resolved
        try:
            for edge_id, source in self.stroke_bindings.items():
                self._slots[edge_id].root.set_z_index(
                    source_z_indices[edge_id], family=True
                )
                self._slots[edge_id].apply_static_style(resolved[edge_id])
            self._apply_frame(frame, plans)
            for edge_id in sorted(snapshots):
                _hide_snapshots(snapshots[edge_id])
            self._source_snapshots = snapshots
            self._attached = True
            # Do not call Scene.add(): it can restructure root ownership.  The
            # overlay is an intentionally independent final Cairo root.
            self.scene.mobjects.append(self.overlay_root)
            self._register_fixed_frame_overlay()
            self._invalidate_cairo_static_image()
        except Exception:
            self._attached = False
            for item in snapshots.values():
                _restore_snapshots(item)
            self._remove_fixed_frame_overlay()
            self._remove_overlay_identity()
            self._invalidate_cairo_static_image()
            self._source_snapshots = {}
            self._resolved_styles = previous_resolved
            raise
        return self

    def update(self, dt: float = 0.0) -> "ManimOcclusionBinding":
        del dt
        if not self._attached:
            raise OcclusionBindingError("occlusion binding is not attached")
        # Prepare every edge first.  A solver/capacity failure therefore leaves
        # every overlay slot at the last-good frame.
        frame, plans, positions = self._prepare_frame()
        self._validate_source_geometry(plans, positions)
        self._apply_frame(frame, plans)
        return self

    def _scene_containers(self) -> tuple[list[object], ...]:
        result: list[list[object]] = []
        for name in ("mobjects", "foreground_mobjects", "moving_mobjects", "static_mobjects"):
            value = getattr(self.scene, name, None)
            if isinstance(value, list) and all(value is not item for item in result):
                result.append(value)
        return tuple(result)

    def _remove_overlay_identity(self) -> None:
        overlay_family_ids = {id(item) for item in self.overlay_root.get_family()}
        for container in self._scene_containers():
            container[:] = [
                item for item in container if id(item) not in overlay_family_ids
            ]

    def _invalidate_cairo_static_image(self) -> None:
        renderer = getattr(self.scene, "renderer", None)
        if renderer is not None and hasattr(renderer, "static_image"):
            renderer.static_image = None

    def restore(self) -> "ManimOcclusionBinding":
        if not self._attached and not self._source_snapshots:
            self._remove_fixed_frame_overlay()
            self._remove_overlay_identity()
            return self
        self._attached = False
        self._remove_fixed_frame_overlay()
        self._remove_overlay_identity()
        for edge_id in sorted(self._source_snapshots):
            _restore_snapshots(self._source_snapshots[edge_id])
        self._invalidate_cairo_static_image()
        self._source_snapshots = {}
        self._resolved_styles = {}
        return self

    @contextmanager
    def session(self) -> Iterator["ManimOcclusionBinding"]:
        self.attach()
        try:
            yield self
        finally:
            self.restore()

    def slot_counts(self, edge_id: str) -> tuple[int, int]:
        slots = self._slots[edge_id]
        return len(slots.visible), len(slots.hidden)

    def slot_identities(self) -> tuple[int, ...]:
        return tuple(
            identity
            for edge_id in sorted(self._slots)
            for identity in self._slots[edge_id].identities()
        )

    def active_overlay_points(self, edge_id: str) -> np.ndarray:
        return self._slots[edge_id].root.get_all_points().copy()

    def slot_snapshot(self) -> tuple[object, ...]:
        values: list[object] = []
        for edge_id in sorted(self._slots):
            for member in self._slots[edge_id].root.get_family():
                points = np.asarray(member.get_all_points(), dtype=float)
                values.append(tuple(np.round(points.reshape(-1), 12)))
                rgbas = getattr(member, "stroke_rgbas", np.empty((0, 4)))
                values.append(tuple(np.round(np.asarray(rgbas).reshape(-1), 12)))
        return tuple(values)


__all__ = [
    "ManimOcclusionBinding",
    "OcclusionBindingError",
    "OcclusionCapacityError",
    "OverlayCapacity",
    "OverlayPlan",
    "PlannedDash",
    "PlannedSegment",
    "DisplayPointProvider",
    "build_overlay_plan",
]
