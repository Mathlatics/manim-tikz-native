from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from manim import ManimColor, Mobject, Polygon, VGroup

from ..api import ParallelProjection
from ..binding import (
    DisplayPointProvider,
    ManimOcclusionBinding,
    OcclusionCapacityError,
    OverlayCapacity,
    OverlayPlan,
    PlannedDash,
    PlannedSegment,
    PositionProvider,
    _StrokeSlots,
    build_overlay_plan,
)
from ..contract import TolerancePolicy
from ..copy_handoff import (
    CopyIdentityHandoffFrame,
    CopyIdentityHandoffMap,
    CopyIdentityHandoffPolicy,
    compute_copy_identity_handoff,
)
from ..style import OcclusionStyle, ResolvedOcclusionStyle
from .authoring import ExtractedDihedralEntity3D
from .compositing import (
    DerivedDihedralTransparentCompositingFrame,
    compute_derived_dihedral_transparent_compositing,
)
from .compositing_manim import (
    DerivedDihedralTransparentLayer,
    PreparedDerivedDihedralTransparentFrame,
)
from .contract import DerivedDihedralModel
from .contract import RigidTransform3D
from .solver import compute_derived_dihedral_visibility
from .trace import DerivedDihedralVisibilityFrame
from .unified_compositing import (
    DerivedDihedralUnifiedCompositingFrame,
    UnifiedStrokeFragment,
    compute_derived_dihedral_unified_compositing,
)
from .unified_compositing_manim import (
    DerivedDihedralUnifiedLayer,
    PreparedDerivedDihedralUnifiedFrame,
)


def _empty_plan() -> OverlayPlan:
    return OverlayPlan((), ())


def _scaled_opacity_style(
    style: ResolvedOcclusionStyle,
    factor: float,
) -> ResolvedOcclusionStyle:
    scale = min(1.0, max(0.0, float(factor)))
    return replace(
        style,
        visible_opacity=style.visible_opacity * scale,
        hidden_opacity=style.hidden_opacity * scale,
        background_opacity=style.background_opacity * scale,
    )


def _unified_capacity(
    style: OcclusionStyle,
    slots_per_style: int,
) -> OverlayCapacity:
    return OverlayCapacity(
        visible_slots=slots_per_style,
        hidden_slots=slots_per_style,
        dash_slots_per_hidden=(
            int(ceil(style.max_projected_length / style.dash_period)) + 2
        ),
        max_projected_length=style.max_projected_length,
    )


def _unified_overlay_plan(
    fragments: Sequence[UnifiedStrokeFragment],
    *,
    display_start: Sequence[float],
    display_end: Sequence[float],
    capacity: OverlayCapacity,
    style: OcclusionStyle,
) -> OverlayPlan:
    first = np.asarray(display_start, dtype=float)
    last = np.asarray(display_end, dtype=float)
    if (
        first.shape != (3,)
        or last.shape != (3,)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(last))
    ):
        raise OcclusionCapacityError(
            "unified display stroke endpoints must be finite three-component points"
        )
    delta = last - first
    display_length = float(np.linalg.norm(delta))
    allowance = max(1.0e-12, capacity.max_projected_length * 1.0e-9)
    if display_length > capacity.max_projected_length + allowance:
        raise OcclusionCapacityError(
            "projected length "
            f"{display_length:.9g} exceeds fixed maximum "
            f"{capacity.max_projected_length:.9g}"
        )
    visible = [item for item in fragments if item.slot_kind == "visible"]
    hidden = [item for item in fragments if item.slot_kind == "hidden"]
    visible.sort(key=lambda item: item.slot_index)
    hidden.sort(key=lambda item: item.slot_index)
    if [item.slot_index for item in visible] != list(range(len(visible))):
        raise OcclusionCapacityError("unified visible slot indices are not contiguous")
    if [item.slot_index for item in hidden] != list(range(len(hidden))):
        raise OcclusionCapacityError("unified hidden slot indices are not contiguous")
    if len(visible) > capacity.visible_slots:
        raise OcclusionCapacityError(
            f"unified visible fragments {len(visible)} exceed fixed capacity "
            f"{capacity.visible_slots}"
        )
    if len(hidden) > capacity.hidden_slots:
        raise OcclusionCapacityError(
            f"unified hidden fragments {len(hidden)} exceed fixed capacity "
            f"{capacity.hidden_slots}"
        )

    def segment(item: UnifiedStrokeFragment) -> PlannedSegment:
        return PlannedSegment(
            item.start_parameter,
            item.end_parameter,
            tuple(float(value) for value in first + item.start_parameter * delta),
            tuple(float(value) for value in first + item.end_parameter * delta),
        )

    visible_segments = tuple(segment(item) for item in visible)
    hidden_segments: list[PlannedSegment] = []
    for item in hidden:
        dashes: list[PlannedDash] = []
        if display_length > 1.0e-12:
            hidden_start = item.start_parameter * display_length
            hidden_end = item.end_parameter * display_length
            period_index = max(
                0,
                int(np.floor((hidden_start - style.dash_length) / style.dash_period))
                + 1,
            )
            while period_index * style.dash_period < hidden_end - 1.0e-12:
                dash_start = period_index * style.dash_period
                dash_end = dash_start + style.dash_length
                period_index += 1
                clipped_start = max(hidden_start, dash_start)
                clipped_end = min(hidden_end, dash_end)
                if clipped_end - clipped_start <= 1.0e-12:
                    continue
                start_parameter = clipped_start / display_length
                end_parameter = clipped_end / display_length
                dashes.append(
                    PlannedDash(
                        tuple(
                            float(value)
                            for value in first + start_parameter * delta
                        ),
                        tuple(
                            float(value)
                            for value in first + end_parameter * delta
                        ),
                    )
                )
        if len(dashes) > capacity.dash_slots_per_hidden:
            raise OcclusionCapacityError(
                f"unified dash count {len(dashes)} exceeds fixed capacity "
                f"{capacity.dash_slots_per_hidden}"
            )
        base = segment(item)
        hidden_segments.append(
            PlannedSegment(
                base.start_parameter,
                base.end_parameter,
                base.start,
                base.end,
                tuple(dashes),
            )
        )
    return OverlayPlan(visible_segments, tuple(hidden_segments))


@dataclass(frozen=True)
class _FaceSnapshot:
    source: Polygon
    fill_rgbas: np.ndarray


class _DerivedDihedralFillLayer:
    """Stable whole-face proxies used before exact intersection splitting."""

    def __init__(
        self,
        model: DerivedDihedralModel,
        face_bindings: Mapping[str, Mobject],
        *,
        tolerance_policy: TolerancePolicy,
        source_coordinate_mode: Literal["world", "display"],
    ) -> None:
        expected = {face.face_id for face in model.overlay_model().faces}
        if set(face_bindings) != expected:
            missing = sorted(expected - set(face_bindings))
            extra = sorted(set(face_bindings) - expected)
            raise RuntimeError(
                "derived-dihedral face binding identity mismatch"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
        if source_coordinate_mode not in {"world", "display"}:
            raise RuntimeError(
                "derived-dihedral face source_coordinate_mode must be 'world' or 'display'"
            )
        self.model = model
        self.tolerance_policy = tolerance_policy
        self.source_coordinate_mode = source_coordinate_mode
        self.sources: dict[str, Polygon] = {}
        self.base_fill: dict[str, np.ndarray] = {}
        for face_id in sorted(expected):
            source = face_bindings[face_id]
            if not isinstance(source, Polygon) or tuple(source.get_family()) != (source,):
                raise RuntimeError(
                    f"derived-dihedral face source {face_id} must be one native Manim Polygon"
                )
            fill = np.asarray(getattr(source, "fill_rgbas", ()), dtype=float)
            if (
                fill.ndim != 2
                or fill.shape[1:] != (4,)
                or not len(fill)
                or not np.all(np.isfinite(fill))
                or any(
                    not np.allclose(item, fill[0], rtol=0.0, atol=1.0e-12)
                    for item in fill[1:]
                )
            ):
                raise RuntimeError(
                    f"derived-dihedral face source {face_id} must use one solid non-gradient fill"
                )
            if (
                float(source.get_stroke_width()) * float(source.get_stroke_opacity())
                > 1.0e-12
                or float(source.get_stroke_width(background=True))
                * float(source.get_stroke_opacity(background=True))
                > 1.0e-12
            ):
                raise RuntimeError(
                    f"derived-dihedral face source {face_id} must be fill-only"
                )
            self.sources[face_id] = source
            self.base_fill[face_id] = fill[0].copy()
        dummy = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        self.proxies = {
            face_id: Polygon(*dummy).set_stroke(opacity=0.0)
            for face_id in sorted(expected)
        }
        self.root = VGroup(*(self.proxies[key] for key in sorted(self.proxies)))
        self.snapshots: dict[str, _FaceSnapshot] = {}

    @staticmethod
    def _scene_ids(containers: Sequence[list[object]]) -> set[int]:
        return {
            id(member)
            for container in containers
            for root in container
            for member in root.get_family()
        }

    def configure(self, containers: Sequence[list[object]]) -> None:
        if self.snapshots:
            return
        scene_ids = self._scene_ids(containers)
        for face_id, source in self.sources.items():
            if id(source) not in scene_ids:
                raise RuntimeError(
                    f"derived-dihedral face source {face_id} is not owned by the current Scene"
                )
        self.snapshots = {
            face_id: _FaceSnapshot(source, source.fill_rgbas.copy())
            for face_id, source in self.sources.items()
        }

    def prepare(
        self,
        frame: DerivedDihedralVisibilityFrame,
        *,
        positions: Mapping[str, np.ndarray],
        display_point_provider: object,
        containers: Sequence[list[object]],
    ) -> tuple[str, ...]:
        self.configure(containers)
        display = display_point_provider
        if display is not None and not callable(display):
            raise RuntimeError("display_point_provider must be callable")
        overlay_model = self.model.overlay_model()
        for face in overlay_model.faces:
            source = self.sources[face.face_id]
            world = [positions[item] for item in face.vertex_ids]
            expected = world if display is None else [np.asarray(display(item), dtype=float) for item in world]
            actual = np.asarray(source.get_vertices(), dtype=float)
            source_expected = world if self.source_coordinate_mode == "world" else expected
            tolerance = self.tolerance_policy.resolve(source_expected).boundary
            if actual.shape != np.asarray(source_expected).shape:
                raise RuntimeError(
                    f"derived-dihedral face source {face.face_id} no longer matches its polygon"
                )
            matched = False
            expected_array = np.asarray(source_expected, dtype=float)
            for candidate in (expected_array, expected_array[::-1]):
                for offset in range(len(candidate)):
                    if float(
                        np.max(
                            np.linalg.norm(
                                actual - np.roll(candidate, -offset, axis=0), axis=1
                            )
                        )
                    ) <= tolerance:
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                raise RuntimeError(
                    f"derived-dihedral face source {face.face_id} no longer matches its polygon"
                )
            proxy = self.proxies[face.face_id]
            proxy.set_points_as_corners([*expected, expected[0]])

        suppressed_faces = {
            self.model.solid_face_id(face_id)
            for face_id in frame.coincident_source_face_ids
        }
        order = tuple(
            face_id
            for face_id in frame.line_visibility.face_draw_order
            if face_id not in suppressed_faces
        )
        return order

    def apply(
        self,
        order: Sequence[str],
        *,
        opacity_scales: Mapping[str, float] | None = None,
    ) -> None:
        active = set(order)
        scales = opacity_scales or {}
        denominator = max(1, len(order) - 1)
        for face_id, proxy in self.proxies.items():
            rgba = self.base_fill[face_id]
            if face_id not in active:
                proxy.set_fill(opacity=0.0)
                continue
            rank = order.index(face_id)
            proxy.set_fill(
                color=ManimColor.from_rgb(rgba[:3]),
                opacity=float(rgba[3]) * float(scales.get(face_id, 1.0)),
            )
            proxy.set_stroke(opacity=0.0)
            proxy.set_z_index(float(rank) / denominator, family=True)

    def capture_and_hide(self) -> None:
        for source in self.sources.values():
            source.set_fill(opacity=0.0)

    def hide(self) -> None:
        if self.snapshots:
            self.capture_and_hide()

    def restore(self) -> None:
        for face_id, snapshot in self.snapshots.items():
            snapshot.source.fill_rgbas = snapshot.fill_rgbas.copy()
        self.snapshots = {}


class ExtractedDihedralOcclusion3D(ManimOcclusionBinding):
    """Stable line binding for one closed solid and one derived dihedral."""

    def __init__(
        self,
        scene: object,
        model: DerivedDihedralModel,
        *,
        entity: ExtractedDihedralEntity3D,
        solid_position_provider: PositionProvider,
        stroke_bindings: Mapping[str, Mobject],
        face_fill_bindings: Mapping[str, Mobject] | None,
        projection: ParallelProjection,
        display_point_provider: DisplayPointProvider | None = None,
        style: OcclusionStyle,
        tolerance_policy: TolerancePolicy | None = None,
        source_coordinate_mode: Literal["world", "display"] = "world",
        accurate_transparency: bool = False,
        unified_compositing: bool | None = None,
        unified_fragment_slots_per_style: int = 12,
        global_transform_provider: Callable[[], RigidTransform3D] | None = None,
        identity_handoff_distance: float = 0.12,
    ) -> None:
        if not isinstance(model, DerivedDihedralModel):
            raise TypeError("model must be a DerivedDihedralModel")
        if not isinstance(accurate_transparency, bool):
            raise TypeError("accurate_transparency must be boolean")
        if accurate_transparency and face_fill_bindings is None:
            raise RuntimeError(
                "accurate_transparency requires face_fill_bindings for every "
                "solid and extracted face"
            )
        if unified_compositing is None:
            unified_compositing = accurate_transparency
        if not isinstance(unified_compositing, bool):
            raise TypeError("unified_compositing must be boolean or None")
        if unified_compositing and not accurate_transparency:
            raise RuntimeError(
                "unified_compositing requires accurate_transparency so faces "
                "and stroke fragments share the same exact painter graph"
            )
        if (
            not isinstance(unified_fragment_slots_per_style, int)
            or isinstance(unified_fragment_slots_per_style, bool)
            or not 1 <= unified_fragment_slots_per_style <= 64
        ):
            raise ValueError(
                "unified_fragment_slots_per_style must be an integer from 1 to 64"
            )
        if (
            isinstance(identity_handoff_distance, bool)
            or not isinstance(identity_handoff_distance, (int, float))
            or not np.isfinite(float(identity_handoff_distance))
            or float(identity_handoff_distance) <= 0.0
        ):
            raise ValueError(
                "identity_handoff_distance must be finite and positive"
            )
        self.accurate_transparency = accurate_transparency
        self.unified_compositing = unified_compositing
        self.unified_fragment_slots_per_style = (
            unified_fragment_slots_per_style
        )
        self.identity_handoff_distance = float(identity_handoff_distance)
        self.world_model = model
        self.identity_handoff = CopyIdentityHandoffMap.from_visibility_model(
            f"{model.visibility_group_id}:{model.extraction.entity_id}:identity-handoff",
            model.solid,
            source_entity_id="solid",
            copy_entity_id=model.extraction.entity_id,
            face_ids=model.extraction.source_face_ids,
            stroke_ids=tuple(
                item.source_stroke_id for item in model.extraction.boundary_strokes
            ),
            policy=CopyIdentityHandoffPolicy(
                activation_distance=self.identity_handoff_distance
            ),
        )
        self.entity = entity
        self.solid_position_provider = solid_position_provider
        self.projection = projection
        if global_transform_provider is not None and not callable(
            global_transform_provider
        ):
            raise TypeError("global_transform_provider must be callable")
        self.global_transform_provider = global_transform_provider
        self._manage_solid_geometry = global_transform_provider is not None
        self.face_fill_bindings = dict(face_fill_bindings or {})
        self.last_global_transform: RigidTransform3D | None = None
        self.last_identity_handoff_separation: float | None = None
        self.last_identity_handoff_weight: float | None = None
        self.last_identity_handoff_frame: CopyIdentityHandoffFrame | None = None
        self.last_extraction_frame: DerivedDihedralVisibilityFrame | None = None
        self.last_transparent_compositing: (
            DerivedDihedralTransparentCompositingFrame | None
        ) = None
        self.last_unified_compositing: (
            DerivedDihedralUnifiedCompositingFrame | None
        ) = None
        self._prepared_extraction_frame: DerivedDihedralVisibilityFrame | None = None
        self._prepared_transparent_compositing: (
            DerivedDihedralTransparentCompositingFrame | None
        ) = None
        self._prepared_transparent: (
            PreparedDerivedDihedralTransparentFrame | None
        ) = None
        self._prepared_unified_compositing: (
            DerivedDihedralUnifiedCompositingFrame | None
        ) = None
        self._prepared_unified: PreparedDerivedDihedralUnifiedFrame | None = None
        self._frame_extracted_positions: dict[str, np.ndarray] | None = None
        self._frame_solid_positions: dict[str, np.ndarray] | None = None
        self._prepared_global_transform: RigidTransform3D | None = None
        self._prepared_identity_handoff_separation = 0.0
        self._prepared_identity_handoff_weight = 0.0
        self._prepared_identity_handoff_frame: CopyIdentityHandoffFrame | None = None
        self._prepared_face_opacity_scales: dict[str, float] = {}
        self._prepared_edge_opacity_scales: dict[str, float] = {}
        overlay_model = model.overlay_model()

        def world_positions() -> dict[str, Sequence[float]]:
            solid = (
                self.solid_position_provider()
                if self._frame_solid_positions is None
                else self._frame_solid_positions
            )
            extracted = (
                self.entity.current_positions()
                if self._frame_extracted_positions is None
                else self._frame_extracted_positions
            )
            return {
                **{
                    model.solid_vertex_id(key): value
                    for key, value in solid.items()
                },
                **{
                    model.extracted_vertex_id(key): value
                    for key, value in extracted.items()
                },
            }

        super().__init__(
            scene,
            overlay_model,
            position_provider=world_positions,
            stroke_bindings=stroke_bindings,
            projection_provider=projection.current_matrix,
            display_point_provider=display_point_provider,
            style=style,
            tolerance_policy=tolerance_policy,
            require_closed_convex_manifold=False,
            source_coordinate_mode=source_coordinate_mode,
        )
        if self.unified_compositing:
            self.capacities = {
                stroke.source_edge_id: _unified_capacity(
                    style,
                    self.unified_fragment_slots_per_style,
                )
                for stroke in self.model.strokes
            }
            self._slots = {
                stroke.source_edge_id: _StrokeSlots(
                    self.capacities[stroke.source_edge_id]
                )
                for stroke in self.model.strokes
            }
            self.overlay_root = VGroup(
                *(self._slots[key].root for key in sorted(self._slots))
            )

            # Keep the same real Manim updater contract as the base binding.
            # No Mobject is allocated while an animation is running.
            def update_unified_overlay(mobject: Mobject, dt: float) -> None:
                del mobject
                if self._attached:
                    self.update(dt)

            self.overlay_root.add_updater(update_unified_overlay)
        self._transparent_layer = (
            DerivedDihedralTransparentLayer(
                model,
                self.face_fill_bindings,
                tolerance_policy=self.tolerance_policy,
                source_coordinate_mode=source_coordinate_mode,
                managed_peer_sources=(
                    self.stroke_bindings if self.unified_compositing else None
                ),
            )
            if accurate_transparency
            else None
        )
        self._face_layer = (
            None
            if not self.face_fill_bindings or accurate_transparency
            else _DerivedDihedralFillLayer(
                model,
                self.face_fill_bindings,
                tolerance_policy=self.tolerance_policy,
                source_coordinate_mode=source_coordinate_mode,
            )
        )
        self._prepared_face_order: tuple[str, ...] = ()
        self._unified_layer = (
            DerivedDihedralUnifiedLayer(
                face_sources=self.face_fill_bindings,
                stroke_sources=self.stroke_bindings,
                managed_roots=(
                    *(
                        (self._transparent_layer.root,)
                        if self._transparent_layer is not None
                        else ()
                    ),
                    *(slot.root for slot in self._slots.values()),
                ),
            )
            if self.unified_compositing
            else None
        )
        active_face_layer = self._transparent_layer or self._face_layer
        if active_face_layer is not None:
            line_root = self.overlay_root
            self.overlay_root = VGroup(active_face_layer.root, line_root)

    def _current_global_transform(self) -> RigidTransform3D:
        if self.global_transform_provider is None:
            return RigidTransform3D.identity()
        value = self.global_transform_provider()
        if not isinstance(value, RigidTransform3D):
            raise RuntimeError(
                "global_transform_provider must return RigidTransform3D"
            )
        return value

    def _identity_handoff_state(
        self,
        projection: Sequence[Sequence[float]],
    ) -> CopyIdentityHandoffFrame:
        if self._frame_solid_positions is None or self._frame_extracted_positions is None:
            raise RuntimeError("derived-dihedral handoff geometry was not prepared")
        matrix = np.asarray(projection, dtype=float)

        def final_coordinates(point: Sequence[float]) -> np.ndarray:
            if self.display_point_provider is None:
                result = matrix @ np.asarray(point, dtype=float)
            else:
                result = np.asarray(self.display_point_provider(point), dtype=float)
            if result.shape != (3,) or not np.all(np.isfinite(result)):
                raise RuntimeError(
                    "derived-dihedral handoff coordinates must be finite three-component points"
                )
            return result

        return compute_copy_identity_handoff(
            self.identity_handoff,
            source_positions={
                self.world_model.solid_vertex_id(vertex_id): point
                for vertex_id, point in self._frame_solid_positions.items()
            },
            copy_positions={
                self.world_model.extracted_vertex_id(vertex_id): point
                for vertex_id, point in self._frame_extracted_positions.items()
            },
            final_point_provider=final_coordinates,
        )

    def _update_managed_solid_geometry(
        self,
        positions: Mapping[str, np.ndarray],
    ) -> None:
        if not self._manage_solid_geometry:
            return

        def displayed(point: Sequence[float]) -> np.ndarray:
            if self.source_coordinate_mode == "world":
                result = np.asarray(point, dtype=float)
            else:
                if self.display_point_provider is None:
                    raise RuntimeError(
                        "display source mode requires display_point_provider"
                    )
                result = np.asarray(self.display_point_provider(point), dtype=float)
            if result.shape != (3,) or not np.all(np.isfinite(result)):
                raise RuntimeError(
                    "display_point_provider must return a finite three-component point"
                )
            return result

        for stroke in self.world_model.solid.strokes:
            source = self.stroke_bindings[
                self.world_model.solid_stroke_id(stroke.source_edge_id)
            ]
            source.put_start_and_end_on(
                displayed(positions[stroke.vertex_ids[0]]),
                displayed(positions[stroke.vertex_ids[1]]),
            )
        for face in self.world_model.solid.faces:
            face_id = self.world_model.solid_face_id(face.face_id)
            source = self.face_fill_bindings.get(face_id)
            if source is None:
                continue
            points = [displayed(positions[item]) for item in face.vertex_ids]
            source.set_points_as_corners([*points, points[0]])

    def _prepare_frame(self):
        global_transform = self._current_global_transform()
        local_transform = self.entity.current_transform()
        # Apply the shared center-relative motion first, then place the copied
        # dihedral.  Consequently the copy's rotation center is the source
        # solid center transformed by its own local placement.  A translated
        # copy therefore rotates about its translated center instead of
        # orbiting the source solid's fixed world-space pivot.
        transform = local_transform.compose(global_transform)
        raw_solid_positions = self.solid_position_provider()
        self._frame_solid_positions = {
            vertex_id: global_transform.apply(raw_solid_positions[vertex_id])
            for vertex_id in self.world_model.solid.vertex_map
        }
        self._frame_extracted_positions = self.entity.positions_for_transform(
            transform
        )
        positions, projection = self._current_inputs()
        solid_positions = {
            vertex_id: positions[self.world_model.solid_vertex_id(vertex_id)]
            for vertex_id in self.world_model.solid.vertex_map
        }
        self._prepared_identity_handoff_frame = self._identity_handoff_state(
            projection
        )
        self._prepared_identity_handoff_separation = (
            self._prepared_identity_handoff_frame.maximum_separation
        )
        self._prepared_identity_handoff_weight = (
            self._prepared_identity_handoff_frame.source_opacity_scale
        )
        self._prepared_face_opacity_scales = (
            self._prepared_identity_handoff_frame.source_face_opacity_scales
        )
        self._prepared_edge_opacity_scales = (
            self._prepared_identity_handoff_frame.source_stroke_opacity_scales
        )
        transparent_compositing: (
            DerivedDihedralTransparentCompositingFrame | None
        ) = None
        unified_compositing: DerivedDihedralUnifiedCompositingFrame | None = None
        if self._unified_layer is not None:
            unified_compositing = compute_derived_dihedral_unified_compositing(
                self.world_model,
                transform=transform,
                projection_matrix=projection,
                solid_vertex_positions=solid_positions,
                tolerance_policy=self.tolerance_policy,
            )
            transparent_compositing = unified_compositing.transparent
            frame = transparent_compositing.visibility
        elif self._transparent_layer is None:
            frame = compute_derived_dihedral_visibility(
                self.world_model,
                transform=transform,
                projection_matrix=projection,
                solid_vertex_positions=solid_positions,
                tolerance_policy=self.tolerance_policy,
            )
        else:
            transparent_compositing = (
                compute_derived_dihedral_transparent_compositing(
                    self.world_model,
                    transform=transform,
                    projection_matrix=projection,
                    solid_vertex_positions=solid_positions,
                    tolerance_policy=self.tolerance_policy,
                )
            )
            frame = transparent_compositing.visibility
        plans: dict[str, OverlayPlan] = {}
        suppressed = set(frame.suppressed_source_stroke_ids)
        unified_fragments: dict[str, list[UnifiedStrokeFragment]] = {}
        if unified_compositing is not None:
            for fragment in unified_compositing.stroke_fragments:
                unified_fragments.setdefault(fragment.source_edge_id, []).append(
                    fragment
                )
        for stroke in self.model.strokes:
            if stroke.source_edge_id in suppressed:
                plans[stroke.source_edge_id] = _empty_plan()
                continue
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
            if unified_compositing is None:
                plans[stroke.source_edge_id] = build_overlay_plan(
                    frame.line_visibility.edge_map[stroke.source_edge_id],
                    display_start=display_start,
                    display_end=display_end,
                    capacity=self.capacities[stroke.source_edge_id],
                    style=self.style,
                )
            else:
                plans[stroke.source_edge_id] = _unified_overlay_plan(
                    unified_fragments.get(stroke.source_edge_id, ()),
                    display_start=display_start,
                    display_end=display_end,
                    capacity=self.capacities[stroke.source_edge_id],
                    style=self.style,
                )
        entity_display = (
            self.display_point_provider
            if self.source_coordinate_mode == "display"
            else None
        )
        self._update_managed_solid_geometry(self._frame_solid_positions)
        self.entity.update_mobjects(
            entity_display,
            positions=self._frame_extracted_positions,
        )
        self._prepared_extraction_frame = frame
        self._prepared_global_transform = global_transform
        self._prepared_transparent_compositing = transparent_compositing
        self._prepared_unified_compositing = unified_compositing
        self._prepared_transparent = None
        self._prepared_unified = None
        if self._face_layer is not None:
            self._prepared_face_order = self._face_layer.prepare(
                frame,
                positions=positions,
                display_point_provider=self.display_point_provider,
                containers=self._scene_containers(),
            )
        if self._transparent_layer is not None:
            if transparent_compositing is None:
                raise RuntimeError(
                    "derived-dihedral transparent compositing frame was not prepared"
                )
            self._prepared_transparent = self._transparent_layer.prepare(
                transparent_compositing,
                world_points=positions,
                display_point_provider=self.display_point_provider,
                containers=self._scene_containers(),
                opacity_scales=self._prepared_face_opacity_scales,
            )
        if self._unified_layer is not None:
            if (
                unified_compositing is None
                or self._transparent_layer is None
                or self._prepared_transparent is None
            ):
                raise RuntimeError(
                    "derived-dihedral unified compositing frame was not prepared"
                )
            self._prepared_unified = self._unified_layer.prepare(
                unified_compositing,
                plans=plans,
                stroke_slots=self._slots,
                transparent_layer=self._transparent_layer,
                transparent_prepared=self._prepared_transparent,
                containers=self._scene_containers(),
            )
        return frame.line_visibility, plans, positions

    def _validate_source_geometry(self, plans, positions) -> None:
        if self.source_coordinate_mode != "display":
            super()._validate_source_geometry(plans, positions)
            return
        validation_plans = dict(plans)
        for stroke in self.model.strokes:
            edge_id = stroke.source_edge_id
            plan = validation_plans[edge_id]
            if plan.visible_segments or plan.hidden_segments:
                continue
            if self.display_point_provider is None:
                raise RuntimeError(
                    "display source mode requires display_point_provider"
                )
            start = tuple(
                float(item)
                for item in self.display_point_provider(
                    positions[stroke.vertex_ids[0]]
                )
            )
            end = tuple(
                float(item)
                for item in self.display_point_provider(
                    positions[stroke.vertex_ids[1]]
                )
            )
            validation_plans[edge_id] = OverlayPlan(
                (PlannedSegment(0.0, 1.0, start, end),),
                (),
            )
        super()._validate_source_geometry(validation_plans, positions)

    def _apply_frame(self, frame, plans) -> None:
        if self._face_layer is not None:
            self._face_layer.apply(
                self._prepared_face_order,
                opacity_scales=self._prepared_face_opacity_scales,
            )
        if self._transparent_layer is not None:
            if self._prepared_transparent is None:
                raise RuntimeError(
                    "derived-dihedral transparent frame was not prepared"
                )
            self._transparent_layer.apply(self._prepared_transparent)
        for edge_id in sorted(self._slots):
            style = self._resolved_styles[edge_id]
            factor = self._prepared_edge_opacity_scales.get(edge_id, 1.0)
            if factor < 1.0:
                style = _scaled_opacity_style(style, factor)
            self._slots[edge_id].apply(plans[edge_id], style)
        self.last_frame = frame
        if self._unified_layer is not None:
            if self._prepared_unified is None:
                raise RuntimeError(
                    "derived-dihedral unified Manim frame was not prepared"
                )
            self._unified_layer.apply(self._prepared_unified)
        self.last_extraction_frame = self._prepared_extraction_frame
        self.last_global_transform = self._prepared_global_transform
        self.last_identity_handoff_separation = (
            self._prepared_identity_handoff_separation
        )
        self.last_identity_handoff_weight = self._prepared_identity_handoff_weight
        self.last_identity_handoff_frame = self._prepared_identity_handoff_frame
        self.last_transparent_compositing = (
            self._prepared_transparent_compositing
        )
        self.last_unified_compositing = self._prepared_unified_compositing

    def attach(self) -> "ExtractedDihedralOcclusion3D":
        if self.attached:
            return self
        try:
            super().attach()
            if self._face_layer is not None:
                self._face_layer.capture_and_hide()
            if self._transparent_layer is not None:
                self._transparent_layer.capture_and_hide()
            return self
        except Exception:
            if self._face_layer is not None:
                self._face_layer.restore()
            if self._transparent_layer is not None:
                self._transparent_layer.restore()
            if self._unified_layer is not None:
                self._unified_layer.restore()
            super().restore()
            raise

    def update(self, dt: float = 0.0) -> "ExtractedDihedralOcclusion3D":
        try:
            super().update(dt)
        finally:
            if self._face_layer is not None:
                self._face_layer.hide()
            if self._transparent_layer is not None:
                self._transparent_layer.hide()
        return self

    def restore(self) -> "ExtractedDihedralOcclusion3D":
        try:
            super().restore()
        finally:
            if self._face_layer is not None:
                self._face_layer.restore()
            if self._transparent_layer is not None:
                self._transparent_layer.restore()
            if self._unified_layer is not None:
                self._unified_layer.restore()
            self._prepared_extraction_frame = None
            self._prepared_transparent_compositing = None
            self._prepared_transparent = None
            self._prepared_unified_compositing = None
            self._prepared_unified = None
            self._prepared_global_transform = None
            self._prepared_identity_handoff_frame = None
            self.last_extraction_frame = None
            self.last_transparent_compositing = None
            self.last_unified_compositing = None
            self.last_global_transform = None
            self.last_identity_handoff_separation = None
            self.last_identity_handoff_weight = None
            self.last_identity_handoff_frame = None
        return self

    def face_fill_identities(self) -> tuple[int, ...]:
        if self._transparent_layer is not None:
            return self._transparent_layer.identities()
        if self._face_layer is None:
            return ()
        return tuple(id(item) for item in self._face_layer.root.get_family())

    def active_transparent_fragment_ids(self) -> tuple[str, ...]:
        if self._transparent_layer is None:
            return ()
        return self._transparent_layer.active_fragment_ids

    def active_transparent_fragment_z_indices(self) -> dict[str, float]:
        if self._transparent_layer is None:
            return {}
        return self._transparent_layer.active_fragment_z_indices()

    def active_transparent_draw_batch_count(self) -> int:
        if self._transparent_layer is None:
            return 0
        return self._transparent_layer.active_draw_batch_count

    def active_unified_draw_order(self) -> tuple[str, ...]:
        if self.last_unified_compositing is None:
            return ()
        return self.last_unified_compositing.draw_order

    def active_unified_z_indices(self) -> dict[str, float]:
        if self._unified_layer is None:
            return {}
        return self._unified_layer.active_z_indices


__all__ = ["ExtractedDihedralOcclusion3D"]
