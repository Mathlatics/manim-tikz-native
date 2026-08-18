from __future__ import annotations

from colorsys import hls_to_rgb, rgb_to_hls
from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

import numpy as np
from manim import ManimColor, Mobject, Polygon, VGroup

from ..api import ParallelProjection
from ..binding import (
    DisplayPointProvider,
    ManimOcclusionBinding,
    OcclusionBindingError,
    OverlayPlan,
    PositionProvider,
)
from ..contract import TolerancePolicy, VisibilityModel
from ..style import OcclusionStyle, ResolvedOcclusionStyle
from ..trace import VisibilityFrame
from .contract import EdgeDepthCue, FaceDepthCueFrame, FaceDepthCueStyle
from .solver import compute_face_depth_cue


@dataclass(frozen=True)
class _FaceFillSnapshot:
    source: Polygon
    fill_rgbas: np.ndarray
    fill_opacity: object


@dataclass(frozen=True)
class _PreparedDepthCue:
    frame: FaceDepthCueFrame
    face_points: Mapping[str, np.ndarray]


def _face_points_match(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float,
) -> bool:
    if actual.shape != expected.shape or len(actual) < 3:
        return False
    for candidate in (expected, expected[::-1]):
        for offset in range(len(candidate)):
            rotated = np.roll(candidate, -offset, axis=0)
            if float(np.max(np.linalg.norm(actual - rotated, axis=1))) <= tolerance:
                return True
    return False


def _shaded_rgb(
    rgb: np.ndarray,
    *,
    brightness: float,
    saturation_scale: float,
    hue_shift_turns: float,
    fog_strength: float,
    fog_color_rgb: Sequence[float],
) -> tuple[float, float, float]:
    red, green, blue = (float(np.clip(item, 0.0, 1.0)) for item in rgb)
    hue, lightness, saturation = rgb_to_hls(red, green, blue)
    hue = (hue + hue_shift_turns) % 1.0
    saturation = float(np.clip(saturation * saturation_scale, 0.0, 1.0))
    if brightness <= 1.0:
        lightness *= brightness
    else:
        lightness += (1.0 - lightness) * min(1.0, brightness - 1.0)
    shaded = np.asarray(
        hls_to_rgb(hue, np.clip(lightness, 0.0, 1.0), saturation),
        dtype=float,
    )
    fog = np.asarray(fog_color_rgb, dtype=float)
    mixed = (1.0 - fog_strength) * shaded + fog_strength * fog
    return tuple(float(np.clip(item, 0.0, 1.0)) for item in mixed)


def depth_cued_stroke_style(
    style: ResolvedOcclusionStyle,
    cue: EdgeDepthCue,
) -> ResolvedOcclusionStyle:
    """Scale only visible strokes; hidden dash semantics stay unchanged."""

    return replace(
        style,
        visible_width=style.visible_width * cue.visible_width_scale,
    )


class FaceDepthCueLayer:
    """Stable fill-only Polygon proxies for one registered face system."""

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
            raise OcclusionBindingError(
                "face depth-cue binding identity mismatch"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
        if source_coordinate_mode not in {"world", "display"}:
            raise OcclusionBindingError(
                "face depth-cue source_coordinate_mode must be 'world' or 'display'"
            )
        self.model = model
        self.tolerance_policy = tolerance_policy
        self.source_coordinate_mode = source_coordinate_mode
        self.sources: dict[str, Polygon] = {}
        self.proxies: dict[str, Polygon] = {}
        for face in sorted(model.faces, key=lambda item: item.face_id):
            source = face_bindings[face.face_id]
            if not isinstance(source, Polygon) or tuple(source.get_family()) != (source,):
                raise OcclusionBindingError(
                    f"face depth-cue source {face.face_id} must be one native Manim Polygon"
                )
            points = [
                np.asarray(model.vertex_map[vertex_id].entry_position, dtype=float)
                for vertex_id in face.vertex_ids
            ]
            proxy = Polygon(*points)
            proxy.set_stroke(opacity=0.0)
            self.sources[face.face_id] = source
            self.proxies[face.face_id] = proxy
        self.root = VGroup(
            *(self.proxies[face_id] for face_id in sorted(self.proxies))
        )
        self._snapshots: dict[str, _FaceFillSnapshot] = {}
        self._base_fill_rgba: dict[str, np.ndarray] = {}
        self._z_slots: tuple[float, ...] = ()
        self._source_z: dict[str, float] = {}

    @staticmethod
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

    @staticmethod
    def _solid_fill(source: Polygon, face_id: str) -> np.ndarray:
        raw = np.asarray(getattr(source, "fill_rgbas", ()), dtype=float)
        if (
            raw.ndim != 2
            or raw.shape[1:] != (4,)
            or not len(raw)
            or not np.all(np.isfinite(raw))
            or any(
                not np.allclose(row, raw[0], rtol=0.0, atol=1.0e-12)
                for row in raw[1:]
            )
        ):
            raise OcclusionBindingError(
                f"face depth-cue source {face_id} must use one solid non-gradient fill"
            )
        foreground_stroke = float(source.get_stroke_width()) * float(
            source.get_stroke_opacity()
        )
        background_stroke = float(
            source.get_stroke_width(background=True)
        ) * float(source.get_stroke_opacity(background=True))
        if (
            not np.isfinite(foreground_stroke)
            or not np.isfinite(background_stroke)
            or foreground_stroke > 1.0e-12
            or background_stroke > 1.0e-12
        ):
            raise OcclusionBindingError(
                f"face depth-cue source {face_id} must be fill-only; register edges as semantic Lines"
            )
        return raw[0].copy()

    def configure(self, containers: Sequence[list[object]]) -> None:
        if self._z_slots:
            return
        scene_family = self._scene_family(containers)
        scene_ids = {id(item) for item in scene_family}
        managed_ids = {
            id(member)
            for source in self.sources.values()
            for member in source.get_family()
        }
        source_z: dict[str, float] = {}
        base_fill: dict[str, np.ndarray] = {}
        for face_id, source in self.sources.items():
            if id(source) not in scene_ids:
                raise OcclusionBindingError(
                    f"face depth-cue source {face_id} is not owned by the current Scene"
                )
            z_index = float(source.z_index)
            if not np.isfinite(z_index):
                raise OcclusionBindingError(
                    f"face depth-cue source {face_id} has a non-finite z_index"
                )
            source_z[face_id] = z_index
            base_fill[face_id] = self._solid_fill(source, face_id)
        slots = tuple(sorted(source_z.values()))
        if len(set(slots)) != len(slots):
            raise OcclusionBindingError(
                "managed face depth-cue fills must occupy distinct authored z_index slots"
            )
        slot_low, slot_high = min(slots), max(slots)
        slot_set = set(slots)
        for member in scene_family:
            if id(member) in managed_ids or not getattr(member, "has_points", lambda: False)():
                continue
            member_z = float(getattr(member, "z_index", float("nan")))
            if member_z in slot_set:
                raise OcclusionBindingError(
                    "a managed face depth-cue z_index is shared by another Scene drawable"
                )
            if slot_low < member_z < slot_high:
                raise OcclusionBindingError(
                    "an unrelated Scene drawable sits inside the managed face depth-cue z band"
                )
        self._source_z = source_z
        self._base_fill_rgba = base_fill
        self._z_slots = slots

    def prepare(
        self,
        frame: FaceDepthCueFrame,
        *,
        world_points: Mapping[str, np.ndarray],
        display_points: Mapping[str, np.ndarray],
        containers: Sequence[list[object]],
    ) -> dict[str, np.ndarray]:
        self.configure(containers)
        if set(frame.face_draw_order) != set(self.model.face_map):
            raise OcclusionBindingError(
                "face depth-cue draw order does not cover every managed face"
            )
        plans: dict[str, np.ndarray] = {}
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
                raise OcclusionBindingError(
                    f"face depth-cue source {face.face_id} no longer matches its registered polygon"
                )
            if float(self.sources[face.face_id].z_index) != self._source_z[face.face_id]:
                raise OcclusionBindingError(
                    f"face depth-cue source {face.face_id} changed its authored z_index"
                )
            current_fill = self._solid_fill(
                self.sources[face.face_id], face.face_id
            )
            if not np.allclose(
                current_fill[:3],
                self._base_fill_rgba[face.face_id][:3],
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise OcclusionBindingError(
                    f"face depth-cue source {face.face_id} changed its authored fill color"
                )
            plans[face.face_id] = np.asarray(
                [display_points[vertex_id] for vertex_id in face.vertex_ids],
                dtype=float,
            )
        return plans

    def apply(
        self,
        frame: FaceDepthCueFrame,
        plans: Mapping[str, np.ndarray],
    ) -> None:
        face_map = frame.face_map
        for rank, face_id in enumerate(frame.face_draw_order):
            cue = face_map[face_id]
            points = plans[face_id]
            proxy = self.proxies[face_id]
            proxy.set_points_as_corners([*points, points[0]])
            base = self._base_fill_rgba[face_id]
            opacity = float(np.clip(base[3] * cue.opacity_scale, 0.0, 1.0))
            proxy.set_fill(
                ManimColor.from_rgb(
                    _shaded_rgb(
                        base[:3],
                        brightness=cue.brightness,
                        saturation_scale=cue.saturation_scale,
                        hue_shift_turns=cue.hue_shift_turns,
                        fog_strength=cue.fog_strength,
                        fog_color_rgb=frame.fog_color_rgb,
                    )
                ),
                opacity=opacity,
            )
            proxy.set_stroke(opacity=0.0)
            proxy.set_z_index(self._z_slots[rank], family=True)

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

    def identities(self) -> tuple[int, ...]:
        return tuple(id(item) for item in self.root.get_family())


class DepthCuedAutoOcclusion3D(ManimOcclusionBinding):
    """Closed-convex hidden lines plus stable didactic face depth cues."""

    def __init__(
        self,
        scene: object,
        model: VisibilityModel,
        *,
        position_provider: PositionProvider,
        stroke_bindings: Mapping[str, Mobject],
        face_fill_bindings: Mapping[str, Mobject],
        projection: ParallelProjection,
        style: OcclusionStyle,
        face_style: FaceDepthCueStyle | None = None,
        display_point_provider: DisplayPointProvider | None = None,
        tolerance_policy: TolerancePolicy | None = None,
        source_coordinate_mode: Literal["world", "display"] = "world",
    ) -> None:
        self.projection = projection
        self.face_style = face_style or FaceDepthCueStyle()
        super().__init__(
            scene,
            model,
            position_provider=position_provider,
            stroke_bindings=stroke_bindings,
            projection_provider=projection.current_matrix,
            display_point_provider=display_point_provider,
            style=style,
            tolerance_policy=tolerance_policy,
            require_closed_convex_manifold=True,
            source_coordinate_mode=source_coordinate_mode,
        )
        self._face_layer = FaceDepthCueLayer(
            model,
            face_fill_bindings,
            tolerance_policy=self.tolerance_policy,
            source_coordinate_mode=source_coordinate_mode,
        )
        line_overlay_root = self.overlay_root
        self.overlay_root = VGroup(self._face_layer.root, line_overlay_root)
        self._prepared_depth_cue: _PreparedDepthCue | None = None
        self.last_face_depth_cue: FaceDepthCueFrame | None = None

    def _prepare_frame(
        self,
    ) -> tuple[VisibilityFrame, dict[str, OverlayPlan], dict[str, np.ndarray]]:
        frame, plans, positions = super()._prepare_frame()
        display_points = {
            vertex_id: (
                positions[vertex_id]
                if self.display_point_provider is None
                else np.asarray(
                    self.display_point_provider(positions[vertex_id]), dtype=float
                )
            )
            for vertex_id in self.model.vertex_map
        }
        cue = compute_face_depth_cue(
            self.model,
            projection_matrix=frame.projection_matrix,
            vertex_positions=positions,
            style=self.face_style,
            tolerance_policy=self.tolerance_policy,
            face_draw_order=frame.face_draw_order,
            require_closed_convex_manifold=True,
        )
        face_points = self._face_layer.prepare(
            cue,
            world_points=positions,
            display_points=display_points,
            containers=self._scene_containers(),
        )
        self._prepared_depth_cue = _PreparedDepthCue(cue, face_points)
        return frame, plans, positions

    def _apply_frame(
        self,
        frame: VisibilityFrame,
        plans: Mapping[str, OverlayPlan],
    ) -> None:
        prepared = self._prepared_depth_cue
        if prepared is None:
            raise OcclusionBindingError("face depth-cue frame was not prepared")
        self._face_layer.apply(prepared.frame, prepared.face_points)
        cue_map = prepared.frame.edge_map
        for edge_id in sorted(self._slots):
            self._slots[edge_id].apply(
                plans[edge_id],
                depth_cued_stroke_style(
                    self._resolved_styles[edge_id], cue_map[edge_id]
                ),
            )
        self.last_frame = frame
        self.last_face_depth_cue = prepared.frame

    def attach(self) -> "DepthCuedAutoOcclusion3D":
        if self.attached:
            return self
        try:
            super().attach()
            self._face_layer.capture_and_hide()
            return self
        except Exception:
            self._face_layer.restore()
            super().restore()
            raise

    def update(self, dt: float = 0.0) -> "DepthCuedAutoOcclusion3D":
        try:
            super().update(dt)
        finally:
            self._face_layer.hide()
        return self

    def restore(self) -> "DepthCuedAutoOcclusion3D":
        try:
            super().restore()
        finally:
            self._face_layer.restore()
            self._prepared_depth_cue = None
        return self

    def face_fill_identities(self) -> tuple[int, ...]:
        return self._face_layer.identities()


__all__ = [
    "DepthCuedAutoOcclusion3D",
    "FaceDepthCueLayer",
    "depth_cued_stroke_style",
]
