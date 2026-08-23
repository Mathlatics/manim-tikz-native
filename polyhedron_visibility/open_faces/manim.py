from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal, Mapping, Sequence

import numpy as np
from manim import Mobject, Polygon, VGroup, VMobject

from ..api import ParallelProjection
from ..binding import (
    DisplayPointProvider,
    ManimOcclusionBinding,
    OcclusionBindingError,
    OverlayPlan,
    PlannedSegment,
    PositionProvider,
    _capture_family_style,
    _hide_snapshots,
    _restore_snapshots,
    _using_cairo_renderer,
    build_overlay_plan,
)
from ..contract import TolerancePolicy
from ..style import OcclusionStyle
from .contract import OpenFaceVisibilityModel
from .solver import compute_open_face_visibility
from .trace import OpenFaceVisibilityFrame
from .unified_compositing import (
    OPEN_FACE_UNIFIED_COMPOSITING_LIMITS,
    OpenFacePaintPolicy,
    OpenFaceUnifiedCompositingFrame,
    OpenFaceUnifiedCompositingLimits,
    compute_open_face_unified_compositing,
)
from .unified_manim import (
    OPEN_FACE_UNIFIED_BINDING_SCALE_LIMITS,
    OpenFaceUnifiedBindingScaleLimits,
    OpenFaceUnifiedManimRuntime,
    PreparedOpenFaceUnifiedManimFrame,
)


class OpenFaceBindingScaleError(OcclusionBindingError):
    """Raised before allocation when a realtime open-face model is too large."""


@dataclass(frozen=True)
class OpenFaceBindingScaleLimits:
    """Fixed public bounds for the v1 realtime Cairo binding."""

    max_faces: int = 64
    max_strokes: int = 128
    max_seams: int = 64
    max_candidate_pairs: int = 4096
    max_overlay_line_slots: int = 65536


OPEN_FACE_BINDING_SCALE_LIMITS = OpenFaceBindingScaleLimits()


@dataclass(frozen=True)
class _FaceFillSnapshot:
    source: Polygon
    fill_rgbas: np.ndarray
    fill_opacity: object


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


class _OpenFaceFillLayer:
    """Stable fill-only Polygon proxies ordered by one solved frame trace."""

    def __init__(
        self,
        model: OpenFaceVisibilityModel,
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
                "face fill binding identity mismatch"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
        if source_coordinate_mode not in {"world", "display"}:
            raise OcclusionBindingError(
                "face fill source_coordinate_mode must be 'world' or 'display'"
            )
        self.model = model
        self.tolerance_policy = tolerance_policy
        self.source_coordinate_mode = source_coordinate_mode
        self.sources: dict[str, Polygon] = {}
        self.proxies: dict[str, Polygon] = {}
        self._base_fill_rgbas: dict[str, np.ndarray] = {}
        self._base_fill_opacity: dict[str, object] = {}
        for face in sorted(model.faces, key=lambda item: item.face_id):
            source = face_bindings[face.face_id]
            if not isinstance(source, Polygon) or tuple(source.get_family()) != (source,):
                raise OcclusionBindingError(
                    f"face fill source {face.face_id} must be one native Manim Polygon"
                )
            raw_fill = np.asarray(getattr(source, "fill_rgbas", ()), dtype=float)
            if (
                raw_fill.ndim != 2
                or raw_fill.shape[1:] != (4,)
                or not len(raw_fill)
                or not np.all(np.isfinite(raw_fill))
                or any(
                    not np.allclose(row, raw_fill[0], rtol=0.0, atol=1.0e-12)
                    for row in raw_fill[1:]
                )
            ):
                raise OcclusionBindingError(
                    f"face fill source {face.face_id} must use one solid non-gradient fill"
                )
            points = [
                np.asarray(model.vertex_map[vertex_id].entry_position, dtype=float)
                for vertex_id in face.vertex_ids
            ]
            proxy = Polygon(*points)
            proxy.set_stroke(opacity=0.0)
            proxy.fill_rgbas = raw_fill.copy()
            if hasattr(source, "fill_opacity"):
                proxy.fill_opacity = getattr(source, "fill_opacity")
            self.sources[face.face_id] = source
            self.proxies[face.face_id] = proxy
            self._base_fill_rgbas[face.face_id] = raw_fill.copy()
            self._base_fill_opacity[face.face_id] = getattr(
                source, "fill_opacity", None
            )
        self.root = VGroup(
            *(self.proxies[face_id] for face_id in sorted(self.proxies))
        )
        self._snapshots: dict[str, _FaceFillSnapshot] = {}
        self._z_slots: tuple[float, ...] = ()
        self._source_z: dict[str, float] = {}

    def _scene_family(self, containers: Sequence[list[object]]) -> tuple[object, ...]:
        result: list[object] = []
        seen: set[int] = set()
        for container in containers:
            for root in container:
                for member in root.get_family():
                    if id(member) not in seen:
                        seen.add(id(member))
                        result.append(member)
        return tuple(result)

    def configure_z_slots(self, containers: Sequence[list[object]]) -> None:
        scene_family = self._scene_family(containers)
        scene_ids = {id(item) for item in scene_family}
        managed_ids = {
            id(member)
            for source in self.sources.values()
            for member in source.get_family()
        }
        source_z: dict[str, float] = {}
        for face_id, source in self.sources.items():
            if id(source) not in scene_ids:
                raise OcclusionBindingError(
                    f"face fill source {face_id} is not owned by the current Scene"
                )
            value = float(source.z_index)
            if not np.isfinite(value):
                raise OcclusionBindingError(
                    f"face fill source {face_id} has a non-finite z_index"
                )
            source_z[face_id] = value
        slots = tuple(sorted(source_z.values()))
        if len(set(slots)) != len(slots):
            raise OcclusionBindingError(
                "managed face fills must occupy distinct authored z_index slots"
            )
        slot_set = set(slots)
        slot_low, slot_high = min(slots), max(slots)
        for member in scene_family:
            if id(member) in managed_ids or not getattr(member, "has_points", lambda: False)():
                continue
            member_z = float(getattr(member, "z_index", float("nan")))
            if member_z in slot_set:
                raise OcclusionBindingError(
                    "a managed face fill z_index is shared by another Scene drawable"
                )
            if slot_low < member_z < slot_high:
                raise OcclusionBindingError(
                    "an unrelated Scene drawable sits inside the managed face fill z band"
                )
        self._source_z = source_z
        self._z_slots = slots

    def prepare_geometry(
        self,
        *,
        world_points: Mapping[str, np.ndarray],
        display_points: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
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
                    f"face fill source {face.face_id} no longer matches its registered polygon"
                )
            plans[face.face_id] = np.asarray(
                [display_points[vertex_id] for vertex_id in face.vertex_ids],
                dtype=float,
            )
        return plans

    def prepare(
        self,
        frame: OpenFaceVisibilityFrame,
        *,
        world_points: Mapping[str, np.ndarray],
        display_points: Mapping[str, np.ndarray],
        containers: Sequence[list[object]],
    ) -> dict[str, np.ndarray]:
        if not self._z_slots:
            self.configure_z_slots(containers)
        if set(frame.advisory_face_draw_order) != set(self.model.face_map):
            raise OcclusionBindingError(
                "open-face draw order does not cover every managed face"
            )
        plans = self.prepare_geometry(
            world_points=world_points,
            display_points=display_points,
        )
        for face_id, source in self.sources.items():
            if float(source.z_index) != self._source_z[face_id]:
                raise OcclusionBindingError(
                    f"face fill source {face_id} changed its authored z_index"
                )
        return plans

    def apply_geometry(
        self,
        plans: Mapping[str, np.ndarray],
        *,
        opacity_multiplier: float = 1.0,
    ) -> None:
        multiplier = float(opacity_multiplier)
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise OcclusionBindingError(
                "face fill opacity multiplier must be finite and non-negative"
            )
        for face_id, points in plans.items():
            proxy = self.proxies[face_id]
            proxy.set_points_as_corners([*points, points[0]])
            fill = self._base_fill_rgbas[face_id].copy()
            fill[..., 3] *= multiplier
            proxy.fill_rgbas = fill
            base_opacity = self._base_fill_opacity[face_id]
            if base_opacity is not None and hasattr(proxy, "fill_opacity"):
                proxy.fill_opacity = float(base_opacity) * multiplier

    def apply(
        self,
        frame: OpenFaceVisibilityFrame,
        plans: Mapping[str, np.ndarray],
    ) -> None:
        self.apply_geometry(plans)
        for rank, face_id in enumerate(frame.advisory_face_draw_order):
            self.proxies[face_id].set_z_index(self._z_slots[rank], family=True)

    def capture_and_hide(self) -> None:
        snapshots: dict[str, _FaceFillSnapshot] = {}
        for face_id, source in self.sources.items():
            snapshots[face_id] = _FaceFillSnapshot(
                source,
                np.asarray(source.fill_rgbas, dtype=float).copy(),
                getattr(source, "fill_opacity", None),
            )
        self._snapshots = snapshots
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


def _guard_realtime_scale(
    model: OpenFaceVisibilityModel,
    style: OcclusionStyle,
) -> None:
    limits = OPEN_FACE_BINDING_SCALE_LIMITS
    counts = {
        "faces": len(model.faces),
        "strokes": len(model.strokes),
        "seams": len(model.seams),
    }
    fixed_bounds = (
        ("faces", counts["faces"], limits.max_faces),
        ("strokes", counts["strokes"], limits.max_strokes),
        ("seams", counts["seams"], limits.max_seams),
    )
    for label, value, maximum in fixed_bounds:
        if value > maximum:
            raise OpenFaceBindingScaleError(
                f"open-face realtime binding {label}={value} exceeds fixed v1 limit {maximum}"
            )

    candidate_pairs = 0
    overlay_line_slots = 0
    base_dash_slots = int(ceil(style.max_projected_length / style.dash_period))
    for stroke in model.strokes:
        candidate_count = sum(
            1
            for face in model.faces
            if face.occludes_strokes and face.face_id not in stroke.incident_face_ids
        )
        candidate_pairs += candidate_count
        hidden_slots = candidate_count
        if stroke.visibility_mode == "always_hidden":
            hidden_slots = max(1, hidden_slots)
        visible_slots = candidate_count + 1
        dashes_per_hidden = base_dash_slots + hidden_slots + 1
        overlay_line_slots += visible_slots + hidden_slots * dashes_per_hidden
    if candidate_pairs > limits.max_candidate_pairs:
        raise OpenFaceBindingScaleError(
            "open-face realtime binding candidate_pairs="
            f"{candidate_pairs} exceeds fixed v1 limit {limits.max_candidate_pairs}"
        )
    if overlay_line_slots > limits.max_overlay_line_slots:
        raise OpenFaceBindingScaleError(
            "open-face realtime binding overlay_line_slots="
            f"{overlay_line_slots} exceeds fixed v1 limit "
            f"{limits.max_overlay_line_slots}"
        )


class OpenFaceOcclusion3D(ManimOcclusionBinding):
    """Cairo binding for finite independent convex faces and articulated hinges.

    ``legacy`` preserves the released split face/line z-order implementation.
    ``unified`` consumes the renderer-neutral painter graph and maps every face,
    solid fragment, and hidden dash group into one explicit reserved z band.
    """

    def __init__(
        self,
        scene: object,
        model: OpenFaceVisibilityModel,
        *,
        position_provider: PositionProvider,
        stroke_bindings: Mapping[str, Mobject],
        face_fill_bindings: Mapping[str, Mobject] | None = None,
        projection: ParallelProjection,
        display_point_provider: DisplayPointProvider | None = None,
        style: OcclusionStyle,
        tolerance_policy: TolerancePolicy | None = None,
        source_coordinate_mode: Literal["world", "display"] = "world",
        compositing_mode: Literal["legacy", "unified"] = "legacy",
        paint_policy: OpenFacePaintPolicy | str = OpenFacePaintPolicy.DIAGRAMMATIC,
        painter_z_band: tuple[float, float] | None = None,
        unified_compositing_limits: OpenFaceUnifiedCompositingLimits = (
            OPEN_FACE_UNIFIED_COMPOSITING_LIMITS
        ),
        unified_binding_scale_limits: OpenFaceUnifiedBindingScaleLimits = (
            OPEN_FACE_UNIFIED_BINDING_SCALE_LIMITS
        ),
        stroke_styles: Mapping[str, OcclusionStyle] | None = None,
    ) -> None:
        if not isinstance(model, OpenFaceVisibilityModel):
            raise OcclusionBindingError("model must be an OpenFaceVisibilityModel")
        if compositing_mode not in {"legacy", "unified"}:
            raise OcclusionBindingError(
                "compositing_mode must be 'legacy' or 'unified'"
            )
        selected_policy = OpenFacePaintPolicy.parse(paint_policy)
        if compositing_mode == "legacy":
            _guard_realtime_scale(model, style)
        elif face_fill_bindings is None:
            raise OcclusionBindingError(
                "unified open-face compositing requires a source Polygon for every face"
            )
        elif painter_z_band is None:
            raise OcclusionBindingError(
                "unified open-face compositing requires an explicit painter_z_band"
            )

        self.projection = projection
        self.compositing_mode = compositing_mode
        self.paint_policy = selected_policy
        self.painter_z_band = painter_z_band
        self.unified_compositing_limits = unified_compositing_limits
        super().__init__(
            scene,
            model,  # type: ignore[arg-type] -- compatible fixed-topology protocol
            position_provider=position_provider,
            stroke_bindings=stroke_bindings,
            projection_provider=projection.current_matrix,
            display_point_provider=display_point_provider,
            style=style,
            tolerance_policy=tolerance_policy,
            require_closed_convex_manifold=False,
            source_coordinate_mode=source_coordinate_mode,
            allocate_overlay_slots=compositing_mode == "legacy",
        )
        self.model: OpenFaceVisibilityModel = model
        self._face_fill_layer = (
            None
            if face_fill_bindings is None
            else _OpenFaceFillLayer(
                model,
                face_fill_bindings,
                tolerance_policy=self.tolerance_policy,
                source_coordinate_mode=source_coordinate_mode,
            )
        )
        self._prepared_face_plans: dict[str, np.ndarray] = {}
        self._unified_runtime: OpenFaceUnifiedManimRuntime | None = None
        self._unified_update_driver: VMobject | None = None
        self.last_unified_frame: OpenFaceUnifiedCompositingFrame | None = None
        if compositing_mode == "unified":
            assert self._face_fill_layer is not None
            assert face_fill_bindings is not None
            assert painter_z_band is not None
            self._unified_runtime = OpenFaceUnifiedManimRuntime(
                model,
                face_layer=self._face_fill_layer,
                stroke_sources=stroke_bindings,
                face_sources=face_fill_bindings,
                style=style,
                painter_z_band=painter_z_band,
                scale_limits=unified_binding_scale_limits,
                stroke_styles=stroke_styles,
            )
            self._unified_update_driver = VMobject()
            for updater in tuple(self.overlay_root.updaters):
                self._unified_update_driver.add_updater(updater)
            self.overlay_root.clear_updaters()
            self.overlay_root.add(
                self._unified_runtime.root,
                self._unified_update_driver,
            )
        elif self._face_fill_layer is not None:
            line_overlay_root = self.overlay_root
            self.overlay_root = VGroup(self._face_fill_layer.root, line_overlay_root)

    @property
    def display_mobject(self) -> Mobject:
        """Display proxy to target with opacity animations while attached.

        Unified mode keeps its updater on an invisible sibling driver.  A
        ``FadeOut``/``FadeIn`` cycle can therefore remove and restore the
        display proxy without interrupting live geometry computation.
        """

        if self._unified_runtime is not None:
            return self._unified_runtime.display_mobject
        return self.overlay_root

    def set_painter_z_band(self, value: tuple[float, float]) -> None:
        """Reconfigure the unified band between detach and reattach cycles."""

        if self.attached or self._unified_runtime is None:
            raise OcclusionBindingError(
                "painter z band can only change on a restored unified binding"
            )
        self._unified_runtime.set_painter_z_band(value)
        self.painter_z_band = value

    def _display_positions(
        self,
        positions: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return {
            vertex_id: (
                np.asarray(positions[vertex_id], dtype=float)
                if self.display_point_provider is None
                else np.asarray(
                    self.display_point_provider(positions[vertex_id]), dtype=float
                )
            )
            for vertex_id in self.model.vertex_map
        }

    def _prepare_frame(
        self,
    ) -> tuple[OpenFaceVisibilityFrame, dict[str, OverlayPlan], dict[str, np.ndarray]]:
        if self.compositing_mode == "unified":
            raise OcclusionBindingError(
                "legacy frame preparation is unavailable in unified mode"
            )
        positions, projection = self._current_inputs()
        frame = compute_open_face_visibility(
            self.model,
            projection_matrix=projection,
            vertex_positions=positions,
            tolerance_policy=self.tolerance_policy,
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
                frame.edge_map[stroke.source_edge_id],  # type: ignore[arg-type]
                display_start=display_start,
                display_end=display_end,
                capacity=self.capacities[stroke.source_edge_id],
                style=self.style,
            )
        if self._face_fill_layer is not None:
            display_positions = self._display_positions(positions)
            self._prepared_face_plans = self._face_fill_layer.prepare(
                frame,
                world_points=positions,
                display_points=display_positions,
                containers=self._scene_containers(),
            )
        return frame, plans, positions

    def _apply_frame(
        self,
        frame: OpenFaceVisibilityFrame,
        plans: Mapping[str, OverlayPlan],
    ) -> None:
        if self._face_fill_layer is not None:
            self._face_fill_layer.apply(frame, self._prepared_face_plans)
        super()._apply_frame(frame, plans)  # type: ignore[arg-type]

    def _prepare_unified_frame(
        self,
    ) -> tuple[PreparedOpenFaceUnifiedManimFrame, dict[str, np.ndarray]]:
        runtime = self._unified_runtime
        face_layer = self._face_fill_layer
        if runtime is None or face_layer is None:
            raise OcclusionBindingError("unified open-face runtime is not configured")
        positions, projection = self._current_inputs()
        frame = compute_open_face_unified_compositing(
            self.model,
            projection_matrix=projection,
            vertex_positions=positions,
            tolerance_policy=self.tolerance_policy,
            paint_policy=self.paint_policy,
            limits=self.unified_compositing_limits,
        )
        display_positions = self._display_positions(positions)
        face_plans = face_layer.prepare_geometry(
            world_points=positions,
            display_points=display_positions,
        )
        prepared = runtime.prepare(
            frame,
            face_plans=face_plans,
            display_positions=display_positions,
            containers=self._scene_containers(),
        )
        validation_plans = {
            stroke.source_edge_id: OverlayPlan(
                visible_segments=(
                    PlannedSegment(
                        0.0,
                        1.0,
                        tuple(
                            float(item)
                            for item in display_positions[stroke.vertex_ids[0]]
                        ),
                        tuple(
                            float(item)
                            for item in display_positions[stroke.vertex_ids[1]]
                        ),
                    ),
                ),
                hidden_segments=(),
            )
            for stroke in self.model.strokes
        }
        self._validate_source_geometry(validation_plans, positions)
        return prepared, positions

    def _hide_unified_sources(self) -> None:
        for edge_id in sorted(self._source_snapshots):
            _hide_snapshots(self._source_snapshots[edge_id])
        if self._face_fill_layer is not None:
            self._face_fill_layer.hide()

    def _attach_unified(self) -> "OpenFaceOcclusion3D":
        if self.attached:
            return self
        if not _using_cairo_renderer():
            raise OcclusionBindingError(
                "automatic occlusion binding v1 supports the Cairo renderer only"
            )
        if any(
            any(item is self.overlay_root for item in container)
            for container in self._scene_containers()
        ):
            raise OcclusionBindingError("overlay root is already owned by the Scene")
        runtime = self._unified_runtime
        face_layer = self._face_fill_layer
        assert runtime is not None and face_layer is not None

        previous_resolved = self._resolved_styles
        resolved = runtime.configure_styles()
        self._resolved_styles = resolved
        prepared, _positions = self._prepare_unified_frame()
        snapshots = {
            edge_id: _capture_family_style(source)
            for edge_id, source in self.stroke_bindings.items()
        }
        self._source_snapshots = snapshots

        def finalize_sources() -> None:
            for edge_id in sorted(snapshots):
                _hide_snapshots(snapshots[edge_id])
            face_layer.capture_and_hide()

        try:
            runtime.apply(prepared, after_apply=finalize_sources)
            self.last_unified_frame = prepared.frame
            self.last_frame = prepared.frame.visibility
            self._attached = True
            self.scene.mobjects.append(self.overlay_root)
            self._register_fixed_frame_overlay()
            self._invalidate_cairo_static_image()
        except Exception:
            self._attached = False
            for values in snapshots.values():
                _restore_snapshots(values)
            face_layer.restore()
            runtime.restore()
            self._remove_fixed_frame_overlay()
            self._remove_overlay_identity()
            self._invalidate_cairo_static_image()
            self._source_snapshots = {}
            self._resolved_styles = previous_resolved
            self.last_unified_frame = None
            raise
        return self

    def attach(self) -> "OpenFaceOcclusion3D":
        if self.compositing_mode == "unified":
            return self._attach_unified()
        if self.attached:
            return self
        try:
            super().attach()
            if self._face_fill_layer is not None:
                self._face_fill_layer.capture_and_hide()
            return self
        except Exception:
            if self._face_fill_layer is not None:
                self._face_fill_layer.restore()
            super().restore()
            raise

    def update(self, dt: float = 0.0) -> "OpenFaceOcclusion3D":
        if self.compositing_mode == "unified":
            del dt
            if not self.attached:
                raise OcclusionBindingError("occlusion binding is not attached")
            runtime = self._unified_runtime
            assert runtime is not None
            prepared, _positions = self._prepare_unified_frame()
            try:
                runtime.apply(prepared, after_apply=self._hide_unified_sources)
            except Exception:
                # The binding owns source visibility while attached.
                self._hide_unified_sources()
                raise
            self.last_unified_frame = prepared.frame
            self.last_frame = prepared.frame.visibility
            return self
        try:
            super().update(dt)
        finally:
            if self._face_fill_layer is not None:
                self._face_fill_layer.hide()
        return self

    def restore(self) -> "OpenFaceOcclusion3D":
        if self.compositing_mode == "unified":
            runtime = self._unified_runtime
            if not self.attached and not self._source_snapshots:
                self._remove_fixed_frame_overlay()
                self._remove_overlay_identity()
                if runtime is not None:
                    runtime.restore()
                return self
            self._attached = False
            self._remove_fixed_frame_overlay()
            self._remove_overlay_identity()
            for edge_id in sorted(self._source_snapshots):
                _restore_snapshots(self._source_snapshots[edge_id])
            if self._face_fill_layer is not None:
                self._face_fill_layer.restore()
            if runtime is not None:
                runtime.restore()
            self._invalidate_cairo_static_image()
            self._source_snapshots = {}
            self._resolved_styles = {}
            self.last_unified_frame = None
            return self
        try:
            super().restore()
        finally:
            if self._face_fill_layer is not None:
                self._face_fill_layer.restore()
        return self

    def detach(self) -> "OpenFaceOcclusion3D":
        return self.restore()

    def face_fill_identities(self) -> tuple[int, ...]:
        if self._face_fill_layer is None:
            return ()
        return self._face_fill_layer.identities()

    def slot_counts(self, edge_id: str) -> tuple[int, int]:
        if self._unified_runtime is not None:
            return self._unified_runtime.slot_counts(edge_id)
        return super().slot_counts(edge_id)

    def slot_identities(self) -> tuple[int, ...]:
        if self._unified_runtime is not None:
            return self._unified_runtime.slot_identities()
        return super().slot_identities()

    def slot_snapshot(self) -> tuple[object, ...]:
        if self._unified_runtime is not None:
            return self._unified_runtime.slot_snapshot()
        return super().slot_snapshot()

    @property
    def active_painter_z_indices(self) -> dict[str, float]:
        if self._unified_runtime is None:
            return {}
        return self._unified_runtime.active_z_indices


__all__ = [
    "OPEN_FACE_BINDING_SCALE_LIMITS",
    "OPEN_FACE_UNIFIED_BINDING_SCALE_LIMITS",
    "OpenFaceBindingScaleError",
    "OpenFaceBindingScaleLimits",
    "OpenFaceOcclusion3D",
    "OpenFaceUnifiedBindingScaleLimits",
]
