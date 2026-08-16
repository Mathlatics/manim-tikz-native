from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal, Mapping

import numpy as np
from manim import Mobject

from ..api import ParallelProjection
from ..binding import (
    DisplayPointProvider,
    ManimOcclusionBinding,
    OcclusionBindingError,
    OverlayPlan,
    PositionProvider,
    build_overlay_plan,
)
from ..contract import TolerancePolicy
from ..style import OcclusionStyle
from .contract import OpenFaceVisibilityModel
from .solver import compute_open_face_visibility
from .trace import OpenFaceVisibilityFrame


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

    The binding deliberately inherits the frozen closed-polyhedron binding's
    lifecycle, fixed slot allocation, Scene ownership checks, straight-Line
    gate, source-style restoration, and last-good-frame transaction.  Only
    frame preparation is replaced with the open-face solver.
    """

    def __init__(
        self,
        scene: object,
        model: OpenFaceVisibilityModel,
        *,
        position_provider: PositionProvider,
        stroke_bindings: Mapping[str, Mobject],
        projection: ParallelProjection,
        display_point_provider: DisplayPointProvider | None = None,
        style: OcclusionStyle,
        tolerance_policy: TolerancePolicy | None = None,
        source_coordinate_mode: Literal["world", "display"] = "world",
    ) -> None:
        if not isinstance(model, OpenFaceVisibilityModel):
            raise OcclusionBindingError(
                "model must be an OpenFaceVisibilityModel"
            )
        _guard_realtime_scale(model, style)
        self.projection = projection
        super().__init__(
            scene,
            model,  # type: ignore[arg-type] -- compatible frozen-slot protocol
            position_provider=position_provider,
            stroke_bindings=stroke_bindings,
            projection_provider=projection.current_matrix,
            display_point_provider=display_point_provider,
            style=style,
            tolerance_policy=tolerance_policy,
            require_closed_convex_manifold=False,
            source_coordinate_mode=source_coordinate_mode,
        )
        self.model: OpenFaceVisibilityModel = model

    def _prepare_frame(
        self,
    ) -> tuple[OpenFaceVisibilityFrame, dict[str, OverlayPlan], dict[str, np.ndarray]]:
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
        return frame, plans, positions


__all__ = [
    "OPEN_FACE_BINDING_SCALE_LIMITS",
    "OpenFaceBindingScaleError",
    "OpenFaceBindingScaleLimits",
    "OpenFaceOcclusion3D",
]
