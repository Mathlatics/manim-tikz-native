from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..contract import ContractError, TolerancePolicy, VisibilityModel
from ..parallel_solver import ParallelView, SolverError
from .contract import EdgeDepthCue, FaceDepthCue, FaceDepthCueFrame, FaceDepthCueStyle


class FaceDepthCueError(ValueError):
    """Raised when one frame cannot produce deterministic depth cues."""


def _unit(value: np.ndarray, label: str) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 1.0e-15:
        raise FaceDepthCueError(f"{label} must be non-zero")
    return value / length


def _face_normal(points: np.ndarray, label: str, world_epsilon: float) -> np.ndarray:
    origin = points[0]
    for index in range(1, len(points) - 1):
        normal = np.cross(points[index] - origin, points[index + 1] - origin)
        if float(np.linalg.norm(normal)) > world_epsilon * world_epsilon:
            return _unit(normal, f"face {label} normal")
    raise FaceDepthCueError(f"face {label} is degenerate")


def _validated_positions(
    model: VisibilityModel,
    vertex_positions: Mapping[str, Sequence[float]] | None,
    *,
    tolerance_policy: TolerancePolicy,
    require_closed_convex_manifold: bool,
) -> dict[str, np.ndarray]:
    raw = model.entry_positions if vertex_positions is None else vertex_positions
    try:
        model.validate(
            vertex_positions=raw,
            tolerance_policy=tolerance_policy,
            require_closed_convex_manifold=require_closed_convex_manifold,
        )
    except ContractError as exc:
        raise FaceDepthCueError(f"invalid face depth-cue frame: {exc}") from exc
    return {
        vertex_id: np.asarray(raw[vertex_id], dtype=float)
        for vertex_id in sorted(raw)
    }


def _light_direction(view: ParallelView, style: FaceDepthCueStyle) -> np.ndarray:
    matrix = np.asarray(view.projection_matrix, dtype=float)
    screen_right = _unit(matrix[0], "projection screen-right axis")
    screen_up = _unit(matrix[1], "projection screen-up axis")
    camera_facing = np.asarray(view.view_direction, dtype=float)
    right_weight, up_weight, facing_weight = style.light_direction_view
    return _unit(
        right_weight * screen_right
        + up_weight * screen_up
        + facing_weight * camera_facing,
        "derived light direction",
    )


def compute_face_depth_cue(
    model: VisibilityModel,
    *,
    projection_matrix: Sequence[Sequence[float]],
    vertex_positions: Mapping[str, Sequence[float]] | None = None,
    style: FaceDepthCueStyle | None = None,
    tolerance_policy: TolerancePolicy | None = None,
    face_draw_order: Sequence[str] | None = None,
    require_closed_convex_manifold: bool = True,
) -> FaceDepthCueFrame:
    """Compute face shading, depth opacity, and silhouette-edge emphasis.

    The result is pure data.  It does not allocate or mutate Manim objects and
    therefore can be shared by static preview, live animation, and tests.
    """

    cue_style = style or FaceDepthCueStyle()
    policy = tolerance_policy or TolerancePolicy()
    try:
        view = ParallelView.from_matrix(projection_matrix)
    except SolverError as exc:
        raise FaceDepthCueError(str(exc)) from exc
    positions = _validated_positions(
        model,
        vertex_positions,
        tolerance_policy=policy,
        require_closed_convex_manifold=require_closed_convex_manifold,
    )
    surface_vertex_ids = sorted(
        {vertex_id for face in model.faces for vertex_id in face.vertex_ids}
    )
    if not surface_vertex_ids:
        raise FaceDepthCueError("face depth cue requires at least one face")
    solid_points = np.asarray([positions[item] for item in surface_vertex_ids])
    solid_centroid = np.mean(solid_points, axis=0)
    light_direction = _light_direction(view, cue_style)
    projection = np.asarray(view.projection_matrix, dtype=float)
    hue_axis = _unit(
        0.80 * _unit(projection[0], "projection screen-right axis")
        + 0.60 * _unit(projection[1], "projection screen-up axis"),
        "derived hue axis",
    )

    raw_faces: dict[str, dict[str, object]] = {}
    for face in sorted(model.faces, key=lambda item: item.face_id):
        points = np.asarray([positions[item] for item in face.vertex_ids])
        tolerance = policy.resolve(points)
        normal = _face_normal(points, face.face_id, tolerance.world)
        centroid = np.mean(points, axis=0)
        outward_hint = centroid - solid_centroid
        if float(np.linalg.norm(outward_hint)) <= tolerance.world:
            raise FaceDepthCueError(
                f"face {face.face_id} has no stable outward direction"
            )
        if float(np.dot(normal, outward_hint)) < 0:
            normal *= -1.0
        facing = float(np.clip(np.dot(normal, view.view_direction), -1.0, 1.0))
        # Half-Lambert lighting keeps back-turned faces in the same continuous
        # tonal scale instead of collapsing every negative dot product to one
        # identical dark color.  That distinction matters in transparent
        # classroom diagrams where several faces overlap on a light canvas.
        light_score = 0.5 * (
            float(np.clip(np.dot(normal, light_direction), -1.0, 1.0)) + 1.0
        )
        raw_faces[face.face_id] = {
            "normal": normal,
            "facing": facing,
            "light": light_score,
            "depth": float(np.dot(centroid, view.view_direction)),
        }

    depths = [float(item["depth"]) for item in raw_faces.values()]
    depth_min = min(depths)
    depth_max = max(depths)
    depth_range = depth_max - depth_min
    depth_tolerance = policy.resolve(solid_points).depth
    normalized_depth = {
        face_id: (
            0.5
            if depth_range <= depth_tolerance
            else (float(item["depth"]) - depth_min) / depth_range
        )
        for face_id, item in raw_faces.items()
    }

    if face_draw_order is None:
        order = tuple(
            sorted(
                raw_faces,
                key=lambda face_id: (float(raw_faces[face_id]["depth"]), face_id),
            )
        )
    else:
        order = tuple(str(item) for item in face_draw_order)
        if len(order) != len(set(order)) or set(order) != set(raw_faces):
            raise FaceDepthCueError(
                "face_draw_order must contain every face identity exactly once"
            )
    rank = {face_id: index for index, face_id in enumerate(order)}
    opacity_weight = (
        cue_style.facing_opacity_weight + cue_style.depth_opacity_weight
    )
    face_cues: list[FaceDepthCue] = []
    for face_id in sorted(raw_faces):
        raw = raw_faces[face_id]
        facing_score = float(raw["facing"])
        facing_unit = 0.5 * (facing_score + 1.0)
        depth_unit = normalized_depth[face_id]
        opacity_position = (
            cue_style.facing_opacity_weight * facing_unit
            + cue_style.depth_opacity_weight * depth_unit
        ) / opacity_weight
        opacity_scale = cue_style.minimum_opacity_scale + opacity_position * (
            cue_style.maximum_opacity_scale - cue_style.minimum_opacity_scale
        )
        # Keep back surfaces available as a faint structural ghost, but stop
        # them from washing every front facet with the same transparent color.
        # The cubic smoothstep avoids a visible pop when a face crosses the
        # silhouette during camera motion.
        smooth_facing = facing_unit * facing_unit * (3.0 - 2.0 * facing_unit)
        surface_visibility = cue_style.back_facing_opacity_scale + (
            1.0 - cue_style.back_facing_opacity_scale
        ) * smooth_facing
        opacity_scale *= surface_visibility
        saturation_scale = cue_style.minimum_saturation_scale + opacity_position * (
            cue_style.maximum_saturation_scale
            - cue_style.minimum_saturation_scale
        )
        light_score = float(raw["light"])
        face_cues.append(
            FaceDepthCue(
                face_id=face_id,
                outward_normal=tuple(float(item) for item in raw["normal"]),
                facing_score=facing_score,
                normalized_depth=depth_unit,
                light_score=light_score,
                near_score=opacity_position,
                brightness=(
                    cue_style.ambient_brightness
                    + cue_style.diffuse_brightness * light_score
                ),
                saturation_scale=saturation_scale,
                hue_shift_turns=(
                    cue_style.maximum_hue_shift_turns
                    * float(np.clip(np.dot(raw["normal"], hue_axis), -1.0, 1.0))
                ),
                fog_strength=cue_style.far_fog_strength * (1.0 - opacity_position),
                surface_visibility=surface_visibility,
                opacity_scale=opacity_scale,
                draw_rank=rank[face_id],
            )
        )
    face_map = {item.face_id: item for item in face_cues}

    edge_cues: list[EdgeDepthCue] = []
    for stroke in sorted(model.strokes, key=lambda item: item.source_edge_id):
        incident = tuple(sorted(stroke.incident_face_ids))
        silhouette = False
        if len(incident) == 1:
            silhouette = True
        elif len(incident) == 2:
            first = face_map[incident[0]].facing_score
            second = face_map[incident[1]].facing_score
            epsilon = policy.angular
            first_edge_on = abs(first) <= epsilon
            second_edge_on = abs(second) <= epsilon
            silhouette = (
                first * second < -(epsilon * epsilon)
                or first_edge_on != second_edge_on
            )
        edge_cues.append(
            EdgeDepthCue(
                source_edge_id=stroke.source_edge_id,
                incident_face_ids=incident,
                is_silhouette=silhouette,
                visible_width_scale=(
                    cue_style.silhouette_visible_width_scale
                    if silhouette
                    else cue_style.regular_visible_width_scale
                ),
            )
        )

    return FaceDepthCueFrame(
        visibility_group_id=model.visibility_group_id,
        projection_matrix=view.projection_matrix,
        view_direction=view.view_direction,
        light_direction=tuple(float(item) for item in light_direction),
        hue_axis=tuple(float(item) for item in hue_axis),
        fog_color_rgb=cue_style.fog_color_rgb,
        face_draw_order=order,
        faces=tuple(face_cues),
        edges=tuple(edge_cues),
    )


__all__ = ["FaceDepthCueError", "compute_face_depth_cue"]
