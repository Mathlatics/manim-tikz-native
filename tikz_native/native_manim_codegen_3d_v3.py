from __future__ import annotations

"""Readable multi-driver Manim source with open-face auto occlusion.

This module is intentionally additive.  It consumes the frozen v2 geometry /
camera source and the separately versioned open-face adapter, then emits one
self-contained ``numpy + manim`` definition source.  Generated projects do
not import a Provider runtime helper.
"""

from copy import deepcopy
from hashlib import sha256
from math import atan2, pi
from pprint import pformat
from typing import Any, Mapping

import numpy as np

from .compiler import PictureSpec
from .native_manim_codegen_3d_v2 import (
    NativeManimCodegen3DV2Error,
    generate_native_manim_source_3d_v2,
)
from .open_face_visibility_3d_adapter import (
    TikzNativeOpenFaceVisibility3DAdapterError,
    adapt_picture_open_face_visibility_3d,
)


NATIVE_MANIM_SOURCE_3D_V3_SCHEMA = "tikz-native-manim-source-3d/v3"
NATIVE_MANIM_AUTHORING_3D_V2_SCHEMA = "tikz-native-manim-authoring-3d/v2"
NATIVE_MANIM_OPEN_FACE_VISIBILITY_3D_SCHEMA = (
    "tikz-native-manim-open-face-visibility-3d/v1"
)


class NativeManimCodegen3DV3Error(ValueError):
    """The picture cannot safely publish an open-face authoring source."""


def _literal(value: object) -> str:
    return pformat(value, width=100, sort_dicts=True)


def _source_hash(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


def _style_payload(value: Mapping[str, object]) -> dict[str, object]:
    raw_dash = value.get("dashPatternPt")
    dash = None
    if raw_dash is not None:
        if not isinstance(raw_dash, (tuple, list)) or len(raw_dash) != 2:
            raise NativeManimCodegen3DV3Error(
                "open-face hidden style must contain one dash/gap pair"
            )
        dash = tuple(float(item) for item in raw_dash)
    color = value.get("drawColor")
    if not isinstance(color, str) or not color.strip():
        raise NativeManimCodegen3DV3Error(
            "open-face stroke style requires an explicit draw color"
        )
    opacity = float(value.get("opacity", 1.0)) * float(
        value.get("drawOpacity", 1.0)
    )
    width = float(value.get("lineWidthPt", 0.9))
    if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise NativeManimCodegen3DV3Error(
            "open-face stroke opacity must lie in [0, 1]"
        )
    if not np.isfinite(width) or width <= 0.0:
        raise NativeManimCodegen3DV3Error(
            "open-face stroke width must be finite and positive"
        )
    return {
        "draw_color": color.strip(),
        "opacity": opacity,
        "line_width_pt": width,
        "dash_pattern_pt": dash,
        "line_cap": str(value.get("lineCap") or "auto"),
        "line_join": str(value.get("lineJoin") or "auto"),
    }


def _entry_spans(analysis: object) -> dict[str, tuple[tuple[float, float, str], ...]]:
    frame = getattr(analysis, "entry_trace")
    return {
        edge.source_edge_id: tuple(
            (
                float(span.start),
                float(span.end),
                str(span.kind),
            )
            for span in edge.spans
        )
        for edge in frame.edges
    }


def _signed_hinge_orientation(
    picture: PictureSpec,
) -> dict[tuple[str, str], float]:
    """Return the authored sign for each directed hinge axis.

    The legacy Geometry Rig exposes an unsigned [0, pi] author value.  The
    generated v3 source keeps that friendly range while multiplying the
    runtime delta by the authored orientation sign.  Consequently author
    values 0 and pi really mean same-normal and opposite-normal coplanarity.
    """

    coordinates = {
        name: np.asarray(value, dtype=float)
        for name, value in picture.coordinates.items()
    }
    result: dict[tuple[str, str], float] = {}
    for relation in picture.hinge_relations:
        axis = tuple(str(value) for value in relation.axis_names)
        if len(axis) != 2:
            continue
        try:
            axis_vector = coordinates[axis[1]] - coordinates[axis[0]]
            fixed = [coordinates[name] for name in relation.fixed_face_names]
            moving = [coordinates[name] for name in relation.moving_face_names]
        except KeyError as exc:
            raise NativeManimCodegen3DV3Error(
                "hinge relation references an unknown coordinate"
            ) from exc
        axis_length = float(np.linalg.norm(axis_vector))
        if axis_length <= 1.0e-14 or len(fixed) < 3 or len(moving) < 3:
            raise NativeManimCodegen3DV3Error(
                "hinge relation cannot prove a directed non-degenerate axis"
            )
        axis_unit = axis_vector / axis_length
        fixed_normal = np.cross(fixed[1] - fixed[0], fixed[2] - fixed[0])
        moving_normal = np.cross(moving[1] - moving[0], moving[2] - moving[0])
        fixed_length = float(np.linalg.norm(fixed_normal))
        moving_length = float(np.linalg.norm(moving_normal))
        if fixed_length <= 1.0e-14 or moving_length <= 1.0e-14:
            raise NativeManimCodegen3DV3Error(
                "hinge relation contains a degenerate face"
            )
        fixed_normal /= fixed_length
        moving_normal /= moving_length
        sine = float(np.dot(axis_unit, np.cross(fixed_normal, moving_normal)))
        cosine = float(np.clip(np.dot(fixed_normal, moving_normal), -1.0, 1.0))
        signed = atan2(sine, cosine)
        result[axis] = -1.0 if signed < 0.0 else 1.0
    return result


_OPEN_FACE_SOURCE_HELPERS = r'''

# ===== Provider 展开的开放凸面全局遮挡 =====

def _open_face_point3(value, label="open-face point"):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{label} must be a finite three-component point")
    return point


def _open_face_resolved_tolerance(points, edge_length=None):
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("open-face tolerance requires finite 3D points")
    extent = np.max(values, axis=0) - np.min(values, axis=0)
    scale = max(float(np.linalg.norm(extent)), 1.0e-14)
    if edge_length is not None:
        scale = max(scale, float(edge_length))
    world = max(1.0e-14, 1.0e-9 * scale)
    return {
        "world": world,
        "boundary": 8.0 * world,
        "depth": 8.0 * world,
        "angular": 1.0e-10,
        "parameter": min(1.0, world / max(float(edge_length or 0.0), world)),
    }


def _open_face_clip_greater_equal(low, high, constant, slope, threshold, parameter_eps):
    if abs(slope) <= threshold:
        return None if constant < threshold else (low, high)
    boundary = (threshold - constant) / slope
    if slope > 0.0:
        low = max(low, boundary)
    else:
        high = min(high, boundary)
    if low >= high - parameter_eps:
        return None
    return max(0.0, low), min(1.0, high)


def _open_face_interval(start, end, face_ids, positions, inclusive_edges, view_direction):
    start = _open_face_point3(start)
    end = _open_face_point3(end)
    points = np.asarray([positions[name] for name in face_ids], dtype=float)
    segment = end - start
    segment_length = float(np.linalg.norm(segment))
    edge_tol = _open_face_resolved_tolerance((start, end), segment_length)
    face_tol = _open_face_resolved_tolerance(points)
    if segment_length <= edge_tol["world"]:
        raise ValueError("open-face semantic stroke collapsed")

    origin = points[0]
    normal = None
    for index in range(1, len(points) - 1):
        candidate = np.cross(points[index] - origin, points[index + 1] - origin)
        length = float(np.linalg.norm(candidate))
        if length > face_tol["world"] * face_tol["world"]:
            normal = candidate / length
            break
    if normal is None:
        raise ValueError("open-face occluding face is degenerate")
    distances = np.abs((points - origin) @ normal)
    if float(np.max(distances, initial=0.0)) > face_tol["boundary"]:
        raise ValueError("open-face occluding face is not planar")
    winding = []
    for index, point in enumerate(points):
        edge = points[(index + 1) % len(points)] - point
        turn = float(np.dot(np.cross(edge, points[(index + 2) % len(points)] - points[(index + 1) % len(points)]), normal))
        if abs(turn) > face_tol["world"] * face_tol["world"]:
            winding.append(turn)
    if not winding or (min(winding) < 0.0 < max(winding)):
        raise ValueError("open-face occluding face is not strictly convex")
    if max(winding) < 0.0:
        normal = -normal

    view = _open_face_point3(view_direction, "parallel view direction")
    denominator = float(np.dot(view, normal))
    if abs(denominator) <= face_tol["angular"]:
        return None
    lambda_zero = float(np.dot(points[0] - start, normal) / denominator)
    lambda_slope = float(-np.dot(segment, normal) / denominator)
    interval = _open_face_clip_greater_equal(
        0.0,
        1.0,
        lambda_zero,
        lambda_slope,
        face_tol["depth"],
        edge_tol["parameter"],
    )
    if interval is None:
        return None

    projected_zero = start + lambda_zero * view
    projected_slope = segment + lambda_slope * view
    for index, edge_start in enumerate(points):
        next_index = (index + 1) % len(points)
        edge_end = points[next_index]
        face_edge = edge_end - edge_start
        value_zero = float(np.dot(np.cross(face_edge, projected_zero - edge_start), normal))
        value_slope = float(np.dot(np.cross(face_edge, projected_slope), normal))
        edge_key = tuple(sorted((face_ids[index], face_ids[next_index])))
        threshold = 0.0 if edge_key in inclusive_edges else (
            face_tol["boundary"] * max(float(np.linalg.norm(face_edge)), face_tol["world"])
        )
        interval = _open_face_clip_greater_equal(
            interval[0],
            interval[1],
            value_zero,
            value_slope,
            threshold,
            edge_tol["parameter"],
        )
        if interval is None:
            return None
    return interval


def _open_face_union(intervals, parameter_eps):
    ordered = sorted(
        (max(0.0, float(start)), min(1.0, float(end)))
        for start, end in intervals
        if float(end) - float(start) > parameter_eps
    )
    merged = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + parameter_eps:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((item[0], item[1]) for item in merged)


def _open_face_spans(hidden, parameter_eps):
    spans = []
    cursor = 0.0
    for start, end in hidden:
        if start > cursor + parameter_eps:
            spans.append((cursor, start, "visible"))
        spans.append((start, end, "hidden"))
        cursor = max(cursor, end)
    if cursor < 1.0 - parameter_eps:
        spans.append((cursor, 1.0, "visible"))
    if not spans:
        spans.append((0.0, 1.0, "visible"))
    if spans[0][0] <= parameter_eps:
        spans[0] = (0.0, spans[0][1], spans[0][2])
    if 1.0 - spans[-1][1] <= parameter_eps:
        spans[-1] = (spans[-1][0], 1.0, spans[-1][2])
    return tuple(spans)


def compute_open_face_visibility_3d(geometry_state):
    positions = {
        name: _open_face_point3(value, f"open-face vertex {name!r}")
        for name, value in geometry_state["coordinates"]().items()
        if name in OPEN_FACE_VERTEX_IDS
    }
    if set(positions) != set(OPEN_FACE_VERTEX_IDS):
        raise ValueError("open-face runtime coordinate identity changed")
    projection = np.asarray(local_camera_matrix(geometry_state), dtype=float)
    view = parallel_view_direction(projection)
    face_map = {item["face_id"]: item for item in OPEN_FACE_FACES}
    spans_by_edge = {}
    for stroke in OPEN_FACE_STROKES:
        start = positions[stroke["vertex_ids"][0]]
        end = positions[stroke["vertex_ids"][1]]
        length = float(np.linalg.norm(end - start))
        tolerance = _open_face_resolved_tolerance((start, end), length)
        mode = stroke["visibility_mode"]
        if mode == "always_visible":
            hidden = ()
        elif mode == "always_hidden":
            hidden = ((0.0, 1.0),)
        elif mode == "auto":
            ignored = set(stroke["incident_face_ids"]) | set(stroke["excluded_face_ids"])
            raw = []
            for face_id in sorted(face_map):
                face = face_map[face_id]
                if face_id in ignored or not face["occludes_strokes"]:
                    continue
                interval = _open_face_interval(
                    start,
                    end,
                    face["vertex_ids"],
                    positions,
                    set(tuple(value) for value in OPEN_FACE_INCLUSIVE_EDGES.get(face_id, ())),
                    view,
                )
                if interval is not None:
                    raw.append(interval)
            hidden = _open_face_union(raw, tolerance["parameter"])
        else:
            raise ValueError(f"unsupported open-face visibility mode {mode!r}")
        spans_by_edge[stroke["source_edge_id"]] = _open_face_spans(
            hidden, tolerance["parameter"]
        )
    return spans_by_edge


def _open_face_assert_entry(spans_by_edge):
    if set(spans_by_edge) != set(OPEN_FACE_ENTRY_SPANS):
        raise RuntimeError("open-face entry trace changed semantic stroke identity")
    for edge_id, expected in OPEN_FACE_ENTRY_SPANS.items():
        actual = spans_by_edge[edge_id]
        if len(actual) != len(expected):
            raise RuntimeError(f"open-face entry trace changed span count for {edge_id!r}")
        for left, right in zip(actual, expected):
            if left[2] != right[2] or not np.allclose(left[:2], right[:2], atol=1.0e-7, rtol=0.0):
                raise RuntimeError(f"open-face entry trace changed spans for {edge_id!r}")
    return True


def _open_face_capture_sources(sources):
    snapshots = []
    for source in sources:
        for member in source.get_family():
            attributes = {}
            for name in (
                "stroke_rgbas",
                "background_stroke_rgbas",
                "stroke_opacity",
                "background_stroke_opacity",
            ):
                if hasattr(member, name):
                    value = getattr(member, name)
                    attributes[name] = value.copy() if isinstance(value, np.ndarray) else value
            snapshots.append((member, attributes))
    return tuple(snapshots)


def _open_face_hide_sources(snapshots):
    for member, attributes in snapshots:
        for name in ("stroke_rgbas", "background_stroke_rgbas"):
            if name not in attributes:
                continue
            value = np.asarray(attributes[name], dtype=float).copy()
            if value.ndim >= 1 and value.shape[-1] >= 4:
                value[..., 3] = 0.0
            setattr(member, name, value)
        for name in ("stroke_opacity", "background_stroke_opacity"):
            if name in attributes:
                setattr(member, name, 0.0)


def _open_face_restore_sources(snapshots):
    for member, attributes in snapshots:
        for name, value in attributes.items():
            setattr(member, name, value.copy() if isinstance(value, np.ndarray) else value)


def _open_face_sources(objects, geometry_state):
    groups = tuple(geometry_state.get("temporary_groups", ()))
    relation_index = {
        relation["relation_id"]: index for index, relation in enumerate(OCCLUSION_RELATIONS)
    }
    result = {}
    for binding in OPEN_FACE_BINDINGS:
        sources = []
        for relation_id in binding["relation_ids"]:
            if relation_id not in relation_index or relation_index[relation_id] >= len(groups):
                raise RuntimeError(f"open-face relation {relation_id!r} lost its Geometry Rig group")
            sources.append(groups[relation_index[relation_id]])
        for object_id in binding["plain_object_ids"]:
            if object_id not in objects:
                raise RuntimeError(f"open-face source object {object_id!r} is missing")
            sources.append(objects[object_id])
        if not sources:
            raise RuntimeError(f"open-face stroke {binding['source_edge_id']!r} has no source")
        result[binding["source_edge_id"]] = tuple(sources)
    return result


def _open_face_safe_length(stroke, geometry_state):
    bounds = geometry_state["coordinate_bounds"]
    start_name, end_name = stroke["vertex_ids"]
    return (
        1.05
        * float(geometry_state["logical_scale"])
        * float(geometry_state["projection_bound"])
        * (float(bounds[start_name]) + float(bounds[end_name]))
        + 1.0e-6
    )


def _open_face_allocate_slots(stroke, binding, geometry_state, ordinal):
    ignored = set(stroke["incident_face_ids"]) | set(stroke["excluded_face_ids"])
    candidate_count = sum(
        1
        for face in OPEN_FACE_FACES
        if face["occludes_strokes"] and face["face_id"] not in ignored
    )
    hidden_count = max(1 if stroke["visibility_mode"] == "always_hidden" else 0, candidate_count)
    safe_length = _open_face_safe_length(stroke, geometry_state)
    allocation_start = np.array((0.0, 0.0, 0.0), dtype=float)
    allocation_end = np.array((safe_length, 0.0, 0.0), dtype=float)
    z_index = float(binding["z_index"]) + (ordinal + 1) * 1.0e-6
    visible = tuple(
        _make_stroke_slot(
            allocation_start,
            allocation_end,
            binding["visible_style"],
            z_index,
            geometry_state["scene_unit_per_cm"],
            geometry_state["stroke_width_per_pt"],
        )
        for _index in range(candidate_count + 1)
    )
    hidden = tuple(
        _make_stroke_slot(
            allocation_start,
            allocation_end,
            binding["hidden_style"],
            z_index,
            geometry_state["scene_unit_per_cm"],
            geometry_state["stroke_width_per_pt"],
        )
        for _index in range(hidden_count)
    )
    root = VGroup(
        *(line for slot in visible for line in slot["lines"]),
        *(line for slot in hidden for line in slot["lines"]),
    )
    root.set_z_index(z_index, family=True)
    return {"visible": visible, "hidden": hidden, "root": root}


def _open_face_hidden_dash_plan(slot, full_start, full_end, start_t, end_t):
    vector = full_end - full_start
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12 or end_t - start_t <= 1.0e-12:
        return ()
    pattern = slot["dash_pattern"]
    if pattern is None:
        return ((full_start + start_t * vector, full_start + end_t * vector),)
    on_length, off_length = pattern
    period = on_length + off_length
    if period <= 0.0:
        raise RuntimeError("open-face hidden dash period collapsed")
    low = start_t * length
    high = end_t * length
    first_index = int(np.floor(low / period))
    result = []
    index = max(0, first_index)
    while index * period < high - 1.0e-10:
        dash_low = max(low, index * period)
        dash_high = min(high, index * period + on_length)
        if dash_high > dash_low + 1.0e-10:
            result.append(
                (
                    full_start + (dash_low / length) * vector,
                    full_start + (dash_high / length) * vector,
                )
            )
        index += 1
    if len(result) > len(slot["lines"]):
        raise RuntimeError("open-face hidden line exceeded its stable dash capacity")
    return tuple(result)


def _open_face_prepare_frame(state):
    spans = compute_open_face_visibility_3d(state["geometry_state"])
    coordinates = state["geometry_state"]["coordinates"]()
    project_scene = state["geometry_state"]["project_scene"]
    plans = {}
    for stroke in OPEN_FACE_STROKES:
        edge_id = stroke["source_edge_id"]
        full_start = _open_face_point3(project_scene(coordinates[stroke["vertex_ids"][0]]))
        full_end = _open_face_point3(project_scene(coordinates[stroke["vertex_ids"][1]]))
        visible = []
        hidden = []
        for start_t, end_t, visibility in spans[edge_id]:
            if visibility == "visible":
                visible.append((full_start + start_t * (full_end - full_start), full_start + end_t * (full_end - full_start)))
            else:
                hidden.append((start_t, end_t))
        slot_group = state["slots"][edge_id]
        if len(visible) > len(slot_group["visible"]) or len(hidden) > len(slot_group["hidden"]):
            raise RuntimeError("open-face span count exceeded stable slot capacity")
        hidden_dashes = tuple(
            _open_face_hidden_dash_plan(slot_group["hidden"][index], full_start, full_end, start_t, end_t)
            for index, (start_t, end_t) in enumerate(hidden)
        )
        plans[edge_id] = {"visible": tuple(visible), "hidden": hidden_dashes}
    return spans, plans


def _open_face_apply_frame(state, spans, plans):
    for edge_id in sorted(state["slots"]):
        slots = state["slots"][edge_id]
        plan = plans[edge_id]
        for index, slot in enumerate(slots["visible"]):
            if index < len(plan["visible"]):
                start, end = plan["visible"][index]
                _update_stroke_slot(slot, start, end, True)
            else:
                _hide_stroke_slot(slot)
        for index, slot in enumerate(slots["hidden"]):
            dashes = plan["hidden"][index] if index < len(plan["hidden"]) else ()
            if slot["dash_pattern"] is None and dashes:
                _update_stroke_slot(slot, dashes[0][0], dashes[0][1], True)
                continue
            for dash_index, line in enumerate(slot["lines"]):
                if dash_index < len(dashes):
                    line.put_start_and_end_on(dashes[dash_index][0], dashes[dash_index][1])
                    line.set_stroke(opacity=slot["opacity"])
                else:
                    line.set_stroke(opacity=0.0)
    state["last_spans"] = spans


def _open_face_remove_overlay(state):
    family_ids = {id(item) for item in state["overlay_root"].get_family()}
    for name in ("mobjects", "foreground_mobjects", "moving_mobjects", "static_mobjects"):
        container = getattr(state["scene"], name, None)
        if isinstance(container, list):
            container[:] = [item for item in container if id(item) not in family_ids]


def install_open_face_visibility_3d(scene, shape, objects, geometry_state):
    """Attach one fixed-capacity, global open-face visibility overlay."""
    if geometry_state.get("shape") is not shape:
        raise RuntimeError("open-face visibility received a foreign Geometry Rig state")
    if getattr(shape, "_mathppt_open_face_visibility_owner", None) is not None:
        raise RuntimeError("TikZ ShapeState already has an open-face visibility owner")
    sources = _open_face_sources(objects, geometry_state)
    snapshots = _open_face_capture_sources(
        tuple(source for edge_id in sorted(sources) for source in sources[edge_id])
    )
    stroke_map = {item["source_edge_id"]: item for item in OPEN_FACE_STROKES}
    binding_map = {item["source_edge_id"]: item for item in OPEN_FACE_BINDINGS}
    slots = {
        edge_id: _open_face_allocate_slots(
            stroke_map[edge_id], binding_map[edge_id], geometry_state, ordinal
        )
        for ordinal, edge_id in enumerate(sorted(stroke_map))
    }
    overlay_root = VGroup(*(slots[edge_id]["root"] for edge_id in sorted(slots)))
    state = {
        "scene": scene,
        "shape": shape,
        "objects": objects,
        "geometry_state": geometry_state,
        "sources": sources,
        "source_snapshots": snapshots,
        "slots": slots,
        "overlay_root": overlay_root,
        "attached": False,
        "last_spans": None,
    }

    def update_overlay(mobject, dt):
        del mobject, dt
        if not state["attached"]:
            return
        try:
            spans, plans = _open_face_prepare_frame(state)
            _open_face_apply_frame(state, spans, plans)
        finally:
            _open_face_hide_sources(state["source_snapshots"])

    overlay_root.add_updater(update_overlay)
    try:
        spans, plans = _open_face_prepare_frame(state)
        _open_face_assert_entry(spans)
        _open_face_apply_frame(state, spans, plans)
        _open_face_hide_sources(snapshots)
        state["attached"] = True
        shape._mathppt_open_face_visibility_owner = state
        scene.mobjects.append(overlay_root)
        return state
    except Exception:
        state["attached"] = False
        _open_face_restore_sources(snapshots)
        _open_face_remove_overlay(state)
        if getattr(shape, "_mathppt_open_face_visibility_owner", None) is state:
            delattr(shape, "_mathppt_open_face_visibility_owner")
        raise


def restore_open_face_visibility_3d(state):
    """Remove only this overlay and restore every temporarily hidden source."""
    if not isinstance(state, dict):
        return
    state["attached"] = False
    root = state.get("overlay_root")
    if root is not None:
        root.clear_updaters()
    _open_face_remove_overlay(state)
    _open_face_restore_sources(state.get("source_snapshots", ()))
    shape = state.get("shape")
    if shape is not None and getattr(shape, "_mathppt_open_face_visibility_owner", None) is state:
        delattr(shape, "_mathppt_open_face_visibility_owner")
'''


def generate_native_manim_source_3d_v3(
    picture: PictureSpec,
    rig: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a self-contained v3 source for proven open convex faces."""

    v3_rig = deepcopy(dict(rig))
    # Discovery mode has no committed motionSpecCore range.  Open-face v3 can
    # safely traverse the complete unsigned dihedral interval, including both
    # coplanar seam states.  A ready/author-confirmed rig keeps its exact range.
    if not isinstance(v3_rig.get("motionSpecCore"), Mapping):
        for candidate in v3_rig.get("motionCandidates", []):
            if (
                isinstance(candidate, dict)
                and candidate.get("candidateKind") == "geometry_driver"
                and candidate.get("driverType") == "hinge_fold"
                and candidate.get("status") in {"available", "recommended"}
            ):
                candidate["suggestedRange"] = {"minimum": 0.0, "maximum": pi}
    try:
        v2 = generate_native_manim_source_3d_v2(picture, v3_rig)
        analysis = adapt_picture_open_face_visibility_3d(picture)
    except (
        NativeManimCodegen3DV2Error,
        TikzNativeOpenFaceVisibility3DAdapterError,
    ) as exc:
        raise NativeManimCodegen3DV3Error(str(exc)) from exc

    source = str(v2["sourceText"])
    orientation_by_axis = _signed_hinge_orientation(picture)
    authoring_v1 = dict(v2["authoringSpec"])
    orientation_by_driver = {
        str(driver["driverId"]): orientation_by_axis.get(
            tuple(str(value) for value in driver.get("axis", ())), 1.0
        )
        for driver in authoring_v1.get("drivers", [])
        if isinstance(driver, Mapping) and driver.get("type") == "hinge_fold"
    }
    source = source.replace(
        "DRIVER_INITIAL_VALUES = {driver_id: spec['initial'] for driver_id, spec in DRIVER_SPECS.items()}\n",
        "DRIVER_INITIAL_VALUES = {driver_id: spec['initial'] for driver_id, spec in DRIVER_SPECS.items()}\n"
        + f"HINGE_ORIENTATION_SIGNS = {_literal(orientation_by_driver)}\n",
        1,
    )
    old_delta = '        delta = values[driver_id] - spec["initial"]\n'
    new_delta = (
        '        delta = HINGE_ORIENTATION_SIGNS.get(driver_id, 1.0) * '
        '(values[driver_id] - spec["initial"])\n'
    )
    if source.count(old_delta) != 1:
        raise NativeManimCodegen3DV3Error(
            "v2 source template drifted at the signed hinge delta"
        )
    source = source.replace(old_delta, new_delta, 1)
    state_patch = (
        '    state["driver_values"] = driver_values\n'
        '    state["coordinates"] = coordinates\n'
        '    state["project_scene"] = project_scene\n'
    )
    state_replacement = (
        '    state["driver_values"] = driver_values\n'
        '    state["coordinates"] = coordinates\n'
        '    state["project_scene"] = project_scene\n'
        '    state["logical_scale"] = logical_scale\n'
        '    state["scene_unit_per_cm"] = scene_unit_per_cm\n'
    )
    if source.count(state_patch) != 1:
        raise NativeManimCodegen3DV3Error(
            "v2 source template drifted at the runtime state API"
        )
    source = source.replace(state_patch, state_replacement, 1)
    projection_patch = (
        "        projection_bound = max(\n"
        "            float(np.linalg.norm(matrix[:2], ord=2))\n"
        "            for matrix in (ENTRY_PROJECTION_MATRIX, *CAMERA_PRESET_MATRICES.values())\n"
        "        )\n"
    )
    projection_replacement = projection_patch + (
        '        state["coordinate_bounds"] = dict(coordinate_bounds)\n'
        '        state["projection_bound"] = projection_bound\n'
        '        state["stroke_width_per_pt"] = stroke_width_per_pt\n'
    )
    if source.count(projection_patch) != 1:
        raise NativeManimCodegen3DV3Error(
            "v2 source template drifted at the capacity bounds"
        )
    source = source.replace(projection_patch, projection_replacement, 1)

    model = analysis.model
    inclusive_edges: dict[str, tuple[tuple[str, str], ...]] = {
        face.face_id: tuple(
            sorted(
                tuple(sorted(seam.vertex_ids))
                for seam in model.seams
                if face.face_id in seam.face_ids
            )
        )
        for face in model.faces
    }
    stroke_payload = tuple(
        {
            "source_edge_id": item.source_edge_id,
            "vertex_ids": tuple(item.vertex_ids),
            "incident_face_ids": tuple(item.incident_face_ids),
            "excluded_face_ids": tuple(item.excluded_occluder_face_ids),
            "visibility_mode": item.visibility_mode,
        }
        for item in model.strokes
    )
    face_payload = tuple(
        {
            "face_id": item.face_id,
            "vertex_ids": tuple(item.vertex_ids),
            "occludes_strokes": bool(item.occludes_strokes),
        }
        for item in model.faces
    )
    binding_payload = tuple(
        {
            "source_edge_id": item.source_edge_id,
            "relation_ids": tuple(item.relation_ids),
            "plain_object_ids": (
                tuple(item.object_ids) if item.source_kind == "named_line" else ()
            ),
            "visible_style": _style_payload(item.visible_style),
            "hidden_style": _style_payload(item.hidden_style),
            "z_index": int(item.z_index),
        }
        for item in analysis.stroke_bindings
    )
    constants = "\n".join(
        [
            "",
            f"OPEN_FACE_VERTEX_IDS = {_literal(tuple(sorted(model.vertex_map)))}",
            f"OPEN_FACE_FACES = {_literal(face_payload)}",
            f"OPEN_FACE_INCLUSIVE_EDGES = {_literal(inclusive_edges)}",
            f"OPEN_FACE_STROKES = {_literal(stroke_payload)}",
            f"OPEN_FACE_BINDINGS = {_literal(binding_payload)}",
            f"OPEN_FACE_ENTRY_SPANS = {_literal(_entry_spans(analysis))}",
            f"OPEN_FACE_ENTRY_TRACE_SHA256 = {analysis.entry_trace_sha256!r}",
            f"OPEN_FACE_MODEL_SHA256 = {analysis.model_sha256!r}",
            f"OPEN_FACE_ADAPTER_RESULT_SHA256 = {analysis.result_sha256!r}",
        ]
    )
    source = source.rstrip() + constants + _OPEN_FACE_SOURCE_HELPERS + "\n"
    compile(source, "<tikz-native-manim-source-3d-v3>", "exec")

    visibility_spec = {
        "schema": NATIVE_MANIM_OPEN_FACE_VISIBILITY_3D_SCHEMA,
        "adapterSchema": analysis.schema,
        "modelSchema": model.schema,
        "traceSchema": analysis.entry_trace.schema,
        "topology": model.topology,
        "modelSha256": analysis.model_sha256,
        "entryTraceSha256": analysis.entry_trace_sha256,
        "adapterResultSha256": analysis.result_sha256,
        "faceCount": len(model.faces),
        "strokeCount": len(model.strokes),
        "seamCount": len(model.seams),
        "requiresExplicitStaticAssetRecompile": True,
    }
    authoring_spec = {
        **authoring_v1,
        "schema": NATIVE_MANIM_AUTHORING_3D_V2_SCHEMA,
        "visibility": visibility_spec,
    }
    return {
        "schema": NATIVE_MANIM_SOURCE_3D_V3_SCHEMA,
        "sourceText": source,
        "sourceSha256": _source_hash(source),
        "authoringSpec": authoring_spec,
        "visibilitySpec": visibility_spec,
    }


__all__ = [
    "NATIVE_MANIM_AUTHORING_3D_V2_SCHEMA",
    "NATIVE_MANIM_OPEN_FACE_VISIBILITY_3D_SCHEMA",
    "NATIVE_MANIM_SOURCE_3D_V3_SCHEMA",
    "NativeManimCodegen3DV3Error",
    "generate_native_manim_source_3d_v3",
]
