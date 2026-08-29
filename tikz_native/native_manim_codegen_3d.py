"""Generate readable, self-contained Manim helpers for one 3D Geometry Rig.

The emitted source is author-facing Python.  It intentionally embeds the
reviewed semantic coordinates and bindings instead of importing the Provider's
opaque motion runtime or reopening frozen JSON files.  The only third-party
imports in the generated file are NumPy and Manim.
"""

from __future__ import annotations

import hashlib
from pprint import pformat
from typing import Any, Mapping

from .compiler import ObjectSpec, PictureSpec, StyleSpec
from .occlusion_3d import standalone_occlusion_source


NATIVE_MANIM_SOURCE_3D_SCHEMA = "tikz-native-manim-source-3d/v1"


class NativeManimCodegen3DError(ValueError):
    """The selected 3D rig cannot be represented by the source contract."""


def _literal(value: object) -> str:
    return pformat(value, width=100, sort_dicts=False)


def _source_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def _style_payload(style: StyleSpec) -> dict[str, Any]:
    return {
        "draw_color": style.draw_color or "#20242A",
        "opacity": float(style.opacity)
        * (1.0 if style.draw_opacity is None else float(style.draw_opacity)),
        "line_width_pt": float(style.line_width_pt),
        "line_cap": style.line_cap,
        "line_join": style.line_join,
        "dash_pattern_pt": (
            None
            if style.dash_pattern_pt is None
            else tuple(float(item) for item in style.dash_pattern_pt)
        ),
    }


def _object_payload(item: ObjectSpec) -> dict[str, Any]:
    geometry = item.geometry
    payload: dict[str, Any] = {
        "kind": item.kind,
        "z_index": int(item.z_index),
        "line_width_pt": float(item.style.line_width_pt),
    }
    if item.kind in {"line", "arrow", "path_label"}:
        payload.update(
            {
                "start_name": geometry.get("start_name"),
                "end_name": geometry.get("end_name"),
                "start": tuple(float(value) for value in geometry["start"]),
                "end": tuple(float(value) for value in geometry["end"]),
            }
        )
        if item.kind == "path_label":
            payload["pos"] = float(geometry.get("pos", 0.5))
            payload["sloped"] = bool(
                item.placement is not None and item.placement.sloped
            )
    elif item.kind == "polygon":
        names = geometry.get("point_names")
        values = geometry.get("points")
        if not isinstance(values, (list, tuple)):
            raise NativeManimCodegen3DError(
                f"polygon {item.id!r} has no explicit points"
            )
        payload["point_names"] = (
            tuple(str(value) for value in names)
            if isinstance(names, list) and len(names) == len(values)
            else None
        )
        payload["points"] = tuple(
            tuple(float(component) for component in point) for point in values
        )
    elif item.kind == "dot":
        payload["center_name"] = geometry.get("center_name")
        payload["center"] = tuple(float(value) for value in geometry["center"])
    elif item.kind == "label":
        payload["at_name"] = geometry.get("at_name")
        payload["at"] = tuple(float(value) for value in geometry["at"])
    else:
        raise NativeManimCodegen3DError(
            f"object {item.id!r} has unsupported local-projection kind {item.kind!r}"
        )
    return payload


def _authored_coordinates(picture: PictureSpec) -> dict[str, tuple[float, float, float]]:
    result = {
        str(name): tuple(float(component) for component in value)
        for name, value in picture.coordinates.items()
    }
    fields = (
        ("start_name", "start"),
        ("end_name", "end"),
        ("at_name", "at"),
        ("center_name", "center"),
        ("first_name", "first"),
        ("vertex_name", "vertex"),
        ("third_name", "third"),
    )
    for item in picture.objects:
        for name_field, value_field in fields:
            name = item.geometry.get(name_field)
            value = item.geometry.get(value_field)
            if isinstance(name, str) and name and value is not None and name not in result:
                point = tuple(float(component) for component in value)
                if len(point) != 3:
                    raise NativeManimCodegen3DError(
                        f"object coordinate {name!r} is not three-dimensional"
                    )
                result[name] = point  # type: ignore[assignment]
    return result


_SOURCE_HELPERS = r'''


def _point3(value, field="point"):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{field} must be a finite 3D point")
    return point


def rotate_point_about_axis(point, axis_start, axis_end, angle):
    """Rodrigues formula for one point around an arbitrary directed axis."""
    value = _point3(point)
    start = _point3(axis_start, "axis_start")
    direction = _point3(axis_end, "axis_end") - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("hinge axis must not have zero length")
    direction /= length
    relative = value - start
    return start + (
        relative * np.cos(angle)
        + np.cross(direction, relative) * np.sin(angle)
        + direction * float(np.dot(direction, relative)) * (1.0 - np.cos(angle))
    )


def point_on_segment_3d(start, end, parameter):
    first = _point3(start, "segment start")
    second = _point3(end, "segment end")
    return first + float(parameter) * (second - first)


def project_point_to_line_3d(point, line_start, line_end):
    value = _point3(point)
    start = _point3(line_start, "line_start")
    direction = _point3(line_end, "line_end") - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-18:
        raise ValueError("cannot project onto a zero-length line")
    return start + float(np.dot(value - start, direction)) / denominator * direction


def geometry_coordinates_3d(hinge_angle):
    """Evaluate the hinge and every explicit derived coordinate in logical 3D."""
    value = float(hinge_angle)
    if not HINGE_ANGLE_MINIMUM <= value <= HINGE_ANGLE_MAXIMUM:
        raise ValueError(
            f"{HINGE_PARAMETER_ID}={value} is outside "
            f"[{HINGE_ANGLE_MINIMUM}, {HINGE_ANGLE_MAXIMUM}]"
        )
    coordinates = {
        name: np.array(point, dtype=float)
        for name, point in AUTHORED_COORDINATES.items()
    }
    axis_start = coordinates[HINGE_AXIS_COORDINATE_IDS[0]]
    axis_end = coordinates[HINGE_AXIS_COORDINATE_IDS[1]]
    delta = value - HINGE_ANGLE_INITIAL
    for name in HINGE_MOVING_COORDINATE_IDS:
        coordinates[name] = rotate_point_about_axis(
            AUTHORED_COORDINATES[name], axis_start, axis_end, delta
        )
    unresolved = {item["name"]: dict(item) for item in DERIVED_COORDINATES}
    while unresolved:
        progressed = False
        for name, relation in tuple(unresolved.items()):
            if relation["type"] == "point_on_segment":
                dependencies = (relation["start"], relation["end"])
                if any(item not in coordinates for item in dependencies):
                    continue
                coordinates[name] = point_on_segment_3d(
                    coordinates[relation["start"]],
                    coordinates[relation["end"]],
                    relation["parameter"],
                )
            elif relation["type"] == "project_point_to_line":
                dependencies = (
                    relation["point"],
                    relation["line_start"],
                    relation["line_end"],
                )
                if any(item not in coordinates for item in dependencies):
                    continue
                coordinates[name] = project_point_to_line_3d(
                    coordinates[relation["point"]],
                    coordinates[relation["line_start"]],
                    coordinates[relation["line_end"]],
                )
            else:
                raise ValueError(f"unsupported derived coordinate type: {relation['type']!r}")
            unresolved.pop(name)
            progressed = True
        if not progressed:
            raise ValueError(
                "derived coordinate dependency cycle or missing input: "
                + ", ".join(sorted(unresolved))
            )
    return coordinates


def _project_point(matrix, point):
    return np.asarray(matrix, dtype=float) @ _point3(point)


def _is_rotation_matrix(matrix, tolerance=1e-7):
    value = np.asarray(matrix, dtype=float)
    return bool(
        value.shape == (3, 3)
        and np.allclose(value @ value.T, np.identity(3), atol=tolerance)
        and np.isclose(np.linalg.det(value), 1.0, atol=tolerance)
    )


def _axis_angle_rotation(axis, angle):
    direction = _point3(axis, "rotation axis")
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("rotation axis must be nonzero")
    x, y, z = direction / length
    cross_matrix = np.array(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float
    )
    return (
        np.identity(3)
        + np.sin(angle) * cross_matrix
        + (1.0 - np.cos(angle)) * (cross_matrix @ cross_matrix)
    )


def _rotation_slerp(source, target, alpha):
    relative = target @ source.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return source.copy()
    sine = float(np.sin(angle))
    if abs(sine) < 1e-7:
        values, vectors = np.linalg.eig(relative)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    else:
        axis = np.array(
            (
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ),
            dtype=float,
        ) / (2.0 * sine)
    return _axis_angle_rotation(axis, float(alpha) * angle) @ source


def _frame_from_view_direction(view_direction, horizontal_hint):
    normal = _point3(view_direction, "view direction")
    normal /= np.linalg.norm(normal)
    horizontal = np.cross(np.array((0.0, 0.0, 1.0)), normal)
    if np.linalg.norm(horizontal) < 1e-7:
        hint = _point3(horizontal_hint, "horizontal hint")
        horizontal = hint - np.dot(hint, normal) * normal
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    vertical /= np.linalg.norm(vertical)
    return np.vstack((horizontal, vertical, normal))


def _orbit_control_matrix(source, target, arc_height):
    bend = -np.cross(source[2], target[2])
    if np.linalg.norm(bend) < 1e-7:
        bend = source[1]
    bend /= np.linalg.norm(bend)
    direction = source[2] + target[2] + float(arc_height) * bend
    direction /= np.linalg.norm(direction)
    hint = source[0] + target[0]
    if np.linalg.norm(hint) < 1e-7:
        hint = source[0]
    return _frame_from_view_direction(direction, hint)


def _spherical_bezier_matrix(source, control, target, alpha):
    first = _rotation_slerp(source, control, alpha)
    second = _rotation_slerp(control, target, alpha)
    return _rotation_slerp(first, second, alpha)


def _polar_projection_parts(matrix):
    """Split an invertible projection into rotation and symmetric stretch."""
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("local camera projection must be a finite invertible 3x3 matrix")
    row_scales = np.max(np.abs(value), axis=1)
    if np.any(row_scales == 0.0) or not np.all(np.isfinite(row_scales)):
        raise ValueError("local camera projection must be a finite invertible 3x3 matrix")
    normalized = value / row_scales[:, np.newaxis]
    row_norms = np.linalg.norm(normalized, axis=1)
    if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
        raise ValueError("local camera projection must be a finite invertible 3x3 matrix")
    normalized /= row_norms[:, np.newaxis]
    determinant = float(np.linalg.det(normalized))
    if not np.isfinite(determinant) or abs(determinant) <= 1.0e-12:
        raise ValueError("local camera projection must be a finite invertible 3x3 matrix")
    left, singular_values, right = np.linalg.svd(value)
    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        singular_values[-1] *= -1.0
        rotation = left @ right
    stretch = right.T @ np.diag(singular_values) @ right
    return rotation, stretch


def local_camera_matrix(state):
    camera = state["local_camera"]
    alpha = float(np.clip(state["camera_progress"].get_value(), 0.0, 1.0))
    if camera["transition"] == "orbit":
        rotation = _spherical_bezier_matrix(
            camera["source_rotation"],
            camera["control_rotation"],
            camera["target_rotation"],
            alpha,
        )
        stretch = (1.0 - alpha) * camera["source_stretch"] + alpha * camera["target_stretch"]
        return rotation @ stretch
    return (1.0 - alpha) * camera["source"] + alpha * camera["target"]


def prepare_local_camera(state, mode, transition="orbit", arc_height=0.2):
    """Choose the next object-local parallel camera; animate camera_progress to 1."""
    if transition not in {"linear", "orbit"}:
        raise ValueError("local camera transition must be 'linear' or 'orbit'")
    if mode == "tikz":
        target = ENTRY_PROJECTION_MATRIX.copy()
    elif mode in CAMERA_PRESET_MATRICES:
        target = np.array(CAMERA_PRESET_MATRICES[mode], dtype=float)
    else:
        raise ValueError(
            f"unknown local camera mode {mode!r}; available: "
            + ", ".join(("tikz", *sorted(CAMERA_PRESET_MATRICES)))
        )
    source = local_camera_matrix(state).copy()
    camera_state = {
        "source": source,
        "target": target.copy(),
        "transition": transition,
    }
    if transition == "orbit":
        source_rotation, source_stretch = _polar_projection_parts(source)
        target_rotation, target_stretch = _polar_projection_parts(target)
        camera_state.update(
            {
                "source_rotation": source_rotation,
                "target_rotation": target_rotation,
                "control_rotation": _orbit_control_matrix(
                    source_rotation, target_rotation, float(arc_height)
                ),
                "source_stretch": source_stretch,
                "target_stretch": target_stretch,
            }
        )
    state["local_camera"] = camera_state
    state["camera_progress"].set_value(0.0)


def _similarity_mapper(logical_start, logical_end, scene_start, scene_end):
    source = np.asarray(logical_end, dtype=float) - np.asarray(logical_start, dtype=float)
    first = _point3(scene_start, "scene axis start")
    target = _point3(scene_end, "scene axis end")[:2] - first[:2]
    denominator = float(np.dot(source, source))
    if denominator <= 1e-18 or float(np.linalg.norm(target)) <= 1e-12:
        raise ValueError("3D hinge anchors must not collapse in the ShapeState")
    real = float(np.dot(target, source)) / denominator
    imaginary = float(target[1] * source[0] - target[0] * source[1]) / denominator
    matrix = np.array(((real, -imaginary), (imaginary, real)), dtype=float)
    scale = float(np.hypot(real, imaginary))
    def map_screen(value):
        point = np.asarray(value, dtype=float)
        if point.shape != (2,):
            raise ValueError("local projection must produce a 2D point")
        mapped = first[:2] + matrix @ (point - logical_start)
        return np.array((mapped[0], mapped[1], first[2]), dtype=float)
    return map_screen, scale


def _coordinate_anchor(name, objects):
    authored = np.array(AUTHORED_COORDINATES[name], dtype=float)
    for object_id, spec in OBJECT_SPECS.items():
        item = objects[object_id]
        if spec["kind"] == "dot" and spec.get("center_name") == name:
            return _point3(item.get_center(), f"dot anchor {name!r}").copy()
    for object_id, spec in OBJECT_SPECS.items():
        item = objects[object_id]
        if spec["kind"] not in {"line", "arrow"}:
            continue
        for name_field, value_field, getter_name in (
            ("start_name", "start", "get_start"),
            ("end_name", "end", "get_end"),
        ):
            getter = getattr(item, getter_name, None)
            if (
                spec.get(name_field) == name
                and np.allclose(spec[value_field], authored, atol=1e-9, rtol=0.0)
                and callable(getter)
            ):
                try:
                    return _point3(getter(), f"line anchor {name!r}").copy()
                except Exception:
                    continue
    raise ValueError(f"ShapeState has no stable semantic anchor for {name!r}")


def _object_logical_point(spec, coordinates, name_field, value_field):
    name = spec.get(name_field)
    if isinstance(name, str) and name:
        try:
            return coordinates[name]
        except KeyError as exc:
            raise ValueError(f"semantic object uses unknown coordinate {name!r}") from exc
    return _point3(spec[value_field], value_field)


# __TIKZ_NATIVE_OCCLUSION_3D_KERNEL__


def _visible_stroke_width(mobject):
    widths = []
    for member in mobject.get_family():
        has_points = getattr(member, "has_points", None)
        if not callable(has_points) or not has_points():
            continue
        opacity_getter = getattr(member, "get_stroke_opacity", None)
        if callable(opacity_getter):
            opacity = np.asarray(opacity_getter(), dtype=float).reshape(-1)
            if not np.any(np.isfinite(opacity) & (opacity > 0)):
                continue
        width_getter = getattr(member, "get_stroke_width", None)
        if callable(width_getter):
            values = np.asarray(width_getter(), dtype=float).reshape(-1)
            widths.extend(float(value) for value in values if np.isfinite(value) and value > 0)
    return None if not widths else float(np.median(widths))


def _make_stroke_slot(start, end, style, z_index, scene_unit_per_cm, stroke_width_per_pt):
    pattern = style["dash_pattern_pt"]
    dash_pattern = None
    capacity = 1
    if pattern is not None:
        on_length = max(float(pattern[0]) * scene_unit_per_cm / TEX_POINTS_PER_CM, 1e-6)
        off_length = max(float(pattern[1]) * scene_unit_per_cm / TEX_POINTS_PER_CM, 0.0)
        dash_pattern = (on_length, off_length)
        capacity = max(
            1,
            int(np.ceil(float(np.linalg.norm(end - start)) / max(on_length + off_length, 1e-6))) + 1,
        )
    width_scale = (
        scene_unit_per_cm / TEX_POINTS_PER_CM
        if stroke_width_per_pt is None
        else stroke_width_per_pt
    )
    cap_style = {
        "round": CapStyleType.ROUND,
        "butt": CapStyleType.BUTT,
        "square": CapStyleType.SQUARE,
    }.get(style["line_cap"], CapStyleType.AUTO)
    joint_type = {
        "round": LineJointType.ROUND,
        "bevel": LineJointType.BEVEL,
        "miter": LineJointType.MITER,
    }.get(style["line_join"], LineJointType.AUTO)
    lines = []
    for _index in range(capacity):
        line = Line(start, end, cap_style=cap_style, joint_type=joint_type)
        line.set_stroke(
            color=style["draw_color"],
            width=float(style["line_width_pt"]) * width_scale,
            opacity=0.0,
        )
        line.set_z_index(z_index)
        lines.append(line)
    return {
        "lines": tuple(lines),
        "dash_pattern": dash_pattern,
        "opacity": float(style["opacity"]),
    }


def _hide_stroke_slot(slot, first_unused=0):
    for line in slot["lines"][first_unused:]:
        line.set_stroke(opacity=0.0)


def _update_stroke_slot(slot, start, end, active):
    vector = end - start
    length = float(np.linalg.norm(vector))
    if not active or length <= 1e-10:
        _hide_stroke_slot(slot)
        return
    pattern = slot["dash_pattern"]
    if pattern is None:
        slot["lines"][0].put_start_and_end_on(start, end)
        slot["lines"][0].set_stroke(opacity=slot["opacity"])
        _hide_stroke_slot(slot, 1)
        return
    on_length, off_length = pattern
    direction = vector / length
    cursor = 0.0
    used = 0
    while cursor < length - 1e-9:
        if used >= len(slot["lines"]):
            raise ValueError("animated occlusion line exceeded its stable dash capacity")
        dash_end = min(cursor + on_length, length)
        slot["lines"][used].put_start_and_end_on(
            start + cursor * direction, start + dash_end * direction
        )
        slot["lines"][used].set_stroke(opacity=slot["opacity"])
        used += 1
        cursor += on_length + off_length
    _hide_stroke_slot(slot, used)


def _update_occlusion_slots(slots, start, end, interval):
    vector = end - start
    if interval is None:
        _update_stroke_slot(slots["visible_before"], start, end, True)
        _hide_stroke_slot(slots["hidden"])
        _hide_stroke_slot(slots["visible_after"])
        return
    hidden_start, hidden_end = interval
    _update_stroke_slot(
        slots["visible_before"], start, start + hidden_start * vector, hidden_start > 1e-7
    )
    _update_stroke_slot(
        slots["hidden"],
        start + hidden_start * vector,
        start + hidden_end * vector,
        hidden_end - hidden_start > 1e-7,
    )
    _update_stroke_slot(
        slots["visible_after"], start + hidden_end * vector, end, hidden_end < 1.0 - 1e-7
    )


def _reset_local_camera(state):
    state["local_camera"] = {
        "source": ENTRY_PROJECTION_MATRIX.copy(),
        "target": ENTRY_PROJECTION_MATRIX.copy(),
        "control": ENTRY_PROJECTION_MATRIX.copy(),
        "transition": "linear",
    }
    state["camera_progress"].set_value(CAMERA_PROGRESS_INITIAL)


def install_geometry_3d_updaters(shape, objects, hinge_angle, camera_progress):
    """Attach readable 3D geometry, local-camera and occlusion updaters."""
    missing = sorted(set(OBJECT_SPECS) - set(objects))
    if missing:
        raise RuntimeError("TikZ ShapeState is missing semantic objects: " + ", ".join(missing))
    if shape.updaters or any(objects[object_id].updaters for object_id in OBJECT_SPECS):
        raise RuntimeError("TikZ ShapeState already has active updaters")
    original_shape_children = list(shape.submobjects)
    original_family_state = [
        (
            member,
            list(getattr(member, "updaters", ())),
            bool(getattr(member, "updating_suspended", False)),
            float(getattr(member, "z_index", 0.0)),
        )
        for member in shape.get_family()
    ]
    entry_snapshots = {object_id: objects[object_id].copy() for object_id in OBJECT_SPECS}
    axis_start_name, axis_end_name = HINGE_AXIS_COORDINATE_IDS
    scene_start = _coordinate_anchor(axis_start_name, objects)
    scene_end = _coordinate_anchor(axis_end_name, objects)
    logical_start = _project_point(ENTRY_PROJECTION_MATRIX, AUTHORED_COORDINATES[axis_start_name])[:2]
    logical_end = _project_point(ENTRY_PROJECTION_MATRIX, AUTHORED_COORDINATES[axis_end_name])[:2]
    map_screen, logical_scale = _similarity_mapper(
        logical_start, logical_end, scene_start, scene_end
    )
    scene_unit_per_cm = logical_scale / PICTURE_SCALE
    authored_points = np.asarray(list(AUTHORED_COORDINATES.values()), dtype=float)
    pivot = 0.5 * (authored_points.min(axis=0) + authored_points.max(axis=0))
    entry_pivot = _project_point(ENTRY_PROJECTION_MATRIX, pivot)[:2]
    state = {
        "shape": shape,
        "objects": objects,
        "hinge_angle": hinge_angle,
        "camera_progress": camera_progress,
        "local_camera": {
            "source": ENTRY_PROJECTION_MATRIX.copy(),
            "target": ENTRY_PROJECTION_MATRIX.copy(),
            "control": ENTRY_PROJECTION_MATRIX.copy(),
            "transition": "linear",
        },
        "entry_snapshots": entry_snapshots,
        "original_shape_children": original_shape_children,
        "original_family_state": original_family_state,
        "temporary_groups": [],
        "scene_start": scene_start,
        "scene_end": scene_end,
        "map_screen": map_screen,
        "pivot": pivot,
        "entry_pivot": entry_pivot,
        "coordinate_cache": {"parameter": None, "coordinates": None},
    }
    camera_progress.set_value(CAMERA_PROGRESS_INITIAL)

    def coordinates():
        value = float(hinge_angle.get_value())
        cache = state["coordinate_cache"]
        if cache["coordinates"] is None or cache["parameter"] != value:
            cache["parameter"] = value
            cache["coordinates"] = geometry_coordinates_3d(value)
        return cache["coordinates"]

    def project_scene(value):
        local = _project_point(local_camera_matrix(state), _point3(value) - pivot)[:2]
        return map_screen(local + entry_pivot)

    state["coordinates"] = coordinates
    state["project_scene"] = project_scene
    label_states = {}
    stroke_ratios = []
    for object_id, spec in OBJECT_SPECS.items():
        if spec["kind"] not in {"line", "arrow", "polygon"}:
            continue
        original_width = _visible_stroke_width(objects[object_id])
        # Only relation styles have a frozen authored point width.  Their
        # visible leaves give an exact ShapeState-to-Manim stroke scale.
        authored_width = float(spec["line_width_pt"])
        if original_width is not None and authored_width > 0:
            stroke_ratios.append(original_width / authored_width)
    stroke_width_per_pt = None if not stroke_ratios else float(np.median(stroke_ratios))

    try:
        for member, _updaters, _suspended, _z_index in original_family_state:
            member.clear_updaters(recursive=False)
        for object_id, spec in OBJECT_SPECS.items():
            item = objects[object_id]
            if object_id in OCCLUSION_FRAGMENT_OBJECT_IDS:
                item.set_opacity(0.0)
                continue
            kind = spec["kind"]
            follows_driver = object_id in DYNAMIC_OBJECT_IDS
            if kind in {"line", "arrow"}:
                def update_line(mobject, _dt=0.0, current=spec, dynamic=follows_driver):
                    logical = coordinates() if dynamic else AUTHORED_COORDINATES
                    start = project_scene(
                        _object_logical_point(current, logical, "start_name", "start")
                    )
                    end = project_scene(
                        _object_logical_point(current, logical, "end_name", "end")
                    )
                    mobject.put_start_and_end_on(start, end)
                item.add_updater(update_line)
            elif kind == "polygon":
                def update_polygon(mobject, _dt=0.0, current=spec, dynamic=follows_driver):
                    logical = coordinates() if dynamic else AUTHORED_COORDINATES
                    names = current["point_names"]
                    points = (
                        [project_scene(logical[name]) for name in names]
                        if names is not None
                        else [project_scene(value) for value in current["points"]]
                    )
                    if len(points) < 3:
                        raise ValueError("animated polygon has fewer than three points")
                    mobject.set_points_as_corners([*points, points[0]])
                item.add_updater(update_polygon)
            elif kind == "dot":
                def update_dot(mobject, _dt=0.0, current=spec, dynamic=follows_driver):
                    logical = coordinates() if dynamic else AUTHORED_COORDINATES
                    mobject.move_to(
                        project_scene(
                            _object_logical_point(current, logical, "center_name", "center")
                        )
                    )
                item.add_updater(update_dot)
            elif kind in {"label", "path_label"}:
                logical = coordinates() if follows_driver else AUTHORED_COORDINATES
                if kind == "label":
                    anchor = project_scene(
                        _object_logical_point(spec, logical, "at_name", "at")
                    )
                    label_states[object_id] = {"offset": _point3(item.get_center()) - anchor}
                else:
                    start = project_scene(
                        _object_logical_point(spec, logical, "start_name", "start")
                    )
                    end = project_scene(
                        _object_logical_point(spec, logical, "end_name", "end")
                    )
                    anchor = start + spec["pos"] * (end - start)
                    offset = _point3(item.get_center()) - anchor
                    record = {"offset": offset}
                    if spec["sloped"]:
                        vector = end[:2] - start[:2]
                        length = float(np.linalg.norm(vector))
                        if length <= 1e-12:
                            raise ValueError(f"path label {object_id!r} has a collapsed entry path")
                        tangent = vector / length
                        normal = np.array((-tangent[1], tangent[0]), dtype=float)
                        record.update(
                            {
                                "tangent_offset": float(np.dot(offset[:2], tangent)),
                                "normal_offset": float(np.dot(offset[:2], normal)),
                                "angle": float(np.arctan2(tangent[1], tangent[0])),
                            }
                        )
                    label_states[object_id] = record
                def update_label(
                    mobject,
                    _dt=0.0,
                    current=spec,
                    current_id=object_id,
                    dynamic=follows_driver,
                ):
                    record = label_states[current_id]
                    logical = coordinates() if dynamic else AUTHORED_COORDINATES
                    if current["kind"] == "label":
                        current_anchor = project_scene(
                            _object_logical_point(current, logical, "at_name", "at")
                        )
                        offset = record["offset"]
                    else:
                        start = project_scene(
                            _object_logical_point(current, logical, "start_name", "start")
                        )
                        end = project_scene(
                            _object_logical_point(current, logical, "end_name", "end")
                        )
                        current_anchor = start + current["pos"] * (end - start)
                        if "angle" not in record:
                            offset = record["offset"]
                        else:
                            vector = end[:2] - start[:2]
                            length = float(np.linalg.norm(vector))
                            if length <= 1e-12:
                                raise ValueError(f"path label {current_id!r} collapsed during motion")
                            tangent = vector / length
                            normal = np.array((-tangent[1], tangent[0]), dtype=float)
                            angle = float(np.arctan2(tangent[1], tangent[0]))
                            mobject.rotate(angle - record["angle"], about_point=mobject.get_center())
                            record["angle"] = angle
                            offset_2d = (
                                record["tangent_offset"] * tangent
                                + record["normal_offset"] * normal
                            )
                            offset = np.array((offset_2d[0], offset_2d[1], record["offset"][2]))
                    mobject.move_to(current_anchor + offset)
                item.add_updater(update_label)

        coordinate_bounds = {
            name: float(np.linalg.norm(point)) for name, point in AUTHORED_COORDINATES.items()
        }
        axis_origin_norm = float(np.linalg.norm(AUTHORED_COORDINATES[axis_start_name]))
        for name in HINGE_MOVING_COORDINATE_IDS:
            coordinate_bounds[name] = axis_origin_norm + float(
                np.linalg.norm(
                    np.asarray(AUTHORED_COORDINATES[name])
                    - np.asarray(AUTHORED_COORDINATES[axis_start_name])
                )
            )
        for relation in DERIVED_COORDINATES:
            if relation["type"] == "point_on_segment":
                coordinate_bounds[relation["name"]] = max(
                    coordinate_bounds[relation["start"]], coordinate_bounds[relation["end"]]
                )
            else:
                coordinate_bounds[relation["name"]] = (
                    coordinate_bounds[relation["point"]]
                    + 2.0 * coordinate_bounds[relation["line_start"]]
                )
        projection_bound = max(
            float(np.linalg.norm(matrix[:2], ord=2))
            for matrix in (ENTRY_PROJECTION_MATRIX, *CAMERA_PRESET_MATRICES.values())
        )
        for relation in OCCLUSION_RELATIONS:
            start = project_scene(coordinates()[relation["start_name"]])
            safe_length = (
                1.05
                * logical_scale
                * projection_bound
                * (
                    coordinate_bounds[relation["start_name"]]
                    + coordinate_bounds[relation["end_name"]]
                )
                + 1e-6
            )
            allocation_start = np.array((0.0, 0.0, start[2]), dtype=float)
            allocation_end = np.array((safe_length, 0.0, start[2]), dtype=float)
            slots = {
                "visible_before": _make_stroke_slot(
                    allocation_start,
                    allocation_end,
                    relation["visible_style"],
                    relation["z_index"],
                    scene_unit_per_cm,
                    stroke_width_per_pt,
                ),
                "hidden": _make_stroke_slot(
                    allocation_start,
                    allocation_end,
                    relation["hidden_style"],
                    relation["z_index"],
                    scene_unit_per_cm,
                    stroke_width_per_pt,
                ),
                "visible_after": _make_stroke_slot(
                    allocation_start,
                    allocation_end,
                    relation["visible_style"],
                    relation["z_index"],
                    scene_unit_per_cm,
                    stroke_width_per_pt,
                ),
            }
            container = VGroup(
                *slots["visible_before"]["lines"],
                *slots["hidden"]["lines"],
                *slots["visible_after"]["lines"],
            )
            container.set_z_index(relation["z_index"])
            shape.add(container)
            state["temporary_groups"].append(container)
            def update_occlusion(_mobject, _dt=0.0, current=relation, stable=slots):
                current_coordinates = coordinates()
                start_world = current_coordinates[current["start_name"]]
                end_world = current_coordinates[current["end_name"]]
                face = [current_coordinates[name] for name in current["face_names"]]
                interval = parallel_occlusion_interval(
                    start_world,
                    end_world,
                    face,
                    parallel_view_direction(local_camera_matrix(state)),
                )
                _update_occlusion_slots(
                    stable, project_scene(start_world), project_scene(end_world), interval
                )
            container.add_updater(update_occlusion)
        shape.update(0.0)
        assert_shape_state_3d_entry(state)
    except Exception:
        restore_geometry_3d_objects(state)
        raise
    return state


def assert_shape_state_3d_entry(state):
    """Fail if installing the readable 3D rig changes the visible entry frame."""
    for object_id, item in state["objects"].items():
        if object_id in OCCLUSION_FRAGMENT_OBJECT_IDS or object_id not in state["entry_snapshots"]:
            continue
        original = state["entry_snapshots"][object_id]
        current_points = item.get_all_points()
        original_points = original.get_all_points()
        if current_points.shape != original_points.shape or not np.allclose(
            current_points, original_points, atol=1e-7, rtol=0.0
        ):
            raise RuntimeError(
                "local 3D entry projection does not align with semantic object "
                + repr(object_id)
            )
    if not np.allclose(
        _coordinate_anchor(HINGE_AXIS_COORDINATE_IDS[0], state["objects"]),
        state["scene_start"],
        atol=1e-7,
        rtol=0.0,
    ) or not np.allclose(
        _coordinate_anchor(HINGE_AXIS_COORDINATE_IDS[1], state["objects"]),
        state["scene_end"],
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("hinge driver initial state does not align with the ShapeState")
    return True


def restore_geometry_3d_objects(state):
    """Remove temporary slots and restore every object/updater/z value exactly."""
    state["hinge_angle"].set_value(HINGE_ANGLE_INITIAL)
    _reset_local_camera(state)
    shape = state["shape"]
    shape.clear_updaters(recursive=False)
    for object_id, original in state["entry_snapshots"].items():
        item = state["objects"][object_id]
        item.clear_updaters(recursive=False)
        item.become(original)
        item.clear_updaters(recursive=False)
    for group in state["temporary_groups"]:
        group.clear_updaters()
        shape.remove(group)
    shape.submobjects[:] = state["original_shape_children"]
    for member, updaters, suspended, z_index in state["original_family_state"]:
        member.updaters[:] = updaters
        member.updating_suspended = suspended
        member.set_z_index(z_index, family=False)
'''


def _current_source_helpers() -> str:
    """Use the runtime occlusion kernel in the portable generated source."""

    placeholder = "# __TIKZ_NATIVE_OCCLUSION_3D_KERNEL__"
    if _SOURCE_HELPERS.count(placeholder) != 1:
        raise RuntimeError("generated 3D source has an invalid occlusion placeholder")
    return _SOURCE_HELPERS.replace(placeholder, standalone_occlusion_source())


def generate_native_manim_source_3d(
    picture: PictureSpec,
    rig: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the readable Manim definition section for one ready hinge rig."""

    if picture.dimension != 3 or picture.projection_3d is None:
        raise NativeManimCodegen3DError("native Manim 3D source requires a 3D picture")
    if rig.get("status") != "ready":
        raise NativeManimCodegen3DError("native Manim 3D source requires a ready rig")
    selected = rig.get("selectedMotionCandidate")
    motion = rig.get("motionSpecCore")
    if not isinstance(selected, Mapping) or not isinstance(motion, Mapping):
        raise NativeManimCodegen3DError("ready rig has no selected motion core")
    driver = motion.get("driver")
    bindings = motion.get("bindings")
    derived = motion.get("derived_coordinates")
    if not isinstance(driver, Mapping) or driver.get("type") != "hinge_fold":
        raise NativeManimCodegen3DError("motion core has no hinge_fold driver")
    if not isinstance(bindings, list) or not bindings:
        raise NativeManimCodegen3DError("motion core has no native object bindings")
    if not isinstance(derived, list):
        raise NativeManimCodegen3DError("motion core derived_coordinates is invalid")
    axis = tuple(str(value) for value in driver.get("axis", ()))
    moving_points = tuple(str(value) for value in driver.get("moving_points", ()))
    raw_range = driver.get("range")
    if len(axis) != 2 or not moving_points:
        raise NativeManimCodegen3DError("hinge driver axis/moving_points is incomplete")
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise NativeManimCodegen3DError("hinge driver range is invalid")

    object_specs = {item.id: _object_payload(item) for item in picture.objects}
    binding_payload = [
        {
            "object_id": str(item["object_id"]),
            "type": str(item["type"]),
            "points": tuple(str(value) for value in item["points"]),
        }
        for item in bindings
    ]
    dynamic_ids = tuple(item["object_id"] for item in binding_payload)
    active_object_id = str(selected.get("activeObjectId") or "")
    if not active_object_id:
        raise NativeManimCodegen3DError("selected hinge has no activeObjectId")
    occlusion_payload = [
        {
            "relation_id": relation.id,
            "start_name": relation.start_name,
            "end_name": relation.end_name,
            "face_names": tuple(relation.face_names),
            "object_ids": tuple(relation.object_ids),
            "visible_style": _style_payload(relation.visible_style),
            "hidden_style": _style_payload(relation.hidden_style),
            "z_index": int(relation.z_index),
        }
        for relation in picture.occlusion_relations
    ]
    relation_members = tuple(
        dict.fromkeys(
            object_id
            for relation in picture.occlusion_relations
            for object_id in relation.object_ids
        )
    )
    root_two = 2.0 ** 0.5
    root_six = 6.0 ** 0.5
    root_three = 3.0 ** 0.5
    oblique_r = root_two / 4.0
    oblique_direction = (1.0, oblique_r, oblique_r)
    oblique_norm = sum(value * value for value in oblique_direction) ** 0.5
    presets = {
        "front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        "side": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        "top": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "oblique": (
            (-oblique_r, 1.0, 0.0),
            (-oblique_r, 0.0, 1.0),
            tuple(value / oblique_norm for value in oblique_direction),
        ),
        "isometric": (
            (-1.0 / root_two, 1.0 / root_two, 0.0),
            (-1.0 / root_six, -1.0 / root_six, 2.0 / root_six),
            (1.0 / root_three, 1.0 / root_three, 1.0 / root_three),
        ),
    }
    source_lines = [
        "import numpy as np",
        "from manim import *",
        "",
        "# ===== Provider 展开的三维几何、局部相机与动态遮挡 =====",
        f"HINGE_PARAMETER_ID = {str(driver.get('id') or selected.get('relationId'))!r}",
        f"HINGE_AXIS_COORDINATE_IDS = {_literal(axis)}",
        f"HINGE_MOVING_COORDINATE_IDS = {_literal(moving_points)}",
        f"HINGE_ANGLE_INITIAL = {float(driver['initial'])!r}",
        f"HINGE_ANGLE_MINIMUM = {float(raw_range[0])!r}",
        f"HINGE_ANGLE_MAXIMUM = {float(raw_range[1])!r}",
        "CAMERA_PROGRESS_INITIAL = 1.0",
        f"ACTIVE_OBJECT_ID = {active_object_id!r}",
        f"DYNAMIC_OBJECT_IDS = {_literal(dynamic_ids)}",
        f"FOLLOWER_OBJECT_IDS = {_literal(tuple(item for item in dynamic_ids if item != active_object_id))}",
        f"FIXED_OBJECT_IDS = {_literal(tuple(str(item) for item in rig.get('fixedObjectIds', [])))}",
        f"DISABLED_OBJECT_IDS = {_literal(tuple(str(item) for item in rig.get('excludedObjectIds', [])))}",
        f"DYNAMIC_BINDINGS = {_literal(binding_payload)}",
        f"DERIVED_COORDINATES = {_literal([dict(item) for item in derived])}",
        f"AUTHORED_COORDINATES = {_literal(_authored_coordinates(picture))}",
        f"OBJECT_SPECS = {_literal(object_specs)}",
        f"OCCLUSION_RELATIONS = {_literal(occlusion_payload)}",
        f"OCCLUSION_FRAGMENT_OBJECT_IDS = frozenset({_literal(relation_members)})",
        f"ENTRY_PROJECTION_MATRIX = np.array({_literal(tuple(tuple(float(value) for value in row) for row in picture.projection_3d.matrix))}, dtype=float)",
        f"CAMERA_PRESET_MATRICES = {{name: np.array(value, dtype=float) for name, value in {_literal(presets)}.items()}}",
        f"PICTURE_SCALE = {float(picture.scale)!r}",
        "TEX_POINTS_PER_CM = 72.27 / 2.54",
    ]
    source_text = "\n".join(source_lines) + _current_source_helpers() + "\n"
    compile(source_text, "<tikz-native-manim-source-3d>", "exec")
    return {
        "schema": NATIVE_MANIM_SOURCE_3D_SCHEMA,
        "sourceText": source_text,
        "sourceSha256": _source_hash(source_text),
    }


__all__ = [
    "NATIVE_MANIM_SOURCE_3D_SCHEMA",
    "NativeManimCodegen3DError",
    "generate_native_manim_source_3d",
]
