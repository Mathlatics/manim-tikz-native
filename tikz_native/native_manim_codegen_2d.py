from __future__ import annotations

"""Expand one supported TikZ 2D geometry rig into readable Manim source.

The emitted source deliberately contains the geometry equations and ordinary
Manim updaters.  It does not import the Provider runtime or hide motion behind
``NativeGeometryRig2D``.  Host-specific frozen-file checks remain outside the
fragment and can therefore live in a managed project module.
"""

import hashlib
from math import sqrt
from pprint import pformat
from typing import Any, Mapping

from .compiler import ObjectSpec, PictureSpec


NATIVE_MANIM_SOURCE_2D_SCHEMA = "tikz-native-manim-source-2d/v1"


class NativeManimCodegen2DError(ValueError):
    """The selected semantic rig cannot be expanded safely."""


def _stable_source_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def _literal(value: Any) -> str:
    """Stable readable Python literal for JSON-compatible Provider data."""

    return pformat(value, width=88, indent=4, compact=False, sort_dicts=True)


def _identifier_tuple(values: list[str]) -> str:
    return _literal(tuple(values))


def _selected_geometry(
    picture: PictureSpec,
    rig: Mapping[str, Any],
) -> dict[str, Any]:
    if rig.get("status") != "ready":
        raise NativeManimCodegen2DError("only a ready geometry rig can emit source")
    selected = rig.get("selectedDriver")
    core = rig.get("motionSpecCore")
    if not isinstance(selected, Mapping) or not isinstance(core, Mapping):
        raise NativeManimCodegen2DError("ready geometry rig has no selected driver")
    if selected.get("type") != "rotate_named_line":
        raise NativeManimCodegen2DError("only rotate_named_line can emit v1 source")

    active_path_name = str(selected.get("activePath") or "")
    active_path = picture.named_paths.get(active_path_name)
    if active_path is None or active_path.kind != "line":
        raise NativeManimCodegen2DError("active path is not an oriented named line")
    start_name = str(active_path.geometry.get("start_name") or "")
    end_name = str(active_path.geometry.get("end_name") or "")
    pivot_name = str(selected.get("pivot") or "")
    if not start_name or not end_name or pivot_name not in picture.coordinates:
        raise NativeManimCodegen2DError("active line endpoints or pivot are missing")

    relation_index = int(selected.get("intersectionIndex", -1))
    try:
        relation = picture.intersections[relation_index]
    except IndexError as exc:
        raise NativeManimCodegen2DError("selected intersection is unavailable") from exc
    if relation.sort_by != active_path_name or len(relation.coordinate_names) != 2:
        raise NativeManimCodegen2DError(
            "selected intersection is not a two-point oriented line intersection"
        )
    ellipse_name = (
        relation.path_b if relation.path_a == relation.sort_by else relation.path_a
    )
    ellipse = picture.named_paths.get(ellipse_name)
    if ellipse is None or ellipse.kind != "ellipse":
        raise NativeManimCodegen2DError("selected driver does not intersect an ellipse")

    pivot = picture.coordinates[pivot_name]
    start = tuple(float(value) for value in active_path.geometry["start"])
    end = tuple(float(value) for value in active_path.geometry["end"])
    direction = (end[0] - start[0], end[1] - start[1])
    length = sqrt(direction[0] ** 2 + direction[1] ** 2)
    if length <= 1e-12:
        raise NativeManimCodegen2DError("active line has zero length")
    unit = (direction[0] / length, direction[1] / length)

    def signed_distance(point: tuple[float, float]) -> float:
        relative = (point[0] - pivot[0], point[1] - pivot[1])
        perpendicular = relative[0] * unit[1] - relative[1] * unit[0]
        if abs(perpendicular) > 1e-8:
            raise NativeManimCodegen2DError("pivot is not on the active line")
        return relative[0] * unit[0] + relative[1] * unit[1]

    start_distance = signed_distance(start)
    end_distance = signed_distance(end)
    if start_distance >= 0 or end_distance <= 0:
        raise NativeManimCodegen2DError(
            "active line must run from the negative to the positive pivot side"
        )

    raw_bindings = core.get("bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise NativeManimCodegen2DError("ready geometry rig has no bindings")
    object_specs = {item.id: item for item in picture.objects}
    bindings: list[dict[str, Any]] = []
    for index, item in enumerate(raw_bindings):
        if not isinstance(item, Mapping):
            raise NativeManimCodegen2DError(f"binding {index} is not an object")
        object_id = str(item.get("object_id") or "")
        binding_type = str(item.get("type") or "")
        points = item.get("points")
        spec = object_specs.get(object_id)
        if spec is None:
            raise NativeManimCodegen2DError(f"unknown bound object {object_id!r}")
        if not isinstance(points, list) or not points or any(
            not isinstance(name, str) or name not in picture.coordinates
            for name in points
        ):
            raise NativeManimCodegen2DError(
                f"binding {object_id!r} has invalid coordinate names"
            )
        if binding_type not in {
            "line",
            "dot",
            "polygon",
            "label",
            "path_label",
            "angle",
            "angle_label",
            "right_angle",
        }:
            raise NativeManimCodegen2DError(
                f"binding {object_id!r} uses unsupported type {binding_type!r}"
            )
        bindings.append(
            {
                "object_id": object_id,
                "type": binding_type,
                "points": list(points),
                "object_kind": spec.kind,
            }
        )

    return {
        "selected": dict(selected),
        "core": dict(core),
        "active_path_name": active_path_name,
        "start_name": start_name,
        "end_name": end_name,
        "pivot_name": pivot_name,
        "backward_length": -start_distance,
        "forward_length": end_distance,
        "intersection_names": tuple(relation.coordinate_names),
        "ellipse_name": ellipse_name,
        "ellipse_center": tuple(float(value) for value in ellipse.geometry["center"]),
        "ellipse_rx": float(ellipse.geometry["rx"]),
        "ellipse_ry": float(ellipse.geometry["ry"]),
        "bindings": bindings,
        "object_specs": object_specs,
    }


def _dependency_lines(
    picture: PictureSpec,
    *,
    selected_names: set[str],
    affected_names: set[str],
) -> list[str]:
    emitted = set(selected_names)
    pending = [
        name
        for name in picture.coordinates
        if name in affected_names and name not in emitted
    ]
    lines: list[str] = []
    while pending:
        progressed = False
        for name in tuple(pending):
            dependency = picture.coordinate_dependencies.get(name)
            if not isinstance(dependency, Mapping):
                pending.remove(name)
                emitted.add(name)
                progressed = True
                continue
            operation = str(dependency.get("operation") or "")
            if operation == "reference":
                required = [str(dependency.get("coordinate") or "")]
            elif operation == "interpolation":
                required = [
                    str(dependency.get("start") or ""),
                    str(dependency.get("end") or ""),
                ]
            elif operation == "translation":
                required = [str(dependency.get("base") or "")]
            elif operation == "projection":
                required = [
                    str(dependency.get("line_start") or ""),
                    str(dependency.get("point") or ""),
                    str(dependency.get("line_end") or ""),
                ]
            elif operation == "intersection":
                raise NativeManimCodegen2DError(
                    f"unselected dynamic intersection {name!r} cannot emit source"
                )
            else:
                raise NativeManimCodegen2DError(
                    f"coordinate {name!r} uses unsupported operation {operation!r}"
                )
            if not all(item in emitted or item not in affected_names for item in required):
                continue
            if operation == "reference":
                lines.append(
                    f"    coordinates[{name!r}] = coordinates[{required[0]!r}].copy()"
                )
            elif operation == "interpolation":
                amount = float(dependency["parameter"])
                lines.extend(
                    [
                        f"    coordinates[{name!r}] = (",
                        f"        coordinates[{required[0]!r}]",
                        f"        + {amount!r} * (coordinates[{required[1]!r}] - coordinates[{required[0]!r}])",
                        "    )",
                    ]
                )
            elif operation == "translation":
                offset = tuple(float(item) for item in dependency["offset"])
                lines.append(
                    f"    coordinates[{name!r}] = coordinates[{required[0]!r}] + np.array({offset!r}, dtype=float)"
                )
            elif operation == "projection":
                lines.extend(
                    [
                        f"    coordinates[{name!r}] = project_point_to_line(",
                        f"        coordinates[{required[1]!r}],",
                        f"        coordinates[{required[0]!r}],",
                        f"        coordinates[{required[2]!r}],",
                        "    )",
                    ]
                )
            pending.remove(name)
            emitted.add(name)
            progressed = True
        if not progressed:
            raise NativeManimCodegen2DError(
                "coordinate dependencies cannot be emitted in a stable order: "
                + ", ".join(pending)
            )
    return lines


def _binding_lines(
    binding: Mapping[str, Any],
    spec: ObjectSpec,
    *,
    index: int,
) -> list[str]:
    object_id = str(binding["object_id"])
    binding_type = str(binding["type"])
    points = [str(item) for item in binding["points"]]
    function_name = f"update_object_{index}"
    comment = f"    # {object_id}: {binding_type}({', '.join(points)})"
    lines = [comment]
    if binding_type == "line":
        if len(points) != 2:
            raise NativeManimCodegen2DError(f"line {object_id!r} needs two points")
        if spec.style.dash_pattern_pt:
            on_pt, off_pt = spec.style.dash_pattern_pt
            lines.extend(
                [
                    f"    line_template_{index} = objects[{object_id!r}].copy()",
                    f"    def {function_name}(item):",
                    "        item.become(",
                    "            native_dashed_line(",
                    f"                point({points[0]!r}), point({points[1]!r}),",
                    f"                on_pt={float(on_pt)!r}, off_pt={float(off_pt)!r},",
                    f"                scene_unit_per_cm=scene_unit_per_cm, template=line_template_{index},",
                    "            )",
                    "        )",
                ]
            )
        else:
            lines.extend(
                [
                    f"    def {function_name}(item):",
                    f"        item.put_start_and_end_on(point({points[0]!r}), point({points[1]!r}))",
                ]
            )
    elif binding_type == "dot":
        if len(points) != 1:
            raise NativeManimCodegen2DError(f"dot {object_id!r} needs one point")
        lines.extend(
            [
                f"    def {function_name}(item):",
                f"        item.move_to(point({points[0]!r}))",
            ]
        )
    elif binding_type == "polygon":
        if len(points) < 3:
            raise NativeManimCodegen2DError(f"polygon {object_id!r} needs three points")
        point_list = ", ".join(f"point({name!r})" for name in points)
        lines.extend(
            [
                f"    polygon_template_{index} = objects[{object_id!r}].copy()",
                f"    def {function_name}(item):",
                f"        updated = Polygon({point_list})",
                f"        match_geometry_style(updated, polygon_template_{index})",
                f"        updated.set_z_index(polygon_template_{index}.z_index)",
                "        item.become(updated)",
            ]
        )
    elif binding_type == "label":
        if len(points) != 1:
            raise NativeManimCodegen2DError(f"label {object_id!r} needs one point")
        lines.extend(
            [
                f"    label_offset_{index} = objects[{object_id!r}].get_center() - point({points[0]!r})",
                f"    def {function_name}(item):",
                f"        item.move_to(point({points[0]!r}) + label_offset_{index})",
            ]
        )
    elif binding_type == "path_label":
        if len(points) != 2 or spec.placement is None:
            raise NativeManimCodegen2DError(
                f"path label {object_id!r} has incomplete placement"
            )
        path_pos = float(spec.geometry["pos"])
        sloped = bool(spec.placement.sloped)
        lines.extend(
            [
                f"    path_label_template_{index} = objects[{object_id!r}].copy()",
                f"    path_start_{index} = point({points[0]!r})",
                f"    path_end_{index} = point({points[1]!r})",
                f"    path_vector_{index} = path_end_{index} - path_start_{index}",
                f"    path_tangent_{index} = path_vector_{index} / np.linalg.norm(path_vector_{index})",
                f"    path_normal_{index} = np.array([-path_tangent_{index}[1], path_tangent_{index}[0], 0.0])",
                f"    path_base_{index} = path_start_{index} + {path_pos!r} * path_vector_{index}",
                f"    path_offset_{index} = objects[{object_id!r}].get_center() - path_base_{index}",
                f"    path_tangent_offset_{index} = float(np.dot(path_offset_{index}, path_tangent_{index}))",
                f"    path_normal_offset_{index} = float(np.dot(path_offset_{index}, path_normal_{index}))",
                f"    path_angle_{index} = display_angle(path_start_{index}, path_end_{index})",
                f"    def {function_name}(item):",
                f"        start = point({points[0]!r})",
                f"        end = point({points[1]!r})",
                "        vector = end - start",
                "        tangent = vector / np.linalg.norm(vector)",
                "        normal = np.array([-tangent[1], tangent[0], 0.0])",
                f"        base = start + {path_pos!r} * vector",
            ]
        )
        if sloped:
            lines.extend(
                [
                    f"        center = base + path_tangent_offset_{index} * tangent + path_normal_offset_{index} * normal",
                    f"        updated = path_label_template_{index}.copy()",
                    f"        updated.rotate(display_angle(start, end) - path_angle_{index}, about_point=updated.get_center())",
                ]
            )
        else:
            lines.extend(
                [
                    f"        center = base + path_offset_{index}",
                    f"        updated = path_label_template_{index}.copy()",
                ]
            )
        lines.extend(
            [
                "        updated.move_to(center)",
                f"        updated.set_z_index(path_label_template_{index}.z_index)",
                "        item.become(updated)",
            ]
        )
    elif binding_type == "angle":
        if len(points) != 3:
            raise NativeManimCodegen2DError(f"angle {object_id!r} needs three points")
        radius_pt = float(spec.geometry["radius_pt"])
        lines.extend(
            [
                f"    angle_template_{index} = objects[{object_id!r}].copy()",
                f"    angle_radius_{index} = {radius_pt!r} * scene_unit_per_cm / TEX_POINTS_PER_CM",
                f"    def {function_name}(item):",
                f"        first, vertex, third = point({points[0]!r}), point({points[1]!r}), point({points[2]!r})",
                "        start_angle, end_angle = directed_angles(first, vertex, third)",
                f"        updated = Arc(radius=angle_radius_{index}, start_angle=start_angle, angle=end_angle - start_angle, arc_center=vertex)",
                f"        match_geometry_style(updated, angle_template_{index})",
                f"        updated.set_z_index(angle_template_{index}.z_index)",
                "        item.become(updated)",
            ]
        )
    elif binding_type == "angle_label":
        if len(points) != 3:
            raise NativeManimCodegen2DError(
                f"angle label {object_id!r} needs three points"
            )
        radius_pt = float(spec.geometry["radius_pt"])
        eccentricity = float(spec.geometry["eccentricity"])
        lines.extend(
            [
                f"    angle_label_radius_{index} = {radius_pt!r} * scene_unit_per_cm / TEX_POINTS_PER_CM",
                f"    def angle_label_target_{index}():",
                f"        first, vertex, third = point({points[0]!r}), point({points[1]!r}), point({points[2]!r})",
                "        start_angle, end_angle = directed_angles(first, vertex, third)",
                "        midpoint = 0.5 * (start_angle + end_angle)",
                f"        return vertex + {eccentricity!r} * angle_label_radius_{index} * np.array([np.cos(midpoint), np.sin(midpoint), 0.0])",
                f"    angle_label_offset_{index} = objects[{object_id!r}].get_center() - angle_label_target_{index}()",
                f"    def {function_name}(item):",
                f"        item.move_to(angle_label_target_{index}() + angle_label_offset_{index})",
            ]
        )
    elif binding_type == "right_angle":
        if len(points) != 3:
            raise NativeManimCodegen2DError(
                f"right angle {object_id!r} needs three points"
            )
        radius_pt = float(spec.geometry["radius_pt"])
        lines.extend(
            [
                f"    right_angle_template_{index} = objects[{object_id!r}].copy()",
                f"    right_angle_length_{index} = {radius_pt!r} * scene_unit_per_cm / TEX_POINTS_PER_CM",
                f"    def {function_name}(item):",
                f"        first, vertex, third = point({points[0]!r}), point({points[1]!r}), point({points[2]!r})",
                "        updated = RightAngle(Line(vertex, first), Line(vertex, third), length=right_angle_length_" + str(index) + ")",
                f"        match_geometry_style(updated, right_angle_template_{index})",
                f"        updated.set_z_index(right_angle_template_{index}.z_index)",
                "        item.become(updated)",
            ]
        )
    else:  # pragma: no cover - selected geometry validates the enum.
        raise NativeManimCodegen2DError(f"unsupported binding type {binding_type!r}")
    lines.append(f"    objects[{object_id!r}].add_updater({function_name})")
    return lines


def generate_native_manim_source_2d(
    picture: PictureSpec,
    rig: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a self-contained readable Manim source fragment for one rig."""

    model = _selected_geometry(picture, rig)
    selected = model["selected"]
    bindings = model["bindings"]
    object_specs: dict[str, ObjectSpec] = model["object_specs"]
    coordinate_records = rig.get("coordinates")
    if not isinstance(coordinate_records, list):
        raise NativeManimCodegen2DError("geometry rig has no coordinate records")
    affected_names = {
        str(item["coordinateId"])
        for item in coordinate_records
        if isinstance(item, Mapping) and item.get("affectedByDriver") is True
    }
    q_name, p_name = model["intersection_names"]
    selected_names = {
        model["start_name"],
        model["end_name"],
        q_name,
        p_name,
    }
    dependency_lines = _dependency_lines(
        picture,
        selected_names=selected_names,
        affected_names=affected_names,
    )
    dynamic_ids = [str(item["object_id"]) for item in bindings]
    follower_ids = [
        object_id
        for object_id in dynamic_ids
        if object_id != str(selected["activeObjectId"])
    ]
    fixed_coordinate_ids = [str(item) for item in rig.get("fixedCoordinateIds", [])]
    fixed_object_ids = [str(item) for item in rig.get("fixedObjectIds", [])]
    disabled_object_ids = [str(item) for item in rig.get("excludedObjectIds", [])]
    logical_coordinates = {
        name: tuple(float(value) for value in picture.coordinates[name])
        for name in picture.coordinates
    }
    binding_payload = [
        {
            "object_id": item["object_id"],
            "type": item["type"],
            "points": list(item["points"]),
        }
        for item in bindings
    ]
    lower, upper = (float(value) for value in selected["range"])
    lines = [
        "import numpy as np",
        "from manim import *",
        "",
        "# ===== Provider 展开的二维几何关系 =====",
        f"ACTIVE_OBJECT_ID = {str(selected['activeObjectId'])!r}",
        f"PIVOT_COORDINATE_ID = {model['pivot_name']!r}",
        f"PARAMETER_ID = {str(selected['id'])!r}",
        f"PARAMETER_INITIAL = {float(selected['initial'])!r}",
        f"PARAMETER_RANGE = {(lower, upper)!r}",
        "PARAMETER_MINIMUM, PARAMETER_MAXIMUM = PARAMETER_RANGE",
        f"FOLLOWER_OBJECT_IDS = {_identifier_tuple(follower_ids)}",
        f"FIXED_COORDINATE_IDS = {_identifier_tuple(fixed_coordinate_ids)}",
        f"FIXED_OBJECT_IDS = {_identifier_tuple(fixed_object_ids)}",
        f"DISABLED_OBJECT_IDS = {_identifier_tuple(disabled_object_ids)}",
        f"DYNAMIC_OBJECT_IDS = {_identifier_tuple(dynamic_ids)}",
        f"DYNAMIC_BINDINGS = {_literal(binding_payload)}",
        f"LOGICAL_COORDINATES = {_literal(logical_coordinates)}",
        f"ACTIVE_START_COORDINATE_ID = {model['start_name']!r}",
        f"ACTIVE_END_COORDINATE_ID = {model['end_name']!r}",
        f"INTERSECTION_COORDINATE_IDS = {(q_name, p_name)!r}",
        f"ELLIPSE_CENTER = np.array({model['ellipse_center']!r}, dtype=float)",
        f"ELLIPSE_SEMI_MAJOR = {model['ellipse_rx']!r}",
        f"ELLIPSE_SEMI_MINOR = {model['ellipse_ry']!r}",
        f"ACTIVE_BACKWARD_LENGTH = {model['backward_length']!r}",
        f"ACTIVE_FORWARD_LENGTH = {model['forward_length']!r}",
        f"PICTURE_SCALE = {float(picture.scale)!r}",
        "TEX_POINTS_PER_CM = 72.27 / 2.54",
        "",
        "",
        "def project_point_to_line(point, line_start, line_end):",
        "    direction = line_end - line_start",
        "    denominator = float(np.dot(direction, direction))",
        "    if denominator <= 1e-18:",
        "        raise ValueError('Cannot project onto a zero-length line')",
        "    amount = float(np.dot(point - line_start, direction)) / denominator",
        "    return line_start + amount * direction",
        "",
        "",
        "def geometry_coordinates(theta):",
        "    if not PARAMETER_MINIMUM <= theta <= PARAMETER_MAXIMUM:",
        "        raise ValueError(f'{PARAMETER_ID}={theta} is outside PARAMETER_RANGE')",
        "    direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)",
        "    focus = np.array(LOGICAL_COORDINATES[PIVOT_COORDINATE_ID], dtype=float)",
        "    relative_focus = focus - ELLIPSE_CENTER",
        "    quadratic_a = (",
        "        direction[0] ** 2 / ELLIPSE_SEMI_MAJOR ** 2",
        "        + direction[1] ** 2 / ELLIPSE_SEMI_MINOR ** 2",
        "    )",
        "    quadratic_b = 2 * (",
        "        relative_focus[0] * direction[0] / ELLIPSE_SEMI_MAJOR ** 2",
        "        + relative_focus[1] * direction[1] / ELLIPSE_SEMI_MINOR ** 2",
        "    )",
        "    quadratic_c = (",
        "        relative_focus[0] ** 2 / ELLIPSE_SEMI_MAJOR ** 2",
        "        + relative_focus[1] ** 2 / ELLIPSE_SEMI_MINOR ** 2",
        "        - 1",
        "    )",
        "    discriminant = quadratic_b ** 2 - 4 * quadratic_a * quadratic_c",
        "    if discriminant < -1e-12:",
        "        raise ValueError('The moving line does not intersect the ellipse')",
        "    root = np.sqrt(max(discriminant, 0.0))",
        "    parameters = sorted((",
        "        (-quadratic_b - root) / (2 * quadratic_a),",
        "        (-quadratic_b + root) / (2 * quadratic_a),",
        "    ))",
        "    coordinates = {",
        "        name: np.array(value, dtype=float)",
        "        for name, value in LOGICAL_COORDINATES.items()",
        "    }",
        "    coordinates[ACTIVE_START_COORDINATE_ID] = focus - ACTIVE_BACKWARD_LENGTH * direction",
        "    coordinates[ACTIVE_END_COORDINATE_ID] = focus + ACTIVE_FORWARD_LENGTH * direction",
        "    coordinates[INTERSECTION_COORDINATE_IDS[0]] = focus + parameters[0] * direction",
        "    coordinates[INTERSECTION_COORDINATE_IDS[1]] = focus + parameters[1] * direction",
        *dependency_lines,
        "    return coordinates",
        "",
        "",
        "def scene_mapper(objects):",
        "    logical_start = np.array(LOGICAL_COORDINATES[ACTIVE_START_COORDINATE_ID], dtype=float)",
        "    logical_end = np.array(LOGICAL_COORDINATES[ACTIVE_END_COORDINATE_ID], dtype=float)",
        "    scene_start = np.array(objects[ACTIVE_OBJECT_ID].get_start(), dtype=float)",
        "    scene_end = np.array(objects[ACTIVE_OBJECT_ID].get_end(), dtype=float)",
        "    source_vector = logical_end - logical_start",
        "    target_vector = scene_end[:2] - scene_start[:2]",
        "    denominator = float(np.dot(source_vector, source_vector))",
        "    real = float(np.dot(target_vector, source_vector)) / denominator",
        "    imaginary = float(target_vector[1] * source_vector[0] - target_vector[0] * source_vector[1]) / denominator",
        "    matrix = np.array([[real, -imaginary], [imaginary, real]], dtype=float)",
        "    logical_scale = float(np.hypot(real, imaginary))",
        "    def to_scene(value):",
        "        mapped = scene_start[:2] + matrix @ (np.array(value, dtype=float) - logical_start)",
        "        return np.array([mapped[0], mapped[1], scene_start[2]], dtype=float)",
        "    return to_scene, logical_scale / PICTURE_SCALE",
        "",
        "",
        "def directed_angles(first, vertex, third):",
        "    first_vector = first - vertex",
        "    third_vector = third - vertex",
        "    start = float(np.arctan2(first_vector[1], first_vector[0]))",
        "    end = float(np.arctan2(third_vector[1], third_vector[0]))",
        "    while end < start:",
        "        end += TAU",
        "    return start, end",
        "",
        "",
        "def display_angle(start, end):",
        "    vector = end - start",
        "    angle = float(np.arctan2(vector[1], vector[0]))",
        "    if angle > PI / 2 or angle < -PI / 2:",
        "        angle += PI",
        "    return angle",
        "",
        "",
        "def match_geometry_style(mobject, template):",
        "    mobject.match_style(template)",
        "    for current, source in zip(mobject.get_family(), template.get_family()):",
        "        if hasattr(current, 'joint_type') and hasattr(source, 'joint_type'):",
        "            current.joint_type = source.joint_type",
        "        if hasattr(current, 'cap_style') and hasattr(source, 'cap_style'):",
        "            current.cap_style = source.cap_style",
        "    return mobject",
        "",
        "",
        "def native_dashed_line(start, end, *, on_pt, off_pt, scene_unit_per_cm, template):",
        "    vector = end - start",
        "    length = float(np.linalg.norm(vector))",
        "    if length <= 1e-12:",
        "        return VGroup()",
        "    direction = vector / length",
        "    on_length = max(on_pt * scene_unit_per_cm / TEX_POINTS_PER_CM, 1e-6)",
        "    off_length = max(off_pt * scene_unit_per_cm / TEX_POINTS_PER_CM, 0.0)",
        "    style_source = next((item for item in template.get_family() if len(item.points)), template)",
        "    dashes = []",
        "    cursor = 0.0",
        "    while cursor < length - 1e-9:",
        "        dash_end = min(cursor + on_length, length)",
        "        dash = Line(start + cursor * direction, start + dash_end * direction)",
        "        match_geometry_style(dash, style_source)",
        "        dashes.append(dash)",
        "        cursor += on_length + off_length",
        "    result = VGroup(*dashes)",
        "    result.set_z_index(template.z_index)",
        "    return result",
        "",
        "",
        "def install_geometry_updaters(objects, parameter):",
        "    missing = [object_id for object_id in DYNAMIC_OBJECT_IDS if object_id not in objects]",
        "    if missing:",
        "        raise RuntimeError('TikZ ShapeState is missing semantic objects: ' + ', '.join(missing))",
        "    active_updaters = [object_id for object_id in DYNAMIC_OBJECT_IDS if objects[object_id].updaters]",
        "    if active_updaters:",
        "        raise RuntimeError('TikZ ShapeState already has active updaters: ' + ', '.join(active_updaters))",
        "    entry_snapshots = {object_id: item.copy() for object_id, item in objects.items()}",
        "    entry_updaters = {object_id: tuple(item.updaters) for object_id, item in objects.items()}",
        "    to_scene, scene_unit_per_cm = scene_mapper(objects)",
        "    cache = {'parameter': None, 'coordinates': None}",
        "    def coordinates():",
        "        value = float(parameter.get_value())",
        "        if cache['coordinates'] is None or value != cache['parameter']:",
        "            cache['parameter'] = value",
        "            cache['coordinates'] = geometry_coordinates(value)",
        "        return cache['coordinates']",
        "    def point(name):",
        "        return to_scene(coordinates()[name])",
        "    geometry_state = {",
        "        'objects': objects,",
        "        'point': point,",
        "        'entry_snapshots': entry_snapshots,",
        "        'entry_updaters': entry_updaters,",
        "    }",
        "    try:",
    ]
    for index, binding in enumerate(bindings):
        lines.extend(
            "    " + line if line else line
            for line in _binding_lines(
                binding,
                object_specs[str(binding["object_id"])],
                index=index,
            )
        )
    lines.extend(
        [
            "    except Exception:",
            "        restore_geometry_objects(geometry_state)",
            "        raise",
            "    return geometry_state",
            "",
            "",
            "def restore_geometry_objects(state):",
            "    for object_id, item in state['objects'].items():",
            "        item.clear_updaters()",
            "        item.become(state['entry_snapshots'][object_id])",
            "        for updater in state['entry_updaters'][object_id]:",
            "            item.add_updater(updater)",
        ]
    )
    source_text = "\n".join(lines) + "\n"
    # Parse as Python without importing Manim; syntax is part of the Provider contract.
    compile(source_text, "<tikz-native-manim-source-2d>", "exec")
    return {
        "schema": NATIVE_MANIM_SOURCE_2D_SCHEMA,
        "sourceText": source_text,
        "sourceSha256": _stable_source_hash(source_text),
    }


__all__ = [
    "NATIVE_MANIM_SOURCE_2D_SCHEMA",
    "NativeManimCodegen2DError",
    "generate_native_manim_source_2d",
]
