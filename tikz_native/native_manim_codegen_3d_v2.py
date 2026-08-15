"""Generate multi-driver readable Manim helpers for one 3D TikZ picture.

Version 2 is additive to ``tikz-native-manim-source-3d/v1``.  It keeps the
same entry-alignment, projection, semantic-object and stable-occlusion code,
but replaces the single hinge tracker with an explicit driver tracker map.
Every point-on-segment driver is evaluated in logical 3D before projection;
the generated author source never drags a screen-space Dot directly.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import keyword
from math import isfinite
import re
from typing import Any, Mapping, Sequence

from .compiler import ObjectSpec, PictureSpec
from .native_manim_codegen_3d import (
    _SOURCE_HELPERS,
    _authored_coordinates,
    _literal,
    _object_payload,
    _source_hash,
    _style_payload,
)


NATIVE_MANIM_SOURCE_3D_V2_SCHEMA = "tikz-native-manim-source-3d/v2"
NATIVE_MANIM_AUTHORING_3D_SCHEMA = "tikz-native-manim-authoring-3d/v1"
POINT_ON_SEGMENT_CANDIDATE_ID_PREFIX = "point_on_segment:"


class NativeManimCodegen3DV2Error(ValueError):
    """The 3D semantic graph cannot be represented by the v2 source contract."""


def _dependency_names(dependency: Mapping[str, Any] | None) -> tuple[str, ...]:
    if dependency is None:
        return ()
    operation = str(dependency.get("operation") or "")
    fields = {
        "reference": ("coordinate",),
        "interpolation": ("start", "end"),
        "translation": ("base",),
        "projection": ("point", "line_start", "line_end"),
    }.get(operation, ())
    return tuple(
        str(dependency[field])
        for field in fields
        if isinstance(dependency.get(field), str) and dependency.get(field)
    )


def _affected_coordinates(
    picture: PictureSpec,
    roots: Sequence[str],
) -> set[str]:
    affected = set(roots)
    changed = True
    while changed:
        changed = False
        for name, dependency in picture.coordinate_dependencies.items():
            if name in affected:
                continue
            if any(parent in affected for parent in _dependency_names(dependency)):
                affected.add(name)
                changed = True
    return affected


def _segment_length(picture: PictureSpec, start: str, end: str) -> float | None:
    try:
        first = tuple(float(value) for value in picture.coordinates[start])
        second = tuple(float(value) for value in picture.coordinates[end])
    except (KeyError, TypeError, ValueError):
        return None
    if len(first) != 3 or len(second) != 3:
        return None
    return sum((second[index] - first[index]) ** 2 for index in range(3)) ** 0.5


def _derived_dependency_cycle(picture: PictureSpec) -> tuple[str, ...]:
    """Return one explicit v2-derived dependency cycle, if present."""

    derived_names = {
        name
        for name, dependency in picture.coordinate_dependencies.items()
        if str(dependency.get("operation") or "") in {"interpolation", "projection"}
    }
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> tuple[str, ...]:
        if name in visiting:
            start = visiting.index(name)
            return tuple((*visiting[start:], name))
        if name in visited:
            return ()
        visiting.append(name)
        dependency = picture.coordinate_dependencies.get(name)
        for parent in _dependency_names(dependency):
            if parent not in derived_names:
                continue
            cycle = visit(parent)
            if cycle:
                return cycle
        visiting.pop()
        visited.add(name)
        return ()

    for name in picture.coordinate_dependencies:
        if name not in derived_names:
            continue
        cycle = visit(name)
        if cycle:
            return cycle
    return ()


def point_on_segment_driver_candidates(picture: PictureSpec) -> list[dict[str, Any]]:
    """Return explicit, reviewable drivers without guessing from object IDs."""

    candidates: list[dict[str, Any]] = []
    dependency_cycle = _derived_dependency_cycle(picture)
    for coordinate_id, dependency in picture.coordinate_dependencies.items():
        if str(dependency.get("operation") or "") != "interpolation":
            continue
        candidate_id = f"{POINT_ON_SEGMENT_CANDIDATE_ID_PREFIX}{coordinate_id}"
        start = dependency.get("start")
        end = dependency.get("end")
        parameter = dependency.get("parameter")
        problems: list[str] = []
        if coordinate_id not in picture.coordinates:
            problems.append("driven coordinate is unknown")
        if not isinstance(start, str) or not start:
            problems.append("segment start is missing")
        if not isinstance(end, str) or not end:
            problems.append("segment end is missing")
        if isinstance(start, str) and isinstance(end, str) and start == end:
            problems.append("segment endpoints use the same coordinate")
        if isinstance(start, str) and start not in picture.coordinates:
            problems.append(f"unknown segment start {start!r}")
        if isinstance(end, str) and end not in picture.coordinates:
            problems.append(f"unknown segment end {end!r}")
        if (
            isinstance(parameter, bool)
            or not isinstance(parameter, (int, float))
            or not isfinite(float(parameter))
            or not 0.0 <= float(parameter) <= 1.0
        ):
            problems.append("authored segment parameter must be inside [0, 1]")
        if isinstance(start, str) and isinstance(end, str):
            length = _segment_length(picture, start, end)
            if length is not None and length <= 1e-12:
                problems.append("authored segment has zero length")
        if dependency_cycle:
            problems.append(
                "derived coordinate dependency cycle: "
                + " -> ".join(dependency_cycle)
            )
        affected = _affected_coordinates(picture, (coordinate_id,))
        candidates.append(
            {
                "candidateId": candidate_id,
                "driverId": candidate_id,
                "candidateKind": "geometry_driver",
                "driverType": "point_on_segment",
                "status": "blocked" if problems else "available",
                "coordinateId": coordinate_id,
                "segment": [
                    str(start) if isinstance(start, str) else "",
                    str(end) if isinstance(end, str) else "",
                ],
                "affectedCoordinates": [
                    name for name in picture.coordinates if name in affected
                ],
                "initial": {
                    "value": (
                        0.0
                        if isinstance(parameter, bool)
                        or not isinstance(parameter, (int, float))
                        else float(parameter)
                    ),
                    "unit": "ratio",
                },
                "suggestedRange": {
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "unit": "ratio",
                    "source": "explicit-segment-domain",
                },
                "reason": (
                    "; ".join(problems)
                    if problems
                    else (
                        "The compiler records one explicit coordinate as a ratio "
                        "on one named 3D segment."
                    )
                ),
            }
        )
    return candidates


def _safe_python_names(driver_ids: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for driver_id in driver_ids:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", driver_id).strip("_").lower()
        if not base:
            base = "geometry_driver"
        if base[0].isdigit():
            base = "driver_" + base
        if keyword.iskeyword(base):
            base += "_value"
        candidate = base
        if candidate in used:
            suffix = hashlib.sha256(driver_id.encode("utf-8")).hexdigest()[:8]
            candidate = f"{base}_{suffix}"
        if candidate in used:
            raise NativeManimCodegen3DV2Error(
                f"driver Python name is ambiguous for {driver_id!r}"
            )
        used.add(candidate)
        result[driver_id] = candidate
    return result


def _camera_authoring_spec(
    picture: PictureSpec,
    rig: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entry_matrix = picture.projection_3d.matrix if picture.projection_3d else ()

    def is_rotation(matrix: Sequence[Sequence[object]]) -> bool:
        try:
            rows = tuple(tuple(float(value) for value in row) for row in matrix)
        except (TypeError, ValueError):
            return False
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            return False
        for first_index, first in enumerate(rows):
            for second_index, second in enumerate(rows):
                dot = sum(left * right for left, right in zip(first, second))
                expected = 1.0 if first_index == second_index else 0.0
                if abs(dot - expected) > 1e-7:
                    return False
        determinant = (
            rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
        )
        return abs(determinant - 1.0) <= 1e-7

    orthogonal_modes = {"front", "side", "top", "isometric"}
    cameras = [
        item
        for item in rig.get("motionCandidates", [])
        if isinstance(item, Mapping)
        and item.get("candidateKind") == "camera_operation"
    ]
    modes = [
        {
            "mode": str(item["mode"]),
            "transitionTypes": [str(value) for value in item["transitionTypes"]],
            "orthogonal": str(item["mode"]) in orthogonal_modes,
        }
        for item in cameras
    ]
    return {"mode": "tikz", "orthogonal": is_rotation(entry_matrix)}, modes


def _authoring_drivers(
    picture: PictureSpec,
    rig: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, str]]:
    hinges = [
        item
        for item in rig.get("motionCandidates", [])
        if isinstance(item, Mapping)
        and item.get("candidateKind") == "geometry_driver"
        and item.get("driverType") == "hinge_fold"
        and item.get("status") != "blocked"
    ]
    selected = rig.get("selectedMotionCandidate")
    selected_id = (
        str(selected.get("candidateId"))
        if isinstance(selected, Mapping) and selected.get("candidateId")
        else None
    )
    if selected_id is not None:
        hinges = [item for item in hinges if item.get("candidateId") == selected_id]
    if len(hinges) != 1:
        raise NativeManimCodegen3DV2Error(
            "v2 readable source requires exactly one explicit hinge candidate"
        )
    hinge = hinges[0]
    point_candidates = [
        item
        for item in rig.get("motionCandidates", [])
        if isinstance(item, Mapping)
        and item.get("candidateKind") == "geometry_driver"
        and item.get("driverType") == "point_on_segment"
    ]
    blocked = [
        str(item.get("candidateId"))
        + " ("
        + str(item.get("reason") or "blocked")
        + ")"
        for item in point_candidates
        if item.get("status") == "blocked"
    ]
    if blocked:
        raise NativeManimCodegen3DV2Error(
            "blocked point-on-segment driver candidates: " + ", ".join(blocked)
        )

    motion = rig.get("motionSpecCore")
    motion_driver = motion.get("driver") if isinstance(motion, Mapping) else None
    hinge_range = (
        [float(value) for value in motion_driver["range"]]
        if isinstance(motion_driver, Mapping)
        and isinstance(motion_driver.get("range"), list)
        and len(motion_driver["range"]) == 2
        else [
            float(hinge["suggestedRange"]["minimum"]),
            float(hinge["suggestedRange"]["maximum"]),
        ]
    )
    hinge_driver_id = str(hinge.get("driverId") or hinge["candidateId"])
    driver_ids = [
        hinge_driver_id,
        *(str(item.get("driverId") or item["candidateId"]) for item in point_candidates),
    ]
    if len(driver_ids) != len(set(driver_ids)):
        raise NativeManimCodegen3DV2Error("3D driver IDs are ambiguous")
    python_names = _safe_python_names(driver_ids)
    authoring = [
        {
            "driverId": hinge_driver_id,
            "candidateId": str(hinge["candidateId"]),
            "type": "hinge_fold",
            "pythonName": python_names[hinge_driver_id],
            "initial": float(hinge["initial"]["value"]),
            "range": hinge_range,
            "unit": "radians",
            "axis": [str(value) for value in hinge["axis"]],
        }
    ]
    internal = [
        {
            **deepcopy(authoring[0]),
            "moving_coordinates": tuple(str(value) for value in hinge["movingCoordinates"]),
        }
    ]
    for item in point_candidates:
        driver_id = str(item.get("driverId") or item["candidateId"])
        record = {
            "driverId": driver_id,
            "candidateId": str(item["candidateId"]),
            "type": "point_on_segment",
            "pythonName": python_names[driver_id],
            "initial": float(item["initial"]["value"]),
            "range": [
                float(item["suggestedRange"]["minimum"]),
                float(item["suggestedRange"]["maximum"]),
            ],
            "unit": "ratio",
            "coordinateId": str(item["coordinateId"]),
            "segment": [str(value) for value in item["segment"]],
        }
        authoring.append(record)
        internal.append(deepcopy(record))
    axis = tuple(str(value) for value in hinge["axis"])
    if len(axis) != 2:
        raise NativeManimCodegen3DV2Error("hinge axis must name exactly two coordinates")
    return authoring, internal, (axis[0], axis[1])


def _derived_relations(
    picture: PictureSpec,
    internal_drivers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    point_driver_ids = {
        str(item["coordinateId"]): str(item["driverId"])
        for item in internal_drivers
        if item["type"] == "point_on_segment"
    }
    hinge_roots = {
        str(name)
        for item in internal_drivers
        if item["type"] == "hinge_fold"
        for name in item["moving_coordinates"]
    }
    affected = _affected_coordinates(
        picture,
        (*hinge_roots, *point_driver_ids),
    )
    relations: dict[str, dict[str, Any]] = {}
    for name, dependency in picture.coordinate_dependencies.items():
        operation = str(dependency.get("operation") or "")
        if operation == "interpolation":
            start = dependency.get("start")
            end = dependency.get("end")
            parameter = dependency.get("parameter")
            if not isinstance(start, str) or not start:
                raise NativeManimCodegen3DV2Error(
                    f"point-on-segment coordinate {name!r} has no named start"
                )
            if not isinstance(end, str) or not end:
                raise NativeManimCodegen3DV2Error(
                    f"point-on-segment coordinate {name!r} has no named end"
                )
            if start not in picture.coordinates or end not in picture.coordinates:
                raise NativeManimCodegen3DV2Error(
                    f"point-on-segment coordinate {name!r} references an unknown segment"
                )
            if (
                isinstance(parameter, bool)
                or not isinstance(parameter, (int, float))
                or not isfinite(float(parameter))
                or not 0.0 <= float(parameter) <= 1.0
            ):
                raise NativeManimCodegen3DV2Error(
                    f"point-on-segment coordinate {name!r} has an illegal authored parameter"
                )
            driver_id = point_driver_ids.get(name)
            if driver_id is None:
                raise NativeManimCodegen3DV2Error(
                    f"point-on-segment coordinate {name!r} has no explicit v2 driver"
                )
            relations[name] = {
                "name": name,
                "type": "point_on_segment",
                "start": start,
                "end": end,
                "driver_id": driver_id,
            }
        elif operation == "projection":
            point = dependency.get("point")
            line_start = dependency.get("line_start")
            line_end = dependency.get("line_end")
            if not all(
                isinstance(value, str) and value
                for value in (point, line_start, line_end)
            ):
                raise NativeManimCodegen3DV2Error(
                    f"projected coordinate {name!r} has incomplete named inputs"
                )
            unknown = [
                str(value)
                for value in (point, line_start, line_end)
                if str(value) not in picture.coordinates
            ]
            if unknown:
                raise NativeManimCodegen3DV2Error(
                    f"projected coordinate {name!r} references unknown coordinates: "
                    + ", ".join(unknown)
                )
            relations[name] = {
                "name": name,
                "type": "project_point_to_line",
                "point": str(point),
                "line_start": str(line_start),
                "line_end": str(line_end),
            }
        elif name in affected:
            raise NativeManimCodegen3DV2Error(
                f"driver-dependent coordinate {name!r} uses unsupported operation {operation!r}"
            )

    unresolved = dict(relations)
    available = set(picture.coordinates) - set(unresolved)
    ordered: list[dict[str, Any]] = []
    while unresolved:
        progressed = False
        for name, relation in tuple(unresolved.items()):
            dependencies = (
                (relation["start"], relation["end"])
                if relation["type"] == "point_on_segment"
                else (
                    relation["point"],
                    relation["line_start"],
                    relation["line_end"],
                )
            )
            if any(value not in available for value in dependencies):
                continue
            ordered.append(relation)
            available.add(name)
            unresolved.pop(name)
            progressed = True
        if not progressed:
            raise NativeManimCodegen3DV2Error(
                "derived coordinate dependency cycle: "
                + ", ".join(sorted(unresolved))
            )
    return ordered


def _object_point_names(item: ObjectSpec) -> tuple[str, ...]:
    geometry = item.geometry
    if item.kind in {"line", "arrow", "path_label"}:
        raw = (geometry.get("start_name"), geometry.get("end_name"))
    elif item.kind == "polygon":
        values = geometry.get("point_names")
        raw = tuple(values) if isinstance(values, list) else ()
    elif item.kind == "dot":
        raw = (geometry.get("center_name"),)
    elif item.kind == "label":
        raw = (geometry.get("at_name"),)
    else:
        raw = ()
    return tuple(str(value) for value in raw if isinstance(value, str) and value)


def _object_driver_ids(
    picture: PictureSpec,
    internal_drivers: Sequence[Mapping[str, Any]],
    derived: Sequence[Mapping[str, Any]],
    excluded_object_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    coordinate_drivers: dict[str, set[str]] = {
        name: set() for name in picture.coordinates
    }
    driver_order = [str(item["driverId"]) for item in internal_drivers]
    for driver in internal_drivers:
        driver_id = str(driver["driverId"])
        if driver["type"] == "hinge_fold":
            for name in driver["moving_coordinates"]:
                if name not in coordinate_drivers:
                    raise NativeManimCodegen3DV2Error(
                        f"hinge driver references unknown moving coordinate {name!r}"
                    )
                coordinate_drivers[str(name)].add(driver_id)
    for relation in derived:
        if relation["type"] == "point_on_segment":
            dependencies = (relation["start"], relation["end"])
            inherited = set().union(
                *(coordinate_drivers[name] for name in dependencies)
            )
            inherited.add(str(relation["driver_id"]))
        else:
            dependencies = (
                relation["point"],
                relation["line_start"],
                relation["line_end"],
            )
            inherited = set().union(
                *(coordinate_drivers[name] for name in dependencies)
            )
        coordinate_drivers[str(relation["name"])] = inherited

    excluded = set(excluded_object_ids)
    result: dict[str, tuple[str, ...]] = {}
    for item in picture.objects:
        dependencies = set().union(
            *(coordinate_drivers[name] for name in _object_point_names(item)),
        ) if _object_point_names(item) else set()
        if item.id in excluded:
            dependencies.clear()
        result[item.id] = tuple(
            driver_id for driver_id in driver_order if driver_id in dependencies
        )
    return result


def _function_span(source: str, function_name: str) -> tuple[int, int]:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1 or matches[0].end_lineno is None:
        raise NativeManimCodegen3DV2Error(
            f"v1 source template has no unique function {function_name!r}"
        )
    node = matches[0]
    return node.lineno - 1, node.end_lineno


def _replace_function(source: str, function_name: str, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    start, end = _function_span(source, function_name)
    normalized = replacement.strip("\n") + "\n"
    lines[start:end] = [normalized]
    return "".join(lines)


def _replace_exact(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise NativeManimCodegen3DV2Error(
            f"v1 source template drifted at {label}: expected one match, got {count}"
        )
    return source.replace(old, new)


_GEOMETRY_COORDINATES_V2 = r'''
def geometry_coordinates_3d(driver_values):
    """Evaluate every logical 3D coordinate from the complete driver value map."""
    expected = set(DRIVER_SPECS)
    actual = set(driver_values)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            "driver value map mismatch; missing=" + repr(missing) + ", unknown=" + repr(unknown)
        )
    values = {}
    for driver_id, spec in DRIVER_SPECS.items():
        value = float(driver_values[driver_id])
        minimum, maximum = spec["range"]
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{driver_id}={value} is outside [{minimum}, {maximum}]"
            )
        values[driver_id] = value

    derived_ids = {item["name"] for item in DERIVED_COORDINATES}
    coordinates = {
        name: np.array(point, dtype=float)
        for name, point in AUTHORED_COORDINATES.items()
        if name not in derived_ids
    }
    for driver_id in DRIVER_ORDER:
        spec = DRIVER_SPECS[driver_id]
        if spec["type"] != "hinge_fold":
            continue
        axis_start = coordinates[spec["axis"][0]]
        axis_end = coordinates[spec["axis"][1]]
        delta = values[driver_id] - spec["initial"]
        for name in spec["moving_coordinates"]:
            if name not in AUTHORED_COORDINATES or name in derived_ids:
                raise ValueError(
                    f"hinge moving coordinate {name!r} must be an authored base coordinate"
                )
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
                    values[relation["driver_id"]],
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
'''


def _v2_source_helpers() -> str:
    helpers = _replace_function(
        _SOURCE_HELPERS,
        "geometry_coordinates_3d",
        _GEOMETRY_COORDINATES_V2,
    )
    install_start, install_end = _function_span(
        helpers,
        "install_geometry_3d_updaters",
    )
    lines = helpers.splitlines(keepends=True)
    install = "".join(lines[install_start:install_end])
    install = _replace_exact(
        install,
        "def install_geometry_3d_updaters(shape, objects, hinge_angle, camera_progress):",
        "def install_geometry_3d_updaters(shape, objects, driver_trackers, camera_progress):",
        "v2 install signature",
    )
    install = _replace_exact(
        install,
        '''    if shape.updaters or any(objects[object_id].updaters for object_id in OBJECT_SPECS):
        raise RuntimeError("TikZ ShapeState already has active updaters")
''',
        '''    expected_driver_ids = set(DRIVER_SPECS)
    actual_driver_ids = set(driver_trackers)
    if actual_driver_ids != expected_driver_ids:
        raise RuntimeError(
            "driver tracker map mismatch; missing="
            + repr(sorted(expected_driver_ids - actual_driver_ids))
            + ", unknown="
            + repr(sorted(actual_driver_ids - expected_driver_ids))
        )
    for driver_id, tracker in driver_trackers.items():
        if not callable(getattr(tracker, "get_value", None)) or not callable(
            getattr(tracker, "set_value", None)
        ):
            raise RuntimeError(f"driver tracker {driver_id!r} must behave like ValueTracker")
    if shape.updaters or any(objects[object_id].updaters for object_id in OBJECT_SPECS):
        raise RuntimeError("TikZ ShapeState already has active updaters")
''',
        "v2 tracker validation",
    )
    install = _replace_exact(
        install,
        '        "hinge_angle": hinge_angle,\n',
        '        "driver_trackers": dict(driver_trackers),\n',
        "v2 state trackers",
    )
    install = _replace_exact(
        install,
        '        "coordinate_cache": {"parameter": None, "coordinates": None},\n',
        '        "coordinate_cache": {"values": None, "coordinates": None},\n',
        "v2 coordinate cache",
    )
    install = _replace_exact(
        install,
        '''    camera_progress.set_value(CAMERA_PROGRESS_INITIAL)

    def coordinates():
        value = float(hinge_angle.get_value())
        cache = state["coordinate_cache"]
        if cache["coordinates"] is None or cache["parameter"] != value:
            cache["parameter"] = value
            cache["coordinates"] = geometry_coordinates_3d(value)
        return cache["coordinates"]
''',
        '''    for driver_id, tracker in driver_trackers.items():
        tracker.set_value(DRIVER_INITIAL_VALUES[driver_id])
    camera_progress.set_value(CAMERA_PROGRESS_INITIAL)

    def driver_values():
        return {
            driver_id: float(driver_trackers[driver_id].get_value())
            for driver_id in DRIVER_ORDER
        }

    def coordinates():
        values = driver_values()
        cache_key = tuple(values[driver_id] for driver_id in DRIVER_ORDER)
        cache = state["coordinate_cache"]
        if cache["coordinates"] is None or cache["values"] != cache_key:
            cache["values"] = cache_key
            cache["coordinates"] = geometry_coordinates_3d(values)
        return cache["coordinates"]
''',
        "v2 coordinate provider",
    )
    install = _replace_exact(
        install,
        '''    state["coordinates"] = coordinates
    state["project_scene"] = project_scene
''',
        '''    state["driver_values"] = driver_values
    state["coordinates"] = coordinates
    state["project_scene"] = project_scene
''',
        "v2 state coordinate API",
    )
    install = _replace_exact(
        install,
        "            follows_driver = object_id in DYNAMIC_OBJECT_IDS\n",
        "            follows_driver = bool(OBJECT_DRIVER_IDS.get(object_id))\n",
        "v2 object driver classification",
    )
    install = _replace_exact(
        install,
        '''        axis_origin_norm = float(np.linalg.norm(AUTHORED_COORDINATES[axis_start_name]))
        for name in HINGE_MOVING_COORDINATE_IDS:
            coordinate_bounds[name] = axis_origin_norm + float(
                np.linalg.norm(
                    np.asarray(AUTHORED_COORDINATES[name])
                    - np.asarray(AUTHORED_COORDINATES[axis_start_name])
                )
            )
''',
        '''        for driver_id in DRIVER_ORDER:
            driver = DRIVER_SPECS[driver_id]
            if driver["type"] != "hinge_fold":
                continue
            hinge_origin = driver["axis"][0]
            axis_origin_norm = float(np.linalg.norm(AUTHORED_COORDINATES[hinge_origin]))
            for name in driver["moving_coordinates"]:
                coordinate_bounds[name] = axis_origin_norm + float(
                    np.linalg.norm(
                        np.asarray(AUTHORED_COORDINATES[name])
                        - np.asarray(AUTHORED_COORDINATES[hinge_origin])
                    )
                )
''',
        "v2 coordinate bounds",
    )
    lines[install_start:install_end] = [install]
    helpers = "".join(lines)

    restore_start, restore_end = _function_span(
        helpers,
        "restore_geometry_3d_objects",
    )
    lines = helpers.splitlines(keepends=True)
    restore = "".join(lines[restore_start:restore_end])
    restore = _replace_exact(
        restore,
        "    state[\"hinge_angle\"].set_value(HINGE_ANGLE_INITIAL)\n",
        '''    for driver_id, tracker in state["driver_trackers"].items():
        tracker.set_value(DRIVER_INITIAL_VALUES[driver_id])
''',
        "v2 restore trackers",
    )
    lines[restore_start:restore_end] = [restore]
    helpers = "".join(lines)
    compile(helpers, "<tikz-native-manim-source-3d-v2-helpers>", "exec")
    return helpers


def generate_native_manim_source_3d_v2(
    picture: PictureSpec,
    rig: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a self-contained multi-driver Manim definition source."""

    if picture.dimension != 3 or picture.projection_3d is None:
        raise NativeManimCodegen3DV2Error(
            "native Manim 3D v2 source requires a 3D picture"
        )
    if rig.get("status") not in {"needs_selection", "ready"}:
        raise NativeManimCodegen3DV2Error(
            "native Manim 3D v2 source requires a reviewable rig"
        )

    authoring_drivers, internal_drivers, anchor_axis = _authoring_drivers(
        picture,
        rig,
    )
    derived = _derived_relations(picture, internal_drivers)
    object_specs = {item.id: _object_payload(item) for item in picture.objects}
    object_driver_ids = _object_driver_ids(
        picture,
        internal_drivers,
        derived,
        tuple(str(value) for value in rig.get("excludedObjectIds", [])),
    )
    entry_camera, camera_modes = _camera_authoring_spec(picture, rig)
    authoring_spec = {
        "schema": NATIVE_MANIM_AUTHORING_3D_SCHEMA,
        "drivers": authoring_drivers,
        "entryCamera": entry_camera,
        "cameraModes": camera_modes,
        "endPolicy": "restore_entry",
    }

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
    root_two = 2.0**0.5
    root_six = 6.0**0.5
    root_three = 3.0**0.5
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
    driver_specs = {
        str(item["driverId"]): {
            "type": str(item["type"]),
            "initial": float(item["initial"]),
            "range": tuple(float(value) for value in item["range"]),
            **(
                {
                    "axis": tuple(str(value) for value in item["axis"]),
                    "moving_coordinates": tuple(
                        str(value) for value in item["moving_coordinates"]
                    ),
                }
                if item["type"] == "hinge_fold"
                else {
                    "coordinate_id": str(item["coordinateId"]),
                    "segment": tuple(str(value) for value in item["segment"]),
                }
            ),
        }
        for item in internal_drivers
    }
    driver_order = tuple(str(item["driverId"]) for item in internal_drivers)
    source_lines = [
        "import numpy as np",
        "from manim import *",
        "",
        "# ===== Provider 展开的三维多驱动几何、局部相机与动态遮挡 =====",
        f"DRIVER_ORDER = {_literal(driver_order)}",
        f"DRIVER_SPECS = {_literal(driver_specs)}",
        (
            "DRIVER_INITIAL_VALUES = {driver_id: spec['initial'] "
            "for driver_id, spec in DRIVER_SPECS.items()}"
        ),
        f"HINGE_AXIS_COORDINATE_IDS = {_literal(anchor_axis)}",
        "CAMERA_PROGRESS_INITIAL = 1.0",
        f"AUTHORED_COORDINATES = {_literal(_authored_coordinates(picture))}",
        f"DERIVED_COORDINATES = {_literal(derived)}",
        f"OBJECT_SPECS = {_literal(object_specs)}",
        f"OBJECT_DRIVER_IDS = {_literal(object_driver_ids)}",
        f"OCCLUSION_RELATIONS = {_literal(occlusion_payload)}",
        f"OCCLUSION_FRAGMENT_OBJECT_IDS = frozenset({_literal(relation_members)})",
        "ENTRY_PROJECTION_MATRIX = np.array("
        + _literal(
            tuple(
                tuple(float(value) for value in row)
                for row in picture.projection_3d.matrix
            )
        )
        + ", dtype=float)",
        "CAMERA_PRESET_MATRICES = {name: np.array(value, dtype=float) for name, value in "
        + _literal(presets)
        + ".items()}",
        f"PICTURE_SCALE = {float(picture.scale)!r}",
        "TEX_POINTS_PER_CM = 72.27 / 2.54",
    ]
    source_text = "\n".join(source_lines) + _v2_source_helpers() + "\n"
    compile(source_text, "<tikz-native-manim-source-3d-v2>", "exec")
    return {
        "schema": NATIVE_MANIM_SOURCE_3D_V2_SCHEMA,
        "sourceText": source_text,
        "sourceSha256": _source_hash(source_text),
        "authoringSpec": authoring_spec,
    }


__all__ = [
    "NATIVE_MANIM_AUTHORING_3D_SCHEMA",
    "NATIVE_MANIM_SOURCE_3D_V2_SCHEMA",
    "NativeManimCodegen3DV2Error",
    "POINT_ON_SEGMENT_CANDIDATE_ID_PREFIX",
    "generate_native_manim_source_3d_v2",
    "point_on_segment_driver_candidates",
]
