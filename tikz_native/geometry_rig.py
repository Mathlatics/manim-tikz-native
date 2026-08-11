from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from math import atan2, cos, isclose, isfinite, pi, sin, sqrt
from typing import Any, Iterable, Mapping, Sequence

from .compiler import ObjectSpec, PictureSpec
from .native_manim_codegen_2d import (
    NativeManimCodegen2DError,
    generate_native_manim_source_2d,
)


GEOMETRY_RIG_SCHEMA = "tikz-native-geometry-rig/v1"
GEOMETRY_SEMANTIC_MODEL_SCHEMA = "tikz-native-geometry-semantic-model/v1"
MOTION_SCHEMA = "tikz-native-motion/v1"

RIG_STATUS_READY = "ready"
RIG_STATUS_NEEDS_SELECTION = "needs_selection"
RIG_STATUS_BLOCKED = "blocked"

_SUPPORTED_BINDINGS = {
    "line": "line",
    "arrow": "line",
    "dot": "dot",
    "polygon": "polygon",
    "label": "label",
    "path_label": "path_label",
    "angle": "angle",
    "angle_label": "angle_label",
    "right_angle": "right_angle",
}


class GeometryRigError(ValueError):
    """Raised when a rig request is malformed rather than merely unsupported."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not isfinite(value):
            raise GeometryRigError("semantic geometry contains a non-finite number")
        return 0.0 if value == 0 else value
    return value


def semantic_model_payload(picture: PictureSpec) -> dict[str, Any]:
    """Return a portable projection of the geometry that owns rig identity."""

    return {
        "schema": GEOMETRY_SEMANTIC_MODEL_SCHEMA,
        "pictureIndex": picture.index,
        "dimension": picture.dimension,
        "coordinates": [
            {
                "coordinateId": name,
                "value": list(value),
                "dependency": picture.coordinate_dependencies.get(name),
            }
            for name, value in sorted(picture.coordinates.items())
        ],
        "namedPaths": [
            {
                "pathId": name,
                "kind": path.kind,
                "geometry": path.geometry,
            }
            for name, path in sorted(picture.named_paths.items())
        ],
        "intersections": [
            {
                "intersectionIndex": index,
                "pathA": relation.path_a,
                "pathB": relation.path_b,
                "sortBy": relation.sort_by,
                "coordinateIds": list(relation.coordinate_names),
            }
            for index, relation in enumerate(picture.intersections)
        ],
        "objects": [
            {
                "objectId": item.id,
                "kind": item.kind,
                "geometry": item.geometry,
            }
            for item in sorted(picture.objects, key=lambda candidate: candidate.id)
        ],
    }


def semantic_model_hash(picture: PictureSpec) -> str:
    payload = json.dumps(
        _canonical_value(semantic_model_payload(picture)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _diagnostic(
    severity: str,
    code: str,
    message: str,
    **context: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    result.update({key: value for key, value in context.items() if value})
    return result


def _dependency_names(dependency: Mapping[str, Any] | None) -> tuple[str, ...]:
    if dependency is None:
        return ()
    operation = dependency.get("operation")
    keys = {
        "reference": ("coordinate",),
        "interpolation": ("start", "end"),
        "translation": ("base",),
        "projection": ("line_start", "point", "line_end"),
    }.get(str(operation), ())
    return tuple(
        str(dependency[key])
        for key in keys
        if isinstance(dependency.get(key), str) and dependency.get(key)
    )


def _point_equal(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        isclose(float(a), float(b), abs_tol=1e-9, rel_tol=0.0)
        for a, b in zip(left, right)
    )


def _id_coordinate_tokens(object_id: str, prefix: str) -> list[str]:
    stem = object_id[len(prefix) + 1 :] if object_id.startswith(prefix + ".") else ""
    parts = stem.split(".") if stem else []
    if parts and parts[-1].isdigit():
        parts.pop()
    return parts


def _resolve_point_name(
    picture: PictureSpec,
    point: Sequence[float],
    *,
    preferred: str | None = None,
) -> str | None:
    if preferred in picture.coordinates and _point_equal(
        picture.coordinates[str(preferred)], point
    ):
        return str(preferred)
    matches = [
        name
        for name, value in picture.coordinates.items()
        if _point_equal(value, point)
    ]
    return matches[0] if len(matches) == 1 else None


def _object_point_names(
    picture: PictureSpec,
    item: ObjectSpec,
) -> tuple[str, ...] | None:
    geometry = item.geometry
    if item.kind in {"line", "arrow"}:
        names = (geometry.get("start_name"), geometry.get("end_name"))
        return tuple(str(name) for name in names) if all(names) else None
    if item.kind == "path_label":
        names = (geometry.get("start_name"), geometry.get("end_name"))
        return tuple(str(name) for name in names) if all(names) else None
    if item.kind == "label":
        name = geometry.get("at_name")
        return (str(name),) if name else None
    if item.kind in {"angle", "angle_label", "right_angle"}:
        names = (
            geometry.get("first_name"),
            geometry.get("vertex_name"),
            geometry.get("third_name"),
        )
        return tuple(str(name) for name in names) if all(names) else None
    if item.kind == "dot":
        center = geometry.get("center")
        if not isinstance(center, (list, tuple)):
            return None
        tokens = _id_coordinate_tokens(item.id, "dot")
        preferred = tokens[0] if tokens else None
        name = _resolve_point_name(picture, center, preferred=preferred)
        return (name,) if name else None
    if item.kind in {"circle", "ellipse"}:
        center = geometry.get("center")
        if not isinstance(center, (list, tuple)):
            return None
        tokens = _id_coordinate_tokens(item.id, item.kind)
        preferred = tokens[0] if tokens else None
        name = _resolve_point_name(picture, center, preferred=preferred)
        return (name,) if name else None
    if item.kind == "polygon":
        points = geometry.get("points")
        if not isinstance(points, (list, tuple)):
            return None
        tokens = _id_coordinate_tokens(item.id, "fill")
        names: list[str] = []
        for index, point in enumerate(points):
            if not isinstance(point, (list, tuple)):
                return None
            preferred = tokens[index] if index < len(tokens) else None
            name = _resolve_point_name(picture, point, preferred=preferred)
            if name is None:
                return None
            names.append(name)
        return tuple(names)
    return None


def _matching_named_lines(
    picture: PictureSpec,
    active: ObjectSpec,
) -> list[str]:
    names = _object_point_names(picture, active)
    if names is None or len(names) != 2:
        return []
    start_name, end_name = names
    matches: list[str] = []
    for name, path in picture.named_paths.items():
        if path.kind != "line":
            continue
        endpoints = (
            path.geometry.get("start_name"),
            path.geometry.get("end_name"),
        )
        if endpoints in {(start_name, end_name), (end_name, start_name)}:
            matches.append(name)
    return sorted(matches)


def _pivot_preference(
    picture: PictureSpec,
    start_name: str,
    end_name: str,
    coordinate_name: str,
) -> tuple[int, str]:
    start_dependency = picture.coordinate_dependencies.get(start_name, {})
    end_dependency = picture.coordinate_dependencies.get(end_name, {})
    score = 0
    for key, points in (("start", 120), ("base", 110), ("end", 90)):
        if (
            start_dependency.get(key) == coordinate_name
            and end_dependency.get(key) == coordinate_name
        ):
            score = max(score, points)
    lowered = coordinate_name.lower()
    if lowered in {"f", "focus", "pivot"} or lowered.startswith("focus"):
        score += 20
    return (-score, coordinate_name)


def _pivot_candidates(
    picture: PictureSpec,
    active_path: str,
) -> list[dict[str, Any]]:
    path = picture.named_paths[active_path]
    start_name = path.geometry.get("start_name")
    end_name = path.geometry.get("end_name")
    if not isinstance(start_name, str) or not isinstance(end_name, str):
        return []
    start = picture.coordinates[start_name]
    end = picture.coordinates[end_name]
    direction = (float(end[0] - start[0]), float(end[1] - start[1]))
    length = sqrt(direction[0] ** 2 + direction[1] ** 2)
    if length <= 1e-12:
        return []
    unit = (direction[0] / length, direction[1] / length)
    candidates: list[str] = []
    for name, point in picture.coordinates.items():
        if len(point) != 2:
            continue
        # A pivot is the fixed origin of the driver's motion.  Coordinates
        # produced by interpolation, translation, projection, reference, or
        # intersection are consequences of another relation and must never be
        # promoted to that origin merely because their initial point happens
        # to lie on the active line.
        if name in picture.coordinate_dependencies:
            continue
        if name in {start_name, end_name}:
            continue
        relative = (float(point[0] - start[0]), float(point[1] - start[1]))
        perpendicular = relative[0] * unit[1] - relative[1] * unit[0]
        position = relative[0] * unit[0] + relative[1] * unit[1]
        if abs(perpendicular) <= 1e-9 and 1e-9 < position < length - 1e-9:
            candidates.append(name)
    candidates.sort(
        key=lambda name: _pivot_preference(
            picture, start_name, end_name, name
        )
    )
    return [
        {
            "coordinateId": name,
            "status": "recommended" if index == 0 else "available",
            "reason": (
                "The coordinate is the shared construction origin of both line endpoints."
                if index == 0
                else "The coordinate lies strictly inside the oriented active line."
            ),
        }
        for index, name in enumerate(candidates)
    ]


def _line_angle(picture: PictureSpec, active_path: str) -> float:
    path = picture.named_paths[active_path]
    start = path.geometry["start"]
    end = path.geometry["end"]
    return atan2(float(end[1] - start[1]), float(end[0] - start[0]))


def _line_ellipse_discriminant(
    angle: float,
    *,
    pivot: Sequence[float],
    ellipse: Mapping[str, Any],
) -> float:
    center = ellipse["center"]
    rx = float(ellipse["rx"])
    ry = float(ellipse["ry"])
    direction = (cos(angle), sin(angle))
    offset = (float(pivot[0] - center[0]), float(pivot[1] - center[1]))
    quadratic_a = direction[0] ** 2 / rx**2 + direction[1] ** 2 / ry**2
    quadratic_b = 2 * (
        offset[0] * direction[0] / rx**2
        + offset[1] * direction[1] / ry**2
    )
    quadratic_c = offset[0] ** 2 / rx**2 + offset[1] ** 2 / ry**2 - 1
    return quadratic_b**2 - 4 * quadratic_a * quadratic_c


def _suggested_range(
    picture: PictureSpec,
    relation_index: int,
    pivot_name: str,
    initial: float,
) -> tuple[float, float] | None:
    relation = picture.intersections[relation_index]
    ellipse_name = (
        relation.path_b if relation.path_a == relation.sort_by else relation.path_a
    )
    ellipse = picture.named_paths.get(ellipse_name)
    if ellipse is None or ellipse.kind != "ellipse":
        return None
    pivot = picture.coordinates[pivot_name]
    half_width = pi / 12
    for _ in range(9):
        minimum = initial - half_width
        maximum = initial + half_width
        samples = [minimum + (maximum - minimum) * index / 16 for index in range(17)]
        if all(
            _line_ellipse_discriminant(
                value,
                pivot=pivot,
                ellipse=ellipse.geometry,
            )
            > 1e-10
            for value in samples
        ):
            return (minimum, maximum)
        half_width /= 2
    return None


def _candidate_id(active_path: str, intersection_index: int) -> str:
    return f"rotate_named_line:{active_path}:{intersection_index}"


def _driver_candidates(
    picture: PictureSpec,
    active: ObjectSpec,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for active_path in _matching_named_lines(picture, active):
        pivots = _pivot_candidates(picture, active_path)
        for relation_index, relation in enumerate(picture.intersections):
            if relation.sort_by != active_path:
                continue
            status = "available"
            if len(relation.coordinate_names) != 2 or not pivots:
                status = "blocked"
            initial = _line_angle(picture, active_path)
            suggested = (
                _suggested_range(
                    picture,
                    relation_index,
                    pivots[0]["coordinateId"],
                    initial,
                )
                if pivots
                else None
            )
            if suggested is None:
                status = "blocked"
            candidates.append(
                {
                    "candidateId": _candidate_id(active_path, relation_index),
                    "status": status,
                    "driverType": "rotate_named_line",
                    "activeObjectId": active.id,
                    "activePath": active_path,
                    "intersectionIndex": relation_index,
                    "intersectionCoordinates": list(relation.coordinate_names),
                    "pivotCandidates": pivots,
                    "initial": {"value": initial, "unit": "radians"},
                    "suggestedRange": (
                        {
                            "minimum": suggested[0],
                            "maximum": suggested[1],
                            "unit": "radians",
                            "source": "initial-window",
                        }
                        if suggested is not None
                        else None
                    ),
                }
            )
    usable = [item for item in candidates if item["status"] != "blocked"]
    if usable:
        usable[0]["status"] = "recommended"
    return candidates


def _active_object_candidates(picture: PictureSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in picture.objects:
        if item.kind not in {"line", "arrow"}:
            continue
        candidates = _driver_candidates(picture, item)
        usable = [candidate for candidate in candidates if candidate["status"] != "blocked"]
        records.append(
            {
                "objectId": item.id,
                "type": item.kind,
                "status": "available" if usable else "unsupported",
                "driverTypes": ["rotate_named_line"] if usable else [],
                "candidateIds": [candidate["candidateId"] for candidate in usable],
                "reason": (
                    "The object is backed by an oriented named line and a supported intersection."
                    if usable
                    else "The line has no supported named-path intersection driver."
                ),
            }
        )
    return records


def _affected_coordinates(
    picture: PictureSpec,
    *,
    active_path: str,
    intersection_index: int,
) -> tuple[set[str], list[int]]:
    path = picture.named_paths[active_path]
    affected = {
        str(path.geometry["start_name"]),
        str(path.geometry["end_name"]),
        *picture.intersections[intersection_index].coordinate_names,
    }
    unselected: list[int] = []
    for index, relation in enumerate(picture.intersections):
        if index != intersection_index and active_path in {
            relation.path_a,
            relation.path_b,
            relation.sort_by,
        }:
            unselected.append(index)
    changed = True
    while changed:
        changed = False
        for name, dependency in picture.coordinate_dependencies.items():
            if name in affected:
                continue
            if any(parent in affected for parent in _dependency_names(dependency)):
                affected.add(name)
                changed = True
    return affected, unselected


def _intersection_dependencies(
    picture: PictureSpec,
    relation_index: int,
) -> tuple[str, ...]:
    relation = picture.intersections[relation_index]
    names: list[str] = []
    for path_name in (relation.path_a, relation.path_b):
        path = picture.named_paths[path_name]
        for key in ("start_name", "end_name", "center_name"):
            value = path.geometry.get(key)
            if isinstance(value, str) and value not in names:
                names.append(value)
    return tuple(names)


def _coordinate_records(
    picture: PictureSpec,
    *,
    active_path: str,
    intersection_index: int,
    affected: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    path = picture.named_paths[active_path]
    endpoints = {
        str(path.geometry["start_name"]),
        str(path.geometry["end_name"]),
    }
    intersections = set(
        picture.intersections[intersection_index].coordinate_names
    )
    records: list[dict[str, Any]] = []
    fixed: list[str] = []
    for name, value in picture.coordinates.items():
        dependency = picture.coordinate_dependencies.get(name)
        if name in endpoints:
            classification = "driver_endpoint"
        elif name in intersections:
            classification = "intersection"
        elif dependency is not None:
            classification = "derived"
        else:
            classification = "fixed"
        if not name in affected:
            fixed.append(name)
        depends_on = list(_dependency_names(dependency))
        if name in intersections:
            depends_on = list(
                _intersection_dependencies(picture, intersection_index)
            )
        records.append(
            {
                "coordinateId": name,
                "value": list(value),
                "classification": classification,
                "dependency": dependency,
                "dependsOn": depends_on,
                "affectedByDriver": name in affected,
            }
        )
    return records, fixed


def _intersection_records(
    picture: PictureSpec,
    *,
    active_path: str,
    intersection_index: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, relation in enumerate(picture.intersections):
        if index == intersection_index:
            status = "driven"
        elif active_path in {relation.path_a, relation.path_b, relation.sort_by}:
            status = "unsupported"
        else:
            status = "fixed"
        records.append(
            {
                "intersectionIndex": index,
                "pathA": relation.path_a,
                "pathB": relation.path_b,
                "sortBy": relation.sort_by,
                "coordinateIds": list(relation.coordinate_names),
                "status": status,
            }
        )
    return records


def _selected_ids(selection: Mapping[str, Any], key: str) -> set[str]:
    raw = selection.get(key, [])
    if raw is None:
        return set()
    if not isinstance(raw, list) or any(
        not isinstance(value, str) or not value.strip() for value in raw
    ):
        raise GeometryRigError(f"selection.{key} must be an array of object IDs")
    if len(set(raw)) != len(raw):
        raise GeometryRigError(f"selection.{key} must not contain duplicates")
    return set(raw)


def _binding_records(
    picture: PictureSpec,
    *,
    affected: set[str],
    active_object_id: str,
    selection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    include_ids = _selected_ids(selection, "include_object_ids")
    exclude_ids = _selected_ids(selection, "exclude_object_ids")
    known_ids = {item.id for item in picture.objects}
    unknown = sorted((include_ids | exclude_ids) - known_ids)
    if unknown:
        raise GeometryRigError(
            "selection references unknown object IDs: " + ", ".join(unknown)
        )
    overlap = sorted(include_ids & exclude_ids)
    if overlap:
        raise GeometryRigError(
            "selection includes and excludes the same objects: " + ", ".join(overlap)
        )
    if active_object_id in exclude_ids:
        raise GeometryRigError("the active object cannot be excluded from bindings")
    if include_ids and active_object_id not in include_ids:
        raise GeometryRigError(
            "selection.include_object_ids must retain the active object"
        )

    records: list[dict[str, Any]] = []
    motion_bindings: list[dict[str, Any]] = []
    fixed_ids: list[str] = []
    excluded_ids: list[str] = []
    for item in picture.objects:
        points = _object_point_names(picture, item)
        dynamic = points is not None and any(name in affected for name in points)
        binding_type = _SUPPORTED_BINDINGS.get(item.kind)
        if not dynamic:
            status = "fixed"
            reason = "No coordinate used by this object changes with the selected driver."
            fixed_ids.append(item.id)
        elif item.id in exclude_ids or (include_ids and item.id not in include_ids):
            status = "excluded"
            reason = "The user chose to keep this dependent object out of the rig."
            excluded_ids.append(item.id)
        elif binding_type is None or points is None:
            status = "unsupported"
            reason = "motion/v1 cannot update this dependent native object safely."
        else:
            status = "included"
            reason = "At least one referenced coordinate changes with the selected driver."
            motion_bindings.append(
                {
                    "object_id": item.id,
                    "type": binding_type,
                    "points": list(points),
                }
            )
        records.append(
            {
                "objectId": item.id,
                "objectKind": item.kind,
                "bindingType": binding_type,
                "type": binding_type or item.kind,
                "points": list(points or ()),
                "enabled": status == "included",
                "status": status,
                "reason": reason,
                "evidence": [
                    *(f"uses:{name}" for name in points or ()),
                    *(f"affected:{name}" for name in points or () if name in affected),
                ],
            }
        )
    return records, motion_bindings, fixed_ids, excluded_ids


def _selection_range(
    selection: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[float, float] | None:
    raw = selection.get("range")
    if raw is None:
        suggested = candidate.get("suggestedRange")
        if not isinstance(suggested, Mapping):
            return None
        return (float(suggested["minimum"]), float(suggested["maximum"]))
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw)
    ):
        raise GeometryRigError("selection.range must be [minimum, maximum]")
    minimum, maximum = float(raw[0]), float(raw[1])
    if not isfinite(minimum) or not isfinite(maximum) or minimum >= maximum:
        raise GeometryRigError("selection.range must contain increasing finite values")
    initial = float(candidate["initial"]["value"])
    if not minimum <= initial <= maximum:
        raise GeometryRigError("selection.range must contain the initial angle")
    return (minimum, maximum)


def _range_is_safe(
    picture: PictureSpec,
    *,
    relation_index: int,
    pivot_name: str,
    value_range: tuple[float, float],
) -> bool:
    relation = picture.intersections[relation_index]
    ellipse_name = (
        relation.path_b if relation.path_a == relation.sort_by else relation.path_a
    )
    ellipse = picture.named_paths.get(ellipse_name)
    if ellipse is None or ellipse.kind != "ellipse":
        return False
    pivot = picture.coordinates[pivot_name]
    minimum, maximum = value_range
    return all(
        _line_ellipse_discriminant(
            minimum + (maximum - minimum) * index / 32,
            pivot=pivot,
            ellipse=ellipse.geometry,
        )
        > 1e-10
        for index in range(33)
    )


def analyze_geometry_rig(
    picture: PictureSpec,
    active_object_id: str,
    *,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze one 2D active native object into a reviewable motion rig draft."""

    if not isinstance(active_object_id, str) or not active_object_id.strip():
        raise GeometryRigError("active_object_id must be a non-empty stable object ID")
    active_object_id = active_object_id.strip()
    selection = {} if selection is None else selection
    if not isinstance(selection, Mapping):
        raise GeometryRigError("selection must be an object")
    allowed_selection = {
        "candidate_id",
        "pivot",
        "range",
        "include_object_ids",
        "exclude_object_ids",
    }
    unknown_selection = sorted(set(selection) - allowed_selection)
    if unknown_selection:
        raise GeometryRigError(
            "selection contains unsupported fields: " + ", ".join(unknown_selection)
        )

    diagnostics: list[dict[str, Any]] = []
    model_hash = semantic_model_hash(picture)
    base: dict[str, Any] = {
        "schema": GEOMETRY_RIG_SCHEMA,
        "status": RIG_STATUS_BLOCKED,
        "pictureIndex": picture.index,
        "activeObjectId": active_object_id,
        "semanticModelHash": model_hash,
        "activeObjectCandidates": _active_object_candidates(picture),
        "driverCandidates": [],
        "selectedDriver": None,
        "coordinates": [],
        "fixedCoordinateIds": [],
        "intersections": [],
        "bindings": [],
        "fixedObjectIds": [],
        "excludedObjectIds": [],
        "diagnostics": diagnostics,
        "motionSpecCore": None,
        "nativeManimSource": None,
        "rigDraft": {
            "driver": None,
            "bindings": [],
            "dependencies": [],
            "coordinates": [],
            "intersections": [],
        },
    }
    for finding in picture.unsupported:
        diagnostics.append(
            _diagnostic(
                "error",
                "PICTURE_SEMANTICS_INCOMPLETE",
                f"The selected picture contains unsupported TikZ semantics: {finding}",
            )
        )
    for warning in picture.warnings:
        diagnostics.append(
            _diagnostic(
                "warning",
                "PICTURE_COMPILER_WARNING",
                str(warning),
            )
        )
    if picture.dimension != 2:
        diagnostics.append(
            _diagnostic(
                "error",
                "DIMENSION_UNSUPPORTED",
                "Geometry rig v1 supports two-dimensional TikZ pictures only.",
            )
        )
        return base

    objects = {item.id: item for item in picture.objects}
    active = objects.get(active_object_id)
    if active is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "ACTIVE_OBJECT_NOT_FOUND",
                "The selected stable object ID is not present in this picture.",
                objectId=active_object_id,
            )
        )
        return base
    if active.kind not in {"line", "arrow"}:
        diagnostics.append(
            _diagnostic(
                "error",
                "ACTIVE_OBJECT_KIND_UNSUPPORTED",
                "Geometry rig v1 can drive only a native line or arrow.",
                objectId=active_object_id,
            )
        )
        return base

    candidates = _driver_candidates(picture, active)
    base["driverCandidates"] = candidates
    usable = [item for item in candidates if item["status"] != "blocked"]
    if not candidates:
        diagnostics.append(
            _diagnostic(
                "error",
                "NO_MATCHING_NAMED_PATH",
                "The selected line is not backed by a stable named TikZ line path and intersection.",
                objectId=active_object_id,
            )
        )
        return base
    if not usable:
        diagnostics.append(
            _diagnostic(
                "error",
                "NO_USABLE_DRIVER",
                "No rotate_named_line candidate has two intersections, a valid pivot, and a safe initial range.",
                objectId=active_object_id,
            )
        )
        return base

    requested_candidate = selection.get("candidate_id")
    if requested_candidate is not None and (
        not isinstance(requested_candidate, str) or not requested_candidate.strip()
    ):
        raise GeometryRigError("selection.candidate_id must be a non-empty string")
    selected_candidate = next(
        (
            item
            for item in usable
            if requested_candidate is not None
            and item["candidateId"] == requested_candidate
        ),
        None,
    )
    if requested_candidate is not None and selected_candidate is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "DRIVER_CANDIDATE_NOT_AVAILABLE",
                "The requested driver candidate is not available for this semantic model.",
            )
        )
        base["status"] = RIG_STATUS_NEEDS_SELECTION
        return base
    if selected_candidate is None:
        selected_candidate = usable[0]

    requested_pivot = selection.get("pivot")
    if requested_pivot is not None and (
        not isinstance(requested_pivot, str) or not requested_pivot.strip()
    ):
        raise GeometryRigError("selection.pivot must be a non-empty coordinate ID")
    pivot_candidates = selected_candidate["pivotCandidates"]
    selected_pivot = next(
        (
            item
            for item in pivot_candidates
            if requested_pivot is not None
            and item["coordinateId"] == requested_pivot
        ),
        None,
    )
    if requested_pivot is not None and selected_pivot is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "PIVOT_NOT_AVAILABLE",
                "The requested pivot is not strictly inside the oriented active line.",
                coordinateId=str(requested_pivot),
            )
        )
        base["status"] = RIG_STATUS_NEEDS_SELECTION
        return base
    if selected_pivot is None:
        selected_pivot = pivot_candidates[0]

    active_path = str(selected_candidate["activePath"])
    intersection_index = int(selected_candidate["intersectionIndex"])
    if selection.get("range") is None:
        selected_range = _suggested_range(
            picture,
            intersection_index,
            selected_pivot["coordinateId"],
            float(selected_candidate["initial"]["value"]),
        )
    else:
        selected_range = _selection_range(selection, selected_candidate)
    if selected_range is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "SAFE_RANGE_NOT_AVAILABLE",
                "A valid rotation interval could not be inferred for the selected pivot.",
                coordinateId=selected_pivot["coordinateId"],
            )
        )
        return base
    if not _range_is_safe(
        picture,
        relation_index=intersection_index,
        pivot_name=selected_pivot["coordinateId"],
        value_range=selected_range,
    ):
        raise GeometryRigError(
            "selection.range crosses a line position without two stable ellipse intersections"
        )
    affected, unselected_intersections = _affected_coordinates(
        picture,
        active_path=active_path,
        intersection_index=intersection_index,
    )
    coordinates, fixed_coordinates = _coordinate_records(
        picture,
        active_path=active_path,
        intersection_index=intersection_index,
        affected=affected,
    )
    intersections = _intersection_records(
        picture,
        active_path=active_path,
        intersection_index=intersection_index,
    )
    bindings, motion_bindings, fixed_objects, excluded_objects = _binding_records(
        picture,
        affected=affected,
        active_object_id=active_object_id,
        selection=selection,
    )
    base.update(
        {
            "coordinates": coordinates,
            "fixedCoordinateIds": fixed_coordinates,
            "intersections": intersections,
            "bindings": bindings,
            "fixedObjectIds": fixed_objects,
            "excludedObjectIds": excluded_objects,
        }
    )

    for index in unselected_intersections:
        diagnostics.append(
            _diagnostic(
                "error",
                "UNSELECTED_ACTIVE_INTERSECTION",
                "The active path also drives another intersection relation that motion/v1 cannot update implicitly.",
            )
        )
    unsupported = [item for item in bindings if item["status"] == "unsupported"]
    for item in unsupported:
        diagnostics.append(
            _diagnostic(
                "error",
                "DYNAMIC_OBJECT_BINDING_UNSUPPORTED",
                "A dependent native object has no safe motion/v1 binding.",
                objectId=item["objectId"],
            )
        )
    if excluded_objects:
        diagnostics.append(
            _diagnostic(
                "warning",
                "DEPENDENT_OBJECTS_EXCLUDED",
                "Excluded dependent objects will remain at their initial geometry during playback.",
            )
        )
    if len(pivot_candidates) > 1 and requested_pivot is None:
        diagnostics.append(
            _diagnostic(
                "warning",
                "PIVOT_RECOMMENDED_NOT_CONFIRMED",
                "The first inferred pivot was selected automatically; the user should confirm it.",
                coordinateId=selected_pivot["coordinateId"],
            )
        )
    if selection.get("range") is None:
        diagnostics.append(
            _diagnostic(
                "warning",
                "RANGE_REQUIRES_CONFIRMATION",
                "The rotation interval is a conservative window around the initial angle and should be confirmed.",
            )
        )
    if not motion_bindings:
        diagnostics.append(
            _diagnostic(
                "error",
                "NO_DYNAMIC_BINDINGS",
                "The selected driver has no included drawable object bindings.",
            )
        )

    selected_driver = {
        "candidateId": selected_candidate["candidateId"],
        "id": "theta",
        "type": "rotate_named_line",
        "activeObjectId": active_object_id,
        "activePath": active_path,
        "pivot": selected_pivot["coordinateId"],
        "intersectionIndex": intersection_index,
        "initial": float(selected_candidate["initial"]["value"]),
        "range": [selected_range[0], selected_range[1]],
        "unit": "radians",
    }
    motion_driver = {
        "id": selected_driver["id"],
        "type": selected_driver["type"],
        "active_path": active_path,
        "pivot": selected_driver["pivot"],
        "intersection_index": intersection_index,
        "initial": selected_driver["initial"],
        "range": list(selected_driver["range"]),
        "unit": selected_driver["unit"],
    }
    base["selectedDriver"] = selected_driver
    base["rigDraft"] = {
        "driver": selected_driver,
        "bindings": [
            {
                "objectId": item["objectId"],
                "type": item["type"],
                "points": list(item["points"]),
                "enabled": True,
                "status": "included",
                "evidence": list(item["evidence"]),
            }
            for item in bindings
            if item["status"] == "included"
        ],
        "dependencies": [
            {
                "objectId": item["objectId"],
                "type": item["type"],
                "points": list(item["points"]),
                "enabled": bool(item["enabled"]),
                "status": item["status"],
                "evidence": list(item["evidence"]),
            }
            for item in bindings
        ],
        "coordinates": coordinates,
        "intersections": intersections,
    }
    base["motionSpecCore"] = {
        "driver": motion_driver,
        "bindings": motion_bindings,
    }
    if any(item["severity"] == "error" for item in diagnostics):
        base["status"] = RIG_STATUS_BLOCKED
        base["motionSpecCore"] = None
    else:
        base["status"] = RIG_STATUS_READY
        try:
            base["nativeManimSource"] = generate_native_manim_source_2d(
                picture,
                base,
            )
        except NativeManimCodegen2DError as exc:
            base["status"] = RIG_STATUS_BLOCKED
            base["motionSpecCore"] = None
            diagnostics.append(
                _diagnostic(
                    "error",
                    "NATIVE_MANIM_SOURCE_UNAVAILABLE",
                    f"The selected relation cannot be expanded as native Manim source: {exc}",
                )
            )
            return base
        diagnostics.insert(
            0,
            _diagnostic(
                "info",
                "RIG_READY",
                "The inferred driver and native object bindings reproduce the initial semantic geometry.",
            ),
        )
    return base


def motion_spec_payload(
    rig: Mapping[str, Any],
    *,
    timeline: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Complete a ready rig core for the existing MotionSpec consumer."""

    if rig.get("schema") != GEOMETRY_RIG_SCHEMA:
        raise GeometryRigError(f"rig schema must be {GEOMETRY_RIG_SCHEMA!r}")
    if rig.get("status") != RIG_STATUS_READY:
        raise GeometryRigError("only a ready geometry rig can compile a motion spec")
    core = rig.get("motionSpecCore")
    if not isinstance(core, Mapping):
        raise GeometryRigError("ready geometry rig is missing motionSpecCore")
    steps = [dict(step) for step in timeline]
    return {
        "schema": MOTION_SCHEMA,
        "picture_index": rig["pictureIndex"],
        "driver": dict(core["driver"]),
        "bindings": [dict(binding) for binding in core["bindings"]],
        "timeline": steps,
    }


def attach_geometry_rig_identity(
    rig: Mapping[str, Any],
    *,
    source_sha256: str,
    provider_revision: str,
    expected_asset_provider_revision: str,
) -> dict[str, Any]:
    """Bind a portable analysis result to source and Provider identities."""

    if rig.get("schema") != GEOMETRY_RIG_SCHEMA:
        raise GeometryRigError(f"rig schema must be {GEOMETRY_RIG_SCHEMA!r}")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in source_sha256)
    ):
        raise GeometryRigError("source_sha256 must be a 64-character hexadecimal digest")
    for name, value in (
        ("provider_revision", provider_revision),
        ("expected_asset_provider_revision", expected_asset_provider_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise GeometryRigError(f"{name} must be a non-empty string")

    result = deepcopy(dict(rig))
    result["sourceSha256"] = source_sha256.lower()
    result["providerRevision"] = provider_revision.strip()
    result["expectedAssetProviderRevision"] = expected_asset_provider_revision.strip()
    revision_match = provider_revision.strip() == expected_asset_provider_revision.strip()
    result["revisionMatch"] = revision_match
    if not revision_match:
        diagnostics = result.setdefault("diagnostics", [])
        diagnostics.insert(
            0,
            _diagnostic(
                "warning",
                "PROVIDER_REVISION_MISMATCH",
                "The selected TikZ asset was compiled by a different Provider revision; analysis and migration simulation may continue, but the Host must block formal application.",
            ),
        )
    return result


__all__ = [
    "GEOMETRY_RIG_SCHEMA",
    "GEOMETRY_SEMANTIC_MODEL_SCHEMA",
    "GeometryRigError",
    "RIG_STATUS_BLOCKED",
    "RIG_STATUS_NEEDS_SELECTION",
    "RIG_STATUS_READY",
    "analyze_geometry_rig",
    "attach_geometry_rig_identity",
    "motion_spec_payload",
    "semantic_model_hash",
    "semantic_model_payload",
]
