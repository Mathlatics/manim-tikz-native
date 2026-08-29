"""Provider-owned relationship analysis for explicit native TikZ 3D geometry.

The analyzer consumes only versioned compiler semantics.  In particular, it
never recovers coordinate identities by parsing generated object IDs or by
guessing from coincident points.  A motion core is emitted only after the
author confirms both a candidate and its numeric range.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from math import acos, isfinite, pi, sqrt
from typing import Any, Mapping, Sequence

from .compiler import HINGE_RELATION_SCHEMA, HingeRelationSpec, ObjectSpec, PictureSpec
from .native_manim_codegen_3d import (
    NativeManimCodegen3DError,
    generate_native_manim_source_3d,
)
from .native_manim_codegen_3d_v2 import (
    NativeManimCodegen3DV2Error,
    generate_native_manim_source_3d_v2,
    point_on_segment_driver_candidates,
)
from .planar_curves_3d import (
    PlanarTikz3DError,
    restore_registered_planar_curve_geometry,
)


GEOMETRY_RIG_3D_SCHEMA = "tikz-native-geometry-rig-3d/v1"
GEOMETRY_SEMANTIC_MODEL_3D_SCHEMA = "tikz-native-geometry-semantic-model-3d/v1"
HINGE_FOLD_CANDIDATE_ID_PREFIX = "hinge_fold:"
MOTION_3D_SCHEMA = "tikz-native-motion-3d/v1"

RIG_STATUS_READY = "ready"
RIG_STATUS_NEEDS_SELECTION = "needs_selection"
RIG_STATUS_BLOCKED = "blocked"

CAMERA_OPERATION_MODES = ("front", "side", "top", "oblique", "isometric")
_ORTHOGONAL_CAMERA_OPERATION_MODES = frozenset(
    {"front", "side", "top", "isometric"}
)
_MOTION_BINDING_TYPES = {
    "line": "line",
    "arrow": "line",
    "dot": "dot",
    "polygon": "polygon",
    "label": "label",
    "path_label": "path_label",
}


class GeometryRig3DError(ValueError):
    """Raised for a malformed author selection or non-portable semantic input."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _canonical_value(value.tolist())
    if isinstance(value, float):
        if not isfinite(value):
            raise GeometryRig3DError("3D semantic geometry contains a non-finite number")
        return 0.0 if value == 0 else value
    return value


def semantic_model_3d_payload(picture: PictureSpec) -> dict[str, Any]:
    projection = picture.projection_3d
    return {
        "schema": GEOMETRY_SEMANTIC_MODEL_3D_SCHEMA,
        "pictureIndex": picture.index,
        "dimension": picture.dimension,
        "projection": (
            None
            if projection is None
            else {
                "source": projection.source,
                "matrix": _canonical_value(projection.matrix),
                "azimuthDegrees": projection.azimuth_degrees,
                "elevationDegrees": projection.elevation_degrees,
            }
        ),
        "coordinates": [
            {
                "coordinateId": name,
                "value": list(value),
                "dependency": picture.coordinate_dependencies.get(name),
            }
            for name, value in picture.coordinates.items()
        ],
        "planarFrames3D": [
            {
                "planeId": plane_id,
                "geometry": geometry,
            }
            for plane_id, geometry in sorted(picture.planar_frames_3d.items())
        ],
        "hingeRelations": [
            {
                "schema": item.schema,
                "relationId": item.id,
                "axis": list(item.axis_names),
                "fixedFace": list(item.fixed_face_names),
                "movingFace": list(item.moving_face_names),
            }
            for item in picture.hinge_relations
        ],
        "objects": [
            {
                "objectId": item.id,
                "kind": item.kind,
                "geometry": item.geometry,
                "zIndex": item.z_index,
            }
            for item in picture.objects
        ],
        "occlusionRelations": [
            {
                "relationId": item.id,
                "linePointNames": [item.start_name, item.end_name],
                "facePointNames": list(item.face_names),
                "objectIds": list(item.object_ids),
                "zIndex": item.z_index,
            }
            for item in picture.occlusion_relations
        ],
    }


def semantic_model_3d_hash(picture: PictureSpec) -> str:
    payload = json.dumps(
        _canonical_value(semantic_model_3d_payload(picture)),
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
    **context: object,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    result.update(
        {
            key: value
            for key, value in context.items()
            if value is not None and value != ""
        }
    )
    return result


def _point3(value: Sequence[float], field: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise GeometryRig3DError(f"{field} must be a three-dimensional point")
    result = tuple(float(item) for item in value)
    if any(not isfinite(item) for item in result):
        raise GeometryRig3DError(f"{field} contains a non-finite number")
    return result  # type: ignore[return-value]


def _subtract(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _scale(value: Sequence[float], amount: float) -> tuple[float, float, float]:
    return tuple(float(item) * amount for item in value)  # type: ignore[return-value]


def _length(value: Sequence[float]) -> float:
    return sqrt(_dot(value, value))


def _perpendicular_reference(
    picture: PictureSpec,
    names: Sequence[str],
    axis: tuple[str, str],
) -> tuple[float, float, float]:
    start = _point3(picture.coordinates[axis[0]], f"coordinate {axis[0]}")
    end = _point3(picture.coordinates[axis[1]], f"coordinate {axis[1]}")
    direction = _subtract(end, start)
    axis_length = _length(direction)
    if axis_length <= 1e-12:
        raise GeometryRig3DError("hinge axis must not have zero length")
    unit = _scale(direction, 1.0 / axis_length)
    for name in names:
        if name in axis:
            continue
        relative = _subtract(
            _point3(picture.coordinates[name], f"coordinate {name}"), start
        )
        perpendicular = _subtract(relative, _scale(unit, _dot(relative, unit)))
        size = _length(perpendicular)
        if size > 1e-10:
            return _scale(perpendicular, 1.0 / size)
    raise GeometryRig3DError("hinge face has no point outside the directed axis")


def _hinge_angle(picture: PictureSpec, relation: HingeRelationSpec) -> float:
    axis = (relation.axis_names[0], relation.axis_names[1])
    fixed = _perpendicular_reference(picture, relation.fixed_face_names, axis)
    moving = _perpendicular_reference(picture, relation.moving_face_names, axis)
    return acos(max(-1.0, min(1.0, _dot(fixed, moving))))


def _cyclic_names_equal(left: Sequence[str], right: Sequence[str]) -> bool:
    if len(left) != len(right) or set(left) != set(right):
        return False
    if not left:
        return True
    doubled = list(right) * 2
    reverse = list(reversed(right))
    reversed_doubled = reverse * 2
    size = len(left)
    return any(
        list(left) == candidate[index : index + size]
        for candidate in (doubled, reversed_doubled)
        for index in range(size)
    )


def _polygon_for_face(
    picture: PictureSpec,
    point_names: Sequence[str],
) -> ObjectSpec | None:
    matches: list[ObjectSpec] = []
    for item in picture.objects:
        if item.kind != "polygon":
            continue
        names = item.geometry.get("point_names")
        if (
            isinstance(names, list)
            and all(isinstance(name, str) and name for name in names)
            and _cyclic_names_equal(point_names, names)
        ):
            matches.append(item)
    return matches[0] if len(matches) == 1 else None


def _hinge_candidate(
    picture: PictureSpec,
    relation: HingeRelationSpec,
) -> tuple[dict[str, Any], ObjectSpec | None, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    moving_object = _polygon_for_face(picture, relation.moving_face_names)
    fixed_object = _polygon_for_face(picture, relation.fixed_face_names)
    if moving_object is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "MOVING_FACE_OBJECT_NOT_FOUND",
                "The explicit moving face has no unique polygon with matching point_names.",
                relationId=relation.id,
            )
        )
    if fixed_object is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "FIXED_FACE_OBJECT_NOT_FOUND",
                "The explicit fixed face has no unique polygon with matching point_names.",
                relationId=relation.id,
            )
        )
    try:
        initial = _hinge_angle(picture, relation)
    except GeometryRig3DError as error:
        initial = 0.0
        diagnostics.append(
            _diagnostic(
                "error",
                "HINGE_GEOMETRY_INVALID",
                str(error),
                relationId=relation.id,
            )
        )
    epsilon = pi / 36.0
    half_window = pi / 4.0
    minimum = max(epsilon, initial - half_window)
    maximum = min(pi - epsilon, initial + half_window)
    if not minimum < initial < maximum:
        minimum = initial - half_window
        maximum = initial + half_window
    axis = set(relation.axis_names)
    moving_coordinates = [
        name for name in relation.moving_face_names if name not in axis
    ]
    status = "blocked" if diagnostics else "recommended"
    return (
        {
            "candidateId": f"{HINGE_FOLD_CANDIDATE_ID_PREFIX}{relation.id}",
            "driverId": f"{HINGE_FOLD_CANDIDATE_ID_PREFIX}{relation.id}",
            "candidateKind": "geometry_driver",
            "driverType": "hinge_fold",
            "status": status,
            "relationId": relation.id,
            "activeObjectId": moving_object.id if moving_object is not None else None,
            "fixedObjectId": fixed_object.id if fixed_object is not None else None,
            "axis": list(relation.axis_names),
            "fixedFace": list(relation.fixed_face_names),
            "movingFace": list(relation.moving_face_names),
            "movingCoordinates": moving_coordinates,
            "initial": {"value": initial, "unit": "radians"},
            "suggestedRange": {
                "minimum": minimum,
                "maximum": maximum,
                "unit": "radians",
                "source": "conservative-hinge-window",
            },
            "reason": (
                "The source explicitly declares the directed hinge, fixed face, and moving face."
            ),
        },
        moving_object,
        diagnostics,
    )


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
    moving_coordinates: Sequence[str],
) -> set[str]:
    affected = set(moving_coordinates)
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


def _derived_relation_candidate(
    name: str,
    dependency: Mapping[str, Any],
    *,
    affected: set[str],
) -> dict[str, Any] | None:
    operation = str(dependency.get("operation") or "")
    if operation == "interpolation":
        start = dependency.get("start")
        end = dependency.get("end")
        parameter = dependency.get("parameter")
        if not isinstance(start, str) or not isinstance(end, str):
            return None
        if isinstance(parameter, bool) or not isinstance(parameter, (int, float)):
            return None
        relation_type = "point_on_segment"
        depends_on = [start, end]
        relation = {
            "type": relation_type,
            "start": start,
            "end": end,
            "parameter": float(parameter),
        }
    elif operation == "projection":
        point = dependency.get("point")
        line_start = dependency.get("line_start")
        line_end = dependency.get("line_end")
        if not all(isinstance(item, str) and item for item in (point, line_start, line_end)):
            return None
        relation_type = "project_point_to_line"
        depends_on = [str(point), str(line_start), str(line_end)]
        relation = {
            "type": relation_type,
            "point": str(point),
            "line_start": str(line_start),
            "line_end": str(line_end),
        }
    else:
        return None
    return {
        "candidateId": f"derived_relation:{name}",
        "candidateKind": "derived_relation",
        "relationType": relation_type,
        "status": "included" if name in affected else "fixed",
        "coordinateId": name,
        "dependsOn": depends_on,
        "affectedByDriver": name in affected,
        "relation": relation,
        "reason": "The relation is explicitly preserved by the compiled TikZ coordinate dependency.",
    }


def _is_right_handed_rotation_matrix(
    matrix: Sequence[Sequence[object]],
    *,
    atol: float = 1e-7,
) -> bool:
    """Return whether ``matrix`` is a proper orthogonal 3D frame.

    Geometry analysis intentionally keeps this small check independent from
    the Manim camera implementation.  ``orbit`` is only a valid operation for
    two proper rotation frames; TikZ oblique bases are general invertible
    projections and must use linear matrix interpolation instead.
    """

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
            if abs(dot - expected) > atol:
                return False
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    return abs(determinant - 1.0) <= atol


def _camera_candidates(picture: PictureSpec) -> list[dict[str, Any]]:
    entry_is_rotation = bool(
        picture.projection_3d is not None
        and _is_right_handed_rotation_matrix(picture.projection_3d.matrix)
    )
    return [
        {
            "candidateId": f"camera:{mode}",
            "candidateKind": "camera_operation",
            "operationType": "camera",
            "status": "available",
            "mode": mode,
            "transitionTypes": (
                ["linear", "orbit"]
                if entry_is_rotation
                and mode in _ORTHOGONAL_CAMERA_OPERATION_MODES
                else ["linear"]
            ),
            "restoresEntry": True,
            "reason": (
                "The Provider owns this parallel-projection camera preset. "
                "Orbit is offered only when both endpoints are right-handed "
                "orthogonal frames."
            ),
        }
        for mode in CAMERA_OPERATION_MODES
    ]


def _object_point_names(item: ObjectSpec) -> tuple[str, ...] | None:
    geometry = item.geometry
    if item.kind in {"line", "arrow", "path_label"}:
        values = (geometry.get("start_name"), geometry.get("end_name"))
    elif item.kind == "polygon":
        raw = geometry.get("point_names")
        values = tuple(raw) if isinstance(raw, list) else ()
    elif item.kind in {"dot", "circle", "ellipse"}:
        values = (geometry.get("center_name"),)
    elif item.kind in {"planar_circle_3d", "planar_ellipse_3d"}:
        raw = geometry.get("plane_point_names")
        values = tuple(raw) if isinstance(raw, list) else ()
    elif item.kind == "label":
        values = (geometry.get("at_name"),)
    elif item.kind in {"angle", "angle_label", "right_angle"}:
        values = (
            geometry.get("first_name"),
            geometry.get("vertex_name"),
            geometry.get("third_name"),
        )
    else:
        return None
    if not values or any(not isinstance(value, str) or not value for value in values):
        return None
    return tuple(str(value) for value in values)


def _unique_strings(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _undirected_edge(left: str, right: str) -> frozenset[str]:
    return frozenset((left, right))


def _face_edges(names: Sequence[str]) -> set[frozenset[str]]:
    return {
        _undirected_edge(names[index], names[(index + 1) % len(names)])
        for index in range(len(names))
    }


def _objects_for_edges(
    picture: PictureSpec,
    edges: set[frozenset[str]],
) -> list[str]:
    result: list[str] = []
    for item in picture.objects:
        names = _object_point_names(item)
        if names is None or len(names) != 2:
            continue
        if _undirected_edge(names[0], names[1]) in edges:
            result.append(item.id)
    return _unique_strings(result)


def _objects_for_coordinate(
    picture: PictureSpec,
    coordinate_id: str,
) -> list[str]:
    return _unique_strings(
        [
            item.id
            for item in picture.objects
            if (
                (names := _object_point_names(item)) is not None
                and coordinate_id in names
            )
        ]
    )


def _semantic_groups(
    picture: PictureSpec,
    *,
    candidate: Mapping[str, Any] | None,
    affected: set[str],
) -> list[dict[str, Any]]:
    """Group raw native fragments by explicit geometry, never by object-ID text."""

    if candidate is None:
        return []
    relation_id = str(candidate.get("relationId") or "")
    relation = next(
        (item for item in picture.hinge_relations if item.id == relation_id),
        None,
    )
    if relation is None:
        return []

    fixed_edges = _face_edges(relation.fixed_face_names)
    moving_edges = _face_edges(relation.moving_face_names)
    axis_edge = {
        _undirected_edge(relation.axis_names[0], relation.axis_names[1])
    }
    fixed_object_ids = _unique_strings(
        [
            str(candidate.get("fixedObjectId") or ""),
            *_objects_for_edges(picture, fixed_edges),
        ]
    )
    moving_object_ids = _unique_strings(
        [
            str(candidate.get("activeObjectId") or ""),
            *_objects_for_edges(picture, moving_edges),
        ]
    )
    groups: list[dict[str, Any]] = [
        {
            "groupId": f"hinge:{relation.id}:fixed-face",
            "label": "Fixed face " + "–".join(relation.fixed_face_names),
            "roles": ["fixed", "face", "occluder"],
            "objectIds": fixed_object_ids,
            "coordinateIds": list(relation.fixed_face_names),
            "required": True,
        },
        {
            "groupId": f"hinge:{relation.id}:moving-face",
            "label": "Moving face " + "–".join(relation.moving_face_names),
            "roles": ["active", "driver-body", "face", "occluder"],
            "objectIds": moving_object_ids,
            "coordinateIds": list(relation.moving_face_names),
            "required": True,
        },
        {
            "groupId": f"hinge:{relation.id}:axis",
            "label": "Hinge axis " + "–".join(relation.axis_names),
            "roles": ["axis", "fixed"],
            "objectIds": _objects_for_edges(picture, axis_edge),
            "coordinateIds": list(relation.axis_names),
            "required": True,
        },
    ]

    for coordinate_id in picture.coordinates:
        if coordinate_id not in affected:
            continue
        dependency = picture.coordinate_dependencies.get(coordinate_id)
        if _derived_relation_candidate(
            coordinate_id,
            dependency or {},
            affected=affected,
        ) is None:
            continue
        groups.append(
            {
                "groupId": f"derived:{coordinate_id}",
                "label": f"Derived point {coordinate_id}",
                "roles": ["derived", "follower"],
                "objectIds": _objects_for_coordinate(picture, coordinate_id),
                "coordinateIds": [coordinate_id],
                "required": True,
            }
        )

    face_edges = fixed_edges | moving_edges
    for relation_spec in picture.occlusion_relations:
        relation_edge = _undirected_edge(
            relation_spec.start_name,
            relation_spec.end_name,
        )
        is_probe = relation_edge not in face_edges
        dynamic = any(
            name in affected
            for name in (
                relation_spec.start_name,
                relation_spec.end_name,
                *relation_spec.face_names,
            )
        )
        groups.append(
            {
                "groupId": f"occlusion:{relation_spec.id}",
                "label": (
                    "Occlusion probe " if is_probe else "Occlusion relation "
                )
                + f"{relation_spec.start_name}–{relation_spec.end_name} / face "
                + "–".join(relation_spec.face_names),
                "roles": [
                    *(["probe"] if is_probe else []),
                    "occlusion",
                    "follower" if dynamic else "fixed",
                ],
                "objectIds": list(relation_spec.object_ids),
                "coordinateIds": _unique_strings(
                    [
                        relation_spec.start_name,
                        relation_spec.end_name,
                        *relation_spec.face_names,
                    ]
                ),
                "required": not is_probe,
            }
        )
    return groups


def _selected_ids(selection: Mapping[str, Any], key: str) -> set[str]:
    raw = selection.get(key, [])
    if raw is None:
        return set()
    if not isinstance(raw, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw
    ):
        raise GeometryRig3DError(f"selection.{key} must be an array of object IDs")
    if len(raw) != len(set(raw)):
        raise GeometryRig3DError(f"selection.{key} must not contain duplicates")
    return set(raw)


def _binding_records(
    picture: PictureSpec,
    *,
    affected: set[str],
    active_object_id: str | None,
    selection: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[str],
    list[str],
    list[dict[str, Any]],
]:
    include_ids = _selected_ids(selection, "include_object_ids")
    exclude_ids = _selected_ids(selection, "exclude_object_ids")
    known_ids = {item.id for item in picture.objects}
    unknown = sorted((include_ids | exclude_ids) - known_ids)
    if unknown:
        raise GeometryRig3DError(
            "selection references unknown object IDs: " + ", ".join(unknown)
        )
    overlap = sorted(include_ids & exclude_ids)
    if overlap:
        raise GeometryRig3DError(
            "selection includes and excludes the same objects: "
            + ", ".join(overlap)
        )
    occlusion_member_ids = {
        object_id
        for relation in picture.occlusion_relations
        for object_id in relation.object_ids
    }
    touched_occlusion = sorted((include_ids | exclude_ids) & occlusion_member_ids)
    if touched_occlusion:
        raise GeometryRig3DError(
            "occlusion-managed segments cannot be selected individually: "
            + ", ".join(touched_occlusion)
        )
    if active_object_id and active_object_id in exclude_ids:
        raise GeometryRig3DError("the active moving face cannot be excluded")
    if include_ids and active_object_id and active_object_id not in include_ids:
        raise GeometryRig3DError(
            "selection.include_object_ids must retain the active moving face"
        )

    records: list[dict[str, Any]] = []
    motion_bindings: list[dict[str, Any]] = []
    dynamic_ids: list[str] = []
    fixed_ids: list[str] = []
    excluded_ids: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for item in picture.objects:
        if item.id in occlusion_member_ids:
            continue
        names = _object_point_names(item)
        dynamic = names is not None and any(name in affected for name in names)
        binding_type = _MOTION_BINDING_TYPES.get(item.kind)
        planar_kind = item.kind in {"planar_circle_3d", "planar_ellipse_3d"}
        planar_error: str | None = None
        if planar_kind:
            try:
                restore_registered_planar_curve_geometry(
                    item.geometry,
                    picture.planar_frames_3d,
                    expected_curve_id=item.id,
                )
            except PlanarTikz3DError as exc:
                planar_error = str(exc)
        if planar_error is not None:
            role = "unsupported"
            enabled = False
            reason = (
                "The planar curve does not agree with its registered supporting "
                f"plane: {planar_error}"
            )
            diagnostics.append(
                _diagnostic(
                    "error",
                    "PLANAR_CURVE_REGISTRY_MISMATCH",
                    reason,
                    objectId=item.id,
                )
            )
        elif planar_kind:
            role = "unsupported"
            enabled = False
            reason = (
                "embedded motion-3d/v1 cannot safely retain an explicit static "
                "planar curve while another geometry driver is active."
            )
            diagnostics.append(
                _diagnostic(
                    "error",
                    (
                        "DYNAMIC_OBJECT_BINDING_UNSUPPORTED"
                        if dynamic
                        else "EMBEDDED_RUNTIME_OBJECT_UNSUPPORTED"
                    ),
                    reason,
                    objectId=item.id,
                )
            )
        elif not dynamic:
            role = "fixed"
            enabled = False
            reason = "No explicitly named coordinate used by this object changes with the hinge."
            fixed_ids.append(item.id)
        elif item.id in exclude_ids or (include_ids and item.id not in include_ids):
            role = "excluded"
            enabled = False
            reason = "The author excluded this follower from the 3D rig."
            excluded_ids.append(item.id)
        elif binding_type is None or names is None:
            role = "unsupported"
            enabled = False
            reason = "motion-3d/v1 has no safe explicit binding for this dependent object."
            diagnostics.append(
                _diagnostic(
                    "error",
                    "DYNAMIC_OBJECT_BINDING_UNSUPPORTED",
                    reason,
                    objectId=item.id,
                )
            )
        else:
            role = "active" if item.id == active_object_id else "follower"
            enabled = True
            reason = (
                "This is the explicitly declared moving face."
                if role == "active"
                else "At least one explicit coordinate dependency follows the hinge."
            )
            dynamic_ids.append(item.id)
            motion_bindings.append(
                {
                    "object_id": item.id,
                    "type": binding_type,
                    "points": list(names),
                }
            )
        records.append(
            {
                "objectId": item.id,
                "objectKind": item.kind,
                "bindingType": binding_type,
                "pointNames": list(names or ()),
                "role": role,
                "enabled": enabled,
                "reason": reason,
                "evidence": [
                    *(f"uses:{name}" for name in names or ()),
                    *(f"affected:{name}" for name in names or () if name in affected),
                ],
            }
        )
    return (
        records,
        motion_bindings,
        dynamic_ids,
        fixed_ids,
        excluded_ids,
        diagnostics,
    )


def _occlusion_records(
    picture: PictureSpec,
    *,
    affected: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relation in picture.occlusion_relations:
        names = [relation.start_name, relation.end_name, *relation.face_names]
        dynamic = any(name in affected for name in names)
        records.append(
            {
                "relationId": relation.id,
                "linePointNames": [relation.start_name, relation.end_name],
                "facePointNames": list(relation.face_names),
                "objectIds": list(relation.object_ids),
                "role": "dynamic" if dynamic else "fixed",
                "dynamicByGeometry": dynamic,
                "cameraSensitive": True,
                "status": "included",
                "reason": (
                    "A line endpoint or occluding-face point follows the hinge."
                    if dynamic
                    else "Its geometry is fixed, but a selected camera operation still recomputes visibility."
                ),
                "evidence": [
                    *(f"line:{name}" for name in (relation.start_name, relation.end_name)),
                    *(f"face:{name}" for name in relation.face_names),
                    *(f"affected:{name}" for name in names if name in affected),
                ],
            }
        )
    return records


def _selection_range(
    selection: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[float, float] | None:
    raw = selection.get("range")
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw)
    ):
        raise GeometryRig3DError("selection.range must be [minimum, maximum]")
    minimum, maximum = float(raw[0]), float(raw[1])
    if not isfinite(minimum) or not isfinite(maximum) or minimum >= maximum:
        raise GeometryRig3DError("selection.range must contain increasing finite values")
    initial = float(candidate["initial"]["value"])
    if not minimum <= initial <= maximum:
        raise GeometryRig3DError("selection.range must contain the authored hinge angle")
    return minimum, maximum


def _derived_motion_specs(
    picture: PictureSpec,
    *,
    affected: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for name in picture.coordinates:
        if name not in affected:
            continue
        dependency = picture.coordinate_dependencies.get(name)
        if dependency is None:
            continue
        candidate = _derived_relation_candidate(name, dependency, affected=affected)
        if candidate is None:
            diagnostics.append(
                _diagnostic(
                    "error",
                    "DERIVED_RELATION_UNSUPPORTED",
                    "An affected coordinate uses a relation unsupported by motion-3d/v1.",
                    coordinateId=name,
                )
            )
            continue
        relation = candidate["relation"]
        specs.append({"name": name, **deepcopy(relation)})
    return specs, diagnostics


def _attach_native_manim_source_v2(
    picture: PictureSpec,
    base: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> None:
    try:
        base["nativeManimSourceV2"] = generate_native_manim_source_3d_v2(
            picture,
            base,
        )
    except NativeManimCodegen3DV2Error as exc:
        base["nativeManimSourceV2"] = None
        diagnostics.append(
            _diagnostic(
                "warning",
                "NATIVE_MANIM_SOURCE_V2_UNAVAILABLE",
                "The legacy 3D rig and v1 source remain available, but the "
                f"multi-driver authoring source is unavailable: {exc}",
            )
        )


def analyze_geometry_rig_3d(
    picture: PictureSpec,
    *,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze one explicit 3D picture into a reviewable, restoring rig draft."""

    selection = {} if selection is None else selection
    if not isinstance(selection, Mapping):
        raise GeometryRig3DError("selection must be an object")
    allowed_selection = {
        "candidate_id",
        "range",
        "include_object_ids",
        "exclude_object_ids",
    }
    unknown_selection = sorted(set(selection) - allowed_selection)
    if unknown_selection:
        raise GeometryRig3DError(
            "selection contains unsupported fields: " + ", ".join(unknown_selection)
        )

    diagnostics: list[dict[str, Any]] = []
    base: dict[str, Any] = {
        "schema": GEOMETRY_RIG_3D_SCHEMA,
        "dimension": 3,
        "status": RIG_STATUS_BLOCKED,
        "pictureIndex": picture.index,
        "semanticModelHash": semantic_model_3d_hash(picture),
        "motionCandidates": [],
        "selectedMotionCandidate": None,
        "semanticGroups": [],
        "coordinateRoles": [],
        "affectedCoordinateIds": [],
        "fixedCoordinateIds": [],
        "bindings": [],
        "dynamicObjectIds": [],
        "fixedObjectIds": [],
        "excludedObjectIds": [],
        "occlusionBindings": [],
        "diagnostics": diagnostics,
        "motionSpecCore": None,
        "nativeManimSource": None,
        "nativeManimSourceV2": None,
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
            _diagnostic("warning", "PICTURE_COMPILER_WARNING", str(warning))
        )
    if picture.dimension != 3 or picture.projection_3d is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "DIMENSION_UNSUPPORTED",
                "Geometry Rig 3D v1 requires a compiled three-dimensional TikZ picture.",
            )
        )
        return base
    if not picture.hinge_relations:
        diagnostics.append(
            _diagnostic(
                "error",
                "NO_EXPLICIT_HINGE_RELATION",
                "Add a versioned DeclareSpaceHinge relation before authoring a fold.",
            )
        )
        return base
    if any(item.schema != HINGE_RELATION_SCHEMA for item in picture.hinge_relations):
        diagnostics.append(
            _diagnostic(
                "error",
                "HINGE_RELATION_SCHEMA_UNSUPPORTED",
                "The picture contains an unsupported hinge-relation schema.",
            )
        )
        return base

    hinge_candidates: list[dict[str, Any]] = []
    moving_objects: dict[str, ObjectSpec | None] = {}
    for relation in picture.hinge_relations:
        candidate, moving_object, candidate_diagnostics = _hinge_candidate(
            picture, relation
        )
        hinge_candidates.append(candidate)
        moving_objects[candidate["candidateId"]] = moving_object
        diagnostics.extend(candidate_diagnostics)

    usable = [item for item in hinge_candidates if item["status"] != "blocked"]
    recommended = usable[0] if usable else None
    affected = (
        _affected_coordinates(picture, recommended["movingCoordinates"])
        if recommended is not None
        else set()
    )
    point_driver_candidates = point_on_segment_driver_candidates(picture)
    all_driver_affected = set(affected)
    for candidate in point_driver_candidates:
        if candidate["status"] != "blocked":
            all_driver_affected.update(candidate["affectedCoordinates"])
    derived_candidates = [
        candidate
        for name, dependency in picture.coordinate_dependencies.items()
        if (
            candidate := _derived_relation_candidate(
                name,
                dependency,
                affected=all_driver_affected,
            )
        )
        is not None
    ]
    base["motionCandidates"] = [
        *hinge_candidates,
        *point_driver_candidates,
        *derived_candidates,
        *_camera_candidates(picture),
    ]

    requested_candidate = selection.get("candidate_id")
    if requested_candidate is not None and (
        not isinstance(requested_candidate, str) or not requested_candidate.strip()
    ):
        raise GeometryRig3DError("selection.candidate_id must be a non-empty string")
    selected = next(
        (
            item
            for item in usable
            if requested_candidate is not None
            and item["candidateId"] == requested_candidate
        ),
        None,
    )
    if requested_candidate is not None and selected is None:
        diagnostics.append(
            _diagnostic(
                "error",
                "DRIVER_CANDIDATE_NOT_AVAILABLE",
                "The requested explicit hinge candidate is not available.",
            )
        )
    selected_range = _selection_range(selection, selected) if selected is not None else None
    active_candidate = selected or recommended
    if active_candidate is not None:
        affected = _affected_coordinates(
            picture, active_candidate["movingCoordinates"]
        )

    base["affectedCoordinateIds"] = [
        name for name in picture.coordinates if name in affected
    ]
    base["fixedCoordinateIds"] = [
        name for name in picture.coordinates if name not in affected
    ]
    axis_names = set(active_candidate["axis"] if active_candidate else ())
    moving_names = set(
        active_candidate["movingCoordinates"] if active_candidate else ()
    )
    base["coordinateRoles"] = [
        {
            "coordinateId": name,
            "value": list(value),
            "role": (
                "hinge_axis"
                if name in axis_names
                else "driver_coordinate"
                if name in moving_names
                else "derived"
                if name in affected
                else "fixed"
            ),
            "dependency": picture.coordinate_dependencies.get(name),
            "dependsOn": list(
                _dependency_names(picture.coordinate_dependencies.get(name))
            ),
            "affectedByDriver": name in affected,
        }
        for name, value in picture.coordinates.items()
    ]

    active_object = (
        moving_objects.get(active_candidate["candidateId"])
        if active_candidate is not None
        else None
    )
    (
        bindings,
        motion_bindings,
        dynamic_ids,
        fixed_ids,
        excluded_ids,
        binding_diagnostics,
    ) = _binding_records(
        picture,
        affected=affected,
        active_object_id=active_object.id if active_object is not None else None,
        selection=selection,
    )
    diagnostics.extend(binding_diagnostics)
    base.update(
        {
            "bindings": bindings,
            "semanticGroups": _semantic_groups(
                picture,
                candidate=active_candidate,
                affected=affected,
            ),
            "dynamicObjectIds": dynamic_ids,
            "fixedObjectIds": fixed_ids,
            "excludedObjectIds": excluded_ids,
            "occlusionBindings": _occlusion_records(
                picture,
                affected=affected,
            ),
        }
    )
    if excluded_ids:
        diagnostics.append(
            _diagnostic(
                "warning",
                "DEPENDENT_OBJECTS_EXCLUDED",
                "Excluded followers remain at authored geometry during playback.",
            )
        )

    if selected is None or selected_range is None:
        if not any(item["severity"] == "error" for item in diagnostics):
            base["status"] = RIG_STATUS_NEEDS_SELECTION
            diagnostics.append(
                _diagnostic(
                    "warning",
                    "AUTHOR_CONFIRMATION_REQUIRED",
                    "Confirm the explicit hinge candidate and an angle range before preview or Native Clip generation.",
                )
            )
            _attach_native_manim_source_v2(picture, base, diagnostics)
        return base

    selected_record = {
        "candidateId": selected["candidateId"],
        "driverId": selected["driverId"],
        "candidateKind": "geometry_driver",
        "driverType": "hinge_fold",
        "relationId": selected["relationId"],
        "activeObjectId": selected["activeObjectId"],
        "axis": list(selected["axis"]),
        "movingCoordinates": list(selected["movingCoordinates"]),
        "initial": float(selected["initial"]["value"]),
        "range": [selected_range[0], selected_range[1]],
        "unit": "radians",
    }
    base["selectedMotionCandidate"] = selected_record
    derived_specs, derived_diagnostics = _derived_motion_specs(
        picture,
        affected=affected,
    )
    diagnostics.extend(derived_diagnostics)
    if not motion_bindings:
        diagnostics.append(
            _diagnostic(
                "error",
                "NO_DYNAMIC_BINDINGS",
                "The selected hinge has no supported native-object bindings.",
            )
        )
    if any(item["severity"] == "error" for item in diagnostics):
        base["status"] = RIG_STATUS_BLOCKED
        return base

    base["motionSpecCore"] = {
        "end_policy": "restore_entry",
        "driver": {
            "id": selected["relationId"].replace("-", "_"),
            "type": "hinge_fold",
            "axis": list(selected["axis"]),
            "moving_points": list(selected["movingCoordinates"]),
            "initial": float(selected["initial"]["value"]),
            "range": [selected_range[0], selected_range[1]],
            "unit": "radians",
        },
        "derived_coordinates": derived_specs,
        "bindings": motion_bindings,
        "camera": {
            "entry_mode": "tikz",
            # The last authored camera mode is not known yet.  It may be the
            # non-orthogonal oblique preset, so restoration must use the one
            # transition that is valid for every invertible projection.
            "restore_transition": "linear",
            "restore_duration": 1.6,
        },
    }
    base["status"] = RIG_STATUS_READY
    try:
        base["nativeManimSource"] = generate_native_manim_source_3d(
            picture,
            base,
        )
    except NativeManimCodegen3DError as exc:
        # Readable source is an additive authoring surface.  A source-generator
        # boundary must not revoke the already validated legacy rig/runtime
        # contract; Host can fail closed only for the new native-first draft.
        base["nativeManimSource"] = None
        diagnostics.append(
            _diagnostic(
                "warning",
                "NATIVE_MANIM_SOURCE_UNAVAILABLE",
                "The selected 3D relation remains available to the legacy "
                "preview/runtime, but cannot be expanded as readable Manim "
                f"source: {exc}",
            )
        )
    _attach_native_manim_source_v2(picture, base, diagnostics)
    diagnostics.insert(
        0,
        _diagnostic(
            "info",
            "RIG_READY",
            "The explicit 3D hinge and all supported followers reproduce the authored entry geometry.",
        ),
    )
    return base


def attach_geometry_rig_3d_identity(
    rig: Mapping[str, Any],
    *,
    source_sha256: str,
    provider_revision: str,
    expected_asset_provider_revision: str,
) -> dict[str, Any]:
    """Bind analysis to immutable source/Provider identity and fail closed."""

    if rig.get("schema") != GEOMETRY_RIG_3D_SCHEMA:
        raise GeometryRig3DError(f"rig schema must be {GEOMETRY_RIG_3D_SCHEMA!r}")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in source_sha256)
    ):
        raise GeometryRig3DError("source_sha256 must be a SHA-256 digest")
    for field, value in (
        ("provider_revision", provider_revision),
        ("expected_asset_provider_revision", expected_asset_provider_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise GeometryRig3DError(f"{field} must be a non-empty string")

    result = deepcopy(dict(rig))
    result["sourceSha256"] = source_sha256.lower()
    result["providerRevision"] = provider_revision.strip()
    result["expectedAssetProviderRevision"] = expected_asset_provider_revision.strip()
    revision_match = provider_revision.strip() == expected_asset_provider_revision.strip()
    result["revisionMatch"] = revision_match
    if not revision_match:
        result["status"] = RIG_STATUS_BLOCKED
        result["motionSpecCore"] = None
        result["nativeManimSource"] = None
        result["nativeManimSourceV2"] = None
        result.setdefault("diagnostics", []).insert(
            0,
            _diagnostic(
                "error",
                "PROVIDER_REVISION_MISMATCH",
                "The frozen ShapeAsset Provider revision differs from this 3D Geometry Rig Provider. Analysis remains reviewable, but preview, Native Clip generation, and formal rendering are blocked.",
            ),
        )
    return result


__all__ = [
    "CAMERA_OPERATION_MODES",
    "GEOMETRY_RIG_3D_SCHEMA",
    "GEOMETRY_SEMANTIC_MODEL_3D_SCHEMA",
    "GeometryRig3DError",
    "HINGE_FOLD_CANDIDATE_ID_PREFIX",
    "RIG_STATUS_BLOCKED",
    "RIG_STATUS_NEEDS_SELECTION",
    "RIG_STATUS_READY",
    "analyze_geometry_rig_3d",
    "attach_geometry_rig_3d_identity",
    "semantic_model_3d_hash",
    "semantic_model_3d_payload",
]
