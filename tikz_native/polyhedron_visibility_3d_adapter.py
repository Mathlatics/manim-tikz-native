"""Adapt compiled TikZ Native 3D semantics to the visibility-core contract.

This module deliberately has no Manim dependency.  It translates one frozen
``PictureSpec`` into a versioned, deterministic description that a later
renderer binding can consume.  Existing compiler and legacy occlusion objects
remain untouched; the adapter merely records which objects a binding would
suppress after it has attached successfully.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from math import isfinite
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from polyhedron_visibility import (
    ContractError,
    FaceSpec,
    StrokeSpec,
    VertexSpec,
    VisibilityFrame,
    VisibilityModel,
    canonical_trace_json,
    compute_frame_visibility,
)

from .compiler import ObjectSpec, PictureSpec, StyleSpec


ADAPTER_RESULT_SCHEMA = "tikz-native-polyhedron-visibility-3d-adapter/v1"
VALIDATION_MODES = frozenset(
    {"closed_convex_polyhedron", "independent_convex_faces"}
)
DEFAULT_HIDDEN_STYLE: Mapping[str, object] = {
    "drawColor": "#808080",
    "drawOpacity": 1.0,
    "dashPatternPt": [2.0, 2.0],
}


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_cycle(values: Sequence[str]) -> tuple[str, ...]:
    cycle = tuple(values)
    if not cycle:
        return ()
    candidates: list[tuple[str, ...]] = []
    for current in (cycle, tuple(reversed(cycle))):
        candidates.extend(
            current[index:] + current[:index] for index in range(len(current))
        )
    return min(candidates)


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


def _identity(prefix: str, values: Sequence[str]) -> str:
    readable = ".".join(values)
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}.{readable}.{digest}"


def _json_value(value: object, label: str) -> object:
    """Return a detached JSON value and reject lossy/non-finite settings."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TikzNativeVisibility3DAdapterError(
            "INVALID_STYLE",
            f"{label} must contain only finite JSON values",
        ) from exc


def _style_payload(style: StyleSpec) -> dict[str, object]:
    raw = asdict(style)
    keys = {
        "draw_color": "drawColor",
        "fill_color": "fillColor",
        "opacity": "opacity",
        "fill_opacity": "fillOpacity",
        "draw_opacity": "drawOpacity",
        "line_width_pt": "lineWidthPt",
        "line_cap": "lineCap",
        "line_join": "lineJoin",
        "dash_pattern_pt": "dashPatternPt",
        "arrow_tip": "arrowTip",
        "arrow_length_pt": "arrowLengthPt",
        "arrow_width_pt": "arrowWidthPt",
    }
    payload: dict[str, object] = {}
    for source_key, output_key in keys.items():
        value = raw[source_key]
        if value is None:
            continue
        if isinstance(value, tuple):
            value = list(value)
        payload[output_key] = value
    return payload


@dataclass(frozen=True)
class AdapterDiagnostic:
    code: str
    severity: str
    message: str
    object_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "objectIds": list(self.object_ids),
        }


@dataclass(frozen=True)
class FaceBinding3D:
    face_id: str
    vertex_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    authored_cycles: tuple[tuple[str, ...], ...]
    relation_ids: tuple[str, ...]
    hinge_ids: tuple[str, ...]
    fill_alphas: tuple[float, ...]
    occludes_strokes: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "faceId": self.face_id,
            "vertexIds": list(self.vertex_ids),
            "objectIds": list(self.object_ids),
            "authoredCycles": [list(item) for item in self.authored_cycles],
            "relationIds": list(self.relation_ids),
            "hingeIds": list(self.hinge_ids),
            "fillAlphas": list(self.fill_alphas),
            "occludesStrokes": self.occludes_strokes,
        }


@dataclass(frozen=True)
class StrokeBinding3D:
    source_edge_id: str
    vertex_ids: tuple[str, str]
    authored_vertex_pairs: tuple[tuple[str, str], ...]
    object_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    source_kind: str
    visible_style: Mapping[str, object]
    hidden_style: Mapping[str, object]
    z_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceEdgeId": self.source_edge_id,
            "vertexIds": list(self.vertex_ids),
            "authoredVertexPairs": [list(item) for item in self.authored_vertex_pairs],
            "objectIds": list(self.object_ids),
            "relationIds": list(self.relation_ids),
            "sourceKind": self.source_kind,
            "visibleStyle": dict(self.visible_style),
            "hiddenStyle": dict(self.hidden_style),
            "zIndex": self.z_index,
        }


@dataclass(frozen=True)
class TikzNativeVisibility3DAdapterResult:
    validation_mode: str
    model: VisibilityModel
    face_bindings: tuple[FaceBinding3D, ...]
    stroke_bindings: tuple[StrokeBinding3D, ...]
    coordinate_vertex_ids: tuple[tuple[str, str], ...]
    suppressed_object_ids: tuple[str, ...]
    unmanaged_object_ids: tuple[str, ...]
    entry_projection: tuple[tuple[float, float, float], ...]
    entry_trace: VisibilityFrame
    diagnostics: tuple[AdapterDiagnostic, ...]
    model_sha256: str
    entry_trace_sha256: str
    result_sha256: str
    schema: str = ADAPTER_RESULT_SCHEMA

    @property
    def coordinate_vertex_map(self) -> Mapping[str, str]:
        return dict(self.coordinate_vertex_ids)

    def _payload(self, *, include_result_hash: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "validationMode": self.validation_mode,
            "model": self.model.to_dict(),
            "faceBindings": [item.to_dict() for item in self.face_bindings],
            "strokeBindings": [item.to_dict() for item in self.stroke_bindings],
            "coordinateVertexIds": {
                authored: canonical
                for authored, canonical in self.coordinate_vertex_ids
            },
            "suppressedObjectIds": list(self.suppressed_object_ids),
            "unmanagedObjectIds": list(self.unmanaged_object_ids),
            "entryProjection": [list(row) for row in self.entry_projection],
            "entryTrace": self.entry_trace.to_dict(),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "modelSha256": self.model_sha256,
            "entryTraceSha256": self.entry_trace_sha256,
        }
        if include_result_hash:
            payload["resultSha256"] = self.result_sha256
        return payload

    def to_dict(self) -> dict[str, object]:
        return self._payload(include_result_hash=True)


class TikzNativeVisibility3DAdapterError(ValueError):
    """Fail-closed adapter error with a stable, machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        diagnostics: Sequence[AdapterDiagnostic] = (),
    ) -> None:
        self.code = code
        self.diagnostics = tuple(diagnostics)
        super().__init__(f"{code}: {message}")


@dataclass
class _FaceCandidate:
    cycle: tuple[str, ...]
    authored_cycles: set[tuple[str, ...]] = field(default_factory=set)
    object_ids: set[str] = field(default_factory=set)
    relation_ids: set[str] = field(default_factory=set)
    hinge_ids: set[str] = field(default_factory=set)
    fill_alphas: set[float] = field(default_factory=set)


@dataclass
class _StrokeCandidate:
    vertex_ids: tuple[str, str]
    authored_vertex_pairs: set[tuple[str, str]] = field(default_factory=set)
    object_ids: set[str] = field(default_factory=set)
    relation_ids: set[str] = field(default_factory=set)
    source_kinds: set[str] = field(default_factory=set)
    styles: list[tuple[int, str, dict[str, object], dict[str, object]]] = field(
        default_factory=list
    )


def _coordinates_3d(picture: PictureSpec) -> dict[str, tuple[float, float, float]]:
    coordinates: dict[str, tuple[float, float, float]] = {}
    for name, raw in sorted(picture.coordinates.items()):
        try:
            point = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise TikzNativeVisibility3DAdapterError(
                "INVALID_COORDINATE",
                f"coordinate {name!r} is not numeric",
            ) from exc
        if len(point) != 3 or not all(isfinite(value) for value in point):
            raise TikzNativeVisibility3DAdapterError(
                "INVALID_COORDINATE",
                f"coordinate {name!r} must be a finite 3D point",
            )
        coordinates[name] = point  # type: ignore[assignment]
    return coordinates


def _weld_coordinates(
    coordinates: Mapping[str, tuple[float, float, float]],
) -> tuple[dict[str, str], dict[str, tuple[float, float, float]], list[AdapterDiagnostic]]:
    """Give exactly coincident TikZ aliases one core vertex identity."""

    by_position: dict[tuple[float, float, float], list[str]] = {}
    for name, point in coordinates.items():
        by_position.setdefault(point, []).append(name)
    alias_map: dict[str, str] = {}
    canonical_positions: dict[str, tuple[float, float, float]] = {}
    diagnostics: list[AdapterDiagnostic] = []
    for point, names in sorted(by_position.items(), key=lambda item: sorted(item[1])):
        ordered = sorted(names)
        canonical = ordered[0]
        canonical_positions[canonical] = point
        for name in ordered:
            alias_map[name] = canonical
        if len(ordered) > 1:
            diagnostics.append(
                AdapterDiagnostic(
                    code="WELDED_COORDINATE_ALIASES",
                    severity="info",
                    message=(
                        f"coincident coordinates {', '.join(ordered)} reuse core vertex "
                        f"{canonical}"
                    ),
                )
            )
    return alias_map, canonical_positions, diagnostics


def _named_cycle(
    raw: object,
    *,
    alias_map: Mapping[str, str],
    coordinates: Mapping[str, tuple[float, float, float]],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise TikzNativeVisibility3DAdapterError(
            "UNPROVEN_FACE",
            f"{label} has no cycle of at least three named coordinates",
        )
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str) or item not in coordinates:
            raise TikzNativeVisibility3DAdapterError(
                "UNPROVEN_FACE",
                f"{label} references an unknown coordinate {item!r}",
            )
        names.append(alias_map[item])
    if len(set(names)) != len(names):
        raise TikzNativeVisibility3DAdapterError(
            "DEGENERATE_WELDED_FACE",
            f"{label} collapses after coincident coordinate aliases are welded",
        )
    return _canonical_cycle(names)


def _fill_alpha(item: ObjectSpec) -> float:
    value = item.style.fill_opacity
    if value is None:
        value = item.style.opacity
    return float(value)


def _polygon_faces(
    picture: PictureSpec,
    *,
    alias_map: Mapping[str, str],
    coordinates: Mapping[str, tuple[float, float, float]],
    diagnostics: list[AdapterDiagnostic],
) -> tuple[list[_FaceCandidate], set[str]]:
    by_cycle: dict[tuple[str, ...], _FaceCandidate] = {}
    managed_objects: set[str] = set()
    for item in sorted(picture.objects, key=lambda value: value.id):
        if item.kind != "polygon":
            continue
        raw_names = item.geometry.get("point_names")
        if not isinstance(raw_names, (list, tuple)) or len(raw_names) < 3:
            diagnostics.append(
                AdapterDiagnostic(
                    "UNMANAGED_UNNAMED_FACE",
                    "warning",
                    f"polygon {item.id} has no complete point_names evidence",
                    (item.id,),
                )
            )
            continue
        raw_cycle = tuple(str(value) for value in raw_names)
        cycle = _named_cycle(
            raw_names,
            alias_map=alias_map,
            coordinates=coordinates,
            label=f"polygon {item.id}",
        )
        candidate = by_cycle.setdefault(cycle, _FaceCandidate(cycle))
        candidate.authored_cycles.add(raw_cycle)
        candidate.object_ids.add(item.id)
        candidate.fill_alphas.add(_fill_alpha(item))
        managed_objects.add(item.id)
    return [by_cycle[key] for key in sorted(by_cycle)], managed_objects


def _face_evidence(
    picture: PictureSpec,
    faces: Sequence[_FaceCandidate],
    *,
    alias_map: Mapping[str, str],
    coordinates: Mapping[str, tuple[float, float, float]],
    diagnostics: list[AdapterDiagnostic],
) -> None:
    by_cycle: dict[tuple[str, ...], _FaceCandidate] = {}
    for candidate in faces:
        by_cycle[candidate.cycle] = candidate
        # After coplanar mesh faces are merged, a legacy declaration can name
        # either the maximal surface or one of its authored source polygons.
        for authored in candidate.authored_cycles:
            welded = tuple(alias_map[name] for name in authored)
            by_cycle[_canonical_cycle(welded)] = candidate

    def record(raw: Sequence[str], evidence_id: str, kind: str) -> None:
        try:
            cycle = _named_cycle(
                raw,
                alias_map=alias_map,
                coordinates=coordinates,
                label=f"{kind} {evidence_id}",
            )
        except TikzNativeVisibility3DAdapterError:
            raise
        candidate = by_cycle.get(cycle)
        if candidate is None:
            diagnostics.append(
                AdapterDiagnostic(
                    "UNBOUND_FACE_EVIDENCE",
                    "warning",
                    f"{kind} {evidence_id} names a face with no polygon.point_names owner",
                )
            )
            return
        if kind == "relation":
            candidate.relation_ids.add(evidence_id)
        else:
            candidate.hinge_ids.add(evidence_id)

    for relation in sorted(picture.occlusion_relations, key=lambda item: item.id):
        record(relation.face_names, relation.id, "relation")
    for hinge in sorted(picture.hinge_relations, key=lambda item: item.id):
        record(hinge.fixed_face_names, hinge.id, "hinge")
        record(hinge.moving_face_names, hinge.id, "hinge")


def _face_normal(
    cycle: Sequence[str],
    positions: Mapping[str, tuple[float, float, float]],
) -> np.ndarray:
    points = np.asarray([positions[name] for name in cycle], dtype=float)
    origin = points[0]
    for index in range(1, len(points) - 1):
        normal = np.cross(points[index] - origin, points[index + 1] - origin)
        length = float(np.linalg.norm(normal))
        if length > 1.0e-28:
            return normal / length
    raise TikzNativeVisibility3DAdapterError(
        "INVALID_FACE_SYSTEM", "face evidence is geometrically degenerate"
    )


def _coplanar(
    first: _FaceCandidate,
    second: _FaceCandidate,
    positions: Mapping[str, tuple[float, float, float]],
) -> bool:
    first_points = np.asarray([positions[name] for name in first.cycle], dtype=float)
    second_points = np.asarray([positions[name] for name in second.cycle], dtype=float)
    all_points = np.concatenate((first_points, second_points), axis=0)
    extent = float(np.linalg.norm(np.max(all_points, axis=0) - np.min(all_points, axis=0)))
    tolerance = max(1.0e-14, 8.0e-9 * max(extent, 1.0e-14))
    first_normal = _face_normal(first.cycle, positions)
    second_normal = _face_normal(second.cycle, positions)
    if abs(float(np.dot(first_normal, second_normal))) < 1.0 - 1.0e-10:
        return False
    return bool(
        np.max(np.abs(np.dot(second_points - first_points[0], first_normal)))
        <= tolerance
    )


def _boundary_union_cycle(
    first: Sequence[str], second: Sequence[str]
) -> tuple[str, ...] | None:
    def edges(cycle: Sequence[str]) -> set[tuple[str, str]]:
        return {
            _canonical_pair(cycle[index], cycle[(index + 1) % len(cycle)])
            for index in range(len(cycle))
        }

    first_edges = edges(first)
    second_edges = edges(second)
    shared = first_edges & second_edges
    if len(shared) != 1:
        return None
    boundary = (first_edges | second_edges) - shared
    adjacency: dict[str, set[str]] = {}
    for start, end in boundary:
        adjacency.setdefault(start, set()).add(end)
        adjacency.setdefault(end, set()).add(start)
    if not adjacency or any(len(neighbours) != 2 for neighbours in adjacency.values()):
        return None
    start = min(adjacency)
    possible: list[tuple[str, ...]] = []
    for first_step in sorted(adjacency[start]):
        cycle = [start]
        previous = start
        current = first_step
        while current != start and len(cycle) <= len(adjacency):
            cycle.append(current)
            next_values = sorted(adjacency[current] - {previous})
            if len(next_values) != 1:
                break
            previous, current = current, next_values[0]
        if current == start and len(cycle) == len(adjacency):
            possible.append(tuple(cycle))
    if not possible:
        return None
    return _canonical_cycle(min(possible))


def _validate_one_face(
    cycle: Sequence[str], positions: Mapping[str, tuple[float, float, float]]
) -> None:
    model = VisibilityModel(
        visibility_group_id="adapter-face-validation",
        vertices=tuple(VertexSpec(name, positions[name]) for name in sorted(set(cycle))),
        faces=(FaceSpec("face", tuple(cycle)),),
        strokes=(),
    )
    model.validate()


def _merge_coplanar_faces(
    source: Sequence[_FaceCandidate],
    positions: Mapping[str, tuple[float, float, float]],
) -> list[_FaceCandidate]:
    faces = list(source)
    while True:
        merged_pair: tuple[int, int, tuple[str, ...]] | None = None
        for first_index, first in enumerate(faces):
            for second_index in range(first_index + 1, len(faces)):
                second = faces[second_index]
                boundary = _boundary_union_cycle(first.cycle, second.cycle)
                if boundary is None or not _coplanar(first, second, positions):
                    continue
                try:
                    _validate_one_face(boundary, positions)
                except ContractError as exc:
                    raise TikzNativeVisibility3DAdapterError(
                        "COPLANAR_FACE_MERGE_FAILED",
                        "coplanar adjacent polygons cannot form one maximal convex face: "
                        f"{first.cycle}, {second.cycle}: {exc}",
                    ) from exc
                merged_pair = (first_index, second_index, boundary)
                break
            if merged_pair is not None:
                break
        if merged_pair is None:
            return sorted(faces, key=lambda item: item.cycle)
        first_index, second_index, boundary = merged_pair
        first = faces[first_index]
        second = faces[second_index]
        combined = _FaceCandidate(
            cycle=boundary,
            authored_cycles=first.authored_cycles | second.authored_cycles,
            object_ids=first.object_ids | second.object_ids,
            relation_ids=first.relation_ids | second.relation_ids,
            hinge_ids=first.hinge_ids | second.hinge_ids,
            fill_alphas=first.fill_alphas | second.fill_alphas,
        )
        faces = [
            value
            for index, value in enumerate(faces)
            if index not in {first_index, second_index}
        ]
        faces.append(combined)
        faces.sort(key=lambda item: item.cycle)


def _normalise_default_hidden_style(
    raw: Mapping[str, object] | None,
) -> dict[str, object]:
    if raw is None:
        return dict(DEFAULT_HIDDEN_STYLE)
    if not isinstance(raw, Mapping):
        raise TikzNativeVisibility3DAdapterError(
            "INVALID_STYLE", "default_hidden_style must be a mapping"
        )
    value = _json_value(dict(raw), "default_hidden_style")
    assert isinstance(value, dict)
    return value


def _relation_strokes(
    picture: PictureSpec,
    *,
    alias_map: Mapping[str, str],
    coordinates: Mapping[str, tuple[float, float, float]],
    diagnostics: list[AdapterDiagnostic],
) -> tuple[dict[tuple[str, str], _StrokeCandidate], set[str]]:
    object_map = {item.id: item for item in picture.objects}
    candidates: dict[tuple[str, str], _StrokeCandidate] = {}
    relation_members: set[str] = set()
    for relation in sorted(picture.occlusion_relations, key=lambda item: item.id):
        if relation.start_name not in coordinates or relation.end_name not in coordinates:
            raise TikzNativeVisibility3DAdapterError(
                "UNPROVEN_RELATION_STROKE",
                f"relation {relation.id} has unknown full-line endpoints",
            )
        start = alias_map[relation.start_name]
        end = alias_map[relation.end_name]
        if start == end:
            raise TikzNativeVisibility3DAdapterError(
                "UNPROVEN_RELATION_STROKE",
                f"relation {relation.id} collapses to a zero-length stroke",
            )
        pair = _canonical_pair(start, end)
        candidate = candidates.setdefault(pair, _StrokeCandidate(pair))
        candidate.authored_vertex_pairs.add((relation.start_name, relation.end_name))
        candidate.relation_ids.add(relation.id)
        candidate.source_kinds.add("legacy_occlusion_relation")
        if not relation.object_ids:
            raise TikzNativeVisibility3DAdapterError(
                "UNPROVEN_RELATION_STROKE",
                f"relation {relation.id} has no compiled fragments to bind",
            )
        for object_id in sorted(set(relation.object_ids)):
            item = object_map.get(object_id)
            if item is None or item.kind != "line":
                raise TikzNativeVisibility3DAdapterError(
                    "UNPROVEN_RELATION_STROKE",
                    f"relation {relation.id} references missing/non-line object {object_id}",
                )
            if item.style.arrow_tip is not None:
                raise TikzNativeVisibility3DAdapterError(
                    "UNPROVEN_RELATION_STROKE",
                    f"relation {relation.id} contains an arrow fragment that cannot be rebuilt safely",
                )
            geometry_pair = (
                item.geometry.get("start_name"),
                item.geometry.get("end_name"),
            )
            if set(geometry_pair) != {relation.start_name, relation.end_name}:
                raise TikzNativeVisibility3DAdapterError(
                    "UNPROVEN_RELATION_STROKE",
                    f"relation {relation.id} fragment {object_id} disagrees with its full endpoints",
                )
            candidate.object_ids.add(object_id)
            relation_members.add(object_id)
        candidate.styles.append(
            (
                int(relation.z_index),
                relation.id,
                _style_payload(relation.visible_style),
                _style_payload(relation.hidden_style),
            )
        )
    return candidates, relation_members


def _plain_named_lines(
    picture: PictureSpec,
    candidates: dict[tuple[str, str], _StrokeCandidate],
    relation_members: set[str],
    *,
    alias_map: Mapping[str, str],
    coordinates: Mapping[str, tuple[float, float, float]],
    default_hidden_style: Mapping[str, object],
    diagnostics: list[AdapterDiagnostic],
) -> None:
    for item in sorted(picture.objects, key=lambda value: value.id):
        if item.kind != "line" or item.id in relation_members:
            continue
        start_name = item.geometry.get("start_name")
        end_name = item.geometry.get("end_name")
        if not isinstance(start_name, str) or not isinstance(end_name, str):
            diagnostics.append(
                AdapterDiagnostic(
                    "UNMANAGED_UNNAMED_STROKE",
                    "warning",
                    f"line {item.id} has no named endpoints",
                    (item.id,),
                )
            )
            continue
        if start_name not in coordinates or end_name not in coordinates:
            diagnostics.append(
                AdapterDiagnostic(
                    "UNMANAGED_UNPROVEN_STROKE",
                    "warning",
                    f"line {item.id} references an unknown coordinate",
                    (item.id,),
                )
            )
            continue
        if item.style.arrow_tip is not None:
            diagnostics.append(
                AdapterDiagnostic(
                    "UNMANAGED_ARROW_STROKE",
                    "warning",
                    f"arrow line {item.id} remains author-managed",
                    (item.id,),
                )
            )
            continue
        pair = _canonical_pair(alias_map[start_name], alias_map[end_name])
        if pair[0] == pair[1]:
            diagnostics.append(
                AdapterDiagnostic(
                    "UNMANAGED_ZERO_LENGTH_STROKE",
                    "warning",
                    f"line {item.id} collapses after coordinate welding",
                    (item.id,),
                )
            )
            continue
        candidate = candidates.setdefault(pair, _StrokeCandidate(pair))
        candidate.authored_vertex_pairs.add((start_name, end_name))
        candidate.object_ids.add(item.id)
        candidate.source_kinds.add("named_line")
        visible = _style_payload(item.style)
        hidden = dict(visible)
        hidden.update(default_hidden_style)
        candidate.styles.append((int(item.z_index), item.id, visible, hidden))


def _override_maps(
    overrides: Mapping[str, object] | None,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if overrides is None:
        return {}, {}, {}
    if not isinstance(overrides, Mapping):
        raise TikzNativeVisibility3DAdapterError(
            "INVALID_OVERRIDES", "overrides must be a mapping"
        )
    allowed = {"faceOccludesStrokes", "strokeVisibilityModes", "hiddenStyles"}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise TikzNativeVisibility3DAdapterError(
            "INVALID_OVERRIDES", f"unknown override fields: {', '.join(unknown)}"
        )
    values: list[Mapping[str, object]] = []
    for key in ("faceOccludesStrokes", "strokeVisibilityModes", "hiddenStyles"):
        value = overrides.get(key, {})
        if not isinstance(value, Mapping):
            raise TikzNativeVisibility3DAdapterError(
                "INVALID_OVERRIDES", f"overrides.{key} must be a mapping"
            )
        values.append(value)
    return values[0], values[1], values[2]


def _selector_values(
    selectors: Mapping[str, object], available: Mapping[str, set[str]], label: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for selector in sorted(selectors):
        matches = sorted(
            identity for identity, aliases in available.items() if selector in aliases
        )
        if not matches:
            raise TikzNativeVisibility3DAdapterError(
                "UNKNOWN_OVERRIDE_TARGET", f"{label} selector {selector!r} matches nothing"
            )
        if len(matches) > 1:
            raise TikzNativeVisibility3DAdapterError(
                "AMBIGUOUS_OVERRIDE_TARGET",
                f"{label} selector {selector!r} matches {', '.join(matches)}",
            )
        identity = matches[0]
        value = selectors[selector]
        previous = result.get(identity, value)
        if previous != value:
            raise TikzNativeVisibility3DAdapterError(
                "CONFLICTING_OVERRIDES", f"conflicting {label} settings for {identity}"
            )
        result[identity] = value
    return result


def _adapter_result(
    *,
    validation_mode: str,
    model: VisibilityModel,
    face_bindings: tuple[FaceBinding3D, ...],
    stroke_bindings: tuple[StrokeBinding3D, ...],
    coordinate_vertex_ids: tuple[tuple[str, str], ...],
    suppressed_object_ids: tuple[str, ...],
    unmanaged_object_ids: tuple[str, ...],
    projection: tuple[tuple[float, float, float], ...],
    trace: VisibilityFrame,
    diagnostics: tuple[AdapterDiagnostic, ...],
) -> TikzNativeVisibility3DAdapterResult:
    model_hash = _sha256(model.to_dict())
    trace_hash = hashlib.sha256(canonical_trace_json(trace).encode("utf-8")).hexdigest()
    provisional = TikzNativeVisibility3DAdapterResult(
        validation_mode=validation_mode,
        model=model,
        face_bindings=face_bindings,
        stroke_bindings=stroke_bindings,
        coordinate_vertex_ids=coordinate_vertex_ids,
        suppressed_object_ids=suppressed_object_ids,
        unmanaged_object_ids=unmanaged_object_ids,
        entry_projection=projection,
        entry_trace=trace,
        diagnostics=diagnostics,
        model_sha256=model_hash,
        entry_trace_sha256=trace_hash,
        result_sha256="",
    )
    result_hash = _sha256(provisional._payload(include_result_hash=False))
    return TikzNativeVisibility3DAdapterResult(
        **{
            **provisional.__dict__,
            "result_sha256": result_hash,
        }
    )


def adapt_picture_visibility_3d(
    picture: PictureSpec,
    validation_mode: Literal[
        "closed_convex_polyhedron", "independent_convex_faces"
    ] = "closed_convex_polyhedron",
    default_hidden_style: Mapping[str, object] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> TikzNativeVisibility3DAdapterResult:
    """Build one deterministic visibility contract from a compiled 3D picture.

    ``closed_convex_polyhedron`` is the safe production gate.  The
    ``independent_convex_faces`` mode intentionally permits open teaching
    constructions such as a dihedral angle, but every individual face must
    still be planar, maximal, and strictly convex.

    Face occlusion is semantic and therefore independent of TikZ fill alpha.
    It can only be changed explicitly through ``overrides.faceOccludesStrokes``.
    Selectors may be generated IDs or bound source object/relation IDs.
    """

    if validation_mode not in VALIDATION_MODES:
        raise TikzNativeVisibility3DAdapterError(
            "INVALID_VALIDATION_MODE",
            f"validation_mode must be one of {', '.join(sorted(VALIDATION_MODES))}",
        )
    if not isinstance(picture, PictureSpec):
        raise TikzNativeVisibility3DAdapterError(
            "INVALID_PICTURE", "picture must be a compiled PictureSpec"
        )
    if picture.dimension != 3 or picture.projection_3d is None:
        raise TikzNativeVisibility3DAdapterError(
            "NOT_THREE_DIMENSIONAL",
            "picture must have a supported parallel 3D projection",
        )
    if picture.unsupported:
        raise TikzNativeVisibility3DAdapterError(
            "PICTURE_NOT_READY",
            "picture contains unsupported compiler features; visibility adaptation is closed",
        )

    projection = tuple(
        tuple(float(component) for component in row)
        for row in picture.projection_3d.matrix
    )
    coordinates = _coordinates_3d(picture)
    alias_map, canonical_positions, diagnostics = _weld_coordinates(coordinates)
    hidden_default = _normalise_default_hidden_style(default_hidden_style)

    faces, face_object_ids = _polygon_faces(
        picture,
        alias_map=alias_map,
        coordinates=coordinates,
        diagnostics=diagnostics,
    )
    if not faces:
        raise TikzNativeVisibility3DAdapterError(
            "NO_PROVEN_FACES",
            "no polygon.point_names face can participate in visibility",
            diagnostics=diagnostics,
        )
    try:
        faces = _merge_coplanar_faces(faces, canonical_positions)
    except ContractError as exc:
        raise TikzNativeVisibility3DAdapterError(
            "INVALID_FACE_SYSTEM", str(exc), diagnostics=diagnostics
        ) from exc
    _face_evidence(
        picture,
        faces,
        alias_map=alias_map,
        coordinates=coordinates,
        diagnostics=diagnostics,
    )

    stroke_candidates, relation_members = _relation_strokes(
        picture,
        alias_map=alias_map,
        coordinates=coordinates,
        diagnostics=diagnostics,
    )
    _plain_named_lines(
        picture,
        stroke_candidates,
        relation_members,
        alias_map=alias_map,
        coordinates=coordinates,
        default_hidden_style=hidden_default,
        diagnostics=diagnostics,
    )

    face_occlusion_raw, stroke_modes_raw, hidden_styles_raw = _override_maps(overrides)
    face_ids = {_identity("face", face.cycle): face for face in faces}
    face_selectors = {
        face_id: {face_id, *face.object_ids, *face.relation_ids, *face.hinge_ids}
        for face_id, face in face_ids.items()
    }
    face_overrides = _selector_values(
        face_occlusion_raw, face_selectors, "faceOccludesStrokes"
    )

    face_specs: list[FaceSpec] = []
    face_bindings: list[FaceBinding3D] = []
    for face_id in sorted(face_ids):
        candidate = face_ids[face_id]
        occludes = face_overrides.get(face_id, True)
        if not isinstance(occludes, bool):
            raise TikzNativeVisibility3DAdapterError(
                "INVALID_OVERRIDES",
                f"faceOccludesStrokes value for {face_id} must be boolean",
            )
        face_specs.append(FaceSpec(face_id, candidate.cycle, occludes))
        face_bindings.append(
            FaceBinding3D(
                face_id=face_id,
                vertex_ids=candidate.cycle,
                object_ids=tuple(sorted(candidate.object_ids)),
                authored_cycles=tuple(sorted(candidate.authored_cycles)),
                relation_ids=tuple(sorted(candidate.relation_ids)),
                hinge_ids=tuple(sorted(candidate.hinge_ids)),
                fill_alphas=tuple(sorted(candidate.fill_alphas)),
                occludes_strokes=occludes,
            )
        )

    stroke_ids = {
        _identity("stroke", pair): candidate
        for pair, candidate in sorted(stroke_candidates.items())
    }
    stroke_selectors = {
        stroke_id: {
            stroke_id,
            *candidate.object_ids,
            *candidate.relation_ids,
        }
        for stroke_id, candidate in stroke_ids.items()
    }
    stroke_modes = _selector_values(
        stroke_modes_raw, stroke_selectors, "strokeVisibilityModes"
    )
    hidden_style_overrides = _selector_values(
        hidden_styles_raw, stroke_selectors, "hiddenStyles"
    )

    face_sets = {
        face.face_id: set(face.vertex_ids)
        for face in face_specs
    }
    stroke_specs: list[StrokeSpec] = []
    stroke_bindings: list[StrokeBinding3D] = []
    suppressed: set[str] = set()
    for stroke_id in sorted(stroke_ids):
        candidate = stroke_ids[stroke_id]
        mode = stroke_modes.get(stroke_id, "auto")
        if mode not in {"auto", "always_visible", "always_hidden"}:
            raise TikzNativeVisibility3DAdapterError(
                "INVALID_OVERRIDES",
                f"strokeVisibilityModes value for {stroke_id} is unsupported",
            )
        incidents = tuple(
            sorted(
                face_id
                for face_id, face_vertices in face_sets.items()
                if set(candidate.vertex_ids).issubset(face_vertices)
            )
        )
        stroke_specs.append(
            StrokeSpec(
                source_edge_id=stroke_id,
                vertex_ids=candidate.vertex_ids,
                incident_face_ids=incidents,
                render_binding_id=stroke_id,
                visibility_mode=str(mode),
            )
        )
        # Highest authored z-index wins coincident duplicate styling.  The ID
        # tie-break makes the choice independent of input list order.
        selected_style = max(candidate.styles, key=lambda value: (value[0], value[1]))
        hidden_style = hidden_style_overrides.get(stroke_id, selected_style[3])
        if not isinstance(hidden_style, Mapping):
            raise TikzNativeVisibility3DAdapterError(
                "INVALID_STYLE", f"hiddenStyles value for {stroke_id} must be a mapping"
            )
        hidden_style_json = _json_value(dict(hidden_style), f"hiddenStyles.{stroke_id}")
        assert isinstance(hidden_style_json, dict)
        source_kind = (
            next(iter(candidate.source_kinds))
            if len(candidate.source_kinds) == 1
            else "combined_named_sources"
        )
        binding = StrokeBinding3D(
            source_edge_id=stroke_id,
            vertex_ids=candidate.vertex_ids,
            authored_vertex_pairs=tuple(sorted(candidate.authored_vertex_pairs)),
            object_ids=tuple(sorted(candidate.object_ids)),
            relation_ids=tuple(sorted(candidate.relation_ids)),
            source_kind=source_kind,
            visible_style=selected_style[2],
            hidden_style=hidden_style_json,
            z_index=selected_style[0],
        )
        stroke_bindings.append(binding)
        suppressed.update(candidate.object_ids)
        style_fingerprints = {
            _canonical_json((item[2], item[3])) for item in candidate.styles
        }
        if len(style_fingerprints) > 1:
            diagnostics.append(
                AdapterDiagnostic(
                    "COINCIDENT_STROKE_STYLE_RESOLVED",
                    "info",
                    f"coincident sources for {stroke_id} use the highest z-index style",
                    tuple(sorted(candidate.object_ids)),
                )
            )

    used_vertex_ids = sorted(
        {
            vertex_id
            for face in face_specs
            for vertex_id in face.vertex_ids
        }
        | {
            vertex_id
            for stroke in stroke_specs
            for vertex_id in stroke.vertex_ids
        }
    )
    model = VisibilityModel(
        visibility_group_id=f"tikz-native-picture-{picture.index}-visibility-3d",
        vertices=tuple(
            VertexSpec(name, canonical_positions[name]) for name in used_vertex_ids
        ),
        faces=tuple(sorted(face_specs, key=lambda item: item.face_id)),
        strokes=tuple(sorted(stroke_specs, key=lambda item: item.source_edge_id)),
    )
    require_closed = validation_mode == "closed_convex_polyhedron"
    try:
        model.validate(require_closed_convex_manifold=require_closed)
    except ContractError as exc:
        message = str(exc)
        code = (
            "OPEN_FACE_SYSTEM"
            if require_closed and "closed two-manifold" in message
            else "INVALID_FACE_SYSTEM"
        )
        raise TikzNativeVisibility3DAdapterError(
            code, message, diagnostics=diagnostics
        ) from exc

    try:
        entry_trace = compute_frame_visibility(
            model,
            projection_matrix=projection,
        )
    except (ContractError, ValueError) as exc:
        raise TikzNativeVisibility3DAdapterError(
            "ENTRY_TRACE_FAILED", str(exc), diagnostics=diagnostics
        ) from exc

    object_ids = {item.id for item in picture.objects}
    managed_face_and_stroke = face_object_ids | suppressed
    unmanaged = tuple(sorted(object_ids - managed_face_and_stroke))
    diagnostics_tuple = tuple(
        sorted(
            diagnostics,
            key=lambda item: (item.severity, item.code, item.message, item.object_ids),
        )
    )
    return _adapter_result(
        validation_mode=validation_mode,
        model=model,
        face_bindings=tuple(sorted(face_bindings, key=lambda item: item.face_id)),
        stroke_bindings=tuple(
            sorted(stroke_bindings, key=lambda item: item.source_edge_id)
        ),
        coordinate_vertex_ids=tuple(sorted(alias_map.items())),
        suppressed_object_ids=tuple(sorted(suppressed)),
        unmanaged_object_ids=unmanaged,
        projection=projection,
        trace=entry_trace,
        diagnostics=diagnostics_tuple,
    )


__all__ = [
    "ADAPTER_RESULT_SCHEMA",
    "AdapterDiagnostic",
    "DEFAULT_HIDDEN_STYLE",
    "FaceBinding3D",
    "StrokeBinding3D",
    "TikzNativeVisibility3DAdapterError",
    "TikzNativeVisibility3DAdapterResult",
    "VALIDATION_MODES",
    "adapt_picture_visibility_3d",
]
