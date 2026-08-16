"""Prove and adapt compiled TikZ open faces for dynamic visibility.

The frozen polyhedron adapter already discovers maximal finite faces and
logical strokes.  This module adds the stricter evidence needed by an open-face
Manim binding: articulated seams, coplanar owner exclusions, and a complete
proof that every legacy relation fragment is an ordered partition of one
straight source segment.  No drawable is mutated here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from typing import Literal, Mapping, Sequence

import numpy as np

from polyhedron_visibility import StrokeSpec, TolerancePolicy
from polyhedron_visibility.open_faces import (
    ARTICULATED_HINGE_POLICY,
    OpenFaceContractError,
    OpenFaceSeamSpec,
    OpenFaceSpec,
    OpenFaceSolverError,
    OpenFaceStrokeSpec,
    OpenFaceVertexSpec,
    OpenFaceVisibilityFrame,
    OpenFaceVisibilityModel,
    canonical_open_face_trace_json,
    compute_open_face_visibility,
)

from .compiler import ObjectSpec, PictureSpec
from .polyhedron_visibility_3d_adapter import (
    FaceBinding3D,
    StrokeBinding3D,
    TikzNativeVisibility3DAdapterError,
    TikzNativeVisibility3DAdapterResult,
    _style_payload,
    adapt_picture_visibility_3d,
)


OPEN_FACE_ADAPTER_RESULT_SCHEMA = "tikz-native-open-face-visibility-3d-adapter/v1"
LEGACY_TOPOLOGY_DISCOVERY_SCHEMA = (
    "tikz-native-open-face-visibility-3d-legacy-topology/v1"
)
_HINGE_TOPOLOGY_PROBE_RADIANS = 1.0e-4


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
    candidates: list[tuple[str, ...]] = []
    for current in (cycle, tuple(reversed(cycle))):
        candidates.extend(
            current[index:] + current[:index] for index in range(len(current))
        )
    return min(candidates)


def _point3(value: object, label: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TikzNativeOpenFaceVisibility3DAdapterError(
            "INVALID_FRAGMENT_GEOMETRY", f"{label} must be numeric"
        ) from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise TikzNativeOpenFaceVisibility3DAdapterError(
            "INVALID_FRAGMENT_GEOMETRY",
            f"{label} must be a finite three-component point",
        )
    return point


@dataclass(frozen=True)
class LegacyRelationFragment3D:
    object_id: str
    start_parameter: float
    end_parameter: float
    visibility: str

    def to_dict(self) -> dict[str, object]:
        return {
            "objectId": self.object_id,
            "startParameter": self.start_parameter,
            "endParameter": self.end_parameter,
            "visibility": self.visibility,
        }


@dataclass(frozen=True)
class LegacyRelationProof3D:
    relation_id: str
    source_edge_id: str
    authored_vertex_ids: tuple[str, str]
    canonical_vertex_ids: tuple[str, str]
    fragments: tuple[LegacyRelationFragment3D, ...]

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.fragments)

    def to_dict(self) -> dict[str, object]:
        return {
            "relationId": self.relation_id,
            "sourceEdgeId": self.source_edge_id,
            "authoredVertexIds": list(self.authored_vertex_ids),
            "canonicalVertexIds": list(self.canonical_vertex_ids),
            "fragments": [item.to_dict() for item in self.fragments],
        }


@dataclass(frozen=True)
class LegacyTopologyDiscovery3D:
    """Geometry-free subset of the frozen adapter result.

    A seam may be exactly flat at entry.  In that case the frozen independent-
    face adapter would weld coincident names or merge the two faces.  We use a
    private, deterministically perturbed copy only to discover stable authored
    topology, then immediately discard its positions and trace.  This value is
    the only legacy evidence exposed by the open-face adapter.
    """

    face_bindings: tuple[FaceBinding3D, ...]
    stroke_bindings: tuple[StrokeBinding3D, ...]
    stroke_specs: tuple[StrokeSpec, ...]
    coordinate_vertex_ids: tuple[tuple[str, str], ...]
    suppressed_object_ids: tuple[str, ...]
    unmanaged_object_ids: tuple[str, ...]
    entry_projection: tuple[tuple[float, float, float], ...]
    result_sha256: str
    schema: str = LEGACY_TOPOLOGY_DISCOVERY_SCHEMA

    @property
    def coordinate_vertex_map(self) -> Mapping[str, str]:
        return dict(self.coordinate_vertex_ids)

    def _payload(self, *, include_result_hash: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "faceBindings": [item.to_dict() for item in self.face_bindings],
            "strokeBindings": [item.to_dict() for item in self.stroke_bindings],
            "strokeSpecs": [item.to_dict() for item in self.stroke_specs],
            "coordinateVertexIds": {
                authored: canonical
                for authored, canonical in self.coordinate_vertex_ids
            },
            "suppressedObjectIds": list(self.suppressed_object_ids),
            "unmanagedObjectIds": list(self.unmanaged_object_ids),
            "entryProjection": [list(row) for row in self.entry_projection],
        }
        if include_result_hash:
            payload["resultSha256"] = self.result_sha256
        return payload

    def to_dict(self) -> dict[str, object]:
        return self._payload(include_result_hash=True)


@dataclass(frozen=True)
class TikzNativeOpenFaceVisibility3DAdapterResult:
    legacy_analysis: LegacyTopologyDiscovery3D
    model: OpenFaceVisibilityModel
    relation_proofs: tuple[LegacyRelationProof3D, ...]
    entry_trace: OpenFaceVisibilityFrame
    model_sha256: str
    entry_trace_sha256: str
    result_sha256: str
    schema: str = OPEN_FACE_ADAPTER_RESULT_SCHEMA

    @property
    def face_bindings(self) -> tuple[FaceBinding3D, ...]:
        return self.legacy_analysis.face_bindings

    @property
    def stroke_bindings(self) -> tuple[StrokeBinding3D, ...]:
        return self.legacy_analysis.stroke_bindings

    @property
    def coordinate_vertex_ids(self) -> tuple[tuple[str, str], ...]:
        return self.legacy_analysis.coordinate_vertex_ids

    @property
    def coordinate_vertex_map(self) -> Mapping[str, str]:
        return self.legacy_analysis.coordinate_vertex_map

    @property
    def suppressed_object_ids(self) -> tuple[str, ...]:
        return self.legacy_analysis.suppressed_object_ids

    @property
    def unmanaged_object_ids(self) -> tuple[str, ...]:
        return self.legacy_analysis.unmanaged_object_ids

    @property
    def entry_projection(self) -> tuple[tuple[float, float, float], ...]:
        return self.legacy_analysis.entry_projection

    @property
    def relation_proof_map(self) -> Mapping[str, LegacyRelationProof3D]:
        return {item.relation_id: item for item in self.relation_proofs}

    def _payload(self, *, include_result_hash: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "legacyAnalysisSha256": self.legacy_analysis.result_sha256,
            "model": self.model.to_dict(),
            "relationProofs": [item.to_dict() for item in self.relation_proofs],
            "suppressedObjectIds": list(self.suppressed_object_ids),
            "unmanagedObjectIds": list(self.unmanaged_object_ids),
            "entryProjection": [list(row) for row in self.entry_projection],
            "entryTrace": self.entry_trace.to_dict(),
            "modelSha256": self.model_sha256,
            "entryTraceSha256": self.entry_trace_sha256,
        }
        if include_result_hash:
            payload["resultSha256"] = self.result_sha256
        return payload

    def to_dict(self) -> dict[str, object]:
        return self._payload(include_result_hash=True)


class TikzNativeOpenFaceVisibility3DAdapterError(ValueError):
    """Stable fail-closed error for TikZ open-face evidence."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _rotate_about_axis(
    point: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    angle: float,
) -> np.ndarray:
    direction = axis_end - axis_start
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        raise TikzNativeOpenFaceVisibility3DAdapterError(
            "UNPROVEN_HINGE", "hinge topology probe has a zero-length axis"
        )
    direction /= length
    relative = point - axis_start
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return axis_start + (
        relative * cosine
        + np.cross(direction, relative) * sine
        + direction * float(np.dot(direction, relative)) * (1.0 - cosine)
    )


def _legacy_topology_probe_picture(
    picture: PictureSpec,
    *,
    tolerance_policy: TolerancePolicy,
) -> PictureSpec:
    """Return a private seam-open copy for frozen topology discovery only."""

    if not picture.hinge_relations:
        return picture
    probe = copy.deepcopy(picture)
    hinge_memberships: dict[str, set[str]] = {}
    for hinge in picture.hinge_relations:
        for name in {
            *hinge.axis_names,
            *hinge.fixed_face_names,
            *hinge.moving_face_names,
        }:
            hinge_memberships.setdefault(name, set()).add(hinge.id)

    claimed_moving_names: set[str] = set()
    for hinge in sorted(picture.hinge_relations, key=lambda item: item.id):
        if len(hinge.axis_names) != 2:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "UNPROVEN_HINGE",
                f"hinge {hinge.id} must have exactly two axis coordinates",
            )
        axis_names = tuple(hinge.axis_names)
        fixed_names = set(hinge.fixed_face_names)
        moving_names = set(hinge.moving_face_names)
        axis_set = set(axis_names)
        if fixed_names & moving_names != axis_set:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "UNPROVEN_HINGE",
                f"hinge {hinge.id} faces must share exactly their declared axis",
            )
        unknown = sorted((fixed_names | moving_names | axis_set) - set(picture.coordinates))
        if unknown:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "UNPROVEN_HINGE",
                f"hinge {hinge.id} references unknown coordinates: {', '.join(unknown)}",
            )
        non_axis_moving = moving_names - axis_set
        overlap = claimed_moving_names & non_axis_moving
        if overlap:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "AMBIGUOUS_HINGE_TOPOLOGY",
                "multiple hinges claim the same moving coordinates: "
                + ", ".join(sorted(overlap)),
            )
        cross_hinge = sorted(
            name
            for name in non_axis_moving
            if hinge_memberships.get(name, {hinge.id}) != {hinge.id}
        )
        if cross_hinge:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "AMBIGUOUS_HINGE_TOPOLOGY",
                "a moving coordinate participates in another hinge: "
                + ", ".join(cross_hinge),
            )
        claimed_moving_names.update(non_axis_moving)

        axis_start = _point3(picture.coordinates[axis_names[0]], axis_names[0])
        axis_end = _point3(picture.coordinates[axis_names[1]], axis_names[1])
        resolved = tolerance_policy.resolve((axis_start, axis_end))
        if float(np.linalg.norm(axis_end - axis_start)) <= resolved.world:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "UNPROVEN_HINGE", f"hinge {hinge.id} has a zero-length axis"
            )
        for name in sorted(non_axis_moving):
            point = _point3(picture.coordinates[name], name)
            rotated = _rotate_about_axis(
                point,
                axis_start,
                axis_end,
                _HINGE_TOPOLOGY_PROBE_RADIANS,
            )
            if float(np.linalg.norm(rotated - point)) <= resolved.world:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "UNPROVEN_HINGE",
                    f"hinge {hinge.id} moving vertex {name} lies on its axis",
                )
            probe.coordinates[name] = tuple(float(value) for value in rotated)
    return probe


def _legacy_topology_discovery(
    result: TikzNativeVisibility3DAdapterResult,
) -> LegacyTopologyDiscovery3D:
    provisional = LegacyTopologyDiscovery3D(
        face_bindings=result.face_bindings,
        stroke_bindings=result.stroke_bindings,
        stroke_specs=result.model.strokes,
        coordinate_vertex_ids=result.coordinate_vertex_ids,
        suppressed_object_ids=result.suppressed_object_ids,
        unmanaged_object_ids=result.unmanaged_object_ids,
        entry_projection=result.entry_projection,
        result_sha256="",
    )
    return LegacyTopologyDiscovery3D(
        face_bindings=provisional.face_bindings,
        stroke_bindings=provisional.stroke_bindings,
        stroke_specs=provisional.stroke_specs,
        coordinate_vertex_ids=provisional.coordinate_vertex_ids,
        suppressed_object_ids=provisional.suppressed_object_ids,
        unmanaged_object_ids=provisional.unmanaged_object_ids,
        entry_projection=provisional.entry_projection,
        result_sha256=_sha256(provisional._payload(include_result_hash=False)),
    )


def _relation_stroke_map(
    analysis: LegacyTopologyDiscovery3D,
) -> dict[str, StrokeBinding3D]:
    result: dict[str, StrokeBinding3D] = {}
    for stroke in analysis.stroke_bindings:
        for relation_id in stroke.relation_ids:
            if relation_id in result:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "AMBIGUOUS_RELATION_BINDING",
                    f"relation {relation_id} belongs to more than one logical stroke",
                )
            result[relation_id] = stroke
    return result


def _prove_relation_fragments(
    picture: PictureSpec,
    analysis: LegacyTopologyDiscovery3D,
    *,
    tolerance_policy: TolerancePolicy,
) -> tuple[LegacyRelationProof3D, ...]:
    objects = {item.id: item for item in picture.objects}
    relation_strokes = _relation_stroke_map(analysis)
    alias_map = analysis.coordinate_vertex_map
    object_owner: dict[str, str] = {}
    proofs: list[LegacyRelationProof3D] = []

    for relation in sorted(picture.occlusion_relations, key=lambda item: item.id):
        stroke = relation_strokes.get(relation.id)
        if stroke is None:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "UNBOUND_RELATION",
                f"legacy relation {relation.id} has no logical stroke binding",
            )
        if relation.start_name not in picture.coordinates or relation.end_name not in picture.coordinates:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "UNPROVEN_RELATION",
                f"relation {relation.id} has unknown authored endpoints",
            )
        if len(set(relation.object_ids)) != len(relation.object_ids) or not relation.object_ids:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "INVALID_FRAGMENT_PARTITION",
                f"relation {relation.id} must own a non-empty unique fragment list",
            )

        full_start = _point3(picture.coordinates[relation.start_name], relation.start_name)
        full_end = _point3(picture.coordinates[relation.end_name], relation.end_name)
        delta = full_end - full_start
        length = float(np.linalg.norm(delta))
        tolerance = tolerance_policy.resolve(
            tuple(picture.coordinates.values()), edge_length=length
        )
        if length <= tolerance.world:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "DEGENERATE_RELATION",
                f"relation {relation.id} has zero length",
            )

        fragments: list[LegacyRelationFragment3D] = []
        for object_id in relation.object_ids:
            previous = object_owner.get(object_id)
            if previous is not None:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "REUSED_RELATION_FRAGMENT",
                    f"fragment {object_id} is shared by {previous} and {relation.id}",
                )
            object_owner[object_id] = relation.id
            item = objects.get(object_id)
            if item is None or item.kind != "line" or item.style.arrow_tip is not None:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "UNPROVEN_RELATION_FRAGMENT",
                    f"relation {relation.id} fragment {object_id} is missing, curved, or arrowed",
                )
            if (
                item.geometry.get("start_name") != relation.start_name
                or item.geometry.get("end_name") != relation.end_name
            ):
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "UNPROVEN_RELATION_FRAGMENT",
                    f"fragment {object_id} does not preserve the authored relation direction",
                )
            if tuple(item.geometry.get("occluding_face", ())) != tuple(relation.face_names):
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "UNPROVEN_RELATION_FRAGMENT",
                    f"fragment {object_id} disagrees with relation {relation.id}'s face",
                )
            raw_range = item.geometry.get("source_parameter_range")
            if not isinstance(raw_range, (tuple, list)) or len(raw_range) != 2:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "INVALID_FRAGMENT_PARTITION",
                    f"fragment {object_id} has no two-value source_parameter_range",
                )
            try:
                first, last = (float(raw_range[0]), float(raw_range[1]))
            except (TypeError, ValueError) as exc:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "INVALID_FRAGMENT_PARTITION",
                    f"fragment {object_id} has a non-numeric source_parameter_range",
                ) from exc
            if (
                not isfinite(first)
                or not isfinite(last)
                or first < -tolerance.parameter
                or last > 1.0 + tolerance.parameter
                or last - first <= tolerance.parameter
            ):
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "INVALID_FRAGMENT_PARTITION",
                    f"fragment {object_id} has invalid parameter interval {first}, {last}",
                )
            first = min(1.0, max(0.0, first))
            last = min(1.0, max(0.0, last))
            expected_start = full_start + first * delta
            expected_end = full_start + last * delta
            actual_start = _point3(item.geometry.get("start"), f"fragment {object_id} start")
            actual_end = _point3(item.geometry.get("end"), f"fragment {object_id} end")
            if (
                float(np.linalg.norm(actual_start - expected_start)) > tolerance.boundary
                or float(np.linalg.norm(actual_end - expected_end)) > tolerance.boundary
            ):
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "NONCOLLINEAR_RELATION_FRAGMENT",
                    f"fragment {object_id} does not match its declared full-line interval",
                )
            visibility = item.geometry.get("visibility")
            if visibility not in {"visible", "hidden"}:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "INVALID_FRAGMENT_VISIBILITY",
                    f"fragment {object_id} has unsupported visibility {visibility!r}",
                )
            expected_style = (
                _style_payload(relation.visible_style)
                if visibility == "visible"
                else _style_payload(relation.hidden_style)
            )
            if _style_payload(item.style) != expected_style:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "FRAGMENT_STYLE_MISMATCH",
                    f"fragment {object_id} style disagrees with relation {relation.id}",
                )
            fragments.append(
                LegacyRelationFragment3D(object_id, first, last, str(visibility))
            )

        fragments.sort(key=lambda item: (item.start_parameter, item.end_parameter, item.object_id))
        if abs(fragments[0].start_parameter) > tolerance.parameter:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "INVALID_FRAGMENT_PARTITION",
                f"relation {relation.id} fragments do not begin at 0",
            )
        for previous, current in zip(fragments, fragments[1:]):
            if abs(previous.end_parameter - current.start_parameter) > tolerance.parameter:
                kind = (
                    "overlap"
                    if current.start_parameter < previous.end_parameter
                    else "gap"
                )
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "INVALID_FRAGMENT_PARTITION",
                    f"relation {relation.id} fragments contain a {kind}",
                )
        if abs(fragments[-1].end_parameter - 1.0) > tolerance.parameter:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "INVALID_FRAGMENT_PARTITION",
                f"relation {relation.id} fragments do not end at 1",
            )
        proofs.append(
            LegacyRelationProof3D(
                relation.id,
                stroke.source_edge_id,
                (relation.start_name, relation.end_name),
                (alias_map[relation.start_name], alias_map[relation.end_name]),
                tuple(fragments),
            )
        )

    proofs_by_edge: dict[str, list[LegacyRelationProof3D]] = {}
    for proof in proofs:
        proofs_by_edge.setdefault(proof.source_edge_id, []).append(proof)
    for stroke in analysis.stroke_bindings:
        edge_proofs = proofs_by_edge.get(stroke.source_edge_id, [])
        proved_objects = {
            object_id for proof in edge_proofs for object_id in proof.object_ids
        }
        if stroke.relation_ids:
            if set(stroke.relation_ids) != {proof.relation_id for proof in edge_proofs}:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "INCOMPLETE_RELATION_PROOF",
                    f"logical stroke {stroke.source_edge_id} has incomplete relation evidence",
                )
            if set(stroke.object_ids) != proved_objects:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "AMBIGUOUS_STROKE_SOURCES",
                    f"logical stroke {stroke.source_edge_id} mixes proven fragments with other objects",
                )
        elif len(stroke.object_ids) != 1 or stroke.source_kind != "named_line":
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "AMBIGUOUS_STROKE_SOURCES",
                f"plain logical stroke {stroke.source_edge_id} must own exactly one named line",
            )
    return tuple(sorted(proofs, key=lambda item: item.relation_id))


def _face_id_by_authored_cycle(
    analysis: LegacyTopologyDiscovery3D,
    alias_map: Mapping[str, str],
) -> Mapping[tuple[str, ...], str]:
    result: dict[tuple[str, ...], str] = {}
    for face in analysis.face_bindings:
        cycles = [face.vertex_ids]
        cycles.extend(
            tuple(alias_map[name] for name in authored)
            for authored in face.authored_cycles
        )
        for cycle in cycles:
            key = _canonical_cycle(cycle)
            previous = result.get(key)
            if previous is not None and previous != face.face_id:
                raise TikzNativeOpenFaceVisibility3DAdapterError(
                    "AMBIGUOUS_HINGE_FACE",
                    f"face cycle {cycle!r} matches both {previous} and {face.face_id}",
                )
            result[key] = face.face_id
    return result


def _seams(
    picture: PictureSpec,
    analysis: LegacyTopologyDiscovery3D,
) -> tuple[OpenFaceSeamSpec, ...]:
    alias_map = analysis.coordinate_vertex_map
    faces = _face_id_by_authored_cycle(analysis, alias_map)
    result: list[OpenFaceSeamSpec] = []
    for hinge in sorted(picture.hinge_relations, key=lambda item: item.id):
        try:
            fixed_cycle = tuple(alias_map[name] for name in hinge.fixed_face_names)
            moving_cycle = tuple(alias_map[name] for name in hinge.moving_face_names)
            axis = tuple(alias_map[name] for name in hinge.axis_names)
        except KeyError as exc:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "UNPROVEN_HINGE",
                f"hinge {hinge.id} references unknown coordinate {exc.args[0]}",
            ) from exc
        if len(axis) != 2 or axis[0] == axis[1]:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "UNPROVEN_HINGE", f"hinge {hinge.id} needs two distinct axis vertices"
            )
        fixed_id = faces.get(_canonical_cycle(fixed_cycle))
        moving_id = faces.get(_canonical_cycle(moving_cycle))
        if fixed_id is None or moving_id is None or fixed_id == moving_id:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "UNPROVEN_HINGE",
                f"hinge {hinge.id} does not name two distinct proven finite faces",
            )
        result.append(
            OpenFaceSeamSpec(
                seam_id=hinge.id,
                policy=ARTICULATED_HINGE_POLICY,
                face_ids=(fixed_id, moving_id),
                vertex_ids=(axis[0], axis[1]),
            )
        )
    return tuple(result)


def _face_normal(points: np.ndarray, tolerance: float) -> np.ndarray:
    origin = points[0]
    for index in range(1, len(points) - 1):
        normal = np.cross(points[index] - origin, points[index + 1] - origin)
        length = float(np.linalg.norm(normal))
        if length > tolerance * tolerance:
            return normal / length
    raise TikzNativeOpenFaceVisibility3DAdapterError(
        "INVALID_OWNER_FACE", "coplanar owner face is degenerate"
    )


def _point_in_convex_face(
    point: np.ndarray,
    polygon: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    normal = _face_normal(polygon, tolerance)
    if abs(float(np.dot(point - polygon[0], normal))) > tolerance:
        return False
    signs: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        value = float(np.dot(np.cross(end - start, point - start), normal))
        threshold = tolerance * max(float(np.linalg.norm(end - start)), tolerance)
        if abs(value) > threshold:
            signs.append(value)
    return not signs or min(signs) >= 0.0 or max(signs) <= 0.0


def _owned_coplanar_faces(
    vertex_ids: tuple[str, str],
    incident_face_ids: Sequence[str],
    faces: Sequence[OpenFaceSpec],
    positions: Mapping[str, tuple[float, float, float]],
    *,
    tolerance_policy: TolerancePolicy,
) -> tuple[str, ...]:
    start = np.asarray(positions[vertex_ids[0]], dtype=float)
    end = np.asarray(positions[vertex_ids[1]], dtype=float)
    excluded: list[str] = []
    incidents = set(incident_face_ids)
    for face in faces:
        if face.face_id in incidents:
            continue
        polygon = np.asarray([positions[name] for name in face.vertex_ids], dtype=float)
        tolerance = tolerance_policy.resolve(polygon).boundary
        if _point_in_convex_face(start, polygon, tolerance=tolerance) and _point_in_convex_face(
            end, polygon, tolerance=tolerance
        ):
            excluded.append(face.face_id)
    return tuple(sorted(excluded))


def _oriented_stroke_vertices(
    binding: StrokeBinding3D,
    proofs: Sequence[LegacyRelationProof3D],
    alias_map: Mapping[str, str],
) -> tuple[str, str]:
    """Keep one stable authored direction as the global dash-phase origin."""

    if proofs:
        directions = {item.canonical_vertex_ids for item in proofs}
        if len(directions) != 1:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "AMBIGUOUS_STROKE_DIRECTION",
                f"logical stroke {binding.source_edge_id} has conflicting authored directions",
            )
        result = next(iter(directions))
    else:
        if len(binding.authored_vertex_pairs) != 1:
            raise TikzNativeOpenFaceVisibility3DAdapterError(
                "AMBIGUOUS_STROKE_DIRECTION",
                f"plain stroke {binding.source_edge_id} has no unique authored direction",
            )
        authored = binding.authored_vertex_pairs[0]
        result = (alias_map[authored[0]], alias_map[authored[1]])
    if set(result) != set(binding.vertex_ids) or result[0] == result[1]:
        raise TikzNativeOpenFaceVisibility3DAdapterError(
            "AMBIGUOUS_STROKE_DIRECTION",
            f"logical stroke {binding.source_edge_id} direction disagrees with its endpoints",
        )
    return result


def _result(
    legacy: LegacyTopologyDiscovery3D,
    model: OpenFaceVisibilityModel,
    proofs: tuple[LegacyRelationProof3D, ...],
    trace: OpenFaceVisibilityFrame,
) -> TikzNativeOpenFaceVisibility3DAdapterResult:
    model_hash = _sha256(model.to_dict())
    trace_hash = hashlib.sha256(
        canonical_open_face_trace_json(trace).encode("utf-8")
    ).hexdigest()
    provisional = TikzNativeOpenFaceVisibility3DAdapterResult(
        legacy,
        model,
        proofs,
        trace,
        model_hash,
        trace_hash,
        "",
    )
    return TikzNativeOpenFaceVisibility3DAdapterResult(
        legacy,
        model,
        proofs,
        trace,
        model_hash,
        trace_hash,
        _sha256(provisional._payload(include_result_hash=False)),
    )


def adapt_picture_open_face_visibility_3d(
    picture: PictureSpec,
    *,
    default_hidden_style: Mapping[str, object] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> TikzNativeOpenFaceVisibility3DAdapterResult:
    """Upgrade one proven independent-face analysis using the fixed v1 tolerance."""

    # The frozen discovery adapter has a fixed default numerical contract.
    # Reuse exactly that default through fragment proof and open-face solving;
    # exposing a second tolerance here would make the two proof phases disagree.
    policy = TolerancePolicy()
    topology_picture = _legacy_topology_probe_picture(
        picture,
        tolerance_policy=policy,
    )
    try:
        frozen_legacy = adapt_picture_visibility_3d(
            topology_picture,
            validation_mode="independent_convex_faces",
            default_hidden_style=default_hidden_style,
            overrides=overrides,
        )
    except TikzNativeVisibility3DAdapterError as exc:
        raise TikzNativeOpenFaceVisibility3DAdapterError(exc.code, str(exc)) from exc
    legacy = _legacy_topology_discovery(frozen_legacy)

    proofs = _prove_relation_fragments(
        picture,
        legacy,
        tolerance_policy=policy,
    )
    faces = tuple(
        OpenFaceSpec(
            face.face_id,
            face.face_id,
            face.vertex_ids,
            face.occludes_strokes,
        )
        for face in legacy.face_bindings
    )
    proofs_by_edge: dict[str, list[LegacyRelationProof3D]] = {}
    for proof in proofs:
        proofs_by_edge.setdefault(proof.source_edge_id, []).append(proof)
    legacy_strokes = {item.source_edge_id: item for item in legacy.stroke_specs}
    used_vertex_ids = sorted(
        {
            vertex_id
            for face in faces
            for vertex_id in face.vertex_ids
        }
        | {
            vertex_id
            for binding in legacy.stroke_bindings
            for vertex_id in binding.vertex_ids
        }
    )
    entry_positions = {
        vertex_id: tuple(
            float(value)
            for value in _point3(
                picture.coordinates.get(vertex_id),
                f"entry coordinate {vertex_id}",
            )
        )
        for vertex_id in used_vertex_ids
    }
    strokes = tuple(
        OpenFaceStrokeSpec(
            source_edge_id=binding.source_edge_id,
            vertex_ids=_oriented_stroke_vertices(
                binding,
                proofs_by_edge.get(binding.source_edge_id, ()),
                legacy.coordinate_vertex_map,
            ),
            incident_face_ids=legacy_strokes[binding.source_edge_id].incident_face_ids,
            excluded_occluder_face_ids=_owned_coplanar_faces(
                _oriented_stroke_vertices(
                    binding,
                    proofs_by_edge.get(binding.source_edge_id, ()),
                    legacy.coordinate_vertex_map,
                ),
                legacy_strokes[binding.source_edge_id].incident_face_ids,
                faces,
                entry_positions,
                tolerance_policy=policy,
            ),
            render_binding_id=legacy_strokes[binding.source_edge_id].render_binding_id,
            visibility_mode=legacy_strokes[binding.source_edge_id].visibility_mode,
        )
        for binding in legacy.stroke_bindings
    )
    model = OpenFaceVisibilityModel(
        visibility_group_id=(
            f"tikz-native-picture-{picture.index}-open-face-visibility-3d"
        ),
        vertices=tuple(
            OpenFaceVertexSpec(vertex_id, entry_positions[vertex_id])
            for vertex_id in used_vertex_ids
        ),
        faces=faces,
        seams=_seams(picture, legacy),
        strokes=strokes,
    )
    try:
        model.validate(tolerance_policy=policy)
        trace = compute_open_face_visibility(
            model,
            projection_matrix=legacy.entry_projection,
            tolerance_policy=policy,
        )
    except (OpenFaceContractError, OpenFaceSolverError) as exc:
        code = getattr(exc, "code", "OPEN_FACE_MODEL_FAILED")
        raise TikzNativeOpenFaceVisibility3DAdapterError(code, str(exc)) from exc
    return _result(legacy, model, proofs, trace)


__all__ = [
    "LegacyRelationFragment3D",
    "LegacyRelationProof3D",
    "OPEN_FACE_ADAPTER_RESULT_SCHEMA",
    "TikzNativeOpenFaceVisibility3DAdapterError",
    "TikzNativeOpenFaceVisibility3DAdapterResult",
    "adapt_picture_open_face_visibility_3d",
]
