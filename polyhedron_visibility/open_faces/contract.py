from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from ..contract import (
    ContractError as FrozenContractError,
    ResolvedTolerance,
    TolerancePolicy,
    _validate_convex_face_points,
)


OPEN_FACE_MODEL_SCHEMA = "manim-open-convex-face-visibility/v1"
OPEN_FACE_TOPOLOGY = "finite_independent_convex_faces"
ARTICULATED_HINGE_POLICY = "articulated_hinge"


class OpenFaceContractError(ValueError):
    """A stable fail-closed error for the open-face visibility contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise OpenFaceContractError(code, message)


def _strict_keys(payload: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        _fail("UNKNOWN_FIELD", f"{label} contains unknown fields: {', '.join(unknown)}")


def _required_text(payload: Mapping[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail("INVALID_IDENTITY", f"{label}.{key} must be a non-empty string")
    return value.strip()


def _text_tuple(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    exact: int | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        _fail("INVALID_ARRAY", f"{label} must be an array of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            _fail("INVALID_IDENTITY", f"{label}[{index}] must be a non-empty string")
        result.append(item.strip())
    if len(result) < minimum:
        _fail("INVALID_ARRAY", f"{label} must contain at least {minimum} items")
    if exact is not None and len(result) != exact:
        _fail("INVALID_ARRAY", f"{label} must contain exactly {exact} items")
    if len(set(result)) != len(result):
        _fail("DUPLICATE_IDENTITY", f"{label} contains duplicate identities")
    return tuple(result)


def _point3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        _fail("INVALID_POINT", f"{label} must be a three-component point")
    try:
        point = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise OpenFaceContractError(
            "INVALID_POINT", f"{label} must contain numeric components"
        ) from exc
    if not all(isfinite(component) for component in point):
        _fail("INVALID_POINT", f"{label} components must be finite")
    return point  # type: ignore[return-value]


def _unique(items: Sequence[object], attribute: str, label: str) -> None:
    seen: set[str] = set()
    for item in items:
        identity = str(getattr(item, attribute))
        if identity in seen:
            _fail("DUPLICATE_IDENTITY", f"duplicate {label}: {identity}")
        seen.add(identity)


@dataclass(frozen=True)
class OpenFaceVertexSpec:
    vertex_id: str
    entry_position: tuple[float, float, float]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OpenFaceVertexSpec":
        _strict_keys(payload, {"vertexId", "entryPosition"}, "vertex")
        vertex_id = _required_text(payload, "vertexId", "vertex")
        return cls(vertex_id, _point3(payload.get("entryPosition"), f"vertex {vertex_id}"))

    def to_dict(self) -> dict[str, object]:
        return {"vertexId": self.vertex_id, "entryPosition": list(self.entry_position)}


@dataclass(frozen=True)
class OpenFaceSpec:
    face_id: str
    logical_surface_id: str
    vertex_ids: tuple[str, ...]
    occludes_strokes: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OpenFaceSpec":
        _strict_keys(
            payload,
            {"faceId", "logicalSurfaceId", "vertexIds", "occludesStrokes"},
            "face",
        )
        face_id = _required_text(payload, "faceId", "face")
        logical_surface_id = _required_text(payload, "logicalSurfaceId", f"face {face_id}")
        occludes = payload.get("occludesStrokes", True)
        if not isinstance(occludes, bool):
            _fail("INVALID_FACE", f"face {face_id}.occludesStrokes must be boolean")
        return cls(
            face_id,
            logical_surface_id,
            _text_tuple(payload.get("vertexIds"), f"face {face_id}.vertexIds", minimum=3),
            occludes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "faceId": self.face_id,
            "logicalSurfaceId": self.logical_surface_id,
            "vertexIds": list(self.vertex_ids),
            "occludesStrokes": self.occludes_strokes,
        }


@dataclass(frozen=True)
class OpenFaceSeamSpec:
    seam_id: str
    policy: str
    face_ids: tuple[str, str]
    vertex_ids: tuple[str, str]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OpenFaceSeamSpec":
        _strict_keys(payload, {"seamId", "policy", "faceIds", "vertexIds"}, "seam")
        seam_id = _required_text(payload, "seamId", "seam")
        policy = _required_text(payload, "policy", f"seam {seam_id}")
        face_ids = _text_tuple(payload.get("faceIds"), f"seam {seam_id}.faceIds", exact=2)
        vertex_ids = _text_tuple(
            payload.get("vertexIds"), f"seam {seam_id}.vertexIds", exact=2
        )
        return cls(
            seam_id,
            policy,
            tuple(sorted((face_ids[0], face_ids[1]))),
            (vertex_ids[0], vertex_ids[1]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seamId": self.seam_id,
            "policy": self.policy,
            "faceIds": list(self.face_ids),
            "vertexIds": list(self.vertex_ids),
        }


@dataclass(frozen=True)
class OpenFaceStrokeSpec:
    source_edge_id: str
    vertex_ids: tuple[str, str]
    incident_face_ids: tuple[str, ...] = ()
    excluded_occluder_face_ids: tuple[str, ...] = ()
    render_binding_id: str | None = None
    visibility_mode: str = "auto"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OpenFaceStrokeSpec":
        _strict_keys(
            payload,
            {
                "sourceEdgeId",
                "vertexIds",
                "incidentFaceIds",
                "excludedOccluderFaceIds",
                "renderBindingId",
                "visibilityMode",
            },
            "stroke",
        )
        edge_id = _required_text(payload, "sourceEdgeId", "stroke")
        vertex_ids = _text_tuple(
            payload.get("vertexIds"), f"stroke {edge_id}.vertexIds", exact=2
        )
        incidents = _text_tuple(
            payload.get("incidentFaceIds", []), f"stroke {edge_id}.incidentFaceIds"
        )
        excluded = _text_tuple(
            payload.get("excludedOccluderFaceIds", []),
            f"stroke {edge_id}.excludedOccluderFaceIds",
        )
        if set(incidents) & set(excluded):
            _fail(
                "REDUNDANT_OCCLUDER_EXCLUSION",
                f"stroke {edge_id} cannot mark one face both incident and excluded",
            )
        binding = payload.get("renderBindingId")
        if binding is not None and (not isinstance(binding, str) or not binding.strip()):
            _fail(
                "INVALID_BINDING_ID",
                f"stroke {edge_id}.renderBindingId must be a non-empty string",
            )
        mode = payload.get("visibilityMode", "auto")
        if mode not in {"auto", "always_visible", "always_hidden"}:
            _fail("INVALID_VISIBILITY_MODE", f"stroke {edge_id}.visibilityMode is unsupported")
        return cls(
            edge_id,
            (vertex_ids[0], vertex_ids[1]),
            tuple(sorted(incidents)),
            tuple(sorted(excluded)),
            binding.strip() if isinstance(binding, str) else None,
            str(mode),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "sourceEdgeId": self.source_edge_id,
            "vertexIds": list(self.vertex_ids),
            "incidentFaceIds": list(self.incident_face_ids),
            "excludedOccluderFaceIds": list(self.excluded_occluder_face_ids),
            "visibilityMode": self.visibility_mode,
        }
        if self.render_binding_id is not None:
            result["renderBindingId"] = self.render_binding_id
        return result


@dataclass(frozen=True)
class _ValidatedOpenFaceFrame:
    positions: Mapping[str, np.ndarray]
    face_normals: Mapping[str, np.ndarray]
    face_tolerances: Mapping[str, ResolvedTolerance]


@dataclass(frozen=True)
class OpenFaceVisibilityModel:
    visibility_group_id: str
    vertices: tuple[OpenFaceVertexSpec, ...]
    faces: tuple[OpenFaceSpec, ...]
    seams: tuple[OpenFaceSeamSpec, ...]
    strokes: tuple[OpenFaceStrokeSpec, ...]
    topology: str = OPEN_FACE_TOPOLOGY
    schema: str = OPEN_FACE_MODEL_SCHEMA

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OpenFaceVisibilityModel":
        if not isinstance(payload, Mapping):
            _fail("INVALID_MODEL", "open-face visibility model must be an object")
        _strict_keys(
            payload,
            {
                "schema",
                "topology",
                "visibilityGroupId",
                "vertices",
                "faces",
                "seams",
                "strokes",
            },
            "open-face visibility model",
        )
        if payload.get("schema") != OPEN_FACE_MODEL_SCHEMA:
            _fail("INVALID_SCHEMA", f"model schema must be {OPEN_FACE_MODEL_SCHEMA}")
        if payload.get("topology") != OPEN_FACE_TOPOLOGY:
            _fail("INVALID_TOPOLOGY", f"model topology must be {OPEN_FACE_TOPOLOGY}")
        group_id = _required_text(payload, "visibilityGroupId", "open-face visibility model")
        raw_vertices = payload.get("vertices")
        raw_faces = payload.get("faces")
        raw_seams = payload.get("seams")
        raw_strokes = payload.get("strokes")
        if not all(isinstance(value, list) for value in (raw_vertices, raw_faces, raw_seams, raw_strokes)):
            _fail("INVALID_ARRAY", "vertices, faces, seams, and strokes must be arrays")
        assert isinstance(raw_vertices, list)
        assert isinstance(raw_faces, list)
        assert isinstance(raw_seams, list)
        assert isinstance(raw_strokes, list)
        if not all(isinstance(item, Mapping) for item in (*raw_vertices, *raw_faces, *raw_seams, *raw_strokes)):
            _fail("INVALID_ARRAY", "model arrays must contain objects")
        model = cls(
            visibility_group_id=group_id,
            vertices=tuple(
                sorted(
                    (OpenFaceVertexSpec.from_dict(item) for item in raw_vertices),
                    key=lambda item: item.vertex_id,
                )
            ),
            faces=tuple(
                sorted(
                    (OpenFaceSpec.from_dict(item) for item in raw_faces),
                    key=lambda item: item.face_id,
                )
            ),
            seams=tuple(
                sorted(
                    (OpenFaceSeamSpec.from_dict(item) for item in raw_seams),
                    key=lambda item: item.seam_id,
                )
            ),
            strokes=tuple(
                sorted(
                    (OpenFaceStrokeSpec.from_dict(item) for item in raw_strokes),
                    key=lambda item: item.source_edge_id,
                )
            ),
        )
        model._validate_static_contract()
        return model

    @property
    def vertex_map(self) -> dict[str, OpenFaceVertexSpec]:
        return {item.vertex_id: item for item in self.vertices}

    @property
    def face_map(self) -> dict[str, OpenFaceSpec]:
        return {item.face_id: item for item in self.faces}

    @property
    def seam_map(self) -> dict[str, OpenFaceSeamSpec]:
        return {item.seam_id: item for item in self.seams}

    @property
    def stroke_map(self) -> dict[str, OpenFaceStrokeSpec]:
        return {item.source_edge_id: item for item in self.strokes}

    @property
    def entry_positions(self) -> dict[str, tuple[float, float, float]]:
        return {item.vertex_id: item.entry_position for item in self.vertices}

    @staticmethod
    def _boundary_edges(face: OpenFaceSpec) -> set[tuple[str, str]]:
        return {
            tuple(sorted((start, face.vertex_ids[(index + 1) % len(face.vertex_ids)])))
            for index, start in enumerate(face.vertex_ids)
        }

    def _validate_static_contract(self) -> None:
        if self.schema != OPEN_FACE_MODEL_SCHEMA:
            _fail("INVALID_SCHEMA", f"model schema must be {OPEN_FACE_MODEL_SCHEMA}")
        if self.topology != OPEN_FACE_TOPOLOGY:
            _fail("INVALID_TOPOLOGY", f"model topology must be {OPEN_FACE_TOPOLOGY}")
        if not isinstance(self.visibility_group_id, str) or not self.visibility_group_id.strip():
            _fail("INVALID_IDENTITY", "visibilityGroupId must be a non-empty string")
        _unique(self.vertices, "vertex_id", "vertexId")
        _unique(self.faces, "face_id", "faceId")
        _unique(self.seams, "seam_id", "seamId")
        _unique(self.strokes, "source_edge_id", "sourceEdgeId")
        if not self.faces:
            _fail("NO_FACES", "open-face visibility requires at least one finite face")

        vertex_ids: set[str] = set()
        for vertex in self.vertices:
            if not isinstance(vertex.vertex_id, str) or not vertex.vertex_id.strip():
                _fail("INVALID_IDENTITY", "vertexId must be a non-empty string")
            _point3(vertex.entry_position, f"vertex {vertex.vertex_id}")
            vertex_ids.add(vertex.vertex_id)

        logical_surface_owners: dict[str, str] = {}
        for face in self.faces:
            if not isinstance(face.face_id, str) or not face.face_id.strip():
                _fail("INVALID_IDENTITY", "faceId must be a non-empty string")
            if not isinstance(face.logical_surface_id, str) or not face.logical_surface_id.strip():
                _fail("INVALID_IDENTITY", f"face {face.face_id}.logicalSurfaceId must be non-empty")
            previous = logical_surface_owners.get(face.logical_surface_id)
            if previous is not None:
                _fail(
                    "MULTIPLE_FACES_PER_LOGICAL_SURFACE",
                    "v1 requires one maximal convex face per logical surface: "
                    f"{previous}, {face.face_id}",
                )
            logical_surface_owners[face.logical_surface_id] = face.face_id
            if len(face.vertex_ids) < 3 or len(set(face.vertex_ids)) != len(face.vertex_ids):
                _fail("INVALID_FACE", f"face {face.face_id} must contain unique vertices")
            missing = sorted(set(face.vertex_ids) - vertex_ids)
            if missing:
                _fail(
                    "MISSING_VERTEX",
                    f"face {face.face_id} references missing vertices: {', '.join(missing)}",
                )
            if not isinstance(face.occludes_strokes, bool):
                _fail("INVALID_FACE", f"face {face.face_id}.occludesStrokes must be boolean")

        face_map = self.face_map
        seam_keys: dict[tuple[tuple[str, str], tuple[str, str]], str] = {}
        seam_face_pairs: dict[tuple[str, str], str] = {}
        for seam in self.seams:
            if not isinstance(seam.seam_id, str) or not seam.seam_id.strip():
                _fail("INVALID_IDENTITY", "seamId must be a non-empty string")
            if seam.policy != ARTICULATED_HINGE_POLICY:
                _fail("INVALID_SEAM_POLICY", f"seam {seam.seam_id} policy is unsupported")
            if len(seam.face_ids) != 2 or seam.face_ids[0] == seam.face_ids[1]:
                _fail("INVALID_SEAM", f"seam {seam.seam_id} must join two distinct faces")
            if len(seam.vertex_ids) != 2 or seam.vertex_ids[0] == seam.vertex_ids[1]:
                _fail("INVALID_SEAM", f"seam {seam.seam_id} must use two distinct vertices")
            missing_faces = sorted(set(seam.face_ids) - set(face_map))
            if missing_faces:
                _fail(
                    "MISSING_FACE",
                    f"seam {seam.seam_id} references missing faces: {', '.join(missing_faces)}",
                )
            missing_vertices = sorted(set(seam.vertex_ids) - vertex_ids)
            if missing_vertices:
                _fail(
                    "MISSING_VERTEX",
                    f"seam {seam.seam_id} references missing vertices: {', '.join(missing_vertices)}",
                )
            first, second = (face_map[item] for item in seam.face_ids)
            if first.logical_surface_id == second.logical_surface_id:
                _fail(
                    "SEAM_WITHIN_LOGICAL_SURFACE",
                    f"seam {seam.seam_id} must join different logical surfaces",
                )
            edge = tuple(sorted(seam.vertex_ids))
            if any(edge not in self._boundary_edges(face) for face in (first, second)):
                _fail(
                    "SEAM_NOT_SHARED_BOUNDARY",
                    f"seam {seam.seam_id} vertices are not one boundary edge of both faces",
                )
            face_pair = tuple(sorted(seam.face_ids))
            key = (face_pair, edge)
            if key in seam_keys:
                _fail("DUPLICATE_SEAM", f"duplicate seam boundary: {seam_keys[key]}, {seam.seam_id}")
            if face_pair in seam_face_pairs:
                _fail(
                    "MULTIPLE_SEAMS_PER_FACE_PAIR",
                    f"faces {face_pair[0]} and {face_pair[1]} share more than one seam",
                )
            seam_keys[key] = seam.seam_id
            seam_face_pairs[face_pair] = seam.seam_id

        boundary_owners: dict[tuple[str, str], list[str]] = {}
        for face in self.faces:
            for edge in self._boundary_edges(face):
                boundary_owners.setdefault(edge, []).append(face.face_id)
        for edge, owners in sorted(boundary_owners.items()):
            if len(owners) == 1:
                continue
            if len(owners) != 2:
                _fail(
                    "NON_MANIFOLD_SHARED_EDGE",
                    f"shared boundary {edge[0]}-{edge[1]} belongs to {len(owners)} faces",
                )
            key = (tuple(sorted(owners)), edge)
            if key not in seam_keys:
                _fail(
                    "UNDECLARED_SHARED_BOUNDARY",
                    f"shared boundary {edge[0]}-{edge[1]} must declare an articulated_hinge seam",
                )
        for key, seam_id in seam_keys.items():
            owners = tuple(sorted(boundary_owners.get(key[1], ())))
            if owners != key[0]:
                _fail(
                    "SEAM_NOT_SHARED_BOUNDARY",
                    f"seam {seam_id} does not exactly own its declared shared boundary",
                )

        for stroke in self.strokes:
            if not isinstance(stroke.source_edge_id, str) or not stroke.source_edge_id.strip():
                _fail("INVALID_IDENTITY", "sourceEdgeId must be a non-empty string")
            if len(stroke.vertex_ids) != 2 or stroke.vertex_ids[0] == stroke.vertex_ids[1]:
                _fail("INVALID_STROKE", f"stroke {stroke.source_edge_id} needs two distinct vertices")
            if len(set(stroke.incident_face_ids)) != len(stroke.incident_face_ids):
                _fail(
                    "DUPLICATE_IDENTITY",
                    f"stroke {stroke.source_edge_id}.incidentFaceIds contains duplicates",
                )
            if len(set(stroke.excluded_occluder_face_ids)) != len(
                stroke.excluded_occluder_face_ids
            ):
                _fail(
                    "DUPLICATE_IDENTITY",
                    f"stroke {stroke.source_edge_id}.excludedOccluderFaceIds contains duplicates",
                )
            if stroke.render_binding_id is not None and (
                not isinstance(stroke.render_binding_id, str)
                or not stroke.render_binding_id.strip()
            ):
                _fail(
                    "INVALID_BINDING_ID",
                    f"stroke {stroke.source_edge_id}.renderBindingId must be a non-empty string",
                )
            missing = sorted(set(stroke.vertex_ids) - vertex_ids)
            if missing:
                _fail(
                    "MISSING_VERTEX",
                    f"stroke {stroke.source_edge_id} references missing vertices: {', '.join(missing)}",
                )
            if set(stroke.incident_face_ids) & set(stroke.excluded_occluder_face_ids):
                _fail(
                    "REDUNDANT_OCCLUDER_EXCLUSION",
                    f"stroke {stroke.source_edge_id} cannot mark one face both incident and excluded",
                )
            for face_id in (*stroke.incident_face_ids, *stroke.excluded_occluder_face_ids):
                if face_id not in face_map:
                    _fail(
                        "MISSING_FACE",
                        f"stroke {stroke.source_edge_id} references missing face {face_id}",
                    )
            for face_id in stroke.incident_face_ids:
                if not set(stroke.vertex_ids).issubset(face_map[face_id].vertex_ids):
                    _fail(
                        "INVALID_INCIDENT_FACE",
                        f"incident face {face_id} does not contain both endpoints of stroke "
                        f"{stroke.source_edge_id}",
                    )
            if stroke.visibility_mode not in {"auto", "always_visible", "always_hidden"}:
                _fail(
                    "INVALID_VISIBILITY_MODE",
                    f"stroke {stroke.source_edge_id}.visibilityMode is unsupported",
                )

    def _positions(
        self, vertex_positions: Mapping[str, Sequence[float]] | None
    ) -> dict[str, np.ndarray]:
        raw = self.entry_positions if vertex_positions is None else vertex_positions
        if set(raw) != set(self.vertex_map):
            missing = sorted(set(self.vertex_map) - set(raw))
            extra = sorted(set(raw) - set(self.vertex_map))
            _fail(
                "VERTEX_IDENTITY_MISMATCH",
                "vertex position identity mismatch"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else ""),
            )
        result: dict[str, np.ndarray] = {}
        for vertex_id in sorted(raw):
            point = np.asarray(raw[vertex_id], dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                _fail(
                    "INVALID_POINT",
                    f"vertex {vertex_id} position must be a finite three-component point",
                )
            result[vertex_id] = point
        return result

    def _validated_frame(
        self,
        *,
        vertex_positions: Mapping[str, Sequence[float]] | None = None,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> _ValidatedOpenFaceFrame:
        self._validate_static_contract()
        positions = self._positions(vertex_positions)
        policy = tolerance_policy or TolerancePolicy()
        normals: dict[str, np.ndarray] = {}
        tolerances: dict[str, ResolvedTolerance] = {}
        for face in self.faces:
            points = np.asarray([positions[item] for item in face.vertex_ids], dtype=float)
            tolerance = policy.resolve(points)
            try:
                normal = _validate_convex_face_points(points, tolerance, face.face_id)
            except FrozenContractError as exc:
                raise OpenFaceContractError("INVALID_FACE_GEOMETRY", str(exc)) from exc
            tolerances[face.face_id] = tolerance
            normals[face.face_id] = normal

        for seam in self.seams:
            start, end = (positions[item] for item in seam.vertex_ids)
            length = float(np.linalg.norm(end - start))
            seam_tolerance = policy.resolve((start, end), edge_length=length)
            if length <= seam_tolerance.world:
                _fail("DEGENERATE_SEAM", f"seam {seam.seam_id} has zero-length hinge axis")

        for stroke in self.strokes:
            start, end = (positions[item] for item in stroke.vertex_ids)
            length = float(np.linalg.norm(end - start))
            stroke_tolerance = policy.resolve((start, end), edge_length=length)
            if length <= stroke_tolerance.world:
                _fail("DEGENERATE_STROKE", f"stroke {stroke.source_edge_id} has zero length")
            for face_id in stroke.excluded_occluder_face_ids:
                face = self.face_map[face_id]
                face_anchor = positions[face.vertex_ids[0]]
                normal = normals[face_id]
                distances = np.dot(np.asarray((start, end)) - face_anchor, normal)
                if float(np.max(np.abs(distances))) > tolerances[face_id].boundary:
                    _fail(
                        "UNPROVEN_COPLANAR_EXCLUSION",
                        f"stroke {stroke.source_edge_id} is not wholly coplanar with excluded face "
                        f"{face_id}",
                    )
        return _ValidatedOpenFaceFrame(positions, normals, tolerances)

    def validate(
        self,
        *,
        vertex_positions: Mapping[str, Sequence[float]] | None = None,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> None:
        self._validated_frame(
            vertex_positions=vertex_positions,
            tolerance_policy=tolerance_policy,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "topology": self.topology,
            "visibilityGroupId": self.visibility_group_id,
            "vertices": [item.to_dict() for item in self.vertices],
            "faces": [item.to_dict() for item in self.faces],
            "seams": [item.to_dict() for item in self.seams],
            "strokes": [item.to_dict() for item in self.strokes],
        }


__all__ = [
    "ARTICULATED_HINGE_POLICY",
    "OPEN_FACE_MODEL_SCHEMA",
    "OPEN_FACE_TOPOLOGY",
    "OpenFaceContractError",
    "OpenFaceSeamSpec",
    "OpenFaceSpec",
    "OpenFaceStrokeSpec",
    "OpenFaceVertexSpec",
    "OpenFaceVisibilityModel",
]
