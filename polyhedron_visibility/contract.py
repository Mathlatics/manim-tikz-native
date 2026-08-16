from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np


VISIBILITY_MODEL_SCHEMA = "manim-convex-polyhedron-visibility/v1"


class ContractError(ValueError):
    """Raised when a visibility model cannot be interpreted safely."""


def _strict_keys(payload: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ContractError(f"{label} contains unknown fields: {', '.join(unknown)}")


def _required_text(payload: Mapping[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _text_tuple(value: object, label: str, *, minimum: int = 0) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{label} must be an array of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ContractError(f"{label}[{index}] must be a non-empty string")
        items.append(item.strip())
    if len(items) < minimum:
        raise ContractError(f"{label} must contain at least {minimum} items")
    if len(set(items)) != len(items):
        raise ContractError(f"{label} contains duplicate identities")
    return tuple(items)


def _point3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ContractError(f"{label} must be a three-component point")
    try:
        point = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must contain numeric components") from exc
    if not all(isfinite(component) for component in point):
        raise ContractError(f"{label} components must be finite")
    return point  # type: ignore[return-value]


@dataclass(frozen=True)
class TolerancePolicy:
    """Scale-aware numerical policy shared by validation and frame solving."""

    relative: float = 1.0e-9
    absolute_floor: float = 1.0e-14
    angular: float = 1.0e-10
    boundary_factor: float = 8.0
    depth_factor: float = 8.0

    def __post_init__(self) -> None:
        values = (
            self.relative,
            self.absolute_floor,
            self.angular,
            self.boundary_factor,
            self.depth_factor,
        )
        if not all(isfinite(float(value)) and float(value) > 0 for value in values):
            raise ContractError("tolerance values must be finite and positive")

    def resolve(
        self,
        positions: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
        *,
        edge_length: float | None = None,
    ) -> "ResolvedTolerance":
        if isinstance(positions, Mapping):
            values = list(positions.values())
        else:
            values = list(positions)
        points = np.asarray(values, dtype=float)
        if points.size == 0:
            scale = 1.0
        else:
            if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
                raise ContractError("tolerance positions must be finite three-component points")
            extent = np.max(points, axis=0) - np.min(points, axis=0)
            scale = float(np.linalg.norm(extent))
        if edge_length is not None:
            if not isfinite(float(edge_length)) or float(edge_length) < 0:
                raise ContractError("edge length used by tolerance must be finite and non-negative")
            scale = max(scale, float(edge_length))
        scale = max(scale, self.absolute_floor)
        world = max(self.absolute_floor, self.relative * scale)
        denominator = max(float(edge_length or scale), world)
        return ResolvedTolerance(
            scale=scale,
            world=world,
            parameter=world / denominator,
            angular=self.angular,
            boundary=self.boundary_factor * world,
            depth=self.depth_factor * world,
        )


@dataclass(frozen=True)
class ResolvedTolerance:
    scale: float
    world: float
    parameter: float
    angular: float
    boundary: float
    depth: float

    def to_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale,
            "world": self.world,
            "parameter": self.parameter,
            "angular": self.angular,
            "boundary": self.boundary,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class VertexSpec:
    vertex_id: str
    entry_position: tuple[float, float, float]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "VertexSpec":
        _strict_keys(payload, {"vertexId", "entryPosition"}, "vertex")
        vertex_id = _required_text(payload, "vertexId", "vertex")
        return cls(vertex_id, _point3(payload.get("entryPosition"), f"vertex {vertex_id}"))

    def to_dict(self) -> dict[str, object]:
        return {"vertexId": self.vertex_id, "entryPosition": list(self.entry_position)}


@dataclass(frozen=True)
class FaceSpec:
    face_id: str
    vertex_ids: tuple[str, ...]
    occludes_strokes: bool = True
    logical_surface_id: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FaceSpec":
        _strict_keys(
            payload,
            {"faceId", "vertexIds", "occludesStrokes", "logicalSurfaceId"},
            "face",
        )
        face_id = _required_text(payload, "faceId", "face")
        occludes = payload.get("occludesStrokes", True)
        if not isinstance(occludes, bool):
            raise ContractError(f"face {face_id}.occludesStrokes must be boolean")
        logical = payload.get("logicalSurfaceId")
        if logical is not None and (not isinstance(logical, str) or not logical.strip()):
            raise ContractError(f"face {face_id}.logicalSurfaceId must be a non-empty string")
        return cls(
            face_id=face_id,
            vertex_ids=_text_tuple(payload.get("vertexIds"), f"face {face_id}.vertexIds", minimum=3),
            occludes_strokes=occludes,
            logical_surface_id=logical.strip() if isinstance(logical, str) else None,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "faceId": self.face_id,
            "vertexIds": list(self.vertex_ids),
            "occludesStrokes": self.occludes_strokes,
        }
        if self.logical_surface_id is not None:
            result["logicalSurfaceId"] = self.logical_surface_id
        return result


@dataclass(frozen=True)
class StrokeSpec:
    source_edge_id: str
    vertex_ids: tuple[str, str]
    incident_face_ids: tuple[str, ...] = ()
    render_binding_id: str | None = None
    visibility_mode: str = "auto"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StrokeSpec":
        _strict_keys(
            payload,
            {
                "sourceEdgeId",
                "vertexIds",
                "incidentFaceIds",
                "renderBindingId",
                "visibilityMode",
            },
            "stroke",
        )
        edge_id = _required_text(payload, "sourceEdgeId", "stroke")
        vertex_ids = _text_tuple(
            payload.get("vertexIds"), f"stroke {edge_id}.vertexIds", minimum=2
        )
        if len(vertex_ids) != 2:
            raise ContractError(f"stroke {edge_id}.vertexIds must contain exactly two vertices")
        incidents = _text_tuple(
            payload.get("incidentFaceIds", []), f"stroke {edge_id}.incidentFaceIds"
        )
        binding = payload.get("renderBindingId")
        if binding is not None and (not isinstance(binding, str) or not binding.strip()):
            raise ContractError(f"stroke {edge_id}.renderBindingId must be a non-empty string")
        mode = payload.get("visibilityMode", "auto")
        if mode not in {"auto", "always_visible", "always_hidden"}:
            raise ContractError(f"stroke {edge_id}.visibilityMode is unsupported")
        return cls(
            source_edge_id=edge_id,
            vertex_ids=(vertex_ids[0], vertex_ids[1]),
            incident_face_ids=tuple(sorted(incidents)),
            render_binding_id=binding.strip() if isinstance(binding, str) else None,
            visibility_mode=str(mode),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "sourceEdgeId": self.source_edge_id,
            "vertexIds": list(self.vertex_ids),
            "incidentFaceIds": list(self.incident_face_ids),
            "visibilityMode": self.visibility_mode,
        }
        if self.render_binding_id is not None:
            result["renderBindingId"] = self.render_binding_id
        return result


def _unique(items: Sequence[object], attribute: str, label: str) -> None:
    seen: set[str] = set()
    for item in items:
        identity = str(getattr(item, attribute))
        if identity in seen:
            raise ContractError(f"duplicate {label}: {identity}")
        seen.add(identity)


def _face_normal(points: np.ndarray, world_epsilon: float, label: str) -> np.ndarray:
    origin = points[0]
    for index in range(1, len(points) - 1):
        normal = np.cross(points[index] - origin, points[index + 1] - origin)
        length = float(np.linalg.norm(normal))
        if length > world_epsilon * world_epsilon:
            return normal / length
    raise ContractError(f"face {label} is degenerate")


@dataclass(frozen=True)
class VisibilityModel:
    visibility_group_id: str
    vertices: tuple[VertexSpec, ...]
    faces: tuple[FaceSpec, ...]
    strokes: tuple[StrokeSpec, ...]
    schema: str = VISIBILITY_MODEL_SCHEMA

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "VisibilityModel":
        if not isinstance(payload, Mapping):
            raise ContractError("visibility model must be an object")
        _strict_keys(
            payload,
            {"schema", "visibilityGroupId", "vertices", "faces", "strokes"},
            "visibility model",
        )
        if payload.get("schema") != VISIBILITY_MODEL_SCHEMA:
            raise ContractError(f"visibility model schema must be {VISIBILITY_MODEL_SCHEMA}")
        group_id = _required_text(payload, "visibilityGroupId", "visibility model")
        raw_vertices = payload.get("vertices")
        raw_faces = payload.get("faces")
        raw_strokes = payload.get("strokes")
        if not isinstance(raw_vertices, list) or not isinstance(raw_faces, list) or not isinstance(raw_strokes, list):
            raise ContractError("visibility model vertices, faces, and strokes must be arrays")
        vertices = tuple(
            sorted(
                (VertexSpec.from_dict(item) for item in raw_vertices if isinstance(item, Mapping)),
                key=lambda item: item.vertex_id,
            )
        )
        faces = tuple(
            sorted(
                (FaceSpec.from_dict(item) for item in raw_faces if isinstance(item, Mapping)),
                key=lambda item: item.face_id,
            )
        )
        strokes = tuple(
            sorted(
                (StrokeSpec.from_dict(item) for item in raw_strokes if isinstance(item, Mapping)),
                key=lambda item: item.source_edge_id,
            )
        )
        if len(vertices) != len(raw_vertices) or len(faces) != len(raw_faces) or len(strokes) != len(raw_strokes):
            raise ContractError("visibility model arrays must contain objects")
        _unique(vertices, "vertex_id", "vertexId")
        _unique(faces, "face_id", "faceId")
        _unique(strokes, "source_edge_id", "sourceEdgeId")
        model = cls(group_id, vertices, faces, strokes)
        model._validate_references()
        return model

    @property
    def vertex_map(self) -> dict[str, VertexSpec]:
        return {item.vertex_id: item for item in self.vertices}

    @property
    def face_map(self) -> dict[str, FaceSpec]:
        return {item.face_id: item for item in self.faces}

    @property
    def stroke_map(self) -> dict[str, StrokeSpec]:
        return {item.source_edge_id: item for item in self.strokes}

    @property
    def entry_positions(self) -> dict[str, tuple[float, float, float]]:
        return {item.vertex_id: item.entry_position for item in self.vertices}

    def _validate_references(self) -> None:
        vertex_ids = set(self.vertex_map)
        face_map = self.face_map
        for face in self.faces:
            missing = sorted(set(face.vertex_ids) - vertex_ids)
            if missing:
                raise ContractError(f"face {face.face_id} references missing vertices: {', '.join(missing)}")
        for stroke in self.strokes:
            missing = sorted(set(stroke.vertex_ids) - vertex_ids)
            if missing:
                raise ContractError(
                    f"stroke {stroke.source_edge_id} references missing vertices: {', '.join(missing)}"
                )
            for face_id in stroke.incident_face_ids:
                face = face_map.get(face_id)
                if face is None:
                    raise ContractError(
                        f"stroke {stroke.source_edge_id} references missing incident face {face_id}"
                    )
                if not set(stroke.vertex_ids).issubset(face.vertex_ids):
                    raise ContractError(
                        f"incident face {face_id} does not contain both endpoints of stroke {stroke.source_edge_id}"
                    )

    def _validated_positions(
        self, vertex_positions: Mapping[str, Sequence[float]] | None
    ) -> dict[str, np.ndarray]:
        raw = self.entry_positions if vertex_positions is None else vertex_positions
        if set(raw) != set(self.vertex_map):
            missing = sorted(set(self.vertex_map) - set(raw))
            extra = sorted(set(raw) - set(self.vertex_map))
            raise ContractError(
                "vertex position identity mismatch"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
        result: dict[str, np.ndarray] = {}
        for vertex_id in sorted(raw):
            point = np.asarray(raw[vertex_id], dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ContractError(f"vertex {vertex_id} position must be a finite three-component point")
            result[vertex_id] = point
        return result

    def validate(
        self,
        *,
        vertex_positions: Mapping[str, Sequence[float]] | None = None,
        require_closed_convex_manifold: bool = False,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> None:
        self._validate_references()
        positions = self._validated_positions(vertex_positions)
        policy = tolerance_policy or TolerancePolicy()
        tolerance = policy.resolve(positions)
        face_normals: dict[str, np.ndarray] = {}
        for face in self.faces:
            points = np.asarray([positions[item] for item in face.vertex_ids], dtype=float)
            normal = _face_normal(points, tolerance.world, face.face_id)
            distances = np.dot(points - points[0], normal)
            if float(np.max(np.abs(distances))) > tolerance.boundary:
                raise ContractError(f"face {face.face_id} is not planar")
            turn_signs: list[float] = []
            for index in range(len(points)):
                before = points[index - 1]
                current = points[index]
                after = points[(index + 1) % len(points)]
                signed = float(np.dot(np.cross(current - before, after - current), normal))
                if abs(signed) <= tolerance.world * tolerance.world:
                    raise ContractError(f"face {face.face_id} is not strictly convex")
                turn_signs.append(signed)
            if min(turn_signs) < 0 < max(turn_signs):
                raise ContractError(f"face {face.face_id} is not strictly convex")
            face_normals[face.face_id] = normal

        if not require_closed_convex_manifold:
            return
        edge_occurrences: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        for face in self.faces:
            for index, start in enumerate(face.vertex_ids):
                end = face.vertex_ids[(index + 1) % len(face.vertex_ids)]
                key = tuple(sorted((start, end)))
                edge_occurrences.setdefault(key, []).append((face.face_id, start, end))
        invalid = {key: value for key, value in edge_occurrences.items() if len(value) != 2}
        if invalid:
            raise ContractError("faces do not form a closed two-manifold")
        for key, occurrences in edge_occurrences.items():
            first, second = occurrences
            if first[1:] != (second[2], second[1]):
                raise ContractError(
                    f"closed two-manifold edge {key[0]}-{key[1]} has inconsistent winding"
                )
        all_points = np.asarray([positions[item.vertex_id] for item in self.vertices])
        for face in self.faces:
            point = positions[face.vertex_ids[0]]
            normal = face_normals[face.face_id]
            signed = np.dot(all_points - point, normal)
            positive = bool(np.any(signed > tolerance.boundary))
            negative = bool(np.any(signed < -tolerance.boundary))
            if positive and negative:
                raise ContractError(f"closed two-manifold is not convex at face {face.face_id}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "visibilityGroupId": self.visibility_group_id,
            "vertices": [item.to_dict() for item in self.vertices],
            "faces": [item.to_dict() for item in self.faces],
            "strokes": [item.to_dict() for item in self.strokes],
        }


__all__ = [
    "ContractError",
    "FaceSpec",
    "ResolvedTolerance",
    "StrokeSpec",
    "TolerancePolicy",
    "VertexSpec",
    "VISIBILITY_MODEL_SCHEMA",
    "VisibilityModel",
]
