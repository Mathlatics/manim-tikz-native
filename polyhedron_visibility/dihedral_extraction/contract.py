from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Mapping, Sequence

import numpy as np

from ..contract import (
    ContractError,
    FaceSpec,
    StrokeSpec,
    TolerancePolicy,
    VertexSpec,
    VisibilityModel,
)


DERIVED_DIHEDRAL_MODEL_SCHEMA = "manim-derived-dihedral-visibility/v1"


class DerivedDihedralContractError(ValueError):
    """Stable fail-closed error for a solid-derived dihedral contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise DerivedDihedralContractError(code, message)


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("INVALID_IDENTITY", f"{label} must be a non-empty string")
    return value.strip()


def _point3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    try:
        point = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise DerivedDihedralContractError(
            "INVALID_POINT", f"{label} must contain three numeric components"
        ) from exc
    if len(point) != 3 or not all(isfinite(item) for item in point):
        _fail("INVALID_POINT", f"{label} must be a finite three-component point")
    return point  # type: ignore[return-value]


@dataclass(frozen=True)
class RigidTransform3D:
    """A right-handed rigid transform used by one extracted dihedral.

    The matrix is validated as a proper rotation.  Translation, rotation, or
    their composition may be driven by ordinary Manim ``ValueTracker`` values
    through a provider that returns a fresh transform each frame.
    """

    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    translation: tuple[float, float, float]

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            _fail("INVALID_TRANSFORM", "rotation must be 3x3 and translation must be 3D")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            _fail("INVALID_TRANSFORM", "rigid transform components must be finite")
        identity = rotation.T @ rotation
        if not np.allclose(identity, np.eye(3), rtol=0.0, atol=1.0e-9):
            _fail("NON_RIGID_TRANSFORM", "rotation matrix must be orthonormal")
        determinant = float(np.linalg.det(rotation))
        if abs(determinant - 1.0) > 1.0e-9:
            _fail("NON_RIGID_TRANSFORM", "rotation matrix must be right-handed")
        object.__setattr__(
            self,
            "rotation",
            tuple(tuple(float(item) for item in row) for row in rotation),
        )
        object.__setattr__(
            self,
            "translation",
            tuple(float(item) for item in translation),
        )

    @classmethod
    def identity(cls) -> "RigidTransform3D":
        return cls(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.0),
        )

    @classmethod
    def translation_by(cls, offset: Sequence[float]) -> "RigidTransform3D":
        return cls(cls.identity().rotation, _point3(offset, "translation"))

    @classmethod
    def rotation_about_axis(
        cls,
        axis: Sequence[float],
        angle: float,
        *,
        about_point: Sequence[float] = (0.0, 0.0, 0.0),
        translation: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "RigidTransform3D":
        direction = np.asarray(_point3(axis, "rotation axis"), dtype=float)
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-12:
            _fail("INVALID_TRANSFORM", "rotation axis must be non-zero")
        if not isfinite(float(angle)):
            _fail("INVALID_TRANSFORM", "rotation angle must be finite")
        direction /= length
        x, y, z = direction
        c = cos(float(angle))
        s = sin(float(angle))
        one_minus = 1.0 - c
        rotation = np.asarray(
            (
                (c + x * x * one_minus, x * y * one_minus - z * s, x * z * one_minus + y * s),
                (y * x * one_minus + z * s, c + y * y * one_minus, y * z * one_minus - x * s),
                (z * x * one_minus - y * s, z * y * one_minus + x * s, c + z * z * one_minus),
            ),
            dtype=float,
        )
        center = np.asarray(_point3(about_point, "rotation center"), dtype=float)
        extra = np.asarray(_point3(translation, "translation"), dtype=float)
        offset = center - rotation @ center + extra
        return cls(
            tuple(tuple(float(item) for item in row) for row in rotation),
            tuple(float(item) for item in offset),
        )

    def apply(self, point: Sequence[float]) -> np.ndarray:
        value = np.asarray(_point3(point, "transform point"), dtype=float)
        return np.asarray(self.rotation, dtype=float) @ value + np.asarray(
            self.translation, dtype=float
        )

    def compose(self, inner: "RigidTransform3D") -> "RigidTransform3D":
        """Apply ``inner`` first, then this transform.

        For example, ``placement.compose(center_rotation)`` first rotates an
        entity about its authored geometric center and then moves that center
        to the entity's independently translated position.
        """

        if not isinstance(inner, RigidTransform3D):
            _fail("INVALID_TRANSFORM", "inner transform must be RigidTransform3D")
        outer_rotation = np.asarray(self.rotation, dtype=float)
        inner_rotation = np.asarray(inner.rotation, dtype=float)
        rotation = outer_rotation @ inner_rotation
        translation = outer_rotation @ np.asarray(
            inner.translation, dtype=float
        ) + np.asarray(self.translation, dtype=float)
        return RigidTransform3D(
            tuple(tuple(float(item) for item in row) for row in rotation),
            tuple(float(item) for item in translation),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rotation": [list(row) for row in self.rotation],
            "translation": list(self.translation),
        }


@dataclass(frozen=True)
class DerivedBoundaryStrokeSpec:
    source_stroke_id: str
    vertex_ids: tuple[str, str]
    incident_source_face_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceStrokeId": self.source_stroke_id,
            "vertexIds": list(self.vertex_ids),
            "incidentSourceFaceIds": list(self.incident_source_face_ids),
        }


@dataclass(frozen=True)
class DerivedDihedralSpec:
    entity_id: str
    source_face_ids: tuple[str, str]
    hinge_vertex_ids: tuple[str, str]
    boundary_strokes: tuple[DerivedBoundaryStrokeSpec, ...]
    entry_transform: RigidTransform3D

    def to_dict(self) -> dict[str, object]:
        return {
            "entityId": self.entity_id,
            "sourceFaceIds": list(self.source_face_ids),
            "hingeVertexIds": list(self.hinge_vertex_ids),
            "boundaryStrokes": [item.to_dict() for item in self.boundary_strokes],
            "entryTransform": self.entry_transform.to_dict(),
        }


@dataclass(frozen=True)
class DerivedDihedralModel:
    visibility_group_id: str
    solid: VisibilityModel
    extraction: DerivedDihedralSpec
    schema: str = DERIVED_DIHEDRAL_MODEL_SCHEMA

    @staticmethod
    def _boundary_edges(vertex_ids: Sequence[str]) -> set[tuple[str, str]]:
        return {
            tuple(sorted((start, vertex_ids[(index + 1) % len(vertex_ids)])))
            for index, start in enumerate(vertex_ids)
        }

    @classmethod
    def from_solid(
        cls,
        visibility_group_id: str,
        solid: VisibilityModel,
        *,
        entity_id: str,
        source_face_ids: Sequence[str],
        entry_transform: RigidTransform3D | None = None,
        vertex_positions: Mapping[str, Sequence[float]] | None = None,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> "DerivedDihedralModel":
        group = _identity(visibility_group_id, "visibility_group_id")
        entity = _identity(entity_id, "entity_id")
        face_ids = tuple(_identity(item, "source_face_id") for item in source_face_ids)
        if len(face_ids) != 2 or face_ids[0] == face_ids[1]:
            _fail("INVALID_FACE_SELECTION", "source_face_ids must contain two distinct faces")
        missing = sorted(set(face_ids) - set(solid.face_map))
        if missing:
            _fail("MISSING_SOURCE_FACE", "missing source faces: " + ", ".join(missing))
        try:
            solid.validate(
                vertex_positions=vertex_positions,
                require_closed_convex_manifold=True,
                tolerance_policy=tolerance_policy,
            )
        except ContractError as exc:
            raise DerivedDihedralContractError(
                "INVALID_SOURCE_SOLID", str(exc)
            ) from exc
        first, second = (solid.face_map[item] for item in face_ids)
        shared = cls._boundary_edges(first.vertex_ids) & cls._boundary_edges(
            second.vertex_ids
        )
        if len(shared) != 1:
            _fail(
                "FACES_NOT_ADJACENT",
                "selected faces must share exactly one complete boundary edge",
            )
        hinge = next(iter(shared))
        selected_edges = cls._boundary_edges(first.vertex_ids) | cls._boundary_edges(
            second.vertex_ids
        )
        strokes_by_edge: dict[tuple[str, str], list[object]] = {}
        for stroke in solid.strokes:
            strokes_by_edge.setdefault(tuple(sorted(stroke.vertex_ids)), []).append(stroke)
        boundary_strokes: list[DerivedBoundaryStrokeSpec] = []
        for edge in sorted(selected_edges):
            candidates = strokes_by_edge.get(edge, [])
            if len(candidates) != 1:
                _fail(
                    "AMBIGUOUS_BOUNDARY_STROKE",
                    f"boundary {edge[0]}-{edge[1]} must map to exactly one source stroke",
                )
            stroke = candidates[0]
            incidents = tuple(
                sorted(
                    face_id
                    for face_id in face_ids
                    if set(edge).issubset(solid.face_map[face_id].vertex_ids)
                )
            )
            boundary_strokes.append(
                DerivedBoundaryStrokeSpec(
                    source_stroke_id=str(getattr(stroke, "source_edge_id")),
                    vertex_ids=(edge[0], edge[1]),
                    incident_source_face_ids=incidents,
                )
            )
        return cls(
            group,
            solid,
            DerivedDihedralSpec(
                entity,
                (face_ids[0], face_ids[1]),
                (hinge[0], hinge[1]),
                tuple(boundary_strokes),
                entry_transform or RigidTransform3D.identity(),
            ),
        )

    @property
    def extracted_vertex_ids(self) -> tuple[str, ...]:
        values = {
            vertex_id
            for face_id in self.extraction.source_face_ids
            for vertex_id in self.solid.face_map[face_id].vertex_ids
        }
        return tuple(sorted(values))

    @staticmethod
    def solid_vertex_id(vertex_id: str) -> str:
        return f"solid:{vertex_id}"

    def extracted_vertex_id(self, vertex_id: str) -> str:
        return f"{self.extraction.entity_id}:{vertex_id}"

    @staticmethod
    def solid_face_id(face_id: str) -> str:
        return f"solid:{face_id}"

    def extracted_face_id(self, face_id: str) -> str:
        return f"{self.extraction.entity_id}:{face_id}"

    @staticmethod
    def solid_stroke_id(stroke_id: str) -> str:
        return f"solid:{stroke_id}"

    def extracted_stroke_id(self, stroke_id: str) -> str:
        return f"{self.extraction.entity_id}:{stroke_id}"

    def overlay_model(self) -> VisibilityModel:
        """Return the stable line-slot protocol shared by the Manim binding.

        The returned object deliberately is not revalidated as one closed
        polyhedron: it is the union of a valid closed solid and a valid open
        two-face copy, so duplicate entry coordinates are expected.
        """

        entry_transform = self.extraction.entry_transform
        vertices = [
            VertexSpec(
                self.solid_vertex_id(vertex.vertex_id),
                vertex.entry_position,
            )
            for vertex in self.solid.vertices
        ]
        for vertex_id in self.extracted_vertex_ids:
            point = entry_transform.apply(
                self.solid.vertex_map[vertex_id].entry_position
            )
            vertices.append(
                VertexSpec(
                    self.extracted_vertex_id(vertex_id),
                    tuple(float(item) for item in point),
                )
            )
        faces = [
            FaceSpec(
                self.solid_face_id(face.face_id),
                tuple(self.solid_vertex_id(item) for item in face.vertex_ids),
                face.occludes_strokes,
            )
            for face in self.solid.faces
        ]
        for face_id in self.extraction.source_face_ids:
            source = self.solid.face_map[face_id]
            faces.append(
                FaceSpec(
                    self.extracted_face_id(face_id),
                    tuple(self.extracted_vertex_id(item) for item in source.vertex_ids),
                    source.occludes_strokes,
                )
            )
        strokes = [
            StrokeSpec(
                self.solid_stroke_id(stroke.source_edge_id),
                tuple(self.solid_vertex_id(item) for item in stroke.vertex_ids),
                tuple(self.solid_face_id(item) for item in stroke.incident_face_ids),
                stroke.render_binding_id,
                stroke.visibility_mode,
            )
            for stroke in self.solid.strokes
        ]
        for stroke in self.extraction.boundary_strokes:
            strokes.append(
                StrokeSpec(
                    self.extracted_stroke_id(stroke.source_stroke_id),
                    tuple(self.extracted_vertex_id(item) for item in stroke.vertex_ids),
                    tuple(
                        self.extracted_face_id(item)
                        for item in stroke.incident_source_face_ids
                    ),
                    None,
                    "auto",
                )
            )
        return VisibilityModel(
            self.visibility_group_id,
            tuple(sorted(vertices, key=lambda item: item.vertex_id)),
            tuple(sorted(faces, key=lambda item: item.face_id)),
            tuple(sorted(strokes, key=lambda item: item.source_edge_id)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "visibilityGroupId": self.visibility_group_id,
            "solid": self.solid.to_dict(),
            "extraction": self.extraction.to_dict(),
        }


__all__ = [
    "DERIVED_DIHEDRAL_MODEL_SCHEMA",
    "DerivedBoundaryStrokeSpec",
    "DerivedDihedralContractError",
    "DerivedDihedralModel",
    "DerivedDihedralSpec",
    "RigidTransform3D",
]
