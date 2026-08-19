from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from ..contract import VisibilityModel


COPY_IDENTITY_HANDOFF_SCHEMA = "manim-copy-identity-handoff/v1"


class CopyHandoffContractError(ValueError):
    """Stable fail-closed error for one source-to-copy identity map."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise CopyHandoffContractError(code, message)


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("INVALID_IDENTITY", f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class CopyVertexPair:
    """One semantic vertex and its source/copy runtime identities."""

    semantic_vertex_id: str
    source_vertex_id: str
    copy_vertex_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_vertex_id",
            _identity(self.semantic_vertex_id, "semantic_vertex_id"),
        )
        object.__setattr__(
            self,
            "source_vertex_id",
            _identity(self.source_vertex_id, "source_vertex_id"),
        )
        object.__setattr__(
            self,
            "copy_vertex_id",
            _identity(self.copy_vertex_id, "copy_vertex_id"),
        )
        if self.source_vertex_id == self.copy_vertex_id:
            _fail(
                "AMBIGUOUS_VERTEX_IDENTITY",
                "source and copy vertex identities must be distinct",
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "semanticVertexId": self.semantic_vertex_id,
            "sourceVertexId": self.source_vertex_id,
            "copyVertexId": self.copy_vertex_id,
        }


@dataclass(frozen=True)
class CopyPrimitivePair:
    """One corresponding face or stroke from the source and the copy."""

    semantic_primitive_id: str
    source_primitive_id: str
    copy_primitive_id: str
    vertex_pair_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_primitive_id",
            _identity(self.semantic_primitive_id, "semantic_primitive_id"),
        )
        object.__setattr__(
            self,
            "source_primitive_id",
            _identity(self.source_primitive_id, "source_primitive_id"),
        )
        object.__setattr__(
            self,
            "copy_primitive_id",
            _identity(self.copy_primitive_id, "copy_primitive_id"),
        )
        if self.source_primitive_id == self.copy_primitive_id:
            _fail(
                "AMBIGUOUS_PRIMITIVE_IDENTITY",
                "source and copy primitive identities must be distinct",
            )
        vertex_pair_ids = tuple(
            _identity(item, "vertex_pair_id") for item in self.vertex_pair_ids
        )
        if not vertex_pair_ids:
            _fail(
                "EMPTY_PRIMITIVE_TOPOLOGY",
                f"primitive {self.semantic_primitive_id!r} has no vertices",
            )
        if len(set(vertex_pair_ids)) != len(vertex_pair_ids):
            _fail(
                "DUPLICATE_PRIMITIVE_VERTEX",
                f"primitive {self.semantic_primitive_id!r} repeats a vertex",
            )
        object.__setattr__(self, "vertex_pair_ids", vertex_pair_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "semanticPrimitiveId": self.semantic_primitive_id,
            "sourcePrimitiveId": self.source_primitive_id,
            "copyPrimitiveId": self.copy_primitive_id,
            "vertexPairIds": list(self.vertex_pair_ids),
        }


@dataclass(frozen=True)
class CopyIdentityHandoffPolicy:
    """How far copied geometry travels before both entities are fully visible.

    The copy owns coincident pixels at distance zero.  The corresponding source
    primitives fade back in with a cubic smoothstep and reach their authored
    opacity at ``activation_distance``.
    """

    activation_distance: float = 0.12
    curve: str = "smoothstep"

    def __post_init__(self) -> None:
        if (
            isinstance(self.activation_distance, bool)
            or not isinstance(self.activation_distance, (int, float))
            or not isfinite(float(self.activation_distance))
            or float(self.activation_distance) <= 0.0
        ):
            _fail(
                "INVALID_ACTIVATION_DISTANCE",
                "activation_distance must be finite and positive",
            )
        if self.curve != "smoothstep":
            _fail(
                "UNSUPPORTED_HANDOFF_CURVE",
                "copy handoff v1 supports only the smoothstep curve",
            )
        object.__setattr__(
            self,
            "activation_distance",
            float(self.activation_distance),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "activationDistance": self.activation_distance,
            "curve": self.curve,
            "coincidentOwner": "copy",
        }


@dataclass(frozen=True)
class CopyIdentityHandoffMap:
    """Frozen semantic lineage between one source entity and one copied entity."""

    handoff_id: str
    vertex_pairs: tuple[CopyVertexPair, ...]
    face_pairs: tuple[CopyPrimitivePair, ...]
    stroke_pairs: tuple[CopyPrimitivePair, ...]
    policy: CopyIdentityHandoffPolicy = CopyIdentityHandoffPolicy()
    schema: str = COPY_IDENTITY_HANDOFF_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COPY_IDENTITY_HANDOFF_SCHEMA:
            _fail(
                "UNSUPPORTED_SCHEMA",
                f"expected {COPY_IDENTITY_HANDOFF_SCHEMA!r}",
            )
        object.__setattr__(self, "handoff_id", _identity(self.handoff_id, "handoff_id"))
        vertex_pairs = tuple(
            sorted(self.vertex_pairs, key=lambda item: item.semantic_vertex_id)
        )
        face_pairs = tuple(
            sorted(self.face_pairs, key=lambda item: item.semantic_primitive_id)
        )
        stroke_pairs = tuple(
            sorted(self.stroke_pairs, key=lambda item: item.semantic_primitive_id)
        )
        if not vertex_pairs:
            _fail("EMPTY_COPY_LINEAGE", "copy handoff requires at least one vertex pair")
        if not face_pairs and not stroke_pairs:
            _fail(
                "EMPTY_COPY_LINEAGE",
                "copy handoff requires at least one face or stroke pair",
            )
        self._validate_unique(
            "vertex",
            vertex_pairs,
            semantic=lambda item: item.semantic_vertex_id,
            source=lambda item: item.source_vertex_id,
            copy=lambda item: item.copy_vertex_id,
        )
        self._validate_unique(
            "face",
            face_pairs,
            semantic=lambda item: item.semantic_primitive_id,
            source=lambda item: item.source_primitive_id,
            copy=lambda item: item.copy_primitive_id,
        )
        self._validate_unique(
            "stroke",
            stroke_pairs,
            semantic=lambda item: item.semantic_primitive_id,
            source=lambda item: item.source_primitive_id,
            copy=lambda item: item.copy_primitive_id,
        )
        vertex_ids = {item.semantic_vertex_id for item in vertex_pairs}
        for kind, pairs, minimum, maximum in (
            ("face", face_pairs, 3, None),
            ("stroke", stroke_pairs, 2, 2),
        ):
            for pair in pairs:
                missing = sorted(set(pair.vertex_pair_ids) - vertex_ids)
                if missing:
                    _fail(
                        "MISSING_VERTEX_PAIR",
                        f"{kind} {pair.semantic_primitive_id!r} references "
                        + ", ".join(missing),
                    )
                count = len(pair.vertex_pair_ids)
                if count < minimum or (maximum is not None and count > maximum):
                    expected = (
                        f"exactly {minimum}"
                        if maximum == minimum
                        else f"at least {minimum}"
                    )
                    _fail(
                        "INVALID_PRIMITIVE_TOPOLOGY",
                        f"{kind} {pair.semantic_primitive_id!r} requires {expected} vertices",
                    )
        if not isinstance(self.policy, CopyIdentityHandoffPolicy):
            _fail("INVALID_POLICY", "policy must be CopyIdentityHandoffPolicy")
        object.__setattr__(self, "vertex_pairs", vertex_pairs)
        object.__setattr__(self, "face_pairs", face_pairs)
        object.__setattr__(self, "stroke_pairs", stroke_pairs)

    @staticmethod
    def _validate_unique(kind, items, *, semantic, source, copy) -> None:
        for label, getter in (
            ("semantic", semantic),
            ("source", source),
            ("copy", copy),
        ):
            values = [getter(item) for item in items]
            duplicates = sorted({item for item in values if values.count(item) > 1})
            if duplicates:
                _fail(
                    "DUPLICATE_COPY_LINEAGE",
                    f"duplicate {label} {kind} identities: " + ", ".join(duplicates),
                )

    @classmethod
    def from_visibility_model(
        cls,
        handoff_id: str,
        model: VisibilityModel,
        *,
        source_entity_id: str,
        copy_entity_id: str,
        face_ids: Sequence[str] | None = None,
        stroke_ids: Sequence[str] | None = None,
        policy: CopyIdentityHandoffPolicy | None = None,
    ) -> "CopyIdentityHandoffMap":
        """Build lineage for a whole copied solid or a selected copied subset."""

        if not isinstance(model, VisibilityModel):
            _fail("INVALID_SOURCE_MODEL", "model must be VisibilityModel")
        source_entity = _identity(source_entity_id, "source_entity_id")
        copy_entity = _identity(copy_entity_id, "copy_entity_id")
        if source_entity == copy_entity:
            _fail(
                "AMBIGUOUS_ENTITY_IDENTITY",
                "source_entity_id and copy_entity_id must be distinct",
            )
        selected_faces = tuple(
            sorted(model.face_map if face_ids is None else {
                _identity(item, "face_id") for item in face_ids
            })
        )
        selected_strokes = tuple(
            sorted(model.stroke_map if stroke_ids is None else {
                _identity(item, "stroke_id") for item in stroke_ids
            })
        )
        missing_faces = sorted(set(selected_faces) - set(model.face_map))
        missing_strokes = sorted(set(selected_strokes) - set(model.stroke_map))
        if missing_faces or missing_strokes:
            details = []
            if missing_faces:
                details.append("faces=" + ",".join(missing_faces))
            if missing_strokes:
                details.append("strokes=" + ",".join(missing_strokes))
            _fail("MISSING_SOURCE_PRIMITIVE", "; ".join(details))

        used_vertices = {
            vertex_id
            for face_id in selected_faces
            for vertex_id in model.face_map[face_id].vertex_ids
        } | {
            vertex_id
            for stroke_id in selected_strokes
            for vertex_id in model.stroke_map[stroke_id].vertex_ids
        }
        missing_vertices = sorted(used_vertices - set(model.vertex_map))
        if missing_vertices:
            _fail(
                "MISSING_SOURCE_VERTEX",
                "selected primitives reference missing vertices: "
                + ", ".join(missing_vertices),
            )

        def qualified(entity_id: str, semantic_id: str) -> str:
            return f"{entity_id}:{semantic_id}"

        return cls(
            handoff_id=_identity(handoff_id, "handoff_id"),
            vertex_pairs=tuple(
                CopyVertexPair(
                    vertex_id,
                    qualified(source_entity, vertex_id),
                    qualified(copy_entity, vertex_id),
                )
                for vertex_id in sorted(used_vertices)
            ),
            face_pairs=tuple(
                CopyPrimitivePair(
                    face_id,
                    qualified(source_entity, face_id),
                    qualified(copy_entity, face_id),
                    tuple(model.face_map[face_id].vertex_ids),
                )
                for face_id in selected_faces
            ),
            stroke_pairs=tuple(
                CopyPrimitivePair(
                    stroke_id,
                    qualified(source_entity, stroke_id),
                    qualified(copy_entity, stroke_id),
                    tuple(model.stroke_map[stroke_id].vertex_ids),
                )
                for stroke_id in selected_strokes
            ),
            policy=policy or CopyIdentityHandoffPolicy(),
        )

    @property
    def vertex_pair_map(self) -> dict[str, CopyVertexPair]:
        return {item.semantic_vertex_id: item for item in self.vertex_pairs}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "handoffId": self.handoff_id,
            "policy": self.policy.to_dict(),
            "vertexPairs": [item.to_dict() for item in self.vertex_pairs],
            "facePairs": [item.to_dict() for item in self.face_pairs],
            "strokePairs": [item.to_dict() for item in self.stroke_pairs],
        }


__all__ = [
    "COPY_IDENTITY_HANDOFF_SCHEMA",
    "CopyHandoffContractError",
    "CopyIdentityHandoffMap",
    "CopyIdentityHandoffPolicy",
    "CopyPrimitivePair",
    "CopyVertexPair",
]
