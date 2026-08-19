from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .contract import CopyIdentityHandoffMap, CopyPrimitivePair


COPY_IDENTITY_HANDOFF_FRAME_SCHEMA = "manim-copy-identity-handoff-frame/v1"

PointProvider = Callable[[Sequence[float]], Sequence[float]]


def _smoothstep01(value: float) -> float:
    normalized = min(1.0, max(0.0, float(value)))
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite three-component point") from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{label} must be a finite three-component point")
    return point


@dataclass(frozen=True)
class CopyVertexSeparation:
    semantic_vertex_id: str
    source_vertex_id: str
    copy_vertex_id: str
    separation: float

    def to_dict(self) -> dict[str, object]:
        return {
            "semanticVertexId": self.semantic_vertex_id,
            "sourceVertexId": self.source_vertex_id,
            "copyVertexId": self.copy_vertex_id,
            "separation": self.separation,
        }


@dataclass(frozen=True)
class CopyPrimitiveActivation:
    semantic_primitive_id: str
    source_primitive_id: str
    copy_primitive_id: str
    separation: float
    source_opacity_scale: float
    copy_opacity_scale: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "semanticPrimitiveId": self.semantic_primitive_id,
            "sourcePrimitiveId": self.source_primitive_id,
            "copyPrimitiveId": self.copy_primitive_id,
            "separation": self.separation,
            "sourceOpacityScale": self.source_opacity_scale,
            "copyOpacityScale": self.copy_opacity_scale,
        }


@dataclass(frozen=True)
class CopyIdentityHandoffFrame:
    handoff_id: str
    activation_distance: float
    maximum_separation: float
    source_opacity_scale: float
    vertex_separations: tuple[CopyVertexSeparation, ...]
    face_activations: tuple[CopyPrimitiveActivation, ...]
    stroke_activations: tuple[CopyPrimitiveActivation, ...]
    schema: str = COPY_IDENTITY_HANDOFF_FRAME_SCHEMA

    @property
    def source_face_opacity_scales(self) -> dict[str, float]:
        return {
            item.source_primitive_id: item.source_opacity_scale
            for item in self.face_activations
        }

    @property
    def source_stroke_opacity_scales(self) -> dict[str, float]:
        return {
            item.source_primitive_id: item.source_opacity_scale
            for item in self.stroke_activations
        }

    @property
    def copy_face_opacity_scales(self) -> dict[str, float]:
        return {
            item.copy_primitive_id: item.copy_opacity_scale
            for item in self.face_activations
        }

    @property
    def copy_stroke_opacity_scales(self) -> dict[str, float]:
        return {
            item.copy_primitive_id: item.copy_opacity_scale
            for item in self.stroke_activations
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "handoffId": self.handoff_id,
            "activationDistance": self.activation_distance,
            "maximumSeparation": self.maximum_separation,
            "sourceOpacityScale": self.source_opacity_scale,
            "coincidentOwner": "copy",
            "vertexSeparations": [item.to_dict() for item in self.vertex_separations],
            "faceActivations": [item.to_dict() for item in self.face_activations],
            "strokeActivations": [item.to_dict() for item in self.stroke_activations],
        }


def compute_copy_identity_handoff(
    handoff: CopyIdentityHandoffMap,
    *,
    source_positions: Mapping[str, Sequence[float]],
    copy_positions: Mapping[str, Sequence[float]],
    final_point_provider: PointProvider | None = None,
) -> CopyIdentityHandoffFrame:
    """Evaluate one deterministic source/copy display-ownership handoff.

    ``source_positions`` and ``copy_positions`` use the runtime vertex IDs
    frozen in ``handoff``.  ``final_point_provider`` may project world points
    into the exact coordinate space used by the overlay.  The copy always
    remains at full authored opacity; only paired source primitives are scaled.
    """

    if not isinstance(handoff, CopyIdentityHandoffMap):
        raise TypeError("handoff must be CopyIdentityHandoffMap")
    if final_point_provider is not None and not callable(final_point_provider):
        raise TypeError("final_point_provider must be callable or None")

    def shown(point: Sequence[float], label: str) -> np.ndarray:
        value = _point3(point, label)
        if final_point_provider is not None:
            value = _point3(final_point_provider(value), f"displayed {label}")
        return value

    vertex_separations: list[CopyVertexSeparation] = []
    separation_by_semantic_id: dict[str, float] = {}
    for pair in handoff.vertex_pairs:
        try:
            source = source_positions[pair.source_vertex_id]
        except KeyError as exc:
            raise ValueError(
                f"missing source vertex position {pair.source_vertex_id!r}"
            ) from exc
        try:
            copied = copy_positions[pair.copy_vertex_id]
        except KeyError as exc:
            raise ValueError(
                f"missing copy vertex position {pair.copy_vertex_id!r}"
            ) from exc
        separation = float(
            np.linalg.norm(
                shown(copied, pair.copy_vertex_id)
                - shown(source, pair.source_vertex_id)
            )
        )
        if not np.isfinite(separation):
            raise ValueError(
                f"copy separation for {pair.semantic_vertex_id!r} is not finite"
            )
        separation_by_semantic_id[pair.semantic_vertex_id] = separation
        vertex_separations.append(
            CopyVertexSeparation(
                pair.semantic_vertex_id,
                pair.source_vertex_id,
                pair.copy_vertex_id,
                separation,
            )
        )

    distance = handoff.policy.activation_distance

    def activation(pair: CopyPrimitivePair) -> CopyPrimitiveActivation:
        separation = max(
            separation_by_semantic_id[item] for item in pair.vertex_pair_ids
        )
        return CopyPrimitiveActivation(
            pair.semantic_primitive_id,
            pair.source_primitive_id,
            pair.copy_primitive_id,
            separation,
            _smoothstep01(separation / distance),
        )

    maximum = max(separation_by_semantic_id.values(), default=0.0)
    return CopyIdentityHandoffFrame(
        handoff_id=handoff.handoff_id,
        activation_distance=distance,
        maximum_separation=maximum,
        source_opacity_scale=_smoothstep01(maximum / distance),
        vertex_separations=tuple(vertex_separations),
        face_activations=tuple(activation(item) for item in handoff.face_pairs),
        stroke_activations=tuple(activation(item) for item in handoff.stroke_pairs),
    )


__all__ = [
    "COPY_IDENTITY_HANDOFF_FRAME_SCHEMA",
    "CopyIdentityHandoffFrame",
    "CopyPrimitiveActivation",
    "CopyVertexSeparation",
    "PointProvider",
    "compute_copy_identity_handoff",
]
