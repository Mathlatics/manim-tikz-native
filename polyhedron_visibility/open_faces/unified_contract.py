"""Contracts for renderer-neutral open-face unified compositing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

import numpy as np

from ..topology import ParameterInterval
from ..visibility import VisibilityKind
from .trace import OpenFaceVisibilityFrame


OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA = (
    "manim-open-convex-face-unified-compositing/v1"
)


class OpenFaceUnifiedCompositingError(ValueError):
    """One exact deterministic painter graph could not be produced."""


class OpenFacePaintPolicy(str, Enum):
    DIAGRAMMATIC = "diagrammatic"
    PHYSICAL = "physical"

    @classmethod
    def parse(cls, value: "OpenFacePaintPolicy | str") -> "OpenFacePaintPolicy":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise OpenFaceUnifiedCompositingError(
                "paint_policy must be 'diagrammatic' or 'physical'"
            ) from exc


@dataclass(frozen=True, slots=True)
class OpenFaceUnifiedCompositingLimits:
    max_faces: int = 64
    max_paths: int = 128
    max_line_face_pairs: int = 4096
    max_line_line_pairs: int = 4096
    max_fragments_per_path: int = 1024
    max_total_fragments: int = 32768
    max_relations: int = 262144

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


OPEN_FACE_UNIFIED_COMPOSITING_LIMITS = OpenFaceUnifiedCompositingLimits()


@dataclass(frozen=True, slots=True)
class OpenFacePaintFace:
    item_id: str
    face_id: str
    logical_surface_id: str

    def __post_init__(self) -> None:
        if self.item_id != f"face:{self.face_id}":
            raise ValueError("paint-face item_id must be derived from face_id")
        if not self.face_id or not self.logical_surface_id:
            raise ValueError("paint-face identities must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "itemId": self.item_id,
            "faceId": self.face_id,
            "logicalSurfaceId": self.logical_surface_id,
        }


@dataclass(frozen=True, slots=True)
class PaintPathFragment:
    fragment_id: str
    source_path_id: str
    parameter_interval: ParameterInterval
    visibility_kind: VisibilityKind
    occluder_face_ids: tuple[str, ...] = ()
    occluder_logical_surface_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.fragment_id or not self.source_path_id:
            raise ValueError("path-fragment identities must be non-empty")
        if not isinstance(self.parameter_interval, ParameterInterval):
            raise TypeError("parameter_interval must be a ParameterInterval")
        if not isinstance(self.visibility_kind, VisibilityKind):
            raise TypeError("visibility_kind must be a VisibilityKind")
        if tuple(sorted(set(self.occluder_face_ids))) != self.occluder_face_ids:
            raise ValueError("occluder_face_ids must be sorted and unique")
        if (
            tuple(sorted(set(self.occluder_logical_surface_ids)))
            != self.occluder_logical_surface_ids
        ):
            raise ValueError(
                "occluder_logical_surface_ids must be sorted and unique"
            )
        if self.visibility_kind is VisibilityKind.VISIBLE and (
            self.occluder_face_ids or self.occluder_logical_surface_ids
        ):
            raise ValueError("visible fragments cannot carry occluders")
        if self.visibility_kind is VisibilityKind.HIDDEN and not self.occluder_face_ids:
            raise ValueError("hidden fragments require an occluder identity")

    @property
    def item_id(self) -> str:
        return self.fragment_id

    def to_dict(self) -> dict[str, object]:
        return {
            "fragmentId": self.fragment_id,
            "sourcePathId": self.source_path_id,
            "parameterInterval": {
                "start": self.parameter_interval.start,
                "end": self.parameter_interval.end,
            },
            "visibilityKind": self.visibility_kind.value,
            "occluderFaceIds": list(self.occluder_face_ids),
            "occluderLogicalSurfaceIds": list(
                self.occluder_logical_surface_ids
            ),
        }


@dataclass(frozen=True, slots=True)
class OpenFacePaintRelation:
    far_item_id: str
    near_item_id: str
    reason: str
    minimum_depth_difference: float
    maximum_depth_difference: float
    overlap_measure: float

    def __post_init__(self) -> None:
        if not self.far_item_id or not self.near_item_id or not self.reason:
            raise ValueError("paint relation identities and reason must be non-empty")
        if self.far_item_id == self.near_item_id:
            raise ValueError("paint relation cannot be a self relation")
        values = (
            self.minimum_depth_difference,
            self.maximum_depth_difference,
            self.overlap_measure,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("paint relation metrics must be finite")
        if self.minimum_depth_difference > self.maximum_depth_difference:
            raise ValueError("paint relation depth range is reversed")
        if self.overlap_measure < 0.0:
            raise ValueError("overlap_measure must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "farItemId": self.far_item_id,
            "nearItemId": self.near_item_id,
            "reason": self.reason,
            "minimumDepthDifference": self.minimum_depth_difference,
            "maximumDepthDifference": self.maximum_depth_difference,
            "overlapMeasure": self.overlap_measure,
        }


@dataclass(frozen=True, slots=True)
class OpenFaceUnifiedCompositingFrame:
    visibility: OpenFaceVisibilityFrame
    paint_policy: OpenFacePaintPolicy
    faces: tuple[OpenFacePaintFace, ...]
    path_fragments: tuple[PaintPathFragment, ...]
    order_relations: tuple[OpenFacePaintRelation, ...]
    draw_order: tuple[str, ...]
    schema: str = OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA:
            raise ValueError("invalid unified-compositing schema")
        if not isinstance(self.paint_policy, OpenFacePaintPolicy):
            raise TypeError("paint_policy must be an OpenFacePaintPolicy")
        item_ids = self.item_ids
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("paint item identities must be unique")
        if set(self.draw_order) != set(item_ids) or len(self.draw_order) != len(item_ids):
            raise ValueError("draw_order must cover every paint item exactly once")

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(
            (
                *(item.item_id for item in self.faces),
                *(item.fragment_id for item in self.path_fragments),
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "paintPolicy": self.paint_policy.value,
            "visibility": self.visibility.to_dict(),
            "faces": [item.to_dict() for item in self.faces],
            "pathFragments": [item.to_dict() for item in self.path_fragments],
            "orderRelations": [item.to_dict() for item in self.order_relations],
            "drawOrder": list(self.draw_order),
        }


def canonical_open_face_unified_compositing_json(
    frame: OpenFaceUnifiedCompositingFrame,
) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "OPEN_FACE_UNIFIED_COMPOSITING_LIMITS",
    "OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA",
    "OpenFacePaintFace",
    "OpenFacePaintPolicy",
    "OpenFacePaintRelation",
    "OpenFaceUnifiedCompositingError",
    "OpenFaceUnifiedCompositingFrame",
    "OpenFaceUnifiedCompositingLimits",
    "PaintPathFragment",
    "canonical_open_face_unified_compositing_json",
]
