from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

from .contract import ResolvedTolerance


VISIBILITY_TRACE_SCHEMA = "manim-convex-polyhedron-visibility-trace/v1"


@dataclass(frozen=True)
class RawOcclusionInterval:
    face_id: str
    start: float
    end: float

    def to_dict(self) -> dict[str, object]:
        return {"faceId": self.face_id, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class SkippedFace:
    face_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"faceId": self.face_id, "reason": self.reason}


@dataclass(frozen=True)
class VisibilitySpan:
    start: float
    end: float
    kind: str
    occluder_face_ids: tuple[str, ...] = ()
    level: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "occluderFaceIds": list(self.occluder_face_ids),
            "level": self.level,
        }


@dataclass(frozen=True)
class EdgeVisibility:
    source_edge_id: str
    raw_intervals: tuple[RawOcclusionInterval, ...]
    skipped_faces: tuple[SkippedFace, ...]
    spans: tuple[VisibilitySpan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceEdgeId": self.source_edge_id,
            "rawIntervals": [item.to_dict() for item in self.raw_intervals],
            "skippedFaces": [item.to_dict() for item in self.skipped_faces],
            "spans": [item.to_dict() for item in self.spans],
        }


@dataclass(frozen=True)
class VisibilityFrame:
    visibility_group_id: str
    projection_matrix: tuple[tuple[float, float, float], ...]
    view_direction: tuple[float, float, float]
    tolerance: ResolvedTolerance
    edges: tuple[EdgeVisibility, ...]
    face_draw_order: tuple[str, ...]
    schema: str = VISIBILITY_TRACE_SCHEMA

    @property
    def edge_map(self) -> Mapping[str, EdgeVisibility]:
        return {item.source_edge_id: item for item in self.edges}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "visibilityGroupId": self.visibility_group_id,
            "projectionMatrix": [list(row) for row in self.projection_matrix],
            "viewDirection": list(self.view_direction),
            "tolerance": self.tolerance.to_dict(),
            "faceDrawOrder": list(self.face_draw_order),
            "edges": [item.to_dict() for item in self.edges],
        }


def canonical_trace_json(frame: VisibilityFrame) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "EdgeVisibility",
    "RawOcclusionInterval",
    "SkippedFace",
    "VISIBILITY_TRACE_SCHEMA",
    "VisibilityFrame",
    "VisibilitySpan",
    "canonical_trace_json",
]
