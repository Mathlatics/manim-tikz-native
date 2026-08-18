from __future__ import annotations

from dataclasses import dataclass
import json

from ..trace import VisibilityFrame
from .contract import SectionPlane3D


SECTION_TRACE_SCHEMA = "manim-convex-polyhedron-section-trace/v1"
SECTIONED_VISIBILITY_TRACE_SCHEMA = (
    "manim-convex-polyhedron-sectioned-visibility-trace/v1"
)


@dataclass(frozen=True)
class SolidBoundaryHit:
    role: str
    parameter: float
    position: tuple[float, float, float]
    face_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "parameter": self.parameter,
            "position": list(self.position),
            "faceIds": list(self.face_ids),
        }


@dataclass(frozen=True)
class SegmentSolidIntersection:
    kind: str
    inside_parameter_interval: tuple[float, float] | None
    starts_inside: bool
    ends_inside: bool
    hits: tuple[SolidBoundaryHit, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "insideParameterInterval": (
                None
                if self.inside_parameter_interval is None
                else list(self.inside_parameter_interval)
            ),
            "startsInside": self.starts_inside,
            "endsInside": self.ends_inside,
            "hits": [item.to_dict() for item in self.hits],
        }


@dataclass(frozen=True)
class SectionPoint:
    point_id: str
    position: tuple[float, float, float]
    source_edge_ids: tuple[str, ...]
    source_vertex_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "pointId": self.point_id,
            "position": list(self.position),
            "sourceEdgeIds": list(self.source_edge_ids),
            "sourceVertexIds": list(self.source_vertex_ids),
        }


@dataclass(frozen=True)
class SectionBoundarySegment:
    segment_id: str
    start_point_id: str
    end_point_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "segmentId": self.segment_id,
            "startPointId": self.start_point_id,
            "endPointId": self.end_point_id,
        }


@dataclass(frozen=True)
class ConvexSectionFrame:
    section_id: str
    plane: SectionPlane3D
    kind: str
    points: tuple[SectionPoint, ...]
    boundary_segments: tuple[SectionBoundarySegment, ...]
    schema: str = SECTION_TRACE_SCHEMA

    @property
    def point_map(self) -> dict[str, SectionPoint]:
        return {item.point_id: item for item in self.points}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sectionId": self.section_id,
            "plane": self.plane.to_dict(),
            "kind": self.kind,
            "points": [item.to_dict() for item in self.points],
            "boundarySegments": [item.to_dict() for item in self.boundary_segments],
        }


@dataclass(frozen=True)
class NamedStrokeSolidIntersection:
    source_edge_id: str
    intersection: SegmentSolidIntersection

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceEdgeId": self.source_edge_id,
            "intersection": self.intersection.to_dict(),
        }


@dataclass(frozen=True)
class SectionedVisibilityFrame:
    section: ConvexSectionFrame
    source_visibility: VisibilityFrame
    boundary_visibility: VisibilityFrame | None
    stroke_intersections: tuple[NamedStrokeSolidIntersection, ...]
    schema: str = SECTIONED_VISIBILITY_TRACE_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "section": self.section.to_dict(),
            "sourceVisibility": self.source_visibility.to_dict(),
            "boundaryVisibility": (
                None
                if self.boundary_visibility is None
                else self.boundary_visibility.to_dict()
            ),
            "strokeIntersections": [
                item.to_dict() for item in self.stroke_intersections
            ],
        }


def canonical_section_trace_json(frame: ConvexSectionFrame) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sectioned_trace_json(frame: SectionedVisibilityFrame) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "ConvexSectionFrame",
    "NamedStrokeSolidIntersection",
    "SECTION_TRACE_SCHEMA",
    "SECTIONED_VISIBILITY_TRACE_SCHEMA",
    "SectionBoundarySegment",
    "SectionPoint",
    "SectionedVisibilityFrame",
    "SegmentSolidIntersection",
    "SolidBoundaryHit",
    "canonical_section_trace_json",
    "canonical_sectioned_trace_json",
]
