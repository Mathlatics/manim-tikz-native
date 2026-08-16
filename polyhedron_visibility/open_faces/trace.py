from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

from ..contract import ResolvedTolerance


OPEN_FACE_TRACE_SCHEMA = "manim-open-convex-face-visibility-trace/v1"


@dataclass(frozen=True)
class OpenFaceRawOcclusionInterval:
    face_id: str
    logical_surface_id: str
    start: float
    end: float

    def to_dict(self) -> dict[str, object]:
        return {
            "faceId": self.face_id,
            "logicalSurfaceId": self.logical_surface_id,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class OpenFaceSkippedOccluder:
    face_id: str
    logical_surface_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "faceId": self.face_id,
            "logicalSurfaceId": self.logical_surface_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OpenFaceToleranceTrace:
    face_id: str
    logical_surface_id: str
    world: float
    boundary: float
    depth: float
    angular: float

    def to_dict(self) -> dict[str, object]:
        return {
            "faceId": self.face_id,
            "logicalSurfaceId": self.logical_surface_id,
            "world": self.world,
            "boundary": self.boundary,
            "depth": self.depth,
            "angular": self.angular,
        }


@dataclass(frozen=True)
class OpenFaceSeamState:
    seam_id: str
    policy: str
    face_ids: tuple[str, str]
    logical_surface_ids: tuple[str, str]
    vertex_ids: tuple[str, str]
    state: str
    dihedral_radians: float
    cosine: float
    signed_sine: float
    world_tolerance: float
    angular_tolerance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "seamId": self.seam_id,
            "policy": self.policy,
            "faceIds": list(self.face_ids),
            "logicalSurfaceIds": list(self.logical_surface_ids),
            "vertexIds": list(self.vertex_ids),
            "state": self.state,
            "dihedralRadians": self.dihedral_radians,
            "cosine": self.cosine,
            "signedSine": self.signed_sine,
            "worldTolerance": self.world_tolerance,
            "angularTolerance": self.angular_tolerance,
        }


@dataclass(frozen=True)
class OpenFaceVisibilitySpan:
    start: float
    end: float
    kind: str
    occluder_face_ids: tuple[str, ...] = ()
    occluder_logical_surface_ids: tuple[str, ...] = ()
    face_level: int = 0
    surface_level: int = 0

    @property
    def level(self) -> int:
        """Compatibility convenience: rendering only needs the face count."""

        return self.face_level

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "occluderFaceIds": list(self.occluder_face_ids),
            "occluderLogicalSurfaceIds": list(self.occluder_logical_surface_ids),
            "faceLevel": self.face_level,
            "surfaceLevel": self.surface_level,
        }


@dataclass(frozen=True)
class OpenFaceEdgeVisibility:
    source_edge_id: str
    raw_intervals: tuple[OpenFaceRawOcclusionInterval, ...]
    skipped_occluders: tuple[OpenFaceSkippedOccluder, ...]
    spans: tuple[OpenFaceVisibilitySpan, ...]
    parameter_epsilon: float

    @property
    def skipped_faces(self) -> tuple[OpenFaceSkippedOccluder, ...]:
        return self.skipped_occluders

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceEdgeId": self.source_edge_id,
            "rawIntervals": [item.to_dict() for item in self.raw_intervals],
            "skippedOccluders": [item.to_dict() for item in self.skipped_occluders],
            "spans": [item.to_dict() for item in self.spans],
            "parameterEpsilon": self.parameter_epsilon,
        }


@dataclass(frozen=True)
class OpenFaceVisibilityFrame:
    visibility_group_id: str
    model_schema: str
    topology: str
    projection_matrix: tuple[tuple[float, float, float], ...]
    view_direction: tuple[float, float, float]
    tolerance: ResolvedTolerance
    face_tolerances: tuple[OpenFaceToleranceTrace, ...]
    seam_states: tuple[OpenFaceSeamState, ...]
    edges: tuple[OpenFaceEdgeVisibility, ...]
    advisory_face_draw_order: tuple[str, ...]
    schema: str = OPEN_FACE_TRACE_SCHEMA

    @property
    def edge_map(self) -> Mapping[str, OpenFaceEdgeVisibility]:
        return {item.source_edge_id: item for item in self.edges}

    @property
    def seam_state_map(self) -> Mapping[str, OpenFaceSeamState]:
        return {item.seam_id: item for item in self.seam_states}

    @property
    def face_draw_order(self) -> tuple[str, ...]:
        return self.advisory_face_draw_order

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "modelSchema": self.model_schema,
            "topology": self.topology,
            "visibilityGroupId": self.visibility_group_id,
            "projectionMatrix": [list(row) for row in self.projection_matrix],
            "viewDirection": list(self.view_direction),
            "tolerance": self.tolerance.to_dict(),
            "faceTolerances": [item.to_dict() for item in self.face_tolerances],
            "seams": [item.to_dict() for item in self.seam_states],
            "advisoryFaceDrawOrder": list(self.advisory_face_draw_order),
            "edges": [item.to_dict() for item in self.edges],
        }


def canonical_open_face_trace_json(frame: OpenFaceVisibilityFrame) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "OPEN_FACE_TRACE_SCHEMA",
    "OpenFaceEdgeVisibility",
    "OpenFaceRawOcclusionInterval",
    "OpenFaceSeamState",
    "OpenFaceSkippedOccluder",
    "OpenFaceToleranceTrace",
    "OpenFaceVisibilityFrame",
    "OpenFaceVisibilitySpan",
    "canonical_open_face_trace_json",
]
