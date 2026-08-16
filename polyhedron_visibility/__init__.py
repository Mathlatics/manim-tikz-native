"""Projection-aware hidden-line removal for registered convex face systems."""

from .contract import (
    ContractError,
    FaceSpec,
    ResolvedTolerance,
    StrokeSpec,
    TolerancePolicy,
    VertexSpec,
    VISIBILITY_MODEL_SCHEMA,
    VisibilityModel,
)
from .parallel_solver import (
    ParallelView,
    SolverError,
    compute_frame_visibility,
    segment_face_occlusion_interval,
)
from .trace import (
    EdgeVisibility,
    RawOcclusionInterval,
    SkippedFace,
    VISIBILITY_TRACE_SCHEMA,
    VisibilityFrame,
    VisibilitySpan,
    canonical_trace_json,
)

__all__ = [
    "ContractError",
    "EdgeVisibility",
    "FaceSpec",
    "ParallelView",
    "RawOcclusionInterval",
    "ResolvedTolerance",
    "SkippedFace",
    "SolverError",
    "StrokeSpec",
    "TolerancePolicy",
    "VISIBILITY_MODEL_SCHEMA",
    "VISIBILITY_TRACE_SCHEMA",
    "VertexSpec",
    "VisibilityFrame",
    "VisibilityModel",
    "VisibilitySpan",
    "canonical_trace_json",
    "compute_frame_visibility",
    "segment_face_occlusion_interval",
]
