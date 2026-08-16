"""Projection-aware hidden-line removal for registered convex face systems."""

from .api import AutoOcclusion3D, ParallelProjection
from .authoring import OcclusionAuthoringError, OcclusionScene3D
from .binding import OcclusionBindingError, OcclusionCapacityError
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
    FaceToleranceTrace,
    RawOcclusionInterval,
    SkippedFace,
    VISIBILITY_TRACE_SCHEMA,
    VisibilityFrame,
    VisibilitySpan,
    canonical_trace_json,
)
from .style import OcclusionStyle, OcclusionStyleError

__all__ = [
    "AutoOcclusion3D",
    "ContractError",
    "EdgeVisibility",
    "FaceToleranceTrace",
    "FaceSpec",
    "OcclusionAuthoringError",
    "OcclusionBindingError",
    "OcclusionCapacityError",
    "OcclusionScene3D",
    "OcclusionStyle",
    "OcclusionStyleError",
    "ParallelView",
    "ParallelProjection",
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
