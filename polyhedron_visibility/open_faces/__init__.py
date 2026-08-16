"""Finite independent convex-face visibility under parallel projection.

This package is deliberately separate from the frozen closed-polyhedron v1
contract.  It models named zero-thickness occluding panels and explicit
articulated hinges without weakening the closed-manifold validator.
"""

from ..contract import ResolvedTolerance, TolerancePolicy
from .contract import (
    ARTICULATED_HINGE_POLICY,
    OPEN_FACE_MODEL_SCHEMA,
    OPEN_FACE_TOPOLOGY,
    OpenFaceContractError,
    OpenFaceSeamSpec,
    OpenFaceSpec,
    OpenFaceStrokeSpec,
    OpenFaceVertexSpec,
    OpenFaceVisibilityModel,
)
from .solver import OpenFaceSolverError, compute_open_face_visibility
from .trace import (
    OPEN_FACE_TRACE_SCHEMA,
    OpenFaceEdgeVisibility,
    OpenFaceRawOcclusionInterval,
    OpenFaceSeamState,
    OpenFaceSkippedOccluder,
    OpenFaceToleranceTrace,
    OpenFaceVisibilityFrame,
    OpenFaceVisibilitySpan,
    canonical_open_face_trace_json,
)


__all__ = [
    "ARTICULATED_HINGE_POLICY",
    "OPEN_FACE_MODEL_SCHEMA",
    "OPEN_FACE_TOPOLOGY",
    "OPEN_FACE_TRACE_SCHEMA",
    "OpenFaceContractError",
    "OpenFaceEdgeVisibility",
    "OpenFaceRawOcclusionInterval",
    "OpenFaceSeamSpec",
    "OpenFaceSeamState",
    "OpenFaceSkippedOccluder",
    "OpenFaceSolverError",
    "OpenFaceSpec",
    "OpenFaceStrokeSpec",
    "OpenFaceToleranceTrace",
    "OpenFaceVertexSpec",
    "OpenFaceVisibilityFrame",
    "OpenFaceVisibilityModel",
    "OpenFaceVisibilitySpan",
    "ResolvedTolerance",
    "TolerancePolicy",
    "canonical_open_face_trace_json",
    "compute_open_face_visibility",
]
