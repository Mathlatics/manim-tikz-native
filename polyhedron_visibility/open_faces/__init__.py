"""Finite independent convex-face visibility under parallel projection.

This package is deliberately separate from the frozen closed-polyhedron v1
contract.  It models named zero-thickness occluding panels and explicit
articulated hinges without weakening the closed-manifold validator.
"""

from ..contract import ResolvedTolerance, TolerancePolicy
from .authoring import (
    OpenFaceAuthoringError,
    OpenFaceScene3D,
    OpenFaceVertexPositionProvider,
)
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
from .manim import (
    OPEN_FACE_BINDING_SCALE_LIMITS,
    OpenFaceBindingScaleError,
    OpenFaceBindingScaleLimits,
    OpenFaceOcclusion3D,
)
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
    "OPEN_FACE_BINDING_SCALE_LIMITS",
    "OPEN_FACE_TOPOLOGY",
    "OPEN_FACE_TRACE_SCHEMA",
    "OpenFaceContractError",
    "OpenFaceAuthoringError",
    "OpenFaceBindingScaleError",
    "OpenFaceBindingScaleLimits",
    "OpenFaceEdgeVisibility",
    "OpenFaceRawOcclusionInterval",
    "OpenFaceOcclusion3D",
    "OpenFaceSeamSpec",
    "OpenFaceScene3D",
    "OpenFaceSeamState",
    "OpenFaceSkippedOccluder",
    "OpenFaceSolverError",
    "OpenFaceSpec",
    "OpenFaceStrokeSpec",
    "OpenFaceToleranceTrace",
    "OpenFaceVertexSpec",
    "OpenFaceVertexPositionProvider",
    "OpenFaceVisibilityFrame",
    "OpenFaceVisibilityModel",
    "OpenFaceVisibilitySpan",
    "ResolvedTolerance",
    "TolerancePolicy",
    "canonical_open_face_trace_json",
    "compute_open_face_visibility",
]
