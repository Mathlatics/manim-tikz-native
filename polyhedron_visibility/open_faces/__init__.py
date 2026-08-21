"""Finite independent convex-face visibility under parallel projection.

The package exports its public API lazily so renderer-neutral modules such as
``open_faces.unified_compositing`` can be imported without importing Manim.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "ResolvedTolerance": ("..contract", "ResolvedTolerance"),
    "TolerancePolicy": ("..contract", "TolerancePolicy"),
    "OpenFaceAuthoringError": (".authoring", "OpenFaceAuthoringError"),
    "OpenFaceScene3D": (".authoring", "OpenFaceScene3D"),
    "OpenFaceVertexPositionProvider": (
        ".authoring",
        "OpenFaceVertexPositionProvider",
    ),
    "ARTICULATED_HINGE_POLICY": (".contract", "ARTICULATED_HINGE_POLICY"),
    "OPEN_FACE_MODEL_SCHEMA": (".contract", "OPEN_FACE_MODEL_SCHEMA"),
    "OPEN_FACE_TOPOLOGY": (".contract", "OPEN_FACE_TOPOLOGY"),
    "OpenFaceContractError": (".contract", "OpenFaceContractError"),
    "OpenFaceSeamSpec": (".contract", "OpenFaceSeamSpec"),
    "OpenFaceSpec": (".contract", "OpenFaceSpec"),
    "OpenFaceStrokeSpec": (".contract", "OpenFaceStrokeSpec"),
    "OpenFaceVertexSpec": (".contract", "OpenFaceVertexSpec"),
    "OpenFaceVisibilityModel": (".contract", "OpenFaceVisibilityModel"),
    "OpenFaceSolverError": (".solver", "OpenFaceSolverError"),
    "compute_open_face_visibility": (".solver", "compute_open_face_visibility"),
    "OPEN_FACE_BINDING_SCALE_LIMITS": (
        ".manim",
        "OPEN_FACE_BINDING_SCALE_LIMITS",
    ),
    "OpenFaceBindingScaleError": (".manim", "OpenFaceBindingScaleError"),
    "OpenFaceBindingScaleLimits": (".manim", "OpenFaceBindingScaleLimits"),
    "OpenFaceOcclusion3D": (".manim", "OpenFaceOcclusion3D"),
    "OPEN_FACE_TRACE_SCHEMA": (".trace", "OPEN_FACE_TRACE_SCHEMA"),
    "OpenFaceEdgeVisibility": (".trace", "OpenFaceEdgeVisibility"),
    "OpenFaceRawOcclusionInterval": (
        ".trace",
        "OpenFaceRawOcclusionInterval",
    ),
    "OpenFaceSeamState": (".trace", "OpenFaceSeamState"),
    "OpenFaceSkippedOccluder": (".trace", "OpenFaceSkippedOccluder"),
    "OpenFaceToleranceTrace": (".trace", "OpenFaceToleranceTrace"),
    "OpenFaceVisibilityFrame": (".trace", "OpenFaceVisibilityFrame"),
    "OpenFaceVisibilitySpan": (".trace", "OpenFaceVisibilitySpan"),
    "canonical_open_face_trace_json": (
        ".trace",
        "canonical_open_face_trace_json",
    ),
    "OPEN_FACE_UNIFIED_COMPOSITING_LIMITS": (
        ".unified_compositing",
        "OPEN_FACE_UNIFIED_COMPOSITING_LIMITS",
    ),
    "OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA": (
        ".unified_compositing",
        "OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA",
    ),
    "OpenFacePaintFace": (".unified_compositing", "OpenFacePaintFace"),
    "OpenFacePaintPolicy": (".unified_compositing", "OpenFacePaintPolicy"),
    "OpenFacePaintRelation": (
        ".unified_compositing",
        "OpenFacePaintRelation",
    ),
    "OpenFaceUnifiedCompositingError": (
        ".unified_compositing",
        "OpenFaceUnifiedCompositingError",
    ),
    "OpenFaceUnifiedCompositingFrame": (
        ".unified_compositing",
        "OpenFaceUnifiedCompositingFrame",
    ),
    "OpenFaceUnifiedCompositingLimits": (
        ".unified_compositing",
        "OpenFaceUnifiedCompositingLimits",
    ),
    "PaintPathFragment": (".unified_compositing", "PaintPathFragment"),
    "canonical_open_face_unified_compositing_json": (
        ".unified_compositing",
        "canonical_open_face_unified_compositing_json",
    ),
    "compute_open_face_unified_compositing": (
        ".unified_compositing",
        "compute_open_face_unified_compositing",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
