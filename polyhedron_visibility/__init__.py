"""Projection-aware hidden-line removal for registered convex face systems.

The public API is loaded lazily.  This preserves all existing root-level
exports while allowing pure modules such as ``polyhedron_visibility.kernel``
to be imported without importing Manim and its renderer bindings first.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "AutoOcclusion3D": (".api", "AutoOcclusion3D"),
    "ParallelProjection": (".api", "ParallelProjection"),
    "OcclusionAuthoringError": (".authoring", "OcclusionAuthoringError"),
    "OcclusionScene3D": (".authoring", "OcclusionScene3D"),
    "OcclusionBindingError": (".binding", "OcclusionBindingError"),
    "OcclusionCapacityError": (".binding", "OcclusionCapacityError"),
    "ContractError": (".contract", "ContractError"),
    "FaceSpec": (".contract", "FaceSpec"),
    "ResolvedTolerance": (".contract", "ResolvedTolerance"),
    "StrokeSpec": (".contract", "StrokeSpec"),
    "TolerancePolicy": (".contract", "TolerancePolicy"),
    "VertexSpec": (".contract", "VertexSpec"),
    "VISIBILITY_MODEL_SCHEMA": (".contract", "VISIBILITY_MODEL_SCHEMA"),
    "VisibilityModel": (".contract", "VisibilityModel"),
    "ParallelView": (".parallel_solver", "ParallelView"),
    "SolverError": (".parallel_solver", "SolverError"),
    "compute_frame_visibility": (".parallel_solver", "compute_frame_visibility"),
    "segment_face_occlusion_interval": (
        ".parallel_solver",
        "segment_face_occlusion_interval",
    ),
    "EdgeVisibility": (".trace", "EdgeVisibility"),
    "FaceToleranceTrace": (".trace", "FaceToleranceTrace"),
    "RawOcclusionInterval": (".trace", "RawOcclusionInterval"),
    "SkippedFace": (".trace", "SkippedFace"),
    "VISIBILITY_TRACE_SCHEMA": (".trace", "VISIBILITY_TRACE_SCHEMA"),
    "VisibilityFrame": (".trace", "VisibilityFrame"),
    "VisibilitySpan": (".trace", "VisibilitySpan"),
    "canonical_trace_json": (".trace", "canonical_trace_json"),
    "OcclusionStyle": (".style", "OcclusionStyle"),
    "OcclusionStyleError": (".style", "OcclusionStyleError"),
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
