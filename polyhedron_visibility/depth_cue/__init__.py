"""Didactic face shading and silhouette emphasis for parallel 3D views."""

from .contract import (
    FACE_DEPTH_CUE_TRACE_SCHEMA,
    EdgeDepthCue,
    FaceDepthCue,
    FaceDepthCueFrame,
    FaceDepthCueStyle,
)
from .manim import DepthCuedAutoOcclusion3D, FaceDepthCueLayer
from .solver import FaceDepthCueError, compute_face_depth_cue

__all__ = [
    "DepthCuedAutoOcclusion3D",
    "EdgeDepthCue",
    "FACE_DEPTH_CUE_TRACE_SCHEMA",
    "FaceDepthCue",
    "FaceDepthCueError",
    "FaceDepthCueFrame",
    "FaceDepthCueLayer",
    "FaceDepthCueStyle",
    "compute_face_depth_cue",
]
