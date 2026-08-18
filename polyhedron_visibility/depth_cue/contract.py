from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence


FACE_DEPTH_CUE_TRACE_SCHEMA = "manim-face-depth-cue-trace/v1"


class FaceDepthCueContractError(ValueError):
    """Raised when a depth-cue style cannot be interpreted safely."""


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise FaceDepthCueContractError(f"{label} must be finite")
    return result


def _positive(value: float, label: str) -> float:
    result = _finite(value, label)
    if result <= 0:
        raise FaceDepthCueContractError(f"{label} must be positive")
    return result


def _non_negative(value: float, label: str) -> float:
    result = _finite(value, label)
    if result < 0:
        raise FaceDepthCueContractError(f"{label} must be non-negative")
    return result


def _unit_interval(value: float, label: str) -> float:
    result = _non_negative(value, label)
    if result > 1.0:
        raise FaceDepthCueContractError(f"{label} must not exceed 1")
    return result


def _vector3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise FaceDepthCueContractError(f"{label} must contain three numbers")
    result = tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))
    if sum(item * item for item in result) <= 0:
        raise FaceDepthCueContractError(f"{label} must be non-zero")
    return result  # type: ignore[return-value]


def _rgb3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    result = _vector3(value, label)
    if any(item < 0.0 or item > 1.0 for item in result):
        raise FaceDepthCueContractError(
            f"{label} components must stay between 0 and 1"
        )
    return result


@dataclass(frozen=True)
class FaceDepthCueStyle:
    """Conservative classroom-oriented depth cues.

    Face opacity is multiplied rather than replaced, so every authored face
    keeps its own base opacity.  Distant faces are also desaturated and blended
    toward ``fog_color_rgb``.  This atmospheric cue is deliberately stronger
    than physically neutral lighting because classroom diagrams must remain
    readable when several transparent faces overlap.  ``light_direction_view``
    is expressed in screen-right, screen-up, and camera-facing coordinates.
    """

    minimum_opacity_scale: float = 0.25
    maximum_opacity_scale: float = 2.55
    facing_opacity_weight: float = 0.35
    depth_opacity_weight: float = 0.65
    ambient_brightness: float = 0.52
    diffuse_brightness: float = 0.78
    minimum_saturation_scale: float = 0.32
    maximum_saturation_scale: float = 1.10
    far_fog_strength: float = 0.58
    fog_color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    maximum_hue_shift_turns: float = 0.055
    back_facing_opacity_scale: float = 0.10
    light_direction_view: tuple[float, float, float] = (-0.45, 0.65, 0.75)
    regular_visible_width_scale: float = 1.0
    silhouette_visible_width_scale: float = 1.45

    def __post_init__(self) -> None:
        minimum = _non_negative(
            self.minimum_opacity_scale, "minimum_opacity_scale"
        )
        maximum = _positive(
            self.maximum_opacity_scale, "maximum_opacity_scale"
        )
        if maximum < minimum:
            raise FaceDepthCueContractError(
                "maximum_opacity_scale must be at least minimum_opacity_scale"
            )
        facing_weight = _non_negative(
            self.facing_opacity_weight, "facing_opacity_weight"
        )
        depth_weight = _non_negative(
            self.depth_opacity_weight, "depth_opacity_weight"
        )
        if facing_weight + depth_weight <= 0:
            raise FaceDepthCueContractError(
                "at least one opacity weight must be positive"
            )
        ambient = _non_negative(self.ambient_brightness, "ambient_brightness")
        diffuse = _non_negative(self.diffuse_brightness, "diffuse_brightness")
        if ambient + diffuse > 2.0:
            raise FaceDepthCueContractError(
                "ambient_brightness + diffuse_brightness must not exceed 2"
            )
        minimum_saturation = _non_negative(
            self.minimum_saturation_scale, "minimum_saturation_scale"
        )
        maximum_saturation = _positive(
            self.maximum_saturation_scale, "maximum_saturation_scale"
        )
        if maximum_saturation < minimum_saturation:
            raise FaceDepthCueContractError(
                "maximum_saturation_scale must be at least minimum_saturation_scale"
            )
        object.__setattr__(self, "minimum_opacity_scale", minimum)
        object.__setattr__(self, "maximum_opacity_scale", maximum)
        object.__setattr__(self, "facing_opacity_weight", facing_weight)
        object.__setattr__(self, "depth_opacity_weight", depth_weight)
        object.__setattr__(self, "ambient_brightness", ambient)
        object.__setattr__(self, "diffuse_brightness", diffuse)
        object.__setattr__(
            self, "minimum_saturation_scale", minimum_saturation
        )
        object.__setattr__(
            self, "maximum_saturation_scale", maximum_saturation
        )
        object.__setattr__(
            self,
            "far_fog_strength",
            _unit_interval(self.far_fog_strength, "far_fog_strength"),
        )
        object.__setattr__(
            self,
            "fog_color_rgb",
            _rgb3(self.fog_color_rgb, "fog_color_rgb"),
        )
        hue_shift = _non_negative(
            self.maximum_hue_shift_turns, "maximum_hue_shift_turns"
        )
        if hue_shift > 0.25:
            raise FaceDepthCueContractError(
                "maximum_hue_shift_turns must not exceed 0.25"
            )
        object.__setattr__(self, "maximum_hue_shift_turns", hue_shift)
        object.__setattr__(
            self,
            "back_facing_opacity_scale",
            _unit_interval(
                self.back_facing_opacity_scale,
                "back_facing_opacity_scale",
            ),
        )
        object.__setattr__(
            self,
            "light_direction_view",
            _vector3(self.light_direction_view, "light_direction_view"),
        )
        object.__setattr__(
            self,
            "regular_visible_width_scale",
            _positive(
                self.regular_visible_width_scale,
                "regular_visible_width_scale",
            ),
        )
        object.__setattr__(
            self,
            "silhouette_visible_width_scale",
            _positive(
                self.silhouette_visible_width_scale,
                "silhouette_visible_width_scale",
            ),
        )


@dataclass(frozen=True)
class FaceDepthCue:
    face_id: str
    outward_normal: tuple[float, float, float]
    facing_score: float
    normalized_depth: float
    light_score: float
    near_score: float
    brightness: float
    saturation_scale: float
    hue_shift_turns: float
    fog_strength: float
    surface_visibility: float
    opacity_scale: float
    draw_rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "faceId": self.face_id,
            "outwardNormal": list(self.outward_normal),
            "facingScore": self.facing_score,
            "normalizedDepth": self.normalized_depth,
            "lightScore": self.light_score,
            "nearScore": self.near_score,
            "brightness": self.brightness,
            "saturationScale": self.saturation_scale,
            "hueShiftTurns": self.hue_shift_turns,
            "fogStrength": self.fog_strength,
            "surfaceVisibility": self.surface_visibility,
            "opacityScale": self.opacity_scale,
            "drawRank": self.draw_rank,
        }


@dataclass(frozen=True)
class EdgeDepthCue:
    source_edge_id: str
    incident_face_ids: tuple[str, ...]
    is_silhouette: bool
    visible_width_scale: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceEdgeId": self.source_edge_id,
            "incidentFaceIds": list(self.incident_face_ids),
            "isSilhouette": self.is_silhouette,
            "visibleWidthScale": self.visible_width_scale,
        }


@dataclass(frozen=True)
class FaceDepthCueFrame:
    visibility_group_id: str
    projection_matrix: tuple[tuple[float, float, float], ...]
    view_direction: tuple[float, float, float]
    light_direction: tuple[float, float, float]
    hue_axis: tuple[float, float, float]
    fog_color_rgb: tuple[float, float, float]
    face_draw_order: tuple[str, ...]
    faces: tuple[FaceDepthCue, ...]
    edges: tuple[EdgeDepthCue, ...]
    schema: str = FACE_DEPTH_CUE_TRACE_SCHEMA

    @property
    def face_map(self) -> Mapping[str, FaceDepthCue]:
        return {item.face_id: item for item in self.faces}

    @property
    def edge_map(self) -> Mapping[str, EdgeDepthCue]:
        return {item.source_edge_id: item for item in self.edges}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "visibilityGroupId": self.visibility_group_id,
            "projectionMatrix": [list(row) for row in self.projection_matrix],
            "viewDirection": list(self.view_direction),
            "lightDirection": list(self.light_direction),
            "hueAxis": list(self.hue_axis),
            "fogColorRgb": list(self.fog_color_rgb),
            "faceDrawOrder": list(self.face_draw_order),
            "faces": [item.to_dict() for item in self.faces],
            "edges": [item.to_dict() for item in self.edges],
        }


__all__ = [
    "EdgeDepthCue",
    "FACE_DEPTH_CUE_TRACE_SCHEMA",
    "FaceDepthCue",
    "FaceDepthCueContractError",
    "FaceDepthCueFrame",
    "FaceDepthCueStyle",
]
