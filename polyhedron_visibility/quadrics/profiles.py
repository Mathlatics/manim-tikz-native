"""Opt-in preview and final-render recipes for finite quadric scenes.

The profiles collect display resolution and the matching Manim approximation
limits in one immutable value.  They do not change global Manim configuration
or a controller by themselves; authors explicitly consume ``manim_config()``
and ``controller_kwargs()`` at the scene boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from .manim import QuadricManimLimits, QuadricManimStyle


QUADRIC_RENDER_PROFILE_SCHEMA = "manim-quadric-render-profile/v1"


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


@dataclass(frozen=True, slots=True)
class QuadricRenderProfile:
    """One explicit authoring/render-quality recipe.

    ``component_shading=False`` removes cone component colour bands from a
    supplied style.  It never changes the mathematical surface, section
    curves, semantic boundaries, or painter policy.
    """

    profile_id: str
    pixel_width: int
    pixel_height: int
    frame_rate: float
    max_chord_error: float
    section_max_screen_error: float
    limits: QuadricManimLimits
    component_shading: bool
    include_surface_boundaries: bool = True
    schema: str = QUADRIC_RENDER_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUADRIC_RENDER_PROFILE_SCHEMA:
            raise ValueError("invalid quadric render-profile schema")
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        for name in ("pixel_width", "pixel_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self,
            "frame_rate",
            _positive_number(self.frame_rate, "frame_rate"),
        )
        object.__setattr__(
            self,
            "max_chord_error",
            _positive_number(self.max_chord_error, "max_chord_error"),
        )
        object.__setattr__(
            self,
            "section_max_screen_error",
            _positive_number(
                self.section_max_screen_error,
                "section_max_screen_error",
            ),
        )
        if not isinstance(self.limits, QuadricManimLimits):
            raise TypeError("limits must be a QuadricManimLimits")
        for name in ("component_shading", "include_surface_boundaries"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    def manim_config(self) -> dict[str, object]:
        """Return the values intended for Manim's ``tempconfig``."""

        return {
            "renderer": "cairo",
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
            "frame_rate": self.frame_rate,
        }

    def apply_style(self, style: QuadricManimStyle) -> QuadricManimStyle:
        """Apply only the profile's component-shading recommendation."""

        if not isinstance(style, QuadricManimStyle):
            raise TypeError("style must be a QuadricManimStyle")
        if self.component_shading:
            return style
        return replace(
            style,
            cone_lateral_fill_colors=None,
            cone_cap_fill_colors=None,
        )

    def controller_kwargs(
        self,
        *,
        style: QuadricManimStyle = QuadricManimStyle(),
        limits: QuadricManimLimits | None = None,
    ) -> dict[str, object]:
        """Return explicit keyword arguments for a quadric controller."""

        selected_limits = self.limits if limits is None else limits
        if not isinstance(selected_limits, QuadricManimLimits):
            raise TypeError("limits must be a QuadricManimLimits")
        return {
            "style": self.apply_style(style),
            "limits": selected_limits,
            "max_chord_error": self.max_chord_error,
            "section_max_screen_error": self.section_max_screen_error,
            "include_surface_boundaries": self.include_surface_boundaries,
        }

    def to_dict(self) -> dict[str, object]:
        limits = self.limits
        return {
            "schema": self.schema,
            "profileId": self.profile_id,
            "manim": {
                "renderer": "cairo",
                "pixelWidth": self.pixel_width,
                "pixelHeight": self.pixel_height,
                "frameRate": self.frame_rate,
            },
            "geometry": {
                "maxChordError": self.max_chord_error,
                "sectionMaxScreenError": self.section_max_screen_error,
            },
            "componentShading": self.component_shading,
            "includeSurfaceBoundaries": self.include_surface_boundaries,
            "limits": {
                name: getattr(limits, name)
                for name in QuadricManimLimits.__dataclass_fields__
            },
        }


QUADRIC_PREVIEW_PROFILE = QuadricRenderProfile(
    profile_id="preview",
    pixel_width=480,
    pixel_height=270,
    frame_rate=15.0,
    max_chord_error=0.025,
    section_max_screen_error=0.14,
    limits=QuadricManimLimits(
        max_surfaces=16,
        max_curves=32,
        max_fragments_per_curve=16,
        max_segments_per_fragment=160,
        max_surface_segments=256,
        max_dashes_per_fragment=48,
        max_projected_length=16.0,
        max_total_mobjects=10000,
        max_boundary_sources=48,
        max_boundary_styles=64,
    ),
    component_shading=False,
)


QUADRIC_FINAL_PROFILE = QuadricRenderProfile(
    profile_id="final",
    pixel_width=960,
    pixel_height=540,
    frame_rate=30.0,
    max_chord_error=0.001,
    section_max_screen_error=0.08,
    limits=QuadricManimLimits(),
    component_shading=True,
)


QUADRIC_RENDER_PROFILES: Mapping[str, QuadricRenderProfile] = MappingProxyType(
    {
        QUADRIC_PREVIEW_PROFILE.profile_id: QUADRIC_PREVIEW_PROFILE,
        QUADRIC_FINAL_PROFILE.profile_id: QUADRIC_FINAL_PROFILE,
    }
)


__all__ = [
    "QUADRIC_FINAL_PROFILE",
    "QUADRIC_PREVIEW_PROFILE",
    "QUADRIC_RENDER_PROFILE_SCHEMA",
    "QUADRIC_RENDER_PROFILES",
    "QuadricRenderProfile",
]
