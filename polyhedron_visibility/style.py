from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np


class OcclusionStyleError(ValueError):
    """Raised when a render style cannot provide stable fixed-capacity slots."""


def _positive(value: float, label: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise OcclusionStyleError(f"{label} must be finite and positive")
    return result


def _non_negative(value: float, label: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0:
        raise OcclusionStyleError(f"{label} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ResolvedOcclusionStyle:
    visible_color: Any
    hidden_color: Any
    visible_width: float
    hidden_width: float
    visible_opacity: float
    hidden_opacity: float
    background_color: Any
    background_width: float
    background_opacity: float
    cap_style: Any | None
    joint_type: Any | None


@dataclass(frozen=True)
class OcclusionStyle:
    """Static Cairo styling plus an explicit allocation bound.

    ``max_projected_length`` is deliberately required and is measured in the
    chosen overlay coordinate space: world-space when no display provider is
    supplied, or final camera-frame Scene coordinates for a custom provider.
    The binding allocates every dashed Line before animation starts, so it must
    know a safe upper bound instead of growing a VGroup inside an updater.
    """

    max_projected_length: float
    dash_length: float = 0.08
    dash_gap: float = 0.06
    visible_color: Any | None = None
    hidden_color: Any | None = None
    visible_width_scale: float = 1.0
    hidden_width_scale: float = 0.82
    visible_opacity_scale: float = 1.0
    hidden_opacity_scale: float = 0.78

    def __post_init__(self) -> None:
        _positive(self.max_projected_length, "max_projected_length")
        _positive(self.dash_length, "dash_length")
        _non_negative(self.dash_gap, "dash_gap")
        _positive(self.visible_width_scale, "visible_width_scale")
        _positive(self.hidden_width_scale, "hidden_width_scale")
        _non_negative(self.visible_opacity_scale, "visible_opacity_scale")
        _non_negative(self.hidden_opacity_scale, "hidden_opacity_scale")

    @property
    def dash_period(self) -> float:
        return self.dash_length + self.dash_gap

    def resolve_for(self, source: object) -> ResolvedOcclusionStyle:
        family = getattr(source, "get_family", lambda: [source])()
        for member in family:
            for attribute in ("stroke_rgbas", "background_stroke_rgbas"):
                if not hasattr(member, attribute):
                    continue
                colors = np.asarray(getattr(member, attribute), dtype=float)
                if colors.ndim != 2 or colors.shape[1:] != (4,) or len(colors) < 1:
                    raise OcclusionStyleError(
                        f"source stroke {attribute} has an unsupported color layout"
                    )
                if len(colors) > 1 and not np.allclose(
                    colors, colors[0], rtol=0.0, atol=1.0e-12
                ):
                    raise OcclusionStyleError(
                        f"source stroke gradient in {attribute} is unsupported"
                    )
        vector = next(
            (
                item
                for item in family
                if callable(getattr(item, "get_stroke_width", None))
                and callable(getattr(item, "get_stroke_opacity", None))
            ),
            None,
        )
        if vector is None:
            raise OcclusionStyleError("source stroke has no Manim vector stroke style")
        width = float(vector.get_stroke_width())
        opacity = float(vector.get_stroke_opacity())
        background_width = float(vector.get_stroke_width(background=True))
        background_opacity = float(vector.get_stroke_opacity(background=True))
        if not all(
            isfinite(item) and item >= 0
            for item in (width, opacity, background_width, background_opacity)
        ):
            raise OcclusionStyleError("source stroke style must be finite and non-negative")
        source_color = vector.get_stroke_color()
        return ResolvedOcclusionStyle(
            visible_color=source_color if self.visible_color is None else self.visible_color,
            hidden_color=source_color if self.hidden_color is None else self.hidden_color,
            visible_width=width * self.visible_width_scale,
            hidden_width=width * self.hidden_width_scale,
            visible_opacity=min(1.0, opacity * self.visible_opacity_scale),
            hidden_opacity=min(1.0, opacity * self.hidden_opacity_scale),
            background_color=vector.get_stroke_color(background=True),
            background_width=background_width,
            background_opacity=min(1.0, background_opacity),
            cap_style=getattr(vector, "cap_style", None),
            joint_type=getattr(vector, "joint_type", None),
        )


__all__ = [
    "OcclusionStyle",
    "OcclusionStyleError",
    "ResolvedOcclusionStyle",
]
