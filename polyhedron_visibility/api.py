from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from manim import Mobject

from .binding import DisplayPointProvider, ManimOcclusionBinding, PositionProvider
from .contract import TolerancePolicy, VisibilityModel
from .style import OcclusionStyle


ProjectionSource = (
    Sequence[Sequence[float]]
    | Callable[[object], Sequence[Sequence[float]]]
)


@dataclass(frozen=True)
class ParallelProjection:
    """An explicit parallel projection, static or driven by Scene state."""

    source: ProjectionSource

    def current_matrix(self, scene: object) -> tuple[tuple[float, float, float], ...]:
        raw = self.source(scene) if callable(self.source) else self.source
        matrix = np.asarray(raw, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("parallel projection must be a finite 3x3 matrix")
        return tuple(tuple(float(item) for item in row) for row in matrix)

    @classmethod
    def identity(cls) -> "ParallelProjection":
        return cls(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


class AutoOcclusion3D(ManimOcclusionBinding):
    """Public Manim API for fixed-topology convex-face hidden-line removal."""

    def __init__(
        self,
        scene: object,
        model: VisibilityModel,
        *,
        position_provider: PositionProvider,
        stroke_bindings: Mapping[str, Mobject],
        projection: ParallelProjection,
        display_point_provider: DisplayPointProvider | None = None,
        style: OcclusionStyle,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> None:
        self.projection = projection
        super().__init__(
            scene,
            model,
            position_provider=position_provider,
            stroke_bindings=stroke_bindings,
            projection_provider=projection.current_matrix,
            display_point_provider=display_point_provider,
            style=style,
            tolerance_policy=tolerance_policy,
        )


__all__ = ["AutoOcclusion3D", "ParallelProjection"]
