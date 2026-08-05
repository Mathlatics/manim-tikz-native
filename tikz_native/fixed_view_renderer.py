from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from manim import RIGHT, UP

from .compiler import PictureSpec
from .manim_renderer import NativeFigure, NativeManimRenderer
from .projection_3d import project_point


class NativeFixedViewRenderer(NativeManimRenderer):
    """Instantiate 2D and authored-view 3D pictures as ordinary 2D Mobjects.

    A 3D ``PictureSpec`` retains its original world coordinates and TikZ view
    matrix in the manifest.  This renderer only projects the runtime Mobjects
    into the fixed authored view, which lets the PPT editor use the same normal
    ``Scene`` and timeline path for 2D diagrams and simple polyhedra.

    It deliberately does not implement camera motion.  Consumers that need to
    orbit a figure must use ``NativeManim3DRenderer`` instead.
    """

    def render(self, picture: PictureSpec) -> NativeFigure:
        if picture.dimension not in {2, 3}:
            raise ValueError(f"unsupported TikZ picture dimension: {picture.dimension}")
        if picture.dimension == 3 and picture.projection_3d is None:
            raise ValueError("fixed-view 3D picture has no TikZ projection")
        if picture.dimension == 3:
            unsupported_canvas_nodes = [
                item.id
                for item in picture.objects
                if item.style.transform_shape
                and item.style.native_canvas_plane is not None
            ]
            if unsupported_canvas_nodes:
                raise ValueError(
                    "fixed-view 3D renderer does not yet support transform-shape "
                    "canvas nodes: " + ", ".join(unsupported_canvas_nodes)
                )
        return super().render(picture)

    def point(
        self,
        value: Sequence[float],
        picture: PictureSpec,
    ) -> np.ndarray:
        values = tuple(float(component) for component in value)
        if len(values) == 2:
            return super().point(values, picture)
        if len(values) != 3:
            raise ValueError(
                f"fixed-view renderer received {len(values)}D point: {values}"
            )
        if picture.projection_3d is None:
            raise ValueError("3D point has no TikZ projection matrix")
        screen_x, screen_y, _depth = project_point(
            picture.projection_3d.matrix,
            values,
        )
        return self.unit * picture.scale * (
            screen_x * RIGHT + screen_y * UP
        )


__all__ = ["NativeFixedViewRenderer"]
