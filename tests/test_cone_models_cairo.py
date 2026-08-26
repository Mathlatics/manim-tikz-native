"""Actual Cairo evidence for closed bases and open cone mouths."""

from __future__ import annotations

from math import pi
import unittest

import numpy as np
from manim import Scene, config, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import ConeModel, ConeSpec
from polyhedron_visibility.quadrics.manim import (
    QuadricManimStyle,
    QuadricOcclusion3D,
)


try:
    import cairo as _cairo  # noqa: F401
    from manim.renderer.cairo_renderer import (  # noqa: F401
        CairoRenderer as _CairoRenderer,
    )
except (ImportError, OSError):
    CAIRO_AVAILABLE = False
else:
    CAIRO_AVAILABLE = True


WIDTH = 320
HEIGHT = 180
AXIAL_VIEW = ParallelView.from_matrix(np.eye(3))


def _capture(model: ConeModel) -> tuple[np.ndarray, QuadricOcclusion3D]:
    scene = Scene()
    scene.camera.background_color = "#000000"
    cone = ConeSpec(
        "pixel-cone",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (0.0, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )
    controller = QuadricOcclusion3D(
        scene,
        surfaces=(cone,),
        curves=(),
        projection=AXIAL_VIEW,
        paint_policy="physical",
        boundary_visibility_mode="unified",
        include_surface_boundaries=False,
        max_chord_error=0.01,
        style=QuadricManimStyle(
            surface_fill_color="#0044CC",
            surface_fill_opacity=0.64,
            surface_stroke_opacity=0.0,
            cone_lateral_fill_colors=("#0044CC",),
            cone_cap_fill_colors=("#FF2200",),
        ),
    ).attach()
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    pixels = scene.camera.pixel_array[:, :, :3].astype(float).copy()
    return pixels, controller


@unittest.skipUnless(CAIRO_AVAILABLE, "Manim Cairo renderer is unavailable")
class ConeModelCairoTests(unittest.TestCase):
    def test_closed_base_and_open_mouth_have_distinct_center_pixels(self) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": WIDTH,
                "pixel_height": HEIGHT,
                "frame_rate": 8,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        ):
            closed_pixels, closed = _capture(ConeModel.CLOSED_SINGLE)
            open_pixels, opened = _capture(ConeModel.OPEN_SINGLE)
            row = int(round(0.5 * (HEIGHT - 1)))
            column = int(round(0.5 * (WIDTH - 1)))
            closed_rgb = closed_pixels[row, column]
            open_rgb = open_pixels[row, column]

            # A closed base adds the red cap sheet. The open mouth keeps only
            # the blue far-side lateral sheet at the same projected point.
            self.assertGreater(closed_rgb[0], open_rgb[0] + 60.0)
            self.assertGreater(open_rgb[2], closed_rgb[2] + 20.0)
            self.assertLess(open_rgb[0], 8.0)

            closed_slot = closed._surface_paint_slots[0]
            open_slot = opened._surface_paint_slots[0]
            self.assertGreater(len(closed_slot.front_cap.points), 0)
            self.assertEqual(len(open_slot.front_cap.points), 0)
            self.assertGreater(len(open_slot.back_lateral.points), 0)

            closed.restore()
            opened.restore()


if __name__ == "__main__":
    unittest.main()
