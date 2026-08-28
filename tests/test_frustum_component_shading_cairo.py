from __future__ import annotations

import json
from math import pi
from pathlib import Path
import unittest

import numpy as np
from manim import Scene, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.manim import (
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
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


BASELINE_PATH = (
    Path(__file__).parent
    / "baselines"
    / "frustum-component-shading-cairo.json"
)
with BASELINE_PATH.open(encoding="utf-8") as _baseline_file:
    BASELINE = json.load(_baseline_file)

PROFILE = BASELINE["profile"]
INVARIANTS = BASELINE["invariants"]
VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)


def _capture(
    *,
    component_shading: bool,
) -> tuple[np.ndarray, QuadricOcclusion3D, tuple[object, ...]]:
    scene = Scene()
    scene.camera.background_color = "#101820"
    frustum = ConeSpec(
        "pixel-component-frustum",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (0.75, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.CLOSED_SINGLE,
    )
    plane = SectionPlane(
        "pixel-component-plane",
        (0.2, 0.0, 1.3),
        (1.0, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
    )
    curves = compute_quadric_section_boundary_curves(
        "pixel-component-section",
        frustum,
        plane,
    )
    controller = QuadricOcclusion3D(
        scene,
        surfaces=(frustum,),
        curves=curves,
        projection=VIEW,
        paint_policy="depth_aware_diagrammatic",
        section_plane=plane,
        boundary_visibility_mode="unified",
        include_surface_boundaries=True,
        max_chord_error=0.015,
        style=QuadricManimStyle(
            surface_fill_color="#315A8A",
            surface_fill_opacity=0.76,
            surface_stroke_opacity=0.0,
            cone_lateral_fill_colors=(
                ("#173753", "#4F84B3", "#1D4368")
                if component_shading
                else None
            ),
            cone_cap_fill_colors=(
                ("#8A6A3D", "#D4A85F") if component_shading else None
            ),
            visible_curve_color="#FFD166",
            visible_curve_width=5.0,
            hidden_curve_color="#F59E0B",
            hidden_curve_width=4.0,
            hidden_curve_opacity=0.7,
            section_plane_fill_color="#43D9C0",
            section_plane_fill_opacity=0.28,
            section_plane_stroke_opacity=0.4,
        ),
    ).attach()
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    pixels = scene.camera.pixel_array[:, :, :3].astype(float).copy()
    return pixels, controller, curves


@unittest.skipUnless(CAIRO_AVAILABLE, "Cairo renderer is unavailable")
class FrustumComponentShadingCairoTests(unittest.TestCase):
    def test_two_caps_two_chords_and_lateral_sheets_render_together(self) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": int(PROFILE["pixel_width"]),
                "pixel_height": int(PROFILE["pixel_height"]),
                "frame_rate": int(PROFILE["frame_rate"]),
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        ):
            uniform_pixels, uniform, _uniform_curves = _capture(
                component_shading=False
            )
            component_pixels, component, component_curves = _capture(
                component_shading=True
            )
            try:
                background = np.asarray((16.0, 24.0, 32.0), dtype=float)
                changed = (
                    np.linalg.norm(component_pixels - background, axis=2) > 12.0
                )
                component_difference = (
                    np.linalg.norm(component_pixels - uniform_pixels, axis=2) > 8.0
                )
                yellow = (
                    (component_pixels[:, :, 0] > 180.0)
                    & (component_pixels[:, :, 1] > 120.0)
                    & (component_pixels[:, :, 2] < 170.0)
                )
                self.assertGreater(
                    int(np.count_nonzero(changed)),
                    int(INVARIANTS["changed_pixel_count_min"]),
                )
                self.assertGreater(
                    int(np.count_nonzero(component_difference)),
                    int(INVARIANTS["component_difference_count_min"]),
                )
                self.assertGreater(
                    int(np.count_nonzero(yellow)),
                    int(INVARIANTS["yellow_section_pixel_count_min"]),
                )

                chords = tuple(
                    item
                    for item in component_curves
                    if isinstance(item, SegmentCurve)
                )
                self.assertEqual(
                    len(chords),
                    int(INVARIANTS["terminal_cap_count"]),
                )
                frame = component.last_boundary_frame
                assert frame is not None
                self.assertEqual(
                    sum(
                        item.source_kind is BoundarySourceKind.SURFACE_CAP_RIM
                        for item in frame.sources
                    ),
                    int(INVARIANTS["surface_cap_rim_source_count"]),
                )
                self.assertEqual(
                    sum(
                        item.source_kind is BoundarySourceKind.SECTION_CAP_CHORD
                        for item in frame.sources
                    ),
                    int(INVARIANTS["section_cap_chord_source_count"]),
                )
                back = component._section_surface_paint_slots[1]
                front = component._section_surface_paint_slots[4]
                self.assertGreater(len(back.back_lateral.points), 0)
                self.assertGreater(len(front.front_lateral.points), 0)
                self.assertGreater(len(back.back_cap.points), 0)
                self.assertGreater(len(front.front_cap.points), 0)
            finally:
                component.restore()
                uniform.restore()


if __name__ == "__main__":
    unittest.main()
