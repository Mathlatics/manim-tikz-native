"""Actual Cairo evidence for the open-double section coordinator."""

from __future__ import annotations

from math import pi
import unittest

import numpy as np
from manim import Scene, config, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricBoundaryStyle,
    QuadricManimLimits,
    QuadricManimStyle,
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


VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)


def _pixel_for_screen(point: tuple[float, float]) -> tuple[int, int]:
    x, y = point
    column = int(
        round(
            (x + config.frame_width / 2.0)
            / config.frame_width
            * (config.pixel_width - 1)
        )
    )
    row = int(
        round(
            (config.frame_height / 2.0 - y)
            / config.frame_height
            * (config.pixel_height - 1)
        )
    )
    return row, column


@unittest.skipUnless(CAIRO_AVAILABLE, "Cairo is required for pixel evidence")
class CompositeQuadricSectionCairoTests(unittest.TestCase):
    def test_two_nappes_share_one_plane_alpha_and_retain_both_section_branches(
        self,
    ) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 480,
                "pixel_height": 270,
                "frame_rate": 6,
            }
        ):
            scene = Scene()
            background = np.asarray((11.0, 23.0, 35.0))
            scene.camera.background_color = "#0B1723"
            cone = ConeSpec(
                "pixel-double",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                pi / 4.0,
                (-2.15, 2.15),
                radial_axis=(1.0, 0.0, 0.0),
                model=ConeModel.OPEN_DOUBLE,
            )
            plane = SectionPlane(
                "pixel-double-cut",
                (0.0, 0.48, 0.0),
                (0.0, 1.0, 0.16),
                u_axis=(1.0, 0.0, 0.0),
            )
            limits = QuadricManimLimits(
                max_surfaces=2,
                max_curves=8,
                max_fragments_per_curve=24,
                max_segments_per_fragment=256,
                max_surface_segments=384,
                max_dashes_per_fragment=64,
                max_projected_length=24.0,
                max_total_mobjects=30000,
                max_boundary_sources=32,
            )
            controller = CompositeQuadricSection3D(
                scene,
                surface=cone,
                section_id="pixel-double-section",
                plane=plane,
                projection=VIEW,
                paint_policy="depth_aware_diagrammatic",
                style=QuadricManimStyle(
                    surface_fill_color="#2B6F9F",
                    surface_fill_opacity=0.62,
                    surface_stroke_opacity=0.0,
                    section_plane_fill_color="#2CB9A4",
                    section_plane_fill_opacity=0.18,
                    section_plane_stroke_opacity=0.0,
                    cone_lateral_fill_colors=(
                        "#173753",
                        "#4F9AC1",
                        "#1D4368",
                    ),
                ),
                boundary_styles={
                    "style:curve": QuadricBoundaryStyle(
                        visible_color="#FFD866",
                        visible_width=4.0,
                        visible_opacity=1.0,
                        hidden_color="#FFD866",
                        hidden_width=3.0,
                        hidden_opacity=0.46,
                    ),
                    "style:surface-boundary": QuadricBoundaryStyle(
                        visible_color="#61DDF2",
                        visible_width=3.2,
                        visible_opacity=0.95,
                        hidden_color="#61DDF2",
                        hidden_width=2.4,
                        hidden_opacity=0.34,
                    ),
                    "style:surface-silhouette": QuadricBoundaryStyle(
                        visible_color="#61DDF2",
                        visible_width=3.2,
                        visible_opacity=0.95,
                        hidden_color="#61DDF2",
                        hidden_width=2.4,
                        hidden_opacity=0.34,
                    ),
                },
                limits=limits,
                max_chord_error=0.03,
                section_max_screen_error=0.16,
                plane_patch_margin=0.17,
            ).attach()
            try:
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                pixels = scene.camera.pixel_array[:, :, :3].astype(float).copy()
                yellow = (
                    (pixels[:, :, 0] > 180.0)
                    & (pixels[:, :, 1] > 130.0)
                    & (pixels[:, :, 2] < 140.0)
                )
                cyan = (
                    (pixels[:, :, 0] < 150.0)
                    & (pixels[:, :, 1] > 130.0)
                    & (pixels[:, :, 2] > 150.0)
                )
                midpoint = pixels.shape[0] // 2
                self.assertGreater(int(np.count_nonzero(yellow[:midpoint])), 35)
                self.assertGreater(int(np.count_nonzero(yellow[midpoint:])), 35)
                self.assertGreater(int(np.count_nonzero(cyan)), 90)

                # This point is inside the fitted plane patch but outside both
                # cone projections. Its RGB must equal one alpha blend of the
                # authored plane color over the background. Drawing one local
                # plane per nappe would make it substantially brighter.
                plane_row, plane_column = _pixel_for_screen((2.25, 0.0))
                actual_plane = pixels[plane_row, plane_column]
                plane_color = np.asarray((44.0, 185.0, 164.0))
                expected_once = 0.82 * background + 0.18 * plane_color
                expected_twice = 0.82 * expected_once + 0.18 * plane_color
                self.assertLess(
                    float(np.linalg.norm(actual_plane - expected_once)),
                    7.0,
                )
                self.assertGreater(
                    float(np.linalg.norm(actual_plane - expected_twice)),
                    16.0,
                )

                apex_row, apex_column = _pixel_for_screen((0.0, 0.0))
                self.assertGreater(
                    float(
                        np.linalg.norm(
                            pixels[apex_row, apex_column] - background
                        )
                    ),
                    20.0,
                )
                frame = controller.last_composite_frame
                assert frame is not None
                self.assertEqual(frame.shared_apex.projected_overlap_area, 0.0)
                self.assertEqual(len(controller.branch_lineage), 2)
            finally:
                controller.restore()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
