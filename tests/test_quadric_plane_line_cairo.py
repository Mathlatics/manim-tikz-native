"""Actual Cairo pixel evidence for a finite plane's AREA/LINE handoff."""

from __future__ import annotations

import unittest

import numpy as np
from manim import Scene, config, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import (
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlanePatchProjectionKind,
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


AREA_VIEW = ParallelView.from_matrix(
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
)
LINE_VIEW = ParallelView.from_matrix(np.identity(3))
PLANE = SectionPlane(
    "pixel-line-plane",
    (0.3, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    u_axis=(0.0, 0.0, 1.0),
)
PATCH = PlaneDisplayPatchSpec(
    "pixel-line-patch",
    PLANE.plane_id,
    1.2,
    1.6,
)
SURFACE = SphereSpec("pixel-line-sphere", (0.0, 0.0, 0.0), 1.0)
BACKGROUND = np.asarray((7, 17, 27), dtype=np.uint8)


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=1,
        max_curves=1,
        max_fragments_per_curve=8,
        max_segments_per_fragment=128,
        max_surface_segments=256,
        max_dashes_per_fragment=32,
        max_projected_length=12.0,
        max_total_mobjects=4000,
        max_boundary_sources=16,
    )


def _style() -> QuadricManimStyle:
    return QuadricManimStyle(
        surface_fill_opacity=0.0,
        surface_stroke_opacity=0.0,
        section_plane_fill_color="#35D0BA",
        section_plane_fill_opacity=0.25,
        section_plane_stroke_color="#F9D65C",
        section_plane_stroke_width=4.0,
        section_plane_stroke_opacity=1.0,
    )


def _controller(
    scene: Scene,
    projection: object,
) -> QuadricOcclusion3D:
    return QuadricOcclusion3D(
        scene,
        surfaces=(SURFACE,),
        curves=(),
        projection=projection,
        section_plane=PLANE,
        section_patch=PATCH,
        boundary_visibility_mode="unified",
        include_surface_boundaries=False,
        style=_style(),
        limits=_limits(),
    )


def _capture(scene: Scene) -> np.ndarray:
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    return scene.camera.pixel_array[:, :, :3].copy()


def _screen_to_pixel(point: tuple[float, float]) -> tuple[int, int]:
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
class QuadricFinitePlaneLineCairoTests(unittest.TestCase):
    def test_warm_edge_on_frame_matches_cold_finite_line_pixels(self) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 480,
                "pixel_height": 270,
                "frame_rate": 6,
            }
        ):
            state: dict[str, object] = {"view": AREA_VIEW}
            warm_scene = Scene()
            warm_scene.camera.background_color = "#07111B"
            warm = _controller(
                warm_scene,
                lambda _scene: state["view"],
            ).attach()
            cold_scene = Scene()
            cold_scene.camera.background_color = "#07111B"
            cold: QuadricOcclusion3D | None = None
            try:
                state["view"] = LINE_VIEW
                warm.update()
                warm_pixels = _capture(warm_scene)

                cold = _controller(cold_scene, LINE_VIEW).attach()
                cold_pixels = _capture(cold_scene)

                np.testing.assert_array_equal(warm_pixels, cold_pixels)
                frame = warm.last_section_frame
                assert frame is not None
                self.assertIs(
                    frame.projection_kind,
                    PlanePatchProjectionKind.LINE,
                )
                self.assertEqual(frame.plane_fragments, ())

                difference = np.linalg.norm(
                    warm_pixels.astype(float) - BACKGROUND.astype(float),
                    axis=2,
                )
                changed = difference > 8.0
                yellow = (
                    (warm_pixels[:, :, 0] > 180)
                    & (warm_pixels[:, :, 1] > 145)
                    & (warm_pixels[:, :, 2] < 150)
                )
                self.assertGreater(int(np.count_nonzero(changed)), 140)
                self.assertGreater(int(np.count_nonzero(yellow)), 70)

                line_row, line_column = _screen_to_pixel((0.3, 0.0))
                self.assertGreater(
                    float(
                        np.linalg.norm(
                            warm_pixels[line_row, line_column].astype(float)
                            - BACKGROUND.astype(float)
                        )
                    ),
                    80.0,
                )
                for point in ((0.8, 0.0), (0.8, 0.6), (-0.3, -0.5), (0.3, 2.0)):
                    row, column = _screen_to_pixel(point)
                    np.testing.assert_array_equal(
                        warm_pixels[row, column],
                        BACKGROUND,
                    )
            finally:
                warm.restore()
                if cold is not None:
                    cold.restore()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
