"""Actual Cairo pixel evidence for unified semantic boundary strokes."""

from __future__ import annotations

from math import pi
from typing import Sequence
import unittest

import numpy as np
from manim import Scene, config, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundaryRenderIntent,
    BoundarySemanticKind,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import ConeSpec, SectionPlane
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.visibility import VisibilityKind


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
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)
WIDTH = 480
HEIGHT = 270


def _hex_rgb(value: str) -> np.ndarray:
    text = value.lstrip("#")
    return np.asarray(
        tuple(int(text[index : index + 2], 16) for index in (0, 2, 4)),
        dtype=float,
    )


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=2,
        max_curves=8,
        max_fragments_per_curve=32,
        max_segments_per_fragment=384,
        max_surface_segments=768,
        max_dashes_per_fragment=100,
        max_projected_length=18.0,
        max_total_mobjects=60000,
        max_boundary_sources=48,
    )


def _style() -> QuadricManimStyle:
    return QuadricManimStyle(
        surface_fill_color="#5275A8",
        surface_fill_opacity=0.72,
        surface_stroke_color="#17324F",
        surface_stroke_width=2.0,
        surface_stroke_opacity=1.0,
        visible_curve_color="#F6C344",
        visible_curve_width=4.0,
        hidden_curve_color="#F6C344",
        hidden_curve_width=3.0,
        hidden_curve_opacity=0.94,
        dash_length=0.12,
        dash_gap=0.09,
        section_plane_fill_color="#63C7B2",
        section_plane_fill_opacity=0.14,
        section_plane_stroke_color="#7E57C2",
        section_plane_stroke_width=2.2,
        section_plane_stroke_opacity=1.0,
    )


def _capture_pixels(scene: Scene) -> np.ndarray:
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    return scene.camera.pixel_array[:, :, :3].astype(float).copy()


def _screen_to_pixel(point: Sequence[float]) -> tuple[int, int]:
    x, y = float(point[0]), float(point[1])
    column = int(
        round((x / float(config.frame_width) + 0.5) * (WIDTH - 1))
    )
    row = int(
        round((0.5 - y / float(config.frame_height)) * (HEIGHT - 1))
    )
    return (
        max(0, min(HEIGHT - 1, row)),
        max(0, min(WIDTH - 1, column)),
    )


def _nearest_ink_pixel(
    pixels: np.ndarray,
    point: Sequence[float],
    target: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    row, column = _screen_to_pixel(point)
    radius = 4
    row_start = max(0, row - radius)
    row_end = min(HEIGHT, row + radius + 1)
    column_start = max(0, column - radius)
    column_end = min(WIDTH, column + radius + 1)
    patch = pixels[row_start:row_end, column_start:column_end]
    distances = np.linalg.norm(patch - target, axis=2)
    local = np.unravel_index(int(np.argmin(distances)), distances.shape)
    selected_row = row_start + local[0]
    selected_column = column_start + local[1]
    return (
        selected_row,
        selected_column,
        pixels[selected_row, selected_column].copy(),
    )


def _scene_for(policy: QuadricPaintPolicy):
    cone = ConeSpec(
        "cairo-boundary-cone",
        (0.0, 0.0, -2.4),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    plane = SectionPlane(
        "cairo-boundary-plane",
        (0.0, 0.0, -0.35),
        (0.82, 0.0, 1.0),
        u_axis=(0.0, 1.0, 0.0),
    )
    scene = Scene()
    scene.camera.background_color = "#F7FAFC"
    controller = QuadricOcclusion3D(
        scene,
        surfaces=(cone,),
        curves=(),
        projection=VIEW,
        paint_policy=policy,
        style=_style(),
        limits=_limits(),
        max_chord_error=0.008,
        section_plane=plane,
        boundary_visibility_mode="unified",
        include_surface_boundaries=True,
    ).attach()
    return scene, controller


def _first_hidden_plane_dash(controller: QuadricOcclusion3D):
    frame = controller.last_boundary_frame
    assert frame is not None
    candidates = [
        item
        for item in frame.fragments
        if item.source_id.startswith(
            "boundary:plane:cairo-boundary-plane:edge:"
        )
        and item.visibility_kind is VisibilityKind.HIDDEN
        and item.painted
        and item.render_intent is BoundaryRenderIntent.DASHED
    ]
    if not candidates:
        raise AssertionError(
            "scene produced no painted hidden plane-outline fragment"
        )
    fragment = sorted(candidates, key=lambda item: item.item_id)[0]
    slot_index = controller._fragment_slot_maps[fragment.source_id][
        fragment.item_id
    ]
    slot = controller._curve_slots[fragment.source_id].fragments[slot_index]
    dashes = [dash for dash in slot.dashes if len(dash.points)]
    if not dashes:
        raise AssertionError(
            "hidden plane-outline fragment has no active dash"
        )
    dash = dashes[len(dashes) // 2]
    point = 0.5 * (
        np.asarray(dash.get_start()) + np.asarray(dash.get_end())
    )
    return fragment, point


@unittest.skipUnless(CAIRO_AVAILABLE, "Manim Cairo renderer is unavailable")
class UnifiedBoundaryCairoTests(unittest.TestCase):
    def test_depth_aware_hidden_outline_is_attenuated_by_front_sheet(
        self,
    ) -> None:
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
            diagram_scene, diagram = _scene_for(
                QuadricPaintPolicy.DIAGRAMMATIC
            )
            depth_scene, depth = _scene_for(
                QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC
            )
            try:
                diagram_fragment, point = _first_hidden_plane_dash(diagram)
                depth_fragment, depth_point = _first_hidden_plane_dash(depth)
                self.assertEqual(
                    diagram_fragment.source_id, depth_fragment.source_id
                )
                np.testing.assert_allclose(point, depth_point, atol=1.0e-8)
                diagram_pixels = _capture_pixels(diagram_scene)
                depth_pixels = _capture_pixels(depth_scene)
                target = _hex_rgb("#7E57C2")
                surface = _hex_rgb("#5275A8")
                row, column, diagram_rgb = _nearest_ink_pixel(
                    diagram_pixels, point, target
                )
                depth_rgb = depth_pixels[row, column]
                self.assertLess(
                    float(np.linalg.norm(diagram_rgb - target)) + 3.0,
                    float(np.linalg.norm(depth_rgb - target)),
                )
                self.assertLess(
                    float(np.linalg.norm(depth_rgb - surface)),
                    float(np.linalg.norm(diagram_rgb - surface)),
                )
            finally:
                depth.restore()
                diagram.restore()

    def test_physical_omits_hidden_boundaries_and_silhouette_stays_solid(
        self,
    ) -> None:
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
            scene, controller = _scene_for(QuadricPaintPolicy.PHYSICAL)
            try:
                frame = controller.last_boundary_frame
                self.assertIsNotNone(frame)
                assert frame is not None
                hidden = [
                    item
                    for item in frame.fragments
                    if item.visibility_kind is VisibilityKind.HIDDEN
                ]
                self.assertTrue(hidden)
                self.assertTrue(all(not item.painted for item in hidden))
                source_map = {
                    item.source_id: item for item in frame.sources
                }
                silhouettes = [
                    item
                    for item in frame.fragments
                    if source_map[item.source_id].semantic_kind
                    is BoundarySemanticKind.TRUE_SILHOUETTE
                ]
                self.assertTrue(silhouettes)
                self.assertTrue(
                    all(
                        item.render_intent is BoundaryRenderIntent.SOLID
                        for item in silhouettes
                        if item.painted
                    )
                )
                pixels = _capture_pixels(scene)
                background = _hex_rgb("#F7FAFC")
                self.assertGreater(
                    int(
                        np.count_nonzero(
                            np.linalg.norm(
                                pixels - background, axis=2
                            )
                            > 12.0
                        )
                    ),
                    500,
                )
            finally:
                controller.restore()

    def test_boundary_slots_and_pixels_remain_stable_after_update(self) -> None:
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
            scene, controller = _scene_for(
                QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC
            )
            try:
                identities = controller.slot_identities()
                first = _capture_pixels(scene)
                controller.update()
                second = _capture_pixels(scene)
                self.assertEqual(controller.slot_identities(), identities)
                np.testing.assert_array_equal(first, second)
            finally:
                controller.restore()


if __name__ == "__main__":
    unittest.main()
