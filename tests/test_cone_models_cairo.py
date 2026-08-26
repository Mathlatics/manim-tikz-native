"""Actual Cairo evidence for closed bases and open cone mouths."""

from __future__ import annotations

from math import pi
import unittest

import numpy as np
from manim import Scene, config, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import ConeModel, ConeSpec, SectionPlane
from polyhedron_visibility.quadrics.manim import (
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
    section_cap_chord_curve_ids,
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
SECTION_VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)
OPEN_SHELL_REGRESSION_FRONT_POINTS = (
    # Opposite sides of the former false chord; both are truly in front.
    (0.68337608, 0.21844136),
    (0.39171439, 1.17496290),
)


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


def _capture_section(
    model: ConeModel,
    *,
    with_curves: bool,
    component_shading: bool = True,
) -> tuple[np.ndarray, QuadricOcclusion3D, tuple[object, ...]]:
    scene = Scene()
    scene.camera.background_color = "#101820"
    cone = ConeSpec(
        "pixel-section-cone",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (0.0, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )
    plane = SectionPlane(
        "pixel-section-plane",
        (0.0, 0.0, 1.5),
        (0.5, 0.0, 1.0),
        u_axis=(0.0, 1.0, 0.0),
    )
    section_curves = compute_quadric_section_boundary_curves(
        "pixel-finite-section",
        cone,
        plane,
    )
    controller = QuadricOcclusion3D(
        scene,
        surfaces=(cone,),
        curves=section_curves if with_curves else (),
        projection=SECTION_VIEW,
        paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        boundary_visibility_mode="unified",
        include_surface_boundaries=False,
        section_plane=plane,
        max_chord_error=0.015,
        style=QuadricManimStyle(
            surface_fill_color="#315A8A",
            surface_fill_opacity=0.72,
            surface_stroke_opacity=0.0,
            cone_lateral_fill_colors=(
                ("#173753", "#4F84B3", "#1D4368")
                if component_shading
                else None
            ),
            cone_cap_fill_colors=(
                ("#557A99", "#294B6B") if component_shading else None
            ),
            visible_curve_color="#FFD166",
            visible_curve_width=5.0,
            hidden_curve_color="#F59E0B",
            hidden_curve_width=4.0,
            hidden_curve_opacity=0.72,
            section_plane_fill_color="#43D9C0",
            section_plane_fill_opacity=0.34,
            section_plane_stroke_opacity=0.0,
        ),
    ).attach()
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    pixels = scene.camera.pixel_array[:, :, :3].astype(float).copy()
    return pixels, controller, section_curves


def _capture_open_shell_trim_partition() -> tuple[np.ndarray, QuadricOcclusion3D]:
    scene = Scene()
    scene.camera.background_color = "#101820"
    cone = ConeSpec(
        "pixel-open-shell-trim-cone",
        (0.0, 0.0, -2.4),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.OPEN_SINGLE,
    )
    normal = np.asarray((0.82, 0.0, 1.0), dtype=float)
    normal /= np.linalg.norm(normal)
    plane = SectionPlane(
        "pixel-open-shell-trim-plane",
        tuple(np.asarray((0.0, 0.0, -0.35)) + 0.48 * normal),
        (0.82, 0.0, 1.0),
        u_axis=(0.0, 1.0, 0.0),
    )
    controller = QuadricOcclusion3D(
        scene,
        surfaces=(cone,),
        curves=(),
        projection=SECTION_VIEW,
        paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        boundary_visibility_mode="unified",
        include_surface_boundaries=False,
        section_plane=plane,
        max_chord_error=0.015,
        style=QuadricManimStyle(
            surface_fill_color="#315A8A",
            surface_fill_opacity=1.0,
            surface_stroke_opacity=0.0,
            cone_lateral_fill_colors=None,
            cone_cap_fill_colors=None,
            section_plane_fill_color="#43D9C0",
            section_plane_fill_opacity=0.7,
            section_plane_stroke_opacity=0.0,
        ),
    ).attach()
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    pixels = scene.camera.pixel_array[:, :, :3].astype(float).copy()
    return pixels, controller


def _world_to_pixel(point: object) -> tuple[int, int]:
    screen = SECTION_VIEW.matrix[:2] @ np.asarray(point, dtype=float)
    column = int(round((screen[0] / float(config.frame_width) + 0.5) * (WIDTH - 1)))
    row = int(round((0.5 - screen[1] / float(config.frame_height)) * (HEIGHT - 1)))
    return row, column


def _screen_to_pixel(point: tuple[float, float]) -> tuple[int, int]:
    column = int(round((point[0] / float(config.frame_width) + 0.5) * (WIDTH - 1)))
    row = int(round((0.5 - point[1] / float(config.frame_height)) * (HEIGHT - 1)))
    return row, column


def _neighborhood_max(values: np.ndarray, row: int, column: int, radius: int = 3) -> float:
    row_start = max(0, row - radius)
    row_end = min(values.shape[0], row + radius + 1)
    column_start = max(0, column - radius)
    column_end = min(values.shape[1], column + radius + 1)
    return float(np.max(values[row_start:row_end, column_start:column_end]))


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result |= padded[
                row_offset : row_offset + mask.shape[0],
                column_offset : column_offset + mask.shape[1],
            ]
    return result


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

    def test_true_section_ink_and_closed_cap_chord_survive_component_shading(
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
            controllers: list[QuadricOcclusion3D] = []
            try:
                closed_off, closed_off_controller, closed_curves = _capture_section(
                    ConeModel.CLOSED_SINGLE,
                    with_curves=False,
                )
                controllers.append(closed_off_controller)
                closed_on, closed_on_controller, _ = _capture_section(
                    ConeModel.CLOSED_SINGLE,
                    with_curves=True,
                )
                controllers.append(closed_on_controller)
                open_off, open_off_controller, open_curves = _capture_section(
                    ConeModel.OPEN_SINGLE,
                    with_curves=False,
                )
                controllers.append(open_off_controller)
                open_on, open_on_controller, _ = _capture_section(
                    ConeModel.OPEN_SINGLE,
                    with_curves=True,
                )
                controllers.append(open_on_controller)
                legacy_off, legacy_off_controller, _ = _capture_section(
                    ConeModel.CLOSED_SINGLE,
                    with_curves=False,
                    component_shading=False,
                )
                controllers.append(legacy_off_controller)
                legacy_on, legacy_on_controller, _ = _capture_section(
                    ConeModel.CLOSED_SINGLE,
                    with_curves=True,
                    component_shading=False,
                )
                controllers.append(legacy_on_controller)

                closed_difference = np.linalg.norm(closed_on - closed_off, axis=2)
                open_difference = np.linalg.norm(open_on - open_off, axis=2)
                legacy_difference = np.linalg.norm(legacy_on - legacy_off, axis=2)
                self.assertGreater(
                    int(np.count_nonzero(closed_difference > 12.0)),
                    80,
                )
                self.assertGreater(
                    int(np.count_nonzero(open_difference > 12.0)),
                    60,
                )
                self.assertGreater(
                    int(np.count_nonzero(np.linalg.norm(closed_off - legacy_off, axis=2) > 12.0)),
                    100,
                )
                component_mask = closed_difference > 12.0
                legacy_mask = legacy_difference > 12.0
                self.assertGreater(
                    float(np.count_nonzero(component_mask & _dilate(legacy_mask)))
                    / float(np.count_nonzero(component_mask)),
                    0.9,
                )
                self.assertGreater(
                    float(np.count_nonzero(legacy_mask & _dilate(component_mask)))
                    / float(np.count_nonzero(legacy_mask)),
                    0.9,
                )

                chord_id = section_cap_chord_curve_ids(
                    "pixel-finite-section",
                    ConeSpec(
                        "identity-cone",
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0),
                        pi / 4.0,
                        (0.0, 2.0),
                        radial_axis=(1.0, 0.0, 0.0),
                        model=ConeModel.CLOSED_SINGLE,
                    ),
                )[0]
                chord = next(
                    item
                    for item in closed_curves
                    if getattr(item, "curve_id", None) == chord_id
                )
                self.assertFalse(
                    any(
                        getattr(item, "curve_id", "").endswith(":chord")
                        for item in open_curves
                    )
                )
                closed_frame = closed_on_controller.last_boundary_frame
                open_frame = open_on_controller.last_boundary_frame
                assert closed_frame is not None and open_frame is not None
                self.assertIn(chord_id, {item.source_id for item in closed_frame.sources})
                self.assertNotIn(chord_id, {item.source_id for item in open_frame.sources})

                closed_corridor = []
                open_corridor = []
                for fraction in (0.3, 0.5, 0.7):
                    parameter = (
                        chord.domain.start + fraction * chord.domain.length
                    )
                    row, column = _world_to_pixel(chord.point(parameter))
                    closed_corridor.append(
                        _neighborhood_max(closed_difference, row, column)
                    )
                    open_corridor.append(
                        _neighborhood_max(open_difference, row, column)
                    )
                self.assertGreater(min(closed_corridor), 20.0)
                self.assertLess(max(open_corridor), 8.0)
            finally:
                for controller in reversed(controllers):
                    controller.restore()

    def test_open_shell_trim_rim_removes_the_false_chord_fill_band(self) -> None:
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
            pixels, controller = _capture_open_shell_trim_partition()
            try:
                samples = []
                for point in OPEN_SHELL_REGRESSION_FRONT_POINTS:
                    row, column = _screen_to_pixel(point)
                    samples.append(pixels[row, column])

                # Both points are geometrically in front of the open shell.
                # The old false chord put the first point below the front
                # surface and produced the dark band from the reported image.
                self.assertGreater(min(sample[1] for sample in samples), 160.0)
                self.assertLess(
                    float(np.max(np.abs(samples[0] - samples[1]))),
                    8.0,
                )
            finally:
                controller.restore()


if __name__ == "__main__":
    unittest.main()
