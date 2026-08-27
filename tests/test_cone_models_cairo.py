"""Actual Cairo evidence for closed bases and open cone mouths."""

from __future__ import annotations

import json
from math import pi, tau
from pathlib import Path
import unittest

import numpy as np
from manim import Scene, config, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import ConeModel, ConeSpec, SectionPlane
from polyhedron_visibility.quadrics.manim import (
    QuadricBoundaryStyle,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
    section_cap_chord_curve_ids,
)
from polyhedron_visibility.quadrics.section_compositing import PlaneDepthRole


try:
    import cairo as _cairo  # noqa: F401
    from manim.renderer.cairo_renderer import (  # noqa: F401
        CairoRenderer as _CairoRenderer,
    )
except (ImportError, OSError):
    CAIRO_AVAILABLE = False
else:
    CAIRO_AVAILABLE = True


CAIRO_BASELINE_PATH = (
    Path(__file__).parent / "baselines" / "quadric-section-v1-cairo.json"
)
with CAIRO_BASELINE_PATH.open(encoding="utf-8") as _baseline_file:
    CAIRO_BASELINE = json.load(_baseline_file)

_STANDARD_PROFILE = CAIRO_BASELINE["profiles"]["standard"]
_COMPLETE_PROFILE = CAIRO_BASELINE["profiles"]["complete_release_frame"]
_CLOSED_OPEN_BASELINE = CAIRO_BASELINE["fixtures"][
    "closed_vs_open_finite_cone_semantics"
]["invariants"]
_SECTION_INK_BASELINE = CAIRO_BASELINE["fixtures"][
    "section_ink_component_shading"
]["invariants"]
_OPEN_SHELL_BASELINE = CAIRO_BASELINE["fixtures"][
    "open_shell_oblique_offset_0_48"
]
_COMPLETE_FRAME_BASELINE = CAIRO_BASELINE["fixtures"][
    "complete_plane_comparison_offset_0_48"
]["invariants"]

WIDTH = int(_STANDARD_PROFILE["pixel_width"])
HEIGHT = int(_STANDARD_PROFILE["pixel_height"])
FULL_WIDTH = int(_COMPLETE_PROFILE["pixel_width"])
FULL_HEIGHT = int(_COMPLETE_PROFILE["pixel_height"])
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
    *(
        tuple(float(value) for value in point)
        for point in _OPEN_SHELL_BASELINE["screen_probe_points"]
    ),
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


def _capture_complete_plane_comparison() -> tuple[
    np.ndarray,
    tuple[QuadricOcclusion3D, QuadricOcclusion3D],
    tuple[tuple[object, ...], tuple[object, ...]],
]:
    """Capture the demo's exact offset=0.48 geometry with every feature on."""

    scene = Scene()
    scene.camera.background_color = "#101820"
    normal = np.asarray((0.82, 0.0, 1.0), dtype=float)
    normal /= np.linalg.norm(normal)
    style = QuadricManimStyle(
        surface_fill_color="#315A8A",
        surface_fill_opacity=0.76,
        surface_stroke_opacity=0.0,
        cone_lateral_fill_colors=("#173753", "#4F84B3", "#1D4368"),
        cone_cap_fill_colors=("#557A99", "#294B6B"),
        cone_lateral_sheen_direction=(1.0, 0.0, 0.0),
        cone_cap_sheen_direction=(-1.0, 1.0, 0.0),
        visible_curve_color="#FFD166",
        visible_curve_width=4.0,
        hidden_curve_color="#F59E0B",
        hidden_curve_width=3.0,
        hidden_curve_opacity=0.66,
        section_plane_fill_color="#43D9C0",
        section_plane_fill_opacity=0.34,
        section_plane_stroke_color="#B39DDB",
        section_plane_stroke_width=1.8,
        section_plane_stroke_opacity=0.9,
        dash_length=0.12,
        dash_gap=0.09,
    )
    boundary_style = QuadricBoundaryStyle(
        visible_color="#5CE1E6",
        visible_width=4.4,
        visible_opacity=1.0,
        hidden_color="#5CE1E6",
        hidden_width=3.0,
        hidden_opacity=0.24,
        dash_length=0.12,
        dash_gap=0.09,
    )
    limits = QuadricManimLimits(
        max_surfaces=1,
        max_curves=4,
        max_fragments_per_curve=16,
        max_segments_per_fragment=384,
        max_surface_segments=768,
        max_dashes_per_fragment=80,
        max_projected_length=18.0,
        max_total_mobjects=30000,
        max_boundary_sources=24,
    )
    controllers = []
    curves_by_model = []
    vertical_shift = -0.55 * np.asarray(SECTION_VIEW.matrix[1], dtype=float)
    for index, (model, horizontal) in enumerate(
        (
            (ConeModel.CLOSED_SINGLE, -3.35),
            (ConeModel.OPEN_SINGLE, 3.35),
        )
    ):
        shift = horizontal * np.asarray(SECTION_VIEW.matrix[0], dtype=float)
        shift += vertical_shift
        cone = ConeSpec(
            f"complete-plane-cone-{index}",
            tuple(shift + np.asarray((0.0, 0.0, -2.4))),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=model,
        )
        plane = SectionPlane(
            f"complete-plane-cut-{index}",
            tuple(
                shift
                + np.asarray((0.0, 0.0, -0.35))
                + 0.48 * normal
            ),
            (0.82, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        section_id = f"complete-plane-section-{index}"
        section_curves = compute_quadric_section_boundary_curves(
            section_id,
            cone,
            plane,
        )
        curves_by_model.append(tuple(section_curves))
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(cone,),
            curves=section_curves,
            projection=SECTION_VIEW,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            style=style,
            boundary_styles={
                "style:surface-silhouette": boundary_style,
                "style:surface-boundary": boundary_style,
            },
            limits=limits,
            max_chord_error=0.008,
            section_plane=plane,
            boundary_visibility_mode="unified",
            include_surface_boundaries=True,
            painter_z_band=(20.0 + 20.0 * index, 30.0 + 20.0 * index),
        ).attach()
        controllers.append(controller)
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    pixels = scene.camera.pixel_array[:, :, :3].astype(float).copy()
    return (
        pixels,
        (controllers[0], controllers[1]),
        (curves_by_model[0], curves_by_model[1]),
    )


def _world_to_pixel(point: object) -> tuple[int, int]:
    screen = SECTION_VIEW.matrix[:2] @ np.asarray(point, dtype=float)
    width = int(config.pixel_width)
    height = int(config.pixel_height)
    column = int(round((screen[0] / float(config.frame_width) + 0.5) * (width - 1)))
    row = int(round((0.5 - screen[1] / float(config.frame_height)) * (height - 1)))
    return row, column


def _screen_to_pixel(point: tuple[float, float]) -> tuple[int, int]:
    width = int(config.pixel_width)
    height = int(config.pixel_height)
    column = int(round((point[0] / float(config.frame_width) + 0.5) * (width - 1)))
    row = int(round((0.5 - point[1] / float(config.frame_height)) * (height - 1)))
    return row, column


def _roles_at_screen(
    controller: QuadricOcclusion3D,
    point: tuple[float, float],
) -> set[PlaneDepthRole]:
    frame = controller.last_section_frame
    assert frame is not None
    target = np.asarray(point, dtype=float)
    roles = set()
    for fragment in frame.plane_fragments:
        triangle = np.asarray(fragment.screen_vertices, dtype=float)
        cross = tuple(
            float(
                (triangle[(index + 1) % 3, 0] - triangle[index, 0])
                * (target[1] - triangle[index, 1])
                - (triangle[(index + 1) % 3, 1] - triangle[index, 1])
                * (target[0] - triangle[index, 0])
            )
            for index in range(3)
        )
        if min(cross) >= -1.0e-9 or max(cross) <= 1.0e-9:
            roles.add(fragment.role)
    return roles


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
                "frame_rate": int(_STANDARD_PROFILE["frame_rate"]),
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
            self.assertGreater(
                closed_rgb[0],
                open_rgb[0] + _CLOSED_OPEN_BASELINE["closed_red_over_open_min"],
            )
            self.assertGreater(
                open_rgb[2],
                closed_rgb[2]
                + _CLOSED_OPEN_BASELINE["open_blue_over_closed_min"],
            )
            self.assertLess(
                open_rgb[0],
                _CLOSED_OPEN_BASELINE["open_red_max"],
            )

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
                "frame_rate": int(_STANDARD_PROFILE["frame_rate"]),
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
                difference_threshold = _SECTION_INK_BASELINE[
                    "rgb_difference_threshold"
                ]
                self.assertGreater(
                    int(
                        np.count_nonzero(
                            closed_difference > difference_threshold
                        )
                    ),
                    _SECTION_INK_BASELINE["closed_changed_pixels_min"],
                )
                self.assertGreater(
                    int(
                        np.count_nonzero(
                            open_difference > difference_threshold
                        )
                    ),
                    _SECTION_INK_BASELINE["open_changed_pixels_min"],
                )
                self.assertGreater(
                    int(
                        np.count_nonzero(
                            np.linalg.norm(closed_off - legacy_off, axis=2)
                            > difference_threshold
                        )
                    ),
                    _SECTION_INK_BASELINE[
                        "component_shading_changed_pixels_min"
                    ],
                )
                component_mask = closed_difference > difference_threshold
                legacy_mask = legacy_difference > difference_threshold
                self.assertGreater(
                    float(np.count_nonzero(component_mask & _dilate(legacy_mask)))
                    / float(np.count_nonzero(component_mask)),
                    _SECTION_INK_BASELINE["component_to_legacy_overlap_min"],
                )
                self.assertGreater(
                    float(np.count_nonzero(legacy_mask & _dilate(component_mask)))
                    / float(np.count_nonzero(legacy_mask)),
                    _SECTION_INK_BASELINE["component_to_legacy_overlap_min"],
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
                self.assertGreater(
                    min(closed_corridor),
                    _SECTION_INK_BASELINE[
                        "closed_chord_corridor_difference_min"
                    ],
                )
                self.assertLess(
                    max(open_corridor),
                    _SECTION_INK_BASELINE[
                        "open_chord_corridor_difference_max"
                    ],
                )
            finally:
                for controller in reversed(controllers):
                    controller.restore()

    def test_open_shell_trim_rim_removes_the_false_chord_fill_band(self) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": WIDTH,
                "pixel_height": HEIGHT,
                "frame_rate": int(_STANDARD_PROFILE["frame_rate"]),
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
                invariants = _OPEN_SHELL_BASELINE["invariants"]
                self.assertGreater(
                    min(sample[1] for sample in samples),
                    invariants["green_channel_min"],
                )
                self.assertLess(
                    float(np.max(np.abs(samples[0] - samples[1]))),
                    invariants["probe_rgb_delta_max"],
                )
            finally:
                controller.restore()

    def test_complete_offset_point_48_demo_frame_has_no_open_shell_corridor_leak(
        self,
    ) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": FULL_WIDTH,
                "pixel_height": FULL_HEIGHT,
                "frame_rate": int(_COMPLETE_PROFILE["frame_rate"]),
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        ):
            pixels, controllers, curves_by_model = (
                _capture_complete_plane_comparison()
            )
            try:
                closed_controller, open_controller = controllers
                closed_curves, open_curves = curves_by_model

                # The exact production combination is active: analytic yellow
                # section ink, cyan semantic boundaries, component-aware cone
                # shading, and the translucent unified section plane.
                yellow = _COMPLETE_FRAME_BASELINE["yellow_mask"]
                cyan = _COMPLETE_FRAME_BASELINE["cyan_mask"]
                yellow_mask = (
                    (pixels[:, :, 0] > yellow["red_min"])
                    & (pixels[:, :, 1] > yellow["green_min"])
                    & (pixels[:, :, 2] < yellow["blue_max"])
                )
                cyan_mask = (
                    (pixels[:, :, 0] < cyan["red_max"])
                    & (pixels[:, :, 1] > cyan["green_min"])
                    & (pixels[:, :, 2] > cyan["blue_min"])
                )
                self.assertGreater(
                    int(np.count_nonzero(yellow_mask)),
                    yellow["pixel_count_min"],
                )
                self.assertGreater(
                    int(np.count_nonzero(cyan_mask)),
                    cyan["pixel_count_min"],
                )
                self.assertTrue(open_controller.style.cone_component_shading)
                self.assertTrue(closed_controller.style.cone_component_shading)

                self.assertTrue(
                    any(
                        getattr(curve, "curve_id", "").endswith(":chord")
                        for curve in closed_curves
                    )
                )
                self.assertFalse(
                    any(
                        getattr(curve, "curve_id", "").endswith(":chord")
                        for curve in open_curves
                    )
                )

                boundary_frame = open_controller.last_boundary_frame
                self.assertIsNotNone(boundary_frame)
                trim_sources = tuple(
                    source
                    for source in boundary_frame.sources
                    if source.source_kind is BoundarySourceKind.SURFACE_TRIM_RIM
                )
                self.assertEqual(
                    len(trim_sources),
                    _COMPLETE_FRAME_BASELINE["open_trim_rim_count"],
                )
                self.assertEqual(
                    sum(
                        source.source_kind is BoundarySourceKind.SECTION_CAP_CHORD
                        for source in boundary_frame.sources
                    ),
                    _COMPLETE_FRAME_BASELINE["open_section_cap_chord_count"],
                )
                trim_fragments = tuple(
                    sorted(
                        (
                            fragment
                            for fragment in boundary_frame.fragments
                            if fragment.painted
                            and fragment.source_id == trim_sources[0].source_id
                        ),
                        key=lambda fragment: fragment.interval.start,
                    )
                )
                self.assertTrue(trim_fragments)
                self.assertAlmostEqual(trim_fragments[0].interval.start, 0.0)
                self.assertAlmostEqual(trim_fragments[-1].interval.end, tau)
                for previous, current in zip(
                    trim_fragments,
                    trim_fragments[1:],
                ):
                    self.assertAlmostEqual(
                        previous.interval.end,
                        current.interval.start,
                    )

                # These eleven samples cross the exact corridor circled in the
                # report.  Every sample remains one certified front-plane role
                # and every rendered pixel stays filled; neither a false cap
                # chord nor a role-boundary background seam may cross it.
                shift = np.asarray((3.35, -0.55), dtype=float)
                first = (
                    np.asarray(OPEN_SHELL_REGRESSION_FRONT_POINTS[0]) + shift
                )
                second = (
                    np.asarray(OPEN_SHELL_REGRESSION_FRONT_POINTS[1]) + shift
                )
                for fraction in np.linspace(
                    0.0,
                    1.0,
                    int(_COMPLETE_FRAME_BASELINE["corridor_sample_count"]),
                ):
                    point = (1.0 - fraction) * first + fraction * second
                    self.assertEqual(
                        _roles_at_screen(open_controller, tuple(point)),
                        {PlaneDepthRole.IN_FRONT_OF_SURFACE},
                    )
                    row, column = _screen_to_pixel(tuple(point))
                    pixel = pixels[row, column]
                    self.assertGreater(
                        pixel[1],
                        _COMPLETE_FRAME_BASELINE["corridor_green_min"],
                    )
                    self.assertGreater(
                        float(
                            np.linalg.norm(
                                pixel
                                - np.asarray(
                                    _COMPLETE_FRAME_BASELINE[
                                        "corridor_background_rgb"
                                    ]
                                )
                            )
                        ),
                        _COMPLETE_FRAME_BASELINE[
                            "corridor_background_distance_min"
                        ],
                    )
            finally:
                for controller in reversed(controllers):
                    controller.restore()


if __name__ == "__main__":
    unittest.main()
