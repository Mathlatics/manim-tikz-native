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
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.manim import (
    QuadricBoundaryStyle,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.surface_boundaries import GeneratorBoundarySpec
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


def _scene_for(
    policy: QuadricPaintPolicy,
    *,
    silhouette_style: QuadricBoundaryStyle | None = None,
):
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
        boundary_styles=(
            None
            if silhouette_style is None
            else {"style:surface-silhouette": silhouette_style}
        ),
    ).attach()
    return scene, controller


def _visible_curve_between_surface_and_plane_scene(
    policy: QuadricPaintPolicy = QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
):
    view = ParallelView.from_matrix(np.eye(3))
    sphere = SphereSpec("bracket-sphere", (0.0, 0.0, 0.0), 1.0)
    curve = SegmentCurve(
        "bracket-curve",
        (-0.7, 0.0, 1.2),
        (0.7, 0.0, 1.2),
    )
    plane = SectionPlane(
        "bracket-plane",
        (0.0, 0.0, 1.5),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    patch = PlaneDisplayPatchSpec(
        "bracket-patch",
        plane.plane_id,
        1.5,
        1.2,
    )
    style = QuadricManimStyle(
        surface_fill_color="#0000FF",
        surface_fill_opacity=1.0,
        surface_stroke_opacity=0.0,
        visible_curve_color="#FF0000",
        visible_curve_width=10.0,
        visible_curve_opacity=1.0,
        hidden_curve_color="#FF0000",
        hidden_curve_width=10.0,
        hidden_curve_opacity=1.0,
        dash_length=0.18,
        dash_gap=0.12,
        section_plane_fill_color="#00FF00",
        section_plane_fill_opacity=0.25,
        section_plane_stroke_opacity=0.0,
    )
    scene = Scene()
    scene.camera.background_color = "#000000"
    controller = QuadricOcclusion3D(
        scene,
        surfaces=(sphere,),
        curves=(curve,),
        projection=view,
        paint_policy=policy,
        style=style,
        limits=_limits(),
        max_chord_error=0.008,
        section_plane=plane,
        section_patch=patch,
        boundary_visibility_mode="unified",
        include_surface_boundaries=False,
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
        and item.effective_visibility_kind is VisibilityKind.HIDDEN
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


def _first_plane_occluded_silhouette_dash(
    controller: QuadricOcclusion3D,
):
    frame = controller.last_boundary_frame
    assert frame is not None
    source_map = {item.source_id: item for item in frame.sources}
    candidates = [
        item
        for item in frame.fragments
        if source_map[item.source_id].semantic_kind
        is BoundarySemanticKind.TRUE_SILHOUETTE
        and item.plane_occluded
        and item.painted
        and item.render_intent is BoundaryRenderIntent.DASHED
    ]
    if not candidates:
        raise AssertionError(
            "scene produced no painted plane-occluded silhouette fragment"
        )
    fragment = sorted(candidates, key=lambda item: item.item_id)[0]
    slot_index = controller._fragment_slot_maps[fragment.source_id][
        fragment.item_id
    ]
    slot = controller._curve_slots[fragment.source_id].fragments[slot_index]
    dashes = [dash for dash in slot.dashes if len(dash.points)]
    if not dashes:
        raise AssertionError("plane-occluded silhouette has no active dash")
    dash = dashes[len(dashes) // 2]
    point = 0.5 * (
        np.asarray(dash.get_start()) + np.asarray(dash.get_end())
    )
    return fragment, point


@unittest.skipUnless(CAIRO_AVAILABLE, "Manim Cairo renderer is unavailable")
class UnifiedBoundaryCairoTests(unittest.TestCase):
    def test_generator_style_ids_produce_distinct_cairo_pixels(self) -> None:
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
            cylinder = CylinderSpec(
                "style-cylinder",
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
                1.0,
                (0.0, 2.0),
                radial_axis=(1.0, 0.0, 0.0),
            )
            side_view = ParallelView.from_matrix(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, -1.0),
                    (0.0, 1.0, 0.0),
                )
            )
            scene = Scene()
            scene.camera.background_color = "#000000"
            controller = QuadricOcclusion3D(
                scene,
                surfaces=(cylinder,),
                curves=(),
                projection=side_view,
                paint_policy="physical",
                style=QuadricManimStyle(
                    surface_fill_opacity=0.0,
                    surface_stroke_opacity=0.0,
                ),
                boundary_styles={
                    "style:red": QuadricBoundaryStyle(
                        visible_color="#FF2020",
                        visible_width=10.0,
                    ),
                    "style:blue": QuadricBoundaryStyle(
                        visible_color="#2080FF",
                        visible_width=10.0,
                    ),
                },
                limits=_limits(),
                max_chord_error=0.008,
                boundary_visibility_mode="unified",
                include_surface_boundaries=False,
                generator_boundaries=(
                    GeneratorBoundarySpec(
                        "red-generator",
                        cylinder.surface_id,
                        pi / 4.0,
                        style_id="style:red",
                    ),
                    GeneratorBoundarySpec(
                        "blue-generator",
                        cylinder.surface_id,
                        3.0 * pi / 4.0,
                        style_id="style:blue",
                    ),
                ),
            ).attach()
            try:
                pixels = _capture_pixels(scene)
                _row, _column, red = _nearest_ink_pixel(
                    pixels,
                    (2.0 ** -0.5, 0.0),
                    _hex_rgb("#FF2020"),
                )
                _row, _column, blue = _nearest_ink_pixel(
                    pixels,
                    (-2.0 ** -0.5, 0.0),
                    _hex_rgb("#2080FF"),
                )
                self.assertGreater(red[0], red[2] + 120.0)
                self.assertGreater(blue[2], blue[0] + 120.0)
            finally:
                controller.restore()

    def test_cone_and_cylinder_cap_rims_render_front_solid_and_rear_dash(
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
            surfaces = (
                CylinderSpec(
                    "cap-cylinder",
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, 1.0),
                    1.0,
                    (0.0, 2.0),
                    radial_axis=(1.0, 0.0, 0.0),
                ),
                ConeSpec(
                    "cap-cone",
                    (0.0, 0.0, -1.4),
                    (0.0, 0.0, 1.0),
                    pi / 6.0,
                    (0.0, 3.0),
                    radial_axis=(1.0, 0.0, 0.0),
                ),
            )
            side_view = ParallelView.from_matrix(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, -1.0),
                    (0.0, 1.0, 0.0),
                )
            )
            for surface in surfaces:
                with self.subTest(surface=surface.surface_id):
                    scene = Scene()
                    scene.camera.background_color = "#07111F"
                    controller = QuadricOcclusion3D(
                        scene,
                        surfaces=(surface,),
                        curves=(),
                        projection=side_view,
                        paint_policy="diagrammatic",
                        style=QuadricManimStyle(
                            surface_fill_color="#355070",
                            surface_fill_opacity=0.22,
                            surface_stroke_color="#FFB000",
                            surface_stroke_width=7.0,
                            surface_stroke_opacity=1.0,
                            hidden_curve_opacity=1.0,
                            dash_length=0.14,
                            dash_gap=0.10,
                        ),
                        limits=_limits(),
                        max_chord_error=0.008,
                        boundary_visibility_mode="unified",
                    ).attach()
                    try:
                        frame = controller.last_boundary_frame
                        assert frame is not None
                        cap_ids = {
                            source.source_id
                            for source in frame.sources
                            if source.source_kind.value == "surface_cap_rim"
                        }
                        self.assertTrue(cap_ids)
                        pixels = _capture_pixels(scene)
                        target = _hex_rgb("#FFB000")
                        for source_id in cap_ids:
                            fragments = [
                                item
                                for item in frame.fragments
                                if item.source_id == source_id and item.painted
                            ]
                            solid = next(
                                item
                                for item in fragments
                                if item.render_intent
                                is BoundaryRenderIntent.SOLID
                            )
                            dashed = next(
                                item
                                for item in fragments
                                if item.render_intent
                                is BoundaryRenderIntent.DASHED
                            )
                            solid_slot = controller._curve_slots[
                                source_id
                            ].fragments[
                                controller._fragment_slot_maps[source_id][
                                    solid.item_id
                                ]
                            ]
                            dash_slot = controller._curve_slots[
                                source_id
                            ].fragments[
                                controller._fragment_slot_maps[source_id][
                                    dashed.item_id
                                ]
                            ]
                            solid_point = solid_slot.solid.points[
                                len(solid_slot.solid.points) // 2
                            ]
                            active_dashes = [
                                item for item in dash_slot.dashes if len(item.points)
                            ]
                            self.assertTrue(active_dashes)
                            dash = active_dashes[len(active_dashes) // 2]
                            dash_point = dash.points[len(dash.points) // 2]
                            for point in (solid_point, dash_point):
                                _row, _column, rgb = _nearest_ink_pixel(
                                    pixels,
                                    point,
                                    target,
                                )
                                self.assertLess(
                                    float(np.linalg.norm(rgb - target)),
                                    75.0,
                                )
                    finally:
                        controller.restore()

    def test_custom_cap_rim_style_draws_plane_occluded_dashes_when_mesh_hidden(
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
            normal = np.asarray((0.82, 0.0, 1.0), dtype=float)
            normal /= np.linalg.norm(normal)
            plane_point = (
                np.asarray((0.0, 0.0, -0.35), dtype=float)
                + 0.48 * normal
            )
            cone = ConeSpec(
                "styled-cap-cone",
                (0.0, 0.0, -2.4),
                (0.0, 0.0, 1.0),
                pi / 6.0,
                (0.0, 4.0),
                radial_axis=(1.0, 0.0, 0.0),
            )
            plane = SectionPlane(
                "styled-cap-plane",
                tuple(float(value) for value in plane_point),
                (0.82, 0.0, 1.0),
                u_axis=(0.0, 1.0, 0.0),
            )
            rim_style = QuadricBoundaryStyle(
                visible_color="#5CE1E6",
                visible_width=7.0,
                visible_opacity=1.0,
                hidden_color="#FF8BD1",
                hidden_width=7.0,
                hidden_opacity=1.0,
                dash_length=0.14,
                dash_gap=0.10,
            )
            scene = Scene()
            scene.camera.background_color = "#101820"
            controller = QuadricOcclusion3D(
                scene,
                surfaces=(cone,),
                curves=(),
                projection=VIEW,
                paint_policy=QuadricPaintPolicy.DIAGRAMMATIC,
                style=QuadricManimStyle(
                    surface_fill_color="#315A8A",
                    surface_fill_opacity=0.78,
                    surface_stroke_opacity=0.0,
                    section_plane_fill_color="#43D9C0",
                    section_plane_fill_opacity=0.34,
                    section_plane_stroke_opacity=0.0,
                    dash_length=0.14,
                    dash_gap=0.10,
                ),
                boundary_styles={"style:surface-boundary": rim_style},
                limits=_limits(),
                max_chord_error=0.008,
                section_plane=plane,
                boundary_visibility_mode="unified",
                include_surface_boundaries=True,
            ).attach()
            try:
                frame = controller.last_boundary_frame
                assert frame is not None
                rim_source = next(
                    item
                    for item in frame.sources
                    if item.source_kind.value == "surface_cap_rim"
                )
                fragments = [
                    item
                    for item in frame.fragments
                    if item.source_id == rim_source.source_id
                ]
                plane_hidden = [
                    item
                    for item in fragments
                    if item.plane_occluded
                    and item.render_intent is BoundaryRenderIntent.DASHED
                    and item.painted
                ]
                visible = [
                    item
                    for item in fragments
                    if not item.plane_occluded
                    and item.effective_visibility_kind
                    is VisibilityKind.VISIBLE
                    and item.render_intent is BoundaryRenderIntent.SOLID
                    and item.painted
                ]
                self.assertTrue(plane_hidden and visible)
                self.assertTrue(
                    all(
                        item.surface_visibility_kind is VisibilityKind.VISIBLE
                        and item.effective_visibility_kind
                        is VisibilityKind.HIDDEN
                        for item in plane_hidden
                    )
                )
                self.assertEqual(
                    controller.boundary_styles["style:surface-boundary"],
                    rim_style,
                )

                hidden = max(
                    plane_hidden,
                    key=lambda item: item.interval.end - item.interval.start,
                )
                solid = max(
                    visible,
                    key=lambda item: item.interval.end - item.interval.start,
                )
                hidden_slot = controller._curve_slots[
                    hidden.source_id
                ].fragments[
                    controller._fragment_slot_maps[hidden.source_id][
                        hidden.item_id
                    ]
                ]
                solid_slot = controller._curve_slots[
                    solid.source_id
                ].fragments[
                    controller._fragment_slot_maps[solid.source_id][
                        solid.item_id
                    ]
                ]
                active_dashes = [
                    item for item in hidden_slot.dashes if len(item.points)
                ]
                self.assertTrue(active_dashes)
                dash = active_dashes[len(active_dashes) // 2]
                dash_point = dash.points[len(dash.points) // 2]
                solid_point = solid_slot.solid.points[
                    len(solid_slot.solid.points) // 2
                ]

                pixels = _capture_pixels(scene)
                for point, target in (
                    (solid_point, _hex_rgb("#5CE1E6")),
                    (dash_point, _hex_rgb("#FF8BD1")),
                ):
                    _row, _column, rgb = _nearest_ink_pixel(
                        pixels,
                        point,
                        target,
                    )
                    self.assertLess(
                        float(np.linalg.norm(rgb - target)),
                        75.0,
                    )
            finally:
                controller.restore()

    def test_second_surface_physically_occludes_true_silhouette_pixels(
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
            view = ParallelView.from_matrix(np.eye(3))
            scene = Scene()
            scene.camera.background_color = "#07111F"
            controller = QuadricOcclusion3D(
                scene,
                surfaces=(
                    SphereSpec("far", (0.0, 0.0, -2.0), 1.0),
                    SphereSpec("near", (0.55, 0.0, 2.0), 0.75),
                ),
                curves=(),
                projection=view,
                paint_policy="physical",
                style=QuadricManimStyle(
                    surface_fill_color="#204060",
                    surface_fill_opacity=1.0,
                    surface_stroke_color="#FFD166",
                    surface_stroke_width=8.0,
                    surface_stroke_opacity=1.0,
                ),
                limits=_limits(),
                max_chord_error=0.008,
                boundary_visibility_mode="unified",
            ).attach()
            try:
                frame = controller.last_boundary_frame
                assert frame is not None
                hidden = next(
                    item
                    for item in frame.fragments
                    if item.source_id == "boundary:far:silhouette"
                    and item.effective_visibility_kind is VisibilityKind.HIDDEN
                )
                self.assertFalse(hidden.painted)
                source = next(
                    item
                    for item in frame.sources
                    if item.source_id == hidden.source_id
                )
                world = source.curve.point(hidden.interval.midpoint)
                screen = view.matrix[:2] @ np.asarray(world, dtype=float)
                pixels = _capture_pixels(scene)
                row, column = _screen_to_pixel(screen)
                rgb = pixels[row, column]
                fill = _hex_rgb("#204060")
                stroke = _hex_rgb("#FFD166")
                self.assertLess(float(np.linalg.norm(rgb - fill)), 35.0)
                self.assertGreater(float(np.linalg.norm(rgb - stroke)), 100.0)
            finally:
                controller.restore()

    def test_crossing_boundaries_use_far_to_near_cairo_order(self) -> None:
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
            view = ParallelView.from_matrix(np.eye(3))
            far = CylinderSpec(
                "far-cross",
                (-1.0, 0.0, -2.0),
                (1.0, 0.0, 0.0),
                0.18,
                (0.0, 2.0),
                radial_axis=(0.0, 1.0, 0.0),
            )
            near = CylinderSpec(
                "near-cross",
                (0.0, -1.0, 2.0),
                (0.0, 1.0, 0.0),
                0.18,
                (0.0, 2.0),
                radial_axis=(1.0, 0.0, 0.0),
            )
            scene = Scene()
            scene.camera.background_color = "#000000"
            controller = QuadricOcclusion3D(
                scene,
                surfaces=(far, near),
                curves=(),
                projection=view,
                paint_policy="diagrammatic",
                style=QuadricManimStyle(
                    surface_fill_color="#404040",
                    surface_fill_opacity=0.2,
                    surface_stroke_opacity=0.0,
                ),
                boundary_styles={
                    "style:far-red": QuadricBoundaryStyle(
                        visible_color="#FF2020",
                        visible_width=12.0,
                        hidden_color="#FF2020",
                        hidden_width=12.0,
                        dash_length=10.0,
                        dash_gap=0.0,
                    ),
                    "style:near-blue": QuadricBoundaryStyle(
                        visible_color="#2080FF",
                        visible_width=12.0,
                        hidden_color="#2080FF",
                        hidden_width=12.0,
                        dash_length=10.0,
                        dash_gap=0.0,
                    ),
                },
                limits=_limits(),
                max_chord_error=0.005,
                boundary_visibility_mode="unified",
                include_surface_boundaries=False,
                generator_boundaries=(
                    GeneratorBoundarySpec(
                        "far-line",
                        far.surface_id,
                        pi / 2.0,
                        style_id="style:far-red",
                    ),
                    GeneratorBoundarySpec(
                        "near-line",
                        near.surface_id,
                        3.0 * pi / 2.0,
                        style_id="style:near-blue",
                    ),
                ),
            ).attach()
            try:
                frame = controller.last_boundary_frame
                assert frame is not None
                far_items = [
                    item.item_id
                    for item in frame.fragments
                    if item.source_id == "far-line" and item.painted
                ]
                near_items = [
                    item.item_id
                    for item in frame.fragments
                    if item.source_id == "near-line" and item.painted
                ]
                self.assertTrue(far_items and near_items)
                self.assertLess(
                    max(frame.draw_order.index(item) for item in far_items),
                    min(frame.draw_order.index(item) for item in near_items),
                )
                pixels = _capture_pixels(scene)
                _row, _column, crossing = _nearest_ink_pixel(
                    pixels,
                    (0.0, 0.0),
                    _hex_rgb("#2080FF"),
                )
                self.assertGreater(crossing[2], crossing[0] + 120.0)
            finally:
                controller.restore()

    def test_plane_occluded_visible_curve_is_dashed_between_surface_and_plane(
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
            diagram_scene, diagram = _visible_curve_between_surface_and_plane_scene(
                QuadricPaintPolicy.DIAGRAMMATIC
            )
            depth_scene, depth = _visible_curve_between_surface_and_plane_scene(
                QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC
            )
            try:
                diagram_frame = diagram.last_boundary_frame
                depth_frame = depth.last_boundary_frame
                self.assertIsNotNone(diagram_frame)
                self.assertIsNotNone(depth_frame)
                assert diagram_frame is not None and depth_frame is not None
                diagram_fragment = next(
                    item
                    for item in diagram_frame.fragments
                    if item.source_id == "bracket-curve"
                )
                fragment = next(
                    item
                    for item in depth_frame.fragments
                    if item.source_id == "bracket-curve"
                )
                self.assertIs(
                    fragment.surface_visibility_kind,
                    VisibilityKind.VISIBLE,
                )
                self.assertIs(
                    fragment.effective_visibility_kind,
                    VisibilityKind.HIDDEN,
                )
                self.assertTrue(fragment.plane_occluded)
                self.assertEqual(fragment.occluder_surface_ids, ())
                self.assertIs(
                    fragment.render_intent,
                    BoundaryRenderIntent.DASHED,
                )
                surface_front = next(
                    item
                    for item in depth_frame.draw_order
                    if item.endswith("projection-sheet:front")
                )
                self.assertLess(
                    depth_frame.draw_order.index(surface_front),
                    depth_frame.draw_order.index(fragment.item_id),
                )
                for plane_item in fragment.plane_occluder_item_ids:
                    self.assertLess(
                        depth_frame.draw_order.index(fragment.item_id),
                        depth_frame.draw_order.index(plane_item),
                    )
                outline_front = next(
                    item
                    for item in diagram_frame.draw_order
                    if item.endswith("bracket-plane:plane:outline:front")
                )
                self.assertLess(
                    diagram_frame.draw_order.index(outline_front),
                    diagram_frame.draw_order.index(diagram_fragment.item_id),
                )

                def dash_midpoint(controller, item):
                    slot_index = controller._fragment_slot_maps[item.source_id][
                        item.item_id
                    ]
                    slot = controller._curve_slots[item.source_id].fragments[
                        slot_index
                    ]
                    dash = next(dash for dash in slot.dashes if len(dash.points))
                    return 0.5 * (
                        np.asarray(dash.get_start())
                        + np.asarray(dash.get_end())
                    )

                point = dash_midpoint(diagram, diagram_fragment)
                depth_point = dash_midpoint(depth, fragment)
                np.testing.assert_allclose(point, depth_point, atol=1.0e-8)
                target = _hex_rgb("#FF0000")
                diagram_pixels = _capture_pixels(diagram_scene)
                depth_pixels = _capture_pixels(depth_scene)
                row, column, diagram_rgb = _nearest_ink_pixel(
                    diagram_pixels, point, target
                )
                depth_rgb = depth_pixels[row, column]
                self.assertLess(
                    float(np.linalg.norm(diagram_rgb - target)) + 10.0,
                    float(np.linalg.norm(depth_rgb - target)),
                )
                self.assertGreater(depth_rgb[1], diagram_rgb[1] + 20.0)
            finally:
                depth.restore()
                diagram.restore()

    def test_section_plane_occludes_cone_silhouette_under_all_policies(
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
            silhouette_style = QuadricBoundaryStyle(
                visible_color="#FF2020",
                visible_width=8.0,
                visible_opacity=1.0,
                hidden_color="#FF2020",
                hidden_width=8.0,
                hidden_opacity=1.0,
                dash_length=0.14,
                dash_gap=0.10,
            )
            physical_scene, physical = _scene_for(
                QuadricPaintPolicy.PHYSICAL,
                silhouette_style=silhouette_style,
            )
            diagram_scene, diagram = _scene_for(
                QuadricPaintPolicy.DIAGRAMMATIC,
                silhouette_style=silhouette_style,
            )
            depth_scene, depth = _scene_for(
                QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                silhouette_style=silhouette_style,
            )
            try:
                frames = {
                    QuadricPaintPolicy.PHYSICAL: physical.last_boundary_frame,
                    QuadricPaintPolicy.DIAGRAMMATIC: diagram.last_boundary_frame,
                    QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC: (
                        depth.last_boundary_frame
                    ),
                }
                self.assertTrue(all(frame is not None for frame in frames.values()))
                for policy, frame in frames.items():
                    assert frame is not None
                    source_map = {item.source_id: item for item in frame.sources}
                    silhouettes = [
                        item
                        for item in frame.fragments
                        if source_map[item.source_id].semantic_kind
                        is BoundarySemanticKind.TRUE_SILHOUETTE
                    ]
                    plane_hidden = [
                        item for item in silhouettes if item.plane_occluded
                    ]
                    unoccluded = [
                        item for item in silhouettes if not item.plane_occluded
                    ]
                    self.assertTrue(plane_hidden and unoccluded)
                    self.assertTrue(
                        all(
                            item.surface_visibility_kind
                            is VisibilityKind.VISIBLE
                            and item.effective_visibility_kind
                            is VisibilityKind.HIDDEN
                            and not item.occluder_surface_ids
                            for item in plane_hidden
                        )
                    )
                    self.assertTrue(
                        all(
                            item.effective_visibility_kind
                            is VisibilityKind.VISIBLE
                            and item.render_intent is BoundaryRenderIntent.SOLID
                            and item.painted
                            for item in unoccluded
                        )
                    )
                    expected = (
                        BoundaryRenderIntent.OMIT
                        if policy is QuadricPaintPolicy.PHYSICAL
                        else BoundaryRenderIntent.DASHED
                    )
                    self.assertTrue(
                        all(item.render_intent is expected for item in plane_hidden)
                    )

                diagram_fragment, point = _first_plane_occluded_silhouette_dash(
                    diagram
                )
                depth_fragment, depth_point = (
                    _first_plane_occluded_silhouette_dash(depth)
                )
                self.assertEqual(
                    diagram_fragment.source_id,
                    depth_fragment.source_id,
                )
                np.testing.assert_allclose(point, depth_point, atol=1.0e-8)
                depth_frame = depth.last_boundary_frame
                assert depth_frame is not None
                surface_front = next(
                    item
                    for item in depth_frame.draw_order
                    if item.endswith("projection-sheet:front")
                )
                self.assertLess(
                    depth_frame.draw_order.index(surface_front),
                    depth_frame.draw_order.index(depth_fragment.item_id),
                )
                for plane_item in depth_fragment.plane_occluder_item_ids:
                    self.assertLess(
                        depth_frame.draw_order.index(depth_fragment.item_id),
                        depth_frame.draw_order.index(plane_item),
                    )

                target = _hex_rgb("#FF2020")
                physical_pixels = _capture_pixels(physical_scene)
                diagram_pixels = _capture_pixels(diagram_scene)
                depth_pixels = _capture_pixels(depth_scene)
                row, column, diagram_rgb = _nearest_ink_pixel(
                    diagram_pixels, point, target
                )
                depth_rgb = depth_pixels[row, column]
                physical_rgb = physical_pixels[row, column]
                self.assertLess(
                    float(np.linalg.norm(diagram_rgb - target)) + 5.0,
                    float(np.linalg.norm(depth_rgb - target)),
                )
                self.assertLess(
                    float(np.linalg.norm(depth_rgb - target)) + 20.0,
                    float(np.linalg.norm(physical_rgb - target)),
                )

                identities = depth.slot_identities()
                depth.update()
                updated_fragment, updated_point = (
                    _first_plane_occluded_silhouette_dash(depth)
                )
                self.assertEqual(depth.slot_identities(), identities)
                self.assertEqual(updated_fragment.item_id, depth_fragment.item_id)
                np.testing.assert_allclose(updated_point, depth_point, atol=1.0e-8)
            finally:
                depth.restore()
                diagram.restore()
                physical.restore()

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

    def test_physical_omits_hidden_and_keeps_unoccluded_silhouette_solid(
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
                    if item.effective_visibility_kind is VisibilityKind.HIDDEN
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
                plane_occluded = [
                    item for item in silhouettes if item.plane_occluded
                ]
                self.assertTrue(plane_occluded)
                self.assertTrue(
                    all(
                        item.surface_visibility_kind
                        is VisibilityKind.VISIBLE
                        and item.effective_visibility_kind
                        is VisibilityKind.HIDDEN
                        and item.render_intent is BoundaryRenderIntent.OMIT
                        and not item.painted
                        and not item.occluder_surface_ids
                        for item in plane_occluded
                    )
                )
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
