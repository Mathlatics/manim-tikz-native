"""Conditional Cairo pixel regressions for boundary-conforming sections.

The fill-only fixtures suppress authored curve and outline ink.  Each role mask
is then eroded before pixel comparison, excluding the true silhouette, role
boundaries, patch outline, and image border without hiding an interior crack.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from math import pi, sqrt
from time import perf_counter
from typing import Sequence
import unittest

import numpy as np
from PIL import Image, ImageDraw
from manim import Scene, ValueTracker, config, tempconfig

from diagnostics.quadrics_section_boundary_partition.scene import (
    BACKGROUND_COLOR,
    PLANE_COLOR,
    STATES,
    SURFACE_COLOR,
    build_controller,
    style_for_mode,
)
from polyhedron_visibility.quadrics.compositing import (
    QuadricPaintKind,
    QuadricPaintPolicy,
)
from polyhedron_visibility.quadrics.contract import ConeSpec, SectionPlane
from polyhedron_visibility.quadrics.manim import QuadricManimLimits
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    QuadricSectionCompositingFrame,
)
from polyhedron_visibility.quadrics.transition_manim import (
    QuadricSectionTransition3D,
)


try:
    import cairo as _cairo  # noqa: F401
    from manim.renderer.cairo_renderer import CairoRenderer as _CairoRenderer  # noqa: F401
except (ImportError, OSError):
    CAIRO_AVAILABLE = False
else:
    CAIRO_AVAILABLE = True


STATIC_PIXEL_WIDTH = 480
STATIC_PIXEL_HEIGHT = 270
DYNAMIC_PIXEL_WIDTH = 320
DYNAMIC_PIXEL_HEIGHT = 180
BOUNDARY_PIXEL_WIDTH = 960
BOUNDARY_PIXEL_HEIGHT = 540
RGB_ERROR_THRESHOLD = 8.0
BOUNDARY_RGB_ERROR_THRESHOLD = 18.0
BOUNDARY_EROSION_PIXELS = 3

ROLE_ORDER = (
    PlaneDepthRole.BEHIND_SURFACE,
    PlaneDepthRole.OUTSIDE_PROJECTION,
    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
    PlaneDepthRole.IN_FRONT_OF_SURFACE,
)


def _hex_rgb(value: str) -> np.ndarray:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a six-digit RGB color, received {value!r}")
    return np.asarray(
        tuple(int(text[index : index + 2], 16) for index in (0, 2, 4)),
        dtype=float,
    )


def _source_over(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return foreground * alpha + background * (1.0 - alpha)


def _expected_role_rgb(role: PlaneDepthRole, mode: str) -> np.ndarray:
    style = style_for_mode(mode)
    background = _hex_rgb(BACKGROUND_COLOR)
    surface = _hex_rgb(SURFACE_COLOR)
    plane = _hex_rgb(PLANE_COLOR)
    sheet_alpha = 1.0 - sqrt(1.0 - style.surface_fill_opacity)
    plane_alpha = style.section_plane_fill_opacity

    if role is PlaneDepthRole.OUTSIDE_PROJECTION:
        return _source_over(background, plane, plane_alpha)
    if role is PlaneDepthRole.BEHIND_SURFACE:
        result = _source_over(background, plane, plane_alpha)
        result = _source_over(result, surface, sheet_alpha)
        return _source_over(result, surface, sheet_alpha)
    if role is PlaneDepthRole.BETWEEN_SURFACE_SHEETS:
        result = _source_over(background, surface, sheet_alpha)
        result = _source_over(result, plane, plane_alpha)
        return _source_over(result, surface, sheet_alpha)
    if role is PlaneDepthRole.IN_FRONT_OF_SURFACE:
        result = _source_over(background, surface, sheet_alpha)
        result = _source_over(result, surface, sheet_alpha)
        return _source_over(result, plane, plane_alpha)
    raise AssertionError(role)


def _screen_to_pixel(
    point: Sequence[float],
    *,
    width: int,
    height: int,
    frame_width: float,
    frame_height: float,
) -> tuple[float, float]:
    x, y = (float(value) for value in point[:2])
    return (
        (x / frame_width + 0.5) * (width - 1),
        (0.5 - y / frame_height) * (height - 1),
    )


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result &= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return result


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return result


def _polygon_mask(
    polygons: Sequence[Sequence[Sequence[float]]],
    *,
    width: int,
    height: int,
    frame_width: float,
    frame_height: float,
    erosion_pixels: int = BOUNDARY_EROSION_PIXELS,
) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(
            tuple(
                _screen_to_pixel(
                    point,
                    width=width,
                    height=height,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                for point in polygon
            ),
            fill=255,
        )
    return _erode(np.asarray(image, dtype=np.uint8) > 0, erosion_pixels)


def _role_mask(
    frame: QuadricSectionCompositingFrame,
    role: PlaneDepthRole,
    *,
    width: int,
    height: int,
    frame_width: float,
    frame_height: float,
    erosion_pixels: int = BOUNDARY_EROSION_PIXELS,
) -> np.ndarray:
    return _polygon_mask(
        tuple(
            fragment.screen_vertices
            for fragment in frame.fragments_by_role[role]
        ),
        width=width,
        height=height,
        frame_width=frame_width,
        frame_height=frame_height,
        erosion_pixels=erosion_pixels,
    )


def _rgb_segment_distance(
    pixels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    delta = second - first
    length_squared = float(np.dot(delta, delta))
    if length_squared <= 0.0:
        return np.linalg.norm(pixels - first, axis=2)
    ratio = np.clip(
        np.sum((pixels - first) * delta, axis=2) / length_squared,
        0.0,
        1.0,
    )
    expected = first + ratio[:, :, np.newaxis] * delta
    return np.linalg.norm(pixels - expected, axis=2)


def _rgb_triangle_distance(
    pixels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> np.ndarray:
    """Distance to the legal Cairo AA hull of two fills and the background."""

    first_axis = second - first
    second_axis = third - first
    gram = np.asarray(
        (
            (np.dot(first_axis, first_axis), np.dot(first_axis, second_axis)),
            (np.dot(first_axis, second_axis), np.dot(second_axis, second_axis)),
        ),
        dtype=float,
    )
    determinant = float(np.linalg.det(gram))
    edge_error = np.minimum.reduce(
        (
            _rgb_segment_distance(pixels, first, second),
            _rgb_segment_distance(pixels, first, third),
            _rgb_segment_distance(pixels, second, third),
        )
    )
    if abs(determinant) <= 1.0e-12:
        return edge_error
    inverse = np.linalg.inv(gram)
    relative = pixels - first
    right_hand_side = np.stack(
        (
            np.sum(relative * first_axis, axis=2),
            np.sum(relative * second_axis, axis=2),
        ),
        axis=2,
    )
    coordinates = right_hand_side @ inverse.T
    first_weight = coordinates[:, :, 0]
    second_weight = coordinates[:, :, 1]
    inside = (
        (first_weight >= 0.0)
        & (second_weight >= 0.0)
        & (first_weight + second_weight <= 1.0)
    )
    projection = (
        first
        + first_weight[:, :, np.newaxis] * first_axis
        + second_weight[:, :, np.newaxis] * second_axis
    )
    plane_error = np.linalg.norm(pixels - projection, axis=2)
    return np.where(inside, plane_error, edge_error)


def _role_boundary_pixel_issues(
    frame: QuadricSectionCompositingFrame,
    pixels: np.ndarray,
    mode: str,
    *,
    width: int,
    height: int,
    frame_width: float,
    frame_height: float,
) -> tuple[int, int, int, float]:
    """Inspect internal role boundaries without eroding either neighboring role."""

    raw_masks = {
        role: _role_mask(
            frame,
            role,
            width=width,
            height=height,
            frame_width=frame_width,
            frame_height=frame_height,
            erosion_pixels=0,
        )
        for role in ROLE_ORDER
    }
    patch_mask = np.logical_or.reduce(tuple(raw_masks.values()))
    # Remove only the outer display-patch edge.  The surface silhouette and
    # every behind/between/front transition remain inside this mask.
    patch_interior = _erode(patch_mask, 2)
    neighborhoods = {
        role: _dilate(mask, 1) for role, mask in raw_masks.items()
    }
    neighborhood_count = sum(
        mask.astype(np.uint8) for mask in neighborhoods.values()
    )
    pair_masks: list[tuple[PlaneDepthRole, PlaneDepthRole, np.ndarray]] = []
    for first_role, second_role in combinations(ROLE_ORDER, 2):
        pair_mask = (
            neighborhoods[first_role]
            & neighborhoods[second_role]
            & patch_interior
            & (neighborhood_count == 2)
        )
        if np.any(pair_mask):
            pair_masks.append((first_role, second_role, pair_mask))

    if not pair_masks:
        return 0, 0, 0, 0.0
    boundary_mask = np.logical_or.reduce(
        tuple(item[2] for item in pair_masks)
    )
    background = _hex_rgb(BACKGROUND_COLOR)
    # At a shared role boundary Cairo may partially cover the plane, the
    # surface sheet, and the background in one pixel.  Those source colors form
    # the complete legal antialias hull; a pure background pixel is rejected
    # separately below as an actual gap.
    legal_error = _rgb_triangle_distance(
        pixels,
        background,
        _hex_rgb(PLANE_COLOR),
        _hex_rgb(SURFACE_COLOR),
    )

    background_error = np.linalg.norm(
        pixels - background,
        axis=2,
    )
    boundary_count = int(np.count_nonzero(boundary_mask))
    background_gaps = int(
        np.count_nonzero(
            boundary_mask & (background_error <= RGB_ERROR_THRESHOLD)
        )
    )
    illegal_colors = int(
        np.count_nonzero(
            boundary_mask
            & (legal_error > BOUNDARY_RGB_ERROR_THRESHOLD)
        )
    )
    maximum_error = float(np.max(legal_error[boundary_mask]))
    return boundary_count, background_gaps, illegal_colors, maximum_error


def _capture_pixels(scene: Scene) -> np.ndarray:
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    return scene.camera.pixel_array[:, :, :3].copy()


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    values = np.asarray(points, dtype=float)
    return 0.5 * abs(
        sum(
            values[index, 0] * values[(index + 1) % len(values), 1]
            - values[index, 1] * values[(index + 1) % len(values), 0]
            for index in range(len(values))
        )
    )


def _role_areas(
    frame: QuadricSectionCompositingFrame,
) -> tuple[float, float, float, float]:
    return tuple(
        sum(
            _polygon_area(fragment.screen_vertices)
            for fragment in frame.fragments_by_role[role]
        )
        for role in ROLE_ORDER
    )


def _transition_schedule():
    cone = ConeSpec(
        "cairo-transition-cone",
        (0.0, 0.0, -1.5),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    motion = AxisAnglePlaneMotion(
        "cairo-transition-motion",
        SectionPlane(
            "cairo-transition-plane",
            (0.0, 0.0, 0.2),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        (0.0, 0.0, 0.2),
        (0.0, 1.0, 0.0),
        0.0,
        1.2,
    )
    return track_scheduled_plane_section(
        "cairo-transition-section",
        cone,
        motion,
    )


def _transition_limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=2,
        # Eight lateral transition-bank slots plus one stable cap chord.
        max_curves=9,
        max_fragments_per_curve=16,
        max_segments_per_fragment=256,
        max_surface_segments=512,
        max_dashes_per_fragment=72,
        max_projected_length=18.0,
        max_total_mobjects=12000,
    )


@unittest.skipUnless(CAIRO_AVAILABLE, "Manim Cairo renderer is unavailable")
class QuadricSectionCairoRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.static_frames: dict[
            tuple[str, str],
            dict[str, object],
        ] = {}
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": STATIC_PIXEL_WIDTH,
                "pixel_height": STATIC_PIXEL_HEIGHT,
                "frame_rate": 12,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        ):
            cls.static_frame_width = float(config.frame_width)
            cls.static_frame_height = float(config.frame_height)
            for mode in ("opaque_fill", "translucent_fill"):
                state = {"name": STATES[0].name}
                scene = Scene()
                scene.camera.background_color = BACKGROUND_COLOR
                controller = build_controller(
                    scene,
                    lambda: state["name"],
                    mode,
                ).attach()
                identities = controller.slot_identities()
                try:
                    for index, definition in enumerate(STATES):
                        state["name"] = definition.name
                        if index:
                            controller.update()
                        frame = controller.last_section_frame
                        if frame is None:
                            raise AssertionError(
                                f"missing section frame for {definition.name}"
                            )
                        slots = dict(
                            zip(
                                frame.paint_items.ordered,
                                controller._section_slots,
                            )
                        )
                        cls.static_frames[(mode, definition.name)] = {
                            "frame": frame,
                            "pixels": _capture_pixels(scene),
                            "identity_stable": (
                                controller.slot_identities() == identities
                            ),
                            "active_z": dict(
                                controller.active_painter_z_indices
                            ),
                            "outline_point_counts": {
                                role: int(
                                    len(
                                        slots[
                                            frame.paint_items.outline_by_role[role]
                                        ].points
                                    )
                                )
                                for role in ROLE_ORDER
                            },
                        }
                finally:
                    controller.restore()

    def test_high_resolution_role_boundaries_have_no_cairo_gaps(self) -> None:
        boundary_pixels_by_mode = {
            "opaque_fill": 0,
            "translucent_fill": 0,
        }
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": BOUNDARY_PIXEL_WIDTH,
                "pixel_height": BOUNDARY_PIXEL_HEIGHT,
                "frame_rate": 12,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        ):
            for mode in ("opaque_fill", "translucent_fill"):
                state = {"name": STATES[0].name}
                scene = Scene()
                scene.camera.background_color = BACKGROUND_COLOR
                controller = build_controller(
                    scene,
                    lambda: state["name"],
                    mode,
                ).attach()
                try:
                    for index, definition in enumerate(STATES):
                        with self.subTest(mode=mode, state=definition.name):
                            state["name"] = definition.name
                            if index:
                                controller.update()
                            frame = controller.last_section_frame
                            self.assertIsNotNone(frame)
                            assert frame is not None
                            pixels = _capture_pixels(scene).astype(float)
                            (
                                boundary_count,
                                background_gaps,
                                illegal_colors,
                                maximum_error,
                            ) = _role_boundary_pixel_issues(
                                frame,
                                pixels,
                                mode,
                                width=BOUNDARY_PIXEL_WIDTH,
                                height=BOUNDARY_PIXEL_HEIGHT,
                                frame_width=float(config.frame_width),
                                frame_height=float(config.frame_height),
                            )
                            boundary_pixels_by_mode[mode] += boundary_count
                            self.assertGreater(boundary_count, 20)
                            self.assertEqual(
                                background_gaps,
                                0,
                                "an internal role boundary contains a "
                                "background-colored Cairo gap",
                            )
                            self.assertEqual(
                                illegal_colors,
                                0,
                                "an internal role boundary contains a color "
                                "outside the adjacent-role antialias range "
                                f"(maximum RGB distance {maximum_error:.6g})",
                            )
                finally:
                    controller.restore()

        for mode, boundary_count in boundary_pixels_by_mode.items():
            with self.subTest(mode=mode):
                self.assertGreater(boundary_count, 500)

    def test_five_state_role_interiors_have_no_cairo_seam_pixels(self) -> None:
        safe_pixels_by_role = {role: 0 for role in ROLE_ORDER}
        background = _hex_rgb(BACKGROUND_COLOR)

        for mode in ("opaque_fill", "translucent_fill"):
            for definition in STATES:
                with self.subTest(mode=mode, state=definition.name):
                    evidence = self.static_frames[(mode, definition.name)]
                    frame = evidence["frame"]
                    pixels = np.asarray(evidence["pixels"], dtype=float)
                    self.assertTrue(evidence["identity_stable"])
                    assert isinstance(frame, QuadricSectionCompositingFrame)

                    frame_seams = 0
                    for role in ROLE_ORDER:
                        mask = _role_mask(
                            frame,
                            role,
                            width=STATIC_PIXEL_WIDTH,
                            height=STATIC_PIXEL_HEIGHT,
                            frame_width=self.static_frame_width,
                            frame_height=self.static_frame_height,
                        )
                        safe_pixels_by_role[role] += int(np.count_nonzero(mask))
                        expected = _expected_role_rgb(role, mode)
                        role_error = np.linalg.norm(pixels - expected, axis=2)
                        frame_seams += int(
                            np.count_nonzero(
                                mask & (role_error > RGB_ERROR_THRESHOLD)
                            )
                        )
                        background_error = np.linalg.norm(
                            pixels - background,
                            axis=2,
                        )
                        self.assertEqual(
                            int(
                                np.count_nonzero(
                                    mask
                                    & (
                                        background_error
                                        <= RGB_ERROR_THRESHOLD
                                    )
                                )
                            ),
                            0,
                            "a safe role interior contains a background-colored gap",
                        )
                    self.assertEqual(frame_seams, 0)

        for role, pixel_count in safe_pixels_by_role.items():
            with self.subTest(role=role.value):
                self.assertGreater(pixel_count, 100)

    def test_five_state_role_colors_match_cairo_over_composites(self) -> None:
        for mode in ("opaque_fill", "translucent_fill"):
            for definition in STATES:
                evidence = self.static_frames[(mode, definition.name)]
                frame = evidence["frame"]
                pixels = np.asarray(evidence["pixels"], dtype=float)
                assert isinstance(frame, QuadricSectionCompositingFrame)
                for role in ROLE_ORDER:
                    with self.subTest(
                        mode=mode,
                        state=definition.name,
                        role=role.value,
                    ):
                        mask = _role_mask(
                            frame,
                            role,
                            width=STATIC_PIXEL_WIDTH,
                            height=STATIC_PIXEL_HEIGHT,
                            frame_width=self.static_frame_width,
                            frame_height=self.static_frame_height,
                        )
                        if not np.any(mask):
                            continue
                        expected = _expected_role_rgb(role, mode)
                        median = np.median(pixels[mask], axis=0)
                        self.assertLessEqual(
                            float(np.linalg.norm(median - expected)),
                            2.1,
                        )

    def test_surface_only_exact_parabola_restores_target_opacity(self) -> None:
        style = replace(
            style_for_mode("translucent_fill"),
            section_plane_fill_opacity=0.0,
            section_plane_stroke_opacity=0.0,
        )
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": STATIC_PIXEL_WIDTH,
                "pixel_height": STATIC_PIXEL_HEIGHT,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        ):
            scene = Scene()
            scene.camera.background_color = BACKGROUND_COLOR
            controller = build_controller(
                scene,
                lambda: "exact_parabola",
                "translucent_fill",
                style=style,
            ).attach()
            try:
                frame = controller.last_section_frame
                self.assertIsNotNone(frame)
                assert frame is not None
                pixels = _capture_pixels(scene).astype(float)
                mask = _polygon_mask(
                    (frame.surface_proxy.boundary_points,),
                    width=STATIC_PIXEL_WIDTH,
                    height=STATIC_PIXEL_HEIGHT,
                    frame_width=float(config.frame_width),
                    frame_height=float(config.frame_height),
                    erosion_pixels=4,
                )
                self.assertGreater(int(np.count_nonzero(mask)), 1000)

                expected = _source_over(
                    _hex_rgb(BACKGROUND_COLOR),
                    _hex_rgb(SURFACE_COLOR),
                    style.surface_fill_opacity,
                )
                errors = np.linalg.norm(pixels - expected, axis=2)
                self.assertEqual(
                    int(
                        np.count_nonzero(
                            mask & (errors > RGB_ERROR_THRESHOLD)
                        )
                    ),
                    0,
                )

                slots = dict(
                    zip(frame.paint_items.ordered, controller._section_slots)
                )
                back = slots[frame.paint_items.surface_back]
                front = slots[frame.paint_items.surface_front]
                np.testing.assert_array_equal(back.points, front.points)
                expected_sheet_opacity = 1.0 - sqrt(
                    1.0 - style.surface_fill_opacity
                )
                self.assertAlmostEqual(
                    float(back.get_fill_opacity()),
                    expected_sheet_opacity,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(front.get_fill_opacity()),
                    expected_sheet_opacity,
                    places=12,
                )
                combined_opacity = 1.0 - (
                    1.0 - expected_sheet_opacity
                ) ** 2
                self.assertAlmostEqual(
                    combined_opacity,
                    style.surface_fill_opacity,
                    places=12,
                )
            finally:
                controller.restore()

    def test_hidden_curve_policies_remain_overlay_and_physical_omission(
        self,
    ) -> None:
        style = replace(
            style_for_mode("translucent_fill"),
            visible_curve_color="#00A96B",
            visible_curve_width=3.2,
            visible_curve_opacity=1.0,
            hidden_curve_color="#E50046",
            hidden_curve_width=3.2,
            hidden_curve_opacity=1.0,
        )
        rendered: dict[str, np.ndarray] = {}

        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": STATIC_PIXEL_WIDTH,
                "pixel_height": STATIC_PIXEL_HEIGHT,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        ):
            for policy in ("diagrammatic", "physical"):
                scene = Scene()
                scene.camera.background_color = BACKGROUND_COLOR
                controller = build_controller(
                    scene,
                    lambda: "exact_parabola",
                    "translucent_fill",
                    paint_policy=policy,
                    style=style,
                ).attach()
                try:
                    frame = controller.last_section_frame
                    self.assertIsNotNone(frame)
                    assert frame is not None
                    hidden = tuple(
                        fragment
                        for fragment in frame.base_frame.curve_fragments
                        if fragment.kind is QuadricPaintKind.HIDDEN_CURVE
                    )
                    self.assertTrue(hidden)
                    if policy == "diagrammatic":
                        self.assertIs(
                            frame.base_frame.paint_policy,
                            QuadricPaintPolicy.DIAGRAMMATIC,
                        )
                        self.assertTrue(all(item.painted for item in hidden))
                        self.assertTrue(
                            all(item.render_intent == "dashed" for item in hidden)
                        )
                        depth_ceiling = max(
                            frame.draw_order.index(item_id)
                            for item_id in frame.paint_items.depth_chain
                        )
                        self.assertTrue(
                            all(
                                frame.draw_order.index(item.item_id)
                                > depth_ceiling
                                for item in hidden
                            )
                        )
                    else:
                        self.assertIs(
                            frame.base_frame.paint_policy,
                            QuadricPaintPolicy.PHYSICAL,
                        )
                        self.assertTrue(
                            all(not item.painted for item in hidden)
                        )
                        self.assertTrue(
                            all(item.render_intent == "omit" for item in hidden)
                        )
                        self.assertTrue(
                            all(
                                item.item_id not in frame.draw_order
                                for item in hidden
                            )
                        )
                    rendered[policy] = _capture_pixels(scene)
                finally:
                    controller.restore()

        changed = np.any(
            np.abs(
                rendered["diagrammatic"].astype(int)
                - rendered["physical"].astype(int)
            )
            > 4,
            axis=2,
        )
        self.assertGreater(int(np.count_nonzero(changed)), 20)

    def test_depth_aware_hidden_dashes_are_attenuated_by_front_sheet(
        self,
    ) -> None:
        hidden_color = "#E50046"
        style = replace(
            style_for_mode("translucent_fill"),
            visible_curve_color="#00A96B",
            visible_curve_width=3.2,
            visible_curve_opacity=1.0,
            hidden_curve_color=hidden_color,
            hidden_curve_width=3.2,
            hidden_curve_opacity=1.0,
        )
        rendered: dict[str, np.ndarray] = {}

        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": STATIC_PIXEL_WIDTH,
                "pixel_height": STATIC_PIXEL_HEIGHT,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        ):
            for policy in (
                "diagrammatic",
                "depth_aware_diagrammatic",
                "physical",
            ):
                scene = Scene()
                scene.camera.background_color = BACKGROUND_COLOR
                controller = build_controller(
                    scene,
                    lambda: "exact_parabola",
                    "translucent_fill",
                    paint_policy=policy,
                    style=style,
                ).attach()
                try:
                    frame = controller.last_section_frame
                    self.assertIsNotNone(frame)
                    assert frame is not None
                    hidden = tuple(
                        item
                        for item in frame.base_frame.curve_fragments
                        if item.kind is QuadricPaintKind.HIDDEN_CURVE
                    )
                    visible = tuple(
                        item
                        for item in frame.base_frame.curve_fragments
                        if item.kind is QuadricPaintKind.VISIBLE_CURVE
                    )
                    self.assertTrue(hidden)
                    self.assertTrue(visible)

                    if policy == "depth_aware_diagrammatic":
                        self.assertIs(
                            frame.base_frame.paint_policy,
                            QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                        )
                        self.assertTrue(all(item.painted for item in hidden))
                        self.assertTrue(
                            all(item.render_intent == "dashed" for item in hidden)
                        )
                        ranks = {
                            item_id: index
                            for index, item_id in enumerate(frame.draw_order)
                        }
                        self.assertLess(
                            ranks[frame.paint_items.plane_outline_between],
                            min(ranks[item.item_id] for item in hidden),
                        )
                        self.assertLess(
                            max(ranks[item.item_id] for item in hidden),
                            ranks[frame.paint_items.surface_front],
                        )
                        self.assertLess(
                            ranks[frame.paint_items.plane_outline],
                            min(ranks[item.item_id] for item in visible),
                        )
                    rendered[policy] = _capture_pixels(scene).astype(float)
                finally:
                    controller.restore()

        hidden_rgb = _hex_rgb(hidden_color)
        diagrammatic_hidden_pixels = (
            np.linalg.norm(rendered["diagrammatic"] - hidden_rgb, axis=2)
            <= RGB_ERROR_THRESHOLD
        )
        hidden_pixel_count = int(np.count_nonzero(diagrammatic_hidden_pixels))
        self.assertGreater(hidden_pixel_count, 8)

        sheet_alpha = 1.0 - sqrt(1.0 - style.surface_fill_opacity)
        expected_attenuated = _source_over(
            hidden_rgb,
            _hex_rgb(SURFACE_COLOR),
            sheet_alpha,
        )
        depth_errors = np.linalg.norm(
            rendered["depth_aware_diagrammatic"] - expected_attenuated,
            axis=2,
        )
        self.assertEqual(
            int(
                np.count_nonzero(
                    diagrammatic_hidden_pixels
                    & (depth_errors <= RGB_ERROR_THRESHOLD)
                )
            ),
            hidden_pixel_count,
        )
        depth_pixels = rendered["depth_aware_diagrammatic"][
            diagrammatic_hidden_pixels
        ]
        self.assertLess(
            float(
                np.median(
                    np.linalg.norm(depth_pixels - expected_attenuated, axis=1)
                )
            ),
            4.0,
        )
        self.assertGreater(
            float(np.median(np.linalg.norm(depth_pixels - hidden_rgb, axis=1))),
            40.0,
        )

        diagrammatic_changed = np.any(
            np.abs(
                rendered["diagrammatic"]
                - rendered["depth_aware_diagrammatic"]
            )
            > 4,
            axis=2,
        )
        physical_changed = np.any(
            np.abs(
                rendered["depth_aware_diagrammatic"]
                - rendered["physical"]
            )
            > 4,
            axis=2,
        )
        self.assertGreater(int(np.count_nonzero(diagrammatic_changed)), 50)
        self.assertGreater(int(np.count_nonzero(physical_changed)), 50)

    def test_outline_roles_keep_distinct_depth_owned_cairo_slots(self) -> None:
        observed_roles: set[PlaneDepthRole] = set()
        for definition in STATES:
            with self.subTest(state=definition.name):
                evidence = self.static_frames[
                    ("translucent_fill", definition.name)
                ]
                frame = evidence["frame"]
                active_z = evidence["active_z"]
                point_counts = evidence["outline_point_counts"]
                assert isinstance(frame, QuadricSectionCompositingFrame)
                assert isinstance(active_z, dict)
                assert isinstance(point_counts, dict)

                outline_by_role = frame.paint_items.outline_by_role
                self.assertEqual(set(outline_by_role), set(ROLE_ORDER))
                self.assertEqual(len(set(outline_by_role.values())), 4)
                self.assertEqual(
                    tuple(
                        item_id
                        for item_id in frame.draw_order
                        if item_id in set(frame.paint_items.depth_chain)
                    ),
                    frame.paint_items.depth_chain,
                )
                outline_z = tuple(
                    float(active_z[outline_by_role[role]])
                    for role in ROLE_ORDER
                )
                self.assertEqual(len(set(outline_z)), 4)

                for role in ROLE_ORDER:
                    fragments = frame.outline_fragments_by_role[role]
                    if not fragments:
                        continue
                    observed_roles.add(role)
                    self.assertGreater(int(point_counts[role]), 0)
                    self.assertIn(outline_by_role[role], frame.draw_order)

        # The fitted diagnostic patch does not force its outer rectangle
        # through the solid interior, so a BETWEEN outline interval need not
        # occur in these five states.  Its dedicated slot and painter layer are
        # nevertheless present and distinct above; geometry is populated only
        # for roles that genuinely occur on the outline.
        self.assertEqual(
            observed_roles,
            {
                PlaneDepthRole.BEHIND_SURFACE,
                PlaneDepthRole.OUTSIDE_PROJECTION,
                PlaneDepthRole.IN_FRONT_OF_SURFACE,
            },
        )

    def test_continuous_cairo_motion_has_no_seam_or_near_tangent_jump(
        self,
    ) -> None:
        scheduled = _transition_schedule()
        progress = ValueTracker(0.0)

        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": DYNAMIC_PIXEL_WIDTH,
                "pixel_height": DYNAMIC_PIXEL_HEIGHT,
                "frame_rate": 12,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        ):
            scene = Scene()
            scene.camera.background_color = BACKGROUND_COLOR
            controller = QuadricSectionTransition3D(
                scene,
                scheduled=scheduled,
                progress=progress,
                style=style_for_mode("translucent_fill"),
                limits=_transition_limits(),
                max_chord_error=0.025,
                transition_fraction=0.055,
                section_max_screen_error=0.05,
            ).attach()
            identities = controller.slot_identities()
            painter_band = controller.controller._band.z_band
            parabolic = next(
                item
                for item in controller.plan.knots
                if "cone_parabolic" in item.critical_kinds
            )
            near_progress = tuple(
                parabolic.progress + offset
                for offset in (-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04)
            )
            sample_progress = tuple(
                sorted(
                    {
                        0.0,
                        0.25,
                        0.5,
                        0.75,
                        1.0,
                        *near_progress,
                    }
                )
            )
            near_labels: list[np.ndarray] = []
            near_areas: list[tuple[float, float, float, float]] = []
            start = perf_counter()
            try:
                for value in sample_progress:
                    with self.subTest(progress=value):
                        progress.set_value(value)
                        controller.update()
                        self.assertEqual(
                            controller.slot_identities(),
                            identities,
                        )
                        frame = controller.controller.last_section_frame
                        self.assertIsNotNone(frame)
                        assert frame is not None
                        self.assertEqual(
                            controller.controller._band.z_band,
                            painter_band,
                        )
                        self.assertEqual(
                            tuple(
                                item_id
                                for item_id in frame.draw_order
                                if item_id in set(frame.paint_items.depth_chain)
                            ),
                            frame.paint_items.depth_chain,
                        )
                        active_z = controller.controller.active_painter_z_indices
                        depth_z = tuple(
                            float(active_z[item_id])
                            for item_id in frame.paint_items.depth_chain
                        )
                        self.assertEqual(depth_z, tuple(sorted(depth_z)))
                        self.assertTrue(
                            all(
                                painter_band[0] <= item <= painter_band[1]
                                for item in depth_z
                            )
                        )
                        self.assertLessEqual(len(frame.plane_fragments), 8192)
                        self.assertLessEqual(
                            frame.ray_classification_count,
                            65536,
                        )

                        pixels = _capture_pixels(scene).astype(float)
                        labels = np.zeros(
                            (DYNAMIC_PIXEL_HEIGHT, DYNAMIC_PIXEL_WIDTH),
                            dtype=np.uint8,
                        )
                        seam_pixels = 0
                        for label, role in enumerate(ROLE_ORDER, start=1):
                            mask = _role_mask(
                                frame,
                                role,
                                width=DYNAMIC_PIXEL_WIDTH,
                                height=DYNAMIC_PIXEL_HEIGHT,
                                frame_width=float(config.frame_width),
                                frame_height=float(config.frame_height),
                            )
                            labels[mask] = label
                            error = np.linalg.norm(
                                pixels
                                - _expected_role_rgb(
                                    role,
                                    "translucent_fill",
                                ),
                                axis=2,
                            )
                            seam_pixels += int(
                                np.count_nonzero(
                                    mask
                                    & (error > RGB_ERROR_THRESHOLD)
                                )
                            )
                        self.assertEqual(seam_pixels, 0)

                        if any(
                            abs(value - item) <= 1.0e-12
                            for item in near_progress
                        ):
                            (
                                boundary_count,
                                background_gaps,
                                illegal_colors,
                                maximum_error,
                            ) = _role_boundary_pixel_issues(
                                frame,
                                pixels,
                                "translucent_fill",
                                width=DYNAMIC_PIXEL_WIDTH,
                                height=DYNAMIC_PIXEL_HEIGHT,
                                frame_width=float(config.frame_width),
                                frame_height=float(config.frame_height),
                            )
                            self.assertGreater(boundary_count, 4)
                            self.assertEqual(background_gaps, 0)
                            self.assertEqual(
                                illegal_colors,
                                0,
                                "near-tangent role boundary contains an "
                                "illegal Cairo color "
                                f"(maximum RGB distance {maximum_error:.6g})",
                            )
                            near_labels.append(labels)
                            near_areas.append(_role_areas(frame))
            finally:
                controller.restore()
            elapsed = perf_counter() - start

        self.assertEqual(len(near_labels), len(near_progress))
        for index in range(1, len(near_labels)):
            previous_labels = near_labels[index - 1]
            current_labels = near_labels[index]
            valid = (previous_labels != 0) | (current_labels != 0)
            changed_fraction = float(
                np.count_nonzero(
                    valid & (previous_labels != current_labels)
                )
                / np.count_nonzero(valid)
            )
            self.assertLess(changed_fraction, 0.12)

            previous_areas = near_areas[index - 1]
            current_areas = near_areas[index]
            normalizer = max(sum(previous_areas), sum(current_areas))
            maximum_role_delta = max(
                abs(before - after)
                for before, after in zip(previous_areas, current_areas)
            ) / normalizer
            self.assertLess(maximum_role_delta, 0.12)

        # This is deliberately generous: it rejects the former multi-minute
        # over-refinement regression without turning shared-runner speed into
        # a release promise.
        self.assertLess(elapsed, 60.0)


if __name__ == "__main__":
    unittest.main()
