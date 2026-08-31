from __future__ import annotations

import copy
from math import ceil, sqrt
from pathlib import Path
import unittest

import numpy as np
from manim import DashedVMobject, Scene, VGroup, tempconfig

from polyhedron_visibility.quadrics.section_compositing import PlaneDepthRole
from tikz_native import compile_document
from tikz_native.provider import instantiate_picture


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND_RGB = np.asarray((16, 24, 32), dtype=np.int16)
FRAME_WIDTH = 14.222
FRAME_HEIGHT = 8.0


def _source() -> str:
    return r"""
\begin{tikzpicture}[3d view={38}{24}]
  \coordinate (A) at (0,0,0);
  \coordinate (Z) at (0,0,1);
  \coordinate (R) at (1,0,0);
  \coordinate (O) at (0,0,2);
  \coordinate (U) at (0,1,2);
  \coordinate (V) at (-0.8,0,2.6);
  \DeclareSpacePlane{cut}{O/U/V}
  \DeclareSpaceRightCone{cone}{A/Z/R}{30}{0/9}{open_single}
  \DeclareDandelinConstruction{dan}{cone}{cut}
  \DrawDandelinDiagram[
    view=spatial,
    mode=depth_aware_teaching_transparent,
    show-directrices=false
  ]{dan}
\end{tikzpicture}
"""


def _capture(group: object) -> np.ndarray:
    scene = Scene()
    scene.camera.background_color = "#101820"
    scene.add(group)
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    return scene.camera.pixel_array[:, :, :3].copy()


def _paint_roots(group: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in group.get_family():
        metadata = getattr(item, "dandelin_metadata", None)
        if not isinstance(metadata, dict):
            continue
        item_id = metadata.get("paintItemId")
        if isinstance(item_id, str) and item_id:
            result[item_id] = item
    return result


def _hide_feature_strokes(group: object) -> None:
    for item in group.get_family():
        metadata = getattr(item, "dandelin_metadata", None)
        if not isinstance(metadata, dict):
            continue
        if metadata.get("renderIntent") in {"solid", "dashed"} or metadata.get(
            "semanticKind"
        ) == "focus":
            item.set_opacity(0.0)


def _median_near_point(
    pixels: np.ndarray,
    point: np.ndarray,
    *,
    frame_width: float,
    frame_height: float,
) -> np.ndarray:
    height, width = pixels.shape[:2]
    column = int(round((float(point[0]) / frame_width + 0.5) * (width - 1)))
    row = int(round((0.5 - float(point[1]) / frame_height) * (height - 1)))
    radius = 3
    window = pixels[
        max(0, row - radius) : min(height, row + radius + 1),
        max(0, column - radius) : min(width, column + radius + 1),
    ]
    return np.median(window, axis=(0, 1))


def _hex_rgb(value: str) -> np.ndarray:
    text = value.lstrip("#")
    return np.asarray(
        tuple(int(text[index : index + 2], 16) for index in (0, 2, 4)),
        dtype=float,
    )


def _plane_fill_mask(pixels: np.ndarray) -> np.ndarray:
    difference = np.max(
        np.abs(pixels.astype(np.int16) - BACKGROUND_RGB),
        axis=2,
    )
    ink = difference > 2
    if int(np.count_nonzero(ink)) < 100:
        raise AssertionError("plane fill did not produce enough Cairo pixels")
    return ink


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    height, width = mask.shape
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result &= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return result


def _source_over(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return foreground * alpha + background * (1.0 - alpha)


def _outline_ink_stats(
    pixels: np.ndarray,
    paths: object,
    runtime: object,
) -> tuple[float, int, int]:
    height, width = pixels.shape[:2]
    translation = np.asarray(runtime.metadata["displayTranslation"], dtype=float)
    scale = float(runtime.dandelin_metadata["sceneScale"])
    all_states: list[bool] = []
    transitions = 0
    longest_gap = 0
    for raw_path in paths:
        points = (np.asarray(raw_path, dtype=float) + translation) * scale
        states: list[bool] = []
        for segment_index, (start, end) in enumerate(zip(points, points[1:])):
            pixel_delta = np.asarray(
                (
                    (end[0] - start[0]) * (width - 1) / FRAME_WIDTH,
                    (end[1] - start[1]) * (height - 1) / FRAME_HEIGHT,
                ),
                dtype=float,
            )
            sample_count = max(2, int(ceil(1.5 * np.linalg.norm(pixel_delta))))
            parameters = np.linspace(
                0.0,
                1.0,
                sample_count,
                endpoint=segment_index == len(points) - 2,
            )
            for parameter in parameters:
                point = start + float(parameter) * (end - start)
                column = int(
                    round((point[0] / FRAME_WIDTH + 0.5) * (width - 1))
                )
                row = int(round((0.5 - point[1] / FRAME_HEIGHT) * (height - 1)))
                window = pixels[
                    max(0, row - 1) : min(height, row + 2),
                    max(0, column - 1) : min(width, column + 2),
                ]
                states.append(
                    bool(
                        np.max(
                            np.abs(window.astype(np.int16) - BACKGROUND_RGB)
                        )
                        > 8
                    )
                )
        transitions += sum(
            left != right for left, right in zip(states, states[1:])
        )
        run = 0
        for state in states:
            run = 0 if state else run + 1
            longest_gap = max(longest_gap, run)
        all_states.extend(states)
    if not all_states:
        raise AssertionError("plane outline produced no sample points")
    return (
        float(sum(all_states)) / float(len(all_states)),
        transitions,
        longest_gap,
    )


class TikzNativeDandelinCairoTests(unittest.TestCase):
    def test_plane_occlusion_dashes_and_tint_reach_real_cairo_pixels(
        self,
    ) -> None:
        source = (
            ROOT
            / "examples"
            / "tikz_dandelin_views"
            / "tikz_dandelin_views.tex"
        ).read_text(encoding="utf-8")
        picture = compile_document(source_text=source).pictures[0]
        self.assertFalse(picture.unsupported)
        with tempconfig(
            {
                "pixel_width": 960,
                "pixel_height": 540,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
            }
        ):
            runtime = instantiate_picture(picture).objects["dan:view:spatial"]
            roots = _paint_roots(runtime)
            frame = runtime.surface_layer_frame

            fill_pixels = {
                layer.role: _capture(roots[layer.item_id].copy())
                for layer in frame.plane_layers
            }
            fill_masks = {
                role: _erode(_plane_fill_mask(pixels), 2)
                for role, pixels in fill_pixels.items()
            }
            fill_medians = {
                role: np.median(pixels[fill_masks[role]], axis=0)
                for role, pixels in fill_pixels.items()
            }
            surface_ids = {
                item_id
                for layer in frame.cone_layers
                for item_id in (layer.back_item_id, layer.front_item_id)
            } | {layer.item_id for layer in frame.plane_layers}
            surface_composite = VGroup(
                *(
                    roots[item_id].copy()
                    for item_id in frame.draw_order
                    if item_id in surface_ids
                )
            )
            surface_pixels = _capture(surface_composite)
            surface_medians = {
                role: np.median(surface_pixels[mask], axis=0)
                for role, mask in fill_masks.items()
            }
            outline_layers = {
                layer.role: layer for layer in frame.plane_outline_layers
            }
            behind_layer = outline_layers[PlaneDepthRole.BEHIND_SURFACE]
            outside_layer = outline_layers[PlaneDepthRole.OUTSIDE_PROJECTION]
            behind_outline = roots[behind_layer.item_id]
            outside_outline = roots[outside_layer.item_id]
            behind_pixels = _capture(behind_outline.copy())
            outside_pixels = _capture(outside_outline.copy())
            behind_stats = _outline_ink_stats(
                behind_pixels,
                behind_layer.paths,
                runtime,
            )
            outside_stats = _outline_ink_stats(
                outside_pixels,
                outside_layer.paths,
                runtime,
            )

        self.assertEqual(set(fill_medians), set(PlaneDepthRole))
        background = BACKGROUND_RGB.astype(float)
        plane_colors = {
            "normal": _hex_rgb("#2CB9A4"),
            "occluded": _hex_rgb("#6B7C93"),
        }
        plane_opacities = {"normal": 0.12, "occluded": 0.18}
        expected = {
            variant: _source_over(
                background,
                plane_colors[variant],
                plane_opacities[variant],
            )
            for variant in plane_colors
        }
        for role, median in fill_medians.items():
            variant = (
                "occluded"
                if role
                in {
                    PlaneDepthRole.BEHIND_SURFACE,
                    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                }
                else "normal"
            )
            np.testing.assert_allclose(median, expected[variant], atol=3.0)
        self.assertGreater(
            float(np.max(np.abs(expected["occluded"] - expected["normal"]))),
            6.0,
        )

        cone_color = _hex_rgb("#173753")
        cone_sheet_opacity = 1.0 - sqrt(1.0 - 0.13)
        expected_surface: dict[PlaneDepthRole, np.ndarray] = {}
        for role in PlaneDepthRole:
            variant = (
                "occluded"
                if role
                in {
                    PlaneDepthRole.BEHIND_SURFACE,
                    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                }
                else "normal"
            )
            plane = plane_colors[variant]
            plane_opacity = plane_opacities[variant]
            value = background.copy()
            if role is PlaneDepthRole.BEHIND_SURFACE:
                value = _source_over(value, plane, plane_opacity)
                value = _source_over(value, cone_color, cone_sheet_opacity)
                value = _source_over(value, cone_color, cone_sheet_opacity)
            elif role is PlaneDepthRole.OUTSIDE_PROJECTION:
                value = _source_over(value, plane, plane_opacity)
            elif role is PlaneDepthRole.BETWEEN_SURFACE_SHEETS:
                value = _source_over(value, cone_color, cone_sheet_opacity)
                value = _source_over(value, plane, plane_opacity)
                value = _source_over(value, cone_color, cone_sheet_opacity)
            elif role is PlaneDepthRole.IN_FRONT_OF_SURFACE:
                value = _source_over(value, cone_color, cone_sheet_opacity)
                value = _source_over(value, cone_color, cone_sheet_opacity)
                value = _source_over(value, plane, plane_opacity)
            expected_surface[role] = value
        for role, median in surface_medians.items():
            np.testing.assert_allclose(
                median,
                expected_surface[role],
                atol=4.0,
            )

        self.assertIsInstance(behind_outline, VGroup)
        self.assertEqual(
            len(behind_outline.submobjects),
            len(behind_layer.paths),
        )
        self.assertTrue(
            all(
                isinstance(path, DashedVMobject)
                for path in behind_outline.submobjects
            )
        )
        self.assertGreater(behind_stats[0], 0.55)
        self.assertLess(behind_stats[0], 0.95)
        self.assertGreaterEqual(behind_stats[1], 8)
        self.assertGreaterEqual(behind_stats[2], 2)
        self.assertNotIsInstance(outside_outline, DashedVMobject)
        self.assertGreater(outside_stats[0], 0.95)

    def test_provider_surface_order_changes_the_real_cairo_composite(self) -> None:
        picture = compile_document(source_text=_source()).pictures[0]
        self.assertFalse(picture.unsupported)
        with tempconfig(
            {
                "pixel_width": 640,
                "pixel_height": 360,
                "frame_width": 14.222,
                "frame_height": 8.0,
            }
        ):
            runtime = instantiate_picture(picture).objects["dan:view:spatial"]
            certified = _capture(runtime)
            flattened = runtime.copy()
            flattened.set_z_index(picture.objects[0].z_index, family=True)
            flattened_pixels = _capture(flattened)

            surface_forward = runtime.copy()
            _hide_feature_strokes(surface_forward)
            surface_forward_pixels = _capture(surface_forward)

            reverse_picture = copy.deepcopy(picture)
            assert reverse_picture.projection_3d is not None
            reverse_matrix = reverse_picture.projection_3d.matrix
            reverse_picture.projection_3d.matrix = (
                reverse_matrix[0],
                reverse_matrix[1],
                tuple(-value for value in reverse_matrix[2]),
            )
            surface_reverse = instantiate_picture(reverse_picture).objects[
                "dan:view:spatial"
            ]
            _hide_feature_strokes(surface_reverse)
            surface_reverse_pixels = _capture(surface_reverse)

        background = np.asarray((16, 24, 32), dtype=np.int16)
        ink = np.any(
            np.abs(certified.astype(np.int16) - background) > 5,
            axis=2,
        )
        difference = np.max(
            np.abs(
                certified.astype(np.int16)
                - flattened_pixels.astype(np.int16)
            ),
            axis=2,
        )
        self.assertGreater(int(np.count_nonzero(ink)), 10_000)
        self.assertGreater(int(np.count_nonzero(difference > 8)), 500)
        self.assertGreater(int(np.max(difference)), 20)

        surface_difference = np.max(
            np.abs(
                surface_forward_pixels.astype(np.int16)
                - surface_reverse_pixels.astype(np.int16)
            ),
            axis=2,
        )
        self.assertGreater(int(np.count_nonzero(surface_difference > 2)), 10_000)

        frame = surface_forward.surface_layer_frame
        pair = frame.sphere_pair_evidence[0]
        roots = _paint_roots(surface_forward)
        far_center = roots[
            f"surface:{pair.farther_sphere_id}:teaching-fill"
        ].get_center()
        near_center = roots[
            f"surface:{pair.nearer_sphere_id}:teaching-fill"
        ].get_center()
        far_forward = _median_near_point(
            surface_forward_pixels,
            far_center,
            frame_width=14.222,
            frame_height=8.0,
        )
        far_reverse = _median_near_point(
            surface_reverse_pixels,
            far_center,
            frame_width=14.222,
            frame_height=8.0,
        )
        near_forward = _median_near_point(
            surface_forward_pixels,
            near_center,
            frame_width=14.222,
            frame_height=8.0,
        )
        near_reverse = _median_near_point(
            surface_reverse_pixels,
            near_center,
            frame_width=14.222,
            frame_height=8.0,
        )
        self.assertGreaterEqual(far_reverse[0] - far_forward[0], 2.0)
        self.assertGreaterEqual(near_forward[0] - near_reverse[0], 2.0)
        # The plane's role tint now contributes a different grey-blue base in
        # the two depth orders, so green is no longer a stable direction cue.
        # The sphere's coral channel must still swap in opposite directions at
        # the two authenticated centres, with a visible total RGB change.
        self.assertGreaterEqual(
            float(np.max(np.abs(far_reverse - far_forward))),
            4.0,
        )
        self.assertGreaterEqual(
            float(np.max(np.abs(near_forward - near_reverse))),
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
