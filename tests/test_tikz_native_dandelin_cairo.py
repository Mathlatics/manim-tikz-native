from __future__ import annotations

import copy
import unittest

import numpy as np
from manim import Scene, tempconfig

from tikz_native import compile_document
from tikz_native.provider import instantiate_picture


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


class TikzNativeDandelinCairoTests(unittest.TestCase):
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
        self.assertGreaterEqual(far_forward[1] - far_reverse[1], 2.0)
        self.assertGreaterEqual(near_forward[0] - near_reverse[0], 2.0)
        self.assertGreaterEqual(near_reverse[1] - near_forward[1], 2.0)


if __name__ == "__main__":
    unittest.main()
