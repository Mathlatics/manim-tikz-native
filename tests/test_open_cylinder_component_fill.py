from __future__ import annotations

import unittest

import numpy as np
from manim import Scene, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import (
    CylinderModel,
    CylinderSpec,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.projection import (
    _build_open_cylinder_projection_layers,
)


HEAD_ON_VIEW = ParallelView.from_matrix(np.eye(3))
OBLIQUE_VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)


def _cylinder(surface_id: str, model: CylinderModel) -> CylinderSpec:
    return CylinderSpec(
        surface_id,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
        (-0.5, 0.5),
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )


def _winding_number(
    point: np.ndarray,
    path: tuple[tuple[float, float], ...],
) -> int:
    result = 0
    values = tuple(np.asarray(item, dtype=float) for item in path)
    for start, end in zip(values, (*values[1:], values[0])):
        edge = end - start
        offset = point - start
        cross = float(edge[0] * offset[1] - edge[1] * offset[0])
        if start[1] <= point[1] < end[1] and cross > 0.0:
            result += 1
        elif end[1] <= point[1] < start[1] and cross < 0.0:
            result -= 1
    return result


def _signed_area(path: tuple[tuple[float, float], ...]) -> float:
    values = np.asarray(path, dtype=float)
    shifted = np.roll(values, -1, axis=0)
    return 0.5 * float(
        np.sum(values[:, 0] * shifted[:, 1] - values[:, 1] * shifted[:, 0])
    )


class OpenCylinderProjectionLayerTests(unittest.TestCase):
    def test_head_on_and_oblique_openings_cancel_the_outer_fill(self) -> None:
        cylinder = _cylinder("open-cylinder", CylinderModel.OPEN)
        terminal_by_id = {item.rim_id: item for item in cylinder.trim_rims}

        for label, view in (("head-on", HEAD_ON_VIEW), ("oblique", OBLIQUE_VIEW)):
            with self.subTest(view=label):
                layers = _build_open_cylinder_projection_layers(
                    cylinder,
                    view,
                    max_chord_error=0.005,
                )

                self.assertEqual(layers.opaque_cap_paths, ())
                self.assertEqual(layers.back.cap_paths, ())
                self.assertEqual(layers.front.cap_paths, ())
                self.assertEqual(len(layers.opaque_lateral_paths), 2)
                facing = dict(layers.terminal_front_facing_by_id)
                self.assertEqual(set(facing.values()), {False, True})

                opaque_outer, opaque_hole = layers.opaque_lateral_paths
                self.assertGreater(_signed_area(opaque_outer), 0.0)
                self.assertLess(_signed_area(opaque_hole), 0.0)
                center = np.asarray(view.matrix[:2], dtype=float) @ np.asarray(
                    cylinder.origin,
                    dtype=float,
                )
                self.assertEqual(
                    sum(
                        _winding_number(center, path)
                        for path in layers.opaque_lateral_paths
                    ),
                    0,
                )
                if label == "oblique":
                    self.assertGreater(
                        _signed_area(opaque_outer) + _signed_area(opaque_hole),
                        0.1,
                    )

                for is_front, sheet in (
                    (False, layers.back),
                    (True, layers.front),
                ):
                    self.assertEqual(len(sheet.lateral_paths), 2)
                    terminal_id = next(
                        item_id
                        for item_id, value in facing.items()
                        if value is is_front
                    )
                    terminal = terminal_by_id[terminal_id]
                    center = np.asarray(view.matrix[:2], dtype=float) @ np.asarray(
                        terminal.center,
                        dtype=float,
                    )
                    self.assertEqual(
                        sum(
                            _winding_number(center, path)
                            for path in sheet.lateral_paths
                        ),
                        0,
                    )


class OpenCylinderManimBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 6,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    @staticmethod
    def _controller(surface: CylinderSpec, view: ParallelView) -> QuadricOcclusion3D:
        return QuadricOcclusion3D(
            Scene(),
            surfaces=(surface,),
            curves=(),
            projection=view,
            style=QuadricManimStyle(
                surface_fill_color="#315F91",
                surface_fill_opacity=1.0,
                surface_stroke_opacity=0.0,
                # These palettes remain cone-only; the open cylinder must use
                # its ordinary surface color while still using component masks.
                cone_lateral_fill_colors=("#FF0000",),
                cone_cap_fill_colors=("#00FF00",),
            ),
            max_chord_error=0.005,
        ).attach()

    def test_open_cylinder_always_prepares_two_open_side_sheets(self) -> None:
        for label, view in (("head-on", HEAD_ON_VIEW), ("oblique", OBLIQUE_VIEW)):
            with self.subTest(view=label):
                controller = self._controller(
                    _cylinder(f"open-{label}", CylinderModel.OPEN),
                    view,
                )
                try:
                    prepared = controller.prepare().numeric.surfaces[0].cone_fill
                    self.assertIsNotNone(prepared)
                    assert prepared is not None
                    self.assertFalse(prepared.use_cone_palette)
                    self.assertEqual(prepared.back_cap_paths, ())
                    self.assertEqual(prepared.front_cap_paths, ())
                    self.assertEqual(len(prepared.back_lateral_paths), 2)
                    self.assertEqual(len(prepared.front_lateral_paths), 2)
                    slot = controller._surface_paint_slots[0]
                    self.assertGreater(len(slot.back_lateral.points), 0)
                    self.assertGreater(len(slot.front_lateral.points), 0)
                    self.assertEqual(len(slot.back_cap.points), 0)
                    self.assertEqual(len(slot.front_cap.points), 0)
                finally:
                    controller.restore()

    def test_closed_cylinder_keeps_the_historical_single_proxy_fill(self) -> None:
        controller = self._controller(
            _cylinder("closed-cylinder", CylinderModel.CLOSED),
            HEAD_ON_VIEW,
        )
        try:
            prepared = controller.prepare().numeric.surfaces[0]
            self.assertIsNone(prepared.cone_fill)
            slot = controller._surface_paint_slots[0]
            self.assertGreater(len(slot.base.points), 0)
            self.assertTrue(all(len(item.points) == 0 for item in slot.components))
        finally:
            controller.restore()

    def test_head_on_cairo_keeps_open_center_clear_and_closed_center_filled(
        self,
    ) -> None:
        def center_and_background_pixels(
            model: CylinderModel,
        ) -> tuple[np.ndarray, np.ndarray]:
            controller = self._controller(
                _cylinder(f"pixel-{model.value}", model),
                HEAD_ON_VIEW,
            )
            try:
                scene = controller.scene
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                pixels = scene.camera.pixel_array[:, :, :3]
                return (
                    pixels[pixels.shape[0] // 2, pixels.shape[1] // 2].astype(int),
                    pixels[0, 0].astype(int),
                )
            finally:
                controller.restore()

        open_center, open_background = center_and_background_pixels(
            CylinderModel.OPEN
        )
        closed_center, closed_background = center_and_background_pixels(
            CylinderModel.CLOSED
        )
        self.assertLess(
            float(np.linalg.norm(open_center - open_background)),
            5.0,
        )
        self.assertGreater(
            float(np.linalg.norm(closed_center - closed_background)),
            50.0,
        )


if __name__ == "__main__":
    unittest.main()
