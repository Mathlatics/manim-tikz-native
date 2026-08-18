from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from manim import BLUE, RED, Dot, Polygon, Scene, tempconfig

from polyhedron_visibility import OcclusionBindingError, OcclusionStyle, ParallelProjection
from polyhedron_visibility.open_faces import (
    OpenFaceOcclusion3D,
    OpenFaceScene3D,
    OpenFaceSolverError,
)


class _FillOrderFixture:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.positions = {
            "F0": np.array((-1.0, -1.0, -2.0)),
            "F1": np.array((1.0, -1.0, -2.0)),
            "F2": np.array((1.0, 1.0, -2.0)),
            "F3": np.array((-1.0, 1.0, -2.0)),
            "N0": np.array((-1.0, -1.0, 2.0)),
            "N1": np.array((1.0, -1.0, 2.0)),
            "N2": np.array((1.0, 1.0, 2.0)),
            "N3": np.array((-1.0, 1.0, 2.0)),
        }
        self.far = Polygon(
            *(self.positions[f"F{index}"] for index in range(4)),
            color=BLUE,
            fill_opacity=0.45,
            stroke_opacity=0.0,
        ).set_z_index(4)
        self.near = Polygon(
            *(self.positions[f"N{index}"] for index in range(4)),
            color=RED,
            fill_opacity=0.55,
            stroke_opacity=0.0,
        ).set_z_index(5)
        scene.add(self.far, self.near)

        builder = OpenFaceScene3D("fill-order-manim")
        for vertex_id in sorted(self.positions):
            builder.vertex(vertex_id, lambda key=vertex_id: self.positions[key].copy())
        builder.face(
            "far",
            ("F0", "F1", "F2", "F3"),
            logical_surface_id="surface-far",
            source_mobject=self.far,
        )
        builder.face(
            "near",
            ("N0", "N1", "N2", "N3"),
            logical_surface_id="surface-near",
            source_mobject=self.near,
        )
        self.controller = builder.controller(
            scene,
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(max_projected_length=4.0),
        )

    def update_polygons(self) -> None:
        far_z = self.positions["F0"][2]
        near_z = self.positions["N0"][2]
        self.far.become(
            Polygon(
                *(self.positions[f"F{index}"] for index in range(4)),
                color=BLUE,
                fill_opacity=0.45,
                stroke_opacity=0.0,
            ).set_z_index(4)
        )
        self.near.become(
            Polygon(
                *(self.positions[f"N{index}"] for index in range(4)),
                color=RED,
                fill_opacity=0.55,
                stroke_opacity=0.0,
            ).set_z_index(5)
        )
        self.positions["F0"][2] = far_z
        self.positions["N0"][2] = near_z


class OpenFaceFillOrderManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "frame_rate": 8,
                "pixel_width": 320,
                "pixel_height": 180,
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_fixed_fill_proxies_follow_order_and_restore_sources(self) -> None:
        fixture = _FillOrderFixture(Scene())
        original_opacities = (
            fixture.far.get_fill_opacity(),
            fixture.near.get_fill_opacity(),
        )
        controller = fixture.controller.attach()

        self.assertIsInstance(controller, OpenFaceOcclusion3D)
        self.assertEqual(controller.last_frame.advisory_face_draw_order, ("far", "near"))
        self.assertEqual(fixture.far.get_fill_opacity(), 0.0)
        self.assertEqual(fixture.near.get_fill_opacity(), 0.0)
        identities = controller.face_fill_identities()
        layer = controller._face_fill_layer
        self.assertIsNotNone(layer)
        assert layer is not None
        self.assertEqual(float(layer.proxies["far"].z_index), 4.0)
        self.assertEqual(float(layer.proxies["near"].z_index), 5.0)
        fixture.scene.camera.reset()
        fixture.scene.camera.capture_mobjects(fixture.scene.mobjects)
        center_before = fixture.scene.camera.pixel_array[90, 160].copy()

        for key in ("F0", "F1", "F2", "F3"):
            fixture.positions[key][2] = 3.0
        for key in ("N0", "N1", "N2", "N3"):
            fixture.positions[key][2] = -3.0
        fixture.update_polygons()
        controller.update()

        self.assertEqual(controller.last_frame.advisory_face_draw_order, ("near", "far"))
        self.assertEqual(controller.face_fill_identities(), identities)
        self.assertEqual(float(layer.proxies["near"].z_index), 4.0)
        self.assertEqual(float(layer.proxies["far"].z_index), 5.0)
        fixture.scene.camera.reset()
        fixture.scene.camera.capture_mobjects(fixture.scene.mobjects)
        center_after = fixture.scene.camera.pixel_array[90, 160].copy()
        self.assertFalse(np.array_equal(center_before, center_after))

        controller.restore()
        self.assertEqual(
            (fixture.far.get_fill_opacity(), fixture.near.get_fill_opacity()),
            original_opacities,
        )
        self.assertNotIn(controller.overlay_root, fixture.scene.mobjects)

    def test_crossing_frame_keeps_last_good_fill_order(self) -> None:
        fixture = _FillOrderFixture(Scene())
        controller = fixture.controller.attach()
        layer = controller._face_fill_layer
        assert layer is not None
        last_good = controller.last_frame
        z_snapshot = {
            face_id: float(proxy.z_index) for face_id, proxy in layer.proxies.items()
        }
        for index, key in enumerate(("F0", "F1", "F2", "F3")):
            fixture.positions[key][2] = -1.0 if index in {0, 3} else 1.0
        for key in ("N0", "N1", "N2", "N3"):
            fixture.positions[key][2] = 0.0
        fixture.update_polygons()

        with self.assertRaises(OpenFaceSolverError):
            controller.update()
        self.assertIs(controller.last_frame, last_good)
        self.assertEqual(
            {face_id: float(proxy.z_index) for face_id, proxy in layer.proxies.items()},
            z_snapshot,
        )
        self.assertEqual(fixture.far.get_fill_opacity(), 0.0)
        self.assertEqual(fixture.near.get_fill_opacity(), 0.0)
        controller.restore()

    def test_unmanaged_drawable_inside_face_layer_band_fails_before_hiding(self) -> None:
        scene = Scene()
        fixture = _FillOrderFixture(scene)
        scene.add(Dot().set_z_index(4.5))
        before = (fixture.far.get_fill_opacity(), fixture.near.get_fill_opacity())

        with self.assertRaisesRegex(OcclusionBindingError, "face fill z band"):
            fixture.controller.attach()

        self.assertEqual(
            (fixture.far.get_fill_opacity(), fixture.near.get_fill_opacity()),
            before,
        )
        self.assertNotIn(fixture.controller.overlay_root, scene.mobjects)

    def test_real_cairo_render_uses_and_removes_fill_proxy_family(self) -> None:
        class FillScene(Scene):
            def construct(inner_self) -> None:
                fixture = _FillOrderFixture(inner_self)
                with fixture.controller.session():
                    inner_self.wait(0.1)
                    inner_self.proxy_ids = fixture.controller.face_fill_identities()
                inner_self.overlay_removed = fixture.controller.overlay_root not in inner_self.mobjects
                inner_self.fill_opacities = (
                    fixture.far.get_fill_opacity(),
                    fixture.near.get_fill_opacity(),
                )

        with TemporaryDirectory() as media_dir, tempconfig(
            {
                "renderer": "cairo",
                "media_dir": media_dir,
                "pixel_width": 160,
                "pixel_height": 90,
                "frame_rate": 5,
                "disable_caching": True,
                "write_to_movie": True,
                "save_last_frame": False,
            }
        ):
            scene = FillScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertTrue(scene.proxy_ids)
            self.assertTrue(scene.overlay_removed)
            self.assertEqual(scene.fill_opacities, (0.45, 0.55))


if __name__ == "__main__":
    unittest.main()
