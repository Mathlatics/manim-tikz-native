from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from manim import Line, Polygon, Scene, VGroup, tempconfig

from polyhedron_visibility import OcclusionStyle, ParallelProjection
from polyhedron_visibility.depth_cue import DepthCuedAutoOcclusion3D

from examples.convex_sections.other_convex_solids_demo import (
    CONVEX_SOLIDS,
    _OtherSolidFixture,
)
from tests.test_face_depth_cue import FACES, VERTICES, cube_model, surface_edges


class _DepthCueFixture:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.model = cube_model()
        self.positions = {
            key: np.asarray(value, dtype=float) for key, value in VERTICES.items()
        }
        self.faces: dict[str, Polygon] = {}
        for index, (face_id, cycle) in enumerate(FACES.items()):
            face = Polygon(
                *(self.positions[item] for item in cycle),
                fill_color="#5B8FF9",
                fill_opacity=0.18,
                stroke_opacity=0.0,
            ).set_z_index(index)
            self.faces[face_id] = face
        self.lines: dict[str, Line] = {}
        for index, (start, end) in enumerate(surface_edges()):
            edge_id = f"edge.{start}.{end}"
            self.lines[edge_id] = Line(
                self.positions[start],
                self.positions[end],
                buff=0,
                color="#263238",
                stroke_width=3.0,
            ).set_z_index(20 + index)
        self.lines["probe.X.Y"] = Line(
            self.positions["X"],
            self.positions["Y"],
            buff=0,
            color="#D1495B",
            stroke_width=4.0,
        ).set_z_index(50)
        scene.add(VGroup(*self.faces.values(), *self.lines.values()))
        self.controller = DepthCuedAutoOcclusion3D(
            scene,
            self.model,
            position_provider=lambda: self.positions,
            stroke_bindings=self.lines,
            face_fill_bindings=self.faces,
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(
                max_projected_length=5.0,
                dash_length=0.20,
                dash_gap=0.14,
            ),
        )


class FaceDepthCueManimTests(unittest.TestCase):
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

    def test_face_proxies_shade_and_restore_without_reallocation(self) -> None:
        fixture = _DepthCueFixture(Scene())
        original_fill = {
            face_id: source.fill_rgbas.copy()
            for face_id, source in fixture.faces.items()
        }
        controller = fixture.controller.attach()
        identities = controller.face_fill_identities()
        self.assertTrue(identities)
        self.assertTrue(
            all(source.get_fill_opacity() == 0.0 for source in fixture.faces.values())
        )
        proxy_opacities = {
            face_id: float(proxy.get_fill_opacity())
            for face_id, proxy in controller._face_layer.proxies.items()
        }
        proxy_colors = {
            str(proxy.get_fill_color())
            for proxy in controller._face_layer.proxies.values()
        }
        self.assertGreater(max(proxy_opacities.values()), min(proxy_opacities.values()))
        self.assertGreater(
            max(proxy_opacities.values()) - min(proxy_opacities.values()),
            0.25,
        )
        self.assertGreater(len(proxy_colors), 1)
        proxy_rgbs = [
            np.asarray(proxy.get_fill_color().to_rgb(), dtype=float)
            for proxy in controller._face_layer.proxies.values()
        ]
        self.assertGreater(
            max(
                float(np.linalg.norm(first - second))
                for first in proxy_rgbs
                for second in proxy_rgbs
            ),
            0.25,
        )
        controller.update()
        self.assertEqual(controller.face_fill_identities(), identities)
        controller.restore()
        for face_id, source in fixture.faces.items():
            self.assertTrue(
                np.array_equal(source.fill_rgbas, original_fill[face_id])
            )
        self.assertNotIn(controller.overlay_root, fixture.scene.mobjects)

    def test_silhouette_width_changes_only_visible_stroke_style(self) -> None:
        fixture = _DepthCueFixture(Scene())
        controller = fixture.controller.attach()
        cue = controller.last_face_depth_cue
        assert cue is not None
        silhouette_widths: list[float] = []
        regular_widths: list[float] = []
        for edge_id, edge_cue in cue.edge_map.items():
            active = [
                line
                for line in controller._slots[edge_id].visible
                if float(line.get_stroke_opacity()) > 0
            ]
            if not active:
                continue
            target = silhouette_widths if edge_cue.is_silhouette else regular_widths
            target.extend(float(line.get_stroke_width()) for line in active)
        self.assertTrue(silhouette_widths)
        self.assertTrue(regular_widths)
        self.assertGreater(min(silhouette_widths), max(regular_widths))
        probe_style = controller._resolved_styles["probe.X.Y"]
        hidden_lines = [
            line
            for group in controller._slots["probe.X.Y"].hidden
            for line in group
            if float(line.get_stroke_opacity()) > 0
        ]
        self.assertTrue(hidden_lines)
        self.assertTrue(
            all(
                abs(float(line.get_stroke_width()) - probe_style.hidden_width)
                <= 1.0e-12
                for line in hidden_lines
            )
        )
        controller.restore()

    def test_invalid_face_update_keeps_last_good_proxy_state(self) -> None:
        fixture = _DepthCueFixture(Scene())
        controller = fixture.controller.attach()
        snapshot = {
            face_id: (
                proxy.get_all_points().copy(),
                proxy.fill_rgbas.copy(),
                float(proxy.z_index),
            )
            for face_id, proxy in controller._face_layer.proxies.items()
        }
        fixture.faces["front"].shift((0.25, 0.0, 0.0))
        with self.assertRaisesRegex(Exception, "no longer matches"):
            controller.update()
        for face_id, proxy in controller._face_layer.proxies.items():
            points, rgba, z_index = snapshot[face_id]
            self.assertTrue(np.array_equal(proxy.get_all_points(), points))
            self.assertTrue(np.array_equal(proxy.fill_rgbas, rgba))
            self.assertEqual(float(proxy.z_index), z_index)
        controller.restore()

    def test_face_source_with_its_own_outline_fails_before_mutation(self) -> None:
        fixture = _DepthCueFixture(Scene())
        fixture.faces["front"].set_stroke(
            color="#000000", width=2.0, opacity=1.0
        )
        before = {
            face_id: polygon.fill_rgbas.copy()
            for face_id, polygon in fixture.faces.items()
        }
        with self.assertRaisesRegex(Exception, "must be fill-only"):
            fixture.controller.attach()
        for face_id, polygon in fixture.faces.items():
            self.assertTrue(np.array_equal(polygon.fill_rgbas, before[face_id]))
        self.assertNotIn(fixture.controller.overlay_root, fixture.scene.mobjects)

    def test_convex_section_keeps_plane_style_above_depth_cued_faces(self) -> None:
        scene = Scene()
        fixture = _OtherSolidFixture(scene, CONVEX_SOLIDS["tetrahedron"])
        original = {
            face_id: polygon.fill_rgbas.copy()
            for face_id, polygon in fixture.face_polygons.items()
        }
        controller = fixture.controller.attach()
        self.assertTrue(controller.face_fill_identities())
        layer = controller._face_depth_layer
        assert layer is not None
        self.assertLess(
            max(float(proxy.z_index) for proxy in layer.proxies.values()),
            float(controller.plane_patch.z_index),
        )
        self.assertLess(
            float(controller.plane_patch.z_index),
            float(controller.section_fill.z_index),
        )
        self.assertEqual(str(controller.plane_patch.get_fill_color()), "#52B6A8")
        fixture.plane_offset.set_value(0.0)
        identities = controller.face_fill_identities()
        controller.update()
        self.assertEqual(controller.face_fill_identities(), identities)
        self.assertEqual(
            controller.last_sectioned_frame.section.kind,
            "polygon",
        )
        controller.restore()
        for face_id, polygon in fixture.face_polygons.items():
            self.assertTrue(np.array_equal(polygon.fill_rgbas, original[face_id]))

    def test_real_cairo_render_removes_depth_cue_family(self) -> None:
        class CueScene(Scene):
            def construct(inner_self) -> None:
                fixture = _DepthCueFixture(inner_self)
                with fixture.controller.session():
                    inner_self.wait(0.15)
                    inner_self.cue_ids = fixture.controller.face_fill_identities()
                inner_self.overlay_removed = (
                    fixture.controller.overlay_root not in inner_self.mobjects
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
            scene = CueScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertTrue(scene.cue_ids)
            self.assertTrue(scene.overlay_removed)


if __name__ == "__main__":
    unittest.main()
