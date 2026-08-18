from __future__ import annotations

from pathlib import Path
import unittest

from manim import Scene, tempconfig

from polyhedron_visibility import OcclusionStyle
from polyhedron_visibility.sections import ConvexSectionStyle, SectionPlane3D
from tikz_native.compiler import compile_document
from tikz_native.convex_section_3d_manim import (
    bind_picture_convex_section_3d,
)
from tikz_native.provider import instantiate_picture


ROOT = Path(__file__).resolve().parents[1]
CUBE = ROOT / "examples" / "convex_polyhedron_visibility_demo" / "cube.tex"


def _cube_with_probe_source() -> str:
    return CUBE.read_text(encoding="utf-8").replace(
        r"\end{tikzpicture}",
        "\n".join(
            (
                r"  \coordinate (X) at (-2,0,0);",
                r"  \coordinate (Y) at ( 2,0,0);",
                r"  \draw[edge,red] (X)--(Y);",
                r"\end{tikzpicture}",
            )
        ),
    )


class TikzNativeConvexSection3DManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig({"renderer": "cairo"})
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_real_tikz_cube_free_line_and_section_share_one_binding(self) -> None:
        picture = compile_document(source_text=_cube_with_probe_source()).pictures[0]
        figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        binding = bind_picture_convex_section_3d(
            scene,
            picture,
            figure,
            plane_provider=lambda: SectionPlane3D(
                "cut",
                (0, 0, 0),
                (1, 1, 1),
                3.0,
                3.0,
                u_axis=(1, -1, 0),
            ),
            source_style=OcclusionStyle(
                max_projected_length=10.0,
                dash_length=0.30,
                dash_gap=0.20,
            ),
            section_style=ConvexSectionStyle(
                max_boundary_projected_length=8.0,
                dash_length=0.30,
                dash_gap=0.20,
            ),
        ).attach()

        self.assertEqual(binding.last_frame.section.kind, "polygon")
        self.assertEqual(len(binding.last_frame.section.points), 6)
        free = [
            item
            for item in binding.last_frame.stroke_intersections
            if item.intersection.inside_parameter_interval == (0.25, 0.75)
        ]
        self.assertEqual(len(free), 1)
        source_edge = binding.last_frame.source_visibility.edge_map[
            free[0].source_edge_id
        ]
        self.assertTrue(
            any(
                interval.face_id == "section-plane:cut"
                for interval in source_edge.raw_intervals
            )
        )
        binding.restore()
        self.assertNotIn(binding.controller.overlay_root, scene.mobjects)

    def test_real_tikz_cube_auto_fits_a_tiny_authored_patch(self) -> None:
        picture = compile_document(source_text=_cube_with_probe_source()).pictures[0]
        figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        binding = bind_picture_convex_section_3d(
            scene,
            picture,
            figure,
            plane_provider=lambda: SectionPlane3D(
                "cut",
                (0, 0, 0),
                (1, 1, 1),
                0.01,
                0.01,
                u_axis=(1, -1, 0),
            ),
            source_style=OcclusionStyle(max_projected_length=10.0),
            section_style=ConvexSectionStyle(
                max_boundary_projected_length=8.0
            ),
            accurate_transparency=True,
        ).attach()
        self.assertEqual(binding.last_frame.section.kind, "polygon")
        self.assertIsNotNone(binding.controller.last_display_plane)
        assert binding.controller.last_display_plane is not None
        self.assertGreater(binding.controller.last_display_plane.half_width, 1.0)
        self.assertGreater(binding.controller.last_display_plane.half_height, 1.0)
        binding.restore()

    def test_real_tikz_cube_uses_exact_transparent_fragment_order(self) -> None:
        picture = compile_document(source_text=_cube_with_probe_source()).pictures[0]
        figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        face_ids = tuple(
            item.id for item in picture.objects if item.kind == "polygon"
        )
        original_opacities = {
            object_id: float(figure.objects[object_id].get_fill_opacity())
            for object_id in face_ids
        }
        binding = bind_picture_convex_section_3d(
            scene,
            picture,
            figure,
            plane_provider=lambda: SectionPlane3D(
                "cut",
                (0, 0, 0),
                (1, 1, 1),
                3.0,
                3.0,
                u_axis=(1, -1, 0),
            ),
            source_style=OcclusionStyle(
                max_projected_length=10.0,
                dash_length=0.30,
                dash_gap=0.20,
            ),
            section_style=ConvexSectionStyle(
                max_boundary_projected_length=8.0,
                dash_length=0.30,
                dash_gap=0.20,
            ),
            accurate_transparency=True,
        ).attach()

        frame = binding.last_transparent_compositing
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.section.kind, "polygon")
        self.assertEqual(len(frame.section.points), 6)
        self.assertTrue(
            all(
                float(figure.objects[object_id].get_fill_opacity()) == 0.0
                for object_id in face_ids
            )
        )
        z_indices = binding.controller.active_transparent_fragment_z_indices()
        for relation in frame.order_relations:
            self.assertLess(
                z_indices[relation.far_fragment_id],
                z_indices[relation.near_fragment_id],
            )
        binding.restore()
        self.assertEqual(
            {
                object_id: float(figure.objects[object_id].get_fill_opacity())
                for object_id in face_ids
            },
            original_opacities,
        )


if __name__ == "__main__":
    unittest.main()
