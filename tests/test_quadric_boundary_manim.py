from __future__ import annotations

from math import pi
import unittest

from manim import Scene, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.surface_boundaries import GeneratorBoundarySpec
from polyhedron_visibility.visibility import VisibilityKind


VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)


def limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=4,
        max_curves=8,
        max_fragments_per_curve=32,
        max_segments_per_fragment=256,
        max_surface_segments=512,
        max_dashes_per_fragment=128,
        max_projected_length=16.0,
        max_total_mobjects=30000,
        max_boundary_sources=32,
    )


class UnifiedBoundaryManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 8,
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_sphere_uses_semantic_silhouette_with_stable_slots(self) -> None:
        scene = Scene()
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=(),
            projection=VIEW,
            paint_policy="physical",
            boundary_visibility_mode="unified",
            limits=limits(),
        ).attach()
        before = tuple(
            id(item)
            for source in controller._curve_slots.values()
            for item in source.root.get_family()
        )
        self.assertIsNotNone(controller.last_boundary_frame)
        self.assertIn(
            "boundary:sphere:silhouette",
            controller.allocated_boundary_ids,
        )
        frame = controller.last_boundary_frame
        assert frame is not None
        silhouette = [
            item
            for item in frame.fragments
            if item.source_id == "boundary:sphere:silhouette"
        ]
        self.assertTrue(silhouette)
        self.assertTrue(all(item.painted for item in silhouette))
        controller.update()
        after = tuple(
            id(item)
            for source in controller._curve_slots.values()
            for item in source.root.get_family()
        )
        self.assertEqual(before, after)
        controller.restore()

    def test_section_outline_has_policy_owned_solid_and_dash_fragments(self) -> None:
        cone = ConeSpec(
            "cone",
            (0.0, 0.0, -2.4),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, -0.4),
            (0.8, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        scene = Scene()
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(cone,),
            curves=(),
            projection=VIEW,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            boundary_visibility_mode="unified",
            section_plane=plane,
            limits=limits(),
            style=QuadricManimStyle(surface_fill_opacity=0.65),
        ).attach()
        frame = controller.last_boundary_frame
        section = controller.last_section_frame
        assert frame is not None and section is not None
        outline = [
            item
            for item in frame.fragments
            if item.source_id.startswith("boundary:plane:cut:edge:")
        ]
        self.assertTrue(
            any(item.visibility_kind is VisibilityKind.VISIBLE for item in outline)
        )
        self.assertTrue(
            any(item.visibility_kind is VisibilityKind.HIDDEN for item in outline)
        )
        hidden = [
            item for item in outline
            if item.visibility_kind is VisibilityKind.HIDDEN and item.painted
        ]
        self.assertTrue(hidden)
        for item in hidden:
            if item.depth_role == "behind_surface":
                self.assertLess(
                    frame.draw_order.index(item.item_id),
                    frame.draw_order.index(section.paint_items.surface_back),
                )
            elif item.depth_role == "between_surface_sheets":
                self.assertLess(
                    frame.draw_order.index(item.item_id),
                    frame.draw_order.index(section.paint_items.surface_front),
                )
        controller.restore()

    def test_cylinder_cap_rims_and_authored_generator_share_fixed_pool(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
            1.0,
            (0.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        scene = Scene()
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(cylinder,),
            curves=(),
            projection=VIEW,
            paint_policy="diagrammatic",
            boundary_visibility_mode="unified",
            generator_boundaries=(
                GeneratorBoundarySpec("generator:teaching", "cylinder", 0.4),
            ),
            limits=limits(),
        ).attach()
        frame = controller.last_boundary_frame
        assert frame is not None
        ids = {item.source_id for item in frame.sources}
        self.assertIn("boundary:cylinder:cap_min:rim", ids)
        self.assertIn("boundary:cylinder:cap_max:rim", ids)
        self.assertIn("generator:teaching", ids)
        self.assertTrue(
            any(
                item.source_id.endswith(":rim")
                and item.visibility_kind is VisibilityKind.HIDDEN
                and item.painted
                for item in frame.fragments
            )
        )
        controller.restore()


if __name__ == "__main__":
    unittest.main()
