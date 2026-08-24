from __future__ import annotations

from math import sqrt
import unittest
from unittest.mock import patch

import numpy as np
from manim import Mobject, Scene, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import SectionPlane, SphereSpec
from polyhedron_visibility.quadrics.manim import (
    QuadricManimError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.section_compositing import PlaneDepthRole


IDENTITY_VIEW = ParallelView.from_matrix(np.eye(3))


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=1,
        max_curves=1,
        max_fragments_per_curve=8,
        max_segments_per_fragment=128,
        max_surface_segments=256,
        max_dashes_per_fragment=32,
        max_projected_length=8.0,
        max_total_mobjects=1000,
    )


class QuadricSectionManimTests(unittest.TestCase):
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

    def test_section_groups_and_surface_sheets_use_one_managed_order(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        style = QuadricManimStyle(
            surface_fill_color="#315F91",
            surface_fill_opacity=0.76,
            section_plane_fill_opacity=0.15,
        )
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(sphere,),
            curves=(),
            projection=IDENTITY_VIEW,
            style=style,
            limits=_limits(),
            section_plane=plane,
            section_max_screen_error=0.12,
        ).attach()

        frame = controller.last_section_frame
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(
            set(controller.active_painter_z_indices),
            set(frame.draw_order),
        )
        ordered_z = [
            controller.active_painter_z_indices[item_id]
            for item_id in frame.draw_order
        ]
        self.assertEqual(ordered_z, sorted(ordered_z))
        self.assertTrue(
            {
                PlaneDepthRole.BEHIND_SURFACE,
                PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                PlaneDepthRole.IN_FRONT_OF_SURFACE,
            }.issubset(
                {item.role for item in frame.plane_fragments}
            )
        )
        expected_sheet_opacity = 1.0 - sqrt(1.0 - style.surface_fill_opacity)
        self.assertAlmostEqual(
            float(controller._section_slots[1].get_fill_opacity()),
            expected_sheet_opacity,
            places=12,
        )
        self.assertAlmostEqual(
            float(controller._section_slots[4].get_fill_opacity()),
            expected_sheet_opacity,
            places=12,
        )
        controller.restore()

    def test_dynamic_plane_preserves_slots_and_rolls_back_edge_on_failure(self) -> None:
        state = {"normal": (0.7, 0.0, 1.0), "u": (0.0, 1.0, 0.0)}

        def plane() -> SectionPlane:
            return SectionPlane(
                "cut",
                (0.0, 0.0, 0.0),
                state["normal"],
                u_axis=state["u"],
            )

        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=(),
            projection=IDENTITY_VIEW,
            limits=_limits(),
            section_plane=plane,
            section_max_screen_error=0.12,
        ).attach()
        identities = controller.slot_identities()
        state["normal"] = (0.25, 0.0, 1.0)
        with patch.object(
            Mobject,
            "__init__",
            side_effect=AssertionError("section update allocated a Mobject"),
        ):
            controller.update()
        self.assertEqual(controller.slot_identities(), identities)
        snapshot = controller.slot_snapshot()

        state["normal"] = (1.0, 0.0, 0.0)
        state["u"] = (0.0, 0.0, 1.0)
        with self.assertRaisesRegex(QuadricManimError, "projects edge-on"):
            controller.update()
        self.assertEqual(controller.slot_snapshot(), snapshot)
        controller.restore()


if __name__ == "__main__":
    unittest.main()
