from __future__ import annotations

from math import pi
import unittest

import numpy as np
from manim import Scene, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics import (
    ConeModel,
    ConeSpec,
    DandelinManimError,
    DandelinOcclusion3D,
    QuadricManimError,
    SectionPlane,
    compute_dandelin_construction,
)


VIEW = ParallelView.from_matrix(
    np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.8, 0.6),
            (0.0, -0.6, 0.8),
        ),
        dtype=float,
    )
)
ROTATED_VIEW = ParallelView.from_matrix(
    np.asarray(
        (
            (0.8, -0.6, 0.0),
            (0.3, 0.4, -0.8660254037844386),
            (0.5196152422706632, 0.6928203230275509, 0.5),
        ),
        dtype=float,
    )
)


def _construction():
    return compute_dandelin_construction(
        "live-dandelin",
        ConeSpec(
            "live-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 9.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_SINGLE,
        ),
        SectionPlane(
            "live-plane",
            (0.0, 0.0, 2.0),
            (0.6, 0.0, 0.8),
            u_axis=(0.0, 1.0, 0.0),
        ),
    )


class DandelinOcclusion3DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 6,
                "disable_caching": True,
                "write_to_movie": False,
                "save_last_frame": False,
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_live_camera_reuses_one_frame_and_fixed_slot_family(self) -> None:
        state = {"view": VIEW}
        scene = Scene()
        controller = DandelinOcclusion3D(
            scene,
            construction=_construction(),
            projection=lambda _scene: state["view"],
        ).attach()
        try:
            first = controller.last_visibility_frame
            self.assertIsNotNone(first)
            assert first is not None
            slot_ids = controller.slot_identities()
            first_json = first.canonical_json()
            first_source_ids = tuple(item.source_id for item in first.strokes)
            self.assertIs(first.compositing_frame, controller.last_boundary_frame)
            self.assertGreater(first.hidden_span_count, 0)
            self.assertIn(
                "plane_boundary",
                {item.role for item in first.strokes},
            )
            self.assertEqual(
                len(first.tangent_contacts),
                len(controller.construction.spheres),
            )

            state["view"] = ROTATED_VIEW
            controller.update()
            second = controller.last_visibility_frame
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(controller.slot_identities(), slot_ids)
            self.assertEqual(
                tuple(item.source_id for item in second.strokes),
                first_source_ids,
            )
            self.assertNotEqual(second.canonical_json(), first_json)
            self.assertIs(second.compositing_frame, controller.last_boundary_frame)
            self.assertTrue(
                set(second.compositing_frame.draw_order).issubset(
                    controller.active_painter_z_indices
                )
            )
        finally:
            controller.restore()
        self.assertEqual(scene.mobjects, [])
        self.assertFalse(controller.attached)
        self.assertIsNone(controller.last_visibility_frame)

    def test_failed_camera_update_restores_display_and_evidence(self) -> None:
        state: dict[str, object] = {"view": VIEW}
        controller = DandelinOcclusion3D(
            Scene(),
            construction=_construction(),
            projection=lambda _scene: state["view"],
        ).attach()
        try:
            previous_frame = controller.last_visibility_frame
            previous_slots = controller.slot_snapshot()
            previous_z = controller.active_painter_z_indices
            state["view"] = np.zeros((3, 3), dtype=float)
            with self.assertRaises((DandelinManimError, QuadricManimError)):
                controller.update()
            self.assertIs(controller.last_visibility_frame, previous_frame)
            self.assertEqual(controller.slot_snapshot(), previous_slots)
            self.assertEqual(controller.active_painter_z_indices, previous_z)
        finally:
            controller.restore()


if __name__ == "__main__":
    unittest.main()
