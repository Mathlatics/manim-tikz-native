"""Actual Cairo movie evidence for live Dandelin hidden-line updates."""

from __future__ import annotations

from math import pi
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from manim import Scene, ValueTracker, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics import (
    ConeModel,
    ConeSpec,
    DandelinOcclusion3D,
    SectionPlane,
    compute_dandelin_construction,
)


try:
    import cairo as _cairo  # noqa: F401
    from manim.renderer.cairo_renderer import (  # noqa: F401
        CairoRenderer as _CairoRenderer,
    )
except (ImportError, OSError):
    CAIRO_AVAILABLE = False
else:
    CAIRO_AVAILABLE = True


FIRST_VIEW = ParallelView.from_matrix(
    (
        (0.72, 0.0, 0.0),
        (0.0, 0.576, 0.432),
        (0.0, -0.6, 0.8),
    )
)
SECOND_VIEW = ParallelView.from_matrix(
    (
        (0.576, -0.432, 0.0),
        (0.216, 0.288, -0.6235382907247958),
        (0.5196152422706632, 0.6928203230275509, 0.5),
    )
)


def _construction():
    return compute_dandelin_construction(
        "cairo-live-dandelin",
        ConeSpec(
            "cairo-live-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 9.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_SINGLE,
        ),
        SectionPlane(
            "cairo-live-plane",
            (0.0, 0.0, 2.0),
            (0.6, 0.0, 0.8),
            u_axis=(0.0, 1.0, 0.0),
        ),
    )


@unittest.skipUnless(CAIRO_AVAILABLE, "Cairo renderer is unavailable")
class DandelinOcclusion3DCairoTests(unittest.TestCase):
    def test_real_cairo_movie_recomputes_hidden_lines_for_camera_change(
        self,
    ) -> None:
        class LiveDandelinScene(Scene):
            def construct(inner_self) -> None:
                phase = ValueTracker(0.0)
                controller = DandelinOcclusion3D(
                    inner_self,
                    construction=_construction(),
                    projection=lambda _scene: (
                        FIRST_VIEW
                        if phase.get_value() < 0.5
                        else SECOND_VIEW
                    ),
                ).attach()
                initial = controller.last_visibility_frame
                assert initial is not None
                inner_self.initial_json = initial.canonical_json()
                inner_self.initial_source_ids = tuple(
                    item.source_id for item in initial.strokes
                )
                inner_self.slot_ids = controller.slot_identities()
                inner_self.play(phase.animate.set_value(1.0), run_time=0.5)
                final = controller.last_visibility_frame
                assert final is not None
                inner_self.final_json = final.canonical_json()
                inner_self.final_source_ids = tuple(
                    item.source_id for item in final.strokes
                )
                inner_self.shared_committed_frame = (
                    final.compositing_frame is controller.last_boundary_frame
                )
                inner_self.fixed_slots = (
                    inner_self.slot_ids == controller.slot_identities()
                )
                frame = inner_self.renderer.get_frame()
                background = frame[0, 0, :3]
                inner_self.non_background_pixels = int(
                    np.count_nonzero(
                        np.any(frame[:, :, :3] != background, axis=2)
                    )
                )
                owned_ids = set(controller.slot_identities())
                controller.restore()
                scene_family_ids = {
                    id(member)
                    for root in inner_self.mobjects
                    for member in root.get_family()
                }
                inner_self.restored = (
                    not controller.attached
                    and not (owned_ids & scene_family_ids)
                )

        with (
            TemporaryDirectory() as media_dir,
            tempconfig(
                {
                    "renderer": "cairo",
                    "media_dir": media_dir,
                    "pixel_width": 240,
                    "pixel_height": 136,
                    "frame_rate": 4,
                    "disable_caching": True,
                    "write_to_movie": True,
                    "save_last_frame": False,
                }
            ),
        ):
            scene = LiveDandelinScene()
            scene.render()
            movie = Path(scene.renderer.file_writer.movie_file_path)
            self.assertTrue(movie.is_file())
            self.assertGreater(movie.stat().st_size, 0)
            self.assertNotEqual(scene.initial_json, scene.final_json)
            self.assertEqual(scene.initial_source_ids, scene.final_source_ids)
            self.assertTrue(scene.shared_committed_frame)
            self.assertTrue(scene.fixed_slots)
            self.assertGreater(scene.non_background_pixels, 100)
            self.assertTrue(scene.restored)


if __name__ == "__main__":
    unittest.main()
