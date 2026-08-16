from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from manim import Scene, tempconfig

from polyhedron_visibility import OcclusionStyle, canonical_trace_json
from tikz_native.compiler import compile_document
from tikz_native.polyhedron_visibility_3d_manim import (
    TikzNativeVisibility3DManimError,
    bind_picture_visibility_3d,
)
from tikz_native.provider import instantiate_picture


ROOT = Path(__file__).resolve().parents[1]
CUBE = ROOT / "examples" / "convex_polyhedron_visibility_demo" / "cube.tex"
DIHEDRAL = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"


class TikzNativePolyhedronVisibility3DManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig({"renderer": "cairo"})
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_real_tikz_cube_attaches_with_the_same_entry_trace_and_restores(self) -> None:
        picture = compile_document(CUBE).pictures[0]
        figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
        figure.group.scale(0.73).rotate(0.18).shift((1.2, -0.4, 0.0))
        scene = Scene()
        scene.add(figure.group)
        source_opacities = {
            binding.source_edge_id: float(
                figure.objects[binding.object_ids[0]].get_stroke_opacity()
            )
            for binding in bind_picture_visibility_3d(
                scene,
                picture,
                figure,
                style=OcclusionStyle(max_projected_length=8.0),
            ).analysis.stroke_bindings
        }
        binding = bind_picture_visibility_3d(
            scene,
            picture,
            figure,
            style=OcclusionStyle(max_projected_length=8.0),
        )
        self.assertTrue(binding.controller.require_closed_convex_manifold)

        binding.attach()
        self.assertEqual(
            canonical_trace_json(binding.last_frame),
            canonical_trace_json(binding.analysis.entry_trace),
        )
        for stroke in binding.analysis.stroke_bindings:
            self.assertEqual(
                float(figure.objects[stroke.object_ids[0]].get_stroke_opacity()),
                0.0,
            )
        self.assertGreater(binding.controller.overlay_root.get_all_points().size, 0)

        binding.restore()
        for stroke in binding.analysis.stroke_bindings:
            self.assertAlmostEqual(
                float(figure.objects[stroke.object_ids[0]].get_stroke_opacity()),
                source_opacities[stroke.source_edge_id],
            )

    def test_dynamic_coordinates_and_projection_are_recomputed_without_topology_guessing(self) -> None:
        picture = compile_document(CUBE).pictures[0]
        figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        coordinates = {
            name: np.asarray(point, dtype=float)
            for name, point in picture.coordinates.items()
        }
        angle = 0.0

        def coordinate_provider():
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.asarray(
                ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
            )
            return {name: rotation @ point for name, point in coordinates.items()}

        binding = bind_picture_visibility_3d(
            scene,
            picture,
            figure,
            coordinate_provider=coordinate_provider,
            style=OcclusionStyle(max_projected_length=8.0),
        ).attach()
        before = binding.controller.slot_snapshot()
        angle = 0.31
        binding.update()
        self.assertNotEqual(binding.controller.slot_snapshot(), before)
        binding.restore()

    def test_open_dihedral_is_not_silently_promoted_to_closed_polyhedron(self) -> None:
        picture = compile_document(DIHEDRAL).pictures[0]
        figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        with self.assertRaisesRegex(Exception, "OPEN_FACE_SYSTEM"):
            bind_picture_visibility_3d(
                scene,
                picture,
                figure,
                style=OcclusionStyle(max_projected_length=10.0),
            )

        with self.assertRaisesRegex(
            TikzNativeVisibility3DManimError,
            "one complete source Mobject",
        ):
            bind_picture_visibility_3d(
                scene,
                picture,
                figure,
                validation_mode="independent_convex_faces",
                style=OcclusionStyle(max_projected_length=10.0),
            )


if __name__ == "__main__":
    unittest.main()
