from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from manim import Scene, tempconfig

from polyhedron_visibility import OcclusionStyle, canonical_trace_json
from tikz_native.compiler import compile_document
from tikz_native.polyhedron_visibility_3d_manim import (
    TikzNativeVisibility3DManimError,
    _canonical_position_provider,
    bind_picture_visibility_3d,
)
from tikz_native.polyhedron_visibility_3d_adapter import adapt_picture_visibility_3d
from tikz_native.provider import instantiate_picture


ROOT = Path(__file__).resolve().parents[1]
CUBE = ROOT / "examples" / "convex_polyhedron_visibility_demo" / "cube.tex"
DIHEDRAL = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"
ALIAS_SOURCE = r"""
\begin{tikzpicture}[3d view={40}{24}]
  \coordinate (A) at (-1,-1,1);
  \coordinate (B) at (1,-1,1);
  \coordinate (C) at (1,1,1);
  \coordinate (D) at (-1,1,1);
  \coordinate (S) at (-2,0,0);
  \coordinate (S_alias) at (-2,0,0);
  \coordinate (E) at (2,0,0);
  \fill (A)--(B)--(C)--(D)--cycle;
  \draw (S_alias)--(E);
\end{tikzpicture}
"""


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
        current_positions = binding.controller.position_provider()
        assert binding.controller.display_point_provider is not None
        for stroke in binding.controller.model.strokes:
            source = binding.controller.stroke_bindings[stroke.source_edge_id]
            source.put_start_and_end_on(
                binding.controller.display_point_provider(
                    current_positions[stroke.vertex_ids[0]]
                ),
                binding.controller.display_point_provider(
                    current_positions[stroke.vertex_ids[1]]
                ),
            )
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

    def test_dashed_tikz_source_edge_fails_with_an_explicit_v1_contract_error(self) -> None:
        with TemporaryDirectory(prefix="tikz-visibility-dashed-") as temporary:
            source_path = Path(temporary) / "cube-dashed.tex"
            source_path.write_text(
                CUBE.read_text(encoding="utf-8").replace(
                    r"\draw[edge] (A)--(B);",
                    r"\draw[edge,dashed] (A)--(B);",
                    1,
                ),
                encoding="utf-8",
            )
            picture = compile_document(source_path).pictures[0]
            figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
            scene = Scene()
            scene.add(figure.group)

            with self.assertRaisesRegex(
                TikzNativeVisibility3DManimError,
                "continuous straight Line.*compound or dashed",
            ):
                bind_picture_visibility_3d(
                    scene,
                    picture,
                    figure,
                    style=OcclusionStyle(max_projected_length=10.0),
                )

    def test_dynamic_welded_alias_tolerance_scales_with_the_whole_model(self) -> None:
        picture = compile_document(source_text=ALIAS_SOURCE).pictures[0]
        analysis = adapt_picture_visibility_3d(
            picture,
            validation_mode="independent_convex_faces",
        )
        authored = {
            name: np.asarray(point, dtype=float)
            for name, point in picture.coordinates.items()
        }

        large = {name: point * 1.0e6 for name, point in authored.items()}
        large["S_alias"] = large["S_alias"] + np.asarray((1.0e-8, 0, 0))
        _canonical_position_provider(analysis, lambda: large)()

        tiny = {name: point * 1.0e-12 for name, point in authored.items()}
        tiny["S_alias"] = tiny["S_alias"] + np.asarray((5.0e-11, 0, 0))
        with self.assertRaisesRegex(
            TikzNativeVisibility3DManimError,
            "welded aliases.*disagree",
        ):
            _canonical_position_provider(analysis, lambda: tiny)()

if __name__ == "__main__":
    unittest.main()
