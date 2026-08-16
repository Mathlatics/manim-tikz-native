from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from tikz_native.compiler import compile_document
from tikz_native.polyhedron_visibility_3d_adapter import (
    adapt_picture_visibility_3d,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "convex_polyhedron_visibility_demo" / "cube.tex"


class TikzNativeConvexPolyhedronVisibilityDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]
        cls.result = adapt_picture_visibility_3d(cls.picture)

    def test_real_tikz_cube_needs_no_per_edge_occlusion_relations(self) -> None:
        self.assertEqual(self.picture.dimension, 3)
        self.assertEqual(self.picture.occlusion_relations, [])
        self.assertEqual(len(self.result.model.vertices), 8)
        self.assertEqual(len(self.result.model.faces), 6)
        self.assertEqual(len(self.result.model.strokes), 12)
        self.assertEqual(len(self.result.stroke_bindings), 12)
        self.assertTrue(
            all(item.source_kind == "named_line" for item in self.result.stroke_bindings)
        )

    def test_every_cube_edge_has_exactly_two_incident_faces(self) -> None:
        for stroke in self.result.model.strokes:
            self.assertEqual(
                len(stroke.incident_face_ids),
                2,
                f"{stroke.source_edge_id} should be a closed-manifold edge",
            )

    def test_entry_trace_is_global_deterministic_and_versioned(self) -> None:
        self.assertEqual(
            self.result.entry_trace.schema,
            "manim-convex-polyhedron-visibility-trace/v1",
        )
        self.assertEqual(
            set(self.result.entry_trace.edge_map),
            {item.source_edge_id for item in self.result.model.strokes},
        )
        self.assertRegex(self.result.entry_trace_sha256, r"^[0-9a-f]{64}$")
        again = adapt_picture_visibility_3d(self.picture)
        self.assertEqual(again.result_sha256, self.result.result_sha256)
        self.assertEqual(again.entry_trace_sha256, self.result.entry_trace_sha256)

    def test_closed_face_orientation_accepts_the_supported_scale_range(self) -> None:
        expected_faces = {
            frozenset(face.vertex_ids) for face in self.result.model.faces
        }
        for factor in (1.0e-6, 1.0e6):
            with self.subTest(factor=factor):
                picture = deepcopy(self.picture)
                picture.coordinates = {
                    name: tuple(float(value) * factor for value in point)
                    for name, point in picture.coordinates.items()
                }
                result = adapt_picture_visibility_3d(picture)
                self.assertEqual(
                    {frozenset(face.vertex_ids) for face in result.model.faces},
                    expected_faces,
                )
                result.model.validate(require_closed_convex_manifold=True)


if __name__ == "__main__":
    unittest.main()
