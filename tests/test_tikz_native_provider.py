from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from manim import Dot, Line, MathTex, Polygon, VGroup

from tikz_native import (
    ASSET_SCHEMA,
    REQUEST_SCHEMA,
    TikzNativeProviderError,
    compile_asset,
    compile_document,
    instantiate_picture,
    provider_info,
)
from tikz_native.provider import ERROR_HASH_MISMATCH, sha256_file


FIXTURES = Path(__file__).with_name("fixtures")
SOURCE_2D = FIXTURES / "tikz_native_bridge_2d.tex"
SOURCE_3D = FIXTURES / "tikz_native_bridge_3d_fixed.tex"


class TikzNativeProviderTests(unittest.TestCase):
    def test_health_exposes_versioned_fixed_view_capability(self) -> None:
        info = provider_info()
        self.assertEqual(info["request_schema"], REQUEST_SCHEMA)
        self.assertTrue(info["revision"].startswith("source-sha256:"))
        self.assertTrue(info["capabilities"]["compile_2d"])
        self.assertTrue(info["capabilities"]["compile_3d_fixed_view"])
        self.assertTrue(info["capabilities"]["native_rig_2d_authoring_v1"])
        self.assertTrue(info["capabilities"]["native_manim_source_2d_v1"])
        self.assertTrue(info["capabilities"]["native_manim_source_3d_v1"])
        self.assertTrue(info["capabilities"]["native_manim_source_3d_v2"])
        self.assertTrue(info["capabilities"]["polyhedron_visibility_parallel_v1"])
        self.assertTrue(
            info["capabilities"]["tikz_polyhedron_visibility_3d_v1"]
        )
        self.assertTrue(
            info["capabilities"]["open_convex_face_visibility_parallel_v1"]
        )
        self.assertTrue(
            info["capabilities"]["tikz_open_face_visibility_3d_v1"]
        )
        self.assertFalse(info["capabilities"]["dynamic_camera_in_fixed_view"])

    def test_two_d_asset_uses_semantic_native_objects(self) -> None:
        compiled = compile_asset(
            SOURCE_2D,
            source_sha256=sha256_file(SOURCE_2D),
        )
        self.assertEqual(compiled.picture.dimension, 2)
        self.assertEqual(compiled.asset["schema"], ASSET_SCHEMA)
        self.assertEqual(compiled.asset["placement_mode"], "native_cm")
        self.assertEqual(
            set(compiled.figure.objects),
            {item.id for item in compiled.picture.objects},
        )
        self.assertEqual(
            set(compiled.figure.group._tikz_native_object_map),
            set(compiled.figure.objects),
        )
        classes = {type(item) for item in compiled.figure.objects.values()}
        self.assertTrue(classes & {Dot, Line, MathTex, Polygon, VGroup})
        self.assertGreater(compiled.figure.group.width, 0)
        self.assertGreater(compiled.figure.group.height, 0)

    def test_fixed_view_three_d_asset_uses_same_ordinary_scene_objects(self) -> None:
        compiled = compile_asset(
            SOURCE_3D,
            source_sha256=sha256_file(SOURCE_3D),
        )
        self.assertEqual(compiled.picture.dimension, 3)
        self.assertEqual(compiled.asset["dimension"], 3)
        projection = compiled.asset["projection"]
        self.assertEqual(projection["source"], "3d view")
        self.assertAlmostEqual(projection["azimuth_degrees"], 40.4)
        self.assertAlmostEqual(projection["elevation_degrees"], 23.8)
        self.assertEqual(
            set(compiled.figure.objects),
            {item.id for item in compiled.picture.objects},
        )
        self.assertFalse(
            any(type(item).__name__ == "Dot3D" for item in compiled.figure.objects.values())
        )
        self.assertTrue(
            all(
                isinstance(item, (Dot, Line, MathTex, Polygon, VGroup))
                for item in compiled.figure.objects.values()
            )
        )
        points = compiled.figure.group.get_all_points()
        self.assertGreater(len(points), 0)
        np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-12)

    def test_runtime_entry_can_instantiate_directly_from_frozen_source(self) -> None:
        figure = instantiate_picture(
            source_path=SOURCE_3D,
            entry_macro=None,
            picture_index=1,
            scene_unit_per_cm=1.0,
        )
        self.assertEqual(figure.picture.dimension, 3)
        self.assertEqual(set(figure.objects), set(figure.group._tikz_native_object_map))

    def test_runtime_entry_can_instantiate_from_inline_source_text(self) -> None:
        source_text = SOURCE_2D.read_text(encoding="utf-8")
        document = compile_document(source_text=source_text)
        figure = instantiate_picture(
            source_text=source_text,
            entry_macro=None,
            picture_index=1,
            scene_unit_per_cm=1.0,
        )
        self.assertEqual(document.source_path, "<inline>")
        self.assertEqual(figure.picture.dimension, 2)
        self.assertEqual(
            set(figure.objects),
            {item.id for item in document.pictures[0].objects},
        )

    def test_source_hash_mismatch_is_a_typed_error(self) -> None:
        with self.assertRaises(TikzNativeProviderError) as context:
            compile_asset(SOURCE_2D, source_sha256="0" * 64)
        self.assertEqual(context.exception.code, ERROR_HASH_MISMATCH)
        self.assertEqual(context.exception.phase, "verify_input")


if __name__ == "__main__":
    unittest.main()
