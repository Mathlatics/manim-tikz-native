from __future__ import annotations

from pathlib import Path
import unittest

from tikz_native import compile_document
from tikz_native.animation import semantic_animation_layers
from tikz_native.provider import compile_asset
from tikz_native.regression import build_semantic_snapshot


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "planar_curves_3d" / "planar_curves_3d.tex"


class TikzPlanarCurves3DProductPathTests(unittest.TestCase):
    def test_semantic_layers_and_snapshot_cover_both_planar_curve_kinds(self) -> None:
        document = compile_document(SOURCE)
        picture = document.pictures[0]

        layers = {
            layer.name: layer.object_ids
            for layer in semantic_animation_layers(picture, include_empty=True)
        }
        self.assertEqual(
            layers["coordinate_frame"],
            ("circle-oblique", "ellipse-oblique"),
        )
        self.assertEqual(
            sum(len(object_ids) for object_ids in layers.values()),
            len(picture.objects),
        )

        snapshot = build_semantic_snapshot(document)
        self.assertEqual(snapshot["picture_count"], 3)
        self.assertEqual(snapshot["object_count"], 6)
        self.assertEqual(snapshot["object_kind_counts"]["planar_circle_3d"], 3)
        self.assertEqual(snapshot["object_kind_counts"]["planar_ellipse_3d"], 3)
        self.assertEqual(snapshot["animation_layer_counts"]["coordinate_frame"], 6)

    def test_provider_compiles_the_real_fixed_view_asset_end_to_end(self) -> None:
        compiled = compile_asset(SOURCE, picture_index=1)

        self.assertEqual(compiled.selected_compatibility["overall_level"], "B")
        self.assertEqual(compiled.selected_compatibility["static_status"], "pass")
        self.assertEqual(
            [item["id"] for item in compiled.asset["object_index"]],
            ["circle-oblique", "ellipse-oblique"],
        )
        coordinate_layer = next(
            item
            for item in compiled.animation_plan["layers"]
            if item["name"] == "coordinate_frame"
        )
        self.assertEqual(
            coordinate_layer["object_ids"],
            ["circle-oblique", "ellipse-oblique"],
        )
        self.assertEqual(set(compiled.figure.objects), {"circle-oblique", "ellipse-oblique"})
        self.assertGreater(compiled.asset["bounds"]["width_scene"], 0.0)
        self.assertGreater(compiled.asset["bounds"]["height_scene"], 0.0)


if __name__ == "__main__":
    unittest.main()
