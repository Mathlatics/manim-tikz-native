from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tikz_native import compile_asset, instantiate_picture
from tikz_native.planar_curves_3d import restore_planar_curve_geometry
from tikz_native.source_project import (
    SOURCE_PROJECT_SCHEMA_VERSION,
    build_project,
)


SOURCE_TEXT = r"""
\begin{tikzpicture}[3d view={40.4}{23.8}]
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (2,0,0);
  \coordinate (V) at (0,2,1);
  \DeclareSpacePlane{section-plane}{O/U/V};
  \DrawSpaceCircle[draw=red]{section-circle}{section-plane}{0,0}{1};
  \DrawSpaceEllipse[draw=blue]{section-ellipse}{section-plane}{0.5,-0.25}{2}{0.75};
\end{tikzpicture}
"""


class TikzPlanarCurves3DSourceProjectTests(unittest.TestCase):
    def test_source_project_provider_path_preserves_static_planar_semantics(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "figure.tex"
            source.write_text(SOURCE_TEXT, encoding="utf-8")
            manifest = root / "project.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schemaVersion": SOURCE_PROJECT_SCHEMA_VERSION,
                        "tikzSource": source.name,
                        "derivedOutput": ".derived",
                        "renderIntent": {
                            "paintPolicy": "diagrammatic",
                            "projection": {
                                "kind": "orthographic",
                                "direction": [1, -1, -1],
                            },
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            compiled = compile_asset(source)
            self.assertEqual(set(compiled.picture.planar_frames_3d), {"section-plane"})
            self.assertFalse(compiled.picture.unsupported)
            self.assertEqual(
                [(item.id, item.kind) for item in compiled.picture.objects],
                [
                    ("section-circle", "planar_circle_3d"),
                    ("section-ellipse", "planar_ellipse_3d"),
                ],
            )
            restored = [
                restore_planar_curve_geometry(
                    item.geometry,
                    expected_curve_id=item.id,
                )
                for item in compiled.picture.objects
            ]
            self.assertEqual(
                [geometry.curve.curve_id for geometry in restored],
                ["section-circle", "section-ellipse"],
            )
            self.assertTrue(
                all(geometry.frame.frame_id == "section-plane" for geometry in restored)
            )
            self.assertEqual(
                set(compiled.figure.objects),
                {"section-circle", "section-ellipse"},
            )

            runtime_figure = instantiate_picture(source_path=source)
            self.assertEqual(
                set(runtime_figure.objects),
                {"section-circle", "section-ellipse"},
            )
            self.assertEqual(
                [item.kind for item in runtime_figure.picture.objects],
                ["planar_circle_3d", "planar_ellipse_3d"],
            )
            self.assertEqual(
                set(runtime_figure.picture.planar_frames_3d),
                {"section-plane"},
            )

            result = build_project(manifest)
            self.assertIn("shape", result.built)
            shape_asset = json.loads(
                (root / ".derived" / "shape-asset.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [
                    (item["id"], item["kind"])
                    for item in shape_asset["object_index"]
                ],
                [
                    ("section-circle", "planar_circle_3d"),
                    ("section-ellipse", "planar_ellipse_3d"),
                ],
            )
            feature_counts = shape_asset["compatibility"]["feature_counts"]
            self.assertEqual(feature_counts["geometry.planar_frame_3d"], 1)
            self.assertEqual(feature_counts["object.planar_circle_3d"], 1)
            self.assertEqual(feature_counts["object.planar_ellipse_3d"], 1)
            coordinate_layer = next(
                layer
                for layer in shape_asset["animation_plan"]["layers"]
                if layer["name"] == "coordinate_frame"
            )
            self.assertEqual(
                coordinate_layer["object_ids"],
                ["section-circle", "section-ellipse"],
            )


if __name__ == "__main__":
    unittest.main()
