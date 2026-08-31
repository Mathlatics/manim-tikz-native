from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tikz_native import (
    audit_document_compatibility,
    compile_document,
    load_subset_spec,
)


PROVIDER_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROVIDER_ROOT / "tests" / "fixtures" / "national_2026_18_tikz.tex"


class TikzNativeCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = compile_document(SOURCE)
        cls.report = audit_document_compatibility(cls.document)

    def test_subset_registry_has_unique_valid_features(self) -> None:
        subset = load_subset_spec()
        feature_ids = [feature["id"] for feature in subset["features"]]
        self.assertEqual(len(feature_ids), len(set(feature_ids)))
        self.assertEqual(
            {feature["level"] for feature in subset["features"]},
            {"A", "B", "C"},
        )

    def test_current_document_has_no_c_level_feature(self) -> None:
        self.assertEqual(self.report["static_status"], "pass")
        self.assertEqual(
            self.report["dynamic_status"],
            "native-relations-ready-explicit-driver-required",
        )
        self.assertEqual(
            self.report["encountered_feature_counts_by_level"],
            {"A": 23, "B": 4, "C": 0},
        )
        self.assertEqual(self.report["c_findings"], [])

    def test_current_document_b_level_findings_are_explicit(self) -> None:
        features = {
            feature["id"]: feature
            for feature in self.report["encountered_features"]
        }
        self.assertEqual(
            {
                feature_id
                for feature_id, feature in features.items()
                if feature["level"] == "B"
            },
            {
                "layout.baseline",
                "layout.trim_right",
                "scope.redundant_draw_none",
                "style.dash_keyword",
            },
        )
        self.assertEqual(
            features["relation.intersection.line_ellipse"]["count"],
            1,
        )
        self.assertIn(
            "显式选择主动对象、运动参数和有效区间",
            self.report["dynamic_requirements"],
        )

    def test_unsupported_clip_is_classified_as_c_and_blocks(self) -> None:
        source_text = r"""
\begin{tikzpicture}
  \clip (0,0) rectangle (1,1);
  \draw (0,0) -- (1,1);
\end{tikzpicture}
"""
        with TemporaryDirectory() as directory:
            source = Path(directory) / "clip.tex"
            source.write_text(source_text, encoding="utf-8")
            document = compile_document(source)
            report = audit_document_compatibility(document)

        self.assertEqual(report["static_status"], "blocked")
        self.assertEqual(report["dynamic_status"], "blocked")
        self.assertEqual(report["c_findings"][0]["feature"], "layout.clip")

    def test_explicit_static_three_dimensional_planar_curves_are_b_level_and_pass(
        self,
    ) -> None:
        document = compile_document(
            source_text=r"""
\begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{section-plane}{O/U/V};
  \DrawSpaceCircle{section-circle}{section-plane}{0,0}{1};
  \DrawSpaceEllipse{section-ellipse}{section-plane}{1,-0.5}{2}{0.75};
\end{tikzpicture}
"""
        )
        picture = document.pictures[0]
        self.assertFalse(picture.unsupported)

        report = audit_document_compatibility(document)
        features = {
            feature["id"]: feature for feature in report["encountered_features"]
        }
        self.assertEqual(report["static_status"], "pass")
        self.assertEqual(
            report["dynamic_status"],
            "native-relations-ready-explicit-driver-required",
        )
        self.assertEqual(report["pictures"][0]["overall_level"], "B")
        self.assertEqual(report["c_findings"], [])
        self.assertEqual(
            {
                feature_id
                for feature_id, feature in features.items()
                if feature["level"] == "B"
            },
            {
                "geometry.planar_frame_3d",
                "object.planar_circle_3d",
                "object.planar_ellipse_3d",
            },
        )
        for feature_id in (
            "geometry.planar_frame_3d",
            "object.planar_circle_3d",
            "object.planar_ellipse_3d",
        ):
            self.assertEqual(features[feature_id]["count"], 1)

    def test_legacy_three_dimensional_circle_and_ellipse_are_c_level_and_block(
        self,
    ) -> None:
        paths = (
            (r"\draw (O) circle (1);", "DrawSpaceCircle"),
            (
                r"\draw (O) ellipse [x radius=2,y radius=1];",
                "DrawSpaceEllipse",
            ),
        )
        for path, replacement in paths:
            with self.subTest(path=path):
                document = compile_document(
                    source_text=rf"""
\begin{{tikzpicture}}[space view={{(-0.35,-0.35),(1,0),(0,1)}}]
  \coordinate (O) at (0,0,0);
  {path}
\end{{tikzpicture}}
"""
                )
                report = audit_document_compatibility(document)

                self.assertEqual(report["static_status"], "blocked")
                self.assertEqual(report["dynamic_status"], "blocked")
                self.assertEqual(report["pictures"][0]["overall_level"], "C")
                self.assertEqual(len(report["c_findings"]), 1)
                finding = report["c_findings"][0]
                self.assertEqual(finding["level"], "C")
                self.assertEqual(finding["feature"], "syntax.unsupported")
                self.assertIn("explicit supporting plane", finding["message"])
                self.assertIn(replacement, finding["message"])

    def test_static_dandelin_diagram_is_explicitly_b_level(self) -> None:
        document = compile_document(
            source_text=r"""
\begin{tikzpicture}[3d view={38}{24}]
  \coordinate (A) at (0,0,0);
  \coordinate (Z) at (0,0,1);
  \coordinate (R) at (1,0,0);
  \coordinate (O) at (0,0,2);
  \coordinate (U) at (0,1,2);
  \coordinate (V) at (-0.8,0,2.6);
  \DeclareSpacePlane{cut}{O/U/V};
  \DeclareSpaceRightCone{cone}{A/Z/R}{30}{0/9}{open_single};
  \DeclareDandelinConstruction{dan}{cone}{cut};
  \DrawDandelinDiagram[view=spatial]{dan};
\end{tikzpicture}
"""
        )
        report = audit_document_compatibility(document)
        picture = report["pictures"][0]

        self.assertEqual(report["static_status"], "pass")
        self.assertEqual(picture["overall_level"], "B")
        self.assertEqual(report["c_findings"], [])
        self.assertEqual(
            {
                key: picture["feature_counts"][key]
                for key in (
                    "geometry.planar_frame_3d",
                    "geometry.space_right_cone_3d",
                    "geometry.dandelin_construction_3d",
                    "object.dandelin_diagram_static",
                )
            },
            {
                "geometry.planar_frame_3d": 1,
                "geometry.space_right_cone_3d": 1,
                "geometry.dandelin_construction_3d": 1,
                "object.dandelin_diagram_static": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
