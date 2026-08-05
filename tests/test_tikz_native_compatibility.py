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


if __name__ == "__main__":
    unittest.main()
