"""Contracts for the fast/extended Cairo CI split and evidence bundle."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TIER_MANIFEST = ROOT / ".github" / "quadric-test-tiers.json"
BASELINE = ROOT / "tests" / "baselines" / "quadric-extended-acceptance-v1.json"
FAST_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
EXTENDED_WORKFLOW = (
    ROOT / ".github" / "workflows" / "extended-quadric-acceptance.yml"
)
GENERATOR = ROOT / "scripts" / "generate_quadric_extended_acceptance.py"
SCENES = ROOT / "examples" / "quadrics" / "extended_acceptance_demo.py"


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _rendering_test_ids() -> set[str]:
    result: set[str] = set()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for class_node in (
            item for item in tree.body if isinstance(item, ast.ClassDef)
        ):
            for method in (
                item
                for item in class_node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith("test_")
            ):
                method_source = ast.get_source_segment(source, method) or ""
                if ".render()" not in method_source:
                    continue
                result.add(f"{path.stem}.{class_node.name}.{method.name}")
    return result


class ExtendedQuadricCIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tiers = _load_json(TIER_MANIFEST)
        cls.baseline = _load_json(BASELINE)

    def test_every_direct_movie_render_is_deliberately_tiered(self) -> None:
        assigned = {
            identity
            for identities in self.tiers["tiers"].values()
            for identity in identities
        }
        self.assertTrue(_rendering_test_ids() <= assigned)
        self.assertFalse(
            set(self.tiers["tiers"]["cairo-smoke"])
            & set(self.tiers["tiers"]["extended-cairo"])
        )

    def test_fast_tier_keeps_small_animations_and_defers_full_frames(self) -> None:
        smoke = set(self.tiers["tiers"]["cairo-smoke"])
        extended = set(self.tiers["tiers"]["extended-cairo"])
        self.assertEqual(
            self.tiers["tier_environment"]["extended-cairo"],
            {
                "RUN_TIKZ_NATIVE_MOTION_3D_RENDER_TEST": "1",
                "RUN_TIKZ_NATIVE_MOTION_RENDER_TEST": "1",
            },
        )
        self.assertEqual(len(smoke), 5)
        self.assertIn(
            "test_composite_quadric_section_cairo."
            "CompositeQuadricSectionCairoTests."
            "test_two_nappes_share_one_plane_alpha_and_retain_both_section_branches",
            smoke,
        )
        self.assertIn(
            "test_quadric_section_authoring.QuadricSectionAuthoringTests."
            "test_real_cairo_animation_uses_the_facade_update_path",
            smoke,
        )
        self.assertIn(
            "test_quadric_transition_manim."
            "QuadricSectionTransitionControllerTests."
            "test_real_cairo_animation_visits_all_three_conic_families",
            smoke,
        )
        self.assertIn(
            "test_cone_models_cairo.ConeModelCairoTests."
            "test_complete_offset_point_48_demo_frame_has_no_open_shell_corridor_leak",
            extended,
        )
        self.assertIn(
            "test_quadric_section_cairo.QuadricSectionCairoRegressionTests."
            "test_high_resolution_role_boundaries_have_no_cairo_gaps",
            extended,
        )

    def test_baseline_covers_every_requested_acceptance_artifact(self) -> None:
        self.assertEqual(
            self.baseline["schema"],
            "manim-quadric-extended-acceptance-baseline/v1",
        )
        self.assertEqual(self.baseline["contract_id"], "quadric-section-v1")
        self.assertEqual(
            self.baseline["profile"],
            {"pixel_width": 960, "pixel_height": 540, "frame_rate": 8},
        )
        scenarios = {
            item["id"]: item for item in self.baseline["scenarios"]
        }
        self.assertEqual(
            set(scenarios),
            {
                "closed_open_comparison",
                "section_topology",
                "hidden_curve_policies",
                "side_view_trim_rim",
                "cap_chord_activation",
            },
        )
        for scenario in scenarios.values():
            self.assertTrue(scenario["video_scene"])
            self.assertTrue(scenario["keyframes"])
            self.assertEqual(
                len(scenario["keyframes"]), len(scenario["keyframe_labels"])
            )
            self.assertGreaterEqual(scenario["motion_samples"], 33)
            self.assertTrue(str(scenario["contact_sheet"]).endswith(".png"))
        self.assertFalse(self.baseline["pixel_policy"]["whole_image_hash"])

    def test_generator_and_scene_modules_are_importable_from_their_files(self) -> None:
        for path in (GENERATOR, SCENES):
            with self.subTest(path=path.name):
                self.assertIsNotNone(
                    importlib.util.spec_from_file_location(path.stem, path)
                )
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_workflows_run_exact_tiers_and_upload_only_the_evidence_dir(self) -> None:
        fast = FAST_WORKFLOW.read_text(encoding="utf-8")
        extended = EXTENDED_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "  push:\n    branches:\n      - main\n  pull_request:", fast
        )
        self.assertEqual(fast.count("actions/checkout@v7"), 2)
        self.assertEqual(fast.count("actions/setup-python@v7"), 2)
        self.assertEqual(extended.count("actions/checkout@v7"), 1)
        self.assertEqual(extended.count("actions/setup-python@v7"), 1)
        self.assertEqual(extended.count("actions/upload-artifact@v7"), 1)
        for obsolete in (
            "actions/checkout@v4",
            "actions/setup-python@v5",
            "actions/upload-artifact@v4",
        ):
            self.assertNotIn(obsolete, fast)
            self.assertNotIn(obsolete, extended)
        self.assertIn("scripts/run_ci_test_tier.py core", fast)
        self.assertIn("scripts/run_ci_test_tier.py cairo-smoke", fast)
        self.assertIn('python-version: ["3.11", "3.12"]', fast)
        self.assertIn("python -m build", fast)
        self.assertIn("workflow_dispatch:", extended)
        self.assertIn("schedule:", extended)
        self.assertIn("release:", extended)
        self.assertIn("scripts/run_ci_test_tier.py extended-cairo", extended)
        self.assertIn("--render-videos", extended)
        self.assertIn(
            "path: ${{ runner.temp }}/quadric-section-acceptance", extended
        )
        self.assertNotIn("path: .\n", extended)
        self.assertNotIn("include-hidden-files: true", extended)


if __name__ == "__main__":
    unittest.main()
