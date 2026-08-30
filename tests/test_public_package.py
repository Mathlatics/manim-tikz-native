from __future__ import annotations

from importlib import resources
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

import tikz_native
from polyhedron_visibility import OcclusionScene3D
from polyhedron_visibility.open_faces import OpenFaceScene3D
from polyhedron_visibility.quadrics import (
    ConeSpec,
    CylinderSpec,
    QUADRIC_FINAL_PROFILE,
    QUADRIC_PREVIEW_PROFILE,
    QuadricCapacityPlanner,
    QuadricRenderProfile,
    QuadricSectionAction,
    QuadricSectionBoundary,
    QuadricSection3D,
    QuadricSectionRig,
    QuadricSectionRigError,
    QuadricSectionTransition3D,
    SectionState,
    SectionPlane,
    SectionTimeline,
    SectionTimelineTransitionPlan,
    SectionTransitionPlan,
    SphereSpec,
    build_section_transition_plan,
    compile_section_timeline,
    compute_quadric_section,
    compute_quadric_section_boundary,
    compute_quadric_section_boundary_curves,
    section_cap_chord_curve_ids,
)
from tikz_native.manim_renderer import DEFAULT_TEX_TEMPLATE


ROOT = Path(__file__).resolve().parents[1]


class PublicPackageTests(unittest.TestCase):
    def test_distribution_and_runtime_versions_match(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["name"], "manim-tikz-native")
        self.assertEqual(metadata["project"]["version"], tikz_native.__version__)
        self.assertEqual(metadata["project"]["requires-python"], ">=3.11")

    def test_public_authoring_and_tikz_entry_points_import(self) -> None:
        self.assertEqual(OcclusionScene3D.__name__, "OcclusionScene3D")
        self.assertEqual(OpenFaceScene3D.__name__, "OpenFaceScene3D")
        for name in (
            "NativeManimRenderer",
            "NativeManim3DRenderer",
            "adapt_picture_visibility_3d",
            "adapt_picture_open_face_visibility_3d",
            "bind_picture_visibility_3d",
            "bind_picture_open_face_visibility_3d",
            "bake_open_face_static_entry_3d",
            "generate_native_manim_source_3d_v3",
            "ParallelCameraSafeFrame",
            "ParallelCameraShot",
            "ParallelCameraShotSequence",
            "ParallelCameraTargetFollowController",
            "canonical_parallel_camera_shot_sequence_json",
            "fit_points_to_parallel_camera_state",
            "play_parallel_camera_shot",
            "play_parallel_camera_shot_sequence",
            "play_parallel_section_sequence",
            "compile_parallel_section_sequence_from_shots",
            "parallel_section_frame_grid",
            "parallel_section_preflight_gate",
            "section_bank_frame_participant",
            "section_display_frame_participant",
            "section_plane_patch_participant",
            "section_painter_order_participant",
        ):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(tikz_native, name)))

    def test_renderer_neutral_quadric_entry_points_import(self) -> None:
        self.assertEqual(SphereSpec.__name__, "SphereSpec")
        self.assertEqual(CylinderSpec.__name__, "CylinderSpec")
        self.assertEqual(ConeSpec.__name__, "ConeSpec")
        self.assertEqual(SectionPlane.__name__, "SectionPlane")
        self.assertEqual(SectionTransitionPlan.__name__, "SectionTransitionPlan")
        self.assertEqual(SectionTimeline.__name__, "SectionTimeline")
        self.assertEqual(
            SectionTimelineTransitionPlan.__name__,
            "SectionTimelineTransitionPlan",
        )
        self.assertEqual(QuadricSectionBoundary.__name__, "QuadricSectionBoundary")
        self.assertEqual(QuadricSection3D.__name__, "QuadricSection3D")
        self.assertEqual(QuadricSectionRig.__name__, "QuadricSectionRig")
        self.assertEqual(QuadricSectionAction.__name__, "QuadricSectionAction")
        self.assertEqual(QuadricSectionRigError.__name__, "QuadricSectionRigError")
        self.assertEqual(SectionState.__name__, "SectionState")
        self.assertEqual(QuadricCapacityPlanner.__name__, "QuadricCapacityPlanner")
        self.assertIsInstance(QUADRIC_PREVIEW_PROFILE, QuadricRenderProfile)
        self.assertIsInstance(QUADRIC_FINAL_PROFILE, QuadricRenderProfile)
        self.assertEqual(
            QuadricSectionTransition3D.__name__, "QuadricSectionTransition3D"
        )
        self.assertTrue(callable(build_section_transition_plan))
        self.assertTrue(callable(compile_section_timeline))
        self.assertTrue(callable(compute_quadric_section))
        self.assertTrue(callable(compute_quadric_section_boundary))
        self.assertTrue(callable(compute_quadric_section_boundary_curves))
        self.assertTrue(callable(section_cap_chord_curve_ids))

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from polyhedron_visibility.quadrics import "
                    "SectionTransitionPlan, build_section_transition_plan, "
                    "QuadricSectionBoundary, compute_quadric_section_boundary, "
                    "compute_quadric_section_boundary_curves, "
                    "section_cap_chord_curve_ids; "
                    "assert SectionTransitionPlan.__name__; "
                    "assert QuadricSectionBoundary.__name__; "
                    "assert callable(compute_quadric_section_boundary); "
                    "assert callable(build_section_transition_plan); "
                    "assert callable(compute_quadric_section_boundary_curves); "
                    "assert callable(section_cap_chord_curve_ids); "
                    "assert 'manim' not in sys.modules"
                ),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_default_tex_template_has_no_machine_specific_font_path(self) -> None:
        preamble = DEFAULT_TEX_TEMPLATE.preamble
        self.assertNotIn("/Users/", preamble)
        self.assertNotIn("Path=", preamble)
        self.assertIn("FandolSong", preamble)
        self.assertIn("latinmodern-math.otf", preamble)

    def test_wheel_package_data_is_declared_and_present(self) -> None:
        package_root = resources.files("tikz_native")
        self.assertTrue(package_root.joinpath("subset_v0_1.json").is_file())
        self.assertTrue(package_root.joinpath("examples/native_friendly_figure.tex").is_file())
        schemas = package_root.joinpath("schemas")
        self.assertTrue(schemas.joinpath("request-v1.schema.json").is_file())
        self.assertTrue(
            schemas.joinpath("geometry-rig-3d-source-v3-v1.schema.json").is_file()
        )
        self.assertTrue(
            schemas.joinpath("parallel-shot-sequence-v1.schema.json").is_file()
        )

    def test_public_tree_contains_no_personal_or_editor_checkout_paths(self) -> None:
        forbidden = (
            "/Users/",
            "Documents/" + "\u8bb2\u8bc4\u8bfe",
            "/tools/tikz-native-provider",
        )
        roots = (
            ROOT / "README.md",
            ROOT / "README.zh-CN.md",
            ROOT / "docs",
            ROOT / "examples",
            ROOT / "polyhedron_visibility",
            ROOT / "tikz_native",
        )
        suffixes = {".md", ".py", ".tex", ".json", ".toml", ".yml", ".yaml"}
        paths: list[Path] = []
        for item in roots:
            paths.extend([item] if item.is_file() else item.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in suffixes:
                continue
            payload = path.read_text(encoding="utf-8", errors="replace")
            for marker in forbidden:
                with self.subTest(path=path.relative_to(ROOT), marker=marker):
                    self.assertNotIn(marker, payload)


if __name__ == "__main__":
    unittest.main()
