from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from manim import Scene, tempconfig
import numpy as np
from PIL import Image

from examples.classroom_cone_sections.classroom_cone_sections import (
    CAP_CONTACT_PROGRESS,
    CAP_FIRST_VISIBLE_PROGRESS,
    PARABOLA_PROGRESS,
    PROJECTION_OBLIQUE_PROGRESS,
    CapChordTopologyLesson,
    ClosedVsOpenConeLesson,
    ConicFamilyTransitionLesson,
    HiddenCurvePoliciesLesson,
    ProjectionDegenerationLesson,
    build_classroom_state,
    classroom_lesson_specs,
    classroom_projection_view,
)
from polyhedron_visibility.quadrics.authoring import QuadricSection3D


ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "examples" / "classroom_cone_sections" / "gallery"
MANIFEST = GALLERY / "manifest.json"
SCENE_SOURCE = (
    ROOT
    / "examples"
    / "classroom_cone_sections"
    / "classroom_cone_sections.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ClassroomConeSectionMetadataTests(unittest.TestCase):
    def test_gallery_contract_contains_exactly_five_complete_lessons(self) -> None:
        specs = classroom_lesson_specs()
        self.assertEqual(len(specs), 5)
        self.assertEqual(len({item.lesson_id for item in specs}), 5)
        self.assertEqual(len({item.scene_name for item in specs}), 5)
        self.assertEqual(
            {item.scene_name for item in specs},
            {
                ConicFamilyTransitionLesson.__name__,
                ClosedVsOpenConeLesson.__name__,
                HiddenCurvePoliciesLesson.__name__,
                ProjectionDegenerationLesson.__name__,
                CapChordTopologyLesson.__name__,
            },
        )
        for spec in specs:
            self.assertTrue(spec.title.strip())
            self.assertGreaterEqual(len(spec.parameters), 3)
            self.assertTrue(spec.conclusion.strip())
            self.assertGreaterEqual(len(spec.teacher_prompts), 3)
            self.assertEqual(len(spec.keyframes), 3)
            progresses = [item.progress for item in spec.keyframes]
            self.assertEqual(progresses, sorted(progresses))
            self.assertEqual(progresses[0], 0.0)
            self.assertEqual(progresses[-1], 1.0)
            self.assertTrue(all(item.label for item in spec.keyframes))
            self.assertTrue(all(item.teaching_point for item in spec.keyframes))

    def test_analytic_teaching_stops_are_in_the_expected_order(self) -> None:
        self.assertGreater(PARABOLA_PROGRESS, 0.0)
        self.assertLess(PARABOLA_PROGRESS, 1.0)
        self.assertGreater(CAP_CONTACT_PROGRESS, 0.0)
        self.assertLess(CAP_CONTACT_PROGRESS, 1.0)
        self.assertGreater(CAP_FIRST_VISIBLE_PROGRESS, CAP_CONTACT_PROGRESS)
        self.assertLess(CAP_FIRST_VISIBLE_PROGRESS, 1.0)
        self.assertGreater(PROJECTION_OBLIQUE_PROGRESS, 0.0)
        self.assertLess(PROJECTION_OBLIQUE_PROGRESS, 1.0)

    def test_projection_path_stays_certified_and_ends_at_rank_one_rim(self) -> None:
        radial_basis = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
        ratios = []
        for progress in np.linspace(0.0, 1.0, 21):
            view = classroom_projection_view(float(progress))
            matrix = view.matrix
            self.assertEqual(matrix.shape, (3, 3))
            self.assertGreater(abs(float(np.linalg.det(matrix))), 1.0e-8)
            singular_values = np.linalg.svd(
                matrix[:2] @ radial_basis, compute_uv=False
            )
            ratios.append(float(singular_values[-1] / singular_values[0]))
        self.assertGreater(ratios[0], 0.2)
        self.assertGreater(ratios[-2], 0.0)
        self.assertLess(ratios[-1], 1.0e-12)


class ClassroomConeSectionIdentityTests(unittest.TestCase):
    def test_every_lesson_keeps_scene_and_slot_identity_at_teaching_stops(self) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 8,
                "disable_caching": True,
            }
        ):
            for spec in classroom_lesson_specs():
                with self.subTest(lesson=spec.lesson_id):
                    scene = Scene()
                    state = build_classroom_state(
                        scene,
                        spec.lesson_id,
                        progress=spec.keyframes[0].progress,
                        with_labels=False,
                    )
                    try:
                        if spec.lesson_id != "projection_degeneration":
                            self.assertTrue(
                                all(
                                    isinstance(authoring, QuadricSection3D)
                                    for authoring in state.authorings
                                )
                            )
                        scene_ids = tuple(id(item) for item in scene.mobjects)
                        slot_ids = tuple(
                            (label, controller.slot_identities())
                            for label, controller in state.controllers
                        )
                        for keyframe in spec.keyframes[1:]:
                            state.set_progress(keyframe.progress)
                            self.assertEqual(
                                tuple(id(item) for item in scene.mobjects),
                                scene_ids,
                            )
                            self.assertEqual(
                                tuple(
                                    (label, controller.slot_identities())
                                    for label, controller in state.controllers
                                ),
                                slot_ids,
                            )
                            for _label, controller in state.controllers:
                                self.assertIsNotNone(controller.last_frame)
                                draw_order = (
                                    controller.last_boundary_frame.draw_order
                                    if controller.last_boundary_frame is not None
                                    else controller.last_frame.draw_order
                                )
                                self.assertTrue(draw_order)
                    finally:
                        state.restore()


class ClassroomConeSectionGalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_covers_all_lesson_keyframes(self) -> None:
        self.assertEqual(
            self.manifest["schema"],
            "manim-tikz-native-classroom-cone-sections/v1",
        )
        self.assertEqual(self.manifest["renderer"], "cairo")
        self.assertEqual(
            self.manifest["profile"],
            {"pixel_width": 960, "pixel_height": 540, "frame_rate": 8},
        )
        self.assertEqual(
            self.manifest["scene_source_sha256"], _sha256(SCENE_SOURCE)
        )
        expected = {
            spec.lesson_id: [item.label for item in spec.keyframes]
            for spec in classroom_lesson_specs()
        }
        actual = {
            item["lesson_id"]: [frame["label"] for frame in item["keyframes"]]
            for item in self.manifest["lessons"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(sum(len(item) for item in actual.values()), 15)

    def test_every_image_matches_its_manifest_digest_and_dimensions(self) -> None:
        paths = set()
        for lesson in self.manifest["lessons"]:
            frames = lesson["keyframes"]
            self.assertEqual(len(frames), 3)
            for frame in frames:
                path = GALLERY / frame["path"]
                self.assertNotIn(path, paths)
                paths.add(path)
                self.assertEqual(_sha256(path), frame["sha256"])
                with Image.open(path) as image:
                    self.assertEqual(image.size, (960, 540))
                self.assertTrue(frame["controllers"])
                for controller in frame["controllers"]:
                    self.assertTrue(controller["draw_order"])
            contact = lesson["contact_sheet"]
            path = GALLERY / contact["path"]
            self.assertNotIn(path, paths)
            paths.add(path)
            self.assertEqual(_sha256(path), contact["sha256"])
            with Image.open(path) as image:
                self.assertEqual(image.size, (1440, 300))
        self.assertEqual(len(paths), 20)

    def test_open_shell_side_view_owns_one_finite_trim_rim_source(self) -> None:
        lesson = next(
            item
            for item in self.manifest["lessons"]
            if item["lesson_id"] == "projection_degeneration"
        )
        controller = lesson["keyframes"][-1]["controllers"][0]
        self.assertEqual(
            controller["boundary_source_counts"].get("surface_trim_rim"), 1
        )


if __name__ == "__main__":
    unittest.main()
