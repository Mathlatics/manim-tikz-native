from __future__ import annotations

import ast
import json
import math
import unittest
from pathlib import Path

import numpy as np

from tikz_native import compile_document
from tikz_native.motion_3d import Motion3DSpec, NativeMotion3DRuntime


PROVIDER_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROVIDER_ROOT / "examples" / "dihedral_fold_3d_demo"
SOURCE = DEMO_ROOT / "dihedral_fold.tex"
MOTION = DEMO_ROOT / "motion-3d.json"
SCENE = DEMO_ROOT / "scene.py"
README = DEMO_ROOT / "README.md"


class TikzNativeDihedralFold3DDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]
        cls.payload = json.loads(MOTION.read_text(encoding="utf-8"))
        cls.motion = Motion3DSpec.load(MOTION)

    def test_demo_is_portable_and_scene_is_valid_python(self) -> None:
        for path in (SOURCE, MOTION, SCENE, README):
            self.assertTrue(path.is_file(), path)
            self.assertNotIn("/Users/", path.read_text(encoding="utf-8"))
        ast.parse(SCENE.read_text(encoding="utf-8"), filename=str(SCENE))

    def test_tikz_compiles_to_one_supported_three_d_picture(self) -> None:
        picture = self.picture
        self.assertEqual(picture.dimension, 3)
        self.assertFalse(picture.unsupported)
        self.assertIsNotNone(picture.projection_3d)
        self.assertEqual(
            {
                "A",
                "B",
                "Alpha0",
                "Alpha1",
                "Beta0",
                "Beta1",
                "S",
                "E",
                "M",
                "N",
            },
            set(picture.coordinates),
        )
        self.assertEqual(
            picture.coordinate_dependencies["N"],
            {
                "operation": "projection",
                "line_start": "A",
                "point": "M",
                "line_end": "B",
            },
        )
        self.assertEqual(len(picture.hinge_relations), 1)
        self.assertEqual(picture.hinge_relations[0].id, "fold-angle")

    def test_entry_geometry_matches_declared_hinge_angle(self) -> None:
        coordinates = self.picture.coordinates
        axis_start = np.asarray(coordinates["A"], dtype=float)
        alpha_ray = np.asarray(coordinates["Alpha0"], dtype=float) - axis_start
        beta_ray = np.asarray(coordinates["Beta0"], dtype=float) - axis_start
        cosine = float(
            np.dot(alpha_ray, beta_ray)
            / (np.linalg.norm(alpha_ray) * np.linalg.norm(beta_ray))
        )
        actual = math.acos(max(-1.0, min(1.0, cosine)))
        self.assertAlmostEqual(actual, self.payload["driver"]["initial"], places=4)

    def test_motion_contract_names_hinge_derived_points_camera_and_restore(self) -> None:
        payload = self.payload
        self.assertEqual(payload["schema"], "tikz-native-motion-3d/v1")
        self.assertEqual(payload["picture_index"], 1)
        self.assertEqual(payload["end_policy"], "restore_entry")

        driver = payload["driver"]
        self.assertEqual(driver["type"], "hinge_fold")
        self.assertEqual(driver["axis"], ["A", "B"])
        self.assertEqual(driver["moving_points"], ["Beta0", "Beta1"])
        self.assertLessEqual(driver["range"][0], driver["initial"])
        self.assertGreaterEqual(driver["range"][1], driver["initial"])

        derived = {item["name"]: item for item in payload["derived_coordinates"]}
        self.assertEqual(derived["M"]["type"], "point_on_segment")
        self.assertEqual(
            (derived["M"]["start"], derived["M"]["end"]),
            ("Beta0", "Beta1"),
        )
        self.assertEqual(derived["N"]["type"], "project_point_to_line")
        self.assertEqual(derived["N"]["point"], "M")
        self.assertEqual(
            (derived["N"]["line_start"], derived["N"]["line_end"]),
            ("A", "B"),
        )

        self.assertEqual(payload["camera"]["entry_mode"], "tikz")
        self.assertEqual(payload["camera"]["restore_transition"], "orbit")
        self.assertEqual(
            [
                step["mode"]
                for step in payload["timeline"]
                if step["type"] == "camera"
            ],
            ["isometric", "side"],
        )
        self.assertNotEqual(payload["timeline"][-1].get("mode"), "tikz")

    def test_motion_spec_validates_and_initial_runtime_reproduces_tikz(self) -> None:
        self.motion.validate_picture(self.picture)
        runtime = NativeMotion3DRuntime(
            self.motion,
            self.picture,
            lambda: self.motion.driver.initial,
        )
        actual = runtime.coordinates()
        for name in ("Beta0", "Beta1", "M", "N"):
            np.testing.assert_allclose(
                actual[name],
                self.picture.coordinates[name],
                atol=1e-9,
                rtol=0.0,
            )

    def test_fold_updates_dependents_and_can_return_to_entry_geometry(self) -> None:
        parameter = [self.motion.driver.initial]
        runtime = NativeMotion3DRuntime(
            self.motion,
            self.picture,
            lambda: parameter[0],
        )
        entry = dict(runtime.coordinates())

        parameter[0] = self.motion.driver.maximum
        folded = dict(runtime.coordinates())
        self.assertFalse(np.allclose(folded["Beta0"], entry["Beta0"]))
        self.assertFalse(np.allclose(folded["M"], entry["M"]))
        axis = np.asarray(folded["B"]) - np.asarray(folded["A"])
        projection_segment = np.asarray(folded["M"]) - np.asarray(folded["N"])
        self.assertAlmostEqual(float(np.dot(axis, projection_segment)), 0.0, places=10)

        parameter[0] = self.motion.driver.initial
        restored = runtime.coordinates()
        for name in ("A", "B", "Beta0", "Beta1", "M", "N"):
            np.testing.assert_allclose(
                restored[name],
                entry[name],
                atol=1e-12,
                rtol=0.0,
            )

    def test_every_binding_targets_a_compiled_object_and_known_coordinate(self) -> None:
        object_ids = {item.id for item in self.picture.objects}
        coordinate_names = set(self.picture.coordinates)
        for binding in self.payload["bindings"]:
            self.assertIn(binding["object_id"], object_ids)
            self.assertTrue(set(binding["points"]).issubset(coordinate_names))

    def test_fixture_records_dynamic_line_face_occlusion(self) -> None:
        relations = self.picture.occlusion_relations
        self.assertEqual(len(relations), 9)
        self.assertTrue(
            any(
                relation.start_name == "Beta1"
                and relation.end_name == "Beta0"
                and relation.face_names == ["A", "B", "Alpha1", "Alpha0"]
                for relation in relations
            )
        )
        self.assertTrue(
            any(
                relation.start_name == "S"
                and relation.end_name == "E"
                and relation.face_names == ["A", "B", "Beta1", "Beta0"]
                for relation in relations
            )
        )
        self.assertTrue(
            any(
                relation.start_name == "Alpha1"
                and relation.end_name == "Alpha0"
                and relation.face_names == ["A", "B", "Beta1", "Beta0"]
                for relation in relations
            )
        )


if __name__ == "__main__":
    unittest.main()
