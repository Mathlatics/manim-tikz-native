from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json

import numpy as np
from manim import ValueTracker, tempconfig
from jsonschema import Draft202012Validator

from tikz_native import compile_document
from tikz_native.camera_3d import ISOMETRIC_MATRIX, MultiProjectionCamera
from tikz_native.manim_renderer_3d import NativeManim3DRenderer
from tikz_native.motion_3d import (
    MOTION_3D_SCHEMA,
    Motion3DConfigError,
    Motion3DSpec,
    NativeMotion3DRuntime,
    point_on_segment_3d,
    project_point_to_line_3d,
    rotate_point_about_axis,
)


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "dihedral_fold_3d_demo"
SOURCE = DEMO / "dihedral_fold.tex"
MOTION = DEMO / "motion-3d.json"


class TikzNativeMotion3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = compile_document(SOURCE)
        cls.picture = cls.document.pictures[0]
        cls.spec = Motion3DSpec.load(MOTION)

    def test_demo_contract_and_picture_are_strict_native_3d(self) -> None:
        self.assertEqual(self.spec.schema, MOTION_3D_SCHEMA)
        self.assertEqual(self.spec.end_policy, "restore_entry")
        self.assertEqual(self.picture.dimension, 3)
        self.assertIsNotNone(self.picture.projection_3d)
        self.assertEqual(self.picture.unsupported, [])
        self.spec.validate_picture(self.picture)

    def test_public_json_schema_accepts_the_demo_contract(self) -> None:
        schema = json.loads(
            (ROOT / "tikz_native" / "schemas" / "motion-3d-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(schema)
        payload = json.loads(MOTION.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(payload)

    def test_rodrigues_rotates_about_an_arbitrary_axis(self) -> None:
        axis_start = np.array((1.0, -2.0, 0.5))
        axis_end = np.array((2.0, -1.0, 1.5))
        point = np.array((3.0, -2.0, 1.0))
        rotated = np.asarray(
            rotate_point_about_axis(point, axis_start, axis_end, np.pi / 2)
        )
        axis = axis_end - axis_start
        axis /= np.linalg.norm(axis)
        before = point - axis_start
        after = rotated - axis_start
        self.assertAlmostEqual(np.dot(before, axis), np.dot(after, axis), places=12)
        self.assertAlmostEqual(np.linalg.norm(before), np.linalg.norm(after), places=12)
        self.assertAlmostEqual(np.dot(before, after), np.dot(before, axis) ** 2, places=12)

    def test_derived_coordinate_math_is_true_3d(self) -> None:
        midpoint = point_on_segment_3d((1, 2, 3), (5, 6, 7), 0.25)
        self.assertEqual(midpoint, (2.0, 3.0, 4.0))
        foot = project_point_to_line_3d(
            (2.0, 4.0, 3.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        )
        np.testing.assert_allclose(foot, (3.0, 3.0, 3.0), atol=1e-12)

    def test_initial_state_reproduces_tikz_without_mutating_authored_coordinates(self) -> None:
        authored = deepcopy(self.picture.coordinates)
        runtime = NativeMotion3DRuntime(
            self.spec,
            self.picture,
            lambda: self.spec.driver.initial,
        )
        coordinates = runtime.coordinates()
        self.assertEqual(self.picture.coordinates, authored)
        for name, expected in authored.items():
            np.testing.assert_allclose(coordinates[name], expected, atol=1e-10)

    def test_fold_preserves_rigid_face_and_updates_dependencies(self) -> None:
        parameter = [self.spec.driver.initial]
        runtime = NativeMotion3DRuntime(self.spec, self.picture, lambda: parameter[0])
        initial = dict(runtime.coordinates())
        initial_lengths = {
            pair: np.linalg.norm(np.asarray(initial[pair[1]]) - initial[pair[0]])
            for pair in (("A", "Beta0"), ("B", "Beta1"), ("Beta0", "Beta1"))
        }
        parameter[0] = self.spec.driver.maximum
        folded = runtime.coordinates()
        for pair, expected in initial_lengths.items():
            actual = np.linalg.norm(np.asarray(folded[pair[1]]) - folded[pair[0]])
            self.assertAlmostEqual(actual, expected, places=10)
        expected_m = point_on_segment_3d(folded["Beta0"], folded["Beta1"], 0.67)
        expected_n = project_point_to_line_3d(expected_m, folded["A"], folded["B"])
        np.testing.assert_allclose(folded["M"], expected_m, atol=1e-10)
        np.testing.assert_allclose(folded["N"], expected_n, atol=1e-10)
        direction = np.asarray(folded["B"]) - folded["A"]
        self.assertAlmostEqual(
            np.dot(np.asarray(folded["M"]) - folded["N"], direction),
            0.0,
            places=10,
        )

    def test_validation_rejects_non_restoring_policy_and_dependency_cycle(self) -> None:
        payload = deepcopy(json.loads(MOTION.read_text(encoding="utf-8")))
        payload["end_policy"] = "keep_result"
        with self.assertRaisesRegex(Motion3DConfigError, "restore_entry"):
            Motion3DSpec.from_dict(payload)

        payload["end_policy"] = "restore_entry"
        payload["derived_coordinates"] = [
            {
                "name": "M",
                "type": "point_on_segment",
                "start": "N",
                "end": "A",
                "parameter": 0.5,
            },
            {
                "name": "N",
                "type": "point_on_segment",
                "start": "M",
                "end": "B",
                "parameter": 0.5,
            },
        ]
        cyclic = Motion3DSpec.from_dict(payload)
        runtime = NativeMotion3DRuntime(cyclic, self.picture, lambda: cyclic.driver.initial)
        with self.assertRaisesRegex(Motion3DConfigError, "dependency cycle"):
            runtime.coordinates()

    def test_portable_camera_registers_tikz_view_and_restores_snapshot(self) -> None:
        camera = MultiProjectionCamera(initial_mode="front")
        original = camera.snapshot()
        runtime = NativeMotion3DRuntime(
            self.spec,
            self.picture,
            lambda: self.spec.driver.initial,
        )
        center = np.array((1.2, 0.3, -0.4))
        runtime.prepare_camera(camera, view_center=center)
        self.assertEqual(camera.current_mode, "tikz")
        np.testing.assert_allclose(camera.get_view_center(), center, atol=1e-12)
        camera.animate_orbit_to("isometric", arc_height=0.25)
        camera.transition_tracker.set_value(1.0)
        np.testing.assert_allclose(
            camera.get_projection_matrix(), ISOMETRIC_MATRIX, atol=1e-12
        )
        camera.restore(original)
        np.testing.assert_allclose(
            camera.get_projection_matrix(), original.matrix, atol=1e-12
        )
        np.testing.assert_allclose(
            camera.get_view_center(), original.view_center, atol=1e-12
        )

    def test_binding_and_occlusion_keep_native_object_identity(self) -> None:
        with TemporaryDirectory() as directory, tempconfig({"media_dir": directory}):
            renderer = NativeManim3DRenderer(scene_unit_per_cm=1.0)
            figure = renderer.render(self.picture)
            tracker = ValueTracker(self.spec.driver.initial)
            runtime = NativeMotion3DRuntime(
                self.spec,
                self.picture,
                tracker.get_value,
            )
            camera = MultiProjectionCamera(initial_mode="isometric")
            runtime.prepare_camera(camera, view_center=figure.view_center)
            runtime.bind(figure, renderer, camera=camera)
            occlusions = runtime.bind_occlusions(figure, renderer, camera)

            object_ids = {
                binding.object_id: id(figure.objects[binding.object_id])
                for binding in self.spec.bindings
            }
            occlusion_ids = {
                key: (id(group), tuple(id(child) for child in group.submobjects))
                for key, group in occlusions.items()
            }
            moving_dot = figure.objects["dot.M"]
            moving_label = figure.objects["label.M.M"]
            dot_before = moving_dot.get_center().copy()
            label_before = moving_label.get_center().copy()

            tracker.set_value(self.spec.driver.maximum)
            figure.world_group.update(0)
            for label in figure.fixed_orientation_labels:
                label.update(0)

            self.assertFalse(np.allclose(moving_dot.get_center(), dot_before))
            self.assertFalse(np.allclose(moving_label.get_center(), label_before))
            self.assertEqual(
                object_ids,
                {
                    binding.object_id: id(figure.objects[binding.object_id])
                    for binding in self.spec.bindings
                },
            )
            self.assertEqual(
                occlusion_ids,
                {
                    key: (id(group), tuple(id(child) for child in group.submobjects))
                    for key, group in occlusions.items()
                },
            )


if __name__ == "__main__":
    unittest.main()
