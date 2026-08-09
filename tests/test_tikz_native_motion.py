from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import json
from math import cos, sin, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from manim import ValueTracker, tempconfig

from tikz_native import compile_document
from tikz_native.dynamic_geometry import EllipseChordDriver
from tikz_native.motion_runtime import (
    MotionConfigError,
    NativeMotionRuntime,
    ellipse_chord_metrics,
    load_motion_spec,
)
from tikz_native.manim_renderer import NativeManimRenderer


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "analytic_geometry_ellipse_demo"
SOURCE = DEMO / "ellipse_problem.tex"
MOTION = DEMO / "ellipse_problem.motion.json"


class TikzNativeMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = compile_document(SOURCE)
        cls.picture = cls.document.pictures[0]
        cls.spec = load_motion_spec(MOTION)

    def test_demo_source_is_strict_native_and_config_is_complete(self) -> None:
        self.assertEqual(len(self.document.pictures), 1)
        self.assertEqual(len(self.picture.objects), 28)
        self.assertEqual(self.picture.warnings, [])
        self.assertEqual(self.picture.unsupported, [])
        self.spec.validate_picture(self.picture)
        self.assertEqual(len(self.spec.bindings), 18)
        self.assertEqual(
            {binding.object_id for binding in self.spec.bindings},
            {
                "line.Lstart.Lend",
                "label.Lend.l",
                "fill.P.Q.R",
                "fill.P.F.O",
                "line.P.Q",
                "line.Q.R",
                "line.R.P",
                "line.P.F",
                "line.O.P",
                "line.Q.O",
                "angle.R.Q.P",
                "label_angle.R.Q.P.varphi",
                "dot.P",
                "dot.Q",
                "dot.R",
                "label.P.P",
                "label.Q.Q",
                "label.R.R",
            },
        )

    def test_initial_motion_state_exactly_reproduces_tikz(self) -> None:
        runtime = NativeMotionRuntime(
            self.spec,
            self.picture,
            lambda: self.spec.driver.initial,
        )
        coordinates = runtime.coordinates()
        for name, expected in self.picture.coordinates.items():
            self.assertAlmostEqual(coordinates[name][0], expected[0], places=12)
            self.assertAlmostEqual(coordinates[name][1], expected[1], places=12)

    def test_keyframes_preserve_all_analytic_geometry_constraints(self) -> None:
        parameter = [self.spec.driver.initial]
        runtime = NativeMotionRuntime(self.spec, self.picture, lambda: parameter[0])
        keyframes = [
            self.spec.driver.minimum,
            0.7137243789447656,
            0.8410686705679303,
            self.spec.driver.maximum,
        ]
        for theta in keyframes:
            parameter[0] = theta
            coordinates = runtime.coordinates()
            p, q, r = coordinates["P"], coordinates["Q"], coordinates["R"]
            for point in (p, q, r):
                self.assertAlmostEqual(
                    point[0] ** 2 / 4 + point[1] ** 2 / 3,
                    1.0,
                    places=11,
                )
            direction = (cos(theta), sin(theta))
            for point in (p, q):
                relative = (point[0] + 1, point[1])
                self.assertAlmostEqual(
                    relative[0] * direction[1] - relative[1] * direction[0],
                    0.0,
                    places=11,
                )
            self.assertLess(q[0], 0)
            self.assertLess(q[1], 0)
            self.assertAlmostEqual(r[0], -p[0], places=12)
            self.assertAlmostEqual(r[1], -p[1], places=12)
            signed_q = (q[0] + 1) * direction[0] + q[1] * direction[1]
            signed_p = (p[0] + 1) * direction[0] + p[1] * direction[1]
            self.assertLess(signed_q, 0)
            self.assertGreater(signed_p, 0)

    def test_problem_answer_keyframes_match_exact_values(self) -> None:
        parameter = [0.8410686705679303]
        runtime = NativeMotionRuntime(self.spec, self.picture, lambda: parameter[0])
        area_metrics = ellipse_chord_metrics(runtime.coordinates())
        self.assertAlmostEqual(area_metrics.slope, sqrt(5) / 2, places=12)
        self.assertAlmostEqual(area_metrics.area_ratio, 3.0, places=12)
        self.assertAlmostEqual(runtime.coordinates()["P"][0], 0.5, places=12)
        self.assertAlmostEqual(runtime.coordinates()["Q"][0], -1.75, places=12)

        parameter[0] = 0.7137243789447656
        tangent_metrics = ellipse_chord_metrics(runtime.coordinates())
        self.assertAlmostEqual(tangent_metrics.slope, sqrt(3) / 2, places=12)
        self.assertAlmostEqual(
            tangent_metrics.angle_tangent,
            4 * sqrt(3),
            places=11,
        )
        for delta in (-0.08, 0.08):
            parameter[0] = 0.7137243789447656 + delta
            self.assertGreater(
                ellipse_chord_metrics(runtime.coordinates()).angle_tangent,
                4 * sqrt(3),
            )

    def test_binding_updates_native_objects_and_keeps_fixed_objects_still(self) -> None:
        with TemporaryDirectory() as directory, tempconfig({"media_dir": directory}):
            renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
            figure = renderer.render(self.picture)
            tracker = ValueTracker(self.spec.driver.initial)
            runtime = NativeMotionRuntime(self.spec, self.picture, tracker.get_value)
            runtime.bind(
                figure,
                renderer,
                lambda point: renderer.point(point, self.picture),
            )

            original_p = figure.objects["dot.P"].get_center().copy()
            original_angle_start = figure.objects["angle.R.Q.P"].get_start().copy()
            fixed_ellipse_center = figure.objects["ellipse.O"].get_center().copy()
            fixed_ellipse_width = float(figure.objects["ellipse.O"].width)
            object_identities = {key: id(value) for key, value in figure.objects.items()}

            tracker.set_value(self.spec.driver.maximum)
            figure.group.update(0)
            expected_p = renderer.point(runtime.coordinates()["P"], self.picture)
            np.testing.assert_allclose(
                figure.objects["dot.P"].get_center(), expected_p, atol=1e-9
            )
            self.assertFalse(
                np.allclose(figure.objects["dot.P"].get_center(), original_p)
            )
            self.assertFalse(
                np.allclose(
                    figure.objects["angle.R.Q.P"].get_start(),
                    original_angle_start,
                )
            )
            angle_object_spec = next(
                item for item in self.picture.objects if item.id == "angle.R.Q.P"
            )
            expected_angle = renderer.native_angle_from_points(
                renderer.point(runtime.coordinates()["R"], self.picture),
                renderer.point(runtime.coordinates()["Q"], self.picture),
                renderer.point(runtime.coordinates()["P"], self.picture),
                radius=angle_object_spec.geometry["radius_pt"] * renderer.pt,
                style=angle_object_spec.style,
            )
            np.testing.assert_allclose(
                figure.objects["angle.R.Q.P"].get_arc_center(),
                expected_angle.get_arc_center(),
                atol=1e-9,
            )
            np.testing.assert_allclose(
                figure.objects["angle.R.Q.P"].get_start(),
                expected_angle.get_start(),
                atol=1e-9,
            )
            np.testing.assert_allclose(
                figure.objects["angle.R.Q.P"].get_end(),
                expected_angle.get_end(),
                atol=1e-9,
            )
            angle_spec = next(
                item
                for item in self.picture.objects
                if item.id == "label_angle.R.Q.P.varphi"
            )
            expected_angle_label = renderer.native_angle_label_position(
                renderer.point(runtime.coordinates()["R"], self.picture),
                renderer.point(runtime.coordinates()["Q"], self.picture),
                renderer.point(runtime.coordinates()["P"], self.picture),
                radius=angle_spec.geometry["radius_pt"] * renderer.pt,
                eccentricity=angle_spec.geometry["eccentricity"],
            )
            np.testing.assert_allclose(
                figure.objects["label_angle.R.Q.P.varphi"].get_center(),
                expected_angle_label,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                figure.objects["ellipse.O"].get_center(),
                fixed_ellipse_center,
                atol=1e-12,
            )
            self.assertAlmostEqual(
                float(figure.objects["ellipse.O"].width), fixed_ellipse_width
            )
            self.assertEqual(
                {key: id(value) for key, value in figure.objects.items()},
                object_identities,
            )

            tracker.set_value(self.spec.driver.initial)
            figure.group.update(0)
            np.testing.assert_allclose(
                figure.objects["dot.P"].get_center(), original_p, atol=1e-9
            )

    def test_invalid_object_and_out_of_range_parameter_fail_closed(self) -> None:
        bad_binding = replace(self.spec.bindings[0], object_id="missing.object")
        bad_spec = replace(
            self.spec,
            bindings=(bad_binding, *self.spec.bindings[1:]),
        )
        with self.assertRaisesRegex(MotionConfigError, "unknown object id"):
            bad_spec.validate_picture(self.picture)

        runtime = NativeMotionRuntime(
            self.spec,
            self.picture,
            lambda: self.spec.driver.maximum + 0.01,
        )
        with self.assertRaisesRegex(MotionConfigError, "outside driver.range"):
            runtime.coordinates()

        payload = json.loads(MOTION.read_text(encoding="utf-8"))
        payload["driver"]["python"] = "__import__('os').system('false')"
        with self.assertRaisesRegex(MotionConfigError, "unsupported fields: python"):
            type(self.spec).from_dict(payload)

    def test_schema_required_fields_nonfinite_values_and_cues_fail_closed(self) -> None:
        missing_unit = json.loads(MOTION.read_text(encoding="utf-8"))
        del missing_unit["driver"]["unit"]
        with self.assertRaisesRegex(MotionConfigError, "driver.unit"):
            type(self.spec).from_dict(missing_unit)

        missing_timeline = json.loads(MOTION.read_text(encoding="utf-8"))
        del missing_timeline["timeline"]
        with self.assertRaisesRegex(MotionConfigError, "timeline must be an array"):
            type(self.spec).from_dict(missing_timeline)

        nonfinite = json.loads(MOTION.read_text(encoding="utf-8"))
        nonfinite["driver"]["initial"] = float("nan")
        with self.assertRaisesRegex(MotionConfigError, "must be finite"):
            type(self.spec).from_dict(nonfinite)

        bad_step = replace(self.spec.timeline[0], cue="misspelled_cue")
        bad_cues = replace(
            self.spec,
            timeline=(bad_step, *self.spec.timeline[1:]),
        )
        with self.assertRaisesRegex(MotionConfigError, "misspelled_cue"):
            bad_cues.validate_cues({step.cue for step in self.spec.timeline if step.cue})

    def test_failed_dependency_evaluation_never_returns_old_cached_geometry(self) -> None:
        picture = deepcopy(self.picture)
        parameter = [self.spec.driver.initial]
        runtime = NativeMotionRuntime(self.spec, picture, lambda: parameter[0])
        original = runtime.coordinates()["P"]
        parameter[0] = self.spec.driver.maximum
        picture.coordinate_dependencies["Pguide"] = {
            "operation": "projection",
            "line_start": "O",
            "point": "P",
            "line_end": "O",
        }
        for _ in range(2):
            with self.assertRaisesRegex(
                MotionConfigError, "zero-length line"
            ):
                runtime.coordinates()
        self.assertEqual(runtime._cached_coordinates["P"], original)
        self.assertEqual(runtime._cached_parameter, self.spec.driver.initial)

    def test_driver_failure_does_not_commit_a_new_cache_key(self) -> None:
        parameter = [self.spec.driver.initial]
        driver = EllipseChordDriver.from_named_intersection(
            lambda: parameter[0],
            self.picture,
            relation_index=self.spec.driver.intersection_index,
            pivot_name=self.spec.driver.pivot,
        )
        original = driver.state()
        parameter[0] = self.spec.driver.maximum
        driver.parameters["semi_major"] = 0.0
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "semiaxes must be positive"):
                driver.state()
        self.assertIs(driver._cached_state, original)
        self.assertEqual(driver._cached_angle, self.spec.driver.initial)

    def test_unselected_intersection_using_active_path_is_rejected(self) -> None:
        picture = deepcopy(self.picture)
        picture.coordinates["S"] = picture.coordinates["P"]
        picture.coordinate_dependencies["S"] = {
            "operation": "intersection",
            "path_a": "Cpath",
            "path_b": "Lpath",
            "sort_by": "Lpath",
            "sorted_index": 0,
        }
        runtime = NativeMotionRuntime(
            self.spec,
            picture,
            lambda: self.spec.driver.initial,
        )
        with self.assertRaisesRegex(
            MotionConfigError, "unselected intersection coordinate 'S'"
        ):
            runtime.coordinates()

    def test_motion_runtime_and_demo_do_not_use_generic_visual_fallbacks(self) -> None:
        forbidden = {"VMobject", "SVGMobject", "ImageMobject", "set_points_as_corners"}
        hits = []
        for path in (ROOT / "tikz_native" / "motion_runtime.py", DEMO / "scene.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                else:
                    continue
                if name in forbidden:
                    hits.append((path.name, node.lineno, name))
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
