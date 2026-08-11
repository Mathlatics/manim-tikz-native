from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import Mobject, Scene, linear, tempconfig

from tikz_native import (
    NATIVE_RIG_2D_API_SCHEMA,
    NativeGeometryRig2D,
    NativeRig2D,
    compile_document,
    load_motion_spec,
)
from tikz_native.manim_renderer import NativeManimRenderer
from tikz_native.motion_runtime import MotionConfigError


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "analytic_geometry_ellipse_demo"
SOURCE = DEMO / "ellipse_problem.tex"
MOTION = DEMO / "ellipse_problem.motion.json"


class TikzNativeNativeRig2DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]
        cls.motion_payload = json.loads(MOTION.read_text(encoding="utf-8"))
        cls.spec = load_motion_spec(MOTION)
        cls.media_directory = TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.media_directory.cleanup()

    def _figure(self, *, provider_metadata: bool = False):
        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        figure = renderer.render(self.picture)
        figure.group.scale(0.63).rotate(0.27).shift(
            np.array([1.2, -0.4, 0.0])
        )
        if provider_metadata:
            figure.group._tikz_native_object_map = figure.objects
            figure.group._tikz_native_picture = figure.picture
        else:
            figure.group._codex_tikz_native_objects = figure.objects
            figure.group._codex_tikz_native_picture = figure.picture
        return figure

    def _rig(self, figure) -> NativeGeometryRig2D:
        return NativeGeometryRig2D(
            figure.group,
            self.motion_payload,
            active_object_id="line.Lstart.Lend",
        )

    def test_public_contract_exposes_tracker_range_objects_and_coordinates(self) -> None:
        with tempconfig({"media_dir": self.media_directory.name}):
            figure = self._figure(provider_metadata=True)
            rig = self._rig(figure)

            self.assertEqual(rig.api_schema, NATIVE_RIG_2D_API_SCHEMA)
            self.assertIs(NativeRig2D, NativeGeometryRig2D)
            self.assertEqual(rig.initial, self.spec.driver.initial)
            self.assertEqual(rig.minimum, self.spec.driver.minimum)
            self.assertEqual(rig.maximum, self.spec.driver.maximum)
            self.assertEqual(rig.range, (rig.minimum, rig.maximum))
            self.assertEqual(rig.tracker.get_value(), rig.initial)
            self.assertIs(rig.object("dot.P"), figure.objects["dot.P"])
            np.testing.assert_allclose(
                rig.coordinate("P"),
                figure.objects["dot.P"].get_center(),
                atol=1e-7,
            )
            np.testing.assert_allclose(
                rig.logical_coordinate("P"),
                self.picture.coordinates["P"],
                atol=1e-12,
            )
            with self.assertRaises(TypeError):
                rig.objects["dot.P"] = Mobject()

    def test_context_updates_in_place_and_restores_every_semantic_object(self) -> None:
        with tempconfig({"media_dir": self.media_directory.name}):
            figure = self._figure()
            identities = {key: id(value) for key, value in figure.objects.items()}
            entry_points = {
                key: value.get_all_points().copy()
                for key, value in figure.objects.items()
            }
            fixed_ellipse = figure.objects["ellipse.O"]
            entry_opacity = fixed_ellipse.get_stroke_opacity()
            entry_p = figure.objects["dot.P"].get_center().copy()

            rig = self._rig(figure)
            with rig:
                self.assertTrue(rig.attached)
                rig.tracker.set_value(rig.maximum)
                figure.group.update(0)
                np.testing.assert_allclose(
                    rig.coordinate("P"),
                    figure.objects["dot.P"].get_center(),
                    atol=1e-7,
                )
                self.assertFalse(
                    np.allclose(figure.objects["dot.P"].get_center(), entry_p)
                )
                fixed_ellipse.set_stroke(opacity=0.17)

            self.assertFalse(rig.attached)
            self.assertEqual(
                {key: id(value) for key, value in figure.objects.items()},
                identities,
            )
            for key, value in figure.objects.items():
                np.testing.assert_allclose(
                    value.get_all_points(), entry_points[key], atol=1e-9
                )
                self.assertEqual(list(value.updaters), [])
            self.assertEqual(fixed_ellipse.get_stroke_opacity(), entry_opacity)

    def test_attach_has_no_entry_jump_for_any_transformed_shape_state_binding(self) -> None:
        with tempconfig({"media_dir": self.media_directory.name}):
            figure = self._figure()
            rig = self._rig(figure)
            entry = {
                binding.object_id: figure.objects[binding.object_id].copy()
                for binding in rig.spec.bindings
            }

            with rig:
                for binding in rig.spec.bindings:
                    current = figure.objects[binding.object_id]
                    original = entry[binding.object_id]
                    self.assertIsNone(
                        NativeGeometryRig2D._entry_alignment_error(
                            current,
                            original,
                        ),
                        binding.object_id,
                    )
                    np.testing.assert_allclose(
                        current.get_all_points(),
                        original.get_all_points(),
                        atol=1e-9,
                        err_msg=binding.object_id,
                    )

                np.testing.assert_allclose(
                    figure.objects["label_angle.R.Q.P.varphi"].get_center(),
                    entry["label_angle.R.Q.P.varphi"].get_center(),
                    atol=1e-9,
                )

    def test_author_can_drive_tracker_with_ordinary_scene_play(self) -> None:
        with tempconfig(
            {
                "media_dir": self.media_directory.name,
                "dry_run": True,
                "frame_rate": 5,
            }
        ):
            figure = self._figure()
            entry_p = figure.objects["dot.P"].get_center().copy()
            scene = Scene()
            scene.add(figure.group)

            with self._rig(figure) as rig:
                scene.play(
                    rig.tracker.animate.set_value(rig.maximum),
                    run_time=0.2,
                    rate_func=linear,
                )
                self.assertFalse(
                    np.allclose(figure.objects["dot.P"].get_center(), entry_p)
                )
                np.testing.assert_allclose(
                    figure.objects["dot.P"].get_center(),
                    rig.coordinate("P"),
                    atol=1e-7,
                )

            np.testing.assert_allclose(
                figure.objects["dot.P"].get_center(), entry_p, atol=1e-9
            )

    def test_context_restores_entry_after_author_exception(self) -> None:
        with tempconfig({"media_dir": self.media_directory.name}):
            figure = self._figure()
            entry_p = figure.objects["dot.P"].get_center().copy()
            rig = self._rig(figure)

            with self.assertRaisesRegex(RuntimeError, "author failure"):
                with rig:
                    rig.tracker.set_value(rig.maximum)
                    figure.group.update(0)
                    raise RuntimeError("author failure")

            np.testing.assert_allclose(
                figure.objects["dot.P"].get_center(), entry_p, atol=1e-9
            )
            for value in figure.objects.values():
                self.assertEqual(list(value.updaters), [])

    def test_detach_freezes_current_frame_until_explicit_restore(self) -> None:
        with tempconfig({"media_dir": self.media_directory.name}):
            figure = self._figure()
            entry_p = figure.objects["dot.P"].get_center().copy()
            rig = self._rig(figure).attach()
            rig.tracker.set_value(rig.maximum)
            figure.group.update(0)
            detached_p = figure.objects["dot.P"].get_center().copy()

            rig.detach()
            self.assertFalse(rig.attached)
            rig.tracker.set_value(rig.minimum)
            figure.group.update(0)
            np.testing.assert_allclose(
                figure.objects["dot.P"].get_center(), detached_p, atol=1e-9
            )

            self.assertIs(rig.restore_entry(), figure.group)
            np.testing.assert_allclose(
                figure.objects["dot.P"].get_center(), entry_p, atol=1e-9
            )

    def test_invalid_semantic_access_and_parameter_fail_closed(self) -> None:
        with tempconfig({"media_dir": self.media_directory.name}):
            figure = self._figure()
            rig = self._rig(figure)
            with self.assertRaisesRegex(MotionConfigError, "unknown semantic object"):
                rig.object("missing.object")
            with self.assertRaisesRegex(MotionConfigError, "unknown semantic coordinate"):
                rig.coordinate("MissingPoint")
            rig.tracker.set_value(rig.maximum + 0.01)
            with self.assertRaisesRegex(MotionConfigError, "outside driver.range"):
                rig.logical_coordinate("P")

    def test_preexisting_bound_updater_is_rejected_without_removing_it(self) -> None:
        with tempconfig({"media_dir": self.media_directory.name}):
            figure = self._figure()
            active = figure.objects["line.Lstart.Lend"]
            updater = lambda item: item
            active.add_updater(updater)
            rig = self._rig(figure)

            with self.assertRaisesRegex(MotionConfigError, "already has active updaters"):
                rig.attach()
            self.assertEqual(list(active.updaters), [updater])

    def test_partial_attach_failure_restores_objects_and_updaters(self) -> None:
        with tempconfig({"media_dir": self.media_directory.name}):
            figure = self._figure()
            active = figure.objects["line.Lstart.Lend"]
            entry_points = active.get_all_points().copy()
            rig = self._rig(figure)

            def fail_after_one_updater(*_args, **_kwargs):
                active.add_updater(lambda item: item.shift(np.array([0.1, 0, 0])))
                raise RuntimeError("binding failed")

            with patch.object(rig.runtime, "bind", side_effect=fail_after_one_updater):
                with self.assertRaisesRegex(RuntimeError, "binding failed"):
                    rig.attach()
            np.testing.assert_allclose(active.get_all_points(), entry_points, atol=1e-9)
            self.assertEqual(list(active.updaters), [])

    def test_attach_rejects_a_jump_in_any_nonactive_binding(self) -> None:
        with tempconfig({"media_dir": self.media_directory.name}):
            figure = self._figure()
            follower = figure.objects["dot.P"]
            entry_points = follower.get_all_points().copy()
            rig = self._rig(figure)
            original_bind = rig.runtime.bind

            def bind_with_follower_jump(*args, **kwargs):
                result = original_bind(*args, **kwargs)
                follower.add_updater(
                    lambda item: item.shift(np.array([0.01, 0.0, 0.0]))
                )
                return result

            with patch.object(
                rig.runtime,
                "bind",
                side_effect=bind_with_follower_jump,
            ):
                with self.assertRaisesRegex(
                    MotionConfigError,
                    "ShapeState object 'dot.P'.*moved",
                ):
                    rig.attach()

            np.testing.assert_allclose(
                follower.get_all_points(), entry_points, atol=1e-9
            )
            self.assertEqual(list(follower.updaters), [])


if __name__ == "__main__":
    unittest.main()
