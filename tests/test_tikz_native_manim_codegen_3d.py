from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import unittest

import numpy as np
from manim import Scene, ValueTracker, tempconfig
from manim.animation.animation import prepare_animation

from tikz_native.compiler import compile_document
from tikz_native.geometry_rig_3d import analyze_geometry_rig_3d
from tikz_native.native_manim_codegen_3d import _style_payload
from tikz_native.provider import instantiate_picture


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"
SELECTION = {
    "candidate_id": "hinge_fold:fold-angle",
    "range": [0.3141592653589793, 1.9547687622336491],
}


class _InstantScene(Scene):
    """Exercise ordinary Manim AnimationBuilders without writing media."""

    def play(self, *builders, **_kwargs) -> None:  # type: ignore[override]
        animations = [prepare_animation(builder) for builder in builders]
        for animation in animations:
            animation.begin()
        for alpha in (0.5, 1.0):
            for animation in animations:
                animation.interpolate(alpha)
            for mobject in self.mobjects:
                mobject.update(0.0)
        for animation in animations:
            animation.finish()


class TikzNativeReadableManim3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]

    def _analysis(self, *, exclude_object_ids=()):
        selection = dict(SELECTION)
        if exclude_object_ids:
            selection["exclude_object_ids"] = list(exclude_object_ids)
        result = analyze_geometry_rig_3d(self.picture, selection=selection)
        self.assertEqual(result["status"], "ready", result["diagnostics"])
        return result

    def _namespace(self, *, exclude_object_ids=()):
        payload = self._analysis(
            exclude_object_ids=exclude_object_ids
        )["nativeManimSource"]
        namespace: dict[str, object] = {}
        exec(compile(payload["sourceText"], "<native-manim-source-3d>", "exec"), namespace)
        return payload, namespace

    def _figure(self):
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        figure.group.scale(0.72).rotate(0.13).shift((1.1, -0.45, 0.0))
        return figure

    def test_ready_rig_emits_hashed_readable_native_manim_source(self) -> None:
        payload, namespace = self._namespace()
        source = payload["sourceText"]
        self.assertEqual(payload["schema"], "tikz-native-manim-source-3d/v1")
        self.assertEqual(
            payload["sourceSha256"],
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )
        self.assertTrue(source.startswith("import numpy as np\nfrom manim import *\n"))
        for required in (
            "def rotate_point_about_axis(",
            "def geometry_coordinates_3d(",
            "def prepare_local_camera(",
            "def parallel_occlusion_interval(",
            "def install_geometry_3d_updaters(",
            "def assert_shape_state_3d_entry(",
            "def restore_geometry_3d_objects(",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "import tikz_native",
            "from tikz_native",
            "play_motion_3d_on_native_shape",
            "Motion3DSpec",
            "NativeManim3DRenderer",
            "/Users/",
        ):
            self.assertNotIn(forbidden, source)
        for symbol in (
            "HINGE_ANGLE_INITIAL",
            "HINGE_ANGLE_MINIMUM",
            "HINGE_ANGLE_MAXIMUM",
            "CAMERA_PROGRESS_INITIAL",
            "install_geometry_3d_updaters",
            "prepare_local_camera",
            "assert_shape_state_3d_entry",
            "restore_geometry_3d_objects",
        ):
            self.assertIn(symbol, namespace)

    def test_occlusion_style_uses_renderer_opacity_and_color_fallback(self) -> None:
        style = deepcopy(self.picture.occlusion_relations[0].visible_style)
        style.opacity = 0.4
        style.draw_opacity = 0.25
        style.draw_color = None
        payload = _style_payload(style)
        self.assertAlmostEqual(payload["opacity"], 0.1)
        self.assertEqual(payload["draw_color"], "#20242A")

    def test_real_shape_state_executes_hinge_camera_occlusion_and_exact_restore(self) -> None:
        _payload, namespace = self._namespace()
        figure = self._figure()
        shape = figure.group
        objects = figure.objects
        entry_points = {
            object_id: item.get_all_points().copy()
            for object_id, item in objects.items()
        }
        original_child_ids = tuple(id(item) for item in shape.submobjects)
        hinge = ValueTracker(namespace["HINGE_ANGLE_INITIAL"])
        camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
        scene = _InstantScene()
        scene.add(shape)
        state = namespace["install_geometry_3d_updaters"](
            shape, objects, hinge, camera
        )
        self.assertTrue(namespace["assert_shape_state_3d_entry"](state))
        slot_identity = [
            tuple(id(item) for item in group.submobjects)
            for group in state["temporary_groups"]
        ]
        moving_entry = objects["dot.M"].get_center().copy()
        fixed_entry = objects["fill.A.B.Alpha1.Alpha0"].get_all_points().copy()

        scene.play(
            hinge.animate.set_value(namespace["HINGE_ANGLE_MAXIMUM"]),
            run_time=0.1,
        )
        self.assertFalse(np.allclose(objects["dot.M"].get_center(), moving_entry))
        np.testing.assert_allclose(
            objects["fill.A.B.Alpha1.Alpha0"].get_all_points(),
            fixed_entry,
            atol=1e-8,
        )

        namespace["prepare_local_camera"](
            state, "side", transition="linear", arc_height=0.2
        )
        scene.play(camera.animate.set_value(1.0), run_time=0.1)
        self.assertFalse(
            np.allclose(
                objects["fill.A.B.Alpha1.Alpha0"].get_all_points(),
                fixed_entry,
            )
        )
        self.assertEqual(
            slot_identity,
            [
                tuple(id(item) for item in group.submobjects)
                for group in state["temporary_groups"]
            ],
        )

        namespace["restore_geometry_3d_objects"](state)
        self.assertEqual(original_child_ids, tuple(id(item) for item in shape.submobjects))
        for object_id, entry in entry_points.items():
            np.testing.assert_allclose(
                objects[object_id].get_all_points(),
                entry,
                atol=1e-10,
                err_msg=object_id,
            )
            self.assertFalse(objects[object_id].updaters, object_id)

    def test_excluded_follower_stays_at_entry_world_coordinate_until_camera_changes(self) -> None:
        _payload, namespace = self._namespace(
            exclude_object_ids=("label.Beta0.beta",)
        )
        figure = self._figure()
        shape = figure.group
        label = figure.objects["label.Beta0.beta"]
        entry = label.get_center().copy()
        hinge = ValueTracker(namespace["HINGE_ANGLE_INITIAL"])
        camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
        state = namespace["install_geometry_3d_updaters"](
            shape, figure.objects, hinge, camera
        )
        hinge.set_value(namespace["HINGE_ANGLE_MAXIMUM"])
        shape.update(0.0)
        np.testing.assert_allclose(label.get_center(), entry, atol=1e-8)
        namespace["prepare_local_camera"](state, "side", transition="linear")
        camera.set_value(1.0)
        shape.update(0.0)
        self.assertFalse(np.allclose(label.get_center(), entry))
        namespace["restore_geometry_3d_objects"](state)

    def test_oblique_camera_and_linear_return_to_tikz_preserve_the_entry(self) -> None:
        _payload, namespace = self._namespace()
        figure = self._figure()
        shape = figure.group
        entry_points = {
            object_id: item.get_all_points().copy()
            for object_id, item in figure.objects.items()
        }
        hinge = ValueTracker(namespace["HINGE_ANGLE_INITIAL"])
        camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
        state = namespace["install_geometry_3d_updaters"](
            shape,
            figure.objects,
            hinge,
            camera,
        )

        namespace["prepare_local_camera"](
            state,
            "oblique",
            transition="linear",
        )
        camera.set_value(1.0)
        shape.update(0.0)
        self.assertFalse(
            np.allclose(
                figure.objects["fill.A.B.Alpha1.Alpha0"].get_all_points(),
                entry_points["fill.A.B.Alpha1.Alpha0"],
            )
        )

        namespace["prepare_local_camera"](
            state,
            "tikz",
            transition="linear",
        )
        camera.set_value(1.0)
        shape.update(0.0)
        for object_id, entry in entry_points.items():
            np.testing.assert_allclose(
                figure.objects[object_id].get_all_points(),
                entry,
                atol=1e-8,
                err_msg=object_id,
            )
        namespace["restore_geometry_3d_objects"](state)

    def test_install_is_pixel_identical_to_the_real_shape_state_entry(self) -> None:
        _payload, namespace = self._namespace()
        with tempconfig({"pixel_width": 640, "pixel_height": 360, "frame_rate": 5}):
            figure = self._figure()
            shape = figure.group
            scene = Scene()
            scene.add(shape)
            scene.camera.reset()
            scene.camera.capture_mobjects(scene.mobjects)
            before = scene.camera.pixel_array.copy()
            hinge = ValueTracker(namespace["HINGE_ANGLE_INITIAL"])
            camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
            state = namespace["install_geometry_3d_updaters"](
                shape, figure.objects, hinge, camera
            )
            shape.update(0.0)
            scene.camera.reset()
            scene.camera.capture_mobjects(scene.mobjects)
            after = scene.camera.pixel_array.copy()
            np.testing.assert_array_equal(after, before)
            namespace["restore_geometry_3d_objects"](state)


if __name__ == "__main__":
    unittest.main()
