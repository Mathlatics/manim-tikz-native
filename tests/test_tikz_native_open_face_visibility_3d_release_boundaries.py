from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from manim import Scene, ValueTracker, tempconfig

from polyhedron_visibility import OcclusionStyle, ParallelProjection
from tikz_native.compiler import compile_document
from tikz_native.geometry_rig_3d import analyze_geometry_rig_3d
from tikz_native.open_face_visibility_3d_manim import (
    TikzNativeOpenFaceVisibility3DManimError,
    bind_picture_open_face_visibility_3d,
)
from tikz_native.provider import instantiate_picture


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"


def _array_tuple(value: object) -> tuple[float, ...]:
    return tuple(np.asarray(value, dtype=float).reshape(-1))


def _figure_style_snapshot(figure: object) -> tuple[object, ...]:
    values: list[object] = []
    objects = getattr(figure, "objects")
    for object_id in sorted(objects):
        for member in objects[object_id].get_family():
            values.append(
                (
                    object_id,
                    id(member),
                    _array_tuple(member.get_all_points()),
                    _array_tuple(getattr(member, "stroke_rgbas", ())),
                    _array_tuple(getattr(member, "background_stroke_rgbas", ())),
                    _array_tuple(getattr(member, "stroke_width", ())),
                    _array_tuple(getattr(member, "background_stroke_width", ())),
                    float(member.z_index),
                )
            )
    return tuple(values)


def _cairo_pixels(scene: Scene) -> np.ndarray:
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    return scene.camera.pixel_array.copy()


class TikzNativeOpenFaceVisibility3DReleaseBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]
        rig = analyze_geometry_rig_3d(cls.picture)
        cls.native_source = rig["nativeManimSourceV2"]["sourceText"]

    def setUp(self) -> None:
        self.config = tempconfig(
            {"renderer": "cairo", "pixel_width": 320, "pixel_height": 180}
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    @staticmethod
    def _style() -> OcclusionStyle:
        return OcclusionStyle(max_projected_length=12.0)

    def _static_figure(self):
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        return scene, figure

    def _installed_rig(self):
        namespace: dict[str, object] = {}
        exec(
            compile(self.native_source, "<native-manim-source-3d-v2>", "exec"),
            namespace,
        )
        scene, figure = self._static_figure()
        pristine_children = tuple(id(item) for item in figure.group.submobjects)
        pristine_style = _figure_style_snapshot(figure)
        pristine_scene_roots = tuple(id(item) for item in scene.mobjects)
        pristine_pixels = _cairo_pixels(scene)
        trackers = {
            driver_id: ValueTracker(initial)
            for driver_id, initial in namespace["DRIVER_INITIAL_VALUES"].items()
        }
        camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
        state = namespace["install_geometry_3d_updaters"](
            figure.group,
            figure.objects,
            trackers,
            camera,
        )
        return (
            namespace,
            scene,
            figure,
            trackers,
            state,
            pristine_children,
            pristine_style,
            pristine_scene_roots,
            pristine_pixels,
        )

    @staticmethod
    def _exercise_rig_frame(namespace, scene, trackers) -> None:
        hinge_driver = next(
            driver_id
            for driver_id, spec in namespace["DRIVER_SPECS"].items()
            if spec["type"] == "hinge_fold"
        )
        trackers[hinge_driver].set_value(
            namespace["DRIVER_SPECS"][hinge_driver]["range"][1]
        )
        for root in tuple(scene.mobjects):
            root.update(0.0)

    def test_one_native_figure_has_one_transactional_owner_and_session_releases_it(self) -> None:
        scene, figure = self._static_figure()
        pristine = _figure_style_snapshot(figure)
        first = bind_picture_open_face_visibility_3d(
            scene,
            self.picture,
            figure,
            style=self._style(),
        )
        second = bind_picture_open_face_visibility_3d(
            scene,
            self.picture,
            figure,
            style=self._style(),
        )
        try:
            first.attach()
            first_slots = first.controller.slot_snapshot()
            roots_before_rejection = tuple(id(item) for item in scene.mobjects)
            hidden_before_rejection = _figure_style_snapshot(figure)
            with self.assertRaisesRegex(
                TikzNativeOpenFaceVisibility3DManimError,
                "already has an attached open-face visibility binding",
            ):
                second.attach()
            self.assertTrue(first.controller.attached)
            self.assertFalse(second.controller.attached)
            self.assertEqual(first.controller.slot_snapshot(), first_slots)
            self.assertEqual(
                tuple(id(item) for item in scene.mobjects),
                roots_before_rejection,
            )
            self.assertEqual(_figure_style_snapshot(figure), hidden_before_rejection)

            first.restore()
            self.assertEqual(_figure_style_snapshot(figure), pristine)
            second.attach()
            self.assertTrue(second.controller.attached)
            second.restore()
            self.assertEqual(_figure_style_snapshot(figure), pristine)

            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                with first.session():
                    raise RuntimeError("sentinel")
            self.assertFalse(first.controller.attached)
            self.assertEqual(_figure_style_snapshot(figure), pristine)
            second.attach().restore()
            self.assertEqual(_figure_style_snapshot(figure), pristine)
        finally:
            second.restore()
            first.restore()

    def test_geometry_rig_and_binding_restore_orders_are_both_exact(self) -> None:
        for order in ("binding_then_rig", "rig_then_binding"):
            with self.subTest(order=order):
                (
                    namespace,
                    scene,
                    figure,
                    trackers,
                    state,
                    pristine_children,
                    pristine_style,
                    pristine_scene_roots,
                    pristine_pixels,
                ) = self._installed_rig()
                binding = bind_picture_open_face_visibility_3d(
                    scene,
                    self.picture,
                    figure,
                    geometry_rig_state=state,
                    style=self._style(),
                )
                binding.attach()
                self._exercise_rig_frame(namespace, scene, trackers)
                if order == "binding_then_rig":
                    binding.restore()
                    namespace["restore_geometry_3d_objects"](state)
                else:
                    namespace["restore_geometry_3d_objects"](state)
                    binding.restore()

                self.assertFalse(binding.controller.attached)
                self.assertNotIn(binding.controller.overlay_root, scene.mobjects)
                self.assertEqual(
                    tuple(id(item) for item in figure.group.submobjects),
                    pristine_children,
                )
                self.assertEqual(_figure_style_snapshot(figure), pristine_style)
                self.assertEqual(
                    tuple(id(item) for item in scene.mobjects),
                    pristine_scene_roots,
                )
                np.testing.assert_array_equal(_cairo_pixels(scene), pristine_pixels)

    def test_geometry_rig_rejects_explicit_projection_and_display_provider(self) -> None:
        conflicts = (
            {"projection": ParallelProjection.identity()},
            {"display_point_provider": lambda point: np.asarray(point, dtype=float)},
        )
        for extra in conflicts:
            with self.subTest(argument=next(iter(extra))):
                (
                    namespace,
                    scene,
                    figure,
                    _trackers,
                    state,
                    _pristine_children,
                    _pristine_style,
                    _pristine_scene_roots,
                    _pristine_pixels,
                ) = self._installed_rig()
                before_children = tuple(id(item) for item in figure.group.submobjects)
                before_style = _figure_style_snapshot(figure)
                before_roots = tuple(id(item) for item in scene.mobjects)
                try:
                    with self.assertRaisesRegex(
                        TikzNativeOpenFaceVisibility3DManimError,
                        "geometry_rig_state.*mutually exclusive",
                    ):
                        bind_picture_open_face_visibility_3d(
                            scene,
                            self.picture,
                            figure,
                            geometry_rig_state=state,
                            style=self._style(),
                            **extra,
                        )
                    self.assertEqual(
                        tuple(id(item) for item in figure.group.submobjects),
                        before_children,
                    )
                    self.assertEqual(_figure_style_snapshot(figure), before_style)
                    self.assertEqual(
                        tuple(id(item) for item in scene.mobjects),
                        before_roots,
                    )
                finally:
                    namespace["restore_geometry_3d_objects"](state)


if __name__ == "__main__":
    unittest.main()
