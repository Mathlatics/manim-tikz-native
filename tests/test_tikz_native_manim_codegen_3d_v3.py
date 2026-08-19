from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np
from manim import Dot, Scene, ValueTracker
from manim.animation.animation import prepare_animation

from tikz_native.compiler import compile_document
from tikz_native.geometry_rig_3d import analyze_geometry_rig_3d
from tikz_native.native_manim_codegen_3d_v3 import (
    NATIVE_MANIM_AUTHORING_3D_V2_SCHEMA,
    NATIVE_MANIM_OPEN_FACE_VISIBILITY_3D_SCHEMA,
    NATIVE_MANIM_SOURCE_3D_V3_SCHEMA,
    generate_native_manim_source_3d_v3,
)
from tikz_native.provider import instantiate_picture


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"


class _InstantScene(Scene):
    def play(self, *builders, **_kwargs) -> None:  # type: ignore[override]
        animations = [prepare_animation(builder) for builder in builders]
        for animation in animations:
            animation.begin()
        for alpha in (0.25, 0.5, 0.75, 1.0):
            for animation in animations:
                animation.interpolate(alpha)
            for mobject in self.mobjects:
                mobject.update(0.0)
        for animation in animations:
            animation.finish()


class TikzNativeReadableManim3DV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]
        cls.rig = analyze_geometry_rig_3d(cls.picture)
        cls.payload = generate_native_manim_source_3d_v3(cls.picture, cls.rig)

    def _namespace(self):
        namespace: dict[str, object] = {}
        exec(
            compile(self.payload["sourceText"], "<native-manim-source-3d-v3>", "exec"),
            namespace,
        )
        return namespace

    def _figure(self):
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        figure.group.scale(0.72).rotate(0.13).shift((1.1, -0.45, 0.0))
        return figure

    def test_payload_is_self_contained_versioned_and_readable(self) -> None:
        payload = self.payload
        self.assertEqual(
            set(payload),
            {"schema", "sourceText", "sourceSha256", "authoringSpec", "visibilitySpec"},
        )
        self.assertEqual(payload["schema"], NATIVE_MANIM_SOURCE_3D_V3_SCHEMA)
        self.assertEqual(
            payload["sourceSha256"],
            hashlib.sha256(payload["sourceText"].encode("utf-8")).hexdigest(),
        )
        self.assertTrue(
            payload["sourceText"].startswith(
                "from __future__ import annotations\n"
                "import numpy as np\n"
                "from manim import *\n"
            )
        )
        for forbidden in (
            "import tikz_native",
            "from tikz_native",
            "import polyhedron_visibility",
            "from polyhedron_visibility",
            "bind_picture_open_face_visibility_3d",
            "/Users/",
        ):
            self.assertNotIn(forbidden, payload["sourceText"])
        self.assertIn("def compute_open_face_visibility_3d", payload["sourceText"])
        self.assertIn("def compute_open_face_fill_order_3d", payload["sourceText"])
        self.assertIn("def install_open_face_visibility_3d", payload["sourceText"])
        self.assertIn("def restore_open_face_visibility_3d", payload["sourceText"])
        self.assertEqual(
            payload["authoringSpec"]["schema"],
            NATIVE_MANIM_AUTHORING_3D_V2_SCHEMA,
        )
        self.assertEqual(
            payload["visibilitySpec"]["schema"],
            NATIVE_MANIM_OPEN_FACE_VISIBILITY_3D_SCHEMA,
        )
        self.assertEqual(payload["visibilitySpec"]["faceCount"], 2)
        self.assertEqual(payload["visibilitySpec"]["strokeCount"], 9)
        self.assertEqual(payload["visibilitySpec"]["seamCount"], 1)
        self.assertTrue(
            payload["visibilitySpec"]["requiresExplicitStaticAssetRecompile"]
        )

    def test_signed_hinge_maps_author_values_zero_and_pi_to_coplanarity(self) -> None:
        namespace = self._namespace()
        driver_id = "hinge_fold:fold-angle"
        self.assertEqual(namespace["HINGE_ORIENTATION_SIGNS"][driver_id], -1.0)
        self.assertEqual(
            namespace["DRIVER_SPECS"][driver_id]["range"],
            (0.0, np.pi),
        )
        initial = namespace["DRIVER_INITIAL_VALUES"].copy()
        for target, expected_dot in ((0.0, 1.0), (np.pi, -1.0)):
            values = dict(initial)
            values[driver_id] = target
            coordinates = namespace["geometry_coordinates_3d"](values)
            fixed = np.cross(
                coordinates["B"] - coordinates["A"],
                coordinates["Alpha1"] - coordinates["A"],
            )
            moving = np.cross(
                coordinates["B"] - coordinates["A"],
                coordinates["Beta1"] - coordinates["A"],
            )
            fixed /= np.linalg.norm(fixed)
            moving /= np.linalg.norm(moving)
            self.assertAlmostEqual(float(np.dot(fixed, moving)), expected_dot, places=10)

    def test_real_scene_updates_global_slots_and_restores_every_identity(self) -> None:
        namespace = self._namespace()
        figure = self._figure()
        shape = figure.group
        entry_points = {
            object_id: item.get_all_points().copy()
            for object_id, item in figure.objects.items()
        }
        entry_children = tuple(id(item) for item in shape.submobjects)
        face_object_ids = tuple(
            item["object_id"] for item in namespace["OPEN_FACE_FACE_BINDINGS"]
        )
        entry_fill_opacities = {
            object_id: figure.objects[object_id].get_fill_opacity()
            for object_id in face_object_ids
        }
        entry_roots = None
        trackers = {
            driver_id: ValueTracker(initial)
            for driver_id, initial in namespace["DRIVER_INITIAL_VALUES"].items()
        }
        camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
        scene = _InstantScene()
        scene.add(shape)
        entry_roots = tuple(id(item) for item in scene.mobjects)
        geometry = namespace["install_geometry_3d_updaters"](
            shape, figure.objects, trackers, camera
        )
        visibility = namespace["install_open_face_visibility_3d"](
            scene, shape, figure.objects, geometry
        )
        probe_id = "stroke.E.S.e67987f11d"
        self.assertEqual(
            tuple(item[2] for item in visibility["last_spans"][probe_id]),
            ("visible", "hidden", "visible"),
        )
        self.assertEqual(
            {
                object_id: figure.objects[object_id].get_fill_opacity()
                for object_id in face_object_ids
            },
            {object_id: 0.0 for object_id in face_object_ids},
        )
        self.assertEqual(
            [
                float(visibility["face_proxies"][face_id].z_index)
                for face_id in visibility["last_face_order"]
            ],
            list(visibility["face_z_slots"]),
        )
        slot_ids = tuple(
            id(item) for item in visibility["overlay_root"].get_family()
        )
        face_proxy_ids = tuple(
            id(item) for item in visibility["face_proxies"].values()
        )

        scene.play(
            trackers["hinge_fold:fold-angle"].animate.set_value(0.0),
            run_time=0.1,
        )
        self.assertEqual(
            visibility["last_spans"][probe_id], ((0.0, 1.0, "visible"),)
        )
        scene.play(
            trackers["hinge_fold:fold-angle"].animate.set_value(np.pi),
            run_time=0.1,
        )
        self.assertEqual(
            visibility["last_spans"][probe_id], ((0.0, 1.0, "visible"),)
        )
        namespace["prepare_local_camera"](
            geometry, "side", transition="linear", arc_height=0.2
        )
        scene.play(camera.animate.set_value(1.0), run_time=0.1)
        self.assertEqual(
            slot_ids,
            tuple(id(item) for item in visibility["overlay_root"].get_family()),
        )
        self.assertEqual(
            face_proxy_ids,
            tuple(id(item) for item in visibility["face_proxies"].values()),
        )

        namespace["restore_open_face_visibility_3d"](visibility)
        namespace["restore_geometry_3d_objects"](geometry)
        self.assertEqual(entry_roots, tuple(id(item) for item in scene.mobjects))
        self.assertEqual(entry_children, tuple(id(item) for item in shape.submobjects))
        self.assertEqual(
            {
                object_id: figure.objects[object_id].get_fill_opacity()
                for object_id in face_object_ids
            },
            entry_fill_opacities,
        )
        for object_id, points in entry_points.items():
            np.testing.assert_allclose(
                figure.objects[object_id].get_all_points(),
                points,
                atol=1.0e-10,
                err_msg=object_id,
            )

    def test_install_failure_is_transactional_and_releases_owner(self) -> None:
        namespace = self._namespace()
        figure = self._figure()
        scene = _InstantScene()
        scene.add(figure.group)
        trackers = {
            driver_id: ValueTracker(initial)
            for driver_id, initial in namespace["DRIVER_INITIAL_VALUES"].items()
        }
        geometry = namespace["install_geometry_3d_updaters"](
            figure.group,
            figure.objects,
            trackers,
            ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"]),
        )
        original = namespace["OPEN_FACE_ENTRY_SPANS"]
        namespace["OPEN_FACE_ENTRY_SPANS"] = {"wrong": ()}
        roots = tuple(id(item) for item in scene.mobjects)
        with self.assertRaisesRegex(RuntimeError, "entry trace"):
            namespace["install_open_face_visibility_3d"](
                scene, figure.group, figure.objects, geometry
            )
        self.assertEqual(roots, tuple(id(item) for item in scene.mobjects))
        self.assertFalse(
            hasattr(figure.group, "_mathppt_open_face_visibility_owner")
        )
        namespace["OPEN_FACE_ENTRY_SPANS"] = original
        visibility = namespace["install_open_face_visibility_3d"](
            scene, figure.group, figure.objects, geometry
        )
        namespace["restore_open_face_visibility_3d"](visibility)
        namespace["restore_geometry_3d_objects"](geometry)

    def test_unmanaged_drawable_inside_face_z_band_fails_transactionally(self) -> None:
        namespace = self._namespace()
        figure = self._figure()
        scene = _InstantScene()
        scene.add(figure.group, Dot().set_z_index(11.5))
        trackers = {
            driver_id: ValueTracker(initial)
            for driver_id, initial in namespace["DRIVER_INITIAL_VALUES"].items()
        }
        geometry = namespace["install_geometry_3d_updaters"](
            figure.group,
            figure.objects,
            trackers,
            ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"]),
        )
        roots = tuple(id(item) for item in scene.mobjects)

        with self.assertRaisesRegex(RuntimeError, "face fill z band"):
            namespace["install_open_face_visibility_3d"](
                scene, figure.group, figure.objects, geometry
            )

        self.assertEqual(roots, tuple(id(item) for item in scene.mobjects))
        self.assertFalse(
            hasattr(figure.group, "_mathppt_open_face_visibility_owner")
        )
        namespace["restore_geometry_3d_objects"](geometry)


if __name__ == "__main__":
    unittest.main()
