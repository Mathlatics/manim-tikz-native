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
from tikz_native.native_manim_codegen_3d_v2 import (
    point_on_segment_driver_candidates,
)
from tikz_native.provider import instantiate_picture


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"
SELECTION = {
    "candidate_id": "hinge_fold:fold-angle",
    "range": [0.3141592653589793, 1.9547687622336491],
}


class _InstantScene(Scene):
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


class TikzNativeReadableManim3DV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]

    def _rig(self, *, selected: bool = False, picture=None):
        result = analyze_geometry_rig_3d(
            self.picture if picture is None else picture,
            selection=SELECTION if selected else None,
        )
        self.assertIn(result["status"], {"needs_selection", "ready"}, result["diagnostics"])
        return result

    def _namespace(self, *, selected: bool = False):
        rig = self._rig(selected=selected)
        payload = rig["nativeManimSourceV2"]
        self.assertIsNotNone(payload, rig["diagnostics"])
        namespace: dict[str, object] = {}
        exec(compile(payload["sourceText"], "<native-manim-source-3d-v2>", "exec"), namespace)
        return rig, payload, namespace

    def _figure(self):
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        figure.group.scale(0.72).rotate(0.13).shift((1.1, -0.45, 0.0))
        return figure

    def _trackers(self, namespace):
        return {
            driver_id: ValueTracker(initial)
            for driver_id, initial in namespace["DRIVER_INITIAL_VALUES"].items()
        }

    def test_discovery_returns_v2_source_without_choosing_an_animation(self) -> None:
        rig, payload, namespace = self._namespace()
        self.assertEqual(rig["status"], "needs_selection")
        self.assertIsNone(rig["nativeManimSource"])
        self.assertEqual(
            set(payload),
            {"schema", "sourceText", "sourceSha256", "authoringSpec"},
        )
        self.assertEqual(payload["schema"], "tikz-native-manim-source-3d/v2")
        self.assertEqual(
            payload["sourceSha256"],
            hashlib.sha256(payload["sourceText"].encode("utf-8")).hexdigest(),
        )
        self.assertTrue(payload["sourceText"].startswith("import numpy as np\nfrom manim import *\n"))
        for forbidden in (
            "import tikz_native",
            "from tikz_native",
            "play_motion_3d_on_native_shape",
            "Motion3DSpec",
            "/Users/",
        ):
            self.assertNotIn(forbidden, payload["sourceText"])
        self.assertIn(
            "def install_geometry_3d_updaters(shape, objects, driver_trackers, camera_progress):",
            payload["sourceText"],
        )
        self.assertIn("DRIVER_SPECS", namespace)

    def test_authoring_spec_and_motion_candidates_cross_reference_exact_drivers(self) -> None:
        rig, payload, _namespace = self._namespace()
        authoring = payload["authoringSpec"]
        self.assertEqual(
            set(authoring),
            {"schema", "drivers", "entryCamera", "cameraModes", "endPolicy"},
        )
        self.assertEqual(authoring["schema"], "tikz-native-manim-authoring-3d/v1")
        self.assertEqual(authoring["endPolicy"], "restore_entry")
        self.assertEqual(authoring["entryCamera"], {"mode": "tikz", "orthogonal": True})
        cameras = {item["mode"]: item for item in authoring["cameraModes"]}
        self.assertFalse(cameras["oblique"]["orthogonal"])
        self.assertEqual(cameras["oblique"]["transitionTypes"], ["linear"])
        self.assertTrue(cameras["side"]["orthogonal"])
        self.assertEqual(cameras["side"]["transitionTypes"], ["linear", "orbit"])

        drivers = {item["driverId"]: item for item in authoring["drivers"]}
        self.assertEqual(set(drivers), {"hinge_fold:fold-angle", "point_on_segment:M"})
        hinge = drivers["hinge_fold:fold-angle"]
        self.assertEqual(
            set(hinge),
            {
                "driverId",
                "candidateId",
                "type",
                "pythonName",
                "initial",
                "range",
                "unit",
                "axis",
            },
        )
        self.assertEqual(hinge["candidateId"], hinge["driverId"])
        self.assertEqual(hinge["type"], "hinge_fold")
        self.assertEqual(hinge["axis"], ["A", "B"])
        self.assertEqual(hinge["unit"], "radians")
        point = drivers["point_on_segment:M"]
        self.assertEqual(
            set(point),
            {
                "driverId",
                "candidateId",
                "type",
                "pythonName",
                "initial",
                "range",
                "unit",
                "coordinateId",
                "segment",
            },
        )
        self.assertEqual(point["candidateId"], point["driverId"])
        self.assertEqual(point["type"], "point_on_segment")
        self.assertEqual(point["coordinateId"], "M")
        self.assertEqual(point["segment"], ["Beta0", "Beta1"])
        self.assertEqual(point["initial"], 0.67)
        self.assertEqual(point["range"], [0.0, 1.0])
        self.assertRegex(point["pythonName"], r"^[A-Za-z_][A-Za-z0-9_]*$")

        candidates = {
            item["driverId"]: item
            for item in rig["motionCandidates"]
            if item["candidateKind"] == "geometry_driver"
        }
        self.assertEqual(set(candidates), set(drivers))
        self.assertEqual(candidates["point_on_segment:M"]["affectedCoordinates"], ["M", "N"])
        self.assertEqual(candidates["point_on_segment:M"]["initial"], {"value": 0.67, "unit": "ratio"})

    def test_v1_source_is_byte_stable_when_selected(self) -> None:
        rig = self._rig(selected=True)
        self.assertEqual(rig["status"], "ready")
        self.assertEqual(
            rig["nativeManimSource"]["sourceSha256"],
            "734d692c78e90a61ff706721a3f7c4c4297c5b1fb8a6dd7223624df24d399d84",
        )
        self.assertIsNotNone(rig["nativeManimSourceV2"])

    def test_point_driver_is_logical_segment_math_and_composes_with_hinge(self) -> None:
        _rig, _payload, namespace = self._namespace()
        initial = dict(namespace["DRIVER_INITIAL_VALUES"])
        first = namespace["geometry_coordinates_3d"](initial)
        point_only = dict(initial)
        point_only["point_on_segment:M"] = 0.2
        second = namespace["geometry_coordinates_3d"](point_only)
        np.testing.assert_allclose(second["Beta0"], first["Beta0"], atol=1e-12)
        np.testing.assert_allclose(second["Beta1"], first["Beta1"], atol=1e-12)
        np.testing.assert_allclose(
            second["M"],
            second["Beta0"] + 0.2 * (second["Beta1"] - second["Beta0"]),
            atol=1e-12,
        )
        self.assertFalse(np.allclose(second["M"], first["M"]))
        self.assertFalse(np.allclose(second["N"], first["N"]))

        combined = dict(initial)
        combined["hinge_fold:fold-angle"] = namespace["DRIVER_SPECS"]["hinge_fold:fold-angle"]["range"][1]
        combined["point_on_segment:M"] = 0.8
        third = namespace["geometry_coordinates_3d"](combined)
        np.testing.assert_allclose(
            third["M"],
            third["Beta0"] + 0.8 * (third["Beta1"] - third["Beta0"]),
            atol=1e-12,
        )
        axis = third["B"] - third["A"]
        np.testing.assert_allclose(np.dot(axis, third["M"] - third["N"]), 0.0, atol=1e-11)

    def test_real_scene_supports_simultaneous_play_stable_occlusion_and_exact_restore(self) -> None:
        _rig, _payload, namespace = self._namespace()
        figure = self._figure()
        shape = figure.group
        entry_points = {object_id: item.get_all_points().copy() for object_id, item in figure.objects.items()}
        original_children = tuple(id(item) for item in shape.submobjects)
        trackers = self._trackers(namespace)
        camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
        scene = _InstantScene()
        scene.add(shape)
        state = namespace["install_geometry_3d_updaters"](shape, figure.objects, trackers, camera)
        self.assertTrue(namespace["assert_shape_state_3d_entry"](state))
        slot_ids = [tuple(id(item) for item in group.submobjects) for group in state["temporary_groups"]]
        fixed_entry = figure.objects["dot.A"].get_center().copy()
        moving_face_id = "plane_interaction_fill.A.B.Beta1.Beta0"
        moving_face_entry = figure.objects[moving_face_id].get_all_points().copy()
        m_entry = figure.objects["dot.M"].get_center().copy()
        n_entry = figure.objects["dot.N"].get_center().copy()
        line_entry = figure.objects["line.M.N"].get_all_points().copy()
        label_entry = figure.objects["label.M.M"].get_center().copy()

        scene.play(
            trackers["hinge_fold:fold-angle"].animate.set_value(
                namespace["DRIVER_SPECS"]["hinge_fold:fold-angle"]["range"][1]
            ),
            trackers["point_on_segment:M"].animate.set_value(0.15),
            run_time=0.1,
        )
        self.assertFalse(
            np.allclose(figure.objects[moving_face_id].get_all_points(), moving_face_entry)
        )
        self.assertFalse(np.allclose(figure.objects["dot.M"].get_center(), m_entry))
        self.assertFalse(np.allclose(figure.objects["dot.N"].get_center(), n_entry))
        self.assertFalse(
            np.allclose(figure.objects["line.M.N"].get_all_points(), line_entry)
        )
        self.assertFalse(
            np.allclose(figure.objects["label.M.M"].get_center(), label_entry)
        )
        self.assertEqual(
            slot_ids,
            [tuple(id(item) for item in group.submobjects) for group in state["temporary_groups"]],
        )
        namespace["prepare_local_camera"](state, "side", transition="linear")
        scene.play(camera.animate.set_value(1.0), run_time=0.1)
        self.assertFalse(
            np.allclose(figure.objects["dot.A"].get_center(), fixed_entry)
        )

        namespace["restore_geometry_3d_objects"](state)
        self.assertEqual(original_children, tuple(id(item) for item in shape.submobjects))
        for object_id, entry in entry_points.items():
            np.testing.assert_allclose(
                figure.objects[object_id].get_all_points(),
                entry,
                atol=1e-10,
                err_msg=object_id,
            )
            self.assertFalse(figure.objects[object_id].updaters, object_id)

    def test_install_rejects_incomplete_or_unknown_driver_tracker_maps(self) -> None:
        _rig, _payload, namespace = self._namespace()
        figure = self._figure()
        camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
        trackers = self._trackers(namespace)
        trackers.pop("point_on_segment:M")
        with self.assertRaisesRegex(RuntimeError, "driver tracker map mismatch"):
            namespace["install_geometry_3d_updaters"](
                figure.group, figure.objects, trackers, camera
            )

    def test_entry_install_is_pixel_identical(self) -> None:
        _rig, _payload, namespace = self._namespace()
        with tempconfig({"pixel_width": 640, "pixel_height": 360, "frame_rate": 5}):
            figure = self._figure()
            scene = Scene()
            scene.add(figure.group)
            scene.camera.reset()
            scene.camera.capture_mobjects(scene.mobjects)
            before = scene.camera.pixel_array.copy()
            state = namespace["install_geometry_3d_updaters"](
                figure.group,
                figure.objects,
                self._trackers(namespace),
                ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"]),
            )
            figure.group.update(0.0)
            scene.camera.reset()
            scene.camera.capture_mobjects(scene.mobjects)
            np.testing.assert_array_equal(scene.camera.pixel_array, before)
            namespace["restore_geometry_3d_objects"](state)

    def test_invalid_segment_cycle_and_multiple_hinges_fail_closed_only_for_v2(self) -> None:
        illegal = deepcopy(self.picture)
        illegal.coordinate_dependencies["M"]["parameter"] = 1.5
        candidates = point_on_segment_driver_candidates(illegal)
        self.assertEqual(candidates[0]["status"], "blocked")
        illegal_rig = self._rig(picture=illegal)
        self.assertIsNone(illegal_rig["nativeManimSourceV2"])
        self.assertIn(
            "NATIVE_MANIM_SOURCE_V2_UNAVAILABLE",
            {item["code"] for item in illegal_rig["diagnostics"]},
        )

        unknown_segment = deepcopy(self.picture)
        unknown_segment.coordinate_dependencies["M"]["start"] = "MissingPoint"
        unknown_candidates = point_on_segment_driver_candidates(unknown_segment)
        self.assertEqual(unknown_candidates[0]["status"], "blocked")
        self.assertIn("unknown segment start", unknown_candidates[0]["reason"])

        cyclic = deepcopy(self.picture)
        cyclic.coordinate_dependencies["M"]["start"] = "N"
        cyclic_candidates = point_on_segment_driver_candidates(cyclic)
        self.assertEqual(cyclic_candidates[0]["status"], "blocked")
        self.assertIn("dependency cycle", cyclic_candidates[0]["reason"])
        cyclic_rig = self._rig(picture=cyclic)
        self.assertIsNone(cyclic_rig["nativeManimSourceV2"])
        diagnostic = next(
            item
            for item in cyclic_rig["diagnostics"]
            if item["code"] == "NATIVE_MANIM_SOURCE_V2_UNAVAILABLE"
        )
        self.assertIn("dependency cycle", diagnostic["message"])

        ambiguous = deepcopy(self.picture)
        second_hinge = deepcopy(ambiguous.hinge_relations[0])
        second_hinge.id = "second-fold"
        ambiguous.hinge_relations.append(second_hinge)
        ambiguous_rig = self._rig(picture=ambiguous)
        self.assertIsNone(ambiguous_rig["nativeManimSourceV2"])
        diagnostic = next(
            item
            for item in ambiguous_rig["diagnostics"]
            if item["code"] == "NATIVE_MANIM_SOURCE_V2_UNAVAILABLE"
        )
        self.assertIn("exactly one explicit hinge", diagnostic["message"])


if __name__ == "__main__":
    unittest.main()
