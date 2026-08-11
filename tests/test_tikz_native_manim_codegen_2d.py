from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import ValueTracker, tempconfig

from tikz_native.compiler import compile_document
from tikz_native.geometry_rig import analyze_geometry_rig
from tikz_native.manim_renderer import NativeManimRenderer
from tikz_native.native_manim_codegen_2d import (
    NATIVE_MANIM_SOURCE_2D_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "analytic_geometry_ellipse_demo" / "ellipse_problem.tex"


class TikzNativeManimCodegen2DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]
        cls.media_directory = TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.media_directory.cleanup()

    def _rig(self, *, exclude: list[str] | None = None) -> dict:
        recommendation = analyze_geometry_rig(
            self.picture,
            "line.Lstart.Lend",
        )["selectedDriver"]
        selection = {
            "candidate_id": recommendation["candidateId"],
            "pivot": recommendation["pivot"],
            "range": recommendation["range"],
        }
        if exclude:
            selection["exclude_object_ids"] = exclude
        return analyze_geometry_rig(
            self.picture,
            "line.Lstart.Lend",
            selection=selection,
        )

    def test_ready_rig_emits_hashed_readable_native_manim_source(self) -> None:
        rig = self._rig()
        self.assertEqual(rig["status"], "ready")
        payload = rig["nativeManimSource"]
        self.assertEqual(payload["schema"], NATIVE_MANIM_SOURCE_2D_SCHEMA)
        source = payload["sourceText"]
        self.assertEqual(
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
            payload["sourceSha256"],
        )
        compile(source, "<native-manim-source-2d>", "exec")

        self.assertIn("from manim import *", source)
        self.assertIn("def geometry_coordinates(theta):", source)
        self.assertIn("def install_geometry_updaters(objects, parameter):", source)
        self.assertIn(".add_updater(", source)
        self.assertIn(".put_start_and_end_on(", source)
        self.assertIn("updated = Polygon(", source)
        self.assertIn("updated = Arc(", source)
        self.assertIn("def match_geometry_style(mobject, template):", source)
        self.assertNotIn("NativeGeometryRig2D", source)
        self.assertNotIn("tikz_native", source)
        self.assertNotIn("hashlib", source)
        self.assertNotIn("pathlib", source)
        self.assertNotIn(".motion.json", source)
        self.assertNotIn(".source.tex", source)

    def test_excluded_follower_is_absent_from_generated_updaters(self) -> None:
        source = self._rig(exclude=["label.R.R"])["nativeManimSource"][
            "sourceText"
        ]
        namespace: dict[str, object] = {}
        exec(compile(source, "<native-manim-source-2d>", "exec"), namespace)
        self.assertNotIn("label.R.R", namespace["DYNAMIC_OBJECT_IDS"])
        self.assertIn("label.R.R", namespace["DISABLED_OBJECT_IDS"])
        self.assertNotIn("objects['label.R.R'].add_updater", source)

    def test_updaters_preserve_transformed_entry_move_and_restore(self) -> None:
        source = self._rig()["nativeManimSource"]["sourceText"]
        namespace: dict[str, object] = {}
        exec(compile(source, "<native-manim-source-2d>", "exec"), namespace)

        with tempconfig({"media_dir": self.media_directory.name}):
            figure = NativeManimRenderer(scene_unit_per_cm=1.0).render(self.picture)
            figure.group.scale(0.63).rotate(0.27).shift(
                np.array([1.2, -0.4, 0.0])
            )
            entry = {
                object_id: mobject.copy()
                for object_id, mobject in figure.objects.items()
            }
            tracker = ValueTracker(namespace["PARAMETER_INITIAL"])
            state = namespace["install_geometry_updaters"](
                figure.objects,
                tracker,
            )
            figure.group.update(0)

            for object_id in namespace["DYNAMIC_OBJECT_IDS"]:
                np.testing.assert_allclose(
                    figure.objects[object_id].get_all_points(),
                    entry[object_id].get_all_points(),
                    atol=1e-9,
                    err_msg=object_id,
                )

            entry_p = figure.objects["dot.P"].get_center().copy()
            tracker.set_value(namespace["PARAMETER_MAXIMUM"])
            figure.group.update(0)
            self.assertFalse(
                np.allclose(figure.objects["dot.P"].get_center(), entry_p)
            )
            for object_id in ("fill.P.Q.R", "fill.P.F.O", "angle.R.Q.P"):
                current_family = figure.objects[object_id].get_family()
                entry_family = entry[object_id].get_family()
                self.assertEqual(len(current_family), len(entry_family))
                for current, original in zip(current_family, entry_family):
                    for attribute in (
                        "stroke_rgbas",
                        "fill_rgbas",
                        "background_stroke_rgbas",
                        "stroke_width",
                        "background_stroke_width",
                    ):
                        np.testing.assert_allclose(
                            getattr(current, attribute),
                            getattr(original, attribute),
                            atol=1e-9,
                            err_msg=f"{object_id}:{attribute}",
                        )
                    self.assertEqual(current.joint_type, original.joint_type)
                    self.assertEqual(current.cap_style, original.cap_style)

            namespace["restore_geometry_objects"](state)
            for object_id, mobject in figure.objects.items():
                np.testing.assert_allclose(
                    mobject.get_all_points(),
                    entry[object_id].get_all_points(),
                    atol=1e-9,
                    err_msg=object_id,
                )
                self.assertEqual(list(mobject.updaters), [])

    def test_partial_updater_install_failure_restores_every_object(self) -> None:
        source = self._rig()["nativeManimSource"]["sourceText"]
        namespace: dict[str, object] = {}
        exec(compile(source, "<native-manim-source-2d>", "exec"), namespace)

        with tempconfig({"media_dir": self.media_directory.name}):
            figure = NativeManimRenderer(scene_unit_per_cm=1.0).render(self.picture)
            entry = {
                object_id: mobject.get_all_points().copy()
                for object_id, mobject in figure.objects.items()
            }
            tracker = ValueTracker(namespace["PARAMETER_INITIAL"])
            with patch.object(
                figure.objects["line.P.Q"],
                "add_updater",
                side_effect=RuntimeError("injected updater failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected updater failure"):
                    namespace["install_geometry_updaters"](
                        figure.objects,
                        tracker,
                    )

            for object_id, mobject in figure.objects.items():
                np.testing.assert_allclose(
                    mobject.get_all_points(),
                    entry[object_id],
                    atol=1e-9,
                    err_msg=object_id,
                )
                self.assertEqual(list(mobject.updaters), [])


if __name__ == "__main__":
    unittest.main()
