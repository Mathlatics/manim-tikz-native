from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from manim import (
    CapStyleType,
    LineJointType,
    Polygon,
    Scene,
    VGroup,
    ValueTracker,
    tempconfig,
)

from polyhedron_visibility.painter_band import scene_painter_band_allocations
from tikz_native.compiler import compile_document
from tikz_native.geometry_rig_3d import analyze_geometry_rig_3d
from tikz_native.native_manim_codegen_3d_v3 import generate_native_manim_source_3d_v3
from tikz_native.provider import instantiate_picture
from tikz_native.source_project import (
    SOURCE_PROJECT_SCHEMA_VERSION,
    PainterZBand,
    rewrite_generated_source,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"


class SourceProjectRealV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]
        rig = analyze_geometry_rig_3d(cls.picture)
        source = generate_native_manim_source_3d_v3(cls.picture, rig)["sourceText"]
        cls.rewritten = rewrite_generated_source(
            source,
            paint_policy="diagrammatic",
            painter_z_band=PainterZBand(10_000.0, 11_024.0),
        )

    def _prepare_source(self, scene: Scene, source: str):
        namespace: dict[str, object] = {}
        exec(compile(source, "<source-project-v3>", "exec"), namespace)
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        scene.add(figure.group)
        trackers = {
            key: ValueTracker(value)
            for key, value in namespace["DRIVER_INITIAL_VALUES"].items()
        }
        geometry = namespace["install_geometry_3d_updaters"](
            figure.group,
            figure.objects,
            trackers,
            ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"]),
        )
        return namespace, figure, geometry

    def _install_source(self, scene: Scene, source: str):
        namespace, figure, geometry = self._prepare_source(scene, source)
        controller = namespace["install_open_face_visibility_3d"](
            scene, figure.group, figure.objects, geometry
        )
        return namespace, figure, geometry, controller

    def _install(self, scene: Scene):
        return self._install_source(scene, self.rewritten)

    def test_real_v3_rewrite_is_idempotent_and_runs_unified_frame(self) -> None:
        self.assertEqual(
            self.rewritten,
            rewrite_generated_source(
                self.rewritten,
                paint_policy="diagrammatic",
                painter_z_band=PainterZBand(10_000.0, 11_024.0),
            ),
        )
        scene = Scene()
        namespace, figure, geometry, controller = self._install(scene)
        self.assertEqual(controller.compositing_mode, "unified")
        self.assertIsNotNone(controller.last_unified_frame)
        controller.update(0.0)
        self.assertIsNotNone(controller.last_unified_frame)
        namespace["restore_open_face_visibility_3d"](controller)
        namespace["restore_geometry_3d_objects"](geometry)
        self.assertFalse(
            hasattr(figure.group, "_mathppt_open_face_visibility_owner")
        )

    def test_generated_styles_preserve_hidden_cap_join_and_legacy_fallback(self) -> None:
        scene = Scene()
        namespace, figure, geometry = self._prepare_source(scene, self.rewritten)
        bindings = namespace["OPEN_FACE_BINDINGS"]
        explicit = bindings[0]
        fallback = bindings[1]
        explicit["visible_style"] = {
            **explicit["visible_style"],
            "line_cap": "butt",
            "line_join": "miter",
        }
        explicit["hidden_style"] = {
            **explicit["hidden_style"],
            "line_cap": "round",
            "line_join": "bevel",
        }
        fallback["visible_style"] = {
            **fallback["visible_style"],
            "line_cap": "square",
            "line_join": "bevel",
        }
        fallback["hidden_style"] = dict(fallback["hidden_style"])
        fallback["hidden_style"].pop("line_cap", None)
        fallback["hidden_style"].pop("line_join", None)

        controller = namespace["install_open_face_visibility_3d"](
            scene, figure.group, figure.objects, geometry
        )
        runtime = controller._unified_runtime
        assert runtime is not None

        explicit_slots = runtime.path_slots[explicit["source_edge_id"]].fragments
        self.assertTrue(
            all(slot.solid.cap_style == CapStyleType.BUTT for slot in explicit_slots)
        )
        self.assertTrue(
            all(
                line.cap_style == CapStyleType.ROUND
                and line.joint_type == LineJointType.BEVEL
                for slot in explicit_slots
                for line in slot.dashes
            )
        )
        fallback_slots = runtime.path_slots[fallback["source_edge_id"]].fragments
        self.assertTrue(
            all(
                line.cap_style == CapStyleType.SQUARE
                and line.joint_type == LineJointType.BEVEL
                for slot in fallback_slots
                for line in slot.dashes
            )
        )
        namespace["restore_open_face_visibility_3d"](controller)
        namespace["restore_geometry_3d_objects"](geometry)

    def test_two_generated_assets_receive_non_overlapping_scene_bands(self) -> None:
        scene = Scene()
        first = self._install(scene)
        second = self._install(scene)
        first_band = first[3].painter_z_band
        second_band = second[3].painter_z_band
        self.assertTrue(
            first_band[1] < second_band[0] or second_band[1] < first_band[0]
        )
        for namespace, _figure, geometry, controller in (second, first):
            namespace["restore_open_face_visibility_3d"](controller)
            namespace["restore_geometry_3d_objects"](geometry)

    def test_scene_band_registry_releases_reuses_and_reallocates(self) -> None:
        scene = Scene()
        first = self._install(scene)
        second = self._install(scene)
        first_preferred = first[3].preferred_painter_z_band
        first[0]["restore_open_face_visibility_3d"](first[3])

        third = self._install(scene)
        self.assertEqual(third[3].painter_z_band, first_preferred)
        first[3].attach()
        bands = [
            first[3].painter_z_band,
            second[3].painter_z_band,
            third[3].painter_z_band,
        ]
        for index, band in enumerate(bands):
            for other in bands[index + 1 :]:
                self.assertTrue(band[1] < other[0] or other[1] < band[0])

        for namespace, _figure, geometry, controller in (first, second, third):
            namespace["restore_open_face_visibility_3d"](controller)
            namespace["restore_geometry_3d_objects"](geometry)
        self.assertEqual(scene_painter_band_allocations(scene), ())

    def test_controller_construction_failure_releases_scene_band(self) -> None:
        from tikz_native import generated_open_face_visibility_3d as adapter

        scene = Scene()
        namespace, figure, geometry = self._prepare_source(
            scene, self.rewritten
        )
        with mock.patch.object(
            adapter,
            "GeneratedOpenFaceVisibility3D",
            side_effect=RuntimeError("constructor failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "constructor failure"):
                namespace["install_open_face_visibility_3d"](
                    scene, figure.group, figure.objects, geometry
                )
        self.assertEqual(scene_painter_band_allocations(scene), ())
        namespace["restore_geometry_3d_objects"](geometry)

    def test_reattach_band_configuration_failure_releases_scene_band(self) -> None:
        scene = Scene()
        namespace, _figure, geometry, controller = self._install(scene)
        namespace["restore_open_face_visibility_3d"](controller)
        self.assertEqual(scene_painter_band_allocations(scene), ())

        with mock.patch.object(
            controller,
            "set_painter_z_band",
            side_effect=RuntimeError("band configuration failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "band configuration failure"):
                controller.attach()
        self.assertEqual(scene_painter_band_allocations(scene), ())
        namespace["restore_geometry_3d_objects"](geometry)

    def test_attach_preparation_failure_rolls_back_proxy_and_band(self) -> None:
        scene = Scene()
        namespace, figure, geometry = self._prepare_source(
            scene, self.rewritten
        )
        scene_before = tuple(scene.mobjects)
        namespace["_open_face_detach_static_entry"] = mock.Mock(
            side_effect=RuntimeError("detach failure")
        )
        with self.assertRaisesRegex(RuntimeError, "detach failure"):
            namespace["install_open_face_visibility_3d"](
                scene, figure.group, figure.objects, geometry
            )
        self.assertEqual(
            tuple(map(id, scene.mobjects)), tuple(map(id, scene_before))
        )
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertFalse(
            hasattr(figure.group, "_mathppt_open_face_visibility_owner")
        )
        namespace["restore_geometry_3d_objects"](geometry)

    def test_face_only_generated_asset_uses_and_releases_a_band(self) -> None:
        from tikz_native.generated_open_face_visibility_3d import (
            install_generated_open_face_visibility_3d,
        )

        scene = Scene()
        face = Polygon((0, 0, 0), (2, 0, 0), (0, 2, 0))
        shape = VGroup(face)
        scene.add(shape)
        positions = {
            "A": np.array((0.0, 0.0, 0.0)),
            "B": np.array((2.0, 0.0, 0.0)),
            "C": np.array((0.0, 2.0, 0.0)),
        }
        geometry = {
            "shape": shape,
            "coordinates": lambda: positions,
            "project_scene": lambda point: np.asarray(point, dtype=float),
            "scene_unit_per_cm": 1.0,
            "stroke_width_per_pt": 1.0,
        }
        controller = install_generated_open_face_visibility_3d(
            scene,
            shape,
            {"face": face},
            geometry,
            open_face_vertex_ids=("A", "B", "C"),
            open_face_faces=(
                {
                    "face_id": "face",
                    "vertex_ids": ("A", "B", "C"),
                    "occludes_strokes": True,
                },
            ),
            open_face_face_bindings=(),
            open_face_inclusive_edges={"face": ()},
            open_face_strokes=(),
            open_face_bindings=(),
            source_resolver=lambda _objects, _geometry: {},
            face_source_resolver=lambda _objects: {"face": face},
            detach_static_entry=lambda _shape: None,
            restore_static_entry=lambda _shape, _entry: None,
            safe_length=lambda _stroke, _geometry: 1.0,
            projection_matrix=lambda: np.eye(3),
            paint_policy="diagrammatic",
            preferred_painter_z_band=(10_000.0, 11_024.0),
            visibility_group_id="face-only",
        )
        self.assertTrue(controller.attached)
        self.assertIsNotNone(controller.last_unified_frame)
        controller.restore()
        self.assertEqual(scene_painter_band_allocations(scene), ())

    def test_real_source_project_build_reaches_v3_bridge_and_unified_runtime(self) -> None:
        with tempfile.TemporaryDirectory(prefix="source-project-real-v3-") as temporary:
            root = Path(temporary)
            source_path = root / "figure.tex"
            shutil.copy2(SOURCE, source_path)
            (root / "bridge.json").write_text("{}\n", encoding="utf-8")
            project_path = root / "project.json"
            project_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": SOURCE_PROJECT_SCHEMA_VERSION,
                        "tikzSource": source_path.name,
                        "bridgeRequestTemplate": "bridge.json",
                        "derivedOutput": ".derived",
                        "renderIntent": {
                            "paintPolicy": "diagrammatic",
                            "projection": {
                                "kind": "orthographic",
                                "direction": [1, -1, -1],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(ROOT), environment.get("PYTHONPATH", "")))
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tikz_native.source_project",
                    "build",
                    str(project_path),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(
                result["built"], ["shape", "compositing", "generated_source"]
            )
            shape_asset = json.loads(
                (root / ".derived/shape-asset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(shape_asset["schema"], "tikz-native-asset/v1")
            self.assertEqual(
                set(shape_asset["provider"]),
                {"name", "asset_schema", "revision", "revision_component"},
            )
            self.assertEqual(
                shape_asset["provider"]["revision_component"], "asset_compiler"
            )
            generated = (root / ".derived/generated_scene.py").read_text(
                encoding="utf-8"
            )
            self.assertIn("tikz-native unified open-face override v1", generated)
            scene = Scene()
            namespace, _figure, geometry, controller = self._install_source(
                scene, generated
            )
            self.assertEqual(controller.compositing_mode, "unified")
            self.assertIsNotNone(controller.last_unified_frame)
            namespace["restore_open_face_visibility_3d"](controller)
            namespace["restore_geometry_3d_objects"](geometry)

            outer = self

            class BuiltSourceScene(Scene):
                def construct(inner_self) -> None:
                    (
                        built_namespace,
                        _built_figure,
                        built_geometry,
                        built_controller,
                    ) = outer._install_source(inner_self, generated)
                    inner_self.wait(0.2)
                    inner_self.has_unified_frame = (
                        built_controller.last_unified_frame is not None
                    )
                    built_namespace["restore_open_face_visibility_3d"](
                        built_controller
                    )
                    built_namespace["restore_geometry_3d_objects"](
                        built_geometry
                    )

            with tempfile.TemporaryDirectory(
                prefix="source-project-real-v3-media-"
            ) as media_dir, tempconfig(
                {
                    "renderer": "cairo",
                    "media_dir": media_dir,
                    "pixel_width": 160,
                    "pixel_height": 90,
                    "frame_rate": 5,
                    "disable_caching": True,
                    "write_to_movie": True,
                    "save_last_frame": False,
                }
            ):
                scene = BuiltSourceScene()
                scene.render()
                self.assertTrue(
                    Path(scene.renderer.file_writer.movie_file_path).is_file()
                )
                self.assertTrue(scene.has_unified_frame)


if __name__ == "__main__":
    unittest.main()
