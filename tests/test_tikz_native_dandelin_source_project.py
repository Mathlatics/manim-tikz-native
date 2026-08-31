from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tikz_native import compile_document
from tikz_native.geometry_rig_3d import GeometryRig3DError, analyze_geometry_rig_3d
from tikz_native.geometry_rig_3d_bridge import (
    GEOMETRY_RIG_3D_BRIDGE_OPERATION,
    GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA,
    execute_geometry_rig_3d_request,
)
from tikz_native.geometry_rig_3d_source_v3_bridge import (
    GEOMETRY_RIG_3D_SOURCE_V3_OPERATION,
    GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA,
    execute_source_v3_request,
)
from tikz_native.native_manim_codegen_3d import (
    NativeManimCodegen3DError,
    generate_native_manim_source_3d,
)
from tikz_native.native_manim_codegen_3d_v2 import (
    NativeManimCodegen3DV2Error,
    generate_native_manim_source_3d_v2,
)
from tikz_native.native_manim_codegen_3d_v3 import (
    NativeManimCodegen3DV3Error,
    generate_native_manim_source_3d_v3,
)
from tikz_native.source_project import (
    SOURCE_PROJECT_SCHEMA_VERSION,
    SourceProjectBuildError,
    build_project,
)
from tikz_native.version import (
    COMPONENT_ASSET_COMPILER,
    provider_component_revision,
)


DANDELIN_SOURCE = r"""
\begin{tikzpicture}[3d view={38}{24},scale=0.58]
  \coordinate (A) at (0,0,0);
  \coordinate (Z) at (0,0,1);
  \coordinate (R) at (1,0,0);
  \coordinate (O) at (0,0,2);
  \coordinate (U) at (0,1,2);
  \coordinate (V) at (-0.8,0,2.6);
  \DeclareSpacePlane{cut}{O/U/V};
  \DeclareSpaceRightCone{cone}{A/Z/R}{30}{0/9}{open_single};
  \DeclareDandelinConstruction{dan}{cone}{cut};
  \DrawDandelinDiagram[view=spatial,preset=classroom]{dan};
\end{tikzpicture}
"""


def _macro_composed_dandelin_source(*, entry_macro: bool) -> str:
    body = DANDELIN_SOURCE.replace(
        r"\DrawDandelinDiagram",
        r"\Draw\DandelinWord\DiagramWord",
    )
    definitions = (
        "\\gdef\\DandelinWord{Dandelin}\n"
        "\\gdef\\DiagramWord{Diagram}\n"
    )
    if not entry_macro:
        return definitions + body
    return definitions + "\\newcommand{\\DandelinFigure}{%\n" + body + "}\n"


class TikzNativeDandelinSourceProjectTests(unittest.TestCase):
    @staticmethod
    def _manifest() -> dict[str, object]:
        return {
            "schemaVersion": SOURCE_PROJECT_SCHEMA_VERSION,
            "tikzSource": "dandelin.tex",
            "derivedOutput": ".derived",
            "renderIntent": {
                "paintPolicy": "diagrammatic",
                "projection": {
                    "kind": "orthographic",
                    "direction": [1, -1, -1],
                },
            },
        }

    @staticmethod
    def _snapshot(output: Path) -> dict[str, bytes]:
        return {
            item.relative_to(output).as_posix(): item.read_bytes()
            for item in output.rglob("*")
            if item.is_file()
        }

    @staticmethod
    def _bridge_request(source: Path, *, v3: bool) -> dict[str, object]:
        return {
            "schema": (
                GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA
                if v3
                else GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA
            ),
            "operation": (
                GEOMETRY_RIG_3D_SOURCE_V3_OPERATION
                if v3
                else GEOMETRY_RIG_3D_BRIDGE_OPERATION
            ),
            "job_id": "dandelin-static-boundary-test",
            "input": {
                "source_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "entry_macro": None,
                "picture_index": 1,
                "expected_asset_provider_revision": provider_component_revision(
                    COMPONENT_ASSET_COMPILER
                ),
            },
        }

    def test_static_diagrammatic_project_builds_a_normal_shape_asset(self) -> None:
        with TemporaryDirectory(prefix="tikz-dandelin-source-project-") as directory:
            root = Path(directory)
            source = root / "dandelin.tex"
            source.write_text(DANDELIN_SOURCE, encoding="utf-8")
            manifest = root / "project.json"
            manifest.write_text(
                json.dumps(self._manifest(), indent=2),
                encoding="utf-8",
            )

            result = build_project(manifest)

            self.assertIn("shape", result.built)
            shape_asset = json.loads(
                (root / ".derived" / "shape-asset.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [(item["id"], item["kind"]) for item in shape_asset["object_index"]],
                [("dan:view:spatial", "dandelin_diagram")],
            )
            self.assertFalse((root / ".derived" / "generated_scene.py").exists())

    def test_depth_aware_hidden_lines_remain_a_static_shape_asset(self) -> None:
        automatic_source = DANDELIN_SOURCE.replace(
            "view=spatial,preset=classroom",
            "view=spatial,mode=depth_aware_diagrammatic,preset=classroom",
        )
        with TemporaryDirectory(prefix="tikz-dandelin-source-project-") as directory:
            root = Path(directory)
            (root / "dandelin.tex").write_text(
                automatic_source,
                encoding="utf-8",
            )
            manifest = root / "project.json"
            manifest.write_text(
                json.dumps(self._manifest(), indent=2),
                encoding="utf-8",
            )

            result = build_project(manifest)

            self.assertEqual(result.built, ("shape", "compositing"))
            shape_asset = json.loads(
                (root / ".derived" / "shape-asset.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                shape_asset["object_index"][0]["kind"],
                "dandelin_diagram",
            )
            self.assertFalse((root / ".derived" / "generated_scene.py").exists())

    def test_macro_composed_draw_is_rejected_before_staging_with_or_without_entry_macro(
        self,
    ) -> None:
        for entry_macro in (False, True):
            with self.subTest(entry_macro=entry_macro):
                with TemporaryDirectory(
                    prefix="tikz-dandelin-source-project-"
                ) as directory:
                    root = Path(directory)
                    source = root / "dandelin.tex"
                    source.write_text(
                        _macro_composed_dandelin_source(
                            entry_macro=entry_macro,
                        ),
                        encoding="utf-8",
                    )
                    manifest_value = self._manifest()
                    if entry_macro:
                        manifest_value["entryMacro"] = "DandelinFigure"
                    manifest = root / "project.json"
                    manifest.write_text(
                        json.dumps(manifest_value, indent=2),
                        encoding="utf-8",
                    )

                    build_project(manifest)
                    output = root / ".derived"
                    before = self._snapshot(output)
                    render_intent = copy.deepcopy(
                        manifest_value["renderIntent"]
                    )
                    assert isinstance(render_intent, dict)
                    render_intent["paintPolicy"] = "physical"
                    manifest_value["renderIntent"] = render_intent
                    manifest.write_text(
                        json.dumps(manifest_value, indent=2),
                        encoding="utf-8",
                    )
                    builder_calls: list[str] = []

                    def forbidden_builder(
                        _project: object,
                        _source: str,
                    ) -> object:
                        builder_calls.append("called")
                        raise AssertionError(
                            "custom ShapeAsset builder must not run"
                        )

                    with patch(
                        "tikz_native.source_project._create_staged_directory"
                    ) as create_stage:
                        with self.assertRaisesRegex(
                            SourceProjectBuildError,
                            r"Dandelin.*paintPolicy|paintPolicy.*Dandelin",
                        ):
                            build_project(
                                manifest,
                                shape_asset_builder=forbidden_builder,
                                component_revisions={
                                    "asset_compiler": "test-builder/v1"
                                },
                            )
                    create_stage.assert_not_called()
                    self.assertFalse(builder_calls)
                    self.assertEqual(self._snapshot(output), before)
                    self.assertFalse(
                        any(
                            item.name.startswith("..derived.")
                            for item in root.iterdir()
                        )
                    )

    def test_non_dandelin_custom_builder_keeps_its_non_native_entry_extension(
        self,
    ) -> None:
        with TemporaryDirectory(prefix="tikz-custom-source-project-") as directory:
            root = Path(directory)
            (root / "custom.tex").write_text(
                "custom builder authoritative source\n",
                encoding="utf-8",
            )
            manifest_value = self._manifest()
            manifest_value["tikzSource"] = "custom.tex"
            manifest_value["entryMacro"] = "ExternalFigure"
            manifest_value["pictureIndex"] = 7
            render_intent = copy.deepcopy(manifest_value["renderIntent"])
            assert isinstance(render_intent, dict)
            render_intent["paintPolicy"] = "physical"
            manifest_value["renderIntent"] = render_intent
            manifest = root / "project.json"
            manifest.write_text(
                json.dumps(manifest_value, indent=2),
                encoding="utf-8",
            )
            builder_calls: list[str] = []

            def custom_builder(project: object, source_text: str) -> object:
                builder_calls.append(source_text)
                return {
                    "compiled": True,
                    "entryMacro": getattr(project, "entry_macro"),
                }

            result = build_project(
                manifest,
                shape_asset_builder=custom_builder,
                component_revisions={"asset_compiler": "test-builder/v1"},
            )

            self.assertEqual(result.built, ("shape", "compositing"))
            self.assertEqual(
                builder_calls,
                ["custom builder authoritative source\n"],
            )

    def test_dynamic_intents_fail_before_staging_and_cannot_be_builder_bypassed(
        self,
    ) -> None:
        with TemporaryDirectory(prefix="tikz-dandelin-source-project-") as directory:
            root = Path(directory)
            source = root / "dandelin.tex"
            source.write_text(DANDELIN_SOURCE, encoding="utf-8")
            manifest = root / "project.json"
            base = self._manifest()
            manifest.write_text(json.dumps(base, indent=2), encoding="utf-8")
            build_project(manifest)
            output = root / ".derived"
            before = self._snapshot(output)

            (root / "motion.json").write_text("{}", encoding="utf-8")
            (root / "camera.json").write_text("{}", encoding="utf-8")
            (root / "bridge.json").write_text("{}", encoding="utf-8")
            (root / "hooks.py").write_text("", encoding="utf-8")
            physical_intent = copy.deepcopy(base["renderIntent"])
            assert isinstance(physical_intent, dict)
            physical_intent["paintPolicy"] = "physical"

            cases: tuple[tuple[str, dict[str, object], str], ...] = (
                (
                    "physical",
                    {"renderIntent": physical_intent},
                    "paintPolicy",
                ),
                ("motion", {"motionJson": "motion.json"}, "motionJson"),
                ("camera", {"cameraShots": "camera.json"}, "cameraShots"),
                (
                    "bridge",
                    {"bridgeRequestTemplate": "bridge.json"},
                    "bridgeRequestTemplate",
                ),
                (
                    "hooks",
                    {
                        "bridgeRequestTemplate": "bridge.json",
                        "hooksSource": "hooks.py",
                    },
                    "hooksSource",
                ),
                (
                    "selection",
                    {
                        "bridgeRequestTemplate": "bridge.json",
                        "selection": {"include_object_ids": ["dan:view:spatial"]},
                    },
                    "selection",
                ),
            )

            for name, changes, expected in cases:
                with self.subTest(name=name):
                    authored = copy.deepcopy(base)
                    authored.update(changes)
                    manifest.write_text(
                        json.dumps(authored, indent=2),
                        encoding="utf-8",
                    )
                    builder_calls: list[str] = []

                    def forbidden_builder(_project: object, _source: str) -> object:
                        builder_calls.append("called")
                        raise AssertionError("custom ShapeAsset builder must not run")

                    with patch(
                        "tikz_native.source_project._create_staged_directory"
                    ) as create_stage:
                        with self.assertRaisesRegex(
                            SourceProjectBuildError,
                            rf"Dandelin.*{expected}|{expected}.*Dandelin",
                        ):
                            build_project(
                                manifest,
                                shape_asset_builder=forbidden_builder,
                                component_revisions={
                                    "asset_compiler": "test-builder/v1"
                                },
                            )
                    create_stage.assert_not_called()
                    self.assertFalse(builder_calls)
                    self.assertEqual(self._snapshot(output), before)
                    self.assertFalse(
                        any(
                            item.name.startswith("..derived.")
                            for item in root.iterdir()
                        )
                    )

    def test_custom_builder_cannot_reinterpret_rejected_dandelin_semantics(
        self,
    ) -> None:
        with TemporaryDirectory(prefix="tikz-dandelin-source-project-") as directory:
            root = Path(directory)
            source = root / "dandelin.tex"
            source.write_text(
                DANDELIN_SOURCE.replace(
                    "view=spatial,preset=classroom",
                    "view=spatial,mode=physical",
                ),
                encoding="utf-8",
            )
            manifest = root / "project.json"
            manifest.write_text(
                json.dumps(self._manifest(), indent=2),
                encoding="utf-8",
            )
            builder_calls: list[str] = []

            def forbidden_builder(_project: object, _source: str) -> object:
                builder_calls.append("called")
                return {"forged": True}

            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "compiler could not certify.*Dandelin",
            ):
                build_project(
                    manifest,
                    shape_asset_builder=forbidden_builder,
                    component_revisions={"asset_compiler": "test-builder/v1"},
                )
            self.assertFalse(builder_calls)
            self.assertFalse((root / ".derived").exists())

    def test_rig_and_all_native_3d_codegen_entry_points_reject_curves(self) -> None:
        picture = compile_document(source_text=DANDELIN_SOURCE).pictures[0]

        with self.assertRaisesRegex(GeometryRig3DError, "Dandelin.*curved"):
            analyze_geometry_rig_3d(picture)
        with self.assertRaisesRegex(NativeManimCodegen3DError, "Dandelin.*static"):
            generate_native_manim_source_3d(picture, {"status": "ready"})
        with self.assertRaisesRegex(NativeManimCodegen3DV2Error, "Dandelin.*static"):
            generate_native_manim_source_3d_v2(picture, {"status": "ready"})
        with self.assertRaisesRegex(NativeManimCodegen3DV3Error, "Dandelin.*static"):
            generate_native_manim_source_3d_v3(picture, {"status": "ready"})

    def test_legacy_and_source_v3_bridges_reject_before_codegen(self) -> None:
        with TemporaryDirectory(prefix="tikz-dandelin-bridge-") as directory:
            source = Path(directory) / "dandelin.tex"
            source.write_text(DANDELIN_SOURCE, encoding="utf-8")

            legacy = execute_geometry_rig_3d_request(
                self._bridge_request(source, v3=False)
            )
            self.assertFalse(legacy["ok"])
            self.assertEqual(legacy["error"]["phase"], "analyze_geometry_rig_3d")
            self.assertIn("Dandelin", legacy["error"]["message"])
            self.assertNotIn("result", legacy)

            with patch(
                "tikz_native.geometry_rig_3d_source_v3_bridge."
                "generate_native_manim_source_3d_v3"
            ) as codegen:
                source_v3 = execute_source_v3_request(
                    self._bridge_request(source, v3=True)
                )
            self.assertFalse(source_v3["ok"])
            self.assertEqual(
                source_v3["error"]["phase"],
                "analyze_geometry_rig_3d",
            )
            self.assertIn("Dandelin", source_v3["error"]["message"])
            self.assertNotIn("result", source_v3)
            codegen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
