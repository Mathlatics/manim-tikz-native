from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tikz_native.source_project import (
    BUILD_MANIFEST_SCHEMA_VERSION,
    PROVIDER_CAPABILITY,
    PROVIDER_COMPONENT,
    SOURCE_PROJECT_SCHEMA_VERSION,
    PainterZBand,
    SourceProjectBuildError,
    SourceProjectError,
    build_project,
    clean_project,
    load_source_project,
    main,
    provider_component_descriptor,
    rebuild_project,
    rewrite_generated_source,
    status_project,
)


class SourceProjectTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_project(
        self,
        *,
        motion: bool = False,
        hooks: bool = False,
        bridge: bool = False,
        paint_policy: str = "source-order",
        projection: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (1,1);\\end{tikzpicture}\n",
        )
        manifest: dict[str, Any] = {
            "schemaVersion": SOURCE_PROJECT_SCHEMA_VERSION,
            "tikzSource": "figure.tex",
            "derivedOutput": ".derived",
            "renderIntent": {
                "paintPolicy": paint_policy,
                "projection": projection
                if projection is not None
                else {"kind": "orthographic", "direction": [1, -1, -1]},
            },
        }
        if motion:
            self.write("motion.json", json.dumps({"tracks": [{"name": "fold"}]}))
            manifest["motionJson"] = "motion.json"
        if hooks:
            self.write("hooks.py", "def user_hook(scene):\n    return scene\n")
            manifest["hooksSource"] = "hooks.py"
        if bridge:
            generated = (
                "from tikz_native.occlusion import OpenFaceOcclusion3D\n"
                "controller = OpenFaceOcclusion3D(shape)\n"
                "FadeIn(figure)\n"
                "FadeOut(figure)\n"
            )
            request = {
                "generatedSource": generated,
                "wholeFigureTargets": ["figure"],
            }
            self.write("bridge-request.json", json.dumps(request))
            manifest["bridgeRequestTemplate"] = "bridge-request.json"
        if extra:
            manifest.update(extra)
        return self.write("project.json", json.dumps(manifest, indent=2))

    @staticmethod
    def fake_shape_builder(project: Any, source: str) -> dict[str, Any]:
        return {
            "compiled": True,
            "source": source,
            "projection": project.projection,
        }

    def build(self, project: Path, **kwargs: Any):
        return build_project(
            project,
            shape_asset_builder=self.fake_shape_builder,
            **kwargs,
        )

    def test_loads_only_authoritative_inputs(self) -> None:
        project_path = self.write_project(motion=True, hooks=True, bridge=True)
        project = load_source_project(project_path)
        self.assertEqual(project.tikz_source, self.root / "figure.tex")
        self.assertEqual(project.motion_json, self.root / "motion.json")
        self.assertEqual(project.hooks_source, self.root / "hooks.py")
        self.assertEqual(
            project.bridge_request_template, self.root / "bridge-request.json"
        )
        self.assertEqual(project.output_directory, self.root / ".derived")
        self.assertFalse(hasattr(project, "compositing_mode"))

    def test_rejects_persisted_implementation_mode_at_any_depth(self) -> None:
        project_path = self.write_project()
        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw["renderIntent"]["compositingMode"] = "legacy"
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(SourceProjectError, "must not persist"):
            load_source_project(project_path)

    def test_rejects_path_traversal_absolute_paths_and_external_symlinks(self) -> None:
        project_path = self.write_project()
        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw["tikzSource"] = "../outside.tex"
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(SourceProjectError, "traverse"):
            load_source_project(project_path)

        raw["tikzSource"] = str((self.root / "figure.tex").resolve())
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(SourceProjectError, "must be relative"):
            load_source_project(project_path)

        external_root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(external_root, ignore_errors=True))
        external = external_root / "outside.tex"
        external.write_text("external", encoding="utf-8")
        link = self.root / "linked.tex"
        try:
            link.symlink_to(external)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable on this platform")
        raw["tikzSource"] = "linked.tex"
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(SourceProjectError, "escapes"):
            load_source_project(project_path)

    def test_deterministic_build_and_cache_reuse(self) -> None:
        project_path = self.write_project(motion=True, bridge=True)
        first = self.build(project_path)
        output = self.root / ".derived"
        files = [
            output / "shape-asset.json",
            output / "motion-asset.json",
            output / "unified-compositing.json",
            output / "generated_scene.py",
            output / "build-manifest.json",
        ]
        first_bytes = {path.name: path.read_bytes() for path in files}
        first_mtimes = {path.name: path.stat().st_mtime_ns for path in files[:-1]}

        time.sleep(0.01)
        second = self.build(project_path)
        second_bytes = {path.name: path.read_bytes() for path in files}
        second_mtimes = {path.name: path.stat().st_mtime_ns for path in files[:-1]}

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_mtimes, second_mtimes)
        self.assertEqual(first.built, ("shape", "motion", "compositing", "generated_source"))
        self.assertEqual(second.built, ())
        self.assertEqual(
            second.reused,
            ("shape", "motion", "compositing", "generated_source"),
        )

    def test_tikz_change_invalidates_every_node(self) -> None:
        project_path = self.write_project(motion=True, hooks=True, bridge=True)
        self.build(project_path)
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) circle (1);\\end{tikzpicture}\n",
        )
        result = self.build(project_path)
        self.assertEqual(
            result.built,
            ("shape", "motion", "compositing", "generated_source"),
        )

    def test_motion_only_change_reuses_shape(self) -> None:
        project_path = self.write_project(motion=True, bridge=True)
        first = self.build(project_path)
        original_band = first.painter_z_band
        self.write("motion.json", json.dumps({"tracks": [{"name": "turn"}]}))
        second = self.build(project_path)
        self.assertEqual(second.reused, ("shape",))
        self.assertEqual(
            second.built, ("motion", "compositing", "generated_source")
        )
        self.assertEqual(second.painter_z_band, original_band)

    def test_paint_policy_change_reuses_shape_and_motion(self) -> None:
        project_path = self.write_project(motion=True, bridge=True)
        first = self.build(project_path)
        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw["renderIntent"]["paintPolicy"] = "faces-first"
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        second = self.build(project_path)
        self.assertEqual(second.reused, ("shape", "motion"))
        self.assertEqual(second.built, ("compositing", "generated_source"))
        self.assertEqual(second.painter_z_band, first.painter_z_band)

    def test_hooks_and_bridge_template_change_only_generated_source(self) -> None:
        project_path = self.write_project(hooks=True, bridge=True)
        first = self.build(project_path)
        self.write("hooks.py", "def user_hook(scene):\n    return None\n")
        second = self.build(project_path)
        self.assertEqual(second.reused, ("shape", "compositing"))
        self.assertEqual(second.built, ("generated_source",))
        self.assertEqual(second.painter_z_band, first.painter_z_band)

        request = json.loads((self.root / "bridge-request.json").read_text())
        request["generatedSource"] += "# changed template\n"
        self.write("bridge-request.json", json.dumps(request))
        third = self.build(project_path)
        self.assertEqual(third.reused, ("shape", "compositing"))
        self.assertEqual(third.built, ("generated_source",))

    def test_component_revisions_invalidate_only_dependants(self) -> None:
        project_path = self.write_project(motion=True, bridge=True)
        revisions = {
            "tikz_compiler": 10,
            "motion_asset": 20,
            "unified_compositor": 30,
            "bridge_codegen": 40,
        }
        self.build(project_path, component_revisions=revisions)

        bridge_changed = dict(revisions)
        bridge_changed["bridge_codegen"] = 41
        bridge_result = self.build(
            project_path, component_revisions=bridge_changed
        )
        self.assertEqual(
            bridge_result.reused, ("shape", "motion", "compositing")
        )
        self.assertEqual(bridge_result.built, ("generated_source",))

        compiler_changed = dict(bridge_changed)
        compiler_changed["tikz_compiler"] = 11
        compiler_result = self.build(
            project_path, component_revisions=compiler_changed
        )
        self.assertEqual(
            compiler_result.built,
            ("shape", "motion", "compositing", "generated_source"),
        )

    def test_missing_manifest_forces_regeneration(self) -> None:
        project_path = self.write_project(motion=True)
        self.build(project_path)
        manifest = self.root / ".derived/build-manifest.json"
        manifest.unlink()
        result = self.build(project_path)
        self.assertEqual(result.built, ("shape", "motion", "compositing"))

    def test_rebuild_ignores_cache_hits(self) -> None:
        project_path = self.write_project(motion=True, bridge=True)
        self.build(project_path)
        result = rebuild_project(
            project_path, shape_asset_builder=self.fake_shape_builder
        )
        self.assertEqual(
            result.built,
            ("shape", "motion", "compositing", "generated_source"),
        )
        self.assertEqual(result.reused, ())

    def test_status_and_clean_are_safe(self) -> None:
        project_path = self.write_project()
        before = status_project(
            project_path, shape_asset_builder=self.fake_shape_builder
        )
        self.assertFalse(before.fresh)
        self.assertTrue(all(node.action == "missing" for node in before.nodes))

        self.build(project_path)
        after = status_project(
            project_path, shape_asset_builder=self.fake_shape_builder
        )
        self.assertTrue(after.fresh)

        sentinel = self.write("keep-me.txt", "authored")
        removed = clean_project(project_path)
        self.assertEqual(removed, self.root / ".derived")
        self.assertFalse(removed.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "authored")
        self.assertTrue(project_path.exists())

    def test_generated_source_is_unified_and_redirects_whole_figure_fades(self) -> None:
        source = (
            "from tikz_native.occlusion import OpenFaceOcclusion3D\n"
            "controller = OpenFaceOcclusion3D(shape, compositing_mode='legacy')\n"
            "FadeIn(figure)\n"
            "FadeOut(figure, run_time=2)\n"
        )
        rewritten = rewrite_generated_source(
            source,
            paint_policy="source-order",
            painter_z_band=PainterZBand(100.0, 200.0),
            whole_figure_targets=("figure",),
        )
        self.assertIn('compositing_mode="unified"', rewritten)
        self.assertIn("paint_policy='source-order'", rewritten)
        self.assertIn("painter_z_band=(100.0, 200.0)", rewritten)
        self.assertNotIn("legacy", rewritten)
        self.assertIn("FadeIn(controller.display_mobject)", rewritten)
        self.assertIn(
            "FadeOut(controller.display_mobject, run_time=2)", rewritten
        )
        compile(rewritten, "generated_scene.py", "exec")

    def test_generated_open_face_code_fails_closed_without_current_binding(self) -> None:
        with self.assertRaisesRegex(SourceProjectBuildError, "without exposing"):
            rewrite_generated_source(
                "controller = OpenFaceOcclusion3D(shape)\n",
                paint_policy="source-order",
                painter_z_band=PainterZBand(0.0, 1.0),
            )
        with self.assertRaisesRegex(SourceProjectBuildError, "does not expose"):
            rewrite_generated_source(
                "def legacy_open_face_renderer():\n    return None\n",
                paint_policy="source-order",
                painter_z_band=PainterZBand(0.0, 1.0),
            )

    def test_build_rewrites_bridge_generated_source(self) -> None:
        project_path = self.write_project(bridge=True)
        self.build(project_path)
        generated = (self.root / ".derived/generated_scene.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('compositing_mode="unified"', generated)
        self.assertIn("FadeIn(controller.display_mobject)", generated)
        self.assertIn("FadeOut(controller.display_mobject)", generated)

    def test_schema_files_are_packaged_and_validate_real_outputs(self) -> None:
        schema_root = resources.files("tikz_native").joinpath("schemas")
        source_schema = json.loads(
            schema_root.joinpath(
                "tikz-native-source-project-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        build_schema = json.loads(
            schema_root.joinpath(
                "tikz-native-build-manifest-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(source_schema)
        Draft202012Validator.check_schema(build_schema)

        project_path = self.write_project(motion=True, bridge=True)
        project_value = json.loads(project_path.read_text(encoding="utf-8"))
        Draft202012Validator(source_schema).validate(project_value)
        self.build(project_path)
        build_value = json.loads(
            (self.root / ".derived/build-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(build_schema).validate(build_value)
        self.assertEqual(
            build_value["schemaVersion"], BUILD_MANIFEST_SCHEMA_VERSION
        )

    def test_provider_component_descriptor_declares_ownership(self) -> None:
        descriptor = provider_component_descriptor()
        self.assertEqual(descriptor["name"], PROVIDER_COMPONENT)
        self.assertIn(PROVIDER_CAPABILITY, descriptor["capabilities"])
        self.assertIn("tikz_native.source_project", descriptor["owns"])
        self.assertTrue(
            any(
                path.endswith("tikz-native-source-project-v1.schema.json")
                for path in descriptor["owns"]
            )
        )

    def test_cli_build_status_rebuild_and_clean(self) -> None:
        project_path = self.write_project()
        standard_output = io.StringIO()
        standard_error = io.StringIO()
        with contextlib.redirect_stdout(standard_output), contextlib.redirect_stderr(
            standard_error
        ):
            self.assertEqual(main(["build", str(project_path)]), 0)
        payload = json.loads(standard_output.getvalue())
        self.assertEqual(payload["mode"], "build")
        self.assertTrue((self.root / ".derived/shape-asset.json").exists())

        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            self.assertEqual(main(["status", str(project_path)]), 0)
        self.assertTrue(json.loads(standard_output.getvalue())["fresh"])

        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            self.assertEqual(main(["rebuild", str(project_path)]), 0)
        self.assertEqual(json.loads(standard_output.getvalue())["mode"], "rebuild")

        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            self.assertEqual(main(["clean", str(project_path)]), 0)
        self.assertFalse((self.root / ".derived").exists())

    def test_status_cli_is_nonzero_for_stale_project(self) -> None:
        project_path = self.write_project()
        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            self.assertEqual(main(["status", str(project_path)]), 1)
        self.assertFalse(json.loads(standard_output.getvalue())["fresh"])


if __name__ == "__main__":
    unittest.main()
