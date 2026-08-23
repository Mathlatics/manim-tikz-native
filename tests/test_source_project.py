from __future__ import annotations

import contextlib
import io
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tikz_native.source_project import (
    BUILD_MANIFEST_SCHEMA_VERSION,
    COMMAND_RESULT_FORMAT_VERSION,
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


ROOT = Path(__file__).resolve().parents[1]
TEST_SHAPE_BUILDER_REVISION = "test-shape-builder/v1"
TEST_BRIDGE_GENERATOR_REVISION = "test-v3-bridge/v1"


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
        paint_policy: str = "diagrammatic",
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
            request = {"wholeFigureTargets": ["figure"]}
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
            "pictureIndex": project.picture_index,
            "entryMacro": project.entry_macro,
        }

    @staticmethod
    def fake_bridge_generator(request: Any) -> str:
        return (
            "from tikz_native.occlusion import OpenFaceOcclusion3D\n"
            "controller = OpenFaceOcclusion3D(shape)\n"
            "FadeIn(figure)\n"
            "FadeOut(figure)\n"
        )

    def build(self, project: Path, **kwargs: Any):
        revisions = dict(kwargs.pop("component_revisions", {}) or {})
        if not {"asset_compiler", "tikz_compiler"}.intersection(revisions):
            revisions["asset_compiler"] = TEST_SHAPE_BUILDER_REVISION
        if not {
            "generated_open_face_visibility_3d",
            "bridge_codegen",
        }.intersection(revisions):
            revisions[
                "generated_open_face_visibility_3d"
            ] = TEST_BRIDGE_GENERATOR_REVISION
        return build_project(
            project,
            shape_asset_builder=self.fake_shape_builder,
            bridge_generator=self.fake_bridge_generator,
            component_revisions=revisions,
            **kwargs,
        )

    @staticmethod
    def builder_revisions(**overrides: str | int) -> dict[str, str | int]:
        return {
            "asset_compiler": TEST_SHAPE_BUILDER_REVISION,
            "generated_open_face_visibility_3d": TEST_BRIDGE_GENERATOR_REVISION,
            **overrides,
        }

    def test_loads_only_authoritative_inputs(self) -> None:
        project_path = self.write_project(motion=True, hooks=True, bridge=True)
        project = load_source_project(project_path)
        self.assertEqual(project.tikz_source, (self.root / "figure.tex").resolve())
        self.assertEqual(project.motion_json, (self.root / "motion.json").resolve())
        self.assertEqual(project.hooks_source, (self.root / "hooks.py").resolve())
        self.assertEqual(
            project.bridge_request_template,
            (self.root / "bridge-request.json").resolve(),
        )
        self.assertEqual(project.output_directory.resolve(), (self.root / ".derived").resolve())
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

    def test_cache_hit_and_status_do_not_call_the_builder(self) -> None:
        project_path = self.write_project()
        calls = 0

        def counted_builder(project: Any, source: str) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return self.fake_shape_builder(project, source)

        revisions = self.builder_revisions()
        build_project(
            project_path,
            shape_asset_builder=counted_builder,
            component_revisions=revisions,
        )
        self.assertEqual(calls, 1)
        self.assertTrue(
            status_project(
                project_path,
                shape_asset_builder=counted_builder,
                component_revisions=revisions,
            ).fresh
        )
        build_project(
            project_path,
            shape_asset_builder=counted_builder,
            component_revisions=revisions,
        )
        self.assertEqual(calls, 1)

    def test_injected_builders_require_explicit_cache_revisions(self) -> None:
        project_path = self.write_project(bridge=True)
        with self.assertRaisesRegex(
            SourceProjectError, "custom shape_asset_builder"
        ):
            build_project(
                project_path, shape_asset_builder=self.fake_shape_builder
            )
        with self.assertRaisesRegex(
            SourceProjectError, "custom bridge_generator"
        ):
            build_project(
                project_path,
                bridge_generator=lambda _request: "value = 1\n",
            )
        with self.assertRaisesRegex(
            SourceProjectError, "unknown Provider component revision"
        ):
            status_project(
                project_path,
                component_revisions=self.builder_revisions(
                    open_face_unified_manim="unused-override"
                ),
                shape_asset_builder=self.fake_shape_builder,
            )

    def test_bridge_template_cannot_persist_generated_python(self) -> None:
        project_path = self.write_project(bridge=True)
        self.write(
            "bridge-request.json",
            json.dumps({"generatedSource": "value = 1\n"}),
        )
        with self.assertRaisesRegex(
            SourceProjectBuildError, "must not embed generated Python"
        ):
            self.build(project_path)

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
        raw["renderIntent"]["paintPolicy"] = "physical"
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        second = self.build(project_path)
        self.assertEqual(second.reused, ("shape", "motion"))
        self.assertEqual(second.built, ("compositing", "generated_source"))
        self.assertEqual(second.painter_z_band, first.painter_z_band)

    def test_projection_change_reuses_shape_and_motion(self) -> None:
        project_path = self.write_project(motion=True, bridge=True)
        self.build(project_path)
        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw["renderIntent"]["projection"] = {
            "kind": "orthographic",
            "direction": [-1, -1, -1],
        }
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        result = self.build(project_path)
        self.assertEqual(result.reused, ("shape", "motion"))
        self.assertEqual(result.built, ("compositing", "generated_source"))

    def test_picture_entry_and_selection_have_narrow_invalidation(self) -> None:
        project_path = self.write_project(bridge=True)
        self.build(project_path)

        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw["pictureIndex"] = 2
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        picture_changed = self.build(project_path)
        self.assertEqual(
            picture_changed.built, ("shape", "compositing", "generated_source")
        )

        raw["entryMacro"] = "figureTwo"
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        entry_changed = self.build(project_path)
        self.assertEqual(
            entry_changed.built, ("shape", "compositing", "generated_source")
        )

        raw["selection"] = {"include_object_ids": ["edge.AB"]}
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        selection_changed = self.build(project_path)
        self.assertEqual(selection_changed.reused, ("shape", "compositing"))
        self.assertEqual(selection_changed.built, ("generated_source",))

    def test_hooks_and_bridge_template_change_only_generated_source(self) -> None:
        project_path = self.write_project(hooks=True, bridge=True)
        first = self.build(project_path)
        self.write("hooks.py", "def user_hook(scene):\n    return None\n")
        second = self.build(project_path)
        self.assertEqual(second.reused, ("shape", "compositing"))
        self.assertEqual(second.built, ("generated_source",))
        self.assertEqual(second.painter_z_band, first.painter_z_band)

        request = json.loads((self.root / "bridge-request.json").read_text())
        request["wholeFigureTargets"] = ["figure", "shape"]
        self.write("bridge-request.json", json.dumps(request))
        third = self.build(project_path)
        self.assertEqual(third.reused, ("shape", "compositing"))
        self.assertEqual(third.built, ("generated_source",))

    def test_hooks_are_appended_after_rewrite_and_preserved_byte_for_byte(self) -> None:
        project_path = self.write_project(hooks=True, bridge=True)
        hooks = (
            "# OpenFaceOcclusion3D and FadeIn(figure) are documentation only.\n"
            "LEGACY_WORDS = 'install_open_face_visibility_3d FadeOut(figure)'\n"
            "def manual_animation(scene):\n"
            "    return LEGACY_WORDS\n"
        )
        self.write("hooks.py", hooks)
        self.build(project_path)
        generated_path = self.root / ".derived/generated_scene.py"
        first = generated_path.read_text(encoding="utf-8")
        expected_block = (
            "# >>> TIKZ_NATIVE_USER_HOOKS_V1\n"
            + hooks
            + "# <<< TIKZ_NATIVE_USER_HOOKS_V1\n"
        )
        self.assertIn(expected_block, first)

        rebuild_project(
            project_path,
            shape_asset_builder=self.fake_shape_builder,
            bridge_generator=self.fake_bridge_generator,
            component_revisions=self.builder_revisions(),
        )
        self.assertEqual(first, generated_path.read_text(encoding="utf-8"))

    def test_component_revisions_invalidate_only_dependants(self) -> None:
        project_path = self.write_project(motion=True, bridge=True)
        revisions = {
            "tikz_compiler": 10,
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

    def test_unused_generated_revision_does_not_stale_a_project_without_source(self) -> None:
        project_path = self.write_project()
        self.build(
            project_path,
            component_revisions={"bridge_codegen": "generated-revision-a"},
        )
        status = status_project(
            project_path,
            component_revisions=self.builder_revisions(
                bridge_codegen="generated-revision-b"
            ),
            shape_asset_builder=self.fake_shape_builder,
        )
        self.assertTrue(status.fresh)

    def test_missing_manifest_forces_regeneration(self) -> None:
        project_path = self.write_project(motion=True)
        self.build(project_path)
        manifest = self.root / ".derived/build-manifest.json"
        manifest.unlink()
        result = self.build(project_path)
        self.assertEqual(result.built, ("shape", "motion", "compositing"))

    def test_failed_or_racy_rebuild_never_publishes_a_partial_directory(self) -> None:
        project_path = self.write_project(motion=True)
        self.build(project_path)
        output = self.root / ".derived"

        def snapshot_output() -> dict[str, bytes]:
            return {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }

        baseline = snapshot_output()

        def changing_builder(project: Any, source: str) -> dict[str, Any]:
            self.write(
                "figure.tex",
                "\\begin{tikzpicture}\\draw (0,0) circle (2);\\end{tikzpicture}\n",
            )
            return self.fake_shape_builder(project, source)

        with self.assertRaisesRegex(
            SourceProjectBuildError, "changed during build"
        ):
            rebuild_project(
                project_path,
                shape_asset_builder=changing_builder,
                component_revisions=self.builder_revisions(),
            )
        self.assertEqual(snapshot_output(), baseline)
        self.assertFalse(any(self.root.glob(".derived.stage-*")))

        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (1,1);\\end{tikzpicture}\n",
        )

        def failing_builder(project: Any, source: str) -> dict[str, Any]:
            raise RuntimeError("compiler probe failure")

        with self.assertRaisesRegex(RuntimeError, "compiler probe failure"):
            rebuild_project(
                project_path,
                shape_asset_builder=failing_builder,
                component_revisions=self.builder_revisions(),
            )
        self.assertEqual(snapshot_output(), baseline)
        self.assertFalse(any(self.root.glob(".derived.stage-*")))

    def test_manifest_selection_overrides_or_removes_template_selection(self) -> None:
        project_path = self.write_project(bridge=True)
        self.write(
            "bridge-request.json",
            json.dumps({"selection": {"candidate_id": "template-only"}}),
        )
        captured: list[dict[str, Any]] = []

        def bridge(request: Any) -> str:
            captured.append(dict(request))
            return self.fake_bridge_generator(request)

        build_project(
            project_path,
            shape_asset_builder=self.fake_shape_builder,
            bridge_generator=bridge,
            component_revisions=self.builder_revisions(),
        )
        self.assertNotIn("selection", captured[-1])

        manifest = json.loads(project_path.read_text(encoding="utf-8"))
        manifest["selection"] = {"include_object_ids": ["edge.AB"]}
        project_path.write_text(json.dumps(manifest), encoding="utf-8")
        build_project(
            project_path,
            shape_asset_builder=self.fake_shape_builder,
            bridge_generator=bridge,
            component_revisions=self.builder_revisions(),
        )
        self.assertEqual(
            captured[-1]["selection"],
            {"include_object_ids": ["edge.AB"]},
        )

    def test_rebuild_ignores_cache_hits(self) -> None:
        project_path = self.write_project(motion=True, bridge=True)
        self.build(project_path)
        result = rebuild_project(
            project_path,
            shape_asset_builder=self.fake_shape_builder,
            bridge_generator=self.fake_bridge_generator,
            component_revisions=self.builder_revisions(),
        )
        self.assertEqual(
            result.built,
            ("shape", "motion", "compositing", "generated_source"),
        )
        self.assertEqual(result.reused, ())

    def test_status_and_clean_are_safe(self) -> None:
        project_path = self.write_project()
        before = status_project(
            project_path,
            shape_asset_builder=self.fake_shape_builder,
            component_revisions=self.builder_revisions(),
        )
        self.assertFalse(before.fresh)
        self.assertEqual(before.manifest_action, "missing")
        self.assertTrue(all(node.action == "missing" for node in before.nodes))

        self.build(project_path)
        after = status_project(
            project_path,
            shape_asset_builder=self.fake_shape_builder,
            component_revisions=self.builder_revisions(),
        )
        self.assertTrue(after.fresh)
        self.assertEqual(after.manifest_action, "fresh")

        sentinel = self.write("keep-me.txt", "authored")
        removed = clean_project(project_path)
        self.assertEqual(removed.resolve(), (self.root / ".derived").resolve())
        self.assertFalse(removed.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "authored")
        self.assertTrue(project_path.exists())

    def test_status_detects_stale_build_manifest_intent(self) -> None:
        project_path = self.write_project(bridge=True)
        self.build(project_path)
        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw["selection"] = {"include_object_ids": ["edge.AB"]}
        project_path.write_text(json.dumps(raw), encoding="utf-8")

        self.assertFalse(
            status_project(
                project_path,
                shape_asset_builder=self.fake_shape_builder,
                component_revisions=self.builder_revisions(),
            ).fresh
        )
        result = self.build(project_path)
        self.assertEqual(result.reused, ("shape", "compositing"))
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["authoringIntent"]["selection"],
            {"include_object_ids": ["edge.AB"]},
        )

    def test_status_reports_stale_manifest_when_nodes_are_fresh(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        manifest_path = self.root / ".derived/build-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["provider"]["revision"] = "tampered-revision"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        status = status_project(
            project_path,
            shape_asset_builder=self.fake_shape_builder,
            component_revisions=self.builder_revisions(),
        )
        self.assertFalse(status.fresh)
        self.assertEqual(status.manifest_action, "stale")
        self.assertTrue(all(node.action == "fresh" for node in status.nodes))
        self.assertEqual(status.as_dict()["manifestAction"], "stale")

    def test_status_fails_closed_if_source_changes_during_check(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        from tikz_native import source_project as source_project_module

        original_cache_hit = source_project_module._cache_hit
        changed = False

        def changing_cache_hit(*args: Any, **kwargs: Any) -> bool:
            nonlocal changed
            result = original_cache_hit(*args, **kwargs)
            if not changed:
                changed = True
                self.write(
                    "figure.tex",
                    "\\begin{tikzpicture}\\draw (0,0) circle (1);\\end{tikzpicture}\n",
                )
            return result

        with mock.patch.object(
            source_project_module, "_cache_hit", side_effect=changing_cache_hit
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError, "changed during build"
            ):
                status_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

    def test_default_nested_output_is_created_safely(self) -> None:
        project_path = self.write_project()
        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw.pop("derivedOutput")
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        result = self.build(project_path)
        self.assertEqual(
            result.manifest_path.resolve(),
            (self.root / ".tikz-native/derived/build-manifest.json").resolve(),
        )
        self.assertTrue(result.manifest_path.is_file())

    def test_missing_output_status_and_clean_are_read_only(self) -> None:
        project_path = self.write_project()
        before = sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
        )
        status = status_project(
            project_path,
            shape_asset_builder=self.fake_shape_builder,
            component_revisions=self.builder_revisions(),
        )
        self.assertFalse(status.fresh)
        self.assertEqual(
            clean_project(project_path).resolve(),
            (self.root / ".derived").resolve(),
        )
        after = sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
        )
        self.assertEqual(after, before)

    def test_owned_output_remains_valid_when_project_is_moved(self) -> None:
        project_path = self.write_project(motion=True, bridge=True)
        self.build(project_path)
        moved_container = tempfile.TemporaryDirectory()
        self.addCleanup(moved_container.cleanup)
        moved_root = Path(moved_container.name) / "moved-project"
        shutil.copytree(self.root, moved_root)
        moved_project = moved_root / project_path.name

        status = status_project(
            moved_project,
            shape_asset_builder=self.fake_shape_builder,
            component_revisions=self.builder_revisions(),
        )
        self.assertTrue(status.fresh)
        result = self.build(moved_project)
        self.assertEqual(
            result.reused,
            ("shape", "motion", "compositing", "generated_source"),
        )
        clean_project(moved_project)
        self.assertFalse((moved_root / ".derived").exists())

    def test_clean_refuses_an_unowned_existing_directory(self) -> None:
        project_path = self.write_project()
        self.write(".derived/keep.txt", "authored")
        with self.assertRaisesRegex(SourceProjectBuildError, "ownership marker"):
            clean_project(project_path)
        self.assertEqual(
            (self.root / ".derived/keep.txt").read_text(encoding="utf-8"),
            "authored",
        )

    def test_clean_refuses_a_directory_disguised_as_generated_output(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        shape_output = self.root / ".derived/shape-asset.json"
        shape_output.unlink()
        shape_output.mkdir()
        authored = shape_output / "user.txt"
        authored.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(
            SourceProjectBuildError, "not regular files"
        ):
            self.build(project_path)
        with self.assertRaisesRegex(
            SourceProjectBuildError, "not regular files"
        ):
            clean_project(project_path)
        self.assertEqual(authored.read_text(encoding="utf-8"), "keep")

    def test_build_and_clean_refuse_unknown_files_in_owned_output(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        authored = self.root / ".derived/user-notes.txt"
        authored.write_text("keep", encoding="utf-8")

        for operation in (self.build, clean_project):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(
                    SourceProjectBuildError, "unowned entries"
                ):
                    operation(project_path)
                self.assertEqual(authored.read_text(encoding="utf-8"), "keep")

    def test_owned_output_identity_rejects_tampering_and_retargeting(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        marker = self.root / ".derived/.tikz-native-owned.json"
        marker_payload = marker.read_bytes()
        marker.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(SourceProjectBuildError, "does not match"):
            clean_project(project_path)
        marker.write_bytes(marker_payload)

        other_output = self.root / ".other-derived"
        shutil.copytree(self.root / ".derived", other_output)
        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw["derivedOutput"] = ".other-derived"
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(SourceProjectBuildError, "does not match"):
            self.build(project_path)
        self.assertTrue(other_output.is_dir())

    def test_owned_output_identity_rejects_a_renamed_manifest(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        renamed = self.root / "renamed-project.json"
        project_path.rename(renamed)
        with self.assertRaisesRegex(SourceProjectBuildError, "does not match"):
            clean_project(renamed)
        self.assertTrue((self.root / ".derived").is_dir())

    def test_removing_optional_motion_removes_its_derived_node(self) -> None:
        project_path = self.write_project(motion=True)
        self.build(project_path)
        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw.pop("motionJson")
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        result = self.build(project_path)
        self.assertFalse((self.root / ".derived/motion-asset.json").exists())
        self.assertNotIn("motion", {node.name for node in result.nodes})

    def test_generated_source_is_unified_and_redirects_whole_figure_fades(self) -> None:
        source = (
            "from tikz_native.occlusion import OpenFaceOcclusion3D\n"
            "controller = OpenFaceOcclusion3D(shape, compositing_mode='legacy')\n"
            "FadeIn(figure)\n"
            "FadeOut(figure, run_time=2)\n"
        )
        rewritten = rewrite_generated_source(
            source,
            paint_policy="diagrammatic",
            painter_z_band=PainterZBand(100.0, 200.0),
            whole_figure_targets=("figure",),
        )
        self.assertIn("compositing_mode='unified'", rewritten)
        self.assertIn("paint_policy='diagrammatic'", rewritten)
        self.assertIn("painter_z_band=(100.0, 200.0)", rewritten)
        self.assertNotIn("legacy", rewritten)
        self.assertIn("FadeIn(controller.display_mobject)", rewritten)
        self.assertIn(
            "FadeOut(controller.display_mobject, run_time=2)", rewritten
        )
        compile(rewritten, "generated_scene.py", "exec")

    def test_generated_open_face_code_fails_closed_without_current_binding(self) -> None:
        with self.assertRaisesRegex(SourceProjectBuildError, "explicit import"):
            rewrite_generated_source(
                "controller = OpenFaceOcclusion3D(shape)\n",
                paint_policy="diagrammatic",
                painter_z_band=PainterZBand(0.0, 1.0),
            )
        untouched = rewrite_generated_source(
            "def legacy_open_face_renderer():\n    return None\n",
            paint_policy="diagrammatic",
            painter_z_band=PainterZBand(0.0, 1.0),
        )
        self.assertIn("legacy_open_face_renderer", untouched)

    def test_build_rewrites_bridge_generated_source(self) -> None:
        project_path = self.write_project(bridge=True)
        self.build(project_path)
        generated = (self.root / ".derived/generated_scene.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("compositing_mode='unified'", generated)
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
        build_validator = Draft202012Validator(build_schema)
        invalid_builds = {
            "unknown paint policy": {
                **build_value,
                "renderIntent": {
                    **build_value["renderIntent"],
                    "paintPolicy": "legacy",
                },
            },
            "missing TikZ input": {**build_value, "inputs": {}},
            "missing component revisions": {
                **build_value,
                "componentRevisions": {},
            },
            "escaping shape output": {
                **build_value,
                "nodes": {
                    **build_value["nodes"],
                    "shape": {
                        **build_value["nodes"]["shape"],
                        "output": "../../authored.tex",
                    },
                },
            },
            "motion input without motion node": {
                **build_value,
                "nodes": {
                    key: value
                    for key, value in build_value["nodes"].items()
                    if key != "motion"
                },
            },
            "motion node without motion input": {
                **build_value,
                "inputs": {
                    key: value
                    for key, value in build_value["inputs"].items()
                    if key != "motionJson"
                },
            },
            "Bridge input without generated node": {
                **build_value,
                "nodes": {
                    key: value
                    for key, value in build_value["nodes"].items()
                    if key != "generated_source"
                },
            },
            "generated node without Bridge input": {
                **build_value,
                "inputs": {
                    key: value
                    for key, value in build_value["inputs"].items()
                    if key != "bridgeRequestTemplate"
                },
            },
        }
        for label, value in invalid_builds.items():
            with self.subTest(label=label):
                self.assertFalse(build_validator.is_valid(value))

    def test_schema_and_loader_reject_the_same_unsafe_manifest_shapes(self) -> None:
        schema = json.loads(
            resources.files("tikz_native")
            .joinpath("schemas/tikz-native-source-project-v1.schema.json")
            .read_text(encoding="utf-8")
        )
        validator = Draft202012Validator(schema)
        project_path = self.write_project()
        baseline = json.loads(project_path.read_text(encoding="utf-8"))
        self.write("hooks.py", "def hook():\n    return None\n")
        cases = {
            "nested implementation mode": {
                **baseline,
                "renderIntent": {
                    **baseline["renderIntent"],
                    "projection": {"compositing-mode": "legacy"},
                },
            },
            "unsupported selection property": {
                **baseline,
                "selection": {"objects": ["edge.AB"]},
            },
            "selection without Bridge": {
                **baseline,
                "selection": {"include_object_ids": ["edge.AB"]},
            },
            "derived output is project root": {
                **baseline,
                "derivedOutput": "./.",
            },
            "hooks without Bridge": {**baseline, "hooksSource": "hooks.py"},
            "parent traversal": {**baseline, "tikzSource": "../figure.tex"},
            "absolute path": {
                **baseline,
                "tikzSource": str((self.root / "figure.tex").resolve()),
            },
            "unknown property": {**baseline, "generatedScene": "scene.py"},
            "noncanonical painter band keys": {
                **baseline,
                "renderIntent": {
                    **baseline["renderIntent"],
                    "painterZBand": {"minimum": 0, "maximum": 1},
                },
            },
            "non-string paint policy": {
                **baseline,
                "renderIntent": {
                    **baseline["renderIntent"],
                    "paintPolicy": [],
                },
            },
        }
        for label, value in cases.items():
            with self.subTest(label=label):
                self.assertFalse(validator.is_valid(value))
                project_path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(SourceProjectError):
                    load_source_project(project_path)

    def test_provider_component_descriptor_declares_ownership(self) -> None:
        descriptor = provider_component_descriptor()
        self.assertEqual(descriptor["name"], PROVIDER_COMPONENT)
        from tikz_native.version import (
            COMPONENT_SOURCE_PROJECT_BUILD,
            provider_component_revision,
        )

        self.assertEqual(
            descriptor["revision"],
            provider_component_revision(COMPONENT_SOURCE_PROJECT_BUILD),
        )
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
        self.assertEqual(
            payload["resultFormat"], COMMAND_RESULT_FORMAT_VERSION
        )
        self.assertNotIn("schemaVersion", payload)
        shape_asset = json.loads(
            (self.root / ".derived/shape-asset.json").read_text(encoding="utf-8")
        )
        self.assertEqual(shape_asset["schema"], "tikz-native-asset/v1")

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

    def test_module_cli_runs_in_a_fresh_process(self) -> None:
        project_path = self.write_project()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(ROOT), environment.get("PYTHONPATH", "")))
        )

        def run(command: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tikz_native.source_project",
                    command,
                    str(project_path),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )

        stale = run("status")
        self.assertEqual(stale.returncode, 1, stale.stderr)
        self.assertFalse(json.loads(stale.stdout)["fresh"])
        built = run("build")
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertEqual(json.loads(built.stdout)["mode"], "build")
        fresh = run("status")
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        fresh_payload = json.loads(fresh.stdout)
        self.assertEqual(fresh_payload["mode"], "status")
        self.assertTrue(fresh_payload["fresh"])

        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\clip (0,0) rectangle (1,1);"
            "\\draw (0,0) -- (1,1);\\end{tikzpicture}\n",
        )
        failed = run("build")
        self.assertEqual(failed.returncode, 2)
        self.assertEqual(failed.stdout, "")
        self.assertNotIn("Traceback", failed.stderr)
        self.assertIn("strict native gate", failed.stderr)

        invalid_manifest = json.loads(project_path.read_text(encoding="utf-8"))
        invalid_manifest["renderIntent"]["paintPolicy"] = []
        project_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
        invalid = run("build")
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(invalid.stdout, "")
        self.assertNotIn("Traceback", invalid.stderr)
        self.assertIn("paintPolicy", invalid.stderr)

    def test_directory_lock_survives_atomic_manifest_replacement(self) -> None:
        project_path = self.write_project()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(ROOT), environment.get("PYTHONPATH", "")))
        )
        script = """
import sys
from tikz_native.source_project import _project_lock, load_source_project
project = load_source_project(sys.argv[1])
with _project_lock(project):
    print('LOCKED', flush=True)
    sys.stdin.readline()
"""

        def start() -> subprocess.Popen[str]:
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(project_path)],
                cwd=ROOT,
                env=environment,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            def cleanup() -> None:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()

            self.addCleanup(cleanup)
            return process

        first = start()
        assert first.stdout is not None and first.stdin is not None
        self.assertEqual(first.stdout.readline().strip(), "LOCKED")
        replacement = self.root / "project.replacement.json"
        replacement.write_bytes(project_path.read_bytes())
        os.replace(replacement, project_path)

        second = start()
        assert second.stdout is not None and second.stdin is not None
        ready, _, _ = select.select([second.stdout], [], [], 0.3)
        self.assertEqual(ready, [])

        first.stdin.write("\n")
        first.stdin.flush()
        self.assertEqual(first.wait(timeout=5), 0)
        output, error = second.communicate(input="\n", timeout=5)
        self.assertEqual(second.returncode, 0, error)
        self.assertEqual(output.strip(), "LOCKED")

    def test_status_cli_is_nonzero_for_stale_project(self) -> None:
        project_path = self.write_project()
        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            self.assertEqual(main(["status", str(project_path)]), 1)
        self.assertFalse(json.loads(standard_output.getvalue())["fresh"])


if __name__ == "__main__":
    unittest.main()
