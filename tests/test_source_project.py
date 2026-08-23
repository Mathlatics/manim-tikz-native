from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

import tikz_native.source_project as source_project_module
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
    derive_painter_z_band,
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

    def output_snapshot(self, output_name: str = ".derived") -> dict[str, bytes]:
        output = self.root / output_name
        return {
            path.relative_to(output).as_posix(): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }

    def transaction_siblings(self, output_name: str = ".derived") -> list[str]:
        prefix = f".{output_name}."
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.name.startswith(prefix)
        )

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
            "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
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

    def test_derived_painter_bands_do_not_overlap_across_distinct_hash_slots(
        self,
    ) -> None:
        project = load_source_project(self.write_project())
        first = derive_painter_z_band(project, b"source-0\n")
        second = derive_painter_z_band(project, b"source-3\n")

        self.assertNotEqual(first, second)
        self.assertTrue(
            first.maximum < second.minimum
            or second.maximum < first.minimum
        )

    def test_selection_rejects_contradictory_object_ids(self) -> None:
        project_path = self.write_project(
            bridge=True,
            extra={
                "selection": {
                    "include_object_ids": ["line.M.N"],
                    "exclude_object_ids": ["line.M.N"],
                }
            },
        )
        with self.assertRaisesRegex(
            SourceProjectError,
            "includes and excludes the same objects",
        ):
            load_source_project(project_path)

    def test_selection_rejects_whitespace_ids_and_descending_range(self) -> None:
        for selection, message in (
            ({"candidate_id": "   "}, "candidate_id"),
            ({"include_object_ids": ["   "]}, "include_object_ids"),
            ({"range": [2.0, 1.0]}, "increasing"),
        ):
            with self.subTest(selection=selection):
                project_path = self.write_project(
                    bridge=True,
                    extra={"selection": selection},
                )
                with self.assertRaisesRegex(SourceProjectError, message):
                    load_source_project(project_path)

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
        original = self.build(project_path)

        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw["pictureIndex"] = 2
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        picture_changed = self.build(project_path)
        self.assertEqual(
            picture_changed.built, ("shape", "compositing", "generated_source")
        )
        self.assertNotEqual(picture_changed.painter_z_band, original.painter_z_band)

        raw["entryMacro"] = "figureTwo"
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        entry_changed = self.build(project_path)
        self.assertEqual(
            entry_changed.built, ("shape", "compositing", "generated_source")
        )
        self.assertNotEqual(entry_changed.painter_z_band, picture_changed.painter_z_band)

        raw["selection"] = {"include_object_ids": ["edge.AB"]}
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        selection_changed = self.build(project_path)
        self.assertEqual(selection_changed.reused, ("shape", "compositing"))
        self.assertEqual(selection_changed.built, ("generated_source",))
        self.assertEqual(selection_changed.painter_z_band, entry_changed.painter_z_band)

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

    def test_status_rejects_reserved_generated_source_without_a_bridge_node(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        generated = self.root / ".derived/generated_scene.py"
        generated.write_text("value = 'not planned by this project'\n", encoding="utf-8")

        status = status_project(
            project_path,
            shape_asset_builder=self.fake_shape_builder,
            component_revisions=self.builder_revisions(),
        )

        self.assertFalse(status.fresh)
        self.assertEqual(
            [
                (node.name, node.action, node.output)
                for node in status.nodes
                if node.output == "generated_scene.py"
            ],
            [("unexpected:generated_scene.py", "obsolete", "generated_scene.py")],
        )

    def test_status_fails_closed_if_reserved_file_appears_after_final_node_digest(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        original = source_project_module._sha256_regular_at
        digest_counts: dict[str, int] = {}
        inserted = False

        def insert_after_final_digest(
            directory_descriptor: int,
            name: str,
        ) -> str:
            nonlocal inserted
            digest = original(directory_descriptor, name)
            digest_counts[name] = digest_counts.get(name, 0) + 1
            # For this no-Bridge project each planned node is hashed once by
            # _cache_hit, once for the reported NodeState, and once during the
            # final consistency pass.  Compositing is the final planned node.
            if (
                name == "unified-compositing.json"
                and digest_counts[name] == 3
                and not inserted
            ):
                descriptor = os.open(
                    "generated_scene.py",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_descriptor,
                )
                try:
                    os.write(descriptor, b"value = 'late reserved file'\n")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                inserted = True
            return digest

        with mock.patch.object(
            source_project_module,
            "_sha256_regular_at",
            side_effect=insert_after_final_digest,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "derived output changed concurrently",
            ):
                status_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertTrue(inserted)
        self.assertEqual(
            (self.root / ".derived/generated_scene.py").read_text(
                encoding="utf-8"
            ),
            "value = 'late reserved file'\n",
        )

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
        self.assertEqual(self.transaction_siblings(), [])

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
        self.assertEqual(self.transaction_siblings(), [])

    def test_build_preserves_unknown_file_inserted_before_publish(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        builder_started = threading.Event()
        resume_builder = threading.Event()
        outcome: list[BaseException | object] = []

        def blocking_builder(project: Any, source: str) -> dict[str, Any]:
            builder_started.set()
            if not resume_builder.wait(timeout=5):
                raise RuntimeError("test timed out waiting to resume builder")
            return self.fake_shape_builder(project, source)

        def worker() -> None:
            try:
                outcome.append(
                    rebuild_project(
                        project_path,
                        shape_asset_builder=blocking_builder,
                        component_revisions=self.builder_revisions(),
                    )
                )
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(builder_started.wait(timeout=5))
        authored = self.root / ".derived/user-notes.txt"
        authored.write_text("keep", encoding="utf-8")
        resume_builder.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], SourceProjectBuildError)
        self.assertIn("unowned entries", str(outcome[0]))
        self.assertEqual(authored.read_text(encoding="utf-8"), "keep")
        current = self.output_snapshot()
        current.pop("user-notes.txt")
        self.assertEqual(current, baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_first_build_preserves_destination_created_at_no_replace_boundary(self) -> None:
        project_path = self.write_project()
        original = source_project_module._rename_no_replace
        created_identity: tuple[int, int] | None = None

        def create_destination_before_rename(
            parent_descriptor: int,
            source_name: str,
            destination_name: str,
        ) -> bool:
            nonlocal created_identity
            if (
                created_identity is None
                and source_name.startswith("..derived.stage-")
                and destination_name == ".derived"
            ):
                os.mkdir(destination_name, 0o700, dir_fd=parent_descriptor)
                created = os.stat(
                    destination_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                created_identity = (created.st_dev, created.st_ino)
            return original(parent_descriptor, source_name, destination_name)

        with mock.patch.object(
            source_project_module,
            "_rename_no_replace",
            side_effect=create_destination_before_rename,
        ):
            with self.assertRaises(SourceProjectBuildError):
                self.build(project_path)

        destination = self.root / ".derived"
        self.assertIsNotNone(created_identity)
        current = destination.stat(follow_symlinks=False)
        self.assertEqual((current.st_dev, current.st_ino), created_identity)
        self.assertEqual(list(destination.iterdir()), [])
        self.assertEqual(self.transaction_siblings(), [])

    def test_portable_publish_preserves_output_created_after_rollback(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original = source_project_module._rename_no_replace
        concurrent_identity: tuple[int, int] | None = None
        inserted = False

        def create_output_before_stage_publish(
            parent_descriptor: int,
            source_name: str,
            destination_name: str,
        ) -> bool:
            nonlocal concurrent_identity, inserted
            if not inserted and source_name.startswith("..derived.stage-"):
                os.mkdir(destination_name, 0o700, dir_fd=parent_descriptor)
                created = os.stat(
                    destination_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                concurrent_identity = (created.st_dev, created.st_ino)
                inserted = True
            return original(parent_descriptor, source_name, destination_name)

        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            return_value=False,
        ), mock.patch.object(
            source_project_module,
            "_rename_no_replace",
            side_effect=create_output_before_stage_publish,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "derived output appeared concurrently",
            ):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertEqual(self.output_snapshot(), baseline)
        self.assertTrue(inserted)
        self.assertIsNotNone(concurrent_identity)
        preserved = [
            path
            for path in self.root.iterdir()
            if path.is_dir()
            and (
                path.stat(follow_symlinks=False).st_dev,
                path.stat(follow_symlinks=False).st_ino,
            )
            == concurrent_identity
        ]
        self.assertEqual(len(preserved), 1)
        self.assertTrue(preserved[0].name.startswith("..derived.concurrent-"))
        self.assertEqual(list(preserved[0].iterdir()), [])
        self.assertFalse(
            any(
                path.name.startswith("..derived.rollback-")
                for path in self.root.iterdir()
            )
        )

    def test_portable_publish_preserves_all_directories_if_rollback_name_is_replaced(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original = source_project_module._named_directory_matches
        identities: dict[str, tuple[int, int]] = {}
        swapped = False

        def identity_at(parent_descriptor: int, name: str) -> tuple[int, int]:
            value = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            return value.st_dev, value.st_ino

        def write_in_directory(
            parent_descriptor: int,
            directory_name: str,
            file_name: str,
            payload: bytes,
        ) -> None:
            directory_descriptor = os.open(
                directory_name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=parent_descriptor,
            )
            try:
                descriptor = os.open(
                    file_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_descriptor,
                )
                try:
                    os.write(descriptor, payload)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                os.close(directory_descriptor)

        def replace_rollback_before_identity_check(
            parent_descriptor: int,
            name: str,
            descriptor: int,
        ) -> bool:
            nonlocal swapped
            if ".rollback-" in name and not swapped:
                swapped = True
                original_name = f"{name}.original"
                os.rename(
                    name,
                    original_name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                identities["original"] = identity_at(
                    parent_descriptor,
                    original_name,
                )

                os.mkdir(name, 0o700, dir_fd=parent_descriptor)
                write_in_directory(
                    parent_descriptor,
                    name,
                    "rollback-replacement.txt",
                    b"replacement must survive\n",
                )
                identities["replacement"] = identity_at(parent_descriptor, name)

                os.mkdir(".derived", 0o700, dir_fd=parent_descriptor)
                write_in_directory(
                    parent_descriptor,
                    ".derived",
                    "concurrent-output.txt",
                    b"concurrent output must survive\n",
                )
                identities["concurrent"] = identity_at(
                    parent_descriptor,
                    ".derived",
                )
            return original(parent_descriptor, name, descriptor)

        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            return_value=False,
        ), mock.patch.object(
            source_project_module,
            "_named_directory_matches",
            side_effect=replace_rollback_before_identity_check,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "derived output changed concurrently",
            ):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertTrue(swapped)
        found: dict[str, Path] = {}
        for label, identity in identities.items():
            matches = [
                path
                for path in self.root.iterdir()
                if path.is_dir()
                and (
                    path.stat(follow_symlinks=False).st_dev,
                    path.stat(follow_symlinks=False).st_ino,
                )
                == identity
            ]
            self.assertEqual(len(matches), 1, label)
            found[label] = matches[0]

        original_snapshot = {
            path.relative_to(found["original"]).as_posix(): path.read_bytes()
            for path in found["original"].rglob("*")
            if path.is_file()
        }
        self.assertEqual(original_snapshot, baseline)
        self.assertEqual(
            (found["replacement"] / "rollback-replacement.txt").read_text(
                encoding="utf-8"
            ),
            "replacement must survive\n",
        )
        self.assertEqual(
            (found["concurrent"] / "concurrent-output.txt").read_text(
                encoding="utf-8"
            ),
            "concurrent output must survive\n",
        )
        self.assertTrue(found["concurrent"].name.startswith("..derived.concurrent-"))

    def test_build_rejects_an_unplanned_reserved_file_inserted_into_stage(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original = source_project_module._publish_staged_directory
        inserted = False

        def insert_reserved_file(*args: Any, **kwargs: Any) -> None:
            nonlocal inserted
            stage_descriptor = args[3]
            descriptor = os.open(
                "generated_scene.py",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=stage_descriptor,
            )
            try:
                os.write(descriptor, b"value = 'concurrent reserved file'\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            inserted = True
            return original(*args, **kwargs)

        with mock.patch.object(
            source_project_module,
            "_publish_staged_directory",
            side_effect=insert_reserved_file,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "unexpected generated_scene.py",
            ):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertTrue(inserted)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_nested_output_parent_symlink_swap_cannot_escape_project(self) -> None:
        project_path = self.write_project()
        manifest = json.loads(project_path.read_text(encoding="utf-8"))
        manifest["derivedOutput"] = "a/b/.derived"
        project_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.build(project_path)

        external_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(external_temporary.cleanup)
        external_root = Path(external_temporary.name)
        (external_root / "b").mkdir()
        shutil.copytree(
            self.root / "a/b/.derived",
            external_root / "b/.derived",
        )
        (external_root / "b/external-sentinel.txt").write_text(
            "must survive",
            encoding="utf-8",
        )

        def external_snapshot() -> dict[str, bytes]:
            return {
                path.relative_to(external_root).as_posix(): path.read_bytes()
                for path in external_root.rglob("*")
                if path.is_file()
            }

        baseline_external = external_snapshot()
        original_lock = source_project_module._project_lock

        for label, operation in (
            ("build", lambda: self.build(project_path)),
            ("clean", lambda: clean_project(project_path)),
        ):
            with self.subTest(operation=label):
                boundary_reached = threading.Event()
                resume_operation = threading.Event()
                outcome: list[BaseException | object] = []

                @contextlib.contextmanager
                def paused_lock(project: Any):
                    boundary_reached.set()
                    if not resume_operation.wait(timeout=5):
                        raise RuntimeError("test timed out waiting to resume operation")
                    with original_lock(project) as descriptor:
                        yield descriptor

                def worker() -> None:
                    try:
                        outcome.append(operation())
                    except BaseException as exc:
                        outcome.append(exc)

                with mock.patch.object(
                    source_project_module,
                    "_project_lock",
                    side_effect=paused_lock,
                ):
                    thread = threading.Thread(target=worker)
                    thread.start()
                    self.assertTrue(boundary_reached.wait(timeout=5))
                    safe_parent = self.root / "a"
                    held_parent = self.root / f"a-held-{label}"
                    safe_parent.rename(held_parent)
                    try:
                        safe_parent.symlink_to(external_root, target_is_directory=True)
                    except (OSError, NotImplementedError):
                        held_parent.rename(safe_parent)
                        resume_operation.set()
                        thread.join(timeout=5)
                        self.skipTest("symlinks are unavailable on this platform")
                    resume_operation.set()
                    thread.join(timeout=5)

                self.assertFalse(thread.is_alive())
                self.assertEqual(len(outcome), 1)
                self.assertIsInstance(outcome[0], SourceProjectBuildError)
                self.assertEqual(external_snapshot(), baseline_external)
                safe_parent.unlink()
                held_parent.rename(safe_parent)

        self.assertTrue((self.root / "a/b/.derived").is_dir())

    def test_build_rolls_back_if_nested_parent_changes_at_final_validation(self) -> None:
        project_path = self.write_project()
        manifest = json.loads(project_path.read_text(encoding="utf-8"))
        manifest["derivedOutput"] = "a/b/.derived"
        project_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.build(project_path)
        old_output = self.root / "a/b/.derived"
        baseline = {
            path.relative_to(old_output).as_posix(): path.read_bytes()
            for path in old_output.rglob("*")
            if path.is_file()
        }
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (7,7);\\end{tikzpicture}\n",
        )
        original = source_project_module._validate_input_snapshot
        validation_count = 0
        swapped = False
        held_parent = self.root / "a-held-build"

        def replace_parent(snapshot: Any) -> None:
            nonlocal validation_count, swapped
            validation_count += 1
            original(snapshot)
            if validation_count == 2:
                (self.root / "a").rename(held_parent)
                (self.root / "a/b").mkdir(parents=True)
                (self.root / "a/b/concurrent.txt").write_text(
                    "keep",
                    encoding="utf-8",
                )
                swapped = True

        with mock.patch.object(
            source_project_module,
            "_validate_input_snapshot",
            side_effect=replace_parent,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "output parent changed concurrently",
            ):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertTrue(swapped)
        self.assertFalse((self.root / "a/b/.derived").exists())
        self.assertEqual(
            (self.root / "a/b/concurrent.txt").read_text(encoding="utf-8"),
            "keep",
        )
        held_output = held_parent / "b/.derived"
        held_snapshot = {
            path.relative_to(held_output).as_posix(): path.read_bytes()
            for path in held_output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(held_snapshot, baseline)

    def test_clean_restores_output_if_nested_parent_changes_before_commit(self) -> None:
        project_path = self.write_project()
        manifest = json.loads(project_path.read_text(encoding="utf-8"))
        manifest["derivedOutput"] = "a/b/.derived"
        project_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.build(project_path)
        old_output = self.root / "a/b/.derived"
        baseline = {
            path.relative_to(old_output).as_posix(): path.read_bytes()
            for path in old_output.rglob("*")
            if path.is_file()
        }
        original = source_project_module._require_output_parent_identity
        swapped = False
        held_parent = self.root / "a-held-clean"

        def replace_parent_then_validate(*args: Any, **kwargs: Any) -> None:
            nonlocal swapped
            if not swapped:
                (self.root / "a").rename(held_parent)
                (self.root / "a/b").mkdir(parents=True)
                (self.root / "a/b/concurrent.txt").write_text(
                    "keep",
                    encoding="utf-8",
                )
                swapped = True
            return original(*args, **kwargs)

        with mock.patch.object(
            source_project_module,
            "_require_output_parent_identity",
            side_effect=replace_parent_then_validate,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "output parent changed concurrently",
            ):
                clean_project(project_path)

        self.assertTrue(swapped)
        self.assertFalse((self.root / "a/b/.derived").exists())
        self.assertEqual(
            (self.root / "a/b/concurrent.txt").read_text(encoding="utf-8"),
            "keep",
        )
        held_output = held_parent / "b/.derived"
        held_snapshot = {
            path.relative_to(held_output).as_posix(): path.read_bytes()
            for path in held_output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(held_snapshot, baseline)

    def test_stage_name_swap_after_validation_cannot_publish_replacement(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original = source_project_module._validate_staged_artifacts
        swapped = False

        def swap_after_validation(*args: Any, **kwargs: Any) -> set[str]:
            nonlocal swapped
            entries = original(*args, **kwargs)
            if swapped:
                return entries
            swapped = True
            stage_descriptor = args[1]
            stage_stat = os.fstat(stage_descriptor)
            stage_path = next(
                path
                for path in self.root.iterdir()
                if path.is_dir()
                and path.stat().st_dev == stage_stat.st_dev
                and path.stat().st_ino == stage_stat.st_ino
            )
            validated_stage = self.root / f"{stage_path.name}.validated"
            stage_path.rename(validated_stage)
            stage_path.mkdir()
            (stage_path / "unvalidated.txt").write_text(
                "must not be published or deleted",
                encoding="utf-8",
            )
            return entries

        with mock.patch.object(
            source_project_module,
            "_validate_staged_artifacts",
            side_effect=swap_after_validation,
        ):
            with self.assertRaises(SourceProjectBuildError):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertTrue(swapped)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertFalse((self.root / ".derived/unvalidated.txt").exists())
        preserved = list(self.root.rglob("unvalidated.txt"))
        self.assertEqual(len(preserved), 1)
        self.assertNotEqual(preserved[0].parent, self.root / ".derived")

    def test_failed_builder_does_not_delete_a_replacement_at_the_stage_name(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        builder_started = threading.Event()
        resume_builder = threading.Event()
        outcome: list[BaseException | object] = []

        def failing_builder(project: Any, source: str) -> dict[str, Any]:
            builder_started.set()
            if not resume_builder.wait(timeout=5):
                raise RuntimeError("test timed out waiting to resume builder")
            raise RuntimeError("intentional builder failure")

        def worker() -> None:
            try:
                outcome.append(
                    rebuild_project(
                        project_path,
                        shape_asset_builder=failing_builder,
                        component_revisions=self.builder_revisions(),
                    )
                )
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(builder_started.wait(timeout=5))
        stage_path = next(
            path
            for path in self.root.iterdir()
            if path.name.startswith("..derived.stage-") and path.is_dir()
        )
        held_stage = self.root / f"{stage_path.name}.held"
        stage_path.rename(held_stage)
        stage_path.mkdir()
        authored = stage_path / "authored-after-swap.txt"
        authored.write_text("must survive", encoding="utf-8")
        resume_builder.set()
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], RuntimeError)
        self.assertIn("intentional builder failure", str(outcome[0]))
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(authored.read_text(encoding="utf-8"), "must survive")

    def test_clean_preserves_unknown_file_inserted_at_detach_boundary(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        boundary_reached = threading.Event()
        resume_detach = threading.Event()
        outcome: list[BaseException | object] = []
        original = source_project_module._detach_and_validate_owned_output

        def paused_detach(*args: Any, **kwargs: Any) -> str | None:
            if kwargs.get("purpose") == "clean":
                boundary_reached.set()
                if not resume_detach.wait(timeout=5):
                    raise RuntimeError("test timed out waiting to resume clean")
            return original(*args, **kwargs)

        def worker() -> None:
            try:
                outcome.append(clean_project(project_path))
            except BaseException as exc:
                outcome.append(exc)

        with mock.patch.object(
            source_project_module,
            "_detach_and_validate_owned_output",
            side_effect=paused_detach,
        ):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(boundary_reached.wait(timeout=5))
            authored = self.root / ".derived/user-notes.txt"
            authored.write_text("keep", encoding="utf-8")
            resume_detach.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], SourceProjectBuildError)
        self.assertEqual(authored.read_text(encoding="utf-8"), "keep")
        self.assertEqual(self.transaction_siblings(), [])

    def test_clean_restores_every_file_when_unknown_appears_after_final_validation(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original = source_project_module._validate_owned_directory_descriptor
        validation_count = 0
        inserted = False

        def insert_after_final_validation(*args: Any, **kwargs: Any) -> set[str]:
            nonlocal validation_count, inserted
            entries = original(*args, **kwargs)
            validation_count += 1
            if validation_count == 3:
                directory_descriptor = args[1]
                descriptor = os.open(
                    "unknown-after-validation.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_descriptor,
                )
                try:
                    os.write(descriptor, b"must survive\n")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                inserted = True
            return entries

        with mock.patch.object(
            source_project_module,
            "_validate_owned_directory_descriptor",
            side_effect=insert_after_final_validation,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "changed concurrently",
            ):
                clean_project(project_path)

        self.assertTrue(inserted)
        output = self.root / ".derived"
        self.assertEqual(
            (output / "unknown-after-validation.txt").read_text(encoding="utf-8"),
            "must survive\n",
        )
        current = self.output_snapshot()
        current.pop("unknown-after-validation.txt")
        self.assertEqual(current, baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_build_refuses_output_directory_identity_swap_after_validation(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        output = self.root / ".derived"
        preserved = self.root / "preserved-derived"
        baseline = self.output_snapshot()
        builder_started = threading.Event()
        resume_builder = threading.Event()
        outcome: list[BaseException | object] = []

        def blocking_builder(project: Any, source: str) -> dict[str, Any]:
            builder_started.set()
            if not resume_builder.wait(timeout=5):
                raise RuntimeError("test timed out waiting to resume builder")
            return self.fake_shape_builder(project, source)

        def worker() -> None:
            try:
                outcome.append(
                    rebuild_project(
                        project_path,
                        shape_asset_builder=blocking_builder,
                        component_revisions=self.builder_revisions(),
                    )
                )
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(builder_started.wait(timeout=5))
        output.rename(preserved)
        shutil.copytree(preserved, output)
        intruder = output / "concurrent-user-file.txt"
        intruder.write_text("keep", encoding="utf-8")
        resume_builder.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], SourceProjectBuildError)
        self.assertTrue(output.is_dir())
        self.assertEqual(intruder.read_text(encoding="utf-8"), "keep")
        self.assertEqual(
            {
                path.relative_to(preserved).as_posix(): path.read_bytes()
                for path in preserved.rglob("*")
                if path.is_file()
            },
            baseline,
        )
        self.assertEqual(self.transaction_siblings(), [])

    def test_clean_refuses_output_directory_identity_swap_at_detach_boundary(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        output = self.root / ".derived"
        preserved = self.root / "preserved-derived"
        baseline = self.output_snapshot()
        boundary_reached = threading.Event()
        resume_detach = threading.Event()
        outcome: list[BaseException | object] = []
        original = source_project_module._detach_and_validate_owned_output

        def paused_detach(*args: Any, **kwargs: Any) -> str | None:
            if kwargs.get("purpose") == "clean":
                boundary_reached.set()
                if not resume_detach.wait(timeout=5):
                    raise RuntimeError("test timed out waiting to resume clean")
            return original(*args, **kwargs)

        def worker() -> None:
            try:
                outcome.append(clean_project(project_path))
            except BaseException as exc:
                outcome.append(exc)

        with mock.patch.object(
            source_project_module,
            "_detach_and_validate_owned_output",
            side_effect=paused_detach,
        ):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(boundary_reached.wait(timeout=5))
            output.rename(preserved)
            shutil.copytree(preserved, output)
            intruder = output / "concurrent-user-file.txt"
            intruder.write_text("keep", encoding="utf-8")
            resume_detach.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], SourceProjectBuildError)
        self.assertTrue(output.is_dir())
        self.assertEqual(intruder.read_text(encoding="utf-8"), "keep")
        self.assertEqual(
            {
                path.relative_to(preserved).as_posix(): path.read_bytes()
                for path in preserved.rglob("*")
                if path.is_file()
            },
            baseline,
        )
        self.assertEqual(self.transaction_siblings(), [])

    def test_detach_recovery_restores_held_inode_not_replacement_name(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original = source_project_module._require_named_directory_identity
        swapped = False
        held_name = ""

        def replace_detached_name(
            parent_descriptor: int,
            name: str,
            descriptor: int,
            *,
            label: str,
        ) -> None:
            nonlocal swapped, held_name
            if not swapped and name.startswith("..derived.clean-"):
                held_name = f"{name}.held"
                os.rename(
                    name,
                    held_name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                os.mkdir(name, 0o700, dir_fd=parent_descriptor)
                replacement_directory_descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=parent_descriptor,
                )
                try:
                    replacement_descriptor = os.open(
                        "replacement.txt",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=replacement_directory_descriptor,
                    )
                    try:
                        os.write(replacement_descriptor, b"keep replacement\n")
                    finally:
                        os.close(replacement_descriptor)
                finally:
                    os.close(replacement_directory_descriptor)
                swapped = True
            original(
                parent_descriptor,
                name,
                descriptor,
                label=label,
            )

        with mock.patch.object(
            source_project_module,
            "_require_named_directory_identity",
            side_effect=replace_detached_name,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "original output was restored.*concurrent replacements were preserved",
            ):
                clean_project(project_path)

        self.assertTrue(swapped)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertFalse((self.root / held_name).exists())
        replacements = list(self.root.rglob("replacement.txt"))
        self.assertEqual(len(replacements), 1)
        self.assertEqual(replacements[0].read_bytes(), b"keep replacement\n")

    def test_cache_copy_rejects_changed_source_and_rebuilds_node(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        shape_path = self.root / ".derived/shape-asset.json"
        baseline = shape_path.read_bytes()
        original = source_project_module._copy_verified_cache_entry
        changed = False

        def racing_copy(*args: Any, **kwargs: Any) -> str | None:
            nonlocal changed
            plan = args[1]
            if plan.name == "shape" and not changed:
                changed = True
                shape_path.write_bytes(b'{"tampered":true}\n')
            return original(*args, **kwargs)

        with mock.patch.object(
            source_project_module,
            "_copy_verified_cache_entry",
            side_effect=racing_copy,
        ):
            result = self.build(project_path)
        self.assertTrue(changed)
        self.assertIn("shape", result.built)
        self.assertEqual(shape_path.read_bytes(), baseline)
        shape_node = next(node for node in result.nodes if node.name == "shape")
        self.assertEqual(
            shape_node.sha256,
            __import__("hashlib").sha256(shape_path.read_bytes()).hexdigest(),
        )
        self.assertTrue(
            status_project(
                project_path,
                component_revisions=self.builder_revisions(),
                shape_asset_builder=self.fake_shape_builder,
            ).fresh
        )

    def test_cache_manifest_hashes_the_staged_copy_not_mutable_old_path(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        shape_path = self.root / ".derived/shape-asset.json"
        baseline = shape_path.read_bytes()
        original = source_project_module._copy_verified_cache_entry
        changed = False

        def racing_copy(*args: Any, **kwargs: Any) -> str | None:
            nonlocal changed
            digest = original(*args, **kwargs)
            plan = args[1]
            if plan.name == "shape" and digest is not None and not changed:
                changed = True
                shape_path.write_bytes(b'{"changedAfterCopy":true}\n')
            return digest

        with mock.patch.object(
            source_project_module,
            "_copy_verified_cache_entry",
            side_effect=racing_copy,
        ):
            result = self.build(project_path)
        self.assertTrue(changed)
        self.assertIn("shape", result.reused)
        self.assertEqual(shape_path.read_bytes(), baseline)
        shape_node = next(node for node in result.nodes if node.name == "shape")
        self.assertEqual(
            shape_node.sha256,
            __import__("hashlib").sha256(baseline).hexdigest(),
        )

    def test_cache_copy_rehashes_staged_bytes_after_its_first_digest(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        shape_path = self.root / ".derived/shape-asset.json"
        baseline = shape_path.read_bytes()
        original = source_project_module._sha256_regular_at
        tampered = False

        def tamper_after_first_staged_digest(
            directory_descriptor: int,
            name: str,
        ) -> str:
            nonlocal tampered
            digest = original(directory_descriptor, name)
            if name.endswith(".cache") and not tampered:
                tampered = True
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_TRUNC,
                    dir_fd=directory_descriptor,
                )
                try:
                    os.write(descriptor, b'{"tamperedAfterDigest":true}\n')
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            return digest

        with mock.patch.object(
            source_project_module,
            "_sha256_regular_at",
            side_effect=tamper_after_first_staged_digest,
        ):
            result = self.build(project_path)

        self.assertTrue(tampered)
        self.assertIn("shape", result.built)
        self.assertEqual(shape_path.read_bytes(), baseline)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["nodes"]["shape"]["sha256"],
            __import__("hashlib").sha256(shape_path.read_bytes()).hexdigest(),
        )
        self.assertTrue(
            status_project(
                project_path,
                component_revisions=self.builder_revisions(),
                shape_asset_builder=self.fake_shape_builder,
            ).fresh
        )

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

    def test_missing_output_status_rejects_nested_parent_replacement(self) -> None:
        project_path = self.write_project()
        manifest = json.loads(project_path.read_text(encoding="utf-8"))
        manifest["derivedOutput"] = "a/b/.derived"
        project_path.write_text(json.dumps(manifest), encoding="utf-8")
        (self.root / "a/b").mkdir(parents=True)
        held_parent = self.root / "a-held-status-missing"
        original = source_project_module._validate_input_snapshot
        swapped = False

        def replace_parent_after_validation(snapshot: Any) -> None:
            nonlocal swapped
            original(snapshot)
            if not swapped:
                (self.root / "a").rename(held_parent)
                intruder = self.root / "a/b/.derived/intruder.txt"
                intruder.parent.mkdir(parents=True)
                intruder.write_text("keep", encoding="utf-8")
                swapped = True

        with mock.patch.object(
            source_project_module,
            "_validate_input_snapshot",
            side_effect=replace_parent_after_validation,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "output parent changed concurrently",
            ):
                status_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertTrue(swapped)
        self.assertEqual(
            (self.root / "a/b/.derived/intruder.txt").read_text(encoding="utf-8"),
            "keep",
        )
        self.assertFalse((held_parent / "b/.derived").exists())

    def test_missing_output_clean_rejects_nested_parent_replacement(self) -> None:
        project_path = self.write_project()
        manifest = json.loads(project_path.read_text(encoding="utf-8"))
        manifest["derivedOutput"] = "a/b/.derived"
        project_path.write_text(json.dumps(manifest), encoding="utf-8")
        (self.root / "a/b").mkdir(parents=True)
        held_parent = self.root / "a-held-clean-missing"
        original = source_project_module._detach_and_validate_owned_output
        swapped = False

        def replace_parent_after_absent(*args: Any, **kwargs: Any) -> Any:
            nonlocal swapped
            result = original(*args, **kwargs)
            if result is None and not swapped:
                (self.root / "a").rename(held_parent)
                intruder = self.root / "a/b/.derived/intruder.txt"
                intruder.parent.mkdir(parents=True)
                intruder.write_text("keep", encoding="utf-8")
                swapped = True
            return result

        with mock.patch.object(
            source_project_module,
            "_detach_and_validate_owned_output",
            side_effect=replace_parent_after_absent,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "output parent changed concurrently",
            ):
                clean_project(project_path)

        self.assertTrue(swapped)
        self.assertEqual(
            (self.root / "a/b/.derived/intruder.txt").read_text(encoding="utf-8"),
            "keep",
        )
        self.assertFalse((held_parent / "b/.derived").exists())

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
            "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
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
        with self.assertRaisesRegex(SourceProjectBuildError, "must be bound from"):
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

    def test_direct_open_face_rewrite_accepts_canonical_import_forms(self) -> None:
        cases = {
            "public": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n",
                "OpenFaceOcclusion3D",
            ),
            "aliased class": (
                "from polyhedron_visibility.open_faces.manim import "
                "OpenFaceOcclusion3D as OFO\n",
                "OFO",
            ),
            "aliased module": (
                "import polyhedron_visibility.open_faces as open_faces\n",
                "open_faces.OpenFaceOcclusion3D",
            ),
            "from-imported module": (
                "from polyhedron_visibility import open_faces as open_faces\n",
                "open_faces.OpenFaceOcclusion3D",
            ),
            "fully qualified with sibling module": (
                "import polyhedron_visibility.open_faces\n"
                "import polyhedron_visibility.style\n",
                "polyhedron_visibility.open_faces.OpenFaceOcclusion3D",
            ),
        }
        for label, (import_line, constructor) in cases.items():
            with self.subTest(label=label):
                source = (
                    import_line
                    + f"controller = {constructor}(shape, 'positional')\n"
                    + "FadeIn(figure)\n"
                )
                rewritten = rewrite_generated_source(
                    source,
                    paint_policy="physical",
                    painter_z_band=PainterZBand(10.0, 20.0),
                    whole_figure_targets=("figure",),
                )
                self.assertIn("compositing_mode='unified'", rewritten)
                self.assertIn("paint_policy='physical'", rewritten)
                self.assertIn("painter_z_band=(10.0, 20.0)", rewritten)
                self.assertIn("FadeIn(controller.display_mobject)", rewritten)
                self.assertIn("shape, 'positional'", rewritten)
                compile(rewritten, "generated_scene.py", "exec")
                self.assertEqual(
                    rewrite_generated_source(
                        rewritten,
                        paint_policy="physical",
                        painter_z_band=PainterZBand(10.0, 20.0),
                        whole_figure_targets=("figure",),
                    ),
                    rewritten,
                )

    def test_direct_open_face_rewrite_rejects_untrusted_or_shadowed_bindings(self) -> None:
        invalid_sources = {
            "untrusted source": (
                "from unrelated import OpenFaceOcclusion3D as OFO\n"
                "controller = OFO(shape)\n"
            ),
            "wildcard": (
                "from polyhedron_visibility.open_faces import *\n"
                "controller = OpenFaceOcclusion3D(shape)\n"
            ),
            "ambiguous class": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
                "from unrelated import OpenFaceOcclusion3D\n"
                "controller = OpenFaceOcclusion3D(shape)\n"
            ),
            "reassigned class": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
                "OpenFaceOcclusion3D = factory\n"
                "controller = OpenFaceOcclusion3D(shape)\n"
            ),
            "ambiguous module alias": (
                "import polyhedron_visibility.open_faces as faces\n"
                "import unrelated as faces\n"
                "controller = faces.OpenFaceOcclusion3D(shape)\n"
            ),
            "nested shadow": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
                "def helper(OpenFaceOcclusion3D):\n"
                "    return OpenFaceOcclusion3D(shape)\n"
                "controller = OpenFaceOcclusion3D(shape)\n"
            ),
            "relative import": (
                "from .polyhedron_visibility.open_faces import "
                "OpenFaceOcclusion3D\n"
                "controller = OpenFaceOcclusion3D(shape)\n"
            ),
            "nested reimport": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
                "def helper():\n"
                "    from unrelated import OpenFaceOcclusion3D\n"
                "    return OpenFaceOcclusion3D(shape)\n"
                "controller = OpenFaceOcclusion3D(shape)\n"
            ),
            "conditional reimport": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
                "if enabled:\n"
                "    from unrelated import OpenFaceOcclusion3D\n"
                "controller = OpenFaceOcclusion3D(shape)\n"
            ),
            "indirect constructor alias": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
                "factory = OpenFaceOcclusion3D\n"
                "controller = factory(shape)\n"
            ),
            "dynamic getattr": (
                "import polyhedron_visibility.open_faces as faces\n"
                "controller = getattr(faces, 'OpenFaceOcclusion3D')(shape)\n"
            ),
            "dynamic dunder getattribute": (
                "import polyhedron_visibility.open_faces as faces\n"
                "controller = faces.__getattribute__('OpenFaceOcclusion3D')(shape)\n"
            ),
            "aliased builtin getattr": (
                "from builtins import getattr as ga\n"
                "import polyhedron_visibility.open_faces as faces\n"
                "controller = ga(faces, 'OpenFaceOcclusion3D')(shape)\n"
            ),
            "operator attrgetter": (
                "from operator import attrgetter as ag\n"
                "import polyhedron_visibility.open_faces as faces\n"
                "controller = ag('OpenFaceOcclusion3D')(faces)(shape)\n"
            ),
            "stored operator attrgetter": (
                "import operator\n"
                "import polyhedron_visibility.open_faces as faces\n"
                "getter = operator.attrgetter('OpenFaceOcclusion3D')\n"
                "controller = getter(faces)(shape)\n"
            ),
            "stored builtin getattr": (
                "import polyhedron_visibility.open_faces as faces\n"
                "getter = getattr\n"
                "controller = getter(faces, 'OpenFaceOcclusion3D')(shape)\n"
            ),
            "stored module getattribute": (
                "import polyhedron_visibility.open_faces as faces\n"
                "getter = faces.__getattribute__\n"
                "controller = getter('OpenFaceOcclusion3D')(shape)\n"
            ),
            "lambda reflected lookup": (
                "import polyhedron_visibility.open_faces as faces\n"
                "getter = lambda module: getattr(module, 'OpenFaceOcclusion3D')\n"
                "controller = getter(faces)(shape)\n"
            ),
            "runtime eval": (
                "import polyhedron_visibility.open_faces as faces\n"
                "controller = eval('faces.OpenFaceOcclusion3D')(shape)\n"
            ),
            "runtime exec rebind": (
                "import polyhedron_visibility.open_faces as faces\n"
                "exec('faces.OpenFaceOcclusion3D = factory')\n"
                "controller = faces.OpenFaceOcclusion3D(shape)\n"
            ),
            "globals constructor alias": (
                "from polyhedron_visibility.open_faces import "
                "OpenFaceOcclusion3D as OFO\n"
                "controller = globals()['OFO'](shape)\n"
            ),
            "dynamic vars lookup": (
                "import polyhedron_visibility.open_faces as faces\n"
                "controller = vars(faces)['OpenFaceOcclusion3D'](shape)\n"
            ),
            "dynamic module dictionary lookup": (
                "import polyhedron_visibility.open_faces as faces\n"
                "controller = faces.__dict__['OpenFaceOcclusion3D'](shape)\n"
            ),
            "dynamic setattr rebinding": (
                "import polyhedron_visibility.open_faces as faces\n"
                "setattr(faces, 'OpenFaceOcclusion3D', factory)\n"
                "controller = faces.OpenFaceOcclusion3D(shape)\n"
            ),
            "dynamic module dictionary rebinding": (
                "import polyhedron_visibility.open_faces as faces\n"
                "faces.__dict__['OpenFaceOcclusion3D'] = factory\n"
                "controller = faces.OpenFaceOcclusion3D(shape)\n"
            ),
            "relative indirect constructor alias": (
                "from .polyhedron_visibility.open_faces import "
                "OpenFaceOcclusion3D as OFO\n"
                "factory = OFO\n"
                "controller = factory(shape)\n"
            ),
            "relative module indirect constructor alias": (
                "from .polyhedron_visibility import open_faces as faces\n"
                "factory = faces.OpenFaceOcclusion3D\n"
                "controller = factory(shape)\n"
            ),
            "dynamic builtins vars lookup": (
                "import builtins\n"
                "import polyhedron_visibility.open_faces as faces\n"
                "controller = builtins.vars(faces)"
                "['OpenFaceOcclusion3D'](shape)\n"
            ),
            "reassigned constructor attribute": (
                "import polyhedron_visibility.open_faces as faces\n"
                "faces.OpenFaceOcclusion3D = factory\n"
                "controller = faces.OpenFaceOcclusion3D(shape)\n"
            ),
        }
        for label, source in invalid_sources.items():
            with self.subTest(label=label):
                with self.assertRaises(SourceProjectBuildError):
                    rewrite_generated_source(
                        source,
                        paint_policy="diagrammatic",
                        painter_z_band=PainterZBand(0.0, 1.0),
                    )

    def test_direct_open_face_rewrite_rejects_reflection_even_when_unrelated(self) -> None:
        source = (
            "import polyhedron_visibility.open_faces as faces\n"
            "import unrelated as other\n"
            "module_name = getattr(faces, '__name__')\n"
            "setattr(faces, 'diagnostic', 1)\n"
            "other_factory = getattr(other, 'OpenFaceOcclusion3D')\n"
            "other_value = faces.__dict__['diagnostic']\n"
            "other_value_2 = vars(faces)['diagnostic']\n"
            "controller = faces.OpenFaceOcclusion3D(shape)\n"
        )
        with self.assertRaisesRegex(SourceProjectBuildError, "dynamic module"):
            rewrite_generated_source(
                source,
                paint_policy="diagrammatic",
                painter_z_band=PainterZBand(0.0, 1.0),
            )

    def test_non_open_face_source_keeps_ordinary_reflection_unchanged(self) -> None:
        source = (
            "note = 'OpenFaceOcclusion3D is documentation only'\n"
            "value = getattr(other, 'value')\n"
            "namespace = globals()\n"
            "exec('ordinary_value = 1')\n"
        )
        self.assertEqual(
            rewrite_generated_source(
                source,
                paint_policy="diagrammatic",
                painter_z_band=PainterZBand(0.0, 1.0),
            ),
            source,
        )

    def test_whole_figure_fades_require_one_constructor_call_and_assignment(self) -> None:
        invalid_sources = {
            "same name assigned twice": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
                "controller = OpenFaceOcclusion3D(shape_a)\n"
                "controller = OpenFaceOcclusion3D(shape_b)\n"
                "FadeIn(figure)\n"
            ),
            "one assigned and one discarded": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
                "OpenFaceOcclusion3D(shape_a)\n"
                "controller = OpenFaceOcclusion3D(shape_b)\n"
                "FadeIn(figure)\n"
            ),
            "constructor result not assigned": (
                "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
                "OpenFaceOcclusion3D(shape)\n"
                "FadeIn(figure)\n"
            ),
        }
        for label, source in invalid_sources.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    SourceProjectBuildError,
                    "exactly one directly assigned",
                ):
                    rewrite_generated_source(
                        source,
                        paint_policy="diagrammatic",
                        painter_z_band=PainterZBand(0.0, 1.0),
                        whole_figure_targets=("figure",),
                    )

    def test_v3_dispatch_requires_a_top_level_installer_definition(self) -> None:
        for source in (
            "note = 'install_open_face_visibility_3d'\n",
            "# install_open_face_visibility_3d\nvalue = 1\n",
            "def outer():\n"
            "    def install_open_face_visibility_3d():\n"
            "        return None\n",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    rewrite_generated_source(
                        source,
                        paint_policy="diagrammatic",
                        painter_z_band=PainterZBand(0.0, 1.0),
                    ),
                    source,
                )

    def test_v3_adapter_contract_errors_are_wrapped_narrowly(self) -> None:
        import tikz_native.generated_open_face_visibility_3d as adapter

        malformed = (
            "def install_open_face_visibility_3d(scene, shape, objects, state):\n"
            "    return None\n"
        )
        with self.assertRaises(SourceProjectBuildError) as raised:
            rewrite_generated_source(
                malformed,
                paint_policy="diagrammatic",
                painter_z_band=PainterZBand(0.0, 1.0),
            )
        self.assertIsInstance(
            raised.exception.__cause__,
            adapter.GeneratedOpenFaceVisibility3DError,
        )
        self.assertIn("adapter contract", str(raised.exception))

        invalid_after_adaptation = (
            "OPEN_FACE_VERTEX_IDS = ()\n"
            "OPEN_FACE_FACES = ()\n"
            "OPEN_FACE_FACE_BINDINGS = ()\n"
            "OPEN_FACE_INCLUSIVE_EDGES = ()\n"
            "OPEN_FACE_STROKES = ()\n"
            "OPEN_FACE_BINDINGS = ()\n"
            "OPEN_FACE_MODEL_SHA256 = '0' * 64\n"
            "def install_open_face_visibility_3d(scene, shape, objects, state):\n"
            "    return None\n"
            "def restore_open_face_visibility_3d(state):\n    return None\n"
            "def _open_face_sources(*args):\n    return None\n"
            "def _open_face_face_sources(*args):\n    return None\n"
            "def _open_face_detach_static_entry(*args):\n    return None\n"
            "def _open_face_restore_static_entry(*args):\n    return None\n"
            "def _open_face_safe_length(*args):\n    return 1.0\n"
            "def local_camera_matrix(*args):\n    return None\n"
            "return\n"
        )
        with self.assertRaises(SourceProjectBuildError) as invalid_raised:
            rewrite_generated_source(
                invalid_after_adaptation,
                paint_policy="diagrammatic",
                painter_z_band=PainterZBand(0.0, 1.0),
            )
        self.assertIsInstance(
            invalid_raised.exception.__cause__,
            adapter.GeneratedOpenFaceVisibility3DError,
        )
        self.assertIsInstance(
            invalid_raised.exception.__cause__.__cause__,
            SyntaxError,
        )

        with mock.patch.object(
            adapter,
            "rewrite_legacy_open_face_source",
            side_effect=RuntimeError("unexpected adapter bug"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected adapter bug"):
                rewrite_generated_source(
                    malformed,
                    paint_policy="diagrammatic",
                    painter_z_band=PainterZBand(0.0, 1.0),
                )

        duplicate_tree = __import__("ast").parse(
            "state = install_open_face_visibility_3d(scene, a, objects, geometry)\n"
            "state = install_open_face_visibility_3d(scene, b, objects, geometry)\n"
            "FadeIn(figure)\n"
        )
        call_count, controller_names = adapter._controller_bindings(duplicate_tree)
        self.assertEqual(call_count, 2)
        self.assertEqual(controller_names, ("state", "state"))
        with self.assertRaisesRegex(
            adapter.GeneratedOpenFaceVisibility3DError,
            "exactly one directly assigned",
        ):
            adapter._rewrite_exact_fades(
                __import__("ast").unparse(duplicate_tree) + "\n",
                targets=("figure",),
                controller_names=controller_names,
                controller_call_count=call_count,
            )

    def test_adapter_contract_failure_does_not_publish_partial_build(self) -> None:
        project_path = self.write_project(bridge=True)
        self.build(project_path)
        baseline = self.output_snapshot()

        def malformed_bridge(request: Any) -> str:
            return (
                "def install_open_face_visibility_3d(scene, shape, objects, state):\n"
                "    return None\n"
            )

        with self.assertRaisesRegex(SourceProjectBuildError, "adapter contract"):
            build_project(
                project_path,
                shape_asset_builder=self.fake_shape_builder,
                bridge_generator=malformed_bridge,
                component_revisions=self.builder_revisions(
                    generated_open_face_visibility_3d="malformed-v3"
                ),
            )
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_build_rewrites_bridge_generated_source(self) -> None:
        project_path = self.write_project(bridge=True)
        self.build(project_path)
        generated = (self.root / ".derived/generated_scene.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("compositing_mode='unified'", generated)
        self.assertIn("FadeIn(controller.display_mobject)", generated)
        self.assertIn("FadeOut(controller.display_mobject)", generated)

    def test_authoritative_json_rejects_duplicate_keys_at_any_depth(self) -> None:
        project_path = self.write_project()
        manifest_source = project_path.read_text(encoding="utf-8").replace(
            '"tikzSource": "figure.tex",',
            '"tikzSource": "figure.tex",\n  "tikzSource": "figure.tex",',
            1,
        )
        project_path.write_text(manifest_source, encoding="utf-8")
        with self.assertRaisesRegex(SourceProjectError, "duplicate JSON object key"):
            load_source_project(project_path)

        project_path = self.write_project(motion=True)
        self.write(
            "motion.json",
            '{"tracks":[{"name":"fold","name":"duplicate"}]}',
        )
        with self.assertRaisesRegex(SourceProjectError, "duplicate JSON object key"):
            self.build(project_path)

        project_path = self.write_project(bridge=True)
        self.write(
            "bridge-request.json",
            '{"selection":{"candidate_id":"a","candidate_id":"b"}}',
        )
        with self.assertRaisesRegex(SourceProjectError, "duplicate JSON object key"):
            self.build(project_path)

    def test_compiler_json_text_rejects_duplicate_keys(self) -> None:
        project_path = self.write_project()

        def ambiguous_builder(project: Any, source: str) -> str:
            return '{"shape":1,"shape":2}'

        with self.assertRaisesRegex(
            SourceProjectBuildError,
            "duplicate JSON object key",
        ):
            build_project(
                project_path,
                shape_asset_builder=ambiguous_builder,
                component_revisions=self.builder_revisions(),
            )
        self.assertFalse((self.root / ".derived").exists())

    def test_authoritative_json_rejects_overflowing_floats(self) -> None:
        project_path = self.write_project()
        manifest_source = project_path.read_text(encoding="utf-8").replace(
            '"paintPolicy": "diagrammatic",',
            '"paintPolicy": "diagrammatic",\n    "overflow": 1e999,',
            1,
        )
        project_path.write_text(manifest_source, encoding="utf-8")
        with self.assertRaisesRegex(SourceProjectError, "non-finite JSON number"):
            load_source_project(project_path)

        project_path = self.write_project(motion=True)
        self.write("motion.json", '{"tracks":[{"weight":1e999}]}')
        with self.assertRaisesRegex(SourceProjectError, "non-finite JSON number"):
            self.build(project_path)

        project_path = self.write_project(bridge=True)
        self.write("bridge-request.json", '{"selection":{"weight":1e999}}')
        with self.assertRaisesRegex(SourceProjectError, "non-finite JSON number"):
            self.build(project_path)

    def test_authoritative_json_rejects_unpaired_utf16_surrogates(self) -> None:
        project_path = self.write_project()
        manifest_source = project_path.read_text(encoding="utf-8").replace(
            '"paintPolicy": "diagrammatic",',
            '"paintPolicy": "diagrammatic",\n    "bad": "\\ud800",',
            1,
        )
        project_path.write_text(manifest_source, encoding="utf-8")

        with self.assertRaisesRegex(SourceProjectError, "unpaired UTF-16 surrogate"):
            load_source_project(project_path)

        standard_output = io.StringIO()
        standard_error = io.StringIO()
        with contextlib.redirect_stdout(standard_output), contextlib.redirect_stderr(
            standard_error
        ):
            result = main(["build", str(project_path)])
        self.assertEqual(result, 2)
        self.assertEqual(standard_output.getvalue(), "")
        self.assertIn("unpaired UTF-16 surrogate", standard_error.getvalue())
        self.assertNotIn("Traceback", standard_error.getvalue())

        project_path = self.write_project(motion=True)
        self.write("motion.json", '{"tracks":[{"label":"\\ud800"}]}')
        with self.assertRaisesRegex(SourceProjectError, "unpaired UTF-16 surrogate"):
            self.build(project_path)

        project_path = self.write_project(bridge=True)
        self.write("bridge-request.json", '{"selection":{"label":"\\ud800"}}')
        with self.assertRaisesRegex(SourceProjectError, "unpaired UTF-16 surrogate"):
            self.build(project_path)

    def test_compiler_json_text_rejects_overflowing_floats(self) -> None:
        project_path = self.write_project()

        def overflowing_builder(project: Any, source: str) -> str:
            return '{"shape":1e999}'

        with self.assertRaisesRegex(
            SourceProjectBuildError,
            "non-finite JSON number",
        ):
            build_project(
                project_path,
                shape_asset_builder=overflowing_builder,
                component_revisions=self.builder_revisions(),
            )
        self.assertFalse((self.root / ".derived").exists())

    def test_compiler_mapping_rejects_non_finite_values_without_partial_output(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()

        def non_finite_builder(project: Any, source: str) -> dict[str, Any]:
            return {"shape": float("inf")}

        with self.assertRaisesRegex(
            SourceProjectBuildError,
            "cannot be serialized as canonical JSON",
        ):
            rebuild_project(
                project_path,
                shape_asset_builder=non_finite_builder,
                component_revisions=self.builder_revisions(
                    asset_compiler="non-finite-builder/v1"
                ),
            )
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(self.transaction_siblings(), [])

        standard_output = io.StringIO()
        standard_error = io.StringIO()
        with mock.patch.object(
            source_project_module,
            "_default_shape_asset_builder",
            return_value={
                "schema": "tikz-native-asset/v1",
                "shape": float("inf"),
            },
        ), contextlib.redirect_stdout(standard_output), contextlib.redirect_stderr(
            standard_error
        ):
            result = main(["rebuild", str(project_path)])
        self.assertEqual(result, 2)
        self.assertEqual(standard_output.getvalue(), "")
        self.assertIn("cannot be serialized as canonical JSON", standard_error.getvalue())
        self.assertNotIn("Traceback", standard_error.getvalue())
        self.assertEqual(self.output_snapshot(), baseline)

    def test_duplicate_keys_in_previous_manifest_invalidate_cache(self) -> None:
        project_path = self.write_project(motion=True)
        self.build(project_path)
        manifest_path = self.root / ".derived/build-manifest.json"
        manifest_source = manifest_path.read_text(encoding="utf-8").replace(
            '"schemaVersion":"tikz-native-build-manifest/v1",',
            '"schemaVersion":"tikz-native-build-manifest/v1",'
            '"schemaVersion":"tikz-native-build-manifest/v1",',
            1,
        )
        manifest_path.write_text(manifest_source, encoding="utf-8")
        result = self.build(project_path)
        self.assertEqual(result.built, ("shape", "motion", "compositing"))
        self.assertTrue(
            status_project(
                project_path,
                component_revisions=self.builder_revisions(),
                shape_asset_builder=self.fake_shape_builder,
            ).fresh
        )

    def test_cli_rejects_duplicate_project_json_without_traceback(self) -> None:
        project_path = self.write_project()
        manifest_source = project_path.read_text(encoding="utf-8").replace(
            '"paintPolicy": "diagrammatic",',
            '"paintPolicy": "diagrammatic",\n'
            '    "paintPolicy": "physical",',
            1,
        )
        project_path.write_text(manifest_source, encoding="utf-8")
        standard_output = io.StringIO()
        standard_error = io.StringIO()
        with contextlib.redirect_stdout(standard_output), contextlib.redirect_stderr(
            standard_error
        ):
            result = main(["build", str(project_path)])
        self.assertEqual(result, 2)
        self.assertEqual(standard_output.getvalue(), "")
        self.assertIn("duplicate JSON object key", standard_error.getvalue())
        self.assertNotIn("Traceback", standard_error.getvalue())

    def test_cli_rejects_overflowing_project_number_without_traceback(self) -> None:
        project_path = self.write_project()
        manifest_source = project_path.read_text(encoding="utf-8").replace(
            '"paintPolicy": "diagrammatic",',
            '"paintPolicy": "diagrammatic",\n    "overflow": 1e999,',
            1,
        )
        project_path.write_text(manifest_source, encoding="utf-8")
        standard_output = io.StringIO()
        standard_error = io.StringIO()
        with contextlib.redirect_stdout(standard_output), contextlib.redirect_stderr(
            standard_error
        ):
            result = main(["build", str(project_path)])
        self.assertEqual(result, 2)
        self.assertEqual(standard_output.getvalue(), "")
        self.assertIn("non-finite JSON number", standard_error.getvalue())
        self.assertNotIn("Traceback", standard_error.getvalue())

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

    def test_clean_keeps_old_output_intact_if_empty_tombstone_changes(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original = source_project_module._rename_exchange
        inserted = False

        def exchange_then_change_empty_directory(
            parent_descriptor: int,
            left_name: str,
            right_name: str,
        ) -> bool:
            nonlocal inserted
            result = original(parent_descriptor, left_name, right_name)
            if result and not inserted and ".clean-" in left_name:
                empty_descriptor = os.open(
                    left_name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=parent_descriptor,
                )
                try:
                    descriptor = os.open(
                        "concurrent-note.txt",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=empty_descriptor,
                    )
                    try:
                        os.write(descriptor, b"must survive\n")
                    finally:
                        os.close(descriptor)
                finally:
                    os.close(empty_descriptor)
                inserted = True
            return result

        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            side_effect=exchange_then_change_empty_directory,
        ):
            with self.assertRaises(SourceProjectBuildError):
                clean_project(project_path)

        self.assertTrue(inserted)
        self.assertEqual(self.output_snapshot(), baseline)
        preserved = list(self.root.rglob("concurrent-note.txt"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), b"must survive\n")

    def test_clean_parent_fsync_failure_restores_old_output(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        root_stat = self.root.stat()
        original = os.fsync
        failed = False

        def fail_clean_parent_fsync(descriptor: int) -> None:
            nonlocal failed
            current = os.fstat(descriptor)
            if (
                not failed
                and current.st_dev == root_stat.st_dev
                and current.st_ino == root_stat.st_ino
            ):
                failed = True
                raise OSError(errno.EIO, "simulated clean directory fsync failure")
            original(descriptor)

        with mock.patch.object(os, "fsync", side_effect=fail_clean_parent_fsync):
            with self.assertRaisesRegex(
                OSError,
                "simulated clean directory fsync failure",
            ):
                clean_project(project_path)

        self.assertTrue(failed)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_clean_uses_portable_rollback_if_reverse_exchange_fails(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original_exchange = source_project_module._rename_exchange
        original_parent_check = source_project_module._require_output_parent_identity
        exchange_count = 0
        parent_check_count = 0

        def fail_reverse_exchange(*args: Any, **kwargs: Any) -> bool:
            nonlocal exchange_count
            exchange_count += 1
            if exchange_count == 1:
                return original_exchange(*args, **kwargs)
            raise OSError(errno.EIO, "simulated reverse exchange failure")

        def fail_final_parent_check(*args: Any, **kwargs: Any) -> None:
            nonlocal parent_check_count
            parent_check_count += 1
            if parent_check_count == 2:
                raise SourceProjectBuildError(
                    "simulated final parent validation failure"
                )
            original_parent_check(*args, **kwargs)

        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            side_effect=fail_reverse_exchange,
        ), mock.patch.object(
            source_project_module,
            "_require_output_parent_identity",
            side_effect=fail_final_parent_check,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "simulated final parent validation failure",
            ):
                clean_project(project_path)

        self.assertEqual(exchange_count, 2)
        self.assertEqual(parent_check_count, 2)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_clean_reports_recovery_name_and_closes_fd_if_portable_rollback_fails(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original_exchange = source_project_module._rename_exchange
        original_no_replace = source_project_module._rename_no_replace
        original_parent_check = source_project_module._require_output_parent_identity
        exchange_count = 0
        parent_check_count = 0

        def fail_reverse_exchange(*args: Any, **kwargs: Any) -> bool:
            nonlocal exchange_count
            exchange_count += 1
            if exchange_count == 1:
                return original_exchange(*args, **kwargs)
            raise OSError(errno.EIO, "simulated reverse exchange failure")

        def fail_old_tombstone_restore(
            parent_descriptor: int,
            source_name: str,
            destination_name: str,
        ) -> bool:
            if ".discard-" in source_name and ".derived.clean-" in destination_name:
                raise OSError(errno.EIO, "simulated portable restore failure")
            return original_no_replace(
                parent_descriptor,
                source_name,
                destination_name,
            )

        def fail_final_parent_check(*args: Any, **kwargs: Any) -> None:
            nonlocal parent_check_count
            parent_check_count += 1
            if parent_check_count == 2:
                raise SourceProjectBuildError(
                    "simulated final parent validation failure"
                )
            original_parent_check(*args, **kwargs)

        descriptor_count = len(os.listdir("/dev/fd"))
        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            side_effect=fail_reverse_exchange,
        ), mock.patch.object(
            source_project_module,
            "_rename_no_replace",
            side_effect=fail_old_tombstone_restore,
        ), mock.patch.object(
            source_project_module,
            "_require_output_parent_identity",
            side_effect=fail_final_parent_check,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "clean rollback could not restore.*preserved as",
            ):
                clean_project(project_path)

        self.assertEqual(len(os.listdir("/dev/fd")), descriptor_count)
        self.assertFalse((self.root / ".derived").exists())
        preserved_old = [
            path
            for path in self.root.iterdir()
            if path.is_dir()
            and {
                child.relative_to(path).as_posix(): child.read_bytes()
                for child in path.rglob("*")
                if child.is_file()
            }
            == baseline
        ]
        self.assertEqual(len(preserved_old), 1)
        self.assertIn(".discard-", preserved_old[0].name)

    def test_clean_post_commit_cleanup_preserves_dirfd_replacement(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        old_output_descriptor = os.open(
            self.root / ".derived",
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        original_unlink = os.unlink
        inserted = False

        def insert_replacement_before_quarantine_unlink(
            path: Any,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            nonlocal inserted
            name = os.fsdecode(path)
            if not inserted and name.startswith(".shape-asset.json.cleanup-"):
                descriptor = os.open(
                    "shape-asset.json",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=old_output_descriptor,
                )
                try:
                    os.write(descriptor, b"CONCURRENT USER DATA\n")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                inserted = True
            original_unlink(path, *args, **kwargs)

        try:
            with mock.patch.object(
                os,
                "unlink",
                side_effect=insert_replacement_before_quarantine_unlink,
            ):
                clean_project(project_path)
        finally:
            os.close(old_output_descriptor)

        self.assertTrue(inserted)
        self.assertFalse((self.root / ".derived").exists())
        preserved = [
            path
            for path in self.root.rglob("shape-asset.json")
            if path.read_bytes() == b"CONCURRENT USER DATA\n"
        ]
        self.assertEqual(len(preserved), 1)
        self.assertIn(".derived.clean-", preserved[0].parent.name)

    def test_publish_rolls_back_when_final_source_validation_fails(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original = source_project_module._validate_input_snapshot
        validation_count = 0

        def change_source_at_publication(snapshot: Any) -> None:
            nonlocal validation_count
            validation_count += 1
            if validation_count == 2:
                self.write(
                    "figure.tex",
                    "\\begin{tikzpicture}\\draw (0,0) -- (2,2);\\end{tikzpicture}\n",
                )
            original(snapshot)

        with mock.patch.object(
            source_project_module,
            "_validate_input_snapshot",
            side_effect=change_source_at_publication,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "changed during build",
            ):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )
        self.assertGreaterEqual(validation_count, 2)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_publish_rolls_back_if_final_validator_sees_node_bytes_change(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (4,4);\\end{tikzpicture}\n",
        )
        original = source_project_module._validate_input_snapshot
        validation_count = 0
        changed = False

        def mutate_published_node(snapshot: Any) -> None:
            nonlocal validation_count, changed
            validation_count += 1
            original(snapshot)
            if validation_count == 2:
                (self.root / ".derived/shape-asset.json").write_text(
                    '{"concurrent":"change"}\n',
                    encoding="utf-8",
                )
                changed = True

        with mock.patch.object(
            source_project_module,
            "_validate_input_snapshot",
            side_effect=mutate_published_node,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "staged derived output changed",
            ):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )
        self.assertTrue(changed)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_exchange_failure_restores_old_output_when_new_output_is_moved(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (5,5);\\end{tikzpicture}\n",
        )
        original = source_project_module._require_named_directory_identity
        moved = False
        held_new = self.root / ".derived-new-held"

        def move_new_output_before_identity_check(*args: Any, **kwargs: Any) -> None:
            nonlocal moved
            if kwargs.get("label") == "published output" and not moved:
                (self.root / ".derived").rename(held_new)
                moved = True
            return original(*args, **kwargs)

        with mock.patch.object(
            source_project_module,
            "_require_named_directory_identity",
            side_effect=move_new_output_before_identity_check,
        ):
            with self.assertRaises(SourceProjectBuildError):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertTrue(moved)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertTrue((held_new / "shape-asset.json").is_file())

    def test_exchange_failure_never_leaves_new_build_visible(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (6,6);\\end{tikzpicture}\n",
        )
        original = source_project_module._rename_exchange
        injected = False
        held_old = self.root / ".derived-old-held"

        def replace_output_before_exchange(
            parent_descriptor: int,
            left_name: str,
            right_name: str,
        ) -> bool:
            nonlocal injected
            if left_name.startswith("..derived.stage-") and not injected:
                (self.root / ".derived").rename(held_old)
                (self.root / ".derived").mkdir()
                (self.root / ".derived/concurrent.txt").write_text(
                    "keep",
                    encoding="utf-8",
                )
                injected = True
            return original(parent_descriptor, left_name, right_name)

        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            side_effect=replace_output_before_exchange,
        ):
            with self.assertRaises(SourceProjectBuildError):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertTrue(injected)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertFalse(held_old.exists())
        preserved_concurrent = list(self.root.rglob("concurrent.txt"))
        self.assertEqual(len(preserved_concurrent), 1)
        self.assertEqual(
            preserved_concurrent[0].read_text(encoding="utf-8"),
            "keep",
        )
        self.assertTrue(
            preserved_concurrent[0].parent.name.startswith("..derived.stage-")
        )

    def test_publish_fsync_failure_rolls_back_visible_output(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (3,3);\\end{tikzpicture}\n",
        )
        root_stat = self.root.stat()
        original = os.fsync
        failed = False

        def fail_parent_fsync(descriptor: int) -> None:
            nonlocal failed
            current = os.fstat(descriptor)
            if (
                not failed
                and current.st_dev == root_stat.st_dev
                and current.st_ino == root_stat.st_ino
            ):
                failed = True
                raise OSError(errno.EIO, "simulated directory fsync failure")
            original(descriptor)

        with mock.patch.object(os, "fsync", side_effect=fail_parent_fsync):
            with self.assertRaisesRegex(OSError, "simulated directory fsync failure"):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )
        self.assertTrue(failed)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_publish_uses_portable_rollback_if_reverse_exchange_fails(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (8,8);\\end{tikzpicture}\n",
        )
        original_exchange = source_project_module._rename_exchange
        exchange_count = 0

        def fail_reverse_exchange(*args: Any, **kwargs: Any) -> bool:
            nonlocal exchange_count
            exchange_count += 1
            if exchange_count == 1:
                return original_exchange(*args, **kwargs)
            raise OSError(errno.EIO, "simulated reverse exchange failure")

        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            side_effect=fail_reverse_exchange,
        ), mock.patch.object(
            source_project_module,
            "_require_output_parent_identity",
            side_effect=SourceProjectBuildError(
                "simulated final parent validation failure"
            ),
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "simulated final parent validation failure",
            ):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        self.assertGreaterEqual(exchange_count, 2)
        self.assertEqual(self.output_snapshot(), baseline)
        self.assertEqual(self.transaction_siblings(), [])

    def test_publish_reports_exact_recovery_name_if_portable_rollback_fails(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (9,9);\\end{tikzpicture}\n",
        )
        original_exchange = source_project_module._rename_exchange
        original_no_replace = source_project_module._rename_no_replace
        exchange_count = 0

        def fail_reverse_exchange(*args: Any, **kwargs: Any) -> bool:
            nonlocal exchange_count
            exchange_count += 1
            if exchange_count == 1:
                return original_exchange(*args, **kwargs)
            raise OSError(errno.EIO, "simulated reverse exchange failure")

        def fail_old_output_restore(
            parent_descriptor: int,
            source_name: str,
            destination_name: str,
        ) -> bool:
            if source_name.startswith("..derived.stage-") and destination_name == ".derived":
                raise OSError(errno.EIO, "simulated portable restore failure")
            return original_no_replace(
                parent_descriptor,
                source_name,
                destination_name,
            )

        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            side_effect=fail_reverse_exchange,
        ), mock.patch.object(
            source_project_module,
            "_rename_no_replace",
            side_effect=fail_old_output_restore,
        ), mock.patch.object(
            source_project_module,
            "_require_output_parent_identity",
            side_effect=SourceProjectBuildError(
                "simulated final parent validation failure"
            ),
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "publication rollback could not restore.*preserved as",
            ):
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        preserved_old = [
            path
            for path in self.root.iterdir()
            if path.is_dir()
            and {
                child.relative_to(path).as_posix(): child.read_bytes()
                for child in path.rglob("*")
                if child.is_file()
            }
            == baseline
        ]
        self.assertEqual(len(preserved_old), 1)
        self.assertNotEqual(preserved_old[0], self.root / ".derived")
        self.assertTrue(preserved_old[0].name.startswith("..derived.stage-"))

    def test_portable_publication_reports_old_name_if_visible_move_fails(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (10,10);\\end{tikzpicture}\n",
        )
        original_no_replace = source_project_module._rename_no_replace

        def fail_visible_move(
            parent_descriptor: int,
            source_name: str,
            destination_name: str,
        ) -> bool:
            if source_name == ".derived" and destination_name.startswith(
                "..derived.concurrent-"
            ):
                raise OSError(errno.EIO, "simulated visible-output move failure")
            return original_no_replace(
                parent_descriptor,
                source_name,
                destination_name,
            )

        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            return_value=False,
        ), mock.patch.object(
            source_project_module,
            "_rename_no_replace",
            side_effect=fail_visible_move,
        ), mock.patch.object(
            source_project_module,
            "_require_output_parent_identity",
            side_effect=SourceProjectBuildError(
                "simulated final parent validation failure"
            ),
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "portable publication rollback could not restore.*preserved as",
            ) as raised:
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        preserved_old = [
            path
            for path in self.root.iterdir()
            if path.is_dir()
            and {
                child.relative_to(path).as_posix(): child.read_bytes()
                for child in path.rglob("*")
                if child.is_file()
            }
            == baseline
        ]
        self.assertEqual(len(preserved_old), 1)
        self.assertIn(preserved_old[0].name, str(raised.exception))

    def test_portable_publication_reports_old_name_if_old_restore_fails(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (11,11);\\end{tikzpicture}\n",
        )
        original_no_replace = source_project_module._rename_no_replace

        def fail_old_restore(
            parent_descriptor: int,
            source_name: str,
            destination_name: str,
        ) -> bool:
            if source_name.startswith("..derived.rollback-") and destination_name == ".derived":
                raise OSError(errno.EIO, "simulated old-output restore failure")
            return original_no_replace(
                parent_descriptor,
                source_name,
                destination_name,
            )

        with mock.patch.object(
            source_project_module,
            "_rename_exchange",
            return_value=False,
        ), mock.patch.object(
            source_project_module,
            "_rename_no_replace",
            side_effect=fail_old_restore,
        ), mock.patch.object(
            source_project_module,
            "_require_output_parent_identity",
            side_effect=SourceProjectBuildError(
                "simulated final parent validation failure"
            ),
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "portable publication rollback could not restore.*preserved as",
            ) as raised:
                rebuild_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )

        preserved_old = [
            path
            for path in self.root.iterdir()
            if path.is_dir()
            and {
                child.relative_to(path).as_posix(): child.read_bytes()
                for child in path.rglob("*")
                if child.is_file()
            }
            == baseline
        ]
        self.assertEqual(len(preserved_old), 1)
        self.assertIn(preserved_old[0].name, str(raised.exception))

    def test_status_rejects_final_output_name_swap(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        original = source_project_module._validate_input_snapshot
        swapped = False

        def swap_after_input_validation(snapshot: Any) -> None:
            nonlocal swapped
            original(snapshot)
            if not swapped:
                output = self.root / ".derived"
                output.rename(self.root / ".derived-held-by-test")
                output.mkdir()
                (output / "intruder.txt").write_text("keep", encoding="utf-8")
                swapped = True

        with mock.patch.object(
            source_project_module,
            "_validate_input_snapshot",
            side_effect=swap_after_input_validation,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "derived output changed concurrently",
            ):
                status_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )
        self.assertTrue(swapped)
        self.assertEqual(
            (self.root / ".derived/intruder.txt").read_text(encoding="utf-8"),
            "keep",
        )
        held = self.root / ".derived-held-by-test"
        held_snapshot = {
            path.relative_to(held).as_posix(): path.read_bytes()
            for path in held.rglob("*")
            if path.is_file()
        }
        self.assertEqual(held_snapshot, baseline)

    def test_status_rechecks_node_bytes_after_input_validation(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        original = source_project_module._validate_input_snapshot
        changed = False

        def mutate_node_after_input_validation(snapshot: Any) -> None:
            nonlocal changed
            original(snapshot)
            if not changed:
                (self.root / ".derived/shape-asset.json").write_text(
                    '{"late":"change"}\n',
                    encoding="utf-8",
                )
                changed = True

        with mock.patch.object(
            source_project_module,
            "_validate_input_snapshot",
            side_effect=mutate_node_after_input_validation,
        ):
            with self.assertRaisesRegex(
                SourceProjectBuildError,
                "derived output changed while status was checked",
            ):
                status_project(
                    project_path,
                    shape_asset_builder=self.fake_shape_builder,
                    component_revisions=self.builder_revisions(),
                )
        self.assertTrue(changed)

    def test_bridge_snapshot_name_swap_never_deletes_replacement(self) -> None:
        project_path = self.write_project(bridge=True)
        self.build(project_path)
        baseline = self.output_snapshot()
        reached = threading.Event()
        resume = threading.Event()
        snapshot_path: list[Path] = []
        outcome: list[BaseException | object] = []

        def blocking_bridge(request: Any) -> str:
            snapshot_path.append(Path(request["input"]["source_path"]))
            reached.set()
            if not resume.wait(timeout=5):
                raise RuntimeError("test timed out waiting to resume bridge")
            return self.fake_bridge_generator(request)

        def worker() -> None:
            try:
                outcome.append(
                    rebuild_project(
                        project_path,
                        shape_asset_builder=self.fake_shape_builder,
                        bridge_generator=blocking_bridge,
                        component_revisions=self.builder_revisions(
                            generated_open_face_visibility_3d="blocking-bridge/v1"
                        ),
                    )
                )
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(reached.wait(timeout=5))
        directory = snapshot_path[0].parent
        held = directory.with_name(directory.name + ".held-by-test")
        directory.rename(held)
        directory.mkdir()
        authored = directory / "authored.txt"
        authored.write_text("must survive", encoding="utf-8")
        self.addCleanup(shutil.rmtree, directory, True)
        self.addCleanup(shutil.rmtree, held, True)
        resume.set()
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], SourceProjectBuildError)
        self.assertEqual(authored.read_text(encoding="utf-8"), "must survive")
        self.assertTrue((held / snapshot_path[0].name).is_file())
        self.assertEqual(self.output_snapshot(), baseline)

    def test_default_compiler_snapshot_name_swap_never_deletes_replacement(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()
        reached = threading.Event()
        resume = threading.Event()
        snapshot_path: list[Path] = []
        outcome: list[BaseException | object] = []

        def blocking_compile(path: Path, **_kwargs: Any) -> dict[str, Any]:
            snapshot_path.append(Path(path))
            reached.set()
            if not resume.wait(timeout=5):
                raise RuntimeError("test timed out waiting to resume compiler")
            return {"schema": "tikz-native-asset/v1"}

        def worker() -> None:
            try:
                outcome.append(rebuild_project(project_path))
            except BaseException as exc:
                outcome.append(exc)

        import tikz_native.provider as provider

        with mock.patch.object(provider, "compile_asset", side_effect=blocking_compile):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(reached.wait(timeout=5))
            directory = snapshot_path[0].parent
            held = directory.with_name(directory.name + ".held-by-test")
            directory.rename(held)
            directory.mkdir()
            authored = directory / "authored.txt"
            authored.write_text("must survive", encoding="utf-8")
            self.addCleanup(shutil.rmtree, directory, True)
            self.addCleanup(shutil.rmtree, held, True)
            resume.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], SourceProjectBuildError)
        self.assertEqual(authored.read_text(encoding="utf-8"), "must survive")
        self.assertTrue((held / snapshot_path[0].name).is_file())
        self.assertEqual(self.output_snapshot(), baseline)

    def test_status_cli_is_nonzero_for_stale_project(self) -> None:
        project_path = self.write_project()
        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            self.assertEqual(main(["status", str(project_path)]), 1)
        self.assertFalse(json.loads(standard_output.getvalue())["fresh"])


if __name__ == "__main__":
    unittest.main()
