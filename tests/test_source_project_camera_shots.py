from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from jsonschema import Draft202012Validator
from manim import tempconfig

from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_shots import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
    canonical_parallel_camera_shot_sequence_json,
)
from tikz_native.source_project import (
    SOURCE_PROJECT_SCHEMA_VERSION,
    SourceProjectBuildError,
    SourceProjectError,
    _append_camera_shots_binding,
    build_project,
    load_source_project,
    status_project,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_V3_SOURCE = ROOT / "examples/dihedral_fold_3d_demo/dihedral_fold.tex"
TEST_ASSET_REVISION = "camera-shots-test-asset-v1"
TEST_BRIDGE_REVISION = "camera-shots-test-bridge-v1"
TEST_CAMERA_REVISION = "camera-shots-test-runtime-v1"
TEST_CAMERA_CORE_REVISION = "camera-shots-test-core-v1"


def _sequence(*, second_zoom: float = 0.85) -> ParallelCameraShotSequence:
    return ParallelCameraShotSequence(
        (
            ParallelCameraShot(
                "overview",
                ParallelCameraState.from_view_direction(
                    (1.0, 1.0, 0.8),
                    target=(0.25, -0.2, 0.4),
                    screen_anchor=(-0.5, 0.2),
                    zoom=1.1,
                ),
                duration=1.0,
                hold=0.2,
                cue="overview",
            ),
            ParallelCameraShot(
                "section",
                ParallelCameraState.from_view_direction(
                    (-0.5, 1.0, 1.2),
                    target=(0.0, 0.0, 0.25),
                    screen_anchor=(0.35, -0.1),
                    zoom=second_zoom,
                ),
                duration=0.8,
                transition="shortest",
                arc_height=0.0,
            ),
        )
    )


class SourceProjectCameraShotsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="source-project-camera-shots-"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, source: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def write_camera_shots(
        self,
        *,
        sequence: ParallelCameraShotSequence | None = None,
        pretty: bool = True,
    ) -> Path:
        value = (sequence or _sequence()).to_dict()
        source = (
            json.dumps(value, ensure_ascii=False, indent=2)
            if pretty
            else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        return self.write("camera-shots.json", source + "\n")

    def write_project(
        self,
        *,
        bridge: bool = True,
        hooks: bool = False,
        motion: bool = False,
    ) -> Path:
        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) -- (1,1);\\end{tikzpicture}\n",
        )
        self.write_camera_shots()
        manifest: dict[str, object] = {
            "schemaVersion": SOURCE_PROJECT_SCHEMA_VERSION,
            "tikzSource": "figure.tex",
            "cameraShots": "camera-shots.json",
            "derivedOutput": ".derived",
            "renderIntent": {
                "paintPolicy": "diagrammatic",
                "projection": {"kind": "identity"},
            },
        }
        if bridge:
            self.write("bridge.json", "{}\n")
            manifest["bridgeRequestTemplate"] = "bridge.json"
        if hooks:
            self.write(
                "hooks.py",
                "AUTHORED_CAMERA_SHOT_IDS = tuple(\n"
                "    shot.id for shot in TIKZ_NATIVE_CAMERA_SHOTS.shots\n"
                ")\n",
            )
            manifest["hooksSource"] = "hooks.py"
        if motion:
            self.write("motion.json", '{"timeline":[]}\n')
            manifest["motionJson"] = "motion.json"
        return self.write(
            "project.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

    @staticmethod
    def shape_builder(project, source: str):
        return {
            "compiled": True,
            "source": source,
            "pictureIndex": project.picture_index,
        }

    @staticmethod
    def bridge_generator(_request) -> str:
        return (
            "from polyhedron_visibility.open_faces import OpenFaceOcclusion3D\n"
            "controller = OpenFaceOcclusion3D(shape)\n"
            "FadeIn(figure)\n"
            "FadeOut(figure)\n"
        )

    @staticmethod
    def revisions(**overrides: str) -> dict[str, str]:
        return {
            "asset_compiler": TEST_ASSET_REVISION,
            "generated_open_face_visibility_3d": TEST_BRIDGE_REVISION,
            "embedded_motion_3d": TEST_CAMERA_REVISION,
            "parallel_camera_core": TEST_CAMERA_CORE_REVISION,
            **overrides,
        }

    def build(self, project: Path, **kwargs):
        return build_project(
            project,
            shape_asset_builder=kwargs.pop("shape_asset_builder", self.shape_builder),
            bridge_generator=self.bridge_generator,
            component_revisions=kwargs.pop("component_revisions", self.revisions()),
            **kwargs,
        )

    def output_snapshot(self) -> dict[str, bytes]:
        output = self.root / ".derived"
        return {
            path.relative_to(output).as_posix(): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }

    def test_loader_tracks_camera_shots_and_rejects_a_second_camera_owner(self) -> None:
        project_path = self.write_project(bridge=False)
        project = load_source_project(project_path)
        self.assertEqual(
            project.camera_shots,
            (self.root / "camera-shots.json").resolve(),
        )

        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw["motionJson"] = "motion.json"
        self.write("motion.json", '{"timeline":[]}\n')
        project_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(
            SourceProjectError,
            "cameraShots and motionJson cannot both be present",
        ):
            load_source_project(project_path)

    def test_build_emits_canonical_node_manifest_and_immutable_source_binding(
        self,
    ) -> None:
        project_path = self.write_project(hooks=True)
        result = self.build(project_path)
        self.assertEqual(
            result.built,
            ("shape", "camera_shots", "compositing", "generated_source"),
        )

        output = self.root / ".derived"
        expected_camera = canonical_parallel_camera_shot_sequence_json(_sequence())
        camera_bytes = (output / "camera-shots.json").read_bytes()
        self.assertEqual(camera_bytes, expected_camera.encode("utf-8"))

        compositing = json.loads(
            (output / "unified-compositing.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            compositing["cameraShotsAssetSha256"],
            hashlib.sha256(camera_bytes).hexdigest(),
        )
        self.assertIsNone(compositing["motionAssetSha256"])

        manifest = json.loads(
            (output / "build-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["inputs"]["cameraShots"]["path"],
            "camera-shots.json",
        )
        self.assertEqual(
            manifest["nodes"]["camera_shots"]["output"],
            "camera-shots.json",
        )
        self.assertEqual(
            manifest["componentRevisions"]["parallel_camera_core"],
            TEST_CAMERA_CORE_REVISION,
        )

        generated = (output / "generated_scene.py").read_text(encoding="utf-8")
        self.assertIn("TIKZ_NATIVE_CAMERA_SHOTS =", generated)
        self.assertIn("# >>> TIKZ_NATIVE_CAMERA_SHOTS_V1", generated)
        self.assertIn("# >>> TIKZ_NATIVE_USER_HOOKS_V1", generated)
        self.assertLess(
            generated.index("# >>> TIKZ_NATIVE_CAMERA_SHOTS_V1"),
            generated.index("# >>> TIKZ_NATIVE_USER_HOOKS_V1"),
        )
        self.assertNotIn("play_parallel_camera_shot", generated)
        compile(generated, "generated_scene.py", "exec")

    def test_generated_binding_reconstructs_frozen_data_without_playback(self) -> None:
        canonical = canonical_parallel_camera_shot_sequence_json(_sequence())
        generated = _append_camera_shots_binding("BRIDGE_VALUE = 7\n", canonical)
        namespace: dict[str, object] = {}

        exec(compile(generated, "generated_scene.py", "exec"), namespace)

        sequence = namespace["TIKZ_NATIVE_CAMERA_SHOTS"]
        self.assertIsInstance(sequence, ParallelCameraShotSequence)
        self.assertEqual(
            canonical_parallel_camera_shot_sequence_json(sequence),
            canonical,
        )
        self.assertFalse(sequence.shots[0].state.matrix.flags.writeable)
        self.assertNotIn("play_parallel_camera_shot_sequence", namespace)

        with self.assertRaisesRegex(
            SourceProjectBuildError,
            "reserved cameraShots binding",
        ):
            _append_camera_shots_binding(
                "TIKZ_NATIVE_CAMERA_SHOTS = 'bridge-owned'\n",
                canonical,
            )

    def test_camera_change_has_narrow_cache_invalidation(self) -> None:
        project_path = self.write_project()
        first = self.build(project_path)
        second = self.build(project_path)
        self.assertEqual(second.built, ())
        self.assertEqual(
            second.reused,
            ("shape", "camera_shots", "compositing", "generated_source"),
        )

        self.write_camera_shots(sequence=_sequence(second_zoom=1.35), pretty=False)
        camera_changed = self.build(project_path)
        self.assertEqual(camera_changed.reused, ("shape",))
        self.assertEqual(
            camera_changed.built,
            ("camera_shots", "compositing", "generated_source"),
        )
        self.assertEqual(camera_changed.painter_z_band, first.painter_z_band)

        self.write(
            "figure.tex",
            "\\begin{tikzpicture}\\draw (0,0) circle (1);\\end{tikzpicture}\n",
        )
        tikz_changed = self.build(project_path)
        self.assertEqual(tikz_changed.reused, ("camera_shots",))
        self.assertEqual(
            tikz_changed.built,
            ("shape", "compositing", "generated_source"),
        )

    def test_camera_core_revision_invalidates_only_camera_dependents(self) -> None:
        project_path = self.write_project(hooks=True)
        self.build(project_path)

        changed = self.build(
            project_path,
            component_revisions=self.revisions(
                parallel_camera_core="camera-shots-test-core-v2"
            ),
        )

        self.assertEqual(changed.reused, ("shape",))
        self.assertEqual(
            changed.built,
            ("camera_shots", "compositing", "generated_source"),
        )

    def test_legacy_embedded_revision_override_still_invalidates_camera(self) -> None:
        project_path = self.write_project(hooks=True)
        legacy = self.revisions()
        legacy.pop("parallel_camera_core")
        self.build(project_path, component_revisions=legacy)

        changed = self.build(
            project_path,
            component_revisions={
                **legacy,
                "embedded_motion_3d": "camera-shots-test-runtime-v2",
            },
        )

        self.assertEqual(changed.reused, ("shape",))
        self.assertEqual(
            changed.built,
            ("camera_shots", "compositing", "generated_source"),
        )

    def test_removing_camera_input_removes_derived_node_and_output(self) -> None:
        project_path = self.write_project(bridge=False)
        self.build(project_path)
        raw = json.loads(project_path.read_text(encoding="utf-8"))
        raw.pop("cameraShots")
        project_path.write_text(json.dumps(raw), encoding="utf-8")

        status = status_project(
            project_path,
            shape_asset_builder=self.shape_builder,
            bridge_generator=self.bridge_generator,
            component_revisions=self.revisions(),
        )
        obsolete = {node.name: node.action for node in status.nodes}
        self.assertEqual(obsolete["camera_shots"], "obsolete")

        rebuilt = self.build(project_path)
        self.assertNotIn("camera_shots", {node.name for node in rebuilt.nodes})
        self.assertFalse((self.root / ".derived/camera-shots.json").exists())
        manifest = json.loads(
            (self.root / ".derived/build-manifest.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("cameraShots", manifest["inputs"])
        self.assertNotIn("camera_shots", manifest["nodes"])
        compositing = json.loads(
            (self.root / ".derived/unified-compositing.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("cameraShotsAssetSha256", compositing)

    def test_invalid_or_concurrently_changed_camera_input_never_publishes(
        self,
    ) -> None:
        project_path = self.write_project()
        self.build(project_path)
        baseline = self.output_snapshot()

        self.write("camera-shots.json", '{"schema":"parallel-shot-sequence/v1"}\n')
        with self.assertRaisesRegex(SourceProjectError, "invalid cameraShots sequence"):
            self.build(project_path)
        self.assertEqual(self.output_snapshot(), baseline)

        self.write_camera_shots(sequence=_sequence(second_zoom=1.4))

        def mutating_builder(project, source: str):
            self.write_camera_shots(sequence=_sequence(second_zoom=1.6))
            return self.shape_builder(project, source)

        with self.assertRaisesRegex(
            SourceProjectBuildError,
            "authoritative input changed during build",
        ):
            self.build(
                project_path,
                force=True,
                shape_asset_builder=mutating_builder,
            )
        self.assertEqual(self.output_snapshot(), baseline)

    def test_public_schemas_accept_complete_build_and_reject_dual_inputs(self) -> None:
        project_path = self.write_project()
        self.build(project_path)
        source_schema = json.loads(
            (ROOT / "tikz_native/schemas/tikz-native-source-project-v1.schema.json")
            .read_text(encoding="utf-8")
        )
        build_schema = json.loads(
            (ROOT / "tikz_native/schemas/tikz-native-build-manifest-v1.schema.json")
            .read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(source_schema)
        Draft202012Validator.check_schema(build_schema)
        project_value = json.loads(project_path.read_text(encoding="utf-8"))
        build_value = json.loads(
            (self.root / ".derived/build-manifest.json").read_text(encoding="utf-8")
        )
        Draft202012Validator(source_schema).validate(project_value)
        Draft202012Validator(build_schema).validate(build_value)

        legacy_build_value = json.loads(json.dumps(build_value))
        legacy_revisions = legacy_build_value["componentRevisions"]
        legacy_revisions["embedded_motion_3d"] = legacy_revisions.pop(
            "parallel_camera_core"
        )
        Draft202012Validator(build_schema).validate(legacy_build_value)

        project_value["motionJson"] = "motion.json"
        errors = list(Draft202012Validator(source_schema).iter_errors(project_value))
        self.assertTrue(errors)

        build_value["inputs"]["motionJson"] = build_value["inputs"]["cameraShots"]
        errors = list(Draft202012Validator(build_schema).iter_errors(build_value))
        self.assertTrue(errors)

    def test_real_v3_bridge_keeps_its_local_camera(self) -> None:
        shutil.copy2(REAL_V3_SOURCE, self.root / "figure.tex")
        self.write_camera_shots()
        self.write("bridge.json", "{}\n")
        project_path = self.write(
            "project.json",
            json.dumps(
                {
                    "schemaVersion": SOURCE_PROJECT_SCHEMA_VERSION,
                    "tikzSource": "figure.tex",
                    "cameraShots": "camera-shots.json",
                    "bridgeRequestTemplate": "bridge.json",
                    "derivedOutput": ".derived",
                }
            ),
        )

        with tempfile.TemporaryDirectory(
            prefix="source-camera-shots-real-v3-media-"
        ) as media_directory, tempconfig({"media_dir": media_directory}):
            result = build_project(project_path)

        self.assertEqual(
            result.built,
            ("shape", "camera_shots", "compositing", "generated_source"),
        )
        generated = (self.root / ".derived/generated_scene.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def local_camera_matrix(state):", generated)
        self.assertIn("TIKZ_NATIVE_CAMERA_SHOTS =", generated)
        self.assertNotIn("play_parallel_camera_shot", generated)
        namespace: dict[str, object] = {}
        exec(compile(generated, "generated_scene.py", "exec"), namespace)
        self.assertIsInstance(
            namespace["TIKZ_NATIVE_CAMERA_SHOTS"],
            ParallelCameraShotSequence,
        )


if __name__ == "__main__":
    unittest.main()
