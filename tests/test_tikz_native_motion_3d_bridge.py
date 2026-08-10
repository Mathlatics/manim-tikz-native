from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
from PIL import Image

from tikz_native.motion_3d_bridge import (
    MOTION_3D_BRIDGE_OPERATION,
    MOTION_3D_BRIDGE_PROFILE,
    MOTION_3D_BRIDGE_PROJECTION,
    MOTION_3D_BRIDGE_REQUEST_SCHEMA,
    MOTION_3D_BRIDGE_RESPONSE_SCHEMA,
    MOTION_3D_ASSET_SCHEMA,
    execute_motion_3d_request,
    health_response,
)
from tikz_native.motion_3d_render import (
    ASSET_SCHEMA,
    MOTION_3D_PREVIEW_PROFILE,
    Motion3DRenderError,
    build_motion_3d_trace,
    compile_motion_3d_asset,
)
from tikz_native.motion_3d import Motion3DSpec
from tikz_native.provider import provider_info, sha256_file


PROVIDER_ROOT = Path(__file__).parents[1]
EXAMPLE = PROVIDER_ROOT / "examples" / "dihedral_fold_3d_demo"
SOURCE = EXAMPLE / "dihedral_fold.tex"
MOTION = EXAMPLE / "motion-3d.json"
SCHEMAS = PROVIDER_ROOT / "tikz_native" / "schemas"
BRIDGE_MODULE = PROVIDER_ROOT / "tikz_native" / "motion_3d_bridge.py"


def _request(source: Path, motion: Path, output_dir: Path) -> dict:
    return {
        "schema": MOTION_3D_BRIDGE_REQUEST_SCHEMA,
        "operation": MOTION_3D_BRIDGE_OPERATION,
        "job_id": "dihedral-motion-3d-test",
        "provider_revision": provider_info()["revision"],
        "input": {
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "motion_path": str(motion),
            "motion_sha256": sha256_file(motion),
            "entry_macro": None,
            "picture_index": 1,
        },
        "conversion": {
            "subset_version": "0.1.0",
            "scene_unit_per_cm": 1.0,
            "strict_native": True,
            "view_mode": "world_3d",
        },
        "render": {
            "profile": MOTION_3D_BRIDGE_PROFILE,
            "projection": MOTION_3D_BRIDGE_PROJECTION,
        },
        "output_dir": str(output_dir),
    }


def _fake_render(_compiled, _motion, output_root, *, profile, trace):
    preview = output_root / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    (preview / "motion-3d.mp4").write_bytes(b"fake-h264-preview")
    (preview / "first.png").write_bytes(b"fake-first-frame")
    (preview / "last.png").write_bytes(b"fake-last-frame")
    (preview / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "video": "preview/motion-3d.mp4",
        "first_frame": "preview/first.png",
        "last_frame": "preview/last.png",
        "trace": "preview/trace.json",
        "media": {
            "codec": "h264",
            "width": profile.pixel_width,
            "height": profile.pixel_height,
            "frame_rate": f"{profile.frame_rate}/1",
            "frame_count": 3,
            "duration_seconds": 0.2,
        },
    }


def _incomplete_render(compiled, motion, output_root, *, profile, trace):
    rendered = _fake_render(
        compiled,
        motion,
        output_root,
        profile=profile,
        trace=trace,
    )
    (output_root / rendered["last_frame"]).unlink()
    return rendered


class TikzNativeMotion3DBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.request_schema = json.loads(
            (SCHEMAS / "motion-3d-bridge-request-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.response_schema = json.loads(
            (SCHEMAS / "motion-3d-bridge-response-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(cls.request_schema)
        Draft202012Validator.check_schema(cls.response_schema)
        cls.compiled = compile_motion_3d_asset(
            SOURCE,
            source_sha256=sha256_file(SOURCE),
            picture_index=1,
            scene_unit_per_cm=1.0,
        )

    def test_public_schemas_health_and_fixed_profile_are_consistent(self) -> None:
        self.assertEqual(
            self.request_schema["properties"]["schema"]["const"],
            MOTION_3D_BRIDGE_REQUEST_SCHEMA,
        )
        self.assertEqual(
            self.response_schema["properties"]["schema"]["const"],
            MOTION_3D_BRIDGE_RESPONSE_SCHEMA,
        )
        request = _request(SOURCE, MOTION, Path("unused-output"))
        Draft202012Validator(self.request_schema).validate(request)
        self.assertEqual(MOTION_3D_PREVIEW_PROFILE.pixel_width, 854)
        self.assertEqual(MOTION_3D_PREVIEW_PROFILE.pixel_height, 480)
        self.assertEqual(MOTION_3D_PREVIEW_PROFILE.frame_rate, 15)
        response = health_response()
        Draft202012Validator(self.response_schema).validate(response)
        self.assertTrue(response["ok"])
        provider = response["provider"]
        self.assertEqual(
            provider["request_schema"], MOTION_3D_BRIDGE_REQUEST_SCHEMA
        )
        self.assertEqual(
            provider["response_schema"], MOTION_3D_BRIDGE_RESPONSE_SCHEMA
        )
        self.assertEqual(provider["operation"], MOTION_3D_BRIDGE_OPERATION)
        self.assertEqual(provider["asset_schema"], MOTION_3D_ASSET_SCHEMA)
        self.assertEqual(provider["asset_schema"], ASSET_SCHEMA)
        self.assertTrue(provider["capabilities"]["render_motion_3d_preview"])
        self.assertTrue(provider["capabilities"]["dynamic_camera_3d_parallel"])

    def test_health_does_not_resolve_3d_render_dependencies(self) -> None:
        with patch(
            "tikz_native.motion_3d_bridge._render_dependencies",
            side_effect=AssertionError("health must not load the 3D renderer"),
        ):
            response = health_response()
        self.assertTrue(response["ok"])
        self.assertEqual(response["operation"], "health")

    def test_bridge_entry_has_no_eager_manim_or_renderer_import(self) -> None:
        tree = ast.parse(BRIDGE_MODULE.read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        forbidden = {
            "manim",
            "motion_3d",
            "motion_3d_render",
            "provider",
            "tikz_native.motion_3d",
            "tikz_native.motion_3d_render",
            "tikz_native.provider",
        }
        self.assertTrue(imports.isdisjoint(forbidden), sorted(imports & forbidden))

    def test_provider_revision_mismatch_fails_before_reading_or_staging(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "provider-mismatch"
            request = _request(SOURCE, MOTION, output_dir)
            request["provider_revision"] = "source-sha256:" + "0" * 64
            response = execute_motion_3d_request(request)
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "PROVIDER_MISMATCH")
            self.assertEqual(response["error"]["phase"], "validate_provider")
            self.assertFalse(output_dir.exists())
            Draft202012Validator(self.response_schema).validate(response)

    def test_hash_mismatches_fail_before_staging(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            for field, digest, expected_asset in (
                ("source_sha256", "0" * 64, "source"),
                ("motion_sha256", "f" * 64, "motion"),
            ):
                output_dir = temporary / field
                request = _request(SOURCE, MOTION, output_dir)
                request["input"][field] = digest
                response = execute_motion_3d_request(request)
                self.assertFalse(response["ok"])
                self.assertEqual(response["error"]["code"], "HASH_MISMATCH")
                self.assertEqual(
                    response["error"]["details"]["asset"], expected_asset
                )
                self.assertFalse(output_dir.exists())

    def test_trace_evaluates_authored_boundaries_and_restores_entry(self) -> None:
        motion = Motion3DSpec.load(MOTION)
        trace = build_motion_3d_trace(
            self.compiled,
            motion,
            provider_revision=provider_info()["revision"],
            source_sha256=sha256_file(SOURCE),
            motion_sha256=sha256_file(MOTION),
            profile=MOTION_3D_PREVIEW_PROFILE,
        )
        self.assertEqual(trace["schema"], "tikz-native-motion-3d-trace/v1")
        self.assertEqual(trace["boundaries"][0]["boundary"], "initial")
        self.assertEqual(trace["boundaries"][-1]["boundary"], "restore")
        self.assertEqual(
            trace["boundaries"][0]["coordinates"],
            trace["boundaries"][-1]["coordinates"],
        )
        self.assertEqual(trace["boundaries"][-1]["camera_mode"], "tikz")
        beta_values = [
            boundary["coordinates"]["Beta0"]
            for boundary in trace["boundaries"]
            if boundary["step_type"] == "driver"
        ]
        self.assertGreater(len({tuple(value) for value in beta_values}), 1)

    def test_invalid_motion_fails_closed_without_replacing_previous_package(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            invalid_motion = temporary / "invalid.motion-3d.json"
            payload = deepcopy(json.loads(MOTION.read_text(encoding="utf-8")))
            payload["python"] = "__import__('os').system('false')"
            invalid_motion.write_text(json.dumps(payload), encoding="utf-8")
            output_dir = temporary / "published"
            output_dir.mkdir()
            marker = output_dir / "previous.mp4"
            marker.write_bytes(b"previous-complete-package")
            response = execute_motion_3d_request(
                _request(SOURCE, invalid_motion, output_dir)
            )
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "validate_motion")
            self.assertEqual(marker.read_bytes(), b"previous-complete-package")
            self.assertFalse(
                any(output_dir.parent.glob(f".{output_dir.name}.motion-3d-stage-*"))
            )

    def test_render_failure_preserves_previous_package(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "published"
            output_dir.mkdir()
            marker = output_dir / "previous.mp4"
            marker.write_bytes(b"previous-complete-package")
            with (
                patch(
                    "tikz_native.motion_3d_bridge.compile_motion_3d_asset",
                    return_value=self.compiled,
                ),
                patch(
                    "tikz_native.motion_3d_bridge.render_motion_3d_preview",
                    side_effect=Motion3DRenderError("synthetic render failure"),
                ),
            ):
                response = execute_motion_3d_request(
                    _request(SOURCE, MOTION, output_dir)
                )
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "render_motion_3d")
            self.assertEqual(marker.read_bytes(), b"previous-complete-package")
            self.assertFalse(
                any(output_dir.parent.glob(f".{output_dir.name}.motion-3d-stage-*"))
            )

    def test_complete_candidate_replaces_previous_package_only_after_validation(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "published"
            output_dir.mkdir()
            (output_dir / "previous.mp4").write_bytes(b"previous-complete-package")
            with (
                patch(
                    "tikz_native.motion_3d_bridge.compile_motion_3d_asset",
                    return_value=self.compiled,
                ),
                patch(
                    "tikz_native.motion_3d_bridge.render_motion_3d_preview",
                    side_effect=_fake_render,
                ),
            ):
                response = execute_motion_3d_request(
                    _request(SOURCE, MOTION, output_dir)
                )
            self.assertTrue(response["ok"], response)
            Draft202012Validator(self.response_schema).validate(response)
            self.assertFalse((output_dir / "previous.mp4").exists())
            self.assertTrue((output_dir / "manifest.json").is_file())
            files = response["package"]["files"]
            expected = {
                "source",
                "motion",
                "request",
                "document_manifest",
                "compatibility",
                "asset",
                "video",
                "first_frame",
                "last_frame",
                "trace",
                "manifest",
            }
            self.assertEqual(set(files), expected)
            for record in files.values():
                artifact = output_dir / record["path"]
                self.assertTrue(artifact.is_file())
                self.assertEqual(sha256_file(artifact), record["sha256"])
                self.assertEqual(artifact.stat().st_size, record["bytes"])
            self.assertFalse(
                any(output_dir.parent.glob(f".{output_dir.name}.motion-3d-backup-*"))
            )

    def test_incomplete_candidate_never_replaces_previous_package(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "published"
            output_dir.mkdir()
            marker = output_dir / "previous.mp4"
            marker.write_bytes(b"previous-complete-package")
            with (
                patch(
                    "tikz_native.motion_3d_bridge.compile_motion_3d_asset",
                    return_value=self.compiled,
                ),
                patch(
                    "tikz_native.motion_3d_bridge.render_motion_3d_preview",
                    side_effect=_incomplete_render,
                ),
            ):
                response = execute_motion_3d_request(
                    _request(SOURCE, MOTION, output_dir)
                )
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "package")
            self.assertEqual(marker.read_bytes(), b"previous-complete-package")
            self.assertFalse(
                any(output_dir.parent.glob(f".{output_dir.name}.motion-3d-stage-*"))
            )

    def test_cli_failure_stdout_is_one_clean_json_document(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            request = _request(SOURCE, MOTION, temporary / "unused")
            request["unknown"] = True
            request_path = temporary / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tikz_native.motion_3d_bridge",
                    "run",
                    "--request",
                    str(request_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            response = json.loads(result.stdout)
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "validate_request")

    @unittest.skipUnless(
        os.environ.get("RUN_TIKZ_NATIVE_MOTION_3D_RENDER_TEST") == "1",
        "set RUN_TIKZ_NATIVE_MOTION_3D_RENDER_TEST=1 for the real Manim render",
    )
    def test_real_fixed_profile_render_publishes_complete_atomic_package(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "published"
            request = _request(SOURCE, MOTION, output_dir)
            request_path = Path(directory) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tikz_native.motion_3d_bridge",
                    "run",
                    "--request",
                    str(request_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            response = json.loads(result.stdout)
            self.assertTrue(response["ok"], response)
            Draft202012Validator(self.response_schema).validate(response)
            media = response["package"]["media"]
            self.assertEqual(media["codec"], "h264")
            self.assertEqual((media["width"], media["height"]), (854, 480))
            for name in ("first_frame", "last_frame"):
                with Image.open(output_dir / response["package"]["files"][name]["path"]) as image:
                    self.assertEqual(image.size, (854, 480))
            self.assertEqual(
                response["package"]["manifest_sha256"],
                sha256_file(output_dir / "manifest.json"),
            )


if __name__ == "__main__":
    unittest.main()
