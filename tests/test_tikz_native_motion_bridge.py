from __future__ import annotations

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

from tikz_native.motion_bridge import (
    MOTION_BRIDGE_OPERATION,
    MOTION_BRIDGE_REQUEST_SCHEMA,
    MOTION_BRIDGE_RESPONSE_SCHEMA,
    execute_motion_request,
    health_response,
)
from tikz_native.motion_render import MotionRenderError
from tikz_native.provider import sha256_file


PROVIDER_ROOT = Path(__file__).parents[1]
EXAMPLE = PROVIDER_ROOT / "examples" / "analytic_geometry_ellipse_demo"
SOURCE = EXAMPLE / "ellipse_problem.tex"
MOTION = EXAMPLE / "ellipse_problem.motion.json"
SCHEMAS = PROVIDER_ROOT / "tikz_native" / "schemas"


def _request(source: Path, motion: Path, output_dir: Path) -> dict:
    return {
        "schema": MOTION_BRIDGE_REQUEST_SCHEMA,
        "operation": MOTION_BRIDGE_OPERATION,
        "job_id": "ellipse-motion-test",
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
            "scene_unit_per_cm": 0.92,
            "strict_native": True,
            "view_mode": "tikz_fixed",
        },
        "render": {
            "profile": "preview",
            "layout": "ellipse_chord_analysis",
            "pixel_width": 320,
            "pixel_height": 180,
            "frame_rate": 5,
            "background": "#F6F8FC",
        },
        "output_dir": str(output_dir),
    }


class TikzNativeMotionBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.request_schema = json.loads(
            (SCHEMAS / "motion-bridge-request-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.response_schema = json.loads(
            (SCHEMAS / "motion-bridge-response-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(cls.request_schema)
        Draft202012Validator.check_schema(cls.response_schema)

    def test_public_schemas_and_health_response_are_self_consistent(self) -> None:
        self.assertEqual(
            self.request_schema["properties"]["schema"]["const"],
            MOTION_BRIDGE_REQUEST_SCHEMA,
        )
        self.assertEqual(
            self.response_schema["properties"]["schema"]["const"],
            MOTION_BRIDGE_RESPONSE_SCHEMA,
        )
        request = _request(SOURCE, MOTION, Path("unused-output"))
        Draft202012Validator(self.request_schema).validate(request)
        response = health_response()
        Draft202012Validator(self.response_schema).validate(response)
        self.assertTrue(response["ok"])
        self.assertIn("revision", response["provider"])

    def test_unknown_request_field_is_rejected_without_creating_output(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "unknown-field"
            request = _request(SOURCE, MOTION, output_dir)
            request["run_python"] = "arbitrary"
            response = execute_motion_request(request)
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "validate_request")
            self.assertIn("unsupported fields", response["error"]["message"])
            self.assertFalse(output_dir.exists())
            Draft202012Validator(self.response_schema).validate(response)

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
                    "tikz_native.motion_bridge",
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

    def test_source_and_motion_hash_mismatches_fail_before_staging(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            source_request = _request(SOURCE, MOTION, temporary / "source-mismatch")
            source_request["input"]["source_sha256"] = "0" * 64
            source_response = execute_motion_request(source_request)
            self.assertFalse(source_response["ok"])
            self.assertEqual(source_response["error"]["code"], "HASH_MISMATCH")
            self.assertEqual(source_response["error"]["details"]["asset"], "source")
            self.assertFalse((temporary / "source-mismatch").exists())

            motion_request = _request(SOURCE, MOTION, temporary / "motion-mismatch")
            motion_request["input"]["motion_sha256"] = "f" * 64
            motion_response = execute_motion_request(motion_request)
            self.assertFalse(motion_response["ok"])
            self.assertEqual(motion_response["error"]["code"], "HASH_MISMATCH")
            self.assertEqual(motion_response["error"]["details"]["asset"], "motion")
            self.assertFalse((temporary / "motion-mismatch").exists())

    def test_invalid_motion_config_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            invalid_motion = temporary / "invalid.motion.json"
            payload = json.loads(MOTION.read_text(encoding="utf-8"))
            payload["python"] = "__import__('os').system('false')"
            invalid_motion.write_text(json.dumps(payload), encoding="utf-8")
            output_dir = temporary / "invalid-output"
            response = execute_motion_request(
                _request(SOURCE, invalid_motion, output_dir)
            )
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "validate_motion")
            self.assertIn("unsupported fields", response["error"]["message"])
            self.assertFalse(output_dir.exists())

    def test_render_failure_leaves_existing_empty_target_empty(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "empty-target"
            output_dir.mkdir()
            request = _request(SOURCE, MOTION, output_dir)
            with patch(
                "tikz_native.motion_bridge.render_motion_preview",
                side_effect=MotionRenderError("synthetic render failure"),
            ):
                response = execute_motion_request(request)
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "render_motion")
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(list(output_dir.iterdir()), [])
            self.assertFalse(
                any(output_dir.parent.glob(f".{output_dir.name}.motion-stage-*"))
            )

    def test_nonempty_complete_target_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "complete-target"
            output_dir.mkdir()
            marker = output_dir / "keep.mp4"
            marker.write_bytes(b"previous-complete-package")
            response = execute_motion_request(_request(SOURCE, MOTION, output_dir))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "prepare_job")
            self.assertEqual(marker.read_bytes(), b"previous-complete-package")
            self.assertEqual(list(output_dir.iterdir()), [marker])

    @unittest.skipUnless(
        os.environ.get("RUN_TIKZ_NATIVE_MOTION_RENDER_TEST") == "1",
        "set RUN_TIKZ_NATIVE_MOTION_RENDER_TEST=1 for the real Manim render",
    )
    def test_real_low_resolution_render_publishes_a_complete_atomic_package(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            compact_motion = temporary / "compact.motion.json"
            payload = deepcopy(json.loads(MOTION.read_text(encoding="utf-8")))
            payload["timeline"] = [
                {
                    "to": payload["driver"]["range"][0],
                    "duration": 0.25,
                    "hold": 0.0,
                }
            ]
            compact_motion.write_text(json.dumps(payload), encoding="utf-8")
            output_dir = temporary / "published"
            request = _request(SOURCE, compact_motion, output_dir)
            request["render"].update(
                {"pixel_width": 854, "pixel_height": 480, "frame_rate": 15}
            )
            request_path = temporary / "render-request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tikz_native.motion_bridge",
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
            self.assertEqual(response["package"]["media"]["codec"], "h264")
            self.assertEqual(response["package"]["media"]["width"], 854)
            self.assertEqual(response["package"]["media"]["height"], 480)
            self.assertEqual(
                response["provider"]["revision"],
                json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))[
                    "provider"
                ]["revision"],
            )
            files = response["package"]["files"]
            expected = {
                "source",
                "motion",
                "request",
                "document_manifest",
                "compatibility",
                "animation_plan",
                "asset",
                "video",
                "first_frame",
                "last_frame",
                "trace",
                "manifest",
            }
            self.assertEqual(set(files), expected)
            for record in files.values():
                relative = Path(record["path"])
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                artifact = output_dir / relative
                self.assertTrue(artifact.is_file())
                self.assertEqual(sha256_file(artifact), record["sha256"])
                self.assertEqual(artifact.stat().st_size, record["bytes"])

            self.assertEqual(
                response["package"]["manifest_sha256"],
                sha256_file(output_dir / response["package"]["manifest_path"]),
            )
            for key in ("first_frame", "last_frame"):
                with Image.open(output_dir / files[key]["path"]) as image:
                    self.assertEqual(image.size, (854, 480))
            trace = json.loads(
                (output_dir / files["trace"]["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(trace["schema"], "tikz-native-motion-trace/v1")
            self.assertEqual(len(trace["boundaries"]), 2)
            self.assertTrue((output_dir / "response.json").is_file())
            self.assertFalse(
                any(output_dir.parent.glob(f".{output_dir.name}.motion-stage-*"))
            )


if __name__ == "__main__":
    unittest.main()
