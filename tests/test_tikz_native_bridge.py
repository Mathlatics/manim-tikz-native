from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from tikz_native.bridge import execute_request, health_response
from tikz_native.provider import sha256_file
from tikz_native.version import REQUEST_SCHEMA, RESPONSE_SCHEMA


FIXTURES = Path(__file__).with_name("fixtures")
SOURCE_2D = FIXTURES / "tikz_native_bridge_2d.tex"
SOURCE_3D = FIXTURES / "tikz_native_bridge_3d_fixed.tex"
SCHEMAS = Path(__file__).parents[1] / "tikz_native" / "schemas"


def _request(
    source: Path,
    output_dir: Path,
    *,
    operation: str,
) -> dict:
    return {
        "schema": REQUEST_SCHEMA,
        "operation": operation,
        "job_id": f"test-{operation}",
        "input": {
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "entry_macro": None,
            "picture_index": 1,
        },
        "conversion": {
            "subset_version": "0.1.0",
            "scene_unit_per_cm": 1.0,
            "strict_native": True,
            "view_mode": "tikz_fixed",
        },
        "render": {
            "quality": "preview",
            "pixel_width": 320,
            "pixel_height": 180,
            "frame_rate": 15,
            "background": "transparent",
        },
        "output_dir": str(output_dir),
    }


class TikzNativeBridgeTests(unittest.TestCase):
    def test_health_is_machine_readable(self) -> None:
        response = health_response()
        self.assertEqual(response["schema"], RESPONSE_SCHEMA)
        self.assertTrue(response["ok"])
        self.assertEqual(response["operation"], "health")

    def test_public_json_schemas_are_valid_json(self) -> None:
        schemas = {}
        for name in (
            "request-v1.schema.json",
            "response-v1.schema.json",
            "asset-v1.schema.json",
        ):
            payload = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
            schemas[name] = payload
            self.assertEqual(payload["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertTrue(payload["$id"].startswith("urn:tikz-native:"))
        self.assertEqual(
            schemas["response-v1.schema.json"]["properties"]["asset"]["$ref"],
            schemas["asset-v1.schema.json"]["$id"],
        )

    def test_compile_asset_job_writes_only_relative_asset_links(self) -> None:
        with TemporaryDirectory() as directory:
            job_dir = Path(directory) / "job-2d"
            response = execute_request(
                _request(SOURCE_2D, job_dir, operation="compile_asset")
            )
            self.assertTrue(response["ok"], response)
            self.assertEqual(response["asset"]["dimension"], 2)
            for relative in response["files"].values():
                self.assertFalse(Path(relative).is_absolute())
                self.assertTrue((job_dir / relative).is_file())
            stored = json.loads(
                (job_dir / "response.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored["asset"]["source_sha256"], sha256_file(SOURCE_2D))

    def test_fixed_view_three_d_job_renders_a_static_png(self) -> None:
        with TemporaryDirectory() as directory:
            job_dir = Path(directory) / "job-3d"
            response = execute_request(
                _request(SOURCE_3D, job_dir, operation="render_static")
            )
            self.assertTrue(response["ok"], response)
            self.assertEqual(response["asset"]["dimension"], 3)
            static_path = job_dir / response["files"]["static"]
            self.assertTrue(static_path.is_file())
            with Image.open(static_path) as image:
                self.assertEqual(image.size, (320, 180))
                self.assertIn(image.mode, {"RGBA", "RGB"})

    def test_nonempty_job_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            job_dir = Path(directory) / "existing"
            job_dir.mkdir()
            (job_dir / "keep.txt").write_text("keep", encoding="utf-8")
            response = execute_request(
                _request(SOURCE_2D, job_dir, operation="compile_asset")
            )
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "prepare_job")
            self.assertEqual((job_dir / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_runtime_validation_rejects_wrong_subset_and_coerced_boolean(self) -> None:
        with TemporaryDirectory() as directory:
            request = _request(
                SOURCE_2D,
                Path(directory) / "wrong-subset",
                operation="compile_asset",
            )
            request["conversion"]["subset_version"] = "999.0"
            response = execute_request(request)
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "PROVIDER_MISMATCH")

            request = _request(
                SOURCE_2D,
                Path(directory) / "wrong-boolean",
                operation="compile_asset",
            )
            request["conversion"]["strict_native"] = "false"
            response = execute_request(request)
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["phase"], "validate_request")

    def test_cli_render_stdout_is_one_clean_json_document(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            request_path = temporary / "request.json"
            request_path.write_text(
                json.dumps(
                    _request(
                        SOURCE_2D,
                        temporary / "job-cli",
                        operation="render_static",
                    )
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tikz_native.bridge",
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
            self.assertEqual(response["operation"], "render_static")


if __name__ == "__main__":
    unittest.main()
