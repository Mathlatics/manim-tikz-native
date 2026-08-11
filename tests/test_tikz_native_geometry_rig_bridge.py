from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from tikz_native.geometry_rig_bridge import (
    GEOMETRY_RIG_BRIDGE_OPERATION,
    GEOMETRY_RIG_BRIDGE_REQUEST_SCHEMA,
    GEOMETRY_RIG_BRIDGE_RESPONSE_SCHEMA,
    execute_geometry_rig_request,
    health_response,
)
from tikz_native.provider import provider_info, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "analytic_geometry_ellipse_demo"
SOURCE = DEMO / "ellipse_problem.tex"
SCHEMAS = ROOT / "tikz_native" / "schemas"


def _request(*, expected_revision: str | None = None) -> dict:
    return {
        "schema": GEOMETRY_RIG_BRIDGE_REQUEST_SCHEMA,
        "operation": GEOMETRY_RIG_BRIDGE_OPERATION,
        "job_id": "ellipse-rig-test",
        "input": {
            "source_path": str(SOURCE),
            "source_sha256": sha256_file(SOURCE),
            "entry_macro": None,
            "picture_index": 1,
            "active_object_id": "line.Lstart.Lend",
            "expected_asset_provider_revision": (
                expected_revision or provider_info()["revision"]
            ),
        },
    }


class TikzNativeGeometryRigBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.request_schema = json.loads(
            (SCHEMAS / "geometry-rig-bridge-request-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.response_schema = json.loads(
            (SCHEMAS / "geometry-rig-bridge-response-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.rig_schema = json.loads(
            (SCHEMAS / "geometry-rig-v1.schema.json").read_text(encoding="utf-8")
        )
        for schema in (cls.request_schema, cls.response_schema, cls.rig_schema):
            Draft202012Validator.check_schema(schema)
        registry = Registry().with_resource(
            cls.rig_schema["$id"],
            Resource.from_contents(cls.rig_schema),
        )
        cls.response_validator = Draft202012Validator(
            cls.response_schema,
            registry=registry,
        )

    def test_public_schemas_health_and_request_are_self_consistent(self) -> None:
        request = _request()
        Draft202012Validator(self.request_schema).validate(request)
        response = health_response()
        self.response_validator.validate(response)
        self.assertTrue(response["ok"])
        self.assertEqual(response["operation"], "health")
        self.assertTrue(
            response["provider"]["capabilities"]["analyze_geometry_rig_2d"]
        )
        self.assertTrue(
            response["provider"]["capabilities"]["native_rig_2d_authoring_v1"]
        )
        self.assertTrue(
            response["provider"]["capabilities"]["native_manim_source_2d_v1"]
        )

    def test_analysis_response_is_portable_strict_and_motion_core_has_no_timeline(self) -> None:
        request = _request()
        response = execute_geometry_rig_request(request)
        self.assertTrue(response["ok"], response)
        self.response_validator.validate(response)
        result = response["result"]
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["revisionMatch"])
        self.assertEqual(result["sourceSha256"], request["input"]["source_sha256"])
        self.assertEqual(result["providerRevision"], response["provider"]["revision"])
        self.assertNotIn("source_path", json.dumps(result))
        self.assertEqual(
            set(result["motionSpecCore"]),
            {"driver", "bindings"},
        )
        self.assertEqual(len(result["rigDraft"]["bindings"]), 18)
        native_source = result["nativeManimSource"]
        self.assertEqual(
            native_source["schema"],
            "tikz-native-manim-source-2d/v1",
        )
        self.assertIn("def geometry_coordinates(theta):", native_source["sourceText"])
        self.assertNotIn("NativeGeometryRig2D", native_source["sourceText"])

    def test_revision_mismatch_is_an_explicit_nonfatal_analysis_warning(self) -> None:
        response = execute_geometry_rig_request(
            _request(expected_revision="source-sha256:" + "0" * 64)
        )
        self.assertTrue(response["ok"], response)
        self.response_validator.validate(response)
        result = response["result"]
        self.assertFalse(result["revisionMatch"])
        self.assertEqual(result["status"], "ready")
        self.assertIsNotNone(result["motionSpecCore"])
        self.assertEqual(
            result["diagnostics"][0]["code"], "PROVIDER_REVISION_MISMATCH"
        )

    def test_hash_mismatch_and_unknown_fields_fail_before_analysis(self) -> None:
        hash_request = _request()
        hash_request["input"]["source_sha256"] = "0" * 64
        response = execute_geometry_rig_request(hash_request)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "HASH_MISMATCH")
        self.assertEqual(response["error"]["phase"], "verify_input")
        self.response_validator.validate(response)

        unknown_request = _request()
        unknown_request["run_python"] = "arbitrary"
        response = execute_geometry_rig_request(unknown_request)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["phase"], "validate_request")
        self.assertIn("unsupported fields", response["error"]["message"])
        self.response_validator.validate(response)

    def test_missing_expected_revision_is_rejected(self) -> None:
        request = _request()
        del request["input"]["expected_asset_provider_revision"]
        response = execute_geometry_rig_request(request)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["phase"], "validate_request")
        self.assertIn("expected_asset_provider_revision", response["error"]["message"])

    def test_semantically_unsupported_active_object_returns_reviewable_blocked_result(self) -> None:
        request = _request()
        request["input"]["active_object_id"] = "dot.P"
        response = execute_geometry_rig_request(request)
        self.assertTrue(response["ok"], response)
        self.response_validator.validate(response)
        result = response["result"]
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["motionSpecCore"])
        self.assertEqual(
            result["diagnostics"][0]["code"],
            "ACTIVE_OBJECT_KIND_UNSUPPORTED",
        )

    def test_selection_is_applied_without_writing_source_or_artifacts(self) -> None:
        before = sha256_file(SOURCE)
        request = _request()
        request["selection"] = {
            "candidate_id": "rotate_named_line:Lpath:0",
            "pivot": "F",
            "range": [0.4, 0.9],
            "exclude_object_ids": ["label.R.R"],
        }
        response = execute_geometry_rig_request(request)
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["result"]["excludedObjectIds"], ["label.R.R"])
        self.assertEqual(sha256_file(SOURCE), before)

    def test_real_cli_subprocess_emits_one_clean_json_document(self) -> None:
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(
                json.dumps(_request(), ensure_ascii=False),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tikz_native.geometry_rig_bridge",
                    "run",
                    "--request",
                    str(request_path),
                ],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            response = json.loads(result.stdout)
            self.assertTrue(response["ok"], response)
            self.response_validator.validate(response)
            self.assertEqual(response["result"]["status"], "ready")
            self.assertEqual(list(Path(directory).iterdir()), [request_path])


if __name__ == "__main__":
    unittest.main()
