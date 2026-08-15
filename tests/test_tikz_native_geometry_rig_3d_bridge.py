from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from tikz_native.geometry_rig_3d_bridge import (
    GEOMETRY_RIG_3D_BRIDGE_OPERATION,
    GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA,
    GEOMETRY_RIG_3D_BRIDGE_RESPONSE_SCHEMA,
    execute_geometry_rig_3d_request,
    health_response,
)
from tikz_native.motion_3d_bridge import provider_info, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "dihedral_fold_3d_demo"
SOURCE = DEMO / "dihedral_fold.tex"
SCHEMAS = ROOT / "tikz_native" / "schemas"


def _request(*, expected_revision: str | None = None) -> dict:
    return {
        "schema": GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA,
        "operation": GEOMETRY_RIG_3D_BRIDGE_OPERATION,
        "job_id": "dihedral-rig-3d-test",
        "input": {
            "source_path": str(SOURCE),
            "source_sha256": sha256_file(SOURCE),
            "entry_macro": None,
            "picture_index": 1,
            "expected_asset_provider_revision": (
                expected_revision or provider_info()["revision"]
            ),
        },
        "selection": {
            "candidate_id": "hinge_fold:fold-angle",
            "range": [0.3141592653589793, 1.9547687622336491],
        },
    }


class TikzNativeGeometryRig3DBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.request_schema = json.loads(
            (SCHEMAS / "geometry-rig-3d-bridge-request-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.response_schema = json.loads(
            (SCHEMAS / "geometry-rig-3d-bridge-response-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.rig_schema = json.loads(
            (SCHEMAS / "geometry-rig-3d-v1.schema.json").read_text(encoding="utf-8")
        )
        for schema in (cls.request_schema, cls.response_schema, cls.rig_schema):
            Draft202012Validator.check_schema(schema)
        registry = Registry().with_resource(
            cls.rig_schema["$id"], Resource.from_contents(cls.rig_schema)
        )
        cls.response_validator = Draft202012Validator(
            cls.response_schema,
            registry=registry,
        )

    def test_public_schemas_and_health_are_self_consistent(self) -> None:
        request = _request()
        Draft202012Validator(self.request_schema).validate(request)
        response = health_response()
        self.response_validator.validate(response)
        self.assertTrue(response["ok"])
        self.assertEqual(response["operation"], "health")
        provider = response["provider"]
        self.assertTrue(provider["capabilities"]["analyze_geometry_rig_3d"])
        self.assertTrue(provider["capabilities"]["native_manim_source_3d_v1"])
        self.assertTrue(provider["capabilities"]["native_manim_source_3d_v2"])
        self.assertEqual(provider["geometry_rig_3d_request_schema"], GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA)
        self.assertEqual(provider["geometry_rig_3d_response_schema"], GEOMETRY_RIG_3D_BRIDGE_RESPONSE_SCHEMA)

    def test_health_reports_embedded_runtime_only_for_the_exact_public_api(self) -> None:
        good_runtime = SimpleNamespace(
            EMBEDDED_MOTION_3D_RUNTIME_CONTRACT="tikz-native-embedded-motion-3d/v1",
            play_motion_3d_on_native_shape=lambda *args, **kwargs: None,
        )
        with patch(
            "tikz_native.geometry_rig_3d_bridge.import_module",
            return_value=good_runtime,
        ):
            provider = health_response()["provider"]
        self.assertEqual(
            provider["embeddedShapeRuntime3D"],
            "tikz-native-embedded-motion-3d/v1",
        )
        self.assertTrue(provider["capabilities"]["embedded_motion_3d_runtime"])

        incomplete_runtime = SimpleNamespace(
            EMBEDDED_MOTION_3D_RUNTIME_CONTRACT="tikz-native-embedded-motion-3d/v0",
            play_motion_3d_on_native_shape=lambda *args, **kwargs: None,
        )
        with patch(
            "tikz_native.geometry_rig_3d_bridge.import_module",
            return_value=incomplete_runtime,
        ):
            provider = health_response()["provider"]
        self.assertNotIn("embeddedShapeRuntime3D", provider)
        self.assertNotIn(
            "embedded_motion_3d_runtime",
            provider["capabilities"],
        )

    def test_bridge_returns_portable_ready_analysis_without_paths(self) -> None:
        request = _request()
        response = execute_geometry_rig_3d_request(request)
        self.assertTrue(response["ok"], response)
        self.response_validator.validate(response)
        result = response["result"]
        self.assertEqual(result["dimension"], 3)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["revisionMatch"])
        self.assertNotIn("source_path", json.dumps(result))
        self.assertEqual(result["sourceSha256"], request["input"]["source_sha256"])
        native_source = result["nativeManimSource"]
        self.assertEqual(
            native_source["schema"],
            "tikz-native-manim-source-3d/v1",
        )
        self.assertIn("def prepare_local_camera(", native_source["sourceText"])
        self.assertNotIn("play_motion_3d_on_native_shape", native_source["sourceText"])
        native_source_v2 = result["nativeManimSourceV2"]
        self.assertEqual(
            native_source_v2["schema"],
            "tikz-native-manim-source-3d/v2",
        )
        self.assertEqual(
            native_source_v2["authoringSpec"]["schema"],
            "tikz-native-manim-authoring-3d/v1",
        )

    def test_discovery_returns_semantic_groups_but_no_executable_motion(self) -> None:
        request = _request()
        request.pop("selection")
        response = execute_geometry_rig_3d_request(request)
        self.assertTrue(response["ok"], response)
        self.response_validator.validate(response)
        result = response["result"]
        self.assertEqual(result["status"], "needs_selection")
        self.assertTrue(result["semanticGroups"])
        self.assertIsNone(result["selectedMotionCandidate"])
        self.assertIsNone(result["motionSpecCore"])
        self.assertIsNone(result["nativeManimSource"])
        self.assertIsNotNone(result["nativeManimSourceV2"])

    def test_revision_mismatch_is_reviewable_but_fail_closed(self) -> None:
        response = execute_geometry_rig_3d_request(
            _request(expected_revision="source-sha256:" + "0" * 64)
        )
        self.assertTrue(response["ok"], response)
        self.response_validator.validate(response)
        result = response["result"]
        self.assertFalse(result["revisionMatch"])
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["motionSpecCore"])
        self.assertIsNone(result["nativeManimSource"])
        self.assertIsNone(result["nativeManimSourceV2"])
        self.assertTrue(result["motionCandidates"])

    def test_hash_mismatch_unknown_fields_and_2d_source_fail_closed(self) -> None:
        request = _request()
        request["input"]["source_sha256"] = "0" * 64
        response = execute_geometry_rig_3d_request(request)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "HASH_MISMATCH")
        self.assertEqual(response["error"]["phase"], "verify_input")

        request = _request()
        request["python"] = "arbitrary"
        response = execute_geometry_rig_3d_request(request)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["phase"], "validate_request")

        two_d = ROOT / "examples" / "analytic_geometry_ellipse_demo" / "ellipse_problem.tex"
        request = _request()
        request["input"]["source_path"] = str(two_d)
        request["input"]["source_sha256"] = sha256_file(two_d)
        response = execute_geometry_rig_3d_request(request)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["phase"], "select_picture")
        self.assertIn("three-dimensional", response["error"]["message"])

    def test_real_cli_emits_one_clean_json_document(self) -> None:
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(
                json.dumps(_request(), ensure_ascii=False), encoding="utf-8"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tikz_native.geometry_rig_3d_bridge",
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


if __name__ == "__main__":
    unittest.main()
