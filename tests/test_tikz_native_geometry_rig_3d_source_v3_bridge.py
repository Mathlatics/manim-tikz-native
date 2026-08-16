from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jsonschema import Draft202012Validator, RefResolver

from tikz_native.geometry_rig_3d_source_v3_bridge import (
    GEOMETRY_RIG_3D_SOURCE_V3_OPERATION,
    GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA,
    execute_source_v3_request,
    health_response,
)
from tikz_native.version import (
    COMPONENT_ASSET_COMPILER,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
    provider_component_revision,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"
SCHEMA_ROOT = ROOT / "tikz_native" / "schemas"


class TikzNativeGeometryRig3DSourceV3BridgeTests(unittest.TestCase):
    def _request(self, source_path: Path, *, source_sha: str | None = None):
        return {
            "schema": GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA,
            "operation": GEOMETRY_RIG_3D_SOURCE_V3_OPERATION,
            "job_id": "source-v3-test",
            "input": {
                "source_path": str(source_path),
                "source_sha256": source_sha or hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "entry_macro": None,
                "picture_index": 1,
                "expected_asset_provider_revision": provider_component_revision(
                    COMPONENT_ASSET_COMPILER
                ),
            },
        }

    def test_health_uses_the_independent_v3_component(self) -> None:
        payload = health_response()
        self.assertTrue(payload["ok"])
        self.assertEqual(
            payload["provider"]["revision_component"],
            COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
        )
        self.assertEqual(
            payload["provider"]["revision"],
            provider_component_revision(COMPONENT_NATIVE_MANIM_SOURCE_3D_V3),
        )
        self.assertTrue(
            payload["provider"]["capabilities"]["native_manim_source_3d_v3"]
        )

    def test_real_dihedral_request_returns_cross_checkable_v3_source(self) -> None:
        with TemporaryDirectory(prefix="tikz-native-source-v3-test-") as temporary:
            source = Path(temporary) / "source.tex"
            source.write_bytes(SOURCE.read_bytes())
            payload = execute_source_v3_request(self._request(source))
        self.assertTrue(payload["ok"], payload.get("error"))
        result = payload["result"]
        self.assertEqual(result["schema"], "tikz-native-geometry-rig-3d-source-v3/v1")
        self.assertEqual(result["dimension"], 3)
        self.assertTrue(result["revisionMatch"])
        self.assertEqual(
            result["assetProviderRevision"],
            provider_component_revision(COMPONENT_ASSET_COMPILER),
        )
        source_v3 = result["nativeManimSourceV3"]
        self.assertEqual(source_v3["schema"], "tikz-native-manim-source-3d/v3")
        self.assertEqual(
            source_v3["sourceSha256"],
            hashlib.sha256(source_v3["sourceText"].encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            source_v3["visibilitySpec"], source_v3["authoringSpec"]["visibility"]
        )

    def test_hash_mismatch_fails_before_source_generation(self) -> None:
        payload = execute_source_v3_request(self._request(SOURCE, source_sha="0" * 64))
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "HASH_MISMATCH")
        self.assertNotIn("result", payload)

    def test_public_v3_schemas_validate_real_response(self) -> None:
        with TemporaryDirectory(prefix="tikz-native-source-v3-schema-") as temporary:
            source = Path(temporary) / "source.tex"
            source.write_bytes(SOURCE.read_bytes())
            response = execute_source_v3_request(self._request(source))
        schemas = {
            path.as_uri(): json.loads(path.read_text(encoding="utf-8"))
            for path in SCHEMA_ROOT.glob("*.schema.json")
        }
        by_id = {
            payload["$id"]: payload
            for payload in schemas.values()
            if isinstance(payload.get("$id"), str)
        }
        result_schema = by_id["urn:tikz-native:geometry-rig-3d-source-v3:v1"]
        response_schema = by_id[
            "urn:tikz-native:geometry-rig-3d-source-v3-bridge:response:v1"
        ]
        resolver = RefResolver.from_schema(response_schema, store=by_id)
        Draft202012Validator(response_schema, resolver=resolver).validate(response)
        Draft202012Validator(result_schema).validate(response["result"])


if __name__ == "__main__":
    unittest.main()
