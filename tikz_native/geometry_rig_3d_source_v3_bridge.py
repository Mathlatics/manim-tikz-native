from __future__ import annotations

"""Independent Bridge for the additive open-face Manim source v3.

The legacy Geometry Rig 3D response stays byte-for-byte compatible.  This
Bridge reuses that frozen analysis as its identity proof, then publishes only
the separately versioned v3 authoring source.
"""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from .compiler import compile_document
from .geometry_rig_3d import analyze_geometry_rig_3d
from .geometry_rig_3d_bridge import (
    GEOMETRY_RIG_3D_BRIDGE_OPERATION,
    GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA,
    execute_geometry_rig_3d_request,
)
from .motion_3d_bridge import (
    ERROR_INPUT,
    ERROR_PROVIDER,
    TikzNativeProviderError,
)
from .native_manim_codegen_3d_v3 import (
    NativeManimCodegen3DV3Error,
    generate_native_manim_source_3d_v3,
)
from .provider_metadata import provider_info
from .version import (
    COMPONENT_ASSET_COMPILER,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
    provider_component_revision,
)


GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA = (
    "tikz-native-geometry-rig-3d-source-v3-bridge.request/v1"
)
GEOMETRY_RIG_3D_SOURCE_V3_RESPONSE_SCHEMA = (
    "tikz-native-geometry-rig-3d-source-v3-bridge.response/v1"
)
GEOMETRY_RIG_3D_SOURCE_V3_RESULT_SCHEMA = (
    "tikz-native-geometry-rig-3d-source-v3/v1"
)
GEOMETRY_RIG_3D_SOURCE_V3_OPERATION = "generate_native_manim_source_3d_v3"


def _bridge_provider_info() -> dict[str, Any]:
    info = dict(provider_info(revision_component=COMPONENT_NATIVE_MANIM_SOURCE_3D_V3))
    capabilities = dict(info.get("capabilities") or {})
    capabilities["generate_native_manim_source_3d_v3"] = True
    capabilities["native_manim_source_3d_v3"] = True
    info["capabilities"] = capabilities
    info["geometry_rig_3d_source_v3_request_schema"] = (
        GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA
    )
    info["geometry_rig_3d_source_v3_response_schema"] = (
        GEOMETRY_RIG_3D_SOURCE_V3_RESPONSE_SCHEMA
    )
    info["geometry_rig_3d_source_v3_result_schema"] = (
        GEOMETRY_RIG_3D_SOURCE_V3_RESULT_SCHEMA
    )
    return info


def _error_response(
    operation: str,
    job_id: str | None,
    error: TikzNativeProviderError,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": GEOMETRY_RIG_3D_SOURCE_V3_RESPONSE_SCHEMA,
        "ok": False,
        "operation": operation,
        "provider": _bridge_provider_info(),
        "error": error.to_dict(),
    }
    if job_id:
        payload["job_id"] = job_id
    return payload


def health_response() -> dict[str, Any]:
    return {
        "schema": GEOMETRY_RIG_3D_SOURCE_V3_RESPONSE_SCHEMA,
        "ok": True,
        "operation": "health",
        "provider": _bridge_provider_info(),
    }


def _validate_request(request: object) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not isinstance(request, dict):
        raise TikzNativeProviderError(
            ERROR_INPUT, "validate_request", "request root must be an object"
        )
    allowed = {"schema", "operation", "job_id", "input", "selection"}
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "request contains unsupported fields: " + ", ".join(unknown),
        )
    if request.get("schema") != GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"request schema must be {GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA!r}",
        )
    if request.get("operation") != GEOMETRY_RIG_3D_SOURCE_V3_OPERATION:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"operation must be {GEOMETRY_RIG_3D_SOURCE_V3_OPERATION!r}",
        )
    job_id = request.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise TikzNativeProviderError(
            ERROR_INPUT, "validate_request", "job_id must be a non-empty string"
        )
    input_payload = request.get("input")
    if not isinstance(input_payload, dict):
        raise TikzNativeProviderError(
            ERROR_INPUT, "validate_request", "input must be an object"
        )
    selection = request.get("selection")
    if selection is None:
        selection_payload: dict[str, Any] = {}
    elif isinstance(selection, dict):
        selection_payload = dict(selection)
    else:
        raise TikzNativeProviderError(
            ERROR_INPUT, "validate_request", "selection must be an object"
        )
    return job_id.strip(), dict(input_payload), selection_payload


def execute_source_v3_request(request: object) -> dict[str, Any]:
    job_id: str | None = None
    try:
        job_id, input_payload, selection = _validate_request(request)
        legacy_request: dict[str, Any] = {
            "schema": GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA,
            "operation": GEOMETRY_RIG_3D_BRIDGE_OPERATION,
            "job_id": job_id,
            "input": input_payload,
        }
        if selection:
            legacy_request["selection"] = selection
        legacy_response = execute_geometry_rig_3d_request(legacy_request)
        if legacy_response.get("ok") is not True:
            error = legacy_response.get("error")
            error = error if isinstance(error, Mapping) else {}
            raise TikzNativeProviderError(
                str(error.get("code") or ERROR_PROVIDER),
                str(error.get("phase") or "geometry_rig_3d"),
                str(error.get("message") or "legacy 3D geometry analysis failed"),
                details=(
                    dict(error.get("details"))
                    if isinstance(error.get("details"), Mapping)
                    else {}
                ),
            )

        source_path = Path(str(input_payload["source_path"])).expanduser().resolve()
        document = compile_document(
            source_path,
            entry_macro=input_payload.get("entry_macro"),
        )
        picture_index = int(input_payload["picture_index"])
        picture = next(
            (item for item in document.pictures if item.index == picture_index),
            None,
        )
        if picture is None or picture.dimension != 3 or picture.projection_3d is None:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "select_picture",
                "source v3 requires one selected three-dimensional picture",
            )
        rig = analyze_geometry_rig_3d(picture, selection=selection)
        try:
            source_v3 = generate_native_manim_source_3d_v3(picture, rig)
        except NativeManimCodegen3DV3Error as exc:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "generate_native_manim_source_3d_v3",
                str(exc),
                details={"exception_type": type(exc).__name__},
            ) from exc

        legacy_result = dict(legacy_response["result"])
        asset_revision = provider_component_revision(COMPONENT_ASSET_COMPILER)
        expected_revision = str(
            input_payload["expected_asset_provider_revision"]
        ).strip()
        result = {
            "schema": GEOMETRY_RIG_3D_SOURCE_V3_RESULT_SCHEMA,
            "dimension": 3,
            "pictureIndex": picture_index,
            "sourceSha256": str(legacy_result["sourceSha256"]),
            "semanticModelHash": str(legacy_result["semanticModelHash"]),
            "status": str(legacy_result["status"]),
            "expectedAssetProviderRevision": expected_revision,
            "assetProviderRevision": asset_revision,
            "revisionMatch": asset_revision == expected_revision,
            "nativeManimSourceV3": source_v3,
        }
        return {
            "schema": GEOMETRY_RIG_3D_SOURCE_V3_RESPONSE_SCHEMA,
            "ok": True,
            "operation": GEOMETRY_RIG_3D_SOURCE_V3_OPERATION,
            "job_id": job_id,
            "provider": _bridge_provider_info(),
            "result": result,
        }
    except TikzNativeProviderError as exc:
        return _error_response(GEOMETRY_RIG_3D_SOURCE_V3_OPERATION, job_id, exc)
    except Exception as exc:
        return _error_response(
            GEOMETRY_RIG_3D_SOURCE_V3_OPERATION,
            job_id,
            TikzNativeProviderError(
                ERROR_PROVIDER,
                "geometry_rig_3d_source_v3_bridge",
                f"source v3 bridge request failed: {exc}",
                details={"exception_type": type(exc).__name__},
            ),
        )


def _emit(payload: Mapping[str, Any], output: Path | None = None) -> None:
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Bridge for TikZ-native open-face Manim source v3."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("health")
    health.add_argument("--output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--response", type=Path)
    args = parser.parse_args(argv)
    if args.command == "health":
        _emit(health_response(), args.output)
        return 0
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
    except Exception as exc:
        payload = _error_response(
            "unknown",
            None,
            TikzNativeProviderError(
                ERROR_INPUT,
                "read_request",
                f"cannot read request JSON: {exc}",
                details={"exception_type": type(exc).__name__},
            ),
        )
        _emit(payload, args.response)
        return 2
    with redirect_stdout(sys.stderr):
        payload = execute_source_v3_request(request)
    _emit(payload, args.response)
    return 0 if payload.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GEOMETRY_RIG_3D_SOURCE_V3_OPERATION",
    "GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA",
    "GEOMETRY_RIG_3D_SOURCE_V3_RESPONSE_SCHEMA",
    "GEOMETRY_RIG_3D_SOURCE_V3_RESULT_SCHEMA",
    "execute_source_v3_request",
    "health_response",
    "main",
]
