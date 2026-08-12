from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from .compiler import compile_document
from .geometry_rig_3d import (
    GEOMETRY_RIG_3D_SCHEMA,
    GeometryRig3DError,
    analyze_geometry_rig_3d,
    attach_geometry_rig_3d_identity,
)
from .motion_3d_bridge import (
    ERROR_HASH_MISMATCH,
    ERROR_INPUT,
    ERROR_PROVIDER,
    TikzNativeProviderError,
    sha256_file,
)
from .provider_metadata import provider_info


GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA = (
    "tikz-native-geometry-rig-3d-bridge.request/v1"
)
GEOMETRY_RIG_3D_BRIDGE_RESPONSE_SCHEMA = (
    "tikz-native-geometry-rig-3d-bridge.response/v1"
)
GEOMETRY_RIG_3D_BRIDGE_OPERATION = "analyze_geometry_rig_3d"
EMBEDDED_MOTION_3D_RUNTIME_CONTRACT = "tikz-native-embedded-motion-3d/v1"


def _embedded_runtime_contract() -> str | None:
    """Report the embedded helper only when its public API actually exists."""

    try:
        module = import_module(".motion_3d_runtime", package=__package__)
    except (ImportError, ModuleNotFoundError):
        return None
    contract = getattr(module, "EMBEDDED_MOTION_3D_RUNTIME_CONTRACT", None)
    callback = getattr(module, "play_motion_3d_on_native_shape", None)
    if contract != EMBEDDED_MOTION_3D_RUNTIME_CONTRACT or not callable(callback):
        return None
    return str(contract)


def _bridge_provider_info() -> dict[str, Any]:
    info = dict(provider_info())
    capabilities = dict(info.get("capabilities", {}))
    capabilities["analyze_geometry_rig_3d"] = True
    embedded = _embedded_runtime_contract()
    if embedded is not None:
        capabilities["embedded_motion_3d_runtime"] = True
        info["embeddedShapeRuntime3D"] = embedded
    info["capabilities"] = capabilities
    info["geometry_rig_3d_request_schema"] = GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA
    info["geometry_rig_3d_response_schema"] = GEOMETRY_RIG_3D_BRIDGE_RESPONSE_SCHEMA
    info["geometry_rig_3d_result_schema"] = GEOMETRY_RIG_3D_SCHEMA
    return info


def _reject_unknown(
    payload: Mapping[str, Any], allowed: set[str], field: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{field} contains unsupported fields: {', '.join(unknown)}",
        )


def _object(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"request field {key!r} must be an object",
        )
    return value


def _string(payload: Mapping[str, Any], key: str, *, field: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{field} must be a non-empty string",
        )
    return value.strip()


def _sha256(payload: Mapping[str, Any], key: str, *, field: str) -> str:
    value = _string(payload, key, field=field)
    if len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{field} must be a 64-character hexadecimal digest",
        )
    return value.lower()


def _positive_integer(payload: Mapping[str, Any], key: str, *, field: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{field} must be an integer >= 1",
        )
    return value


def _string_array(value: object, field: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{field} must be an array of non-empty strings",
        )
    if len(value) != len(set(value)):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{field} must not contain duplicates",
        )


def _validate_selection(selection: Mapping[str, Any]) -> None:
    _reject_unknown(
        selection,
        {"candidate_id", "range", "include_object_ids", "exclude_object_ids"},
        "selection",
    )
    if "candidate_id" in selection:
        _string(selection, "candidate_id", field="selection.candidate_id")
    if "range" in selection:
        raw = selection["range"]
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in raw
            )
        ):
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_request",
                "selection.range must be a two-number array",
            )
    for key in ("include_object_ids", "exclude_object_ids"):
        if key in selection:
            _string_array(selection[key], f"selection.{key}")


def _validate_request(request: Mapping[str, Any]) -> None:
    _reject_unknown(
        request,
        {"schema", "operation", "job_id", "input", "selection"},
        "request",
    )
    if request.get("schema") != GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"request schema must be {GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA!r}",
        )
    if request.get("operation") != GEOMETRY_RIG_3D_BRIDGE_OPERATION:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"operation must be {GEOMETRY_RIG_3D_BRIDGE_OPERATION!r}",
        )
    _string(request, "job_id", field="job_id")
    input_payload = _object(request, "input")
    _reject_unknown(
        input_payload,
        {
            "source_path",
            "source_sha256",
            "entry_macro",
            "picture_index",
            "expected_asset_provider_revision",
        },
        "input",
    )
    required = {
        "source_path",
        "source_sha256",
        "picture_index",
        "expected_asset_provider_revision",
    }
    missing = sorted(required - set(input_payload))
    if missing:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "input is missing required fields: " + ", ".join(missing),
        )
    _string(input_payload, "source_path", field="input.source_path")
    _sha256(input_payload, "source_sha256", field="input.source_sha256")
    _positive_integer(input_payload, "picture_index", field="input.picture_index")
    _string(
        input_payload,
        "expected_asset_provider_revision",
        field="input.expected_asset_provider_revision",
    )
    entry_macro = input_payload.get("entry_macro")
    if entry_macro is not None and not isinstance(entry_macro, str):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "input.entry_macro must be a string or null",
        )
    selection = request.get("selection")
    if selection is not None:
        if not isinstance(selection, dict):
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_request",
                "selection must be an object",
            )
        _validate_selection(selection)


def _response_operation(value: object) -> str:
    return (
        GEOMETRY_RIG_3D_BRIDGE_OPERATION
        if value == GEOMETRY_RIG_3D_BRIDGE_OPERATION
        else "unknown"
    )


def _error_response(
    operation: str,
    job_id: str | None,
    error: TikzNativeProviderError,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "schema": GEOMETRY_RIG_3D_BRIDGE_RESPONSE_SCHEMA,
        "ok": False,
        "operation": operation,
        "provider": _bridge_provider_info(),
        "error": error.to_dict(),
    }
    if job_id:
        response["job_id"] = job_id
    return response


def health_response() -> dict[str, Any]:
    return {
        "schema": GEOMETRY_RIG_3D_BRIDGE_RESPONSE_SCHEMA,
        "ok": True,
        "operation": "health",
        "provider": _bridge_provider_info(),
    }


def _selected_picture(document: Any, picture_index: int):
    picture = next(
        (item for item in document.pictures if item.index == picture_index),
        None,
    )
    if picture is None:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "select_picture",
            f"picture_index {picture_index} is not available",
            details={
                "available_picture_indices": [item.index for item in document.pictures]
            },
        )
    if picture.dimension != 3 or picture.projection_3d is None:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "select_picture",
            "Geometry Rig 3D requires a three-dimensional compiled picture",
            details={"actual_dimension": picture.dimension},
        )
    return picture


def execute_geometry_rig_3d_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = _response_operation(
        request.get("operation") if isinstance(request, dict) else None
    )
    job_id_value = request.get("job_id") if isinstance(request, dict) else None
    job_id = (
        job_id_value
        if isinstance(job_id_value, str) and job_id_value.strip()
        else None
    )
    try:
        if not isinstance(request, dict):
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_request",
                "request root must be an object",
            )
        _validate_request(request)
        input_payload = _object(request, "input")
        assert job_id is not None
        source = Path(
            _string(input_payload, "source_path", field="input.source_path")
        ).expanduser().resolve()
        if not source.is_file():
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "read_input",
                f"source file does not exist: {source}",
            )
        expected_hash = _sha256(
            input_payload,
            "source_sha256",
            field="input.source_sha256",
        )
        actual_hash = sha256_file(source)
        if actual_hash != expected_hash:
            raise TikzNativeProviderError(
                ERROR_HASH_MISMATCH,
                "verify_input",
                "source SHA-256 does not match the requested snapshot",
                details={
                    "asset": "source",
                    "expected": expected_hash,
                    "actual": actual_hash,
                },
            )
        try:
            document = compile_document(
                source,
                entry_macro=input_payload.get("entry_macro"),
            )
            picture = _selected_picture(document, int(input_payload["picture_index"]))
            rig = analyze_geometry_rig_3d(
                picture,
                selection=request.get("selection"),
            )
            provider = _bridge_provider_info()
            result = attach_geometry_rig_3d_identity(
                rig,
                source_sha256=expected_hash,
                provider_revision=str(provider["revision"]),
                expected_asset_provider_revision=str(
                    input_payload["expected_asset_provider_revision"]
                ),
            )
        except GeometryRig3DError as error:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "analyze_geometry_rig_3d",
                str(error),
                details={"exception_type": type(error).__name__},
            ) from error
        except TikzNativeProviderError:
            raise
        except Exception as error:
            raise TikzNativeProviderError(
                ERROR_PROVIDER,
                "compile_source",
                f"TikZ-native 3D geometry analysis failed: {error}",
                details={"exception_type": type(error).__name__},
            ) from error
        return {
            "schema": GEOMETRY_RIG_3D_BRIDGE_RESPONSE_SCHEMA,
            "ok": True,
            "operation": GEOMETRY_RIG_3D_BRIDGE_OPERATION,
            "job_id": job_id,
            "provider": provider,
            "result": result,
        }
    except TikzNativeProviderError as error:
        return _error_response(operation, job_id, error)
    except Exception as error:
        wrapped = TikzNativeProviderError(
            ERROR_PROVIDER,
            "geometry_rig_3d_bridge",
            f"Geometry Rig 3D bridge request failed: {error}",
            details={"exception_type": type(error).__name__},
        )
        return _error_response(operation, job_id, wrapped)


def _emit(payload: Mapping[str, Any], output: Path | None = None) -> None:
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only JSON bridge for explicit TikZ-native 3D Geometry Rig analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    health_parser = subparsers.add_parser("health")
    health_parser.add_argument("--output", type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--request", type=Path, required=True)
    run_parser.add_argument("--response", type=Path)
    args = parser.parse_args(argv)

    if args.command == "health":
        _emit(health_response(), args.output)
        return 0
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
    except Exception as error:
        payload = _error_response(
            "unknown",
            None,
            TikzNativeProviderError(
                ERROR_INPUT,
                "read_request",
                f"cannot read request JSON: {error}",
                details={"exception_type": type(error).__name__},
            ),
        )
        _emit(payload, args.response)
        return 2
    with redirect_stdout(sys.stderr):
        payload = execute_geometry_rig_3d_request(request)
    _emit(payload, args.response)
    return 0 if payload.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EMBEDDED_MOTION_3D_RUNTIME_CONTRACT",
    "GEOMETRY_RIG_3D_BRIDGE_OPERATION",
    "GEOMETRY_RIG_3D_BRIDGE_REQUEST_SCHEMA",
    "GEOMETRY_RIG_3D_BRIDGE_RESPONSE_SCHEMA",
    "execute_geometry_rig_3d_request",
    "health_response",
    "main",
]
