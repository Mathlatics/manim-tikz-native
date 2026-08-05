from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from .provider import (
    ERROR_INPUT,
    ERROR_PROVIDER,
    TikzNativeProviderError,
    compile_asset,
    provider_info,
    render_static_png,
)
from .version import REQUEST_SCHEMA, RESPONSE_SCHEMA


SUPPORTED_OPERATIONS = {"compile_asset", "render_static"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_object(
    payload: dict[str, Any],
    key: str,
    *,
    phase: str = "validate_request",
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            phase,
            f"request field {key!r} must be an object",
        )
    return value


def _require_string(
    payload: dict[str, Any],
    key: str,
    *,
    phase: str = "validate_request",
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TikzNativeProviderError(
            ERROR_INPUT,
            phase,
            f"request field {key!r} must be a non-empty string",
        )
    return value.strip()


def _validate_request(request: dict[str, Any]) -> None:
    if request.get("schema") != REQUEST_SCHEMA:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"request schema must be {REQUEST_SCHEMA!r}",
            details={"actual": request.get("schema")},
        )
    operation = _require_string(request, "operation")
    if operation not in SUPPORTED_OPERATIONS:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"unsupported operation: {operation}",
            details={"supported": sorted(SUPPORTED_OPERATIONS)},
        )
    _require_string(request, "job_id")
    _require_string(request, "output_dir")
    input_payload = _require_object(request, "input")
    _require_string(input_payload, "source_path")
    expected_hash = _require_string(input_payload, "source_sha256")
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_hash
    ):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "input.source_sha256 must be a 64-character hexadecimal digest",
        )
    picture_index = input_payload.get("picture_index")
    if (
        not isinstance(picture_index, int)
        or isinstance(picture_index, bool)
        or picture_index < 1
    ):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "input.picture_index must be a positive integer",
        )
    entry_macro = input_payload.get("entry_macro")
    if entry_macro is not None and not isinstance(entry_macro, str):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "input.entry_macro must be a string or null",
        )

    conversion = request.get("conversion", {})
    if not isinstance(conversion, dict):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "request field 'conversion' must be an object",
        )
    subset_version = conversion.get("subset_version")
    supported_subsets = provider_info()["subset_versions"]
    if subset_version is not None and subset_version not in supported_subsets:
        raise TikzNativeProviderError(
            ERROR_PROVIDER,
            "validate_request",
            f"unsupported TikZ-native subset version: {subset_version!r}",
            details={"supported_subset_versions": supported_subsets},
        )
    scene_unit_per_cm = conversion.get("scene_unit_per_cm", 1.0)
    if (
        not isinstance(scene_unit_per_cm, (int, float))
        or isinstance(scene_unit_per_cm, bool)
        or scene_unit_per_cm <= 0
    ):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "conversion.scene_unit_per_cm must be a positive number",
        )
    strict_native = conversion.get("strict_native", True)
    if not isinstance(strict_native, bool):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "conversion.strict_native must be a boolean",
        )
    view_mode = conversion.get("view_mode", "tikz_fixed")
    if view_mode != "tikz_fixed":
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "conversion.view_mode must be 'tikz_fixed'",
        )

    render = request.get("render", {})
    if not isinstance(render, dict):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "request field 'render' must be an object",
        )
    quality = render.get("quality", "preview")
    if quality not in {"preview", "formal"}:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "render.quality must be 'preview' or 'formal'",
        )
    for key, default in (
        ("pixel_width", 1280),
        ("pixel_height", 720),
        ("frame_rate", 30),
    ):
        value = render.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_request",
                f"render.{key} must be a positive integer",
            )
    background = render.get("background", "#FFFFFF")
    if not isinstance(background, str) or not background.strip():
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "render.background must be a non-empty string",
        )


def _success_response(
    operation: str,
    job_id: str,
    *,
    asset: dict[str, Any],
    files: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": RESPONSE_SCHEMA,
        "ok": True,
        "operation": operation,
        "job_id": job_id,
        "provider": provider_info(),
        "asset": asset,
        "files": files,
    }


def _error_response(
    operation: str | None,
    job_id: str | None,
    error: TikzNativeProviderError,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": RESPONSE_SCHEMA,
        "ok": False,
        "operation": operation or "unknown",
        "provider": provider_info(),
        "error": error.to_dict(),
    }
    if job_id:
        payload["job_id"] = job_id
    return payload


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = request.get("operation") if isinstance(request, dict) else None
    job_id = request.get("job_id") if isinstance(request, dict) else None
    job_dir: Path | None = None
    try:
        if not isinstance(request, dict):
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_request",
                "request root must be an object",
            )
        _validate_request(request)
        assert isinstance(operation, str)
        assert isinstance(job_id, str)
        input_payload = _require_object(request, "input")
        conversion = request.get("conversion", {})
        render = request.get("render", {})

        source = Path(_require_string(input_payload, "source_path")).expanduser().resolve()
        if not source.is_file():
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "read_input",
                f"source file does not exist: {source}",
            )
        job_dir = Path(_require_string(request, "output_dir")).expanduser().resolve()
        if job_dir.exists() and any(job_dir.iterdir()):
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "prepare_job",
                f"output_dir must be empty for an isolated job: {job_dir}",
            )

        snapshot = job_dir / "input" / "source.tex"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        _write_json(job_dir / "input" / "request.json", request)

        expected_hash = _require_string(input_payload, "source_sha256").lower()
        picture_index = int(input_payload["picture_index"])
        entry_macro = input_payload.get("entry_macro")
        scene_unit_per_cm = float(conversion.get("scene_unit_per_cm", 1.0))
        strict_native = conversion.get("strict_native", True)
        view_mode = conversion.get("view_mode", "tikz_fixed")

        compiled = compile_asset(
            snapshot,
            source_sha256=expected_hash,
            entry_macro=entry_macro,
            picture_index=picture_index,
            scene_unit_per_cm=scene_unit_per_cm,
            strict_native=strict_native,
            view_mode=view_mode,
        )

        compile_dir = job_dir / "compile"
        manifest_path = compile_dir / "manifest.json"
        compatibility_path = compile_dir / "compatibility.json"
        animation_plan_path = compile_dir / "animation_plan.json"
        asset_path = compile_dir / "asset.json"
        compiled.document.write_json(manifest_path)
        _write_json(compatibility_path, compiled.compatibility)
        _write_json(animation_plan_path, compiled.animation_plan)

        files = {
            "source": "input/source.tex",
            "request": "input/request.json",
            "manifest": "compile/manifest.json",
            "compatibility": "compile/compatibility.json",
            "animation_plan": "compile/animation_plan.json",
            "asset": "compile/asset.json",
        }
        if operation == "render_static":
            static_path = job_dir / "preview" / "static.png"
            background = str(render.get("background", "#FFFFFF"))
            transparent = background.lower() == "transparent"
            render_static_png(
                compiled,
                static_path,
                pixel_width=int(render.get("pixel_width", 1280)),
                pixel_height=int(render.get("pixel_height", 720)),
                frame_rate=int(render.get("frame_rate", 30)),
                background="#FFFFFF" if transparent else background,
                transparent=transparent,
                media_dir=job_dir / "preview" / "media",
            )
            files["static"] = "preview/static.png"

        asset = dict(compiled.asset)
        asset["files"] = dict(files)
        _write_json(asset_path, asset)
        response = _success_response(
            operation,
            job_id,
            asset=asset,
            files=files,
        )
        _write_json(job_dir / "response.json", response)
        return response
    except TikzNativeProviderError as error:
        response = _error_response(
            operation if isinstance(operation, str) else None,
            job_id if isinstance(job_id, str) else None,
            error,
        )
        if job_dir is not None:
            _write_json(job_dir / "response.json", response)
        return response
    except Exception as error:
        wrapped = TikzNativeProviderError(
            ERROR_INPUT,
            "bridge",
            f"bridge request failed: {error}",
            details={"exception_type": type(error).__name__},
        )
        response = _error_response(
            operation if isinstance(operation, str) else None,
            job_id if isinstance(job_id, str) else None,
            wrapped,
        )
        if job_dir is not None:
            _write_json(job_dir / "response.json", response)
        return response


def health_response() -> dict[str, Any]:
    return {
        "schema": RESPONSE_SCHEMA,
        "ok": True,
        "operation": "health",
        "provider": provider_info(),
    }


def _emit(payload: dict[str, Any], output: Path | None = None) -> None:
    if output is not None:
        _write_json(output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Versioned JSON bridge for TikZ-native Manim assets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    health_parser = subparsers.add_parser("health")
    health_parser.add_argument("--output", type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--request", type=Path, required=True)
    run_parser.add_argument("--response", type=Path)
    args = parser.parse_args(argv)

    if args.command == "health":
        payload = health_response()
        _emit(payload, args.output)
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
    # Manim and XeLaTeX may emit progress messages while a request is running.
    # Keep stdout as a clean machine-readable response channel; diagnostics go
    # to stderr for the editor's job log.
    with redirect_stdout(sys.stderr):
        payload = execute_request(request)
    _emit(payload, args.response)
    return 0 if payload.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SUPPORTED_OPERATIONS",
    "execute_request",
    "health_response",
    "main",
]
