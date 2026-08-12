from __future__ import annotations

import argparse
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import sys
import uuid
from contextlib import redirect_stdout
from typing import Any, Mapping

from .motion_render import (
    LAYOUT_ELLIPSE_CHORD_ANALYSIS,
    MotionPreviewProfile,
    MotionRenderError,
    build_motion_trace,
    render_motion_preview,
)
from .motion_runtime import MotionConfigError, MotionSpec
from .provider import (
    ERROR_HASH_MISMATCH,
    ERROR_INPUT,
    ERROR_PROVIDER,
    ERROR_RENDER,
    TikzNativeProviderError,
    compile_asset,
    sha256_file,
)
from .provider_metadata import provider_info as _base_provider_info
from .version import COMPONENT_MOTION_PREVIEW_2D


MOTION_BRIDGE_REQUEST_SCHEMA = "tikz-native-motion-bridge.request/v1"
MOTION_BRIDGE_RESPONSE_SCHEMA = "tikz-native-motion-bridge.response/v1"
MOTION_BRIDGE_OPERATION = "render_motion_preview"


def provider_info() -> dict[str, Any]:
    """Return the compatibility identity for the 2D motion renderer."""

    return _base_provider_info(revision_component=COMPONENT_MOTION_PREVIEW_2D)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    field: str,
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


def _string(payload: Mapping[str, Any], key: str, *, field: str | None = None) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        name = field or key
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{name} must be a non-empty string",
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


def _positive_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    field: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    value = payload.get(key)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        upper = "" if maximum is None else f" and <= {maximum}"
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{field} must be an integer >= {minimum}{upper}",
        )
    return value


def _validate_request(request: Mapping[str, Any]) -> None:
    _reject_unknown(
        request,
        {
            "schema",
            "operation",
            "job_id",
            "input",
            "conversion",
            "render",
            "output_dir",
        },
        "request",
    )
    if request.get("schema") != MOTION_BRIDGE_REQUEST_SCHEMA:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"request schema must be {MOTION_BRIDGE_REQUEST_SCHEMA!r}",
            details={"actual": request.get("schema")},
        )
    if request.get("operation") != MOTION_BRIDGE_OPERATION:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"operation must be {MOTION_BRIDGE_OPERATION!r}",
        )
    _string(request, "job_id", field="job_id")
    _string(request, "output_dir", field="output_dir")

    input_payload = _object(request, "input")
    _reject_unknown(
        input_payload,
        {
            "source_path",
            "source_sha256",
            "motion_path",
            "motion_sha256",
            "entry_macro",
            "picture_index",
        },
        "input",
    )
    _string(input_payload, "source_path", field="input.source_path")
    _sha256(input_payload, "source_sha256", field="input.source_sha256")
    _string(input_payload, "motion_path", field="input.motion_path")
    _sha256(input_payload, "motion_sha256", field="input.motion_sha256")
    _positive_integer(
        input_payload,
        "picture_index",
        field="input.picture_index",
    )
    entry_macro = input_payload.get("entry_macro")
    if entry_macro is not None and not isinstance(entry_macro, str):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "input.entry_macro must be a string or null",
        )

    conversion = _object(request, "conversion")
    _reject_unknown(
        conversion,
        {"subset_version", "scene_unit_per_cm", "strict_native", "view_mode"},
        "conversion",
    )
    required_conversion = {
        "subset_version",
        "scene_unit_per_cm",
        "strict_native",
        "view_mode",
    }
    missing_conversion = sorted(required_conversion - set(conversion))
    if missing_conversion:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "conversion is missing required fields: " + ", ".join(missing_conversion),
        )
    subset_version = conversion.get("subset_version")
    supported_subsets = provider_info()["subset_versions"]
    if subset_version not in supported_subsets:
        raise TikzNativeProviderError(
            ERROR_PROVIDER,
            "validate_request",
            f"unsupported TikZ-native subset version: {subset_version!r}",
            details={"supported_subset_versions": supported_subsets},
        )
    scene_unit = conversion.get("scene_unit_per_cm")
    if (
        isinstance(scene_unit, bool)
        or not isinstance(scene_unit, (int, float))
        or not isfinite(float(scene_unit))
        or float(scene_unit) <= 0
    ):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "conversion.scene_unit_per_cm must be a finite positive number",
        )
    if conversion.get("strict_native") is not True:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "conversion.strict_native must be true for motion bridge v1",
        )
    if conversion.get("view_mode") != "tikz_fixed":
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "conversion.view_mode must be 'tikz_fixed'",
        )

    render = _object(request, "render")
    _reject_unknown(
        render,
        {
            "profile",
            "layout",
            "pixel_width",
            "pixel_height",
            "frame_rate",
            "background",
        },
        "render",
    )
    required_render = {
        "profile",
        "layout",
        "pixel_width",
        "pixel_height",
        "frame_rate",
        "background",
    }
    missing_render = sorted(required_render - set(render))
    if missing_render:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "render is missing required fields: " + ", ".join(missing_render),
        )
    if render.get("profile") != "preview":
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "render.profile must be 'preview'",
        )
    if render.get("layout") != LAYOUT_ELLIPSE_CHORD_ANALYSIS:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"render.layout must be {LAYOUT_ELLIPSE_CHORD_ANALYSIS!r}",
        )
    _positive_integer(
        render,
        "pixel_width",
        field="render.pixel_width",
        minimum=160,
        maximum=1920,
    )
    _positive_integer(
        render,
        "pixel_height",
        field="render.pixel_height",
        minimum=90,
        maximum=1080,
    )
    _positive_integer(
        render,
        "frame_rate",
        field="render.frame_rate",
        minimum=1,
        maximum=60,
    )
    background = _string(render, "background", field="render.background")
    if (
        len(background) != 7
        or not background.startswith("#")
        or any(character not in "0123456789abcdefABCDEF" for character in background[1:])
    ):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "render.background must be a six-digit hexadecimal color",
        )


def _response_operation(value: object) -> str:
    return MOTION_BRIDGE_OPERATION if value == MOTION_BRIDGE_OPERATION else "unknown"


def _error_response(
    operation: str,
    job_id: str | None,
    error: TikzNativeProviderError,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "schema": MOTION_BRIDGE_RESPONSE_SCHEMA,
        "ok": False,
        "operation": operation,
        "provider": provider_info(),
        "error": error.to_dict(),
    }
    if job_id:
        response["job_id"] = job_id
    return response


def health_response() -> dict[str, Any]:
    return {
        "schema": MOTION_BRIDGE_RESPONSE_SCHEMA,
        "ok": True,
        "operation": "health",
        "provider": provider_info(),
    }


def _verify_hash(path: Path, expected: str, *, asset: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise TikzNativeProviderError(
            ERROR_HASH_MISMATCH,
            "verify_input",
            f"{asset} SHA-256 does not match the requested snapshot",
            details={"asset": asset, "expected": expected, "actual": actual},
        )


def _prepare_output_target(target: Path) -> bool:
    """Validate the publication target without changing it.

    Returns whether the caller supplied an existing empty directory.  A
    symlink is never accepted as an isolated job directory.
    """

    if target.is_symlink():
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "prepare_job",
            f"output_dir must not be a symlink: {target}",
        )
    if not target.exists():
        return False
    if not target.is_dir():
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "prepare_job",
            f"output_dir must be a directory: {target}",
        )
    if any(target.iterdir()):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "prepare_job",
            f"output_dir must be empty for an isolated job: {target}",
        )
    return True


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "package",
            f"package file path must be relative and contained: {relative}",
        )
    path = root / candidate
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise TikzNativeProviderError(
            ERROR_RENDER,
            "package",
            f"required package file is missing or empty: {relative}",
        )
    return {
        "path": candidate.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _publish(staging: Path, target: Path, *, target_was_empty: bool) -> None:
    if target_was_empty:
        if not target.is_dir() or target.is_symlink() or any(target.iterdir()):
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "publish",
                "output_dir changed while the motion preview was rendering",
            )
        target.rmdir()
    elif target.exists():
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "publish",
            "output_dir appeared while the motion preview was rendering",
        )
    try:
        os.replace(staging, target)
    except Exception:
        if target_was_empty and not target.exists():
            target.mkdir(parents=False, exist_ok=True)
        raise


def execute_motion_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = _response_operation(request.get("operation") if isinstance(request, dict) else None)
    job_id_value = request.get("job_id") if isinstance(request, dict) else None
    job_id = job_id_value if isinstance(job_id_value, str) and job_id_value.strip() else None
    staging: Path | None = None
    try:
        if not isinstance(request, dict):
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_request",
                "request root must be an object",
            )
        _validate_request(request)
        input_payload = _object(request, "input")
        conversion = _object(request, "conversion")
        render = _object(request, "render")
        assert job_id is not None

        source = Path(_string(input_payload, "source_path")).expanduser().resolve()
        motion_path = Path(_string(input_payload, "motion_path")).expanduser().resolve()
        for label, path in (("source", source), ("motion", motion_path)):
            if not path.is_file():
                raise TikzNativeProviderError(
                    ERROR_INPUT,
                    "read_input",
                    f"{label} file does not exist: {path}",
                )
        source_hash = _sha256(
            input_payload,
            "source_sha256",
            field="input.source_sha256",
        )
        motion_hash = _sha256(
            input_payload,
            "motion_sha256",
            field="input.motion_sha256",
        )
        _verify_hash(source, source_hash, asset="source")
        _verify_hash(motion_path, motion_hash, asset="motion")

        target = Path(_string(request, "output_dir")).expanduser().resolve()
        target_was_empty = _prepare_output_target(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.motion-stage-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)

        source_snapshot = staging / "input" / "source.tex"
        motion_snapshot = staging / "input" / "motion.json"
        source_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, source_snapshot)
        shutil.copy2(motion_path, motion_snapshot)
        _write_json_atomic(staging / "input" / "request.json", request)
        _verify_hash(source_snapshot, source_hash, asset="source snapshot")
        _verify_hash(motion_snapshot, motion_hash, asset="motion snapshot")

        picture_index = int(input_payload["picture_index"])
        compiled = compile_asset(
            source_snapshot,
            source_sha256=source_hash,
            entry_macro=input_payload.get("entry_macro"),
            picture_index=picture_index,
            scene_unit_per_cm=float(conversion["scene_unit_per_cm"]),
            strict_native=True,
            view_mode="tikz_fixed",
        )
        try:
            motion = MotionSpec.load(motion_snapshot)
            motion.validate_picture(compiled.picture)
        except MotionConfigError as error:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_motion",
                f"motion config is incompatible with the selected TikZ picture: {error}",
                details={"exception_type": type(error).__name__},
            ) from error

        profile = MotionPreviewProfile(
            pixel_width=int(render["pixel_width"]),
            pixel_height=int(render["pixel_height"]),
            frame_rate=int(render["frame_rate"]),
            background=str(render["background"]),
            layout=str(render["layout"]),
        )
        try:
            trace = build_motion_trace(
                compiled,
                motion,
                provider_revision=provider_info()["revision"],
                source_sha256=source_hash,
                motion_sha256=motion_hash,
                profile=profile,
            )
        except MotionConfigError as error:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_motion",
                f"motion config cannot produce the requested preview layout: {error}",
                details={"exception_type": type(error).__name__},
            ) from error

        compile_dir = staging / "compile"
        compiled.document.write_json(compile_dir / "document-manifest.json")
        _write_json_atomic(compile_dir / "compatibility.json", compiled.compatibility)
        _write_json_atomic(compile_dir / "animation-plan.json", compiled.animation_plan)

        try:
            rendered = render_motion_preview(
                compiled,
                motion,
                staging,
                profile=profile,
                trace=trace,
            )
        except MotionRenderError as error:
            raise TikzNativeProviderError(
                ERROR_RENDER,
                "render_motion",
                str(error),
                details={"exception_type": type(error).__name__},
            ) from error

        relative_paths = {
            "source": "input/source.tex",
            "motion": "input/motion.json",
            "request": "input/request.json",
            "document_manifest": "compile/document-manifest.json",
            "compatibility": "compile/compatibility.json",
            "animation_plan": "compile/animation-plan.json",
            "asset": "compile/asset.json",
            "video": rendered["video"],
            "first_frame": rendered["first_frame"],
            "last_frame": rendered["last_frame"],
            "trace": rendered["trace"],
        }
        asset = dict(compiled.asset)
        asset["motion"] = {
            "schema": motion.schema,
            "source_sha256": motion_hash,
            "layout": profile.layout,
        }
        asset["files"] = dict(relative_paths)
        _write_json_atomic(compile_dir / "asset.json", asset)

        file_records = {
            name: _file_record(staging, relative)
            for name, relative in relative_paths.items()
        }
        manifest = {
            "schema": "tikz-native-motion-package/v1",
            "job_id": job_id,
            "provider": provider_info(),
            "source_sha256": source_hash,
            "motion_sha256": motion_hash,
            "picture_index": picture_index,
            "conversion": dict(conversion),
            "render": {**dict(render), "media": rendered["media"]},
            "files": file_records,
        }
        manifest_path = staging / "manifest.json"
        _write_json_atomic(manifest_path, manifest)
        manifest_record = _file_record(staging, "manifest.json")
        response_files = {**file_records, "manifest": manifest_record}
        response = {
            "schema": MOTION_BRIDGE_RESPONSE_SCHEMA,
            "ok": True,
            "operation": MOTION_BRIDGE_OPERATION,
            "job_id": job_id,
            "provider": provider_info(),
            "package": {
                "manifest_path": "manifest.json",
                "manifest_sha256": manifest_record["sha256"],
                "media": rendered["media"],
                "files": response_files,
            },
        }
        _write_json_atomic(staging / "response.json", response)

        for record in response_files.values():
            verified = _file_record(staging, str(record["path"]))
            if verified != record:
                raise TikzNativeProviderError(
                    ERROR_RENDER,
                    "package",
                    f"package file changed before publication: {record['path']}",
                )
        if not (staging / "response.json").is_file():
            raise TikzNativeProviderError(
                ERROR_RENDER,
                "package",
                "motion package response is missing",
            )

        _publish(staging, target, target_was_empty=target_was_empty)
        staging = None
        return response
    except TikzNativeProviderError as error:
        return _error_response(operation, job_id, error)
    except Exception as error:
        wrapped = TikzNativeProviderError(
            ERROR_RENDER,
            "motion_bridge",
            f"motion bridge request failed: {error}",
            details={"exception_type": type(error).__name__},
        )
        return _error_response(operation, job_id, wrapped)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _emit(payload: Mapping[str, Any], output: Path | None = None) -> None:
    if output is not None:
        _write_json_atomic(output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Versioned JSON bridge for TikZ-native analytic motion previews."
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

    with redirect_stdout(sys.stderr):
        payload = execute_motion_request(request)
    _emit(payload, args.response)
    return 0 if payload.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MOTION_BRIDGE_OPERATION",
    "MOTION_BRIDGE_REQUEST_SCHEMA",
    "MOTION_BRIDGE_RESPONSE_SCHEMA",
    "execute_motion_request",
    "health_response",
    "main",
]
