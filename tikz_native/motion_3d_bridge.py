from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import PackageNotFoundError, version as distribution_version
import json
from math import isfinite
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping
import uuid

from .compatibility import load_subset_spec
from .version import (
    PROTOCOL_VERSION,
    __version__,
    provider_revision,
)


MOTION_3D_BRIDGE_REQUEST_SCHEMA = "tikz-native-motion-3d-bridge.request/v1"
MOTION_3D_BRIDGE_RESPONSE_SCHEMA = "tikz-native-motion-3d-bridge.response/v1"
MOTION_3D_BRIDGE_OPERATION = "render_motion_3d_preview"
MOTION_3D_BRIDGE_PROFILE = "preview_854x480_15fps"
MOTION_3D_BRIDGE_PROJECTION = "parallel"
MOTION_3D_ASSET_SCHEMA = "tikz-native-motion-3d-asset/v1"

ERROR_INPUT = "INPUT_ERROR"
ERROR_HASH_MISMATCH = "HASH_MISMATCH"
ERROR_PROVIDER = "PROVIDER_MISMATCH"
ERROR_RENDER = "RENDER_FAILED"


class TikzNativeProviderError(RuntimeError):
    """Lightweight bridge error that does not import the Manim provider."""

    def __init__(
        self,
        code: str,
        phase: str,
        message: str,
        *,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "phase": self.phase,
            "message": str(self),
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provider_info() -> dict[str, Any]:
    """Return health metadata without importing the Manim render runtime."""

    subset = load_subset_spec()
    try:
        manim_version = distribution_version("manim")
    except PackageNotFoundError:
        manim_version = "unavailable"
    return {
        "name": "tikz-native-manim",
        "version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "request_schema": MOTION_3D_BRIDGE_REQUEST_SCHEMA,
        "response_schema": MOTION_3D_BRIDGE_RESPONSE_SCHEMA,
        "operation": MOTION_3D_BRIDGE_OPERATION,
        "asset_schema": MOTION_3D_ASSET_SCHEMA,
        "revision": provider_revision(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "manim_version": manim_version,
        "subset_versions": [subset["subset_version"]],
        "capabilities": {
            "compile_2d": True,
            "compile_3d_fixed_view": True,
            "semantic_object_ids": True,
            "semantic_animation_layers": True,
            "render_static": True,
            "dynamic_camera_in_fixed_view": False,
            "render_motion_3d_preview": True,
            "dynamic_camera_3d_parallel": True,
        },
    }


def _render_dependencies():
    """Import Manim and the 3D runtime only for an explicit render request."""

    from manim import tempconfig

    from .motion_3d import Motion3DConfigError, Motion3DSpec
    from .motion_3d_render import (
        MOTION_3D_PREVIEW_PROFILE,
        Motion3DRenderError,
    )

    return (
        tempconfig,
        Motion3DConfigError,
        Motion3DSpec,
        MOTION_3D_PREVIEW_PROFILE,
        Motion3DRenderError,
    )


def compile_motion_3d_asset(*args, **kwargs):
    from .motion_3d_render import compile_motion_3d_asset as implementation

    return implementation(*args, **kwargs)


def build_motion_3d_trace(*args, **kwargs):
    from .motion_3d_render import build_motion_3d_trace as implementation

    return implementation(*args, **kwargs)


def render_motion_3d_preview(*args, **kwargs):
    from .motion_3d_render import render_motion_3d_preview as implementation

    return implementation(*args, **kwargs)


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
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{field or key} must be a non-empty string",
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
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"{field} must be an integer >= 1",
        )
    return value


def _validate_request(request: Mapping[str, Any]) -> None:
    _reject_unknown(
        request,
        {
            "schema",
            "operation",
            "job_id",
            "provider_revision",
            "input",
            "conversion",
            "render",
            "output_dir",
        },
        "request",
    )
    if request.get("schema") != MOTION_3D_BRIDGE_REQUEST_SCHEMA:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"request schema must be {MOTION_3D_BRIDGE_REQUEST_SCHEMA!r}",
        )
    if request.get("operation") != MOTION_3D_BRIDGE_OPERATION:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"operation must be {MOTION_3D_BRIDGE_OPERATION!r}",
        )
    _string(request, "job_id", field="job_id")
    expected_revision = _string(
        request,
        "provider_revision",
        field="provider_revision",
    )
    actual_revision = str(provider_info()["revision"])
    if expected_revision != actual_revision:
        raise TikzNativeProviderError(
            ERROR_PROVIDER,
            "validate_provider",
            "provider revision does not match the frozen preview request",
            details={"expected": expected_revision, "actual": actual_revision},
        )
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
    _positive_integer(input_payload, "picture_index", field="input.picture_index")
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
            "conversion.strict_native must be true for 3D motion bridge v1",
        )
    if conversion.get("view_mode") != "world_3d":
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "conversion.view_mode must be 'world_3d'",
        )

    render = _object(request, "render")
    _reject_unknown(render, {"profile", "projection"}, "render")
    if render.get("profile") != MOTION_3D_BRIDGE_PROFILE:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            f"render.profile must be {MOTION_3D_BRIDGE_PROFILE!r}",
        )
    if render.get("projection") != MOTION_3D_BRIDGE_PROJECTION:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "validate_request",
            "render.projection must be 'parallel'",
        )


def _response_operation(value: object) -> str:
    return MOTION_3D_BRIDGE_OPERATION if value == MOTION_3D_BRIDGE_OPERATION else "unknown"


def _error_response(
    operation: str,
    job_id: str | None,
    error: TikzNativeProviderError,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "schema": MOTION_3D_BRIDGE_RESPONSE_SCHEMA,
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
        "schema": MOTION_3D_BRIDGE_RESPONSE_SCHEMA,
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


def _target_token(target: Path) -> tuple[int, int, int] | None:
    if target.is_symlink():
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "prepare_job",
            f"output_dir must not be a symlink: {target}",
        )
    if not target.exists():
        return None
    if not target.is_dir():
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "prepare_job",
            f"output_dir must be a directory: {target}",
        )
    stat = target.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_mtime_ns))


def _publish(staging: Path, target: Path, expected_token: tuple[int, int, int] | None) -> None:
    if _target_token(target) != expected_token:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "publish",
            "output_dir changed while the 3D preview was rendering",
        )

    backup: Path | None = None
    if target.exists():
        backup = target.parent / f".{target.name}.motion-3d-backup-{uuid.uuid4().hex}"
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None and backup.exists():
        # Publication already succeeded.  A best-effort cleanup must not turn
        # that success into a false failure response.
        shutil.rmtree(backup, ignore_errors=True)


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


def execute_motion_3d_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = _response_operation(
        request.get("operation") if isinstance(request, dict) else None
    )
    job_id_value = request.get("job_id") if isinstance(request, dict) else None
    job_id = (
        job_id_value
        if isinstance(job_id_value, str) and job_id_value.strip()
        else None
    )
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
        target.parent.mkdir(parents=True, exist_ok=True)
        target_token = _target_token(target)
        staging = target.parent / f".{target.name}.motion-3d-stage-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)

        source_snapshot = staging / "input" / "source.tex"
        motion_snapshot = staging / "input" / "motion.json"
        source_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, source_snapshot)
        shutil.copy2(motion_path, motion_snapshot)
        _write_json_atomic(staging / "input" / "request.json", request)
        _verify_hash(source_snapshot, source_hash, asset="source snapshot")
        _verify_hash(motion_snapshot, motion_hash, asset="motion snapshot")

        (
            tempconfig,
            Motion3DConfigError,
            Motion3DSpec,
            preview_profile,
            Motion3DRenderError,
        ) = _render_dependencies()

        try:
            motion = Motion3DSpec.load(motion_snapshot)
        except Motion3DConfigError as error:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_motion",
                f"3D motion config is invalid: {error}",
                details={"exception_type": type(error).__name__},
            ) from error

        picture_index = int(input_payload["picture_index"])
        compile_media = staging / ".compile-media"
        with tempconfig(
            {
                "media_dir": str(compile_media),
                "disable_caching": True,
            }
        ):
            compiled = compile_motion_3d_asset(
                source_snapshot,
                source_sha256=source_hash,
                entry_macro=input_payload.get("entry_macro"),
                picture_index=picture_index,
                scene_unit_per_cm=float(conversion["scene_unit_per_cm"]),
                strict_native=True,
            )
        shutil.rmtree(compile_media, ignore_errors=True)
        try:
            motion.validate_picture(compiled.picture)
        except Motion3DConfigError as error:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_motion",
                f"3D motion config is incompatible with the selected TikZ picture: {error}",
                details={"exception_type": type(error).__name__},
            ) from error

        try:
            trace = build_motion_3d_trace(
                compiled,
                motion,
                provider_revision=str(request["provider_revision"]),
                source_sha256=source_hash,
                motion_sha256=motion_hash,
                profile=preview_profile,
            )
        except Motion3DConfigError as error:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "validate_motion",
                f"3D motion config cannot produce a complete preview trace: {error}",
                details={"exception_type": type(error).__name__},
            ) from error

        compile_dir = staging / "compile"
        compiled.document.write_json(compile_dir / "document-manifest.json")
        _write_json_atomic(compile_dir / "compatibility.json", compiled.compatibility)

        try:
            rendered = render_motion_3d_preview(
                compiled,
                motion,
                staging,
                profile=preview_profile,
                trace=trace,
            )
        except Motion3DRenderError as error:
            raise TikzNativeProviderError(
                ERROR_RENDER,
                "render_motion_3d",
                str(error),
                details={"exception_type": type(error).__name__},
            ) from error

        relative_paths = {
            "source": "input/source.tex",
            "motion": "input/motion.json",
            "request": "input/request.json",
            "document_manifest": "compile/document-manifest.json",
            "compatibility": "compile/compatibility.json",
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
            "profile": MOTION_3D_BRIDGE_PROFILE,
            "projection": MOTION_3D_BRIDGE_PROJECTION,
        }
        asset["files"] = dict(relative_paths)
        _write_json_atomic(compile_dir / "asset.json", asset)

        file_records = {
            name: _file_record(staging, relative)
            for name, relative in relative_paths.items()
        }
        manifest = {
            "schema": "tikz-native-motion-3d-package/v1",
            "job_id": job_id,
            "provider": provider_info(),
            "source_sha256": source_hash,
            "motion_sha256": motion_hash,
            "picture_index": picture_index,
            "conversion": dict(conversion),
            "render": {**dict(render), "media": rendered["media"]},
            "files": file_records,
        }
        _write_json_atomic(staging / "manifest.json", manifest)
        manifest_record = _file_record(staging, "manifest.json")
        response_files = {**file_records, "manifest": manifest_record}
        response = {
            "schema": MOTION_3D_BRIDGE_RESPONSE_SCHEMA,
            "ok": True,
            "operation": MOTION_3D_BRIDGE_OPERATION,
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
        _file_record(staging, "response.json")

        for record in response_files.values():
            verified = _file_record(staging, str(record["path"]))
            if verified != record:
                raise TikzNativeProviderError(
                    ERROR_RENDER,
                    "package",
                    f"package file changed before publication: {record['path']}",
                )
        _publish(staging, target, target_token)
        staging = None
        return response
    except Exception as error:
        if isinstance(error, TikzNativeProviderError) or (
            callable(getattr(error, "to_dict", None))
            and isinstance(getattr(error, "code", None), str)
            and isinstance(getattr(error, "phase", None), str)
        ):
            return _error_response(operation, job_id, error)
        wrapped = TikzNativeProviderError(
            ERROR_RENDER,
            "motion_3d_bridge",
            f"3D motion bridge request failed: {error}",
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
        description="Versioned JSON bridge for TikZ-native 3D motion previews."
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
        payload = execute_motion_3d_request(request)
    _emit(payload, args.response)
    return 0 if payload.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MOTION_3D_BRIDGE_OPERATION",
    "MOTION_3D_BRIDGE_PROFILE",
    "MOTION_3D_BRIDGE_PROJECTION",
    "MOTION_3D_BRIDGE_REQUEST_SCHEMA",
    "MOTION_3D_BRIDGE_RESPONSE_SCHEMA",
    "MOTION_3D_ASSET_SCHEMA",
    "execute_motion_3d_request",
    "health_response",
    "main",
]
