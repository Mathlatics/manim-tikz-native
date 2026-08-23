"""Source-authoritative TikZ project builds.

The project manifest stores only authored inputs and render intent.  Every
artifact produced by this module is derived, deterministic, and disposable.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

SOURCE_PROJECT_SCHEMA_VERSION = "tikz-native-source-project/v1"
BUILD_MANIFEST_SCHEMA_VERSION = "tikz-native-build-manifest/v1"
SHAPE_ASSET_SCHEMA_VERSION = "tikz-native-shape-asset/v1"
MOTION_ASSET_SCHEMA_VERSION = "tikz-native-motion-asset/v1"
COMPOSITING_SCHEMA_VERSION = "tikz-native-unified-compositing/v1"

PROVIDER_COMPONENT = "source_project_build"
PROVIDER_CAPABILITY = "source_authoritative_project_build_v1"
COMMAND_RESULT_FORMAT_VERSION = "tikz-native-project-command-result/v1"

OWNED_OUTPUT_SCHEMA_VERSION = "tikz-native-owned-output/v1"
OWNED_OUTPUT_MARKER = ".tikz-native-owned.json"
BUILD_MANIFEST_NAME = "build-manifest.json"
_KNOWN_OUTPUT_NAMES = frozenset(
    {
        OWNED_OUTPUT_MARKER,
        BUILD_MANIFEST_NAME,
        "shape-asset.json",
        "motion-asset.json",
        "unified-compositing.json",
        "generated_scene.py",
    }
)

_FORBIDDEN_MANIFEST_KEYS = {
    "compositingmode",
    "implementationmode",
}


class SourceProjectError(ValueError):
    """Raised when a source project is invalid or unsafe."""


class SourceProjectBuildError(RuntimeError):
    """Raised when a derived artifact cannot be generated safely."""


@dataclass(frozen=True)
class PainterZBand:
    minimum: float
    maximum: float

    def as_list(self) -> list[float]:
        return [self.minimum, self.maximum]


@dataclass(frozen=True)
class SourceProject:
    manifest_path: Path
    root: Path
    tikz_source: Path
    motion_json: Path | None
    hooks_source: Path | None
    bridge_request_template: Path | None
    output_directory: Path
    paint_policy: str
    projection: Any
    painter_z_band_override: PainterZBand | None
    picture_index: int
    entry_macro: str | None
    selection: Mapping[str, Any]


@dataclass(frozen=True)
class NodePlan:
    name: str
    output_name: str
    key: str
    component_revisions: Mapping[str, str | int]
    build_payload: Callable[["_BuildContext"], bytes]


@dataclass(frozen=True)
class InputSnapshot:
    project: SourceProject
    manifest_payload: bytes
    payloads: Mapping[Path, bytes]

    def payload(self, path: Path | None) -> bytes | None:
        if path is None:
            return None
        return self.payloads[path]

    def text(self, path: Path, *, label: str) -> str:
        payload = self.payloads[path]
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceProjectError(f"{label} must be UTF-8: {path}") from exc


@dataclass
class _BuildContext:
    project: SourceProject
    snapshot: InputSnapshot
    revisions: Mapping[str, str | int]
    painter_z_band: PainterZBand
    motion_value: Any
    shape_asset_builder: "ShapeAssetBuilder | None"
    bridge_generator: "BridgeGenerator | None"
    digests: MutableMapping[str, str]
    _snapshot_directory: tempfile.TemporaryDirectory[str] | None = None
    _snapshot_source_path: Path | None = None

    def source_path(self) -> Path:
        if self._snapshot_source_path is None:
            self._snapshot_directory = tempfile.TemporaryDirectory(
                prefix=".tikz-native-input-", dir=self.project.root
            )
            path = Path(self._snapshot_directory.name) / self.project.tikz_source.name
            path.write_bytes(self.snapshot.payloads[self.project.tikz_source])
            self._snapshot_source_path = path
        return self._snapshot_source_path

    def close(self) -> None:
        if self._snapshot_directory is not None:
            self._snapshot_directory.cleanup()
            self._snapshot_directory = None
            self._snapshot_source_path = None


@dataclass(frozen=True)
class NodeState:
    name: str
    action: str
    key: str
    output: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "action": self.action,
            "key": self.key,
            "output": self.output,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class BuildResult:
    project: SourceProject
    mode: str
    nodes: tuple[NodeState, ...]
    manifest_path: Path
    painter_z_band: PainterZBand

    @property
    def built(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes if node.action == "built")

    @property
    def reused(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes if node.action == "reused")

    def as_dict(self) -> dict[str, Any]:
        return {
            "resultFormat": COMMAND_RESULT_FORMAT_VERSION,
            "mode": self.mode,
            "project": str(self.project.manifest_path),
            "outputDirectory": str(self.project.output_directory),
            "manifest": str(self.manifest_path),
            "painterZBand": self.painter_z_band.as_list(),
            "built": list(self.built),
            "reused": list(self.reused),
            "nodes": [node.as_dict() for node in self.nodes],
        }


@dataclass(frozen=True)
class ProjectStatus:
    project: SourceProject
    fresh: bool
    nodes: tuple[NodeState, ...]
    manifest_path: Path
    manifest_action: str
    painter_z_band: PainterZBand

    def as_dict(self) -> dict[str, Any]:
        return {
            "resultFormat": COMMAND_RESULT_FORMAT_VERSION,
            "mode": "status",
            "project": str(self.project.manifest_path),
            "outputDirectory": str(self.project.output_directory),
            "manifest": str(self.manifest_path),
            "manifestAction": self.manifest_action,
            "fresh": self.fresh,
            "painterZBand": self.painter_z_band.as_list(),
            "nodes": [node.as_dict() for node in self.nodes],
        }


ShapeAssetBuilder = Callable[[SourceProject, str], Any]
BridgeGenerator = Callable[[Mapping[str, Any]], str]


def provider_component_descriptor() -> dict[str, Any]:
    """Return the Provider component record owned by this module."""

    version = importlib.import_module("tikz_native.version")
    revision = version.provider_component_revision(
        version.COMPONENT_SOURCE_PROJECT_BUILD
    )

    return {
        "name": PROVIDER_COMPONENT,
        "revision": revision,
        "capabilities": [PROVIDER_CAPABILITY],
        "owns": [
            "tikz_native.source_project",
            "tikz_native/schemas/tikz-native-source-project-v1.schema.json",
            "tikz_native/schemas/tikz-native-build-manifest-v1.schema.json",
        ],
    }


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_key(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _read_utf8(path: Path, *, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SourceProjectError(f"{label} must be UTF-8: {path}") from exc
    except OSError as exc:
        raise SourceProjectError(f"cannot read {label}: {path}: {exc}") from exc


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_utf8(path, label=label),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise SourceProjectError(
            f"invalid JSON in {label} {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise SourceProjectError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceProjectError(f"{label} must contain a JSON object: {path}")
    return value


def _normalise_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", key).lower()


def _reject_persisted_implementation_modes(value: Any, *, location: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and _normalise_key(key) in _FORBIDDEN_MANIFEST_KEYS:
                raise SourceProjectError(
                    f"{location} must not persist implementation mode {key!r}; "
                    "generated open-face code always uses the current unified compositor"
                )
            _reject_persisted_implementation_modes(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_persisted_implementation_modes(child, location=f"{location}[{index}]")


def _one_of(mapping: Mapping[str, Any], names: Sequence[str], *, required: bool = False) -> Any:
    present = [(name, mapping[name]) for name in names if name in mapping]
    if len(present) > 1:
        joined = ", ".join(name for name, _ in present)
        raise SourceProjectError(f"use only one of these aliases: {joined}")
    if present:
        return present[0][1]
    if required:
        raise SourceProjectError(f"missing required property {names[0]!r}")
    return None


def _ensure_keys(mapping: Mapping[str, Any], allowed: Iterable[str], *, location: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise SourceProjectError(
            f"unknown {location} propert{'y' if len(unknown) == 1 else 'ies'}: "
            + ", ".join(unknown)
        )


def _resolve_project_path(
    root: Path,
    raw: Any,
    *,
    label: str,
    must_exist: bool,
    allow_directory: bool = False,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SourceProjectError(f"{label} must be a non-empty relative path")
    supplied = Path(raw)
    if supplied.is_absolute():
        raise SourceProjectError(f"{label} must be relative to the project manifest")
    if any(part == ".." for part in supplied.parts):
        raise SourceProjectError(f"{label} must not traverse outside the project directory")
    candidate = root.joinpath(supplied)
    current = root
    for part in supplied.parts:
        current = current / part
        try:
            if current.is_symlink():
                raise SourceProjectError(
                    f"{label} symlink escapes the stable project path: {raw!r}"
                )
        except OSError as exc:
            raise SourceProjectError(f"cannot inspect {label} {raw!r}: {exc}") from exc
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise SourceProjectError(f"cannot resolve {label} {raw!r}: {exc}") from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SourceProjectError(f"{label} escapes the project directory: {raw!r}") from exc
    if must_exist:
        if allow_directory:
            if not resolved.is_dir():
                raise SourceProjectError(f"{label} is not a directory: {raw!r}")
        elif not resolved.is_file():
            raise SourceProjectError(f"{label} is not a regular file: {raw!r}")
    return resolved


def _resolve_output_path(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SourceProjectError("derivedOutput must be a non-empty relative path")
    supplied = Path(raw)
    if supplied.is_absolute():
        raise SourceProjectError("derivedOutput must be relative to the project manifest")
    if any(part == ".." for part in supplied.parts):
        raise SourceProjectError("derivedOutput must not traverse outside the project directory")
    candidate = root.joinpath(supplied)
    current = root
    for part in supplied.parts:
        current = current / part
        if current.is_symlink():
            raise SourceProjectError(
                f"derivedOutput must not contain symlinks: {raw!r}"
            )
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SourceProjectError(f"derivedOutput escapes the project directory: {raw!r}") from exc
    return candidate


def _parse_painter_z_band(value: Any) -> PainterZBand | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        _ensure_keys(value, {"min", "max"}, location="painterZBand")
        minimum = _one_of(value, ("min",), required=True)
        maximum = _one_of(value, ("max",), required=True)
    elif isinstance(value, list) and len(value) == 2:
        minimum, maximum = value
    else:
        raise SourceProjectError(
            "painterZBand must be [minimum, maximum] or an object with min/max"
        )
    if isinstance(minimum, bool) or isinstance(maximum, bool):
        raise SourceProjectError("painterZBand values must be finite numbers")
    try:
        minimum_float = float(minimum)
        maximum_float = float(maximum)
    except (TypeError, ValueError) as exc:
        raise SourceProjectError("painterZBand values must be finite numbers") from exc
    if not math.isfinite(minimum_float) or not math.isfinite(maximum_float):
        raise SourceProjectError("painterZBand values must be finite numbers")
    if minimum_float >= maximum_float:
        raise SourceProjectError("painterZBand minimum must be less than maximum")
    return PainterZBand(minimum_float, maximum_float)


def _parse_selection(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SourceProjectError("selection must be an object")
    _ensure_keys(
        value,
        {
            "candidate_id",
            "range",
            "include_object_ids",
            "exclude_object_ids",
        },
        location="selection",
    )
    result: dict[str, Any] = {}
    candidate_id = value.get("candidate_id")
    if candidate_id is not None:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise SourceProjectError("selection.candidate_id must be a non-empty string")
        result["candidate_id"] = candidate_id
    selected_range = value.get("range")
    if selected_range is not None:
        if not isinstance(selected_range, list) or len(selected_range) != 2:
            raise SourceProjectError("selection.range must contain exactly two numbers")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in selected_range
        ):
            raise SourceProjectError("selection.range must contain exactly two finite numbers")
        result["range"] = list(selected_range)
    for key in ("include_object_ids", "exclude_object_ids"):
        object_ids = value.get(key)
        if object_ids is None:
            continue
        if (
            not isinstance(object_ids, list)
            or any(not isinstance(item, str) or not item for item in object_ids)
            or len(set(object_ids)) != len(object_ids)
        ):
            raise SourceProjectError(
                f"selection.{key} must contain unique non-empty strings"
            )
        result[key] = list(object_ids)
    return result


def load_source_project(manifest_path: str | os.PathLike[str]) -> SourceProject:
    """Load and validate a ``tikz-native-source-project/v1`` manifest."""

    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    if not manifest.is_file():
        raise SourceProjectError(f"project manifest is not a regular file: {manifest}")
    root = manifest.parent.resolve(strict=True)
    raw = _load_json_object(manifest, label="source project manifest")
    _reject_persisted_implementation_modes(raw)
    _ensure_keys(
        raw,
        {
            "$schema",
            "schemaVersion",
            "tikzSource",
            "motionJson",
            "hooksSource",
            "bridgeRequestTemplate",
            "derivedOutput",
            "renderIntent",
            "pictureIndex",
            "entryMacro",
            "selection",
        },
        location="manifest",
    )
    version = raw.get("schemaVersion")
    if version != SOURCE_PROJECT_SCHEMA_VERSION:
        raise SourceProjectError(
            f"schemaVersion must be {SOURCE_PROJECT_SCHEMA_VERSION!r}, got {version!r}"
        )

    tikz_raw = _one_of(raw, ("tikzSource",), required=True)
    motion_raw = _one_of(raw, ("motionJson",))
    hooks_raw = _one_of(raw, ("hooksSource",))
    bridge_raw = _one_of(raw, ("bridgeRequestTemplate",))
    output_raw = _one_of(raw, ("derivedOutput",))
    if output_raw is None:
        output_raw = ".tikz-native/derived"

    render_intent = raw.get("renderIntent", {})
    if not isinstance(render_intent, Mapping):
        raise SourceProjectError("renderIntent must be an object")
    _ensure_keys(
        render_intent,
        {"paintPolicy", "projection", "painterZBand"},
        location="renderIntent",
    )
    paint_policy = render_intent.get("paintPolicy", "diagrammatic")
    if not isinstance(paint_policy, str) or paint_policy not in {
        "diagrammatic",
        "physical",
    }:
        raise SourceProjectError(
            "renderIntent.paintPolicy must be 'diagrammatic' or 'physical'"
        )
    projection = render_intent.get("projection", {"kind": "identity"})
    try:
        _canonical_json(projection)
    except (TypeError, ValueError) as exc:
        raise SourceProjectError("renderIntent.projection must be JSON-serializable") from exc

    tikz_source = _resolve_project_path(
        root, tikz_raw, label="tikzSource", must_exist=True
    )
    motion_json = (
        _resolve_project_path(root, motion_raw, label="motionJson", must_exist=True)
        if motion_raw is not None
        else None
    )
    hooks_source = (
        _resolve_project_path(root, hooks_raw, label="hooksSource", must_exist=True)
        if hooks_raw is not None
        else None
    )
    bridge_template = (
        _resolve_project_path(
            root, bridge_raw, label="bridgeRequestTemplate", must_exist=True
        )
        if bridge_raw is not None
        else None
    )
    output_directory = _resolve_output_path(root, output_raw)
    if output_directory == root:
        raise SourceProjectError("derivedOutput must not be the project root")
    if manifest == output_directory or manifest.is_relative_to(output_directory):
        raise SourceProjectError("derivedOutput must not contain the source manifest")

    authored_paths = tuple(
        path
        for path in (tikz_source, motion_json, hooks_source, bridge_template)
        if path is not None
    )
    for authored in authored_paths:
        if authored == output_directory or authored.is_relative_to(output_directory):
            raise SourceProjectError(
                "derivedOutput must not contain authoritative input "
                f"{authored.relative_to(root).as_posix()!r}"
            )

    picture_index = raw.get("pictureIndex", 1)
    if (
        isinstance(picture_index, bool)
        or not isinstance(picture_index, int)
        or picture_index < 1
    ):
        raise SourceProjectError("pictureIndex must be an integer >= 1")
    entry_macro = raw.get("entryMacro")
    if entry_macro is not None and not isinstance(entry_macro, str):
        raise SourceProjectError("entryMacro must be a string or null")
    selection = _parse_selection(raw.get("selection"))
    if hooks_source is not None and bridge_template is None:
        raise SourceProjectError("hooksSource requires bridgeRequestTemplate")
    if "selection" in raw and bridge_template is None:
        raise SourceProjectError("selection requires bridgeRequestTemplate")

    return SourceProject(
        manifest_path=manifest,
        root=root,
        tikz_source=tikz_source,
        motion_json=motion_json,
        hooks_source=hooks_source,
        bridge_request_template=bridge_template,
        output_directory=output_directory,
        paint_policy=paint_policy.strip(),
        projection=projection,
        painter_z_band_override=_parse_painter_z_band(
            render_intent.get("painterZBand")
        ),
        picture_index=picture_index,
        entry_macro=entry_macro,
        selection=selection,
    )


def _relative(project: SourceProject, path: Path | None) -> str | None:
    if path is None:
        return None
    return path.relative_to(project.root).as_posix()


def derive_painter_z_band(project: SourceProject, tikz_bytes: bytes | None = None) -> PainterZBand:
    """Derive a stable z band from TikZ and projection only."""

    if project.painter_z_band_override is not None:
        return project.painter_z_band_override
    if tikz_bytes is None:
        tikz_bytes = project.tikz_source.read_bytes()
    digest = hashlib.sha256()
    digest.update(tikz_bytes)
    digest.update(b"\0projection\0")
    digest.update(_canonical_json(project.projection))
    offset = int(digest.hexdigest()[:8], 16) % 4096
    minimum = float(10_000 + offset * 2)
    return PainterZBand(minimum, minimum + 1024.0)


def _normalise_compiler_result(value: Any) -> Any:
    asset = getattr(value, "asset", None)
    if isinstance(asset, Mapping):
        return dict(asset)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, str, int, float, bool)) or value is None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return json.loads(
                        stripped, parse_constant=_reject_json_constant
                    )
                except json.JSONDecodeError:
                    return {"generated": stripped}
                except ValueError as exc:
                    raise SourceProjectBuildError(
                        f"compiler returned invalid JSON: {exc}"
                    ) from exc
        return value
    for name in ("to_dict", "as_dict", "model_dump", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return dict(result)
    raise SourceProjectBuildError(
        f"current TikZ compiler returned unsupported value {type(value).__name__}"
    )


def _call_candidate(function: Callable[..., Any], project: SourceProject, source: str) -> Any:
    signature = inspect.signature(function)
    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        name = parameter.name.lower()
        value: Any
        if name in {"source", "source_text", "tikz", "tikz_source", "text"}:
            value = source
        elif name in {"path", "source_path", "tikz_path", "filename"}:
            value = project.tikz_source
        elif name in {"projection", "projection_spec"}:
            value = project.projection
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(f"unsupported required compiler parameter {parameter.name!r}")
        else:
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            kwargs[parameter.name] = value
    return function(*positional, **kwargs)


def _default_shape_asset_builder(project: SourceProject, source: str) -> Any:
    provider = importlib.import_module("tikz_native.provider")
    compile_asset = getattr(provider, "compile_asset")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".tikz-native-compile-", dir=project.root
        ) as temporary:
            snapshot = Path(temporary) / project.tikz_source.name
            snapshot.write_text(source, encoding="utf-8")
            compiled = compile_asset(
                snapshot,
                source_sha256=_sha256_bytes(source.encode("utf-8")),
                entry_macro=project.entry_macro,
                picture_index=project.picture_index,
                strict_native=True,
            )
    except Exception as exc:
        provider_error = getattr(provider, "TikzNativeProviderError", ())
        if not isinstance(exc, provider_error):
            raise
        code = getattr(exc, "code", "PROVIDER_ERROR")
        phase = getattr(exc, "phase", "compile")
        raise SourceProjectBuildError(
            f"TikZ Provider {phase} failed ({code}): {exc}"
        ) from exc
    return _normalise_compiler_result(compiled)


def _provider_revision_defaults() -> dict[str, str | int]:
    version = importlib.import_module("tikz_native.version")
    provider_revisions = dict(version.provider_component_revisions())
    names = {
        "source_project_build": getattr(
            version, "COMPONENT_SOURCE_PROJECT_BUILD", "source_project_build"
        ),
        "asset_compiler": version.COMPONENT_ASSET_COMPILER,
        "generated_open_face_visibility_3d": getattr(
            version,
            "COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D",
            version.COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
        ),
        "open_face_unified_compositing": version.COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING,
        "managed_painter_band": getattr(
            version, "COMPONENT_MANAGED_PAINTER_BAND", "managed_painter_band"
        ),
    }
    result: dict[str, str | int] = {}
    for public_name, component in names.items():
        revision = provider_revisions.get(component)
        if revision is None:
            # Parallel worktrees may import this module before the new component
            # registration lands.  The complete Provider build hash remains a
            # fail-closed fallback rather than the old constant integer.
            revision = version.provider_revision()
        result[public_name] = revision
    return result


_REVISION_ALIASES = {
    "tikz_compiler": "asset_compiler",
    "bridge_codegen": "generated_open_face_visibility_3d",
    "unified_compositor": "open_face_unified_compositing",
}


def _normalise_revisions(
    revisions: Mapping[str, str | int] | None,
) -> dict[str, str | int]:
    result = _provider_revision_defaults()
    if revisions is not None:
        for supplied_name, revision in revisions.items():
            if not isinstance(supplied_name, str) or not supplied_name:
                raise SourceProjectError("component revision names must be non-empty strings")
            name = _REVISION_ALIASES.get(supplied_name, supplied_name)
            if name not in result:
                raise SourceProjectError(f"unknown Provider component revision {supplied_name!r}")
            if isinstance(revision, bool) or not isinstance(revision, (str, int)):
                raise SourceProjectError(
                    f"component revision {supplied_name!r} must be a string or integer"
                )
            if isinstance(revision, int) and revision < 0:
                raise SourceProjectError(
                    f"component revision {supplied_name!r} must be non-negative"
                )
            if isinstance(revision, str) and not revision.strip():
                raise SourceProjectError(
                    f"component revision {supplied_name!r} must not be empty"
                )
            result[name] = revision
    return dict(sorted(result.items()))


def _validate_injected_builder_revisions(
    revisions: Mapping[str, str | int] | None,
    *,
    shape_asset_builder: ShapeAssetBuilder | None,
    bridge_generator: BridgeGenerator | None,
) -> None:
    supplied = set(revisions or {})
    if shape_asset_builder is not None and supplied.isdisjoint(
        {"asset_compiler", "tikz_compiler"}
    ):
        raise SourceProjectError(
            "a custom shape_asset_builder requires an explicit "
            "asset_compiler component revision"
        )
    if bridge_generator is not None and supplied.isdisjoint(
        {"generated_open_face_visibility_3d", "bridge_codegen"}
    ):
        raise SourceProjectError(
            "a custom bridge_generator requires an explicit "
            "generated_open_face_visibility_3d component revision"
        )


def _capture_input_snapshot(project: SourceProject) -> InputSnapshot:
    paths = tuple(
        path
        for path in (
            project.tikz_source,
            project.motion_json,
            project.hooks_source,
            project.bridge_request_template,
        )
        if path is not None
    )
    manifest_payload = project.manifest_path.read_bytes()
    payloads = {path: path.read_bytes() for path in paths}
    reloaded = load_source_project(project.manifest_path)
    if reloaded != project or project.manifest_path.read_bytes() != manifest_payload:
        raise SourceProjectBuildError("source project changed while inputs were snapshotted")
    for path, payload in payloads.items():
        if path.read_bytes() != payload:
            raise SourceProjectBuildError(
                f"authoritative input changed while it was snapshotted: {path}"
            )
    return InputSnapshot(project, manifest_payload, MappingProxyType(payloads))


def _validate_input_snapshot(snapshot: InputSnapshot) -> None:
    project = snapshot.project
    if project.manifest_path.read_bytes() != snapshot.manifest_payload:
        raise SourceProjectBuildError("source project manifest changed during build")
    if load_source_project(project.manifest_path) != project:
        raise SourceProjectBuildError("source project paths or render intent changed during build")
    for path, payload in snapshot.payloads.items():
        try:
            current = path.read_bytes()
        except OSError as exc:
            raise SourceProjectBuildError(
                f"authoritative input became unreadable during build: {path}: {exc}"
            ) from exc
        if current != payload:
            raise SourceProjectBuildError(
                f"authoritative input changed during build: {path}"
            )


def _source_inputs(snapshot: InputSnapshot) -> dict[str, Any]:
    project = snapshot.project
    result: dict[str, Any] = {
        "tikzSource": {
            "path": _relative(project, project.tikz_source),
            "sha256": _sha256_bytes(snapshot.payloads[project.tikz_source]),
        }
    }
    for key, path in (
        ("motionJson", project.motion_json),
        ("hooksSource", project.hooks_source),
        ("bridgeRequestTemplate", project.bridge_request_template),
    ):
        if path is not None:
            result[key] = {
                "path": _relative(project, path),
                "sha256": _sha256_bytes(snapshot.payloads[path]),
            }
    return result


_BRIDGE_SOURCE_KEYS = (
    "sourceText",
    "generatedSource",
    "generated_source",
    "pythonSource",
    "python_source",
    "python",
    "code",
)


def _extract_bridge_python(value: Any) -> str | None:
    """Extract generated Python from both direct and nested Bridge envelopes."""

    if isinstance(value, Mapping):
        if value.get("ok") is False:
            error = value.get("error")
            message = error.get("message") if isinstance(error, Mapping) else error
            raise SourceProjectBuildError(f"v3 Bridge failed: {message or 'unknown error'}")
        for key in _BRIDGE_SOURCE_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
            nested = _extract_bridge_python(candidate)
            if nested is not None:
                return nested
        for key in ("result", "nativeManimSourceV3", "payload", "data"):
            nested = _extract_bridge_python(value.get(key))
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = _extract_bridge_python(child)
            if nested is not None:
                return nested
    return None


def _call_bridge_generator(request: Mapping[str, Any]) -> str | None:
    module = importlib.import_module("tikz_native.geometry_rig_3d_source_v3_bridge")
    response = module.execute_source_v3_request(dict(request))
    return _extract_bridge_python(response)


def _replace_json_placeholders(value: Any, replacements: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{([A-Z][A-Z0-9_]*)\}", value)
        if match:
            name = match.group(1)
            if name not in replacements:
                raise SourceProjectBuildError(
                    f"unsupported Bridge template placeholder: {name}"
                )
            return replacements[name]
        embedded = re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", value)
        if embedded:
            raise SourceProjectBuildError(
                "Bridge JSON placeholders must occupy the complete JSON value: "
                + ", ".join(sorted(set(embedded)))
            )
        return value
    if isinstance(value, list):
        return [_replace_json_placeholders(child, replacements) for child in value]
    if isinstance(value, Mapping):
        return {
            str(key): _replace_json_placeholders(child, replacements)
            for key, child in value.items()
        }
    return value


def _local_bridge_metadata(request: MutableMapping[str, Any]) -> tuple[str, ...]:
    values: Any = None
    for key in (
        "wholeFigureTargets",
        "whole_figure_targets",
        "wholeFigureFadeTargets",
        "whole_figure_fade_targets",
    ):
        if key in request:
            if values is not None:
                raise SourceProjectBuildError(
                    "Bridge template must use only one wholeFigureTargets alias"
                )
            values = request.pop(key)
    return _whole_figure_targets({"wholeFigureTargets": values} if values is not None else None)


_HOOKS_BEGIN = "# >>> TIKZ_NATIVE_USER_HOOKS_V1"
_HOOKS_END = "# <<< TIKZ_NATIVE_USER_HOOKS_V1"


def _append_authored_hooks(source: str, hooks_source: str | None) -> str:
    if hooks_source is None:
        return source if source.endswith("\n") else source + "\n"
    if _HOOKS_BEGIN in hooks_source or _HOOKS_END in hooks_source:
        raise SourceProjectBuildError("hooksSource must not contain reserved hook sentinels")
    prefix = source.rstrip() + "\n\n"
    body = hooks_source
    if body and not body.endswith("\n"):
        body += "\n"
    return prefix + _HOOKS_BEGIN + "\n" + body + _HOOKS_END + "\n"


def _whole_figure_targets(request: Mapping[str, Any] | None) -> tuple[str, ...]:
    if request is None:
        return ()
    values: Any = None
    for key in (
        "wholeFigureTargets",
        "whole_figure_targets",
        "wholeFigureFadeTargets",
        "whole_figure_fade_targets",
    ):
        if key in request:
            values = request[key]
            break
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise SourceProjectBuildError(
            "wholeFigureTargets must be a string or an array of non-empty identifiers"
        )
    return tuple(values)


class _DirectOpenFaceTransformer(ast.NodeTransformer):
    def __init__(
        self,
        *,
        paint_policy: str,
        painter_z_band: PainterZBand,
        whole_figure_targets: Sequence[str],
        controller_name: str | None,
    ) -> None:
        self.paint_policy = paint_policy
        self.painter_z_band = painter_z_band
        self.targets = set(whole_figure_targets)
        self.controller_name = controller_name
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "OpenFaceOcclusion3D":
            if any(keyword.arg is None for keyword in node.keywords):
                raise SourceProjectBuildError(
                    "OpenFaceOcclusion3D calls with **kwargs cannot be rewritten safely"
                )
            replacements = {
                "compositing_mode": ast.Constant("unified"),
                "paint_policy": ast.Constant(self.paint_policy),
                "painter_z_band": ast.Tuple(
                    elts=[
                        ast.Constant(self.painter_z_band.minimum),
                        ast.Constant(self.painter_z_band.maximum),
                    ],
                    ctx=ast.Load(),
                ),
            }
            retained = [
                keyword
                for keyword in node.keywords
                if keyword.arg not in replacements
            ]
            retained.extend(
                ast.keyword(arg=name, value=value)
                for name, value in replacements.items()
            )
            node.keywords = retained
            self.changed = True
            return node
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"FadeIn", "FadeOut"}
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.targets
        ):
            if self.controller_name is None:
                raise SourceProjectBuildError(
                    "wholeFigureTargets requires exactly one OpenFaceOcclusion3D controller"
                )
            node.args[0] = ast.Attribute(
                value=ast.Name(id=self.controller_name, ctx=ast.Load()),
                attr="display_mobject",
                ctx=ast.Load(),
            )
            self.changed = True
        return node


def _rewrite_direct_open_face_source(
    source: str,
    *,
    paint_policy: str,
    painter_z_band: PainterZBand,
    whole_figure_targets: Sequence[str],
) -> str:
    try:
        tree = ast.parse(source, filename="<generated_scene.py>")
    except SyntaxError as exc:
        raise SourceProjectBuildError(
            f"generated Python is invalid before unified rewrite: line {exc.lineno}: {exc.msg}"
        ) from exc
    imported = False
    controller_names: list[str] = []
    has_open_face_name = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported = imported or any(
                alias.name == "OpenFaceOcclusion3D" and alias.asname in {None, "OpenFaceOcclusion3D"}
                for alias in node.names
            )
        if isinstance(node, ast.Name) and node.id == "OpenFaceOcclusion3D":
            has_open_face_name = True
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "OpenFaceOcclusion3D"
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            controller_names.extend(
                target.id for target in targets if isinstance(target, ast.Name)
            )
    if not has_open_face_name:
        if whole_figure_targets:
            raise SourceProjectBuildError(
                "wholeFigureTargets requires generated OpenFaceOcclusion3D source"
            )
        return source if not source or source.endswith("\n") else source + "\n"
    if not imported:
        raise SourceProjectBuildError(
            "generated source calls OpenFaceOcclusion3D without an explicit import binding"
        )
    unique_controllers = sorted(set(controller_names))
    controller_name = unique_controllers[0] if len(unique_controllers) == 1 else None
    transformer = _DirectOpenFaceTransformer(
        paint_policy=paint_policy,
        painter_z_band=painter_z_band,
        whole_figure_targets=whole_figure_targets,
        controller_name=controller_name,
    )
    rewritten_tree = transformer.visit(tree)
    ast.fix_missing_locations(rewritten_tree)
    rewritten = ast.unparse(rewritten_tree)
    return rewritten.rstrip() + "\n"


def project_literal(value: Any) -> str:
    """Return a deterministic Python literal for generated source."""

    return repr(value)


def rewrite_generated_source(
    source: str,
    *,
    paint_policy: str,
    painter_z_band: PainterZBand,
    whole_figure_targets: Sequence[str] = (),
) -> str:
    """Rewrite disposable generated Python to current unified behavior."""

    if paint_policy not in {"diagrammatic", "physical"}:
        raise SourceProjectBuildError(
            "paint_policy must be 'diagrammatic' or 'physical'"
        )
    if any(not isinstance(target, str) or not target.isidentifier() for target in whole_figure_targets):
        raise SourceProjectBuildError(
            "wholeFigureTargets must contain Python identifiers"
        )
    if "install_open_face_visibility_3d" in source:
        try:
            adapter = importlib.import_module(
                "tikz_native.generated_open_face_visibility_3d"
            )
            rewritten = adapter.rewrite_legacy_open_face_source(
                source,
                paint_policy=paint_policy,
                preferred_painter_z_band=tuple(painter_z_band.as_list()),
                whole_figure_targets=whole_figure_targets,
            )
        except (ImportError, AttributeError) as exc:
            raise SourceProjectBuildError(
                "generated v3 source requires the unified open-face source adapter"
            ) from exc
    else:
        rewritten = _rewrite_direct_open_face_source(
            source,
            paint_policy=paint_policy,
            painter_z_band=painter_z_band,
            whole_figure_targets=whole_figure_targets,
        )
    if rewritten and not rewritten.endswith("\n"):
        rewritten += "\n"
    try:
        compile(rewritten, "<generated_scene.py>", "exec")
    except SyntaxError as exc:
        raise SourceProjectBuildError(
            f"generated Python is invalid after unified rewrite: line {exc.lineno}: {exc.msg}"
        ) from exc
    return rewritten


def _generate_scene_source(
    project: SourceProject,
    *,
    snapshot: InputSnapshot,
    snapshot_source_path: Path,
    template_value: Mapping[str, Any],
    motion_value: Any,
    hooks_source: str | None,
    painter_z_band: PainterZBand,
    revisions: Mapping[str, str | int],
    bridge_generator: BridgeGenerator | None,
) -> str:
    tikz_payload = snapshot.payloads[project.tikz_source]
    replacements: dict[str, Any] = {
        "TIKZ_SOURCE": snapshot.text(project.tikz_source, label="TikZ source"),
        "TIKZ_PATH": str(snapshot_source_path),
        "TIKZ_SHA256": _sha256_bytes(tikz_payload),
        "MOTION_JSON": motion_value,
        "HOOKS_SOURCE": hooks_source,
        "PAINT_POLICY": project.paint_policy,
        "PAINTER_Z_BAND": painter_z_band.as_list(),
        "PICTURE_INDEX": project.picture_index,
        "ENTRY_MACRO": project.entry_macro,
        "EXPECTED_ASSET_PROVIDER_REVISION": revisions["asset_compiler"],
    }
    request = _replace_json_placeholders(dict(template_value), replacements)
    if not isinstance(request, dict):
        raise SourceProjectBuildError("Bridge request template root must be an object")
    whole_figure_targets = _local_bridge_metadata(request)

    if _extract_bridge_python(request) is not None:
        raise SourceProjectBuildError(
            "Bridge request templates must not embed generated Python; "
            "generated source is disposable and must come from the current v3 Bridge"
        )
    bridge = importlib.import_module("tikz_native.geometry_rig_3d_source_v3_bridge")
    request["schema"] = bridge.GEOMETRY_RIG_3D_SOURCE_V3_REQUEST_SCHEMA
    request["operation"] = bridge.GEOMETRY_RIG_3D_SOURCE_V3_OPERATION
    request["job_id"] = "source-project-" + _sha256_bytes(tikz_payload)[:16]
    request["input"] = {
        "source_path": str(snapshot_source_path),
        "source_sha256": _sha256_bytes(tikz_payload),
        "entry_macro": project.entry_macro,
        "picture_index": project.picture_index,
        "expected_asset_provider_revision": str(revisions["asset_compiler"]),
    }
    # The manifest, including an empty selection, is authoritative.  A request
    # template cannot retain a hidden selection that would be absent from the
    # generated-source cache key and build-manifest authoring intent.
    if project.selection:
        request["selection"] = dict(project.selection)
    else:
        request.pop("selection", None)
    if bridge_generator is not None:
        generated_value = bridge_generator(request)
        generated = (
            generated_value
            if isinstance(generated_value, str)
            else _extract_bridge_python(generated_value)
        )
    else:
        generated = _call_bridge_generator(request)
    if generated is None:
        raise SourceProjectBuildError("v3 Bridge did not return generated Python")

    rewritten = rewrite_generated_source(
        generated,
        paint_policy=project.paint_policy,
        painter_z_band=painter_z_band,
        whole_figure_targets=whole_figure_targets,
    )
    rewritten = _append_authored_hooks(rewritten, hooks_source)
    try:
        compile(rewritten, "<generated_scene.py>", "exec")
    except SyntaxError as exc:
        raise SourceProjectBuildError(
            f"generated Python with authored hooks is invalid: line {exc.lineno}: {exc.msg}"
        ) from exc
    return rewritten


def _plan_nodes(
    project: SourceProject,
    *,
    snapshot: InputSnapshot,
    component_revisions: Mapping[str, str | int] | None,
    shape_asset_builder: ShapeAssetBuilder | None,
    bridge_generator: BridgeGenerator | None,
) -> tuple[list[NodePlan], PainterZBand, dict[str, Any], Any]:
    revisions = _normalise_revisions(component_revisions)
    tikz_bytes = snapshot.payloads[project.tikz_source]
    tikz_source = snapshot.text(project.tikz_source, label="TikZ source")
    painter_z_band = derive_painter_z_band(project, tikz_bytes)
    projection_digest = _sha256_bytes(_canonical_json(project.projection))

    compiler_revision = {
        name: revisions[name]
        for name in ("source_project_build", "asset_compiler")
    }
    shape_key = _build_key(
        {
            "node": "shape",
            "schemaVersion": SHAPE_ASSET_SCHEMA_VERSION,
            "tikzSha256": _sha256_bytes(tikz_bytes),
            "pictureIndex": project.picture_index,
            "entryMacro": project.entry_macro,
            "componentRevisions": compiler_revision,
        }
    )
    def build_shape(context: _BuildContext) -> bytes:
        builder = context.shape_asset_builder or _default_shape_asset_builder
        compiled = _normalise_compiler_result(builder(project, tikz_source))
        if context.shape_asset_builder is None and (
            not isinstance(compiled, Mapping)
            or compiled.get("schema") != "tikz-native-asset/v1"
        ):
            raise SourceProjectBuildError(
                "the real Provider compiler did not return a tikz-native-asset/v1 ShapeAsset"
            )
        if context.shape_asset_builder is None:
            compiled = dict(compiled)
            # Full health metadata contains process diagnostics and revisions
            # for unrelated components.  Keeping those fields in a disposable
            # ShapeAsset would make identical compiler inputs produce different
            # bytes after an unrelated renderer change.  Persist only the
            # asset-facing identity that is already part of this node's key.
            compiled["provider"] = {
                "name": "manim-tikz-native",
                "asset_schema": "tikz-native-asset/v1",
                "revision": revisions["asset_compiler"],
                "revision_component": "asset_compiler",
            }
        # `shape-asset.json` is the existing public ShapeAsset itself.  Build
        # keys and compiler revisions belong in build-manifest.json; wrapping
        # the asset would make ordinary Provider consumers unable to load it.
        return _canonical_json(compiled)
    plans = [
        NodePlan(
            name="shape",
            output_name="shape-asset.json",
            key=shape_key,
            component_revisions=compiler_revision,
            build_payload=build_shape,
        )
    ]

    motion_value: Any = None
    motion_key: str | None = None
    if project.motion_json is not None:
        motion_source = snapshot.text(project.motion_json, label="motion JSON")
        try:
            motion_value = json.loads(
                motion_source, parse_constant=_reject_json_constant
            )
        except json.JSONDecodeError as exc:
            raise SourceProjectError(
                f"invalid motion JSON {project.motion_json}: line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc
        except ValueError as exc:
            raise SourceProjectError(
                f"invalid motion JSON {project.motion_json}: {exc}"
            ) from exc
        motion_revision = {
            "source_project_build": revisions["source_project_build"]
        }
        motion_key = _build_key(
            {
                "node": "motion",
                "schemaVersion": MOTION_ASSET_SCHEMA_VERSION,
                "shapeKey": shape_key,
                "motionSha256": _sha256_bytes(motion_source.encode("utf-8")),
                "componentRevisions": motion_revision,
            }
        )
        def build_motion(context: _BuildContext) -> bytes:
            return _canonical_json({
                "schemaVersion": MOTION_ASSET_SCHEMA_VERSION,
                "buildKey": motion_key,
                "source": {
                    "path": _relative(project, project.motion_json),
                    "sha256": _sha256_bytes(motion_source.encode("utf-8")),
                },
                "shapeAssetSha256": context.digests["shape"],
                "motion": motion_value,
            })
        plans.append(
            NodePlan(
                name="motion",
                output_name="motion-asset.json",
                key=motion_key,
                component_revisions=motion_revision,
                build_payload=build_motion,
            )
        )

    compositing_revision = {
        name: revisions[name]
        for name in (
            "source_project_build",
            "open_face_unified_compositing",
            "managed_painter_band",
        )
    }
    compositing_key = _build_key(
        {
            "node": "compositing",
            "schemaVersion": COMPOSITING_SCHEMA_VERSION,
            "shapeKey": shape_key,
            "motionKey": motion_key,
            "paintPolicy": project.paint_policy,
            "projectionSha256": projection_digest,
            "painterZBand": painter_z_band.as_list(),
            "componentRevisions": compositing_revision,
        }
    )
    def build_compositing(context: _BuildContext) -> bytes:
        return _canonical_json({
            "schemaVersion": COMPOSITING_SCHEMA_VERSION,
            "buildKey": compositing_key,
            "compositingMode": "unified",
            "paintPolicy": project.paint_policy,
            "projection": project.projection,
            "painterZBand": painter_z_band.as_list(),
            "shapeAssetSha256": context.digests["shape"],
            "motionAssetSha256": context.digests.get("motion"),
            "component": {
                "name": "unified_compositor",
                "revision": revisions["open_face_unified_compositing"],
            },
        })
    plans.append(
        NodePlan(
            name="compositing",
            output_name="unified-compositing.json",
            key=compositing_key,
            component_revisions=compositing_revision,
            build_payload=build_compositing,
        )
    )

    hooks_source: str | None = None
    if project.hooks_source is not None:
        hooks_source = snapshot.text(project.hooks_source, label="hooks source")

    if project.bridge_request_template is not None:
        template = snapshot.text(
            project.bridge_request_template, label="Bridge request template"
        )
        try:
            template_value = json.loads(
                template, parse_constant=_reject_json_constant
            )
        except json.JSONDecodeError as exc:
            raise SourceProjectError(
                f"Bridge request template must be JSON: line {exc.lineno}, column {exc.colno}"
            ) from exc
        except ValueError as exc:
            raise SourceProjectError(
                f"Bridge request template must be JSON: {exc}"
            ) from exc
        if not isinstance(template_value, Mapping):
            raise SourceProjectError("Bridge request template root must be an object")
        generated_revision = {
            name: revisions[name]
            for name in (
                "source_project_build",
                "generated_open_face_visibility_3d",
            )
        }
        generated_key = _build_key(
            {
                "node": "generated_source",
                "compositingKey": compositing_key,
                "bridgeTemplateSha256": _sha256_bytes(template.encode("utf-8")),
                "hooksSha256": (
                    _sha256_bytes(hooks_source.encode("utf-8"))
                    if hooks_source is not None
                    else None
                ),
                "paintPolicy": project.paint_policy,
                "painterZBand": painter_z_band.as_list(),
                "selection": project.selection,
                "componentRevisions": generated_revision,
            }
        )
        def build_generated(context: _BuildContext) -> bytes:
            generated = _generate_scene_source(
                project,
                snapshot=snapshot,
                snapshot_source_path=context.source_path(),
                template_value=template_value,
                motion_value=motion_value,
                hooks_source=hooks_source,
                painter_z_band=painter_z_band,
                revisions=revisions,
                bridge_generator=context.bridge_generator,
            )
            return generated.encode("utf-8")
        plans.append(
            NodePlan(
                name="generated_source",
                output_name="generated_scene.py",
                key=generated_key,
                component_revisions=generated_revision,
                build_payload=build_generated,
            )
        )

    inputs = _source_inputs(snapshot)
    return plans, painter_z_band, inputs, motion_value


def _load_previous_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("schemaVersion") != BUILD_MANIFEST_SCHEMA_VERSION:
        return None
    nodes = value.get("nodes")
    if not isinstance(nodes, dict):
        return None
    return value


def _safe_output_path(project: SourceProject, output_name: str) -> Path:
    path = project.output_directory.joinpath(output_name)
    try:
        path.resolve(strict=False).relative_to(project.output_directory.resolve(strict=False))
    except ValueError as exc:
        raise SourceProjectBuildError(f"derived output escapes output directory: {output_name}") from exc
    return path


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _owned_output_payload(project: SourceProject) -> bytes:
    # Ownership is deliberately relative to the source-project root.  A source
    # project and its disposable output may be copied or moved together without
    # turning a safe rebuild/clean into a false ownership mismatch.
    return _canonical_json(
        {
            "schemaVersion": OWNED_OUTPUT_SCHEMA_VERSION,
            "project": project.manifest_path.name,
            "outputDirectory": project.output_directory.relative_to(
                project.root
            ).as_posix(),
        }
    )


def _validate_owned_output(
    project: SourceProject, *, allow_absent: bool, allow_empty: bool = False
) -> bool:
    output = project.output_directory
    if output.is_symlink():
        raise SourceProjectBuildError("refusing to use a symlink output directory")
    if not output.exists():
        if allow_absent:
            return False
        raise SourceProjectBuildError("derived output directory does not exist")
    if not output.is_dir():
        raise SourceProjectBuildError("derived output path is not a directory")
    entries = {path.name for path in output.iterdir()}
    if not entries and allow_empty:
        return True
    marker = output / OWNED_OUTPUT_MARKER
    if not marker.is_file() or marker.is_symlink():
        raise SourceProjectBuildError(
            "refusing to replace or clean an output directory without its ownership marker"
        )
    if marker.read_bytes() != _owned_output_payload(project):
        raise SourceProjectBuildError("derived output ownership marker does not match this project")
    unknown = sorted(entries - _KNOWN_OUTPUT_NAMES)
    if unknown:
        raise SourceProjectBuildError(
            "refusing to replace or clean output containing unowned entries: "
            + ", ".join(unknown)
        )
    invalid_generated = sorted(
        name
        for name in entries - {OWNED_OUTPUT_MARKER}
        if (output / name).is_symlink() or not (output / name).is_file()
    )
    if invalid_generated:
        raise SourceProjectBuildError(
            "refusing to replace or clean a derived output whose generated "
            "entries are not regular files: " + ", ".join(invalid_generated)
        )
    return True


@contextlib.contextmanager
def _project_lock(project: SourceProject) -> Iterator[None]:
    # Lock the already-existing project directory.  It is stable even if an
    # editor atomically replaces project.json with a new inode, and it avoids a
    # surprising write from read-only `status` or a no-op `clean`.  Manifests
    # sharing one directory are intentionally serialised.
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(project.root, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _ensure_output_parent(project: SourceProject) -> None:
    lock_parent = project.output_directory.parent
    try:
        lock_parent.mkdir(parents=True, exist_ok=True)
        lock_parent.resolve(strict=True).relative_to(project.root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SourceProjectBuildError(
            "cannot create a safe parent for the derived output"
        ) from exc
    current = project.root
    for part in lock_parent.relative_to(project.root).parts:
        current = current / part
        if current.is_symlink():
            raise SourceProjectBuildError(
                "refusing to lock a derived output through a symlink"
            )


def _publish_staged_directory(stage: Path, output: Path) -> None:
    if not output.exists():
        os.replace(stage, output)
        return
    # macOS provides an atomic directory exchange.  Keep a rollback fallback
    # for platforms where renameatx_np is unavailable.
    try:
        rename_swap = ctypes.CDLL(None, use_errno=True).renameatx_np
        rename_swap.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename_swap.restype = ctypes.c_int
        if rename_swap(-2, os.fsencode(stage), -2, os.fsencode(output), 2) == 0:
            # Publication has committed: ``stage`` now names the old output.
            # A cleanup failure must not turn a successful build into a false
            # failure whose visible output has already changed.
            shutil.rmtree(stage, ignore_errors=True)
            return
        error = ctypes.get_errno()
        if error not in {errno.ENOSYS, errno.ENOTSUP, errno.EINVAL}:
            raise OSError(error, os.strerror(error))
    except AttributeError:
        pass
    backup = output.parent / f".{output.name}.rollback-{uuid.uuid4().hex}"
    os.replace(output, backup)
    try:
        os.replace(stage, output)
    except BaseException:
        os.replace(backup, output)
        raise
    # As above, publication has already committed.  Best-effort cleanup keeps
    # the command result truthful if the platform refuses to remove a stale
    # backup immediately.
    shutil.rmtree(backup, ignore_errors=True)


def _cache_hit(
    project: SourceProject,
    plan: NodePlan,
    previous_nodes: Mapping[str, Any],
) -> bool:
    previous = previous_nodes.get(plan.name)
    if not isinstance(previous, Mapping) or previous.get("key") != plan.key:
        return False
    output = _safe_output_path(project, plan.output_name)
    if not output.is_file():
        return False
    expected_digest = previous.get("sha256")
    return isinstance(expected_digest, str) and _sha256_file(output) == expected_digest


def _manifest_payload(
    project: SourceProject,
    *,
    nodes: Sequence[NodeState],
    inputs: Mapping[str, Any],
    painter_z_band: PainterZBand,
    revisions: Mapping[str, str | int],
) -> bytes:
    return _canonical_json(
        {
            "schemaVersion": BUILD_MANIFEST_SCHEMA_VERSION,
            "sourceProjectSchemaVersion": SOURCE_PROJECT_SCHEMA_VERSION,
            "provider": provider_component_descriptor(),
            "project": project.manifest_path.name,
            "inputs": inputs,
            "authoringIntent": {
                "pictureIndex": project.picture_index,
                "entryMacro": project.entry_macro,
                "selection": project.selection,
            },
            "renderIntent": {
                "paintPolicy": project.paint_policy,
                "projection": project.projection,
                "painterZBand": painter_z_band.as_list(),
            },
            "componentRevisions": dict(sorted(revisions.items())),
            "nodes": {
                node.name: {
                    "key": node.key,
                    "output": node.output,
                    "sha256": node.sha256,
                }
                for node in nodes
            },
        }
    )


def _planned_component_revisions(
    plans: Sequence[NodePlan],
) -> dict[str, str | int]:
    result: dict[str, str | int] = {}
    for plan in plans:
        for name, revision in plan.component_revisions.items():
            previous = result.setdefault(name, revision)
            if previous != revision:
                raise SourceProjectBuildError(
                    f"component revision {name!r} is inconsistent across build nodes"
                )
    return dict(sorted(result.items()))


def build_project(
    manifest_path: str | os.PathLike[str],
    *,
    force: bool = False,
    component_revisions: Mapping[str, str | int] | None = None,
    shape_asset_builder: ShapeAssetBuilder | None = None,
    bridge_generator: BridgeGenerator | None = None,
) -> BuildResult:
    """Build or incrementally refresh all derived project artifacts."""

    _validate_injected_builder_revisions(
        component_revisions,
        shape_asset_builder=shape_asset_builder,
        bridge_generator=bridge_generator,
    )
    project = load_source_project(manifest_path)
    snapshot = _capture_input_snapshot(project)
    revisions = _normalise_revisions(component_revisions)
    plans, painter_z_band, inputs, motion_value = _plan_nodes(
        project,
        snapshot=snapshot,
        component_revisions=revisions,
        shape_asset_builder=shape_asset_builder,
        bridge_generator=bridge_generator,
    )
    manifest_revisions = _planned_component_revisions(plans)
    manifest_path_obj = project.output_directory / BUILD_MANIFEST_NAME
    states: list[NodeState] = []
    context = _BuildContext(
        project, snapshot, revisions, painter_z_band, motion_value,
        shape_asset_builder, bridge_generator, {},
    )
    stage: Path | None = None
    try:
        with _project_lock(project):
            _ensure_output_parent(project)
            _validate_owned_output(project, allow_absent=True, allow_empty=True)
            previous = None if force else _load_previous_manifest(manifest_path_obj)
            previous_nodes: Mapping[str, Any] = (
                previous.get("nodes", {}) if isinstance(previous, Mapping) else {}
            )
            stage = Path(tempfile.mkdtemp(
                prefix=f".{project.output_directory.name}.stage-",
                dir=project.output_directory.parent,
            ))
            for plan in plans:
                old_output = _safe_output_path(project, plan.output_name)
                staged_output = stage / plan.output_name
                if not force and _cache_hit(project, plan, previous_nodes):
                    shutil.copy2(old_output, staged_output)
                    action = "reused"
                    digest = _sha256_file(old_output)
                else:
                    payload = plan.build_payload(context)
                    _atomic_write(staged_output, payload)
                    action = "built"
                    digest = _sha256_bytes(payload)
                context.digests[plan.name] = digest
                states.append(NodeState(
                    name=plan.name, action=action, key=plan.key,
                    output=plan.output_name, sha256=digest,
                ))
            _atomic_write(stage / BUILD_MANIFEST_NAME, _manifest_payload(
                project, nodes=states, inputs=inputs,
                painter_z_band=painter_z_band, revisions=manifest_revisions,
            ))
            _atomic_write(stage / OWNED_OUTPUT_MARKER, _owned_output_payload(project))
            _validate_input_snapshot(snapshot)
            _publish_staged_directory(stage, project.output_directory)
            stage = None
    finally:
        context.close()
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    return BuildResult(
        project=project,
        mode="rebuild" if force else "build",
        nodes=tuple(states),
        manifest_path=manifest_path_obj,
        painter_z_band=painter_z_band,
    )


def rebuild_project(
    manifest_path: str | os.PathLike[str],
    **kwargs: Any,
) -> BuildResult:
    """Rebuild every derived node, ignoring cache hits."""

    kwargs.pop("force", None)
    return build_project(manifest_path, force=True, **kwargs)


def status_project(
    manifest_path: str | os.PathLike[str],
    *,
    component_revisions: Mapping[str, str | int] | None = None,
    shape_asset_builder: ShapeAssetBuilder | None = None,
    bridge_generator: BridgeGenerator | None = None,
) -> ProjectStatus:
    """Return whether each expected derived node is fresh."""

    _validate_injected_builder_revisions(
        component_revisions,
        shape_asset_builder=shape_asset_builder,
        bridge_generator=bridge_generator,
    )
    project = load_source_project(manifest_path)
    snapshot = _capture_input_snapshot(project)
    revisions = _normalise_revisions(component_revisions)
    plans, painter_z_band, inputs, _ = _plan_nodes(
        project,
        snapshot=snapshot,
        component_revisions=revisions,
        shape_asset_builder=shape_asset_builder,
        bridge_generator=bridge_generator,
    )
    manifest_revisions = _planned_component_revisions(plans)
    manifest_path_obj = project.output_directory / BUILD_MANIFEST_NAME
    states: list[NodeState] = []
    with _project_lock(project):
        owned = _validate_owned_output(project, allow_absent=True)
        previous = _load_previous_manifest(manifest_path_obj) if owned else None
        previous_nodes: Mapping[str, Any] = (
            previous.get("nodes", {}) if isinstance(previous, Mapping) else {}
        )
        for plan in plans:
            output = _safe_output_path(project, plan.output_name)
            hit = owned and _cache_hit(project, plan, previous_nodes)
            action = "fresh" if hit else (
                "stale" if owned and (output.exists() or previous is not None) else "missing"
            )
            digest = _sha256_file(output) if output.is_file() else ""
            states.append(NodeState(plan.name, action, plan.key, plan.output_name, digest))
        expected_names = {plan.name for plan in plans}
        for name in sorted(set(previous_nodes) - expected_names):
            old = previous_nodes[name]
            output_name = old.get("output", "") if isinstance(old, Mapping) else ""
            states.append(NodeState(
                name, "obsolete", "", output_name if isinstance(output_name, str) else "", ""
            ))
        _validate_input_snapshot(snapshot)
    nodes_fresh = previous is not None and all(
        node.action == "fresh" for node in states
    )
    expected_manifest = _manifest_payload(
        project,
        nodes=states,
        inputs=inputs,
        painter_z_band=painter_z_band,
        revisions=manifest_revisions,
    )
    try:
        manifest_fresh = manifest_path_obj.read_bytes() == expected_manifest
        manifest_action = "fresh" if manifest_fresh else "stale"
    except OSError:
        manifest_fresh = False
        manifest_action = "missing"
    fresh = nodes_fresh and manifest_fresh
    return ProjectStatus(
        project=project,
        fresh=fresh,
        nodes=tuple(states),
        manifest_path=manifest_path_obj,
        manifest_action=manifest_action,
        painter_z_band=painter_z_band,
    )


def clean_project(manifest_path: str | os.PathLike[str]) -> Path:
    """Remove only the validated derived output directory."""

    project = load_source_project(manifest_path)
    output = project.output_directory
    with _project_lock(project):
        if not output.exists() and not output.is_symlink():
            return output
        _validate_owned_output(project, allow_absent=False)
        tombstone = output.parent / f".{output.name}.clean-{uuid.uuid4().hex}"
        os.replace(output, tombstone)
    shutil.rmtree(tombstone)
    return output


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tikz-native-project",
        description="Build disposable artifacts from an authoritative TikZ source project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "status", "rebuild", "clean"):
        child = subparsers.add_parser(command)
        child.add_argument("project", help="path to project.json")
    return parser


@contextlib.contextmanager
def _redirect_operation_output_to_stderr() -> Iterator[None]:
    """Keep the CLI's stdout as one machine-readable JSON document.

    Manim's Rich logger is created while the package is imported and keeps a
    reference to the then-current stdout.  Redirecting ``sys.stdout`` alone
    therefore does not catch first-build TeX/compiler messages.  Temporarily
    retarget the default Manim consoles as well, while leaving an explicitly
    configured log file untouched.
    """

    consoles: list[tuple[Any, Any, bool]] = []
    seen: set[int] = set()
    manim_module = sys.modules.get("manim")
    candidates: list[Any] = []
    if manim_module is not None:
        candidates.append(getattr(manim_module, "console", None))
        logger = getattr(manim_module, "logger", None)
        for handler in getattr(logger, "handlers", ()):
            candidates.append(getattr(handler, "console", None))
    for console in candidates:
        if console is None or id(console) in seen:
            continue
        seen.add(id(console))
        original_override = getattr(console, "_file", None)
        current_file = getattr(console, "file", None)
        if original_override is not None and current_file not in {
            sys.stdout,
            sys.__stdout__,
        }:
            continue
        original_stderr = bool(getattr(console, "stderr", False))
        consoles.append((console, original_override, original_stderr))
        console.file = sys.stderr
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        for console, original_override, original_stderr in reversed(consoles):
            console.file = original_override
            console.stderr = original_stderr


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        with _redirect_operation_output_to_stderr():
            if arguments.command == "build":
                payload = build_project(arguments.project).as_dict()
                exit_code = 0
            elif arguments.command == "rebuild":
                payload = rebuild_project(arguments.project).as_dict()
                exit_code = 0
            elif arguments.command == "status":
                status = status_project(arguments.project)
                payload = status.as_dict()
                exit_code = 0 if status.fresh else 1
            else:
                removed = clean_project(arguments.project)
                payload = {
                    "resultFormat": COMMAND_RESULT_FORMAT_VERSION,
                    "mode": "clean",
                    "project": str(Path(arguments.project).resolve()),
                    "removed": str(removed),
                }
                exit_code = 0
    except (SourceProjectError, SourceProjectBuildError, OSError) as exc:
        print(f"tikz-native-project: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return exit_code


__all__ = [
    "BUILD_MANIFEST_SCHEMA_VERSION",
    "BuildResult",
    "COMMAND_RESULT_FORMAT_VERSION",
    "PROVIDER_CAPABILITY",
    "PROVIDER_COMPONENT",
    "PainterZBand",
    "ProjectStatus",
    "SOURCE_PROJECT_SCHEMA_VERSION",
    "SourceProject",
    "SourceProjectBuildError",
    "SourceProjectError",
    "build_project",
    "clean_project",
    "derive_painter_z_band",
    "load_source_project",
    "main",
    "provider_component_descriptor",
    "rebuild_project",
    "rewrite_generated_source",
    "status_project",
]


if __name__ == "__main__":
    raise SystemExit(main())
