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
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

from .parallel_shots import (
    PARALLEL_CAMERA_SHOT_SEQUENCE_SCHEMA,
    ParallelCameraShotSequence,
    canonical_parallel_camera_shot_sequence_json,
    parallel_camera_shot_sequence_from_dict,
)

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
        "camera-shots.json",
        "unified-compositing.json",
        "generated_scene.py",
    }
)

_FORBIDDEN_MANIFEST_KEYS = {
    "compositingmode",
    "implementationmode",
}

_PAINTER_Z_BAND_BASE = 10_000.0
_PAINTER_Z_BAND_WIDTH = 1024.0
_PAINTER_Z_BAND_GAP = 1024.0
_PAINTER_Z_BAND_STRIDE = _PAINTER_Z_BAND_WIDTH + _PAINTER_Z_BAND_GAP
_PAINTER_Z_BAND_SLOT_COUNT = 4096


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
    camera_shots: Path | None
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
    _snapshot_directory: "_SafeSourceSnapshot | None" = None
    _snapshot_source_path: Path | None = None

    def source_path(self) -> Path:
        if self._snapshot_source_path is None:
            self._snapshot_directory = _SafeSourceSnapshot.create(
                prefix="tikz-native-input-",
                source_name=self.project.tikz_source.name,
                payload=self.snapshot.payloads[self.project.tikz_source],
            )
            self._snapshot_source_path = self._snapshot_directory.path
        return self._snapshot_source_path

    def close(self) -> None:
        if self._snapshot_directory is not None:
            self._snapshot_directory.close()
            self._snapshot_directory = None
            self._snapshot_source_path = None


@dataclass
class _SafeSourceSnapshot:
    """One path-backed input whose cleanup never recursively follows a name."""

    parent_path: Path
    parent_descriptor: int
    directory_name: str
    directory_descriptor: int
    source_name: str
    source_identity: tuple[int, int]
    source_sha256: str
    _closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        prefix: str,
        source_name: str,
        payload: bytes,
    ) -> "_SafeSourceSnapshot":
        source_name = _simple_entry_name(source_name)
        parent_path = Path(tempfile.gettempdir()).resolve()
        parent_descriptor = os.open(parent_path, _directory_open_flags())
        directory_name = ""
        directory_descriptor: int | None = None
        try:
            for _ in range(16):
                candidate = f".{prefix}{uuid.uuid4().hex}"
                try:
                    os.mkdir(candidate, 0o700, dir_fd=parent_descriptor)
                except FileExistsError:
                    continue
                directory_name = candidate
                break
            if not directory_name:
                raise SourceProjectBuildError(
                    "cannot allocate a secure source snapshot directory"
                )
            directory_descriptor = os.open(
                directory_name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
            _atomic_write_at(directory_descriptor, source_name, payload)
            source_descriptor = _open_regular_at(directory_descriptor, source_name)
            try:
                source_stat = os.fstat(source_descriptor)
                source_identity = (source_stat.st_dev, source_stat.st_ino)
                source_sha256 = _sha256_descriptor(source_descriptor)
            finally:
                os.close(source_descriptor)
            snapshot = cls(
                parent_path=parent_path,
                parent_descriptor=parent_descriptor,
                directory_name=directory_name,
                directory_descriptor=directory_descriptor,
                source_name=source_name,
                source_identity=source_identity,
                source_sha256=source_sha256,
            )
            snapshot.validate()
            return snapshot
        except BaseException:
            if directory_descriptor is not None:
                os.close(directory_descriptor)
            # Preserve any partially created directory.  Guessing which inode
            # a mutable name now denotes would make error cleanup destructive.
            os.close(parent_descriptor)
            raise

    @property
    def path(self) -> Path:
        return self.parent_path / self.directory_name / self.source_name

    def validate(self) -> None:
        if self._closed:
            raise SourceProjectBuildError("source snapshot is already closed")
        _require_named_directory_identity(
            self.parent_descriptor,
            self.directory_name,
            self.directory_descriptor,
            label="source snapshot directory",
        )
        if set(os.listdir(self.directory_descriptor)) != {self.source_name}:
            raise SourceProjectBuildError("source snapshot changed concurrently")
        source_descriptor = _open_regular_at(
            self.directory_descriptor,
            self.source_name,
        )
        try:
            source_stat = os.fstat(source_descriptor)
            if (source_stat.st_dev, source_stat.st_ino) != self.source_identity:
                raise SourceProjectBuildError("source snapshot changed concurrently")
            if _sha256_descriptor(source_descriptor) != self.source_sha256:
                raise SourceProjectBuildError("source snapshot changed concurrently")
        finally:
            os.close(source_descriptor)
        if set(os.listdir(self.directory_descriptor)) != {self.source_name}:
            raise SourceProjectBuildError("source snapshot changed concurrently")
        _require_named_directory_identity(
            self.parent_descriptor,
            self.directory_name,
            self.directory_descriptor,
            label="source snapshot directory",
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.validate()
            detached_name = _move_to_unique_sibling(
                self.parent_descriptor,
                self.directory_name,
                prefix=f"{self.directory_name}.cleanup",
            )
            self.directory_name = detached_name
            self.validate()
            if not _unlink_matching_identity(
                self.directory_descriptor,
                self.source_name,
                self.source_identity,
            ):
                raise SourceProjectBuildError("source snapshot changed concurrently")
            _require_named_directory_identity(
                self.parent_descriptor,
                self.directory_name,
                self.directory_descriptor,
                label="source snapshot directory",
            )
            if os.listdir(self.directory_descriptor):
                raise SourceProjectBuildError("source snapshot changed concurrently")
            os.rmdir(self.directory_name, dir_fd=self.parent_descriptor)
            os.fsync(self.parent_descriptor)
        finally:
            self._closed = True
            os.close(self.directory_descriptor)
            os.close(self.parent_descriptor)


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


def _parse_finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _reject_json_constant(value)
    return result


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is not allowed: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(value: str) -> Any:
    """Decode authoritative JSON without ambiguous keys or non-finite numbers."""

    decoded = json.loads(
        value,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    return _normalize_json_strings(decoded)


def _normalize_json_strings(value: Any) -> Any:
    """Normalize escaped surrogate pairs and reject unpaired surrogates.

    Python's JSON decoder intentionally accepts isolated ``\\uD800`` style
    code units. They cannot be encoded as canonical UTF-8 later, so an
    authoritative document must fail at its parsing boundary instead of
    leaking a ``UnicodeEncodeError`` from a later build step.
    """

    if isinstance(value, str):
        try:
            return value.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "unpaired UTF-16 surrogate is not allowed in JSON strings"
            ) from exc
    if isinstance(value, list):
        return [_normalize_json_strings(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _normalize_json_strings(raw_key)
            if key in normalized:
                raise ValueError(
                    "duplicate JSON object key is not allowed after Unicode "
                    f"normalization: {key!r}"
                )
            normalized[key] = _normalize_json_strings(raw_value)
        return normalized
    return value


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
        value = _strict_json_loads(_read_utf8(path, label=label))
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
        if not isinstance(candidate_id, str) or not candidate_id.strip():
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
        if float(selected_range[0]) >= float(selected_range[1]):
            raise SourceProjectError("selection.range must contain increasing values")
        result["range"] = list(selected_range)
    for key in ("include_object_ids", "exclude_object_ids"):
        object_ids = value.get(key)
        if object_ids is None:
            continue
        if (
            not isinstance(object_ids, list)
            or any(not isinstance(item, str) or not item.strip() for item in object_ids)
            or len(set(object_ids)) != len(object_ids)
        ):
            raise SourceProjectError(
                f"selection.{key} must contain unique non-empty strings"
            )
        result[key] = list(object_ids)
    contradictory_ids = sorted(
        set(result.get("include_object_ids", ()))
        & set(result.get("exclude_object_ids", ()))
    )
    if contradictory_ids:
        raise SourceProjectError(
            "selection includes and excludes the same objects: "
            + ", ".join(contradictory_ids)
        )
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
            "cameraShots",
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
    camera_shots_raw = _one_of(raw, ("cameraShots",))
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
    camera_shots = (
        _resolve_project_path(
            root,
            camera_shots_raw,
            label="cameraShots",
            must_exist=True,
        )
        if camera_shots_raw is not None
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
        for path in (
            tikz_source,
            motion_json,
            camera_shots,
            hooks_source,
            bridge_template,
        )
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
    if motion_json is not None and camera_shots is not None:
        raise SourceProjectError(
            "cameraShots and motionJson cannot both be present until one "
            "coordinated timeline owns the scene camera"
        )

    return SourceProject(
        manifest_path=manifest,
        root=root,
        tikz_source=tikz_source,
        motion_json=motion_json,
        camera_shots=camera_shots,
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
    """Derive a stable preferred z band from the semantic figure identity."""

    if project.painter_z_band_override is not None:
        return project.painter_z_band_override
    if tikz_bytes is None:
        tikz_bytes = project.tikz_source.read_bytes()
    digest = hashlib.sha256()
    digest.update(tikz_bytes)
    digest.update(b"\0projection\0")
    digest.update(_canonical_json(project.projection))
    # ``pictureIndex`` and ``entryMacro`` select different semantic figures
    # from the same TikZ bytes. They therefore need distinct automatic painter
    # reservations. Motion/paint/selection edits intentionally keep the same
    # band because they do not change that figure identity.
    digest.update(b"\0picture-index\0")
    digest.update(str(project.picture_index).encode("ascii"))
    digest.update(b"\0entry-macro\0")
    digest.update(_canonical_json(project.entry_macro))
    offset = (
        int(digest.hexdigest()[:8], 16) % _PAINTER_Z_BAND_SLOT_COUNT
    )
    # Managed-band bounds are inclusive. Distinct hash slots therefore need a
    # stride greater than the complete band width, rather than merely distinct
    # starting z-indices. A true hash collision yields the same preferred band;
    # the generated Scene binding reserves an available actual band at attach
    # time instead of allowing two controllers to overlap.
    minimum = _PAINTER_Z_BAND_BASE + offset * _PAINTER_Z_BAND_STRIDE
    return PainterZBand(minimum, minimum + _PAINTER_Z_BAND_WIDTH)


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
                    return _strict_json_loads(stripped)
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
    snapshot = _SafeSourceSnapshot.create(
        prefix="tikz-native-compile-",
        source_name=project.tikz_source.name,
        payload=source.encode("utf-8"),
    )
    try:
        try:
            compiled = compile_asset(
                snapshot.path,
                source_sha256=_sha256_bytes(source.encode("utf-8")),
                entry_macro=project.entry_macro,
                picture_index=project.picture_index,
                strict_native=True,
            )
            snapshot.validate()
        finally:
            snapshot.close()
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
        "embedded_motion_3d": version.COMPONENT_EMBEDDED_MOTION_3D,
        "parallel_camera_core": getattr(
            version, "COMPONENT_PARALLEL_CAMERA_CORE", "parallel_camera_core"
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
    supplied_names: set[str] = set()
    if revisions is not None:
        for supplied_name, revision in revisions.items():
            if not isinstance(supplied_name, str) or not supplied_name:
                raise SourceProjectError("component revision names must be non-empty strings")
            name = _REVISION_ALIASES.get(supplied_name, supplied_name)
            supplied_names.add(name)
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
    # Before the parallel-camera core acquired its own component identity,
    # callers injected ``embedded_motion_3d`` for camera-shot cache tests.  Keep
    # that override fail-closed: unless the new identity is supplied explicitly,
    # the legacy value invalidates both components.
    if (
        "embedded_motion_3d" in supplied_names
        and "parallel_camera_core" not in supplied_names
    ):
        result["parallel_camera_core"] = result["embedded_motion_3d"]
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
            project.camera_shots,
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
        ("cameraShots", project.camera_shots),
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
_CAMERA_SHOTS_BEGIN = "# >>> TIKZ_NATIVE_CAMERA_SHOTS_V1"
_CAMERA_SHOTS_END = "# <<< TIKZ_NATIVE_CAMERA_SHOTS_V1"


def _append_camera_shots_binding(
    source: str,
    camera_shots_json: str | None,
) -> str:
    prefix = source if source.endswith("\n") else source + "\n"
    if camera_shots_json is None:
        return prefix
    if any(
        reserved in source
        for reserved in (
            _CAMERA_SHOTS_BEGIN,
            _CAMERA_SHOTS_END,
            "TIKZ_NATIVE_CAMERA_SHOTS",
            "_tikz_native_camera_shots_from_json",
        )
    ):
        raise SourceProjectBuildError(
            "Bridge generated source contains a reserved cameraShots binding"
        )
    block = (
        "\n"
        + _CAMERA_SHOTS_BEGIN
        + "\n"
        + "from tikz_native.parallel_shots import (\n"
        + "    parallel_camera_shot_sequence_from_json as "
        + "_tikz_native_camera_shots_from_json,\n"
        + ")\n"
        + "TIKZ_NATIVE_CAMERA_SHOTS = "
        + "_tikz_native_camera_shots_from_json("
        + project_literal(camera_shots_json)
        + ")\n"
        + "del _tikz_native_camera_shots_from_json\n"
        + _CAMERA_SHOTS_END
        + "\n"
    )
    return prefix.rstrip() + "\n" + block


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


_DIRECT_OPEN_FACE_MODULES = frozenset(
    {
        "polyhedron_visibility.open_faces",
        "polyhedron_visibility.open_faces.manim",
    }
)


def _dotted_python_name(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        prefix = _dotted_python_name(node.value)
        if prefix is not None:
            return (*prefix, node.attr)
    return None


def _imported_open_face_calls(
    tree: ast.Module,
) -> tuple[set[int], list[str]]:
    """Resolve supported constructor calls without trusting a matching spelling."""

    direct_names: set[str] = set()
    imported_constructor_names: set[str] = set()
    untrusted_constructor_names: set[str] = set()
    untrusted_module_prefixes: set[tuple[str, ...]] = set()
    module_prefixes: set[tuple[str, ...]] = set()
    full_package_roots: set[str] = set()
    module_root_origins: dict[str, set[str]] = {}
    import_bindings: dict[
        str, list[tuple[str, str, bool, bool, int]]
    ] = {}
    wildcard_modules: set[str] = set()
    dynamic_builtin_names = {"getattr", "setattr", "delattr"}
    dynamic_builtin_aliases: set[str] = set()
    namespace_builtin_names = {"globals", "locals", "vars"}
    namespace_builtin_aliases: set[str] = set()
    operator_attrgetter_aliases: set[str] = set()
    operator_module_aliases: set[str] = set()
    top_level_nodes = {id(node) for node in tree.body}

    def record(
        local_name: str,
        origin: str,
        *,
        kind: str,
        top_level: bool,
        plain_import: bool,
        level: int,
    ) -> None:
        import_bindings.setdefault(local_name, []).append(
            (origin, kind, top_level, plain_import, level)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    if node.level == 0:
                        wildcard_modules.add(module)
                    continue
                local_name = alias.asname or alias.name
                origin = f"{module}.{alias.name}" if module else alias.name
                record(
                    local_name,
                    origin,
                    kind="from",
                    top_level=id(node) in top_level_nodes,
                    plain_import=False,
                    level=node.level,
                )
                if node.level == 0 and module == "builtins":
                    if alias.name in dynamic_builtin_names:
                        dynamic_builtin_aliases.add(local_name)
                    if alias.name in namespace_builtin_names:
                        namespace_builtin_aliases.add(local_name)
                if (
                    node.level == 0
                    and module == "operator"
                    and alias.name == "attrgetter"
                ):
                    operator_attrgetter_aliases.add(local_name)
                if alias.name == "OpenFaceOcclusion3D":
                    imported_constructor_names.add(local_name)
                    if not (
                        id(node) in top_level_nodes
                        and node.level == 0
                        and module in _DIRECT_OPEN_FACE_MODULES
                    ):
                        untrusted_constructor_names.add(local_name)
                imported_module = f"{module}.{alias.name}" if module else alias.name
                if (
                    imported_module in _DIRECT_OPEN_FACE_MODULES
                    and (id(node) not in top_level_nodes or node.level != 0)
                ):
                    untrusted_module_prefixes.add((local_name,))
                if id(node) not in top_level_nodes or node.level != 0:
                    continue
                if (
                    module in _DIRECT_OPEN_FACE_MODULES
                    and alias.name == "OpenFaceOcclusion3D"
                ):
                    direct_names.add(local_name)
                if imported_module in _DIRECT_OPEN_FACE_MODULES:
                    module_prefixes.add((local_name,))
                    module_root_origins.setdefault(local_name, set()).add(
                        imported_module
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                record(
                    local_name,
                    alias.name,
                    kind="import",
                    top_level=id(node) in top_level_nodes,
                    plain_import=alias.asname is None,
                    level=0,
                )
                if alias.name == "operator":
                    operator_module_aliases.add(local_name)
                if id(node) not in top_level_nodes:
                    continue
                if alias.name in _DIRECT_OPEN_FACE_MODULES:
                    prefix = (
                        (alias.asname,)
                        if alias.asname
                        else tuple(alias.name.split("."))
                    )
                    module_prefixes.add(prefix)
                    if alias.asname is None:
                        full_package_roots.add(prefix[0])
                    else:
                        module_root_origins.setdefault(prefix[0], set()).add(
                            alias.name
                        )

    canonical_direct_origins = {
        f"{module}.OpenFaceOcclusion3D" for module in _DIRECT_OPEN_FACE_MODULES
    }
    for name in sorted(direct_names):
        for origin, kind, top_level, _, level in import_bindings.get(name, ()):
            if not (
                top_level
                and level == 0
                and kind == "from"
                and origin in canonical_direct_origins
            ):
                raise SourceProjectBuildError(
                    f"generated OpenFaceOcclusion3D binding {name!r} is ambiguous"
                )

    module_binding_roots = {prefix[0] for prefix in module_prefixes if prefix}
    for name in sorted(module_binding_roots):
        allowed_origins = module_root_origins.get(name, set())
        for origin, kind, top_level, plain_import, level in import_bindings.get(
            name, ()
        ):
            package_import = (
                name in full_package_roots
                and kind == "import"
                and plain_import
                and (origin == name or origin.startswith(f"{name}."))
            )
            explicit_module_import = origin in allowed_origins
            if not (
                top_level
                and level == 0
                and (package_import or explicit_module_import)
            ):
                raise SourceProjectBuildError(
                    f"generated OpenFaceOcclusion3D module binding {name!r} is ambiguous"
                )

    binding_roots = direct_names | module_binding_roots
    shadowed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node.id in binding_roots:
                shadowed.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            candidate = _dotted_python_name(node)
            if candidate in module_prefixes or (
                candidate is not None
                and candidate[-1] == "OpenFaceOcclusion3D"
                and candidate[:-1] in module_prefixes
            ):
                shadowed.add(".".join(candidate))
        elif isinstance(node, ast.arg) and node.arg in binding_roots:
            shadowed.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in binding_roots:
                shadowed.add(node.name)
        elif isinstance(node, ast.ExceptHandler):
            if node.name in binding_roots:
                shadowed.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name in binding_roots:
                shadowed.add(node.name)
        elif isinstance(node, ast.MatchMapping):
            if node.rest in binding_roots:
                shadowed.add(node.rest)
    if shadowed:
        raise SourceProjectBuildError(
            "generated OpenFaceOcclusion3D import binding is reassigned or shadowed: "
            + ", ".join(sorted(shadowed))
        )

    loaded_untrusted_bindings = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in untrusted_constructor_names
    }
    loaded_untrusted_modules = {
        ".".join(candidate)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Load)
        and (candidate := _dotted_python_name(node)) is not None
        and candidate[-1] == "OpenFaceOcclusion3D"
        and candidate[:-1] in untrusted_module_prefixes
    }
    if loaded_untrusted_bindings or loaded_untrusted_modules:
        raise SourceProjectBuildError(
            "generated OpenFaceOcclusion3D bindings must come from an "
            "absolute canonical import: "
            + ", ".join(
                sorted(loaded_untrusted_bindings | loaded_untrusted_modules)
            )
        )

    supported_calls: set[int] = set()
    unsupported_calls: list[str] = []
    dynamic_calls: list[str] = []
    controller_names: list[str] = []
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    def targets_constructor_name(
        node: ast.AST | None,
        *,
        names: set[str] | None = None,
    ) -> bool:
        target_names = {"OpenFaceOcclusion3D"}
        if names is not None:
            target_names.update(names)
        return not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value not in target_names
        )

    def is_dynamic_builtin(name: tuple[str, ...] | None) -> bool:
        return bool(
            name
            and (
                name[-1] in dynamic_builtin_names
                or len(name) == 1 and name[0] in dynamic_builtin_aliases
            )
        )

    def is_namespace_builtin(name: tuple[str, ...] | None) -> bool:
        return bool(
            name
            and (
                name[-1] in namespace_builtin_names
                or len(name) == 1 and name[0] in namespace_builtin_aliases
            )
        )

    def is_operator_attrgetter(name: tuple[str, ...] | None) -> bool:
        return bool(
            name
            and (
                len(name) == 1 and name[0] in operator_attrgetter_aliases
                or len(name) == 2
                and name[0] in operator_module_aliases
                and name[1] == "attrgetter"
                or name == ("operator", "attrgetter")
            )
        )

    # Once a canonical constructor/module binding exists, reject reflection and
    # runtime code evaluation outright. Following their data flow would turn a
    # narrow deterministic rewrite into an unsafe partial Python interpreter.
    forbidden_runtime_names = {
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "eval",
        "exec",
        "compile",
        "__import__",
    }
    forbidden_runtime_aliases = (
        dynamic_builtin_aliases | namespace_builtin_aliases
    )
    strict_reflection = bool(binding_roots)
    if strict_reflection:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {
                "__dict__",
                "__getattr__",
                "__getattribute__",
                "__setattr__",
                "__delattr__",
            }:
                dynamic_calls.append(node.attr)
            if isinstance(node, ast.Call):
                callable_name = _dotted_python_name(node.func)
                if is_operator_attrgetter(callable_name) and any(
                    targets_constructor_name(argument) for argument in node.args
                ):
                    dynamic_calls.append("operator.attrgetter")
            if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(
                node.ctx, ast.Load
            ):
                callable_name = _dotted_python_name(node)
                if callable_name and (
                    callable_name[-1] in forbidden_runtime_names
                    or len(callable_name) == 1
                    and callable_name[0] in forbidden_runtime_aliases
                ):
                    dynamic_calls.append(callable_name[-1])
    else:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callable_name = _dotted_python_name(node.func)
            if not callable_name or callable_name[-1] not in {
                "eval",
                "exec",
                "compile",
                "__import__",
            }:
                continue
            if any(
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and "OpenFaceOcclusion3D" in argument.value
                for argument in node.args
            ):
                dynamic_calls.append(callable_name[-1])

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        getter_name = _dotted_python_name(node.func)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {"__getattr__", "__getattribute__", "__setattr__", "__delattr__"}
        ):
            base_name = _dotted_python_name(node.func.value)
            requested_name = node.args[0] if node.args else None
            if (
                base_name in module_prefixes
                and targets_constructor_name(requested_name)
            ):
                dynamic_calls.append(node.func.attr)
        if (
            is_dynamic_builtin(getter_name)
            and node.args
        ):
            base_name = _dotted_python_name(node.args[0])
            requested_name = node.args[1] if len(node.args) >= 2 else None
            if (
                base_name in module_prefixes
                and targets_constructor_name(requested_name)
            ):
                dynamic_calls.append(getter_name[-1] if getter_name else "builtin")
        if is_namespace_builtin(getter_name) and node.args:
            base_name = _dotted_python_name(node.args[0])
            if base_name in module_prefixes:
                parent = parents.get(id(node))
                if not (
                    isinstance(parent, ast.Subscript)
                    and parent.value is node
                    and not targets_constructor_name(parent.slice)
                ):
                    dynamic_calls.append("vars")
        if is_operator_attrgetter(getter_name) and any(
            targets_constructor_name(argument) for argument in node.args
        ):
            parent = parents.get(id(node))
            if (
                isinstance(parent, ast.Call)
                and parent.func is node
                and any(
                    _dotted_python_name(argument) in module_prefixes
                    for argument in parent.args
                )
            ):
                dynamic_calls.append("operator.attrgetter")

        # Direct constructor aliases can otherwise be recovered from the
        # current global/local namespace without ever loading the imported
        # Name node. Treat only lookups that may target that binding as unsafe;
        # unrelated introspection remains allowed.
        if is_namespace_builtin(getter_name) and not node.args:
            parent = parents.get(id(node))
            requested_name: ast.AST | None = None
            if isinstance(parent, ast.Subscript) and parent.value is node:
                requested_name = parent.slice
            elif isinstance(parent, ast.Attribute) and parent.value is node:
                outer = parents.get(id(parent))
                if (
                    isinstance(outer, ast.Call)
                    and outer.func is parent
                    and parent.attr
                    in {"get", "pop", "setdefault", "__getitem__", "__setitem__"}
                ):
                    requested_name = outer.args[0] if outer.args else None
            if requested_name is not None and targets_constructor_name(
                requested_name,
                names=direct_names,
            ):
                dynamic_calls.append("namespace lookup")
        name = _dotted_python_name(node.func)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "OpenFaceOcclusion3D"
            and name is None
        ):
            unsupported_calls.append("dynamic.OpenFaceOcclusion3D")
            continue
        if name is None:
            continue
        supported = (
            len(name) == 1
            and name[0] in direct_names
            or len(name) >= 2
            and name[-1] == "OpenFaceOcclusion3D"
            and name[:-1] in module_prefixes
        )
        looks_like_constructor = (
            name[-1] == "OpenFaceOcclusion3D"
            or len(name) == 1
            and name[0] in imported_constructor_names
        )
        if supported:
            supported_calls.add(id(node))
        elif looks_like_constructor:
            unsupported_calls.append(".".join(name))

    if dynamic_calls:
        raise SourceProjectBuildError(
            "generated OpenFaceOcclusion3D source must call the imported "
            "constructor directly; dynamic module lookup or rebinding is not "
            "supported: "
            + ", ".join(sorted(set(dynamic_calls)))
        )
    if supported_calls and wildcard_modules.intersection(_DIRECT_OPEN_FACE_MODULES):
        raise SourceProjectBuildError(
            "generated OpenFaceOcclusion3D source must not use wildcard imports"
        )
    if unsupported_calls:
        raise SourceProjectBuildError(
            "generated OpenFaceOcclusion3D calls must be bound from "
            "polyhedron_visibility.open_faces or "
            "polyhedron_visibility.open_faces.manim: "
            + ", ".join(sorted(set(unsupported_calls)))
        )

    indirect_references: set[str] = set()
    for node in ast.walk(tree):
        name: tuple[str, ...] | None = None
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in direct_names:
                name = (node.id,)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            candidate = _dotted_python_name(node)
            if (
                candidate is not None
                and candidate[-1] == "OpenFaceOcclusion3D"
                and candidate[:-1] in module_prefixes
            ):
                name = candidate
        if name is None:
            continue
        parent = parents.get(id(node))
        if (
            isinstance(parent, ast.Call)
            and parent.func is node
            and id(parent) in supported_calls
        ):
            continue
        indirect_references.add(".".join(name))
    if indirect_references:
        raise SourceProjectBuildError(
            "generated OpenFaceOcclusion3D source must call the imported "
            "constructor directly; indirect constructor aliases are not supported: "
            + ", ".join(sorted(indirect_references))
        )

    for node in ast.walk(tree):
        value: ast.AST | None = None
        targets: Sequence[ast.AST] = ()
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign):
            value, targets = node.value, (node.target,)
        if not isinstance(value, ast.Call) or id(value) not in supported_calls:
            continue
        controller_names.extend(
            target.id for target in targets if isinstance(target, ast.Name)
        )
    return supported_calls, controller_names


class _DirectOpenFaceTransformer(ast.NodeTransformer):
    def __init__(
        self,
        *,
        paint_policy: str,
        painter_z_band: PainterZBand,
        whole_figure_targets: Sequence[str],
        controller_name: str | None,
        constructor_call_ids: set[int],
    ) -> None:
        self.paint_policy = paint_policy
        self.painter_z_band = painter_z_band
        self.targets = set(whole_figure_targets)
        self.controller_name = controller_name
        self.constructor_call_ids = constructor_call_ids
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if id(node) in self.constructor_call_ids:
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
    constructor_call_ids, controller_names = _imported_open_face_calls(tree)
    if not constructor_call_ids:
        if whole_figure_targets:
            raise SourceProjectBuildError(
                "wholeFigureTargets requires generated OpenFaceOcclusion3D source"
            )
        return source if not source or source.endswith("\n") else source + "\n"
    controller_name: str | None = None
    if whole_figure_targets:
        if len(constructor_call_ids) != 1 or len(controller_names) != 1:
            raise SourceProjectBuildError(
                "wholeFigureTargets requires exactly one directly assigned "
                "OpenFaceOcclusion3D controller"
            )
        controller_name = controller_names[0]
    transformer = _DirectOpenFaceTransformer(
        paint_policy=paint_policy,
        painter_z_band=painter_z_band,
        whole_figure_targets=whole_figure_targets,
        controller_name=controller_name,
        constructor_call_ids=constructor_call_ids,
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
    try:
        source_tree = ast.parse(source, filename="<generated_scene.py>")
    except SyntaxError as exc:
        raise SourceProjectBuildError(
            f"generated Python is invalid before unified rewrite: line {exc.lineno}: {exc.msg}"
        ) from exc
    has_v3_installer = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "install_open_face_visibility_3d"
        for node in source_tree.body
    )
    if has_v3_installer:
        try:
            adapter = importlib.import_module(
                "tikz_native.generated_open_face_visibility_3d"
            )
            rewrite_adapter = adapter.rewrite_legacy_open_face_source
            adapter_error = adapter.GeneratedOpenFaceVisibility3DError
        except (ImportError, AttributeError) as exc:
            raise SourceProjectBuildError(
                "generated v3 source requires the unified open-face source adapter"
            ) from exc
        try:
            rewritten = rewrite_adapter(
                source,
                paint_policy=paint_policy,
                preferred_painter_z_band=tuple(painter_z_band.as_list()),
                whole_figure_targets=whole_figure_targets,
            )
        except adapter_error as exc:
            raise SourceProjectBuildError(
                f"generated v3 source violates the unified adapter contract: {exc}"
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
    camera_shots_json: str | None,
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
    rewritten = _append_camera_shots_binding(rewritten, camera_shots_json)
    rewritten = _append_authored_hooks(rewritten, hooks_source)
    try:
        compile(rewritten, "<generated_scene.py>", "exec")
    except SyntaxError as exc:
        raise SourceProjectBuildError(
            f"generated Python with authored hooks is invalid: line {exc.lineno}: {exc.msg}"
        ) from exc
    return rewritten


def _certify_dandelin_source_project_boundary(
    project: SourceProject,
    *,
    source_text: str,
    custom_shape_asset_builder: bool,
) -> None:
    """Keep curved Dandelin diagrams on the certified static asset path.

    This check deliberately compiles the captured source text instead of
    trusting a possibly injected ShapeAsset builder.  It runs while planning,
    before a staged output directory can be allocated.
    """

    compiler = importlib.import_module("tikz_native.compiler")

    def has_dandelin_evidence(picture: Any) -> bool:
        return bool(
            getattr(picture, "dandelin_diagrams", {})
            or getattr(picture, "dandelin_constructions_3d", {})
            or any(
                "Dandelin" in str(finding)
                for finding in getattr(picture, "unsupported", ())
            )
        )

    try:
        document = compiler.compile_document(
            source_text=source_text,
            entry_macro=project.entry_macro,
        )
    except Exception as exc:
        # A custom builder is allowed to understand a non-native entry macro,
        # but it must not use that freedom to reinterpret compiler-supported
        # Dandelin semantics.  Recompile the complete authoritative snapshot
        # without entry selection: if that succeeds and contains no Dandelin
        # evidence, the custom builder keeps its existing extension point.
        if custom_shape_asset_builder and project.entry_macro:
            try:
                fallback_document = compiler.compile_document(
                    source_text=source_text,
                )
            except Exception:
                pass
            else:
                if not any(
                    has_dandelin_evidence(picture)
                    for picture in fallback_document.pictures
                ):
                    return
        raise SourceProjectBuildError(
            "cannot certify the authoritative TikZ source snapshot before "
            f"planning derived outputs: {exc}"
        ) from exc
    picture = next(
        (
            item
            for item in document.pictures
            if item.index == project.picture_index
        ),
        None,
    )
    if picture is None:
        if custom_shape_asset_builder:
            if not any(
                has_dandelin_evidence(candidate)
                for candidate in document.pictures
            ):
                return
        raise SourceProjectBuildError(
            "cannot certify the Dandelin static boundary because "
            f"picture_index {project.picture_index} is not available in the "
            "authoritative TikZ source snapshot"
        )
    dandelin_findings = tuple(
        str(finding)
        for finding in getattr(picture, "unsupported", ())
        if "Dandelin" in str(finding)
    )
    if dandelin_findings:
        raise SourceProjectBuildError(
            "the authoritative compiler could not certify the selected "
            "Dandelin picture: " + "; ".join(dandelin_findings)
        )
    if not getattr(picture, "dandelin_diagrams", {}):
        return

    unsupported: list[str] = []
    if project.paint_policy != "diagrammatic":
        unsupported.append("renderIntent.paintPolicy must be 'diagrammatic'")
    for field, value in (
        ("motionJson", project.motion_json),
        ("cameraShots", project.camera_shots),
        ("bridgeRequestTemplate", project.bridge_request_template),
        ("hooksSource", project.hooks_source),
    ):
        if value is not None:
            unsupported.append(f"{field} must be absent")
    if project.selection:
        unsupported.append("selection must be empty or absent")
    if unsupported:
        raise SourceProjectBuildError(
            "Dandelin TikZ diagrams are static, diagrammatic ShapeAssets and "
            "cannot enter Geometry Rig/source-v2/source-v3 authoring: "
            + "; ".join(unsupported)
        )


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
    _certify_dandelin_source_project_boundary(
        project,
        source_text=tikz_source,
        custom_shape_asset_builder=shape_asset_builder is not None,
    )
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
        try:
            return _canonical_json(compiled)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise SourceProjectBuildError(
                "the TikZ compiler returned a ShapeAsset that cannot be "
                f"serialized as canonical JSON: {exc}"
            ) from exc
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
            motion_value = _strict_json_loads(motion_source)
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

    camera_shots_sequence: ParallelCameraShotSequence | None = None
    camera_shots_json: str | None = None
    camera_shots_key: str | None = None
    if project.camera_shots is not None:
        camera_shots_source = snapshot.text(
            project.camera_shots,
            label="cameraShots JSON",
        )
        try:
            camera_shots_value = _strict_json_loads(camera_shots_source)
        except json.JSONDecodeError as exc:
            raise SourceProjectError(
                f"invalid cameraShots JSON {project.camera_shots}: "
                f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        except ValueError as exc:
            raise SourceProjectError(
                f"invalid cameraShots JSON {project.camera_shots}: {exc}"
            ) from exc
        try:
            camera_shots_sequence = parallel_camera_shot_sequence_from_dict(
                camera_shots_value
            )
        except (TypeError, ValueError) as exc:
            raise SourceProjectError(
                f"invalid cameraShots sequence {project.camera_shots}: {exc}"
            ) from exc
        camera_shots_json = canonical_parallel_camera_shot_sequence_json(
            camera_shots_sequence
        )
        camera_shots_revision = {
            name: revisions[name]
            for name in ("source_project_build", "parallel_camera_core")
        }
        camera_shots_key = _build_key(
            {
                "node": "camera_shots",
                "schemaVersion": PARALLEL_CAMERA_SHOT_SEQUENCE_SCHEMA,
                "cameraShotsSha256": _sha256_bytes(
                    snapshot.payloads[project.camera_shots]
                ),
                "componentRevisions": camera_shots_revision,
            }
        )

        def build_camera_shots(_context: _BuildContext) -> bytes:
            assert camera_shots_json is not None
            return camera_shots_json.encode("utf-8")

        plans.append(
            NodePlan(
                name="camera_shots",
                output_name="camera-shots.json",
                key=camera_shots_key,
                component_revisions=camera_shots_revision,
                build_payload=build_camera_shots,
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
    compositing_key_value: dict[str, Any] = {
        "node": "compositing",
        "schemaVersion": COMPOSITING_SCHEMA_VERSION,
        "shapeKey": shape_key,
        "motionKey": motion_key,
        "paintPolicy": project.paint_policy,
        "projectionSha256": projection_digest,
        "painterZBand": painter_z_band.as_list(),
        "componentRevisions": compositing_revision,
    }
    if camera_shots_key is not None:
        compositing_key_value["cameraShotsKey"] = camera_shots_key
    compositing_key = _build_key(compositing_key_value)
    def build_compositing(context: _BuildContext) -> bytes:
        value = {
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
        }
        if camera_shots_key is not None:
            value["cameraShotsAssetSha256"] = context.digests["camera_shots"]
        return _canonical_json(value)
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
            template_value = _strict_json_loads(template)
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
        generated_revision_names = [
            "source_project_build",
            "generated_open_face_visibility_3d",
        ]
        if camera_shots_sequence is not None:
            generated_revision_names.append("parallel_camera_core")
        generated_revision = {
            name: revisions[name] for name in generated_revision_names
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
                camera_shots_json=camera_shots_json,
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


def _decode_previous_manifest(payload: bytes) -> dict[str, Any] | None:
    try:
        value = _strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("schemaVersion") != BUILD_MANIFEST_SCHEMA_VERSION:
        return None
    nodes = value.get("nodes")
    if not isinstance(nodes, dict):
        return None
    return value


def _simple_entry_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise SourceProjectBuildError(f"unsafe derived output name: {name!r}")
    return name


def _regular_open_flags(*, write: bool = False) -> int:
    flags = os.O_WRONLY if write else os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if not write and hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _open_regular_at(directory_descriptor: int, name: str) -> int:
    _simple_entry_name(name)
    try:
        descriptor = os.open(
            name,
            _regular_open_flags(),
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise SourceProjectBuildError(
            f"cannot safely open derived output file {name!r}"
        ) from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise SourceProjectBuildError(
            f"derived output entry is not a regular file: {name!r}"
        )
    return descriptor


def _entry_identity_at(directory_descriptor: int, name: str) -> tuple[int, int]:
    entry = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    return entry.st_dev, entry.st_ino


def _unlink_matching_identity(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int],
) -> bool:
    """Quarantine a matching inode under a private name before unlinking it.

    A direct stat-then-unlink can delete a replacement inserted between those
    syscalls. Renaming first means a mismatching replacement is preserved under
    the quarantine name rather than deleted.
    """

    try:
        if _entry_identity_at(directory_descriptor, name) != identity:
            return False
        quarantine_name = f".{name}.cleanup-{uuid.uuid4().hex}"
        if not _rename_no_replace(
            directory_descriptor,
            name,
            quarantine_name,
        ):
            return False
        if _entry_identity_at(
            directory_descriptor,
            quarantine_name,
        ) != identity:
            return False
        os.unlink(quarantine_name, dir_fd=directory_descriptor)
        return True
    except FileNotFoundError:
        return False


def _read_regular_at(directory_descriptor: int, name: str) -> bytes:
    descriptor = _open_regular_at(directory_descriptor, name)
    try:
        return _read_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _sha256_regular_at(directory_descriptor: int, name: str) -> str:
    descriptor = _open_regular_at(directory_descriptor, name)
    try:
        return _sha256_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_at(directory_descriptor: int, name: str, payload: bytes) -> None:
    """Write one derived file without resolving a mutable directory path."""

    _simple_entry_name(name)
    temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = _regular_open_flags(write=True) | os.O_CREAT | os.O_EXCL
    descriptor: int | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        temporary_stat = os.fstat(descriptor)
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            published = _rename_no_replace(
                directory_descriptor,
                temporary_name,
                name,
            )
        except OSError as exc:
            raise SourceProjectBuildError(
                f"derived output name appeared concurrently: {name!r}"
            ) from exc
        if not published:
            raise SourceProjectBuildError(
                "this platform cannot publish a derived file without a race"
            )
        temporary_name = ""
        if _entry_identity_at(directory_descriptor, name) != temporary_identity:
            raise SourceProjectBuildError(
                f"derived output file changed concurrently: {name!r}"
            )
        os.fsync(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name and temporary_identity is not None:
            with contextlib.suppress(OSError):
                _unlink_matching_identity(
                    directory_descriptor,
                    temporary_name,
                    temporary_identity,
                )


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


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


@contextlib.contextmanager
def _output_parent_descriptor(
    project: SourceProject,
    project_root_descriptor: int,
    *,
    create: bool,
) -> Iterator[int | None]:
    """Open every output-parent component relative to the locked project fd.

    Opening the full path would only protect its final component with
    ``O_NOFOLLOW``.  Walking it with ``openat`` also prevents an intermediate
    directory from being exchanged for a symlink after the manifest was read.
    """

    relative_parent = project.output_directory.parent.relative_to(project.root)
    descriptor = os.dup(project_root_descriptor)
    try:
        for raw_part in relative_parent.parts:
            part = _simple_entry_name(raw_part)
            try:
                child = os.open(
                    part,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    os.close(descriptor)
                    descriptor = -1
                    yield None
                    return
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                try:
                    child = os.open(
                        part,
                        _directory_open_flags(),
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise SourceProjectBuildError(
                        "cannot create a safe parent for the derived output"
                    ) from exc
            except OSError as exc:
                raise SourceProjectBuildError(
                    "refusing to use a derived output through a symlink or unsafe parent"
                ) from exc
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_output_parent_identity(
    project: SourceProject,
    project_root_descriptor: int,
    output_parent_descriptor: int,
) -> None:
    """Confirm the authored path still resolves to the held output parent."""

    held_root = os.fstat(project_root_descriptor)
    relative_parent = project.output_directory.parent.relative_to(project.root)

    def resolve_once() -> None:
        descriptors: list[int] = []
        links: list[tuple[int, str, int]] = []
        try:
            try:
                descriptor = os.open(project.root, _directory_open_flags())
            except OSError as exc:
                raise SourceProjectBuildError(
                    "source project directory changed concurrently"
                ) from exc
            descriptors.append(descriptor)
            named_root = os.fstat(descriptor)
            if (
                named_root.st_dev != held_root.st_dev
                or named_root.st_ino != held_root.st_ino
            ):
                raise SourceProjectBuildError(
                    "source project directory changed concurrently"
                )

            for raw_part in relative_parent.parts:
                part = _simple_entry_name(raw_part)
                try:
                    child = os.open(
                        part,
                        _directory_open_flags(),
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise SourceProjectBuildError(
                        "derived output parent changed concurrently"
                    ) from exc
                descriptors.append(child)
                links.append((descriptor, part, child))
                descriptor = child

            # Recheck every named edge after the complete walk. This catches a
            # parent renamed after its descriptor was opened but before a
            # deeper component was resolved.
            for parent, part, child in links:
                try:
                    named = os.stat(part, dir_fd=parent, follow_symlinks=False)
                except OSError as exc:
                    raise SourceProjectBuildError(
                        "derived output parent changed concurrently"
                    ) from exc
                held_child = os.fstat(child)
                if (
                    not stat.S_ISDIR(named.st_mode)
                    or named.st_dev != held_child.st_dev
                    or named.st_ino != held_child.st_ino
                ):
                    raise SourceProjectBuildError(
                        "derived output parent changed concurrently"
                    )

            current = os.fstat(descriptor)
            held = os.fstat(output_parent_descriptor)
            if current.st_dev != held.st_dev or current.st_ino != held.st_ino:
                raise SourceProjectBuildError(
                    "derived output parent changed concurrently"
                )
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    # A second independent resolution prevents a swap performed during the
    # first descriptor walk from being mistaken for the still-current path.
    resolve_once()
    resolve_once()


def _require_output_absent(
    project: SourceProject,
    project_root_descriptor: int,
    output_parent_descriptor: int | None,
    *,
    message: str,
) -> None:
    """Revalidate both the authored path and the absence of its output name."""

    if output_parent_descriptor is not None:
        for _ in range(2):
            _require_output_parent_identity(
                project,
                project_root_descriptor,
                output_parent_descriptor,
            )
            if _entry_exists_at(
                output_parent_descriptor,
                project.output_directory.name,
            ):
                raise SourceProjectBuildError(message)
        return

    held_root = os.fstat(project_root_descriptor)
    for _ in range(2):
        try:
            current_root_descriptor = os.open(
                project.root,
                _directory_open_flags(),
            )
        except OSError as exc:
            raise SourceProjectBuildError(
                "source project directory changed concurrently"
            ) from exc
        try:
            current_root = os.fstat(current_root_descriptor)
            if (
                current_root.st_dev != held_root.st_dev
                or current_root.st_ino != held_root.st_ino
            ):
                raise SourceProjectBuildError(
                    "source project directory changed concurrently"
                )
        finally:
            os.close(current_root_descriptor)
        if os.path.lexists(project.output_directory):
            raise SourceProjectBuildError(message)


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _validate_owned_directory_descriptor(
    project: SourceProject,
    descriptor: int,
    *,
    allow_empty: bool = False,
    expected_entries: set[str] | None = None,
) -> set[str]:
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        raise SourceProjectBuildError("derived output path is not a directory")
    entries = set(os.listdir(descriptor))
    if not entries and allow_empty:
        return entries
    marker_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        marker_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        marker_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        marker_flags |= os.O_NONBLOCK
    try:
        marker_descriptor = os.open(
            OWNED_OUTPUT_MARKER,
            marker_flags,
            dir_fd=descriptor,
        )
    except OSError as exc:
        raise SourceProjectBuildError(
            "refusing to replace or clean an output directory without its ownership marker"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(marker_descriptor).st_mode):
            raise SourceProjectBuildError(
                "refusing to replace or clean an output directory without its ownership marker"
            )
        marker_payload = _read_descriptor(marker_descriptor)
    finally:
        os.close(marker_descriptor)
    if marker_payload != _owned_output_payload(project):
        raise SourceProjectBuildError("derived output ownership marker does not match this project")
    unknown = sorted(entries - _KNOWN_OUTPUT_NAMES)
    if unknown:
        raise SourceProjectBuildError(
            "refusing to replace or clean output containing unowned entries: "
            + ", ".join(unknown)
        )
    if expected_entries is not None and entries != expected_entries:
        missing = sorted(expected_entries - entries)
        extra = sorted(entries - expected_entries)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise SourceProjectBuildError(
            "derived output file set does not match this build: " + "; ".join(details)
        )
    invalid_generated: list[str] = []
    for name in entries - {OWNED_OUTPUT_MARKER}:
        try:
            entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError:
            invalid_generated.append(name)
            continue
        if not stat.S_ISREG(entry.st_mode):
            invalid_generated.append(name)
    invalid_generated.sort()
    if invalid_generated:
        raise SourceProjectBuildError(
            "refusing to replace or clean a derived output whose generated "
            "entries are not regular files: " + ", ".join(invalid_generated)
        )
    return entries


def _require_directory_entries_unchanged(
    descriptor: int,
    expected_entries: set[str],
    *,
    label: str,
) -> None:
    if set(os.listdir(descriptor)) != expected_entries:
        raise SourceProjectBuildError(f"{label} changed concurrently")


def _open_owned_output_descriptor(
    project: SourceProject,
    parent_descriptor: int,
    *,
    allow_absent: bool,
    allow_empty: bool = False,
) -> tuple[int, set[str]] | None:
    output_name = _simple_entry_name(project.output_directory.name)
    try:
        descriptor = os.open(
            output_name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except FileNotFoundError:
        if allow_absent:
            return None
        raise SourceProjectBuildError("derived output directory does not exist")
    except OSError as exc:
        raise SourceProjectBuildError(
            "refusing to use an unsafe derived output directory"
        ) from exc
    try:
        entries = _validate_owned_directory_descriptor(
            project,
            descriptor,
            allow_empty=allow_empty,
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, entries


@contextlib.contextmanager
def _project_lock(project: SourceProject) -> Iterator[int]:
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
        yield descriptor
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _entry_exists_at(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _move_to_unique_sibling(
    parent_descriptor: int,
    source_name: str,
    *,
    prefix: str,
) -> str:
    for _ in range(16):
        destination_name = f".{prefix}-{uuid.uuid4().hex}"
        try:
            moved = _rename_no_replace(
                parent_descriptor,
                source_name,
                destination_name,
            )
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                continue
            raise
        if not moved:
            raise SourceProjectBuildError(
                "this platform cannot isolate a derived output without a race"
            )
        return destination_name
    raise SourceProjectBuildError("cannot allocate a unique recovery name")


def _named_directory_matches(
    parent_descriptor: int,
    name: str,
    descriptor: int,
) -> bool:
    try:
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        return False
    held = os.fstat(descriptor)
    return (
        stat.S_ISDIR(named.st_mode)
        and named.st_dev == held.st_dev
        and named.st_ino == held.st_ino
    )


def _names_matching_directory_descriptor(
    parent_descriptor: int,
    descriptor: int,
) -> list[str]:
    """Return the sibling names that still refer to one held directory."""

    matches: list[str] = []
    for name in os.listdir(parent_descriptor):
        if _named_directory_matches(parent_descriptor, name, descriptor):
            matches.append(name)
    return sorted(matches)


def _require_named_directory_identity(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    *,
    label: str,
) -> None:
    if not _named_directory_matches(parent_descriptor, name, descriptor):
        raise SourceProjectBuildError(f"{label} changed concurrently")


def _restore_detached_output(
    parent_descriptor: int,
    *,
    detached_name: str,
    output_name: str,
) -> str | None:
    # Restore the exact directory entry that this operation isolated.  Do not
    # require it to retain a previously observed inode here: if another writer
    # exchanged the name at the validation boundary, moving that entry back is
    # the only non-destructive way to preserve the concurrent writer's data.
    concurrent_name: str | None = None
    if _entry_exists_at(parent_descriptor, output_name):
        concurrent_name = _move_to_unique_sibling(
            parent_descriptor,
            output_name,
            prefix=f"{output_name}.concurrent",
        )
    try:
        restored = _rename_no_replace(
            parent_descriptor,
            detached_name,
            output_name,
        )
    except OSError as exc:
        restored = False
        restore_error: BaseException = exc
    else:
        restore_error = SourceProjectBuildError(
            "this platform cannot restore a derived output without a race"
        )
    if not restored:
        # If the destination was isolated successfully but restoring the old
        # directory failed, put the displaced destination back when possible.
        # Never leave the visible name empty merely because the recovery
        # primitive failed; both directories remain explicitly named if even
        # this best-effort step cannot complete.
        displaced_restored = False
        if (
            concurrent_name is not None
            and not _entry_exists_at(parent_descriptor, output_name)
        ):
            try:
                displaced_restored = _rename_no_replace(
                    parent_descriptor,
                    concurrent_name,
                    output_name,
                )
            except OSError:
                displaced_restored = False
            if displaced_restored:
                concurrent_name = None
        details = [f"previous directory is preserved as {detached_name!r}"]
        if concurrent_name is not None:
            details.append(
                f"displaced visible directory is preserved as {concurrent_name!r}"
            )
        elif displaced_restored:
            details.append("the displaced visible directory was restored")
        raise SourceProjectBuildError(
            "could not restore the previous derived output; " + "; ".join(details)
        ) from restore_error
    return concurrent_name


def _detach_and_validate_owned_output(
    project: SourceProject,
    parent_descriptor: int,
    *,
    purpose: str,
    allow_absent: bool,
    allow_empty: bool = False,
) -> tuple[str, int] | None:
    output_name = project.output_directory.name
    opened = _open_owned_output_descriptor(
        project,
        parent_descriptor,
        allow_absent=allow_absent,
        allow_empty=allow_empty,
    )
    if opened is None:
        return None
    descriptor, _ = opened
    try:
        _require_named_directory_identity(
            parent_descriptor,
            output_name,
            descriptor,
            label="derived output",
        )
    except BaseException:
        os.close(descriptor)
        raise
    try:
        detached_name = _move_to_unique_sibling(
            parent_descriptor,
            output_name,
            prefix=f"{output_name}.{purpose}",
        )
    except FileNotFoundError:
        os.close(descriptor)
        raise SourceProjectBuildError("derived output changed concurrently")
    except BaseException as exc:
        os.close(descriptor)
        if isinstance(exc, SourceProjectBuildError):
            raise
        raise SourceProjectBuildError(
            "cannot atomically isolate the derived output directory"
        ) from exc

    try:
        _require_named_directory_identity(
            parent_descriptor,
            detached_name,
            descriptor,
            label="derived output",
        )
        _validate_owned_directory_descriptor(
            project,
            descriptor,
            allow_empty=allow_empty,
        )
    except BaseException as exc:
        old_names = _names_matching_directory_descriptor(
            parent_descriptor,
            descriptor,
        )
        if output_name in old_names:
            concurrent_name = None
        elif old_names:
            old_name = detached_name if detached_name in old_names else old_names[0]
            try:
                concurrent_name = _restore_detached_output(
                    parent_descriptor,
                    detached_name=old_name,
                    output_name=output_name,
                )
                _require_named_directory_identity(
                    parent_descriptor,
                    output_name,
                    descriptor,
                    label="restored derived output",
                )
            except BaseException as restore_exc:
                preserved = _names_matching_directory_descriptor(
                    parent_descriptor,
                    descriptor,
                )
                location = (
                    ", ".join(repr(name) for name in preserved)
                    or "an unnamed held directory"
                )
                os.close(descriptor)
                raise SourceProjectBuildError(
                    "derived output validation failed and the previous output "
                    f"could not be restored; it is preserved as {location}"
                ) from restore_exc
        else:
            os.close(descriptor)
            raise SourceProjectBuildError(
                "derived output validation failed and the previous output "
                "could not be located under its output parent"
            ) from exc

        preserved_replacements: list[str] = []
        if (
            _entry_exists_at(parent_descriptor, detached_name)
            and not _named_directory_matches(
                parent_descriptor,
                detached_name,
                descriptor,
            )
        ):
            preserved_replacements.append(detached_name)
        if concurrent_name is not None:
            preserved_replacements.append(concurrent_name)
        os.close(descriptor)
        if preserved_replacements:
            locations = ", ".join(repr(name) for name in preserved_replacements)
            raise SourceProjectBuildError(
                "derived output changed concurrently; the original output was "
                f"restored and concurrent replacements were preserved as {locations}"
            ) from exc
        if isinstance(exc, SourceProjectBuildError):
            raise
        raise SourceProjectBuildError(
            "refusing to replace or clean an unsafe derived output directory"
        ) from exc
    return detached_name, descriptor


def _rollback_removal_exchange(
    parent_descriptor: int,
    *,
    detached_name: str,
    discard_name: str,
    detached_descriptor: int,
    discard_descriptor: int,
) -> None:
    detached_is_empty = _named_directory_matches(
        parent_descriptor,
        detached_name,
        discard_descriptor,
    )
    discard_is_old = _named_directory_matches(
        parent_descriptor,
        discard_name,
        detached_descriptor,
    )
    if detached_is_empty and discard_is_old:
        try:
            exchanged = _rename_exchange(
                parent_descriptor,
                detached_name,
                discard_name,
            )
        except OSError:
            exchanged = False
        if exchanged and _named_directory_matches(
            parent_descriptor,
            detached_name,
            detached_descriptor,
        ):
            return

    old_names = _names_matching_directory_descriptor(
        parent_descriptor,
        detached_descriptor,
    )
    if detached_name in old_names:
        return
    if not old_names:
        raise SourceProjectBuildError(
            "clean rollback could not locate the previous derived output"
        )
    old_name = discard_name if discard_name in old_names else old_names[0]
    try:
        displaced_name = _restore_detached_output(
            parent_descriptor,
            detached_name=old_name,
            output_name=detached_name,
        )
        _require_named_directory_identity(
            parent_descriptor,
            detached_name,
            detached_descriptor,
            label="restored derived output",
        )
    except BaseException as exc:
        preserved = _names_matching_directory_descriptor(
            parent_descriptor,
            detached_descriptor,
        )
        location = ", ".join(repr(name) for name in preserved) or "an unnamed held directory"
        raise SourceProjectBuildError(
            "clean rollback could not restore the previous derived output; "
            f"it is preserved as {location}"
        ) from exc

    if (
        displaced_name is not None
        and _named_directory_matches(
            parent_descriptor,
            displaced_name,
            discard_descriptor,
        )
        and not os.listdir(discard_descriptor)
    ):
        with contextlib.suppress(OSError):
            os.rmdir(displaced_name, dir_fd=parent_descriptor)


def _remove_detached_owned_output(
    project: SourceProject,
    parent_descriptor: int,
    detached_name: str,
    detached_descriptor: int,
    *,
    allow_empty: bool = False,
    final_validator: Callable[[], None] | None = None,
) -> None:
    _require_named_directory_identity(
        parent_descriptor,
        detached_name,
        detached_descriptor,
        label="isolated derived output",
    )
    entries = _validate_owned_directory_descriptor(
        project,
        detached_descriptor,
        allow_empty=allow_empty,
    )
    _require_directory_entries_unchanged(
        detached_descriptor,
        entries,
        label="isolated derived output",
    )
    identities = {
        name: (
            entry.st_dev,
            entry.st_ino,
        )
        for name in entries
        for entry in (
            os.stat(
                name,
                dir_fd=detached_descriptor,
                follow_symlinks=False,
            ),
        )
    }

    # Exchange the complete old output with one empty directory.  The old
    # files remain an intact unit until rmdir(detached_name) succeeds, which is
    # the sole commit point for clean/removal.  This avoids the former partial
    # deletion when a concurrent unknown file appeared mid-clean.
    discard_name, discard_descriptor = _create_empty_transaction_directory(
        parent_descriptor,
        prefix=f"{detached_name}.discard",
    )
    exchanged = False
    committed = False
    rollback_failed = False
    try:
        _require_named_directory_identity(
            parent_descriptor,
            detached_name,
            detached_descriptor,
            label="isolated derived output",
        )
        _require_directory_entries_unchanged(
            detached_descriptor,
            entries,
            label="isolated derived output",
        )
        if final_validator is not None:
            final_validator()
        exchanged = _rename_exchange(
            parent_descriptor,
            detached_name,
            discard_name,
        )
        if not exchanged:
            raise SourceProjectBuildError(
                "this platform cannot atomically remove a derived output"
            )
        _require_named_directory_identity(
            parent_descriptor,
            detached_name,
            discard_descriptor,
            label="empty clean transaction directory",
        )
        _require_named_directory_identity(
            parent_descriptor,
            discard_name,
            detached_descriptor,
            label="isolated derived output",
        )
        if os.listdir(discard_descriptor):
            raise SourceProjectBuildError(
                "empty clean transaction directory changed concurrently"
            )
        if set(os.listdir(detached_descriptor)) != entries:
            raise SourceProjectBuildError(
                "isolated derived output changed concurrently"
            )
        if final_validator is not None:
            final_validator()
        os.rmdir(detached_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        committed = True
    except BaseException:
        if exchanged and not committed:
            try:
                _rollback_removal_exchange(
                    parent_descriptor,
                    detached_name=detached_name,
                    discard_name=discard_name,
                    detached_descriptor=detached_descriptor,
                    discard_descriptor=discard_descriptor,
                )
            except BaseException:
                rollback_failed = True
                raise
        raise
    finally:
        if not committed:
            if not rollback_failed:
                # Remove only a still-empty transaction directory.  Any changed
                # directory is preserved for diagnosis/recovery.
                for name, descriptor in (
                    (discard_name, discard_descriptor),
                    (detached_name, discard_descriptor),
                ):
                    if (
                        _named_directory_matches(
                            parent_descriptor,
                            name,
                            descriptor,
                        )
                        and not os.listdir(descriptor)
                    ):
                        with contextlib.suppress(OSError):
                            os.rmdir(name, dir_fd=parent_descriptor)
                        break
            os.close(discard_descriptor)

    # Removal is committed.  Cleanup of the now-disposable old output is
    # deliberately best-effort and can no longer change the command result.
    try:
        _require_named_directory_identity(
            parent_descriptor,
            discard_name,
            detached_descriptor,
            label="discarded derived output",
        )
        if set(os.listdir(detached_descriptor)) != entries:
            return
        for name, identity in identities.items():
            current = os.stat(
                name,
                dir_fd=detached_descriptor,
                follow_symlinks=False,
            )
            if (current.st_dev, current.st_ino) != identity:
                return
        ordered = sorted(entries - {OWNED_OUTPUT_MARKER})
        if OWNED_OUTPUT_MARKER in entries:
            ordered.append(OWNED_OUTPUT_MARKER)
        for name in ordered:
            if not _unlink_matching_identity(
                detached_descriptor,
                name,
                identities[name],
            ):
                return
        if not os.listdir(detached_descriptor) and _named_directory_matches(
            parent_descriptor,
            discard_name,
            detached_descriptor,
        ):
            os.rmdir(discard_name, dir_fd=parent_descriptor)
    except (OSError, SourceProjectBuildError):
        return
    finally:
        os.close(discard_descriptor)


def _create_empty_transaction_directory(
    parent_descriptor: int,
    *,
    prefix: str,
) -> tuple[str, int]:
    for _ in range(16):
        name = f".{prefix}-{uuid.uuid4().hex}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(
                name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
        except BaseException:
            with contextlib.suppress(OSError):
                os.rmdir(name, dir_fd=parent_descriptor)
            raise
        return name, descriptor
    raise SourceProjectBuildError(
        "cannot allocate an empty derived-output transaction directory"
    )


def _create_staged_directory(
    parent_descriptor: int,
    output_name: str,
) -> tuple[str, int]:
    for _ in range(16):
        stage_name = f".{output_name}.stage-{uuid.uuid4().hex}"
        try:
            os.mkdir(stage_name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(
                stage_name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
        except BaseException:
            with contextlib.suppress(OSError):
                os.rmdir(stage_name, dir_fd=parent_descriptor)
            raise
        return stage_name, descriptor
    raise SourceProjectBuildError("cannot allocate a unique staged output directory")


def _cleanup_partial_stage(
    parent_descriptor: int,
    stage_name: str,
    stage_descriptor: int,
) -> None:
    """Best-effort cleanup that never follows or recursively deletes a path."""

    if not _named_directory_matches(parent_descriptor, stage_name, stage_descriptor):
        return
    entries = set(os.listdir(stage_descriptor))
    if entries - _KNOWN_OUTPUT_NAMES:
        return
    identities: dict[str, tuple[int, int]] = {}
    for name in sorted(entries):
        try:
            entry = os.stat(name, dir_fd=stage_descriptor, follow_symlinks=False)
        except OSError:
            return
        if stat.S_ISDIR(entry.st_mode):
            return
        identities[name] = (entry.st_dev, entry.st_ino)
    for name in sorted(entries):
        if not _unlink_matching_identity(
            stage_descriptor,
            name,
            identities[name],
        ):
            return
    if _named_directory_matches(parent_descriptor, stage_name, stage_descriptor):
        with contextlib.suppress(OSError):
            os.rmdir(stage_name, dir_fd=parent_descriptor)


def _rename_exchange(
    parent_descriptor: int,
    left_name: str,
    right_name: str,
) -> bool:
    """Atomically exchange sibling names where the host kernel supports it."""

    library = ctypes.CDLL(None, use_errno=True)
    encoded_left = os.fsencode(left_name)
    encoded_right = os.fsencode(right_name)
    function: Any
    flags: int
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flags = 0x00000002  # RENAME_SWAP
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        flags = 0x00000002  # RENAME_EXCHANGE
    else:
        return False
    if function is None:
        return False
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(parent_descriptor, encoded_left, parent_descriptor, encoded_right, flags) == 0:
        return True
    error_number = ctypes.get_errno()
    unsupported = {errno.ENOSYS, errno.EINVAL}
    for candidate in (getattr(errno, "ENOTSUP", None), getattr(errno, "EOPNOTSUPP", None)):
        if candidate is not None:
            unsupported.add(candidate)
    if error_number in unsupported:
        return False
    raise OSError(error_number, os.strerror(error_number))


def _rename_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> bool:
    """Atomically rename only when the destination name is still absent."""

    library = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source_name)
    encoded_destination = os.fsencode(destination_name)
    function: Any
    flags: int
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flags = 0x00000004  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        flags = 0x00000001  # RENAME_NOREPLACE
    else:
        return False
    if function is None:
        return False
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    if function(
        parent_descriptor,
        encoded_source,
        parent_descriptor,
        encoded_destination,
        flags,
    ) == 0:
        return True
    error_number = ctypes.get_errno()
    unsupported = {errno.ENOSYS, errno.EINVAL}
    for candidate in (
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    ):
        if candidate is not None:
            unsupported.add(candidate)
    if error_number in unsupported:
        return False
    raise OSError(error_number, os.strerror(error_number))


def _validate_staged_artifacts(
    project: SourceProject,
    stage_descriptor: int,
    *,
    nodes: Sequence[NodeState],
    manifest_payload: bytes,
) -> set[str]:
    expected_entries = {
        *(node.output for node in nodes),
        BUILD_MANIFEST_NAME,
        OWNED_OUTPUT_MARKER,
    }
    entries = _validate_owned_directory_descriptor(
        project,
        stage_descriptor,
        expected_entries=expected_entries,
    )
    for node in nodes:
        if _sha256_regular_at(stage_descriptor, node.output) != node.sha256:
            raise SourceProjectBuildError(
                f"staged derived output changed while it was built: {node.output}"
            )
    if _read_regular_at(stage_descriptor, BUILD_MANIFEST_NAME) != manifest_payload:
        raise SourceProjectBuildError("staged build manifest changed before publication")
    _require_directory_entries_unchanged(
        stage_descriptor,
        entries,
        label="staged output",
    )
    return entries


def _rollback_portable_publication(
    parent_descriptor: int,
    *,
    output_name: str,
    stage_descriptor: int,
    rollback_name: str,
    old_descriptor: int,
) -> None:
    old_names = _names_matching_directory_descriptor(
        parent_descriptor,
        old_descriptor,
    )
    if output_name in old_names:
        return
    if not old_names:
        raise SourceProjectBuildError(
            "portable publication rollback could not locate the previous derived output"
        )
    old_name = rollback_name if rollback_name in old_names else old_names[0]
    try:
        displaced_name = _restore_detached_output(
            parent_descriptor,
            detached_name=old_name,
            output_name=output_name,
        )
        _require_named_directory_identity(
            parent_descriptor,
            output_name,
            old_descriptor,
            label="restored derived output",
        )
    except BaseException as exc:
        preserved = _names_matching_directory_descriptor(
            parent_descriptor,
            old_descriptor,
        )
        location = ", ".join(repr(name) for name in preserved) or "an unnamed held directory"
        raise SourceProjectBuildError(
            "portable publication rollback could not restore the previous derived output; "
            f"it is preserved as {location}"
        ) from exc

    if displaced_name is not None and _named_directory_matches(
        parent_descriptor,
        displaced_name,
        stage_descriptor,
    ):
        _cleanup_partial_stage(
            parent_descriptor,
            displaced_name,
            stage_descriptor,
        )


def _rollback_exchange_publication(
    parent_descriptor: int,
    *,
    output_name: str,
    stage_name: str,
    stage_descriptor: int,
    old_descriptor: int,
) -> None:
    """Reverse a successful publication or fail with an exact recovery name."""

    output_is_new = _named_directory_matches(
        parent_descriptor,
        output_name,
        stage_descriptor,
    )
    stage_is_old = _named_directory_matches(
        parent_descriptor,
        stage_name,
        old_descriptor,
    )
    if output_is_new and stage_is_old:
        try:
            exchanged = _rename_exchange(
                parent_descriptor,
                stage_name,
                output_name,
            )
        except OSError:
            exchanged = False
        if exchanged and _named_directory_matches(
            parent_descriptor,
            output_name,
            old_descriptor,
        ):
            return

    old_names = _names_matching_directory_descriptor(
        parent_descriptor,
        old_descriptor,
    )
    if output_name in old_names:
        return
    if not old_names:
        raise SourceProjectBuildError(
            "publication rollback could not locate the previous derived output"
        )
    old_name = stage_name if stage_name in old_names else old_names[0]
    try:
        displaced_name = _restore_detached_output(
            parent_descriptor,
            detached_name=old_name,
            output_name=output_name,
        )
        _require_named_directory_identity(
            parent_descriptor,
            output_name,
            old_descriptor,
            label="restored derived output",
        )
    except BaseException as exc:
        preserved = _names_matching_directory_descriptor(
            parent_descriptor,
            old_descriptor,
        )
        location = ", ".join(repr(name) for name in preserved) or "an unnamed held directory"
        raise SourceProjectBuildError(
            "publication rollback could not restore the previous derived output; "
            f"it is preserved as {location}"
        ) from exc

    if displaced_name is not None and _named_directory_matches(
        parent_descriptor,
        displaced_name,
        stage_descriptor,
    ):
        _cleanup_partial_stage(
            parent_descriptor,
            displaced_name,
            stage_descriptor,
        )


def _publish_staged_directory(
    project: SourceProject,
    parent_descriptor: int,
    stage_name: str,
    stage_descriptor: int,
    old_descriptor: int | None,
    *,
    nodes: Sequence[NodeState],
    manifest_payload: bytes,
    final_validator: Callable[[], None] | None = None,
) -> None:
    output_name = project.output_directory.name
    _require_named_directory_identity(
        parent_descriptor,
        stage_name,
        stage_descriptor,
        label="staged output",
    )
    _validate_staged_artifacts(
        project,
        stage_descriptor,
        nodes=nodes,
        manifest_payload=manifest_payload,
    )

    if old_descriptor is None:
        try:
            published = _rename_no_replace(
                parent_descriptor,
                stage_name,
                output_name,
            )
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise SourceProjectBuildError(
                    "derived output appeared concurrently"
                ) from exc
            raise SourceProjectBuildError("cannot publish the staged output") from exc
        if not published:
            raise SourceProjectBuildError(
                "this platform cannot publish a new derived output without a race"
            )
        try:
            _require_named_directory_identity(
                parent_descriptor,
                output_name,
                stage_descriptor,
                label="published output",
            )
            _validate_staged_artifacts(
                project,
                stage_descriptor,
                nodes=nodes,
                manifest_payload=manifest_payload,
            )
            if final_validator is not None:
                final_validator()
            os.fsync(parent_descriptor)
            _require_named_directory_identity(
                parent_descriptor,
                output_name,
                stage_descriptor,
                label="published output",
            )
            if final_validator is not None:
                final_validator()
            _validate_staged_artifacts(
                project,
                stage_descriptor,
                nodes=nodes,
                manifest_payload=manifest_payload,
            )
        except BaseException:
            if _entry_exists_at(parent_descriptor, output_name):
                if not _named_directory_matches(
                    parent_descriptor,
                    output_name,
                    stage_descriptor,
                ):
                    _move_to_unique_sibling(
                        parent_descriptor,
                        output_name,
                        prefix=f"{output_name}.recovery",
                    )
                else:
                    try:
                        recovered = _rename_no_replace(
                            parent_descriptor,
                            output_name,
                            stage_name,
                        )
                    except OSError as recovery_exc:
                        if recovery_exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                            raise
                        _move_to_unique_sibling(
                            parent_descriptor,
                            output_name,
                            prefix=f"{output_name}.recovery",
                        )
                    else:
                        if not recovered:
                            raise SourceProjectBuildError(
                                "this platform cannot recover a staged output without a race"
                            )
            raise
        return

    _require_named_directory_identity(
        parent_descriptor,
        output_name,
        old_descriptor,
        label="derived output",
    )
    old_entries = _validate_owned_directory_descriptor(
        project,
        old_descriptor,
        allow_empty=True,
    )
    _require_directory_entries_unchanged(
        old_descriptor,
        old_entries,
        label="derived output",
    )

    try:
        exchanged = _rename_exchange(parent_descriptor, stage_name, output_name)
    except OSError as exc:
        raise SourceProjectBuildError("cannot atomically publish the staged output") from exc
    if exchanged:
        try:
            _require_named_directory_identity(
                parent_descriptor,
                output_name,
                stage_descriptor,
                label="published output",
            )
            _require_named_directory_identity(
                parent_descriptor,
                stage_name,
                old_descriptor,
                label="previous derived output",
            )
            _validate_staged_artifacts(
                project,
                stage_descriptor,
                nodes=nodes,
                manifest_payload=manifest_payload,
            )
            current_old_entries = _validate_owned_directory_descriptor(
                project,
                old_descriptor,
                allow_empty=True,
            )
            if current_old_entries != old_entries:
                raise SourceProjectBuildError(
                    "previous derived output changed during publication"
                )
            _require_directory_entries_unchanged(
                old_descriptor,
                old_entries,
                label="previous derived output",
            )
            if final_validator is not None:
                final_validator()
            os.fsync(parent_descriptor)
            _require_named_directory_identity(
                parent_descriptor,
                output_name,
                stage_descriptor,
                label="published output",
            )
            _require_named_directory_identity(
                parent_descriptor,
                stage_name,
                old_descriptor,
                label="previous derived output",
            )
            if final_validator is not None:
                final_validator()
            _validate_staged_artifacts(
                project,
                stage_descriptor,
                nodes=nodes,
                manifest_payload=manifest_payload,
            )
        except BaseException:
            _rollback_exchange_publication(
                parent_descriptor,
                output_name=output_name,
                stage_name=stage_name,
                stage_descriptor=stage_descriptor,
                old_descriptor=old_descriptor,
            )
            raise
        try:
            _remove_detached_owned_output(
                project,
                parent_descriptor,
                stage_name,
                old_descriptor,
                allow_empty=True,
            )
        except (OSError, SourceProjectBuildError):
            # The new output is already committed.  Preserve a concurrently
            # changed old directory under its quarantined stage name.
            pass
        return

    rollback_name = _move_to_unique_sibling(
        parent_descriptor,
        output_name,
        prefix=f"{output_name}.rollback",
    )
    if not _named_directory_matches(parent_descriptor, rollback_name, old_descriptor):
        if _entry_exists_at(parent_descriptor, rollback_name):
            _restore_detached_output(
                parent_descriptor,
                detached_name=rollback_name,
                output_name=output_name,
            )
        raise SourceProjectBuildError("derived output changed concurrently")
    try:
        try:
            published = _rename_no_replace(
                parent_descriptor,
                stage_name,
                output_name,
            )
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise SourceProjectBuildError(
                    "derived output appeared concurrently"
                ) from exc
            raise SourceProjectBuildError(
                "cannot publish the staged output safely"
            ) from exc
        if not published:
            raise SourceProjectBuildError(
                "this platform cannot publish a derived output without a race"
            )
        _require_named_directory_identity(
            parent_descriptor,
            output_name,
            stage_descriptor,
            label="published output",
        )
        _validate_staged_artifacts(
            project,
            stage_descriptor,
            nodes=nodes,
            manifest_payload=manifest_payload,
        )
        current_old_entries = _validate_owned_directory_descriptor(
            project,
            old_descriptor,
            allow_empty=True,
        )
        if current_old_entries != old_entries:
            raise SourceProjectBuildError(
                "previous derived output changed during publication"
            )
        _require_directory_entries_unchanged(
            old_descriptor,
            old_entries,
            label="previous derived output",
        )
        if final_validator is not None:
            final_validator()
        os.fsync(parent_descriptor)
        _require_named_directory_identity(
            parent_descriptor,
            output_name,
            stage_descriptor,
            label="published output",
        )
        _require_named_directory_identity(
            parent_descriptor,
            rollback_name,
            old_descriptor,
            label="previous derived output",
        )
        if final_validator is not None:
            final_validator()
        _validate_staged_artifacts(
            project,
            stage_descriptor,
            nodes=nodes,
            manifest_payload=manifest_payload,
        )
    except BaseException:
        _rollback_portable_publication(
            parent_descriptor,
            output_name=output_name,
            stage_descriptor=stage_descriptor,
            rollback_name=rollback_name,
            old_descriptor=old_descriptor,
        )
        raise
    try:
        _remove_detached_owned_output(
            project,
            parent_descriptor,
            rollback_name,
            old_descriptor,
            allow_empty=True,
        )
    except (OSError, SourceProjectBuildError):
        pass


def _cache_hit(
    project: SourceProject,
    plan: NodePlan,
    previous_nodes: Mapping[str, Any],
    output_descriptor: int,
) -> bool:
    previous = previous_nodes.get(plan.name)
    if not isinstance(previous, Mapping) or previous.get("key") != plan.key:
        return False
    expected_digest = previous.get("sha256")
    if not isinstance(expected_digest, str):
        return False
    try:
        return _sha256_regular_at(output_descriptor, plan.output_name) == expected_digest
    except SourceProjectBuildError:
        return False


def _copy_verified_cache_entry(
    project: SourceProject,
    plan: NodePlan,
    previous_nodes: Mapping[str, Any],
    output_descriptor: int | None,
    stage_descriptor: int,
) -> str | None:
    """Copy one cache entry from a stable fd and verify the staged bytes."""

    previous = previous_nodes.get(plan.name)
    if not isinstance(previous, Mapping) or previous.get("key") != plan.key:
        return None
    expected_digest = previous.get("sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_digest
    ):
        return None
    if output_descriptor is None:
        return None
    try:
        source_descriptor = _open_regular_at(output_descriptor, plan.output_name)
    except SourceProjectBuildError:
        return None
    temporary_descriptor: int | None = None
    temporary_name = f".{plan.output_name}.{uuid.uuid4().hex}.cache"
    temporary_identity: tuple[int, int] | None = None
    try:
        source_stat = os.fstat(source_descriptor)
        temporary_descriptor = os.open(
            temporary_name,
            _regular_open_flags(write=True) | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=stage_descriptor,
        )
        temporary_stat = os.fstat(temporary_descriptor)
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        with os.fdopen(source_descriptor, "rb", closefd=False) as source, os.fdopen(
            temporary_descriptor, "wb", closefd=False
        ) as destination:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        # Re-open the staged destination and hash those bytes.  The mutable old
        # cache directory is no longer authoritative once the copy completes.
        staged_digest = _sha256_regular_at(stage_descriptor, temporary_name)
        if staged_digest != expected_digest:
            return None
        if _entry_identity_at(stage_descriptor, temporary_name) != temporary_identity:
            return None
        os.utime(
            temporary_descriptor,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )
        try:
            published = _rename_no_replace(
                stage_descriptor,
                temporary_name,
                plan.output_name,
            )
        except OSError as exc:
            raise SourceProjectBuildError(
                f"staged cache output appeared concurrently: {plan.output_name!r}"
            ) from exc
        if not published:
            raise SourceProjectBuildError(
                "this platform cannot publish a staged cache entry without a race"
            )
        temporary_name = ""
        if (
            _entry_identity_at(stage_descriptor, plan.output_name)
            != temporary_identity
        ):
            raise SourceProjectBuildError(
                f"staged cache output changed concurrently: {plan.output_name!r}"
            )
        final_digest = _sha256_regular_at(stage_descriptor, plan.output_name)
        if final_digest != expected_digest:
            if temporary_identity is not None:
                with contextlib.suppress(OSError):
                    _unlink_matching_identity(
                        stage_descriptor,
                        plan.output_name,
                        temporary_identity,
                    )
            return None
        return final_digest
    finally:
        os.close(source_descriptor)
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_name and temporary_identity is not None:
            with contextlib.suppress(OSError):
                _unlink_matching_identity(
                    stage_descriptor,
                    temporary_name,
                    temporary_identity,
                )


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
    stage_name: str | None = None
    stage_descriptor: int | None = None
    stage_published = False
    try:
        with _project_lock(project) as project_root_descriptor:
            with _output_parent_descriptor(
                project,
                project_root_descriptor,
                create=True,
            ) as parent_descriptor:
                assert parent_descriptor is not None
                opened_output = _open_owned_output_descriptor(
                    project,
                    parent_descriptor,
                    allow_absent=True,
                    allow_empty=True,
                )
                old_descriptor = opened_output[0] if opened_output is not None else None
                try:
                    previous = None
                    if not force and old_descriptor is not None:
                        try:
                            previous = _decode_previous_manifest(
                                _read_regular_at(old_descriptor, BUILD_MANIFEST_NAME)
                            )
                        except SourceProjectBuildError:
                            previous = None
                    previous_nodes: Mapping[str, Any] = (
                        previous.get("nodes", {})
                        if isinstance(previous, Mapping)
                        else {}
                    )
                    stage_name, stage_descriptor = _create_staged_directory(
                        parent_descriptor,
                        project.output_directory.name,
                    )
                    for plan in plans:
                        cached_digest = None if force else _copy_verified_cache_entry(
                            project,
                            plan,
                            previous_nodes,
                            old_descriptor,
                            stage_descriptor,
                        )
                        if cached_digest is not None:
                            action = "reused"
                            digest = cached_digest
                        else:
                            payload = plan.build_payload(context)
                            _atomic_write_at(stage_descriptor, plan.output_name, payload)
                            action = "built"
                            digest = _sha256_bytes(payload)
                        context.digests[plan.name] = digest
                        states.append(NodeState(
                            name=plan.name, action=action, key=plan.key,
                            output=plan.output_name, sha256=digest,
                        ))
                    # A path-backed compiler/bridge snapshot is disposable but
                    # must be validated and safely closed before publication.
                    # No snapshot cleanup error may occur after the commit.
                    context.close()
                    manifest_payload = _manifest_payload(
                        project,
                        nodes=states,
                        inputs=inputs,
                        painter_z_band=painter_z_band,
                        revisions=manifest_revisions,
                    )
                    _atomic_write_at(
                        stage_descriptor,
                        BUILD_MANIFEST_NAME,
                        manifest_payload,
                    )
                    _atomic_write_at(
                        stage_descriptor,
                        OWNED_OUTPUT_MARKER,
                        _owned_output_payload(project),
                    )
                    _validate_input_snapshot(snapshot)
                    def validate_publication_inputs() -> None:
                        _validate_input_snapshot(snapshot)
                        _require_output_parent_identity(
                            project,
                            project_root_descriptor,
                            parent_descriptor,
                        )

                    _publish_staged_directory(
                        project,
                        parent_descriptor,
                        stage_name,
                        stage_descriptor,
                        old_descriptor,
                        nodes=states,
                        manifest_payload=manifest_payload,
                        final_validator=validate_publication_inputs,
                    )
                    stage_published = True
                finally:
                    if stage_descriptor is not None:
                        if not stage_published and stage_name is not None:
                            _cleanup_partial_stage(
                                parent_descriptor,
                                stage_name,
                                stage_descriptor,
                            )
                        os.close(stage_descriptor)
                        stage_descriptor = None
                    if old_descriptor is not None:
                        os.close(old_descriptor)
    finally:
        context.close()

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
    manifest_bytes: bytes | None = None
    with _project_lock(project) as project_root_descriptor:
        with _output_parent_descriptor(
            project,
            project_root_descriptor,
            create=False,
        ) as parent_descriptor:
            opened_output = None
            if parent_descriptor is not None:
                opened_output = _open_owned_output_descriptor(
                    project,
                    parent_descriptor,
                    allow_absent=True,
                )
            output_descriptor = (
                opened_output[0] if opened_output is not None else None
            )
            output_entries = (
                opened_output[1] if opened_output is not None else set()
            )
            owned = output_descriptor is not None
            try:
                previous = None
                if output_descriptor is not None:
                    try:
                        manifest_bytes = _read_regular_at(
                            output_descriptor,
                            BUILD_MANIFEST_NAME,
                        )
                    except SourceProjectBuildError:
                        manifest_bytes = None
                    if manifest_bytes is not None:
                        previous = _decode_previous_manifest(manifest_bytes)
                previous_nodes: Mapping[str, Any] = (
                    previous.get("nodes", {})
                    if isinstance(previous, Mapping)
                    else {}
                )
                for plan in plans:
                    hit = (
                        output_descriptor is not None
                        and _cache_hit(
                            project,
                            plan,
                            previous_nodes,
                            output_descriptor,
                        )
                    )
                    try:
                        digest = (
                            _sha256_regular_at(output_descriptor, plan.output_name)
                            if output_descriptor is not None
                            else ""
                        )
                        output_exists = bool(digest)
                    except SourceProjectBuildError:
                        digest = ""
                        output_exists = False
                    action = "fresh" if hit else (
                        "stale"
                        if owned and (output_exists or previous is not None)
                        else "missing"
                    )
                    states.append(NodeState(
                        plan.name,
                        action,
                        plan.key,
                        plan.output_name,
                        digest,
                    ))
                expected_names = {plan.name for plan in plans}
                for name in sorted(set(previous_nodes) - expected_names):
                    old = previous_nodes[name]
                    output_name = (
                        old.get("output", "") if isinstance(old, Mapping) else ""
                    )
                    states.append(NodeState(
                        name,
                        "obsolete",
                        "",
                        output_name if isinstance(output_name, str) else "",
                        "",
                    ))
                expected_files = {
                    *(plan.output_name for plan in plans),
                    BUILD_MANIFEST_NAME,
                    OWNED_OUTPUT_MARKER,
                }
                def validate_current_output() -> set[str]:
                    assert output_descriptor is not None
                    current_entries = _validate_owned_directory_descriptor(
                        project,
                        output_descriptor,
                    )
                    if current_entries != output_entries:
                        raise SourceProjectBuildError(
                            "derived output changed while status was checked"
                        )
                    _require_directory_entries_unchanged(
                        output_descriptor,
                        current_entries,
                        label="derived output",
                    )
                    try:
                        current_manifest_bytes = _read_regular_at(
                            output_descriptor,
                            BUILD_MANIFEST_NAME,
                        )
                    except SourceProjectBuildError:
                        current_manifest_bytes = None
                    if current_manifest_bytes != manifest_bytes:
                        raise SourceProjectBuildError(
                            "derived output changed while status was checked"
                        )
                    for node in states:
                        if node.output not in expected_files or not node.sha256:
                            continue
                        if (
                            _sha256_regular_at(output_descriptor, node.output)
                            != node.sha256
                        ):
                            raise SourceProjectBuildError(
                                "derived output changed while status was checked: "
                                + node.output
                            )
                    _require_directory_entries_unchanged(
                        output_descriptor,
                        current_entries,
                        label="derived output",
                    )
                    return current_entries

                if output_descriptor is not None:
                    output_entries = validate_current_output()
                _validate_input_snapshot(snapshot)
                if output_descriptor is not None:
                    assert parent_descriptor is not None
                    _require_output_parent_identity(
                        project,
                        project_root_descriptor,
                        parent_descriptor,
                    )
                    _require_named_directory_identity(
                        parent_descriptor,
                        project.output_directory.name,
                        output_descriptor,
                        label="derived output",
                    )
                    output_entries = validate_current_output()
                    _require_output_parent_identity(
                        project,
                        project_root_descriptor,
                        parent_descriptor,
                    )
                    _require_named_directory_identity(
                        parent_descriptor,
                        project.output_directory.name,
                        output_descriptor,
                        label="derived output",
                    )
                else:
                    _require_output_absent(
                        project,
                        project_root_descriptor,
                        parent_descriptor,
                        message="derived output appeared while status was checked",
                    )
                for name in sorted(output_entries - expected_files):
                    states.append(NodeState(
                        f"unexpected:{name}",
                        "obsolete",
                        "",
                        name,
                        "",
                    ))
            finally:
                if output_descriptor is not None:
                    os.close(output_descriptor)
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
    manifest_fresh = manifest_bytes == expected_manifest
    manifest_action = (
        "fresh" if manifest_fresh else "stale" if manifest_bytes is not None else "missing"
    )
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
    with _project_lock(project) as project_root_descriptor:
        with _output_parent_descriptor(
            project,
            project_root_descriptor,
            create=False,
        ) as parent_descriptor:
            if parent_descriptor is None:
                _require_output_absent(
                    project,
                    project_root_descriptor,
                    None,
                    message="derived output appeared while clean was checked",
                )
                return output
            detached = _detach_and_validate_owned_output(
                project,
                parent_descriptor,
                purpose="clean",
                allow_absent=True,
            )
            if detached is None:
                _require_output_absent(
                    project,
                    project_root_descriptor,
                    parent_descriptor,
                    message="derived output appeared while clean was checked",
                )
                return output
            tombstone_name, tombstone_descriptor = detached
            try:
                def validate_clean_parent() -> None:
                    _require_output_parent_identity(
                        project,
                        project_root_descriptor,
                        parent_descriptor,
                    )

                _remove_detached_owned_output(
                    project,
                    parent_descriptor,
                    tombstone_name,
                    tombstone_descriptor,
                    final_validator=validate_clean_parent,
                )
            except BaseException:
                if _named_directory_matches(
                    parent_descriptor,
                    tombstone_name,
                    tombstone_descriptor,
                ):
                    _restore_detached_output(
                        parent_descriptor,
                        detached_name=tombstone_name,
                        output_name=output.name,
                    )
                raise
            finally:
                os.close(tombstone_descriptor)
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
