"""Source-authoritative TikZ project builds.

The project manifest stores only authored inputs and render intent.  Every
artifact produced by this module is derived, deterministic, and disposable.
"""

from __future__ import annotations

import argparse
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
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

SOURCE_PROJECT_SCHEMA_VERSION = "tikz-native-source-project/v1"
BUILD_MANIFEST_SCHEMA_VERSION = "tikz-native-build-manifest/v1"
SHAPE_ASSET_SCHEMA_VERSION = "tikz-native-shape-asset/v1"
MOTION_ASSET_SCHEMA_VERSION = "tikz-native-motion-asset/v1"
COMPOSITING_SCHEMA_VERSION = "tikz-native-unified-compositing/v1"

PROVIDER_COMPONENT = "source_project_build"
PROVIDER_CAPABILITY = "source_authoritative_project_build_v1"
PROVIDER_COMPONENT_REVISION = 1

DEFAULT_COMPONENT_REVISIONS: Mapping[str, int] = MappingProxyType(
    {
        "source_project_build": PROVIDER_COMPONENT_REVISION,
        "tikz_compiler": 1,
        "motion_asset": 1,
        "unified_compositor": 1,
        "bridge_codegen": 1,
    }
)

_FORBIDDEN_MANIFEST_KEYS = {
    "compositingmode",
    "compositing_mode",
    "implementationmode",
    "implementation_mode",
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


@dataclass(frozen=True)
class NodePlan:
    name: str
    output_name: str
    key: str
    payload: bytes
    component_revisions: Mapping[str, int]


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
            "schemaVersion": BUILD_MANIFEST_SCHEMA_VERSION,
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
    painter_z_band: PainterZBand

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": BUILD_MANIFEST_SCHEMA_VERSION,
            "project": str(self.project.manifest_path),
            "outputDirectory": str(self.project.output_directory),
            "manifest": str(self.manifest_path),
            "fresh": self.fresh,
            "painterZBand": self.painter_z_band.as_list(),
            "nodes": [node.as_dict() for node in self.nodes],
        }


ShapeAssetBuilder = Callable[[SourceProject, str], Any]
BridgeGenerator = Callable[[Mapping[str, Any]], str]


def provider_component_descriptor() -> dict[str, Any]:
    """Return the Provider component record owned by this module."""

    return {
        "name": PROVIDER_COMPONENT,
        "revision": PROVIDER_COMPONENT_REVISION,
        "capabilities": [PROVIDER_CAPABILITY],
        "owns": [
            "tikz_native.source_project",
            "tikz_native/schemas/tikz-native-source-project-v1.schema.json",
            "tikz_native/schemas/tikz-native-build-manifest-v1.schema.json",
        ],
    }


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


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
        value = json.loads(_read_utf8(path, label=label))
    except json.JSONDecodeError as exc:
        raise SourceProjectError(
            f"invalid JSON in {label} {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SourceProjectError(f"{label} must contain a JSON object: {path}")
    return value


def _normalise_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", key).lower()


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


def _parse_painter_z_band(value: Any) -> PainterZBand | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        _ensure_keys(value, {"min", "max", "minimum", "maximum"}, location="painterZBand")
        minimum = _one_of(value, ("min", "minimum"), required=True)
        maximum = _one_of(value, ("max", "maximum"), required=True)
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
            "tikz",
            "motionJson",
            "motionSource",
            "motion",
            "hooksSource",
            "hooks",
            "bridgeRequestTemplate",
            "bridgeTemplate",
            "derivedOutput",
            "outputDirectory",
            "renderIntent",
        },
        location="manifest",
    )
    version = raw.get("schemaVersion")
    if version != SOURCE_PROJECT_SCHEMA_VERSION:
        raise SourceProjectError(
            f"schemaVersion must be {SOURCE_PROJECT_SCHEMA_VERSION!r}, got {version!r}"
        )

    tikz_raw = _one_of(raw, ("tikzSource", "tikz"), required=True)
    motion_raw = _one_of(raw, ("motionJson", "motionSource", "motion"))
    hooks_raw = _one_of(raw, ("hooksSource", "hooks"))
    bridge_raw = _one_of(raw, ("bridgeRequestTemplate", "bridgeTemplate"))
    output_raw = _one_of(raw, ("derivedOutput", "outputDirectory"))
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
    paint_policy = render_intent.get("paintPolicy", "source")
    if not isinstance(paint_policy, str) or not paint_policy.strip():
        raise SourceProjectError("renderIntent.paintPolicy must be a non-empty string")
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
    output_directory = _resolve_project_path(
        root, output_raw, label="derivedOutput", must_exist=False
    )
    if output_directory == root:
        raise SourceProjectError("derivedOutput must not be the project root")
    if manifest == output_directory or manifest.is_relative_to(output_directory):
        raise SourceProjectError("derivedOutput must not contain the source manifest")

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
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, str, int, float, bool)) or value is None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    return {"generated": stripped}
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
    candidates = (
        ("tikz_native.compiler", "compile_tikz_source"),
        ("tikz_native.compiler", "compile_tikz"),
        ("tikz_native.compiler", "compile_source"),
        ("tikz_native.bridge", "compile_tikz_source"),
        ("tikz_native.bridge", "compile_tikz"),
    )
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
        except (ImportError, AttributeError):
            continue
        if not callable(function):
            continue
        try:
            return _normalise_compiler_result(
                _call_candidate(function, project, source)
            )
        except TypeError:
            continue

    # The source envelope remains deterministic and rebuildable.  Providers
    # with a native compiler entry point replace it through shape_asset_builder.
    return {
        "kind": "source-authoritative-compiler-input",
        "compiled": False,
        "tikz": source,
        "projection": project.projection,
    }


def _normalise_revisions(
    revisions: Mapping[str, int] | None,
) -> dict[str, int]:
    result = dict(DEFAULT_COMPONENT_REVISIONS)
    if revisions is not None:
        for name, revision in revisions.items():
            if not isinstance(name, str) or not name:
                raise SourceProjectError("component revision names must be non-empty strings")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise SourceProjectError(
                    f"component revision {name!r} must be a non-negative integer"
                )
            result[name] = revision
    return dict(sorted(result.items()))


def _source_inputs(project: SourceProject) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tikzSource": {
            "path": _relative(project, project.tikz_source),
            "sha256": _sha256_file(project.tikz_source),
        }
    }
    for key, path in (
        ("motionJson", project.motion_json),
        ("hooksSource", project.hooks_source),
        ("bridgeRequestTemplate", project.bridge_request_template),
    ):
        if path is not None:
            result[key] = {"path": _relative(project, path), "sha256": _sha256_file(path)}
    return result


def _extract_bridge_python(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in (
            "generatedSource",
            "generated_source",
            "pythonSource",
            "python_source",
            "source",
            "python",
            "code",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return None


def _call_bridge_generator(request: Mapping[str, Any]) -> str | None:
    candidates = (
        ("tikz_native.bridge", "generate_source"),
        ("tikz_native.bridge", "generate_manim_source"),
        ("tikz_native.bridge", "process_request"),
        ("tikz_native.bridge", "handle_request"),
        ("tikz_native.bridge", "compile_request"),
    )
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
        except (ImportError, AttributeError):
            continue
        if not callable(function):
            continue
        try:
            signature = inspect.signature(function)
            if len(signature.parameters) == 1:
                result = function(dict(request))
            else:
                continue
        except TypeError:
            continue
        source = _extract_bridge_python(result)
        if source is not None:
            return source
    return None


def _placeholder_values(
    project: SourceProject,
    *,
    tikz_source: str,
    motion_source: str | None,
    hooks_source: str | None,
    painter_z_band: PainterZBand,
) -> dict[str, str]:
    return {
        "TIKZ_SOURCE": tikz_source,
        "MOTION_JSON": motion_source if motion_source is not None else "null",
        "HOOKS_SOURCE": hooks_source if hooks_source is not None else "",
        "PAINT_POLICY": project.paint_policy,
        "PAINTER_Z_BAND": json.dumps(painter_z_band.as_list(), separators=(",", ":")),
    }


def _substitute_placeholders(template: str, values: Mapping[str, str]) -> str:
    result = template
    known = set(values)
    for name, value in values.items():
        result = result.replace("${" + name + "}", value)
    remaining = sorted(set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", result)))
    unsupported = [name for name in remaining if name not in known]
    if unsupported:
        raise SourceProjectBuildError(
            "unsupported Bridge template placeholder(s): " + ", ".join(unsupported)
        )
    if remaining:
        raise SourceProjectBuildError(
            "Bridge template placeholder substitution did not converge: "
            + ", ".join(remaining)
        )
    return result


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


def _insert_open_face_arguments(
    source: str, *, paint_policy: str, painter_z_band: PainterZBand
) -> str:
    call_pattern = re.compile(r"OpenFaceOcclusion3D\s*\(")
    if not call_pattern.search(source):
        if re.search(r"open[_ -]?face|OpenFace", source, flags=re.IGNORECASE):
            raise SourceProjectBuildError(
                "generated source contains an open-face implementation but does not "
                "expose the current OpenFaceOcclusion3D binding"
            )
        return source

    binding_patterns = (
        r"\bfrom\s+[\w.]+\s+import\s+[^\n]*\bOpenFaceOcclusion3D\b",
        r"\bOpenFaceOcclusion3D\s*=",
        r"\bclass\s+OpenFaceOcclusion3D\b",
        r"\bimport\s+[\w.]+\s+as\s+\w+",  # accepted only when a direct alias is established below
    )
    has_binding = any(re.search(pattern, source) for pattern in binding_patterns[:3])
    if not has_binding:
        raise SourceProjectBuildError(
            "generated source calls OpenFaceOcclusion3D without exposing the current binding"
        )

    source = re.sub(
        r"compositing_mode\s*=\s*(['\"])[^'\"]*\1",
        'compositing_mode="unified"',
        source,
    )

    arguments = [
        'compositing_mode="unified"',
        f"paint_policy={project_literal(paint_policy)}",
        f"painter_z_band={project_literal(tuple(painter_z_band.as_list()))}",
    ]

    def replace_call(match: re.Match[str]) -> str:
        start = match.end()
        tail = source[start : start + 1000]
        existing: set[str] = set()
        for name in ("compositing_mode", "paint_policy", "painter_z_band"):
            if re.search(rf"\b{name}\s*=", tail):
                existing.add(name)
        missing = [argument for argument in arguments if argument.split("=", 1)[0] not in existing]
        if not missing:
            return match.group(0)
        return match.group(0) + ", ".join(missing) + ", "

    return call_pattern.sub(replace_call, source)


def project_literal(value: Any) -> str:
    """Return a deterministic Python literal for generated source."""

    return repr(value)


def _rewrite_fades(source: str, targets: Sequence[str]) -> str:
    for target in targets:
        escaped = re.escape(target)
        source = re.sub(
            rf"\b(FadeIn|FadeOut)\s*\(\s*{escaped}\b",
            lambda match: f"{match.group(1)}(controller.display_mobject",
            source,
        )
    return source


def rewrite_generated_source(
    source: str,
    *,
    paint_policy: str,
    painter_z_band: PainterZBand,
    whole_figure_targets: Sequence[str] = (),
) -> str:
    """Rewrite disposable generated Python to current unified behavior."""

    rewritten = _insert_open_face_arguments(
        source,
        paint_policy=paint_policy,
        painter_z_band=painter_z_band,
    )
    rewritten = _rewrite_fades(rewritten, whole_figure_targets)
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
    template: str,
    tikz_source: str,
    motion_source: str | None,
    hooks_source: str | None,
    painter_z_band: PainterZBand,
    bridge_generator: BridgeGenerator | None,
) -> str:
    substituted = _substitute_placeholders(
        template,
        _placeholder_values(
            project,
            tikz_source=tikz_source,
            motion_source=motion_source,
            hooks_source=hooks_source,
            painter_z_band=painter_z_band,
        ),
    )
    request: Mapping[str, Any] | None = None
    try:
        parsed = json.loads(substituted)
        if isinstance(parsed, Mapping):
            request = parsed
    except json.JSONDecodeError:
        request = None

    generated: str | None = None
    if request is not None:
        if bridge_generator is not None:
            generated = bridge_generator(request)
        else:
            generated = _call_bridge_generator(request)
        if generated is None:
            generated = _extract_bridge_python(request)
        if generated is None:
            raise SourceProjectBuildError(
                "Bridge request template produced JSON, but the current Bridge did not "
                "return generated Python"
            )
    else:
        generated = substituted

    return rewrite_generated_source(
        generated,
        paint_policy=project.paint_policy,
        painter_z_band=painter_z_band,
        whole_figure_targets=_whole_figure_targets(request),
    )


def _plan_nodes(
    project: SourceProject,
    *,
    component_revisions: Mapping[str, int] | None,
    shape_asset_builder: ShapeAssetBuilder | None,
    bridge_generator: BridgeGenerator | None,
) -> tuple[list[NodePlan], PainterZBand, dict[str, Any]]:
    revisions = _normalise_revisions(component_revisions)
    tikz_bytes = project.tikz_source.read_bytes()
    tikz_source = _read_utf8(project.tikz_source, label="TikZ source")
    painter_z_band = derive_painter_z_band(project, tikz_bytes)
    projection_digest = _sha256_bytes(_canonical_json(project.projection))

    compiler_revision = {"tikz_compiler": revisions["tikz_compiler"]}
    shape_key = _build_key(
        {
            "node": "shape",
            "schemaVersion": SHAPE_ASSET_SCHEMA_VERSION,
            "tikzSha256": _sha256_bytes(tikz_bytes),
            "projectionSha256": projection_digest,
            "componentRevisions": compiler_revision,
        }
    )
    builder = shape_asset_builder or _default_shape_asset_builder
    compiled = _normalise_compiler_result(builder(project, tikz_source))
    shape_payload = _canonical_json(
        {
            "schemaVersion": SHAPE_ASSET_SCHEMA_VERSION,
            "buildKey": shape_key,
            "source": {
                "path": _relative(project, project.tikz_source),
                "sha256": _sha256_bytes(tikz_bytes),
            },
            "projection": project.projection,
            "compiler": {
                "component": "tikz_compiler",
                "revision": revisions["tikz_compiler"],
            },
            "asset": compiled,
        }
    )
    plans = [
        NodePlan(
            name="shape",
            output_name="shape-asset.json",
            key=shape_key,
            payload=shape_payload,
            component_revisions=compiler_revision,
        )
    ]

    motion_source: str | None = None
    motion_payload: bytes | None = None
    motion_digest: str | None = None
    if project.motion_json is not None:
        motion_source = _read_utf8(project.motion_json, label="motion JSON")
        try:
            motion_value = json.loads(motion_source)
        except json.JSONDecodeError as exc:
            raise SourceProjectError(
                f"invalid motion JSON {project.motion_json}: line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc
        motion_revision = {"motion_asset": revisions["motion_asset"]}
        motion_key = _build_key(
            {
                "node": "motion",
                "schemaVersion": MOTION_ASSET_SCHEMA_VERSION,
                "shapeSha256": _sha256_bytes(shape_payload),
                "motionSha256": _sha256_bytes(motion_source.encode("utf-8")),
                "componentRevisions": motion_revision,
            }
        )
        motion_payload = _canonical_json(
            {
                "schemaVersion": MOTION_ASSET_SCHEMA_VERSION,
                "buildKey": motion_key,
                "source": {
                    "path": _relative(project, project.motion_json),
                    "sha256": _sha256_bytes(motion_source.encode("utf-8")),
                },
                "shapeAssetSha256": _sha256_bytes(shape_payload),
                "motion": motion_value,
            }
        )
        motion_digest = _sha256_bytes(motion_payload)
        plans.append(
            NodePlan(
                name="motion",
                output_name="motion-asset.json",
                key=motion_key,
                payload=motion_payload,
                component_revisions=motion_revision,
            )
        )

    compositing_revision = {
        "unified_compositor": revisions["unified_compositor"]
    }
    compositing_key = _build_key(
        {
            "node": "compositing",
            "schemaVersion": COMPOSITING_SCHEMA_VERSION,
            "shapeSha256": _sha256_bytes(shape_payload),
            "motionSha256": motion_digest,
            "paintPolicy": project.paint_policy,
            "projectionSha256": projection_digest,
            "painterZBand": painter_z_band.as_list(),
            "componentRevisions": compositing_revision,
        }
    )
    compositing_payload = _canonical_json(
        {
            "schemaVersion": COMPOSITING_SCHEMA_VERSION,
            "buildKey": compositing_key,
            "compositingMode": "unified",
            "paintPolicy": project.paint_policy,
            "projection": project.projection,
            "painterZBand": painter_z_band.as_list(),
            "shapeAssetSha256": _sha256_bytes(shape_payload),
            "motionAssetSha256": motion_digest,
            "component": {
                "name": "unified_compositor",
                "revision": revisions["unified_compositor"],
            },
        }
    )
    plans.append(
        NodePlan(
            name="compositing",
            output_name="unified-compositing.json",
            key=compositing_key,
            payload=compositing_payload,
            component_revisions=compositing_revision,
        )
    )

    hooks_source: str | None = None
    if project.hooks_source is not None:
        hooks_source = _read_utf8(project.hooks_source, label="hooks source")

    if project.bridge_request_template is not None:
        template = _read_utf8(
            project.bridge_request_template, label="Bridge request template"
        )
        generated = _generate_scene_source(
            project,
            template=template,
            tikz_source=tikz_source,
            motion_source=motion_source,
            hooks_source=hooks_source,
            painter_z_band=painter_z_band,
            bridge_generator=bridge_generator,
        )
        generated_payload = generated.encode("utf-8")
        generated_revision = {"bridge_codegen": revisions["bridge_codegen"]}
        generated_key = _build_key(
            {
                "node": "generated_source",
                "compositingSha256": _sha256_bytes(compositing_payload),
                "bridgeTemplateSha256": _sha256_bytes(template.encode("utf-8")),
                "hooksSha256": (
                    _sha256_bytes(hooks_source.encode("utf-8"))
                    if hooks_source is not None
                    else None
                ),
                "paintPolicy": project.paint_policy,
                "painterZBand": painter_z_band.as_list(),
                "componentRevisions": generated_revision,
            }
        )
        plans.append(
            NodePlan(
                name="generated_source",
                output_name="generated_scene.py",
                key=generated_key,
                payload=generated_payload,
                component_revisions=generated_revision,
            )
        )

    inputs = _source_inputs(project)
    return plans, painter_z_band, inputs


def _load_previous_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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
    revisions: Mapping[str, int],
) -> bytes:
    return _canonical_json(
        {
            "schemaVersion": BUILD_MANIFEST_SCHEMA_VERSION,
            "sourceProjectSchemaVersion": SOURCE_PROJECT_SCHEMA_VERSION,
            "provider": provider_component_descriptor(),
            "project": project.manifest_path.name,
            "inputs": inputs,
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


def build_project(
    manifest_path: str | os.PathLike[str],
    *,
    force: bool = False,
    component_revisions: Mapping[str, int] | None = None,
    shape_asset_builder: ShapeAssetBuilder | None = None,
    bridge_generator: BridgeGenerator | None = None,
) -> BuildResult:
    """Build or incrementally refresh all derived project artifacts."""

    project = load_source_project(manifest_path)
    revisions = _normalise_revisions(component_revisions)
    plans, painter_z_band, inputs = _plan_nodes(
        project,
        component_revisions=revisions,
        shape_asset_builder=shape_asset_builder,
        bridge_generator=bridge_generator,
    )
    manifest_path_obj = project.output_directory / "build-manifest.json"
    previous = None if force else _load_previous_manifest(manifest_path_obj)
    previous_nodes: Mapping[str, Any] = (
        previous.get("nodes", {}) if isinstance(previous, Mapping) else {}
    )

    states: list[NodeState] = []
    for plan in plans:
        output = _safe_output_path(project, plan.output_name)
        if not force and _cache_hit(project, plan, previous_nodes):
            action = "reused"
            digest = _sha256_file(output)
        else:
            _atomic_write(output, plan.payload)
            action = "built"
            digest = _sha256_bytes(plan.payload)
        states.append(
            NodeState(
                name=plan.name,
                action=action,
                key=plan.key,
                output=plan.output_name,
                sha256=digest,
            )
        )

    manifest_bytes = _manifest_payload(
        project,
        nodes=states,
        inputs=inputs,
        painter_z_band=painter_z_band,
        revisions=revisions,
    )
    if not manifest_path_obj.is_file() or manifest_path_obj.read_bytes() != manifest_bytes:
        _atomic_write(manifest_path_obj, manifest_bytes)

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
    component_revisions: Mapping[str, int] | None = None,
    shape_asset_builder: ShapeAssetBuilder | None = None,
    bridge_generator: BridgeGenerator | None = None,
) -> ProjectStatus:
    """Return whether each expected derived node is fresh."""

    project = load_source_project(manifest_path)
    revisions = _normalise_revisions(component_revisions)
    plans, painter_z_band, _ = _plan_nodes(
        project,
        component_revisions=revisions,
        shape_asset_builder=shape_asset_builder,
        bridge_generator=bridge_generator,
    )
    manifest_path_obj = project.output_directory / "build-manifest.json"
    previous = _load_previous_manifest(manifest_path_obj)
    previous_nodes: Mapping[str, Any] = (
        previous.get("nodes", {}) if isinstance(previous, Mapping) else {}
    )
    states: list[NodeState] = []
    for plan in plans:
        output = _safe_output_path(project, plan.output_name)
        if _cache_hit(project, plan, previous_nodes):
            action = "fresh"
            digest = _sha256_file(output)
        else:
            action = "stale" if output.exists() or previous is not None else "missing"
            digest = _sha256_file(output) if output.is_file() else ""
        states.append(
            NodeState(
                name=plan.name,
                action=action,
                key=plan.key,
                output=plan.output_name,
                sha256=digest,
            )
        )
    expected_names = {plan.name for plan in plans}
    extra_names = sorted(set(previous_nodes) - expected_names)
    for name in extra_names:
        previous_node = previous_nodes[name]
        output_name = (
            previous_node.get("output", "")
            if isinstance(previous_node, Mapping)
            else ""
        )
        states.append(
            NodeState(
                name=name,
                action="obsolete",
                key="",
                output=output_name if isinstance(output_name, str) else "",
                sha256="",
            )
        )
    fresh = previous is not None and all(node.action == "fresh" for node in states)
    return ProjectStatus(
        project=project,
        fresh=fresh,
        nodes=tuple(states),
        manifest_path=manifest_path_obj,
        painter_z_band=painter_z_band,
    )


def clean_project(manifest_path: str | os.PathLike[str]) -> Path:
    """Remove only the validated derived output directory."""

    project = load_source_project(manifest_path)
    output = project.output_directory
    if output == project.root or project.manifest_path.is_relative_to(output):
        raise SourceProjectBuildError("refusing to clean an unsafe output directory")
    if output.is_symlink():
        resolved = output.resolve(strict=True)
        try:
            resolved.relative_to(project.root)
        except ValueError as exc:
            raise SourceProjectBuildError("refusing to clean an external symlink") from exc
        output.unlink()
    elif output.exists():
        shutil.rmtree(output)
    return output


def _parse_revision_arguments(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SourceProjectError(
                f"component revision must use NAME=INTEGER syntax: {value!r}"
            )
        name, raw_revision = value.split("=", 1)
        try:
            revision = int(raw_revision)
        except ValueError as exc:
            raise SourceProjectError(
                f"component revision must be an integer: {value!r}"
            ) from exc
        result[name] = revision
    return _normalise_revisions(result)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tikz-native-project",
        description="Build disposable artifacts from an authoritative TikZ source project.",
    )
    parser.add_argument(
        "--component-revision",
        action="append",
        default=[],
        metavar="NAME=INTEGER",
        help="override a Provider component revision for cache invalidation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "status", "rebuild", "clean"):
        child = subparsers.add_parser(command)
        child.add_argument("project", help="path to project.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        revisions = _parse_revision_arguments(arguments.component_revision)
        if arguments.command == "build":
            payload = build_project(
                arguments.project, component_revisions=revisions
            ).as_dict()
            exit_code = 0
        elif arguments.command == "rebuild":
            payload = rebuild_project(
                arguments.project, component_revisions=revisions
            ).as_dict()
            exit_code = 0
        elif arguments.command == "status":
            status = status_project(
                arguments.project, component_revisions=revisions
            )
            payload = status.as_dict()
            exit_code = 0 if status.fresh else 1
        else:
            removed = clean_project(arguments.project)
            payload = {
                "schemaVersion": BUILD_MANIFEST_SCHEMA_VERSION,
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
    "DEFAULT_COMPONENT_REVISIONS",
    "PROVIDER_CAPABILITY",
    "PROVIDER_COMPONENT",
    "PROVIDER_COMPONENT_REVISION",
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
