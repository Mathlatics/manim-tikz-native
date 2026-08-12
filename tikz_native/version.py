from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Final


__version__ = "0.1.0"
PROTOCOL_VERSION = 1
REQUEST_SCHEMA = "tikz-native-bridge.request/v1"
RESPONSE_SCHEMA = "tikz-native-bridge.response/v1"
ASSET_SCHEMA = "tikz-native-asset/v1"

COMPONENT_REVISION_SCHEMA = "tikz-native-component-revisions/v1"
COMPONENT_ASSET_COMPILER = "asset_compiler"
COMPONENT_GEOMETRY_RIG_2D = "geometry_rig_2d"
COMPONENT_NATIVE_MANIM_SOURCE_2D = "native_manim_source_2d"
COMPONENT_NATIVE_RIG_2D = "native_rig_2d"
COMPONENT_MOTION_PREVIEW_2D = "motion_preview_2d"
COMPONENT_GEOMETRY_RIG_3D = "geometry_rig_3d"
COMPONENT_EMBEDDED_MOTION_3D = "embedded_motion_3d"
COMPONENT_MOTION_PREVIEW_3D = "motion_preview_3d"


# A component revision is the build identity at which that compatibility
# surface last changed.  The paired implementation digest makes this
# fail-closed: editing one of the declared files without deliberately updating
# its component contract immediately yields a fresh component-sha256 revision.
#
# The asset/3D components deliberately retain the 6920... identity.  The
# capability implementations were reviewed across that cut and remain
# compatible.  Metadata dispatch was later separated from provider.py without
# changing compile/render semantics; its current digest is explicitly recorded
# below rather than pretending that the whole source tree stayed byte-identical.
_LEGACY_6920: Final = (
    "source-sha256:6920c63acf10ec22c3f94a1eeb9374799f5ce467419cb610a447675e9678c0ab"
)
_NATIVE_SOURCE_01DF: Final = (
    "source-sha256:01df91473770e47746d4f14de2d94297a2e846be7dcb250652b65168fcda30d6"
)


_COMPONENT_DEFINITIONS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    COMPONENT_ASSET_COMPILER: {
        "dependencies": (),
        "files": (
            "animation.py",
            "bridge.py",
            "compatibility.py",
            "compiler.py",
            "fixed_view_renderer.py",
            "macro_frontend.py",
            "manim_renderer.py",
            "occlusion_3d.py",
            "projection_3d.py",
            "provider.py",
            "schemas/asset-v1.schema.json",
            "schemas/request-v1.schema.json",
            "schemas/response-v1.schema.json",
            "subset_v0_1.json",
        ),
    },
    COMPONENT_GEOMETRY_RIG_2D: {
        "dependencies": (COMPONENT_ASSET_COMPILER,),
        "files": (
            "geometry_rig.py",
            "geometry_rig_bridge.py",
            "schemas/geometry-rig-bridge-request-v1.schema.json",
            "schemas/geometry-rig-bridge-response-v1.schema.json",
            "schemas/geometry-rig-v1.schema.json",
        ),
    },
    COMPONENT_NATIVE_MANIM_SOURCE_2D: {
        "dependencies": (COMPONENT_GEOMETRY_RIG_2D,),
        "files": ("native_manim_codegen_2d.py",),
    },
    COMPONENT_NATIVE_RIG_2D: {
        "dependencies": (COMPONENT_ASSET_COMPILER,),
        "files": (
            "dynamic_geometry.py",
            "motion_runtime.py",
            "native_rig_2d.py",
            "schemas/motion-v1.schema.json",
        ),
    },
    COMPONENT_MOTION_PREVIEW_2D: {
        "dependencies": (COMPONENT_NATIVE_RIG_2D,),
        "files": (
            "motion_bridge.py",
            "motion_render.py",
            "schemas/motion-bridge-request-v1.schema.json",
            "schemas/motion-bridge-response-v1.schema.json",
        ),
    },
    COMPONENT_GEOMETRY_RIG_3D: {
        "dependencies": (COMPONENT_ASSET_COMPILER,),
        "files": (
            "geometry_rig_3d.py",
            "geometry_rig_3d_bridge.py",
            "schemas/geometry-rig-3d-bridge-request-v1.schema.json",
            "schemas/geometry-rig-3d-bridge-response-v1.schema.json",
            "schemas/geometry-rig-3d-v1.schema.json",
        ),
    },
    COMPONENT_EMBEDDED_MOTION_3D: {
        "dependencies": (
            COMPONENT_ASSET_COMPILER,
            COMPONENT_GEOMETRY_RIG_3D,
        ),
        "files": (
            "camera_3d.py",
            "dynamic_geometry.py",
            "manim_renderer_3d.py",
            "motion_3d.py",
            "motion_3d_runtime.py",
            "schemas/motion-3d-v1.schema.json",
        ),
    },
    COMPONENT_MOTION_PREVIEW_3D: {
        "dependencies": (COMPONENT_EMBEDDED_MOTION_3D,),
        "files": (
            "motion_3d_bridge.py",
            "motion_3d_render.py",
            "schemas/motion-3d-bridge-request-v1.schema.json",
            "schemas/motion-3d-bridge-response-v1.schema.json",
        ),
    },
}


_DECLARED_COMPONENT_REVISIONS: Final[dict[str, str]] = {
    COMPONENT_ASSET_COMPILER: _LEGACY_6920,
    COMPONENT_GEOMETRY_RIG_2D: _NATIVE_SOURCE_01DF,
    COMPONENT_NATIVE_MANIM_SOURCE_2D: _NATIVE_SOURCE_01DF,
    COMPONENT_NATIVE_RIG_2D: _LEGACY_6920,
    COMPONENT_MOTION_PREVIEW_2D: _LEGACY_6920,
    COMPONENT_GEOMETRY_RIG_3D: _LEGACY_6920,
    COMPONENT_EMBEDDED_MOTION_3D: _LEGACY_6920,
    COMPONENT_MOTION_PREVIEW_3D: _LEGACY_6920,
}


# Filled with the verified digests after the component manifest was defined.
# A mismatch does not silently retain the declaration; it returns the actual
# component-sha256 identity and therefore invalidates only that component.
_DECLARED_IMPLEMENTATION_DIGESTS: Final[dict[str, str]] = {
    COMPONENT_ASSET_COMPILER: (
        "d7391d62dbcb394c188003e8f4fb6fcd1cf36985f7ef10364ebd7861348f30d2"
    ),
    COMPONENT_GEOMETRY_RIG_2D: (
        "ad6b968b537cc18a37d719d8d4a7aebfefff814e85c41ed394cebe86cc689879"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_2D: (
        "aa5321c5a8826eb73da6c572be97f57931420f887844b82815757ac995133e15"
    ),
    COMPONENT_NATIVE_RIG_2D: (
        "5e43b99023231256eb4cad4fd44fcba3b46c293227ce1f1a29a3e42d5c2ba0f9"
    ),
    COMPONENT_MOTION_PREVIEW_2D: (
        "95c450c576ed90bc729ffe8ce04c4db08b79023c3d717420ad6b983ae3428231"
    ),
    COMPONENT_GEOMETRY_RIG_3D: (
        "33bdba7b65adac19487ca8148eed63413ee67cfe421b18ffa93816576c073e20"
    ),
    COMPONENT_EMBEDDED_MOTION_3D: (
        "ebfa568b28c292f0a668328e7af7ccced486b4ae887e5f89f8631e5327c188b6"
    ),
    COMPONENT_MOTION_PREVIEW_3D: (
        "0f520c0cd5516363b97a557232d67b267d009ab8eaffabf69847e3a269f5532a"
    ),
}


_COMPONENT_NEUTRAL_FILES: Final[frozenset[str]] = frozenset(
    {
        "__init__.py",
        "__main__.py",
        "provider_metadata.py",
        "regression.py",
        "version.py",
    }
)


@lru_cache(maxsize=1)
def provider_revision() -> str:
    """Return a deterministic revision for cache invalidation.

    A release pipeline may inject a Git commit with
    ``TIKZ_NATIVE_PROVIDER_REVISION``.  Editable local installs currently live
    outside a Git repository, so their fallback revision hashes the installed
    provider sources, subset registry, and public JSON schemas.
    """

    explicit = os.environ.get("TIKZ_NATIVE_PROVIDER_REVISION", "").strip()
    if explicit:
        return explicit

    package_root = Path(__file__).resolve().parent
    candidates = sorted(
        (
            path
            for path in package_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".json"}
        ),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in candidates:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"source-sha256:{digest.hexdigest()}"


def _component_implementation_digest(
    component: str,
    *,
    memo: dict[str, str],
) -> str:
    if component in memo:
        return memo[component]
    definition = _COMPONENT_DEFINITIONS.get(component)
    if definition is None:
        raise ValueError(f"unknown TikZ Native component: {component!r}")
    digest = hashlib.sha256()
    digest.update(COMPONENT_REVISION_SCHEMA.encode("utf-8"))
    digest.update(component.encode("utf-8"))
    for dependency in sorted(definition["dependencies"]):
        dependency_digest = _component_implementation_digest(
            dependency,
            memo=memo,
        )
        digest.update(dependency.encode("utf-8"))
        digest.update(bytes.fromhex(dependency_digest))
    package_root = Path(__file__).resolve().parent
    for relative_text in sorted(definition["files"]):
        relative = Path(relative_text)
        path = package_root / relative
        if not path.is_file():
            raise RuntimeError(
                f"TikZ Native component {component!r} is missing {relative_text!r}"
            )
        relative_bytes = relative.as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(4, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    memo[component] = digest.hexdigest()
    return memo[component]


@lru_cache(maxsize=1)
def provider_component_implementation_revisions() -> dict[str, str]:
    memo: dict[str, str] = {}
    return {
        component: "component-sha256:"
        + _component_implementation_digest(component, memo=memo)
        for component in _COMPONENT_DEFINITIONS
    }


@lru_cache(maxsize=1)
def provider_component_revisions() -> dict[str, str]:
    implementations = provider_component_implementation_revisions()
    result: dict[str, str] = {}
    for component, implementation_revision in implementations.items():
        actual_digest = implementation_revision.removeprefix("component-sha256:")
        expected_digest = _DECLARED_IMPLEMENTATION_DIGESTS.get(component, "")
        declared_revision = _DECLARED_COMPONENT_REVISIONS.get(component, "")
        result[component] = (
            declared_revision
            if declared_revision and expected_digest == actual_digest
            else implementation_revision
        )
    return result


def provider_component_revision(component: str) -> str:
    try:
        return provider_component_revisions()[component]
    except KeyError as exc:
        raise ValueError(f"unknown TikZ Native component: {component!r}") from exc


def provider_component_files() -> dict[str, tuple[str, ...]]:
    return {
        component: tuple(definition["files"])
        for component, definition in _COMPONENT_DEFINITIONS.items()
    }


def provider_component_neutral_files() -> frozenset[str]:
    return _COMPONENT_NEUTRAL_FILES


def provider_component_revision_matches(component: str, recorded: object) -> bool:
    value = str(recorded or "").strip()
    if not value:
        return False
    return value == provider_component_revision(component)


__all__ = [
    "ASSET_SCHEMA",
    "COMPONENT_ASSET_COMPILER",
    "COMPONENT_EMBEDDED_MOTION_3D",
    "COMPONENT_GEOMETRY_RIG_2D",
    "COMPONENT_GEOMETRY_RIG_3D",
    "COMPONENT_MOTION_PREVIEW_2D",
    "COMPONENT_MOTION_PREVIEW_3D",
    "COMPONENT_NATIVE_MANIM_SOURCE_2D",
    "COMPONENT_NATIVE_RIG_2D",
    "COMPONENT_REVISION_SCHEMA",
    "PROTOCOL_VERSION",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "__version__",
    "provider_component_files",
    "provider_component_implementation_revisions",
    "provider_component_neutral_files",
    "provider_component_revision",
    "provider_component_revision_matches",
    "provider_component_revisions",
    "provider_revision",
]
