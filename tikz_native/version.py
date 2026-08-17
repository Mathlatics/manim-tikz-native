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
COMPONENT_CONTRACT_REVISION_SCHEMA = (
    "tikz-native-component-contract-revisions/v1"
)
COMPONENT_ASSET_COMPILER = "asset_compiler"
COMPONENT_GEOMETRY_RIG_2D = "geometry_rig_2d"
COMPONENT_NATIVE_MANIM_SOURCE_2D = "native_manim_source_2d"
COMPONENT_NATIVE_RIG_2D = "native_rig_2d"
COMPONENT_MOTION_PREVIEW_2D = "motion_preview_2d"
COMPONENT_GEOMETRY_RIG_3D = "geometry_rig_3d"
COMPONENT_NATIVE_MANIM_SOURCE_3D = "native_manim_source_3d"
COMPONENT_NATIVE_MANIM_SOURCE_3D_V2 = "native_manim_source_3d_v2"
COMPONENT_NATIVE_MANIM_SOURCE_3D_V3 = "native_manim_source_3d_v3"
COMPONENT_EMBEDDED_MOTION_3D = "embedded_motion_3d"
COMPONENT_MOTION_PREVIEW_3D = "motion_preview_3d"
COMPONENT_POLYHEDRON_VISIBILITY = "polyhedron_visibility"
COMPONENT_OPEN_FACE_VISIBILITY = "open_face_visibility"
COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D = "tikz_polyhedron_visibility_3d"
COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D = "tikz_open_face_visibility_3d"
COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D = "tikz_open_face_static_asset_3d"


# Existing manifests are relative to ``tikz_native/``.  New independent
# runtime packages can opt into Provider-tool-root addressing without changing
# the bytes that define any already frozen component digest.
_TOOL_ROOT_FILE_PREFIX: Final = "@tool/"


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
_NATIVE_SOURCE_3D_270A: Final = (
    "source-sha256:270a1d6d04659bb16f1ce2b5239fda2c19315691e103c12491e4eed84b460a45"
)
_NATIVE_SOURCE_3D_V2_26D7: Final = (
    "source-sha256:26d702598f6300ece385972271346a571e9fb28b273732584ce14fca98445769"
)
_NATIVE_SOURCE_3D_V3_4960: Final = (
    "source-sha256:7c368f4d7681ebe716d4cf2e0a73d64822ec5a49cdd64b7e67b9f5bf4689a8ca"
)
_POLYHEDRON_VISIBILITY_AA45: Final = (
    "source-sha256:aa45310ff3c70ac1922ddf61b457cafeb789f9011ec67069b70c23d63fb3a8ae"
)
_TIKZ_POLYHEDRON_VISIBILITY_3D_F0E4: Final = (
    "source-sha256:f0e495883f71691f8b1ca3728ab78953ab8132b3e316a7a911a6e345f74fcd3c"
)
_OPEN_FACE_VISIBILITY_REVISION: Final = (
    "source-sha256:8af538cfe89b38936ee374c7c71932398986bdc17732de45f59b33f6719d71a8"
)
_TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION: Final = (
    "source-sha256:e5854f0f7bcd83f2a86cbbf53ab8237ad07471c84668ee3f13af695c367d4a29"
)
_TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION: Final = (
    "source-sha256:e46f69d6a2167da0c3aa0dd8e66f9762865a4627f4acd987ff9c58489ce5eea2"
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
    COMPONENT_NATIVE_MANIM_SOURCE_3D: {
        "dependencies": (COMPONENT_GEOMETRY_RIG_3D,),
        "files": ("native_manim_codegen_3d.py",),
    },
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: {
        "dependencies": (COMPONENT_NATIVE_MANIM_SOURCE_3D,),
        "files": ("native_manim_codegen_3d_v2.py",),
    },
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: {
        "dependencies": (
            COMPONENT_NATIVE_MANIM_SOURCE_3D_V2,
            COMPONENT_OPEN_FACE_VISIBILITY,
            COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
        ),
        "files": (
            "geometry_rig_3d_source_v3_bridge.py",
            "native_manim_codegen_3d_v3.py",
            "schemas/geometry-rig-3d-source-v3-bridge-request-v1.schema.json",
            "schemas/geometry-rig-3d-source-v3-bridge-response-v1.schema.json",
            "schemas/geometry-rig-3d-source-v3-v1.schema.json",
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
    COMPONENT_POLYHEDRON_VISIBILITY: {
        "dependencies": (),
        "files": (
            "@tool/polyhedron_visibility/__init__.py",
            "@tool/polyhedron_visibility/api.py",
            "@tool/polyhedron_visibility/authoring.py",
            "@tool/polyhedron_visibility/binding.py",
            "@tool/polyhedron_visibility/contract.py",
            "@tool/polyhedron_visibility/parallel_solver.py",
            "@tool/polyhedron_visibility/style.py",
            "@tool/polyhedron_visibility/trace.py",
        ),
    },
    COMPONENT_OPEN_FACE_VISIBILITY: {
        "dependencies": (COMPONENT_POLYHEDRON_VISIBILITY,),
        "files": (
            "@tool/polyhedron_visibility/open_faces/__init__.py",
            "@tool/polyhedron_visibility/open_faces/authoring.py",
            "@tool/polyhedron_visibility/open_faces/contract.py",
            "@tool/polyhedron_visibility/open_faces/manim.py",
            "@tool/polyhedron_visibility/open_faces/solver.py",
            "@tool/polyhedron_visibility/open_faces/trace.py",
        ),
    },
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: {
        "dependencies": (
            COMPONENT_ASSET_COMPILER,
            COMPONENT_POLYHEDRON_VISIBILITY,
        ),
        "files": (
            "polyhedron_visibility_3d_adapter.py",
            "polyhedron_visibility_3d_manim.py",
        ),
    },
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: {
        "dependencies": (
            COMPONENT_ASSET_COMPILER,
            COMPONENT_OPEN_FACE_VISIBILITY,
            COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
        ),
        "files": (
            "open_face_visibility_3d_adapter.py",
            "open_face_visibility_3d_manim.py",
        ),
    },
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: {
        "dependencies": (
            COMPONENT_ASSET_COMPILER,
            COMPONENT_OPEN_FACE_VISIBILITY,
            COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
        ),
        "files": ("open_face_static_asset_3d.py",),
    },
}


_DECLARED_COMPONENT_REVISIONS: Final[dict[str, str]] = {
    COMPONENT_ASSET_COMPILER: _LEGACY_6920,
    COMPONENT_GEOMETRY_RIG_2D: _NATIVE_SOURCE_01DF,
    COMPONENT_NATIVE_MANIM_SOURCE_2D: _NATIVE_SOURCE_01DF,
    COMPONENT_NATIVE_RIG_2D: _LEGACY_6920,
    COMPONENT_MOTION_PREVIEW_2D: _LEGACY_6920,
    COMPONENT_GEOMETRY_RIG_3D: _LEGACY_6920,
    COMPONENT_NATIVE_MANIM_SOURCE_3D: _NATIVE_SOURCE_3D_270A,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: _NATIVE_SOURCE_3D_V2_26D7,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: _NATIVE_SOURCE_3D_V3_4960,
    COMPONENT_EMBEDDED_MOTION_3D: _LEGACY_6920,
    COMPONENT_MOTION_PREVIEW_3D: _LEGACY_6920,
    COMPONENT_POLYHEDRON_VISIBILITY: _POLYHEDRON_VISIBILITY_AA45,
    COMPONENT_OPEN_FACE_VISIBILITY: _OPEN_FACE_VISIBILITY_REVISION,
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: (
        _TIKZ_POLYHEDRON_VISIBILITY_3D_F0E4
    ),
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: (
        _TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION
    ),
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: (
        _TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION
    ),
}


# Contract revisions describe the persisted author-data ABI, not the bytes of
# the current implementation.  They must change only when an already-saved
# asset or Native Clip can no longer be read safely.  In contrast,
# ``provider_component_revisions()`` remains the render/cache identity and may
# change whenever implementation bytes change.
_DECLARED_COMPONENT_CONTRACT_VERSIONS: Final[dict[str, int]] = {
    component: 1 for component in _COMPONENT_DEFINITIONS
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
        "8063037e999aaaa269d9e37c3a477beb7c7f5dcdff5adb6df71834362d57072e"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D: (
        "624ffd1b2d508aac64f90e83949efe8a46606c9ca735865b9ff4931a419cbd69"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: (
        "26d702598f6300ece385972271346a571e9fb28b273732584ce14fca98445769"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: (
        "7c368f4d7681ebe716d4cf2e0a73d64822ec5a49cdd64b7e67b9f5bf4689a8ca"
    ),
    COMPONENT_EMBEDDED_MOTION_3D: (
        "f47e6a693fdf4dd19f9487b7762a637bcec3ec981b8381a1eb14913e42e5fbcd"
    ),
    COMPONENT_MOTION_PREVIEW_3D: (
        "ad0ae4862e87fd7fcee52d2dce6cf87d5fb69c35babb2d2f457628b1f1d715b2"
    ),
    COMPONENT_POLYHEDRON_VISIBILITY: (
        "aa45310ff3c70ac1922ddf61b457cafeb789f9011ec67069b70c23d63fb3a8ae"
    ),
    COMPONENT_OPEN_FACE_VISIBILITY: (
        "8af538cfe89b38936ee374c7c71932398986bdc17732de45f59b33f6719d71a8"
    ),
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: (
        "f0e495883f71691f8b1ca3728ab78953ab8132b3e316a7a911a6e345f74fcd3c"
    ),
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: (
        "e5854f0f7bcd83f2a86cbbf53ab8237ad07471c84668ee3f13af695c367d4a29"
    ),
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: (
        "e46f69d6a2167da0c3aa0dd8e66f9762865a4627f4acd987ff9c58489ce5eea2"
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
    tool_root = package_root.parent
    candidates = [
        (path.relative_to(package_root).as_posix(), path)
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".json"}
    ]
    visibility_root = tool_root / "polyhedron_visibility"
    candidates.extend(
        (
            _TOOL_ROOT_FILE_PREFIX + path.relative_to(tool_root).as_posix(),
            path,
        )
        for path in visibility_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".json"}
    )
    digest = hashlib.sha256()
    for relative_text, path in sorted(candidates, key=lambda item: item[0]):
        relative = relative_text.encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"source-sha256:{digest.hexdigest()}"


def _component_file_path(relative_text: str) -> Path:
    package_root = Path(__file__).resolve().parent
    if relative_text.startswith(_TOOL_ROOT_FILE_PREFIX):
        tool_relative = relative_text.removeprefix(_TOOL_ROOT_FILE_PREFIX)
        relative = Path(tool_relative)
        if not tool_relative or relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"invalid TikZ Native tool-root component path: {relative_text!r}"
            )
        path = package_root.parent / relative
    else:
        path = package_root / relative_text
    return path


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
    for relative_text in sorted(definition["files"]):
        path = _component_file_path(relative_text)
        if not path.is_file():
            raise RuntimeError(
                f"TikZ Native component {component!r} is missing {relative_text!r}"
            )
        relative_bytes = relative_text.encode("utf-8")
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


@lru_cache(maxsize=1)
def provider_component_contract_revisions() -> dict[str, str]:
    return {
        component: f"tikz-native-contract:{component}/v{version}"
        for component, version in _DECLARED_COMPONENT_CONTRACT_VERSIONS.items()
    }


def provider_component_contract_revision(component: str) -> str:
    try:
        return provider_component_contract_revisions()[component]
    except KeyError as exc:
        raise ValueError(f"unknown TikZ Native component: {component!r}") from exc


def provider_component_contract_revision_matches(
    component: str, recorded: object
) -> bool:
    value = str(recorded or "").strip()
    if not value:
        return False
    return value == provider_component_contract_revision(component)


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
    "COMPONENT_CONTRACT_REVISION_SCHEMA",
    "COMPONENT_EMBEDDED_MOTION_3D",
    "COMPONENT_GEOMETRY_RIG_2D",
    "COMPONENT_GEOMETRY_RIG_3D",
    "COMPONENT_MOTION_PREVIEW_2D",
    "COMPONENT_MOTION_PREVIEW_3D",
    "COMPONENT_NATIVE_MANIM_SOURCE_2D",
    "COMPONENT_NATIVE_MANIM_SOURCE_3D",
    "COMPONENT_NATIVE_MANIM_SOURCE_3D_V2",
    "COMPONENT_NATIVE_MANIM_SOURCE_3D_V3",
    "COMPONENT_NATIVE_RIG_2D",
    "COMPONENT_OPEN_FACE_VISIBILITY",
    "COMPONENT_POLYHEDRON_VISIBILITY",
    "COMPONENT_REVISION_SCHEMA",
    "COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D",
    "COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D",
    "COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D",
    "PROTOCOL_VERSION",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "__version__",
    "provider_component_files",
    "provider_component_contract_revision",
    "provider_component_contract_revision_matches",
    "provider_component_contract_revisions",
    "provider_component_implementation_revisions",
    "provider_component_neutral_files",
    "provider_component_revision",
    "provider_component_revision_matches",
    "provider_component_revisions",
    "provider_revision",
]
