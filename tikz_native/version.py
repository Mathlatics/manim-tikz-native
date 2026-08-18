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
COMPONENT_FACE_DEPTH_CUE_3D = "face_depth_cue_3d"
COMPONENT_CONVEX_SECTION_3D = "convex_section_3d"
COMPONENT_OPEN_FACE_VISIBILITY = "open_face_visibility"
COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D = "tikz_polyhedron_visibility_3d"
COMPONENT_TIKZ_CONVEX_SECTION_3D = "tikz_convex_section_3d"
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
# The public 0.1.0 release replaces one machine-specific TeX font template with
# portable TeX Live filenames.  Persisted schemas remain contract-v1, but every
# render/cache component that depends on that template intentionally receives a
# new identity.  The pure visibility solvers keep their already verified bytes.
_PUBLIC_0_1_COMPONENT_REVISIONS: Final[dict[str, str]] = {
    COMPONENT_ASSET_COMPILER: "source-sha256:ab7f963405fc04de38e394aaf6a87d83535dcabcf07c5fb55ef3965ff0fe24a8",
    COMPONENT_GEOMETRY_RIG_2D: "source-sha256:b1fc6424921a77536a7ac03b81bb07e7da2fd94280151e6d780f8674a91c87b5",
    COMPONENT_NATIVE_MANIM_SOURCE_2D: "source-sha256:67595e78d46c2285fa7f2de6e2e0516f69d1403e59675ea5015d5c6eb1e925cb",
    COMPONENT_NATIVE_RIG_2D: "source-sha256:9eb920be8b6368cc4eb658523084753e98657049dac37a7e1071ff32b4079756",
    COMPONENT_MOTION_PREVIEW_2D: "source-sha256:257e982c33a4ab94dc5104e78c3495940b54a626df063a0bb05615ce844d7023",
    COMPONENT_GEOMETRY_RIG_3D: "source-sha256:3f580038dcc69ff0e96eb0c27e4fa55ac38a4fa1051114c61d09eb4b988bb3f0",
    COMPONENT_NATIVE_MANIM_SOURCE_3D: "source-sha256:374519e01a15a7bd7d99fb65dc36e882d938029dcdb728cdc27a027930ec4b20",
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: "source-sha256:5e18054bb19a47e03eea9966a5783928f5bac81e1fa649567d5739f662a702a8",
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: "source-sha256:8d711b47f68c4171177b4ee7d651b0fb1c167d4e6eb59803f80d1ea6d905978b",
    COMPONENT_EMBEDDED_MOTION_3D: "source-sha256:73b1c79e7d9178bb3be451e6b94977ba05361f2c187b6d8cf9df8dbb47c5ae59",
    COMPONENT_MOTION_PREVIEW_3D: "source-sha256:6958a1284eb328e1dc07e3f1ee7658b9525f810a1f386449ce02088b05e93cbc",
    COMPONENT_POLYHEDRON_VISIBILITY: "source-sha256:aa45310ff3c70ac1922ddf61b457cafeb789f9011ec67069b70c23d63fb3a8ae",
    COMPONENT_FACE_DEPTH_CUE_3D: "source-sha256:be2a87b144147f49ed7f47c4955c366d00ad48b5cae98ff58e55ae63570da0fa",
    COMPONENT_CONVEX_SECTION_3D: "source-sha256:03581834d1a596f4e678153cf4780329e5c7f424031b91ed20e8981f340d3a4f",
    COMPONENT_OPEN_FACE_VISIBILITY: "source-sha256:8c831f441d21e2ceb39aed78ac3428936ac50fe86fa726ea548a52a4bf426341",
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: "source-sha256:76bf703ef380738af5ee0f463e9d2a43f4537d8d156d2af372953457dca6cc48",
    COMPONENT_TIKZ_CONVEX_SECTION_3D: "source-sha256:30d0a7bfc0a9a838975eeeedec5a0ca4feb003f227590d988d5c75af761aba2e",
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: "source-sha256:adb3cfad37ceff50c968984fbc779985c70c5a7cba583fd44ce1af66a2955e58",
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: "source-sha256:a11291d775923e57cfe36a3debb6ab0813098a11193093a0901ba82356fba2d2",
}


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
    COMPONENT_FACE_DEPTH_CUE_3D: {
        "dependencies": (COMPONENT_POLYHEDRON_VISIBILITY,),
        "files": (
            "@tool/polyhedron_visibility/depth_cue/__init__.py",
            "@tool/polyhedron_visibility/depth_cue/contract.py",
            "@tool/polyhedron_visibility/depth_cue/manim.py",
            "@tool/polyhedron_visibility/depth_cue/solver.py",
        ),
    },
    COMPONENT_CONVEX_SECTION_3D: {
        "dependencies": (
            COMPONENT_POLYHEDRON_VISIBILITY,
            COMPONENT_FACE_DEPTH_CUE_3D,
        ),
        "files": (
            "@tool/polyhedron_visibility/sections/__init__.py",
            "@tool/polyhedron_visibility/sections/authoring.py",
            "@tool/polyhedron_visibility/sections/compositing.py",
            "@tool/polyhedron_visibility/sections/compositing_manim.py",
            "@tool/polyhedron_visibility/sections/contract.py",
            "@tool/polyhedron_visibility/sections/manim.py",
            "@tool/polyhedron_visibility/sections/solver.py",
            "@tool/polyhedron_visibility/sections/trace.py",
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
    COMPONENT_TIKZ_CONVEX_SECTION_3D: {
        "dependencies": (
            COMPONENT_ASSET_COMPILER,
            COMPONENT_CONVEX_SECTION_3D,
            COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
        ),
        "files": ("convex_section_3d_manim.py",),
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


_DECLARED_COMPONENT_REVISIONS: Final[dict[str, str]] = dict(
    _PUBLIC_0_1_COMPONENT_REVISIONS
)


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
        "ab7f963405fc04de38e394aaf6a87d83535dcabcf07c5fb55ef3965ff0fe24a8"
    ),
    COMPONENT_GEOMETRY_RIG_2D: (
        "b1fc6424921a77536a7ac03b81bb07e7da2fd94280151e6d780f8674a91c87b5"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_2D: (
        "67595e78d46c2285fa7f2de6e2e0516f69d1403e59675ea5015d5c6eb1e925cb"
    ),
    COMPONENT_NATIVE_RIG_2D: (
        "9eb920be8b6368cc4eb658523084753e98657049dac37a7e1071ff32b4079756"
    ),
    COMPONENT_MOTION_PREVIEW_2D: (
        "257e982c33a4ab94dc5104e78c3495940b54a626df063a0bb05615ce844d7023"
    ),
    COMPONENT_GEOMETRY_RIG_3D: (
        "3f580038dcc69ff0e96eb0c27e4fa55ac38a4fa1051114c61d09eb4b988bb3f0"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D: (
        "374519e01a15a7bd7d99fb65dc36e882d938029dcdb728cdc27a027930ec4b20"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: (
        "5e18054bb19a47e03eea9966a5783928f5bac81e1fa649567d5739f662a702a8"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: (
        "8d711b47f68c4171177b4ee7d651b0fb1c167d4e6eb59803f80d1ea6d905978b"
    ),
    COMPONENT_EMBEDDED_MOTION_3D: (
        "73b1c79e7d9178bb3be451e6b94977ba05361f2c187b6d8cf9df8dbb47c5ae59"
    ),
    COMPONENT_MOTION_PREVIEW_3D: (
        "6958a1284eb328e1dc07e3f1ee7658b9525f810a1f386449ce02088b05e93cbc"
    ),
    COMPONENT_POLYHEDRON_VISIBILITY: (
        "aa45310ff3c70ac1922ddf61b457cafeb789f9011ec67069b70c23d63fb3a8ae"
    ),
    COMPONENT_FACE_DEPTH_CUE_3D: (
        "be2a87b144147f49ed7f47c4955c366d00ad48b5cae98ff58e55ae63570da0fa"
    ),
    COMPONENT_CONVEX_SECTION_3D: (
        "03581834d1a596f4e678153cf4780329e5c7f424031b91ed20e8981f340d3a4f"
    ),
    COMPONENT_OPEN_FACE_VISIBILITY: (
        "8c831f441d21e2ceb39aed78ac3428936ac50fe86fa726ea548a52a4bf426341"
    ),
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: (
        "76bf703ef380738af5ee0f463e9d2a43f4537d8d156d2af372953457dca6cc48"
    ),
    COMPONENT_TIKZ_CONVEX_SECTION_3D: (
        "30d0a7bfc0a9a838975eeeedec5a0ca4feb003f227590d988d5c75af761aba2e"
    ),
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: (
        "adb3cfad37ceff50c968984fbc779985c70c5a7cba583fd44ce1af66a2955e58"
    ),
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: (
        "a11291d775923e57cfe36a3debb6ab0813098a11193093a0901ba82356fba2d2"
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
    "COMPONENT_CONVEX_SECTION_3D",
    "COMPONENT_CONTRACT_REVISION_SCHEMA",
    "COMPONENT_EMBEDDED_MOTION_3D",
    "COMPONENT_FACE_DEPTH_CUE_3D",
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
    "COMPONENT_TIKZ_CONVEX_SECTION_3D",
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
