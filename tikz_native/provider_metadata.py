from __future__ import annotations

"""Provider health metadata, kept outside asset implementation hashing.

Capabilities and authoring helpers can evolve without pretending that the
TikZ compiler or fixed-view renderer changed.  Each Bridge selects the
component revision that owns its operation while ``build_revision`` retains
the complete source-tree identity for diagnostics.
"""

import platform
import sys
from typing import Any

import manim

from .compatibility import load_subset_spec
from .version import (
    ASSET_SCHEMA,
    COMPONENT_ASSET_COMPILER,
    COMPONENT_CONTRACT_REVISION_SCHEMA,
    COMPONENT_REVISION_SCHEMA,
    PROTOCOL_VERSION,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    __version__,
    provider_component_contract_revisions,
    provider_component_revision,
    provider_component_revisions,
    provider_revision,
)


def provider_info(
    *,
    revision_component: str = COMPONENT_ASSET_COMPILER,
) -> dict[str, Any]:
    subset = load_subset_spec()
    component_revisions = provider_component_revisions()
    component_contract_revisions = provider_component_contract_revisions()
    return {
        "name": "manim-tikz-native",
        "version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "request_schema": REQUEST_SCHEMA,
        "response_schema": RESPONSE_SCHEMA,
        "asset_schema": ASSET_SCHEMA,
        # ``revision`` remains the operation-facing compatibility identity for
        # old clients.  The complete source-tree hash is separately visible as
        # ``build_revision`` and never gates unrelated persisted assets.
        "revision": provider_component_revision(revision_component),
        "revision_component": revision_component,
        "build_revision": provider_revision(),
        "component_contract_revision_schema": (
            COMPONENT_CONTRACT_REVISION_SCHEMA
        ),
        "component_contract_revisions": component_contract_revisions,
        "component_revision_schema": COMPONENT_REVISION_SCHEMA,
        "component_revisions": component_revisions,
        # Explicit aliases make the new semantics clear while the original
        # names stay available to old Host versions.
        "component_render_revision_schema": COMPONENT_REVISION_SCHEMA,
        "component_render_revisions": component_revisions,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "manim_version": manim.__version__,
        "subset_versions": [subset["subset_version"]],
        "capabilities": {
            "compile_2d": True,
            "compile_3d_fixed_view": True,
            "semantic_object_ids": True,
            "semantic_animation_layers": True,
            "render_static": True,
            "native_rig_2d_authoring_v1": True,
            "native_manim_source_2d_v1": True,
            "native_manim_source_3d_v1": True,
            "native_manim_source_3d_v2": True,
            "native_manim_source_3d_v3": True,
            "polyhedron_visibility_parallel_v1": True,
            "tikz_polyhedron_visibility_3d_v1": True,
            "open_convex_face_visibility_parallel_v1": True,
            "tikz_open_face_visibility_3d_v1": True,
            "tikz_open_face_static_asset_3d_v1": True,
            "provider_component_revisions_v1": True,
            "provider_component_contract_revisions_v1": True,
            "dynamic_camera_in_fixed_view": False,
        },
    }


__all__ = ["provider_info"]
