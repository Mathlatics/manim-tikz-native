"""Provider revision facade extended for split quadric implementation modules."""

from __future__ import annotations

from . import _version_impl as _impl
from ._version_impl import *  # noqa: F401,F403


_impl._COMPONENT_DEFINITIONS[COMPONENT_QUADRIC_VISIBILITY]["files"] = (
    *_impl._COMPONENT_DEFINITIONS[COMPONENT_QUADRIC_VISIBILITY]["files"],
    "@tool/polyhedron_visibility/quadrics/_section_compositing_impl.py",
)
_impl._COMPONENT_DEFINITIONS[COMPONENT_QUADRIC_MANIM]["files"] = (
    *_impl._COMPONENT_DEFINITIONS[COMPONENT_QUADRIC_MANIM]["files"],
    "@tool/polyhedron_visibility/quadrics/_manim_impl.py",
)
_impl._COMPONENT_NEUTRAL_FILES = frozenset(
    (*_impl._COMPONENT_NEUTRAL_FILES, "_version_impl.py")
)

# The depth-split plane outline changes both the renderer-neutral compositor and
# the Cairo binding. Freeze their new implementation identities so Provider
# metadata, cache invalidation, and public release constants agree.
_QUADRIC_VISIBILITY_IMPLEMENTATION_DIGEST = (
    "70c2c21e28f1cee03e834244006da779b4e4936dc7ec13aaa67e6f0005fea213"
)
_QUADRIC_MANIM_IMPLEMENTATION_DIGEST = (
    "c6b0d8517089b60f5ec6156ab1ee4bcbfb2131629fba31a136ca6f5b3512c63c"
)
QUADRIC_VISIBILITY_REVISION = (
    "source-sha256:" + _QUADRIC_VISIBILITY_IMPLEMENTATION_DIGEST
)
QUADRIC_MANIM_REVISION = "source-sha256:" + _QUADRIC_MANIM_IMPLEMENTATION_DIGEST

_impl.QUADRIC_VISIBILITY_REVISION = QUADRIC_VISIBILITY_REVISION
_impl.QUADRIC_MANIM_REVISION = QUADRIC_MANIM_REVISION
_impl._PUBLIC_0_1_COMPONENT_REVISIONS[COMPONENT_QUADRIC_VISIBILITY] = (
    QUADRIC_VISIBILITY_REVISION
)
_impl._PUBLIC_0_1_COMPONENT_REVISIONS[COMPONENT_QUADRIC_MANIM] = (
    QUADRIC_MANIM_REVISION
)
_impl._DECLARED_COMPONENT_REVISIONS[COMPONENT_QUADRIC_VISIBILITY] = (
    QUADRIC_VISIBILITY_REVISION
)
_impl._DECLARED_COMPONENT_REVISIONS[COMPONENT_QUADRIC_MANIM] = QUADRIC_MANIM_REVISION
_impl._DECLARED_IMPLEMENTATION_DIGESTS[COMPONENT_QUADRIC_VISIBILITY] = (
    _QUADRIC_VISIBILITY_IMPLEMENTATION_DIGEST
)
_impl._DECLARED_IMPLEMENTATION_DIGESTS[COMPONENT_QUADRIC_MANIM] = (
    _QUADRIC_MANIM_IMPLEMENTATION_DIGEST
)

# Definitions and frozen identities changed after the implementation module was
# imported. Clear every cached map so the first public query observes the full
# ownership graph and the updated fail-closed component revisions.
_impl.provider_revision.cache_clear()
_impl.provider_component_implementation_revisions.cache_clear()
_impl.provider_component_revisions.cache_clear()
_impl.provider_component_contract_revisions.cache_clear()


def __getattr__(name: str) -> object:
    return getattr(_impl, name)


__all__ = _impl.__all__
