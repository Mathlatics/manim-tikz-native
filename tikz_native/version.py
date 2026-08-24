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

# Definitions changed after the implementation module was imported. Clear all
# cached revision maps so the first public query sees the complete ownership
# graph and computes the new fail-closed component digests.
_impl.provider_revision.cache_clear()
_impl.provider_component_implementation_revisions.cache_clear()
_impl.provider_component_revisions.cache_clear()
_impl.provider_component_contract_revisions.cache_clear()


def __getattr__(name: str) -> object:
    return getattr(_impl, name)


__all__ = _impl.__all__
