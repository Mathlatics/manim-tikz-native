"""Restricted TikZ-to-native-Manim compiler.

The package intentionally supports a documented subset of TikZ.  Unsupported
syntax is reported instead of being flattened into SVG or a generic VMobject.
"""

from .compiler import (
    DocumentSpec,
    IntersectionSpec,
    NamedPathSpec,
    OcclusionRelationSpec,
    ObjectSpec,
    PictureSpec,
    Projection3DSpec,
    TikzNativeCompiler,
    compile_document,
)
from .compatibility import (
    audit_document_compatibility,
    load_subset_spec,
)
from .fixed_view_renderer import NativeFixedViewRenderer
from .provider import (
    CompiledAsset,
    TikzNativeProviderError,
    compile_asset,
    instantiate_picture,
    provider_info,
    render_static_png,
)
from .version import (
    ASSET_SCHEMA,
    PROTOCOL_VERSION,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    __version__,
    provider_revision,
)

__all__ = [
    "DocumentSpec",
    "IntersectionSpec",
    "NamedPathSpec",
    "NativeFixedViewRenderer",
    "OcclusionRelationSpec",
    "ObjectSpec",
    "PictureSpec",
    "Projection3DSpec",
    "ASSET_SCHEMA",
    "CompiledAsset",
    "PROTOCOL_VERSION",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "TikzNativeProviderError",
    "TikzNativeCompiler",
    "__version__",
    "audit_document_compatibility",
    "compile_asset",
    "compile_document",
    "instantiate_picture",
    "load_subset_spec",
    "provider_info",
    "provider_revision",
    "render_static_png",
]
