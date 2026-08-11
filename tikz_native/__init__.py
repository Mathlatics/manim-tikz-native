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
from .motion_runtime import (
    MOTION_SCHEMA,
    EllipseChordMetrics,
    MotionConfigError,
    MotionSpec,
    NativeMotionRuntime,
    ellipse_chord_metrics,
    load_motion_spec,
)
from .native_rig_2d import (
    NATIVE_RIG_2D_API_SCHEMA,
    NativeGeometryRig2D,
    NativeRig2D,
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
    "EllipseChordMetrics",
    "MOTION_SCHEMA",
    "MotionConfigError",
    "MotionSpec",
    "NATIVE_RIG_2D_API_SCHEMA",
    "NativeGeometryRig2D",
    "NativeMotionRuntime",
    "NativeRig2D",
    "PROTOCOL_VERSION",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "TikzNativeProviderError",
    "TikzNativeCompiler",
    "__version__",
    "audit_document_compatibility",
    "compile_asset",
    "compile_document",
    "ellipse_chord_metrics",
    "instantiate_picture",
    "load_subset_spec",
    "load_motion_spec",
    "provider_info",
    "provider_revision",
    "render_static_png",
]
