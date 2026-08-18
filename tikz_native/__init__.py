"""Restricted TikZ-to-native-Manim compiler and visibility adapters.

The package intentionally supports a documented subset of TikZ. Unsupported
syntax is reported instead of being flattened into SVG or a generic VMobject.
"""

from .compatibility import audit_document_compatibility, load_subset_spec
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
from .fixed_view_renderer import NativeFixedViewRenderer
from .manim_renderer import NativeFigure, NativeManimRenderer
from .manim_renderer_3d import Native3DFigure, NativeManim3DRenderer
from .motion_runtime import (
    MOTION_SCHEMA,
    EllipseChordMetrics,
    MotionConfigError,
    MotionSpec,
    NativeMotionRuntime,
    ellipse_chord_metrics,
    load_motion_spec,
)
from .native_manim_codegen_3d_v3 import (
    NATIVE_MANIM_SOURCE_3D_V3_SCHEMA,
    NativeManimCodegen3DV3Error,
    generate_native_manim_source_3d_v3,
)
from .native_rig_2d import (
    NATIVE_RIG_2D_API_SCHEMA,
    NativeGeometryRig2D,
    NativeRig2D,
)
from .open_face_static_asset_3d import (
    TikzNativeOpenFaceStaticAsset3DError,
    bake_open_face_static_entry_3d,
    validate_open_face_static_asset_3d_contract,
)
from .open_face_visibility_3d_adapter import (
    TikzNativeOpenFaceVisibility3DAdapterError,
    TikzNativeOpenFaceVisibility3DAdapterResult,
    adapt_picture_open_face_visibility_3d,
)
from .open_face_visibility_3d_manim import (
    TikzNativeOpenFaceAutoOcclusion3D,
    bind_picture_open_face_visibility_3d,
)
from .polyhedron_visibility_3d_adapter import (
    TikzNativeVisibility3DAdapterError,
    TikzNativeVisibility3DAdapterResult,
    adapt_picture_visibility_3d,
)
from .polyhedron_visibility_3d_manim import (
    TikzNativeAutoOcclusion3D,
    bind_picture_visibility_3d,
)
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
    COMPONENT_ASSET_COMPILER,
    COMPONENT_EMBEDDED_MOTION_3D,
    COMPONENT_GEOMETRY_RIG_2D,
    COMPONENT_GEOMETRY_RIG_3D,
    COMPONENT_MOTION_PREVIEW_2D,
    COMPONENT_MOTION_PREVIEW_3D,
    COMPONENT_NATIVE_MANIM_SOURCE_2D,
    COMPONENT_NATIVE_MANIM_SOURCE_3D,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
    COMPONENT_NATIVE_RIG_2D,
    COMPONENT_OPEN_FACE_VISIBILITY,
    COMPONENT_POLYHEDRON_VISIBILITY,
    COMPONENT_REVISION_SCHEMA,
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
    PROTOCOL_VERSION,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    __version__,
    provider_component_revision,
    provider_component_revisions,
    provider_revision,
)

__all__ = [
    "ASSET_SCHEMA",
    "COMPONENT_ASSET_COMPILER",
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
    "COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D",
    "COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D",
    "COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D",
    "CompiledAsset",
    "DocumentSpec",
    "EllipseChordMetrics",
    "IntersectionSpec",
    "MOTION_SCHEMA",
    "MotionConfigError",
    "MotionSpec",
    "NATIVE_MANIM_SOURCE_3D_V3_SCHEMA",
    "NATIVE_RIG_2D_API_SCHEMA",
    "NamedPathSpec",
    "Native3DFigure",
    "NativeFigure",
    "NativeFixedViewRenderer",
    "NativeGeometryRig2D",
    "NativeManim3DRenderer",
    "NativeManimCodegen3DV3Error",
    "NativeManimRenderer",
    "NativeMotionRuntime",
    "NativeRig2D",
    "ObjectSpec",
    "OcclusionRelationSpec",
    "PROTOCOL_VERSION",
    "PictureSpec",
    "Projection3DSpec",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "TikzNativeAutoOcclusion3D",
    "TikzNativeCompiler",
    "TikzNativeOpenFaceAutoOcclusion3D",
    "TikzNativeOpenFaceStaticAsset3DError",
    "TikzNativeOpenFaceVisibility3DAdapterError",
    "TikzNativeOpenFaceVisibility3DAdapterResult",
    "TikzNativeProviderError",
    "TikzNativeVisibility3DAdapterError",
    "TikzNativeVisibility3DAdapterResult",
    "__version__",
    "adapt_picture_open_face_visibility_3d",
    "adapt_picture_visibility_3d",
    "audit_document_compatibility",
    "bake_open_face_static_entry_3d",
    "bind_picture_open_face_visibility_3d",
    "bind_picture_visibility_3d",
    "compile_asset",
    "compile_document",
    "ellipse_chord_metrics",
    "generate_native_manim_source_3d_v3",
    "instantiate_picture",
    "load_motion_spec",
    "load_subset_spec",
    "provider_component_revision",
    "provider_component_revisions",
    "provider_info",
    "provider_revision",
    "render_static_png",
    "validate_open_face_static_asset_3d_contract",
]
