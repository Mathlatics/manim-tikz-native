from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Final


__version__ = "0.1.1"
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
COMPONENT_COPY_IDENTITY_HANDOFF = "copy_identity_handoff"
COMPONENT_DERIVED_DIHEDRAL_VISIBILITY = "derived_dihedral_visibility"
COMPONENT_OPEN_FACE_VISIBILITY = "open_face_visibility"
COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING = "open_face_unified_compositing"
COMPONENT_MANAGED_PAINTER_BAND = "managed_painter_band"
COMPONENT_OPEN_FACE_UNIFIED_MANIM = "open_face_unified_manim"
COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D = "tikz_polyhedron_visibility_3d"
COMPONENT_TIKZ_CONVEX_SECTION_3D = "tikz_convex_section_3d"
COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D = "tikz_open_face_visibility_3d"
COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D = "tikz_open_face_static_asset_3d"
COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D = (
    "generated_open_face_visibility_3d"
)
COMPONENT_SOURCE_PROJECT_BUILD = "source_project_build"
COMPONENT_QUADRIC_GEOMETRY = "quadric_geometry"
COMPONENT_QUADRIC_VISIBILITY = "quadric_visibility"
COMPONENT_QUADRIC_MANIM = "quadric_manim"


# Existing manifests are relative to ``tikz_native/``.  New independent
# runtime packages can opt into Provider-tool-root addressing without changing
# the bytes that define any already frozen component digest.
_TOOL_ROOT_FILE_PREFIX: Final = "@tool/"


# A component revision is the build identity at which that compatibility
# surface last changed.  The paired implementation digest makes this
# fail-closed: editing one of the declared files without deliberately updating
# its component contract immediately yields a fresh component-sha256 revision.
#
# Persisted schemas remain contract-v1, while each render/cache identity is
# refreshed deliberately whenever the owned implementation bytes or one of
# their declared dependencies changes.  The paired digest table below makes an
# unrecorded source edit fail closed to a fresh component-sha256 identity.
_PUBLIC_0_1_COMPONENT_REVISIONS: Final[dict[str, str]] = {
    COMPONENT_ASSET_COMPILER: "source-sha256:5599d2a0f6e6c5e561c6b9ea5ee420eae1fc4f21b88d20b13836170cafec54ff",
    COMPONENT_GEOMETRY_RIG_2D: "source-sha256:2a845be9cdfeffba8c345aaa346b4ac84127ace890dac5d9eda71f1d199f536e",
    COMPONENT_NATIVE_MANIM_SOURCE_2D: "source-sha256:8d9abee202dd3246e97f05cb5911f4b9b7cea7a5891fa695fdc2a11cb68ddea4",
    COMPONENT_NATIVE_RIG_2D: "source-sha256:8b48e5128a90b5ce5320e12e7e6ef3bfe511105a3a907081e3e4082c5c34f816",
    COMPONENT_MOTION_PREVIEW_2D: "source-sha256:567ba36480e1fc41a0c28d78a2f02618c776a326e16cce88910e40744221bfab",
    COMPONENT_GEOMETRY_RIG_3D: "source-sha256:3673185517871945742b8eea609a99c67f34b90d718d0ca10e2a2d407d4af534",
    COMPONENT_NATIVE_MANIM_SOURCE_3D: "source-sha256:177257d74cd12ddcb7b87920610f9a300cda8dd2bd52604f9007bc838d7d5278",
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: "source-sha256:3783dd61ed9c86d89f0b45c403c3af4a9b5cf5181bb7d771133cfa3cb35a7912",
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: "source-sha256:6050f4132e08d8424ecfe596684b78cb625f806a47c903e691999cf796a616fd",
    COMPONENT_EMBEDDED_MOTION_3D: "source-sha256:c8eaf866ddb03e78ac5a6b2ee7953fc8a1663cfe11299298aac72b9761cdbfa6",
    COMPONENT_MOTION_PREVIEW_3D: "source-sha256:5ac22e8e5b01c4ce5449f8eb1f314f436e23d6b4e992444084de905d51e02d4b",
    COMPONENT_POLYHEDRON_VISIBILITY: "source-sha256:745b8cb5b9cb832b4ce6c831094d961cbec7d506ec9110dd4f47e042b7a0799b",
    COMPONENT_FACE_DEPTH_CUE_3D: "source-sha256:65b662f5503915686c0a2b9829f6dbb04b0ca0cda74bb049adde3442898001ff",
    COMPONENT_CONVEX_SECTION_3D: "source-sha256:5cfd664e136caf4f68876ac76f81674d99c15b3b275c3859a84c174d738729b5",
    COMPONENT_COPY_IDENTITY_HANDOFF: "source-sha256:bf8aa2d0fe3ec9921320305279f2e23c8ab71d68b5613d19d19f467326d293b7",
    COMPONENT_DERIVED_DIHEDRAL_VISIBILITY: "source-sha256:5145d018dc54d5dba7e7b727d60ed6106360b9e6889488029080065ed6ee6b29",
    COMPONENT_OPEN_FACE_VISIBILITY: "source-sha256:0a23a2cc27ac2a82338608a0a5ab374c343e3215ee4c162e8d573e5baced008c",
    COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING: "source-sha256:c985c56a84b6414e5bcdcbc47ec522e84712726d123b0e56c3d7c2cbe630b7c9",
    COMPONENT_MANAGED_PAINTER_BAND: "source-sha256:6cfc31d3b50aa4768d17eaf10812ab6dc4b253227cc1d28a2c6bd3734b84b827",
    COMPONENT_OPEN_FACE_UNIFIED_MANIM: "source-sha256:b5859b4c2f70893b31bf9897c7a868138fb8d00aa2e1f3e1cceeccbd8421c6d5",
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: "source-sha256:60658221fd42359fb23310004b8a5be89afbb4c8fe1036ecec80242189e0bc2a",
    COMPONENT_TIKZ_CONVEX_SECTION_3D: "source-sha256:22804f08f377e20a28bd5cf07a396bdd105578e44e93bb49746c771d57bd51d8",
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: "source-sha256:0872cb5b5c3c7566a22816701b4ea59c51f3ab5c0f3d6ff1ac77bb6543a73d60",
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: "source-sha256:f74e221297d3444a17b9165ae9759ae41ca4cf7bda013333e28ab3b9d157a54e",
    COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D: "source-sha256:d75c0f26754b524da806a747ac91d43b8dec444343a8762fb9f6fe57fb81278b",
    COMPONENT_SOURCE_PROJECT_BUILD: "source-sha256:00579e342012a96443c098c7004be672d265d32fe6f82d66b0c6b11ba64305b7",
    COMPONENT_QUADRIC_GEOMETRY: "source-sha256:2d70a1abb2f75a76896a072808fb00f8fe2df91f70b98f0fa2b65fe90a21a62c",
    COMPONENT_QUADRIC_VISIBILITY: "source-sha256:e7f7a2a5dd28370daa45fb71404bc9b8866285f506eb37d828980908d5271262",
    COMPONENT_QUADRIC_MANIM: "source-sha256:e73c5aa28575265b5dfb06d612a007ca54991eb0694dd000a05606a99633423d",
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
            "@tool/polyhedron_visibility/geometry.py",
            "@tool/polyhedron_visibility/topology.py",
            "@tool/polyhedron_visibility/visibility.py",
            "@tool/polyhedron_visibility/compositor.py",
            "@tool/polyhedron_visibility/kernel.py",
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
    COMPONENT_COPY_IDENTITY_HANDOFF: {
        "dependencies": (),
        "files": (
            "@tool/polyhedron_visibility/copy_handoff/__init__.py",
            "@tool/polyhedron_visibility/copy_handoff/contract.py",
            "@tool/polyhedron_visibility/copy_handoff/solver.py",
        ),
    },
    COMPONENT_MANAGED_PAINTER_BAND: {
        "dependencies": (COMPONENT_POLYHEDRON_VISIBILITY,),
        "files": (
            "@tool/polyhedron_visibility/painter_band.py",
        ),
    },
    COMPONENT_DERIVED_DIHEDRAL_VISIBILITY: {
        "dependencies": (
            COMPONENT_POLYHEDRON_VISIBILITY,
            COMPONENT_CONVEX_SECTION_3D,
            COMPONENT_COPY_IDENTITY_HANDOFF,
            COMPONENT_MANAGED_PAINTER_BAND,
        ),
        "files": (
            "@tool/polyhedron_visibility/dihedral_extraction/__init__.py",
            "@tool/polyhedron_visibility/dihedral_extraction/authoring.py",
            "@tool/polyhedron_visibility/dihedral_extraction/base_plane.py",
            "@tool/polyhedron_visibility/dihedral_extraction/compositing.py",
            "@tool/polyhedron_visibility/dihedral_extraction/compositing_manim.py",
            "@tool/polyhedron_visibility/dihedral_extraction/contract.py",
            "@tool/polyhedron_visibility/dihedral_extraction/manim.py",
            "@tool/polyhedron_visibility/dihedral_extraction/solver.py",
            "@tool/polyhedron_visibility/dihedral_extraction/trace.py",
            "@tool/polyhedron_visibility/dihedral_extraction/unified_compositing.py",
            "@tool/polyhedron_visibility/dihedral_extraction/unified_compositing_manim.py",
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
    COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING: {
        "dependencies": (
            COMPONENT_POLYHEDRON_VISIBILITY,
            COMPONENT_OPEN_FACE_VISIBILITY,
        ),
        "files": (
            "@tool/polyhedron_visibility/path_compositing.py",
            "@tool/polyhedron_visibility/open_faces/unified_contract.py",
            "@tool/polyhedron_visibility/open_faces/unified_fragments.py",
            "@tool/polyhedron_visibility/open_faces/unified_compositing.py",
        ),
    },
    COMPONENT_OPEN_FACE_UNIFIED_MANIM: {
        "dependencies": (
            COMPONENT_POLYHEDRON_VISIBILITY,
            COMPONENT_OPEN_FACE_VISIBILITY,
            COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING,
            COMPONENT_MANAGED_PAINTER_BAND,
        ),
        "files": (
            "@tool/polyhedron_visibility/open_faces/unified_manim.py",
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
    COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D: {
        "dependencies": (
            COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
            COMPONENT_OPEN_FACE_UNIFIED_MANIM,
        ),
        "files": ("generated_open_face_visibility_3d.py",),
    },
    # The source-project orchestrator owns its persisted contracts and its own
    # build semantics.  Other relevant Provider identities are incorporated
    # into the appropriate node cache keys and their used-revision union is
    # recorded in the build manifest.  They are not broad dependencies here,
    # which preserves narrow invalidation.
    COMPONENT_SOURCE_PROJECT_BUILD: {
        "dependencies": (),
        "files": (
            "source_project.py",
            "schemas/tikz-native-source-project-v1.schema.json",
            "schemas/tikz-native-build-manifest-v1.schema.json",
        ),
    },
    # The finite-quadric package is split into three independently versioned
    # layers.  The geometry component owns the Manim-neutral authoring and
    # section-animation contracts; the visibility component owns projected
    # occlusion and painter ordering; the final component is the only layer
    # that imports Manim and mutates scene objects.
    COMPONENT_QUADRIC_GEOMETRY: {
        "dependencies": (COMPONENT_POLYHEDRON_VISIBILITY,),
        "files": (
            "@tool/polyhedron_visibility/quadrics/__init__.py",
            "@tool/polyhedron_visibility/quadrics/algebra.py",
            "@tool/polyhedron_visibility/quadrics/animation.py",
            "@tool/polyhedron_visibility/quadrics/conics.py",
            "@tool/polyhedron_visibility/quadrics/contract.py",
            "@tool/polyhedron_visibility/quadrics/curves.py",
            "@tool/polyhedron_visibility/quadrics/plane_motion.py",
            "@tool/polyhedron_visibility/quadrics/plane_patch.py",
            "@tool/polyhedron_visibility/quadrics/roots.py",
            "@tool/polyhedron_visibility/quadrics/sections.py",
            "@tool/polyhedron_visibility/quadrics/trace.py",
            "@tool/polyhedron_visibility/quadrics/transition.py",
        ),
    },
    COMPONENT_QUADRIC_VISIBILITY: {
        "dependencies": (
            COMPONENT_POLYHEDRON_VISIBILITY,
            COMPONENT_QUADRIC_GEOMETRY,
        ),
        "files": (
            "@tool/polyhedron_visibility/quadrics/boundary_compositing.py",
            "@tool/polyhedron_visibility/quadrics/boundary_section.py",
            "@tool/polyhedron_visibility/quadrics/composite_section.py",
            "@tool/polyhedron_visibility/quadrics/compositing.py",
            "@tool/polyhedron_visibility/quadrics/critical.py",
            "@tool/polyhedron_visibility/quadrics/curve_intersections.py",
            "@tool/polyhedron_visibility/quadrics/global_occlusion.py",
            "@tool/polyhedron_visibility/quadrics/projection.py",
            "@tool/polyhedron_visibility/quadrics/section_compositing.py",
            "@tool/polyhedron_visibility/quadrics/surface_boundaries.py",
            "@tool/polyhedron_visibility/quadrics/visibility.py",
        ),
    },
    COMPONENT_QUADRIC_MANIM: {
        "dependencies": (
            COMPONENT_POLYHEDRON_VISIBILITY,
            COMPONENT_QUADRIC_GEOMETRY,
            COMPONENT_QUADRIC_VISIBILITY,
            COMPONENT_MANAGED_PAINTER_BAND,
        ),
        "files": (
            "@tool/polyhedron_visibility/quadrics/authoring.py",
            "@tool/polyhedron_visibility/quadrics/capacity.py",
            "@tool/polyhedron_visibility/quadrics/composite_authoring.py",
            "@tool/polyhedron_visibility/quadrics/manim.py",
            "@tool/polyhedron_visibility/quadrics/manim_runtime.py",
            "@tool/polyhedron_visibility/quadrics/performance.py",
            "@tool/polyhedron_visibility/quadrics/profiles.py",
            "@tool/polyhedron_visibility/quadrics/transition_manim.py",
        ),
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
    component: (
        2 if component == COMPONENT_QUADRIC_VISIBILITY else 1
    )
    for component in _COMPONENT_DEFINITIONS
}


# Filled with the verified digests after the component manifest was defined.
# A mismatch does not silently retain the declaration; it returns the actual
# component-sha256 identity and therefore invalidates only that component.
_DECLARED_IMPLEMENTATION_DIGESTS: Final[dict[str, str]] = {
    COMPONENT_ASSET_COMPILER: (
        "5599d2a0f6e6c5e561c6b9ea5ee420eae1fc4f21b88d20b13836170cafec54ff"
    ),
    COMPONENT_GEOMETRY_RIG_2D: (
        "2a845be9cdfeffba8c345aaa346b4ac84127ace890dac5d9eda71f1d199f536e"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_2D: (
        "8d9abee202dd3246e97f05cb5911f4b9b7cea7a5891fa695fdc2a11cb68ddea4"
    ),
    COMPONENT_NATIVE_RIG_2D: (
        "8b48e5128a90b5ce5320e12e7e6ef3bfe511105a3a907081e3e4082c5c34f816"
    ),
    COMPONENT_MOTION_PREVIEW_2D: (
        "567ba36480e1fc41a0c28d78a2f02618c776a326e16cce88910e40744221bfab"
    ),
    COMPONENT_GEOMETRY_RIG_3D: (
        "3673185517871945742b8eea609a99c67f34b90d718d0ca10e2a2d407d4af534"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D: (
        "177257d74cd12ddcb7b87920610f9a300cda8dd2bd52604f9007bc838d7d5278"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: (
        "3783dd61ed9c86d89f0b45c403c3af4a9b5cf5181bb7d771133cfa3cb35a7912"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: (
        "6050f4132e08d8424ecfe596684b78cb625f806a47c903e691999cf796a616fd"
    ),
    COMPONENT_EMBEDDED_MOTION_3D: (
        "c8eaf866ddb03e78ac5a6b2ee7953fc8a1663cfe11299298aac72b9761cdbfa6"
    ),
    COMPONENT_MOTION_PREVIEW_3D: (
        "5ac22e8e5b01c4ce5449f8eb1f314f436e23d6b4e992444084de905d51e02d4b"
    ),
    COMPONENT_POLYHEDRON_VISIBILITY: (
        "745b8cb5b9cb832b4ce6c831094d961cbec7d506ec9110dd4f47e042b7a0799b"
    ),
    COMPONENT_FACE_DEPTH_CUE_3D: (
        "65b662f5503915686c0a2b9829f6dbb04b0ca0cda74bb049adde3442898001ff"
    ),
    COMPONENT_CONVEX_SECTION_3D: (
        "5cfd664e136caf4f68876ac76f81674d99c15b3b275c3859a84c174d738729b5"
    ),
    COMPONENT_COPY_IDENTITY_HANDOFF: (
        "bf8aa2d0fe3ec9921320305279f2e23c8ab71d68b5613d19d19f467326d293b7"
    ),
    COMPONENT_DERIVED_DIHEDRAL_VISIBILITY: (
        "5145d018dc54d5dba7e7b727d60ed6106360b9e6889488029080065ed6ee6b29"
    ),
    COMPONENT_OPEN_FACE_VISIBILITY: (
        "0a23a2cc27ac2a82338608a0a5ab374c343e3215ee4c162e8d573e5baced008c"
    ),
    COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING: (
        "c985c56a84b6414e5bcdcbc47ec522e84712726d123b0e56c3d7c2cbe630b7c9"
    ),
    COMPONENT_MANAGED_PAINTER_BAND: (
        "6cfc31d3b50aa4768d17eaf10812ab6dc4b253227cc1d28a2c6bd3734b84b827"
    ),
    COMPONENT_OPEN_FACE_UNIFIED_MANIM: (
        "b5859b4c2f70893b31bf9897c7a868138fb8d00aa2e1f3e1cceeccbd8421c6d5"
    ),
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: (
        "60658221fd42359fb23310004b8a5be89afbb4c8fe1036ecec80242189e0bc2a"
    ),
    COMPONENT_TIKZ_CONVEX_SECTION_3D: (
        "22804f08f377e20a28bd5cf07a396bdd105578e44e93bb49746c771d57bd51d8"
    ),
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: (
        "0872cb5b5c3c7566a22816701b4ea59c51f3ab5c0f3d6ff1ac77bb6543a73d60"
    ),
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: (
        "f74e221297d3444a17b9165ae9759ae41ca4cf7bda013333e28ab3b9d157a54e"
    ),
    COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D: (
        "d75c0f26754b524da806a747ac91d43b8dec444343a8762fb9f6fe57fb81278b"
    ),
    COMPONENT_SOURCE_PROJECT_BUILD: (
        "00579e342012a96443c098c7004be672d265d32fe6f82d66b0c6b11ba64305b7"
    ),
    COMPONENT_QUADRIC_GEOMETRY: (
        "2d70a1abb2f75a76896a072808fb00f8fe2df91f70b98f0fa2b65fe90a21a62c"
    ),
    COMPONENT_QUADRIC_VISIBILITY: (
        "e7f7a2a5dd28370daa45fb71404bc9b8866285f506eb37d828980908d5271262"
    ),
    COMPONENT_QUADRIC_MANIM: (
        "e73c5aa28575265b5dfb06d612a007ca54991eb0694dd000a05606a99633423d"
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
    "COMPONENT_COPY_IDENTITY_HANDOFF",
    "COMPONENT_DERIVED_DIHEDRAL_VISIBILITY",
    "COMPONENT_CONTRACT_REVISION_SCHEMA",
    "COMPONENT_EMBEDDED_MOTION_3D",
    "COMPONENT_FACE_DEPTH_CUE_3D",
    "COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D",
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
    "COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING",
    "COMPONENT_MANAGED_PAINTER_BAND",
    "COMPONENT_OPEN_FACE_UNIFIED_MANIM",
    "COMPONENT_POLYHEDRON_VISIBILITY",
    "COMPONENT_QUADRIC_GEOMETRY",
    "COMPONENT_QUADRIC_MANIM",
    "COMPONENT_QUADRIC_VISIBILITY",
    "COMPONENT_REVISION_SCHEMA",
    "COMPONENT_SOURCE_PROJECT_BUILD",
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
