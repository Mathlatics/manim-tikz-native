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
    COMPONENT_ASSET_COMPILER: "source-sha256:47e72918791518076d0d6b0c2857c1b3162c29217265e218b2f5a44433c6dd7d",
    COMPONENT_GEOMETRY_RIG_2D: "source-sha256:55b5fe153e05e572f146556f69ca2652741071fa813bdde5df150258b4142080",
    COMPONENT_NATIVE_MANIM_SOURCE_2D: "source-sha256:8e0067c597c086c046bc3dc0eb97795ac491815bf390ad35de40b63c5d40c5d1",
    COMPONENT_NATIVE_RIG_2D: "source-sha256:70201c67a2703cecf3b210834977a25508b4aff972b8627443a250978f113870",
    COMPONENT_MOTION_PREVIEW_2D: "source-sha256:d61945b9d99fa962288cc566de55849df42b71b640a620a7aa7cb89d7ed5c267",
    COMPONENT_GEOMETRY_RIG_3D: "source-sha256:9b0de81c87e32a599c145a2fb78b32403e264a98d34494c277e58d7154b1bb60",
    COMPONENT_NATIVE_MANIM_SOURCE_3D: "source-sha256:1ba17fb69e455e3694ae477ebd69de321043555af05c8e7712c3ee22d20307c1",
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: "source-sha256:9fb59765ba981108a73a0a4a340de5a32ce4c03707d4925cdb729b9a7ecdcb9c",
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: "source-sha256:513935788725e3c72b74905268695a90c443df208a4f321229f05efcdd3ef5aa",
    COMPONENT_EMBEDDED_MOTION_3D: "source-sha256:5365474d68e36c8f0cac666d3c5befa0872247a6bfd8bfba90a009010b95df88",
    COMPONENT_MOTION_PREVIEW_3D: "source-sha256:1b9383e7d289c7160dc41e1fa944e1b83ba2b2263e5fc674a41b896cce49baff",
    COMPONENT_POLYHEDRON_VISIBILITY: "source-sha256:8fff612f011f5b67cecaa66dc251af3126fe091cfe6b753e7fd5e9301cfcc53f",
    COMPONENT_FACE_DEPTH_CUE_3D: "source-sha256:499495b399f1078f0532413690821764208bbb87695f1f40019ce396f2ac347a",
    COMPONENT_CONVEX_SECTION_3D: "source-sha256:a6c71249ce429884b0fcc3341eeea33dacf2d64e14d875d1e0f222c1530637bf",
    COMPONENT_COPY_IDENTITY_HANDOFF: "source-sha256:bf8aa2d0fe3ec9921320305279f2e23c8ab71d68b5613d19d19f467326d293b7",
    COMPONENT_DERIVED_DIHEDRAL_VISIBILITY: "source-sha256:1fcf6ff43a119d35afe4294157f5f0145bed1749153da248c2335e1582216f22",
    COMPONENT_OPEN_FACE_VISIBILITY: "source-sha256:583f95c7e3a9056b306d90e14f85e580442cf0c2cfd9d0043795f1670dfc43ae",
    COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING: "source-sha256:5b2bdd7146a0e548f395637653b61b16cb5cf4a8758399030df934c071b72832",
    COMPONENT_MANAGED_PAINTER_BAND: "source-sha256:0aac25e610d7055edf5ce989ce5a91989a62439534645023c4f5f43cdbe25475",
    COMPONENT_OPEN_FACE_UNIFIED_MANIM: "source-sha256:4aeeb8388f50c6193d534b77bd60cb3fe0e202d2e439513eecb4d7e51c82410d",
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: "source-sha256:b17799b6605896e53da89b59d692944848bda0ce634a792933f318abf45eba12",
    COMPONENT_TIKZ_CONVEX_SECTION_3D: "source-sha256:b0623617cf182f17eaaa7ef260540c12749b532e0fbf9fe079316cfbc8da166a",
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: "source-sha256:cdf237694a76c8c6c7869bfa9bf391cedbab410f56409d10c8808a21ecc8710d",
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: "source-sha256:5602140f9269ac819e0f84abebf12c133decce44927e4725c4e05ca5272d9d4c",
    COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D: "source-sha256:612dac37f6851e70c1ec8175455166111412f33ad9c65a43f98ffdce8c87c603",
    COMPONENT_SOURCE_PROJECT_BUILD: "source-sha256:c108980773c47c6dc8a070c01cec9fb4bc4c26620b9cba9de03f3efe12fbee70",
    COMPONENT_QUADRIC_GEOMETRY: "source-sha256:aabb9e54831a6c1ee326f310580dc9c18598cb32ab3c16e2353a7e7699157950",
    COMPONENT_QUADRIC_VISIBILITY: "source-sha256:8d4aab96c05b1fef95032925484f233e2f79ec3862e02c9c021fd0de751f4b03",
    COMPONENT_QUADRIC_MANIM: "source-sha256:d67e127651db8429e56d07b2d95a1ae52bcadc766efd3fec9746d8731ee539d1",
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
            "parallel_camera.py",
            "parallel_shots.py",
            "parallel_shots_manim.py",
            "dynamic_geometry.py",
            "manim_renderer_3d.py",
            "motion_3d.py",
            "motion_3d_runtime.py",
            "schemas/parallel-shot-sequence-v1.schema.json",
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
        "47e72918791518076d0d6b0c2857c1b3162c29217265e218b2f5a44433c6dd7d"
    ),
    COMPONENT_GEOMETRY_RIG_2D: (
        "55b5fe153e05e572f146556f69ca2652741071fa813bdde5df150258b4142080"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_2D: (
        "8e0067c597c086c046bc3dc0eb97795ac491815bf390ad35de40b63c5d40c5d1"
    ),
    COMPONENT_NATIVE_RIG_2D: (
        "70201c67a2703cecf3b210834977a25508b4aff972b8627443a250978f113870"
    ),
    COMPONENT_MOTION_PREVIEW_2D: (
        "d61945b9d99fa962288cc566de55849df42b71b640a620a7aa7cb89d7ed5c267"
    ),
    COMPONENT_GEOMETRY_RIG_3D: (
        "9b0de81c87e32a599c145a2fb78b32403e264a98d34494c277e58d7154b1bb60"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D: (
        "1ba17fb69e455e3694ae477ebd69de321043555af05c8e7712c3ee22d20307c1"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: (
        "9fb59765ba981108a73a0a4a340de5a32ce4c03707d4925cdb729b9a7ecdcb9c"
    ),
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: (
        "513935788725e3c72b74905268695a90c443df208a4f321229f05efcdd3ef5aa"
    ),
    COMPONENT_EMBEDDED_MOTION_3D: (
        "5365474d68e36c8f0cac666d3c5befa0872247a6bfd8bfba90a009010b95df88"
    ),
    COMPONENT_MOTION_PREVIEW_3D: (
        "1b9383e7d289c7160dc41e1fa944e1b83ba2b2263e5fc674a41b896cce49baff"
    ),
    COMPONENT_POLYHEDRON_VISIBILITY: (
        "8fff612f011f5b67cecaa66dc251af3126fe091cfe6b753e7fd5e9301cfcc53f"
    ),
    COMPONENT_FACE_DEPTH_CUE_3D: (
        "499495b399f1078f0532413690821764208bbb87695f1f40019ce396f2ac347a"
    ),
    COMPONENT_CONVEX_SECTION_3D: (
        "a6c71249ce429884b0fcc3341eeea33dacf2d64e14d875d1e0f222c1530637bf"
    ),
    COMPONENT_COPY_IDENTITY_HANDOFF: (
        "bf8aa2d0fe3ec9921320305279f2e23c8ab71d68b5613d19d19f467326d293b7"
    ),
    COMPONENT_DERIVED_DIHEDRAL_VISIBILITY: (
        "1fcf6ff43a119d35afe4294157f5f0145bed1749153da248c2335e1582216f22"
    ),
    COMPONENT_OPEN_FACE_VISIBILITY: (
        "583f95c7e3a9056b306d90e14f85e580442cf0c2cfd9d0043795f1670dfc43ae"
    ),
    COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING: (
        "5b2bdd7146a0e548f395637653b61b16cb5cf4a8758399030df934c071b72832"
    ),
    COMPONENT_MANAGED_PAINTER_BAND: (
        "0aac25e610d7055edf5ce989ce5a91989a62439534645023c4f5f43cdbe25475"
    ),
    COMPONENT_OPEN_FACE_UNIFIED_MANIM: (
        "4aeeb8388f50c6193d534b77bd60cb3fe0e202d2e439513eecb4d7e51c82410d"
    ),
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: (
        "b17799b6605896e53da89b59d692944848bda0ce634a792933f318abf45eba12"
    ),
    COMPONENT_TIKZ_CONVEX_SECTION_3D: (
        "b0623617cf182f17eaaa7ef260540c12749b532e0fbf9fe079316cfbc8da166a"
    ),
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: (
        "cdf237694a76c8c6c7869bfa9bf391cedbab410f56409d10c8808a21ecc8710d"
    ),
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: (
        "5602140f9269ac819e0f84abebf12c133decce44927e4725c4e05ca5272d9d4c"
    ),
    COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D: (
        "612dac37f6851e70c1ec8175455166111412f33ad9c65a43f98ffdce8c87c603"
    ),
    COMPONENT_SOURCE_PROJECT_BUILD: (
        "c108980773c47c6dc8a070c01cec9fb4bc4c26620b9cba9de03f3efe12fbee70"
    ),
    COMPONENT_QUADRIC_GEOMETRY: (
        "aabb9e54831a6c1ee326f310580dc9c18598cb32ab3c16e2353a7e7699157950"
    ),
    COMPONENT_QUADRIC_VISIBILITY: (
        "8d4aab96c05b1fef95032925484f233e2f79ec3862e02c9c021fd0de751f4b03"
    ),
    COMPONENT_QUADRIC_MANIM: (
        "d67e127651db8429e56d07b2d95a1ae52bcadc766efd3fec9746d8731ee539d1"
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
