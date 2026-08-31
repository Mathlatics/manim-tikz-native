from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import tikz_native.version as version_module

from tikz_native.geometry_rig_3d_bridge import (
    _bridge_provider_info as geometry_rig_3d_provider_info,
)
from tikz_native.geometry_rig_3d_source_v3_bridge import (
    _bridge_provider_info as geometry_rig_3d_source_v3_provider_info,
)
from tikz_native.geometry_rig_bridge import (
    _bridge_provider_info as geometry_rig_2d_provider_info,
)
from tikz_native.motion_3d_bridge import provider_info as motion_3d_provider_info
from tikz_native.motion_bridge import provider_info as motion_2d_provider_info
from tikz_native.provider import provider_info as asset_provider_info
from tikz_native.version import (
    COMPONENT_ASSET_COMPILER,
    COMPONENT_CONVEX_SECTION_3D,
    COMPONENT_COPY_IDENTITY_HANDOFF,
    COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
    COMPONENT_EMBEDDED_MOTION_3D,
    COMPONENT_FACE_DEPTH_CUE_3D,
    COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
    COMPONENT_GEOMETRY_RIG_2D,
    COMPONENT_GEOMETRY_RIG_3D,
    COMPONENT_MOTION_PREVIEW_2D,
    COMPONENT_MOTION_PREVIEW_3D,
    COMPONENT_MANAGED_PAINTER_BAND,
    COMPONENT_NATIVE_MANIM_SOURCE_2D,
    COMPONENT_NATIVE_MANIM_SOURCE_3D,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
    COMPONENT_NATIVE_RIG_2D,
    COMPONENT_OPEN_FACE_VISIBILITY,
    COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING,
    COMPONENT_OPEN_FACE_UNIFIED_MANIM,
    COMPONENT_PARALLEL_CAMERA_CORE,
    COMPONENT_POLYHEDRON_VISIBILITY,
    COMPONENT_QUADRIC_GEOMETRY,
    COMPONENT_QUADRIC_MANIM,
    COMPONENT_QUADRIC_VISIBILITY,
    COMPONENT_SOURCE_PROJECT_BUILD,
    COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
    COMPONENT_TIKZ_CONVEX_SECTION_3D,
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
    COMPONENT_CONTRACT_REVISION_SCHEMA,
    provider_component_contract_revisions,
    provider_component_files,
    provider_component_neutral_files,
    provider_component_revision_matches,
    provider_component_revisions,
    provider_revision,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "tikz_native"
VISIBILITY_ROOT = ROOT / "polyhedron_visibility"
ASSET_REVISION = (
    "source-sha256:94e7617efe8c5aeb0724c9a50b9616d9a50b071e21b79ad6ab642db76691bc83"
)
GEOMETRY_RIG_2D_REVISION = (
    "source-sha256:9fcc5f40b0e8753e517a9dc16a893ad1847bc1ccabdd3516306505571fcecba5"
)
NATIVE_SOURCE_2D_REVISION = (
    "source-sha256:d0ec5302c39c0cc7d3eacb0557cee50e12955bb7d0df0c42896d1630278881f3"
)
NATIVE_RIG_2D_REVISION = (
    "source-sha256:41aa7938fcc29836b11da28fdf427f3842c3eade96268f82878ae75f0bc94f25"
)
MOTION_PREVIEW_2D_REVISION = (
    "source-sha256:70f1694003985ce4c23572d37ecc6cb98584150bfd944b14db9512b15d26d1a2"
)
GEOMETRY_RIG_3D_REVISION = (
    "source-sha256:1f7b1cfd07f9cd8425ab4bbf0267ccb4750afa4a29488af73df1e1999ac0bddd"
)
NATIVE_SOURCE_3D_REVISION = (
    "source-sha256:336d506c3872354aac04bd40e3d2fcba038146bf9ff10d2897d6410ba148f2e6"
)
NATIVE_SOURCE_3D_V2_REVISION = (
    "source-sha256:1a0f6725e1d18f502414c664da6a1f0cb7cd568b5ade89cf2ca34989be8f34ee"
)
NATIVE_SOURCE_3D_V3_REVISION = (
    "source-sha256:89be859ac3cc61671d528b05beb9caa73d04853aa763eb31e951378bfba7382c"
)
PARALLEL_CAMERA_CORE_REVISION = (
    "source-sha256:a6ca69204f5d7cbdaf0eaeff9c47fa8a2cc26d1f6c30436e04eb0aeaa717ab15"
)
EMBEDDED_MOTION_3D_REVISION = (
    "source-sha256:7e15127e685f60ee2222f0bbbc54fb21a7500b02de03b96173d4ff9e49b49ba9"
)
MOTION_PREVIEW_3D_REVISION = (
    "source-sha256:ed6b79a041c8ee174531e938fba20b534e07e0ffc611087e4017b055fb45f03d"
)
POLYHEDRON_VISIBILITY_REVISION = (
    "source-sha256:8fff612f011f5b67cecaa66dc251af3126fe091cfe6b753e7fd5e9301cfcc53f"
)
FACE_DEPTH_CUE_3D_REVISION = (
    "source-sha256:499495b399f1078f0532413690821764208bbb87695f1f40019ce396f2ac347a"
)
TIKZ_POLYHEDRON_VISIBILITY_3D_REVISION = (
    "source-sha256:c6b58189c0e6f3dd6e25bf07046141a1090361f54d08ffa27f8dcd225de1949f"
)
OPEN_FACE_VISIBILITY_REVISION = (
    "source-sha256:583f95c7e3a9056b306d90e14f85e580442cf0c2cfd9d0043795f1670dfc43ae"
)
OPEN_FACE_UNIFIED_COMPOSITING_REVISION = (
    "source-sha256:5b2bdd7146a0e548f395637653b61b16cb5cf4a8758399030df934c071b72832"
)
MANAGED_PAINTER_BAND_REVISION = (
    "source-sha256:9e9bde612f6fad601c97c47b8d1cce9c9cd03360566135d17e224d1049c3ee7a"
)
OPEN_FACE_UNIFIED_MANIM_REVISION = (
    "source-sha256:6ea6cf42d61d2cc8c4331b6da5f25ea3b61b659c75ae4c7a493425f0efddd934"
)
TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION = (
    "source-sha256:489b865a3a2adc93887d1e4e240f62828ed97d7c1c2c34e7a425528bbf75d91e"
)
TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION = (
    "source-sha256:8ce701bdc9694c234fb3ee36bab10f271548d5b17d78cb3e833314b50b6736a7"
)
GENERATED_OPEN_FACE_VISIBILITY_3D_REVISION = (
    "source-sha256:6ef37590c16faa499466b23aff94a1203ef33048b8b6d9cad65bbc392342cb52"
)
SOURCE_PROJECT_BUILD_REVISION = (
    "source-sha256:34980c51d6190e21bcbf5bfbe19e565eff17fddfc0bc2a84f1f379b0da73432b"
)
QUADRIC_GEOMETRY_REVISION = (
    "source-sha256:4366a6f4d7b5a29649a6d93fb55e5bb76560a27ad0162876747e0bfa3c8af08c"
)
QUADRIC_VISIBILITY_REVISION = (
    "source-sha256:0b8bb3bed0dd28c26c9de07405ec82500c675895142c5a8cd5e414e2a1380f47"
)
QUADRIC_MANIM_REVISION = (
    "source-sha256:a5616e18d5ae79858da8bc2865103f5785b472e608a0026ae01f38de2278098f"
)
CONVEX_SECTION_3D_REVISION = (
    "source-sha256:a6c71249ce429884b0fcc3341eeea33dacf2d64e14d875d1e0f222c1530637bf"
)
COPY_IDENTITY_HANDOFF_REVISION = (
    "source-sha256:bf8aa2d0fe3ec9921320305279f2e23c8ab71d68b5613d19d19f467326d293b7"
)
DERIVED_DIHEDRAL_VISIBILITY_REVISION = (
    "source-sha256:2b49f6a0bfa4c0a8e850f94af2446dd888a469b0472463c0df5127aec8185ae2"
)
TIKZ_CONVEX_SECTION_3D_REVISION = (
    "source-sha256:9e5326f8cb634aa3e892a4e229384916f289100d909f05ab570399b82c2ebe4f"
)

QUADRIC_GEOMETRY_RECURSIVE_COMPONENTS = frozenset(
    {
        COMPONENT_ASSET_COMPILER,
        COMPONENT_EMBEDDED_MOTION_3D,
        COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
        COMPONENT_GEOMETRY_RIG_2D,
        COMPONENT_GEOMETRY_RIG_3D,
        COMPONENT_MOTION_PREVIEW_2D,
        COMPONENT_MOTION_PREVIEW_3D,
        COMPONENT_NATIVE_MANIM_SOURCE_2D,
        COMPONENT_NATIVE_MANIM_SOURCE_3D,
        COMPONENT_NATIVE_MANIM_SOURCE_3D_V2,
        COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
        COMPONENT_NATIVE_RIG_2D,
        COMPONENT_QUADRIC_GEOMETRY,
        COMPONENT_QUADRIC_MANIM,
        COMPONENT_QUADRIC_VISIBILITY,
        COMPONENT_TIKZ_CONVEX_SECTION_3D,
        COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
        COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
        COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
    }
)


def _owned_tool_path(relative: str) -> str:
    if relative.startswith("@tool/"):
        return relative.removeprefix("@tool/")
    return f"tikz_native/{relative}"


class TikzNativeComponentRevisionTests(unittest.TestCase):
    def test_reloading_version_registry_is_idempotent(self) -> None:
        baseline_files = version_module.provider_component_files()
        baseline_revisions = version_module.provider_component_revisions()

        reloaded = version_module
        for _index in range(3):
            reloaded = importlib.reload(reloaded)

        self.assertEqual(reloaded.provider_component_files(), baseline_files)
        self.assertEqual(
            reloaded.provider_component_revisions(),
            baseline_revisions,
        )

    def test_every_provider_source_or_schema_has_an_explicit_component_owner(self) -> None:
        declared = {
            _owned_tool_path(relative)
            for files in provider_component_files().values()
            for relative in files
        } | {
            f"tikz_native/{relative}"
            for relative in provider_component_neutral_files()
        }
        actual = {
            path.relative_to(ROOT).as_posix()
            for path in PACKAGE_ROOT.rglob("*")
            if path.is_file() and path.suffix in {".py", ".json"}
        } | {
            path.relative_to(ROOT).as_posix()
            for path in VISIBILITY_ROOT.rglob("*")
            if path.is_file() and path.suffix in {".py", ".json"}
        }
        self.assertEqual(actual, declared)

    def test_every_quadric_module_has_exactly_one_component_owner(self) -> None:
        prefix = "@tool/polyhedron_visibility/quadrics/"
        owned = [
            relative
            for files in provider_component_files().values()
            for relative in files
            if relative.startswith(prefix)
        ]
        actual = {
            "@tool/" + path.relative_to(ROOT).as_posix()
            for path in (VISIBILITY_ROOT / "quadrics").glob("*.py")
            if path.is_file()
        }
        self.assertEqual(set(owned), actual)
        for relative in actual:
            with self.subTest(relative=relative):
                self.assertEqual(owned.count(relative), 1)

    def test_bridge_health_uses_the_component_that_owns_its_operation(self) -> None:
        revisions = provider_component_revisions()
        contracts = provider_component_contract_revisions()
        asset = asset_provider_info()
        geometry_2d = geometry_rig_2d_provider_info()
        geometry_3d = geometry_rig_3d_provider_info()
        motion_2d = motion_2d_provider_info()
        motion_3d = motion_3d_provider_info()
        source_v3 = geometry_rig_3d_source_v3_provider_info()

        self.assertEqual(asset["revision"], revisions[COMPONENT_ASSET_COMPILER])
        self.assertEqual(asset["revision_component"], COMPONENT_ASSET_COMPILER)
        self.assertTrue(
            asset["capabilities"][
                "open_convex_face_unified_compositing_parallel_v1"
            ]
        )
        self.assertTrue(
            asset["capabilities"]["open_convex_face_unified_manim_binding_v1"]
        )
        self.assertTrue(
            asset["capabilities"]["generated_open_face_visibility_3d_v1"]
        )
        self.assertTrue(
            asset["capabilities"]["source_authoritative_project_build_v1"]
        )
        self.assertTrue(
            asset["capabilities"]["quadric_occlusion_parallel_v1"]
        )
        self.assertTrue(
            asset["capabilities"]["quadric_boundary_compositing_v2"]
        )
        self.assertTrue(
            asset["capabilities"]["quadric_section_animation_trace_v1"]
        )
        self.assertTrue(
            asset["capabilities"][
                "quadric_section_topology_transition_manim_v1"
            ]
        )
        self.assertTrue(
            asset["capabilities"][
                "quadric_open_double_section_compositing_v1"
            ]
        )
        self.assertTrue(
            asset["capabilities"]["quadric_global_compositing_v1"]
        )
        self.assertTrue(
            asset["capabilities"][
                "dandelin_teaching_transparent_compositing_parallel_v1"
            ]
        )
        for component in (
            COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            COMPONENT_SOURCE_PROJECT_BUILD,
            COMPONENT_QUADRIC_GEOMETRY,
            COMPONENT_QUADRIC_VISIBILITY,
            COMPONENT_QUADRIC_MANIM,
        ):
            self.assertIn(component, revisions)
            self.assertIn(component, contracts)
        self.assertEqual(
            contracts[COMPONENT_QUADRIC_GEOMETRY],
            "tikz-native-contract:quadric_geometry/v1",
        )
        self.assertEqual(
            contracts[COMPONENT_ASSET_COMPILER],
            "tikz-native-contract:asset_compiler/v2",
        )
        self.assertEqual(
            contracts[COMPONENT_QUADRIC_VISIBILITY],
            "tikz-native-contract:quadric_visibility/v2",
        )
        self.assertEqual(
            contracts[COMPONENT_QUADRIC_MANIM],
            "tikz-native-contract:quadric_manim/v1",
        )
        self.assertEqual(
            motion_2d["revision"], revisions[COMPONENT_MOTION_PREVIEW_2D]
        )
        self.assertEqual(
            motion_3d["revision"], revisions[COMPONENT_MOTION_PREVIEW_3D]
        )
        self.assertEqual(
            source_v3["revision"],
            revisions[COMPONENT_NATIVE_MANIM_SOURCE_3D_V3],
        )
        self.assertEqual(
            source_v3["revision_component"],
            COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
        )
        # Geometry analysis first validates an existing ShapeAsset, so its
        # primary revision stays asset-facing; the distinct rig identity is
        # read from the component map by Host and frozen definitions.
        for record in (geometry_2d, geometry_3d):
            self.assertEqual(record["revision"], revisions[COMPONENT_ASSET_COMPILER])
            self.assertEqual(record["revision_component"], COMPONENT_ASSET_COMPILER)
        for record in (
            asset,
            geometry_2d,
            geometry_3d,
            motion_2d,
            motion_3d,
            source_v3,
        ):
            self.assertEqual(record["build_revision"], provider_revision())
            self.assertEqual(record["component_revisions"], revisions)
            # The legacy 3D motion bridge intentionally avoids importing the
            # full Provider metadata module.  Primary asset/analysis/source
            # health records expose both new maps; its frozen response stays
            # byte-compatible until that bridge gets a separately versioned
            # metadata envelope.
            if record is not motion_3d:
                self.assertEqual(
                    record["component_render_revisions"], revisions
                )
                self.assertEqual(
                    record["component_contract_revision_schema"],
                    COMPONENT_CONTRACT_REVISION_SCHEMA,
                )
                self.assertEqual(
                    record["component_contract_revisions"], contracts
                )

    def test_public_0_1_component_history_is_not_relabelled(self) -> None:
        historical = {
            COMPONENT_PARALLEL_CAMERA_CORE: (
                "source-sha256:6ea0e6870ffafbad664676e0a7429c4240ed56d1f1fdc073a075c6800da276cc"
            ),
            COMPONENT_EMBEDDED_MOTION_3D: (
                "source-sha256:d289d7c39fed4b95b856d61736ed84ef3e607c657c94cd10682d692a2f6eff49"
            ),
            COMPONENT_MOTION_PREVIEW_3D: (
                "source-sha256:0d0a8bb97386aefd0f74ef9896dbe359e0827e1fe2a2ca13c141b8d0815822c7"
            ),
            COMPONENT_QUADRIC_GEOMETRY: (
                "source-sha256:37e2f79b974a4329f97bfb355378b6bb11383e1e7423a01a9cf62709eb633d4a"
            ),
            COMPONENT_QUADRIC_VISIBILITY: (
                "source-sha256:8d5a40e6f1b01d6d6e5285a1cae4218fec5892d6f6ced928ce9cad2431a03420"
            ),
            COMPONENT_QUADRIC_MANIM: (
                "source-sha256:761a98f07a3b8d2b400c44a4e615084db1ffe9c97f15376ac93744a948172b94"
            ),
        }
        public_history = version_module._PUBLIC_0_1_COMPONENT_REVISIONS
        for component, revision in historical.items():
            self.assertEqual(public_history[component], revision)
            self.assertNotEqual(
                version_module._UNRELEASED_COMPONENT_REVISIONS[component],
                revision,
            )

    def test_current_provider_components_have_declared_render_identities(self) -> None:
        revisions = provider_component_revisions()
        expected = {
            COMPONENT_ASSET_COMPILER: ASSET_REVISION,
            COMPONENT_GEOMETRY_RIG_2D: GEOMETRY_RIG_2D_REVISION,
            COMPONENT_NATIVE_MANIM_SOURCE_2D: NATIVE_SOURCE_2D_REVISION,
            COMPONENT_NATIVE_RIG_2D: NATIVE_RIG_2D_REVISION,
            COMPONENT_MOTION_PREVIEW_2D: MOTION_PREVIEW_2D_REVISION,
            COMPONENT_GEOMETRY_RIG_3D: GEOMETRY_RIG_3D_REVISION,
            COMPONENT_NATIVE_MANIM_SOURCE_3D: NATIVE_SOURCE_3D_REVISION,
            COMPONENT_NATIVE_MANIM_SOURCE_3D_V2: NATIVE_SOURCE_3D_V2_REVISION,
            COMPONENT_NATIVE_MANIM_SOURCE_3D_V3: NATIVE_SOURCE_3D_V3_REVISION,
            COMPONENT_PARALLEL_CAMERA_CORE: PARALLEL_CAMERA_CORE_REVISION,
            COMPONENT_EMBEDDED_MOTION_3D: EMBEDDED_MOTION_3D_REVISION,
            COMPONENT_MOTION_PREVIEW_3D: MOTION_PREVIEW_3D_REVISION,
            COMPONENT_POLYHEDRON_VISIBILITY: POLYHEDRON_VISIBILITY_REVISION,
            COMPONENT_FACE_DEPTH_CUE_3D: FACE_DEPTH_CUE_3D_REVISION,
            COMPONENT_CONVEX_SECTION_3D: CONVEX_SECTION_3D_REVISION,
            COMPONENT_COPY_IDENTITY_HANDOFF: COPY_IDENTITY_HANDOFF_REVISION,
            COMPONENT_DERIVED_DIHEDRAL_VISIBILITY: (
                DERIVED_DIHEDRAL_VISIBILITY_REVISION
            ),
            COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: (
                TIKZ_POLYHEDRON_VISIBILITY_3D_REVISION
            ),
            COMPONENT_TIKZ_CONVEX_SECTION_3D: (
                TIKZ_CONVEX_SECTION_3D_REVISION
            ),
            COMPONENT_OPEN_FACE_VISIBILITY: OPEN_FACE_VISIBILITY_REVISION,
            COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING: (
                OPEN_FACE_UNIFIED_COMPOSITING_REVISION
            ),
            COMPONENT_MANAGED_PAINTER_BAND: MANAGED_PAINTER_BAND_REVISION,
            COMPONENT_OPEN_FACE_UNIFIED_MANIM: OPEN_FACE_UNIFIED_MANIM_REVISION,
            COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: (
                TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION
            ),
            COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: (
                TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION
            ),
            COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D: (
                GENERATED_OPEN_FACE_VISIBILITY_3D_REVISION
            ),
            COMPONENT_SOURCE_PROJECT_BUILD: SOURCE_PROJECT_BUILD_REVISION,
            COMPONENT_QUADRIC_GEOMETRY: QUADRIC_GEOMETRY_REVISION,
            COMPONENT_QUADRIC_VISIBILITY: QUADRIC_VISIBILITY_REVISION,
            COMPONENT_QUADRIC_MANIM: QUADRIC_MANIM_REVISION,
        }
        self.assertEqual(revisions, expected)
        self.assertEqual(len(set(expected.values())), len(expected))

    def test_visibility_components_have_independent_frozen_identities(self) -> None:
        revisions = provider_component_revisions()
        expected = {
            COMPONENT_POLYHEDRON_VISIBILITY: POLYHEDRON_VISIBILITY_REVISION,
            COMPONENT_FACE_DEPTH_CUE_3D: FACE_DEPTH_CUE_3D_REVISION,
            COMPONENT_CONVEX_SECTION_3D: CONVEX_SECTION_3D_REVISION,
            COMPONENT_COPY_IDENTITY_HANDOFF: COPY_IDENTITY_HANDOFF_REVISION,
            COMPONENT_DERIVED_DIHEDRAL_VISIBILITY: (
                DERIVED_DIHEDRAL_VISIBILITY_REVISION
            ),
            COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D: (
                TIKZ_POLYHEDRON_VISIBILITY_3D_REVISION
            ),
            COMPONENT_TIKZ_CONVEX_SECTION_3D: (
                TIKZ_CONVEX_SECTION_3D_REVISION
            ),
            COMPONENT_OPEN_FACE_VISIBILITY: OPEN_FACE_VISIBILITY_REVISION,
            COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING: (
                OPEN_FACE_UNIFIED_COMPOSITING_REVISION
            ),
            COMPONENT_MANAGED_PAINTER_BAND: MANAGED_PAINTER_BAND_REVISION,
            COMPONENT_OPEN_FACE_UNIFIED_MANIM: OPEN_FACE_UNIFIED_MANIM_REVISION,
            COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: (
                TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION
            ),
            COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: (
                TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION
            ),
            COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D: (
                GENERATED_OPEN_FACE_VISIBILITY_3D_REVISION
            ),
            COMPONENT_QUADRIC_GEOMETRY: QUADRIC_GEOMETRY_REVISION,
            COMPONENT_QUADRIC_VISIBILITY: QUADRIC_VISIBILITY_REVISION,
            COMPONENT_QUADRIC_MANIM: QUADRIC_MANIM_REVISION,
        }
        self.assertEqual(
            {component: revisions[component] for component in expected},
            expected,
        )
        self.assertEqual(len(set(expected.values())), len(expected))

    def test_unknown_legacy_revision_is_not_treated_as_compatible(self) -> None:
        self.assertFalse(
            provider_component_revision_matches(
                COMPONENT_ASSET_COMPILER,
                "source-sha256:" + "f" * 64,
            )
        )
        self.assertFalse(
            provider_component_revision_matches(
                COMPONENT_EMBEDDED_MOTION_3D,
                "source-sha256:" + "f" * 64,
            )
        )

    def _probe_copy(self, mutation_path: str) -> dict[str, object]:
        with TemporaryDirectory(prefix="tikz-component-revision-") as temporary:
            copied_root = Path(temporary) / "provider"
            shutil.copytree(ROOT, copied_root)
            if mutation_path.startswith("@tool/"):
                target = copied_root / mutation_path.removeprefix("@tool/")
            else:
                target = copied_root / "tikz_native" / mutation_path
            target.write_text(
                target.read_text(encoding="utf-8")
                + "\n# component revision isolation probe\n",
                encoding="utf-8",
            )
            script = """
import json
from tikz_native.version import provider_component_revisions, provider_revision
print(json.dumps({
    'build': provider_revision(),
    'components': provider_component_revisions(),
}, sort_keys=True))
"""
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(copied_root)
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=copied_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=True,
            )
            return json.loads(completed.stdout)

    def test_editing_generated_source_adapter_changes_only_its_component(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("generated_open_face_visibility_3d.py")
        components = mutated["components"]
        for component in baseline:
            if component == COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D:
                self.assertNotEqual(components[component], baseline[component])
            else:
                self.assertEqual(components[component], baseline[component])

    def test_editing_source_project_contract_changes_only_its_component(self) -> None:
        baseline = provider_component_revisions()
        for path in (
            "source_project.py",
            "schemas/tikz-native-source-project-v1.schema.json",
            "schemas/tikz-native-build-manifest-v1.schema.json",
        ):
            with self.subTest(path=path):
                mutated = self._probe_copy(path)
                components = mutated["components"]
                for component in baseline:
                    if component == COMPONENT_SOURCE_PROJECT_BUILD:
                        self.assertNotEqual(
                            components[component], baseline[component]
                        )
                    else:
                        self.assertEqual(components[component], baseline[component])

    def test_editing_parallel_camera_core_changes_its_true_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("parallel_camera.py")
        components = mutated["components"]
        changed = set(QUADRIC_GEOMETRY_RECURSIVE_COMPONENTS) | {
            COMPONENT_PARALLEL_CAMERA_CORE,
            COMPONENT_EMBEDDED_MOTION_3D,
            COMPONENT_MOTION_PREVIEW_3D,
        }
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component not in changed:
                self.assertEqual(components[component], baseline[component])

    def test_editing_parallel_viewport_changes_camera_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("parallel_viewport.py")
        components = mutated["components"]
        changed = set(QUADRIC_GEOMETRY_RECURSIVE_COMPONENTS) | {
            COMPONENT_PARALLEL_CAMERA_CORE,
            COMPONENT_EMBEDDED_MOTION_3D,
            COMPONENT_MOTION_PREVIEW_3D,
        }
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component not in changed:
                self.assertEqual(components[component], baseline[component])

    def test_editing_quadric_geometry_changes_its_recursive_dependents(self) -> None:
        baseline = provider_component_revisions()
        changed = set(QUADRIC_GEOMETRY_RECURSIVE_COMPONENTS)
        for path in (
            "@tool/polyhedron_visibility/quadrics/sections.py",
            "@tool/polyhedron_visibility/quadrics/planar_curves.py",
            "@tool/polyhedron_visibility/quadrics/dandelin.py",
            "@tool/polyhedron_visibility/quadrics/dandelin_overlay.py",
            "@tool/polyhedron_visibility/quadrics/dandelin_views.py",
        ):
            with self.subTest(path=path):
                mutated = self._probe_copy(path)
                components = mutated["components"]
                for component in changed:
                    self.assertNotEqual(
                        components[component],
                        baseline[component],
                    )
                for component in baseline:
                    if component not in changed:
                        self.assertEqual(
                            components[component],
                            baseline[component],
                        )

    def test_editing_semantic_compositing_changes_geometry_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/quadrics/semantic_compositing.py"
        )
        components = mutated["components"]
        changed = set(QUADRIC_GEOMETRY_RECURSIVE_COMPONENTS)
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component not in changed:
                self.assertEqual(components[component], baseline[component])

    def test_editing_quadric_visibility_changes_recursive_dependents(self) -> None:
        baseline = provider_component_revisions()
        changed = set(QUADRIC_GEOMETRY_RECURSIVE_COMPONENTS) - {
            COMPONENT_QUADRIC_GEOMETRY
        }
        for path in (
            "@tool/polyhedron_visibility/quadrics/visibility.py",
            "@tool/polyhedron_visibility/quadrics/dandelin_visibility.py",
        ):
            with self.subTest(path=path):
                mutated = self._probe_copy(path)
                components = mutated["components"]
                for component in changed:
                    self.assertNotEqual(
                        components[component],
                        baseline[component],
                    )
                for component in baseline:
                    if component not in changed:
                        self.assertEqual(
                            components[component],
                            baseline[component],
                        )

    def test_editing_quadric_manim_changes_only_manim(self) -> None:
        baseline = provider_component_revisions()
        for path in (
            "@tool/polyhedron_visibility/quadrics/manim.py",
            "@tool/polyhedron_visibility/quadrics/dandelin_authoring.py",
        ):
            with self.subTest(path=path):
                mutated = self._probe_copy(path)
                components = mutated["components"]
                self.assertNotEqual(
                    components[COMPONENT_QUADRIC_MANIM],
                    baseline[COMPONENT_QUADRIC_MANIM],
                )
                for component in baseline:
                    if component != COMPONENT_QUADRIC_MANIM:
                        self.assertEqual(components[component], baseline[component])

    def test_editing_global_parallel_rig_changes_only_manim(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("global_parallel_rig.py")
        components = mutated["components"]
        self.assertNotEqual(
            components[COMPONENT_QUADRIC_MANIM],
            baseline[COMPONENT_QUADRIC_MANIM],
        )
        for component in baseline:
            if component != COMPONENT_QUADRIC_MANIM:
                self.assertEqual(components[component], baseline[component])

    def test_editing_quadric_rig_changes_only_manim(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/quadrics/rig.py"
        )
        components = mutated["components"]
        self.assertNotEqual(
            components[COMPONENT_QUADRIC_MANIM],
            baseline[COMPONENT_QUADRIC_MANIM],
        )
        for component in baseline:
            if component != COMPONENT_QUADRIC_MANIM:
                self.assertEqual(components[component], baseline[component])

    def test_editing_shared_quadric_manim_runtime_changes_only_manim(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/quadrics/manim_runtime.py"
        )
        components = mutated["components"]
        self.assertNotEqual(
            components[COMPONENT_QUADRIC_MANIM],
            baseline[COMPONENT_QUADRIC_MANIM],
        )
        for component in baseline:
            if component != COMPONENT_QUADRIC_MANIM:
                self.assertEqual(components[component], baseline[component])

    def test_editing_quadric_performance_trace_changes_only_manim(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/quadrics/performance.py"
        )
        components = mutated["components"]
        self.assertNotEqual(
            components[COMPONENT_QUADRIC_MANIM],
            baseline[COMPONENT_QUADRIC_MANIM],
        )
        for component in baseline:
            if component != COMPONENT_QUADRIC_MANIM:
                self.assertEqual(components[component], baseline[component])

    def test_editing_quadric_transition_plan_changes_recursive_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/quadrics/transition.py"
        )
        components = mutated["components"]
        changed = set(QUADRIC_GEOMETRY_RECURSIVE_COMPONENTS)
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component not in changed:
                self.assertEqual(components[component], baseline[component])

    def test_editing_quadric_transition_manim_changes_only_manim(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/quadrics/transition_manim.py"
        )
        components = mutated["components"]
        self.assertNotEqual(
            components[COMPONENT_QUADRIC_MANIM],
            baseline[COMPONENT_QUADRIC_MANIM],
        )
        for component in baseline:
            if component != COMPONENT_QUADRIC_MANIM:
                self.assertEqual(components[component], baseline[component])

    def test_editing_2d_codegen_does_not_invalidate_asset_or_3d_runtime(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("native_manim_codegen_2d.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_NATIVE_MANIM_SOURCE_2D],
            baseline[COMPONENT_NATIVE_MANIM_SOURCE_2D],
        )
        for component in (
            COMPONENT_ASSET_COMPILER,
            COMPONENT_GEOMETRY_RIG_3D,
            COMPONENT_EMBEDDED_MOTION_3D,
            COMPONENT_MOTION_PREVIEW_3D,
        ):
            self.assertEqual(components[component], baseline[component])

    def test_editing_v1_3d_codegen_changes_v1_and_recursive_source_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("native_manim_codegen_3d.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_NATIVE_MANIM_SOURCE_3D],
            baseline[COMPONENT_NATIVE_MANIM_SOURCE_3D],
        )
        self.assertNotEqual(
            components[COMPONENT_NATIVE_MANIM_SOURCE_3D_V2],
            baseline[COMPONENT_NATIVE_MANIM_SOURCE_3D_V2],
        )
        self.assertNotEqual(
            components[COMPONENT_NATIVE_MANIM_SOURCE_3D_V3],
            baseline[COMPONENT_NATIVE_MANIM_SOURCE_3D_V3],
        )
        for component in baseline:
            if component in {
                COMPONENT_NATIVE_MANIM_SOURCE_3D,
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V2,
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
                COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_v2_3d_codegen_changes_v2_and_v3_source_components(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("native_manim_codegen_3d_v2.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_NATIVE_MANIM_SOURCE_3D_V2],
            baseline[COMPONENT_NATIVE_MANIM_SOURCE_3D_V2],
        )
        self.assertNotEqual(
            components[COMPONENT_NATIVE_MANIM_SOURCE_3D_V3],
            baseline[COMPONENT_NATIVE_MANIM_SOURCE_3D_V3],
        )
        for component in baseline:
            if component in {
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V2,
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
                COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_v3_3d_codegen_changes_only_the_v3_source_component(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("native_manim_codegen_3d_v3.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_NATIVE_MANIM_SOURCE_3D_V3],
            baseline[COMPONENT_NATIVE_MANIM_SOURCE_3D_V3],
        )
        for component in baseline:
            if component in {
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
                COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_visibility_core_changes_only_its_recursive_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/parallel_solver.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        changed = set(QUADRIC_GEOMETRY_RECURSIVE_COMPONENTS) | {
            COMPONENT_POLYHEDRON_VISIBILITY,
            COMPONENT_FACE_DEPTH_CUE_3D,
            COMPONENT_CONVEX_SECTION_3D,
            COMPONENT_MANAGED_PAINTER_BAND,
            COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
            COMPONENT_OPEN_FACE_VISIBILITY,
            COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING,
            COMPONENT_OPEN_FACE_UNIFIED_MANIM,
        }
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component in changed:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_convex_section_core_changes_only_it_and_tikz_binding(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/sections/solver.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        for component in (
            COMPONENT_CONVEX_SECTION_3D,
            COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
            COMPONENT_TIKZ_CONVEX_SECTION_3D,
        ):
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component in {
                COMPONENT_CONVEX_SECTION_3D,
                COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
                COMPONENT_TIKZ_CONVEX_SECTION_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_face_depth_cue_changes_it_and_section_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/depth_cue/solver.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        for component in (
            COMPONENT_FACE_DEPTH_CUE_3D,
            COMPONENT_CONVEX_SECTION_3D,
            COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
            COMPONENT_TIKZ_CONVEX_SECTION_3D,
        ):
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component in {
                COMPONENT_FACE_DEPTH_CUE_3D,
                COMPONENT_CONVEX_SECTION_3D,
                COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
                COMPONENT_TIKZ_CONVEX_SECTION_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_tikz_convex_section_binding_changes_only_itself(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("convex_section_3d_manim.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_TIKZ_CONVEX_SECTION_3D],
            baseline[COMPONENT_TIKZ_CONVEX_SECTION_3D],
        )
        for component in baseline:
            if component == COMPONENT_TIKZ_CONVEX_SECTION_3D:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_derived_dihedral_changes_only_its_component(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/dihedral_extraction/solver.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_DERIVED_DIHEDRAL_VISIBILITY],
            baseline[COMPONENT_DERIVED_DIHEDRAL_VISIBILITY],
        )
        for component in baseline:
            if component == COMPONENT_DERIVED_DIHEDRAL_VISIBILITY:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_copy_handoff_changes_it_and_derived_dihedral_only(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/copy_handoff/solver.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        for component in (
            COMPONENT_COPY_IDENTITY_HANDOFF,
            COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
        ):
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component in {
                COMPONENT_COPY_IDENTITY_HANDOFF,
                COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_open_face_core_changes_its_recursive_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/open_faces/solver.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        changed = {
            COMPONENT_OPEN_FACE_VISIBILITY,
            COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING,
            COMPONENT_OPEN_FACE_UNIFIED_MANIM,
            COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
            COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
            COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
            COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
        }
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component in changed:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_open_face_unified_core_changes_computation_and_manim(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/open_faces/unified_compositing.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        changed = {
            COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING,
            COMPONENT_OPEN_FACE_UNIFIED_MANIM,
            COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
        }
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component in changed:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_painter_band_changes_its_consumers_only(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/painter_band.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        changed = {
            COMPONENT_MANAGED_PAINTER_BAND,
            COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
            COMPONENT_OPEN_FACE_UNIFIED_MANIM,
            COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            COMPONENT_QUADRIC_MANIM,
        }
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component in changed:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_open_face_unified_manim_changes_only_its_component(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/open_faces/unified_manim.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_OPEN_FACE_UNIFIED_MANIM],
            baseline[COMPONENT_OPEN_FACE_UNIFIED_MANIM],
        )
        self.assertNotEqual(
            components[COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D],
            baseline[COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D],
        )
        for component in baseline:
            if component in {
                COMPONENT_OPEN_FACE_UNIFIED_MANIM,
                COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_tikz_visibility_adapter_changes_it_and_tikz_open_face(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("polyhedron_visibility_3d_adapter.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
            baseline[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
        )
        self.assertNotEqual(
            components[COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D],
            baseline[COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D],
        )
        self.assertNotEqual(
            components[COMPONENT_TIKZ_CONVEX_SECTION_3D],
            baseline[COMPONENT_TIKZ_CONVEX_SECTION_3D],
        )
        for component in baseline:
            if component in {
                COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
                COMPONENT_TIKZ_CONVEX_SECTION_3D,
                COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
                COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
                COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_tikz_open_face_adapter_changes_it_and_v3_source(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("open_face_visibility_3d_adapter.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D],
            baseline[COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D],
        )
        self.assertNotEqual(
            components[COMPONENT_NATIVE_MANIM_SOURCE_3D_V3],
            baseline[COMPONENT_NATIVE_MANIM_SOURCE_3D_V3],
        )
        self.assertNotEqual(
            components[COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D],
            baseline[COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D],
        )
        for component in baseline:
            if component in {
                COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
                COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
                COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_static_open_face_baker_changes_only_its_component(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("open_face_static_asset_3d.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D],
            baseline[COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D],
        )
        for component in baseline:
            if component in {
                COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
                COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_compiler_invalidates_every_dependent_component(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("compiler.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertEqual(set(components), set(baseline))
        for component in baseline:
            if component in {
                COMPONENT_POLYHEDRON_VISIBILITY,
                COMPONENT_FACE_DEPTH_CUE_3D,
                COMPONENT_OPEN_FACE_VISIBILITY,
                COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING,
                COMPONENT_MANAGED_PAINTER_BAND,
                COMPONENT_OPEN_FACE_UNIFIED_MANIM,
                COMPONENT_CONVEX_SECTION_3D,
                COMPONENT_COPY_IDENTITY_HANDOFF,
                COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
                COMPONENT_SOURCE_PROJECT_BUILD,
                COMPONENT_PARALLEL_CAMERA_CORE,
                COMPONENT_QUADRIC_GEOMETRY,
                COMPONENT_QUADRIC_VISIBILITY,
                COMPONENT_QUADRIC_MANIM,
            }:
                self.assertEqual(components[component], baseline[component])
            else:
                self.assertNotEqual(components[component], baseline[component])


if __name__ == "__main__":
    unittest.main()
