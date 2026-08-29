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
    "source-sha256:47e72918791518076d0d6b0c2857c1b3162c29217265e218b2f5a44433c6dd7d"
)
GEOMETRY_RIG_2D_REVISION = (
    "source-sha256:55b5fe153e05e572f146556f69ca2652741071fa813bdde5df150258b4142080"
)
NATIVE_SOURCE_2D_REVISION = (
    "source-sha256:8e0067c597c086c046bc3dc0eb97795ac491815bf390ad35de40b63c5d40c5d1"
)
NATIVE_RIG_2D_REVISION = (
    "source-sha256:70201c67a2703cecf3b210834977a25508b4aff972b8627443a250978f113870"
)
MOTION_PREVIEW_2D_REVISION = (
    "source-sha256:d61945b9d99fa962288cc566de55849df42b71b640a620a7aa7cb89d7ed5c267"
)
GEOMETRY_RIG_3D_REVISION = (
    "source-sha256:9b0de81c87e32a599c145a2fb78b32403e264a98d34494c277e58d7154b1bb60"
)
NATIVE_SOURCE_3D_REVISION = (
    "source-sha256:1ba17fb69e455e3694ae477ebd69de321043555af05c8e7712c3ee22d20307c1"
)
NATIVE_SOURCE_3D_V2_REVISION = (
    "source-sha256:9fb59765ba981108a73a0a4a340de5a32ce4c03707d4925cdb729b9a7ecdcb9c"
)
NATIVE_SOURCE_3D_V3_REVISION = (
    "source-sha256:513935788725e3c72b74905268695a90c443df208a4f321229f05efcdd3ef5aa"
)
EMBEDDED_MOTION_3D_REVISION = (
    "source-sha256:731320aa99d0c61ad55e6f456cf1781c657a0adeada14e0a8cd27ed9055b904a"
)
MOTION_PREVIEW_3D_REVISION = (
    "source-sha256:1111ec6e635702b225d3c1b0a9c47b7a30861bafeabee762e2312318f03ecab2"
)
POLYHEDRON_VISIBILITY_REVISION = (
    "source-sha256:8fff612f011f5b67cecaa66dc251af3126fe091cfe6b753e7fd5e9301cfcc53f"
)
FACE_DEPTH_CUE_3D_REVISION = (
    "source-sha256:499495b399f1078f0532413690821764208bbb87695f1f40019ce396f2ac347a"
)
TIKZ_POLYHEDRON_VISIBILITY_3D_REVISION = (
    "source-sha256:b17799b6605896e53da89b59d692944848bda0ce634a792933f318abf45eba12"
)
OPEN_FACE_VISIBILITY_REVISION = (
    "source-sha256:583f95c7e3a9056b306d90e14f85e580442cf0c2cfd9d0043795f1670dfc43ae"
)
OPEN_FACE_UNIFIED_COMPOSITING_REVISION = (
    "source-sha256:5b2bdd7146a0e548f395637653b61b16cb5cf4a8758399030df934c071b72832"
)
MANAGED_PAINTER_BAND_REVISION = (
    "source-sha256:0aac25e610d7055edf5ce989ce5a91989a62439534645023c4f5f43cdbe25475"
)
OPEN_FACE_UNIFIED_MANIM_REVISION = (
    "source-sha256:4aeeb8388f50c6193d534b77bd60cb3fe0e202d2e439513eecb4d7e51c82410d"
)
TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION = (
    "source-sha256:cdf237694a76c8c6c7869bfa9bf391cedbab410f56409d10c8808a21ecc8710d"
)
TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION = (
    "source-sha256:5602140f9269ac819e0f84abebf12c133decce44927e4725c4e05ca5272d9d4c"
)
GENERATED_OPEN_FACE_VISIBILITY_3D_REVISION = (
    "source-sha256:612dac37f6851e70c1ec8175455166111412f33ad9c65a43f98ffdce8c87c603"
)
SOURCE_PROJECT_BUILD_REVISION = (
    "source-sha256:c108980773c47c6dc8a070c01cec9fb4bc4c26620b9cba9de03f3efe12fbee70"
)
QUADRIC_GEOMETRY_REVISION = (
    "source-sha256:aa2cb1ad9c06bfba01ab9c168d5c2a74ac7fd89066bb8ca0770ab92ed5b9984b"
)
QUADRIC_VISIBILITY_REVISION = (
    "source-sha256:8efa5d2f53b8441f0949f59f68c7cd7b6ba4f83dc8ed9208f71c9f9204954033"
)
QUADRIC_MANIM_REVISION = (
    "source-sha256:ff5c7bb858376a29f462d7499f3ffb3bd27ad5cbb24487e6d53f6318f366cad9"
)
CONVEX_SECTION_3D_REVISION = (
    "source-sha256:a6c71249ce429884b0fcc3341eeea33dacf2d64e14d875d1e0f222c1530637bf"
)
COPY_IDENTITY_HANDOFF_REVISION = (
    "source-sha256:bf8aa2d0fe3ec9921320305279f2e23c8ab71d68b5613d19d19f467326d293b7"
)
DERIVED_DIHEDRAL_VISIBILITY_REVISION = (
    "source-sha256:1fcf6ff43a119d35afe4294157f5f0145bed1749153da248c2335e1582216f22"
)
TIKZ_CONVEX_SECTION_3D_REVISION = (
    "source-sha256:b0623617cf182f17eaaa7ef260540c12749b532e0fbf9fe079316cfbc8da166a"
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
            contracts[COMPONENT_QUADRIC_VISIBILITY],
            "tikz-native-contract:quadric_visibility/v2",
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

    def test_public_release_components_have_frozen_render_identities(self) -> None:
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

    def test_editing_quadric_geometry_changes_its_recursive_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/quadrics/sections.py"
        )
        components = mutated["components"]
        changed = {
            COMPONENT_QUADRIC_GEOMETRY,
            COMPONENT_QUADRIC_VISIBILITY,
            COMPONENT_QUADRIC_MANIM,
        }
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component not in changed:
                self.assertEqual(components[component], baseline[component])

    def test_editing_quadric_visibility_changes_visibility_and_manim(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/quadrics/visibility.py"
        )
        components = mutated["components"]
        changed = {
            COMPONENT_QUADRIC_VISIBILITY,
            COMPONENT_QUADRIC_MANIM,
        }
        for component in changed:
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component not in changed:
                self.assertEqual(components[component], baseline[component])

    def test_editing_quadric_manim_changes_only_manim(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/quadrics/manim.py"
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
        changed = {
            COMPONENT_QUADRIC_GEOMETRY,
            COMPONENT_QUADRIC_VISIBILITY,
            COMPONENT_QUADRIC_MANIM,
        }
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
        changed = {
            COMPONENT_POLYHEDRON_VISIBILITY,
            COMPONENT_FACE_DEPTH_CUE_3D,
            COMPONENT_CONVEX_SECTION_3D,
            COMPONENT_MANAGED_PAINTER_BAND,
            COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
            COMPONENT_OPEN_FACE_VISIBILITY,
            COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING,
            COMPONENT_OPEN_FACE_UNIFIED_MANIM,
            COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
            COMPONENT_TIKZ_CONVEX_SECTION_3D,
            COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
            COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
            COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
            COMPONENT_GENERATED_OPEN_FACE_VISIBILITY_3D,
            COMPONENT_QUADRIC_GEOMETRY,
            COMPONENT_QUADRIC_VISIBILITY,
            COMPONENT_QUADRIC_MANIM,
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
                COMPONENT_QUADRIC_GEOMETRY,
                COMPONENT_QUADRIC_VISIBILITY,
                COMPONENT_QUADRIC_MANIM,
            }:
                self.assertEqual(components[component], baseline[component])
            else:
                self.assertNotEqual(components[component], baseline[component])


if __name__ == "__main__":
    unittest.main()
