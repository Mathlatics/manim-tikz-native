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
    "source-sha256:5599d2a0f6e6c5e561c6b9ea5ee420eae1fc4f21b88d20b13836170cafec54ff"
)
GEOMETRY_RIG_2D_REVISION = (
    "source-sha256:2a845be9cdfeffba8c345aaa346b4ac84127ace890dac5d9eda71f1d199f536e"
)
NATIVE_SOURCE_2D_REVISION = (
    "source-sha256:8d9abee202dd3246e97f05cb5911f4b9b7cea7a5891fa695fdc2a11cb68ddea4"
)
NATIVE_RIG_2D_REVISION = (
    "source-sha256:8b48e5128a90b5ce5320e12e7e6ef3bfe511105a3a907081e3e4082c5c34f816"
)
MOTION_PREVIEW_2D_REVISION = (
    "source-sha256:567ba36480e1fc41a0c28d78a2f02618c776a326e16cce88910e40744221bfab"
)
GEOMETRY_RIG_3D_REVISION = (
    "source-sha256:3673185517871945742b8eea609a99c67f34b90d718d0ca10e2a2d407d4af534"
)
NATIVE_SOURCE_3D_REVISION = (
    "source-sha256:177257d74cd12ddcb7b87920610f9a300cda8dd2bd52604f9007bc838d7d5278"
)
NATIVE_SOURCE_3D_V2_REVISION = (
    "source-sha256:3783dd61ed9c86d89f0b45c403c3af4a9b5cf5181bb7d771133cfa3cb35a7912"
)
NATIVE_SOURCE_3D_V3_REVISION = (
    "source-sha256:6050f4132e08d8424ecfe596684b78cb625f806a47c903e691999cf796a616fd"
)
EMBEDDED_MOTION_3D_REVISION = (
    "source-sha256:c8eaf866ddb03e78ac5a6b2ee7953fc8a1663cfe11299298aac72b9761cdbfa6"
)
MOTION_PREVIEW_3D_REVISION = (
    "source-sha256:5ac22e8e5b01c4ce5449f8eb1f314f436e23d6b4e992444084de905d51e02d4b"
)
POLYHEDRON_VISIBILITY_REVISION = (
    "source-sha256:745b8cb5b9cb832b4ce6c831094d961cbec7d506ec9110dd4f47e042b7a0799b"
)
FACE_DEPTH_CUE_3D_REVISION = (
    "source-sha256:65b662f5503915686c0a2b9829f6dbb04b0ca0cda74bb049adde3442898001ff"
)
TIKZ_POLYHEDRON_VISIBILITY_3D_REVISION = (
    "source-sha256:60658221fd42359fb23310004b8a5be89afbb4c8fe1036ecec80242189e0bc2a"
)
OPEN_FACE_VISIBILITY_REVISION = (
    "source-sha256:0a23a2cc27ac2a82338608a0a5ab374c343e3215ee4c162e8d573e5baced008c"
)
OPEN_FACE_UNIFIED_COMPOSITING_REVISION = (
    "source-sha256:c985c56a84b6414e5bcdcbc47ec522e84712726d123b0e56c3d7c2cbe630b7c9"
)
MANAGED_PAINTER_BAND_REVISION = (
    "source-sha256:785bc38fc11be44d8a2678f4337a80ff0d27a99cdd470565b663710f20a7cda9"
)
OPEN_FACE_UNIFIED_MANIM_REVISION = (
    "source-sha256:0babe56861cbb40b50c92d77fc522c39eeb6b89453a0abb2b2affd5d05352cf5"
)
TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION = (
    "source-sha256:0872cb5b5c3c7566a22816701b4ea59c51f3ab5c0f3d6ff1ac77bb6543a73d60"
)
TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION = (
    "source-sha256:f74e221297d3444a17b9165ae9759ae41ca4cf7bda013333e28ab3b9d157a54e"
)
GENERATED_OPEN_FACE_VISIBILITY_3D_REVISION = (
    "source-sha256:d3b8917a867f754c586bef39acc1ac387726dbb7ce192f8b61a0053505b63503"
)
SOURCE_PROJECT_BUILD_REVISION = (
    "source-sha256:00579e342012a96443c098c7004be672d265d32fe6f82d66b0c6b11ba64305b7"
)
QUADRIC_GEOMETRY_REVISION = (
    "source-sha256:3874f85d3dd7b45717555b713eac5707b13201eff4b98e71d3905cf6c0c76b64"
)
QUADRIC_VISIBILITY_REVISION = (
    "source-sha256:08d76d9326089bdf865aac97ee83bcde239fc1a4c3ba45d783a80b54ca17ab0e"
)
QUADRIC_MANIM_REVISION = (
    "source-sha256:8e3a9e6f2272323e8968f39aaff093682907e517f80f48d1c239fda9519eb19c"
)
CONVEX_SECTION_3D_REVISION = (
    "source-sha256:5cfd664e136caf4f68876ac76f81674d99c15b3b275c3859a84c174d738729b5"
)
COPY_IDENTITY_HANDOFF_REVISION = (
    "source-sha256:bf8aa2d0fe3ec9921320305279f2e23c8ab71d68b5613d19d19f467326d293b7"
)
DERIVED_DIHEDRAL_VISIBILITY_REVISION = (
    "source-sha256:1fba0fc481ccf1df92ed6920146fd26f0a739009d5e49e0770304b44b4e40f1a"
)
TIKZ_CONVEX_SECTION_3D_REVISION = (
    "source-sha256:22804f08f377e20a28bd5cf07a396bdd105578e44e93bb49746c771d57bd51d8"
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
