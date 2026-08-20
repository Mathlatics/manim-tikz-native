from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

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
    "source-sha256:1437b15a177a7a7b15b78892abdc9d92dfa1e46353ed708daa47e2180d15554e"
)
GEOMETRY_RIG_2D_REVISION = (
    "source-sha256:88353bb23707044dd3084e257bbd27ecd1d31f962c86ca5d841d810cdb5f2deb"
)
NATIVE_SOURCE_2D_REVISION = (
    "source-sha256:5bfda64bbbafc3ffc3b4dcfa249a5b8c7e7deca75ba2c1378240fc3e38eabd88"
)
NATIVE_RIG_2D_REVISION = (
    "source-sha256:ec8c374d5571b9e75d3443e6fdb39a0b7111db06014263b24e06503495d36926"
)
MOTION_PREVIEW_2D_REVISION = (
    "source-sha256:621781d9b7cac57910c736cdbd299061d109029eb6f6acb624e25ba408745403"
)
GEOMETRY_RIG_3D_REVISION = (
    "source-sha256:a6623495db194eb4cbc49813d60b77ef7c83bfd9dc74b447cbd3f593874c5744"
)
NATIVE_SOURCE_3D_REVISION = (
    "source-sha256:5e93cec53ae402f5545359d8c475b6a112ce7ea0f1291392c595cc7a39a01b41"
)
NATIVE_SOURCE_3D_V2_REVISION = (
    "source-sha256:68d4b86fe2fe533cb63eaae21f91a882465fb2759527e60f4fd53b58ed0ce19a"
)
NATIVE_SOURCE_3D_V3_REVISION = (
    "source-sha256:4c69a45e880146ab552261821ac193d5734f8a20c1ec2129d52b220b49a5b772"
)
EMBEDDED_MOTION_3D_REVISION = (
    "source-sha256:08dab27593aee5781fc7b74d47bb42fd02e18b6b1a9389a071f493c2d26f7ef0"
)
MOTION_PREVIEW_3D_REVISION = (
    "source-sha256:da1a28f14d814545ccf5015da769c88b0d3c538e9b7eec92dd026a2ca92ffae7"
)
POLYHEDRON_VISIBILITY_REVISION = (
    "source-sha256:aa45310ff3c70ac1922ddf61b457cafeb789f9011ec67069b70c23d63fb3a8ae"
)
FACE_DEPTH_CUE_3D_REVISION = (
    "source-sha256:be2a87b144147f49ed7f47c4955c366d00ad48b5cae98ff58e55ae63570da0fa"
)
TIKZ_POLYHEDRON_VISIBILITY_3D_REVISION = (
    "source-sha256:0f4a8985f49ada3319c96c376fc74ba5b1bfd85a1349ca1677831c6b363fa4e4"
)
OPEN_FACE_VISIBILITY_REVISION = (
    "source-sha256:8c831f441d21e2ceb39aed78ac3428936ac50fe86fa726ea548a52a4bf426341"
)
TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION = (
    "source-sha256:8020f51bc3f7b54a492015e3e43863c0e365d2615b2fb145793e78966b93bcdd"
)
TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION = (
    "source-sha256:4d50f4cbc709d5b361072abcd693ed081a998a083eeab1985b2c4fc0bae67b84"
)
CONVEX_SECTION_3D_REVISION = (
    "source-sha256:03581834d1a596f4e678153cf4780329e5c7f424031b91ed20e8981f340d3a4f"
)
COPY_IDENTITY_HANDOFF_REVISION = (
    "source-sha256:bf8aa2d0fe3ec9921320305279f2e23c8ab71d68b5613d19d19f467326d293b7"
)
DERIVED_DIHEDRAL_VISIBILITY_REVISION = (
    "source-sha256:000fc2b3fbd8bf381daff710400e93d8f20387766876f25cc2e2b429c21ec7a1"
)
TIKZ_CONVEX_SECTION_3D_REVISION = (
    "source-sha256:34f700fca1b4547b61020acccdc3c945c06ae305eca9255904488f1302c931ae"
)


def _owned_tool_path(relative: str) -> str:
    if relative.startswith("@tool/"):
        return relative.removeprefix("@tool/")
    return f"tikz_native/{relative}"


class TikzNativeComponentRevisionTests(unittest.TestCase):
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
            COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: (
                TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION
            ),
            COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: (
                TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION
            ),
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
            COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D: (
                TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION
            ),
            COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D: (
                TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION
            ),
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
            if component == COMPONENT_NATIVE_MANIM_SOURCE_3D_V3:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_visibility_core_changes_only_its_recursive_dependents(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/parallel_solver.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_POLYHEDRON_VISIBILITY],
            baseline[COMPONENT_POLYHEDRON_VISIBILITY],
        )
        self.assertNotEqual(
            components[COMPONENT_FACE_DEPTH_CUE_3D],
            baseline[COMPONENT_FACE_DEPTH_CUE_3D],
        )
        self.assertNotEqual(
            components[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
            baseline[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
        )
        self.assertNotEqual(
            components[COMPONENT_OPEN_FACE_VISIBILITY],
            baseline[COMPONENT_OPEN_FACE_VISIBILITY],
        )
        self.assertNotEqual(
            components[COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D],
            baseline[COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D],
        )
        self.assertNotEqual(
            components[COMPONENT_CONVEX_SECTION_3D],
            baseline[COMPONENT_CONVEX_SECTION_3D],
        )
        self.assertNotEqual(
            components[COMPONENT_DERIVED_DIHEDRAL_VISIBILITY],
            baseline[COMPONENT_DERIVED_DIHEDRAL_VISIBILITY],
        )
        self.assertNotEqual(
            components[COMPONENT_TIKZ_CONVEX_SECTION_3D],
            baseline[COMPONENT_TIKZ_CONVEX_SECTION_3D],
        )
        for component in baseline:
            if component in {
                COMPONENT_POLYHEDRON_VISIBILITY,
                COMPONENT_FACE_DEPTH_CUE_3D,
                COMPONENT_CONVEX_SECTION_3D,
                COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
                COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
                COMPONENT_TIKZ_CONVEX_SECTION_3D,
                COMPONENT_OPEN_FACE_VISIBILITY,
                COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
                COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
            }:
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

    def test_editing_open_face_core_changes_only_it_and_tikz_open_face(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy(
            "@tool/polyhedron_visibility/open_faces/solver.py"
        )
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        for component in (
            COMPONENT_OPEN_FACE_VISIBILITY,
            COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
            COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
            COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
        ):
            self.assertNotEqual(components[component], baseline[component])
        for component in baseline:
            if component in {
                COMPONENT_OPEN_FACE_VISIBILITY,
                COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
                COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D,
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
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
            if component == COMPONENT_TIKZ_OPEN_FACE_STATIC_ASSET_3D:
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
                COMPONENT_CONVEX_SECTION_3D,
                COMPONENT_COPY_IDENTITY_HANDOFF,
                COMPONENT_DERIVED_DIHEDRAL_VISIBILITY,
            }:
                self.assertEqual(components[component], baseline[component])
            else:
                self.assertNotEqual(components[component], baseline[component])


if __name__ == "__main__":
    unittest.main()
