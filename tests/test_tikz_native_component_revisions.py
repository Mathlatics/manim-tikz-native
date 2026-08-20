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
    "source-sha256:af91f83c84b9f54a598e0fef3daae04bf2c488c2f5da2228b1ab620a4bf3c1e1"
)
GEOMETRY_RIG_2D_REVISION = (
    "source-sha256:2bd4f2d994897170dcc5f4f47b1a146dcbf28d07e0bf6fe6f8176884d7fcafbb"
)
NATIVE_SOURCE_2D_REVISION = (
    "source-sha256:a2a0a3bab177c193d5f4707ef63f504818bdf4ff9168737576f4e305b252f602"
)
NATIVE_RIG_2D_REVISION = (
    "source-sha256:1f4b2906846a3a605ba07eda10df4b3d70c1f23746d4f89f570cb2c0c78e83c8"
)
MOTION_PREVIEW_2D_REVISION = (
    "source-sha256:e14a7b31c787ff7b382602efef59c8612eb3680a02916432e44d8aee5ea9f431"
)
GEOMETRY_RIG_3D_REVISION = (
    "source-sha256:e356e6f49986890ab7837f5532416a8336b2f4aac13eeddedf4f5da0bb756eae"
)
NATIVE_SOURCE_3D_REVISION = (
    "source-sha256:6a8c1832c3ce8e7f9649bcb8f4a0eadcb6f09318a2ff93c4e29cdd6ef5d452d0"
)
NATIVE_SOURCE_3D_V2_REVISION = (
    "source-sha256:9d43cda60594b6f463c9fcb2701a47338ece7d6d8e574c9356450f3c9daa0afb"
)
NATIVE_SOURCE_3D_V3_REVISION = (
    "source-sha256:b9ff8759218918c952c1d2e1ee8bc6a7acd6ec451c572fa57eb4e58919924147"
)
EMBEDDED_MOTION_3D_REVISION = (
    "source-sha256:e70e83ae4fe8dc345437c3481d02facee33a24741b52ad7a63ce22eb58123a1e"
)
MOTION_PREVIEW_3D_REVISION = (
    "source-sha256:98304c7d7ce6444bb0685be44dd540a5e6fe76afa26b11f362deb532f64ddb7e"
)
POLYHEDRON_VISIBILITY_REVISION = (
    "source-sha256:aa45310ff3c70ac1922ddf61b457cafeb789f9011ec67069b70c23d63fb3a8ae"
)
FACE_DEPTH_CUE_3D_REVISION = (
    "source-sha256:be2a87b144147f49ed7f47c4955c366d00ad48b5cae98ff58e55ae63570da0fa"
)
TIKZ_POLYHEDRON_VISIBILITY_3D_REVISION = (
    "source-sha256:44fe047176386fea2c05cddbcc3b5355c69c44de09e1cd293b2e7c350407edd5"
)
OPEN_FACE_VISIBILITY_REVISION = (
    "source-sha256:8c831f441d21e2ceb39aed78ac3428936ac50fe86fa726ea548a52a4bf426341"
)
TIKZ_OPEN_FACE_VISIBILITY_3D_REVISION = (
    "source-sha256:daaf17ed28d07e2504ffbca76bfc21fb7abfcd5a6c65c3fae19bfc22437ef6dc"
)
TIKZ_OPEN_FACE_STATIC_ASSET_3D_REVISION = (
    "source-sha256:f4d14b704e2d3ec0d978330d6331385b460b7b062b548e11362fc5f24940e097"
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
    "source-sha256:e25da984319fb732d75aeef12b762707fcd7a37be8405e9eb8b3b7af462165c9"
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
