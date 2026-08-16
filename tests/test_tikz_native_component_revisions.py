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
from tikz_native.geometry_rig_bridge import (
    _bridge_provider_info as geometry_rig_2d_provider_info,
)
from tikz_native.motion_3d_bridge import provider_info as motion_3d_provider_info
from tikz_native.motion_bridge import provider_info as motion_2d_provider_info
from tikz_native.provider import provider_info as asset_provider_info
from tikz_native.version import (
    COMPONENT_ASSET_COMPILER,
    COMPONENT_EMBEDDED_MOTION_3D,
    COMPONENT_GEOMETRY_RIG_3D,
    COMPONENT_MOTION_PREVIEW_2D,
    COMPONENT_MOTION_PREVIEW_3D,
    COMPONENT_NATIVE_MANIM_SOURCE_2D,
    COMPONENT_NATIVE_MANIM_SOURCE_3D,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V2,
    COMPONENT_NATIVE_RIG_2D,
    COMPONENT_POLYHEDRON_VISIBILITY,
    COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
    provider_component_files,
    provider_component_neutral_files,
    provider_component_revision_matches,
    provider_component_revisions,
    provider_revision,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "tikz_native"
VISIBILITY_ROOT = ROOT / "polyhedron_visibility"
LEGACY_ASSET_REVISION = (
    "source-sha256:6920c63acf10ec22c3f94a1eeb9374799f5ce467419cb610a447675e9678c0ab"
)
NATIVE_SOURCE_REVISION = (
    "source-sha256:01df91473770e47746d4f14de2d94297a2e846be7dcb250652b65168fcda30d6"
)
NATIVE_SOURCE_3D_REVISION = (
    "source-sha256:270a1d6d04659bb16f1ce2b5239fda2c19315691e103c12491e4eed84b460a45"
)
NATIVE_SOURCE_3D_V2_REVISION = (
    "source-sha256:26d702598f6300ece385972271346a571e9fb28b273732584ce14fca98445769"
)
POLYHEDRON_VISIBILITY_REVISION = (
    "source-sha256:aa45310ff3c70ac1922ddf61b457cafeb789f9011ec67069b70c23d63fb3a8ae"
)
TIKZ_POLYHEDRON_VISIBILITY_3D_REVISION = (
    "source-sha256:f0e495883f71691f8b1ca3728ab78953ab8132b3e316a7a911a6e345f74fcd3c"
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
        asset = asset_provider_info()
        geometry_2d = geometry_rig_2d_provider_info()
        geometry_3d = geometry_rig_3d_provider_info()
        motion_2d = motion_2d_provider_info()
        motion_3d = motion_3d_provider_info()

        self.assertEqual(asset["revision"], revisions[COMPONENT_ASSET_COMPILER])
        self.assertEqual(asset["revision_component"], COMPONENT_ASSET_COMPILER)
        self.assertEqual(
            motion_2d["revision"], revisions[COMPONENT_MOTION_PREVIEW_2D]
        )
        self.assertEqual(
            motion_3d["revision"], revisions[COMPONENT_MOTION_PREVIEW_3D]
        )
        # Geometry analysis first validates an existing ShapeAsset, so its
        # primary revision stays asset-facing; the distinct rig identity is
        # read from the component map by Host and frozen definitions.
        for record in (geometry_2d, geometry_3d):
            self.assertEqual(record["revision"], revisions[COMPONENT_ASSET_COMPILER])
            self.assertEqual(record["revision_component"], COMPONENT_ASSET_COMPILER)
        for record in (asset, geometry_2d, geometry_3d, motion_2d, motion_3d):
            self.assertEqual(record["build_revision"], provider_revision())
            self.assertEqual(record["component_revisions"], revisions)

    def test_verified_page8_and_page9_components_keep_their_frozen_identity(self) -> None:
        revisions = provider_component_revisions()
        self.assertEqual(revisions[COMPONENT_ASSET_COMPILER], LEGACY_ASSET_REVISION)
        self.assertEqual(revisions[COMPONENT_NATIVE_RIG_2D], LEGACY_ASSET_REVISION)
        self.assertEqual(
            revisions[COMPONENT_MOTION_PREVIEW_2D], LEGACY_ASSET_REVISION
        )
        self.assertEqual(revisions[COMPONENT_GEOMETRY_RIG_3D], LEGACY_ASSET_REVISION)
        self.assertEqual(revisions[COMPONENT_EMBEDDED_MOTION_3D], LEGACY_ASSET_REVISION)
        self.assertEqual(
            revisions[COMPONENT_MOTION_PREVIEW_3D], LEGACY_ASSET_REVISION
        )
        self.assertEqual(
            revisions[COMPONENT_NATIVE_MANIM_SOURCE_2D], NATIVE_SOURCE_REVISION
        )
        self.assertEqual(
            revisions[COMPONENT_NATIVE_MANIM_SOURCE_3D],
            NATIVE_SOURCE_3D_REVISION,
        )
        self.assertEqual(
            revisions[COMPONENT_NATIVE_MANIM_SOURCE_3D_V2],
            NATIVE_SOURCE_3D_V2_REVISION,
        )
        self.assertNotIn(
            revisions[COMPONENT_NATIVE_MANIM_SOURCE_3D_V2],
            {LEGACY_ASSET_REVISION, NATIVE_SOURCE_3D_REVISION},
        )

    def test_new_visibility_components_have_independent_frozen_identities(self) -> None:
        revisions = provider_component_revisions()
        self.assertEqual(
            revisions[COMPONENT_POLYHEDRON_VISIBILITY],
            POLYHEDRON_VISIBILITY_REVISION,
        )
        self.assertEqual(
            revisions[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
            TIKZ_POLYHEDRON_VISIBILITY_3D_REVISION,
        )
        self.assertNotEqual(
            revisions[COMPONENT_POLYHEDRON_VISIBILITY],
            revisions[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
        )

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

    def test_editing_v1_3d_codegen_changes_v1_and_its_v2_dependent_only(self) -> None:
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
        for component in baseline:
            if component in {
                COMPONENT_NATIVE_MANIM_SOURCE_3D,
                COMPONENT_NATIVE_MANIM_SOURCE_3D_V2,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_v2_3d_codegen_changes_only_the_v2_source_component(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("native_manim_codegen_3d_v2.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_NATIVE_MANIM_SOURCE_3D_V2],
            baseline[COMPONENT_NATIVE_MANIM_SOURCE_3D_V2],
        )
        for component in baseline:
            if component == COMPONENT_NATIVE_MANIM_SOURCE_3D_V2:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_visibility_core_changes_only_core_and_adapter(self) -> None:
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
            components[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
            baseline[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
        )
        for component in baseline:
            if component in {
                COMPONENT_POLYHEDRON_VISIBILITY,
                COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D,
            }:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_tikz_visibility_adapter_changes_only_its_component(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("polyhedron_visibility_3d_adapter.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertNotEqual(
            components[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
            baseline[COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D],
        )
        for component in baseline:
            if component == COMPONENT_TIKZ_POLYHEDRON_VISIBILITY_3D:
                continue
            self.assertEqual(components[component], baseline[component])

    def test_editing_compiler_invalidates_every_dependent_component(self) -> None:
        baseline = provider_component_revisions()
        mutated = self._probe_copy("compiler.py")
        components = mutated["components"]
        self.assertNotEqual(mutated["build"], provider_revision())
        self.assertEqual(set(components), set(baseline))
        for component in baseline:
            if component == COMPONENT_POLYHEDRON_VISIBILITY:
                self.assertEqual(components[component], baseline[component])
            else:
                self.assertNotEqual(components[component], baseline[component])


if __name__ == "__main__":
    unittest.main()
