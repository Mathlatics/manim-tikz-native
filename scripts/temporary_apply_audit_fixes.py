from __future__ import annotations

import importlib.util
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def replace_once_or_verify(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    old_count = text.count(old)
    new_count = text.count(new)
    if new_count == 1:
        return
    if old_count == 1:
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    raise RuntimeError(
        f"{path}: expected exactly one old block or one already-updated block; "
        f"found old={old_count}, new={new_count}"
    )


def refresh_component_revisions() -> tuple[str, ...]:
    version_path = ROOT / "tikz_native/version.py"
    tests_path = ROOT / "tests/test_tikz_native_component_revisions.py"

    spec = importlib.util.spec_from_file_location(
        "_tikz_native_audit_version",
        version_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tikz_native/version.py")
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)

    implementations = version.provider_component_implementation_revisions()
    constant_names = {
        value: name
        for name, value in vars(version).items()
        if (
            name.startswith("COMPONENT_")
            and isinstance(value, str)
            and value in implementations
        )
    }
    if set(constant_names) != set(implementations):
        missing = sorted(set(implementations) - set(constant_names))
        raise RuntimeError(f"missing component constant names: {missing}")

    version_text = version_path.read_text(encoding="utf-8")
    tests_text = tests_path.read_text(encoding="utf-8")
    changed: list[str] = []

    for component, implementation_revision in implementations.items():
        digest = implementation_revision.removeprefix("component-sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(
                f"unexpected implementation revision for {component}: "
                f"{implementation_revision!r}"
            )
        constant = constant_names[component]
        public_pattern = re.compile(
            rf'({re.escape(constant)}\s*:\s*(?:\(\s*)?"source-sha256:)'
            r"([0-9a-f]{64})"
            r'(")',
            re.DOTALL,
        )
        implementation_pattern = re.compile(
            rf'({re.escape(constant)}\s*:\s*(?:\(\s*)?")'
            r"([0-9a-f]{64})"
            r'(")',
            re.DOTALL,
        )
        public_matches = list(public_pattern.finditer(version_text))
        implementation_matches = list(implementation_pattern.finditer(version_text))
        if len(public_matches) != 1 or len(implementation_matches) != 1:
            raise RuntimeError(
                f"{component}: expected one public and one implementation revision; "
                f"found {len(public_matches)} and {len(implementation_matches)}"
            )
        old_public = public_matches[0].group(2)
        old_implementation = implementation_matches[0].group(2)
        if old_public != old_implementation:
            raise RuntimeError(
                f"{component}: public and implementation declarations diverged"
            )
        if old_implementation == digest:
            continue

        version_text = public_pattern.sub(
            lambda match: match.group(1) + digest + match.group(3),
            version_text,
            count=1,
        )
        version_text = implementation_pattern.sub(
            lambda match: match.group(1) + digest + match.group(3),
            version_text,
            count=1,
        )
        old_literal = f"source-sha256:{old_public}"
        new_literal = f"source-sha256:{digest}"
        occurrences = tests_text.count(old_literal)
        if occurrences < 1:
            raise RuntimeError(
                f"{component}: frozen revision literal missing from component tests"
            )
        tests_text = tests_text.replace(old_literal, new_literal)
        changed.append(component)

    version_path.write_text(version_text, encoding="utf-8")
    tests_path.write_text(tests_text, encoding="utf-8")
    return tuple(changed)


def main() -> None:
    replace_once_or_verify(
        "tikz_native/source_project.py",
        '''_FORBIDDEN_MANIFEST_KEYS = {
    "compositingmode",
    "implementationmode",
}


class SourceProjectError(ValueError):
''',
        '''_FORBIDDEN_MANIFEST_KEYS = {
    "compositingmode",
    "implementationmode",
}

_PAINTER_Z_BAND_BASE = 10_000.0
_PAINTER_Z_BAND_WIDTH = 1024.0
_PAINTER_Z_BAND_GAP = 1024.0
_PAINTER_Z_BAND_STRIDE = _PAINTER_Z_BAND_WIDTH + _PAINTER_Z_BAND_GAP
_PAINTER_Z_BAND_SLOT_COUNT = 4096


class SourceProjectError(ValueError):
''',
    )
    replace_once_or_verify(
        "tikz_native/source_project.py",
        '''        result[key] = list(object_ids)
    return result


def load_source_project''',
        '''        result[key] = list(object_ids)
    contradictory_ids = sorted(
        set(result.get("include_object_ids", ()))
        & set(result.get("exclude_object_ids", ()))
    )
    if contradictory_ids:
        raise SourceProjectError(
            "selection cannot include and exclude the same object IDs: "
            + ", ".join(contradictory_ids)
        )
    return result


def load_source_project''',
    )
    replace_once_or_verify(
        "tikz_native/source_project.py",
        '''    offset = int(digest.hexdigest()[:8], 16) % 4096
    minimum = float(10_000 + offset * 2)
    return PainterZBand(minimum, minimum + 1024.0)
''',
        '''    offset = (
        int(digest.hexdigest()[:8], 16) % _PAINTER_Z_BAND_SLOT_COUNT
    )
    # A managed band uses inclusive bounds, so adjacent slots must be separated
    # by more than the full band width.  A true slot collision still fails
    # closed at Scene binding time and can be resolved with painterZBand.
    minimum = _PAINTER_Z_BAND_BASE + offset * _PAINTER_Z_BAND_STRIDE
    return PainterZBand(minimum, minimum + _PAINTER_Z_BAND_WIDTH)
''',
    )
    replace_once_or_verify(
        "tikz_native/geometry_rig_3d.py",
        '''    include_ids = _selected_ids(selection, "include_object_ids")
    exclude_ids = _selected_ids(selection, "exclude_object_ids")
    known_ids = {item.id for item in picture.objects}
''',
        '''    include_ids = _selected_ids(selection, "include_object_ids")
    exclude_ids = _selected_ids(selection, "exclude_object_ids")
    contradictory_ids = sorted(include_ids & exclude_ids)
    if contradictory_ids:
        raise GeometryRig3DError(
            "selection cannot include and exclude the same object IDs: "
            + ", ".join(contradictory_ids)
        )
    known_ids = {item.id for item in picture.objects}
''',
    )
    replace_once_or_verify(
        "tests/test_source_project.py",
        '''    clean_project,
    load_source_project,
''',
        '''    clean_project,
    derive_painter_z_band,
    load_source_project,
''',
    )
    replace_once_or_verify(
        "tests/test_source_project.py",
        '''    def test_rejects_persisted_implementation_mode_at_any_depth(self) -> None:
''',
        '''    def test_derived_painter_bands_do_not_overlap_across_distinct_hash_slots(
        self,
    ) -> None:
        project = load_source_project(self.write_project())
        first = derive_painter_z_band(project, b"source-0\\n")
        second = derive_painter_z_band(project, b"source-3\\n")

        self.assertNotEqual(first, second)
        self.assertTrue(
            first.maximum < second.minimum
            or second.maximum < first.minimum
        )

    def test_selection_rejects_contradictory_object_ids(self) -> None:
        project_path = self.write_project(
            bridge=True,
            extra={
                "selection": {
                    "include_object_ids": ["line.M.N"],
                    "exclude_object_ids": ["line.M.N"],
                }
            },
        )
        with self.assertRaisesRegex(
            SourceProjectError,
            "include and exclude the same object IDs",
        ):
            load_source_project(project_path)

    def test_rejects_persisted_implementation_mode_at_any_depth(self) -> None:
''',
    )
    replace_once_or_verify(
        "tests/test_tikz_native_geometry_rig_3d.py",
        '''    def test_nonorthogonal_tikz_entry_advertises_no_orbit_transition(self) -> None:
''',
        '''    def test_selection_rejects_object_ids_in_both_include_and_exclude(
        self,
    ) -> None:
        selection = self._selection()
        selection["include_object_ids"] = ["line.M.N"]
        selection["exclude_object_ids"] = ["line.M.N"]

        with self.assertRaisesRegex(
            ValueError,
            "include and exclude the same object IDs",
        ):
            analyze_geometry_rig_3d(self.picture, selection=selection)

    def test_nonorthogonal_tikz_entry_advertises_no_orbit_transition(self) -> None:
''',
    )
    replace_once_or_verify(
        "docs/source-authoritative-projects.md",
        '''Use `tikz-native health` to inspect the current component render and contract
revisions.

## Unified output, not automatic legacy fallback
''',
        '''Use `tikz-native health` to inspect the current component render and contract
revisions.

When `renderIntent.painterZBand` is omitted, the Provider hashes the TikZ bytes
and projection into one of 4096 deterministic slots. Each slot reserves a
1024-wide band and starts 2048 z-index units after the previous slot, so
different slots cannot overlap even though managed-band bounds are inclusive.
A true hash-slot collision, or an unrelated authored drawable inside the chosen
band, still fails closed; set an explicit `painterZBand` to coordinate those
figures deliberately.

## Unified output, not automatic legacy fallback
''',
    )

    changed_components = refresh_component_revisions()
    print(
        "applied audit fixes; refreshed component revisions: "
        + ", ".join(changed_components)
    )


if __name__ == "__main__":
    main()
