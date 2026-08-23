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


def _load_version(label: str):
    version_path = ROOT / "tikz_native/version.py"
    spec = importlib.util.spec_from_file_location(
        f"_tikz_native_audit_version_{label}",
        version_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tikz_native/version.py")
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    return version


def _constant_names(version, implementations: dict[str, str]) -> dict[str, str]:
    result = {
        value: name
        for name, value in vars(version).items()
        if (
            name.startswith("COMPONENT_")
            and isinstance(value, str)
            and value in implementations
        )
    }
    if set(result) != set(implementations):
        missing = sorted(set(implementations) - set(result))
        raise RuntimeError(f"missing component constant names: {missing}")
    return result


def _revision_patterns(constant: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    public = re.compile(
        rf'({re.escape(constant)}\s*:\s*(?:\(\s*)?"source-sha256:)'
        r"([0-9a-f]{64})"
        r'(")',
        re.DOTALL,
    )
    implementation = re.compile(
        rf'({re.escape(constant)}\s*:\s*(?:\(\s*)?")'
        r"([0-9a-f]{64})"
        r'(")',
        re.DOTALL,
    )
    return public, implementation


def refresh_component_revisions() -> tuple[str, ...]:
    version_path = ROOT / "tikz_native/version.py"
    tests_path = ROOT / "tests/test_tikz_native_component_revisions.py"
    initial = _load_version("initial")
    initial_revisions = dict(initial._DECLARED_COMPONENT_REVISIONS)
    changed: set[str] = set()

    final = initial
    for attempt in range(3):
        version = _load_version(f"pass_{attempt}")
        implementations = version.provider_component_implementation_revisions()
        constant_names = _constant_names(version, implementations)
        version_text = version_path.read_text(encoding="utf-8")
        touched = False

        for component, implementation_revision in implementations.items():
            digest = implementation_revision.removeprefix("component-sha256:")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise RuntimeError(
                    f"unexpected implementation revision for {component}: "
                    f"{implementation_revision!r}"
                )
            public_pattern, implementation_pattern = _revision_patterns(
                constant_names[component]
            )
            public_matches = list(public_pattern.finditer(version_text))
            implementation_matches = list(
                implementation_pattern.finditer(version_text)
            )
            if len(public_matches) != 1 or len(implementation_matches) != 1:
                raise RuntimeError(
                    f"{component}: expected one public and one implementation "
                    f"revision; found {len(public_matches)} and "
                    f"{len(implementation_matches)}"
                )
            old_public = public_matches[0].group(2)
            old_implementation = implementation_matches[0].group(2)
            if old_public == digest and old_implementation == digest:
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
            changed.add(component)
            touched = True

        if touched:
            version_path.write_text(version_text, encoding="utf-8")

        final = _load_version(f"verify_{attempt}")
        final_implementations = final.provider_component_implementation_revisions()
        final_revisions = final.provider_component_revisions()
        mismatches = [
            component
            for component, implementation_revision in final_implementations.items()
            if final_revisions[component]
            != "source-sha256:"
            + implementation_revision.removeprefix("component-sha256:")
        ]
        if not mismatches:
            break
        if not touched:
            details = {
                component: {
                    "implementation": final_implementations[component],
                    "declared": final._DECLARED_COMPONENT_REVISIONS.get(component),
                    "expected_digest": final._DECLARED_IMPLEMENTATION_DIGESTS.get(
                        component
                    ),
                    "resolved": final_revisions[component],
                }
                for component in mismatches
            }
            raise RuntimeError(
                "component revision refresh made no progress: " + repr(details)
            )
    else:
        raise RuntimeError("component revision refresh did not converge")

    final_revisions = final.provider_component_revisions()
    tests_text = tests_path.read_text(encoding="utf-8")
    for component in sorted(changed):
        old_revision = initial_revisions[component]
        new_revision = final_revisions[component]
        occurrences = tests_text.count(old_revision)
        if occurrences < 1:
            raise RuntimeError(
                f"{component}: frozen revision literal missing from component tests"
            )
        tests_text = tests_text.replace(old_revision, new_revision)
    tests_path.write_text(tests_text, encoding="utf-8")
    return tuple(sorted(changed))


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
        "tikz_native/projection_3d.py",
        "from math import cos, radians, sin, sqrt\n",
        "from math import cos, hypot, isfinite, radians, sin\n",
    )
    replace_once_or_verify(
        "tikz_native/projection_3d.py",
        '''def _normalized(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    length = sqrt(sum(component * component for component in vector))
    if length <= 1e-12:
        raise ValueError("TikZ 三维投影的两个屏幕方向线性相关")
    return tuple(component / length for component in vector)  # type: ignore[return-value]
''',
        '''_RELATIVE_GRAM_DETERMINANT_TOLERANCE = 1e-12


def _normalized(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    length = hypot(*vector)
    if not isfinite(length) or length == 0.0:
        raise ValueError("TikZ 三维投影的屏幕方向必须是有限非零向量")
    return tuple(component / length for component in vector)  # type: ignore[return-value]
''',
    )
    replace_once_or_verify(
        "tikz_native/projection_3d.py",
        '''    screen_u = (x_basis[0], y_basis[0], z_basis[0])
    screen_v = (x_basis[1], y_basis[1], z_basis[1])
    depth = _normalized(_cross(screen_u, screen_v))
    return (screen_u, screen_v, depth)
''',
        '''    screen_u = (x_basis[0], y_basis[0], z_basis[0])
    screen_v = (x_basis[1], y_basis[1], z_basis[1])
    unit_u = _normalized(screen_u)
    unit_v = _normalized(screen_v)
    depth_vector = _cross(unit_u, unit_v)
    depth_length = hypot(*depth_vector)
    if (
        not isfinite(depth_length)
        or depth_length * depth_length
        <= _RELATIVE_GRAM_DETERMINANT_TOLERANCE
    ):
        raise ValueError("TikZ 三维投影的两个屏幕方向线性相关")
    depth = tuple(
        component / depth_length for component in depth_vector
    )
    return (screen_u, screen_v, depth)  # type: ignore[return-value]
''',
    )
    replace_once_or_verify(
        "tikz_native/projection_3d.py",
        '''    first, second = matrix[0], matrix[1]
    aa = sum(value * value for value in first)
    ab = sum(a * b for a, b in zip(first, second, strict=True))
    bb = sum(value * value for value in second)
    determinant = aa * bb - ab * ab
    if abs(determinant) <= 1e-12:
        raise ValueError("TikZ 三维投影无法反解屏幕偏移")
    coefficient_a = (bb * delta_u - ab * delta_v) / determinant
    coefficient_b = (aa * delta_v - ab * delta_u) / determinant
    return tuple(
        coefficient_a * first[index] + coefficient_b * second[index]
        for index in range(3)
    )  # type: ignore[return-value]
''',
        '''    first, second = matrix[0], matrix[1]
    first_length = hypot(*first)
    second_length = hypot(*second)
    if (
        not isfinite(first_length)
        or not isfinite(second_length)
        or first_length == 0.0
        or second_length == 0.0
    ):
        raise ValueError("TikZ 三维投影无法反解屏幕偏移")
    unit_first = tuple(value / first_length for value in first)
    unit_second = tuple(value / second_length for value in second)
    cosine = sum(
        first_value * second_value
        for first_value, second_value in zip(
            unit_first,
            unit_second,
            strict=True,
        )
    )
    cosine = max(-1.0, min(1.0, cosine))
    determinant = 1.0 - cosine * cosine
    if determinant <= _RELATIVE_GRAM_DETERMINANT_TOLERANCE:
        raise ValueError("TikZ 三维投影无法反解屏幕偏移")
    normalized_u = float(delta_u) / first_length
    normalized_v = float(delta_v) / second_length
    coefficient_a = (normalized_u - cosine * normalized_v) / determinant
    coefficient_b = (normalized_v - cosine * normalized_u) / determinant
    return tuple(
        coefficient_a * unit_first[index]
        + coefficient_b * unit_second[index]
        for index in range(3)
    )  # type: ignore[return-value]
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
        "tests/test_tikz_native_3d.py",
        '''from tikz_native.projection_3d import (
    matrix_from_tikz_three_d_view,
    project_point,
)
''',
        '''from tikz_native.projection_3d import (
    matrix_from_tikz_basis,
    matrix_from_tikz_three_d_view,
    project_point,
    screen_delta_to_world,
)
''',
    )
    replace_once_or_verify(
        "tests/test_tikz_native_3d.py",
        '''    def test_demo_compiles_to_native_semantic_inventory(self) -> None:
''',
        '''    def test_projection_basis_independence_is_scale_invariant(self) -> None:
        for scale in (1e-20, 1e20):
            with self.subTest(scale=scale):
                matrix = matrix_from_tikz_basis(
                    (scale, 0.0),
                    (0.0, scale),
                    (0.0, 0.0),
                )
                np.testing.assert_allclose(
                    np.asarray(matrix)[:2],
                    np.array(
                        [
                            [scale, 0.0, 0.0],
                            [0.0, scale, 0.0],
                        ]
                    ),
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    matrix[2],
                    (0.0, 0.0, 1.0),
                    rtol=0.0,
                    atol=1e-15,
                )

    def test_screen_delta_inverse_is_scale_invariant(self) -> None:
        for scale in (1e-20, 1e20):
            with self.subTest(scale=scale):
                matrix = matrix_from_tikz_basis(
                    (scale, 0.0),
                    (0.0, scale),
                    (0.0, 0.0),
                )
                displacement = screen_delta_to_world(
                    matrix,
                    2.0 * scale,
                    -3.0 * scale,
                )
                np.testing.assert_allclose(
                    displacement,
                    (2.0, -3.0, 0.0),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_nearly_parallel_projection_basis_is_rejected_relatively(self) -> None:
        with self.assertRaisesRegex(ValueError, "线性相关"):
            matrix_from_tikz_basis(
                (1.0, 1.0),
                (0.0, 1e-7),
                (0.0, 0.0),
            )

    def test_demo_compiles_to_native_semantic_inventory(self) -> None:
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
