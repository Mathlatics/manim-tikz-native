from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from polyhedron_visibility.quadrics.semantic_display import (
    SECTION_DISPLAY_CATALOG_SCHEMA,
    SectionDisplayCatalog,
    SectionDisplayInstruction,
    SectionDisplayMode,
    SectionDisplayPolicy,
    SectionDisplayRole,
    SectionSemanticDisplayError,
    SectionSemanticSlot,
    compile_section_display,
)


ROOT = Path(__file__).resolve().parents[1]


def _slots() -> tuple[SectionSemanticSlot, ...]:
    return (
        SectionSemanticSlot("surface-fill:0", "surface-fill"),
        SectionSemanticSlot("surface-outline:0", "surface-outline"),
        SectionSemanticSlot("plane-fill:0", "plane-fill"),
        SectionSemanticSlot("plane-outline:0", "plane-outline"),
        SectionSemanticSlot(
            "section-bank:ellipse:0",
            "section-curve",
            topology_bank="ellipse",
        ),
        SectionSemanticSlot(
            "section-bank:parabola:0",
            "section-curve",
            topology_bank="parabola",
        ),
        SectionSemanticSlot(
            "section-bank:hyperbola:0",
            "section-curve",
            topology_bank="hyperbola",
        ),
        SectionSemanticSlot(
            "section-point:ellipse:0",
            "section-point",
            topology_bank="ellipse",
        ),
        SectionSemanticSlot(
            "generator:0",
            "generator",
            source_id="boundary:cone:generator:positive",
        ),
        SectionSemanticSlot(
            "generator:1",
            "generator",
            source_id="boundary:cone:generator:negative",
        ),
        SectionSemanticSlot(
            "contour:0",
            "contour",
            source_id="boundary:cone:silhouette:0",
        ),
        SectionSemanticSlot(
            "cap-rim:0",
            "cap-rim",
            source_id="boundary:cone:cap-rim:upper",
        ),
        SectionSemanticSlot(
            "cap-chord:0",
            "cap-chord",
            source_id="section:cap-chord:upper",
        ),
    )


def _catalog() -> SectionDisplayCatalog:
    return SectionDisplayCatalog("lesson-section", _slots())


class QuadricSemanticDisplayTests(unittest.TestCase):
    def test_module_import_does_not_import_manim(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import polyhedron_visibility.quadrics.semantic_display; "
                    "print('manim' in sys.modules)"
                ),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "False")

    def test_catalog_exposes_stable_compound_handles(self) -> None:
        catalog = _catalog()

        self.assertEqual(
            catalog.plane.slot_ids,
            ("plane-fill:0", "plane-outline:0"),
        )
        self.assertEqual(
            catalog.surface.slot_ids,
            (
                "cap-rim:0",
                "contour:0",
                "generator:0",
                "generator:1",
                "surface-fill:0",
                "surface-outline:0",
            ),
        )
        self.assertEqual(
            catalog.section_curve.slot_ids,
            (
                "cap-chord:0",
                "section-bank:ellipse:0",
                "section-bank:hyperbola:0",
                "section-bank:parabola:0",
                "section-point:ellipse:0",
            ),
        )

    def test_section_curve_handle_spans_every_topology_bank(self) -> None:
        catalog = _catalog()
        handle = catalog.section_curve
        banks = {
            slot.topology_bank
            for slot in catalog.slots
            if slot.slot_id in handle.slot_ids
            and slot.role is SectionDisplayRole.SECTION_CURVE
        }
        self.assertEqual(banks, {"ellipse", "parabola", "hyperbola"})
        self.assertEqual(handle.handle_id, "lesson-section:display:section-curve")

        frame = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        compiled_banks = {
            slot.topology_bank
            for slot in frame.slots
            if slot.role is SectionDisplayRole.SECTION_CURVE
        }
        self.assertEqual(compiled_banks, banks)
        self.assertEqual(frame.catalog_digest, catalog.digest)
        self.assertEqual(
            json.loads(frame.to_json())["catalogDigest"],
            catalog.digest,
        )

    def test_boundary_handles_support_kind_and_exact_source(self) -> None:
        catalog = _catalog()
        all_generators = catalog.boundary("generator")
        positive = catalog.boundary(
            SectionDisplayRole.GENERATOR,
            source_id="boundary:cone:generator:positive",
        )

        self.assertEqual(all_generators.slot_ids, ("generator:0", "generator:1"))
        self.assertEqual(positive.slot_ids, ("generator:0",))
        with self.assertRaisesRegex(SectionSemanticDisplayError, "not a boundary"):
            catalog.boundary("plane-fill")
        with self.assertRaisesRegex(SectionSemanticDisplayError, "unavailable"):
            catalog.boundary(
                "generator",
                source_id="boundary:cone:generator:missing",
            )

    def test_named_modes_compile_expected_role_multipliers(self) -> None:
        catalog = _catalog()
        painted = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        outline = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("outline-only"),
        )
        section = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("section-only"),
        )
        hidden = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("hidden"),
        )

        self.assertTrue(all(item.opacity_multiplier == 1.0 for item in painted.slots))
        self.assertEqual(outline.opacity_for("surface-fill:0"), 0.0)
        self.assertEqual(outline.opacity_for("plane-fill:0"), 0.0)
        self.assertEqual(outline.opacity_for("generator:0"), 1.0)
        self.assertEqual(section.opacity_for("section-bank:ellipse:0"), 1.0)
        self.assertEqual(section.opacity_for("cap-chord:0"), 1.0)
        self.assertEqual(section.opacity_for("section-point:ellipse:0"), 1.0)
        self.assertEqual(section.opacity_for("plane-outline:0"), 0.0)
        self.assertEqual(section.opacity_for("contour:0"), 0.0)
        self.assertTrue(all(item.opacity_multiplier == 0.0 for item in hidden.slots))

    def test_emphasis_dims_only_unselected_existing_ink(self) -> None:
        catalog = _catalog()
        frame = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode(
                "outline-only",
                emphasized_handles=(catalog.boundary("generator").handle_id,),
                dim_unemphasized=0.2,
            ),
        )

        self.assertEqual(frame.opacity_for("generator:0"), 1.0)
        self.assertEqual(frame.opacity_for("generator:1"), 1.0)
        self.assertEqual(frame.opacity_for("contour:0"), 0.2)
        self.assertEqual(frame.opacity_for("plane-outline:0"), 0.2)
        self.assertEqual(frame.opacity_for("surface-fill:0"), 0.0)
        emphasized = {
            item.slot_id for item in frame.slots if item.emphasized
        }
        self.assertEqual(emphasized, {"generator:0", "generator:1"})

    def test_emphasis_never_activates_policy_suppressed_slots(self) -> None:
        catalog = _catalog()
        frame = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode(
                "section-only",
                emphasized_handles=(catalog.plane.handle_id,),
                dim_unemphasized=0.1,
            ),
        )

        self.assertEqual(frame.opacity_for("plane-fill:0"), 0.0)
        self.assertEqual(frame.opacity_for("plane-outline:0"), 0.0)
        self.assertEqual(frame.opacity_for("section-bank:ellipse:0"), 0.1)

    def test_unknown_emphasis_fails_before_a_frame_is_returned(self) -> None:
        with self.assertRaisesRegex(SectionSemanticDisplayError, "unavailable"):
            compile_section_display(
                _catalog(),
                SectionDisplayInstruction.for_mode(
                    "painted",
                    emphasized_handles=("missing-handle",),
                ),
            )

    def test_catalog_and_instruction_have_strict_canonical_json(self) -> None:
        catalog = _catalog()
        reversed_catalog = SectionDisplayCatalog(
            "lesson-section",
            tuple(reversed(_slots())),
        )
        instruction = SectionDisplayInstruction.for_mode(
            SectionDisplayMode.PAINTED,
            emphasized_handles=(catalog.section_curve.handle_id,),
            dim_unemphasized=0.35,
        )

        self.assertEqual(catalog.to_json(), reversed_catalog.to_json())
        self.assertEqual(
            SectionDisplayCatalog.from_json(catalog.to_json()).to_json(),
            catalog.to_json(),
        )
        self.assertEqual(
            SectionDisplayInstruction.from_json(instruction.to_json()),
            instruction,
        )
        frame = compile_section_display(catalog, instruction)
        compiled_again = compile_section_display(catalog, instruction)
        self.assertEqual(frame.digest, compiled_again.digest)
        self.assertEqual(
            json.loads(catalog.to_json())["schema"],
            SECTION_DISPLAY_CATALOG_SCHEMA,
        )

    def test_strict_json_rejects_unknown_duplicate_and_nonfinite_values(self) -> None:
        catalog_value = _catalog().to_dict()
        catalog_value["unknown"] = True
        with self.assertRaisesRegex(SectionSemanticDisplayError, "unknown fields"):
            SectionDisplayCatalog.from_dict(catalog_value)
        with self.assertRaisesRegex(SectionSemanticDisplayError, "duplicate key"):
            SectionDisplayCatalog.from_json(
                '{"schema":"quadric-section-display-catalog/v1",'
                '"sectionId":"a","sectionId":"b","slots":[]}'
            )
        with self.assertRaisesRegex(SectionSemanticDisplayError, "non-finite"):
            SectionDisplayInstruction.from_json(
                '{"schema":"quadric-section-display-instruction/v1",'
                '"mode":"painted","emphasizedHandles":[],'
                '"dimUnemphasized":NaN}'
            )

    def test_invalid_slots_and_instructions_fail_closed(self) -> None:
        with self.assertRaisesRegex(SectionSemanticDisplayError, "requires source_id"):
            SectionSemanticSlot("generator", "generator")
        with self.assertRaisesRegex(SectionSemanticDisplayError, "only for section"):
            SectionSemanticSlot(
                "plane",
                "plane-fill",
                topology_bank="ellipse",
            )
        with self.assertRaisesRegex(SectionSemanticDisplayError, "ids must be unique"):
            SectionDisplayCatalog(
                "section",
                (
                    SectionSemanticSlot("slot", "surface-fill"),
                    SectionSemanticSlot("slot", "surface-outline"),
                ),
            )
        with self.assertRaisesRegex(SectionSemanticDisplayError, "must be unique"):
            SectionDisplayInstruction.for_mode(
                "painted",
                emphasized_handles=("handle", "handle"),
            )
        for invalid in (-0.1, 1.1, float("nan"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    SectionSemanticDisplayError,
                    "between 0 and 1",
                ):
                    SectionDisplayInstruction.for_mode(
                        "painted",
                        dim_unemphasized=invalid,
                    )

    def test_unavailable_compound_handle_fails_explicitly(self) -> None:
        catalog = SectionDisplayCatalog(
            "surface-only",
            (SectionSemanticSlot("surface", "surface-fill"),),
        )
        with self.assertRaisesRegex(SectionSemanticDisplayError, "unavailable"):
            _ = catalog.plane
        with self.assertRaisesRegex(SectionSemanticDisplayError, "source_id requires"):
            catalog.boundary(source_id="anything")


if __name__ == "__main__":
    unittest.main()
