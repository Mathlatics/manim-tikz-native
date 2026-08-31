from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from polyhedron_visibility.quadrics.semantic_compositing import (
    SECTION_COMPOSITING_FRAME_SCHEMA,
    SECTION_COMPOSITING_INSTRUCTION_SCHEMA,
    SectionCompositingAxes,
    SectionCompositingFrame,
    SectionCompositingInstruction,
    SectionCompositingOverride,
    SectionCompositingSlotState,
    SectionCompositingTargetKind,
    SectionDepthPresentationPolicy,
    SectionOcclusionParticipation,
    SectionSemanticCompositingError,
    compile_section_compositing,
)
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayCatalog,
    SectionSemanticSlot,
)


ROOT = Path(__file__).resolve().parents[1]


def _catalog() -> SectionDisplayCatalog:
    return SectionDisplayCatalog(
        "lesson-section",
        (
            SectionSemanticSlot("surface-fill:0", "surface-fill"),
            SectionSemanticSlot("plane-fill:0", "plane-fill"),
            SectionSemanticSlot(
                "section-bank:ellipse:0",
                "section-curve",
                topology_bank="ellipse",
            ),
            SectionSemanticSlot(
                "contour:0",
                "contour",
                source_id="boundary:cone:silhouette",
            ),
            SectionSemanticSlot(
                "contour:1",
                "contour",
                source_id="boundary:cone:silhouette",
            ),
        ),
    )


class QuadricSemanticCompositingTests(unittest.TestCase):
    def test_top_level_lazy_api_exports_complete_compiled_contract(self) -> None:
        from polyhedron_visibility import quadrics

        self.assertIs(
            quadrics.SectionCompositingSlotState,
            SectionCompositingSlotState,
        )
        self.assertIs(
            quadrics.SectionCompositingTargetKind,
            SectionCompositingTargetKind,
        )

    def test_module_is_renderer_neutral(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import polyhedron_visibility.quadrics.semantic_compositing; "
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

    def test_defaults_bind_all_three_axes_to_every_fixed_slot(self) -> None:
        catalog = _catalog()
        frame = compile_section_compositing(
            catalog,
            SectionCompositingInstruction.for_catalog(catalog),
        )

        self.assertEqual(
            tuple(item.slot_id for item in frame.slots),
            tuple(item.slot_id for item in catalog.slots),
        )
        for item in frame.slots:
            self.assertEqual(item.display_opacity, 1.0)
            self.assertIs(
                item.occlusion_participation,
                SectionOcclusionParticipation.CERTIFIED,
            )
            self.assertIs(
                item.depth_presentation,
                SectionDepthPresentationPolicy.PHYSICAL,
            )
        frame.validate_catalog(catalog)

    def test_axes_are_independent_and_never_inferred_from_opacity(self) -> None:
        catalog = _catalog()
        instruction = SectionCompositingInstruction.for_catalog(
            catalog,
            overrides=(
                SectionCompositingOverride.for_slot(
                    "surface-fill:0",
                    display_opacity=0.0,
                ),
                SectionCompositingOverride.for_slot(
                    "plane-fill:0",
                    occlusion_participation="paint-only",
                ),
                SectionCompositingOverride.for_slot(
                    "section-bank:ellipse:0",
                    depth_presentation="depth-aware-diagrammatic",
                ),
            ),
        )
        frame = compile_section_compositing(catalog, instruction)

        invisible = frame.state_for_slot("surface-fill:0")
        self.assertEqual(invisible.display_opacity, 0.0)
        self.assertIs(
            invisible.occlusion_participation,
            SectionOcclusionParticipation.CERTIFIED,
        )
        self.assertIs(
            invisible.depth_presentation,
            SectionDepthPresentationPolicy.PHYSICAL,
        )

        paint_only = frame.state_for_slot("plane-fill:0")
        self.assertEqual(paint_only.display_opacity, 1.0)
        self.assertIs(
            paint_only.occlusion_participation,
            SectionOcclusionParticipation.PAINT_ONLY,
        )
        self.assertIs(
            paint_only.depth_presentation,
            SectionDepthPresentationPolicy.PHYSICAL,
        )

        diagrammatic = frame.state_for_slot("section-bank:ellipse:0")
        self.assertEqual(diagrammatic.display_opacity, 1.0)
        self.assertIs(
            diagrammatic.occlusion_participation,
            SectionOcclusionParticipation.CERTIFIED,
        )
        self.assertIs(
            diagrammatic.depth_presentation,
            SectionDepthPresentationPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        )

    def test_handle_and_slot_can_set_different_axes_without_precedence(self) -> None:
        catalog = _catalog()
        instruction = SectionCompositingInstruction.for_catalog(
            catalog,
            overrides=(
                SectionCompositingOverride.for_handle(
                    catalog.surface.handle_id,
                    display_opacity=0.35,
                ),
                SectionCompositingOverride.for_slot(
                    "contour:0",
                    occlusion_participation="paint-only",
                    depth_presentation="diagrammatic",
                ),
            ),
        )
        frame = compile_section_compositing(catalog, instruction)

        surface = frame.state_for_slot("surface-fill:0")
        contour = frame.state_for_slot("contour:0")
        self.assertEqual(surface.display_opacity, 0.35)
        self.assertEqual(contour.display_opacity, 0.35)
        self.assertIs(
            contour.occlusion_participation,
            SectionOcclusionParticipation.PAINT_ONLY,
        )
        self.assertIs(
            contour.depth_presentation,
            SectionDepthPresentationPolicy.DIAGRAMMATIC,
        )

    def test_overlapping_assignment_of_the_same_axis_fails_closed(self) -> None:
        catalog = _catalog()
        instruction = SectionCompositingInstruction.for_catalog(
            catalog,
            overrides=(
                SectionCompositingOverride.for_handle(
                    catalog.surface.handle_id,
                    display_opacity=0.25,
                ),
                SectionCompositingOverride.for_slot(
                    "contour:0",
                    display_opacity=0.5,
                ),
            ),
        )

        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "overlapping targets",
        ):
            compile_section_compositing(catalog, instruction)

    def test_frame_queries_preserve_slot_source_and_handle_identity(self) -> None:
        catalog = _catalog()
        frame = compile_section_compositing(
            catalog,
            SectionCompositingInstruction.for_catalog(catalog),
        )

        source_states = frame.states_for_source("boundary:cone:silhouette")
        self.assertEqual(
            tuple(item.slot_id for item in source_states),
            ("contour:0", "contour:1"),
        )
        surface_states = frame.states_for_handle(
            catalog,
            catalog.surface.handle_id,
        )
        self.assertEqual(
            tuple(item.slot_id for item in surface_states),
            ("contour:0", "contour:1", "surface-fill:0"),
        )
        ellipse = frame.state_for_slot("section-bank:ellipse:0")
        self.assertEqual(ellipse.topology_bank, "ellipse")
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "unknown compositing slot",
        ):
            frame.state_for_slot("missing")
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "unknown compositing source",
        ):
            frame.states_for_source("missing")

    def test_instruction_and_frame_have_strict_canonical_json(self) -> None:
        catalog = _catalog()
        instruction = SectionCompositingInstruction.for_catalog(
            catalog,
            defaults=SectionCompositingAxes(
                display_opacity=0.8,
                occlusion_participation="paint-only",
                depth_presentation="diagrammatic",
            ),
            overrides=(
                SectionCompositingOverride.for_slot(
                    "surface-fill:0",
                    display_opacity=0.2,
                ),
                SectionCompositingOverride.for_handle(
                    catalog.section_curve.handle_id,
                    depth_presentation="physical",
                ),
            ),
        )
        reversed_instruction = SectionCompositingInstruction.for_catalog(
            catalog,
            defaults=instruction.defaults,
            overrides=tuple(reversed(instruction.overrides)),
        )

        self.assertEqual(instruction.to_json(), reversed_instruction.to_json())
        self.assertEqual(
            SectionCompositingInstruction.from_json(instruction.to_json()),
            instruction,
        )
        frame = compile_section_compositing(catalog, instruction)
        loaded_frame = SectionCompositingFrame.from_json(frame.to_json())
        self.assertEqual(loaded_frame, frame)
        self.assertEqual(loaded_frame.digest, frame.digest)
        loaded_frame.validate_catalog(catalog)
        self.assertEqual(
            json.loads(instruction.to_json())["schema"],
            SECTION_COMPOSITING_INSTRUCTION_SCHEMA,
        )
        self.assertEqual(
            json.loads(frame.to_json())["schema"],
            SECTION_COMPOSITING_FRAME_SCHEMA,
        )

    def test_catalog_binding_rejects_section_digest_and_identity_drift(self) -> None:
        catalog = _catalog()
        good = SectionCompositingInstruction.for_catalog(catalog)
        wrong_section = SectionCompositingInstruction(
            "other-section",
            catalog.digest,
        )
        changed_catalog = SectionDisplayCatalog(
            catalog.section_id,
            catalog.slots
            + (SectionSemanticSlot("plane-outline:0", "plane-outline"),),
        )
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "section_id does not match",
        ):
            compile_section_compositing(catalog, wrong_section)
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "catalog_digest does not match",
        ):
            compile_section_compositing(changed_catalog, good)

        frame = compile_section_compositing(catalog, good)
        changed_state = SectionCompositingSlotState(
            "contour:0",
            "contour",
            "boundary:cone:different",
            None,
            frame.state_for_slot("contour:0").axes,
        )
        drifted = SectionCompositingFrame(
            frame.section_id,
            frame.catalog_digest,
            tuple(
                changed_state if item.slot_id == "contour:0" else item
                for item in frame.slots
            ),
        )
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "slot/source identity",
        ):
            drifted.validate_catalog(catalog)

    def test_unknown_targets_and_duplicate_targets_are_rejected(self) -> None:
        catalog = _catalog()
        missing_slot = SectionCompositingInstruction.for_catalog(
            catalog,
            overrides=(
                SectionCompositingOverride.for_slot(
                    "missing-slot",
                    display_opacity=0.0,
                ),
            ),
        )
        missing_handle = SectionCompositingInstruction.for_catalog(
            catalog,
            overrides=(
                SectionCompositingOverride.for_handle(
                    "missing-handle",
                    display_opacity=0.0,
                ),
            ),
        )
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "unavailable",
        ):
            compile_section_compositing(catalog, missing_slot)
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "unavailable",
        ):
            compile_section_compositing(catalog, missing_handle)
        duplicate = SectionCompositingOverride.for_slot(
            "surface-fill:0",
            display_opacity=0.2,
        )
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "targets must be unique",
        ):
            SectionCompositingInstruction.for_catalog(
                catalog,
                overrides=(duplicate, duplicate),
            )

    def test_invalid_numbers_enums_and_empty_overrides_are_rejected(self) -> None:
        for value in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    SectionSemanticCompositingError,
                    "finite number between 0 and 1",
                ):
                    SectionCompositingAxes(display_opacity=value)
                with self.assertRaisesRegex(
                    SectionSemanticCompositingError,
                    "finite number between 0 and 1",
                ):
                    SectionCompositingOverride.for_slot(
                        "slot",
                        display_opacity=value,
                    )
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "SectionOcclusionParticipation",
        ):
            SectionCompositingAxes(occlusion_participation="hidden")
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "SectionDepthPresentationPolicy",
        ):
            SectionCompositingAxes(depth_presentation="front-z-index")
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "at least one axis",
        ):
            SectionCompositingOverride.for_slot("slot")

    def test_strict_json_rejects_unknown_duplicate_and_nonfinite_values(self) -> None:
        catalog = _catalog()
        value = SectionCompositingInstruction.for_catalog(catalog).to_dict()
        value["unknown"] = True
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "unknown fields",
        ):
            SectionCompositingInstruction.from_dict(value)
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "duplicate key",
        ):
            SectionCompositingInstruction.from_json(
                '{"schema":"quadric-section-compositing-instruction/v1",'
                '"sectionId":"a","sectionId":"b",'
                '"catalogDigest":"sha256:' + "0" * 64 + '",'
                '"defaults":{"displayOpacity":1.0,'
                '"occlusionParticipation":"certified",'
                '"depthPresentation":"physical"},"overrides":[]}'
            )
        with self.assertRaisesRegex(
            SectionSemanticCompositingError,
            "non-finite",
        ):
            SectionCompositingInstruction.from_json(
                '{"schema":"quadric-section-compositing-instruction/v1",'
                '"sectionId":"a",'
                '"catalogDigest":"sha256:' + "0" * 64 + '",'
                '"defaults":{"displayOpacity":NaN,'
                '"occlusionParticipation":"certified",'
                '"depthPresentation":"physical"},"overrides":[]}'
            )


if __name__ == "__main__":
    unittest.main()
