from __future__ import annotations

from math import pi
import unittest
from unittest.mock import patch

from manim import Mobject, Scene, ValueTracker

from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import SectionPlane
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.section_compositing import (
    QUADRIC_SECTION_COMPOSITING_LIMITS,
)
from polyhedron_visibility.quadrics.transition_manim import (
    QuadricSectionTransition3D,
    QuadricSectionTransitionManimError,
)
from tests.test_quadric_section_parameter_sweep import (
    SURFACE_RECORDS,
    SWEEP,
    VIEW_RECORDS,
    _surface,
    _view,
)


def _limits(**overrides: object) -> QuadricManimLimits:
    values: dict[str, object] = {
        "max_surfaces": 2,
        "max_curves": 10,
        "max_fragments_per_curve": 32,
        "max_segments_per_fragment": 256,
        "max_surface_segments": 512,
        "max_dashes_per_fragment": 72,
        "max_projected_length": 18.0,
        "max_total_mobjects": 60000,
        "max_boundary_sources": 64,
    }
    values.update(overrides)
    return QuadricManimLimits(**values)  # type: ignore[arg-type]


def _scheduled(record: object):
    surface = _surface(SURFACE_RECORDS[str(record["surface"])])
    pivot = (0.0, 0.0, float(record["pivotHeight"]))
    base_plane = SectionPlane(
        f"sweep:{record['id']}:plane",
        pivot,
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    motion = AxisAnglePlaneMotion(
        f"sweep:{record['id']}:motion",
        base_plane,
        pivot,
        (0.0, 1.0, 0.0),
        float(record["startAngle"]) * pi / 180.0,
        float(record["endAngle"]) * pi / 180.0,
    )
    return track_scheduled_plane_section(
        f"sweep:{record['id']}:section",
        surface,
        motion,
    )


class DeterministicConeSectionDynamicSweepTests(unittest.TestCase):
    def test_dynamic_sequences_keep_identity_capacity_and_last_good_frame(
        self,
    ) -> None:
        progress_samples = (0.0, 0.1, 0.25, 0.49, 0.5, 0.51, 0.75, 0.9, 1.0)
        for record in SWEEP["dynamicSequences"]:
            with self.subTest(sequence=record["id"]):
                scene = Scene()
                original_scene_mobjects = tuple(id(item) for item in scene.mobjects)
                progress = ValueTracker(0.0)
                controller = QuadricSectionTransition3D(
                    scene,
                    scheduled=_scheduled(record),
                    progress=progress,
                    projection=_view(VIEW_RECORDS[str(record["view"])]),
                    transition_fraction=0.04,
                    paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                    boundary_visibility_mode="unified",
                    style=QuadricManimStyle(),
                    limits=_limits(),
                    max_chord_error=0.01,
                ).attach()
                try:
                    slot_identities = controller.slot_identities()
                    slot_count = len(slot_identities)
                    scene_mobjects = tuple(id(item) for item in scene.mobjects)
                    scene_mobject_count = len(scene.mobjects)
                    allocated_curve_ids = controller.controller.allocated_curve_ids

                    for sample in progress_samples:
                        progress.set_value(sample)
                        with patch.object(
                            Mobject,
                            "__init__",
                            side_effect=AssertionError(
                                "parameter-sweep updater allocated a Mobject"
                            ),
                        ):
                            controller.update()
                        self.assertEqual(controller.slot_identities(), slot_identities)
                        self.assertEqual(len(controller.slot_identities()), slot_count)
                        self.assertEqual(
                            tuple(id(item) for item in scene.mobjects),
                            scene_mobjects,
                        )
                        self.assertEqual(len(scene.mobjects), scene_mobject_count)
                        self.assertEqual(
                            controller.controller.allocated_curve_ids,
                            allocated_curve_ids,
                        )
                        section_frame = controller.controller.last_section_frame
                        self.assertIsNotNone(section_frame)
                        assert section_frame is not None
                        self.assertLessEqual(
                            len(section_frame.plane_fragments),
                            QUADRIC_SECTION_COMPOSITING_LIMITS.max_plane_fragments,
                        )
                        self.assertLessEqual(
                            section_frame.ray_classification_count,
                            QUADRIC_SECTION_COMPOSITING_LIMITS.max_ray_classifications,
                        )
                        numeric_frame = controller.controller.last_frame
                        self.assertIsNotNone(numeric_frame)
                        assert numeric_frame is not None
                        self.assertLessEqual(
                            len(numeric_frame.curve_fragments),
                            _limits().max_curves * _limits().max_fragments_per_curve,
                        )

                    snapshot = controller.controller.slot_snapshot()
                    committed = (
                        controller.controller.last_frame,
                        controller.controller.last_global_frame,
                        controller.controller.last_section_frame,
                        controller.controller.last_boundary_frame,
                    )
                    progress.set_value(1.5)
                    with self.assertRaisesRegex(
                        QuadricSectionTransitionManimError,
                        r"lie in \[0, 1\]",
                    ):
                        controller.update()
                    self.assertEqual(controller.controller.slot_snapshot(), snapshot)
                    self.assertEqual(
                        (
                            controller.controller.last_frame,
                            controller.controller.last_global_frame,
                            controller.controller.last_section_frame,
                            controller.controller.last_boundary_frame,
                        ),
                        committed,
                    )
                    self.assertEqual(controller.slot_identities(), slot_identities)
                    self.assertEqual(
                        tuple(id(item) for item in scene.mobjects),
                        scene_mobjects,
                    )

                    progress.set_value(1.0)
                    with patch.object(
                        Mobject,
                        "__init__",
                        side_effect=AssertionError(
                            "recovery update allocated a Mobject"
                        ),
                    ):
                        controller.update()
                    self.assertEqual(controller.slot_identities(), slot_identities)
                finally:
                    controller.restore()
                self.assertEqual(
                    tuple(id(item) for item in scene.mobjects),
                    original_scene_mobjects,
                )


if __name__ == "__main__":
    unittest.main()
