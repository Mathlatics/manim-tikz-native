from __future__ import annotations

from math import pi
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import Mobject, Scene, ValueTracker, linear, tempconfig

from polyhedron_visibility.geometry import GeometryContext
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import ConeSpec, SectionPlane, SphereSpec
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.manim import (
    QuadricManimCapacityError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.transition import SectionTransitionRole
from polyhedron_visibility.quadrics.transition_manim import (
    QuadricSectionTransition3D,
    QuadricSectionTransitionManimError,
)


VIEW = ParallelView.from_matrix(
    ((1.0, 0.0, 0.15), (0.0, 0.45, 0.85), (0.0, -0.85, 0.45))
)


def _limits(**overrides: object) -> QuadricManimLimits:
    values: dict[str, object] = {
        "max_surfaces": 2,
        "max_curves": 8,
        "max_fragments_per_curve": 16,
        "max_segments_per_fragment": 256,
        "max_surface_segments": 512,
        "max_dashes_per_fragment": 72,
        "max_projected_length": 18.0,
        "max_total_mobjects": 12000,
    }
    values.update(overrides)
    return QuadricManimLimits(**values)  # type: ignore[arg-type]


def _scheduled():
    cone = ConeSpec(
        "cone",
        (0.0, 0.0, -1.5),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    motion = AxisAnglePlaneMotion(
        "motion",
        SectionPlane(
            "plane",
            (0.0, 0.0, 0.2),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        (0.0, 0.0, 0.2),
        (0.0, 1.0, 0.0),
        0.0,
        1.2,
    )
    return track_scheduled_plane_section("section", cone, motion)


def _style() -> QuadricManimStyle:
    return QuadricManimStyle(
        surface_fill_color="#315f91",
        surface_fill_opacity=1.0,
        surface_stroke_color="#17324f",
        surface_stroke_width=1.0,
        visible_curve_color="#f4c542",
        visible_curve_width=4.0,
        hidden_curve_color="#f4c542",
        hidden_curve_width=3.0,
        dash_length=0.10,
        dash_gap=0.08,
    )


class AllocatedCurveBankTests(unittest.TestCase):
    def test_controller_accepts_preallocated_active_subsets_and_opacity(self) -> None:
        scene = Scene()
        state = {"ids": ("a",), "opacity": 0.25}

        def curves():
            records = {
                "a": SegmentCurve("a", (-1.0, 0.0, 1.2), (1.0, 0.0, 1.2)),
                "b": SegmentCurve("b", (0.0, -1.0, 1.2), (0.0, 1.0, 1.2)),
            }
            return tuple(records[item] for item in state["ids"])

        def opacities():
            return {item: state["opacity"] for item in state["ids"]}

        controller = QuadricOcclusion3D(
            scene,
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=curves,
            allocated_curve_ids=("a", "b"),
            curve_opacities=opacities,
            projection=ParallelView.from_matrix(np.eye(3)),
            limits=_limits(max_curves=2),
        ).attach()
        identities = controller.slot_identities()
        prepared = controller.prepare()
        self.assertEqual(prepared.numeric.curve_opacities, {"a": 0.25})

        state["ids"] = ("b",)
        state["opacity"] = 0.8
        controller.update()
        prepared = controller.prepare()
        self.assertEqual(prepared.numeric.curve_opacities, {"b": 0.8})
        self.assertEqual(controller.slot_identities(), identities)

        state["ids"] = ()
        controller.update()
        self.assertFalse(controller.last_frame.curve_fragments)
        self.assertEqual(controller.slot_identities(), identities)
        controller.restore()

    def test_unallocated_curve_fails_before_commit(self) -> None:
        state = {"curve": "a"}

        def curves():
            return (
                SegmentCurve(
                    state["curve"], (-1.0, 0.0, 1.2), (1.0, 0.0, 1.2)
                ),
            )

        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=curves,
            allocated_curve_ids=("a",),
            projection=ParallelView.from_matrix(np.eye(3)),
            limits=_limits(max_curves=1),
        ).attach()
        snapshot = controller.slot_snapshot()
        state["curve"] = "unknown"
        with self.assertRaisesRegex(QuadricManimCapacityError, "not preallocated"):
            controller.update()
        self.assertEqual(controller.slot_snapshot(), snapshot)
        controller.restore()


class QuadricSectionTransitionControllerTests(unittest.TestCase):
    def test_explicit_geometry_context_is_shared_by_live_frames(self) -> None:
        context = GeometryContext(screen_tolerance=1.0e-4)
        base = _scheduled()
        scheduled = track_scheduled_plane_section(
            "section",
            base.schedule.samples[0].surface,
            base.schedule.motion,
            context=context,
        )
        controller = QuadricSectionTransition3D(
            Scene(),
            scheduled=scheduled,
            progress=0.25,
            projection=VIEW,
            context=context,
            limits=_limits(),
        ).attach()
        self.assertIs(controller.context, context)
        controller.update()
        controller.restore()

    def test_crossfade_keeps_slots_and_full_occlusion_controller(self) -> None:
        progress = ValueTracker(0.0)
        controller = QuadricSectionTransition3D(
            Scene(),
            scheduled=_scheduled(),
            progress=progress,
            projection=VIEW,
            transition_fraction=0.05,
            style=_style(),
            limits=_limits(),
            max_chord_error=0.015,
        ).attach()
        identities = controller.slot_identities()
        parabolic = next(
            item
            for item in controller.plan.knots
            if "cone_parabolic" in item.critical_kinds
        )

        progress.set_value(0.5 * (parabolic.left_start + parabolic.progress))
        with patch.object(
            Mobject,
            "__init__",
            side_effect=AssertionError("transition updater allocated a Mobject"),
        ):
            controller.update()
        frame = controller.transition_frame
        self.assertTrue(frame.transitioning)
        self.assertEqual(
            tuple(item.role for item in frame.layers),
            (SectionTransitionRole.LIVE, SectionTransitionRole.CRITICAL),
        )
        self.assertEqual(
            {item.conic_family.value for item in controller.active_signatures},
            {"oval", "parabola"},
        )
        prepared = controller.controller.prepare()
        self.assertEqual(
            set(prepared.numeric.curve_opacities),
            {item.curve_id for item in controller._resolve_geometry().curves},
        )
        self.assertEqual(controller.slot_identities(), identities)

        progress.set_value(parabolic.progress)
        controller.update()
        self.assertEqual(
            tuple(item.conic_family.value for item in controller.active_signatures),
            ("parabola",),
        )
        self.assertEqual(controller.slot_identities(), identities)

        progress.set_value(0.5 * (parabolic.progress + parabolic.right_end))
        controller.update()
        self.assertEqual(
            {item.conic_family.value for item in controller.active_signatures},
            {"parabola", "hyperbola"},
        )
        self.assertEqual(controller.slot_identities(), identities)

        progress.set_value(0.2)
        controller.update()
        self.assertEqual(controller.slot_identities(), identities)
        controller.restore()

    def test_invalid_progress_fails_without_changing_committed_frame(self) -> None:
        progress = ValueTracker(0.0)
        controller = QuadricSectionTransition3D(
            Scene(),
            scheduled=_scheduled(),
            progress=progress,
            projection=VIEW,
            limits=_limits(),
        ).attach()
        snapshot = controller.controller.slot_snapshot()
        progress.set_value(1.5)
        with self.assertRaisesRegex(
            QuadricSectionTransitionManimError, r"lie in \[0, 1\]"
        ):
            controller.update()
        self.assertEqual(controller.controller.slot_snapshot(), snapshot)
        controller.restore()

    def test_real_cairo_animation_visits_all_three_conic_families(self) -> None:
        class TransitionScene(Scene):
            def construct(inner_self) -> None:
                progress = ValueTracker(0.0)
                controller = QuadricSectionTransition3D(
                    inner_self,
                    scheduled=_scheduled(),
                    progress=progress,
                    projection=VIEW,
                    transition_fraction=0.055,
                    style=_style(),
                    limits=_limits(),
                    max_chord_error=0.025,
                ).attach()
                identities = controller.slot_identities()
                families: set[str] = set()
                maximum_layers = 0

                def capture(value: Mobject, dt: float) -> None:
                    nonlocal maximum_layers
                    del value, dt
                    families.update(
                        item.conic_family.value
                        for item in controller.active_signatures
                    )
                    maximum_layers = max(
                        maximum_layers, len(controller.transition_frame.layers)
                    )

                controller.controller._update_driver.add_updater(capture)
                inner_self.play(
                    progress.animate.set_value(1.0),
                    run_time=3.0,
                    rate_func=linear,
                )
                inner_self.families = families
                inner_self.maximum_layers = maximum_layers
                inner_self.identity_stable = identities == controller.slot_identities()
                controller.restore()

        with (
            TemporaryDirectory() as media_dir,
            tempconfig(
                {
                    "renderer": "cairo",
                    "media_dir": media_dir,
                    "pixel_width": 240,
                    "pixel_height": 136,
                    "frame_rate": 12,
                    "disable_caching": True,
                    "write_to_movie": True,
                    "save_last_frame": False,
                }
            ),
        ):
            scene = TransitionScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertEqual(scene.families, {"oval", "parabola", "hyperbola"})
            self.assertEqual(scene.maximum_layers, 2)
            self.assertTrue(scene.identity_stable)


if __name__ == "__main__":
    unittest.main()
