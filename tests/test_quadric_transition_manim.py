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
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.manim import (
    DEFAULT_QUADRIC_VIEW,
    QuadricBoundaryStyle,
    QuadricManimCapacityError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.sections import section_cap_chord_curve_ids
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
        "max_curves": 10,
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


def _translated_isometric_scheduled():
    shift = 3.25 * np.asarray(DEFAULT_QUADRIC_VIEW.matrix[0], dtype=float)
    apex = shift + np.asarray((0.0, 0.0, -2.45), dtype=float)
    center = shift + np.asarray((0.0, 0.0, 0.2), dtype=float)
    cone = ConeSpec(
        "translated-cone",
        tuple(float(value) for value in apex),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    motion = AxisAnglePlaneMotion(
        "translated-motion",
        SectionPlane(
            "translated-plane",
            tuple(float(value) for value in center),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        tuple(float(value) for value in center),
        (0.0, 1.0, 0.0),
        0.72,
        1.35,
    )
    return track_scheduled_plane_section(
        "translated-section", cone, motion
    )


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
    def test_transition_rejects_double_cone_section_models_before_allocation(
        self,
    ) -> None:
        for model in (ConeModel.OPEN_DOUBLE, ConeModel.ANALYTIC_DOUBLE):
            with self.subTest(model=model.value):
                cone = ConeSpec(
                    f"{model.value}-cone",
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    pi / 4.0,
                    (-2.0, 2.0),
                    radial_axis=(1.0, 0.0, 0.0),
                    model=model,
                )
                motion = AxisAnglePlaneMotion(
                    f"{model.value}-motion",
                    SectionPlane(
                        f"{model.value}-plane",
                        (0.0, 0.0, 0.2),
                        (0.0, 0.0, 1.0),
                        u_axis=(1.0, 0.0, 0.0),
                    ),
                    (0.0, 0.0, 0.2),
                    (0.0, 1.0, 0.0),
                    0.0,
                    0.2,
                )
                scheduled = track_scheduled_plane_section(
                    f"{model.value}-section",
                    cone,
                    motion,
                )
                with self.assertRaisesRegex(
                    QuadricSectionTransitionManimError,
                    model.name,
                ):
                    QuadricSectionTransition3D(
                        Scene(),
                        scheduled=scheduled,
                        progress=0.0,
                        projection=VIEW,
                        limits=_limits(),
                    )

    def test_boundary_style_registry_is_forwarded_to_inner_controller(self) -> None:
        accent = QuadricBoundaryStyle(visible_color="#E53935")
        controller = QuadricSectionTransition3D(
            Scene(),
            scheduled=_scheduled(),
            progress=ValueTracker(0.0),
            projection=VIEW,
            boundary_visibility_mode="unified",
            boundary_styles={"style:curve": accent},
            limits=_limits(max_total_mobjects=30000),
        )
        self.assertIs(
            controller.controller.boundary_styles["style:curve"], accent
        )

    def test_transition_controller_inherits_isometric_quadric_default(self) -> None:
        controller = QuadricSectionTransition3D(
            Scene(),
            scheduled=_scheduled(),
            progress=0.25,
            limits=_limits(),
        )

        np.testing.assert_allclose(
            controller._controller._resolve_view().matrix,
            DEFAULT_QUADRIC_VIEW.matrix,
            atol=1.0e-12,
        )

    def test_scheduled_motion_can_reuse_one_certified_plane_patch(self) -> None:
        scheduled = _scheduled()
        with patch(
            "polyhedron_visibility.quadrics.manim.fit_plane_display_patch",
            side_effect=AssertionError("dynamic patch fitting was entered"),
        ):
            controller = QuadricSectionTransition3D(
                Scene(),
                scheduled=scheduled,
                progress=0.0,
                projection=VIEW,
                use_plane_patch_envelope=True,
                limits=_limits(),
            )
            envelope = controller.plane_patch_envelope
            self.assertIsNotNone(envelope)
            assert envelope is not None
            surface = scheduled.schedule.samples[0].surface
            patches = tuple(
                controller.controller._resolve_section_patch(
                    surface,
                    scheduled.schedule.motion.plane_at(progress),
                )
                for progress in (0.0, 0.25, 0.5, 0.75, 1.0)
            )

        self.assertTrue(all(item is envelope.patch for item in patches))
        self.assertEqual(
            {item.patch_id for item in patches},
            {"plane:motion-display-patch"},
        )

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
        self.assertIs(controller.controller.context, context)
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

    def test_scheduled_closed_cone_keeps_one_current_plane_cap_chord(self) -> None:
        scheduled = _scheduled()
        progress = ValueTracker(0.85)
        controller = QuadricSectionTransition3D(
            Scene(),
            scheduled=scheduled,
            progress=progress,
            projection=VIEW,
            transition_fraction=0.05,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            boundary_visibility_mode="unified",
            style=_style(),
            limits=_limits(max_total_mobjects=30000, max_boundary_sources=32),
            max_chord_error=0.02,
        ).attach()
        identities = controller.slot_identities()
        surface = scheduled.schedule.samples[0].surface
        semantic_cap_ids = section_cap_chord_curve_ids(
            controller.plan.section_id,
            surface,
        )
        self.assertEqual(semantic_cap_ids, ("section:cap:cap_max:chord",))
        allocated_cap_ids = {"section:cap:cap_max:chord"}
        self.assertTrue(
            allocated_cap_ids.issubset(controller.controller.allocated_curve_ids)
        )

        prepared = controller._resolve_geometry()
        active_chords = {
            item.curve_id: item
            for item in prepared.curves
            if isinstance(item, SegmentCurve)
        }
        expected_active_ids = {"section:cap:cap_max:chord"}
        self.assertEqual(set(active_chords), expected_active_ids)
        self.assertEqual(prepared.curve_opacities["section:cap:cap_max:chord"], 1.0)

        frame = controller.controller.last_boundary_frame
        assert frame is not None
        source_ids = {item.source_id for item in frame.sources}
        self.assertTrue(expected_active_ids.issubset(source_ids))

        progress.set_value(0.2)
        with patch.object(
            Mobject,
            "__init__",
            side_effect=AssertionError("cap-chord updater allocated a Mobject"),
        ):
            controller.update()
        self.assertFalse(
            any(
                isinstance(item, SegmentCurve)
                for item in controller._resolve_geometry().curves
            )
        )
        self.assertEqual(controller.slot_identities(), identities)
        controller.restore()

    def test_scheduled_cylinder_reserves_both_semantic_cap_chords(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        motion = AxisAnglePlaneMotion(
            "cylinder-motion",
            SectionPlane(
                "cylinder-plane",
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.2),
                u_axis=(0.0, 1.0, 0.0),
            ),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            0.0,
            0.2,
        )
        scheduled = track_scheduled_plane_section(
            "cylinder-section",
            cylinder,
            motion,
        )
        controller = QuadricSectionTransition3D(
            Scene(),
            scheduled=scheduled,
            progress=0.5,
            projection=VIEW,
            boundary_visibility_mode="unified",
            limits=_limits(max_total_mobjects=30000, max_boundary_sources=32),
        ).attach()
        try:
            cap_ids = {
                "cylinder-section:cap:cap_min:chord",
                "cylinder-section:cap:cap_max:chord",
            }
            self.assertTrue(
                cap_ids.issubset(controller.controller.allocated_curve_ids)
            )
            active = {
                item.curve_id
                for item in controller._resolve_geometry().curves
                if isinstance(item, SegmentCurve)
            }
            self.assertEqual(active, cap_ids)
        finally:
            controller.restore()

    def test_scheduled_cap_slots_fail_closed_when_curve_capacity_is_too_small(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            QuadricManimCapacityError,
            "curve count exceeds fixed limit 8",
        ):
            QuadricSectionTransition3D(
                Scene(),
                scheduled=_scheduled(),
                progress=0.0,
                projection=VIEW,
                limits=_limits(max_curves=8),
            )

    def test_unified_boundaries_survive_all_topology_families(self) -> None:
        progress = ValueTracker(0.0)
        controller = QuadricSectionTransition3D(
            Scene(),
            scheduled=_scheduled(),
            progress=progress,
            projection=VIEW,
            transition_fraction=0.05,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            boundary_visibility_mode="unified",
            style=_style(),
            limits=_limits(
                max_total_mobjects=30000,
                max_boundary_sources=32,
            ),
            max_chord_error=0.02,
        ).attach()
        identities = controller.slot_identities()
        parabolic = next(
            item
            for item in controller.plan.knots
            if "cone_parabolic" in item.critical_kinds
        )
        samples = (
            max(0.0, 0.5 * parabolic.left_start),
            parabolic.progress,
            min(1.0, 0.5 * (parabolic.right_end + 1.0)),
        )
        families: set[str] = set()
        for sample in samples:
            progress.set_value(sample)
            with patch.object(
                Mobject,
                "__init__",
                side_effect=AssertionError(
                    "unified transition updater allocated a Mobject"
                ),
            ):
                controller.update()
            families.update(
                item.conic_family.value for item in controller.active_signatures
            )
            boundary = controller.controller.last_boundary_frame
            self.assertIsNotNone(boundary)
            assert boundary is not None
            source_ids = {item.source_id for item in boundary.sources}
            self.assertTrue(
                any(item.startswith("boundary:plane:plane:edge:") for item in source_ids)
            )
            self.assertIn("boundary:cone:cap_max:rim", source_ids)
            self.assertEqual(controller.slot_identities(), identities)
        self.assertEqual(families, {"oval", "parabola", "hyperbola"})
        controller.restore()

    def test_translated_crossfade_uses_exact_plane_roles_without_cycle(self) -> None:
        controller = QuadricSectionTransition3D(
            Scene(),
            scheduled=_translated_isometric_scheduled(),
            progress=0.475,
            projection=DEFAULT_QUADRIC_VIEW,
            transition_fraction=0.055,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            boundary_visibility_mode="unified",
            style=_style(),
            limits=_limits(
                max_fragments_per_curve=32,
                max_total_mobjects=60000,
                max_boundary_sources=48,
            ),
            max_chord_error=0.008,
        ).attach()
        frame = controller.controller.last_boundary_frame
        self.assertIsNotNone(frame)
        assert frame is not None
        hidden_transition = tuple(
            item
            for item in frame.fragments
            if ":transition:bank:1:" in item.source_id
            and item.effective_visibility_kind.value == "hidden"
        )
        self.assertTrue(hidden_transition)
        self.assertTrue(
            all(
                "outside_projection" not in item.plane_depth_roles
                for item in hidden_transition
            )
        )
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
                    paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                    boundary_visibility_mode="unified",
                    style=_style(),
                    limits=_limits(
                        max_total_mobjects=30000,
                        max_boundary_sources=32,
                    ),
                    max_chord_error=0.025,
                ).attach()
                identities = controller.slot_identities()
                families: set[str] = set()
                maximum_layers = 0
                unified_frame_count = 0
                cap_chord_frame_count = 0

                def capture(value: Mobject, dt: float) -> None:
                    nonlocal maximum_layers, unified_frame_count, cap_chord_frame_count
                    del value, dt
                    families.update(
                        item.conic_family.value
                        for item in controller.active_signatures
                    )
                    maximum_layers = max(
                        maximum_layers, len(controller.transition_frame.layers)
                    )
                    if controller.controller.last_boundary_frame is not None:
                        unified_frame_count += 1
                        source_ids = {
                            item.source_id
                            for item in controller.controller.last_boundary_frame.sources
                        }
                        if any(
                            ":cap:cap_max:chord" in item for item in source_ids
                        ):
                            cap_chord_frame_count += 1

                controller.controller._update_driver.add_updater(capture)
                inner_self.play(
                    progress.animate.set_value(1.0),
                    run_time=3.0,
                    rate_func=linear,
                )
                inner_self.families = families
                inner_self.maximum_layers = maximum_layers
                inner_self.unified_frame_count = unified_frame_count
                inner_self.cap_chord_frame_count = cap_chord_frame_count
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
            self.assertGreater(scene.unified_frame_count, 0)
            self.assertGreater(scene.cap_chord_frame_count, 0)
            self.assertTrue(scene.identity_stable)


if __name__ == "__main__":
    unittest.main()
