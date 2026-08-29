from __future__ import annotations

from dataclasses import replace
from math import pi
import unittest

from manim import Scene, ValueTracker, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.authoring import QuadricSection3D
from polyhedron_visibility.quadrics.capacity import (
    QUADRIC_CAPACITY_PLAN_SCHEMA,
    QuadricCapacityHeadroom,
    QuadricCapacityPlanner,
    QuadricCapacityPlanningError,
    canonical_quadric_capacity_plan_json,
    scheduled_capacity_progresses,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
    estimate_quadric_mobject_count,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.profiles import (
    QUADRIC_FINAL_PROFILE,
    QUADRIC_PREVIEW_PROFILE,
)


VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.15),
        (0.0, 0.45, 0.85),
        (0.0, -0.85, 0.45),
    )
)


def _cone(model: ConeModel) -> ConeSpec:
    return ConeSpec(
        f"{model.value}-capacity-cone",
        (0.0, 0.0, -1.5),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )


def _static_authoring(
    scene: Scene,
    tracker: ValueTracker,
    model: ConeModel,
    *,
    limits: QuadricManimLimits,
) -> QuadricSection3D:
    def plane() -> SectionPlane:
        progress = float(tracker.get_value())
        return SectionPlane(
            f"{model.value}-capacity-plane",
            (0.0, 0.0, -0.2 + 0.8 * progress),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )

    return QuadricSection3D(
        scene,
        surface=_cone(model),
        section_id=f"{model.value}-capacity-section",
        plane=plane,
        projection=VIEW,
        **QUADRIC_PREVIEW_PROFILE.controller_kwargs(limits=limits),
    )


def _scheduled_cone():
    cone = _cone(ConeModel.CLOSED_SINGLE)
    motion = AxisAnglePlaneMotion(
        "capacity-motion",
        SectionPlane(
            "capacity-motion-plane",
            (0.0, 0.0, 0.2),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        (0.0, 0.0, 0.2),
        (0.0, 1.0, 0.0),
        0.0,
        1.2,
    )
    return track_scheduled_plane_section("capacity-transition", cone, motion)


class QuadricRenderProfileTests(unittest.TestCase):
    def test_preview_and_final_profiles_are_explicit_and_non_mutating(self) -> None:
        style = QuadricManimStyle(
            cone_lateral_fill_colors=("#123456", "#789ABC"),
            cone_cap_fill_colors=("#ABCDEF",),
        )
        preview_style = QUADRIC_PREVIEW_PROFILE.apply_style(style)
        self.assertIsNone(preview_style.cone_lateral_fill_colors)
        self.assertIsNone(preview_style.cone_cap_fill_colors)
        self.assertEqual(style.cone_lateral_fill_colors, ("#123456", "#789ABC"))
        self.assertIs(QUADRIC_FINAL_PROFILE.apply_style(style), style)

        self.assertEqual(
            QUADRIC_PREVIEW_PROFILE.manim_config(),
            {
                "renderer": "cairo",
                "pixel_width": 480,
                "pixel_height": 270,
                "frame_rate": 15.0,
            },
        )
        self.assertEqual(
            QUADRIC_PREVIEW_PROFILE.limits.max_fragments_per_curve,
            16,
        )
        self.assertEqual(
            QUADRIC_PREVIEW_PROFILE.limits.max_dashes_per_fragment,
            48,
        )
        self.assertEqual(QUADRIC_FINAL_PROFILE.pixel_width, 960)
        self.assertEqual(QUADRIC_FINAL_PROFILE.pixel_height, 540)
        self.assertTrue(QUADRIC_FINAL_PROFILE.component_shading)

    def test_shared_mobject_estimator_uses_compact_dash_slots(self) -> None:
        self.assertEqual(
            estimate_quadric_mobject_count(
                surface_count=1,
                source_count=7,
                max_fragments_per_curve=16,
                section_enabled=True,
            ),
            374,
        )
        with self.assertRaisesRegex(ValueError, "source_count"):
            estimate_quadric_mobject_count(
                surface_count=1,
                source_count=-1,
                max_fragments_per_curve=16,
                section_enabled=True,
            )


class QuadricCapacityPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 8,
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_static_closed_and_open_cones_produce_compact_replayable_limits(
        self,
    ) -> None:
        for model in (ConeModel.CLOSED_SINGLE, ConeModel.OPEN_SINGLE):
            with self.subTest(model=model.value):
                tracker = ValueTracker(0.25)
                scene = Scene()
                authoring = _static_authoring(
                    scene,
                    tracker,
                    model,
                    limits=QUADRIC_PREVIEW_PROFILE.limits,
                )
                plan = QuadricCapacityPlanner(
                    authoring,
                    progress=tracker,
                ).scan((0.0, 0.5, 1.0))

                self.assertEqual(plan.schema, QUADRIC_CAPACITY_PLAN_SCHEMA)
                self.assertFalse(plan.continuous_interval_certified)
                self.assertEqual(plan.coverage, "listed_progresses")
                self.assertEqual(tracker.get_value(), 0.25)
                self.assertFalse(authoring.attached)
                self.assertEqual(scene.mobjects, [])
                self.assertGreater(plan.peaks.plane_fragment_count, 0)
                self.assertGreater(plan.peaks.ray_classification_count, 0)
                self.assertGreater(plan.peaks.max_projected_length, 0.0)
                self.assertEqual(
                    plan.recommended_limits.max_segments_per_fragment,
                    plan.peaks.max_projected_source_segments + 8,
                )
                self.assertLess(
                    plan.recommended_limits.max_total_mobjects,
                    QUADRIC_PREVIEW_PROFILE.limits.max_total_mobjects,
                )
                self.assertEqual(
                    plan.estimated_mobject_total,
                    plan.recommended_limits.max_total_mobjects,
                )
                self.assertEqual(
                    canonical_quadric_capacity_plan_json(plan),
                    canonical_quadric_capacity_plan_json(plan),
                )

                replay_tracker = ValueTracker(0.0)
                replay = _static_authoring(
                    Scene(),
                    replay_tracker,
                    model,
                    limits=plan.recommended_limits,
                ).attach()
                identities = replay.slot_identities()
                for progress in (0.0, 0.5, 1.0):
                    replay_tracker.set_value(progress)
                    replay.update(0.0)
                    self.assertEqual(replay.slot_identities(), identities)
                replay.restore()

    def test_scene_factory_scan_has_human_summary_and_calls_factory_once(
        self,
    ) -> None:
        created: list[tuple[Scene, QuadricSection3D]] = []

        def scene_factory(progress: ValueTracker) -> QuadricSection3D:
            scene = Scene()

            def plane() -> SectionPlane:
                return SectionPlane(
                    "factory-plane",
                    (0.0, 0.0, -0.2 + 0.8 * progress.get_value()),
                    (0.7, 0.0, 1.0),
                    u_axis=(0.0, 1.0, 0.0),
                )

            authoring = QuadricSection3D(
                scene,
                surface=_cone(ConeModel.CLOSED_SINGLE),
                section_id="factory-section",
                plane=plane,
                projection=VIEW,
                render_profile="preview",
            )
            created.append((scene, authoring))
            return authoring

        plan = QuadricCapacityPlanner.scan(
            scene_factory,
            frames=range(0, 3),
        )

        self.assertEqual(len(created), 1)
        scene, authoring = created[0]
        self.assertFalse(authoring.attached)
        self.assertEqual(scene.mobjects, [])
        self.assertEqual(len(plan.samples), 3)
        self.assertEqual(
            tuple(sample.progress for sample in plan.samples),
            (0.0, 0.5, 1.0),
        )
        self.assertEqual(plan.profile_id, "preview")
        self.assertEqual(plan.recommended_profile_id, "preview")
        self.assertIn("Peak boundary sources:", plan.summary())
        chinese = plan.summary(locale="zh-CN")
        self.assertIn("边界源峰值：", chinese)
        self.assertIn("建议 profile：preview", chinese)
        self.assertIsInstance(plan.recommended_limits, QuadricManimLimits)

    def test_scene_factory_scan_rejects_invalid_frame_indices(self) -> None:
        with self.assertRaisesRegex(
            QuadricCapacityPlanningError,
            "non-negative integer",
        ):
            QuadricCapacityPlanner.scan(lambda progress: object(), frames=(-1, 0))

    def test_schedule_grid_includes_eased_frames_and_every_analytic_knot(
        self,
    ) -> None:
        scheduled = _scheduled_cone()
        progresses = scheduled_capacity_progresses(
            scheduled,
            frame_rate=2.0,
            rate_function=lambda value: value * value,
        )
        self.assertIn(0.25, progresses)
        self.assertEqual(progresses[0], 0.0)
        self.assertEqual(progresses[-1], 1.0)
        for required in scheduled.schedule.progresses:
            self.assertTrue(
                any(abs(required - value) <= 1.0e-12 for value in progresses)
            )

    def test_topology_schedule_scan_records_knots_and_restores_identity(self) -> None:
        scheduled = _scheduled_cone()
        profile = replace(
            QUADRIC_PREVIEW_PROFILE,
            profile_id="capacity-test-preview",
            frame_rate=2.0,
            section_max_screen_error=0.2,
            include_surface_boundaries=False,
        )
        tracker = ValueTracker(0.0)
        scene = Scene()
        authoring = QuadricSection3D(
            scene,
            scheduled=scheduled,
            progress=tracker,
            projection=VIEW,
            **profile.controller_kwargs(),
        )
        plan = QuadricCapacityPlanner(
            authoring,
            progress=tracker,
        ).scan_schedule(scheduled, profile=profile)

        self.assertEqual(
            plan.coverage,
            "uniform_render_grid_plus_schedule_knots",
        )
        self.assertEqual(plan.profile_id, profile.profile_id)
        self.assertEqual(plan.frame_rate, 2.0)
        self.assertIsNotNone(plan.schedule_digest)
        self.assertEqual(
            plan.required_progresses,
            scheduled.schedule.progresses,
        )
        self.assertEqual(tracker.get_value(), 0.0)
        self.assertFalse(authoring.attached)
        self.assertEqual(scene.mobjects, [])
        self.assertGreaterEqual(plan.peaks.active_curve_count, 2)

    def test_missing_required_progress_and_probe_overflow_fail_closed(self) -> None:
        tracker = ValueTracker(0.25)
        scene = Scene()
        authoring = _static_authoring(
            scene,
            tracker,
            ConeModel.CLOSED_SINGLE,
            limits=QUADRIC_PREVIEW_PROFILE.limits,
        )
        planner = QuadricCapacityPlanner(authoring, progress=tracker)
        with self.assertRaisesRegex(
            QuadricCapacityPlanningError,
            "omits required progress",
        ):
            planner.scan((0.0, 1.0), required_progresses=(0.5,))
        self.assertEqual(tracker.get_value(), 0.25)
        self.assertFalse(authoring.attached)
        self.assertEqual(scene.mobjects, [])

        tiny_limits = replace(
            QUADRIC_PREVIEW_PROFILE.limits,
            max_fragments_per_curve=1,
            max_total_mobjects=1000,
        )
        tiny_tracker = ValueTracker(0.25)
        tiny_scene = Scene()
        tiny = _static_authoring(
            tiny_scene,
            tiny_tracker,
            ConeModel.CLOSED_SINGLE,
            limits=tiny_limits,
        )
        with self.assertRaisesRegex(
            QuadricCapacityPlanningError,
            "capacity scan failed",
        ):
            QuadricCapacityPlanner(tiny, progress=tiny_tracker).scan((0.5,))
        self.assertEqual(tiny_tracker.get_value(), 0.25)
        self.assertFalse(tiny.attached)
        self.assertEqual(tiny_scene.mobjects, [])

    def test_attached_probe_failure_restores_last_good_frame_and_slots(self) -> None:
        limits = replace(
            QUADRIC_PREVIEW_PROFILE.limits,
            max_fragments_per_curve=5,
        )
        tracker = ValueTracker(0.25)
        scene = Scene()
        authoring = _static_authoring(
            scene,
            tracker,
            ConeModel.CLOSED_SINGLE,
            limits=limits,
        ).attach()
        identities = authoring.slot_identities()
        snapshot = authoring.slot_snapshot()
        scene_identity = tuple(id(item) for item in scene.mobjects)

        with self.assertRaisesRegex(
            QuadricCapacityPlanningError,
            "progress 0.75",
        ):
            QuadricCapacityPlanner(authoring, progress=tracker).scan((0.75,))

        self.assertTrue(authoring.attached)
        self.assertEqual(tracker.get_value(), 0.25)
        self.assertEqual(authoring.slot_identities(), identities)
        self.assertEqual(authoring.slot_snapshot(), snapshot)
        self.assertEqual(
            tuple(id(item) for item in scene.mobjects),
            scene_identity,
        )
        authoring.restore()

    def test_schedule_scan_rejects_a_different_controller_schedule(self) -> None:
        scheduled = _scheduled_cone()
        different_motion = replace(
            scheduled.schedule.motion,
            end_angle=1.0,
        )
        different = track_scheduled_plane_section(
            scheduled.animation.section_id,
            scheduled.schedule.samples[0].surface,
            different_motion,
        )
        profile = replace(
            QUADRIC_PREVIEW_PROFILE,
            profile_id="capacity-mismatch-preview",
            frame_rate=2.0,
            include_surface_boundaries=False,
        )
        tracker = ValueTracker(0.0)
        authoring = QuadricSection3D(
            Scene(),
            scheduled=scheduled,
            progress=tracker,
            projection=VIEW,
            **profile.controller_kwargs(),
        )
        with self.assertRaisesRegex(
            QuadricCapacityPlanningError,
            "does not match",
        ):
            QuadricCapacityPlanner(
                authoring,
                progress=tracker,
            ).scan_schedule(different, profile=profile)
        self.assertFalse(authoring.attached)

    def test_headroom_validation_rejects_false_underplanning(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be smaller than 1"):
            QuadricCapacityHeadroom(projected_length_scale=0.99)


if __name__ == "__main__":
    unittest.main()
