from __future__ import annotations

import json
from math import pi
import unittest

from polyhedron_visibility.quadrics.animation import (
    BranchContinuityError,
    match_tracked_section_frame,
)
from polyhedron_visibility.quadrics.contract import ConeSpec, SectionPlane
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    PlaneMotionCriticalKind,
    PlaneMotionSchedule,
    ScheduledSectionAnimation,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.sections import compute_quadric_section
from polyhedron_visibility.quadrics.transition import (
    SectionTransitionError,
    SectionTransitionMode,
    SectionTransitionRole,
    build_section_transition_plan,
    canonical_section_transition_plan_json,
)


def _cone() -> ConeSpec:
    return ConeSpec(
        "cone",
        (0.0, 0.0, -1.5),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )


def _motion() -> AxisAnglePlaneMotion:
    return AxisAnglePlaneMotion(
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
        start_time=2.0,
        end_time=8.0,
    )


def _scheduled() -> ScheduledSectionAnimation:
    return track_scheduled_plane_section("section", _cone(), _motion())


def _parabolic_knot(plan):
    return next(
        item
        for item in plan.knots
        if PlaneMotionCriticalKind.CONE_PARABOLIC.value in item.critical_kinds
    )


class SectionTransitionPlanTests(unittest.TestCase):
    def test_cone_family_change_uses_two_banks_around_exact_parabola(self) -> None:
        plan = build_section_transition_plan(_scheduled(), transition_fraction=0.04)
        knot = _parabolic_knot(plan)
        self.assertTrue(knot.left_crossfade)
        self.assertTrue(knot.right_crossfade)
        self.assertNotEqual(
            plan.frame_banks[knot.before_frame_index],
            plan.frame_banks[knot.critical_frame_index],
        )
        self.assertNotEqual(
            plan.frame_banks[knot.critical_frame_index],
            plan.frame_banks[knot.after_frame_index],
        )

        left = plan.sample(0.5 * (knot.left_start + knot.progress))
        self.assertTrue(left.transitioning)
        self.assertEqual(
            tuple(item.role for item in left.layers),
            (SectionTransitionRole.LIVE, SectionTransitionRole.CRITICAL),
        )
        self.assertAlmostEqual(sum(item.opacity for item in left.layers), 1.0)
        self.assertEqual(len({item.bank_index for item in left.layers}), 2)

        exact = plan.sample(knot.progress)
        self.assertFalse(exact.transitioning)
        self.assertIs(exact.layers[0].role, SectionTransitionRole.CRITICAL)
        self.assertEqual(exact.layers[0].geometry_progress, knot.progress)
        critical_frame = plan.scheduled.animation.frames[
            exact.layers[0].reference_frame_index
        ]
        self.assertEqual(critical_frame.signature.conic_family.value, "parabola")

        right = plan.sample(0.5 * (knot.progress + knot.right_end))
        self.assertTrue(right.transitioning)
        self.assertAlmostEqual(sum(item.opacity for item in right.layers), 1.0)
        self.assertEqual(
            tuple(item.role for item in right.layers),
            (SectionTransitionRole.LIVE, SectionTransitionRole.CRITICAL),
        )

    def test_trim_tangency_is_an_instantaneous_bank_handoff(self) -> None:
        plan = build_section_transition_plan(_scheduled())
        knot = next(
            item
            for item in plan.knots
            if item.critical_kinds
            == (PlaneMotionCriticalKind.CONE_TRIM_TANGENCY.value,)
        )
        self.assertFalse(knot.left_crossfade)
        self.assertFalse(knot.right_crossfade)
        exact = plan.sample(knot.progress)
        self.assertFalse(exact.transitioning)
        self.assertNotEqual(exact.layers[0].geometry_progress, knot.progress)
        before = plan.sample(max(0.0, knot.progress - 1.0e-5))
        after = plan.sample(min(1.0, knot.progress + 1.0e-5))
        if knot.left_changes or knot.right_changes:
            self.assertNotEqual(before.layers[0].bank_index, after.layers[0].bank_index)

    def test_sampling_is_stateless_and_reversible(self) -> None:
        plan = build_section_transition_plan(_scheduled())
        knot = _parabolic_knot(plan)
        progresses = (
            knot.left_start,
            0.5 * (knot.left_start + knot.progress),
            knot.progress,
            0.5 * (knot.progress + knot.right_end),
            knot.right_end,
        )
        forward = tuple(plan.sample(item).to_dict() for item in progresses)
        reverse = tuple(
            reversed(tuple(plan.sample(item).to_dict() for item in reversed(progresses)))
        )
        self.assertEqual(forward, reverse)

    def test_cut_mode_never_blends_banks_but_keeps_exact_parabola(self) -> None:
        plan = build_section_transition_plan(
            _scheduled(), transition_fraction=0.0, mode=SectionTransitionMode.CUT
        )
        knot = _parabolic_knot(plan)
        for progress in (
            knot.progress - 1.0e-5,
            knot.progress,
            knot.progress + 1.0e-5,
        ):
            self.assertEqual(len(plan.sample(progress).layers), 1)
        exact = plan.sample(knot.progress)
        self.assertEqual(exact.layers[0].geometry_progress, knot.progress)
        self.assertEqual(
            plan.scheduled.animation.frames[
                exact.layers[0].reference_frame_index
            ].signature.conic_family.value,
            "parabola",
        )

    def test_unscheduled_topology_change_fails_closed(self) -> None:
        scheduled = _scheduled()
        schedule = PlaneMotionSchedule(
            motion=scheduled.schedule.motion,
            surface_id=scheduled.schedule.surface_id,
            progresses=scheduled.schedule.progresses,
            samples=scheduled.schedule.samples,
            critical_events=(),
        )
        broken = ScheduledSectionAnimation(schedule, scheduled.animation)
        with self.assertRaisesRegex(SectionTransitionError, "analytic critical"):
            build_section_transition_plan(broken)

    def test_plan_json_is_strict_and_deterministic(self) -> None:
        plan = build_section_transition_plan(_scheduled())
        first = canonical_section_transition_plan_json(plan)
        self.assertEqual(first, canonical_section_transition_plan_json(plan))
        payload = json.loads(first)
        self.assertEqual(payload["mode"], "crossfade")
        self.assertTrue(payload["knots"])

    def test_invalid_transition_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(SectionTransitionError, "positive"):
            build_section_transition_plan(_scheduled(), transition_fraction=0.0)
        with self.assertRaisesRegex(SectionTransitionError, "crossfade.*cut"):
            build_section_transition_plan(_scheduled(), mode="morph")


class ReferenceFrameMatchingTests(unittest.TestCase):
    def test_live_same_topology_section_reuses_reference_capacity_slots(self) -> None:
        scheduled = _scheduled()
        reference = scheduled.animation.frames[0]
        section = compute_quadric_section(
            scheduled.animation.section_id,
            _cone(),
            _motion().plane_at(0.1),
        )
        matched = match_tracked_section_frame(
            reference, section, frame_index=7, time=3.0
        )
        self.assertEqual(matched.frame_index, 7)
        self.assertEqual(
            tuple(item.capacity_slot for item in matched.branches),
            tuple(item.capacity_slot for item in reference.branches),
        )

    def test_reference_matching_rejects_a_different_family(self) -> None:
        scheduled = _scheduled()
        oval = scheduled.animation.frames[0]
        hyperbola = scheduled.animation.frames[-1].section
        with self.assertRaisesRegex(BranchContinuityError, "topology differs"):
            match_tracked_section_frame(oval, hyperbola)


if __name__ == "__main__":
    unittest.main()
