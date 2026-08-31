from __future__ import annotations

from dataclasses import replace
import json
import unittest

from polyhedron_visibility.quadrics.contract import (
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.parallel_plane_motion import (
    ParallelPlaneTranslation,
)
from polyhedron_visibility.quadrics.section_timeline import (
    compile_section_timeline,
)
from polyhedron_visibility.quadrics.section_timeline_transition import (
    SECTION_TIMELINE_TRANSITION_SCHEMA,
    SectionTimelineLayerRole,
    SectionTimelineTransitionError,
    SectionTimelineTransitionLayer,
    build_section_timeline_transition_plan,
    section_timeline_transition_state_at,
)


def _sphere_timeline():
    sphere = SphereSpec("transition-sphere", (0.0, 0.0, 0.0), 1.0)
    plane = SectionPlane(
        "transition-plane",
        (0.0, 0.0, -2.0),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    return compile_section_timeline(
        "transition-section",
        sphere,
        (
            ParallelPlaneTranslation(
                "transition-motion",
                plane,
                (0.0, 0.0, 4.0),
                start_time=0.0,
                end_time=2.0,
            ),
        ),
    )


class QuadricSectionTimelineTransitionTests(unittest.TestCase):
    def test_sphere_critical_frames_get_global_two_bank_crossfades(self) -> None:
        timeline = _sphere_timeline()
        plan = build_section_timeline_transition_plan(
            timeline,
            transition_fraction=0.25,
        )
        self.assertEqual(plan.schema, SECTION_TIMELINE_TRANSITION_SCHEMA)
        self.assertEqual(len(plan.knots), 2)
        for knot in plan.knots:
            self.assertTrue(knot.left_crossfade)
            self.assertTrue(knot.right_crossfade)
            self.assertEqual(knot.before_bank, knot.after_bank)
            self.assertNotEqual(knot.before_bank, knot.critical_bank)

            left_time = 0.5 * (knot.left_start + knot.critical_time)
            left = section_timeline_transition_state_at(plan, left_time)
            self.assertEqual(len(left.layers), 2)
            self.assertEqual(
                {item.bank_index for item in left.layers},
                {knot.before_bank, knot.critical_bank},
            )
            self.assertAlmostEqual(sum(item.opacity for item in left.layers), 1.0)

            exact = section_timeline_transition_state_at(
                plan,
                knot.critical_time,
            )
            self.assertEqual(len(exact.layers), 1)
            self.assertIs(
                exact.layers[0].role,
                SectionTimelineLayerRole.EXACT_CRITICAL,
            )
            self.assertEqual(exact.layers[0].bank_index, knot.critical_bank)

            right_time = 0.5 * (knot.critical_time + knot.right_end)
            right = section_timeline_transition_state_at(plan, right_time)
            self.assertEqual(len(right.layers), 2)
            self.assertEqual(
                {item.bank_index for item in right.layers},
                {knot.critical_bank, knot.after_bank},
            )

        payload = json.loads(plan.to_json())
        self.assertEqual(payload["schema"], SECTION_TIMELINE_TRANSITION_SCHEMA)
        self.assertEqual(payload["frameBanks"], list(timeline.topology_frame_banks))
        self.assertTrue(payload["knots"][0]["leftTopologyEventIds"])
        self.assertTrue(payload["knots"][0]["leftCriticalEventIds"])
        self.assertEqual(
            plan.to_json(),
            build_section_timeline_transition_plan(timeline).to_json(),
        )

    def test_crossfade_uses_smoothstep_and_has_single_layer_boundaries(self) -> None:
        timeline = _sphere_timeline()
        plan = build_section_timeline_transition_plan(timeline)
        knot = plan.knots[0]

        left_boundary = section_timeline_transition_state_at(
            plan,
            knot.left_start,
        )
        right_boundary = section_timeline_transition_state_at(
            plan,
            knot.right_end,
        )
        self.assertEqual(len(left_boundary.layers), 1)
        self.assertEqual(len(right_boundary.layers), 1)
        self.assertEqual(left_boundary.layers[0].bank_index, knot.before_bank)
        self.assertEqual(right_boundary.layers[0].bank_index, knot.after_bank)
        self.assertTrue(
            all(
                item.opacity > 0.0
                for state in (left_boundary, right_boundary)
                for item in state.layers
            )
        )

        quarter = knot.left_start + 0.25 * (
            knot.critical_time - knot.left_start
        )
        blended = section_timeline_transition_state_at(plan, quarter)
        critical = next(
            item
            for item in blended.layers
            if item.role is SectionTimelineLayerRole.EXACT_CRITICAL
        )
        live = next(
            item
            for item in blended.layers
            if item.role is SectionTimelineLayerRole.LIVE_BEFORE
        )
        self.assertAlmostEqual(critical.opacity, 0.15625)
        self.assertEqual(critical.geometry_time, knot.critical_time)
        self.assertEqual(live.geometry_time, quarter)

    def test_cut_mode_never_blends_banks(self) -> None:
        timeline = _sphere_timeline()
        plan = build_section_timeline_transition_plan(
            timeline,
            transition_fraction=0.0,
            mode="cut",
        )
        self.assertTrue(
            all(
                not knot.left_crossfade and not knot.right_crossfade
                for knot in plan.knots
            )
        )
        for sample in timeline.samples:
            state = section_timeline_transition_state_at(plan, sample.time)
            self.assertEqual(len(state.layers), 1)

    def test_finite_trim_topology_uses_instantaneous_handoffs(self) -> None:
        cylinder = CylinderSpec(
            "transition-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "transition-cylinder-plane",
            (0.0, 0.0, -3.0),
            (1.0, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        timeline = compile_section_timeline(
            "transition-cylinder-section",
            cylinder,
            (
                ParallelPlaneTranslation(
                    "transition-cylinder-motion",
                    plane,
                    (0.0, 0.0, 6.0),
                    start_time=0.0,
                    end_time=6.0,
                ),
            ),
        )
        plan = build_section_timeline_transition_plan(timeline)
        self.assertTrue(plan.knots)
        self.assertTrue(
            all(
                not knot.left_crossfade and not knot.right_crossfade
                for knot in plan.knots
            )
        )
        for knot in plan.knots:
            state = section_timeline_transition_state_at(
                plan,
                knot.critical_time,
            )
            self.assertEqual(len(state.layers), 1)
            self.assertTrue(knot.pure_trim_tangency)
            self.assertIs(state.layers[0].role, SectionTimelineLayerRole.LIVE)
            self.assertNotEqual(state.layers[0].geometry_time, knot.critical_time)
            if knot.before_frame_index is not None:
                neighbor = timeline.samples[knot.before_frame_index].time
                self.assertEqual(
                    state.layers[0].reference_frame_index,
                    knot.before_frame_index,
                )
                self.assertGreaterEqual(state.layers[0].geometry_time, neighbor)
                self.assertLess(state.layers[0].geometry_time, knot.critical_time)
            else:
                assert knot.after_frame_index is not None
                neighbor = timeline.samples[knot.after_frame_index].time
                self.assertEqual(
                    state.layers[0].reference_frame_index,
                    knot.after_frame_index,
                )
                self.assertGreater(state.layers[0].geometry_time, knot.critical_time)
                self.assertLessEqual(state.layers[0].geometry_time, neighbor)

    def test_public_plan_rebuilds_and_rejects_missing_or_forged_knots(self) -> None:
        timeline = _sphere_timeline()
        plan = build_section_timeline_transition_plan(timeline)
        with self.assertRaisesRegex(
            SectionTimelineTransitionError,
            "canonical SectionTimeline evidence",
        ):
            replace(plan, knots=())

        forged = replace(
            plan.knots[0],
            critical_frame_index=len(timeline.samples),
        )
        with self.assertRaisesRegex(
            SectionTimelineTransitionError,
            "canonical SectionTimeline evidence",
        ):
            replace(plan, knots=(forged, *plan.knots[1:]))

    def test_frame_indices_and_banks_strictly_reject_bool(self) -> None:
        plan = build_section_timeline_transition_plan(_sphere_timeline())
        with self.assertRaisesRegex(
            SectionTimelineTransitionError,
            "non-negative integer",
        ):
            replace(plan.knots[0], critical_frame_index=True)
        with self.assertRaisesRegex(
            SectionTimelineTransitionError,
            "integer 0 or 1",
        ):
            replace(plan.knots[0], critical_bank=True)
        with self.assertRaisesRegex(
            SectionTimelineTransitionError,
            "integer 0 or 1",
        ):
            SectionTimelineTransitionLayer(
                bank_index=True,
                geometry_time=0.0,
                opacity=1.0,
                role=SectionTimelineLayerRole.LIVE,
                reference_frame_index=0,
            )
        with self.assertRaisesRegex(
            SectionTimelineTransitionError,
            "non-negative integer",
        ):
            SectionTimelineTransitionLayer(
                bank_index=0,
                geometry_time=0.0,
                opacity=1.0,
                role=SectionTimelineLayerRole.LIVE,
                reference_frame_index=True,
            )

    def test_segment_join_groups_both_sides_with_causal_evidence(self) -> None:
        sphere = SphereSpec("joined-transition-sphere", (0.0, 0.0, 0.0), 1.0)
        initial_plane = SectionPlane(
            "joined-transition-plane",
            (0.0, 0.0, -2.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        first = ParallelPlaneTranslation(
            "joined-transition-first",
            initial_plane,
            (0.0, 0.0, 1.0),
            start_time=0.0,
            end_time=1.0,
        )
        second = ParallelPlaneTranslation(
            "joined-transition-second",
            first.plane_at(1.0),
            (0.0, 0.0, 2.0),
            start_time=1.0,
            end_time=3.0,
        )
        timeline = compile_section_timeline(
            "joined-transition-section",
            sphere,
            (first, second),
        )
        plan = build_section_timeline_transition_plan(timeline)
        joined = tuple(
            item for item in plan.knots if item.critical_time == 1.0
        )
        self.assertEqual(len(joined), 1)
        knot = joined[0]
        self.assertEqual(len(knot.left_topology_event_ids), 1)
        self.assertEqual(len(knot.right_topology_event_ids), 1)
        self.assertTrue(knot.left_critical_event_ids)
        self.assertTrue(knot.right_critical_event_ids)
        self.assertEqual(
            set(knot.topology_event_ids),
            set(knot.left_topology_event_ids)
            | set(knot.right_topology_event_ids),
        )
        self.assertEqual(
            set(knot.critical_event_ids),
            set(knot.left_critical_event_ids)
            | set(knot.right_critical_event_ids),
        )
        self.assertTrue(
            all(
                left.right_end <= right.left_start
                for left, right in zip(plan.knots, plan.knots[1:])
            )
        )

    def test_invalid_fraction_and_out_of_range_time_fail_explicitly(self) -> None:
        timeline = _sphere_timeline()
        with self.assertRaisesRegex(
            SectionTimelineTransitionError,
            r"\[0, 0.5\]",
        ):
            build_section_timeline_transition_plan(
                timeline,
                transition_fraction=0.75,
            )
        plan = build_section_timeline_transition_plan(timeline)
        with self.assertRaisesRegex(
            SectionTimelineTransitionError,
            "outside the SectionTimeline",
        ):
            section_timeline_transition_state_at(plan, -1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
