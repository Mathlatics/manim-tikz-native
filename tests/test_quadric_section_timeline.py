"""Renderer-neutral acceptance for analytic quadric SectionTimeline plans."""

from __future__ import annotations

from dataclasses import replace
import json
from math import pi, sin, sqrt
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from polyhedron_visibility.quadrics.animation import (
    SectionConicFamily,
    track_quadric_section_animation,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.parallel_plane_motion import (
    PARALLEL_PLANE_MOTION_SCHEDULE_SCHEMA,
    ParallelPlaneMotionError,
    ParallelPlaneTranslation,
    canonical_parallel_plane_motion_schedule_json,
    compute_parallel_plane_motion_schedule,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    PlaneMotionCriticalKind,
    compute_plane_motion_schedule,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.section_timeline import (
    SECTION_TIMELINE_SCHEMA,
    SectionTimeline,
    SectionTimelineError,
    canonical_section_timeline_json,
    compile_section_timeline,
)
import polyhedron_visibility.quadrics.section_timeline as timeline_module


ROOT = Path(__file__).resolve().parents[1]


def _plane(
    point: tuple[float, float, float],
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    *,
    plane_id: str = "timeline-plane",
) -> SectionPlane:
    return SectionPlane(
        plane_id,
        point,
        normal,
        u_axis=(1.0, 0.0, 0.0) if normal == (0.0, 0.0, 1.0) else (0.0, 1.0, 0.0),
    )


def _closed_cone() -> ConeSpec:
    return ConeSpec(
        "timeline-cone",
        (0.0, 0.0, -1.5),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.CLOSED_SINGLE,
    )


class ParallelPlaneMotionTests(unittest.TestCase):
    def test_sphere_tangencies_are_analytic_and_json_is_canonical(self) -> None:
        sphere = SphereSpec("timeline-sphere", (0.0, 0.0, 0.0), 1.0)
        motion = ParallelPlaneTranslation(
            "sphere-translation",
            _plane((0.0, 0.0, -2.0)),
            (0.0, 0.0, 4.0),
            start_time=2.0,
            end_time=6.0,
        )
        schedule = compute_parallel_plane_motion_schedule(
            sphere,
            motion,
            authored_progresses=(0.123456789,),
            include_interval_midpoints=False,
        )
        self.assertEqual(schedule.schema, PARALLEL_PLANE_MOTION_SCHEDULE_SCHEMA)
        self.assertEqual(
            tuple(item.progress for item in schedule.critical_events),
            (0.25, 0.75),
        )
        self.assertEqual(
            tuple(item.time for item in schedule.critical_events),
            (3.0, 5.0),
        )
        self.assertEqual(
            tuple(item.normal_offsets for item in schedule.critical_events),
            ((1.0,), (3.0,)),
        )
        self.assertIn(0.123456789, schedule.progresses)
        first = canonical_parallel_plane_motion_schedule_json(schedule)
        second = canonical_parallel_plane_motion_schedule_json(
            compute_parallel_plane_motion_schedule(
                sphere,
                motion,
                authored_progresses=(0.123456789,),
                include_interval_midpoints=False,
            )
        )
        self.assertEqual(first, second)
        self.assertEqual(
            json.loads(first)["schema"],
            PARALLEL_PLANE_MOTION_SCHEDULE_SCHEMA,
        )

    def test_large_common_translation_uses_relative_height_evidence(self) -> None:
        shift = 1.0e12
        sphere = SphereSpec("shifted-sphere", (0.0, 0.0, shift), 0.002)
        motion = ParallelPlaneTranslation(
            "shifted-translation",
            _plane((0.0, 0.0, shift - 0.005)),
            (0.0, 0.0, 0.01),
        )
        schedule = compute_parallel_plane_motion_schedule(
            sphere,
            motion,
            include_interval_midpoints=False,
        )
        self.assertEqual(len(schedule.critical_events), 2)
        np.testing.assert_allclose(
            tuple(item.progress for item in schedule.critical_events),
            (0.3, 0.7),
            atol=6.0e-4,
            rtol=0.0,
        )
        self.assertTrue(all(not item.persistent for item in schedule.critical_events))

    def test_distinct_critical_levels_that_collapse_in_progress_fail_closed(
        self,
    ) -> None:
        sphere = SphereSpec("wide-sweep-sphere", (0.0, 0.0, 0.0), 1.0)
        motion = ParallelPlaneTranslation(
            "wide-sweep",
            _plane((0.0, 0.0, -1.0e15)),
            (0.0, 0.0, 2.0e15),
        )
        with self.assertRaisesRegex(
            ParallelPlaneMotionError,
            "distinct critical levels collapse",
        ):
            compute_parallel_plane_motion_schedule(sphere, motion)

        rounded_levels = ParallelPlaneTranslation(
            "rounded-wide-sweep",
            _plane((0.0, 0.0, -1.0e20)),
            (0.0, 0.0, 2.0e20),
        )
        with self.assertRaisesRegex(
            ParallelPlaneMotionError,
            "equality is uncertified",
        ):
            compute_parallel_plane_motion_schedule(sphere, rounded_levels)

        almost_past_tangency = ParallelPlaneTranslation(
            "ambiguous-endpoint",
            _plane((0.0, 0.0, -1.0 + 1.0e-13)),
            (0.0, 0.0, 1.0),
        )
        with self.assertRaisesRegex(
            ParallelPlaneMotionError,
            "ambiguous at a motion endpoint",
        ):
            compute_parallel_plane_motion_schedule(
                sphere,
                almost_past_tangency,
            )

    def test_persistent_orientation_and_unresolved_normal_motion_are_explicit(
        self,
    ) -> None:
        cylinder = CylinderSpec(
            "parallel-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-2.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        parallel_plane = SectionPlane(
            "parallel-plane",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        schedule = compute_parallel_plane_motion_schedule(
            cylinder,
            ParallelPlaneTranslation(
                "tangential-translation",
                parallel_plane,
                (0.0, 1.0, 0.0),
            ),
        )
        persistent = tuple(item for item in schedule.critical_events if item.persistent)
        self.assertEqual(len(persistent), 1)
        self.assertEqual(
            persistent[0].kinds,
            (PlaneMotionCriticalKind.CYLINDER_AXIS_PARALLEL,),
        )

        with self.assertRaisesRegex(
            ParallelPlaneMotionError,
            "below numeric resolution",
        ):
            cancellation_plane = SectionPlane(
                "cancellation-plane",
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                u_axis=(0.0, 0.0, 1.0),
            )
            compute_parallel_plane_motion_schedule(
                cylinder,
                ParallelPlaneTranslation(
                    "unresolved-translation",
                    cancellation_plane,
                    (1.0, -1.0 + 1.0e-16, 0.0),
                ),
            )

    def test_structural_parallel_and_parabolic_levels_are_canonicalized(
        self,
    ) -> None:
        cylinder = CylinderSpec(
            "skew-axis-cylinder",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            1.0,
            (-2.0e6, 3.0e6),
            radial_axis=(1.0, -1.0, 0.0),
        )
        cylinder_normal = cylinder.frame.x_axis
        cylinder_plane = SectionPlane(
            "skew-axis-plane",
            tuple(-2.0 * item for item in cylinder_normal),
            cylinder_normal,
            u_axis=cylinder.frame.y_axis,
        )
        cylinder_schedule = compute_parallel_plane_motion_schedule(
            cylinder,
            ParallelPlaneTranslation(
                "skew-axis-translation",
                cylinder_plane,
                tuple(4.0 * item for item in cylinder_normal),
            ),
            include_interval_midpoints=False,
        )
        cylinder_events = tuple(
            item for item in cylinder_schedule.critical_events if not item.persistent
        )
        self.assertEqual(len(cylinder_events), 2)
        self.assertTrue(all(len(item.equations) == 2 for item in cylinder_events))

        beta = 0.61
        cone = ConeSpec(
            "parabolic-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            beta,
            (0.7, 2.4),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.CLOSED_SINGLE,
        )
        normal = (np.cos(beta), 0.0, np.sin(beta))
        cone_plane = SectionPlane(
            "parabolic-plane",
            tuple(-1.0 * item for item in normal),
            normal,
            u_axis=(0.0, 1.0, 0.0),
        )
        cone_schedule = compute_parallel_plane_motion_schedule(
            cone,
            ParallelPlaneTranslation(
                "parabolic-translation",
                cone_plane,
                tuple(2.0 * item for item in normal),
            ),
            include_interval_midpoints=False,
        )
        apex = next(
            item
            for item in cone_schedule.critical_events
            if PlaneMotionCriticalKind.CONE_APEX_DEGENERACY in item.kinds
        )
        self.assertIn(PlaneMotionCriticalKind.CONE_TRIM_TANGENCY, apex.kinds)
        self.assertEqual(len(apex.equations), 3)

        tiny_sphere = SphereSpec("tiny-sphere", (0.0, 0.0, 0.0), 1.0e-6)
        near_tangent = SectionPlane(
            "tiny-plane",
            (0.0, 0.0, 1.0e-6 - 5.0e-13),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        tiny_schedule = compute_parallel_plane_motion_schedule(
            tiny_sphere,
            ParallelPlaneTranslation(
                "tiny-tangential",
                near_tangent,
                (1.0e-6, 0.0, 0.0),
            ),
        )
        self.assertFalse(
            any(
                PlaneMotionCriticalKind.SPHERE_TANGENCY in item.kinds
                for item in tiny_schedule.critical_events
            )
        )

    def test_near_parallel_long_cylinder_is_not_snapped_to_parallel(self) -> None:
        alignment = 1.0e-15
        normal = (sqrt(1.0 - alignment * alignment), 0.0, alignment)
        cylinder = CylinderSpec(
            "long-near-parallel-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0e20, 1.0e20),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "long-near-parallel-plane",
            tuple(-2.0 * item for item in normal),
            normal,
            u_axis=(0.0, 1.0, 0.0),
        )
        schedule = compute_parallel_plane_motion_schedule(
            cylinder,
            ParallelPlaneTranslation(
                "long-near-parallel-motion",
                plane,
                tuple(4.0 * item for item in normal),
            ),
            include_interval_midpoints=False,
        )
        self.assertFalse(
            any(
                PlaneMotionCriticalKind.CYLINDER_AXIS_PARALLEL in item.kinds
                for item in schedule.critical_events
            )
        )
        self.assertFalse(
            any(
                PlaneMotionCriticalKind.CYLINDER_TRIM_TANGENCY in item.kinds
                for item in schedule.critical_events
            )
        )

    def test_near_parabolic_long_cone_is_not_snapped_to_parabola(self) -> None:
        beta = pi / 4.0
        alignment = sin(beta) + 1.0e-15
        normal = (sqrt(1.0 - alignment * alignment), 0.0, alignment)
        cone = ConeSpec(
            "long-near-parabolic-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            beta,
            (0.0, 1.0e20),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.CLOSED_SINGLE,
        )
        plane = SectionPlane(
            "long-near-parabolic-plane",
            tuple(-2.0 * item for item in normal),
            normal,
            u_axis=(0.0, 1.0, 0.0),
        )
        with self.assertRaisesRegex(
            ParallelPlaneMotionError,
            "parabolic orientation is numerically ambiguous",
        ):
            compute_parallel_plane_motion_schedule(
                cone,
                ParallelPlaneTranslation(
                    "long-near-parabolic-motion",
                    plane,
                    tuple(4.0 * item for item in normal),
                ),
                include_interval_midpoints=False,
            )

    def test_overlapping_roundoff_is_not_a_proof_of_equal_levels(self) -> None:
        alignment = 1.0e-12
        normal = (sqrt(1.0 - alignment * alignment), 0.0, alignment)
        cylinder = CylinderSpec(
            "short-near-parallel-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (0.0, 0.001),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "short-near-parallel-plane",
            tuple(-2.0 * item for item in normal),
            normal,
            u_axis=(0.0, 1.0, 0.0),
        )
        with self.assertRaisesRegex(
            ParallelPlaneMotionError,
            "equality is uncertified",
        ):
            compute_parallel_plane_motion_schedule(
                cylinder,
                ParallelPlaneTranslation(
                    "short-near-parallel-motion",
                    plane,
                    tuple(4.0 * item for item in normal),
                ),
                include_interval_midpoints=False,
            )

    def test_rounded_zero_dot_product_is_not_structural_parallelism(self) -> None:
        cylinder = CylinderSpec(
            "rounded-dot-cylinder",
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            1.0,
            (-1.0e20, 1.0e20),
            radial_axis=(1.0, -1.0, 0.0),
        )
        raw_normal = (
            1.0,
            float(np.nextafter(1.0, -np.inf)),
            -2.0,
        )
        plane = SectionPlane(
            "rounded-dot-plane",
            tuple(-2.0 * item for item in raw_normal),
            raw_normal,
            u_axis=(1.0, -1.0, 0.0),
        )
        with self.assertRaisesRegex(
            ParallelPlaneMotionError,
            "axis-parallel orientation is numerically ambiguous",
        ):
            compute_parallel_plane_motion_schedule(
                cylinder,
                ParallelPlaneTranslation(
                    "rounded-dot-motion",
                    plane,
                    tuple(4.0 * item for item in plane.normal),
                ),
                include_interval_midpoints=False,
            )


class SectionTimelineCompilationTests(unittest.TestCase):
    def test_sphere_crossing_tracks_lineage_once_and_certifies_topology(self) -> None:
        sphere = SphereSpec("timeline-sphere", (0.0, 0.0, 0.0), 1.0)
        motion = ParallelPlaneTranslation(
            "sphere-crossing",
            _plane((0.0, 0.0, -2.0)),
            (0.0, 0.0, 4.0),
            start_time=0.0,
            end_time=2.0,
        )
        with patch.object(
            timeline_module,
            "track_quadric_section_animation",
            wraps=timeline_module.track_quadric_section_animation,
        ) as tracker:
            timeline = compile_section_timeline(
                "sphere-section",
                sphere,
                (motion,),
            )
        tracker.assert_called_once()
        self.assertEqual(len(timeline.animation.topology_events), 4)
        self.assertEqual(len(timeline.topology_certifications), 4)
        self.assertEqual(
            timeline.topology_frame_banks,
            (0, 0, 1, 0, 1, 0, 0),
        )
        for index, (left, right) in enumerate(
            zip(timeline.animation.frames, timeline.animation.frames[1:])
        ):
            equivalent = left.signature.topologically_equivalent(right.signature)
            self.assertEqual(
                equivalent,
                timeline.topology_frame_banks[index]
                == timeline.topology_frame_banks[index + 1],
            )
        certified = {
            critical_id
            for item in timeline.topology_certifications
            for critical_id in item.critical_event_ids
        }
        self.assertEqual(
            certified,
            {item.event_id for item in timeline.critical_events},
        )
        self.assertEqual(
            tuple(frame.signature.conic_family for frame in timeline.animation.frames),
            (
                SectionConicFamily.EMPTY,
                SectionConicFamily.EMPTY,
                SectionConicFamily.POINT,
                SectionConicFamily.OVAL,
                SectionConicFamily.POINT,
                SectionConicFamily.EMPTY,
                SectionConicFamily.EMPTY,
            ),
        )

    def test_axis_angle_segment_matches_existing_schedule_and_tracking(self) -> None:
        cone = _closed_cone()
        motion = AxisAnglePlaneMotion(
            "cone-rotation",
            _plane((0.0, 0.0, 0.2)),
            (0.0, 0.0, 0.2),
            (0.0, 1.0, 0.0),
            0.0,
            1.2,
            start_time=2.0,
            end_time=8.0,
        )
        expected_schedule = compute_plane_motion_schedule(cone, motion)
        expected_animation = track_scheduled_plane_section(
            "cone-section",
            cone,
            motion,
        ).animation
        timeline = compile_section_timeline(
            "cone-section",
            cone,
            (motion,),
        )
        self.assertEqual(
            timeline.segment_schedules[0].to_dict(),
            expected_schedule.to_dict(),
        )
        self.assertEqual(timeline.animation.to_dict(), expected_animation.to_dict())
        families = tuple(
            frame.signature.conic_family for frame in timeline.animation.frames
        )
        self.assertIn(SectionConicFamily.OVAL, families)
        self.assertIn(SectionConicFamily.PARABOLA, families)
        self.assertIn(SectionConicFamily.HYPERBOLA, families)
        self.assertTrue(timeline.topology_certifications)

    def test_rotation_and_translation_join_once_with_global_lineage(self) -> None:
        sphere = SphereSpec("joined-sphere", (0.0, 0.0, 0.0), 2.0)
        start = _plane((0.0, 0.0, 0.2))
        rotation = AxisAnglePlaneMotion(
            "joined-rotation",
            start,
            start.point,
            (0.0, 1.0, 0.0),
            0.0,
            0.4,
            start_time=0.0,
            end_time=1.0,
        )
        rotated = rotation.plane_at(1.0)
        translation = ParallelPlaneTranslation(
            "joined-translation",
            rotated,
            tuple(0.3 * item for item in rotated.normal),
            start_time=1.0,
            end_time=2.0,
        )
        timeline = compile_section_timeline(
            "joined-section",
            sphere,
            (rotation, translation),
        )
        times = tuple(item.time for item in timeline.samples)
        self.assertEqual(times.count(1.0), 1)
        self.assertTrue(all(right > left for left, right in zip(times, times[1:])))
        stable_ids = {
            branch.stable_branch_id
            for frame in timeline.animation.frames
            for branch in frame.branches
        }
        self.assertEqual(stable_ids, {"joined-section:epoch:0000:branch:00"})
        self.assertEqual(timeline.animation.capacity_plan.required_slots, 1)

    def test_cylinder_cap_chord_events_have_analytic_evidence(self) -> None:
        cylinder = CylinderSpec(
            "cap-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cap-plane",
            (0.0, 0.0, -3.0),
            (1.0, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        timeline = compile_section_timeline(
            "cap-section",
            cylinder,
            (
                ParallelPlaneTranslation(
                    "cap-translation",
                    plane,
                    (0.0, 0.0, 6.0),
                    start_time=0.0,
                    end_time=6.0,
                ),
            ),
        )
        self.assertEqual(
            timeline.cap_chord_ids,
            (
                "cap-section:cap:cap_max:chord",
                "cap-section:cap:cap_min:chord",
            ),
        )
        self.assertEqual(len(timeline.cap_chord_events), 4)
        self.assertTrue(
            all(item.critical_event_ids for item in timeline.cap_chord_events)
        )
        critical_by_id = {
            item.event_id: item for item in timeline.critical_events
        }
        for event in timeline.cap_chord_events:
            for critical_id in event.critical_event_ids:
                self.assertIn(
                    "cylinder_trim_tangency",
                    critical_by_id[critical_id].kinds,
                )
        activated = {
            curve_id
            for event in timeline.cap_chord_events
            for curve_id in event.activated_curve_ids
        }
        deactivated = {
            curve_id
            for event in timeline.cap_chord_events
            for curve_id in event.deactivated_curve_ids
        }
        self.assertEqual(activated, set(timeline.cap_chord_ids))
        self.assertEqual(deactivated, set(timeline.cap_chord_ids))

    def test_json_is_deterministic_strict_and_contains_complete_evidence(self) -> None:
        sphere = SphereSpec("json-sphere", (0.0, 0.0, 0.0), 1.0)
        motion = ParallelPlaneTranslation(
            "json-motion",
            _plane((0.0, 0.0, -2.0)),
            (0.0, 0.0, 4.0),
        )
        first = canonical_section_timeline_json(
            compile_section_timeline("json-section", sphere, (motion,))
        )
        second = canonical_section_timeline_json(
            compile_section_timeline("json-section", sphere, (motion,))
        )
        self.assertEqual(first, second)
        self.assertNotIn("NaN", first)
        self.assertNotIn("Infinity", first)
        payload = json.loads(first)
        self.assertEqual(payload["schema"], SECTION_TIMELINE_SCHEMA)
        self.assertTrue(payload["criticalEvents"])
        self.assertTrue(payload["topologyCertifications"])
        self.assertEqual(
            payload["topologyFrameBanks"],
            list(
                compile_section_timeline(
                    "json-section",
                    sphere,
                    (motion,),
                ).topology_frame_banks
            ),
        )

    def test_missing_analytic_event_is_rejected_instead_of_sampled_through(
        self,
    ) -> None:
        sphere = SphereSpec("missing-event-sphere", (0.0, 0.0, 0.0), 1.0)
        motion = ParallelPlaneTranslation(
            "missing-event-motion",
            _plane((0.0, 0.0, -2.0)),
            (0.0, 0.0, 4.0),
        )
        with (
            patch.object(timeline_module, "_timeline_critical_events", return_value=()),
            self.assertRaisesRegex(
                SectionTimelineError,
                "not bracketed by analytic critical evidence",
            ),
        ):
            compile_section_timeline("missing-event", sphere, (motion,))

    def test_topology_event_rejects_incompatible_critical_kind(self) -> None:
        sphere = SphereSpec("cause-sphere", (0.0, 0.0, 0.0), 1.0)
        motion = ParallelPlaneTranslation(
            "cause-motion",
            _plane((0.0, 0.0, -2.0)),
            (0.0, 0.0, 4.0),
        )
        schedule = compute_parallel_plane_motion_schedule(sphere, motion)
        genuine = timeline_module._timeline_critical_events(
            "cause-section",
            (schedule,),
        )
        incompatible = tuple(
            replace(
                event,
                kinds=("cylinder_trim_tangency",),
                equations=("unrelated-event",),
            )
            for event in genuine
        )
        with (
            patch.object(
                timeline_module,
                "_timeline_critical_events",
                return_value=incompatible,
            ),
            self.assertRaisesRegex(
                SectionTimelineError,
                "compatible with its surface and topology reason",
            ),
        ):
            compile_section_timeline("cause-section", sphere, (motion,))

    def test_finite_topology_rejects_same_surface_orientation_event(self) -> None:
        cylinder = CylinderSpec(
            "cause-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cause-cylinder-plane",
            (0.0, 0.0, -3.0),
            (1.0, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        timeline = compile_section_timeline(
            "cause-cylinder-section",
            cylinder,
            (
                ParallelPlaneTranslation(
                    "cause-cylinder-motion",
                    plane,
                    (0.0, 0.0, 6.0),
                ),
            ),
        )
        incompatible = tuple(
            replace(
                event,
                kinds=("cylinder_axis_parallel",),
                equations=("unrelated-orientation-event",),
            )
            if "cylinder_trim_tangency" in event.kinds
            else event
            for event in timeline.critical_events
        )
        with self.assertRaisesRegex(
            SectionTimelineError,
            "compatible with its surface and topology reason",
        ):
            timeline_module._topology_certifications(
                timeline.animation,
                incompatible,
                cylinder,
            )

    def test_public_timeline_rejects_segment_time_gap(self) -> None:
        sphere = SphereSpec("gap-sphere", (0.0, 0.0, 0.0), 10.0)
        first_motion = ParallelPlaneTranslation(
            "gap-first",
            _plane((0.0, 0.0, 0.0)),
            (0.0, 0.0, 0.1),
            start_time=0.0,
            end_time=1.0,
        )
        second_motion = ParallelPlaneTranslation(
            "gap-second",
            first_motion.plane_at(1.0),
            (0.0, 0.0, 0.1),
            start_time=2.0,
            end_time=3.0,
        )
        schedules = tuple(
            compute_parallel_plane_motion_schedule(sphere, motion)
            for motion in (first_motion, second_motion)
        )
        samples = tuple(
            sample for schedule in schedules for sample in schedule.samples
        )
        animation = track_quadric_section_animation(
            "gap-section",
            samples,
        )
        with self.assertRaisesRegex(
            SectionTimelineError,
            "join exactly without gaps",
        ):
            SectionTimeline(
                section_id="gap-section",
                surface_id=sphere.surface_id,
                plane_id=first_motion.base_plane.plane_id,
                segment_schedules=schedules,
                samples=samples,
                critical_events=(),
                animation=animation,
                topology_certifications=(),
                topology_frame_banks=tuple(0 for _ in samples),
                cap_chord_ids=(),
                cap_chord_states=(),
                cap_chord_events=(),
            )

    def test_critical_event_ids_are_namespaced_by_section(self) -> None:
        sphere = SphereSpec("namespace-sphere", (0.0, 0.0, 0.0), 1.0)
        motion = ParallelPlaneTranslation(
            "namespace-motion",
            _plane((0.0, 0.0, -2.0)),
            (0.0, 0.0, 4.0),
        )
        first = compile_section_timeline("section-a", sphere, (motion,))
        second = compile_section_timeline("section-b", sphere, (motion,))
        first_ids = {item.event_id for item in first.critical_events}
        second_ids = {item.event_id for item in second.critical_events}
        self.assertTrue(first_ids)
        self.assertTrue(second_ids)
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_invalid_timeline_contracts_fail_before_compilation(self) -> None:
        sphere = SphereSpec("invalid-sphere", (0.0, 0.0, 0.0), 2.0)
        first = ParallelPlaneTranslation(
            "first",
            _plane((0.0, 0.0, 0.0)),
            (0.0, 0.0, 0.2),
            0.0,
            1.0,
        )
        correct_endpoint = first.plane_at(1.0)
        cases = (
            (
                (
                    first,
                    ParallelPlaneTranslation(
                        "gap",
                        correct_endpoint,
                        (0.0, 0.0, 0.1),
                        2.0,
                        3.0,
                    ),
                ),
                "join exactly",
            ),
            (
                (
                    first,
                    ParallelPlaneTranslation(
                        "wrong-plane",
                        _plane((0.0, 0.0, 0.3)),
                        (0.0, 0.0, 0.1),
                        1.0,
                        2.0,
                    ),
                ),
                "one exact endpoint",
            ),
            (
                (
                    first,
                    ParallelPlaneTranslation(
                        "first",
                        correct_endpoint,
                        (0.0, 0.0, 0.1),
                        1.0,
                        2.0,
                    ),
                ),
                "ids must be unique",
            ),
        )
        for motions, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SectionTimelineError,
                message,
            ):
                compile_section_timeline("invalid-section", sphere, motions)

        with self.assertRaisesRegex(SectionTimelineError, "interval midpoint"):
            compile_section_timeline(
                "invalid-section",
                sphere,
                (first,),
                include_interval_midpoints=False,
            )
        with self.assertRaisesRegex(SectionTimelineError, "unknown motion ids"):
            compile_section_timeline(
                "invalid-section",
                sphere,
                (first,),
                authored_progresses={"missing": (0.4,)},
            )
        with self.assertRaisesRegex(
            SectionTimelineError,
            "duplicate normalized motion ids",
        ):
            compile_section_timeline(
                "invalid-section",
                sphere,
                (first,),
                authored_progresses={"first": (0.2,), " first ": (0.3,)},
            )
        with self.assertRaisesRegex(
            SectionTimelineError,
            "finite and positive",
        ):
            compile_section_timeline(
                "invalid-section",
                sphere,
                (first,),
                coefficient_tolerance=0.0,
            )

    def test_public_timeline_rejects_inconsistent_cap_chord_evidence(self) -> None:
        cylinder = CylinderSpec(
            "tamper-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "tamper-plane",
            (0.0, 0.0, -3.0),
            (1.0, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        timeline = compile_section_timeline(
            "tamper-section",
            cylinder,
            (
                ParallelPlaneTranslation(
                    "tamper-motion",
                    plane,
                    (0.0, 0.0, 6.0),
                ),
            ),
        )
        original = timeline.cap_chord_events[0]
        wrong_curve_id = next(
            item
            for item in timeline.cap_chord_ids
            if item not in original.activated_curve_ids
        )
        inconsistent = replace(
            original,
            activated_curve_ids=(wrong_curve_id,),
        )
        with self.assertRaisesRegex(
            SectionTimelineError,
            "does not describe its state transition",
        ):
            replace(
                timeline,
                cap_chord_events=(
                    inconsistent,
                    *timeline.cap_chord_events[1:],
                ),
            )

    def test_public_timeline_rejects_tampered_topology_bank_plan(self) -> None:
        sphere = SphereSpec("bank-sphere", (0.0, 0.0, 0.0), 1.0)
        timeline = compile_section_timeline(
            "bank-section",
            sphere,
            (
                ParallelPlaneTranslation(
                    "bank-motion",
                    _plane((0.0, 0.0, -2.0)),
                    (0.0, 0.0, 4.0),
                ),
            ),
        )
        tampered = list(timeline.topology_frame_banks)
        tampered[1] = 1 - tampered[1]
        with self.assertRaisesRegex(
            SectionTimelineError,
            "change exactly at topology events",
        ):
            replace(timeline, topology_frame_banks=tuple(tampered))

    def test_open_double_and_analytic_double_are_outside_single_timeline(self) -> None:
        plane = _plane((0.0, 0.0, 0.2))
        motion = ParallelPlaneTranslation("double-motion", plane, (0.0, 0.0, 0.1))
        for model in (ConeModel.OPEN_DOUBLE, ConeModel.ANALYTIC_DOUBLE):
            cone = ConeSpec(
                f"double-{model.value}",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                pi / 4.0,
                (-1.0, 1.0),
                radial_axis=(1.0, 0.0, 0.0),
                model=model,
            )
            with self.subTest(model=model), self.assertRaisesRegex(
                SectionTimelineError,
                "one directly renderable cone nappe",
            ):
                compile_section_timeline("double-section", cone, (motion,))


class SectionTimelineImportBoundaryTests(unittest.TestCase):
    def test_renderer_neutral_modules_do_not_import_manim(self) -> None:
        script = """
import sys
assert 'manim' not in sys.modules
from polyhedron_visibility.quadrics.parallel_plane_motion import (
    ParallelPlaneTranslation,
)
from polyhedron_visibility.quadrics.section_timeline import compile_section_timeline
assert 'manim' not in sys.modules
print(ParallelPlaneTranslation.__name__, compile_section_timeline.__name__)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ParallelPlaneTranslation", result.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
