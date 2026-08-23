from __future__ import annotations

import json
from math import cos, pi, sin, tau
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

from polyhedron_visibility.quadrics.animation import (
    BranchCapacityPlan,
    BranchContinuityError,
    MovingPointContinuityError,
    PointParameterMode,
    PointTrackSelection,
    SectionAnimationError,
    SectionAnimationSample,
    SectionConicFamily,
    SectionTopologySignature,
    TopologyEventKind,
    canonical_quadric_section_animation_json,
    track_moving_section_point,
    track_quadric_section_animation,
)
from polyhedron_visibility.quadrics.conics import ConicKind
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.trace import FiniteSectionTopology


ROOT = Path(__file__).resolve().parents[1]


def _sphere_samples(
    heights: tuple[float, ...],
) -> tuple[SectionAnimationSample, ...]:
    sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 2.0)
    return tuple(
        SectionAnimationSample(
            index,
            sphere,
            SectionPlane(
                "section-plane",
                (0.0, 0.0, height),
                (0.0, 0.0, 1.0),
                u_axis=(1.0, 0.0, 0.0),
            ),
        )
        for index, height in enumerate(heights)
    )


def _cone() -> ConeSpec:
    return ConeSpec(
        "cone",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (-20.0, 20.0),
        radial_axis=(1.0, 0.0, 0.0),
    )


def _cone_sample(time: float, slope: float) -> SectionAnimationSample:
    return SectionAnimationSample(
        time,
        _cone(),
        SectionPlane(
            "section-plane",
            (0.0, 0.0, 3.0),
            (-slope, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        ),
    )


class AnimationContractTests(unittest.TestCase):
    def test_signature_and_animation_are_strictly_serializable(self) -> None:
        animation = track_quadric_section_animation(
            "sphere-section", _sphere_samples((0.0, 0.25, 0.5))
        )
        signature = animation.frames[0].signature
        self.assertIs(signature.supporting_kind, ConicKind.CIRCLE)
        self.assertIs(signature.conic_family, SectionConicFamily.OVAL)
        self.assertIs(signature.finite_topology, FiniteSectionTopology.CLOSED_CURVE)
        self.assertEqual(signature.component_closedness, (True,))
        self.assertFalse(signature.degenerate)
        self.assertEqual(animation.capacity_plan.maximum_slots, 2)
        self.assertEqual(animation.capacity_plan.required_slots, 1)
        payload = canonical_quadric_section_animation_json(animation)
        self.assertEqual(payload, canonical_quadric_section_animation_json(animation))
        parsed = json.loads(payload)
        self.assertEqual(parsed["capacityPlan"]["maximumSlots"], 2)
        self.assertEqual(len(parsed["frames"]), 3)

    def test_invalid_capacity_and_animation_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(SectionAnimationError, "exactly two"):
            BranchCapacityPlan(("a", "b"), 1, maximum_slots=3)
        with self.assertRaisesRegex(SectionAnimationError, "increase strictly"):
            track_quadric_section_animation(
                "bad-time",
                (
                    SectionAnimationSample(1.0, _cone(), _cone_sample(0.0, 0.5).plane),
                    SectionAnimationSample(1.0, _cone(), _cone_sample(0.0, 0.6).plane),
                ),
            )
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        with self.assertRaisesRegex(SectionAnimationError, "plane_id"):
            track_quadric_section_animation(
                "bad-plane",
                (
                    SectionAnimationSample(
                        0.0, sphere, SectionPlane("a", (0, 0, 0), (0, 0, 1))
                    ),
                    SectionAnimationSample(
                        1.0, sphere, SectionPlane("b", (0, 0, 0), (0, 0, 1))
                    ),
                ),
            )

    def test_one_exact_section_solve_is_used_per_authored_frame(self) -> None:
        import polyhedron_visibility.quadrics.animation as animation_module

        samples = _sphere_samples((0.0, 0.2, 0.4, 0.6))
        original = animation_module.compute_quadric_section
        with mock.patch.object(
            animation_module,
            "compute_quadric_section",
            wraps=original,
        ) as solve:
            track_quadric_section_animation("counted", samples)
        self.assertEqual(solve.call_count, len(samples))

    def test_import_does_not_load_manim(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import polyhedron_visibility.quadrics.animation; "
                    "assert not any(name == 'manim' or name.startswith('manim.') "
                    "for name in sys.modules)"
                ),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class SameTopologyContinuityTests(unittest.TestCase):
    def test_stable_identity_and_first_frame_have_no_parameter_jump(self) -> None:
        animation = track_quadric_section_animation(
            "sphere-section", _sphere_samples((0.0, 0.2, 0.4))
        )
        stable_ids = tuple(frame.branches[0].stable_branch_id for frame in animation.frames)
        self.assertEqual(len(set(stable_ids)), 1)
        self.assertFalse(animation.topology_events)

        first = animation.frames[0]
        tracked = first.branches[0]
        branch = first.section.branch_map[tracked.source_branch_id]
        natural = branch.parameterization.natural_domain
        self.assertIsNotNone(natural)
        self.assertEqual(tracked.orientation, 1)
        self.assertAlmostEqual(tracked.phase_offset, natural.start)
        np.testing.assert_allclose(
            first.world_point(tracked.stable_branch_id, 0.0),
            branch.world_point(natural.start),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_repeated_frame_is_bitwise_deterministic(self) -> None:
        samples = _sphere_samples((0.25, 0.25, 0.25))
        first = track_quadric_section_animation("repeat", samples)
        second = track_quadric_section_animation("repeat", samples)
        self.assertEqual(
            canonical_quadric_section_animation_json(first),
            canonical_quadric_section_animation_json(second),
        )
        points = [
            frame.world_point(frame.branches[0].stable_branch_id, 0.37)
            for frame in first.frames
        ]
        for point in points[1:]:
            np.testing.assert_allclose(point, points[0], rtol=0.0, atol=1.0e-12)

    def test_coordinate_frame_flip_preserves_circle_phase_and_direction(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 2.0)
        planes = (
            SectionPlane(
                "plane", (0, 0, 0), (0, 0, 1), u_axis=(1, 0, 0)
            ),
            SectionPlane(
                "plane", (0, 0, 0), (0, 0, 1), u_axis=(-1, 0, 0)
            ),
        )
        animation = track_quadric_section_animation(
            "gauge",
            tuple(
                SectionAnimationSample(index, sphere, plane)
                for index, plane in enumerate(planes)
            ),
        )
        for fraction in (0.0, 0.125, 0.25, 0.75):
            points = [
                frame.world_point(frame.branches[0].stable_branch_id, fraction)
                for frame in animation.frames
            ]
            np.testing.assert_allclose(points[1], points[0], rtol=0.0, atol=1.0e-11)

    def test_closed_phase_is_unwrapped_across_the_periodic_seam(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 2.0)
        samples = tuple(
            SectionAnimationSample(
                index,
                sphere,
                SectionPlane(
                    "plane",
                    (0, 0, 0),
                    (0, 0, 1),
                    u_axis=(cos(angle), sin(angle), 0.0),
                ),
            )
            for index, angle in enumerate((0.0, 0.1, 0.2))
        )
        animation = track_quadric_section_animation("phase", samples)
        phases = tuple(frame.branches[0].phase_offset for frame in animation.frames)
        np.testing.assert_allclose(phases, (0.0, -0.1, -0.2), rtol=0.0, atol=1.0e-12)
        for frame in animation.frames:
            point = frame.world_point(frame.branches[0].stable_branch_id, 0.0)
            np.testing.assert_allclose(point, (2.0, 0.0, 0.0), rtol=0.0, atol=1.0e-11)

    def test_hyperbola_branch_swap_and_reversal_are_tracked(self) -> None:
        cone = _cone()
        planes = (
            SectionPlane(
                "plane", (0, 0, 3), (-1.5, 0, 1), u_axis=(0, 1, 0)
            ),
            SectionPlane(
                "plane", (0, 0, 3), (-1.5, 0, 1), u_axis=(0, -1, 0)
            ),
        )
        animation = track_quadric_section_animation(
            "hyperbola",
            tuple(
                SectionAnimationSample(index, cone, plane)
                for index, plane in enumerate(planes)
            ),
        )
        self.assertEqual(animation.capacity_plan.required_slots, 2)
        self.assertFalse(animation.topology_events)
        first, second = animation.frames
        self.assertEqual(
            tuple(item.stable_branch_id for item in first.branches),
            tuple(item.stable_branch_id for item in second.branches),
        )
        self.assertEqual(tuple(item.orientation for item in second.branches), (-1, -1))
        self.assertNotEqual(
            tuple(item.source_component_id for item in first.branches),
            tuple(item.source_component_id for item in second.branches),
        )
        for slot in (0, 1):
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                np.testing.assert_allclose(
                    first.world_point(first.slot_map[slot].stable_branch_id, fraction),
                    second.world_point(second.slot_map[slot].stable_branch_id, fraction),
                    rtol=0.0,
                    atol=2.0e-10,
                )

    def test_rigid_translation_and_rotation_remain_continuous(self) -> None:
        first_surface = SphereSpec("sphere", (1.0, 0.0, 0.0), 2.0)
        first_plane = SectionPlane(
            "plane", (1.0, 0.0, 0.25), (0.0, 0.0, 1.0), u_axis=(1, 0, 0)
        )
        angle = 0.35
        rotation = np.asarray(
            (
                (np.cos(angle), 0.0, np.sin(angle)),
                (0.0, 1.0, 0.0),
                (-np.sin(angle), 0.0, np.cos(angle)),
            )
        )
        translation = np.asarray((0.4, -0.3, 0.2))

        def moved(point):
            return tuple(rotation @ np.asarray(point, dtype=float) + translation)

        second_surface = SphereSpec("sphere", moved(first_surface.center), 2.0)
        second_plane = SectionPlane(
            "plane",
            moved(first_plane.point),
            tuple(rotation @ np.asarray(first_plane.normal)),
            u_axis=tuple(rotation @ np.asarray(first_plane.u_axis)),
        )
        animation = track_quadric_section_animation(
            "rigid",
            (
                SectionAnimationSample(0.0, first_surface, first_plane),
                SectionAnimationSample(1.0, second_surface, second_plane),
            ),
        )
        first, second = animation.frames
        for fraction in (0.0, 0.2, 0.5, 0.85):
            expected = moved(first.world_point(first.branches[0].stable_branch_id, fraction))
            actual = second.world_point(second.branches[0].stable_branch_id, fraction)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-10)

    def test_circle_to_ellipse_is_one_oval_epoch(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            2.0,
            (-10.0, 10.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        samples = (
            SectionAnimationSample(
                0.0,
                cylinder,
                SectionPlane("plane", (0, 0, 0), (0, 0, 1), u_axis=(1, 0, 0)),
            ),
            SectionAnimationSample(
                1.0,
                cylinder,
                SectionPlane("plane", (0, 0, 0), (0.2, 0, 1), u_axis=(0, 1, 0)),
            ),
        )
        animation = track_quadric_section_animation("oval", samples)
        self.assertEqual(
            tuple(frame.signature.supporting_kind for frame in animation.frames),
            (ConicKind.CIRCLE, ConicKind.ELLIPSE),
        )
        self.assertFalse(animation.topology_events)
        self.assertEqual(animation.frames[0].topology_epoch, animation.frames[1].topology_epoch)

        first_frame, second_frame = animation.frames
        first_tracked = first_frame.branches[0]
        second_tracked = second_frame.branches[0]
        first_branch = first_frame.section.branch_map[first_tracked.source_branch_id]
        second_branch = second_frame.section.branch_map[second_tracked.source_branch_id]

        def center(branch):
            embedding = np.asarray(branch.plane_embedding, dtype=float)
            return (
                embedding[:3, :2]
                @ np.asarray(branch.parameterization.origin, dtype=float)
                + embedding[:3, 2]
            )

        first_normal = np.cross(
            np.asarray(first_frame.section.plane_embedding)[:3, 0],
            np.asarray(first_frame.section.plane_embedding)[:3, 1],
        )
        second_normal = np.cross(
            np.asarray(second_frame.section.plane_embedding)[:3, 0],
            np.asarray(second_frame.section.plane_embedding)[:3, 1],
        )
        first_normal /= np.linalg.norm(first_normal)
        second_normal /= np.linalg.norm(second_normal)
        axis = np.cross(first_normal, second_normal)
        sine = np.linalg.norm(axis)
        cosine = float(np.dot(first_normal, second_normal))
        axis /= sine
        skew = np.asarray(
            (
                (0.0, -axis[2], axis[1]),
                (axis[2], 0.0, -axis[0]),
                (-axis[1], axis[0], 0.0),
            )
        )
        rotation = np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
        first_radial = (
            first_frame.world_point(first_tracked.stable_branch_id, 0.0)
            - center(first_branch)
        )
        actual_radial = (
            second_frame.world_point(second_tracked.stable_branch_id, 0.0)
            - center(second_branch)
        )
        expected_direction = rotation @ first_radial
        expected_direction /= np.linalg.norm(expected_direction)
        actual_radial /= np.linalg.norm(actual_radial)
        np.testing.assert_allclose(
            actual_radial,
            expected_direction,
            rtol=0.0,
            atol=2.0e-10,
        )

    def test_seam_wrapped_finite_ellipse_is_traversed_without_a_jump(self) -> None:
        cone = ConeSpec(
            "cone",
            (0, 0, 0),
            (0, 0, 1),
            pi / 4,
            (0.1, 20.0),
            radial_axis=(1, 0, 0),
        )
        samples = tuple(
            SectionAnimationSample(
                index,
                cone,
                SectionPlane(
                    "plane", (0, 0, 3), (-slope, 0, 1), u_axis=(0, 1, 0)
                ),
            )
            for index, slope in enumerate((0.99, 0.991))
        )
        animation = track_quadric_section_animation("wrapped", samples)
        self.assertFalse(animation.topology_events)
        for frame in animation.frames:
            component = frame.section.components[0]
            self.assertEqual(len(component.parameter_intervals), 2)
            branch_id = frame.branches[0].stable_branch_id
            parameters = tuple(
                frame.source_parameter(branch_id, fraction)
                for fraction in (0.0, 0.5, 1.0)
            )
            self.assertGreater(parameters[0], tau - 1.0)
            self.assertTrue(
                abs(parameters[1]) < 1.0e-8 or abs(parameters[1] - tau) < 1.0e-8
            )
            self.assertLess(parameters[2], 1.0)


class TopologyEventTests(unittest.TestCase):
    def test_rotating_cone_plane_emits_ellipse_parabola_hyperbola_events(self) -> None:
        samples = tuple(
            _cone_sample(index, slope)
            for index, slope in enumerate((0.5, 1.0, 1.5))
        )
        animation = track_quadric_section_animation("transition", samples)
        self.assertEqual(
            tuple(frame.signature.supporting_kind for frame in animation.frames),
            (ConicKind.ELLIPSE, ConicKind.PARABOLA, ConicKind.HYPERBOLA),
        )
        self.assertEqual(tuple(frame.topology_epoch for frame in animation.frames), (0, 1, 2))
        self.assertEqual(len(animation.topology_events), 2)
        for event in animation.topology_events:
            self.assertIn(TopologyEventKind.CONIC_FAMILY_CHANGED, event.reasons)
        self.assertEqual(
            len(
                {
                    frame.branches[0].stable_branch_id
                    for frame in animation.frames
                    if frame.branches
                }
            ),
            3,
        )

    def test_reverse_authored_sampling_has_the_reverse_exact_families(self) -> None:
        forward = track_quadric_section_animation(
            "forward",
            tuple(
                _cone_sample(index, slope)
                for index, slope in enumerate((0.5, 1.0, 1.5))
            ),
        )
        reverse = track_quadric_section_animation(
            "reverse",
            tuple(
                _cone_sample(index, slope)
                for index, slope in enumerate((1.5, 1.0, 0.5))
            ),
        )
        self.assertEqual(
            tuple(frame.signature.supporting_kind for frame in reverse.frames),
            tuple(
                reversed(
                    tuple(frame.signature.supporting_kind for frame in forward.frames)
                )
            ),
        )
        self.assertEqual(len(forward.topology_events), 2)
        self.assertEqual(len(reverse.topology_events), 2)

    def test_opposite_plane_normals_fail_closed_inside_one_epoch(self) -> None:
        sphere = SphereSpec("sphere", (0, 0, 0), 2.0)
        samples = (
            SectionAnimationSample(
                0.0, sphere, SectionPlane("plane", (0, 0, 0), (0, 0, 1))
            ),
            SectionAnimationSample(
                1.0, sphere, SectionPlane("plane", (0, 0, 0), (0, 0, -1))
            ),
        )
        with self.assertRaisesRegex(BranchContinuityError, "opposite"):
            track_quadric_section_animation("ambiguous", samples)

    def test_degenerate_entry_and_exit_are_named(self) -> None:
        sphere = SphereSpec("sphere", (0, 0, 0), 2.0)
        samples = tuple(
            SectionAnimationSample(
                index,
                sphere,
                SectionPlane("plane", (0, 0, height), (0, 0, 1)),
            )
            for index, height in enumerate((1.5, 2.0, 1.5))
        )
        animation = track_quadric_section_animation("tangent", samples)
        self.assertEqual(
            tuple(frame.signature.supporting_kind for frame in animation.frames),
            (ConicKind.CIRCLE, ConicKind.POINT, ConicKind.CIRCLE),
        )
        self.assertIn(
            TopologyEventKind.ENTERED_DEGENERACY,
            animation.topology_events[0].reasons,
        )
        self.assertIn(
            TopologyEventKind.EXITED_DEGENERACY,
            animation.topology_events[1].reasons,
        )


class MovingPointTests(unittest.TestCase):
    def test_normalized_parameter_point_is_continuous_inside_an_epoch(self) -> None:
        animation = track_quadric_section_animation(
            "point", _sphere_samples((0.0, 0.1, 0.2, 0.3))
        )
        point = track_moving_section_point(
            animation,
            PointTrackSelection(0, 0.25),
        )
        self.assertEqual(len(point.samples), len(animation.frames))
        self.assertFalse(point.crossed_topology_events)
        for sample in point.samples:
            self.assertAlmostEqual(sample.world_point[0], 0.0, places=10)
            self.assertGreater(sample.world_point[1], 0.0)

    def test_arc_length_fraction_uses_analytic_tangents(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0, 0, 0),
            (0, 0, 1),
            2.0,
            (-20, 20),
            radial_axis=(1, 0, 0),
        )
        animation = track_quadric_section_animation(
            "ellipse-point",
            (
                SectionAnimationSample(
                    0.0,
                    cylinder,
                    SectionPlane(
                        "plane", (0, 0, 0), (0.8, 0, 1), u_axis=(0, 1, 0)
                    ),
                ),
            ),
        )
        frame = animation.frames[0]
        branch_id = frame.branches[0].stable_branch_id
        normalized = frame.source_parameter(branch_id, 0.125)
        arc_length = frame.source_parameter(
            branch_id,
            0.125,
            mode=PointParameterMode.ARC_LENGTH_FRACTION,
        )
        self.assertGreater(normalized, 0.0)
        self.assertLess(normalized, pi / 2.0)
        self.assertGreater(arc_length, 0.0)
        self.assertLess(arc_length, pi / 2.0)
        self.assertNotAlmostEqual(normalized, arc_length, places=5)
        endpoints = (
            frame.source_parameter(
                branch_id, value, mode=PointParameterMode.ARC_LENGTH_FRACTION
            )
            for value in (0.0, 1.0)
        )
        self.assertEqual(tuple(endpoints), (0.0, tau))

    def test_topology_crossing_requires_explicit_auxiliary_rule(self) -> None:
        animation = track_quadric_section_animation(
            "transition-point",
            tuple(
                _cone_sample(index, slope)
                for index, slope in enumerate((0.5, 1.0, 1.5))
            ),
        )
        selection = PointTrackSelection(0, 0.4)
        with self.assertRaisesRegex(MovingPointContinuityError, "auxiliary rule"):
            track_moving_section_point(animation, selection)

        calls = []

        def remap(context):
            calls.append(context.event.event_id)
            return PointTrackSelection(0, 0.4)

        point = track_moving_section_point(
            animation,
            selection,
            auxiliary_rule=remap,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(point.crossed_topology_events, tuple(calls))
        self.assertEqual(len(point.samples), 3)
        self.assertTrue(
            all(np.all(np.isfinite(sample.world_point)) for sample in point.samples)
        )

    def test_auxiliary_rule_cannot_select_an_empty_slot(self) -> None:
        sphere = SphereSpec("sphere", (0, 0, 0), 2.0)
        samples = tuple(
            SectionAnimationSample(
                index,
                sphere,
                SectionPlane("plane", (0, 0, height), (0, 0, 1)),
            )
            for index, height in enumerate((0.0, 2.0))
        )
        animation = track_quadric_section_animation("empty-slot", samples)
        with self.assertRaisesRegex(MovingPointContinuityError, "slot is empty"):
            track_moving_section_point(
                animation,
                PointTrackSelection(0, 0.5),
                auxiliary_rule=lambda context: PointTrackSelection(0, 0.5),
            )


if __name__ == "__main__":
    unittest.main()
