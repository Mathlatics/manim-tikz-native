from __future__ import annotations

import json
from math import atan, pi
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from polyhedron_visibility.quadrics.animation import SectionConicFamily
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    PlaneMotionCriticalKind,
    PlaneMotionError,
    canonical_plane_motion_schedule_json,
    compute_plane_motion_schedule,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.roots import PolynomialRootError


ROOT = Path(__file__).resolve().parents[1]


def _cone() -> ConeSpec:
    return ConeSpec(
        "cone",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (-20.0, 20.0),
        radial_axis=(1.0, 0.0, 0.0),
    )


def _cone_motion(*, reverse: bool = False) -> AxisAnglePlaneMotion:
    return AxisAnglePlaneMotion(
        "cone-motion",
        SectionPlane(
            "plane",
            (0.0, 0.0, 3.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        (0.0, 0.0, 3.0),
        (0.0, 1.0, 0.0),
        pi / 2.0 if reverse else 0.0,
        0.0 if reverse else pi / 2.0,
        start_time=2.0,
        end_time=6.0,
    )


class ConePlaneMotionTests(unittest.TestCase):
    def test_exact_parabolic_frame_is_inserted_between_ellipse_and_hyperbola(self) -> None:
        result = track_scheduled_plane_section(
            "cone-section", _cone(), _cone_motion()
        )
        event = next(
            item
            for item in result.schedule.critical_events
            if PlaneMotionCriticalKind.CONE_PARABOLIC in item.kinds
        )
        self.assertAlmostEqual(event.progress, 0.5)
        self.assertAlmostEqual(event.angle, pi / 4.0)
        self.assertAlmostEqual(event.time, 4.0)
        frame_by_time = {round(item.time, 12): item for item in result.animation.frames}
        self.assertIs(
            frame_by_time[4.0].signature.conic_family,
            SectionConicFamily.PARABOLA,
        )
        families = tuple(item.signature.conic_family for item in result.animation.frames)
        self.assertIn(SectionConicFamily.OVAL, families)
        self.assertIn(SectionConicFamily.HYPERBOLA, families)

    def test_reverse_motion_finds_the_same_world_angle(self) -> None:
        forward = compute_plane_motion_schedule(_cone(), _cone_motion())
        reverse = compute_plane_motion_schedule(_cone(), _cone_motion(reverse=True))
        forward_parabola = next(
            item
            for item in forward.critical_events
            if PlaneMotionCriticalKind.CONE_PARABOLIC in item.kinds
        )
        reverse_parabola = next(
            item
            for item in reverse.critical_events
            if PlaneMotionCriticalKind.CONE_PARABOLIC in item.kinds
        )
        self.assertAlmostEqual(forward_parabola.angle, reverse_parabola.angle)
        self.assertAlmostEqual(forward_parabola.progress, 0.5)
        self.assertAlmostEqual(reverse_parabola.progress, 0.5)

    def test_cone_apex_degeneracy_is_a_separate_event(self) -> None:
        schedule = compute_plane_motion_schedule(_cone(), _cone_motion())
        apex = next(
            item
            for item in schedule.critical_events
            if PlaneMotionCriticalKind.CONE_APEX_DEGENERACY in item.kinds
        )
        self.assertAlmostEqual(apex.progress, 1.0)
        self.assertFalse(apex.persistent)


class SphereAndCylinderPlaneMotionTests(unittest.TestCase):
    def test_sphere_tangent_is_solved_without_frame_sampling(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 0.25)
        motion = AxisAnglePlaneMotion(
            "sphere-motion",
            SectionPlane("plane", (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.5),
            (0.0, 1.0, 0.0),
            0.0,
            pi / 2.0,
        )
        schedule = compute_plane_motion_schedule(
            sphere,
            motion,
            authored_progresses=(0.123456789,),
            include_interval_midpoints=False,
        )
        tangencies = [
            item
            for item in schedule.critical_events
            if PlaneMotionCriticalKind.SPHERE_TANGENCY in item.kinds
        ]
        self.assertEqual(len(tangencies), 1)
        self.assertAlmostEqual(tangencies[0].angle, pi / 3.0)
        self.assertAlmostEqual(tangencies[0].progress, 2.0 / 3.0)
        self.assertIn(0.123456789, schedule.progresses)

    def test_cylinder_parallel_position_is_inserted_and_classified(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-5.0, 5.0),
        )
        motion = AxisAnglePlaneMotion(
            "cylinder-motion",
            SectionPlane("plane", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            0.0,
            pi,
        )
        result = track_scheduled_plane_section("cylinder-section", cylinder, motion)
        parallel = next(
            item
            for item in result.schedule.critical_events
            if PlaneMotionCriticalKind.CYLINDER_AXIS_PARALLEL in item.kinds
        )
        self.assertAlmostEqual(parallel.progress, 0.5)
        critical_frame = min(
            result.animation.frames, key=lambda item: abs(item.time - 0.5)
        )
        self.assertIn(
            critical_frame.signature.conic_family,
            {SectionConicFamily.PARALLEL_LINES, SectionConicFamily.COINCIDENT_LINE},
        )

    def test_persistent_parallel_relation_is_explicit(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-5.0, 5.0),
        )
        motion = AxisAnglePlaneMotion(
            "persistent",
            SectionPlane("plane", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            0.0,
            pi / 2.0,
        )
        schedule = compute_plane_motion_schedule(cylinder, motion)
        self.assertEqual(len(schedule.critical_events), 1)
        self.assertTrue(schedule.critical_events[0].persistent)
        self.assertEqual(schedule.critical_events[0].progress, 0.0)

    def test_finite_cylinder_trim_entry_and_exit_are_exact_schedule_events(
        self,
    ) -> None:
        cylinder = CylinderSpec(
            "finite-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
        )
        motion = AxisAnglePlaneMotion(
            "trim-motion",
            SectionPlane("plane", (0.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 3.0),
            (0.0, 1.0, 0.0),
            0.0,
            pi / 2.0,
        )
        schedule = compute_plane_motion_schedule(
            cylinder,
            motion,
            include_interval_midpoints=False,
        )
        trim_events = [
            item
            for item in schedule.critical_events
            if PlaneMotionCriticalKind.CYLINDER_TRIM_TANGENCY in item.kinds
        ]

        expected = (
            atan(2.0) / (pi / 2.0),
            atan(4.0) / (pi / 2.0),
        )
        self.assertEqual(len(trim_events), 2)
        for event, progress in zip(trim_events, expected):
            self.assertAlmostEqual(event.progress, progress, places=12)
            self.assertIn(event.progress, schedule.progresses)
            self.assertFalse(event.persistent)

    def test_reverse_finite_cylinder_motion_reverses_trim_progresses(self) -> None:
        cylinder = CylinderSpec(
            "finite-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
        )
        base_plane = SectionPlane(
            "plane", (0.0, 0.0, 3.0), (0.0, 0.0, 1.0)
        )
        forward = compute_plane_motion_schedule(
            cylinder,
            AxisAnglePlaneMotion(
                "forward",
                base_plane,
                (0.0, 0.0, 3.0),
                (0.0, 1.0, 0.0),
                0.0,
                pi / 2.0,
            ),
            include_interval_midpoints=False,
        )
        reverse = compute_plane_motion_schedule(
            cylinder,
            AxisAnglePlaneMotion(
                "reverse",
                base_plane,
                (0.0, 0.0, 3.0),
                (0.0, 1.0, 0.0),
                pi / 2.0,
                0.0,
            ),
            include_interval_midpoints=False,
        )
        forward_progresses = tuple(
            item.progress
            for item in forward.critical_events
            if PlaneMotionCriticalKind.CYLINDER_TRIM_TANGENCY in item.kinds
        )
        reverse_progresses = tuple(
            item.progress
            for item in reverse.critical_events
            if PlaneMotionCriticalKind.CYLINDER_TRIM_TANGENCY in item.kinds
        )
        self.assertEqual(len(forward_progresses), len(reverse_progresses))
        for first, second in zip(forward_progresses, reversed(reverse_progresses)):
            self.assertAlmostEqual(first, 1.0 - second, places=12)

    def test_trim_tangencies_are_scale_and_translation_covariant(self) -> None:
        def trim_progresses(scale: float, shift: tuple[float, float, float]):
            cylinder = CylinderSpec(
                f"cylinder-{scale:g}",
                shift,
                (0.0, 0.0, 1.0),
                scale,
                (-scale, scale),
            )
            plane_point = (
                shift[0],
                shift[1],
                shift[2] + 3.0 * scale,
            )
            schedule = compute_plane_motion_schedule(
                cylinder,
                AxisAnglePlaneMotion(
                    f"motion-{scale:g}",
                    SectionPlane("plane", plane_point, (0.0, 0.0, 1.0)),
                    plane_point,
                    (0.0, 1.0, 0.0),
                    0.0,
                    pi / 2.0,
                ),
                include_interval_midpoints=False,
            )
            return tuple(
                item.progress
                for item in schedule.critical_events
                if PlaneMotionCriticalKind.CYLINDER_TRIM_TANGENCY in item.kinds
            )

        expected = trim_progresses(1.0, (0.0, 0.0, 0.0))
        for scale, shift, places in (
            (1.0e-6, (0.0, 0.0, 0.0), 11),
            (1.0e6, (0.0, 0.0, 0.0), 11),
            (1.0e6, (1.0e12, -2.0e12, 3.0e12), 8),
        ):
            with self.subTest(scale=scale, shift=shift):
                actual = trim_progresses(scale, shift)
                self.assertEqual(len(actual), len(expected))
                for first, second in zip(actual, expected):
                    self.assertAlmostEqual(first, second, places=places)


class ConeTrimPlaneMotionTests(unittest.TestCase):
    def test_frustum_trim_circles_are_scheduled_analytically(self) -> None:
        frustum = ConeSpec(
            "frustum",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (1.0, 3.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        motion = AxisAnglePlaneMotion(
            "frustum-motion",
            SectionPlane("plane", (0.0, 0.0, 5.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 5.0),
            (0.0, 1.0, 0.0),
            0.0,
            pi / 2.0,
        )
        schedule = compute_plane_motion_schedule(
            frustum,
            motion,
            include_interval_midpoints=False,
        )
        trim_events = [
            item
            for item in schedule.critical_events
            if PlaneMotionCriticalKind.CONE_TRIM_TANGENCY in item.kinds
        ]
        expected = (
            atan(2.0 / 3.0) / (pi / 2.0),
            atan(4.0) / (pi / 2.0),
        )
        self.assertEqual(len(trim_events), 2)
        for event, progress in zip(trim_events, expected):
            self.assertAlmostEqual(event.progress, progress, places=12)
            self.assertIn(event.progress, schedule.progresses)


class PlaneMotionContractTests(unittest.TestCase):
    def test_schedule_is_deterministic_strict_json(self) -> None:
        first = compute_plane_motion_schedule(_cone(), _cone_motion())
        second = compute_plane_motion_schedule(_cone(), _cone_motion())
        payload = canonical_plane_motion_schedule_json(first)
        self.assertEqual(payload, canonical_plane_motion_schedule_json(second))
        self.assertEqual(json.loads(payload), first.to_dict())

    def test_invalid_intervals_fail_closed(self) -> None:
        plane = SectionPlane("plane", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        with self.assertRaisesRegex(PlaneMotionError, "non-zero"):
            AxisAnglePlaneMotion("bad", plane, (0, 0, 0), (0, 1, 0), 0.0, 0.0)
        with self.assertRaisesRegex(PlaneMotionError, "one revolution"):
            AxisAnglePlaneMotion(
                "bad", plane, (0, 0, 0), (0, 1, 0), 0.0, 2.1 * pi
            )
        with self.assertRaisesRegex(PlaneMotionError, r"\[0, 1\]"):
            compute_plane_motion_schedule(
                _cone(), _cone_motion(), authored_progresses=(-0.1,)
            )

    def test_uncertifiable_trim_polynomial_fails_closed(self) -> None:
        cylinder = CylinderSpec(
            "finite-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 1.0),
        )
        motion = AxisAnglePlaneMotion(
            "trim-motion",
            SectionPlane("plane", (0.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 3.0),
            (0.0, 1.0, 0.0),
            0.0,
            pi / 2.0,
        )
        with patch(
            "polyhedron_visibility.quadrics.plane_motion.solve_real_polynomial",
            side_effect=PolynomialRootError("synthetic uncertifiable roots"),
        ):
            with self.assertRaisesRegex(PlaneMotionError, "roots are ambiguous"):
                compute_plane_motion_schedule(cylinder, motion)

    def test_import_does_not_load_manim(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import polyhedron_visibility.quadrics.plane_motion; "
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


if __name__ == "__main__":
    unittest.main()
