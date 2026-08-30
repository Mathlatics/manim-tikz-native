from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from math import acos, atan, cos, pi, sin, sqrt, tau
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import (
    AnimationGroup,
    Mobject,
    Scene,
    linear,
    tempconfig,
    there_and_back,
)

from polyhedron_visibility.painter_band import (
    ManagedPainterBand,
    ScenePainterBandError,
    scene_painter_band_allocations,
)
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.animation import SectionConicFamily
from polyhedron_visibility.quadrics.conics import ConicKind
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.manim import QuadricManimLimits
import polyhedron_visibility.quadrics.manim as quadric_manim_module
from polyhedron_visibility.quadrics.plane_motion import AxisAnglePlaneMotion
from polyhedron_visibility.quadrics.rig import (
    QuadricSectionRig,
    QuadricSectionRigError,
    SectionState,
)
from tikz_native.parallel_camera import ParallelCameraState


VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)


def _limits(**overrides: object) -> QuadricManimLimits:
    values: dict[str, object] = {
        "max_surfaces": 2,
        "max_curves": 8,
        "max_fragments_per_curve": 8,
        "max_segments_per_fragment": 64,
        "max_surface_segments": 128,
        "max_dashes_per_fragment": 32,
        "max_projected_length": 18.0,
        "max_total_mobjects": 10000,
        "max_boundary_sources": 16,
    }
    values.update(overrides)
    return QuadricManimLimits(**values)  # type: ignore[arg-type]


def _band_only_limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=1,
        max_curves=1,
        max_fragments_per_curve=1,
        max_segments_per_fragment=16,
        max_surface_segments=64,
        max_dashes_per_fragment=1,
        max_projected_length=18.0,
        max_total_mobjects=100,
        max_boundary_sources=1,
        max_boundary_styles=5,
    )


def _sphere_plane(
    height: float = 0.0,
    *,
    plane_id: str = "plane",
) -> SectionPlane:
    return SectionPlane(
        plane_id,
        (0.0, 0.0, height),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )


def _display_options() -> dict[str, object]:
    return {
        "projection": VIEW,
        "limits": _limits(),
        "max_chord_error": 0.08,
        "section_max_screen_error": 0.08,
        "include_surface_boundaries": False,
    }


def _band_only_options() -> dict[str, object]:
    return {
        "projection": VIEW,
        "limits": _band_only_limits(),
        "max_chord_error": 0.1,
        "section_max_screen_error": 0.1,
        "include_surface_boundaries": False,
        "draw_section_boundary": False,
        "show_plane": False,
    }


def _assert_plane_close(
    case: unittest.TestCase,
    actual: SectionPlane,
    expected: SectionPlane,
) -> None:
    case.assertEqual(actual.plane_id, expected.plane_id)
    np.testing.assert_allclose(actual.point, expected.point, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual.normal, expected.normal, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual.u_axis, expected.u_axis, rtol=0.0, atol=1.0e-12)


def _scene_ownership_snapshot(scene: Scene) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(getattr(scene, name, ()))
        for name in (
            "mobjects",
            "foreground_mobjects",
            "moving_mobjects",
            "static_mobjects",
        )
    )


class QuadricSectionRigTests(unittest.TestCase):
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

    def test_section_state_is_frozen_and_rejects_non_plane_values(self) -> None:
        plane = _sphere_plane()
        state = SectionState(plane=plane)

        self.assertIs(state.plane, plane)
        with self.assertRaises(FrozenInstanceError):
            state.plane = _sphere_plane(0.2)  # type: ignore[misc]
        with self.assertRaisesRegex(TypeError, "SectionPlane"):
            SectionState(plane=object())  # type: ignore[arg-type]

    def test_phase1_projection_is_static_and_initial_plane_has_display_rank(
        self,
    ) -> None:
        scene = Scene()
        sphere = SphereSpec("view-sphere", (0.0, 0.0, 0.0), 1.0)
        plane = _sphere_plane(plane_id="view-plane")
        before_scene = _scene_ownership_snapshot(scene)
        calls = 0

        def live_projection(_scene: object) -> ParallelView:
            nonlocal calls
            calls += 1
            return VIEW

        with self.assertRaisesRegex(
            QuadricSectionRigError,
            "static projection.*callable projection is unsupported",
        ):
            QuadricSectionRig(
                scene,
                surface=sphere,
                plane=plane,
                section_id="callable-view-section",
                projection=live_projection,
            )
        self.assertEqual(calls, 0)
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), ())

        mutable_matrix = np.eye(3).tolist()
        rig = QuadricSectionRig(
            scene,
            surface=sphere,
            plane=plane,
            section_id="static-view-section",
            projection=mutable_matrix,
        )
        mutable_matrix[0][0] = 0.0
        np.testing.assert_array_equal(rig.view.matrix, np.eye(3))

        semantic_projection = ParallelCameraState(
            np.eye(3),
            target=np.asarray((1.0, -0.5, 0.25)),
            screen_anchor=np.asarray((2.0, 3.0)),
            zoom=1.75,
        )
        semantic_options = _band_only_options()
        semantic_options["projection"] = semantic_projection
        semantic_rig = QuadricSectionRig(
            scene,
            surface=sphere,
            plane=plane,
            section_id="semantic-static-view-section",
            **semantic_options,
        )
        expected_matrix = np.eye(3)
        expected_matrix[:2] *= semantic_projection.zoom
        expected_offset = (
            semantic_projection.screen_anchor
            - expected_matrix[:2] @ semantic_projection.target
        )
        np.testing.assert_allclose(
            semantic_rig.view.matrix,
            expected_matrix,
            rtol=0.0,
            atol=0.0,
        )
        semantic_rig.attach()
        try:
            frozen_frame = semantic_rig.controller._resolve_projection_frame()
            np.testing.assert_allclose(
                frozen_frame.screen_offset,
                expected_offset,
                rtol=0.0,
                atol=0.0,
            )
            self.assertTrue(frozen_frame.viewport_relative)
        finally:
            semantic_rig.restore()
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), ())

        edge_on = SectionPlane(
            "edge-on-plane",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            u_axis=(0.0, 0.0, 1.0),
        )
        with self.assertRaisesRegex(
            QuadricSectionRigError,
            r"initial rig view.*edge-on.*progress=0",
        ):
            QuadricSectionRig(
                scene,
                surface=sphere,
                plane=edge_on,
                section_id="edge-on-view-section",
                projection=semantic_projection,
            )
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), ())

    def test_great_circle_rotation_rejects_interior_edge_on_before_playback(
        self,
    ) -> None:
        scene = Scene()
        sphere = SphereSpec("great-circle-sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "great-circle-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        rig = QuadricSectionRig(
            scene,
            surface=sphere,
            plane=plane,
            section_id="great-circle-section",
            projection=np.eye(3),
        )
        before_scene = _scene_ownership_snapshot(scene)
        before_bands = scene_painter_band_allocations(scene)

        with self.assertRaisesRegex(
            QuadricSectionRigError,
            r"edge-on.*display rank.*progress=0\.666666666667",
        ):
            rig.animate_plane_rotation(
                axis=(0.0, 1.0, 0.0),
                angle=3.0 * pi / 4.0,
                pivot=(0.0, 0.0, 0.0),
            )

        self.assertEqual(rig.state, SectionState(plane=plane))
        self.assertEqual(rig.frame_state, rig.state)
        self.assertFalse(rig.attached)
        self.assertIsNone(rig.painter_z_band)
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), before_bands)

        curve_only = QuadricSectionRig(
            scene,
            surface=sphere,
            plane=plane,
            section_id="great-circle-curve-only",
            projection=np.eye(3),
            show_plane=False,
        )
        with self.assertRaisesRegex(
            QuadricSectionRigError,
            r"edge-on.*display rank.*progress=0\.666666666667",
        ):
            curve_only.animate_plane_rotation(
                axis=(0.0, 1.0, 0.0),
                angle=3.0 * pi / 4.0,
                pivot=(0.0, 0.0, 0.0),
            )

        fully_hidden = QuadricSectionRig(
            scene,
            surface=sphere,
            plane=plane,
            section_id="great-circle-fully-hidden",
            projection=np.eye(3),
            show_plane=False,
            draw_section_boundary=False,
        )
        action = fully_hidden.animate_plane_rotation(
            axis=(0.0, 1.0, 0.0),
            angle=3.0 * pi / 4.0,
            pivot=(0.0, 0.0, 0.0),
        )
        self.assertEqual(action.target_state.plane, action._compiled.plane_at(1.0))
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), before_bands)

    def test_rotation_rank_proof_keeps_sub_tolerance_harmonic_extrema(
        self,
    ) -> None:
        scene = Scene()
        sphere = SphereSpec("micro-swing-sphere", (0.0, 0.0, 0.0), 1.0)
        center_alignment = 1.5e-12
        transverse_alignment = sqrt(1.0 - center_alignment**2)
        axis = np.asarray(
            (transverse_alignment, 0.0, center_alignment),
            dtype=float,
        )
        perpendicular = np.asarray(
            (-center_alignment, 0.0, transverse_alignment),
            dtype=float,
        )
        swing = 8.0e-13
        normal = cos(swing) * axis + sin(swing) * perpendicular
        plane = SectionPlane(
            "micro-swing-plane",
            (0.0, 0.0, 0.0),
            tuple(float(item) for item in normal),
            u_axis=(0.0, 1.0, 0.0),
        )
        motion = AxisAnglePlaneMotion(
            "micro-swing-motion",
            plane,
            (0.0, 0.0, 0.0),
            tuple(float(item) for item in axis),
            0.0,
            tau,
        )
        self.assertGreater(abs(plane.normal[2]), 1.0e-12)
        self.assertLess(abs(motion.plane_at(0.5).normal[2]), 1.0e-12)

        rig = QuadricSectionRig(
            scene,
            surface=sphere,
            plane=plane,
            section_id="micro-swing-section",
            projection=np.eye(3),
        )
        before_scene = _scene_ownership_snapshot(scene)
        before_bands = scene_painter_band_allocations(scene)
        with self.assertRaisesRegex(
            QuadricSectionRigError,
            r"rank-deficient.*display rank.*progress=0\.5",
        ):
            rig.animate_plane_rotation(
                axis=axis,
                angle=tau,
                pivot=(0.0, 0.0, 0.0),
            )

        self.assertEqual(rig.state, SectionState(plane=plane))
        self.assertEqual(rig.frame_state, rig.state)
        self.assertFalse(rig.attached)
        self.assertIsNone(rig.painter_z_band)
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), before_bands)

    def test_rotation_rejects_interior_numeric_rank_loss_without_a_zero_root(
        self,
    ) -> None:
        scene = Scene()
        sphere = SphereSpec("near-edge-sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "near-edge-plane",
            (0.0, 0.0, 0.2),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        view = ParallelView.from_matrix(
            (
                (0.1, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        delta = 5.0e-12
        beta = 0.5 * acos(delta)
        axis = (sin(beta), 0.0, cos(beta))
        rig = QuadricSectionRig(
            scene,
            surface=sphere,
            plane=plane,
            section_id="near-edge-section",
            projection=view,
        )
        before_scene = _scene_ownership_snapshot(scene)
        before_bands = scene_painter_band_allocations(scene)

        with self.assertRaisesRegex(
            QuadricSectionRigError,
            r"rank-deficient.*display rank.*progress=0\.5",
        ):
            rig.animate_plane_rotation(
                axis=axis,
                angle=tau,
                pivot=(0.0, 0.0, 0.0),
            )

        self.assertEqual(rig.state, SectionState(plane=plane))
        self.assertEqual(rig.frame_state, rig.state)
        self.assertFalse(rig.attached)
        self.assertIsNone(rig.painter_z_band)
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), before_bands)

        fully_hidden = QuadricSectionRig(
            scene,
            surface=sphere,
            plane=plane,
            section_id="near-edge-fully-hidden",
            projection=view,
            show_plane=False,
            draw_section_boundary=False,
        )
        fully_hidden.animate_plane_rotation(
            axis=axis,
            angle=tau,
            pivot=(0.0, 0.0, 0.0),
        )
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), before_bands)

    def test_shift_rotation_and_parallel_plane_to_have_exact_targets(self) -> None:
        sphere = SphereSpec("math-sphere", (0.0, 0.0, 0.0), 2.0)
        start = _sphere_plane(0.0, plane_id="math-plane")

        shift_rig = QuadricSectionRig(
            Scene(),
            surface=sphere,
            plane=start,
            section_id="shift-math",
        )
        shifted = shift_rig.animate_plane_shift(0.35, direction=(0.0, 0.0, 2.0))
        _assert_plane_close(
            self,
            shifted.target_state.plane,
            SectionPlane(
                start.plane_id,
                (0.0, 0.0, 0.35),
                start.normal,
                u_axis=start.u_axis,
            ),
        )

        pivot = (0.1, -0.2, 0.05)
        axis = (0.0, 1.0, 0.0)
        angle = 0.3
        rotation_rig = QuadricSectionRig(
            Scene(),
            surface=sphere,
            plane=start,
            section_id="rotation-math",
        )
        rotated = rotation_rig.animate_plane_rotation(axis, angle, pivot)
        expected_motion = AxisAnglePlaneMotion(
            "expected-rotation",
            start,
            pivot,
            axis,
            0.0,
            angle,
        )
        _assert_plane_close(
            self,
            rotated.target_state.plane,
            expected_motion.plane_at(1.0),
        )

        target = SectionPlane(
            start.plane_id,
            (0.2, -0.1, 0.25),
            start.normal,
            u_axis=start.u_axis,
        )
        to_scene = Scene()
        to_rig = QuadricSectionRig(
            to_scene,
            surface=sphere,
            plane=start,
            section_id="plane-to-math",
        )
        moved_to = to_rig.animate_plane_to(target)
        self.assertEqual(moved_to.target_state, SectionState(plane=target))

        unsupported_targets = (
            (
                SectionPlane(
                    start.plane_id,
                    start.point,
                    (0.0, 0.2, 1.0),
                    u_axis=(1.0, 0.0, 0.0),
                ),
                "normal-changing",
            ),
            (
                SectionPlane(
                    start.plane_id,
                    start.point,
                    start.normal,
                    u_axis=(-1.0, 0.0, 0.0),
                ),
                "in-plane axis",
            ),
            (
                SectionPlane(
                    "other-plane",
                    start.point,
                    start.normal,
                    u_axis=start.u_axis,
                ),
                "preserve plane_id",
            ),
        )
        before_mobjects = tuple(to_scene.mobjects)
        for unsupported, message in unsupported_targets:
            with self.subTest(message=message), self.assertRaisesRegex(
                QuadricSectionRigError,
                message,
            ):
                to_rig.animate_plane_to(unsupported)
            self.assertEqual(tuple(to_scene.mobjects), before_mobjects)
            self.assertEqual(to_rig.state, SectionState(plane=start))

    def test_topology_crossing_fails_before_animation_or_scene_mutation(self) -> None:
        scene = Scene()
        initial = _sphere_plane(0.0, plane_id="unsafe-plane")
        rig = QuadricSectionRig(
            scene,
            surface=SphereSpec("unsafe-sphere", (0.0, 0.0, 0.0), 1.0),
            plane=initial,
            section_id="unsafe-section",
        )
        before_mobjects = tuple(scene.mobjects)

        with self.assertRaisesRegex(
            QuadricSectionRigError,
            "crosses a section topology.*scheduled transition",
        ):
            rig.animate_plane_shift(2.0)

        self.assertEqual(tuple(scene.mobjects), before_mobjects)
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertEqual(rig.state, SectionState(plane=initial))
        self.assertFalse(rig.attached)

        capped_cylinder = CylinderSpec(
            "cap-change-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-2.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        oblique = SectionPlane(
            "cap-change-plane",
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        cap_rig = QuadricSectionRig(
            scene,
            surface=capped_cylinder,
            plane=oblique,
            section_id="cap-change-section",
        )
        with self.assertRaisesRegex(
            QuadricSectionRigError,
            "changes cap-chord activation.*scheduled transition",
        ):
            cap_rig.animate_plane_shift(1.6)
        self.assertEqual(tuple(scene.mobjects), before_mobjects)
        self.assertEqual(scene_painter_band_allocations(scene), ())

    def test_circle_ellipse_knot_commits_with_stable_slots_and_zero_allocation(
        self,
    ) -> None:
        theta = 0.2
        cylinder = CylinderSpec(
            "oval-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-3.0, 3.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        start = SectionPlane(
            "oval-plane",
            (0.0, 0.0, 0.0),
            (-sin(theta), 0.0, cos(theta)),
            u_axis=(0.0, 1.0, 0.0),
        )
        scene = Scene()
        rig = QuadricSectionRig(
            scene,
            surface=cylinder,
            plane=start,
            section_id="oval-section",
            **_display_options(),
        ).attach()
        try:
            identities = rig.slot_identities()
            allocated = rig.allocated_curve_ids
            painter = scene_painter_band_allocations(scene)
            initial_boundary = rig.last_boundary_frame
            action = rig.animate_plane_rotation(
                (0.0, 1.0, 0.0),
                2.0 * theta,
                (0.0, 0.0, 0.0),
                rate_func=linear,
            )
            self.assertTrue(
                any(
                    abs(value - 0.5) <= 1.0e-12
                    for value in action._compiled.certificate.certified_progresses
                )
            )
            tracked_frames = action._compiled.tracking.frames
            self.assertIs(
                tracked_frames[0].signature.supporting_kind,
                ConicKind.ELLIPSE,
            )
            self.assertIs(
                tracked_frames[-1].signature.supporting_kind,
                ConicKind.ELLIPSE,
            )
            exact_knot = tuple(
                frame
                for frame in tracked_frames
                if abs(frame.time - 0.5) <= 1.0e-12
            )
            self.assertEqual(len(exact_knot), 1)
            self.assertIs(
                exact_knot[0].signature.supporting_kind,
                ConicKind.CIRCLE,
            )
            self.assertTrue(
                all(
                    frame.signature.conic_family is SectionConicFamily.OVAL
                    for frame in tracked_frames
                )
            )
            self.assertFalse(action._compiled.tracking.topology_events)

            action.begin()
            with patch.object(
                Mobject,
                "__init__",
                side_effect=AssertionError("rig updater allocated a Mobject"),
            ):
                for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
                    action.interpolate_mobject(progress)
                    rig.update()
                    self.assertEqual(rig.slot_identities(), identities)
                    self.assertEqual(rig.allocated_curve_ids, allocated)
                    self.assertEqual(scene_painter_band_allocations(scene), painter)
                    self.assertEqual(rig.state, rig.frame_state)
                    frame = rig.last_boundary_frame
                    assert frame is not None
                    active_section_ids = {
                        item.source_id
                        for item in frame.sources
                        if item.source_id.startswith("oval-section:rig:slot:")
                    }
                    self.assertEqual(
                        active_section_ids,
                        {"oval-section:rig:slot:0:interval:0"},
                    )
            action.finish()

            self.assertEqual(rig.state, action.target_state)
            self.assertIsNot(rig.last_boundary_frame, initial_boundary)
            self.assertEqual(rig.slot_identities(), identities)
        finally:
            rig.restore()

    def test_seam_wrapped_ellipse_and_hyperbola_keep_tracked_identity(self) -> None:
        cone = ConeSpec(
            "tracking-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.1, 20.0),
            radial_axis=(1.0, 0.0, 0.0),
        )

        seam_start = SectionPlane(
            "seam-plane",
            (0.0, 0.0, 3.0),
            (-0.99, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        seam_action = QuadricSectionRig(
            Scene(),
            surface=cone,
            plane=seam_start,
            section_id="seam-section",
        ).animate_plane_rotation(
            (0.0, 1.0, 0.0),
            atan(0.99) - atan(0.991),
            seam_start.point,
        )
        seam_frames = seam_action._compiled.tracking.frames
        self.assertTrue(
            all(frame.signature.conic_family is SectionConicFamily.OVAL for frame in seam_frames)
        )
        self.assertTrue(
            all(
                len(frame.section.components) == 1
                and len(frame.section.components[0].parameter_intervals) == 2
                for frame in seam_frames
            )
        )
        self.assertEqual(
            len(
                {
                    tuple(branch.stable_branch_id for branch in frame.branches)
                    for frame in seam_frames
                }
            ),
            1,
        )

        hyper_start = SectionPlane(
            "hyper-plane",
            (0.0, 0.0, 3.0),
            (-1.5, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        hyper_action = QuadricSectionRig(
            Scene(),
            surface=cone,
            plane=hyper_start,
            section_id="hyper-section",
        ).animate_plane_rotation(
            hyper_start.normal,
            pi,
            hyper_start.point,
        )
        hyper_frames = hyper_action._compiled.tracking.frames
        self.assertTrue(
            all(
                frame.signature.conic_family is SectionConicFamily.HYPERBOLA
                and frame.signature.branch_count == 2
                and frame.signature.component_count == 1
                for frame in hyper_frames
            )
        )
        self.assertEqual(
            len(
                {
                    tuple(branch.stable_branch_id for branch in frame.branches)
                    for frame in hyper_frames
                }
            ),
            1,
        )

    def test_late_frame_fault_rolls_back_author_display_and_painter_state(self) -> None:
        scene = Scene()
        initial = _sphere_plane(0.0, plane_id="rollback-plane")
        rig = QuadricSectionRig(
            scene,
            surface=SphereSpec("rollback-sphere", (0.0, 0.0, 0.0), 1.0),
            plane=initial,
            section_id="rollback-section",
            **_display_options(),
        ).attach()
        try:
            state = rig.state
            snapshot = rig.slot_snapshot()
            identities = rig.slot_identities()
            boundary = rig.last_boundary_frame
            z_indices = rig.active_painter_z_indices
            allocations = scene_painter_band_allocations(scene)
            action = rig.animate_plane_shift(0.2, rate_func=linear)
            action.begin()
            action.interpolate_mobject(1.0)

            original_signature = quadric_manim_module._painter_band_signature
            original_apply = ManagedPainterBand.apply

            def changed_signature(prepared):
                return (
                    *original_signature(prepared),
                    ("__forced_test_change__", -1, -1.0),
                )

            def fail_after_painter_apply(band, prepared) -> None:
                original_apply(band, prepared)
                raise RuntimeError("injected painter commit fault")

            with (
                patch.object(
                    quadric_manim_module,
                    "_painter_band_signature",
                    side_effect=changed_signature,
                ),
                patch.object(
                    ManagedPainterBand,
                    "apply",
                    new=fail_after_painter_apply,
                ),
                self.assertRaisesRegex(RuntimeError, "injected painter commit fault"),
            ):
                rig.update()

            self.assertEqual(rig.state, state)
            self.assertEqual(rig.frame_state, state)
            self.assertEqual(rig.slot_snapshot(), snapshot)
            self.assertEqual(rig.slot_identities(), identities)
            self.assertIs(rig.last_boundary_frame, boundary)
            self.assertEqual(rig.active_painter_z_indices, z_indices)
            self.assertEqual(scene_painter_band_allocations(scene), allocations)
            self.assertTrue(rig.attached)
        finally:
            rig.restore()

    def test_author_commit_fault_rolls_back_display_cache_and_author_together(
        self,
    ) -> None:
        scene = Scene()
        initial = _sphere_plane(0.0, plane_id="author-fault-plane")
        rig = QuadricSectionRig(
            scene,
            surface=SphereSpec("author-fault-sphere", (0.0, 0.0, 0.0), 1.0),
            plane=initial,
            section_id="author-fault-section",
            **_display_options(),
        ).attach()
        try:
            controller = rig.controller
            self.assertIsNone(rig._frame_token)
            self.assertIsNone(rig._committed_frame_token)
            state = rig.state
            frame_state = rig.frame_state
            idle_reference = rig._idle_reference
            staged_reference = rig._staged_reference
            scene_state = _scene_ownership_snapshot(scene)
            painter_band = rig.painter_z_band
            allocations = scene_painter_band_allocations(scene)
            snapshot = rig.slot_snapshot()
            identities = rig.slot_identities()
            z_indices = rig.active_painter_z_indices
            last_frame = rig.last_frame
            last_global_frame = rig.last_global_frame
            last_section_frame = rig.last_section_frame
            last_boundary_frame = rig.last_boundary_frame
            fragment_slot_maps = {
                source_id: dict(slots)
                for source_id, slots in controller._fragment_slot_maps.items()
            }
            display_slot_state = dict(controller._display_slot_state)
            painter_signature = controller._last_painter_band_signature
            geometry_signature = controller._last_input_geometry_signature
            draw_signature = controller._last_input_draw_signature
            opacity = controller._last_input_opacity
            prepared_frame = controller._last_prepared_frame
            performance_counts = dict(
                controller._last_prepared_performance_counts
            )

            action = rig.animate_plane_shift(0.2, rate_func=linear)
            action.begin()
            action.interpolate_mobject(1.0)
            original_author_commit = rig._commit_quadric_frame

            def fail_author_commit(token: object) -> None:
                self.assertNotEqual(rig.slot_snapshot(), snapshot)
                self.assertIsNot(controller._last_prepared_frame, prepared_frame)
                self.assertNotEqual(
                    controller._last_input_geometry_signature,
                    geometry_signature,
                )
                original_author_commit(token)
                self.assertNotEqual(rig.state, state)
                self.assertIs(rig._committed_frame_token, token)
                raise RuntimeError("injected author commit fault")

            with (
                patch.object(
                    rig,
                    "_commit_quadric_frame",
                    side_effect=fail_author_commit,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "injected author commit fault",
                ),
            ):
                rig.update()

            self.assertEqual(rig.state, state)
            self.assertEqual(rig.frame_state, frame_state)
            self.assertIs(rig._idle_reference, idle_reference)
            self.assertIs(rig._staged_reference, staged_reference)
            self.assertIsNone(rig._active_action)
            self.assertEqual(rig._active_progress, 0.0)
            self.assertIsNone(rig._resolved_frame_state)
            self.assertIsNone(rig._resolved_action)
            self.assertIsNone(rig._frame_token)
            self.assertIsNone(rig._committed_frame_token)
            self.assertEqual(rig.slot_snapshot(), snapshot)
            self.assertEqual(rig.slot_identities(), identities)
            self.assertEqual(rig.active_painter_z_indices, z_indices)
            self.assertIs(rig.last_frame, last_frame)
            self.assertIs(rig.last_global_frame, last_global_frame)
            self.assertIs(rig.last_section_frame, last_section_frame)
            self.assertIs(rig.last_boundary_frame, last_boundary_frame)
            self.assertEqual(controller._fragment_slot_maps, fragment_slot_maps)
            self.assertEqual(controller._display_slot_state, display_slot_state)
            self.assertEqual(
                controller._last_painter_band_signature,
                painter_signature,
            )
            self.assertEqual(
                controller._last_input_geometry_signature,
                geometry_signature,
            )
            self.assertEqual(controller._last_input_draw_signature, draw_signature)
            self.assertEqual(controller._last_input_opacity, opacity)
            self.assertIs(controller._last_prepared_frame, prepared_frame)
            self.assertEqual(
                controller._last_prepared_performance_counts,
                performance_counts,
            )
            self.assertEqual(_scene_ownership_snapshot(scene), scene_state)
            self.assertEqual(scene_painter_band_allocations(scene), allocations)
            self.assertEqual(rig.painter_z_band, painter_band)
            self.assertTrue(rig.attached)

            finalize_action = rig.animate_plane_shift(0.2, rate_func=linear)
            finalize_action.begin()
            finalize_action.interpolate_mobject(1.0)
            with (
                patch.object(
                    rig,
                    "_finalize_quadric_frame",
                    side_effect=RuntimeError("injected author finalize fault"),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "injected author finalize fault",
                ),
            ):
                rig.update()

            self.assertEqual(rig.state, state)
            self.assertEqual(rig.frame_state, frame_state)
            self.assertIsNone(rig._frame_token)
            self.assertIsNone(rig._committed_frame_token)
            self.assertEqual(rig.slot_snapshot(), snapshot)
            self.assertEqual(rig.active_painter_z_indices, z_indices)
            self.assertIs(rig.last_section_frame, last_section_frame)
            self.assertEqual(
                controller._last_input_geometry_signature,
                geometry_signature,
            )
            self.assertIs(controller._last_prepared_frame, prepared_frame)
            self.assertEqual(_scene_ownership_snapshot(scene), scene_state)
            self.assertEqual(scene_painter_band_allocations(scene), allocations)
            self.assertTrue(rig.attached)
        finally:
            rig.restore()

    def test_rate_failure_and_restore_failure_still_clear_action_and_band(self) -> None:
        scene = Scene()
        initial = _sphere_plane(0.0, plane_id="cleanup-plane")
        rig = QuadricSectionRig(
            scene,
            surface=SphereSpec("cleanup-sphere", (0.0, 0.0, 0.0), 1.0),
            plane=initial,
            section_id="cleanup-section",
            **_band_only_options(),
        ).attach()

        invalid = rig.animate_plane_shift(
            0.1,
            rate_func=lambda _progress: float("nan"),
        )
        with self.assertRaisesRegex(
            QuadricSectionRigError,
            r"rate_func\(0\) must be finite",
        ):
            invalid.begin()
        self.assertEqual(rig.state, SectionState(plane=initial))
        self.assertEqual(rig.frame_state, rig.state)

        fresh = rig.animate_plane_shift(0.0, rate_func=linear)
        fresh.begin()
        fresh.finish()
        self.assertEqual(rig.state, fresh.target_state)

        facade = rig._facade
        assert facade is not None
        with (
            patch.object(
                facade,
                "restore",
                side_effect=RuntimeError("injected facade restore fault"),
            ),
            self.assertRaisesRegex(RuntimeError, "injected facade restore fault"),
        ):
            rig.restore()
        self.assertFalse(rig.attached)
        self.assertIsNone(rig.painter_z_band)
        self.assertEqual(rig.state, SectionState(plane=initial))
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertEqual(scene.mobjects, [])

    def test_scene_driver_never_leaks_and_rate_endpoints_fail_before_play(self) -> None:
        scene = Scene()
        initial = _sphere_plane(0.0, plane_id="driver-plane")
        rig = QuadricSectionRig(
            scene,
            surface=SphereSpec("driver-sphere", (0.0, 0.0, 0.0), 1.0),
            plane=initial,
            section_id="driver-section",
            **_band_only_options(),
        )
        unattached = rig.animate_plane_shift(0.1, rate_func=linear)
        scene.compile_animation_data(unattached)
        self.assertEqual(scene.mobjects, [unattached.mobject])
        with self.assertRaisesRegex(QuadricSectionRigError, "attach the rig"):
            scene.begin_animations()
        self.assertEqual(scene.mobjects, [])

        rig.attach()
        try:
            owned = tuple(scene.mobjects)
            invalid = rig.animate_plane_shift(0.1, rate_func=there_and_back)
            scene.compile_animation_data(invalid)
            self.assertEqual(tuple(scene.mobjects), (*owned, invalid.mobject))
            with self.assertRaisesRegex(
                QuadricSectionRigError,
                "rate_func must map animation endpoints to 0 and 1",
            ):
                scene.begin_animations()
            self.assertEqual(tuple(scene.mobjects), owned)
            self.assertEqual(rig.state, SectionState(plane=initial))

            valid = rig.animate_plane_shift(0.0, rate_func=linear)
            valid.begin()
            valid.finish()
            self.assertEqual(rig.state, valid.target_state)
        finally:
            rig.restore()

    def test_reversed_play_kwargs_are_rejected_but_bounded_backtracking_is_allowed(
        self,
    ) -> None:
        scene = Scene()
        initial = _sphere_plane(0.0, plane_id="rate-plane")
        rig = QuadricSectionRig(
            scene,
            surface=SphereSpec("rate-sphere", (0.0, 0.0, 0.0), 1.0),
            plane=initial,
            section_id="rate-section",
            **_band_only_options(),
        ).attach()
        try:
            owned = _scene_ownership_snapshot(scene)
            bands = scene_painter_band_allocations(scene)

            reversed_at_construction = rig.animate_plane_shift(
                0.2,
                rate_func=linear,
                reverse_rate_function=True,
            )
            with self.assertRaisesRegex(
                QuadricSectionRigError,
                "reverse_rate_function=True is unsupported",
            ):
                reversed_at_construction.begin()
            self.assertEqual(rig.state, SectionState(plane=initial))
            self.assertEqual(rig.frame_state, rig.state)
            self.assertEqual(_scene_ownership_snapshot(scene), owned)
            self.assertEqual(scene_painter_band_allocations(scene), bands)

            reversed_by_play = rig.animate_plane_shift(0.2, rate_func=linear)

            def compile_and_begin(
                current_scene: Scene,
                *animations: object,
                **play_kwargs: object,
            ) -> None:
                current_scene.compile_animation_data(
                    *animations,  # type: ignore[arg-type]
                    **play_kwargs,
                )
                current_scene.begin_animations()

            with (
                patch.object(
                    scene.renderer,
                    "play",
                    side_effect=compile_and_begin,
                ),
                self.assertRaisesRegex(
                    QuadricSectionRigError,
                    "reverse_rate_function=True is unsupported",
                ),
            ):
                scene.play(reversed_by_play, reverse_rate_function=True)
            self.assertTrue(reversed_by_play.reverse_rate_function)
            self.assertEqual(rig.state, SectionState(plane=initial))
            self.assertEqual(rig.frame_state, rig.state)
            self.assertEqual(_scene_ownership_snapshot(scene), owned)
            self.assertEqual(scene_painter_band_allocations(scene), bands)

            grouped_action = rig.animate_plane_shift(0.2, rate_func=linear)
            grouped = AnimationGroup(grouped_action)
            with (
                patch.object(
                    scene.renderer,
                    "play",
                    side_effect=compile_and_begin,
                ),
                self.assertRaisesRegex(
                    QuadricSectionRigError,
                    "reverse_rate_function=True is unsupported",
                ),
            ):
                scene.play(grouped, reverse_rate_function=True)
            self.assertTrue(grouped.reverse_rate_function)
            self.assertFalse(grouped_action.reverse_rate_function)
            self.assertEqual(rig.state, SectionState(plane=initial))
            self.assertEqual(rig.frame_state, rig.state)
            self.assertEqual(_scene_ownership_snapshot(scene), owned)
            self.assertEqual(scene_painter_band_allocations(scene), bands)

            def bounded_backtracking(alpha: float) -> float:
                if alpha <= 0.25:
                    return 2.4 * alpha
                if alpha <= 0.5:
                    return 0.7 - 0.4 * alpha
                return alpha

            backtracking = rig.animate_plane_shift(
                0.2,
                rate_func=bounded_backtracking,
            )
            backtracking.begin()
            backtracking.interpolate_mobject(0.25)
            rig.update()
            forward_height = rig.plane.point[2]
            backtracking.interpolate_mobject(0.5)
            rig.update()
            backward_height = rig.plane.point[2]
            self.assertGreater(forward_height, backward_height)
            backtracking.interpolate_mobject(1.0)
            rig.update()
            backtracking.finish()
            self.assertEqual(rig.state, backtracking.target_state)
            self.assertEqual(_scene_ownership_snapshot(scene), owned)
            self.assertEqual(scene_painter_band_allocations(scene), bands)
        finally:
            rig.restore()

    def test_exact_plane_to_endpoint_and_stale_play_recovery(self) -> None:
        scene = Scene()
        start = _sphere_plane(-0.4, plane_id="endpoint-plane")
        target = _sphere_plane(0.2, plane_id="endpoint-plane")
        rig = QuadricSectionRig(
            scene,
            surface=SphereSpec("endpoint-sphere", (0.0, 0.0, 0.0), 1.0),
            plane=start,
            section_id="endpoint-section",
            **_band_only_options(),
        ).attach()
        try:
            exact = rig.animate_plane_to(target, rate_func=linear)
            exact.begin()
            exact.interpolate_mobject(1.0)
            rig.update()
            exact.finish()
            self.assertEqual(rig.state, SectionState(plane=target))

            interrupted = rig.animate_plane_shift(0.05, rate_func=linear)
            interrupted.begin()
            interrupted.interpolate_mobject(0.5)
            rig.update()

            recovered = rig.animate_plane_shift(0.0, rate_func=linear)
            scene.animations = [recovered]
            recovered.begin()
            recovered.finish()
            self.assertEqual(rig.state, recovered.target_state)

            concurrent_a = rig.animate_plane_shift(0.0, rate_func=linear)
            concurrent_b = rig.animate_plane_shift(0.0, rate_func=linear)
            nested_a = AnimationGroup(concurrent_a)
            owned = tuple(scene.mobjects)
            scene.compile_animation_data(nested_a, concurrent_b)
            with self.assertRaisesRegex(
                QuadricSectionRigError,
                "only one mathematical action",
            ):
                scene.begin_animations()
            self.assertEqual(rig.frame_state, rig.state)
            self.assertEqual(tuple(scene.mobjects), owned)

            stale = rig.animate_plane_shift(0.0, rate_func=linear)
            stale.begin()
            nested_recovery = rig.animate_plane_shift(0.0, rate_func=linear)
            recovery_group = AnimationGroup(nested_recovery)
            scene.compile_animation_data(recovery_group)
            scene.begin_animations()
            recovery_group.finish()
            recovery_group.clean_up_from_scene(scene)
            self.assertEqual(rig.state, nested_recovery.target_state)
            self.assertEqual(tuple(scene.mobjects), owned)
        finally:
            rig.restore()

    def test_multiple_rigs_reject_conflicts_and_reuse_released_band(self) -> None:
        scene = Scene()
        first = QuadricSectionRig(
            scene,
            surface=SphereSpec("first-sphere", (-1.5, 0.0, 0.0), 0.75),
            plane=_sphere_plane(0.0, plane_id="first-plane"),
            section_id="first-section",
            **_band_only_options(),
        )
        second = QuadricSectionRig(
            scene,
            surface=SphereSpec("second-sphere", (1.5, 0.0, 0.0), 0.75),
            plane=_sphere_plane(0.0, plane_id="second-plane"),
            section_id="second-section",
            **_band_only_options(),
        )
        first.attach()
        second.attach()
        try:
            first_band = first.painter_z_band
            second_band = second.painter_z_band
            assert first_band is not None and second_band is not None
            self.assertTrue(
                first_band[1] < second_band[0]
                or second_band[1] < first_band[0]
            )
            self.assertEqual(len(scene_painter_band_allocations(scene)), 2)

            before_mobjects = tuple(scene.mobjects)
            duplicate = QuadricSectionRig(
                scene,
                surface=SphereSpec("duplicate-sphere", (0.0, 0.0, 0.0), 0.5),
                plane=_sphere_plane(0.0, plane_id="duplicate-plane"),
                section_id="first-section",
                **_band_only_options(),
            )
            with self.assertRaisesRegex(ScenePainterBandError, "duplicate.*owner"):
                duplicate.attach()
            self.assertEqual(tuple(scene.mobjects), before_mobjects)
            self.assertEqual(len(scene_painter_band_allocations(scene)), 2)

            explicit = QuadricSectionRig(
                scene,
                surface=SphereSpec("explicit-sphere", (0.0, 0.0, 0.0), 0.5),
                plane=_sphere_plane(0.0, plane_id="explicit-plane"),
                section_id="explicit-section",
                painter_z_band=first_band,
                **_band_only_options(),
            )
            with self.assertRaisesRegex(ScenePainterBandError, "exact.*conflicts"):
                explicit.attach()
            self.assertEqual(tuple(scene.mobjects), before_mobjects)
            self.assertEqual(len(scene_painter_band_allocations(scene)), 2)

            first.restore()
            self.assertFalse(first.attached)
            self.assertIsNone(first.painter_z_band)
            self.assertEqual(len(scene_painter_band_allocations(scene)), 1)
            first.attach()
            self.assertEqual(first.painter_z_band, first_band)
            self.assertEqual(len(scene_painter_band_allocations(scene)), 2)
        finally:
            first.restore()
            second.restore()
        self.assertEqual(scene_painter_band_allocations(scene), ())

    def test_real_cairo_scene_play_writes_mp4_and_commits_target(self) -> None:
        class RigScene(Scene):
            def construct(inner_self) -> None:
                rig = QuadricSectionRig(
                    inner_self,
                    surface=SphereSpec("movie-sphere", (0.0, 0.0, 0.0), 1.0),
                    plane=_sphere_plane(0.0, plane_id="movie-plane"),
                    section_id="movie-section",
                    **_display_options(),
                ).attach()
                identities = rig.slot_identities()
                action = rig.animate_plane_shift(
                    0.2,
                    run_time=0.5,
                    rate_func=linear,
                )
                inner_self.play(action)
                inner_self.target_committed = rig.state == action.target_state
                inner_self.identity_stable = identities == rig.slot_identities()
                rig.restore()
                inner_self.scene_clean = not inner_self.mobjects

        with (
            TemporaryDirectory() as media_dir,
            tempconfig(
                {
                    "renderer": "cairo",
                    "media_dir": media_dir,
                    "pixel_width": 160,
                    "pixel_height": 90,
                    "frame_rate": 4,
                    "disable_caching": True,
                    "write_to_movie": True,
                    "save_last_frame": False,
                }
            ),
        ):
            scene = RigScene()
            scene.render()
            movie = Path(scene.renderer.file_writer.movie_file_path)
            self.assertTrue(movie.is_file())
            self.assertEqual(movie.suffix, ".mp4")
            self.assertGreater(movie.stat().st_size, 0)
            self.assertTrue(scene.target_committed)
            self.assertTrue(scene.identity_stable)
            self.assertTrue(scene.scene_clean)

            probe = subprocess.run(
                (
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    (
                        "stream=codec_name,width,height,nb_read_frames,duration:"
                        "format=duration"
                    ),
                    "-of",
                    "json",
                    str(movie),
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            metadata = json.loads(probe.stdout)
            self.assertEqual(len(metadata["streams"]), 1)
            stream = metadata["streams"][0]
            self.assertEqual((stream["width"], stream["height"]), (160, 90))
            self.assertEqual(stream["codec_name"], "h264")
            self.assertGreaterEqual(int(stream["nb_read_frames"]), 2)
            duration = float(
                stream.get("duration", metadata["format"]["duration"])
            )
            self.assertGreater(duration, 0.0)

            decode = subprocess.run(
                (
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(movie),
                    "-f",
                    "null",
                    "-",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(decode.returncode, 0, decode.stderr)


if __name__ == "__main__":
    unittest.main()
