from __future__ import annotations

from math import tau
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import (
    FadeIn,
    FadeOut,
    Line,
    Mobject,
    Scene,
    ValueTracker,
    config,
    tempconfig,
)

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.compositing import (
    QuadricPaintKind,
    QuadricPaintPolicy,
)
from polyhedron_visibility.quadrics.contract import SphereSpec
from polyhedron_visibility.quadrics.curves import CircleArcCurve, SegmentCurve
from polyhedron_visibility.quadrics.global_occlusion import (
    GlobalQuadricOcclusionError,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimCapacityError,
    QuadricManimError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
    _adaptive_project_curve,
)
from polyhedron_visibility.quadrics.projection import (
    ProjectionProxyError,
    ProjectionSubdivisionError,
)

IDENTITY_VIEW = ParallelView.from_matrix(np.eye(3))


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    delta = end - start
    squared = float(np.dot(delta, delta))
    if squared == 0.0:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, delta) / squared)
    ratio = min(1.0, max(0.0, ratio))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def _dense_projected_curve_error(
    curve: CircleArcCurve,
    points: np.ndarray,
    *,
    samples: int = 2401,
) -> float:
    screen_points = points[:, :2]
    return max(
        min(
            _point_segment_distance(projected, start, end)
            for start, end in zip(screen_points, screen_points[1:])
        )
        for parameter in np.linspace(0.0, tau, samples)
        for projected in (np.asarray(curve.point(float(parameter)))[:2],)
    )


def _limits(**overrides: object) -> QuadricManimLimits:
    values: dict[str, object] = {
        "max_surfaces": 2,
        "max_curves": 2,
        "max_fragments_per_curve": 8,
        "max_segments_per_fragment": 128,
        "max_surface_segments": 256,
        "max_dashes_per_fragment": 48,
        "max_projected_length": 8.0,
        "max_total_mobjects": 2000,
    }
    values.update(overrides)
    return QuadricManimLimits(**values)  # type: ignore[arg-type]


class _QuadricFixture:
    def __init__(
        self,
        scene: Scene,
        *,
        paint_policy: str = "diagrammatic",
        limits: QuadricManimLimits | None = None,
    ) -> None:
        self.scene = scene
        self.offset = 0.0

        def surfaces() -> tuple[SphereSpec, ...]:
            return (SphereSpec("sphere", (self.offset, 0.0, 0.0), 1.0),)

        def curves() -> tuple[CircleArcCurve, ...]:
            return (
                CircleArcCurve(
                    "great-circle",
                    (self.offset, 0.0, 0.0),
                    1.0,
                    (1.0, 0.0, 0.0),
                    radial_axis=(0.0, 1.0, 0.0),
                ),
            )

        self.controller = QuadricOcclusion3D(
            scene,
            surfaces=surfaces,
            curves=curves,
            projection=IDENTITY_VIEW,
            paint_policy=paint_policy,
            style=QuadricManimStyle(
                surface_fill_color="#315f91",
                surface_fill_opacity=1.0,
                surface_stroke_color="#17324f",
                surface_stroke_width=1.0,
                visible_curve_color="#f4c542",
                visible_curve_width=4.0,
                hidden_curve_color="#f4c542",
                hidden_curve_width=3.0,
                dash_length=0.12,
                dash_gap=0.08,
            ),
            limits=limits or _limits(),
            max_chord_error=0.015,
            painter_z_band=(20.0, 30.0),
        )


class QuadricManimBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "frame_rate": 8,
                "pixel_width": 320,
                "pixel_height": 180,
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_default_projection_is_true_isometric_with_vertical_world_z(self) -> None:
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=(),
            limits=_limits(max_curves=1),
        )
        matrix = controller._resolve_view().matrix

        np.testing.assert_allclose(matrix @ matrix.T, np.identity(3), atol=1e-12)
        projected_axes = matrix[:2]
        lengths = np.linalg.norm(projected_axes, axis=0)
        np.testing.assert_allclose(lengths, np.full(3, lengths[0]), atol=1e-12)
        self.assertAlmostEqual(float(matrix[0, 2]), 0.0, places=12)
        self.assertGreater(float(matrix[1, 2]), 0.0)

    def test_curve_display_subdivision_is_stable_after_large_translation(
        self,
    ) -> None:
        tolerance = 1.0e-3
        shift = np.asarray((1.0e12, -1.0e12, 0.0))
        base_curve = CircleArcCurve(
            "base",
            (0.0, 0.0, 0.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        moved_curve = CircleArcCurve(
            "moved",
            tuple(shift),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        base = _adaptive_project_curve(
            base_curve,
            IDENTITY_VIEW,
            0.0,
            tau,
            max_chord_error=tolerance,
            max_segments=4096,
        )
        moved = _adaptive_project_curve(
            moved_curve,
            IDENTITY_VIEW,
            0.0,
            tau,
            max_chord_error=tolerance,
            max_segments=4096,
        )

        self.assertEqual(len(base), len(moved))
        screen_ulp = max(abs(float(np.spacing(value))) for value in shift[:2])
        np.testing.assert_allclose(
            moved[:, :2] - shift[:2],
            base[:, :2],
            rtol=0.0,
            atol=2.0 * screen_ulp,
        )
        self.assertLessEqual(
            _dense_projected_curve_error(moved_curve, moved),
            tolerance,
        )

    def test_curve_display_fails_below_large_coordinate_resolution(self) -> None:
        curve = CircleArcCurve(
            "moved",
            (1.0e12, -1.0e12, 0.0),
            1.0,
            (0.0, 0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(
            QuadricManimError,
            "floating-point screen resolution",
        ):
            _adaptive_project_curve(
                curve,
                IDENTITY_VIEW,
                0.0,
                tau,
                max_chord_error=1.0e-5,
                max_segments=4096,
            )

    def test_curve_display_clamps_only_floating_point_domain_roundoff(self) -> None:
        curve = SegmentCurve("domain-roundoff", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        one_ulp_past_end = float(np.nextafter(1.0, np.inf))
        points = _adaptive_project_curve(
            curve,
            IDENTITY_VIEW,
            0.0,
            one_ulp_past_end,
            max_chord_error=0.01,
            max_segments=16,
        )
        np.testing.assert_allclose(points[-1], (1.0, 0.0, 0.0))

        with self.assertRaisesRegex(
            QuadricManimError,
            "outside its authored parameter domain",
        ):
            _adaptive_project_curve(
                curve,
                IDENTITY_VIEW,
                0.0,
                1.0 + 1.0e-8,
                max_chord_error=0.01,
                max_segments=16,
            )

    def test_diagrammatic_and_physical_follow_complete_draw_order(self) -> None:
        for policy in ("diagrammatic", "physical"):
            with self.subTest(policy=policy):
                scene = Scene()
                fixture = _QuadricFixture(scene, paint_policy=policy)
                controller = fixture.controller.attach()
                frame = controller.last_frame
                assert frame is not None
                self.assertEqual(
                    set(controller.active_painter_z_indices),
                    set(frame.draw_order),
                )
                ordered_z = [
                    controller.active_painter_z_indices[item_id]
                    for item_id in frame.draw_order
                ]
                self.assertEqual(ordered_z, sorted(ordered_z))
                hidden = next(
                    item
                    for item in frame.curve_fragments
                    if item.kind is QuadricPaintKind.HIDDEN_CURVE
                )
                if policy == "diagrammatic":
                    self.assertIs(frame.paint_policy, QuadricPaintPolicy.DIAGRAMMATIC)
                    self.assertTrue(hidden.painted)
                    self.assertEqual(hidden.render_intent, "dashed")
                    self.assertIn(hidden.item_id, frame.draw_order)
                else:
                    self.assertIs(frame.paint_policy, QuadricPaintPolicy.PHYSICAL)
                    self.assertFalse(hidden.painted)
                    self.assertEqual(hidden.render_intent, "omit")
                    self.assertNotIn(hidden.item_id, frame.draw_order)
                controller.restore()

    def test_depth_aware_diagrammatic_binding_uses_compositor_depth(self) -> None:
        controller = _QuadricFixture(
            Scene(),
            paint_policy="depth_aware_diagrammatic",
        ).controller.attach()
        try:
            frame = controller.last_frame
            self.assertIsNotNone(frame)
            assert frame is not None
            self.assertIs(
                frame.paint_policy,
                QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            )
            hidden = tuple(
                item
                for item in frame.curve_fragments
                if item.kind is QuadricPaintKind.HIDDEN_CURVE
            )
            visible = tuple(
                item
                for item in frame.curve_fragments
                if item.kind is QuadricPaintKind.VISIBLE_CURVE
            )
            self.assertTrue(hidden)
            self.assertTrue(visible)
            surface_id = frame.surface_items[0].item_id
            ranks = {
                item_id: index for index, item_id in enumerate(frame.draw_order)
            }
            self.assertTrue(
                all(ranks[item.item_id] < ranks[surface_id] for item in hidden)
            )
            self.assertTrue(
                all(ranks[surface_id] < ranks[item.item_id] for item in visible)
            )
            active_z = controller.active_painter_z_indices
            self.assertTrue(
                all(
                    active_z[item.item_id] < active_z[surface_id]
                    for item in hidden
                )
            )
            self.assertTrue(
                all(
                    active_z[surface_id] < active_z[item.item_id]
                    for item in visible
                )
            )
        finally:
            controller.restore()

    def test_depth_aware_multiple_surfaces_bracket_hidden_curve(self) -> None:
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(
                SphereSpec("alpha", (0.0, 0.0, 2.0), 0.8),
                SphereSpec("beta", (0.0, 0.0, -2.0), 0.8),
            ),
            curves=(
                SegmentCurve("between", (-0.6, 0.0, 0.0), (0.6, 0.0, 0.0)),
            ),
            projection=IDENTITY_VIEW,
            paint_policy="depth_aware_diagrammatic",
            limits=_limits(max_fragments_per_curve=4),
            max_chord_error=0.015,
        ).attach()
        try:
            frame = controller.last_frame
            assert frame is not None
            hidden = next(
                item
                for item in frame.curve_fragments
                if item.kind is QuadricPaintKind.HIDDEN_CURVE
            )
            ranks = {
                item_id: index for index, item_id in enumerate(frame.draw_order)
            }
            beta = "surface:beta:opaque-projection"
            alpha = "surface:alpha:opaque-projection"
            self.assertLess(ranks[beta], ranks[hidden.item_id])
            self.assertLess(ranks[hidden.item_id], ranks[alpha])
            active_z = controller.active_painter_z_indices
            self.assertLess(active_z[beta], active_z[hidden.item_id])
            self.assertLess(active_z[hidden.item_id], active_z[alpha])
        finally:
            controller.restore()

    def test_automatic_surface_order_uses_depth_instead_of_identity_order(
        self,
    ) -> None:
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(
                SphereSpec("a-near", (0.0, 0.0, 3.0), 0.5),
                SphereSpec("z-far", (0.0, 0.0, 0.0), 0.5),
            ),
            curves=(),
            projection=IDENTITY_VIEW,
            limits=_limits(),
            max_chord_error=0.015,
        ).attach()

        self.assertEqual(
            controller.last_frame.draw_order,  # type: ignore[union-attr]
            (
                "surface:z-far:opaque-projection",
                "surface:a-near:opaque-projection",
            ),
        )
        global_frame = controller.last_global_frame
        self.assertIsNotNone(global_frame)
        assert global_frame is not None
        self.assertIs(controller.last_frame, global_frame.frame)
        self.assertEqual(
            tuple(
                (item.farther_surface_id, item.nearer_surface_id)
                for item in global_frame.surface_constraints
            ),
            (("z-far", "a-near"),),
        )
        prepared = controller.prepare()
        self.assertIsNotNone(prepared.global_frame)
        assert prepared.global_frame is not None
        self.assertIs(prepared.frame, prepared.global_frame.frame)
        self.assertIs(controller.last_global_frame, global_frame)
        controller.restore()

    def test_explicit_surface_order_mode_keeps_legacy_compositor_path(self) -> None:
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(
                SphereSpec("a-near", (0.0, 0.0, 3.0), 0.5),
                SphereSpec("z-far", (0.0, 0.0, 0.0), 0.5),
            ),
            curves=(),
            projection=IDENTITY_VIEW,
            surface_order_mode="explicit",
            limits=_limits(),
            max_chord_error=0.015,
        ).attach()

        self.assertIsNone(controller.last_global_frame)
        self.assertIsNone(controller.prepare().global_frame)
        self.assertEqual(
            controller.last_frame.draw_order,  # type: ignore[union-attr]
            (
                "surface:a-near:opaque-projection",
                "surface:z-far:opaque-projection",
            ),
        )
        controller.restore()

    def test_automatic_order_recomputes_dynamic_depth_without_stale_constraints(
        self,
    ) -> None:
        depths = {"a": 0.0, "b": 3.0}

        def surfaces() -> tuple[SphereSpec, SphereSpec]:
            return (
                SphereSpec("a", (0.0, 0.0, depths["a"]), 0.5),
                SphereSpec("b", (0.0, 0.0, depths["b"]), 0.5),
            )

        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=surfaces,
            curves=(),
            projection=IDENTITY_VIEW,
            limits=_limits(),
            max_chord_error=0.015,
        ).attach()
        first_global = controller.last_global_frame
        self.assertEqual(
            controller.last_frame.draw_order,  # type: ignore[union-attr]
            (
                "surface:a:opaque-projection",
                "surface:b:opaque-projection",
            ),
        )

        depths.update(a=3.0, b=0.0)
        controller.update()
        self.assertIsNot(controller.last_global_frame, first_global)
        self.assertEqual(
            controller.last_frame.draw_order,  # type: ignore[union-attr]
            (
                "surface:b:opaque-projection",
                "surface:a:opaque-projection",
            ),
        )
        global_frame = controller.last_global_frame
        assert global_frame is not None
        self.assertEqual(
            tuple(
                (item.farther_surface_id, item.nearer_surface_id)
                for item in global_frame.surface_constraints
            ),
            (("b", "a"),),
        )
        controller.restore()

    def test_automatic_global_failure_rolls_back_every_committed_state(self) -> None:
        depths = {"a": 0.0, "b": 3.0}

        def surfaces() -> tuple[SphereSpec, SphereSpec]:
            return (
                SphereSpec("a", (0.0, 0.0, depths["a"]), 0.5),
                SphereSpec("b", (0.0, 0.0, depths["b"]), 0.5),
            )

        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=surfaces,
            curves=(),
            projection=IDENTITY_VIEW,
            limits=_limits(),
            max_chord_error=0.015,
        ).attach()
        snapshot = controller.slot_snapshot()
        previous_z = controller.active_painter_z_indices
        previous_frame = controller.last_frame
        previous_global = controller.last_global_frame

        depths.update(a=1.0, b=1.0)
        with self.assertRaisesRegex(
            QuadricManimError,
            "automatic global quadric ordering failed",
        ) as caught:
            controller.update()

        self.assertIsInstance(caught.exception.__cause__, GlobalQuadricOcclusionError)
        self.assertEqual(controller.slot_snapshot(), snapshot)
        self.assertEqual(controller.active_painter_z_indices, previous_z)
        self.assertIs(controller.last_frame, previous_frame)
        self.assertIs(controller.last_global_frame, previous_global)
        controller.restore()

    def test_automatic_additional_constraints_report_conflict_and_cycle(self) -> None:
        conflict = QuadricOcclusion3D(
            Scene(),
            surfaces=(
                SphereSpec("a-near", (0.0, 0.0, 3.0), 0.5),
                SphereSpec("z-far", (0.0, 0.0, 0.0), 0.5),
            ),
            curves=(),
            projection=IDENTITY_VIEW,
            surface_constraints=(("a-near", "z-far"),),
            limits=_limits(),
            max_chord_error=0.015,
        )
        with self.assertRaisesRegex(QuadricManimError, "contradictory direct evidence"):
            conflict.attach()

        cycle_scene = Scene()
        cycle = QuadricOcclusion3D(
            cycle_scene,
            surfaces=(
                SphereSpec("a", (-5.0, 0.0, 0.0), 0.5),
                SphereSpec("b", (0.0, 0.0, 0.0), 0.5),
                SphereSpec("c", (5.0, 0.0, 0.0), 0.5),
            ),
            curves=(),
            projection=IDENTITY_VIEW,
            surface_constraints=(("a", "b"), ("b", "c"), ("c", "a")),
            limits=_limits(max_surfaces=3),
            max_chord_error=0.015,
        )
        with self.assertRaisesRegex(QuadricManimError, "cycle") as caught:
            cycle.attach()
        self.assertIsInstance(caught.exception.__cause__, GlobalQuadricOcclusionError)
        self.assertEqual(cycle_scene.mobjects, [])
        self.assertFalse(cycle.attached)

    def test_automatic_global_and_projection_failures_use_public_errors(self) -> None:
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=(),
            projection=IDENTITY_VIEW,
            limits=_limits(),
            max_chord_error=0.015,
        )
        with patch(
            "polyhedron_visibility.quadrics.manim.compute_global_quadric_frame",
            side_effect=GlobalQuadricOcclusionError("synthetic global failure"),
        ):
            with self.assertRaisesRegex(QuadricManimError, "synthetic global failure"):
                controller.attach()

        with patch(
            "polyhedron_visibility.quadrics.manim.compute_global_quadric_frame",
            side_effect=ProjectionProxyError("synthetic projection failure"),
        ):
            with self.assertRaisesRegex(
                QuadricManimError,
                "synthetic projection failure",
            ) as caught:
                controller.attach()
        self.assertIsInstance(caught.exception.__cause__, ProjectionProxyError)

        with self.assertRaisesRegex(
            QuadricManimError,
            "surface_order_mode must be",
        ):
            QuadricOcclusion3D(
                Scene(),
                surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
                curves=(),
                projection=IDENTITY_VIEW,
                surface_order_mode="guess",
            )

    def test_physical_mode_ignores_fully_hidden_coincident_strokes(self) -> None:
        scene = Scene()
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 3.0), 1.0),),
            curves=(
                SegmentCurve("far", (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
                SegmentCurve("near", (-0.5, 0.0, 1.0), (0.5, 0.0, 1.0)),
            ),
            projection=IDENTITY_VIEW,
            paint_policy="physical",
            limits=_limits(),
            max_chord_error=0.015,
        ).attach()
        frame = controller.last_frame
        assert frame is not None
        self.assertFalse(any(item.painted for item in frame.curve_fragments))
        self.assertEqual(frame.curve_crossings, ())
        controller.restore()

    def test_update_constructs_no_mobject_and_preserves_every_slot_identity(
        self,
    ) -> None:
        fixture = _QuadricFixture(Scene())
        controller = fixture.controller.attach()
        identities = controller.slot_identities()
        previous_frame = controller.last_frame
        fixture.offset = 0.35

        with patch.object(
            Mobject,
            "__init__",
            side_effect=AssertionError("updater allocated a Mobject"),
        ):
            controller.update()

        self.assertEqual(controller.slot_identities(), identities)
        self.assertIsNot(controller.last_frame, previous_frame)
        controller.restore()

    def test_capacity_failure_happens_before_scene_ownership_changes(self) -> None:
        scene = Scene()
        fixture = _QuadricFixture(
            scene,
            limits=_limits(max_dashes_per_fragment=1),
        )
        before = tuple(scene.mobjects)
        with self.assertRaisesRegex(QuadricManimCapacityError, "dash count exceeds"):
            fixture.controller.attach()
        self.assertEqual(tuple(scene.mobjects), before)
        self.assertFalse(fixture.controller.attached)
        self.assertIsNone(fixture.controller.last_frame)

    def test_curve_subdivision_capacity_fails_closed(self) -> None:
        scene = Scene()
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=(
                CircleArcCurve(
                    "silhouette",
                    (0.0, 0.0, 0.0),
                    1.0,
                    (0.0, 0.0, 1.0),
                    radial_axis=(1.0, 0.0, 0.0),
                ),
            ),
            projection=IDENTITY_VIEW,
            limits=_limits(max_segments_per_fragment=1),
            max_chord_error=0.015,
        )
        before = tuple(scene.mobjects)
        with self.assertRaisesRegex(QuadricManimCapacityError, "display segments"):
            controller.attach()
        self.assertEqual(tuple(scene.mobjects), before)

    def test_surface_subdivision_capacity_fails_closed(self) -> None:
        scene = Scene()
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=(),
            projection=IDENTITY_VIEW,
            limits=_limits(max_surface_segments=8),
            max_chord_error=1.0e-6,
        )
        before = tuple(scene.mobjects)
        with self.assertRaisesRegex(
            QuadricManimCapacityError, "adaptive projection outline"
        ) as caught:
            controller.attach()
        self.assertIsInstance(caught.exception.__cause__, ProjectionSubdivisionError)
        self.assertEqual(tuple(scene.mobjects), before)

    def test_apply_failure_restores_geometry_order_mapping_and_last_frame(self) -> None:
        fixture = _QuadricFixture(Scene())
        controller = fixture.controller.attach()
        snapshot = controller.slot_snapshot()
        identities = controller.slot_identities()
        previous_z = controller.active_painter_z_indices
        previous_frame = controller.last_frame
        previous_global = controller.last_global_frame
        previous_maps = {
            key: dict(value) for key, value in controller._fragment_slot_maps.items()
        }
        fixture.offset = -0.4
        original_apply = controller._band.apply

        def fail_after_commit(prepared) -> None:
            original_apply(prepared)
            raise RuntimeError("synthetic painter commit failure")

        with patch.object(controller._band, "apply", side_effect=fail_after_commit):
            with self.assertRaisesRegex(RuntimeError, "synthetic painter"):
                controller.update()

        self.assertEqual(controller.slot_snapshot(), snapshot)
        self.assertEqual(controller.slot_identities(), identities)
        self.assertEqual(controller.active_painter_z_indices, previous_z)
        self.assertEqual(controller._fragment_slot_maps, previous_maps)
        self.assertIs(controller.last_frame, previous_frame)
        self.assertIs(controller.last_global_frame, previous_global)
        controller.restore()

    def test_band_intruder_is_rejected_without_leaking_controller_roots(self) -> None:
        scene = Scene()
        fixture = _QuadricFixture(scene)
        intruder = Line((-2, 2, 0), (2, 2, 0)).set_z_index(25.0)
        scene.add(intruder)
        with self.assertRaisesRegex(QuadricManimError, "managed painter z band"):
            fixture.controller.attach()
        self.assertEqual(scene.mobjects, [intruder])
        self.assertFalse(fixture.controller.attached)
        self.assertIsNone(fixture.controller.last_global_frame)

    def test_display_opacity_multiplier_survives_geometry_update_and_reattach(
        self,
    ) -> None:
        fixture = _QuadricFixture(Scene())
        controller = fixture.controller.attach()
        identities = controller.slot_identities()
        controller.display_mobject.set_opacity(0.25)
        fixture.offset = 0.2
        controller.update()

        surface = controller._surface_slots[0]
        self.assertLessEqual(float(surface.get_fill_opacity()), 0.2500001)
        active_strokes = []
        for slots in controller._curve_slots.values():
            for slot in slots.fragments:
                active_strokes.append(float(slot.solid.get_stroke_opacity()))
                active_strokes.extend(
                    float(dash.get_stroke_opacity()) for dash in slot.dashes
                )
        self.assertTrue(any(0.0 < value <= 0.2500001 for value in active_strokes))

        controller.detach()
        controller.attach()
        self.assertEqual(controller.slot_identities(), identities)
        self.assertAlmostEqual(controller.root.opacity_multiplier, 1.0)
        controller.restore()


class QuadricManimCairoRenderTests(unittest.TestCase):
    def test_depth_aware_hidden_curve_is_attenuated_only_by_nearer_surface(
        self,
    ) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 640,
                "pixel_height": 360,
                "frame_rate": 5,
                "disable_caching": True,
                "write_to_movie": False,
                "save_last_frame": False,
            }
        ):
            scene = Scene()
            scene.camera.background_color = "#FFFFFF"
            controller = QuadricOcclusion3D(
                scene,
                surfaces=(
                    SphereSpec("alpha", (0.0, 0.0, 2.0), 0.8),
                    SphereSpec("beta", (0.0, 0.0, -2.0), 0.8),
                ),
                curves=(
                    SegmentCurve(
                        "between",
                        (-0.6, 0.0, 0.0),
                        (0.6, 0.0, 0.0),
                    ),
                ),
                projection=IDENTITY_VIEW,
                paint_policy="depth_aware_diagrammatic",
                style=QuadricManimStyle(
                    surface_fill_color="#2050A0",
                    surface_fill_opacity=0.5,
                    surface_stroke_opacity=0.0,
                    hidden_curve_color="#F00000",
                    hidden_curve_opacity=1.0,
                    hidden_curve_width=8.0,
                    dash_length=0.3,
                    dash_gap=0.15,
                ),
                limits=_limits(),
                max_chord_error=0.01,
            ).attach()
            try:
                frame = controller.last_frame
                assert frame is not None
                hidden = next(
                    item
                    for item in frame.curve_fragments
                    if item.kind is QuadricPaintKind.HIDDEN_CURVE
                )
                slot_index = controller._fragment_slot_maps["between"][hidden.item_id]
                slot = controller._curve_slots["between"].fragments[slot_index]
                dashes = tuple(dash for dash in slot.dashes if len(dash.points))
                self.assertTrue(dashes)
                dash = dashes[len(dashes) // 2]
                point = 0.5 * (
                    np.asarray(dash.get_start()) + np.asarray(dash.get_end())
                )

                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                pixels = scene.camera.pixel_array[:, :, :3]
                column = int(
                    round(
                        (point[0] / float(config.frame_width) + 0.5)
                        * (int(config.pixel_width) - 1)
                    )
                )
                row = int(
                    round(
                        (0.5 - point[1] / float(config.frame_height))
                        * (int(config.pixel_height) - 1)
                    )
                )
                # beta is painted first, the opaque red dash second, and the
                # 50%-opaque alpha fill last: 0.5 * #2050A0 + 0.5 * #F00000.
                np.testing.assert_allclose(
                    pixels[row, column].astype(float),
                    np.asarray((136.0, 40.0, 80.0)),
                    atol=2.0,
                )
            finally:
                controller.restore()

    def test_real_cairo_recomputes_automatic_global_surface_order(self) -> None:
        class AutomaticGlobalScene(Scene):
            def construct(inner_self) -> None:
                phase = ValueTracker(0.0)

                def surfaces() -> tuple[SphereSpec, SphereSpec]:
                    first_depth, second_depth = (
                        (0.0, 3.0)
                        if phase.get_value() < 0.5
                        else (3.0, 0.0)
                    )
                    return (
                        SphereSpec("a", (0.0, 0.0, first_depth), 0.55),
                        SphereSpec("b", (0.0, 0.0, second_depth), 0.55),
                    )

                controller = QuadricOcclusion3D(
                    inner_self,
                    surfaces=surfaces,
                    curves=(),
                    projection=IDENTITY_VIEW,
                    limits=_limits(),
                    max_chord_error=0.015,
                ).attach()
                inner_self.initial_order = controller.last_frame.draw_order
                inner_self.play(phase.animate.set_value(1.0), run_time=0.4)
                inner_self.final_order = controller.last_frame.draw_order
                inner_self.has_global_evidence = bool(
                    controller.last_global_frame
                    and controller.last_global_frame.surface_constraints
                )
                frame = inner_self.renderer.get_frame()
                background = frame[0, 0, :3]
                inner_self.non_background_pixels = int(
                    np.count_nonzero(np.any(frame[:, :, :3] != background, axis=2))
                )
                controller.restore()

        with (
            TemporaryDirectory() as media_dir,
            tempconfig(
                {
                    "renderer": "cairo",
                    "media_dir": media_dir,
                    "pixel_width": 160,
                    "pixel_height": 90,
                    "frame_rate": 5,
                    "disable_caching": True,
                    "write_to_movie": True,
                    "save_last_frame": False,
                }
            ),
        ):
            scene = AutomaticGlobalScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertEqual(
                scene.initial_order,
                (
                    "surface:a:opaque-projection",
                    "surface:b:opaque-projection",
                ),
            )
            self.assertEqual(
                scene.final_order,
                (
                    "surface:b:opaque-projection",
                    "surface:a:opaque-projection",
                ),
            )
            self.assertTrue(scene.has_global_evidence)
            self.assertGreater(scene.non_background_pixels, 10)

    def test_real_cairo_keeps_preexisting_static_art_during_motion_and_wait(
        self,
    ) -> None:
        class StaticBackgroundScene(Scene):
            @staticmethod
            def marker_pixels(frame: np.ndarray) -> int:
                rgb = np.asarray(frame[:, :, :3], dtype=np.int16)
                return int(
                    np.count_nonzero(
                        (rgb[:, :, 0] > 180)
                        & (rgb[:, :, 1] < 90)
                        & (rgb[:, :, 2] > 180)
                    )
                )

            def construct(inner_self) -> None:
                marker = Line((-5.5, -1.2, 0), (-5.5, 1.2, 0))
                marker.set_stroke("#ff00ff", width=12)
                inner_self.add(marker)

                fixture = _QuadricFixture(inner_self)
                controller = fixture.controller.attach()
                motion = ValueTracker(0.0)

                def sync_fixture(value: Mobject, dt: float) -> None:
                    del value, dt
                    fixture.offset = motion.get_value()

                controller._update_driver.add_updater(sync_fixture, index=0)
                inner_self.play(motion.animate.set_value(0.4), run_time=0.4)
                inner_self.motion_marker_pixels = inner_self.marker_pixels(
                    inner_self.renderer.get_frame()
                )

                inner_self.wait(0.4, frozen_frame=False)
                inner_self.wait_marker_pixels = inner_self.marker_pixels(
                    inner_self.renderer.get_frame()
                )
                inner_self.marker_still_owned = marker in inner_self.mobjects
                controller.restore()

        with (
            TemporaryDirectory() as media_dir,
            tempconfig(
                {
                    "renderer": "cairo",
                    "media_dir": media_dir,
                    "pixel_width": 200,
                    "pixel_height": 120,
                    "frame_rate": 5,
                    "disable_caching": True,
                    "write_to_movie": True,
                    "save_last_frame": False,
                }
            ),
        ):
            scene = StaticBackgroundScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertGreater(scene.motion_marker_pixels, 10)
            self.assertGreater(scene.wait_marker_pixels, 10)
            self.assertTrue(scene.marker_still_owned)

    def test_real_cairo_render_updates_during_fade_without_losing_lifecycle(
        self,
    ) -> None:
        class QuadricScene(Scene):
            def construct(inner_self) -> None:
                fixture = _QuadricFixture(inner_self)
                controller = fixture.controller.attach()
                identities = controller.slot_identities()
                inner_self.camera.reset()
                inner_self.camera.capture_mobjects(inner_self.mobjects)
                pixels = inner_self.camera.pixel_array.copy()
                background = pixels[0, 0, :3]
                inner_self.non_background_pixels = int(
                    np.count_nonzero(np.any(pixels[:, :, :3] != background, axis=2))
                )
                motion = ValueTracker(0.0)
                opacity_samples: list[float] = []

                def sync_fixture(value: Mobject, dt: float) -> None:
                    del value, dt
                    fixture.offset = motion.get_value()

                def capture_opacity(value: Mobject, dt: float) -> None:
                    del value, dt
                    opacity_samples.append(controller.root.opacity_multiplier)

                controller._update_driver.add_updater(sync_fixture, index=0)
                controller._update_driver.add_updater(capture_opacity)
                before = controller.last_frame
                inner_self.play(
                    FadeOut(controller.display_mobject),
                    motion.animate.set_value(0.4),
                    run_time=0.4,
                )
                inner_self.fade_out_updated = (
                    controller.last_frame is not before
                    and np.isclose(fixture.offset, 0.4)
                )
                inner_self.fade_out_minimum = min(opacity_samples)
                inner_self.fade_out_ownership = (
                    controller.display_mobject not in inner_self.mobjects
                    and controller._update_driver in inner_self.mobjects
                )

                opacity_samples.clear()
                inner_self.play(FadeIn(controller.display_mobject), run_time=0.4)
                inner_self.fade_in_minimum = min(opacity_samples)
                inner_self.fade_in_final = controller.root.opacity_multiplier
                inner_self.single_ownership = (
                    inner_self.mobjects.count(controller.display_mobject) == 1
                    and inner_self.mobjects.count(controller._update_driver) == 1
                )
                inner_self.identity_stable = identities == controller.slot_identities()
                controller.restore()
                inner_self.restored = (
                    controller.display_mobject not in inner_self.mobjects
                    and controller._update_driver not in inner_self.mobjects
                )

        with (
            TemporaryDirectory() as media_dir,
            tempconfig(
                {
                    "renderer": "cairo",
                    "media_dir": media_dir,
                    "pixel_width": 160,
                    "pixel_height": 90,
                    "frame_rate": 5,
                    "disable_caching": True,
                    "write_to_movie": True,
                    "save_last_frame": False,
                }
            ),
        ):
            scene = QuadricScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertGreater(scene.non_background_pixels, 10)
            self.assertTrue(scene.fade_out_updated)
            self.assertLess(scene.fade_out_minimum, 1.0)
            self.assertTrue(scene.fade_out_ownership)
            self.assertLessEqual(scene.fade_in_minimum, 1.0e-6)
            self.assertGreaterEqual(scene.fade_in_final, 1.0 - 1.0e-6)
            self.assertTrue(scene.single_ownership)
            self.assertTrue(scene.identity_stable)
            self.assertTrue(scene.restored)


if __name__ == "__main__":
    unittest.main()
