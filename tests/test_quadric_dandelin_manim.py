from __future__ import annotations

from math import pi
import unittest
from unittest.mock import patch

import numpy as np
from manim import Scene, ThreeDScene, config

from polyhedron_visibility.geometry import GeometryContext
from polyhedron_visibility.painter_band import (
    ScenePainterBandError,
    ScenePainterBandReservation,
    release_scene_painter_band,
    reserve_scene_painter_band,
    scene_painter_band_allocations,
)
from polyhedron_visibility.quadrics import (
    ConeModel,
    ConeSpec,
    DandelinSection3D,
    DandelinSectionAuthoringError,
    QuadricManimError,
    QuadricManimLimits,
    SectionPlane,
)
from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
)
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
)
from polyhedron_visibility.visibility import VisibilityKind
from tikz_native.parallel_camera import ParallelCameraState


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=4,
        max_curves=8,
        max_fragments_per_curve=20,
        max_segments_per_fragment=384,
        max_surface_segments=512,
        max_dashes_per_fragment=96,
        max_projected_length=32.0,
        max_total_mobjects=10000,
        max_boundary_sources=32,
    )


def _ellipse_facade(
    scene: Scene,
    *,
    projection: object = None,
    context: GeometryContext | None = None,
    construction_id: str = "dandelin-authoring",
    **options: object,
) -> DandelinSection3D:
    values: dict[str, object] = {
        "cone": ConeSpec(
            "dandelin-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 9.0),
            model=ConeModel.OPEN_SINGLE,
        ),
        "plane": SectionPlane(
            "dandelin-plane",
            (0.0, 0.0, 2.0),
            (0.6, 0.0, 0.8),
            u_axis=(0.0, 1.0, 0.0),
        ),
        "construction_id": construction_id,
        "projection": projection,
        "context": context,
        "limits": _limits(),
        "max_chord_error": 0.02,
        "section_max_screen_error": 0.12,
    }
    values.update(options)
    return DandelinSection3D(scene, **values)  # type: ignore[arg-type]


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


def _low_level_section_controller(facade: DandelinSection3D) -> object:
    controller = facade.section_controller
    return getattr(controller, "controller", controller)


def _hyperbola_facade(scene: Scene) -> DandelinSection3D:
    return DandelinSection3D(
        scene,
        cone=ConeSpec(
            "double-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (-4.0, 4.0),
            model=ConeModel.OPEN_DOUBLE,
        ),
        plane=SectionPlane(
            "hyperbola-plane",
            (0.0, 0.0, 2.0),
            ((1.0 - 0.2**2) ** 0.5, 0.0, 0.2),
            u_axis=(0.0, 1.0, 0.0),
        ),
        construction_id="hyperbola-authoring",
        limits=_limits(),
        max_chord_error=0.02,
        section_max_screen_error=0.12,
    )


class DandelinSection3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config.renderer = "cairo"

    def test_static_facade_builds_slots_lazily_and_reattaches_cleanly(self) -> None:
        scene = Scene()
        facade = _ellipse_facade(scene)
        display_identity = id(facade.display_mobject)
        with self.assertRaisesRegex(
            DandelinSectionAuthoringError,
            "slot_identities.*only while attached",
        ):
            facade.slot_identities()
        self.assertEqual(tuple(facade.display_mobject.submobjects), ())
        self.assertIsNone(facade.painter_z_band)
        self.assertIsNone(facade.painter_subbands)
        self.assertIsNone(facade.focus_z)

        facade.attach()

        self.assertTrue(facade.slot_identities())
        self.assertTrue(facade.attached)
        self.assertEqual(id(facade.display_mobject), display_identity)
        self.assertFalse(facade.visibility_authoritative)
        self.assertEqual(facade.overlay_mode, "diagrammatic")
        self.assertEqual(facade.painter_z_band, (10.0, 32.0))
        self.assertEqual(facade.section_painter_z_band, (10.0, 20.0))
        self.assertEqual(facade.overlay_painter_z_band, (21.0, 31.0))
        self.assertEqual(facade.focus_z, 32.0)
        self.assertEqual(len(scene_painter_band_allocations(scene)), 1)
        self.assertGreater(len(scene.mobjects), 0)
        first_section_controller = facade.section_controller

        facade.restore()

        self.assertFalse(facade.attached)
        self.assertEqual(scene.mobjects, [])
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertIsNone(facade.painter_z_band)
        self.assertEqual(tuple(facade.display_mobject.submobjects), ())
        with self.assertRaisesRegex(
            DandelinSectionAuthoringError,
            "slot_identities.*only while attached",
        ):
            facade.slot_identities()

        facade.attach()
        try:
            self.assertEqual(id(facade.display_mobject), display_identity)
            self.assertIsNot(facade.section_controller, first_section_controller)
            self.assertTrue(facade.slot_identities())
            self.assertEqual(facade.painter_z_band, (10.0, 32.0))
        finally:
            facade.restore()
        self.assertEqual(scene_painter_band_allocations(scene), ())

    def test_attach_failure_rolls_back_the_already_attached_section(self) -> None:
        scene = Scene()
        facade = _ellipse_facade(scene)
        before_scene = _scene_ownership_snapshot(scene)
        before_bands = scene_painter_band_allocations(scene)

        with patch.object(
            facade,
            "_attach_overlay_layer",
            side_effect=RuntimeError("injected overlay failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected overlay failure"):
                facade.attach()

        self.assertFalse(facade.attached)
        self.assertIsNone(facade._section_controller)
        self.assertIsNone(facade._overlay_controller)
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), before_bands)

    def test_semantic_camera_is_frozen_once_for_every_layer_and_focus(self) -> None:
        scene = ThreeDScene()
        camera = scene.camera
        camera.frame_center = np.asarray((0.8, -0.45, 0.2), dtype=float)
        camera.set_zoom(1.3)
        state = ParallelCameraState(
            np.asarray(
                (
                    (0.8, -0.6, 0.0),
                    (0.3, 0.4, -0.8660254037844386),
                    (0.5196152422706632, 0.6928203230275509, 0.5),
                ),
                dtype=float,
            ),
            target=(1.4, -0.8, 0.6),
            screen_anchor=(-0.9, 0.55),
            zoom=1.25,
        )
        expected_matrix = np.array(state.matrix, dtype=float, copy=True)
        expected_matrix[:2] *= state.zoom * 1.3
        expected_offset = (
            np.asarray(state.screen_anchor, dtype=float)
            - expected_matrix[:2] @ np.asarray(state.target, dtype=float)
            + np.asarray((0.8, -0.45), dtype=float)
        )
        before_scene = _scene_ownership_snapshot(scene)
        before_fixed = set(camera.fixed_in_frame_mobjects)
        context = GeometryContext(screen_tolerance=0.001)

        facade = _ellipse_facade(scene, projection=state, context=context)
        np.testing.assert_allclose(
            facade.view.matrix,
            expected_matrix,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            facade.display_offset,
            expected_offset,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertFalse(facade._projection_frame.viewport_relative)
        self.assertIs(
            facade.resolved_context,
            facade.construction.certification_context,
        )
        self.assertIs(facade.context, context)

        # Changing the ambient camera after construction must not silently
        # reinterpret this static facade's already-authored camera state.
        camera.frame_center = np.asarray((-1.2, 0.75, -0.3), dtype=float)
        camera.set_zoom(0.7)
        facade.attach()
        try:
            self.assertIs(
                facade.section_controller.context,
                facade.resolved_context,
            )
            self.assertIs(
                facade.overlay_controller.context,
                facade.resolved_context,
            )
            section = _low_level_section_controller(facade)
            self.assertIs(
                section._resolve_projection_frame(),
                facade._projection_frame,
            )
            self.assertIs(
                facade.overlay_controller._resolve_projection_frame(),
                facade._projection_frame,
            )
            for dot, sphere in zip(
                facade.focus_group.submobjects,
                facade.construction.spheres,
                strict=True,
            ):
                expected = (
                    expected_matrix[:2]
                    @ np.asarray(sphere.focus.world_point, dtype=float)
                    + expected_offset
                )
                np.testing.assert_allclose(
                    dot.get_center(),
                    (float(expected[0]), float(expected[1]), 0.0),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            self.assertTrue(
                set(facade.focus_group.get_family()).issubset(
                    camera.fixed_in_frame_mobjects
                )
            )
        finally:
            facade.restore()

        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(set(camera.fixed_in_frame_mobjects), before_fixed)

    def test_projection_callback_is_rejected_without_call_or_scene_changes(
        self,
    ) -> None:
        scene = Scene()
        calls = 0
        before_scene = _scene_ownership_snapshot(scene)
        before_bands = scene_painter_band_allocations(scene)

        def live_projection(_scene: object) -> object:
            nonlocal calls
            calls += 1
            return ((1, 0, 0), (0, 1, 0), (0, 0, 1))

        with self.assertRaisesRegex(
            DandelinSectionAuthoringError,
            "immutable parallel projection.*callable projection is unsupported",
        ):
            DandelinSection3D(
                scene,
                cone=ConeSpec(
                    "callback-cone",
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    pi / 6.0,
                    (0.0, 9.0),
                    model=ConeModel.OPEN_SINGLE,
                ),
                plane=SectionPlane(
                    "callback-plane",
                    (0.0, 0.0, 2.0),
                    (0.6, 0.0, 0.8),
                    u_axis=(0.0, 1.0, 0.0),
                ),
                construction_id="callback-dandelin",
                projection=live_projection,
            )
        self.assertEqual(calls, 0)
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), before_bands)

    def test_author_commit_failure_rolls_back_display_cache_and_fixed_focus(
        self,
    ) -> None:
        scene = ThreeDScene()
        camera = scene.camera
        facade = _ellipse_facade(scene)
        before_scene = _scene_ownership_snapshot(scene)
        before_fixed = set(camera.fixed_in_frame_mobjects)
        before_bands = scene_painter_band_allocations(scene)
        captured: dict[str, object] = {}

        def fail_author_commit(*_args: object) -> None:
            section_facade = facade._section_controller
            overlay = facade._overlay_controller
            assert section_facade is not None and overlay is not None
            captured["section"] = getattr(
                section_facade,
                "controller",
                section_facade,
            )
            captured["overlay"] = overlay
            raise RuntimeError("injected author commit failure")

        with patch.object(
            facade,
            "_commit_author_state",
            side_effect=fail_author_commit,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected author commit failure",
            ):
                facade.attach()

        self.assertFalse(facade.attached)
        self.assertIsNone(facade._section_controller)
        self.assertIsNone(facade._overlay_controller)
        self.assertIsNone(facade._focus_group)
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(set(camera.fixed_in_frame_mobjects), before_fixed)
        self.assertEqual(scene_painter_band_allocations(scene), before_bands)
        for controller in (captured["section"], captured["overlay"]):
            self.assertIsNone(controller._last_prepared_frame)
            self.assertIsNone(controller._last_input_geometry_signature)
            self.assertIsNone(controller._last_input_draw_signature)
            self.assertIsNone(controller._last_input_opacity)
            self.assertEqual(controller._surface_view_cache._entries, {})
            self.assertEqual(controller.active_painter_z_indices, {})

    def test_default_scene_bands_auto_shift_without_overlap_and_release(self) -> None:
        scene = Scene()
        first = _ellipse_facade(
            scene,
            construction_id="dandelin-auto-first",
        )
        second = _ellipse_facade(
            scene,
            construction_id="dandelin-auto-second",
        )

        first.attach()
        second.attach()
        try:
            self.assertEqual(first.painter_z_band, (10.0, 32.0))
            self.assertEqual(second.painter_z_band, (33.0, 55.0))
            assert first.painter_z_band is not None
            assert second.painter_z_band is not None
            self.assertLess(
                first.painter_z_band[1],
                second.painter_z_band[0],
            )
            self.assertEqual(first.painter_subbands, ((10.0, 20.0), (21.0, 31.0)))
            self.assertEqual(second.painter_subbands, ((33.0, 43.0), (44.0, 54.0)))
            self.assertEqual(first.focus_z, 32.0)
            self.assertEqual(second.focus_z, 55.0)
            self.assertEqual(
                tuple(item.z_band for item in scene_painter_band_allocations(scene)),
                ((10.0, 32.0), (33.0, 55.0)),
            )
        finally:
            first.restore()
        self.assertEqual(
            tuple(item.z_band for item in scene_painter_band_allocations(scene)),
            ((33.0, 55.0),),
        )
        second.restore()
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertEqual(scene.mobjects, [])

    def test_legacy_exact_bands_are_paired_and_conflicts_fail_before_build(
        self,
    ) -> None:
        scene = Scene()
        with self.assertRaisesRegex(
            DandelinSectionAuthoringError,
            "must be provided together",
        ):
            _ellipse_facade(
                scene,
                construction_id="unpaired-band",
                section_painter_z_band=(100.0, 110.0),
            )
        for invalid_options in (
            {"preferred_painter_z_band": (False, 32.0)},
            {
                "section_painter_z_band": (np.bool_(True), 110.0),
                "overlay_painter_z_band": (111.0, 121.0),
            },
        ):
            with self.subTest(invalid_options=invalid_options):
                with self.assertRaisesRegex(
                    DandelinSectionAuthoringError,
                    "finite increasing values",
                ):
                    _ellipse_facade(
                        scene,
                        construction_id="boolean-band",
                        **invalid_options,
                    )

        options = {
            "section_painter_z_band": (100.0, 110.0),
            "overlay_painter_z_band": (111.0, 121.0),
        }
        first = _ellipse_facade(
            scene,
            construction_id="exact-band-first",
            **options,
        )
        second = _ellipse_facade(
            scene,
            construction_id="exact-band-second",
            **options,
        )
        first.attach()
        try:
            self.assertEqual(first.painter_z_band, (100.0, 122.0))
            self.assertEqual(first.section_painter_z_band, (100.0, 110.0))
            self.assertEqual(first.overlay_painter_z_band, (111.0, 121.0))
            self.assertEqual(first.focus_z, 122.0)
            before_scene = _scene_ownership_snapshot(scene)
            before_bands = scene_painter_band_allocations(scene)
            with (
                patch.object(
                    second,
                    "_build_section_controller",
                    side_effect=AssertionError(
                        "exact conflict reached controller construction"
                    ),
                ) as builder,
                self.assertRaisesRegex(
                    ScenePainterBandError,
                    "exact Scene painter z band conflicts",
                ),
            ):
                second.attach()
            builder.assert_not_called()
            self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
            self.assertEqual(scene_painter_band_allocations(scene), before_bands)
            self.assertFalse(second.attached)
            self.assertIsNone(second._section_controller)
        finally:
            first.restore()
        self.assertEqual(scene_painter_band_allocations(scene), ())

    def test_duplicate_construction_owner_is_diagnosed_without_reallocation(
        self,
    ) -> None:
        scene = Scene()
        first = _ellipse_facade(scene)
        duplicate = _ellipse_facade(scene)
        first.attach()
        try:
            before_scene = _scene_ownership_snapshot(scene)
            before_bands = scene_painter_band_allocations(scene)
            with self.assertRaisesRegex(
                ScenePainterBandError,
                "duplicate Scene painter-band owner",
            ):
                duplicate.attach()
            self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
            self.assertEqual(scene_painter_band_allocations(scene), before_bands)
            self.assertFalse(duplicate.attached)
            self.assertFalse(duplicate._reservation_active)
            self.assertIsNone(duplicate._section_controller)
        finally:
            first.restore()
        self.assertEqual(scene_painter_band_allocations(scene), ())

    def test_automatic_band_overflow_fails_without_reservation_or_scene_leak(
        self,
    ) -> None:
        scene = Scene()
        blocker = ScenePainterBandReservation(
            ("test", "automatic-overflow-blocker"),
            (10.0, float(np.finfo(float).max)),
            exact=True,
        )
        reserve_scene_painter_band(scene, blocker)
        facade = _ellipse_facade(
            scene,
            construction_id="automatic-overflow",
        )
        before_scene = _scene_ownership_snapshot(scene)
        before_bands = scene_painter_band_allocations(scene)
        try:
            with self.assertRaisesRegex(
                ScenePainterBandError,
                "overflowed or could not advance",
            ):
                facade.attach()
            self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
            self.assertEqual(scene_painter_band_allocations(scene), before_bands)
            self.assertFalse(facade.attached)
            self.assertIsNone(facade.painter_z_band)
            self.assertIsNone(facade._section_controller)
        finally:
            release_scene_painter_band(scene, blocker)
        self.assertEqual(scene_painter_band_allocations(scene), ())

    def test_subband_rounding_stall_releases_the_aggregate_fail_closed(
        self,
    ) -> None:
        scene = Scene()
        low = 1.0e300
        high = float(np.nextafter(low, np.inf))
        facade = _ellipse_facade(
            scene,
            construction_id="subband-rounding-stall",
            preferred_painter_z_band=(low, high),
        )
        before_scene = _scene_ownership_snapshot(scene)

        with self.assertRaisesRegex(
            DandelinSectionAuthoringError,
            "sub-band split lost strict ordering",
        ):
            facade.attach()

        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertFalse(facade.attached)
        self.assertIsNone(facade._section_controller)
        self.assertIsNone(facade.painter_z_band)

    def test_large_strict_subbands_still_reject_internal_z_collapse(
        self,
    ) -> None:
        scene = ThreeDScene()
        camera = scene.camera
        low = 1.0e300
        high = low
        for _ in range(16):
            high = float(np.nextafter(high, np.inf))
        facade = _ellipse_facade(
            scene,
            construction_id="internal-z-collapse",
            preferred_painter_z_band=(low, high),
        )
        before_scene = _scene_ownership_snapshot(scene)
        before_fixed = set(camera.fixed_in_frame_mobjects)

        with self.assertRaisesRegex(
            QuadricManimError,
            "finite, remain within.*strictly increasing.*insufficient "
            "floating-point",
        ):
            facade.attach()

        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(set(camera.fixed_in_frame_mobjects), before_fixed)
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertFalse(facade.attached)
        self.assertIsNone(facade._section_controller)
        self.assertIsNone(facade._overlay_controller)
        self.assertIsNone(facade._focus_group)
        self.assertFalse(facade._reservation_active)

    def test_fixed_frame_registration_failure_rolls_back_partial_camera_add(
        self,
    ) -> None:
        scene = ThreeDScene()
        camera = scene.camera
        facade = _ellipse_facade(
            scene,
            construction_id="fixed-frame-registration-failure",
        )
        before_scene = _scene_ownership_snapshot(scene)
        before_fixed = set(camera.fixed_in_frame_mobjects)
        original_add = camera.add_fixed_in_frame_mobjects

        def add_then_fail(*mobjects: object) -> None:
            original_add(*mobjects)
            raise RuntimeError("injected fixed-frame registration failure")

        with patch.object(
            camera,
            "add_fixed_in_frame_mobjects",
            side_effect=add_then_fail,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected fixed-frame registration failure",
            ):
                facade.attach()

        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(set(camera.fixed_in_frame_mobjects), before_fixed)
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertFalse(facade.attached)
        self.assertFalse(facade._reservation_active)
        self.assertIsNone(facade._section_controller)
        self.assertIsNone(facade._overlay_controller)
        self.assertIsNone(facade._focus_group)

    def test_pre_cleanup_failure_retains_runtime_for_restore_retry(self) -> None:
        scene = ThreeDScene()
        camera = scene.camera
        facade = _ellipse_facade(
            scene,
            construction_id="cleanup-retry",
        ).attach()
        overlay = facade.overlay_controller
        before_fixed = set(camera.fixed_in_frame_mobjects)
        self.assertTrue(before_fixed)

        with patch.object(
            overlay,
            "restore",
            side_effect=RuntimeError("injected pre-cleanup overlay failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected pre-cleanup overlay failure",
            ) as raised:
                facade.restore()

        self.assertTrue(
            any(
                "retained for a later restore() retry" in note
                for note in getattr(raised.exception, "__notes__", ())
            )
        )
        self.assertFalse(facade.attached)
        self.assertTrue(overlay.attached)
        self.assertIs(facade._overlay_controller, overlay)
        self.assertIsNotNone(facade._section_controller)
        self.assertIsNotNone(facade._focus_group)
        self.assertTrue(facade._reservation_active)
        self.assertEqual(len(scene_painter_band_allocations(scene)), 1)
        self.assertTrue(scene.mobjects)
        self.assertTrue(camera.fixed_in_frame_mobjects)
        self.assertIsNone(facade.painter_z_band)
        with self.assertRaisesRegex(
            DandelinSectionAuthoringError,
            "previous Dandelin painter-band release did not complete",
        ):
            facade.attach()

        facade.restore()
        self.assertFalse(facade.attached)
        self.assertFalse(facade._reservation_active)
        self.assertIsNone(facade._section_controller)
        self.assertIsNone(facade._overlay_controller)
        self.assertIsNone(facade._focus_group)
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertEqual(scene.mobjects, [])
        self.assertEqual(set(camera.fixed_in_frame_mobjects), set())

    def test_restore_aggregates_cleanup_errors_but_still_releases_and_clears(
        self,
    ) -> None:
        scene = Scene()
        facade = _ellipse_facade(
            scene,
            construction_id="cleanup-errors",
        ).attach()
        section = facade.section_controller
        overlay = facade.overlay_controller
        original_section_restore = section.restore
        original_overlay_restore = overlay.restore

        def fail_overlay_restore() -> object:
            original_overlay_restore()
            raise RuntimeError("injected overlay cleanup failure")

        def fail_section_restore() -> object:
            original_section_restore()
            raise RuntimeError("injected section cleanup failure")

        with (
            patch.object(overlay, "restore", side_effect=fail_overlay_restore),
            patch.object(section, "restore", side_effect=fail_section_restore),
            self.assertRaisesRegex(
                RuntimeError,
                "injected overlay cleanup failure",
            ) as raised,
        ):
            facade.restore()

        self.assertTrue(
            any(
                "section cleanup also failed" in note
                for note in getattr(raised.exception, "__notes__", ())
            )
        )
        self.assertFalse(facade.attached)
        self.assertIsNone(facade._section_controller)
        self.assertIsNone(facade._overlay_controller)
        self.assertIsNone(facade.painter_z_band)
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertEqual(scene.mobjects, [])

    def test_circular_section_attaches_two_spheres_without_fake_directrices(
        self,
    ) -> None:
        scene = Scene()
        facade = DandelinSection3D(
            scene,
            cone=ConeSpec(
                "circle-cone",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                pi / 6.0,
                (0.0, 10.0),
                model=ConeModel.OPEN_SINGLE,
            ),
            plane=SectionPlane(
                "circle-plane",
                (0.0, 0.0, 2.0),
                (0.0, 0.0, 1.0),
                u_axis=(1.0, 0.0, 0.0),
            ),
            construction_id="circle-authoring",
            limits=_limits(),
            max_chord_error=0.02,
            section_max_screen_error=0.12,
        )

        facade.attach()
        try:
            identities = facade.slot_identities()
            self.assertEqual(facade.construction.supporting_kind.value, "circle")
            self.assertEqual(len(facade.construction.spheres), 2)
            self.assertEqual(facade.construction.directrices, ())
            self.assertEqual(facade.slot_identities(), identities)
        finally:
            facade.restore()

        self.assertEqual(scene.mobjects, [])
        self.assertEqual(scene_painter_band_allocations(scene), ())
        with self.assertRaisesRegex(
            DandelinSectionAuthoringError,
            "slot_identities.*only while attached",
        ):
            facade.slot_identities()

    def test_open_double_hyperbola_reuses_the_existing_composite_controller(self) -> None:
        scene = Scene()
        facade = _hyperbola_facade(scene)
        self.assertEqual(
            {item.nappe_sign for item in facade.construction.spheres},
            {-1, 1},
        )

        facade.attach()
        try:
            self.assertIsInstance(
                facade.section_controller,
                CompositeQuadricSection3D,
            )
            identities = facade.slot_identities()
            self.assertEqual(facade.slot_identities(), identities)
            boundary = facade.section_controller.last_boundary_frame
            self.assertIsNotNone(boundary)
            assert boundary is not None
            sources = {item.source_id: item for item in boundary.sources}
            hidden_plane_edges = tuple(
                item
                for item in boundary.fragments
                if sources[item.source_id].source_kind
                is BoundarySourceKind.PLANE_PATCH_EDGE
                and item.effective_visibility_kind is VisibilityKind.HIDDEN
            )
            self.assertTrue(hidden_plane_edges)
            child_ids = {
                item.surface_id
                for item in facade.section_controller.children
            }
            self.assertTrue(
                all(
                    len(item.occluder_surface_ids) == 1
                    and item.occluder_surface_ids[0] in child_ids
                    for item in hidden_plane_edges
                )
            )
            self.assertEqual(
                {
                    item.occluder_surface_ids[0]
                    for item in hidden_plane_edges
                },
                child_ids,
            )
        finally:
            facade.restore()

        self.assertEqual(scene.mobjects, [])
        self.assertEqual(scene_painter_band_allocations(scene), ())
        with self.assertRaisesRegex(
            DandelinSectionAuthoringError,
            "slot_identities.*only while attached",
        ):
            facade.slot_identities()

    def test_open_double_overlay_failure_restores_both_controller_layers(self) -> None:
        scene = Scene()
        facade = _hyperbola_facade(scene)
        before_scene = _scene_ownership_snapshot(scene)
        before_bands = scene_painter_band_allocations(scene)

        with patch.object(
            facade,
            "_attach_overlay_layer",
            side_effect=RuntimeError("injected double overlay failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected double overlay failure",
            ):
                facade.attach()

        self.assertFalse(facade.attached)
        self.assertIsNone(facade._section_controller)
        self.assertIsNone(facade._overlay_controller)
        self.assertEqual(_scene_ownership_snapshot(scene), before_scene)
        self.assertEqual(scene_painter_band_allocations(scene), before_bands)


if __name__ == "__main__":
    unittest.main()
