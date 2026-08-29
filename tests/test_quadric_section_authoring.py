from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from math import pi
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from manim import Mobject, Scene, ValueTracker, linear, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.animation import SectionConicFamily
from polyhedron_visibility.quadrics.authoring import (
    QuadricSection3D,
    QuadricSectionAuthoringError,
)
from polyhedron_visibility.quadrics.boundary_compositing import (
    canonical_quadric_boundary_compositing_json,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimCapacityError,
    QuadricManimError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)
from polyhedron_visibility.quadrics.section_compositing import (
    canonical_quadric_section_compositing_json,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
    section_cap_chord_curve_ids,
)


VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)
SIDE_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)


def _limits(**overrides: object) -> QuadricManimLimits:
    values: dict[str, object] = {
        "max_surfaces": 2,
        "max_curves": 10,
        "max_fragments_per_curve": 32,
        "max_segments_per_fragment": 256,
        "max_surface_segments": 512,
        "max_dashes_per_fragment": 128,
        "max_projected_length": 18.0,
        "max_total_mobjects": 50000,
        "max_boundary_sources": 48,
    }
    values.update(overrides)
    return QuadricManimLimits(**values)  # type: ignore[arg-type]


def _cone(
    surface_id: str = "cone",
    *,
    model: ConeModel = ConeModel.CLOSED_SINGLE,
) -> ConeSpec:
    return ConeSpec(
        surface_id,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (0.0, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )


def _plane(height: float, plane_id: str = "cut") -> SectionPlane:
    return SectionPlane(
        plane_id,
        (0.0, 0.0, height),
        (0.5, 0.0, 1.0),
        u_axis=(0.0, 1.0, 0.0),
    )


def _scheduled():
    cone = ConeSpec(
        "transition-cone",
        (0.0, 0.0, -1.5),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    motion = AxisAnglePlaneMotion(
        "transition-motion",
        SectionPlane(
            "transition-plane",
            (0.0, 0.0, 0.2),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        (0.0, 0.0, 0.2),
        (0.0, 1.0, 0.0),
        0.0,
        1.2,
    )
    return track_scheduled_plane_section(
        "transition-section",
        cone,
        motion,
    )


class QuadricSectionAuthoringTests(unittest.TestCase):
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

    def test_facade_matches_manual_canonical_static_frame(self) -> None:
        cone = _cone("equivalent-cone")
        plane = _plane(1.5, "equivalent-plane")
        section_id = "equivalent-section"
        curves = compute_quadric_section_boundary_curves(
            section_id,
            cone,
            plane,
        )
        allocated = tuple(
            sorted(
                {
                    *(item.curve_id for item in curves),
                    *section_cap_chord_curve_ids(section_id, cone),
                }
            )
        )
        style = QuadricManimStyle(surface_fill_opacity=0.72)
        manual = QuadricOcclusion3D(
            Scene(),
            surfaces=(cone,),
            curves=curves,
            allocated_curve_ids=allocated,
            section_plane=plane,
            projection=VIEW,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            boundary_visibility_mode="unified",
            style=style,
            limits=_limits(),
            max_chord_error=0.01,
        ).attach()
        facade = QuadricSection3D(
            Scene(),
            surface=cone,
            section_id=section_id,
            plane=plane,
            projection=VIEW,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            style=style,
            limits=_limits(),
            max_chord_error=0.01,
        ).attach()
        try:
            manual_section = manual.last_section_frame
            facade_section = facade.last_section_frame
            manual_boundary = manual.last_boundary_frame
            facade_boundary = facade.last_boundary_frame
            assert manual_section is not None and facade_section is not None
            assert manual_boundary is not None and facade_boundary is not None
            self.assertEqual(
                canonical_quadric_section_compositing_json(manual_section),
                canonical_quadric_section_compositing_json(facade_section),
            )
            self.assertEqual(
                canonical_quadric_boundary_compositing_json(manual_boundary),
                canonical_quadric_boundary_compositing_json(facade_boundary),
            )
            self.assertEqual(facade.controller.boundary_visibility_mode, "unified")
        finally:
            facade.restore()
            manual.restore()

    def test_closed_and_open_cones_get_their_correct_boundary_contract(self) -> None:
        for model, has_cap_chord, rim_suffix in (
            (ConeModel.CLOSED_SINGLE, True, None),
            (ConeModel.OPEN_SINGLE, False, "trim_max"),
        ):
            with self.subTest(model=model.value):
                cone = _cone(f"{model.value}-cone", model=model)
                section_id = f"{model.value}-section"
                controller = QuadricSection3D(
                    Scene(),
                    surface=cone,
                    section_id=section_id,
                    plane=_plane(1.5, f"{model.value}-plane"),
                    projection=VIEW,
                    limits=_limits(),
                    max_chord_error=0.01,
                ).attach()
                try:
                    cap_ids = section_cap_chord_curve_ids(section_id, cone)
                    self.assertEqual(bool(cap_ids), has_cap_chord)
                    self.assertTrue(
                        set(cap_ids).issubset(controller.allocated_curve_ids)
                    )
                    frame = controller.last_boundary_frame
                    assert frame is not None
                    source_ids = {item.source_id for item in frame.sources}
                    self.assertTrue(
                        any(item.startswith(section_id + ":") for item in source_ids)
                    )
                    if rim_suffix is not None:
                        self.assertIn(
                            f"boundary:{cone.surface_id}:{rim_suffix}:rim",
                            source_ids,
                        )
                finally:
                    controller.restore()

    def test_moving_plane_automatically_reserves_and_toggles_cap_chord(self) -> None:
        cone = _cone("moving-cone")
        state = {"height": 0.8, "calls": 0}

        def plane() -> SectionPlane:
            state["calls"] += 1
            return _plane(state["height"], "moving-plane")

        controller = QuadricSection3D(
            Scene(),
            surface=cone,
            section_id="moving-section",
            plane=plane,
            projection=VIEW,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            limits=_limits(),
            max_chord_error=0.01,
        ).attach()
        identities = controller.slot_identities()
        chord_id = section_cap_chord_curve_ids("moving-section", cone)[0]
        self.assertIn(chord_id, controller.allocated_curve_ids)

        def active_sources() -> set[str]:
            frame = controller.last_boundary_frame
            assert frame is not None
            return {item.source_id for item in frame.sources}

        self.assertNotIn(chord_id, active_sources())
        for height, expected in ((1.5, True), (1.7, True), (0.8, False)):
            before_calls = state["calls"]
            state["height"] = height
            with patch.object(
                Mobject,
                "__init__",
                side_effect=AssertionError("facade updater allocated a Mobject"),
            ):
                controller.update()
            self.assertEqual(state["calls"], before_calls + 1)
            self.assertEqual(chord_id in active_sources(), expected)
            self.assertEqual(controller.slot_identities(), identities)
        controller.restore()

    def test_draw_section_boundary_false_keeps_plane_partition_only(self) -> None:
        controller = QuadricSection3D(
            Scene(),
            surface=_cone("partition-cone", model=ConeModel.OPEN_SINGLE),
            section_id="silent-section",
            plane=_plane(1.2, "partition-plane"),
            projection=VIEW,
            draw_section_boundary=False,
            limits=_limits(),
            max_chord_error=0.01,
        ).attach()
        try:
            self.assertEqual(controller.allocated_curve_ids, ())
            self.assertIsNotNone(controller.last_section_frame)
            frame = controller.last_boundary_frame
            assert frame is not None
            source_ids = {item.source_id for item in frame.sources}
            self.assertFalse(
                any(item.startswith("silent-section:") for item in source_ids)
            )
            self.assertTrue(
                any(
                    item.startswith("boundary:plane:partition-plane:edge:")
                    for item in source_ids
                )
            )
            self.assertIn("boundary:partition-cone:trim_max:rim", source_ids)
        finally:
            controller.restore()

    def test_scheduled_facade_visits_all_topologies_with_fixed_slots(self) -> None:
        progress = ValueTracker(0.0)
        controller = QuadricSection3D(
            Scene(),
            scheduled=_scheduled(),
            progress=progress,
            projection=VIEW,
            transition_fraction=0.05,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            limits=_limits(
                max_fragments_per_curve=16,
                max_dashes_per_fragment=72,
                max_total_mobjects=30000,
            ),
            max_chord_error=0.02,
        ).attach()
        transition = controller.transition_controller
        assert transition is not None
        identities = controller.slot_identities()
        parabolic = next(
            item
            for item in transition.plan.knots
            if "cone_parabolic" in item.critical_kinds
        )
        samples = (
            max(0.0, 0.5 * parabolic.left_start),
            parabolic.progress,
            min(1.0, 0.5 * (parabolic.right_end + 1.0)),
        )
        families: set[SectionConicFamily] = set()
        saw_cap_chord = False
        for index, sample in enumerate(samples):
            progress.set_value(sample)
            context = (
                patch.object(
                    Mobject,
                    "__init__",
                    side_effect=AssertionError(
                        "scheduled facade updater allocated a Mobject"
                    ),
                )
                if index == 1
                else nullcontext()
            )
            with context:
                controller.update()
            families.update(item.conic_family for item in controller.active_signatures)
            self.assertEqual(controller.slot_identities(), identities)
            self.assertIsNotNone(controller.last_boundary_frame)
            assert controller.last_boundary_frame is not None
            saw_cap_chord = saw_cap_chord or any(
                ":cap:cap_max:chord" in item.source_id
                for item in controller.last_boundary_frame.sources
            )
        self.assertEqual(
            families,
            {
                SectionConicFamily.OVAL,
                SectionConicFamily.PARABOLA,
                SectionConicFamily.HYPERBOLA,
            },
        )
        self.assertTrue(saw_cap_chord)
        controller.restore()

    def test_scheduled_boundary_opt_out_keeps_the_live_plane_partition(self) -> None:
        progress = ValueTracker(0.2)
        controller = QuadricSection3D(
            Scene(),
            scheduled=_scheduled(),
            progress=progress,
            projection=VIEW,
            draw_section_boundary=False,
            limits=_limits(
                max_fragments_per_curve=16,
                max_dashes_per_fragment=72,
                max_total_mobjects=30000,
            ),
            max_chord_error=0.02,
        ).attach()
        try:
            self.assertEqual(controller.allocated_curve_ids, ())
            self.assertIsNotNone(controller.last_section_frame)
            frame = controller.last_boundary_frame
            assert frame is not None
            self.assertFalse(
                any(
                    ":transition:bank:" in item.source_id
                    for item in frame.sources
                )
            )
            self.assertTrue(controller.active_signatures)
        finally:
            controller.restore()

    def test_failure_rolls_back_the_complete_facade_frame(self) -> None:
        state = {"edge_on": False}

        def plane() -> SectionPlane:
            if state["edge_on"]:
                return SectionPlane(
                    "rollback-plane",
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 1.0),
                    u_axis=(1.0, 0.0, 0.0),
                )
            return SectionPlane(
                "rollback-plane",
                (0.0, 0.35, 1.0),
                (0.0, 1.0, 0.0),
                u_axis=(1.0, 0.0, 0.0),
            )

        controller = QuadricSection3D(
            Scene(),
            surface=_cone("rollback-cone"),
            section_id="rollback-section",
            plane=plane,
            projection=SIDE_VIEW,
            draw_section_boundary=False,
            limits=replace(_limits(), max_total_mobjects=60000),
            max_chord_error=0.01,
        ).attach()
        snapshot = controller.slot_snapshot()
        identities = controller.slot_identities()
        frame = controller.last_boundary_frame
        z_indices = controller.active_painter_z_indices
        state["edge_on"] = True
        with self.assertRaisesRegex(QuadricManimError, "projects edge-on"):
            controller.update()
        self.assertEqual(controller.slot_snapshot(), snapshot)
        self.assertEqual(controller.slot_identities(), identities)
        self.assertIs(controller.last_boundary_frame, frame)
        self.assertEqual(controller.active_painter_z_indices, z_indices)
        controller.restore()

    def test_static_and_scheduled_authorities_cannot_be_mixed(self) -> None:
        with self.assertRaisesRegex(
            QuadricSectionAuthoringError,
            "cannot also define surface",
        ):
            QuadricSection3D(
                Scene(),
                surface=_cone(),
                scheduled=_scheduled(),
                progress=0.0,
            )
        with self.assertRaisesRegex(
            QuadricSectionAuthoringError,
            "requires progress",
        ):
            QuadricSection3D(Scene(), scheduled=_scheduled())
        with self.assertRaisesRegex(
            QuadricSectionAuthoringError,
            "requires scheduled mode",
        ):
            QuadricSection3D(
                Scene(),
                surface=_cone(),
                section_id="section",
                plane=SectionPlane(
                    "plane",
                    (0.0, 0.0, 0.2),
                    (0.0, 0.0, 1.0),
                    u_axis=(1.0, 0.0, 0.0),
                ),
                use_plane_patch_envelope=True,
            )

    def test_facade_rejects_double_cone_section_models_before_scene_mutation(
        self,
    ) -> None:
        for model in (ConeModel.OPEN_DOUBLE, ConeModel.ANALYTIC_DOUBLE):
            with self.subTest(model=model.value):
                scene = Scene()
                cone = ConeSpec(
                    f"{model.value}-cone",
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    pi / 4.0,
                    (-2.0, 2.0),
                    radial_axis=(1.0, 0.0, 0.0),
                    model=model,
                )
                with self.assertRaisesRegex(
                    QuadricSectionAuthoringError,
                    model.name,
                ):
                    QuadricSection3D(
                        scene,
                        surface=cone,
                        section_id=f"{model.value}-section",
                        plane=SectionPlane(
                            f"{model.value}-plane",
                            (0.0, 0.0, 0.2),
                            (0.0, 0.0, 1.0),
                            u_axis=(1.0, 0.0, 0.0),
                        ),
                    )
                self.assertEqual(scene.mobjects, [])

    def test_show_plane_false_disables_the_plane_compositor_not_section_ink(
        self,
    ) -> None:
        controller = QuadricSection3D(
            Scene(),
            surface=_cone("curve-only-cone"),
            section_id="curve-only-section",
            plane=_plane(1.5, "curve-only-plane"),
            projection=VIEW,
            show_plane=False,
            limits=_limits(),
            max_chord_error=0.01,
        ).attach()
        try:
            self.assertIsNone(controller.last_section_frame)
            frame = controller.last_boundary_frame
            assert frame is not None
            source_ids = {item.source_id for item in frame.sources}
            self.assertTrue(
                any(item.startswith("curve-only-section:") for item in source_ids)
            )
            self.assertFalse(
                any(
                    item.startswith("boundary:plane:curve-only-plane:edge:")
                    for item in source_ids
                )
            )
        finally:
            controller.restore()

    def test_fixed_topology_callback_rejects_a_conic_family_change_and_rolls_back(
        self,
    ) -> None:
        scheduled = _scheduled()
        motion = scheduled.schedule.motion
        surface = scheduled.schedule.samples[0].surface
        state = {"progress": 0.2}
        controller = QuadricSection3D(
            Scene(),
            surface=surface,
            section_id="fixed-topology-section",
            plane=lambda: motion.plane_at(state["progress"]),
            projection=VIEW,
            limits=_limits(),
            max_chord_error=0.02,
        ).attach()
        snapshot = controller.slot_snapshot()
        identities = controller.slot_identities()
        committed = controller.last_boundary_frame
        state["progress"] = 0.95
        with self.assertRaisesRegex(
            QuadricManimCapacityError,
            "curve identities were not preallocated",
        ):
            controller.update()
        self.assertEqual(controller.slot_snapshot(), snapshot)
        self.assertEqual(controller.slot_identities(), identities)
        self.assertIs(controller.last_boundary_frame, committed)
        controller.restore()

    def test_real_cairo_animation_uses_the_facade_update_path(self) -> None:
        class FacadeScene(Scene):
            def construct(inner_self) -> None:
                height = ValueTracker(0.8)
                cone = _cone("cairo-facade-cone")
                section_id = "cairo-facade-section"

                def plane() -> SectionPlane:
                    return _plane(height.get_value(), "cairo-facade-plane")

                controller = QuadricSection3D(
                    inner_self,
                    surface=cone,
                    section_id=section_id,
                    plane=plane,
                    projection=VIEW,
                    limits=_limits(),
                    max_chord_error=0.03,
                ).attach()
                identities = controller.slot_identities()
                inner_self.play(
                    height.animate.set_value(1.5),
                    run_time=0.5,
                    rate_func=linear,
                )
                chord_id = section_cap_chord_curve_ids(section_id, cone)[0]
                frame = controller.last_boundary_frame
                assert frame is not None
                inner_self.cap_chord_active = chord_id in {
                    item.source_id for item in frame.sources
                }
                inner_self.identity_stable = (
                    identities == controller.slot_identities()
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
                    "frame_rate": 4,
                    "disable_caching": True,
                    "write_to_movie": True,
                    "save_last_frame": False,
                }
            ),
        ):
            scene = FacadeScene()
            scene.render()
            self.assertTrue(
                Path(scene.renderer.file_writer.movie_file_path).is_file()
            )
            self.assertTrue(scene.cap_chord_active)
            self.assertTrue(scene.identity_stable)


if __name__ == "__main__":
    unittest.main()
