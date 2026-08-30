from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
import unittest
from unittest.mock import patch

import numpy as np
from manim import Mobject, ThreeDScene, tempconfig

from polyhedron_visibility.quadrics.authoring import QuadricSection3D
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
    canonical_quadric_boundary_compositing_json,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.conics import ConicKind
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionKind,
    canonical_quadric_section_compositing_json,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary,
    section_cap_chord_curve_ids,
)
from polyhedron_visibility.visibility import VisibilityKind
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_shots import ParallelCameraShot
from tikz_native.parallel_shots_manim import play_parallel_camera_shot


@dataclass(frozen=True, slots=True)
class _ConicScenario:
    name: str
    kind: ConicKind
    surface: ConeSpec
    plane: SectionPlane
    section_id: str


class _ParallelCameraScene(ThreeDScene):
    def __init__(self) -> None:
        super().__init__(camera_class=MultiProjectionCamera)


def _scenario(name: str) -> _ConicScenario:
    surface = ConeSpec(
        f"rank-one:{name}:cone",
        (0.0, 0.0, -1.5),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.CLOSED_SINGLE,
    )
    plane_id = f"rank-one:{name}:plane"
    if name == "circle":
        kind = ConicKind.CIRCLE
        plane = SectionPlane(
            plane_id,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
    elif name == "ellipse":
        kind = ConicKind.ELLIPSE
        plane = SectionPlane(
            plane_id,
            (0.0, 0.0, 0.0),
            (0.35, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
    elif name == "parabola":
        kind = ConicKind.PARABOLA
        plane = SectionPlane(
            plane_id,
            (0.0, 0.0, 0.0),
            (cos(pi / 6.0), 0.0, -sin(pi / 6.0)),
            u_axis=(0.0, 1.0, 0.0),
        )
    elif name == "hyperbola":
        kind = ConicKind.HYPERBOLA
        plane = SectionPlane(
            plane_id,
            (0.4, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            u_axis=(0.0, 1.0, 0.0),
        )
    else:  # pragma: no cover - the test table below is closed.
        raise AssertionError(f"unknown conic scenario {name!r}")
    return _ConicScenario(
        name,
        kind,
        surface,
        plane,
        f"rank-one:{name}:section",
    )


SCENARIOS = tuple(
    _scenario(name) for name in ("circle", "ellipse", "parabola", "hyperbola")
)


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=1,
        max_curves=8,
        max_fragments_per_curve=24,
        max_segments_per_fragment=160,
        max_surface_segments=320,
        max_dashes_per_fragment=64,
        max_projected_length=20.0,
        max_total_mobjects=20000,
        max_boundary_sources=32,
    )


def _playback_limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=1,
        max_curves=8,
        max_fragments_per_curve=12,
        max_segments_per_fragment=96,
        max_surface_segments=192,
        max_dashes_per_fragment=64,
        max_projected_length=20.0,
        max_total_mobjects=8000,
        max_boundary_sources=32,
    )


def _camera_states(
    scenario: _ConicScenario,
) -> tuple[ParallelCameraState, ParallelCameraState, ParallelCameraState]:
    plane = scenario.plane
    initial = ParallelCameraState.relative_to_plane(
        plane,
        inclination_degrees=18.0,
        azimuth_degrees=37.0,
        target=plane.point,
        screen_anchor=(-0.18, 0.12),
        zoom=0.92,
    )
    line = ParallelCameraState.along_plane(
        plane,
        direction=(0.0, -1.0, 0.0),
        target=plane.point,
        screen_anchor=(0.16, -0.11),
        zoom=1.06,
    )
    final = ParallelCameraState.relative_to_plane(
        plane,
        inclination_degrees=31.0,
        azimuth_degrees=23.0,
        target=(0.08, -0.12, 0.18),
        screen_anchor=(-0.14, 0.17),
        zoom=0.88,
    )
    return initial, line, final


def _build(
    scene: ThreeDScene,
    scenario: _ConicScenario,
    projection: object,
    *,
    include_surface_boundaries: bool,
    paint_policy: QuadricPaintPolicy,
    limits: QuadricManimLimits | None = None,
) -> QuadricSection3D:
    return QuadricSection3D(
        scene,
        surface=scenario.surface,
        section_id=scenario.section_id,
        plane=scenario.plane,
        projection=projection,
        draw_section_boundary=True,
        include_surface_boundaries=include_surface_boundaries,
        paint_policy=paint_policy,
        limits=_limits() if limits is None else limits,
        max_chord_error=0.025,
        section_max_screen_error=0.16,
    ).attach()


def _scene_ownership(
    controller: QuadricOcclusion3D,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(id(item) for item in container)
        for container in controller._scene_containers()
    )


def _rgba_tuple(value: object, name: str) -> tuple[float, ...]:
    return tuple(
        float(item)
        for item in np.round(
            np.asarray(getattr(value, name, np.empty((0, 4))), dtype=float),
            12,
        ).reshape(-1)
    )


def _active_display_snapshot(
    controller: QuadricOcclusion3D,
) -> tuple[object, ...]:
    """Compare only active fixed-slot contents, not stale invisible buffers."""

    prepared = controller._last_prepared_frame
    assert prepared is not None
    result: list[object] = []
    for item_id in prepared.numeric.painter_draw_order:
        root = prepared.numeric.item_mobjects[item_id]
        members: list[object] = []
        for index, member in enumerate(root.get_family()):
            points = np.asarray(
                getattr(member, "points", np.empty((0, 3))),
                dtype=float,
            )
            fill = _rgba_tuple(member, "fill_rgbas")
            stroke = _rgba_tuple(member, "stroke_rgbas")
            background = _rgba_tuple(member, "background_stroke_rgbas")
            own_alpha = max(
                (*fill[3::4], *stroke[3::4], *background[3::4], 0.0)
            )
            if index == 0 or (len(points) > 0 and own_alpha > 0.0):
                members.append(
                    (
                        tuple(
                            float(item)
                            for item in np.round(points, 12).reshape(-1)
                        ),
                        fill,
                        stroke,
                        background,
                        float(getattr(member, "z_index", 0.0)),
                    )
                )
        result.append((item_id, tuple(members)))
    return tuple(result)


def _polyline_length(points: object) -> float:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)))


class QuadricRankOneConicCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "frame_rate": 4,
                "pixel_width": 160,
                "pixel_height": 90,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
                "progress_bar": "none",
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def _assert_line_frame(
        self,
        facade: QuadricSection3D,
        scenario: _ConicScenario,
        *,
        paint_policy: QuadricPaintPolicy,
    ) -> None:
        section = facade.last_section_frame
        boundary = facade.last_boundary_frame
        prepared = facade.controller._last_prepared_frame
        self.assertIsNotNone(section)
        self.assertIsNotNone(boundary)
        self.assertIsNotNone(prepared)
        assert section is not None and boundary is not None and prepared is not None

        self.assertIs(section.projection_kind, PlanePatchProjectionKind.LINE)
        self.assertFalse(section.has_plane_fill)
        self.assertEqual(section.plane_fragments, ())
        layers = prepared.numeric.section_layers
        self.assertIsNotNone(layers)
        assert layers is not None
        self.assertTrue(
            all(not layers.plane_polygons[role] for role in PlaneDepthRole)
        )

        line_start = np.asarray(section.patch_projection.line_screen_start, dtype=float)
        line_end = np.asarray(section.patch_projection.line_screen_end, dtype=float)
        line_length = float(np.linalg.norm(line_end - line_start))
        self.assertGreater(line_length, 1.0e-8)
        self.assertTrue(section.plane_outline_fragments)
        for fragment in section.plane_outline_fragments:
            self.assertGreater(
                float(
                    np.linalg.norm(
                        np.asarray(fragment.screen_end, dtype=float)
                        - np.asarray(fragment.screen_start, dtype=float)
                    )
                ),
                1.0e-8,
            )

        # A committed LINE frame may omit a source which becomes a certified
        # screen point, but it must never emit a tiny surrogate stroke for it.
        for fragment in boundary.fragments:
            self.assertGreater(fragment.interval.length, 1.0e-12)
        for fragments in (prepared.numeric.boundary_fragments or {}).values():
            for fragment in fragments:
                self.assertGreater(_polyline_length(fragment.points), 1.0e-8)
                for dash in fragment.dashes:
                    self.assertGreater(_polyline_length(dash.points), 1.0e-8)

        source_by_id = {item.source_id: item for item in boundary.sources}
        for crossing in boundary.crossings:
            first = source_by_id[crossing.first_curve_id]
            second = source_by_id[crossing.second_curve_id]
            for section_source, owner_source in (
                (first, second),
                (second, first),
            ):
                is_section_source = (
                    section_source.source_kind
                    in {
                        BoundarySourceKind.SECTION_CURVE,
                        BoundarySourceKind.SECTION_CAP_CHORD,
                    }
                    and section_source.section_surface_id
                    == scenario.surface.surface_id
                    and section_source.section_plane_id
                    == scenario.plane.plane_id
                )
                is_owner_source = (
                    owner_source.owner_surface_id == scenario.surface.surface_id
                )
                self.assertFalse(
                    is_section_source and is_owner_source,
                    msg=(
                        "rank-one section/owner pair leaked into projected "
                        f"crossings: {section_source.source_id!r}, "
                        f"{owner_source.source_id!r}"
                    ),
                )

        section_fragments = tuple(
            item
            for item in boundary.fragments
            if source_by_id[item.source_id].source_kind
            in {
                BoundarySourceKind.SECTION_CURVE,
                BoundarySourceKind.SECTION_CAP_CHORD,
            }
        )
        hidden = tuple(
            item
            for item in section_fragments
            if item.effective_visibility_kind is VisibilityKind.HIDDEN
        )
        visible = tuple(
            item
            for item in section_fragments
            if item.effective_visibility_kind is VisibilityKind.VISIBLE
        )
        self.assertTrue(hidden)
        self.assertTrue(visible)
        positions = {item_id: index for index, item_id in enumerate(boundary.draw_order)}
        if paint_policy is QuadricPaintPolicy.DIAGRAMMATIC:
            self.assertTrue(all(item.painted for item in hidden))
            self.assertTrue(all(item.painted for item in visible))
            self.assertLess(
                max(positions[item.item_id] for item in hidden),
                min(positions[item.item_id] for item in visible),
            )
        else:
            self.assertTrue(all(not item.painted for item in hidden))
            self.assertTrue(all(item.painted for item in visible))
            self.assertTrue(
                all(item.item_id not in positions for item in hidden)
            )

        cap_ids = section_cap_chord_curve_ids(
            scenario.section_id,
            scenario.surface,
        )
        self.assertTrue(
            set(cap_ids).issubset(facade.allocated_curve_ids)
        )
        self.assertTrue(
            set(cap_ids).issubset(facade.controller._curve_slots)
        )
        if scenario.kind in {ConicKind.PARABOLA, ConicKind.HYPERBOLA}:
            self.assertEqual(len(cap_ids), 1)
            cap_id = cap_ids[0]
            cap_source = source_by_id.get(cap_id)
            self.assertIsNotNone(cap_source)
            assert cap_source is not None
            self.assertIs(
                cap_source.source_kind,
                BoundarySourceKind.SECTION_CAP_CHORD,
            )
            self.assertEqual(
                tuple(
                    item for item in boundary.fragments if item.source_id == cap_id
                ),
                (),
            )
            self.assertEqual(
                (prepared.numeric.boundary_fragments or {}).get(cap_id, ()),
                (),
            )
            self.assertFalse(
                any(
                    cap_id in {item.first_curve_id, item.second_curve_id}
                    for item in boundary.crossings
                )
            )

    def test_four_conics_commit_area_line_area_with_fixed_slots(self) -> None:
        for scenario_index, scenario in enumerate(SCENARIOS):
            exact = compute_quadric_section_boundary(
                scenario.section_id,
                scenario.surface,
                scenario.plane,
            )
            self.assertIs(exact.trace.supporting_kind, scenario.kind)
            configurations = (
                (False, QuadricPaintPolicy.DIAGRAMMATIC),
                (True, QuadricPaintPolicy.PHYSICAL),
            )
            if scenario_index % 2:
                configurations = tuple(reversed(configurations))
                configurations = tuple(
                    (include, (
                        QuadricPaintPolicy.DIAGRAMMATIC
                        if policy is QuadricPaintPolicy.PHYSICAL
                        else QuadricPaintPolicy.PHYSICAL
                    ))
                    for include, policy in configurations
                )
            for include_surface_boundaries, paint_policy in configurations:
                with self.subTest(
                    family=scenario.name,
                    include_surface_boundaries=include_surface_boundaries,
                    paint_policy=paint_policy.value,
                ):
                    initial, line, final = _camera_states(scenario)
                    state: dict[str, object] = {"projection": initial}
                    scene = _ParallelCameraScene()
                    facade: QuadricSection3D | None = None
                    cold: QuadricSection3D | None = None
                    try:
                        facade = _build(
                            scene,
                            scenario,
                            lambda _scene: state["projection"],
                            include_surface_boundaries=include_surface_boundaries,
                            paint_policy=paint_policy,
                        )
                        initial_frame = facade.last_section_frame
                        self.assertIsNotNone(initial_frame)
                        assert initial_frame is not None
                        self.assertIs(
                            initial_frame.projection_kind,
                            PlanePatchProjectionKind.AREA,
                        )

                        identities = facade.slot_identities()
                        ownership = _scene_ownership(facade.controller)
                        scene_mobjects = tuple(id(item) for item in scene.mobjects)
                        state["projection"] = line
                        with (
                            patch.object(
                                Mobject,
                                "__init__",
                                side_effect=AssertionError(
                                    "AREA-to-LINE update allocated a Mobject"
                                ),
                            ),
                            patch.object(
                                scene,
                                "add",
                                side_effect=AssertionError(
                                    "AREA-to-LINE update changed Scene ownership"
                                ),
                            ),
                            patch.object(
                                scene,
                                "remove",
                                side_effect=AssertionError(
                                    "AREA-to-LINE update changed Scene ownership"
                                ),
                            ),
                        ):
                            facade.update()
                        self.assertEqual(facade.slot_identities(), identities)
                        self.assertEqual(
                            _scene_ownership(facade.controller), ownership
                        )
                        self.assertEqual(
                            tuple(id(item) for item in scene.mobjects),
                            scene_mobjects,
                        )
                        self._assert_line_frame(
                            facade,
                            scenario,
                            paint_policy=paint_policy,
                        )

                        state["projection"] = final
                        with (
                            patch.object(
                                Mobject,
                                "__init__",
                                side_effect=AssertionError(
                                    "LINE-to-AREA update allocated a Mobject"
                                ),
                            ),
                            patch.object(
                                scene,
                                "add",
                                side_effect=AssertionError(
                                    "LINE-to-AREA update changed Scene ownership"
                                ),
                            ),
                            patch.object(
                                scene,
                                "remove",
                                side_effect=AssertionError(
                                    "LINE-to-AREA update changed Scene ownership"
                                ),
                            ),
                        ):
                            facade.update()
                        final_section = facade.last_section_frame
                        final_boundary = facade.last_boundary_frame
                        self.assertIsNotNone(final_section)
                        self.assertIsNotNone(final_boundary)
                        assert final_section is not None and final_boundary is not None
                        self.assertIs(
                            final_section.projection_kind,
                            PlanePatchProjectionKind.AREA,
                        )
                        self.assertEqual(facade.slot_identities(), identities)
                        self.assertEqual(
                            _scene_ownership(facade.controller), ownership
                        )

                        cold = _build(
                            _ParallelCameraScene(),
                            scenario,
                            final,
                            include_surface_boundaries=include_surface_boundaries,
                            paint_policy=paint_policy,
                        )
                        cold_section = cold.last_section_frame
                        cold_boundary = cold.last_boundary_frame
                        assert cold_section is not None and cold_boundary is not None
                        self.assertEqual(
                            canonical_quadric_section_compositing_json(
                                final_section
                            ),
                            canonical_quadric_section_compositing_json(
                                cold_section
                            ),
                        )
                        self.assertEqual(
                            canonical_quadric_boundary_compositing_json(
                                final_boundary
                            ),
                            canonical_quadric_boundary_compositing_json(
                                cold_boundary
                            ),
                        )
                        self.assertEqual(
                            _active_display_snapshot(facade.controller),
                            _active_display_snapshot(cold.controller),
                        )
                    finally:
                        if cold is not None:
                            cold.restore()
                        if facade is not None:
                            facade.restore()

    def test_real_camera_shots_keep_rank_one_slots_and_scene_ownership(self) -> None:
        scenario = _scenario("ellipse")
        initial, line, final = _camera_states(scenario)
        scene = _ParallelCameraScene()
        camera = scene.camera
        self.assertIsInstance(camera, MultiProjectionCamera)
        camera.set_parallel_state(initial)
        facade: QuadricSection3D | None = None
        controller: QuadricOcclusion3D | None = None
        owned_ids: set[int] = set()
        fixed_ids: set[int] = set()
        try:
            facade = _build(
                scene,
                scenario,
                lambda active_scene: active_scene.camera.snapshot_parallel_state(),
                include_surface_boundaries=False,
                paint_policy=QuadricPaintPolicy.DIAGRAMMATIC,
                limits=_playback_limits(),
            )
            controller = facade.controller
            identities = facade.slot_identities()
            root_family = tuple(controller.root.get_family())
            root_family_ids = {id(item) for item in root_family}
            driver_family_ids = {
                id(item) for item in controller._update_driver.get_family()
            }
            owned_ids = root_family_ids | driver_family_ids
            update_count = 0

            def assert_owned_state() -> None:
                self.assertEqual(facade.slot_identities(), identities)
                self.assertEqual(scene.mobjects.count(controller.root), 1)
                self.assertEqual(
                    scene.mobjects.count(controller._update_driver),
                    1,
                )
                top_level_ids = {id(item) for item in scene.mobjects}
                self.assertEqual(
                    top_level_ids & root_family_ids,
                    {id(controller.root)},
                )
                current_fixed = {
                    id(item) for item in camera.fixed_in_frame_mobjects
                }
                self.assertTrue(root_family_ids.issubset(current_fixed))

            original_update = controller.update

            def audited_update(dt: float = 0.0) -> QuadricOcclusion3D:
                nonlocal update_count
                result = original_update(dt)
                update_count += 1
                assert_owned_state()
                return result

            line_shot = ParallelCameraShot(
                "rank-one-exact-side-view",
                line,
                duration=0.5,
                transition="orbit",
                arc_height=0.65,
            )
            area_shot = ParallelCameraShot(
                "rank-one-return-area-view",
                final,
                duration=0.5,
                transition="orbit",
                arc_height=0.65,
            )
            with patch.object(controller, "update", side_effect=audited_update):
                first_endpoint = play_parallel_camera_shot(scene, line_shot)
                facade.update()
                self.assertIs(first_endpoint, line)
                self._assert_line_frame(
                    facade,
                    scenario,
                    paint_policy=QuadricPaintPolicy.DIAGRAMMATIC,
                )
                assert_owned_state()

                second_endpoint = play_parallel_camera_shot(scene, area_shot)
                facade.update()
                self.assertIs(second_endpoint, final)
                final_section = facade.last_section_frame
                self.assertIsNotNone(final_section)
                assert final_section is not None
                self.assertIs(
                    final_section.projection_kind,
                    PlanePatchProjectionKind.AREA,
                )
                assert_owned_state()

            self.assertGreaterEqual(update_count, 6)
            final_state = camera.snapshot_parallel_state()
            np.testing.assert_array_equal(final_state.matrix, final.matrix)
            np.testing.assert_array_equal(final_state.target, final.target)
            np.testing.assert_array_equal(
                final_state.screen_anchor,
                final.screen_anchor,
            )
            self.assertEqual(final_state.zoom, final.zoom)
            fixed_ids = {id(item) for item in camera.fixed_in_frame_mobjects}
            self.assertTrue(root_family_ids.issubset(fixed_ids))
        finally:
            if facade is not None:
                facade.restore()
            if controller is not None:
                self.assertTrue(
                    all(
                        id(item) not in owned_ids
                        for container in controller._scene_containers()
                        for item in container
                    )
                )
                remaining_fixed_ids = {
                    id(item) for item in camera.fixed_in_frame_mobjects
                }
                self.assertTrue(fixed_ids.isdisjoint(remaining_fixed_ids))


if __name__ == "__main__":
    unittest.main()
