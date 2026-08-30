"""Parallel-camera acceptance for exact edge-on open-double sections.

These tests intentionally exercise the complete CompositeQuadricSection3D
binding.  The renderer-neutral child frames, semantic boundary compositor,
fixed-capacity Manim slots, and Cairo display must all agree on the same
AREA/LINE camera state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
import unittest
from unittest.mock import patch

import numpy as np
from manim import Mobject, ThreeDScene, config, tempconfig

from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
    canonical_quadric_boundary_compositing_json,
)
from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
    CompositeQuadricSectionAuthoringError,
)
from polyhedron_visibility.quadrics.composite_section import (
    canonical_composite_quadric_section_compositing_json,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionKind,
)
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import ParallelCameraState


try:
    import cairo as _cairo  # noqa: F401
    from manim.renderer.cairo_renderer import (  # noqa: F401
        CairoRenderer as _CairoRenderer,
    )
except (ImportError, OSError):
    CAIRO_AVAILABLE = False
else:
    CAIRO_AVAILABLE = True


@dataclass(frozen=True, slots=True)
class _CompositeScenario:
    name: str
    surface: ConeSpec
    plane: SectionPlane
    section_id: str
    apex_plane: bool


class _ParallelCameraScene(ThreeDScene):
    def __init__(self) -> None:
        super().__init__(camera_class=MultiProjectionCamera)


def _scenario(name: str) -> _CompositeScenario:
    if name not in {"apex", "offset-hyperbola"}:
        raise AssertionError(f"unknown composite scenario {name!r}")
    offset = 0.0 if name == "apex" else 0.48
    return _CompositeScenario(
        name,
        ConeSpec(
            f"composite-line:{name}:double",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (-2.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_DOUBLE,
        ),
        SectionPlane(
            f"composite-line:{name}:plane",
            (0.0, offset, 0.0),
            (0.0, 1.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        f"composite-line:{name}:section",
        offset == 0.0,
    )


SCENARIOS = tuple(_scenario(name) for name in ("apex", "offset-hyperbola"))
CONFIGURATIONS = tuple(
    (include_surface_boundaries, paint_policy)
    for include_surface_boundaries in (False, True)
    for paint_policy in (
        QuadricPaintPolicy.PHYSICAL,
        QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
    )
)


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=2,
        max_curves=8,
        max_fragments_per_curve=24,
        max_segments_per_fragment=192,
        max_surface_segments=320,
        max_dashes_per_fragment=64,
        max_projected_length=24.0,
        max_total_mobjects=30000,
        max_boundary_sources=32,
    )


def _camera_states(
    scenario: _CompositeScenario,
) -> tuple[ParallelCameraState, ParallelCameraState, ParallelCameraState]:
    plane = scenario.plane
    # Keep the opening AREA view slightly oblique.  For the apex cut, an
    # exactly normal view makes the mathematical generator coincide with an
    # ordinary cone silhouette before the rank-one transition is exercised;
    # that is a separate same-support authoring case.
    initial = ParallelCameraState.relative_to_plane(
        plane,
        inclination_degrees=14.0,
        azimuth_degrees=0.0,
        target=plane.point,
        screen_anchor=(-0.12, 0.09),
        zoom=0.93,
    )
    line = ParallelCameraState.along_plane(
        plane,
        direction=(1.0, 0.0, 0.0),
        target=plane.point,
        screen_anchor=(0.14, -0.09),
        zoom=1.04,
    )
    final = ParallelCameraState.relative_to_plane(
        plane,
        inclination_degrees=23.0,
        azimuth_degrees=0.0,
        target=(0.11, float(plane.point[1]) - 0.08, 0.17),
        screen_anchor=(-0.15, 0.13),
        zoom=0.89,
    )
    return initial, line, final


def _build(
    scene: ThreeDScene,
    scenario: _CompositeScenario,
    projection: object,
    *,
    include_surface_boundaries: bool,
    paint_policy: QuadricPaintPolicy,
    style: QuadricManimStyle | None = None,
) -> CompositeQuadricSection3D:
    return CompositeQuadricSection3D(
        scene,
        surface=scenario.surface,
        section_id=scenario.section_id,
        plane=scenario.plane,
        projection=projection,
        draw_section_boundary=True,
        include_surface_boundaries=include_surface_boundaries,
        paint_policy=paint_policy,
        style=QuadricManimStyle() if style is None else style,
        limits=_limits(),
        max_chord_error=0.035,
        section_max_screen_error=0.16,
        plane_patch_margin=0.16,
    ).attach()


def _scene_ownership(
    controller: CompositeQuadricSection3D,
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
    controller: CompositeQuadricSection3D,
) -> tuple[object, ...]:
    """Compare active contents while ignoring stale invisible fixed slots."""

    prepared = controller._last_prepared_frame
    assert prepared is not None
    result: list[object] = []
    for item_id in prepared.numeric.draw_order:
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


def _assert_no_mobject_or_scene_allocation(
    testcase: unittest.TestCase,
    controller: CompositeQuadricSection3D,
    scene: ThreeDScene,
    update,
    *,
    label: str,
) -> None:
    identities = controller.slot_identities()
    child_identities = controller.child_slot_identities()
    ownership = _scene_ownership(controller)
    scene_mobjects = tuple(id(item) for item in scene.mobjects)
    with (
        patch.object(
            Mobject,
            "__init__",
            side_effect=AssertionError(f"{label} allocated a Mobject"),
        ),
        patch.object(
            scene,
            "add",
            side_effect=AssertionError(f"{label} changed Scene ownership"),
        ),
        patch.object(
            scene,
            "remove",
            side_effect=AssertionError(f"{label} changed Scene ownership"),
        ),
    ):
        update()
    testcase.assertEqual(controller.slot_identities(), identities)
    testcase.assertEqual(controller.child_slot_identities(), child_identities)
    testcase.assertEqual(_scene_ownership(controller), ownership)
    testcase.assertEqual(
        tuple(id(item) for item in scene.mobjects),
        scene_mobjects,
    )


class CompositeQuadricSectionLineManimTests(unittest.TestCase):
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

    def _assert_line_frame(
        self,
        controller: CompositeQuadricSection3D,
        scenario: _CompositeScenario,
    ) -> None:
        frame = controller.last_composite_frame
        boundary = controller.last_boundary_frame
        prepared = controller._last_prepared_frame
        self.assertIsNotNone(frame)
        self.assertIsNotNone(boundary)
        self.assertIsNotNone(prepared)
        assert frame is not None and boundary is not None and prepared is not None

        self.assertIs(frame.projection_kind, PlanePatchProjectionKind.LINE)
        self.assertFalse(frame.has_plane_fill)
        self.assertEqual(frame.plane_fragments, ())
        self.assertTrue(frame.plane_outline_fragments)
        self.assertTrue(
            all(
                child.projection_kind is PlanePatchProjectionKind.LINE
                and not child.has_plane_fill
                and not child.plane_fragments
                for child in frame.child_frames
            )
        )

        self.assertTrue(
            all(not prepared.numeric.plane_polygons[role] for role in PlaneDepthRole)
        )
        for item_id in controller._plane_item_ids.values():
            self.assertEqual(len(controller._plane_slots[item_id].points), 0)

        # The two child compositors must collapse to one shared finite outline
        # bank.  LINE does not retain four semantic rectangle-edge banks.
        self.assertFalse(
            any(
                source.source_kind is BoundarySourceKind.PLANE_PATCH_EDGE
                for source in boundary.sources
            )
        )
        for edge_index in range(4):
            source_id = (
                f"boundary:plane:{scenario.plane.plane_id}:edge:{edge_index}"
            )
            self.assertEqual(
                prepared.numeric.boundary_fragments.get(source_id, ()),
                (),
            )

        outline_paths = tuple(
            path
            for role in PlaneDepthRole
            for path in prepared.numeric.plane_outline_paths[role]
        )
        self.assertTrue(outline_paths)
        self.assertEqual(len(outline_paths), len(frame.plane_outline_fragments))
        line_start = np.asarray(frame.patch_projection.line_screen_start, dtype=float)
        line_end = np.asarray(frame.patch_projection.line_screen_end, dtype=float)
        line_vector = line_end - line_start
        line_length = float(np.linalg.norm(line_vector))
        self.assertGreater(line_length, 1.0e-8)
        axis = line_vector / line_length
        intervals = []
        for path in outline_paths:
            values = np.asarray(path, dtype=float)
            self.assertGreater(_polyline_length(values), 1.0e-8)
            first = float(np.dot(values[0, :2], axis))
            second = float(np.dot(values[-1, :2], axis))
            intervals.append(tuple(sorted((first, second))))
        total_length = sum(upper - lower for lower, upper in intervals)
        union_length = max(upper for _lower, upper in intervals) - min(
            lower for lower, _upper in intervals
        )
        tolerance = max(1.0e-9, line_length * 1.0e-9)
        self.assertAlmostEqual(total_length, line_length, delta=tolerance)
        self.assertAlmostEqual(union_length, line_length, delta=tolerance)

        # Each nappe carries an independent geometry certificate.  In
        # particular, an opposite-nappe curve may not inherit the same-group
        # coincident-support exception.
        groups = boundary.rank_one_section_source_groups
        self.assertEqual(len(groups), 2)
        self.assertIsNone(boundary.rank_one_section_source_group)
        self.assertEqual(
            tuple(group.surface_id for group in groups),
            tuple(
                sorted(
                    child.surface_id
                    for child in scenario.surface.render_components
                )
            ),
        )
        source_by_id = {item.source_id: item for item in boundary.sources}
        group_by_source_id = {
            source_id: group
            for group in groups
            for source_id in group.source_ids
        }
        for group in groups:
            self.assertEqual(group.plane_id, scenario.plane.plane_id)
            expected_source_ids = tuple(
                sorted(
                    source.source_id
                    for source in boundary.sources
                    if source.section_surface_id == group.surface_id
                    and source.section_plane_id == group.plane_id
                    and source.source_kind
                    in {
                        BoundarySourceKind.SECTION_CURVE,
                        BoundarySourceKind.SECTION_CAP_CHORD,
                    }
                )
            )
            self.assertEqual(group.source_ids, expected_source_ids)
            for source_id in group.point_source_ids:
                self.assertFalse(
                    any(
                        fragment.source_id == source_id
                        for fragment in boundary.fragments
                    )
                )
                self.assertEqual(
                    prepared.numeric.boundary_fragments.get(source_id, ()),
                    (),
                )

        if scenario.apex_plane:
            cross_nappe = tuple(
                crossing
                for crossing in boundary.crossings
                if crossing.first_curve_id in group_by_source_id
                and crossing.second_curve_id in group_by_source_id
                and group_by_source_id[crossing.first_curve_id].surface_id
                != group_by_source_id[crossing.second_curve_id].surface_id
            )
            self.assertTrue(
                cross_nappe,
                "the shared-apex crossing between nappes was silently skipped",
            )

        for fragment in boundary.fragments:
            self.assertGreater(fragment.interval.length, 1.0e-12)
            self.assertIn(fragment.source_id, source_by_id)
        for fragments in prepared.numeric.boundary_fragments.values():
            for fragment in fragments:
                self.assertGreater(_polyline_length(fragment.points), 1.0e-8)
                for dash in fragment.dashes:
                    self.assertGreater(_polyline_length(dash.points), 1.0e-8)

    def test_area_line_area_keeps_fixed_slots_and_matches_cold_frames(self) -> None:
        for scenario in SCENARIOS:
            for include_surface_boundaries, paint_policy in CONFIGURATIONS:
                with self.subTest(
                    scenario=scenario.name,
                    include_surface_boundaries=include_surface_boundaries,
                    paint_policy=paint_policy.value,
                ):
                    initial, line, final = _camera_states(scenario)
                    state: dict[str, object] = {"projection": initial}
                    scene = _ParallelCameraScene()
                    warm: CompositeQuadricSection3D | None = None
                    cold_line: CompositeQuadricSection3D | None = None
                    cold_area: CompositeQuadricSection3D | None = None
                    try:
                        warm = _build(
                            scene,
                            scenario,
                            lambda _scene: state["projection"],
                            include_surface_boundaries=include_surface_boundaries,
                            paint_policy=paint_policy,
                        )
                        initial_frame = warm.last_composite_frame
                        self.assertIsNotNone(initial_frame)
                        assert initial_frame is not None
                        self.assertIs(
                            initial_frame.projection_kind,
                            PlanePatchProjectionKind.AREA,
                        )
                        self.assertTrue(initial_frame.has_plane_fill)

                        state["projection"] = line
                        _assert_no_mobject_or_scene_allocation(
                            self,
                            warm,
                            scene,
                            warm.update,
                            label="Composite AREA-to-LINE update",
                        )
                        self._assert_line_frame(warm, scenario)
                        warm_line_frame = warm.last_composite_frame
                        warm_line_boundary = warm.last_boundary_frame
                        assert (
                            warm_line_frame is not None
                            and warm_line_boundary is not None
                        )
                        warm_line_display = _active_display_snapshot(warm)

                        cold_line = _build(
                            _ParallelCameraScene(),
                            scenario,
                            line,
                            include_surface_boundaries=include_surface_boundaries,
                            paint_policy=paint_policy,
                        )
                        self._assert_line_frame(cold_line, scenario)
                        cold_line_frame = cold_line.last_composite_frame
                        cold_line_boundary = cold_line.last_boundary_frame
                        assert (
                            cold_line_frame is not None
                            and cold_line_boundary is not None
                        )
                        self.assertEqual(
                            canonical_composite_quadric_section_compositing_json(
                                warm_line_frame
                            ),
                            canonical_composite_quadric_section_compositing_json(
                                cold_line_frame
                            ),
                        )
                        self.assertEqual(
                            canonical_quadric_boundary_compositing_json(
                                warm_line_boundary
                            ),
                            canonical_quadric_boundary_compositing_json(
                                cold_line_boundary
                            ),
                        )
                        self.assertEqual(
                            warm_line_display,
                            _active_display_snapshot(cold_line),
                        )

                        state["projection"] = final
                        _assert_no_mobject_or_scene_allocation(
                            self,
                            warm,
                            scene,
                            warm.update,
                            label="Composite LINE-to-AREA update",
                        )
                        returned_area = warm.last_composite_frame
                        returned_boundary = warm.last_boundary_frame
                        self.assertIsNotNone(returned_area)
                        self.assertIsNotNone(returned_boundary)
                        assert (
                            returned_area is not None
                            and returned_boundary is not None
                        )
                        self.assertIs(
                            returned_area.projection_kind,
                            PlanePatchProjectionKind.AREA,
                        )
                        self.assertTrue(returned_area.has_plane_fill)

                        cold_area = _build(
                            _ParallelCameraScene(),
                            scenario,
                            final,
                            include_surface_boundaries=include_surface_boundaries,
                            paint_policy=paint_policy,
                        )
                        cold_area_frame = cold_area.last_composite_frame
                        cold_area_boundary = cold_area.last_boundary_frame
                        assert (
                            cold_area_frame is not None
                            and cold_area_boundary is not None
                        )
                        self.assertEqual(
                            canonical_composite_quadric_section_compositing_json(
                                returned_area
                            ),
                            canonical_composite_quadric_section_compositing_json(
                                cold_area_frame
                            ),
                        )
                        self.assertEqual(
                            canonical_quadric_boundary_compositing_json(
                                returned_boundary
                            ),
                            canonical_quadric_boundary_compositing_json(
                                cold_area_boundary
                            ),
                        )
                        self.assertEqual(
                            _active_display_snapshot(warm),
                            _active_display_snapshot(cold_area),
                        )
                    finally:
                        for controller in (cold_area, cold_line, warm):
                            if controller is not None:
                                controller.restore()

    def test_positive_area_nappe_overlap_rolls_back_then_line_recovers(self) -> None:
        scenario = _scenario("offset-hyperbola")
        initial, line, _final = _camera_states(scenario)
        state: dict[str, object] = {"projection": initial}
        scene = _ParallelCameraScene()
        controller: CompositeQuadricSection3D | None = None
        try:
            controller = _build(
                scene,
                scenario,
                lambda _scene: state["projection"],
                include_surface_boundaries=True,
                paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            )
            state["projection"] = line
            controller.update()
            self._assert_line_frame(controller, scenario)

            snapshot = controller.slot_snapshot()
            identities = controller.slot_identities()
            child_identities = controller.child_slot_identities()
            ownership = _scene_ownership(controller)
            scene_mobjects = tuple(id(item) for item in scene.mobjects)
            frame = controller.last_composite_frame
            boundary = controller.last_boundary_frame
            prepared = controller._last_prepared_frame
            z_indices = controller.active_painter_z_indices

            # Looking along the cone axis keeps the cutting plane rank one but
            # projects the two finite nappes over the same positive-area disk.
            state["projection"] = ParallelCameraState.along_plane(
                scenario.plane,
                direction=(0.0, 0.0, 1.0),
                target=scenario.plane.point,
                screen_anchor=(-0.08, 0.05),
                zoom=1.01,
            )
            with self.assertRaisesRegex(
                CompositeQuadricSectionAuthoringError,
                "positive-area overlap|contact is two-dimensional",
            ):
                controller.update()

            self.assertEqual(controller.slot_snapshot(), snapshot)
            self.assertEqual(controller.slot_identities(), identities)
            self.assertEqual(controller.child_slot_identities(), child_identities)
            self.assertEqual(_scene_ownership(controller), ownership)
            self.assertEqual(
                tuple(id(item) for item in scene.mobjects),
                scene_mobjects,
            )
            self.assertIs(controller.last_composite_frame, frame)
            self.assertIs(controller.last_boundary_frame, boundary)
            self.assertIs(controller._last_prepared_frame, prepared)
            self.assertEqual(controller.active_painter_z_indices, z_indices)

            recovered = line.with_screen_anchor((0.22, 0.04)).with_zoom(0.97)
            state["projection"] = recovered
            _assert_no_mobject_or_scene_allocation(
                self,
                controller,
                scene,
                controller.update,
                label="Composite rollback recovery LINE update",
            )
            self._assert_line_frame(controller, scenario)
            self.assertIsNot(controller.last_composite_frame, frame)
        finally:
            if controller is not None:
                controller.restore()


def _pixel_for_screen(point: tuple[float, float]) -> tuple[int, int]:
    x, y = point
    column = int(
        round(
            (x + config.frame_width / 2.0)
            / config.frame_width
            * (config.pixel_width - 1)
        )
    )
    row = int(
        round(
            (config.frame_height / 2.0 - y)
            / config.frame_height
            * (config.pixel_height - 1)
        )
    )
    return row, column


def _capture_pixels(scene: ThreeDScene) -> np.ndarray:
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    return scene.camera.pixel_array[:, :, :3].copy()


@unittest.skipUnless(CAIRO_AVAILABLE, "Cairo is required for pixel evidence")
class CompositeQuadricSectionLineCairoTests(unittest.TestCase):
    def test_line_frame_has_both_nappes_apex_finite_outline_and_no_double_draw(
        self,
    ) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 6,
            }
        ):
            scenario = _scenario("offset-hyperbola")
            initial, line, _final = _camera_states(scenario)
            line = line.with_screen_anchor((0.0, 0.0)).with_zoom(1.0)
            style = QuadricManimStyle(
                surface_fill_color="#245A7A",
                surface_fill_opacity=0.54,
                surface_stroke_color="#61DDF2",
                surface_stroke_width=3.0,
                surface_stroke_opacity=0.94,
                visible_curve_color="#FFD866",
                visible_curve_width=4.0,
                visible_curve_opacity=1.0,
                hidden_curve_color="#FFD866",
                hidden_curve_width=3.0,
                hidden_curve_opacity=0.48,
                section_plane_fill_color="#2CB9A4",
                section_plane_fill_opacity=0.2,
                section_plane_stroke_color="#F05BC8",
                section_plane_stroke_width=3.2,
                section_plane_stroke_opacity=1.0,
                cone_lateral_fill_colors=("#173753", "#4F9AC1", "#1D4368"),
            )
            warm_scene = _ParallelCameraScene()
            cold_scene = _ParallelCameraScene()
            background = np.asarray((11, 23, 35), dtype=np.uint8)
            warm_scene.camera.background_color = "#0B1723"
            cold_scene.camera.background_color = "#0B1723"
            state: dict[str, object] = {"projection": initial}
            warm: CompositeQuadricSection3D | None = None
            cold: CompositeQuadricSection3D | None = None
            try:
                warm = _build(
                    warm_scene,
                    scenario,
                    lambda _scene: state["projection"],
                    include_surface_boundaries=True,
                    paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                    style=style,
                )
                state["projection"] = line
                warm.update()
                warm_pixels = _capture_pixels(warm_scene)

                cold = _build(
                    cold_scene,
                    scenario,
                    line,
                    include_surface_boundaries=True,
                    paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                    style=style,
                )
                cold_pixels = _capture_pixels(cold_scene)
                self.assertTrue(
                    np.array_equal(warm_pixels, cold_pixels),
                    "warm LINE retained a duplicate or stale Cairo stroke",
                )

                rgb = warm_pixels.astype(int)
                yellow = (
                    (rgb[:, :, 0] > 180)
                    & (rgb[:, :, 1] > 130)
                    & (rgb[:, :, 2] < 150)
                )
                magenta = (
                    (rgb[:, :, 0] > 70)
                    & (rgb[:, :, 2] > 70)
                    & (
                        10 * rgb[:, :, 1]
                        < 7
                        * np.minimum(
                            rgb[:, :, 0],
                            rgb[:, :, 2],
                        )
                    )
                    & (
                        np.abs(
                            rgb[:, :, 0]
                            - rgb[:, :, 2]
                        )
                        < 40
                    )
                )
                midpoint = warm_pixels.shape[0] // 2
                self.assertGreater(int(np.count_nonzero(yellow[:midpoint])), 20)
                self.assertGreater(int(np.count_nonzero(yellow[midpoint:])), 20)
                self.assertGreater(int(np.count_nonzero(magenta)), 45)

                apex = line.project_point(scenario.surface.apex)[:2]
                apex_row, apex_column = _pixel_for_screen(
                    (float(apex[0]), float(apex[1]))
                )
                apex_patch = warm_pixels[
                    max(0, apex_row - 3) : apex_row + 4,
                    max(0, apex_column - 3) : apex_column + 4,
                ].astype(float)
                self.assertGreater(
                    float(
                        np.max(
                            np.linalg.norm(
                                apex_patch - background.astype(float),
                                axis=2,
                            )
                        )
                    ),
                    20.0,
                )

                frame = warm.last_composite_frame
                prepared = warm._last_prepared_frame
                assert frame is not None
                assert prepared is not None
                start = np.asarray(
                    frame.patch_projection.line_screen_start,
                    dtype=float,
                )
                end = np.asarray(frame.patch_projection.line_screen_end, dtype=float)
                direction = end - start
                direction /= np.linalg.norm(direction)
                displayed_endpoints = tuple(
                    np.asarray(point, dtype=float)[:2]
                    for role in PlaneDepthRole
                    for path in prepared.numeric.plane_outline_paths[role]
                    for point in (path[0], path[-1])
                )
                outside = max(
                    displayed_endpoints,
                    key=lambda point: float(np.dot(point, direction)),
                ) + 0.45 * direction
                outside_row, outside_column = _pixel_for_screen(
                    (float(outside[0]), float(outside[1]))
                )
                actual_background = warm_pixels[outside_row, outside_column]
                self.assertLessEqual(
                    int(
                        np.max(
                            np.abs(
                                actual_background.astype(int)
                                - background.astype(int)
                            )
                        )
                    ),
                    1,
                )
            finally:
                for controller in (cold, warm):
                    if controller is not None:
                        controller.restore()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
