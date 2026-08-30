from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from manim import Mobject, ThreeDScene, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_compositing import (
    canonical_quadric_boundary_compositing_json,
)
from polyhedron_visibility.quadrics.contract import (
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimError,
    QuadricManimLimits,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionKind,
    QuadricSectionCompositingError,
    canonical_quadric_section_compositing_json,
)
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import ParallelCameraState


PLANE = SectionPlane(
    "finite-cut",
    (0.0, 0.0, 0.3),
    (0.0, 0.0, 1.0),
    u_axis=(1.0, 0.0, 0.0),
)
PATCH = PlaneDisplayPatchSpec(
    "finite-cut-patch",
    PLANE.plane_id,
    0.7,
    0.7,
)
SURFACE = SphereSpec("finite-cut-sphere", (0.0, 0.0, 0.0), 1.0)

INITIAL_AREA_STATE = ParallelCameraState.normal_to_plane(PLANE)
LINE_STATE = ParallelCameraState.along_plane(
    PLANE,
    direction=(0.0, -1.0, 0.0),
    target=(0.1, -0.2, 0.3),
    screen_anchor=(0.25, -0.15),
    zoom=1.1,
)
FINAL_AREA_STATE = ParallelCameraState.relative_to_plane(
    PLANE,
    inclination_degrees=32.0,
    azimuth_degrees=24.0,
    target=(-0.15, 0.1, 0.25),
    screen_anchor=(-0.2, 0.18),
    zoom=0.95,
)


class _ParallelCameraScene(ThreeDScene):
    def __init__(self) -> None:
        super().__init__(camera_class=MultiProjectionCamera)


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=1,
        max_curves=1,
        max_fragments_per_curve=8,
        max_segments_per_fragment=128,
        max_surface_segments=128,
        max_dashes_per_fragment=32,
        max_projected_length=16.0,
        max_total_mobjects=5000,
        max_boundary_sources=16,
    )


def _legacy_view(state: ParallelCameraState) -> ParallelView:
    matrix = np.array(state.matrix, dtype=float, copy=True)
    matrix[:2] *= state.zoom
    return ParallelView.from_matrix(matrix)


def _projection_value(
    boundary_mode: str,
    state: ParallelCameraState,
) -> ParallelView | ParallelCameraState:
    if boundary_mode == "legacy":
        return _legacy_view(state)
    return state


def _controller(
    scene: ThreeDScene,
    projection: object,
    *,
    boundary_mode: str,
) -> QuadricOcclusion3D:
    return QuadricOcclusion3D(
        scene,
        surfaces=(SURFACE,),
        curves=(),
        projection=projection,
        section_plane=PLANE,
        section_patch=PATCH,
        section_max_screen_error=0.2,
        boundary_visibility_mode=boundary_mode,
        include_surface_boundaries=False,
        limits=_limits(),
    )


def _scene_ownership(controller: QuadricOcclusion3D) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(id(item) for item in container)
        for container in controller._scene_containers()
    )


def _committed_evidence(controller: QuadricOcclusion3D) -> dict[str, object]:
    return {
        "slot_identities": controller.slot_identities(),
        "slot_snapshot": controller.slot_snapshot(),
        "scene_ownership": _scene_ownership(controller),
        "active_z": controller.active_painter_z_indices,
        "last_frame": controller.last_frame,
        "last_global_frame": controller.last_global_frame,
        "last_section_frame": controller.last_section_frame,
        "last_boundary_frame": controller.last_boundary_frame,
        "last_prepared_frame": controller._last_prepared_frame,
        "geometry_signature": controller._last_input_geometry_signature,
        "draw_signature": controller._last_input_draw_signature,
    }


def _rgba_tuple(value: object, name: str) -> tuple[float, ...]:
    return tuple(
        float(item)
        for item in np.round(
            np.asarray(getattr(value, name, np.empty((0, 4))), dtype=float),
            12,
        ).reshape(-1)
    )


def _active_display_snapshot(controller: QuadricOcclusion3D) -> tuple[object, ...]:
    """Capture active render state while ignoring invisible stale slot buffers."""

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
                (
                    *fill[3::4],
                    *stroke[3::4],
                    *background[3::4],
                    0.0,
                )
            )
            # Keep the root even when it has no points: hiding a VGroup changes
            # its opacity and can suppress otherwise-valid active children.
            if index == 0 or (len(points) > 0 and own_alpha > 0.0):
                members.append(
                    (
                        tuple(float(item) for item in np.round(points, 12).reshape(-1)),
                        fill,
                        stroke,
                        background,
                        float(getattr(member, "z_index", 0.0)),
                    )
                )
        result.append((item_id, tuple(members)))
    return tuple(result)


class QuadricFinitePlaneLineManimTests(unittest.TestCase):
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

    def _assert_line_display_is_finite_and_not_duplicated(
        self,
        controller: QuadricOcclusion3D,
        *,
        boundary_mode: str,
    ) -> None:
        frame = controller.last_section_frame
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertIs(frame.projection_kind, PlanePatchProjectionKind.LINE)
        self.assertFalse(frame.has_plane_fill)
        self.assertEqual(frame.plane_fragments, ())

        prepared = controller._last_prepared_frame
        self.assertIsNotNone(prepared)
        assert prepared is not None
        layers = prepared.numeric.section_layers
        self.assertIsNotNone(layers)
        assert layers is not None
        self.assertTrue(
            all(not polygons for polygons in layers.plane_polygons.values())
        )

        slots = dict(zip(frame.paint_items.ordered, controller._section_slots))
        fill_ids = (
            frame.paint_items.plane_behind,
            frame.paint_items.plane_outside,
            frame.paint_items.plane_between,
            frame.paint_items.plane_front,
        )
        self.assertTrue(
            all(len(np.asarray(slots[item_id].points)) == 0 for item_id in fill_ids)
        )

        paths = tuple(
            np.asarray(path, dtype=float)
            for role in PlaneDepthRole
            for path in layers.plane_outline_paths[role]
        )
        self.assertEqual(len(paths), len(frame.plane_outline_fragments))
        self.assertGreater(len(paths), 0)
        self.assertTrue(
            all(
                path.shape == (2, 3) and np.all(np.isfinite(path))
                for path in paths
            )
        )

        line_start = np.asarray(frame.patch_projection.line_screen_start, dtype=float)
        line_end = np.asarray(frame.patch_projection.line_screen_end, dtype=float)
        line_length = float(np.linalg.norm(line_end - line_start))
        axis = (line_end - line_start) / line_length
        intervals = sorted(
            tuple(
                sorted(
                    (
                        float(np.dot(path[0, :2], axis)),
                        float(np.dot(path[-1, :2], axis)),
                    )
                )
            )
            for path in paths
        )
        tolerance = 1.0e-9 * max(1.0, line_length)
        cursor = intervals[0][0]
        total_length = 0.0
        for lower, upper in intervals:
            self.assertAlmostEqual(lower, cursor, delta=tolerance)
            self.assertGreater(upper, lower)
            total_length += upper - lower
            cursor = upper
        self.assertAlmostEqual(total_length, line_length, delta=tolerance)
        self.assertAlmostEqual(
            intervals[-1][1] - intervals[0][0],
            line_length,
            delta=tolerance,
        )

        outline_id_by_role = frame.paint_items.outline_by_role
        for role in PlaneDepthRole:
            slot = slots[outline_id_by_role[role]]
            if layers.plane_outline_paths[role]:
                self.assertGreater(len(np.asarray(slot.points)), 0)
                self.assertGreater(float(slot.get_stroke_opacity()), 0.0)
            else:
                self.assertEqual(len(np.asarray(slot.points)), 0)

        if boundary_mode == "unified":
            boundary_fragments = prepared.numeric.boundary_fragments or {}
            prepared_plane_edges = tuple(
                fragment
                for source_id, fragments in boundary_fragments.items()
                if source_id.startswith("boundary:plane:finite-cut:edge:")
                for fragment in fragments
            )
            self.assertEqual(prepared_plane_edges, ())
            boundary_frame = controller.last_boundary_frame
            self.assertIsNotNone(boundary_frame)
            assert boundary_frame is not None
            self.assertFalse(
                any(
                    fragment.source_id.startswith(
                        "boundary:plane:finite-cut:edge:"
                    )
                    for fragment in boundary_frame.fragments
                )
            )

    def test_area_line_area_commits_with_fixed_slots_and_matches_cold_frame(
        self,
    ) -> None:
        for boundary_mode in ("legacy", "unified"):
            with self.subTest(boundary_mode=boundary_mode):
                state: dict[str, object] = {
                    "projection": _projection_value(
                        boundary_mode,
                        INITIAL_AREA_STATE,
                    )
                }
                scene = _ParallelCameraScene()
                controller = _controller(
                    scene,
                    lambda _scene: state["projection"],
                    boundary_mode=boundary_mode,
                ).attach()
                cold: QuadricOcclusion3D | None = None
                try:
                    initial_frame = controller.last_section_frame
                    self.assertIsNotNone(initial_frame)
                    assert initial_frame is not None
                    self.assertIs(
                        initial_frame.projection_kind,
                        PlanePatchProjectionKind.AREA,
                    )
                    self.assertTrue(initial_frame.has_plane_fill)
                    self.assertGreater(len(initial_frame.plane_fragments), 0)

                    identities = controller.slot_identities()
                    ownership = _scene_ownership(controller)
                    scene_mobjects = tuple(id(item) for item in scene.mobjects)
                    state["projection"] = _projection_value(
                        boundary_mode,
                        LINE_STATE,
                    )
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
                        controller.update()
                    self.assertEqual(controller.slot_identities(), identities)
                    self.assertEqual(_scene_ownership(controller), ownership)
                    self.assertEqual(
                        tuple(id(item) for item in scene.mobjects),
                        scene_mobjects,
                    )
                    self._assert_line_display_is_finite_and_not_duplicated(
                        controller,
                        boundary_mode=boundary_mode,
                    )

                    state["projection"] = _projection_value(
                        boundary_mode,
                        FINAL_AREA_STATE,
                    )
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
                        controller.update()
                    final_frame = controller.last_section_frame
                    self.assertIsNotNone(final_frame)
                    assert final_frame is not None
                    self.assertIs(
                        final_frame.projection_kind,
                        PlanePatchProjectionKind.AREA,
                    )
                    self.assertTrue(final_frame.has_plane_fill)
                    self.assertEqual(controller.slot_identities(), identities)
                    self.assertEqual(_scene_ownership(controller), ownership)

                    cold = _controller(
                        _ParallelCameraScene(),
                        _projection_value(boundary_mode, FINAL_AREA_STATE),
                        boundary_mode=boundary_mode,
                    ).attach()
                    self.assertEqual(
                        _active_display_snapshot(controller),
                        _active_display_snapshot(cold),
                    )
                    assert cold.last_section_frame is not None
                    self.assertEqual(
                        canonical_quadric_section_compositing_json(final_frame),
                        canonical_quadric_section_compositing_json(
                            cold.last_section_frame
                        ),
                    )
                    if boundary_mode == "unified":
                        assert controller.last_boundary_frame is not None
                        assert cold.last_boundary_frame is not None
                        self.assertEqual(
                            canonical_quadric_boundary_compositing_json(
                                controller.last_boundary_frame
                            ),
                            canonical_quadric_boundary_compositing_json(
                                cold.last_boundary_frame
                            ),
                        )
                finally:
                    if cold is not None:
                        cold.restore()
                    controller.restore()

    def test_invalid_total_collapse_and_other_errors_keep_last_line_frame(
        self,
    ) -> None:
        for boundary_mode in ("legacy", "unified"):
            with self.subTest(boundary_mode=boundary_mode):
                state: dict[str, object] = {
                    "projection": _projection_value(boundary_mode, LINE_STATE)
                }
                controller = _controller(
                    _ParallelCameraScene(),
                    lambda _scene: state["projection"],
                    boundary_mode=boundary_mode,
                ).attach()
                try:
                    self._assert_line_display_is_finite_and_not_duplicated(
                        controller,
                        boundary_mode=boundary_mode,
                    )
                    committed = _committed_evidence(controller)

                    state["projection"] = (
                        np.zeros((3, 3), dtype=float)
                        if boundary_mode == "legacy"
                        else SimpleNamespace(
                            matrix=np.zeros((3, 3), dtype=float),
                            target=np.zeros(3, dtype=float),
                            screen_anchor=np.zeros(2, dtype=float),
                            zoom=1.0,
                        )
                    )
                    with self.assertRaisesRegex(
                        QuadricManimError,
                        "invalid .*parallel (projection|camera)",
                    ):
                        controller.update()
                    self.assertEqual(_committed_evidence(controller), committed)

                    state["projection"] = _projection_value(
                        boundary_mode,
                        FINAL_AREA_STATE,
                    )
                    with patch(
                        "polyhedron_visibility.quadrics.manim."
                        "compute_quadric_section_compositing",
                        side_effect=QuadricSectionCompositingError(
                            "synthetic section topology failure"
                        ),
                    ):
                        with self.assertRaisesRegex(
                            QuadricManimError,
                            "synthetic section topology failure",
                        ):
                            controller.update()
                    self.assertEqual(_committed_evidence(controller), committed)
                finally:
                    controller.restore()


if __name__ == "__main__":
    unittest.main()
