from __future__ import annotations

from math import sqrt
import unittest
from unittest.mock import patch

import numpy as np
from manim import Mobject, Scene, tempconfig

from diagnostics.quadrics_section_boundary_partition.scene import (
    STATES as DIAGNOSTIC_STATES,
    build_controller as build_diagnostic_controller,
)
from polyhedron_visibility.painter_band import ManagedPainterBandError
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import SectionPlane, SphereSpec
from polyhedron_visibility.quadrics.manim import (
    QuadricManimError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    QuadricSectionCompositingError,
    QuadricSectionCompositingLimits,
)


IDENTITY_VIEW = ParallelView.from_matrix(np.eye(3))


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=1,
        max_curves=1,
        max_fragments_per_curve=8,
        max_segments_per_fragment=128,
        max_surface_segments=256,
        max_dashes_per_fragment=32,
        max_projected_length=8.0,
        max_total_mobjects=1000,
    )


class QuadricSectionManimTests(unittest.TestCase):
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

    @staticmethod
    def _identity_evidence(controller: QuadricOcclusion3D) -> dict[str, object]:
        curve_slots = tuple(
            (
                curve_id,
                tuple(
                    id(member)
                    for member in controller._curve_slots[
                        curve_id
                    ].root.get_family()
                ),
            )
            for curve_id in controller.allocated_curve_ids
        )
        return {
            "all_slots": controller.slot_identities(),
            "surface_slots": tuple(id(item) for item in controller._surface_slots),
            "section_slots": tuple(id(item) for item in controller._section_slots),
            "curve_slots": curve_slots,
            "display_root": id(controller.root),
            "root_groups": tuple(
                id(item) for item in controller.root.submobjects[:3]
            ),
            "scene_mobjects": tuple(id(item) for item in controller.scene.mobjects),
            "scene_containers": tuple(
                tuple(id(item) for item in container)
                for container in controller._scene_containers()
            ),
        }

    @classmethod
    def _committed_evidence(
        cls,
        controller: QuadricOcclusion3D,
    ) -> dict[str, object]:
        return {
            "identity": cls._identity_evidence(controller),
            "slot_snapshot": controller.slot_snapshot(),
            "active_z": controller.active_painter_z_indices,
            "band_state": controller._band.capture_active_state(),
            "fragment_maps": tuple(
                (
                    curve_id,
                    tuple(sorted(values.items())),
                )
                for curve_id, values in sorted(
                    controller._fragment_slot_maps.items()
                )
            ),
            "last_frame": controller.last_frame,
            "last_global_frame": controller.last_global_frame,
            "last_section_frame": controller.last_section_frame,
        }

    def _assert_committed_evidence_unchanged(
        self,
        controller: QuadricOcclusion3D,
        evidence: dict[str, object],
    ) -> None:
        current = self._committed_evidence(controller)
        self.assertEqual(current["identity"], evidence["identity"])
        self.assertEqual(current["slot_snapshot"], evidence["slot_snapshot"])
        self.assertEqual(current["active_z"], evidence["active_z"])
        self.assertEqual(current["band_state"], evidence["band_state"])
        self.assertEqual(current["fragment_maps"], evidence["fragment_maps"])
        self.assertIs(current["last_frame"], evidence["last_frame"])
        self.assertIs(
            current["last_global_frame"],
            evidence["last_global_frame"],
        )
        self.assertIs(
            current["last_section_frame"],
            evidence["last_section_frame"],
        )

    @staticmethod
    def _five_state_controller():
        state = {"name": DIAGNOSTIC_STATES[0].name}
        scene = Scene()
        controller = build_diagnostic_controller(
            scene,
            lambda: state["name"],
            "opaque_fill",
        ).attach()
        return scene, state, controller

    def test_section_groups_and_surface_sheets_use_one_managed_order(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        style = QuadricManimStyle(
            surface_fill_color="#315F91",
            surface_fill_opacity=0.76,
            section_plane_fill_opacity=0.15,
        )
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(sphere,),
            curves=(),
            projection=IDENTITY_VIEW,
            style=style,
            limits=_limits(),
            section_plane=plane,
            section_max_screen_error=0.12,
        ).attach()

        frame = controller.last_section_frame
        self.assertIsNotNone(frame)
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
        self.assertTrue(
            {
                PlaneDepthRole.BEHIND_SURFACE,
                PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                PlaneDepthRole.IN_FRONT_OF_SURFACE,
            }.issubset(
                {item.role for item in frame.plane_fragments}
            )
        )
        expected_sheet_opacity = 1.0 - sqrt(1.0 - style.surface_fill_opacity)
        self.assertAlmostEqual(
            float(controller._section_slots[1].get_fill_opacity()),
            expected_sheet_opacity,
            places=12,
        )
        self.assertAlmostEqual(
            float(controller._section_slots[4].get_fill_opacity()),
            expected_sheet_opacity,
            places=12,
        )
        controller.restore()

    def test_dynamic_plane_preserves_slots_and_rolls_back_edge_on_failure(self) -> None:
        state = {"normal": (0.7, 0.0, 1.0), "u": (0.0, 1.0, 0.0)}

        def plane() -> SectionPlane:
            return SectionPlane(
                "cut",
                (0.0, 0.0, 0.0),
                state["normal"],
                u_axis=state["u"],
            )

        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=(),
            projection=IDENTITY_VIEW,
            limits=_limits(),
            section_plane=plane,
            section_max_screen_error=0.12,
        ).attach()
        identities = controller.slot_identities()
        state["normal"] = (0.25, 0.0, 1.0)
        with patch.object(
            Mobject,
            "__init__",
            side_effect=AssertionError("section update allocated a Mobject"),
        ):
            controller.update()
        self.assertEqual(controller.slot_identities(), identities)
        snapshot = controller.slot_snapshot()

        state["normal"] = (1.0, 0.0, 0.0)
        state["u"] = (0.0, 0.0, 1.0)
        with self.assertRaisesRegex(QuadricManimError, "projects edge-on"):
            controller.update()
        self.assertEqual(controller.slot_snapshot(), snapshot)
        controller.restore()

    def test_five_state_updates_keep_ten_slots_identity_and_painter_order(
        self,
    ) -> None:
        scene, state, controller = self._five_state_controller()
        try:
            identities = self._identity_evidence(controller)
            self.assertEqual(len(controller._section_slots), 10)
            first_frame = controller.last_section_frame
            self.assertIsNotNone(first_frame)
            assert first_frame is not None
            section_slot_by_item = {
                item_id: id(slot)
                for item_id, slot in zip(
                    first_frame.paint_items.ordered,
                    controller._section_slots,
                )
            }
            painter_band = controller._band.z_band

            for diagnostic_state in DIAGNOSTIC_STATES:
                with self.subTest(state=diagnostic_state.name):
                    state["name"] = diagnostic_state.name
                    with (
                        patch.object(
                            Mobject,
                            "__init__",
                            side_effect=AssertionError(
                                "five-state updater allocated a Mobject"
                            ),
                        ),
                        patch.object(
                            scene,
                            "add",
                            side_effect=AssertionError(
                                "five-state updater added a Scene object"
                            ),
                        ),
                        patch.object(
                            scene,
                            "remove",
                            side_effect=AssertionError(
                                "five-state updater removed a Scene object"
                            ),
                        ),
                    ):
                        controller.update()

                    self.assertEqual(
                        self._identity_evidence(controller),
                        identities,
                    )
                    frame = controller.last_section_frame
                    self.assertIsNotNone(frame)
                    assert frame is not None
                    items = frame.paint_items
                    expected_depth_chain = (
                        items.plane_behind,
                        items.plane_outline_behind,
                        items.surface_back,
                        items.plane_outside,
                        items.plane_outline_outside,
                        items.plane_between,
                        items.plane_outline_between,
                        items.surface_front,
                        items.plane_front,
                        items.plane_outline,
                    )
                    self.assertEqual(items.depth_chain, expected_depth_chain)
                    self.assertEqual(len(items.ordered), 10)
                    self.assertEqual(len(set(items.ordered)), 10)
                    self.assertEqual(
                        {
                            item_id: id(slot)
                            for item_id, slot in zip(
                                items.ordered,
                                controller._section_slots,
                            )
                        },
                        section_slot_by_item,
                    )
                    self.assertEqual(
                        tuple(
                            item_id
                            for item_id in frame.draw_order
                            if item_id in set(items.depth_chain)
                        ),
                        items.depth_chain,
                    )
                    active_z = controller.active_painter_z_indices
                    self.assertEqual(set(active_z), set(frame.draw_order))
                    self.assertEqual(
                        [active_z[item_id] for item_id in frame.draw_order],
                        sorted(active_z.values()),
                    )
                    self.assertEqual(controller._band.z_band, painter_band)
        finally:
            controller.restore()

    def test_geometry_prepare_faults_leave_every_committed_value_unchanged(
        self,
    ) -> None:
        _scene, state, controller = self._five_state_controller()
        try:
            state["name"] = "exact_parabola"
            fault_patchers = (
                (
                    "polygon partition cannot close",
                    patch(
                        "polyhedron_visibility.quadrics.section_compositing."
                        "_partition_triangle_by_convex_proxy",
                        side_effect=QuadricSectionCompositingError(
                            "synthetic polygon partition cannot close"
                        ),
                    ),
                ),
                (
                    "triangulation is degenerate",
                    patch(
                        "polyhedron_visibility.quadrics.section_compositing."
                        "_triangulate_plane_partition_polygon",
                        side_effect=QuadricSectionCompositingError(
                            "synthetic triangulation is degenerate"
                        ),
                    ),
                ),
                (
                    "contour union is open",
                    patch(
                        "polyhedron_visibility.quadrics.manim."
                        "quadric_plane_fragment_contours",
                        side_effect=QuadricSectionCompositingError(
                            "synthetic contour union is open"
                        ),
                    ),
                ),
            )
            for label, fault_patcher in fault_patchers:
                with self.subTest(fault=label):
                    evidence = self._committed_evidence(controller)
                    with fault_patcher:
                        with self.assertRaisesRegex(
                            QuadricManimError,
                            "synthetic",
                        ):
                            controller.update()
                    self._assert_committed_evidence_unchanged(
                        controller,
                        evidence,
                    )
        finally:
            controller.restore()

    def test_fragment_capacity_failure_leaves_committed_state_unchanged(
        self,
    ) -> None:
        _scene, state, controller = self._five_state_controller()
        original_limits = controller.section_compositing_limits
        try:
            evidence = self._committed_evidence(controller)
            state["name"] = "intersects"
            controller.section_compositing_limits = (
                QuadricSectionCompositingLimits(max_plane_fragments=8)
            )
            with self.assertRaisesRegex(
                QuadricManimError,
                "more than 8 plane fragments",
            ):
                controller.update()
            self._assert_committed_evidence_unchanged(controller, evidence)
        finally:
            controller.section_compositing_limits = original_limits
            controller.restore()

    def test_painter_prepare_and_commit_faults_restore_complete_section_state(
        self,
    ) -> None:
        _scene, state, controller = self._five_state_controller()
        try:
            state["name"] = "mainly_front"
            evidence = self._committed_evidence(controller)
            with patch.object(
                controller._band,
                "prepare",
                side_effect=ManagedPainterBandError(
                    "synthetic painter prepare failure"
                ),
            ):
                with self.assertRaisesRegex(
                    QuadricManimError,
                    "synthetic painter prepare failure",
                ):
                    controller.update()
            self._assert_committed_evidence_unchanged(controller, evidence)

            original_apply = controller._band.apply

            def fail_after_painter_commit(prepared) -> None:
                original_apply(prepared)
                raise RuntimeError("synthetic painter commit failure")

            with patch.object(
                controller._band,
                "apply",
                side_effect=fail_after_painter_commit,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "synthetic painter commit failure",
                ):
                    controller.update()
            self._assert_committed_evidence_unchanged(controller, evidence)
        finally:
            controller.restore()


if __name__ == "__main__":
    unittest.main()
