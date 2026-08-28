from __future__ import annotations

from math import cos, pi, sin
import unittest
from unittest.mock import patch

from manim import Mobject, Scene

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
    canonical_quadric_boundary_compositing_json,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.manim import (
    QuadricManimError,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.projection import ProjectionProxyError
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
    section_cap_chord_curve_ids,
)


OBLIQUE_VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)


def _near_side_view(angle: float) -> ParallelView:
    return ParallelView.from_matrix(
        (
            (1.0, 0.0, 0.0),
            (0.0, -sin(angle), cos(angle)),
            (0.0, -cos(angle), -sin(angle)),
        )
    )


def _frustum(surface_id: str = "component-frustum") -> ConeSpec:
    return ConeSpec(
        surface_id,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (0.75, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.CLOSED_SINGLE,
    )


def _section_plane() -> SectionPlane:
    return SectionPlane(
        "component-frustum-plane",
        (0.2, 0.0, 1.3),
        (1.0, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
    )


def _component_style() -> QuadricManimStyle:
    return QuadricManimStyle(
        surface_fill_opacity=0.72,
        surface_stroke_opacity=0.0,
        cone_lateral_fill_colors=("#173753", "#4F84B3", "#1D4368"),
        cone_cap_fill_colors=("#557A99", "#294B6B"),
    )


class FrustumComponentShadingTests(unittest.TestCase):
    def test_two_cap_chords_and_rims_share_every_supported_painter_policy(
        self,
    ) -> None:
        frustum = _frustum()
        plane = _section_plane()
        curves = compute_quadric_section_boundary_curves(
            "component-frustum-section",
            frustum,
            plane,
        )
        chords = tuple(item for item in curves if isinstance(item, SegmentCurve))
        self.assertEqual(len(chords), 2)
        self.assertEqual(
            {item.curve_id for item in chords},
            set(section_cap_chord_curve_ids("component-frustum-section", frustum)),
        )

        for policy in QuadricPaintPolicy:
            with self.subTest(policy=policy.value):
                controller = QuadricOcclusion3D(
                    Scene(),
                    surfaces=(frustum,),
                    curves=curves,
                    projection=OBLIQUE_VIEW,
                    paint_policy=policy,
                    section_plane=plane,
                    boundary_visibility_mode="unified",
                    include_surface_boundaries=True,
                    max_chord_error=0.01,
                    style=_component_style(),
                ).attach()
                try:
                    frame = controller.last_boundary_frame
                    section = controller.last_section_frame
                    assert frame is not None and section is not None
                    cap_chord_sources = tuple(
                        item
                        for item in frame.sources
                        if item.source_kind is BoundarySourceKind.SECTION_CAP_CHORD
                    )
                    cap_rim_sources = tuple(
                        item
                        for item in frame.sources
                        if item.source_kind is BoundarySourceKind.SURFACE_CAP_RIM
                    )
                    self.assertEqual(len(cap_chord_sources), 2)
                    self.assertEqual(len(cap_rim_sources), 2)
                    self.assertEqual(
                        len(frame.draw_order),
                        len(set(frame.draw_order)),
                    )
                    painted = {
                        item.item_id for item in frame.fragments if item.painted
                    }
                    self.assertEqual(
                        set(frame.draw_order),
                        set(frame.parent_item_ids) | painted,
                    )
                    back_slot = controller._section_surface_paint_slots[1]
                    front_slot = controller._section_surface_paint_slots[4]
                    self.assertGreater(len(back_slot.back_lateral.points), 0)
                    self.assertGreater(len(front_slot.front_lateral.points), 0)
                    self.assertGreater(len(back_slot.back_cap.points), 0)
                    self.assertGreater(len(front_slot.front_cap.points), 0)
                finally:
                    controller.restore()

    def test_component_masks_do_not_change_the_section_painter_graph(self) -> None:
        frustum = _frustum("graph-frustum")
        plane = _section_plane()
        curves = compute_quadric_section_boundary_curves(
            "graph-frustum-section",
            frustum,
            plane,
        )

        def build(style: QuadricManimStyle) -> QuadricOcclusion3D:
            return QuadricOcclusion3D(
                Scene(),
                surfaces=(frustum,),
                curves=curves,
                projection=OBLIQUE_VIEW,
                paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                section_plane=plane,
                boundary_visibility_mode="unified",
                include_surface_boundaries=True,
                max_chord_error=0.01,
                style=style,
            ).attach()

        uniform = build(QuadricManimStyle(surface_fill_opacity=0.72))
        component = build(_component_style())
        try:
            uniform_frame = uniform.last_boundary_frame
            component_frame = component.last_boundary_frame
            assert uniform_frame is not None and component_frame is not None
            self.assertEqual(
                canonical_quadric_boundary_compositing_json(uniform_frame),
                canonical_quadric_boundary_compositing_json(component_frame),
            )
        finally:
            component.restore()
            uniform.restore()

    def test_near_side_rank_switch_keeps_slots_and_scene_membership_fixed(
        self,
    ) -> None:
        state = {"angle": 0.02}

        def current_view(scene: object) -> ParallelView:
            del scene
            return _near_side_view(state["angle"])

        scene = Scene()
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(_frustum("rank-switch-frustum"),),
            curves=(),
            projection=current_view,
            boundary_visibility_mode="unified",
            include_surface_boundaries=True,
            max_chord_error=0.01,
            style=_component_style(),
        ).attach()
        try:
            identities = controller.slot_identities()
            scene_members = tuple(id(item) for item in scene.mobjects)
            committed_frames = []
            with patch.object(
                Mobject,
                "__init__",
                side_effect=AssertionError("frustum update allocated a Mobject"),
            ):
                for angle in (0.01, 0.0025, 0.0, -0.0025, -0.01):
                    state["angle"] = angle
                    controller.update()
                    committed_frames.append(controller.last_boundary_frame)
                    self.assertEqual(controller.slot_identities(), identities)
                    self.assertEqual(
                        tuple(id(item) for item in scene.mobjects),
                        scene_members,
                    )
                    if angle == 0.0:
                        slot = controller._surface_paint_slots[0]
                        self.assertEqual(len(slot.back_cap.points), 0)
                        self.assertEqual(len(slot.front_cap.points), 0)
            self.assertTrue(all(item is not None for item in committed_frames))
            self.assertEqual(len({id(item) for item in committed_frames}), 5)
        finally:
            controller.restore()

    def test_component_projection_failure_preserves_last_good_frame(self) -> None:
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(_frustum("rollback-frustum"),),
            curves=(),
            projection=OBLIQUE_VIEW,
            boundary_visibility_mode="unified",
            include_surface_boundaries=True,
            max_chord_error=0.01,
            style=_component_style(),
        ).attach()
        try:
            snapshot = controller.slot_snapshot()
            identities = controller.slot_identities()
            previous_frame = controller.last_boundary_frame
            with patch(
                "polyhedron_visibility.quadrics.manim.build_cone_projection_layers",
                side_effect=ProjectionProxyError("forced terminal-mask failure"),
            ):
                with self.assertRaisesRegex(
                    QuadricManimError,
                    "forced terminal-mask failure",
                ):
                    controller.update()
            self.assertEqual(controller.slot_snapshot(), snapshot)
            self.assertEqual(controller.slot_identities(), identities)
            self.assertIs(controller.last_boundary_frame, previous_frame)
        finally:
            controller.restore()


if __name__ == "__main__":
    unittest.main()
