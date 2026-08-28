from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from math import cos, pi, sin
import unittest

import numpy as np
from manim import Scene, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
    CompositeQuadricSectionAuthoringError,
)
from polyhedron_visibility.quadrics.composite_section import (
    CompositeQuadricSectionCompositingError,
    CompositeSectionBranchLineage,
    canonical_composite_quadric_section_compositing_json,
    compute_composite_quadric_section_compositing,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.global_occlusion import (
    compute_global_quadric_frame,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimCapacityError,
    QuadricManimLimits,
    QuadricManimStyle,
)
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    QuadricSectionCompositingError,
    compute_quadric_section_compositing,
    merge_quadric_plane_fragment_contours,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
)


SIDE_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)
AXIAL_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
)


def _double(surface_id: str = "double") -> ConeSpec:
    return ConeSpec(
        surface_id,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        (-2.0, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.OPEN_DOUBLE,
    )


def _side_plane(offset: float = 0.5) -> SectionPlane:
    return SectionPlane(
        "double-cut",
        (0.0, offset, 0.0),
        (0.0, 1.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
    )


def _view_at_axis_angle(angle: float) -> ParallelView:
    return ParallelView.from_matrix(
        (
            (1.0, 0.0, 0.0),
            (0.0, cos(angle), -sin(angle)),
            (0.0, sin(angle), cos(angle)),
        )
    )


def _limits(**overrides: object) -> QuadricManimLimits:
    values: dict[str, object] = {
        "max_surfaces": 2,
        "max_curves": 8,
        "max_fragments_per_curve": 24,
        "max_segments_per_fragment": 256,
        "max_surface_segments": 384,
        "max_dashes_per_fragment": 64,
        "max_projected_length": 24.0,
        "max_total_mobjects": 30000,
        "max_boundary_sources": 32,
    }
    values.update(overrides)
    return QuadricManimLimits(**values)  # type: ignore[arg-type]


def _physical_curves(
    section_id: str,
    child: ConeSpec,
    plane: SectionPlane,
) -> tuple[tuple[object, ...], tuple[CompositeSectionBranchLineage, ...]]:
    role = child.surface_id.rsplit(":", 1)[-1]
    curves = []
    lineage = []
    for curve in compute_quadric_section_boundary_curves(
        section_id,
        child,
        plane,
    ):
        physical_id = (
            f"{section_id}:nappe:{role}:"
            f"{curve.curve_id[len(section_id) + 1:]}"
        )
        physical = replace(curve, curve_id=physical_id)
        curves.append(physical)
        lineage.append(
            CompositeSectionBranchLineage(
                physical_id,
                f"{section_id}:component:{curve.parameterization.branch_label}",
                child.surface_id,
                role,
            )
        )
    return tuple(curves), tuple(lineage)


def _renderer_neutral_frame(
    cone: ConeSpec,
    plane: SectionPlane,
    view: ParallelView = SIDE_VIEW,
    *,
    max_plane_fragments: int | None = None,
):
    children = cone.render_components
    patch = fit_plane_display_patch(
        f"{plane.plane_id}:test-patch",
        plane,
        children,
        margin_ratio=0.15,
    ).patch
    frames = []
    lineage = []
    for child in children:
        base = compute_global_quadric_frame(
            (),
            (child,),
            view,
            paint_policy=QuadricPaintPolicy.PHYSICAL,
            max_chord_error=0.02,
            max_segments=384,
        )
        frames.append(
            compute_quadric_section_compositing(
                base.frame,
                child,
                plane,
                patch,
                view,
                max_screen_error=0.12,
            )
        )
        _curves, child_lineage = _physical_curves(
            "double-section", child, plane
        )
        lineage.extend(child_lineage)
    arguments = {}
    if max_plane_fragments is not None:
        arguments["max_plane_fragments"] = max_plane_fragments
    return compute_composite_quadric_section_compositing(
        cone,
        "double-section",
        frames,
        lineage,
        **arguments,
    )


def _frame_with_proxy_vertices(
    frame,
    vertices: tuple[tuple[float, float], ...],
):
    area_twice = sum(
        start[0] * end[1] - start[1] * end[0]
        for start, end in zip(vertices, (*vertices[1:], vertices[0]))
    )
    ordered = vertices if area_twice > 0.0 else tuple(reversed(vertices))
    closed = (*ordered, ordered[0])
    metadata = replace(
        frame.surface_proxy.metadata,
        segment_count=len(ordered),
    )
    proxy = replace(
        frame.surface_proxy,
        boundary_points=closed,
        metadata=metadata,
    )
    return replace(frame, surface_proxy=proxy)


class CompositeRendererNeutralTests(unittest.TestCase):
    def test_two_local_frames_merge_into_one_area_conserving_plane_partition(
        self,
    ) -> None:
        frame = _renderer_neutral_frame(_double(), _side_plane())
        self.assertEqual(
            tuple(item.surface_id for item in frame.child_frames),
            ("double:nappe:negative", "double:nappe:positive"),
        )
        self.assertEqual(frame.shared_apex.projected_overlap_area, 0.0)
        self.assertEqual(frame.shared_apex.contact_dimension, 0)
        self.assertEqual(frame.shared_apex.contact_extent, 0.0)
        self.assertEqual(frame.shared_apex.max_contact_distance_from_apex, 0.0)
        self.assertEqual(frame.shared_apex.contact_points, ((0.0, 0.0),))
        self.assertEqual(len(frame.paint_items.surface_sheets), 2)
        self.assertEqual(len(frame.paint_items.ordered), 12)
        self.assertEqual(set(frame.draw_order), set(frame.paint_items.ordered))
        area = sum(
            abs(
                0.5
                * (
                    (
                        np.asarray(item.screen_vertices[1])
                        - np.asarray(item.screen_vertices[0])
                    )[0]
                    * (
                        np.asarray(item.screen_vertices[2])
                        - np.asarray(item.screen_vertices[0])
                    )[1]
                    - (
                        np.asarray(item.screen_vertices[1])
                        - np.asarray(item.screen_vertices[0])
                    )[1]
                    * (
                        np.asarray(item.screen_vertices[2])
                        - np.asarray(item.screen_vertices[0])
                    )[0]
                )
            )
            for item in frame.plane_fragments
        )
        expected = 4.0 * frame.patch.half_width * frame.patch.half_height
        self.assertAlmostEqual(area, expected, delta=expected * 2.0e-9)
        contours = merge_quadric_plane_fragment_contours(
            frame.plane,
            frame.patch,
            frame.child_frames[0].base_frame.visibility.projection_matrix,
            frame.plane_fragments,
        )
        self.assertTrue(contours[PlaneDepthRole.OUTSIDE_PROJECTION])
        self.assertTrue(contours[PlaneDepthRole.BETWEEN_SURFACE_SHEETS])
        first = canonical_composite_quadric_section_compositing_json(frame)
        second = canonical_composite_quadric_section_compositing_json(frame)
        self.assertEqual(first, second)
        shared_apex = json.loads(first)["sharedApex"]
        self.assertEqual(shared_apex["contactDimension"], 0)
        self.assertEqual(shared_apex["contactExtent"], 0.0)
        self.assertEqual(shared_apex["maxContactDistanceFromApex"], 0.0)
        self.assertEqual(shared_apex["contactPoints"], [[0.0, 0.0]])

    def test_shared_mathematical_generator_lineage_keeps_separate_physical_ids(
        self,
    ) -> None:
        frame = _renderer_neutral_frame(_double(), _side_plane(0.0))
        counts = Counter(item.mathematical_branch_id for item in frame.branch_lineage)
        self.assertEqual(len(frame.branch_lineage), 4)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(
            len({item.physical_curve_id for item in frame.branch_lineage}),
            4,
        )

    def test_positive_area_axial_projection_overlap_fails_explicitly(self) -> None:
        cone = _double("axial-double")
        plane = SectionPlane(
            "axial-cut",
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        children = cone.render_components
        patch = fit_plane_display_patch(
            "axial-patch",
            plane,
            children,
            margin_ratio=0.15,
        ).patch
        frames = []
        for child in children:
            base = compute_global_quadric_frame(
                (),
                (child,),
                AXIAL_VIEW,
                paint_policy=QuadricPaintPolicy.PHYSICAL,
                max_chord_error=0.02,
                max_segments=384,
            )
            frames.append(
                compute_quadric_section_compositing(
                    base.frame,
                    child,
                    plane,
                    patch,
                    AXIAL_VIEW,
                    max_screen_error=0.12,
                )
            )
        with self.assertRaisesRegex(
            CompositeQuadricSectionCompositingError,
            "positive-area overlap",
        ):
            compute_composite_quadric_section_compositing(
                cone,
                "axial-section",
                frames,
            )

    def test_far_single_point_contact_is_not_mistaken_for_shared_apex(self) -> None:
        cone = _double("far-contact-double")
        ordinary = _renderer_neutral_frame(cone, _side_plane())
        delta = 1.0e-10
        frames = (
            _frame_with_proxy_vertices(
                ordinary.child_frames[0],
                ((-delta, delta), (1.0, 0.0), (0.5, 0.8)),
            ),
            _frame_with_proxy_vertices(
                ordinary.child_frames[1],
                ((delta, -delta), (0.5, -0.8), (1.0, 0.0)),
            ),
        )
        with self.assertRaisesRegex(
            CompositeQuadricSectionCompositingError,
            "contact lies away from the shared apex",
        ):
            compute_composite_quadric_section_compositing(
                cone,
                "far-contact-section",
                frames,
            )

    def test_nonzero_collinear_contact_through_apex_fails_explicitly(self) -> None:
        cone = _double("segment-contact-double")
        ordinary = _renderer_neutral_frame(cone, _side_plane())
        frames = (
            _frame_with_proxy_vertices(
                ordinary.child_frames[0],
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            ),
            _frame_with_proxy_vertices(
                ordinary.child_frames[1],
                ((0.0, 0.0), (0.0, -1.0), (1.0, 0.0)),
            ),
        )
        with self.assertRaisesRegex(
            CompositeQuadricSectionCompositingError,
            "nonzero-length contact segment",
        ):
            compute_composite_quadric_section_compositing(
                cone,
                "segment-contact-section",
                frames,
            )

    def test_critical_view_scan_never_accepts_uncertified_contact(self) -> None:
        cone = _double("critical-scan-double")
        plane = _side_plane()
        critical = pi / 4.0
        for offset in (0.08, 0.01, 1.0e-4):
            frame = _renderer_neutral_frame(
                cone,
                plane,
                _view_at_axis_angle(critical + offset),
            )
            self.assertEqual(frame.shared_apex.contact_dimension, 0)
            self.assertLessEqual(
                frame.shared_apex.contact_extent,
                frame.shared_apex.boundary_tolerance,
            )
        critical_frame = _renderer_neutral_frame(
            cone,
            plane,
            _view_at_axis_angle(critical),
        )
        self.assertEqual(critical_frame.shared_apex.contact_dimension, 0)
        self.assertLessEqual(
            critical_frame.shared_apex.max_contact_distance_from_apex,
            critical_frame.shared_apex.boundary_tolerance,
        )
        for offset in (-1.0e-4, -0.01):
            with self.assertRaisesRegex(
                CompositeQuadricSectionCompositingError,
                "positive-area overlap|contact is two-dimensional",
            ):
                _renderer_neutral_frame(
                    cone,
                    plane,
                    _view_at_axis_angle(critical + offset),
                )
        for offset in (
            1.0e-6,
            1.0e-8,
            1.0e-10,
            1.0e-12,
            -1.0e-12,
            -1.0e-10,
            -1.0e-8,
            -1.0e-6,
        ):
            try:
                frame = _renderer_neutral_frame(
                    cone,
                    plane,
                    _view_at_axis_angle(critical + offset),
                )
            except (
                CompositeQuadricSectionCompositingError,
                QuadricSectionCompositingError,
            ):
                continue
            evidence = frame.shared_apex
            self.assertEqual(evidence.contact_dimension, 0)
            self.assertLessEqual(
                evidence.contact_extent,
                evidence.boundary_tolerance,
            )
            self.assertLessEqual(
                evidence.max_contact_distance_from_apex,
                evidence.boundary_tolerance,
            )
            self.assertTrue(
                all(
                    np.linalg.norm(
                        np.asarray(point) - np.asarray(evidence.screen_point)
                    )
                    <= evidence.boundary_tolerance
                    for point in evidence.contact_points
                )
            )

    def test_composite_plane_fragment_capacity_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(
            CompositeQuadricSectionCompositingError,
            "fragment count.*capacity 1",
        ):
            _renderer_neutral_frame(
                _double("fragment-capacity-double"),
                _side_plane(),
                max_plane_fragments=1,
            )


class CompositeManimBindingTests(unittest.TestCase):
    def test_attach_update_and_restore_keep_two_fixed_slot_groups(self) -> None:
        state = {"offset": 0.5}

        def plane() -> SectionPlane:
            return _side_plane(state["offset"])

        with tempconfig({"renderer": "cairo"}):
            scene = Scene()
            controller = CompositeQuadricSection3D(
                scene,
                surface=_double("binding-double"),
                section_id="binding-section",
                plane=plane,
                projection=SIDE_VIEW,
                limits=_limits(),
                max_chord_error=0.03,
                section_max_screen_error=0.16,
                style=QuadricManimStyle(
                    surface_fill_opacity=0.58,
                    cone_lateral_fill_colors=(
                        "#173753",
                        "#4F84B3",
                        "#1D4368",
                    ),
                ),
            ).attach()
            try:
                scene_ids = tuple(id(item) for item in scene.mobjects)
                identities = controller.slot_identities()
                child_groups = controller.child_slot_identities()
                first_frame = controller.last_composite_frame
                assert first_frame is not None
                self.assertEqual(len(child_groups), 2)
                self.assertTrue(all(child_groups.values()))
                self.assertEqual(
                    len(controller.branch_lineage),
                    len(controller.allocated_curve_ids),
                )
                self.assertEqual(
                    set(controller.active_painter_z_indices),
                    set(controller.last_boundary_frame.draw_order),
                )
                state["offset"] = 0.68
                controller.update()
                self.assertEqual(controller.slot_identities(), identities)
                self.assertEqual(controller.child_slot_identities(), child_groups)
                self.assertEqual(
                    tuple(id(item) for item in scene.mobjects), scene_ids
                )
                self.assertNotEqual(
                    canonical_composite_quadric_section_compositing_json(
                        controller.last_composite_frame
                    ),
                    canonical_composite_quadric_section_compositing_json(
                        first_frame
                    ),
                )
            finally:
                controller.restore()
            self.assertEqual(scene.mobjects, [])

    def test_capacity_overflow_fails_before_scene_mutation(self) -> None:
        scene = Scene()
        with self.assertRaisesRegex(
            QuadricManimCapacityError,
            "preallocated Mobject count",
        ):
            CompositeQuadricSection3D(
                scene,
                surface=_double("capacity-double"),
                section_id="capacity-section",
                plane=_side_plane(),
                projection=SIDE_VIEW,
                limits=_limits(max_total_mobjects=128),
            )
        self.assertEqual(scene.mobjects, [])

    def test_topology_change_rolls_back_without_replacing_slots(self) -> None:
        state = {"horizontal": False}

        def plane() -> SectionPlane:
            if state["horizontal"]:
                return SectionPlane(
                    "double-cut",
                    (0.0, 0.0, 0.5),
                    (0.0, 0.0, 1.0),
                    u_axis=(1.0, 0.0, 0.0),
                )
            return _side_plane()

        with tempconfig({"renderer": "cairo"}):
            controller = CompositeQuadricSection3D(
                Scene(),
                surface=_double("rollback-double"),
                section_id="rollback-section",
                plane=plane,
                projection=SIDE_VIEW,
                limits=_limits(),
                max_chord_error=0.03,
                section_max_screen_error=0.16,
            ).attach()
            try:
                snapshot = controller.slot_snapshot()
                identities = controller.slot_identities()
                frame = controller.last_composite_frame
                boundary = controller.last_boundary_frame
                z_indices = controller.active_painter_z_indices
                state["horizontal"] = True
                with self.assertRaisesRegex(
                    QuadricManimCapacityError,
                    "curve identities changed",
                ):
                    controller.update()
                self.assertEqual(controller.slot_snapshot(), snapshot)
                self.assertEqual(controller.slot_identities(), identities)
                self.assertIs(controller.last_composite_frame, frame)
                self.assertIs(controller.last_boundary_frame, boundary)
                self.assertEqual(controller.active_painter_z_indices, z_indices)
            finally:
                controller.restore()

    def test_wrong_cone_model_fails_before_scene_mutation(self) -> None:
        scene = Scene()
        single = ConeSpec(
            "single",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_SINGLE,
        )
        with self.assertRaisesRegex(
            CompositeQuadricSectionAuthoringError,
            "OPEN_DOUBLE",
        ):
            CompositeQuadricSection3D(
                scene,
                surface=single,
                section_id="bad-section",
                plane=_side_plane(),
            )
        self.assertEqual(scene.mobjects, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
