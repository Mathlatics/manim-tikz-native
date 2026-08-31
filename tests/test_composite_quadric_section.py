from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from math import cos, pi, sin
import unittest
from unittest.mock import patch

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.visibility import VisibilityKind
from polyhedron_visibility.quadrics.composite_section import (
    COMPOSITE_QUADRIC_SECTION_COMPOSITING_SCHEMA,
    CompositeQuadricSectionCompositingError,
    CompositeSectionBranchLineage,
    canonical_composite_quadric_section_compositing_json,
    compute_composite_quadric_section_compositing,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.global_occlusion import (
    compute_global_quadric_frame,
)
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionKind,
    QuadricSectionCompositingError,
    compute_quadric_section_compositing,
    merge_quadric_plane_fragment_contours,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
)
from polyhedron_visibility.visibility import VisibilityKind

try:
    from manim import Scene, tempconfig

    from polyhedron_visibility.quadrics.composite_authoring import (
        CompositeQuadricSection3D,
        CompositeQuadricSectionAuthoringError,
    )
    from polyhedron_visibility.quadrics.manim import (
        QuadricManimCapacityError,
        QuadricManimLimits,
        QuadricManimStyle,
    )
except ModuleNotFoundError as exc:
    if exc.name != "manim":
        raise
    MANIM_AVAILABLE = False
else:
    MANIM_AVAILABLE = True


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


def _edge_on_side_plane_view(angle: float = 0.0) -> ParallelView:
    """Look along one in-plane direction while keeping plane normal horizontal."""

    return ParallelView.from_matrix(
        (
            (0.0, 1.0, 0.0),
            (-sin(angle), 0.0, cos(angle)),
            (cos(angle), 0.0, sin(angle)),
        )
    )


EDGE_ON_VIEW = _edge_on_side_plane_view()


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


def _line_outline_scalar_intervals(frame) -> tuple[tuple[float, float], ...]:
    start = np.asarray(frame.patch_projection.line_screen_start, dtype=float)
    end = np.asarray(frame.patch_projection.line_screen_end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    axis = direction / length
    result = []
    for fragment in frame.plane_outline_fragments:
        values = tuple(
            float(np.dot(np.asarray(point, dtype=float) - start, axis))
            for point in (fragment.screen_start, fragment.screen_end)
        )
        result.append(tuple(sorted(values)))
    return tuple(sorted(result))


class CompositeRendererNeutralTests(unittest.TestCase):
    def _assert_finite_non_overlapping_line_chain(self, frame) -> None:
        self.assertIs(frame.projection_kind, PlanePatchProjectionKind.LINE)
        self.assertFalse(frame.has_plane_fill)
        self.assertEqual(frame.plane_fragments, ())
        intervals = _line_outline_scalar_intervals(frame)
        self.assertTrue(intervals)
        start = np.asarray(frame.patch_projection.line_screen_start, dtype=float)
        end = np.asarray(frame.patch_projection.line_screen_end, dtype=float)
        length = float(np.linalg.norm(end - start))
        tolerance = max(1.0e-10, frame.max_screen_error * 1.0e-9)
        cursor = 0.0
        for lower, upper in intervals:
            self.assertLessEqual(lower, cursor + tolerance)
            self.assertGreaterEqual(lower, cursor - tolerance)
            self.assertGreater(upper, lower)
            cursor = upper
        self.assertAlmostEqual(cursor, length, delta=tolerance)

    def test_two_local_frames_merge_into_one_area_conserving_plane_partition(
        self,
    ) -> None:
        frame = _renderer_neutral_frame(_double(), _side_plane())
        self.assertIs(frame.projection_kind, PlanePatchProjectionKind.AREA)
        self.assertTrue(frame.has_plane_fill)
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

    def test_apex_plane_merges_two_line_children_without_plane_fill(self) -> None:
        frame = _renderer_neutral_frame(
            _double("apex-line-double"),
            _side_plane(0.0),
            EDGE_ON_VIEW,
        )

        self._assert_finite_non_overlapping_line_chain(frame)
        self.assertEqual(len(frame.branch_lineage), 4)
        self.assertEqual(frame.shared_apex.contact_dimension, 0)
        self.assertEqual(frame.shared_apex.contact_extent, 0.0)
        self.assertEqual(
            {item.role for item in frame.plane_outline_fragments},
            {
                PlaneDepthRole.OUTSIDE_PROJECTION,
                PlaneDepthRole.IN_FRONT_OF_SURFACE,
            },
        )
        first = canonical_composite_quadric_section_compositing_json(frame)
        second = canonical_composite_quadric_section_compositing_json(frame)
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(
            payload["schema"],
            COMPOSITE_QUADRIC_SECTION_COMPOSITING_SCHEMA,
        )
        self.assertEqual(
            payload["schema"],
            "manim-composite-quadric-section-compositing/v3",
        )
        self.assertEqual(payload["patchProjection"]["kind"], "line")
        self.assertEqual(payload["projectionKind"], "line")
        self.assertFalse(payload["hasPlaneFill"])
        self.assertEqual(payload["planeFragments"], [])

    def test_offset_hyperbola_plane_retains_two_physical_line_branches(self) -> None:
        frame = _renderer_neutral_frame(
            _double("offset-line-double"),
            _side_plane(0.5),
            EDGE_ON_VIEW,
        )

        self._assert_finite_non_overlapping_line_chain(frame)
        self.assertEqual(len(frame.branch_lineage), 2)
        self.assertEqual(
            {item.nappe_role for item in frame.branch_lineage},
            {"negative", "positive"},
        )
        self.assertTrue(
            any(
                item.role is PlaneDepthRole.OUTSIDE_PROJECTION
                for item in frame.plane_outline_fragments
            )
        )
        self.assertTrue(
            any(
                item.role is PlaneDepthRole.IN_FRONT_OF_SURFACE
                for item in frame.plane_outline_fragments
            )
        )

    def test_oblique_in_plane_view_merges_adjacent_near_patch_edges(self) -> None:
        frame = _renderer_neutral_frame(
            _double("oblique-line-double"),
            _side_plane(0.0),
            _edge_on_side_plane_view(0.2),
        )

        self._assert_finite_non_overlapping_line_chain(frame)
        self.assertEqual(
            {item.edge_index for item in frame.plane_outline_fragments},
            {0, 1},
        )
        start = np.asarray(frame.patch_projection.line_screen_start, dtype=float)
        end = np.asarray(frame.patch_projection.line_screen_end, dtype=float)
        direction = end - start
        axis = direction / np.linalg.norm(direction)
        normal = np.asarray((-axis[1], axis[0]), dtype=float)
        self.assertLessEqual(
            max(
                abs(
                    float(
                        np.dot(
                            np.asarray(point, dtype=float) - start,
                            normal,
                        )
                    )
                )
                for fragment in frame.plane_outline_fragments
                for point in (fragment.screen_start, fragment.screen_end)
            ),
            1.0e-10,
        )

    def test_line_path_still_rejects_nonzero_shared_proxy_segment(self) -> None:
        cone = _double("line-contact-double")
        ordinary = _renderer_neutral_frame(
            cone,
            _side_plane(0.0),
            EDGE_ON_VIEW,
        )
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
                "line-contact-section",
                frames,
            )

    def test_line_children_must_share_canonical_endpoint_evidence(self) -> None:
        cone = _double("line-endpoint-double")
        ordinary = _renderer_neutral_frame(
            cone,
            _side_plane(0.0),
            EDGE_ON_VIEW,
        )
        second = ordinary.child_frames[1]
        evidence = second.patch_projection
        reversed_evidence = replace(
            evidence,
            line_screen_start=evidence.line_screen_end,
            line_screen_end=evidence.line_screen_start,
        )
        reversed_second = replace(
            second,
            patch_projection=reversed_evidence,
        )

        with self.assertRaisesRegex(
            CompositeQuadricSectionCompositingError,
            "disagree on finite projection endpoints",
        ):
            compute_composite_quadric_section_compositing(
                cone,
                "line-endpoint-section",
                (ordinary.child_frames[0], reversed_second),
            )

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


@unittest.skipUnless(MANIM_AVAILABLE, "manim is not installed")
class CompositeManimBindingTests(unittest.TestCase):
    def test_hidden_plane_outline_spans_name_the_actual_child_occluder(
        self,
    ) -> None:
        frame = _renderer_neutral_frame(
            _double("outline-occluder-double"),
            _side_plane(),
            _view_at_axis_angle(0.8),
        )
        spans = CompositeQuadricSection3D._plane_outline_visibility(frame)
        hidden = tuple(
            span
            for values in spans.values()
            for span in values
            if span.kind is VisibilityKind.HIDDEN
        )
        child_ids = {item.surface_id for item in frame.child_frames}
        self.assertTrue(hidden)
        self.assertTrue(
            all(
                len(span.occluder_surface_ids) == 1
                and set(span.occluder_surface_ids) <= child_ids
                for span in hidden
            )
        )

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

    def test_apply_failure_restores_composite_sparse_display_transaction(
        self,
    ) -> None:
        with tempconfig({"renderer": "cairo"}):
            controller = CompositeQuadricSection3D(
                Scene(),
                surface=_double("apply-rollback-double"),
                section_id="apply-rollback-section",
                plane=_side_plane(),
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
                maps = {
                    key: dict(value)
                    for key, value in controller._fragment_slot_maps.items()
                }
                prepared = controller.prepare()
                shifted_items = (
                    replace(
                        prepared.painter_band.items[0],
                        z_index=prepared.painter_band.items[0].z_index + 0.125,
                    ),
                    *prepared.painter_band.items[1:],
                )
                prepared = replace(
                    prepared,
                    painter_band=replace(
                        prepared.painter_band,
                        items=shifted_items,
                    ),
                )
                original_apply = controller._band.apply

                def fail_after_commit(value) -> None:
                    original_apply(value)
                    raise RuntimeError("synthetic composite painter failure")

                with patch.object(
                    controller._band,
                    "apply",
                    side_effect=fail_after_commit,
                ):
                    with self.assertRaisesRegex(RuntimeError, "synthetic composite"):
                        controller.apply(prepared)

                self.assertEqual(controller.slot_snapshot(), snapshot)
                self.assertEqual(controller.slot_identities(), identities)
                self.assertIs(controller.last_composite_frame, frame)
                self.assertIs(controller.last_boundary_frame, boundary)
                self.assertEqual(controller.active_painter_z_indices, z_indices)
                self.assertEqual(controller._fragment_slot_maps, maps)
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
