from __future__ import annotations

from math import acos, pi, sqrt
import unittest

import numpy as np

from polyhedron_visibility.geometry import resolve_geometry_context
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_section import (
    BoundaryPlaneRelation,
    QuadricBoundarySectionLimits,
    _projected_segment_intersection_parameters,
    compute_boundary_section_spans,
)
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
    QuadricBoundaryCompositingError,
    compute_boundary_visibility,
)
from polyhedron_visibility.quadrics.conics import ConicKind, ConicParameterization
from polyhedron_visibility.quadrics.compositing import compute_quadric_compositing
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.curve_intersections import (
    compute_projected_curve_crossings,
)
from polyhedron_visibility.quadrics.curves import (
    CircleArcCurve,
    ParametricConicBranch,
    SegmentCurve,
)
from polyhedron_visibility.quadrics.projection import build_opaque_projection_proxy
from polyhedron_visibility.quadrics.section_compositing import (
    compute_quadric_section_compositing,
)
from polyhedron_visibility.quadrics.sections import (
    QuadricSectionError,
    compute_quadric_section,
    compute_quadric_section_boundary_curves,
    compute_section_cap_chord_curves,
)
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.surface_boundaries import (
    build_surface_boundary_sources,
    curve_boundary_source,
    plane_outline_sources,
    section_curve_boundary_source,
)
from polyhedron_visibility.quadrics.visibility import compute_quadric_visibility
from polyhedron_visibility.quadrics.trace import section_trace_curves
from polyhedron_visibility.topology import ParameterInterval


VIEW = ParallelView.from_matrix(
    (
        (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
        (-1.0 / sqrt(6.0), -1.0 / sqrt(6.0), 2.0 / sqrt(6.0)),
        (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    )
)
SIDE_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)
FRONT_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
)


def _cylinder_section_case():
    cylinder = CylinderSpec(
        "cylinder",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
        (0.0, 2.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    plane = SectionPlane(
        "cut",
        (0.0, 0.5, 1.0),
        (0.0, 1.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    patch = PlaneDisplayPatchSpec("patch", "cut", 1.4, 1.4)
    proxy = build_opaque_projection_proxy(
        cylinder, SIDE_VIEW, max_chord_error=0.002
    )
    base = compute_quadric_compositing(
        compute_quadric_visibility((), (cylinder,), SIDE_VIEW),
        (proxy,),
    )
    section = compute_quadric_section_compositing(
        base,
        cylinder,
        plane,
        patch,
        SIDE_VIEW,
        max_screen_error=0.03,
    )
    return build_surface_boundary_sources((cylinder,), SIDE_VIEW), section


class BoundarySectionPlacementTests(unittest.TestCase):
    def test_rank_one_plane_has_no_boundary_placement_or_occlusion(self) -> None:
        sphere = SphereSpec("line-sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "line-plane",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            u_axis=(0.0, 0.0, 1.0),
        )
        patch = fit_plane_display_patch(
            "line-patch",
            plane,
            (sphere,),
        ).patch
        proxy = build_opaque_projection_proxy(
            sphere,
            FRONT_VIEW,
            max_chord_error=0.002,
        )
        base = compute_quadric_compositing(
            compute_quadric_visibility((), (sphere,), FRONT_VIEW),
            (proxy,),
        )
        section = compute_quadric_section_compositing(
            base,
            sphere,
            plane,
            patch,
            FRONT_VIEW,
        )
        source = curve_boundary_source(
            SegmentCurve(
                "unrelated-boundary",
                (-1.5, -0.5, 0.25),
                (1.5, 0.5, 0.25),
            )
        )

        self.assertEqual(
            compute_boundary_section_spans(
                (source,),
                section,
                FRONT_VIEW,
            ),
            {},
        )

    def test_side_view_circle_keeps_screen_coincident_depth_events(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "plane",
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        # The patch spans x in [-2, 0.9].  Its other three edges do not meet
        # the projected circle line y=3, while the rectangle still overlaps
        # the sphere projection as required by the section compositor.
        patch = PlaneDisplayPatchSpec(
            "patch",
            plane.plane_id,
            1.45,
            2.2,
            center_coordinates=(-0.55, 1.4),
        )
        proxy = build_opaque_projection_proxy(
            sphere,
            FRONT_VIEW,
            max_chord_error=0.002,
        )
        base = compute_quadric_compositing(
            compute_quadric_visibility((), (sphere,), FRONT_VIEW),
            (proxy,),
        )
        section = compute_quadric_section_compositing(
            base,
            sphere,
            plane,
            patch,
            FRONT_VIEW,
            max_screen_error=0.03,
        )
        curve = CircleArcCurve(
            "side-view-circle",
            (0.0, 3.0, 0.0),
            1.0,
            (0.0, -1.0, 0.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        source = curve_boundary_source(curve)

        spans = compute_boundary_section_spans(
            (source,),
            section,
            FRONT_VIEW,
        )[source.source_id]

        first = acos(0.9)
        second = 2.0 * pi - first
        first_world = np.asarray(curve.point(first), dtype=float)
        second_world = np.asarray(curve.point(second), dtype=float)
        np.testing.assert_allclose(
            FRONT_VIEW.matrix[:2] @ first_world,
            FRONT_VIEW.matrix[:2] @ second_world,
            atol=1.0e-12,
        )
        self.assertGreater(
            abs(
                float(
                    np.dot(
                        first_world - second_world,
                        FRONT_VIEW.view_direction,
                    )
                )
            ),
            0.8,
        )
        self.assertEqual(len(spans), 3)
        expected_boundaries = (0.0, first, second, 2.0 * pi)
        for span, start, end in zip(
            spans,
            expected_boundaries,
            expected_boundaries[1:],
        ):
            self.assertAlmostEqual(span.interval.start, start, places=10)
            self.assertAlmostEqual(span.interval.end, end, places=10)
        self.assertEqual(
            tuple((item.relation, item.plane_depth_roles) for item in spans),
            (
                (BoundaryPlaneRelation.OUTSIDE_PATCH, ()),
                (
                    BoundaryPlaneRelation.BOUNDARY_BEHIND_PLANE,
                    ("outside_projection",),
                ),
                (BoundaryPlaneRelation.OUTSIDE_PATCH, ()),
            ),
        )

    def test_degenerate_conic_overlap_is_split_at_finite_segment_ends(self) -> None:
        curve = ParametricConicBranch(
            "line",
            ConicParameterization(
                kind=ConicKind.COINCIDENT_LINE,
                branch_label="line",
                origin=(0.0, 0.0),
                first_axis=(1.0, 0.0),
            ),
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            ParameterInterval(-2.0, 2.0),
        )
        source = curve_boundary_source(curve)
        context = resolve_geometry_context(
            None,
            positions=(curve.point(-2.0), curve.point(2.0)),
        )

        parameters = _projected_segment_intersection_parameters(
            source,
            (-0.5, 0.0),
            (0.5, 0.0),
            SIDE_VIEW,
            context,
        )

        self.assertEqual(len(parameters), 2)
        self.assertAlmostEqual(parameters[0], -0.5)
        self.assertAlmostEqual(parameters[1], 0.5)

    def test_boundary_section_limits_reject_invalid_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            QuadricBoundarySectionLimits(max_role_boundary_segments=True)

    def test_boundary_section_capacity_fails_explicitly(self) -> None:
        sources, section = _cylinder_section_case()
        with self.assertRaisesRegex(
            QuadricBoundaryCompositingError,
            "max_role_boundary_segments=1",
        ):
            compute_boundary_section_spans(
                sources,
                section,
                SIDE_VIEW,
                limits=QuadricBoundarySectionLimits(
                    max_role_boundary_segments=1
                ),
            )
        with self.assertRaisesRegex(
            QuadricBoundaryCompositingError,
            "max_split_parameters_per_source=1",
        ):
            compute_boundary_section_spans(
                sources,
                section,
                SIDE_VIEW,
                limits=QuadricBoundarySectionLimits(
                    max_split_parameters_per_source=1
                ),
            )

    def test_exact_section_curve_does_not_fragment_at_mesh_chords(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        patch = PlaneDisplayPatchSpec("patch", "cut", 1.25, 1.25)
        curves = section_trace_curves(
            compute_quadric_section("section", sphere, plane)
        )
        self.assertEqual(len(curves), 1)
        sources = tuple(curve_boundary_source(curve) for curve in curves)
        proxy = build_opaque_projection_proxy(
            sphere, VIEW, max_chord_error=0.002
        )
        base = compute_quadric_compositing(
            compute_quadric_visibility((), (sphere,), VIEW),
            (proxy,),
        )
        section = compute_quadric_section_compositing(
            base,
            sphere,
            plane,
            patch,
            VIEW,
            max_screen_error=0.03,
        )
        visibility = compute_boundary_visibility(sources, (sphere,), VIEW)

        spans = compute_boundary_section_spans(
            sources,
            section,
            VIEW,
            surface=sphere,
            visibility_spans_by_source=visibility,
        )[sources[0].source_id]

        # A closed conic may express one wrapped side as two authored
        # intervals around the stable parameter seam, but mesh density must
        # not create any additional fragments.
        self.assertLessEqual(len(spans), 3)
        self.assertEqual(
            {item.plane_depth_roles for item in spans},
            {
                ("behind_surface", "between_surface_sheets"),
                ("between_surface_sheets", "in_front_of_surface"),
            },
        )

    def test_cap_chord_is_certified_as_an_exact_section_boundary(self) -> None:
        cone = ConeSpec(
            "closed-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 1.5),
            (0.5, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "patch", plane, (cone,), margin_ratio=0.08
        ).patch
        curves = compute_quadric_section_boundary_curves(
            "finite-section",
            cone,
            plane,
        )
        sources = tuple(
            section_curve_boundary_source(curve, cone, plane)
            for curve in curves
        )
        chord_source = next(
            item
            for item in sources
            if isinstance(item.curve, SegmentCurve)
        )
        self.assertIs(
            chord_source.source_kind,
            BoundarySourceKind.SECTION_CAP_CHORD,
        )
        self.assertEqual(chord_source.owner_id, cone.end_caps[0].cap_id)

        proxy = build_opaque_projection_proxy(
            cone,
            VIEW,
            max_chord_error=0.002,
        )
        base = compute_quadric_compositing(
            compute_quadric_visibility(curves, (cone,), VIEW),
            (proxy,),
        )
        section = compute_quadric_section_compositing(
            base,
            cone,
            plane,
            patch,
            VIEW,
            max_screen_error=0.03,
        )
        visibility = compute_boundary_visibility(sources, (cone,), VIEW)
        spans = compute_boundary_section_spans(
            sources,
            section,
            VIEW,
            surface=cone,
            visibility_spans_by_source=visibility,
        )[chord_source.source_id]

        self.assertTrue(spans)
        self.assertEqual(
            {item.relation for item in spans},
            {BoundaryPlaneRelation.COINCIDENT},
        )
        self.assertTrue(
            all(
                item.plane_depth_roles
                in {
                    ("behind_surface", "between_surface_sheets"),
                    ("between_surface_sheets", "in_front_of_surface"),
                }
                for item in spans
            )
        )

        forged = curve_boundary_source(
            SegmentCurve(
                chord_source.source_id,
                chord_source.curve.start,
                (0.0, 0.0, 2.0),
            ),
            source_kind=BoundarySourceKind.SECTION_CAP_CHORD,
            owner_id=cone.end_caps[0].cap_id,
        )
        forged_visibility = compute_boundary_visibility(
            (forged,),
            (cone,),
            VIEW,
        )
        with self.assertRaisesRegex(
            QuadricBoundaryCompositingError,
            "does not end on",
        ):
            compute_boundary_section_spans(
                (forged,),
                section,
                VIEW,
                surface=cone,
                visibility_spans_by_source=forged_visibility,
            )

    def test_cap_chord_suffix_collision_remains_an_ordinary_curve(self) -> None:
        cone = ConeSpec(
            "collision-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "collision-cut",
            (0.0, 0.0, 1.5),
            (0.5, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        ordinary = SegmentCurve(
            "ordinary-feature:cap:cap_max:chord",
            (-0.25, -0.25, 0.5),
            (0.25, 0.25, 0.5),
        )
        source = section_curve_boundary_source(ordinary, cone, plane)
        self.assertIs(source.source_kind, BoundarySourceKind.ANALYTIC_CURVE)
        self.assertEqual(source.owner_id, ordinary.curve_id)

    def test_stale_cap_chord_is_not_downgraded_to_a_free_curve(self) -> None:
        cone = ConeSpec(
            "stale-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        cap = cone.end_caps[0]
        previous_plane = SectionPlane(
            "previous-cut",
            (0.0, 0.0, 1.5),
            (0.5, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        stale = compute_section_cap_chord_curves(
            "stale-section",
            cone,
            previous_plane,
        )[0]
        unresolved_plane = SectionPlane(
            "unresolved-cut",
            cap.center,
            (1.0e-15, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        with self.assertRaisesRegex(
            QuadricSectionError,
            "below the configured angular resolution",
        ):
            section_curve_boundary_source(stale, cone, unresolved_plane)

    def test_cap_source_reuses_complete_boundary_topology_check(self) -> None:
        cone = ConeSpec(
            "topology-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        cap = cone.end_caps[0]
        tolerance_mismatch_plane = SectionPlane(
            "topology-mismatch-cut",
            cap.center,
            (1.0e-9, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        standalone_chord = compute_section_cap_chord_curves(
            "topology-section",
            cone,
            tolerance_mismatch_plane,
        )[0]
        with self.assertRaisesRegex(
            QuadricSectionError,
            "lateral and cap clipping disagree",
        ):
            section_curve_boundary_source(
                standalone_chord,
                cone,
                tolerance_mismatch_plane,
            )

    def test_cap_rim_splits_at_every_plane_depth_role_boundary(self) -> None:
        sources, section = _cylinder_section_case()
        spans = compute_boundary_section_spans(sources, section, SIDE_VIEW)
        cap = next(
            item for item in sources if item.source_id.endswith("cap_max:rim")
        )
        cap_spans = spans[cap.source_id]

        expected_boundaries = (
            0.0,
            pi / 6.0,
            5.0 * pi / 6.0,
            7.0 * pi / 6.0,
            11.0 * pi / 6.0,
            2.0 * pi,
        )
        self.assertEqual(len(cap_spans), 5)
        for span, start, end in zip(
            cap_spans, expected_boundaries, expected_boundaries[1:]
        ):
            self.assertAlmostEqual(span.interval.start, start, places=8)
            self.assertAlmostEqual(span.interval.end, end, places=8)
        self.assertEqual(
            tuple(item.plane_depth_roles for item in cap_spans),
            (
                ("in_front_of_surface", "outside_projection"),
                ("between_surface_sheets", "outside_projection"),
                ("in_front_of_surface", "outside_projection"),
                ("between_surface_sheets", "outside_projection"),
                ("in_front_of_surface", "outside_projection"),
            ),
        )

    def test_surface_boundaries_split_at_plane_and_patch_events(self) -> None:
        cone = ConeSpec(
            "cone",
            (0.0, 0.0, -2.4),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, -0.4),
            (0.8, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "patch", plane, (cone,), margin_ratio=0.08
        ).patch
        proxy = build_opaque_projection_proxy(cone, VIEW, max_chord_error=0.002)
        base = compute_quadric_compositing(
            compute_quadric_visibility((), (cone,), VIEW),
            (proxy,),
        )
        section = compute_quadric_section_compositing(
            base, cone, plane, patch, VIEW, max_screen_error=0.08
        )
        surface_sources = build_surface_boundary_sources((cone,), VIEW)
        plane_sources = plane_outline_sources(plane, patch)
        sources = (*surface_sources, *plane_sources)
        crossings = tuple(
            item
            for first in surface_sources
            for second in plane_sources
            for item in compute_projected_curve_crossings(
                (first.curve, second.curve), VIEW
            )
        )
        spans = compute_boundary_section_spans(
            sources, section, VIEW, crossings
        )
        self.assertEqual(set(spans), {item.source_id for item in surface_sources})
        self.assertTrue(
            any(
                item.relation is BoundaryPlaneRelation.BOUNDARY_BEHIND_PLANE
                for values in spans.values()
                for item in values
            )
        )
        self.assertTrue(
            any(
                item.relation is BoundaryPlaneRelation.BOUNDARY_IN_FRONT_OF_PLANE
                for values in spans.values()
                for item in values
            )
        )
        for source in surface_sources:
            values = spans[source.source_id]
            self.assertAlmostEqual(values[0].interval.start, source.curve.domain.start)
            self.assertAlmostEqual(values[-1].interval.end, source.curve.domain.end)
            for left, right in zip(values, values[1:]):
                self.assertAlmostEqual(left.interval.end, right.interval.start)


if __name__ == "__main__":
    unittest.main()
