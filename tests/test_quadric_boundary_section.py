from __future__ import annotations

from math import pi, sqrt
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
from polyhedron_visibility.quadrics.curves import ParametricConicBranch
from polyhedron_visibility.quadrics.projection import build_opaque_projection_proxy
from polyhedron_visibility.quadrics.section_compositing import (
    compute_quadric_section_compositing,
)
from polyhedron_visibility.quadrics.sections import compute_quadric_section
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.surface_boundaries import (
    build_surface_boundary_sources,
    curve_boundary_source,
    plane_outline_sources,
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
