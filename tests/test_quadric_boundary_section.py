from __future__ import annotations

from math import pi, sqrt
import unittest

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_section import (
    BoundaryPlaneRelation,
    compute_boundary_section_spans,
)
from polyhedron_visibility.quadrics.compositing import compute_quadric_compositing
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.curve_intersections import (
    compute_projected_curve_crossings,
)
from polyhedron_visibility.quadrics.projection import build_opaque_projection_proxy
from polyhedron_visibility.quadrics.section_compositing import (
    compute_quadric_section_compositing,
)
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.surface_boundaries import (
    build_surface_boundary_sources,
    plane_outline_sources,
)
from polyhedron_visibility.quadrics.visibility import compute_quadric_visibility


VIEW = ParallelView.from_matrix(
    (
        (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
        (-1.0 / sqrt(6.0), -1.0 / sqrt(6.0), 2.0 / sqrt(6.0)),
        (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    )
)


class BoundarySectionPlacementTests(unittest.TestCase):
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
