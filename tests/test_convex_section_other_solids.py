from __future__ import annotations

import unittest

import numpy as np

from examples.convex_sections.other_convex_solids_demo import (
    CONVEX_SOLIDS,
    SECTION_NORMAL,
    incident_faces,
    surface_edges,
    visibility_payload,
)
from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.sections import (
    SectionPlane3D,
    intersect_plane_with_convex_polyhedron,
    intersect_segment_with_convex_polyhedron,
)


def _section_plane(offset: float = 0.0) -> SectionPlane3D:
    normal = np.asarray(SECTION_NORMAL, dtype=float)
    u_axis = np.cross(normal, np.asarray((0.0, 0.0, 1.0)))
    u_axis /= np.linalg.norm(u_axis)
    return SectionPlane3D(
        "test-cut",
        tuple(float(value) for value in normal * offset),
        tuple(float(value) for value in normal),
        3.0,
        3.0,
        u_axis=tuple(float(value) for value in u_axis),
    )


class ConvexSectionOtherSolidsTests(unittest.TestCase):
    def test_all_four_models_are_strict_closed_convex_manifolds(self) -> None:
        expected_edge_counts = {
            "tetrahedron": 6,
            "triangular-prism": 9,
            "square-pyramid": 8,
            "octahedron": 12,
        }
        for solid_id, spec in CONVEX_SOLIDS.items():
            with self.subTest(solid=solid_id):
                model = VisibilityModel.from_dict(visibility_payload(spec))
                model.validate(require_closed_convex_manifold=True)
                edges = surface_edges(spec.faces)
                self.assertEqual(len(edges), expected_edge_counts[solid_id])
                for start, end in edges:
                    self.assertEqual(
                        len(incident_faces(spec.faces, start, end)),
                        2,
                        f"{solid_id}:{start}-{end}",
                    )

    def test_center_plane_produces_shape_specific_sections(self) -> None:
        for solid_id, spec in CONVEX_SOLIDS.items():
            with self.subTest(solid=solid_id):
                model = VisibilityModel.from_dict(
                    visibility_payload(spec, include_probe=False)
                )
                section = intersect_plane_with_convex_polyhedron(
                    "center", model, _section_plane()
                )
                self.assertEqual(section.kind, "polygon")
                self.assertEqual(
                    len(section.points), spec.expected_center_section_vertices
                )
                self.assertEqual(
                    len(section.boundary_segments),
                    spec.expected_center_section_vertices,
                )

    def test_plane_enters_and_leaves_every_solid_without_stale_geometry(self) -> None:
        for solid_id, spec in CONVEX_SOLIDS.items():
            with self.subTest(solid=solid_id):
                model = VisibilityModel.from_dict(
                    visibility_payload(spec, include_probe=False)
                )
                support = max(
                    abs(float(np.dot(SECTION_NORMAL, np.asarray(point))))
                    for point in spec.vertices.values()
                )
                before = intersect_plane_with_convex_polyhedron(
                    "before", model, _section_plane(support + 0.2)
                )
                middle = intersect_plane_with_convex_polyhedron(
                    "middle", model, _section_plane(0.0)
                )
                after = intersect_plane_with_convex_polyhedron(
                    "after", model, _section_plane(-support - 0.2)
                )
                self.assertEqual(before.kind, "empty")
                self.assertEqual(middle.kind, "polygon")
                self.assertEqual(after.kind, "empty")

    def test_free_line_crosses_every_solid_at_two_computed_points(self) -> None:
        for solid_id, spec in CONVEX_SOLIDS.items():
            with self.subTest(solid=solid_id):
                model = VisibilityModel.from_dict(
                    visibility_payload(spec, include_probe=False)
                )
                crossing = intersect_segment_with_convex_polyhedron(
                    model, (-2.4, 0.0, 0.0), (2.4, 0.0, 0.0)
                )
                self.assertEqual(crossing.kind, "segment")
                self.assertEqual(len(crossing.hits), 2)
                self.assertIsNotNone(crossing.inside_parameter_interval)
                start, end = crossing.inside_parameter_interval or (0.0, 0.0)
                self.assertGreater(start, 0.0)
                self.assertLess(end, 1.0)
                self.assertLess(start, end)


if __name__ == "__main__":
    unittest.main()
