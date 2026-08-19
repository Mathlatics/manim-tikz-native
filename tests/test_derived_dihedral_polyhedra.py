from __future__ import annotations

import unittest

import numpy as np

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.dihedral_extraction import (
    BasePlaneRotation3D,
    DerivedDihedralModel,
    RigidTransform3D,
    compute_derived_dihedral_transparent_compositing,
    compute_derived_dihedral_visibility,
)

from tests.test_derived_dihedral_manim import isometric_projection


def closed_model(
    group_id: str,
    points: dict[str, tuple[float, float, float]],
    raw_faces: dict[str, tuple[str, ...]],
) -> VisibilityModel:
    center = np.mean([points[key] for key in sorted(points)], axis=0)
    faces: dict[str, tuple[str, ...]] = {}
    for face_id, raw_vertex_ids in raw_faces.items():
        vertex_ids = list(raw_vertex_ids)
        vertices = np.asarray([points[item] for item in vertex_ids], dtype=float)
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        if float(np.dot(normal, np.mean(vertices, axis=0) - center)) < 0:
            vertex_ids.reverse()
        faces[face_id] = tuple(vertex_ids)

    edge_faces: dict[tuple[str, str], list[str]] = {}
    for face_id, vertex_ids in faces.items():
        for index, start in enumerate(vertex_ids):
            end = vertex_ids[(index + 1) % len(vertex_ids)]
            edge_faces.setdefault(tuple(sorted((start, end))), []).append(face_id)
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": group_id,
            "vertices": [
                {"vertexId": vertex_id, "entryPosition": points[vertex_id]}
                for vertex_id in sorted(points)
            ],
            "faces": [
                {"faceId": face_id, "vertexIds": list(faces[face_id])}
                for face_id in sorted(faces)
            ],
            "strokes": [
                {
                    "sourceEdgeId": f"edge.{start}.{end}",
                    "vertexIds": [start, end],
                    "incidentFaceIds": sorted(owner_ids),
                }
                for (start, end), owner_ids in sorted(edge_faces.items())
            ],
        }
    )


def rectangular_box_model() -> VisibilityModel:
    return closed_model(
        "rectangular-box",
        {
            "A": (-1.6, -1.0, -0.8),
            "B": (1.6, -1.0, -0.8),
            "C": (1.6, 1.0, -0.8),
            "D": (-1.6, 1.0, -0.8),
            "E": (-1.6, -1.0, 0.8),
            "F": (1.6, -1.0, 0.8),
            "G": (1.6, 1.0, 0.8),
            "H": (-1.6, 1.0, 0.8),
        },
        {
            "back": ("A", "B", "C", "D"),
            "front": ("E", "F", "G", "H"),
            "bottom": ("A", "B", "F", "E"),
            "right": ("B", "C", "G", "F"),
            "top": ("D", "C", "G", "H"),
            "left": ("A", "D", "H", "E"),
        },
    )


def tetrahedron_model() -> VisibilityModel:
    return closed_model(
        "tetrahedron",
        {
            "A": (1.25, 1.05, 1.0),
            "B": (-1.3, -1.0, 0.95),
            "C": (-1.05, 1.25, -1.0),
            "D": (1.15, -1.15, -0.95),
        },
        {
            "ABC": ("A", "B", "C"),
            "ABD": ("A", "B", "D"),
            "ACD": ("A", "C", "D"),
            "BCD": ("B", "C", "D"),
        },
    )


def square_pyramid_model() -> VisibilityModel:
    return closed_model(
        "square-pyramid",
        {
            "A": (-1.25, -1.05, -0.75),
            "B": (1.25, -1.05, -0.75),
            "C": (1.25, 1.05, -0.75),
            "D": (-1.25, 1.05, -0.75),
            "S": (0.0, 0.0, 1.65),
        },
        {
            "base": ("A", "B", "C", "D"),
            "side.AB": ("A", "B", "S"),
            "side.BC": ("B", "C", "S"),
            "side.CD": ("C", "D", "S"),
            "side.DA": ("D", "A", "S"),
        },
    )


class DerivedDihedralPolyhedraTests(unittest.TestCase):
    CASES = (
        (
            "rectangular-box",
            rectangular_box_model,
            ("front", "top"),
            "right",
            (0.45, 0.15, -0.55),
        ),
        (
            "tetrahedron",
            tetrahedron_model,
            ("ABC", "ABD"),
            "ACD",
            (0.4, -0.35, -0.45),
        ),
        (
            "square-pyramid",
            square_pyramid_model,
            ("side.AB", "side.BC"),
            "side.CD",
            (0.55, -0.2, -0.45),
        ),
    )

    def test_three_convex_solids_support_identity_handoff_and_moved_copy(self) -> None:
        for label, factory, source_faces, _base_face, translation in self.CASES:
            with self.subTest(label=label):
                model = DerivedDihedralModel.from_solid(
                    f"{label}-extraction",
                    factory(),
                    entity_id="copy",
                    source_face_ids=source_faces,
                )
                identity = compute_derived_dihedral_visibility(
                    model,
                    transform=RigidTransform3D.identity(),
                    projection_matrix=isometric_projection(),
                )
                self.assertEqual(
                    set(identity.coincident_source_face_ids),
                    set(source_faces),
                )
                self.assertTrue(identity.suppressed_source_stroke_ids)

                moved = compute_derived_dihedral_visibility(
                    model,
                    transform=RigidTransform3D.translation_by(translation),
                    projection_matrix=isometric_projection(),
                )
                self.assertEqual(moved.coincident_source_face_ids, ())
                self.assertEqual(
                    len(moved.line_visibility.edges),
                    len(model.overlay_model().strokes),
                )

                transparent = compute_derived_dihedral_transparent_compositing(
                    model,
                    transform=RigidTransform3D.translation_by(translation),
                    projection_matrix=isometric_projection(),
                )
                self.assertEqual(
                    set(transparent.draw_order),
                    set(transparent.fragment_map),
                )
                self.assertTrue(transparent.fragments)

    def test_three_solids_keep_copy_synchronized_while_base_face_turns_down(self) -> None:
        for label, factory, source_faces, base_face_id, translation in self.CASES:
            with self.subTest(label=label):
                solid = factory()
                model = DerivedDihedralModel.from_solid(
                    f"{label}-base-plane",
                    solid,
                    entity_id="copy",
                    source_face_ids=source_faces,
                )
                base_motion = BasePlaneRotation3D.from_model(
                    solid,
                    base_face_id,
                )
                global_transform = base_motion.final_transform()
                solid_positions = {
                    vertex_id: global_transform.apply(vertex.entry_position)
                    for vertex_id, vertex in solid.vertex_map.items()
                }
                local_transform = RigidTransform3D.translation_by(translation)
                copy_transform = local_transform.compose(global_transform)
                solid_center = np.mean(
                    [
                        np.asarray(vertex.entry_position, dtype=float)
                        for vertex in solid.vertex_map.values()
                    ],
                    axis=0,
                )
                np.testing.assert_allclose(
                    copy_transform.apply(solid_center),
                    local_transform.apply(solid_center),
                    atol=1.0e-12,
                )
                frame = compute_derived_dihedral_visibility(
                    model,
                    transform=copy_transform,
                    projection_matrix=isometric_projection(),
                    solid_vertex_positions=solid_positions,
                )

                base_face = solid.face_map[base_face_id]
                points = np.asarray(
                    [solid_positions[item] for item in base_face.vertex_ids]
                )
                normal = np.cross(
                    points[1] - points[0], points[2] - points[0]
                )
                normal /= np.linalg.norm(normal)
                np.testing.assert_allclose(
                    normal,
                    (0.0, 0.0, -1.0),
                    atol=1.0e-12,
                )
                self.assertEqual(frame.coincident_source_face_ids, ())
                self.assertEqual(
                    len(frame.line_visibility.edges),
                    len(model.overlay_model().strokes),
                )
                transparent = compute_derived_dihedral_transparent_compositing(
                    model,
                    transform=copy_transform,
                    projection_matrix=isometric_projection(),
                    solid_vertex_positions=solid_positions,
                )
                self.assertEqual(
                    set(transparent.draw_order),
                    set(transparent.fragment_map),
                )


if __name__ == "__main__":
    unittest.main()
