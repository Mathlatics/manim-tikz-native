from __future__ import annotations

from math import pi
import unittest

import numpy as np

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.dihedral_extraction import (
    BASE_PLANE_ROTATION_SCHEMA,
    DERIVED_DIHEDRAL_MODEL_SCHEMA,
    BasePlaneRotation3D,
    DerivedDihedralContractError,
    DerivedDihedralModel,
    RigidTransform3D,
    canonical_derived_dihedral_trace_json,
    compute_derived_dihedral_visibility,
)


def cube_model() -> VisibilityModel:
    points = {
        "A": (-1, -1, -1),
        "B": (1, -1, -1),
        "C": (1, 1, -1),
        "D": (-1, 1, -1),
        "E": (-1, -1, 1),
        "F": (1, -1, 1),
        "G": (1, 1, 1),
        "H": (-1, 1, 1),
    }
    faces = {
        "back": ("A", "D", "C", "B"),
        "front": ("E", "F", "G", "H"),
        "bottom": ("A", "B", "F", "E"),
        "right": ("B", "C", "G", "F"),
        "top": ("D", "H", "G", "C"),
        "left": ("A", "E", "H", "D"),
    }
    edge_faces: dict[tuple[str, str], list[str]] = {}
    for face_id, vertex_ids in faces.items():
        for index, start in enumerate(vertex_ids):
            end = vertex_ids[(index + 1) % len(vertex_ids)]
            edge_faces.setdefault(tuple(sorted((start, end))), []).append(face_id)
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "cube",
            "vertices": [
                {"vertexId": vertex_id, "entryPosition": point}
                for vertex_id, point in points.items()
            ],
            "faces": [
                {"faceId": face_id, "vertexIds": list(vertex_ids)}
                for face_id, vertex_ids in faces.items()
            ],
            "strokes": [
                {
                    "sourceEdgeId": f"edge.{start}.{end}",
                    "vertexIds": [start, end],
                    "incidentFaceIds": sorted(owners),
                }
                for (start, end), owners in sorted(edge_faces.items())
            ],
        }
    )


class RigidTransform3DTests(unittest.TestCase):
    def test_translation_and_rotation_are_explicit_and_right_handed(self) -> None:
        translated = RigidTransform3D.translation_by((3, -2, 1))
        np.testing.assert_allclose(translated.apply((1, 2, 3)), (4, 0, 4))

        rotated = RigidTransform3D.rotation_about_axis(
            (0, 0, 1), pi / 2, about_point=(1, 0, 0), translation=(0, 2, 0)
        )
        np.testing.assert_allclose(rotated.apply((2, 0, 0)), (1, 3, 0), atol=1e-12)

        with self.assertRaisesRegex(DerivedDihedralContractError, "orthonormal"):
            RigidTransform3D(
                ((2, 0, 0), (0, 1, 0), (0, 0, 1)),
                (0, 0, 0),
            )

    def test_global_and_local_rigid_transforms_compose_in_documented_order(self) -> None:
        local = RigidTransform3D.translation_by((2.0, 0.0, 0.0))
        global_transform = RigidTransform3D.rotation_about_axis(
            (0.0, 0.0, 1.0), pi / 2
        )
        combined = global_transform.compose(local)
        point = np.asarray((1.0, 0.0, 0.0))
        np.testing.assert_allclose(
            combined.apply(point),
            global_transform.apply(local.apply(point)),
            atol=1.0e-12,
        )

    def test_local_placement_moves_the_rotation_center_with_the_copy(self) -> None:
        center = np.asarray((0.25, -0.5, 0.75), dtype=float)
        placement = RigidTransform3D.translation_by((2.0, -1.0, 0.5))
        center_rotation = RigidTransform3D.rotation_about_axis(
            (0.0, 0.0, 1.0),
            pi / 2,
            about_point=center,
        )
        combined = placement.compose(center_rotation)
        np.testing.assert_allclose(
            combined.apply(center),
            placement.apply(center),
            atol=1.0e-12,
        )


class BasePlaneRotation3DTests(unittest.TestCase):
    def test_selected_face_becomes_horizontal_bottom_about_solid_center(self) -> None:
        model = cube_model()
        motion = BasePlaneRotation3D.from_model(model, "right")
        transform = motion.final_transform()
        solid_center = np.mean(
            [
                np.asarray(model.vertex_map[item].entry_position, dtype=float)
                for item in sorted(model.vertex_map)
            ],
            axis=0,
        )
        face = model.face_map["right"]
        points = np.asarray(
            [transform.apply(model.vertex_map[item].entry_position) for item in face.vertex_ids]
        )
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal /= np.linalg.norm(normal)

        self.assertEqual(motion.schema, BASE_PLANE_ROTATION_SCHEMA)
        np.testing.assert_allclose(normal, (0.0, 0.0, -1.0), atol=1.0e-12)
        self.assertLess(float(np.ptp(points[:, 2])), 1.0e-12)
        np.testing.assert_allclose(
            motion.anchor,
            solid_center,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            transform.apply(solid_center),
            solid_center,
            atol=1.0e-12,
        )

    def test_explicit_anchor_overrides_the_solid_center(self) -> None:
        model = cube_model()
        anchor = (0.25, -0.5, 0.75)
        motion = BasePlaneRotation3D.from_model(
            model,
            "right",
            anchor=anchor,
        )
        np.testing.assert_allclose(motion.anchor, anchor, atol=1.0e-12)
        np.testing.assert_allclose(
            motion.final_transform().apply(anchor),
            anchor,
            atol=1.0e-12,
        )

    def test_progress_zero_is_identity_and_invalid_face_fails_closed(self) -> None:
        model = cube_model()
        motion = BasePlaneRotation3D.from_model(model, "right")
        np.testing.assert_allclose(
            motion.transform(0.0).apply((0.25, -0.5, 0.75)),
            (0.25, -0.5, 0.75),
        )
        with self.assertRaisesRegex(ValueError, "unknown base face"):
            BasePlaneRotation3D.from_model(model, "missing")

    def test_parallel_and_antiparallel_face_normals_are_deterministic(self) -> None:
        model = cube_model()
        already_bottom = BasePlaneRotation3D.from_model(model, "back")
        opposite = BasePlaneRotation3D.from_model(model, "front")
        self.assertEqual(already_bottom.total_angle, 0.0)
        self.assertAlmostEqual(opposite.total_angle, pi)
        face = model.face_map["front"]
        points = np.asarray(
            [
                opposite.final_transform().apply(
                    model.vertex_map[item].entry_position
                )
                for item in face.vertex_ids
            ]
        )
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(normal, (0.0, 0.0, -1.0), atol=1.0e-12)

    def test_base_plane_alignment_is_scale_independent(self) -> None:
        for scale in (1.0e-6, 1.0, 1.0e6):
            with self.subTest(scale=scale):
                payload = cube_model().to_dict()
                for vertex in payload["vertices"]:
                    vertex["entryPosition"] = [
                        scale * value for value in vertex["entryPosition"]
                    ]
                model = VisibilityModel.from_dict(payload)
                motion = BasePlaneRotation3D.from_model(
                    model,
                    "right",
                    target_outward_normal=(0.0, 0.0, -5.0),
                )
                face = model.face_map["right"]
                points = np.asarray(
                    [
                        motion.final_transform().apply(
                            model.vertex_map[item].entry_position
                        )
                        for item in face.vertex_ids
                    ]
                )
                self.assertLess(
                    float(np.ptp(points[:, 2])),
                    max(1.0e-12, scale * 1.0e-12),
                )


class DerivedDihedralContractTests(unittest.TestCase):
    def test_adjacent_faces_freeze_one_hinge_and_complete_boundary(self) -> None:
        model = DerivedDihedralModel.from_solid(
            "cube-with-extracted-dihedral",
            cube_model(),
            entity_id="top-front-copy",
            source_face_ids=("front", "top"),
        )

        self.assertEqual(model.schema, DERIVED_DIHEDRAL_MODEL_SCHEMA)
        self.assertEqual(model.extraction.hinge_vertex_ids, ("G", "H"))
        self.assertEqual(len(model.extraction.boundary_strokes), 7)
        self.assertEqual(model.extracted_vertex_ids, ("C", "D", "E", "F", "G", "H"))
        self.assertEqual(
            set(model.to_dict()),
            {"schema", "visibilityGroupId", "solid", "extraction"},
        )

    def test_non_adjacent_faces_and_missing_boundary_strokes_fail_closed(self) -> None:
        with self.assertRaisesRegex(DerivedDihedralContractError, "share exactly one"):
            DerivedDihedralModel.from_solid(
                "bad",
                cube_model(),
                entity_id="copy",
                source_face_ids=("front", "back"),
            )


class DerivedDihedralSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DerivedDihedralModel.from_solid(
            "cube-with-extracted-dihedral",
            cube_model(),
            entity_id="copy",
            source_face_ids=("front", "top"),
        )

    def test_identity_frame_suppresses_only_the_duplicate_source_edges(self) -> None:
        frame = compute_derived_dihedral_visibility(
            self.model,
            transform=RigidTransform3D.identity(),
            projection_matrix=np.eye(3),
        )

        self.assertEqual(frame.coincident_source_face_ids, ("front", "top"))
        self.assertEqual(len(frame.suppressed_source_stroke_ids), 7)
        self.assertEqual(
            len(frame.line_visibility.edges),
            len(self.model.solid.strokes)
            + len(self.model.extraction.boundary_strokes),
        )
        self.assertIn(
            "coincident_source_incident",
            {
                item.reason
                for item in frame.line_visibility.edge_map[
                    self.model.extracted_stroke_id("edge.G.H")
                ].skipped_faces
            },
        )

    def test_moved_copy_joins_the_same_global_occluder_set(self) -> None:
        view = np.asarray((1.0, 1.0, 1.0), dtype=float)
        view /= np.linalg.norm(view)
        screen_right = np.cross(np.asarray((0.0, 0.0, 1.0)), view)
        screen_right /= np.linalg.norm(screen_right)
        screen_up = np.cross(view, screen_right)
        projection = np.asarray((screen_right, screen_up, view), dtype=float)
        frame = compute_derived_dihedral_visibility(
            self.model,
            transform=RigidTransform3D.translation_by((0.0, 0.0, -2.5)),
            projection_matrix=projection,
        )

        self.assertEqual(frame.coincident_source_face_ids, ())
        self.assertEqual(frame.suppressed_source_stroke_ids, ())
        derived_edges = [
            edge
            for edge in frame.line_visibility.edges
            if edge.source_edge_id.startswith("copy:")
        ]
        self.assertTrue(
            any(
                interval.face_id.startswith("solid:")
                for edge in derived_edges
                for interval in edge.raw_intervals
            ),
            "at least one moved dihedral edge must be hidden by the source solid",
        )
        payload = canonical_derived_dihedral_trace_json(frame)
        self.assertEqual(payload, canonical_derived_dihedral_trace_json(frame))
        self.assertNotIn("NaN", payload)

        payload = cube_model().to_dict()
        payload["strokes"] = payload["strokes"][:-1]
        incomplete = VisibilityModel.from_dict(payload)
        with self.assertRaisesRegex(
            DerivedDihedralContractError, "map to exactly one source stroke"
        ):
            DerivedDihedralModel.from_solid(
                "bad",
                incomplete,
                entity_id="copy",
                source_face_ids=("front", "top"),
            )


if __name__ == "__main__":
    unittest.main()
