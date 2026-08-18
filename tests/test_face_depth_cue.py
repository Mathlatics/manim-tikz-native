from __future__ import annotations

import unittest

import numpy as np

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.depth_cue import FaceDepthCueStyle, compute_face_depth_cue


VERTICES = {
    "A": (-1.0, -1.0, -1.0),
    "B": (1.0, -1.0, -1.0),
    "C": (1.0, 1.0, -1.0),
    "D": (-1.0, 1.0, -1.0),
    "E": (-1.0, -1.0, 1.0),
    "F": (1.0, -1.0, 1.0),
    "G": (1.0, 1.0, 1.0),
    "H": (-1.0, 1.0, 1.0),
    "X": (-2.0, 0.0, 0.0),
    "Y": (2.0, 0.0, 0.0),
}
FACES = {
    "back": ("A", "D", "C", "B"),
    "front": ("E", "F", "G", "H"),
    "bottom": ("A", "B", "F", "E"),
    "right": ("B", "C", "G", "F"),
    "top": ("D", "H", "G", "C"),
    "left": ("A", "E", "H", "D"),
}


def surface_edges() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                tuple(sorted((start, face[(index + 1) % len(face)])))
                for face in FACES.values()
                for index, start in enumerate(face)
            }
        )
    )


def cube_model(*, reverse_faces: bool = False) -> VisibilityModel:
    incidents = {edge: [] for edge in surface_edges()}
    for face_id, cycle in FACES.items():
        for index, start in enumerate(cycle):
            edge = tuple(sorted((start, cycle[(index + 1) % len(cycle)])))
            incidents[edge].append(face_id)
    faces = {
        face_id: tuple(reversed(cycle)) if reverse_faces else cycle
        for face_id, cycle in FACES.items()
    }
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "depth-cued-cube",
            "vertices": [
                {"vertexId": key, "entryPosition": value}
                for key, value in VERTICES.items()
            ],
            "faces": [
                {"faceId": key, "vertexIds": list(value)}
                for key, value in faces.items()
            ],
            "strokes": [
                {
                    "sourceEdgeId": f"edge.{start}.{end}",
                    "vertexIds": [start, end],
                    "incidentFaceIds": sorted(incidents[(start, end)]),
                }
                for start, end in surface_edges()
            ]
            + [
                {
                    "sourceEdgeId": "probe.X.Y",
                    "vertexIds": ["X", "Y"],
                    "incidentFaceIds": [],
                }
            ],
        }
    )


class FaceDepthCueTests(unittest.TestCase):
    def test_face_winding_does_not_change_outward_cues(self) -> None:
        normal = compute_face_depth_cue(
            cube_model(), projection_matrix=np.eye(3)
        )
        reversed_frame = compute_face_depth_cue(
            cube_model(reverse_faces=True), projection_matrix=np.eye(3)
        )
        for face_id in normal.face_map:
            with self.subTest(face=face_id):
                self.assertTrue(
                    np.allclose(
                        normal.face_map[face_id].outward_normal,
                        reversed_frame.face_map[face_id].outward_normal,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
                self.assertAlmostEqual(
                    normal.face_map[face_id].opacity_scale,
                    reversed_frame.face_map[face_id].opacity_scale,
                )

    def test_front_near_face_is_more_opaque_than_back_far_face(self) -> None:
        frame = compute_face_depth_cue(
            cube_model(), projection_matrix=np.eye(3)
        )
        front = frame.face_map["front"]
        back = frame.face_map["back"]
        self.assertGreater(front.facing_score, back.facing_score)
        self.assertGreater(front.normalized_depth, back.normalized_depth)
        self.assertGreater(front.opacity_scale, back.opacity_scale)
        self.assertGreater(front.near_score, back.near_score)
        self.assertGreater(front.saturation_scale, back.saturation_scale)
        self.assertLess(front.fog_strength, back.fog_strength)
        self.assertGreater(front.surface_visibility, back.surface_visibility)
        self.assertGreater(front.opacity_scale / back.opacity_scale, 4.0)
        self.assertGreater(
            max(item.hue_shift_turns for item in frame.faces)
            - min(item.hue_shift_turns for item in frame.faces),
            0.05,
        )
        self.assertGreater(max(item.brightness for item in frame.faces), min(item.brightness for item in frame.faces))

    def test_back_turned_faces_keep_distinct_half_lambert_tones(self) -> None:
        frame = compute_face_depth_cue(
            cube_model(),
            projection_matrix=(
                (1.0, 0.0, -0.35),
                (0.15, 1.0, -0.20),
                (0.35, 0.20, 1.0),
            ),
        )
        back_turned = sorted(
            item.light_score for item in frame.faces if item.light_score < 0.5
        )
        self.assertGreaterEqual(len(back_turned), 2)
        self.assertGreater(back_turned[-1] - back_turned[0], 0.10)

    def test_free_line_is_never_mistaken_for_a_silhouette(self) -> None:
        frame = compute_face_depth_cue(
            cube_model(),
            projection_matrix=(
                (1.0, 0.0, -0.35),
                (0.15, 1.0, -0.20),
                (0.35, 0.20, 1.0),
            ),
        )
        self.assertFalse(frame.edge_map["probe.X.Y"].is_silhouette)
        silhouettes = [item for item in frame.edges if item.is_silhouette]
        self.assertTrue(silhouettes)
        self.assertTrue(
            all(
                item.visible_width_scale
                == FaceDepthCueStyle().silhouette_visible_width_scale
                for item in silhouettes
            )
        )

    def test_camera_direction_changes_face_and_silhouette_cues(self) -> None:
        front = compute_face_depth_cue(
            cube_model(), projection_matrix=np.eye(3)
        )
        side = compute_face_depth_cue(
            cube_model(),
            projection_matrix=((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        )
        self.assertNotEqual(
            front.face_map["front"].facing_score,
            side.face_map["front"].facing_score,
        )
        self.assertNotEqual(
            {
                item.source_edge_id
                for item in front.edges
                if item.is_silhouette
            },
            {
                item.source_edge_id
                for item in side.edges
                if item.is_silhouette
            },
        )

    def test_explicit_draw_order_is_preserved_and_trace_is_deterministic(self) -> None:
        model = cube_model()
        order = tuple(reversed(sorted(model.face_map)))
        frame = compute_face_depth_cue(
            model,
            projection_matrix=np.eye(3),
            face_draw_order=order,
        )
        self.assertEqual(frame.face_draw_order, order)
        self.assertEqual(
            [frame.face_map[face_id].draw_rank for face_id in order],
            list(range(len(order))),
        )
        self.assertEqual(frame.to_dict(), frame.to_dict())


if __name__ == "__main__":
    unittest.main()
