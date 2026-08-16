from __future__ import annotations

import unittest

from polyhedron_visibility import VisibilityModel, compute_frame_visibility
from polyhedron_visibility.binding import (
    OcclusionCapacityError,
    OverlayCapacity,
    build_overlay_plan,
)
from polyhedron_visibility.style import OcclusionStyle


def _model() -> VisibilityModel:
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "state-fixture",
            "vertices": [
                {"vertexId": "a", "entryPosition": [-2, 0, 0]},
                {"vertexId": "b", "entryPosition": [2, 0, 0]},
                {"vertexId": "p0", "entryPosition": [-1, -1, 1]},
                {"vertexId": "p1", "entryPosition": [1, -1, 1]},
                {"vertexId": "p2", "entryPosition": [1, 1, 1]},
                {"vertexId": "p3", "entryPosition": [-1, 1, 1]},
            ],
            "faces": [
                {
                    "faceId": "front",
                    "vertexIds": ["p0", "p1", "p2", "p3"],
                }
            ],
            "strokes": [
                {"sourceEdgeId": "probe", "vertexIds": ["a", "b"]}
            ],
        }
    )


class OverlayPlanningTests(unittest.TestCase):
    def test_plan_coalesces_provenance_changes_without_losing_visible_gaps(self) -> None:
        model = VisibilityModel.from_dict(
            {
                "schema": "manim-convex-polyhedron-visibility/v1",
                "visibilityGroupId": "overlap",
                "vertices": [
                    {"vertexId": "a", "entryPosition": [-2, 0, 0]},
                    {"vertexId": "b", "entryPosition": [2, 0, 0]},
                    {"vertexId": "l0", "entryPosition": [-1.5, -1, 1]},
                    {"vertexId": "l1", "entryPosition": [0.2, -1, 1]},
                    {"vertexId": "l2", "entryPosition": [0.2, 1, 1]},
                    {"vertexId": "l3", "entryPosition": [-1.5, 1, 1]},
                    {"vertexId": "r0", "entryPosition": [-0.2, -1, 2]},
                    {"vertexId": "r1", "entryPosition": [1.5, -1, 2]},
                    {"vertexId": "r2", "entryPosition": [1.5, 1, 2]},
                    {"vertexId": "r3", "entryPosition": [-0.2, 1, 2]},
                ],
                "faces": [
                    {"faceId": "left", "vertexIds": ["l0", "l1", "l2", "l3"]},
                    {"faceId": "right", "vertexIds": ["r0", "r1", "r2", "r3"]},
                ],
                "strokes": [{"sourceEdgeId": "probe", "vertexIds": ["a", "b"]}],
            }
        )
        frame = compute_frame_visibility(model, projection_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        style = OcclusionStyle(max_projected_length=5.0, dash_length=0.1, dash_gap=0.1)
        capacity = OverlayCapacity.for_stroke(model.stroke_map["probe"], model, style)

        plan = build_overlay_plan(
            frame.edge_map["probe"],
            display_start=(-2, 0, 0),
            display_end=(2, 0, 0),
            capacity=capacity,
            style=style,
        )

        self.assertEqual(len(plan.visible_segments), 2)
        # The core trace has three adjacent hidden spans ([left], [left,right],
        # [right]).  Rendering intentionally coalesces those into one hidden
        # component because all three use the same dashed style.
        self.assertEqual(len(plan.hidden_segments), 1)
        self.assertAlmostEqual(plan.hidden_segments[0].start_parameter, 0.125, places=6)
        self.assertAlmostEqual(plan.hidden_segments[0].end_parameter, 0.875, places=6)

    def test_capacity_failure_is_detected_before_any_render_state_is_needed(self) -> None:
        model = _model()
        frame = compute_frame_visibility(model, projection_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        style = OcclusionStyle(max_projected_length=1.0, dash_length=0.1, dash_gap=0.1)
        capacity = OverlayCapacity.for_stroke(model.stroke_map["probe"], model, style)

        with self.assertRaisesRegex(OcclusionCapacityError, "projected length"):
            build_overlay_plan(
                frame.edge_map["probe"],
                display_start=(-2, 0, 0),
                display_end=(2, 0, 0),
                capacity=capacity,
                style=style,
            )

    def test_dash_phase_is_anchored_to_source_start_when_occlusion_boundary_moves(self) -> None:
        model = _model()
        style = OcclusionStyle(max_projected_length=5.0, dash_length=0.2, dash_gap=0.2)
        capacity = OverlayCapacity.for_stroke(model.stroke_map["probe"], model, style)
        entry = compute_frame_visibility(
            model,
            projection_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        shifted_positions = dict(model.entry_positions)
        for vertex_id in ("p0", "p1", "p2", "p3"):
            x, y, z = shifted_positions[vertex_id]
            shifted_positions[vertex_id] = (x + 0.18, y, z)
        shifted = compute_frame_visibility(
            model,
            vertex_positions=shifted_positions,
            projection_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )

        first_plan = build_overlay_plan(
            entry.edge_map["probe"],
            display_start=(-2, 0, 0),
            display_end=(2, 0, 0),
            capacity=capacity,
            style=style,
        )
        shifted_plan = build_overlay_plan(
            shifted.edge_map["probe"],
            display_start=(-2, 0, 0),
            display_end=(2, 0, 0),
            capacity=capacity,
            style=style,
        )

        first_interior_starts = {
            round(dash.start[0], 6)
            for dash in first_plan.hidden_segments[0].dashes[1:-1]
        }
        shifted_interior_starts = {
            round(dash.start[0], 6)
            for dash in shifted_plan.hidden_segments[0].dashes[1:-1]
        }
        self.assertTrue(first_interior_starts & shifted_interior_starts)
        # A boundary-anchored implementation would restart at the new face
        # boundary, shifting every interior dash by 0.18.
        self.assertGreaterEqual(len(first_interior_starts & shifted_interior_starts), 2)


if __name__ == "__main__":
    unittest.main()
