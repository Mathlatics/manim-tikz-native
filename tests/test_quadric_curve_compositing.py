from __future__ import annotations

import json
import unittest

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.compositing import (
    QuadricCompositingError,
    canonical_quadric_compositing_json,
    compute_quadric_compositing,
)
from polyhedron_visibility.quadrics.curve_intersections import (
    ProjectedCurveCrossing,
    compute_projected_curve_crossings,
)
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.visibility import (
    CurveVisibilityFrame,
    CurveVisibilityRecord,
)
from polyhedron_visibility.topology import ParameterInterval
from polyhedron_visibility.visibility import VisibilityKind, VisibilitySpan


VIEW = ParallelView.from_matrix(
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)


def _record(curve_id: str, *, hidden: bool = False) -> CurveVisibilityRecord:
    interval = ParameterInterval(0.0, 1.0)
    return CurveVisibilityRecord(
        curve_id,
        interval,
        (),
        (
            VisibilitySpan(
                interval,
                VisibilityKind.HIDDEN if hidden else VisibilityKind.VISIBLE,
                ("solid",) if hidden else (),
            ),
        ),
        1.0e-9,
    )


def _frame(*records: CurveVisibilityRecord, with_surface: bool = False) -> CurveVisibilityFrame:
    return CurveVisibilityFrame(
        VIEW.projection_matrix,
        VIEW.view_direction,
        ("solid",) if with_surface else (),
        tuple(sorted(records, key=lambda item: item.curve_id)),
    )


class CurveCrossingCompositingTests(unittest.TestCase):
    def test_crossing_adds_a_real_curve_to_curve_painter_edge(self) -> None:
        far = SegmentCurve("far", (-1.0, 0.0, -2.0), (1.0, 0.0, -2.0))
        near = SegmentCurve("near", (0.0, -1.0, 2.0), (0.0, 1.0, 2.0))
        crossings = compute_projected_curve_crossings((far, near), VIEW)
        frame = compute_quadric_compositing(
            _frame(_record("far"), _record("near")),
            (),
            curve_crossings=crossings,
        )
        far_item = next(
            item.item_id for item in frame.curve_fragments if item.curve_id == "far"
        )
        near_item = next(
            item.item_id for item in frame.curve_fragments if item.curve_id == "near"
        )
        relation = next(
            item
            for item in frame.order_relations
            if item.reason.startswith("projected_curve_crossing:")
        )
        self.assertEqual((relation.far_item_id, relation.near_item_id), (far_item, near_item))
        self.assertLess(frame.draw_order.index(far_item), frame.draw_order.index(near_item))
        payload = json.loads(canonical_quadric_compositing_json(frame))
        self.assertEqual(payload["curveCrossings"], [crossings[0].to_dict()])

    def test_true_3d_intersection_does_not_invent_a_painter_edge(self) -> None:
        first = SegmentCurve("a", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        second = SegmentCurve("b", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))
        crossings = compute_projected_curve_crossings((first, second), VIEW)
        frame = compute_quadric_compositing(
            _frame(_record("a"), _record("b")),
            (),
            curve_crossings=crossings,
        )
        self.assertTrue(crossings[0].coincident_depth)
        self.assertFalse(
            any(
                item.reason.startswith("projected_curve_crossing:")
                for item in frame.order_relations
            )
        )

    def test_opposite_depth_evidence_for_unsplit_fragments_fails_closed(self) -> None:
        first = ProjectedCurveCrossing(
            "crossing:a:b:0",
            "a",
            "b",
            0.25,
            0.25,
            (0.0, 0.0),
            -1.0,
            1.0,
            "a",
            "b",
        )
        second = ProjectedCurveCrossing(
            "crossing:a:b:1",
            "a",
            "b",
            0.75,
            0.75,
            (1.0, 1.0),
            1.0,
            -1.0,
            "b",
            "a",
        )
        with self.assertRaisesRegex(QuadricCompositingError, "contradictory"):
            compute_quadric_compositing(
                _frame(_record("a"), _record("b")),
                (),
                curve_crossings=(second, first),
            )


if __name__ == "__main__":
    unittest.main()
