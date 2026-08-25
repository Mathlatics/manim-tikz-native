from __future__ import annotations

from dataclasses import replace
import json
import unittest

from polyhedron_visibility.compositor import PainterConstraint
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.compositing import (
    QuadricCompositingError,
    QuadricPaintKind,
    QuadricPaintPolicy,
    canonical_quadric_compositing_json,
    compute_quadric_compositing,
)
from polyhedron_visibility.quadrics.projection import (
    OpaqueProjectionProxy,
    ProjectionApproximationMetadata,
)
from polyhedron_visibility.quadrics.visibility import (
    CurveVisibilityFrame,
    CurveVisibilityRecord,
)
from polyhedron_visibility.style import OcclusionStyle
from polyhedron_visibility.topology import ParameterInterval
from polyhedron_visibility.visibility import VisibilityKind, VisibilitySpan


IDENTITY_VIEW = ParallelView.from_matrix(
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)


def _proxy(surface_id: str, *, patch_id: str | None = None) -> OpaqueProjectionProxy:
    offset = float(sum(ord(item) for item in surface_id) % 7)
    return OpaqueProjectionProxy(
        patch_id or f"{surface_id}:proxy",
        surface_id,
        (
            (offset, 0.0),
            (offset + 1.0, 0.0),
            (offset + 1.0, 1.0),
            (offset, 1.0),
            (offset, 0.0),
        ),
        ProjectionApproximationMetadata(
            max_chord_error=0.01,
            observed_chord_error=0.005,
            max_segments=16,
            segment_count=4,
            adaptive_interval_count=4,
            support_evaluation_count=8,
        ),
    )


def _record(curve_id: str, occluder: str) -> CurveVisibilityRecord:
    domain = ParameterInterval(0.0, 3.0)
    return CurveVisibilityRecord(
        curve_id,
        domain,
        (),
        (
            VisibilitySpan(
                ParameterInterval(0.0, 1.0),
                VisibilityKind.VISIBLE,
            ),
            VisibilitySpan(
                ParameterInterval(1.0, 2.0),
                VisibilityKind.HIDDEN,
                (occluder,),
            ),
            VisibilitySpan(
                ParameterInterval(2.0, 3.0),
                VisibilityKind.VISIBLE,
            ),
        ),
        1.0e-9,
    )


def _visibility(
    surface_ids: tuple[str, ...] = ("alpha", "beta"),
    *,
    records: tuple[CurveVisibilityRecord, ...] | None = None,
) -> CurveVisibilityFrame:
    if records is None:
        records = (_record("curve", surface_ids[0]),)
    return CurveVisibilityFrame(
        IDENTITY_VIEW.projection_matrix,
        IDENTITY_VIEW.view_direction,
        tuple(sorted(surface_ids)),
        records,
    )


def _rank(frame: object) -> dict[str, int]:
    return {
        item_id: index
        for index, item_id in enumerate(frame.draw_order)  # type: ignore[attr-defined]
    }


class QuadricCompositingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.visibility = _visibility()
        self.proxies = (_proxy("alpha"), _proxy("beta"))
        self.style = OcclusionStyle(
            max_projected_length=12.0,
            dash_length=0.09,
            dash_gap=0.07,
            hidden_color="#7080ff",
        )

    def test_diagrammatic_registers_surfaces_visible_and_hidden_together(self) -> None:
        frame = compute_quadric_compositing(
            self.visibility,
            self.proxies,
            paint_policy="diagrammatic",
            curve_styles=self.style,
            surface_constraints=(PainterConstraint("alpha", "beta"),),
        )
        self.assertIs(frame.paint_policy, QuadricPaintPolicy.DIAGRAMMATIC)
        hidden = next(
            item
            for item in frame.curve_fragments
            if item.kind is QuadricPaintKind.HIDDEN_CURVE
        )
        visible = [
            item
            for item in frame.curve_fragments
            if item.kind is QuadricPaintKind.VISIBLE_CURVE
        ]
        self.assertTrue(hidden.painted)
        self.assertEqual(hidden.render_intent, "dashed")
        self.assertEqual(hidden.occluder_surface_ids, ("alpha",))
        self.assertIn(hidden.item_id, frame.item_ids)
        self.assertEqual(frame.omitted_fragment_ids, ())

        ranks = _rank(frame)
        surface_ids = [item.item_id for item in frame.surface_items]
        self.assertLess(ranks[surface_ids[0]], ranks[surface_ids[1]])
        for surface_id in surface_ids:
            self.assertLess(ranks[surface_id], ranks[hidden.item_id])
            for fragment in visible:
                self.assertLess(ranks[surface_id], ranks[fragment.item_id])
        self.assertEqual(frame.styles[0].style_id, "style:default")
        self.assertEqual(frame.styles[0].dash_length, 0.09)
        self.assertEqual(frame.styles[0].hidden_color, "#7080ff")

    def test_physical_keeps_hidden_trace_but_omits_it_from_paint_graph(self) -> None:
        frame = compute_quadric_compositing(
            self.visibility,
            self.proxies,
            paint_policy=QuadricPaintPolicy.PHYSICAL,
        )
        hidden = next(
            item
            for item in frame.curve_fragments
            if item.kind is QuadricPaintKind.HIDDEN_CURVE
        )
        self.assertFalse(hidden.painted)
        self.assertEqual(hidden.render_intent, "omit")
        self.assertNotIn(hidden.item_id, frame.item_ids)
        self.assertEqual(frame.omitted_fragment_ids, (hidden.item_id,))
        self.assertEqual(frame.styles, ())

        ranks = _rank(frame)
        for surface in frame.surface_items:
            for fragment in frame.curve_fragments:
                if fragment.painted:
                    self.assertLess(ranks[surface.item_id], ranks[fragment.item_id])

    def test_depth_aware_diagrammatic_places_hidden_behind_its_occluder(self) -> None:
        frame = compute_quadric_compositing(
            _visibility(("alpha",)),
            (_proxy("alpha"),),
            paint_policy="depth_aware_diagrammatic",
            curve_styles=self.style,
        )
        self.assertIs(
            frame.paint_policy,
            QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        )
        hidden = next(
            item
            for item in frame.curve_fragments
            if item.kind is QuadricPaintKind.HIDDEN_CURVE
        )
        visible = tuple(
            item
            for item in frame.curve_fragments
            if item.kind is QuadricPaintKind.VISIBLE_CURVE
        )
        surface = frame.surface_items[0]

        self.assertTrue(hidden.painted)
        self.assertEqual(hidden.render_intent, "dashed")
        self.assertEqual(hidden.occluder_surface_ids, ("alpha",))
        ranks = _rank(frame)
        self.assertLess(ranks[hidden.item_id], ranks[surface.item_id])
        self.assertTrue(
            all(ranks[surface.item_id] < ranks[item.item_id] for item in visible)
        )
        self.assertIn(
            (
                hidden.item_id,
                surface.item_id,
                "depth_aware_hidden_occlusion",
            ),
            {
                (item.far_item_id, item.near_item_id, item.reason)
                for item in frame.order_relations
            },
        )

    def test_depth_aware_hidden_is_bracketed_by_farther_surface_and_occluder(
        self,
    ) -> None:
        frame = compute_quadric_compositing(
            _visibility(("alpha", "beta")),
            (_proxy("alpha"), _proxy("beta")),
            paint_policy="depth_aware_diagrammatic",
            surface_constraints=(("beta", "alpha"),),
        )
        hidden = next(
            item
            for item in frame.curve_fragments
            if item.kind is QuadricPaintKind.HIDDEN_CURVE
        )
        ranks = _rank(frame)
        beta = next(
            item.item_id for item in frame.surface_items if item.surface_id == "beta"
        )
        alpha = next(
            item.item_id for item in frame.surface_items if item.surface_id == "alpha"
        )

        self.assertLess(ranks[beta], ranks[hidden.item_id])
        self.assertLess(ranks[hidden.item_id], ranks[alpha])
        self.assertIn(
            (
                beta,
                hidden.item_id,
                "depth_aware_hidden_after_farther_surface",
            ),
            {
                (item.far_item_id, item.near_item_id, item.reason)
                for item in frame.order_relations
            },
        )

    def test_depth_aware_does_not_move_one_occluder_in_front_of_the_stroke(
        self,
    ) -> None:
        record = _record("curve", "alpha")
        hidden = record.spans[1]
        visibility = _visibility(
            records=(
                CurveVisibilityRecord(
                    record.curve_id,
                    record.domain,
                    record.critical_events,
                    (
                        record.spans[0],
                        VisibilitySpan(
                            hidden.interval,
                            hidden.kind,
                            ("alpha", "beta"),
                        ),
                        record.spans[2],
                    ),
                    record.parameter_tolerance,
                ),
            )
        )
        frame = compute_quadric_compositing(
            visibility,
            (_proxy("alpha"), _proxy("beta")),
            paint_policy="depth_aware_diagrammatic",
            surface_constraints=(("beta", "alpha"),),
        )
        hidden_fragment = next(
            item
            for item in frame.curve_fragments
            if item.kind is QuadricPaintKind.HIDDEN_CURVE
        )
        ranks = _rank(frame)
        surface_ranks = {
            item.surface_id: ranks[item.item_id] for item in frame.surface_items
        }
        self.assertLess(ranks[hidden_fragment.item_id], surface_ranks["beta"])
        self.assertLess(surface_ranks["beta"], surface_ranks["alpha"])

    def test_depth_aware_multi_occluder_chain_remains_acyclic(self) -> None:
        record = _record("curve", "alpha")
        hidden = record.spans[1]
        visibility = _visibility(
            records=(
                CurveVisibilityRecord(
                    record.curve_id,
                    record.domain,
                    record.critical_events,
                    (
                        record.spans[0],
                        VisibilitySpan(
                            hidden.interval,
                            hidden.kind,
                            ("alpha", "gamma"),
                        ),
                        record.spans[2],
                    ),
                    record.parameter_tolerance,
                ),
            ),
            surface_ids=("alpha", "beta", "gamma"),
        )
        frame = compute_quadric_compositing(
            visibility,
            (_proxy("alpha"), _proxy("beta"), _proxy("gamma")),
            paint_policy="depth_aware_diagrammatic",
            surface_constraints=(("gamma", "beta"), ("beta", "alpha")),
        )
        hidden_fragment = next(
            item
            for item in frame.curve_fragments
            if item.kind is QuadricPaintKind.HIDDEN_CURVE
        )
        ranks = _rank(frame)
        surface_ranks = {
            item.surface_id: ranks[item.item_id] for item in frame.surface_items
        }
        self.assertLess(ranks[hidden_fragment.item_id], surface_ranks["gamma"])
        self.assertLess(surface_ranks["gamma"], surface_ranks["beta"])
        self.assertLess(surface_ranks["beta"], surface_ranks["alpha"])

    def test_visible_curve_is_above_every_surface_not_only_its_occluder(self) -> None:
        frame = compute_quadric_compositing(
            self.visibility,
            self.proxies,
            paint_policy="physical",
        )
        visible = next(
            item
            for item in frame.curve_fragments
            if item.kind is QuadricPaintKind.VISIBLE_CURVE
        )
        related_surfaces = {
            relation.far_item_id
            for relation in frame.order_relations
            if relation.near_item_id == visible.item_id
        }
        self.assertEqual(
            related_surfaces,
            {item.item_id for item in frame.surface_items},
        )


class QuadricCompositingDeterminismTests(unittest.TestCase):
    def test_proxy_constraint_and_style_input_order_do_not_change_trace(self) -> None:
        visibility = _visibility(
            records=(
                _record("a-curve", "alpha"),
                _record("z-curve", "beta"),
            )
        )
        styles_a = {
            "z-curve": OcclusionStyle(max_projected_length=9.0),
            "a-curve": OcclusionStyle(max_projected_length=8.0),
        }
        styles_b = {
            "a-curve": OcclusionStyle(max_projected_length=8.0),
            "z-curve": OcclusionStyle(max_projected_length=9.0),
        }
        first = compute_quadric_compositing(
            visibility,
            (_proxy("beta"), _proxy("alpha")),
            curve_styles=styles_a,
            surface_constraints=(("alpha", "beta"), ("alpha", "beta")),
        )
        second = compute_quadric_compositing(
            visibility,
            (_proxy("alpha"), _proxy("beta")),
            curve_styles=styles_b,
            surface_constraints=(
                PainterConstraint("alpha", "beta"),
            ),
        )
        self.assertEqual(first, second)
        self.assertEqual(
            canonical_quadric_compositing_json(first),
            canonical_quadric_compositing_json(second),
        )

    def test_canonical_json_is_complete_and_round_trips(self) -> None:
        frame = compute_quadric_compositing(
            _visibility(),
            (_proxy("alpha"), _proxy("beta")),
            paint_policy="diagrammatic",
            curve_styles=OcclusionStyle(max_projected_length=10.0),
        )
        payload = json.loads(canonical_quadric_compositing_json(frame))
        self.assertEqual(payload, frame.to_dict())
        self.assertEqual(set(payload["drawOrder"]), set(frame.item_ids))
        self.assertEqual(
            len(payload["curveFragments"]),
            len(frame.visibility.records[0].spans),
        )
        self.assertFalse(
            payload["surfaceItems"][0]["proxy"]["metadata"][
                "visibilityAuthoritative"
            ]
        )

    def test_item_id_namespaces_prevent_patch_curve_collision(self) -> None:
        visibility = _visibility(
            ("surface",),
            records=(_record("surface:surface:proxy", "surface"),),
        )
        frame = compute_quadric_compositing(
            visibility,
            (_proxy("surface", patch_id="surface:proxy"),),
        )
        self.assertEqual(len(frame.item_ids), len(set(frame.item_ids)))
        self.assertTrue(frame.surface_items[0].item_id.startswith("surface:"))
        self.assertTrue(frame.curve_fragments[0].item_id.startswith("curve:"))


class QuadricCompositingFailureTests(unittest.TestCase):
    def test_surface_proxy_coverage_must_be_exact(self) -> None:
        visibility = _visibility()
        with self.assertRaisesRegex(QuadricCompositingError, "cover"):
            compute_quadric_compositing(visibility, (_proxy("alpha"),))
        with self.assertRaisesRegex(QuadricCompositingError, "exactly one"):
            compute_quadric_compositing(
                visibility,
                (_proxy("alpha", patch_id="one"), _proxy("alpha", patch_id="two")),
            )

    def test_unknown_surface_constraint_fails_closed(self) -> None:
        with self.assertRaisesRegex(QuadricCompositingError, "unknown surfaces"):
            compute_quadric_compositing(
                _visibility(),
                (_proxy("alpha"), _proxy("beta")),
                surface_constraints=(("alpha", "missing"),),
            )

    def test_directly_contradictory_surface_order_fails_closed(self) -> None:
        with self.assertRaisesRegex(QuadricCompositingError, "contradictory"):
            compute_quadric_compositing(
                _visibility(),
                (_proxy("alpha"), _proxy("beta")),
                surface_constraints=(
                    ("alpha", "beta"),
                    ("beta", "alpha"),
                ),
            )

    def test_three_surface_painter_cycle_fails_closed(self) -> None:
        visibility = _visibility(("alpha", "beta", "gamma"))
        with self.assertRaisesRegex(QuadricCompositingError, "cycle"):
            compute_quadric_compositing(
                visibility,
                (_proxy("gamma"), _proxy("alpha"), _proxy("beta")),
                surface_constraints=(
                    ("alpha", "beta"),
                    ("beta", "gamma"),
                    ("gamma", "alpha"),
                ),
            )

    def test_curve_style_mapping_must_cover_records_exactly(self) -> None:
        visibility = _visibility()
        with self.assertRaisesRegex(QuadricCompositingError, "missing curve"):
            compute_quadric_compositing(
                visibility,
                (_proxy("alpha"), _proxy("beta")),
                curve_styles={},
            )
        with self.assertRaisesRegex(QuadricCompositingError, "unknown extra"):
            compute_quadric_compositing(
                visibility,
                (_proxy("alpha"), _proxy("beta")),
                curve_styles={
                    "curve": OcclusionStyle(max_projected_length=5.0),
                    "extra": OcclusionStyle(max_projected_length=5.0),
                },
            )

    def test_noncanonical_style_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            QuadricCompositingError, "canonically serializable"
        ):
            compute_quadric_compositing(
                _visibility(),
                (_proxy("alpha"), _proxy("beta")),
                curve_styles=OcclusionStyle(
                    max_projected_length=5.0,
                    hidden_color=object(),
                ),
            )

    def test_frame_rejects_noncanonical_draw_order(self) -> None:
        frame = compute_quadric_compositing(
            _visibility(),
            (_proxy("alpha"), _proxy("beta")),
        )
        with self.assertRaisesRegex(QuadricCompositingError, "canonical order"):
            replace(frame, draw_order=tuple(reversed(frame.draw_order)))


if __name__ == "__main__":
    unittest.main()
