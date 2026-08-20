from __future__ import annotations

import unittest

from polyhedron_visibility.compositor import (
    CompositorCycleError,
    PainterConstraint,
    painter_ranks,
    stable_topological_sort,
)
from polyhedron_visibility.contract import TolerancePolicy
from polyhedron_visibility.geometry import (
    GeometryContext,
    GeometryQuantity,
    GeometryScale,
    coordinate_scale,
    resolve_geometry_context,
)
from polyhedron_visibility.topology import (
    ParameterInterval,
    TaggedInterval,
    assert_exact_partition,
    coalesce_tagged_intervals,
    partition_parameter_domain,
)
from polyhedron_visibility.visibility import (
    OcclusionInterval,
    VisibilityKind,
    partition_visibility,
)


class GeometryContextTests(unittest.TestCase):
    def test_default_context_uses_the_existing_policy_type(self) -> None:
        context = GeometryContext()
        self.assertIsInstance(context.tolerance, TolerancePolicy)
        self.assertIs(resolve_geometry_context(context), context)

    def test_quantity_scales_and_overrides_are_separate(self) -> None:
        context = GeometryContext(
            scale=GeometryScale(length=100.0, parameter=0.5),
            overrides={GeometryQuantity.PARAMETER: 2.5e-6},
        )
        self.assertEqual(
            context.epsilon(GeometryQuantity.PARAMETER),
            2.5e-6,
        )
        self.assertNotEqual(
            context.scale.for_quantity(GeometryQuantity.LENGTH),
            context.scale.for_quantity(GeometryQuantity.PARAMETER),
        )

    def test_coordinate_scale_rejects_non_finite_geometry(self) -> None:
        self.assertEqual(coordinate_scale([[1.0, -4.0], [2.0, 3.0]]), 4.0)
        with self.assertRaises(ValueError):
            coordinate_scale([float("inf")])


class TopologyLayerTests(unittest.TestCase):
    def test_single_cell_preserves_exact_domain_boundaries(self) -> None:
        domain = ParameterInterval(0.0, 1.0)
        cells = partition_parameter_domain(
            domain,
            [0.0, 1.0, 1.0e-14],
            tolerance=1.0e-10,
        )
        self.assertEqual(cells, (domain,))
        self.assertEqual(assert_exact_partition(domain, cells), cells)

    def test_coalescing_respects_semantic_identity(self) -> None:
        same = coalesce_tagged_intervals(
            [
                TaggedInterval(ParameterInterval(0.0, 0.4), "face-a"),
                TaggedInterval(ParameterInterval(0.4, 0.8), "face-a"),
            ]
        )
        self.assertEqual(
            same,
            (TaggedInterval(ParameterInterval(0.0, 0.8), "face-a"),),
        )
        different = coalesce_tagged_intervals(
            [
                TaggedInterval(ParameterInterval(0.0, 0.4), "face-a"),
                TaggedInterval(ParameterInterval(0.4, 0.8), "face-b"),
            ]
        )
        self.assertEqual(len(different), 2)


class VisibilityLayerTests(unittest.TestCase):
    def test_boundary_only_contact_remains_visible(self) -> None:
        domain = ParameterInterval(0.0, 1.0)
        spans = partition_visibility(
            domain,
            [OcclusionInterval(ParameterInterval(0.5, 0.5), "face")],
            parameter_tolerance=1.0e-9,
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].interval, domain)
        self.assertIs(spans[0].kind, VisibilityKind.VISIBLE)

    def test_same_occluder_handoff_keeps_one_hidden_slot(self) -> None:
        domain = ParameterInterval(0.0, 1.0)
        spans = partition_visibility(
            domain,
            [
                OcclusionInterval(ParameterInterval(0.2, 0.4), "face-a"),
                OcclusionInterval(ParameterInterval(0.4, 0.7), "face-a"),
            ],
            parameter_tolerance=0.0,
        )
        self.assertEqual(
            [(span.kind, span.interval.start, span.interval.end) for span in spans],
            [
                (VisibilityKind.VISIBLE, 0.0, 0.2),
                (VisibilityKind.HIDDEN, 0.2, 0.7),
                (VisibilityKind.VISIBLE, 0.7, 1.0),
            ],
        )

    def test_different_occluders_keep_distinct_hidden_slots(self) -> None:
        domain = ParameterInterval(0.0, 1.0)
        spans = partition_visibility(
            domain,
            [
                OcclusionInterval(ParameterInterval(0.2, 0.4), "face-a"),
                OcclusionInterval(ParameterInterval(0.4, 0.7), "face-b"),
            ],
            parameter_tolerance=0.0,
        )
        hidden = [span for span in spans if span.kind is VisibilityKind.HIDDEN]
        self.assertEqual(
            [span.occluders for span in hidden],
            [("face-a",), ("face-b",)],
        )


class CompositorLayerTests(unittest.TestCase):
    def test_unrelated_nodes_keep_authored_order(self) -> None:
        order = stable_topological_sort(["back", "label", "front"], [])
        self.assertEqual(order, ("back", "label", "front"))
        self.assertEqual(painter_ranks(order)["front"], 2)

    def test_constraints_produce_stable_far_to_near_order(self) -> None:
        order = stable_topological_sort(
            ["label", "near", "far"],
            [PainterConstraint("far", "near")],
        )
        self.assertLess(order.index("far"), order.index("near"))
        self.assertEqual(order.index("label"), 0)

    def test_cycles_are_explicit(self) -> None:
        with self.assertRaises(CompositorCycleError):
            stable_topological_sort(
                ["a", "b"],
                [("a", "b"), ("b", "a")],
            )


if __name__ == "__main__":
    unittest.main()
