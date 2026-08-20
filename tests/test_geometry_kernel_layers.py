from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
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
    ResolvedGeometryContext,
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
    VisibilitySpan,
    partition_visibility,
)


ROOT = Path(__file__).resolve().parents[1]


class GeometryContextTests(unittest.TestCase):
    def test_resolution_exactly_preserves_legacy_policy_values(self) -> None:
        policies = (
            TolerancePolicy(),
            TolerancePolicy(
                relative=3.0e-8,
                absolute_floor=2.0e-15,
                angular=4.0e-11,
                boundary_factor=5.0,
                depth_factor=7.0,
            ),
        )
        scales = (0.0, 1.0e-20, 1.0e-10, 1.0e-6, 1.0, 1.0e6)
        for policy in policies:
            for scale in scales:
                positions = ((0.0, 0.0, 0.0), (scale, 0.0, 0.0))
                for edge_length in (None, scale, 2.0 * scale):
                    with self.subTest(
                        policy=policy,
                        scale=scale,
                        edge_length=edge_length,
                    ):
                        legacy = policy.resolve(
                            positions,
                            edge_length=edge_length,
                        )
                        context = GeometryContext(policy).resolve(
                            positions,
                            edge_length=edge_length,
                        )
                        self.assertEqual(context.resolved, legacy)
                        self.assertEqual(
                            context.epsilon(GeometryQuantity.LENGTH),
                            legacy.world,
                        )
                        self.assertEqual(
                            context.epsilon(GeometryQuantity.BOUNDARY),
                            legacy.boundary,
                        )
                        self.assertEqual(
                            context.epsilon(GeometryQuantity.DEPTH),
                            legacy.depth,
                        )
                        self.assertEqual(
                            context.epsilon(GeometryQuantity.PARAMETER),
                            legacy.parameter,
                        )
                        self.assertEqual(
                            context.epsilon(GeometryQuantity.ANGULAR),
                            legacy.angular,
                        )

    def test_small_geometry_is_not_clamped_to_unit_scale(self) -> None:
        policy = TolerancePolicy()
        positions = ((0.0, 0.0, 0.0), (1.0e-10, 0.0, 0.0))
        legacy = policy.resolve(positions, edge_length=1.0e-10)
        context = GeometryContext(policy).resolve(
            positions,
            edge_length=1.0e-10,
        )
        self.assertEqual(context.epsilon(GeometryQuantity.LENGTH), 1.0e-14)
        self.assertEqual(context.epsilon(GeometryQuantity.LENGTH), legacy.world)
        self.assertEqual(
            context.epsilon(GeometryQuantity.PARAMETER),
            legacy.parameter,
        )

    def test_resolved_context_is_not_resolved_twice(self) -> None:
        resolved = GeometryContext().resolve()
        self.assertIs(resolve_geometry_context(resolved), resolved)
        with self.assertRaises(ValueError):
            resolve_geometry_context(
                resolved,
                positions=((0.0, 0.0, 0.0),),
            )

    def test_quantity_overrides_are_explicit_and_local(self) -> None:
        context = GeometryContext(
            overrides={GeometryQuantity.PARAMETER: 2.5e-6},
        ).resolve()
        self.assertEqual(
            context.epsilon(GeometryQuantity.PARAMETER),
            2.5e-6,
        )
        self.assertEqual(
            context.epsilon(GeometryQuantity.LENGTH),
            context.resolved.world,
        )

    def test_coordinate_scale_keeps_subunit_magnitudes(self) -> None:
        self.assertEqual(coordinate_scale([[1.0e-10, -4.0e-10]]), 4.0e-10)
        with self.assertRaises(ValueError):
            coordinate_scale([float("inf")])

    def test_resolved_context_rejects_non_finite_screen_tolerance(self) -> None:
        with self.assertRaises(ValueError):
            GeometryContext(screen_tolerance=float("nan"))
        self.assertIsInstance(GeometryContext().resolve(), ResolvedGeometryContext)


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

    def test_equal_sort_keys_use_first_authored_order(self) -> None:
        domain = ParameterInterval(0.0, 1.0)
        hidden = [
            OcclusionInterval(domain, "face-b"),
            OcclusionInterval(domain, "face-a"),
        ]
        spans = partition_visibility(
            domain,
            hidden,
            parameter_tolerance=0.0,
            occluder_key=lambda _owner: 0,
        )
        self.assertEqual(spans[0].occluders, ("face-b", "face-a"))

    def test_equal_sort_keys_are_hash_seed_independent(self) -> None:
        script = r'''
import json
from polyhedron_visibility.topology import ParameterInterval
from polyhedron_visibility.visibility import OcclusionInterval, partition_visibility

domain = ParameterInterval(0.0, 1.0)
spans = partition_visibility(
    domain,
    [
        OcclusionInterval(domain, "face-b"),
        OcclusionInterval(domain, "face-a"),
        OcclusionInterval(domain, "face-c"),
    ],
    parameter_tolerance=0.0,
    occluder_key=lambda _owner: 0,
)
print(json.dumps(spans[0].occluders))
'''
        outputs: set[str] = set()
        for seed in ("1", "2", "3", "17", "101"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            environment["PYTHONPATH"] = str(ROOT)
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=environment,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            outputs.add(completed.stdout.strip())
        self.assertEqual(outputs, {'["face-b", "face-a", "face-c"]'})

    def test_visibility_span_rejects_illegal_states(self) -> None:
        domain = ParameterInterval(0.0, 1.0)
        with self.assertRaises(TypeError):
            VisibilitySpan(domain, "visible")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            VisibilitySpan(domain, VisibilityKind.VISIBLE, ("face",))
        with self.assertRaises(ValueError):
            VisibilitySpan(domain, VisibilityKind.HIDDEN)
        with self.assertRaises(TypeError):
            VisibilitySpan(
                domain,
                VisibilityKind.HIDDEN,
                ["face"],  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            VisibilitySpan(
                domain,
                VisibilityKind.HIDDEN,
                ("face", "face"),
            )


class CompositorLayerTests(unittest.TestCase):
    def test_unrelated_nodes_keep_authored_order(self) -> None:
        order = stable_topological_sort(["back", "label", "front"], [])
        self.assertEqual(order, ("back", "label", "front"))
        self.assertEqual(painter_ranks(order)["front"], 2)

    def test_equal_keys_keep_authored_order(self) -> None:
        order = stable_topological_sort(
            ["c", "a", "b"],
            [],
            key=lambda _node: 0,
        )
        self.assertEqual(order, ("c", "a", "b"))

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


class ImportBoundaryTests(unittest.TestCase):
    def test_importing_kernel_does_not_import_manim(self) -> None:
        script = """
import sys
import polyhedron_visibility.kernel
assert 'manim' not in sys.modules, sorted(
    name for name in sys.modules if name == 'manim' or name.startswith('manim.')
)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_legacy_root_exports_remain_available_lazily(self) -> None:
        script = """
import sys
import polyhedron_visibility
assert 'manim' not in sys.modules
from polyhedron_visibility import TolerancePolicy
assert TolerancePolicy.__name__ == 'TolerancePolicy'
assert 'manim' not in sys.modules
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


if __name__ == "__main__":
    unittest.main()
