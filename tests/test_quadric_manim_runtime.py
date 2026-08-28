from __future__ import annotations

import ast
import inspect
import unittest

import numpy as np
from manim import VGroup, VMobject

import polyhedron_visibility.quadrics.composite_authoring as composite_binding
import polyhedron_visibility.quadrics.manim as single_binding
import polyhedron_visibility.quadrics.manim_runtime as runtime


class _FakePainterBand:
    def __init__(self) -> None:
        self.active_state = {"item": 1.0}
        self.restore_calls = 0

    def capture_active_state(self) -> dict[str, float]:
        return dict(self.active_state)

    def restore_active_state(self, state: dict[str, float]) -> None:
        self.active_state = dict(state)
        self.restore_calls += 1


class SharedQuadricManimRuntimeTests(unittest.TestCase):
    def test_public_contracts_remain_reexported_from_manim(self) -> None:
        self.assertIs(single_binding.QuadricBoundaryStyle, runtime.QuadricBoundaryStyle)
        self.assertIs(single_binding.QuadricManimError, runtime.QuadricManimError)
        self.assertIs(
            single_binding.QuadricManimCapacityError,
            runtime.QuadricManimCapacityError,
        )

    def test_both_controllers_use_the_same_slot_and_transaction_runtime(self) -> None:
        for binding in (single_binding, composite_binding):
            with self.subTest(binding=binding.__name__):
                self.assertIs(binding._CurveSlots, runtime._CurveSlots)
                self.assertIs(binding._SurfacePaintSlot, runtime._SurfacePaintSlot)
                self.assertIs(
                    binding._prepare_boundary_fragments,
                    runtime._prepare_boundary_fragments,
                )
                self.assertIs(
                    binding._apply_surface_sheet_pair,
                    runtime._apply_surface_sheet_pair,
                )
                self.assertIs(
                    binding._rollback_display_transaction,
                    runtime._rollback_display_transaction,
                )
                self.assertIs(
                    binding._prepare_display_delta,
                    runtime._prepare_display_delta,
                )
                self.assertIs(
                    binding._apply_display_delta,
                    runtime._apply_display_delta,
                )
        self.assertIs(
            single_binding._apply_opaque_surface_slot,
            runtime._apply_opaque_surface_slot,
        )

    def test_dash_capacity_does_not_allocate_one_mobject_per_dash(self) -> None:
        slots = runtime._CurveSlots(fragment_capacity=32, dash_capacity=100)
        family = slots.root.get_family()

        self.assertEqual(len(family), 97)
        self.assertEqual(
            len(family),
            runtime._curve_slots_family_capacity(32),
        )
        for fragment in slots.fragments:
            self.assertEqual(fragment.dash_capacity, 100)
            self.assertEqual(
                tuple(fragment.root.submobjects),
                (fragment.solid, fragment.dashed),
            )

    def test_compact_dash_vmobject_preserves_disconnected_open_subpaths(
        self,
    ) -> None:
        slot = runtime._CurveFragmentSlot(dash_capacity=8)
        paths = (
            np.asarray(((0.0, 0.0, 0.0), (0.4, 0.0, 0.0))),
            np.asarray(((0.8, 0.0, 0.0), (1.2, 0.2, 0.0))),
            np.asarray(((1.6, 0.2, 0.0), (2.0, 0.2, 0.0))),
        )

        runtime._set_open_subpaths(slot.dashed, paths)
        actual = tuple(slot.dashed.get_subpaths())

        self.assertEqual(len(actual), len(paths))
        for observed, expected in zip(actual, paths):
            np.testing.assert_allclose(observed[0], expected[0], atol=0.0, rtol=0.0)
            np.testing.assert_allclose(
                observed[-1], expected[-1], atol=0.0, rtol=0.0
            )

    def test_display_delta_mutates_only_changed_or_removed_fixed_slots(self) -> None:
        first = VMobject()
        second = VMobject()
        calls: list[str] = []
        first_action = runtime._PreparedDisplayAction(
            "first",
            (first,),
            runtime._display_digest("first", np.asarray((1.0, 2.0))),
            lambda: calls.append("first"),
        )
        second_action = runtime._PreparedDisplayAction(
            "second",
            (second,),
            runtime._display_digest("second", np.asarray((3.0, 4.0))),
            lambda: calls.append("second"),
        )

        initial = runtime._prepare_display_delta({}, (first_action, second_action))
        runtime._apply_display_delta(initial)
        self.assertEqual(calls, ["first", "second"])
        repeated = runtime._prepare_display_delta(
            initial.next_state,
            (first_action, second_action),
        )
        self.assertEqual(repeated.changed, ())
        self.assertEqual(repeated.mutation_roots, ())
        self.assertEqual(repeated.unchanged_slot_ids, ("first", "second"))

        first.set_points_as_corners(
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        ).set_stroke(opacity=1.0)
        changed_second = runtime._PreparedDisplayAction(
            "second",
            (second,),
            runtime._display_digest("second", np.asarray((5.0, 6.0))),
            lambda: calls.append("second-changed"),
        )
        delta = runtime._prepare_display_delta(
            initial.next_state,
            (changed_second,),
        )
        runtime._apply_display_delta(delta)

        self.assertEqual(calls[-1], "second-changed")
        self.assertEqual(len(delta.hidden), 1)
        self.assertEqual(float(first.get_stroke_opacity()), 0.0)
        self.assertEqual(
            {id(root) for root in delta.mutation_roots},
            {id(first), id(second)},
        )

    def test_composite_binding_imports_no_private_manim_runtime_objects(self) -> None:
        tree = ast.parse(inspect.getsource(composite_binding))
        private_imports = tuple(
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "manim"
            for alias in node.names
            if alias.name.startswith("_")
        )
        self.assertEqual(private_imports, ())

    def test_shared_transaction_restores_display_band_and_controller_state(
        self,
    ) -> None:
        path = VMobject().set_points_as_corners(
            np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                dtype=float,
            )
        )
        root = VGroup(path)
        points_before = path.points.copy()
        band = _FakePainterBand()
        controller_state = {"frame": "last-good"}

        def capture_controller_state() -> dict[str, str]:
            return dict(controller_state)

        def restore_controller_state(state: dict[str, str]) -> None:
            controller_state.clear()
            controller_state.update(state)

        with self.assertRaisesRegex(RuntimeError, "synthetic apply failure"):
            with runtime._rollback_display_transaction(
                root,
                band,  # type: ignore[arg-type]
                capture_controller_state=capture_controller_state,
                restore_controller_state=restore_controller_state,
            ):
                path.shift(np.asarray((2.0, 3.0, 0.0), dtype=float))
                band.active_state = {"item": 8.0}
                controller_state["frame"] = "partial"
                raise RuntimeError("synthetic apply failure")

        np.testing.assert_allclose(path.points, points_before, atol=0.0, rtol=0.0)
        self.assertEqual(band.active_state, {"item": 1.0})
        self.assertEqual(band.restore_calls, 1)
        self.assertEqual(controller_state, {"frame": "last-good"})

    def test_shared_transaction_snapshots_only_declared_mutation_roots(self) -> None:
        changed = VMobject().set_points_as_corners(
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        )
        untouched = VMobject().set_points_as_corners(
            np.asarray(((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)))
        )
        root = VGroup(changed, untouched)
        changed_before = changed.points.copy()
        untouched_before = untouched.points.copy()
        band = _FakePainterBand()

        with self.assertRaisesRegex(RuntimeError, "sparse rollback"):
            with runtime._rollback_display_transaction(
                root,
                band,  # type: ignore[arg-type]
                capture_controller_state=lambda: None,
                restore_controller_state=lambda state: None,
                mutation_roots=(changed,),
            ):
                changed.shift(np.asarray((2.0, 0.0, 0.0)))
                raise RuntimeError("sparse rollback")

        np.testing.assert_allclose(changed.points, changed_before, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            untouched.points,
            untouched_before,
            atol=0.0,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
