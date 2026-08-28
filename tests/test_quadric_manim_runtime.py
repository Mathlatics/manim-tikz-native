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
            single_binding._apply_opaque_surface_slot,
            runtime._apply_opaque_surface_slot,
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


if __name__ == "__main__":
    unittest.main()
