from __future__ import annotations

from math import cos, sin
import unittest

import numpy as np
from manim import Line

from polyhedron_visibility.api import ParallelProjection
from polyhedron_visibility.open_faces import (
    ARTICULATED_HINGE_POLICY,
    OpenFaceAuthoringError,
    OpenFaceOcclusion3D,
    OpenFaceScene3D,
)
from polyhedron_visibility.style import OcclusionStyle


class OpenFaceScene3DAuthoringTests(unittest.TestCase):
    def _builder(self) -> tuple[OpenFaceScene3D, dict[str, float], Line, Line]:
        state = {"theta": 0.0}

        def beta_point(x: float) -> np.ndarray:
            theta = state["theta"]
            return np.array((x, -2.0 * cos(theta), -2.0 * sin(theta)))

        positions = {
            "A": lambda: np.array((-1.0, 0.0, 0.0)),
            "B": lambda: np.array((1.0, 0.0, 0.0)),
            "C": lambda: np.array((1.0, 2.0, 0.0)),
            "D": lambda: np.array((-1.0, 2.0, 0.0)),
            "E": lambda: beta_point(1.0),
            "F": lambda: beta_point(-1.0),
            "M": lambda: 0.25 * positions["A"]() + 0.75 * positions["F"](),
            "N": lambda: 0.25 * positions["B"]() + 0.75 * positions["E"](),
            "P": lambda: np.array((-2.0, 0.0, -1.0)),
            "Q": lambda: np.array((2.0, 0.0, -1.0)),
        }
        hinge_line = Line(positions["A"](), positions["B"](), buff=0)
        excluded_line = Line(positions["M"](), positions["N"](), buff=0)
        builder = OpenFaceScene3D("ordinary-open-dihedral")
        for vertex_id, provider in positions.items():
            builder.vertex(vertex_id, provider)
        builder.face(
            "alpha",
            ("A", "B", "C", "D"),
            logical_surface_id="surface-alpha",
        )
        builder.face(
            "beta",
            ("B", "A", "F", "E"),
            logical_surface_id="surface-beta",
        )
        builder.hinge("axis", "alpha", "beta", "A", "B")
        builder.stroke("AB", "A", "B", hinge_line)
        builder.stroke(
            "MN",
            "M",
            "N",
            excluded_line,
            excluded_occluder_face_ids=("beta",),
        )
        return builder, state, hinge_line, excluded_line

    def test_builder_freezes_explicit_open_face_contract_and_derives_incidence(self) -> None:
        builder, _state, hinge_line, excluded_line = self._builder()
        model = builder.freeze()

        self.assertTrue(builder.frozen)
        self.assertEqual(model.topology, "finite_independent_convex_faces")
        self.assertEqual(model.face_map["alpha"].logical_surface_id, "surface-alpha")
        self.assertEqual(model.seam_map["axis"].policy, ARTICULATED_HINGE_POLICY)
        self.assertEqual(model.stroke_map["AB"].incident_face_ids, ("alpha", "beta"))
        self.assertEqual(model.stroke_map["MN"].incident_face_ids, ())
        self.assertEqual(
            model.stroke_map["MN"].excluded_occluder_face_ids,
            ("beta",),
        )
        self.assertEqual(
            builder.stroke_bindings,
            {"AB": hinge_line, "MN": excluded_line},
        )
        self.assertIs(builder.freeze(), model)
        with self.assertRaisesRegex(OpenFaceAuthoringError, "already frozen"):
            builder.vertex("late", lambda: (0.0, 0.0, 0.0))

    def test_excluded_faces_are_never_inferred_and_are_revalidated_at_freeze(self) -> None:
        builder, state, _hinge_line, _excluded_line = self._builder()
        state["theta"] = 0.35
        model = builder.freeze()
        self.assertEqual(model.stroke_map["AB"].excluded_occluder_face_ids, ())
        self.assertEqual(model.stroke_map["MN"].excluded_occluder_face_ids, ("beta",))

        invalid = OpenFaceScene3D("invalid-exclusion")
        invalid_points = {
            "A": (-1.0, 0.0, 0.0),
            "B": (1.0, 0.0, 0.0),
            "E": (1.0, -2.0, 0.0),
            "F": (-1.0, -2.0, 0.0),
            "M": (-0.5, -1.0, 0.0),
            "N": (0.5, -1.0, 1.0),
        }
        for vertex_id, point in invalid_points.items():
            invalid.vertex(vertex_id, lambda point=point: point)
        invalid.face(
            "beta",
            ("B", "A", "F", "E"),
            logical_surface_id="surface-beta",
        )
        invalid.stroke(
            "MN",
            "M",
            "N",
            Line(invalid_points["M"], invalid_points["N"], buff=0),
            excluded_occluder_face_ids=("beta",),
        )
        with self.assertRaisesRegex(OpenFaceAuthoringError, "coplanar"):
            invalid.freeze()

    def test_duplicate_logical_surface_and_bad_hinge_fail_closed(self) -> None:
        duplicate = OpenFaceScene3D("duplicate-surface")
        duplicate_points = {
            "A": (0, 0, 0),
            "B": (1, 0, 0),
            "C": (0, 1, 0),
            "D": (3, 0, 0),
            "E": (4, 0, 0),
            "F": (3, 1, 0),
        }
        for vertex_id, point in duplicate_points.items():
            duplicate.vertex(vertex_id, lambda point=point: point)
        duplicate.face(
            "one", ("A", "B", "C"), logical_surface_id="same-surface"
        )
        duplicate.face(
            "two", ("D", "E", "F"), logical_surface_id="same-surface"
        )
        with self.assertRaisesRegex(OpenFaceAuthoringError, "one maximal convex face"):
            duplicate.freeze()

        bad = OpenFaceScene3D("bad-hinge")
        points = {
            "A": (0, 0, 0),
            "B": (1, 0, 0),
            "C": (0, 1, 0),
            "D": (0, -1, 0),
            "E": (1, -1, 0),
        }
        for vertex_id, point in points.items():
            bad.vertex(vertex_id, lambda point=point: point)
        bad.face("one", ("A", "B", "C"), logical_surface_id="one")
        bad.face("two", ("B", "A", "D", "E"), logical_surface_id="two")
        bad.hinge("wrong", "one", "two", "A", "C")
        with self.assertRaisesRegex(OpenFaceAuthoringError, "boundary edge"):
            bad.freeze()

    def test_controller_uses_open_solver_binding_without_topology_guessing(self) -> None:
        builder, _state, _hinge_line, _excluded_line = self._builder()

        class SceneProbe:
            mobjects: list[object] = []

        controller = builder.controller(
            SceneProbe(),
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(max_projected_length=8.0),
        )
        self.assertIsInstance(controller, OpenFaceOcclusion3D)
        self.assertEqual(controller.model.visibility_group_id, "ordinary-open-dihedral")
        self.assertFalse(controller.require_closed_convex_manifold)


if __name__ == "__main__":
    unittest.main()
