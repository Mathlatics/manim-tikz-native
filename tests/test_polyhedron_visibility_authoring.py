from __future__ import annotations

import unittest

import numpy as np
from manim import Line

from polyhedron_visibility import (
    AutoOcclusion3D,
    OcclusionAuthoringError,
    OcclusionScene3D,
    OcclusionStyle,
    ParallelProjection,
)


POINTS = {
    "A": np.array((1.0, 1.0, 1.0)),
    "B": np.array((1.0, -1.0, -1.0)),
    "C": np.array((-1.0, 1.0, -1.0)),
    "D": np.array((-1.0, -1.0, 1.0)),
}


class OcclusionScene3DAuthoringTests(unittest.TestCase):
    def _scene_model(self) -> tuple[OcclusionScene3D, dict[str, Line]]:
        authoring = OcclusionScene3D("tetrahedron")
        for vertex_id, point in POINTS.items():
            authoring.vertex(vertex_id, lambda point=point: point)
        authoring.face("ABC", ("A", "C", "B"))
        authoring.face("ABD", ("A", "B", "D"))
        authoring.face("ACD", ("A", "D", "C"))
        authoring.face("BCD", ("B", "C", "D"))
        lines: dict[str, Line] = {}
        for start, end in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")):
            edge_id = start + end
            line = Line(POINTS[start], POINTS[end], buff=0)
            lines[edge_id] = line
            authoring.stroke(edge_id, start, end, line)
        return authoring, lines

    def test_builder_derives_incidence_and_freezes_closed_convex_topology(self) -> None:
        authoring, lines = self._scene_model()
        model = authoring.freeze()

        self.assertTrue(authoring.frozen)
        self.assertEqual(set(model.stroke_map["AB"].incident_face_ids), {"ABC", "ABD"})
        self.assertEqual(set(authoring.stroke_bindings), set(lines))
        self.assertIs(authoring.freeze(), model)
        with self.assertRaisesRegex(OcclusionAuthoringError, "already frozen"):
            authoring.vertex("E", lambda: (0, 0, 0))

    def test_public_top_level_api_builds_controller_without_guessing_vgroup(self) -> None:
        authoring, _lines = self._scene_model()

        class SceneProbe:
            mobjects: list[object] = []

        controller = authoring.controller(
            SceneProbe(),
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(max_projected_length=4.0),
        )
        self.assertIsInstance(controller, AutoOcclusion3D)
        self.assertEqual(controller.model.visibility_group_id, "tetrahedron")

    def test_open_face_system_requires_an_explicit_experimental_mode(self) -> None:
        strict = OcclusionScene3D("open")
        for vertex_id, point in {"A": (0, 0, 0), "B": (1, 0, 0), "C": (0, 1, 0)}.items():
            strict.vertex(vertex_id, lambda point=point: point)
        strict.face("ABC", ("A", "B", "C"))
        with self.assertRaisesRegex(OcclusionAuthoringError, "closed two-manifold"):
            strict.freeze()

        experimental = OcclusionScene3D(
            "open", topology_mode="independent_convex_faces"
        )
        for vertex_id, point in {"A": (0, 0, 0), "B": (1, 0, 0), "C": (0, 1, 0)}.items():
            experimental.vertex(vertex_id, lambda point=point: point)
        experimental.face("ABC", ("A", "B", "C"))
        experimental.freeze()


if __name__ == "__main__":
    unittest.main()
