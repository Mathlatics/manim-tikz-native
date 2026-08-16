from __future__ import annotations

from dataclasses import replace
from math import cos, pi, sin
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import BLUE, Dot, Line, Polygon, Scene, ValueTracker, VGroup, tempconfig

from polyhedron_visibility.api import ParallelProjection
from polyhedron_visibility.binding import OcclusionBindingError
from polyhedron_visibility.open_faces import (
    OPEN_FACE_BINDING_SCALE_LIMITS,
    OpenFaceBindingScaleError,
    OpenFaceOcclusion3D,
    OpenFaceScene3D,
    OpenFaceSolverError,
)
from polyhedron_visibility.style import OcclusionStyle


class _OpenDihedralFixture:
    def __init__(self, scene: Scene, *, initial_angle: float = 0.0) -> None:
        self.scene = scene
        self.angle = ValueTracker(initial_angle)
        self.invalid = False

        self.alpha_points = (
            np.array((-1.0, 0.0, 0.0)),
            np.array((1.0, 0.0, 0.0)),
            np.array((1.0, 2.0, 0.0)),
            np.array((-1.0, 2.0, 0.0)),
        )

        def beta_far(x: float) -> np.ndarray:
            if self.invalid:
                return self.alpha_points[0 if x < 0 else 1].copy()
            theta = self.angle.get_value()
            return np.array((x, -2.0 * cos(theta), -2.0 * sin(theta)))

        self.providers = {
            "A": lambda: self.alpha_points[0].copy(),
            "B": lambda: self.alpha_points[1].copy(),
            "C": lambda: self.alpha_points[2].copy(),
            "D": lambda: self.alpha_points[3].copy(),
            "E": lambda: beta_far(1.0),
            "F": lambda: beta_far(-1.0),
            "P": lambda: np.array((-2.0, 0.0, -1.0)),
            "Q": lambda: np.array((2.0, 0.0, -1.0)),
        }
        self.alpha = Polygon(*self.alpha_points, fill_opacity=0.12, stroke_opacity=0.35)
        self.beta = Polygon(
            self.providers["B"](),
            self.providers["A"](),
            self.providers["F"](),
            self.providers["E"](),
            color=BLUE,
            fill_opacity=0.12,
            stroke_opacity=0.35,
        ).set_z_index(1)
        self.source = Line(
            self.providers["P"](),
            self.providers["Q"](),
            buff=0,
            stroke_width=7,
            stroke_opacity=0.63,
        ).set_z_index(10)
        self.geometry = VGroup(self.alpha, self.beta, self.source)
        scene.add(self.geometry)

        self.builder = OpenFaceScene3D("ordinary-manim-dihedral")
        for vertex_id, provider in self.providers.items():
            self.builder.vertex(vertex_id, provider)
        self.builder.face(
            "alpha",
            ("A", "B", "C", "D"),
            logical_surface_id="surface-alpha",
        )
        self.builder.face(
            "beta",
            ("B", "A", "F", "E"),
            logical_surface_id="surface-beta",
        )
        self.builder.articulated_hinge("axis", "alpha", "beta", "A", "B")
        self.builder.stroke("probe", "P", "Q", self.source)
        self.controller = self.builder.controller(
            scene,
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(
                max_projected_length=8.0,
                dash_length=0.12,
                dash_gap=0.08,
            ),
        )

    def update_beta_mobject(self) -> None:
        replacement = Polygon(
            self.providers["B"](),
            self.providers["A"](),
            self.providers["F"](),
            self.providers["E"](),
            color=BLUE,
            fill_opacity=0.12,
            stroke_opacity=0.35,
        ).set_z_index(1)
        self.beta.become(replacement)


class OpenFaceOcclusion3DManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig({"renderer": "cairo", "frame_rate": 12})
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_exact_zero_and_pi_hinge_frames_are_successful(self) -> None:
        fixture = _OpenDihedralFixture(Scene(), initial_angle=0.0)
        controller = fixture.controller.attach()

        self.assertIsInstance(controller, OpenFaceOcclusion3D)
        self.assertEqual(
            controller.last_frame.seam_state_map["axis"].state,
            "coplanar_same_normal",
        )
        fixture.angle.set_value(pi)
        controller.update()
        self.assertEqual(
            controller.last_frame.seam_state_map["axis"].state,
            "coplanar_opposite_normal",
        )
        self.assertAlmostEqual(
            controller.last_frame.seam_state_map["axis"].dihedral_radians,
            pi,
        )
        controller.restore()

    def test_invalid_dynamic_frame_preserves_slots_and_last_good_then_recovers(self) -> None:
        fixture = _OpenDihedralFixture(Scene(), initial_angle=0.45)
        controller = fixture.controller.attach()
        snapshot = controller.slot_snapshot()
        last_good = controller.last_frame
        fixture.invalid = True

        with self.assertRaises(OpenFaceSolverError):
            controller.update()
        self.assertEqual(controller.slot_snapshot(), snapshot)
        self.assertIs(controller.last_frame, last_good)
        self.assertTrue(controller.attached)

        fixture.invalid = False
        fixture.angle.set_value(0.8)
        controller.update()
        self.assertIsNot(controller.last_frame, last_good)
        controller.restore()

    def test_exception_session_restores_source_and_preserves_unrelated_scene_state(self) -> None:
        fixture = _OpenDihedralFixture(Scene(), initial_angle=0.3)
        marker = Dot((2.5, 1.0, 0.0))

        with self.assertRaisesRegex(RuntimeError, "author failure"):
            with fixture.controller.session():
                fixture.scene.add(marker)
                raise RuntimeError("author failure")

        self.assertFalse(fixture.controller.attached)
        self.assertNotIn(fixture.controller.overlay_root, fixture.scene.mobjects)
        self.assertIn(marker, fixture.scene.mobjects)
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), 0.63)

    def test_inherits_complete_line_cairo_and_scene_owner_gates(self) -> None:
        fixture = _OpenDihedralFixture(Scene(), initial_angle=0.3)
        roots = tuple(fixture.scene.mobjects)
        opacity = float(fixture.source.get_stroke_opacity())

        with self.assertRaisesRegex(OcclusionBindingError, "complete straight Manim Line"):
            OpenFaceOcclusion3D(
                fixture.scene,
                fixture.builder.freeze(),
                position_provider=fixture.builder.current_positions,
                stroke_bindings={
                    "probe": Polygon((-2, 0, -1), (2, 0, -1), (0, 1, -1))
                },
                projection=ParallelProjection.identity(),
                style=OcclusionStyle(max_projected_length=8.0),
            )
        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), opacity)

        with patch(
            "polyhedron_visibility.binding._using_cairo_renderer",
            return_value=False,
        ), self.assertRaisesRegex(OcclusionBindingError, "Cairo"):
            fixture.controller.attach()
        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), opacity)

        detached = Line((-2, 0, -1), (2, 0, -1), buff=0).set_z_index(22)
        controller = OpenFaceOcclusion3D(
            fixture.scene,
            fixture.builder.freeze(),
            position_provider=fixture.builder.current_positions,
            stroke_bindings={"probe": detached},
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(max_projected_length=8.0),
        )
        with self.assertRaisesRegex(OcclusionBindingError, "not owned"):
            controller.attach()
        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        self.assertAlmostEqual(float(detached.get_stroke_opacity()), 1.0)

    def test_realtime_scale_gate_fails_before_any_overlay_allocation(self) -> None:
        fixture = _OpenDihedralFixture(Scene(), initial_angle=0.3)
        model = fixture.builder.freeze()
        limits = OPEN_FACE_BINDING_SCALE_LIMITS
        self.assertEqual(
            (
                limits.max_faces,
                limits.max_strokes,
                limits.max_seams,
                limits.max_candidate_pairs,
                limits.max_overlay_line_slots,
            ),
            (64, 128, 64, 4096, 65536),
        )
        cases = (
            (
                "faces",
                replace(
                    model,
                    faces=model.faces * (limits.max_faces // len(model.faces) + 1),
                ),
                OcclusionStyle(max_projected_length=8.0),
            ),
            (
                "strokes",
                replace(model, strokes=model.strokes * (limits.max_strokes + 1)),
                OcclusionStyle(max_projected_length=8.0),
            ),
            (
                "seams",
                replace(model, seams=model.seams * (limits.max_seams + 1)),
                OcclusionStyle(max_projected_length=8.0),
            ),
            (
                "candidate_pairs",
                replace(
                    model,
                    faces=model.faces * (limits.max_faces // len(model.faces)),
                    strokes=model.strokes * 65,
                ),
                OcclusionStyle(max_projected_length=8.0),
            ),
            (
                "overlay_line_slots",
                model,
                OcclusionStyle(
                    max_projected_length=8.0,
                    dash_length=1.0e-6,
                    dash_gap=0.0,
                ),
            ),
        )
        for label, oversized, style in cases:
            with self.subTest(label=label), patch(
                "polyhedron_visibility.binding.Line"
            ) as line_constructor, patch(
                "polyhedron_visibility.binding.VGroup"
            ) as group_constructor, self.assertRaisesRegex(
                OpenFaceBindingScaleError,
                f"{label}=.*fixed v1 limit",
            ):
                OpenFaceOcclusion3D(
                    fixture.scene,
                    oversized,
                    position_provider=fixture.builder.current_positions,
                    stroke_bindings={"probe": fixture.source},
                    projection=ParallelProjection.identity(),
                    style=style,
                )
            line_constructor.assert_not_called()
            group_constructor.assert_not_called()

    def test_real_cairo_minimal_fold_renders_and_restores(self) -> None:
        class FoldScene(Scene):
            def construct(inner_self) -> None:
                fixture = _OpenDihedralFixture(inner_self, initial_angle=0.0)
                fixture.beta.add_updater(lambda _mobject: fixture.update_beta_mobject())
                with fixture.controller.session():
                    inner_self.play(
                        fixture.angle.animate.set_value(0.7),
                        run_time=0.2,
                    )
                    inner_self.play(
                        fixture.angle.animate.set_value(pi),
                        run_time=0.2,
                    )
                    inner_self.wait(0.1)
                    inner_self.final_seam_state = (
                        fixture.controller.last_frame.seam_state_map["axis"].state
                    )
                inner_self.overlay_removed = (
                    fixture.controller.overlay_root not in inner_self.mobjects
                )
                inner_self.source_opacity = float(
                    fixture.source.get_stroke_opacity()
                )

        with TemporaryDirectory() as media_dir, tempconfig(
            {
                "renderer": "cairo",
                "media_dir": media_dir,
                "pixel_width": 160,
                "pixel_height": 90,
                "frame_rate": 5,
                "disable_caching": True,
                "write_to_movie": True,
                "save_last_frame": False,
            }
        ):
            scene = FoldScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertEqual(scene.final_seam_state, "coplanar_opposite_normal")
            self.assertTrue(scene.overlay_removed)
            self.assertAlmostEqual(scene.source_opacity, 0.63)


if __name__ == "__main__":
    unittest.main()
