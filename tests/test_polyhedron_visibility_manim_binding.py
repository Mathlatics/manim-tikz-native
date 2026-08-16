from __future__ import annotations

import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import (
    CapStyleType,
    Dot,
    Line,
    LineJointType,
    Polygon,
    Rotate,
    Scene,
    ValueTracker,
    VGroup,
    tempconfig,
)
from manim.animation.animation import prepare_animation

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.api import AutoOcclusion3D, ParallelProjection
from polyhedron_visibility.binding import OcclusionCapacityError
from polyhedron_visibility.style import OcclusionStyle


IDENTITY_PROJECTION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class _InstantScene(Scene):
    def play(self, *builders, **_kwargs) -> None:  # type: ignore[override]
        animations = [prepare_animation(builder) for builder in builders]
        for animation in animations:
            animation.begin()
        for alpha in (0.0, 0.5, 1.0):
            for animation in animations:
                animation.interpolate(alpha)
            for mobject in tuple(self.mobjects):
                mobject.update(0.0)
        for animation in animations:
            animation.finish()
        # This is the update Manim performs after finish().  A Scene updater
        # would miss it; the independent overlay Mobject updater must not.
        for mobject in tuple(self.mobjects):
            mobject.update(0.0)

    def wait(self, *_args, **_kwargs) -> None:  # type: ignore[override]
        for mobject in tuple(self.mobjects):
            mobject.update(0.0)


def _model() -> VisibilityModel:
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "manim-fixture",
            "vertices": [
                {"vertexId": "a", "entryPosition": [-2, 0, 0]},
                {"vertexId": "b", "entryPosition": [2, 0, 0]},
                {"vertexId": "p0", "entryPosition": [-1, -1, 1]},
                {"vertexId": "p1", "entryPosition": [1, -1, 1]},
                {"vertexId": "p2", "entryPosition": [1, 1, 1]},
                {"vertexId": "p3", "entryPosition": [-1, 1, 1]},
            ],
            "faces": [{"faceId": "front", "vertexIds": ["p0", "p1", "p2", "p3"]}],
            "strokes": [{"sourceEdgeId": "probe", "vertexIds": ["a", "b"]}],
        }
    )


class _Fixture:
    def __init__(
        self,
        scene: Scene,
        *,
        max_projected_length: float = 6.0,
        project_display: bool = False,
    ) -> None:
        self.scene = scene
        self.face = Polygon(
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
            fill_opacity=0.2,
            stroke_opacity=0.4,
        )
        self.source = Line((-2, 0, 0), (2, 0, 0), stroke_width=7, stroke_opacity=0.63)
        self.geometry = VGroup(self.face, self.source)
        self.scene.add(self.geometry)
        self.projection_tracker = ValueTracker(0.0)

        def positions():
            vertices = self.face.get_vertices()
            return {
                "a": self.source.get_start(),
                "b": self.source.get_end(),
                "p0": vertices[0],
                "p1": vertices[1],
                "p2": vertices[2],
                "p3": vertices[3],
            }

        def projection(_scene):
            angle = self.projection_tracker.get_value()
            cosine, sine = np.cos(angle), np.sin(angle)
            return (
                (cosine, 0.0, sine),
                (0.0, 1.0, 0.0),
                (-sine, 0.0, cosine),
            )

        def display_point(world):
            matrix = np.asarray(projection(self.scene), dtype=float)
            projected = matrix @ np.asarray(world, dtype=float)
            return (projected[0], projected[1], 0.0)

        self.controller = AutoOcclusion3D(
            self.scene,
            _model(),
            position_provider=positions,
            stroke_bindings={"probe": self.source},
            projection=ParallelProjection(projection),
            display_point_provider=display_point if project_display else None,
            style=OcclusionStyle(
                max_projected_length=max_projected_length,
                dash_length=0.12,
                dash_gap=0.08,
            ),
        )


class AutoOcclusion3DManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig({"renderer": "cairo", "frame_rate": 12})
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_attach_uses_independent_root_real_dt_updater_and_fixed_slots(self) -> None:
        fixture = _Fixture(_InstantScene())
        fixture.source.set_cap_style(CapStyleType.ROUND)
        fixture.source.set_joint_type(LineJointType.ROUND)
        controller = fixture.controller.attach()

        self.assertTrue(controller.attached)
        self.assertIs(controller.overlay_root, fixture.scene.mobjects[-1])
        self.assertNotIn(controller.overlay_root, fixture.geometry.get_family())
        self.assertEqual(
            list(inspect.signature(controller.overlay_root.updaters[0]).parameters),
            ["mobject", "dt"],
        )
        self.assertEqual(fixture.source.get_stroke_opacity(), 0)
        self.assertEqual(controller.slot_counts("probe"), (2, 1))
        slot_lines = [
            *controller._slots["probe"].visible,
            *(line for slot in controller._slots["probe"].hidden for line in slot),
        ]
        self.assertTrue(all(line.cap_style == CapStyleType.ROUND for line in slot_lines))
        self.assertTrue(all(line.joint_type == LineJointType.ROUND for line in slot_lines))
        identities = controller.slot_identities()

        for index in range(100):
            fixture.projection_tracker.set_value(0.15 * np.sin(index / 9))
            controller.overlay_root.update(1 / 60)

        self.assertEqual(controller.slot_identities(), identities)
        controller.restore()
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), 0.63)

    def test_display_projection_moves_overlay_when_world_geometry_is_unchanged(self) -> None:
        fixture = _Fixture(_InstantScene(), project_display=True)
        fixture.controller.attach()
        world_points = fixture.source.get_all_points().copy()
        overlay_before = fixture.controller.active_overlay_points("probe")

        fixture.projection_tracker.set_value(0.28)
        fixture.controller.overlay_root.update(1 / 60)

        np.testing.assert_allclose(fixture.source.get_all_points(), world_points)
        self.assertFalse(
            np.allclose(fixture.controller.active_overlay_points("probe"), overlay_before)
        )
        fixture.controller.restore()

    def test_rotate_value_tracker_and_wait_keep_overlay_live_while_target_updaters_suspend(self) -> None:
        fixture = _Fixture(_InstantScene())
        entry_source_points = fixture.source.get_all_points().copy()
        controller = fixture.controller.attach()
        overlay_points_before = controller.active_overlay_points("probe")

        fixture.scene.play(Rotate(fixture.geometry, angle=0.35, axis=(1, 0, 0)), run_time=0.1)
        fixture.scene.play(fixture.projection_tracker.animate.set_value(0.2), run_time=0.1)
        fixture.scene.wait(0.1)

        self.assertFalse(np.allclose(fixture.source.get_all_points(), entry_source_points))
        self.assertFalse(np.allclose(controller.active_overlay_points("probe"), overlay_points_before))
        controller.restore()
        # Restore returns opacity/style only.  The author's Rotate remains.
        self.assertFalse(np.allclose(fixture.source.get_all_points(), entry_source_points))
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), 0.63)

    def test_first_release_fails_closed_outside_cairo_without_scene_mutation(self) -> None:
        fixture = _Fixture(_InstantScene())
        entry_opacity = float(fixture.source.get_stroke_opacity())
        roots = tuple(fixture.scene.mobjects)

        with patch("polyhedron_visibility.binding._using_cairo_renderer", return_value=False):
            with self.assertRaisesRegex(Exception, "Cairo"):
                fixture.controller.attach()

        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), entry_opacity)
        self.assertFalse(fixture.controller.attached)

    def test_capacity_failure_keeps_last_good_frame_and_attach_is_transactional(self) -> None:
        scene = _InstantScene()
        fixture = _Fixture(scene, max_projected_length=4.1)
        fixture.controller.attach()
        last_good = fixture.controller.slot_snapshot()
        last_good_frame = fixture.controller.last_frame
        fixture.source.put_start_and_end_on((-8, 0, 0), (8, 0, 0))

        with self.assertRaises(OcclusionCapacityError):
            fixture.controller.update(0.0)
        self.assertEqual(fixture.controller.slot_snapshot(), last_good)
        self.assertIs(fixture.controller.last_frame, last_good_frame)
        fixture.controller.restore()

        failed = _Fixture(_InstantScene(), max_projected_length=1.0)
        original_opacity = float(failed.source.get_stroke_opacity())
        roots = tuple(failed.scene.mobjects)
        with self.assertRaises(OcclusionCapacityError):
            failed.controller.attach()
        self.assertFalse(failed.controller.attached)
        self.assertEqual(tuple(failed.scene.mobjects), roots)
        self.assertAlmostEqual(float(failed.source.get_stroke_opacity()), original_opacity)

    def test_session_exception_removes_only_its_overlay_and_preserves_new_scene_state(self) -> None:
        fixture = _Fixture(_InstantScene())
        foreground = Dot((0, 2, 0))
        fixture.scene.add_foreground_mobject(foreground)
        added_during_session = Dot((2, 2, 0))

        with self.assertRaisesRegex(RuntimeError, "author failure"):
            with fixture.controller.session():
                fixture.scene.add(added_during_session)
                raise RuntimeError("author failure")

        self.assertFalse(fixture.controller.attached)
        self.assertNotIn(fixture.controller.overlay_root, fixture.scene.mobjects)
        self.assertIn(added_during_session, fixture.scene.mobjects)
        self.assertIn(foreground, fixture.scene.foreground_mobjects)
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), 0.63)

    def test_restore_is_idempotent_and_does_not_remove_unrelated_foreground_or_roots(self) -> None:
        fixture = _Fixture(_InstantScene())
        fixture.scene.add_foreground_mobject(fixture.geometry)
        fixture.controller.attach()
        unrelated = Dot((-2, 2, 0))
        fixture.scene.mobjects.append(unrelated)

        fixture.controller.restore()
        fixture.controller.restore()

        self.assertIn(fixture.geometry, fixture.scene.foreground_mobjects)
        self.assertIn(unrelated, fixture.scene.mobjects)
        self.assertNotIn(fixture.controller.overlay_root, fixture.scene.mobjects)

    def test_source_fill_is_never_hidden_and_slots_inherit_source_z_index(self) -> None:
        scene = _InstantScene()
        fixture = _Fixture(scene)
        filled_source = Polygon(
            (-2, 0, 0),
            (2, 0, 0),
            (0, -0.5, 0),
            fill_opacity=0.47,
            stroke_opacity=0.63,
            stroke_width=7,
        )
        scene.remove(fixture.geometry)
        fixture.geometry = VGroup(fixture.face, filled_source)
        fixture.source = filled_source
        fixture.source.set_z_index(17)
        scene.add(fixture.geometry)

        def positions():
            vertices = fixture.face.get_vertices()
            source_vertices = filled_source.get_vertices()
            return {
                "a": source_vertices[0],
                "b": source_vertices[1],
                "p0": vertices[0],
                "p1": vertices[1],
                "p2": vertices[2],
                "p3": vertices[3],
            }

        controller = AutoOcclusion3D(
            scene,
            _model(),
            position_provider=positions,
            stroke_bindings={"probe": filled_source},
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(max_projected_length=6.0),
        )
        cache_sentinel = object()
        scene.renderer.static_image = cache_sentinel
        controller.attach()

        self.assertAlmostEqual(float(filled_source.get_fill_opacity()), 0.47)
        self.assertIsNone(scene.renderer.static_image)
        slot_family = controller._slots["probe"].root.get_family()
        self.assertEqual({float(item.z_index) for item in slot_family}, {17.0})

        scene.renderer.static_image = cache_sentinel
        controller.restore()
        self.assertAlmostEqual(float(filled_source.get_fill_opacity()), 0.47)
        self.assertAlmostEqual(float(filled_source.get_stroke_opacity()), 0.63)
        self.assertIsNone(scene.renderer.static_image)

    def test_real_cairo_render_accepts_rotate_tracker_and_wait(self) -> None:
        class RenderedScene(Scene):
            def construct(inner_self) -> None:
                fixture = _Fixture(inner_self)
                with fixture.controller.session():
                    inner_self.play(
                        Rotate(fixture.geometry, angle=0.2, axis=(1, 0, 0)),
                        run_time=0.12,
                    )
                    inner_self.play(
                        fixture.projection_tracker.animate.set_value(0.12),
                        run_time=0.12,
                    )
                    inner_self.wait(0.12)

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
            scene = RenderedScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())


if __name__ == "__main__":
    unittest.main()
