from __future__ import annotations

import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import (
    BLUE,
    CapStyleType,
    Dot,
    Line,
    LineJointType,
    Polygon,
    RED,
    Rotate,
    Scene,
    ThreeDScene,
    ValueTracker,
    VGroup,
    tempconfig,
)
from manim.animation.animation import prepare_animation

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.api import AutoOcclusion3D, ParallelProjection
from polyhedron_visibility.binding import OcclusionBindingError, OcclusionCapacityError
from polyhedron_visibility.style import OcclusionStyle
from polyhedron_visibility.style import OcclusionStyleError


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
        self.source.set_z_index(10)
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
        fixture.source.set_stroke(
            color=RED,
            width=11,
            opacity=0.44,
            background=True,
        )
        controller = fixture.controller.attach()

        self.assertTrue(controller.attached)
        self.assertIs(controller.overlay_root, fixture.scene.mobjects[-1])
        self.assertNotIn(controller.overlay_root, fixture.geometry.get_family())
        self.assertEqual(
            list(inspect.signature(controller.overlay_root.updaters[0]).parameters),
            ["mobject", "dt"],
        )
        self.assertEqual(fixture.source.get_stroke_opacity(), 0)
        self.assertEqual(fixture.source.get_stroke_opacity(background=True), 0)
        self.assertEqual(controller.slot_counts("probe"), (2, 1))
        slot_lines = [
            *controller._slots["probe"].visible,
            *(line for slot in controller._slots["probe"].hidden for line in slot),
        ]
        self.assertTrue(all(line.cap_style == CapStyleType.ROUND for line in slot_lines))
        self.assertTrue(all(line.joint_type == LineJointType.ROUND for line in slot_lines))
        for line in slot_lines:
            np.testing.assert_allclose(
                line.background_stroke_rgbas[0, :3],
                RED.to_rgb(),
                atol=1e-12,
            )
        self.assertTrue(all(line.get_stroke_width(background=True) == 11 for line in slot_lines))
        for line in slot_lines:
            expected_background_opacity = (
                0.44 if float(line.get_stroke_opacity()) > 0 else 0.0
            )
            self.assertAlmostEqual(
                float(line.get_stroke_opacity(background=True)),
                expected_background_opacity,
            )
        identities = controller.slot_identities()

        for index in range(100):
            fixture.projection_tracker.set_value(0.15 * np.sin(index / 9))
            controller.overlay_root.update(1 / 60)

        self.assertEqual(controller.slot_identities(), identities)
        controller.restore()
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), 0.63)
        self.assertAlmostEqual(
            float(fixture.source.get_stroke_opacity(background=True)),
            0.44,
        )

    def test_gradient_strokes_fail_closed_before_source_or_scene_mutation(self) -> None:
        for background in (False, True):
            with self.subTest(background=background):
                fixture = _Fixture(_InstantScene())
                fixture.source.set_stroke(
                    color=[RED, BLUE],
                    background=background,
                )
                roots = tuple(fixture.scene.mobjects)
                foreground_opacity = float(fixture.source.get_stroke_opacity())
                background_opacity = float(
                    fixture.source.get_stroke_opacity(background=True)
                )

                with self.assertRaisesRegex(OcclusionStyleError, "gradient"):
                    fixture.controller.attach()

                self.assertEqual(tuple(fixture.scene.mobjects), roots)
                self.assertAlmostEqual(
                    float(fixture.source.get_stroke_opacity()),
                    foreground_opacity,
                )
                self.assertAlmostEqual(
                    float(fixture.source.get_stroke_opacity(background=True)),
                    background_opacity,
                )
                self.assertFalse(fixture.controller.attached)

    def test_group_style_uses_drawable_members_and_rejects_mixed_styles(self) -> None:
        first = Line((-1, 0, 0), (0, 0, 0), color=RED, stroke_width=9)
        second = Line((0, 0, 0), (1, 0, 0), color=RED, stroke_width=9)
        first.set_stroke(opacity=0.37)
        second.set_stroke(opacity=0.37)
        style = OcclusionStyle(max_projected_length=4.0)

        resolved = style.resolve_for(VGroup(first, second))
        self.assertAlmostEqual(resolved.visible_width, 9.0)
        self.assertAlmostEqual(resolved.visible_opacity, 0.37)
        np.testing.assert_allclose(resolved.visible_color.to_rgb(), RED.to_rgb())

        second.set_stroke(color=BLUE)
        with self.assertRaisesRegex(OcclusionStyleError, "share one style"):
            style.resolve_for(VGroup(first, second))

    def test_compound_or_non_line_source_fails_before_scene_mutation(self) -> None:
        fixture = _Fixture(_InstantScene())
        roots = tuple(fixture.scene.mobjects)
        with self.assertRaisesRegex(OcclusionBindingError, "complete straight Manim Line"):
            AutoOcclusion3D(
                fixture.scene,
                _model(),
                position_provider=lambda: {
                    "a": (-2, 0, 0),
                    "b": (2, 0, 0),
                    "p0": (-1, -1, 1),
                    "p1": (1, -1, 1),
                    "p2": (1, 1, 1),
                    "p3": (-1, 1, 1),
                },
                stroke_bindings={
                    "probe": Polygon((-2, 0, 0), (2, 0, 0), (0, -0.5, 0))
                },
                projection=ParallelProjection.identity(),
                style=OcclusionStyle(max_projected_length=6.0),
            )
        self.assertEqual(tuple(fixture.scene.mobjects), roots)

    def test_detached_source_line_cannot_create_new_scene_content(self) -> None:
        fixture = _Fixture(_InstantScene())
        detached = Line((-2, 0, 0), (2, 0, 0)).set_z_index(31)
        controller = AutoOcclusion3D(
            fixture.scene,
            _model(),
            position_provider=lambda: {
                "a": (-2, 0, 0),
                "b": (2, 0, 0),
                "p0": (-1, -1, 1),
                "p1": (1, -1, 1),
                "p2": (1, 1, 1),
                "p3": (-1, 1, 1),
            },
            stroke_bindings={"probe": detached},
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(max_projected_length=6.0),
        )
        roots = tuple(fixture.scene.mobjects)
        with self.assertRaisesRegex(OcclusionBindingError, "not owned"):
            controller.attach()
        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        self.assertFalse(controller.attached)

    def test_source_line_must_match_the_registered_entry_segment(self) -> None:
        fixture = _Fixture(_InstantScene())
        wrong = Line((-2, 0.4, 0), (2, 0.4, 0)).set_z_index(31)
        fixture.scene.add(wrong)
        controller = AutoOcclusion3D(
            fixture.scene,
            _model(),
            position_provider=lambda: {
                "a": (-2, 0, 0),
                "b": (2, 0, 0),
                "p0": (-1, -1, 1),
                "p1": (1, -1, 1),
                "p2": (1, 1, 1),
                "p3": (-1, 1, 1),
            },
            stroke_bindings={"probe": wrong},
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(max_projected_length=6.0),
        )
        roots = tuple(fixture.scene.mobjects)
        with self.assertRaisesRegex(OcclusionBindingError, "registered straight segment"):
            controller.attach()
        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        self.assertAlmostEqual(float(wrong.get_stroke_opacity()), 1.0)

    def test_dynamic_source_mismatch_preserves_the_last_good_overlay(self) -> None:
        fixture = _Fixture(_InstantScene())
        fixed_positions = {
            "a": (-2, 0, 0),
            "b": (2, 0, 0),
            "p0": (-1, -1, 1),
            "p1": (1, -1, 1),
            "p2": (1, 1, 1),
            "p3": (-1, 1, 1),
        }
        controller = AutoOcclusion3D(
            fixture.scene,
            _model(),
            position_provider=lambda: fixed_positions,
            stroke_bindings={"probe": fixture.source},
            projection=ParallelProjection.identity(),
            style=OcclusionStyle(max_projected_length=6.0),
        ).attach()
        last_good = controller.slot_snapshot()
        fixture.source.shift((0, 0.4, 0))

        with self.assertRaisesRegex(OcclusionBindingError, "registered straight segment"):
            controller.update()

        self.assertEqual(controller.slot_snapshot(), last_good)
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), 0.0)
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

    def test_same_z_drawable_rejected_and_distinct_z_accepted_transactionally(self) -> None:
        ambiguous = _Fixture(_InstantScene())
        ambiguous.source.set_z_index(ambiguous.face.z_index)
        roots = tuple(ambiguous.scene.mobjects)
        opacity = float(ambiguous.source.get_stroke_opacity())
        with self.assertRaisesRegex(Exception, "z_index"):
            ambiguous.controller.attach()
        self.assertEqual(tuple(ambiguous.scene.mobjects), roots)
        self.assertAlmostEqual(float(ambiguous.source.get_stroke_opacity()), opacity)

        distinct = _Fixture(_InstantScene())
        distinct.source.set_z_index(23)
        distinct.controller.attach()
        slot_family = distinct.controller._slots["probe"].root.get_family()
        self.assertEqual({float(item.z_index) for item in slot_family}, {23.0})
        distinct.controller.restore()

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

    def test_attach_tail_rollback_removes_overlay_family_from_every_scene_container(self) -> None:
        fixture = _Fixture(_InstantScene())
        original_invalidate = fixture.controller._invalidate_cairo_static_image
        calls = 0

        def fail_once_after_scene_insertion() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                fixture.scene.moving_mobjects.extend(
                    fixture.controller.overlay_root.get_family()
                )
                fixture.scene.static_mobjects.extend(
                    fixture.controller.overlay_root.get_family()[1:]
                )
                raise RuntimeError("attach tail failure")
            original_invalidate()

        fixture.controller._invalidate_cairo_static_image = fail_once_after_scene_insertion
        with self.assertRaisesRegex(RuntimeError, "attach tail failure"):
            fixture.controller.attach()

        overlay_family_ids = {
            id(item) for item in fixture.controller.overlay_root.get_family()
        }
        for container in (
            fixture.scene.mobjects,
            fixture.scene.foreground_mobjects,
            fixture.scene.moving_mobjects,
            fixture.scene.static_mobjects,
        ):
            self.assertFalse(any(id(item) in overlay_family_ids for item in container))
        self.assertFalse(fixture.controller.attached)
        self.assertAlmostEqual(float(fixture.source.get_stroke_opacity()), 0.63)

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
        filled_source = Line(
            (-2, 0, 0),
            (2, 0, 0),
            stroke_opacity=0.63,
            stroke_width=7,
        )
        filled_source.set_fill(opacity=0.47)
        scene.remove(fixture.geometry)
        fixture.geometry = VGroup(fixture.face, filled_source)
        fixture.source = filled_source
        fixture.source.set_z_index(17)
        scene.add(fixture.geometry)

        def positions():
            vertices = fixture.face.get_vertices()
            return {
                "a": filled_source.get_start(),
                "b": filled_source.get_end(),
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
                overlay_family_ids = {
                    id(item) for item in fixture.controller.overlay_root.get_family()
                }
                inner_self.overlay_removed_from_all_containers = all(
                    all(id(item) not in overlay_family_ids for item in container)
                    for container in (
                        inner_self.mobjects,
                        inner_self.foreground_mobjects,
                        inner_self.moving_mobjects,
                        inner_self.static_mobjects,
                    )
                )
                # Regression: stale family members used to survive in Cairo's
                # moving/static containers and poison the next animation.
                marker = Dot((2, 1, 0))
                inner_self.add(marker)
                inner_self.play(marker.animate.shift((0.2, 0, 0)), run_time=0.12)
                inner_self.wait(0.12)
                inner_self.marker_center = marker.get_center().copy()

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
            self.assertTrue(scene.overlay_removed_from_all_containers)
            np.testing.assert_allclose(scene.marker_center, (2.2, 1.0, 0.0), atol=1e-9)

    def test_real_three_d_scene_alignment_for_world_and_fixed_display_paths(self) -> None:
        for custom_display in (False, True):
            with self.subTest(custom_display=custom_display), TemporaryDirectory() as media_dir, tempconfig(
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
                class AlignmentScene(ThreeDScene):
                    def construct(inner_self) -> None:
                        inner_self.set_camera_orientation(phi=0.91, theta=-0.63)
                        fixture = _Fixture(inner_self)
                        if custom_display:
                            fixture.controller.display_point_provider = lambda world: (
                                inner_self.camera.project_points(
                                    np.asarray([world], dtype=float)
                                )[0]
                            )
                            display = fixture.controller.display_point_provider
                        else:
                            display = lambda world: np.asarray(world, dtype=float)

                        fixture.controller.attach()
                        start = fixture.source.get_start()
                        end = fixture.source.get_end()
                        delta = end - start
                        visible_spans = [
                            span
                            for span in fixture.controller.last_frame.edge_map["probe"].spans
                            if span.kind == "visible"
                        ]
                        inner_self.endpoint_aligned = True
                        for line, span in zip(
                            fixture.controller._slots["probe"].visible,
                            visible_spans,
                        ):
                            expected_start = display(start + span.start * delta)
                            expected_end = display(start + span.end * delta)
                            inner_self.endpoint_aligned &= bool(
                                np.allclose(line.get_start(), expected_start, atol=1e-9)
                                and np.allclose(line.get_end(), expected_end, atol=1e-9)
                            )

                        overlay_family = fixture.controller.overlay_root.get_family()
                        fixed = inner_self.camera.fixed_in_frame_mobjects
                        inner_self.fixed_registration_during = all(
                            item in fixed for item in overlay_family
                        )
                        inner_self.wait(0.2)
                        fixture.controller.restore()
                        inner_self.fixed_registration_after = any(
                            item in fixed for item in overlay_family
                        )

                scene = AlignmentScene()
                scene.render()
                self.assertTrue(scene.endpoint_aligned)
                self.assertEqual(scene.fixed_registration_during, custom_display)
                self.assertFalse(scene.fixed_registration_after)
                self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())


if __name__ == "__main__":
    unittest.main()
