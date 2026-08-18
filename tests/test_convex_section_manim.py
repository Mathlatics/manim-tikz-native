from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import Line, Polygon, Scene, ValueTracker, VGroup, linear, tempconfig

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.api import ParallelProjection
from polyhedron_visibility.sections import (
    ConvexSection3D,
    ConvexSectionBindingScaleError,
    ConvexSectionManimError,
    ConvexSectionScene3D,
    ConvexSectionStyle,
    SectionPlane3D,
)
from polyhedron_visibility.style import OcclusionStyle


_VERTICES = {
    "A": (-1.0, -1.0, -1.0),
    "B": (1.0, -1.0, -1.0),
    "C": (1.0, 1.0, -1.0),
    "D": (-1.0, 1.0, -1.0),
    "E": (-1.0, -1.0, 1.0),
    "F": (1.0, -1.0, 1.0),
    "G": (1.0, 1.0, 1.0),
    "H": (-1.0, 1.0, 1.0),
    "X": (-2.0, 0.0, 0.0),
    "Y": (2.0, 0.0, 0.0),
}

_FACES = {
    "back": ("A", "D", "C", "B"),
    "front": ("E", "F", "G", "H"),
    "bottom": ("A", "B", "F", "E"),
    "right": ("B", "C", "G", "F"),
    "top": ("D", "H", "G", "C"),
    "left": ("A", "E", "H", "D"),
}

_ISOMETRIC = np.asarray(
    (
        (0.7071067811865476, -0.7071067811865476, 0.0),
        (0.4082482904638631, 0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, -0.5773502691896258),
    )
)


def _surface_edges() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                tuple(sorted((start, face[(index + 1) % len(face)])))
                for face in _FACES.values()
                for index, start in enumerate(face)
            }
        )
    )


def _cube_with_probe() -> VisibilityModel:
    incident: dict[tuple[str, str], list[str]] = {
        edge: [] for edge in _surface_edges()
    }
    for face_id, face in _FACES.items():
        for index, start in enumerate(face):
            edge = tuple(sorted((start, face[(index + 1) % len(face)])))
            incident[edge].append(face_id)
    strokes = [
        {
            "sourceEdgeId": f"edge.{start}.{end}",
            "vertexIds": [start, end],
            "incidentFaceIds": sorted(incident[(start, end)]),
        }
        for start, end in _surface_edges()
    ]
    strokes.append(
        {
            "sourceEdgeId": "probe.X.Y",
            "vertexIds": ["X", "Y"],
            "incidentFaceIds": [],
        }
    )
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "sectioned-cube",
            "vertices": [
                {"vertexId": key, "entryPosition": value}
                for key, value in _VERTICES.items()
            ],
            "faces": [
                {"faceId": key, "vertexIds": list(value)}
                for key, value in _FACES.items()
            ],
            "strokes": strokes,
        }
    )


class _SectionFixture:
    def __init__(
        self,
        scene: Scene,
        *,
        initial_offset: float = 4.0,
        half_extent: float = 3.0,
        accurate_transparency: bool = False,
        plane_patch_mode: str = "auto",
    ) -> None:
        self.scene = scene
        self.model = _cube_with_probe()
        self.offset = ValueTracker(initial_offset)
        self.projection_matrix = np.eye(3)
        self.half_extent = half_extent
        self.invalid_contract = False
        self.face_sources: dict[str, Polygon] = {}
        if accurate_transparency:
            for index, face in enumerate(self.model.faces):
                source = Polygon(
                    *[_VERTICES[item] for item in face.vertex_ids],
                    fill_color=("#7BA3D8" if index % 2 else "#8CC8C0"),
                    fill_opacity=0.24,
                    stroke_opacity=0.0,
                )
                source.set_z_index(2.0 + index)
                self.face_sources[face.face_id] = source
            scene.add(VGroup(*self.face_sources.values()))
        self.sources: dict[str, Line] = {}
        for index, stroke in enumerate(self.model.strokes):
            start = np.asarray(_VERTICES[stroke.vertex_ids[0]], dtype=float)
            end = np.asarray(_VERTICES[stroke.vertex_ids[1]], dtype=float)
            source = Line(start, end, buff=0, stroke_width=3.5)
            source.set_z_index(10.0 + index)
            self.sources[stroke.source_edge_id] = source
        self.geometry = VGroup(*self.sources.values())
        scene.add(self.geometry)

        def current_plane() -> SectionPlane3D:
            extent = self.half_extent + (0.25 if self.invalid_contract else 0.0)
            offset = self.offset.get_value()
            return SectionPlane3D(
                "moving-cut",
                (offset / 3.0, offset / 3.0, offset / 3.0),
                (1.0, 1.0, 1.0),
                extent,
                extent,
                u_axis=(1.0, -1.0, 0.0),
            )

        self.controller = ConvexSection3D(
            scene,
            self.model,
            position_provider=lambda: dict(_VERTICES),
            stroke_bindings=self.sources,
            plane_provider=current_plane,
            projection=ParallelProjection(
                lambda _scene: self.projection_matrix
            ),
            source_style=OcclusionStyle(
                max_projected_length=6.0,
                dash_length=0.30,
                dash_gap=0.20,
            ),
            section_style=ConvexSectionStyle(
                max_boundary_projected_length=4.0,
                dash_length=0.24,
                dash_gap=0.16,
            ),
            face_fill_bindings=(
                self.face_sources if accurate_transparency else None
            ),
            accurate_transparency=accurate_transparency,
            plane_patch_mode=plane_patch_mode,
        )


class ConvexSection3DManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig({"renderer": "cairo", "frame_rate": 12})
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_dynamic_topology_uses_stable_slots_and_restores_sources(self) -> None:
        fixture = _SectionFixture(Scene())
        controller = fixture.controller.attach()
        identities = controller.section_slot_identities()
        roots = tuple(fixture.scene.mobjects)
        self.assertEqual(controller.last_sectioned_frame.section.kind, "empty")
        self.assertEqual(
            controller.active_intersection_points("probe.X.Y"),
            ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        )
        self.assertTrue(
            all(float(source.get_stroke_opacity()) == 0 for source in fixture.sources.values())
        )

        samples = (
            (3.0, "point", 1),
            (2.0, "polygon", 3),
            (0.0, "polygon", 6),
            (-2.0, "polygon", 3),
            (-3.0, "point", 1),
            (-4.0, "empty", 0),
        )
        for offset, kind, point_count in samples:
            with self.subTest(offset=offset):
                fixture.offset.set_value(offset)
                controller.update()
                section = controller.last_sectioned_frame.section
                self.assertEqual(section.kind, kind)
                self.assertEqual(len(section.points), point_count)
                self.assertEqual(controller.section_slot_identities(), identities)
                intersection = {
                    item.source_edge_id: item.intersection
                    for item in controller.last_sectioned_frame.stroke_intersections
                }["probe.X.Y"]
                self.assertEqual(intersection.inside_parameter_interval, (0.25, 0.75))

        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        controller.restore()
        self.assertNotIn(controller.overlay_root, fixture.scene.mobjects)
        self.assertTrue(
            all(
                abs(float(source.get_stroke_opacity()) - 1.0) <= 1.0e-12
                for source in fixture.sources.values()
            )
        )

    def test_accurate_transparency_splits_and_locally_orders_every_surface(self) -> None:
        fixture = _SectionFixture(
            Scene(), initial_offset=0.0, accurate_transparency=True
        )
        roots = tuple(fixture.scene.mobjects)
        controller = fixture.controller.attach()
        pool_identities = controller.face_fill_identities()
        frame = controller.last_transparent_compositing
        self.assertIsNotNone(frame)
        assert frame is not None
        roles = {item.role for item in frame.fragments}
        self.assertIn("section_inside", roles)
        self.assertIn("plane_outside", roles)
        self.assertTrue(any(item.startswith("solid_face") for item in roles))
        self.assertEqual(
            set(controller.active_transparent_fragment_ids()),
            set(frame.fragment_map),
        )
        z_indices = controller.active_transparent_fragment_z_indices()
        for relation in frame.order_relations:
            self.assertLess(
                z_indices[relation.far_fragment_id],
                z_indices[relation.near_fragment_id],
            )
        self.assertEqual(float(controller.plane_patch.get_fill_opacity()), 0.0)
        self.assertEqual(float(controller.section_fill.get_fill_opacity()), 0.0)
        self.assertTrue(
            all(
                float(source.get_fill_opacity()) == 0.0
                for source in fixture.face_sources.values()
            )
        )

        for offset, kind in ((2.0, "polygon"), (4.0, "empty"), (-2.0, "polygon"), (0.0, "polygon")):
            with self.subTest(offset=offset):
                fixture.offset.set_value(offset)
                controller.update()
                self.assertEqual(
                    controller.last_transparent_compositing.section.kind,
                    kind,
                )
                self.assertEqual(
                    controller.face_fill_identities(), pool_identities
                )

        controller.restore()
        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        self.assertIsNone(controller.last_transparent_compositing)
        self.assertTrue(
            all(
                abs(float(source.get_fill_opacity()) - 0.24) <= 1.0e-12
                for source in fixture.face_sources.values()
            )
        )

    def test_accurate_transparency_failure_keeps_last_good_fragments(self) -> None:
        fixture = _SectionFixture(
            Scene(), initial_offset=0.0, accurate_transparency=True
        )
        controller = fixture.controller.attach()
        last_good = controller.last_transparent_compositing
        z_indices = controller.active_transparent_fragment_z_indices()
        fixture.invalid_contract = True
        with self.assertRaisesRegex(ConvexSectionManimError, "must stay fixed"):
            controller.update()
        self.assertIs(controller.last_transparent_compositing, last_good)
        self.assertEqual(
            controller.active_transparent_fragment_z_indices(), z_indices
        )
        self.assertTrue(
            all(
                float(source.get_fill_opacity()) == 0.0
                for source in fixture.face_sources.values()
            )
        )
        controller.restore()

    def test_accurate_transparency_reorders_for_a_new_parallel_view(self) -> None:
        fixture = _SectionFixture(
            Scene(), initial_offset=0.0, accurate_transparency=True
        )
        controller = fixture.controller.attach()
        identities = controller.face_fill_identities()
        first_order = controller.last_transparent_compositing.draw_order
        fixture.projection_matrix = _ISOMETRIC
        controller.update()
        second = controller.last_transparent_compositing
        self.assertEqual(
            second.projection_matrix,
            tuple(
                tuple(float(value) for value in row) for row in _ISOMETRIC
            ),
        )
        self.assertNotEqual(second.draw_order, first_order)
        self.assertEqual(controller.face_fill_identities(), identities)
        z_indices = controller.active_transparent_fragment_z_indices()
        self.assertTrue(
            all(
                z_indices[item.far_fragment_id]
                < z_indices[item.near_fragment_id]
                for item in second.order_relations
            )
        )
        controller.restore()

    def test_accurate_transparency_exception_session_restores_faces_and_lines(self) -> None:
        fixture = _SectionFixture(
            Scene(), initial_offset=0.0, accurate_transparency=True
        )
        roots = tuple(fixture.scene.mobjects)
        with self.assertRaisesRegex(RuntimeError, "author failure"):
            with fixture.controller.session():
                raise RuntimeError("author failure")
        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        self.assertFalse(fixture.controller.attached)
        self.assertTrue(
            all(
                abs(float(source.get_fill_opacity()) - 0.24) <= 1.0e-12
                for source in fixture.face_sources.values()
            )
        )
        self.assertTrue(
            all(
                abs(float(source.get_stroke_opacity()) - 1.0) <= 1.0e-12
                for source in fixture.sources.values()
            )
        )

    def test_public_builder_exposes_section_and_free_line_intersection(self) -> None:
        scene = Scene()
        model = _cube_with_probe()
        sources: dict[str, Line] = {}
        for index, stroke in enumerate(model.strokes):
            line = Line(
                _VERTICES[stroke.vertex_ids[0]],
                _VERTICES[stroke.vertex_ids[1]],
                buff=0,
            ).set_z_index(30 + index)
            sources[stroke.source_edge_id] = line
        scene.add(VGroup(*sources.values()))

        visibility = ConvexSectionScene3D("public-section-builder")
        for vertex_id, point in _VERTICES.items():
            visibility.vertex(
                vertex_id,
                lambda point=point: point,
            )
        for face_id, cycle in _FACES.items():
            visibility.face(face_id, cycle)
        for stroke in model.strokes:
            visibility.stroke(
                stroke.source_edge_id,
                stroke.vertex_ids[0],
                stroke.vertex_ids[1],
                sources[stroke.source_edge_id],
                incident_face_ids=stroke.incident_face_ids,
            )
        visibility.cutting_plane(
            "public-cut",
            lambda: SectionPlane3D(
                "public-cut",
                (0, 0, 0),
                (1, 1, 1),
                3.0,
                3.0,
                u_axis=(1, -1, 0),
            ),
        )

        self.assertEqual(len(visibility.current_section().points), 6)
        self.assertEqual(
            visibility.current_stroke_intersections()[
                "probe.X.Y"
            ].inside_parameter_interval,
            (0.25, 0.75),
        )
        controller = visibility.controller(
            scene,
            projection=ParallelProjection.identity(),
            source_style=OcclusionStyle(
                max_projected_length=6.0,
                dash_length=0.30,
                dash_gap=0.20,
            ),
            section_style=ConvexSectionStyle(
                max_boundary_projected_length=4.0,
                dash_length=0.24,
                dash_gap=0.16,
            ),
        ).attach()
        self.assertEqual(len(controller.last_sectioned_frame.section.points), 6)
        controller.restore()

    def test_plane_is_an_additional_global_occluder(self) -> None:
        fixture = _SectionFixture(Scene(), initial_offset=0.0)
        controller = fixture.controller.attach()
        probe = controller.last_sectioned_frame.source_visibility.edge_map[
            "probe.X.Y"
        ]
        self.assertTrue(
            any(item.face_id == "section-plane:moving-cut" for item in probe.raw_intervals)
        )
        self.assertTrue(
            any(
                item.face_id != "section-plane:moving-cut"
                for item in probe.raw_intervals
            )
        )
        self.assertEqual(len(controller.last_sectioned_frame.section.points), 6)
        controller.restore()

    def test_invalid_dynamic_plane_keeps_the_last_good_frame(self) -> None:
        fixture = _SectionFixture(Scene(), initial_offset=0.0)
        controller = fixture.controller.attach()
        last_good = controller.last_sectioned_frame
        points = controller.active_overlay_points("probe.X.Y")
        identities = controller.section_slot_identities()
        fixture.invalid_contract = True

        with self.assertRaisesRegex(ConvexSectionManimError, "must stay fixed"):
            controller.update()

        self.assertIs(controller.last_sectioned_frame, last_good)
        np.testing.assert_allclose(
            controller.active_overlay_points("probe.X.Y"), points
        )
        self.assertEqual(controller.section_slot_identities(), identities)
        self.assertTrue(controller.attached)
        controller.restore()

    def test_small_authored_patch_auto_expands_by_default(self) -> None:
        fixture = _SectionFixture(
            Scene(),
            initial_offset=0.0,
            half_extent=0.1,
            accurate_transparency=True,
        )
        controller = fixture.controller.attach()
        self.assertEqual(controller.last_sectioned_frame.section.kind, "polygon")
        self.assertIsNotNone(controller.last_transparent_compositing)
        self.assertIsNotNone(controller.last_display_plane)
        assert controller.last_display_plane is not None
        self.assertGreater(controller.last_display_plane.half_width, 1.0)
        self.assertGreater(controller.last_display_plane.half_height, 1.0)
        self.assertEqual(
            controller.last_sectioned_frame.section.plane.half_width,
            0.1,
        )
        controller.restore()

    def test_auto_patch_expands_monotonically_as_the_solid_changes(self) -> None:
        scene = Scene()
        model = _cube_with_probe()
        scale = ValueTracker(1.0)

        def positions() -> dict[str, tuple[float, float, float]]:
            factor = scale.get_value()
            return {
                key: tuple(float(item * factor) for item in value)
                for key, value in _VERTICES.items()
            }

        sources: dict[str, Line] = {}
        for index, stroke in enumerate(model.strokes):
            current = positions()
            source = Line(
                current[stroke.vertex_ids[0]],
                current[stroke.vertex_ids[1]],
                buff=0,
            )
            source.set_z_index(10.0 + index)
            sources[stroke.source_edge_id] = source
        scene.add(VGroup(*sources.values()))

        def sync_sources() -> None:
            current = positions()
            for stroke in model.strokes:
                sources[stroke.source_edge_id].put_start_and_end_on(
                    current[stroke.vertex_ids[0]],
                    current[stroke.vertex_ids[1]],
                )

        controller = ConvexSection3D(
            scene,
            model,
            position_provider=positions,
            stroke_bindings=sources,
            plane_provider=lambda: SectionPlane3D(
                "cut", (0, 0, 0), (0, 0, 1), 0.01, 0.01, u_axis=(1, 0, 0)
            ),
            projection=ParallelProjection.identity(),
            source_style=OcclusionStyle(max_projected_length=12.0),
            section_style=ConvexSectionStyle(
                max_boundary_projected_length=8.0
            ),
        ).attach()
        assert controller.last_display_plane is not None
        initial_width = controller.last_display_plane.half_width

        scale.set_value(2.0)
        sync_sources()
        controller.update()
        assert controller.last_display_plane is not None
        expanded_width = controller.last_display_plane.half_width
        self.assertGreater(expanded_width, initial_width)

        scale.set_value(0.5)
        sync_sources()
        controller.update()
        assert controller.last_display_plane is not None
        self.assertEqual(controller.last_display_plane.half_width, expanded_width)
        controller.restore()

    def test_strict_patch_too_small_fails_before_scene_or_source_mutation(self) -> None:
        fixture = _SectionFixture(
            Scene(),
            initial_offset=0.0,
            half_extent=0.1,
            plane_patch_mode="strict",
        )
        roots = tuple(fixture.scene.mobjects)
        opacities = {
            key: float(source.get_stroke_opacity())
            for key, source in fixture.sources.items()
        }
        with self.assertRaisesRegex(
            ConvexSectionManimError, "does not cover"
        ):
            fixture.controller.attach()
        self.assertEqual(tuple(fixture.scene.mobjects), roots)
        self.assertEqual(
            {
                key: float(source.get_stroke_opacity())
                for key, source in fixture.sources.items()
            },
            opacities,
        )

    def test_plane_patch_options_are_strictly_validated(self) -> None:
        with self.assertRaisesRegex(
            ConvexSectionManimError, "plane_patch_mode"
        ):
            _SectionFixture(Scene(), plane_patch_mode="guess")
        fixture = _SectionFixture(Scene())
        with self.assertRaisesRegex(
            ConvexSectionManimError, "plane_patch_margin"
        ):
            ConvexSection3D(
                fixture.scene,
                fixture.model,
                position_provider=lambda: dict(_VERTICES),
                stroke_bindings=fixture.sources,
                plane_provider=lambda: SectionPlane3D(
                    "cut", (0, 0, 0), (0, 0, 1), 1, 1
                ),
                projection=ParallelProjection.identity(),
                source_style=OcclusionStyle(max_projected_length=6.0),
                plane_patch_margin=-0.1,
            )

    def test_realtime_scale_gate_runs_before_overlay_allocation(self) -> None:
        payload = _cube_with_probe().to_dict()
        for index in range(200):
            payload["strokes"].append(
                {
                    "sourceEdgeId": f"extra.{index}",
                    "vertexIds": ["X", "Y"],
                    "incidentFaceIds": [],
                }
            )
        oversized = VisibilityModel.from_dict(payload)
        with patch(
            "polyhedron_visibility.sections.manim.Line"
        ) as section_line, patch(
            "polyhedron_visibility.binding.Line"
        ) as base_line, self.assertRaisesRegex(
            ConvexSectionBindingScaleError,
            "strokes=.*fixed v1 limit",
        ):
            ConvexSection3D(
                Scene(),
                oversized,
                position_provider=lambda: dict(_VERTICES),
                stroke_bindings={},
                plane_provider=lambda: SectionPlane3D(
                    "cut", (0, 0, 0), (1, 1, 1), 3, 3
                ),
                projection=ParallelProjection.identity(),
                source_style=OcclusionStyle(max_projected_length=6.0),
            )
        section_line.assert_not_called()
        base_line.assert_not_called()

    def test_exception_session_restores_every_source_and_overlay_root(self) -> None:
        fixture = _SectionFixture(Scene(), initial_offset=0.0)
        with self.assertRaisesRegex(RuntimeError, "author failure"):
            with fixture.controller.session():
                raise RuntimeError("author failure")
        self.assertFalse(fixture.controller.attached)
        self.assertNotIn(fixture.controller.overlay_root, fixture.scene.mobjects)
        self.assertTrue(
            all(float(source.get_stroke_opacity()) == 1 for source in fixture.sources.values())
        )

    def test_real_cairo_moving_section_renders_and_restores(self) -> None:
        class MovingSectionScene(Scene):
            def construct(inner_self) -> None:
                fixture = _SectionFixture(
                    inner_self,
                    initial_offset=4.0,
                    half_extent=0.02,
                )
                with fixture.controller.session():
                    inner_self.play(
                        fixture.offset.animate.set_value(0.0),
                        run_time=0.4,
                        rate_func=linear,
                    )
                    inner_self.wait(0.1)
                    inner_self.final_point_count = len(
                        fixture.controller.last_sectioned_frame.section.points
                    )
                inner_self.overlay_removed = (
                    fixture.controller.overlay_root not in inner_self.mobjects
                )
                inner_self.sources_restored = all(
                    abs(float(source.get_stroke_opacity()) - 1.0) <= 1.0e-12
                    for source in fixture.sources.values()
                )

        with TemporaryDirectory() as media_dir, tempconfig(
            {
                "renderer": "cairo",
                "media_dir": media_dir,
                "pixel_width": 192,
                "pixel_height": 108,
                "frame_rate": 6,
                "disable_caching": True,
                "write_to_movie": True,
                "save_last_frame": False,
            }
        ):
            scene = MovingSectionScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertEqual(scene.final_point_count, 6)
            self.assertTrue(scene.overlay_removed)
            self.assertTrue(scene.sources_restored)

    def test_real_cairo_exact_transparency_renders_and_restores(self) -> None:
        class AccurateSectionScene(Scene):
            def construct(inner_self) -> None:
                fixture = _SectionFixture(
                    inner_self,
                    initial_offset=3.2,
                    half_extent=0.02,
                    accurate_transparency=True,
                )
                with fixture.controller.session():
                    inner_self.play(
                        fixture.offset.animate.set_value(0.0),
                        run_time=0.4,
                        rate_func=linear,
                    )
                    inner_self.wait(0.1)
                    frame = fixture.controller.last_transparent_compositing
                    inner_self.fragment_count = len(frame.fragments)
                    z_indices = (
                        fixture.controller.active_transparent_fragment_z_indices()
                    )
                    inner_self.local_order_valid = all(
                        z_indices[item.far_fragment_id]
                        < z_indices[item.near_fragment_id]
                        for item in frame.order_relations
                    )
                inner_self.faces_restored = all(
                    abs(float(source.get_fill_opacity()) - 0.24) <= 1.0e-12
                    for source in fixture.face_sources.values()
                )
                inner_self.overlay_removed = (
                    fixture.controller.overlay_root not in inner_self.mobjects
                )

        with TemporaryDirectory() as media_dir, tempconfig(
            {
                "renderer": "cairo",
                "media_dir": media_dir,
                "pixel_width": 192,
                "pixel_height": 108,
                "frame_rate": 6,
                "disable_caching": True,
                "write_to_movie": True,
                "save_last_frame": False,
            }
        ):
            scene = AccurateSectionScene()
            scene.render()
            self.assertTrue(
                Path(scene.renderer.file_writer.movie_file_path).is_file()
            )
            self.assertGreater(scene.fragment_count, 20)
            self.assertTrue(scene.local_order_valid)
            self.assertTrue(scene.faces_restored)
            self.assertTrue(scene.overlay_removed)


if __name__ == "__main__":
    unittest.main()
