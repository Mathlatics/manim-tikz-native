from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from manim import BLUE, GOLD, Line, Polygon, Scene, ValueTracker, VGroup, tempconfig

from polyhedron_visibility import OcclusionStyle, ParallelProjection
from polyhedron_visibility.dihedral_extraction import (
    BasePlaneRotation3D,
    ExtractedDihedralOcclusion3D,
    ExtractedDihedralScene3D,
    RigidTransform3D,
)

from examples.derived_dihedral_extraction.derived_dihedral_extraction_demo import (
    DISPLAY_SCENE_UNITS_PER_CM,
    HIDDEN_DASH_PATTERN_PT,
    TEX_POINTS_PER_CM,
    hidden_dash_pattern_scene_units,
)
from tests.test_derived_dihedral_contract import cube_model


def isometric_projection() -> np.ndarray:
    view = np.asarray((1.0, 1.0, 1.0), dtype=float)
    view /= np.linalg.norm(view)
    right = np.cross(np.asarray((0.0, 0.0, 1.0)), view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    return np.asarray((right, up, view), dtype=float)


class _CubeExtractionFixture:
    def __init__(
        self,
        scene: Scene,
        *,
        accurate_transparency: bool = False,
        unified_compositing: bool | None = None,
        unified_fragment_slots_per_style: int = 12,
        projected_display: bool = False,
        synchronized_base_face: str | None = None,
        identity_handoff_distance: float = 0.12,
    ) -> None:
        self.scene = scene
        self.model = cube_model()
        self.positions = {
            key: np.asarray(value.entry_position, dtype=float)
            for key, value in self.model.vertex_map.items()
        }
        self.shift = ValueTracker(0.0)
        self.base_progress = ValueTracker(0.0)
        self.projection_matrix = isometric_projection()
        self.display = (
            (lambda point: self.projection_matrix @ np.asarray(point, dtype=float))
            if projected_display
            else None
        )

        def shown(point):
            return point if self.display is None else self.display(point)

        self.faces: dict[str, Polygon] = {}
        for index, face in enumerate(self.model.faces):
            polygon = Polygon(
                *(shown(self.positions[item]) for item in face.vertex_ids),
                color=BLUE,
                fill_opacity=0.16,
                stroke_opacity=0.0,
            ).set_z_index(index)
            self.faces[face.face_id] = polygon
        self.lines: dict[str, Line] = {}
        for index, stroke in enumerate(self.model.strokes):
            line = Line(
                shown(self.positions[stroke.vertex_ids[0]]),
                shown(self.positions[stroke.vertex_ids[1]]),
                buff=0,
                stroke_width=4.0,
            ).set_z_index(20 + index)
            self.lines[stroke.source_edge_id] = line
        scene.add(
            VGroup(*(self.faces[key] for key in sorted(self.faces))),
            VGroup(*(self.lines[key] for key in sorted(self.lines))),
        )

        builder = ExtractedDihedralScene3D("cube-extraction")
        for vertex_id in sorted(self.positions):
            builder.vertex(
                vertex_id,
                lambda vertex_id=vertex_id: self.positions[vertex_id],
            )
        for face in self.model.faces:
            builder.face(
                face.face_id,
                face.vertex_ids,
                source_mobject=self.faces[face.face_id],
            )
        for stroke in self.model.strokes:
            builder.stroke(
                stroke.source_edge_id,
                stroke.vertex_ids[0],
                stroke.vertex_ids[1],
                self.lines[stroke.source_edge_id],
                incident_face_ids=stroke.incident_face_ids,
            )
        self.builder = builder
        self.entity = builder.extract_dihedral(
            "copy",
            ("front", "top"),
            transform_provider=lambda: RigidTransform3D.translation_by(
                (0.0, 0.0, self.shift.get_value())
            ),
            edge_color=GOLD,
            face_color=GOLD,
            face_opacity=0.30,
        )
        self.base_rotation: BasePlaneRotation3D | None = (
            builder.base_plane_rotation(synchronized_base_face)
            if synchronized_base_face is not None
            else None
        )
        scene.add(self.entity.mobject)
        self.controller = builder.controller(
            scene,
            projection=ParallelProjection(self.projection_matrix),
            display_point_provider=self.display,
            style=OcclusionStyle(
                max_projected_length=8.0,
                dash_length=0.14,
                dash_gap=0.09,
            ),
            source_coordinate_mode=("display" if projected_display else "world"),
            accurate_transparency=accurate_transparency,
            unified_compositing=unified_compositing,
            unified_fragment_slots_per_style=(
                unified_fragment_slots_per_style
            ),
            global_transform_provider=(
                (
                    lambda: self.base_rotation.transform(
                        self.base_progress.get_value()
                    )
                )
                if self.base_rotation is not None
                else None
            ),
            identity_handoff_distance=identity_handoff_distance,
        )


class ExtractedDihedralManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig({"renderer": "cairo", "frame_rate": 12})
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_demo_dash_pattern_is_exactly_on_2pt_off_2pt(self) -> None:
        dash_length, dash_gap = hidden_dash_pattern_scene_units()

        self.assertEqual(HIDDEN_DASH_PATTERN_PT, (2.0, 2.0))
        self.assertAlmostEqual(
            dash_length,
            2.0 * DISPLAY_SCENE_UNITS_PER_CM / TEX_POINTS_PER_CM,
        )
        self.assertAlmostEqual(
            dash_gap,
            2.0 * DISPLAY_SCENE_UNITS_PER_CM / TEX_POINTS_PER_CM,
        )

    def test_unified_compositing_requires_exact_transparent_faces(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "accurate_transparency"):
            _CubeExtractionFixture(
                Scene(),
                accurate_transparency=False,
                unified_compositing=True,
            )

        legacy_exact = _CubeExtractionFixture(
            Scene(),
            accurate_transparency=True,
            unified_compositing=False,
        )
        controller = legacy_exact.controller.attach()
        self.assertIsNone(controller._unified_layer)
        self.assertIsNone(controller.last_unified_compositing)
        controller.restore()

    def test_identity_handoff_distance_fails_closed_when_invalid(self) -> None:
        for value in (0.0, -0.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "identity_handoff_distance",
            ):
                _CubeExtractionFixture(
                    Scene(),
                    identity_handoff_distance=value,
                )

    def test_unified_managed_stroke_may_occupy_the_authored_face_band(self) -> None:
        fixture = _CubeExtractionFixture(
            Scene(),
            accurate_transparency=True,
        )
        fixture.lines["edge.A.B"].set_z_index(0.5)

        controller = fixture.controller.attach()
        self.assertIsNotNone(controller.last_unified_compositing)
        self.assertTrue(controller.active_unified_z_indices())
        controller.restore()

    @staticmethod
    def _maximum_slot_opacity(controller, edge_id: str) -> float:
        values = []
        for item in controller._slots[edge_id].root.get_family():
            if np.asarray(getattr(item, "points", ()), dtype=float).size == 0:
                continue
            rgba = np.asarray(getattr(item, "stroke_rgbas", ()), dtype=float)
            if rgba.ndim >= 2 and rgba.shape[-1] >= 4:
                values.extend(rgba[..., 3].reshape(-1))
        return max((float(item) for item in values), default=0.0)

    def test_identity_handoff_and_moved_global_occlusion_use_fixed_slots(self) -> None:
        fixture = _CubeExtractionFixture(Scene())
        source_opacity = float(fixture.lines["edge.G.H"].get_stroke_opacity())
        source_face_fill = fixture.faces["front"].fill_rgbas.copy()
        controller = fixture.controller.attach()

        self.assertIsInstance(controller, ExtractedDihedralOcclusion3D)
        self.assertEqual(
            controller.last_extraction_frame.coincident_source_face_ids,
            ("front", "top"),
        )
        self.assertEqual(
            self._maximum_slot_opacity(controller, "solid:edge.G.H"), 0.0
        )
        self.assertGreater(
            self._maximum_slot_opacity(controller, "copy:edge.G.H"), 0.0
        )
        self.assertIsNotNone(controller._face_layer)
        self.assertEqual(
            float(
                controller._face_layer.proxies["solid:front"].get_fill_opacity()
            ),
            0.0,
        )
        self.assertGreater(
            float(controller._face_layer.proxies["copy:front"].get_fill_opacity()),
            0.0,
        )
        self.assertEqual(float(fixture.faces["front"].get_fill_opacity()), 0.0)
        identities = controller.slot_identities()

        fixture.shift.set_value(-2.5)
        controller.update()
        self.assertEqual(
            controller.last_extraction_frame.coincident_source_face_ids,
            (),
        )
        self.assertEqual(controller.slot_identities(), identities)
        self.assertGreater(
            self._maximum_slot_opacity(controller, "solid:edge.G.H"),
            0.0,
        )
        self.assertGreater(
            float(
                controller._face_layer.proxies["solid:front"].get_fill_opacity()
            ),
            0.0,
        )
        self.assertTrue(
            any(
                interval.face_id.startswith("solid:")
                for edge in controller.last_extraction_frame.line_visibility.edges
                if edge.source_edge_id.startswith("copy:")
                for interval in edge.raw_intervals
            )
        )

        controller.restore()
        self.assertFalse(controller.attached)
        self.assertAlmostEqual(
            float(fixture.lines["edge.G.H"].get_stroke_opacity()),
            source_opacity,
        )
        self.assertIn(fixture.entity.mobject, fixture.scene.mobjects)
        np.testing.assert_allclose(
            fixture.faces["front"].fill_rgbas,
            source_face_fill,
        )

    def test_identity_handoff_fades_the_reappearing_source_without_a_binary_jump(self) -> None:
        fixture = _CubeExtractionFixture(
            Scene(),
            identity_handoff_distance=0.20,
        )
        controller = fixture.controller.attach()
        identities = controller.slot_identities()

        self.assertEqual(controller.last_identity_handoff_separation, 0.0)
        self.assertEqual(controller.last_identity_handoff_weight, 0.0)
        self.assertIsNotNone(controller.last_identity_handoff_frame)
        self.assertEqual(
            len(controller.identity_handoff.face_pairs),
            2,
        )
        self.assertEqual(
            len(controller.identity_handoff.stroke_pairs),
            len(fixture.entity.model.extraction.boundary_strokes),
        )
        self.assertEqual(
            set(
                controller.last_identity_handoff_frame.copy_face_opacity_scales.values()
            ),
            {1.0},
        )

        fixture.shift.set_value(-0.10)
        controller.update()
        self.assertAlmostEqual(
            controller.last_identity_handoff_separation,
            0.10,
            places=9,
        )
        self.assertAlmostEqual(controller.last_identity_handoff_weight, 0.5)
        source_stroke_opacity = self._maximum_slot_opacity(
            controller,
            "solid:edge.G.H",
        )
        copy_stroke_opacity = self._maximum_slot_opacity(
            controller,
            "copy:edge.G.H",
        )
        self.assertGreater(source_stroke_opacity, 0.0)
        self.assertLessEqual(source_stroke_opacity, 0.5 + 1.0e-12)
        self.assertGreater(copy_stroke_opacity, source_stroke_opacity)
        self.assertAlmostEqual(
            float(
                controller._face_layer.proxies["solid:front"].get_fill_opacity()
            ),
            0.08,
        )

        fixture.shift.set_value(-0.20)
        controller.update()
        self.assertEqual(controller.last_identity_handoff_weight, 1.0)
        self.assertAlmostEqual(
            float(
                controller._face_layer.proxies["solid:front"].get_fill_opacity()
            ),
            0.16,
        )
        self.assertEqual(controller.slot_identities(), identities)

        fixture.shift.set_value(-0.10)
        controller.update()
        self.assertAlmostEqual(controller.last_identity_handoff_weight, 0.5)
        fixture.shift.set_value(0.0)
        controller.update()
        self.assertEqual(controller.last_identity_handoff_weight, 0.0)
        self.assertEqual(
            controller.last_identity_handoff_frame.source_opacity_scale,
            0.0,
        )
        self.assertEqual(
            self._maximum_slot_opacity(controller, "solid:edge.G.H"),
            0.0,
        )
        controller.restore()

    def test_exact_transparent_handoff_scales_only_the_reappearing_solid_faces(self) -> None:
        fixture = _CubeExtractionFixture(
            Scene(),
            accurate_transparency=True,
            identity_handoff_distance=0.20,
        )
        controller = fixture.controller.attach()
        fixture.shift.set_value(-0.10)
        controller.update()

        self.assertAlmostEqual(controller.last_identity_handoff_weight, 0.5)
        self.assertIsNotNone(controller._prepared_transparent)
        frame = controller._prepared_transparent.frame
        opacities: dict[str, set[float]] = {}
        for batch in controller._prepared_transparent.batches:
            source_ids = {
                frame.fragment_map[fragment_id].source_face_id
                for fragment_id in batch.fragment_ids
            }
            self.assertEqual(len(source_ids), 1)
            source_id = next(iter(source_ids))
            opacities.setdefault(source_id, set()).add(batch.fill_opacity)

        for opacity in opacities["solid:front"] | opacities["solid:top"]:
            self.assertAlmostEqual(opacity, 0.08)
        for opacity in opacities["copy:front"] | opacities["copy:top"]:
            self.assertAlmostEqual(opacity, 0.30)
        controller.restore()

    def test_entity_must_be_scene_owned_before_attach(self) -> None:
        scene = Scene()
        fixture = _CubeExtractionFixture(scene)
        scene.remove(fixture.entity.mobject)

        with self.assertRaisesRegex(RuntimeError, "not owned"):
            fixture.controller.attach()

    def test_exact_transparent_fragments_split_and_restore_transactionally(self) -> None:
        fixture = _CubeExtractionFixture(
            Scene(),
            accurate_transparency=True,
        )
        source_fill = {
            face_id: polygon.fill_rgbas.copy()
            for face_id, polygon in fixture.faces.items()
        }
        controller = fixture.controller.attach()

        self.assertIsNone(controller._face_layer)
        self.assertIsNotNone(controller._transparent_layer)
        self.assertIsNotNone(controller._unified_layer)
        self.assertIsNotNone(controller.last_unified_compositing)
        self.assertIsNotNone(controller.last_transparent_compositing)
        identity_fragments = controller.active_transparent_fragment_ids()
        self.assertTrue(identity_fragments)
        self.assertFalse(
            any(
                controller.last_transparent_compositing.fragment_map[item].surface_id
                in {"solid:front", "solid:top"}
                for item in identity_fragments
            )
        )
        identities = controller.face_fill_identities()

        fixture.shift.set_value(-0.5)
        controller.update()
        moved_fragments = controller.active_transparent_fragment_ids()
        draw_batch_count = controller.active_transparent_draw_batch_count()
        self.assertEqual(controller.face_fill_identities(), identities)
        self.assertGreater(len(moved_fragments), len(identity_fragments))
        self.assertGreater(draw_batch_count, 0)
        self.assertLess(draw_batch_count, len(moved_fragments))
        active_slot_indices = set(
            controller._transparent_layer._fragment_slot_map.values()
        )
        self.assertTrue(
            any(
                len(controller._transparent_layer.slots[index].get_subpaths()) > 1
                for index in active_slot_indices
            )
        )
        self.assertTrue(controller.last_transparent_compositing.order_relations)
        unified_order = controller.active_unified_draw_order()
        unified_z = controller.active_unified_z_indices()
        self.assertEqual(set(unified_order), set(unified_z))
        self.assertEqual(
            tuple(sorted(unified_order, key=unified_z.__getitem__)),
            unified_order,
        )
        unified_kinds = tuple(
            "stroke" if item_id.startswith("stroke:") else "face"
            for item_id in unified_order
        )
        self.assertGreaterEqual(
            sum(
                first != second
                for first, second in zip(unified_kinds, unified_kinds[1:])
            ),
            3,
        )
        self.assertTrue(
            any(
                relation.reason == "stroke_crossing_depth"
                for relation in controller.last_unified_compositing.order_relations
            )
        )
        self.assertTrue(
            any(
                item.role == "solid_face"
                for item in controller.last_transparent_compositing.fragments
            )
        )
        self.assertTrue(
            any(
                item.role == "section_inside"
                for item in controller.last_transparent_compositing.fragments
            )
        )
        last_good_fragments = moved_fragments
        last_good_slot_points = tuple(
            np.asarray(slot.get_all_points(), dtype=float).copy()
            for slot in controller._transparent_layer.slots
        )
        last_good_unified_z = controller.active_unified_z_indices()

        fixture.shift.set_value(float("nan"))
        with self.assertRaises(Exception):
            controller.update()
        self.assertEqual(
            controller.active_transparent_fragment_ids(),
            last_good_fragments,
        )
        for actual, expected in zip(
            (slot.get_all_points() for slot in controller._transparent_layer.slots),
            last_good_slot_points,
        ):
            np.testing.assert_allclose(actual, expected)
        self.assertEqual(
            controller.active_unified_z_indices(),
            last_good_unified_z,
        )
        self.assertTrue(
            all(float(face.get_fill_opacity()) == 0.0 for face in fixture.faces.values())
        )

        controller.restore()
        self.assertEqual(controller.active_unified_draw_order(), ())
        self.assertEqual(controller.active_unified_z_indices(), {})
        for face_id, expected in source_fill.items():
            np.testing.assert_allclose(
                fixture.faces[face_id].fill_rgbas,
                expected,
            )

    def test_unified_fragment_capacity_failure_keeps_last_good_frame(self) -> None:
        fixture = _CubeExtractionFixture(
            Scene(),
            accurate_transparency=True,
            unified_fragment_slots_per_style=1,
        )
        controller = fixture.controller.attach()
        identities = controller.slot_identities()
        last_good = controller.slot_snapshot()
        last_good_order = controller.active_unified_draw_order()

        fixture.shift.set_value(-0.5)
        with self.assertRaisesRegex(Exception, "capacity"):
            controller.update()

        self.assertEqual(controller.slot_identities(), identities)
        self.assertEqual(controller.slot_snapshot(), last_good)
        self.assertEqual(controller.active_unified_draw_order(), last_good_order)
        controller.restore()

    def test_projected_display_updates_derived_geometry_before_validation(self) -> None:
        fixture = _CubeExtractionFixture(
            Scene(),
            accurate_transparency=True,
            projected_display=True,
        )
        controller = fixture.controller.attach()
        fixture.shift.set_value(-0.75)
        controller.update()

        boundary = fixture.entity.model.extraction.boundary_strokes[0]
        line = fixture.entity.stroke_mobjects[
            fixture.entity.model.extracted_stroke_id(boundary.source_stroke_id)
        ]
        positions = fixture.entity.current_positions()
        expected_start = fixture.display(positions[boundary.vertex_ids[0]])
        expected_end = fixture.display(positions[boundary.vertex_ids[1]])
        self.assertTrue(
            (
                np.allclose(line.get_start(), expected_start)
                and np.allclose(line.get_end(), expected_end)
            )
            or (
                np.allclose(line.get_start(), expected_end)
                and np.allclose(line.get_end(), expected_start)
            )
        )
        controller.restore()

    def test_transform_provider_is_sampled_once_per_visibility_frame(self) -> None:
        fixture = _CubeExtractionFixture(Scene())
        calls: list[int] = []

        def stateful_transform_provider() -> RigidTransform3D:
            frame_number = len(calls) + 1
            calls.append(frame_number)
            return RigidTransform3D.translation_by(
                (0.0, 0.0, -0.25 * frame_number)
            )

        fixture.entity.transform_provider = stateful_transform_provider
        controller = fixture.controller.attach()
        self.assertEqual(calls, [1])
        self.assertEqual(
            controller.last_extraction_frame.transform.translation,
            (0.0, 0.0, -0.25),
        )

        controller.update()
        self.assertEqual(calls, [1, 2])
        self.assertEqual(
            controller.last_extraction_frame.transform.translation,
            (0.0, 0.0, -0.5),
        )
        controller.restore()

    def test_base_face_rotation_uses_each_translated_entity_center(self) -> None:
        fixture = _CubeExtractionFixture(
            Scene(),
            accurate_transparency=True,
            projected_display=True,
            synchronized_base_face="right",
        )
        global_transform_calls: list[float] = []

        def global_transform_provider() -> RigidTransform3D:
            progress = fixture.base_progress.get_value()
            global_transform_calls.append(progress)
            separation = np.asarray(
                (0.0, 0.0, fixture.shift.get_value()),
                dtype=float,
            )
            solid_shift = RigidTransform3D.translation_by(-0.5 * separation)
            return solid_shift.compose(fixture.base_rotation.transform(progress))

        fixture.controller.global_transform_provider = global_transform_provider
        controller = fixture.controller.attach()
        self.assertEqual(global_transform_calls, [0.0])
        fixture.shift.set_value(-0.75)
        fixture.base_progress.set_value(1.0)
        controller.update()
        self.assertEqual(global_transform_calls, [0.0, 1.0])

        base_transform = fixture.base_rotation.final_transform()
        separation = np.asarray((0.0, 0.0, -0.75), dtype=float)
        solid_shift = RigidTransform3D.translation_by(-0.5 * separation)
        global_transform = solid_shift.compose(base_transform)
        solid_center = np.mean(
            [fixture.positions[item] for item in sorted(fixture.positions)],
            axis=0,
        )
        np.testing.assert_allclose(
            fixture.base_rotation.anchor,
            solid_center,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            global_transform.apply(solid_center),
            solid_center - 0.5 * separation,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            controller.last_global_transform.rotation,
            base_transform.rotation,
            atol=1.0e-12,
        )
        base_face = fixture.model.face_map["right"]
        base_points = np.asarray(
            [
                global_transform.apply(fixture.positions[item])
                for item in base_face.vertex_ids
            ]
        )
        base_normal = np.cross(
            base_points[1] - base_points[0],
            base_points[2] - base_points[0],
        )
        base_normal /= np.linalg.norm(base_normal)
        np.testing.assert_allclose(base_normal, (0.0, 0.0, -1.0), atol=1.0e-12)

        solid_stroke = fixture.model.strokes[0]
        solid_line = fixture.lines[solid_stroke.source_edge_id]
        solid_expected = tuple(
            fixture.display(global_transform.apply(fixture.positions[item]))
            for item in solid_stroke.vertex_ids
        )
        self.assertTrue(
            (
                np.allclose(solid_line.get_start(), solid_expected[0])
                and np.allclose(solid_line.get_end(), solid_expected[1])
            )
            or (
                np.allclose(solid_line.get_start(), solid_expected[1])
                and np.allclose(solid_line.get_end(), solid_expected[0])
            )
        )

        boundary = fixture.entity.model.extraction.boundary_strokes[0]
        copy_line = fixture.entity.stroke_mobjects[
            fixture.entity.model.extracted_stroke_id(boundary.source_stroke_id)
        ]
        local_transform = RigidTransform3D.translation_by((0.0, 0.0, -0.75))
        copy_transform = local_transform.compose(global_transform)
        np.testing.assert_allclose(
            copy_transform.apply(solid_center),
            solid_center + 0.5 * separation,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            0.5
            * (
                global_transform.apply(solid_center)
                + copy_transform.apply(solid_center)
            ),
            solid_center,
            atol=1.0e-12,
        )
        copy_expected = tuple(
            fixture.display(copy_transform.apply(fixture.positions[item]))
            for item in boundary.vertex_ids
        )
        self.assertTrue(
            (
                np.allclose(copy_line.get_start(), copy_expected[0])
                and np.allclose(copy_line.get_end(), copy_expected[1])
            )
            or (
                np.allclose(copy_line.get_start(), copy_expected[1])
                and np.allclose(copy_line.get_end(), copy_expected[0])
            )
        )
        self.assertTrue(controller.active_transparent_fragment_ids())
        controller.restore()

    def test_real_cairo_render_splits_intersecting_faces_and_cleans_up(self) -> None:
        class RenderedScene(Scene):
            def construct(inner_self) -> None:
                fixture = _CubeExtractionFixture(
                    inner_self,
                    accurate_transparency=True,
                    projected_display=True,
                    synchronized_base_face="right",
                )
                original_fill = fixture.faces["front"].fill_rgbas.copy()
                with fixture.controller.session():
                    inner_self.wait(0.2)
                    inner_self.play(
                        fixture.shift.animate.set_value(-0.5),
                        run_time=0.25,
                    )
                    inner_self.play(
                        fixture.base_progress.animate.set_value(1.0),
                        run_time=0.25,
                    )
                    inner_self.wait(0.2)
                    inner_self.fragment_count = len(
                        fixture.controller.active_transparent_fragment_ids()
                    )
                    inner_self.relation_count = len(
                        fixture.controller.last_transparent_compositing.order_relations
                    )
                    inner_self.unified_relation_count = len(
                        fixture.controller.last_unified_compositing.order_relations
                    )
                    inner_self.unified_line_face_count = sum(
                        relation.far_item_id.startswith("stroke:")
                        != relation.near_item_id.startswith("stroke:")
                        for relation in fixture.controller.last_unified_compositing.order_relations
                    )
                inner_self.overlay_removed = (
                    fixture.controller.overlay_root not in inner_self.mobjects
                )
                inner_self.fill_restored = np.allclose(
                    fixture.faces["front"].fill_rgbas,
                    original_fill,
                )

        with TemporaryDirectory() as media_dir, tempconfig(
            {
                "renderer": "cairo",
                "media_dir": media_dir,
                "pixel_width": 160,
                "pixel_height": 90,
                "frame_rate": 6,
                "disable_caching": True,
                "write_to_movie": True,
                "save_last_frame": False,
            }
        ):
            scene = RenderedScene()
            scene.render()
            self.assertTrue(
                Path(scene.renderer.file_writer.movie_file_path).is_file()
            )
            self.assertGreater(scene.fragment_count, 16)
            self.assertGreater(scene.relation_count, 0)
            self.assertGreater(scene.unified_relation_count, scene.relation_count)
            self.assertGreater(scene.unified_line_face_count, 0)
            self.assertTrue(scene.overlay_removed)
            self.assertTrue(scene.fill_restored)


if __name__ == "__main__":
    unittest.main()
