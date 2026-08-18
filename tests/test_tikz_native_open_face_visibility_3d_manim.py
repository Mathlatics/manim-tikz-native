from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from manim import Arc, Line, Polygon, Scene, ValueTracker, tempconfig

from polyhedron_visibility import OcclusionStyle
from tikz_native.compiler import compile_document
from tikz_native.geometry_rig_3d import analyze_geometry_rig_3d
from tikz_native.open_face_visibility_3d_manim import (
    TikzNativeOpenFaceVisibility3DManimError,
    bind_picture_open_face_visibility_3d,
)
from tikz_native.provider import instantiate_picture


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"
PLAIN_OPEN_FACES = r"""
\begin{tikzpicture}[3d view={40.4}{23.8}]
  \coordinate (A) at (0,-1.8,0);
  \coordinate (B) at (0,1.8,0);
  \coordinate (Alpha0) at (3.4,-1.8,0);
  \coordinate (Alpha1) at (3.4,1.8,0);
  \coordinate (Beta0) at (1.8017254984,-1.8,2.8833635269);
  \coordinate (Beta1) at (1.8017254984,1.8,2.8833635269);
  \coordinate (S) at (-1,0,1.5);
  \coordinate (E) at (4,0,1.5);
  \fill[fill opacity=.3] (A)--(B)--(Alpha1)--(Alpha0)--cycle;
  \fill[fill opacity=.3] (A)--(B)--(Beta1)--(Beta0)--cycle;
  \DeclareSpaceHinge{fold-angle}{A/B}{A/B/Alpha1/Alpha0}{A/B/Beta1/Beta0}
  \draw[purple] (S)--(E);
\end{tikzpicture}
"""


def _family_style_snapshot(figure, object_ids):
    values = []
    for object_id in sorted(object_ids):
        for member in figure.objects[object_id].get_family():
            values.append(
                (
                    object_id,
                    id(member),
                    tuple(np.asarray(getattr(member, "stroke_rgbas", ())).reshape(-1)),
                    tuple(
                        np.asarray(getattr(member, "background_stroke_rgbas", ())).reshape(-1)
                    ),
                    getattr(member, "stroke_opacity", None),
                    getattr(member, "background_stroke_opacity", None),
                    float(member.z_index),
                )
            )
    return tuple(values)


def _replace_direct_child(figure, object_id, replacement):
    original = figure.objects[object_id]
    index = next(
        index for index, item in enumerate(figure.group.submobjects) if item is original
    )
    replacement.set_z_index(original.z_index, family=True)
    figure.group.submobjects[index] = replacement
    figure.objects[object_id] = replacement


class TikzNativeOpenFaceVisibility3DManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {"renderer": "cairo", "pixel_width": 320, "pixel_height": 180}
        )
        self.config.__enter__()
        self.picture = compile_document(SOURCE).pictures[0]

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def _binding(self, *, coordinate_provider=None, max_length=12.0):
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        binding = bind_picture_open_face_visibility_3d(
            scene,
            self.picture,
            figure,
            coordinate_provider=coordinate_provider,
            style=OcclusionStyle(max_projected_length=max_length),
        )
        return scene, figure, binding

    def test_real_dihedral_uses_complete_proxies_three_probe_spans_and_exact_restore(self) -> None:
        scene, figure, binding = self._binding()
        managed = binding.analysis.suppressed_object_ids
        before_style = _family_style_snapshot(figure, managed)
        before_slots = binding.controller.slot_identities()
        face_fill_opacities = {
            face_id: source.get_fill_opacity()
            for face_id, source in binding.controller.face_fill_sources.items()
        }
        face_proxy_ids = binding.controller._face_fill_layer.identities()

        binding.attach()
        self.assertEqual(before_slots, binding.controller.slot_identities())
        self.assertTrue(all(isinstance(item, Line) for item in binding.controller.proxies.values()))
        self.assertEqual(len(binding.controller.proxies), 9)
        self.assertEqual(
            binding.controller._face_fill_layer.identities(),
            face_proxy_ids,
        )
        self.assertEqual(
            {
                face_id: float(source.get_fill_opacity())
                for face_id, source in binding.controller.face_fill_sources.items()
            },
            {face_id: 0.0 for face_id in face_fill_opacities},
        )
        z_slots = sorted(float(source.z_index) for source in binding.controller.face_fill_sources.values())
        self.assertEqual(
            [
                float(binding.controller._face_fill_layer.proxies[face_id].z_index)
                for face_id in binding.last_frame.advisory_face_draw_order
            ],
            z_slots,
        )
        for object_id in managed:
            drawable = [
                item
                for item in figure.objects[object_id].get_family()
                if np.asarray(getattr(item, "points", ())).size > 0
            ]
            self.assertTrue(drawable)
            self.assertTrue(all(float(item.get_stroke_opacity()) == 0.0 for item in drawable))

        se = next(
            stroke
            for stroke in binding.analysis.model.strokes
            if set(stroke.vertex_ids) == {"S", "E"}
        )
        spans = binding.last_frame.edge_map[se.source_edge_id].spans
        self.assertEqual([item.kind for item in spans], ["visible", "hidden", "visible"])
        self.assertEqual(len(spans[1].occluder_face_ids), 1)

        roots = {
            edge_id: float(slots.root.z_index)
            for edge_id, slots in binding.controller._slots.items()
        }
        self.assertEqual(len(set(roots.values())), len(roots))
        for stroke in binding.analysis.stroke_bindings:
            self.assertGreater(roots[stroke.source_edge_id], float(stroke.z_index))
            self.assertLess(roots[stroke.source_edge_id], float(stroke.z_index) + 1.0)

        binding.restore()
        self.assertEqual(_family_style_snapshot(figure, managed), before_style)
        self.assertEqual(
            {
                face_id: source.get_fill_opacity()
                for face_id, source in binding.controller.face_fill_sources.items()
            },
            face_fill_opacities,
        )
        self.assertNotIn(binding.controller.overlay_root, scene.mobjects)

    def test_normal_and_exception_restore_are_pixel_exact_in_real_cairo_scene(self) -> None:
        for exceptional in (False, True):
            with self.subTest(exceptional=exceptional):
                scene, _figure, binding = self._binding()
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                before = scene.camera.pixel_array.copy()
                if exceptional:
                    with self.assertRaisesRegex(RuntimeError, "sentinel"):
                        with binding.session():
                            raise RuntimeError("sentinel")
                else:
                    binding.attach().restore()
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                np.testing.assert_array_equal(scene.camera.pixel_array, before)

    def test_plain_complete_line_binds_global_face_trace_without_legacy_relation(self) -> None:
        picture = compile_document(source_text=PLAIN_OPEN_FACES).pictures[0]
        self.assertFalse(picture.occlusion_relations)
        figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        binding = bind_picture_open_face_visibility_3d(
            scene,
            picture,
            figure,
            style=OcclusionStyle(max_projected_length=12.0),
        ).attach()
        stroke = binding.analysis.model.strokes[0]
        self.assertEqual(
            [span.kind for span in binding.last_frame.edge_map[stroke.source_edge_id].spans],
            ["visible", "hidden", "visible"],
        )
        self.assertEqual(figure.objects["line.S.E"].get_stroke_opacity(), 0.0)
        self.assertGreater(
            binding.controller._slots[stroke.source_edge_id].root.get_all_points().size,
            0,
        )
        binding.restore()
        self.assertGreater(figure.objects["line.S.E"].get_stroke_opacity(), 0.0)

    def test_exact_zero_and_pi_hinges_bind_without_a_visibility_seam(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        cases = (
            (3.4, 0.0, "coplanar_same_normal"),
            (-3.4, 0.0, "coplanar_opposite_normal"),
            (3.4, 1.0e-11, "coplanar_same_normal"),
            (-3.4, 1.0e-11, "coplanar_opposite_normal"),
        )
        for x_value, z_value, expected_state in cases:
            with self.subTest(x=x_value, z=z_value):
                current = source.replace(
                    "(1.8017254984,-1.8,2.8833635269)",
                    f"({x_value},-1.8,{z_value})",
                ).replace(
                    "(1.8017254984,1.8,2.8833635269)",
                    f"({x_value},1.8,{z_value})",
                )
                picture = compile_document(source_text=current).pictures[0]
                figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
                scene = Scene()
                scene.add(figure.group)
                binding = bind_picture_open_face_visibility_3d(
                    scene,
                    picture,
                    figure,
                    style=OcclusionStyle(max_projected_length=12.0),
                ).attach()
                self.assertEqual(len(binding.last_frame.seam_states), 1)
                self.assertEqual(
                    binding.last_frame.seam_states[0].state,
                    expected_state,
                )
                self.assertEqual(
                    set(binding.last_frame.advisory_face_draw_order),
                    set(binding.analysis.model.face_map),
                )
                binding.restore()

    def test_scale_gate_runs_before_proxy_or_slot_allocation_and_scene_mutation(self) -> None:
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        managed = tuple(
            object_id
            for relation in self.picture.occlusion_relations
            for object_id in relation.object_ids
        ) + ("line.M.N",)
        before_style = _family_style_snapshot(figure, managed)
        before_roots = tuple(id(item) for item in scene.mobjects)
        with patch(
            "tikz_native.open_face_visibility_3d_manim._complete_proxy_line",
            side_effect=AssertionError("proxy allocated before scale gate"),
        ) as proxy_factory, patch(
            "tikz_native.open_face_visibility_3d_manim._StrokeSlots",
            side_effect=AssertionError("slot allocated before scale gate"),
        ) as slot_factory:
            with self.assertRaisesRegex(
                TikzNativeOpenFaceVisibility3DManimError,
                "overlay_line_slots=.*exceeds fixed v1 limit 65536",
            ):
                bind_picture_open_face_visibility_3d(
                    scene,
                    self.picture,
                    figure,
                    style=OcclusionStyle(max_projected_length=1.0e6),
                )
        proxy_factory.assert_not_called()
        slot_factory.assert_not_called()
        self.assertEqual(tuple(id(item) for item in scene.mobjects), before_roots)
        self.assertEqual(_family_style_snapshot(figure, managed), before_style)

    def test_real_geometry_rig_order_rehides_relation_groups_and_restores_pixels(self) -> None:
        rig = analyze_geometry_rig_3d(self.picture)
        payload = rig["nativeManimSourceV2"]
        namespace: dict[str, object] = {}
        exec(compile(payload["sourceText"], "<native-manim-source-3d-v2>", "exec"), namespace)

        for exceptional in (False, True):
            with self.subTest(exceptional=exceptional):
                figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
                figure.group.scale(0.72).rotate(0.13).shift((1.1, -0.45, 0.0))
                scene = Scene()
                scene.add(figure.group)
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                pristine = scene.camera.pixel_array.copy()
                pristine_children = tuple(id(item) for item in figure.group.submobjects)
                trackers = {
                    driver_id: ValueTracker(initial)
                    for driver_id, initial in namespace["DRIVER_INITIAL_VALUES"].items()
                }
                camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
                state = namespace["install_geometry_3d_updaters"](
                    figure.group, figure.objects, trackers, camera
                )
                binding = bind_picture_open_face_visibility_3d(
                    scene,
                    self.picture,
                    figure,
                    geometry_rig_state=state,
                    style=OcclusionStyle(max_projected_length=12.0),
                )
                identities = binding.controller.slot_identities()

                def exercise() -> None:
                    se = next(
                        stroke
                        for stroke in binding.analysis.model.strokes
                        if set(stroke.vertex_ids) == {"S", "E"}
                    )
                    self.assertEqual(
                        [span.kind for span in binding.last_frame.edge_map[se.source_edge_id].spans],
                        ["visible", "hidden", "visible"],
                    )
                    scene.camera.reset()
                    scene.camera.capture_mobjects(scene.mobjects)
                    entry_overlay = scene.camera.pixel_array.copy()
                    trackers["hinge_fold:fold-angle"].set_value(
                        namespace["DRIVER_SPECS"]["hinge_fold:fold-angle"]["range"][1]
                    )
                    for root in tuple(scene.mobjects):
                        root.update(0.0)
                    self.assertEqual(binding.controller.slot_identities(), identities)
                    relation_opacities = [
                        float(line.get_stroke_opacity())
                        for group in state["temporary_groups"]
                        for line in group.get_family()
                        if np.asarray(getattr(line, "points", ())).size > 0
                    ]
                    self.assertTrue(relation_opacities)
                    self.assertEqual(max(relation_opacities), 0.0)
                    # Also cover a fragment whose style is explicitly restored
                    # by some later updater: the binding must suppress it again.
                    figure.objects["occluded_visible.S.E.0"].set_stroke(opacity=1.0)
                    binding.update()
                    self.assertEqual(
                        figure.objects["occluded_visible.S.E.0"].get_stroke_opacity(),
                        0.0,
                    )
                    scene.camera.reset()
                    scene.camera.capture_mobjects(scene.mobjects)
                    self.assertFalse(
                        np.array_equal(scene.camera.pixel_array, entry_overlay),
                        "folded Cairo keyframe must differ from entry",
                    )

                try:
                    if exceptional:
                        with self.assertRaisesRegex(RuntimeError, "sentinel"):
                            with binding.session():
                                exercise()
                                raise RuntimeError("sentinel")
                    else:
                        binding.attach()
                        exercise()
                        binding.restore()
                finally:
                    if binding.controller.attached:
                        binding.restore()
                    namespace["restore_geometry_3d_objects"](state)
                self.assertNotIn(binding.controller.overlay_root, scene.mobjects)
                self.assertEqual(
                    tuple(id(item) for item in figure.group.submobjects),
                    pristine_children,
                )
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                np.testing.assert_array_equal(scene.camera.pixel_array, pristine)

    def test_geometry_rig_camera_modes_and_orbit_midpoint_share_one_live_projection(self) -> None:
        rig = analyze_geometry_rig_3d(self.picture)
        namespace: dict[str, object] = {}
        exec(
            compile(
                rig["nativeManimSourceV2"]["sourceText"],
                "<native-manim-source-3d-v2>",
                "exec",
            ),
            namespace,
        )
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        figure.group.scale(0.72).rotate(0.13).shift((1.1, -0.45, 0.0))
        scene = Scene()
        scene.add(figure.group)
        trackers = {
            driver_id: ValueTracker(initial)
            for driver_id, initial in namespace["DRIVER_INITIAL_VALUES"].items()
        }
        camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
        state = namespace["install_geometry_3d_updaters"](
            figure.group, figure.objects, trackers, camera
        )
        binding = bind_picture_open_face_visibility_3d(
            scene,
            self.picture,
            figure,
            geometry_rig_state=state,
            style=OcclusionStyle(max_projected_length=12.0),
        ).attach()
        mn = next(
            stroke
            for stroke in binding.analysis.model.strokes
            if set(stroke.vertex_ids) == {"M", "N"}
        )

        try:
            requests = [
                ("side", "linear", 1.0),
                ("front", "linear", 1.0),
                ("top", "linear", 1.0),
                ("isometric", "linear", 1.0),
                ("side", "orbit", 0.5),
            ]
            matrices = []
            for mode, transition, progress in requests:
                namespace["prepare_local_camera"](state, mode, transition, 0.2)
                camera.set_value(progress)
                # The Geometry Rig root runs before the independent overlay
                # root in a real Scene frame.
                figure.group.update(0.0)
                binding.update()
                live_matrix = binding.controller.projection.current_matrix(scene)
                np.testing.assert_allclose(
                    np.asarray(binding.last_frame.projection_matrix),
                    np.asarray(live_matrix),
                    atol=1.0e-12,
                    rtol=0.0,
                )
                native = figure.objects["line.M.N"]
                proxy = binding.controller.proxies[mn.source_edge_id]
                np.testing.assert_allclose(proxy.get_start(), native.get_start(), atol=1e-9)
                np.testing.assert_allclose(proxy.get_end(), native.get_end(), atol=1e-9)
                matrices.append(np.asarray(live_matrix))
            self.assertTrue(
                any(not np.allclose(matrices[0], matrix) for matrix in matrices[1:])
            )
        finally:
            binding.restore()
            namespace["restore_geometry_3d_objects"](state)

    def test_runtime_fragment_offset_fails_atomically_and_rehides_sources(self) -> None:
        scene, figure, binding = self._binding()
        binding.attach()
        last_frame = binding.last_frame
        last_slots = binding.controller.slot_snapshot()
        object_id = "occluded_visible.S.E.0"
        source = figure.objects[object_id]
        source.shift((0.0, 0.05, 0.0)).set_stroke(opacity=1.0)
        with self.assertRaisesRegex(
            TikzNativeOpenFaceVisibility3DManimError,
            "offset from its current logical Line",
        ):
            binding.update()
        self.assertIs(binding.last_frame, last_frame)
        self.assertEqual(binding.controller.slot_snapshot(), last_slots)
        self.assertEqual(source.get_stroke_opacity(), 0.0)
        source.shift((0.0, -0.05, 0.0))
        binding.restore()
        self.assertNotIn(binding.controller.overlay_root, scene.mobjects)

    def test_tikz_binding_is_scale_equivalent_from_1e_minus_9_to_1e9(self) -> None:
        for scale in (1.0e-9, 1.0e-6, 1.0, 1.0e6, 1.0e9):
            with self.subTest(scale=scale):
                figure = instantiate_picture(self.picture, scene_unit_per_cm=scale)
                scene = Scene()
                scene.add(figure.group)
                binding = bind_picture_open_face_visibility_3d(
                    scene,
                    self.picture,
                    figure,
                    style=OcclusionStyle(max_projected_length=12.0 * scale),
                ).attach()
                se = next(
                    stroke
                    for stroke in binding.analysis.model.strokes
                    if set(stroke.vertex_ids) == {"S", "E"}
                )
                self.assertEqual(
                    [
                        span.kind
                        for span in binding.last_frame.edge_map[se.source_edge_id].spans
                    ],
                    ["visible", "hidden", "visible"],
                )
                binding.restore()
                self.assertNotIn(binding.controller.overlay_root, scene.mobjects)

    def test_real_cairo_page9_fold_and_camera_render_restores_transactionally(self) -> None:
        picture = self.picture
        rig = analyze_geometry_rig_3d(picture)
        namespace: dict[str, object] = {}
        exec(
            compile(
                rig["nativeManimSourceV2"]["sourceText"],
                "<native-manim-source-3d-v2>",
                "exec",
            ),
            namespace,
        )

        class PageNineOpenFaceScene(Scene):
            def construct(inner_self) -> None:
                figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
                inner_self.add(figure.group)
                entry_children = tuple(id(item) for item in figure.group.submobjects)
                trackers = {
                    driver_id: ValueTracker(initial)
                    for driver_id, initial in namespace["DRIVER_INITIAL_VALUES"].items()
                }
                camera = ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"])
                state = namespace["install_geometry_3d_updaters"](
                    figure.group,
                    figure.objects,
                    trackers,
                    camera,
                )
                binding = bind_picture_open_face_visibility_3d(
                    inner_self,
                    picture,
                    figure,
                    geometry_rig_state=state,
                    style=OcclusionStyle(max_projected_length=12.0),
                )
                with binding.session():
                    driver_id = "hinge_fold:fold-angle"
                    upper = namespace["DRIVER_SPECS"][driver_id]["range"][1]
                    inner_self.play(
                        trackers[driver_id].animate.set_value(upper),
                        run_time=0.2,
                    )
                    namespace["prepare_local_camera"](
                        state,
                        "side",
                        "linear",
                        0.2,
                    )
                    inner_self.play(camera.animate.set_value(1.0), run_time=0.2)
                    inner_self.wait(0.1)
                    inner_self.trace_schema = binding.last_frame.schema
                    inner_self.slot_ids = binding.controller.slot_identities()
                namespace["restore_geometry_3d_objects"](state)
                inner_self.overlay_removed = (
                    binding.controller.overlay_root not in inner_self.mobjects
                )
                inner_self.children_restored = (
                    tuple(id(item) for item in figure.group.submobjects)
                    == entry_children
                )
                inner_self.wait(0.05)

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
            scene = PageNineOpenFaceScene()
            scene.render()
            self.assertTrue(Path(scene.renderer.file_writer.movie_file_path).is_file())
            self.assertEqual(
                scene.trace_schema,
                "manim-open-convex-face-visibility-trace/v1",
            )
            self.assertTrue(scene.slot_ids)
            self.assertTrue(scene.overlay_removed)
            self.assertTrue(scene.children_restored)

    def test_tikz_binding_does_not_accept_a_custom_tolerance_policy(self) -> None:
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        with self.assertRaisesRegex(TypeError, "tolerance_policy"):
            bind_picture_open_face_visibility_3d(
                scene,
                self.picture,
                figure,
                style=OcclusionStyle(max_projected_length=12.0),
                tolerance_policy=object(),  # type: ignore[call-arg]
            )

    def test_dynamic_update_keeps_slot_identity_and_failed_capacity_is_atomic(self) -> None:
        picture = compile_document(source_text=PLAIN_OPEN_FACES).pictures[0]
        authored = {
            name: np.asarray(point, dtype=float) for name, point in picture.coordinates.items()
        }
        scale = 1.0

        def coordinates():
            return {name: scale * point for name, point in authored.items()}

        figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
        scene = Scene()
        scene.add(figure.group)
        binding = bind_picture_open_face_visibility_3d(
            scene,
            picture,
            figure,
            coordinate_provider=coordinates,
            style=OcclusionStyle(max_projected_length=12.0),
        )

        def sync_source() -> None:
            current = coordinates()
            projection = binding.analysis.entry_projection
            mapper = binding.controller._mapper
            for face_binding in binding.analysis.face_bindings:
                source = figure.objects[face_binding.object_ids[0]]
                points = [
                    mapper.map_point(
                        current[vertex_id],
                        projection,
                    )
                    for vertex_id in face_binding.authored_cycles[0]
                ]
                replacement = Polygon(*points).match_style(source)
                replacement.set_z_index(source.z_index, family=True)
                source.become(replacement)
            figure.objects["line.S.E"].put_start_and_end_on(
                mapper.map_point(current["S"], projection),
                mapper.map_point(current["E"], projection),
            )

        binding.attach()
        identities = binding.controller.slot_identities()
        before = binding.controller.slot_snapshot()
        scale = 1.2
        sync_source()
        binding.update()
        self.assertEqual(binding.controller.slot_identities(), identities)
        self.assertNotEqual(binding.controller.slot_snapshot(), before)

        last_good_frame = binding.last_frame
        last_good_slots = binding.controller.slot_snapshot()
        scale = 100.0
        sync_source()
        restored_fragment = "line.S.E"
        figure.objects[restored_fragment].set_stroke(opacity=1.0)
        with self.assertRaisesRegex(Exception, "exceeds fixed maximum"):
            binding.update()
        self.assertIs(binding.last_frame, last_good_frame)
        self.assertEqual(binding.controller.slot_snapshot(), last_good_slots)
        fragment_opacities = [
            float(item.get_stroke_opacity())
            for item in figure.objects[restored_fragment].get_family()
            if np.asarray(getattr(item, "points", ())).size > 0
        ]
        self.assertTrue(fragment_opacities)
        self.assertEqual(max(fragment_opacities), 0.0)
        binding.restore()
        self.assertNotIn(binding.controller.overlay_root, scene.mobjects)

    def test_unmanaged_same_z_drawable_fails_before_fragments_are_hidden(self) -> None:
        scene, figure, binding = self._binding()
        managed = binding.analysis.suppressed_object_ids
        before = _family_style_snapshot(figure, managed)
        conflict = Line((-1, -1, 0), (1, -1, 0)).set_z_index(14)
        scene.add(conflict)

        with self.assertRaisesRegex(
            TikzNativeOpenFaceVisibility3DManimError,
            "unmanaged drawable shares managed z_index 14",
        ):
            binding.attach()
        self.assertEqual(_family_style_snapshot(figure, managed), before)
        self.assertNotIn(binding.controller.overlay_root, scene.mobjects)

    def test_unproven_dashed_curve_and_arrow_plain_sources_all_fail_closed(self) -> None:
        variants = ("dashed", "curve", "arrow")
        for variant in variants:
            with self.subTest(variant=variant):
                if variant == "dashed":
                    source = SOURCE.read_text(encoding="utf-8").replace(
                        r"\draw[helper] (M)--(N);",
                        r"\draw[helper,dashed] (M)--(N);",
                    )
                    picture = compile_document(source_text=source).pictures[0]
                else:
                    picture = copy.deepcopy(self.picture)
                figure = instantiate_picture(picture, scene_unit_per_cm=1.0)
                if variant == "curve":
                    _replace_direct_child(figure, "line.M.N", Arc(radius=1.0))
                elif variant == "arrow":
                    arrow = Line(
                        figure.objects["line.M.N"].get_start(),
                        figure.objects["line.M.N"].get_end(),
                    ).add_tip()
                    _replace_direct_child(figure, "line.M.N", arrow)
                scene = Scene()
                scene.add(figure.group)
                with self.assertRaises(TikzNativeOpenFaceVisibility3DManimError):
                    bind_picture_open_face_visibility_3d(
                        scene,
                        picture,
                        figure,
                        style=OcclusionStyle(max_projected_length=12.0),
                    )


if __name__ == "__main__":
    unittest.main()
