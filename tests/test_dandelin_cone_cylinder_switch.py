from __future__ import annotations

from copy import deepcopy
from math import cos, pi, sin
import unittest
from unittest.mock import patch

import numpy as np
from manim import Scene, ValueTracker, config, smooth, tempconfig
from polyhedron_visibility.visibility import VisibilityKind
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundaryOcclusionScope,
    BoundarySourceKind,
)
from polyhedron_visibility.quadrics.contract import CylinderModel

from examples.dandelin_cone_cylinder_switch.dandelin_cone_cylinder_switch import (
    AXIAL_RANGE,
    BACKGROUND_COLOR,
    CONE_SLOPE,
    DandelinConeCylinderSwitch,
    PLANE_NORMAL,
    PLANE_OFFSET,
    PlaneDepthRole,
    PROJECTION_SCALE,
    SURFACE_RADIUS,
    TEACHING_UI_Z_INDEX,
    _header,
    _progress_legend,
    _surface_spec,
    build_switch_diagram,
    compute_switch_occlusion_frame,
    compute_switch_frame,
    project_point,
    refresh_switch_diagram,
    section_point,
)


class DandelinConeCylinderSwitchGeometryTests(unittest.TestCase):
    def test_endpoints_are_one_true_cone_and_one_true_cylinder(self) -> None:
        cone = compute_switch_frame(0.0)
        cylinder = compute_switch_frame(1.0)

        self.assertEqual(DandelinConeCylinderSwitch.__name__, "DandelinConeCylinderSwitch")
        self.assertEqual(cone.surface_kind, "cone")
        self.assertAlmostEqual(cone.slope, CONE_SLOPE)
        self.assertAlmostEqual(cone.apex_z, AXIAL_RANGE[0])
        self.assertAlmostEqual(cone.radius_at(AXIAL_RANGE[0]), 0.0)

        self.assertEqual(cylinder.surface_kind, "cylinder")
        self.assertEqual(cylinder.slope, 0.0)
        self.assertIsNone(cylinder.apex_z)
        for z in AXIAL_RANGE:
            self.assertAlmostEqual(cylinder.radius_at(z), SURFACE_RADIUS)

        cylinder_surface = _surface_spec(cylinder)
        self.assertIs(cylinder_surface.model, CylinderModel.OPEN)
        self.assertEqual(cylinder_surface.end_caps, ())
        self.assertEqual(
            tuple(item.role for item in cylinder_surface.trim_rims),
            ("trim_min", "trim_max"),
        )
        for sphere in cylinder.spheres:
            self.assertAlmostEqual(sphere.radius, SURFACE_RADIUS)
            self.assertAlmostEqual(sphere.surface_contact_radius, SURFACE_RADIUS)
            self.assertAlmostEqual(sphere.surface_contact_z, sphere.center[2])

    def test_two_spheres_remain_tangent_to_surface_and_plane(self) -> None:
        normal = np.asarray(PLANE_NORMAL, dtype=float)
        for progress in np.linspace(0.0, 1.0, 9):
            with self.subTest(progress=float(progress)):
                frame = compute_switch_frame(float(progress))
                self.assertEqual(tuple(item.plane_side for item in frame.spheres), (-1, 1))
                for sphere in frame.spheres:
                    center = np.asarray(sphere.center, dtype=float)
                    plane_contact = np.asarray(sphere.plane_contact, dtype=float)
                    signed_distance = float(np.dot(normal, center) - PLANE_OFFSET)
                    self.assertAlmostEqual(
                        signed_distance,
                        sphere.plane_side * sphere.radius,
                        places=11,
                    )
                    self.assertAlmostEqual(
                        float(np.dot(normal, plane_contact)),
                        PLANE_OFFSET,
                        places=11,
                    )
                    self.assertAlmostEqual(
                        float(np.linalg.norm(plane_contact - center)),
                        sphere.radius,
                        places=11,
                    )

                    contact_point = np.asarray(
                        (
                            sphere.surface_contact_radius,
                            0.0,
                            sphere.surface_contact_z,
                        ),
                        dtype=float,
                    )
                    self.assertAlmostEqual(
                        sphere.surface_contact_radius,
                        frame.radius_at(sphere.surface_contact_z),
                        places=11,
                    )
                    self.assertAlmostEqual(
                        float(np.linalg.norm(contact_point - center)),
                        sphere.radius,
                        places=11,
                    )
                    self.assertGreaterEqual(sphere.center[2] - sphere.radius, AXIAL_RANGE[0])
                    self.assertLessEqual(sphere.center[2] + sphere.radius, AXIAL_RANGE[1])

    def test_section_parameterization_lies_on_both_constraints(self) -> None:
        normal = np.asarray(PLANE_NORMAL, dtype=float)
        for progress in (0.0, 0.2, 0.5, 0.8, 1.0):
            frame = compute_switch_frame(progress)
            for theta in np.linspace(0.0, 2.0 * pi, 25)[:-1]:
                point = np.asarray(section_point(frame, float(theta)), dtype=float)
                radial = float(np.hypot(point[0], point[1]))
                self.assertAlmostEqual(radial, frame.radius_at(float(point[2])), places=11)
                self.assertAlmostEqual(float(np.dot(normal, point)), PLANE_OFFSET, places=11)
                expected = np.asarray(
                    (
                        radial * cos(float(theta)),
                        radial * sin(float(theta)),
                    )
                )
                np.testing.assert_allclose(point[:2], expected, atol=1.0e-11)

    def test_invalid_progress_fails_closed(self) -> None:
        for value in (-0.01, 1.01, float("nan"), True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "progress must lie"):
                    compute_switch_frame(value)


class DandelinConeCylinderSwitchOcclusionTests(unittest.TestCase):
    def test_teaching_ui_layer_stays_above_every_geometry_fragment(self) -> None:
        diagram = build_switch_diagram(0.5)
        header = _header()
        legend = _progress_legend(ValueTracker(0.5))

        self.assertTrue(diagram.submobjects)
        self.assertLess(
            max(item.z_index for item in diagram.get_family()),
            TEACHING_UI_Z_INDEX,
        )
        self.assertTrue(
            all(item.z_index == TEACHING_UI_Z_INDEX for item in header.get_family())
        )
        self.assertTrue(
            all(item.z_index == TEACHING_UI_Z_INDEX for item in legend.get_family())
        )

    def test_every_geometric_stroke_has_one_registered_boundary_source(self) -> None:
        common = {
            *(f"boundary:plane:switch-plane:edge:{index}" for index in range(4)),
            "boundary:switch-sphere:+1:silhouette",
            "boundary:switch-sphere:-1:silhouette",
            "boundary:switch-surface:silhouette:generator:0",
            "boundary:switch-surface:silhouette:generator:1",
            "boundary:switch-surface:trim_max:rim",
            "switch-axis",
            "switch-contact:+1",
            "switch-contact:-1",
            "switch-section:component:ellipse",
        }
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(progress=progress):
                boundary = compute_switch_occlusion_frame(
                    progress
                ).scene_frame.boundary_frame
                sources = boundary.sources
                source_ids = {item.source_id for item in sources}
                expected = set(common)
                if progress > 0.0:
                    expected.add("boundary:switch-surface:trim_min:rim")
                self.assertEqual(source_ids, expected)
                self.assertEqual(len(sources), len(source_ids))
                self.assertEqual(
                    {item.source_id for item in boundary.fragments},
                    source_ids,
                )
                plane_edges = tuple(
                    item
                    for item in sources
                    if item.source_kind is BoundarySourceKind.PLANE_PATCH_EDGE
                )
                self.assertEqual(len(plane_edges), 4)
                self.assertTrue(
                    all(
                        item.occlusion_scope
                        is BoundaryOcclusionScope.ALL_SURFACES
                        for item in plane_edges
                    )
                )

    def test_five_keyframes_have_certified_plane_roles_and_sphere_order(self) -> None:
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(progress=progress):
                frame = compute_switch_occlusion_frame(progress)
                self.assertTrue(frame.surface_layering_authoritative)
                self.assertFalse(frame.physical_surface_visibility_authoritative)
                self.assertEqual(
                    {role for role, _paths in frame.plane_contours},
                    set(PlaneDepthRole),
                )
                self.assertTrue(
                    all(frame.contours_for(role) for role in PlaneDepthRole)
                )

                far = next(
                    item for item in frame.sphere_layers if item.plane_is_in_front
                )
                near = next(
                    item
                    for item in frame.sphere_layers
                    if not item.plane_is_in_front
                )
                self.assertEqual(far.plane_side, -1)
                self.assertEqual(near.plane_side, 1)
                rank = {
                    item_id: index
                    for index, item_id in enumerate(frame.draw_order)
                }
                plane_ids = {
                    role: (fill_id, outline_id)
                    for role, fill_id, outline_id in frame.plane_item_ids
                }
                for item_id in plane_ids[PlaneDepthRole.BEHIND_SURFACE]:
                    self.assertLess(rank[item_id], rank[far.item_id])
                for role in (
                    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                    PlaneDepthRole.IN_FRONT_OF_SURFACE,
                ):
                    for item_id in plane_ids[role]:
                        self.assertLess(rank[far.item_id], rank[item_id])
                for role in (
                    PlaneDepthRole.BEHIND_SURFACE,
                    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                ):
                    for item_id in plane_ids[role]:
                        self.assertLess(rank[item_id], rank[near.item_id])
                for item_id in plane_ids[PlaneDepthRole.IN_FRONT_OF_SURFACE]:
                    self.assertLess(rank[near.item_id], rank[item_id])
                section_fragments = tuple(
                    item
                    for item in frame.curve_fragments
                    if item.source_id.startswith("switch-section:")
                    and item.painted
                )
                hidden_section = tuple(
                    item
                    for item in section_fragments
                    if item.surface_visibility_kind is VisibilityKind.HIDDEN
                )
                visible_section = tuple(
                    item
                    for item in section_fragments
                    if item.surface_visibility_kind is VisibilityKind.VISIBLE
                )
                self.assertTrue(hidden_section)
                self.assertTrue(visible_section)
                for fragment in hidden_section:
                    self.assertLess(rank[far.item_id], rank[fragment.item_id])
                    self.assertLess(rank[fragment.item_id], rank[near.item_id])
                for fragment in visible_section:
                    self.assertLess(rank[near.item_id], rank[fragment.item_id])
                sphere_hidden = tuple(
                    item
                    for item in hidden_section
                    if near.item_id in item.occluder_surface_ids
                )
                self.assertTrue(sphere_hidden)

    def test_near_cylinder_plane_partition_converges_without_role_loss(self) -> None:
        near = compute_switch_occlusion_frame(0.9999)
        cylinder = compute_switch_occlusion_frame(1.0)

        def signed_area(
            contours: tuple[tuple[tuple[float, float], ...], ...],
        ) -> float:
            result = 0.0
            for contour in contours:
                points = np.asarray(contour, dtype=float)
                result += 0.5 * float(
                    np.sum(
                        points[:, 0] * np.roll(points[:, 1], -1)
                        - points[:, 1] * np.roll(points[:, 0], -1)
                    )
                )
            return result

        for role in PlaneDepthRole:
            with self.subTest(role=role.value):
                self.assertAlmostEqual(
                    signed_area(near.contours_for(role)),
                    signed_area(cylinder.contours_for(role)),
                    delta=1.0e-3,
                )

    def test_near_cylinder_nested_tangencies_create_no_crossing_slivers(
        self,
    ) -> None:
        sphere_ids = {
            "boundary:switch-sphere:-1:silhouette",
            "boundary:switch-sphere:+1:silhouette",
        }
        mother_ids = {
            "boundary:switch-surface:silhouette:generator:0",
            "boundary:switch-surface:silhouette:generator:1",
        }
        contact_by_sphere = {
            "boundary:switch-sphere:-1:silhouette": "switch-contact:-1",
            "boundary:switch-sphere:+1:silhouette": "switch-contact:+1",
        }
        certified_pairs = {
            tuple(sorted((sphere_id, other_id)))
            for sphere_id, contact_id in contact_by_sphere.items()
            for other_id in (*mother_ids, contact_id)
        } | {
            tuple(sorted((contact_id, mother_id)))
            for contact_id in contact_by_sphere.values()
            for mother_id in mother_ids
        }
        for progress in (0.9999, 0.99999999, 1.0):
            with self.subTest(progress=progress):
                scene_frame = compute_switch_occlusion_frame(
                    progress
                ).scene_frame
                boundary = scene_frame.boundary_frame
                crossing_pairs = {
                    (item.first_curve_id, item.second_curve_id)
                    for item in boundary.crossings
                }
                self.assertTrue(certified_pairs.isdisjoint(crossing_pairs))
                parameter_epsilon = (
                    scene_frame.global_frame.geometry_context.epsilon(
                        "parameter"
                    )
                )
                self.assertTrue(
                    all(
                        fragment.interval.length
                        > 100.0 * parameter_epsilon
                        for fragment in boundary.fragments
                        if fragment.source_id in sphere_ids
                    )
                )

    def test_smooth_animation_tail_has_no_uncertified_intermediate_frame(
        self,
    ) -> None:
        for frame_index in range(37, 46):
            progress = float(smooth(frame_index / 45.0))
            with self.subTest(frame_index=frame_index, progress=progress):
                frame = compute_switch_occlusion_frame(progress)
                self.assertEqual(
                    len(frame.draw_order),
                    len(set(frame.draw_order)),
                )

    def test_contact_curves_are_not_embedded_in_sphere_body_items(self) -> None:
        diagram = build_switch_diagram(1.0)
        frame = diagram.switch_occlusion_frame
        by_id = {
            item.switch_paint_item_id: item
            for item in diagram.submobjects
            if hasattr(item, "switch_paint_item_id")
        }
        self.assertEqual(set(by_id), set(frame.draw_order))
        contact_ids = {
            item.item_id
            for item in frame.curve_fragments
            if item.source_id.startswith("switch-contact:") and item.painted
        }
        self.assertTrue(contact_ids)
        self.assertTrue(contact_ids.issubset(by_id))
        for sphere in frame.sphere_layers:
            body = by_id[sphere.item_id]
            self.assertEqual(body.switch_metadata["switchPaintKind"], "sphere_body")
            self.assertFalse(body.switch_metadata["contactCurvesEmbedded"])
            self.assertFalse(body.switch_metadata["silhouetteEmbedded"])
            self.assertTrue(contact_ids.isdisjoint({body.switch_paint_item_id}))

        structural = tuple(
            item
            for item in by_id.values()
            if "boundaryFragmentsEmbedded" in item.switch_metadata
        )
        self.assertEqual(len(structural), 10)
        self.assertTrue(
            all(
                item.switch_metadata["boundaryFragmentsEmbedded"] is False
                for item in structural
            )
        )

    def test_manim_can_deepcopy_immutable_scene_evidence_for_fade(self) -> None:
        diagram = build_switch_diagram(0.5)
        copied = deepcopy(diagram)
        self.assertIs(
            copied.switch_occlusion_frame,
            diagram.switch_occlusion_frame,
        )

    def test_live_refresh_matches_direct_fragment_topology_and_z_order(self) -> None:
        live = build_switch_diagram(0.0)
        root_identity = id(live)
        for progress in (0.25, 0.5, 0.75, 0.9999, 1.0, 0.5, 0.0):
            with self.subTest(progress=progress):
                refresh_switch_diagram(live, progress)
                direct = build_switch_diagram(progress)

                def trace(diagram):
                    return tuple(
                        (
                            getattr(item, "switch_paint_item_id", None),
                            item.z_index,
                        )
                        for item in diagram.submobjects
                    )

                self.assertEqual(trace(live), trace(direct))
                self.assertEqual(
                    live.switch_occlusion_frame.draw_order,
                    direct.switch_occlusion_frame.draw_order,
                )
                self.assertEqual(id(live), root_identity)

    def test_live_refresh_rolls_back_before_mutating_on_failed_frame(self) -> None:
        diagram = build_switch_diagram(0.5)
        original_children = tuple(diagram.submobjects)
        original_frame = diagram.switch_occlusion_frame

        with patch(
            "examples.dandelin_cone_cylinder_switch."
            "dandelin_cone_cylinder_switch.build_switch_diagram",
            side_effect=RuntimeError("failed certification"),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed certification"):
                refresh_switch_diagram(diagram, 0.75)

        self.assertEqual(len(diagram.submobjects), len(original_children))
        self.assertTrue(
            all(
                actual is expected
                for actual, expected in zip(
                    diagram.submobjects,
                    original_children,
                )
            )
        )
        self.assertIs(diagram.switch_occlusion_frame, original_frame)

    def test_hidden_dashes_are_anchored_to_their_semantic_source(self) -> None:
        diagram = build_switch_diagram(0.5)
        dashed = tuple(
            item
            for item in diagram.submobjects
            if getattr(item, "switch_metadata", {}).get("strokePattern")
            == "dashed"
        )

        self.assertTrue(dashed)
        self.assertTrue(
            all(item.switch_metadata["dashPhaseAnchored"] for item in dashed)
        )
        self.assertTrue(
            all(item.switch_metadata["dashPeriod"] > 0.0 for item in dashed)
        )

    def test_scene_keeps_dynamic_diagram_as_one_cairo_moving_root(self) -> None:
        scene = DandelinConeCylinderSwitch()
        diagram = build_switch_diagram(0.0)
        foreground = build_switch_diagram(1.0)
        scene.add(diagram)
        scene.add_foreground_mobject(foreground)

        moving, static = scene.get_moving_and_static_mobjects(())

        self.assertEqual(static, [])
        self.assertEqual(len(moving), 2)
        self.assertIs(moving[0], diagram)
        self.assertIs(moving[1], foreground)
        self.assertFalse(any(item is diagram.submobjects[0] for item in moving))


class DandelinConeCylinderSwitchCairoTests(unittest.TestCase):
    def test_keyframes_have_sphere_surface_and_section_pixels(self) -> None:
        frames: list[np.ndarray] = []
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 6,
                "disable_caching": True,
                "write_to_movie": False,
                "save_last_frame": False,
            }
        ):
            for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
                scene = Scene()
                scene.camera.background_color = BACKGROUND_COLOR
                scene.add(build_switch_diagram(progress))
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                pixels = scene.camera.pixel_array[:, :, :3].copy()
                frames.append(pixels)

                background = np.asarray((11, 22, 34), dtype=int)
                non_background = np.linalg.norm(
                    pixels.astype(int) - background,
                    axis=2,
                ) > 4.0
                red = pixels[:, :, 0].astype(int)
                green = pixels[:, :, 1].astype(int)
                blue = pixels[:, :, 2].astype(int)
                sphere_orange = (
                    (red > 135)
                    & (green > 60)
                    & (green < 190)
                    & ((red - blue) > 25)
                )
                section_yellow = (
                    (red > 170)
                    & (green > 130)
                    & (blue < 165)
                    & ((red - blue) > 35)
                )
                self.assertGreater(int(np.count_nonzero(non_background)), 1300)
                self.assertGreater(int(np.count_nonzero(sphere_orange)), 35)
                self.assertGreater(int(np.count_nonzero(section_yellow)), 8)

        self.assertGreater(int(np.count_nonzero(frames[0] != frames[1])), 1500)
        self.assertGreater(int(np.count_nonzero(frames[1] != frames[2])), 1200)
        self.assertGreater(int(np.count_nonzero(frames[2] != frames[3])), 1000)
        self.assertGreater(int(np.count_nonzero(frames[3] != frames[4])), 800)
        self.assertGreater(int(np.count_nonzero(frames[0] != frames[4])), 2500)

    def test_cylinder_plane_outline_switches_order_across_the_two_spheres(self) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 960,
                "pixel_height": 540,
                "frame_rate": 12,
                "disable_caching": True,
                "write_to_movie": False,
                "save_last_frame": False,
            }
        ):
            scene = Scene()
            scene.camera.background_color = BACKGROUND_COLOR
            scene.add(build_switch_diagram(1.0))
            scene.camera.reset()
            scene.camera.capture_mobjects(scene.mobjects)
            pixels = scene.camera.pixel_array[:, :, :3].astype(int)

        far_sphere_boundary = pixels[382:387, 448:453]
        near_sphere_boundary = pixels[233:238, 478:483]
        far_cyan = float(
            np.mean(
                far_sphere_boundary[:, :, 1]
                - far_sphere_boundary[:, :, 0]
            )
        )
        near_cyan = float(
            np.mean(
                near_sphere_boundary[:, :, 1]
                - near_sphere_boundary[:, :, 0]
            )
        )
        self.assertGreater(far_cyan - near_cyan, 12.0)

    def test_cylinder_contact_endpoints_and_section_sphere_cutoff_are_clean(self) -> None:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 960,
                "pixel_height": 540,
                "frame_rate": 12,
                "disable_caching": True,
                "write_to_movie": False,
                "save_last_frame": False,
            }
        ):
            scene = Scene()
            scene.camera.background_color = BACKGROUND_COLOR
            diagram = build_switch_diagram(1.0)
            scene.add(diagram)
            scene.camera.reset()
            scene.camera.capture_mobjects(scene.mobjects)
            pixels = scene.camera.pixel_array[:, :, :3].astype(int)
            frame = diagram.switch_occlusion_frame

            def pixel(point: np.ndarray) -> tuple[int, int]:
                x = int(
                    round(
                        (point[0] + 0.5 * config.frame_width)
                        * config.pixel_width
                        / config.frame_width
                    )
                )
                y = int(
                    round(
                        (0.5 * config.frame_height - point[1])
                        * config.pixel_height
                        / config.frame_height
                    )
                )
                return x, y

            source = next(
                item
                for item in frame.curve_sources
                if item.source_id == "switch-contact:+1"
            )
            contact_fragments = tuple(
                item
                for item in frame.curve_fragments
                if item.source_id == source.source_id
            )
            visible_ends = {
                value
                for item in contact_fragments
                if item.surface_visibility_kind is VisibilityKind.VISIBLE
                for value in (item.interval.start, item.interval.end)
            }
            hidden_ends = {
                value
                for item in contact_fragments
                if item.surface_visibility_kind is VisibilityKind.HIDDEN
                for value in (item.interval.start, item.interval.end)
            }
            transition_parameters = tuple(
                sorted(
                    value
                    for value in visible_ends
                    if any(abs(value - other) < 1.0e-10 for other in hidden_ends)
                )
            )
            self.assertEqual(len(transition_parameters), 2)
            for parameter in transition_parameters:
                x, y = pixel(project_point(source.curve.point(parameter)))
                neighborhood = pixels[y - 3 : y + 4, x - 3 : x + 4]
                orange_score = (
                    neighborhood[:, :, 0] - neighborhood[:, :, 2]
                )
                self.assertGreater(int(np.max(orange_score)), 90)

            geometry = compute_switch_frame(1.0)
            near_layer = next(
                item
                for item in frame.sphere_layers
                if not item.plane_is_in_front
            )
            near_sphere = next(
                item
                for item in geometry.spheres
                if item.plane_side == near_layer.plane_side
            )
            center_x, center_y = pixel(project_point(near_sphere.center))
            radius_pixels = (
                PROJECTION_SCALE
                * near_sphere.radius
                * config.pixel_height
                / config.frame_height
            )
            rows, columns = np.indices(pixels.shape[:2])
            inside = (
                np.hypot(columns - center_x, rows - center_y)
                < radius_pixels - 2.0
            )
            red = pixels[:, :, 0]
            green = pixels[:, :, 1]
            blue = pixels[:, :, 2]
            strong_section_yellow = (
                (red > 200)
                & (green > 160)
                & (green < 230)
                & (blue < 145)
                & ((red - blue) > 70)
            )
            self.assertEqual(
                int(np.count_nonzero(strong_section_yellow & inside)),
                0,
            )


if __name__ == "__main__":
    unittest.main()
