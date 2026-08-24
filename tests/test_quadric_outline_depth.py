from __future__ import annotations

import unittest

import numpy as np
from manim import Scene, tempconfig

from polyhedron_visibility.geometry import GeometryContext, GeometryQuantity
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.topology import ParameterInterval, assert_exact_partition
from polyhedron_visibility.quadrics.compositing import compute_quadric_compositing
from polyhedron_visibility.quadrics.contract import SectionPlane, SphereSpec
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.projection import build_opaque_projection_proxy
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    compute_quadric_section_compositing,
)
from polyhedron_visibility.quadrics.visibility import compute_quadric_visibility


IDENTITY_VIEW = ParallelView.from_matrix(np.eye(3))


def _sphere_frame(surface: SphereSpec):
    proxy = build_opaque_projection_proxy(
        surface,
        IDENTITY_VIEW,
        max_chord_error=0.01,
    )
    visibility = compute_quadric_visibility((), (surface,), IDENTITY_VIEW)
    return compute_quadric_compositing(visibility, (proxy,))


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=1,
        max_curves=1,
        max_fragments_per_curve=8,
        max_segments_per_fragment=128,
        max_surface_segments=256,
        max_dashes_per_fragment=32,
        max_projected_length=8.0,
        max_total_mobjects=1000,
    )


class QuadricOutlineGeometryTests(unittest.TestCase):
    def test_tilted_patch_outline_is_exactly_partitioned_by_depth(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "cut-patch",
            plane,
            (sphere,),
            margin_ratio=0.1,
        ).patch
        frame = compute_quadric_section_compositing(
            _sphere_frame(sphere),
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        roles = {item.role for item in frame.plane_outline_fragments}
        self.assertTrue(
            {
                PlaneDepthRole.BEHIND_SURFACE,
                PlaneDepthRole.OUTSIDE_PROJECTION,
                PlaneDepthRole.IN_FRONT_OF_SURFACE,
            }.issubset(roles)
        )
        for edge_index in range(4):
            edge = tuple(
                sorted(
                    (
                        item
                        for item in frame.plane_outline_fragments
                        if item.edge_index == edge_index
                    ),
                    key=lambda item: item.interval.start,
                )
            )
            assert_exact_partition(
                ParameterInterval(0.0, 1.0),
                (item.interval for item in edge),
            )

        context = GeometryContext().resolve(
            (*sphere.characteristic_points, *patch.corners(plane))
        )
        boundary = context.epsilon(GeometryQuantity.BOUNDARY)
        direction = np.asarray(IDENTITY_VIEW.view_direction, dtype=float)
        for item in frame.plane_outline_fragments:
            midpoint = 0.5 * (
                np.asarray(item.world_start, dtype=float)
                + np.asarray(item.world_end, dtype=float)
            )
            parameters: list[float] = []
            for hit in sphere.ray_hits(
                midpoint,
                direction,
                context=context,
                include_caps=True,
                forward_only=False,
            ):
                if (
                    not parameters
                    or abs(float(hit.parameter) - parameters[-1]) > boundary
                ):
                    parameters.append(float(hit.parameter))
            if not parameters:
                expected = PlaneDepthRole.OUTSIDE_PROJECTION
            elif min(parameters) > boundary:
                expected = PlaneDepthRole.BEHIND_SURFACE
            elif max(parameters) < -boundary:
                expected = PlaneDepthRole.IN_FRONT_OF_SURFACE
            else:
                expected = PlaneDepthRole.BETWEEN_SURFACE_SHEETS
            self.assertIs(item.role, expected)

        order = frame.draw_order
        self.assertLess(
            order.index(frame.paint_items.plane_outline_behind),
            order.index(frame.paint_items.surface_back),
        )
        self.assertLess(
            order.index(frame.paint_items.plane_outline_between),
            order.index(frame.paint_items.surface_front),
        )
        self.assertGreater(
            order.index(frame.paint_items.plane_outline),
            order.index(frame.paint_items.surface_front),
        )


class QuadricOutlineManimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "frame_rate": 8,
                "pixel_width": 320,
                "pixel_height": 180,
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_outline_roles_bind_to_distinct_fixed_slots(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(sphere,),
            curves=(),
            projection=IDENTITY_VIEW,
            limits=_limits(),
            section_plane=plane,
            section_max_screen_error=0.12,
        ).attach()
        frame = controller.last_section_frame
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(len(controller._section_slots), 10)
        slots = dict(zip(frame.paint_items.ordered, controller._section_slots))
        outline_ids = tuple(frame.paint_items.outline_by_role.values())
        self.assertEqual(len(set(outline_ids)), 4)
        for role, item_id in frame.paint_items.outline_by_role.items():
            self.assertEqual(
                slots[item_id].has_points(),
                bool(frame.outline_fragments_by_role[role]),
            )
        z_indices = controller.active_painter_z_indices
        self.assertLess(
            z_indices[frame.paint_items.plane_outline_behind],
            z_indices[frame.paint_items.surface_back],
        )
        self.assertGreater(
            z_indices[frame.paint_items.plane_outline],
            z_indices[frame.paint_items.surface_front],
        )
        controller.restore()


if __name__ == "__main__":
    unittest.main()
