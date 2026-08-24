from __future__ import annotations

from math import pi
import unittest

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.geometry import GeometryContext, GeometryQuantity
from polyhedron_visibility.quadrics.compositing import (
    QuadricPaintPolicy,
    compute_quadric_compositing,
)
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    SectionPlane,
    SphereSpec,
    CylinderSpec,
)
from polyhedron_visibility.quadrics.curves import CircleArcCurve
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.projection import build_opaque_projection_proxy
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    QuadricSectionCompositingError,
    QuadricSectionCompositingLimits,
    canonical_quadric_section_compositing_json,
    compute_quadric_section_compositing,
    quadric_plane_fragment_contours,
    _surface_ray_solver,
)
from polyhedron_visibility.quadrics.visibility import compute_quadric_visibility


IDENTITY_VIEW = ParallelView.from_matrix(np.eye(3))


def _base_frame(
    surface: SphereSpec | ConeSpec,
    curves: tuple[CircleArcCurve, ...] = (),
    *,
    paint_policy: QuadricPaintPolicy | str = QuadricPaintPolicy.DIAGRAMMATIC,
):
    proxy = build_opaque_projection_proxy(
        surface,
        IDENTITY_VIEW,
        max_chord_error=0.01,
    )
    visibility = compute_quadric_visibility(
        curves,
        (surface,),
        IDENTITY_VIEW,
    )
    return compute_quadric_compositing(
        visibility,
        (proxy,),
        paint_policy=paint_policy,
    )


def _triangle_area(vertices: tuple[tuple[float, float, float], ...]) -> float:
    values = np.asarray(vertices, dtype=float)
    return 0.5 * float(
        np.linalg.norm(np.cross(values[1] - values[0], values[2] - values[0]))
    )


def _screen_signed_area(vertices) -> float:
    values = np.asarray(vertices, dtype=float)
    return 0.5 * sum(
        values[index, 0] * values[(index + 1) % len(values), 1]
        - values[index, 1] * values[(index + 1) % len(values), 0]
        for index in range(len(values))
    )


class QuadricSectionCompositingTests(unittest.TestCase):
    def test_display_ray_classifier_matches_authoritative_finite_hits(self) -> None:
        surfaces = (
            SphereSpec("sphere", (0.2, -0.1, 0.3), 1.1),
            CylinderSpec(
                "cylinder",
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
                0.9,
                (-0.4, 2.1),
                radial_axis=(1.0, 0.0, 0.0),
            ),
            ConeSpec(
                "cone",
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
                pi / 6.0,
                (0.0, 3.0),
                radial_axis=(1.0, 0.0, 0.0),
            ),
        )
        direction = np.asarray((0.31, -0.27, 0.91), dtype=float)
        direction /= np.linalg.norm(direction)
        generator = np.random.default_rng(20260824)
        for surface in surfaces:
            with self.subTest(surface=surface.surface_id):
                context = GeometryContext().resolve(surface.characteristic_points)
                boundary = context.epsilon(GeometryQuantity.BOUNDARY)
                solver = _surface_ray_solver(
                    surface,
                    direction,
                    boundary_epsilon=boundary,
                    angular_epsilon=context.epsilon(GeometryQuantity.ANGULAR),
                )
                for point in generator.uniform(-2.5, 2.5, size=(40, 3)):
                    expected: list[float] = []
                    for hit in surface.ray_hits(
                        point,
                        direction,
                        context=context,
                        include_caps=True,
                        forward_only=False,
                    ):
                        if (
                            not expected
                            or abs(float(hit.parameter) - expected[-1]) > boundary
                        ):
                            expected.append(float(hit.parameter))
                    np.testing.assert_allclose(
                        solver(point),
                        tuple(expected),
                        rtol=0.0,
                        atol=max(boundary, 1.0e-11),
                    )

    def test_tilted_sphere_plane_splits_all_three_depth_regions(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "cut-patch", plane, (sphere,), margin_ratio=0.1
        ).patch
        frame = compute_quadric_section_compositing(
            _base_frame(sphere),
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        roles = {item.role for item in frame.plane_fragments}
        self.assertTrue(
            {
                PlaneDepthRole.BEHIND_SURFACE,
                PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
                PlaneDepthRole.IN_FRONT_OF_SURFACE,
            }.issubset(roles)
        )
        restored_area = sum(
            _triangle_area(item.world_vertices) for item in frame.plane_fragments
        )
        self.assertAlmostEqual(
            restored_area,
            4.0 * patch.half_width * patch.half_height,
            places=9,
        )
        self.assertEqual(set(frame.draw_order), {
            *frame.paint_items.ordered,
        })

    def test_plane_fragments_merge_to_equivalent_renderer_contours(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "cut-patch", plane, (sphere,), margin_ratio=0.1
        ).patch
        frame = compute_quadric_section_compositing(
            _base_frame(sphere),
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        first = quadric_plane_fragment_contours(frame)
        second = quadric_plane_fragment_contours(frame)
        self.assertEqual(first, second)
        for role in PlaneDepthRole:
            source_area = sum(
                _screen_signed_area(item.screen_vertices)
                for item in frame.fragments_by_role[role]
            )
            contour_area = sum(
                _screen_signed_area(item) for item in first[role]
            )
            self.assertAlmostEqual(source_area, contour_area, places=10)
        self.assertLess(
            sum(len(items) for items in first.values()),
            len(frame.plane_fragments) // 10,
        )

    def test_small_near_tangent_section_is_not_missed_by_coarse_samples(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        plane = SectionPlane(
            "near-tangent",
            (0.0, 0.0, 0.95),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "near-tangent-patch", plane, (sphere,), margin_ratio=0.1
        ).patch
        frame = compute_quadric_section_compositing(
            _base_frame(sphere),
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        self.assertTrue(frame.fragments_by_role[PlaneDepthRole.BETWEEN_SURFACE_SHEETS])
        between_area = sum(
            abs(_screen_signed_area(item.screen_vertices))
            for item in frame.fragments_by_role[PlaneDepthRole.BETWEEN_SURFACE_SHEETS]
        )
        self.assertGreater(between_area, 0.1)

    def test_surface_sheets_plane_groups_and_curves_share_one_draw_order(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        curve = CircleArcCurve(
            "equator",
            (0.0, 0.0, 0.0),
            1.0,
            (0.0, 1.0, 0.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch("patch", plane, (sphere,)).patch
        base = _base_frame(sphere, (curve,))
        frame = compute_quadric_section_compositing(
            base,
            sphere,
            plane,
            patch,
            IDENTITY_VIEW,
        )

        active_curves = {
            item.item_id for item in base.curve_fragments if item.painted
        }
        self.assertEqual(
            set(frame.draw_order),
            {*frame.paint_items.ordered, *active_curves},
        )
        outline_rank = frame.draw_order.index(frame.paint_items.plane_outline)
        self.assertTrue(
            all(frame.draw_order.index(item_id) > outline_rank for item_id in active_curves)
        )

    def test_cone_transition_style_frame_is_deterministic(self) -> None:
        cone = ConeSpec(
            "cone",
            (0.0, 0.0, -1.5),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 0.2),
            (0.5, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "patch", plane, (cone,), margin_ratio=0.08
        ).patch
        first = compute_quadric_section_compositing(
            _base_frame(cone),
            cone,
            plane,
            patch,
            IDENTITY_VIEW,
        )
        second = compute_quadric_section_compositing(
            _base_frame(cone),
            cone,
            plane,
            patch,
            IDENTITY_VIEW,
        )
        self.assertEqual(
            canonical_quadric_section_compositing_json(first),
            canonical_quadric_section_compositing_json(second),
        )

    def test_edge_on_plane_and_capacity_overflow_fail_closed(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        edge_on = SectionPlane(
            "edge-on",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            u_axis=(0.0, 0.0, 1.0),
        )
        edge_patch = fit_plane_display_patch("edge", edge_on, (sphere,)).patch
        with self.assertRaisesRegex(
            QuadricSectionCompositingError,
            "projects edge-on",
        ):
            compute_quadric_section_compositing(
                _base_frame(sphere),
                sphere,
                edge_on,
                edge_patch,
                IDENTITY_VIEW,
            )

        tilted = SectionPlane(
            "tilted",
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        tilted_patch = fit_plane_display_patch("tilted", tilted, (sphere,)).patch
        with self.assertRaisesRegex(
            QuadricSectionCompositingError,
            "more than 8 plane fragments",
        ):
            compute_quadric_section_compositing(
                _base_frame(sphere),
                sphere,
                tilted,
                tilted_patch,
                IDENTITY_VIEW,
                limits=QuadricSectionCompositingLimits(
                    minimum_subdivision_depth=0,
                    maximum_subdivision_depth=10,
                    max_plane_fragments=8,
                    max_ray_classifications=4096,
                ),
            )


if __name__ == "__main__":
    unittest.main()
