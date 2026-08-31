from __future__ import annotations

from dataclasses import replace
from math import pi, sin, sqrt
import unittest

import numpy as np

import polyhedron_visibility.quadrics.dandelin_compositing as compositing_module
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.dandelin import compute_dandelin_construction
from polyhedron_visibility.quadrics.dandelin_compositing import (
    DandelinContactSheet,
    DandelinPlanePosition,
    DandelinSurfaceCompositingError,
    canonical_dandelin_surface_layer_json,
    compute_dandelin_surface_layer_frame,
)
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch


HALF_ANGLE = pi / 6.0
VIEW_MATRIX = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.8, 0.6),
        (0.0, -0.6, 0.8),
    ),
    dtype=float,
)


def _normal(axis_dot: float) -> tuple[float, float, float]:
    return (sqrt(max(0.0, 1.0 - axis_dot * axis_dot)), 0.0, axis_dot)


def _construction(
    construction_id: str,
    axis_dot: float,
    *,
    model: ConeModel = ConeModel.OPEN_SINGLE,
    axial_range: tuple[float, float] = (0.0, 20.0),
):
    cone = ConeSpec(
        "cone",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        HALF_ANGLE,
        axial_range,
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )
    plane = SectionPlane(
        "section-plane",
        (0.0, 0.0, 2.0),
        _normal(axis_dot),
        u_axis=(0.0, 1.0, 0.0),
    )
    return compute_dandelin_construction(construction_id, cone, plane)


def _frame(construction: object, matrix: np.ndarray = VIEW_MATRIX):
    patch = fit_plane_display_patch(
        "dandelin-layer-patch",
        construction.plane,
        construction.cone.render_components,
        margin_ratio=0.14,
    ).patch
    return compute_dandelin_surface_layer_frame(
        construction,
        ParallelView.from_matrix(matrix),
        patch,
    )


class DandelinSurfaceCompositingTests(unittest.TestCase):
    def test_ellipse_certifies_cone_spheres_plane_and_equal_depth_seams(
        self,
    ) -> None:
        construction = _construction("layered-ellipse", 0.8)
        frame = _frame(construction)

        self.assertTrue(frame.surface_layering_authoritative)
        self.assertFalse(frame.physical_surface_visibility_authoritative)
        self.assertEqual(len(frame.cone_layers), 1)
        self.assertEqual(len(frame.sphere_layers), 2)
        self.assertEqual(len(frame.equal_depth_contacts), 2)
        self.assertGreater(frame.plane_fragment_count, 0)
        self.assertEqual(
            {item.role for item in frame.plane_layers},
            set(type(frame.plane_layers[0].role)),
        )
        rank = {item_id: index for index, item_id in enumerate(frame.draw_order)}
        cone = frame.cone_layers[0]
        for sphere in frame.sphere_layers:
            self.assertLess(rank[cone.back_item_id], rank[sphere.item_id])
            self.assertLess(rank[sphere.item_id], rank[cone.front_item_id])
        self.assertEqual(
            {item.plane_position for item in frame.sphere_layers},
            {
                DandelinPlanePosition.IN_FRONT_OF_SPHERE,
                DandelinPlanePosition.BEHIND_SPHERE,
            },
        )
        plane_items = (*frame.plane_layers, *frame.plane_outline_layers)
        far_sphere = next(
            item
            for item in frame.sphere_layers
            if item.plane_position is DandelinPlanePosition.IN_FRONT_OF_SPHERE
        )
        near_sphere = next(
            item
            for item in frame.sphere_layers
            if item.plane_position is DandelinPlanePosition.BEHIND_SPHERE
        )
        for item in plane_items:
            if item.role.value == "behind_surface":
                self.assertLess(rank[item.item_id], rank[far_sphere.item_id])
            else:
                self.assertLess(rank[far_sphere.item_id], rank[item.item_id])
            if item.role.value == "in_front_of_surface":
                self.assertLess(rank[near_sphere.item_id], rank[item.item_id])
            else:
                self.assertLess(rank[item.item_id], rank[near_sphere.item_id])
        for contact in frame.equal_depth_contacts:
            self.assertTrue(contact.feature_stroke_owns_equal_depth)
            self.assertEqual(
                {item.sheet for item in contact.spans},
                {DandelinContactSheet.BACK, DandelinContactSheet.FRONT},
            )
            self.assertEqual(len(contact.transition_parameters), 2)

    def test_same_screen_reverse_depth_reverses_sphere_chain(self) -> None:
        construction = _construction("reverse-depth", 0.8)
        forward = _frame(construction)
        reversed_matrix = VIEW_MATRIX.copy()
        reversed_matrix[2] *= -1.0
        reverse = _frame(construction, reversed_matrix)

        self.assertEqual(
            forward.projection_matrix[:2],
            reverse.projection_matrix[:2],
        )
        self.assertEqual(
            tuple(item.proxy for item in forward.sphere_layers),
            tuple(item.proxy for item in reverse.sphere_layers),
        )
        self.assertEqual(
            tuple(item.plane_position for item in forward.sphere_layers),
            tuple(
                DandelinPlanePosition.BEHIND_SPHERE
                if item.plane_position
                is DandelinPlanePosition.IN_FRONT_OF_SPHERE
                else DandelinPlanePosition.IN_FRONT_OF_SPHERE
                for item in reverse.sphere_layers
            ),
        )
        self.assertEqual(
            forward.sphere_pair_evidence[0].farther_sphere_id,
            reverse.sphere_pair_evidence[0].nearer_sphere_id,
        )
        self.assertEqual(
            forward.sphere_pair_evidence[0].nearer_sphere_id,
            reverse.sphere_pair_evidence[0].farther_sphere_id,
        )
        self.assertNotEqual(forward.draw_order, reverse.draw_order)
        for frame in (forward, reverse):
            rank = {
                item_id: index for index, item_id in enumerate(frame.draw_order)
            }
            far_sphere = next(
                item
                for item in frame.sphere_layers
                if item.plane_position
                is DandelinPlanePosition.IN_FRONT_OF_SPHERE
            )
            near_sphere = next(
                item
                for item in frame.sphere_layers
                if item.plane_position is DandelinPlanePosition.BEHIND_SPHERE
            )
            for item in (*frame.plane_layers, *frame.plane_outline_layers):
                if item.role.value == "behind_surface":
                    self.assertLess(rank[item.item_id], rank[far_sphere.item_id])
                else:
                    self.assertLess(rank[far_sphere.item_id], rank[item.item_id])
                if item.role.value == "in_front_of_surface":
                    self.assertLess(rank[near_sphere.item_id], rank[item.item_id])
                else:
                    self.assertLess(rank[item.item_id], rank[near_sphere.item_id])

    def test_parabola_has_one_sphere_and_circle_records_common_focus_tie(
        self,
    ) -> None:
        parabola = _frame(_construction("layered-parabola", sin(HALF_ANGLE)))
        circle = _frame(_construction("layered-circle", 1.0))

        self.assertEqual(len(parabola.sphere_layers), 1)
        self.assertFalse(parabola.sphere_pair_evidence)
        self.assertEqual(len(circle.sphere_layers), 2)
        self.assertEqual(
            circle.sphere_pair_evidence[0].relation,
            "external_tangent",
        )
        self.assertIsNotNone(circle.sphere_pair_evidence[0].tangent_point)

    def test_frame_is_canonical_and_capacity_or_edge_on_fail_closed(self) -> None:
        construction = _construction("canonical-layer-frame", 0.8)
        first = _frame(construction)
        second = _frame(construction)
        self.assertEqual(
            canonical_dandelin_surface_layer_json(first),
            canonical_dandelin_surface_layer_json(second),
        )
        patch = first.patch
        with self.assertRaises(DandelinSurfaceCompositingError):
            compute_dandelin_surface_layer_frame(
                construction,
                ParallelView.from_matrix(VIEW_MATRIX),
                patch,
                max_screen_error=1.0e-6,
                max_segments=8,
            )

        edge_on_matrix = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        with self.assertRaisesRegex(
            DandelinSurfaceCompositingError,
            "AREA|edge-on|rank-one",
        ):
            compute_dandelin_surface_layer_frame(
                construction,
                ParallelView.from_matrix(edge_on_matrix),
                patch,
            )

    def test_public_replacement_cannot_forge_layer_evidence(self) -> None:
        frame = _frame(_construction("unforgeable-layer-frame", 0.8))
        fragment_index = next(
            index
            for index, fragment in enumerate(frame.section_frame.plane_fragments)
            if fragment.role.value != "outside_projection"
        )
        fragment = frame.section_frame.plane_fragments[fragment_index]
        forged_fragments = list(frame.section_frame.plane_fragments)
        forged_fragments[fragment_index] = replace(
            fragment,
            role=type(fragment.role).OUTSIDE_PROJECTION,
        )
        forged_section = replace(
            frame.section_frame,
            plane_fragments=tuple(forged_fragments),
        )
        forged_planes, forged_outlines = compositing_module._plane_layers(
            forged_section,
            frame.projection_matrix,
        )
        with self.assertRaisesRegex(
            DandelinSurfaceCompositingError,
            "construction-derived geometry",
        ):
            replace(
                frame,
                section_frame=forged_section,
                plane_layers=forged_planes,
                plane_outline_layers=forged_outlines,
            )

        plane = frame.plane_layers[0]
        forged_plane = replace(
            plane,
            contours=(tuple(reversed(plane.contours[0])), *plane.contours[1:]),
        )
        with self.assertRaisesRegex(
            DandelinSurfaceCompositingError,
            "disagree with certified section geometry",
        ):
            replace(
                frame,
                plane_layers=(forged_plane, *frame.plane_layers[1:]),
            )

        contact = frame.equal_depth_contacts[0]
        with self.assertRaisesRegex(
            DandelinSurfaceCompositingError,
            "do not cover the certified sphere layers",
        ):
            replace(
                frame,
                equal_depth_contacts=(
                    replace(contact, sphere_id="invented-sphere"),
                    *frame.equal_depth_contacts[1:],
                ),
            )

        sphere = frame.sphere_layers[0]
        forged_sphere = replace(
            sphere,
            plane_position=(
                DandelinPlanePosition.BEHIND_SPHERE
                if sphere.plane_position
                is DandelinPlanePosition.IN_FRONT_OF_SPHERE
                else DandelinPlanePosition.IN_FRONT_OF_SPHERE
            ),
            plane_ray_parameter=-sphere.plane_ray_parameter,
        )
        with self.assertRaisesRegex(
            DandelinSurfaceCompositingError,
            "disagree with construction evidence",
        ):
            replace(
                frame,
                sphere_layers=(forged_sphere, *frame.sphere_layers[1:]),
            )

        forged_contact = replace(
            contact,
            spans=tuple(
                replace(
                    span,
                    sheet=(
                        DandelinContactSheet.FRONT
                        if span.sheet is DandelinContactSheet.BACK
                        else DandelinContactSheet.BACK
                    ),
                )
                for span in contact.spans
            ),
        )
        with self.assertRaisesRegex(
            DandelinSurfaceCompositingError,
            "equal-depth seams disagree with construction evidence",
        ):
            replace(
                frame,
                equal_depth_contacts=(
                    forged_contact,
                    *frame.equal_depth_contacts[1:],
                ),
            )

    def test_open_double_unsupported_partition_fails_before_rendering(self) -> None:
        construction = _construction(
            "layered-hyperbola",
            0.2,
            model=ConeModel.OPEN_DOUBLE,
            axial_range=(-20.0, 20.0),
        )
        with self.assertRaises(DandelinSurfaceCompositingError):
            _frame(construction)


if __name__ == "__main__":
    unittest.main()
