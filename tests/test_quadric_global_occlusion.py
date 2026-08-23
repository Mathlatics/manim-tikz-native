from __future__ import annotations

import json
from math import pi
import unittest

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    SphereSpec,
)
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.global_occlusion import (
    GlobalQuadricOcclusionError,
    canonical_global_quadric_frame_json,
    compute_global_quadric_frame,
    verify_strict_quadric_separation,
)


IDENTITY_VIEW = ParallelView.from_matrix(np.eye(3))
OBLIQUE_VIEW = ParallelView.from_matrix(
    ((1.0, 0.0, 0.35), (0.0, 1.0, 0.2), (0.0, 0.0, 1.0))
)


def _constraint_pairs(frame: object) -> tuple[tuple[str, str], ...]:
    return tuple(
        (item.farther_surface_id, item.nearer_surface_id)
        for item in frame.surface_constraints  # type: ignore[attr-defined]
    )


class GlobalQuadricPairOrderingTests(unittest.TestCase):
    def test_sphere_sphere_overlap_uses_exact_ray_depth(self) -> None:
        far = SphereSpec("far", (0.0, 0.0, 0.0), 1.0)
        near = SphereSpec("near", (0.0, 0.0, 4.0), 1.0)
        result = compute_global_quadric_frame([], (near, far), IDENTITY_VIEW)

        self.assertEqual(_constraint_pairs(result), (("far", "near"),))
        self.assertEqual(len(result.separation_evidence), 1)
        self.assertGreater(
            result.separation_evidence[0].support_gap_lower_bound,
            0.0,
        )
        depth = result.surface_depth_evidence[0]
        self.assertEqual(depth.projection_relation, "overlap")
        self.assertGreaterEqual(len(depth.witnesses), 1)
        self.assertFalse(depth.proxy_visibility_authoritative)
        self.assertTrue(
            all(
                witness.first_depth_interval[1]
                < witness.second_depth_interval[0]
                for witness in depth.witnesses
            )
        )

    def test_sphere_and_finite_cylinder_are_ordered(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 0.8)
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, 3.0),
            (0.0, 1.0, 0.0),
            0.6,
            (-1.25, 1.25),
        )
        result = compute_global_quadric_frame(
            [], (cylinder, sphere), IDENTITY_VIEW
        )
        self.assertEqual(_constraint_pairs(result), (("sphere", "cylinder"),))
        self.assertEqual(result.surface_depth_evidence[0].projection_relation, "overlap")

    def test_rotated_cylinder_uses_finite_caps_and_lateral_surface(self) -> None:
        sphere = SphereSpec("base", (0.0, 0.0, 0.0), 0.7)
        cylinder = CylinderSpec(
            "rotated",
            (0.0, 0.0, 3.0),
            (1.0, 2.0, 0.35),
            0.55,
            (-1.2, 1.2),
            radial_axis=(2.0, -1.0, 0.0),
        )
        result = compute_global_quadric_frame(
            [], (sphere, cylinder), OBLIQUE_VIEW
        )
        self.assertEqual(_constraint_pairs(result), (("base", "rotated"),))
        self.assertTrue(result.surface_depth_evidence[0].witnesses)

    def test_single_nappe_cone_frustum_is_convex_and_ordered(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 0.65)
        frustum = ConeSpec(
            "frustum",
            (0.0, -2.0, 3.0),
            (0.0, 1.0, 0.0),
            0.25,
            (1.0, 3.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        result = compute_global_quadric_frame(
            [], (frustum, sphere), IDENTITY_VIEW
        )
        self.assertEqual(_constraint_pairs(result), (("sphere", "frustum"),))
        self.assertEqual(result.surface_depth_evidence[0].projection_relation, "overlap")

    def test_negative_nappe_frustum_is_also_supported(self) -> None:
        first = ConeSpec(
            "negative",
            (0.0, 2.0, 3.0),
            (0.0, 1.0, 0.0),
            pi / 12.0,
            (-3.0, -1.0),
        )
        second = SphereSpec("sphere", (0.0, 0.0, 0.0), 0.6)
        result = compute_global_quadric_frame([], (first, second), IDENTITY_VIEW)
        self.assertEqual(_constraint_pairs(result), (("sphere", "negative"),))

    def test_apex_to_cap_cone_is_closed_under_an_axial_view(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 0.5)
        cone = ConeSpec(
            "cone",
            (0.0, 0.0, 3.0),
            (0.0, 0.0, 1.0),
            0.3,
            (0.0, 2.0),
        )
        result = compute_global_quadric_frame([], (cone, sphere), IDENTITY_VIEW)
        self.assertEqual(_constraint_pairs(result), (("sphere", "cone"),))

    def test_projection_disjoint_pair_adds_no_painter_constraint(self) -> None:
        left = SphereSpec("left", (-4.0, 0.0, 0.0), 0.75)
        right = SphereSpec("right", (4.0, 0.0, 1.0), 0.75)
        result = compute_global_quadric_frame([], (right, left), IDENTITY_VIEW)
        self.assertEqual(result.surface_constraints, ())
        self.assertEqual(result.surface_depth_evidence[0].projection_relation, "disjoint")
        self.assertGreater(
            result.surface_depth_evidence[0].projected_separation_gap or 0.0,
            0.0,
        )


class GlobalQuadricSceneTests(unittest.TestCase):
    def test_three_overlapping_spheres_form_one_acyclic_global_order(self) -> None:
        surfaces = (
            SphereSpec("middle", (0.0, 0.0, 3.0), 0.8),
            SphereSpec("near", (0.0, 0.0, 6.0), 0.8),
            SphereSpec("far", (0.0, 0.0, 0.0), 0.8),
        )
        result = compute_global_quadric_frame([], surfaces, IDENTITY_VIEW)
        self.assertEqual(
            _constraint_pairs(result),
            (("far", "middle"), ("far", "near"), ("middle", "near")),
        )
        surface_order = tuple(
            item.removeprefix("surface:").removesuffix(":opaque-projection")
            for item in result.frame.draw_order
        )
        self.assertEqual(surface_order, ("far", "middle", "near"))

    def test_oblique_projection_overlap_of_spatially_separated_entities(self) -> None:
        # Their world centers differ in x and z.  The oblique screen x row
        # brings their silhouettes back onto the same display region.
        first = SphereSpec("first", (0.0, 0.0, 0.0), 0.7)
        second = SphereSpec("second", (-1.4, -0.8, 4.0), 0.7)
        result = compute_global_quadric_frame([], (second, first), OBLIQUE_VIEW)
        self.assertEqual(
            result.surface_depth_evidence[0].projection_relation,
            "overlap",
        )
        self.assertEqual(_constraint_pairs(result), (("first", "second"),))

    def test_curve_visibility_and_policy_are_passed_through(self) -> None:
        sphere = SphereSpec("sphere", (0.0, 0.0, 0.0), 1.0)
        curve = SegmentCurve("axis", (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        physical = compute_global_quadric_frame(
            (curve,), (sphere,), IDENTITY_VIEW, paint_policy="physical"
        )
        teaching = compute_global_quadric_frame(
            (curve,), (sphere,), IDENTITY_VIEW, paint_policy="diagrammatic"
        )
        self.assertEqual(physical.frame.paint_policy.value, "physical")
        self.assertEqual(teaching.frame.paint_policy.value, "diagrammatic")
        self.assertTrue(physical.frame.omitted_fragment_ids)
        self.assertFalse(teaching.frame.omitted_fragment_ids)

    def test_input_order_does_not_change_canonical_result(self) -> None:
        surfaces = (
            SphereSpec("a", (0.0, 0.0, 0.0), 0.75),
            CylinderSpec("b", (0.0, 0.0, 3.0), (0.0, 1.0, 0.0), 0.5, (-1.0, 1.0)),
        )
        curves = (
            SegmentCurve("z", (-2.0, 0.0, -1.0), (2.0, 0.0, -1.0)),
            SegmentCurve("a", (-2.0, 0.5, -1.0), (2.0, 0.5, -1.0)),
        )
        first = compute_global_quadric_frame(curves, surfaces, IDENTITY_VIEW)
        second = compute_global_quadric_frame(
            tuple(reversed(curves)), tuple(reversed(surfaces)), IDENTITY_VIEW
        )
        self.assertEqual(first, second)
        self.assertEqual(
            canonical_global_quadric_frame_json(first),
            canonical_global_quadric_frame_json(second),
        )
        payload = json.loads(canonical_global_quadric_frame_json(first))
        self.assertEqual(payload, first.to_dict())

    def test_injected_three_surface_cycle_fails_explicitly(self) -> None:
        surfaces = (
            SphereSpec("a", (-5.0, 0.0, 0.0), 0.5),
            SphereSpec("b", (0.0, 0.0, 0.0), 0.5),
            SphereSpec("c", (5.0, 0.0, 0.0), 0.5),
        )
        with self.assertRaisesRegex(
            GlobalQuadricOcclusionError,
            "cycle",
        ):
            compute_global_quadric_frame(
                [],
                surfaces,
                IDENTITY_VIEW,
                additional_surface_constraints=(
                    ("a", "b"),
                    ("b", "c"),
                    ("c", "a"),
                ),
            )

    def test_three_real_cylinders_form_a_geometric_painter_cycle(self) -> None:
        def cylinder(
            surface_id: str,
            center: tuple[float, float, float],
            direction: tuple[float, float, float],
            half_parameter: float,
        ) -> CylinderSpec:
            vector = np.asarray(direction, dtype=float)
            length = float(np.linalg.norm(vector))
            return CylinderSpec(
                surface_id,
                center,
                tuple(float(value) for value in vector / length),
                0.12,
                (-half_parameter * length, half_parameter * length),
            )

        # The three projected axes cross at different screen points.  Their
        # affine z values certify a<b, b<c, and c<a at those respective
        # crossings, while the thin finite cylinders remain strictly disjoint.
        surfaces = (
            cylinder("a", (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), 3.0),
            cylinder("b", (-1.0, 1.0, 1.5), (0.0, 1.0, 0.5), 2.0),
            cylinder("c", (0.0, 1.0, 2.0), (1.0, -1.0, -1.0), 2.0),
        )
        self.assertEqual(len(verify_strict_quadric_separation(surfaces)), 3)
        with self.assertRaisesRegex(GlobalQuadricOcclusionError, "cycle"):
            compute_global_quadric_frame([], surfaces, IDENTITY_VIEW)

    def test_fully_hidden_coincident_curves_do_not_block_physical_paint(self) -> None:
        sphere = SphereSpec("occluder", (0.0, 0.0, 3.0), 2.0)
        curves = (
            SegmentCurve("a", (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
            SegmentCurve("b", (-0.5, 0.0, -1.0), (0.5, 0.0, -1.0)),
        )
        physical = compute_global_quadric_frame(
            curves,
            (sphere,),
            IDENTITY_VIEW,
            paint_policy="physical",
        )
        self.assertEqual(len(physical.frame.omitted_fragment_ids), 2)
        self.assertEqual(physical.frame.curve_crossings, ())
        with self.assertRaisesRegex(
            GlobalQuadricOcclusionError,
            "projected curve ordering cannot be certified",
        ):
            compute_global_quadric_frame(
                curves,
                (sphere,),
                IDENTITY_VIEW,
                paint_policy="diagrammatic",
            )


class GlobalQuadricFailClosedTests(unittest.TestCase):
    def test_touching_and_intersecting_spheres_are_rejected(self) -> None:
        first = SphereSpec("first", (0.0, 0.0, 0.0), 1.0)
        for name, center in (("touch", 2.0), ("overlap", 1.5)):
            with self.subTest(name=name), self.assertRaisesRegex(
                GlobalQuadricOcclusionError,
                "touching, intersecting, or numerically inseparable",
            ):
                verify_strict_quadric_separation(
                    (first, SphereSpec(name, (center, 0.0, 0.0), 1.0))
                )

    def test_two_nappe_cone_is_rejected_before_geometry_changes(self) -> None:
        invalid = ConeSpec(
            "double",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            0.3,
            (-1.0, 1.0),
        )
        with self.assertRaisesRegex(GlobalQuadricOcclusionError, "two nappes"):
            compute_global_quadric_frame([], (invalid,), IDENTITY_VIEW)

    def test_similarity_scales_keep_the_same_order(self) -> None:
        normalized: list[tuple[tuple[str, str], ...]] = []
        normalized_gaps: list[float] = []
        for scale in (1.0e-6, 1.0, 1.0e6, 1.0e9):
            far = SphereSpec("far", (0.0, 0.0, 0.0), 0.75 * scale)
            near = SphereSpec("near", (0.0, 0.0, 4.0 * scale), 0.75 * scale)
            result = compute_global_quadric_frame([], (near, far), IDENTITY_VIEW)
            normalized.append(_constraint_pairs(result))
            normalized_gaps.append(
                result.separation_evidence[0].support_gap_lower_bound / scale
            )
        self.assertEqual(normalized, [(("far", "near"),)] * 4)
        np.testing.assert_allclose(
            normalized_gaps,
            np.broadcast_to(normalized_gaps[0], (4,)),
            rtol=1.0e-9,
            atol=1.0e-12,
        )

    def test_large_common_translation_preserves_depth_order(self) -> None:
        translation = 1.0e6
        far = SphereSpec(
            "far",
            (translation, translation, translation),
            0.75,
        )
        near = SphereSpec(
            "near",
            (translation, translation, translation + 4.0),
            0.75,
        )
        result = compute_global_quadric_frame([], (near, far), IDENTITY_VIEW)
        self.assertEqual(_constraint_pairs(result), (("far", "near"),))

    def test_common_screen_row_scale_is_geometrically_equivalent(self) -> None:
        tiny_screen_view = ParallelView.from_matrix(
            ((1.0e-9, 0.0, 0.0), (0.0, 1.0e-9, 0.0), (0.0, 0.0, 1.0))
        )
        far = SphereSpec("far", (0.0, 0.0, 0.0), 0.75)
        near = SphereSpec("near", (0.0, 0.0, 4.0), 0.75)
        normal = compute_global_quadric_frame([], (near, far), IDENTITY_VIEW)
        scaled = compute_global_quadric_frame([], (near, far), tiny_screen_view)
        self.assertEqual(_constraint_pairs(normal), _constraint_pairs(scaled))

    def test_valid_near_singular_depth_row_does_not_affect_screen_rays(self) -> None:
        tiny_depth_view = ParallelView.from_matrix(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0e-15))
        )
        far = SphereSpec("far", (0.0, 0.0, 0.0), 0.75)
        near = SphereSpec("near", (0.0, 0.0, 4.0), 0.75)
        result = compute_global_quadric_frame([], (near, far), tiny_depth_view)
        self.assertEqual(_constraint_pairs(result), (("far", "near"),))


if __name__ == "__main__":
    unittest.main()
