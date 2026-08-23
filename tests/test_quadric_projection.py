from __future__ import annotations

from math import cos, pi, sin, tau
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    SphereSpec,
)
from polyhedron_visibility.quadrics.projection import (
    ProjectionProxyError,
    ProjectionSubdivisionError,
    build_opaque_projection_proxy,
    canonical_opaque_projection_proxy_json,
)


ROOT = Path(__file__).resolve().parents[1]
ORTHOGONAL_VIEW = ParallelView.from_matrix(np.eye(3))
OBLIQUE_MATRIX = np.asarray(
    ((1.0, 0.0, 0.35), (0.0, 1.0, 0.2), (0.0, 0.0, 1.0))
)
OBLIQUE_VIEW = ParallelView.from_matrix(OBLIQUE_MATRIX)


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    displacement = end - start
    squared_length = float(np.dot(displacement, displacement))
    if squared_length == 0.0:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, displacement) / squared_length)
    ratio = min(1.0, max(0.0, ratio))
    return float(np.linalg.norm(point - start - ratio * displacement))


def _distance_to_boundary(point: np.ndarray, boundary: np.ndarray) -> float:
    return min(
        _point_segment_distance(point, start, end)
        for start, end in zip(boundary, boundary[1:])
    )


def _analytic_support_point(
    surface: SphereSpec | CylinderSpec | ConeSpec,
    screen_matrix: np.ndarray,
    angle: float,
) -> np.ndarray:
    direction = np.asarray((cos(angle), sin(angle)), dtype=float)
    covector = screen_matrix.T @ direction
    if isinstance(surface, SphereSpec):
        world = (
            np.asarray(surface.center, dtype=float)
            + surface.radius * covector / float(np.linalg.norm(covector))
        )
        return screen_matrix @ world

    frame = surface.frame
    axis = np.asarray(frame.z_axis, dtype=float)
    x_axis = np.asarray(frame.x_axis, dtype=float)
    y_axis = np.asarray(frame.y_axis, dtype=float)
    radial_x = float(np.dot(covector, x_axis))
    radial_y = float(np.dot(covector, y_axis))
    radial_norm = float(np.hypot(radial_x, radial_y))
    candidates = []
    for axial in surface.axial_range:
        radius = (
            surface.radius
            if isinstance(surface, CylinderSpec)
            else abs(axial) * surface.slope
        )
        radial = (
            np.zeros(3, dtype=float)
            if radial_norm == 0.0
            else radius
            * (radial_x * x_axis + radial_y * y_axis)
            / radial_norm
        )
        origin = (
            np.asarray(surface.origin, dtype=float)
            if isinstance(surface, CylinderSpec)
            else np.asarray(surface.apex, dtype=float)
        )
        candidates.append(origin + axial * axis + radial)
    world = max(candidates, key=lambda point: float(np.dot(covector, point)))
    return screen_matrix @ world


def _dense_outline_error(
    surface: SphereSpec | CylinderSpec | ConeSpec,
    screen_matrix: np.ndarray,
    boundary: tuple[tuple[float, float], ...],
    *,
    samples: int = 1201,
) -> float:
    polygon = np.asarray(boundary, dtype=float)
    return max(
        _distance_to_boundary(
            _analytic_support_point(surface, screen_matrix, angle),
            polygon,
        )
        for angle in np.linspace(0.0, tau, samples)
    )


def _assert_counter_clockwise_convex(
    test: unittest.TestCase,
    boundary: tuple[tuple[float, float], ...],
) -> None:
    test.assertEqual(boundary[0], boundary[-1])
    vertices = np.asarray(boundary[:-1], dtype=float)
    scale = max(float(np.max(np.abs(vertices))), 1.0)
    tolerance = 1.0e-10 * scale * scale
    area_twice = 0.0
    for index, point in enumerate(vertices):
        following = vertices[(index + 1) % len(vertices)]
        area_twice += point[0] * following[1] - point[1] * following[0]
        previous = vertices[index - 1]
        first = point - previous
        second = following - point
        cross = first[0] * second[1] - first[1] * second[0]
        test.assertGreaterEqual(cross, -tolerance)
    test.assertGreater(area_twice, 0.0)


class SphereProjectionProxyTests(unittest.TestCase):
    def test_orthogonal_projection_is_a_closed_counter_clockwise_circle(self) -> None:
        sphere = SphereSpec("sphere", (1.0, -2.0, 3.0), 2.0)
        proxy = build_opaque_projection_proxy(
            sphere,
            ORTHOGONAL_VIEW,
            max_chord_error=0.01,
        )
        self.assertEqual(proxy.patch_id, "sphere:opaque-projection")
        self.assertEqual(proxy.surface_id, "sphere")
        self.assertEqual(proxy.boundary_points[0], (3.0, -2.0))
        for point in proxy.vertices:
            self.assertAlmostEqual(
                float(np.linalg.norm(np.asarray(point) - (1.0, -2.0))),
                2.0,
                places=11,
            )
        _assert_counter_clockwise_convex(self, proxy.boundary_points)
        self.assertFalse(proxy.metadata.visibility_authoritative)
        self.assertNotIn("depth", proxy.to_dict())
        self.assertNotIn("occluders", proxy.to_dict())

    def test_oblique_projection_pulls_screen_support_back_to_world(self) -> None:
        sphere = SphereSpec("sphere", (0.5, -1.0, 2.0), 1.75)
        proxy = build_opaque_projection_proxy(
            sphere,
            OBLIQUE_VIEW,
            max_chord_error=0.005,
        )
        screen = OBLIQUE_MATRIX[:2]
        direction = np.asarray((1.0, 0.0))
        covector = screen.T @ direction
        expected_world = (
            np.asarray(sphere.center)
            + sphere.radius * covector / float(np.linalg.norm(covector))
        )
        np.testing.assert_allclose(
            proxy.boundary_points[0],
            screen @ expected_world,
            rtol=0.0,
            atol=1.0e-12,
        )

        # Every sampled point lies on the exact projected ellipse.
        center = screen @ np.asarray(sphere.center)
        covariance = sphere.radius * sphere.radius * (screen @ screen.T)
        inverse = np.linalg.inv(covariance)
        for point in proxy.vertices:
            offset = np.asarray(point) - center
            self.assertAlmostEqual(float(offset @ inverse @ offset), 1.0, places=10)

    def test_similarity_scale_scales_points_error_and_segment_count(self) -> None:
        base = SphereSpec("base", (0.5, -1.0, 2.0), 1.5)
        factor = 1.0e4
        scaled = SphereSpec(
            "scaled",
            tuple(factor * np.asarray(base.center)),
            factor * base.radius,
        )
        first = build_opaque_projection_proxy(
            base,
            OBLIQUE_VIEW,
            max_chord_error=0.004,
        )
        second = build_opaque_projection_proxy(
            scaled,
            OBLIQUE_VIEW,
            max_chord_error=factor * 0.004,
        )
        self.assertEqual(first.metadata.segment_count, second.metadata.segment_count)
        np.testing.assert_allclose(
            np.asarray(second.boundary_points) / factor,
            first.boundary_points,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        self.assertAlmostEqual(
            second.metadata.observed_chord_error / factor,
            first.metadata.observed_chord_error,
            places=11,
        )


class CylinderAndConeProjectionProxyTests(unittest.TestCase):
    def test_rotated_translated_cylinder_keeps_stable_order_and_translation(self) -> None:
        axis = (1.0, 2.0, 3.0)
        radial_axis = (2.0, -1.0, 0.0)
        base = CylinderSpec(
            "base-cylinder",
            (0.0, 0.0, 0.0),
            axis,
            1.5,
            (-2.0, 3.0),
            radial_axis=radial_axis,
        )
        translation = np.asarray((4.0, -3.0, 2.0))
        moved = CylinderSpec(
            "moved-cylinder",
            tuple(translation),
            axis,
            1.5,
            (-2.0, 3.0),
            radial_axis=radial_axis,
        )
        first = build_opaque_projection_proxy(
            base, OBLIQUE_VIEW, max_chord_error=0.005
        )
        repeated = build_opaque_projection_proxy(
            base, OBLIQUE_VIEW, max_chord_error=0.005
        )
        second = build_opaque_projection_proxy(
            moved, OBLIQUE_VIEW, max_chord_error=0.005
        )
        self.assertEqual(
            canonical_opaque_projection_proxy_json(first),
            canonical_opaque_projection_proxy_json(repeated),
        )
        self.assertEqual(len(first.vertices), len(second.vertices))
        expected_shift = OBLIQUE_MATRIX[:2] @ translation
        np.testing.assert_allclose(
            np.asarray(second.boundary_points)
            - np.asarray(first.boundary_points),
            np.broadcast_to(
                expected_shift,
                np.asarray(first.boundary_points).shape,
            ),
            rtol=0.0,
            atol=2.0e-12,
        )
        _assert_counter_clockwise_convex(self, first.boundary_points)

    def test_side_view_single_nappe_cone_contains_apex_and_is_convex(self) -> None:
        side_matrix = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        )
        side_view = ParallelView.from_matrix(side_matrix)
        cone = ConeSpec(
            "cone",
            (1.0, -2.0, 0.5),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        proxy = build_opaque_projection_proxy(
            cone,
            side_view,
            max_chord_error=0.002,
        )
        projected_apex = tuple(
            float(value) for value in side_matrix[:2] @ np.asarray(cone.apex)
        )
        self.assertIn(projected_apex, proxy.vertices)
        _assert_counter_clockwise_convex(self, proxy.boundary_points)

    def test_cross_nappe_cone_is_rejected_before_a_proxy_is_returned(self) -> None:
        cone = ConeSpec(
            "double-cone",
            (0, 0, 0),
            (0, 0, 1),
            pi / 5,
            (-2.0, 3.0),
        )
        with self.assertRaisesRegex(ProjectionProxyError, "two nappes"):
            build_opaque_projection_proxy(cone, ORTHOGONAL_VIEW)


class AdaptiveProjectionTests(unittest.TestCase):
    def test_large_translation_preserves_shape_and_truthful_error_metadata(
        self,
    ) -> None:
        tolerance = 1.0e-3
        shift = np.asarray((1.0e12, -1.0e12, 0.0))
        base_surface = SphereSpec("base", (0.0, 0.0, 0.0), 1.0)
        moved_surface = SphereSpec("moved", tuple(shift), 1.0)
        base = build_opaque_projection_proxy(
            base_surface,
            ORTHOGONAL_VIEW,
            max_chord_error=tolerance,
        )
        moved = build_opaque_projection_proxy(
            moved_surface,
            ORTHOGONAL_VIEW,
            max_chord_error=tolerance,
        )

        self.assertEqual(base.metadata.segment_count, moved.metadata.segment_count)
        screen_shift = shift[:2]
        screen_ulp = max(abs(float(np.spacing(value))) for value in screen_shift)
        np.testing.assert_allclose(
            np.asarray(moved.boundary_points) - screen_shift,
            base.boundary_points,
            rtol=0.0,
            atol=2.0 * screen_ulp,
        )
        dense_error = _dense_outline_error(
            moved_surface,
            np.eye(3)[:2],
            moved.boundary_points,
            samples=2401,
        )
        self.assertLessEqual(dense_error, tolerance)
        self.assertLessEqual(dense_error, moved.metadata.observed_chord_error)

    def test_large_translation_below_float_resolution_fails_closed(self) -> None:
        sphere = SphereSpec("sphere", (1.0e12, -1.0e12, 0.0), 1.0)
        with self.assertRaisesRegex(
            ProjectionProxyError,
            "floating-point screen resolution",
        ):
            build_opaque_projection_proxy(
                sphere,
                ORTHOGONAL_VIEW,
                max_chord_error=1.0e-5,
            )

    def test_tighter_error_converges_for_oblique_sphere_cylinder_and_cone(self) -> None:
        surfaces = (
            SphereSpec("sphere", (1, -2, 0.5), 2.0),
            CylinderSpec(
                "cylinder", (1, -2, 0.5), (1, 2, 3), 1.5, (-2, 3)
            ),
            ConeSpec(
                "cone", (1, -2, 0.5), (1, 2, 3), pi / 6, (0, 4)
            ),
        )
        screen = OBLIQUE_MATRIX[:2]
        for surface in surfaces:
            with self.subTest(surface=surface.surface_id):
                coarse = build_opaque_projection_proxy(
                    surface,
                    OBLIQUE_VIEW,
                    max_chord_error=0.04,
                )
                fine = build_opaque_projection_proxy(
                    surface,
                    OBLIQUE_VIEW,
                    max_chord_error=0.004,
                )
                coarse_error = _dense_outline_error(
                    surface, screen, coarse.boundary_points
                )
                fine_error = _dense_outline_error(
                    surface, screen, fine.boundary_points
                )
                self.assertLessEqual(coarse_error, 0.04 * 1.01)
                self.assertLessEqual(fine_error, 0.004 * 1.01)
                self.assertLess(fine_error, coarse_error)
                self.assertGreaterEqual(
                    fine.metadata.adaptive_interval_count,
                    coarse.metadata.adaptive_interval_count,
                )
                self.assertLessEqual(
                    fine.metadata.observed_chord_error,
                    fine.metadata.max_chord_error,
                )
                _assert_counter_clockwise_convex(self, fine.boundary_points)

    def test_max_segments_fails_closed_with_structured_error_metadata(self) -> None:
        sphere = SphereSpec("sphere", (0, 0, 0), 1)
        with self.assertRaises(ProjectionSubdivisionError) as caught:
            build_opaque_projection_proxy(
                sphere,
                ORTHOGONAL_VIEW,
                max_chord_error=1.0e-12,
                max_segments=8,
            )
        self.assertEqual(caught.exception.max_segments, 8)
        self.assertGreater(caught.exception.observed_chord_error, 1.0e-12)
        self.assertEqual(
            caught.exception.to_dict()["kind"], "max_segments_exceeded"
        )

    def test_singular_parallel_projection_is_rejected(self) -> None:
        sphere = SphereSpec("sphere", (0, 0, 0), 1)
        with self.assertRaisesRegex(ProjectionProxyError, "singular"):
            build_opaque_projection_proxy(
                sphere,
                ((1, 0, 0), (2, 0, 0), (0, 0, 1)),
            )


class RendererNeutralProjectionImportTests(unittest.TestCase):
    def test_projection_module_does_not_import_manim(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import polyhedron_visibility.quadrics.projection; "
                    "assert 'manim' not in sys.modules; "
                    "assert not any(name.startswith('manim.') for name in sys.modules)"
                ),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
