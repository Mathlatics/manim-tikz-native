from __future__ import annotations

from math import pi
import unittest

import numpy as np

from polyhedron_visibility.quadrics import Circle3DSpec, Ellipse3DSpec, PlanarFrame3D
from polyhedron_visibility.topology import ParameterInterval
from tikz_native.planar_curve_projection import (
    PlanarCurveProjectionError,
    project_planar_curve_2d,
)


class PlanarCurveProjectionTests(unittest.TestCase):
    def test_rank_two_projection_retains_the_direct_affine_basis(self) -> None:
        frame = PlanarFrame3D(
            "oblique-plane",
            (1.0, 2.0, 3.0),
            (1.0, 1.0, 1.0),
            (1.0, -1.0, 0.0),
        )
        curve = Ellipse3DSpec.from_plane_coordinates(
            "ellipse",
            frame,
            (0.5, -0.25),
            2.5,
            0.75,
        )
        projection = np.array(
            [
                [1.0, 0.25, -0.1],
                [-0.2, 0.8, 0.6],
                [0.0, 0.0, 1.0],
            ]
        )

        result = project_planar_curve_2d(curve, projection)

        self.assertEqual(result.rank, 2)
        self.assertIsNone(result.segment_start)
        self.assertIsNone(result.segment_end)
        self.assertIsNone(result.segment_start_offset)
        self.assertIsNone(result.segment_end_offset)
        analytic = curve.lower_to_analytic_curve()
        expected_center = projection[:2] @ np.asarray(analytic.center)
        expected_basis = projection[:2] @ np.column_stack(
            (analytic.first_axis, analytic.second_axis)
        )
        np.testing.assert_allclose(result.center, expected_center, atol=1.0e-12)
        np.testing.assert_allclose(result.screen_basis, expected_basis, atol=1.0e-12)
        for parameter in np.linspace(0.0, 2.0 * pi, 17):
            projected_point = projection[:2] @ np.asarray(analytic.point(parameter))
            affine_point = (
                np.asarray(result.center)
                + np.cos(parameter) * result.screen_basis[:, 0]
                + np.sin(parameter) * result.screen_basis[:, 1]
            )
            np.testing.assert_allclose(affine_point, projected_point, atol=1.0e-12)

    def test_exact_side_view_circle_becomes_its_finite_diameter(self) -> None:
        frame = PlanarFrame3D(
            "xy-plane",
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        curve = Circle3DSpec.from_plane_coordinates(
            "side-view-circle",
            frame,
            (0.0, 0.0),
            2.0,
        )
        view_along_y = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )

        result = project_planar_curve_2d(curve, view_along_y)

        self.assertEqual(result.rank, 1)
        np.testing.assert_allclose(result.segment_start, (-2.0, 1.0), atol=1.0e-12)
        np.testing.assert_allclose(result.segment_end, (2.0, 1.0), atol=1.0e-12)
        np.testing.assert_allclose(result.segment_start_offset, (-2.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(result.segment_end_offset, (2.0, 0.0), atol=1.0e-12)
        self.assertAlmostEqual(result.singular_values[0], 2.0)
        self.assertAlmostEqual(result.singular_values[1], 0.0)

    def test_rank_one_arc_uses_only_its_finite_parameter_extent(self) -> None:
        frame = PlanarFrame3D(
            "xy-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        curve = Circle3DSpec.from_plane_coordinates(
            "quarter-circle",
            frame,
            (0.0, 0.0),
            3.0,
            domain=ParameterInterval(0.0, pi / 2.0),
        )
        view_along_y = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )

        result = project_planar_curve_2d(curve, view_along_y)

        self.assertEqual(result.rank, 1)
        np.testing.assert_allclose(result.segment_start, (0.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(result.segment_end, (3.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(result.segment_start_offset, (0.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(result.segment_end_offset, (3.0, 0.0), atol=1.0e-12)

    def test_thin_but_resolved_ellipse_is_not_silently_collapsed(self) -> None:
        frame = PlanarFrame3D(
            "xy-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        curve = Ellipse3DSpec.from_plane_coordinates(
            "thin-ellipse",
            frame,
            (0.0, 0.0),
            2.0,
            1.0e-10,
        )

        exact = project_planar_curve_2d(curve, np.identity(3))
        approximated = project_planar_curve_2d(
            curve,
            np.identity(3),
            absolute_rank_tolerance=1.0e-9,
        )

        self.assertEqual(exact.rank, 2)
        self.assertEqual(approximated.rank, 1)

    def test_invalid_or_point_projection_fails_closed(self) -> None:
        frame = PlanarFrame3D(
            "xy-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )
        curve = Circle3DSpec.from_plane_coordinates(
            "circle",
            frame,
            (0.0, 0.0),
            1.0,
        )

        for projection in (
            np.zeros((2, 3)),
            np.zeros((4, 4)),
            np.array([[1.0, 0.0, float("nan")], [0.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            [[True, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ):
            with self.subTest(shape=np.asarray(projection).shape):
                with self.assertRaises(PlanarCurveProjectionError):
                    project_planar_curve_2d(curve, projection)

        with self.assertRaisesRegex(
            PlanarCurveProjectionError,
            "smaller than one",
        ):
            project_planar_curve_2d(
                curve,
                np.identity(3),
                relative_rank_tolerance=1.0,
            )


if __name__ == "__main__":
    unittest.main()
