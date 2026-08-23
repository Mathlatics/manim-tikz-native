from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np
from numpy.polynomial import Polynomial

from polyhedron_visibility.geometry import GeometryContext, GeometryQuantity
from polyhedron_visibility.quadrics.roots import (
    MAX_POLYNOMIAL_DEGREE,
    PolynomialRootError,
    RealRoot,
    cluster_real_roots,
    solve_real_polynomial,
)
from polyhedron_visibility.topology import ParameterInterval


class QuadricPolynomialRootTests(unittest.TestCase):
    def assert_root_values(
        self,
        roots: tuple[RealRoot, ...],
        expected: tuple[float, ...],
        *,
        places: int = 10,
    ) -> None:
        self.assertEqual(len(roots), len(expected))
        for root, value in zip(roots, expected):
            self.assertAlmostEqual(root.value, value, places=places)

    def test_linear_and_quadratic_roots_are_sorted(self) -> None:
        self.assert_root_values(solve_real_polynomial((-4.0, 2.0)), (2.0,))
        roots = solve_real_polynomial((-1.0, 0.0, 1.0))
        self.assert_root_values(roots, (-1.0, 1.0))
        self.assertEqual(tuple(root.multiplicity for root in roots), (1, 1))

    def test_cubic_roots_are_deterministic(self) -> None:
        coefficients = (-6.0, 11.0, -6.0, 1.0)
        first = solve_real_polynomial(coefficients)
        second = solve_real_polynomial(coefficients)
        self.assertEqual(first, second)
        self.assert_root_values(first, (1.0, 2.0, 3.0))
        self.assertTrue(all(root.residual < 1.0e-14 for root in first))

    def test_even_double_root_is_not_lost_without_a_sign_change(self) -> None:
        roots = solve_real_polynomial((0.0625, -0.5, 1.0), domain=(-1.0, 1.0))
        self.assert_root_values(roots, (0.25,))
        self.assertEqual(roots[0].multiplicity, 2)
        self.assertLessEqual(roots[0].residual, 1.0e-15)

    def test_two_distinct_even_roots_keep_their_multiplicity(self) -> None:
        coefficients = Polynomial.fromroots((-0.5, -0.5, 0.25, 0.25)).coef
        roots = solve_real_polynomial(coefficients, domain=(-1.0, 1.0))
        self.assert_root_values(roots, (-0.5, 0.25))
        self.assertEqual(tuple(root.multiplicity for root in roots), (2, 2))

    def test_finite_domain_filters_roots_and_keeps_endpoints(self) -> None:
        coefficients = Polynomial.fromroots((-2.0, 0.0, 0.5, 1.0, 3.0)).coef
        roots = solve_real_polynomial(
            coefficients,
            domain=ParameterInterval(0.0, 1.0),
        )
        self.assert_root_values(roots, (0.0, 0.5, 1.0))

    def test_stationary_positive_minimum_is_not_a_false_root(self) -> None:
        self.assertEqual(
            solve_real_polynomial((1.0e-6, 0.0, 1.0), domain=(-1.0, 1.0)),
            (),
        )

    def test_close_simple_roots_cluster_by_parameter_tolerance(self) -> None:
        coefficients = Polynomial.fromroots((0.25, 0.2500005)).coef
        separate = solve_real_polynomial(
            coefficients,
            domain=(-1.0, 1.0),
            parameter_tolerance=1.0e-9,
        )
        self.assertEqual(len(separate), 2)
        self.assertEqual(tuple(root.multiplicity for root in separate), (1, 1))

        clustered = solve_real_polynomial(
            coefficients,
            domain=(-1.0, 1.0),
            parameter_tolerance=1.0e-3,
        )
        self.assertEqual(len(clustered), 1)
        self.assertAlmostEqual(clustered[0].value, 0.25, places=7)
        self.assertEqual(clustered[0].multiplicity, 2)

    def test_cluster_real_roots_uses_geometry_parameter_tolerance(self) -> None:
        context = GeometryContext(
            overrides={GeometryQuantity.PARAMETER: 1.0e-3}
        )
        roots = cluster_real_roots(
            (
                RealRoot(2.0, 1, 0.0),
                RealRoot(1.0005, 1, 2.0e-16),
                RealRoot(1.0, 1, 1.0e-16),
            ),
            context=context,
        )
        self.assertEqual(
            roots,
            (
                RealRoot(1.0, 2, 1.0e-16),
                RealRoot(2.0, 1, 0.0),
            ),
        )

    def test_degree_eight_with_simple_roots_is_supported(self) -> None:
        expected = (-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8)
        coefficients = Polynomial.fromroots(expected).coef
        roots = solve_real_polynomial(
            coefficients,
            domain=(-1.0, 1.0),
            parameter_tolerance=1.0e-10,
        )
        self.assert_root_values(roots, expected)
        self.assertEqual(
            tuple(root.multiplicity for root in roots),
            (1,) * MAX_POLYNOMIAL_DEGREE,
        )

    def test_coefficient_scale_does_not_change_roots(self) -> None:
        coefficients = np.asarray((-6.0, 11.0, -6.0, 1.0)) * 1.0e200
        roots = solve_real_polynomial(coefficients)
        self.assert_root_values(roots, (1.0, 2.0, 3.0))

    def test_zero_length_domain_checks_the_single_parameter(self) -> None:
        at_root = solve_real_polynomial(
            (-1.0, 1.0), domain=ParameterInterval(1.0, 1.0)
        )
        self.assertEqual(at_root, (RealRoot(1.0, 1, 0.0),))
        self.assertEqual(
            solve_real_polynomial(
                (-1.0, 1.0), domain=ParameterInterval(2.0, 2.0)
            ),
            (),
        )

    def test_invalid_polynomials_fail_closed(self) -> None:
        with self.assertRaisesRegex(PolynomialRootError, "zero polynomial"):
            solve_real_polynomial((0.0, 0.0, 0.0))
        with self.assertRaisesRegex(PolynomialRootError, "finite"):
            solve_real_polynomial((1.0, float("nan")))
        with self.assertRaisesRegex(PolynomialRootError, "exceeds supported degree"):
            solve_real_polynomial((1.0,) + (0.0,) * 8 + (1.0,))
        with self.assertRaisesRegex(ValueError, "parameter_tolerance"):
            solve_real_polynomial((-1.0, 1.0), parameter_tolerance=-1.0)
        with self.assertRaisesRegex(ValueError, "residual_tolerance"):
            solve_real_polynomial((-1.0, 1.0), residual_tolerance=-1.0)

    def test_real_root_validates_its_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            RealRoot(float("inf"), 1, 0.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            RealRoot(0.0, 0, 0.0)
        with self.assertRaisesRegex(TypeError, "integer"):
            RealRoot(0.0, True, 0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            RealRoot(0.0, 1, -1.0)

    def test_roots_module_does_not_import_manim(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import polyhedron_visibility.quadrics.roots; "
                    "assert 'manim' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
