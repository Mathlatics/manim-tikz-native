from __future__ import annotations

import json
from math import pi, tau
import subprocess
import sys
import unittest

import numpy as np

from polyhedron_visibility.quadrics.conics import (
    ConicKind,
    ConicParameterization,
)
from polyhedron_visibility.quadrics.curves import (
    ANALYTIC_CURVE_SCHEMA,
    CircleArcCurve,
    CurveContractError,
    EllipseArcCurve,
    ParametricConicBranch,
    SegmentCurve,
)
from polyhedron_visibility.topology import ParameterInterval


class AnalyticCurveTests(unittest.TestCase):
    def assert_point_close(
        self,
        actual: tuple[float, float, float],
        expected: tuple[float, float, float],
    ) -> None:
        self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=1.0e-12))

    def test_segment_uses_its_authored_parameter_domain(self) -> None:
        curve = SegmentCurve(
            " segment ",
            (1.0, 2.0, 3.0),
            (5.0, 4.0, 3.0),
            ParameterInterval(2.0, 6.0),
        )
        self.assertEqual(curve.curve_id, "segment")
        self.assert_point_close(curve.point(2.0), (1.0, 2.0, 3.0))
        self.assert_point_close(curve.point(4.0), (3.0, 3.0, 3.0))
        self.assert_point_close(curve.point(6.0), (5.0, 4.0, 3.0))
        self.assert_point_close(curve.tangent(3.0), (1.0, 0.5, 0.0))
        self.assertAlmostEqual(curve.length, np.sqrt(20.0))

    def test_segment_rejects_invalid_or_degenerate_geometry(self) -> None:
        with self.assertRaisesRegex(CurveContractError, "curve_id"):
            SegmentCurve(" ", (0, 0, 0), (1, 0, 0))
        with self.assertRaisesRegex(CurveContractError, "finite"):
            SegmentCurve("s", (0, 0, float("nan")), (1, 0, 0))
        with self.assertRaisesRegex(CurveContractError, "distinct"):
            SegmentCurve("s", (0, 0, 0), (0, 0, 0))
        with self.assertRaisesRegex(CurveContractError, "positive length"):
            SegmentCurve(
                "s",
                (0, 0, 0),
                (1, 0, 0),
                ParameterInterval(1.0, 1.0),
            )

    def test_curve_evaluation_rejects_parameters_outside_domain(self) -> None:
        curve = SegmentCurve("s", (0, 0, 0), (1, 0, 0))
        with self.assertRaisesRegex(CurveContractError, "outside"):
            curve.point(-0.01)
        with self.assertRaisesRegex(CurveContractError, "outside"):
            curve.tangent(1.01)
        with self.assertRaisesRegex(CurveContractError, "finite"):
            curve.point(float("inf"))

    def test_ellipse_arc_has_exact_point_tangent_and_plane_normal(self) -> None:
        curve = EllipseArcCurve(
            "ellipse",
            (1.0, 2.0, 3.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            ParameterInterval(0.0, pi),
        )
        self.assert_point_close(curve.point(0.0), (3.0, 2.0, 3.0))
        self.assert_point_close(curve.point(pi / 2.0), (1.0, 3.0, 3.0))
        self.assert_point_close(curve.tangent(0.0), (0.0, 1.0, 0.0))
        self.assert_point_close(curve.tangent(pi / 2.0), (-2.0, 0.0, 0.0))
        self.assert_point_close(curve.normal, (0.0, 0.0, 1.0))
        self.assertEqual(curve.semi_axis_lengths, (2.0, 1.0))
        self.assertFalse(curve.closed)
        self.assertFalse(curve.circular)

    def test_full_ellipse_is_closed_and_never_exceeds_one_revolution(self) -> None:
        curve = EllipseArcCurve(
            "full",
            (0, 0, 0),
            (2, 0, 0),
            (0, 1, 0),
            ParameterInterval(-pi, pi),
        )
        self.assertTrue(curve.closed)
        with self.assertRaisesRegex(CurveContractError, "one revolution"):
            EllipseArcCurve(
                "too-long",
                (0, 0, 0),
                (2, 0, 0),
                (0, 1, 0),
                ParameterInterval(0.0, tau + 1.0e-6),
            )

    def test_ellipse_axes_must_be_nonzero_and_orthogonal(self) -> None:
        with self.assertRaisesRegex(CurveContractError, "first_axis"):
            EllipseArcCurve("e", (0, 0, 0), (0, 0, 0), (0, 1, 0))
        with self.assertRaisesRegex(CurveContractError, "orthogonal"):
            EllipseArcCurve("e", (0, 0, 0), (1, 0, 0), (1, 1, 0))

    def test_circle_factory_builds_a_deterministic_right_handed_frame(self) -> None:
        curve = CircleArcCurve(
            "circle",
            (1, 2, 3),
            2.0,
            (0, 0, 4),
            domain=ParameterInterval(0.0, pi),
        )
        self.assertEqual(curve.radius, 2.0)
        self.assertTrue(curve.circular)
        self.assert_point_close(curve.first_axis, (2.0, 0.0, 0.0))
        self.assert_point_close(curve.second_axis, (0.0, 2.0, 0.0))
        self.assert_point_close(curve.normal, (0.0, 0.0, 1.0))
        self.assert_point_close(curve.point(0.0), (3.0, 2.0, 3.0))

    def test_circle_factory_respects_an_authored_radial_axis(self) -> None:
        curve = CircleArcCurve(
            "circle",
            (0, 0, 0),
            3.0,
            (0, 0, 1),
            radial_axis=(1, 1, 2),
        )
        expected = np.asarray((1.0, 1.0, 0.0)) / np.sqrt(2.0) * 3.0
        self.assertTrue(np.allclose(curve.first_axis, expected, atol=1.0e-12))
        with self.assertRaisesRegex(CurveContractError, "parallel"):
            CircleArcCurve(
                "bad",
                (0, 0, 0),
                1.0,
                (0, 0, 1),
                radial_axis=(0, 0, 2),
            )

    def test_circle_factory_rejects_invalid_radius_or_normal(self) -> None:
        with self.assertRaisesRegex(CurveContractError, "radius"):
            CircleArcCurve("bad", (0, 0, 0), 0.0, (0, 0, 1))
        with self.assertRaisesRegex(CurveContractError, "normal"):
            CircleArcCurve("bad", (0, 0, 0), 1.0, (0, 0, 0))

    def test_parametric_conic_branch_maps_exact_analytic_geometry_to_world(self) -> None:
        parameterization = ConicParameterization(
            kind=ConicKind.ELLIPSE,
            branch_label="closed",
            origin=(0.0, 0.0),
            first_axis=(2.0, 0.0),
            second_axis=(0.0, 1.0),
            natural_domain=ParameterInterval(0.0, tau),
            closed=True,
        )
        embedding = (
            (1.0, 0.0, 10.0),
            (0.0, 0.0, 20.0),
            (0.0, 1.0, 30.0),
            (0.0, 0.0, 1.0),
        )
        curve = ParametricConicBranch(
            "section:closed",
            parameterization,
            embedding,
            ParameterInterval(0.0, pi),
        )
        self.assert_point_close(curve.point(0.0), (12.0, 20.0, 30.0))
        self.assert_point_close(curve.point(pi / 2.0), (10.0, 20.0, 31.0))
        self.assert_point_close(curve.tangent(0.0), (0.0, 0.0, 1.0))

    def test_parametric_conic_branch_validates_embedding_and_domain(self) -> None:
        parameterization = ConicParameterization(
            kind=ConicKind.ELLIPSE,
            branch_label="closed",
            origin=(0, 0),
            first_axis=(1, 0),
            second_axis=(0, 1),
            natural_domain=ParameterInterval(0.0, tau),
            closed=True,
        )
        valid_embedding = (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 0),
            (0, 0, 1),
        )
        with self.assertRaisesRegex(CurveContractError, "outside"):
            ParametricConicBranch(
                "branch",
                parameterization,
                valid_embedding,
                ParameterInterval(-0.1, 1.0),
            )
        with self.assertRaisesRegex(CurveContractError, "independent"):
            ParametricConicBranch(
                "branch",
                parameterization,
                (
                    (1, 1, 0),
                    (0, 0, 0),
                    (0, 0, 0),
                    (0, 0, 1),
                ),
                ParameterInterval(0.0, 1.0),
            )

    def test_all_curve_contracts_are_json_serializable(self) -> None:
        parameterization = ConicParameterization(
            kind=ConicKind.PARABOLA,
            branch_label="main",
            origin=(0, 0),
            first_axis=(1, 0),
            second_axis=(0, 1),
        )
        curves = (
            SegmentCurve("s", (0, 0, 0), (1, 0, 0)),
            EllipseArcCurve("e", (0, 0, 0), (2, 0, 0), (0, 1, 0)),
            CircleArcCurve("c", (0, 0, 0), 1, (0, 0, 1)),
            ParametricConicBranch(
                "p",
                parameterization,
                (
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 0),
                    (0, 0, 1),
                ),
                ParameterInterval(-2.0, 2.0),
            ),
        )
        payloads = [curve.to_dict() for curve in curves]
        encoded = json.dumps(payloads, ensure_ascii=False, sort_keys=True)
        self.assertIn(ANALYTIC_CURVE_SCHEMA, encoded)
        self.assertEqual(
            [payload["curveId"] for payload in payloads],
            ["s", "e", "c", "p"],
        )

    def test_curves_module_does_not_import_manim(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import polyhedron_visibility.quadrics.curves; "
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
