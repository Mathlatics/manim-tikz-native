from __future__ import annotations

import copy
from math import pi
import unittest

import numpy as np
from manim import Circle, Line

from tikz_native import compile_document
from tikz_native.fixed_view_renderer import NativeFixedViewRenderer
from tikz_native.manim_renderer_3d import NativeManim3DRenderer
from tikz_native.planar_curves_3d import (
    restore_planar_curve_geometry,
)


OBLIQUE_VIEW = "space view={(-0.35,-0.35),(1,0),(0,1)}"
SIDE_VIEW = "x={(1cm,0cm)},y={(0cm,0cm)},z={(0cm,1cm)}"


def _picture(body: str, *, view: str = OBLIQUE_VIEW):
    document = compile_document(
        source_text=(
            f"\\begin{{tikzpicture}}[{view}]\n"
            f"{body.strip()}\n"
            "\\end{tikzpicture}\n"
        )
    )
    assert len(document.pictures) == 1
    picture = document.pictures[0]
    assert not picture.unsupported, picture.unsupported
    return picture


def _ellipse_picture(*, view: str = OBLIQUE_VIEW):
    return _picture(
        r"""
  \coordinate (O) at (1,2,3);
  \coordinate (U) at (3,2,3);
  \coordinate (V) at (1,5,3);
  \DeclareSpacePlane{plane-a}{O/U/V};
  \DrawSpaceEllipse[draw=blue,line width=1.2pt,draw opacity=0.6]
    {ellipse-a}{plane-a}{0.25,-0.5}{2.5}{0.75};
""",
        view=view,
    )


class TikzPlanarCurves3DRenderingTests(unittest.TestCase):
    def test_fixed_view_rank_two_matches_four_directly_projected_cardinal_points(
        self,
    ) -> None:
        picture = _ellipse_picture()
        renderer = NativeFixedViewRenderer()
        figure = renderer.render(picture)
        rendered = figure.objects["ellipse-a"]
        self.assertIsInstance(rendered, Circle)

        geometry = restore_planar_curve_geometry(
            picture.objects[0].geometry,
            expected_curve_id="ellipse-a",
        )
        analytic = geometry.curve.lower_to_analytic_curve()
        np.testing.assert_allclose(
            rendered.get_center(),
            renderer.point(analytic.center, picture),
            atol=1.0e-12,
        )
        anchors = rendered.get_anchors()
        for anchor_index, parameter in (
            (0, 0.0),
            (3, pi / 2.0),
            (7, pi),
            (11, 3.0 * pi / 2.0),
        ):
            expected = renderer.point(analytic.point(parameter), picture)
            np.testing.assert_allclose(
                anchors[anchor_index],
                expected,
                atol=1.0e-12,
            )
        self.assertAlmostEqual(
            float(rendered.get_stroke_opacity()),
            0.6,
            places=12,
        )

    def test_fixed_view_exact_side_view_is_one_finite_segment(self) -> None:
        picture = _picture(
            r"""
  \coordinate (O) at (0,0,1);
  \coordinate (U) at (1,0,1);
  \coordinate (V) at (0,1,1);
  \DeclareSpacePlane{plane-a}{O/U/V};
  \DrawSpaceCircle{circle-a}{plane-a}{0.5,0.25}{2};
""",
            view=SIDE_VIEW,
        )

        rendered = NativeFixedViewRenderer().render(picture).objects["circle-a"]

        self.assertIsInstance(rendered, Line)
        np.testing.assert_allclose(rendered.get_start(), (-1.5, 1.0, 0.0), atol=1.0e-12)
        np.testing.assert_allclose(rendered.get_end(), (2.5, 1.0, 0.0), atol=1.0e-12)

    def test_fixed_view_does_not_collapse_a_resolved_thin_ellipse(self) -> None:
        picture = _picture(
            r"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{plane-a}{O/U/V};
  \DrawSpaceEllipse{ellipse-a}{plane-a}{0,0}{2}{1e-10};
""",
            view="x={(1cm,0cm)},y={(0cm,1cm)},z={(0cm,0cm)}",
        )

        rendered = NativeFixedViewRenderer().render(picture).objects["ellipse-a"]

        self.assertIsInstance(rendered, Circle)
        self.assertNotIsInstance(rendered, Line)

    def test_world_renderer_retains_authored_center_axes_and_object_identity(
        self,
    ) -> None:
        picture = _ellipse_picture()
        renderer = NativeManim3DRenderer()
        figure = renderer.render(picture)
        rendered = figure.objects["ellipse-a"]
        identity = id(rendered)
        geometry = restore_planar_curve_geometry(
            picture.objects[0].geometry,
            expected_curve_id="ellipse-a",
        )
        analytic = geometry.curve.lower_to_analytic_curve()

        np.testing.assert_allclose(
            rendered.get_center(),
            renderer.point(analytic.center, picture),
            atol=1.0e-12,
        )

        anchors = rendered.get_anchors()
        for anchor_index, parameter in (
            (0, 0.0),
            (3, pi / 2.0),
            (7, pi),
            (11, 3.0 * pi / 2.0),
        ):
            expected = renderer.point(analytic.point(parameter), picture)
            np.testing.assert_allclose(
                anchors[anchor_index],
                expected,
                atol=1.0e-12,
            )
        self.assertEqual(id(figure.objects["ellipse-a"]), identity)
        self.assertIn(rendered, figure.world_group.submobjects)
        self.assertNotIn(rendered, figure.fixed_orientation_labels)

    def test_both_renderers_reject_a_canonical_partial_revolution_payload(
        self,
    ) -> None:
        picture = _ellipse_picture()
        spec = picture.objects[0]
        geometry = restore_planar_curve_geometry(
            spec.geometry,
            expected_curve_id=spec.id,
        )
        spec.geometry = copy.deepcopy(spec.geometry)
        spec.geometry["curve"]["domain"] = [0.0, pi]

        for renderer in (NativeFixedViewRenderer(), NativeManim3DRenderer()):
            with self.subTest(renderer=type(renderer).__name__):
                with self.assertRaisesRegex(RuntimeError, "revolution"):
                    renderer.render(picture)

    def test_both_renderers_revalidate_style_and_payload_kind(self) -> None:
        for mutation, expected in (
            ("fill", "fill"),
            ("dash", "dashed"),
            ("arrow", "arrow"),
            ("boolean-width", "line width"),
            ("boolean-opacity", "opacity"),
            ("kind", "disagrees"),
        ):
            with self.subTest(mutation=mutation):
                picture = _ellipse_picture()
                spec = picture.objects[0]
                if mutation == "fill":
                    spec.style.fill_color = "#FF0000"
                elif mutation == "dash":
                    spec.style.dash_pattern_pt = (2.0, 2.0)
                elif mutation == "arrow":
                    spec.style.arrow_tip = "Stealth"
                elif mutation == "boolean-width":
                    spec.style.line_width_pt = True
                elif mutation == "boolean-opacity":
                    spec.style.opacity = True
                else:
                    spec.kind = "planar_circle_3d"
                for renderer in (
                    NativeFixedViewRenderer(),
                    NativeManim3DRenderer(),
                ):
                    with self.subTest(renderer=type(renderer).__name__):
                        with self.assertRaisesRegex(RuntimeError, expected):
                            renderer.render(picture)

    def test_both_renderers_reject_valid_but_conflicting_plane_evidence(self) -> None:
        picture = _ellipse_picture()
        conflicting = _picture(
            r"""
  \coordinate (O) at (10,20,30);
  \coordinate (U) at (10,22,30);
  \coordinate (V) at (10,20,34);
  \DeclareSpacePlane{plane-a}{O/U/V};
  \DrawSpaceEllipse[draw=blue]{ellipse-a}{plane-a}{0.25,-0.5}{2.5}{0.75};
"""
        )
        picture.objects[0].geometry = copy.deepcopy(
            conflicting.objects[0].geometry
        )

        for renderer in (NativeFixedViewRenderer(), NativeManim3DRenderer()):
            with self.subTest(renderer=type(renderer).__name__):
                with self.assertRaisesRegex(RuntimeError, "registered supporting plane"):
                    renderer.render(picture)

    def test_both_renderers_fail_before_nonfinite_display_geometry_reaches_manim(
        self,
    ) -> None:
        for unit in (
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            5.0e-324,
            1.0e-323,
            1.0e308,
        ):
            with self.subTest(unit=unit):
                picture = _ellipse_picture()
                for renderer_type in (
                    NativeFixedViewRenderer,
                    NativeManim3DRenderer,
                ):
                    with self.subTest(renderer=renderer_type.__name__):
                        renderer = renderer_type(scene_unit_per_cm=unit)
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "finite|display scale|representable|underflow",
                        ):
                            renderer.render(picture)

    def test_fixed_view_rejects_a_radius_lost_at_a_large_projected_center(
        self,
    ) -> None:
        picture = _picture(
            r"""
  \coordinate (O) at (1,0,0);
  \coordinate (U) at (1,1,0);
  \coordinate (V) at (1,0,1);
  \DeclareSpacePlane{plane-a}{O/U/V};
  \DrawSpaceCircle{circle-a}{plane-a}{0,0}{1e-5};
""",
            view="x={(1e12cm,0cm)},y={(0cm,1cm)},z={(1cm,0cm)}",
        )

        with self.assertRaisesRegex(RuntimeError, "display center"):
            NativeFixedViewRenderer().render(picture)

        world = NativeManim3DRenderer().render(picture).objects["circle-a"]
        extents = np.ptp(world.get_all_points(), axis=0)
        self.assertEqual(int(np.count_nonzero(extents > 0.0)), 2)

    def test_fixed_view_rejects_a_rank_one_direction_lost_at_a_large_center(
        self,
    ) -> None:
        picture = _picture(
            r"""
  \coordinate (O) at (1,0,0);
  \coordinate (U) at (1,1,0);
  \coordinate (V) at (1,0,1);
  \DeclareSpacePlane{plane-a}{O/U/V};
  \DrawSpaceCircle{circle-a}{plane-a}{0,0}{5e-5};
""",
            view="x={(1e12cm,0cm)},y={(0cm,0cm)},z={(1cm,1cm)}",
        )

        with self.assertRaisesRegex(RuntimeError, "display center"):
            NativeFixedViewRenderer().render(picture)

    def test_world_view_center_includes_a_far_plane_local_curve(self) -> None:
        picture = _picture(
            r"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{plane-a}{O/U/V};
  \DrawSpaceCircle{circle-a}{plane-a}{100,0}{2};
""",
            view="x={(1cm,0cm)},y={(0cm,1cm)},z={(0cm,0cm)}",
        )

        figure = NativeManim3DRenderer().render(picture)

        # Named plane evidence ends at x=1.  Including the circle's x=[98,102]
        # extent moves the combined authored bounding-box center to x=51.
        np.testing.assert_allclose(
            figure.view_center,
            (51.0, 0.0, 0.0),
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
