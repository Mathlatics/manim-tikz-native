from __future__ import annotations

import unittest

import numpy as np

from polyhedron_visibility.quadrics import Circle3DSpec, Ellipse3DSpec
from tikz_native import compile_document
from tikz_native.planar_curves_3d import restore_planar_curve_geometry


SPACE_VIEW = "space view={(-0.35,-0.35),(1,0),(0,1)}"


class TikzPlanarCurves3DCompilerTests(unittest.TestCase):
    def compile_picture(self, body: str, *, options: str = SPACE_VIEW):
        document = compile_document(
            source_text=(
                f"\\begin{{tikzpicture}}[{options}]\n"
                f"{body.strip()}\n"
                "\\end{tikzpicture}\n"
            )
        )
        self.assertEqual(len(document.pictures), 1)
        return document.pictures[0]

    def assert_has_unsupported(
        self,
        picture,
        *expected_fragments: str,
    ) -> None:
        self.assertTrue(picture.unsupported)
        joined = "\n".join(picture.unsupported).lower()
        for fragment in expected_fragments:
            self.assertIn(fragment.lower(), joined, picture.unsupported)

    def test_ordinary_two_dimensional_circles_ellipses_and_named_paths_remain_unchanged(
        self,
    ) -> None:
        picture = self.compile_picture(
            r"""
  \coordinate (O) at (1,2);
  \draw[red] (O) circle (1.5);
  \draw[blue] (O) ellipse [x radius=3,y radius=1];
  \path[name path=named-circle] (O) circle (2);
  \path[name path=named-ellipse] (O) ellipse [x radius=4,y radius=2];
""",
            options="scale=1",
        )

        self.assertEqual(picture.dimension, 2)
        self.assertFalse(picture.unsupported)
        self.assertEqual([item.kind for item in picture.objects], ["circle", "ellipse"])
        self.assertEqual(picture.objects[0].geometry["center"], (1.0, 2.0))
        self.assertEqual(picture.objects[0].geometry["radius"], 1.5)
        self.assertEqual(picture.objects[1].geometry["center"], (1.0, 2.0))
        self.assertEqual(picture.objects[1].geometry["rx"], 3.0)
        self.assertEqual(picture.objects[1].geometry["ry"], 1.0)
        self.assertEqual(set(picture.named_paths), {"named-circle", "named-ellipse"})
        self.assertEqual(picture.named_paths["named-circle"].kind, "ellipse")
        self.assertEqual(picture.named_paths["named-circle"].geometry["rx"], 2.0)
        self.assertEqual(picture.named_paths["named-circle"].geometry["ry"], 2.0)
        self.assertEqual(picture.named_paths["named-ellipse"].kind, "ellipse")
        self.assertEqual(picture.named_paths["named-ellipse"].geometry["rx"], 4.0)
        self.assertEqual(picture.named_paths["named-ellipse"].geometry["ry"], 2.0)

    def test_ordinary_three_dimensional_circle_and_ellipse_paths_fail_closed(
        self,
    ) -> None:
        statements = (
            r"\draw (O) circle (1);",
            r"\draw (O) ellipse [x radius=2,y radius=1];",
            r"\filldraw[fill=gray] (O) circle (1);",
            r"\filldraw[fill=gray] (O) ellipse [x radius=2,y radius=1];",
        )
        for statement in statements:
            with self.subTest(statement=statement):
                picture = self.compile_picture(
                    rf"""
  \coordinate (O) at (0,0,0);
  {statement}
"""
                )
                self.assertEqual(picture.dimension, 3)
                self.assertFalse(picture.objects)
                self.assert_has_unsupported(picture, "plane")

    def test_ordinary_three_dimensional_named_circle_and_ellipse_fail_closed(
        self,
    ) -> None:
        statements = (
            r"\path[name path=C] (O) circle (1);",
            r"\path[name path=E] (O) ellipse [x radius=2,y radius=1];",
        )
        for statement in statements:
            with self.subTest(statement=statement):
                picture = self.compile_picture(
                    rf"""
  \coordinate (O) at (0,0,0);
  {statement}
"""
                )
                self.assertFalse(picture.named_paths)
                self.assert_has_unsupported(picture, "plane")

    def test_three_dimensional_point_marker_remains_a_dot(self) -> None:
        picture = self.compile_picture(
            r"""
  \coordinate (P) at (1,2,3);
  \fill[red] (P) circle (1pt);
"""
        )

        self.assertFalse(picture.unsupported)
        self.assertEqual(len(picture.objects), 1)
        dot = picture.objects[0]
        self.assertEqual(dot.kind, "dot")
        self.assertEqual(dot.geometry["center"], (1.0, 2.0, 3.0))
        self.assertEqual(dot.geometry["center_name"], "P")
        self.assertEqual(dot.geometry["radius_pt"], 1.0)

    def test_explicit_space_plane_circle_and_ellipse_compile_to_certified_payloads(
        self,
    ) -> None:
        picture = self.compile_picture(
            r"""
  \coordinate (O) at (1,2,3);
  \coordinate (U) at (3,2,3);
  \coordinate (V) at (1,5,3);
  \DeclareSpacePlane{base-plane}{O/U/V};
  \DrawSpaceCircle[draw=red,line width=1.2pt]{circle-a}{base-plane}{0.25,-0.5}{1.5};
  \DrawSpaceEllipse[draw=blue]{ellipse-a}{base-plane}{-1,2}{3}{1};
"""
        )

        self.assertFalse(picture.unsupported)
        self.assertEqual(set(picture.planar_frames_3d), {"base-plane"})
        self.assertEqual(
            [item.id for item in picture.objects],
            ["circle-a", "ellipse-a"],
        )
        self.assertEqual(
            [item.kind for item in picture.objects],
            ["planar_circle_3d", "planar_ellipse_3d"],
        )

        expected_geometry_keys = {
            "plane_id",
            "plane_point_names",
            "frame",
            "curve",
            "static",
        }
        for item in picture.objects:
            self.assertEqual(set(item.geometry), expected_geometry_keys)
            self.assertEqual(item.geometry["plane_id"], "base-plane")
            self.assertEqual(item.geometry["plane_point_names"], ["O", "U", "V"])
            self.assertIs(item.geometry["static"], True)
            self.assertEqual(
                item.geometry["frame"],
                picture.planar_frames_3d["base-plane"]["frame"],
            )

        circle_geometry = restore_planar_curve_geometry(
            picture.objects[0].geometry,
            expected_curve_id="circle-a",
        )
        self.assertIsInstance(circle_geometry.curve, Circle3DSpec)
        self.assertEqual(circle_geometry.plane_id, "base-plane")
        self.assertEqual(circle_geometry.plane_point_names, ("O", "U", "V"))
        self.assertEqual(circle_geometry.frame.frame_id, "base-plane")
        self.assertEqual(circle_geometry.frame.point, (1.0, 2.0, 3.0))
        np.testing.assert_allclose(circle_geometry.frame.u_axis, (1.0, 0.0, 0.0))
        np.testing.assert_allclose(circle_geometry.frame.v_axis, (0.0, 1.0, 0.0))
        np.testing.assert_allclose(circle_geometry.frame.normal, (0.0, 0.0, 1.0))
        self.assertEqual(circle_geometry.curve.center_coordinates, (0.25, -0.5))
        self.assertEqual(circle_geometry.curve.center, (1.25, 1.5, 3.0))
        self.assertEqual(circle_geometry.curve.radius, 1.5)
        self.assertEqual(picture.objects[0].style.draw_color, "#FF0000")
        self.assertEqual(picture.objects[0].style.line_width_pt, 1.2)

        ellipse_geometry = restore_planar_curve_geometry(
            picture.objects[1].geometry,
            expected_curve_id="ellipse-a",
        )
        self.assertIsInstance(ellipse_geometry.curve, Ellipse3DSpec)
        self.assertEqual(ellipse_geometry.curve.center_coordinates, (-1.0, 2.0))
        self.assertEqual(ellipse_geometry.curve.center, (0.0, 4.0, 3.0))
        self.assertEqual(ellipse_geometry.curve.semi_u, 3.0)
        self.assertEqual(ellipse_geometry.curve.semi_v, 1.0)
        self.assertEqual(picture.objects[1].style.draw_color, "#0000FF")
        self.assertIsNone(picture.objects[1].style.dash_pattern_pt)

    def test_space_plane_requires_a_three_dimensional_picture(self) -> None:
        picture = self.compile_picture(
            r"""
  \coordinate (O) at (0,0);
  \coordinate (U) at (1,0);
  \coordinate (V) at (0,1);
  \DeclareSpacePlane{base-plane}{O/U/V};
""",
            options="scale=1",
        )

        self.assertFalse(picture.planar_frames_3d)
        self.assert_has_unsupported(picture, "three-dimensional")

    def test_space_plane_rejects_unknown_repeated_and_collinear_points(self) -> None:
        cases = (
            ("O/U/MISSING", "unknown"),
            ("O/U/U", "distinct"),
            ("O/U/W", "collinear"),
        )
        for point_names, expected_error in cases:
            with self.subTest(point_names=point_names):
                picture = self.compile_picture(
                    rf"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \coordinate (W) at (2,0,0);
  \DeclareSpacePlane{{base-plane}}{{{point_names}}};
"""
                )
                self.assertFalse(picture.planar_frames_3d)
                self.assert_has_unsupported(picture, expected_error)

    def test_space_plane_and_curve_reject_empty_delimited_fields(self) -> None:
        for point_names in ("/U/V", "O//V", "O/U/", "O/U//V"):
            with self.subTest(point_names=point_names):
                picture = self.compile_picture(
                    rf"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{{base-plane}}{{{point_names}}};
"""
                )
                self.assertFalse(picture.planar_frames_3d)
                self.assert_has_unsupported(picture, "three", "non-empty")

        for center in (",0", "0,", "0,,1", ",0,1"):
            with self.subTest(center=center):
                picture = self.compile_picture(
                    rf"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{{base-plane}}{{O/U/V}};
  \DrawSpaceCircle{{circle-a}}{{base-plane}}{{{center}}}{{1}};
"""
                )
                self.assertFalse(picture.objects)
                self.assert_has_unsupported(picture, "two", "comma-separated")

    def test_space_plane_identifier_must_be_unique(self) -> None:
        picture = self.compile_picture(
            r"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{base-plane}{O/U/V};
  \DeclareSpacePlane{base-plane}{O/U/V};
"""
        )

        self.assertEqual(set(picture.planar_frames_3d), {"base-plane"})
        self.assert_has_unsupported(picture, "duplicate", "base-plane")

    def test_space_curve_rejects_unknown_plane_and_duplicate_curve_id(self) -> None:
        unknown = self.compile_picture(
            r"""
  \coordinate (O) at (0,0,0);
  \DrawSpaceCircle{circle-a}{missing-plane}{0,0}{1};
"""
        )
        self.assertFalse(unknown.objects)
        self.assert_has_unsupported(unknown, "unknown", "missing-plane")

        duplicate = self.compile_picture(
            r"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{base-plane}{O/U/V};
  \DrawSpaceCircle{shared-id}{base-plane}{0,0}{1};
  \DrawSpaceEllipse{shared-id}{base-plane}{0,0}{2}{1};
"""
        )
        self.assertEqual([item.id for item in duplicate.objects], ["shared-id"])
        self.assert_has_unsupported(duplicate, "duplicate", "shared-id")

    def test_explicit_curve_ids_reserve_the_global_object_namespace(self) -> None:
        picture = self.compile_picture(
            r"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \coordinate (A) at (0,0,0);
  \coordinate (B) at (1,1,0);
  \DeclareSpacePlane{base-plane}{O/U/V};
  \DrawSpaceCircle{line.A.B}{base-plane}{0,0}{1};
  \DrawSpaceEllipse{line.A.B.2}{base-plane}{0,0}{2}{1};
  \draw (A) -- (B);
  \draw (A) -- (B);
"""
        )

        self.assertFalse(picture.unsupported)
        self.assertEqual(
            [item.id for item in picture.objects],
            ["line.A.B", "line.A.B.2", "line.A.B.3", "line.A.B.4"],
        )
        self.assertEqual(len({item.id for item in picture.objects}), 4)

    def test_plane_ids_also_reserve_the_global_semantic_namespace(self) -> None:
        picture = self.compile_picture(
            r"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \coordinate (A) at (0,0,0);
  \coordinate (B) at (1,1,0);
  \DeclareSpacePlane{line.A.B}{O/U/V};
  \DrawSpaceCircle{circle-a}{line.A.B}{0,0}{1};
  \draw (A) -- (B);
  \draw (A) -- (B);
"""
        )

        self.assertFalse(picture.unsupported)
        self.assertEqual(set(picture.planar_frames_3d), {"line.A.B"})
        self.assertEqual(
            [item.id for item in picture.objects],
            ["circle-a", "line.A.B.2", "line.A.B.3"],
        )

    def test_space_curves_reject_nonpositive_or_nonfinite_radii(self) -> None:
        statements = (
            (r"\DrawSpaceCircle{curve}{base-plane}{0,0}{0};", "positive"),
            (r"\DrawSpaceCircle{curve}{base-plane}{0,0}{-1};", "positive"),
            (r"\DrawSpaceCircle{curve}{base-plane}{0,0}{1e309};", "finite"),
            (r"\DrawSpaceEllipse{curve}{base-plane}{0,0}{0}{1};", "positive"),
            (r"\DrawSpaceEllipse{curve}{base-plane}{0,0}{2}{-1};", "positive"),
            (r"\DrawSpaceEllipse{curve}{base-plane}{0,0}{1e309}{1};", "finite"),
        )
        for statement, expected_error in statements:
            with self.subTest(statement=statement):
                picture = self.compile_picture(
                    rf"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{{base-plane}}{{O/U/V}};
  {statement}
"""
                )
                self.assertFalse(picture.objects)
                self.assert_has_unsupported(picture, expected_error)

    def test_space_curve_v1_is_stroke_only(self) -> None:
        unsupported_styles = ("fill=red", "dashed")
        for style in unsupported_styles:
            with self.subTest(style=style):
                picture = self.compile_picture(
                    rf"""
  \coordinate (O) at (0,0,0);
  \coordinate (U) at (1,0,0);
  \coordinate (V) at (0,1,0);
  \DeclareSpacePlane{{base-plane}}{{O/U/V}};
  \DrawSpaceCircle[{style}]{{circle-a}}{{base-plane}}{{0,0}}{{1}};
"""
                )

                self.assertFalse(picture.objects)
                self.assert_has_unsupported(picture, "solid", "stroke")


if __name__ == "__main__":
    unittest.main()
