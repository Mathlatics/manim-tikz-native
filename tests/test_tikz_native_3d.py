from __future__ import annotations

import copy
import math
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from manim import Dot3D, Line, MathTex, ParametricFunction, Polygon, VGroup

from tikz_native import compile_document
from tikz_native.manim_renderer_3d import NativeManim3DRenderer
from tikz_native.projection_3d import (
    matrix_from_tikz_basis,
    matrix_from_tikz_three_d_view,
    project_point,
    screen_delta_to_world,
)


PROVIDER_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROVIDER_ROOT / "tests" / "fixtures" / "tikz_native_3d_demo.tex"


class _ParallelCameraProbe:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=float)

    def get_projection_matrix(self) -> np.ndarray:
        return self.matrix

    def get_perspective_strength(self) -> float:
        return 0.0


def _active_lines(mobject: VGroup) -> list[Line]:
    return [
        item
        for item in mobject.get_family()
        if isinstance(item, Line) and float(item.get_stroke_opacity()) > 1e-9
    ]


def _line_points(lines: list[Line]) -> np.ndarray:
    return np.concatenate([line.get_all_points() for line in lines], axis=0)


class TikzNative3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]

    def test_demo_retains_xyz_and_exact_view_angles(self) -> None:
        picture = self.picture
        self.assertEqual(picture.dimension, 3)
        self.assertEqual(picture.coordinates["C1"], (2.0, 2.0, 2.0))
        self.assertIsNotNone(picture.projection_3d)
        projection = picture.projection_3d
        assert projection is not None
        self.assertEqual(projection.source, "3d view")
        self.assertAlmostEqual(projection.azimuth_degrees or 0.0, 40.4)
        self.assertAlmostEqual(projection.elevation_degrees or 0.0, 23.8)
        np.testing.assert_allclose(
            projection.matrix,
            matrix_from_tikz_three_d_view(40.4, 23.8),
            atol=1e-12,
        )

    def test_three_d_view_matrix_matches_tikz_basis_formula(self) -> None:
        azimuth = math.radians(40.4)
        elevation = math.radians(23.8)
        point = (1.7, -0.4, 2.2)
        actual = project_point(
            matrix_from_tikz_three_d_view(40.4, 23.8),
            point,
        )
        expected_u = point[0] * math.cos(azimuth) + point[1] * math.sin(azimuth)
        expected_v = (
            -point[0] * math.sin(azimuth) * math.sin(elevation)
            + point[1] * math.cos(azimuth) * math.sin(elevation)
            + point[2] * math.cos(elevation)
        )
        self.assertAlmostEqual(actual[0], expected_u, places=12)
        self.assertAlmostEqual(actual[1], expected_v, places=12)

    def test_projection_basis_independence_is_scale_invariant(self) -> None:
        for scale in (1.0e-20, 1.0e150):
            with self.subTest(scale=scale):
                matrix = matrix_from_tikz_basis(
                    (scale, 0.0),
                    (0.0, scale),
                    (0.0, 0.0),
                )
                np.testing.assert_allclose(
                    np.asarray(matrix)[:2],
                    np.array(
                        [
                            [scale, 0.0, 0.0],
                            [0.0, scale, 0.0],
                        ]
                    ),
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    matrix[2],
                    (0.0, 0.0, 1.0),
                    rtol=0.0,
                    atol=1.0e-15,
                )

    def test_screen_delta_inverse_is_scale_invariant(self) -> None:
        for scale in (1.0e-20, 1.0e150):
            with self.subTest(scale=scale):
                matrix = matrix_from_tikz_basis(
                    (scale, 0.0),
                    (0.0, scale),
                    (0.0, 0.0),
                )
                displacement = screen_delta_to_world(
                    matrix,
                    2.0 * scale,
                    -3.0 * scale,
                )
                np.testing.assert_allclose(
                    displacement,
                    (2.0, -3.0, 0.0),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )

    def test_screen_delta_inverse_matches_nonorthogonal_row_span(self) -> None:
        matrix = matrix_from_tikz_basis(
            (2.0, 1.0),
            (1.0, 3.0),
            (0.5, -0.4),
        )
        first = np.asarray(matrix[0])
        second = np.asarray(matrix[1])
        expected = 0.7 * first - 0.2 * second
        displacement = screen_delta_to_world(
            matrix,
            float(np.dot(first, expected)),
            float(np.dot(second, expected)),
        )
        np.testing.assert_allclose(
            displacement,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_nearly_parallel_projection_basis_is_rejected_relatively(self) -> None:
        with self.assertRaisesRegex(ValueError, "线性相关"):
            matrix_from_tikz_basis(
                (1.0, 1.0),
                (0.0, 1.0e-7),
                (0.0, 0.0),
            )

    def test_nonfinite_projection_basis_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "有限非零向量"):
            matrix_from_tikz_basis(
                (math.nan, 0.0),
                (0.0, 1.0),
                (0.0, 0.0),
            )

    def test_demo_compiles_to_native_semantic_inventory(self) -> None:
        self.assertFalse(self.picture.unsupported)
        self.assertEqual(
            Counter(item.kind for item in self.picture.objects),
            Counter(
                {
                    "polygon": 3,
                    "line": 12,
                    "arrow": 3,
                    "dot": 8,
                    "label": 11,
                }
            ),
        )
        figure = NativeManim3DRenderer().render(self.picture)
        self.assertEqual(len(figure.objects), 37)
        self.assertEqual(
            sum(isinstance(item, Dot3D) for item in figure.objects.values()),
            8,
        )
        self.assertEqual(
            sum(isinstance(item, Polygon) for item in figure.objects.values()),
            3,
        )
        self.assertEqual(
            sum(isinstance(item, MathTex) for item in figure.objects.values()),
            11,
        )

    def test_three_d_label_size_is_independent_of_picture_scale(self) -> None:
        renderer = NativeManim3DRenderer()
        original = renderer.render(self.picture)
        enlarged_picture = copy.deepcopy(self.picture)
        enlarged_picture.scale *= 1.8
        enlarged = renderer.render(enlarged_picture)
        original_label = next(
            original.objects[item.id]
            for item in self.picture.objects
            if item.kind == "label"
        )
        enlarged_label = next(
            enlarged.objects[item.id]
            for item in enlarged_picture.objects
            if item.kind == "label"
        )
        self.assertAlmostEqual(original_label.width, enlarged_label.width, places=12)
        self.assertAlmostEqual(original_label.height, enlarged_label.height, places=12)

    def test_transform_shape_label_explicitly_follows_picture_scale(self) -> None:
        source = r"""
\begin{tikzpicture}[scale=0.5,space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (A) at (0,0,0);
  \node[transform shape,rectangle,draw=black] at (A) {$A$};
\end{tikzpicture}
\begin{tikzpicture}[scale=1,space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (A) at (0,0,0);
  \node[transform shape,rectangle,draw=black] at (A) {$A$};
\end{tikzpicture}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "transform-shape.tex"
            path.write_text(source, encoding="utf-8")
            pictures = compile_document(path).pictures
        renderer = NativeManim3DRenderer()
        small = renderer.render(pictures[0]).fixed_orientation_labels[0]
        regular = renderer.render(pictures[1]).fixed_orientation_labels[0]
        self.assertAlmostEqual(small.width / regular.width, 0.5, places=12)
        self.assertAlmostEqual(small.height / regular.height, 0.5, places=12)

    def test_canvas_plane_transform_shape_retains_local_affine_basis(self) -> None:
        source = r"""
\begin{tikzpicture}[
  scale=0.8,
  x={(1cm,0cm)},
  y={(0.4cm,0.5cm)},
  z={(0.1cm,1cm)}
]
  \begin{scope}[canvas is yz plane at x=2]
    \node[transform shape,rectangle,draw=black] at (1,2) {$A$};
  \end{scope}
\end{tikzpicture}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "canvas-transform-shape.tex"
            path.write_text(source, encoding="utf-8")
            picture = compile_document(path).pictures[0]
        label_spec = next(item for item in picture.objects if item.kind == "label")
        self.assertEqual(label_spec.style.native_canvas_plane, "yz")
        self.assertEqual(label_spec.geometry["at"], (2.0, 1.0, 2.0))
        label = NativeManim3DRenderer().render(picture).objects[label_spec.id]
        actual = getattr(label, "_tikz_canvas_screen_matrix")
        np.testing.assert_allclose(
            actual,
            np.array([[0.32, 0.08], [0.4, 0.8]]),
            atol=1e-12,
        )

    def test_anglemark_uses_vertex_text_and_pic_text_inner_sep(self) -> None:
        source = r"""
\begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (A) at (1,0,0);
  \coordinate (O) at (0,0,0);
  \coordinate (B) at (0,1,1);
  \pic[
    anglemark={\varphi},
    angle radius=10pt,
    pic text options={inner sep=0pt,above=11pt,right=6pt}
  ] {angle=A--O--B};
\end{tikzpicture}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "anglemark.tex"
            path.write_text(source, encoding="utf-8")
            picture = compile_document(path).pictures[0]
        angle = next(item for item in picture.objects if item.kind == "angle")
        label = next(item for item in picture.objects if item.kind == "angle_label")
        self.assertEqual(angle.geometry["eccentricity"], 0.0)
        self.assertEqual(label.geometry["eccentricity"], 0.0)
        self.assertEqual(label.style.inner_xsep_pt, 0.0)
        self.assertEqual(label.style.inner_ysep_pt, 0.0)
        self.assertEqual(label.placement.anchor, "west")
        self.assertEqual(label.placement.dx_pt, 6.0)
        self.assertEqual(label.placement.dy_pt, 11.0)

    def test_three_d_angle_and_right_angle_are_native_objects(self) -> None:
        source = r"""
\begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (A) at (1,0,0);
  \coordinate (O) at (0,0,0);
  \coordinate (B) at (0,1,1);
  \pic[draw,angle radius=8pt] {angle=A--O--B};
  \pic[draw,angle radius=4pt] {right angle=A--O--B};
\end{tikzpicture}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "angles.tex"
            path.write_text(source, encoding="utf-8")
            picture = compile_document(path).pictures[0]
        self.assertFalse(picture.unsupported)
        figure = NativeManim3DRenderer().render(picture)
        angle = next(
            figure.objects[item.id] for item in picture.objects if item.kind == "angle"
        )
        right_angle = next(
            figure.objects[item.id]
            for item in picture.objects
            if item.kind == "right_angle"
        )
        self.assertIsInstance(angle, ParametricFunction)
        self.assertIsInstance(right_angle, VGroup)
        self.assertEqual(len(right_angle), 2)

    def test_explicit_xyz_basis_becomes_camera_matrix(self) -> None:
        source = r"""
\documentclass{standalone}
\begin{document}
\begin{tikzpicture}[
  x={(-0.3cm,-0.4cm)},
  y={(1cm,0cm)},
  z={(0cm,1cm)}
]
  \coordinate (A) at (0,0,0);
  \coordinate (B) at (1,2,3);
  \draw (A)--(B);
\end{tikzpicture}
\end{document}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "basis.tex"
            path.write_text(source, encoding="utf-8")
            picture = compile_document(path).pictures[0]
        self.assertFalse(picture.unsupported)
        self.assertEqual(picture.dimension, 3)
        projection = picture.projection_3d
        assert projection is not None
        self.assertEqual(projection.source, "basis")
        np.testing.assert_allclose(
            np.asarray(projection.matrix)[:2],
            np.array(
                [
                    [-0.3, 1.0, 0.0],
                    [-0.4, 0.0, 1.0],
                ]
            ),
            atol=1e-12,
        )

    def test_macro_entry_materializes_space_view_and_point_helpers(self) -> None:
        source = r"""
\newcommand{\BaseFigure}[2]{%
  \begin{tikzpicture}[
    scale=#1,
    space view={(-0.35,-0.35),(1,0),(0,1)},
    edge/.style={blue!70!black,line width=1pt},
    pt/.style={circle,fill=black,inner sep=0.8pt}
  ]
    \defPoint{A}{0}{0}{0}
    \defPoint{B}{2*cos(60)}{2*sin(30)}{3}
    \pointOnSpaceLine{M}{A}{B}{0.5}
    \draw[edge] (A)--(B);
    \node[pt] at (M) {};
    #2
  \end{tikzpicture}
}
\newcommand{\ConcreteFigure}{%
  \BaseFigure{0.85}{\node[above=2pt] at (B) {$B$};}%
}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "macro-entry.tex"
            path.write_text(source, encoding="utf-8")
            document = compile_document(path, entry_macro="ConcreteFigure")
        self.assertEqual(document.entry_macro, "ConcreteFigure")
        self.assertEqual(len(document.pictures), 1)
        picture = document.pictures[0]
        self.assertFalse(picture.unsupported)
        self.assertAlmostEqual(picture.scale, 0.85)
        np.testing.assert_allclose(picture.coordinates["B"], (1.0, 1.0, 3.0))
        np.testing.assert_allclose(picture.coordinates["M"], (0.5, 0.5, 1.5))
        self.assertIsNotNone(picture.projection_3d)
        assert picture.projection_3d is not None
        self.assertEqual(picture.projection_3d.source, "space view")
        self.assertEqual(
            Counter(item.kind for item in picture.objects),
            Counter({"line": 1, "dot": 1, "label": 1}),
        )

    def test_entry_preludes_feed_view_macros_to_each_picture(self) -> None:
        source = r"""
\newcommand{\BaseFigure}[1]{%
  \pgfmathsetmacro{\viewx}{cos(#1)}
  \begin{tikzpicture}[x={(\viewx cm,0cm)},y={(0cm,1cm)},z={(0cm,0cm)}]
    \coordinate (A) at (0,0,0);
    \coordinate (B) at (1,0,0);
    \draw (A)--(B);
  \end{tikzpicture}
}
\newcommand{\CompareFigure}{\BaseFigure{0}\BaseFigure{60}}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "macro-prelude.tex"
            path.write_text(source, encoding="utf-8")
            document = compile_document(path, entry_macro="CompareFigure")
        self.assertEqual(len(document.pictures), 2)
        self.assertFalse(document.pictures[0].unsupported)
        self.assertFalse(document.pictures[1].unsupported)
        first = document.pictures[0].projection_3d
        second = document.pictures[1].projection_3d
        assert first is not None and second is not None
        self.assertAlmostEqual(first.x_basis_cm[0], 1.0)
        self.assertAlmostEqual(second.x_basis_cm[0], 0.5)

    def test_native_common_tikz_orderings_and_filldraw(self) -> None:
        source = r"""
\newcommand{\ConcreteFigure}{%
  \begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
    \coordinate (A) at (0,0,0);
    \coordinate (B) at (2,0,0);
    \coordinate (C) at (0,2,0);
    \coordinate (H) at (0.5,0.5,0);
    \filldraw[fill=gray!10,draw=black] (A)--(B)--(C)--cycle;
    \fill (H) circle (0.5pt) node[above=2pt] {$H$};
    \node at (A) [right=2pt] {$A$};
  \end{tikzpicture}
}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "common-orderings.tex"
            path.write_text(source, encoding="utf-8")
            picture = compile_document(
                path,
                entry_macro="ConcreteFigure",
            ).pictures[0]
        self.assertFalse(picture.unsupported)
        self.assertEqual(
            Counter(item.kind for item in picture.objects),
            Counter({"polygon": 1, "dot": 1, "label": 2}),
        )

    def test_disconnected_subpaths_are_never_bridged(self) -> None:
        source = r"""
\begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (A) at (0,0,0);
  \coordinate (B) at (1,0,0);
  \coordinate (C) at (0,1,0);
  \coordinate (D) at (1,1,0);
  \draw (A)--(B) (C)--(D);
\end{tikzpicture}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "disconnected.tex"
            path.write_text(source, encoding="utf-8")
            picture = compile_document(path).pictures[0]
        self.assertFalse(picture.unsupported)
        lines = [item for item in picture.objects if item.kind == "line"]
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            [(item.geometry["start_name"], item.geometry["end_name"]) for item in lines],
            [("A", "B"), ("C", "D")],
        )

    def test_unknown_semantic_prefix_is_not_silently_skipped(self) -> None:
        source = r"""
\begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (A) at (0,0,0);
  \coordinate (B) at (1,0,0);
  \CustomOcclusion{A}{B}
  \draw (A)--(B);
\end{tikzpicture}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "semantic-prefix.tex"
            path.write_text(source, encoding="utf-8")
            picture = compile_document(path).pictures[0]
        self.assertTrue(picture.unsupported)
        self.assertIn("statement contains no supported TikZ command", picture.unsupported[0])
        self.assertFalse([item for item in picture.objects if item.kind == "line"])

    @staticmethod
    def _occlusion_source(*, side_view: bool = False) -> str:
        basis = (
            "x={(0cm,0cm)},y={(1cm,0cm)},z={(0cm,1cm)}"
            if side_view
            else "x={(1cm,0cm)},y={(0cm,1cm)},z={(0cm,0cm)}"
        )
        return rf"""
\begin{{tikzpicture}}[
  {basis},
  edge/.style={{black,thick}},
  hidden/.style={{black,densely dashed,thin}}
]
  \coordinate (S) at (-2,0,-1);
  \coordinate (E) at (2,0,-1);
  \coordinate (A) at (-1,-1,0);
  \coordinate (B) at (1,-1,0);
  \coordinate (C) at (1,1,0);
  \coordinate (D) at (-1,1,0);
  \DrawSpaceLineBehindParallelogramFace[edge][hidden]
    {{S}}{{E}}{{A}}{{B}}{{C}}{{D}}
\end{{tikzpicture}}
"""

    def test_dynamic_occlusion_matches_authored_static_split(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic-occlusion-static.tex"
            path.write_text(self._occlusion_source(), encoding="utf-8")
            picture = compile_document(path).pictures[0]

        relation = picture.occlusion_relations[0]
        self.assertEqual(len(picture.occlusion_relations), 1)
        self.assertEqual(
            [
                next(item for item in picture.objects if item.id == object_id)
                .geometry["visibility"]
                for object_id in relation.object_ids
            ],
            ["visible", "hidden", "visible"],
        )
        renderer = NativeManim3DRenderer()
        reference = renderer.render(picture)
        expected = VGroup(
            *(reference.objects[object_id] for object_id in relation.object_ids)
        )
        dynamic = renderer.render(picture)
        camera = _ParallelCameraProbe(np.identity(3))
        groups = renderer.bind_occlusions_to_camera(dynamic, camera)
        actual = groups[relation.id]
        stable_line_ids = {
            id(item) for item in actual.get_family() if isinstance(item, Line)
        }
        expected_lines = _active_lines(expected)
        actual_lines = _active_lines(actual)

        np.testing.assert_allclose(
            _line_points(actual_lines),
            _line_points(expected_lines),
            atol=1e-12,
        )
        self.assertEqual(len(actual_lines), len(expected_lines))

        camera.matrix = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        )
        actual.update(0.0)
        self.assertEqual(len(_active_lines(actual)), 1)
        self.assertEqual(
            {
                id(item)
                for item in actual.get_family()
                if isinstance(item, Line)
            },
            stable_line_ids,
        )

    def test_dynamic_occlusion_can_create_a_later_hidden_piece(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic-occlusion-new-piece.tex"
            path.write_text(
                self._occlusion_source(side_view=True),
                encoding="utf-8",
            )
            picture = compile_document(path).pictures[0]

        relation = picture.occlusion_relations[0]
        self.assertEqual(len(relation.object_ids), 1)
        self.assertIsNotNone(relation.hidden_style.dash_pattern_pt)
        renderer = NativeManim3DRenderer()
        figure = renderer.render(picture)
        side = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        )
        camera = _ParallelCameraProbe(side)
        actual = renderer.bind_occlusions_to_camera(figure, camera)[relation.id]
        stable_line_ids = {
            id(item) for item in actual.get_family() if isinstance(item, Line)
        }
        self.assertEqual(len(_active_lines(actual)), 1)

        camera.matrix = np.identity(3)
        actual.update(0.0)
        self.assertGreater(len(_active_lines(actual)), 3)
        self.assertEqual(
            {
                id(item)
                for item in actual.get_family()
                if isinstance(item, Line)
            },
            stable_line_ids,
        )


if __name__ == "__main__":
    unittest.main()
