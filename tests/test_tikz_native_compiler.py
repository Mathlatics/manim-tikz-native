from __future__ import annotations

import ast
import copy
import json
import unittest
from collections import Counter
from math import atan, atan2, cos, radians, sin, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

from manim import AnimationGroup, Create, FadeIn, GrowFromCenter, RightAngle, Write

from tikz_native import compile_document
from tikz_native.animation import (
    SEMANTIC_LAYER_ORDER,
    native_reveal_animation,
    semantic_animation_layers,
    semantic_layer_name,
)
from tikz_native.compiler import StyleSpec, TikzNativeError
from tikz_native.dynamic_geometry import (
    EllipseChordDriver,
    ellipse_chord_state,
    project_point_to_line,
)
from tikz_native.manim_renderer import (
    DEFAULT_TEX_TEMPLATE,
    MANIM_FONT_SIZE_PER_TEX_CM,
    NativeManimRenderer,
)
from tikz_native.regression import build_semantic_snapshot


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "fixtures" / "national_2026_18_tikz.tex"


class TikzNativeCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = compile_document(SOURCE)

    def test_all_sixteen_pictures_compile_without_fallback(self) -> None:
        self.assertEqual(len(self.document.pictures), 16)
        self.assertEqual(
            sum(len(picture.objects) for picture in self.document.pictures), 262
        )
        self.assertFalse(
            [
                (picture.index, item)
                for picture in self.document.pictures
                for item in picture.unsupported
            ]
        )

    def test_expected_semantic_inventory(self) -> None:
        counts = Counter(
            item.kind
            for picture in self.document.pictures
            for item in picture.objects
        )
        self.assertEqual(counts["label"], 85)
        self.assertEqual(counts["path_label"], 10)
        self.assertEqual(counts["angle"], 9)
        self.assertEqual(counts["angle_label"], 9)
        self.assertEqual(counts["right_angle"], 5)
        self.assertEqual(counts["dot"], 45)

    def test_animation_layers_cover_every_object_exactly_once(self) -> None:
        for picture in self.document.pictures:
            layers = semantic_animation_layers(picture, include_empty=True)
            self.assertEqual(tuple(layer.name for layer in layers), SEMANTIC_LAYER_ORDER)
            layered_ids = [
                object_id for layer in layers for object_id in layer.object_ids
            ]
            source_ids = [spec.id for spec in picture.objects]
            self.assertCountEqual(layered_ids, source_ids)
            self.assertEqual(len(layered_ids), len(set(layered_ids)))
            for spec in picture.objects:
                containing_layer = next(
                    layer.name for layer in layers if spec.id in layer.object_ids
                )
                self.assertEqual(containing_layer, semantic_layer_name(spec))

    def test_figure_four_construction_ids_are_stable(self) -> None:
        ids = {spec.id for spec in self.document.pictures[3].objects}
        expected = {
            "fill.P.Q.R",
            "arrow",
            "label.x",
            "arrow.2",
            "label.y",
            "ellipse",
            "line",
            "label.l",
            "line.P.Q",
            "line.Q.R",
            "line.R.P",
            "label_path.P.Q.PQ",
            "line.R.H",
            "label_path.R.H.d",
            "right_angle.R.H.P",
            "dot.P",
            "dot.Q",
            "dot.R",
            "label.P.P",
            "label.Q.Q",
            "label.R.R",
        }
        self.assertEqual(ids, expected)

    def test_lines_points_labels_and_dashes_have_native_animations(self) -> None:
        picture = self.document.pictures[3]
        renderer = NativeManimRenderer(scene_unit_per_cm=0.7)
        figure = renderer.render(picture)
        specs = {spec.id: spec for spec in picture.objects}

        self.assertIsInstance(
            native_reveal_animation(
                specs["line.P.Q"], figure.objects["line.P.Q"]
            ),
            Create,
        )
        self.assertIsInstance(
            native_reveal_animation(specs["dot.P"], figure.objects["dot.P"]),
            GrowFromCenter,
        )
        self.assertIsInstance(
            native_reveal_animation(
                specs["label.P.P"], figure.objects["label.P.P"]
            ),
            Write,
        )
        self.assertIsInstance(
            native_reveal_animation(
                specs["fill.P.Q.R"], figure.objects["fill.P.Q.R"]
            ),
            FadeIn,
        )
        dashed = native_reveal_animation(
            specs["line.R.H"], figure.objects["line.R.H"]
        )
        self.assertIsInstance(dashed, AnimationGroup)
        self.assertEqual(
            len(dashed.animations),
            len(figure.objects["line.R.H"]),
        )

    def test_shared_ellipse_coordinates_are_evaluated(self) -> None:
        coordinates = self.document.pictures[0].coordinates
        self.assertAlmostEqual(coordinates["P"][0], 1.0, places=12)
        self.assertAlmostEqual(coordinates["P"][1], 1.5, places=12)
        self.assertAlmostEqual(coordinates["Q"][0], -13 / 7, places=12)
        self.assertAlmostEqual(coordinates["Q"][1], -9 / 14, places=12)
        self.assertAlmostEqual(coordinates["R"][0], -1.0, places=12)
        self.assertAlmostEqual(coordinates["R"][1], -1.5, places=12)

    def test_figure_one_retains_native_named_intersection_semantics(self) -> None:
        picture = self.document.pictures[0]
        self.assertEqual(set(picture.named_paths), {"analysisfigC", "analysisfigL"})
        self.assertEqual(picture.named_paths["analysisfigC"].kind, "ellipse")
        self.assertEqual(picture.named_paths["analysisfigL"].kind, "line")
        self.assertEqual(len(picture.intersections), 1)
        relation = picture.intersections[0]
        self.assertEqual(relation.path_a, "analysisfigC")
        self.assertEqual(relation.path_b, "analysisfigL")
        self.assertEqual(relation.sort_by, "analysisfigL")
        self.assertEqual(tuple(relation.coordinate_names), ("Q", "P"))
        self.assertLess(relation.sort_parameters[0], relation.sort_parameters[1])
        self.assertEqual(
            picture.coordinate_dependencies["Q"]["sorted_index"],
            0,
        )
        self.assertEqual(
            picture.coordinate_dependencies["P"]["sorted_index"],
            1,
        )
        self.assertEqual(
            picture.coordinate_dependencies["R"]["operation"],
            "interpolation",
        )

    def test_native_intersection_order_follows_oriented_line_direction(self) -> None:
        source_text = r"""
\begin{tikzpicture}
  \coordinate (O) at (0,0);
  \path[name path=C] (O) ellipse [x radius=2,y radius=1];
  \path[name path=forward] (-3,0) -- (3,0);
  \path[name intersections={of=C and forward,sort by=forward,by={A,B}}];
\end{tikzpicture}
\begin{tikzpicture}
  \coordinate (O) at (0,0);
  \path[name path=C] (O) ellipse [x radius=2,y radius=1];
  \path[name path=reverse] (3,0) -- (-3,0);
  \path[name intersections={of=C and reverse,sort by=reverse,by={A,B}}];
\end{tikzpicture}
"""
        with TemporaryDirectory() as directory:
            source = Path(directory) / "oriented_intersections.tex"
            source.write_text(source_text, encoding="utf-8")
            document = compile_document(source)
        self.assertEqual(document.pictures[0].coordinates["A"], (-2.0, 0.0))
        self.assertEqual(document.pictures[0].coordinates["B"], (2.0, 0.0))
        self.assertEqual(document.pictures[1].coordinates["A"], (2.0, 0.0))
        self.assertEqual(document.pictures[1].coordinates["B"], (-2.0, 0.0))

    def test_dynamic_driver_is_built_from_the_native_tikz_relation(self) -> None:
        picture = self.document.pictures[0]
        relation = picture.intersections[0]
        line = picture.named_paths[relation.sort_by]
        direction = (
            line.geometry["end"][0] - line.geometry["start"][0],
            line.geometry["end"][1] - line.geometry["start"][1],
        )
        angle = atan2(direction[1], direction[0])
        driver = EllipseChordDriver.from_named_intersection(
            lambda: angle,
            picture,
            pivot_name="F",
        )
        state = driver.state()
        for name in ("P", "Q", "R"):
            expected = picture.coordinates[name]
            actual = getattr(state, name.lower())
            self.assertAlmostEqual(actual[0], expected[0], places=12)
            self.assertAlmostEqual(actual[1], expected[1], places=12)

    def test_source_driven_intersection_identity_stays_ordered_during_motion(self) -> None:
        picture = self.document.pictures[0]
        current_angle = [atan(3 / 4)]
        driver = EllipseChordDriver.from_named_intersection(
            lambda: current_angle[0],
            picture,
            pivot_name="F",
        )
        for degrees in (22, 36.86989764584402, 54, 89, 140):
            current_angle[0] = radians(degrees)
            state = driver.state()
            direction = (cos(current_angle[0]), sin(current_angle[0]))

            def signed_parameter(point):
                return (
                    (point[0] - state.focus[0]) * direction[0]
                    + (point[1] - state.focus[1]) * direction[1]
                )

            self.assertLess(signed_parameter(state.q), 0)
            self.assertGreater(signed_parameter(state.p), 0)

    def test_projection_is_perpendicular(self) -> None:
        coordinates = self.document.pictures[3].coordinates
        p, r, h = coordinates["P"], coordinates["R"], coordinates["H"]
        q = coordinates["Q"]
        rh = (h[0] - r[0], h[1] - r[1])
        pq = (q[0] - p[0], q[1] - p[1])
        self.assertAlmostEqual(rh[0] * pq[0] + rh[1] * pq[1], 0.0, places=12)

    def test_figure_one_driven_state_matches_the_original_tikz_state(self) -> None:
        state = ellipse_chord_state(
            atan(3 / 4),
            semi_major=2.0,
            semi_minor=sqrt(3),
            focus=(-1.0, 0.0),
            backward_length=1.5625,
            forward_length=3.1875,
        )
        coordinates = self.document.pictures[0].coordinates
        for name in ("P", "Q", "R"):
            actual = getattr(state, name.lower())
            self.assertAlmostEqual(actual[0], coordinates[name][0], places=12)
            self.assertAlmostEqual(actual[1], coordinates[name][1], places=12)
        line = next(
            spec for spec in self.document.pictures[0].objects if spec.id == "line"
        )
        for actual, key in (
            (state.line_start, "start"),
            (state.line_end, "end"),
        ):
            self.assertAlmostEqual(actual[0], line.geometry[key][0], places=12)
            self.assertAlmostEqual(actual[1], line.geometry[key][1], places=12)

    def test_driven_points_remain_on_their_geometric_constraints(self) -> None:
        for degrees in (22, 36.86989764584402, 58):
            angle = radians(degrees)
            state = ellipse_chord_state(
                angle,
                semi_major=2.0,
                semi_minor=sqrt(3),
                focus=(-1.0, 0.0),
                backward_length=1.5625,
                forward_length=3.1875,
            )
            for point in (state.p, state.q, state.r):
                self.assertAlmostEqual(
                    point[0] ** 2 / 4 + point[1] ** 2 / 3,
                    1.0,
                    places=12,
                )
            direction = (cos(angle), sin(angle))
            for point in (state.p, state.q):
                relative = (
                    point[0] - state.focus[0],
                    point[1] - state.focus[1],
                )
                self.assertAlmostEqual(
                    relative[0] * direction[1] - relative[1] * direction[0],
                    0.0,
                    places=12,
                )
            self.assertAlmostEqual(state.r[0], -state.p[0], places=12)
            self.assertAlmostEqual(state.r[1], -state.p[1], places=12)

    def test_dynamic_projection_remains_perpendicular_to_the_chord(self) -> None:
        for degrees in (22, 36.86989764584402, 54):
            state = ellipse_chord_state(
                radians(degrees),
                semi_major=2.0,
                semi_minor=sqrt(3),
                focus=(-1.0, 0.0),
                backward_length=1.5625,
                forward_length=3.1875,
            )
            foot = project_point_to_line(state.r, state.p, state.q)
            chord = (state.q[0] - state.p[0], state.q[1] - state.p[1])
            projection = (foot[0] - state.r[0], foot[1] - state.r[1])
            self.assertAlmostEqual(
                chord[0] * projection[0] + chord[1] * projection[1],
                0.0,
                places=12,
            )

    def test_invalid_pgf_arithmetic_fails_closed(self) -> None:
        invalid_expressions = (
            "1 / 0",
            "sqrt(-1)",
            "10^10000",
            "1e309",
        )
        for expression in invalid_expressions:
            source_text = rf"""
\begin{{tikzpicture}}
  \pgfmathsetmacro{{\bad}}{{{expression}}}
  \coordinate (A) at (\bad,0);
  \fill (A) circle (1pt);
\end{{tikzpicture}}
"""
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(TikzNativeError, "PGF math"):
                    compile_document(source_text=source_text)

    def test_finite_pgf_arithmetic_remains_supported(self) -> None:
        document = compile_document(
            source_text=r"""
\begin{tikzpicture}
  \pgfmathsetmacro{\value}{1e150 * 1e-150 + sqrt(4)}
  \coordinate (A) at (\value,0);
  \fill (A) circle (1pt);
\end{tikzpicture}
"""
        )
        self.assertEqual(document.pictures[0].coordinates["A"], (3.0, 0.0))

    def test_xcolor_mix_is_not_opacity(self) -> None:
        polygon = self.document.pictures[0].objects[0]
        self.assertEqual(polygon.style.fill_color, "#DEECEB")
        self.assertIsNone(polygon.style.draw_color)
        self.assertAlmostEqual(polygon.style.opacity, 1.0)
        gold = self.document.pictures[0].objects[1]
        self.assertEqual(gold.style.fill_color, "#EADBB6")
        self.assertAlmostEqual(gold.style.opacity, 0.72)

    def test_inline_node_between_coordinates_defaults_to_segment_midpoint(self) -> None:
        picture = self.document.pictures[13]
        horizontal = next(
            item
            for item in picture.objects
            if item.id == "label_path.A.C.Delta_x_a_Delta_X"
        )
        vertical = next(
            item
            for item in picture.objects
            if item.id == "label_path.C.B.Delta_y_b_Delta_Y"
        )
        self.assertEqual(horizontal.kind, "path_label")
        self.assertEqual(horizontal.geometry["pos"], 0.5)
        self.assertEqual(vertical.geometry["pos"], 0.5)
        self.assertFalse(vertical.placement.sloped)

        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        figure = renderer.render(picture)
        rendered_vertical = figure.objects[vertical.id]
        self.assertGreater(rendered_vertical.width, rendered_vertical.height)

    def test_renderer_uses_native_right_angle(self) -> None:
        renderer = NativeManimRenderer(scene_unit_per_cm=0.7)
        figure = renderer.render(self.document.pictures[3])
        marker = figure.objects["right_angle.R.H.P"]
        self.assertIsInstance(marker, RightAngle)

    def test_font_uses_11pt_document_and_is_independent_of_tikz_scale(self) -> None:
        self.assertIn("11pt", DEFAULT_TEX_TEMPLATE.documentclass)
        renderer = NativeManimRenderer(scene_unit_per_cm=0.56)
        self.assertAlmostEqual(
            renderer.base_font_size,
            0.56 * MANIM_FONT_SIZE_PER_TEX_CM,
            places=12,
        )

        small_picture = copy.deepcopy(self.document.pictures[0])
        large_picture = copy.deepcopy(self.document.pictures[0])
        small_picture.scale = 0.8
        large_picture.scale = 1.5
        small_figure = renderer.render(small_picture)
        large_figure = renderer.render(large_picture)

        small_label = small_figure.objects["label.P.P"]
        large_label = large_figure.objects["label.P.P"]
        self.assertAlmostEqual(small_label.width, large_label.width, places=12)
        self.assertAlmostEqual(small_label.height, large_label.height, places=12)
        self.assertAlmostEqual(
            large_figure.objects["ellipse"].width
            / small_figure.objects["ellipse"].width,
            1.5 / 0.8,
            places=12,
        )

    def test_small_uses_the_11pt_latex_size_table(self) -> None:
        renderer = NativeManimRenderer(scene_unit_per_cm=0.56)
        normal = renderer._make_label("$P$", StyleSpec())
        small = renderer._make_label(r"{\small $P$}", StyleSpec())
        self.assertAlmostEqual(small.height / normal.height, 10 / 10.95, places=6)

    def test_every_label_formula_explicitly_uses_displaystyle(self) -> None:
        renderer = NativeManimRenderer(scene_unit_per_cm=0.56)
        missing = []
        for picture in self.document.pictures:
            figure = renderer.render(picture)
            for spec in picture.objects:
                if not spec.label or "$" not in spec.label:
                    continue
                tex_string = getattr(figure.objects[spec.id], "tex_string", "")
                if r"\displaystyle" not in tex_string:
                    missing.append((picture.index, spec.id, tex_string))
        self.assertEqual(missing, [])

        mixed = renderer._make_label(r"{\small $x$正方向}", StyleSpec())
        self.assertIn(r"$\displaystyle x$正方向", mixed.tex_string)
        already_display = renderer._make_label(r"$\displaystyle x$", StyleSpec())
        self.assertEqual(already_display.tex_string.count(r"\displaystyle"), 1)

    def test_white_outline_measures_directional_nested_tikz_node(self) -> None:
        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        outline_style = StyleSpec(raw_options=["above left=-1pt"])
        outlined = renderer._make_label(
            r"$\mathWhiteOutline[1.0]{\beta}$",
            outline_style,
        )
        metric_box, visible_formula = outlined
        offset = metric_box.get_center() - visible_formula.get_center()

        # In TikZ, the directional shift reaches the macro's nested
        # tikzpicture: the outer node grows to about 18.55 x 23.49 pt and the
        # black beta sits right/up of its geometric center.  A plain TeX hbox
        # cannot reproduce this relationship.
        self.assertGreater(offset[0] / renderer.pt, 1.0)
        self.assertLess(offset[0] / renderer.pt, 1.3)
        self.assertLess(offset[1] / renderer.pt, -1.8)
        self.assertGreater(offset[1] / renderer.pt, -2.2)
        self.assertGreater(outlined.width / renderer.pt, 18.4)
        self.assertLess(outlined.width / renderer.pt, 18.7)
        self.assertGreater(outlined.height / renderer.pt, 23.3)
        self.assertLess(outlined.height / renderer.pt, 23.7)
        self.assertAlmostEqual(outlined.width, metric_box.width, places=12)
        self.assertAlmostEqual(outlined.height, metric_box.height, places=12)
        self.assertEqual(
            renderer._outer_node_padding(outlined, outline_style),
            (0.0, 0.0),
        )

        colored_parent = renderer._make_label(
            r"$\mathWhiteOutline[1.0]{m'}$",
            StyleSpec(
                draw_color="#0000FF",
                raw_options=["above=2pt", "right=-7pt"],
            ),
        )
        _, foreground = colored_parent
        foreground_colors = {
            str(item.get_fill_color())
            for item in foreground.get_family()
            if hasattr(item, "get_num_points") and item.get_num_points()
        }
        self.assertEqual(foreground_colors, {"#000000"})

    def test_white_outline_uses_real_tikz_default_inner_sep(self) -> None:
        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        ordinary = renderer._make_label("$P$", StyleSpec())
        outlined = renderer._make_label(
            r"$\mathWhiteOutline[1.0]{P}$",
            StyleSpec(),
        )

        ordinary_pad = renderer._node_padding(ordinary, StyleSpec())
        outlined_pad = renderer._node_padding(outlined, StyleSpec())
        explicit_zero = renderer._node_padding(
            outlined,
            StyleSpec(inner_xsep_pt=0.0, inner_ysep_pt=0.0),
        )

        self.assertAlmostEqual(ordinary_pad[0] / renderer.pt, 4.7)
        self.assertAlmostEqual(ordinary_pad[1] / renderer.pt, 3.55)
        self.assertAlmostEqual(outlined_pad[0] / renderer.pt, 3.6496)
        self.assertAlmostEqual(outlined_pad[1] / renderer.pt, 3.6496)
        self.assertEqual(explicit_zero, (0.0, 0.0))

    def test_geometry_strokes_are_thicker_without_changing_label_halo(self) -> None:
        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        one_pt = StyleSpec(line_width_pt=1.0)

        self.assertAlmostEqual(renderer._stroke_width(one_pt), 3.8)
        self.assertAlmostEqual(renderer.label_outline_stroke_width_per_pt, 2.15)

    def test_gallery_does_not_rescale_individual_content_groups(self) -> None:
        scene_path = ROOT / "examples" / "national_2026_18_native.py"
        tree = ast.parse(scene_path.read_text(encoding="utf-8"))
        gallery_names = {
            "National2026TikzNativeRepresentativeGallery",
            "National2026TikzNativeAllGallery",
            "National2026TikzNativeAllGalleryReveal",
        }
        forbidden = {"scale", "scale_to_fit_width", "scale_to_fit_height"}
        hits = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name not in gallery_names:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if child.func.attr in forbidden:
                        hits.append((node.name, child.lineno, child.func.attr))
        self.assertEqual(hits, [])

    def test_no_explicit_generic_path_or_image_fallback(self) -> None:
        forbidden = {"VMobject", "SVGMobject", "ImageMobject", "set_points_as_corners"}
        paths = [
            ROOT / "tikz_native" / "manim_renderer.py",
            ROOT / "tikz_native" / "animation.py",
            ROOT / "tikz_native" / "dynamic_geometry.py",
            ROOT / "examples" / "national_2026_18_native.py",
        ]
        hits = []
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                else:
                    continue
                if name in forbidden:
                    hits.append((path.name, node.lineno, name))
        self.assertEqual(hits, [])

    def test_documented_native_friendly_example_stays_convertible(self) -> None:
        example = ROOT / "tikz_native" / "examples" / "native_friendly_figure.tex"
        document = compile_document(example)
        self.assertEqual(len(document.pictures), 1)
        self.assertFalse(document.pictures[0].unsupported)
        self.assertEqual(len(document.pictures[0].objects), 15)

    def test_v0_1_semantic_baseline_is_frozen(self) -> None:
        baseline_path = (
            ROOT
            / "tests"
            / "baselines"
            / "national_2026_18-v0.1.json"
        )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        self.assertEqual(build_semantic_snapshot(self.document), baseline["semantic"])


if __name__ == "__main__":
    unittest.main()
