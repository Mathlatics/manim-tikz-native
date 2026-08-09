from __future__ import annotations

from math import degrees
from pathlib import Path

import numpy as np
from manim import (
    BLUE_D,
    DOWN,
    DecimalNumber,
    FadeIn,
    Indicate,
    LEFT,
    MathTex,
    RIGHT,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    UP,
    VGroup,
    ValueTracker,
    WHITE,
)

from tikz_native import (
    NativeMotionRuntime,
    compile_document,
    ellipse_chord_metrics,
    load_motion_spec,
)
from tikz_native.manim_renderer import NativeManimRenderer


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ellipse_problem.tex"
MOTION = ROOT / "ellipse_problem.motion.json"
BACKGROUND = "#F6F8FC"
INK = "#20242A"
MUTED = "#667085"
TEAL = "#157A6E"
GOLD = "#B8860B"
RED = "#A23E48"
CUE_TEXT = {
    "sweep_low": "低斜率：交点仍在椭圆上",
    "sweep_high": "高斜率：R 仍始终等于 -P",
    "area_ratio_3": "面积比停在 3，得 k=√5/2",
    "tangent_min": "tan∠PQR 到达最小值 4√3",
    "min_check": "离开极小点后，tan∠PQR 变大",
    "tangent_min_return": "回到 k=√3/2，数值再次最小",
    "initial": "回到 TikZ 初始帧，对象不重建",
}


class EllipseAnalyticGeometryDriverDemo(Scene):
    """TikZ-native figure driven by one declared analytic-geometry parameter."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        picture = compile_document(SOURCE).pictures[0]
        motion_spec = load_motion_spec(MOTION)
        motion_spec.validate_cues(frozenset(CUE_TEXT))
        motion_spec.validate_picture(picture)

        renderer = NativeManimRenderer(scene_unit_per_cm=0.92)
        figure = renderer.render(picture)
        target_center = np.array([-3.3, -0.35, 0.0])
        group_shift = target_center - figure.group.get_center()
        figure.group.shift(group_shift)

        coordinate_origin = renderer.point((0.0, 0.0), picture) + group_shift
        geometry_scale = renderer.unit * picture.scale

        def to_scene_point(point):
            return coordinate_origin + geometry_scale * np.array(
                [point[0], point[1], 0.0]
            )

        theta = ValueTracker(motion_spec.driver.initial)
        runtime = NativeMotionRuntime(motion_spec, picture, theta.get_value)
        runtime.bind(figure, renderer, to_scene_point)

        title = Text(
            "TikZ → Manim 解析几何驱动 Demo",
            font="PingFang SC",
            font_size=31,
            color=INK,
        ).to_edge(UP, buff=0.28)
        subtitle = Text(
            "只改变直线角度 θ，P、Q、R 和所有从属图形每帧依约束重算",
            font="PingFang SC",
            font_size=18,
            color=MUTED,
        ).next_to(title, DOWN, buff=0.09)

        panel = RoundedRectangle(
            width=5.15,
            height=5.7,
            corner_radius=0.22,
            stroke_color="#CAD4E3",
            stroke_width=1.5,
            fill_color=WHITE,
            fill_opacity=0.96,
        ).move_to([3.7, -0.42, 0.0])
        equation = MathTex(
            r"C:\ \frac{x^2}{4}+\frac{y^2}{3}=1",
            color=INK,
            font_size=34,
        ).move_to(panel.get_top() + DOWN * 0.48)
        relation = MathTex(
            r"l:\ y=k(x+1),\qquad R=-P",
            color=INK,
            font_size=28,
        ).next_to(equation, DOWN, buff=0.17)

        labels = VGroup(
            Text("θ（度）", font="PingFang SC", font_size=20, color=MUTED),
            Text("斜率 k", font="PingFang SC", font_size=20, color=MUTED),
            Text("面积比", font="PingFang SC", font_size=20, color=MUTED),
            Text("tan∠PQR", font="PingFang SC", font_size=20, color=MUTED),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        labels.move_to(panel.get_center() + LEFT * 1.25 + UP * 0.12)

        values = VGroup(
            DecimalNumber(degrees(theta.get_value()), num_decimal_places=2, color=BLUE_D),
            DecimalNumber(0.75, num_decimal_places=4, color=RED),
            DecimalNumber(0.0, num_decimal_places=4, color=TEAL),
            DecimalNumber(0.0, num_decimal_places=4, color=GOLD),
        )
        for value, label in zip(values, labels):
            value.scale(0.62)
            value.next_to(label, RIGHT, buff=0.62)

        def current_metrics():
            return ellipse_chord_metrics(runtime.coordinates())

        values[0].add_updater(lambda item: item.set_value(degrees(theta.get_value())))
        values[1].add_updater(lambda item: item.set_value(current_metrics().slope))
        values[2].add_updater(lambda item: item.set_value(current_metrics().area_ratio))
        values[3].add_updater(lambda item: item.set_value(current_metrics().angle_tangent))

        area_formula = MathTex(
            r"\frac{S_{PQR}}{S_{PFO}}=3\ \Longrightarrow\ k=\frac{\sqrt5}{2}",
            color=TEAL,
            font_size=22,
        ).move_to(panel.get_bottom() + UP * 1.42)
        tangent_formula = MathTex(
            r"\tan\angle PQR=4k+\frac3k\ge 4\sqrt3",
            color=GOLD,
            font_size=22,
        ).next_to(area_formula, DOWN, buff=0.14)
        cue = Text(
            "有向直线确保 Q、P 身份不互换",
            font="PingFang SC",
            font_size=16,
            color=MUTED,
        ).move_to(panel.get_bottom() + UP * 0.24)

        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.6)
        self.play(FadeIn(figure.group), FadeIn(panel), run_time=0.8)
        self.play(
            FadeIn(equation),
            FadeIn(relation),
            FadeIn(labels),
            FadeIn(values),
            FadeIn(area_formula),
            FadeIn(tangent_formula),
            FadeIn(cue),
            run_time=0.8,
        )
        self.wait(0.35)

        def on_cue(step) -> None:
            updated = Text(
                CUE_TEXT[step.cue],
                font="PingFang SC",
                font_size=16,
                color=INK,
            ).move_to(cue)
            self.play(Transform(cue, updated), run_time=0.28)
            if step.cue == "area_ratio_3":
                self.play(Indicate(values[2], color=TEAL), run_time=0.55)
            elif step.cue in {"tangent_min", "tangent_min_return"}:
                self.play(Indicate(values[3], color=GOLD), run_time=0.55)

        runtime.play_timeline(self, theta, on_cue=on_cue)
        self.wait(0.45)
