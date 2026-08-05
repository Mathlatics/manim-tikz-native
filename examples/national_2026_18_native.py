from __future__ import annotations

from math import atan, sqrt
from pathlib import Path

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    FadeIn,
    FadeOut,
    LaggedStart,
    MovingCameraScene,
    Scene,
    Text,
    Transform,
    UP,
    VGroup,
    ValueTracker,
    config,
)

from tikz_native import compile_document
from tikz_native.animation import play_named_reveal, play_semantic_reveal
from tikz_native.dynamic_geometry import (
    EllipseChordDriver,
    NativeMotionBinder,
    project_point_to_line,
)
from tikz_native.manim_renderer import NativeManimRenderer


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "national_2026_18_tikz.tex"
)
BACKGROUND = "#F8F6F0"
TEXT = "#20242A"


def load_document():
    return compile_document(SOURCE)


def frame_without_rescaling_objects(
    scene: MovingCameraScene,
    content: VGroup,
    *,
    margin: float = 0.45,
) -> None:
    """Expand the common camera view; never scale an individual figure."""

    aspect_ratio = config.frame_width / config.frame_height
    required_width = max(
        float(content.width) + 2 * margin,
        (float(content.height) + 2 * margin) * aspect_ratio,
    )
    scene.camera.frame.scale(required_width / scene.camera.frame.width)
    scene.camera.frame.move_to(content)


class _NativeFigureScene(Scene):
    picture_index = 1
    scene_unit_per_cm = 1.0

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        document = load_document()
        renderer = NativeManimRenderer(scene_unit_per_cm=self.scene_unit_per_cm)
        figure = renderer.render(document.pictures[self.picture_index - 1])
        figure.group.move_to([0, 0, 0])
        self.add(figure.group)


class National2026TikzNativeFigure01(_NativeFigureScene):
    picture_index = 1


class National2026TikzNativeFigure02(_NativeFigureScene):
    picture_index = 2


class National2026TikzNativeFigure03(_NativeFigureScene):
    picture_index = 3


class National2026TikzNativeFigure04(_NativeFigureScene):
    picture_index = 4


class National2026TikzNativeFigure05(_NativeFigureScene):
    picture_index = 5


class National2026TikzNativeFigure06(_NativeFigureScene):
    picture_index = 6


class National2026TikzNativeFigure07(_NativeFigureScene):
    picture_index = 7


class National2026TikzNativeFigure08(_NativeFigureScene):
    picture_index = 8


class National2026TikzNativeFigure09(_NativeFigureScene):
    picture_index = 9


class National2026TikzNativeFigure10(_NativeFigureScene):
    picture_index = 10


class National2026TikzNativeFigure11(_NativeFigureScene):
    picture_index = 11


class National2026TikzNativeFigure12(_NativeFigureScene):
    picture_index = 12


class National2026TikzNativeFigure13(_NativeFigureScene):
    picture_index = 13


class National2026TikzNativeFigure14(_NativeFigureScene):
    picture_index = 14


class National2026TikzNativeFigure15(_NativeFigureScene):
    picture_index = 15


class National2026TikzNativeFigure16(_NativeFigureScene):
    picture_index = 16


class National2026TikzNativeRepresentativeGallery(MovingCameraScene):
    """Static QA gallery for the five high-risk representative figures."""

    indices = (1, 4, 7, 10, 16)

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        document = load_document()
        renderer = NativeManimRenderer(scene_unit_per_cm=0.72)
        panels = []
        for index in self.indices:
            figure = renderer.render(document.pictures[index - 1])
            content = figure.group
            title = Text(f"TikZ {index:02d}", font="PingFang SC", font_size=19, color=TEXT)
            title.next_to(content, UP, buff=0.10)
            panels.append(VGroup(title, content))
        row1 = VGroup(*panels[:3]).arrange(buff=0.55)
        row2 = VGroup(*panels[3:]).arrange(buff=1.0)
        gallery = VGroup(row1, row2).arrange(DOWN, buff=0.55).move_to([0, 0, 0])
        frame_without_rescaling_objects(self, gallery)
        self.add(gallery)


class National2026TikzNativeAllGallery(MovingCameraScene):
    """Compact 4-by-4 visual regression sheet for every extracted picture."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        document = load_document()
        renderer = NativeManimRenderer(scene_unit_per_cm=0.56)
        panels = []
        for index, picture in enumerate(document.pictures, start=1):
            figure = renderer.render(picture)
            content = figure.group
            title = Text(
                f"{index:02d}", font="PingFang SC", font_size=14, color=TEXT
            )
            title.next_to(content, UP, buff=0.05)
            panels.append(VGroup(title, content))
        gallery = VGroup(*panels).arrange_in_grid(
            rows=4,
            cols=4,
            buff=(0.30, 0.24),
        )
        gallery.move_to([0, 0, 0])
        frame_without_rescaling_objects(self, gallery)
        self.add(gallery)


class National2026TikzNativeAllGalleryReveal(MovingCameraScene):
    """Animate all 262 native objects without scaling any individual panel."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        document = load_document()
        renderer = NativeManimRenderer(scene_unit_per_cm=0.56)
        figures = []
        titles = []
        panels = []
        for index, picture in enumerate(document.pictures, start=1):
            figure = renderer.render(picture)
            title = Text(
                f"{index:02d}", font="PingFang SC", font_size=14, color=TEXT
            )
            title.next_to(figure.group, UP, buff=0.05)
            figures.append(figure)
            titles.append(title)
            panels.append(VGroup(title, figure.group))
        gallery = VGroup(*panels).arrange_in_grid(
            rows=4,
            cols=4,
            buff=(0.30, 0.24),
        )
        gallery.move_to([0, 0, 0])
        frame_without_rescaling_objects(self, gallery)

        self.add(*titles)
        play_semantic_reveal(
            self,
            figures,
            object_lag_ratio=0.012,
            label_mode="fade",
        )
        self.wait(0.6)


class National2026TikzNativeFigure04Reveal(Scene):
    """Generic semantic reveal applied to one representative figure."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        document = load_document()
        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        figure = renderer.render(document.pictures[3])
        figure.group.move_to([0, 0, 0])
        play_semantic_reveal(self, [figure])
        self.wait(0.6)


class National2026TikzNativeFigure04Construction(Scene):
    """Pedagogical construction driven only by stable native-object IDs."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        document = load_document()
        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        figure = renderer.render(document.pictures[3])
        figure.group.move_to([0, 0, 0])

        play_named_reveal(
            self, figure, ("arrow", "arrow.2"), run_time=0.85, lag_ratio=0.12
        )
        play_named_reveal(
            self, figure, ("label.x", "label.y"), run_time=0.50, lag_ratio=0.18
        )
        play_named_reveal(
            self, figure, ("ellipse", "line"), run_time=1.10, lag_ratio=0.18
        )
        play_named_reveal(self, figure, ("label.l",), run_time=0.38)
        play_named_reveal(
            self,
            figure,
            ("dot.P", "dot.Q", "dot.R"),
            run_time=0.65,
            lag_ratio=0.18,
        )
        play_named_reveal(
            self,
            figure,
            ("label.P.P", "label.Q.Q", "label.R.R"),
            run_time=0.85,
            lag_ratio=0.16,
        )
        play_named_reveal(self, figure, ("fill.P.Q.R",), run_time=0.45)
        play_named_reveal(
            self,
            figure,
            ("line.P.Q", "line.Q.R", "line.R.P"),
            run_time=1.35,
            lag_ratio=0.16,
        )
        play_named_reveal(
            self, figure, ("label_path.P.Q.PQ",), run_time=0.55
        )
        play_named_reveal(self, figure, ("line.R.H",), run_time=0.90)
        play_named_reveal(
            self, figure, ("label_path.R.H.d",), run_time=0.45
        )
        play_named_reveal(
            self, figure, ("right_angle.R.H.P",), run_time=0.48
        )
        self.wait(0.8)


class National2026TikzNativeFigure01DrivenMotion(Scene):
    """Rotate line l and recompute every dependent native object each frame."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        document = load_document()
        picture = document.pictures[0]
        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        figure = renderer.render(picture)
        group_shift = -figure.group.get_center().copy()
        figure.group.shift(group_shift)

        intersection = picture.intersections[0]
        source_line = picture.named_paths[intersection.sort_by]
        source_direction = np.array(source_line.geometry["end"]) - np.array(
            source_line.geometry["start"]
        )
        initial_angle = float(np.arctan2(source_direction[1], source_direction[0]))
        angle = ValueTracker(initial_angle)
        coordinate_origin = renderer.point((0.0, 0.0), picture) + group_shift
        geometry_scale = renderer.unit * picture.scale
        driver = EllipseChordDriver.from_named_intersection(
            angle.get_value,
            picture,
            pivot_name="F",
        )

        def state():
            return driver.state()

        def scene_point(point):
            return coordinate_origin + geometry_scale * np.array(
                [point[0], point[1], 0.0]
            )

        binder = NativeMotionBinder(figure, renderer)
        binder.bind_line(
            "line",
            lambda: scene_point(state().line_start),
            lambda: scene_point(state().line_end),
        )
        binder.bind_line(
            "line.P.Q",
            lambda: scene_point(state().p),
            lambda: scene_point(state().q),
        )
        binder.bind_line(
            "line.Q.R",
            lambda: scene_point(state().q),
            lambda: scene_point(state().r),
        )
        binder.bind_line(
            "line.R.P",
            lambda: scene_point(state().r),
            lambda: scene_point(state().p),
        )
        binder.bind_line(
            "line.P.F",
            lambda: scene_point(state().p),
            lambda: scene_point(state().focus),
        )
        binder.bind_line(
            "line.O.P",
            lambda: scene_point(state().center),
            lambda: scene_point(state().p),
        )
        binder.bind_line(
            "line.Q.O",
            lambda: scene_point(state().q),
            lambda: scene_point(state().center),
        )
        binder.bind_polygon(
            "fill.P.Q.R",
            lambda: [
                scene_point(state().p),
                scene_point(state().q),
                scene_point(state().r),
            ],
        )
        binder.bind_polygon(
            "fill.P.F.O",
            lambda: [
                scene_point(state().p),
                scene_point(state().focus),
                scene_point(state().center),
            ],
        )
        for name in ("p", "q", "r"):
            object_name = name.upper()
            binder.bind_dot(
                f"dot.{object_name}",
                lambda point_name=name: scene_point(getattr(state(), point_name)),
            )
            binder.bind_label(
                f"label.{object_name}.{object_name}",
                lambda point_name=name: scene_point(getattr(state(), point_name)),
            )
        binder.bind_label("label.l", lambda: scene_point(state().line_end))

        self.add(figure.group)
        self.wait(0.6)
        self.play(angle.animate.set_value(22 * DEGREES), run_time=2.2)
        self.play(angle.animate.set_value(54 * DEGREES), run_time=3.4)
        self.play(angle.animate.set_value(initial_angle), run_time=2.2)
        self.wait(0.8)


class National2026TikzNativeFigure04DrivenMotion(Scene):
    """Drive a chord, projection, path labels and right angle together."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        document = load_document()
        picture = document.pictures[3]
        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        figure = renderer.render(picture)
        group_shift = -figure.group.get_center().copy()
        figure.group.shift(group_shift)

        initial_angle = atan(3 / 4)
        angle = ValueTracker(initial_angle)
        coordinate_origin = renderer.point((0.0, 0.0), picture) + group_shift
        geometry_scale = renderer.unit * picture.scale
        driver = EllipseChordDriver(
            angle.get_value,
            semi_major=2.0,
            semi_minor=sqrt(3),
            focus=(-1.0, 0.0),
            backward_length=1.5625,
            forward_length=3.1875,
        )

        def state():
            return driver.state()

        def foot():
            current = state()
            return project_point_to_line(current.r, current.p, current.q)

        def scene_point(point):
            return coordinate_origin + geometry_scale * np.array(
                [point[0], point[1], 0.0]
            )

        binder = NativeMotionBinder(figure, renderer)
        binder.bind_line(
            "line",
            lambda: scene_point(state().line_start),
            lambda: scene_point(state().line_end),
        )
        binder.bind_line(
            "line.P.Q",
            lambda: scene_point(state().p),
            lambda: scene_point(state().q),
        )
        binder.bind_line(
            "line.Q.R",
            lambda: scene_point(state().q),
            lambda: scene_point(state().r),
        )
        binder.bind_line(
            "line.R.P",
            lambda: scene_point(state().r),
            lambda: scene_point(state().p),
        )
        binder.bind_line(
            "line.R.H",
            lambda: scene_point(state().r),
            lambda: scene_point(foot()),
        )
        binder.bind_polygon(
            "fill.P.Q.R",
            lambda: [
                scene_point(state().p),
                scene_point(state().q),
                scene_point(state().r),
            ],
        )
        binder.bind_path_label(
            "label_path.P.Q.PQ",
            lambda: scene_point(state().p),
            lambda: scene_point(state().q),
        )
        binder.bind_path_label(
            "label_path.R.H.d",
            lambda: scene_point(state().r),
            lambda: scene_point(foot()),
        )
        binder.bind_right_angle(
            "right_angle.R.H.P",
            lambda: scene_point(state().r),
            lambda: scene_point(foot()),
            lambda: scene_point(state().p),
        )
        for name in ("p", "q", "r"):
            object_name = name.upper()
            binder.bind_dot(
                f"dot.{object_name}",
                lambda point_name=name: scene_point(getattr(state(), point_name)),
            )
            binder.bind_label(
                f"label.{object_name}.{object_name}",
                lambda point_name=name: scene_point(getattr(state(), point_name)),
            )
        binder.bind_label("label.l", lambda: scene_point(state().line_end))

        self.add(figure.group)
        self.wait(0.6)
        self.play(angle.animate.set_value(22 * DEGREES), run_time=2.2)
        self.play(angle.animate.set_value(54 * DEGREES), run_time=3.4)
        self.play(angle.animate.set_value(initial_angle), run_time=2.2)
        self.wait(0.8)


class National2026TikzNativeFigure13To14(Scene):
    """Object-by-object affine-diagram transition using stable semantic IDs."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        document = load_document()
        renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
        source = renderer.render(document.pictures[12])
        target = renderer.render(document.pictures[13])
        source.group.move_to([0, 0, 0])
        target.group.move_to([0, 0, 0])

        mapping = {
            "arrow": "arrow",
            "label.X": "label.x",
            "arrow.2": "arrow.2",
            "label.Y": "label.y",
            "line.A.B": "line.A.B",
            "line.A.C": "line.A.C",
            "label_path.A.C.Delta_X": "label_path.A.C.Delta_x_a_Delta_X",
            "line.C.B": "line.C.B",
            "label_path.C.B.Delta_Y": "label_path.C.B.Delta_y_b_Delta_Y",
            "dot.A": "dot.A",
            "dot.B": "dot.B",
            "label.A.A": "label.A.A",
            "label.B.B": "label.B.B",
            "label.K_frac_Delta_Y_Delta_X": (
                "label.k_frac_Delta_y_Delta_x_frac_baK"
            ),
        }
        source_specs = {item.id: item for item in source.picture.objects}
        label_pairs = [
            (source_id, target_id)
            for source_id, target_id in mapping.items()
            if source_specs[source_id].kind in {"label", "path_label"}
        ]
        geometry_pairs = [
            (source_id, target_id)
            for source_id, target_id in mapping.items()
            if source_specs[source_id].kind not in {"label", "path_label"}
        ]
        self.add(source.group)
        self.play(
            LaggedStart(
                *(FadeOut(source.objects[source_id]) for source_id, _ in label_pairs),
                lag_ratio=0.03,
            ),
            run_time=0.45,
        )
        self.play(
            LaggedStart(
                *(
                    Transform(source.objects[source_id], target.objects[target_id])
                    for source_id, target_id in geometry_pairs
                ),
                lag_ratio=0.04,
            ),
            run_time=1.45,
        )
        self.play(
            LaggedStart(
                *(FadeIn(target.objects[target_id]) for _, target_id in label_pairs),
                lag_ratio=0.05,
            ),
            run_time=0.7,
        )
        self.wait(0.5)
