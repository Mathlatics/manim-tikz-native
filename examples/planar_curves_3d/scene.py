from __future__ import annotations

from pathlib import Path

from manim import (
    DEGREES,
    DOWN,
    LEFT,
    RIGHT,
    Scene,
    Text,
    ThreeDScene,
    UP,
    VGroup,
)

from tikz_native import (
    NativeFixedViewRenderer,
    NativeManim3DRenderer,
    compile_document,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "planar_curves_3d.tex"
BACKGROUND = "#F7F9FC"
INK = "#20242A"
MUTED = "#667085"


def _document():
    document = compile_document(SOURCE)
    if len(document.pictures) != 3:
        raise RuntimeError(f"Expected three TikZ pictures, got {len(document.pictures)}")
    for picture in document.pictures:
        if picture.unsupported:
            raise RuntimeError("Unsupported TikZ: " + "; ".join(picture.unsupported))
    return document


class ExplicitPlanarCurvesFixedViewDemo(Scene):
    """Compare the certified rank-two and exact rank-one fixed views."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        pictures = _document().pictures
        renderer = NativeFixedViewRenderer(scene_unit_per_cm=0.9)
        groups = [renderer.render(picture).group for picture in pictures[:2]]
        groups[0].move_to(LEFT * 3.1 + DOWN * 0.2)
        groups[1].move_to(RIGHT * 3.1 + DOWN * 0.2)

        title = Text(
            "显式三维圆与椭圆：一般投影与精确侧视",
            font="PingFang SC",
            font_size=30,
            color=INK,
        ).to_edge(UP, buff=0.35)
        labels = VGroup(
            Text("rank 2：仿射椭圆", font="PingFang SC", font_size=20, color=MUTED),
            Text("rank 1：有限线段", font="PingFang SC", font_size=20, color=MUTED),
        )
        labels[0].next_to(groups[0], DOWN, buff=0.5)
        labels[1].next_to(groups[1], DOWN, buff=0.5)
        self.add(title, *groups, labels)
        self.wait(1.0)


class ExplicitPlanarCurvesWorldCameraDemo(ThreeDScene):
    """Keep the same authored curves while only the Manim camera moves."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        picture = _document().pictures[2]
        figure = NativeManim3DRenderer(scene_unit_per_cm=1.5).render(picture)
        figure.world_group.move_to([0.0, 0.0, 0.0])
        self.set_camera_orientation(phi=66 * DEGREES, theta=-48 * DEGREES)
        self.add(figure.world_group)
        self.begin_ambient_camera_rotation(rate=0.32)
        self.wait(2.4)
        self.stop_ambient_camera_rotation()
        self.wait(0.2)


__all__ = [
    "ExplicitPlanarCurvesFixedViewDemo",
    "ExplicitPlanarCurvesWorldCameraDemo",
]
