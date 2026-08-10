from __future__ import annotations

from pathlib import Path

from manim import ThreeDScene, ValueTracker

from tikz_native import compile_document
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.manim_renderer_3d import NativeManim3DRenderer
from tikz_native.motion_3d import Motion3DSpec, NativeMotion3DRuntime


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dihedral_fold.tex"
MOTION = ROOT / "motion-3d.json"


class DihedralFold3DAnimationDemo(ThreeDScene):
    """Animate one compiled TikZ-native 3D figure without rebuilding it."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=MultiProjectionCamera, **kwargs)

    def construct(self) -> None:
        self.camera.background_color = "#F6F8FC"

        document = compile_document(SOURCE)
        if len(document.pictures) != 1:
            raise RuntimeError(f"Expected one TikZ picture, got {len(document.pictures)}")
        picture = document.pictures[0]
        if picture.unsupported:
            raise RuntimeError("Unsupported TikZ: " + "; ".join(picture.unsupported))

        motion = Motion3DSpec.load(MOTION)
        renderer = NativeManim3DRenderer(scene_unit_per_cm=1.0)
        figure = renderer.render(picture)
        fold_angle = ValueTracker(motion.driver.initial)
        runtime = NativeMotion3DRuntime(motion, picture, fold_angle.get_value)

        runtime.prepare_camera(self.camera, view_center=figure.view_center)
        runtime.bind(figure, renderer, camera=self.camera)
        runtime.bind_occlusions(figure, renderer, self.camera)

        self.add(figure.world_group)
        self.add_fixed_orientation_mobjects(*figure.fixed_orientation_labels)
        runtime.play_timeline(self, fold_angle, self.camera)
        self.wait(0.2)


__all__ = ["DihedralFold3DAnimationDemo"]
