from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import degrees
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np
from PIL import Image
from manim import (
    BLUE_D,
    DOWN,
    DecimalNumber,
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
    tempconfig,
)

from .fixed_view_renderer import NativeFixedViewRenderer
from .motion_runtime import (
    MotionConfigError,
    MotionSpec,
    MotionTimelineStep,
    NativeMotionRuntime,
    ellipse_chord_metrics,
)
from .provider import CompiledAsset


TRACE_SCHEMA = "tikz-native-motion-trace/v1"
LAYOUT_ELLIPSE_CHORD_ANALYSIS = "ellipse_chord_analysis"

BACKGROUND_DEFAULT = "#F6F8FC"
INK = "#20242A"
MUTED = "#667085"
TEAL = "#157A6E"
GOLD = "#B8860B"
RED = "#A23E48"

ELLIPSE_CUE_TEXT = {
    "sweep_low": "低斜率：交点仍在椭圆上",
    "sweep_high": "高斜率：R 仍始终等于 -P",
    "area_ratio_3": "面积比停在 3，得 k=√5/2",
    "tangent_min": "tan∠PQR 到达最小值 4√3",
    "min_check": "离开极小点后，tan∠PQR 变大",
    "tangent_min_return": "回到 k=√3/2，数值再次最小",
    "initial": "回到 TikZ 初始帧，对象不重建",
}


class MotionRenderError(RuntimeError):
    """Raised when a validated motion package cannot be rendered completely."""


@dataclass(frozen=True)
class MotionPreviewProfile:
    pixel_width: int
    pixel_height: int
    frame_rate: int
    background: str = BACKGROUND_DEFAULT
    layout: str = LAYOUT_ELLIPSE_CHORD_ANALYSIS


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _json_coordinates(coordinates: Mapping[str, tuple[float, float]]) -> dict[str, list[float]]:
    return {
        name: [float(value[0]), float(value[1])]
        for name, value in sorted(coordinates.items())
    }


def build_motion_trace(
    compiled: CompiledAsset,
    motion: MotionSpec,
    *,
    provider_revision: str,
    source_sha256: str,
    motion_sha256: str,
    profile: MotionPreviewProfile,
) -> dict[str, Any]:
    """Evaluate every authored boundary before Manim starts rendering.

    This makes an invalid dependency, non-finite coordinate, or incompatible
    analytic layout fail before a partially rendered package can be published.
    """

    parameter = [motion.driver.initial]
    runtime = NativeMotionRuntime(motion, compiled.picture, lambda: parameter[0])
    boundaries: list[tuple[str, float, str | None]] = [
        ("initial", motion.driver.initial, None)
    ]
    boundaries.extend(
        (f"timeline:{index}", step.to, step.cue)
        for index, step in enumerate(motion.timeline)
    )
    frames: list[dict[str, Any]] = []
    for boundary, value, cue in boundaries:
        parameter[0] = value
        coordinates = runtime.coordinates()
        try:
            metrics = ellipse_chord_metrics(coordinates)
        except (KeyError, ValueError, ZeroDivisionError) as error:
            raise MotionConfigError(
                "ellipse_chord_analysis requires valid P, Q, R, F and O geometry: "
                f"{error}"
            ) from error
        frames.append(
            {
                "boundary": boundary,
                "parameter": float(value),
                "cue": cue,
                "coordinates": _json_coordinates(coordinates),
                "metrics": asdict(metrics),
            }
        )
    return {
        "schema": TRACE_SCHEMA,
        "provider_revision": provider_revision,
        "source_sha256": source_sha256,
        "motion_sha256": motion_sha256,
        "picture_index": compiled.picture.index,
        "motion_schema": motion.schema,
        "driver": asdict(motion.driver),
        "bindings": [asdict(binding) for binding in motion.bindings],
        "render": asdict(profile),
        "boundaries": frames,
    }


def _ellipse_scene_class(
    compiled: CompiledAsset,
    motion: MotionSpec,
    profile: MotionPreviewProfile,
) -> type[Scene]:
    motion.validate_cues(frozenset(ELLIPSE_CUE_TEXT))
    figure = compiled.figure
    picture = compiled.picture
    scene_unit_per_cm = float(compiled.asset["scene_unit_per_cm"])
    renderer = NativeFixedViewRenderer(scene_unit_per_cm=scene_unit_per_cm)

    class TikzNativeEllipseMotionPreview(Scene):
        def construct(self) -> None:
            self.camera.background_color = profile.background

            target_center = np.array([-3.3, -0.35, 0.0])
            group_shift = target_center - figure.group.get_center()
            figure.group.shift(group_shift)
            coordinate_origin = renderer.point((0.0, 0.0), picture) + group_shift
            geometry_scale = renderer.unit * picture.scale

            def to_scene_point(point: tuple[float, float]) -> np.ndarray:
                return coordinate_origin + geometry_scale * np.array(
                    [point[0], point[1], 0.0]
                )

            theta = ValueTracker(motion.driver.initial)
            runtime = NativeMotionRuntime(motion, picture, theta.get_value)
            runtime.bind(figure, renderer, to_scene_point)
            initial_metrics = ellipse_chord_metrics(runtime.coordinates())

            title = Text(
                "TikZ → Manim 解析几何驱动",
                font="PingFang SC",
                font_size=31,
                color=INK,
            ).to_edge(UP, buff=0.28)
            subtitle = Text(
                "只改变直线角度 θ，P、Q、R 和从属图形按约束逐帧重算",
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
                DecimalNumber(initial_metrics.slope, num_decimal_places=4, color=RED),
                DecimalNumber(initial_metrics.area_ratio, num_decimal_places=4, color=TEAL),
                DecimalNumber(initial_metrics.angle_tangent, num_decimal_places=4, color=GOLD),
            )
            for value, label in zip(values, labels):
                value.scale(0.62)
                value.next_to(label, RIGHT, buff=0.62)

            def current_metrics():
                return ellipse_chord_metrics(runtime.coordinates())

            values[0].add_updater(
                lambda item: item.set_value(degrees(theta.get_value()))
            )
            values[1].add_updater(
                lambda item: item.set_value(current_metrics().slope)
            )
            values[2].add_updater(
                lambda item: item.set_value(current_metrics().area_ratio)
            )
            values[3].add_updater(
                lambda item: item.set_value(current_metrics().angle_tangent)
            )

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

            self.add(
                title,
                subtitle,
                figure.group,
                panel,
                equation,
                relation,
                labels,
                values,
                area_formula,
                tangent_formula,
                cue,
            )
            self.wait(max(1.0 / profile.frame_rate, 0.05))

            def on_cue(step: MotionTimelineStep) -> None:
                if step.cue is None:
                    return
                updated = Text(
                    ELLIPSE_CUE_TEXT[step.cue],
                    font="PingFang SC",
                    font_size=16,
                    color=INK,
                ).move_to(cue)
                self.play(
                    Transform(cue, updated),
                    run_time=max(0.05, min(0.28, step.duration * 0.2)),
                )

            runtime.play_timeline(self, theta, on_cue=on_cue)
            self.wait(max(1.0 / profile.frame_rate, 0.05))

    return TikzNativeEllipseMotionPreview


def _run_checked(command: list[str], *, phase: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise MotionRenderError(f"{phase} failed: {message}")
    return result


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise MotionRenderError("ffprobe is required to validate motion previews")
    result = _run_checked(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        phase="ffprobe",
    )
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        frame_count = int(stream["nb_read_frames"])
        duration = float(payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MotionRenderError(f"ffprobe returned incomplete video metadata: {error}") from error
    if frame_count < 2 or duration <= 0:
        raise MotionRenderError("motion preview must contain at least two frames")
    return {
        "codec": str(stream["codec_name"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": str(stream["avg_frame_rate"]),
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def _extract_frame(video: Path, target: Path, frame_index: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise MotionRenderError("ffmpeg is required to extract motion preview frames")
    target.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            str(target),
        ],
        phase=f"extract frame {frame_index}",
    )


def _validate_png(path: Path, profile: MotionPreviewProfile) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MotionRenderError(f"rendered frame is missing or empty: {path}")
    with Image.open(path) as image:
        image.load()
        if image.size != (profile.pixel_width, profile.pixel_height):
            raise MotionRenderError(
                f"rendered frame has size {image.size}, expected "
                f"{(profile.pixel_width, profile.pixel_height)}"
            )


def render_motion_preview(
    compiled: CompiledAsset,
    motion: MotionSpec,
    output_root: Path,
    *,
    profile: MotionPreviewProfile,
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    if profile.layout != LAYOUT_ELLIPSE_CHORD_ANALYSIS:
        raise MotionRenderError(f"unsupported motion preview layout: {profile.layout}")
    motion.validate_picture(compiled.picture)
    preview_dir = output_root / "preview"
    media_dir = preview_dir / "media"
    video_path = preview_dir / "motion.mp4"
    first_path = preview_dir / "first.png"
    last_path = preview_dir / "last.png"
    trace_path = preview_dir / "trace.json"
    preview_dir.mkdir(parents=True, exist_ok=True)

    scene_class = _ellipse_scene_class(compiled, motion, profile)
    config = {
        "media_dir": str(media_dir),
        "pixel_width": profile.pixel_width,
        "pixel_height": profile.pixel_height,
        "frame_rate": profile.frame_rate,
        "format": "mp4",
        "write_to_movie": True,
        "save_last_frame": False,
        "disable_caching": True,
        "output_file": "tikz_native_motion_preview",
    }
    try:
        with tempconfig(config):
            scene = scene_class()
            scene.render()
            rendered_video = Path(scene.renderer.file_writer.movie_file_path)
            if not rendered_video.is_file():
                raise MotionRenderError(
                    f"Manim did not produce the expected movie: {rendered_video}"
                )
            shutil.copy2(rendered_video, video_path)
    except MotionRenderError:
        raise
    except Exception as error:
        raise MotionRenderError(
            f"Manim motion preview failed: {type(error).__name__}: {error}"
        ) from error

    media = _probe_video(video_path)
    if (media["width"], media["height"]) != (
        profile.pixel_width,
        profile.pixel_height,
    ):
        raise MotionRenderError(
            "motion preview dimensions do not match the requested profile"
        )
    _extract_frame(video_path, first_path, 0)
    _extract_frame(video_path, last_path, media["frame_count"] - 1)
    _validate_png(first_path, profile)
    _validate_png(last_path, profile)
    _write_json(trace_path, trace)
    return {
        "video": "preview/motion.mp4",
        "first_frame": "preview/first.png",
        "last_frame": "preview/last.png",
        "trace": "preview/trace.json",
        "media": media,
    }


__all__ = [
    "LAYOUT_ELLIPSE_CHORD_ANALYSIS",
    "MotionPreviewProfile",
    "MotionRenderError",
    "TRACE_SCHEMA",
    "build_motion_trace",
    "render_motion_preview",
]
