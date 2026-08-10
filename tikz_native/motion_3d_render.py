from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np
from PIL import Image
from manim import ThreeDScene, ValueTracker, tempconfig

from .camera_3d import MultiProjectionCamera
from .compatibility import audit_document_compatibility
from .compiler import DocumentSpec, PictureSpec, compile_document
from .manim_renderer_3d import Native3DFigure, NativeManim3DRenderer
from .motion_3d import Motion3DConfigError, Motion3DSpec, NativeMotion3DRuntime
from .provider import (
    ERROR_COMPATIBILITY,
    ERROR_HASH_MISMATCH,
    ERROR_INPUT,
    ERROR_INSTANTIATION,
    TikzNativeProviderError,
)


TRACE_SCHEMA = "tikz-native-motion-3d-trace/v1"
ASSET_SCHEMA = "tikz-native-motion-3d-asset/v1"
BACKGROUND_DEFAULT = "#F6F8FC"


class Motion3DRenderError(RuntimeError):
    """Raised when a validated 3D preview package cannot render completely."""


@dataclass(frozen=True)
class Motion3DPreviewProfile:
    name: str = "preview_854x480_15fps"
    pixel_width: int = 854
    pixel_height: int = 480
    frame_rate: int = 15
    background: str = BACKGROUND_DEFAULT
    projection: str = "parallel"
    camera_zoom: float = 1.35


MOTION_3D_PREVIEW_PROFILE = Motion3DPreviewProfile()


@dataclass
class CompiledMotion3DAsset:
    document: DocumentSpec
    picture: PictureSpec
    compatibility: dict[str, Any]
    selected_compatibility: dict[str, Any]
    renderer: NativeManim3DRenderer
    figure: Native3DFigure
    asset: dict[str, Any]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _select_picture(document: DocumentSpec, picture_index: int) -> PictureSpec:
    matches = [picture for picture in document.pictures if picture.index == picture_index]
    if not matches:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "select_picture",
            f"picture_index {picture_index} is not available",
            details={"available_picture_indices": [item.index for item in document.pictures]},
        )
    return matches[0]


def _selected_compatibility(
    compatibility: Mapping[str, Any],
    picture_index: int,
) -> dict[str, Any]:
    selected = next(
        (
            item
            for item in compatibility.get("pictures", [])
            if item.get("picture") == picture_index
        ),
        None,
    )
    if selected is None:
        raise TikzNativeProviderError(
            ERROR_COMPATIBILITY,
            "compatibility",
            "compatibility report omitted the selected 3D picture",
        )
    blocked = any(
        finding.get("level") == "C"
        for finding in selected.get("findings", [])
    )
    return {
        "picture": picture_index,
        "static_status": "blocked" if blocked else "pass",
        "overall_level": selected.get("overall_level", "A"),
        "feature_counts": selected.get("feature_counts", {}),
        "findings": selected.get("findings", []),
    }


def _projection_payload(picture: PictureSpec) -> dict[str, Any]:
    projection = picture.projection_3d
    assert projection is not None
    return {
        "source": projection.source,
        "matrix": [list(row) for row in projection.matrix],
        "x_basis_cm": list(projection.x_basis_cm),
        "y_basis_cm": list(projection.y_basis_cm),
        "z_basis_cm": list(projection.z_basis_cm),
        "azimuth_degrees": projection.azimuth_degrees,
        "elevation_degrees": projection.elevation_degrees,
    }


def compile_motion_3d_asset(
    source_path: str | Path,
    *,
    source_sha256: str | None = None,
    entry_macro: str | None = None,
    picture_index: int = 1,
    scene_unit_per_cm: float = 1.0,
    strict_native: bool = True,
) -> CompiledMotion3DAsset:
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "read_input",
            f"source file does not exist: {source}",
        )
    try:
        document = compile_document(source, entry_macro=entry_macro)
    except Exception as error:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "compile",
            f"TikZ-native 3D compilation failed: {error}",
            details={"exception_type": type(error).__name__},
        ) from error
    if source_sha256 is not None and document.source_sha256 != source_sha256:
        raise TikzNativeProviderError(
            ERROR_HASH_MISMATCH,
            "verify_input",
            "source SHA-256 does not match the requested 3D snapshot",
            details={"expected": source_sha256, "actual": document.source_sha256},
        )

    picture = _select_picture(document, picture_index)
    if picture.dimension != 3 or picture.projection_3d is None:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "compile",
            "motion-3d preview requires a true 3D TikZ picture",
            details={"picture_index": picture_index, "dimension": picture.dimension},
        )
    compatibility = audit_document_compatibility(document)
    selected = _selected_compatibility(compatibility, picture_index)
    if strict_native and (
        selected["static_status"] != "pass" or picture.unsupported
    ):
        raise TikzNativeProviderError(
            ERROR_COMPATIBILITY,
            "compatibility",
            f"picture {picture_index} is blocked by the strict native 3D gate",
            details={
                "compatibility": selected,
                "unsupported": list(picture.unsupported),
            },
        )

    try:
        renderer = NativeManim3DRenderer(scene_unit_per_cm=scene_unit_per_cm)
        figure = renderer.render(picture)
    except Exception as error:
        raise TikzNativeProviderError(
            ERROR_INSTANTIATION,
            "instantiate_3d",
            f"failed to instantiate picture {picture.index} as native 3D objects: {error}",
            details={"exception_type": type(error).__name__},
        ) from error

    asset = {
        "schema": ASSET_SCHEMA,
        "source_sha256": document.source_sha256,
        "entry_macro": document.entry_macro,
        "picture_index": picture.index,
        "dimension": 3,
        "projection": _projection_payload(picture),
        "scene_unit_per_cm": float(scene_unit_per_cm),
        "view_mode": "world_3d",
        "placement_mode": "native_cm",
        "object_index": [
            {
                "id": item.id,
                "kind": item.kind,
                "z_index": item.z_index,
                "source_line": item.source_line,
                "label": item.label,
            }
            for item in picture.objects
        ],
        "occlusion_relation_ids": [
            relation.id for relation in picture.occlusion_relations
        ],
        "compatibility": selected,
        "warnings": list(figure.warnings),
        "files": {},
    }
    return CompiledMotion3DAsset(
        document=document,
        picture=picture,
        compatibility=compatibility,
        selected_compatibility=selected,
        renderer=renderer,
        figure=figure,
        asset=asset,
    )


def _json_coordinates(
    coordinates: Mapping[str, tuple[float, float, float]],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name, value in sorted(coordinates.items()):
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise Motion3DConfigError(
                f"trace coordinate {name!r} is not a finite 3D point"
            )
        result[name] = [float(component) for component in point]
    return result


def build_motion_3d_trace(
    compiled: CompiledMotion3DAsset,
    motion: Motion3DSpec,
    *,
    provider_revision: str,
    source_sha256: str,
    motion_sha256: str,
    profile: Motion3DPreviewProfile,
) -> dict[str, Any]:
    """Evaluate every authored boundary before Manim creates package files."""

    motion.validate_picture(compiled.picture)
    value = [motion.driver.initial]
    runtime = NativeMotion3DRuntime(motion, compiled.picture, lambda: value[0])
    camera_mode = motion.camera.entry_mode
    boundaries: list[dict[str, Any]] = [
        {
            "boundary": "initial",
            "step_type": "initial",
            "parameter": float(value[0]),
            "camera_mode": camera_mode,
            "cue": None,
            "coordinates": _json_coordinates(runtime.coordinates()),
        }
    ]
    for index, step in enumerate(motion.timeline):
        if step.type == "driver":
            assert step.to is not None
            value[0] = step.to
        elif step.type == "camera":
            assert step.mode is not None
            camera_mode = step.mode
        boundaries.append(
            {
                "boundary": f"timeline:{index}",
                "step_type": step.type,
                "parameter": float(value[0]),
                "camera_mode": camera_mode,
                "cue": step.cue,
                "coordinates": _json_coordinates(runtime.coordinates()),
            }
        )
    value[0] = motion.driver.initial
    boundaries.append(
        {
            "boundary": "restore",
            "step_type": "restore_entry",
            "parameter": float(value[0]),
            "camera_mode": motion.camera.entry_mode,
            "cue": None,
            "coordinates": _json_coordinates(runtime.coordinates()),
        }
    )
    if boundaries[0]["coordinates"] != boundaries[-1]["coordinates"]:
        raise Motion3DConfigError(
            "restore_entry did not reproduce the initial logical coordinates"
        )
    return {
        "schema": TRACE_SCHEMA,
        "provider_revision": provider_revision,
        "source_sha256": source_sha256,
        "motion_sha256": motion_sha256,
        "picture_index": compiled.picture.index,
        "motion_schema": motion.schema,
        "end_policy": motion.end_policy,
        "driver": asdict(motion.driver),
        "derived_coordinates": [
            asdict(item) for item in motion.derived_coordinates
        ],
        "bindings": [asdict(item) for item in motion.bindings],
        "camera": asdict(motion.camera),
        "timeline": [asdict(item) for item in motion.timeline],
        "render": asdict(profile),
        "stable_object_ids": [item.object_id for item in motion.bindings],
        "boundaries": boundaries,
    }


def _scene_class(
    compiled: CompiledMotion3DAsset,
    motion: Motion3DSpec,
    profile: Motion3DPreviewProfile,
) -> type[ThreeDScene]:
    class TikzNativeMotion3DPreview(ThreeDScene):
        def __init__(self, **kwargs):
            super().__init__(camera_class=MultiProjectionCamera, **kwargs)

        def construct(self) -> None:
            self.camera.background_color = profile.background
            camera: MultiProjectionCamera = self.camera
            camera.set_zoom(profile.camera_zoom)
            tracker = ValueTracker(motion.driver.initial)
            runtime = NativeMotion3DRuntime(
                motion,
                compiled.picture,
                tracker.get_value,
            )
            runtime.prepare_camera(
                camera,
                view_center=compiled.figure.view_center,
            )
            runtime.bind(
                compiled.figure,
                compiled.renderer,
                camera=camera,
            )
            runtime.bind_occlusions(
                compiled.figure,
                compiled.renderer,
                camera,
            )
            self.add(compiled.figure.world_group)
            self.add_fixed_orientation_mobjects(
                *compiled.figure.fixed_orientation_labels
            )
            self.wait(max(1.0 / profile.frame_rate, 0.05))
            runtime.play_timeline(self, tracker, camera)
            self.wait(max(1.0 / profile.frame_rate, 0.05))

    return TikzNativeMotion3DPreview


def _run_checked(command: list[str], *, phase: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise Motion3DRenderError(f"{phase} failed: {message}")
    return result


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise Motion3DRenderError("ffprobe is required to validate 3D previews")
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
        raise Motion3DRenderError(
            f"ffprobe returned incomplete video metadata: {error}"
        ) from error
    if frame_count < 2 or duration <= 0:
        raise Motion3DRenderError("3D preview must contain at least two frames")
    return {
        "codec": str(stream["codec_name"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": str(stream["avg_frame_rate"]),
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def _frame_rate_value(value: str) -> float:
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            result = float(numerator) / float(denominator)
        else:
            result = float(value)
    except (ValueError, ZeroDivisionError) as error:
        raise Motion3DRenderError(
            f"ffprobe returned an invalid frame rate: {value!r}"
        ) from error
    if not np.isfinite(result) or result <= 0:
        raise Motion3DRenderError(
            f"ffprobe returned an invalid frame rate: {value!r}"
        )
    return result


def _extract_frame(video: Path, target: Path, frame_index: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise Motion3DRenderError("ffmpeg is required to extract 3D preview frames")
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


def _validate_png(path: Path, profile: Motion3DPreviewProfile) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise Motion3DRenderError(f"rendered frame is missing or empty: {path}")
    with Image.open(path) as image:
        image.load()
        if image.size != (profile.pixel_width, profile.pixel_height):
            raise Motion3DRenderError(
                f"rendered frame has size {image.size}, expected "
                f"{(profile.pixel_width, profile.pixel_height)}"
            )


def render_motion_3d_preview(
    compiled: CompiledMotion3DAsset,
    motion: Motion3DSpec,
    output_root: Path,
    *,
    profile: Motion3DPreviewProfile,
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    if profile != MOTION_3D_PREVIEW_PROFILE:
        raise Motion3DRenderError("motion-3d/v1 only accepts the fixed preview profile")
    if profile.projection != "parallel":
        raise Motion3DRenderError("motion-3d/v1 only supports parallel projection")
    motion.validate_picture(compiled.picture)
    preview_dir = output_root / "preview"
    media_dir = preview_dir / "media"
    video_path = preview_dir / "motion-3d.mp4"
    first_path = preview_dir / "first.png"
    last_path = preview_dir / "last.png"
    trace_path = preview_dir / "trace.json"
    preview_dir.mkdir(parents=True, exist_ok=True)

    scene_class = _scene_class(compiled, motion, profile)
    config = {
        "media_dir": str(media_dir),
        "pixel_width": profile.pixel_width,
        "pixel_height": profile.pixel_height,
        "frame_rate": profile.frame_rate,
        "format": "mp4",
        "write_to_movie": True,
        "save_last_frame": False,
        "disable_caching": True,
        "output_file": "tikz_native_motion_3d_preview",
    }
    try:
        with tempconfig(config):
            scene = scene_class()
            scene.render()
            rendered_video = Path(scene.renderer.file_writer.movie_file_path)
            if not rendered_video.is_file():
                raise Motion3DRenderError(
                    f"Manim did not produce the expected movie: {rendered_video}"
                )
            shutil.copy2(rendered_video, video_path)
    except Motion3DRenderError:
        raise
    except Exception as error:
        raise Motion3DRenderError(
            f"Manim 3D motion preview failed: {type(error).__name__}: {error}"
        ) from error

    # Only the normalized public MP4 belongs in the package.  Manim's partial
    # movies and temporary render tree are reproducible workspace artifacts.
    shutil.rmtree(media_dir, ignore_errors=True)

    media = _probe_video(video_path)
    if (media["width"], media["height"]) != (
        profile.pixel_width,
        profile.pixel_height,
    ):
        raise Motion3DRenderError(
            "3D preview dimensions do not match the fixed profile"
        )
    expected_rate = f"{profile.frame_rate}/1"
    # MP4 time bases can make ffprobe report values such as
    # 15300000/1019971 even though Manim was configured at exactly 15 fps.
    if abs(_frame_rate_value(media["frame_rate"]) - profile.frame_rate) > 0.02:
        raise Motion3DRenderError(
            f"3D preview frame rate is {media['frame_rate']}, expected {expected_rate}"
        )
    _extract_frame(video_path, first_path, 0)
    _extract_frame(video_path, last_path, media["frame_count"] - 1)
    _validate_png(first_path, profile)
    _validate_png(last_path, profile)
    _write_json(trace_path, trace)
    return {
        "video": "preview/motion-3d.mp4",
        "first_frame": "preview/first.png",
        "last_frame": "preview/last.png",
        "trace": "preview/trace.json",
        "media": media,
    }


__all__ = [
    "ASSET_SCHEMA",
    "BACKGROUND_DEFAULT",
    "CompiledMotion3DAsset",
    "MOTION_3D_PREVIEW_PROFILE",
    "Motion3DPreviewProfile",
    "Motion3DRenderError",
    "TRACE_SCHEMA",
    "build_motion_3d_trace",
    "compile_motion_3d_asset",
    "render_motion_3d_preview",
]
