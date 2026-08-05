from __future__ import annotations

import hashlib
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import manim
from manim import Scene, tempconfig

from .animation import semantic_animation_layers
from .compatibility import (
    DEFAULT_SUBSET_PATH,
    audit_document_compatibility,
    load_subset_spec,
)
from .compiler import DocumentSpec, PictureSpec, compile_document
from .fixed_view_renderer import NativeFixedViewRenderer
from .manim_renderer import NativeFigure
from .version import (
    ASSET_SCHEMA,
    PROTOCOL_VERSION,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    __version__,
    provider_revision,
)


ERROR_INPUT = "INPUT_ERROR"
ERROR_HASH_MISMATCH = "HASH_MISMATCH"
ERROR_COMPATIBILITY = "COMPATIBILITY_BLOCKED"
ERROR_INSTANTIATION = "INSTANTIATION_FAILED"
ERROR_RENDER = "RENDER_FAILED"
ERROR_PROVIDER = "PROVIDER_MISMATCH"


class TikzNativeProviderError(RuntimeError):
    def __init__(
        self,
        code: str,
        phase: str,
        message: str,
        *,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "phase": self.phase,
            "message": str(self),
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


@dataclass
class CompiledAsset:
    document: DocumentSpec
    picture: PictureSpec
    compatibility: dict[str, Any]
    selected_compatibility: dict[str, Any]
    figure: NativeFigure
    animation_plan: dict[str, Any]
    asset: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provider_info() -> dict[str, Any]:
    subset = load_subset_spec()
    return {
        "name": "tikz-native-manim",
        "version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "request_schema": REQUEST_SCHEMA,
        "response_schema": RESPONSE_SCHEMA,
        "asset_schema": ASSET_SCHEMA,
        "revision": provider_revision(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "manim_version": manim.__version__,
        "subset_versions": [subset["subset_version"]],
        "capabilities": {
            "compile_2d": True,
            "compile_3d_fixed_view": True,
            "semantic_object_ids": True,
            "semantic_animation_layers": True,
            "render_static": True,
            "dynamic_camera_in_fixed_view": False,
        },
    }


def _picture_by_report_index(
    document: DocumentSpec,
    picture_index: int,
) -> PictureSpec:
    matches = [picture for picture in document.pictures if picture.index == picture_index]
    if not matches:
        available = [picture.index for picture in document.pictures]
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "select_picture",
            f"picture_index {picture_index} is not available",
            details={"available_picture_indices": available},
        )
    return matches[0]


def _selected_compatibility(
    compatibility: dict[str, Any],
    picture_index: int,
) -> dict[str, Any]:
    selected = next(
        (
            picture
            for picture in compatibility.get("pictures", [])
            if picture.get("picture") == picture_index
        ),
        None,
    )
    if selected is None:
        raise TikzNativeProviderError(
            ERROR_PROVIDER,
            "compatibility",
            "compatibility report omitted the selected picture",
            details={"picture_index": picture_index},
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


def instantiate_picture(
    picture: PictureSpec | None = None,
    *,
    source: str | Path | None = None,
    source_path: str | Path | None = None,
    source_text: str | None = None,
    entry_macro: str | None = None,
    picture_index: int = 1,
    scene_unit_per_cm: float = 1.0,
    view_mode: str = "tikz_fixed",
) -> NativeFigure:
    """Instantiate one semantic picture for an editor runtime.

    Provider internals pass an already compiled ``PictureSpec``.  Generated
    Manim runtime templates may instead pass frozen ``source_text`` or a
    ``source``/``source_path``, entry macro, and report picture index.  All
    forms return the same ``NativeFigure`` with ``group`` and ``objects``.
    """

    supplied_sources = [value for value in (source, source_path) if value is not None]
    if picture is not None and (supplied_sources or source_text is not None):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "instantiate",
            "pass either a PictureSpec, source/source_path, or source_text",
        )
    if source_text is not None and supplied_sources:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "instantiate",
            "source_text cannot be combined with source/source_path",
        )
    if source_text is not None and not isinstance(source_text, str):
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "instantiate",
            "source_text must be a string",
        )
    if len(supplied_sources) > 1:
        first = Path(supplied_sources[0]).expanduser().resolve()
        second = Path(supplied_sources[1]).expanduser().resolve()
        if first != second:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "instantiate",
                "source and source_path refer to different files",
            )
    if picture is None:
        if not supplied_sources and source_text is None:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "instantiate",
                "a PictureSpec, source/source_path, or source_text is required",
            )
        try:
            if source_text is not None:
                runtime_document = compile_document(
                    source_text=source_text,
                    entry_macro=entry_macro,
                )
            else:
                selected_source = Path(supplied_sources[0]).expanduser().resolve()
                if not selected_source.is_file():
                    raise TikzNativeProviderError(
                        ERROR_INPUT,
                        "instantiate",
                        f"source file does not exist: {selected_source}",
                    )
                runtime_document = compile_document(
                    selected_source,
                    entry_macro=entry_macro,
                )
        except TikzNativeProviderError:
            raise
        except Exception as error:
            raise TikzNativeProviderError(
                ERROR_INPUT,
                "instantiate",
                f"TikZ-native compilation failed: {error}",
                details={"exception_type": type(error).__name__},
            ) from error
        picture = _picture_by_report_index(runtime_document, picture_index)

    if scene_unit_per_cm <= 0:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "instantiate",
            "scene_unit_per_cm must be positive",
        )
    if view_mode != "tikz_fixed":
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "instantiate",
            f"unsupported view_mode: {view_mode}",
        )
    try:
        renderer = NativeFixedViewRenderer(scene_unit_per_cm=scene_unit_per_cm)
        figure = renderer.render(picture)
    except TikzNativeProviderError:
        raise
    except Exception as error:
        raise TikzNativeProviderError(
            ERROR_INSTANTIATION,
            "instantiate",
            f"failed to instantiate picture {picture.index}: {error}",
            details={"exception_type": type(error).__name__},
        ) from error

    # The editor runtime can recover semantic children without depending on the
    # renderer's internal traversal order.
    figure.group._tikz_native_object_map = figure.objects  # type: ignore[attr-defined]
    figure.group._tikz_native_picture = picture  # type: ignore[attr-defined]
    figure.group._tikz_native_dimension = picture.dimension  # type: ignore[attr-defined]
    return figure


def _projection_payload(picture: PictureSpec) -> dict[str, Any] | None:
    projection = picture.projection_3d
    if projection is None:
        return None
    return {
        "source": projection.source,
        "matrix": [list(row) for row in projection.matrix],
        "x_basis_cm": list(projection.x_basis_cm),
        "y_basis_cm": list(projection.y_basis_cm),
        "z_basis_cm": list(projection.z_basis_cm),
        "azimuth_degrees": projection.azimuth_degrees,
        "elevation_degrees": projection.elevation_degrees,
    }


def _animation_plan(picture: PictureSpec) -> dict[str, Any]:
    return {
        "mode": "semantic_reveal",
        "picture": picture.index,
        "layers": [
            {
                "name": layer.name,
                "object_ids": list(layer.object_ids),
            }
            for layer in semantic_animation_layers(picture, include_empty=True)
        ],
        "note": (
            "Deterministic semantic baseline only; teaching order and timing "
            "remain explicit editor data."
        ),
    }


def _asset_payload(
    document: DocumentSpec,
    picture: PictureSpec,
    selected_compatibility: dict[str, Any],
    figure: NativeFigure,
    animation_plan: dict[str, Any],
    *,
    scene_unit_per_cm: float,
    view_mode: str,
) -> dict[str, Any]:
    center = figure.group.get_center()
    return {
        "schema": ASSET_SCHEMA,
        "provider": provider_info(),
        "source_sha256": document.source_sha256,
        "entry_macro": document.entry_macro,
        "picture_index": picture.index,
        "dimension": picture.dimension,
        "projection": _projection_payload(picture),
        "scene_unit_per_cm": scene_unit_per_cm,
        "view_mode": view_mode,
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
        "bounds": {
            "width_scene": float(figure.group.width),
            "height_scene": float(figure.group.height),
            "center_scene": [float(center[0]), float(center[1])],
        },
        "compatibility": selected_compatibility,
        "animation_plan": animation_plan,
        "warnings": list(figure.warnings),
        "files": {},
    }


def compile_asset(
    source_path: str | Path,
    *,
    source_sha256: str | None = None,
    entry_macro: str | None = None,
    picture_index: int = 1,
    subset_path: Path = DEFAULT_SUBSET_PATH,
    scene_unit_per_cm: float = 1.0,
    strict_native: bool = True,
    view_mode: str = "tikz_fixed",
) -> CompiledAsset:
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
            f"TikZ-native compilation failed: {error}",
            details={"exception_type": type(error).__name__},
        ) from error

    if source_sha256 is not None and document.source_sha256 != source_sha256:
        raise TikzNativeProviderError(
            ERROR_HASH_MISMATCH,
            "verify_input",
            "source SHA-256 does not match the requested snapshot",
            details={
                "expected": source_sha256,
                "actual": document.source_sha256,
            },
        )

    picture = _picture_by_report_index(document, picture_index)
    compatibility = audit_document_compatibility(document, subset_path)
    selected_compatibility = _selected_compatibility(
        compatibility,
        picture_index,
    )
    if strict_native and (
        selected_compatibility["static_status"] != "pass"
        or picture.unsupported
    ):
        raise TikzNativeProviderError(
            ERROR_COMPATIBILITY,
            "compatibility",
            f"picture {picture_index} is blocked by the strict native gate",
            details={
                "compatibility": selected_compatibility,
                "unsupported": list(picture.unsupported),
            },
        )

    figure = instantiate_picture(
        picture,
        scene_unit_per_cm=scene_unit_per_cm,
        view_mode=view_mode,
    )
    animation_plan = _animation_plan(picture)
    asset = _asset_payload(
        document,
        picture,
        selected_compatibility,
        figure,
        animation_plan,
        scene_unit_per_cm=scene_unit_per_cm,
        view_mode=view_mode,
    )
    return CompiledAsset(
        document=document,
        picture=picture,
        compatibility=compatibility,
        selected_compatibility=selected_compatibility,
        figure=figure,
        animation_plan=animation_plan,
        asset=asset,
    )


def render_static_png(
    compiled: CompiledAsset,
    output_path: str | Path,
    *,
    pixel_width: int = 1280,
    pixel_height: int = 720,
    frame_rate: int = 30,
    background: str = "#FFFFFF",
    transparent: bool = False,
    media_dir: str | Path | None = None,
) -> Path:
    if pixel_width <= 0 or pixel_height <= 0 or frame_rate <= 0:
        raise TikzNativeProviderError(
            ERROR_INPUT,
            "render_static",
            "pixel dimensions and frame_rate must be positive",
        )
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    render_media_dir = (
        Path(media_dir).expanduser().resolve()
        if media_dir is not None
        else target.parent / "media"
    )

    figure = compiled.figure

    class TikzNativeBridgeStaticScene(Scene):
        def construct(self) -> None:
            self.camera.background_color = background
            self.add(figure.group)

    config = {
        "media_dir": str(render_media_dir),
        "pixel_width": int(pixel_width),
        "pixel_height": int(pixel_height),
        "frame_rate": int(frame_rate),
        "format": "png",
        "save_last_frame": True,
        "write_to_movie": False,
        "disable_caching": True,
        "transparent": bool(transparent),
    }
    try:
        with tempconfig(config):
            scene = TikzNativeBridgeStaticScene()
            scene.render()
            rendered = Path(scene.renderer.file_writer.image_file_path)
            shutil.copy2(rendered, target)
    except Exception as error:
        raise TikzNativeProviderError(
            ERROR_RENDER,
            "render_static",
            f"static render failed: {error}",
            details={"exception_type": type(error).__name__},
        ) from error
    return target


__all__ = [
    "CompiledAsset",
    "ERROR_COMPATIBILITY",
    "ERROR_HASH_MISMATCH",
    "ERROR_INPUT",
    "ERROR_INSTANTIATION",
    "ERROR_PROVIDER",
    "ERROR_RENDER",
    "TikzNativeProviderError",
    "compile_asset",
    "instantiate_picture",
    "provider_info",
    "render_static_png",
    "sha256_file",
]
