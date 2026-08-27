#!/usr/bin/env python3
"""Generate the reviewed keyframes for the classroom cone-section gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manim import Scene, tempconfig
from PIL import Image, ImageDraw

from examples.classroom_cone_sections.classroom_cone_sections import (
    BACKGROUND_COLOR,
    build_classroom_state,
    classroom_lesson_specs,
)


DEFAULT_OUTPUT = ROOT / "examples" / "classroom_cone_sections" / "gallery"
SCENE_PATH = (
    ROOT
    / "examples"
    / "classroom_cone_sections"
    / "classroom_cone_sections.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _draw_order(controller: object) -> tuple[str, ...]:
    boundary = controller.last_boundary_frame
    if boundary is not None:
        return tuple(boundary.draw_order)
    section = controller.last_section_frame
    if section is not None:
        return tuple(section.draw_order)
    frame = controller.last_frame
    return () if frame is None else tuple(frame.draw_order)


def _controller_evidence(label: str, controller: object) -> dict[str, object]:
    frame = controller.last_frame
    section = controller.last_section_frame
    boundary = controller.last_boundary_frame
    if frame is None:
        raise RuntimeError(f"controller {label!r} has no committed frame")
    role_counts: dict[str, int] = {}
    if section is not None:
        for fragment in section.plane_fragments:
            role = fragment.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
    boundary_kind_counts: dict[str, int] = {}
    if boundary is not None:
        for source in boundary.sources:
            kind = source.source_kind.value
            boundary_kind_counts[kind] = boundary_kind_counts.get(kind, 0) + 1
    return {
        "controller_id": label,
        "surface_count": len(frame.surface_items),
        "curve_fragment_count": len(frame.curve_fragments),
        "painted_curve_fragment_count": sum(
            item.painted for item in frame.curve_fragments
        ),
        "plane_fragment_count": (
            0 if section is None else len(section.plane_fragments)
        ),
        "plane_role_counts": role_counts,
        "boundary_source_counts": boundary_kind_counts,
        "painted_boundary_fragment_count": (
            0 if boundary is None else len(boundary.painted_fragments)
        ),
        "draw_order": list(_draw_order(controller)),
    }


def _capture(
    lesson_id: str,
    label: str,
    progress: float,
    *,
    output: Path,
    width: int,
    height: int,
    frame_rate: int,
) -> dict[str, object]:
    image_path = output / "keyframes" / lesson_id / f"{label}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="classroom-cone-keyframe-") as media_dir:
        with tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": width,
                "pixel_height": height,
                "frame_rate": frame_rate,
                "disable_caching": True,
                "media_dir": media_dir,
            }
        ):
            scene = Scene()
            state = build_classroom_state(
                scene,
                lesson_id,
                progress=progress,
                with_labels=True,
            )
            try:
                state.set_progress(progress)
                scene.camera.reset()
                scene.camera.capture_mobjects(scene.mobjects)
                pixels = scene.camera.pixel_array[:, :, :3].copy()
                Image.fromarray(pixels, mode="RGB").save(image_path)
                controllers = [
                    _controller_evidence(controller_id, controller)
                    for controller_id, controller in state.controllers
                ]
            finally:
                state.restore()
    return {
        "label": label,
        "progress": round(float(progress), 12),
        "path": image_path.relative_to(output).as_posix(),
        "width": width,
        "height": height,
        "sha256": _sha256(image_path),
        "controllers": controllers,
    }


def _rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))


def _contact_sheet(
    frames: Sequence[dict[str, object]],
    *,
    output: Path,
    lesson_id: str,
) -> dict[str, object]:
    tile_width = 480
    tile_height = 270
    label_height = 30
    sheet = Image.new(
        "RGB",
        (tile_width * len(frames), tile_height + label_height),
        _rgb(BACKGROUND_COLOR),
    )
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        source_path = output / str(frame["path"])
        with Image.open(source_path) as source:
            tile = source.convert("RGB").resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
        x = index * tile_width
        sheet.paste(tile, (x, 0))
        draw.text(
            (x + 9, tile_height + 8),
            str(frame["label"]).replace("-", " ").upper(),
            fill=(220, 230, 242),
        )
    path = output / "contact-sheets" / f"{lesson_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return {
        "path": path.relative_to(output).as_posix(),
        "width": sheet.width,
        "height": sheet.height,
        "sha256": _sha256(path),
    }


def generate(
    output: Path,
    *,
    width: int,
    height: int,
    frame_rate: int,
) -> dict[str, object]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    lessons = []
    for spec in classroom_lesson_specs():
        frames = [
            _capture(
                spec.lesson_id,
                keyframe.label,
                keyframe.progress,
                output=output,
                width=width,
                height=height,
                frame_rate=frame_rate,
            )
            for keyframe in spec.keyframes
        ]
        lessons.append(
            {
                "lesson_id": spec.lesson_id,
                "scene_name": spec.scene_name,
                "title": spec.title,
                "parameters": list(spec.parameters),
                "conclusion": spec.conclusion,
                "teacher_prompts": list(spec.teacher_prompts),
                "keyframes": frames,
                "contact_sheet": _contact_sheet(
                    frames,
                    output=output,
                    lesson_id=spec.lesson_id,
                ),
            }
        )
    manifest = {
        "schema": "manim-tikz-native-classroom-cone-sections/v1",
        "renderer": "cairo",
        "profile": {
            "pixel_width": width,
            "pixel_height": height,
            "frame_rate": frame_rate,
        },
        "scene_source": SCENE_PATH.relative_to(ROOT).as_posix(),
        "scene_source_sha256": _sha256(SCENE_PATH),
        "lessons": lessons,
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=8)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    manifest = generate(
        output,
        width=int(args.width),
        height=int(args.height),
        frame_rate=int(args.fps),
    )
    print(
        f"generated {sum(len(item['keyframes']) for item in manifest['lessons'])} "
        f"keyframes in {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
