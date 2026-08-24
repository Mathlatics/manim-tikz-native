"""Supplement the Cairo evidence with proxy-clipped interior role samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image
from manim import Scene, config, tempconfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as base  # noqa: E402
import scene as diagnostic  # noqa: E402

from polyhedron_visibility.quadrics.section_compositing import PlaneDepthRole  # noqa: E402


ROLE_ORDER = (
    PlaneDepthRole.BEHIND_SURFACE,
    PlaneDepthRole.OUTSIDE_PROJECTION,
    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
    PlaneDepthRole.IN_FRONT_OF_SURFACE,
)


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _signed_area(points: Sequence[np.ndarray]) -> float:
    return 0.5 * sum(
        _cross2(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def _clip_convex(subject: Sequence[np.ndarray], clipper: Sequence[np.ndarray]) -> list[np.ndarray]:
    output = [np.asarray(item, dtype=float) for item in subject]
    clip = [np.asarray(item, dtype=float) for item in clipper]
    if _signed_area(clip) < 0.0:
        clip.reverse()
    epsilon = 1.0e-10
    for index, edge_start in enumerate(clip):
        edge_end = clip[(index + 1) % len(clip)]
        direction = edge_end - edge_start
        if not output:
            break
        input_values = output
        output = []
        previous = input_values[-1]
        previous_value = _cross2(direction, previous - edge_start)
        previous_inside = previous_value >= -epsilon
        for current in input_values:
            current_value = _cross2(direction, current - edge_start)
            current_inside = current_value >= -epsilon
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                if abs(denominator) > epsilon:
                    parameter = previous_value / denominator
                    output.append(previous + parameter * (current - previous))
            if current_inside:
                output.append(current)
            previous = current
            previous_value = current_value
            previous_inside = current_inside
    return output


def _polygon_centroid(points: Sequence[np.ndarray]) -> np.ndarray:
    values = [np.asarray(item, dtype=float) for item in points]
    area_twice = sum(
        _cross2(values[index], values[(index + 1) % len(values)])
        for index in range(len(values))
    )
    if abs(area_twice) <= 1.0e-14:
        return np.mean(values, axis=0)
    numerator = np.zeros(2, dtype=float)
    for index, first in enumerate(values):
        second = values[(index + 1) % len(values)]
        cross = _cross2(first, second)
        numerator += (first + second) * cross
    return numerator / (3.0 * area_twice)


def _proxy_polygon(section_frame: object) -> list[np.ndarray]:
    points = [
        np.asarray(item, dtype=float)
        for item in section_frame.surface_proxy.boundary_points
    ]
    if len(points) > 1 and float(np.linalg.norm(points[0] - points[-1])) <= 1.0e-10:
        points.pop()
    if _signed_area(points) < 0.0:
        points.reverse()
    return points


def _select_point(section_frame: object, role: PlaneDepthRole) -> tuple[np.ndarray, float]:
    fragments = section_frame.fragments_by_role[role]
    if not fragments:
        raise RuntimeError(f"no fragments for {role.value}")
    if role is PlaneDepthRole.OUTSIDE_PROJECTION:
        fragment = max(
            fragments,
            key=lambda item: base._triangle_area(item.screen_vertices),
        )
        return np.mean(np.asarray(fragment.screen_vertices, dtype=float), axis=0), base._triangle_area(fragment.screen_vertices)

    proxy = _proxy_polygon(section_frame)
    candidates: list[tuple[float, np.ndarray]] = []
    for fragment in fragments:
        clipped = _clip_convex(
            [np.asarray(item, dtype=float) for item in fragment.screen_vertices],
            proxy,
        )
        if len(clipped) < 3:
            continue
        area = abs(_signed_area(clipped))
        if area <= 1.0e-12:
            continue
        candidates.append((area, _polygon_centroid(clipped)))
    if not candidates:
        raise RuntimeError(f"no proxy-interior area for {role.value}")
    area, point = max(candidates, key=lambda item: item[0])
    return point, area


def _sample_record(mode: str, state_name: str, image_path: Path) -> dict[str, object]:
    progress = float(diagnostic.STATE_INDEX[state_name])
    with tempconfig(
        {
            "renderer": "cairo",
            "pixel_width": diagnostic.PIXEL_WIDTH,
            "pixel_height": diagnostic.PIXEL_HEIGHT,
            "frame_rate": diagnostic.FPS,
            "write_to_movie": False,
            "save_last_frame": False,
            "disable_caching": True,
        }
    ):
        scene = Scene()
        scene.camera.background_color = diagnostic.BACKGROUND_COLOR
        controller = diagnostic.build_controller(scene, lambda: progress, mode).attach()
        frame = controller.last_section_frame
        if frame is None:
            raise RuntimeError("section frame missing")
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]
        samples: dict[str, object] = {}
        for role in ROLE_ORDER:
            if not frame.fragments_by_role[role]:
                continue
            point, clipped_area = _select_point(frame, role)
            pixel = base._screen_to_pixel(
                point,
                width=width,
                height=height,
                frame_width=float(config.frame_width),
                frame_height=float(config.frame_height),
            )
            expected = base._expected_fill(role, mode)
            actual = base._median_patch(image, pixel, radius=2)
            label = f"fill_{base.ROLE_LABEL[role]}"
            samples[label] = {
                "screen": [float(point[0]), float(point[1])],
                "pixel": list(pixel),
                "clippedArea": float(clipped_area),
                "expectedRgb": base._rgb(expected),
                "actualMedianRgb": base._rgb(actual),
                "rgbErrorNorm": float(np.linalg.norm(actual - expected)),
            }
        controller.restore()
    return {"mode": mode, "state": state_name, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    records: list[dict[str, object]] = []
    for mode in ("translucent", "opaque"):
        for state in diagnostic.STATE_INDEX:
            records.append(
                _sample_record(
                    mode,
                    state,
                    root / "keyframes" / f"{mode}_{state}.png",
                )
            )
    records.append(
        _sample_record(
            "surface_only",
            "exact_parabola",
            root / "keyframes" / "surface_only_exact_parabola.png",
        )
    )
    payload = {
        "schema": "pr12-cairo-proxy-clipped-role-samples/v1",
        "prHead": "3a0d68443af95384367a48d76e01d930d2dff73c",
        "mergeCommit": "a7811bf4f7d4adbf2e4078e30d647b40bea6693a",
        "records": records,
    }
    (root / "role_samples_clipped.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Proxy-clipped interior Cairo samples",
        "",
        "Non-outside samples are selected only after each adaptive triangle is clipped to the rendered surface proxy.",
        "",
    ]
    for record in records:
        lines.extend((f"## {record['mode']} / {record['state']}", ""))
        lines.append("| Sample | Expected RGB | Actual RGB | Error |")
        lines.append("|---|---:|---:|---:|")
        for label, sample in record["samples"].items():
            lines.append(
                f"| `{label}` | `{sample['expectedRgb']}` | `{sample['actualMedianRgb']}` | `{sample['rgbErrorNorm']:.3f}` |"
            )
        lines.append("")
    (root / "role_samples_clipped.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
