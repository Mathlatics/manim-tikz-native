"""Analyze real Cairo keyframes produced by ``scene.py``.

The script computes renderer-neutral theory from the same public production
objects, then samples the already-rendered PNG files.  It never substitutes a
synthetic rasterizer for Cairo.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import sqrt
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from manim import Scene, config, tempconfig

from polyhedron_visibility.quadrics.compositing import QuadricPaintKind
from polyhedron_visibility.quadrics.curves import ParametricConicBranch
from polyhedron_visibility.quadrics.section_compositing import PlaneDepthRole
from polyhedron_visibility.quadrics.sections import compute_quadric_section
from polyhedron_visibility.quadrics.trace import section_trace_curves

import scene as diagnostic


ROLE_ORDER = (
    PlaneDepthRole.BEHIND_SURFACE,
    PlaneDepthRole.OUTSIDE_PROJECTION,
    PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
    PlaneDepthRole.IN_FRONT_OF_SURFACE,
)
ROLE_LABEL = {
    PlaneDepthRole.BEHIND_SURFACE: "behind",
    PlaneDepthRole.OUTSIDE_PROJECTION: "outside",
    PlaneDepthRole.BETWEEN_SURFACE_SHEETS: "between",
    PlaneDepthRole.IN_FRONT_OF_SURFACE: "front",
}


def _hex_rgb(value: str) -> np.ndarray:
    text = value.lstrip("#")
    return np.asarray(tuple(int(text[index : index + 2], 16) for index in (0, 2, 4)), dtype=float)


def _source_over(background: np.ndarray, foreground: np.ndarray, alpha: float) -> np.ndarray:
    return float(alpha) * foreground + (1.0 - float(alpha)) * background


def _expected_fill(role: PlaneDepthRole, mode: str) -> np.ndarray:
    style = diagnostic.style_for_mode(mode)
    background = _hex_rgb(diagnostic.BACKGROUND_COLOR)
    surface = _hex_rgb(diagnostic.SURFACE_COLOR)
    plane = _hex_rgb(diagnostic.PLANE_COLOR)
    combined = float(style.surface_fill_opacity)
    sheet = 1.0 - sqrt(max(0.0, 1.0 - combined))
    plane_alpha = float(style.section_plane_fill_opacity)
    result = background
    if role is PlaneDepthRole.OUTSIDE_PROJECTION:
        return _source_over(result, plane, plane_alpha)
    if role is PlaneDepthRole.BEHIND_SURFACE:
        result = _source_over(result, plane, plane_alpha)
        result = _source_over(result, surface, sheet)
        result = _source_over(result, surface, sheet)
        return result
    if role is PlaneDepthRole.BETWEEN_SURFACE_SHEETS:
        result = _source_over(result, surface, sheet)
        result = _source_over(result, plane, plane_alpha)
        result = _source_over(result, surface, sheet)
        return result
    if role is PlaneDepthRole.IN_FRONT_OF_SURFACE:
        result = _source_over(result, surface, sheet)
        result = _source_over(result, surface, sheet)
        result = _source_over(result, plane, plane_alpha)
        return result
    raise AssertionError(role)


def _expected_outline(role: PlaneDepthRole, mode: str) -> np.ndarray:
    style = diagnostic.style_for_mode(mode)
    background = _hex_rgb(diagnostic.BACKGROUND_COLOR)
    surface = _hex_rgb(diagnostic.SURFACE_COLOR)
    plane = _hex_rgb(diagnostic.PLANE_COLOR)
    outline = _hex_rgb(diagnostic.PLANE_OUTLINE_COLOR)
    combined = float(style.surface_fill_opacity)
    sheet = 1.0 - sqrt(max(0.0, 1.0 - combined))
    plane_alpha = float(style.section_plane_fill_opacity)
    outline_alpha = float(style.section_plane_stroke_opacity)
    result = background
    if role is PlaneDepthRole.OUTSIDE_PROJECTION:
        result = _source_over(result, plane, plane_alpha)
        return _source_over(result, outline, outline_alpha)
    if role is PlaneDepthRole.BEHIND_SURFACE:
        result = _source_over(result, plane, plane_alpha)
        result = _source_over(result, outline, outline_alpha)
        result = _source_over(result, surface, sheet)
        result = _source_over(result, surface, sheet)
        return result
    if role is PlaneDepthRole.BETWEEN_SURFACE_SHEETS:
        result = _source_over(result, surface, sheet)
        result = _source_over(result, plane, plane_alpha)
        result = _source_over(result, outline, outline_alpha)
        result = _source_over(result, surface, sheet)
        return result
    if role is PlaneDepthRole.IN_FRONT_OF_SURFACE:
        result = _source_over(result, surface, sheet)
        result = _source_over(result, surface, sheet)
        result = _source_over(result, plane, plane_alpha)
        return _source_over(result, outline, outline_alpha)
    raise AssertionError(role)


def _triangle_area(points: Sequence[Sequence[float]]) -> float:
    values = np.asarray(points, dtype=float)
    return 0.5 * abs(
        sum(
            values[index, 0] * values[(index + 1) % len(values), 1]
            - values[index, 1] * values[(index + 1) % len(values), 0]
            for index in range(len(values))
        )
    )


def _screen_to_pixel(
    point: Sequence[float],
    *,
    width: int,
    height: int,
    frame_width: float,
    frame_height: float,
) -> tuple[int, int]:
    x, y = (float(value) for value in point[:2])
    pixel_x = int(round((x / frame_width + 0.5) * (width - 1)))
    pixel_y = int(round((0.5 - y / frame_height) * (height - 1)))
    return (
        min(width - 1, max(0, pixel_x)),
        min(height - 1, max(0, pixel_y)),
    )


def _median_patch(image: np.ndarray, pixel: tuple[int, int], radius: int = 2) -> np.ndarray:
    x, y = pixel
    y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
    return np.median(image[y0:y1, x0:x1].reshape(-1, 3), axis=0)


def _nearest_patch(
    image: np.ndarray,
    pixel: tuple[int, int],
    target: np.ndarray,
    radius: int = 6,
) -> tuple[np.ndarray, tuple[int, int], float]:
    x, y = pixel
    y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
    patch = image[y0:y1, x0:x1].astype(float)
    distances = np.linalg.norm(patch - target.reshape(1, 1, 3), axis=2)
    local_y, local_x = np.unravel_index(int(np.argmin(distances)), distances.shape)
    actual = patch[local_y, local_x]
    actual_pixel = (x0 + int(local_x), y0 + int(local_y))
    return actual, actual_pixel, float(distances[local_y, local_x])


def _rgb(values: np.ndarray) -> list[int]:
    return [int(round(float(value))) for value in values]


def _active_curves(progress: float) -> tuple[ParametricConicBranch, ...]:
    trace = compute_quadric_section(
        "diagnostic-section",
        diagnostic.CONE,
        diagnostic.plane_at_progress(progress),
    )
    source = tuple(
        sorted(
            section_trace_curves(trace),
            key=lambda item: (item.domain.start, item.domain.end, item.curve_id),
        )
    )
    return tuple(
        ParametricConicBranch(
            diagnostic.ALLOCATED_CURVE_IDS[index],
            curve.parameterization,
            curve.plane_embedding,
            curve.domain,
        )
        for index, curve in enumerate(source)
    )


def _curve_sample_points(controller: object, progress: float) -> dict[str, tuple[float, float]]:
    frame = controller.last_frame
    if frame is None:
        return {}
    curves = {curve.curve_id: curve for curve in _active_curves(progress)}
    result: dict[str, tuple[float, float]] = {}
    candidates = sorted(
        (item for item in frame.curve_fragments if item.painted),
        key=lambda item: (
            item.kind.value,
            -(item.interval.length),
            item.item_id,
        ),
    )
    for fragment in candidates:
        label = (
            "visible_curve"
            if fragment.kind is QuadricPaintKind.VISIBLE_CURVE
            else "hidden_curve"
        )
        if label in result:
            continue
        curve = curves.get(fragment.curve_id)
        if curve is None:
            continue
        world = np.asarray(curve.point(fragment.interval.midpoint), dtype=float)
        projected = diagnostic.DIAGNOSTIC_VIEW.matrix[:2] @ world
        result[label] = (float(projected[0]), float(projected[1]))
    return result


def _frame_theory(mode: str, state_name: str, image_path: Path) -> tuple[dict[str, object], dict[str, tuple[int, int]]]:
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
        section_frame = controller.last_section_frame
        base_frame = controller.last_frame
        if section_frame is None or base_frame is None:
            raise RuntimeError("diagnostic controller did not produce a section frame")
        trace = compute_quadric_section(
            "diagnostic-section",
            diagnostic.CONE,
            diagnostic.plane_at_progress(progress),
        )
        frame_width = float(config.frame_width)
        frame_height = float(config.frame_height)

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]
        samples: dict[str, object] = {}
        annotation_pixels: dict[str, tuple[int, int]] = {}
        role_areas = {
            role: sum(
                _triangle_area(item.screen_vertices)
                for item in section_frame.fragments_by_role[role]
            )
            for role in ROLE_ORDER
        }
        inside_total = sum(
            role_areas[role]
            for role in ROLE_ORDER
            if role is not PlaneDepthRole.OUTSIDE_PROJECTION
        )

        for role in ROLE_ORDER:
            fragments = section_frame.fragments_by_role[role]
            if not fragments:
                continue
            fragment = max(fragments, key=lambda item: _triangle_area(item.screen_vertices))
            point = np.mean(np.asarray(fragment.screen_vertices, dtype=float), axis=0)
            pixel = _screen_to_pixel(
                point,
                width=width,
                height=height,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            expected = _expected_fill(role, mode)
            actual = _median_patch(image, pixel, radius=2)
            label = f"fill_{ROLE_LABEL[role]}"
            annotation_pixels[label] = pixel
            samples[label] = {
                "screen": [float(point[0]), float(point[1])],
                "pixel": list(pixel),
                "expectedRgb": _rgb(expected),
                "actualMedianRgb": _rgb(actual),
                "rgbErrorNorm": float(np.linalg.norm(actual - expected)),
            }

        for role in ROLE_ORDER:
            fragments = section_frame.outline_fragments_by_role[role]
            if not fragments:
                continue
            fragment = max(
                fragments,
                key=lambda item: float(
                    np.linalg.norm(
                        np.asarray(item.screen_end) - np.asarray(item.screen_start)
                    )
                ),
            )
            point = 0.5 * (
                np.asarray(fragment.screen_start, dtype=float)
                + np.asarray(fragment.screen_end, dtype=float)
            )
            pixel = _screen_to_pixel(
                point,
                width=width,
                height=height,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            expected = _expected_outline(role, mode)
            actual, actual_pixel, error = _nearest_patch(
                image,
                pixel,
                expected,
                radius=7,
            )
            label = f"outline_{ROLE_LABEL[role]}"
            annotation_pixels[label] = actual_pixel
            samples[label] = {
                "screen": [float(point[0]), float(point[1])],
                "nominalPixel": list(pixel),
                "sampledPixel": list(actual_pixel),
                "expectedRgb": _rgb(expected),
                "actualRgb": _rgb(actual),
                "rgbErrorNorm": error,
            }

        curve_points = _curve_sample_points(controller, progress)
        curve_targets = {
            "visible_curve": _hex_rgb(diagnostic.VISIBLE_CURVE_COLOR),
            "hidden_curve": _hex_rgb(diagnostic.HIDDEN_CURVE_COLOR),
        }
        for label, point in curve_points.items():
            pixel = _screen_to_pixel(
                point,
                width=width,
                height=height,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            actual, actual_pixel, error = _nearest_patch(
                image,
                pixel,
                curve_targets[label],
                radius=10,
            )
            annotation_pixels[label] = actual_pixel
            samples[label] = {
                "screen": list(point),
                "nominalPixel": list(pixel),
                "sampledPixel": list(actual_pixel),
                "targetRgb": _rgb(curve_targets[label]),
                "actualRgb": _rgb(actual),
                "distanceToAuthoredCurveColor": error,
            }

        result: dict[str, object] = {
            "mode": mode,
            "state": state_name,
            "stateDefinition": asdict(diagnostic.STATES[diagnostic.STATE_INDEX[state_name]]),
            "supportingKind": trace.supporting_kind.value,
            "finiteTopology": trace.finite_topology.value,
            "curveRecordCount": len(_active_curves(progress)),
            "visiblePaintFragmentCount": sum(
                item.painted and item.kind is QuadricPaintKind.VISIBLE_CURVE
                for item in base_frame.curve_fragments
            ),
            "hiddenPaintFragmentCount": sum(
                item.painted and item.kind is QuadricPaintKind.HIDDEN_CURVE
                for item in base_frame.curve_fragments
            ),
            "roleScreenAreas": {
                ROLE_LABEL[role]: role_areas[role] for role in ROLE_ORDER
            },
            "insideRoleFractions": {
                ROLE_LABEL[role]: (
                    role_areas[role] / inside_total if inside_total > 0.0 else 0.0
                )
                for role in ROLE_ORDER
                if role is not PlaneDepthRole.OUTSIDE_PROJECTION
            },
            "drawOrder": list(section_frame.draw_order),
            "zIndices": controller.active_painter_z_indices,
            "samples": samples,
            "image": str(image_path),
            "imageSize": [width, height],
            "frameSize": [frame_width, frame_height],
        }
        controller.restore()
        return result, annotation_pixels


def _annotate(
    source: Path,
    destination: Path,
    pixels: Mapping[str, tuple[int, int]],
) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = {
        "fill_behind": "#1D4ED8",
        "fill_outside": "#F97316",
        "fill_between": "#7C3AED",
        "fill_front": "#DC2626",
        "outline_behind": "#1E40AF",
        "outline_outside": "#EA580C",
        "outline_between": "#6D28D9",
        "outline_front": "#B91C1C",
        "visible_curve": diagnostic.VISIBLE_CURVE_COLOR,
        "hidden_curve": diagnostic.HIDDEN_CURVE_COLOR,
    }
    for label, (x, y) in pixels.items():
        color = palette.get(label, "#111827")
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
        draw.text((x + 7, y - 7), label, fill=color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def _contact_sheet(
    paths: Sequence[Path],
    labels: Sequence[str],
    destination: Path,
) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_height = 28
    sheet = Image.new("RGB", (width, (height + label_height) * len(images)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        top = index * (height + label_height)
        sheet.paste(image, (0, top + label_height))
        draw.text((10, top + 7), label, fill="#111827")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def _report_markdown(records: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# PR #12 Cairo diagnostic evidence",
        "",
        "All pixel values below come from PNG frames rendered by Manim Cairo.",
        "Renderer-neutral values are used only to select interior sample points and",
        "to compute the expected Porter–Duff source-over result.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['mode']} / {record['state']}",
                "",
                f"- supporting conic: `{record['supportingKind']}`",
                f"- finite topology: `{record['finiteTopology']}`",
                f"- visible painter fragments: {record['visiblePaintFragmentCount']}",
                f"- hidden dashed painter fragments: {record['hiddenPaintFragmentCount']}",
                f"- inside role fractions: `{json.dumps(record['insideRoleFractions'], sort_keys=True)}`",
                "",
                "| Sample | Expected/target RGB | Actual RGB | Error |",
                "|---|---:|---:|---:|",
            ]
        )
        samples = record["samples"]
        assert isinstance(samples, Mapping)
        for name, raw in samples.items():
            assert isinstance(raw, Mapping)
            expected = raw.get("expectedRgb", raw.get("targetRgb"))
            actual = raw.get("actualMedianRgb", raw.get("actualRgb"))
            error = raw.get("rgbErrorNorm", raw.get("distanceToAuthoredCurveColor"))
            lines.append(f"| `{name}` | `{expected}` | `{actual}` | `{float(error):.3f}` |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.artifact_dir.resolve()
    keyframes = root / "keyframes"
    annotated = root / "annotated"
    records: list[dict[str, object]] = []

    modes_and_states: list[tuple[str, str]] = [
        (mode, state.name)
        for mode in ("translucent", "opaque")
        for state in diagnostic.STATES
    ]
    modes_and_states.append(("surface_only", "exact_parabola"))

    for mode, state_name in modes_and_states:
        image_path = keyframes / f"{mode}_{state_name}.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        record, pixels = _frame_theory(mode, state_name, image_path)
        records.append(record)
        _annotate(
            image_path,
            annotated / f"{mode}_{state_name}_annotated.png",
            pixels,
        )

    evidence = {
        "schema": "pr12-cairo-diagnostic-evidence/v1",
        "mergeCommit": "a7811bf4f7d4adbf2e4078e30d647b40bea6693a",
        "prHead": "3a0d68443af95384367a48d76e01d930d2dff73c",
        "records": records,
    }
    (root / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "report.md").write_text(
        _report_markdown(records),
        encoding="utf-8",
    )

    for mode in ("translucent", "opaque"):
        paths = [keyframes / f"{mode}_{state.name}.png" for state in diagnostic.STATES]
        labels = [f"{mode}: {state.name}" for state in diagnostic.STATES]
        _contact_sheet(paths, labels, root / f"contact_sheet_{mode}.png")
        annotated_paths = [
            annotated / f"{mode}_{state.name}_annotated.png"
            for state in diagnostic.STATES
        ]
        _contact_sheet(
            annotated_paths,
            labels,
            root / f"contact_sheet_{mode}_annotated.png",
        )


if __name__ == "__main__":
    main()
