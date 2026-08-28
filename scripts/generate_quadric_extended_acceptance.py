#!/usr/bin/env python3
"""Generate deterministic extended Cairo evidence for finite-cone sections.

The output is intentionally richer than a successful MP4.  Every reviewed
keyframe includes the renderer-neutral role classification, the complete
painter order, probe coordinates, expected and actual RGB values, and the
applied tolerance.  Motion sweeps additionally record fragment/ray counts and
fixed Manim identity hashes at every rendered progress sample.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import pi
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter, perf_counter_ns
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import manim
from manim import Scene, config, tempconfig
import numpy as np
from PIL import Image, ImageDraw

from examples.quadrics.extended_acceptance_demo import (
    BACKGROUND_COLOR,
    BOUNDARY_STYLES,
    STYLE,
    AcceptanceState,
    acceptance_limits,
    acceptance_scenario_ids,
    build_acceptance_state,
)
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundaryRenderIntent,
    BoundarySourceKind,
)
from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.section_compositing import PlaneDepthRole
from polyhedron_visibility.quadrics.performance import (
    QUADRIC_CAIRO_FRAME_TRACE_ENV,
    QUADRIC_CAIRO_FRAME_TRACE_SCHEMA,
    QUADRIC_PERFORMANCE_TRACE_ENV,
    QUADRIC_PERFORMANCE_TRACE_SCHEMA,
)

BASELINE_PATH = (
    ROOT / "tests" / "baselines" / "quadric-extended-acceptance-v1.json"
)
SCENE_PATH = ROOT / "examples" / "quadrics" / "extended_acceptance_demo.py"
MAX_MOTION_SWEEP_WORKERS = 2
OPEN_DOUBLE_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hex_rgb(value: str) -> np.ndarray:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected six-digit RGB, received {value!r}")
    return np.asarray(
        tuple(int(text[index : index + 2], 16) for index in (0, 2, 4)),
        dtype=float,
    )


def _source_over(background: np.ndarray, foreground: np.ndarray, alpha: float):
    return foreground * alpha + background * (1.0 - alpha)


def _expected_role_rgb(role: PlaneDepthRole) -> np.ndarray:
    background = _hex_rgb(BACKGROUND_COLOR)
    surface = _hex_rgb(STYLE.surface_fill_color)
    plane = _hex_rgb(STYLE.section_plane_fill_color)
    sheet_alpha = 1.0 - np.sqrt(1.0 - STYLE.surface_fill_opacity)
    plane_alpha = STYLE.section_plane_fill_opacity
    if role is PlaneDepthRole.OUTSIDE_PROJECTION:
        return _source_over(background, plane, plane_alpha)
    if role is PlaneDepthRole.BEHIND_SURFACE:
        value = _source_over(background, plane, plane_alpha)
        value = _source_over(value, surface, sheet_alpha)
        return _source_over(value, surface, sheet_alpha)
    if role is PlaneDepthRole.BETWEEN_SURFACE_SHEETS:
        value = _source_over(background, surface, sheet_alpha)
        value = _source_over(value, plane, plane_alpha)
        return _source_over(value, surface, sheet_alpha)
    if role is PlaneDepthRole.IN_FRONT_OF_SURFACE:
        value = _source_over(background, surface, sheet_alpha)
        value = _source_over(value, surface, sheet_alpha)
        return _source_over(value, plane, plane_alpha)
    raise AssertionError(role)


def _screen_to_pixel(point: Sequence[float]) -> tuple[int, int]:
    x, y = (float(value) for value in point[:2])
    column = int(
        round((x / float(config.frame_width) + 0.5) * (config.pixel_width - 1))
    )
    row = int(
        round((0.5 - y / float(config.frame_height)) * (config.pixel_height - 1))
    )
    return (
        min(int(config.pixel_height) - 1, max(0, row)),
        min(int(config.pixel_width) - 1, max(0, column)),
    )


def _nearest_rgb(
    pixels: np.ndarray,
    row: int,
    column: int,
    target: np.ndarray,
    radius: int,
) -> tuple[int, int, np.ndarray]:
    row_start = max(0, row - radius)
    row_end = min(len(pixels), row + radius + 1)
    column_start = max(0, column - radius)
    column_end = min(len(pixels[0]), column + radius + 1)
    patch = pixels[row_start:row_end, column_start:column_end].astype(float)
    distances = np.linalg.norm(patch - target, axis=2)
    local_row, local_column = np.unravel_index(
        int(np.argmin(distances)), distances.shape
    )
    selected_row = row_start + int(local_row)
    selected_column = column_start + int(local_column)
    return selected_row, selected_column, pixels[selected_row, selected_column]


def _area(vertices: Sequence[Sequence[float]]) -> float:
    points = np.asarray(vertices, dtype=float)
    if len(points) < 3:
        return 0.0
    return 0.5 * abs(
        sum(
            float(
                points[index, 0] * points[(index + 1) % len(points), 1]
                - points[index, 1] * points[(index + 1) % len(points), 0]
            )
            for index in range(len(points))
        )
    )


def _fill_probes(
    controller,
    pixels: np.ndarray,
    *,
    tolerance: float,
) -> list[dict[str, object]]:
    frame = controller.last_section_frame
    if frame is None:
        return []
    probes = []
    for role in PlaneDepthRole:
        candidates = tuple(item for item in frame.plane_fragments if item.role is role)
        if not candidates:
            continue
        fragment = max(candidates, key=lambda item: _area(item.screen_vertices))
        point = np.mean(np.asarray(fragment.screen_vertices, dtype=float), axis=0)
        row, column = _screen_to_pixel(point)
        expected = _expected_role_rgb(role)
        actual = pixels[row, column].astype(float)
        error = float(np.linalg.norm(actual - expected))
        probes.append(
            {
                "probe_id": f"fill:{role.value}",
                "theoretical_role": role.value,
                "fragment_id": fragment.fragment_id,
                "screen_point": [float(value) for value in point],
                "pixel": {"row": row, "column": column},
                "expected_rgb": [round(float(value), 6) for value in expected],
                "actual_rgb": [int(value) for value in actual],
                "rgb_euclidean_error": round(error, 6),
                "tolerance": tolerance,
                "passed": error <= tolerance,
            }
        )
    return probes


def _boundary_target(source, fragment) -> tuple[np.ndarray, float]:
    if source.source_kind in {
        BoundarySourceKind.ANALYTIC_CURVE,
        BoundarySourceKind.SECTION_CAP_CHORD,
    }:
        visible = _hex_rgb(STYLE.visible_curve_color)
        hidden = _hex_rgb(STYLE.hidden_curve_color)
        hidden_opacity = STYLE.hidden_curve_opacity
    elif source.source_kind is BoundarySourceKind.PLANE_PATCH_EDGE:
        visible = _hex_rgb(STYLE.section_plane_stroke_color)
        hidden = visible
        hidden_opacity = STYLE.section_plane_stroke_opacity
    else:
        visible = _hex_rgb("#5CE1E6")
        hidden = visible
        hidden_opacity = 0.24
    if fragment.render_intent is BoundaryRenderIntent.SOLID:
        return visible, 1.0
    if fragment.render_intent is BoundaryRenderIntent.DASHED:
        return hidden, hidden_opacity
    raise ValueError("omitted boundary fragments have no pixel probe")


def _boundary_probes(
    controller,
    projection,
    pixels: np.ndarray,
    *,
    tolerance: float,
    search_radius: int,
) -> list[dict[str, object]]:
    frame = controller.last_boundary_frame
    if frame is None:
        return []
    source_map = {item.source_id: item for item in frame.sources}
    best_by_kind: dict[BoundarySourceKind, dict[str, object]] = {}
    ordered_fragments = sorted(
        frame.painted_fragments,
        key=lambda item: (
            item.render_intent is BoundaryRenderIntent.DASHED,
            item.source_id,
            item.interval.start,
        ),
    )
    for fragment in ordered_fragments:
        source = source_map[fragment.source_id]
        parameter = 0.5 * (fragment.interval.start + fragment.interval.end)
        world = np.asarray(source.curve.point(parameter), dtype=float)
        screen = np.asarray(projection.matrix[:2] @ world, dtype=float)
        projected_row, projected_column = _screen_to_pixel(screen)
        ink, opacity = _boundary_target(source, fragment)
        target = _source_over(_hex_rgb(BACKGROUND_COLOR), ink, opacity)
        row, column, actual = _nearest_rgb(
            pixels,
            projected_row,
            projected_column,
            target,
            search_radius,
        )
        actual = actual.astype(float)
        error = float(np.linalg.norm(actual - target))
        candidate = {
            "probe_id": f"boundary:{source.source_kind.value}",
            "theoretical_role": (
                f"{source.source_kind.value}:"
                f"{fragment.render_intent.value}:"
                f"{fragment.plane_relation or 'no_plane_relation'}"
            ),
            "source_id": source.source_id,
            "fragment_id": fragment.item_id,
            "screen_point": [float(value) for value in screen],
            "pixel": {"row": row, "column": column},
            "projected_pixel": {
                "row": projected_row,
                "column": projected_column,
            },
            "pixel_offset_from_projected": {
                "row": row - projected_row,
                "column": column - projected_column,
            },
            "search_radius_pixels": search_radius,
            "expected_rgb": [round(float(value), 6) for value in target],
            "actual_rgb": [int(value) for value in actual],
            "rgb_euclidean_error": round(error, 6),
            "tolerance": tolerance,
            "passed": error <= tolerance,
        }
        previous = best_by_kind.get(source.source_kind)
        if previous is None or error < float(previous["rgb_euclidean_error"]):
            best_by_kind[source.source_kind] = candidate
    return [
        best_by_kind[kind]
        for kind in sorted(best_by_kind, key=lambda item: item.value)[:4]
    ]


def _controller_evidence(label: str, controller) -> dict[str, object]:
    base = controller.last_frame
    section = controller.last_section_frame
    boundary = controller.last_boundary_frame
    if base is None:
        raise RuntimeError(f"controller {label!r} has no committed frame")
    draw_order = (
        boundary.draw_order
        if boundary is not None
        else section.draw_order if section is not None else base.draw_order
    )
    role_counts = (
        {}
        if section is None
        else {
            role.value: sum(
                item.role is role for item in section.plane_fragments
            )
            for role in PlaneDepthRole
        }
    )
    boundary_source_map = (
        {} if boundary is None else {item.source_id: item for item in boundary.sources}
    )
    semantic_curve_fragments = (
        ()
        if boundary is None
        else tuple(
            item
            for item in boundary.fragments
            if boundary_source_map[item.source_id].source_kind
            in {
                BoundarySourceKind.ANALYTIC_CURVE,
                BoundarySourceKind.SECTION_CAP_CHORD,
            }
        )
    )
    cap_chords = tuple(
        item
        for item in semantic_curve_fragments
        if boundary_source_map[item.source_id].source_kind
        is BoundarySourceKind.SECTION_CAP_CHORD
        and item.painted
    )
    trim_sources = () if boundary is None else tuple(
        item
        for item in boundary.sources
        if item.source_kind is BoundarySourceKind.SURFACE_TRIM_RIM
    )
    performance = controller.performance_snapshot()
    performance_counts = {} if performance is None else performance.counts
    return {
        "controller_id": label,
        "paint_policy": controller.paint_policy.value,
        "draw_order": list(draw_order),
        "draw_order_digest": _semantic_digest(list(draw_order)),
        "surface_count": len(base.surface_items),
        "curve_fragment_count": (
            len(base.curve_fragments) + len(semantic_curve_fragments)
        ),
        "painted_curve_fragment_count": (
            sum(item.painted for item in base.curve_fragments)
            + sum(item.painted for item in semantic_curve_fragments)
        ),
        "active_cap_chord_fragment_count": len(cap_chords),
        "plane_fragment_count": 0 if section is None else len(section.plane_fragments),
        "plane_outline_fragment_count": (
            0 if section is None else len(section.plane_outline_fragments)
        ),
        "plane_role_counts": role_counts,
        "ray_classification_count": (
            0 if section is None else section.ray_classification_count
        ),
        "boundary_source_count": 0 if boundary is None else len(boundary.sources),
        "boundary_fragment_count": (
            0 if boundary is None else len(boundary.fragments)
        ),
        "painted_boundary_fragment_count": (
            0 if boundary is None else len(boundary.painted_fragments)
        ),
        "trim_rim_source_count": len(trim_sources),
        "mobject_total_count": int(
            performance_counts.get("mobject_family_count", 0)
        ),
        "active_mobject_count": int(
            performance_counts.get("active_mobject_count", 0)
        ),
        "modified_mobject_count": int(
            performance_counts.get("modified_mobject_count", 0)
        ),
        "cache_hit_count": (
            0 if performance is None else sum(performance.cache_hits.values())
        ),
        "cache_miss_count": (
            0 if performance is None else sum(performance.cache_misses.values())
        ),
        "performance": (
            None if performance is None else performance.to_dict()
        ),
    }


@dataclass(frozen=True, slots=True)
class CaptureResult:
    image_path: Path
    evidence: dict[str, object]
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _MotionSweepTask:
    scenario_id: str
    sample_count: int
    output: str
    critical_progresses: tuple[float, ...]


def _capture_keyframe(
    scenario_id: str,
    progress: float,
    *,
    output: Path,
    profile: Mapping[str, object],
    pixel_policy: Mapping[str, object],
) -> CaptureResult:
    started = perf_counter()
    frame_token = f"p{int(round(progress * 1000.0)):04d}"
    keyframe_dir = output / "keyframes" / scenario_id
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    image_path = keyframe_dir / f"{frame_token}.png"
    with tempconfig(
        {
            "renderer": "cairo",
            "pixel_width": int(profile["pixel_width"]),
            "pixel_height": int(profile["pixel_height"]),
            "frame_rate": int(profile["frame_rate"]),
            "disable_caching": True,
            "media_dir": str(output / "_static-media"),
        }
    ):
        scene = Scene()
        state = build_acceptance_state(
            scene,
            scenario_id,
            progress=progress,
            with_labels=True,
        )
        # Every builder returns a state already committed at ``progress``.
        # Keeping that exact attach/update snapshot avoids replacing useful
        # mutation evidence with a redundant same-value update.
        scene.camera.reset()
        cairo_started_ns = perf_counter_ns()
        scene.camera.capture_mobjects(scene.mobjects)
        cairo_render_ns = max(0, perf_counter_ns() - cairo_started_ns)
        pixels = scene.camera.pixel_array[:, :, :3].astype(np.uint8).copy()
        Image.fromarray(pixels, mode="RGB").save(image_path)
        controllers = []
        probes = []
        rows = []
        for index, (label, controller) in enumerate(state.controllers):
            item = _controller_evidence(label, controller)
            controllers.append(item)
            probes.extend(
                {
                    "controller_id": label,
                    **probe,
                }
                for probe in _fill_probes(
                    controller,
                    pixels,
                    tolerance=float(
                        pixel_policy["fill_rgb_euclidean_tolerance"]
                    ),
                )
            )
            probes.extend(
                {
                    "controller_id": label,
                    **probe,
                }
                for probe in _boundary_probes(
                    controller,
                    state.projections[index](),
                    pixels,
                    tolerance=float(
                        pixel_policy["boundary_rgb_euclidean_tolerance"]
                    ),
                    search_radius=int(
                        pixel_policy["probe_search_radius_pixels"]
                    ),
                )
            )
            rows.append(
                {
                    "scenario": scenario_id,
                    "keyframe": frame_token,
                    "progress": progress,
                    **{
                        name: item[name]
                        for name in (
                            "controller_id",
                            "surface_count",
                            "curve_fragment_count",
                            "painted_curve_fragment_count",
                            "active_cap_chord_fragment_count",
                            "plane_fragment_count",
                            "plane_outline_fragment_count",
                            "ray_classification_count",
                            "boundary_source_count",
                            "boundary_fragment_count",
                            "painted_boundary_fragment_count",
                            "trim_rim_source_count",
                            "mobject_total_count",
                            "active_mobject_count",
                            "modified_mobject_count",
                            "cache_hit_count",
                            "cache_miss_count",
                        )
                    },
                }
            )
        state.restore()
    failed = tuple(item for item in probes if not item["passed"])
    if failed:
        labels = ", ".join(
            f"{item['probe_id']} error={item['rgb_euclidean_error']} "
            f"actual={item['actual_rgb']} expected={item['expected_rgb']}"
            for item in failed
        )
        raise RuntimeError(
            f"{scenario_id} progress {progress:.6g} failed RGB probes: {labels}"
        )
    elapsed = perf_counter() - started
    evidence = {
        "scenario_id": scenario_id,
        "keyframe_id": frame_token,
        "progress": progress,
        "image": str(image_path.relative_to(output)),
        "image_sha256": _sha256(image_path),
        "elapsed_seconds": round(elapsed, 6),
        "cairo_render": {
            "nanoseconds": cairo_render_ns,
            "seconds": cairo_render_ns / 1_000_000_000.0,
        },
        "controllers": controllers,
        "key_pixels": probes,
    }
    return CaptureResult(image_path, evidence, tuple(rows))


def _capture_open_double_shared_apex(
    *,
    output: Path,
    profile: Mapping[str, object],
    pixel_policy: Mapping[str, object],
    specification: Mapping[str, object],
) -> CaptureResult:
    """Capture one full-resolution certificate for the composite apex contact."""

    started = perf_counter()
    image_path = output / str(specification["full_frame"])
    zoom_path = output / str(specification["zoom"])
    trace_path = output / str(specification["painter_trace"])
    evidence_path = output / str(specification["pixel_evidence"])
    for path in (image_path, zoom_path, trace_path, evidence_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    controller = None
    with tempconfig(
        {
            "renderer": "cairo",
            "pixel_width": int(profile["pixel_width"]),
            "pixel_height": int(profile["pixel_height"]),
            "frame_rate": int(profile["frame_rate"]),
            "disable_caching": True,
            "media_dir": str(output / "_static-media" / "open-double-apex"),
        }
    ):
        scene = Scene()
        scene.camera.background_color = BACKGROUND_COLOR
        cone = ConeSpec(
            "acceptance:open-double-apex:cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (-2.15, 2.15),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_DOUBLE,
        )
        plane = SectionPlane(
            "acceptance:open-double-apex:plane",
            (0.0, 0.48, 0.0),
            (0.0, 1.0, 0.16),
            u_axis=(1.0, 0.0, 0.0),
        )
        controller = CompositeQuadricSection3D(
            scene,
            surface=cone,
            section_id="acceptance:open-double-apex:section",
            plane=plane,
            projection=OPEN_DOUBLE_VIEW,
            paint_policy="depth_aware_diagrammatic",
            style=STYLE,
            boundary_styles=BOUNDARY_STYLES,
            limits=acceptance_limits(),
            max_chord_error=0.008,
            section_max_screen_error=0.08,
            plane_patch_margin=0.17,
        ).attach()
        try:
            scene.camera.reset()
            cairo_started_ns = perf_counter_ns()
            scene.camera.capture_mobjects(scene.mobjects)
            cairo_render_ns = max(0, perf_counter_ns() - cairo_started_ns)
            pixels = scene.camera.pixel_array[:, :, :3].astype(np.uint8).copy()
            Image.fromarray(pixels, mode="RGB").save(image_path)

            frame = controller.last_composite_frame
            boundary = controller.last_boundary_frame
            if frame is None or boundary is None:
                raise RuntimeError("open-double controller did not commit its frames")
            apex_row, apex_column = _screen_to_pixel(frame.shared_apex.screen_point)
            half_width = int(specification["apex_zoom_half_width_pixels"])
            half_height = int(specification["apex_zoom_half_height_pixels"])
            left = max(0, apex_column - half_width)
            right = min(int(config.pixel_width), apex_column + half_width)
            top = max(0, apex_row - half_height)
            bottom = min(int(config.pixel_height), apex_row + half_height)
            with Image.open(image_path) as full_frame:
                zoom = full_frame.convert("RGB").crop((left, top, right, bottom))
                zoom = zoom.resize(
                    (2 * (right - left), 2 * (bottom - top)),
                    Image.Resampling.LANCZOS,
                )
                zoom.save(zoom_path)

            background = _hex_rgb(BACKGROUND_COLOR)
            plane_color = _hex_rgb(STYLE.section_plane_fill_color)
            expected_plane = _source_over(
                background,
                plane_color,
                STYLE.section_plane_fill_opacity,
            )
            plane_row, plane_column = _screen_to_pixel((2.25, 0.0))
            actual_plane = pixels[plane_row, plane_column].astype(float)
            plane_error = float(np.linalg.norm(actual_plane - expected_plane))
            apex_rgb = pixels[apex_row, apex_column].astype(float)
            apex_background_distance = float(np.linalg.norm(apex_rgb - background))

            visible_curve = _hex_rgb(STYLE.visible_curve_color)
            boundary_ink = _hex_rgb(BOUNDARY_STYLES[
                "style:surface-boundary"
            ].visible_color)
            rgb = pixels.astype(float)
            yellow_mask = np.linalg.norm(rgb - visible_curve, axis=2) <= 80.0
            cyan_mask = np.linalg.norm(rgb - boundary_ink, axis=2) <= 80.0
            midpoint = len(pixels) // 2
            yellow_upper = int(np.count_nonzero(yellow_mask[:midpoint]))
            yellow_lower = int(np.count_nonzero(yellow_mask[midpoint:]))
            cyan_count = int(np.count_nonzero(cyan_mask))
            plane_tolerance = min(
                24.0,
                float(pixel_policy["fill_rgb_euclidean_tolerance"]),
            )
            probes = [
                {
                    "probe_id": "plane:single-alpha-outside-nappes",
                    "pixel": {"row": plane_row, "column": plane_column},
                    "expected_rgb": [
                        round(float(value), 6) for value in expected_plane
                    ],
                    "actual_rgb": [int(value) for value in actual_plane],
                    "rgb_euclidean_error": round(plane_error, 6),
                    "tolerance": plane_tolerance,
                    "passed": plane_error <= plane_tolerance,
                },
                {
                    "probe_id": "shared-apex:non-background",
                    "pixel": {"row": apex_row, "column": apex_column},
                    "background_rgb": [int(value) for value in background],
                    "actual_rgb": [int(value) for value in apex_rgb],
                    "background_distance": round(apex_background_distance, 6),
                    "minimum_distance": float(
                        specification["apex_background_distance_min"]
                    ),
                    "passed": apex_background_distance
                    >= float(specification["apex_background_distance_min"]),
                },
                {
                    "probe_id": "section-branch:negative-nappe-yellow-pixels",
                    "pixel_count": yellow_upper,
                    "minimum_count": int(
                        specification["yellow_pixels_per_nappe_min"]
                    ),
                    "passed": yellow_upper
                    >= int(specification["yellow_pixels_per_nappe_min"]),
                },
                {
                    "probe_id": "section-branch:positive-nappe-yellow-pixels",
                    "pixel_count": yellow_lower,
                    "minimum_count": int(
                        specification["yellow_pixels_per_nappe_min"]
                    ),
                    "passed": yellow_lower
                    >= int(specification["yellow_pixels_per_nappe_min"]),
                },
                {
                    "probe_id": "surface-boundary:cyan-pixels",
                    "pixel_count": cyan_count,
                    "minimum_count": int(specification["cyan_pixels_min"]),
                    "passed": cyan_count >= int(specification["cyan_pixels_min"]),
                },
            ]

            trace = {
                "schema": "manim-open-double-shared-apex-painter-trace/v1",
                "composite_frame": frame.to_dict(),
                "boundary_frame": boundary.to_dict(),
                "active_painter_z_indices": controller.active_painter_z_indices,
                "slot_identity_digest": _semantic_digest(
                    [int(value) for value in controller.slot_identities()]
                ),
            }
            _atomic_json(trace_path, trace)

            source_map = {item.source_id: item for item in boundary.sources}
            semantic_fragments = tuple(
                item
                for item in boundary.fragments
                if source_map[item.source_id].source_kind
                in {
                    BoundarySourceKind.ANALYTIC_CURVE,
                    BoundarySourceKind.SECTION_CAP_CHORD,
                }
            )
            trim_sources = tuple(
                item
                for item in boundary.sources
                if item.source_kind is BoundarySourceKind.SURFACE_TRIM_RIM
            )
            performance_snapshot = controller.performance_snapshot()
            performance_counts = (
                {} if performance_snapshot is None else performance_snapshot.counts
            )
            row = {
                "scenario": "open_double_shared_apex",
                "keyframe": "p0000",
                "progress": 0.0,
                "controller_id": "composite-open-double",
                "surface_count": len(frame.child_frames),
                "curve_fragment_count": len(semantic_fragments),
                "painted_curve_fragment_count": sum(
                    item.painted for item in semantic_fragments
                ),
                "active_cap_chord_fragment_count": sum(
                    item.painted
                    and source_map[item.source_id].source_kind
                    is BoundarySourceKind.SECTION_CAP_CHORD
                    for item in semantic_fragments
                ),
                "plane_fragment_count": len(frame.plane_fragments),
                "plane_outline_fragment_count": len(frame.plane_outline_fragments),
                "ray_classification_count": sum(
                    item.ray_classification_count for item in frame.child_frames
                ),
                "boundary_source_count": len(boundary.sources),
                "boundary_fragment_count": len(boundary.fragments),
                "painted_boundary_fragment_count": len(boundary.painted_fragments),
                "trim_rim_source_count": len(trim_sources),
                "mobject_total_count": int(
                    performance_counts.get("mobject_family_count", 0)
                ),
                "active_mobject_count": int(
                    performance_counts.get("active_mobject_count", 0)
                ),
                "modified_mobject_count": int(
                    performance_counts.get("modified_mobject_count", 0)
                ),
                "cache_hit_count": (
                    0
                    if performance_snapshot is None
                    else sum(performance_snapshot.cache_hits.values())
                ),
                "cache_miss_count": (
                    0
                    if performance_snapshot is None
                    else sum(performance_snapshot.cache_misses.values())
                ),
            }
            evidence = {
                "schema": "manim-open-double-shared-apex-evidence/v1",
                "scenario_id": "open_double_shared_apex",
                "full_frame": str(image_path.relative_to(output)),
                "full_frame_sha256": _sha256(image_path),
                "zoom": str(zoom_path.relative_to(output)),
                "zoom_sha256": _sha256(zoom_path),
                "zoom_source_box": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
                "painter_trace": str(trace_path.relative_to(output)),
                "shared_apex": frame.shared_apex.to_dict(),
                "branch_lineage": [item.to_dict() for item in frame.branch_lineage],
                "draw_order": list(boundary.draw_order),
                "counts": row,
                "performance": (
                    None
                    if performance_snapshot is None
                    else performance_snapshot.to_dict()
                ),
                "cairo_render": {
                    "nanoseconds": cairo_render_ns,
                    "seconds": cairo_render_ns / 1_000_000_000.0,
                },
                "key_pixels": probes,
                "elapsed_seconds": round(perf_counter() - started, 6),
            }
            failed = tuple(item for item in probes if not item["passed"])
            _atomic_json(
                evidence_path,
                {
                    **evidence,
                    "status": "failed" if failed else "passed",
                },
            )
            if failed:
                labels = ", ".join(str(item["probe_id"]) for item in failed)
                raise RuntimeError(
                    "open-double shared-apex RGB evidence failed: " + labels
                )
            return CaptureResult(image_path, evidence, (row,))
        finally:
            controller.restore()


def _motion_sweep(
    scenario_id: str,
    sample_count: int,
    *,
    output: Path,
    critical_progresses: Sequence[float] = (),
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    started = perf_counter()
    with tempconfig(
        {
            "renderer": "cairo",
            "pixel_width": 320,
            "pixel_height": 180,
            "frame_rate": 8,
            "disable_caching": True,
            "media_dir": str(output / "_sweep-media" / scenario_id),
        }
    ):
        scene = Scene()
        state = build_acceptance_state(scene, scenario_id, with_labels=False)
        scene_identity = tuple(id(item) for item in scene.mobjects)
        slot_identities = tuple(
            (label, controller.slot_identities())
            for label, controller in state.controllers
        )
        identity_digest = _semantic_digest(
            [
                [label, [int(value) for value in identities]]
                for label, identities in slot_identities
            ]
        )
        samples = []
        rows = []
        progresses = sorted(
            {
                *(float(value) for value in np.linspace(0.0, 1.0, sample_count)),
                *(float(value) for value in critical_progresses),
            }
        )
        for progress in progresses:
            sample_started = perf_counter()
            state.set_progress(float(progress))
            if tuple(id(item) for item in scene.mobjects) != scene_identity:
                raise RuntimeError(
                    f"{scenario_id} changed scene.mobjects during motion sweep"
                )
            current_identities = tuple(
                (label, controller.slot_identities())
                for label, controller in state.controllers
            )
            if current_identities != slot_identities:
                raise RuntimeError(
                    f"{scenario_id} replaced a fixed Manim slot during motion sweep"
                )
            controllers = [
                _controller_evidence(label, controller)
                for label, controller in state.controllers
            ]
            samples.append(
                {
                    "progress": round(float(progress), 12),
                    "elapsed_seconds": round(perf_counter() - sample_started, 6),
                    "controllers": controllers,
                }
            )
            for item in controllers:
                rows.append(
                    {
                        "scenario": scenario_id,
                        "keyframe": "motion",
                        "progress": round(float(progress), 12),
                        **{
                            name: item[name]
                            for name in (
                                "controller_id",
                                "surface_count",
                                "curve_fragment_count",
                                "painted_curve_fragment_count",
                                "active_cap_chord_fragment_count",
                                "plane_fragment_count",
                                "plane_outline_fragment_count",
                                "ray_classification_count",
                                "boundary_source_count",
                                "boundary_fragment_count",
                                "painted_boundary_fragment_count",
                                "trim_rim_source_count",
                                "mobject_total_count",
                                "active_mobject_count",
                                "modified_mobject_count",
                                "cache_hit_count",
                                "cache_miss_count",
                            )
                        },
                    }
                )
        state.restore()
    return (
        {
            "scenario_id": scenario_id,
            "output_frame_sample_count": sample_count,
            "critical_progresses": [float(value) for value in critical_progresses],
            "sample_count": len(samples),
            "slot_identity_digest": identity_digest,
            "elapsed_seconds": round(perf_counter() - started, 6),
            "samples": samples,
        },
        tuple(rows),
    )


def _contact_sheet(
    image_paths: Sequence[Path],
    labels: Sequence[str],
    output_path: Path,
) -> None:
    if not image_paths:
        raise ValueError("a contact sheet requires at least one image")
    tile_width = 480
    tile_height = 270
    label_height = 28
    sheet = Image.new(
        "RGB",
        (tile_width * len(image_paths), tile_height + label_height),
        tuple(int(value) for value in _hex_rgb(BACKGROUND_COLOR)),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(zip(image_paths, labels)):
        with Image.open(path) as source:
            tile = source.convert("RGB").resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
        x = index * tile_width
        sheet.paste(tile, (x, 0))
        draw.text((x + 8, tile_height + 7), label, fill=(220, 230, 242))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _render_video(
    scenario: Mapping[str, object],
    *,
    output: Path,
    profile: Mapping[str, object],
    budgets: Mapping[str, object],
) -> dict[str, object]:
    scenario_id = str(scenario["id"])
    scene_name = str(scenario["video_scene"])
    media_dir = output / "_video-media" / scenario_id
    video_dir = output / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_stem = scenario_id.replace("_", "-")
    command = [
        sys.executable,
        "-m",
        "manim",
        "--renderer",
        "cairo",
        "--disable_caching",
        "--format",
        "mp4",
        "--fps",
        str(int(profile["frame_rate"])),
        "-r",
        f"{int(profile['pixel_width'])},{int(profile['pixel_height'])}",
        "--media_dir",
        str(media_dir),
        "--output_file",
        output_stem,
        str(SCENE_PATH),
        scene_name,
    ]
    started = perf_counter()
    log_path = output / "logs" / f"{scenario_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frame_trace_path = (
        output / "evidence" / "video-frame-performance" / f"{scenario_id}.json"
    )
    frame_trace_path.parent.mkdir(parents=True, exist_ok=True)
    frame_trace_path.unlink(missing_ok=True)
    child_environment = os.environ.copy()
    child_environment[QUADRIC_PERFORMANCE_TRACE_ENV] = "1"
    child_environment[QUADRIC_CAIRO_FRAME_TRACE_ENV] = str(frame_trace_path)
    timeout_seconds = float(budgets["single_video_seconds"])
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=child_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or ""
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", errors="replace")
        log_path.write_text(captured, encoding="utf-8")
        raise RuntimeError(
            f"Manim timed out after {timeout_seconds:g}s for {scenario_id}; "
            f"inspect {log_path.relative_to(output)}"
        ) from exc
    elapsed = perf_counter() - started
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"Manim failed for {scenario_id}; inspect {log_path.relative_to(output)}"
        )
    matches = tuple(media_dir.rglob(f"{output_stem}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {output_stem}.mp4, found {len(matches)}"
        )
    destination = video_dir / f"{output_stem}.mp4"
    shutil.copy2(matches[0], destination)
    if not frame_trace_path.is_file():
        raise RuntimeError(
            f"Manim did not publish per-frame performance for {scenario_id}"
        )
    frame_trace = _read_json(frame_trace_path)
    if frame_trace.get("schema") != QUADRIC_CAIRO_FRAME_TRACE_SCHEMA:
        raise RuntimeError(
            f"invalid Cairo performance trace schema for {scenario_id}"
        )
    frames = frame_trace.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(
            f"invalid Cairo performance frames for {scenario_id}"
        )
    frame_count = int(frame_trace.get("frameCount", 0))
    if frame_count <= 0 or frame_count != len(frames):
        raise RuntimeError(
            f"invalid Cairo performance frame count for {scenario_id}"
        )
    return {
        "scenario_id": scenario_id,
        "scene": scene_name,
        "path": str(destination.relative_to(output)),
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "elapsed_seconds": round(elapsed, 6),
        "log": str(log_path.relative_to(output)),
        "frame_performance": str(frame_trace_path.relative_to(output)),
        "frame_performance_sha256": _sha256(frame_trace_path),
        "frame_count": frame_count,
        "cairo_render_seconds": (
            int(frame_trace["cairoRenderNanoseconds"]) / 1_000_000_000.0
        ),
    }


def _write_counts_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = tuple(rows)
    fieldnames = (
        "scenario",
        "keyframe",
        "progress",
        "controller_id",
        "surface_count",
        "curve_fragment_count",
        "painted_curve_fragment_count",
        "active_cap_chord_fragment_count",
        "plane_fragment_count",
        "plane_outline_fragment_count",
        "ray_classification_count",
        "boundary_source_count",
        "boundary_fragment_count",
        "painted_boundary_fragment_count",
        "trim_rim_source_count",
        "mobject_total_count",
        "active_mobject_count",
        "modified_mobject_count",
        "cache_hit_count",
        "cache_miss_count",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _execute_motion_sweep(
    task: _MotionSweepTask,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Run one isolated sweep and persist evidence even if another task fails."""

    output = Path(task.output)
    evidence_directory = output / "evidence" / "motion-sweeps"
    evidence_path = evidence_directory / f"{task.scenario_id}.json"
    counts_path = evidence_directory / f"{task.scenario_id}.csv"
    started = perf_counter()
    try:
        sweep, rows = _motion_sweep(
            task.scenario_id,
            task.sample_count,
            output=output,
            critical_progresses=task.critical_progresses,
        )
        _write_counts_csv(counts_path, rows)
        _atomic_json(
            evidence_path,
            {
                "schema": "manim-quadric-motion-sweep-evidence/v1",
                "status": "passed",
                "scenario_id": task.scenario_id,
                "elapsed_seconds": round(perf_counter() - started, 6),
                "counts_csv": str(counts_path.relative_to(output)),
                "sweep": sweep,
            },
        )
        return sweep, rows
    except Exception as exc:
        _atomic_json(
            evidence_path,
            {
                "schema": "manim-quadric-motion-sweep-evidence/v1",
                "status": "failed",
                "scenario_id": task.scenario_id,
                "elapsed_seconds": round(perf_counter() - started, 6),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def _run_motion_sweeps(
    scenarios: Sequence[Mapping[str, object]],
    *,
    output: Path,
    workers: int,
) -> tuple[tuple[dict[str, object], tuple[dict[str, object], ...]], ...]:
    """Run independent scenarios in a bounded spawn-based process pool."""

    if isinstance(workers, bool) or workers <= 0:
        raise ValueError("motion sweep worker count must be positive")
    if workers > MAX_MOTION_SWEEP_WORKERS:
        raise ValueError(
            "motion sweep worker count must not exceed "
            f"{MAX_MOTION_SWEEP_WORKERS}"
        )
    tasks = tuple(
        _MotionSweepTask(
            scenario_id=str(scenario["id"]),
            sample_count=int(scenario["motion_samples"]),
            output=str(output),
            critical_progresses=tuple(
                float(value)
                for value in scenario.get("critical_progresses", ())
            ),
        )
        for scenario in scenarios
    )
    if not tasks:
        return ()
    worker_count = min(workers, len(tasks))
    if worker_count == 1:
        return tuple(_execute_motion_sweep(task) for task in tasks)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
    ) as executor:
        return tuple(executor.map(_execute_motion_sweep, tasks))


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def _artifact_manifest(output: Path) -> dict[str, object]:
    ignored = {"artifact-manifest.json", "run-status.json"}
    artifacts = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output))
        if relative in ignored or relative.startswith("_video-media/") or relative.startswith("_static-media/") or relative.startswith("_sweep-media/"):
            continue
        artifacts.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": "manim-quadric-acceptance-artifacts/v1",
        "artifacts": artifacts,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=acceptance_scenario_ids(),
        help="Generate only selected scenarios; repeat for more than one.",
    )
    parser.add_argument(
        "--render-videos",
        action="store_true",
        help="Render the complete 960x540 MP4 acceptance scenes.",
    )
    parser.add_argument(
        "--skip-motion-sweeps",
        action="store_true",
        help="Skip the per-frame fixed-identity progress scan.",
    )
    parser.add_argument(
        "--motion-sweep-workers",
        type=int,
        default=1,
        help="Bounded worker-process count for independent scenario sweeps.",
    )
    parser.add_argument(
        "--keyframe-limit",
        type=int,
        help="Development-only limit applied independently to each scenario.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ[QUADRIC_PERFORMANCE_TRACE_ENV] = "1"
    output = args.output.resolve()
    if output == ROOT or output in ROOT.parents:
        raise ValueError("acceptance output cannot be the repository or its parent")
    output.mkdir(parents=True, exist_ok=True)
    baseline = _read_json(BASELINE_PATH)
    profile = baseline["profile"]
    pixel_policy = baseline["pixel_policy"]
    budgets = baseline["performance_budgets"]
    selected = set(args.scenario or acceptance_scenario_ids())
    scenarios = tuple(
        item for item in baseline["scenarios"] if item["id"] in selected
    )
    started = perf_counter()
    status = {
        "schema": "manim-quadric-acceptance-run-status/v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    _atomic_json(output / "run-status.json", status)
    try:
        keyframes = []
        supplemental_evidence = []
        counts = []
        contact_sheets = []
        keyframe_started = perf_counter()
        for scenario in scenarios:
            scenario_images = []
            scenario_labels = []
            progresses = tuple(float(value) for value in scenario["keyframes"])
            labels = tuple(str(value) for value in scenario["keyframe_labels"])
            if len(labels) != len(progresses):
                raise ValueError(
                    f"{scenario['id']} keyframe labels do not match keyframes"
                )
            if args.keyframe_limit is not None:
                if args.keyframe_limit <= 0:
                    raise ValueError("--keyframe-limit must be positive")
                progresses = progresses[: args.keyframe_limit]
                labels = labels[: args.keyframe_limit]
            for progress, keyframe_label in zip(progresses, labels):
                captured = _capture_keyframe(
                    str(scenario["id"]),
                    progress,
                    output=output,
                    profile=profile,
                    pixel_policy=pixel_policy,
                )
                if captured.evidence["elapsed_seconds"] > float(
                    budgets["single_keyframe_seconds"]
                ):
                    raise RuntimeError(
                        f"{scenario['id']} keyframe exceeded the performance budget"
                    )
                keyframes.append(captured.evidence)
                counts.extend(captured.rows)
                scenario_images.append(captured.image_path)
                scenario_labels.append(keyframe_label)
            sheet_path = output / "contact-sheets" / str(scenario["contact_sheet"])
            _contact_sheet(scenario_images, scenario_labels, sheet_path)
            contact_sheets.append(
                {
                    "scenario_id": scenario["id"],
                    "path": str(sheet_path.relative_to(output)),
                    "sha256": _sha256(sheet_path),
                }
            )
        if args.scenario is None:
            supplemental_specification = baseline["supplemental_evidence"][
                "open_double_shared_apex"
            ]
            supplemental = _capture_open_double_shared_apex(
                output=output,
                profile=profile,
                pixel_policy=pixel_policy,
                specification=supplemental_specification,
            )
            if supplemental.evidence["elapsed_seconds"] > float(
                budgets["single_keyframe_seconds"]
            ):
                raise RuntimeError(
                    "open-double shared-apex keyframe exceeded the performance budget"
                )
            supplemental_evidence.append(supplemental.evidence)
            counts.extend(supplemental.rows)
            contact_sheets.append(
                {
                    "scenario_id": "open_double_shared_apex",
                    "path": str(supplemental_specification["zoom"]),
                    "sha256": _sha256(
                        output / str(supplemental_specification["zoom"])
                    ),
                }
            )
        keyframe_elapsed = perf_counter() - keyframe_started
        if keyframe_elapsed > float(budgets["all_keyframes_seconds"]):
            raise RuntimeError("keyframe capture exceeded the performance budget")

        sweeps = []
        sweep_started = perf_counter()
        if not args.skip_motion_sweeps:
            sweep_results = _run_motion_sweeps(
                scenarios,
                output=output,
                workers=args.motion_sweep_workers,
            )
            for sweep, rows in sweep_results:
                sweeps.append(sweep)
                counts.extend(rows)
        sweep_elapsed = perf_counter() - sweep_started
        if sweep_elapsed > float(budgets["all_motion_sweeps_seconds"]):
            raise RuntimeError("motion sweeps exceeded the performance budget")

        videos = []
        video_started = perf_counter()
        if args.render_videos:
            for scenario in scenarios:
                videos.append(
                    _render_video(
                        scenario,
                        output=output,
                        profile=profile,
                        budgets=budgets,
                    )
                )
        video_elapsed = perf_counter() - video_started
        if video_elapsed > float(budgets["all_videos_seconds"]):
            raise RuntimeError("video rendering exceeded the performance budget")

        counts_path = output / "evidence" / "fragment-ray-counts.csv"
        _write_counts_csv(counts_path, counts)
        evidence = {
            "schema": "manim-quadric-extended-acceptance/v1",
            "performance_trace_schema": QUADRIC_PERFORMANCE_TRACE_SCHEMA,
            "contract_id": baseline["contract_id"],
            "git_commit": _git_commit(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "manim": manim.__version__,
                "numpy": np.__version__,
                "renderer": baseline["renderer"],
            },
            "profile": profile,
            "pixel_policy": pixel_policy,
            "performance": {
                "budgets": budgets,
                "keyframes_seconds": round(keyframe_elapsed, 6),
                "motion_sweeps_seconds": round(sweep_elapsed, 6),
                "motion_sweep_workers": (
                    0
                    if args.skip_motion_sweeps
                    else min(args.motion_sweep_workers, len(scenarios))
                ),
                "videos_seconds": round(video_elapsed, 6),
                "total_seconds": round(perf_counter() - started, 6),
            },
            "keyframes": keyframes,
            "supplemental_evidence": supplemental_evidence,
            "motion_sweeps": sweeps,
            "contact_sheets": contact_sheets,
            "videos": videos,
            "counts_csv": str(counts_path.relative_to(output)),
        }
        _atomic_json(output / "evidence" / "acceptance.json", evidence)
        _atomic_json(output / "artifact-manifest.json", _artifact_manifest(output))
        status.update(
            {
                "status": "passed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(perf_counter() - started, 6),
            }
        )
        _atomic_json(output / "run-status.json", status)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "output": str(output),
                    "keyframes": len(keyframes) + len(supplemental_evidence),
                    "motion_sweeps": len(sweeps),
                    "videos": len(videos),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(perf_counter() - started, 6),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(output / "run-status.json", status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
