"""Capture the pre-fix five-state quadric section-boundary baseline.

The structural half records renderer-neutral frame facts and exact canonical
JSON.  The Cairo half renders fill-only frames, erodes legitimate role
boundaries by three pixels, and counts remaining interior pixels whose RGB is
not the mathematically expected flat composite.  Those pixels are the stable
seam/deviation signal used by the follow-up repair batches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from math import sqrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from manim import Scene, config, tempconfig
from scipy.ndimage import binary_erosion


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import scene as diagnostic  # noqa: E402

from polyhedron_visibility.quadrics.section_compositing import (  # noqa: E402
    PlaneDepthRole,
    QuadricSectionCompositingFrame,
    canonical_quadric_section_compositing_json,
)


SCHEMA = "manim-quadric-section-boundary-baseline/v1"
PR12_MERGE_COMMIT = "a7811bf4f7d4adbf2e4078e30d647b40bea6693a"
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
MODES = ("opaque_fill", "translucent_fill")
RGB_ERROR_THRESHOLD = 8.0
BOUNDARY_EROSION_PIXELS = 3


def _run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _file_evidence(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "name": path.name,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _triangle_area(points: Sequence[Sequence[float]]) -> float:
    values = np.asarray(points, dtype=float)
    return 0.5 * abs(
        sum(
            values[index, 0] * values[(index + 1) % len(values), 1]
            - values[index, 1] * values[(index + 1) % len(values), 0]
            for index in range(len(values))
        )
    )


def _hex_rgb(value: str) -> np.ndarray:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected six-digit RGB color, received {value!r}")
    return np.asarray(
        tuple(int(text[index : index + 2], 16) for index in (0, 2, 4)),
        dtype=float,
    )


def _source_over(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return foreground * alpha + background * (1.0 - alpha)


def _expected_role_rgb(role: PlaneDepthRole, mode: str) -> np.ndarray:
    style = diagnostic.style_for_mode(mode)
    background = _hex_rgb(diagnostic.BACKGROUND_COLOR)
    surface = _hex_rgb(diagnostic.SURFACE_COLOR)
    plane = _hex_rgb(diagnostic.PLANE_COLOR)
    sheet_alpha = 1.0 - sqrt(1.0 - style.surface_fill_opacity)
    plane_alpha = style.section_plane_fill_opacity
    if role is PlaneDepthRole.OUTSIDE_PROJECTION:
        return _source_over(background, plane, plane_alpha)
    if role is PlaneDepthRole.BEHIND_SURFACE:
        result = _source_over(background, plane, plane_alpha)
        result = _source_over(result, surface, sheet_alpha)
        return _source_over(result, surface, sheet_alpha)
    if role is PlaneDepthRole.BETWEEN_SURFACE_SHEETS:
        result = _source_over(background, surface, sheet_alpha)
        result = _source_over(result, plane, plane_alpha)
        return _source_over(result, surface, sheet_alpha)
    if role is PlaneDepthRole.IN_FRONT_OF_SURFACE:
        result = _source_over(background, surface, sheet_alpha)
        result = _source_over(result, surface, sheet_alpha)
        return _source_over(result, plane, plane_alpha)
    raise AssertionError(role)


def _accepted_flat_fill_palette(mode: str) -> tuple[np.ndarray, ...]:
    """Return every legitimate flat composite used by the fill-only scene.

    A pre-fix role can occupy the wrong solid layer over a large area.  That is
    a painter-role error, not a Cairo triangulation seam.  The seam baseline
    therefore accepts every mathematically valid flat stack and flags only
    pixels that match none of them.
    """

    style = diagnostic.style_for_mode(mode)
    background = _hex_rgb(diagnostic.BACKGROUND_COLOR)
    surface = _hex_rgb(diagnostic.SURFACE_COLOR)
    plane = _hex_rgb(diagnostic.PLANE_COLOR)
    sheet_alpha = 1.0 - sqrt(1.0 - style.surface_fill_opacity)
    plane_alpha = style.section_plane_fill_opacity
    palette: list[np.ndarray] = []
    for surface_count in range(3):
        without_plane = background.copy()
        for _ in range(surface_count):
            without_plane = _source_over(
                without_plane,
                surface,
                sheet_alpha,
            )
        palette.append(without_plane)
        for plane_position in range(surface_count + 1):
            composite = background.copy()
            for layer_index in range(surface_count + 1):
                if layer_index == plane_position:
                    composite = _source_over(
                        composite,
                        plane,
                        plane_alpha,
                    )
                if layer_index < surface_count:
                    composite = _source_over(
                        composite,
                        surface,
                        sheet_alpha,
                    )
            palette.append(composite)
    unique: dict[tuple[int, int, int], np.ndarray] = {}
    for color in palette:
        unique.setdefault(
            tuple(int(round(channel)) for channel in color),
            color,
        )
    return tuple(unique.values())


def _screen_to_pixel(
    point: Sequence[float],
    *,
    width: int,
    height: int,
    frame_width: float,
    frame_height: float,
) -> tuple[float, float]:
    x, y = (float(value) for value in point[:2])
    return (
        (x / frame_width + 0.5) * (width - 1),
        (0.5 - y / frame_height) * (height - 1),
    )


def _role_mask(
    frame: QuadricSectionCompositingFrame,
    role: PlaneDepthRole,
    *,
    width: int,
    height: int,
    frame_width: float,
    frame_height: float,
) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for fragment in frame.fragments_by_role[role]:
        draw.polygon(
            tuple(
                _screen_to_pixel(
                    point,
                    width=width,
                    height=height,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                for point in fragment.screen_vertices
            ),
            fill=255,
        )
    mask = np.asarray(image, dtype=np.uint8) > 0
    return binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=BOUNDARY_EROSION_PIXELS,
        border_value=0,
    )


def _analyze_cairo_frame(
    image_path: Path,
    frame: QuadricSectionCompositingFrame,
    mode: str,
    *,
    frame_width: float,
    frame_height: float,
) -> tuple[dict[str, object], Image.Image]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    deviation_union = np.zeros((height, width), dtype=bool)
    per_role: dict[str, object] = {}
    palette = _accepted_flat_fill_palette(mode)
    palette_errors = np.stack(
        tuple(
            np.linalg.norm(image.astype(float) - expected, axis=2)
            for expected in palette
        ),
        axis=0,
    )
    nearest_palette_error = np.min(palette_errors, axis=0)
    for role in ROLE_ORDER:
        mask = _role_mask(
            frame,
            role,
            width=width,
            height=height,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        deviations = mask & (nearest_palette_error > RGB_ERROR_THRESHOLD)
        deviation_union |= deviations
        per_role[ROLE_LABEL[role]] = {
            "safeInteriorPixelCount": int(np.count_nonzero(mask)),
            "seamPixelCount": int(np.count_nonzero(deviations)),
            "nominalRoleRgb": [
                int(round(value))
                for value in _expected_role_rgb(role, mode)
            ],
        }

    overlay = image.copy()
    overlay[deviation_union] = np.asarray((255, 0, 0), dtype=np.uint8)
    return (
        {
            "imageSize": [width, height],
            "rgbErrorThreshold": RGB_ERROR_THRESHOLD,
            "boundaryErosionPixels": BOUNDARY_EROSION_PIXELS,
            "acceptedRgbPalette": [
                [int(round(value)) for value in color]
                for color in palette
            ],
            "safeInteriorPixelCount": sum(
                int(record["safeInteriorPixelCount"])
                for record in per_role.values()
            ),
            "seamPixelCount": int(np.count_nonzero(deviation_union)),
            "perRole": per_role,
        },
        Image.fromarray(overlay, mode="RGB"),
    )


def _render_keyframe(
    state_name: str,
    mode: str,
    temporary_root: Path,
) -> Path:
    output_name = f"{mode}_{state_name}"
    media_dir = temporary_root / output_name
    environment = os.environ.copy()
    environment["QUADRIC_BASELINE_STATE"] = state_name
    environment["QUADRIC_BASELINE_MODE"] = mode
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(REPO_ROOT), environment.get("PYTHONPATH", "")),
        )
    )
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "manim",
            "--renderer",
            "cairo",
            "--disable_caching",
            "-r",
            f"{diagnostic.PIXEL_WIDTH},{diagnostic.PIXEL_HEIGHT}",
            "--fps",
            str(diagnostic.FPS),
            "-s",
            "--format",
            "png",
            "--media_dir",
            str(media_dir),
            str(HERE / "scene.py"),
            "BoundaryBaselineStill",
            "-o",
            output_name,
        ),
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Manim failed for {mode}/{state_name}:\n{completed.stdout}"
        )
    candidates = tuple(media_dir.rglob("*.png"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one PNG for {mode}/{state_name}, found {len(candidates)}"
        )
    return candidates[0]


def _capture_structure(
    output_dir: Path,
) -> tuple[
    list[dict[str, object]],
    dict[str, QuadricSectionCompositingFrame],
    float,
    float,
    dict[str, object],
]:
    state = {"name": diagnostic.STATES[0].name}
    records: list[dict[str, object]] = []
    frames: dict[str, QuadricSectionCompositingFrame] = {}
    canonical_dir = output_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
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
        controller = diagnostic.build_controller(
            scene,
            lambda: state["name"],
            "translucent_fill",
        ).attach()
        baseline_identities = controller.slot_identities()
        slot_topology = tuple(
            f"{type(item).__module__}.{type(item).__qualname__}"
            for item in controller.root.get_family()
        )
        topology_payload = json.dumps(
            slot_topology,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        for index, definition in enumerate(diagnostic.STATES):
            state["name"] = definition.name
            if index:
                controller.update()
            frame = controller.last_section_frame
            if frame is None:
                raise RuntimeError(
                    f"missing section frame for {definition.name}"
                )
            frames[definition.name] = frame
            canonical = canonical_quadric_section_compositing_json(frame).encode(
                "utf-8"
            )
            compressed = gzip.compress(canonical, compresslevel=9, mtime=0)
            canonical_path = canonical_dir / f"{definition.name}.json.gz"
            canonical_path.write_bytes(compressed)
            current_identities = controller.slot_identities()
            records.append(
                {
                    "state": definition.name,
                    "stateDefinition": asdict(definition),
                    "planeFragmentCount": len(frame.plane_fragments),
                    "planeOutlineFragmentCount": len(
                        frame.plane_outline_fragments
                    ),
                    "rayClassificationCount": frame.ray_classification_count,
                    "roleScreenAreas": {
                        ROLE_LABEL[role]: sum(
                            _triangle_area(fragment.screen_vertices)
                            for fragment in frame.fragments_by_role[role]
                        )
                        for role in ROLE_ORDER
                    },
                    "canonicalJson": {
                        "path": canonical_path.relative_to(output_dir).as_posix(),
                        "bytes": len(canonical),
                        "sha256": _sha256_bytes(canonical),
                        "gzipBytes": len(compressed),
                        "gzipSha256": _sha256_bytes(compressed),
                    },
                    "slotIdentityStable": (
                        current_identities == baseline_identities
                    ),
                    "changedSlotIdentityIndices": [
                        position
                        for position, (before, after) in enumerate(
                            zip(baseline_identities, current_identities)
                        )
                        if before != after
                    ],
                }
            )
        identity = {
            "semantics": (
                "Python id values are process-local; the baseline records "
                "family size/topology and proves that every allocated identity "
                "is unchanged across all five updates."
            ),
            "slotIdentityCount": len(baseline_identities),
            "slotTopologySha256": _sha256_bytes(topology_payload),
            "stableAcrossAllStates": all(
                bool(record["slotIdentityStable"]) for record in records
            ),
        }
        frame_width = float(config.frame_width)
        frame_height = float(config.frame_height)
        controller.restore()
    return records, frames, frame_width, frame_height, identity


def _write_contact_sheet(
    panels: Mapping[tuple[str, str], tuple[Image.Image, int]],
    destination: Path,
) -> None:
    panel_width = diagnostic.PIXEL_WIDTH
    panel_height = diagnostic.PIXEL_HEIGHT + 24
    sheet = Image.new(
        "RGB",
        (panel_width * len(diagnostic.STATES), panel_height * len(MODES)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, mode in enumerate(MODES):
        for column, state in enumerate(diagnostic.STATES):
            overlay, count = panels[(mode, state.name)]
            left = column * panel_width
            top = row * panel_height
            sheet.paste(overlay, (left, top + 24))
            draw.text(
                (left + 6, top + 6),
                f"{mode} / {state.name}: {count} seam pixels",
                fill="black",
            )
    sheet.save(destination)


def _report_markdown(payload: Mapping[str, object]) -> str:
    identity = payload["fixedMobjectIdentity"]
    lines = [
        "# Quadric section-boundary pre-fix baseline",
        "",
        f"- Source commit: `{payload['sourceCommit']}`",
        f"- Required PR #12 ancestor: `{payload['requiredAncestor']}`",
        f"- Production quadric diff from PR #12: `{payload['productionDiff']}`",
        f"- Fixed Mobject identities: `{identity['slotIdentityCount']}`",
        (
            "- Identity stable across five states: "
            f"`{identity['stableAcrossAllStates']}`"
        ),
        "",
        (
            "The Cairo seam count uses fill-only production rendering. "
            "Legitimate role boundaries are eroded by 3 pixels; a remaining "
            "interior pixel is counted when its RGB distance from every valid "
            "flat-fill composite exceeds 8.0. Solid painter-role errors are "
            "deliberately excluded."
        ),
        "",
        (
            "| State | Plane fragments | Ray classifications | Behind area | "
            "Outside area | Between area | Front area | Opaque seam px | "
            "Translucent seam px | Canonical SHA-256 | IDs stable |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in payload["records"]:
        areas = record["roleScreenAreas"]
        cairo = record["cairo"]
        lines.append(
            "| {state} | {fragments} | {rays} | {behind:.12f} | "
            "{outside:.12f} | {between:.12f} | {front:.12f} | "
            "{opaque} | {translucent} | `{canonical}` | {stable} |".format(
                state=record["state"],
                fragments=record["planeFragmentCount"],
                rays=record["rayClassificationCount"],
                behind=areas["behind"],
                outside=areas["outside"],
                between=areas["between"],
                front=areas["front"],
                opaque=cairo["opaque_fill"]["seamPixelCount"],
                translucent=cairo["translucent_fill"]["seamPixelCount"],
                canonical=record["canonicalJson"]["sha256"],
                stable=record["slotIdentityStable"],
            )
        )
    lines.extend(
        (
            "",
            "## Initial reference attachments",
            "",
        )
    )
    references = payload.get("initialReferences", ())
    if references:
        for reference in references:
            lines.append(
                f"- `{reference['name']}`: `{reference['sha256']}` "
                f"({reference['bytes']} bytes)"
            )
    else:
        lines.append("- None supplied.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        type=Path,
        help="Optional pre-existing evidence file; only its hash is recorded.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_commit = _run_git("rev-parse", "HEAD")
    subprocess.run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            PR12_MERGE_COMMIT,
            "HEAD",
        ),
        cwd=REPO_ROOT,
        check=True,
    )
    production_diff = _run_git(
        "diff",
        "--name-only",
        PR12_MERGE_COMMIT,
        "HEAD",
        "--",
        "polyhedron_visibility/quadrics",
    )
    if production_diff:
        raise RuntimeError(
            "batch 0 requires unchanged quadric production files; found:\n"
            + production_diff
        )
    working_tree_production_diff = _run_git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "polyhedron_visibility/quadrics",
    )
    if working_tree_production_diff:
        raise RuntimeError(
            "batch 0 requires a clean quadric production working tree; found:\n"
            + working_tree_production_diff
        )

    records, frames, frame_width, frame_height, identity = _capture_structure(
        output_dir
    )
    panels: dict[tuple[str, str], tuple[Image.Image, int]] = {}
    by_state = {record["state"]: record for record in records}
    with TemporaryDirectory(prefix="quadric-section-boundary-baseline-") as temp:
        temporary_root = Path(temp)
        for mode in MODES:
            for definition in diagnostic.STATES:
                print(f"rendering {mode}/{definition.name}", flush=True)
                image_path = _render_keyframe(
                    definition.name,
                    mode,
                    temporary_root,
                )
                evidence, overlay = _analyze_cairo_frame(
                    image_path,
                    frames[definition.name],
                    mode,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                by_state[definition.name].setdefault("cairo", {})[mode] = evidence
                panels[(mode, definition.name)] = (
                    overlay,
                    int(evidence["seamPixelCount"]),
                )

    references = [
        _file_evidence(path.resolve())
        for path in args.reference
        if path.is_file()
    ]
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "sourceCommit": source_commit,
        "requiredAncestor": PR12_MERGE_COMMIT,
        "productionDiff": "empty",
        "productionWorkingTree": "clean",
        "states": [state.name for state in diagnostic.STATES],
        "fixedMobjectIdentity": identity,
        "pixelMethod": {
            "renderer": "Manim Cairo",
            "imageSize": [diagnostic.PIXEL_WIDTH, diagnostic.PIXEL_HEIGHT],
            "frameSize": [frame_width, frame_height],
            "rgbErrorThreshold": RGB_ERROR_THRESHOLD,
            "boundaryErosionPixels": BOUNDARY_EROSION_PIXELS,
            "styles": list(MODES),
        },
        "initialReferences": references,
        "records": records,
    }
    (output_dir / "baseline.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _report_markdown(payload),
        encoding="utf-8",
    )
    _write_contact_sheet(panels, output_dir / "seam_evidence.png")
    print(output_dir)


if __name__ == "__main__":
    main()
