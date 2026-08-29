from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SUBSET_PATH = Path(__file__).with_name("subset_v0_1.json")
LEVEL_ORDER = {"A": 0, "B": 1, "C": 2}

OBJECT_FEATURES = {
    "line": "object.line",
    "arrow": "object.arrow_stealth",
    "polygon": "object.polygon_fill",
    "ellipse": "object.ellipse",
    "circle": "object.circle",
    "dot": "object.dot",
    "label": "node.label",
    "path_label": "node.path_label",
    "angle": "marker.angle",
    "angle_label": "marker.angle",
    "right_angle": "marker.right_angle",
    "planar_circle_3d": "object.planar_circle_3d",
    "planar_ellipse_3d": "object.planar_ellipse_3d",
}

DEPENDENCY_FEATURES = {
    "interpolation": "relation.interpolation",
    "translation": "relation.translation",
    "projection": "relation.projection",
    "intersection": None,
}

UNSUPPORTED_PATTERNS = (
    (re.compile(r"\\clip\b"), "layout.clip"),
    (re.compile(r"\bcontrols\b"), "path.bezier"),
    (re.compile(r"\barc\b"), "path.arc_general"),
    (re.compile(r"\b(?:plot|smooth)\b"), "path.plot_smooth"),
    (re.compile(r"\bdecorat"), "style.decoration"),
    (re.compile(r"\b(?:pattern|shade|gradient)\b"), "style.pattern_shade_gradient"),
    (re.compile(r"\\matrix\b|\bmatrix\s*="), "node.matrix"),
    (re.compile(r"\btext width\s*="), "node.complex"),
)


def load_subset_spec(path: Path = DEFAULT_SUBSET_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_ids = [feature["id"] for feature in payload["features"]]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("duplicate feature id in TikZ-native subset specification")
    invalid_levels = {
        feature["level"]
        for feature in payload["features"]
        if feature["level"] not in LEVEL_ORDER
    }
    if invalid_levels:
        raise ValueError(f"invalid support levels: {sorted(invalid_levels)}")
    return payload


def feature_registry(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {feature["id"]: feature for feature in spec["features"]}


def _add(counter: Counter[str], feature_id: str, amount: int = 1) -> None:
    if amount > 0:
        counter[feature_id] += amount


def _style_features(item: Any, counter: Counter[str]) -> None:
    raw_options = [str(option) for option in item.style.raw_options]
    raw = " ".join(raw_options)
    if "line width" in raw:
        _add(counter, "style.line_width_pt")
    if item.style.dash_pattern_pt is not None:
        if "dash pattern" in raw:
            _add(counter, "style.dash_pattern_explicit")
        else:
            _add(counter, "style.dash_keyword")
    if "!" in raw:
        _add(counter, "style.xcolor_mix")
    if "opacity" in raw:
        _add(counter, "style.opacity")
    if item.style.font_command or (
        item.placement is not None and item.placement.font_command
    ):
        _add(counter, "style.font_command")


def _unsupported_feature(raw: str) -> str:
    for pattern, feature_id in UNSUPPORTED_PATTERNS:
        if pattern.search(raw):
            return feature_id
    if "scope" in raw and any(
        token in raw for token in ("rotate", "shift", "scale", "xscale", "yscale")
    ):
        return "scope.transform_nested"
    return "syntax.unsupported"


def _warning_feature(warning: str) -> str:
    if "baseline=" in warning:
        return "layout.baseline"
    if "trim right=" in warning:
        return "layout.trim_right"
    if "scope option" in warning and "draw=none" in warning:
        return "scope.redundant_draw_none"
    return "syntax.unsupported"


def audit_picture_compatibility(
    picture: Any,
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    findings: list[dict[str, Any]] = []

    _add(counts, "coordinate.named", len(picture.coordinates))
    if picture.dimension == 3:
        _add(counts, "coordinate.xyz", len(picture.coordinates))
        if picture.projection_3d is not None:
            _add(
                counts,
                (
                    "projection.3d_view"
                    if picture.projection_3d.source == "3d view"
                    else "projection.basis_3d"
                ),
            )
    _add(counts, "macro.numeric", len(picture.symbols))
    if "line width" in picture.raw_options:
        _add(counts, "style.line_width_pt")

    for dependency in picture.coordinate_dependencies.values():
        feature_id = DEPENDENCY_FEATURES.get(dependency.get("operation"))
        if feature_id:
            _add(counts, feature_id)

    _add(counts, "relation.named_path", len(picture.named_paths))
    _add(counts, "relation.hinge_3d", len(picture.hinge_relations))
    _add(
        counts,
        "geometry.planar_frame_3d",
        len(getattr(picture, "planar_frames_3d", {})),
    )
    for relation in picture.intersections:
        kinds = {
            picture.named_paths[relation.path_a].kind,
            picture.named_paths[relation.path_b].kind,
        }
        if kinds == {"line"}:
            feature_id = "relation.intersection.line_line"
        elif "line" in kinds and kinds & {"ellipse", "circle"}:
            feature_id = "relation.intersection.line_ellipse"
        else:
            feature_id = "relation.intersection.complex"
        _add(counts, feature_id)

    for item in picture.objects:
        feature_id = OBJECT_FEATURES.get(item.kind)
        if feature_id:
            _add(counts, feature_id)
        if item.kind == "path_label" and item.placement and item.placement.sloped:
            _add(counts, "node.path_label.sloped")
        if picture.dimension == 3 and item.kind in {"label", "path_label"}:
            _add(counts, "node.label_billboard_3d")
        _style_features(item, counts)

    for warning in picture.warnings:
        feature_id = _warning_feature(warning)
        _add(counts, feature_id)
        findings.append(
            {
                "level": registry[feature_id]["level"],
                "feature": feature_id,
                "message": warning,
            }
        )

    for unsupported in picture.unsupported:
        feature_id = _unsupported_feature(unsupported)
        _add(counts, feature_id)
        findings.append(
            {
                "level": "C",
                "feature": feature_id,
                "message": unsupported,
            }
        )

    missing_features = sorted(set(counts) - set(registry))
    if missing_features:
        raise KeyError(f"features missing from registry: {missing_features}")

    levels = [registry[feature_id]["level"] for feature_id in counts]
    overall_level = max(levels, key=LEVEL_ORDER.get) if levels else "A"
    return {
        "picture": picture.index,
        "overall_level": overall_level,
        "feature_counts": dict(sorted(counts.items())),
        "findings": findings,
    }


def audit_document_compatibility(
    document: Any,
    subset_path: Path = DEFAULT_SUBSET_PATH,
) -> dict[str, Any]:
    subset = load_subset_spec(subset_path)
    registry = feature_registry(subset)
    pictures = [
        audit_picture_compatibility(picture, registry)
        for picture in document.pictures
    ]
    totals: Counter[str] = Counter()
    for picture in pictures:
        totals.update(picture["feature_counts"])

    encountered = []
    level_feature_counts = Counter()
    for feature_id in sorted(totals):
        feature = registry[feature_id]
        level_feature_counts[feature["level"]] += 1
        encountered.append(
            {
                "id": feature_id,
                "level": feature["level"],
                "count": totals[feature_id],
                "description": feature["description"],
                "requirement": feature["requirement"],
            }
        )

    c_findings = [
        finding
        for picture in pictures
        for finding in picture["findings"]
        if finding["level"] == "C"
    ]
    has_intersections = any(
        feature_id.startswith("relation.intersection.")
        for feature_id in totals
    )
    dynamic_requirements = [
        "显式选择主动对象、运动参数和有效区间",
        "使用稳定对象 ID 编排教学时间线",
    ]
    if has_intersections:
        dynamic_requirements.append(
            "求交运动必须保留有向路径，并为相切或交点消失规定策略"
        )
    if any(picture.dimension == 3 for picture in document.pictures):
        dynamic_requirements.extend(
            [
                "相机运动时逐帧重算标签的屏幕 anchor，不能固定世界偏移",
                "遮挡关系变化时必须选择自动深度判定或显式虚实线策略",
            ]
        )

    return {
        "schema_version": 1,
        "subset_version": subset["subset_version"],
        "source": document.source_path,
        "source_sha256": document.source_sha256,
        "picture_count": len(document.pictures),
        "static_status": "pass" if not c_findings else "blocked",
        "dynamic_status": (
            "native-relations-ready-explicit-driver-required"
            if not c_findings
            else "blocked"
        ),
        "encountered_feature_counts_by_level": {
            level: level_feature_counts[level] for level in ("A", "B", "C")
        },
        "encountered_features": encountered,
        "dynamic_requirements": dynamic_requirements,
        "pictures": pictures,
        "c_findings": c_findings,
        "policy_note": (
            "A 表示对象或关系具备原生动态接口，不表示静态 TikZ 已经包含唯一的"
            "主动对象和教学时间线。B 项必须在报告中保留，但不伪装成 Manim 几何。"
        ),
    }
