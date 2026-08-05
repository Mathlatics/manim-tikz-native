#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tikz_native import compile_document  # noqa: E402
from tikz_native.animation import (  # noqa: E402
    SEMANTIC_LAYER_ORDER,
    semantic_animation_layers,
)


DEFAULT_SOURCE = ROOT / "tests" / "fixtures" / "national_2026_18_tikz.tex"
DEFAULT_OUTPUT = ROOT / "reports" / "tikz_native" / "2026_national_1_18"


def audit_native_source(paths: list[Path]) -> list[str]:
    """Reject explicit generic/path-image fallbacks in generated native code."""

    forbidden_calls = {"VMobject", "SVGMobject", "ImageMobject"}
    issues: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name: str | None = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in forbidden_calls:
                    issues.append(f"{path}:{node.lineno}: forbidden call {name}(...)")
                if name == "set_points_as_corners":
                    issues.append(
                        f"{path}:{node.lineno}: forbidden generic path construction "
                        "set_points_as_corners(...)"
                    )
    return issues


def build_animation_plan(document) -> dict:
    layer_counts = Counter()
    pictures = []
    intersection_relations = []
    for picture in document.pictures:
        layers = semantic_animation_layers(picture, include_empty=True)
        layer_counts.update(
            {layer.name: len(layer.object_ids) for layer in layers}
        )
        pictures.append(
            {
                "picture": picture.index,
                "layers": {
                    layer.name: list(layer.object_ids) for layer in layers
                },
            }
        )
        intersection_relations.extend(
            {
                "picture": picture.index,
                "paths": [relation.path_a, relation.path_b],
                "sort_by": relation.sort_by,
                "coordinates": list(relation.coordinate_names),
                "sort_parameters": list(relation.sort_parameters),
            }
            for relation in picture.intersections
        )
    return {
        "mode": "semantic_reveal",
        "layer_order": list(SEMANTIC_LAYER_ORDER),
        "layer_counts": {
            name: layer_counts[name] for name in SEMANTIC_LAYER_ORDER
        },
        "pictures": pictures,
        "note": (
            "This is a deterministic baseline reveal, not an inferred teaching "
            "narrative. Use stable object IDs for pedagogical ordering."
        ),
        "motion_dependencies": {
            "named_path_intersections": intersection_relations,
            "note": (
                "Native TikZ named paths, oriented intersection sorting, and "
                "coordinate construction dependencies are retained. Choosing the "
                "animation driver remains an explicit scene decision."
            ),
        },
    }


def build_report(document, native_audit: list[str], animation_plan: dict) -> dict:
    kind_counts = Counter(
        item.kind for picture in document.pictures for item in picture.objects
    )
    unsupported = [
        {"picture": picture.index, "items": picture.unsupported}
        for picture in document.pictures
        if picture.unsupported
    ]
    warnings = [
        {"picture": picture.index, "items": picture.warnings}
        for picture in document.pictures
        if picture.warnings
    ]
    return {
        "source": document.source_path,
        "source_sha256": document.source_sha256,
        "entry_macro": document.entry_macro,
        "pictures": len(document.pictures),
        "picture_dimensions": {
            "2d": sum(picture.dimension == 2 for picture in document.pictures),
            "3d": sum(picture.dimension == 3 for picture in document.pictures),
        },
        "objects": sum(len(picture.objects) for picture in document.pictures),
        "object_kinds": dict(sorted(kind_counts.items())),
        "animation_layer_counts": animation_plan["layer_counts"],
        "animation_plan": "animation_plan.json",
        "unsupported": unsupported,
        "warnings": warnings,
        "native_audit": native_audit,
        "strict_passed": not unsupported and not native_audit,
        "policy": {
            "generic_vmobject_fallback": False,
            "svg_fallback": False,
            "bitmap_fallback": False,
            "unsupported_syntax": "report-and-stop",
        },
    }


def report_markdown(report: dict) -> str:
    lines = [
        "# TikZ → 原生 Manim 转换报告",
        "",
        f"- 源文件：`{report['source']}`",
        f"- SHA-256：`{report['source_sha256']}`",
        f"- TikZ 图数：{report['pictures']}",
        f"- 独立语义对象：{report['objects']}",
        f"- 严格门禁：{'通过' if report['strict_passed'] else '未通过'}",
        "",
        "## 对象统计",
        "",
        "| 类型 | 数量 |",
        "| --- | ---: |",
    ]
    if report.get("entry_macro"):
        lines.insert(4, f"- 图形宏入口：`\\{report['entry_macro']}`")
    lines.extend(
        f"| `{kind}` | {count} |"
        for kind, count in report["object_kinds"].items()
    )
    if report.get("instantiated_classes"):
        lines.extend(
            [
                "",
                "## 实际创建的 Manim 类",
                "",
                "| 类 | 顶层对象数 |",
                "| --- | ---: |",
            ]
        )
        lines.extend(
            f"| `{class_name}` | {count} |"
            for class_name, count in report["instantiated_classes"].items()
        )
    lines.extend(
        [
            "",
            "## 自动动画分层",
            "",
            "| 播放层 | 对象数 |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{name}` | {count} |"
        for name, count in report["animation_layer_counts"].items()
    )
    lines.extend(
        [
            "",
            "完整稳定对象 ID 顺序见 `animation_plan.json`。该文件给出可直接播放的"
            "语义基线顺序；教学叙事仍应按对象 ID 显式编排。原生命名路径、按有向"
            "直线排序的交点关系与坐标构造依赖会写入清单；主动对象仍由动画场景显式选择。",
        ]
    )
    lines.extend(["", "## 未支持语法", ""])
    if report["unsupported"]:
        for entry in report["unsupported"]:
            lines.append(f"### 图 {entry['picture']}")
            lines.extend(f"- {item}" for item in entry["items"])
    else:
        lines.append("当前输入中的 TikZ 图没有未支持语句。")
    lines.extend(["", "## 非致命警告", ""])
    if report["warnings"]:
        for entry in report["warnings"]:
            lines.append(
                f"- 图 {entry['picture']}：" + "；".join(entry["items"])
            )
    else:
        lines.append("无。")
    lines.extend(["", "## 原生对象门禁", ""])
    if report["native_audit"]:
        lines.extend(f"- {item}" for item in report["native_audit"])
    else:
        lines.append(
            "通过：转换器和目标场景未直接调用 `VMobject(...)`、"
            "`SVGMobject(...)`、`ImageMobject(...)` 或 "
            "`set_points_as_corners(...)`。"
        )
    lines.extend(
        [
            "",
            "## 当前边界",
            "",
            "- `baseline`、`trim right` 等 TeX 排版选项不属于图形对象，"
            "已记录但不进入 Manim 几何层。",
            "- 普通 `dashed`、`densely dashed` 使用固定基准节距；明确写出的 "
            "`dash pattern=on ... off ...` 可以精确保留。",
            "- 当前只把 `scope[draw=none]` 识别为对 `\\fill` 的安全冗余；"
            "其他 scope 样式或变换会使严格门禁失败。",
            "- 程序可以自动生成填充、框架、实线、辅助线、标记、点、标签的"
            "基线播放顺序；更具体的教学叙事不从静态 TikZ 猜测，需要使用稳定对象 "
            "ID 编排。",
            "- `name path` 与 `name intersections` 当前支持直线—椭圆及直线—直线"
            "求交；动态身份按 `sort by` 指定的有向直线路径保存。复杂曲线多交点和"
            "相切事件仍需继续扩展。",
            "- 结构不同的公式标签不应直接做点级 `Transform`；应淡出/淡入，"
            "或在结构适合时使用 `TransformMatchingTex`。",
            "- 转换程序遇到新语法时会写入 `unsupported`，不会静默改用 SVG 或位图。",
            "- 三维图支持直接三维坐标，并读取 `3d view={方位角}{仰角}` 或 "
            "`x/y/z={(u,v)}` 基向量；点、线、面保留世界坐标，由原生 Manim "
            "三维相机投影。已经语义化的有限凸面线—面遮挡可随平行投影相机逐帧"
            "更新；空间曲线、任意平面圆弧、非凸/曲面和透视遮挡仍需逐项扩展。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a restricted TikZ subset into native-Manim specs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--entry-macro",
        help=(
            "Materialize one zero-argument figure macro before parsing, for "
            "macro-library TeX files. The leading backslash is optional."
        ),
    )
    parser.add_argument(
        "--instantiate",
        action="store_true",
        help="Instantiate every object with Manim after parsing.",
    )
    args = parser.parse_args()

    document = compile_document(args.input, entry_macro=args.entry_macro)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    document.write_json(args.output_dir / "manifest.json")
    animation_plan = build_animation_plan(document)
    (args.output_dir / "animation_plan.json").write_text(
        json.dumps(animation_plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    native_paths = [
        ROOT / "tikz_native" / "manim_renderer.py",
        ROOT / "tikz_native" / "manim_renderer_3d.py",
        ROOT / "tikz_native" / "projection_3d.py",
        ROOT / "tikz_native" / "animation.py",
        ROOT / "tikz_native" / "dynamic_geometry.py",
        ROOT / "scenes" / "national_2026_18_native.py",
        ROOT / "scenes" / "tikz_native_3d_demo.py",
        ROOT / "scenes" / "space_angle_part1_native.py",
    ]
    native_audit = audit_native_source([path for path in native_paths if path.exists()])
    report = build_report(document, native_audit, animation_plan)

    if args.instantiate:
        from tikz_native.manim_renderer import NativeManimRenderer
        from tikz_native.manim_renderer_3d import NativeManim3DRenderer

        # Preserve the physical TikZ coordinate scale during the instantiation
        # audit.  Any later framing belongs to the consuming scene/camera, not
        # to conversion itself.
        renderer_2d = NativeManimRenderer(scene_unit_per_cm=1.0)
        renderer_3d = NativeManim3DRenderer(scene_unit_per_cm=1.0)
        classes = Counter()
        bboxes: list[dict[str, float | int]] = []
        for picture in document.pictures:
            if picture.dimension == 3:
                figure_3d = renderer_3d.render(picture)
                classes.update(
                    type(item).__name__ for item in figure_3d.objects.values()
                )
                bboxes.append(
                    {
                        "picture": picture.index,
                        "dimension": 3,
                        "width": float(figure_3d.world_group.width),
                        "height": float(figure_3d.world_group.height),
                        "depth": float(figure_3d.world_group.depth),
                    }
                )
            else:
                figure_2d = renderer_2d.render(picture)
                classes.update(
                    type(item).__name__ for item in figure_2d.objects.values()
                )
                bboxes.append(
                    {
                        "picture": picture.index,
                        "dimension": 2,
                        "width": float(figure_2d.group.width),
                        "height": float(figure_2d.group.height),
                    }
                )
        report["instantiated_classes"] = dict(sorted(classes.items()))
        report["instantiated_bboxes"] = bboxes

    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        report_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["strict_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
