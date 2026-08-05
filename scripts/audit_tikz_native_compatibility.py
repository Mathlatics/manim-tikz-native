#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tikz_native import compile_document  # noqa: E402
from tikz_native.compatibility import (  # noqa: E402
    DEFAULT_SUBSET_PATH,
    audit_document_compatibility,
)


DEFAULT_SOURCE = ROOT / "tests" / "fixtures" / "national_2026_18_tikz.tex"
DEFAULT_OUTPUT_DIR = (
    ROOT / "reports" / "tikz_native" / "2026_national_1_18"
)


def compatibility_markdown(report: dict) -> str:
    lines = [
        "# TikZ → 原生 Manim 标准化子集兼容性报告",
        "",
        f"- 子集版本：`{report['subset_version']}`",
        f"- 源文件：`{report['source']}`",
        f"- SHA-256：`{report['source_sha256']}`",
        f"- TikZ 图数：{report['picture_count']}",
        f"- 静态状态：`{report['static_status']}`",
        f"- 动态状态：`{report['dynamic_status']}`",
        "",
        "A 表示原生对象或关系已经具备动态接口，不代表转换器可以从静态 TikZ"
        "唯一推断主动对象、运动区间和教学时间线。B 表示静态安全或版面层近似；"
        "C 表示严格模式必须停止。",
        "",
        "## 本文档实际使用的能力",
        "",
        "| 等级 | Feature | 出现次数 | 说明 | 要求 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    lines.extend(
        "| `{level}` | `{id}` | {count} | {description} | {requirement} |".format(
            **feature
        )
        for feature in report["encountered_features"]
    )

    lines.extend(["", "## B 级与 C 级逐图发现", ""])
    any_findings = False
    for picture in report["pictures"]:
        if not picture["findings"]:
            continue
        any_findings = True
        lines.append(f"### 图 {picture['picture']}（总体 `{picture['overall_level']}`）")
        lines.append("")
        for finding in picture["findings"]:
            lines.append(
                f"- `{finding['level']}` / `{finding['feature']}`："
                f"{finding['message']}"
            )
        lines.append("")
    if not any_findings:
        lines.append("无。")

    lines.extend(["", "## 动态使用前仍需明确", ""])
    lines.extend(
        f"- {requirement}" for requirement in report["dynamic_requirements"]
    )

    lines.extend(["", "## C 级门禁", ""])
    if report["c_findings"]:
        lines.extend(
            f"- `{finding['feature']}`：{finding['message']}"
            for finding in report["c_findings"]
        )
    else:
        lines.append("本次输入没有遇到 C 级语法。")

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "当前文档在标准化子集 `v0.1` 下可以继续进行原生 Manim 静态复刻；"
            "已保留的坐标关系可用于动态绑定。B 级项目必须继续留在报告中，"
            "主动对象、运动区间和教学时间线由动画计划显式给出。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a TeX document against the TikZ-native A/B/C subset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--basename",
        default="compatibility-v0.1",
        help="Output basename without .json/.md.",
    )
    args = parser.parse_args()

    document = compile_document(args.input)
    report = audit_document_compatibility(document, args.subset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.basename}.json"
    markdown_path = args.output_dir / f"{args.basename}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(compatibility_markdown(report), encoding="utf-8")

    print(json.dumps({
        "subset_version": report["subset_version"],
        "static_status": report["static_status"],
        "dynamic_status": report["dynamic_status"],
        "features_by_level": report["encountered_feature_counts_by_level"],
        "json": str(json_path),
        "markdown": str(markdown_path),
    }, ensure_ascii=False, indent=2))
    return 0 if report["static_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
