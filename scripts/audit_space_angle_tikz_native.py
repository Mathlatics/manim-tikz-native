#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tikz_native import compile_document  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "reports" / "space_angle_part1_native" / "batch_audit"
PREVIEW_COMMAND_RE = re.compile(r"\\previewFig(?![A-Za-z@])")
ENTRY_MACRO_RE = re.compile(r"\\([A-Za-z@]+)")


@dataclass(frozen=True)
class PreviewEntry:
    index: int
    label: str
    entry_macro: str
    source_line: int


def _blank_tex_comments(text: str) -> str:
    """Remove TeX comments while preserving every source offset and line."""

    output: list[str] = []
    for line in text.splitlines(keepends=True):
        comment_at: int | None = None
        for index, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                comment_at = index
                break
        if comment_at is None:
            output.append(line)
            continue

        newline = ""
        content_end = len(line)
        if line.endswith("\r\n"):
            newline = "\r\n"
            content_end -= 2
        elif line.endswith(("\n", "\r")):
            newline = line[-1]
            content_end -= 1
        output.append(
            line[:comment_at]
            + " " * max(0, content_end - comment_at)
            + newline
        )
    return "".join(output)


def _read_braced_group(text: str, offset: int) -> tuple[str, int] | None:
    while offset < len(text) and text[offset].isspace():
        offset += 1
    if offset >= len(text) or text[offset] != "{":
        return None

    depth = 0
    escaped = False
    for index in range(offset, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[offset + 1 : index], index + 1
    return None


def extract_preview_entries(source_text: str) -> list[PreviewEntry]:
    r"""Extract ordered ``\previewFig{label}{\Macro}`` audit entries."""

    clean_text = _blank_tex_comments(source_text)
    entries: list[PreviewEntry] = []
    for match in PREVIEW_COMMAND_RE.finditer(clean_text):
        label_group = _read_braced_group(clean_text, match.end())
        if label_group is None:
            # This also deliberately skips ``\newcommand{\previewFig}[2]``.
            continue
        macro_group = _read_braced_group(clean_text, label_group[1])
        if macro_group is None:
            continue

        label = label_group[0].strip()
        macro_source = macro_group[0].strip()
        macro_match = ENTRY_MACRO_RE.fullmatch(macro_source)
        source_line = clean_text.count("\n", 0, match.start()) + 1
        if macro_match is None:
            raise ValueError(
                f"Line {source_line}: preview body must be one zero-argument "
                f"control sequence, got {macro_source!r}"
            )
        entries.append(
            PreviewEntry(
                index=len(entries) + 1,
                label=label,
                entry_macro=macro_match.group(1),
                source_line=source_line,
            )
        )
    return entries


def _object_kind_counts(pictures: list[Any]) -> dict[str, int]:
    counts = Counter(
        item.kind for picture in pictures for item in picture.objects
    )
    return dict(sorted(counts.items()))


def _warning_records(document: Any) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = [
        {"scope": "document", "message": message}
        for message in document.warnings
    ]
    for picture in document.pictures:
        warnings.extend(
            {
                "scope": "picture",
                "picture": picture.index,
                "message": message,
            }
            for message in picture.warnings
        )
        for item in picture.objects:
            warnings.extend(
                {
                    "scope": "object",
                    "picture": picture.index,
                    "object_id": item.id,
                    "message": message,
                }
                for message in item.warnings
            )
    return warnings


def _unsupported_records(document: Any) -> list[dict[str, Any]]:
    return [
        {
            "picture": picture.index,
            "source_start_line": picture.start_line,
            "message": message,
        }
        for picture in document.pictures
        for message in picture.unsupported
    ]


def _picture_record(picture: Any) -> dict[str, Any]:
    object_warnings = [
        {"object_id": item.id, "message": message}
        for item in picture.objects
        for message in item.warnings
    ]
    return {
        "picture": picture.index,
        "source_start_line": picture.start_line,
        "source_end_line": picture.end_line,
        "dimension": picture.dimension,
        "projection_source": (
            picture.projection_3d.source if picture.projection_3d else None
        ),
        "coordinate_count": len(picture.coordinates),
        "object_count": len(picture.objects),
        "object_kind_counts": _object_kind_counts([picture]),
        "warning_count": len(picture.warnings) + len(object_warnings),
        "unsupported_count": len(picture.unsupported),
        "warnings": list(picture.warnings),
        "object_warnings": object_warnings,
        "unsupported": list(picture.unsupported),
    }


def _empty_instantiation(
    *,
    requested: bool,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "requested": requested,
        "status": status,
        "reason": reason,
        "eligible_picture_count": 0,
        "skipped_picture_count": 0,
        "instantiated_picture_count": 0,
        "object_count": 0,
        "class_counts": {},
        "warning_count": 0,
        "warnings": [],
        "pictures": [],
        "exception": None,
    }


def _instantiate_document(document: Any) -> dict[str, Any]:
    eligible = [picture for picture in document.pictures if picture.dimension == 3]
    if not eligible:
        return _empty_instantiation(
            requested=True,
            status="skipped",
            reason="no_3d_picture",
        )

    result = _empty_instantiation(requested=True, status="passed")
    result["eligible_picture_count"] = len(eligible)
    result["skipped_picture_count"] = len(document.pictures) - len(eligible)
    try:
        from tikz_native.manim_renderer_3d import NativeManim3DRenderer

        renderer = NativeManim3DRenderer(scene_unit_per_cm=1.0)
    except Exception as error:
        result["status"] = "failed"
        result["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        return result

    class_counts: Counter[str] = Counter()
    warnings: list[dict[str, Any]] = []
    first_exception: dict[str, str] | None = None
    for picture in eligible:
        try:
            figure = renderer.render(picture)
        except Exception as error:
            exception = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            if first_exception is None:
                first_exception = exception
            result["pictures"].append(
                {
                    "picture": picture.index,
                    "status": "failed",
                    "object_count": 0,
                    "class_counts": {},
                    "warning_count": 0,
                    "warnings": [],
                    "exception": exception,
                }
            )
            continue

        picture_classes = Counter(
            type(item).__name__ for item in figure.objects.values()
        )
        picture_warnings = [
            {"picture": picture.index, "message": message}
            for message in figure.warnings
        ]
        class_counts.update(picture_classes)
        warnings.extend(picture_warnings)
        result["instantiated_picture_count"] += 1
        result["object_count"] += len(figure.objects)
        result["pictures"].append(
            {
                "picture": picture.index,
                "status": "passed",
                "object_count": len(figure.objects),
                "class_counts": dict(sorted(picture_classes.items())),
                "warning_count": len(picture_warnings),
                "warnings": picture_warnings,
                "exception": None,
            }
        )

    result["class_counts"] = dict(sorted(class_counts.items()))
    result["warning_count"] = len(warnings)
    result["warnings"] = warnings
    result["exception"] = first_exception
    if first_exception is not None:
        result["status"] = "failed"
    return result


def audit_entry(
    source_path: Path,
    entry: PreviewEntry,
    *,
    instantiate: bool = False,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "index": entry.index,
        "label": entry.label,
        "entry_macro": entry.entry_macro,
        "preview_line": entry.source_line,
        "status": "exception",
        "compiled": False,
        "picture_count": 0,
        "dimensions": [],
        "coordinate_count": 0,
        "object_count": 0,
        "object_kind_counts": {},
        "warning_count": 0,
        "unsupported_count": 0,
        "warnings": [],
        "unsupported": [],
        "pictures": [],
        "exception": None,
        "instantiation": _empty_instantiation(
            requested=instantiate,
            status="not_requested" if not instantiate else "skipped",
            reason=None if not instantiate else "compile_exception",
        ),
    }
    try:
        document = compile_document(source_path, entry_macro=entry.entry_macro)
    except Exception as error:  # A single unsupported entry must not stop the batch.
        base["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        return base

    warnings = _warning_records(document)
    unsupported = _unsupported_records(document)
    pictures = [_picture_record(picture) for picture in document.pictures]
    if unsupported:
        status = "unsupported"
    elif warnings:
        status = "warnings"
    else:
        status = "success"

    base.update(
        {
            "status": status,
            "compiled": True,
            "picture_count": len(document.pictures),
            "dimensions": sorted(
                {picture.dimension for picture in document.pictures}
            ),
            "coordinate_count": sum(
                len(picture.coordinates) for picture in document.pictures
            ),
            "object_count": sum(
                len(picture.objects) for picture in document.pictures
            ),
            "object_kind_counts": _object_kind_counts(document.pictures),
            "warning_count": len(warnings),
            "unsupported_count": len(unsupported),
            "warnings": warnings,
            "unsupported": unsupported,
            "pictures": pictures,
        }
    )
    if instantiate:
        if unsupported:
            base["instantiation"] = _empty_instantiation(
                requested=True,
                status="skipped",
                reason="unsupported",
            )
        else:
            base["instantiation"] = _instantiate_document(document)
    return base


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ").strip()


def _short_issue(record: dict[str, Any], limit: int = 150) -> str:
    if record["exception"]:
        issue = (
            f"{record['exception']['type']}: "
            f"{record['exception']['message']}"
        )
    elif record["unsupported"]:
        issue = record["unsupported"][0]["message"]
    elif record["warnings"]:
        issue = record["warnings"][0]["message"]
    else:
        return "—"
    issue = re.sub(r"\s+", " ", issue).strip()
    return issue if len(issue) <= limit else issue[: limit - 1] + "…"


def _instantiation_detail(record: dict[str, Any], limit: int = 150) -> str:
    instantiation = record["instantiation"]
    status = instantiation["status"]
    if status == "passed":
        detail = ", ".join(
            f"{name}×{count}"
            for name, count in instantiation["class_counts"].items()
        )
        return detail or "0 objects"
    if status == "failed" and instantiation["exception"]:
        exception = instantiation["exception"]
        detail = f"{exception['type']}: {exception['message']}"
    elif status == "skipped":
        detail = f"skipped: {instantiation['reason']}"
    else:
        return "—"
    detail = re.sub(r"\s+", " ", detail).strip()
    return detail if len(detail) <= limit else detail[: limit - 1] + "…"


def support_matrix_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 空间角 TikZ 原生 Manim 批量支持矩阵",
        "",
        f"- 源文件：`{report['source']}`",
        f"- SHA-256：`{report['source_sha256']}`",
        f"- `previewFig` 入口：{report['entry_count']}",
        f"- 状态：success {summary['success']}，warnings {summary['warnings']}，"
        f"unsupported {summary['unsupported']}，exception {summary['exception']}",
        (
            f"- 原生实例化：passed {summary['instantiation_passed']}，"
            f"failed {summary['instantiation_failed']}，"
            f"skipped {summary['instantiation_skipped']}"
            if summary["instantiation_requested"]
            else "- 原生实例化：未请求"
        ),
        "",
        "`warnings` 表示已编译且没有 unsupported；U/W 分别是 unsupported/"
        "warning 数量，完整诊断保存在同目录 JSON。",
        "",
        "| # | Preview label | Entry macro | TeX 行 | 状态 | 图数 | 维度 | 对象 | U | W | 实例化 | 类计数/实例化异常 | 首项编译诊断 |",
        "| ---: | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for entry in report["entries"]:
        dimensions = (
            "/".join(f"{value}D" for value in entry["dimensions"]) or "—"
        )
        lines.append(
            "| {index} | `{label}` | `\\{macro}` | {line} | `{status}` | "
            "{pictures} | {dimensions} | {objects} | {unsupported} | {warnings} | "
            "`{instantiation}` | {instantiation_detail} | {issue} |".format(
                index=entry["index"],
                label=_markdown_cell(entry["label"]),
                macro=_markdown_cell(entry["entry_macro"]),
                line=entry["preview_line"],
                status=entry["status"],
                pictures=entry["picture_count"],
                dimensions=dimensions,
                objects=entry["object_count"],
                unsupported=entry["unsupported_count"],
                warnings=entry["warning_count"],
                instantiation=entry["instantiation"]["status"],
                instantiation_detail=_markdown_cell(
                    _instantiation_detail(entry)
                ),
                issue=_markdown_cell(_short_issue(entry)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_report(
    source_path: Path,
    entries: list[PreviewEntry],
    *,
    instantiate: bool = False,
) -> dict[str, Any]:
    audited = [
        audit_entry(source_path, entry, instantiate=instantiate)
        for entry in entries
    ]
    status_counts = Counter(entry["status"] for entry in audited)
    instantiation_counts = Counter(
        entry["instantiation"]["status"] for entry in audited
    )
    return {
        "schema_version": 2,
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "entry_count": len(entries),
        "summary": {
            "success": status_counts["success"],
            "warnings": status_counts["warnings"],
            "unsupported": status_counts["unsupported"],
            "exception": status_counts["exception"],
            "compiled": sum(entry["compiled"] for entry in audited),
            "native_supported": sum(
                entry["status"] in {"success", "warnings"}
                for entry in audited
            ),
            "picture_count": sum(entry["picture_count"] for entry in audited),
            "object_count": sum(entry["object_count"] for entry in audited),
            "warning_count": sum(entry["warning_count"] for entry in audited),
            "unsupported_count": sum(
                entry["unsupported_count"] for entry in audited
            ),
            "instantiation_requested": instantiate,
            "instantiation_passed": instantiation_counts["passed"],
            "instantiation_failed": instantiation_counts["failed"],
            "instantiation_skipped": instantiation_counts["skipped"],
            "instantiation_not_requested": instantiation_counts[
                "not_requested"
            ],
            "instantiated_picture_count": sum(
                entry["instantiation"]["instantiated_picture_count"]
                for entry in audited
            ),
            "instantiated_object_count": sum(
                entry["instantiation"]["object_count"] for entry in audited
            ),
        },
        "entries": audited,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every \\previewFig entry in a space-angle TikZ macro file "
            "with the native Manim compiler."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="需要审计的空间角 TikZ TeX 文件",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--instantiate",
        action="store_true",
        help=(
            "Instantiate supported 3D pictures with NativeManim3DRenderer "
            "and record native class counts or rendering exceptions."
        ),
    )
    args = parser.parse_args()

    source_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_path.is_file():
        parser.error(f"input is not a file: {source_path}")

    source_text = source_path.read_text(encoding="utf-8")
    try:
        entries = extract_preview_entries(source_text)
    except ValueError as error:
        parser.error(str(error))
    if not entries:
        parser.error(
            f"no \\previewFig{{label}}{{\\Macro}} entries found in {source_path}"
        )

    report = build_report(source_path, entries, instantiate=args.instantiate)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "support-matrix.md"
    report["outputs"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        support_matrix_markdown(report),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "source": report["source"],
                "entry_count": report["entry_count"],
                "summary": report["summary"],
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
